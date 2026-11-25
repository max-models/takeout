"""Field maps: gridded external fields as xarray/netCDF, read, written and interpolated.

A field map holds ``b_field_<i>`` (T) and/or ``e_field_<i>`` (V/m) on node coordinates
``x``, ``y``, ``z`` (m): magnet and coil maps, or fields computed offline on fitted
meshes (NGSolve, FEniCSx). :class:`FieldMap` reads and writes one, checks it with
:func:`validate_field_map` and interpolates it multilinearly to any position.

The layout:

- dims an ordered subset of ``(x, y, z)``, each with an increasing coordinate in
  ``m``; a map may have fewer dims than the positions it is evaluated at and is then
  uniform along the others (a 1D profile ``B_z(x)`` in a 3D region);
- variables ``b_field_<i>`` (``units = "T"``) and/or ``e_field_<i>``
  (``units = "V/m"``), ``<i>`` in ``x, y, z``, each on all the map's dims in that order;
- attributes ``Conventions`` (:data:`FIELD_MAP_FORMAT`), ``frame`` (``"cartesian"``, or
  ``"cylindrical"`` for the axisymmetric ``(x, r)`` plane: ``x`` axial, ``y`` the
  radius, and the ``z`` component the azimuthal one, ``x, r, theta`` right-handed),
  optionally ``frame_axes`` (the dims in a device frame, see
  :func:`~takeout.output.parse_frame_axes`) and ``source`` (what produced it).

Outside the map the field is that of the nearest boundary node (``outside="clip"``),
zero (``"zero"``) or an error (``"error"``)::

    field_map = FieldMap.from_function(
        lambda p: np.stack([0 * p[:, 0], 0 * p[:, 0], 0.01 * np.exp(-p[:, 0] ** 2 / 1e-4)], 1),
        {"x": np.linspace(0, 0.05, 101)},
        kind="magnetic",
        source="analytic",
    )
    field_map.write("magnets.nc")
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr
from methodcard import ConvergenceProblem, method
from scipy.interpolate import RegularGridInterpolator

from takeout._common import AXES, FormatError, check_coordinates, check_frame
from takeout._profiling import PROFILE_TAGS

__all__ = ["FIELD_MAP_FORMAT", "OUTSIDE", "FieldMap", "validate_field_map"]

#: The name and version of the field-map layout, in the ``Conventions`` attribute.
FIELD_MAP_FORMAT = "max-models-field-map-1.0"

#: What the field is outside the map.
OUTSIDE = ("clip", "zero", "error")

_KINDS = {"magnetic": ("b", "T"), "electric": ("e", "V/m")}


def _units(name: str) -> str:
    """Return the units of a field-map variable from its name."""
    return "T" if name.startswith("b_field_") else "V/m"


def validate_field_map(ds: xr.Dataset) -> list[str]:
    """Return what ``ds`` gets wrong against the field-map layout.

    Parameters
    ----------
    ds : xarray.Dataset
        A field map: ``b_field_<i>`` and/or ``e_field_<i>`` on ``(x[, y[, z]])``.

    Returns
    -------
    list of str
        One message per problem; empty when the map follows the layout.
    """
    problems = []
    attrs = ds.attrs
    if attrs.get("Conventions") != FIELD_MAP_FORMAT:
        problems.append(
            f"attribute Conventions is {attrs.get('Conventions')!r}, "
            f"expected {FIELD_MAP_FORMAT!r}"
        )
    problems.extend(check_frame(attrs))
    problems.extend(check_coordinates(ds, dict.fromkeys(AXES, "m")))
    dims = tuple(str(dim) for dim in ds.dims)
    if not dims or any(dim not in AXES for dim in dims):
        problems.append(f"a field map lives on (x, y, z) only, got dims {dims}")
    if not ds.data_vars:
        problems.append("the field map has no b_field_* or e_field_* variable")
    order = tuple(dim for dim in AXES if dim in ds.dims)
    for name, variable in ds.data_vars.items():
        match = re.fullmatch(r"([be])_field_([xyz])", str(name))
        if match is None:
            problems.append(
                f"{name}: not a field-map variable (b_field_<i>, e_field_<i>)"
            )
            continue
        expected = "T" if match.group(1) == "b" else "V/m"
        if variable.attrs.get("units") != expected:
            problems.append(
                f"{name}: units {variable.attrs.get('units')!r}, expected {expected!r}"
            )
        if tuple(str(dim) for dim in variable.dims) != order:
            problems.append(f"{name}: dims {variable.dims}, expected {order}")
    return problems


def _interpolation_problem() -> ConvergenceProblem:
    """``sin x cos y`` from its map, against the function, at the cell centres."""

    def exact(points: np.ndarray) -> np.ndarray:
        return np.sin(points[:, 0]) * np.cos(points[:, 1])

    def error(step: float) -> float:
        nodes = np.arange(0.0, 1.0 + 0.5 * step, step)
        field_map = FieldMap.from_function(
            lambda p: np.stack([exact(p)], axis=1),
            {"x": nodes, "y": nodes},
            kind="electric",
        )
        centres = nodes[:-1] + 0.5 * step
        mesh = np.meshgrid(centres, centres, indexing="ij")
        points = np.stack([axis.ravel() for axis in mesh], axis=-1)
        values = field_map.field("electric", points, 1)[:, 0]
        return float(np.max(np.abs(values - exact(points))))

    return ConvergenceProblem(error, steps=[0.1, 0.05, 0.025, 0.0125])


@method(
    equations={
        "multilinear": r"F(\mathbf x) = \sum_{\mathbf c \in \{0, 1\}^d} F_{\mathbf i + \mathbf c} \prod_{k=1}^{d} \left(c_k w_k + (1 - c_k)(1 - w_k)\right), \qquad w_k = \frac{x_k - x_{k, i_k}}{x_{k, i_k + 1} - x_{k, i_k}}",
        "outside": r"F(\mathbf x) = F(\operatorname{clip}(\mathbf x)) \ \text{(clip)}, \quad 0 \ \text{(zero)}",
    },
    order=2,
    assumptions=(
        r"each component on the nodes of a tensor-product mesh of $d \le 3$ increasing (not necessarily uniform) coordinates",
        r"a map with fewer dims than the positions is uniform along the missing axes",
        r"the order is in the node spacing, for a smooth field",
    ),
    problem=_interpolation_problem,
)
class FieldMap:
    """A gridded external field in the field-map layout, interpolated to positions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The map, checked with :func:`validate_field_map`.
    path : str or Path, optional
        The file it was read from, if any.
    outside : str
        What the field is outside the map: ``"clip"`` (the nearest boundary node's),
        ``"zero"`` or ``"error"``.

    Attributes
    ----------
    dataset : xarray.Dataset
        The map.
    path : Path or None
        The file it was read from or written to, if any.
    outside : str
        What the field is outside the map.
    dims : tuple of str
        The map's dims, an ordered subset of ``("x", "y", "z")``.

    Raises
    ------
    FormatError
        If ``dataset`` does not follow the layout.
    """

    def __init__(
        self, dataset: xr.Dataset, path: str | Path | None = None, outside: str = "clip"
    ) -> None:
        problems = validate_field_map(dataset)
        if problems:
            raise FormatError("not a field map:\n- " + "\n- ".join(problems))
        if outside not in OUTSIDE:
            raise ValueError(f"outside must be one of {OUTSIDE}, got {outside!r}")
        self.dataset = dataset
        self.path = None if path is None else Path(path)
        self.outside = outside
        self.dims = tuple(dim for dim in AXES if dim in dataset.dims)
        self._points = tuple(
            np.asarray(dataset[dim].values, dtype=float) for dim in self.dims
        )
        self._columns = [AXES.index(dim) for dim in self.dims]
        self._lower = np.array([axis[0] for axis in self._points])
        self._upper = np.array([axis[-1] for axis in self._points])
        self._interpolators: dict[str, RegularGridInterpolator] = {}

    @property
    def frame(self) -> str:
        """The ``frame`` attribute: ``"cartesian"`` or ``"cylindrical"``."""
        return str(self.dataset.attrs["frame"])

    @classmethod
    @sp.profile("takeout.field_map_read", tags=PROFILE_TAGS)
    def read(cls, path: str | Path, outside: str = "clip") -> FieldMap:
        """Read a field map from a netCDF file, loaded into memory.

        Parameters
        ----------
        path : str or Path
            The file.
        outside : str
            See :class:`FieldMap`.

        Returns
        -------
        FieldMap
            The map.
        """
        with xr.open_dataset(path) as opened:
            dataset = opened.load()
        return cls(dataset, path=path, outside=outside)

    @classmethod
    def from_arrays(
        cls,
        coordinates: Mapping[str, Any],
        components: Mapping[str, Any],
        frame: str = "cartesian",
        frame_axes: str | None = None,
        source: str = "",
        outside: str = "clip",
    ) -> FieldMap:
        """Build a map from node coordinates and component arrays.

        Parameters
        ----------
        coordinates : mapping of str to array
            ``{"x": nodes, ...}`` in metres, increasing.
        components : mapping of str to array
            ``{"b_field_z": values, ...}`` (T or V/m), each of the shape of the mesh of
            ``coordinates`` (in the order ``x, y, z``).
        frame : str
            The ``frame`` attribute.
        frame_axes : str, optional
            The ``frame_axes`` attribute.
        source : str
            What produced the map (a script, a solver and its mesh, a measurement).
        outside : str
            See :class:`FieldMap`.

        Returns
        -------
        FieldMap
            The map.
        """
        dims = tuple(dim for dim in AXES if dim in coordinates)
        coords = {
            dim: (dim, np.asarray(coordinates[dim], dtype=float), {"units": "m"})
            for dim in dims
        }
        variables = {
            name: (dims, np.asarray(values, dtype=float), {"units": _units(name)})
            for name, values in components.items()
        }
        attrs = {"Conventions": FIELD_MAP_FORMAT, "frame": frame, "source": source}
        if frame_axes:
            attrs["frame_axes"] = frame_axes
        return cls(xr.Dataset(variables, coords=coords, attrs=attrs), outside=outside)

    @classmethod
    def from_function(
        cls,
        function: Callable[[np.ndarray], np.ndarray],
        coordinates: Mapping[str, Any],
        kind: str = "magnetic",
        **kwargs: Any,
    ) -> FieldMap:
        """Sample a field function on the mesh of ``coordinates``.

        Parameters
        ----------
        function : callable
            The field at positions ``(N, ndim)`` (m), as ``(N, num_components)``, the
            components ``x, y, ...`` (beyond the third ignored).
        coordinates : mapping of str to array
            ``{"x": nodes, ...}`` in metres.
        kind : str
            ``"magnetic"`` (``b_field_*``) or ``"electric"`` (``e_field_*``).
        **kwargs
            Passed to :meth:`from_arrays`.

        Returns
        -------
        FieldMap
            The map.
        """
        prefix, _ = _KINDS[kind]
        dims = tuple(dim for dim in AXES if dim in coordinates)
        mesh = np.meshgrid(
            *(np.asarray(coordinates[dim], dtype=float) for dim in dims),
            indexing="ij",
        )
        positions = np.stack([axis.ravel() for axis in mesh], axis=-1)
        values = np.asarray(function(positions), dtype=float)
        values = values.reshape(positions.shape[0], -1)
        components = {
            f"{prefix}_field_{AXES[k]}": values[:, k].reshape(mesh[0].shape)
            for k in range(min(values.shape[1], 3))
        }
        return cls.from_arrays(coordinates, components, **kwargs)

    @sp.profile("takeout.field_map_write", tags=PROFILE_TAGS)
    def write(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the map to the netCDF file ``path`` and remember it there.

        Parameters
        ----------
        path : str or Path
            The file.
        **kwargs
            Passed to ``xarray.Dataset.to_netcdf``.

        Returns
        -------
        Path
            The path written.
        """
        path = Path(path)
        self.dataset.to_netcdf(path, **kwargs)
        self.path = path
        return path

    def components(self, kind: str) -> tuple[str, ...]:
        """Return the names of the components of ``kind`` in the map.

        Parameters
        ----------
        kind : str
            ``"magnetic"`` or ``"electric"``.

        Returns
        -------
        tuple of str
            For example ``("b_field_x", "b_field_z")``.
        """
        prefix, _ = _KINDS[kind]
        return tuple(
            f"{prefix}_field_{axis}"
            for axis in AXES
            if f"{prefix}_field_{axis}" in self.dataset
        )

    def _interpolate(self, name: str, positions: np.ndarray) -> np.ndarray:
        """Return the component ``name`` at ``positions`` ``(N, ndim)``."""
        if name not in self._interpolators:
            # the positions are clipped to the map, so the interpolator never extrapolates
            self._interpolators[name] = RegularGridInterpolator(
                self._points, np.asarray(self.dataset[name].values, dtype=float)
            )
        if max(self._columns) >= positions.shape[1]:
            raise ValueError(
                f"the map on {self.dims} needs positions with at least "
                f"{max(self._columns) + 1} coordinates, got {positions.shape[1]}"
            )
        points = positions[:, self._columns]
        inside = np.all((points >= self._lower) & (points <= self._upper), axis=1)
        if self.outside == "error" and not inside.all():
            bounds = {
                dim: (float(low), float(high))
                for dim, low, high in zip(
                    self.dims, self._lower, self._upper, strict=True
                )
            }
            raise ValueError(
                f"{int((~inside).sum())} positions are outside the field map {bounds}"
            )
        values = self._interpolators[name](np.clip(points, self._lower, self._upper))
        if self.outside == "zero":
            values[~inside] = 0.0
        return values

    def field(self, kind: str, positions: Any, num_components: int) -> np.ndarray:
        """Return the field of ``kind`` at ``positions`` with ``num_components``.

        Parameters
        ----------
        kind : str
            ``"magnetic"`` or ``"electric"``.
        positions : array_like
            ``(N, ndim)`` in metres, the columns ``x, y, z``.
        num_components : int
            The components returned, ``x, y, ...``; those not in the map are zero.

        Returns
        -------
        numpy.ndarray
            ``(N, num_components)`` in T or V/m.
        """
        prefix, _ = _KINDS[kind]
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        available = self.components(kind)
        if not available:
            raise ValueError(f"the field map has no {prefix}_field_* component")
        out = np.zeros((positions.shape[0], num_components))
        for k in range(num_components):
            name = f"{prefix}_field_{AXES[k]}"
            if name in available:
                out[:, k] = self._interpolate(name, positions)
        return out

    def magnetic_field(self, positions: Any) -> np.ndarray:
        """Return B (T) at ``positions`` ``(N, ndim)`` as ``(N, 3)``.

        Parameters
        ----------
        positions : array_like
            ``(N, ndim)`` in metres.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)``.
        """
        return self.field("magnetic", positions, 3)

    def electric_field(self, positions: Any) -> np.ndarray:
        """Return E (V/m) at ``positions`` ``(N, ndim)`` as ``(N, ndim)``.

        Parameters
        ----------
        positions : array_like
            ``(N, ndim)`` in metres.

        Returns
        -------
        numpy.ndarray
            ``(N, ndim)``, the components along the axes of the positions.
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        return self.field("electric", positions, positions.shape[1])

    def evaluate(
        self, coordinates: tuple[Any, ...], kind: str = "magnetic"
    ) -> np.ndarray:
        """Evaluate the field on a mesh.

        Parameters
        ----------
        coordinates : tuple of array_like
            The (up to three) ``indexing="ij"`` coordinate arrays in metres.
        kind : str
            ``"magnetic"`` (three components) or ``"electric"`` (one per coordinate).

        Returns
        -------
        numpy.ndarray
            ``(*mesh, num_components)``.
        """
        shape = np.shape(coordinates[0])
        positions = np.stack(
            [np.broadcast_to(c, shape).ravel() for c in coordinates], axis=-1
        )
        num_components = 3 if kind == "magnetic" else len(coordinates)
        return self.field(kind, positions, num_components).reshape(
            *shape, num_components
        )

    def points_function(self, kind: str) -> Callable[[np.ndarray], np.ndarray]:
        """Return the field of ``kind`` as ``F(points)``, ``(3, N)`` at ``(3, N)``.

        Parameters
        ----------
        kind : str
            ``"magnetic"`` or ``"electric"``.

        Returns
        -------
        callable
            ``F(points)`` for 3D points in a ``"cartesian"`` map, component-major.
        """
        self.field(kind, np.zeros((1, 3)), 3)  # fail now on a missing kind

        def function(points: np.ndarray) -> np.ndarray:
            return self.field(kind, np.asarray(points).T, 3).T

        return function

    def plane_function(
        self, kind: str
    ) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, ...]]:
        """Return the field of ``kind`` as ``F(x, y) -> (F_x, F_y, F_z)`` on a plane.

        Parameters
        ----------
        kind : str
            ``"magnetic"`` or ``"electric"``.

        Returns
        -------
        callable
            ``F(x, y)`` with arrays of any shape: ``(x, y)`` planar, or
            ``(x, r) -> (F_x, F_r, F_theta)`` in a ``"cylindrical"`` map.

        Raises
        ------
        ValueError
            If the map has a ``z`` dim.
        """
        if "z" in self.dims:
            raise ValueError(
                f"a map on {self.dims} does not fit a 2D plane; it needs dims in (x, y)"
            )
        self.field(kind, np.zeros((1, 2)), 3)  # fail now on a missing kind

        def function(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
            values = self.field(kind, np.stack([np.ravel(x), np.ravel(y)], -1), 3)
            shape = np.shape(x)
            return tuple(values[:, k].reshape(shape) for k in range(3))

        return function
