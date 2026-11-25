"""The output format: one ``xarray.Dataset`` layout for the output of any model.

The plotting, the diagnostics and code-to-code comparisons read a run without caring
which code produced it: picnic writes this layout, and so do the readers of other codes
(SAPIC's). :func:`validate` lists what a dataset gets wrong, :func:`check` raises on it,
:func:`write` and :func:`read` store it as netCDF.

**Dimensions and coordinates**

=============  ====================================================================
``t``          time of the record, s, increasing
``species``    species names (strings)
``x, y, z``    node coordinates of the grid, m, increasing; the spatial dims of every
               variable are a prefix of ``(x, y, z)`` and the same for all variables
``sample``     index of a marker sample (no coordinate values)
=============  ====================================================================

The event tables of a marker tracker have one dim each, ``birth``, ``absorption`` and ``crossing`` (one entry per event),
with the variables ``birth_*``, ``absorbed_*`` and ``crossing_*``, each with units.

**Variables** (each with a ``units`` attribute; ``t`` and ``species`` lead when present)

====================================  ====================  ===========================
name                                  dims                  units
====================================  ====================  ===========================
``density``                           (t, species, space)   ``1/m^3``
``flux_<i>``                          (t, species, space)   ``1/(m^2 s)`` (``n u_i``)
``moment_<ij...>`` (order k >= 2)     (t, species, space)   :func:`moment_units`
``potential``                         (t, space)            ``V``
``e_field_<i>``                       (t, space)            ``V/m``
``b_field_<i>``                       (space) or (t, space) ``T``
``sample_id``                         (t, species, sample)  ``1``, marker id, -1 = none
``sample_<x|y|z>``                    (t, species, sample)  ``m``
``sample_v<x|y|z>``                   (t, species, sample)  ``m/s``
``sample_weight``                     (t, species, sample)  ``1`` (particles per marker)
====================================  ====================  ===========================

Components ``<i>`` are ``x``, ``y`` or ``z``; the moment indices are sorted
(``moment_xy``, never ``moment_yx``). Other variables are allowed if they carry
``units``. Missing samples are padded with ``sample_id = -1`` and NaN.

**Global attributes**

- ``Conventions``: :data:`FORMAT` (the version of this layout);
- ``frame``: ``"cartesian"`` (``x, y, z``) or ``"cylindrical"`` (an axisymmetric
  ``(x, r)`` grid: ``x`` axial, ``y`` the radius; the vector components ``_x, _y, _z``
  are then axial, radial and azimuthal, the right-handed set ``(x, r, theta)`` of the
  r-z push, and there is no ``z`` dim); ``frame_axes`` (optional) maps the
  dims to the frame of a magnetron device (``z`` normal to the target into the plasma, ``theta`` or ``s`` along the racetrack, ``r`` across it), with
  a sign where an axis points the other way, e.g. ``"x=z y=-s z=r"`` for Boeuf's
  axial-azimuthal plane with ``+y`` the ``E x B`` direction over ``B_r > 0``; three
  mapped axes must form a right-handed set;
- ``mass``, ``charge``: one value per species (kg, C) when there is a ``species`` dim;
- provenance (:func:`provenance`): ``code``, ``code_version``, ``git_hash``,
  ``parameters`` (the run's configuration as YAML) and ``created`` (ISO 8601, UTC).

Signs follow the coordinates (a positive ``flux_y`` points along ``+y``); SI units
throughout.

**Field maps** (gridded external fields: magnet maps, coil fields, fields computed
offline on fitted meshes) use the same coordinates, names, units and frame attributes,
with ``Conventions`` :data:`takeout.field_map.FIELD_MAP_FORMAT`: dims an ordered subset of ``(x, y,
z)`` (a map may have fewer dims than the model that reads it; it is then uniform along
the others), the variables ``b_field_<i>`` (T) and/or ``e_field_<i>`` (V/m) on all of
the map's dims, and optionally a ``source`` attribute saying what produced it.
:mod:`takeout.field_map` checks, reads, writes and interpolates them.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata, util
from pathlib import Path
from typing import Any

import numpy as np
import scope_profiler as sp
import xarray as xr

from takeout._common import (
    AXES,
    FRAMES,
    FormatError,
    check_coordinates,
    check_frame,
    parse_frame_axes,
)
from takeout._profiling import PROFILE_TAGS

__all__ = [
    "AXES",
    "FORMAT",
    "FRAMES",
    "PROVENANCE",
    "FormatError",
    "check",
    "git_hash",
    "marker_samples",
    "moment_units",
    "parse_frame_axes",
    "provenance",
    "read",
    "validate",
    "write",
]

#: The name and version of the layout, stored in the ``Conventions`` attribute.
FORMAT = "max-models-output-1.0"

#: The provenance attributes every dataset written to a file carries.
PROVENANCE = ("code", "code_version", "git_hash", "parameters", "created")

#: The dims of the event tables of a marker tracker.
EVENT_DIMS = ("birth", "absorption", "crossing")

_COORDINATE_UNITS = {"t": "s", "x": "m", "y": "m", "z": "m"}
_FIXED_UNITS = {
    "density": "1/m^3",
    "potential": "V",
    "sample_id": "1",
    "sample_weight": "1",
}
_PATTERNS = (
    (re.compile(r"flux_([xyz])"), "moment"),
    (re.compile(r"moment_([xyz]{2,})"), "moment"),
    (re.compile(r"e_field_([xyz])"), "field"),
    (re.compile(r"b_field_([xyz])"), "static"),
    (re.compile(r"sample_(v?)([xyz])"), "sample"),
)
_KINDS = {
    "density": "moment",
    "potential": "field",
    "sample_id": "sample",
    "sample_weight": "sample",
}
#: The non-spatial dims each kind of variable must lead with (optional ones in []).
_LEADING = {
    "moment": ("[t]", "[species]"),
    "field": ("[t]",),
    "static": ("[t]",),
    "sample": ("[t]", "[species]", "sample"),
}


def moment_units(order: int) -> str:
    """Return the units of the moment ``n <v^order>`` (``m^(order - 3) s^-order``).

    Parameters
    ----------
    order : int
        The order of the moment: 0 the density, 1 the flux, 2 ``n <v v>``, ...

    Returns
    -------
    str
        ``1/m^3``, ``1/(m^2 s)``, ``1/(m s^2)``, ``1/s^3``, then ``m^k/s^n``.
    """
    named = {0: "1/m^3", 1: "1/(m^2 s)", 2: "1/(m s^2)", 3: "1/s^3"}
    if order in named:
        return named[order]
    return f"m^{order - 3}/s^{order}"


def marker_samples(
    ids: Any,
    positions: Any,
    velocities: Any,
    weights: Any,
    num_samples: int | None = None,
) -> xr.Dataset:
    """Return markers as the sample variables of the format, over a ``sample`` dim.

    Parameters
    ----------
    ids : array_like or None
        The marker ids, ``(n,)``; ``None`` when they are not known (no ``sample_id``).
    positions : array_like
        ``(ndim, n)`` positions, m: ``sample_x``, ``sample_y``, ...
    velocities : array_like
        ``(nvel, n)`` velocities, m/s: ``sample_vx``, ...
    weights : array_like
        ``(n,)`` physical particles per marker: ``sample_weight``.
    num_samples : int, optional
        The length of the ``sample`` dim (default ``n``); the markers past ``n`` are
        padded with ``sample_id = -1`` and NaN, those past ``num_samples`` dropped.

    Returns
    -------
    xarray.Dataset
        The sample variables with their units, no attributes.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    weights = np.asarray(weights, dtype=float).reshape(1, -1)
    count = weights.shape[1]
    size = count if num_samples is None else int(num_samples)
    kept = min(count, size)
    table = np.full((positions.shape[0] + velocities.shape[0] + 1, size), np.nan)
    table[:, :kept] = np.vstack([positions, velocities, weights])[:, :kept]
    variables: dict[str, Any] = {}
    if ids is not None:
        padded = np.full(size, -1, dtype=np.int64)
        padded[:kept] = np.asarray(ids, dtype=np.int64).reshape(-1)[:kept]
        variables["sample_id"] = ("sample", padded, {"units": "1"})
    for k in range(positions.shape[0]):
        variables[f"sample_{AXES[k]}"] = ("sample", table[k], {"units": "m"})
    for k in range(velocities.shape[0]):
        row = table[positions.shape[0] + k]
        variables[f"sample_v{AXES[k]}"] = ("sample", row, {"units": "m/s"})
    variables["sample_weight"] = ("sample", table[-1], {"units": "1"})
    return xr.Dataset(variables)


def git_hash(path: str | Path) -> str:
    """Return the git commit of the checkout holding ``path``.

    Parameters
    ----------
    path : str or Path
        A file or directory inside the checkout.

    Returns
    -------
    str
        The commit hash, with ``-dirty`` appended when tracked files are modified, or
        ``"unknown"`` outside a git checkout (an installed wheel).
    """
    directory = Path(path).resolve()
    if directory.is_file():
        directory = directory.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def _code_version(code: str) -> str:
    """Return the installed version of the distribution ``code``, or ``"unknown"``."""
    try:
        return metadata.version(code)
    except (metadata.PackageNotFoundError, ValueError):
        return "unknown"


def _source(code: str) -> Path | None:
    """Return the directory of the package ``code`` imports as, or None."""
    try:
        spec = util.find_spec(code.replace("-", "_"))
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def provenance(
    code: str, parameters: Any = None, source: str | Path | None = None
) -> dict[str, str]:
    """Return the provenance attributes of a dataset.

    Parameters
    ----------
    code : str
        The code that wrote the data, as its distribution name; its version is looked
        up.
    parameters : Any, optional
        The run's configuration: a string (YAML), an object whose ``str`` is its
        configuration file, or ``None`` (stored as ``""``).
    source : str or Path, optional
        A path inside the code's checkout, for :func:`git_hash`; by default the
        directory of the package ``code`` imports as (dashes read as underscores).

    Returns
    -------
    dict of str to str
        ``code``, ``code_version``, ``git_hash``, ``parameters`` and ``created``.
    """
    if source is None:
        source = _source(code)
    return {
        "code": code,
        "code_version": _code_version(code),
        "git_hash": "unknown" if source is None else git_hash(source),
        "parameters": "" if parameters is None else str(parameters),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _kind(name: str) -> tuple[str, int] | None:
    """Return the kind and (for moments) the order of a known variable name."""
    if name in _KINDS:
        return _KINDS[name], 0
    for pattern, kind in _PATTERNS:
        match = pattern.fullmatch(name)
        if match is None:
            continue
        if name.startswith("moment_"):
            indices = match.group(1)
            if "".join(sorted(indices)) != indices:
                return None
            return kind, len(indices)
        return kind, 1
    return None


def _expected_units(name: str, kind: str, order: int) -> str:
    """Return the units the format prescribes for the known variable ``name``."""
    if name in _FIXED_UNITS:
        return _FIXED_UNITS[name]
    if kind == "moment":
        return moment_units(order)
    if kind == "field":
        return "V/m"
    if kind == "static":
        return "T"
    return "m/s" if name.startswith("sample_v") else "m"


def _check_leading(name: str, kind: str, dims: tuple[str, ...]) -> list[str]:
    """Check that ``dims`` start with the non-spatial dims of ``kind``, in order."""
    problems = []
    position = 0
    for entry in _LEADING[kind]:
        optional = entry.startswith("[")
        dim = entry.strip("[]")
        if position < len(dims) and dims[position] == dim:
            position += 1
        elif not optional:
            problems.append(f"{name}: dims {dims} miss {dim!r}")
    rest = dims[position:]
    if kind == "sample":
        if rest:
            problems.append(f"{name}: dims {dims} end with {rest}, expected 'sample'")
        return problems
    if rest != AXES[: len(rest)]:
        problems.append(f"{name}: dims {dims} are not (t, species, x, y, z)-ordered")
    return problems


def _check_coordinates(ds: xr.Dataset) -> list[str]:
    """Check the units and the order of the coordinates, and the dims."""
    problems = check_coordinates(ds, _COORDINATE_UNITS)
    if "species" in ds.dims and "species" not in ds.coords:
        problems.append("dimension 'species' has no coordinate")
    known = {*_COORDINATE_UNITS, "species", "sample", *EVENT_DIMS}
    unknown = {str(dim) for dim in ds.dims} - known
    if unknown:
        problems.append(f"unknown dimensions {sorted(unknown)}")
    return problems


def _check_attributes(ds: xr.Dataset, require_provenance: bool) -> list[str]:
    """Check the global attributes: format, frame, species and provenance."""
    problems = []
    attrs = ds.attrs
    if attrs.get("Conventions") != FORMAT:
        problems.append(
            f"attribute Conventions is {attrs.get('Conventions')!r}, expected {FORMAT!r}"
        )
    problems.extend(check_frame(attrs))
    if "species" in ds.dims:
        for key in ("mass", "charge"):
            value = np.atleast_1d(np.asarray(attrs.get(key, []), dtype=float))
            if value.size != ds.sizes["species"]:
                problems.append(
                    f"attribute {key} has {value.size} values for "
                    f"{ds.sizes['species']} species"
                )
    if require_provenance:
        for key in PROVENANCE:
            if not isinstance(attrs.get(key), str):
                problems.append(f"provenance attribute {key!r} is missing")
    return problems


def validate(ds: xr.Dataset, require_provenance: bool = True) -> list[str]:
    """Return what ``ds`` gets wrong against the output format.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to check.
    require_provenance : bool
        Whether the provenance attributes (:data:`PROVENANCE`) must be there; a
        snapshot in memory has none, a file always has them.

    Returns
    -------
    list of str
        One message per problem; empty when the dataset follows the format.
    """
    problems = _check_attributes(ds, require_provenance) + _check_coordinates(ds)
    spaces = set()
    for name, variable in ds.data_vars.items():
        dims = tuple(str(dim) for dim in variable.dims)
        units = variable.attrs.get("units")
        known = _kind(str(name))
        if known is None:
            if str(name).startswith(("flux_", "moment_", "e_field_", "b_field_")):
                problems.append(f"{name}: not a valid component name")
            elif units is None:
                problems.append(f"{name}: no units")
            continue
        kind, order = known
        expected = _expected_units(str(name), kind, order)
        if units != expected:
            problems.append(f"{name}: units {units!r}, expected {expected!r}")
        problems.extend(_check_leading(str(name), kind, dims))
        if kind != "sample":
            spaces.add(tuple(dim for dim in dims if dim in AXES))
    if len(spaces) > 1:
        problems.append(f"the variables live on different grids: {sorted(spaces)}")
    return problems


def check(ds: xr.Dataset, require_provenance: bool = True) -> xr.Dataset:
    """Raise if ``ds`` does not follow the format, else return it.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to check.
    require_provenance : bool
        See :func:`validate`.

    Returns
    -------
    xarray.Dataset
        ``ds`` itself.

    Raises
    ------
    FormatError
        Listing every problem :func:`validate` found.
    """
    problems = validate(ds, require_provenance)
    if problems:
        raise FormatError(
            "the dataset does not follow the output format:\n- " + "\n- ".join(problems)
        )
    return ds


def _encodable(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``attrs`` with lists as arrays, the forms netCDF stores."""
    return {
        key: np.asarray(value) if isinstance(value, list | tuple) else value
        for key, value in attrs.items()
    }


@sp.profile("takeout.write", tags=PROFILE_TAGS)
def write(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> Path:
    """Check ``ds`` and write it to the netCDF file ``path``.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset in the format, with provenance.
    path : str or Path
        The file to write.
    **kwargs
        Passed to ``xarray.Dataset.to_netcdf`` (``engine``, ``encoding``, ...).

    Returns
    -------
    Path
        The path written.
    """
    check(ds)
    path = Path(path)
    out = ds.copy()
    out.attrs = _encodable(ds.attrs)
    out.to_netcdf(path, **kwargs)
    return path


@sp.profile("takeout.read", tags=PROFILE_TAGS)
def read(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Read a dataset written by :func:`write` and check it.

    Parameters
    ----------
    path : str or Path
        The netCDF file.
    **kwargs
        Passed to ``xarray.open_dataset``.

    Returns
    -------
    xarray.Dataset
        The dataset, loaded into memory; ``mass`` and ``charge`` as lists.
    """
    with xr.open_dataset(path, **kwargs) as opened:
        ds = opened.load()
    for key in ("mass", "charge"):
        if key in ds.attrs:
            ds.attrs[key] = np.atleast_1d(ds.attrs[key]).tolist()
    return check(ds)
