"""What the output format and the field maps share: axes, frames, the error, the checks."""

from __future__ import annotations

import re

import numpy as np
import xarray as xr

#: The frames a dataset or a field map can be in.
FRAMES = ("cartesian", "cylindrical")

#: The spatial dims, in their order.
AXES = ("x", "y", "z")

#: The axes of the device frame that ``frame_axes`` maps the dims to.
DEVICE_AXES = ("r", "theta", "z")


class FormatError(ValueError):
    """A dataset does not follow the output format or the field-map layout."""


def parse_frame_axes(text: str) -> dict[str, tuple[int, str]]:
    """Parse a ``frame_axes`` attribute such as ``"x=z y=-s z=r"``.

    Parameters
    ----------
    text : str
        Space-separated ``<dim>=[-]<device axis>`` items; the device axes are ``r``,
        ``theta`` (or ``s``, the arc length along it) and ``z``.

    Returns
    -------
    dict of str to (int, str)
        ``{dim: (sign, device axis)}`` with ``s`` reported as ``theta``.

    Raises
    ------
    FormatError
        On a malformed item, an unknown dim or axis, a repeated axis, or three mapped
        axes that are not right-handed.
    """
    mapping: dict[str, tuple[int, str]] = {}
    for item in text.split():
        match = re.fullmatch(r"([xyz])=(-?)(r|theta|s|z)", item)
        if match is None:
            raise FormatError(f"frame_axes: cannot read {item!r}")
        dim, minus, axis = match.groups()
        axis = "theta" if axis == "s" else axis
        if dim in mapping:
            raise FormatError(f"frame_axes: {dim} is mapped twice")
        if any(axis == other for _, other in mapping.values()):
            raise FormatError(f"frame_axes: {axis} is mapped twice")
        mapping[dim] = (-1 if minus else 1, axis)
    if len(mapping) == 3:
        # the matrix taking (x, y, z) to (r, theta, z) must be a rotation
        matrix = np.zeros((3, 3))
        for dim, (sign, axis) in mapping.items():
            matrix[DEVICE_AXES.index(axis), AXES.index(dim)] = sign
        if np.linalg.det(matrix) < 0:
            raise FormatError(
                f"frame_axes {text!r} is left-handed; flip the sign of one axis"
            )
    return mapping


def check_frame(attrs: dict) -> list[str]:
    """Check the ``frame`` and ``frame_axes`` attributes.

    Parameters
    ----------
    attrs : dict
        The global attributes of a dataset.

    Returns
    -------
    list of str
        One message per problem.
    """
    problems = []
    frame = attrs.get("frame")
    if frame not in FRAMES:
        problems.append(f"attribute frame is {frame!r}, expected one of {FRAMES}")
    if "frame_axes" in attrs:
        try:
            mapping = parse_frame_axes(str(attrs["frame_axes"]))
        except FormatError as error:
            problems.append(str(error))
        else:
            if frame == "cylindrical" and mapping:
                problems.append("frame_axes is for a cartesian frame")
    return problems


def check_coordinates(ds: xr.Dataset, units: dict[str, str]) -> list[str]:
    """Check that the dims in ``units`` have coordinates with those units, increasing.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset.
    units : dict of str to str
        The units of each coordinate that may be there, e.g. ``{"x": "m"}``.

    Returns
    -------
    list of str
        One message per problem.
    """
    problems = []
    for dim, expected in units.items():
        if dim not in ds.dims:
            continue
        if dim not in ds.coords:
            problems.append(f"dimension {dim!r} has no coordinate")
            continue
        if ds[dim].attrs.get("units") != expected:
            problems.append(
                f"coordinate {dim!r} has units {ds[dim].attrs.get('units')!r}, "
                f"expected {expected!r}"
            )
        values = np.asarray(ds[dim].values, dtype=float)
        if values.size > 1 and not np.all(np.diff(values) > 0):
            problems.append(f"coordinate {dim!r} is not increasing")
    return problems
