"""takeout: the output format and the field maps shared by the max-models codes.

- :mod:`takeout.output`: one ``xarray.Dataset`` layout for model output (fields,
  moments, marker samples, provenance), its validator and netCDF files;
- :mod:`takeout.field_map`: gridded external fields in the same coordinates and units,
  read, written and interpolated (:class:`~takeout.field_map.FieldMap`).
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from takeout._profiling import PROFILE_TAGS
from takeout.field_map import FIELD_MAP_FORMAT, FieldMap, validate_field_map
from takeout.output import (
    AXES,
    FORMAT,
    FRAMES,
    PROVENANCE,
    FormatError,
    check,
    marker_samples,
    moment_units,
    parse_frame_axes,
    provenance,
    read,
    validate,
    write,
)

__all__ = [
    "AXES",
    "FIELD_MAP_FORMAT",
    "FORMAT",
    "FRAMES",
    "PROVENANCE",
    "FieldMap",
    "FormatError",
    "check",
    "marker_samples",
    "moment_units",
    "parse_frame_axes",
    "provenance",
    "read",
    "validate",
    "validate_field_map",
    "write",
]

try:
    __version__ = version("takeout")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)
