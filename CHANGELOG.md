# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-11-25

First release.

### Added

- `takeout.output`: the output format of the max-models codes (moved from picnic's
  `picnic.input_output.output_format`): the layout, `validate` and `check`, `write` and
  `read` (netCDF, checked both ways), `marker_samples`, `moment_units`, `parse_frame_axes`
  and `provenance`, which now takes the code first and finds its checkout itself.
- `takeout.field_map`: field maps (`FieldMap`, `validate_field_map`, `FIELD_MAP_FORMAT`),
  merged from picnic's `picnic.parameters.field_map` and beamtracer's copy
  `beamtracer.field_map`, with beamtracer's method card of the interpolation and its
  `points_function` and `plane_function`.
- One error class, `FormatError`, for both layouts; opt-in scope-profiler regions on the
  file input and output; the "Output format" and "Field maps" examples.
- `AXES`, the order of the spatial dims.
