---
title: Quickstart
description: Check a dataset against the output format, store it, and interpolate a field map.
---

First, ensure that `takeout` is [installed](/takeout/getting-started/installation/).

## The output format

A dataset in the output format has the dims `t`, `species` and `x` (then `y`, `z`) in that
order, a `units` attribute on every coordinate and variable, and global attributes saying
which layout it is, its frame, the species' mass and charge and its provenance:

```python
import numpy as np
import xarray as xr

import takeout

x = np.linspace(0.0, 0.02, 33)
run = xr.Dataset(
    {"potential": (("x",), np.sin(np.pi * x / 0.02), {"units": "V"})},
    coords={"x": ("x", x, {"units": "m"})},
    attrs={
        "Conventions": takeout.FORMAT,
        "frame": "cartesian",
        **takeout.provenance("takeout"),
    },
)
assert takeout.validate(run) == []
takeout.write(run, "run.nc")
back = takeout.read("run.nc")
```

`validate` returns one message per problem (an empty list when the dataset follows the
format), `check` raises a `FormatError` listing them, and `write` and `read` check the
dataset on the way to and from the file. The [Output format](/takeout/examples/output-format/)
example builds a run with two species and marker samples.

## Field maps

```python
from takeout import FieldMap

field_map = FieldMap.from_arrays(
    {"x": [0.0, 0.05, 0.1]}, {"b_field_x": [0.15, 0.05, 0.15]}, source="three nodes"
)
field_map.write("mirror.nc")
FieldMap.read("mirror.nc").magnetic_field(np.array([[0.025, 0.0, 0.0]]))  # (1, 3), T
```

A map with fewer dims than the positions is uniform along the others; outside the grid the
field is clipped to the boundary nodes (`outside="clip"`), zero (`"zero"`) or an error
(`"error"`). See the [Field maps](/takeout/examples/field-maps/) example.

See the [API reference](/takeout/api/takeout/) for every module and function.

## Profiling

takeout's file input and output are [scope-profiler](https://github.com/max-models/scope-profiler)
regions (`takeout.write`, `takeout.read`, `takeout.field_map_read`, `takeout.field_map_write`),
tagged with the opt-in tag `takeout`: they are off by default and cost a fraction of a
microsecond per call. A code that profiles itself with scope-profiler turns them on with the
tag:

```python
import scope_profiler as sp

sp.setup(enable_tags=["takeout"])
```

or without editing the code, with `SCOPE_PROFILER_ENABLE_TAGS=takeout` or
`scope-profiler run --enable-tag takeout`.
