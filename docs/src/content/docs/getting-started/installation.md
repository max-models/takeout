---
title: Installation
description: Create a Python environment and install takeout.
---

From PyPI:

```bash
pip install takeout
```

From a checkout, create and activate a Python environment (3.10 or newer):

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
```

and install the package and its requirements with pip:

```bash
pip install -e .
```

It needs NumPy, SciPy, xarray, methodcard (the method card of the interpolation) and
scope-profiler.

## Optional extras

| Extra  | Installs                                                |
| ------ | ------------------------------------------------------- |
| `test` | `pytest` and `coverage`                                 |
| `docs` | the notebook runner, griffe for the API reference, plasma-plots for the figures |
| `dev`  | formatters and linters, plus the `test` and `docs` extras |

```bash
pip install -e ".[dev]"
```
