# takeout


<!-- README.md is generated from README.qmd: edit the .qmd and run `make readme`. -->

[![Tests](https://github.com/max-models/takeout/actions/workflows/test_pytest.yml/badge.svg)](https://github.com/max-models/takeout/actions/workflows/test_pytest.yml)
[![Static
analysis](https://github.com/max-models/takeout/actions/workflows/static_analysis.yml/badge.svg)](https://github.com/max-models/takeout/actions/workflows/static_analysis.yml)
[![Docs](https://github.com/max-models/takeout/actions/workflows/docs.yml/badge.svg)](https://max-models.github.io/takeout/)
[![codecov](https://codecov.io/gh/max-models/takeout/branch/main/graph/badge.svg)](https://codecov.io/gh/max-models/takeout)
[![PyPI](https://img.shields.io/pypi/v/takeout.png)](https://pypi.org/project/takeout/)
[![Python](https://img.shields.io/pypi/pyversions/takeout.png)](https://pypi.org/project/takeout/)

Gridded data that crosses between the max-models codes, as xarray
datasets with fixed names, dims and units:

- **the output format** (`takeout.output`): one layout for the output of
  any model, with the grid moments of each species, the fields, marker
  samples and the run’s provenance; a validator that lists what a
  dataset gets wrong, and netCDF files that are checked on writing and
  on reading;
- **field maps** (`takeout.field_map`): external fields on a grid
  (magnet and coil maps, fields computed offline), read, written and
  interpolated multilinearly to any position;
- **analyses** (`takeout.analysis`): functions of these datasets that
  do not need the code that ran: the detection and naming of
  instabilities (with the `analysis` extra), the ion-acoustic
  dispersion relation, the dominant wave of E x B runs and rotating
  frames.

picnic and beamtracer write and read these layouts, and other codes’
readers (SAPIC’s) produce them, so plots, diagnostics and code-to-code
comparisons do not depend on which code made a run.

Documentation: <https://max-models.github.io/takeout/>

## Install

``` bash
pip install takeout
```

or from a checkout, with the test, docs and development tools:

``` bash
make install    # uv sync --extra dev, plus the pre-commit hooks
pip install -e ".[dev]"   # the same with pip
```

## Example

``` python
import numpy as np
import takeout

field_map = takeout.FieldMap.from_function(
    lambda p: np.stack([0.05 + 0 * p[:, 0], 0 * p[:, 0], 0 * p[:, 0]], axis=1),
    {"x": np.linspace(0.0, 0.1, 11)},
    kind="magnetic",
    source="uniform 50 mT",
)
field_map.write("uniform.nc")
takeout.FieldMap.read("uniform.nc").magnetic_field(np.array([[0.03, 0.0, 0.0]]))
```

A run is checked against the output format with `takeout.validate(ds)`
(a list of problems, empty when it follows the format), stored with
`takeout.write(ds, "run.nc")` and loaded with `takeout.read("run.nc")`.

## Development

Formatting and linting use [ruff](https://docs.astral.sh/ruff/) and
[ssort](https://github.com/bwhmather/ssort), type checking
[pyright](https://microsoft.github.io/pyright/), run by
[pre-commit](https://pre-commit.com/) and in CI:

``` bash
make lint     # ruff check, ruff format --check, ssort, pyright
make test     # pytest with coverage
```

## Build docs

The documentation in `docs/` is an Astro + Starlight site: hand-written
pages, the notebooks in `tutorials/` executed and published as pages,
complete Python scripts in `examples/` published with their plots, and
the API reference generated from the docstrings with
[starlight-pydocs](https://ewels.github.io/starlight-pydocs/). It needs
Node 22 or newer.

``` bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py and publish their plots
make docs-dev         # live preview at http://localhost:4321/takeout/
make docs-build       # the static site in docs/dist
```

## Build the README

`README.md` is rendered from `README.qmd` with
[Quarto](https://quarto.org/):

``` bash
make readme
```

## Releases

Before merging a release to `main`, update the version in
`pyproject.toml` and add the release notes to `CHANGELOG.md`. The push
to `main` creates a GitHub release with a `vX.Y.Z` tag and publishes the
package to PyPI with trusted publishing (OIDC). The one-time PyPI and
GitHub configuration is described in the [publishing
tutorial](https://max-models.github.io/takeout/development/publishing/).
