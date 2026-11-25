"""The opt-in scope-profiler regions: off unless the application enables the tag."""

from __future__ import annotations

import pytest
import scope_profiler as sp

from takeout import FieldMap


@pytest.fixture(autouse=True)
def _fresh_profiler(monkeypatch):
    """Keep the profiling state of these tests out of the others."""
    monkeypatch.delenv("SCOPE_PROFILER_ENABLE_TAGS", raising=False)
    sp.reset()
    yield
    sp.reset()


def _region_names(tmp_path, **setup_kwargs) -> set[str]:
    sp.setup(output=None, **setup_kwargs)
    # an application region, recorded either way
    with sp.region("user.run"):
        FieldMap.from_arrays({"x": [0.0, 1.0]}, {"b_field_z": [0.0, 1.0]}).write(
            tmp_path / "map.nc"
        )
    results = sp.finalize(verbose=False, return_results=True)
    return set(results.region_names)


def test_regions_are_off_by_default(tmp_path) -> None:
    assert _region_names(tmp_path) == {"user.run"}


@pytest.mark.parametrize("tags", [["takeout"], ["*"]])
def test_enabling_the_tag_records_the_regions(tmp_path, tags: list[str]) -> None:
    names = _region_names(tmp_path, enable_tags=tags)
    assert names == {"takeout.field_map_write", "user.run"}
