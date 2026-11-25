"""The output format: the validator, the netCDF files, marker samples and provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import takeout
from takeout import output
from takeout.output import (
    FORMAT,
    FormatError,
    check,
    git_hash,
    marker_samples,
    moment_units,
    parse_frame_axes,
    provenance,
    read,
    validate,
    write,
)


@pytest.fixture
def run() -> xr.Dataset:
    """A small 2D run of two species in the format, with provenance."""
    t = np.array([0.0, 1e-9, 2e-9])
    x = np.linspace(0.0, 1e-2, 5)
    y = np.linspace(0.0, 4e-3, 3)
    rng = np.random.default_rng(0)
    moment = (3, 2, 5, 3)
    field = (3, 5, 3)
    samples = xr.concat(
        [
            xr.concat(
                [
                    marker_samples(
                        np.arange(4),
                        rng.uniform(size=(2, 4)),
                        rng.normal(size=(3, 4)),
                        np.ones(4),
                        num_samples=6,
                    )
                    for _ in range(2)
                ],
                dim="species",
            )
            for _ in t
        ],
        dim="t",
    )
    ds = xr.Dataset(
        {
            "density": (("t", "species", "x", "y"), rng.uniform(size=moment)),
            "flux_x": (("t", "species", "x", "y"), rng.normal(size=moment)),
            "moment_xy": (("t", "species", "x", "y"), rng.normal(size=moment)),
            "potential": (("t", "x", "y"), rng.normal(size=field)),
            "e_field_x": (("t", "x", "y"), rng.normal(size=field)),
            "b_field_z": (("x", "y"), np.full((5, 3), 0.02)),
        },
        coords={
            "t": ("t", t, {"units": "s"}),
            "species": ("species", ["electrons", "ions"]),
            "x": ("x", x, {"units": "m"}),
            "y": ("y", y, {"units": "m"}),
        },
        attrs={
            "Conventions": FORMAT,
            "frame": "cartesian",
            "frame_axes": "x=z y=-s",
            "mass": [9.109e-31, 6.6e-26],
            "charge": [-1.602e-19, 1.602e-19],
            **provenance("takeout", parameters="a: 1"),
        },
    )
    for name, units in (
        ("density", "1/m^3"),
        ("flux_x", "1/(m^2 s)"),
        ("moment_xy", "1/(m s^2)"),
        ("potential", "V"),
        ("e_field_x", "V/m"),
        ("b_field_z", "T"),
    ):
        ds[name].attrs["units"] = units
    return ds.merge(samples)


def test_a_dataset_in_the_format_passes(run: xr.Dataset) -> None:
    assert validate(run) == []
    assert check(run) is run
    assert takeout.validate is validate


def test_the_netcdf_file_reads_back_the_same(run: xr.Dataset, tmp_path: Path) -> None:
    path = write(run, tmp_path / "run.nc")
    back = read(path)
    xr.testing.assert_allclose(back, run)
    assert back.attrs["mass"] == run.attrs["mass"]
    assert back.attrs["git_hash"] == run.attrs["git_hash"]


def test_the_validator_reports_each_problem(run: xr.Dataset, tmp_path: Path) -> None:
    bad = run.copy()
    bad.attrs = {**run.attrs, "Conventions": "other", "frame": "polar"}
    del bad.attrs["git_hash"]
    bad.attrs["mass"] = [1.0]
    bad["density"] = bad["density"].assign_attrs(units="1/cm^3")
    bad["flux_x"] = bad["flux_x"].transpose("species", "t", "x", "y")
    bad["moment_yx"] = bad["moment_xy"]
    bad["e_field_q"] = bad["e_field_x"]
    bad["extra"] = bad["potential"].copy()
    del bad["extra"].attrs["units"]
    bad = bad.assign_coords(y=bad["y"].assign_attrs(units="cm"))
    text = "\n".join(validate(bad))
    for expected in (
        "Conventions",
        "frame is 'polar'",
        "'git_hash' is missing",
        "mass has 1 values for 2 species",
        "density: units '1/cm^3'",
        "flux_x: dims",
        "moment_yx: not a valid component name",
        "e_field_q: not a valid component name",
        "extra: no units",
        "coordinate 'y' has units 'cm'",
    ):
        assert expected in text, expected
    with pytest.raises(FormatError, match="does not follow"):
        check(bad)
    with pytest.raises(FormatError):
        write(bad, tmp_path / "never-written.nc")
    assert not (tmp_path / "never-written.nc").exists()


def test_grids_samples_and_coordinates_are_checked(run: xr.Dataset) -> None:
    other_grid = run.copy()
    other_grid["b_field_z"] = other_grid["b_field_z"].isel(y=0)
    assert any("different grids" in p for p in validate(other_grid))
    samples = run.copy()
    samples["sample_x"] = samples["sample_x"].isel(sample=0)
    assert any("miss 'sample'" in p for p in validate(samples))
    samples = run.copy()
    samples["sample_vx"] = samples["sample_vx"].transpose("t", "sample", "species")
    assert any("expected 'sample'" in p for p in validate(samples))
    reversed_t = run.isel(t=slice(None, None, -1))
    assert any("'t' is not increasing" in p for p in validate(reversed_t))
    no_coordinate = run.drop_vars(["x", "species"])
    problems = validate(no_coordinate)
    assert "dimension 'x' has no coordinate" in problems
    assert "dimension 'species' has no coordinate" in problems
    extra_dim = run.expand_dims(run=1)
    assert any("unknown dimensions ['run']" in p for p in validate(extra_dim))
    cylindrical = run.copy()
    cylindrical.attrs = {**run.attrs, "frame": "cylindrical"}
    assert "frame_axes is for a cartesian frame" in validate(cylindrical)
    left = run.copy()
    left.attrs = {**run.attrs, "frame_axes": "x=z y=s z=r"}
    assert any("left-handed" in p for p in validate(left))


def test_a_snapshot_follows_the_format_without_provenance(run: xr.Dataset) -> None:
    snapshot = run.isel(t=-1, drop=True).drop_vars(
        [name for name in run.data_vars if name.startswith("sample_")]
    )
    snapshot.attrs = {
        key: value for key, value in run.attrs.items() if key not in output.PROVENANCE
    }
    assert validate(snapshot, require_provenance=False) == []
    assert any("provenance" in p for p in validate(snapshot))


def test_frame_axes() -> None:
    assert parse_frame_axes("x=z y=-s z=r") == {
        "x": (1, "z"),
        "y": (-1, "theta"),
        "z": (1, "r"),
    }
    assert parse_frame_axes("x=r y=theta") == {"x": (1, "r"), "y": (1, "theta")}
    for text, message in (
        ("x=z y=s z=r", "left-handed"),
        ("x=q", "cannot read"),
        ("x=z x=r", "x is mapped twice"),
        ("x=z y=-z", "z is mapped twice"),
    ):
        with pytest.raises(FormatError, match=message):
            parse_frame_axes(text)


def _missing_git(*args: object, **kwargs: object) -> None:
    raise FileNotFoundError("git")


def test_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert git_hash(tmp_path) == "unknown"
    stamp = provenance("no-such-distribution", parameters="a: 1")
    assert stamp["code_version"] == "unknown"
    assert stamp["git_hash"] == "unknown"
    assert stamp["parameters"] == "a: 1"
    assert set(stamp) == set(output.PROVENANCE)
    own = provenance("takeout")
    assert own["parameters"] == ""
    assert own["git_hash"] == git_hash(Path(output.__file__))
    assert own["git_hash"] == "unknown" or len(own["git_hash"].split("-")[0]) == 40
    assert provenance("takeout", source=tmp_path)["git_hash"] == "unknown"
    assert provenance("")["git_hash"] == "unknown"
    monkeypatch.setattr(output.subprocess, "run", _missing_git)
    assert git_hash(Path(output.__file__)) == "unknown"


def test_moment_units() -> None:
    assert [moment_units(k) for k in range(5)] == [
        "1/m^3",
        "1/(m^2 s)",
        "1/(m s^2)",
        "1/s^3",
        "m^1/s^4",
    ]


def test_marker_samples_pad_and_drop() -> None:
    ds = marker_samples([7, 8], [[0.1, 0.2]], [[1.0, 2.0]], [3.0, 4.0], num_samples=3)
    assert ds["sample_id"].values.tolist() == [7, 8, -1]
    assert np.isnan(ds["sample_x"].values[2])
    assert ds["sample_vx"].attrs["units"] == "m/s"
    no_ids = marker_samples(None, [[0.1, 0.2]], [[1.0, 2.0]], [3.0, 4.0], 1)
    assert "sample_id" not in no_ids and no_ids.sizes["sample"] == 1
    assert no_ids["sample_weight"].values.tolist() == [3.0]
