"""Field maps: the layout, netCDF files and the interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from methodcard.testing import method_suite

from takeout import FIELD_MAP_FORMAT, FieldMap, FormatError, validate_field_map

TestMethodCards = method_suite("takeout")


def _linear(positions: np.ndarray) -> np.ndarray:
    """A field linear in x and y, which multilinear interpolation reproduces."""
    x, y = positions[:, 0], positions[:, 1]
    return np.stack([1.0 + 2.0 * x, 0.5 * y - x, 0.0 * x + 3.0], axis=1)


def _map_2d(kind: str = "magnetic") -> FieldMap:
    return FieldMap.from_function(
        _linear,
        {"x": np.linspace(0.0, 1.0, 11), "y": np.linspace(-1.0, 1.0, 5)},
        kind=kind,
        frame_axes="x=z y=-s",
        source="test: linear field",
    )


def _xarray_map() -> xr.Dataset:
    """A map built with xarray alone, as a code without takeout writes one."""
    x = np.linspace(0.0, 0.1, 11)
    y = np.linspace(-0.02, 0.02, 5)
    bz = 0.01 + 0.1 * x[:, None] + 0.5 * y[None, :]
    return xr.Dataset(
        {
            "b_field_z": (("x", "y"), bz, {"units": "T"}),
            "e_field_x": (("x", "y"), np.full((11, 5), 100.0), {"units": "V/m"}),
        },
        coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"})},
        attrs={
            "Conventions": "max-models-field-map-1.0",
            "frame": "cartesian",
            "source": "xarray",
            "frame_axes": "x=z y=-s z=r",
        },
    )


def test_a_map_round_trips_through_netcdf(tmp_path: Path) -> None:
    field_map = _map_2d()
    assert validate_field_map(field_map.dataset) == []
    path = field_map.write(tmp_path / "linear.nc")
    assert field_map.path == path
    back = FieldMap.read(path, outside="zero")
    xr.testing.assert_identical(back.dataset, field_map.dataset)
    assert back.path == path and back.outside == "zero"
    assert back.frame == "cartesian"
    assert back.dataset.attrs["Conventions"] == FIELD_MAP_FORMAT
    assert back.dataset.attrs["source"] == "test: linear field"
    assert back.dims == ("x", "y")
    assert back.components("magnetic") == ("b_field_x", "b_field_y", "b_field_z")
    assert back.components("electric") == ()


def test_a_map_written_with_xarray_alone_is_read(tmp_path: Path) -> None:
    path = tmp_path / "plain.nc"
    _xarray_map().to_netcdf(path)
    field_map = FieldMap.read(path)
    points = np.array([[0.033, 0.007], [0.091, -0.013]])
    b = field_map.magnetic_field(points)
    expected = 0.01 + 0.1 * points[:, 0] + 0.5 * points[:, 1]
    np.testing.assert_allclose(b[:, 2], expected, rtol=1e-12)
    np.testing.assert_array_equal(b[:, :2], 0.0)
    np.testing.assert_allclose(field_map.electric_field(points), [[100, 0], [100, 0]])
    written = xr.open_dataset(field_map.write(tmp_path / "again.nc")).load()
    assert validate_field_map(written) == []
    assert written.attrs == _xarray_map().attrs


def test_linear_fields_are_interpolated_exactly() -> None:
    field_map = _map_2d()
    rng = np.random.default_rng(1)
    positions = np.stack([rng.uniform(0, 1, 50), rng.uniform(-1, 1, 50)], axis=1)
    np.testing.assert_allclose(field_map.magnetic_field(positions), _linear(positions))
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(-1, 1, 3), indexing="ij")
    mesh = field_map.evaluate((x, y))
    assert mesh.shape == (4, 3, 3)
    np.testing.assert_allclose(mesh[..., 1], 0.5 * y - x)
    electric = _map_2d("electric")
    np.testing.assert_allclose(
        electric.electric_field(positions), _linear(positions)[:, :2]
    )
    assert electric.evaluate((x, y), kind="electric").shape == (4, 3, 2)
    with pytest.raises(ValueError, match="no b_field"):
        electric.magnetic_field(positions)


def test_outside_the_map() -> None:
    field_map = _map_2d()
    outside = np.array([[2.0, 0.0], [0.5, 0.0]])
    clipped = field_map.magnetic_field(outside)
    np.testing.assert_allclose(clipped[0], _linear(np.array([[1.0, 0.0]]))[0])
    zero = FieldMap(field_map.dataset, outside="zero").magnetic_field(outside)
    assert (zero[0] == 0.0).all() and (zero[1] == clipped[1]).all()
    with pytest.raises(ValueError, match="1 positions are outside"):
        FieldMap(field_map.dataset, outside="error").magnetic_field(outside)
    with pytest.raises(ValueError, match="outside must be"):
        FieldMap(field_map.dataset, outside="nan")


def test_a_map_with_fewer_dims_is_uniform_along_the_others() -> None:
    profile = FieldMap.from_arrays(
        {"x": [0.0, 1.0]}, {"b_field_z": [0.0, 2.0]}, source="ramp"
    )
    positions = np.array([[0.25, 5.0, 1.0], [0.75, -3.0, 2.0]])
    np.testing.assert_allclose(profile.magnetic_field(positions)[:, 2], [0.5, 1.5])
    on_y = FieldMap.from_arrays({"y": [0.0, 1.0]}, {"e_field_x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="at least 2 coordinates"):
        on_y.electric_field(np.array([[0.5]]))
    x = np.linspace(0.0, 1.0, 11)
    one_d = FieldMap.from_arrays({"x": x}, {"b_field_y": 2.0 * x})
    assert one_d.evaluate((x[:4],)).shape == (4, 3)


def test_the_validator_rejects_malformed_maps() -> None:
    good = _map_2d().dataset
    bad = good.rename({"b_field_x": "bx"})
    bad["b_field_y"] = bad["b_field_y"].assign_attrs(units="G")
    bad["b_field_z"] = bad["b_field_z"].transpose("y", "x")
    bad.attrs = {"Conventions": "other", "frame": "polar", "frame_axes": "x=z y=s z=r"}
    text = "\n".join(validate_field_map(bad))
    for expected in (
        "Conventions",
        "frame is 'polar'",
        "left-handed",
        "bx: not a field-map variable",
        "b_field_y: units 'G'",
        "b_field_z: dims",
    ):
        assert expected in text, expected
    with pytest.raises(FormatError, match="not a field map"):
        FieldMap(bad)
    timed = good.expand_dims(t=[0.0])
    timed["t"].attrs["units"] = "s"
    assert any("(x, y, z) only" in p for p in validate_field_map(timed))
    assert any(
        "no b_field" in p
        for p in validate_field_map(good.drop_vars(list(good.data_vars)))
    )
    decreasing = good.assign_coords(x=("x", good.x.values[::-1], {"units": "m"}))
    assert any("not increasing" in p for p in validate_field_map(decreasing))
    millimetres = good.assign_coords(x=("x", good.x.values, {"units": "mm"}))
    assert any("has units 'mm'" in p for p in validate_field_map(millimetres))
    assert any("no coordinate" in p for p in validate_field_map(good.drop_vars("y")))
    cylindrical = good.assign_attrs(frame="cylindrical")
    assert "frame_axes is for a cartesian frame" in validate_field_map(cylindrical)


def test_point_and_plane_functions() -> None:
    x = np.linspace(0.0, 1.0, 11)
    field_map = FieldMap.from_arrays({"x": x}, {"b_field_y": 2.0 * x})
    points = np.array([[0.25, 0.55], [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        field_map.points_function("magnetic")(points)[1], [0.5, 1.1]
    )
    bx, by, bz = field_map.plane_function("magnetic")(
        np.array([[0.25, 0.55]]), np.zeros((1, 2))
    )
    np.testing.assert_allclose(by, [[0.5, 1.1]])
    assert bx.shape == bz.shape == (1, 2)
    with pytest.raises(ValueError, match="no e_field"):
        field_map.points_function("electric")
    volume = FieldMap.from_arrays(
        {"x": x, "y": x, "z": x}, {"e_field_x": np.ones((11, 11, 11))}
    )
    with pytest.raises(ValueError, match="at least 3 coordinates"):
        volume.electric_field(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="2D plane"):
        volume.plane_function("electric")
