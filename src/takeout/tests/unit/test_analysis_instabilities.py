"""Tests of the detection, naming and checks of instabilities on synthetic records."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr
from scipy import constants

from takeout.analysis.instabilities import (
    COUPLED,
    GrowingMode,
    PlasmaScales,
    Population,
    _classify,
    _mark_couplings,
    _phase_frequency,
    _prediction,
    _rank,
    instability_report,
    populations,
)

# ---------------------------------------------------------------------------------------------
# The report on synthetic datasets
# ---------------------------------------------------------------------------------------------


def _dataset(
    beams,
    ions=None,
    growth=(),
    nodes=128,
    length=None,
    t_end=60.0,
    dt=0.05,
    magnetic_field=0.0,
    samples=True,
    heating=1.0,
    periodic=True,
) -> xr.Dataset:
    """An instability record: beams ``(density, drift, v_t)`` of electrons, ions
    ``(density, drift, v_t, mass)``, Fourier modes ``(m, gamma, omega)`` growing from the noise
    (rates in omega_pe), ``length`` in Debye lengths of the hottest beam."""
    rng = np.random.default_rng(0)
    n_e = sum(b[0] for b in beams)
    omega_pe = np.sqrt(n_e * constants.e**2 / (constants.epsilon_0 * constants.m_e))
    v_t = max(b[2] for b in beams)
    debye = v_t / omega_pe
    length = (nodes * 0.5 if length is None else length) * debye
    t = np.linspace(0.0, t_end / omega_pe, 600)
    x = np.arange(nodes) * length / nodes
    phi = 1e-6 * rng.standard_normal((t.size, nodes))
    for m, gamma, omega in growth:
        k = 2 * np.pi * m / length
        a = 1e-7 * np.exp(gamma * omega_pe * t)
        phi += (a / (1 + a))[:, None] * np.cos(
            k * x[None] - omega * omega_pe * t[:, None]
        )
    count = 20_000
    v_e = np.concatenate(
        [u + v * rng.standard_normal(int(count * n / n_e)) for n, u, v in beams]
    )
    names, mass, charge, density = ["electrons"], [constants.m_e], [-constants.e], [n_e]
    drift = [[v_e.mean(), 0.0, 0.0]]
    temperature = [[constants.m_e * v_e.var()] * 3]
    velocity = [np.pad(v_e, (0, count - v_e.size), constant_values=np.nan)]
    if ions is not None:
        n, u, v, m = ions
        names.append("ions")
        mass.append(m)
        charge.append(constants.e)
        density.append(n)
        drift.append([u, 0.0, 0.0])
        temperature.append([m * v**2] * 3)
        velocity.append(u + v * rng.standard_normal(count))
    temperature = np.tile(np.array(temperature)[None], (t.size, 1, 1))
    temperature[-1] *= heating
    kinetic = (
        0.5
        * np.array(density)
        * (temperature.sum(-1) + np.array(mass) * np.array(drift)[:, 0] ** 2)
        * length
    )
    variables = {
        "field_energy": ("t", np.full(t.size, 1e-20)),
        "kinetic_energy": (("t", "species"), kinetic),
        "density": (("t", "species"), np.tile(density, (t.size, 1))),
        "drift": (
            ("t", "species", "component"),
            np.tile(np.array(drift)[None], (t.size, 1, 1)),
        ),
        "temperature": (("t", "species", "component"), temperature),
    }
    if periodic:
        variables["potential_x"] = (("t", "x"), phi)
    if samples:
        stacked = np.tile(np.array(velocity)[None, :, None, :], (2, 1, 3, 1))
        variables["sample_velocity"] = (
            ("snapshot", "species", "component", "sample"),
            stacked,
        )
        variables["sample_weight"] = (
            ("snapshot", "species", "sample"),
            np.ones((2, len(names), count)),
        )
    return xr.Dataset(
        variables,
        coords={
            "t": t,
            "x": x,
            "species": names,
            "component": ["x", "y", "z"],
            "snapshot": [t[0], t[-1]],
        },
        attrs={
            "ndim": 1,
            "nvel": 3,
            "periodic": [int(periodic)],
            "lengths": [length],
            "spacing": [length / nodes],
            "dt": dt / omega_pe,
            "mass": mass,
            "charge": charge,
            "magnetic_field": magnetic_field,
            "magnetic_field_direction": [0.0, 0.0, 1.0]
            if magnetic_field
            else [0.0] * 3,
            "model": "synthetic",
        },
    )


PROTON = (1e15, 0.0, 1e3, constants.m_p)


def test_synthetic_buneman_is_found_and_predicted() -> None:
    drift, v_t = 5e6, 2e5
    ds = _dataset(
        [(1e15, drift, v_t)],
        ions=PROTON,
        growth=[(3, 0.05, 0.03)],
        t_end=600.0,
        length=3 * 2 * np.pi * drift / v_t,
    )
    report = instability_report(ds)
    assert [m.mode for m in report.modes] == [3]
    mode = report.modes[0]
    assert mode.name == "Buneman instability"
    assert mode.growth_rate / report.scales.electron_plasma_frequency == pytest.approx(
        0.05, rel=0.05
    )
    assert mode.frequency / report.scales.electron_plasma_frequency == pytest.approx(
        0.03, rel=0.05
    )
    assert mode.phase_velocity > 0
    assert mode.predicted.imag > 0
    assert "drift through the ions" in mode.reason


def test_nothing_grows_in_noise() -> None:
    report = instability_report(_dataset([(1e15, 0.0, 1e6)], ions=PROTON))
    assert not report.unstable
    assert str(report).count("Penrose") == 2


def test_heating_without_growth_is_flagged() -> None:
    report = instability_report(_dataset([(1e15, 0.0, 1e6)], ions=PROTON, heating=1.5))
    assert any("heated by 50 %" in w for w in report.warnings)


def test_the_numerical_warnings() -> None:
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON, nodes=32, length=128.0, dt=2.5)
    warnings = instability_report(ds).warnings
    assert any("lambda_De: an explicit PIC heats" in w for w in warnings)
    assert any("leapfrog is unstable" in w for w in warnings)
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON, dt=0.5)
    assert any("not accurate" in w for w in instability_report(ds).warnings)
    ds["kinetic_energy"][-1] *= 1.1
    assert any("total energy" in w for w in instability_report(ds).warnings)


def test_growth_unresolved_in_time() -> None:
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON, growth=[(2, 0.5, 0.0)], dt=0.3)
    assert any(
        "e-foldings per step" in w
        for w in instability_report(ds, predict=False).warnings
    )


def test_grid_scale_growth_is_numerical() -> None:
    ds = _dataset(
        [(1e15, 0.0, 1e6)],
        ions=PROTON,
        growth=[(50, 0.3, 0.0)],
        nodes=128,
        length=512.0,
    )
    report = instability_report(ds)
    assert report.modes[0].name == "numerical (finite-grid instability)"


def test_without_samples_every_species_is_one_population() -> None:
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON, samples=False)
    pops = populations(ds)
    assert [p.species for p in pops] == ["electrons", "ions"]
    assert pops[0].thermal_speed == pytest.approx(1e6, rel=0.05)
    report = instability_report(ds)
    assert report.penrose["x"] == {}


def test_without_a_periodic_axis_there_is_nothing_to_search() -> None:
    report = instability_report(_dataset([(1e15, 0.0, 1e6)], periodic=False))
    assert report.modes == []
    assert any("no periodic axis" in w for w in report.warnings)


def test_magnetized_across_the_axis_has_no_prediction() -> None:
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON, magnetic_field=0.5)
    report = instability_report(ds)
    assert report.predicted == {}
    assert any("magnetized" in w for w in report.warnings)


# ---------------------------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------------------------

OMEGA_PE = 1e9
ATTRS = {"spacing": [1e-4], "magnetic_field_direction": [0.0, 0.0, 0.0]}


def _scales(**changes) -> PlasmaScales:
    values = {
        "electron_plasma_frequency": OMEGA_PE,
        "ion_plasma_frequency": OMEGA_PE / 43,
        "electron_cyclotron_frequency": 0.0,
        "ion_cyclotron_frequency": 0.0,
        "lower_hybrid_frequency": 0.0,
        "debye_length": 1e-3,
        "electron_thermal_speed": 1e6,
        "sound_speed": 2e4,
        "electron_ion_drift": np.zeros(3),
        "temperature_ratio": 20.0,
    }
    values.update(changes)
    return PlasmaScales(**values)


def _electrons(density, drift, v_t) -> Population:
    return Population("electrons", density, -constants.e, constants.m_e, drift, v_t)


def _ions(density, drift, v_t) -> Population:
    return Population("ions", density, constants.e, constants.m_p, drift, v_t)


@pytest.mark.parametrize(
    ("k", "omega", "scales", "pops", "attrs", "name"),
    [
        (
            100.0,
            0.0,
            _scales(),
            [_electrons(5e14, 3e6, 1e5), _electrons(5e14, -3e6, 1e5)],
            ATTRS,
            "two-stream instability",
        ),
        (
            100.0,
            1e9,
            _scales(),
            [_electrons(1e15, 0.0, 1e6), _electrons(1e13, 8e6, 1e5)],
            ATTRS,
            "beam–plasma instability",
        ),
        (
            100.0,
            1e9,
            _scales(),
            [_electrons(1e15, 0.0, 1e6), _electrons(1e13, 8e6, 2e6)],
            ATTRS,
            "bump-on-tail instability",
        ),
        (
            100.0,
            3e7,
            _scales(electron_ion_drift=np.array([3e6, 0, 0])),
            [_electrons(1e15, 3e6, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "Buneman instability",
        ),
        (
            100.0,
            2e6,
            _scales(electron_ion_drift=np.array([5e5, 0, 0])),
            [_electrons(1e15, 5e5, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "ion-acoustic instability",
        ),
        (
            100.0,
            1e6,
            _scales(),
            [_electrons(1e15, 0, 1e6), _ions(5e14, 2e5, 1e3), _ions(5e14, -2e5, 1e3)],
            ATTRS,
            "ion–ion streaming instability",
        ),
        (
            100.0,
            1e6,
            _scales(),
            [_electrons(1e15, 0, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "unidentified electrostatic mode",
        ),
        (
            2e4,
            0.0,
            _scales(debye_length=1e-5),
            [_electrons(1e15, 0, 1e6)],
            ATTRS,
            "numerical (finite-grid instability)",
        ),
        (
            1000.0,
            1e7,
            _scales(
                electron_cyclotron_frequency=3e8,
                lower_hybrid_frequency=1.5e7,
                electron_ion_drift=np.array([5e5, 0, 0]),
            ),
            [_electrons(1e15, 5e5, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "electron cyclotron drift instability",
        ),
        (
            100.0,
            1e7,
            _scales(
                electron_cyclotron_frequency=3e8,
                lower_hybrid_frequency=1.5e7,
                electron_ion_drift=np.array([5e5, 0, 0]),
            ),
            [_electrons(1e15, 5e5, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "lower-hybrid / modified two-stream instability",
        ),
        (
            100.0,
            1e5,
            _scales(
                electron_cyclotron_frequency=3e8,
                lower_hybrid_frequency=1.5e7,
                electron_ion_drift=np.array([5e5, 0, 0]),
            ),
            [_electrons(1e15, 5e5, 1e6), _ions(1e15, 0, 1e3)],
            ATTRS,
            "low-frequency drift mode (gradient-driven, e.g. Simon–Hoh or a spoke)",
        ),
    ],
    ids=[
        "two-stream",
        "beam-plasma",
        "bump-on-tail",
        "buneman",
        "ion-acoustic",
        "ion-ion",
        "unidentified",
        "finite-grid",
        "ecdi",
        "lower-hybrid",
        "low-frequency",
    ],
)
def test_names(k, omega, scales, pops, attrs, name) -> None:
    found, reason = _classify(k, omega, 0, scales, pops, attrs)
    assert found == name
    assert reason


def _mode(mode: int, rate: float, predicted: complex = complex(np.nan, np.nan)):
    return GrowingMode(
        axis="x",
        mode=mode,
        k=float(mode),
        growth_rate=rate,
        frequency=0.0,
        phase_velocity=0.0,
        efolds=5.0,
        start=0.0,
        stop=1.0,
        r2=0.99,
        name="two-stream instability",
        reason="",
        predicted=predicted,
    )


def test_modes_driven_by_two_growing_modes_are_marked() -> None:
    modes = [
        _mode(2, 1.0),
        _mode(3, 0.9),
        _mode(4, 2.05),
        _mode(5, 1.85),
        _mode(1, 1.95),
    ]
    marked = {m.mode: m for m in _mark_couplings(modes)}
    assert marked[2].name == marked[3].name == "two-stream instability"
    assert marked[4].name == COUPLED and "4 = 2 + 2" in marked[4].reason  # a harmonic
    assert marked[5].name == COUPLED and "5 = 3 + 2" in marked[5].reason
    assert marked[1].name == COUPLED and "1 = 3 - 2" in marked[1].reason
    # a mode growing at neither sum stays linear
    assert _mark_couplings([_mode(2, 1.0), _mode(4, 1.2)])[1].name != COUPLED


def test_modes_the_theory_calls_stable_rank_after_the_confirmed_ones() -> None:
    confirmed = _mode(1, 1.0, predicted=complex(2.0, 1.0))
    stable = _mode(2, 1.5, predicted=complex(3.0, 1e-16))
    unknown = _mode(3, 0.8)
    coupled = replace(_mode(4, 3.0), name=COUPLED)
    ranked = sorted([coupled, stable, unknown, confirmed], key=_rank)
    assert [m.mode for m in ranked] == [1, 3, 2, 4]


def test_the_report_as_a_table() -> None:
    drift, v_t = 5e6, 2e5
    ds = _dataset(
        [(1e15, drift, v_t)],
        ions=PROTON,
        growth=[(3, 0.05, 0.03)],
        t_end=600.0,
        length=3 * 2 * np.pi * drift / v_t,
    )
    report = instability_report(ds)
    text = str(report)
    assert report.names == ["Buneman instability"]
    assert "gamma/wpe" in text and "Buneman instability" in text
    assert "Predicted fastest along x" in text
    assert report.populations["x"][0].plasma_frequency == pytest.approx(
        report.scales.electron_plasma_frequency, rel=1e-6
    )


def test_two_electron_species_are_checked_together() -> None:
    ds = _dataset([(1e15, 0.0, 1e6)], ions=PROTON)
    second = ds.isel(species=[0]).assign_coords(species=["beam"])
    ds = xr.concat([ds, second], dim="species", data_vars="minimal")
    ds = ds.assign_coords(species=["bulk", "ions", "beam"])
    ds.attrs["mass"] = [*ds.attrs["mass"], constants.m_e]
    ds.attrs["charge"] = [*ds.attrs["charge"], -constants.e]
    report = instability_report(ds)
    assert set(report.penrose["x"]) == {"bulk", "ions", "beam", "electrons"}


def test_without_electrons_nothing_is_predicted() -> None:
    pops = [Population("ions", 1e15, constants.e, constants.m_p, 0.0, 1e3)]
    k = np.array([1.0, 2.0])
    report_scales = PlasmaScales(
        electron_plasma_frequency=0.0,
        ion_plasma_frequency=1e7,
        electron_cyclotron_frequency=0.0,
        ion_cyclotron_frequency=0.0,
        lower_hybrid_frequency=0.0,
        debye_length=np.inf,
        electron_thermal_speed=0.0,
        sound_speed=0.0,
        electron_ion_drift=np.zeros(3),
        temperature_ratio=0.0,
    )
    assert np.isnan(_prediction(pops, report_scales, k)).all()
    assert _phase_frequency(np.ones(5), np.arange(5.0), 0.0, 1.0) == 0.0
