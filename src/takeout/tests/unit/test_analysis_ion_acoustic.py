"""Tests of the ion-acoustic wave's dispersion relation and its measurement."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from takeout.analysis.ion_acoustic import (
    dielectric,
    ion_acoustic_root,
    measure,
    plasma_dispersion,
)


def test_the_plasma_dispersion_function_at_zero() -> None:
    assert plasma_dispersion(0.0) == pytest.approx(1j * np.sqrt(np.pi))


def test_the_root_tends_to_the_cold_sound_wave_for_cold_ions() -> None:
    # Ti << Te: omega -> k c_s / sqrt(1 + k^2 lambda_De^2), lightly damped
    k = 0.05
    omega = ion_acoustic_root(k, temperature_ratio=1000.0, mass_ratio=1836.0)
    assert omega.real == pytest.approx(
        k / np.sqrt(1836.0) / np.sqrt(1 + k**2), rel=2e-3
    )
    assert -0.02 < omega.imag / omega.real < 0.0
    assert abs(dielectric(omega, k, 1000.0, 1836.0)) < 1e-8


def test_a_drifting_electron_population_drives_the_wave() -> None:
    assert ion_acoustic_root(0.65, 20.0, 100.0, drift=0.8).imag > 0.0


def _run(omega: complex, periods: float = 2.0) -> xr.Dataset:
    times = np.linspace(0.0, periods * 2.0 * np.pi / omega.real, 200)
    mode = 0.1 * np.exp(-1j * omega.real * times + omega.imag * times)
    return xr.Dataset(
        {"mode_real": ("t", mode.real), "mode_imag": ("t", mode.imag)},
        coords={"t": ("t", times)},
    )


def test_measure_a_damped_mode() -> None:
    omega = ion_acoustic_root(2 * np.pi / 400, 20.0, 100.0)
    frequency, damping = measure(_run(omega))
    assert frequency == pytest.approx(omega.real, rel=1e-6)
    assert damping == pytest.approx(omega.imag, rel=1e-6)


def test_measure_uses_every_sample_when_the_window_is_short() -> None:
    omega = complex(1.0, -0.01)
    frequency, damping = measure(_run(omega, periods=0.1), start=0.25)
    assert frequency == pytest.approx(1.0, rel=1e-6)
    assert damping == pytest.approx(-0.01, rel=1e-6)
