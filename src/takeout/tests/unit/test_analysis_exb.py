"""Tests of the diagnostics of the 2D E x B benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from takeout.analysis.exb import dominant_mode


def test_dominant_mode_finds_a_wave_towards_plus_y() -> None:
    ny, nt = 16, 300
    length_y, dt = 0.0128, 2e-9
    y = np.arange(ny) * length_y / ny
    t = np.arange(nt) * dt
    wavelength, frequency = length_y / 2, 10e6
    field = np.cos(2 * np.pi * (y[None, :] / wavelength - frequency * t[:, None]))
    mode = dominant_mode(field, t, length_y)
    assert mode["wavelength"] == pytest.approx(wavelength)
    assert mode["frequency"] == pytest.approx(frequency, rel=0.02)
    assert mode["phase_speed"] == pytest.approx(wavelength * frequency, rel=0.02)


def test_dominant_mode_of_a_wave_towards_minus_y_has_a_negative_speed() -> None:
    ny, nt = 16, 300
    length_y, dt = 0.0128, 2e-9
    y = np.arange(ny) * length_y / ny
    t = np.arange(nt) * dt
    field = np.cos(2 * np.pi * (y[None, :] / length_y + 10e6 * t[:, None]))
    assert dominant_mode(field, t, length_y)["phase_speed"] < 0


def test_dominant_mode_of_a_window_too_short_is_nan() -> None:
    t = np.arange(3) * 1e-9
    mode = dominant_mode(np.ones((3, 8)), t, 0.01, min_periods=10.0)
    assert np.isnan(mode["frequency"]) and np.isnan(mode["wavelength"])
    assert mode["span"] == pytest.approx(2e-9)
