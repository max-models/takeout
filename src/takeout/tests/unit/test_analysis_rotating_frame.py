"""Tests of the positions in a rotating frame."""

from __future__ import annotations

import numpy as np
import pytest

from takeout.analysis.rotating_frame import rotate, to_rotating_frame


def test_rotate_a_quarter_turn() -> None:
    x, y = rotate(np.array([1.0]), np.array([0.0]), np.pi / 2)
    assert x == pytest.approx([0.0], abs=1e-15)
    assert y == pytest.approx([1.0])


def test_a_point_turning_with_the_frame_stands_still_in_it() -> None:
    omega, radius = 2.0e5, 0.02
    times = np.linspace(0.0, 1e-4, 7)
    x, y = rotate(np.full(times.size, radius), np.zeros(times.size), omega * times)
    frame = to_rotating_frame(np.stack([x, y, np.ones_like(x)]), times, omega)
    assert frame.shape == (2, times.size)
    assert frame[0] == pytest.approx(np.full(times.size, radius))
    assert frame[1] == pytest.approx(np.zeros(times.size), abs=1e-15)


def test_the_center_and_the_initial_angle() -> None:
    frame = to_rotating_frame(
        np.array([[1.0], [2.0]]), 0.0, 1.0, center=(1.0, 1.0), angle0=np.pi / 2
    )
    # (0, 1) from the center, turned back by a quarter turn: (1, 0)
    assert frame[:, 0] == pytest.approx([2.0, 1.0])
