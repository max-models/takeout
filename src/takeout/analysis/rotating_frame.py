"""Positions in a frame that rotates about the ``z`` axis.

A rotating spoke (a rigidly rotating ionization zone) stands still in the frame that
turns with it, so averages over many records in that frame show its structure with far
less noise than any single record in the lab. :func:`to_rotating_frame` takes lab
positions, such as the ``sample_x`` and ``sample_y`` of :mod:`takeout.output`, or the
events of a marker tracker, into that frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["rotate", "to_rotating_frame"]


def rotate(
    x: np.ndarray, y: np.ndarray, angle: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the points ``(x, y)`` turned by ``angle`` about the origin.

    Parameters
    ----------
    x, y : numpy.ndarray
        The coordinates of the points.
    angle : float or numpy.ndarray
        The angle, rad, counter-clockwise; one per point or one for all.

    Returns
    -------
    tuple of numpy.ndarray
        The turned ``x`` and ``y``.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return cos * x - sin * y, sin * x + cos * y


def to_rotating_frame(
    positions: np.ndarray,
    time: float | np.ndarray,
    omega: float,
    *,
    center: Sequence[float] = (0.0, 0.0),
    angle0: float = 0.0,
) -> np.ndarray:
    """Return lab positions in the frame rotating at ``omega`` about ``center``.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(>=2, N)`` lab positions, m; the first two rows are ``x`` and ``y``.
    time : float or numpy.ndarray
        The time of the positions, s; one per point or one for all.
    omega : float
        The angular velocity of the frame, rad/s, positive counter-clockwise seen
        from +z.
    center : sequence of float, optional
        The axis of rotation in the plane, m. Default: the origin.
    angle0 : float, optional
        The angle of the frame at ``t = 0``, rad. Default: ``0.0``.

    Returns
    -------
    numpy.ndarray
        ``(2, N)`` positions in the rotating frame, m.
    """
    cx, cy = float(center[0]), float(center[1])
    x = np.asarray(positions[0], dtype=float) - cx
    y = np.asarray(positions[1], dtype=float) - cy
    xs, ys = rotate(x, y, -(angle0 + omega * np.asarray(time, dtype=float)))
    return np.stack([xs + cx, ys + cy])
