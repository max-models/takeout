"""Diagnostics of the 2D E x B discharge benchmarks.

The axial-azimuthal benchmark of Charoy et al. (2019) and the radial-azimuthal one of
Villafana et al. (2021) are periodic along ``y`` (the azimuth); :func:`dominant_mode`
finds the strongest azimuthal wave in a time series of a field along ``y``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dominant_mode"]


def dominant_mode(
    field: np.ndarray, times: np.ndarray, length_y: float, min_periods: float = 2.0
) -> dict[str, float]:
    """Return the strongest azimuthal wave in ``field[time, y]``.

    A Hann-windowed 2D FFT over y and time; the strongest ``omega > 0`` bin with
    ``k != 0`` (a real signal repeats it at ``-omega, -k``) among the frequencies of at
    least ``min_periods`` periods in the time span (``min_frequency`` in the result:
    below it the window cannot tell a wave from a drift of the mean). The phase
    speed is positive towards +y (``exp(i(k y - omega t))`` has its peak at ``omega > 0``
    and ``k < 0`` in the FFT's convention).

    Parameters
    ----------
    field : numpy.ndarray
        The field, ``(time, y)``, on uniform grids, periodic along ``y``.
    times : numpy.ndarray
        The times of the rows, s.
    length_y : float
        The period along ``y``, m.
    min_periods : float, optional
        The fewest periods of a wave in the time span. Default: ``2.0``.

    Returns
    -------
    dict of str to float
        ``frequency`` (Hz), ``wavelength`` (m), ``phase_speed`` (m/s), ``span`` (the
        time span, s) and ``min_frequency`` (Hz). The wave's values are NaN when the
        window is too short for any frequency of ``min_periods`` periods.
    """
    nt, ny = field.shape
    dt = float(np.mean(np.diff(times)))
    span = float(times[-1] - times[0])
    window = np.hanning(nt)[:, None] * np.hanning(ny)[None, :]
    data = (field - field.mean()) * window
    spectrum = np.abs(np.fft.fft2(data)) ** 2
    omega = np.fft.fftfreq(nt, dt)  # Hz
    wave = np.fft.fftfreq(ny, length_y / ny)  # cycles per metre
    min_frequency = min_periods / span
    keep = (omega[:, None] >= min_frequency) & (wave[None, :] != 0)
    if not keep.any():
        # too few samples in the window for a frequency of min_periods periods
        return {
            "frequency": np.nan,
            "wavelength": np.nan,
            "phase_speed": np.nan,
            "span": span,
            "min_frequency": min_frequency,
        }
    peak = np.unravel_index(np.argmax(np.where(keep, spectrum, 0.0)), spectrum.shape)
    f, cycles = float(omega[peak[0]]), float(wave[peak[1]])
    return {
        "frequency": f,
        "wavelength": 1.0 / abs(cycles),
        "phase_speed": -f / cycles,
        "span": span,
        "min_frequency": min_frequency,
    }
