r"""The ion-acoustic wave: its kinetic dispersion relation and the mode of a run.

The frequency and the damping (or growth) of the ion mode of Maxwellian electrons and
ions are the root of

.. math::

    1 + \frac{1 + \zeta_e Z(\zeta_e)}{k^2 \lambda_{De}^2}
      + \frac{T_e}{T_i}\,\frac{1 + \zeta_i Z(\zeta_i)}{k^2 \lambda_{De}^2} = 0,
    \qquad \zeta_s = \frac{\omega - k u_s}{\sqrt 2\, k v_{ts}},

with :math:`Z` the plasma dispersion function and :math:`u_e` an optional electron drift
(:func:`dielectric`, :func:`ion_acoustic_root`); :func:`measure` reads the frequency and
the damping rate off the recorded Fourier coefficient of a run's ion mode. Frequencies
are in ``omega_pe``, wavenumbers in ``1 / lambda_De`` and speeds in electron thermal
speeds.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import root
from scipy.special import wofz

__all__ = ["dielectric", "ion_acoustic_root", "measure", "plasma_dispersion"]


def plasma_dispersion(z: complex | np.ndarray) -> complex | np.ndarray:
    """Return the plasma dispersion function ``Z(z) = i sqrt(pi) w(z)``.

    Parameters
    ----------
    z : complex or ndarray
        The argument, ``(omega - k u) / (sqrt(2) k v_t)`` of a Maxwellian.

    Returns
    -------
    complex or ndarray
        ``Z(z)``, from the Faddeeva function :func:`scipy.special.wofz`.
    """
    return 1j * np.sqrt(np.pi) * wofz(z)


def dielectric(
    omega: complex,
    k: float,
    temperature_ratio: float,
    mass_ratio: float,
    drift: float = 0.0,
) -> complex:
    """Return the kinetic dielectric function of Maxwellian electrons and ions.

    Parameters
    ----------
    omega : complex
        The frequency in ``omega_pe``.
    k : float
        The wavenumber in ``1 / lambda_De``.
    temperature_ratio : float
        ``T_e / T_i``.
    mass_ratio : float
        ``m_i / m_e``.
    drift : float, optional
        The electron drift in electron thermal speeds. Default: ``0.0``.

    Returns
    -------
    complex
        ``1 + chi_e + chi_i``, zero on the wave's dispersion relation.
    """
    zeta_e = (omega - k * drift) / (np.sqrt(2.0) * k)
    ion_thermal = np.sqrt(1.0 / (temperature_ratio * mass_ratio))
    zeta_i = omega / (np.sqrt(2.0) * k * ion_thermal)
    electrons = (1.0 + zeta_e * plasma_dispersion(zeta_e)) / k**2
    ions = temperature_ratio * (1.0 + zeta_i * plasma_dispersion(zeta_i)) / k**2
    return 1.0 + electrons + ions


def ion_acoustic_root(
    k: float, temperature_ratio: float, mass_ratio: float, drift: float = 0.0
) -> complex:
    """Return the ion-acoustic root ``omega`` [omega_pe] of :func:`dielectric` at ``k``.

    Parameters
    ----------
    k : float
        The wavenumber in ``1 / lambda_De``.
    temperature_ratio : float
        ``T_e / T_i``.
    mass_ratio : float
        ``m_i / m_e``.
    drift : float, optional
        The electron drift in electron thermal speeds. Default: ``0.0``.

    Returns
    -------
    complex
        The root found from the fluid guess ``k c_s / sqrt(1 + k^2 lambda_De^2)``
        (``c_s`` the cold ion sound speed); its imaginary part is the damping
        rate (negative) or the growth rate.
    """
    sound = np.sqrt(1.0 / mass_ratio)
    guess = k * sound / np.sqrt(1.0 + k**2)

    def residual(x: np.ndarray) -> list[float]:
        value = dielectric(x[0] + 1j * x[1], k, temperature_ratio, mass_ratio, drift)
        return [value.real, value.imag]

    solution = root(residual, [guess, 0.0])
    return complex(solution.x[0], solution.x[1])


def measure(result: xr.Dataset, start: float = 0.25) -> tuple[float, float]:
    """Return the frequency and the damping rate of the ion mode of a run.

    Parameters
    ----------
    result : xarray.Dataset
        The run: ``mode_real`` and ``mode_imag``, the Fourier coefficient of the ion
        mode against ``t`` (in any time unit; the rates come out in its inverse).
    start : float, optional
        The wave periods skipped before the fit (all the samples are used when
        fewer than three remain). Default: ``0.25``.

    Returns
    -------
    tuple of float
        The frequency, the slope of the unwrapped phase, and the damping rate,
        the slope of the log of the magnitude (negative for a damped wave).
    """
    mode = result["mode_real"].values + 1j * result["mode_imag"].values
    times = result["t"].values
    period = 2.0 * np.pi / abs(np.polyfit(times, np.unwrap(np.angle(mode)), 1)[0])
    window = times >= start * period
    if np.count_nonzero(window) < 3:
        window = np.ones_like(times, dtype=bool)
    frequency = -np.polyfit(times[window], np.unwrap(np.angle(mode[window])), 1)[0]
    damping = np.polyfit(times[window], np.log(np.abs(mode[window])), 1)[0]
    return float(frequency), float(damping)
