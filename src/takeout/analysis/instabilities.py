"""Detection of instabilities in a run: what grows, how fast, and what it is.

No single diagnostic names every instability, but whether something grows can be told
without knowing what it is: a Fourier mode of the potential whose amplitude grows
exponentially, out of the marker noise, for several e-foldings. :func:`instability_report`
reads an *instability record* (picnic's ``InstabilityMonitor`` writes one) in three
layers:

1. **Detection.** Every Fourier mode of the potential is searched for an exponential growth
   phase from the noise floor to saturation (:func:`plasma_plots.analysis.detect_growth`, at
   least ``min_efolds`` e-foldings, ``R² >= min_r2``). The growth rate γ comes from the fit,
   and the real frequency ω_r from the phase of the mode over the same window.
2. **Classification.** The plasma's scales are measured from the first record (ω_pe, ω_pi,
   ω_ce, ω_LH, λ_De, c_s, the drifts and thermal speeds). The velocity sample of each species
   is split at the dips of its distribution into drifting Maxwellian populations (two beams in
   one species become two). Then three things are reported:

   - the name of each growing mode from its k, ω_r and the populations: two-stream,
     beam–plasma, bump-on-tail, Buneman, ion acoustic, ion–ion streaming, electron cyclotron
     drift, lower hybrid;
   - the Penrose criterion of the measured distributions
     (:func:`plasma_plots.analysis.penrose_criterion`);
   - the growth that the electrostatic dispersion relation of those populations predicts at
     every measured k (:func:`plasma_plots.theory.kinetic.electrostatic_growth`), when the
     plasma is unmagnetized along the axis.
3. **Numerical instabilities** are flagged, not reported as physics:

   - a cell larger than the Debye length (the finite-grid instability heats the plasma);
   - ω_pe Δt past the accuracy (0.3) or the stability (2) limit of the leapfrog;
   - growth faster than the step resolves;
   - drift of the total energy in a periodic box;
   - heating with nothing growing.

**The instability record** (SI units, ``t`` in s; the dims ``t``, ``species`` and the
axis names follow :mod:`takeout.output`)

=========================  ======================================  =======================
name                       dims                                    meaning
=========================  ======================================  =======================
``potential_<axis>``       (t, <axis>)                             the potential along
                                                                   each periodic axis,
                                                                   averaged over the
                                                                   others, V
``field_energy``           (t)                                     J
``kinetic_energy``         (t, species)                            J
``density``                (t, species)                            mean density, m^-3
``drift``                  (t, species, component)                 mean velocity, m/s
``temperature``            (t, species, component)                 m <(v - u)²>, J
``sample_velocity``        (snapshot, species, component, sample)  optional: velocities
                                                                   of a marker sample at
                                                                   the first and the
                                                                   latest record, m/s
``sample_weight``          (snapshot, species, sample)             their weights
=========================  ======================================  =======================

with the attributes ``ndim``, ``nvel``, ``periodic`` (one 0/1 flag per axis),
``lengths`` and ``spacing`` (m, per axis), ``dt`` (s), ``mass`` and ``charge`` (per
species), ``magnetic_field`` (the mean ``|B|``, T) and ``magnetic_field_direction``
(the unit vector of the mean field).

Limits: modes are looked for along the periodic axes only, with the potential averaged over
the others (a bounded direction has no clean Fourier modes). The prediction assumes a
homogeneous, unmagnetized plasma (or k along B), so it misses gradient-driven modes. Several
instabilities at once, or a run that starts saturated, leave no clean linear phase. Growth
below the marker noise, or slower than the run, is not seen. Electromagnetic instabilities
(Weibel, firehose) are out of reach of the electrostatic analysis. The names come from
dimensionless numbers and are a best guess: each mode carries the reason for its name.

This module needs plasma-plots: install ``takeout[analysis]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import xarray as xr
from plasma_plots.analysis import detect_growth, dominant_mode, penrose_criterion
from plasma_plots.theory.kinetic import Maxwellian, electrostatic_growth
from scipy import constants

from takeout._common import AXES

__all__ = [
    "COUPLED",
    "GrowingMode",
    "InstabilityReport",
    "PlasmaScales",
    "Population",
    "instability_report",
    "populations",
]

#: A species lighter than this many electron masses counts as electrons.
_ELECTRON_MASS_LIMIT = 100.0
#: Bins of the velocity histograms, and the [1, 2, 1] smoothing passes applied to them.
_BINS = 96
_SMOOTHING = 2


@dataclass(frozen=True)
class Population:
    """A drifting Maxwellian population, one species or one beam of it.

    Attributes
    ----------
    species : str
        The species it belongs to.
    density : float
        The mean density, m^-3.
    charge : float
        The charge, C.
    mass : float
        The mass, kg.
    drift : float
        The mean velocity along the axis, m/s.
    thermal_speed : float
        sqrt(T/m) along the axis, m/s.
    """

    species: str
    density: float
    charge: float
    mass: float
    drift: float
    thermal_speed: float

    @property
    def is_electron(self) -> bool:
        """Lighter than 100 electron masses."""
        return self.mass < _ELECTRON_MASS_LIMIT * constants.m_e

    @property
    def plasma_frequency(self) -> float:
        """sqrt(n q² / (ε0 m)), rad/s."""
        return float(
            np.sqrt(self.density * self.charge**2 / (constants.epsilon_0 * self.mass))
        )


def _histogram(
    velocities: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed, weighted histogram of the finite velocities: centres, values."""
    keep = np.isfinite(velocities) & np.isfinite(weights)
    velocities, weights = velocities[keep], weights[keep]
    low, high = float(velocities.min()), float(velocities.max())
    pad = 0.05 * (high - low) if high > low else max(abs(low), 1.0) * 1e-3
    counts, edges = np.histogram(
        velocities, bins=_BINS, range=(low - pad, high + pad), weights=weights
    )
    for _ in range(_SMOOTHING):
        counts = np.convolve(np.pad(counts, 1), [0.25, 0.5, 0.25], mode="valid")
    return 0.5 * (edges[1:] + edges[:-1]), counts


def _distribution(velocities: np.ndarray, weights: np.ndarray) -> xr.DataArray:
    """Return the histogram of :func:`_histogram` as f(v)."""
    centres, counts = _histogram(velocities, weights)
    return xr.DataArray(counts, coords={"v": centres}, dims="v")


def populations(ds: xr.Dataset, axis: str = "x", snapshot: int = 0) -> list[Population]:
    """Split every species into drifting Maxwellians along ``axis``, from its velocity sample.

    A species whose velocity distribution has dips (deeper than 10 % of its peak) is cut at
    them into populations, each with its own density, drift and thermal speed: two beams loaded
    as one species become two populations. Without a sample, every species is one population
    with the recorded moments.

    Parameters
    ----------
    ds : xarray.Dataset
        The instability record (see the module docstring).
    axis : str, optional
        The velocity component. Default: ``"x"``.
    snapshot : int, optional
        0 for the first record, 1 for the latest. Default: ``0``.

    Returns
    -------
    list of Population
        The populations of every species.
    """
    out = []
    t = 0 if snapshot == 0 else -1
    for index, name in enumerate(ds.species.values):
        mass, charge = ds.attrs["mass"][index], ds.attrs["charge"][index]
        density = float(ds["density"].isel(t=t, species=index))
        if "sample_velocity" not in ds:
            drift = float(ds["drift"].isel(t=t, species=index).sel(component=axis))
            temperature = float(
                ds["temperature"].isel(t=t, species=index).sel(component=axis)
            )
            out.append(
                Population(
                    str(name),
                    density,
                    charge,
                    mass,
                    drift,
                    np.sqrt(max(temperature, 0.0) / mass),
                )
            )
            continue
        v = (
            ds["sample_velocity"]
            .isel(snapshot=snapshot, species=index)
            .sel(component=axis)
            .values
        )
        w = ds["sample_weight"].isel(snapshot=snapshot, species=index).values
        keep = np.isfinite(v) & np.isfinite(w)
        v, w = v[keep], w[keep]
        if not v.size:
            continue
        cuts = np.sort(penrose_criterion(_distribution(v, w), prominence=0.1).minima)
        groups = np.searchsorted(cuts, v)
        total = w.sum()
        for group in range(cuts.size + 1):
            member = groups == group
            if not member.any() or w[member].sum() <= 0.01 * total:
                continue
            ww = w[member] / w[member].sum()
            u = float(np.sum(ww * v[member]))
            spread = float(np.sqrt(np.sum(ww * (v[member] - u) ** 2)))
            out.append(
                Population(
                    str(name),
                    density * w[member].sum() / total,
                    charge,
                    mass,
                    u,
                    spread,
                )
            )
    return out


@dataclass(frozen=True)
class PlasmaScales:
    """The frequencies, lengths and speeds of the plasma, from the first record.

    Attributes
    ----------
    electron_plasma_frequency : float
        ω_pe of all the electrons, rad/s.
    ion_plasma_frequency : float
        ω_pi of all the ions, rad/s (0 without ions).
    electron_cyclotron_frequency : float
        ω_ce, rad/s (0 without a magnetic field).
    ion_cyclotron_frequency : float
        ω_ci of the lightest ion, rad/s.
    lower_hybrid_frequency : float
        ω_LH, 1/ω_LH² = 1/(ω_ce ω_ci) + 1/ω_pi² (0 without a field or ions), rad/s.
    debye_length : float
        λ_De = v_te / ω_pe of the electrons, m.
    electron_thermal_speed : float
        sqrt(T_e/m_e) of the electrons, m/s: the thermal spread within their populations along
        the first periodic axis (the spread between beams is not thermal).
    sound_speed : float
        c_s = sqrt(T_e / m_i) with the lightest ion, m/s (0 without ions).
    electron_ion_drift : numpy.ndarray
        The drift of the electrons relative to the ions, per component, m/s.
    temperature_ratio : float
        T_e / T_i (inf for cold ions, nan without ions).
    """

    electron_plasma_frequency: float
    ion_plasma_frequency: float
    electron_cyclotron_frequency: float
    ion_cyclotron_frequency: float
    lower_hybrid_frequency: float
    debye_length: float
    electron_thermal_speed: float
    sound_speed: float
    electron_ion_drift: np.ndarray
    temperature_ratio: float


def _scales(ds: xr.Dataset, pops: Sequence[Population] = ()) -> PlasmaScales:
    """Return the :class:`PlasmaScales` of the first record.

    With ``pops`` (the populations along one axis), the electron temperature is the thermal
    spread within the electron populations, without the spread between beams.
    """
    mass = np.array(ds.attrs["mass"])
    charge = np.array(ds.attrs["charge"])
    density = ds["density"].isel(t=0).values
    drift = ds["drift"].isel(t=0).values
    temperature = ds["temperature"].isel(t=0).values.mean(axis=-1)
    electrons = mass < _ELECTRON_MASS_LIMIT * constants.m_e
    ions = ~electrons
    square = density * charge**2 / (constants.epsilon_0 * mass)
    omega_pe = float(np.sqrt(square[electrons].sum()))
    omega_pi = float(np.sqrt(square[ions].sum()))
    b = float(ds.attrs.get("magnetic_field", 0.0))
    omega_ce = constants.e * b / constants.m_e if electrons.any() else 0.0
    light = int(np.argmin(np.where(ions, mass, np.inf))) if ions.any() else None
    omega_ci = abs(charge[light]) * b / mass[light] if light is not None else 0.0
    if omega_ce > 0 and omega_ci > 0 and omega_pi > 0:
        omega_lh = float(1.0 / np.sqrt(1.0 / (omega_ce * omega_ci) + 1.0 / omega_pi**2))
    else:
        omega_lh = 0.0
    n_e = density[electrons].sum()
    # the electrons' temperature, including the spread between their populations
    if n_e > 0:
        u_e = (density[electrons, None] * drift[electrons]).sum(axis=0) / n_e
        spread = (density[electrons, None] * (drift[electrons] - u_e) ** 2).sum(
            axis=0
        ) / n_e
        t_e = float(
            (density[electrons] * temperature[electrons]).sum() / n_e
            + constants.m_e * spread.mean()
        )
    else:
        u_e, t_e = np.zeros(drift.shape[-1]), 0.0
    beams = [p for p in pops if p.is_electron]
    if beams:
        n_b = sum(p.density for p in beams)
        t_e = constants.m_e * sum(p.density * p.thermal_speed**2 for p in beams) / n_b
    v_te = float(np.sqrt(t_e / constants.m_e))
    n_i = density[ions].sum()
    if n_i > 0:
        u_i = (density[ions, None] * drift[ions]).sum(axis=0) / n_i
        t_i = float((density[ions] * temperature[ions]).sum() / n_i)
        c_s = float(np.sqrt(t_e / mass[light]))
        ratio = t_e / t_i if t_i > 0 else np.inf
    else:
        u_i, c_s, ratio = np.zeros_like(u_e), 0.0, np.nan
    return PlasmaScales(
        electron_plasma_frequency=omega_pe,
        ion_plasma_frequency=omega_pi,
        electron_cyclotron_frequency=float(omega_ce),
        ion_cyclotron_frequency=float(omega_ci),
        lower_hybrid_frequency=omega_lh,
        debye_length=v_te / omega_pe if omega_pe > 0 else np.inf,
        electron_thermal_speed=v_te,
        sound_speed=c_s,
        electron_ion_drift=np.asarray(u_e - u_i, dtype=float),
        temperature_ratio=float(ratio),
    )


@dataclass(frozen=True)
class GrowingMode:
    """A Fourier mode of the potential that grows, from :func:`instability_report`.

    Attributes
    ----------
    axis : str
        The periodic axis along which the mode is a Fourier mode.
    mode : int
        The mode number m: m wavelengths fit in the box.
    k : float
        The wavenumber 2π m / L, rad/m.
    growth_rate : float
        γ, 1/s, of the amplitude.
    frequency : float
        The real frequency |ω_r|, rad/s, from the phase over the growth phase (0 for a purely
        growing or a standing mode).
    phase_velocity : float
        ω_r / k with the sign of the direction of travel along the axis, m/s.
    efolds : float
        The number of e-foldings of the growth phase.
    start : float
        The start of the growth phase, s.
    stop : float
        The end of the growth phase, s.
    r2 : float
        The coefficient of determination of the exponential fit.
    name : str
        The most likely instability.
    reason : str
        Why that name: the dimensionless numbers it rests on.
    predicted : complex
        ω (rad/s, Im ω the growth rate) of the most unstable electrostatic root of the measured
        populations at this k; nan when no prediction was made.
    """

    axis: str
    mode: int
    k: float
    growth_rate: float
    frequency: float
    phase_velocity: float
    efolds: float
    start: float
    stop: float
    r2: float
    name: str
    reason: str
    predicted: complex = complex(np.nan, np.nan)


@dataclass
class InstabilityReport:
    """What grows in a run and what it is, from :func:`instability_report`.

    Attributes
    ----------
    modes : list of GrowingMode
        The growing modes, fastest first.
    scales : PlasmaScales
        The plasma's scales at the first record.
    populations : dict
        The drifting Maxwellians along each periodic axis (:func:`populations`).
    penrose : dict
        Per axis, ``{name: PenroseResult}`` of the initial distributions: every species, and
        ``"electrons"`` for several electron species together.
    predicted : dict
        Per axis, ``(k, omega)`` arrays of the most unstable electrostatic root at every mode
        number, rad/m and rad/s; missing where no prediction applies.
    dominant : dict
        Per axis, the strongest wave of the whole run (:func:`plasma_plots.analysis.dominant_mode`).
    warnings : list of str
        Numerical problems and caveats.
    """

    modes: list[GrowingMode]
    scales: PlasmaScales
    populations: dict[str, list[Population]] = field(default_factory=dict)
    penrose: dict[str, dict[str, Any]] = field(default_factory=dict)
    predicted: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    dominant: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def unstable(self) -> bool:
        """Whether any mode grows."""
        return bool(self.modes)

    @property
    def names(self) -> list[str]:
        """The distinct names of the growing modes, fastest first."""
        return list(dict.fromkeys(mode.name for mode in self.modes))

    def __str__(self) -> str:
        """Return a table of the growing modes, the checks and the warnings (plasma units)."""
        s = self.scales
        omega_pe = s.electron_plasma_frequency or 1.0
        lines = [
            f"Plasma: omega_pe = {s.electron_plasma_frequency:.3e} rad/s, "
            f"lambda_De = {s.debye_length:.3e} m, omega_ce/omega_pe = "
            f"{s.electron_cyclotron_frequency / omega_pe:.3g}, Te/Ti = {s.temperature_ratio:.3g}",
        ]
        if not self.modes:
            lines.append("No growing mode.")
        else:
            lines.append(
                f"{'axis':>4} {'m':>4} {'k lDe':>8} {'gamma/wpe':>10} {'w_r/wpe':>9} "
                f"{'e-folds':>7} {'pred. gamma':>11}  name"
            )
            for m in self.modes:
                lines.append(
                    f"{m.axis:>4} {m.mode:>4} {m.k * s.debye_length:8.3f} "
                    f"{m.growth_rate / omega_pe:10.4f} {m.frequency / omega_pe:9.4f} "
                    f"{m.efolds:7.1f} {m.predicted.imag / omega_pe:11.4f}  {m.name}"
                )
        for axis, results in self.penrose.items():
            for name, result in results.items():
                verdict = "unstable" if result.unstable else "stable"
                lines.append(f"Penrose ({axis}, {name}): {verdict}")
        for axis, (k, omega) in self.predicted.items():
            rates = np.imag(omega)
            if np.isfinite(rates).any() and np.nanmax(rates) > 0:
                best = int(np.nanargmax(rates))
                lines.append(
                    f"Predicted fastest along {axis}: k lDe = {k[best] * s.debye_length:.3f}, "
                    f"gamma/wpe = {omega[best].imag / omega_pe:.4f}"
                )
            else:
                lines.append(f"Predicted along {axis}: stable at the measured k")
        lines += [f"Warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _phase_frequency(
    coefficient: np.ndarray, times: np.ndarray, start: float, stop: float
) -> float:
    """ω_r from the phase of a Fourier coefficient ``∝ exp(-i ω t)`` within [start, stop]."""
    window = (times >= start) & (times <= stop)
    if window.sum() < 3:
        return 0.0
    phase = np.unwrap(np.angle(coefficient[window]))
    return -float(np.polyfit(times[window], phase, 1)[0])


def _beam_pairs(
    pops: Sequence[Population], electrons: bool
) -> list[tuple[Population, Population]]:
    """Pairs of populations of one kind drifting apart faster than their thermal speeds."""
    group = [p for p in pops if p.is_electron == electrons]
    return [
        (a, b)
        for i, a in enumerate(group)
        for b in group[i + 1 :]
        if abs(a.drift - b.drift) > 2.0 * (a.thermal_speed + b.thermal_speed)
    ]


def _classify(
    k: float,
    omega: float,
    axis_index: int,
    scales: PlasmaScales,
    pops: Sequence[Population],
    attrs: dict[str, Any],
) -> tuple[str, str]:
    """Return the most likely name of a growing electrostatic mode, and why."""
    omega_pe = scales.electron_plasma_frequency
    dx = attrs["spacing"][axis_index]
    lambda_d = scales.debye_length
    if k * dx > 0.5 * np.pi and dx > lambda_d:
        ratio = f"{dx / lambda_d:.2f} lambda_De" if lambda_d > 0 else "> lambda_De = 0"
        return (
            "numerical (finite-grid instability)",
            f"k dx = {k * dx:.2f} near the grid scale and dx = {ratio}",
        )
    direction = np.asarray(attrs.get("magnetic_field_direction", [0.0, 0.0, 0.0]))
    along_b = abs(direction[axis_index]) if direction.size > axis_index else 0.0
    magnetized = scales.electron_cyclotron_frequency > 0.01 * omega_pe and along_b < 0.5
    u_rel = abs(float(scales.electron_ion_drift[axis_index]))
    if magnetized:
        omega_ce, omega_lh = (
            scales.electron_cyclotron_frequency,
            scales.lower_hybrid_frequency,
        )
        harmonic = k * u_rel / omega_ce
        # the ECDI sits at the Doppler-shifted cyclotron harmonics k u = n omega_ce, far below
        # omega_ce in frequency; the lower-hybrid branch at k u well below omega_ce
        if harmonic >= 0.5 and omega < omega_ce:
            return (
                "electron cyclotron drift instability",
                f"k ⊥ B, k u_d / omega_ce = {harmonic:.2f} (near harmonic {max(round(harmonic), 1)}), "
                f"omega_r / omega_ce = {omega / omega_ce:.3f}",
            )
        if omega_lh > 0 and 0.2 * omega_lh < omega < 5 * omega_lh:
            return (
                "lower-hybrid / modified two-stream instability",
                f"omega_r / omega_LH = {omega / omega_lh:.2f}, k ⊥ B, electron drift {u_rel:.3g} m/s",
            )
        if omega_lh > 0 and omega <= 0.2 * omega_lh:
            return (
                "low-frequency drift mode (gradient-driven, e.g. Simon–Hoh or a spoke)",
                f"omega_r / omega_LH = {omega / omega_lh:.3f}, below the lower hybrid frequency",
            )
    pairs = _beam_pairs(pops, electrons=True)
    if pairs:
        a, b = max(pairs, key=lambda p: abs(p[0].drift - p[1].drift))
        beam, bulk = sorted((a, b), key=lambda p: p.density)
        ratio = beam.density / bulk.density
        relative = abs(beam.drift - bulk.drift)
        if ratio > 0.25:
            return (
                "two-stream instability",
                f"electron streams at {a.drift:.3g} and {b.drift:.3g} m/s, density ratio {ratio:.2f}, "
                f"k u / omega_pe = {k * relative / 2 / omega_pe:.2f}",
            )
        if beam.thermal_speed < ratio ** (1 / 3) * relative:
            return (
                "beam–plasma instability",
                f"a cold electron beam (n_b/n_0 = {ratio:.3g}, v_tb/u = {beam.thermal_speed / relative:.3f} "
                f"< (n_b/n_0)^(1/3)), k u / omega_pe = {k * relative / omega_pe:.2f}",
            )
        return (
            "bump-on-tail instability",
            f"a warm electron beam (n_b/n_0 = {ratio:.3g}, v_tb/u = {beam.thermal_speed / relative:.3f}), "
            f"phase speed {omega / k:.3g} m/s against the beam at {beam.drift:.3g} m/s",
        )
    ion_pairs = _beam_pairs(pops, electrons=False)
    v_te = scales.electron_thermal_speed
    if scales.ion_plasma_frequency > 0 and u_rel > 0:
        if u_rel > v_te:
            return (
                "Buneman instability",
                f"electrons drift through the ions at {u_rel / max(v_te, 1e-300):.2f} v_te, "
                f"k u / omega_pe = {k * u_rel / omega_pe:.2f}, omega_r / omega_pe = {omega / omega_pe:.3f}",
            )
        c_s = scales.sound_speed
        if c_s > 0 and u_rel > c_s and omega < 3 * k * c_s:
            return (
                "ion-acoustic instability",
                f"electron drift {u_rel / v_te:.2f} v_te, Te/Ti = {scales.temperature_ratio:.3g}, "
                f"phase speed {omega / k / c_s:.2f} c_s",
            )
    if ion_pairs:
        a, b = ion_pairs[0]
        return (
            "ion–ion streaming instability",
            f"ion streams at {a.drift:.3g} and {b.drift:.3g} m/s",
        )
    return (
        "unidentified electrostatic mode",
        f"k lambda_De = {k * lambda_d:.3f}, omega_r / omega_pe = {omega / omega_pe:.3f}",
    )


def _penrose(ds: xr.Dataset, axis: str) -> dict[str, Any]:
    """Return the Penrose criterion of the initial distributions along ``axis``.

    For every species, and for all electrons together when there are several electron species
    (each weighted by its density). Streaming of electrons through ions (Buneman, ion
    acoustic) is left to the drifts and the dispersion relation: the ions' share of
    Σ (q²/m) f is too small for a dip between them to stand out of the marker noise.
    """
    out: dict[str, Any] = {}
    if "sample_velocity" not in ds:
        return out
    mass = np.array(ds.attrs["mass"])
    density = ds["density"].isel(t=0).values
    electrons: list[tuple[np.ndarray, np.ndarray]] = []
    for index, name in enumerate(ds.species.values):
        v = (
            ds["sample_velocity"]
            .isel(snapshot=0, species=index)
            .sel(component=axis)
            .values
        )
        w = ds["sample_weight"].isel(snapshot=0, species=index).values
        keep = np.isfinite(v) & np.isfinite(w)
        if not keep.any() or w[keep].sum() <= 0:
            continue
        out[str(name)] = penrose_criterion(
            _distribution(v[keep], w[keep]), prominence=0.1
        )
        if mass[index] < _ELECTRON_MASS_LIMIT * constants.m_e:
            electrons.append((v[keep], w[keep] / w[keep].sum() * density[index]))
    if len(electrons) > 1 and "electrons" not in out:
        v = np.concatenate([g[0] for g in electrons])
        w = np.concatenate([g[1] for g in electrons])
        out["electrons"] = penrose_criterion(_distribution(v, w), prominence=0.1)
    return out


def _prediction(
    pops: Sequence[Population], scales: PlasmaScales, k: np.ndarray
) -> np.ndarray:
    """ω (rad/s) of the most unstable electrostatic root of the populations at each k."""
    omega_pe = scales.electron_plasma_frequency
    electrons = [p for p in pops if p.is_electron]
    if omega_pe <= 0 or not electrons:
        return np.full(k.shape, np.nan + 1j * np.nan)
    n_ref = sum(p.density for p in electrons)
    # a velocity scale: the electrons' spread, beams included
    u_mean = sum(p.density * p.drift for p in electrons) / n_ref
    v_ref = np.sqrt(
        sum(
            p.density * (p.thermal_speed**2 + (p.drift - u_mean) ** 2)
            for p in electrons
        )
        / n_ref
    )
    v_ref = v_ref if v_ref > 0 else max(abs(p.drift) for p in pops) or 1.0
    floor = 1e-3
    species = [
        Maxwellian(
            density=p.density / n_ref,
            charge=p.charge / constants.e,
            mass=p.mass / constants.m_e,
            thermal_speed=max(p.thermal_speed / v_ref, floor),
            drift=p.drift / v_ref,
        )
        for p in pops
        if p.density > 0
    ]
    with np.errstate(all="ignore"):
        omega = np.asarray(
            electrostatic_growth(k * v_ref / omega_pe, species), dtype=complex
        )
    return omega * omega_pe


COUPLED = "nonlinear mode coupling"


def _mark_couplings(
    modes: list[GrowingMode], tolerance: float = 0.2
) -> list[GrowingMode]:
    """Return the modes, those driven by two other growing modes on their axis renamed.

    Two modes a and b growing at γ_a and γ_b drive the modes a + b and |a − b| through the
    nonlinearity of the Vlasov equation, at γ_a + γ_b: faster than any linear mode, and not an
    instability of their own (the harmonics are the case a = b). The modes are taken from the
    slowest-growing up, so a driven mode can drive further ones.
    """
    out: list[GrowingMode] = []
    for mode in sorted(modes, key=lambda m: m.growth_rate):
        for a in out:
            for b in out:
                if a.axis != mode.axis or b.axis != mode.axis or b.mode < a.mode:
                    continue
                if mode.mode not in (a.mode + b.mode, b.mode - a.mode):
                    continue
                total = a.growth_rate + b.growth_rate
                if abs(mode.growth_rate / total - 1) < tolerance:
                    sign = "+" if mode.mode == a.mode + b.mode else "-"
                    mode = replace(
                        mode,
                        name=COUPLED,
                        reason=f"mode {mode.mode} = {b.mode} {sign} {a.mode}, growing at "
                        f"{mode.growth_rate / total:.2f} times the sum of their rates",
                    )
                    break
            if mode.name == COUPLED:
                break
        out.append(mode)
    return out


def _rank(mode: GrowingMode) -> tuple[int, float]:
    """Sort key of the growing modes: linear modes the theory confirms first, fastest first.

    Then the modes the dispersion relation of the measured populations calls stable (numerical,
    e.g. the cold-beam instability of the grid, or outside the electrostatic model), then the
    modes driven by nonlinear coupling.
    """
    stable = bool(mode.predicted.imag <= 1e-9 * abs(mode.predicted))  # False for nan
    return (2 if mode.name == COUPLED else int(stable), -mode.growth_rate)


def _warnings(
    ds: xr.Dataset, scales: PlasmaScales, modes: Sequence[GrowingMode]
) -> list[str]:
    """Numerical problems of the run."""
    out = []
    attrs = ds.attrs
    omega_pe, lambda_d = scales.electron_plasma_frequency, scales.debye_length
    dx = max(attrs["spacing"])
    if dx > lambda_d:
        ratio = f"{dx / lambda_d:.2f} lambda_De" if lambda_d > 0 else "> lambda_De = 0"
        out.append(
            f"dx = {ratio}: an explicit PIC heats numerically (finite-grid "
            "instability) with cells larger than the Debye length"
        )
    step = omega_pe * attrs["dt"]
    if step > 2.0:
        out.append(
            f"omega_pe dt = {step:.2f} > 2: the leapfrog is unstable for plasma oscillations"
        )
    elif step > 0.3:
        out.append(
            f"omega_pe dt = {step:.2f} > 0.3: plasma oscillations are not accurate"
        )
    for mode in modes:
        if mode.growth_rate * attrs["dt"] > 0.1:
            out.append(
                f"mode {mode.mode} along {mode.axis} grows by {mode.growth_rate * attrs['dt']:.2f} "
                "e-foldings per step: not resolved in time"
            )
            break
    total = ds["field_energy"].values + ds["kinetic_energy"].sum("species").values
    if all(attrs["periodic"]) and total.size > 1 and total[0] > 0:
        drift = float(np.max(np.abs(total - total[0])) / total[0])
        if drift > 0.01:
            out.append(
                f"the total energy of the periodic box changed by {100 * drift:.1f} %"
            )
    if not modes and ds.sizes["t"] > 1:
        temperature = ds["temperature"].sum("component")
        start, end = temperature.isel(t=0).values, temperature.isel(t=-1).values
        for name, a, b in zip(ds.species.values, start, end, strict=True):
            if a > 0 and b > 1.2 * a:
                out.append(
                    f"{name} heated by {100 * (b / a - 1):.0f} % with no growing mode: numerical heating?"
                )
    return out


def instability_report(
    ds: xr.Dataset,
    *,
    min_efolds: float = 2.0,
    min_r2: float = 0.95,
    max_mode: int | None = None,
    predict: bool = True,
) -> InstabilityReport:
    """Detect, name and check the instabilities of a run from its instability record.

    Parameters
    ----------
    ds : xarray.Dataset
        The instability record (see the module docstring).
    min_efolds : float, optional
        The shortest growth phase that counts, in e-foldings. Default: ``2.0``.
    min_r2 : float, optional
        The smallest coefficient of determination of the exponential fit. Default: ``0.95``.
    max_mode : int, optional
        The highest mode number searched. Default: half the nodes (all of them).
    predict : bool, optional
        Solve the electrostatic dispersion relation of the measured populations. Default:
        ``True``.

    Returns
    -------
    InstabilityReport
        The growing modes (fastest first) with their names, the scales, the Penrose checks, the
        prediction and the warnings; ``print`` it for a table.
    """
    attrs = ds.attrs
    times = ds.t.values
    first = next(
        (
            AXES[a]
            for a in range(attrs["ndim"])
            if attrs["periodic"][a] and a < attrs["nvel"]
        ),
        "x",
    )
    scales = _scales(ds, populations(ds, first))
    modes: list[GrowingMode] = []
    report = InstabilityReport(modes=modes, scales=scales)
    direction = np.asarray(attrs.get("magnetic_field_direction", [0.0] * 3))
    for axis_index in range(attrs["ndim"]):
        axis = AXES[axis_index]
        if f"potential_{axis}" not in ds:
            continue
        profile = ds[f"potential_{axis}"]
        nodes = profile.sizes[axis]
        length = attrs["lengths"][axis_index]
        coefficients = np.fft.rfft(profile.values, axis=1) / nodes
        top = (
            coefficients.shape[1] - 1
            if max_mode is None
            else min(max_mode, coefficients.shape[1] - 1)
        )
        k = 2.0 * np.pi * np.arange(1, top + 1) / length
        pops = populations(ds, axis) if axis_index < attrs["nvel"] else []
        report.populations[axis] = pops
        if axis_index < attrs["nvel"]:
            report.penrose[axis] = _penrose(ds, axis)
        magnetized = (
            scales.electron_cyclotron_frequency
            > 0.01 * scales.electron_plasma_frequency
        )
        along_b = (
            abs(direction[axis_index]) > 0.99 if direction.size > axis_index else False
        )
        predicted = np.full(k.shape, np.nan + 1j * np.nan)
        if predict and pops and (not magnetized or along_b):
            predicted = _prediction(pops, scales, k)
            report.predicted[axis] = (k, predicted)
        elif predict and magnetized:
            report.warnings.append(
                f"no prediction along {axis}: the plasma is magnetized across it (the dispersion "
                "relation is the unmagnetized one)"
            )
        if ds.sizes["t"] > 2:
            report.dominant[axis] = dominant_mode(profile)
        for m in range(1, top + 1):
            amplitude = xr.DataArray(
                np.abs(coefficients[:, m]), coords={"t": times}, dims="t"
            )
            phase = detect_growth(
                amplitude, min_efolds=min_efolds, min_r2=min_r2
            ) or detect_growth(
                amplitude, min_efolds=min_efolds, min_r2=min_r2, peaks_only=True
            )
            if phase is None:
                continue
            omega_r = _phase_frequency(
                coefficients[:, m], times, phase.start, phase.stop
            )
            name, reason = _classify(
                k[m - 1], abs(omega_r), axis_index, scales, pops, attrs
            )
            modes.append(
                GrowingMode(
                    axis=axis,
                    mode=m,
                    k=float(k[m - 1]),
                    growth_rate=phase.rate,
                    frequency=abs(omega_r),
                    phase_velocity=omega_r / k[m - 1],
                    efolds=phase.efolds,
                    start=phase.start,
                    stop=phase.stop,
                    r2=phase.r2,
                    name=name,
                    reason=reason,
                    predicted=complex(predicted[m - 1]),
                )
            )
    modes[:] = _mark_couplings(modes)
    modes.sort(key=_rank)
    report.warnings[:0] = _warnings(ds, scales, modes)
    if not any(attrs["periodic"][: attrs["ndim"]]):
        report.warnings.append("no periodic axis: no Fourier modes to search")
    return report
