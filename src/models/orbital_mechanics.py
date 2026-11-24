"""Keplerian orbital and eclipse geometry."""

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from trex.foundation.physical_constants import GRAVITATIONAL_CONSTANT_CGS, SECONDS_PER_DAY


def semi_major_axis_cgs(period_days: ArrayLike, M_total_g: ArrayLike) -> float | np.ndarray:
    """Semi-major axis [cm] from Kepler's third law.

    a^3 = G * M_total * P^2 / (4 * pi^2). Period [days] and total mass [g]
    must be finite positive real scalars or broadcastable arrays. Booleans,
    text and invalid physical values raise ``ValueError``. A scalar result
    is returned as float; otherwise the broadcast array shape is preserved.
    """
    period, mass = np.asarray(period_days), np.asarray(M_total_g)
    for value in (period, mass):
        if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("orbital period and total mass must be finite positive real numbers")
    period_seconds = period.astype(float, copy=False) * SECONDS_PER_DAY
    val = (GRAVITATIONAL_CONSTANT_CGS * mass.astype(float, copy=False) * period_seconds ** 2 / (4.0 * math.pi ** 2)) ** (1.0 / 3.0)
    if not np.all(np.isfinite(val)) or np.any(val <= 0):
        raise ValueError("semi-major axis cannot be represented finitely and positively")
    return float(val) if np.ndim(val) == 0 else val


def a_over_Rs(period_days: float, M_s_g: float, R_s_cm: float, M_p_g: float = 0.0) -> float:
    """Scaled semi-major axis a/R_s, for positive period, stellar mass and radius.

    Scalar inputs use days, grams and cm; planet mass may be zero, not negative.
    Nonfinite, non-real and boolean inputs raise ``ValueError``.
    """
    for value in (period_days, M_s_g, R_s_cm):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("period, stellar mass and radius must be finite positive real scalars")
    if isinstance(M_p_g, (bool, np.bool_)) or not isinstance(M_p_g, Real) or not math.isfinite(M_p_g) or M_p_g < 0:
        raise ValueError("planet mass must be a finite nonnegative real scalar")
    a_cm = semi_major_axis_cgs(period_days, M_s_g + M_p_g)
    return a_cm / R_s_cm


def impact_parameter(
    inc_deg: float, a_Rs: float, ecc: float = 0.0, argp_deg: float = 90.0,
) -> float:
    """Impact parameter b for circular or eccentric orbits.

    b = (a/R_s) * abs(cos(i)) * (1-e^2) / (1 + e*sin(w)).
    The unsigned sky separation treats mirrored inclinations identically.
    Angles are degrees, inclination is in [0, 180], a/R_s is positive,
    and eccentricity is in [0, 1). Invalid inputs raise ``ValueError``.
    """
    _validate_eccentric_orbit(ecc, argp_deg)
    if isinstance(inc_deg, (bool, np.bool_)) or not isinstance(inc_deg, Real) or not math.isfinite(inc_deg) or not 0 <= inc_deg <= 180:
        raise ValueError("inclination must be a finite real angle in [0, 180] degrees")
    if isinstance(a_Rs, (bool, np.bool_)) or not isinstance(a_Rs, Real) or not math.isfinite(a_Rs) or a_Rs <= 0:
        raise ValueError("scaled semi-major axis must be finite and positive")
    cos_i = math.cos(math.radians(inc_deg))
    if ecc == 0.0:
        return a_Rs * abs(cos_i)
    sin_w = math.sin(math.radians(argp_deg))
    return a_Rs * abs(cos_i) * (1.0 - ecc ** 2) / (1.0 + ecc * sin_w)


def _validate_eccentric_orbit(ecc: float, argp_deg: float) -> None:
    if isinstance(ecc, (bool, np.bool_)) or not isinstance(ecc, Real) or not math.isfinite(ecc) or not 0.0 <= ecc < 1.0:
        raise ValueError("eccentricity must be finite and in [0, 1)")
    if isinstance(argp_deg, (bool, np.bool_)) or not isinstance(argp_deg, Real) or not math.isfinite(argp_deg):
        raise ValueError("argument of periastron must be finite")


def _mean_anomaly(true_anomaly: float, ecc: float) -> float:
    eccentric_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - ecc) * math.sin(0.5 * true_anomaly),
        math.sqrt(1.0 + ecc) * math.cos(0.5 * true_anomaly),
    )
    return eccentric_anomaly - ecc * math.sin(eccentric_anomaly)


def _solve_kepler(mean: np.ndarray, ecc: float) -> np.ndarray:
    """Solve E - e sin(E) = M for M in [-pi, pi], in float64."""
    target = np.abs(mean)
    lower = np.zeros_like(target)
    upper = np.full_like(target, math.pi)
    estimate = np.minimum(upper, target + ecc)
    for _ in range(64):
        sine = np.sin(estimate)
        half_sine = np.sin(0.5 * estimate)
        square = estimate**2
        # Near periastron, subtracting E - e*sin(E) directly loses the root's
        # significant digits when e is near one. Below x=1/4 the omitted
        # x^15/15! term is smaller than one ulp of x - sin(x).
        difference = np.where(
            estimate < 0.25,
            estimate * square * (
                1/6 - square * (1/120 - square * (1/5040 - square * (
                    1/362880 - square * (1/39916800 - square/6227020800))))
            ),
            estimate - sine,
        )
        residual = (1 - ecc) * estimate + ecc * difference - target
        step = residual / ((1 - ecc) + 2 * ecc * half_sine**2)
        proposal = estimate - step
        spacing = 4 * np.spacing(estimate)
        close_bracket = upper - lower <= spacing
        done = (np.abs(step) <= spacing) | close_bracket | (target == 0)
        if np.all(done):
            return np.copysign(np.where(target == 0, 0.0, np.where(close_bracket, estimate, proposal)), mean)
        lower = np.where(residual < 0, estimate, lower)
        upper = np.where(residual > 0, estimate, upper)
        bisected = (lower + upper) / 2
        estimate = np.where(done, estimate,
                            np.where((proposal > lower) & (proposal < upper), proposal, bisected))
    raise FloatingPointError("Kepler equation did not converge in float64")


def kepler_sky_geometry(
    time_days: np.ndarray, period_days: float, a_rs: float, inc_deg: float,
    ecc: float = 0.0, argp_deg: float = 90.0, t0_days: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected separation in stellar radii and the foreground mask.

    Times, epoch and period share a day scale, and the epoch is defined at
    inferior conjunction (omega + f = pi/2, the edge-on approximation).
    Angles are degrees; distances are in units of the occulted body's radius.
    Both outputs match the input shape. Winn (2010), arXiv:1001.2010, Sec. 2.2,
    Eqs. (2)–(5): +Z is toward the observer. No finite far-side sentinel.
    Invalid geometry raises ValueError; an unresolved float64 solve raises
    FloatingPointError. Mirrors the geometry of ``_jax_orbits._sky_geometry``.
    """
    _validate_eccentric_orbit(ecc, argp_deg)
    if not (math.isfinite(period_days) and period_days > 0 and math.isfinite(a_rs) and a_rs > 0
            and math.isfinite(t0_days) and 0 <= inc_deg <= 180):
        raise ValueError("invalid Keplerian orbit")
    phase = (np.asarray(time_days, dtype=np.float64) - t0_days) * (math.tau / period_days)
    if not np.all(np.isfinite(phase)):
        raise ValueError("Keplerian phase must be finite")
    inc = math.radians(inc_deg)
    if ecc == 0:
        along, toward = np.sin(phase), np.cos(phase)
    else:
        argp = math.radians(argp_deg)
        mean = np.fmod(phase + _mean_anomaly(math.pi / 2 - argp, ecc), math.tau)
        mean = np.where(mean > math.pi, mean - math.tau, np.where(mean < -math.pi, mean + math.tau, mean))
        anomaly = _solve_kepler(mean, ecc)
        x = (1 - ecc) - 2 * np.sin(0.5 * anomaly)**2
        y = math.sqrt((1 - ecc) * (1 + ecc)) * np.sin(anomaly)
        along = math.cos(argp) * x - math.sin(argp) * y
        toward = math.sin(argp) * x + math.cos(argp) * y
    return a_rs * np.hypot(along, toward * math.cos(inc)), toward * math.sin(inc) > 0


def secondary_eclipse_phase(ecc: float, argp_deg: float) -> float:
    """Secondary phase in [0, 1), from Keplerian mean anomalies.

    Eccentricity is a real scalar in [0, 1), and argument of periastron is a
    finite real angle in degrees; booleans are refused. Conjunctions use true
    anomalies pi/2-w and 3pi/2-w (the edge-on conjunction approximation).
    """
    _validate_eccentric_orbit(ecc, argp_deg)
    argp = math.radians(argp_deg)

    transit_mean_anomaly = _mean_anomaly(0.5 * math.pi - argp, ecc)
    occultation_mean_anomaly = _mean_anomaly(1.5 * math.pi - argp, ecc)
    return (occultation_mean_anomaly - transit_mean_anomaly) % (2.0 * math.pi) / (2.0 * math.pi)


__all__ = ["semi_major_axis_cgs", "a_over_Rs", "impact_parameter", "kepler_sky_geometry", "secondary_eclipse_phase"]
