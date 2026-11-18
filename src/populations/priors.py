"""Vectorized prior distributions for the TREX statistical vetting engine.

Implements the astrophysical priors described in Giacalone et al. (2021):

* Planet radius: Giacalone et al. (2021) Sect. 2.4.2 broken power law.
* Orbital inclination: isotropic (uniform in cos i).
* Eccentricity: Kipping (2013) Beta distribution for planets;
  Moe & Di Stefano (2017) Eq. 17 for binaries with P > 10 d.
* Orbital period scenario prior: Giacalone et al. (2021) Sect. 2.4.1
  broken power laws integrated over +/-0.1 d.
* Mass ratio: Moe & Di Stefano (2017) short-period binary distribution.
* Bound companion occurrence: Winters et al. (2019) + Moe & Di Stefano (2017).
* Background star occurrence: TRILEGAL-based.

Samplers admit finite real unit quantiles with their documented endpoints.
Log-prior functions compute dimensionless natural-log weights.

Verified sources, units, and limits
-----------------------------------
The sources are Giacalone et al. (2021), AJ 161, 24,
DOI ``10.3847/1538-3881/abc6af`` (radius-prior coefficients);
Moe & Di Stefano (2017), ApJS 230, 15,
DOI ``10.3847/1538-4365/aa6fb6``; and
Winters et al. (2019), AJ 157, 216, DOI
``10.3847/1538-3881/ab05dc``.  Radius draws are Earth radii, primary mass is
solar mass, periods are days, inclinations/arguments are degrees, separations
are arcsec or AU as named, parallax is mas, and log priors are dimensionless.
The Winters relation is limited to its documented M-dwarf range; out-of-range
or nonfinite draws must be rejected rather than extrapolated. These are
conditional scenario priors, not empirically calibrated population assumptions.
"""

from __future__ import annotations

import math
from numbers import Integral
from functools import lru_cache as lru_cache
from typing import Tuple as Tuple

import numpy as np
from scipy.special import betaincinv as _beta_ppf
from scipy.special import betaln as _beta_ln
from scipy.special import expit as _expit
from scipy.special import logit as _logit

from trex.foundation.physical_constants import FULL_TURN_DEGREES, MILLIARCSECONDS_PER_ARCSECOND as MILLIARCSECONDS_PER_ARCSECOND, SQUARE_ARCSECONDS_PER_SQUARE_DEGREE
from trex.foundation.constants import au as au, G as G, pi, SECONDS_PER_DAY as SECONDS_PER_DAY
from trex.populations._validation import boolean, real_array, real_scalar, unit_quantiles
from trex.populations.mass_ratio_priors import (
    eb_mass_ratio_support as eb_mass_ratio_support,
    _stellar_mass_ratio_floor as _stellar_mass_ratio_floor,
    _stellar_mass_ratio_floors as _stellar_mass_ratio_floors,
    _broken_power_law_mass_ratio_inverse as _broken_power_law_mass_ratio_inverse,
    _power_law_integral as _power_law_integral,
    _power_law_inverse as _power_law_inverse,
)
from trex.populations.companion_priors import (
    moe2017_cumulative_companion_frequency as moe2017_cumulative_companion_frequency,
    moe2017_largeq_slope as moe2017_largeq_slope,
    moe2017_twin_fraction as moe2017_twin_fraction,
    companion_log_separation_au as companion_log_separation_au,
    _companion_log_period_from_log_separation as _companion_log_period_from_log_separation,
    companion_population_frequency as companion_population_frequency,
    sample_companion_log_period as sample_companion_log_period,
    sample_companion_log_projected_separation as sample_companion_log_projected_separation,
    sample_companion_log_semimajor_axis as sample_companion_log_semimajor_axis,
    companion_log_period_from_log_projected_separation as companion_log_period_from_log_projected_separation,
    companion_hierarchical_stability_supported as companion_hierarchical_stability_supported,
    _moe2017_period_inverse_cdf as _moe2017_period_inverse_cdf,
    sample_q_companion as sample_q_companion,
)

from trex.populations._parameters import (
    WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR as WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR,
    WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR as WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR,
    WINTERS_2019_M_DWARF_COMPANION_RATE as WINTERS_2019_M_DWARF_COMPANION_RATE,
    WINTERS_2019_LOG10_SEPARATION_MEAN_AU as WINTERS_2019_LOG10_SEPARATION_MEAN_AU,
    WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX as WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX,
    REGGIANI_M_DWARF_Q_EXPONENT as REGGIANI_M_DWARF_Q_EXPONENT,
    MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR as MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR,
    MOE2017_LOGP_QUAD_C2 as MOE2017_LOGP_QUAD_C2,
    MOE2017_LOGP_QUAD_C1 as MOE2017_LOGP_QUAD_C1,
    MOE2017_LOGP_QUAD_C0 as MOE2017_LOGP_QUAD_C0,
    MOE2017_EXP_SCALE as MOE2017_EXP_SCALE,
    MOE2017_COMPANION_FRACTION_SLOPE as MOE2017_COMPANION_FRACTION_SLOPE,
    MOE2017_COMPANION_LOGPERIOD_BIN_DEX as MOE2017_COMPANION_LOGPERIOD_BIN_DEX,
    MOE2017_LOGPERIOD_MIN_DAYS as MOE2017_LOGPERIOD_MIN_DAYS,
    MOE2017_LOGPERIOD_FIRST_BREAK_DAYS as MOE2017_LOGPERIOD_FIRST_BREAK_DAYS,
    MOE2017_LOGPERIOD_SECOND_BREAK_DAYS as MOE2017_LOGPERIOD_SECOND_BREAK_DAYS,
    MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT as MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT,
    MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT as MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT,
    MOE2017_LOGPERIOD_THIRD_BREAK_DAYS as MOE2017_LOGPERIOD_THIRD_BREAK_DAYS,
    MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE as MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE,
    MOE2017_COMPANION_FRACTION_F1_COEFFS as MOE2017_COMPANION_FRACTION_F1_COEFFS,
    MOE2017_COMPANION_FRACTION_F2_COEFFS as MOE2017_COMPANION_FRACTION_F2_COEFFS,
    MOE2017_COMPANION_FRACTION_F3_COEFFS as MOE2017_COMPANION_FRACTION_F3_COEFFS,
    MOE2017_PERIOD_CUMULATIVE_GRID_POINTS as MOE2017_PERIOD_CUMULATIVE_GRID_POINTS,
    MOE2017_Q_SLOPE_LOW as MOE2017_Q_SLOPE_LOW,
    MOE2017_Q_SLOPE_HIGH as MOE2017_Q_SLOPE_HIGH,
    MOE2017_TWIN_FRACTION as MOE2017_TWIN_FRACTION,
    MOE2017_Q_BREAK_LOW as MOE2017_Q_BREAK_LOW,
    MOE2017_Q_BREAK_MID as MOE2017_Q_BREAK_MID,
    MOE2017_Q_TWIN_THRESHOLD as MOE2017_Q_TWIN_THRESHOLD,
    MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR as MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR,
    MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS as MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS,
    MOE2017_LOGPERIOD_MAX_DAYS as MOE2017_LOGPERIOD_MAX_DAYS,
    MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS as MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS,
    MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX as MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX,
    MOE2017_TWIN_FRACTION_MASS_COEFFICIENT as MOE2017_TWIN_FRACTION_MASS_COEFFICIENT,
    MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS as MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS,
    MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT as MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT,
    MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS as MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS,
    MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR as MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR,
    MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS as MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS,
    _MASS_RATIO_FLOOR_MAX_ULP_STEPS as _MASS_RATIO_FLOOR_MAX_ULP_STEPS,
)

# Kipping (2013), MNRAS, 434, L51, DOI
# ``10.1093/mnrasl/slt075``: Beta-distribution eccentricity prior parameters
# for transiting planets.  Moe & Di Stefano (2017) eccentricity power laws for
# binary companions: e^-0.8 for P <= 10 d, and Eq. 17 for P > 10 d.
KIPPING_2013_ECCENTRICITY_BETA_A = 0.867
KIPPING_2013_ECCENTRICITY_BETA_B = 3.030
MOE2017_ECCENTRICITY_POWERLAW_SHORT = 0.2
MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS = 10.0
# Moe & Di Stefano (2017, ApJS 230, 15), DOI 10.3847/1538-4365/aa6fb6, Sec. 9.2, Eq. 17
# (binstat_ApJv3.tex:1361-1364), late-type primaries:
#   eta = 0.6 - 0.7 / (log10(P/day) - 0.5),  0.5 < log10(P/day) < 6.0,
# fitted for 0.8 < M1 < 3 Msun.  The density is p(e) proportional to e**eta on
# 0 <= e < 1.  TREX applies the period law only above the retained 10-d break;
# below it Eq. 17 also tends to eta = -0.8 at log P = 1, but the paper quotes
# e^-0.8 directly for P < 10 d.
MOE2017_ECCENTRICITY_ETA_INTERCEPT = 0.6
MOE2017_ECCENTRICITY_ETA_PERIOD_COEFFICIENT = 0.7
MOE2017_ECCENTRICITY_ETA_LOGPERIOD_OFFSET = 0.5
MOE2017_ECCENTRICITY_LOGPERIOD_MIN = 0.5
MOE2017_ECCENTRICITY_LOGPERIOD_MAX = 6.0
MOE2017_ECCENTRICITY_PRIMARY_MASS_MIN_SOLAR = 0.8
MOE2017_ECCENTRICITY_PRIMARY_MASS_MAX_SOLAR = 3.0

# Planet radius prior  (Giacalone et al. 2021, broken power law)

# Pre-computed normalisation constants for the broken power-law planet-radius
# prior of Giacalone et al. (2021, AJ, 161, 24), Sect. 2.4.2, Eqs. 10–11,
# DOI 10.3847/1538-3881/abc6af:
# breaks at 3 and 6 R_earth, range 0.5-20 R_earth, indices (0, -4, -0.5) for
# FGK dwarfs and (0, -7, -0.5) for M dwarfs.  The paper gives no numeric
# M/FGK mass boundary for this prior. Its Sect. 2.3 uses 0.63 Msun for
# stellar-property relations, corresponding roughly to 4000 K; adopting that
# threshold for the radius prior is a TREX modeling choice, not a published
# occurrence cutoff.
RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR = 0.63
_R_BREAK1 = 3.0    # R_earth
_R_BREAK2 = 6.0    # R_earth
_R_MIN = 0.5
_R_MAX = 20.0

# Power-law indices for M_s > 0.63 Msun
_P1_HI, _P2_HI, _P3_HI = 0.0, -4.0, -0.5
# Power-law indices for M_s <= 0.63 Msun
_P1_LO, _P2_LO, _P3_LO = 0.0, -7.0, -0.5

_A1 = _R_BREAK1 ** _P1_HI / _R_BREAK1 ** _P2_HI
_A2 = _R_BREAK2 ** _P2_HI / _R_BREAK2 ** _P3_HI
_I1 = (_R_BREAK1 ** (_P1_HI + 1) - _R_MIN ** (_P1_HI + 1)) / (_P1_HI + 1)
_I2 = _A1 * (_R_BREAK2 ** (_P2_HI + 1) - _R_BREAK1 ** (_P2_HI + 1)) / (_P2_HI + 1)
_I3 = _A2 * _A1 * (_R_MAX ** (_P3_HI + 1) - _R_BREAK2 ** (_P3_HI + 1)) / (_P3_HI + 1)
_NORM1 = 1.0 / (_I1 + _I2 + _I3)

_A3 = _R_BREAK1 ** _P1_LO / _R_BREAK1 ** _P2_LO
_A4 = _R_BREAK2 ** _P2_LO / _R_BREAK2 ** _P3_LO
_I4 = (_R_BREAK1 ** (_P1_LO + 1) - _R_MIN ** (_P1_LO + 1)) / (_P1_LO + 1)
_I5 = _A3 * (_R_BREAK2 ** (_P2_LO + 1) - _R_BREAK1 ** (_P2_LO + 1)) / (_P2_LO + 1)
_I6 = _A4 * _A3 * (_R_MAX ** (_P3_LO + 1) - _R_BREAK2 ** (_P3_LO + 1)) / (_P3_LO + 1)
_NORM2 = 1.0 / (_I4 + _I5 + _I6)


def sample_rp(x: np.ndarray, M_s: np.ndarray, flatpriors: bool = False) -> np.ndarray:
    """Sample the inverse-CDF radius prior in Giacalone et al. (2021).

    The piecewise power laws are AJ 161, 24, Sect. 2.4.2, Eqs. 10–11,
    DOI 10.3847/1538-3881/abc6af; the normalizations are defined above.

    Args:
        x: Finite real unit quantiles in [0, 1], scalar or array.
        M_s: Finite positive host masses [Msun], broadcastable with x.
        flatpriors: If True, sample uniformly in [0.5, 20] R_earth.

    Returns:
        Planet radii [R_earth], broadcast shape. Invalid inputs raise ValueError.
    """
    flatpriors = boolean(flatpriors, "flatpriors")
    x = unit_quantiles(x, "radius quantiles")
    M_s = real_array(M_s, "host masses")
    if np.any(M_s <= 0):
        raise ValueError("host masses must be positive")
    x, M_s = np.broadcast_arrays(x, M_s)

    if flatpriors:
        return x * (_R_MAX - _R_MIN) + _R_MIN
    x = x.copy()
    upper_endpoint = x == 1.0

    hi_mass = M_s > RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
    lo_mass = ~hi_mass

    # High-mass branch
    if np.any(hi_mass):
        m1 = hi_mass & (x <= _NORM1 * _I1)
        m2 = hi_mass & (x > _NORM1 * _I1) & (x <= _NORM1 * (_I1 + _I2))
        m3 = hi_mass & (x > _NORM1 * (_I1 + _I2)) & (x <= _NORM1 * (_I1 + _I2 + _I3))
        x[m1] = (x[m1] / _NORM1 * (_P1_HI + 1) + _R_MIN ** (_P1_HI + 1)) ** (1.0 / (_P1_HI + 1))
        x[m2] = ((x[m2] / _NORM1 - _I1) * (_P2_HI + 1) / _A1 + _R_BREAK1 ** (_P2_HI + 1)) ** (1.0 / (_P2_HI + 1))
        xx = x[m3]
        x[m3] = ((xx / _NORM1 - _I1 - _I2) * (_P3_HI + 1) / (_A1 * _A2) + _R_BREAK2 ** (_P3_HI + 1)) ** (1.0 / (_P3_HI + 1))

    # Low-mass branch
    if np.any(lo_mass):
        m4 = lo_mass & (x <= _NORM2 * _I4)
        m5 = lo_mass & (x > _NORM2 * _I4) & (x <= _NORM2 * (_I4 + _I5))
        m6 = lo_mass & (x > _NORM2 * (_I4 + _I5)) & (x <= _NORM2 * (_I4 + _I5 + _I6))
        x[m4] = (x[m4] / _NORM2 * (_P1_LO + 1) + _R_MIN ** (_P1_LO + 1)) ** (1.0 / (_P1_LO + 1))
        x[m5] = ((x[m5] / _NORM2 - _I4) * (_P2_LO + 1) / _A3 + _R_BREAK1 ** (_P2_LO + 1)) ** (1.0 / (_P2_LO + 1))
        xx = x[m6]
        x[m6] = ((xx / _NORM2 - _I4 - _I5) * (_P3_LO + 1) / (_A3 * _A4) + _R_BREAK2 ** (_P3_LO + 1)) ** (1.0 / (_P3_LO + 1))

    x[upper_endpoint] = _R_MAX
    return x


def radius_quantile(radius: np.ndarray, M_s: np.ndarray, flatpriors: bool = False) -> np.ndarray:
    """Radius CDF for finite real broadcast radius [R_earth]/positive mass [Msun].

    Radii outside [0.5,20] are assigned the nearest support endpoint.
    Boolean/text/nonfinite inputs or nonboolean flatpriors raise ValueError.
    """
    flatpriors = boolean(flatpriors, "flatpriors")
    radius, M_s = np.broadcast_arrays(real_array(radius, "radii"), real_array(M_s, "host masses"))
    if np.any(M_s <= 0):
        raise ValueError("host masses must be positive")
    radius = np.clip(radius, _R_MIN, _R_MAX)
    if flatpriors:
        return (radius - _R_MIN) / (_R_MAX - _R_MIN)

    def branch(values, p1, p2, p3, a1, a2, i1, i2, norm):
        # Integrals of the three power-law segments, the constants sample_rp inverts.
        result = np.empty(values.shape)
        first = values <= _R_BREAK1
        second = (~first) & (values <= _R_BREAK2)
        third = ~(first | second)
        result[first] = norm * (values[first] ** (p1 + 1) - _R_MIN ** (p1 + 1)) / (p1 + 1)
        result[second] = norm * (i1 + a1 * (values[second] ** (p2 + 1) - _R_BREAK1 ** (p2 + 1)) / (p2 + 1))
        result[third] = norm * (i1 + i2 + a1 * a2 * (values[third] ** (p3 + 1) - _R_BREAK2 ** (p3 + 1)) / (p3 + 1))
        return result

    high = M_s > RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
    result = np.empty(radius.shape)
    result[high] = branch(radius[high], _P1_HI, _P2_HI, _P3_HI, _A1, _A2, _I1, _I2, _NORM1)
    result[~high] = branch(radius[~high], _P1_LO, _P2_LO, _P3_LO, _A3, _A4, _I4, _I5, _NORM2)
    return result


# Inclination prior  (isotropic: uniform in cos i)

def sample_inc(x: np.ndarray, lower: float = 0.0, upper: float = 90.0) -> np.ndarray:
    """Sample inclinations from an isotropic distribution.

    Args:
        x: Finite real unit quantiles in [0, 1], preserving shape.
        lower: Lower bound [deg], 0 <= lower < upper <= 180.
        upper: Upper bound [deg]. Invalid inputs raise ValueError.

    Returns:
        Inclinations [deg], shape (N,).
    """
    x = unit_quantiles(x, "inclination quantiles")
    lower = real_scalar(lower, "lower inclination")
    upper = real_scalar(upper, "upper inclination")
    if not 0 <= lower < upper <= 180:
        raise ValueError("inclination bounds must satisfy 0 <= lower < upper <= 180 degrees")
    low_rad = math.radians(lower)
    up_rad = math.radians(upper)
    span = math.cos(low_rad) - math.cos(up_rad)
    if span <= 0:
        raise ValueError("inclination bounds are unresolved in float64")
    norm = 1.0 / span
    return np.degrees(np.arccos(np.cos(low_rad) - x / norm))


# Eccentricity prior

# Span of logit x and node spacing of the eccentricity table. A double below one has logit
# under 37, and uniform draws in steps of 2**-53 stay above -37; a smaller quantile from
# another source takes the lower tail. Cubic Hermite interpolation error scales as h**4 / 384.
_ECCENTRICITY_LOGIT_LIMIT: float = 38.0
_ECCENTRICITY_LOGIT_STEP: float = 1.0 / 256.0
# Below the table e < 1e-19 and F(e) = e**a / (a B(a, b)) to double precision, so
# log e = (log x + this) / a.
_ECCENTRICITY_LOWER_TAIL_LOG: float = float(
    math.log(KIPPING_2013_ECCENTRICITY_BETA_A)
    + _beta_ln(KIPPING_2013_ECCENTRICITY_BETA_A, KIPPING_2013_ECCENTRICITY_BETA_B)
)


@lru_cache(maxsize=1)
def _planet_eccentricity_table() -> Tuple[np.ndarray, np.ndarray]:
    """Nodes of logit e and d(logit e)/d(logit x) for Kipping (2013) Beta(a, b).

    The upper half uses the reflected law so 1 - x and 1 - e keep their precision.
    Both tails are power laws, straight in logit-logit space, so nodes stay uniform.
    """
    a, b = KIPPING_2013_ECCENTRICITY_BETA_A, KIPPING_2013_ECCENTRICITY_BETA_B
    nodes = round(2.0 * _ECCENTRICITY_LOGIT_LIMIT / _ECCENTRICITY_LOGIT_STEP) + 1
    logit_x = -_ECCENTRICITY_LOGIT_LIMIT + np.arange(nodes) * _ECCENTRICITY_LOGIT_STEP
    lower = logit_x < 0.0
    root = np.where(lower, _beta_ppf(a, b, _expit(logit_x)), _beta_ppf(b, a, _expit(-logit_x)))
    log_e = np.where(lower, np.log(root), np.log1p(-root))
    log_rest = np.where(lower, np.log1p(-root), np.log(root))
    # dy/dz = x (1 - x) / (e (1 - e) f(e)) for z = logit x, y = logit e and Beta density f.
    log_density = (a - 1.0) * log_e + (b - 1.0) * log_rest - _beta_ln(a, b)
    log_slope = logit_x - 2.0 * np.logaddexp(0.0, logit_x) - log_e - log_rest - log_density
    return log_e - log_rest, np.exp(log_slope)


def _planet_eccentricity(x: np.ndarray) -> np.ndarray:
    """Beta inverse CDF: cubic Hermite in logit space, power-law lower tail; x = 0 is exactly e = 0."""
    values, slopes = _planet_eccentricity_table()
    positive = x > 0.0
    logit_x = _logit(np.where(positive, x, 0.5))
    below = logit_x < -_ECCENTRICITY_LOGIT_LIMIT
    place = (np.maximum(logit_x, -_ECCENTRICITY_LOGIT_LIMIT) + _ECCENTRICITY_LOGIT_LIMIT) / _ECCENTRICITY_LOGIT_STEP
    index = np.floor(place).astype(np.intp)
    t = place - index
    y = ((1.0 + 2.0 * t) * (1.0 - t) ** 2 * values[index] + t * t * (3.0 - 2.0 * t) * values[index + 1]
         + _ECCENTRICITY_LOGIT_STEP * t * (1.0 - t) * ((1.0 - t) * slopes[index] - t * slopes[index + 1]))
    tail = np.exp((np.log(np.where(positive, x, 0.5)) + _ECCENTRICITY_LOWER_TAIL_LOG) / KIPPING_2013_ECCENTRICITY_BETA_A)
    return np.where(positive, np.where(below, tail, _expit(y)), 0.0)


def moe2017_binary_eccentricity_eta(period_days: float) -> float:
    """Eq. 17 power-law index ``eta(log P)`` for a late-type binary.

    Moe & Di Stefano (2017), ApJS 230, 15, Sec. 9.2, Eq. 17, fitted for
    ``0.8 < M1 < 3 Msun`` and ``0.5 < log10(P/day) < 6.0``.  The period is
    capped at log10(P/day)=6 above the fit; at or below its singular lower
    boundary log10(P/day)=0.5 the function raises ValueError.
    """
    period = real_scalar(period_days, "binary period", positive=True)
    log_period = math.log10(period)
    if log_period <= MOE2017_ECCENTRICITY_LOGPERIOD_MIN:
        raise ValueError("Moe & Di Stefano Eq. 17 has no finite eta at log10(P/day) <= 0.5")
    log_period = min(log_period, MOE2017_ECCENTRICITY_LOGPERIOD_MAX)
    return MOE2017_ECCENTRICITY_ETA_INTERCEPT - (
        MOE2017_ECCENTRICITY_ETA_PERIOD_COEFFICIENT
        / (log_period - MOE2017_ECCENTRICITY_ETA_LOGPERIOD_OFFSET)
    )


def sample_ecc(x: np.ndarray, planet: bool, P_orb: float) -> np.ndarray:
    """Sample orbital eccentricities.

    Planets: Kipping (2013) Beta(0.867, 3.030).
    Binaries, P <= 10 d: Moe & Di Stefano (2017) e^-0.8 (nu=0.2).
    Binaries, P > 10 d: Moe & Di Stefano (2017) Eq. 17, e^eta(log P).  The
    inverse CDF ``e = x**(1/(1+eta))`` is the exact inverse of the normalized
    density, so MC, the Python nested transform and any scalar consumer share
    the same law.
    """
    planet = boolean(planet, "planet")
    P_orb = real_scalar(P_orb, "orbital period", positive=True)
    x = unit_quantiles(x, "eccentricity quantiles")
    if np.any(x >= 1.0):
        raise ValueError("eccentricity draws must be finite values in [0, 1)")
    if planet:
        return _planet_eccentricity(x)
    if P_orb <= MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS:
        return np.power(x, 1.0 / MOE2017_ECCENTRICITY_POWERLAW_SHORT)
    eta = moe2017_binary_eccentricity_eta(P_orb)
    return np.power(x, 1.0 / (1.0 + eta))


# Argument of periastron  (uniform)

def sample_w(x: np.ndarray) -> np.ndarray:
    """Finite real [0,1] quantiles -> periastron [0,360] deg (360 equals 0)."""
    return unit_quantiles(x, "periastron quantiles") * FULL_TURN_DEGREES


# Orbital-period scenario prior  (Giacalone et al. 2021, Sect. 2.4.1)
#
# G21 Sect. 2.4.1: the only scenario prior is the probability that a
# transiting planet or eclipsing binary has the orbital period applied to the
# model, integrated over P_orb' +/- 0.1 d.  Both densities are broken power laws
# over 0.1-50 d, normalized over that complete interval:
#   planet: P^1.5 for 0.1 <= P <= 10 d, P^0 for 10 < P <= 50 d
#   EB:     P^5.0 for 0.1 <= P <=  0.3 d, P^0.5 for 0.3 < P <= 50 d
# The integral keeps the paper's normalization; it is not a per-bin density.
PERIOD_PRIOR_MIN_DAYS = 0.1
PERIOD_PRIOR_MAX_DAYS = 50.0
PERIOD_PRIOR_HALF_WIDTH_DAYS = 0.1
PLANET_PERIOD_PRIOR_BREAK_DAYS = 10.0
PLANET_PERIOD_PRIOR_LOW_EXPONENT = 1.5
PLANET_PERIOD_PRIOR_HIGH_EXPONENT = 0.0
EB_PERIOD_PRIOR_BREAK_DAYS = 0.3
EB_PERIOD_PRIOR_LOW_EXPONENT = 5.0
EB_PERIOD_PRIOR_HIGH_EXPONENT = 0.5


@lru_cache(maxsize=2)
def _period_prior_normalization(break_days, low_exponent, high_exponent):
    continuity = break_days ** low_exponent / break_days ** high_exponent
    first = (break_days ** (low_exponent + 1.0) - PERIOD_PRIOR_MIN_DAYS ** (low_exponent + 1.0)) / (low_exponent + 1.0)
    second = continuity * (
        PERIOD_PRIOR_MAX_DAYS ** (high_exponent + 1.0) - break_days ** (high_exponent + 1.0)
    ) / (high_exponent + 1.0)
    return continuity, 1.0 / (first + second)


def _period_prior_interval(
    period_days: float, break_days: float, low_exponent: float, high_exponent: float
) -> float:
    """Normalized integral of one broken power law over ``P +/- 0.1 d``.

    G21 Eq. ``Porb_jnt``.  The lower and upper edges are clamped to the
    distribution's 0.1-50 d support, which is the only extension needed for a
    candidate at the support boundary; outside it the prior stays at the edge
    bin instead of becoming zero for every scenario.
    """
    period = real_scalar(period_days, "scenario period", positive=True)
    half = PERIOD_PRIOR_HALF_WIDTH_DAYS
    center = min(max(period, PERIOD_PRIOR_MIN_DAYS + half), PERIOD_PRIOR_MAX_DAYS - half)
    low, high = center - half, center + half
    continuity, normalization = _period_prior_normalization(break_days, low_exponent, high_exponent)
    if high <= break_days:
        integral = (high ** (low_exponent + 1.0) - low ** (low_exponent + 1.0)) / (low_exponent + 1.0)
    elif low >= break_days:
        integral = continuity * (
            high ** (high_exponent + 1.0) - low ** (high_exponent + 1.0)
        ) / (high_exponent + 1.0)
    else:
        integral = (
            (break_days ** (low_exponent + 1.0) - low ** (low_exponent + 1.0)) / (low_exponent + 1.0)
            + continuity * (
                high ** (high_exponent + 1.0) - break_days ** (high_exponent + 1.0)
            ) / (high_exponent + 1.0)
        )
    return normalization * integral


def planet_period_prior(period_days: float) -> float:
    """G21 planet ``+/-0.1 d`` period prior, normalized over 0.1-50 d."""
    return _period_prior_interval(
        period_days,
        PLANET_PERIOD_PRIOR_BREAK_DAYS,
        PLANET_PERIOD_PRIOR_LOW_EXPONENT,
        PLANET_PERIOD_PRIOR_HIGH_EXPONENT,
    )


def eb_period_prior(period_days: float) -> float:
    """G21 eclipsing-binary ``+/-0.1 d`` period prior, normalized over 0.1-50 d."""
    return _period_prior_interval(
        period_days,
        EB_PERIOD_PRIOR_BREAK_DAYS,
        EB_PERIOD_PRIOR_LOW_EXPONENT,
        EB_PERIOD_PRIOR_HIGH_EXPONENT,
    )


def scenario_period_prior(period_days: float, planet: bool, use_2xP: bool = False) -> float:
    """Period prior for one scenario's applied model period.

    ``use_2xP`` scenarios apply ``2 P_orb`` to the model, so the paper's prior
    is evaluated at that physical period rather than doubled again.
    """
    planet = boolean(planet, "planet")
    use_2xP = boolean(use_2xP, "use_2xP")
    effective = real_scalar(period_days, "scenario period", positive=True) * (2.0 if use_2xP else 1.0)
    return planet_period_prior(effective) if planet else eb_period_prior(effective)


def scenario_period_log_prior(period_days: float, planet: bool, use_2xP: bool = False) -> float:
    """Natural log of :func:`scenario_period_prior` (dimensionless)."""
    return math.log(scenario_period_prior(period_days, planet=planet, use_2xP=use_2xP))


# Mass-ratio prior for short-period binaries  (Moe & Di Stefano 2017)

def sample_q(x: np.ndarray, M_s: float) -> np.ndarray:
    """Sample binary mass ratios q = M_sec / M_prim.

    Broken power-law from Moe & Di Stefano (2017) with twin fraction
    ``MOE2017_TWIN_FRACTION`` for q >= ``MOE2017_Q_TWIN_THRESHOLD``.  All
    population parameters are the named ``MOE2017_*`` constants. The secondary
    mass floor is applied through the same ULP-safe inverse CDF used for
    per-draw hosts. The 0.1 solar-mass boundary itself has zero-width support.

    Source choice: the short-period eclipsing-pair q keeps the Moe
    solar-type broken power law; Giacalone et al. (2021) Sect. 2.4.2
    discloses that it "extrapolate[s] from the results of Moe & Di Stefano
    (2017) for Sun-like stars" to every host. Delfosse et al. (2004) Sect. 5,
    Fig. 4 report that P < 50 d M-dwarf binaries favor nearly equal masses,
    but do not provide a fitted normalized density over the stellar q range.
    For M-dwarf hosts this remains an extrapolation; the bound-companion law for the
    same hosts is the separate Reggiani & Meyer (2013) field-binary q^0.25
    (:func:`sample_q_companion`), a different population, not an inconsistency
    within one population.
    """
    x = unit_quantiles(x, "mass-ratio quantiles")
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if M_s < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
        raise ValueError(
            "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass"
        )
    return _sample_broken_power_law_mass_ratio(
        x,
        _stellar_mass_ratio_floor(M_s), MOE2017_Q_SLOPE_HIGH, MOE2017_TWIN_FRACTION,
    )


def sample_q_host_mass_ratio(
    x: np.ndarray, host_mass_solar: np.ndarray
) -> np.ndarray:
    """Sample eclipsing-pair mass ratios for a per-draw host mass.

    Identical in distribution to :func:`sample_q` for a uniform host mass, but
    the support floor is resolved per draw so a secondary orbiting a low-mass
    host cannot be placed below the stellar mass-radius relation's domain.  The
    short-period ``MOE2017_Q_SLOPE_HIGH``/``MOE2017_TWIN_FRACTION`` parameters
    are used, matching the eclipsing pairs these scenarios model.
    """
    u = unit_quantiles(x, "mass-ratio quantiles")
    masses = real_array(host_mass_solar, "host masses")
    if u.shape != masses.shape:
        raise ValueError("eclipsing-pair quantiles need one per-draw host mass")
    return host_mass_ratio_inverse_cdf(masses)(u)


def host_mass_ratio_inverse_cdf(host_mass_solar: np.ndarray):
    """Return ``u -> q``, the :func:`sample_q_host_mass_ratio` law for these hosts.

    The per-host normalization is computed once, so a caller that inverts the
    same hosts' law many times (the occultor-window bisection) pays only for
    the quantile-dependent part. Finite real ``u`` in [0,1] must have exactly
    the hosts' shape; ``.cdf(q)`` returns quantiles, clipping finite real q to
    the supported stellar range. Invalid inputs raise ValueError.
    """
    masses = real_array(host_mass_solar, "host masses")
    if not np.all(np.isfinite(masses)) or np.any(
        masses < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
    ):
        raise ValueError(
            "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass"
        )
    return _broken_power_law_mass_ratio_inverse(
        _stellar_mass_ratio_floors(masses),
        MOE2017_Q_SLOPE_HIGH,
        MOE2017_TWIN_FRACTION,
        masses.shape,
    )


def _sample_broken_power_law_mass_ratio(
    u: np.ndarray, q_min: np.ndarray, p2: np.ndarray, F_twin: np.ndarray
) -> np.ndarray:
    """Inverse-CDF draw from the Moe & Di Stefano (2017) Fig. 2 mass-ratio law.

    Three segments -- a shallow ``q ** MOE2017_Q_SLOPE_LOW`` branch below the
    mass-ratio break, a steeper ``q ** p2`` branch above it, and an excess twin
    population above ``MOE2017_Q_TWIN_THRESHOLD`` -- with ``q_min``, ``p2``, and
    ``F_twin`` all resolved per draw.
    """
    return _broken_power_law_mass_ratio_inverse(q_min, p2, F_twin, u.shape)(u)


# Log-prior for background stars  (TRILEGAL-based)

def background_population_expectation(
    star_count: int,
    simulated_area_sq_deg: float,
    field_radius_arcsec: float,
) -> float:
    """Expected number of simulated background stars inside a circular field.

    The TRILEGAL cache is a realization of the candidate's own field over a
    known simulated solid angle, so its surface density is
    ``star_count / simulated_area``.  The expected number inside a circle of
    ``field_radius_arcsec`` is that density times the circle's area.

    ``simulated_area_sq_deg`` must come from the retained query that produced
    the cache.  It is deliberately a required argument: the reference
    TRICERATOPS implementation hardcodes 0.1 deg^2 in this prior, which is a
    property of *its* query rather than of any particular candidate's evidence.

    Args:
        star_count: Simulated stars retained from the cache.
        simulated_area_sq_deg: Solid angle of the TRILEGAL query [deg^2].
        field_radius_arcsec: Radius of the circle of interest [arcsec].

    Returns:
        Expected number of background stars inside the field (dimensionless).
    """
    if isinstance(star_count, (bool, np.bool_)) or not isinstance(star_count, Integral) or star_count <= 0:
        raise ValueError("background population requires a positive simulated star count")
    simulated_area_sq_deg = real_scalar(simulated_area_sq_deg, "simulated sky area", positive=True)
    field_radius_arcsec = real_scalar(field_radius_arcsec, "field radius", positive=True)
    try:
        density_per_sq_arcsec = float(star_count) / (
            simulated_area_sq_deg * SQUARE_ARCSECONDS_PER_SQUARE_DEGREE
        )
        expectation = density_per_sq_arcsec * pi * field_radius_arcsec ** 2
    except OverflowError as exc:
        raise ValueError("background population expectation is unrepresentable") from exc
    if not math.isfinite(expectation) or expectation <= 0:
        raise ValueError("background population expectation is unrepresentable")
    return expectation


__all__ = [
    "sample_rp",
    "radius_quantile",
    "sample_inc",
    "sample_ecc",
    "moe2017_binary_eccentricity_eta",
    "sample_w",
    "planet_period_prior",
    "eb_period_prior",
    "scenario_period_prior",
    "scenario_period_log_prior",
    "sample_q",
    "sample_q_host_mass_ratio",
    "eb_mass_ratio_support",
    "host_mass_ratio_inverse_cdf",
    "moe2017_cumulative_companion_frequency",
    "moe2017_largeq_slope",
    "moe2017_twin_fraction",
    "companion_log_separation_au",
    "companion_population_frequency",
    "sample_companion_log_period",
    "sample_companion_log_projected_separation",
    "sample_companion_log_semimajor_axis",
    "companion_log_period_from_log_projected_separation",
    "companion_hierarchical_stability_supported",
    "sample_q_companion",
    "background_population_expectation",
]
