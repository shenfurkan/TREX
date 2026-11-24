"""Bound-companion frequency, separation/period and conditional-q laws from priors.

The M-dwarf branch (0.075--0.60 solar masses) moment-matches a log-normal
semi-major axis to Winters et al. (2019)'s projected-separation fit, assuming
circular orbits and independent isotropic orientations and phases. Kepler's
third law uses both masses (G21 Sect. 2.4.3 step 3).
The higher-mass branch samples the Moe & Di Stefano (2017) Eq. 23 period
distribution.  One shared rule,
:func:`companion_hierarchical_stability_supported`, masks companion periods
that Mardling & Aarseth (2001) Eq. 90 makes dynamically unstable for the
observed inner orbit; MC, scalar, JAX and nested paths all apply it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.special import ndtri as _standard_normal_ppf

from trex.foundation.constants import Msun, au, G, pi, SECONDS_PER_DAY
from trex.populations._validation import boolean, real_array, real_scalar, unit_quantiles
from trex.populations.mass_ratio_priors import (
    _broken_power_law_mass_ratio_inverse, _stellar_mass_ratio_floor,
)
from trex.populations._parameters import (
    WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR, WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR,
    WINTERS_2019_M_DWARF_COMPANION_RATE, WINTERS_2019_LOG10_SEPARATION_MEAN_AU,
    WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX, REGGIANI_M_DWARF_Q_EXPONENT,
    MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR, MOE2017_COMPANION_FRACTION_SLOPE,
    MOE2017_COMPANION_LOGPERIOD_BIN_DEX, MOE2017_LOGPERIOD_MIN_DAYS,
    MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS, MOE2017_LOGPERIOD_FIRST_BREAK_DAYS,
    MOE2017_LOGPERIOD_SECOND_BREAK_DAYS, MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT,
    MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT, MOE2017_LOGPERIOD_THIRD_BREAK_DAYS,
    MOE2017_LOGP_QUAD_C2, MOE2017_LOGP_QUAD_C1, MOE2017_LOGP_QUAD_C0, MOE2017_EXP_SCALE,
    MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE, MOE2017_LOGPERIOD_MAX_DAYS,
    MOE2017_COMPANION_FRACTION_F1_COEFFS, MOE2017_COMPANION_FRACTION_F2_COEFFS,
    MOE2017_COMPANION_FRACTION_F3_COEFFS, MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS,
    MOE2017_Q_SLOPE_HIGH, MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX,
    MOE2017_TWIN_FRACTION, MOE2017_TWIN_FRACTION_MASS_COEFFICIENT,
    MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR, MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT,
    MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS,
    MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS, MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS,
    MOE2017_Q_BREAK_LOW, MOE2017_Q_BREAK_MID, MOE2017_PERIOD_CUMULATIVE_GRID_POINTS,
    MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR,
)


# Mardling & Aarseth (2001), MNRAS 321, 398, Eq. 90,
# DOI ``10.1046/j.1365-8711.2001.03974.x``: a
# hierarchical triple is stable when the outer periastron exceeds
#   R_p^crit / a_in = C [(1 + q_out)(1 + e_out)/(1 - e_out)^(1/2)]^(2/5)
#                     (1 - 0.3 i / pi),
# with C = 2.8 determined empirically, q_out = M_out / M_in the outer-to-inner
# mass ratio, e_out the outer eccentricity and i the mutual inclination
# (radians).  The criterion has no inner mass-ratio or eccentricity dependence;
# Vynatheya et al. (2022, MNRAS 516, 4146, DOI 10.1093/mnras/stac2540, Eq. 2, with Mardling as
# coauthor) quotes the formula verbatim.  In TREX's bound-companion scenarios
# the drawn companion is the outer body and the observed planet or eclipsing
# pair is the inner binary.  Neither outer eccentricity nor mutual inclination
# is drawn: TREX assumes e_out = i = 0 (circular, prograde coplanar).
# G21 assumes circular orbits, not zero mutual inclination. This is not the
# least restrictive MA01 form: i = pi lowers the period threshold by 0.7**1.5.
MARDLING_AARSETH_2001_STABILITY_COEFFICIENT = 2.8
MARDLING_AARSETH_2001_INCLINATION_FACTOR_PER_RADIAN = 0.3


def companion_hierarchical_stability_supported(
    log_period_days,
    observed_period_days,
    inner_binary_mass_solar,
    outer_companion_mass_solar,
    outer_eccentricity=0.0,
    mutual_inclination_rad=0.0,
    array_module=np,
):
    """Whether each drawn companion period is dynamically stable (MA01 Eq. 90).

    The mask compares the outer companion orbit (drawn period, drawn companion
    mass) with the observed inner orbit (fixed period, inner binary total mass
    -- the host mass for a planet host, the host plus secondary mass for an
    eclipsing pair; the planet's own mass is not part of any G21 scenario
    prior and is negligible against the host's).  With Kepler's third law
    using both masses the period form of the criterion is

      P_out >= P_in [C (1+q_out)^(2/5) (1+e_out)^(2/5) (1-e_out)^(-6/5)
                    (1 - 0.3 i/pi)]^(3/2) (M_in / (M_in + M_out))^(1/2),

    which at e_out = i = 0 reduces to
    ``P_out >= P_in 2.8^(3/2) (1 + q_out)^(1/10)``.  Concrete-input validation
    raises only on the numpy path; the JAX call site passes traced scalars
    produced by already-validating samplers through the identical arithmetic.

    Args:
        log_period_days: log10 of the drawn outer companion period [days].
        observed_period_days: Observed (effective) inner period [days].
        inner_binary_mass_solar: Inner binary total mass [Msun].
        outer_companion_mass_solar: Drawn companion mass [Msun].
        outer_eccentricity: Outer orbit eccentricity; 0 in production.
        mutual_inclination_rad: Mutual inclination [rad]; 0 in production.
        array_module: ``numpy`` or the JAX ``jnp`` module for traced calls.

    Returns:
        Boolean support mask with the broadcasted input shape.
    """
    module = array_module
    if module is np:
        log_period_days = real_array(log_period_days, "outer log periods")
        inner_binary_mass_solar = real_array(inner_binary_mass_solar, "inner binary masses")
        outer_companion_mass_solar = real_array(outer_companion_mass_solar, "outer companion masses")
        observed_period_days = real_scalar(observed_period_days, "observed period", positive=True)
        outer_eccentricity = real_scalar(outer_eccentricity, "outer eccentricity")
        mutual_inclination_rad = real_scalar(mutual_inclination_rad, "mutual inclination")
        if (
            not math.isfinite(float(observed_period_days))
            or observed_period_days <= 0.0
            or not math.isfinite(float(outer_eccentricity))
            or not 0.0 <= outer_eccentricity < 1.0
            or not math.isfinite(float(mutual_inclination_rad))
            or not 0.0 <= mutual_inclination_rad <= pi
        ):
            raise ValueError(
                "hierarchical stability requires a finite positive observed period, "
                "0 <= outer eccentricity < 1 and 0 <= mutual inclination <= pi"
            )
        if (
            np.any(inner_binary_mass_solar <= 0.0)
            or np.any(outer_companion_mass_solar <= 0.0)
        ):
            raise ValueError(
                "hierarchical stability requires finite periods and positive masses"
            )
    log_period = module.asarray(log_period_days, dtype=float)
    inner = module.asarray(inner_binary_mass_solar, dtype=float)
    outer = module.asarray(outer_companion_mass_solar, dtype=float)
    q_out = outer / inner
    critical_axis_ratio = (
        MARDLING_AARSETH_2001_STABILITY_COEFFICIENT
        * module.power(1.0 + q_out, 0.4)
        * module.power(1.0 + outer_eccentricity, 0.4)
        * module.power(1.0 - outer_eccentricity, -1.2)
        * (1.0 - MARDLING_AARSETH_2001_INCLINATION_FACTOR_PER_RADIAN * mutual_inclination_rad / pi)
    )
    total_mass = inner + outer
    minimum_log_period = (
        module.log10(observed_period_days)
        + 1.5 * module.log10(critical_axis_ratio)
        + 0.5 * (module.log10(inner) - module.log10(total_mass))
    )
    return log_period >= minimum_log_period


# Dupuy & Liu (2011), ApJ 733, 122, DOI
# ``10.1088/0004-637X/733/2/122``, Appendix A/Table 6: for very low-mass
# binaries the median semi-major axis / projected separation is 1.16 (68.3%
# interval 0.84--1.97), an ensemble statistical conversion, not a per-system
# conditional orbit. Retained for callers of the historical conversion only;
# the population model draws an independent orientation instead.
DUPUY_LIU_2011_MEDIAN_SEMIMAJOR_AXIS_OVER_PROJECTED_SEPARATION = 1.16


def companion_log_period_from_log_projected_separation(
    total_mass_solar, log_projected_separation_au, array_module=np
):
    """Convert a projected separation to log10 orbital period [days].

    The Dupuy & Liu (2011) median semi-major-axis factor converts the
    projected separation to a physical axis, and Kepler's third law uses the
    total pair mass, matching Giacalone et al. (2021) Sect. 2.4.3 step 3
    ("the masses of the target and simulated secondary").  The projected
    separation argument stays the quantity the Winters et al. (2019) sample
    actually measured.
    """
    if array_module is np:
        log_projected_separation_au = real_array(log_projected_separation_au, "log projected separations")
    return _companion_log_period_from_log_separation(
        total_mass_solar,
        array_module.asarray(log_projected_separation_au, dtype=float)
        + math.log10(DUPUY_LIU_2011_MEDIAN_SEMIMAJOR_AXIS_OVER_PROJECTED_SEPARATION),
        array_module=array_module,
    )


def sample_companion_log_projected_separation(x: np.ndarray, M_s: float) -> np.ndarray:
    """Sample the Winters et al. (2019) log10 projected separation [AU].

    This is the measured M-dwarf quantity (Sect. 5.4/Fig. 16): a Gaussian in
    log10 projected separation, mean log10(20 AU), sigma 1.16 dex. Retained
    as the fitted marginal, not the population's orbit generator.
    Finite real quantiles must lie in (0,1), primary mass in [0.075,0.60]
    Msun; invalid inputs raise ValueError. Output preserves quantile shape.
    """
    u = unit_quantiles(x, "companion separation quantiles", open_endpoints=True)
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if (
        not math.isfinite(M_s)
        or M_s < WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR
        or M_s > WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR
    ):
        raise ValueError(
            "Winters et al. (2019) projected separation requires a primary mass "
            "of at least 0.075 and at most 0.60 solar masses"
        )
    return (
        WINTERS_2019_LOG10_SEPARATION_MEAN_AU
        + WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX * _standard_normal_ppf(u)
    )


def winters_log_semimajor_axis_parameters() -> Tuple[float, float]:
    """Moment-matched log10(a/AU) for circular, isotropic Winters binaries.

    For u uniform on [0,1], s/a = sqrt(1-u**2). Its log10 mean is
    (ln 2 - 1)/ln 10 and variance (1 - pi**2/12)/(ln 10)**2. Assuming
    log(a) is normal matches the projected mean and variance, not its full
    Gaussian distribution: log(s/a) itself is not normal.
    """
    projection_mean = (math.log(2.0) - 1.0) / math.log(10.0)
    projection_variance = (1.0 - math.pi**2 / 12.0) / math.log(10.0)**2
    variance = WINTERS_2019_LOG10_SEPARATION_STANDARD_DEVIATION_DEX**2 - projection_variance
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("Winters projected-separation variance must exceed the orbital projection variance")
    return WINTERS_2019_LOG10_SEPARATION_MEAN_AU - projection_mean, math.sqrt(variance)


def sample_companion_log_semimajor_axis(x: np.ndarray, M_s: float) -> np.ndarray:
    """Draw moment-matched log10(a/AU), finite real quantiles strictly in (0,1).

    The primary mass is in [0.075,0.60] Msun; invalid values raise ValueError.
    Endpoint quantiles of the unbounded Gaussian have no finite separation.
    """
    u = unit_quantiles(x, "companion separation quantiles", open_endpoints=True)
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if not WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR <= M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        raise ValueError("Winters et al. (2019) M-dwarf prior requires 0.075 <= primary mass <= 0.60 solar masses")
    mean, sigma = winters_log_semimajor_axis_parameters()
    return mean + sigma * _standard_normal_ppf(u)


def _moe2017_companion_fraction_polynomials(M_s: float) -> Tuple[float, float, float]:
    """Return Eqn. 20-22's (f1, f2, f3) companion frequencies for this primary."""
    M_s = real_scalar(M_s, "primary mass", positive=True)
    log10_M_s = math.log10(M_s)
    values = []
    for coefficients in (
        MOE2017_COMPANION_FRACTION_F1_COEFFS,
        MOE2017_COMPANION_FRACTION_F2_COEFFS,
        MOE2017_COMPANION_FRACTION_F3_COEFFS,
    ):
        c0, c1, c2 = coefficients
        values.append(c0 + c1 * log10_M_s + c2 * log10_M_s ** 2)
    return values[0], values[1], values[2]


def moe2017_cumulative_companion_frequency(
    M_s: float, log_period_days: np.ndarray, *, _uncapped: bool = False
) -> np.ndarray:
    """Companions per primary with q > 0.3 and period below ``log_period_days``.

    This is the integral of Moe & Di Stefano (2017) Eqn. 23 from the domain
    minimum ``log P = 0.2``.  Eqn. 23's first segment is flat at
    ``f_logP<1;q>0.3`` across ``0.2 <= log P < 1.0``; that 0.8 dex of
    short-period companions is part of the occurrence rate and is integrated
    here.  Periods below the domain contribute nothing.

    The returned frequency is capped at one companion per primary, the
    ``w <= 1`` rule of Giacalone et al. (2021) Sect. 2.4.3.
    Moe & Di Stefano's multiplicity-frequency integral counts the mean number
    of companions per primary, which can exceed one; the cap is an adopted
    scenario-weight choice, not a correction of a fit artefact. Primaries
    above the 40-solar-mass calibration end are refused rather than extrapolated.

    Args:
        M_s: Primary mass [Msun].
        log_period_days: log10 of the orbital period [days], shape (N,).

    Returns:
        Cumulative companion frequency, shape (N,), monotone non-decreasing,
        at most one companion per primary.
    """
    _uncapped = boolean(_uncapped, "_uncapped")
    m = real_array(log_period_days, "companion orbital periods")
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if not WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR < M_s <= MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR:
        raise ValueError(
            "Moe & Di Stefano companion frequency is calibrated for primaries "
            "up to 40 solar masses; the TREX Moe branch requires mass above 0.60 solar masses"
        )

    f1, f2, f3 = _moe2017_companion_fraction_polynomials(M_s)
    alpha = MOE2017_COMPANION_FRACTION_SLOPE
    dlogP = MOE2017_COMPANION_LOGPERIOD_BIN_DEX

    # Segment integrals join at logP=1,2,3.4,5.5; f2 is centred at 2.7.
    # Eq. 23's alpha*(logP-2.7) derivative integrates to the named quadratic.
    t1 = f1 * (MOE2017_LOGPERIOD_MIN_DAYS - MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS)
    t2 = 0.5 * (MOE2017_LOGPERIOD_FIRST_BREAK_DAYS - MOE2017_LOGPERIOD_MIN_DAYS) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (MOE2017_LOGPERIOD_FIRST_BREAK_DAYS - MOE2017_LOGPERIOD_MIN_DAYS))
    t3 = 0.5 * alpha * (MOE2017_LOGPERIOD_SECOND_BREAK_DAYS ** 2 - MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT * MOE2017_LOGPERIOD_SECOND_BREAK_DAYS + MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT) + f2 * (MOE2017_LOGPERIOD_SECOND_BREAK_DAYS - MOE2017_LOGPERIOD_FIRST_BREAK_DAYS)
    t4 = (alpha * dlogP * (MOE2017_LOGPERIOD_THIRD_BREAK_DAYS - MOE2017_LOGPERIOD_SECOND_BREAK_DAYS) + f2 * (MOE2017_LOGPERIOD_THIRD_BREAK_DAYS - MOE2017_LOGPERIOD_SECOND_BREAK_DAYS)
          + (f3 - f2 - alpha * dlogP) * (
              MOE2017_LOGP_QUAD_C2 * MOE2017_LOGPERIOD_THIRD_BREAK_DAYS ** 2
              - MOE2017_LOGP_QUAD_C1 * MOE2017_LOGPERIOD_THIRD_BREAK_DAYS
              + MOE2017_LOGP_QUAD_C0
          ))
    t5 = f3 * MOE2017_EXP_SCALE * (1.0 - math.exp(MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE * (MOE2017_LOGPERIOD_MAX_DAYS - MOE2017_LOGPERIOD_THIRD_BREAK_DAYS)))

    f_comp = np.zeros_like(m)

    r01 = (m >= MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS) & (m < MOE2017_LOGPERIOD_MIN_DAYS)
    r12 = (m >= MOE2017_LOGPERIOD_MIN_DAYS) & (m < MOE2017_LOGPERIOD_FIRST_BREAK_DAYS)
    r23 = (m >= MOE2017_LOGPERIOD_FIRST_BREAK_DAYS) & (m < MOE2017_LOGPERIOD_SECOND_BREAK_DAYS)
    r34 = (m >= MOE2017_LOGPERIOD_SECOND_BREAK_DAYS) & (m < MOE2017_LOGPERIOD_THIRD_BREAK_DAYS)
    r45 = (m >= MOE2017_LOGPERIOD_THIRD_BREAK_DAYS) & (m < MOE2017_LOGPERIOD_MAX_DAYS)
    r5p = m >= MOE2017_LOGPERIOD_MAX_DAYS

    f_comp[r01] = f1 * (m[r01] - MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS)
    f_comp[r12] = t1 + 0.5 * (m[r12] - MOE2017_LOGPERIOD_MIN_DAYS) * (2.0 * f1 + (f2 - f1 - alpha * dlogP) * (m[r12] - MOE2017_LOGPERIOD_MIN_DAYS))
    f_comp[r23] = t1 + t2 + 0.5 * alpha * (m[r23] ** 2 - MOE2017_LOGPERIOD_SECOND_INTEGRAL_LINEAR_COEFFICIENT * m[r23] + MOE2017_LOGPERIOD_SECOND_INTEGRAL_CONSTANT) + f2 * (m[r23] - MOE2017_LOGPERIOD_FIRST_BREAK_DAYS)
    f_comp[r34] = t1 + t2 + t3 + alpha * dlogP * (m[r34] - MOE2017_LOGPERIOD_SECOND_BREAK_DAYS) + f2 * (m[r34] - MOE2017_LOGPERIOD_SECOND_BREAK_DAYS) + (
        f3 - f2 - alpha * dlogP) * (
            MOE2017_LOGP_QUAD_C2 * m[r34] ** 2
            - MOE2017_LOGP_QUAD_C1 * m[r34]
            + MOE2017_LOGP_QUAD_C0
        )
    f_comp[r45] = t1 + t2 + t3 + t4 + f3 * MOE2017_EXP_SCALE * (1.0 - np.exp(MOE2017_COMPANION_FRACTION_EXPONENTIAL_SLOPE * (m[r45] - MOE2017_LOGPERIOD_THIRD_BREAK_DAYS)))
    f_comp[r5p] = t1 + t2 + t3 + t4 + t5
    if np.any(f_comp < 0.0):
        raise ValueError("Moe & Di Stefano companion fraction is outside its physical domain")
    # G21 Sect. 2.4.3 caps w at 1; the internal uncapped rate is needed
    # to apply the q-support correction before that cap.
    return f_comp if _uncapped else np.minimum(f_comp, 1.0)


def _validated_companion_log_periods(log_period_days: np.ndarray) -> np.ndarray:
    m = real_array(log_period_days, "companion orbital periods")
    if (
        not np.all(np.isfinite(m))
        or np.any(m < MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS)
        or np.any(m > MOE2017_LOGPERIOD_MAX_DAYS)
    ):
        raise ValueError(
            "companion orbital period must lie inside the measured "
            "0.2 <= log10(P/day) <= 8.0 domain"
        )
    return m


def moe2017_largeq_slope(log_period_days: np.ndarray) -> np.ndarray:
    """Return Eqn. 9's ``gamma_largeq`` for solar-type primaries.

    Flat at ``MOE2017_Q_SLOPE_HIGH`` up to ``log P = 5``, then steepening by
    ``-0.3`` per dex out to the domain maximum.  Moe & Di Stefano measure this
    for 0.8-1.2 Msun primaries; TREX applies it to non-M-dwarf companions,
    including outside that range. That extrapolation is an approximation,
    not a measurement for other spectral types.
    """
    m = _validated_companion_log_periods(log_period_days)
    return _largeq_slope(m)


def _largeq_slope(m: np.ndarray) -> np.ndarray:
    excess = np.maximum(m - MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS, 0.0)
    return MOE2017_Q_SLOPE_HIGH + MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX * excess


def moe2017_twin_fraction(M_s: float, log_period_days: np.ndarray) -> np.ndarray:
    """Eqns. 5--7 twin fraction on TREX's 0.60<M1<=40 Msun Moe branch.

    The 0.60--0.80 interval remains an explicit extrapolation. Periods must
    lie in 0.2<=log10(P/day)<=8; boolean/text/nonfinite values raise ValueError.
    """
    m = _validated_companion_log_periods(log_period_days)
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if not WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR < M_s <= MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR:
        raise ValueError("Moe twin fractions require 0.60 < primary mass <= 40 solar masses")
    return _twin_fraction(M_s, m)


def _twin_fraction(M_s: float, m: np.ndarray) -> np.ndarray:
    short_period_value = (
        MOE2017_TWIN_FRACTION + MOE2017_TWIN_FRACTION_MASS_COEFFICIENT * math.log10(M_s)
    )
    if M_s <= MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR:
        cutoff = (
            MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT
            + MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS * M_s
        )
    else:
        cutoff = MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS
    span = cutoff - MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS
    if span <= 0.0:
        return np.where(m < MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS, short_period_value, 0.0)
    taper = short_period_value * (
        1.0 - (m - MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS) / span
    )
    return np.where(
        m < MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS,
        short_period_value,
        np.where(m < cutoff, taper, 0.0),
    )


def companion_log_separation_au(M_s: float | np.ndarray, log_period_days: np.ndarray) -> np.ndarray:
    """Convert companion orbital periods to log10 semi-major axis [AU].

    Kepler's third law uses ``M_s`` as the total pair mass in solar masses.
    The result is a semi-major axis, not an instantaneous projected separation.
    """
    m = real_array(log_period_days, "companion log periods")
    masses = real_array(M_s, "total pair masses")
    if np.any(masses <= 0):
        raise ValueError("total pair masses must be positive")
    # Kepler's law in log space avoids exponentiating an otherwise valid log period.
    return (np.log10(masses) + 2.0 * m - _KEPLER_LOG_PERIOD_CONSTANT) / 3.0


_KEPLER_LOG_PERIOD_CONSTANT = (
    math.log10(4.0 * pi ** 2) + 3.0 * math.log10(au)
    - math.log10(G * Msun) - 2.0 * math.log10(SECONDS_PER_DAY)
)


def _companion_log_period_from_log_separation(
    M_s: float, log_separation_au: np.ndarray, array_module=np
) -> np.ndarray:
    """Kepler period in log10 days from log10 semi-major axis [AU] and total mass [Msun]."""
    if array_module is np:
        log_separation_au = real_array(log_separation_au, "log semimajor axes")
        M_s = real_array(M_s, "total pair masses")
        if np.any(M_s <= 0):
            raise ValueError("total pair masses must be positive")
    return (3.0 * array_module.asarray(log_separation_au, dtype=float)
            - array_module.log10(M_s) + _KEPLER_LOG_PERIOD_CONSTANT) / 2.0


def companion_population_frequency(M_s: float) -> float:
    """Full-range bound-companion weight before imaging constraints.

    The Winters (2019) branch starts from the corrected mean number of stellar
    companions per primary (section 5.2). It retains only the
    fraction of the assumed M-dwarf q law above the stellar grid's 0.1 Msun
    floor; the original survey's stellar floor is 0.075 Msun. Applying a
    field companion q distribution is an approximation, not a calibrated
    mass-conditioned multiplicity or hierarchical-triple model.
    Primaries whose stellar mass-ratio floor reaches q = 1 (roughly at or
    below 0.1 Msun) have no supported stellar companion and raise rather
    than drawing a boundary twin. The other branch integrates the Moe & Di Stefano
    (2017) Eqn. 23 q > 0.3 rate corrected to the drawn stellar-q support; its
    calibration spans 0.8--40 Msun, so 0.6--0.8 is a documented extrapolation
    and above 40 Msun is refused.
    """
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        if M_s < WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR:
            raise ValueError(
                "Winters et al. (2019) M-dwarf prior requires a primary mass of at least 0.075 solar masses"
            )
        if _stellar_mass_ratio_floor(M_s) >= 1.0:
            raise ValueError("bound-companion model has no supported stellar mass-ratio domain")
        # shortcut: field companion counts do not condition on a planet or inner binary.
        # Integrate q^beta from each secondary-mass floor to q=1; masses in Msun.
        exponent = 1.0 + REGGIANI_M_DWARF_Q_EXPONENT
        survey_floor = WINTERS_2019_M_DWARF_MINIMUM_MASS_SOLAR / M_s
        grid_floor = _stellar_mass_ratio_floor(M_s)
        retained = math.expm1(exponent * math.log(grid_floor)) / math.expm1(
            exponent * math.log(survey_floor)
        )
        return float(WINTERS_2019_M_DWARF_COMPANION_RATE * retained)
    return min(float(_moe2017_supported_period_cumulative(M_s)[1][-1]), 1.0)


def sample_companion_log_period(x: np.ndarray, M_s: float) -> np.ndarray:
    """Sample log10 orbital periods [days] from the q-support-corrected Moe law.

    This sampler serves the non-M-dwarf branch only; below the Winters et al.
    (2019) mass boundary the population samples a moment-matched semi-major
    axis and derives the period from the total mass afterwards.

    Args:
        x: Finite real uniform quantiles in [0, 1], preserving shape.
        M_s: Primary mass [Msun], above the Winters boundary.

    Returns:
        log10(P/day), shape (N,).
    """
    u = unit_quantiles(x, "companion period quantiles")
    M_s = real_scalar(M_s, "primary mass", positive=True)
    if not math.isfinite(M_s) or M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        raise ValueError(
            "Moe & Di Stefano companion period sampling requires a primary mass "
            "above the 0.60 solar-mass Winters boundary"
        )

    grid, normalized_cumulative = _moe2017_period_inverse_cdf(M_s)
    return np.interp(u, normalized_cumulative, grid)


def _moe2017_period_inverse_cdf(M_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Tabulated q-support-corrected cumulative, normalized to unity.

    The typed, bounded cache retains up to 32 host masses, not just the last
    host. Its arrays are read-only. Normalize the full rate, never a pointwise
    capped cumulative: the occurrence cap must not erase the long-period tail.
    """
    M_s = real_scalar(M_s, "primary mass", positive=True)
    return _cached_moe2017_period_inverse_cdf(M_s)


@lru_cache(maxsize=32, typed=True)
def _cached_moe2017_period_inverse_cdf(M_s: float) -> Tuple[np.ndarray, np.ndarray]:
    grid, cumulative = _moe2017_supported_period_cumulative(M_s)
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Moe & Di Stefano companion frequency has no positive normalization")
    normalized = cumulative / total
    grid.setflags(write=False)
    normalized.setflags(write=False)
    return grid, normalized


def _moe2017_q_support_ratio(M_s: float, log_period_days: np.ndarray, q_min: float) -> np.ndarray:
    """Ratio of the adopted q density above q_min to that above 0.3."""
    if not MOE2017_Q_BREAK_LOW <= q_min < 1.0:
        raise ValueError("Moe mass-ratio support must start in [0.1, 1)")
    periods = np.asarray(log_period_days, dtype=float)
    # MDS17 §2 Eq. 2 and Fig. 2: continuous broken q powers plus excess twins,
    # normalized to q > 0.3 at each log10(P/day), not once over all periods.
    distribution = _broken_power_law_mass_ratio_inverse(
        MOE2017_Q_BREAK_LOW, moe2017_largeq_slope(periods),
        moe2017_twin_fraction(M_s, periods), periods.shape,
    )
    return (1.0 - distribution.cdf(np.full_like(periods, q_min))) / (
        1.0 - distribution.cdf(np.full_like(periods, MOE2017_Q_BREAK_MID))
    )


def _moe2017_supported_period_cumulative(M_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uncapped Eq. 23 integral after period-dependent stellar-q correction."""
    M_s = real_scalar(M_s, "primary mass", positive=True)
    return _cached_moe2017_supported_period_cumulative(M_s)


@lru_cache(maxsize=32, typed=True)
def _cached_moe2017_supported_period_cumulative(M_s: float) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS, MOE2017_LOGPERIOD_MAX_DAYS,
                       MOE2017_PERIOD_CUMULATIVE_GRID_POINTS)
    original = moe2017_cumulative_companion_frequency(M_s, grid, _uncapped=True)
    midpoint = (grid[:-1] + grid[1:]) / 2.0
    ratio = _moe2017_q_support_ratio(M_s, midpoint, _stellar_mass_ratio_floor(M_s))
    # Midpoint quadrature uses the exact Eq. 23 integral in each log-period bin.
    cumulative = np.concatenate(([0.0], np.cumsum(np.diff(original) * ratio)))
    if not np.isfinite(cumulative).all() or np.any(np.diff(cumulative) < 0):
        raise ValueError("Moe & Di Stefano period cumulative must be finite and monotone")
    grid.setflags(write=False)
    cumulative.setflags(write=False)
    return grid, cumulative


def _m_dwarf_mass_ratio_inverse(u, floor, array_module=np):
    """Inverse of the Reggiani--Meyer q^beta density on [floor, 1]."""
    exponent = 1.0 + REGGIANI_M_DWARF_Q_EXPONENT
    q = (floor ** exponent + u * (1.0 - floor ** exponent)) ** (1.0 / exponent)
    return array_module.maximum(q, floor)


def sample_q_companion(
    x: np.ndarray, M_s: float, log_period_days: "np.ndarray | None" = None
) -> np.ndarray:
    """Sample bound-companion mass ratios for each separation/period draw.

    M dwarfs use the field-binary q^0.25 model of Reggiani & Meyer (2013),
    independent of the Winters projected separation, with the q floor set by
    the 0.1 Msun stellar grid; the period argument is unused there and may be
    ``None``. The model is extrapolated beyond the M-dwarf observations'
    separation and mass-ratio coverage and is not a calibrated planet-host or
    triple distribution. Higher masses use the solar-primary broken power law
    of Moe & Di Stefano (2017) Fig. 2 and Eqns. 5-7, 9 and 13, which remains
    an extrapolation outside 0.8--1.2 Msun. ``x`` and ``log_period_days``
    must be the same length on the Moe branch.
    """
    u = unit_quantiles(x, "companion mass-ratio quantiles")
    M_s = real_scalar(M_s, "primary mass", positive=True)

    if M_s <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        floor = _stellar_mass_ratio_floor(M_s)
        if floor >= 1.0:
            raise ValueError("bound-companion model has no supported stellar mass-ratio domain")
        # shortcut: q--separation independence is extrapolated; replace with a calibrated joint model.
        return _m_dwarf_mass_ratio_inverse(u, floor)

    if log_period_days is None:
        raise ValueError(
            "Moe & Di Stefano mass-ratio law requires one orbital period per draw"
        )
    m = _validated_companion_log_periods(log_period_days)
    if u.shape != m.shape:
        raise ValueError("companion mass-ratio quantiles need one per-draw orbital period")
    if M_s < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
        raise ValueError(
            "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass"
        )

    if M_s > MOE2017_PRIMARY_MASS_MAXIMUM_SOLAR:
        raise ValueError("Moe twin fractions require 0.60 < primary mass <= 40 solar masses")
    return _broken_power_law_mass_ratio_inverse(
        _stellar_mass_ratio_floor(M_s),
        _largeq_slope(m),
        _twin_fraction(M_s, m),
        u.shape,
    )(u)
