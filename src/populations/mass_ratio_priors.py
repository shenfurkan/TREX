"""Moe--Di Stefano mass-ratio inversion kernels, re-exported by priors."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from trex.populations._parameters import (
    MOE2017_Q_BREAK_LOW, MOE2017_Q_BREAK_MID, MOE2017_Q_SLOPE_LOW,
    MOE2017_Q_TWIN_THRESHOLD, _MASS_RATIO_FLOOR_MAX_ULP_STEPS,
)
from trex.populations._validation import boolean, real_array, real_scalar, unit_quantiles


def _stellar_mass_ratio_floor(primary_mass_solar: float) -> float:
    """Smallest mass ratio whose secondary is still a star.

    ``MOE2017_Q_BREAK_LOW`` is the mass-ratio break Moe & Di Stefano quote for a
    solar-mass primary, i.e. a secondary at the hydrogen-burning boundary.  For
    a lighter primary the same absolute secondary mass is a larger ratio, so the
    support floor rises accordingly -- the reference implementation's
    "prevent M_comp < 0.1 solar masses" condition.
    """
    mass = real_scalar(primary_mass_solar, "primary mass", positive=True)
    return float(_stellar_mass_ratio_floors(np.array([mass]))[0])


def _stellar_mass_ratio_floors(primary_mass_solar: np.ndarray) -> np.ndarray:
    """Per-primary mass-ratio floors whose secondaries stay on the stellar floor.

    NUMERICAL_GUARD: in binary64 ``(MOE2017_Q_BREAK_LOW / M) * M`` falls one ulp
    short of ``MOE2017_Q_BREAK_LOW`` for a fraction of primary masses, so a draw
    at the floor would yield a secondary a hair below the hydrogen-burning
    boundary and outside the stellar mass-radius relation's domain. The floor is
    therefore raised to the smallest representable ratio whose product with the
    primary mass is not below that boundary. The ratios skipped are those whose
    exact secondary mass is below the boundary, so no supported draw is removed;
    because rounded multiplication is monotone, every ratio at or above the
    returned floor yields a secondary mass at or above the boundary.
    """
    masses = real_array(primary_mass_solar, "primary masses")
    if np.any(masses <= 0):
        raise ValueError("primary masses must be positive")
    with np.errstate(over="ignore", divide="ignore"):
        floors = np.where(masses >= 1.0, MOE2017_Q_BREAK_LOW, MOE2017_Q_BREAK_LOW / masses)
    if not np.isfinite(floors).all():
        raise ValueError("mass-ratio floor cannot represent the stellar secondary-mass boundary")
    # The shortfall is at most an ulp of the boundary, so a few increments of
    # the ratio suffice; the bound only guards the loop.
    for _ in range(_MASS_RATIO_FLOOR_MAX_ULP_STEPS):
        short = floors * masses < MOE2017_Q_BREAK_LOW
        if not np.any(short):
            return floors
        floors = np.where(short, np.nextafter(floors, np.inf), floors)
    raise ValueError("mass-ratio floor cannot represent the stellar secondary-mass boundary")


# Public support planners use the same ULP-safe floor as the sampling kernels.
stellar_mass_ratio_floor = _stellar_mass_ratio_floor
stellar_mass_ratio_floors = _stellar_mass_ratio_floors


def broken_power_law_coefficients(q_min: np.ndarray, p2: np.ndarray, F_twin: np.ndarray):
    """Segment bounds and normalization of the Moe & Di Stefano (2017) Fig. 2 law.

    Returns ``(steep_lower, twin_lower, A1, A2, I1, I2, I3)`` broadcast over the
    inputs: the lower bounds of the steep and twin segments, the continuity
    factors and the unnormalized integral of each segment. The NumPy inverse
    CDF and the device background-EB integrand share this one evaluation.
    """
    p1 = MOE2017_Q_SLOPE_LOW
    q_break = MOE2017_Q_BREAK_MID
    q_twin = MOE2017_Q_TWIN_THRESHOLD
    q_min, p2, F_twin = np.broadcast_arrays(
        real_array(q_min, "mass-ratio floors"), real_array(p2, "mass-ratio slopes"),
        real_array(F_twin, "twin fractions"),
    )
    if np.any((q_min <= 0) | (q_min >= 1.0)):
        raise ValueError(
            "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass"
        )
    if np.any((F_twin < 0) | (F_twin >= 1)):
        raise ValueError("twin fractions must lie in [0, 1)")

    s = p2 + 1.0
    # When the support already begins above the mass-ratio break there is no
    # shallow small-q segment left, so the steep segment starts at q_min and
    # carries no continuity rescaling -- the same structure the reference
    # implementation uses for its low-mass primaries.
    has_small_q_segment = q_min < q_break
    steep_lower = np.where(has_small_q_segment, q_break, q_min)
    A1 = np.where(
        has_small_q_segment, q_break ** p1 / np.power(q_break, p2), 1.0
    )

    twin_ratio = (
        _power_law_integral(steep_lower, 1.0, s) / _power_law_integral(q_twin, 1.0, s)
    )
    A2 = 1.0 + (F_twin / (1.0 - F_twin)) * twin_ratio

    I1 = np.where(
        has_small_q_segment,
        (q_break ** (p1 + 1.0) - np.power(q_min, p1 + 1.0)) / (p1 + 1.0),
        0.0,
    )
    # Above the twin threshold only q**p2 remains; its constant twin factor cancels.
    twin_lower = np.maximum(q_twin, q_min)
    I2 = A1 * _power_law_integral(steep_lower, twin_lower, s)
    I3 = A2 * A1 * _power_law_integral(twin_lower, 1.0, s)
    if not all(np.isfinite(item).all() for item in (A1, A2, I1, I2, I3)):
        raise ValueError("Moe & Di Stefano mass-ratio normalization is not positive")
    return steep_lower, twin_lower, A1, A2, I1, I2, I3


def _broken_power_law_mass_ratio_inverse(
    q_min: np.ndarray, p2: np.ndarray, F_twin: np.ndarray, shape: Tuple[int, ...]
):
    """Prepare the dimensionless three-segment law and return ``u -> q``.

    Finite real q_min in (0,1), finite slope p2, and F_twin in [0,1) must
    broadcast to shape. The returned callable requires finite real [0,1]
    quantiles with EXACTLY that shape. Its ``.cdf(q)`` attribute accepts finite
    real q with that shape, clipping q to [q_min,1] before integrating.
    Invalid parameters/shape/quantiles raise ValueError.
    """
    p1 = MOE2017_Q_SLOPE_LOW
    q_twin = MOE2017_Q_TWIN_THRESHOLD
    q_min, p2, F_twin = (
        np.broadcast_to(real_array(value, name), shape) for value, name in
        ((q_min, "mass-ratio floors"), (p2, "mass-ratio slopes"), (F_twin, "twin fractions"))
    )
    s = p2 + 1.0
    steep_lower, twin_lower, A1, A2, I1, I2, I3 = broken_power_law_coefficients(q_min, p2, F_twin)
    total = I1 + I2 + I3
    if np.any(total <= 0.0) or not np.all(np.isfinite(total)):
        raise ValueError("Moe & Di Stefano mass-ratio normalization is not positive")

    q_min_power = np.power(q_min, p1 + 1.0)
    I12 = I1 + I2
    A12 = A1 * A2

    def invert(u: np.ndarray) -> np.ndarray:
        u = unit_quantiles(u, "mass-ratio quantiles")
        if u.shape != shape:
            raise ValueError("mass-ratio quantiles must match the prepared host shape")
        target = u * total
        q = np.empty(u.shape, dtype=float)
        first = target <= I1
        second = (~first) & (target <= I12)
        third = ~(first | second)

        q[first] = np.power(
            target[first] * (p1 + 1.0) + q_min_power[first],
            1.0 / (p1 + 1.0),
        )
        q[second] = _power_law_inverse(
            steep_lower[second], (target[second] - I1[second]) / A1[second], s[second]
        )
        q[third] = _power_law_inverse(
            twin_lower[third],
            (target[third] - I1[third] - I2[third]) / A12[third],
            s[third],
        )
        # NUMERICAL_GUARD: the exact inverse CDF never returns less than q_min, but
        # its power and log evaluations can round a quantile at the floor to a value
        # an ulp below it. Such a value is the floor itself, not a draw outside the
        # support, so it is returned as the floor.
        np.maximum(q, q_min, out=q)
        q[u == 0] = q_min[u == 0]
        q[u == 1] = 1.0
        if not np.isfinite(q).all() or np.any(q > 1):
            raise ValueError("mass-ratio inverse is unrepresentable on its support")
        return q

    def cdf(q: np.ndarray) -> np.ndarray:
        """The exact inverse of ``invert``: quantile of ``q``, clipped to the support."""
        q = real_array(q, "mass ratios")
        if q.shape != shape:
            raise ValueError("mass ratios must match the prepared host shape")
        q = np.clip(q, q_min, 1.0)
        first = q < steep_lower
        second = (~first) & (q < q_twin)
        third = ~(first | second)
        result = np.empty(shape)
        result[first] = (np.power(q[first], p1 + 1.0) - q_min_power[first]) / (p1 + 1.0)
        result[second] = I1[second] + A1[second] * _power_law_integral(steep_lower[second], q[second], s[second])
        result[third] = I12[third] + A12[third] * _power_law_integral(twin_lower[third], q[third], s[third])
        return result / total

    invert.cdf = cdf
    return invert


def _power_law_integral(lower, upper, exponent: np.ndarray) -> np.ndarray:
    """Return the integral of ``q ** (exponent - 1)`` from ``lower`` to ``upper``.

    Bounds/exponents may be broadcast arrays. Evaluated as
    ``(upper**s - lower**s) / s`` but written with ``expm1``/``log``
    so it stays accurate through the removable singularity at ``s = 0``, where
    the integral is ``log(upper / lower)``.  A power-law slope of exactly -1 is
    reachable here because the slope varies continuously with orbital period.
    """
    s = np.asarray(exponent, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    span = np.log(np.asarray(upper, dtype=float)) - np.log(lower_arr)
    scaled = np.power(lower_arr, s)
    ratio = np.where(
        s == 0.0, span, np.expm1(s * span) / np.where(s == 0.0, 1.0, s)
    )
    return scaled * ratio


def _power_law_inverse(
    lower, integral: np.ndarray, exponent: np.ndarray
) -> np.ndarray:
    """Invert ``_power_law_integral(lower, q, exponent)`` for ``q``."""
    s = np.asarray(exponent, dtype=float)
    value = np.asarray(integral, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    log_lower = np.log(lower_arr)
    if not np.any(s == 0.0):
        # Nonzero-exponent inverse of the power-law integral above; q is dimensionless.
        scaled = value * s / np.power(lower_arr, s)
        return np.exp(log_lower + np.log1p(scaled) / s)
    scaled = np.where(
        s == 0.0,
        0.0,
        value * s / np.power(lower_arr, np.where(s == 0.0, 0.0, s)),
    )
    return np.where(
        s == 0.0,
        lower_arr * np.exp(value),
        np.exp(log_lower + np.log1p(scaled) / np.where(s == 0.0, 1.0, s)),
    )


def eb_mass_ratio_support(q, use_2xP):
    """Disjoint EB/twin support without renormalizing the full-q prior.

    Giacalone et al. (2021), AJ 161, 24, §2.4.4,
    DOI 10.3847/1538-3881/abc6af; TRICERATOPS 1.0.20 lnZ_TEB uses >=
    at the measure-zero twin boundary. q and the threshold are dimensionless.
    """
    use_2xP = boolean(use_2xP, "use_2xP")
    q = real_array(q, "mass ratios")
    if np.any((q <= 0) | (q > 1)):
        raise ValueError("EB mass ratios must lie in (0, 1]")
    return q >= MOE2017_Q_TWIN_THRESHOLD if use_2xP else q < MOE2017_Q_TWIN_THRESHOLD
