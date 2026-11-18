"""Monte Carlo importance-sampling proposals."""
from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np


def _facade(name):
    """Look the name up on ``marginal_likelihoods``, the module tests patch."""
    from trex.evidence import marginal_likelihoods

    return getattr(marginal_likelihoods, name)


def _window_quantile(quantile: np.ndarray) -> np.ndarray:
    """Round an exact window quantile to the centre of its dyadic cell.

    These are the values fixed-depth bisection of ``[0, 1]`` returns, so a
    window bound carries the spacing of ``_QUANTILE_BISECTION_ITERATIONS``.
    """
    cells = 2.0 ** _facade("_QUANTILE_BISECTION_ITERATIONS")
    return (np.minimum(np.floor(quantile * cells), cells - 1.0) + 0.5) / cells


def _defensive_window_draws(
    rng: "np.random.Generator",
    n_draws: int,
    x_low: np.ndarray,
    x_high: np.ndarray,
    broad_low: Optional[np.ndarray] = None,
    broad_high: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw uniform variates from a broad-plus-window mixture with their log weights.

    The proposal is ``q(x) = a * U(broad) + (1 - a) * U(x_low, x_high)`` for the
    configured ``_DEFENSIVE_PRIOR_FRACTION`` ``a``. The prior on the variate is ``U(0, 1)`` by
    construction, because every prior here is sampled by inverse CDF, so the
    returned log weight is ``ln(p(x) / q(x)) = -ln q(x)``; the caller adds it to
    the draw's log-likelihood. The linear-evidence estimate is unbiased;
    its logarithm is not generally unbiased.

    ``broad_low``/``broad_high`` default to the whole unit interval, which is the
    prior itself. They are narrowed only where the excluded region provably
    contributes nothing: non-transiting inclinations or the other EB/twin
    mass-ratio branch. Their likelihood is identically zero, so the defensive
    component spends its draws inside the support instead of outside it.

    A draw whose window is empty or degenerate falls back to the broad component
    alone and carries only that component's weight, so an unusable window costs
    nothing but the concentration it would have bought.

    Bounds must be finite scalars or per-draw arrays. The broad interval must
    lie in [0, 1]; the narrow interval is intersected with it. A zero-width
    broad interval has zero prior mass and returns log weight -inf.
    """
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(n_draws, (int, np.integer)) or n_draws < 0:
        raise ValueError("proposal draw count must be a non-negative integer")
    low = _per_draw_values(0.0 if broad_low is None else broad_low, n_draws, "broad lower bound")
    high = _per_draw_values(1.0 if broad_high is None else broad_high, n_draws, "broad upper bound")
    window_low = _per_draw_values(x_low, n_draws, "window lower bound")
    window_high = _per_draw_values(x_high, n_draws, "window upper bound")
    if not all(np.isfinite(bound).all() for bound in (low, high, window_low, window_high)):
        raise ValueError("proposal bounds must be finite")
    if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
        raise ValueError("broad proposal bounds must satisfy 0 <= low <= high <= 1")
    window_low = np.clip(window_low, low, high)
    window_high = np.clip(window_high, low, high)
    broad_width = high - low
    broad_usable = broad_width > 0.0
    safe_broad = np.where(broad_usable, broad_width, 1.0)

    width = np.maximum(window_high - window_low, 0.0)
    usable = (width > 0.0) & broad_usable
    safe_width = np.where(usable, width, 1.0)

    broad = np.minimum(low + broad_width * rng.random(n_draws), high)
    narrow = np.minimum(window_low + width * rng.random(n_draws), window_high)
    fraction = _facade("_DEFENSIVE_PRIOR_FRACTION")
    take_narrow = usable & (rng.random(n_draws) >= fraction)
    x = np.where(take_narrow, narrow, broad)

    in_broad = broad_usable & (x >= low) & (x <= high)
    # Binary64 interpolation can round onto either upper endpoint.
    in_narrow = usable & (x >= window_low) & (x <= window_high)
    # Log densities remain finite even when a reciprocal width would overflow.
    log_density = np.logaddexp(
        np.where(in_broad, np.log(fraction) - np.log(safe_broad), -np.inf),
        np.where(in_narrow, np.log1p(-fraction) - np.log(safe_width), -np.inf),
    )
    log_weight = np.where(
        in_broad,
        np.where(usable, -log_density, np.log(safe_broad)),
        -np.inf,
    )
    return x, log_weight


def _per_draw_values(value: Any, n_draws: int, label: str) -> np.ndarray:
    """Broadcast a scalar to one value per draw, or validate a per-draw array."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not booleans")
    array = array.astype(float, copy=False)
    if array.ndim == 0:
        return np.full(n_draws, float(array), dtype=float)
    if array.shape != (n_draws,):
        raise ValueError(
            "{0} must be a scalar or an array of {1} per-draw values".format(label, n_draws)
        )
    return array


def _proposal_support_state(
    window_low: np.ndarray,
    window_high: np.ndarray,
    ratio_floor: np.ndarray,
    ratio_ceil: np.ndarray,
    supported: np.ndarray,
) -> dict:
    """Classify an occultor-radius proposal window against the prior support.

    The occultor-radius importance window is positioned from the observed
    transit depth, while the scenario's own prior has a hard support floor and
    ceiling -- an eclipsing secondary must be a star, a planet radius prior
    spans a fixed radius range. When the entire requested window lies outside
    that support the window is unusable and draws fall back to the broad
    defensive component. This classifier records that proposal state.

    It does not decide routing or exclude, clip, or reweight a draw, and
    a `below-support` result is not by itself a physical exclusion -- grazing
    geometry can still bring a larger occultor into depth agreement, which is
    exactly what a proper integrator must resolve.

    Args:
        window_low, window_high: Per-draw requested radius-ratio window bounds.
        ratio_floor, ratio_ceil: Per-draw smallest/largest radius ratio the
            scenario prior can produce.
        supported: Boolean mask of draws the scenario prior admits.

    Returns:
        Diagnostic mapping with ``state`` in ``{"usable", "below-support",
        "above-support", "degenerate-width", "no-supported-draws"}`` plus the
        per-draw fractions behind the decision. Radius ratios are
        dimensionless.
    """
    mask = np.asarray(supported, dtype=bool) & np.isfinite(ratio_floor) & np.isfinite(
        ratio_ceil
    ) & (np.asarray(ratio_ceil, dtype=float) > 0.0)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"state": "no-supported-draws", "draws_considered": 0}
    low = np.asarray(window_low, dtype=float)[mask]
    high = np.asarray(window_high, dtype=float)[mask]
    if np.any(np.isnan(low)) or np.any(np.isnan(high)):
        raise ValueError("proposal support window bounds must not be NaN")
    floor = np.asarray(ratio_floor, dtype=float)[mask]
    ceil = np.asarray(ratio_ceil, dtype=float)[mask]
    below_fraction = float(np.count_nonzero(high < floor)) / count
    above_fraction = float(np.count_nonzero(low > ceil)) / count
    if below_fraction == 1.0:
        state = "below-support"
    elif above_fraction == 1.0:
        state = "above-support"
    elif bool(np.all(high == low)):
        state = "degenerate-width"
    else:
        state = "usable"
    return {
        "state": state,
        "draws_considered": count,
        "below_fraction": below_fraction,
        "above_fraction": above_fraction,
    }
