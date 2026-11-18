"""Numerically stable methods for the TREX Monte Carlo integrator.

These pure-numerics routines import only NumPy and SciPy, so they can be
imported and tested without the Astropy chain that ``constants`` loads.

References
----------
* ``_log_mean_exp`` replaces the TRICERATOPS 1.0.20 marginal-likelihood
  estimate ``np.log(np.mean(np.nan_to_num(np.exp(lnL + 600))))``
  (https://github.com/stevengiacalone/triceratops), whose fixed offset
  still underflows for long light curves, with an exact log-sum-exp.
* ``_normalize_probabilities`` implements the Bayesian evidence-to-probability
  normalisation described in Giacalone et al. (2021, AJ, 161, 24,
  DOI ``10.3847/1538-3881/abc6af``).

Log weights/evidences and normalized probabilities are dimensionless. NaN log
weights raise; ``+inf`` and all-``-inf`` evidence are explicit anomalous or
degenerate results (``+inf`` evidence, an undefined ESS, an ``"anomaly"`` or
``"all_neginf"`` status) rather than clipped probabilities.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike
from scipy.special import logsumexp as _logsumexp


def _log_mean_exp(logw: ArrayLike, *, N_total: int) -> float:
    """Numerically stable log(mean(exp(logw))).

    Equivalent to log(Z) = log(mean(exp(lnL))) but avoids float underflow
    when lnL values are very negative (e.g., long light curves with many
    flux points).

    ``-inf`` entries (non-transiting draws) contribute zero weight to the
    sum but still count in the denominator, preserving the same mean
    semantics as ``np.mean(np.nan_to_num(np.exp(lnL + 600)))``.

    ``NaN`` entries are numerical failures and raise rather than becoming
    zero-weight draws or a finite-looking scenario result. ``+inf`` entries
    propagate so the scenario normalizer can report its explicit anomaly state.
    The effective-sample-size gate is evaluated separately, not by this reduction.

    Args:
        logw: 1-D array-like of log-weight values (e.g., lnL or lnL + lnprior).
        N_total: Total number of MC draws (must equal len(logw)).

    Returns:
        log(mean(exp(logw))) as a Python float. Returns ``-inf`` if all
        entries are non-positive-infinite, or ``+inf`` if any entry is
        positive-infinite.
    """
    logw = _log_weights(logw)
    if N_total != logw.size:
        raise ValueError(
            f"N_total ({N_total}) must equal len(logw) ({logw.size}). "
            "Passing len(lnL[finite]) instead of len(lnL) would silently "
            "overestimate evidence for scenarios with geometric exclusions."
        )
    if np.any(np.isposinf(logw)):
        return np.inf
    if not np.any(np.isfinite(logw)):
        return -np.inf
    # logsumexp gives the remaining -inf entries exactly zero weight, so no
    # filtered copy of a million-draw array is needed.
    return float(_logsumexp(logw) - np.log(N_total))


def _log_weight_ess(logw: ArrayLike) -> float:
    """Kish effective sample size of Monte Carlo weights, from their logs.

    ``ESS = (sum w)**2 / sum w**2`` (Kish 1965, *Survey Sampling*, Wiley,
    section 11.7), evaluated with max-shifted weights so it is defined for the same
    underflowing weights ``_log_mean_exp`` handles.  It counts how many draws
    actually carry the estimate: ``ESS == N`` when every draw contributes
    equally, and ``ESS == 1`` when one draw carries all of it, in which case
    ``_log_mean_exp`` has returned an extreme-value statistic rather than an
    integral.

    ``-inf`` entries (non-transiting or zero-prior draws) carry zero weight and
    contribute to neither sum. NaN entries raise, as in ``_log_mean_exp``.
    A ``+inf`` entry makes the weight ratio undefined, so the result is NaN:
    every ESS floor treats a non-finite ESS as unconverged, and the matching
    ``+inf`` evidence is reported as an anomaly by the normalizer.

    Args:
        logw: 1-D array-like of log-weight values (``lnL + lnprior``).

    Returns:
        The effective sample size as a Python float: ``0.0`` when every entry
        is ``-inf``, NaN when any entry is ``+inf``.
    """
    logw = _log_weights(logw)
    if np.any(np.isposinf(logw)):
        return math.nan
    if not np.any(np.isfinite(logw)):
        return 0.0
    # The common log scale cancels; subtract it before reductions can lose ESS.
    with np.errstate(over="ignore"):
        weights = np.exp(logw - np.max(logw))
    return float(np.sum(weights) ** 2 / np.sum(weights ** 2))


def _log_weights(logw: ArrayLike) -> np.ndarray:
    """Validated 1-D float64 log weights; NaN marks a numerical failure and raises."""
    logw = np.asarray(logw)
    if logw.dtype.kind not in "iuf":
        raise ValueError("log weights must contain real numbers, not boolean or text")
    logw = logw.astype(np.float64, copy=False)
    if logw.ndim != 1:
        raise ValueError("log weights must be a 1-D array")
    if np.any(np.isnan(logw)):
        raise ValueError(
            "TREX scenario produced a nonfinite log weight; numerical evidence "
            "cannot be reduced to a scenario probability"
        )
    return logw


# QUADRATURE_POLICY: integrate each linear panel of the two-box
# (phase-bin * exposure) trapezoid kernel with a five-point Gauss-Legendre
# rule; unequal box widths also have a plateau panel.
_TWO_BOX_PANEL_ORDER: int = 5
_PANEL_NODES, _PANEL_WEIGHTS = leggauss(_TWO_BOX_PANEL_ORDER)


def _gauss_legendre_panel(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Five-point Gauss-Legendre nodes and weights mapped onto [lo, hi]."""
    nodes, weights = _PANEL_NODES, _PANEL_WEIGHTS
    midpoint = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    return midpoint + half_width * nodes, half_width * weights


# Distinguish bools from equal numeric widths so cache hits cannot bypass validation.
@lru_cache(maxsize=64, typed=True)
def _two_box_convolution_quadrature(
    bin_integration_days: float, exposure_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for the exact convolution of the phase-bin and exposure boxes.

    Args:
        bin_integration_days: Phase-bin box width [days], finite and positive.
        exposure_days: Detector exposure box width [days], finite and positive.

    Returns:
        ``(offsets, weights)``: read-only node offsets [days] relative to the
        bin centre, spanning the full trapezoid support, and their weights,
        which sum to one.

    The observed folded value at a bin centre is the true light curve averaged
    over the phase bin and over the detector exposure. Both averages are
    uniform boxes, and their convolution is a trapezoid (a triangle when the
    widths match) on ``[-(w_bin + w_exp) / 2, +(w_bin + w_exp) / 2]`` with a
    plateau of half width ``|w_bin - w_exp| / 2`` and linear ramps elsewhere.

    The trapezoid is used exactly rather than supersampled: each of its (at
    most three) linear panels is integrated with a Gauss-Legendre rule against
    the instantaneous forward model. This avoids ``nsamples**2`` nested
    midpoint evaluations (400 at the production default), and constant widths
    let the rule be cached.
    """
    _require_positive_width(bin_integration_days, "bin_integration_days")
    _require_positive_width(exposure_days, "exposure_days")

    half_span = 0.5 * (bin_integration_days + exposure_days)
    plateau_half_width = 0.5 * abs(bin_integration_days - exposure_days)
    # The convolution density is the overlap length divided by the product of the box widths.
    kernel_norm = 1.0 / (bin_integration_days * exposure_days)

    offsets = []
    weights = []
    if plateau_half_width > 0.0:
        nodes, gl_weights = _gauss_legendre_panel(-plateau_half_width, plateau_half_width)
        offsets.append(nodes)
        weights.append(gl_weights / max(bin_integration_days, exposure_days))
    nodes, gl_weights = _gauss_legendre_panel(-half_span, -plateau_half_width)
    offsets.append(nodes)
    weights.append(gl_weights * (nodes + half_span) * kernel_norm)
    nodes, gl_weights = _gauss_legendre_panel(plateau_half_width, half_span)
    offsets.append(nodes)
    weights.append(gl_weights * (half_span - nodes) * kernel_norm)

    offsets_array = np.concatenate(offsets)
    weights_array = np.concatenate(weights)
    # Renormalize so the discrete kernel conserves flux exactly: the panel sum
    # is one analytically, and this removes the accumulated rounding so the
    # weights sum to one at machine precision before they are cached and shared.
    weights_array = weights_array / np.sum(weights_array)
    offsets_array.setflags(write=False)
    weights_array.setflags(write=False)
    return offsets_array, weights_array


def _require_positive_width(value, name: str) -> None:
    """Refuse a box width that is boolean, non-real, non-scalar, non-finite or not positive."""
    if np.ndim(value) != 0 or np.asarray(value).dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real number of days")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def phase_bin_widths(value: ArrayLike, n_points: int) -> float | np.ndarray:
    """Validate a phase-bin width: one scalar for every point, or one width per point.

    Args:
        value: Phase-bin width [days]; a real scalar, or a 1-D real array with
            one entry per time point. Booleans and strings are refused.
        n_points: Number of time points the widths apply to.

    Returns:
        A scalar keeps its historical meaning (``0`` selects exposure-only
        supersampling) and its own float type; an integer becomes a float.
        Per-point widths are returned as a float64 array and must be finite
        and positive; they let the broad baseline bins integrate the model
        over their own span, where an EB secondary eclipse can fall, instead
        of over the transit-bin width.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bin_integration_days must be finite and non-negative")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("bin_integration_days must be real numbers of days")
    array = array.astype(float, copy=False)
    if array.ndim == 0:
        width = float(array)
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("bin_integration_days must be finite and non-negative")
        # Return a numeric scalar unchanged so its historical arithmetic (e.g. float32) is kept.
        return value if isinstance(value, (float, np.floating)) else width
    if array.shape != (n_points,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("per-bin bin_integration_days must be finite, positive and match the time array")
    return array


def _per_bin_two_box_quadrature(widths: ArrayLike, exposure_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise two-box quadrature for per-bin widths, zero-padded to one node count.

    ``widths`` is a non-empty 1-D array-like of finite positive real widths
    [days]; booleans and strings are refused, as in :func:`phase_bin_widths`.
    Each row of the returned ``(n_bins, n_nodes)`` offsets and weights is the
    exact :func:`_two_box_convolution_quadrature` rule of its own bin; padding
    nodes sit at the bin centre with zero weight, so every row still
    integrates only its own trapezoid.
    """
    _require_positive_width(exposure_days, "exposure_days")
    if np.ndim(widths) != 1 or np.size(widths) == 0:
        raise ValueError("per-bin bin_integration_days must be a non-empty 1-D array")
    widths = phase_bin_widths(widths, np.size(widths))
    unique, inverse = np.unique(widths, return_inverse=True)
    rules = [_two_box_convolution_quadrature(float(width), float(exposure_days)) for width in unique]
    size = max(nodes.size for nodes, _ in rules)
    offsets, weights = np.zeros((unique.size, size)), np.zeros((unique.size, size))
    for row, (nodes, node_weights) in enumerate(rules):
        offsets[row, :nodes.size], weights[row, :nodes.size] = nodes, node_weights
    return offsets[inverse], weights[inverse]


def _normalize_probabilities(lnZ: ArrayLike) -> tuple[np.ndarray, str]:
    """Normalize scenario log-evidences to a probability vector.

    Args:
        lnZ: Non-empty 1-D array-like of per-scenario log-evidences; any other
            shape raises ``ValueError``.

    Returns:
        (probs, status) where:
            probs  — normalized probability vector (sums to 1 in the 'ok'
                     case; all-zero for the degenerate cases).
            status — one of:
                'ok'          finite shifted-weight normalization
                'all_neginf'  every scenario has zero sampled evidence
                'anomaly'     NaN or +inf present; distinct failure mode
    """
    lnZ = np.asarray(lnZ)
    if lnZ.dtype.kind not in "iuf":
        raise ValueError("scenario log-evidences must contain real numbers, not boolean or text")
    lnZ = lnZ.astype(np.float64, copy=False)
    if lnZ.ndim != 1 or lnZ.size == 0:
        raise ValueError("scenario log-evidences must be a non-empty 1-D array")
    if np.any(np.isnan(lnZ)) or np.any(np.isposinf(lnZ)):
        return np.zeros(len(lnZ), dtype=float), "anomaly"
    if np.all(np.isneginf(lnZ)):
        return np.zeros(len(lnZ), dtype=float), "all_neginf"
    # Restoring a huge common offset would round away the log-normalizer correction.
    with np.errstate(over="ignore"):
        weights = np.exp(lnZ - np.max(lnZ))
    return weights / np.sum(weights), "ok"


__all__ = [
    "_log_mean_exp",
    "_log_weight_ess",
    "_two_box_convolution_quadrature",
    "_per_bin_two_box_quadrature",
    "phase_bin_widths",
    "_normalize_probabilities",
]
