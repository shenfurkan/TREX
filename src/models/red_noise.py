"""Portable Pont, Zucker & Queloz (2006) photometry-noise diagnostic.

This module implements the time-binned ``beta`` red-noise factor introduced by
Pont, Zucker & Queloz (2006), MNRAS 373, 231,
arXiv:astro-ph/0608597, DOI ``10.1111/j.1365-2966.2006.11012.x``. For a set of
out-of-transit residuals :math:`r_i` with unbinned scatter :math:`\\sigma_1`, the
residuals are averaged into bins of :math:`N` consecutive points and the scatter
of those bin means :math:`\\sigma_N` is compared with the white-noise
expectation :math:`\\sigma_1 / \\sqrt{N}`:

.. math::

    \\beta = \\frac{\\sigma_N}{\\sigma_1 / \\sqrt{N}}.

For white noise :math:`\\beta \\approx 1`; correlated (red) noise inflates the
binned scatter and pushes :math:`\\beta > 1`. Both scatters use sample standard
deviations (ddof=1); no autocovariance estimator is used.

Scientific boundary
-------------------
``beta`` is a **descriptive** noise diagnostic.  It is not a likelihood, a
detection significance, or a validation probability.  Three rules are enforced:

* A value ``beta < 1`` must never be used to shrink a reported uncertainty; the
  adopted factor is floored at :math:`1.0` and the raw value is retained
  separately.
* When a validated correlated-noise (GP/wavelet) likelihood is active, ``beta``
  must not be applied on top of it (no double inflation).  The diagnostic may
  still be reported descriptively with ``inflation_applied: false``.
* When the out-of-transit residuals, cadence, or transit-duration scale cannot
  support a characterized estimate, the result is a durable ``deferred`` record
  rather than a fabricated ``beta``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np


RED_NOISE_SCHEMA_VERSION = 2
KIND = "pont-2006-correlated-noise"

PONT_ETAL_2006: dict[str, str] = {
    "citation": "Pont, Zucker & Queloz (2006)",
    "title": "The effect of red noise on planetary transit detection",
    "journal": "MNRAS 373, 231",
    "bibcode": "2006MNRAS.373..231P",
    "arxiv": "astro-ph/0608597",
    "doi": "10.1111/j.1365-2966.2006.11012.x",
}

# OPERATIONAL_POLICY: minimum evidence required before a beta factor is
# characterized. These are estimator-stability bounds, not physical constants.
MIN_OUT_OF_TRANSIT_SAMPLES = 20
MIN_BIN_COUNT = 4
MIN_BIN_POINTS = 2
# OPERATIONAL_POLICY: a time gap larger than this many cadences breaks a
# contiguous segment so a bin never spans a data gap.
GAP_TOLERANCE_CADENCES = Fraction(3, 1)
# OPERATIONAL_POLICY: transit exclusion window half-width as a multiple of the
# candidate-derived transit duration T14. Points within +-T14/2 are in transit.
TRANSIT_EXCLUSION_HALF_DURATION_FACTOR = Fraction(1, 1)


@dataclass(frozen=True)
class RedNoiseScale:
    """One binned-RMS estimate, with scatter in the input residuals' units.

    ``bin_size_days`` is the nominal point-count times supplied cadence, not
    an elapsed-time measurement for an irregularly sampled bin. The unbinned
    scatter uses every input sample; ``dropped_sample_count`` counts segment
    tails excluded only from the complete-bin means. No residuals are retained.
    """

    bin_size_points: int
    bin_size_days: float
    bin_count: int
    sample_count: int
    dropped_sample_count: int
    binned_rms: float
    unbinned_rms: float
    white_noise_expectation: float
    beta_raw: float
    beta: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable summary; scatter units remain owned by the caller."""
        return asdict(self)


def out_of_transit_mask(
    time_btjd: Sequence[float],
    *,
    period_days: float,
    epoch_btjd: float,
    duration_days: float,
    exclusion_factor: Real = TRANSIT_EXCLUSION_HALF_DURATION_FACTOR,
) -> np.ndarray:
    """Return the boolean out-of-transit mask for a candidate ephemeris.

    The orbital phase is folded with the candidate period and epoch, and every
    cadence whose absolute time from the nearest mid-transit is within
    ``0.5 * exclusion_factor * T14`` is excluded. The rule is defined here so the
    selection is explicit and reproducible from candidate-derived quantities.
    Time/epoch are BTJD_TDB days; T14 is positive and shorter than P/2.
    All inputs must be finite and real (not boolean); the factor is positive.
    Invalid inputs raise ``ValueError``.
    """
    time = np.asarray(time_btjd)
    if (time.dtype.kind not in "iuf" or time.ndim != 1 or not time.size
            or not np.all(np.isfinite(time))):
        raise ValueError("time_btjd must be a non-empty finite real one-dimensional array")
    for value in (period_days, epoch_btjd, duration_days, exclusion_factor):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("ephemeris and exclusion factor must be finite real scalars")
    time = time.astype(float, copy=False)
    period = float(period_days)
    duration = float(duration_days)
    if not 0 < duration < period / 2 or exclusion_factor <= 0:
        raise ValueError("duration must be positive and shorter than P/2; exclusion factor must be positive")
    phase = ((time - float(epoch_btjd) + 0.5 * period) % period) - 0.5 * period
    half_window = 0.5 * duration * float(exclusion_factor)
    return np.abs(phase) > half_window


def median_cadence_days(time_btjd: Sequence[float]) -> float | None:
    """Median positive interval [days]; None for invalid/insufficient real times."""
    time = np.asarray(time_btjd)
    if time.dtype.kind not in "iuf" or time.ndim != 1:
        return None
    time = np.sort(time.astype(float, copy=False))
    if time.size < 2 or not np.all(np.isfinite(time)):
        return None
    differences = np.diff(time)
    positive = differences[differences > 0.0]
    if positive.size == 0:
        return None
    cadence = float(np.median(positive))
    if not math.isfinite(cadence) or cadence <= 0.0:
        return None
    return cadence


def transit_bin_size_points(duration_days: float, cadence_days: float) -> int:
    """Nearest point count to T14/cadence, rounding exact ties to even.

    Inputs are positive finite real days, not booleans. No minimum is imposed:
    a result below two cannot support a binned-RMS estimate. Callers must defer
    rather than silently widen the requested timescale.
    """
    for value in (duration_days, cadence_days):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("duration_days and cadence_days must be finite real scalars")
    if duration_days <= 0.0 or cadence_days <= 0.0:
        raise ValueError("duration_days and cadence_days must be positive")
    ratio = duration_days / cadence_days
    if not math.isfinite(ratio):
        raise ValueError("duration-to-cadence ratio must be finite")
    return int(round(ratio))


def _contiguous_segments(
    time: np.ndarray, *, gap_tolerance_days: float
) -> list[np.ndarray]:
    """Split sorted time indices into contiguous segments separated by gaps."""
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    breaks = np.nonzero(np.diff(sorted_time) > gap_tolerance_days)[0]
    return [segment for segment in np.split(order, breaks + 1) if segment.size > 0]


def _bin_means(
    residuals: np.ndarray, segments: Sequence[np.ndarray], bin_size: int
) -> tuple[np.ndarray, int]:
    """Return complete-bin means and the count of dropped leftover points."""
    means = []
    dropped = 0
    for segment in segments:
        count = segment.size
        complete = count // bin_size
        dropped += count - complete * bin_size
        if complete:
            means.append(residuals[segment[:complete * bin_size]].reshape(complete, bin_size).mean(axis=1))
    return (np.concatenate(means) if means else np.empty(0)), dropped


def _prepare_beta_inputs(residuals, time_btjd, cadence_days, gap_tolerance_cadences):
    residual_array, time_array = np.asarray(residuals), np.asarray(time_btjd)
    if (residual_array.dtype.kind not in "iuf" or time_array.dtype.kind not in "iuf"
            or residual_array.shape != time_array.shape or residual_array.ndim != 1
            or not np.all(np.isfinite(residual_array)) or not np.all(np.isfinite(time_array))):
        raise ValueError("residuals and time_btjd must be aligned finite real one-dimensional arrays")
    for value in (cadence_days, gap_tolerance_cadences):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("cadence and gap tolerance must be finite positive real scalars")
    residual_array = residual_array.astype(float, copy=False)
    time_array = time_array.astype(float, copy=False)
    gap_days = float(gap_tolerance_cadences) * cadence_days
    if not math.isfinite(gap_days):
        raise ValueError("gap tolerance in days must be finite")
    segments = _contiguous_segments(time_array, gap_tolerance_days=gap_days)
    return residual_array, segments


def _validate_bin_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("bin point counts must be nonnegative integers, not booleans")


def pont_2006_beta(
    residuals: Sequence[float],
    time_btjd: Sequence[float],
    *,
    bin_size_points: int,
    cadence_days: float,
    gap_tolerance_cadences: Real = GAP_TOLERANCE_CADENCES,
) -> RedNoiseScale | None:
    """Compute one Pont et al. (2006) beta factor for the requested bin size.

    Returns ``None`` when the evidence cannot support a characterized estimate
    (too few samples, bins, or points).  The adopted ``beta`` is floored at
    ``1.0``; ``beta_raw`` retains the unfloored estimator. Scatter uses the
    residuals' units (fractional flux, ppm, or standardized dimensionless z);
    beta is invariant under a common nonzero rescaling. Nonfinite observations,
    invalid cadence/gap tolerance and non-integer bin counts raise ValueError.
    Complete nonoverlapping equal-point bins approximate duration-scale averaging
    at the supplied cadence; this is not Pont Sect. 2.6's sliding-window V(n)
    estimator for irregular sampling. No bin is formed across a declared gap.
    """
    _validate_bin_count(bin_size_points)
    residual_array, segments = _prepare_beta_inputs(residuals, time_btjd, cadence_days, gap_tolerance_cadences)
    return _pont_scale(residual_array, segments, int(bin_size_points), cadence_days)


def _pont_scale(residual_array, segments, bin_size_points, cadence_days):
    if bin_size_points < MIN_BIN_POINTS:
        return None
    if residual_array.size < MIN_OUT_OF_TRANSIT_SAMPLES:
        return None
    bin_means, dropped = _bin_means(residual_array, segments, bin_size_points)
    if bin_means.size < MIN_BIN_COUNT:
        return None

    unbinned_rms = float(np.std(residual_array, ddof=1))
    binned_rms = float(np.std(bin_means, ddof=1))
    if not (math.isfinite(unbinned_rms) and math.isfinite(binned_rms)) or unbinned_rms <= 0.0:
        return None
    white_expectation = unbinned_rms / math.sqrt(bin_size_points)
    beta_raw = binned_rms / white_expectation
    if not math.isfinite(beta_raw):
        return None
    beta = max(1.0, beta_raw)
    return RedNoiseScale(
        bin_size_points=int(bin_size_points),
        bin_size_days=float(bin_size_points) * cadence_days,
        bin_count=int(bin_means.size),
        sample_count=int(residual_array.size),
        dropped_sample_count=int(dropped),
        binned_rms=binned_rms,
        unbinned_rms=unbinned_rms,
        white_noise_expectation=white_expectation,
        beta_raw=beta_raw,
        beta=beta,
    )


WINN_ETAL_2008 = {
    "citation": "Winn et al. (2008), ApJ 683, 1076, Sect. 3 (time-averaging method)",
    "bibcode": "2008ApJ...683.1076W",
    "doi": "10.1086/589737",
}


def winn_2008_multiscale_beta(
    residuals: Sequence[float],
    time_btjd: Sequence[float],
    *,
    minimum_bin_points: int,
    maximum_bin_points: int,
    cadence_days: float,
    gap_tolerance_cadences: Real = GAP_TOLERANCE_CADENCES,
) -> dict[str, Any] | None:
    """Return the median beta over a range of averaging times (Winn et al. 2008).

    Winn et al. (2008, Sect. 3) found beta to depend weakly on the averaging
    time and adopted the median over averaging times spanning the ingress or
    egress duration. Each per-scale factor is :func:`pont_2006_beta`, whose
    unbiased (``ddof=1``) binned RMS already has the white-noise expectation
    ``sigma_1 / sqrt(N)``; Winn's ``sqrt(M / (M - 1))`` factor corrects their
    zero-mean RMS estimator and would double-count here, so it is not applied.
    Returns ``None`` when fewer than two scales yield a characterized factor.
    Sorting/segmentation is shared by all scales; requested integer bounds
    are intersected with point counts capable of producing MIN_BIN_COUNT bins.
    """
    _validate_bin_count(minimum_bin_points)
    _validate_bin_count(maximum_bin_points)
    if maximum_bin_points < minimum_bin_points:
        raise ValueError("maximum bin points must not be below minimum bin points")
    residual_array, segments = _prepare_beta_inputs(residuals, time_btjd, cadence_days, gap_tolerance_cadences)
    lower = max(MIN_BIN_POINTS, int(minimum_bin_points))
    upper = min(int(maximum_bin_points), residual_array.size // MIN_BIN_COUNT)
    scales = []
    for bin_points in range(lower, upper + 1):
        scale = _pont_scale(residual_array, segments, bin_points, cadence_days)
        if scale is not None:
            scales.append(scale)
    if len(scales) < 2:
        return None
    raw = [float(scale.beta_raw) for scale in scales]
    beta_raw = float(np.median(raw))
    return {
        "beta_raw": beta_raw,
        "beta": max(1.0, beta_raw),
        "bin_size_points": [int(scale.bin_size_points) for scale in scales],
        "bin_size_days": [float(scale.bin_size_days) for scale in scales],
        "per_scale_beta_raw": raw,
        "reference": dict(WINN_ETAL_2008),
    }


def _deferred(reason: str) -> dict[str, Any]:
    return {
        "schema_version": RED_NOISE_SCHEMA_VERSION,
        "kind": KIND,
        "status": "deferred",
        "reason": reason,
        "beta": None,
        "beta_raw": None,
        "inflation_applied": False,
        "claim_eligible": False,
        "validation_eligible": False,
        "reference": dict(PONT_ETAL_2006),
    }


def compute_red_noise_diagnostic(
    time_btjd: Sequence[float],
    residuals: Sequence[float],
    *,
    period_days: float,
    epoch_btjd: float,
    duration_days: float,
    correlated_noise_model_active: bool = False,
    uncertainty_inflation_already_applied: bool = False,
    gap_tolerance_cadences: Real = GAP_TOLERANCE_CADENCES,
    ingress_duration_days: float | None = None,
) -> dict[str, Any]:
    """Characterize Pont et al. (2006) red noise in out-of-transit residuals.

    When the caller knows the ingress duration, the adopted ``beta`` is the
    Winn et al. (2008) median over averaging times from the ingress duration
    to the transit duration; the single transit-scale Pont factor
    is still reported. Without it, the single-scale factor is adopted.

    The caller supplies residuals already evaluated against the adopted transit
    model.  The bin size is derived from the candidate transit duration and the
    measured cadence, never from a hardcoded exposure or duration.
    Residual scatter retains the caller's units. This descriptive result does
    not validate a planet or calibrate false-positive probabilities. Invalid
    ephemeris/scale evidence returns deferred; non-boolean policy switches raise.
    """
    if not isinstance(correlated_noise_model_active, (bool, np.bool_)) or not isinstance(uncertainty_inflation_already_applied, (bool, np.bool_)):
        raise ValueError("noise-model and prior-inflation switches must be boolean")
    time_array = np.asarray(time_btjd)
    residual_array = np.asarray(residuals)
    if time_array.ndim != 1 or residual_array.ndim != 1 or time_array.shape != residual_array.shape:
        raise ValueError("time_btjd and residuals must be aligned one-dimensional arrays")
    if time_array.dtype.kind not in "iuf" or residual_array.dtype.kind not in "iuf":
        return _deferred("red-noise-unavailable: time and residuals must be real, not boolean or text")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(residual_array)):
        return _deferred("red-noise-unavailable: non-finite time or residual values")
    time_array = time_array.astype(float, copy=False)
    residual_array = residual_array.astype(float, copy=False)
    if ingress_duration_days is not None:
        if (isinstance(ingress_duration_days, (bool, np.bool_)) or not isinstance(ingress_duration_days, Real)
                or not math.isfinite(ingress_duration_days) or ingress_duration_days <= 0):
            return _deferred("red-noise-unavailable: ingress duration must be a finite positive real scalar")

    cadence_days = median_cadence_days(time_array)
    if cadence_days is None:
        return _deferred("red-noise-unavailable: cadence could not be determined")

    try:
        mask = out_of_transit_mask(
            time_array,
            period_days=period_days,
            epoch_btjd=epoch_btjd,
            duration_days=duration_days,
        )
        if ingress_duration_days is not None and ingress_duration_days > duration_days:
            return _deferred("red-noise-unavailable: ingress duration exceeds transit duration")
    except ValueError as exc:
        return _deferred("red-noise-unavailable: {0}".format(exc))
    out_time = time_array[mask]
    out_residuals = residual_array[mask]
    if out_residuals.size < MIN_OUT_OF_TRANSIT_SAMPLES:
        return _deferred(
            "red-noise-unavailable: fewer than {0} out-of-transit residuals".format(
                MIN_OUT_OF_TRANSIT_SAMPLES
            )
        )

    if duration_days < MIN_BIN_POINTS * cadence_days:
        return _deferred("red-noise-unavailable: transit duration spans fewer than two cadences")

    try:
        bin_size_points = transit_bin_size_points(duration_days, cadence_days)
        scale = pont_2006_beta(
            out_residuals,
            out_time,
            bin_size_points=bin_size_points,
            cadence_days=cadence_days,
            gap_tolerance_cadences=gap_tolerance_cadences,
        )
    except ValueError as exc:
        return _deferred(f"red-noise-unavailable: {exc}")
    if scale is None:
        return _deferred(
            "red-noise-unavailable: incomplete transit-scale bins for the binned-RMS estimator"
        )

    multiscale: dict[str, Any] | None = None
    if ingress_duration_days is not None:
        multiscale = winn_2008_multiscale_beta(
            out_residuals,
            out_time,
            minimum_bin_points=transit_bin_size_points(ingress_duration_days, cadence_days),
            maximum_bin_points=bin_size_points,
            cadence_days=cadence_days,
            gap_tolerance_cadences=gap_tolerance_cadences,
        )
    adopted_beta = float(multiscale["beta"]) if multiscale is not None else float(scale.beta)
    adopted_beta_raw = float(multiscale["beta_raw"]) if multiscale is not None else float(scale.beta_raw)
    if correlated_noise_model_active:
        inflation_applied = False
        inflation_reason = "correlated-noise-model-active: beta is not applied on top of a validated correlated-noise likelihood"
    elif uncertainty_inflation_already_applied:
        inflation_applied = False
        inflation_reason = "uncertainty-inflation-already-applied: beta is not applied twice"
    else:
        inflation_applied = True
        inflation_reason = "white-noise-likelihood: beta is available for a single conservative inflation"

    record = _deferred(inflation_reason)
    record.update({
        "status": "characterized",
        "beta": adopted_beta,
        "beta_raw": adopted_beta_raw,
        "beta_floored_to_unity": bool(adopted_beta_raw < 1.0),
        "beta_estimator": "winn-2008-multiscale-median" if multiscale is not None else "pont-2006-transit-scale",
        "multiscale": multiscale,
        "transit_scale_beta": float(scale.beta),
        "inflation_applied": bool(inflation_applied),
        "estimator": {
            "method": "Pont et al. (2006) time-binned residual RMS",
            "bin_size_source": "candidate transit duration T14 divided by measured median cadence",
            "bin_size_points": int(scale.bin_size_points),
            "bin_size_days": float(scale.bin_size_days),
            "bin_count": int(scale.bin_count),
            "sample_count": int(scale.sample_count),
            "gap_tolerance_cadences": float(gap_tolerance_cadences),
            "scatter_units": "same as input residuals",
            "bin_time_scale": "nominal bin point count times supplied cadence",
            "dropped_sample_count": int(scale.dropped_sample_count),
            "binned_rms": float(scale.binned_rms),
            "unbinned_rms": float(scale.unbinned_rms),
            "white_noise_expectation": float(scale.white_noise_expectation),
        },
        "selection": {
            "rule": "exclude |phase to nearest mid-transit| <= 0.5 * T14",
            "out_of_transit_sample_count": int(out_residuals.size),
            "in_transit_sample_count": int(time_array.size - out_residuals.size),
            "cadence_days": float(cadence_days),
        },
    })
    return record


__all__ = [
    "GAP_TOLERANCE_CADENCES",
    "KIND",
    "MIN_BIN_COUNT",
    "MIN_BIN_POINTS",
    "MIN_OUT_OF_TRANSIT_SAMPLES",
    "PONT_ETAL_2006",
    "RED_NOISE_SCHEMA_VERSION",
    "RedNoiseScale",
    "TRANSIT_EXCLUSION_HALF_DURATION_FACTOR",
    "WINN_ETAL_2008",
    "compute_red_noise_diagnostic",
    "winn_2008_multiscale_beta",
    "median_cadence_days",
    "out_of_transit_mask",
    "pont_2006_beta",
    "transit_bin_size_points",
]
