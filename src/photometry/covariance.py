"""Descriptive OOT shifted-fold covariance at the actual transit-bin resolution.

The sample covariance identity, not a fitted noise likelihood, is used here.
Disjoint phase windows prevent cadence reuse, but do not prove independent noise
realizations. No significance threshold, matrix inversion or error adoption.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from trex.foundation.errors import NumericalLikelihoodError

from .binning import weighted_phase_bins

# Bound the optional JSON diagnostic, not the observations or scientific draw budget.
MAX_COVARIANCE_WINDOW_CELLS = 100_000


def _event_ephemeris(event) -> tuple[float, float, float]:
    keys = ("period_days", "epoch_btjd", "duration_days")
    if not isinstance(event, Mapping) or any(key not in event for key in keys):
        raise ValueError("known covariance-exclusion event requires a finite BTJD_TDB ephemeris")
    values = tuple(event[key] for key in keys)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("known covariance-exclusion event requires a finite BTJD_TDB ephemeris")
    period, epoch, duration = map(float, values)
    if not 0 < duration < period:
        raise ValueError("known covariance-exclusion event requires a finite BTJD_TDB ephemeris")
    return period, epoch, duration


def _covariance_summary(samples: np.ndarray, mean_modes, envelope: float) -> dict:
    """Ordinary ddof=1 covariance summaries of complete standardized windows."""
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        means = samples.mean(axis=0)
        centred = samples - means
        # C_ab=sum_m((z_ma-zbar_a)*(z_mb-zbar_b))/(M-1) [dimensionless].
        variance = np.sum(centred**2, axis=0) / (len(samples) - 1)
        adjacent = np.sum(centred[:, :-1] * centred[:, 1:], axis=0) / (len(samples) - 1)
        correlations = np.full_like(adjacent, np.nan)
        positive = (variance[:-1] > 0) & (variance[1:] > 0)
        correlations[positive] = adjacent[positive] / np.sqrt(variance[:-1][positive]) / np.sqrt(variance[1:][positive])
        # Var(mean_b(z_b)) includes ALL covariance lags without a dense B×B matrix.
        mean_variance = float(np.var(samples.mean(axis=1), ddof=1))
        diagonal_mean_variance = float(np.sum(variance) / samples.shape[1]**2)
        mean_diagonal_ratio = float(np.mean(variance) / envelope / envelope)
        mean_ratio = mean_variance / envelope / envelope * samples.shape[1]
        offdiagonal = mean_variance - diagonal_mean_variance
        weighted_ratio = float(np.var(mean_modes, ddof=1) / envelope / envelope)
    if (not all(np.isfinite(v).all() for v in (means, variance, adjacent, correlations[positive]))
            or not all(math.isfinite(v) for v in (mean_diagonal_ratio, mean_ratio, offdiagonal, weighted_ratio))):
        raise FloatingPointError("folded covariance summary is nonfinite")
    return dict(standardized_bin_means=means.tolist(), standardized_bin_variances=variance.tolist(),
                adjacent_standardized_covariances=adjacent.tolist(),
                adjacent_correlations=[float(v) if np.isfinite(v) else None for v in correlations],
                mean_diagonal_variance_over_envelope=mean_diagonal_ratio,
                standardized_mean_variance_over_envelope=mean_ratio,
                standardized_mean_offdiagonal_variance=offdiagonal,
                weighted_transit_mean_variance_over_envelope=weighted_ratio)


def folded_covariance(time_btjd, flux, sigma, *, period_days, epoch_btjd,
                      duration_days, exposure_days, transit_bin_count,
                      adopted_uncertainty_factor, known_events=()) -> dict:
    """Measure adjacent-bin covariance on disjoint shifted T14 folds.

    Native time/epoch are BTJD_TDB days; P/T14/exposure are days, flux/sigma are
    relative aperture-total light. Covariances are dimensionless after dividing
    each bin by its propagated reported sigma, before the shared error envelope.
    Primary/half-phase and all supplied known BTJD_TDB events are excluded with
    half an exposure of padding. The caller owns known-event completeness.

    ``windows`` retains all phase centres, per-bin counts and standardized
    residuals (None for missing bins); only complete windows enter covariance.
    ``selected_cadence_count`` includes incomplete windows; ``unique_used_cadence_count``
    counts only rows in the complete windows entering the estimate. Exclusion,
    reuse, complete-window and rank counts describe this selection, not independent
    realizations. Covariance fields are ordinary ddof=1 diagonal/adjacent summaries
    and transit-mean variance ratios against the supplied error envelope.

    Boolean/non-real inputs and malformed events raise ValueError. Excessive
    window×bin output (>MAX_COVARIANCE_WINDOW_CELLS), unresolved float64 geometry,
    insufficient coverage or unrepresentable numerical results return deferred;
    no windows or observations are silently truncated. Returned records are finite
    JSON data, with None only for unavailable measurements.
    """
    arrays = [np.asarray(item) for item in (time_btjd, flux, sigma)]
    if any(item.dtype.kind not in "iuf" for item in arrays):
        raise ValueError("folded covariance requires real, nonboolean photometry")
    time, values, errors = (item.astype(float, copy=False) for item in arrays)
    if (time.ndim != 1 or not time.size or values.shape != time.shape or errors.shape != time.shape
            or not np.all(np.isfinite(time) & np.isfinite(values) & np.isfinite(errors) & (errors > 0))):
        raise ValueError("folded covariance requires aligned finite photometry and positive errors")
    controls = (period_days, epoch_btjd, duration_days, exposure_days, adopted_uncertainty_factor)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) for v in controls)
            or not 0 < duration_days < period_days / 2 or exposure_days <= 0
            or adopted_uncertainty_factor < 1 or isinstance(transit_bin_count, (bool, np.bool_))
            or not isinstance(transit_bin_count, Integral) or transit_bin_count < 1):
        raise ValueError("invalid folded covariance ephemeris, bin count or shared error factor")
    period_days, epoch_btjd, duration_days, exposure_days, adopted_uncertainty_factor = map(float, controls)
    transit_bin_count = int(transit_bin_count)
    if isinstance(known_events, (Mapping, str, bytes)) or known_events is None:
        raise ValueError("known covariance-exclusion events must be an iterable of ephemeris mappings")
    try:
        events = [_event_ephemeris(event) for event in known_events]
    except TypeError as exc:
        raise ValueError("known covariance-exclusion events must be an iterable of ephemeris mappings") from exc
    exclusions = [dict(period_days=period_days, epoch_btjd=epoch_btjd, duration_days=duration_days,
                       role="refined-primary"),
                   dict(period_days=period_days, epoch_btjd=epoch_btjd+period_days/2,
                        duration_days=duration_days, role="half-phase-control")]
    for period, epoch, duration in events:
        exclusions.append(dict(period_days=period, epoch_btjd=epoch, duration_days=duration,
                               role="declared-known-event"))
    record = dict(contract="oot-disjoint-shifted-fold-covariance-v1", status="deferred",
        period_days=period_days, epoch_btjd=epoch_btjd, duration_days=duration_days,
        exposure_days=exposure_days, covariance_bin_count=transit_bin_count,
        bin_width_days=None, phase_spacing_days=None, expected_window_count=None,
        complete_window_count=0, windows=[], excluded_cadence_count=None, exclusions=exclusions,
        selected_cadence_count=0, unique_used_cadence_count=0, cadence_reuse_count=0,
        independent_realization_count=None, adopted_uncertainty_factor=adopted_uncertainty_factor,
        sample_covariance_rank_upper_bound=0,
        covariance_units="dimensionless: bin residual / propagated reported sigma, before envelope",
        limits=("Descriptive stationary-OOT transfer only. Disjoint cadences do not imply independent "
                "windows/events; long correlations and estimated trends can couple them. No p-value, "
                "confidence interval, covariance inverse, full-likelihood adequacy or FPP calibration. "
                 "Incomplete windows remain explicit; covariance uses complete windows only. "
                 "Only T14-wide transit-resolution bins, not broad baseline/other EB durations, are assessed."))
    if not math.isfinite(exclusions[1]["epoch_btjd"]):
        raise ValueError("half-phase covariance-exclusion epoch cannot be represented finitely")
    ratio = period_days / duration_days
    if not math.isfinite(ratio):
        record["reason"] = "folded covariance window geometry cannot be represented finitely"
        return record
    # Decimal integer ratios may round just below an integer in binary64.
    nearest = round(ratio)
    window_count = nearest if math.isclose(ratio, nearest, rel_tol=4 * np.finfo(float).eps) else math.floor(ratio)
    record["expected_window_count"] = window_count
    if window_count * transit_bin_count > MAX_COVARIANCE_WINDOW_CELLS:
        record.update(reason="folded covariance output exceeds the window-bin cell budget",
                      window_bin_cell_budget=MAX_COVARIANCE_WINDOW_CELLS)
        return record
    spacing = period_days / window_count
    edges = np.linspace(-duration_days/2, duration_days/2, transit_bin_count+1)
    if np.any(edges[1:] <= edges[:-1]) or duration_days / transit_bin_count <= 0:
        record["reason"] = "folded covariance bin geometry is unresolved in float64"
        return record
    record.update(bin_width_days=duration_days/transit_bin_count, phase_spacing_days=spacing)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            phase = (time - epoch_btjd + period_days/2) % period_days - period_days/2
            half_window = duration_days/2 + exposure_days/2
            oot = (abs(phase) > half_window) & (period_days/2 - abs(phase) > half_window)
            for period, epoch, duration in events:
                event_phase = (time - epoch + period/2) % period - period/2
                oot &= abs(event_phase) > duration/2 + exposure_days/2
            centres = (np.arange(window_count) + .5) * spacing - period_days/2
            # One half-open spacing cell per cadence also resolves rounded touching edges.
            membership = np.clip(np.floor((phase + period_days/2) / spacing).astype(int), 0, window_count - 1)
            local_phase = (phase - centres[membership] + period_days/2) % period_days - period_days/2
    except FloatingPointError:
        record["reason"] = "folded covariance phase geometry cannot be represented finitely"
        return record
    record["excluded_cadence_count"] = int(np.count_nonzero(~oot))
    selected = np.flatnonzero(oot & (local_phase >= edges[0]) & (local_phase < edges[-1]))
    selected = selected[np.argsort(membership[selected], kind="stable")]
    offsets = np.searchsorted(membership[selected], np.arange(window_count + 1))
    record["selected_cadence_count"] = int(selected.size)
    complete, mean_modes = [], []
    failed_windows = 0
    for index, centre in enumerate(centres):
        members = selected[offsets[index]:offsets[index + 1]]
        try:
            bins = weighted_phase_bins(local_phase[members], values[members], errors[members], edges)
        except NumericalLikelihoodError:
            record["windows"].append(dict(
                centre_phase_days=float(centre), complete=False,
                bin_counts=[None] * transit_bin_count, standardized_residuals=[None] * transit_bin_count,
                reason="window binning cannot be represented finitely"))
            failed_windows += 1
            continue
        counts = np.zeros(transit_bin_count, dtype=int)
        counts[bins["bin_indices"]] = bins["counts"]
        standardized = [None] * transit_bin_count
        full = bool(np.all(counts > 0))
        row = dict(centre_phase_days=float(centre), complete=full,
                   bin_counts=counts.tolist(), standardized_residuals=standardized)
        record["windows"].append(row)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                z = (bins["flux"] - 1) / bins["flux_err"]
                if not np.isfinite(z).all():
                    raise FloatingPointError
                for bin_index, value in zip(bins["bin_indices"], z):
                    standardized[int(bin_index)] = float(value)
                if full:
                    # The same bin weights, scaled to avoid squaring tiny errors.
                    scale = float(np.min(bins["flux_err"]))
                    weights = np.square(scale / bins["flux_err"])
                    mean_mode = float(np.sum(weights * (bins["flux"] - 1)) / math.sqrt(float(np.sum(weights))) / scale)
                    if not math.isfinite(mean_mode):
                        raise FloatingPointError
                    complete.append(z)
                    mean_modes.append(mean_mode)
                    record["unique_used_cadence_count"] += int(members.size)
        except FloatingPointError:
            row.update(complete=False, reason="window standardization cannot be represented finitely")
            failed_windows += 1
    record["complete_window_count"] = len(complete)
    record["sample_covariance_rank_upper_bound"] = min(transit_bin_count, max(len(complete) - 1, 0))
    if failed_windows:
        record.update(reason="folded covariance has numerical window failures", numerical_failure_window_count=failed_windows)
        return record
    if len(complete) < 2 or transit_bin_count < 2:
        record["reason"] = "need at least two complete OOT windows and two transit-resolution bins"
        return record
    try:
        summary = _covariance_summary(np.asarray(complete), mean_modes, adopted_uncertainty_factor)
    except FloatingPointError:
        record["reason"] = "folded covariance summary cannot be represented finitely"
        return record
    record.update(status="measured", **summary)
    return record
