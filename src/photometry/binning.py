"""The observed TREX inverse-variance phase-bin calculation, without new cuts."""
from __future__ import annotations

import math

import numpy as np

from trex.foundation.errors import NumericalLikelihoodError


def weighted_phase_bins(time_days, flux, flux_err, edges) -> dict[str, np.ndarray]:
    """Inverse-variance bins of relative flux, at strictly increasing phase-day edges.

    Inputs are aligned finite real 1-D vectors with positive reported errors;
    lists are accepted, booleans/text are not. Empty observations produce empty
    arrays. Points outside the edges are not selected. Bins are left-closed and
    right-open, except the last includes its right edge. Empty bins are omitted;
    ``bin_indices`` retains each populated bin's original index, not renumbering.
    All six fields are arrays; widths/time are days, flux/errors are relative flux.

    Ordinary inputs keep the observed preparer's reduction order and input order
    within each bin. Scaled inverse variances handle extreme positive errors
    without dropping a bin; unrepresentable results raise NumericalLikelihoodError.
    Invalid inputs raise ValueError. No exposure convolution is performed here.
    """
    arrays = [np.asarray(item) for item in (time_days, flux, flux_err, edges)]
    if any(item.dtype.kind not in "iuf" or item.ndim != 1 or not np.isfinite(item).all() for item in arrays):
        raise ValueError("phase bins require finite real one-dimensional photometry and edges")
    with np.errstate(over="ignore", invalid="ignore"):
        time_days, flux, flux_err, edges = (item.astype(float, copy=False) for item in arrays)
    if not all(np.isfinite(item).all() for item in (time_days, flux, flux_err, edges)):
        raise ValueError("phase-bin inputs must be representable finite float64 values")
    if flux.shape != time_days.shape or flux_err.shape != time_days.shape or np.any(flux_err <= 0):
        raise ValueError("phase bins require aligned flux and strictly positive reported errors")
    if edges.size < 2 or np.any(edges[1:] <= edges[:-1]):
        raise ValueError("phase-bin edges must contain at least two strictly increasing values")
    with np.errstate(over="ignore"):
        widths = np.diff(edges)
    if not np.isfinite(widths).all():
        raise ValueError("phase-bin widths must be representable finite days")

    binned_time, binned_flux, binned_err, binned_width = [], [], [], []
    bin_indices, counts = [], []
    indices = np.searchsorted(edges, time_days, side="right") - 1
    indices[time_days == edges[-1]] = edges.size - 2
    selected = np.flatnonzero((indices >= 0) & (indices < edges.size - 1))
    # Stable grouping preserves the original summation order without B full-data scans.
    selected = selected[np.argsort(indices[selected], kind="stable")]
    groups = np.split(selected, np.flatnonzero(np.diff(indices[selected])) + 1)
    for members in groups:
        if not members.size:
            continue
        index = int(indices[members[0]])
        times, values, errors = time_days[members], flux[members], flux_err[members]
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            weights = np.reciprocal(np.square(errors))
            weight_sum = float(np.sum(weights))
            mean_time = float(np.sum(weights * times) / weight_sum)
            mean_flux = float(np.sum(weights * values) / weight_sum)
            # Var(sum(w*f)/sum(w))=1/sum(w), for independent errors [relative flux²].
            error = math.sqrt(1.0 / weight_sum) if math.isfinite(weight_sum) and weight_sum > 0 else math.nan
        if not (math.isfinite(weight_sum) and weight_sum > 0
                and np.all(weights > 0) and math.isfinite(mean_time)
                and math.isfinite(mean_flux) and math.isfinite(error) and error > 0):
            # Scaling cancels in the means; sigma_min restores the propagated error.
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                scale = float(np.min(errors))
                weights = np.square(scale / errors)
                weight_sum = float(np.sum(weights))
                normalized = weights / weight_sum
                mean_time = float(np.sum(normalized * times))
                mean_flux = float(np.sum(normalized * values))
                error = scale / math.sqrt(weight_sum)
        if not (math.isfinite(mean_time) and math.isfinite(mean_flux) and math.isfinite(error) and error > 0):
            raise NumericalLikelihoodError("phase-bin mean or uncertainty cannot be represented finitely")
        binned_time.append(mean_time)
        binned_flux.append(mean_flux)
        binned_err.append(error)
        binned_width.append(float(widths[index]))
        bin_indices.append(index)
        counts.append(int(members.size))
    return dict(time_days=np.asarray(binned_time), flux=np.asarray(binned_flux),
                flux_err=np.asarray(binned_err), widths_days=np.asarray(binned_width),
                bin_indices=np.asarray(bin_indices, dtype=int), counts=np.asarray(counts, dtype=int))
