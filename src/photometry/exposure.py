"""Declared temporal averaging, separate from cadence and photon accumulation."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from trex.photometry.time_axis import _days


@dataclass(frozen=True, eq=False)
class ExposureResponse:
    """Finite quadrature of one declared temporal response.

    ``offset_days`` and dimensionless ``weights`` have matching shape (K,) for
    a shared rule or (N,K) for one rule per observation. K>=1; each row's
    finite nonnegative weights must sum to one within 32 float64 epsilons.
    Offsets are relative to the model's supplied timestamps in TDB days.
    No cadence, live-time fraction, or ramp estimator is inferred.

    Copies are read-only. Invalid, masked, unit-bearing or nonfinite inputs
    raise ValueError. A mission ramp needs its calibrated response, not a
    box with a duration taken from an effective-exposure keyword.
    """

    offset_days: np.ndarray
    weights: np.ndarray
    source: str

    def __post_init__(self):
        offsets = _days(self.offset_days, "integration offsets")
        weights = _days(self.weights, "integration weights")
        if (offsets.ndim not in (1, 2) or not offsets.size
                or offsets.shape != weights.shape):
            raise ValueError("integration offsets and weights need matching (K,) or (N,K) shapes")
        if np.any(weights < 0) or not np.allclose(
                weights.sum(axis=-1), 1., rtol=0., atol=32 * np.finfo(float).eps):
            raise ValueError("integration weights must be nonnegative and sum to one per row")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("integration response requires an explicit source")
        offsets.flags.writeable = weights.flags.writeable = False
        object.__setattr__(self, "offset_days", offsets)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def boxcar(cls, exposure_days, *, nsamples: int, source: str):
        """Midpoint rule for scalar/per-row nonnegative TDB widths [days].

        Zero is the instantaneous limit. ``nsamples`` is a positive integer,
        not an accuracy guarantee: callers check convergence for the actual
        ingress/egress and required depth/timing precision.
        """
        widths = _days(exposure_days, "exposure duration")
        if widths.ndim > 1 or not widths.size or np.any(widths < 0):
            raise ValueError("exposure duration must be nonnegative scalar or nonempty 1-D")
        if isinstance(nsamples, (bool, np.bool_)) or not isinstance(nsamples, Integral) or nsamples < 1:
            raise ValueError("nsamples must be a positive integer")
        offsets = widths[..., None] * ((np.arange(nsamples) + .5) / nsamples - .5)
        return cls(offsets, np.full(offsets.shape, 1. / nsamples), source)

    @classmethod
    def stacked(cls, start_days, end_days, *, nsamples: int, source: str):
        """Average disjoint uniform live windows, weighted by their durations.

        Starts/ends [TDB days relative to each supplied timestamp] have matching
        shape (W,) or (N,W). Ordered windows must have positive width and not
        overlap; gaps receive no weight. This models a sum of uniform live
        integrations, not an arbitrary weighted coadd or detector ramp.
        """
        start, end = _days(start_days, "window starts"), _days(end_days, "window ends")
        if (start.ndim not in (1, 2) or not start.size or start.shape != end.shape
                or np.any(end <= start) or np.any(start[..., 1:] < end[..., :-1])):
            raise ValueError("live windows must be ordered, disjoint and have positive widths")
        with np.errstate(over="ignore", invalid="ignore"):
            widths = end - start
            duration = widths.sum(axis=-1, keepdims=True)
        if not np.isfinite(duration).all():
            raise ValueError("total live-window duration must be finite")
        rule = cls.boxcar(widths.reshape(-1), nsamples=nsamples, source=source)
        offsets = rule.offset_days.reshape((*start.shape, nsamples))
        offsets = offsets + (start + widths / 2)[..., None]
        weights = np.broadcast_to((widths / duration / nsamples)[..., None], offsets.shape)
        shape = (*start.shape[:-1], -1)
        return cls(offsets.reshape(shape), weights.reshape(shape), source)

    def evaluation_times(self, time_days):
        """Return (N,K) finite evaluation times for the aligned model stamps."""
        times = _days(time_days, "model times")
        if times.ndim != 1 or not times.size:
            raise ValueError("model times must be a nonempty 1-D array")
        if self.offset_days.ndim == 2 and self.offset_days.shape[0] != times.size:
            raise ValueError("per-row integration response must match the model time array")
        with np.errstate(over="ignore", invalid="ignore"):
            result = times[:, None] + self.offset_days
        if not np.isfinite(result).all():
            raise ValueError("integration times must remain finite")
        return result
