"""Shared linear-ephemeris uncertainty propagated to transit epochs.

A single fitted ``(t0, P)`` couples every predicted epoch ``T_E = t0 + E P``.
Its covariance is the linear propagation ``J Sigma J^T`` with rows
``J_E = (1, E)``, i.e. ``Var(T_E) = Var(t0) + E^2 Var(P) + 2 E Cov(t0, P)`` and
``Cov(T_E, T_F) = Var(t0) + (E + F) Cov(t0, P) + E F Var(P)``. This is exact for
the linear ephemeris (no approximation beyond the Gaussian measure supplied by
the caller). Independent per-band or per-visit timing offsets are a different,
separately declared model and are not produced here. No sigma clipping or
support truncation is applied: Gaussian tails remain unbounded.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True, eq=False)
class LinearEphemeris:
    """Gaussian measure of a linear ephemeris.

    ``t0_days`` is the reference epoch as days relative to the caller's stated
    TDB reference ``reference_label``; ``period_days`` in days.
    ``covariance`` is the 2x2 covariance of ``(t0, P)`` in days^2, symmetric
    positive semidefinite. ``source`` names the fit/catalog that produced it.
    """

    t0_days: float
    period_days: float
    covariance: np.ndarray
    reference_label: str
    source: str

    def __post_init__(self):
        values = np.array([self.t0_days, self.period_days], dtype=np.float64)
        if not np.isfinite(values).all() or values[1] <= 0:
            raise ValueError("ephemeris needs finite t0 and a positive finite period [days]")
        covariance = np.array(self.covariance, dtype=np.float64, copy=True)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ValueError("ephemeris covariance must be a finite 2x2 matrix [days^2]")
        if covariance[0, 1] != covariance[1, 0]:
            raise ValueError("ephemeris covariance must be exactly symmetric")
        # Tolerance: eigenvalue roundoff of a 2x2 symmetric matrix, relative to its scale.
        scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
        if np.min(np.linalg.eigvalsh(covariance)) < -8 * np.finfo(float).eps * scale:
            raise ValueError("ephemeris covariance must be positive semidefinite")
        for name in ("reference_label", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ephemeris requires an explicit {name}")
        covariance.flags.writeable = False
        object.__setattr__(self, "t0_days", float(values[0]))
        object.__setattr__(self, "period_days", float(values[1]))
        object.__setattr__(self, "covariance", covariance)

    def epochs(self, epoch_numbers):
        """Predicted mid-times [days] and their full covariance [days^2].

        ``epoch_numbers`` are integers (1-D). The returned covariance is the
        coupled measure for all epochs together; using only its diagonal as
        independent shifts discards the shared ephemeris correlation.
        """
        numbers = np.asarray(epoch_numbers)
        if numbers.ndim != 1 or not numbers.size or numbers.dtype.kind not in "iu":
            raise ValueError("epoch numbers must be a nonempty 1-D integer array")
        jacobian = np.stack([np.ones(numbers.size), numbers.astype(np.float64)], axis=1)
        times = self.t0_days + numbers.astype(np.float64) * self.period_days
        covariance = jacobian @ self.covariance @ jacobian.T
        return times, 0.5 * (covariance + covariance.T)

    def epoch_variance(self, epoch_number):
        """Var(T_E) = Var(t0) + E^2 Var(P) + 2 E Cov(t0, P) [days^2]."""
        if isinstance(epoch_number, bool) or not isinstance(epoch_number, Integral):
            raise ValueError("epoch number must be an integer")
        e = float(epoch_number)
        c = self.covariance
        return float(c[0, 0] + e * e * c[1, 1] + 2.0 * e * c[0, 1])
