"""Exact Gaussian linear-baseline marginalization on native chronological data.

For a fixed physical aperture mean m(theta), y=m+A(theta)b+epsilon,
b~N(mu,Lambda), epsilon~N(0,diag(sigma**2)). Gaussian convolution gives
y~N(m+A mu, diag(sigma**2)+A Lambda A.T), including its determinant.
This is the linear-basis identity discussed by Gibson (2014), MNRAS 445, 3401,
DOI 10.1093/mnras/stu1975, sections 2.2 and 3.3; no Laplace/profile approximation,
ML hyperparameter prior or empirical coefficient is used here.

The caller supplies a proper prior from an independently specified source.
No default basis, scale, conditioning posterior or detrended data is adopted.
Additive bases describe calibrated aperture residuals; multiplicative bases
describe fractional Gaussian residuals about the total aperture mean, NOT a
positive-throughput distribution. Neither model describes arbitrary activity.
For example, an independently calibrated background offset is additive; a
fractional calibration drift about the total aperture mean is multiplicative.
Time is native TDB days from a single conjunction, not folded phase; flux,
design columns and coefficients are dimensionless; covariance is squared
normalized relative flux, not detector-count variance.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from scipy.linalg import cho_solve

from trex.foundation._numerics import phase_bin_widths
from trex.foundation.errors import NumericalLikelihoodError
from trex.models.flux_calibration import validate_flux_uncertainty


@dataclass(frozen=True, eq=False)
class GaussianBaseline:
    """An explicit, proper Gaussian prior on fixed native-time basis weights.

    ``design`` must be evaluated at each cadence (not on folded offsets), with
    the finite-exposure convention declared in ``source``. Multiplicative
    deviations are constant within each exposure; rapidly varying throughput
    within an exposure is unsupported, not silently convolved approximately.
    Priors refer to aperture flux even when scenarios use host coordinates.

    Native TDB offsets [days] must strictly increase, with N>=2 observations.
    Explicit ``stream_ids`` instead requires increasing time within each stream;
    simultaneous samples from independent streams may share timestamps.
    Design has shape (N,K), mean (K,), covariance (K,K), K>=1. All are finite
    real inputs copied into read-only float64 arrays. Covariance must be positive
    definite and symmetric apart from relative roundoff (4 eps, zero absolute
    tolerance); only that roundoff is symmetrized. Invalid priors raise ValueError.
    """

    time_days: np.ndarray
    design: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    mode: str
    source: str
    stream_ids: tuple[str, ...] | None = None
    _prior_cholesky: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("time_days", "design", "prior_mean", "prior_covariance"):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "iuf":
                raise ValueError(f"baseline {name} must be real, not boolean or text")
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.array(value, dtype=np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"baseline {name} must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        t, x, mu, covariance = self.time_days, self.design, self.prior_mean, self.prior_covariance
        if t.ndim != 1 or t.size < 2:
            raise ValueError("baseline requires strictly increasing native TDB days, not folded phase")
        if self.stream_ids is None:
            if np.any(t[1:] <= t[:-1]):
                raise ValueError("baseline requires strictly increasing native TDB days, not folded phase")
        else:
            streams = tuple(self.stream_ids)
            if len(streams) != t.size or any(not isinstance(s, str) or not s.strip() for s in streams):
                raise ValueError("baseline stream_ids must identify every native observation")
            for stream in set(streams):
                values = t[np.array([s == stream for s in streams])]
                if np.any(values[1:] <= values[:-1]):
                    raise ValueError("baseline times must strictly increase within each native stream")
            object.__setattr__(self, "stream_ids", streams)
        if x.ndim != 2 or x.shape[0] != t.size or x.shape[1] < 1:
            raise ValueError("baseline design must align with every retained native cadence")
        if mu.shape != (x.shape[1],) or covariance.shape != (mu.size, mu.size):
            raise ValueError("baseline prior dimensions do not match its design")
        if not np.array_equal(covariance, covariance.T):
            # Permit only relative binary64 roundoff, not a different covariance model.
            if not np.allclose(covariance, covariance.T, rtol=4 * np.finfo(float).eps, atol=0):
                raise ValueError("baseline prior covariance must be symmetric")
            covariance = 0.5 * covariance + 0.5 * covariance.T
            covariance.flags.writeable = False
            object.__setattr__(self, "prior_covariance", covariance)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("baseline prior covariance must be positive definite") from exc
        if not np.isfinite(chol).all():
            raise ValueError("baseline prior covariance Cholesky must be finite")
        chol.flags.writeable = False
        object.__setattr__(self, "_prior_cholesky", chol)
        if (not isinstance(self.mode, str) or self.mode not in ("additive", "multiplicative")
                or not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("baseline needs an explicit mode and independent prior/basis source")

    def prepare(self, time, flux, sigma, bin_integration_days=0.0) -> BaselineLikelihood:
        """Bind finite unbinned aperture data; time must equal the prior's TDB offsets.

        Errors are positive scalar/per-cadence relative flux, never booleans.
        Only a real scalar zero phase-bin width is supported. Invalid input raises
        ValueError; unrepresentable Gaussian algebra raises NumericalLikelihoodError.
        """
        time = np.asarray(time)
        widths = phase_bin_widths(bin_integration_days, self.time_days.size)
        if (time.dtype.kind not in "iuf" or not np.array_equal(self.time_days, time)
                or np.any(widths != 0)):
            raise ValueError("Gaussian baseline requires its exact native-time observations without phase bins")
        return BaselineLikelihood(self, flux, sigma)

    def provenance(self) -> dict:
        return {"model": "native-gaussian-linear-baseline-v1", "mode": self.mode,
                "source": self.source, "coefficient_count": int(self.prior_mean.size),
                "time_domain": ("native chronological TDB days from conjunction" if self.stream_ids is None else
                                "native TDB days; chronological within each explicit stream"),
                "normalization": "full aperture Gaussian density; proper nuisance prior integrated exactly",
                "execution": "scalar float64 reference and native float64 JAX Gaussian convolution",
                "conditioning": "no fitted trend or conditioning-data likelihood is reused",
                "domain": "fixed basis; Gaussian calibrated residuals; multiplicative deviations constant within exposure"}

    def to_dict(self) -> dict:
        """Finite JSON-compatible physical prior, including exact stream order."""
        return {name: value.tolist() if isinstance(value, np.ndarray) else value
                for name in ("time_days", "design", "prior_mean", "prior_covariance", "mode", "source", "stream_ids")
                for value in (getattr(self, name),)}


class BaselineLikelihood:
    """Prepared aperture density; host-coordinate predictions are mapped back once.

    Fixed D, X, Lambda algebra is shared by TP/EB and every population/draw.
    Multiplicative A=diag(m)X is recomputed for each changing physical mean.
    """

    def __init__(self, prior: GaussianBaseline, flux, sigma):
        self.prior = prior
        sigma = validate_flux_uncertainty(sigma, flux)
        if np.shape(flux) != prior.time_days.shape:
            raise ValueError("baseline likelihood needs aligned finite aperture flux and positive reported errors")
        with np.errstate(over="ignore", invalid="ignore"):
            self.flux = np.array(flux, dtype=np.float64, copy=True)
        if not np.isfinite(self.flux).all():
            raise ValueError("baseline likelihood needs aligned finite aperture flux and positive reported errors")
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), self.flux.shape).copy()
        self.flux.flags.writeable = self.sigma.flags.writeable = False
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                self.whitened_factor = (prior.design @ prior._prior_cholesky) / self.sigma[:, None]
                self.mean_shift = prior.design @ prior.prior_mean
                # Normalized Gaussian log|2pi D| [nat], without squaring tiny errors.
                self.normalization = float(self.flux.size * np.log(2 * np.pi) + 2 * np.sum(np.log(self.sigma)))
        except FloatingPointError as exc:
            raise NumericalLikelihoodError("baseline prior prediction cannot be represented finitely in float64") from exc
        if not (np.isfinite(self.whitened_factor).all() and np.isfinite(self.mean_shift).all()
                and math.isfinite(self.normalization)):
            raise NumericalLikelihoodError("baseline prior prediction cannot be represented finitely in float64")
        self.whitened_factor.flags.writeable = self.mean_shift.flags.writeable = False
        self.host_fraction = 1.0
        self.fixed = self._algebra(self.whitened_factor) if prior.mode == "additive" else None

    def for_host(self, fraction: float) -> BaselineLikelihood:
        """Share aperture algebra for a finite real aperture-light fraction in (0,1].

        Boolean/non-real/out-of-domain fractions raise ValueError. The prepared
        observations and their errors stay in the original aperture coordinates.
        """
        if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("baseline host aperture fraction must lie in (0,1]")
        other = copy.copy(self)
        other.host_fraction = float(fraction)
        return other

    def _algebra(self, whitened: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Gaussian convolution/determinant lemma: C=D+UU.T; H=I+U.T D^-1 U [dimensionless].
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                chol = np.linalg.cholesky(np.eye(whitened.shape[1]) + whitened.T @ whitened)
                logdet = float(2 * np.sum(np.log(np.diag(chol))))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise NumericalLikelihoodError("baseline covariance evaluation failed") from exc
        if not (np.isfinite(whitened).all() and np.isfinite(chol).all() and math.isfinite(logdet)):
            raise NumericalLikelihoodError("baseline covariance cannot be represented finitely in float64")
        chol.flags.writeable = False
        return whitened, chol, logdet

    def __call__(self, model) -> float:
        model = np.asarray(model)
        if model.dtype.kind not in "iuf" or model.shape != self.flux.shape:
            raise ValueError("baseline requires one finite physical prediction per native observation")
        model = model.astype(np.float64, copy=False)
        if not np.isfinite(model).all():
            raise NumericalLikelihoodError("baseline physical prediction is nonfinite")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Original aperture coordinates need no extra host prefactor.
                aperture_mean = 1 + self.host_fraction * (model - 1)
                multiplier = aperture_mean if self.prior.mode == "multiplicative" else 1.0
                residual = (self.flux - aperture_mean - multiplier * self.mean_shift) / self.sigma
                algebra = self.fixed if self.fixed is not None else self._algebra(aperture_mean[:, None] * self.whitened_factor)
                whitened, chol, logdet = algebra
                weights = cho_solve((chol, True), whitened.T @ residual, check_finite=False)
                # Completed square avoids subtracting nearly equal large quadratic forms.
                remainder = residual - whitened @ weights
                quadratic = float(remainder @ remainder + weights @ weights)
                value = -0.5 * (quadratic + self.normalization + logdet)
        except FloatingPointError as exc:
            raise NumericalLikelihoodError("baseline likelihood evaluation overflowed") from exc
        if not np.isfinite(value):
            raise NumericalLikelihoodError("baseline likelihood is nonfinite; no zero-support replacement")
        return value

    def on_device(self, device):
        """Return a JIT-compatible float64 aperture Gaussian on the selected device.

        Requires JAX x64 enabled; wrong model shape/type raises ValueError at trace
        time. Nonfinite predictions or density algebra return NaN, not zero support:
        native integrands must propagate that numerical-failure marker. JIT code
        cannot raise the scalar path's NumericalLikelihoodError on a device value.
        """
        from trex.accelerators import _jax_models as models

        jax, jnp = models.jax, models.jnp
        if not jax.config.x64_enabled:
            raise RuntimeError("baseline device density requires JAX float64 mode")
        flux, sigma, factor, shift = (jax.device_put(value, device) for value in
                                      (self.flux, self.sigma, self.whitened_factor, self.mean_shift))
        fixed = None if self.fixed is None else tuple(jax.device_put(value, device) for value in self.fixed)
        fraction, normalization = self.host_fraction, self.normalization

        def density(model):
            if not jax.config.x64_enabled:
                raise RuntimeError("baseline device density requires JAX float64 mode")
            try:
                model = jnp.asarray(model)
            except (TypeError, ValueError) as exc:
                raise ValueError("baseline requires one finite physical prediction per native observation") from exc
            if model.dtype.kind not in "iuf" or model.shape != flux.shape:
                raise ValueError("baseline requires one finite physical prediction per native observation")
            model = model.astype(jnp.float64)
            mean = 1.0 + fraction * (model - 1.0)
            multiplier = mean if self.prior.mode == "multiplicative" else 1.0
            residual = (flux - mean - multiplier * shift) / sigma
            if fixed is None:
                whitened = mean[:, None] * factor
                chol = jnp.linalg.cholesky(jnp.eye(factor.shape[1], dtype=jnp.float64) + whitened.T @ whitened)
                logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
            else:
                whitened, chol, logdet = fixed
            weights = jax.scipy.linalg.cho_solve((chol, True), whitened.T @ residual)
            remainder = residual - whitened @ weights
            quadratic = remainder @ remainder + weights @ weights
            value = -0.5 * (quadratic + normalization + logdet)
            return jnp.where(jnp.isfinite(model).all() & jnp.isfinite(value), value, jnp.nan)

        return density
