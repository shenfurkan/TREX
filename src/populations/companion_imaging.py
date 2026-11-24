"""Unresolved-source contrast weights after the eclipsing pair is drawn.

Giacalone et al. (2021), AJ 161, 24, DOI
10.3847/1538-3881/abc6af, Section 2.4.3, specifies imaging
weights. The SEB/BEB reference sums component/target fluxes before taking
contrast. Image separations are arcsec, contrast is mag, flux
ratios are dimensionless and log occurrence weights are natural logarithms.
The sampled-separation support rule remains TREX's, not official equivalence.

A measured curve bounds only its own filter. Independent nondetections in the
same filter are combined as separate constraints; curves in different filters
are never merged into one envelope, because a non-detection in one band says
nothing about the contrast in another.
"""

from dataclasses import dataclass, field

import numpy as np

from trex.photometry.photometric_relations import flux_ratio_to_delta_mag
from trex.foundation.errors import NumericalLikelihoodError
from trex.populations._validation import boolean, real_array


def image_source_flux_ratio(companion_flux_ratio, secondary_flux_ratio, companion_is_host, *, array_module=np):
    """G21 §2.4.3 image-source contrast ratio, with broadcast dimensionless fluxes.

    Companion flux is positive; secondary flux is nonnegative, both F/F_target.
    Hosted EB light is added to the companion; otherwise it brightens the target
    denominator. NumPy validates concrete inputs and boolean host choice; traced
    JAX callers pass array_module=jnp and own their support/numerical-failure mask.
    """
    companion_is_host = boolean(companion_is_host, "companion_is_host")
    if array_module is np:
        companion_flux_ratio = real_array(companion_flux_ratio, "companion flux ratios")
        secondary_flux_ratio = real_array(secondary_flux_ratio, "secondary flux ratios")
        if np.any(companion_flux_ratio <= 0) or np.any(secondary_flux_ratio < 0):
            raise ValueError("imaging requires positive companion and nonnegative secondary flux")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = (companion_flux_ratio + secondary_flux_ratio if companion_is_host
                  else companion_flux_ratio / (1.0 + secondary_flux_ratio))
    if array_module is np and (not np.isfinite(result).all() or np.any(result <= 0)):
        raise NumericalLikelihoodError("image-source flux ratio is unrepresentable")
    return result


@dataclass(frozen=True)
class ContrastCurve:
    """Immutable measured curve: increasing positive separations [arcsec], contrasts [mag].

    ``band_id`` is the native filter the curve was measured in, ``detection_model``
    the declared non-detection rule and ``source`` the retained origin of the
    numbers. They are keyword-only and default to ``""`` = not declared, so a
    legacy two-argument curve still constructs; the empty value never stands for
    a guessed filter or model, and :class:`MeasuredContrastSet` refuses it.

    No sorting, extrapolation or artificial contrast monotonicity is imposed.
    Finite real 1-D matching vectors with >=2 points are required.
    """

    separations: np.ndarray
    contrasts: np.ndarray
    band_id: str = field(default="", kw_only=True)
    detection_model: str = field(default="", kw_only=True)
    source: str = field(default="", kw_only=True)

    def __post_init__(self):
        s = real_array(self.separations, "contrast separations").copy()
        c = real_array(self.contrasts, "contrast values").copy()
        if s.ndim != 1 or c.ndim != 1 or s.shape != c.shape or s.size < 2:
            raise ValueError("contrast curve requires matching separation and contrast arrays")
        if np.any(s <= 0) or np.any(s[1:] <= s[:-1]):
            raise ValueError("contrast curve separations must be positive and strictly increasing")
        for name in ("band_id", "detection_model", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise ValueError(f"contrast curve {name} must be a nonempty string when declared")
        s.flags.writeable = c.flags.writeable = False
        object.__setattr__(self, "separations", s)
        object.__setattr__(self, "contrasts", c)

    def at(self, separation_arcsec):
        """Finite nonnegative query [arcsec] -> limit [mag], NaN off-curve."""
        query = real_array(separation_arcsec, "projected separations")
        if np.any(query < 0):
            raise ValueError("projected separations must be nonnegative")
        return np.interp(query, self.separations, self.contrasts, left=np.nan, right=np.nan)


@dataclass
class ImagingConstraint:
    """A sampled source and its measured contrast limit, without an EB mask yet."""

    flux_ratio: np.ndarray
    limit_mag: np.ndarray | None
    require_fainter: bool

    def apply(self, lnprior, secondary_flux_ratio=0.0, companion_is_host=False, *, secondary_to_host=False):
        """Apply only to finite-prior draws; existing -inf draws remain rejected.

        None limits (or NaN off-curve entries) mean no measured constraint.
        require_fainter keeps delta_mag>=0: an equal-brightness twin is supported.
        Inputs broadcast to a common shape; NaN/+inf priors are errors, not cuts.
        """
        require_fainter = boolean(self.require_fainter, "require_fainter")
        companion_is_host = boolean(companion_is_host, "companion_is_host")
        secondary_to_host = boolean(secondary_to_host, "secondary_to_host")
        raw = [np.asarray(value) for value in (lnprior, self.flux_ratio, secondary_flux_ratio)]
        if any(value.dtype.kind not in "iuf" for value in raw):
            raise ValueError("imaging priors and fluxes must be real, not boolean or text")
        if self.limit_mag is not None:
            limit = np.asarray(self.limit_mag)
            if limit.dtype.kind not in "iuf" or np.isinf(limit).any():
                raise ValueError("imaging limits must be real finite values or NaN off-curve")
            prior, companion, secondary, limit = np.broadcast_arrays(*raw, limit)
        else:
            prior, companion, secondary = np.broadcast_arrays(*raw)
            limit = None
        prior = np.array(prior, dtype=float, copy=True)
        if np.any(np.isnan(prior) | np.isposinf(prior)):
            raise ValueError("imaging log priors must be finite or negative infinity")
        active = np.isfinite(prior)
        if limit is not None:
            active &= np.isfinite(limit) | require_fainter
        elif not require_fainter:
            return prior
        if not np.any(active):
            return prior
        if secondary_to_host:
            from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT

            c = real_array(companion[active], "companion flux ratios")
            r = real_array(secondary[active], "secondary-to-host flux ratios")
            if np.any(c <= 0) or np.any(r < 0):
                raise ValueError("imaging requires positive companion and nonnegative secondary flux")
            # Log light sums retain a finite contrast without constructing c*r or c*(1+r).
            log_ratio = np.log(c) + (np.log1p(r) if companion_is_host else -np.log1p(r))
            delta_mag = log_ratio / (np.log(10.0) * POGSON_FLUX_EXPONENT)
        else:
            ratio = image_source_flux_ratio(companion[active], secondary[active], companion_is_host)
            delta_mag = flux_ratio_to_delta_mag(ratio)
        retained = np.ones(delta_mag.shape, dtype=bool) if limit is None else ~(delta_mag < limit[active])
        if require_fainter:
            retained &= delta_mag >= 0.0
        prior[active] = np.where(retained, prior[active], -np.inf)
        return prior


@dataclass(frozen=True)
class MeasuredContrastSet:
    """Independent measured non-detection curves, grouped by exact native band.

    Each curve is one observation's limiting contrast [delta-mag] against
    separation [arcsec].  Two curves in the same filter constrain the same
    companion independently: every curve whose measured support covers the
    separation excludes companions brighter than its own limit, so the
    surviving limit is the maximum (tightest) of those limits.  Curves in
    different filters are never merged into one envelope, because a
    non-detection in one band does not bound the contrast in another.
    ``band_id`` matching is exact; nothing is normalised or aliased.

    Outside a curve's measured support its limit is ``NaN`` and places no
    bound; that gap is not a detection and is never treated as one.  The
    imaging weight itself follows Giacalone et al. (2021), AJ 161, 24, DOI
    10.3847/1538-3881/abc6af, Section 2.4.3; the per-band grouping and
    support rule remain TREX's, not official equivalence.

    ``detection_model`` records the declared rule of each curve.  Only a
    step limit is implemented: a companion at least as faint as the limit
    survives, a brighter one is rejected.  A probabilistic detection model
    is not implemented and must not be inferred from this class or from the
    metadata it carries.
    """

    curves: tuple

    def __post_init__(self):
        curves = tuple(self.curves)
        if not curves:
            raise ValueError("measured contrast set requires at least one curve")
        for curve in curves:
            if not isinstance(curve, ContrastCurve):
                raise ValueError("measured contrast set entries must be ContrastCurve instances")
            for name in ("band_id", "detection_model", "source"):
                if not getattr(curve, name):
                    raise ValueError(
                        f"measured contrast curve must declare {name}; it cannot be inferred"
                    )
        object.__setattr__(self, "curves", curves)

    @property
    def band_ids(self) -> tuple:
        """Declared band ids in first-declaration order."""
        return tuple(dict.fromkeys(curve.band_id for curve in self.curves))

    def _band(self, band_id):
        if not isinstance(band_id, str) or not band_id.strip():
            raise ValueError("band id must be a nonempty string")
        selected = tuple(curve for curve in self.curves if curve.band_id == band_id)
        if not selected:
            raise ValueError(f"no measured contrast curve declares band {band_id!r}")
        return selected

    @staticmethod
    def _tightest(band, query):
        """Maximum limit [mag] over the band's curves covering each query [arcsec]."""
        limit = np.full(query.shape, np.nan)
        for curve in band:
            limit = np.fmax(limit, curve.at(query))
        return limit

    def at(self, band_id, separation_arcsec):
        """Tightest surviving limit [mag] in one band, NaN off every curve's support.

        Raises ValueError for an unknown band instead of borrowing another
        band's limit.
        """
        band = self._band(band_id)
        query = real_array(separation_arcsec, "projected separations")
        if np.any(query < 0):
            raise ValueError("projected separations must be nonnegative")
        return self._tightest(band, query)

    def apply(
        self,
        band_id,
        lnprior,
        flux_ratio,
        separation_arcsec,
        *,
        require_fainter=False,
        secondary_flux_ratio=0.0,
        companion_is_host=False,
    ):
        """Apply only the selected band's limits to a companion log prior.

        ``flux_ratio`` is the companion/host flux ratio, ``separation_arcsec``
        the projected separation [arcsec] and the limit is delta-mag.  A NaN
        limit (outside every curve's support in that band) means that band
        places no constraint, exactly as the legacy single-curve path; a
        detected companion is rejected with ``-inf``.  With
        ``require_fainter`` an equal-brightness twin is still allowed where
        no limit covers the separation, while a companion brighter than the
        host is not.
        """
        band = self._band(band_id)
        require_fainter = boolean(require_fainter, "require_fainter")
        companion_is_host = boolean(companion_is_host, "companion_is_host")
        prior = np.asarray(lnprior)
        query = np.asarray(separation_arcsec)
        flux = np.asarray(flux_ratio)
        if prior.dtype.kind not in "iuf" or query.dtype.kind not in "iuf":
            raise ValueError("imaging priors and separations must be real, not boolean or text")
        if flux.dtype.kind not in "iuf":
            raise ValueError("imaging priors and fluxes must be real, not boolean or text")
        if np.any(np.isnan(prior) | np.isposinf(prior)):
            raise ValueError("imaging log priors must be finite or negative infinity")
        if np.any(query < 0):
            raise ValueError("projected separations must be nonnegative")
        prior, query, flux = np.broadcast_arrays(prior, query, flux)
        limit = np.full(prior.shape, np.nan)
        active = np.isfinite(prior)
        limit[active] = self._tightest(band, query[active])
        constraint = ImagingConstraint(flux, limit, require_fainter)
        return constraint.apply(
            prior, secondary_flux_ratio=secondary_flux_ratio, companion_is_host=companion_is_host
        )


def imaging_log_prior(
    flux_ratio, separation_arcsec, contrast_separations, contrast_values, lnprior,
    *, require_fainter=False, deferred=None,
):
    """Apply imaging now, or append a constraint IN PLACE to deferred for the EB.

    Both curve arrays are required together. No curve and no fainter-source rule
    is a true no-op; it need not convert unused fluxes to magnitudes.
    """
    require_fainter = boolean(require_fainter, "require_fainter")
    if (contrast_separations is None) != (contrast_values is None):
        raise ValueError("contrast separations and values must be supplied together")
    prior = np.asarray(lnprior)
    if prior.dtype.kind not in "iuf":
        raise ValueError("imaging priors and separations must be real, not boolean or text")
    if np.any(np.isnan(prior) | np.isposinf(prior)):
        raise ValueError("imaging log priors must be finite or negative infinity")
    if contrast_separations is None:
        limit = None
    else:
        curve = ContrastCurve(contrast_separations, contrast_values)
        query = np.asarray(separation_arcsec)
        if query.dtype.kind not in "iuf":
            raise ValueError("imaging priors and separations must be real, not boolean or text")
        prior, query, _ = np.broadcast_arrays(prior, query, np.asarray(flux_ratio))
        limit = np.full(prior.shape, np.nan)
        active = np.isfinite(prior)
        limit[active] = curve.at(query[active])
    constraint = ImagingConstraint(flux_ratio, limit, require_fainter)
    if deferred is not None:
        if not isinstance(deferred, list):
            raise ValueError("deferred imaging constraints must be a list")
        if limit is not None or require_fainter:
            deferred.append(constraint)
        return lnprior
    return constraint.apply(lnprior)


def binary_imaging_log_prior(lnprior, constraints, secondary_flux_ratio, companion_is_host, *, secondary_to_host=False):
    """Use the actual drawn secondary; never resample or change its prior/proposal."""
    companion_is_host = boolean(companion_is_host, "companion_is_host")
    for constraint in constraints:
        lnprior = constraint.apply(lnprior, secondary_flux_ratio=secondary_flux_ratio,
                                   companion_is_host=companion_is_host, secondary_to_host=secondary_to_host)
    return lnprior


def contrast_limit_at_separation(
    separation_arcsec: np.ndarray,
    separations: np.ndarray,
    contrasts: np.ndarray,
) -> np.ndarray:
    """Return the faintest detectable delta-mag at each projected separation.

    This reads the contrast curve in its measured direction -- separation in,
    limiting contrast out -- so no draw has to be extrapolated past the
    tabulated contrast range.  Separations outside the curve's own coverage
    return ``nan``: inside its inner working angle and outside its outer edge
    the imaging places no constraint, which the caller must handle explicitly
    rather than treating as a detection.

    Args:
        separation_arcsec: Finite nonnegative projected separations [arcsec], scalar or array.
        separations: Contrast-curve separations [arcsec], shape (M,).
        contrasts: Contrast-curve limiting contrasts [delta_mag], shape (M,).

    Returns:
        Limiting delta-mag at each separation, shape (N,), ``nan`` off-curve.
    """
    return ContrastCurve(separations, contrasts).at(separation_arcsec)
