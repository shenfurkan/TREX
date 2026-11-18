"""Aperture-host flux calibration and dilution."""

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike


def companion_flux_ratio(flux_frac: ArrayLike) -> float | np.ndarray:
    """Convert finite real F_comp/(F_comp+F_target) in [0, 1) to F_comp/F_target.

    Booleans, strings and out-of-domain fractions raise ``ValueError``; no
    invalid fraction is replaced with a physical zero. Shape is preserved.
    """
    values = np.asarray(flux_frac)
    if values.dtype.kind not in "iuf":
        raise ValueError("companion flux fractions must be real numbers in [0, 1)")
    values = values.astype(float, copy=False)
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values >= 1.0)):
        raise ValueError("companion flux fractions must be finite and in [0, 1)")
    ratio = values / (1.0 - values)
    return float(ratio) if ratio.ndim == 0 else ratio


def binary_flux_weights(secondary_to_host, companion_fraction, companion_is_host, *, array_module=np):
    """Normalized primary, secondary and constant-light weights, shape broadcast.

    secondary_to_host is finite nonnegative F_secondary/F_primary; the companion
    fraction is F_companion/(F_companion+F_target) in [0, 1]. Endpoints represent
    limiting light shares, not reconstructed infinite ratios. NumPy validates
    inputs; traced callers own their finite/domain admission.
    """
    if array_module is np:
        from trex.populations._validation import boolean, real_array

        companion_is_host = boolean(companion_is_host, "companion_is_host")
        secondary_to_host = real_array(secondary_to_host, "secondary-to-host flux ratio")
        companion_fraction = real_array(companion_fraction, "companion flux fraction")
        if np.any(secondary_to_host < 0) or np.any((companion_fraction < 0) | (companion_fraction > 1)):
            raise ValueError("binary light requires nonnegative ratios and fractions in [0, 1]")
    host = companion_fraction if companion_is_host else 1.0 - companion_fraction
    secondary = secondary_to_host * host
    total = 1.0 + secondary
    return host / total, secondary / total, (1.0 - host) / total


def validate_flux_uncertainty(sigma: ArrayLike, flux: ArrayLike) -> float | np.ndarray:
    """Return a validated scalar or per-point flux uncertainty.

    A scalar applies to every point, as in the TRICERATOPS reference
    implementation.  An array must match ``flux`` and carries each observed
    point's own propagated uncertainty; with equal values it reproduces the
    scalar likelihood exactly. Every uncertainty must be real, finite and
    positive. Flux must be a non-empty finite real vector; booleans and strings
    are not photometry or uncertainties. Invalid inputs raise ``ValueError``.
    """
    values = np.asarray(sigma)
    flux_values = np.asarray(flux)
    if (flux_values.dtype.kind not in "iuf" or flux_values.ndim != 1
            or not flux_values.size or not np.all(np.isfinite(flux_values))):
        raise ValueError("flux must be a non-empty finite real one-dimensional array")
    if values.dtype.kind not in "iuf":
        raise ValueError("flux uncertainty must be real, not boolean or text")
    values = values.astype(float, copy=False)
    if values.ndim == 0:
        scalar = float(values)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("flux uncertainty must be finite and positive")
        return scalar
    if values.shape != flux_values.shape:
        raise ValueError("per-point flux uncertainty must match the flux array shape")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("flux uncertainty must be finite and positive")
    return values


def renorm_flux_to_host(
    flux: np.ndarray, sigma: ArrayLike, host_aperture_flux_fraction: float,
) -> tuple[np.ndarray, float | np.ndarray]:
    """Renormalize aperture photometry onto one resolved host source.

    Giacalone et al. (2021, AJ, 161, 24; DOI 10.3847/1538-3881/abc6af)
    evaluate each candidate host on the light curve it would have produced
    alone, by subtracting the constant flux of every other resolved source in
    the aperture and rescaling to the host's own flux level (the
    ``renorm_flux`` step of the reference implementation).  With ``f`` the
    host's fraction of the total in-aperture flux and a light curve normalized
    to unity out of transit,

    .. math::

        F_{\\mathrm{host}} = \\frac{F - (1 - f)}{f}, \\qquad
        \\sigma_{\\mathrm{host}} = \\frac{\\sigma}{f}.

    ``f`` must come from calibrated per-source aperture evidence; a catalog
    magnitude ratio is not an aperture throughput.  A source contributing no
    aperture flux cannot host the signal and is rejected here rather than
    being silently renormalized by a vanishing denominator.

    Args:
        flux: Normalized aperture flux, shape (N,).
        sigma: Normalized aperture flux uncertainty: a scalar, or one value
            per flux point (see :func:`validate_flux_uncertainty`).
        host_aperture_flux_fraction: Host share of total aperture flux, in
            ``(0, 1]``.

    Returns:
        ``(host_flux, host_sigma)`` in the host's own normalized flux units;
        ``host_sigma`` keeps the scalar or per-point form of ``sigma``.

    A normalized density in these host coordinates must include the
    transformation Jacobian f**(-N) to describe the original aperture data.
    Its log cancels the N*log(f) in the Gaussian determinant; the reduced
    -chi2/2 used by TREX is consequently invariant, also for per-point errors.
    """
    if isinstance(host_aperture_flux_fraction, (bool, np.bool_)) or not isinstance(host_aperture_flux_fraction, Real):
        raise ValueError("host aperture flux fraction must be a real number")
    fraction = float(host_aperture_flux_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            "host aperture flux fraction must be finite and within (0, 1]"
        )
    flux = np.asarray(flux)
    validated_sigma = validate_flux_uncertainty(sigma, flux)
    host_flux = (flux.astype(float, copy=False) - (1.0 - fraction)) / fraction
    return host_flux, validated_sigma / fraction


def dilute_flux(
    flux: np.ndarray, companion_fluxratio: float, companion_is_host: bool = False,
) -> np.ndarray:
    """Dilute normalized host flux using companion/total aperture flux in [0, 1).

    ``companion_is_host`` is a boolean selecting the event host. Zero leaves
    the input unchanged; negative, nonfinite, boolean or >=1 fractions raise
    ``ValueError``. The supported-input dilution algebra is unchanged.
    """
    if not isinstance(companion_is_host, (bool, np.bool_)):
        raise ValueError("companion_is_host must be boolean")
    if np.ndim(companion_fluxratio) != 0:
        raise ValueError("companion flux fraction must be scalar")
    companion_flux_ratio(companion_fluxratio)
    if companion_fluxratio == 0.0:
        return flux
    # Convex weights avoid a reciprocal that overflows for a subnormal companion.
    host_fraction = companion_fluxratio if companion_is_host else 1.0 - companion_fluxratio
    return host_fraction * flux + (1.0 - host_fraction)


__all__ = [
    "companion_flux_ratio", "binary_flux_weights", "validate_flux_uncertainty", "renorm_flux_to_host", "dilute_flux",
]
