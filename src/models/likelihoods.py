"""Mandel-Agol transit light curve models for TREX.

Uses the NumPy Mandel & Agol (2002) model for quadratic limb-darkened
transit and eclipse light curves. With no phase-bin box, the exposure box is averaged over
``nsamples`` sub-exposure midpoints, ``exptime * ((i + 0.5) / n - 0.5)``, the
rule PyTransit 2.2 (TRICERATOPS's engine) uses; batman's native supersampling
weights the exposure endpoints equally and converges only as O(1/n).  With a
phase-bin box, the two boxes are convolved with the exact trapezoid
kernel and Gauss-Legendre nodes from ``_two_box_convolution_quadrature``
instead of nested supersampling.

Times are relative to transit midpoint; native chronological times are also
accepted for the explicit baseline likelihood.

Verified sources, units, and failure boundary
----------------------------------------------
Mandel & Agol (2002), ApJ 580, L171, DOI ``10.1086/345520``. Phase-relative time, period,
epoch, and dynamic exposure use days; semimajor axis uses cm; radii use
``R_earth``/``R_sun`` as named; inclination/periastron use degrees; eccentricity,
limb darkening, flux ratios, and normalized model flux are dimensionless.
Legacy boxcar exposure must be finite and positive. Explicit temporal response
quadrature can instead represent the instantaneous mathematical limit.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from trex.foundation._numerics import (
    _per_bin_two_box_quadrature, _require_positive_width,
    _two_box_convolution_quadrature, phase_bin_widths,
)
from trex.foundation.constants import Rsun, Rearth
from trex.foundation.errors import NumericalLikelihoodError
from trex.models.flux_calibration import companion_flux_ratio, dilute_flux, validate_flux_uncertainty
from trex.models.orbital_mechanics import _validate_eccentric_orbit, kepler_sky_geometry, secondary_eclipse_phase
from trex.models.transit_flux import quadratic_limb_dark_flux



class DrawUnsupportedError(NumericalLikelihoodError):
    """A forward-model draw could not be evaluated numerically.

    Failure to solve the orbit or evaluate the flux is not physical zero
    support; callers retain it as unresolved and do not publish the integral.
    """



def _batman_transit(
    time: np.ndarray,
    period_days: float,
    rp_rs: float,
    a_rs: float,
    inc_deg: float,
    u1: float,
    u2: float,
    exptime_days: float | None,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    nsamples: int = 20,
    t0_days: float = 0.0,
    bin_integration_days: float = 0.0,
    *, angular_profile=None, exposure_response=None,
) -> np.ndarray:
    """Mandel-Agol flux convolved with exposure and phase-bin boxes.

    With a phase-bin box, the exact box * box trapezoid kernel is integrated
    against the instantaneous model at Gauss-Legendre nodes; ``nsamples``
    then only shapes the exposure supersampling of the unboxed path, because
    the boxed path carries the exposure box analytically inside the kernel.
    ``bin_integration_days`` is one width for every point or one per point.
    ``angular_profile`` supplies original (mu, absolute photon intensity) rays;
    The same Keplerian orbit is used, but no fitted angular law is evaluated.
    An explicit ``ExposureResponse`` instead supplies shared/per-row quadrature
    in TDB days, including an instantaneous limit or disjoint live windows.
    It requires ``exptime_days=None`` and zero phase-bin width to prevent
    accidentally integrating an exposure twice. ``nsamples`` then belongs to
    the response producer, not this legacy boxcar path.
    """
    from trex.photometry.exposure import ExposureResponse

    if exposure_response is None:
        _validate_exptime_days(exptime_days)
    elif not isinstance(exposure_response, ExposureResponse) or exptime_days is not None:
        raise ValueError("explicit exposure_response requires ExposureResponse and exptime_days=None")
    _validate_eccentric_orbit(ecc, argp_deg)
    for value in (period_days, rp_rs, a_rs):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("period and scaled radii must be finite positive real scalars")
    if isinstance(inc_deg, (bool, np.bool_)) or not isinstance(inc_deg, Real) or not math.isfinite(inc_deg) or not 0 <= inc_deg <= 180:
        raise ValueError("inclination must be a finite real angle in [0, 180] degrees")
    for value in (u1, u2, t0_days):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("limb-darkening coefficients and epoch must be finite real scalars")
    if isinstance(nsamples, (bool, np.bool_)) or not isinstance(nsamples, Integral) or nsamples < 1:
        raise ValueError("nsamples must be a positive integer")
    time_values = np.asarray(time)
    if (time_values.dtype.kind not in "iuf" or time_values.ndim != 1
            or not time_values.size or not np.all(np.isfinite(time_values))):
        raise ValueError("time must be a non-empty finite real one-dimensional array")
    time_values = time_values.astype(float, copy=False)
    bin_integration_days = phase_bin_widths(bin_integration_days, time_values.size)
    if exposure_response is not None:
        if np.ndim(bin_integration_days) or bin_integration_days != 0:
            raise ValueError("explicit exposure_response cannot also apply phase-bin integration")
        response_times = exposure_response.evaluation_times(time_values)

    try:
        if exposure_response is not None:
            evaluation_times = response_times.reshape(-1)
        elif isinstance(bin_integration_days, np.ndarray):
            # One trapezoid rule per bin: broad baseline bins integrate over
            # their own span, where an EB secondary eclipse can fall.
            row_offsets, row_weights = _per_bin_two_box_quadrature(bin_integration_days, exptime_days)
            evaluation_times = (time_values[:, None] + row_offsets).reshape(-1)
        else:
            if bin_integration_days > 0.0:
                offsets, weights = _two_box_convolution_quadrature(
                    bin_integration_days, exptime_days
                )
            else:
                offsets = exptime_days * ((np.arange(nsamples) + 0.5) / nsamples - 0.5)
                weights = np.full(nsamples, 1.0 / nsamples)
            evaluation_times = (time_values[:, None] + offsets[None, :]).reshape(-1)
        separation, foreground = kepler_sky_geometry(evaluation_times, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days)
        if angular_profile is None:
            flux = quadratic_limb_dark_flux(np.where(foreground, separation, np.inf), rp_rs, u1, u2)
        else:
            from trex.stellar_light.occultation import disc_flux
            flux = np.ones_like(separation)
            flux[foreground] = disc_flux(separation[foreground], rp_rs, angular_profile)
        if not np.all(np.isfinite(flux)):
            raise DrawUnsupportedError("forward model produced nonfinite flux")
        if exposure_response is not None:
            return np.sum(flux.reshape(response_times.shape) * exposure_response.weights, axis=1)
        if isinstance(bin_integration_days, np.ndarray):
            return np.sum(flux.reshape(row_offsets.shape) * row_weights, axis=1)
        return flux.reshape(time_values.size, offsets.size) @ weights
    except NumericalLikelihoodError:
        raise
    except (RuntimeError, ArithmeticError) as exc:
        raise DrawUnsupportedError(str(exc)) from exc


def _validate_exptime_days(exptime_days: float) -> None:
    _require_positive_width(exptime_days, "exptime_days")



def simulate_TP(
    time: np.ndarray,
    R_p_earth: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float | None,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *, angular_profile=None, exposure_response=None,
) -> np.ndarray:
    """Simulate a transiting planet light curve.

    Args:
        time: Phase-folded times [days from transit midpoint].
        R_p_earth: Planet radius [R_earth].
        P_orb: Orbital period [days].
        inc_deg: Inclination [degrees].
        a_cm: Semi-major axis [cm].
        R_s_solar: Stellar radius [R_sun].
        u1, u2: Quadratic limb-darkening coefficients.
        ecc, argp_deg: Eccentricity, argument of periastron.
        companion_fluxratio: F_comp / (F_comp + F_target).
        companion_is_host: True if transit on unresolved companion.
        exptime_days: Exposure time [days].
        nsamples: Positive integer exposure-midpoint count (not a boolean).
        bin_integration_days: Zero for exposure-only integration, a positive
            phase-bin width [days], or one positive width per time point.
        angular_profile: Original (mu, absolute photon intensity) rays;
            bypasses the quadratic angular law, not orbital propagation.
        exposure_response: Explicit temporal quadrature; requires
            exptime_days=None and bin_integration_days=0. See _batman_transit.

    Returns:
        Normalised flux array, one value per finite time. Radii, period and
        axis must be positive finite real scalars; inclination is in [0, 180],
        eccentricity in [0, 1), and both LDC are finite real scalars (no
        imposed sign cut on atmosphere coefficients). Companion fraction is
        in [0, 1); the host selector is boolean. Invalid inputs raise ValueError.
    """
    for value in (R_p_earth, R_s_solar, a_cm):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("physical radii and semi-major axis must be finite positive real scalars")
    if np.ndim(companion_fluxratio) != 0:
        raise ValueError("companion flux fraction must be scalar")
    companion_flux_ratio(companion_fluxratio)
    if not isinstance(companion_is_host, (bool, np.bool_)):
        raise ValueError("companion_is_host must be boolean")
    rp_rs = R_p_earth * Rearth / (R_s_solar * Rsun)
    a_rs = a_cm / (R_s_solar * Rsun)

    flux = _batman_transit(
        time, P_orb, rp_rs, a_rs, inc_deg, u1, u2, exptime_days,
        ecc, argp_deg, nsamples, bin_integration_days=bin_integration_days,
        angular_profile=angular_profile, exposure_response=exposure_response,
    )

    if companion_fluxratio > 0.0:
        flux = dilute_flux(flux, companion_fluxratio, companion_is_host)

    return flux


def simulate_EB(
    time: np.ndarray,
    R_EB_solar: float,
    EB_fluxratio: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float | None,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *,
    u1_EB: float,
    u2_EB: float,
    secondary_to_host_flux_ratio=None,
    angular_profile=None,
    secondary_angular_profile=None,
    exposure_response=None,
) -> tuple[np.ndarray, float]:
    """Simulate an eclipsing binary light curve.

    Returns the full binary model and the secondary-eclipse phase.
    ``u1, u2`` describe the primary; the required dimensionless quadratic
    ``u1_EB, u2_EB`` describe the eclipsed secondary's own atmosphere.
    Neither body's coefficients are substituted for the other body's.
    The spherical quadratic-disc model follows Mandel & Agol (2002), ApJ 580,
    L171, DOI ``10.1086/345520``. Times, period and
    exposure are in days, radii in R_sun, semimajor axis in cm and angles
    in degrees; flux and the returned orbital phase are dimensionless.
    Missing secondary coefficients or nonfinite coefficients are refused;
    unsupported scalar light-curve evaluations raise :class:`DrawUnsupportedError`.
    ``EB_fluxratio`` is ``F_secondary / (F_secondary + F_target)`` and
    ``companion_fluxratio`` is ``F_companion / (F_companion + F_target)``.
    Both reference the target, even when the companion hosts the pair; a
    secondary-to-host ratio may instead be passed explicitly through
    ``secondary_to_host_flux_ratio`` (finite nonnegative F_secondary/F_primary).
    That mode requires EB_fluxratio=None and accepts companion fractions in
    [0, 1], avoiding a lossy fraction-to-ratio round trip for generated draws.
    Radii, period and axis are finite positive real scalars; inclination is
    in [0, 180] degrees, eccentricity in [0, 1), flux fractions in [0, 1),
    and the host selector is boolean. Both bodies' LDC are finite real values;
    their signs are not restricted here. Invalid inputs raise ValueError.
    ``nsamples`` and ``bin_integration_days`` follow :func:`simulate_TP`.
    Original ``angular_profile`` and ``secondary_angular_profile`` rays must
    be supplied together; each replaces only its body's quadratic angular law.
    ``exposure_response`` applies the same declared temporal quadrature to
    both eclipses; its no-double-integration contract follows simulate_TP.
    """
    if (angular_profile is None) != (secondary_angular_profile is None):
        raise ValueError("a tabulated binary requires both bodies' original angular profiles")
    for value in (u1_EB, u2_EB):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("secondary limb-darkening coefficients must be finite real scalars")
    for value in (R_EB_solar, R_s_solar, a_cm):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("physical radii and semi-major axis must be finite positive real scalars")
    if not isinstance(companion_is_host, (bool, np.bool_)):
        raise ValueError("companion_is_host must be boolean")
    if np.ndim(companion_fluxratio) != 0 or np.ndim(EB_fluxratio) != 0:
        raise ValueError("binary flux fractions must be scalar")
    if secondary_to_host_flux_ratio is None:
        F_comp = companion_flux_ratio(companion_fluxratio)
        F_EB = companion_flux_ratio(EB_fluxratio)
        host = F_comp if companion_is_host else 1.0
        other = 1.0 if companion_is_host else F_comp
        total = host + F_EB + other
        weights = host / total, F_EB / total, other / total
    else:
        from trex.models.flux_calibration import binary_flux_weights

        if EB_fluxratio is not None or np.ndim(secondary_to_host_flux_ratio) != 0:
            raise ValueError("explicit secondary-to-host flux requires scalar ratio and EB_fluxratio=None")
        weights = binary_flux_weights(secondary_to_host_flux_ratio, companion_fluxratio, companion_is_host)
    secondary_phase = secondary_eclipse_phase(ecc, argp_deg)

    k = R_EB_solar / R_s_solar
    a_rs = a_cm / (R_s_solar * Rsun)

    primary_flux = _batman_transit(
        time, P_orb, k, a_rs, inc_deg, u1, u2, exptime_days,
        ecc, argp_deg, nsamples, bin_integration_days=bin_integration_days,
        angular_profile=angular_profile, exposure_response=exposure_response,
    )
    secondary_flux = _batman_transit(
        time, P_orb, 1.0 / k, a_rs / k, inc_deg, u1_EB, u2_EB, exptime_days,
        ecc, argp_deg + 180.0, nsamples, t0_days=secondary_phase * P_orb,
        bin_integration_days=bin_integration_days,
        angular_profile=secondary_angular_profile, exposure_response=exposure_response,
    )

    # Both eclipsed bodies and the constant source share one conserved total light.
    flux = weights[0] * primary_flux + weights[1] * secondary_flux + weights[2]

    return flux, secondary_phase



def _validated_observations(time, flux, sigma):
    sigma = validate_flux_uncertainty(sigma, flux)
    flux = np.asarray(flux, dtype=float)
    if np.shape(time) != flux.shape:
        raise ValueError("time and flux must be aligned one-dimensional arrays")
    return flux, sigma


def _model_log_likelihood(model, flux, sigma, baseline_likelihood):
    if baseline_likelihood is not None:
        value = baseline_likelihood(model)
    else:
        # Preserve the reference operation/reduction order, reusing the residual buffer.
        residual = flux - model
        np.square(residual, out=residual)
        residual /= np.square(sigma)
        value = -0.5 * np.sum(residual)
    if not np.isfinite(value):
        raise NumericalLikelihoodError("model likelihood is nonfinite")
    return float(value)


def lnL_TP(
    time: np.ndarray, flux: np.ndarray, sigma: np.ndarray | float,
    R_p_earth: float, P_orb: float, inc_deg: float, a_cm: float,
    R_s_solar: float, u1: float, u2: float, exptime_days: float | None,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *, angular_profile=None, baseline_likelihood=None, exposure_response=None,
) -> float:
    """Log-likelihood for a transiting planet scenario.

    lnL = -0.5 * sum((flux_obs - flux_model)^2 / sigma^2).
    Production sigma includes the shared native-time beta correction. It is
    fixed across scenarios; the aperture-frame Gaussian normalization is a
    common constant and cancels from their FPP/NFPP ratios. Do not fit a
    scenario-specific sigma with this unnormalized expression.
    In host coordinates sigma_host=sigma_ap/f, the N*log(f) density term
    cancels the -N*log(f) transformation Jacobian back to aperture data.
    Explicit ``baseline_likelihood`` instead evaluates the normalized aperture
    Gaussian with its proper nuisance prior integrated, on native unbinned time.
    Its prepared aperture observations/errors are authoritative; ``flux`` and
    ``sigma`` remain validated host-coordinate inputs, not a second likelihood
    or a replacement for those bound observations.
    ``exposure_response`` follows simulate_TP; it does not change the data's
    uncertainty model, baseline prior, or density normalization.
    """
    flux, sigma = _validated_observations(time, flux, sigma)
    model = simulate_TP(
        time, R_p_earth, P_orb, inc_deg, a_cm, R_s_solar, u1, u2,
        exptime_days, ecc, argp_deg, companion_fluxratio, companion_is_host,
        nsamples, bin_integration_days,
        angular_profile=angular_profile, exposure_response=exposure_response,
    )
    return _model_log_likelihood(model, flux, sigma, baseline_likelihood)


def lnL_EB(
    time: np.ndarray, flux: np.ndarray, sigma: np.ndarray | float,
    R_EB_solar: float, EB_fluxratio: float, P_orb: float,
    inc_deg: float, a_cm: float, R_s_solar: float,
    u1: float, u2: float, exptime_days: float | None,
    ecc: float = 0.0, argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0, companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *,
    u1_EB: float,
    u2_EB: float,
    secondary_to_host_flux_ratio=None,
    angular_profile=None,
    secondary_angular_profile=None,
    baseline_likelihood=None,
    exposure_response=None,
) -> float:
    """Log-likelihood for an eclipsing binary scenario.

    Both primary and secondary eclipses are evaluated at the observed times.
    Required secondary LDC and physical units follow :func:`simulate_EB`
    (Mandel & Agol 2002, ApJ 580, L171, DOI ``10.1086/345520``).
    Uses the same fixed, noise-calibrated sigma and omitted common Gaussian
    normalization as :func:`lnL_TP`, including after host renormalization.
    Flux and sigma are dimensionless; the returned log likelihood is in nats.
    Unevaluable draws raise, not substituted profiles. Missing or
    nonfinite secondary LDC is an input error; no claim eligibility is set.
    ``baseline_likelihood`` selects the same normalized joint density and
    prepared-data ownership as TP; host inputs still require aligned finite
    photometry and finite positive scalar/per-point errors.
    """
    flux, sigma = _validated_observations(time, flux, sigma)
    model, _ = simulate_EB(
        time, R_EB_solar, EB_fluxratio, P_orb, inc_deg, a_cm, R_s_solar,
        u1, u2, exptime_days, ecc, argp_deg, companion_fluxratio,
        companion_is_host, nsamples, bin_integration_days,
        u1_EB=u1_EB, u2_EB=u2_EB,
        secondary_to_host_flux_ratio=secondary_to_host_flux_ratio,
        angular_profile=angular_profile,
        secondary_angular_profile=secondary_angular_profile,
        exposure_response=exposure_response,
    )
    return _model_log_likelihood(model, flux, sigma, baseline_likelihood)


__all__ = [
    "DrawUnsupportedError",
    "simulate_TP",
    "simulate_EB",
    "lnL_TP",
    "lnL_EB",
]
