"""Device-native fixed-host TP unit-cube integrand.

This is the integral defined by ``_TransitingPlanetIntegrand``: retained
Giacalone radius priors, Kipping eccentricity prior, isotropic transit-band
change of variables, detached-orbit support, and the unnormalized fixed-sigma
Gaussian likelihood. Production importance refinement can use this callback;
population scenarios use their own adapters.

Sources: Giacalone et al. (2021), AJ 161, 24, DOI
``10.3847/1538-3881/abc6af``; Kipping (2013), MNRAS 434, L51,
DOI ``10.1093/mnrasl/slt075``; Winn (2010), *Exoplanets*, ed. S. Seager,
University of Arizona Press, pp. 55–77, Eq. 7; preprint DOI
``10.48550/arXiv.1001.2010``;
Agol et al. (2020), AJ 159, 123, DOI
``10.3847/1538-3881/ab4fee``. Coefficients come from the retained priors, not
new fitted approximations. The radius-prior M/FGK switch is the shared
``priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR`` (G21's 0.63 Msun).

Times/exposure/bin widths are TDB days relative to conjunction, radius is in
Earth/solar radii as named, angles are degrees, semimajor axis is in stellar
radii, flux/errors are fractional, and log integrands are in nats. The input
photometry is already renormalized by the scenario caller. A numerical failure
returns NaN with an explicit validity flag, never physical zero support.
All evaluations are float64 JAX, without host callbacks or an eccentricity cutoff.
"""

import math
from typing import NamedTuple

import numpy as np

from trex.accelerators import _jax_models as models
from trex.populations import priors
from trex.foundation._numerics import _per_bin_two_box_quadrature, _require_positive_width, _two_box_convolution_quadrature, phase_bin_widths
from trex.foundation.constants import Rearth, Rsun


class UnitIntegrandEvaluation(NamedTuple):
    """Log integral density and numerical validity, separate from physical support."""

    log_integrand: object
    numerically_valid: object
    grid_boundary_clamped: object = False


def _photometry_on_device(reference, device):
    """Validate/retain fractional photometry and TDB-day quadrature at setup."""
    if isinstance(reference.nsamples, (bool, np.bool_)) or not isinstance(reference.nsamples, (int, np.integer)) or reference.nsamples < 1:
        raise ValueError("nsamples must be a positive integer")
    _require_positive_width(reference.exptime_days, "exptime_days")
    time = np.asarray(reference.time, dtype=np.float64)
    flux = np.asarray(reference.flux, dtype=np.float64)
    sigma = np.asarray(reference.sigma, dtype=np.float64)
    if time.ndim != 1 or time.size == 0 or flux.shape != time.shape:
        raise ValueError("time and flux must be matching non-empty one-dimensional arrays")
    if sigma.ndim > 1 or (sigma.ndim == 1 and sigma.shape != time.shape):
        raise ValueError("sigma must be a scalar or match the time array")
    if not (np.isfinite(time).all() and np.isfinite(flux).all()
            and np.isfinite(sigma).all() and (sigma > 0.0).all()):
        raise ValueError("photometry must be finite with positive sigma")
    if not np.isfinite([reference.u1, reference.u2]).all():
        raise ValueError("host limb-darkening coefficients must be finite")
    widths = phase_bin_widths(reference.bin_integration_days, time.size)
    if isinstance(widths, np.ndarray):
        offsets, weights = _per_bin_two_box_quadrature(widths, reference.exptime_days)
    elif widths > 0.0:
        offsets, weights = _two_box_convolution_quadrature(widths, reference.exptime_days)
    else:
        nodes = int(reference.nsamples)
        offsets = reference.exptime_days * ((np.arange(nodes) + 0.5) / nodes - 0.5)
        weights = np.full(nodes, 1.0 / nodes)
    grid = time[:, None] + (offsets if offsets.ndim == 2 else offsets[None, :])
    arrays = (grid.reshape(-1), weights, flux, sigma,
              np.asarray([reference.u1, reference.u2], dtype=np.float64))
    return tuple(models.jax.device_put(value, device) for value in arrays)


def _disc_flux(separation, radius_ratio, u_coeffs, front, *, angular_profile=None):
    """Normalized instantaneous disc flux for one occultation member.

    Quadratic limb darkening is evaluated only for the front-side member: the
    back side is replaced by an infinite separation *before* the quadrature, so
    the disc kernel cannot return a partial overlap for a geometry that cannot
    occult. A tabulated angular profile still maps the raw finite separation
    because its ray quadrature has no infinite-distance branch; callers with a
    back-side sentinel pass the raw finite geometry and the front mask
    separately, and the back-side result is discarded by that mask. Exact
    disjoint/full-disc limits replace finite quadrature values without hiding
    invalid profiles or geometry. This corrects values; skipping tabulated work
    needs a compacted active set, which this helper does not provide.

    ``separation`` is in units of the occulted body's radius and ``front``
    marks the occultor in front of it. The returned flux is 1.0 where the
    occultor is behind; geometric zero support is a finite 1.0, while a
    non-finite input propagates through the caller's validity flag.
    """
    jnp = models.jnp
    if angular_profile is None:
        delta = models.core_limb_dark(u_coeffs, jnp.where(front, separation, jnp.inf), radius_ratio)
    else:
        from trex.stellar_light.occultation import occulted_fraction

        # Map finite working sets, preserving every orbital/exposure/bin sample.
        delta = -models.jax.lax.map(
            lambda d: occulted_fraction(d, radius_ratio, *angular_profile, xp=jnp),
            separation, batch_size=32)
        limits = jnp.where(separation <= radius_ratio - 1.0, -1.0,
                           jnp.where(separation >= 1.0 + radius_ratio, 0.0, delta))
        delta = jnp.where(jnp.isfinite(delta), limits, delta)
    return jnp.where(front, 1.0 + delta, 1.0)


def _quadratic_disc_flux(time, period_days, radius_ratio, a_rs, ecc, argp, inc, u_coeffs, t0_days=0.0,
                         *, angular_profile=None):
    """Full eccentric TDB orbit with the explicit quadratic or tabulated disc.

    Orbit geometry comes from the single owner ``_jax_orbits._sky_geometry``,
    the same mean-anomaly, sky-projection and front-side convention the Monte
    Carlo paths use; the former inline copy is gone. Returned values are
    instantaneous normalized fluxes at the supplied times (exposure/bin
    integration remains the caller's contract). ``valid`` is false only for a
    non-finite geometry or flux, never for a finite geometric zero.
    """
    from trex.accelerators import _jax_orbits

    jnp = models.jnp
    separation, front = _jax_orbits._sky_geometry(time, period_days, a_rs, inc, ecc, argp, t0_days)
    flux = _disc_flux(separation, radius_ratio, u_coeffs, front, angular_profile=angular_profile)
    # A finite separation implies finite sin/cos(f) and sky coordinates here:
    # the separation is built from them, so this retains the former checks.
    valid = jnp.isfinite(separation).all() & jnp.isfinite(flux).all()
    return flux, valid


def _inverse_planet_eccentricity(probability):
    """Device form of ``priors._planet_eccentricity``, the Kipping (2013) Beta inverse.

    Cubic Hermite interpolation of logit e on logit x from exact ``betaincinv`` nodes,
    within 2e-12 relative across both tails, with the exact power law below the table.
    For 512 points, 64 incomplete-beta bisections took 5.6 s versus 1.7 ms
    for interpolation on one GPU. x = 0 is exactly e = 0;
    a probability outside [0, 1) returns NaN as a numerical failure.
    """
    jax, jnp = models.jax, models.jnp
    values, slopes = (jnp.asarray(column) for column in priors._planet_eccentricity_table())
    step, limit = priors._ECCENTRICITY_LOGIT_STEP, priors._ECCENTRICITY_LOGIT_LIMIT
    positive = jnp.where(probability > 0.0, probability, 0.5)
    logit_x = jnp.log(positive) - jnp.log1p(-positive)
    place = (jnp.maximum(logit_x, -limit) + limit) / step
    index = jnp.clip(jnp.floor(place).astype(int), 0, values.size - 2)
    t = place - index
    logit_e = ((1.0 + 2.0 * t) * (1.0 - t) ** 2 * values[index] + t * t * (3.0 - 2.0 * t) * values[index + 1]
               + step * t * (1.0 - t) * ((1.0 - t) * slopes[index] - t * slopes[index + 1]))
    tail = jnp.exp((jnp.log(positive) + priors._ECCENTRICITY_LOWER_TAIL_LOG) / priors.KIPPING_2013_ECCENTRICITY_BETA_A)
    result = jnp.where(probability > 0.0, jnp.where(logit_x < -limit, tail, jax.scipy.special.expit(logit_e)), 0.0)
    return jnp.where((probability >= 0.0) & (probability < 1.0), result, jnp.nan)


def _planet_radius_quantile(quantile, high_mass, flat):
    """Retained radius inverse CDF with a scalar or per-draw host branch."""
    jnp = models.jnp
    if flat:
        return quantile * (priors._R_MAX - priors._R_MIN) + priors._R_MIN
    values = []
    for p1, p2, p3, a1, a2, i1, i2, i3, norm in (
        (priors._P1_HI, priors._P2_HI, priors._P3_HI, priors._A1, priors._A2,
         priors._I1, priors._I2, priors._I3, priors._NORM1),
        (priors._P1_LO, priors._P2_LO, priors._P3_LO, priors._A3, priors._A4,
         priors._I4, priors._I5, priors._I6, priors._NORM2),
    ):
        first = (quantile / norm * (p1 + 1) + priors._R_MIN ** (p1 + 1)) ** (1.0 / (p1 + 1))
        second = ((quantile / norm - i1) * (p2 + 1) / a1 + priors._R_BREAK1 ** (p2 + 1)) ** (1.0 / (p2 + 1))
        third = ((quantile / norm - i1 - i2) * (p3 + 1) / (a1 * a2)
                 + priors._R_BREAK2 ** (p3 + 1)) ** (1.0 / (p3 + 1))
        values.append(jnp.where(quantile <= norm * i1, first,
                                jnp.where(quantile <= norm * (i1 + i2), second,
                                          jnp.where(quantile <= norm * (i1 + i2 + i3), third, jnp.nan))))
    return jnp.where(high_mass, values[0], values[1])


class FixedHostPlanetUnitIntegrand:
    """JIT/vmap-safe TP evaluation on one explicitly selected device.

    Construct outside JIT from the actual scalar integrand after its caller's
    photometry normalization. The caller owns device leases and batching.
    This object draws no RNG and changes no estimator/reduction policy.
    """

    claim_eligible = False

    def __init__(self, reference, *, device):
        if reference.hosted or reference.population is not None:
            raise NotImplementedError("JAX population TP integrands are not implemented")
        if reference.n_dimensions != 4:
            raise ValueError("the fixed-host TP unit cube has exactly four dimensions")
        if not models.HAS_JAX_TREX:
            raise RuntimeError("JAX/jaxoplanet is required for the native TP integrand")
        jax = models.jax
        if not jax.config.jax_enable_x64:
            raise RuntimeError("the native TP integrand requires JAX float64")
        for name in ("period_days", "host_mass", "host_radius", "a_rs", "exptime_days"):
            value = getattr(reference, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        self.device = device
        self.time, self.weights, self.flux, self.sigma, self.u_coeffs = _photometry_on_device(reference, device)
        self.baseline = (None if reference.baseline_likelihood is None
                         else reference.baseline_likelihood.on_device(device))
        self.period_days = float(reference.period_days)
        self.host_radius = float(reference.host_radius)
        self.a_rs = float(reference.a_rs)
        self.flat_radius_prior = bool(reference.use_flat_priors)
        self.high_mass_radius_prior = reference.host_mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
        self.companion_fluxratio = 0.0
        self.companion_is_host = False
        self.photon_light = None
        if getattr(reference, "stellar_light", None) is not None:
            from trex.stellar_light.native import NativeStellarLight
            self.photon_light = NativeStellarLight(reference.stellar_light, device)
            self.host_mass, self.host_teff = reference.host_mass, reference.host_teff
            self.host_metallicity, self.host_av_mag = reference.target_feh, reference.stellar_light.target_av_mag

    def _radius(self, quantile):
        """Earth-radius inverse CDF with the scalar path's coefficients/order."""
        return _planet_radius_quantile(quantile, self.high_mass_radius_prior, self.flat_radius_prior)

    def _band_low(self, radius_ratio, ecc, argp):
        jnp = models.jnp
        factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
        cos_inc_max = (1.0 + radius_ratio) / (factor * self.a_rs)
        # Intersection of geometric inclination support with [0,1], exactly
        # the scalar _transit_band_low change of variables, not clipping e.
        return 1.0 - jnp.minimum(jnp.maximum(cos_inc_max, 0.0), 1.0)

    def prior_transform(self, unit_point):
        """Map four [0,1) quantiles to R_earth, e, omega_deg, i_deg."""
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host TP unit point must have shape (4,)")
        radius = self._radius(point[0])
        ecc = _inverse_planet_eccentricity(point[1])
        argp = point[2] * priors.FULL_TURN_DEGREES
        radius_ratio = radius * Rearth / (self.host_radius * Rsun)
        band_low = self._band_low(radius_ratio, ecc, argp)
        quantile = band_low + point[3] * (1.0 - band_low)
        norm = 1.0 / (math.cos(0.0) - math.cos(math.radians(90.0)))
        inc = jnp.degrees(jnp.arccos(1.0 - quantile / norm))
        return jnp.stack((radius, ecc, argp, inc))

    def _flux_model(self, radius_ratio, ecc, argp, inc):
        """Full eccentric orbit and selected disc, no host callbacks/cutoff."""
        jnp = models.jnp
        profile, profile_valid = None, jnp.asarray(True)
        coefficients = self.u_coeffs
        if self.photon_light is not None:
            component, profile_valid = self.photon_light.prepare_component(
                self.host_teff, self.host_mass, self.host_radius, self.host_metallicity, self.host_av_mag)
            profile, coefficients = component.angular_profile, jnp.stack((component.u1, component.u2))
        instantaneous, valid = _quadratic_disc_flux(
            self.time, self.period_days, radius_ratio, self.a_rs, ecc, argp, inc, coefficients,
            angular_profile=profile,
        )
        model = jnp.sum(instantaneous.reshape(-1, self.weights.shape[-1]) * self.weights, axis=1)
        def diluted(value):
            host_fraction = self.companion_fluxratio if self.companion_is_host else 1.0 - self.companion_fluxratio
            return host_fraction * value + (1.0 - host_fraction)

        model = models.jax.lax.cond(self.companion_fluxratio > 0.0, diluted, lambda value: value, model)
        return model, valid & profile_valid & jnp.isfinite(model).all()

    def evaluate(self, unit_point):
        """Return log density/validity; structural zero is -inf, failure is NaN.

        Finite points outside [0,1) have zero prior density. Nonfinite inputs
        and failed supported model evaluations must stop/defer any future
        estimator; they must not be accumulated as zero likelihood.
        """
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host TP unit point must have shape (4,)")

        def on_cube(_unused):
            parameters = self.prior_transform(point)
            radius, ecc, argp, inc = parameters
            radius_ratio = radius * Rearth / (self.host_radius * Rsun)
            band_factor = 1.0 - self._band_low(radius_ratio, ecc, argp)
            eccentric_factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
            impact = jnp.abs(self.a_rs * jnp.cos(jnp.radians(inc)))
            supported = ((band_factor > 0.0) & (impact * eccentric_factor <= 1.0 + radius_ratio)
                         & (self.a_rs * (1.0 - ecc) >= 1.0 + radius_ratio))

            def on_support(_unused):
                model, valid = self._flux_model(radius_ratio, ecc, argp, inc)
                likelihood = (self.baseline(model) if self.baseline is not None else
                              -0.5 * jnp.sum((self.flux - model) ** 2 / self.sigma ** 2))
                value = likelihood + jnp.log(band_factor)
                valid = valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid)

            return jax.lax.cond(
                jnp.isfinite(parameters).all(),
                lambda _unused: jax.lax.cond(supported, on_support, lambda _unused: zero, None),
                lambda _unused: failure, None,
            )

        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf, dtype=jnp.float64), jnp.asarray(True))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan, dtype=jnp.float64), jnp.asarray(False))
        return jax.lax.cond(
            jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None,
        )
