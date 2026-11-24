"""Device-native joint likelihood for population-hosted/diluted scenarios.

The scalar owners are the joint branches of
:class:`trex.evidence.nested_integrands._TransitingPlanetIntegrand` and
:class:`trex.evidence.nested_integrands._EclipsingPairIntegrand` through
:class:`trex.evidence.joint_observations.ScenarioObservations`: one physical
draw is evaluated against every retained native-time aperture stream with
that stream's own band light, exposure quadrature, aperture share and
per-dataset dilution. This module widens the fixed-host device twin
(:mod:`trex.accelerators._jax_joint_unit_integrand`) to the unresolved
population slots: the bound-companion ``P*``/``S*`` families and the
background-star ``D*``/``B*`` families, so a joint scene that also produces
those slots can still run a native-only estimator instead of deferring.

Each band's atmosphere is prepared per draw from that draw's own host
(temperature, mass, radius, composition and extinction): drawn per row for
the hosted slots, the scene's fixed target for the diluting slots. The
per-dataset dilution fraction comes from the same physical draw in each
band, exactly as ``UnresolvedPopulationDraw.dataset_dilutions`` carries it
(pair ratio per band for a bound companion, native-band magnitudes for a
background star). Streams stay ragged with their own exposure response and
errors; a per-stream validity flag is reduced per stream before combining so
streams of different length are never broadcast against each other. An
exact-zero aperture share predicts flat data
in that stream and keeps the scenario. Failures return NaN with an invalid
flag; -inf marks structural zero support only.

Characterized resolved hosts (a measured stellar-parameter distribution) are
not part of this contract and keep the scalar joint likelihood. No joint
speed claim is made from this module; matched-host timing is recorded by the
parity tests' run records.

Sources and units follow the owning modules: Moe & Di Stefano (2017), ApJS
230, 15, DOI ``10.3847/1538-4365/aa6fb6``; Winters et al. (2019), AJ 157,
216, DOI ``10.3847/1538-3881/ab05dc``; Chen et al. (2019) same-distance
photon ratio; Mandel & Agol (2002), ApJ 580, L171, DOI
``10.1086/345520``; the scenario framework is Giacalone et al. (2021), AJ,
161, 24, DOI ``10.3847/1538-3881/abc6af``. Time is native barycentric TDB
days relative to the observation set's shared split reference minus the
scenario epoch [days]; flux and errors are fractional aperture flux; radii
are Earth (planet) or solar (star) units; angles are degrees; the returned
log density is in nats.
"""

from __future__ import annotations

import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.accelerators._jax_joint_unit_integrand import (
    _JointAperture,
    _aperture_prediction,
    _dilute,
    _finite,
    _positive,
)
from trex.evidence._jax_bound_unit_integrand import (
    BoundPopulationUnitIntegrand,
    _ratio_coefficients,
)
from trex.evidence._jax_eb_unit_integrand import (
    _stellar_geometry_from_splines,
    _stellar_splines_on_device,
)
from trex.evidence._jax_unit_integrand import (
    UnitIntegrandEvaluation,
    _disc_flux,
    _inverse_planet_eccentricity,
    _planet_radius_quantile,
    _quadratic_disc_flux,
)
from trex.foundation import physical_constants as physical
from trex.foundation.constants import Msun, Rearth, Rsun
from trex.models.flux_calibration import binary_flux_weights
from trex.models.orbital_mechanics import semi_major_axis_cgs
from trex.populations import priors
from trex.populations.companion_priors import companion_hierarchical_stability_supported
from trex.populations.mass_ratio_priors import broken_power_law_coefficients
from trex.stellar import stellar_relations as stellar
from trex.stellar_light.native import NativeStellarLight


def _joint_population_geometry(reference):
    """The joint contract a population twin requires before construction."""
    if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
        raise RuntimeError("the native joint likelihood requires JAX/jaxoplanet float64")
    if reference.exptime_days is not None or np.any(np.asarray(reference.bin_integration_days) != 0.0):
        raise ValueError("joint observations own their exposure responses; no boxcar or phase-bin integration")


def _transit_band_low(occultor_rs, ecc, argp, axis):
    """Lower edge of the transiting inclination quantile for one draw."""
    jnp = models.jnp
    factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
    cos_inc_max = (1.0 + occultor_rs) / (factor * axis)
    return 1.0 - jnp.minimum(jnp.maximum(cos_inc_max, 0.0), 1.0)


def _mass_ratio_inverse(quantile, floor, steep_lower, a1, a2, i1, i2, i3):
    """The EB q inverse CDF of ``FixedHostEclipsingPairUnitIntegrand._mass_ratio``.

    The fixed-host object owns single-band photometry and cannot be built
    from a joint reference, so the identical formula is restated here for the
    traced (hosted) and precomputed (diluting) coefficient tuples; parity
    against the scalar integrand is the binding check.
    """
    jnp = models.jnp
    p1, s, q_twin = (priors.MOE2017_Q_SLOPE_LOW, priors.MOE2017_Q_SLOPE_HIGH + 1.0,
                     priors.MOE2017_Q_TWIN_THRESHOLD)
    target = quantile * (i1 + i2 + i3)
    first = (target * (p1 + 1.0) + floor ** (p1 + 1.0)) ** (1.0 / (p1 + 1.0))
    second = jnp.exp(jnp.log(steep_lower) + jnp.log1p(
        (target - i1) / a1 * s / steep_lower ** s) / s)
    twin_lower = jnp.maximum(q_twin, floor)
    log_lower = jnp.where(floor > q_twin, jnp.log(twin_lower), math.log(q_twin))
    lower_power = jnp.where(floor > q_twin, twin_lower ** s, q_twin ** s)
    third = jnp.exp(log_lower + jnp.log1p(
        (target - i1 - i2) / (a1 * a2) * s / lower_power) / s)
    q = jnp.where(target <= i1, first, jnp.where(target <= i1 + i2, second, third))
    q = jnp.maximum(q, floor)
    return jnp.where(quantile == 0.0, floor, jnp.where(quantile == 1.0, 1.0, q))


def _axis_from_mass(axis_mass_solar, period_days, host_radius_solar):
    """Kepler axis in host radii from the total mass, as the fixed-host bases do."""
    return ((physical.GRAVITATIONAL_CONSTANT_CGS * axis_mass_solar * Msun
             * (period_days * physical.SECONDS_PER_DAY) ** 2) / (4.0 * math.pi ** 2)
            ) ** (1.0 / 3.0) / (host_radius_solar * Rsun)


def _inclination(band_low, quantile):
    """Inclination inside the transiting band, the scalar transform's mapping."""
    jnp = models.jnp
    value = band_low + quantile * (1.0 - band_low)
    return jnp.degrees(jnp.arccos(1.0 - value))


class _JointPopulationStreams:
    """Per-band host light resolution shared by the population joint twins.

    ``hosted`` slots prepare every band's atmosphere from the drawn host
    with the draw's own extinction; diluting slots keep the scene's fixed
    target with each band's own field extinction. The occultor models and
    per-stream validity follow the fixed-host joint twins.
    """

    def __init__(self, joint, *, hosted, target_teff, target_mass, target_radius,
                 target_feh, period_days, stellar_splines, device):
        self.joint = _JointAperture(joint, device)
        self.band_lights = tuple(
            NativeStellarLight(joint.band_light[band], device) for band in self.joint.bands)
        self.hosted = bool(hosted)
        self.period_days = float(period_days)
        self.stellar_splines = stellar_splines
        if hosted:
            # The drawn row supplies the host; no fixed target is needed.
            self.target_teff = self.target_mass = self.target_radius = None
            self.target_feh = None
        else:
            self.target_teff = _positive(target_teff, "target_teff")
            self.target_mass = _positive(target_mass, "target_mass")
            self.target_radius = _positive(target_radius, "target_radius")
            self.target_feh = _finite(target_feh, "target_feh")

    def _host_component(self, band_light, host_teff, host_mass, host_radius,
                        host_metallicity, host_av):
        if self.hosted:
            return band_light.prepare_component(
                host_teff, host_mass, host_radius, host_metallicity, host_av)
        # Diluting slots keep the target host; each band's own field extinction.
        return band_light.prepare_component(
            self.target_teff, self.target_mass, self.target_radius,
            self.target_feh, band_light.target_av_mag)

    def predictions(self, *, planet, parameters, axis, host_teff, host_mass,
                    host_radius, host_metallicity, host_av, fractions):
        """Per-stream aperture predictions with each band's own light and share."""
        jnp = models.jnp
        predictions, valid = [], jnp.asarray(True)
        for grid, weights, share, band_light, fraction in zip(
                self.joint.grids, self.joint.weights, self.joint.shares,
                self.band_lights, fractions):
            first, first_valid = self._host_component(
                band_light, host_teff, host_mass, host_radius, host_metallicity, host_av)
            # The scalar joint owner resolves a missing host A_V through each
            # band's own field extinction (ScenarioObservations.evaluate).
            secondary_av = host_av if host_av is not None else band_light.target_av_mag
            if planet:
                radius_earth, ecc, argp, inc = parameters
                k = radius_earth * Rearth / (host_radius * Rsun)
                instantaneous, geometry_valid = _quadratic_disc_flux(
                    grid, self.period_days, k, axis, ecc, argp, inc,
                    jnp.stack((first.u1, first.u2)), angular_profile=first.angular_profile)
                intrinsic = jnp.sum(
                    instantaneous.reshape(-1, weights.shape[-1]) * weights, axis=1)
                intrinsic = _dilute(intrinsic, fraction, self.hosted)
                light_valid = first_valid & geometry_valid & jnp.isfinite(intrinsic).all()
            else:
                from trex.accelerators._jax_orbits import _sky_geometry

                q, ecc, argp, inc = parameters
                secondary_mass = q * host_mass
                secondary_radius, secondary_teff = _stellar_geometry_from_splines(
                    self.stellar_splines, secondary_mass)
                second, second_valid = band_light.prepare_component(
                    secondary_teff, secondary_mass, secondary_radius, host_metallicity, secondary_av)
                # Chen 2019 same-distance photon ratio; extinction is in both surfaces.
                ratio = (secondary_radius / host_radius) ** 2 * (
                    second.surface_photon_flux / first.surface_photon_flux)
                separation, front = _sky_geometry(
                    grid, self.period_days, axis, inc, ecc, argp, 0.0)
                primary_instant = _disc_flux(
                    separation, secondary_radius / host_radius,
                    jnp.stack((first.u1, first.u2)), front,
                    angular_profile=first.angular_profile)
                secondary_instant = _disc_flux(
                    separation / (secondary_radius / host_radius),
                    host_radius / secondary_radius, jnp.stack((second.u1, second.u2)),
                    ~front, angular_profile=second.angular_profile)
                primary = jnp.sum(
                    primary_instant.reshape(-1, weights.shape[-1]) * weights, axis=1)
                secondary = jnp.sum(
                    secondary_instant.reshape(-1, weights.shape[-1]) * weights, axis=1)
                light_weights = binary_flux_weights(
                    ratio, fraction, self.hosted, array_module=jnp)
                intrinsic = (light_weights[0] * primary
                             + light_weights[1] * secondary + light_weights[2])
                light_valid = (first_valid & second_valid & jnp.isfinite(separation).all()
                               & jnp.isfinite(primary_instant).all()
                               & jnp.isfinite(secondary_instant).all()
                               & jnp.isfinite(intrinsic).all())
            prediction, active = _aperture_prediction(intrinsic, share)
            predictions.append(prediction)
            # Streams have different lengths: reduce each stream's cadence
            # validity to one flag before combining, never broadcast across streams.
            valid = valid & (jnp.all(light_valid) | ~active)
        return predictions, valid


class JointBoundPopulationUnitIntegrand(BoundPopulationUnitIntegrand):
    """Joint ``P*``/``S*`` density: one companion draw feeding every band's stream.

    The population draw, retention mask, occurrence weight and imaging
    convention are the single-band bound twin's (``_prepare_bound_population``
    and ``_resolve``); the likelihood reproduces the scalar joint owner. The
    per-dataset dilution is the same physical pair evaluated in each band's
    atmosphere with the population light's A_V, exactly as
    ``_BoundCompanionPopulation._band_pair_ratio`` computes it, and the
    hosted atmosphere uses that draw's A_V in every band.
    """

    claim_eligible = False

    def __init__(self, reference, *, device, planet):
        joint = getattr(reference, "joint_observations", None)
        if joint is None:
            raise ValueError("the native joint population callback requires joint_observations")
        _joint_population_geometry(reference)
        if reference.target_feh is None:
            raise ValueError("joint bound scenarios require the scene metallicity")
        self._prepare_bound_population(reference, device=device, planet=planet)
        if self.population.target_feh is None:
            raise ValueError("joint bound pair ratios require the block's scene metallicity")
        self.planet = bool(planet)
        self.period_days = _positive(
            reference.period_days if planet else reference.effective_period,
            "period_days" if planet else "effective_period")
        self.use_2xP = bool(getattr(reference, "use_2xP", False))
        self.use_flat_priors = bool(getattr(reference, "use_flat_priors", False))
        if not planet:
            self.eccentricity_power = (
                priors.MOE2017_ECCENTRICITY_POWERLAW_SHORT
                if self.period_days <= priors.MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS
                else 1.0 + priors.moe2017_binary_eccentricity_eta(self.period_days))
        # The scalar per-dataset pair ratio and the hosted joint atmosphere
        # both use the population light's A_V in every band
        # (nested_population._band_pair_ratio, ScenarioObservations.evaluate
        # with the draw's own av_mag).
        self.field_av_mag = self.population.stellar_light.target_av_mag
        # The scalar joint owner resolves the diluting target's atmosphere,
        # geometry and q support from the integrand's own host fields, while
        # the population block's host fields feed the pair ratios; production
        # passes identical values, and each owner keeps its own source here.
        self.reference_host_teff = (
            _positive(reference.host_teff, "host_teff") if not self.hosted else None)
        self.reference_host_mass = _positive(reference.host_mass, "host_mass")
        self.reference_host_radius = _positive(reference.host_radius, "host_radius")
        self.stellar_splines = _stellar_splines_on_device(device)
        self.streams = _JointPopulationStreams(
            joint, hosted=self.hosted, target_teff=self.reference_host_teff,
            target_mass=self.reference_host_mass, target_radius=self.reference_host_radius,
            target_feh=self.host_metallicity, period_days=self.period_days,
            stellar_splines=self.stellar_splines, device=device)
        if not self.hosted:
            # Diluting slots keep the target's fixed geometry and q support,
            # exactly as the single-band bound twin's base object does.
            self.fixed_axis = semi_major_axis_cgs(
                self.period_days, self.reference_host_mass * Msun) / (
                self.reference_host_radius * Rsun)
            floor = priors._stellar_mass_ratio_floor(self.reference_host_mass)
            if floor >= 1.0:
                raise ValueError("Moe mass-ratio prior has no support for this host")
            coefficients = broken_power_law_coefficients(
                np.array([floor]), np.array([priors.MOE2017_Q_SLOPE_HIGH]),
                np.array([priors.MOE2017_TWIN_FRACTION]))
            (steep_lower, _twin_lower, a1, a2, i1, i2, i3) = (
                float(value[0]) for value in coefficients)
            self.fixed_q_coefficients = (floor, steep_lower, a1, a2, i1, i2, i3)

    def _band_fractions(self, teff, mass, radius):
        """Per-dataset dilution of the same physical pair, and its validity."""
        jnp = models.jnp
        fractions, valid = [], jnp.asarray(True)
        for band_light in self.streams.band_lights:
            # The metallicity is the block's own scene value, the exact source
            # the scalar _band_pair_ratio feeds the per-dataset pair ratio.
            ratio = band_light.pair_ratio(
                self.population.host_teff, self.population.host_mass,
                self.population.host_radius, teff, mass, radius,
                self.population.target_feh, self.field_av_mag)
            fractions.append(jnp.where(jnp.isposinf(ratio), 1.0, ratio / (1.0 + ratio)))
            valid = valid & (jnp.isfinite(ratio) | jnp.isposinf(ratio))
        return tuple(fractions), valid

    def _orbit(self, point):
        """The draw's parameters and every support term, unconditionally.

        The parameter half mirrors the scalar transform (finite fallback host
        coordinates on unsupported draws); the support half mirrors the
        scalar likelihood's masks so ``prior_transform`` and ``evaluate``
        consume one population inverse-CDF implementation.
        """
        jnp = models.jnp
        mass, radius, teff, _fraction, support, limit, log_period, contrast = self._resolve(point)
        fractions, fraction_valid = self._band_fractions(teff, mass, radius)
        support = support & fraction_valid
        host_mass = mass if self.hosted else self.reference_host_mass
        host_radius = radius if self.hosted else self.reference_host_radius
        host_teff = teff if self.hosted else self.reference_host_teff
        if self.planet:
            high_mass = host_mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
            radius_earth = _planet_radius_quantile(point[0], high_mass, self.use_flat_priors)
            ecc = _inverse_planet_eccentricity(point[1])
            argp = point[2] * physical.FULL_TURN_DEGREES
            k = radius_earth * Rearth / (host_radius * Rsun)
            axis = (_axis_from_mass(host_mass, self.period_days, host_radius)
                    if self.hosted else self.fixed_axis)
            band_low = _transit_band_low(k, ecc, argp, axis)
            inc = _inclination(band_low, point[3])
            parameters = jnp.stack((radius_earth, ecc, argp, inc))
            inner_binary_mass, outer_mass = (
                (host_mass, self.reference_host_mass) if self.hosted
                else (self.reference_host_mass, mass))
        else:
            if self.hosted:
                # The trailing exponent (SLOPE_HIGH + 1) is the one _mass_ratio_inverse fixes itself.
                coefficients = _ratio_coefficients(
                    jnp.maximum(host_mass, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR),
                    priors.MOE2017_Q_SLOPE_HIGH, priors.MOE2017_TWIN_FRACTION)[:7]
                support = support & (host_mass >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
            else:
                coefficients = self.fixed_q_coefficients
            floor, steep_lower, a1, a2, i1, i2, i3 = coefficients
            q = _mass_ratio_inverse(point[0], floor, steep_lower, a1, a2, i1, i2, i3)
            ecc = point[1] ** (1.0 / self.eccentricity_power)
            argp = point[2] * physical.FULL_TURN_DEGREES
            support = support & (
                (q >= priors.MOE2017_Q_TWIN_THRESHOLD) if self.use_2xP
                else (q < priors.MOE2017_Q_TWIN_THRESHOLD))
            secondary_mass = q * host_mass
            support = support & (
                (secondary_mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR)
                & (secondary_mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR))
            secondary_radius, secondary_teff = _stellar_geometry_from_splines(
                self.stellar_splines, secondary_mass)
            # Same transform-only fallback as the scalar EB: the likelihood
            # rejects the draw; the returned point stays finite.
            k = jnp.where(jnp.isfinite(secondary_radius), secondary_radius, host_radius) / host_radius
            axis = _axis_from_mass(host_mass * (1.0 + q), self.period_days, host_radius)
            band_low = _transit_band_low(k, ecc, argp, axis)
            inc = _inclination(band_low, point[3])
            parameters = jnp.stack((q, ecc, argp, inc))
            # Imaging support uses the pair-corrected contrast in log light
            # (the single-band bound twin's convention) with the original
            # secondary geometry, never the transform fallback.
            secondary_ratio = self.photon_light.pair_ratio(
                host_teff, host_mass, host_radius, secondary_teff, secondary_mass,
                secondary_radius, self.host_metallicity, self.field_av_mag)
            correction = jnp.log1p(secondary_ratio) / (
                jnp.log(10.0) * physical.POGSON_FLUX_EXPONENT)
            pair_contrast = contrast + (correction if self.hosted else -correction)
            fainter = (pair_contrast >= 0.0) | ~jnp.isfinite(pair_contrast)
            support = support & fainter & ~(jnp.isfinite(limit) & (pair_contrast < limit))
            inner_binary_mass = (host_mass if self.hosted else self.reference_host_mass) * (1.0 + q)
            outer_mass = self.reference_host_mass if self.hosted else mass
        # Shared Mardling & Aarseth (2001) Eq. 90 mask against the observed
        # period: the inner binary is the event host (plus its eclipsing
        # secondary on the EB slots); the drawn companion is the outer body
        # on the diluting slots and the target otherwise.
        support = support & companion_hierarchical_stability_supported(
            log_period, self.period_days, inner_binary_mass, outer_mass, array_module=jnp)
        return parameters, support, k, axis, ecc, argp, inc, host_teff, host_mass, host_radius, fractions

    def prior_transform(self, unit_point):
        """Map the unit cube to the scalar integrand's parameter vector."""
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError(f"joint bound population point must have shape ({self.n_dimensions},)")
        parameters, *_rest = self._orbit(point)
        return jnp.concatenate((parameters, point[4:]))

    def evaluate(self, unit_point):
        """Return the joint log density and numerical validity on the unit cube."""
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError(f"joint bound population point must have shape ({self.n_dimensions},)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf, dtype=jnp.float64), jnp.asarray(True))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan, dtype=jnp.float64), jnp.asarray(False))

        def on_cube(_unused):
            (parameters, support, k, axis, ecc, argp, inc, host_teff, host_mass,
             host_radius, fractions) = self._orbit(point)
            band_factor = 1.0 - _transit_band_low(k, ecc, argp, axis)
            eccentric_factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
            impact = jnp.abs(axis * jnp.cos(jnp.radians(inc)))
            support = support & ((band_factor > 0.0)
                                 & (impact * eccentric_factor <= 1.0 + k)
                                 & (axis * (1.0 - ecc) >= 1.0 + k))

            def on_support(_unused):
                predictions, valid = self.streams.predictions(
                    planet=self.planet, parameters=parameters, axis=axis,
                    host_teff=host_teff, host_mass=host_mass, host_radius=host_radius,
                    host_metallicity=self.host_metallicity,
                    host_av=self.field_av_mag if self.hosted else None,
                    fractions=fractions)
                value = self.streams.joint.density(predictions) + jnp.log(band_factor)
                valid = valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid)

            return jax.lax.cond(
                jnp.isfinite(parameters).all(),
                lambda _unused: jax.lax.cond(support, on_support, lambda _unused: zero, None),
                lambda _unused: failure, None,
            )

        return jax.lax.cond(
            jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(
                jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None,
        )


class JointBackgroundPopulationUnitIntegrand:
    """Joint ``D*``/``B*`` density: one drawn background row per band's stream.

    Every retained population row stays present; the discrete uniform table
    measure and uniform-area separation are the single-band background
    twin's. The per-dataset dilution is the drawn star's native-band
    magnitude against that dataset's target magnitude, precomputed per row;
    the hosted atmosphere is that row's own temperature, composition and
    extinction. Background stars are not bound: no hierarchical-stability
    mask applies.
    """

    claim_eligible = False

    def __init__(self, reference, *, device, planet):
        joint = getattr(reference, "joint_observations", None)
        if joint is None:
            raise ValueError("the native joint population callback requires joint_observations")
        _joint_population_geometry(reference)
        population = reference.population
        if population.n_dimensions != 2 or reference.n_dimensions != 6:
            raise ValueError("joint background population requires four orbit and two population dimensions")
        if not population.datasets:
            raise ValueError("joint background scenarios require dataset-bound population magnitudes")
        rows = joint.likelihood.observations.observations
        if tuple(dataset.dataset_id for dataset in population.datasets) != tuple(
                row.dataset_id for row in rows):
            raise ValueError("background population datasets must match the observation set order")
        self.device = device
        self.planet = bool(planet)
        self.hosted = bool(reference.hosted)
        self.period_days = _positive(
            reference.period_days if planet else reference.effective_period,
            "period_days" if planet else "effective_period")
        self.use_2xP = bool(getattr(reference, "use_2xP", False))
        self.use_flat_priors = bool(getattr(reference, "use_flat_priors", False))
        if not planet:
            self.eccentricity_power = (
                priors.MOE2017_ECCENTRICITY_POWERLAW_SHORT
                if self.period_days <= priors.MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS
                else 1.0 + priors.moe2017_binary_eccentricity_eta(self.period_days))
        self.log_occurrence_weight = float(population.log_occurrence_weight)
        self.field_radius_arcsec = float(population.field_radius_arcsec)
        selection = np.asarray(population.selection_mag, dtype=np.float64)
        masses = np.asarray(population.mass_solar, dtype=np.float64)
        radii = np.asarray(population.radius_solar, dtype=np.float64)
        temperatures = np.asarray(population.teff_k, dtype=np.float64)
        metallicity = np.asarray(population.mh_dex, dtype=np.float64)
        if (selection.ndim != 1 or not selection.size or any(
                values.shape != selection.shape for values in (
                    masses, radii, temperatures, metallicity))):
            raise ValueError("background population columns must be matching one-dimensional arrays")
        if any(not np.isfinite(values).all() or np.any(values <= 0.0)
               for values in (masses, radii, temperatures)):
            raise ValueError("background masses, radii and temperatures must be finite and positive")
        self.star_count = int(selection.size)
        # The imaging contrast is the retained selection-band magnitude
        # difference, never a fraction-to-ratio reconstruction.
        contrasts = selection - float(population.selection_depth_mag)
        fractions, fraction_valid = [], []
        for dataset in population.datasets:
            values = np.asarray(population.band_magnitudes[dataset.band], dtype=np.float64)
            valid = np.asarray(population.band_valid[dataset.band], dtype=bool)
            if values.shape != selection.shape or valid.shape != selection.shape:
                raise ValueError("background band magnitudes must align with the retained stars")
            delta = values - float(dataset.target_magnitude)
            with np.errstate(over="ignore"):
                ratio = np.power(10.0, physical.POGSON_FLUX_EXPONENT * delta)
            fractions.append(np.where(np.isposinf(ratio), 1.0, ratio / (1.0 + ratio)))
            fraction_valid.append(valid & np.isfinite(values))
        if self.hosted:
            if population.av_mag is None:
                raise ValueError("hosted joint background scenarios require retained A_V per row")
            extinction = np.asarray(population.av_mag, dtype=np.float64)
            if extinction.shape != selection.shape or not np.isfinite(extinction).all():
                raise ValueError("background A_V must be finite and row aligned")
            axes = semi_major_axis_cgs(self.period_days, masses * Msun) / (radii * Rsun)
            high_mass = masses > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
            q_masses = masses
            domain = masses >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
        else:
            if reference.host_teff is None or reference.target_feh is None:
                raise ValueError("diluting joint background scenarios require the scene host atmosphere")
            extinction = None
            axes = np.full(selection.shape, semi_major_axis_cgs(
                self.period_days, reference.host_mass * Msun) / (reference.host_radius * Rsun))
            high_mass = np.full(selection.shape, bool(
                reference.host_mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR))
            q_masses = np.full(selection.shape, float(reference.host_mass))
            domain = np.full(selection.shape, bool(
                reference.host_mass >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR))
        # The q prior's domain floor applies to the drawn row and to a fixed
        # target alike, exactly as the single-band background twin computes it.
        q_masses = np.maximum(q_masses, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
        q_floor = priors._stellar_mass_ratio_floors(q_masses)
        coefficients = broken_power_law_coefficients(
            q_floor, np.full(q_floor.shape, priors.MOE2017_Q_SLOPE_HIGH),
            np.full(q_floor.shape, priors.MOE2017_TWIN_FRACTION))
        (_lower, _twin_lower, a1, a2, i1, i2, i3) = coefficients
        self.rows = tuple(models.jax.device_put(value, device) for value in (
            masses, radii, temperatures, metallicity,
            np.zeros(1) if extinction is None else extinction, axes, high_mass, contrasts,
            q_floor, _lower, a1, a2, i1, i2, i3, domain))
        self.fractions = tuple(models.jax.device_put(value, device) for value in fractions)
        self.fraction_valid = tuple(models.jax.device_put(value, device) for value in fraction_valid)
        self.photon_light = NativeStellarLight(reference.stellar_light, device)
        self.stellar_splines = _stellar_splines_on_device(device)
        self.streams = _JointPopulationStreams(
            joint, hosted=self.hosted, target_teff=reference.host_teff,
            target_mass=reference.host_mass, target_radius=reference.host_radius,
            target_feh=reference.target_feh, period_days=self.period_days,
            stellar_splines=self.stellar_splines, device=device)
        self.measured_curve = bool(population.measured_curve)
        if self.measured_curve:
            curve = population.contrast_curve
            self.curve = tuple(models.jax.device_put(value, device) for value in (
                curve.separations, curve.contrasts))

    def _index(self, point):
        # Row of the discrete uniform table measure; the clip only states the
        # [0, star_count) range that u < 1 already implies in float64.
        jnp = models.jnp
        return jnp.minimum(jnp.floor(point[4] * self.star_count), self.star_count - 1).astype(jnp.int32)

    def _orbit(self, point):
        """The drawn row's parameters and support terms, unconditionally."""
        jnp = models.jnp
        index = self._index(point)
        (masses, radii, temperatures, metallicity, extinction, axes, high_mass,
         contrasts, q_floor, steep_lower, a1, a2, i1, i2, i3, domain) = self.rows
        support = jnp.asarray(True)
        for valid in self.fraction_valid:
            support = support & valid[index]
        if self.hosted:
            host_mass, host_radius, host_teff = masses[index], radii[index], temperatures[index]
            host_metallicity, host_av = metallicity[index], extinction[index]
        else:
            host_mass, host_radius, host_teff = (
                self.streams.target_mass, self.streams.target_radius, self.streams.target_teff)
            host_metallicity, host_av = self.streams.target_feh, None
        if self.planet:
            radius_earth = _planet_radius_quantile(point[0], high_mass[index], self.use_flat_priors)
            ecc = _inverse_planet_eccentricity(point[1])
            argp = point[2] * physical.FULL_TURN_DEGREES
            k = radius_earth * Rearth / (host_radius * Rsun)
            axis = axes[index]
            band_low = _transit_band_low(k, ecc, argp, axis)
            inc = _inclination(band_low, point[3])
            parameters = jnp.stack((radius_earth, ecc, argp, inc))
            # A planet host adds no secondary light; the retained magnitude
            # difference is the whole imaging contrast.
            pair_contrast = contrasts[index]
        else:
            support = support & domain[index]
            q = _mass_ratio_inverse(point[0], q_floor[index], steep_lower[index],
                                    a1[index], a2[index], i1[index], i2[index], i3[index])
            ecc = point[1] ** (1.0 / self.eccentricity_power)
            argp = point[2] * physical.FULL_TURN_DEGREES
            support = support & (
                (q >= priors.MOE2017_Q_TWIN_THRESHOLD) if self.use_2xP
                else (q < priors.MOE2017_Q_TWIN_THRESHOLD))
            secondary_mass = q * host_mass
            support = support & (
                (secondary_mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR)
                & (secondary_mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR))
            secondary_radius, secondary_teff = _stellar_geometry_from_splines(
                self.stellar_splines, secondary_mass)
            # Same transform-only fallback as the scalar EB when the secondary
            # leaves the stellar relation; the likelihood rejects the draw.
            k = jnp.where(jnp.isfinite(secondary_radius), secondary_radius, host_radius) / host_radius
            axis = _axis_from_mass(host_mass * (1.0 + q), self.period_days, host_radius)
            band_low = _transit_band_low(k, ecc, argp, axis)
            inc = _inclination(band_low, point[3])
            parameters = jnp.stack((q, ecc, argp, inc))
            # The imaging contrast is the row's retained magnitude difference,
            # pair-corrected on the EB slots as the single-band background
            # twin does; no fainter requirement applies to background stars.
            secondary_ratio = self.photon_light.pair_ratio(
                host_teff, host_mass, host_radius, secondary_teff, secondary_mass,
                secondary_radius, host_metallicity,
                host_av if self.hosted else self.photon_light.target_av_mag)
            correction = jnp.log1p(secondary_ratio) / (
                jnp.log(10.0) * physical.POGSON_FLUX_EXPONENT)
            pair_contrast = contrasts[index] + (correction if self.hosted else -correction)
        if self.measured_curve:
            separations, values = self.curve
            separation = self.field_radius_arcsec * jnp.sqrt(jnp.maximum(point[5], 0.0))
            limit = jnp.interp(separation, separations, values, left=jnp.nan, right=jnp.nan)
            support = support & ~(jnp.isfinite(limit) & (pair_contrast < limit))
        return parameters, support, k, axis, ecc, argp, inc, host_teff, host_mass, host_radius, host_metallicity, host_av

    def prior_transform(self, unit_point):
        """Map the unit cube to the scalar integrand's parameter vector."""
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("joint background population point must have shape (6,)")
        parameters, *_rest = self._orbit(point)
        return jnp.concatenate((parameters, point[4:]))

    def evaluate(self, unit_point):
        """Return the joint log density and numerical validity on the unit cube."""
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("joint background population point must have shape (6,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf, dtype=jnp.float64), jnp.asarray(True))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan, dtype=jnp.float64), jnp.asarray(False))

        def on_cube(_unused):
            (parameters, support, k, axis, ecc, argp, inc, host_teff, host_mass,
             host_radius, host_metallicity, host_av) = self._orbit(point)
            band_factor = 1.0 - _transit_band_low(k, ecc, argp, axis)
            eccentric_factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
            impact = jnp.abs(axis * jnp.cos(jnp.radians(inc)))
            support = support & ((band_factor > 0.0)
                                 & (impact * eccentric_factor <= 1.0 + k)
                                 & (axis * (1.0 - ecc) >= 1.0 + k))

            def on_support(_unused):
                index = self._index(point)
                fractions = tuple(fraction[index] for fraction in self.fractions)
                predictions, valid = self.streams.predictions(
                    planet=self.planet, parameters=parameters, axis=axis,
                    host_teff=host_teff, host_mass=host_mass, host_radius=host_radius,
                    host_metallicity=host_metallicity,
                    host_av=host_av if self.hosted else None,
                    fractions=fractions)
                value = self.streams.joint.density(predictions) + jnp.log(band_factor)
                valid = valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid)

            return jax.lax.cond(
                jnp.isfinite(parameters).all(),
                lambda _unused: jax.lax.cond(support, on_support, lambda _unused: zero, None),
                lambda _unused: failure, None,
            )

        return jax.lax.cond(
            jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(
                jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None,
        )
