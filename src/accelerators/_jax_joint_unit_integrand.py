"""Device-native joint aperture likelihood for the joint scenario integrands.

The scalar owners are :class:`trex.models.joint_likelihood.JointLikelihood` and
:class:`trex.evidence.joint_observations.ScenarioObservations`: one physical
draw is evaluated against every retained native-time aperture stream with that
stream's own band light, exposure quadrature, aperture share and dilution. This
module is the device twin for the fixed-host transiting-planet and
eclipsing-pair scenarios, so defensive importance refinement and the native
nested estimator can evaluate the same joint density as a batched float64
kernel instead of one scalar draw at a time. A single-band callback is never
reused for joint data.

Time is native barycentric TDB days relative to the observation set's shared
split reference minus the scenario epoch [days]; flux and errors are fractional
aperture flux; radii/axis/period are days, solar radii and Earth radii as
named; angles are degrees; the returned log density is in nats. Streams stay
ragged (their own length, exposure response, errors and photometry) and are
never restacked onto a synthetic cadence. An exact-zero aperture share predicts
flat data in that stream and keeps the scenario. Failures return NaN with an
explicit validity flag; -inf marks structural zero support only.

Only the exactly specified cases are admitted: a fixed scenario host (the
target or a resolved neighbor) with one explicit physical StellarLight per
band, or one of the unresolved population slots through the sibling module
:mod:`trex.accelerators._jax_joint_population_integrand` (bound-companion
``P*``/``S*`` and background-star ``D*``/``B*`` families). Every other joint
configuration — a characterized host distribution above all — returns no
callback and keeps the scalar ``JointLikelihood`` path, so a scenario the
parity does not cover cannot run a single-band model.

Sources: Mandel & Agol (2002), ApJ 580, L171, DOI ``10.1086/345520``;
Agol et al. (2020), AJ 159, 123, DOI ``10.3847/1538-3881/ab4fee``; Chen et al.
(2019) same-distance photon ratio for the binary pair (the retained
``StellarLight.prepare_pair`` contract); Gibson (2014), MNRAS 445, 3401,
DOI ``10.1093/mnras/stu1975`` for the marginal linear-baseline identity the
scalar owner already implements.
"""

from __future__ import annotations

import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.evidence._jax_unit_integrand import (
    FixedHostPlanetUnitIntegrand,
    UnitIntegrandEvaluation,
    _disc_flux,
    _quadratic_disc_flux,
)
from trex.foundation.constants import Rearth, Rsun
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.flux_calibration import binary_flux_weights
from trex.populations import priors
from trex.stellar_light.native import NativeStellarLight

if models.HAS_JAX_TREX:
    # The EB prior/geometry owner needs the jaxoplanet kernel; without the
    # float64 backend the callback factory refuses before any construction.
    from trex.evidence._jax_eb_unit_integrand import (
        FixedHostEclipsingPairUnitIntegrand as _FixedHostEclipsingPair,
    )
else:  # pragma: no cover - exercised only on hosts without JAX/jaxoplanet
    _FixedHostEclipsingPair = object


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite")
    return float(value)


class _JointAperture:
    """Ragged device streams and the scalar owner's normalized joint density.

    ``JointLikelihood.density`` sums one normalized Gaussian per dataset, or
    replaces that sum by one shared Gaussian baseline marginalized once over the
    dataset-ordered predictions. Both variants are reproduced here from the
    same prepared objects: reported errors/normalizations, each dataset's own
    or the one shared proper prior, and the same dataset order.
    """

    def __init__(self, joint, device):
        from trex.evidence.joint_observations import ScenarioObservations

        if not isinstance(joint, ScenarioObservations):
            raise ValueError("native joint likelihood requires a prepared ScenarioObservations")
        jax = models.jax
        likelihood = joint.likelihood
        rows = likelihood.observations.observations
        grids, weights, fluxes, sigmas, shares, baselines, normalizations, bands = (
            [], [], [], [], [], [], [], [])
        for row, time, baseline, normalization in zip(
                rows, likelihood.times, likelihood.baselines, likelihood.normalizations):
            share = row.source_shares.get(joint.host_id)
            if share is None:
                raise TrexEvidenceUnavailableError(
                    f"source {joint.host_id!r} has unknown aperture light in dataset {row.dataset_id!r}")
            response = row.exposure_response
            offsets = np.asarray(response.offset_days, dtype=np.float64)
            response_weights = np.asarray(response.weights, dtype=np.float64)
            times = np.asarray(time, dtype=np.float64) - joint.epoch_days
            if times.ndim != 1 or times.size != row.flux.size or not np.isfinite(times).all():
                raise ValueError("joint stream times must be the finite retained native TDB rows")
            grid = times[:, None] + (offsets if offsets.ndim == 2 else offsets[None, :])
            if not np.isfinite(grid).all():
                raise ValueError("joint exposure quadrature times must remain finite")
            grids.append(jax.device_put(grid.reshape(-1), device))
            weights.append(jax.device_put(response_weights, device))
            fluxes.append(jax.device_put(np.asarray(row.flux, dtype=np.float64), device))
            sigmas.append(jax.device_put(np.asarray(row.sigma, dtype=np.float64), device))
            shares.append(jax.device_put(np.asarray(share, dtype=np.float64), device))
            baselines.append(None if baseline is None else baseline.on_device(device))
            normalizations.append(jax.device_put(np.asarray(normalization, dtype=np.float64), device))
            bands.append(row.band_id)
        self.device = device
        self.bands = tuple(bands)
        self.band_light = dict(joint.band_light)
        self.grids = tuple(grids)
        self.weights = tuple(weights)
        self.fluxes = tuple(fluxes)
        self.sigmas = tuple(sigmas)
        self.shares = tuple(shares)
        self.baselines = tuple(baselines)
        self.normalizations = tuple(normalizations)
        self.shared = None if likelihood.shared is None else likelihood.shared.on_device(device)

    def density(self, predictions):
        """Normalized joint log density of one aperture prediction per stream."""
        jnp = models.jnp
        if len(predictions) != len(self.fluxes):
            raise ValueError("joint density requires exactly one prediction per admitted stream")
        if self.shared is not None:
            return self.shared(jnp.concatenate(tuple(predictions)))
        total = jnp.asarray(0.0, dtype=jnp.float64)
        for prediction, flux, sigma, baseline, normalization in zip(
                predictions, self.fluxes, self.sigmas, self.baselines, self.normalizations):
            if baseline is not None:
                total = total + baseline(prediction)
            else:
                residual = (flux - prediction) / sigma
                total = total + (-0.5 * (jnp.sum(residual * residual) + normalization))
        return total


def _dilute(model, fraction, companion_is_host):
    """The scalar ``dilute_flux`` blend: the event host's own aperture share."""
    jnp = models.jnp
    host = fraction if companion_is_host else 1.0 - fraction
    return jnp.where(fraction > 0.0, host * model + (1.0 - host), model)


def _aperture_prediction(intrinsic, share):
    """Map one host-frame model to its aperture prediction and active flag.

    ``JointLikelihood.models`` uses ``1 + share * (intrinsic - 1)`` and skips a
    source with an exactly-zero aperture share, which predicts flat data rather
    than deleting the scenario. An inactive stream must therefore not turn a
    nonfinite intrinsic model into a numerical failure.
    """
    jnp = models.jnp
    active = jnp.any(share > 0.0)
    prediction = jnp.where(active, 1.0 + share * (intrinsic - 1.0), jnp.ones_like(intrinsic))
    return prediction, active


def _joint_native_geometry(reference):
    """Validate the fixed-host joint contract shared by the TP/EB callbacks."""
    if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
        raise RuntimeError("the native joint likelihood requires JAX/jaxoplanet float64")
    if reference.n_dimensions != 4:
        raise ValueError("the fixed-host joint unit cube has exactly four dimensions")
    if reference.exptime_days is not None or np.any(np.asarray(reference.bin_integration_days) != 0.0):
        raise ValueError("joint observations own their exposure responses; no boxcar or phase-bin integration")
    if reference.host_teff is None or reference.target_feh is None:
        raise ValueError("joint physical light requires this host's own temperature and composition")


def _joint_band_lights(aperture):
    """The one physical atmosphere per admitted band, in dataset order."""
    lights = []
    for band in aperture.bands:
        light = aperture.band_light.get(band)
        if light is None:
            raise TrexEvidenceUnavailableError(f"joint scenario has no physical atmosphere for band {band!r}")
        if light.target_av_mag is None:
            raise TrexEvidenceUnavailableError("joint physical light requires this host's own A_V [mag]")
        lights.append(light)
    return tuple(lights)


class JointPlanetUnitIntegrand(FixedHostPlanetUnitIntegrand):
    """Joint fixed-host TP density: one orbit, every band's own light and share.

    Reuses the fixed-host prior transform and geometric support of the scalar
    nested integrand, then evaluates the per-band instantaneous model on each
    stream's own exposure quadrature. The prior's radius/eccentricity/inclination
    draw is shared, so the band fluxes describe one physical transit.
    """

    claim_eligible = False

    def __init__(self, reference, device):
        joint = getattr(reference, "joint_observations", None)
        if joint is None:
            raise ValueError("the native joint TP callback requires joint_observations")
        if reference.population is not None or getattr(reference, "hosted", False):
            raise NotImplementedError("the native joint TP callback covers fixed hosts only")
        _joint_native_geometry(reference)
        self.device = device
        self.period_days = _positive(reference.period_days, "period_days")
        self.host_mass = _positive(reference.host_mass, "host_mass")
        self.host_radius = _positive(reference.host_radius, "host_radius")
        self.a_rs = _positive(reference.a_rs, "a_rs")
        self.host_teff = _positive(reference.host_teff, "host_teff")
        self.host_metallicity = _finite(reference.target_feh, "target_feh")
        self.flat_radius_prior = bool(reference.use_flat_priors)
        self.high_mass_radius_prior = self.host_mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
        # Fixed-host joint scenarios carry no unresolved dilutor; the scalar
        # ``simulate_TP`` receives the same zero fraction and host flag.
        self.companion_fluxratio = 0.0
        self.companion_is_host = False
        self.joint = _JointAperture(joint, device)
        self.lights = _joint_band_lights(self.joint)
        components = []
        for light in self.lights:
            component, valid = NativeStellarLight(light, device).prepare_component(
                self.host_teff, self.host_mass, self.host_radius,
                self.host_metallicity, light.target_av_mag)
            components.append((models.jnp.stack((component.u1, component.u2)),
                               component.angular_profile, valid))
        self.components = tuple(components)

    def _predictions(self, radius_ratio, ecc, argp, inc):
        jnp = models.jnp
        predictions, valid = [], jnp.asarray(True)
        for grid, weights, share, (coefficients, profile, light_valid) in zip(
                self.joint.grids, self.joint.weights, self.joint.shares, self.components):
            instantaneous, geometry_valid = _quadratic_disc_flux(
                grid, self.period_days, radius_ratio, self.a_rs, ecc, argp, inc,
                coefficients, angular_profile=profile)
            intrinsic = jnp.sum(instantaneous.reshape(-1, weights.shape[-1]) * weights, axis=1)
            intrinsic = _dilute(intrinsic, self.companion_fluxratio, self.companion_is_host)
            prediction, active = _aperture_prediction(intrinsic, share)
            predictions.append(prediction)
            # Streams have different lengths: reduce each stream's cadence validity
            # to one flag before combining, never broadcast across streams.
            valid = valid & (jnp.all(light_valid & geometry_valid & jnp.isfinite(intrinsic)) | ~active)
        return predictions, valid

    def evaluate(self, unit_point):
        """Return the joint log density and numerical validity on the unit cube."""
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host TP unit point must have shape (4,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf, dtype=jnp.float64), jnp.asarray(True))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan, dtype=jnp.float64), jnp.asarray(False))

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
                predictions, valid = self._predictions(radius_ratio, ecc, argp, inc)
                value = self.joint.density(predictions) + jnp.log(band_factor)
                valid = valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid)

            return jax.lax.cond(
                jnp.isfinite(parameters).all(),
                lambda _unused: jax.lax.cond(supported, on_support, lambda _unused: zero, None),
                lambda _unused: failure, None,
            )

        return jax.lax.cond(
            jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None,
        )


class JointEclipsingPairUnitIntegrand(_FixedHostEclipsingPair):
    """Joint fixed-host EB density: one orbit, every band's own pair light.

    Reuses the retained Moe & Di Stefano (2017) mass-ratio/eccentricity prior
    transform and the shared stellar scenario relations of the scalar nested
    integrand. Each band's ``StellarLight.prepare_pair`` supplies that band's
    secondary-to-primary ratio and both bodies' own limb darkening, so no band
    borrows another band's colour.
    """

    claim_eligible = False

    def __init__(self, reference, device):
        from trex.evidence._jax_eb_unit_integrand import (
            _stellar_splines_on_device,
        )
        from trex.populations.mass_ratio_priors import broken_power_law_coefficients

        joint = getattr(reference, "joint_observations", None)
        if joint is None:
            raise ValueError("the native joint EB callback requires joint_observations")
        if reference.population is not None or getattr(reference, "hosted", False):
            raise NotImplementedError("the native joint EB callback covers fixed hosts only")
        _joint_native_geometry(reference)
        self.device = device
        self.period_days = _positive(reference.effective_period, "effective_period")
        self.use_2xP = bool(getattr(reference, "use_2xP", False))
        self.host_mass = _positive(reference.host_mass, "host_mass")
        self.host_radius = _positive(reference.host_radius, "host_radius")
        self.host_teff = _positive(reference.host_teff, "host_teff")
        self.host_metallicity = _finite(reference.target_feh, "target_feh")
        self.companion_fluxratio = 0.0
        self.companion_is_host = False
        # Same inverse CDF as priors.sample_ecc: e = x**(1/power), with MDS17
        # Eq. 17's density e**eta above the 10-d break, i.e. power 1 + eta.
        self.eccentricity_power = (priors.MOE2017_ECCENTRICITY_POWERLAW_SHORT
                                   if self.period_days <= priors.MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS
                                   else 1.0 + priors.moe2017_binary_eccentricity_eta(self.period_days))
        self.q_floor = priors._stellar_mass_ratio_floor(self.host_mass)
        if self.q_floor >= 1.0:
            raise ValueError("Moe mass-ratio prior has no support for this host")
        self.q_twin = priors.MOE2017_Q_TWIN_THRESHOLD
        self.p1 = priors.MOE2017_Q_SLOPE_LOW
        self.s = priors.MOE2017_Q_SLOPE_HIGH + 1.0
        coefficients = broken_power_law_coefficients(
            np.array([self.q_floor]), np.array([priors.MOE2017_Q_SLOPE_HIGH]),
            np.array([priors.MOE2017_TWIN_FRACTION]))
        (self.steep_lower, _twin_lower, self.a1, self.a2,
         self.i1, self.i2, self.i3) = (float(value[0]) for value in coefficients)
        self.splines = _stellar_splines_on_device(device)
        self.joint = _JointAperture(joint, device)
        self.lights = _joint_band_lights(self.joint)
        # The device wrapper returns (ComponentLight, valid); the host-side
        # StellarLight.prepare_component returns only the component.
        self.native_lights = tuple(NativeStellarLight(light, device) for light in self.lights)

    def _predictions(self, q, ecc, argp, inc, radius, teff, axis):
        from trex.accelerators._jax_orbits import _sky_geometry

        jnp = models.jnp
        mass = q * self.host_mass
        radius_ratio = radius / self.host_radius
        predictions, valid = [], jnp.asarray(True)
        for grid, weights, share, light in zip(
                self.joint.grids, self.joint.weights, self.joint.shares, self.native_lights):
            first, first_valid = light.prepare_component(
                self.host_teff, self.host_mass, self.host_radius,
                self.host_metallicity, light.target_av_mag)
            second, second_valid = light.prepare_component(
                teff, mass, radius, self.host_metallicity, light.target_av_mag)
            # Chen 2019 same-distance photon ratio; extinction is already in both surfaces.
            ratio = (radius / self.host_radius) ** 2 * second.surface_photon_flux / first.surface_photon_flux
            separation, front = _sky_geometry(grid, self.period_days, axis, inc, ecc, argp, 0.0)
            primary_instant = _disc_flux(
                separation, radius_ratio, jnp.stack((first.u1, first.u2)), front,
                angular_profile=first.angular_profile)
            secondary_instant = _disc_flux(
                separation / radius_ratio, 1.0 / radius_ratio,
                jnp.stack((second.u1, second.u2)), ~front,
                angular_profile=second.angular_profile)
            primary = jnp.sum(primary_instant.reshape(-1, weights.shape[-1]) * weights, axis=1)
            secondary = jnp.sum(secondary_instant.reshape(-1, weights.shape[-1]) * weights, axis=1)
            light_weights = binary_flux_weights(ratio, self.companion_fluxratio,
                                                self.companion_is_host, array_module=jnp)
            intrinsic = light_weights[0] * primary + light_weights[1] * secondary + light_weights[2]
            prediction, active = _aperture_prediction(intrinsic, share)
            predictions.append(prediction)
            valid = valid & (jnp.all(first_valid & second_valid & jnp.isfinite(separation).all()
                                     & jnp.isfinite(primary_instant).all()
                                     & jnp.isfinite(secondary_instant).all()
                                     & jnp.isfinite(intrinsic)) | ~active)
        return predictions, valid

    def evaluate(self, unit_point):
        """Return the joint log density and numerical validity on the unit cube."""
        from trex.stellar import stellar_relations as stellar

        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host EB unit point must have shape (4,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf, dtype=jnp.float64), jnp.asarray(True))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan, dtype=jnp.float64), jnp.asarray(False))

        def on_cube(_unused):
            parameters = self.prior_transform(point)
            q, ecc, argp, inc = parameters
            radius, teff = self.secondary_geometry(q)
            k = radius / self.host_radius
            axis = self._axis(q)
            band = 1.0 - self._band_low(k, ecc, argp, axis)
            impact = jnp.abs(axis * jnp.cos(jnp.radians(inc)))
            factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
            support = (band > 0.0) & (impact * factor <= 1.0 + k) & (axis * (1.0 - ecc) >= 1.0 + k)
            # G21 Sec. 2.4.4: disjoint EB/twin q slots, preserving each branch's prior mass.
            support &= (q >= self.q_twin) if self.use_2xP else (q < self.q_twin)

            def on_support(_unused):
                predictions, valid = self._predictions(q, ecc, argp, inc, radius, teff, axis)
                value = self.joint.density(predictions) + jnp.log(band)
                valid = valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid)

            # Outside the retained stellar relation there is zero prior support,
            # exactly as the scalar integrand returns -inf before evaluating it.
            companion_mass = q * self.host_mass
            in_domain = ((companion_mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR)
                         & (companion_mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR))
            # The transform returns a finite placeholder outside the relation;
            # only the likelihood may reject that draw as structural zero.
            return jax.lax.cond(
                jnp.isfinite(parameters).all() & (jnp.isfinite(teff) | ~in_domain),
                lambda _unused: jax.lax.cond(in_domain & support, on_support, lambda _unused: zero, None),
                lambda _unused: failure, None,
            )

        return jax.lax.cond(
            jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None,
        )


class _JointCallback:
    """Picklable boundary from the scalar integrand to its device joint twin."""

    def __init__(self, reference, factory):
        self.reference = reference
        self.factory = factory

    def __call__(self, *, device):
        return self.factory(self.reference, device=device)


def joint_planet_callback(reference):
    """Device factory for a supported joint TP scenario, or ``None``.

    The fixed-host family uses the twin in this module; the unresolved
    population slots (diluting ``PTP``/``DTP``, hosted ``STP``/``BTP``) use
    the population twin. Characterized-host configurations keep the scalar
    joint integrand, so the caller neither reuses a single-band callback nor
    fabricates a band flux.
    """
    joint = getattr(reference, "joint_observations", None)
    if joint is None:
        raise ValueError("joint callback requested for a scenario without joint observations")
    if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
        return None
    population = reference.population
    if population is None and not getattr(reference, "hosted", False):
        return _JointCallback(reference, JointPlanetUnitIntegrand)
    from functools import partial

    from trex.evidence.nested_population import (
        _BackgroundStarPopulation,
        _BoundCompanionPopulation,
    )

    if isinstance(population, _BoundCompanionPopulation):
        from trex.accelerators._jax_joint_population_integrand import (
            JointBoundPopulationUnitIntegrand,
        )

        return _JointCallback(reference, partial(JointBoundPopulationUnitIntegrand, planet=True))
    if isinstance(population, _BackgroundStarPopulation):
        from trex.accelerators._jax_joint_population_integrand import (
            JointBackgroundPopulationUnitIntegrand,
        )

        return _JointCallback(
            reference, partial(JointBackgroundPopulationUnitIntegrand, planet=True))
    return None


def joint_eclipsing_pair_callback(reference):
    """Device factory for a supported joint EB scenario, or ``None``.

    The same routing rule as :func:`joint_planet_callback`: the fixed-host
    family uses the twin in this module, the unresolved population slots
    (diluting ``PEB``/``PEBx2P``/``DEB``/``DEBx2P``, hosted
    ``SEB``/``SEBx2P``/``BEB``/``BEBx2P``) use the population twin, and
    characterized hosts keep the scalar joint integrand.
    """
    joint = getattr(reference, "joint_observations", None)
    if joint is None:
        raise ValueError("joint callback requested for a scenario without joint observations")
    if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
        return None
    population = reference.population
    if population is None and not getattr(reference, "hosted", False):
        return _JointCallback(reference, JointEclipsingPairUnitIntegrand)
    from functools import partial

    from trex.evidence.nested_population import (
        _BackgroundStarPopulation,
        _BoundCompanionPopulation,
    )

    if isinstance(population, _BoundCompanionPopulation):
        from trex.accelerators._jax_joint_population_integrand import (
            JointBoundPopulationUnitIntegrand,
        )

        return _JointCallback(reference, partial(JointBoundPopulationUnitIntegrand, planet=False))
    if isinstance(population, _BackgroundStarPopulation):
        from trex.accelerators._jax_joint_population_integrand import (
            JointBackgroundPopulationUnitIntegrand,
        )

        return _JointCallback(
            reference, partial(JointBackgroundPopulationUnitIntegrand, planet=False))
    return None
