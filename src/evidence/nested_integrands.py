"""Nested-evidence likelihood integrands."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from trex.foundation.constants import Msun, Rearth, Rsun
from trex.foundation.errors import NumericalLikelihoodError
from trex.foundation.physical_constants import FULL_TURN_DEGREES
from trex.models.likelihoods import lnL_EB, lnL_TP
from trex.models.orbital_mechanics import semi_major_axis_cgs
from trex.populations._parameters import MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
from trex.populations.mass_ratio_priors import eb_mass_ratio_support
from trex.populations.priors import sample_ecc, sample_inc, sample_q_host_mass_ratio, sample_rp, sample_w
from trex.stellar.stellar_relations import (
    STELLAR_RELATIONS_MASS_MAX_SOLAR,
    STELLAR_RELATIONS_MASS_MIN_SOLAR,
    stellar_relations,
)
from trex.stellar_light.passband import tess_surface_brightness_ratio

from trex.evidence.mc_population import _host_grid_coefficients
from trex.evidence.mc_scenario import _geometric_transit_mask
from trex.evidence.nested_controls import _HOSTED_N_DIMENSIONS, _INCLINATION_MAX_DEGREES, _validate_scalar
from trex.evidence.nested_population import UnresolvedPopulation, UnresolvedPopulationDraw
from trex.evidence.nested_population import _BackgroundStarPopulation as _BackgroundStarPopulation
from trex.evidence.nested_population import _BoundCompanionPopulation as _BoundCompanionPopulation
from trex.populations.companion_priors import companion_hierarchical_stability_supported as companion_hierarchical_stability_supported
from trex.populations._validation import unit_quantiles


def _transit_band_low(
    occultor_rs: float, ecc: float, argp: float, a_rs: float
) -> float:
    """Lower edge of the transiting inclination quantile for one draw.

    The ``cos_inc_max`` condition matches ``_eval_scenario``'s geometric
    transit mask; ``sample_inc`` maps the
    unit variate to ``cos i = 1 - x``, so the transiting band is
    ``x in [band_low, 1]``.
    """
    eccentric_factor = (1.0 - ecc ** 2) / (
        1.0 + ecc * math.sin(math.radians(argp))
    )
    cos_inc_max = (1.0 + occultor_rs) / (eccentric_factor * a_rs)
    return 1.0 - min(max(cos_inc_max, 0.0), 1.0)


def _reject_out_of_prior(
    ecc: float, argp: float, inc: float
) -> bool:
    """True for a parameter vector outside the shared shape/eccentricity prior."""
    return (
        not 0.0 <= ecc < 1.0
        or not 0.0 <= argp <= FULL_TURN_DEGREES
        or not 0.0 <= inc <= _INCLINATION_MAX_DEGREES
    )


def _hosted_draw(
    integrand: "_TransitingPlanetIntegrand | _EclipsingPairIntegrand", unit_point: np.ndarray
) -> "UnresolvedPopulationDraw":
    """The integrand's hosted population draw, reused for the point just transformed.

    dynesty calls ``prior_transform`` and then ``log_likelihood`` on one point
    and both need this draw; ``resolve`` is pure, so the reuse is bit-identical
    and saves what profiling measured as half of a hosted nested call.
    """
    key = unit_point.tobytes()
    cached = getattr(integrand, "_last_hosted_draw", None)
    if cached is None or cached[0] != key:
        cached = (key, integrand.population.resolve(
            unit_point, defer_imaging=isinstance(integrand, _EclipsingPairIntegrand)))
        integrand._last_hosted_draw = cached
    return cached[1]


def _resolved_dilution(
    population: "Optional[UnresolvedPopulation]", unit_point: np.ndarray
) -> Optional[float]:
    """Diluting flux fraction for one draw, or ``None`` where there is no support."""
    if population is None:
        return 0.0
    draw = population.resolve(unit_point)
    if not draw.supported:
        return None
    return float(draw.flux_fraction)


class _TransitingPlanetIntegrand:
    """Prior transform and likelihood for one transiting-planet integral.

    A plain object rather than a pair of closures: with a worker pool
    attached, dynesty evolves points through ``pool.map``, which pickles both
    the callable and its arguments, and a closure over the caller's scope
    would not travel to a worker process.  Every field is plain data or a
    numpy array for the same reason.

    ``hosted`` selects the population-*hosted* slots (``STP``/``BTP``): the
    drawn unresolved source is the host star, the target's flux dilutes the
    event through ``companion_is_host=True``, and the radius prior, orbit
    geometry, and host atmosphere are resolved per draw exactly as
    ``_eval_scenario`` resolves them for those slots.
    """

    def __init__(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        sigma: "np.ndarray | float",
        period_days: float,
        host_mass: float,
        host_radius: float,
        u1: float,
        u2: float,
        exptime_days: float,
        bin_integration_days: float,
        nsamples: int,
        use_flat_priors: bool,
        population: "Optional[UnresolvedPopulation]",
        n_dimensions: int,
        hosted: bool = False,
        target_feh: Optional[float] = None,
        host_teff_k: Optional[float] = None,
        stellar_light=None,
        baseline_likelihood=None,
        joint_observations=None,
    ) -> None:
        self.time = time
        self.flux = flux
        self.sigma = sigma
        self.period_days = period_days
        self.host_mass = host_mass
        self.host_radius = host_radius
        self.u1 = u1
        self.u2 = u2
        self.exptime_days = exptime_days
        self.bin_integration_days = bin_integration_days
        self.nsamples = nsamples
        self.use_flat_priors = use_flat_priors
        self.population = population
        self.n_dimensions = n_dimensions
        self.hosted = bool(hosted)
        self.target_feh = None if target_feh is None else _validate_scalar(
            target_feh, "target_feh", positive=False)
        self.stellar_light = stellar_light
        self.baseline_likelihood = baseline_likelihood
        self.joint_observations = joint_observations
        if joint_observations is not None:
            self.jax_unit_integrand = None
        from trex.stellar_light.inputs import require_same_population_model

        require_same_population_model(population, stellar_light)
        self.host_teff = host_teff_k
        self.host_component = None
        if stellar_light is not None and not hosted:
            self.host_component = stellar_light.prepare_component(
                host_teff_k, host_mass, host_radius, target_feh, stellar_light.target_av_mag)
            self.u1, self.u2 = float(self.host_component.u1), float(self.host_component.u2)
        if self.hosted:
            # The host is the drawn source; every geometric quantity is
            # resolved per point by ``_host_geometry``.
            self.a_cm = 0.0
            self.a_rs = 0.0
        else:
            self.a_cm = semi_major_axis_cgs(period_days, host_mass * Msun)
            self.a_rs = self.a_cm / (host_radius * Rsun)

    def _host_geometry(self, host_mass: float, host_radius: float) -> Tuple[float, float]:
        """Return ``(a_cm, a_rs)`` for one resolved host, as production does."""
        a_cm = semi_major_axis_cgs(self.period_days, host_mass * Msun)
        return a_cm, a_cm / (host_radius * Rsun)

    def occultor_radius_ratio(self, R_p: float, host_radius: Optional[float] = None) -> float:
        radius = self.host_radius if host_radius is None else host_radius
        return R_p * Rearth / (radius * Rsun)

    def jax_unit_integrand(self, *, device):
        """Native likelihood callback for defensive importance refinement.

        Photometry is already host-normalized fractional flux/errors at TDB
        days from conjunction. The explicit caller owns the device lease.
        Fixed-host, bound-companion and background populations have native paths.
        Scalar priors/likelihoods and the production estimator remain unchanged.
        """
        from trex.evidence._jax_unit_integrand import FixedHostPlanetUnitIntegrand

        if isinstance(self.population, _BoundCompanionPopulation):
            from trex.evidence._jax_bound_unit_integrand import BoundPopulationUnitIntegrand

            return BoundPopulationUnitIntegrand(self, device=device, planet=True)
        if isinstance(self.population, _BackgroundStarPopulation):
            from trex.evidence._jax_background_unit_integrand import BackgroundPlanetUnitIntegrand

            return BackgroundPlanetUnitIntegrand(self, device=device)
        return FixedHostPlanetUnitIntegrand(self, device=device)

    def prior_transform(self, unit_point: np.ndarray) -> np.ndarray:
        point = unit_quantiles(unit_point, "planet unit point")
        if point.shape != (self.n_dimensions,):
            raise ValueError("planet unit point does not match the scenario dimensions")
        if self.hosted:
            draw = _hosted_draw(self, point[_HOSTED_N_DIMENSIONS:])
            host_mass = float(draw.mass_solar)
            host_radius = float(draw.radius_solar)
            _, a_rs = self._host_geometry(host_mass, host_radius)
        else:
            host_mass = self.host_mass
            host_radius = self.host_radius
            a_rs = self.a_rs
        R_p = float(
            sample_rp(
                np.array([point[0]]),
                np.array([host_mass]),
                flatpriors=self.use_flat_priors,
            )[0]
        )
        ecc = float(
            sample_ecc(np.array([point[1]]), planet=True, P_orb=self.period_days)[0]
        )
        argp = float(sample_w(np.array([point[2]]))[0])
        band_low = _transit_band_low(
            self.occultor_radius_ratio(R_p, host_radius), ecc, argp, a_rs
        )
        x_inc = band_low + float(point[3]) * (1.0 - band_low)
        inc = float(sample_inc(np.array([x_inc]))[0])
        hosted = np.array([R_p, ecc, argp, inc])
        if self.population is None:
            return hosted
        return np.concatenate((hosted, point[_HOSTED_N_DIMENSIONS:]))

    def log_likelihood(self, parameters: np.ndarray) -> float:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.n_dimensions,):
            raise ValueError("planet parameters do not match the scenario dimensions")
        if not np.all(np.isfinite(values)):
            raise NumericalLikelihoodError("planet integrand received non-finite parameters")
        R_p, ecc, argp, inc = values[:_HOSTED_N_DIMENSIONS]
        if _reject_out_of_prior(ecc, argp, inc) or R_p <= 0.0:
            return float("-inf")
        if self.hosted:
            draw = _hosted_draw(self, values[_HOSTED_N_DIMENSIONS:])
            if not draw.supported:
                return float("-inf")
            host_mass = float(draw.mass_solar)
            host_radius = float(draw.radius_solar)
            if not (
                math.isfinite(host_mass)
                and host_mass > 0.0
                and math.isfinite(host_radius)
                and host_radius > 0.0
            ):
                return float("-inf")
            # Shared Mardling & Aarseth (2001) Eq. 90 mask: the drawn companion
            # is the inner binary here (hosting the planet), the target is the
            # outer body.
            if draw.log_period_days is not None and not companion_hierarchical_stability_supported(
                draw.log_period_days, self.period_days, host_mass, self.host_mass
            ):
                return float("-inf")
            a_cm, a_rs = self._host_geometry(host_mass, host_radius)
            u1_value, u2_value = self.u1, self.u2
            if self.stellar_light is None:
                host_u1, host_u2 = _host_grid_coefficients(
                    np.array([float(draw.teff_k)]), np.array([host_mass]), np.array([host_radius]),
                    draw.mh_dex if draw.mh_dex is not None else self.target_feh, scenario="hosted-TP")
                u1_value, u2_value = float(host_u1[0]), float(host_u2[0])
            companion_fluxratio = float(draw.flux_fraction)
            companion_is_host = True
        else:
            host_mass = self.host_mass
            host_radius = self.host_radius
            a_cm, a_rs = self.a_cm, self.a_rs
            u1_value, u2_value = self.u1, self.u2
            if isinstance(self.population, _BoundCompanionPopulation):
                population_draw = self.population.resolve(values[_HOSTED_N_DIMENSIONS:])
                if not population_draw.supported:
                    return float("-inf")
                # Shared MA01 Eq. 90 mask: the target hosts the planet (inner
                # binary) and the drawn companion is the outer body.
                if not companion_hierarchical_stability_supported(
                    population_draw.log_period_days, self.period_days,
                    self.host_mass, population_draw.mass_solar,
                ):
                    return float("-inf")
                companion_fluxratio = float(population_draw.flux_fraction)
            else:
                population_draw = (None if self.population is None else
                                   self.population.resolve(values[_HOSTED_N_DIMENSIONS:]))
                if population_draw is not None and not population_draw.supported:
                    return float("-inf")
                companion_fluxratio = 0.0 if population_draw is None else float(population_draw.flux_fraction)
            companion_is_host = False
        occultor_rs = self.occultor_radius_ratio(R_p, host_radius)
        band_factor = 1.0 - _transit_band_low(occultor_rs, ecc, argp, a_rs)
        if band_factor <= 0.0:
            return float("-inf")
        b = abs(a_rs * math.cos(math.radians(inc)))
        if not bool(
            _geometric_transit_mask(
                np.array([b]), np.array([occultor_rs]), np.array([ecc]),
                np.array([argp]), np.array([a_rs]),
            )[0]
        ):
            return float("-inf")
        if self.joint_observations is not None:
            population_draw = draw if self.hosted else population_draw
            lnL = self.joint_observations.evaluate(
                host_mass=host_mass, host_radius=host_radius,
                host_teff=float(draw.teff_k) if self.hosted else self.host_teff,
                host_metallicity=(draw.mh_dex if self.hosted and draw.mh_dex is not None else self.target_feh),
                host_av=draw.av_mag if self.hosted else None,
                period_days=self.period_days, inc_deg=inc, a_cm=a_cm,
                ecc=ecc, argp_deg=argp, planet_radius=R_p, companion_is_host=companion_is_host,
                dataset_dilutions=(None if population_draw is None else dict(population_draw.dataset_dilutions)),
            )
            return lnL + math.log(band_factor)
        angular_options = {}
        if self.stellar_light is not None:
            component = self.host_component
            if self.hosted:
                component = self.stellar_light.prepare_component(
                    float(draw.teff_k), host_mass, host_radius,
                    draw.mh_dex if draw.mh_dex is not None else self.target_feh, draw.av_mag)
            u1_value, u2_value = float(component.u1), float(component.u2)
            if component.angular_profile is not None:
                angular_options["angular_profile"] = component.angular_profile
        lnL = lnL_TP(
            self.time, self.flux, self.sigma, R_p, self.period_days, inc,
            a_cm, host_radius, u1_value, u2_value, self.exptime_days,
            ecc=ecc, argp_deg=argp, companion_fluxratio=companion_fluxratio,
            companion_is_host=companion_is_host,
            nsamples=self.nsamples,
            bin_integration_days=self.bin_integration_days,
            **({"baseline_likelihood": self.baseline_likelihood} if self.baseline_likelihood is not None else {}),
            **angular_options,
        )
        if not math.isfinite(lnL):
            raise NumericalLikelihoodError("planet likelihood returned a non-finite value on supported geometry")
        return lnL + math.log(band_factor)


class _EclipsingPairIntegrand:
    """Prior transform and likelihood for one eclipsing-pair integral.

    Picklable for the same reason as :class:`_TransitingPlanetIntegrand`.

    ``hosted`` selects the population-*hosted* slots (``SEB``/``SEBx2P``,
    ``BEB``/``BEBx2P``): the drawn unresolved source is the eclipsing pair's
    primary, its mass-ratio support floor, orbit geometry, secondary stellar
    relations, and host atmosphere are resolved per draw exactly as
    ``_eval_scenario`` resolves them, and the target's flux dilutes the event
    through ``companion_is_host=True``.
    """

    def __init__(
        self,
        *,
        time: np.ndarray,
        flux: np.ndarray,
        sigma: "np.ndarray | float",
        effective_period: float,
        use_2xP: bool = False,
        host_mass: float,
        host_radius: float,
        u1: float,
        u2: float,
        exptime_days: float,
        bin_integration_days: float,
        nsamples: int,
        population: "Optional[UnresolvedPopulation]",
        n_dimensions: int,
        hosted: bool = False,
        target_feh: Optional[float] = None,
        host_teff_k: Optional[float] = None,
        stellar_light=None,
        baseline_likelihood=None,
        joint_observations=None,
    ) -> None:
        self.time = time
        self.flux = flux
        self.sigma = sigma
        self.effective_period = effective_period
        self.use_2xP = bool(use_2xP)
        self.host_mass = host_mass
        self.host_radius = host_radius
        self.u1 = u1
        self.u2 = u2
        self.exptime_days = exptime_days
        self.bin_integration_days = bin_integration_days
        self.nsamples = nsamples
        self.population = population
        self.n_dimensions = n_dimensions
        self.hosted = bool(hosted)
        self.target_feh = None if target_feh is None else _validate_scalar(
            target_feh, "target_feh", positive=False)
        self.stellar_light = stellar_light
        self.baseline_likelihood = baseline_likelihood
        self.joint_observations = joint_observations
        if joint_observations is not None:
            self.jax_unit_integrand = None
        from trex.stellar_light.inputs import require_same_population_model

        require_same_population_model(population, stellar_light)
        if self.hosted:
            self.host_teff = float("nan")
        elif host_teff_k is not None:
            self.host_teff = _validate_scalar(host_teff_k, "host_teff_k", positive=True)
        else:
            if stellar_light is not None:
                raise ValueError("physical stellar light requires the actual fixed-host temperature")
            _, host_teff = stellar_relations(np.array([host_mass]))
            self.host_teff = float(host_teff[0])
        if stellar_light is not None and not hosted:
            component = stellar_light.prepare_component(
                self.host_teff, host_mass, host_radius, target_feh, stellar_light.target_av_mag)
            self.u1, self.u2 = float(component.u1), float(component.u2)

    def secondary_geometry(self, q: float, host_mass: Optional[float] = None) -> Tuple[float, float]:
        """Secondary radius and temperature for this draw's mass ratio."""
        primary = self.host_mass if host_mass is None else host_mass
        key = (q, primary)
        cached = getattr(self, "_last_secondary_geometry", None)
        if cached is None or cached[0] != key:
            # The prior transform and likelihood need the identical stellar draw.
            R_eb, teff_eb = stellar_relations(np.array([q * primary]))
            cached = (key, (float(R_eb[0]), float(teff_eb[0])))
            self._last_secondary_geometry = cached
        return cached[1]

    def jax_unit_integrand(self, *, device):
        """Native EB likelihood callback for defensive importance refinement.

        Uses the same host-normalized fractional photometry and TDB-day primary
        conjunction convention as the scalar integrand. Per-draw secondary grid
        boundaries remain explicit; bound and background populations have native paths.
        """
        from trex.evidence._jax_eb_unit_integrand import FixedHostEclipsingPairUnitIntegrand

        if isinstance(self.population, _BoundCompanionPopulation):
            from trex.evidence._jax_bound_unit_integrand import BoundPopulationUnitIntegrand

            return BoundPopulationUnitIntegrand(self, device=device, planet=False)
        if isinstance(self.population, _BackgroundStarPopulation):
            from trex.evidence._jax_background_eb_integrand import BackgroundEclipsingPairUnitIntegrand

            return BackgroundEclipsingPairUnitIntegrand(self, device=device)
        return FixedHostEclipsingPairUnitIntegrand(self, device=device)

    def draw_axes(
        self, q: float, host_mass: Optional[float] = None, host_radius: Optional[float] = None
    ) -> Tuple[float, float]:
        """Return ``(a_cm, a_rs)`` for this draw's total mass, as production does."""
        primary = self.host_mass if host_mass is None else host_mass
        radius = self.host_radius if host_radius is None else host_radius
        a_cm = semi_major_axis_cgs(
            self.effective_period, primary * (1.0 + q) * Msun
        )
        return a_cm, a_cm / (radius * Rsun)

    def prior_transform(self, unit_point: np.ndarray) -> np.ndarray:
        point = unit_quantiles(unit_point, "binary unit point")
        if point.shape != (self.n_dimensions,):
            raise ValueError("binary unit point does not match the scenario dimensions")
        if self.hosted:
            draw = _hosted_draw(self, point[_HOSTED_N_DIMENSIONS:])
            host_mass = float(draw.mass_solar)
            host_radius = float(draw.radius_solar)
            # A host below the mass-ratio prior's domain carries zero support;
            # the transform still needs a valid q, so it samples just inside
            # the domain edge and the likelihood rejects the draw.
            q_mass = max(
                host_mass,
                float(MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR),
            )
        else:
            host_mass = self.host_mass
            host_radius = self.host_radius
            q_mass = self.host_mass
        q = float(
            sample_q_host_mass_ratio(
                np.array([point[0]]), np.array([q_mass])
            )[0]
        )
        ecc = float(
            sample_ecc(
                np.array([point[1]]), planet=False, P_orb=self.effective_period
            )[0]
        )
        argp = float(sample_w(np.array([point[2]]))[0])
        try:
            R_eb, _ = self.secondary_geometry(q, host_mass)
        except ValueError:
            # Out-of-domain secondary: unrepresentable support, rejected by
            # the likelihood; the band is built on the host radius so the
            # returned point stays finite and deterministic.
            R_eb = host_radius
        _, a_rs = self.draw_axes(q, host_mass, host_radius)
        band_low = _transit_band_low(R_eb / host_radius, ecc, argp, a_rs)
        x_inc = band_low + float(point[3]) * (1.0 - band_low)
        inc = float(sample_inc(np.array([x_inc]))[0])
        hosted = np.array([q, ecc, argp, inc])
        if self.population is None:
            return hosted
        return np.concatenate((hosted, point[_HOSTED_N_DIMENSIONS:]))

    def log_likelihood(self, parameters: np.ndarray) -> float:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.n_dimensions,):
            raise ValueError("binary parameters do not match the scenario dimensions")
        if not np.all(np.isfinite(values)):
            raise NumericalLikelihoodError("binary integrand received non-finite parameters")
        q, ecc, argp, inc = values[:_HOSTED_N_DIMENSIONS]
        if not 0.0 < q <= 1.0 or _reject_out_of_prior(ecc, argp, inc):
            return float("-inf")
        # G21 Sect. 2.4.4: disjoint EB/twin q support [dimensionless]; keep full prior mass.
        if not eb_mass_ratio_support(q, self.use_2xP):
            return float("-inf")
        if self.hosted:
            draw = _hosted_draw(self, values[_HOSTED_N_DIMENSIONS:])
            if not draw.supported:
                return float("-inf")
            host_mass = float(draw.mass_solar)
            host_radius = float(draw.radius_solar)
            host_teff = float(draw.teff_k)
            host_metallicity = draw.mh_dex if draw.mh_dex is not None else self.target_feh
            if host_mass < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
                return float("-inf")
            secondary_mass = q * host_mass
            if not (
                STELLAR_RELATIONS_MASS_MIN_SOLAR
                <= secondary_mass
                <= STELLAR_RELATIONS_MASS_MAX_SOLAR
            ):
                return float("-inf")
            # Shared MA01 Eq. 90 mask: the drawn companion plus its eclipsing
            # secondary is the inner binary; the target is the outer body.
            if draw.log_period_days is not None and not companion_hierarchical_stability_supported(
                draw.log_period_days, self.effective_period,
                host_mass * (1.0 + q), self.host_mass,
            ):
                return float("-inf")
            a_cm, a_rs = self.draw_axes(q, host_mass, host_radius)
            R_eb, teff_eb = self.secondary_geometry(q, host_mass)
            u1_value, u2_value = self.u1, self.u2
            if self.stellar_light is None:
                host_u1, host_u2 = _host_grid_coefficients(
                    np.array([host_teff]), np.array([host_mass]), np.array([host_radius]),
                    host_metallicity, scenario="hosted-EB")
                u1_value, u2_value = float(host_u1[0]), float(host_u2[0])
            companion_fluxratio = float(draw.flux_fraction)
            companion_is_host = True
        else:
            # Match the hosted/native support before requesting a stellar-grid value.
            if not (
                STELLAR_RELATIONS_MASS_MIN_SOLAR
                <= q * self.host_mass
                <= STELLAR_RELATIONS_MASS_MAX_SOLAR
            ):
                return float("-inf")
            host_radius = self.host_radius
            host_metallicity = self.target_feh
            host_teff = self.host_teff
            a_cm, a_rs = self.draw_axes(q)
            u1_value, u2_value = self.u1, self.u2
            draw = None if self.population is None else self.population.resolve(
                values[_HOSTED_N_DIMENSIONS:], defer_imaging=True)
            if draw is not None and not draw.supported:
                return float("-inf")
            # Shared MA01 Eq. 90 mask: the target plus its eclipsing secondary
            # is the inner binary; the drawn companion is the outer body.
            if draw is not None and draw.log_period_days is not None and not companion_hierarchical_stability_supported(
                draw.log_period_days, self.effective_period,
                self.host_mass * (1.0 + q), draw.mass_solar,
            ):
                return float("-inf")
            companion_fluxratio = 0.0 if draw is None else float(draw.flux_fraction)
            companion_is_host = False
            R_eb, teff_eb = self.secondary_geometry(q)
        occultor_rs = R_eb / host_radius
        if self.stellar_light is None:
            flux_ratio = occultor_rs ** 2 * float(
                tess_surface_brightness_ratio(
                    np.array([teff_eb]), np.array([host_teff])
                )[0]
            )
        else:
            host_av = (draw.av_mag if self.hosted else self.stellar_light.target_av_mag)
            ratio, first, second = self.stellar_light.prepare_pair(
                (host_teff, host_mass if self.hosted else self.host_mass, host_radius),
                (teff_eb, q*(host_mass if self.hosted else self.host_mass), R_eb),
                host_metallicity, host_av,
            )
            flux_ratio = float(ratio)
            u1_value, u2_value = float(first.u1), float(first.u2)
        if draw is not None and not np.isfinite(draw.imaging.apply(
                0.0, flux_ratio, companion_is_host, secondary_to_host=True)[0]):
            return float("-inf")
        band_factor = 1.0 - _transit_band_low(occultor_rs, ecc, argp, a_rs)
        if band_factor <= 0.0:
            return float("-inf")
        b = abs(a_rs * math.cos(math.radians(inc)))
        if not bool(
            _geometric_transit_mask(
                np.array([b]), np.array([occultor_rs]), np.array([ecc]),
                np.array([argp]), np.array([a_rs]),
            )[0]
        ):
            return float("-inf")
        secondary_mass = q * (host_mass if self.hosted else self.host_mass)
        if self.joint_observations is not None:
            lnL = self.joint_observations.evaluate(
                host_mass=host_mass if self.hosted else self.host_mass,
                host_radius=host_radius, host_teff=host_teff, host_metallicity=host_metallicity,
                host_av=draw.av_mag if self.hosted else None,
                period_days=self.effective_period, inc_deg=inc, a_cm=a_cm,
                ecc=ecc, argp_deg=argp, secondary_mass=secondary_mass,
                secondary_radius=R_eb, secondary_teff=teff_eb, companion_is_host=companion_is_host,
                dataset_dilutions=None if draw is None else dict(draw.dataset_dilutions),
            )
            return lnL + math.log(band_factor)
        if self.stellar_light is None:
            eb_u1, eb_u2 = _host_grid_coefficients(
                np.array([teff_eb]), np.array([secondary_mass]), np.array([R_eb]),
                host_metallicity, scenario="EB-eclipsed-secondary",
            )
        else:
            eb_u1, eb_u2 = np.atleast_1d(second.u1), np.atleast_1d(second.u2)
        angular_options = {}
        if self.stellar_light is not None:
            if first.angular_profile is not None:
                angular_options.update(angular_profile=first.angular_profile,
                                       secondary_angular_profile=second.angular_profile)
        lnL = lnL_EB(
            self.time, self.flux, self.sigma, R_eb, None,
            self.effective_period, inc, a_cm, host_radius, u1_value,
            u2_value, self.exptime_days,
            ecc=ecc, argp_deg=argp, companion_fluxratio=companion_fluxratio,
            companion_is_host=companion_is_host,
            nsamples=self.nsamples,
            bin_integration_days=self.bin_integration_days,
            **({"baseline_likelihood": self.baseline_likelihood} if self.baseline_likelihood is not None else {}),
            u1_EB=float(eb_u1[0]), u2_EB=float(eb_u2[0]),
            secondary_to_host_flux_ratio=flux_ratio,
            **angular_options,
        )
        if not math.isfinite(lnL):
            raise NumericalLikelihoodError("binary likelihood returned a non-finite value on supported geometry")
        return lnL + math.log(band_factor)
