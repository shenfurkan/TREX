"""Native continuous bound-companion TP/EB unit-cube integrand.

Retains the scalar companion period cumulative, shared M-dwarf mass-ratio
inverse and period-dependent Moe inverse, exact cubic stellar scenario
polynomials, explicit photon light (or the shared TESS-band blackbody
reference), measured imaging and catalog-resolution support. Constant
occurrence remains outside the integral callback. Unsupported population
draws retain the scalar transform's finite host coordinates but never
receive physical likelihood.

Sources/limits: Moe & Di Stefano (2017), ApJS 230, 15, DOI
``10.3847/1538-4365/aa6fb6``; Winters et al. (2019), AJ 157, 216,
DOI ``10.3847/1538-3881/ab05dc``; retained stellar
relations are uncalibrated as documented in ``trex.stellar.stellar_relations``.
Time/exposure: TDB days;
mass/radius: solar units; temperature: K; parallax: mas; angular separation:
arcsec; flux: fractional; orbit angles: degrees; log density: nats. The production
importance stage may select this callback on an admitted accelerator.
"""

import copy
import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.populations import priors
from trex.stellar import stellar_relations as stellar
from trex.stellar_light import passband as bandpass
from trex.foundation import physical_constants as physical
from trex.populations import companion_imaging as imaging
from trex.models import orbital_mechanics as orbit
from trex.evidence._jax_unit_integrand import FixedHostPlanetUnitIntegrand, UnitIntegrandEvaluation
from trex.evidence._jax_eb_unit_integrand import FixedHostEclipsingPairUnitIntegrand, _stellar_splines_on_device, _stellar_geometry_from_splines
from trex.foundation.physical_constants import MILLIARCSECONDS_PER_ARCSECOND
from trex.foundation.constants import Msun, Rsun
from trex.populations.companion_priors import _m_dwarf_mass_ratio_inverse
from trex.populations.companion_priors import _companion_log_period_from_log_separation as _companion_log_period_from_log_separation, winters_log_semimajor_axis_parameters as winters_log_semimajor_axis_parameters, companion_hierarchical_stability_supported as companion_hierarchical_stability_supported


def _integral(lower, upper, exponent):
    jnp = models.jnp
    span = jnp.log(upper) - jnp.log(lower)
    return lower ** exponent * jnp.where(exponent == 0.0, span,
        jnp.expm1(exponent * span) / jnp.where(exponent == 0.0, 1.0, exponent))


def _inverse(lower, integral, exponent):
    jnp = models.jnp
    scaled = jnp.where(exponent == 0.0, 0.0,
                        integral * exponent / lower ** jnp.where(exponent == 0.0, 0.0, exponent))
    return jnp.where(exponent == 0.0, lower * jnp.exp(integral),
                      jnp.exp(jnp.log(lower) + jnp.log1p(scaled) / jnp.where(exponent == 0.0, 1.0, exponent)))


def _ratio_coefficients(primary_mass, slope, twin):
    """JAX mirror of broken_power_law_coefficients for traced primary masses; keep in step."""
    jax, jnp = models.jax, models.jnp
    floor = jnp.where(primary_mass >= 1.0, priors.MOE2017_Q_BREAK_LOW, priors.MOE2017_Q_BREAK_LOW / primary_mass)
    def lift(_index, value):
        return jnp.where(value * primary_mass < priors.MOE2017_Q_BREAK_LOW, jnp.nextafter(value, jnp.inf), value)
    floor = jax.lax.fori_loop(0, priors._MASS_RATIO_FLOOR_MAX_ULP_STEPS, lift, floor)
    small = floor < priors.MOE2017_Q_BREAK_MID
    lower = jnp.where(small, priors.MOE2017_Q_BREAK_MID, floor)
    a1 = jnp.where(small, priors.MOE2017_Q_BREAK_MID ** priors.MOE2017_Q_SLOPE_LOW
                   / priors.MOE2017_Q_BREAK_MID ** slope, 1.0)
    exponent = slope + 1.0
    a2 = 1.0 + twin / (1.0 - twin) * _integral(lower, 1.0, exponent) / _integral(priors.MOE2017_Q_TWIN_THRESHOLD, 1.0, exponent)
    i1 = jnp.where(small, (priors.MOE2017_Q_BREAK_MID ** (priors.MOE2017_Q_SLOPE_LOW + 1.0)
                           - floor ** (priors.MOE2017_Q_SLOPE_LOW + 1.0)) / (priors.MOE2017_Q_SLOPE_LOW + 1.0), 0.0)
    twin_lower = jnp.maximum(priors.MOE2017_Q_TWIN_THRESHOLD, floor)
    i2 = a1 * _integral(lower, twin_lower, exponent)
    i3 = a2 * a1 * _integral(twin_lower, 1.0, exponent)
    return floor, lower, a1, a2, i1, i2, i3, exponent


def _mass_ratio(quantile, coefficients):
    jnp = models.jnp
    floor, lower, a1, a2, i1, i2, i3, exponent = coefficients
    target = quantile * (i1 + i2 + i3)
    first = (target * (priors.MOE2017_Q_SLOPE_LOW + 1.0) + floor ** (priors.MOE2017_Q_SLOPE_LOW + 1.0)) ** (1.0 / (priors.MOE2017_Q_SLOPE_LOW + 1.0))
    second = _inverse(lower, (target - i1) / a1, exponent)
    third = _inverse(jnp.maximum(priors.MOE2017_Q_TWIN_THRESHOLD, floor), (target - i1 - i2) / (a1 * a2), exponent)
    q = jnp.maximum(jnp.where(target <= i1, first, jnp.where(target <= i1 + i2, second, third)), floor)
    return jnp.where(quantile == 0.0, floor, jnp.where(quantile == 1.0, 1.0, q))


class BoundPopulationUnitIntegrand:
    """Pure-JAX P*/S* callback sharing the fixed-host scientific models."""

    claim_eligible = False

    def __init__(self, reference, *, device, planet):
        self._prepare_bound_population(reference, device=device, planet=planet)
        # Fixed objects retain coefficients once. No setup reader is called
        # from the traced resolve/transform/likelihood path.
        base = copy.copy(reference)
        base.population, base.hosted, base.n_dimensions = None, False, 4
        if planet:
            base.a_cm = orbit.semi_major_axis_cgs(base.period_days, base.host_mass * Msun)
            base.a_rs = base.a_cm / (base.host_radius * Rsun)
            self.base = FixedHostPlanetUnitIntegrand(base, device=device)
            if self.hosted:
                binary = copy.copy(base)
                binary.effective_period = base.period_days
                if self.photon_light is not None:
                    binary.host_teff = self.population.host_teff
                else:
                    _, teff = stellar.stellar_relations(np.asarray([binary.host_mass]))
                    binary.host_teff = float(teff[0])
                self.stellar = FixedHostEclipsingPairUnitIntegrand(binary, device=device)
                self.stellar_splines = self.stellar.splines
            else:
                # A diluting companion needs mass/radius/temperature, but no
                # eclipsed atmosphere. Do not require unused scene metallicity
                # merely to reuse the EB adapter's stellar-polynomial code.
                self.stellar = None
                self.stellar_splines = _stellar_splines_on_device(device)
        else:
            if self.hosted:
                # Setup-only host is overwritten by the drawn companion; fixed hosts keep their input Teff [K].
                if self.photon_light is not None:
                    base.host_teff = self.population.host_teff
                else:
                    _, teff = stellar.stellar_relations(np.asarray([base.host_mass]))
                    base.host_teff = float(teff[0])
            self.base = self.stellar = FixedHostEclipsingPairUnitIntegrand(base, device=device)
            self.stellar_splines = self.stellar.splines

    def _prepare_bound_population(self, reference, *, device, planet):
        """Population draw machinery shared with the joint device twin.

        The joint twin (``trex.accelerators._jax_joint_population_integrand``)
        reuses this exact draw, retention and occurrence setup so one
        population inverse-CDF implementation serves both likelihood owners.
        """
        population = reference.population
        expected_population = 3
        self.n_dimensions = 4 + expected_population
        if population.n_dimensions != expected_population or reference.n_dimensions != self.n_dimensions:
            raise ValueError(f"bound population integrand requires {self.n_dimensions} unit dimensions")
        self.device = device
        self.population = population
        self.planet = bool(planet)
        self.hosted = bool(reference.hosted)
        self.log_occurrence_weight = float(population.log_occurrence_weight)
        self.photon_light = None
        if getattr(reference, "stellar_light", None) is not None:
            from trex.stellar_light.native import NativeStellarLight

            self.photon_light = NativeStellarLight(reference.stellar_light, device)
            self.host_metallicity = reference.target_feh
        self.period_table = None
        if population.host_mass > priors.WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
            grid, cumulative = priors._moe2017_period_inverse_cdf(population.host_mass)
            self.period_table = tuple(models.jax.device_put(value, device) for value in (grid, cumulative))
        # The default companion light is the same TESS-band-integrated Planck
        # ratio the scalar and MC paths evaluate through
        # ``bandpass.tess_surface_brightness_ratio``; the traced path reads the
        # shared table instead of the retired pivot formula.
        self.band_table = None
        if self.photon_light is None:
            log_temperature, log_flux = bandpass._tess_band_log_blackbody_flux()
            self.band_table = tuple(models.jax.device_put(value, device) for value in (
                log_temperature, log_flux))
        if population.measured_curve:
            curve = population.contrast_curve or imaging.ContrastCurve(
                population.contrast_separations, population.contrast_values)
            self.curve = tuple(models.jax.device_put(value, device) for value in (
                curve.separations, curve.contrasts))

    def _period(self, quantile):
        from jax.scipy.special import ndtri
        jnp = models.jnp
        if self.period_table is not None:
            grid, cumulative = self.period_table
            return jnp.interp(quantile, cumulative, grid)
        mean, sigma = winters_log_semimajor_axis_parameters()
        return mean + sigma * ndtri(quantile)

    def _resolve(self, point):
        jax, jnp = models.jax, models.jnp
        population = self.population
        m_dwarf = population.host_mass <= priors.WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR
        period_or_separation = self._period(point[4])
        period_supported = (period_or_separation >= priors.MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS) & (period_or_separation <= priors.MOE2017_LOGPERIOD_MAX_DAYS)
        period_supported = period_supported | m_dwarf
        fallback = (jnp.asarray(population.host_mass), jnp.asarray(population.host_radius),
                     jnp.asarray(population.host_teff), jnp.asarray(0.0), jnp.asarray(False), jnp.asarray(jnp.nan),
                     jnp.asarray(jnp.nan), jnp.asarray(jnp.nan))

        def in_period(_unused):
            if population.host_mass <= priors.WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
                q = _m_dwarf_mass_ratio_inverse(
                    point[5], priors._stellar_mass_ratio_floor(population.host_mass), jnp
                )
            else:
                slope = priors.MOE2017_Q_SLOPE_HIGH + priors.MOE2017_Q_SLOPE_HIGH_PERIOD_GRADIENT_PER_DEX * jnp.maximum(
                    period_or_separation - priors.MOE2017_Q_SLOPE_HIGH_PERIOD_BREAK_LOGDAYS, 0.0)
                short_twin = priors.MOE2017_TWIN_FRACTION + priors.MOE2017_TWIN_FRACTION_MASS_COEFFICIENT * math.log10(population.host_mass)
                cutoff = (priors.MOE2017_TWIN_CUTOFF_LOGPERIOD_INTERCEPT + priors.MOE2017_TWIN_CUTOFF_LOGPERIOD_MASS_SLOPE_PER_SOLAR_MASS * population.host_mass
                           if population.host_mass <= priors.MOE2017_TWIN_CUTOFF_PRIMARY_MASS_BREAK_SOLAR else priors.MOE2017_TWIN_CUTOFF_LOGPERIOD_MASSIVE_DAYS)
                span = cutoff - priors.MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS
                twin = jnp.where(period_or_separation < priors.MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS, short_twin,
                    jnp.where(period_or_separation < cutoff, short_twin * (1.0 - (period_or_separation - priors.MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS) / span), 0.0)) if span > 0.0 else jnp.where(period_or_separation < priors.MOE2017_TWIN_PLATEAU_LOGPERIOD_DAYS, short_twin, 0.0)
                q = _mass_ratio(point[5], _ratio_coefficients(population.host_mass, slope, twin))
            mass = q * population.host_mass
            # Circular, isotropic companion orbit: independent line-of-sight cosine.
            projection = jnp.sqrt((1.0 - point[6]) * (1.0 + point[6]))
            if m_dwarf:
                log_period = _companion_log_period_from_log_separation(
                    population.host_mass + mass, period_or_separation, array_module=jnp
                )
                separation = 10.0 ** period_or_separation * projection * population.parallax_mas / MILLIARCSECONDS_PER_ARCSECOND
            else:
                log_period = period_or_separation
                # Kepler's law uses both masses [Msun]; axis is computed in cm.
                axis = jnp.cbrt(priors.G * (population.host_mass + mass) * Msun * (10.0 ** log_period * priors.SECONDS_PER_DAY) ** 2 / (4.0 * math.pi ** 2))
                separation = 10.0 ** jnp.log10(axis / priors.au) * projection * population.parallax_mas / MILLIARCSECONDS_PER_ARCSECOND
            # A unit-cube edge gives an infinite M-dwarf separation: no support, as on the scalar path.
            in_domain = ((mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR) & (mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR)
                         & jnp.isfinite(log_period) & (separation <= population.outer_arcsec))

            def retained_mass(_unused):
                radius, teff = _stellar_geometry_from_splines(self.stellar_splines, mass)
                if self.photon_light is not None:
                    ratio = self.photon_light.pair_ratio(
                        population.host_teff, population.host_mass, population.host_radius,
                        teff, mass, radius, self.host_metallicity, self.photon_light.target_av_mag)
                else:
                    # Same TESS-band blackbody ratio the scalar/MC paths use
                    # (bandpass.tess_surface_brightness_ratio); the shared table
                    # is placed on the device once.  Temperatures outside the
                    # band table carry no support, mirroring the scalar path's
                    # refusal rather than extrapolating the table.
                    log_temperature, log_flux = self.band_table
                    ratio = (radius / population.host_radius) ** 2 * jnp.exp(
                        jnp.interp(jnp.log(teff), log_temperature, log_flux)
                        - jnp.interp(jnp.log(population.host_teff), log_temperature, log_flux)
                    )
                fraction = ratio / (1.0 + ratio)
                contrast = jnp.log10(ratio) / physical.POGSON_FLUX_EXPONENT
                support = separation <= population.outer_arcsec
                if self.photon_light is None:
                    support = support & (
                        (teff >= bandpass.TESS_BAND_TEMPERATURE_MIN_K)
                        & (teff <= bandpass.TESS_BAND_TEMPERATURE_MAX_K)
                        & (population.host_teff >= bandpass.TESS_BAND_TEMPERATURE_MIN_K)
                        & (population.host_teff <= bandpass.TESS_BAND_TEMPERATURE_MAX_K)
                    )
                limit = jnp.asarray(jnp.nan)
                if population.measured_curve:
                    separations, values = self.curve
                    limit = jnp.interp(separation, separations, values, left=jnp.nan, right=jnp.nan)
                if self.planet:
                    # Missing physical light is numerical failure downstream, not a fainter-star prior cut.
                    fainter = (contrast >= 0.0) | ((self.photon_light is not None) & ~jnp.isfinite(contrast))
                    support = support & fainter & ~(jnp.isfinite(limit) & (contrast < limit))
                return mass, radius, teff, fraction, support, limit, log_period, contrast

            return jax.lax.cond(in_domain, retained_mass, lambda _unused: fallback, None)

        return jax.lax.cond(period_supported, in_period, lambda _unused: fallback, None)

    def _view(self, point):
        jnp = models.jnp
        mass, radius, teff, fraction, support, limit, log_period, companion_contrast = self._resolve(point)
        view = copy.copy(self.base)
        clamped = jnp.asarray(False)
        if self.hosted:
            view.host_mass, view.host_radius, view.host_teff = mass, radius, teff
            if self.photon_light is None:
                atmosphere = copy.copy(self.stellar)
                atmosphere.host_mass = mass
                view.u_coeffs, grid_valid, clamped = atmosphere.secondary_grid(1.0, radius, teff)
                support = support & grid_valid
            # Selected physical coefficients/rays are prepared together by view.evaluate;
            # model unavailability there is numerical failure, not a prior cut here.
            if self.planet:
                view.a_rs = (physical.GRAVITATIONAL_CONSTANT_CGS * mass * Msun * (view.period_days * physical.SECONDS_PER_DAY) ** 2
                              / (4.0 * math.pi ** 2)) ** (1.0 / 3.0) / (radius * Rsun)
                view.high_mass_radius_prior = mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
            else:
                q_mass = jnp.maximum(mass, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
                coefficients = _ratio_coefficients(q_mass, priors.MOE2017_Q_SLOPE_HIGH, priors.MOE2017_TWIN_FRACTION)
                view.q_floor, view.steep_lower, view.a1, view.a2, view.i1, view.i2, view.i3, view.s = coefficients
                support = support & (mass >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
        view.companion_fluxratio, view.companion_is_host = fraction, self.hosted
        if not self.planet:
            q = view.prior_transform(point[:4])[0]
            secondary_radius, secondary_teff = view.secondary_geometry(q)
            # Contrast sums stay in log light; no reconstruction from a rounded fraction.
            secondary = view.secondary_flux_ratio(q, secondary_radius, secondary_teff)
            correction = jnp.log1p(secondary) / (jnp.log(10.0) * physical.POGSON_FLUX_EXPONENT)
            contrast = companion_contrast + (correction if self.hosted else -correction)
            fainter = (contrast >= 0.0) | ((self.photon_light is not None) & ~jnp.isfinite(contrast))
            support = support & fainter & ~(jnp.isfinite(limit) & (contrast < limit))
            # Shared MA01 Eq. 90 mask against the observed (effective) EB
            # period: the inner binary is the pair primary (the drawn
            # companion for the hosted slot) plus its secondary.
            inner_binary_mass = (mass if self.hosted else self.population.host_mass) * (1.0 + q)
            outer_mass = self.population.host_mass if self.hosted else mass
            support = support & companion_hierarchical_stability_supported(
                log_period, self.base.period_days, inner_binary_mass, outer_mass,
                array_module=jnp,
            )
        else:
            # Shared MA01 Eq. 90 mask against the observed planet period: the
            # inner binary is the planet host alone (the planet's mass is not
            # part of any G21 prior and is negligible).
            inner_binary_mass = mass if self.hosted else self.population.host_mass
            outer_mass = self.population.host_mass if self.hosted else mass
            support = support & companion_hierarchical_stability_supported(
                log_period, self.base.period_days, inner_binary_mass, outer_mass,
                array_module=jnp,
            )
        return view, support, clamped

    def prior_transform(self, unit_point):
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError(f"bound population point must have shape ({self.n_dimensions},)")
        view, _support, _clamped = self._view(point)
        return jnp.concatenate((view.prior_transform(point[:4]), point[4:]))

    def evaluate(self, unit_point):
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError(f"bound population point must have shape ({self.n_dimensions},)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf), jnp.asarray(True), jnp.asarray(False))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan), jnp.asarray(False), jnp.asarray(False))
        def on_cube(_unused):
            view, support, clamped = self._view(point)
            def retained(_unused):
                result = view.evaluate(point[:4])
                return result._replace(grid_boundary_clamped=result.grid_boundary_clamped | clamped)
            return jax.lax.cond(support, retained, lambda _unused: zero, None)
        return jax.lax.cond(jnp.isfinite(point).all(),
            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
            lambda _unused: failure, None)
