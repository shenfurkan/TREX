"""Pure-JAX background-source TP integrand, retaining every population row.

Uses the production discrete uniform stellar-table measure and uniform-area
separation, measured contrast-curve interpolation, per-row host geometry and
grid selection, and the same inclination-band weight and TP likelihood.
Data/priors originate from the owning scalar integrand; no missing catalog
values are synthesized. Constant occurrence weight remains outside the
likelihood, exactly as ``_with_occurrence_weight`` applies it.

Time/exposure use TDB days from conjunction; mass/radius use solar units,
temperature K, gravity log10(cgs), separation arcsec, flux/errors fractional,
log integrands nats. Sources/limitations are those of ``nested_evidence`` and
``_jax_unit_integrand`` (Giacalone et al. 2021, AJ 161, 24, DOI
``10.3847/1538-3881/abc6af``). The atmosphere-grid boundary contract remains
explicit. The callback supplies likelihood evaluations to importance refinement,
not an independent evidence backend.
"""

import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.populations import priors
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation, _photometry_on_device, _quadratic_disc_flux, _planet_radius_quantile, _inverse_planet_eccentricity
from trex.foundation.constants import G, Msun, Rearth, Rsun
from trex.foundation.physical_constants import FULL_TURN_DEGREES
from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT
from trex.models.orbital_mechanics import semi_major_axis_cgs
from trex.populations.companion_imaging import ContrastCurve
from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid, LDC_GRID_CLAMP_CONTRACT


class BackgroundPlanetUnitIntegrand:
    """All-retained-row DTP/BTP callback with no host operation inside JIT."""

    claim_eligible = False

    def __init__(self, reference, *, device):
        population = reference.population
        if reference.n_dimensions != 6 or population.n_dimensions != 2:
            raise ValueError("background TP requires the four orbit and two population dimensions")
        if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
            raise RuntimeError("native background TP requires JAX/jaxoplanet float64")
        for name in ("period_days", "exptime_days", "host_mass", "host_radius"):
            value = getattr(reference, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        arrays = [np.asarray(getattr(population, name), dtype=np.float64)
                  for name in ("tess_mag", "mass_solar", "radius_solar", "teff_k")]
        count = arrays[0].size
        if count == 0 or any(value.shape != (count,) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("background columns must be matching finite non-empty one-dimensional arrays")
        if any((value <= 0.0).any() for value in arrays[1:]):
            raise ValueError("background masses, radii and temperatures must be positive")
        self.device = device
        self.time, self.weights, self.flux, self.sigma, _target_u = _photometry_on_device(reference, device)
        self.baseline = (None if reference.baseline_likelihood is None
                         else reference.baseline_likelihood.on_device(device))
        self.hosted = bool(reference.hosted)
        self.period_days = float(reference.period_days)
        self.flat_radius_prior = bool(reference.use_flat_priors)
        self.star_count = count
        self.log_occurrence_weight = float(population.log_occurrence_weight)
        self.field_radius_arcsec = float(population.field_radius_arcsec)
        delta_mag = arrays[0] - population.target_tess_mag
        ratio = np.power(10.0, POGSON_FLUX_EXPONENT * delta_mag)
        fraction = ratio / (1.0 + ratio)
        if not np.isfinite(fraction).all() or (fraction <= 0.0).any() or (fraction >= 1.0).any():
            raise ValueError("background apparent flux fractions are not representable in float64")
        if self.hosted:
            masses, radii, temperatures = arrays[1:]
            logg = np.log10(G * masses * Msun / (radii * Rsun) ** 2)
            if getattr(reference, "stellar_light", None) is not None:
                from trex.stellar_light.native import physical_population_coefficients

                coefficients = physical_population_coefficients(reference.stellar_light, population)
                clamped = np.zeros(count, dtype=bool)
                self.grid_boundary_contract = reference.stellar_light.model_id
            else:
                grid = load_triceratops_tess_grid()
                selected = grid.select_many(temperatures, logg, population.mh_dex)
                coefficients = np.stack((selected.u1, selected.u2), axis=-1)
                clamped = selected.boundary_clamped
                self.grid_metallicity_dex = selected.used_metallicity_dex
                self.grid_used_teff_k = selected.used_teff_k
                self.grid_used_logg_cgs = selected.used_logg_cgs
                self.grid_boundary_contract = LDC_GRID_CLAMP_CONTRACT
        else:
            masses = np.full(count, reference.host_mass)
            radii = np.full(count, reference.host_radius)
            coefficients = np.tile([reference.u1, reference.u2], (count, 1))
            clamped = np.zeros(count, dtype=bool)
        axes = semi_major_axis_cgs(self.period_days, masses * Msun) / (radii * Rsun)
        self.rows = tuple(models.jax.device_put(value, device) for value in (
            masses, radii, axes, coefficients, fraction, delta_mag, clamped))
        self.photon_light = None
        if getattr(reference, "stellar_light", None) is not None:
            from trex.stellar_light.native import NativeStellarLight
            self.photon_light = NativeStellarLight(reference.stellar_light, device)
            if self.hosted:
                physical = population.teff_k, population.mh_dex, population.av_mag
            else:
                physical = tuple(np.full(count, value) for value in (
                    reference.host_teff, reference.target_feh, reference.stellar_light.target_av_mag))
            self.physical_rows = tuple(models.jax.device_put(value, device) for value in physical)
        self.measured_curve = bool(population.measured_curve)
        if self.measured_curve:
            curve = population.contrast_curve or ContrastCurve(
                population.contrast_separations, population.contrast_values)
            self.curve = tuple(models.jax.device_put(value, device) for value in (
                curve.separations, curve.contrasts))

    def _index(self, point):
        # Row of the discrete uniform table measure; the clip only states the
        # [0, star_count) range that u < 1 already implies in float64.
        jnp = models.jnp
        return jnp.minimum(jnp.floor(point[4] * self.star_count), self.star_count - 1).astype(jnp.int32)

    def _band_low(self, k, ecc, argp, axis):
        jnp = models.jnp
        factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
        return 1.0 - jnp.minimum(jnp.maximum((1.0 + k) / (factor * axis), 0.0), 1.0)

    def prior_transform(self, unit_point):
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("background TP point must have shape (6,)")
        index = self._index(point)
        masses, radii, axes, _coefficients, _fractions, _contrasts, _clamped = self.rows
        radius = _planet_radius_quantile(point[0], masses[index] > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR, self.flat_radius_prior)
        ecc = _inverse_planet_eccentricity(point[1])
        argp = point[2] * FULL_TURN_DEGREES
        k = radius * Rearth / (radii[index] * Rsun)
        low = self._band_low(k, ecc, argp, axes[index])
        quantile = low + point[3] * (1.0 - low)
        norm = 1.0 / (math.cos(0.0) - math.cos(math.radians(90.0)))
        inc = jnp.degrees(jnp.arccos(1.0 - quantile / norm))
        return jnp.concatenate((jnp.stack((radius, ecc, argp, inc)), point[4:]))

    def evaluate(self, unit_point):
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("background TP point must have shape (6,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf), jnp.asarray(True), jnp.asarray(False))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan), jnp.asarray(False), jnp.asarray(False))

        def on_cube(_unused):
            parameters = self.prior_transform(point)
            radius, ecc, argp, inc = parameters[:4]
            index = self._index(point)
            _mass, radii, axes, coefficients, fractions, contrasts, clamped = self.rows
            k = radius * Rearth / (radii[index] * Rsun)
            axis = axes[index]
            band = 1.0 - self._band_low(k, ecc, argp, axis)
            factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
            support = ((band > 0.0) & (jnp.abs(axis * jnp.cos(jnp.radians(inc))) * factor <= 1.0 + k)
                       & (axis * (1.0 - ecc) >= 1.0 + k))
            if self.measured_curve:
                separations, values = self.curve
                separation = self.field_radius_arcsec * jnp.sqrt(point[5])
                limit = jnp.interp(separation, separations, values, left=jnp.nan, right=jnp.nan)
                support = support & ~(jnp.isfinite(limit) & (contrasts[index] < limit))

            def on_support(_unused):
                profile, profile_valid = None, jnp.asarray(True)
                host_coefficients = coefficients[index]
                if self.photon_light is not None:
                    temperature, mh, av = (row[index] for row in self.physical_rows)
                    component, profile_valid = self.photon_light.prepare_component(
                        temperature, _mass[index], radii[index], mh, av)
                    profile = component.angular_profile
                    host_coefficients = jnp.stack((component.u1, component.u2))
                instantaneous, valid = _quadratic_disc_flux(self.time, self.period_days, k, axis,
                                                            ecc, argp, inc, host_coefficients,
                                                            angular_profile=profile)
                model = jnp.sum(instantaneous.reshape(-1, self.weights.shape[-1]) * self.weights, axis=1)
                companion_flux = fractions[index] / (1.0 - fractions[index])
                dilution = 1.0 / companion_flux if self.hosted else companion_flux
                model = (model + dilution) / (1.0 + dilution)
                likelihood = (self.baseline(model) if self.baseline is not None else
                              -0.5 * jnp.sum((self.flux - model) ** 2 / self.sigma ** 2))
                value = likelihood + jnp.log(band)
                valid = valid & profile_valid & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid, clamped[index])

            return jax.lax.cond(jnp.isfinite(parameters).all(),
                                lambda _unused: jax.lax.cond(support, on_support, lambda _unused: zero, None),
                                lambda _unused: failure, None)

        return jax.lax.cond(jnp.isfinite(point).all(),
                            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
                            lambda _unused: failure, None)
