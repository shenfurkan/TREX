"""Background EB callback sharing the full fixed-host JAX orbital integrand.

Population-row selection, imaging and atmosphere-grid provenance are retained
by the background TP adapter. This module specializes only the binary host
parameters and mass-ratio inverse-CDF coefficients, never a separate forward
model. Runtime indexing is device-native; every table row remains present.
The secondary flux follows the corrected target-frame scalar contract.

Units, sources and interpretation limits follow ``_jax_eb_unit_integrand``:
TDB days, fractional host-normalized photometry, solar mass/radius, K, degrees,
dimensionless quantiles/log density. This is experimental diagnostic code,
not a calibrated evidence estimator or a candidate claim.
"""

import copy

import numpy as np

from trex.accelerators import _jax_models as models
from trex.populations import priors
from trex.populations.mass_ratio_priors import broken_power_law_coefficients
from trex.evidence._jax_background_unit_integrand import BackgroundPlanetUnitIntegrand
from trex.evidence._jax_eb_unit_integrand import FixedHostEclipsingPairUnitIntegrand
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation
from trex.stellar.stellar_relations import stellar_relations
from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT
from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid


class BackgroundEclipsingPairUnitIntegrand:
    """Six-dimensional DEB/BEB callback with dynamic per-row binary hosts."""

    claim_eligible = False

    def __init__(self, reference, *, device):
        # The shared background reader needs the planet-style period metadata,
        # not planet priors. Only setup uses this shallow data view.
        context = copy.copy(reference)
        context.period_days = reference.effective_period
        context.use_flat_priors = False
        self.background = BackgroundPlanetUnitIntegrand(context, device=device)
        self.device = device
        self.log_occurrence_weight = reference.population.log_occurrence_weight
        self.hosted = reference.hosted
        self.host_masses = np.asarray(reference.population.mass_solar) if self.hosted else np.full(
            reference.population.mass_solar.size, reference.host_mass)
        q_masses = np.maximum(self.host_masses, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
        q_floor = priors._stellar_mass_ratio_floors(q_masses)
        # The CPU inverse CDF's own coefficients, so device and CPU q agree.
        lower, _twin_lower, a1, a2, i1, i2, i3 = broken_power_law_coefficients(
            q_floor, np.full(q_floor.shape, priors.MOE2017_Q_SLOPE_HIGH),
            np.full(q_floor.shape, priors.MOE2017_TWIN_FRACTION))
        temperatures = (np.asarray(reference.population.teff_k) if self.hosted
                        else np.full(q_floor.size, reference.host_teff))
        self.row_values = tuple(models.jax.device_put(value, device) for value in (
            q_floor, lower, a1, a2, i1, i2, i3, temperatures, self.host_masses >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR))
        # One fixed-host object owns the exact spline/grid coefficients. Its
        # setup host is a real supported mass from the retained input, not a
        # fabricated missing observation. Every evaluated host is overwritten
        # by its actual indexed row below.
        base = copy.copy(reference)
        base.hosted, base.population, base.n_dimensions = False, None, 4
        base.host_mass = float(q_masses[0])
        if getattr(reference, "stellar_light", None) is not None:
            base.host_teff = float(temperatures[0])
        else:
            _, teff = stellar_relations(np.asarray([base.host_mass]))
            base.host_teff = float(teff[0])
        if self.hosted:
            metallicity = np.asarray(reference.population.mh_dex, dtype=float)
            if metallicity.shape != self.host_masses.shape or not np.all(np.isfinite(metallicity)):
                raise ValueError("hosted background EB requires one finite [M/H] per population row")
            base.target_feh = float(metallicity[0])
        self.base = FixedHostEclipsingPairUnitIntegrand(base, device=device)
        if self.hosted and self.base.photon_light is None:
            grid = load_triceratops_tess_grid()
            planes = np.asarray(grid._metallicity_planes)
            zi = np.argmin(abs(metallicity[:, None] - planes), axis=-1)
            self.secondary_planes = models.jax.device_put(zi, device)
        if self.hosted and self.base.photon_light is not None:
            if reference.population.av_mag is None:
                raise ValueError("physical background EB requires retained A_V")
            extinction = np.asarray(reference.population.av_mag, dtype=float)
            if extinction.shape != metallicity.shape or not np.all(np.isfinite(extinction)):
                raise ValueError("physical background EB requires one finite A_V per population row")
            self.physical_rows = tuple(models.jax.device_put(value, device) for value in (
                metallicity, extinction))

    def _view(self, point):
        """Build a tracer-only parameter view; no loader/scientific host call."""
        index = self.background._index(point)
        masses, radii, _axes, coefficients, fractions, _contrast, clamped = self.background.rows
        q_floor, lower, a1, a2, i1, i2, i3, temperatures, domain = self.row_values
        view = copy.copy(self.base)
        view.host_mass, view.host_radius = masses[index], radii[index]
        view.host_teff, view.u_coeffs = temperatures[index], coefficients[index]
        view.q_floor, view.steep_lower = q_floor[index], lower[index]
        view.a1, view.a2, view.i1, view.i2, view.i3 = a1[index], a2[index], i1[index], i2[index], i3[index]
        view.companion_fluxratio, view.companion_is_host = fractions[index], bool(self.hosted)
        if self.hosted:
            if view.photon_light is not None:
                view.host_metallicity, view.host_av_mag = (a[index] for a in self.physical_rows)
            else:
                view.grid_plane = self.secondary_planes[index]
        return view, domain[index], clamped[index]

    def prior_transform(self, unit_point):
        """Map a point of the unit cube [0, 1)^6 to (q, e, omega, i, row, separation)."""
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("background EB point must have shape (6,)")
        view, _domain, _clamped = self._view(point)
        return jnp.concatenate((view.prior_transform(point[:4]), point[4:]))

    def evaluate(self, unit_point):
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (6,):
            raise ValueError("background EB point must have shape (6,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf), jnp.asarray(True), jnp.asarray(False))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan), jnp.asarray(False), jnp.asarray(False))

        def on_cube(_unused):
            view, supported, clamped = self._view(point)
            if self.background.measured_curve:
                separations, values = self.background.curve
                separation = self.background.field_radius_arcsec * jnp.sqrt(point[5])
                limit = jnp.interp(separation, separations, values, left=jnp.nan, right=jnp.nan)
                q = view._mass_ratio(point[0])
                radius, teff = view.secondary_geometry(q)
                # Use the retained magnitude contrast, not a fraction-to-ratio reconstruction.
                secondary = view.secondary_flux_ratio(q, radius, teff)
                correction = jnp.log1p(secondary) / (jnp.log(10.0) * POGSON_FLUX_EXPONENT)
                contrast = self.background.rows[5][self.background._index(point)] + (
                    correction if self.hosted else -correction)
                supported = supported & ~(jnp.isfinite(limit) & (contrast < limit))

            def retained(_unused):
                result = view.evaluate(point[:4])
                return result._replace(grid_boundary_clamped=result.grid_boundary_clamped | clamped)

            return jax.lax.cond(supported, retained, lambda _unused: zero, None)

        return jax.lax.cond(jnp.isfinite(point).all(),
                            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
                            lambda _unused: failure, None)
