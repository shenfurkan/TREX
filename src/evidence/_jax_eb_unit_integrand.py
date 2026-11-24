"""Fixed-host eclipsing-pair unit-cube integrand, all-device JAX.

Ports the retained Moe & Di Stefano (2017) mass-ratio/eccentricity priors,
stellar scenario splines, explicit photon atmospheres (or a TESS-band blackbody reference),
and primary/secondary disc light curves without per-point host callbacks.
Sources and limitations follow ``trex.populations.priors``,
``trex.stellar.stellar_relations`` and ``trex.models.likelihoods``:
Moe & Di Stefano (2017), ApJS 230, 15, DOI ``10.3847/1538-4365/aa6fb6``;
Torres et al. (2010), A&ARv 18, 67, DOI ``10.1007/s00159-009-0025-1``;
Mandel & Agol (2002), ApJ 580, L171, DOI ``10.1086/345520``. The cool-dwarf
node vector remains uncalibrated; its exact cubic interpolant is preserved,
not replaced by a new empirical relation or linear interpolation.

Photometry is fractional and host-normalized; times/exposure/bin widths are
TDB days relative to primary conjunction. Mass/radius are solar units,
temperature K, gravity log10(cgs), angles degrees and log integrands nats.
Quadratic limb darkening follows the reference grid and Claret extension with
an explicit per-draw clamp flag; explicit photon atmospheres never clamp or
fall back. Out-of-domain stellar splines are not evaluated as physical support.
"""

import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.accelerators import _jax_orbits
from trex.stellar import stellar_relations as stellar
from trex.stellar_light import passband as bandpass
from trex.foundation import physical_constants as physical
from trex.populations import priors
from trex.populations.mass_ratio_priors import broken_power_law_coefficients
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation, _disc_flux, _photometry_on_device
from trex.foundation.constants import G, Msun, Rsun
from trex.stellar_light import host_limb_darkening as ldc


def _stellar_splines_on_device(device):
    """Retain existing FITPACK stellar coefficients without requiring an atmosphere.

    Same uncalibrated scenario relations and mass domain as
    ``trex.stellar.stellar_relations``; this
    prepares no missing observation and introduces no new interpolation fit.
    """
    splines = {}
    for family, (_, _, radius, temperature) in zip(("cdwrf", "Torres"), stellar.stellar_relation_polynomials()):
        for quantity, polynomial in (("radius", radius), ("temperature", temperature)):
            splines[f"{quantity}_{family}"] = (models.jax.device_put(polynomial.x, device),
                                               models.jax.device_put(polynomial.c, device))
    return splines


def _stellar_spline_value(splines, name, mass):
    jnp = models.jnp
    knots, coefficients = splines[name]
    index = jnp.minimum(jnp.maximum(jnp.searchsorted(knots, mass, side="right") - 1, 0), coefficients.shape[1] - 1)
    delta = mass - knots[index]
    return ((coefficients[0, index] * delta + coefficients[1, index]) * delta + coefficients[2, index]) * delta + coefficients[3, index]


def _stellar_geometry_from_splines(splines, mass):
    """Original native stellar relation: solar mass -> solar radius and kelvin.

    Outside the retained scalar relation's documented mass domain returns NaN;
    metallicity is not an input to these particular retained scenario splines.
    """
    jnp = models.jnp
    hot = mass > stellar.STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR
    radius = jnp.where(hot, _stellar_spline_value(splines, "radius_Torres", mass),
                       _stellar_spline_value(splines, "radius_cdwrf", mass))
    teff = jnp.where(hot, _stellar_spline_value(splines, "temperature_Torres", mass),
                     _stellar_spline_value(splines, "temperature_cdwrf", mass))
    domain = (mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR) & (mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR)
    return jnp.where(domain, radius, jnp.nan), jnp.where(domain, teff, jnp.nan)


class FixedHostEclipsingPairUnitIntegrand:
    """Actual EB/EBx2P integrand; setup retains coefficients, evaluation is JAX."""

    claim_eligible = False

    def __init__(self, reference, *, device):
        if reference.hosted or reference.population is not None:
            raise NotImplementedError("JAX population EB integrands are not implemented")
        if reference.n_dimensions != 4:
            raise ValueError("the fixed-host EB unit cube has exactly four dimensions")
        if not models.HAS_JAX_TREX or not models.jax.config.jax_enable_x64:
            raise RuntimeError("the native EB integrand requires JAX/jaxoplanet float64")
        for name in ("effective_period", "host_mass", "host_radius", "host_teff", "exptime_days"):
            if not math.isfinite(float(getattr(reference, name))) or getattr(reference, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if reference.target_feh is None or not math.isfinite(reference.target_feh):
            raise ValueError("target_feh must be finite for secondary limb darkening")
        self.device = device
        self.time, self.weights, self.flux, self.sigma, self.u_coeffs = _photometry_on_device(reference, device)
        self.baseline = (None if reference.baseline_likelihood is None
                         else reference.baseline_likelihood.on_device(device))
        self.period_days = float(reference.effective_period)
        self.use_2xP = bool(getattr(reference, "use_2xP", False))
        self.host_mass = float(reference.host_mass)
        self.host_radius = float(reference.host_radius)
        self.host_teff = float(reference.host_teff)
        self.photon_light = None
        self.host_metallicity = reference.target_feh
        self.host_av_mag = 0.0
        if getattr(reference, "stellar_light", None) is not None:
            from trex.stellar_light.native import NativeStellarLight

            self.photon_light = NativeStellarLight(reference.stellar_light, device)
            self.host_av_mag = reference.stellar_light.target_av_mag
        self.companion_fluxratio = 0.0
        self.companion_is_host = False
        # Same inverse CDF as priors.sample_ecc: e = x**(1/power), with MDS17 Eq. 17's
        # density e**eta above the 10-d break, i.e. power 1 + eta. Moe & Di Stefano
        # (2017), ApJS 230, 15, DOI 10.3847/1538-4365/aa6fb6, Eq. 17.
        self.eccentricity_power = (priors.MOE2017_ECCENTRICITY_POWERLAW_SHORT
                                  if self.period_days <= priors.MOE2017_ECCENTRICITY_PERIOD_BREAK_DAYS
                                  else 1.0 + priors.moe2017_binary_eccentricity_eta(self.period_days))
        # Default component light: the TESS-band blackbody table the scalar and MC paths
        # read through bandpass.tess_surface_brightness_ratio, placed on the device once.
        self.band_table = None
        if self.photon_light is None:
            self.band_table = tuple(models.jax.device_put(value, device)
                                    for value in bandpass._tess_band_log_blackbody_flux())
        # The CPU inverse CDF's own coefficients (priors.sample_q), including the
        # documented ULP-safe hydrogen-burning floor. No interpolation of the inverse CDF.
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
        if self.photon_light is not None:
            self.grid_boundary_contract = self.photon_light.model_id
            self.stellar_relations_calibration_status = stellar.STELLAR_RELATIONS_CALIBRATION_STATUS
            return
        self._prepare_grid(reference.target_feh, device)

    def _prepare_grid(self, metallicity_dex, device):
        """Retain the scalar grid owner independently of a binary host/prior."""
        if metallicity_dex is None or not math.isfinite(float(metallicity_dex)):
            raise ValueError("host limb darkening requires finite metallicity")
        grid = ldc.load_triceratops_tess_grid()
        planes = grid._metallicity_planes
        metallicity = min(planes, key=lambda value: abs(value - metallicity_dex))
        temperatures, gravities, u1, u2 = grid._population_index
        self.grid_plane = planes.index(metallicity)
        self.grid = tuple(models.jax.device_put(value, device) for value in (
            temperatures, gravities, np.stack((u1, u2), axis=-1)))
        ext_t, ext_g, _planes, ext_u1, ext_u2 = grid._extension_tables
        self.extension_grid = (None if not ext_t.size else tuple(
            models.jax.device_put(value, device) for value in (
                ext_t, ext_g, np.stack((ext_u1, ext_u2), axis=-1))))
        ext_t, ext_g, _planes, ext_u1, ext_u2 = grid._phoenix_tables
        self.phoenix_grid = (None if not ext_t.size else tuple(
            models.jax.device_put(value, device) for value in (
                ext_t, ext_g, np.stack((ext_u1, ext_u2), axis=-1))))
        self.solar_grid_plane = planes.index(0.0)
        self.grid_metallicity_dex = metallicity
        self.grid_boundary_contract = ldc.LDC_GRID_CLAMP_CONTRACT
        self.stellar_relations_calibration_status = stellar.STELLAR_RELATIONS_CALIBRATION_STATUS

    def _mass_ratio(self, quantile):
        jnp = models.jnp
        target = quantile * (self.i1 + self.i2 + self.i3)
        first = (target * (self.p1 + 1.0) + self.q_floor ** (self.p1 + 1.0)) ** (1.0 / (self.p1 + 1.0))
        second = jnp.exp(jnp.log(self.steep_lower) + jnp.log1p(
            (target - self.i1) / self.a1 * self.s / self.steep_lower ** self.s) / self.s)
        twin_lower = jnp.maximum(self.q_twin, self.q_floor)
        log_lower = jnp.where(self.q_floor > self.q_twin, jnp.log(twin_lower), math.log(self.q_twin))
        lower_power = jnp.where(self.q_floor > self.q_twin, twin_lower ** self.s, self.q_twin ** self.s)
        third = jnp.exp(log_lower + jnp.log1p(
            (target - self.i1 - self.i2) / (self.a1 * self.a2) * self.s / lower_power) / self.s)
        q = jnp.where(target <= self.i1, first, jnp.where(target <= self.i1 + self.i2, second, third))
        # Same endpoint-rounding correction as the scalar inverse; it cannot
        # move an interior quantile or reduce the supported secondary domain.
        q = jnp.maximum(q, self.q_floor)
        return jnp.where(quantile == 0.0, self.q_floor, jnp.where(quantile == 1.0, 1.0, q))

    def _spline(self, name, mass):
        return _stellar_spline_value(self.splines, name, mass)

    def secondary_geometry(self, q):
        return _stellar_geometry_from_splines(self.splines, q * self.host_mass)

    def _axis(self, q):
        mass_g = self.host_mass * (1.0 + q) * Msun
        return (physical.GRAVITATIONAL_CONSTANT_CGS * mass_g * (self.period_days * physical.SECONDS_PER_DAY) ** 2
                / (4.0 * math.pi ** 2)) ** (1.0 / 3.0) / (self.host_radius * Rsun)

    def _band_low(self, radius_ratio, ecc, argp, axis):
        jnp = models.jnp
        factor = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(jnp.radians(argp)))
        return 1.0 - jnp.minimum(jnp.maximum((1.0 + radius_ratio) / (factor * axis), 0.0), 1.0)

    def prior_transform(self, unit_point):
        jnp = models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host EB unit point must have shape (4,)")
        q = self._mass_ratio(point[0])
        ecc = point[1] ** (1.0 / self.eccentricity_power)
        argp = point[2] * priors.FULL_TURN_DEGREES
        radius, _teff = self.secondary_geometry(q)
        # Same transform-only fallback as scalar EB when the secondary leaves
        # the stellar relation. The likelihood/population support still rejects
        # that draw; this is not a substituted physical light curve.
        radius = jnp.where(jnp.isfinite(radius), radius, self.host_radius)
        low = self._band_low(radius / self.host_radius, ecc, argp, self._axis(q))
        inclination_quantile = low + point[3] * (1.0 - low)
        norm = 1.0 / (math.cos(0.0) - math.cos(math.radians(90.0)))
        inc = jnp.degrees(jnp.arccos(1.0 - inclination_quantile / norm))
        return jnp.stack((q, ecc, argp, inc))

    def secondary_grid(self, q, radius, teff):
        """Host limb-darkening coefficients for a drawn star; explicit photon models use their own light."""
        jnp = models.jnp
        # Device log10 can differ by an ULP from NumPy at selection boundaries.
        logg = jnp.log10(G * q * self.host_mass * Msun / (radius * Rsun) ** 2)
        return self._grid_coefficients(teff, logg)

    def _grid_coefficients(self, teff, logg):
        """Port the owner's node/cell selection for Teff [K] and log10(g [cgs])."""
        jnp = models.jnp
        rounded_t = jnp.round(teff / ldc.TESS_TEFF_GRID_STEP_K) * ldc.TESS_TEFF_GRID_STEP_K
        rounded_g = jnp.round(logg / ldc.TESS_LOGG_GRID_STEP_DEX) * ldc.TESS_LOGG_GRID_STEP_DEX
        used_t = jnp.minimum(jnp.maximum(rounded_t, ldc.TESS_TEFF_GRID_MIN_K), ldc.TESS_TEFF_GRID_MAX_K)
        used_g = jnp.minimum(jnp.maximum(rounded_g, ldc.TESS_LOGG_GRID_MIN_DEX), ldc.TESS_LOGG_GRID_MAX_DEX)
        temperatures, gravities, table = self.grid
        ti, gi = jnp.searchsorted(temperatures, used_t), jnp.searchsorted(gravities, used_g)
        coefficients = table[self.grid_plane, ti, gi]
        valid = (temperatures[ti] == used_t) & (gravities[gi] == used_g) & jnp.isfinite(coefficients).all()
        clamped = ((teff < ldc.TESS_TEFF_GRID_MIN_K) | (teff > ldc.TESS_TEFF_GRID_MAX_K)
                   | (logg < ldc.TESS_LOGG_GRID_MIN_DEX) | (logg > ldc.TESS_LOGG_GRID_MAX_DEX))
        for extension, plane, eligible in (
            (self.phoenix_grid, 0, clamped & (teff < ldc.TESS_TEFF_GRID_MIN_K)
              & (self.grid_plane == self.solar_grid_plane)),
            (self.extension_grid, self.grid_plane, clamped),
        ):
            if extension is None:
                continue
            interpolated, complete = self._interpolate_grid(extension, plane, teff, logg)
            use_extension = clamped & eligible & complete
            coefficients = jnp.where(use_extension, interpolated, coefficients)
            valid = valid | use_extension
            clamped = clamped & ~use_extension
        return coefficients, valid, clamped

    def _interpolate_grid(self, extension, plane, teff, logg):
        jnp = models.jnp
        ext_t, ext_g, ext_table = extension
        ti, gi = jnp.searchsorted(ext_t, teff, side="left"), jnp.searchsorted(ext_g, logg, side="left")
        exact_t = (ti < ext_t.size) & (ext_t[jnp.minimum(ti, ext_t.size - 1)] == teff)
        exact_g = (gi < ext_g.size) & (ext_g[jnp.minimum(gi, ext_g.size - 1)] == logg)
        t0, t1 = jnp.where(exact_t, ti, ti - 1), ti
        g0, g1 = jnp.where(exact_g, gi, gi - 1), gi
        bracketed = (t0 >= 0) & (t1 < ext_t.size) & (g0 >= 0) & (g1 < ext_g.size)
        t0, t1 = jnp.clip(t0, 0, ext_t.size - 1), jnp.clip(t1, 0, ext_t.size - 1)
        g0, g1 = jnp.clip(g0, 0, ext_g.size - 1), jnp.clip(g1, 0, ext_g.size - 1)
        dt, dg = ext_t[t1] - ext_t[t0], ext_g[g1] - ext_g[g0]
        wt = jnp.where(dt > 0.0, (teff - ext_t[t0]) / jnp.where(dt > 0.0, dt, 1.0), 0.0)
        wg = jnp.where(dg > 0.0, (logg - ext_g[g0]) / jnp.where(dg > 0.0, dg, 1.0), 0.0)
        weights = ((1.0 - wg) * (1.0 - wt), (1.0 - wg) * wt, wg * (1.0 - wt), wg * wt)
        corners = (ext_table[plane, g0, t0], ext_table[plane, g0, t1],
                   ext_table[plane, g1, t0], ext_table[plane, g1, t1])
        complete = bracketed
        interpolated = jnp.zeros_like(corners[0])
        # Claret (2017) tables 15/25: bilinear in K and log10(cgs), only published corners.
        for corner, weight in zip(corners, weights):
            complete = complete & (jnp.isfinite(corner) | (weight == 0.0)).all()
            interpolated = interpolated + jnp.where(jnp.isfinite(corner), corner, 0.0) * weight
        return interpolated, complete

    def secondary_flux_ratio(self, q, radius, teff):
        """Same component-light model as scalar/MC, in F_secondary/F_host units."""
        if self.photon_light is not None:
            return self.photon_light.pair_ratio(
                self.host_teff, self.host_mass, self.host_radius,
                teff, q*self.host_mass, radius, self.host_metallicity, self.host_av_mag)
        jnp = models.jnp
        log_temperature, log_flux = self.band_table
        return (radius/self.host_radius)**2 * jnp.exp(
            jnp.interp(jnp.log(teff), log_temperature, log_flux)
            - jnp.interp(jnp.log(self.host_teff), log_temperature, log_flux))

    def in_band_table(self, teff):
        """The scalar ratio refuses temperatures outside the blackbody table; so does this path."""
        low, high = bandpass.TESS_BAND_TEMPERATURE_MIN_K, bandpass.TESS_BAND_TEMPERATURE_MAX_K
        host = models.jnp.asarray(self.host_teff)
        return (teff >= low) & (teff <= high) & (host >= low) & (host <= high)

    def evaluate(self, unit_point):
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(unit_point, dtype=jnp.float64)
        if point.shape != (4,):
            raise ValueError("the fixed-host EB unit point must have shape (4,)")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf), jnp.asarray(True), jnp.asarray(False))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan), jnp.asarray(False), jnp.asarray(False))

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
            # G21 §2.4.4: disjoint q slots, preserving the full-prior branch mass.
            support &= (q >= self.q_twin) if self.use_2xP else (q < self.q_twin)

            def on_support(_unused):
                primary_u = self.u_coeffs
                primary_profile, secondary_profile = None, None
                if self.photon_light is None:
                    secondary_u, grid_valid, clamped = self.secondary_grid(q, radius, teff)
                    ratio = self.secondary_flux_ratio(q, radius, teff)
                    grid_valid = grid_valid & self.in_band_table(teff)
                else:
                    first, first_valid = self.photon_light.prepare_component(
                        self.host_teff, self.host_mass, self.host_radius, self.host_metallicity, self.host_av_mag)
                    second, second_valid = self.photon_light.prepare_component(
                        teff, q*self.host_mass, radius, self.host_metallicity, self.host_av_mag)
                    # Chen 2019 eq_Mag: same-distance R² surface-photon ratio, extinction already applied.
                    ratio = (radius/self.host_radius)**2 * second.surface_photon_flux/first.surface_photon_flux
                    primary_u, secondary_u = jnp.stack((first.u1, first.u2)), jnp.stack((second.u1, second.u2))
                    primary_profile, secondary_profile = first.angular_profile, second.angular_profile
                    grid_valid, clamped = first_valid & second_valid, jnp.asarray(False)
                # One Kepler solve supplies both members: the primary at the
                # host-frame separation with the companion radius ratio k, the
                # secondary at d/k in companion radii with the host radius 1/k
                # and the complementary front flag. This is the same pair
                # geometry as the Monte Carlo paths; it is not a second orbit.
                separation, front = _jax_orbits._sky_geometry(
                    self.time, self.period_days, axis, inc, ecc, argp, 0.0)
                primary_instant = _disc_flux(separation, k, primary_u, front,
                                             angular_profile=primary_profile)
                secondary_instant = _disc_flux(separation / k, 1.0 / k, secondary_u, ~front,
                                               angular_profile=secondary_profile)
                primary = jnp.sum(primary_instant.reshape(-1, self.weights.shape[-1]) * self.weights, axis=1)
                secondary = jnp.sum(secondary_instant.reshape(-1, self.weights.shape[-1]) * self.weights, axis=1)
                valid_p = jnp.isfinite(separation).all() & jnp.isfinite(primary_instant).all()
                valid_s = ((k > 0.0) & jnp.isfinite(separation).all()
                           & jnp.isfinite(secondary_instant).all())
                from trex.models.flux_calibration import binary_flux_weights

                weights = binary_flux_weights(ratio, self.companion_fluxratio,
                                              self.companion_is_host, array_module=jnp)
                model = weights[0] * primary + weights[1] * secondary + weights[2]
                likelihood = (self.baseline(model) if self.baseline is not None else
                              -0.5 * jnp.sum((self.flux - model) ** 2 / self.sigma ** 2))
                value = likelihood + jnp.log(band)
                valid = grid_valid & valid_p & valid_s & jnp.isfinite(value)
                return UnitIntegrandEvaluation(jnp.where(valid, value, jnp.nan), valid, clamped)

            # A companion mass outside the stellar relations is zero prior
            # support (-inf, valid), exactly as the scalar integrand returns
            # -inf before the relation is evaluated; the relation's NaN there is
            # not a numerical failure and must not abort the native sampler.
            mass = q * self.host_mass
            in_relation_domain = ((mass >= stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR)
                                  & (mass <= stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR))

            def on_finite(_unused):
                return jax.lax.cond(in_relation_domain & support, on_support, lambda _unused: zero, None)

            return jax.lax.cond(jnp.isfinite(parameters).all()
                                & (jnp.isfinite(teff) | ~in_relation_domain),
                                on_finite, lambda _unused: failure, None)

        return jax.lax.cond(jnp.isfinite(point).all(),
                            lambda _unused: jax.lax.cond(jnp.all((point >= 0.0) & (point < 1.0)), on_cube, lambda _unused: zero, None),
                            lambda _unused: failure, None)
