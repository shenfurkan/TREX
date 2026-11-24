"""Native source-associated stellar measure, with unchanged orbit/likelihood owners.

Discrete rows retain their joint M/R/Teff weights. Continuous stellar labels
assume independent mass, logg and logTeff marginals, as the scalar measure does;
radius is derived from the same mass and gravity draw. This is not a measured
label covariance or a stellar-population calibration.
"""
import copy
import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.evidence._jax_bound_unit_integrand import _ratio_coefficients
from trex.evidence._jax_eb_unit_integrand import FixedHostEclipsingPairUnitIntegrand
from trex.evidence._jax_unit_integrand import FixedHostPlanetUnitIntegrand, UnitIntegrandEvaluation
from trex.foundation.constants import G, Msun, Rsun
from trex.foundation import physical_constants as physical
from trex.populations import priors


class CharacterizedHostUnitIntegrand:
    """Device twin of ``host_measure.CharacterizedHostIntegrand``.

    The scalar owner admits only the reference light model, so continuous
    marginals take grid limb darkening here and never a ``NativeStellarLight``.
    """

    def __init__(self, reference, *, device):
        self.device = device
        self.measure = reference.measure
        self.n_dimensions = reference.n_dimensions
        self.planet = hasattr(reference.base, "period_days")
        if self.measure.marginals is not None:
            # Continuous draws pick grid coefficients per draw; discrete rows carry their own.
            self.atmosphere = object.__new__(FixedHostEclipsingPairUnitIntegrand)
            self.atmosphere._prepare_grid(reference.base.target_feh, device)
            self.mass_floor = self.measure.mass_floor
        self.base = (FixedHostPlanetUnitIntegrand(reference.base, device=device) if self.planet
                     else FixedHostEclipsingPairUnitIntegrand(reference.base, device=device))
        self.cdf = models.jax.device_put(self.measure.cdf, device)
        if self.measure.marginals is None:
            u1, u2 = self.measure.coefficients()
            self.rows = tuple(models.jax.device_put(value, device) for value in (
                self.measure.values["mass_solar"], self.measure.values["radius_solar"],
                self.measure.values["teff_k"], u1, u2))

    def _view(self, point):
        jax, jnp = models.jax, models.jnp
        if self.measure.marginals is None:
            # cdf[-1] == 1 and u < 1 keep the row in range; the clip states it.
            index = jnp.minimum(jnp.searchsorted(self.cdf, point[-1], side="left"), self.cdf.shape[0] - 1)
            mass, radius, teff, u1, u2 = (row[index] for row in self.rows)
            valid, clamped = jnp.asarray(True), jnp.asarray(False)
        else:
            p = self.measure.marginals
            unit = point[-3:]
            # Only measure-zero endpoints move; all interior tails remain intact.
            unit = jnp.where(unit == 0.0, np.nextafter(0.0, 1.0),
                             jnp.where(unit == 1.0, np.nextafter(1.0, 0.0), unit))
            lower = jax.scipy.special.ndtr(-p["mass_mean"] / p["mass_std"])
            survival = jax.scipy.special.ndtr(p["mass_mean"] / p["mass_std"])
            # Invert the nearer tail, avoiding cancellation at either endpoint.
            normal = jnp.where(unit[0] < .5, jax.scipy.special.ndtri(lower + survival * unit[0]),
                               -jax.scipy.special.ndtri(survival * (1.0 - unit[0])))
            mass = p["mass_mean"] + p["mass_std"] * normal
            mass = jnp.where(mass > 0.0, mass, self.mass_floor)
            logg = p["logg_mean"] + p["logg_std"] * jax.scipy.special.ndtri(unit[1])
            teff = 10.0 ** (p["logteff_mean"] + p["logteff_std"] * jax.scipy.special.ndtri(unit[2]))
            radius = jnp.sqrt(G * Msun / Rsun**2 * mass / 10.0**logg)
            coefficients, valid, clamped = self.atmosphere._grid_coefficients(teff, logg)
            u1, u2 = coefficients
        view = copy.copy(self.base)
        view.host_mass, view.host_radius, view.host_teff = mass, radius, teff
        view.u_coeffs = jnp.stack((u1, u2))
        valid &= (jnp.isfinite(jnp.stack((mass, radius, teff))).all()
                  & (mass > 0) & (radius > 0) & (teff > 0))
        support = jnp.asarray(True)
        if self.planet:
            view.a_rs = (physical.GRAVITATIONAL_CONSTANT_CGS * mass * Msun * (view.period_days * physical.SECONDS_PER_DAY)**2
                         / (4.0 * math.pi**2))**(1.0 / 3.0) / (radius * Rsun)
            view.high_mass_radius_prior = mass > priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
        else:
            support = mass >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
            coefficients = _ratio_coefficients(jnp.maximum(mass, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR),
                                               priors.MOE2017_Q_SLOPE_HIGH, priors.MOE2017_TWIN_FRACTION)
            view.q_floor, view.steep_lower, view.a1, view.a2, view.i1, view.i2, view.i3, view.s = coefficients
        return view, support, valid, clamped

    def prior_transform(self, point):
        jnp = models.jnp
        point = jnp.asarray(point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError("characterized-host point has the wrong dimension")
        view, support, _valid, _clamped = self._view(point)
        # Outside the Moe q domain there is no support; return the same finite
        # placeholder orbit as the scalar CharacterizedHostIntegrand.
        orbit = models.jax.lax.cond(support, lambda _: view.prior_transform(point[:4]),
                                     lambda _: jnp.array([1., 0., 0., 90.]), None)
        return jnp.concatenate((orbit, point[4:]))

    def evaluate(self, point):
        jax, jnp = models.jax, models.jnp
        point = jnp.asarray(point, dtype=jnp.float64)
        if point.shape != (self.n_dimensions,):
            raise ValueError("characterized-host point has the wrong dimension")
        zero = UnitIntegrandEvaluation(jnp.asarray(-jnp.inf), jnp.asarray(True), jnp.asarray(False))
        failure = UnitIntegrandEvaluation(jnp.asarray(jnp.nan), jnp.asarray(False), jnp.asarray(False))

        def on_cube(_):
            view, support, valid, clamped = self._view(point)
            def retained(_):
                result = view.evaluate(point[:4])
                return result._replace(grid_boundary_clamped=result.grid_boundary_clamped | clamped)
            return jax.lax.cond(valid, lambda _: jax.lax.cond(support, retained, lambda _: zero, None),
                                lambda _: failure, None)

        return jax.lax.cond(jnp.isfinite(point).all(),
            lambda _: jax.lax.cond(jnp.all((point >= 0) & (point < 1)), on_cube, lambda _: zero, None),
            lambda _: failure, None)
