"""Device-native interpolation of the identical compiled photon atmosphere."""

from trex.accelerators import _jax_models as models
from trex.foundation.constants import G, Msun, Rsun
from .components import ComponentLight
from .interpolation import interpolate_photons


class NativeStellarLight:
    """No network, source reader or host callback during native evaluation."""

    def __init__(self, light, device):
        """Upload an already-validated grid; require the float64 native backend."""
        if not models.HAS_JAX_TREX or models.jax is None or not models.jax.config.jax_enable_x64:
            raise RuntimeError("native stellar light requires JAX/jaxoplanet float64")
        self.axes = tuple(models.jax.device_put(a, device) for a in light.grid.axes)
        self.table = models.jax.device_put(light.grid.table, device)
        self.target_av_mag = light.target_av_mag
        self.model_id = light.model_id
        self.angular = (None if light.grid.angular is None else
                        tuple(models.jax.device_put(a, device) for a in light.grid.angular))

    def angular_profile(self, temperature, mass, radius, metallicity, av_mag):
        """Same original corner rays on device, without callbacks or averaged mu."""
        if self.angular is None:
            return None, models.jnp.asarray(True)
        from .angular import interpolate_angular
        coordinates, physical = self._coordinates(temperature, mass, radius, metallicity, av_mag)
        mu, intensity, valid = interpolate_angular(
            self.axes, self.angular, coordinates, xp=models.jnp)
        return (mu, intensity), valid & physical

    def _coordinates(self, temperature, mass, radius, metallicity, av_mag):
        jnp = models.jnp
        if any(v is None for v in (temperature, mass, radius, metallicity, av_mag)):
            raise ValueError("physical atmosphere requires stellar coordinates, composition and extinction")
        values = [jnp.asarray(v) for v in (temperature, mass, radius, metallicity, av_mag)]
        if any(v.dtype.kind not in "iuf" for v in values):
            raise ValueError("physical atmosphere coordinates must be real numbers")
        temperature, mass, radius, metallicity, av_mag = jnp.broadcast_arrays(*values)
        valid = jnp.all(jnp.stack([jnp.isfinite(v) for v in (temperature, mass, radius, metallicity, av_mag)]), axis=0)
        valid = valid & (temperature > 0) & (mass > 0) & (radius > 0) & (av_mag >= 0)
        # g=GM/R² [cm s^-2], identical to the host coordinate producer.
        gravity = jnp.log10(G*mass*Msun/(radius*Rsun)**2)
        return (temperature, gravity, metallicity, av_mag), valid

    def atmosphere(self, temperature, mass, radius, metallicity, av_mag, *, tabulated=False):
        """Photon flux, diagnostic (u1,u2), validity; tabulated skips only the law veto."""
        coordinates, physical = self._coordinates(temperature, mass, radius, metallicity, av_mag)
        return self._surface(coordinates, physical, tabulated=tabulated)

    def _surface(self, coordinates, physical, *, tabulated):
        jnp = models.jnp
        values, valid = interpolate_photons(
            self.axes, self.table, coordinates, xp=jnp)
        valid = valid & physical
        if not tabulated:
            from .passband import quadratic_nonnegative

            valid = valid & quadratic_nonnegative(values, xp=jnp)
        coefficients = -values[..., 2:]/values[..., 1, None]
        return jnp.where(valid, values[..., 0], jnp.nan), coefficients, valid

    def prepare_component(self, temperature, mass, radius, metallicity, av_mag):
        """Device-native prepared photon light/profile and numerical validity.

        Inputs are K, solar mass/radius, own [M/H] dex and A_V mag. Invalid
        atmosphere support stays a numerical failure, not a population-prior cut.
        No host callback, cache or altered angular quadrature is introduced.
        """
        coordinates, physical = self._coordinates(temperature, mass, radius, metallicity, av_mag)
        flux, coefficients, valid = self._surface(coordinates, physical, tabulated=self.angular is not None)
        profile = None
        if self.angular is not None:
            from .angular import interpolate_angular
            mu, intensity, angular_valid = interpolate_angular(self.axes, self.angular, coordinates, xp=models.jnp)
            profile = (mu, intensity)
            valid = valid & angular_valid
        return ComponentLight(flux, coefficients[..., 0], coefficients[..., 1], profile), valid

    def pair_ratio(self, host_t, host_m, host_r, secondary_t, secondary_m, secondary_r, mh, av):
        first, _, first_valid = self.atmosphere(host_t, host_m, host_r, mh, av, tabulated=self.angular is not None)
        second, _, second_valid = self.atmosphere(
            secondary_t, secondary_m, secondary_r, mh, av, tabulated=self.angular is not None)
        # Chen 2019 eq_Mag: both surface factors already attenuated, same-distance ratio.
        ratio = (secondary_r/host_r)**2*second/first
        return models.jnp.where(first_valid & second_valid, ratio, models.jnp.nan)


def physical_population_coefficients(light, population):
    """Fixed background rows use their own extinction, never target extinction."""
    if population.av_mag is None:
        raise ValueError("physical background atmosphere requires retained A_V")
    coordinates = light._coordinates(population.teff_k, population.mass_solar, population.radius_solar,
                                     population.mh_dex, population.av_mag)
    values, _ = interpolate_photons(light.grid.axes, light.grid.table, coordinates)
    # Keep every row. Uncovered hosts remain NaN numerical failures, never prior cuts.
    return -values[..., 2:]/values[..., 1, None]
