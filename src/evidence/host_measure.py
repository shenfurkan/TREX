"""Source-associated grid or continuous stellar measures shared by both estimators."""
import copy
import math

import numpy as np


def positive_mass_floor(marginals) -> float:
    """Smallest positive mass [Msun] the positive-conditioned normal mass marginal returns.

    The smallest representable quantile can still round onto the truncation
    point M = 0, which is outside the measure's support. Such a draw takes a
    cumulative probability 1e-12 (relative) above the truncation: far above
    ndtri's rounding there, and a zero-measure change to the integral. Shared
    by the scalar and JAX transforms so both resolve the endpoint identically.
    """
    from scipy.special import ndtr, ndtri

    start = ndtr(-marginals["mass_mean"] / marginals["mass_std"])
    floor = float(marginals["mass_mean"] + marginals["mass_std"]
                  * ndtri(start + max(start * 1e-12, np.finfo(float).tiny)))
    if not floor > 0.0:
        raise ValueError("continuous stellar mass marginal has no representable positive lower endpoint")
    return floor


class HostMeasure:
    def __init__(self, record, metallicity, provenance=None):
        if not isinstance(record, dict) or not isinstance(record.get("method"), str):
            raise ValueError("stellar distribution must name its method")
        self.values = {}
        for key in ("mass_solar", "radius_solar", "teff_k", "weights"):
            raw = record.get(key)
            if (not isinstance(raw, (list, np.ndarray))
                    or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(raw, dtype=object).flat)):
                raise ValueError(f"stellar distribution {key} must be a numeric array")
            values = np.asarray(raw, dtype=np.float64)
            if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"stellar distribution {key} must be finite and positive")
            self.values[key] = values
        weights = self.values["weights"]
        if any(v.shape != weights.shape for v in self.values.values()) or not math.isclose(float(weights.sum()), 1., rel_tol=1e-10):
            raise ValueError("stellar distribution arrays must match and weights sum to one")
        self.cdf = np.cumsum(weights / weights.sum())
        self.cdf[-1] = 1.0  # Floating-point sum, not posterior clipping.
        self.metallicity = metallicity
        self.provenance = provenance
        self._coefficients = None
        self.marginals = record.get("marginals")
        self.n_dimensions = 1 if self.marginals is None else 3
        if self.marginals is not None:
            keys = ("mass_mean", "mass_std", "logg_mean", "logg_std", "logteff_mean", "logteff_std")
            if not isinstance(self.marginals, dict) or set(self.marginals) != set(keys):
                raise ValueError("continuous stellar marginals must declare every mean and uncertainty")
            if any(not isinstance(self.marginals[k], (int, float, np.integer, np.floating))
                   or isinstance(self.marginals[k], (bool, np.bool_))
                   or not math.isfinite(self.marginals[k]) for k in keys):
                raise ValueError("stellar marginal parameters must be finite numbers")
            if any(self.marginals[k] <= 0 for k in ("mass_mean", "mass_std", "logg_std", "logteff_std")):
                raise ValueError("stellar marginal mass and uncertainties must be positive")
            self.mass_floor = positive_mass_floor(self.marginals)

    def resolve(self, unit):
        """Joint M/R/Teff draw; radius follows the same draw's M and gravity."""
        if self.marginals is None:
            indices = self.indices(unit)
            u1, u2 = self.coefficients()
            return tuple(self.values[k][indices] for k in ("mass_solar", "radius_solar", "teff_k")) + (u1[indices], u2[indices])
        from scipy.special import ndtri
        from scipy.stats import truncnorm
        from trex.foundation.constants import G, Msun, Rsun
        from trex.evidence.mc_population import _host_grid_coefficients
        unit = np.asarray(unit, dtype=float)
        if unit.ndim == 0 or unit.shape[-1] != 3 or np.any((unit < 0) | (unit > 1)) or not np.all(np.isfinite(unit)):
            raise ValueError("continuous stellar quantiles must lie in [0,1]")
        # Unit-cube endpoints have zero measure.  Keep a finite, representable
        # endpoint for the prior transform; every interior quantile is untouched.
        unit = np.where(unit == 0.0, np.nextafter(0.0, 1.0),
                        np.where(unit == 1.0, np.nextafter(1.0, 0.0), unit))
        p = self.marginals
        lower = -p["mass_mean"] / p["mass_std"]
        mass = truncnorm.ppf(unit[..., 0], lower, np.inf, loc=p["mass_mean"], scale=p["mass_std"])
        mass = np.where(mass > 0.0, mass, self.mass_floor)
        logg = p["logg_mean"] + p["logg_std"] * ndtri(unit[..., 1])
        temperature = 10.0**(p["logteff_mean"] + p["logteff_std"] * ndtri(unit[..., 2]))
        radius = np.sqrt(G * Msun / Rsun**2 * mass / 10.0**logg)
        scalar = np.ndim(mass) == 0
        u1, u2 = _host_grid_coefficients(np.atleast_1d(temperature), np.atleast_1d(mass), np.atleast_1d(radius),
                                        self.metallicity, scenario=getattr(self.provenance, "scenario", "characterized-neighbor"), provenance=self.provenance)
        if scalar:
            return float(mass), float(radius), float(temperature), u1[0], u2[0]
        return mass, radius, temperature, u1, u2

    def marginal_median(self, name):
        """Median of one marginal (``teff_k`` in K, ``mass_solar``, ``radius_solar``).

        With continuous marginals, Teff (log-normal in log10 T) and mass
        (positive-conditioned normal) have exact medians. Otherwise the grid
        median uses midpoint cumulative weights: plain cumulative weights stop
        at the node below the centre of a symmetric quadrature (12-node
        SHBoost grids), biasing the median low.
        A continuous radius is derived jointly from mass and gravity and has
        no median supplied by this method; requesting it raises ValueError.
        """
        if self.marginals is not None and name == "teff_k":
            return float(10.0 ** self.marginals["logteff_mean"])
        if self.marginals is not None and name == "mass_solar":
            from scipy.special import ndtr, ndtri
            mean, sigma = self.marginals["mass_mean"], self.marginals["mass_std"]
            lower = ndtr(-mean / sigma)
            return float(mean + sigma * ndtri(lower + 0.5 * (1.0 - lower)))
        if self.marginals is not None:
            raise ValueError(f"continuous stellar marginal median is unavailable for {name}")
        values = self.values[name]
        order = np.argsort(values)
        weights = self.values["weights"][order]
        cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
        return float(np.interp(.5, cdf, values[order]))

    def indices(self, unit):
        unit = np.asarray(unit, dtype=float)
        if not np.all(np.isfinite(unit)) or np.any((unit < 0) | (unit > 1)):
            raise ValueError("stellar measure quantile must lie in [0,1]")
        return np.minimum(np.searchsorted(self.cdf, unit, side="left"), self.cdf.size - 1)

    def coefficients(self):
        if self._coefficients is None:
            from trex.evidence.mc_population import _host_grid_coefficients
            self._coefficients = _host_grid_coefficients(
                self.values["teff_k"], self.values["mass_solar"], self.values["radius_solar"],
                self.metallicity, scenario=getattr(self.provenance, "scenario", "characterized-neighbor"), provenance=self.provenance)
        return self._coefficients


class CharacterizedHostIntegrand:
    """Extra stellar unit-CDF dimensions; delegate orbit/occultor physics unchanged.

    MC and nested sampling integrate exactly the same stellar measure.
    No central-host refinement may replace a marginalized-host MC evidence.
    """
    def __init__(self, base, record):
        if base.population is not None or base.stellar_light is not None:
            raise ValueError("characterized resolved hosts currently require the reference light model and no unresolved population")
        self.base = base
        self.time, self.nsamples = base.time, base.nsamples
        self.measure = HostMeasure(record, base.target_feh)
        if self.measure.marginals is None:
            self.measure.coefficients()
        self.n_dimensions = base.n_dimensions + self.measure.n_dimensions
        self.population = None
        self.target_feh = base.target_feh
        self.hosted = False
        self.baseline_likelihood = base.baseline_likelihood

    def _host(self, quantile):
        from trex.models.orbital_mechanics import semi_major_axis_cgs
        from trex.foundation.constants import Msun, Rsun
        host = copy.copy(self.base)
        host.host_mass, host.host_radius, host.host_teff, host.u1, host.u2 = (
            float(v) for v in self.measure.resolve(quantile))
        if hasattr(host, "period_days"):
            host.a_cm = semi_major_axis_cgs(host.period_days, host.host_mass * Msun)
            host.a_rs = host.a_cm / (host.host_radius * Rsun)
        return host

    def prior_transform(self, unit):
        unit = np.asarray(unit, dtype=float)
        if unit.shape != (self.n_dimensions,) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
            raise ValueError("characterized-host unit point must match its dimension and lie in [0,1]")
        width = self.measure.n_dimensions
        quantile = unit[-width:] if width > 1 else unit[-1]
        host = self._host(quantile)
        if hasattr(host, "effective_period"):
            from trex.populations.priors import MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
            if host.host_mass < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
                # Orbit parameters are unused: log_likelihood rejects this host before evaluation.
                return np.r_[np.zeros(self.base.n_dimensions), unit[-width:]]
        return np.r_[host.prior_transform(unit[:-width]), unit[-width:]]

    def log_likelihood(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.n_dimensions,):
            raise ValueError("characterized-host parameters must match the integrand dimension")
        width = self.measure.n_dimensions
        quantile = parameters[-width:] if width > 1 else parameters[-1]
        host = self._host(quantile)
        if hasattr(host, "effective_period"):
            from trex.populations.priors import MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
            if host.host_mass < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
                return -math.inf
        return host.log_likelihood(parameters[:-width])

    def jax_unit_integrand(self, *, device):
        from trex.evidence._jax_host_integrand import CharacterizedHostUnitIntegrand

        return CharacterizedHostUnitIntegrand(self, device=device)
