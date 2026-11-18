"""Host measure, prior transform and likelihood parity against the scalar engine."""
import math

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import truncnorm

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence as nested
from trex.evidence.host_measure import CharacterizedHostIntegrand, HostMeasure
from trex.evidence.nested_integrands import _EclipsingPairIntegrand, _TransitingPlanetIntegrand
from trex.foundation.constants import G, Msun, Rsun
from trex.photometry.baseline import GaussianBaseline


def distribution(continuous):
    record = {"method": "synthetic joint stellar measure", "mass_solar": [.8, 1.1],
              "radius_solar": [.8, 1.1], "teff_k": [5000., 6000.], "weights": [.25, .75]}
    if continuous:
        record["marginals"] = {"mass_mean": 1., "mass_std": .03,
                               "logg_mean": math.log10(G * Msun / Rsun**2), "logg_std": .02,
                               "logteff_mean": math.log10(5772.), "logteff_std": .003}
    return record


def test_continuous_interior_tails_are_not_clipped_or_given_artificial_atoms():
    record = distribution(True)
    measure = HostMeasure(record, 0.)
    quantiles = np.array([[1e-10, 2e-10, 3e-10], [1e-9, 2e-9, 3e-9],
                          [1-1e-10, 1-2e-10, 1-3e-10]])
    mass, radius, teff, _u1, _u2 = measure.resolve(quantiles)
    p = record["marginals"]
    np.testing.assert_allclose(truncnorm.cdf(mass, -p["mass_mean"]/p["mass_std"], np.inf,
                                           loc=p["mass_mean"], scale=p["mass_std"]), quantiles[:, 0],
                               rtol=1e-5, atol=1e-15)
    logg = np.log10(G * Msun / Rsun**2 * mass / radius**2)
    np.testing.assert_allclose(ndtr((logg-p["logg_mean"])/p["logg_std"]), quantiles[:, 1], rtol=1e-5, atol=1e-15)
    np.testing.assert_allclose(ndtr((np.log10(teff)-p["logteff_mean"])/p["logteff_std"]), quantiles[:, 2],
                               rtol=1e-5, atol=1e-15)
    assert mass[0] != mass[1] and radius[0] != radius[1] and teff[0] != teff[1]


@pytest.mark.parametrize("continuous", [False, True], ids=["joint-grid", "continuous-labels"])
@pytest.mark.parametrize("planet", [True, False], ids=["TP", "EB"])
@pytest.mark.parametrize("baseline_mode", [None, "additive", "multiplicative"])
def test_native_characterized_prior_and_likelihood_match_scalar(continuous, planet, baseline_mode):
    time = np.linspace(-.15, .15, 25)
    flux, sigma = np.ones(time.size), np.full(time.size, .003)
    baseline = None
    if baseline_mode is not None:
        baseline = GaussianBaseline(time, np.column_stack((np.ones(time.size), time)), np.array([.001, -.002]),
                                    np.array([[1e-6, 1e-7], [1e-7, 2e-6]]), baseline_mode,
                                    "synthetic independent proper prior").prepare(time, flux, sigma).for_host(.2)
    kwargs = dict(time=time, flux=flux, sigma=sigma, host_mass=1., host_radius=1., u1=.4, u2=.2,
                  exptime_days=120/86400, bin_integration_days=0., nsamples=3, population=None,
                  n_dimensions=4, target_feh=0., host_teff_k=5772., baseline_likelihood=baseline)
    base = (_TransitingPlanetIntegrand(period_days=3., use_flat_priors=False, **kwargs) if planet else
            _EclipsingPairIntegrand(effective_period=3., use_2xP=False, **kwargs))
    scalar = CharacterizedHostIntegrand(base, distribution(continuous))
    assert nested._cpu_refinement_integrand(scalar) is scalar
    device = models.jax.devices("cpu")[0]
    native = scalar.jax_unit_integrand(device=device)
    points = np.random.default_rng(1927).uniform(.05, .95, size=(16, scalar.n_dimensions))
    scalar_parameters = np.array([scalar.prior_transform(point) for point in points])
    native_parameters = np.asarray(models.jax.jit(models.jax.vmap(native.prior_transform))(points))
    np.testing.assert_allclose(native_parameters, scalar_parameters, rtol=2e-11, atol=2e-11)
    scalar_values = np.array([scalar.log_likelihood(parameters) for parameters in scalar_parameters])
    native_values = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    assert np.asarray(native_values.numerically_valid).all()
    values = np.asarray(native_values.log_integrand)
    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(scalar_values))
    assert np.isfinite(values).sum() >= 5
    # Existing scalar/native kernel bar; paired values are never replaced by aggregate lnZ.
    np.testing.assert_allclose(values, scalar_values, rtol=1e-4, atol=1e-6)
