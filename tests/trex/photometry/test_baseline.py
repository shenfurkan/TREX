"""Independent normalized Gaussian convolution for the explicit native baseline."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from trex.foundation.errors import NumericalLikelihoodError
from trex.photometry.baseline import GaussianBaseline


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("fraction", [1., .07])
def test_prepared_density_matches_independent_dense_gaussian(mode, fraction):
    t = np.linspace(-.2, .2, 13)
    design = np.column_stack((np.ones_like(t), t, t**2))
    covariance = np.array([[4., .2, -.3], [.2, 2., .4], [-.3, .4, 1.]])*1e-6
    mu = np.array([.002, -.003, .004])
    host = 1-.04*np.exp(-t*t/.001)
    aperture = 1+fraction*(host-1)
    sigma = np.linspace(.0007, .0014, t.size)
    flux = aperture+.001*np.sin(t*30)
    basis = design if mode == "additive" else aperture[:, None]*design
    expected = multivariate_normal.logpdf(
        flux, mean=aperture+basis@mu,
        cov=np.diag(sigma**2)+basis@covariance@basis.T)
    prior = GaussianBaseline(t, design, mu, covariance, mode, "synthetic independent prior")
    density = prior.prepare(t, flux, sigma).for_host(fraction)
    assert density(host) == pytest.approx(expected, abs=2e-12)


def test_small_proper_prior_recovers_fully_normalized_white_density():
    time = np.linspace(-.1, .1, 7)
    sigma = .002
    model = np.ones(time.size)
    flux = model+np.linspace(-.001, .001, time.size)
    prior = GaussianBaseline(time, np.ones((time.size, 1)), np.zeros(1),
                             np.array([[1e-30]]), "additive", "synthetic white limit")
    expected = -.5*np.sum(((flux-model)/sigma)**2+np.log(2*np.pi*sigma**2))
    assert prior.prepare(time, flux, sigma)(model) == pytest.approx(expected, abs=1e-13)


def test_multiplicative_density_recomputes_covariance_for_changed_physical_mean():
    time = np.linspace(-.2, .2, 9)
    prior = GaussianBaseline(time, np.ones((time.size, 1)), np.array([.01]),
                             np.array([[.03**2]]), "multiplicative", "synthetic fixed prior")
    flux, sigma = np.ones(time.size), .01
    density = prior.prepare(time, flux, sigma)
    for model in (np.ones(time.size), 1-.5*np.exp(-time*time/.005), np.ones(time.size)):
        expected = multivariate_normal.logpdf(flux, mean=model*(1+.01),
            cov=np.eye(time.size)*sigma**2+np.outer(model, model)*.03**2)
        assert density(model) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
def test_supported_nonfinite_baseline_density_is_a_numerical_failure(mode):
    time = np.array([0., .1])
    prior = GaussianBaseline(time, np.zeros((2, 1)), np.zeros(1), np.ones((1, 1)), mode, "zero design")
    density = prior.prepare(time, np.ones(2), np.full(2, 1e-200))
    # The residual is finite, but its square cannot be represented in float64.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(NumericalLikelihoodError, match="baseline likelihood") as failure:
            density(np.array([.9, 1.]))
    assert failure.value.count == 1
    with pytest.raises(NumericalLikelihoodError, match="physical prediction"):
        density(np.array([np.nan, 1.]))
    with pytest.raises(ValueError, match="physical prediction"):
        density(np.ones(3))


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("fraction", [1., .07])
def test_native_density_matches_independent_dense_gaussian(mode, fraction):
    from trex.accelerators import _jax_models as models

    time = np.linspace(-.2, .2, 13)
    design = np.column_stack((np.ones_like(time), time))
    covariance = np.array([[4., .2], [.2, 2.]]) * 1e-6
    mu = np.array([.002, -.003])
    host = 1 - .04 * np.exp(-time*time/.001)
    aperture = 1 + fraction * (host - 1)
    sigma = np.linspace(.0007, .0014, time.size)
    flux = aperture + .001*np.sin(time*30)
    prior = GaussianBaseline(time, design, mu, covariance, mode, "synthetic independent prior")
    density = prior.prepare(time, flux, sigma).for_host(fraction).on_device(models.jax.devices("cpu")[0])
    # Full dense convolution, independently of either Woodbury implementation.
    basis = design if mode == "additive" else aperture[:, None] * design
    expected = multivariate_normal.logpdf(flux, mean=aperture+basis@mu,
                                         cov=np.diag(sigma**2)+basis@covariance@basis.T)
    actual = models.jax.jit(density)(models.jnp.asarray(host))
    assert float(actual) == pytest.approx(expected, abs=2e-10)
