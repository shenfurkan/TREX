"""Full continuous bound-companion population callbacks against retained scalar priors."""

import os

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence.nested_evidence import _TransitingPlanetIntegrand, _EclipsingPairIntegrand, bound_companion_population
from tests.fixtures.synthetic_observations import (
    synthetic_trex_tp_integrand_case, synthetic_trex_eb_integrand_case,
)


@pytest.mark.parametrize("kind", ["tp", "eb"])
@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("measured_curve", [False, True])
@pytest.mark.parametrize("primary_mass", [1.0, 0.3, 1.8])
@pytest.mark.parametrize("execution_device", ["cpu", pytest.param("gpu:0", marks=pytest.mark.skipif(
    os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="physical GPU acceptance requires an approved GPU host")),
    pytest.param("gpu:1", marks=pytest.mark.skipif(os.environ.get("EXONYM_REQUIRE_GPU") != "1",
                                                reason="physical GPU acceptance requires an approved GPU host"))])
def test_complete_bound_population_integrand_matches_scalar(kind, hosted, measured_curve, primary_mass, execution_device):
    # TEST_DESIGN: target-neutral primary and imaging geometry. The expected
    # population period, q, spline, atmosphere and retention values are read
    # through the actual retained scalar production integrand, not synthesized.
    population = bound_companion_population(
        primary_mass, primary_mass, 50.0, resolution_limit_arcsec=1.0,
        contrast_separations=np.asarray([0.1, 0.5, 1.0]) if measured_curve else None,
        contrast_values=np.asarray([1.0, 3.0, 5.0]) if measured_curve else None,
    )
    scene = synthetic_trex_tp_integrand_case() if kind == "tp" else synthetic_trex_eb_integrand_case(2.0)
    constructor = _TransitingPlanetIntegrand if kind == "tp" else _EclipsingPairIntegrand
    kwargs = dict(scene, host_mass=primary_mass, host_radius=primary_mass, nsamples=3,
                  bin_integration_days=scene["exptime_days"], population=population,
                  n_dimensions=4 + population.n_dimensions, hosted=hosted, target_feh=0.0)
    if kind == "tp":
        kwargs["use_flat_priors"] = False
    scalar = constructor(**kwargs)
    platform, _, index = execution_device.partition(":")
    device = models.jax.devices(platform)[int(index or 0)]
    native = scalar.jax_unit_integrand(device=device)
    # D-61: every bound primary adds the sky projection (4 orbit + 3 population coordinates).
    points = np.random.default_rng(20261017).uniform(size=(64, 4 + population.n_dimensions))
    points[:2, 4] = (0.0, np.nextafter(1.0, 0.0))
    parameters = np.asarray([scalar.prior_transform(point) for point in points])
    placed = models.jax.device_put(points, device)
    transformed = models.jax.jit(models.jax.vmap(native.prior_transform))(placed)
    expected = np.asarray([scalar.log_likelihood(value) for value in parameters])
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(placed)
    assert evaluated.log_integrand.devices() == {device}
    assert evaluated.log_integrand.dtype == np.float64
    assert np.asarray(evaluated.numerically_valid).all()
    np.testing.assert_array_equal(np.isneginf(evaluated.log_integrand), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any()
    np.testing.assert_allclose(np.asarray(evaluated.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(transformed, parameters, rtol=1e-12, atol=1e-14)
    assert native.claim_eligible is False
    program = str(models.jax.make_jaxpr(models.jax.vmap(native.evaluate))(points))
    assert "pure_callback" not in program and "io_callback" not in program


def test_bound_population_supported_numerical_failure_is_not_zero_likelihood():
    population = bound_companion_population(1.0, 1.0, 50.0, resolution_limit_arcsec=1.0)
    scene = synthetic_trex_tp_integrand_case()
    scene["sigma"] = np.finfo(np.float64).tiny
    scalar = _TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        nsamples=3, bin_integration_days=0.0, population=population, n_dimensions=4 + population.n_dimensions,
        hosted=True, target_feh=0.0, use_flat_priors=False)
    native = scalar.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    points = np.random.default_rng(20261018).uniform(size=(128, scalar.n_dimensions))
    result = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    numerical_failures = ~np.asarray(result.numerically_valid)
    assert numerical_failures.any(), "supported draws must exercise the numerical overflow boundary"
    assert np.isnan(np.asarray(result.log_integrand)[numerical_failures]).all()
    nonfinite = points[:2].copy()
    nonfinite[0, 4], nonfinite[1, 5] = np.nan, np.inf
    invalid = models.jax.jit(models.jax.vmap(native.evaluate))(nonfinite)
    assert not np.asarray(invalid.numerically_valid).any()
    assert np.isnan(invalid.log_integrand).all()
