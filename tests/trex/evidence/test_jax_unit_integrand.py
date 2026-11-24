"""Full fixed-host TP unit-cube path: SciPy/Batman reference versus native JAX.

This is numerical/software parity, not evidence calibration. Population-hosted
TP/EB, O9, K7 and production-budget evidence remain separate acceptance gates.
"""

import numpy as np
import pytest
import os

from trex.evidence.nested_evidence import _TransitingPlanetIntegrand
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _integrands(mass=1.0, radius=1.0, flat=False, boxed=False, period_days=None):
    from trex.accelerators import _jax_models as models

    scene = synthetic_trex_tp_integrand_case()
    if period_days is not None:
        scene["period_days"] = period_days
    reference = _TransitingPlanetIntegrand(
        **scene, host_mass=mass, host_radius=radius,
        bin_integration_days=scene["exptime_days"] if boxed else 0.0,
        nsamples=3, use_flat_priors=flat, population=None, n_dimensions=4,
    )
    factory = getattr(reference, "jax_unit_integrand", None)
    assert callable(factory), "the production TP integrand has no pure-JAX unit-cube callback"
    native = factory(device=models.jax.devices("cpu")[0])
    return reference, native, models.jax


@pytest.mark.parametrize("mass,radius", [(0.3, 0.3), (0.45, 0.45), (1.0, 1.0)])
@pytest.mark.parametrize("flat,boxed", [(False, False), (True, True)])
def test_complete_unit_cube_tp_transform_and_likelihood_match_reference(mass, radius, flat, boxed):
    reference, native, jax = _integrands(mass, radius, flat, boxed)
    # Declared development points, independent of any P5/calibration scene.
    points = np.random.default_rng(20261013).uniform(size=(32, 4))
    points[:2, 1] = (1e-12, 1 - 1e-12)
    transformed = np.asarray(jax.jit(jax.vmap(native.prior_transform))(points))
    expected_parameters = np.asarray([reference.prior_transform(point) for point in points])
    # Machine-precision transform comparison, including the inverse-beta tail.
    # This is stricter than flux-model parity, not a replacement for O9 exactness.
    np.testing.assert_allclose(transformed, expected_parameters, rtol=1e-12, atol=1e-14)
    evaluated = jax.jit(jax.vmap(native.evaluate))(points)
    assert np.all(evaluated.numerically_valid), "a finite reference draw became a numerical failure"
    actual = np.asarray(evaluated.log_integrand)
    expected = np.asarray([reference.log_likelihood(value) for value in expected_parameters])
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    # Unchanged TREX JAX/Batman likelihood-parity tolerance. Evidence adoption
    # additionally needs the existing 0.1-nat/error-coverage and O9 contracts.
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)
    assert native.claim_eligible is False
    program = str(jax.make_jaxpr(jax.vmap(native.evaluate))(points))
    assert "pure_callback" not in program and "io_callback" not in program


def test_native_integrand_distinguishes_structural_zero_from_numerical_failure():
    reference, native, jax = _integrands()
    points = np.asarray([
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 1 - 1e-12, 0.5, 0.5],  # physically colliding orbit, not a clipped e
        [-0.1, 0.5, 0.5, 0.5],       # outside the unit-cube prior
        [0.5, np.nan, 0.5, 0.5],      # input/numerical defect must stay observable
    ])
    result = jax.jit(jax.vmap(native.evaluate))(points)
    assert np.isfinite(result.log_integrand[0])
    assert np.isneginf(result.log_integrand[1:3]).all()
    assert np.asarray(result.numerically_valid).tolist() == [True, True, True, False]
    assert np.isnan(result.log_integrand[3]), "NaN must not masquerade as physical zero support"
    assert np.isneginf(reference.log_likelihood(reference.prior_transform(points[1])))


def test_population_callback_is_explicitly_unimplemented_not_silent_cpu_fallback():
    reference, _native, jax = _integrands()
    reference.hosted = True
    with pytest.raises(NotImplementedError, match="population"):
        reference.jax_unit_integrand(device=jax.devices("cpu")[0])


def test_complete_prior_transform_preserves_both_inverse_beta_tails():
    reference, native, jax = _integrands()
    quantiles = np.asarray([0.0, 1e-250, 1e-200, 1e-100, 1e-30, 1e-16,
                            1e-12, 0.5, 1 - 1e-12, np.nextafter(1.0, 0.0)])
    points = np.full((quantiles.size, 4), 0.5)
    points[:, 1] = quantiles
    actual = np.asarray(jax.jit(jax.vmap(native.prior_transform))(points))[:, 1]
    expected = np.asarray([reference.prior_transform(point)[1] for point in points])
    # Relative-only comparison prevents an absolute tolerance from hiding a
    # lost small quantile or an inverse-CDF interpolation/underflow floor.
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert actual[0] == 0.0
    assert np.all(np.diff(actual) > 0.0)


def test_complete_native_tp_path_has_no_high_eccentricity_routing_cutoff():
    from fractions import Fraction
    from scipy.stats import beta
    from trex.populations.priors import KIPPING_2013_ECCENTRICITY_BETA_A, KIPPING_2013_ECCENTRICITY_BETA_B

    # TEST_DESIGN: a long period permits detached orbits even at e=999/1000.
    # This is a numerical development case, not a survey/population assumption.
    reference, native, jax = _integrands(period_days=float(Fraction(10000, 1)))
    eccentricities = [float(Fraction(99, 100)), float(Fraction(199, 200)), float(Fraction(999, 1000))]
    quantiles = beta.cdf(eccentricities, KIPPING_2013_ECCENTRICITY_BETA_A,
                         KIPPING_2013_ECCENTRICITY_BETA_B)
    points = np.asarray([[0.5, quantile, omega, 0.5]
                         for quantile in quantiles for omega in (0.0, 0.25, 0.75)])
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    assert np.isfinite(expected).all(), "the detached high-e reference must have physical support"
    evaluated = jax.jit(jax.vmap(native.evaluate))(points)
    assert np.asarray(evaluated.numerically_valid).all()
    np.testing.assert_allclose(evaluated.log_integrand, expected, rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("field,value", [("sigma", 0.0), ("sigma", np.nan),
                                        ("bin_integration_days", -1.0), ("nsamples", True)])
def test_native_tp_factory_refuses_invalid_scientific_inputs(field, value):
    reference, _native, jax = _integrands()
    setattr(reference, field, value)
    with pytest.raises(ValueError, match=field if field != "sigma" else "photometry"):
        reference.jax_unit_integrand(device=jax.devices("cpu")[0])


def test_supported_numerical_overflow_is_not_silently_counted_as_zero_support():
    reference, _native, jax = _integrands()
    reference.sigma = np.finfo(np.float64).tiny
    native = reference.jax_unit_integrand(device=jax.devices("cpu")[0])
    evaluated = jax.jit(native.evaluate)(np.full(4, 0.5))
    assert not evaluated.numerically_valid
    assert np.isnan(evaluated.log_integrand)


@pytest.mark.skipif(os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="physical dual-GPU acceptance requires an approved GPU session")
def test_complete_tp_callback_runs_on_each_physical_gpu_with_float64_parity():
    reference, _cpu_native, jax = _integrands()
    devices = jax.devices("gpu")
    assert len(devices) >= 2, "physical dual-GPU execution is required"
    points = np.random.default_rng(20261013).uniform(size=(32, 4))
    points[:2, 1] = (1e-12, 1 - 1e-12)
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    for device in devices:
        native = reference.jax_unit_integrand(device=device)
        placed = jax.device_put(points, device)
        transformed = jax.jit(jax.vmap(native.prior_transform))(placed)
        np.testing.assert_allclose(transformed, parameters, rtol=1e-12, atol=1e-14)
        evaluated = jax.jit(jax.vmap(native.evaluate))(placed)
        assert evaluated.log_integrand.dtype == np.dtype("float64")
        assert evaluated.log_integrand.devices() == {device}
        assert np.asarray(evaluated.numerically_valid).all()
        np.testing.assert_array_equal(np.isneginf(evaluated.log_integrand), np.isneginf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(np.asarray(evaluated.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)
