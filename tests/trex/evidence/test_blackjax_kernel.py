"""Real TP integrand through the proposed guarded native nested kernel.

This tests constrained state transitions and invalidity propagation, not
evidence calibration. No lnZ is emitted or claimed by this kernel interface.
"""

import numpy as np
import pytest

from trex.evidence import nested_evidence
from trex.accelerators import _jax_models as models
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _kernel():
    factory = getattr(nested_evidence, "native_nested_kernel", None)
    assert callable(factory), "TREX has no guarded device-native constrained sampling kernel"
    scene = synthetic_trex_tp_integrand_case()
    scalar = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        bin_integration_days=0.0, nsamples=3, use_flat_priors=False, population=None, n_dimensions=4)
    native = scalar.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    return factory(native, n_dimensions=4, n_live_points=32, num_delete=4, num_inner_steps=8), scalar


def test_native_kernel_uses_real_tp_integrand_and_retains_dead_and_live_states():
    kernel, scalar = _kernel()
    jax = models.jax
    points = np.random.default_rng(20261019).uniform(size=(32, 4))
    state = kernel.initialize(points, jax.random.PRNGKey(20261019))
    expected = np.asarray([scalar.log_likelihood(scalar.prior_transform(point)) for point in points])
    np.testing.assert_array_equal(np.isneginf(state.particles.loglikelihood), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(state.particles.loglikelihood)[finite], expected[finite], rtol=1e-4, atol=1e-10)
    for index in range(3):
        previous = np.asarray(state.particles.loglikelihood)
        state, info = kernel.step(jax.random.fold_in(jax.random.PRNGKey(20261020), index), state)
        assert state.particles.position.shape == (32, 4)
        assert info.particles.position.shape == (4, 4)
        np.testing.assert_allclose(np.sort(info.particles.loglikelihood), np.sort(previous)[:4], rtol=0.0, atol=0.0)
        assert np.min(state.particles.loglikelihood) >= np.max(info.particles.loglikelihood)
        assert np.asarray(info.update_info.is_accepted).all()
    assert kernel.claim_eligible is False
    assert kernel.evidence_calibrated is False


def test_native_kernel_initialization_exposes_invalid_scientific_draws():
    kernel, _scalar = _kernel()
    points = np.random.default_rng(20261019).uniform(size=(32, 4))
    points[0, 1] = np.nan
    with pytest.raises(Exception, match="nonfinite|numerical|invalid"):
        kernel.initialize(points, models.jax.random.PRNGKey(20261019))


def test_native_kernel_replays_identical_state_and_rng_without_host_callbacks():
    kernel, _scalar = _kernel()
    jax = models.jax
    points = np.random.default_rng(20261019).uniform(size=(32, 4))
    initial = kernel.initialize(points, jax.random.PRNGKey(20261019))
    key = jax.random.PRNGKey(20261020)
    first, info_first = kernel.step(key, initial)
    second, info_second = kernel.step(key, initial)
    for left, right in zip(jax.tree.leaves((first, info_first)), jax.tree.leaves((second, info_second))):
        np.testing.assert_array_equal(left, right)
    program = str(jax.make_jaxpr(kernel.checked_step)(key, initial))
    assert "pure_callback" not in program and "io_callback" not in program


def test_native_kernel_rejects_missing_strict_survivors_before_sampling():
    kernel, _scalar = _kernel()
    points = np.random.default_rng(20261019).uniform(size=(32, 4))
    initial = kernel.initialize(points, models.jax.random.PRNGKey(20261019))
    # TEST_DESIGN: a likelihood plateau explicitly exercises the upstream
    # no-strict-survivor fallback, not a fabricated physical reference.
    state = initial._replace(particles=initial.particles._replace(
        loglikelihood=models.jnp.zeros_like(initial.particles.loglikelihood)))
    with pytest.raises(ValueError, match="no strict live survivor"):
        kernel.step(models.jax.random.PRNGKey(20261020), state)


def test_native_kernel_rejects_singular_live_covariance_without_regularization():
    kernel, _scalar = _kernel()
    points = np.random.default_rng(20261019).uniform(size=(32, 4))
    initial = kernel.initialize(points, models.jax.random.PRNGKey(20261019))
    state = initial._replace(particles=initial.particles._replace(
        position=models.jnp.broadcast_to(initial.particles.position[0], points.shape)))
    with pytest.raises(ValueError, match="invalid numerical live covariance"):
        kernel.step(models.jax.random.PRNGKey(20261020), state)
