"""Experimental integration-boundary controls, not TREX error calibration.

Development domain: IID unit-cube priors; constant, exponential and bounded
support likelihoods in one dimension. Seeds 17/29/43, deletion sizes 1/4.
The original 128-live/100,000-call known-Z controls and their failed receipts
are retained. Following analytic error decomposition and an operator-approved
precision revision (2026-10-14), the known-Z accuracy test uses 1024 live points
and at most 2,000,000 calls; the seeds, eight steps and 0.1-nat bar are unchanged.
Other controls retain their original budgets. This is development, not held-out
coverage: more-live assessment on fresh seeds is retained separately.
"""

import math

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_kernel import NativeNestedKernel
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


class AnalyticIntegrand:
    """Test-only dimensionless likelihoods with independent analytic integrals."""

    def __init__(self, kind, device, *, log_offset=0.0):
        self.kind = kind
        self.device = device
        self.log_offset = log_offset

    def evaluate(self, point):
        jnp = models.jnp
        inside = jnp.all(jnp.isfinite(point) & (point >= 0.0) & (point < 1.0))
        value = -4.0 * point[0]
        valid = jnp.all(jnp.isfinite(point))
        if self.kind == "constant":
            value = jnp.asarray(math.log(2.0), dtype=jnp.float64)
        elif self.kind == "support":
            value = jnp.where(point[0] >= 0.5, 0.0, -jnp.inf)
        elif self.kind == "two-plateaus":
            value = jnp.where(point[0] >= 0.5, math.log(2.0), math.log(0.2))
        elif self.kind == "interior-plateau":
            value = jnp.where(point[0] < 0.25, point[0] - 1.0,
                              jnp.where(point[0] < 0.75, 0.0, point[0] - 0.75))
        elif self.kind == "terminal-plateau":
            value = jnp.where(point[0] < 0.25, point[0] - 1.0, 0.0)
        elif self.kind == "zero":
            value = jnp.asarray(-jnp.inf, dtype=jnp.float64)
        elif self.kind == "invalid":
            valid = jnp.asarray(False)
        elif self.kind == "nan":
            value = jnp.asarray(jnp.nan, dtype=jnp.float64)
        elif self.kind == "positive-infinity":
            value = jnp.asarray(jnp.inf, dtype=jnp.float64)
        elif self.kind == "extreme-range":
            value = jnp.where(point[0] < 0.5, -1e308, 1e308)
        elif self.kind == "invalid-proposal":
            # TEST_DESIGN: fail numerically on rejected outside-cube proposals,
            # after a fully valid initialization. No scientific engine is mocked.
            valid = valid & inside
        return UnitIntegrandEvaluation(jnp.where(inside, value + self.log_offset, -jnp.inf), valid)


@pytest.fixture(params=["cpu", "gpu", "gpu:1"])
def device(request):
    """Exercise both physical GPUs, without substituting unavailable devices."""
    platform, _, index = request.param.partition(":")
    index = int(index or "0")
    try:
        devices = models.jax.devices(platform)
    except RuntimeError:
        devices = []
    if len(devices) <= index:
        pytest.skip(f"no physical {request.param} JAX device on this host")
    with models.jax.default_device(devices[index]):
        yield devices[index]


def _case(kind, device, *, deletion=4, seed=17, live=128):
    kernel = NativeNestedKernel(AnalyticIntegrand(kind, device), n_dimensions=1,
                               n_live_points=live, num_delete=deletion, num_inner_steps=8)
    return kernel, np.random.default_rng(seed).uniform(size=(live, 1)), models.jax.random.PRNGKey(seed)


def _integrate(kernel, positions, key, *, budget=100_000, seed=17):
    method = getattr(kernel, "integrate", None)
    assert callable(method), "native kernel has no complete experimental evidence driver"
    return method(positions, key, evidence_tolerance=0.1,
                  max_likelihood_calls=budget, random_seed=seed)


def _accounting(result):
    volume_weights = np.concatenate((result.dead_log_volume_weights, result.final_live_log_volume_weights))
    np.testing.assert_allclose(np.exp(volume_weights).sum(), 1.0, rtol=0.0, atol=2e-14)
    weights = np.concatenate((result.dead_log_evidence_weights, result.final_live_log_evidence_weights))
    np.testing.assert_allclose(np.logaddexp.reduce(weights), result.log_evidence, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(result.posterior_weights, np.exp(weights - result.log_evidence),
                               rtol=2e-13, atol=0.0)
    assert np.isfinite(result.posterior_weights).all()
    assert (result.posterior_weights >= 0.0).all()
    assert result.posterior_weights.sum() == pytest.approx(1.0, abs=2e-14)
    assert np.isfinite(result.posterior_unit_samples).all()
    assert ((result.posterior_unit_samples >= 0.0) & (result.posterior_unit_samples < 1.0)).all()
    assert result.n_likelihood_calls == result.monitored_likelihood_calls
    assert result.physical_likelihood_calls == result.n_likelihood_calls
    assert result.likelihood_calls_exact
    assert sum(result.likelihood_call_breakdown.values()) == result.n_likelihood_calls
    assert result.claim_eligible is False and result.evidence_calibrated is False


@pytest.mark.parametrize("deletion", [1, 4])
def test_constant_final_live_is_exact_and_has_no_slice_calls(device, deletion):
    kernel, positions, key = _case("constant", device, deletion=deletion)
    result = _integrate(kernel, positions, key)
    _accounting(result)
    assert result.log_evidence == pytest.approx(math.log(2.0), abs=2e-14)
    assert result.log_evidence_error == 0.0
    assert result.n_likelihood_calls == 128
    assert result.n_iterations == 0
    assert result.stop_reason == "likelihood-plateau"
    assert not result.stopped_on_evidence_tolerance
    assert math.isinf(result.base_run_delta_logz)
    assert result.dead_log_volume_weights.size == 0
    np.testing.assert_array_equal(result.posterior_unit_samples, positions)


@pytest.mark.parametrize("deletion", [1, 4])
@pytest.mark.parametrize("seed", [17, 29, 43])
def test_exponential_known_z_joint_deletion_and_prefinal_stop(device, deletion, seed):
    # Precision allowance, not a tolerance relaxation: 128-live compression
    # scatter exceeded the unchanged bar. See test_trex_blackjax_known_z_diagnosis
    # for observed-volume decomposition, tighter-stop and fresh-seed controls.
    live = 1024
    kernel, positions, key = _case("exponential", device, deletion=deletion, seed=seed, live=live)
    result = _integrate(kernel, positions, key, budget=2_000_000, seed=seed)
    _accounting(result)
    assert result.n_live_points == live
    truth = math.log(-math.expm1(-4.0) / 4.0)
    # Development accuracy bar only; not an assertion of error coverage.
    assert abs(result.log_evidence - truth) < 0.1
    assert math.isfinite(result.log_evidence_error) and result.log_evidence_error > 0.0
    assert 1.0 / np.sum(result.posterior_weights ** 2) >= 100.0
    assert result.stop_reason == "evidence-tolerance"
    assert not result.stopped_short and not result.budget_exhausted
    assert result.stopped_on_evidence_tolerance
    assert result.base_run_delta_logz < 0.1
    check = result.stopping_checks[-1]
    assert check.delta_logz == result.base_run_delta_logz
    assert check.n_likelihood_calls == result.n_likelihood_calls
    assert check.evidence_tolerance == 0.1
    independently_remaining = np.logaddexp(0.0, check.log_remaining_volume + check.max_live_loglikelihood
                                           - check.log_dead_evidence)
    assert check.delta_logz == pytest.approx(independently_remaining, abs=2e-14)
    for batch in result.batches:
        if batch.completed:
            expected_counts = live - np.arange(batch.dead_indices.size)
            np.testing.assert_array_equal(batch.effective_live_counts, expected_counts)
    np.testing.assert_allclose(result.dead_log_volumes, -np.cumsum(1.0 / result.effective_live_counts),
                               rtol=0.0, atol=2e-13)


def test_physical_zero_plateau_is_evicted_completely_before_replenishment(device):
    kernel, positions, key = _case("support", device)
    result = _integrate(kernel, positions, key)
    _accounting(result)
    assert abs(result.log_evidence - math.log(0.5)) < 0.1
    unsupported = result.posterior_unit_samples[:, 0] < 0.5
    assert unsupported.any()
    assert np.all(result.posterior_weights[unsupported] == 0.0)
    first = result.batches[0]
    assert first.dead_indices.size == np.count_nonzero(positions[:, 0] < 0.5)
    assert first.dead_indices.size > 4
    assert first.completed and np.isneginf(first.threshold)
    assert result.stop_reason == "likelihood-plateau"


def test_finite_lower_plateau_is_evicted_and_final_upper_plateau_is_retained(device):
    kernel, positions, key = _case("two-plateaus", device)
    result = _integrate(kernel, positions, key)
    _accounting(result)
    assert abs(result.log_evidence - math.log((0.2 + 2.0) / 2.0)) < 0.1
    assert result.batches[0].dead_indices.size == np.count_nonzero(positions[:, 0] < 0.5)
    assert result.batches[0].threshold == pytest.approx(math.log(0.2), abs=0.0)
    assert result.stop_reason == "likelihood-plateau"
    assert np.all(np.isfinite(result.dead_log_evidence_weights))


def test_numerically_invalid_rejected_proposal_is_not_silently_support_zero(device):
    kernel, positions, key = _case("invalid-proposal", device)
    with pytest.raises(RuntimeError, match="numerical") as caught:
        _integrate(kernel, positions, key)
    result = caught.value.result
    assert result.stop_reason == "unresolved" and result.posterior_weights is None
    assert result.n_likelihood_calls == 128 + sum(result.batches[-1].slices[-1].info.likelihood_calls)
    assert not result.batches[-1].completed
    assert not bool(result.batches[-1].slices[-1].info.numerically_valid)
    np.testing.assert_array_equal(result.final_state.particles.position, positions)


def test_singular_covariance_does_not_become_plateau_convergence(device):
    kernel, positions, key = _case("constant", device)
    positions[:] = positions[0]
    with pytest.raises(RuntimeError, match="covariance") as caught:
        _integrate(kernel, positions, key)
    assert caught.value.result.stop_reason == "unresolved"
    assert not caught.value.result.stopped_on_evidence_tolerance


@pytest.mark.parametrize("kind", ["zero", "invalid", "nan", "positive-infinity"])
def test_zero_or_invalid_evidence_fails_with_partial_audit_not_invented_weights(device, kind):
    kernel, positions, key = _case(kind, device)
    with pytest.raises(RuntimeError, match="zero|numerical") as caught:
        _integrate(kernel, positions, key)
    partial = caught.value.result
    assert partial.stop_reason == "unresolved"
    assert not partial.stopped_on_evidence_tolerance
    assert partial.posterior_weights is None
    assert partial.failure_reason
    assert 0 < partial.n_likelihood_calls <= 128
    if kind == "zero":
        assert np.isneginf(partial.log_evidence)
        assert partial.n_likelihood_calls == 128


def test_all_zero_at_exact_initialization_ceiling_is_unresolved_and_budget_bound(device):
    kernel, positions, key = _case("zero", device)
    with pytest.raises(RuntimeError, match="zero") as caught:
        _integrate(kernel, positions, key, budget=128)
    result = caught.value.result
    assert result.n_likelihood_calls == 128 and result.budget_exhausted
    assert result.stop_reason == "unresolved" and result.posterior_weights is None
    assert not result.stopped_on_evidence_tolerance


@pytest.mark.parametrize("budget", [1, 127, 128, 129, 130, 131, 160, 500])
def test_hard_budget_retains_exact_counts_and_incomplete_attempt(device, budget):
    kernel, positions, key = _case("exponential", device)
    result = _integrate(kernel, positions, key, budget=budget)
    assert result.n_likelihood_calls == budget
    assert result.physical_likelihood_calls == budget
    assert result.budget_exhausted and result.stopped_short
    assert result.stop_reason == "truncated"
    assert not result.stopped_on_evidence_tolerance
    assert result.failure_reason
    assert sum(result.likelihood_call_breakdown.values()) == budget
    if budget < 128:
        assert result.final_state is None and result.posterior_weights is None
    else:
        _accounting(result)
        if budget > 128:
            assert result.batches
            # Hitting the cap at a complete batch boundary is also a legitimate
            # truncation. Only the retained completed transitions receive weights.
            assert result.batches[-1].calls_after == budget
            if budget < 128 + 3 * 8 * 4:
                assert not result.batches[-1].completed
                last = result.batches[-1].slices[-1].info
                assert int(last.blocked_phase) in {0, 1, 2}
                assert int(last.remaining_budget) == 0


def test_identical_key_replay_and_functional_budget_kernel(device):
    kernel, positions, key = _case("exponential", device)
    first = _integrate(kernel, positions, key, budget=500)
    second = _integrate(kernel, positions, key, budget=500)
    for name in ("posterior_unit_samples", "posterior_weights", "continuation_key", "dead_log_volumes",
                 "compression_log_evidences"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.log_evidence == second.log_evidence
    assert first.likelihood_call_breakdown == second.likelihood_call_breakdown
    assert first.n_likelihood_calls == second.n_likelihood_calls
    state = first.initial_state.particles
    particle = models.jax.tree.map(lambda array: array[0], state)
    program = str(models.jax.make_jaxpr(kernel.budgeted_slice)(
        key, particle, models.jnp.asarray(-10.0), models.jnp.asarray(100, dtype=models.jnp.int64)))
    assert "pure_callback" not in program and "io_callback" not in program


def test_default_scalar_driver_keeps_four_argument_dispatch(device, monkeypatch):
    kernel, positions, key = _case("exponential", device)
    original = kernel.budgeted_slice
    calls = []

    def slice_step(key, state, threshold, remaining):
        calls.append(1)
        return original(key, state, threshold, remaining)

    monkeypatch.setattr(kernel, "budgeted_slice", slice_step)
    result = _integrate(kernel, positions, key, budget=160)
    assert calls and result.n_likelihood_calls == 160


def test_budgeted_slices_match_pinned_scalar_blackjax_and_its_actual_call_diagnostics(device):
    from blackjax.mcmc.slice import build_kernel, stepping_out
    from blackjax.ns import nss

    kernel, positions, key = _case("exponential", device)
    result = _integrate(kernel, positions, key, budget=500)

    def particle(position, loglikelihood_birth):
        return kernel.integration_evaluate(position, loglikelihood_birth)[0]

    reference = models.jax.jit(nss.slice_constrained_step(
        particle, build_kernel(interval=stepping_out, max_expansions=10, max_shrinkage=100),
        nss.covariance_proposal))
    checked = 0
    for batch in result.batches:
        for attempt in batch.slices:
            if bool(attempt.info.budget_exhausted):
                continue  # An interrupted slice is intentionally not a full transition.
            expected, info = reference(attempt.key, attempt.start, models.jnp.asarray(batch.threshold),
                                       covariance_factor=models.jnp.eye(1, dtype=models.jnp.float64))
            for actual_leaf, expected_leaf in zip(models.jax.tree.leaves(attempt.proposed),
                                                  models.jax.tree.leaves(expected)):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
            for actual_leaf, expected_leaf in zip(models.jax.tree.leaves(attempt.info.slice_info),
                                                  models.jax.tree.leaves(info)):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
            # Verified scalar source has two initial bracket evaluations, one
            # per expansion and one per shrink, with no cached re-evaluations.
            assert sum(attempt.info.likelihood_calls) == 2 + info.num_expansions + info.num_shrink
            checked += 1
    assert checked > 0


def test_batched_four_dimensional_directions_and_replacements_match_scalar(device):
    # Additional development control for the direction-normalization boundary;
    # the identical-key oracle is the unchanged scalar driver, not calibration.
    positions = np.random.default_rng(29).uniform(size=(32, 4))
    key = models.jax.random.PRNGKey(29)
    kernels = [nested_evidence.native_nested_kernel(AnalyticIntegrand("exponential", device),
        n_dimensions=4, n_live_points=32, num_delete=4, num_inner_steps=2, batched=flag)
        for flag in (False, True)]
    expected, actual = [_integrate(kernel, positions, key, budget=5000, seed=29) for kernel in kernels]
    assert any(batch.batched_slices is not None for batch in actual.batches)
    assert actual.log_evidence == expected.log_evidence
    assert actual.log_evidence_error == expected.log_evidence_error
    assert actual.n_likelihood_calls == expected.n_likelihood_calls
    assert actual.likelihood_call_breakdown == expected.likelihood_call_breakdown
    for name in ("posterior_unit_samples", "posterior_weights", "dead_log_volumes", "compression_log_evidences"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for left, right in zip(models.jax.tree.leaves(actual.final_state), models.jax.tree.leaves(expected.final_state)):
        np.testing.assert_array_equal(left, right)
    assert actual.claim_eligible is actual.evidence_calibrated is False


def test_real_tp_integrand_has_auditable_budget_truncation(device):
    scene = synthetic_trex_tp_integrand_case()
    scalar = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        bin_integration_days=0.0, nsamples=3, use_flat_priors=False, population=None, n_dimensions=4)
    native = scalar.jax_unit_integrand(device=device)
    kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=32,
                               num_delete=4, num_inner_steps=8)
    positions = np.random.default_rng(20261019).uniform(size=(32, 4))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(20261019), budget=200)
    assert result.n_likelihood_calls <= 200
    assert result.stop_reason in {"truncated", "evidence-tolerance", "likelihood-plateau"}
    _accounting(result)
    expected = np.asarray([scalar.log_likelihood(scalar.prior_transform(point)) for point in positions])
    actual = np.asarray(result.initial_state.particles.loglikelihood)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    # Same unchanged scientific tolerances as the existing real TP transition test.
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("field,value", [("evidence_tolerance", 0.0), ("evidence_tolerance", math.nan),
    ("max_likelihood_calls", True), ("max_likelihood_calls", 0), ("random_seed", True)])
def test_integrate_rejects_invalid_explicit_controls_before_evaluation(device, field, value):
    kernel, positions, key = _case("constant", device)
    controls = dict(evidence_tolerance=0.1, max_likelihood_calls=1000, random_seed=17)
    controls[field] = value
    method = getattr(kernel, "integrate", None)
    assert callable(method), "native kernel has no complete experimental evidence driver"
    with pytest.raises(ValueError, match=field):
        method(positions, key, **controls)


@pytest.mark.parametrize("offset", [-1e16, 1e16])
def test_large_constant_offset_preserves_evidence_and_relative_weight_audit(device, offset):
    # Independent limiting case: a constant native log integrand integrates to
    # that constant exactly. This targets loss of log-volume terms, not coverage.
    kernel = NativeNestedKernel(AnalyticIntegrand("constant", device, log_offset=offset),
        n_dimensions=1, n_live_points=128, num_delete=4, num_inner_steps=8)
    positions = np.random.default_rng(17).uniform(size=(128, 1))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(17))
    assert result.log_evidence == float(offset + math.log(2.0))
    assert result.log_evidence_relative == pytest.approx(0.0, abs=2e-14)
    assert result.log_evidence_error == 0.0
    relative = np.concatenate((result.dead_log_evidence_weights_relative,
                               result.final_live_log_evidence_weights_relative))
    np.testing.assert_allclose(np.logaddexp.reduce(relative), result.log_evidence_relative,
                               rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(result.posterior_weights, np.exp(relative - result.log_evidence_relative),
                               rtol=2e-13, atol=0.0)
    assert result.posterior_weights.sum() == pytest.approx(1.0, abs=2e-14)
    centered = result._replace(log_evidence=0.0)
    assert centered.log_evidence_offset == 0.0
    np.testing.assert_array_equal(centered.final_live_log_evidence_weights,
                                  result.final_live_log_evidence_weights_relative)


def test_occurrence_weight_wrapper_accepts_native_result_and_preserves_its_audit(device):
    from types import SimpleNamespace

    kernel, positions, key = _case("constant", device)
    result = _integrate(kernel, positions, key)
    shifted = nested_evidence._with_occurrence_weight(result, SimpleNamespace(log_occurrence_weight=-2.0))
    assert shifted.log_evidence == result.log_evidence - 2.0
    assert shifted.log_evidence_error == result.log_evidence_error
    np.testing.assert_array_equal(shifted.posterior_weights, result.posterior_weights)
    np.testing.assert_allclose(shifted.final_live_log_evidence_weights,
                               result.final_live_log_evidence_weights - 2.0, rtol=0.0, atol=2e-14)
    np.testing.assert_array_equal(shifted.final_live_log_evidence_weights_relative,
                                  result.final_live_log_evidence_weights_relative)
    assert shifted.external_log_evidence_shift == pytest.approx(-2.0, abs=2e-14)


def test_nonconstant_large_offset_keeps_compression_dispersion_before_restoring_scale(device):
    # Additional development control, declared before execution. The native
    # float64 likelihood's 1e12 offset has sub-millinat quantization; the known-Z
    # criterion remains the unchanged 0.1-nat development bar, not error coverage.
    offset = 1e12
    kernel = NativeNestedKernel(AnalyticIntegrand("exponential", device, log_offset=offset),
        n_dimensions=1, n_live_points=128, num_delete=4, num_inner_steps=8)
    positions = np.random.default_rng(17).uniform(size=(128, 1))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(17))
    truth = offset + math.log(-math.expm1(-4.0) / 4.0)
    assert abs(result.log_evidence - truth) < 0.1
    assert result.log_evidence_error > 0.0
    samples = result.compression_log_evidences_relative
    np.testing.assert_allclose(result.log_evidence_error, np.std(samples - samples[0], ddof=1),
                               rtol=2e-13, atol=0.0)
    relative = np.concatenate((result.dead_log_evidence_weights_relative,
                               result.final_live_log_evidence_weights_relative))
    np.testing.assert_allclose(result.posterior_weights, np.exp(relative - result.log_evidence_relative),
                               rtol=2e-13, atol=0.0)
    assert result.posterior_weights.sum() == pytest.approx(1.0, abs=2e-14)


@pytest.mark.parametrize("key_kind", ["typed", "legacy"])
def test_single_key_forms_work_and_batched_keys_are_rejected_before_evaluation(device, key_kind):
    kernel, positions, _ = _case("constant", device)
    key = (models.jax.random.key(17) if key_kind == "typed" else models.jax.random.PRNGKey(17))
    result = _integrate(kernel, positions, key)
    assert result.n_likelihood_calls == 128
    batched = models.jax.random.split(key, 2)
    with pytest.raises(ValueError, match="single.*key"):
        _integrate(kernel, positions, batched)


def test_complex_prior_positions_are_rejected_without_discarding_imaginary_data(device):
    kernel, positions, key = _case("constant", device)
    positions = positions.astype(np.complex128)
    positions[0, 0] = complex(0.5, math.nan)
    with pytest.raises(ValueError, match="real.*positions|positions.*real"):
        _integrate(kernel, positions, key)


def test_finalization_arithmetic_failure_retains_known_counts_without_retrying_science(device):
    kernel, positions, key = _case("extreme-range", device)
    with pytest.raises(RuntimeError, match="dynamic range|arithmetic") as caught:
        _integrate(kernel, positions, key, budget=128)
    result = caught.value.result
    assert result.stop_reason == "unresolved" and result.posterior_weights is None
    assert result.n_likelihood_calls == 128 and result.likelihood_calls_exact
    assert result.failed_operation == "finalization"
    np.testing.assert_array_equal(result.final_state.particles.position, positions)
    assert result.budget_exhausted


def test_malformed_native_contract_retains_failure_attempt_and_does_not_invent_a_counter(device):
    class MalformedValidity(AnalyticIntegrand):
        def evaluate(self, point):
            value = super().evaluate(point)
            return value._replace(numerically_valid=models.jnp.ones(2, dtype=models.jnp.bool_))

    kernel = NativeNestedKernel(MalformedValidity("constant", device), n_dimensions=1,
        n_live_points=128, num_delete=4, num_inner_steps=8)
    positions = np.random.default_rng(17).uniform(size=(128, 1))
    with pytest.raises(RuntimeError, match="scalar boolean") as caught:
        _integrate(kernel, positions, models.jax.random.PRNGKey(17))
    result = caught.value.result
    assert result.stop_reason == "unresolved" and result.posterior_weights is None
    assert result.failed_operation == "initialization"
    assert result.n_likelihood_calls == 0  # lower bound, not guessed execution
    assert not result.likelihood_calls_exact and result.physical_likelihood_calls is None
    np.testing.assert_array_equal(result.failed_attempt["position"], positions[0])


def test_budget_interruptions_name_left_right_and_shrinkage_phases(device):
    kernel, positions, key = _case("exponential", device)
    baseline = _integrate(kernel, positions, key, budget=500)
    first = baseline.batches[0].slices[0]
    assert not bool(first.info.budget_exhausted)
    left, right, shrinkage = map(int, first.info.likelihood_calls)
    assert min(left, right, shrinkage) >= 1
    # Metamorphic budget controls, not a second integrator: the same stream
    # must stop before the next evaluation in the designated scalar phase.
    ceilings = {"right": 128 + left, "shrinkage": 128 + left + right,
                "left": 128 + left + right + shrinkage}
    for name, ceiling in ceilings.items():
        result = _integrate(kernel, positions, key, budget=ceiling)
        assert result.stop_reason == "truncated" and result.n_likelihood_calls == ceiling
        assert result.likelihood_calls_exact
        assert not result.batches[-1].completed
        actual_phase = int(result.batches[-1].slices[-1].info.blocked_phase)
        assert actual_phase == {"left": 0, "right": 1, "shrinkage": 2}[name]
        np.testing.assert_array_equal(result.final_state.particles.position, positions)


@pytest.mark.parametrize("kind", ["interior-plateau", "terminal-plateau"])
def test_mixed_rank_batch_never_splits_a_tied_contour_or_evicts_all_survivors(device, kind):
    # New lifecycle development controls, predeclared before execution. The
    # independent oracle below is the analytic piecewise contour partition,
    # not a test-side nested integrator. Only one native proposal is budgeted.
    kernel, positions, key = _case(kind, device, deletion=64)
    result = _integrate(kernel, positions, key, budget=129)
    _accounting(result)
    batch = result.batches[0]
    lower = positions[:, 0] < 0.25
    plateau = ((positions[:, 0] >= 0.25) & (positions[:, 0] < 0.75)
               if kind == "interior-plateau" else ~lower)
    assert np.count_nonzero(lower) < 64 < np.count_nonzero(lower | plateau)
    expected = np.flatnonzero(lower | plateau if kind == "interior-plateau" else lower)
    np.testing.assert_array_equal(np.sort(batch.dead_indices), expected)
    assert batch.dead_indices.size < 128
    assert not batch.completed and result.stop_reason == "truncated"
    assert result.dead_log_volume_weights.size == 0
    np.testing.assert_array_equal(result.final_state.particles.position, positions)
    for attempt in batch.slices:
        assert float(attempt.start.loglikelihood) > batch.threshold
    from trex.evidence.evidence_escalation import _is_unresolved_nested_run

    assert _is_unresolved_nested_run(result)


def test_fixed_isotropic_slice_keeps_functional_counts_under_exhausted_shrinkage(device):
    from blackjax.ns import base

    class SingletonSupport(AnalyticIntegrand):
        def evaluate(self, point):
            # Explicit test-only support; the deliberately impossible contour
            # below exercises the scalar kernel's failure accounting directly.
            value = models.jnp.where(models.jnp.all(point == 0.5), 0.0, -models.jnp.inf)
            return UnitIntegrandEvaluation(value, models.jnp.asarray(True))

    kernel = NativeNestedKernel(SingletonSupport("zero", device), n_dimensions=1,
        n_live_points=128, num_delete=4, num_inner_steps=8)
    state = base.StateWithLogLikelihood(models.jnp.asarray([0.5]), models.jnp.asarray(0.0),
                                       models.jnp.asarray(0.0), models.jnp.asarray(-models.jnp.inf))
    proposed, info = kernel.budgeted_slice(models.jax.random.PRNGKey(17), state,
        models.jnp.asarray(1.0), models.jnp.asarray(1000, dtype=models.jnp.int64))
    assert not bool(info.slice_info.is_accepted)
    assert bool(info.numerically_valid) and not bool(info.budget_exhausted)
    assert int(info.slice_info.num_shrink) == 100
    assert int(info.likelihood_calls.sum()) == 2 + int(info.slice_info.num_expansions) + 100
    assert int(info.remaining_budget) + int(info.likelihood_calls.sum()) == 1000
    for actual, original in zip(models.jax.tree.leaves(proposed), models.jax.tree.leaves(state)):
        np.testing.assert_array_equal(actual, original)


def test_real_tp_initial_likelihood_float64_gpu_matches_same_native_cpu_at_1e_minus10_gate(device):
    if device.platform != "gpu":
        pytest.skip("physical GPU parity gate; CPU accounting is tested separately")
    scene = synthetic_trex_tp_integrand_case()
    scalar = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        bin_integration_days=0.0, nsamples=3, use_flat_priors=False, population=None, n_dimensions=4)
    positions = np.random.default_rng(20261019).uniform(size=(32, 4))
    key = models.jax.random.PRNGKey(20261019)
    results = []
    for selected in (models.jax.devices("cpu")[0], device):
        native = scalar.jax_unit_integrand(device=selected)
        kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=32,
                                   num_delete=4, num_inner_steps=8)
        results.append(_integrate(kernel, positions, key, budget=32))
    reference, accelerated = results
    expected = np.asarray(reference.initial_state.particles.loglikelihood)
    actual = np.asarray(accelerated.initial_state.particles.loglikelihood)
    assert actual.dtype == expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any()
    # Absolute same-native hardware gate on identical points, not independent
    # sampler agreement and not a new relaxation of the Batman reference test.
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=1e-10)
    assert accelerated.stop_reason == reference.stop_reason == "truncated"


def _routing_controls(budget):
    return nested_evidence._SamplerControls(n_live_points=32, evidence_tolerance=0.05,
                                            max_likelihood_calls=budget, random_seed=20261019, n_jobs=1)


def test_opt_in_routes_real_tp_nested_run_to_blackjax(device, monkeypatch):
    scene = synthetic_trex_tp_integrand_case()
    scalar = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        bin_integration_days=0.0, nsamples=3, use_flat_priors=False, population=None, n_dimensions=4)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    result = nested_evidence._integrate(scalar, _routing_controls(300), "TP", 4, None)
    assert result.sampler == "blackjax-nss-experimental"
    assert result.num_delete == 16 and result.num_inner_steps == 12
    assert result.n_likelihood_calls <= 300
    assert not result.evidence_calibrated


@pytest.mark.parametrize("estimator,population", [("blackjax", None), ("blackjax", "bound"), (None, None)])
def test_explicit_unsupported_native_refuses_while_default_keeps_dynesty(estimator, population, monkeypatch):
    from types import SimpleNamespace

    def unsupported(*, device):
        raise NotImplementedError("population-hosted")

    def native_never_called(*, device):
        raise AssertionError("default routing must not build a native integrand")

    if estimator is None:
        monkeypatch.delenv("TREX_NESTED_ESTIMATOR", raising=False)
    else:
        monkeypatch.setenv("TREX_NESTED_ESTIMATOR", estimator)
    # A population scene without target_feh must not reach the native constructor.
    integrand = SimpleNamespace(baseline_likelihood=None, prior_transform=None, log_likelihood=None, population=population, hosted=False,
                                target_feh=None, jax_unit_integrand=(
                                    unsupported if estimator and population is None else native_never_called))
    monkeypatch.setattr(nested_evidence, "_run_nested_sampler", lambda *args: "dynesty-run")
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        if estimator:
            with pytest.raises((NotImplementedError, ValueError)):
                nested_evidence._integrate(integrand, _routing_controls(300), "TP", 4, None)
        else:
            assert nested_evidence._integrate(integrand, _routing_controls(300), "TP", 4, None) == "dynesty-run"


def _batch_case(kind, device, *, deletion=4, batched=True):
    kernel = nested_evidence.native_nested_kernel(
        AnalyticIntegrand(kind, device), n_dimensions=1, n_live_points=32,
        num_delete=deletion, num_inner_steps=2, batched=batched)
    return kernel, np.random.default_rng(29).uniform(size=(32, 1)), models.jax.random.PRNGKey(29)


def _batch_counts(result):
    """Reconcile retained functional counters, including uncommitted work."""
    from trex.evidence._blackjax_types import SLICE_CALL_BOUND

    phases = np.zeros(3, dtype=np.int64)
    for batch in result.batches:
        if batch.batched_slices is None:
            counts = np.asarray([attempt.info.likelihood_calls for attempt in batch.slices])
            total = counts.sum(axis=0)
        else:
            keys, parents, infos = batch.batched_slices
            q = batch.dead_indices.size
            assert keys.shape[0] == parents.size == q
            counts = np.asarray(infos.likelihood_calls)
            assert counts.shape == (q, result.num_inner_steps, 3)
            used = counts.sum(axis=-1)
            assert (used <= SLICE_CALL_BOUND).all()
            np.testing.assert_array_equal(infos.remaining_budget, SLICE_CALL_BOUND - used)
            total = counts.sum(axis=(0, 1))
            assert not batch.slices
            assert batch.replacement_particles.position.shape == (q, 1)
            if batch.completed:
                assert np.asarray(infos.numerically_valid).all()
                assert np.asarray(infos.slice_info.is_accepted).all()
                assert not np.asarray(infos.budget_exhausted).any()
                assert (np.asarray(batch.replacement_particles.loglikelihood) > batch.threshold).all()
                np.testing.assert_array_equal(batch.replacement_particles.loglikelihood_birth,
                                              np.full(q, batch.threshold))
        assert batch.calls_after - batch.calls_before == int(total.sum())
        phases += total
    assert result.n_likelihood_calls == result.monitored_likelihood_calls
    assert result.n_likelihood_calls <= result.max_likelihood_calls
    assert result.likelihood_calls_exact
    assert result.likelihood_call_breakdown == dict(
        initialization=len(result.initial_particles), left=int(phases[0]),
        right=int(phases[1]), shrinkage=int(phases[2]))
    assert sum(result.likelihood_call_breakdown.values()) == result.n_likelihood_calls
    assert not result.claim_eligible and not result.evidence_calibrated


@pytest.mark.parametrize("kind", ["exponential", "two-plateaus"])
@pytest.mark.parametrize("deletion", [1, 4])
def test_batched_identical_key_matches_scalar_replacements_logz_and_volume_accounting(device, kind, deletion):
    scalar, positions, key = _batch_case(kind, device, deletion=deletion, batched=False)
    batched, _, _ = _batch_case(kind, device, deletion=deletion)
    expected = _integrate(scalar, positions, key, budget=5000, seed=29)
    actual = _integrate(batched, positions, key, budget=5000, seed=29)
    _accounting(expected)
    _batch_counts(actual)
    assert any(batch.batched_slices is not None for batch in actual.batches)
    assert actual.slice_execution == "vmapped-chains"
    assert actual.physical_likelihood_calls is None
    assert actual.log_evidence == expected.log_evidence
    assert actual.log_evidence_error == expected.log_evidence_error
    assert actual.stop_reason == expected.stop_reason
    assert actual.n_likelihood_calls == expected.n_likelihood_calls
    assert actual.likelihood_call_breakdown == expected.likelihood_call_breakdown
    assert len(actual.batches) == len(expected.batches)
    for left, right in zip(actual.batches, expected.batches):
        assert left.completed == right.completed
        assert left.threshold == right.threshold
        np.testing.assert_array_equal(left.dead_indices, right.dead_indices)
        np.testing.assert_array_equal(left.effective_live_counts, right.effective_live_counts)
        if left.batched_slices is not None:
            stack = models.jax.tree.map(lambda *leaves: models.jnp.stack(leaves), *right.replacement_particles)
            for received, reference in zip(models.jax.tree.leaves(left.replacement_particles),
                                            models.jax.tree.leaves(stack)):
                np.testing.assert_array_equal(received, reference)
    for name in ("posterior_unit_samples", "posterior_weights", "dead_log_volumes",
                 "dead_log_volume_weights", "final_live_log_volume_weights",
                 "compression_log_evidences", "continuation_key"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for received, reference in zip(models.jax.tree.leaves(actual.final_state),
                                    models.jax.tree.leaves(expected.final_state)):
        np.testing.assert_array_equal(received, reference)


@pytest.mark.parametrize("budget", [33, 500, 919, 5000])
def test_batched_ceiling_falls_back_to_scalar_without_overshoot(device, budget):
    from trex.evidence._blackjax_types import SLICE_CALL_BOUND

    scalar, positions, key = _batch_case("exponential", device, batched=False)
    batched, _, _ = _batch_case("exponential", device)
    controls = dict(evidence_tolerance=1e-12, max_likelihood_calls=budget, random_seed=29)
    expected = scalar.integrate(positions, key, **controls)
    actual = batched.integrate(positions, key, **controls)
    _batch_counts(actual)
    assert actual.n_likelihood_calls == expected.n_likelihood_calls == budget
    assert actual.stop_reason == expected.stop_reason == "truncated"
    assert actual.budget_exhausted and actual.stopped_short
    assert actual.log_evidence == expected.log_evidence
    assert actual.likelihood_call_breakdown == expected.likelihood_call_breakdown
    assert actual.batches[-1].batched_slices is None
    assert actual.batches[-1].calls_after == budget
    assert actual.batches[-1].slices
    for batch in actual.batches:
        reserve = batch.dead_indices.size * actual.num_inner_steps * SLICE_CALL_BOUND
        assert (batch.batched_slices is not None) == (budget - batch.calls_before >= reserve)
    if budget == 5000:
        assert any(batch.batched_slices is not None for batch in actual.batches)
        assert actual.physical_likelihood_calls is None
    else:
        assert actual.slice_execution == "scalar-device-loop"
        assert actual.physical_likelihood_calls == budget
    np.testing.assert_array_equal(actual.final_state.particles.position, expected.final_state.particles.position)
    np.testing.assert_array_equal(actual.dead_log_volumes, expected.dead_log_volumes)


def test_batched_invalid_numerics_fail_whole_batch_without_committing_deaths(device):
    kernel, positions, key = _batch_case("invalid-proposal", device)
    with pytest.raises(RuntimeError, match="invalid numerical") as caught:
        _integrate(kernel, positions, key, budget=2000, seed=29)
    result = caught.value.result
    _batch_counts(result)
    assert result.stop_reason == "unresolved"
    assert result.failed_operation == "constrained slice"
    assert result.posterior_weights is result.posterior_unit_samples is None
    assert len(result.batches) == 1 and not result.batches[0].completed
    assert result.batches[0].batched_slices is not None
    assert not np.asarray(result.batches[0].batched_slices[2].numerically_valid).all()
    # Every budgeted slice stops at its first invalid logical evaluation.
    assert caught.value.numerical_failure_count == int(
        np.count_nonzero(~np.asarray(result.batches[0].batched_slices[2].numerically_valid))
    )
    assert result.physical_likelihood_calls is None
    assert result.n_iterations == 0
    assert result.dead_log_volumes.size == result.dead_log_volume_weights.size == 0
    np.testing.assert_array_equal(result.final_state.particles.position, positions)
    for received, reference in zip(models.jax.tree.leaves(result.final_state),
                                    models.jax.tree.leaves(result.initial_state)):
        np.testing.assert_array_equal(received, reference)


def test_opt_in_estimator_enables_batched_driver_and_retains_explicit_scalar_override(device, monkeypatch):
    class RoutedIntegrand(AnalyticIntegrand):
        baseline_likelihood = None
        population = None
        target_feh = None

        def jax_unit_integrand(self, *, device):
            return AnalyticIntegrand(self.kind, device)

    integrand = RoutedIntegrand("exponential", device)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    controls = _routing_controls(10_000)
    expected = nested_evidence._run_native_nested_sampler(integrand, controls, 1, batched=False)
    actual = nested_evidence._integrate(integrand, controls, "TP", 1, None)
    assert actual.slice_execution == "vmapped-chains"
    assert expected.slice_execution == "scalar-device-loop"
    assert actual.sampler == expected.sampler == "blackjax-nss-experimental"
    assert actual.log_evidence == expected.log_evidence
    assert actual.n_likelihood_calls == expected.n_likelihood_calls
    _batch_counts(actual)
    np.testing.assert_array_equal(actual.posterior_unit_samples, expected.posterior_unit_samples)
    np.testing.assert_array_equal(actual.posterior_weights, expected.posterior_weights)


def _slice_rounding_diagnostic(attempt):
    """Independent exact-rational oracle for fused versus separate float64 rounding."""
    from fractions import Fraction

    from blackjax.ns.nss import sample_direction_from_covariance_factor

    jax, jnp = models.jax, models.jnp
    prop_key, slice_key = jax.random.split(attempt.key)
    _, _, shrink_key = jax.random.split(slice_key, 3)
    direction = float(sample_direction_from_covariance_factor(
        prop_key, attempt.start.position, jnp.eye(1, dtype=jnp.float64))[0])
    start = float(attempt.start.position[0])
    uniforms = []
    for _ in range(int(attempt.info.slice_info.num_shrink)):
        shrink_key, subkey = jax.random.split(shrink_key)
        uniforms.append(float(jax.random.uniform(subkey)))
    candidates = {}
    compiled_direction = jax.jit(lambda key, point: sample_direction_from_covariance_factor(
        key, point, jnp.eye(1, dtype=jnp.float64)))
    candidates["eager_direction"] = direction.hex()
    candidates["compiled_direction"] = float(compiled_direction(prop_key, attempt.start.position)[0]).hex()
    candidates["vmapped_direction"] = float(jax.jit(jax.vmap(compiled_direction))(
        jnp.stack([prop_key] * 4), jnp.stack([attempt.start.position] * 4))[0, 0]).hex()
    for fused_t in (False, True):
        for fused_position in (False, True):
            lo, hi = map(float, (attempt.info.slice_info.bracket_left, attempt.info.slice_info.bracket_right))
            for u in uniforms:
                width = hi - lo
                t = (float(Fraction(lo) + Fraction(u) * Fraction(width)) if fused_t else lo + u * width)
                position = (float(Fraction(start) + Fraction(t) * Fraction(direction))
                            if fused_position else start + t * direction)
                if t < 0.0:
                    lo = t
                else:
                    hi = t
            candidates[f"fused_t={fused_t},fused_position={fused_position}"] = position.hex()
    lo, hi = map(float, (attempt.info.slice_info.bracket_left, attempt.info.slice_info.bracket_right))
    for u in uniforms:
        width = hi - lo
        exact_t = Fraction(lo) + Fraction(u) * Fraction(width)
        position = float(Fraction(start) + exact_t * Fraction(direction))
        t = float(exact_t)
        if t < 0.0:
            lo = t
        else:
            hi = t
    candidates["fused_entire_position"] = position.hex()
    return candidates


def test_batched_each_slice_preserves_scalar_proposal_arithmetic(device, record_property):
    scalar, positions, key = _batch_case("exponential", device, batched=False)
    batched, _, _ = _batch_case("exponential", device)
    expected = _integrate(scalar, positions, key, budget=5000, seed=29)
    actual = _integrate(batched, positions, key, budget=5000, seed=29)
    checked = 0
    for batch_index, (received, reference) in enumerate(zip(actual.batches, expected.batches)):
        if received.batched_slices is None:
            continue
        _, _, infos = received.batched_slices
        for attempt_index, attempt in enumerate(reference.slices):
            lane, step = divmod(attempt_index, actual.num_inner_steps)
            selected = models.jax.tree.map(lambda leaf, lane=lane, step=step: leaf[lane, step], infos)
            for field in ("slice_info", "likelihood_calls", "last_position", "last_log_integrand",
                          "numerically_valid", "budget_exhausted"):
                for left, right in zip(models.jax.tree.leaves(getattr(selected, field)),
                                       models.jax.tree.leaves(getattr(attempt.info, field))):
                    diagnostic = ""
                    if field == "last_position" and not np.array_equal(left, right):
                        record_property("rounding_oracle", str(_slice_rounding_diagnostic(attempt)))
                        scalar_hlo = scalar.budgeted_slice.lower(attempt.key, attempt.start,
                            models.jnp.asarray(reference.threshold),
                            models.jnp.asarray(5000, dtype=models.jnp.int64)).compile().as_text()
                        keys, parents, _ = received.batched_slices
                        starts = models.jax.tree.map(lambda leaf, idx=parents: leaf[idx], actual.initial_state.particles)
                        batched_hlo = batched.batched_chains.lower(keys, starts,
                            models.jnp.asarray(received.threshold)).compile().as_text()
                        record_property("scalar_optimized_hlo", scalar_hlo)
                        record_property("batched_optimized_hlo", batched_hlo)
                        diagnostic = (f"; scalar={float(np.asarray(right)[0]).hex()}"
                                      f"; batched={float(np.asarray(left)[0]).hex()}"
                                      f"; rounding_oracle={_slice_rounding_diagnostic(attempt)}")
                    np.testing.assert_array_equal(left, right,
                        err_msg=f"batch={batch_index}, lane={lane}, step={step}, field={field}{diagnostic}")
            checked += 1
    assert checked > 0


def _padded_initialization_kernel(kind, device, *, block=16, metric="fixed-isotropic", live=30):
    return NativeNestedKernel(AnalyticIntegrand(kind, device), n_dimensions=1, n_live_points=live,
                              num_delete=4, num_inner_steps=2, batched_initialization=True,
                              initialization_block=block, direction_metric=metric)


def test_batched_initialization_matches_sequential_initial_particles(device):
    # Same points, keys and budget; only the evaluation batch shape differs, so
    # the documented XLA cross-shape gate applies. The later trajectory is
    # chaotic and is deliberately not compared.
    positions = np.random.default_rng(17).uniform(size=(30, 2))
    key = models.jax.random.PRNGKey(17)
    controls = dict(n_dimensions=2, n_live_points=30, num_delete=4, num_inner_steps=2)
    sequential = NativeNestedKernel(AnalyticIntegrand("exponential", device), **controls)
    batched = NativeNestedKernel(AnalyticIntegrand("exponential", device),
                                 batched_initialization=True, initialization_block=16, **controls)
    expected = _integrate(sequential, positions, key, budget=60)
    actual = _integrate(batched, positions, key, budget=60)
    np.testing.assert_allclose(np.asarray(actual.initial_state.particles.position),
                               np.asarray(expected.initial_state.particles.position), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual.initial_state.particles.loglikelihood),
                               np.asarray(expected.initial_state.particles.loglikelihood),
                               rtol=1e-13, atol=1e-13)
    for received, reference in zip(actual.initial_particles, expected.initial_particles):
        np.testing.assert_allclose(np.asarray(received.loglikelihood),
                                   np.asarray(reference.loglikelihood), rtol=1e-13, atol=1e-13)
    assert len(actual.initial_particles) == len(expected.initial_particles) == 30
    assert actual.likelihood_call_breakdown["initialization"] == 30
    # 30 points in blocks of 16 dispatch 16 + 16 physical lanes: two padding lanes.
    assert actual.physical_likelihood_calls == actual.n_likelihood_calls + 2
    assert expected.physical_likelihood_calls == expected.n_likelihood_calls


def test_batched_initialization_clamps_exactly_at_the_hard_ceiling(device):
    kernel = _padded_initialization_kernel("exponential", device)
    positions = np.random.default_rng(17).uniform(size=(30, 1))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(17), budget=10)
    assert result.stop_reason == "truncated"
    assert result.n_likelihood_calls == 10 == result.monitored_likelihood_calls
    assert len(result.initial_particles) == 10
    assert result.likelihood_call_breakdown == dict(initialization=10, left=0, right=0, shrinkage=0)
    assert result.physical_likelihood_calls == 16  # one dispatched block of 16 lanes
    assert result.final_state is None and result.posterior_weights is None
    assert result.budget_exhausted and result.stopped_short and result.failure_reason
    for index, particle in enumerate(result.initial_particles):
        np.testing.assert_array_equal(np.asarray(particle.position), positions[index])


def test_batched_initialization_invalid_lane_retains_only_the_sequential_prefix(device):
    class BelowHalf(AnalyticIntegrand):
        def evaluate(self, point):
            return super().evaluate(point)._replace(numerically_valid=point[0] < 0.5)

    positions = np.random.default_rng(17).uniform(size=(30, 1))
    positions[:8] = np.arange(8, dtype=np.float64)[:, None] / 16.0
    positions[5] = 0.75  # first invalid lane, after five valid ones
    kernel = NativeNestedKernel(BelowHalf("exponential", device), n_dimensions=1, n_live_points=30,
                                num_delete=4, num_inner_steps=2, batched_initialization=True,
                                initialization_block=16)
    with pytest.raises(RuntimeError, match="numerical") as caught:
        _integrate(kernel, positions, models.jax.random.PRNGKey(17), budget=100)
    result = caught.value.result
    assert result.stop_reason == "unresolved"
    assert result.failed_operation == "initialization"
    assert caught.value.numerical_failure_count == 1
    assert len(result.initial_particles) == 6 == result.n_likelihood_calls
    assert result.likelihood_call_breakdown["initialization"] == 6
    assert result.physical_likelihood_calls == 16  # 6 committed + 10 discarded lanes
    assert result.final_state is None and result.posterior_weights is None
    assert result.failed_attempt["remaining_budget"] == 100 - 5
    np.testing.assert_array_equal(np.asarray(result.failed_attempt["position"]), positions[5])
    for index, particle in enumerate(result.initial_particles):
        np.testing.assert_array_equal(np.asarray(particle.position), positions[index])


def test_batched_initialization_padding_is_reported_on_a_successful_plateau(device):
    kernel = _padded_initialization_kernel("constant", device)
    positions = np.random.default_rng(17).uniform(size=(30, 1))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(17), budget=1000)
    assert result.stop_reason == "likelihood-plateau"
    assert result.log_evidence == pytest.approx(math.log(2.0), abs=2e-14)
    assert result.n_likelihood_calls == 30
    assert result.likelihood_call_breakdown == dict(initialization=30, left=0, right=0, shrinkage=0)
    assert result.physical_likelihood_calls == 32  # 16 + 16 dispatched, two padding lanes


def test_batched_initialization_pause_after_a_committed_block_keeps_the_prefix(tmp_path, monkeypatch, device):
    import json

    from trex.evidence import _blackjax_evidence as evidence
    from trex.evidence._blackjax_checkpoint import load_native
    from trex.evidence.nested_checkpoint import NestedCheckpointPaused

    tmp_path.chmod(0o700)
    kernel = _padded_initialization_kernel("constant", device)
    positions = np.random.default_rng(17).uniform(size=(30, 1))
    key = models.jax.random.PRNGKey(17)
    path = tmp_path / "batched-initialization.npz"
    options = dict(evidence_tolerance=.1, max_likelihood_calls=1000, random_seed=17,
                   checkpoint=path, checkpoint_identity="batched-init")
    # Clock calls: retained snapshot baseline, first block boundary, second boundary.
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(evidence, "monotonic", lambda: next(ticks, 2.0))
    with pytest.raises(NestedCheckpointPaused):
        kernel.integrate(positions, key, deadline=1.0, **options)
    with np.load(path, allow_pickle=False) as stored:
        snapshot = load_native(path, json.loads(str(stored["metadata"].item()))["identity"])
    assert snapshot["clean_pause"]
    assert len(snapshot["state"][1]) == 16  # only the committed block, no padded lanes
    assert snapshot["state"][6]["initialization"] == 16
    monkeypatch.setattr(evidence, "monotonic", lambda: 0.0)
    resumed = kernel.integrate(positions, key, **options)
    assert len(resumed.initial_particles) == 30
    assert resumed.likelihood_call_breakdown["initialization"] == 30
    assert resumed.physical_likelihood_calls == 32  # the resumed block's two padding lanes persist
    assert resumed.stop_reason == "likelihood-plateau"


@pytest.mark.parametrize("field,value", [("batched_initialization", 1), ("batched_initialization", "yes"),
                                         ("initialization_block", 0), ("initialization_block", 24),
                                         ("initialization_block", True), ("direction_metric", "adaptive"),
                                         ("speculative_expansions", 1), ("speculative_expansions", True)])
def test_invalid_opt_in_execution_policies_are_rejected_before_evaluation(device, field, value):
    class NeverEvaluated(AnalyticIntegrand):
        def evaluate(self, point):
            raise AssertionError("invalid policies must fail before any evaluation")

    controls = dict(n_dimensions=1, n_live_points=32, num_delete=4, num_inner_steps=2)
    controls[field] = value
    with pytest.raises(ValueError, match=field):
        NativeNestedKernel(NeverEvaluated("constant", device), **controls)


@pytest.mark.parametrize("metric", ["fixed-pilot", "split-ensemble"])
def test_opt_in_direction_metrics_keep_scalar_and_batched_identical_key_parity(device, metric):
    # Same strength as the default-metric parity control: identical keys, points
    # and threshold, so bitwise equality is expected; not calibration.
    positions = np.random.default_rng(29).uniform(size=(32, 1))
    key = models.jax.random.PRNGKey(29)
    controls = dict(n_dimensions=1, n_live_points=32, num_delete=4, num_inner_steps=2,
                    direction_metric=metric)
    expected, actual = [
        _integrate(NativeNestedKernel(AnalyticIntegrand("exponential", device), batched=flag, **controls),
                   positions, key, budget=5000, seed=29) for flag in (False, True)]
    assert any(batch.batched_slices is not None for batch in actual.batches)
    assert actual.direction_metric == expected.direction_metric == metric
    assert actual.log_evidence == expected.log_evidence
    assert actual.log_evidence_error == expected.log_evidence_error
    assert actual.stop_reason == expected.stop_reason
    assert actual.n_likelihood_calls == expected.n_likelihood_calls
    assert actual.likelihood_call_breakdown == expected.likelihood_call_breakdown
    assert len(actual.batches) == len(expected.batches)
    for left, right in zip(actual.batches, expected.batches):
        assert left.completed == right.completed
        assert left.threshold == right.threshold
        np.testing.assert_array_equal(left.dead_indices, right.dead_indices)
    for name in ("posterior_unit_samples", "posterior_weights", "dead_log_volumes",
                 "dead_log_volume_weights", "final_live_log_volume_weights",
                 "compression_log_evidences", "continuation_key"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for received, reference in zip(models.jax.tree.leaves(actual.final_state),
                                    models.jax.tree.leaves(expected.final_state)):
        np.testing.assert_array_equal(received, reference)


@pytest.mark.parametrize("policy", [{}, {"direction_metric": "fixed-pilot"},
                                   {"direction_metric": "split-ensemble"},
                                   {"batched_initialization": True},
                                   # Speculation is a batched-chain-only policy.
                                   {"speculative_expansions": 4, "batched": True},
                                   {"speculative_expansions": 8, "batched": True}])
def test_native_result_reports_policy_without_inheriting_default_calibration(device, policy):
    """Check metadata admission only; a plateau run is not a calibration experiment."""
    from dataclasses import replace
    from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling

    controls = dict(n_dimensions=1, n_live_points=32, num_delete=4, num_inner_steps=2)
    kernel = NativeNestedKernel(AnalyticIntegrand("constant", device), **controls, **policy)
    positions = np.random.default_rng(17).uniform(size=(32, 1))
    result = _integrate(kernel, positions, models.jax.random.PRNGKey(17))
    assert result.direction_metric == policy.get("direction_metric", "fixed-isotropic")
    assert result.speculative_expansions == policy.get("speculative_expansions", 0)
    assert result.batched_initialization == policy.get("batched_initialization", False)
    assert result.stop_reason == "likelihood-plateau"
    live = BLACKJAX_CALIBRATED_LIVE_POINTS
    delete, steps = live // 2, 3
    metadata = replace(result, n_live_points=live, num_delete=delete, num_inner_steps=steps,
                       max_likelihood_calls=blackjax_call_ceiling(live, delete, steps))
    assert metadata.configuration_calibrated is (not policy)
    assert not replace(metadata, max_likelihood_calls=metadata.max_likelihood_calls - 1).configuration_calibrated


def test_split_ensemble_factor_for_a_parent_in_the_first_half_comes_from_the_second_half():
    from trex.evidence._blackjax_evidence import _ridge_cholesky_factor, _split_ensemble_chain_factors

    positions = np.random.default_rng(5).uniform(size=(12, 2))
    loglikelihood = np.linspace(-3.0, 0.0, 12)
    threshold = -1.0  # strict survivors 8..11 split into halves [8, 9] and [10, 11]
    parents = np.asarray([8, 9])
    factor = _split_ensemble_chain_factors(positions, loglikelihood, threshold, parents)
    np.testing.assert_array_equal(factor[0], _ridge_cholesky_factor(positions[[10, 11]]))
    np.testing.assert_array_equal(factor[1], _ridge_cholesky_factor(positions[[10, 11]]))
    changed_first = positions.copy()
    changed_first[[8, 9]] += 0.25
    np.testing.assert_array_equal(factor, _split_ensemble_chain_factors(
        changed_first, loglikelihood, threshold, parents))
    changed_second = positions.copy()
    changed_second[10] += 0.25
    assert not np.array_equal(factor, _split_ensemble_chain_factors(
        changed_second, loglikelihood, threshold, parents))
    collapsed = _ridge_cholesky_factor(np.full((4, 2), 0.5))
    assert np.isfinite(collapsed).all() and (np.diag(collapsed) > 0.0).all()
    # One strict survivor is the end-game batch: no ensemble spread exists.
    single = _split_ensemble_chain_factors(positions, loglikelihood, loglikelihood[-2], np.asarray([11, 11]))
    np.testing.assert_array_equal(single, np.repeat(np.eye(2)[None], 2, axis=0))
    with pytest.raises(ValueError, match="strict survivor"):
        _split_ensemble_chain_factors(positions, loglikelihood, loglikelihood[-1], np.asarray([11]))


def test_checkpoint_identity_separates_opt_in_execution_policies(tmp_path, monkeypatch, device):
    from trex.evidence import _blackjax_evidence as evidence
    from trex.evidence.nested_checkpoint import CheckpointInputMismatch, NestedCheckpointPaused

    tmp_path.chmod(0o700)
    positions = np.random.default_rng(17).uniform(size=(32, 1))
    key = models.jax.random.PRNGKey(17)
    path = tmp_path / "policy.npz"
    options = dict(evidence_tolerance=.1, max_likelihood_calls=1000, random_seed=17,
                   checkpoint=path, checkpoint_identity="policy")
    controls = dict(n_dimensions=1, n_live_points=32, num_delete=4, num_inner_steps=2)
    monkeypatch.setattr(evidence, "monotonic", lambda: 2.0)
    default = NativeNestedKernel(AnalyticIntegrand("constant", device), **controls)
    with pytest.raises(NestedCheckpointPaused):
        default.integrate(positions, key, deadline=1.0, **options)  # zero evaluations retained
    for policy in (dict(direction_metric="fixed-pilot"), dict(direction_metric="split-ensemble"),
                   dict(batched_initialization=True)):
        with pytest.raises(CheckpointInputMismatch):
            NativeNestedKernel(AnalyticIntegrand("constant", device), **controls, **policy).integrate(
                positions, key, **options)
    monkeypatch.setattr(evidence, "monotonic", lambda: 0.0)
    resumed = default.integrate(positions, key, **options)
    assert resumed.stop_reason == "likelihood-plateau" and resumed.n_likelihood_calls == 32

    # Speculation exists only on the batched path, so it is checked against a
    # batched default checkpoint; the scalar kernel refuses it outright.
    with pytest.raises(ValueError, match="requires batched=True"):
        NativeNestedKernel(AnalyticIntegrand("constant", device), **controls, speculative_expansions=4)
    batched_options = dict(options, checkpoint=tmp_path / "batched-policy.npz")
    monkeypatch.setattr(evidence, "monotonic", lambda: 2.0)
    batched_default = NativeNestedKernel(AnalyticIntegrand("constant", device), **controls, batched=True)
    with pytest.raises(NestedCheckpointPaused):
        batched_default.integrate(positions, key, deadline=1.0, **batched_options)
    for speculation in (4, 8):
        with pytest.raises(CheckpointInputMismatch):
            NativeNestedKernel(AnalyticIntegrand("constant", device), **controls, batched=True,
                               speculative_expansions=speculation).integrate(positions, key, **batched_options)


@pytest.mark.parametrize("speculation", [4, 8])
def test_speculative_expansions_preserve_the_sequential_batched_transition(device, speculation):
    # The wave's probes are the sequential loop's own iterated positions, so the
    # bracket, state and logical counters must match bitwise, not statistically.
    positions = np.random.default_rng(29).uniform(size=(32, 1))
    key = models.jax.random.PRNGKey(29)
    controls = dict(n_dimensions=1, n_live_points=32, num_delete=4, num_inner_steps=2, batched=True)
    baseline = _integrate(NativeNestedKernel(AnalyticIntegrand("exponential", device), **controls),
                          positions, key, budget=5000, seed=29)
    actual = _integrate(NativeNestedKernel(AnalyticIntegrand("exponential", device),
                                           speculative_expansions=speculation, **controls),
                        positions, key, budget=5000, seed=29)
    assert actual.speculative_expansions == speculation
    assert actual.log_evidence == baseline.log_evidence
    assert actual.n_likelihood_calls == baseline.n_likelihood_calls
    assert actual.likelihood_call_breakdown == baseline.likelihood_call_breakdown
    checked = 0
    for received, reference in zip(actual.batches, baseline.batches):
        assert received.completed == reference.completed and received.threshold == reference.threshold
        np.testing.assert_array_equal(received.dead_indices, reference.dead_indices)
        if received.batched_slices is None:
            continue
        for field in ("slice_info", "likelihood_calls", "remaining_budget", "last_position",
                      "last_log_integrand", "numerically_valid", "budget_exhausted"):
            for left, right in zip(models.jax.tree.leaves(getattr(received.batched_slices[2], field)),
                                    models.jax.tree.leaves(getattr(reference.batched_slices[2], field))):
                np.testing.assert_array_equal(left, right)
        checked += 1
    assert checked > 0
    for left, right in zip(models.jax.tree.leaves(actual.final_state),
                            models.jax.tree.leaves(baseline.final_state)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("speculation", [4, 8])
@pytest.mark.parametrize("kind", ["exponential", "invalid-proposal"])
@pytest.mark.parametrize("step", [.01, .25])
def test_speculative_slice_keeps_budget_interruptions_and_invalid_probe_history(device, speculation, kind, step):
    from trex.evidence._blackjax_slice import _build_budgeted_slice
    from trex.evidence._blackjax_types import SLICE_CALL_BOUND

    kernel, _, _ = _case(kind, device)
    jax, jnp = models.jax, models.jnp
    start, valid = kernel.integration_evaluate(jnp.asarray([.5]), jnp.asarray(-jnp.inf))
    assert bool(valid)
    speculative = jax.jit(_build_budgeted_slice(kernel.integration_evaluate, 1, speculation))
    # Fixed direction isolates expansion/bracket arithmetic from direction sampling.
    direction = jnp.asarray([step])
    for seed in (17, 29):
        for budget in (0, 1, 2, 3, 5, 9, SLICE_CALL_BOUND):
            args = (jax.random.PRNGKey(seed), start, jnp.asarray(-10.), jnp.asarray(budget, dtype=jnp.int64))
            expected = jax.block_until_ready(kernel.budgeted_slice(*args, direction=direction))
            actual = jax.block_until_ready(speculative(*args, direction=direction))
            for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                np.testing.assert_array_equal(left, right)
