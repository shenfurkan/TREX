"""Native restoration is exact at logical boundaries, not a fresh seeded run."""
import math
import json

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import _blackjax_evidence as evidence
from trex.evidence._blackjax_kernel import NativeNestedKernel
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation
from trex.evidence.nested_checkpoint import NestedCheckpointPaused, load_completed, save_completed


class Gaussian:
    def __init__(self):
        self.device = models.jax.devices("cpu")[0]

    def evaluate(self, point):
        jnp = models.jnp
        value = -jnp.sum(((point - .5) / .15)**2) / 2
        return UnitIntegrandEvaluation(jnp.where(jnp.all((point >= 0) & (point < 1)), value, -jnp.inf),
                                       jnp.isfinite(point).all())


class GaussianReference:
    population = None
    target_feh = None

    def jax_unit_integrand(self, *, device):
        result = Gaussian()
        result.device = device
        return result


def kernel(**policy):
    return NativeNestedKernel(Gaussian(), n_dimensions=2, n_live_points=16,
                              num_delete=4, num_inner_steps=2, batched=True, **policy)


@pytest.mark.parametrize("boundary", ["initialization", "deletion"])
@pytest.mark.parametrize("unclean", [False, True], ids=["clean-pause", "unclean-snapshot"])
@pytest.mark.parametrize("policy", [{}, {"direction_metric": "fixed-pilot"},
                                   {"direction_metric": "split-ensemble"}, {"speculative_expansions": 4}])
def test_pause_restores_live_keys_deaths_volumes_and_counters(tmp_path, monkeypatch, boundary, unclean, policy):
    tmp_path.chmod(0o700)
    native = kernel(**policy)
    positions = np.random.default_rng(37).uniform(size=native.shape)
    key = models.jax.random.PRNGKey(41)
    options = dict(evidence_tolerance=.05, max_likelihood_calls=100_000, random_seed=37)
    reference = native.integrate(positions, key, **options)
    paused = False
    evaluated = 0
    evaluate, chains = native.integration_evaluate, native.batched_chains

    def measured_evaluate(*args):
        nonlocal paused, evaluated
        value = evaluate(*args)
        evaluated += 1
        if boundary == "initialization" and evaluated == 5:
            paused = True
        return value

    def measured_chains(*args):
        nonlocal paused
        value = chains(*args)
        paused = True
        return value

    monkeypatch.setattr(native, "integration_evaluate", measured_evaluate)
    monkeypatch.setattr(native, "batched_chains", measured_chains)
    monkeypatch.setattr(evidence, "monotonic", lambda: 2.0 if paused else 0.0)
    path = tmp_path / "native.npz"
    with pytest.raises(NestedCheckpointPaused):
        native.integrate(positions, key, checkpoint=path, checkpoint_identity="known-gaussian", deadline=1., **options)
    assert path.is_file()
    assert evaluated == (5 if boundary == "initialization" else native.shape[0])
    if unclean:
        from trex.evidence._blackjax_checkpoint import load_native, save_native

        # A periodic snapshot carries the same logical state, but no certificate
        # that a killed process did no further uncommitted physical work.
        with np.load(path, allow_pickle=False) as stored:
            identity = json.loads(str(stored["metadata"].item()))["identity"]
        snapshot = load_native(path, identity)
        snapshot["clean_pause"] = False
        save_native(path, identity, snapshot)
    # A new kernel/process starts from the retained particle count/live state.
    restored = kernel(**policy)
    calls = []
    original = restored.integration_evaluate
    def trace(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(restored, "integration_evaluate", trace)
    resumed = restored.integrate(positions, key, checkpoint=path, checkpoint_identity="known-gaussian", **options)
    assert resumed.likelihood_calls_exact is (not unclean)
    if unclean:
        assert resumed.physical_likelihood_calls is None
    assert len(calls) == (native.shape[0] - 5 if boundary == "initialization" else 0)
    for field in ("log_evidence", "log_evidence_error", "n_likelihood_calls", "n_iterations", "stop_reason"):
        assert getattr(resumed, field) == getattr(reference, field)
    for field in ("posterior_unit_samples", "posterior_weights", "dead_log_volumes", "compression_log_evidences",
                  "continuation_key"):
        np.testing.assert_array_equal(getattr(resumed, field), getattr(reference, field))
    completed = tmp_path / "completed.npz"
    save_completed(completed, "input", resumed)
    cached = load_completed(completed, "input")
    assert cached.sampler == "blackjax-nss-experimental"
    assert cached.log_evidence == resumed.log_evidence
    np.testing.assert_array_equal(cached.posterior_weights, resumed.posterior_weights)
    retained = path.read_bytes()
    with pytest.raises(ValueError, match="inputs do not match"):
        restored.integrate(positions, key, checkpoint=path, checkpoint_identity="known-gaussian",
                           **dict(options, random_seed=38))
    assert path.read_bytes() == retained


def test_failed_native_attempt_is_not_silently_retried(tmp_path):
    from trex.evidence._blackjax_checkpoint import save_native
    from trex.evidence._blackjax_types import NativeIntegrationError

    tmp_path.chmod(0o700)
    native = kernel()
    # All-zero live support is not a proof of an empty prior.
    class Empty(Gaussian):
        def evaluate(self, point):
            return UnitIntegrandEvaluation(models.jnp.asarray(-math.inf), models.jnp.asarray(True))
    native.integrand = Empty()
    # Test the recovery reader directly with a genuine failed result, not a fake posterior.
    empty = NativeNestedKernel(Empty(), n_dimensions=2, n_live_points=16,
                               num_delete=4, num_inner_steps=2)
    points = np.random.default_rng(17).uniform(size=empty.shape)
    with pytest.raises(NativeIntegrationError) as caught:
        empty.integrate(points, models.jax.random.PRNGKey(17), evidence_tolerance=.05,
                         max_likelihood_calls=1000, random_seed=17)
    path = tmp_path / "failure.npz"
    save_native(path, "failed-integrand", {"failed": caught.value.result})
    with pytest.raises(NativeIntegrationError, match="retained BlackJAX") as restored:
        evidence._integrate_positions(empty, points, models.jax.random.PRNGKey(17), .05, 1000, 17,
                                      checkpoint=path, identity="failed-integrand")
    assert restored.value.result.stop_reason == "unresolved"
    assert restored.value.result.n_likelihood_calls == 16


def test_selected_blackjax_public_dispatch_uses_its_native_checkpoint(tmp_path, monkeypatch):
    from trex.evidence import nested_evidence as nested

    tmp_path.chmod(0o700)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    # Pass public pre-setup admission, then expire at the native retain boundary.
    monkeypatch.setattr(nested, "_monotonic", lambda: 0.)
    monkeypatch.setattr(evidence, "monotonic", lambda: 2.)
    controls = nested._SamplerControls(16, .05, 100_000, 37, 1)
    prefix = tmp_path / "selected-blackjax"
    with nested.nested_checkpoint(prefix, deadline=1.), pytest.raises(NestedCheckpointPaused):
        nested._integrate(GaussianReference(), controls, "analytic", 2, None)
    assert prefix.with_suffix(".blackjax.npz").is_file()
    with nested.nested_checkpoint(prefix, deadline=math.inf):
        result = nested._integrate(GaussianReference(), controls, "analytic", 2, None)
    assert result.sampler == "blackjax-nss-experimental"
    assert result.n_live_points == 16  # Explicit software-only low-level control, not production calibration.
