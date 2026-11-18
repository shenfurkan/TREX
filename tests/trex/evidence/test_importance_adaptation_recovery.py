"""Real defensive-importance evaluation, deadlines and retained adaptive work."""

import math
import multiprocessing

import numpy as np
import pytest

from trex.evidence import mc_importance, nested_controls, nested_evidence
from trex.evidence.nested_checkpoint import load_importance_adaptation
from trex.foundation.errors import NumericalLikelihoodError


EVALUATED = []


def _identity(point):
    return point


def _linear_likelihood(point):
    EVALUATED.append(tuple(point))
    return math.log1p(point[0])


def _invalid_likelihood(point):
    return float("nan")


class _NativeLinearLikelihood:
    n_dimensions = 2
    time = np.zeros(1)
    nsamples = 1

    def jax_unit_integrand(self, *, device):
        from trex.accelerators import _jax_models as models
        from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation

        class Native:
            def evaluate(self, point):
                return UnitIntegrandEvaluation(models.jnp.log1p(point[0]), models.jnp.isfinite(point).all())

        return Native()


def _run(*, seed=19, n_jobs=1, likelihood=_linear_likelihood, native=None):
    samples = np.array([(x, y) for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75)])
    return nested_evidence._run_importance_sampler(
        _identity, likelihood, [(samples, np.full(len(samples), 1.0 / len(samples)))],
        n_dimensions=2, random_seed=seed, n_jobs=n_jobs, n_draws=4096,
        cpu_native_integrand=native,
    )


@pytest.fixture
def adaptive_stage(monkeypatch):
    from trex.accelerators import _jax_batch

    EVALUATED.clear()
    # Keep multiple legal native recovery waves without replacing the calculation.
    monkeypatch.setattr(_jax_batch, "_BLOCK_MAX_DRAWS", 8)
    monkeypatch.setattr(nested_evidence, "_IMPORTANCE_CENTRES", 16)
    monkeypatch.setattr(nested_controls, "_IMPORTANCE_CENTRES", 16)
    monkeypatch.setattr(nested_evidence, "_IMPORTANCE_EVALUATION_CHUNK", 8)
    monkeypatch.setattr(nested_controls, "_IMPORTANCE_ADAPTATION_DRAWS", 64)
    monkeypatch.setattr(nested_controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 2)
    monkeypatch.setattr(nested_controls, "_IMPORTANCE_ADAPTATION_ESS", 2.0)
    token = mc_importance.native_stage.set(("adaptation-recovery", "linear", mc_importance.stage_controls()))
    try:
        yield
    finally:
        mc_importance.native_stage.reset(token)
        EVALUATED.clear()


def test_expired_deadline_never_evaluates_adaptation(adaptive_stage, monkeypatch, tmp_path):
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 1.0)
    with pytest.raises(nested_evidence.NestedCheckpointPaused):
        with nested_evidence.nested_checkpoint(tmp_path / "linear", deadline=0.0):
            _run()
    assert EVALUATED == []


def test_total_cost_includes_actual_adaptive_evaluations(adaptive_stage):
    result = _run()
    assert result.n_likelihood_calls == len(EVALUATED)
    provenance = result.execution_provenance
    adaptive_calls = sum(record["likelihood_calls"] for record in provenance["adaptation"])
    assert adaptive_calls > 0
    assert provenance["logical_likelihood_calls"] == len(EVALUATED)
    assert provenance["adaptation_likelihood_calls"] == adaptive_calls
    assert provenance["estimation_likelihood_calls"] + adaptive_calls == len(EVALUATED)
    # Integral of 1 + u[0] over the two-dimensional unit cube is exactly 3/2.
    assert result.log_evidence == pytest.approx(math.log(1.5), abs=6 * result.log_evidence_error)


@pytest.mark.parametrize("phase", ["adaptation", "estimation"])
@pytest.mark.parametrize("backend", ["scalar", "native"])
def test_interrupted_adaptation_resumes_without_repeating_likelihoods(
    adaptive_stage, monkeypatch, tmp_path, phase, backend,
):
    native = None
    if backend == "native":
        from trex.accelerators import _jax_models as models
        from trex.accelerators._native_unit_batch import NativeUnitCubeBatch

        if not models.HAS_JAX_TREX:
            pytest.skip("JAX transit kernels unavailable")
        native = _NativeLinearLikelihood()
        evaluate = NativeUnitCubeBatch.evaluate

        def observe_points(batch, *, subset=None):
            points = batch.points if subset is None else batch.points[subset]
            result = evaluate(batch, subset=subset)
            # Observe completed logical points; the native numerical calculation is unchanged.
            EVALUATED.extend(tuple(point) for point in points)
            return result

        monkeypatch.setattr(NativeUnitCubeBatch, "evaluate", observe_points)
    reference = _run(native=native)
    reference_points = list(EVALUATED)
    EVALUATED.clear()
    threshold = 8 if phase == "adaptation" else reference.execution_provenance["adaptation_likelihood_calls"] + 8
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 2.0 if len(EVALUATED) >= threshold else 0.0)
    prefix = tmp_path / "linear"
    with pytest.raises(nested_evidence.NestedCheckpointPaused):
        with nested_evidence.nested_checkpoint(prefix, deadline=1.0):
            _run(native=native)
    assert len(EVALUATED) == threshold
    assert prefix.with_suffix(".adaptation.npz").is_file()
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 0.0)
    with nested_evidence.nested_checkpoint(prefix, deadline=1.0):
        resumed = _run(native=native)
    np.testing.assert_array_equal(EVALUATED, reference_points)
    assert resumed.n_likelihood_calls == reference.n_likelihood_calls == len(EVALUATED)
    for field in ("log_evidence", "log_evidence_error", "effective_sample_size", "maximum_weight_share"):
        assert getattr(resumed, field) == getattr(reference, field)


@pytest.mark.parametrize("defect", ["seed", "weights", "rng"])
def test_changed_or_damaged_adaptation_is_refused_before_more_evaluations(adaptive_stage, monkeypatch, tmp_path, defect):
    prefix = tmp_path / "linear"
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 2.0 if len(EVALUATED) >= 8 else 0.0)
    with pytest.raises(nested_evidence.NestedCheckpointPaused):
        with nested_evidence.nested_checkpoint(prefix, deadline=1.0):
            _run()
    path = prefix.with_suffix(".adaptation.npz")
    if defect != "seed":
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: np.array(saved[key]) for key in saved.files}
        if defect == "weights":
            arrays["weights"][0] = -1.0
        else:
            import json

            metadata = json.loads(str(arrays["metadata"].item()))
            metadata["rng_state"] = {}
            arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 0.0)
    with pytest.raises(ValueError, match="inputs do not match|corrupt or incomplete"):
        with nested_evidence.nested_checkpoint(prefix, deadline=1.0):
            _run(seed=20 if defect == "seed" else 19)
    assert len(EVALUATED) == 8


def test_zero_support_adaptive_pass_still_records_its_real_cost(adaptive_stage):
    from trex.evidence.nested_importance import _LogLikelihoodBlock, _adapt_proposal

    evaluated = []

    def zero_likelihood(point):
        evaluated.append(tuple(point))
        return -math.inf

    samples = np.array([(x, y) for x in (0.25, 0.75) for y in (0.25, 0.75)])
    _, _, passes = _adapt_proposal(
        samples, np.full(len(samples), 1.0 / len(samples)),
        _LogLikelihoodBlock(_identity, zero_likelihood), np.random.default_rng(19),
    )
    assert len(passes) == 1
    assert passes[0]["likelihood_calls"] == len(evaluated) > 0
    assert passes[0]["effective_sample_size"] == 0.0


def test_adaptive_failure_leaves_no_spawn_workers(adaptive_stage):
    before = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(NumericalLikelihoodError):
        _run(n_jobs=2, likelihood=_invalid_likelihood)
    assert {process.pid for process in multiprocessing.active_children()} == before


@pytest.mark.parametrize("draws", [0, 1, True, 2.5])
def test_invalid_estimation_budget_is_refused_before_evaluation(adaptive_stage, draws):
    with pytest.raises(ValueError, match="draw count"):
        nested_evidence._run_importance_sampler(
            _identity, _linear_likelihood, [(np.array([[0.25], [0.75]]), np.array([0.5, 0.5]))],
            n_dimensions=1, random_seed=19, n_jobs=1, n_draws=draws,
        )
    assert EVALUATED == []


def test_saved_adaptation_retains_an_atomic_likelihood_prefix(adaptive_stage, monkeypatch, tmp_path):
    prefix = tmp_path / "linear"
    monkeypatch.setattr(nested_evidence, "_monotonic", lambda: 2.0 if len(EVALUATED) >= 8 else 0.0)
    with pytest.raises(nested_evidence.NestedCheckpointPaused):
        with nested_evidence.nested_checkpoint(prefix, deadline=1.0):
            _run()
    with np.load(prefix.with_suffix(".adaptation.npz"), allow_pickle=False) as saved:
        import json

        identity = json.loads(str(saved["metadata"].item()))["identity"]
    samples, weights, passes, _, values, finished = load_importance_adaptation(prefix, identity, 2, 2, 64)
    assert not finished and passes == []
    assert weights.sum() == pytest.approx(1.0)
    assert samples.shape == (9, 2)
    np.testing.assert_array_equal(values, [math.log1p(point[0]) for point in EVALUATED])
