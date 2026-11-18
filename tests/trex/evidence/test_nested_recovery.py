"""Real dynesty continuation; analytic likelihoods, no candidate evidence runs."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trex.evidence import evidence_escalation as escalation
from trex.evidence import nested_evidence as nested
from tests.trex.evidence.escalation_support import _base_kwargs, _binding_evidence, _result


def _prior(x):
    return x


def _likelihood(x):
    return -20.0 * float(np.sum((x - 0.5) ** 2))


def _process_run(mode, directory, n_jobs, pause_stage="base"):
    """Run the production sampler in an independently imported interpreter."""
    controls = nested._SamplerControls(30, 0.05, 1_000_000, 123, n_jobs)
    directory = Path(directory)
    if mode == "pause":
        if pause_stage == "base":
            clock = iter([0.0] * 12 + [100.0] * 100_000)
            nested._monotonic = lambda: next(clock)
        else:
            original_progress = nested._SamplerCheckpoint.progress
            def progress(self, sampler, *, clock, count):
                def batch_clock():
                    return 100.0 if str(sampler.internal_state).endswith(".INBATCH") else 0.0
                return original_progress(self, sampler, clock=batch_clock, count=count)
            nested._SamplerCheckpoint.progress = progress
            nested._monotonic = lambda: 0.0
    else:
        nested._monotonic = lambda: 0.0
    if mode == "reuse":
        def fail(*_args, **_kwargs):
            raise AssertionError("completed sampler was repeated")
        nested._worker_pool = fail
    context = getattr(nested, "nested_checkpoint", None)
    assert context is not None, "native nested checkpoint support is missing"
    try:
        with context(directory / "state", deadline=50.0):
            result = nested._run_nested_sampler(_prior, _likelihood, controls, "analytic", 2)
    except nested.NestedCheckpointPaused:
        assert mode == "pause"
        assert (directory / "state.dynesty").is_file()
        return
    assert mode != "pause", "deadline did not pause the actual sampler"
    arrays = {}
    values = result._asdict()
    for key, value in list(values.items()):
        if isinstance(value, np.ndarray):
            arrays[key] = values.pop(key)
    np.savez(directory / (mode + ".npz"), metadata=json.dumps(values), **arrays)


@pytest.mark.parametrize(("n_jobs", "pause_stage"), [(1, "base"), (2, "base"), (1, "batch")])
def test_native_sampler_resumes_across_fresh_processes(tmp_path, n_jobs, pause_stage):
    expected = tmp_path / "expected"
    resumed = tmp_path / "resumed"
    expected.mkdir()
    resumed.mkdir()
    for mode, directory, hash_seed in (
        ("complete", expected, "0"), ("pause", resumed, "1"),
        ("resume", resumed, "314159"), ("reuse", resumed, "42"),
    ):
        code = (
            "from tests.trex.evidence.test_nested_recovery import _process_run; "
            f"_process_run({mode!r}, {str(directory)!r}, {n_jobs}, {pause_stage!r})"
        )
        child = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "TREX_N_JOBS": str(n_jobs),
                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "JAX_PLATFORMS": "cpu"},
        )
        assert child.returncode == 0, child.stdout + child.stderr
    with np.load(expected / "complete.npz", allow_pickle=False) as reference:
        values = json.loads(reference["metadata"].item())
        # Independent finite-domain Gaussian integral, not a saved TREX output.
        log_z = 2 * math.log(math.sqrt(math.pi / 20) * math.erf(math.sqrt(20) / 2))
        assert abs(values["log_evidence"] - log_z) < 5 * values["log_evidence_error"]
        assert values["stopped_on_evidence_tolerance"]
        for mode in ("resume", "reuse"):
            with np.load(resumed / (mode + ".npz"), allow_pickle=False) as actual:
                assert json.loads(actual["metadata"].item()) == json.loads(reference["metadata"].item())
                np.testing.assert_array_equal(actual["posterior_unit_samples"], reference["posterior_unit_samples"])
                np.testing.assert_array_equal(actual["posterior_weights"], reference["posterior_weights"])


def test_failed_escalation_retains_attempt_cost(tmp_path, monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("controlled integration failure")
    monkeypatch.setattr(escalation, "eb_scenario_nested_evidence", fail)
    lnz, ess = _binding_evidence()
    kwargs = _base_kwargs(checkpoint_directory=tmp_path)
    outcome = escalation.run_nested_escalation(lnZ=lnz, ess=ess, **kwargs)
    assert outcome.get("attempt_costs"), "failed numerical attempt lost its cost"
    cost = outcome["attempt_costs"][0]
    assert cost["status"] == "failed"
    assert cost["wall_seconds"] >= 0
    assert cost["backend"] == "dynesty-scalar-cpu"
    retained = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert retained == outcome["attempt_costs"]


@pytest.mark.parametrize("failure", ["missing-plan", "unavailable", "numerical", "truncated"])
def test_independent_scenario_completes_and_is_reused_after_an_unresolved_one(tmp_path, monkeypatch, failure):
    from trex.foundation.errors import NumericalLikelihoodError

    calls = []
    def integrate(**kwargs):
        calls.append(kwargs["use_2xP"])
        if not kwargs["use_2xP"]:
            if failure == "unavailable":
                raise escalation.NestedEvidenceUnavailableError("synthetic missing first-scenario evidence")
            if failure == "numerical":
                raise NumericalLikelihoodError("synthetic first-scenario solver failure", 3)
            return _result(-100.2, stopped_short=True)
        return _result(-300.0)

    monkeypatch.setattr(escalation, "eb_scenario_nested_evidence", integrate)
    if failure == "missing-plan":
        original = escalation._target_plan_entry
        monkeypatch.setattr(escalation, "_target_plan_entry", lambda index: None if index == 1 else original(index))
    lnz, ess = _binding_evidence()
    lnz[2], ess[2] = -101.0, 1.0
    kwargs = _base_kwargs(checkpoint_directory=tmp_path)
    for _ in range(2):
        outcome = escalation.run_nested_escalation(lnZ=lnz, ess=ess, **kwargs)
        assert outcome["escalated"] == [2] and outcome["unmet_scenarios"] == ["EB"]
        assert outcome["attempted_scenarios"] == (["EBx2P"] if failure == "missing-plan" else ["EB", "EBx2P"])
        assert outcome["unmet_reasons"] == {"EB": outcome["unmet_reason"]}
        assert outcome["lnZ"][1] == lnz[1] and outcome["lnZ"][2] < -250
        assert outcome["remaining_ess"] == {0: ess[0], 1: ess[1]}
        if failure == "numerical":
            assert outcome["numerical_failures"]["EB"]["count"] == 3
    assert calls.count(True) == 1, "the later completed scenario was recomputed on recovery"


def test_each_failed_scenario_keeps_its_own_reason(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise escalation.NestedEvidenceUnavailableError("second failure" if kwargs["use_2xP"] else "first failure")

    monkeypatch.setattr(escalation, "eb_scenario_nested_evidence", unavailable)
    lnz, ess = _binding_evidence()
    lnz[2], ess[2] = -101.0, 1.0
    outcome = escalation.run_nested_escalation(lnZ=lnz, ess=ess, **_base_kwargs(checkpoint_directory=tmp_path))
    assert outcome["unmet_reasons"] == {"EB": "first failure", "EBx2P": "second failure"}
    assert outcome["unmet_reason"] == "first failure"
    np.testing.assert_array_equal(outcome["lnZ"], lnz)
    assert outcome["remaining_ess"] == ess and outcome["escalated"] == []


@pytest.mark.parametrize(("initial_workers", "resume_workers"), [(2, 1), (1, 2)])
def test_checkpoint_can_resume_when_the_host_worker_count_changes(tmp_path, initial_workers, resume_workers):
    expected = tmp_path / "expected"
    resumed = tmp_path / "resumed"
    expected.mkdir()
    resumed.mkdir()
    for mode, directory, workers in (
        ("complete", expected, initial_workers), ("pause", resumed, initial_workers),
        ("resume", resumed, resume_workers),
    ):
        child = subprocess.run(
            [sys.executable, "-c", ("from tests.trex.evidence.test_nested_recovery import _process_run; "
                                   f"_process_run({mode!r}, {str(directory)!r}, {workers})")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "TREX_N_JOBS": str(workers), "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"},
        )
        assert child.returncode == 0, child.stdout + child.stderr
    with np.load(expected / "complete.npz", allow_pickle=False) as reference:
        with np.load(resumed / "resume.npz", allow_pickle=False) as actual:
            assert json.loads(actual["metadata"].item()) == json.loads(reference["metadata"].item())
            np.testing.assert_array_equal(actual["posterior_unit_samples"], reference["posterior_unit_samples"])
            np.testing.assert_array_equal(actual["posterior_weights"], reference["posterior_weights"])


def test_completed_scenario_survives_interruption(tmp_path, monkeypatch):
    calls = []
    def run(**kwargs):
        calls.append(kwargs["random_seed"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _result(-100.2)
    monkeypatch.setattr(escalation, "eb_scenario_nested_evidence", run)
    lnz, ess = _binding_evidence()
    kwargs = _base_kwargs(checkpoint_directory=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        escalation.run_nested_escalation(lnZ=lnz, ess=ess, **kwargs)
    interrupted_costs = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [cost["status"] for cost in interrupted_costs] == ["completed", "interrupted"]
    first_seed = calls[0]
    outcome = escalation.run_nested_escalation(lnZ=lnz, ess=ess, **kwargs)
    assert calls.count(first_seed) == 1, "completed scenario was integrated again"
    assert any(cost["reused"] for cost in outcome["attempt_costs"])
    assert outcome["attempt_costs"][:2] == interrupted_costs
    assert outcome["cumulative_wall_seconds"] == sum(cost["wall_seconds"] for cost in outcome["attempt_costs"])


def test_expired_deadline_does_not_start_likelihood(tmp_path, monkeypatch):
    monkeypatch.setattr(escalation._time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(escalation, "eb_scenario_nested_evidence", lambda **_k: pytest.fail("started after deadline"))
    lnz, ess = _binding_evidence()
    kwargs = _base_kwargs(checkpoint_directory=tmp_path, deadline_monotonic=99.0)
    with pytest.raises(nested.NestedCheckpointPaused):
        escalation.run_nested_escalation(lnZ=lnz, ess=ess, **kwargs)


def test_importance_refinement_keeps_partial_values_without_repeating_likelihoods(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    posterior = [(rng.uniform(size=(200, 2)), np.full(200, 1 / 200))]
    monkeypatch.setattr(nested, "_IMPORTANCE_EVALUATION_CHUNK", 32)
    kwargs = dict(n_dimensions=2, random_seed=123, n_jobs=1, n_draws=128)
    expected = nested._run_importance_sampler(_prior, _likelihood, posterior, **kwargs)
    calls = []
    original = nested._LogLikelihoodBlock.__call__
    def observe(self, points):
        calls.extend(points.copy())
        return original(self, points)
    monkeypatch.setattr(nested._LogLikelihoodBlock, "__call__", observe)
    clock = iter([0.0] * 3 + [100.0] * 1000)
    monkeypatch.setattr(nested, "_monotonic", lambda: next(clock))
    with pytest.raises(nested.NestedCheckpointPaused):
        with nested.nested_checkpoint(tmp_path / "importance", deadline=50.0):
            nested._run_importance_sampler(_prior, _likelihood, posterior, **kwargs)
    assert 0 < len(calls) < expected.n_likelihood_calls
    monkeypatch.setattr(nested, "_monotonic", lambda: 0.0)
    with nested.nested_checkpoint(tmp_path / "importance", deadline=50.0):
        actual = nested._run_importance_sampler(_prior, _likelihood, posterior, **kwargs)
    assert actual == expected
    assert len(calls) == expected.n_likelihood_calls
    assert len(np.unique(np.asarray(calls), axis=0)) == len(calls)


@pytest.mark.parametrize("change", ["inputs", "payload", "missing-digest", "payload-shape", "payload-dtype"])
def test_wrong_or_corrupt_state_is_refused_before_the_native_pickle_reader(tmp_path, change):
    from trex.evidence.nested_checkpoint import SamplerCheckpoint

    recovery = SamplerCheckpoint(tmp_path / "state.dynesty", "input-a", 100.0)
    fake_sampler = SimpleNamespace(save=lambda path: Path(path).write_bytes(b"trusted state fixture"))
    recovery.save(fake_sampler, 42)
    if change == "inputs":
        recovery.identity = "input-b"
    else:
        with np.load(recovery.path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            payload = np.array(stored["payload"])
        if change == "payload":
            payload[0] ^= 1
        elif change == "missing-digest":
            del metadata["payload_sha256"]
        elif change == "payload-shape":
            payload = payload.reshape(1, -1)
        else:
            payload = payload.astype(np.int16)
        with recovery.path.open("wb") as stream:
            np.savez(stream, metadata=json.dumps(metadata), payload=payload)
    def refuse(*_args, **_kwargs):
        pytest.fail("unsafe native restore was reached")
    dynesty = SimpleNamespace(DynamicNestedSampler=SimpleNamespace(restore=refuse))
    with pytest.raises(ValueError, match="do not match|corrupt"):
        recovery.restore(dynesty, None, None)


def test_restored_counter_keeps_the_original_likelihood_cap(tmp_path, monkeypatch):
    controls = nested._SamplerControls(30, 0.05, 100, 123, 1)
    clock = iter([0.0] * 12 + [100.0] * 10_000)
    monkeypatch.setattr(nested, "_monotonic", lambda: next(clock))
    with pytest.raises(nested.NestedCheckpointPaused):
        with nested.nested_checkpoint(tmp_path / "limited", deadline=50.0):
            nested._run_nested_sampler(_prior, _likelihood, controls, "analytic", 2)
    monkeypatch.setattr(nested, "_monotonic", lambda: 0.0)
    with pytest.raises(RuntimeError, match="after 100 monitored evaluations"):
        with nested.nested_checkpoint(tmp_path / "limited", deadline=50.0):
            nested._run_nested_sampler(_prior, _likelihood, controls, "analytic", 2)


def test_native_importance_recovery_keeps_the_actual_scheduler_and_draw_order(tmp_path, monkeypatch):
    from trex.accelerators import _jax_models as models
    from trex.accelerators import _native_unit_batch
    from tests.test_trex_native_importance_integration import _inputs

    monkeypatch.setattr(nested, "_IMPORTANCE_EVALUATION_CHUNK", 32)
    monkeypatch.setattr(_native_unit_batch, "evaluation_block", lambda *_args: 64)
    kwargs = dict(n_dimensions=4, random_seed=42, n_jobs=1, n_draws=256)
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        integrand, posterior = _inputs()
        expected = nested._run_importance_sampler(
            integrand.prior_transform, integrand.log_likelihood, posterior,
            native_integrand=integrand, **kwargs,
        )
        # Use a fresh object: integrands cache the last transformed point.
        integrand, posterior = _inputs()
        clock = iter([0.0] * 3 + [100.0] * 1000)
        monkeypatch.setattr(nested, "_monotonic", lambda: next(clock))
        with pytest.raises(nested.NestedCheckpointPaused):
            with nested.nested_checkpoint(tmp_path / "native", deadline=50.0):
                nested._run_importance_sampler(
                    integrand.prior_transform, integrand.log_likelihood, posterior,
                    native_integrand=integrand, **kwargs,
                )
        integrand, posterior = _inputs()
        monkeypatch.setattr(nested, "_monotonic", lambda: 0.0)
        with nested.nested_checkpoint(tmp_path / "native", deadline=50.0):
            actual = nested._run_importance_sampler(
                integrand.prior_transform, integrand.log_likelihood, posterior,
                native_integrand=integrand, **kwargs,
            )
    assert actual._replace(execution_provenance=None) == expected._replace(execution_provenance=None)
    waves = actual.execution_provenance["recovery_waves"]
    assert len(waves) > 1
    assert sum(wave["logical_likelihood_calls"] for wave in waves) == actual.n_likelihood_calls
    assert all(row["platform"] == "cpu" for wave in waves for row in wave["device_dispatches"])
