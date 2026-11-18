"""Standalone operational admission and failure records, without evidence draws."""

import copy
import json
import logging
import time

import numpy as np
import pytest

from trex.app import engine
from trex.app.doctor import failure_diagnosis, group_failures, main, probe_mast
from trex.app.example import synthetic_inputs
from trex.app.target import target
from trex.evidence.marginal_likelihoods import _SCENARIO_NAMES
from trex.evidence.nested_checkpoint import NestedCheckpointPaused
from trex.foundation.errors import NumericalLikelihoodError, TrexEvidenceUnavailableError


@pytest.fixture
def inputs(tmp_path):
    return synthetic_inputs(tmp_path / "inputs")


def test_doctor_uses_actual_engine_admission_without_draws_or_mutation(inputs, monkeypatch):
    scene, binding, times, flux, sigma, period = inputs
    obj = target.from_scene(scene, **binding)
    before = copy.deepcopy(obj.__dict__)
    monkeypatch.setattr(engine, "calc_target_evidences", lambda **kw: pytest.fail("doctor drew evidence"))
    report = obj.doctor(times, flux, sigma, period)
    assert report["status"] == "ready" and report["evidence_draws"] == 0
    assert report["cadences"] == 80 and report["neighbors"] == 2
    assert obj.FPP is obj.NFPP is obj.failure_record is None
    assert obj.calculation_provenance == before["calculation_provenance"]


@pytest.mark.parametrize("defect", ["nan", "alignment", "sigma", "exposure", "exposure-type", "budget"])
def test_actual_calc_probs_and_doctor_refuse_the_same_bad_inputs(inputs, defect, tmp_path):
    scene, binding, times, flux, sigma, period = inputs
    options = {"N": 8, "exptime": binding["exptime_days"]}
    if defect == "nan":
        times[0] = np.nan
    elif defect == "alignment":
        flux = flux[:-1]
    elif defect == "sigma":
        sigma[:] = 0
    elif defect == "exposure":
        options["exptime"] *= 2
    elif defect == "exposure-type":
        options["exptime"] = True
    else:
        options["N"] = 0
    obj = target.from_scene(scene, **binding)
    report = obj.doctor(times, flux, sigma, period, **options)
    assert report["status"] == "failed" and report["category"] == "invalid-input"
    with pytest.raises(ValueError) as caught:
        obj.calc_probs(times, flux, sigma, period, **options)
    assert caught.value.trex_diagnosis["stage"] == report["stage"]
    assert obj.FPP is None and obj.failure_record["operational_diagnosis"]["category"] == "invalid-input"
    assert obj.failure_record["operational_diagnosis"]["stage"] == report["stage"]
    output = tmp_path / "refused"
    obj.write_outputs(output)
    record = json.loads((output / "failure.json").read_text())
    assert "FPP" not in record and "NFPP" not in record
    assert record["operational_diagnosis"]["next_action"]


def test_missing_capable_host_is_reported_before_any_rng_or_scenario(inputs, monkeypatch):
    scene, binding, times, flux, sigma, period = inputs
    scene.resolved_neighbors[0]["M_s"] = None
    scene.resolved_neighbors[0]["R_s"] = None
    scene.resolved_neighbors[0]["stellar_parameter_unavailable_reason"] = "synthetic missing host observations"
    obj = target.from_scene(scene, **binding)
    monkeypatch.setattr(engine, "calc_target_evidences", lambda **kw: pytest.fail("drew before admission"))
    report = obj.doctor(times, flux, sigma, period)
    assert report["status"] == "deferred" and report["category"] == "missing-evidence"
    assert report["diagnostics"]["input_requirements"]
    with pytest.raises(TrexEvidenceUnavailableError) as caught:
        obj.calc_probs(times, flux, sigma, period, exptime=binding["exptime_days"], N=8)
    assert caught.value.trex_diagnosis["stage"] == report["stage"]
    assert obj.failure_record["diagnostics"]["input_requirements"]


def test_engine_logs_budget_original_traceback_and_preserves_exception(inputs, caplog):
    scene, binding, times, flux, sigma, period = inputs
    times[0] = np.nan
    with caplog.at_level(logging.INFO, logger="trex"):
        with pytest.raises(ValueError) as caught:
            engine.run_trex_vetting(scene, times, flux, sigma, period, binding["depth_ppm"],
                                    binding["exptime_days"], sector_weights=binding["sector_weights"], n_draws=73)
    stopped = next(row for row in caplog.records if getattr(row, "event", "") == "trex.engine.stopped")
    assert stopped.n_draws == 73 and stopped.wall_seconds >= 0
    assert stopped.exc_info[1] is caught.value
    assert stopped.exc_info[2] is not None
    assert caught.value.trex_diagnosis["category"] == "invalid-input"
    assert "flux" not in stopped.__dict__ and "time" not in stopped.__dict__


def test_convergence_is_not_a_software_failure():
    error = TrexEvidenceUnavailableError("not converged", {"effective_sample_sizes": {"TP": 2.0}})
    diagnosis = failure_diagnosis(error, stage="evidence")
    assert diagnosis["status"] == "deferred" and diagnosis["category"] == "convergence"
    assert failure_diagnosis(NumericalLikelihoodError("solver broke"), stage="evidence")["category"] == "numerical-failure"


def test_grouping_keeps_distinct_errors_and_damaged_records_visible(tmp_path):
    paths = []
    for i, reason in enumerate(("missing host", "missing host", "different missing input")):
        path = tmp_path / f"failure-{i}.json"
        path.write_text(json.dumps({"operational_diagnosis": failure_diagnosis(ValueError(reason), stage="inputs")}))
        paths.append(path)
    damaged = tmp_path / "damaged.json"
    damaged.write_text("{broken")
    report = group_failures([*paths, paths[0], damaged])
    assert [row["affected_runs"] for row in report["groups"]] == [2, 1]
    assert report["unreadable_records"][0]["path"] == str(damaged)


def test_grouping_keeps_checkpoint_phases_distinct(tmp_path):
    def pause(phase):
        error = NestedCheckpointPaused("synthetic deadline")
        error.diagnostics = {phase: {"paused": True}}
        return failure_diagnosis(error, stage="full")

    paths = []
    for index, phase in enumerate(("mc_importance", "nested_escalation", "mc_importance")):
        path = tmp_path / f"pause-{index}.json"
        path.write_text(json.dumps({"operational_diagnosis": pause(phase)}))
        paths.append(path)
    report = group_failures(paths)
    assert [(row["phase"], row["affected_runs"]) for row in report["groups"]] == [
        ("mc_importance", 2), ("nested_escalation", 1)]


def test_cli_missing_prepared_input_is_a_nonzero_diagnosis(tmp_path, capsys):
    assert main([str(tmp_path / "absent")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["category"] == "invalid-input"


def test_mast_probe_hard_timeout_includes_child_startup():
    # No sleep or external request: a budget below spawn startup ends before the query.
    started = time.monotonic()
    report = probe_mast("123", timeout_seconds=1e-9)
    assert report["status"] == "deferred" and report["category"] == "timeout"
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, 61])
def test_mast_probe_rejects_unbounded_or_invalid_budgets(seconds):
    with pytest.raises(ValueError, match="timeout_seconds"):
        probe_mast("123", timeout_seconds=seconds)


def test_doctor_and_engine_share_admission_and_log_engine_lifecycle(inputs, monkeypatch, tmp_path, caplog):
    scene, binding, times, flux, sigma, period = inputs
    scenarios = len(_SCENARIO_NAMES)
    calls = []
    original = engine.prepare_vetting_inputs

    def recorded(*args, **kwargs):
        calls.append((np.asarray(args[1]).size, kwargs["n_draws"] if "n_draws" in kwargs else args[11]))
        return original(*args, **kwargs)

    def fake_calc(**kwargs):
        for index in range(scenarios):
            kwargs["ess_out"][index] = 1000.0
        return np.full(scenarios, -10.0), "ok"

    monkeypatch.setattr(engine, "prepare_vetting_inputs", recorded)
    monkeypatch.setattr(engine, "calc_target_evidences", fake_calc)
    obj = target.from_scene(scene, **binding)
    with caplog.at_level(logging.INFO, logger="trex"):
        report = obj.doctor(times, flux, sigma, period, N=16, nsamples=1)
        assert report["status"] == "ready" and report["stage"] == "input-admission"
        assert calls == [(80, 16)]
        result = engine.run_trex_vetting(
            scene, times, flux, sigma, period, binding["depth_ppm"], binding["exptime_days"],
            sector_weights=binding["sector_weights"], n_draws=16, nsamples=1,
            escalation_checkpoint_directory=str(tmp_path / "escalation"))
    assert calls == [(80, 16), (80, 16)] and len(result.probs) == scenarios
    engine_events = [row for row in caplog.records if row.name == "trex.engine"]
    events = [row.event for row in engine_events]
    assert events[0] == "trex.engine.start" and events[-1] == "trex.engine.finish"
    assert "trex.engine.admission" in events and "trex.engine.phase" in events
    admission = next(row for row in engine_events if row.event == "trex.engine.admission")
    assert admission.n_draws == 16 and admission.phase == "input-admission" and admission.wall_seconds >= 0
    phase = next(row for row in engine_events if row.event == "trex.engine.phase")
    assert phase.phase == "monte-carlo" and phase.wall_seconds >= 0
    finish = next(row for row in engine_events if row.event == "trex.engine.finish")
    assert finish.status == result.status and finish.FPP == result.fpp and finish.wall_seconds >= 0


def test_execution_logging_preserves_engine_numerics(inputs, monkeypatch, tmp_path):
    scene, binding, times, flux, sigma, period = inputs
    scenarios = len(_SCENARIO_NAMES)

    def fake_calc(**kwargs):
        for index in range(scenarios):
            kwargs["ess_out"][index] = 1000.0
        return np.full(scenarios, -10.0), "ok"

    monkeypatch.setattr(engine, "calc_target_evidences", fake_calc)
    arguments = (scene, times, flux, sigma, period, binding["depth_ppm"], binding["exptime_days"])
    options = dict(sector_weights=binding["sector_weights"], n_draws=16, nsamples=1)
    decorated = engine.run_trex_vetting(
        *arguments, **options, escalation_checkpoint_directory=str(tmp_path / "decorated"))
    plain = engine.run_trex_vetting.__wrapped__(
        *arguments, **options, escalation_checkpoint_directory=str(tmp_path / "plain"))
    assert decorated.status == plain.status and decorated.fpp == plain.fpp and decorated.nfpp == plain.nfpp
    np.testing.assert_array_equal(decorated.probs, plain.probs)
    assert decorated.effective_sample_sizes == plain.effective_sample_sizes


def test_doctor_does_not_mutate_retained_results_or_failure_records(inputs):
    scene, binding, times, flux, sigma, period = inputs
    obj = target.from_scene(scene, **binding)
    marker = object()
    obj.trex_result, obj.FPP, obj.NFPP = marker, 0.42, 0.01
    obj.failure_record = {"retained": True}
    obj.calculation_provenance = {"retained": True}
    report = obj.doctor(times, flux, sigma, period)
    assert report["status"] == "ready"
    assert (obj.trex_result is marker and obj.FPP == 0.42 and obj.NFPP == 0.01
            and obj.failure_record == {"retained": True}
            and obj.calculation_provenance == {"retained": True})
    # A changed binding is diagnosed, but the doctor still clears no retained result.
    obj.stars = obj.stars.assign(Tmag=obj.stars["Tmag"] + 1.0)
    report = obj.doctor(times, flux, sigma, period)
    assert report["status"] == "failed" and report["category"] == "invalid-input"
    assert report["stage"] == "target-binding"
    assert (obj.trex_result is marker and obj.FPP == 0.42 and obj.NFPP == 0.01
            and obj.failure_record == {"retained": True})


def test_early_calc_probs_refusal_retains_an_actionable_writable_diagnosis(inputs, tmp_path):
    scene, binding, times, flux, sigma, period = inputs
    obj = target.from_scene(scene, **binding)
    with pytest.raises(NotImplementedError) as caught:
        obj.calc_probs(times, flux, sigma, period, exptime=binding["exptime_days"], filt="J")
    diagnosis = caught.value.trex_diagnosis
    assert diagnosis["status"] == "failed" and diagnosis["category"] == "invalid-input"
    assert diagnosis["stage"] == "target-api" and diagnosis["next_action"]
    assert obj.failure_record["operational_diagnosis"] == diagnosis
    obj.write_outputs(tmp_path / "refused")
    payload = json.loads((tmp_path / "refused" / "failure.json").read_text())
    assert payload["operational_diagnosis"]["category"] == "invalid-input"
    assert payload["operational_diagnosis"]["next_action"]
    assert "FPP" not in payload and "NFPP" not in payload


def test_engine_preserves_the_admission_exception_object(inputs, monkeypatch, tmp_path, caplog):
    scene, binding, times, flux, sigma, period = inputs
    error = ValueError("synthetic admission refusal")

    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(engine, "admitted_photometry", refuse)
    with caplog.at_level(logging.INFO, logger="trex"):
        with pytest.raises(ValueError) as caught:
            engine.run_trex_vetting(
                scene, times, flux, sigma, period, binding["depth_ppm"], binding["exptime_days"],
                sector_weights=binding["sector_weights"], n_draws=16, nsamples=1,
                escalation_checkpoint_directory=str(tmp_path / "escalation"))
    assert caught.value is error
    assert error.trex_diagnosis["stage"] == "input-admission"
    assert error.trex_diagnosis["category"] == "invalid-input"
    stopped = next(row for row in caplog.records if getattr(row, "event", None) == "trex.engine.stopped")
    assert stopped.stage == "full" and stopped.phase == "input-admission"


def test_checkpoint_pause_is_retained_as_a_writable_deferred_diagnosis(inputs, monkeypatch, tmp_path):
    scene, binding, times, flux, sigma, period = inputs
    scenarios = len(_SCENARIO_NAMES)
    ess = {index: 2.0 for index in range(scenarios)}
    lnZ = np.full(scenarios, -10.0)

    def fake_calc(**kwargs):
        kwargs["ess_out"].update(ess)
        return lnZ, "ok"

    def fake_importance(**kwargs):
        return {"lnZ": lnZ, "ess": dict(ess), "records": [], "attempts": {}}

    def pause(**kwargs):
        raise NestedCheckpointPaused("synthetic session deadline")

    monkeypatch.setattr(engine, "calc_target_evidences", fake_calc)
    monkeypatch.setattr(engine, "_run_mc_importance", fake_importance)
    monkeypatch.setattr(engine, "run_nested_escalation", pause)
    obj = target.from_scene(scene, **binding)
    with pytest.raises(NestedCheckpointPaused) as caught:
        obj.calc_probs(times, flux, sigma, period, exptime=binding["exptime_days"], N=8, nsamples=1,
                       escalation_checkpoint_directory=str(tmp_path / "checkpoint"))
    assert caught.value.trex_diagnosis["status"] == "deferred"
    assert caught.value.trex_diagnosis["category"] == "paused"
    assert caught.value.trex_diagnosis["phase"] == "nested_escalation"
    assert obj.failure_record["operational_diagnosis"]["category"] == "paused"
    assert obj.failure_record["diagnostics"]["nested_escalation"]["paused"] is True
    obj.write_outputs(tmp_path / "paused")
    payload = json.loads((tmp_path / "paused" / "failure.json").read_text())
    assert payload["operational_diagnosis"]["category"] == "paused"
    assert "FPP" not in payload and "NFPP" not in payload


def test_numerical_failure_is_not_reported_as_missing_evidence_or_convergence():
    error = TrexEvidenceUnavailableError(
        "aggregate evidence gate",
        {"effective_sample_sizes": {"TP": 2.0}, "numerical_failures": {"TP": {"count": 2}}},
    )
    diagnosis = failure_diagnosis(error, stage="evidence")
    assert (diagnosis["status"], diagnosis["category"]) == ("failed", "numerical-failure")
    missing = failure_diagnosis(
        TrexEvidenceUnavailableError("missing host", {"input_requirements": [{"source_id": "1"}]}),
        stage="input-admission",
    )
    assert (missing["status"], missing["category"]) == ("deferred", "missing-evidence")


def test_numerical_engine_failure_is_retained_and_writable(inputs, monkeypatch, tmp_path):
    scene, binding, times, flux, sigma, period = inputs
    error = NumericalLikelihoodError("synthetic supported likelihood failure", count=2)

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(engine, "calc_target_evidences", fail)
    obj = target.from_scene(scene, **binding)
    with pytest.raises(NumericalLikelihoodError) as caught:
        obj.calc_probs(times, flux, sigma, period, exptime=binding["exptime_days"], N=16, nsamples=1,
                       escalation_checkpoint_directory=str(tmp_path / "checkpoint"))
    assert caught.value is error
    assert caught.value.trex_diagnosis["category"] == "numerical-failure"
    assert caught.value.trex_diagnosis["status"] == "failed"
    assert obj.failure_record["operational_diagnosis"]["category"] == "numerical-failure"
    obj.write_outputs(tmp_path / "numerical")
    payload = json.loads((tmp_path / "numerical" / "failure.json").read_text())
    assert payload["operational_diagnosis"]["category"] == "numerical-failure"
    assert "FPP" not in payload and "NFPP" not in payload

