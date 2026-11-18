"""Runner lifecycle: input refusal, execution failure, deferral, pause and resume."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trex.app import run
from trex.app.doctor import failure_diagnosis
from trex.app.example import synthetic_inputs
from trex.evidence.nested_checkpoint import NestedCheckpointPaused
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.runtime import logging as trex_logging


def prepared_inputs(tmp_path):
    """Write the three portable input files from the synthetic demonstration."""
    scene, binding, time, flux, sigma, period = synthetic_inputs(tmp_path / "synthetic")
    directory = tmp_path / "prepared"
    directory.mkdir()
    (directory / "background.csv").write_bytes((tmp_path / "synthetic" / "synthetic-trilegal.csv").read_bytes())
    payload = {
        "scene": scene.to_dict(),
        "period_days": period,
        "binding": {key: value for key, value in binding.items()
                    if key not in {"input_artifacts", "bin_integration_days"}},
    }
    (directory / "trex-input.json").write_text(json.dumps(payload), encoding="utf-8")
    np.savez(directory / "arrays.npz", time=time, flux=flux, sigma=sigma,
             bin_integration_days=np.asarray(binding["bin_integration_days"]))
    return directory, time, flux, sigma, period, binding


def install(monkeypatch, tmp_path, *, calc=None):
    """Replace load_input with a target double and the calculated controls with inert arrays."""
    digests = [hashlib.sha256(name.encode("utf-8")).hexdigest() for name in run.INPUT_FILES]
    artifacts = dict(zip((str((tmp_path / "inputs" / name).absolute()) for name in run.INPUT_FILES),
                         digests, strict=True))
    bound = SimpleNamespace(failure_record=None, trex_result=None, FPP=None, NFPP=None,
                            written=[], calls=[])
    bound._require_scene = lambda: {"input_artifacts": artifacts}

    def write_outputs(directory, *, existing_empty=False):
        output = Path(directory)
        output.mkdir(exist_ok=existing_empty)
        if existing_empty and (output.is_symlink() or any(output.iterdir())):
            raise FileExistsError(f"output directory is not empty: {output}")
        bound.written.append(output)
        if bound.failure_record is not None:
            (output / "failure.json").write_text(json.dumps(bound.failure_record), encoding="utf-8")
        else:
            (output / "summary.json").write_text(json.dumps({
                "status": bound.trex_result.status, "FPP": bound.FPP, "NFPP": bound.NFPP,
            }), encoding="utf-8")

    def calc_probs(*args, **kwargs):
        bound.calls.append(kwargs)
        return (calc or (lambda *ignored, **ignored_controls: None))(bound, *args, **kwargs)

    bound.write_outputs = write_outputs
    bound.calc_probs = calc_probs
    observed = {"time": np.zeros(4), "flux": np.ones(4), "sigma": np.full(4, 1e-3)}
    monkeypatch.setattr(run, "load_input", lambda directory: (bound, observed, 3.2, 120.0 / 86400.0))
    monkeypatch.delenv("TREX_ESCALATION_DEADLINE_MONOTONIC", raising=False)
    return bound


def runner_args(tmp_path, output, checkpoint=None, *extra):
    arguments = [str(tmp_path / "inputs"), str(output)]
    if checkpoint is not None:
        arguments += ["--escalation-checkpoint-directory", str(checkpoint)]
    return arguments + list(extra)


def read_failure(output):
    return json.loads((output / "failure.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative", [False, True])
def test_load_input_rebuilds_the_bound_target_and_hashes_the_three_files(tmp_path, monkeypatch, relative):
    directory, time, flux, sigma, period, binding = prepared_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    bound, observed, restored_period, exposure = run.load_input(directory.name if relative else directory)
    assert bound.ID == 100001
    assert restored_period == pytest.approx(period)
    assert exposure == pytest.approx(binding["exptime_days"])
    np.testing.assert_array_equal(observed["time"], time)
    np.testing.assert_array_equal(observed["flux"], flux)
    np.testing.assert_array_equal(observed["sigma"], sigma)
    assert float(observed["bin_integration_days"]) == pytest.approx(binding["bin_integration_days"])
    artifacts = bound._require_scene()["input_artifacts"]
    assert list(artifacts) == [str(directory / name) for name in run.INPUT_FILES]
    for name in run.INPUT_FILES:
        assert artifacts[str(directory / name)] == hashlib.sha256((directory / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("missing", ["time", "flux", "sigma", "bin_integration_days"])
def test_missing_required_array_is_an_input_refusal(tmp_path, missing):
    directory, *_ = prepared_inputs(tmp_path)
    path = directory / "arrays.npz"
    with np.load(path, allow_pickle=False) as arrays:
        retained = {name: arrays[name] for name in arrays.files if name != missing}
    np.savez(path, **retained)
    output = tmp_path / "refused"
    assert run.main([str(directory), str(output), "--draws", "4096"]) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["error_type"] == "ValueError" and missing in record["reason"]
    assert record["operational_diagnosis"]["stage"] == "input-loading"
    assert "FPP" not in record and "NFPP" not in record
    assert not output.with_name(output.name + ".escalation").exists()


@pytest.mark.parametrize("name", run.INPUT_FILES)
@pytest.mark.parametrize("defect", ["missing", "directory"])
def test_required_input_path_errors_are_retained_before_inference(tmp_path, name, defect):
    directory, *_ = prepared_inputs(tmp_path)
    path = directory / name
    path.rename(path.with_name(name + ".original"))
    if defect == "directory":
        path.mkdir()
    output = tmp_path / "refused"
    assert run.main([str(directory), str(output)]) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["error_type"] == ("FileNotFoundError" if defect == "missing" else "ValueError")
    assert record["operational_diagnosis"]["stage"] == "input-loading"
    assert not output.with_name(output.name + ".escalation").exists()
    if defect == "missing":
        assert str(path) in record["reason"]


@pytest.mark.parametrize("name", [None, *run.INPUT_FILES])
def test_absolute_input_binding_does_not_follow_a_symlinked_input(tmp_path, name):
    directory, *_ = prepared_inputs(tmp_path)
    if name is None:
        alias = tmp_path / "linked-input"
        alias.symlink_to(directory, target_is_directory=True)
    else:
        path = directory / name
        retained = path.rename(path.with_name(name + ".original"))
        path.symlink_to(retained)
        alias = directory
    with pytest.raises(ValueError, match="non-symlink"):
        run.load_input(alias)


@pytest.mark.slow
def test_prepared_run_example_executes_with_a_relative_input(tmp_path, record_property):
    """Exercise the documented synthetic file/CLI path, not FPP calibration."""
    directory, *_ = prepared_inputs(tmp_path)
    output = tmp_path / "demo-output"
    environment = {**os.environ, "TREX_NESTED_ESTIMATOR": "dynesty"}
    environment.pop("TREX_ESCALATION_DEADLINE_MONOTONIC", None)
    completed = subprocess.run(
        [sys.executable, "-m", "trex.run", "--no-log-file", directory.name, output.name, "--draws", "4096"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300)
    assert completed.returncode in (run.EXIT_COMPLETED, run.EXIT_DEFERRED), completed.stderr[-12000:]
    results = [output / name for name in ("summary.json", "failure.json") if (output / name).is_file()]
    assert len(results) == 1
    payload = json.loads(results[0].read_text())
    record_property("prepared_run", json.dumps({
        "draws": 4096, "exit_code": completed.returncode,
        "status": payload.get("status"), "operational_diagnosis": payload.get("operational_diagnosis"),
        "FPP": payload.get("FPP"), "NFPP": payload.get("NFPP"),
    }))
    if completed.returncode == run.EXIT_COMPLETED:
        assert payload["status"] == "ok"
        assert all(0.0 <= payload[key] <= 1.0 for key in ("FPP", "NFPP"))
    else:
        assert payload.get("status") != "ok"
        if payload.get("operational_diagnosis"):
            assert payload["operational_diagnosis"]["status"] == "deferred"
        assert payload.get("FPP") is None and payload.get("NFPP") is None


def test_input_refusal_records_the_original_exception_without_probabilities(tmp_path, monkeypatch, capsys):
    error = ValueError("scene is missing its measured metallicity")

    def fail(directory):
        raise error

    monkeypatch.setattr(run, "load_input", fail)
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["error_type"] == "ValueError"
    assert record["reason"] == str(error)
    assert record["operational_diagnosis"]["stage"] == "input-loading"
    assert record["operational_diagnosis"]["category"] == "invalid-input"
    assert "FPP" not in record and "NFPP" not in record
    assert [path.name for path in output.iterdir()] == ["failure.json"]
    assert "input-loading" in capsys.readouterr().err


def test_input_refusal_keeps_diagnostics_and_nonfinite_values_visible(tmp_path, monkeypatch):
    class Refusal(ValueError):
        diagnostics = {"effective_sample_sizes": {"TP": float("nan")}, "draws": np.int64(3)}

    def fail(directory):
        raise Refusal("required evidence is missing")

    monkeypatch.setattr(run, "load_input", fail)
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["diagnostics"]["effective_sample_sizes"]["TP"] == "nan"
    assert record["diagnostics"]["draws"] == 3
    assert "FPP" not in record and "NFPP" not in record


@pytest.mark.parametrize("stop", ["input-refusal", "calculation-refusal", "deferral", "deferral-without-record"])
@pytest.mark.parametrize("publication", ["creation", "serialization", "replace"])
def test_required_failure_publication_errors_propagate_with_original_cause(tmp_path, monkeypatch, stop, publication):
    original = (TrexEvidenceUnavailableError("required evidence is missing", {}) if stop.startswith("deferral")
                else ValueError("invalid measured input"))

    def calc(bound, *args, **kwargs):
        if stop != "deferral-without-record":
            bound.failure_record = {"error_type": type(original).__name__, "reason": str(original),
                                    "operational_diagnosis": failure_diagnosis(original, stage="target-api")}
        raise original

    bound = install(monkeypatch, tmp_path, calc=calc)
    # Exercise the stable target writer, not the install() double's serialization.
    bound.write_outputs = lambda directory: run.target.write_outputs(bound, directory)
    if stop == "input-refusal":
        def fail_input(directory):
            raise original

        monkeypatch.setattr(run, "load_input", fail_input)
    output = tmp_path / "result"
    problem = (TypeError("injected failure serialization error") if publication == "serialization"
               else OSError(f"injected failure {publication} error"))
    mkdir, replace, dumps = Path.mkdir, Path.replace, json.dumps

    def fail_mkdir(path, *args, **kwargs):
        if path == output:
            raise problem
        return mkdir(path, *args, **kwargs)

    def fail_replace(path, *args, **kwargs):
        if path.parent == output:
            raise problem
        return replace(path, *args, **kwargs)

    def fail_serialization(payload, *args, **kwargs):
        if isinstance(payload, dict) and payload.get("reason") == str(original):
            raise problem
        return dumps(payload, *args, **kwargs)

    if publication == "creation":
        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    elif publication == "replace":
        monkeypatch.setattr(Path, "replace", fail_replace)
    else:
        monkeypatch.setattr(json, "dumps", fail_serialization)

    with pytest.raises(type(problem)) as caught:
        run.main(runner_args(tmp_path, output))
    assert caught.value is problem
    assert caught.value.__cause__ is original
    assert original.__traceback__ is not None
    assert not (output / "failure.json").exists()
    assert not (output / "summary.json").exists()
    if publication == "creation":
        assert not output.exists()
    else:
        assert output.is_dir() and not any(output.iterdir())


def test_failure_record_atomic_publication_error_propagates_and_cleans_temporary(tmp_path, monkeypatch):
    original = FileNotFoundError("prepared input is missing")
    problem = OSError("injected failure record replacement error")

    def fail_input(directory):
        raise original

    def fail_replace(source, destination):
        assert Path(source).read_text(encoding="utf-8")
        raise problem

    monkeypatch.setattr(run, "load_input", fail_input)
    monkeypatch.setattr(run.os, "replace", fail_replace)
    output = tmp_path / "result"
    with pytest.raises(OSError) as caught:
        run.main(runner_args(tmp_path, output))
    assert caught.value is problem and caught.value.__cause__ is original
    assert not any(output.iterdir())


def test_failure_publication_collision_protects_existing_output(tmp_path, monkeypatch):
    output = tmp_path / "result"
    output.mkdir()
    retained = output / "failure.json"
    retained.write_bytes(b"retained failure record\n")

    def unexpected_record(*args, **kwargs):
        pytest.fail("protected output must not prepare or write a new record")

    monkeypatch.setattr(run, "_failure_record", unexpected_record)
    record, reason = run._write_failure(output, ValueError("later refusal"), stage="input-loading")
    assert record is None and "already exists" in reason
    assert retained.read_bytes() == b"retained failure record\n"
    assert list(output.iterdir()) == [retained]


@pytest.mark.parametrize("publication", ["creation", "replace"])
def test_standalone_failure_publication_error_exits_one_with_chained_cause(tmp_path, publication):
    output = tmp_path / "output"
    script = """
import runpy
import sys
from pathlib import Path

publication, destination = sys.argv[1:3]
output = Path(destination)
operation = 'mkdir' if publication == 'creation' else 'replace'
original = getattr(Path, operation)

def fail(path, *args, **kwargs):
    if path == output or path.parent == output:
        raise OSError('injected required record publication failure')
    return original(path, *args, **kwargs)

setattr(Path, operation, fail)
sys.argv = ['trex.run', '--no-log-file', str(output.parent / 'missing-input'), str(output)]
runpy.run_module('trex.run', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, publication, str(output)],
        capture_output=True, text=True, check=False, timeout=60,
        env={**os.environ, "XDG_STATE_HOME": str(tmp_path / "state")},
    )
    assert result.returncode == 1, result.stderr
    rows = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    event, = [row for row in rows if row.get("event") == "trex.cli.stopped"]
    assert event["exception"]["type"] == "OSError"
    cause = event["exception"]["cause"]
    assert cause["type"] == "FileNotFoundError"
    assert "load_input" in [frame["function"] for frame in cause["frames"]]
    assert rows[-1]["fields"]["exit_code"] == 1
    assert rows[-1]["fields"]["status"] == "failed"
    assert not (output / "failure.json").exists()
    assert not (output / "summary.json").exists()
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("defect", ["missing-input", "malformed-checkpoint"])
def test_caught_refusal_logs_original_traceback_without_input_payload(tmp_path, monkeypatch, caplog, defect):
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"
    if defect == "malformed-checkpoint":
        install(monkeypatch, tmp_path)
        checkpoint.mkdir(mode=0o700)
        handoff = checkpoint / "monte-carlo.json"
        handoff.write_text("private-catalog-payload", encoding="utf-8")
        retained = handoff.read_bytes()
        expected_type, stage, origin = json.JSONDecodeError, "checkpoint", "raw_decode"
    else:
        expected_type, stage, origin = FileNotFoundError, "input-loading", "load_input"

    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    event, = [row for row in caplog.records if getattr(row, "event", None) == "trex.run.refused"]
    assert event.stage == stage and event.exit_code == 2
    assert event.exc_info[0] is expected_type
    assert event.exc_info[2] is event.exc_info[1].__traceback__
    formatted = json.loads(trex_logging._JsonFormatter("refusal-test").format(event))
    assert formatted["exception"]["type"] == expected_type.__name__
    assert origin in [frame["function"] for frame in formatted["exception"]["frames"]]
    assert "private-catalog-payload" not in json.dumps(formatted)
    assert read_failure(output)["error_type"] == expected_type.__name__
    if defect == "malformed-checkpoint":
        assert handoff.read_bytes() == retained


@pytest.mark.parametrize("no_file", [False, True])
def test_standalone_refusal_retains_traceback_with_and_without_log_file(tmp_path, no_file):
    log_root = tmp_path / "logs"
    options = ["--no-log-file"] if no_file else ["--log-dir", str(log_root)]
    result = subprocess.run(
        [sys.executable, "-m", "trex.run", *options, str(tmp_path / "missing"), str(tmp_path / "output")],
        capture_output=True, text=True, check=False, timeout=60,
        env={**os.environ, "XDG_STATE_HOME": str(tmp_path / "state")},
    )
    assert result.returncode == 2 and result.stdout == ""
    rows = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    event, = [row for row in rows if row.get("event") == "trex.run.refused"]
    assert event["fields"]["stage"] == "input-loading"
    assert event["exception"]["type"] == "FileNotFoundError"
    assert "load_input" in [frame["function"] for frame in event["exception"]["frames"]]
    assert rows[-1]["fields"]["exit_code"] == 2
    assert read_failure(tmp_path / "output")["error_type"] == "FileNotFoundError"
    if no_file:
        assert not log_root.exists() and not (tmp_path / "state").exists()
    else:
        path, = log_root.glob("*/*.jsonl")
        retained = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert event in retained


def test_repeated_invocation_explains_a_published_failure_and_keeps_it(tmp_path, monkeypatch, capsys):
    def fail(directory):
        raise ValueError("broken input")

    monkeypatch.setattr(run, "load_input", fail)
    output = tmp_path / "result"
    arguments = runner_args(tmp_path, output)
    assert run.main(arguments) == run.EXIT_REFUSED
    retained = (output / "failure.json").read_bytes()
    capsys.readouterr()
    monkeypatch.setattr(run, "load_input", lambda directory: pytest.fail("must refuse before loading"))
    assert run.main(arguments) == run.EXIT_REFUSED
    assert (output / "failure.json").read_bytes() == retained
    assert [path.name for path in output.iterdir()] == ["failure.json"]
    message = capsys.readouterr().err
    assert "failure.json" in message and "never overwritten" in message


def test_input_refusal_leaves_a_resumable_empty_output_for_continuation(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    handoff = checkpoint / "monte-carlo.json"
    handoff.write_text('{"binding": {}, "state": {"random_seed": 3}}\n', encoding="utf-8")
    output = tmp_path / "result"
    output.mkdir()

    def fail(directory):
        raise FileNotFoundError("trex-input.json is missing")

    monkeypatch.setattr(run, "load_input", fail)
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    assert not any(output.iterdir())
    assert handoff.read_text(encoding="utf-8") == '{"binding": {}, "state": {"random_seed": 3}}\n'
    assert "left untouched" in capsys.readouterr().err


@pytest.mark.parametrize("defect", ["symlink", "permissions"])
def test_checkpoint_refusals_are_published_for_a_new_output(tmp_path, monkeypatch, defect):
    install(monkeypatch, tmp_path)
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"
    if defect == "symlink":
        private = tmp_path / "private"
        private.mkdir(mode=0o700)
        checkpoint.symlink_to(private)
        expected = "symlink"
    else:
        checkpoint.mkdir()
        checkpoint.chmod(0o755)
        expected = "inaccessible to others"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert expected in record["reason"]
    assert "FPP" not in record


def test_checkpoint_inside_the_output_is_refused_and_recorded(tmp_path):
    output = tmp_path / "result"
    checkpoint = output / "escalation"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert "outside the output" in record["reason"]
    assert not checkpoint.exists()


def test_checkpoint_without_a_handoff_explains_that_it_cannot_resume(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    stray = checkpoint / "stray.bin"
    stray.write_bytes(b"retained")
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert "no Monte Carlo handoff" in record["reason"]
    assert stray.read_bytes() == b"retained"


def test_changed_controls_refuse_the_retained_handoff_without_touching_it(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    handoff = checkpoint / "monte-carlo.json"
    handoff.write_text(json.dumps({"binding": {"draws": 1}, "state": {"random_seed": 4}}), encoding="utf-8")
    retained = handoff.read_bytes()
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert "changed" in record["reason"] and "draws" in record["reason"]
    assert handoff.read_bytes() == retained


def test_handoff_binding_binds_a_non_default_native_policy(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    monkeypatch.setenv("TREX_BLACKJAX_PRECOND", "split")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    handoff = checkpoint / "monte-carlo.json"
    handoff.write_text(json.dumps({"binding": {"draws": 1}, "state": {"random_seed": 4}}), encoding="utf-8")
    retained = handoff.read_bytes()
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert "blackjax_policy" in record["reason"]
    assert handoff.read_bytes() == retained


def test_malformed_handoff_is_refused_with_its_damaged_location(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    handoff = checkpoint / "monte-carlo.json"
    handoff.write_text('{"binding": {}, "state": "not a mapping"}\n', encoding="utf-8")
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["operational_diagnosis"]["stage"] == "checkpoint"
    assert "malformed" in record["reason"]
    assert handoff.read_text(encoding="utf-8") == '{"binding": {}, "state": "not a mapping"}\n'


def test_existing_output_is_never_overwritten_or_reinterpreted(tmp_path, monkeypatch, capsys):
    output = tmp_path / "result"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text('{"status": "ok", "FPP": 0.1}\n', encoding="utf-8")
    monkeypatch.setattr(run, "load_input", lambda directory: pytest.fail("must refuse before loading"))
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_REFUSED
    assert summary.read_text(encoding="utf-8") == '{"status": "ok", "FPP": 0.1}\n'
    assert [path.name for path in output.iterdir()] == ["summary.json"]
    assert "never overwritten" in capsys.readouterr().err


def test_output_path_without_a_name_is_refused_without_touching_its_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    assert run.main(["missing-input", "."]) == run.EXIT_REFUSED
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


def test_checkpoint_pause_creates_no_output_and_resume_reuses_handoff(tmp_path, monkeypatch):
    def pause(bound, *args, **kwargs):
        kwargs["escalation_state_out"].update({"random_seed": 424242, "session_remaining_seconds": 0.0})

    bound = install(monkeypatch, tmp_path, calc=pause)
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"
    arguments = runner_args(tmp_path, output, checkpoint, "--draws", "4096", "--nsamples", "4")
    assert run.main(arguments) == run.EXIT_PAUSED
    assert not output.exists()
    retained = (checkpoint / "monte-carlo.json").read_bytes()
    assert json.loads(retained)["state"] == {"random_seed": 424242}
    assert [path.name for path in checkpoint.iterdir()] == ["monte-carlo.json"]
    output.mkdir()  # A retained empty output is accepted only with this handoff.

    def complete(restarted, *args, **kwargs):
        assert kwargs["escalation_state_in"] == {"random_seed": 424242}
        assert kwargs["escalation_state_out"] is None
        restarted.trex_result = SimpleNamespace(status="ok")
        restarted.FPP, restarted.NFPP = 0.25, 0.5

    bound.calc_probs = lambda *args, **kwargs: complete(bound, *args, **kwargs)
    assert run.main(arguments) == run.EXIT_COMPLETED
    assert (output / "summary.json").is_file()
    assert (checkpoint / "monte-carlo.json").read_bytes() == retained


def test_pause_in_nested_stage_keeps_the_handoff_and_creates_no_output(tmp_path, monkeypatch):
    def calc(bound, *args, **kwargs):
        if kwargs.get("escalation_state_out") is not None:
            kwargs["escalation_state_out"].update({"random_seed": 9, "session_remaining_seconds": 60.0})
            return
        raise NestedCheckpointPaused("nested wall deadline reached")

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"
    assert run.main(runner_args(tmp_path, output, checkpoint)) == run.EXIT_PAUSED
    assert not output.exists()
    assert json.loads((checkpoint / "monte-carlo.json").read_text())["state"] == {"random_seed": 9}


def test_execution_failure_retains_the_target_record_and_reraises_the_original(tmp_path, monkeypatch):
    failure = RuntimeError("supported likelihood failed")

    def calc(bound, *args, **kwargs):
        bound.failure_record = {"claim_eligible": False, "error_type": "RuntimeError",
                                "reason": str(failure), "diagnostics": {"scenario": "TP"}}
        raise failure

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    with pytest.raises(RuntimeError) as caught:
        run.main(runner_args(tmp_path, output))
    assert caught.value is failure
    record = read_failure(output)
    assert record["diagnostics"] == {"scenario": "TP"}
    assert "FPP" not in record
    assert not (output / "summary.json").exists()


def test_calculation_input_refusal_returns_two_with_its_record(tmp_path, monkeypatch, capsys, caplog):
    error = ValueError("sigma must be positive")

    def calc(bound, *args, **kwargs):
        bound.failure_record = {"claim_eligible": False, "error_type": "ValueError",
                                "reason": str(error),
                                "operational_diagnosis": failure_diagnosis(error, stage="target-api")}
        raise error

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_REFUSED
    record = read_failure(output)
    assert record["error_type"] == "ValueError"
    assert record["operational_diagnosis"]["category"] == "invalid-input"
    assert "FPP" not in record
    assert "refused (input)" in capsys.readouterr().err
    event, = [row for row in caplog.records if getattr(row, "event", None) == "trex.run.refused"]
    assert event.stage == "target-api" and event.exit_code == 2
    assert event.exc_info[1] is error
    assert "calc" in [frame["function"] for frame in
                      json.loads(trex_logging._JsonFormatter("refusal-test").format(event))["exception"]["frames"]]


def test_handoff_publication_failure_is_recorded_and_leaves_no_partial_handoff(tmp_path, monkeypatch):
    def calc(bound, *args, **kwargs):
        kwargs["escalation_state_out"].update({"random_seed": 5, "session_remaining_seconds": 60.0})

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"

    def fail(*args, **kwargs):
        raise OSError("handoff link failed")

    monkeypatch.setattr(run, "_publish_handoff", fail)
    with pytest.raises(OSError, match="handoff link failed"):
        run.main(runner_args(tmp_path, output, checkpoint))
    record = read_failure(output)
    assert record["error_type"] == "OSError"
    assert record["operational_diagnosis"]["stage"] == "execution"
    assert not (checkpoint / "monte-carlo.json").exists()
    assert [path.name for path in checkpoint.iterdir()] == []


def test_scientific_deferral_returns_three_without_probabilities(tmp_path, monkeypatch, capsys, caplog):
    error = TrexEvidenceUnavailableError("no capable host", {"input_requirements": [{"source_id": "100002"}]})

    def calc(bound, *args, **kwargs):
        bound.failure_record = {"claim_eligible": False, "error_type": type(error).__name__,
                                "reason": str(error), "diagnostics": dict(error.diagnostics),
                                "operational_diagnosis": failure_diagnosis(error, stage="target-api")}
        raise error

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_DEFERRED
    record = read_failure(output)
    assert record["operational_diagnosis"]["status"] == "deferred"
    assert record["diagnostics"]["input_requirements"][0]["source_id"] == "100002"
    assert "FPP" not in record
    assert "scientific deferral" in capsys.readouterr().err
    event, = [row for row in caplog.records if getattr(row, "event", None) == "trex.run.deferred"]
    assert event.stage == "evidence" and event.exit_code == 3
    assert event.exc_info[1] is error


@pytest.mark.parametrize("stop", ["refusal", "deferral", "execution-failure"])
def test_calculation_failure_leaves_empty_resumable_output_untouched(tmp_path, monkeypatch, capsys, stop):
    original = (TrexEvidenceUnavailableError("required evidence is missing", {}) if stop == "deferral"
                else ValueError("invalid measured input") if stop == "refusal"
                else RuntimeError("execution failed"))

    def calc(bound, *args, **kwargs):
        bound.failure_record = {"error_type": type(original).__name__, "reason": str(original),
                                "operational_diagnosis": failure_diagnosis(original, stage="target-api")}
        raise original

    bound = install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    checkpoint = tmp_path / "checkpoint"
    arguments = runner_args(tmp_path, output, checkpoint)

    def pause(*args, **kwargs):
        kwargs["escalation_state_out"].update({"random_seed": 7, "session_remaining_seconds": 0.0})

    bound.calc_probs = pause
    assert run.main(arguments) == run.EXIT_PAUSED
    handoff = checkpoint / "monte-carlo.json"
    retained = handoff.read_bytes()
    output.mkdir()
    bound.calc_probs = lambda *args, **kwargs: calc(bound, *args, **kwargs)

    def unexpected_write(*args, **kwargs):
        pytest.fail("a failure must not publish into a protected resumable output")

    bound.write_outputs = unexpected_write
    monkeypatch.setattr(run, "_write_failure", unexpected_write)
    if stop == "execution-failure":
        with pytest.raises(RuntimeError) as caught:
            run.main(arguments)
        assert caught.value is original
        assert "retained for resume" in " ".join(original.__notes__)
    else:
        assert run.main(arguments) == (run.EXIT_REFUSED if stop == "refusal" else run.EXIT_DEFERRED)
        assert "retained for resume" in capsys.readouterr().err
    assert not any(output.iterdir())
    assert handoff.read_bytes() == retained


def test_recorded_numerical_failure_is_an_execution_failure_not_a_deferral(tmp_path, monkeypatch):
    error = TrexEvidenceUnavailableError("numerical failure", {"numerical_failures": {"TP": 1}})

    def calc(bound, *args, **kwargs):
        bound.failure_record = {"claim_eligible": False, "error_type": type(error).__name__,
                                "reason": str(error), "diagnostics": dict(error.diagnostics),
                                "operational_diagnosis": failure_diagnosis(error, stage="target-api")}
        raise error

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    with pytest.raises(TrexEvidenceUnavailableError) as caught:
        run.main(runner_args(tmp_path, output))
    assert caught.value is error
    record = read_failure(output)
    assert record["operational_diagnosis"]["status"] == "failed"
    assert record["operational_diagnosis"]["category"] == "numerical-failure"
    assert "FPP" not in record


def test_returned_degenerate_result_is_deferred_not_reported_as_success(tmp_path, monkeypatch, capsys):
    def calc(bound, *args, **kwargs):
        bound.trex_result = SimpleNamespace(status="degenerate_evidence")

    install(monkeypatch, tmp_path, calc=calc)
    output = tmp_path / "result"
    assert run.main(runner_args(tmp_path, output)) == run.EXIT_DEFERRED
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "degenerate_evidence"
    assert summary["FPP"] is None and summary["NFPP"] is None
    assert not (output / "failure.json").exists()
    assert "no complete FPP/NFPP" in capsys.readouterr().out


def test_returned_calculation_without_a_result_is_not_success(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path)
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="without a result"):
        run.main(runner_args(tmp_path, output))
    record = read_failure(output)
    assert record["error_type"] == "RuntimeError"
    assert record["operational_diagnosis"]["stage"] == "execution"
    assert not (output / "summary.json").exists()


def test_handoff_publication_never_replaces_retained_state(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(mode=0o700)
    handoff = checkpoint / "monte-carlo.json"
    handoff.write_text('{"binding": {"draws": 1}, "state": {}}\n', encoding="utf-8")
    retained = handoff.read_bytes()
    with pytest.raises(FileExistsError):
        run._publish_handoff(checkpoint, handoff, {"draws": 2}, {"state": "new"})
    assert handoff.read_bytes() == retained
    assert [path.name for path in checkpoint.iterdir()] == ["monte-carlo.json"]


@pytest.mark.slow
def test_real_engine_cli_restores_its_checkpoint_without_repeating_monte_carlo(tmp_path):
    """Reduced-budget execution parity, not converged evidence or native dynesty restore."""
    root = Path(os.environ.get("EXONYM_TEST_RECOVERY_DIR", tmp_path)) / "standalone-real-engine-resume"
    directory, time, flux, _sigma, _period, binding = prepared_inputs(root)
    # A deliberately narrow synthetic likelihood makes the low-draw MC result unresolved.
    np.savez(directory / "arrays.npz", time=time, flux=flux, sigma=np.full(time.size, 1e-5),
             bin_integration_days=np.asarray(binding["bin_integration_days"]))
    output = root / "resumed"
    command = [sys.executable, "-m", "trex.run", "--no-log-file", str(directory.relative_to(root)), str(output),
               "--draws", "3000", "--nsamples", "4"]
    environment = {**os.environ, "TREX_NESTED_ESTIMATOR": "dynesty"}
    environment.pop("TREX_ESCALATION_DEADLINE_MONOTONIC", None)

    def execute(arguments, env, label):
        result = subprocess.run(arguments, cwd=root, env=env, capture_output=True, text=True, timeout=300)
        (root / f"{label}.stdout.txt").write_text(result.stdout)
        (root / f"{label}.stderr.jsonl").write_text(result.stderr)
        return result

    paused = execute(command, {**environment, "TREX_ESCALATION_DEADLINE_MONOTONIC": "0"}, "paused")
    assert paused.returncode == run.EXIT_PAUSED, paused.stderr
    handoff = root / "resumed.escalation/monte-carlo.json"
    retained = handoff.read_bytes()
    state = json.loads(retained)["state"]
    assert state["n_draws"] == 3000 and state["escalation_required"]
    assert not output.exists()
    resumed = execute(command, environment, "resumed")
    assert resumed.returncode == run.EXIT_DEFERRED, resumed.stderr
    events = [json.loads(line) for line in resumed.stderr.splitlines() if line.startswith("{")]
    phases = [row["fields"]["status"] for row in events
              if row.get("event") == "trex.engine.phase" and row["fields"].get("phase") == "monte-carlo"]
    assert phases == ["restored"]
    assert handoff.read_bytes() == retained
    reference_output = root / "reference"
    reference_command = [str(reference_output) if value == str(output) else value for value in command]
    reference = execute(reference_command, environment, "reference")
    assert reference.returncode == run.EXIT_DEFERRED, reference.stderr
    actual, expected = read_failure(output), read_failure(reference_output)
    for key in ("n_draws", "effective_sample_sizes", "unconverged_scenarios", "unconverged_probability_mass"):
        assert actual["diagnostics"][key] == expected["diagnostics"][key]
    assert actual["diagnostics"]["phase_wall_seconds"]["monte_carlo"] == state["monte_carlo_seconds"]
    assert not (output / "summary.json").exists()


CONTRAST_CURVE_ROWS = "0.10, 1.00\n1.00, 3.00\n2.00, 5.00\n"
# A valid curve with fainter limits at every separation: same separations, different consumed values.
CHANGED_CONTRAST_CURVE_ROWS = "0.10, 1.20\n1.00, 3.20\n2.00, 5.20\n"


@pytest.mark.slow
def test_changed_external_contrast_curve_refuses_the_retained_handoff(tmp_path):
    """A curve outside the three-file digest map still binds the restored Monte Carlo state."""
    root = tmp_path / "external-contrast-resume"
    directory, time, flux, _sigma, _period, binding = prepared_inputs(root)
    # The retained real-engine test's narrow likelihood keeps the low-draw Monte Carlo set unresolved.
    np.savez(directory / "arrays.npz", time=time, flux=flux, sigma=np.full(time.size, 1e-5),
             bin_integration_days=np.asarray(binding["bin_integration_days"]))
    curve = root / "measured-contrast.csv"
    curve.write_text(CONTRAST_CURVE_ROWS, encoding="utf-8")
    payload = json.loads((directory / "trex-input.json").read_text(encoding="utf-8"))
    payload["binding"]["contrast_curve"] = str(curve)
    payload["binding"]["contrast_band"] = "TESS"
    (directory / "trex-input.json").write_text(json.dumps(payload), encoding="utf-8")
    primary = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in run.INPUT_FILES}

    checkpoint = tmp_path / "escalation"
    environment = {**os.environ, "TREX_NESTED_ESTIMATOR": "dynesty"}
    environment.pop("TREX_ESCALATION_DEADLINE_MONOTONIC", None)

    def execute(output, env):
        return subprocess.run(
            [sys.executable, "-m", "trex.run", "--no-log-file", str(directory), str(output),
             "--draws", "3000", "--nsamples", "4", "--escalation-checkpoint-directory", str(checkpoint)],
            cwd=root, env=env, capture_output=True, text=True, timeout=600)

    paused = execute(tmp_path / "paused", {**environment, "TREX_ESCALATION_DEADLINE_MONOTONIC": "0"})
    assert paused.returncode == run.EXIT_PAUSED, paused.stderr
    handoff = checkpoint / "monte-carlo.json"
    retained = handoff.read_bytes()
    assert json.loads(retained)["state"]["input_identity"]

    # Only the consumed curve changes; the three prepared files stay byte-identical.
    curve.write_text(CHANGED_CONTRAST_CURVE_ROWS, encoding="utf-8")
    loaded, *_ = run.load_input(directory)
    np.testing.assert_allclose(loaded._require_scene()["scene"].contrast_values, [1.2, 3.2, 5.2])
    changed_output = tmp_path / "changed"
    changed = execute(changed_output, environment)
    assert changed.returncode == run.EXIT_REFUSED, changed.stderr
    # The three-file handoff binding matched; the refusal came from the restored-state identity.
    assert "resuming Monte Carlo handoff" in changed.stdout, changed.stdout + changed.stderr
    record = read_failure(changed_output)
    assert record["operational_diagnosis"]["category"] == "invalid-input"
    assert "escalation state" in record["reason"] and "inputs changed" in record["reason"]
    assert not (changed_output / "summary.json").exists()
    assert handoff.read_bytes() == retained
    assert primary == {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in run.INPUT_FILES}

    # Restoring the curve restores the retained identity: the same handoff is reused.
    curve.write_text(CONTRAST_CURVE_ROWS, encoding="utf-8")
    resumed_output = tmp_path / "resumed"
    resumed = execute(resumed_output, environment)
    assert resumed.returncode == run.EXIT_DEFERRED, resumed.stderr
    assert "resuming Monte Carlo handoff" in resumed.stdout, resumed.stdout + resumed.stderr
    events = [json.loads(line) for line in resumed.stderr.splitlines() if line.startswith("{")]
    phases = [row["fields"]["status"] for row in events
              if row.get("event") == "trex.engine.phase" and row["fields"].get("phase") == "monte-carlo"]
    assert phases == ["restored"]
    assert handoff.read_bytes() == retained
