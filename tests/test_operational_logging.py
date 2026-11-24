"""Standalone CLI and spawned-worker logging, without scientific calculations."""

import errno
import json
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import warnings

import pytest

from trex.runtime import logging as trex_logging


_barrier = None
_WORKER_WARNING = "We have reached the plateau in the likelihood we are stopping sampling"
_INVOCATION = "0123456789abcdef0123456789abcdef"


def _initialize_worker(setup, report, barrier):
    global _barrier
    initializer, args = setup
    initializer(*args)
    trex_logging.install_worker_warning_report(report)
    _barrier = barrier


def _worker_event(index):
    if index == -1:
        error = RuntimeError("password=sample-secret flux=[101.2, 103.4]")
        error.add_note("TREX failure record not written: output directory could not be created: "
                       "[Errno 13] Permission denied: '/private/token=sample-secret'")
        raise error
    if index == -3:
        warnings.warn("magnitude 12.34", RuntimeWarning, stacklevel=1)
        trex_logging.host_logger("engine").warning("magnitude %.2f", 12.34)
        raise RuntimeError("magnitude 12.34")
    if index < 3:
        _barrier.wait(timeout=30)
        warnings.warn(_WORKER_WARNING, RuntimeWarning, stacklevel=1)
    trex_logging.host_logger("engine").info(
        "worker draw %d", index,
        extra={"event": "trex.scenario", "n_draws": index, "flux": [101.2, 103.4]},
    )
    return os.getpid()


def _initialize_blocked_worker(setup, report, barrier, release):
    _initialize_worker(setup, report, barrier)
    _worker_event(0)
    release.wait()


def _run_pool(directory):
    trex_logging._configure_standalone(_INVOCATION, directory)
    try:
        spool = trex_logging.start_worker_warning_spool()
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            3, initializer=_initialize_worker,
            initargs=(trex_logging.worker_log_setup(), trex_logging.worker_warning_report(spool),
                      context.Barrier(3)),
        )
        try:
            pids = pool.map(_worker_event, range(30), chunksize=1)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
            trex_logging.collect_worker_warnings(spool, 3)
        return pids
    finally:
        trex_logging._shutdown_standalone()


def _records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_cli_starts_before_import_keeps_stdout_json_and_supports_no_file(tmp_path):
    environment = {**os.environ, "XDG_STATE_HOME": str(tmp_path / "state")}
    result = subprocess.run(
        [sys.executable, "-m", "trex.doctor", "--no-log-file", str(tmp_path / "missing")],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["inputs"]["category"] == "invalid-input"
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [row["event"] for row in events if row["event"].startswith("trex.cli.")] == [
        "trex.cli.start", "trex.cli.finish",
    ]
    assert events[-1]["fields"]["exit_code"] == 1
    assert not (tmp_path / "state").exists()


def test_run_cli_records_refusal_without_replacing_owning_failure_record(tmp_path):
    destination = tmp_path / "logs"
    result = subprocess.run(
        [sys.executable, "-m", "trex.run", "--log-dir", str(destination),
         str(tmp_path / "missing"), str(tmp_path / "output")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2 and result.stdout == ""
    files = list(destination.glob("*/*.jsonl"))
    assert len(files) == 1
    rows = _records(files[0])
    assert rows[0]["invocation_id"] == rows[-1]["invocation_id"]
    assert rows[-1]["fields"]["exit_code"] == 2 and rows[-1]["fields"]["status"] == "refused"
    failure = json.loads((tmp_path / "output" / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "FileNotFoundError"
    assert failure["operational_diagnosis"]["stage"] == "input-loading"


def test_explicit_unusable_log_root_stops_before_running_the_command(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("existing file", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "trex.doctor", "--log-dir", str(blocked), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2 and result.stdout == ""
    assert "requested log directory unavailable" in result.stderr
    assert blocked.read_text(encoding="utf-8") == "existing file"


@pytest.mark.parametrize(("exit_code", "status"), [
    (0, "finished"), (1, "failed"), (2, "refused"), (3, "deferred"), (75, "paused"),
])
def test_cli_records_the_actual_stop_status(exit_code, status, monkeypatch, capfd):
    from trex.app import run

    monkeypatch.setattr(run, "main", lambda arguments: exit_code)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    assert trex_logging.standalone_cli("trex.app.run") == exit_code
    rows = [json.loads(line) for line in capfd.readouterr().err.splitlines()]
    assert rows[-1]["fields"]["exit_code"] == exit_code
    assert rows[-1]["fields"]["status"] == status


def test_cli_interruption_is_reported_and_propagated(monkeypatch, capfd):
    from trex.app import run

    def interrupted(arguments):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "main", interrupted)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    with pytest.raises(KeyboardInterrupt):
        trex_logging.standalone_cli("trex.app.run")
    rows = [json.loads(line) for line in capfd.readouterr().err.splitlines()]
    stopped = next(row for row in rows if row["event"] == "trex.cli.stopped")
    assert stopped["fields"]["category"] == "interrupted"
    assert rows[-1]["fields"]["exit_code"] == 130


def test_cli_unhandled_failure_retains_the_original_raise_site(monkeypatch, capfd):
    from trex.app import run

    def broken(arguments):
        raise RuntimeError("password=sample-secret flux=[101.2]")

    monkeypatch.setattr(run, "main", broken)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    assert trex_logging.standalone_cli("trex.app.run") == 1
    error = capfd.readouterr().err
    rows = [json.loads(line) for line in error.splitlines()]
    stopped = next(row for row in rows if row["event"] == "trex.cli.stopped")
    assert stopped["exception"]["type"] == "RuntimeError"
    assert any(frame["file"] == "<external>" and frame["line"] == broken.__code__.co_firstlineno + 1
               for frame in stopped["exception"]["frames"])
    assert "sample-secret" not in error and "101.2" not in error


def test_source_literal_exception_cannot_expose_an_observation(monkeypatch, capfd):
    from trex.app import run

    def broken(arguments):
        raise RuntimeError("magnitude 12.34")

    monkeypatch.setattr(run, "main", broken)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    assert trex_logging.standalone_cli("trex.app.run") == 1
    captured = capfd.readouterr().err
    rows = [json.loads(line) for line in captured.splitlines()]
    stopped = next(row for row in rows if row["event"] == "trex.cli.stopped")
    assert stopped["exception"]["message_omitted"] is True
    assert "message" not in stopped["exception"] and "12.34" not in captured


def test_dynamic_exception_keeps_only_the_reviewed_reason(monkeypatch, capfd):
    from trex.app import run

    def broken(arguments):
        detail = "token=sample-secret flux=[101.2]"
        raise RuntimeError(f"native unit numerical/execution integrity failure: {detail}")

    monkeypatch.setattr(run, "main", broken)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    assert trex_logging.standalone_cli("trex.app.run") == 1
    error = capfd.readouterr().err
    rows = [json.loads(line) for line in error.splitlines()]
    stopped = next(row for row in rows if row["event"] == "trex.cli.stopped")
    assert stopped["exception"]["message"] == "native unit numerical/execution integrity failure; details omitted"
    assert stopped["exception"]["message_omitted"] is False
    assert stopped["exception"]["message_redacted"] is True
    assert "sample-secret" not in error and "101.2" not in error


def test_exception_notes_keep_reviewed_reason_and_mark_unrecognized_note(monkeypatch, capfd):
    from trex.app import run

    def broken(arguments):
        error = RuntimeError("no importance draw landed on the likelihood support")
        error.add_note("nested wall deadline reached; sampler checkpoint retained")
        error.add_note("token=sample-secret flux=[101.2]")
        raise error

    monkeypatch.setattr(run, "main", broken)
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file"])
    assert trex_logging.standalone_cli("trex.app.run") == 1
    captured = capfd.readouterr().err
    rows = [json.loads(line) for line in captured.splitlines()]
    stopped = next(row for row in rows if row["event"] == "trex.cli.stopped")
    assert stopped["exception"]["notes"] == ["nested wall deadline reached; sampler checkpoint retained", "<note omitted>"]
    assert "sample-secret" not in captured and "101.2" not in captured


def test_non_scalar_exception_argument_does_not_break_logging():
    class NonScalar:
        def __ne__(self, other):
            raise ValueError("non-scalar comparison")

    exception = trex_logging._exception_frames(ValueError(NonScalar()))
    assert exception["message_omitted"] is True and "message" not in exception


def test_dynamic_budget_reason_keeps_only_its_reviewed_meaning():
    error = RuntimeError("monitored nested likelihood budget of 74123 evaluations exhausted")
    summary = trex_logging._exception_frames(error)
    assert summary["message"] == "monitored nested likelihood budget exhausted"
    assert summary["message_omitted"] is False and summary["message_redacted"] is True
    assert "74123" not in json.dumps(summary)
    assert str(error) == "monitored nested likelihood budget of 74123 evaluations exhausted"


@pytest.mark.parametrize(("reason", "expected"), [
    ("escalation wall deadline reached; retained runs are resumable",
     "escalation wall deadline reached; retained runs are resumable"),
    ("nested wall deadline reached during importance refinement; 73 of 100 likelihoods retained",
     "nested wall deadline reached during importance refinement; likelihood progress retained"),
])
def test_checkpoint_pause_preserves_its_safe_stopping_reason(reason, expected):
    from trex.evidence.nested_checkpoint import NestedCheckpointPaused

    summary = trex_logging._exception_frames(NestedCheckpointPaused(reason))
    assert summary["type"] == "NestedCheckpointPaused" and summary["message"] == expected


def test_cli_keeps_safe_reason_and_actual_failure_record_write_note(tmp_path, monkeypatch, capfd):
    """Publication now raises ENOSPC from the scientific error, rather than adding a swallowed-write note."""
    from trex.app import run

    class FailedTarget:
        failure_record = {"status": "failed"}

        def _require_scene(self):
            return {"input_artifacts": {
                str((tmp_path / "input" / name).absolute()): "a" * 64 for name in run.INPUT_FILES
            }}

        def calc_probs(self, *args, **kwargs):
            raise RuntimeError("no importance draw landed on the likelihood support")

        def write_outputs(self, *args, **kwargs):
            raise OSError(errno.ENOSPC, "token=sample-secret", "/private/observations")

    observed = {"time": [], "flux": [], "sigma": []}
    monkeypatch.setattr(run, "load_input", lambda path: (FailedTarget(), observed, 1, 1))
    monkeypatch.setattr(run, "_selected_nested_estimator", lambda: "dynesty")
    monkeypatch.setattr(trex_logging, "_prime_warning_hook", lambda: None)
    monkeypatch.delenv("TREX_ESCALATION_DEADLINE_MONOTONIC", raising=False)
    monkeypatch.setattr(sys, "argv", ["trex.run", "--no-log-file", str(tmp_path / "input"),
                                       str(tmp_path / "output")])

    assert trex_logging.standalone_cli("trex.app.run") == 1
    captured = capfd.readouterr()
    stopped = next(json.loads(line) for line in captured.err.splitlines()
                   if json.loads(line)["event"] == "trex.cli.stopped")
    exception = stopped["exception"]
    assert exception["type"] == "OSError"
    assert exception["reason"] == os.strerror(errno.ENOSPC)
    assert exception["message_omitted"] is True and "message" not in exception
    assert any(frame["line"] == FailedTarget.write_outputs.__code__.co_firstlineno + 1
               for frame in exception["frames"])
    cause = exception["cause"]
    assert cause["type"] == "RuntimeError"
    assert cause["message"] == "no importance draw landed on the likelihood support"
    assert cause["message_omitted"] is False
    assert any(frame["file"] == "<external>" and frame["line"] == FailedTarget.calc_probs.__code__.co_firstlineno + 1
               for frame in cause["frames"])
    finished = json.loads(captured.err.splitlines()[-1])
    assert finished["fields"]["exit_code"] == 1 and finished["fields"]["status"] == "failed"
    assert "sample-secret" not in captured.err and "/private/observations" not in captured.err
    assert captured.out == "" and not (tmp_path / "output" / "failure.json").exists()


@pytest.mark.parametrize(("detail", "expected"), [
    ("failure record could not be written: [Errno 13] Permission denied: '/private/token=secret'",
     "failure record could not be written (Permission denied)"),
    ("output directory could not be created: [Errno 28] No space left on device: '/private/flux'",
     "output directory could not be created (No space left on device)"),
    ("the empty output is retained for resume", "the empty output is retained for resume"),
    ("unrecognized private detail token=sample-secret", "details omitted"),
])
def test_failure_record_notes_keep_safe_reason_without_private_detail(detail, expected):
    safe = trex_logging._failure_record_note("TREX failure record not written: " + detail)
    assert safe == "TREX failure record not written: " + expected
    assert "sample-secret" not in safe and "/private/" not in safe


def test_standalone_warnings_keep_safe_reasons_and_omit_dynamic_payloads(capfd):
    from trex.evidence.marginal_likelihoods import _warn_jax_batch_fallback

    trex_logging._configure_standalone("warning-reasons", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(_WORKER_WARNING, RuntimeWarning, stacklevel=1)
            secret = "sample-secret"
            warnings.warn(f"token={secret} flux=[101.2, 103.4]", RuntimeWarning, stacklevel=1)
            warnings.warn(f"progress callback display failed: RuntimeError at source:1: {secret}",
                          RuntimeWarning, stacklevel=1)
            warnings.warn("catalog payload: synthetic-source", RuntimeWarning, stacklevel=1)
            warnings.warn("0.35742", RuntimeWarning, stacklevel=1)
            warnings.warn("magnitude 12.34", RuntimeWarning, stacklevel=1)
            _warn_jax_batch_fallback("EB", RuntimeError("token=sample-secret flux=[101.2]"))
        captured = trex_logging._drain_standalone_warnings()
    finally:
        trex_logging._shutdown_standalone()
    error = capfd.readouterr().err
    rows = [json.loads(line) for line in error.splitlines()]
    summaries = [row["fields"]["warning_summary"] for row in rows]
    assert summaries == [
        _WORKER_WARNING,
        "Warning text omitted; inspect the source and owning record",
        "Progress callback failed; details omitted",
        "Warning text omitted; inspect the source and owning record",
        "Warning text omitted; inspect the source and owning record",
        "Warning text omitted; inspect the source and owning record",
        "JAX batched EB likelihood failed (RuntimeError); NumPy reference likelihood used",
    ]
    assert [entry["message"] for entry in captured] == summaries
    assert all(item not in error for item in (
        "sample-secret", "101.2", "103.4", "synthetic-source", "0.35742", "12.34",
    ))


def test_replayed_dynesty_stopping_warning_keeps_its_original_reason(capfd):
    text = "".join([
        "The sampling was stopped short due to maxiter/maxcall limit ",
        "the delta(log(z)) criterion is not achieved; posterior may be poorly sampled",
    ])
    trex_logging._configure_standalone("replayed-warning", None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn_explicit(text, UserWarning, "dynesty/sampler.py", 1082)
    finally:
        trex_logging._shutdown_standalone()
    row = json.loads(capfd.readouterr().err)
    assert row["fields"]["warning_summary"] == text
    assert row["fields"]["warning_location"] == "dynesty/sampler.py:1082"


def test_imports_leave_host_and_public_identity_untouched():
    code = (
        "import importlib, logging, warnings\n"
        "from trex.runtime import logging as log\n"
        "log.set_host_namespace('example_host')\n"
        "def setup(): return (None, ())\n"
        "def show(*args, **kwargs): pass\n"
        "log.set_worker_log_setup(setup)\n"
        "warnings.showwarning = show\n"
        "logger = logging.getLogger('trex')\n"
        "root = logging.getLogger()\n"
        "before = (list(root.handlers), list(logger.handlers), logger.level, logger.propagate, "
        "warnings.showwarning, log.worker_log_setup())\n"
        "assert importlib.import_module('trex.run') is importlib.import_module('trex.app.run')\n"
        "importlib.import_module('trex.doctor')\n"
        "after = (list(root.handlers), list(logger.handlers), logger.level, logger.propagate, "
        "warnings.showwarning, log.worker_log_setup())\n"
        "assert before == after and log.host_logger('engine').name == 'example_host.engine'\n"
        "assert log.worker_log_setup() == (None, ()) and warnings.showwarning is show\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_warning_capture_survives_application_and_library_imports():
    code = (
        "import warnings\n"
        "from trex.runtime import logging as log\n"
        "log._configure_standalone('warning-probe', None)\n"
        "log._prime_warning_hook()\n"
        "import trex.app.run\n"
        "warnings.warn('synthetic warning', RuntimeWarning)\n"
        "log._shutdown_standalone()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stderr.splitlines()]
    assert any(row["event"] == "warning.captured" and row["fields"]["warning_category"] == "RuntimeWarning"
               for row in rows)


def test_warning_hook_setup_failure_is_visible_without_aborting(monkeypatch, capfd):
    def unavailable(name):
        raise ImportError("password=sample-secret")

    monkeypatch.setattr(trex_logging.importlib, "import_module", unavailable)
    trex_logging._configure_standalone("warning-setup-failure", None)
    try:
        trex_logging._prime_warning_hook()
        assert warnings.showwarning is trex_logging._standalone_showwarning
    finally:
        trex_logging._shutdown_standalone()
    error = capfd.readouterr().err
    rows = [json.loads(line) for line in error.splitlines()]
    assert [row["fields"]["warning_module"] for row in rows] == ["astropy", "astroquery"]
    assert all(row["event"] == "warning.setup.degraded" for row in rows)
    assert "sample-secret" not in error


def test_engine_fields_and_chained_traceback_survive_without_payloads(tmp_path):
    from trex.app.doctor import record_engine_execution

    def operation(*, n_draws, nsamples, n_jobs, random_seed, escalation_state_in, escalation_state_out):
        try:
            raise ValueError("password=sample-secret flux=[101.2, 103.4]")
        except ValueError as cause:
            raise RuntimeError("catalog payload and chain data") from cause

    trex_logging._configure_standalone("engine-invocation", str(tmp_path))
    try:
        with pytest.raises(RuntimeError):
            record_engine_execution(operation)(
                n_draws=73, nsamples=4, n_jobs=2, random_seed=19,
                escalation_state_in=None, escalation_state_out=None,
            )
        trex_logging.host_logger("engine").info(
            "TREX calculation started", extra={"event": "trex.device_pilot", "ess": 8.5, "FPP": 0.02,
                                                "NFPP": 0.01, "stop_reason": "evidence-tolerance",
                                                "phase": "nested", "budget": 1000, "elapsed_seconds": 2.5,
                                                "pilot": [{"device": "cpu", "draws": 12,
                                                           "parity": {"passed": True, "tolerance": 0.01,
                                                                      "flux": [101.2]}}],
                                                "flux": [101.2, 103.4], "password": "sample-secret"},
        )
        trex_logging.host_logger("engine").warning("flux=%f", 101.2, extra={"event": "trex.batch.retry"})
    finally:
        trex_logging._shutdown_standalone()

    path, = tmp_path.glob("*.jsonl")
    raw = path.read_text(encoding="utf-8")
    assert all(item not in raw for item in ("sample-secret", "101.2", "103.4", "catalog payload"))
    rows = _records(path)
    stopped = next(row for row in rows if row["event"] == "trex.engine.stopped")
    assert stopped["fields"]["n_draws"] == 73
    assert stopped["fields"]["nsamples"] == 4 and stopped["fields"]["n_jobs"] == 2
    assert stopped["fields"]["stage"] == "full" and stopped["fields"]["wall_seconds"] >= 0
    assert stopped["exception"]["type"] == "RuntimeError"
    assert stopped["exception"]["cause"]["type"] == "ValueError"
    assert any(frame["file"] == "<external>" and frame["line"] == operation.__code__.co_firstlineno + 2
               for frame in stopped["exception"]["cause"]["frames"])
    metrics = next(row["fields"] for row in rows if row["event"] == "trex.device_pilot")
    assert (metrics["ess"], metrics["FPP"], metrics["NFPP"], metrics["stop_reason"]) == (
        8.5, 0.02, 0.01, "evidence-tolerance",
    )
    assert metrics["pilot"] == [{"device": "cpu", "draws": 12,
                                 "parity": {"passed": True, "tolerance": 0.01}}]
    assert (metrics["phase"], metrics["budget"], metrics["elapsed_seconds"]) == ("nested", 1000, 2.5)


@pytest.mark.parametrize("field", sorted(trex_logging._METRIC_FIELDS))
@pytest.mark.parametrize("payload", ["value 12.34", "unreviewedcontext"])
def test_structured_fields_never_copy_unreviewed_text(field, payload):
    record = logging.makeLogRecord({"name": "trex", "msg": "probe", "event": "trex.engine.phase", field: payload})
    row = json.loads(trex_logging._JsonFormatter(_INVOCATION).format(record))
    assert row["fields"][field] == "<omitted>"
    assert payload not in json.dumps(row["fields"])


def test_metadata_and_warning_location_keep_lines_without_copying_private_text():
    record = logging.LogRecord("trex.value12.34", logging.WARNING, "/private/sample-secret/value12.34.py", 73,
                               "probe", (), None, "value12.34")
    record.levelname = "sample-secret"
    record.created = "value 12.34"
    record.process = "value 12.34"
    record.warning_location = "/private/sample-secret/value12.34.py:81"
    record.phase = "unreviewedcontext"
    record.error_type = "unreviewedcontext"
    row = json.loads(trex_logging._JsonFormatter("sample-secret").format(record))
    assert row["logger"] == "<omitted>" and row["invocation_id"] == "<omitted>"
    assert row["source"] == {"file": "<external>", "line": 73, "function": "<omitted>"}
    assert row["fields"]["warning_location"] == "<external>:81"
    assert row["fields"]["phase"] == row["fields"]["error_type"] == "<omitted>"
    assert row["level"] == "WARNING" and row["process"] is None and row["timestamp_utc"] is None
    assert all(value not in json.dumps(row) for value in ("12.34", "sample-secret", "unreviewedcontext"))


def test_reviewed_function_keeps_name_when_its_external_path_is_private():
    assert trex_logging._source("/private/sample-secret/value12.34.py", 73, "calc") == {
        "file": "<external>", "line": 73, "function": "calc",
    }


def test_structural_names_accept_new_trex_code_without_allowlist_edits():
    """New modules, functions and exception types are code identities, not payloads."""
    class NewEvidenceError(RuntimeError):
        pass

    new_module = Path(trex_logging.__file__).with_name("new_evidence.py")
    assert trex_logging._source(str(new_module), 7, "new_evidence_step") == {
        "file": "trex/runtime/new_evidence.py", "line": 7, "function": "new_evidence_step",
    }
    assert trex_logging._source("trex/evidence/new_evidence.py", 7, "new_evidence_step") == {
        "file": "trex/evidence/new_evidence.py", "line": 7, "function": "new_evidence_step",
    }
    assert trex_logging._source("/private/token=sample-secret/new.py", 7, "new_evidence_step")["file"] == "<external>"
    assert trex_logging._source(str(new_module), 7, "value12.34")["function"] == "<omitted>"
    assert trex_logging._exception_frames(NewEvidenceError("token=sample-secret"))["type"] == "NewEvidenceError"
    assert trex_logging._field("error_type", "FutureImportedError") == "FutureImportedError"
    assert trex_logging._field("error_type", "unreviewedcontext") == "<omitted>"


def test_reviewed_source_and_diagnostic_values_remain_useful():
    record = logging.LogRecord("trex.engine", logging.ERROR, trex_logging.__file__, 123, "probe", (), None,
                               "standalone_cli")
    record.event = "trex.engine.stopped"
    record.warning_location = str(trex_logging.__file__) + ":456"
    record.stage, record.phase, record.status = "full", "nested", "failed"
    record.category, record.error_type = "numerical-failure", "NumericalLikelihoodError"
    record.next_action = "Repair the failed supported likelihood; retain every scenario."
    record.stop_reason, record.n_draws, record.wall_seconds = "truncated", 73, 0.5
    row = json.loads(trex_logging._JsonFormatter(_INVOCATION).format(record))
    assert row["source"] == {"file": "trex/runtime/logging.py", "line": 123, "function": "standalone_cli"}
    assert row["fields"]["warning_location"] == "trex/runtime/logging.py:456"
    assert row["logger"] == "trex.engine" and row["invocation_id"] == _INVOCATION
    assert (row["fields"]["stage"], row["fields"]["phase"], row["fields"]["status"]) == ("full", "nested", "failed")
    assert (row["fields"]["category"], row["fields"]["error_type"], row["fields"]["stop_reason"]) == (
        "numerical-failure", "NumericalLikelihoodError", "truncated",
    )
    assert (row["fields"]["n_draws"], row["fields"]["wall_seconds"]) == (73, 0.5)


def test_warning_capture_sanitizes_source_before_worker_transport(capfd):
    trex_logging._configure_standalone(_INVOCATION, None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn_explicit("synthetic warning", RuntimeWarning, "/private/sample-secret/value12.34.py", 83)
        entries = trex_logging._drain_standalone_warnings()
    finally:
        trex_logging._shutdown_standalone()
    row = json.loads(capfd.readouterr().err)
    assert row["fields"]["warning_location"] == entries[0]["location"] == "<external>:83"
    assert row["fields"]["warning_category"] == entries[0]["category"] == "RuntimeWarning"
    assert "sample-secret" not in json.dumps(entries) and "12.34" not in json.dumps(entries)


def test_remote_traceback_hides_unreviewed_filename_and_function_but_keeps_line():
    from multiprocessing.pool import RemoteTraceback

    error = RuntimeError("no importance draw landed on the likelihood support")
    error.__cause__ = RemoteTraceback(
        'Traceback (most recent call last):\n  File "/private/sample-secret/value12.34.py", line 87, '
        'in value12.34\n    sensitive code\nRuntimeError: sample-secret\n'
    )
    summary = trex_logging._exception_frames(error)
    assert summary["message"] == "no importance draw landed on the likelihood support"
    assert summary["cause"]["frames"] == [{"file": "<external>", "line": 87, "function": "<omitted>"}]
    assert "sample-secret" not in json.dumps(summary) and "12.34" not in json.dumps(summary)


def test_json_decode_traceback_retains_reviewed_standard_library_location():
    try:
        json.loads('{"value":')
    except json.JSONDecodeError as error:
        frames = trex_logging._exception_frames(error)["frames"]
    assert any(frame["file"] == "json/decoder.py" and frame["function"] == "raw_decode"
               and isinstance(frame["line"], int) for frame in frames)


@pytest.mark.parametrize(("event", "expected"), [
    ("trex.engine.start", "trex.engine.start"),
    ("trex.device_pilot", "trex.device_pilot"),
    ("warning.transport.lost", "warning.transport.lost"),
    ("value 12.34", "<omitted>"),
    ("trex.engine.start.value.12.34", "<omitted>"),
    ("unknown.event", "<omitted>"),
    ({"token": "sample-secret"}, "<omitted>"),
    (None, None),
])
def test_event_only_keeps_reviewed_identifiers(event, expected):
    record = logging.makeLogRecord({"name": "trex", "msg": "probe", "event": event})
    output = trex_logging._JsonFormatter("event-probe").format(record)
    row = json.loads(output)
    assert row["event"] == expected
    row.pop("timestamp_utc")
    assert "12.34" not in json.dumps(row) and "sample-secret" not in json.dumps(row)


@pytest.mark.parametrize(("device", "expected"), [
    ("cpu:0", "cpu:0"), ("gpu:1", "gpu:1"), ("cuda:0", "cuda:0"), ("rocm:0", "rocm:0"),
    ("tpu:0", "tpu:0"), ("TFRT_CPU_0", "TFRT_CPU_0"), ("cpu", "cpu"),
    ("value 12.34", "<omitted>"), ("cpu:12.34", "<omitted>"),
    ("cuda:0 token=sample-secret", "<omitted>"), (12.34, "<omitted>"),
])
def test_chosen_and_pilot_device_only_keep_recognized_ids(device, expected):
    record = logging.makeLogRecord({"name": "trex", "msg": "probe", "event": "trex.device_pilot",
                                    "chosen": [device], "pilot": [{"device": device, "draws": 73}]})
    output = trex_logging._JsonFormatter("device-probe").format(record)
    fields = json.loads(output)["fields"]
    assert fields["chosen"] == [expected] and fields["pilot"][0]["device"] == expected
    assert fields["pilot"][0]["draws"] == 73
    assert "12.34" not in json.dumps(fields) and "sample-secret" not in json.dumps(fields)


@pytest.mark.parametrize("value", ["value 12.34", 12.34, {"token": "sample-secret"}])
def test_malformed_chosen_and_pilot_containers_do_not_fall_through(value):
    record = logging.makeLogRecord({"name": "trex", "msg": "probe", "event": "trex.device_pilot",
                                    "chosen": value, "pilot": value})
    fields = json.loads(trex_logging._JsonFormatter("device-probe").format(record))["fields"]
    assert fields["chosen"] == "<omitted>" and fields["pilot"] == "<omitted>"


@pytest.mark.parametrize("durable", [True, False])
def test_pilot_rejects_unreviewed_text_without_losing_numeric_metrics(tmp_path, capfd, durable):
    trex_logging._configure_standalone("pilot-sanitization", str(tmp_path) if durable else None)
    try:
        trex_logging.host_logger("trex").info(
            "TREX device pilot: chose %s (%s)", ["cpu:0"], "no accelerator passed float64 parity",
            extra={"event": "trex.device_pilot", "chosen": ["cpu:0", "value 12.34"],
                   "pilot": [{"device": "cpu:0", "draws": 73, "seconds": 0.5, "reference": True,
                              "parity": {"passed": True, "tolerance": 0.01,
                                         "max_abs_lnL_difference": 0.001, "device": "value 12.34"}},
                             {"device": "value 12.34", "draws": "value 12.34", "seconds": "value 12.34",
                              "reference": "value 12.34", "parity": {"passed": "value 12.34",
                                                                         "tolerance": "value 12.34"}}],
                   "n_draws": 1000, "wall_seconds": 2.5, "ess": 8.5},
        )
    finally:
        trex_logging._shutdown_standalone()
    captured = capfd.readouterr()
    disk = "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.jsonl"))
    assert captured.out == ""
    row = json.loads((disk if durable else captured.err).strip())
    fields = row["fields"]
    assert "12.34" not in json.dumps(fields)
    assert json.loads(captured.err)["fields"] == fields
    assert row["event"] == "trex.device_pilot" and fields["chosen"] == ["cpu:0", "<omitted>"]
    assert fields["pilot"][0] == {
        "device": "cpu:0", "draws": 73, "seconds": 0.5, "reference": True,
        "parity": {"passed": True, "tolerance": 0.01, "max_abs_lnL_difference": 0.001},
    }
    assert fields["pilot"][1] == {
        "device": "<omitted>", "draws": "<omitted>", "seconds": "<omitted>", "reference": "<omitted>",
        "parity": {"passed": "<omitted>", "tolerance": "<omitted>"},
    }
    assert (fields["n_draws"], fields["wall_seconds"], fields["ess"]) == (1000, 2.5, 8.5)
    if not durable:
        assert not list(tmp_path.iterdir())


def test_formatted_numeric_messages_cannot_expose_observations(tmp_path, capfd):
    trex_logging._configure_standalone("formatted-input", str(tmp_path))
    try:
        log = trex_logging.host_logger("engine")
        log.warning("magnitude %.2f", 12.34)
        log.warning("value %s", "magnitude 12.34")
        log.warning("magnitude 12.34")
        log.warning("TREX calculation stopped: %s", "magnitude 12.34", extra={"category": "magnitude 12.34"})
        log.warning("TREX calculation stopped: %s", 12.34)
        log.info("TREX Monte Carlo phase %s: %d draws per scenario, %.2f s", "finished", 73, 0.5,
                 extra={"event": "trex.engine.phase", "phase": "monte-carlo", "status": "finished",
                        "n_draws": 73, "wall_seconds": 0.5})
        trex_logging.host_logger("trex").info(
            "TREX %s: batch=%d, compiled working set=%d bytes, memory budget=%d bytes", "gpu", 64,
            4096, 65536, extra={"event": "trex.batch_plan"},
        )
        trex_logging.host_logger("trex").info(
            "TREX device pilot: chose %s (%s)", ["cpu"], "no accelerator passed float64 parity",
            extra={"event": "trex.device_pilot", "chosen": ["cpu"],
                   "reason": "no accelerator passed float64 parity"},
        )
    finally:
        trex_logging._shutdown_standalone()
    captured = capfd.readouterr()
    path, = tmp_path.glob("*.jsonl")
    disk = path.read_text(encoding="utf-8")
    assert "12.34" not in captured.err and "12.34" not in disk
    rows = _records(path)
    assert all(row["message"] == "Unstructured message omitted" for row in rows[:3])
    assert rows[3]["fields"]["category"] == "<omitted>" and "12.34" not in rows[3]["message"]
    assert rows[4]["message"] == "TREX calculation stopped"
    assert rows[5]["fields"]["n_draws"] == 73 and rows[5]["fields"]["wall_seconds"] == 0.5
    assert rows[6]["fields"] == {"batch_size": 64, "compiled_bytes": 4096, "memory_budget_bytes": 65536}
    assert rows[7]["fields"]["reason"] == "no accelerator passed float64 parity"


@pytest.mark.parametrize("durable", [True, False])
def test_actual_spawn_pool_transports_safe_exception_and_hides_observations(tmp_path, capfd, durable):
    from trex.evidence.nested_sampling import _cached_importance_block, _worker_initializer

    directory = str(tmp_path) if durable else None
    trex_logging._configure_standalone("worker-reasons", directory)
    try:
        spool = trex_logging.start_worker_warning_spool()
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            1, initializer=_worker_initializer,
            initargs=(None, None, None, trex_logging.worker_log_setup(), trex_logging.worker_warning_report(spool)),
        )
        try:
            for function, argument, event in (
                (_cached_importance_block, None, "trex.nested.failed"),
                (_worker_event, -3, "trex.engine.stopped"),
            ):
                try:
                    pool.apply_async(function, (argument,)).get(timeout=60)
                except RuntimeError:
                    trex_logging.host_logger("engine").exception("Worker calculation failed", extra={"event": event})
                else:
                    pytest.fail("worker error did not reach its caller")
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
            trex_logging.collect_worker_warnings(spool, 1)
    finally:
        trex_logging._shutdown_standalone()
    captured = capfd.readouterr()
    disk = "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.jsonl"))
    assert captured.out == "" and "12.34" not in captured.err and "12.34" not in disk
    records = [json.loads(line) for line in (disk if durable else captured.err).splitlines()]
    safe = next(row for row in records if row["event"] == "trex.nested.failed")
    assert safe["exception"]["message"] == "importance worker initializer did not receive its block callable"
    assert safe["exception"]["message_omitted"] is False
    assert any(frame["function"] == "_cached_importance_block" and frame["file"].endswith("nested_sampling.py")
               for frame in safe["exception"]["cause"]["frames"])
    sensitive = next(row for row in records if row["event"] == "trex.engine.stopped")
    assert sensitive["exception"]["message_omitted"] is True
    assert "message" not in sensitive["exception"]
    assert any(row["event"] == "warning.worker" and row["fields"]["warning_summary"] == trex_logging._WARNING_OMITTED
               for row in records)
    assert not [row for row in records if row["event"] == "warning.transport.lost"]
    if not durable:
        assert not list(tmp_path.iterdir())


def test_spawned_workers_own_separate_files_and_report_warnings(tmp_path, capfd):
    pids = _run_pool(str(tmp_path))
    assert len(set(pids)) == 3
    rows_by_file = [_records(path) for path in tmp_path.glob("*.jsonl")]
    assert len(rows_by_file) == 4  # parent plus one file per worker
    for rows in rows_by_file:
        assert len({row["process"] for row in rows}) == 1
        assert {row["invocation_id"] for row in rows} == {_INVOCATION}
    records = [row for rows in rows_by_file for row in rows]
    worker_events = [row for row in records if row["event"] == "trex.scenario"]
    assert sorted(row["fields"]["n_draws"] for row in worker_events) == list(range(30))
    assert {row["process"] for row in worker_events} == set(pids)
    assert len([row for row in records if row["event"] == "trex.worker.stop"]) == 3
    adopted = [row for row in records if row["event"] == "warning.worker"]
    assert sum(row["fields"]["count"] for row in adopted) == 3
    assert all(row["fields"]["warning_summary"] == _WORKER_WARNING for row in adopted)
    assert not [row for row in records if row["event"] == "warning.transport.lost"]
    assert "101.2" not in "".join(path.read_text() for path in tmp_path.glob("*.jsonl"))
    capfd.readouterr()


def test_no_file_policy_reaches_spawned_workers_and_warning_transport(tmp_path, capfd, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    pids = _run_pool(None)
    captured = capfd.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert captured.out == ""
    assert len(set(pids)) == 3
    assert {row["process"] for row in records if row["event"] == "trex.scenario"} == set(pids)
    assert len([row for row in records if row["event"] == "warning.worker"]) == 3
    assert all(row["fields"]["warning_summary"] == _WORKER_WARNING
               for row in records if row["event"] == "warning.worker")
    assert not [row for row in records if row["event"] == "warning.transport.lost"]
    assert not list(tmp_path.iterdir())


def test_terminated_workers_are_reported_as_missing_warning_payloads(tmp_path):
    trex_logging._configure_standalone("terminated-workers", str(tmp_path))
    try:
        spool = trex_logging.start_worker_warning_spool()
        context = multiprocessing.get_context("spawn")
        barrier, release = context.Barrier(4), context.Event()
        pool = context.Pool(
            3, initializer=_initialize_blocked_worker,
            initargs=(trex_logging.worker_log_setup(), trex_logging.worker_warning_report(spool),
                      barrier, release),
        )
        try:
            # Idle workers may finalize normally after terminate's sentinel; keep all three blocked.
            barrier.wait(timeout=60)
        finally:
            pool.terminate()
            pool.join()
            trex_logging.collect_worker_warnings(spool, 3)
    finally:
        trex_logging._shutdown_standalone()
    rows = [row for path in tmp_path.glob("*.jsonl") for row in _records(path)]
    lost = next(row for row in rows if row["event"] == "warning.transport.lost")
    assert lost["fields"]["count"] == 3
    assert any(row["event"] == "warning.worker" and row["fields"]["warning_category"] == "WorkerWarningLoss"
               for row in rows)


def test_remote_worker_failure_keeps_worker_raise_site_without_exception_payload(tmp_path):
    trex_logging._configure_standalone("remote-exception", str(tmp_path))
    try:
        spool = trex_logging.start_worker_warning_spool()
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            1, initializer=_initialize_worker,
            initargs=(trex_logging.worker_log_setup(), trex_logging.worker_warning_report(spool), None),
        )
        try:
            try:
                pool.apply(_worker_event, (-1,))
            except RuntimeError:
                trex_logging.host_logger("engine").exception(
                    "Worker calculation failed", extra={"event": "trex.nested.failed"},
                )
            else:
                pytest.fail("worker error did not reach its caller")
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
            trex_logging.collect_worker_warnings(spool, 1)
    finally:
        trex_logging._shutdown_standalone()
    raw = "".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.jsonl"))
    assert "sample-secret" not in raw and "101.2" not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    failure = next(row for row in records if row["event"] == "trex.nested.failed")
    assert failure["exception"]["type"] == "RuntimeError"
    assert failure["exception"]["notes"] == [
        "TREX failure record not written: output directory could not be created (Permission denied)"
    ]
    assert any(frame["file"] == "<external>" and frame["line"] == _worker_event.__code__.co_firstlineno + 5
               for frame in failure["exception"]["cause"]["frames"])


def test_unavailable_warning_transport_reaches_host_record(monkeypatch, caplog):
    adopted = []
    original = trex_logging._worker_warning_hooks
    trex_logging.set_worker_warning_hooks((list, adopted.extend))
    monkeypatch.setattr(trex_logging.tempfile, "mkdtemp", lambda **kwargs: (_ for _ in ()).throw(OSError()))
    try:
        with caplog.at_level(logging.WARNING, logger=trex_logging.host_logger("trex").name):
            assert trex_logging.start_worker_warning_spool() is None
    finally:
        trex_logging.set_worker_warning_hooks(original)
    assert adopted[0]["category"] == "WorkerWarningTransportUnavailable"
    assert any(getattr(record, "event", None) == "warning.transport.unavailable" for record in caplog.records)


def test_damaged_worker_payload_is_counted_as_lost(tmp_path):
    adopted = []
    original = trex_logging._worker_warning_hooks
    trex_logging.set_worker_warning_hooks((list, adopted.extend))
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "started-101").touch()
    (spool / "worker-101.json").write_text('[{"message": "valid"}, null]', encoding="utf-8")
    try:
        trex_logging.collect_worker_warnings(str(spool), 1)
    finally:
        trex_logging.set_worker_warning_hooks(original)
    assert any(entry["category"] == "WorkerWarningLoss" for entry in adopted)
    assert any(entry["message"] == "valid" for entry in adopted)


def test_file_write_failure_is_announced_and_counted(tmp_path, capfd):
    class BrokenStream:
        def write(self, text):
            raise OSError("full disk")

        def flush(self):
            pass

        def close(self):
            pass

    trex_logging._configure_standalone("disk-failure", str(tmp_path))
    try:
        handler = trex_logging._standalone_state["file"]
        handler.stream.close()
        handler.stream = BrokenStream()
        trex_logging.host_logger("cli").info("TREX CLI started", extra={"event": "trex.cli.start"})
        trex_logging.host_logger("cli").info("TREX CLI finished", extra={"event": "trex.cli.finish"})
    finally:
        trex_logging._shutdown_standalone()
    error = capfd.readouterr().err
    assert "file write failed" in error and "2 file record(s) lost" in error
