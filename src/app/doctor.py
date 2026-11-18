"""Operational diagnoses using TREX's real input admission, without evidence draws."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import wraps
import inspect
import json
import logging
import multiprocessing
from pathlib import Path
import time
from time import monotonic

from trex.foundation.errors import NumericalLikelihoodError, TrexEvidenceUnavailableError
from trex.runtime.logging import host_logger, worker_log_setup

# Failure stages name the boundary the engine refused at, not merely the run
# mode: pre-draw admission, a target API contract, or the evidence stage.
INPUT_ADMISSION = "input-admission"
TARGET_BINDING = "target-binding"


def failure_diagnosis(error, *, stage):
    """Separate missing inputs, convergence, numerical defects and checkpoint pauses.

    ``NumericalLikelihoodError`` is a demonstrated supported-likelihood failure.
    A recorded ``numerical_failures`` table takes precedence over the aggregate
    convergence gate in ``TrexEvidenceUnavailableError``: a deferred run with a
    recorded numerical defect is ``failed``, not merely under-sampled.
    """
    from trex.evidence.nested_checkpoint import NestedCheckpointPaused

    diagnostics = getattr(error, "diagnostics", None)
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    if isinstance(error, NestedCheckpointPaused):
        category, action = "paused", "Resume the retained checkpoint with the same inputs and budget."
    elif isinstance(error, NumericalLikelihoodError):
        category, action = "numerical-failure", "Repair the failed supported likelihood; retain every scenario."
    elif isinstance(error, TrexEvidenceUnavailableError):
        if diagnostics.get("numerical_failures"):
            category, action = (
                "numerical-failure",
                "Inspect the recorded scenario failures, repair the supported likelihood and retain every scenario.",
            )
        elif ("effective_sample_sizes" in diagnostics or "unconverged_probability_mass" in diagnostics
              or "unconverged_scenarios" in diagnostics):
            category, action = "convergence", "Inspect retained scenario convergence; do not lower the scientific budget."
        else:
            category, action = "missing-evidence", "Supply the recorded required evidence or supported model configuration."
    elif isinstance(error, (ValueError, TypeError, FileNotFoundError, NotImplementedError)):
        category, action = "invalid-input", "Correct the supplied inputs or unsupported option before retrying."
    else:
        category, action = "execution-error", "Inspect the original traceback and repair the failing production path."
    diagnosis = {"status": "deferred" if category in {"paused", "convergence", "missing-evidence"} else "failed",
                 "stage": stage, "category": category, "error_type": type(error).__name__,
                 "reason": str(error), "next_action": action}
    if category == "paused":
        phase = next((name for name in ("mc_importance", "nested_escalation")
                      if name in diagnostics), None)
        if phase is not None:
            diagnosis["phase"] = phase
    return diagnosis


def _log_input_report(report):
    """Log one read-only input check with its boundary and timing, never inputs."""
    host_logger("trex.doctor").log(
        logging.WARNING if report["status"] != "ready" else logging.INFO,
        "TREX input check: %s", report["status"],
        extra={"event": "trex.doctor.inputs", "status": report["status"],
               "category": report.get("category"), "stage": report.get("stage"),
               "next_action": report.get("next_action"), "wall_seconds": report["wall_seconds"]})
    return report


def failed_input_report(error, *, stage, started):
    """Build and log a writable, no-draw diagnosis at the failing boundary."""
    report = {"kind": "trex-input-health", "evidence_draws": 0,
              **failure_diagnosis(error, stage=getattr(error, "trex_boundary", stage)),
              "wall_seconds": monotonic() - started}
    if isinstance(error, TrexEvidenceUnavailableError):
        report["diagnostics"] = error.diagnostics
    return _log_input_report(report)


def record_engine_execution(operation):
    """Log the actual engine call and retain a diagnosis on its original exception."""
    signature = inspect.signature(operation)

    @wraps(operation)
    def run(*args, **kwargs):
        controls = signature.bind(*args, **kwargs)
        controls.apply_defaults()
        values = controls.arguments
        fields = {key: values[key] for key in ("n_draws", "nsamples", "n_jobs", "random_seed") if key in values}
        fields["stage"] = "resume" if values.get("escalation_state_in") is not None else "full"
        log = host_logger("trex.engine")
        started = time.monotonic()
        log.info("TREX calculation started", extra={"event": "trex.engine.start", **fields})
        try:
            result = operation(*args, **kwargs)
        except Exception as error:
            diagnosis = failure_diagnosis(
                error, stage=getattr(error, "trex_boundary", fields["stage"]))
            error.trex_diagnosis = diagnosis
            if isinstance(error, TrexEvidenceUnavailableError):
                error.diagnostics = {**error.diagnostics, "operational_diagnosis": dict(diagnosis)}
            stopped = {"event": "trex.engine.stopped", **fields,
                       "category": diagnosis["category"], "error_type": diagnosis["error_type"],
                       "next_action": diagnosis["next_action"],
                       "wall_seconds": time.monotonic() - started}
            if diagnosis.get("phase"):
                stopped["phase"] = diagnosis["phase"]
            elif diagnosis["stage"] != fields["stage"]:
                stopped["phase"] = diagnosis["stage"]
            log.log(logging.WARNING if diagnosis["status"] == "deferred" else logging.ERROR,
                    "TREX calculation stopped: %s", diagnosis["category"], exc_info=True,
                    extra=stopped)
            raise
        pending = values.get("escalation_state_out")
        # Logging must not fail a completed scientific result: a substituted
        # operation or result without engine attributes still finishes normally.
        status = getattr(result, "status", None)
        log.info("TREX calculation %s", "paused for handoff" if pending else "finished",
                 extra={"event": "trex.engine.finish", **fields,
                        "wall_seconds": time.monotonic() - started,
                        "status": "paused" if pending else status,
                        "FPP": None if pending else getattr(result, "fpp", None),
                        "NFPP": None if pending else getattr(result, "nfpp", None)})
        return result

    return run


def record_target_failure(operation):
    """Keep early API refusals as well as engine failures available to write_outputs."""
    @wraps(operation)
    def run(self, *args, **kwargs):
        try:
            return operation(self, *args, **kwargs)
        except Exception as error:
            attached = getattr(error, "trex_diagnosis", None)
            diagnosis = attached or failure_diagnosis(
                error, stage=getattr(error, "trex_boundary", "target-api"))
            record = self.failure_record or {"error_type": type(error).__name__, "reason": str(error)}
            self.failure_record = {**record, "operational_diagnosis": diagnosis}
            if attached is None:
                error.trex_diagnosis = diagnosis
                if isinstance(error, TrexEvidenceUnavailableError):
                    error.diagnostics = {**error.diagnostics, "operational_diagnosis": dict(diagnosis)}
                host_logger("trex.target").log(
                    logging.WARNING if diagnosis["status"] == "deferred" else logging.ERROR,
                    "TREX input refused: %s", diagnosis["category"], exc_info=True,
                    extra={"event": "trex.target.refused", "stage": diagnosis["stage"],
                           "category": diagnosis["category"], "error_type": diagnosis["error_type"],
                           "next_action": diagnosis["next_action"]})
            raise
    return run


def inspect_inputs(scene, time, flux, sigma, period_days, depth_ppm, exptime_days, **controls):
    """Run the engine's exact pre-draw checks, never samplers or a network request.

    Ready means admissible inputs at this instant, not converged or valid FPP.
    No observation, catalog payload or posterior is copied into this report.
    """
    from trex.app.engine import (
        _TREX_DEFAULT_N_DRAWS, _TREX_DEFAULT_NSAMPLES, prepare_vetting_inputs,
    )

    started = monotonic()
    report = {"kind": "trex-input-health", "evidence_draws": 0,
              "stage": INPUT_ADMISSION,
              "n_draws": controls.get("n_draws", _TREX_DEFAULT_N_DRAWS),
              "nsamples": controls.get("nsamples", _TREX_DEFAULT_NSAMPLES)}
    try:
        prepared = prepare_vetting_inputs(scene, time, flux, sigma, period_days, depth_ppm, exptime_days, **controls)
        report.update(status="ready", cadences=len(prepared[0]),
                       neighbors=len(prepared[6]), next_action="Inputs are admissible; run TREX to assess evidence and convergence.",
                      wall_seconds=monotonic() - started)
    except Exception as error:
        return failed_input_report(error, stage=INPUT_ADMISSION, started=started)
    return _log_input_report(report)


def _mast_probe(connection, tic_id, timeout, logging_setup):
    """Use the same MAST application transport as acquisition, without downloading FITS."""
    try:
        if logging_setup is not None:
            initializer, args = logging_setup
            initializer(*args)
        from urllib.parse import urlencode
        from trex.acquisition.transport import AcquisitionTransport, NetworkPolicy
        from trex.acquisition.tpf import MAST_INVOKE_URI

        transport = AcquisitionTransport(NetworkPolicy(True, attempts=1, timeout_seconds=timeout,
                                                       max_response_bytes=1_000_000))
        request = {"service": "Mast.Caom.Filtered", "params": {"columns": "obsid",
                   "filters": [{"paramName": "obs_collection", "values": ["TESS"]},
                               {"paramName": "target_name", "values": [str(tic_id)]}]},
                   "format": "json", "pagesize": 1, "page": 1}
        body = transport.post_mast(MAST_INVOKE_URI, urlencode({"request": json.dumps(request)}).encode())
        if body is None:
            if not transport.transport_failures:
                raise RuntimeError("MAST transport returned no response and no failure record")
            failure = transport.transport_failures[-1]
            result = {"status": "deferred", "category": failure.category, "reason": failure.reason,
                      "error_type": failure.exception_class}
        else:
            reply = json.loads(body.decode("utf-8"))
            if not isinstance(reply, dict) or reply.get("status") != "COMPLETE" or not isinstance(reply.get("data"), list):
                result = {"status": "deferred", "category": "application-error",
                          "reason": str(reply.get("msg", "MAST did not return a complete data table"))
                          if isinstance(reply, dict) else "MAST returned a non-object response"}
            else:
                result = {"status": "ready", "rows": len(reply["data"])}
    except Exception as error:
        result = {"status": "deferred", "category": "probe-error",
                  "error_type": type(error).__name__, "reason": str(error)}
    try:
        connection.send(result)
    except (BrokenPipeError, EOFError, OSError):
        # The bounded parent may already have timed out and closed its pipe.
        pass
    finally:
        connection.close()


def probe_mast(tic_id, *, timeout_seconds=15.0):
    """A hard wall-clock bound including DNS, imports and client waits, on this host only."""
    from trex.acquisition.identity import exact_identifier
    from trex.populations._validation import real_scalar

    tic_id = exact_identifier(tic_id, "TIC ID")
    timeout_seconds = real_scalar(timeout_seconds, "timeout_seconds", positive=True)
    if timeout_seconds > 60:
        raise ValueError("timeout_seconds must be finite and in (0, 60]")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_mast_probe, args=(send, tic_id, timeout_seconds, worker_log_setup()))
    started = time.monotonic()
    try:
        process.start()
        send.close()
        if receive.poll(timeout_seconds):
            result = receive.recv()
        else:
            result = {"status": "deferred", "category": "timeout", "error_type": "TimeoutError",
                      "reason": "MAST query exceeded its wall-clock budget; service health is unknown."}
    except (OSError, RuntimeError, EOFError) as error:
        result = {"status": "deferred", "category": "probe-error", "error_type": type(error).__name__, "reason": str(error)}
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            process.join(0.2)
            if process.is_alive():
                process.terminate()
                process.join(0.2)
            if process.is_alive():
                process.kill()
                process.join(0.2)
    if not isinstance(result, dict):
        result = {"status": "deferred", "category": "probe-error", "reason": "Probe returned a non-object result."}
    result.update(kind="mast-application-health", endpoint="mast.stsci.edu", tic_id=tic_id,
                  scope="this host and query only; not a global outage or IP-block diagnosis",
                  timeout_seconds=timeout_seconds, wall_seconds=time.monotonic() - started)
    host_logger("trex.doctor").info("MAST application check: %s", result["status"],
                                   extra={"event": "trex.doctor.mast", "status": result["status"],
                                          "category": result.get("category"), "wall_seconds": result["wall_seconds"]})
    return result


def group_failures(paths):
    """Group identical recorded signatures, not guessed common root causes."""
    groups = defaultdict(list)
    invalid = []
    seen = set()
    for supplied in paths:
        path = Path(supplied).resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            diagnosis = payload["operational_diagnosis"]
            key = tuple(diagnosis[name] for name in ("stage", "category", "error_type", "reason"))
            phase = diagnosis.get("phase")
            if not all(isinstance(value, str) for value in key) or (
                    phase is not None and not isinstance(phase, str)):
                raise ValueError("invalid failure signature")
        except (OSError, ValueError, TypeError, KeyError) as error:
            invalid.append({"path": str(path), "error_type": type(error).__name__, "reason": str(error)})
        else:
            if str(path) not in groups[key + (phase,)]:
                groups[key + (phase,)].append(str(path))
    return {"groups": [{"stage": key[0], "category": key[1], "error_type": key[2], "reason": key[3],
                         **({"phase": key[4]} if key[4] is not None else {}),
                         "records": records, "affected_runs": len(records)} for key, records in groups.items()],
            "unreadable_records": invalid}


def main(arguments=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m trex.doctor", description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="Prepared TREX input directory; no inference is run.")
    parser.add_argument("--mast-tic", help="Explicitly enable a real MAST metadata query for this TIC.")
    parser.add_argument("--timeout", type=float, default=15.0, help="MAST wall-clock seconds (maximum 60).")
    parser.add_argument("--failure", action="append", default=[], type=Path, help="Retained failure.json to group (repeatable).")
    args = parser.parse_args(arguments)
    if args.input is None and args.mast_tic is None and not args.failure:
        parser.error("provide input, --mast-tic or --failure")
    reports = {}
    if args.input is not None:
        from trex.app.run import load_input

        started = monotonic()
        try:
            bound, observed, period, exposure = load_input(args.input)
            reports["inputs"] = bound.doctor(observed["time"], observed["flux"], observed["sigma"], period, exptime=exposure)
        except Exception as error:
            reports["inputs"] = {"kind": "trex-input-health", "evidence_draws": 0,
                                 **failure_diagnosis(error, stage="input-loading"),
                                 "wall_seconds": monotonic() - started}
    if args.mast_tic is not None:
        try:
            reports["mast"] = probe_mast(args.mast_tic, timeout_seconds=args.timeout)
        except (ValueError, TypeError) as error:
            reports["mast"] = failure_diagnosis(error, stage="mast-input")
    if args.failure:
        reports["failures"] = group_failures(args.failure)
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0 if all(report.get("status") == "ready" for key, report in reports.items() if key != "failures") and not (
        reports.get("failures", {}).get("groups") or reports.get("failures", {}).get("unreadable_records")) else 1
