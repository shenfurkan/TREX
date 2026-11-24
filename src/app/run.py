"""Run one prepared target: `python -m trex.run INPUT_DIRECTORY OUTPUT_DIRECTORY`.

INPUT_DIRECTORY holds the prepared files from a calibrated scene producer:
`trex-input.json` (scene, binding, period in days), `arrays.npz` (relative time in days, normalized
flux, sigma, bin integration in days) and `background.csv`. Results or `failure.json` go to the output
directory; a run TREX refuses exits non-zero, so unattended callers see it.

Exit status distinguishes the lifecycle stages:

* 0: the calculation completed with a reported FPP and NFPP.
* 1: execution failed, including failure-record publication; exceptions retain the original cause.
* 2: the input, checkpoint or output was refused before a result existed; a new output retains
  failure.json with the original exception type, reason and any diagnostics when publication succeeds.
* 3: the calculation ran but required evidence stayed unresolved; failure.json (or a non-ok
  summary.json) retains the reason and no FPP/NFPP is reported as a success.
* 75: escalation paused at a retained checkpoint; no output is created and rerunning the same
  command with the same inputs and budget continues it.

Existing outputs are protected; a failure leaves an existing empty resumable output untouched.
Publication errors propagate rather than returning refusal or deferral, so failure.json is not guaranteed.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time

import numpy as np

from trex.app.scene import TargetScene
from trex.app.target import _json_safe, target
from trex.evidence.nested_checkpoint import NestedCheckpointPaused
from trex.evidence.nested_evidence import _selected_nested_estimator
from trex.evidence.nested_controls import _native_kernel_policy_identity, _selected_importance_proposal
from trex.foundation.environment import read_environment
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.runtime.logging import host_logger
from trex.acquisition.storage import _sync_directory, write_json

INPUT_FILES = ("trex-input.json", "arrays.npz", "background.csv")

EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_DEFERRED = 3
EXIT_PAUSED = 75


def load_input(directory):
    """Load all required arrays and bind absolute file identities, accepting a relative directory.

    Symlinked/nonregular inputs are refused before reading; missing paths retain
    FileNotFoundError. TargetScene re-verifies the background bytes.
    """
    directory = Path(directory).absolute()
    if directory.is_symlink() or any((directory / name).is_symlink() for name in INPUT_FILES):
        raise ValueError("prepared inputs must be regular files in a non-symlink directory")
    for name in INPUT_FILES:
        if not stat.S_ISREG((directory / name).stat().st_mode):
            raise ValueError("prepared inputs must be regular files in a non-symlink directory")
    payload = json.loads((directory / INPUT_FILES[0]).read_text(encoding="utf-8"))
    if payload.get("version", 1) not in (1, 2):
        raise ValueError("unsupported prepared-input version")
    payload["scene"]["trilegal_cache"] = str(directory / INPUT_FILES[2])
    binding = payload["binding"]
    binding["sector_weights"] = {int(sector): weight for sector, weight in binding["sector_weights"].items()}
    with np.load(directory / INPUT_FILES[1], allow_pickle=False) as arrays:
        missing = sorted({"time", "flux", "sigma", "bin_integration_days"} - set(arrays.files))
        if missing:
            raise ValueError("prepared arrays.npz is missing required arrays: " + ", ".join(missing))
        observed = {name: arrays[name] for name in arrays.files}
    binding["bin_integration_days"] = observed["bin_integration_days"].tolist()
    binding["input_artifacts"] = {
        str(directory / name): hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in INPUT_FILES
    }
    if payload.get("version") == 2:
        from trex.app.observation_io import load_observation_binding

        physical, assets = load_observation_binding(directory, payload)
        binding.update(physical)
        binding["input_artifacts"].update(assets)
        joint = physical["joint_observations"]
        first = joint.likelihood.observations.observations[0]
        if (np.any(observed["bin_integration_days"] != 0) or any(
                not np.array_equal(observed[key], expected) for key, expected in (
                    ("time", joint.likelihood.times[0] - joint.epoch_days),
                    ("flux", first.flux), ("sigma", first.sigma)))):
            raise ValueError("portable proposal arrays differ from the first bound observation")
    bound = target.from_scene(TargetScene.from_dict(payload["scene"]), **binding)
    return bound, observed, payload["period_days"], binding["exptime_days"]


def _failure_record(error, *, stage):
    """Describe a refused or failed run without inventing probabilities or a success status."""
    from trex.app.doctor import failure_diagnosis
    from trex.stellar_light.host_limb_darkening import summarize_ldc_grid_provenance

    record = {
        "claim_eligible": False,
        "error_type": type(error).__name__,
        "reason": str(error),
        "operational_diagnosis": failure_diagnosis(error, stage=stage),
        "limb_darkening_grid": summarize_ldc_grid_provenance([]),
    }
    diagnostics = getattr(error, "diagnostics", None)
    if diagnostics:
        record["diagnostics"] = copy.deepcopy(diagnostics)
    return record


def _write_failure(output, error, *, stage):
    """Return ``(path, protection_reason)``; raise publication failures from ``error``.

    The directory is created exclusively, so an earlier completed, failed or
    deferred output is never overwritten by a later refusal.
    """
    try:
        try:
            output.mkdir()
        except FileExistsError:
            return None, f"output directory already exists: {output}"
        record = _failure_record(error, stage=stage)
        target_path = output / "failure.json"
        write_json(target_path, _json_safe(record))
    except Exception as problem:
        raise problem from error
    return target_path, None


def _output_refusal(output, handoff_path):
    """Return why the output cannot be used, or None for a new or empty resumable output."""
    if output.is_symlink():
        return f"output path is a symlink, not a directory: {output}"
    if not output.exists():
        return None
    if not output.is_dir():
        return f"output path exists and is not a directory: {output}"
    entries = sorted(path.name for path in output.iterdir())
    if entries:
        return (f"output directory already contains {', '.join(entries)}; a completed, failed or deferred "
                f"output is never overwritten and rerunning cannot continue it in place. Choose a new "
                f"output directory; rerunning with it and this checkpoint reuses any retained recovery "
                f"state under {handoff_path.parent}.")
    if not handoff_path.is_file():
        return (f"output directory is empty but has no Monte Carlo handoff at {handoff_path}; it is not "
                "resumable. Choose a new output directory, or restore the handoff the paused run wrote.")
    return None


def _handoff_json(value):
    """Retain nonfinite numerical state losslessly in strict JSON, unlike display strings."""
    if isinstance(value, dict):
        return {key: _handoff_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _handoff_json(value.tolist())
    if isinstance(value, np.generic):
        return _handoff_json(value.item())
    if isinstance(value, (list, tuple)):
        return [_handoff_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__trex_float__": str(value)}
    return value


def _restore_handoff_float(value):
    if set(value) == {"__trex_float__"}:
        token = value["__trex_float__"]
        if token not in ("-inf", "inf", "nan"):
            raise ValueError("invalid retained nonfinite float token")
        return float(token)
    return value


def _publish_handoff(checkpoint, handoff_path, binding, state):
    """Atomically link checked Monte Carlo state; an earlier handoff is never replaced."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=checkpoint, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(_handoff_json({"binding": binding, "state": state}), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, handoff_path)
            _sync_directory(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)


def _retain_failure(bound, output, error, *, resumable, stage):
    """Publish or deliberately protect resumable state; propagate publication failures."""
    if resumable:
        return None, "the empty output is retained for resume"
    try:
        if bound.failure_record is not None:
            bound.write_outputs(output)
            return output / "failure.json", None
    except Exception as problem:
        raise problem from error
    return _write_failure(output, error, stage=stage)


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m trex.run", description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=1_000_000, help="below 1,000,000 is for software tests only")
    parser.add_argument("--nsamples", type=int, default=20)
    parser.add_argument("--escalation-checkpoint-directory", type=Path,
                        help="trusted recovery directory (default: OUTPUT_DIRECTORY.escalation)")
    args = parser.parse_args(arguments)
    output = args.output
    if args.escalation_checkpoint_directory is not None:
        checkpoint = args.escalation_checkpoint_directory
    elif output.name:
        checkpoint = output.with_name(output.name + ".escalation")
    else:
        # Path(".") and Path("/") have no name for with_name(); refuse them as occupied outputs.
        checkpoint = output / "trex-output.escalation"
    handoff_path = checkpoint / "monte-carlo.json"
    output_present = output.exists() or output.is_symlink()

    def refuse(error, *, stage, message):
        """Report a refusal and publish it only when no earlier output can be overwritten."""
        host_logger("trex.run").warning(
            "TREX input refused", exc_info=(type(error), error, error.__traceback__),
            extra={"event": "trex.run.refused", "stage": stage, "status": "refused",
                   "error_type": type(error).__name__, "exit_code": EXIT_REFUSED},
        )
        published, problem = (None, "the existing output was left untouched")
        if not output_present:
            published, problem = _write_failure(output, error, stage=stage)
        location = (f"record: {published}; retry with a new output directory"
                    if published is not None else f"no record: {problem}")
        print(f"TREX refused ({stage}): {message}: {type(error).__name__}: {error}; {location}",
              file=sys.stderr, flush=True)
        return EXIT_REFUSED

    try:
        checkpoint_inside_output = checkpoint.resolve().is_relative_to(output.resolve())
    except (OSError, RuntimeError) as error:
        return refuse(error, stage="checkpoint",
                      message=f"checkpoint directory {checkpoint} cannot be resolved")
    if checkpoint_inside_output:
        return refuse(ValueError(f"checkpoint directory {checkpoint} must be outside the output directory {output}"),
                      stage="checkpoint", message=f"checkpoint directory {checkpoint} is inside the output")

    refusal = _output_refusal(output, handoff_path)
    if refusal:
        print(f"TREX refused (output): {refusal}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    resumable = output_present

    try:
        bound, observed, period, exposure = load_input(args.input)
    except Exception as error:
        return refuse(error, stage="input-loading",
                      message=f"input directory {args.input} was not accepted")

    try:
        if checkpoint.is_symlink():
            raise ValueError(f"checkpoint directory {checkpoint} must not be a symlink")
        checkpoint.mkdir(parents=True, exist_ok=True, mode=0o700)
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_uid != os.geteuid() or stat.S_IMODE(checkpoint_stat.st_mode) & 0o077:
            raise ValueError(f"checkpoint directory {checkpoint} must be owned by the current user "
                             "and inaccessible to others")
    except Exception as error:
        return refuse(error, stage="checkpoint",
                      message=f"checkpoint directory {checkpoint} was not accepted")

    try:
        binding = {
            "inputs": {Path(path).name: digest for path, digest in bound._require_scene()["input_artifacts"].items()},
            "draws": args.draws, "nsamples": args.nsamples, "estimator": _selected_nested_estimator(),
        }
        # Only a non-default opt-in changes the handoff identity, so retained
        # default controls stay resumable while a policy change cannot reuse them.
        if binding["estimator"] == "blackjax":
            policy = _native_kernel_policy_identity()
            if policy:
                binding["blackjax_policy"] = list(policy)
        proposal = _selected_importance_proposal()
        if proposal != "kernel-mixture":
            binding["importance_proposal"] = proposal
    except Exception as error:
        return refuse(error, stage="input-binding",
                      message="bound input identity or calculation controls were not accepted")

    state_in = None
    if handoff_path.exists():
        try:
            if handoff_path.is_symlink() or not handoff_path.is_file():
                raise ValueError("retained Monte Carlo handoff must be a regular non-symlink file")
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"), object_hook=_restore_handoff_float)
            if (not isinstance(handoff, dict) or not isinstance(handoff.get("binding"), dict)
                    or not isinstance(handoff.get("state"), dict)):
                raise ValueError(f"retained handoff {handoff_path} is malformed; inspect the checkpoint directory")
            retained_binding = handoff["binding"]
            if retained_binding != binding:
                changed = sorted(name for name in set(binding) | set(retained_binding)
                                 if binding.get(name) != retained_binding.get(name))
                raise ValueError(f"checkpoint inputs or calculation controls changed "
                                 f"({', '.join(changed)}) at {checkpoint}; use a new checkpoint directory")
            state_in = handoff["state"]
        except Exception as error:
            return refuse(error, stage="checkpoint",
                          message=f"retained handoff {handoff_path} was not accepted")
        print(f"resuming Monte Carlo handoff: {handoff_path}", flush=True)
    elif any(checkpoint.iterdir()):
        return refuse(ValueError(f"checkpoint directory {checkpoint} has no Monte Carlo handoff; inspect it"),
                      stage="checkpoint", message=f"checkpoint directory {checkpoint} was not accepted")

    try:
        state_out = {} if state_in is None else None
        options = dict(exptime=exposure, N=args.draws, nsamples=args.nsamples, parallel=True,
                       escalation_checkpoint_directory=str(checkpoint))
        # A host session deadline (same machine's monotonic clock, never persisted) pauses before the host is cut.
        host = read_environment("TREX_ESCALATION_DEADLINE_MONOTONIC",
                                "EXONYM_TREX_ESCALATION_DEADLINE_MONOTONIC")
        host_deadline = float(host) if host else None
        if host_deadline is not None and not math.isfinite(host_deadline):
            raise ValueError("host escalation deadline must be a finite monotonic time")
        bound.calc_probs(observed["time"], observed["flux"], observed["sigma"], period,
                         escalation_state_in=state_in, escalation_state_out=state_out,
                         escalation_deadline_monotonic=host_deadline, **options)
        if state_out:
            # A split in this process shares one wall budget; monotonic deadlines are not persisted.
            deadline = time.monotonic() + state_out.pop("session_remaining_seconds")
            if host_deadline is not None:
                deadline = min(deadline, host_deadline)
            # Persist checked MC state before starting the nested stage.
            _publish_handoff(checkpoint, handoff_path, binding, state_out)
            if state_out.get("mc_importance_paused") or time.monotonic() >= deadline:
                print(f"TREX checkpoint pause: escalation paused; no output was created. Rerun the same "
                      f"command with the same inputs and budget; checkpoint directory {checkpoint}", flush=True)
                return EXIT_PAUSED
            bound.calc_probs(observed["time"], observed["flux"], observed["sigma"], period,
                             escalation_state_in=state_out, escalation_deadline_monotonic=deadline, **options)
    except NestedCheckpointPaused as error:
        print(f"TREX checkpoint pause: {error}; no output was created. Rerun the same command with the "
              f"same inputs and budget; checkpoint directory {checkpoint}", flush=True)
        return EXIT_PAUSED
    except TrexEvidenceUnavailableError as error:
        from trex.app.doctor import failure_diagnosis

        diagnosis = ((bound.failure_record.get("operational_diagnosis")
                      if isinstance(bound.failure_record, dict) else None)
                     or failure_diagnosis(error, stage="evidence"))
        record, problem = _retain_failure(bound, output, error, resumable=resumable, stage="evidence")
        location = (f"record: {record}; retry with a new output directory"
                    if record is not None else f"no record written: {problem}")
        if diagnosis.get("status") == "failed":
            # A recorded numerical likelihood failure is an execution failure; it
            # must not be reported as an under-sampled scientific deferral.
            error.add_note(f"TREX failure record: {record}" if record is not None
                           else f"TREX failure record not written: {problem}")
            raise
        host_logger("trex.run").warning(
            "TREX calculation stopped: %s", "deferred", exc_info=(type(error), error, error.__traceback__),
            extra={"event": "trex.run.deferred", "stage": "evidence", "status": "deferred",
                   "error_type": type(error).__name__, "exit_code": EXIT_DEFERRED},
        )
        print(f"TREX scientific deferral: {error}; no FPP/NFPP was produced; {location}",
              file=sys.stderr, flush=True)
        return EXIT_DEFERRED
    except Exception as error:
        record, problem = _retain_failure(bound, output, error, resumable=resumable, stage="execution")
        location = (f"record: {record}; retry with a new output directory"
                    if record is not None else f"no record written: {problem}")
        from trex.app.doctor import failure_diagnosis

        diagnosis = ((bound.failure_record.get("operational_diagnosis")
                      if isinstance(bound.failure_record, dict) else None)
                     or failure_diagnosis(error, stage="target-api"))
        if diagnosis.get("category") == "invalid-input":
            host_logger("trex.run").warning(
                "TREX input refused", exc_info=(type(error), error, error.__traceback__),
                extra={"event": "trex.run.refused", "stage": "target-api", "status": "refused",
                       "error_type": type(error).__name__, "exit_code": EXIT_REFUSED},
            )
            print(f"TREX refused (input): calculation inputs were not accepted: "
                  f"{type(error).__name__}: {error}; {location}", file=sys.stderr, flush=True)
            return EXIT_REFUSED
        error.add_note(f"TREX failure record: {record}" if record is not None
                       else f"TREX failure record not written: {problem}")
        raise

    result = bound.trex_result
    if result is None:
        error = RuntimeError("TREX calculation returned without a result; no probabilities were published")
        record, problem = _retain_failure(bound, output, error, resumable=resumable, stage="execution")
        error.add_note(f"TREX failure record: {record}" if record is not None
                       else f"TREX failure record not written: {problem}")
        raise error
    if result.status == "ok" and (bound.FPP is None or bound.NFPP is None):
        error = RuntimeError("TREX calculation reported ok without FPP/NFPP; no success was published")
        record, problem = _retain_failure(bound, output, error, resumable=resumable, stage="execution")
        error.add_note(f"TREX failure record: {record}" if record is not None
                       else f"TREX failure record not written: {problem}")
        raise error
    bound.write_outputs(output, existing_empty=resumable)
    if result.status != "ok":
        print(f"status={result.status}; no complete FPP/NFPP pair was reported; see {output}", flush=True)
        return EXIT_DEFERRED
    print(f"status={result.status} FPP={bound.FPP} NFPP={bound.NFPP}; see {output}", flush=True)
    return EXIT_COMPLETED


if __name__ == "__main__":
    raise SystemExit(main())
