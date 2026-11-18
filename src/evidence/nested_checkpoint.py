"""Trusted native dynesty state and completed-result recovery for TREX.

These files are execution state, not evidence of estimator accuracy. Native
dynesty state contains Python pickle: only restore user-controlled files
created by the same pinned runtime, never an untrusted uploaded checkpoint.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import zipfile

import numpy as np

from trex.foundation.errors import TrexEvidenceUnavailableError


class NestedCheckpointPaused(TrexEvidenceUnavailableError):
    """An operational deadline retained state for a later bounded session."""


class CheckpointInputMismatch(ValueError):
    """Retained execution state belongs to different integration inputs."""


checkpoint_context = ContextVar("trex_nested_checkpoint", default=None)


@contextmanager
def nested_checkpoint(path, *, deadline):
    """Use owned recovery files and a real monotonic deadline (+inf disables it)."""
    if (isinstance(deadline, (bool, np.bool_)) or not isinstance(deadline, (int, float, np.integer, np.floating))
            or math.isnan(deadline) or deadline == -math.inf):
        raise ValueError("checkpoint deadline must be a real monotonic time or +inf")
    token = checkpoint_context.set((None if path is None else Path(path), float(deadline)))
    try:
        yield
    finally:
        checkpoint_context.reset(token)


def checkpoint_identity(value):
    # Bind actual inputs/controls to avoid wrong-input reuse. Serialization is
    # one-way here; only the trusted native dynesty restore unpickles anything.
    # Also bind TREX's own source bytes and the tables read inside likelihoods,
    # so changed code or tables never resume state written by another version.
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    root = Path(__file__).resolve().parents[1]
    sources = tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).digest())
        for path in sorted(root.rglob("*.py"))
        if not any(part in {"__pycache__", "build", "dist"} or part.endswith(".egg-info")
                   for part in path.relative_to(root).parts)
    )
    if not sources:
        raise RuntimeError("TREX source files are unavailable for checkpoint identity")
    grid = load_triceratops_tess_grid()
    tables = grid.sha256, grid.supplement_sha256, grid.extension_sha256, grid.phoenix_sha256
    return hashlib.sha256(pickle.dumps((sources, tables, value), protocol=5)).hexdigest()


def _save_archive(path, metadata, **arrays):
    """Publish one complete recovery archive; failed writes leave the old one intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_completed(path, identity, result):
    from trex.evidence._blackjax_types import NativeEvidenceResult

    if isinstance(result, NativeEvidenceResult):
        from trex.evidence._blackjax_checkpoint import save_native

        save_native(path, identity, result)
        return
    values = result._asdict()
    arrays = {}
    for key, value in list(values.items()):
        if isinstance(value, np.ndarray):
            arrays[key] = value
            del values[key]
    _save_archive(path, {"identity": identity, "type": type(result).__name__, "values": values}, **arrays)


def load_completed(path, identity):
    from trex.evidence.nested_controls import ImportanceEvidenceResult, NestedEvidenceResult

    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            if metadata["identity"] != identity:
                raise CheckpointInputMismatch("nested checkpoint inputs do not match this integration")
            if "payload" in stored.files:
                from trex.evidence._blackjax_checkpoint import load_native

                return load_native(path, identity)
            values = metadata["values"]
            values.update({key: np.array(stored[key]) for key in stored.files if key != "metadata"})
        result_type = {"NestedEvidenceResult": NestedEvidenceResult,
                       "ImportanceEvidenceResult": ImportanceEvidenceResult}[metadata["type"]]
        return result_type(**values)
    except CheckpointInputMismatch:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        raise ValueError("nested result checkpoint is corrupt or incomplete") from exc


def load_importance_progress(prefix, identity, size):
    path = Path(prefix).with_suffix(".importance.npz")
    if not path.exists():
        return np.empty(0, dtype=float), []
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            if metadata["identity"] != identity:
                raise CheckpointInputMismatch("importance checkpoint inputs do not match this refinement")
            values = np.array(stored["log_l"], dtype=float)
            executions = metadata["executions"]
        if not isinstance(executions, list) or any(not isinstance(item, dict) for item in executions):
            raise ValueError("importance executions must be a list of records")
    except CheckpointInputMismatch:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        raise ValueError("importance checkpoint is corrupt or incomplete") from exc
    if values.ndim != 1 or values.size > size or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("importance checkpoint progress is incomplete or invalid")
    return values, executions


def save_importance_progress(prefix, identity, values, executions):
    path = Path(prefix).with_suffix(".importance.npz")
    _save_archive(path, {"identity": identity, "executions": executions}, log_l=np.asarray(values, dtype=float))


def load_importance_adaptation(prefix, identity, ndim, rounds, draws):
    """Restore a pass boundary or an ordered likelihood prefix within that pass."""
    path = Path(prefix).with_suffix(".adaptation.npz")
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            if metadata["identity"] != identity:
                raise CheckpointInputMismatch("importance checkpoint inputs do not match this refinement")
            samples = np.array(stored["samples"], dtype=float)
            weights = np.array(stored["weights"], dtype=float)
            values = np.array(stored["log_l"], dtype=float)
        if (samples.ndim != 2 or samples.shape[1] != ndim or samples.shape[0] == 0
                or weights.shape != (len(samples),)
                or not np.isfinite(samples).all() or np.any((samples < 0) | (samples > 1))
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=0.0)):
            raise ValueError("invalid adaptive posterior")
        passes = metadata["passes"]
        if not isinstance(passes, list) or len(passes) > rounds:
            raise ValueError("invalid adaptive passes")
        for record in passes:
            if (not isinstance(record, dict) or type(record["likelihood_calls"]) is not int
                    or not 0 <= record["likelihood_calls"] <= draws):
                raise ValueError("invalid adaptive likelihood count")
            exponent, ess = record["exponent"], record["effective_sample_size"]
            if (exponent is not None and
                    (type(exponent) not in (int, float) or not math.isfinite(exponent) or not 0 <= exponent <= 1)):
                raise ValueError("invalid adaptive exponent")
            if type(ess) not in (int, float) or not math.isfinite(ess) or not 0 <= ess <= draws * (1 + 1e-12):
                raise ValueError("invalid adaptive ESS")
        finished = metadata["finished"]
        if type(finished) is not bool or (not finished and len(passes) >= rounds):
            raise ValueError("invalid adaptive completion state")
        stopped = bool(passes and passes[-1]["exponent"] in (None, 1.0))
        if finished != (stopped or len(passes) == rounds):
            raise ValueError("inconsistent adaptive completion state")
        if (values.ndim != 1 or values.size > draws or np.isnan(values).any()
                or np.isposinf(values).any() or (finished and values.size)):
            raise ValueError("invalid adaptive likelihood prefix")
        # PCG64 state assignment validates its schema without drawing any new points.
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata["rng_state"]
        return samples, weights, passes, rng.bit_generator.state, values, finished
    except CheckpointInputMismatch:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, zipfile.BadZipFile) as exc:
        raise ValueError("importance adaptation checkpoint is corrupt or incomplete") from exc


def save_importance_adaptation(prefix, identity, samples, weights, passes, rng_state, values, *, finished):
    """Keep proposal/RNG state and evaluated values in one atomic archive."""
    path = Path(prefix).with_suffix(".adaptation.npz")
    _save_archive(
        path, {"identity": identity, "passes": passes, "rng_state": rng_state, "finished": finished},
        samples=np.asarray(samples, dtype=float), weights=np.asarray(weights, dtype=float),
        log_l=np.asarray(values, dtype=float),
    )


class SamplerCheckpoint:
    """Atomic native-state recovery with cumulative evaluation accounting."""

    def __init__(self, path, identity, deadline):
        self.path = Path(path)
        self.identity = identity
        self.deadline = deadline
        self.last_saved = None

    def save(self, sampler, calls):
        # Parallel workers carry unconsumed budget reservations in the shared
        # counter; withdraw them so the checkpoint records the exact executed
        # count.  A resume restores that value (restore()), so a paused run
        # must not carry reservation slack across processes or its recorded
        # total would overshoot the continuous run's.
        pool = getattr(sampler, "pool", None)
        if pool is not None and getattr(pool, "_processes", 0) > 0:
            from trex.evidence.nested_sampling import _flush_worker_monitored_counter, _monitored_likelihood_count

            pool.map(_flush_worker_monitored_counter, range(int(pool._processes)), chunksize=1)
            calls = _monitored_likelihood_count()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".pending", dir=self.path.parent)
        os.close(descriptor)
        native = Path(name)
        try:
            sampler.save(str(native))
            payload = native.read_bytes()
            _save_archive(self.path, {"identity": self.identity, "monitored_calls": int(calls),
                                     "payload_sha256": hashlib.sha256(payload).hexdigest()},
                          payload=np.frombuffer(payload, dtype=np.uint8))
        finally:
            native.unlink(missing_ok=True)

    def restore(self, dynesty, pool, counter):
        if not self.path.exists():
            return None
        # ZIP CRC alone cannot detect a changed payload repacked with a fresh CRC.
        # This binding detects corruption, not an untrusted checkpoint's authorship.
        try:
            with np.load(self.path, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata"].item()))
                array = stored["payload"]
                if array.dtype != np.uint8 or array.ndim != 1:
                    raise ValueError("native state must be a byte vector")
                payload = array.tobytes()
            if metadata["payload_sha256"] != hashlib.sha256(payload).hexdigest():
                raise ValueError("native state payload does not match its retained bytes")
            identity = metadata["identity"]
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            raise ValueError("nested sampler checkpoint is corrupt or incomplete") from error
        if identity != self.identity:
            raise CheckpointInputMismatch("nested checkpoint inputs do not match this integration")
        calls = metadata.get("monitored_calls")
        if type(calls) is not int or calls < 0:
            raise ValueError("nested sampler checkpoint has no valid monitored likelihood count")
        descriptor, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".restore", dir=self.path.parent)
        os.close(descriptor)
        native = Path(name)
        try:
            native.write_bytes(payload)
            sampler = dynesty.DynamicNestedSampler.restore(str(native), pool=pool)
        finally:
            native.unlink(missing_ok=True)
        if pool is not None and sampler.queue_size == 1:
            # dynesty's single-item queue passes the parent's Generator itself
            # into proposal evolution. Moving that call into a new process
            # would advance a pickled copy, not the retained parent RNG. Keep
            # this run's proposal evolution serial; the new pool can still
            # execute independent initialization/bound work and later runs.
            sampler.use_pool = dict(sampler.use_pool, propose_point=False)
            sampler.use_pool_evolve = False
            for inner in (sampler.sampler, sampler.batch_sampler):
                if inner is not None:
                    inner.use_pool_evolve = False
        if counter is not None:
            counter.value = calls
        return sampler

    def progress(self, sampler, *, clock, count):
        self.last_saved = clock()

        def check(*_args, **_kwargs):
            now = clock()
            # dynesty cannot exactly resume a transient final-live-point append.
            safe = "ADDLIVE" not in str(sampler.internal_state)
            if safe and (now >= self.deadline or now - self.last_saved >= 60.0):
                self.save(sampler, count())
                self.last_saved = clock()
                if now >= self.deadline:
                    raise NestedCheckpointPaused("nested wall deadline reached; sampler checkpoint retained")
        return check
