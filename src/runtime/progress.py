"""Optional presentation callbacks; never scientific evidence or convergence."""

import threading
import warnings
import weakref
from typing import Any

from trex.runtime.logging import _source_file

# Bound methods are recreated on access; key their owner/function, not the wrapper.
_failed_callbacks = {}
_failure_lock = threading.RLock()
_FAILED_CALLBACK_LIMIT = 128


def _failure_origin(exc: BaseException) -> str:
    """The raise site of a callback failure, or a stated unknown."""
    trace = exc.__traceback__
    if trace is None:
        return "unknown origin"
    while trace.tb_next is not None:
        trace = trace.tb_next
    return f"{_source_file(trace.tb_frame.f_code.co_filename)}:{trace.tb_lineno}"


def _report_failure(callback: Any, exc: BaseException) -> None:
    """Warn once per failing callback, naming the exception type and raise site."""
    try:
        owner = getattr(callback, "__self__", None)
        function = getattr(callback, "__func__", None)
        target = owner if owner is not None and function is not None else callback
        key = (id(target), id(function))
        with _failure_lock:
            if key in _failed_callbacks:
                return
            if len(_failed_callbacks) >= _FAILED_CALLBACK_LIMIT:
                _failed_callbacks.pop(next(iter(_failed_callbacks)))
            try:
                def forget(_reference):
                    with _failure_lock:
                        _failed_callbacks.pop(key, None)
                reference = weakref.ref(target, forget)
            except TypeError:
                # Retain non-weakrefable callables only within the bounded registry.
                reference = target
            _failed_callbacks[key] = reference
    except Exception:  # noqa: BLE001 - broken introspection must not abort computation
        pass
    try:
        warnings.warn(
            f"progress callback failed: {type(exc).__name__} at {_failure_origin(exc)}",
            RuntimeWarning,
            stacklevel=3,  # Attribute the warning to the engine's emit_progress call.
        )
    except Exception:  # noqa: BLE001 - a warnings-as-errors policy must not abort the run
        return


def emit_progress(
    callback: Any, *args: Any, **update: Any
) -> None:
    """Deliver one progress update, reporting ordinary callback exceptions.

    Engines call this instead of invoking a caller-supplied callback directly
    so a presentation defect cannot abort a scientific run.
    Only measured quantities belong in ``update``: an engine that cannot
    measure its own completion reports ``total=None`` rather than a guess.

    Failures carry type and normalized raise site, not callback text or paths.
    Deduplication is per live callable/owner-function within 128 recent failures.
    Each callback is invoked once; a TypeError inside it is never retried.
    Caller-specific legacy signature adaptation belongs to the caller.
    """
    if callback is None:
        return
    try:
        callback(*args, **update)
    except Exception as exc:  # noqa: BLE001 - presentation must not abort a run
        _report_failure(callback, exc)


__all__ = ["emit_progress"]
