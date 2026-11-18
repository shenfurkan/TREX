"""Device-scoped, crash-recoverable accelerator admission for independent jobs.

Leases cover a computation, not data acquisition or the candidate's lifetime.
There is no global single-GPU lock or vendor/model capacity rule. A standalone
computation takes all idle devices unless its caller supplies a device-count cap.
Unavailable identities or contested leases carry explicit reasons, never an
unrecorded GPU-to-CPU change.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from numbers import Real

from trex.foundation.environment import read_environment
from trex.runtime.hardware import AcceleratorLease, driver_memory_profile
from trex.runtime.logging import _device, _field, _metric, host_logger

# A shared host or embedding supervisor divides the visible devices with this cap.
DEVICE_SHARE_ENVIRONMENT_VARIABLE = "TREX_ACCELERATOR_MAX_DEVICES"
_active_devices = ContextVar("trex_execution_devices", default=None)
_active_admission = ContextVar("trex_accelerator_admission", default=None)
_admission_observer = ContextVar("trex_admission_observer", default=None)
_EXECUTION_EVENT_LIMIT = 256
_EXECUTION_METRICS = frozenset({
    "batch", "in_flight_blocks", "retry_blocks", "bytes_required", "available_budget",
    "draws", "seconds", "cpu_seconds", "accelerator_seconds", "cpu_draws_per_second",
    "accelerator_draws_per_second", "device_budget_bytes", "host_available_bytes",
    "compiled_block_bytes", "output_block_bytes", "resident_input_bytes", "capacity_blocks",
    "effective_batch_draws", "kernel_lanes_per_second",
})
_EXECUTION_KINDS = frozenset({
    "device-memory-unavailable", "evaluation-block-memory-retry", "device-pilot",
    "compiler-memory-unavailable", "allocation-memory-retry", "adaptive-block-wave",
    "pipeline-memory-retry", "compiler-memory-retry", "native-unit-block-memory-retry",
    "native-unit-compiler-memory-unavailable", "native-unit-compiler-memory-retry",
    "native-unit-memory-headroom-unavailable", "native-unit-measured-memory-plan",
    "native-unit-compiled-memory", "mc-pilot-importance-device-pilot",
})


def record_accelerator_event(kind, device, **details):
    """Retain bounded operational metrics, never arbitrary backend error text."""
    admission = _active_admission.get()
    if admission is not None:
        event = {"kind": kind if kind in _EXECUTION_KINDS else "<omitted>", "device": _device(str(device))}
        for key, value in details.items():
            if key in {"kernel", "chosen", "reason", "pilot", "cpu_reference_overlap",
                       "cpu_timing_condition", "accelerator_timing_condition", "error_type", "parity"}:
                event[key] = _field(key, value)
            elif key in _EXECUTION_METRICS and (value is None or isinstance(value, Real)):
                event[key] = _metric(value)
        events = admission.setdefault("execution_events", [])
        if len(events) < _EXECUTION_EVENT_LIMIT:
            events.append(event)
        else:
            admission["execution_events_omitted"] = admission.get("execution_events_omitted", 0) + 1
            key = event["kind"] + ":" + event["device"]
            aggregate = admission.setdefault("execution_event_overflow", {}).setdefault(key, {"count": 0})
            aggregate.update(last=event, count=aggregate["count"] + 1)


@contextmanager
def observe_accelerator_admissions(observer):
    """Install a caller-owned context manager receiving each scope's mutable record."""
    token = _admission_observer.set(observer)
    try:
        yield
    finally:
        _admission_observer.reset(token)


def execution_devices(devices):
    """Intersect with the enclosing allocation, preserving the requested device order."""
    active = _active_devices.get()
    return tuple(device for device in devices if device in active) if active is not None else tuple(devices)


@contextmanager
def accelerator_execution_scope(devices):
    """Acquire leases in stable identity order, preserving caller execution order."""
    devices = execution_devices(devices)
    if _active_devices.get() is not None:
        yield devices
        return
    override = read_environment(
        DEVICE_SHARE_ENVIRONMENT_VARIABLE, "EXONYM_ACCELERATOR_MAX_DEVICES"
    )
    override = "" if override is None else override.strip()
    maximum = len(devices)
    if override:
        try:
            maximum = int(override)
        except ValueError as exc:
            raise ValueError(f"{DEVICE_SHARE_ENVIRONMENT_VARIABLE} must be a positive integer") from exc
        if maximum < 1:
            raise ValueError(f"{DEVICE_SHARE_ENVIRONMENT_VARIABLE} must be a positive integer")
    cpu_devices = tuple(device for device in devices if device.platform in ("cpu", "interpreter"))
    profiles = [(driver_memory_profile(device), device) for device in devices if device not in cpu_devices]
    identities = [(profile.get("device_key"), index, device)
                  for index, (profile, device) in enumerate(profiles)]
    leases, acquired = [], []
    reasons = []
    token = None
    admission_token = None
    admission = {"release_reasons": []}
    try:
        for identity, _, device in sorted(identities, key=lambda pair: pair[0] or ""):
            if identity is None:
                reasons.append(f"{device}: stable driver identity unavailable")
                continue
            lease = AcceleratorLease(device_key=identity)
            try:
                leased = lease.acquire()
            except OSError:
                leased = False
                lease.reason = "lease metadata or temporary-file cleanup failed"
            if leased:
                leases.append(lease)
                acquired.append(device)
                if len(acquired) >= maximum:
                    break
            else:
                if lease.acquired:
                    leases.append(lease)
                reasons.append(f"{device}: {lease.reason}")
        acquired = [device for device in devices if device in acquired]
        if not devices:
            reasons.append("no execution devices supplied")
        elif not profiles:
            reasons.append("only CPU execution devices supplied")
        if profiles and reasons or not devices:
            host_logger("hardware").warning(
                "Accelerator admission incomplete", extra={"event": "trex.hardware.admission",
                "count": len(reasons), "reason": "accelerator admission incomplete"},
            )
        if profiles and not acquired:
            host_logger("hardware").warning("No idle identifiable accelerator; CPU execution selected",
                                            extra={"event": "trex.hardware.fallback"})
        admitted = tuple(device for device in devices if device in acquired or device in cpu_devices)
        admission = {
            "devices": [dict(device=str(device), **profile) for profile, device in profiles],
            "acquired_devices": [str(device) for device in acquired],
            "maximum_devices": maximum, "reasons": reasons,
            "cpu_fallback": not acquired,
            "release_reasons": admission["release_reasons"],
        }
        admission_token = _active_admission.set(admission)
        token = _active_devices.set(admitted)
        observer = _admission_observer.get()
        with observer(admission) if observer is not None else nullcontext():
            yield admitted
    finally:
        if token is not None:
            _active_devices.reset(token)
        if admission_token is not None:
            _active_admission.reset(admission_token)
        for lease in reversed(leases):
            try:
                lease.release()
            except OSError:
                # Continue releasing peers even if one sidecar cannot be closed.
                lease.reason = "lease metadata cleanup failed; ownership may be retained"
            if lease.reason:
                admission["release_reasons"].append(lease.reason)
                host_logger("hardware").warning(
                    "Accelerator lease cleanup incomplete",
                    extra={"event": "trex.hardware.lease_cleanup"},
                )


__all__ = [
    "DEVICE_SHARE_ENVIRONMENT_VARIABLE",
    "accelerator_execution_scope",
    "execution_devices",
    "observe_accelerator_admissions",
    "record_accelerator_event",
]
