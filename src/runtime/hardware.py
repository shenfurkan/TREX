"""Portable execution policy and observable host resources, not certification.

CPU pools use affinity/cgroup capacity minus the two-CPU host reservation,
floored at one. Explicit positive process budgets are honoured, not silently
clamped. RAM is measured independently: without a measured workload peak RSS
it cannot honestly determine the number of workers. Cohorts measure an isolated
pilot and use live admission; optional CapacityEvidence can seed that estimate.

GPU capacity is never proof of speed or FP64 fitness. Driver-free VRAM and
allocator headroom are distinct counters; both bound working sets. Unknown
counters remain unknown; driver failures carry reasons; invalid policy raises.
The legacy public API is retained. Colab/WSL/container names select no guessed
resource profile: actual OS counters determine capacity on each invocation.

Kernel resource semantics: see hardware_resources. JAX allocation semantics:
https://docs.jax.dev/en/latest/gpu_memory_allocation.html (JAX 0.4.x).
Counts are processes, memory is bytes. No draw count, precision, convergence
threshold, lifecycle, or claim eligibility is changed by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Optional

from trex.foundation.environment import read_environment
from trex.runtime.hardware_drivers import compute_capability, driver_memory_profile
from trex.runtime.hardware_lock import PRIVATE_FILE_MODE, lease_metadata_lock
from trex.runtime.hardware_resources import cpu_resource_profile, memory_resource_profile
from trex.acquisition.storage import _sync_directory

# Logical CPUs held back so the operating system and user processes stay responsive.
RESERVED_LOGICAL_CPU_COUNT: int = 2

# The per-process-tree override. Set it to divide the machine deliberately
# (nested pools, a shared host, a CI runner); leave it unset for the derived
# value.
WORKER_COUNT_ENVIRONMENT_VARIABLE: str = "TREX_N_JOBS"

# No device-model/capacity ban by default. Workload-sized batches determine what
# fits; a working small GPU must not be excluded by a benchmark on another host.
# A user can still explicitly require a minimum addressable capacity.
MINIMUM_ACCELERATOR_MEMORY_BYTES: int = 0

# Per-process-tree override for the accelerator memory floor. ``0`` disables
# the floor entirely and engages any device JAX reports.
ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE: str = "TREX_GPU_MIN_MEMORY_BYTES"

# User override for where the host accelerator lease
# lives. Unset, the lease is a host-keyed file in the system temporary
# directory.
ACCELERATOR_LEASE_PATH_ENVIRONMENT_VARIABLE: str = "TREX_ACCELERATOR_LEASE_PATH"

# /proc/<pid>/stat field 22; the suffix after '(comm)' starts at field 3.
_PROC_STAT_START_TIME_INDEX = 22 - 3


def _process_start_time(pid: int) -> Optional[str]:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return fields[_PROC_STAT_START_TIME_INDEX]
    except (OSError, UnicodeError, IndexError):
        return None


def available_cpu_count() -> int:
    """Affinity-visible logical CPUs, bounded by visible cgroup CPU quotas."""
    return cpu_resource_profile()["effective_cpu_count"]


def get_default_n_jobs() -> int:
    """Worker processes every CPU-bound run uses unless told otherwise.

    ``TREX_N_JOBS`` wins when set; otherwise the scheduler-visible CPU count
    less :data:`RESERVED_LOGICAL_CPU_COUNT`, floored at one.

    Raises:
        ValueError: If ``TREX_N_JOBS`` is set to something that is not a
            positive integer.
    """
    return _worker_count(available_cpu_count())


def _worker_count(available: int) -> int:
    override = read_environment(WORKER_COUNT_ENVIRONMENT_VARIABLE, "EXONYM_N_JOBS")
    if override is not None and override.strip():
        try:
            parsed = int(override.strip())
        except ValueError as exc:
            raise ValueError(
                "{0} must be a positive integer, got {1!r}".format(
                    WORKER_COUNT_ENVIRONMENT_VARIABLE, override
                )
            ) from exc
        if parsed < 1:
            raise ValueError(
                "{0} must be a positive integer, got {1!r}".format(
                    WORKER_COUNT_ENVIRONMENT_VARIABLE, override
                )
            )
        return parsed
    return max(1, available - RESERVED_LOGICAL_CPU_COUNT)


def resolve_n_jobs(requested: Optional[int]) -> int:
    """Validate an explicit worker count, or supply the hardware default.

    Raises:
        ValueError: If ``requested`` is not ``None`` and not a positive
            integer.
    """
    if requested is None:
        return get_default_n_jobs()
    if isinstance(requested, bool) or not isinstance(requested, Integral) or requested < 1:
        raise ValueError("n_jobs must be a positive integer or None")
    return int(requested)


def accelerator_arithmetic_profile(device: Any) -> Dict[str, Any]:
    """Read-only architecture metadata; it never predicts FP64 throughput.

    ``compute_capability`` is the CUDA runtime's architecture pair or None.
    That pair identifies an architecture, not a measured FP64:FP32 ratio, so
    ``fp64_throughput_class`` stays ``"unknown"`` until a workload is profiled.
    Device admission, leases and memory floors are neither consulted nor changed.
    """
    capability = compute_capability(device)
    pair = None
    if capability["major"] is not None and capability["minor"] is not None:
        pair = (capability["major"], capability["minor"])
    return {
        "compute_capability": pair,
        "capability_source": capability["source"],
        "capability_reason": capability["reason"],
        "fp64_throughput_class": "unknown",
        "fp64_throughput_reason": (
            "compute capability identifies an architecture, not a measured FP64:FP32 "
            "throughput ratio; profile the workload"
        ),
    }


def accelerator_memory_floor_bytes() -> int:
    """Device memory a GPU must report before a run will use it.

    ``TREX_GPU_MIN_MEMORY_BYTES`` wins when set; ``0`` disables the floor.

    Raises:
        ValueError: If the override is set to something that is not a
            non-negative integer.
    """
    override = read_environment(
        ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE, "EXONYM_GPU_MIN_MEMORY_BYTES"
    )
    if override is not None and override.strip():
        try:
            parsed = int(override.strip())
        except ValueError as exc:
            raise ValueError(
                "{0} must be a non-negative integer, got {1!r}".format(
                    ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE, override
                )
            ) from exc
        if parsed < 0:
            raise ValueError(
                "{0} must be a non-negative integer, got {1!r}".format(
                    ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE, override
                )
            )
        return parsed
    return MINIMUM_ACCELERATOR_MEMORY_BYTES


def _byte_counter(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if not isinstance(value, Integral) and (not math.isfinite(value) or value != int(value)):
        return None
    return int(value) if value >= 0 else None


def device_memory_profile(device: Any) -> Dict[str, Any]:
    """Sample byte counters; GPU availability needs driver-free VRAM.

    Allocator limits/headroom remain distinct from driver totals/free bytes.
    CPU availability uses OS/cgroup RAM, matching device_memory_available_bytes.
    Missing counters stay None; memory_known refers to the allocator limit.
    """
    reported = None
    reason = None
    try:
        stats = getattr(device, "memory_stats", None)
        reported = stats() if callable(stats) else None
    except Exception:  # noqa: BLE001 - backend refusals stay unknown
        reason = "allocator memory probe failed"
    reported = reported if isinstance(reported, dict) else {}
    limit = _byte_counter(reported.get("bytes_limit"))
    used = _byte_counter(reported.get("bytes_in_use"))
    headroom = max(0, limit - used) if limit is not None and used is not None else None
    driver = driver_memory_profile(device)
    free = _byte_counter(driver.get("free_bytes"))
    pool = _byte_counter(reported.get("pool_bytes"))
    # Driver-free bytes exclude our retained pool. Only measured unused pool
    # bytes can be added back; do not confuse bytes_limit with reserved memory.
    reusable = max(0, pool - used) if pool is not None and used is not None else 0
    bounds = [value for value in (headroom, free + reusable if free is not None else None) if value is not None]
    platform = getattr(device, "platform", None)
    available = min(bounds) if bounds else None
    if platform not in (None, "cpu", "interpreter") and free is None:
        available = None  # allocator headroom cannot replace missing driver-free VRAM
    if platform in ("cpu", "interpreter"):
        memory = memory_resource_profile()
        available = memory["available_bytes"]
        reason = memory["reason"]
    elif available is None:
        reason = reason or "available device memory is unknown"
    return {
        "memory_limit_bytes": limit,
        "bytes_in_use": used,
        "allocator_headroom_bytes": headroom,
        "driver_total_bytes": driver.get("total_bytes"),
        "driver_free_bytes": free,
        "driver_memory_source": driver.get("source"),
        "driver_probe_reason": driver.get("reason"),
        "device_key": driver.get("device_key"),
        "available_bytes": available,
        "memory_known": limit is not None,
        "memory_probe_reason": reason or (
            "backend supplies no valid memory limit" if limit is None and platform not in ("cpu", "interpreter") else None),
    }


def device_memory_limit_bytes(device: Any) -> Optional[int]:
    """Addressable JAX allocator capacity, or ``None`` when not reported."""
    return device_memory_profile(device)["memory_limit_bytes"]


def device_memory_available_bytes(device: Any) -> Optional[int]:
    """Driver-free VRAM bounded by allocator headroom; unknown stays unknown."""
    return device_memory_profile(device)["available_bytes"]


def configure_jax_memory() -> Dict[str, Any]:
    """Request demand allocation before backend initialization, preserving overrides.

    Shared notebooks and display GPUs must not have most of their VRAM seized
    by a probe/first array. This avoids JAX's default preallocation, trading
    some allocation overhead/fragmentation risk for coexistence. A user
    can explicitly enable preallocation for a dedicated, benchmarked device.
    Already imported JAX is reported: in a running notebook this setting may
    require a kernel restart to take effect. No private JAX state is guessed.
    """
    already_imported = "jax" in sys.modules
    previous = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    previous_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Demand allocation is not a reservation. Permit the workload planner to
    # address the whole device instead of inheriting JAX's implicit 75% cap;
    # real working sets still leave measured headroom and back off on OOM.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
    return {
        "preallocate": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "memory_fraction": os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
        "memory_fraction_existing_setting_preserved": previous_fraction is not None,
        # An existing setting may come from the user or an earlier policy
        # call; the environment alone cannot establish who set it.
        "existing_setting_preserved": previous is not None,
        "jax_already_imported": already_imported,
        "reason": "JAX was already imported; allocator setting may require a kernel restart"
        if already_imported and (previous is None or previous_fraction is None) else None,
    }


def accelerator_meets_memory_floor(device: Any) -> bool:
    """Whether capacity satisfies policy (not a prediction of runtime).

    Unknown capacity passes, but does not establish available working memory;
    callers must still size a workload from measured counters.
    """
    floor = accelerator_memory_floor_bytes()
    if floor <= 0:
        return True
    limit = device_memory_limit_bytes(device)
    return limit is None or limit >= floor


def _accelerator_lease_path() -> Path:
    override = read_environment(
        ACCELERATOR_LEASE_PATH_ENVIRONMENT_VARIABLE, "EXONYM_ACCELERATOR_LEASE_PATH"
    )
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "trex-accelerator-{0}.lease".format(
        socket.gethostname().replace(os.sep, "_") or "host"
    )


class AcceleratorLease:
    """Exclusive, crash-recoverable lease for a stable physical device identity.

    ``device_key`` scopes ownership per physical device; distinct devices never
    serialize one another. Omitting it retains the explicit legacy host lease.
    New computation scopes acquire available device leases and release promptly.

    The record lives in the system temporary directory keyed by host name --
    device contention is a property of the host, not of a repository -- and a
    holder that died without releasing is detected by process liveness, never
    by a timeout. Metadata transactions use an OS lock on a persistent sidecar
    file, so two contenders cannot both reclaim the same stale record. Unknown
    or corrupt ownership fails closed. Linux process start time prevents PID
    reuse from preserving a dead owner's lease. Other platforms conservatively
    retain a live PID because its original process identity cannot be verified.
    Context-manager callers must inspect acquired before starting device work.
    """

    def __init__(self, path: Optional[Path] = None, *, device_key: Optional[str] = None) -> None:
        self.path = Path(path) if path is not None else _accelerator_lease_path()
        if device_key is not None:
            if not isinstance(device_key, str) or not device_key.strip():
                raise ValueError("device_key must be a non-empty stable device identity")
            digest = hashlib.sha256(device_key.encode("utf-8")).hexdigest()
            self.path = self.path.with_name(self.path.name + "." + digest)
        self.device_key = device_key
        self._token: Optional[str] = None
        self._owner_pid: Optional[int] = None
        self.reason: Optional[str] = None

    @property
    def acquired(self) -> bool:
        return self._token is not None and self._owner_pid == os.getpid()

    def _holder_is_alive(self) -> bool:
        if self.path.is_symlink():
            self.reason = "lease record is a symlink; device acquisition refused"
            return True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = payload["pid"]
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError("invalid owner PID")
            recorded_start = payload.get("process_start_time")
            if recorded_start is not None and (
                not isinstance(recorded_start, str) or not recorded_start.isdigit()
            ):
                raise ValueError("invalid owner process start time")
            if payload.get("host", socket.gethostname()) != socket.gethostname():
                self.reason = "lease owner belongs to another host; ownership is unknown"
                return True
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            self.reason = "lease ownership is unreadable; device acquisition refused"
            return True
        start_time = _process_start_time(pid)
        if recorded_start is not None and start_time is not None and recorded_start != start_time:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            self.reason = "lease owner liveness is unknown; device acquisition refused"
            return True
        return True

    def acquire(self) -> bool:
        """Take this device's lease, or report that another process holds it."""
        self.reason = None
        if self._token is not None:
            self.reason = "this lease instance already owns a record"
            return False
        token = uuid.uuid4().hex
        payload = {
            "pid": os.getpid(),
            "token": token,
            "host": socket.gethostname(),
            "process_start_time": _process_start_time(os.getpid()),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "device_key": self.device_key,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.reason = "lease directory unavailable; device acquisition refused"
            return False
        with lease_metadata_lock(self.path) as locked:
            if not locked:
                self.reason = "lease metadata lock is busy or unavailable; CPU execution selected"
                return False
            if (self.path.exists() or self.path.is_symlink()) and self._holder_is_alive():
                self.reason = self.reason or "another live process holds the accelerator lease"
                return False
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.path.parent,
                    prefix=self.path.name + ".", delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    os.chmod(temporary_path, PRIVATE_FILE_MODE)
                    json.dump(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.path)
                temporary_path = None
                self._token = token
                self._owner_pid = os.getpid()
                _sync_directory(self.path.parent)
            except OSError:
                self.reason = "lease record publication or directory sync failed"
                if self.acquired:
                    # A complete rename can precede a failed directory sync.
                    try:
                        self.path.unlink()
                        self._token = self._owner_pid = None
                        _sync_directory(self.path.parent)
                    except OSError:
                        self.reason = ("lease publication failed; ownership retained for release" if self.acquired
                                       else "lease publication rolled back but directory sync failed")
                return False
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return True

    def release(self) -> None:
        """Release only the record this instance created."""
        if not self.acquired:
            return
        self.reason = None
        with lease_metadata_lock(self.path, blocking=True) as locked:
            if not locked:
                self.reason = "lease metadata lock unavailable during release; ownership retained"
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or payload.get("token") != self._token:
                    self.reason = "lease record no longer matches this owner"
                    self._token = None
                    self._owner_pid = None
                    return
                self.path.unlink()
                self._token = self._owner_pid = None
                _sync_directory(self.path.parent)
            except FileNotFoundError:
                self._token = self._owner_pid = None
                return
            except (OSError, ValueError):
                self.reason = ("lease release failed; ownership retained" if self.acquired
                               else "lease removed but directory sync failed")
                return
            self._token = None
            self._owner_pid = None

    def __enter__(self) -> "AcceleratorLease":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


__all__ = [
    "RESERVED_LOGICAL_CPU_COUNT",
    "WORKER_COUNT_ENVIRONMENT_VARIABLE",
    "ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE",
    "ACCELERATOR_LEASE_PATH_ENVIRONMENT_VARIABLE",
    "MINIMUM_ACCELERATOR_MEMORY_BYTES",
    "AcceleratorLease",
    "accelerator_arithmetic_profile",
    "accelerator_memory_floor_bytes",
    "accelerator_meets_memory_floor",
    "device_memory_limit_bytes",
    "device_memory_profile",
    "device_memory_available_bytes",
    "configure_jax_memory",
    "available_cpu_count",
    "cpu_resource_profile",
    "memory_resource_profile",
    "get_default_n_jobs",
    "resolve_n_jobs",
]
