"""Read OS resource ceilings without importing numerical or accelerator stacks.

Linux cgroup v1/v2 counters are interpreted per the kernel's admin guides:
https://docs.kernel.org/admin-guide/cgroup-v2.html and
https://docs.kernel.org/admin-guide/cgroup-v1/index.html.
CPU quota/period is an exact ratio, not a count of physical cores. Memory is
bytes; Linux meminfo's ``kB`` is converted as KiB. Visible ancestor limits
apply as well as leaf limits. Missing counters stay unknown, never invented.
These are instantaneous capacity diagnostics, not reservations or evidence
that a scientific workload fits. No environment or scheduler state is changed.
"""

from __future__ import annotations

import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

KIBIBYTE_BYTES = 1024
_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


def _read(path: Path, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return None


def _counter(path: Path, errors: list[str]) -> int | None:
    text = _read(path, errors)
    if text is None or text == "max":
        return None
    try:
        value = int(text)
        if value < 0:
            raise ValueError("negative byte counter")
        return value
    except ValueError as exc:
        errors.append(f"{path}: {exc}")
        return None


def _cgroup_directories(errors: list[str]) -> list[tuple[Path, frozenset[str], str]]:
    """Resolve memberships using mount roots, including namespaced mounts."""
    memberships = _read(Path("/proc/self/cgroup"), errors)
    mounts = _read(Path("/proc/self/mountinfo"), errors)
    if memberships is None or mounts is None:
        return []
    groups: dict[str, str] = {}
    for line in memberships.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            _, controllers, membership = fields
            for controller in controllers.split(","):
                groups[controller] = membership
    result: list[tuple[Path, frozenset[str], str]] = []
    seen = set()
    for line in mounts.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        kind = filesystem[0]
        if kind not in {"cgroup", "cgroup2"}:
            continue
        # V1 controller names may live in mount options, source or mount basename.
        options = ",".join((filesystem[2], filesystem[1], fields[5] if len(fields) > 5 else "",
                            PurePosixPath(fields[4]).name))
        controllers = frozenset(options.split(",")) if kind == "cgroup" else frozenset({"cpu", "memory"})
        # The empty controller membership denotes the unified v2 hierarchy.
        keys = controllers & {"cpu", "memory"} if kind == "cgroup" else {""}
        root, mount = (
            _MOUNT_ESCAPE.sub(lambda match: chr(int(match[1], 8)), field)
            for field in fields[3:5]
        )
        for key in keys:
            membership = groups.get(key)
            if membership is None:
                continue
            member_path = PurePosixPath(membership)
            if not member_path.is_absolute() or ".." in member_path.parts:
                errors.append(f"unsafe cgroup membership: {membership!r}")
                continue
            try:
                relative = member_path.relative_to(PurePosixPath(root))
            except ValueError:
                # A cgroup namespace exposes its membership as '/' even when
                # mountinfo still describes the host's subtree root.
                if membership != "/":
                    continue
                relative = PurePosixPath(".")
            boundary = Path(mount)
            current = boundary / str(relative)
            while True:
                record = (current, controllers, kind)
                if record not in seen:
                    seen.add(record)
                    result.append(record)
                if current == boundary:
                    break
                current = current.parent
    return result


def _cgroup_limits(errors: list[str], controller: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cpu: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for directory, controllers, kind in _cgroup_directories(errors):
        if controller == "cpu" and "cpu" in controllers:
            path = directory / ("cpu.max" if kind == "cgroup2" else "cpu.cfs_quota_us")
            text = _read(path, errors)
            if text is not None:
                try:
                    if kind == "cgroup2":
                        quota_text, period_text = text.split()
                    else:
                        quota_text = text
                        period_text = _read(directory / "cpu.cfs_period_us", errors)
                    if quota_text not in {"max", "-1"}:
                        if period_text is None:
                            errors.append(f"{path}: CPU quota period is unavailable")
                            continue
                        quota, period = int(quota_text), int(period_text)
                        if quota <= 0 or period <= 0:
                            raise ValueError("quota and period must be positive")
                        cpu.append({"source": str(path), "quota_us": quota, "period_us": period})
                except (TypeError, ValueError) as exc:
                    errors.append(f"{path}: invalid CPU quota ({exc})")
        if controller == "memory" and "memory" in controllers:
            names = ("memory.max", "memory.high") if kind == "cgroup2" else ("memory.limit_in_bytes",)
            for name in names:
                path = directory / name
                limit = _counter(path, errors)
                if kind == "cgroup" and limit is not None:
                    try:
                        page = os.sysconf("SC_PAGE_SIZE")
                    except (AttributeError, OSError, ValueError):
                        page = None
                    # Linux v1 PAGE_COUNTER_MAX is LONG_MAX rounded down to pages.
                    if page and limit == sys.maxsize // page * page:
                        continue
                if limit is not None:
                    usage_name = "memory.current" if kind == "cgroup2" else "memory.usage_in_bytes"
                    usage = _counter(directory / usage_name, errors)
                    memory.append({"source": str(path), "limit_bytes": limit, "usage_bytes": usage})
    return cpu, memory


def cpu_resource_profile() -> dict[str, Any]:
    """Effective logical capacity: affinity intersected with cgroup bandwidth.

    Fractional quotas are rounded up so the final partial CPU can be used.
    Host responsiveness reservation is applied by the public hardware policy,
    not here. Unreadable quotas are explicitly listed in ``probe_errors``.
    """
    errors: list[str] = []
    logical = os.cpu_count()
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0)) or None
        except OSError as exc:
            errors.append(f"sched_getaffinity: {exc}")
    cpu, _ = _cgroup_limits(errors, "cpu")
    quota = min((Fraction(item["quota_us"], item["period_us"]) for item in cpu), default=None)
    visible = affinity or logical or 1
    effective = min(visible, max(1, math.ceil(quota))) if quota is not None else visible
    return {
        "logical_cpu_count": logical,
        "affinity_cpu_count": affinity,
        "effective_cpu_count": effective,
        "quota_cpu_capacity": float(quota) if quota is not None else None,
        "quota_sources": cpu,
        "probe_errors": errors,
    }


def memory_resource_profile() -> dict[str, Any]:
    """Host RAM bounded by each visible cgroup's remaining shared headroom.

    ``memory.high`` is treated as an operational ceiling to avoid reclaim
    throttling, not claimed as a hard kernel limit. Swap is never usable RAM.
    Missing usage or an incomplete cgroup probe makes available RAM unknown:
    host headroom alone cannot establish the process's remaining allowance.
    POSIX sysconf is a fallback on systems without Linux meminfo.
    """
    errors: list[str] = []
    text = _read(Path("/proc/meminfo"), errors)
    values: dict[str, int] = {}
    if text is not None:
        for line in text.splitlines():
            key, separator, remainder = line.partition(":")
            fields = remainder.split()
            if separator and len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
                values[key] = int(fields[0]) * KIBIBYTE_BYTES
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    source = "/proc/meminfo" if total is not None else None
    if total is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            if min(page_size, physical_pages) <= 0 or available_pages < 0:
                raise ValueError("invalid OS memory counters")
            total, available = page_size * physical_pages, page_size * available_pages
            source = "os.sysconf"
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            errors.append(f"os.sysconf memory: {exc}")
    error_count = len(errors)
    _, limits = _cgroup_limits(errors, "memory")
    incomplete_cgroup_probe = len(errors) > error_count
    totals = ([total] if total is not None else []) + [item["limit_bytes"] for item in limits]
    headrooms = ([available] if available is not None else []) + [
        max(0, item["limit_bytes"] - item["usage_bytes"])
        for item in limits if item["usage_bytes"] is not None
    ]
    unknown_usage = any(item["usage_bytes"] is None for item in limits)
    effective_available = min(headrooms) if headrooms and not unknown_usage and not incomplete_cgroup_probe else None
    effective_total = min(totals) if totals else None
    if effective_available is not None and effective_total is not None:
        effective_available = min(effective_available, effective_total)
    reasons = ["cgroup memory probe is incomplete"] if incomplete_cgroup_probe else []
    if unknown_usage:
        reasons.append("cgroup memory usage is unavailable")
    if effective_available is None and not reasons:
        reasons.append("operating system memory counters are unavailable")
    return {
        "status": "passed" if effective_total is not None and effective_available is not None else "deferred",
        "source": source,
        "total_bytes": effective_total,
        "available_bytes": effective_available,
        "host_total_bytes": total,
        "host_available_bytes": available,
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "cgroup_limits": limits,
        "probe_errors": errors,
        "reason": "; ".join(reasons) or None,
    }


__all__ = [
    "KIBIBYTE_BYTES",
    "cpu_resource_profile",
    "memory_resource_profile",
]
