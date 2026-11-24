"""Capability-probed CUDA/HIP counters; no OS, model or capacity whitelist.

The runtime's logical ordinal is JAX's ``local_hardware_id`` (so visibility
masks are respected). Memory is queried on that device and the calling thread's
previous device is restored. UUIDs, not ordinals, identify shared-host leases.
Compute capability is read as CUDA runtime architecture metadata and classifies
no device. Unsupported drivers remain unknown. APIs: CUDA Runtime memory/device
management and HIP Runtime memory/device management; all memory quantities are
bytes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
from functools import lru_cache
from numbers import Integral
from pathlib import Path

# UUID ABI: cudaUUID_t / hipUUID, sixteen unsigned bytes (not a memory policy).
UUID_BYTES = 16


@lru_cache
def _runtime_library(api: str):
    library_name = "cudart" if api == "cuda" else "amdhip64"
    candidates = []
    found = ctypes.util.find_library(library_name)
    if found:
        candidates.append(found)
    # CUDA runtime wheels are part of the locked JAX installation. Discover
    # their location, never a host-specific absolute path.
    if api == "cuda":
        try:
            spec = importlib.util.find_spec("nvidia.cuda_runtime")
        except (ImportError, ValueError):
            spec = None
        for root in getattr(spec, "submodule_search_locations", ()) or ():
            candidates.extend(str(path) for path in Path(root).glob("lib/*cudart*"))
    for candidate in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(candidate)
            for suffix in ("GetDevice", "SetDevice", "MemGetInfo"):
                getattr(library, api + suffix)
            if api == "cuda":
                _ = library.cudaFree
            return library
        except (OSError, AttributeError):
            continue
    raise OSError("runtime library unavailable or missing required symbols")


def _function(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


def _checked(function, *arguments):
    code = function(*arguments)
    if code != 0:
        raise RuntimeError(f"{function.__name__} returned driver status {code}")


@lru_cache
def _cuda_driver():
    for name in ("cuda", "nvcuda"):
        found = ctypes.util.find_library(name)
        if found:
            try:
                library = ctypes.CDLL(found)
                _ = library.cuCtxGetDevice
                if not hasattr(library, "cuDeviceGetUuid_v2"):
                    _ = library.cuDeviceGetUuid
                return library
            except (OSError, AttributeError):
                continue
    raise OSError("CUDA driver identity API unavailable")


def _device_identity(api, library, ordinal):
    identity = (ctypes.c_ubyte * UUID_BYTES)()
    if api == "cuda":
        driver = _cuda_driver()
        # Query the CURRENT runtime-selected context. A CUDA driver ordinal is
        # not interchangeable with a visibility-masked runtime ordinal.
        physical = ctypes.c_int()
        current = _function(driver, "cuCtxGetDevice", [ctypes.POINTER(ctypes.c_int)])
        _checked(current, ctypes.byref(physical))
        name = "cuDeviceGetUuid_v2" if hasattr(driver, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
        function = _function(driver, name, [ctypes.c_void_p, ctypes.c_int])
        _checked(function, ctypes.byref(identity), physical.value)
    else:
        function = _function(library, "hipDeviceGetUuid", [ctypes.c_void_p, ctypes.c_int])
        _checked(function, ctypes.byref(identity), ordinal)
    return bytes(identity)


def driver_memory_profile(device) -> dict:
    """Fresh global free/total VRAM and stable identity, or a named unknown.

    An API is chosen only when the execution client advertises that runtime.
    A CPU or unfamiliar accelerator never receives invented GPU counters.
    Fields total_bytes/free_bytes are bytes or None; source names the memory
    API, device_key is a nonzero runtime UUID or None, and reason names failures
    without loader paths. Failed restoration invalidates the entire snapshot.
    """
    report = dict(total_bytes=None, free_bytes=None, source=None, device_key=None, reason=None)
    try:
        version = str(getattr(getattr(device, "client", None), "platform_version", "")).lower()
        api = "cuda" if "cuda" in version else "hip" if "rocm" in version or "hip" in version else None
        ordinal = getattr(device, "local_hardware_id", None)
        if api is None or isinstance(ordinal, bool) or not isinstance(ordinal, Integral) or ordinal < 0:
            report["reason"] = "execution backend exposes no supported driver memory API/ordinal"
            return report
        ordinal = int(ordinal)
        library = _runtime_library(api)
        integer_pointer = ctypes.POINTER(ctypes.c_int)
        size_pointer = ctypes.POINTER(ctypes.c_size_t)
        get_device = _function(library, api + "GetDevice", [integer_pointer])
        set_device = _function(library, api + "SetDevice", [ctypes.c_int])
        memory_info = _function(library, api + "MemGetInfo", [size_pointer, size_pointer])
        previous = ctypes.c_int()
        _checked(get_device, ctypes.byref(previous))
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        reasons = []
        try:
            if previous.value != ordinal:
                _checked(set_device, ordinal)
            try:
                _checked(memory_info, ctypes.byref(free), ctypes.byref(total))
                if total.value <= 0 or free.value > total.value:
                    raise ValueError("invalid driver memory counters")
                report.update(total_bytes=total.value, free_bytes=free.value, source=api + "MemGetInfo")
            except (OSError, AttributeError, RuntimeError, ValueError) as exc:
                reasons.append("driver memory counter unavailable")
                report["memory_error_type"] = type(exc).__name__
            try:
                if api == "cuda":
                    # SetDevice can be lazy; never identify a stale/uncreated context.
                    initialize = _function(library, "cudaFree", [ctypes.c_void_p])
                    _checked(initialize, None)
                identity = _device_identity(api, library, ordinal)
                if not any(identity):
                    raise ValueError("invalid device UUID")
                report["device_key"] = api + "-" + identity.hex()
            except (OSError, AttributeError, RuntimeError, ValueError) as exc:
                reasons.append("driver identity unavailable")
                report["identity_error_type"] = type(exc).__name__
        finally:
            if previous.value != ordinal:
                try:
                    _checked(set_device, previous.value)
                except (OSError, AttributeError, RuntimeError, ValueError) as exc:
                    report.update(total_bytes=None, free_bytes=None, source=None, device_key=None)
                    reasons.append("previous driver device could not be restored")
                    report["restore_error_type"] = type(exc).__name__
        report["reason"] = "; ".join(reasons) or None
    except Exception as exc:  # noqa: BLE001 - arbitrary backend properties are an optional probe boundary
        report.update(total_bytes=None, free_bytes=None, source=None, device_key=None,
                      reason="driver memory probe unavailable", error_type=type(exc).__name__)
    return report


def compute_capability(device) -> dict:
    """CUDA architecture pair from runtime device attributes, or a named unknown.

    The probe queries the visibility-masked runtime ordinal without changing
    the current device. The pair identifies an architecture only: it is not a
    measured FP64:FP32 throughput ratio and must not classify a device. A
    non-CUDA client, an invalid ordinal, a missing symbol or a nonzero driver
    status leaves every value None with a reason and never raises.
    """
    report = dict(major=None, minor=None, source=None, reason=None)
    try:
        version = str(getattr(getattr(device, "client", None), "platform_version", "")).lower()
        if "cuda" not in version:
            report["reason"] = "execution backend exposes no supported CUDA runtime"
            return report
        ordinal = getattr(device, "local_hardware_id", None)
        if isinstance(ordinal, bool) or not isinstance(ordinal, Integral) or ordinal < 0:
            report["reason"] = "CUDA runtime ordinal is missing or invalid"
            return report
        library = _runtime_library("cuda")
        get_attribute = _function(
            library, "cudaDeviceGetAttribute",
            [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        )
        major, minor = ctypes.c_int(), ctypes.c_int()
        # cudaDevAttrComputeCapabilityMajor / cudaDevAttrComputeCapabilityMinor.
        _checked(get_attribute, ctypes.byref(major), 75, int(ordinal))
        _checked(get_attribute, ctypes.byref(minor), 76, int(ordinal))
        report.update(major=major.value, minor=minor.value, source="cudaDeviceGetAttribute")
    except Exception as exc:  # noqa: BLE001 - arbitrary backend properties are an optional probe boundary
        report.update(major=None, minor=None, source=None,
                      reason=f"compute capability probe unavailable ({type(exc).__name__})")
    return report


__all__ = [
    "UUID_BYTES",
    "compute_capability",
    "driver_memory_profile",
]
