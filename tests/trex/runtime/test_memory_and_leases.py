"""Allocation admission uses measured counters and actual OS-backed device leases."""

from types import SimpleNamespace

import pytest

from trex.runtime import hardware
from trex.runtime.hardware_resources import is_memory_exhaustion


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MemoryError(), True),
        (RuntimeError("RESOURCE_EXHAUSTED: Failed to allocate buffer"), True),
        (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"), True),
        (RuntimeError("CUDA_ERROR_INVALID_VALUE"), False),
        (ValueError("invalid light curve shape"), False),
    ],
)
def test_memory_exhaustion_classifies_retryable_allocations(error, expected):
    assert is_memory_exhaustion(error) is expected


def test_driver_free_memory_bounds_allocator_capacity(monkeypatch):
    device = SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"bytes_limit": 16384, "bytes_in_use": 0},
    )
    monkeypatch.setattr(hardware, "driver_memory_profile", lambda _device: {
        "total_bytes": 16384, "free_bytes": 4096, "source": "driver-counter",
        "device_key": "neutral-device", "reason": None,
    })
    assert hardware.device_memory_available_bytes(device) == 4096


def test_unknown_usage_is_not_reported_as_available_capacity(monkeypatch):
    device = SimpleNamespace(platform="gpu", memory_stats=lambda: {"bytes_limit": 4096})
    monkeypatch.setattr(hardware, "driver_memory_profile", lambda _device: {
        "total_bytes": None, "free_bytes": None,
    })
    assert hardware.device_memory_available_bytes(device) is None


def test_missing_driver_free_counter_is_not_allocator_headroom(monkeypatch):
    device = SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"bytes_limit": 16384, "bytes_in_use": 0},
    )
    monkeypatch.setattr(hardware, "driver_memory_profile", lambda _device: {
        "total_bytes": None, "free_bytes": None, "reason": "driver counter unavailable",
    })
    profile = hardware.device_memory_profile(device)
    assert profile["allocator_headroom_bytes"] == 16384
    assert profile["available_bytes"] is None
    assert profile["driver_probe_reason"] == "driver counter unavailable"


def test_device_leases_do_not_serialize_independent_accelerators(tmp_path):
    first = hardware.AcceleratorLease(tmp_path / "pool.lease", device_key="neutral-first")
    peer = hardware.AcceleratorLease(tmp_path / "pool.lease", device_key="neutral-second")
    conflict = hardware.AcceleratorLease(tmp_path / "pool.lease", device_key="neutral-first")
    try:
        assert first.acquire()
        assert peer.acquire()
        assert not conflict.acquire()
    finally:
        first.release()
        peer.release()
    try:
        assert conflict.acquire()
    finally:
        conflict.release()


@pytest.mark.slow
def test_unknown_ram_never_creates_a_fixed_trex_budget(monkeypatch):
    import numpy as np
    from trex.accelerators import _jax_models as models
    from trex.foundation.constants import Rsun
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(hardware, "memory_resource_profile", lambda: {"available_bytes": None})
    monkeypatch.setattr(hardware, "device_memory_available_bytes", lambda _device: None)
    case = synthetic_trex_tp_integrand_case()
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        with pytest.raises(MemoryError, match="memory.*unavailable|unknown.*memory"):
            models.batch_lnL_TP(
                case["time"], case["flux"], case["sigma"], np.ones(1), case["period_days"],
                np.full(1, 90.), 10*Rsun, 1.,
                case["u1"], case["u2"], case["exptime_days"], np.zeros(1),
                np.full(1, 90.), nsamples=4,
            )
