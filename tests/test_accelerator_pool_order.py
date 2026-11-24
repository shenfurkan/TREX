"""Preserve requested execution order while using real identity-scoped leases."""

from collections import namedtuple
from contextlib import contextmanager

import pytest

from trex.runtime import accelerator_pool as pool
from trex.runtime.hardware import ACCELERATOR_LEASE_PATH_ENVIRONMENT_VARIABLE, AcceleratorLease


@pytest.mark.parametrize("contested", [False, True])
def test_stable_lease_acquisition_does_not_reorder_execution_devices(tmp_path, monkeypatch, contested):
    Device = namedtuple("Device", "platform id")
    devices = tuple(Device("gpu", index) for index in (2, 0, 1))
    monkeypatch.setenv(ACCELERATOR_LEASE_PATH_ENVIRONMENT_VARIABLE, str(tmp_path / "accelerator.lease"))
    monkeypatch.delenv(pool.DEVICE_SHARE_ENVIRONMENT_VARIABLE, raising=False)
    # Driver identity is the external boundary; ownership uses actual OS locks.
    monkeypatch.setattr(pool, "driver_memory_profile", lambda device: {"device_key": f"GPU-{device.id}"})
    held = AcceleratorLease(device_key="GPU-0") if contested else None
    if held is not None:
        assert held.acquire()
    acquire = AcceleratorLease.acquire
    attempts = []
    leases = []
    admissions = []

    @contextmanager
    def observe(admission):
        admissions.append(admission)
        try:
            yield
        finally:
            admissions.pop()

    def observed_acquire(lease):
        attempts.append(lease.device_key)
        leases.append(lease)
        return acquire(lease)

    monkeypatch.setattr(AcceleratorLease, "acquire", observed_acquire)
    try:
        expected = tuple(device for device in devices if not contested or device.id != 0)
        with pool.observe_accelerator_admissions(observe), pool.accelerator_execution_scope(devices) as admitted:
            assert admitted == expected
            assert attempts == ["GPU-0", "GPU-1", "GPU-2"]
            assert admissions[0]["acquired_devices"] == list(map(str, expected))
            with pool.accelerator_execution_scope(expected[:1]) as nested:
                assert nested == expected[:1]
                assert len(admissions) == 1
            assert len(attempts) == len(devices)
        assert all(not lease.acquired for lease in leases)
        assert admissions == []
        assert pool.execution_devices(devices) == devices
    finally:
        if held is not None:
            held.release()
