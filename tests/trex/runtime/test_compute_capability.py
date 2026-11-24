"""Compute capability is architecture metadata, never a throughput classifier."""

import warnings
from types import SimpleNamespace

import pytest

from trex.runtime import hardware, hardware_drivers


class _FakeCudaSymbol:
    """Minimal ctypes-function stand-in with settable signature and recorded calls."""

    __name__ = "cudaDeviceGetAttribute"

    def __init__(self, capability, status=0):
        self.argtypes = None
        self.restype = None
        self.calls = []
        self._capability = capability
        self._status = status

    def __call__(self, pointer, attribute, ordinal):
        self.calls.append((attribute, ordinal))
        if self._status == 0:
            pointer._obj.value = self._capability[attribute]
        return self._status


def _cuda_device(ordinal=0, **extra):
    return SimpleNamespace(
        client=SimpleNamespace(platform_version="cuda"), local_hardware_id=ordinal, **extra,
    )


def _install_symbol(monkeypatch, symbol):
    # The fake library exposes only the attribute symbol, so any SetDevice or
    # other discovery call would fail instead of silently succeeding.
    monkeypatch.setattr(
        hardware_drivers, "_runtime_library",
        lambda _api: SimpleNamespace(cudaDeviceGetAttribute=symbol),
    )


def _assert_unknown(report):
    assert set(report) == {"major", "minor", "source", "reason"}
    assert report["major"] is None
    assert report["minor"] is None
    assert report["source"] is None
    assert isinstance(report["reason"], str) and report["reason"]


@pytest.mark.parametrize(
    "capability", [(7, 5), (8, 6), (8, 9), (9, 0), (8, 0), (6, 0)],
    ids=["7-5", "8-6", "8-9", "9-0", "8-0", "6-0"],
)
def test_runtime_capability_pair_is_metadata_not_a_throughput_class(monkeypatch, capability):
    symbol = _FakeCudaSymbol({75: capability[0], 76: capability[1]})
    _install_symbol(monkeypatch, symbol)
    device = _cuda_device(ordinal=3)

    report = hardware_drivers.compute_capability(device)

    assert set(report) == {"major", "minor", "source", "reason"}
    assert (report["major"], report["minor"]) == capability
    assert report["source"] == "cudaDeviceGetAttribute"
    assert report["reason"] is None
    assert symbol.calls == [(75, 3), (76, 3)]

    profile = hardware.accelerator_arithmetic_profile(device)

    assert set(profile) == {
        "compute_capability", "capability_source", "capability_reason",
        "fp64_throughput_class", "fp64_throughput_reason",
    }
    assert profile["compute_capability"] == capability
    assert profile["capability_source"] == "cudaDeviceGetAttribute"
    assert profile["capability_reason"] is None
    assert profile["fp64_throughput_class"] == "unknown"
    assert profile["fp64_throughput_reason"] == (
        "compute capability identifies an architecture, not a measured FP64:FP32 "
        "throughput ratio; profile the workload"
    )


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(SimpleNamespace(platform_version="cpu"), id="cpu"),
        pytest.param(SimpleNamespace(platform_version="1.2.3"), id="non-cuda"),
        pytest.param(None, id="no-client"),
    ],
)
def test_non_cuda_client_reports_named_unknown(monkeypatch, client):
    queried = []
    monkeypatch.setattr(hardware_drivers, "_runtime_library", lambda api: queried.append(api))
    device = SimpleNamespace(client=client, local_hardware_id=0)

    _assert_unknown(hardware_drivers.compute_capability(device))
    profile = hardware.accelerator_arithmetic_profile(device)

    assert queried == []
    assert profile["compute_capability"] is None
    assert profile["capability_source"] is None
    assert profile["capability_reason"]
    assert profile["fp64_throughput_class"] == "unknown"


@pytest.mark.parametrize("ordinal", [pytest.param(-1, id="negative"), pytest.param(True, id="bool")])
def test_invalid_cuda_ordinal_reports_named_unknown(monkeypatch, ordinal):
    queried = []
    monkeypatch.setattr(hardware_drivers, "_runtime_library", lambda api: queried.append(api))

    _assert_unknown(hardware_drivers.compute_capability(_cuda_device(ordinal=ordinal)))
    assert queried == []


def test_missing_cuda_ordinal_reports_named_unknown(monkeypatch):
    queried = []
    monkeypatch.setattr(hardware_drivers, "_runtime_library", lambda api: queried.append(api))
    device = SimpleNamespace(client=SimpleNamespace(platform_version="cuda"))

    _assert_unknown(hardware_drivers.compute_capability(device))
    assert queried == []


def test_nonzero_driver_status_reports_named_unknown(monkeypatch):
    _install_symbol(monkeypatch, _FakeCudaSymbol({75: 7, 76: 5}, status=1))

    _assert_unknown(hardware_drivers.compute_capability(_cuda_device()))


def test_missing_runtime_symbol_reports_named_unknown(monkeypatch):
    monkeypatch.setattr(hardware_drivers, "_runtime_library", lambda _api: SimpleNamespace())

    _assert_unknown(hardware_drivers.compute_capability(_cuda_device()))


def test_unavailable_runtime_library_reports_named_unknown(monkeypatch):
    def unavailable(_api):
        raise OSError("runtime library unavailable")

    monkeypatch.setattr(hardware_drivers, "_runtime_library", unavailable)

    _assert_unknown(hardware_drivers.compute_capability(_cuda_device()))


def test_capability_metadata_does_not_change_memory_admission(monkeypatch):
    _install_symbol(monkeypatch, _FakeCudaSymbol({75: 7, 76: 5}))
    limit = 8 * 1024**3
    device = _cuda_device(memory_stats=lambda: {"bytes_limit": limit, "bytes_in_use": 0})
    monkeypatch.setattr(hardware, "driver_memory_profile", lambda _device: {
        "total_bytes": limit, "free_bytes": limit // 2, "source": "synthetic",
        "device_key": "synthetic-device", "reason": None,
    })
    monkeypatch.setenv(hardware.ACCELERATOR_MEMORY_FLOOR_ENVIRONMENT_VARIABLE, str(16 * 1024**3))

    before = (
        hardware.accelerator_memory_floor_bytes(),
        hardware.accelerator_meets_memory_floor(device),
    )
    hardware_drivers.compute_capability(device)
    profile = hardware.accelerator_arithmetic_profile(device)
    after = (
        hardware.accelerator_memory_floor_bytes(),
        hardware.accelerator_meets_memory_floor(device),
    )

    assert profile["compute_capability"] == (7, 5)
    assert profile["fp64_throughput_class"] == "unknown"
    assert before == after == (16 * 1024**3, False)


def test_retired_environment_name_is_honoured_with_one_warning(monkeypatch):
    from trex.foundation import environment

    monkeypatch.delenv(hardware.WORKER_COUNT_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv("EXONYM_N_JOBS", "7")
    environment._warned.discard("EXONYM_N_JOBS")

    with pytest.warns(FutureWarning, match="TREX_N_JOBS"):
        assert hardware.get_default_n_jobs() == 7

    # A 24-scenario run must report the rename once, not once per read.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert hardware.get_default_n_jobs() == 7

    # The current name always wins over the retired spelling.
    monkeypatch.setenv(hardware.WORKER_COUNT_ENVIRONMENT_VARIABLE, "3")
    assert hardware.get_default_n_jobs() == 3
