"""Real likelihoods on virtual CPU devices exercise multi-device dispatch.

Run with JAX_PLATFORMS=cpu and XLA_FLAGS=--xla_force_host_platform_device_count=2
to test concurrent placement without requiring multiple physical GPUs.
Discovery, live memory counters and injected allocation failures exercise
scheduling; the JIT kernels and reference likelihoods remain real.
"""

from contextlib import contextmanager
import os
from threading import Barrier, Event, Lock

# JAX reads the forced-device count at import; a focused run of this module
# needs two virtual CPU devices, before any other test imports JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import pytest

from tests.fixtures.synthetic_observations import synthetic_batch_likelihood_case


def _one_draw_blocks(monkeypatch, time, nsamples):
    """These budgets are a few draws; keep the evaluation block below them.

    Budgets size chunks but never shrink the evaluation block, so without
    this a small fixture would be one block on one device.
    """
    from trex.accelerators import _jax_batch

    monkeypatch.setattr(_jax_batch, "_BLOCK_ELEMENTS", time.size * nsamples)


@pytest.mark.parametrize("kind", ("TP", "EB"))
@pytest.mark.parametrize("chunk_size_divisor", (1, 2))
def test_heterogeneous_devices_run_concurrently_in_draw_order(monkeypatch, kind, chunk_size_divisor):
    import trex.runtime.hardware as hardware
    from trex.accelerators import _jax_models as models
    from trex.models import likelihoods
    from trex.foundation.constants import Rearth, Rsun

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    jax = models.jax
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices (see module invocation)")
    devices = devices[:2]
    monkeypatch.setattr(models, "JAX_USABLE_DEVICES", tuple(devices), raising=False)
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    time, flux, params = synthetic_batch_likelihood_case()
    _one_draw_blocks(monkeypatch, time, params["nsamples"])
    # TEST_DESIGN: deliberately different worker budgets, not real VRAM values.
    chunks = (2, 3)
    count = sum(chunks) * len(devices) + 1
    per_draw = time.size * np.dtype(float).itemsize * models._JAX_BATCH_MEMORY_OVERHEAD
    headrooms = dict(zip(devices, (
        chunk * per_draw * models._JAX_BATCH_MEMORY_DEVICE_DIVISOR for chunk in chunks
    )))
    probes = []

    def memory_probe(device):
        probes.append(device)
        return headrooms[device]

    monkeypatch.setattr(hardware, "device_memory_available_bytes", memory_probe)
    radii = params["R_EB_solar"] * np.linspace(1, 2, count)
    if kind == "TP":
        radii *= Rsun / Rearth
    inc = np.full(count, params["inc_deg"])
    ecc = np.zeros(count)
    argp = np.full(count, np.rad2deg(np.pi / 2))
    common = dict(
        time=time, flux_obs=flux, sigma=params["exptime_days"],
        p_orb=params["P_orb"], inc_arr=inc, a_cm_arr=params["a_cm"],
        R_host=params["R_s_solar"], u1=params["u1"], u2=params["u2"],
        exptime_days=params["exptime_days"], ecc_arr=ecc, w_arr=argp,
        nsamples=params["nsamples"], chunk_size_divisor=chunk_size_divisor,
    )
    batch = models.batch_lnL_TP if kind == "TP" else models.batch_lnL_EB
    if kind == "TP":
        common["rp_arr"] = radii
    else:
        common.update(reb_arr=radii, feb_arr=np.full(count, params["EB_fluxratio"]),
                      u1_EB=params["u1_EB"], u2_EB=params["u2_EB"])

    barrier = Barrier(len(devices))
    lock = Lock()
    entries = []
    placements = []
    original_context = jax.default_device
    original_asarray = models.jnp.asarray

    @contextmanager
    def observed_placement(device):
        with lock:
            first_wave = len(entries) < len(devices)
            entries.append(device)
        with original_context(device):
            if first_wave:
                # TEST_POLICY: bounded synchronization; never an actual sleep.
                barrier.wait(timeout=30)
            yield

    def observed_array(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and np.shares_memory(value, radii):
            with lock:
                placements.append((jax.config.jax_default_device, value.copy()))
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(jax, "default_device", observed_placement)
    monkeypatch.setattr(models.jnp, "asarray", observed_array)
    with original_context(None):
        actual = batch(**common)
    assert set(probes) == set(devices)
    assert set(entries[:len(devices)]) == set(devices)
    # Contract change: placement is work-taking, no longer a static round-robin
    # assignment. Keep exact coverage, per-device budgets and reference parity.
    assert {device for device, _ in placements} == set(devices)
    budgets = {device: max(1, chunk // chunk_size_divisor) for device, chunk in zip(devices, chunks)}
    assert all(0 < values.size <= budgets[device] for device, values in placements)
    np.testing.assert_array_equal(np.sort(np.concatenate([values for _, values in placements])), radii)
    reference = []
    for radius in radii:
        args = (time, flux, common["sigma"], radius)
        if kind == "EB":
            args += (params["EB_fluxratio"],)
        args += (params["P_orb"], params["inc_deg"], params["a_cm"],
                 params["R_s_solar"], params["u1"], params["u2"], params["exptime_days"])
        extra = {} if kind == "TP" else {"u1_EB": params["u1_EB"], "u2_EB": params["u2_EB"]}
        evaluate = likelihoods.lnL_TP if kind == "TP" else likelihoods.lnL_EB
        reference.append(evaluate(*args, nsamples=params["nsamples"], **extra))
    # The documented JAX/batman model-parity tolerance, not scheduling slack.
    np.testing.assert_allclose(actual, reference, rtol=1e-4)


def test_failed_device_stops_other_workers_before_the_remaining_draws(monkeypatch):
    """An upload failure must not wait for another device's entire 1M-draw queue."""
    from trex.accelerators import _jax_models as models
    from trex.foundation.constants import Rearth, Rsun

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    devices = models.jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices")
    devices = devices[:2]
    monkeypatch.setattr(models, "JAX_USABLE_DEVICES", tuple(devices))
    # TEST_POLICY: one draw per chunk exercises cancellation, not memory speed.
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(1))
    time, flux, params = synthetic_batch_likelihood_case()
    _one_draw_blocks(monkeypatch, time, params["nsamples"])
    radii = np.full(time.size, params["R_EB_solar"] * Rsun / Rearth)
    failed = Event()
    original_asarray = models.jnp.asarray
    other_transfers = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and np.shares_memory(value, radii):
            if models.jax.config.jax_default_device == devices[0]:
                failed.set()
                raise MemoryError("synthetic failed device transfer")
            assert failed.wait(timeout=30)  # TEST_POLICY: bounded synchronization.
            other_transfers.append(value.size)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    with models.jax.default_device(None), pytest.raises(MemoryError, match="failed device transfer"):
        models.batch_lnL_TP(
            time, flux, params["exptime_days"], radii, params["P_orb"],
            np.full(radii.size, params["inc_deg"]), params["a_cm"], params["R_s_solar"],
            params["u1"], params["u2"], params["exptime_days"], np.zeros(radii.size),
            np.full(radii.size, np.rad2deg(np.pi / 2)), nsamples=params["nsamples"],
        )
    assert len(other_transfers) <= 1


@pytest.mark.parametrize("phase", ["compile", "execute"])
def test_allocation_retry_changes_only_the_failing_devices_block(monkeypatch, phase):
    from trex.accelerators import _jax_batch, _jax_models as models
    from trex.foundation.constants import Rearth, Rsun

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    jax = models.jax
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices (see module invocation)")
    from jax.stages import Compiled, Lowered

    failing, healthy = devices[:2]
    time, flux, params = synthetic_batch_likelihood_case()
    block = 4
    monkeypatch.setattr(_jax_batch, "_BLOCK_ELEMENTS", block * time.size * params["nsamples"])
    radii = np.linspace(0.025, 0.1, 2 * block + 3) * Rsun / Rearth
    batch = models.batch_lnL_TP(
        time, flux, .001, radii, params["P_orb"], np.full(radii.size, params["inc_deg"]),
        params["a_cm"], params["R_s_solar"], params["u1"], params["u2"], params["exptime_days"],
        np.zeros(radii.size), np.full(radii.size, 90.), nsamples=params["nsamples"], _prepare_only=True,
    )
    with jax.default_device(healthy):
        reference = batch.evaluate(slice(None))
    assert np.all(np.diff(reference) < 0), reference

    compile_original, execute_original = Lowered.compile, Compiled.__call__
    failures, evaluated = [], []

    def compile_block(lowered, *args, **kwargs):
        count = lowered.in_avals[1]["rp_arr"].shape[0]
        if phase == "compile" and jax.config.jax_default_device == failing and count == block:
            failures.append(count)
            raise MemoryError("synthetic compiler allocation exhaustion")
        return compile_original(lowered, *args, **kwargs)

    def execute_block(executable, **kwargs):
        count = kwargs["rp_arr"].shape[0]
        device = jax.config.jax_default_device
        if phase == "execute" and device == failing and count == block:
            failures.append(count)
            raise MemoryError("synthetic execution allocation exhaustion")
        evaluated.append((device, count))
        return execute_original(executable, **kwargs)

    monkeypatch.setattr(Lowered, "compile", compile_block)
    monkeypatch.setattr(Compiled, "__call__", execute_block)
    models._compile_pilot(batch.evaluate, failing)
    with jax.default_device(failing):
        recovered = batch.evaluate(slice(None))
    with jax.default_device(healthy):
        unaffected = batch.evaluate(slice(None))

    assert failures == [block], failures
    assert set(evaluated) == {(failing, block // 2), (healthy, block)}, evaluated
    np.testing.assert_allclose(recovered, reference, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(unaffected, reference)


def test_idle_device_takes_remaining_draws_instead_of_waiting_for_static_assignments(monkeypatch):
    """Real kernels run while an external transfer on one device is held.

    The old round-robin queue assigned half the draws to the held device and
    left the other idle. Each device must get its first chunk, then whichever
    is ready must be able to take the remaining unstarted chunks.
    """
    from trex.accelerators import _jax_models as models
    from trex.foundation.constants import Rearth, Rsun

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    devices = tuple(models.jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices")
    monkeypatch.setattr(models, "JAX_USABLE_DEVICES", devices)
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(1))
    time, flux, params = synthetic_batch_likelihood_case()
    _one_draw_blocks(monkeypatch, time, params["nsamples"])
    radii = np.linspace(params["R_EB_solar"], params["R_s_solar"], len(devices) ** 2 + 1) * Rsun / Rearth
    original = models.jnp.asarray
    drained = Event()
    advanced = []
    transfers = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and np.shares_memory(value, radii):
            device = models.jax.config.jax_default_device
            transfers.append((device, value.copy()))
            if device == devices[0] and value[0] == radii[0]:
                # TEST_POLICY: bounded hardware-latency simulation, no sleep.
                advanced.append(drained.wait(timeout=30))
            elif value[-1] == radii[-1]:
                drained.set()
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    with models.jax.default_device(None):
        actual = models.batch_lnL_TP(
            time, flux, params["exptime_days"], radii, params["P_orb"],
            np.full(radii.size, params["inc_deg"]), params["a_cm"], params["R_s_solar"],
            params["u1"], params["u2"], params["exptime_days"], np.zeros(radii.size),
            np.full(radii.size, np.rad2deg(np.pi / 2)), nsamples=params["nsamples"],
        )
    assert advanced == [True], "the ready device must drain remaining draws while its peer is held"
    assert actual.shape == radii.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(np.sort(np.concatenate([values for _, values in transfers])), radii)


def test_device_compile_failure_is_recorded_before_the_device_is_skipped(monkeypatch, caplog):
    """A skipped device leaves an admission event and a host warning, not silence."""
    import json
    import logging
    from types import SimpleNamespace

    from trex.accelerators import _jax_dispatch, _jax_models as models
    from trex.runtime import accelerator_pool
    from trex.runtime import logging as trex_logging

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")

    class FailingDevice:
        platform = "gpu"

        def __str__(self):
            return "gpu:0"

    failing = FailingDevice()
    evaluate = SimpleNamespace(kernel=object(), block=1, n_items=1)
    attempted = []

    def compile_pilot(target, device):
        attempted.append(device)
        raise RuntimeError("synthetic device compile failure")

    monkeypatch.setattr(models, "_compile_pilot", compile_pilot)
    monkeypatch.setattr(models, "_usable_jax_devices", lambda: (failing,))
    monkeypatch.setattr(_jax_dispatch, "_cached_decision", lambda key: (failing,))
    # The execution scope installs this record; the speculative prepare may run before it.
    admission = {}
    token = accelerator_pool._active_admission.set(admission)
    try:
        with models.jax.default_device(None), caplog.at_level(logging.WARNING, logger="trex.hardware"):
            _jax_dispatch._PreparedBatch(evaluate, 8, 1).prepare()
    finally:
        accelerator_pool._active_admission.reset(token)

    assert attempted == [failing]
    assert [(event["kind"], event["device"], event["error_type"])
            for event in admission["execution_events"]] == [
        ("device-compile-failed", "gpu:0", "RuntimeError"),
    ]
    warning, = [record for record in caplog.records if record.name == "trex.hardware"]
    assert warning.args == (failing, "RuntimeError")
    row = json.loads(trex_logging._JsonFormatter("device-compile-probe").format(warning))
    assert row["event"] == "trex.hardware.compile_failed"
    assert row["message"] == "Device compile failed; device skipped"
    assert row["fields"] == {"device": "gpu:0", "error_type": "RuntimeError"}
