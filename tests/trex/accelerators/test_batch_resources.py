"""Production batch regressions for transfers, memory sizing and OOM recovery."""

from contextlib import nullcontext

import numpy as np
import pytest

from tests.fixtures.synthetic_observations import synthetic_batch_likelihood_case


def _inputs():
    from trex.foundation.constants import Rearth, Rsun

    time, flux, params = synthetic_batch_likelihood_case()
    # An odd draw count makes the fixed 4-draw blocks end in a partial tail.
    count = (time.size // 2) | 1
    return dict(
        time=time, flux_obs=flux, sigma=params["exptime_days"],
        rp_arr=np.linspace(params["R_EB_solar"], params["R_s_solar"], count) * Rsun / Rearth,
        p_orb=params["P_orb"], inc_arr=np.full(count, params["inc_deg"]),
        a_cm_arr=params["a_cm"], R_host=params["R_s_solar"],
        u1=params["u1"], u2=params["u2"], exptime_days=params["exptime_days"],
        ecc_arr=np.zeros(count), w_arr=np.full(count, np.degrees(np.pi / 2)),
        nsamples=params["nsamples"],
    )


def test_observation_arrays_are_uploaded_once_not_for_every_chunk(monkeypatch):
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    inputs = _inputs()
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(1))
    transfers = {"time": 0, "flux_obs": 0}
    original = models.jnp.asarray

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            for name in transfers:
                if np.shares_memory(value, inputs[name]):
                    transfers[name] += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        actual = models.batch_lnL_TP(**inputs)
    assert actual.shape == inputs["rp_arr"].shape
    assert transfers == {"time": 1, "flux_obs": 1}, "unchanging observations must remain device-resident"


def _accelerator_chosen(monkeypatch, models):
    """Fix the measured device pilot (a hardware probe) to its accelerator outcome.

    These tests exercise the accelerator working-set planner, which runs only
    when the per-kernel pilot chooses an accelerator. The pilot's own choice
    is covered by tests/trex/accelerators/test_device_pilot.py; it is not asserted here.
    """
    monkeypatch.setattr(models, "_device_pilot", lambda evaluate, devices, *args: (
        tuple(device for device in devices if device.platform not in ("cpu", "interpreter")), 0))


def test_gpu_working_set_is_not_limited_by_unrelated_host_model_grid_memory(monkeypatch):
    import trex.runtime.hardware as hardware
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX or not models.JAX_USABLE_DEVICES:
        pytest.skip("requires an actual accelerator for compiler memory analysis")
    _accelerator_chosen(monkeypatch, models)
    inputs = _inputs()
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    # Only the external OS memory counter is substituted. Host arrays already
    # exist; GPU temporaries belong to VRAM, not this host-grid estimate.
    nominal = inputs["time"].size * np.dtype(float).itemsize * models._JAX_BATCH_MEMORY_OVERHEAD
    monkeypatch.setattr(hardware, "memory_resource_profile", lambda: {"available_bytes": nominal})
    plan = {}
    actual = models.batch_lnL_TP(**inputs, _chunk_plan=plan)
    assert any(chunk > 1 for chunk in plan.values()), "host RAM must not force GPU draws into a scalar loop"
    assert np.all(np.isfinite(actual))


def test_batch_reduces_working_set_until_it_fits_instead_of_abandoning_gpu(monkeypatch):
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    inputs = _inputs()
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(1))
    device = models.jax.devices("cpu")[0]
    with models.jax.default_device(device):
        expected = models.batch_lnL_TP(**inputs)
    nominal = inputs["time"].size * np.dtype(float).itemsize * models._JAX_BATCH_MEMORY_OVERHEAD
    # TEST_POLICY: allow just one draw to be uploaded, force repeated halving.
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(nominal * 8))
    original = models.jnp.asarray
    attempted = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and np.shares_memory(value, inputs["rp_arr"]):
            attempted.append(value.size)
            if value.size > 1:
                raise MemoryError("synthetic constrained device upload")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    actual = None
    try:
        with models.jax.default_device(device):
            actual = models.batch_lnL_TP(**inputs)
    except MemoryError:
        pass
    assert actual is not None, "an oversized chunk must shrink to one draw before GPU fallback"
    assert max(attempted) > 1
    np.testing.assert_allclose(actual, expected, rtol=np.finfo(float).eps * inputs["time"].size)


def _force_blocks(monkeypatch, inputs, block):
    """Shrink the evaluation block so a small fixture spans several blocks and a partial tail."""
    from trex.accelerators import _jax_batch

    monkeypatch.setattr(_jax_batch, "_BLOCK_ELEMENTS", block * inputs["time"].size * inputs["nsamples"])
    assert inputs["rp_arr"].size > block and inputs["rp_arr"].size % block, "fixture must end in a partial block"


def test_kernel_blocks_do_not_depend_on_the_chunk_plan(monkeypatch):
    """Each draw is evaluated in the same block, shape and lane under any memory plan.

    XLA gives no bitwise guarantee across batch shapes (one CPU run moved an
    EB lnL by 1.6e-16 between shapes 1 and 2), so plan invariance must come from
    the kernel calls themselves: identical aligned blocks whatever the budget.
    """
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    inputs = _inputs()
    _force_blocks(monkeypatch, inputs, 4)
    original = models.jnp.asarray
    calls = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 1 and np.isin(value, inputs["rp_arr"]).all():
            calls.append(tuple(value))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    observed = []
    for budget in (None, "1"):  # measured headroom (one chunk) and a one-draw budget
        if budget is None:
            monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
        else:
            monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, budget)
        calls.clear()
        with models.jax.default_device(models.jax.devices("cpu")[0]):
            values = models.batch_lnL_TP(**inputs)
        observed.append((values, list(calls)))
    assert len(observed[0][1]) > 1 and {len(block) for block in observed[0][1]} == {4}
    assert observed[0][1] == observed[1][1]
    np.testing.assert_array_equal(observed[0][0], observed[1][0])


def test_the_next_block_is_dispatched_before_the_current_one_is_read(monkeypatch):
    """Real TP kernels respect a measured two-block cap and overlap dispatch.

    The former fake multiplication kernel and fixed event-sequence oracle cannot
    exercise compiler memory analysis or adaptive warm-up. Observe only actual
    uploads/readbacks here; retain byte-exact values, block shapes and ordering.
    """
    from trex.accelerators import _jax_batch, _jax_models as models

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    inputs = _inputs()
    _force_blocks(monkeypatch, inputs, 4)
    monkeypatch.setenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, "1")
    device = models.jax.devices("cpu")[0]
    with models.jax.default_device(device):
        expected = models.batch_lnL_TP(**inputs)
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE)
    monkeypatch.setattr(_jax_batch, "pipeline_capacity", lambda *args, **kwargs: {
        "capacity_blocks": 2, "reason": "explicit test hardware admission"})
    original_upload = models.jnp.asarray
    original_read = _jax_batch.BatchEvaluator._read_block
    events = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 1 and np.isin(value, inputs["rp_arr"]).all():
            events.append(("launch", tuple(value)))
        return original_upload(value, *args, **kwargs)

    def read(output, count):
        events.append(("read", count))
        return original_read(output, count)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    monkeypatch.setattr(_jax_batch.BatchEvaluator, "_read_block", staticmethod(read))
    with models.jax.default_device(device):
        actual = models.batch_lnL_TP(**inputs)
    np.testing.assert_array_equal(actual, expected)
    outstanding = peak = 0
    for kind, _ in events:
        outstanding += 1 if kind == "launch" else -1
        assert 0 <= outstanding <= 2
        peak = max(peak, outstanding)
    assert outstanding == 0 and peak == 2
    assert all(len(payload) == 4 for kind, payload in events if kind == "launch")


def test_auto_gpu_tail_reuses_the_planned_compiled_shape(monkeypatch):
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX or not models.JAX_USABLE_DEVICES:
        pytest.skip("requires accelerator compiler memory analysis")
    _accelerator_chosen(monkeypatch, models)
    inputs = _inputs()
    _force_blocks(monkeypatch, inputs, 4)
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    original = models.jnp.asarray
    transferred = []

    def upload(value, *args, **kwargs):
        # Observe the real radius uploads, including a padded tail. No kernel
        # is replaced; padded repeats are discarded, not new evidence draws.
        if isinstance(value, np.ndarray) and value.ndim == 1 and np.isin(value, inputs["rp_arr"]).all():
            transferred.append(value.size)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    plan = {}
    actual = models.batch_lnL_TP(**inputs, _chunk_plan=plan)
    assert actual.shape == inputs["rp_arr"].shape
    assert len(transferred) > 1, "test must actually cover a nonempty partial tail"
    assert set(transferred) == {4}, "the tail must not compile a new random batch size"


def test_auto_batch_reduces_a_compiler_allocation_failure_before_reference_fallback(monkeypatch):
    import trex.runtime.hardware as hardware
    from trex.runtime.accelerator_pool import observe_accelerator_admissions
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX or not models.JAX_USABLE_DEVICES:
        pytest.skip("requires accelerator compiler memory analysis")
    _accelerator_chosen(monkeypatch, models)
    from jax.stages import Lowered

    inputs = _inputs()
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    actual_headroom = hardware.device_memory_available_bytes
    probed = set()
    nominal = inputs["time"].size * np.dtype(float).itemsize * models._JAX_BATCH_MEMORY_OVERHEAD

    def headroom(device):
        # External counter race: the pilot starts small; newly freed capacity
        # then permits a larger proposal, but its compiler allocation fails.
        if device not in probed:
            probed.add(device)
            return nominal
        return actual_headroom(device)

    compile_original = Lowered.compile
    attempted = []
    failed_shapes = []

    def compile_with_allocation_failure(lowered, *args, **kwargs):
        # Only XLA's allocation boundary is substituted, not the scientific
        # kernel or memory analysis. The successful one-draw compile is real.
        arrays = lowered.in_avals[1]
        if "rp_arr" in arrays:
            count = arrays["rp_arr"].shape[0]
            attempted.append(count)
            if count > 1:
                failed_shapes.append((models.jax.config.jax_default_device, count))
                raise MemoryError("synthetic compiler allocation exhaustion")
        return compile_original(lowered, *args, **kwargs)

    monkeypatch.setattr(hardware, "device_memory_available_bytes", headroom)
    monkeypatch.setattr(Lowered, "compile", compile_with_allocation_failure)
    records = []

    def collect(record):
        records.append(record)
        return nullcontext()

    result = None
    with observe_accelerator_admissions(collect):
        try:
            result = models.batch_lnL_TP(**inputs)
        except MemoryError:
            pass
    assert probed
    assert result is not None, "the proposed compile must halve down to the real one-draw working set"
    assert max(attempted) > 1
    assert np.all(np.isfinite(result))
    assert result.shape == inputs["rp_arr"].shape
    assert len(failed_shapes) == len(set(failed_shapes)), "a failed compile shape must not be reproposed for every subsequent draw"
    retries = [event for record in records for event in record.get("execution_events", [])]
    assert {event["batch"] for event in retries if event["kind"] == "compiler-memory-retry"} == {count for _, count in failed_shapes}
