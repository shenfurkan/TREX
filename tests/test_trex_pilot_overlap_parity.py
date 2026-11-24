"""Real likelihood parity; virtual accelerators and fixed clocks test dispatch, not GPU speed."""

from contextlib import contextmanager
import threading

import numpy as np
import pytest

from tests.test_trex_background_population import _evidences


@pytest.mark.parametrize("accelerator_seconds", (0.25, 4.0))
def test_aot_and_overlapped_pilots_preserve_all_draws_evidence_and_cached_decisions(monkeypatch, accelerator_seconds):
    import trex.runtime.accelerator_pool as accelerator_pool
    import trex.accelerators._jax_batch as batch_module
    from trex.accelerators import _jax_models as models
    from trex.evidence import marginal_likelihoods as ml

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    cpus = models.jax.devices("cpu")
    if len(cpus) < 3:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=3")

    class VirtualAccelerator:
        platform = "gpu"

        def __init__(self, device):
            self.device = device

        def __str__(self):
            return f"virtual:{self.device}"

    devices = tuple(VirtualAccelerator(device) for device in cpus[1:3])
    default_device = models.jax.default_device
    monkeypatch.setattr(models.jax, "default_device",
                        lambda device: default_device(getattr(device, "device", device)))
    monkeypatch.setattr(models, "_usable_jax_devices", lambda: devices)
    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", lambda *args: 64 * 1024**2)
    # Exercise concurrent ownership chunks and a partial tail, not a single kernel block.
    monkeypatch.setattr(batch_module, "_BLOCK_ELEMENTS", 64 * 60 * 5)

    @contextmanager
    def scope(devices):
        yield devices

    monkeypatch.setattr(accelerator_pool, "accelerator_execution_scope", scope)
    monkeypatch.setattr(models, "_DEVICE_DECISIONS", {})
    original_run = models._run_jax_chunks
    original_timed = models._timed_pilot
    original_prepare = models._PreparedBatch.prepare
    original_plan = models._plan_batch_chunk
    original_init = ml._ScenarioOverlap.__init__
    original_evaluate = models.BatchEvaluator.__call__
    default_rng = np.random.default_rng

    class MainThreadOnly:
        def __init__(self, generator):
            self.generator = generator

        def __getattr__(self, name):
            assert threading.current_thread() is threading.main_thread(), name
            return getattr(self.generator, name)

    monkeypatch.setattr(ml.np.random, "default_rng", lambda *args: MainThreadOnly(default_rng(*args)))
    mode = None
    snapshots, likelihoods, decisions, pilot_calls, evaluated_pilots = [], [], [], [], []
    cpu_started, tp_finished = threading.Event(), threading.Event()
    timing = threading.local()
    hold_first = False

    def evaluate(self, subset):
        if getattr(timing, "active", False):
            evaluated_pilots.append((self.kernel.__name__, str(models.jax.config.jax_default_device)))
        return original_evaluate(self, subset)

    def timed(evaluate, device, draws):
        work = evaluate.shared["time"].size * evaluate.shared["nsamples"]
        pilot_calls.append((evaluate.kernel, work, str(device)))
        if (mode == "on" and device is cpus[0] and evaluate.kernel is models._batch_lnL_EB_jit
                and not cpu_started.is_set()):
            assert threading.current_thread() is threading.main_thread()
            cpu_started.set()
        if mode == "on" and device in devices and evaluate.kernel is models._batch_lnL_EB_jit:
            assert tp_finished.is_set(), "GPU pilot must wait for preceding MC to finish"
        timing.active = True
        try:
            if mode == "on":
                values, _, used = original_timed(evaluate, device, draws)
            else:
                # Warm-up and timed evaluation use the real likelihood.
                with models.jax.default_device(device):
                    evaluate(slice(0, draws))
                    values = evaluate(slice(0, draws))
                used = draws
        finally:
            timing.active = False
        return values, 1.0 if device is cpus[0] else accelerator_seconds, used

    def run_batch(n, points, samples, evaluate, *args, **kwargs):
        nonlocal hold_first
        outer = kwargs.get("_allocated_devices") is None
        is_first = outer and hold_first
        if is_first:
            hold_first = False
            assert cpu_started.wait(30), "next scenario did not prepare its CPU pilot during the preceding MC"
        if outer:
            if not snapshots:
                assert n > evaluate.block and n % evaluate.block
            snapshots.append({name: np.array(value, copy=True) for name, value in evaluate.draws.items()})
        result = original_run(n, points, samples, evaluate, *args, **kwargs)
        if outer:
            assert not hasattr(evaluate, "_cpu_pilot"), "CPU reference must not survive dispatch"
            likelihoods.append(result.copy())
        if is_first:
            tp_finished.set()
        return result

    def event(kind, device, **details):
        if kind == "device-pilot":
            decisions.append(details)

    monkeypatch.setattr(models.BatchEvaluator, "__call__", evaluate)
    monkeypatch.setattr(models, "_timed_pilot", timed)
    monkeypatch.setattr(models, "_run_jax_chunks", run_batch)
    monkeypatch.setattr(accelerator_pool, "record_accelerator_event", event)

    def run(optimized, cached):
        nonlocal mode, hold_first
        mode = "on" if optimized else "off"
        snapshots.clear()
        likelihoods.clear()
        decisions.clear()
        pilot_calls.clear()
        evaluated_pilots.clear()
        hold_first = optimized and not cached
        monkeypatch.setattr(models._PreparedBatch, "prepare", original_prepare if optimized else lambda *args: None)
        # The unoptimized arm keeps the conservative initial chunk instead of the
        # compiled-memory plan; aligned blocks make the likelihoods plan-invariant.
        monkeypatch.setattr(models, "_plan_batch_chunk", original_plan if optimized else
                            lambda evaluate, initial, *args: initial)

        def init(self):
            original_init(self)
            if not optimized:
                self._enabled = False

        monkeypatch.setattr(ml._ScenarioOverlap, "__init__", init)
        outputs = {name: {} for name in ("ess_out", "support_state_out", "jax_fallback_out", "representative_fit_out")}
        lnz = _evidences(nearby_stars=[{"M_s": 0.5, "R_s": 0.5, "aperture_flux_fraction": 0.25}],
                         target_aperture_flux_fraction=0.75, **outputs)
        if cached:
            assert not pilot_calls and not evaluated_pilots and not decisions
        else:
            # Screening and the full observation grid have different measured
            # work keys. Each key still gets exactly one pilot per device.
            expected_pilots = {
                (kernel, work, str(device))
                for kernel, work, _device_set in models._DEVICE_DECISIONS
                for device in (cpus[0], *devices)
            }
            assert set(pilot_calls) == expected_pilots
            assert len(pilot_calls) == len(expected_pilots), "a measured key was piloted more than once"
            assert len(evaluated_pilots) == len(pilot_calls) * (1 if optimized else 2)
            expected = devices if accelerator_seconds == 0.25 else (cpus[0],)
            assert all(row["chosen"] == [str(device) for device in expected] for row in decisions)
            if optimized:
                assert any(row["kernel"] == str(models._batch_lnL_EB_jit)
                           and row["cpu_reference_overlap"] == "TP" for row in decisions)
        assert not outputs["jax_fallback_out"]
        # Wall time should differ between dispatch modes; the scientific records must not.
        for fit in outputs["representative_fit_out"].values():
            fit.pop("phase_seconds", None)
        return (lnz, outputs, snapshots.copy(), likelihoods.copy(),
                [(row["chosen"], row["reason"], row["cpu_draws_per_second"],
                  row["accelerator_draws_per_second"], [p.get("parity") for p in row["pilot"]])
                 for row in decisions], dict(models._DEVICE_DECISIONS))

    optimized = run(True, False)
    optimized_cached = run(True, True)
    models._DEVICE_DECISIONS.clear()
    reference = run(False, False)
    reference_cached = run(False, True)
    for left, right in ((optimized, reference), (optimized_cached, reference_cached),
                        (optimized, optimized_cached)):
        np.testing.assert_array_equal(left[0], right[0])
        assert repr(left[1]) == repr(right[1])
        assert left[5] == right[5]
        assert len(left[2]) == len(right[2]) == len(left[3]) == len(right[3])
        for lhs, rhs in zip(left[2], right[2]):
            assert lhs.keys() == rhs.keys()
            for name in lhs:
                np.testing.assert_array_equal(lhs[name], rhs[name])
        for lhs, rhs in zip(left[3], right[3]):
            np.testing.assert_array_equal(lhs, rhs)
    assert optimized[4] == reference[4]
