"""Actual native likelihood execution across measured pipeline capacities.

Only hardware counters and the clock may be replaced. Observations are analytic
target-neutral fixtures, not model-generated truth. No engine is mocked. The
pipeline scheduling tests at the end use synthetic keyed work; only that
pipeline's clock, capacity probe and launch/read boundary are replaced.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from trex.accelerators import _jax_batch, _jax_models as models
from trex.evidence import nested_evidence
from trex.accelerators._jax_batch import OrderedBlockPipeline
from trex.accelerators._native_unit_batch import NativeUnitCubeBatch
from tests.fixtures.synthetic_observations import synthetic_trex_dense_tp_integrand_case


@pytest.fixture(params=["cpu", "gpu:0", "gpu:1"])
def device(request):
    platform, _, index = request.param.partition(":")
    try:
        devices = models.jax.devices(platform)
    except RuntimeError:
        devices = []
    index = int(index or 0)
    if index >= len(devices):
        pytest.skip(f"physical {request.param} unavailable on the approved host")
    with models.jax.default_device(devices[index]):
        yield devices[index]


def _case():
    scene = synthetic_trex_dense_tp_integrand_case()
    reference = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
        nsamples=3, bin_integration_days=0.0, use_flat_priors=False, population=None, n_dimensions=4)
    # Repeat explicit prior points to check original-order reassembly, including
    # a physical zero and a non-power-of-two last block. This is not calibration.
    pool = np.random.default_rng(20261014).uniform(size=(7, 4))
    points = np.resize(pool, (3073, 4))
    return reference, points


def test_real_native_draws_scale_pipeline_without_changing_shape_order_or_values(device, monkeypatch):
    reference, points = _case()
    # TEST_DESIGN: actual compiler memory counters remain authoritative. Only the
    # hardware headroom is explicit; no numerical evaluation or prior is mocked.
    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", lambda device=None: 1 << 30)
    evaluator = NativeUnitCubeBatch(reference, points)
    values = evaluator(slice(0, len(points)))
    provenance = evaluator.provenance()
    assert "adaptive_pipeline" in provenance, "native callback lacks capacity-adaptive execution metadata"
    plans = provenance["adaptive_pipeline"]
    assert plans and any(wave["in_flight_blocks"] >= 2 for plan in plans for wave in plan["waves"])
    assert evaluator.block == 1024
    assert provenance["logical_likelihood_calls"] == len(points)
    assert any(np.isfinite(values))
    # Missing headroom cannot be converted into a guessed pipeline budget.
    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", lambda device=None: None)
    conservative = NativeUnitCubeBatch(reference, points)
    expected = conservative(slice(0, len(points)))
    assert all(wave["in_flight_blocks"] == 1 for plan in conservative.provenance()["adaptive_pipeline"]
               for wave in plan["waves"])
    np.testing.assert_array_equal(values, expected)
    native = reference.jax_unit_integrand(device=device)
    independent_scalar = np.asarray([native.evaluate(point).log_integrand for point in points[:7]])
    np.testing.assert_array_equal(np.isneginf(values[:7]), np.isneginf(independent_scalar))
    finite = np.isfinite(independent_scalar)
    # This dense 513-point observation is not the original 33-point absolute
    # 1e-10 parity gate. Reduction rounding scales with the number of terms and
    # likelihood magnitude. Use the existing resource-test n*epsilon envelope
    # for this additional scalar diagnostic; the scheduling replay above stays
    # BYTE EXACT. Retain the original absolute-envelope failure receipt.
    np.testing.assert_allclose(values[:7][finite], independent_scalar[finite],
                               rtol=np.finfo(float).eps * len(reference.time), atol=1e-10)


def test_live_capacity_shrink_clamps_inflight_not_the_numerical_granule(device, monkeypatch):
    reference, points = _case()
    evaluator = NativeUnitCubeBatch(reference, points)
    required = evaluator._compiled_bytes(device, evaluator.block)
    calls = []

    def headroom(device=None):
        calls.append(device)
        return required * (4 if len(calls) <= 2 else 1)

    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", headroom)
    # Drive the block pipeline itself: above it, chunk ownership splits these
    # points into spans that never hold three blocks, so the capacity
    # observation would depend on chunk planning rather than on headroom.
    blocks = evaluator._run_blocks(device, 0, len(points), evaluator.block)
    waves = evaluator.provenance()["adaptive_pipeline"][0]["waves"]
    assert any(wave["in_flight_blocks"] == 2 for wave in waves)
    assert any(wave["capacity_blocks"] == wave["in_flight_blocks"] == 1 for wave in waves)
    assert all(flags.all() for _, flags in blocks)
    values = np.concatenate([values for values, _ in blocks])
    assert evaluator.block == 1024 and values.shape == (len(points),)


@pytest.mark.parametrize("exception", [MemoryError, RuntimeError])
def test_true_upload_boundary_failure_reduces_concurrency_only_for_oom(device, monkeypatch, exception):
    reference, points = _case()
    evaluator = NativeUnitCubeBatch(reference, points)
    required = evaluator._compiled_bytes(device, evaluator.block)
    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", lambda device=None: required * 4)
    original = models.jax.device_put
    uploads = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.shape == (1024, 4):
            uploads.append(value.copy())
            if len(uploads) == 3:
                raise exception("controlled external upload boundary failure")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(models.jax, "device_put", upload)
    if exception is RuntimeError:
        with pytest.raises(RuntimeError, match="controlled external upload"):
            evaluator(slice(0, len(points)))
        assert evaluator.provenance()["adaptive_pipeline"][0]["oom_depth_limit"] is None
        assert evaluator._integrity_failures
        return
    actual = evaluator(slice(0, len(points)))
    assert evaluator.block == 1024
    assert evaluator.provenance()["adaptive_pipeline"][0]["oom_depth_limit"] == 1
    # The interrupted wave repeats the same retained input, not a fresh draw.
    np.testing.assert_array_equal(uploads[1], uploads[3])
    monkeypatch.setattr(models, "_jax_batch_memory_budget_bytes", lambda device=None: None)
    expected = NativeUnitCubeBatch(reference, points)(slice(0, len(points)))
    np.testing.assert_array_equal(actual, expected)


class _WaveClock:
    """Synthetic perf_counter stamps; an unexpected extra wave fails clearly."""

    def __init__(self, seconds):
        self._stamps = []
        now = 0.0
        for duration in seconds:
            self._stamps.extend((now, now + duration))
            now += duration
        self._stamps = iter(self._stamps)

    def perf_counter(self):
        try:
            return next(self._stamps)
        except StopIteration:
            raise AssertionError("pipeline executed an unexpected extra wave") from None


def _run_synthetic(pipeline, starts, seconds, monkeypatch, *, capacity_blocks=8, block=4, probe=None):
    """Run the real pipeline; only its clock, capacity probe and work units are fake."""
    launched, read_keys = [], []

    def launch(start, size):
        launched.append(start)
        return start, size

    def read(start, size):
        read_keys.append(start)
        return start

    probe_row = dict(probe) if probe is not None else {"capacity_blocks": capacity_blocks}
    monkeypatch.setattr(_jax_batch, "time", _WaveClock(seconds))
    values = pipeline.run(list(starts), launch, read, lambda: dict(probe_row),
                          device="synthetic", block=block)
    return values, launched, read_keys


def test_one_slower_wave_keeps_probing_at_the_retained_best_depth(monkeypatch):
    pipeline = OrderedBlockPipeline()
    values, launched, read_keys = _run_synthetic(pipeline, range(3), [1.0, 3.0], monkeypatch)
    assert pipeline.settled is False
    assert pipeline.best_depth == 1 and pipeline.depth == 2
    assert read_keys == launched == [0, 1, 2]
    assert values == [0, 1, 2]


def test_two_consecutive_slower_waves_settle_at_the_best_observed_depth(monkeypatch):
    pipeline = OrderedBlockPipeline()
    values, launched, read_keys = _run_synthetic(
        pipeline, range(6), [1.0, 3.0, 3.0, 1.0], monkeypatch)
    assert pipeline.settled is True
    assert pipeline.depth == pipeline.best_depth == 1
    assert pipeline.best_rate == 4.0
    assert pipeline.waves[2]["wave_count"] == 2, "both misses must be observed before settling"
    assert pipeline.waves[1]["wave_count"] == 2
    assert read_keys == launched == list(range(6))
    assert values == list(range(6))


def test_improving_wave_resets_the_settle_confirmation_count(monkeypatch):
    pipeline = OrderedBlockPipeline()
    values, launched, read_keys = _run_synthetic(
        pipeline, range(15), [1.0, 3.0, 1.5, 4.0, 4.0, 2.0], monkeypatch)
    assert pipeline.settled is True
    assert pipeline.depth == pipeline.best_depth == 2
    # The first post-improvement miss must not settle; both width-4 waves run.
    assert pipeline.waves[4]["wave_count"] == 2
    assert pipeline.waves[2]["wave_count"] == 3
    assert read_keys == launched == list(range(15))
    assert values == list(range(15))


def test_tail_wave_cannot_settle_the_pipeline(monkeypatch):
    pipeline = OrderedBlockPipeline()
    values, launched, read_keys = _run_synthetic(pipeline, range(4), [1.0, 3.0, 100.0], monkeypatch)
    assert pipeline.settled is False
    assert pipeline.best_depth == 1 and pipeline.depth == 2
    assert pipeline.waves[1]["wave_count"] == 2, "the one-key tail runs as a width-1 wave"
    assert pipeline.waves[2]["wave_count"] == 1
    assert read_keys == launched == list(range(4))
    assert values == list(range(4))


def test_clamped_waves_never_count_as_tuning_evidence(monkeypatch):
    pipeline = OrderedBlockPipeline()
    first = _run_synthetic(pipeline, range(3), [1.0, 3.0], monkeypatch)
    assert pipeline.settled is False and pipeline.depth == 2
    # With capacity at one block, two full-width waves at the clamped depth must
    # not count toward settling or update the best depth/rate of the real target.
    second = _run_synthetic(pipeline, range(3, 5), [100.0, 0.01], monkeypatch, capacity_blocks=1)
    assert pipeline.settled is False
    assert pipeline.best_depth == 1 and pipeline.depth == 2
    assert pipeline.best_rate == 4.0
    assert first[1] + second[1] == first[2] + second[2] == list(range(5))


def test_capacity_recovery_resumes_depth_exploration(monkeypatch):
    pipeline = OrderedBlockPipeline()
    first = _run_synthetic(pipeline, range(3), [1.0, 3.0], monkeypatch)
    second = _run_synthetic(pipeline, range(3, 5), [100.0, 0.01], monkeypatch, capacity_blocks=1)
    # Recovery is new evidence: the next full-width wave must still be able to
    # grow the depth the clamped waves were forbidden to judge.
    third = _run_synthetic(pipeline, range(5, 8), [1.0, 2.0], monkeypatch)
    assert first[0] == [0, 1, 2] and second[0] == [3, 4] and third[0] == [5, 6, 7]
    assert pipeline.settled is False
    assert pipeline.best_depth == 2 and pipeline.depth == 4


def test_tail_wave_preserves_pending_miss_evidence(monkeypatch):
    pipeline = OrderedBlockPipeline()
    first = _run_synthetic(pipeline, range(4), [1.0, 3.0, 100.0], monkeypatch)
    assert pipeline.settled is False, "a tail wave cannot settle the pipeline"
    assert first[0] == [0, 1, 2, 3]
    # The tail proved nothing about depth 2, so the earlier full-width miss
    # stays on record and one more miss settles at the retained best depth.
    second = _run_synthetic(pipeline, range(2), [10.0], monkeypatch)
    assert second[0] == [0, 1]
    assert pipeline.settled is True
    assert pipeline.depth == pipeline.best_depth == 1


def test_allocation_ceiling_survives_capacity_recovery(monkeypatch):
    pipeline = OrderedBlockPipeline()
    failures = {"left": 1}
    launched, read_keys = [], []

    def launch(start, size):
        launched.append(start)
        return start, size

    def read(start, size):
        read_keys.append(start)
        if failures["left"] and start == 1:
            failures["left"] -= 1
            raise MemoryError("controlled synthetic allocation failure")
        return start

    ticks = itertools.count(0.5, 0.5)
    monkeypatch.setattr(_jax_batch, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    probe = {"capacity_blocks": 8}
    values = pipeline.run([0, 1, 2, 3], launch, read, lambda: dict(probe), device="synthetic", block=4)
    # The failed width-2 wave is withdrawn and re-launched, yet the committed
    # result still holds each key exactly once, in order.
    assert values == [0, 1, 2, 3]
    assert launched == [0, 1, 2, 1, 2, 3]
    assert read_keys == [0, 1, 2, 1, 2, 3]
    assert pipeline.settled is True
    assert pipeline.oom_limit == pipeline.depth == 1

    # Recovered free memory is not evidence that the width that failed now fits.
    launched.clear()
    recovered = pipeline.run([0, 1], launch, read, lambda: dict(probe), device="synthetic", block=4)
    assert recovered == [0, 1]
    assert launched == [0, 1]
    assert all(row["in_flight_blocks"] == 1 for row in pipeline.provenance()["waves"])


def test_adaptive_wave_measurement_carries_counters_and_bounds_provenance(monkeypatch):
    from trex.runtime import accelerator_pool

    events = []

    def record(kind, device, **details):
        events.append((kind, device, dict(details)))

    monkeypatch.setattr(accelerator_pool, "record_accelerator_event", record)
    probe = {
        "device_budget_bytes": 1 << 30, "host_available_bytes": 1 << 31,
        "compiled_block_bytes": 512, "output_block_bytes": 256,
        "resident_input_bytes": 128, "capacity_blocks": 4,
        "reason": "synthetic measured capacity",
    }
    pipeline = OrderedBlockPipeline(max_depth=2)
    values, launched, read_keys = _run_synthetic(
        pipeline, range(6), [1.0, 3.0, 3.0, 1.0], monkeypatch, probe=probe)
    assert [kind for kind, _, _ in events] == ["adaptive-block-wave"] * 4
    for _, device, details in events:
        assert device == "synthetic"
        assert details["in_flight_blocks"] in (1, 2)
        assert details["effective_batch_draws"] == details["in_flight_blocks"] * 4
        assert details["seconds"] > 0
        assert details["kernel_lanes_per_second"] == pytest.approx(
            details["effective_batch_draws"] / details["seconds"])
        assert details["capacity_blocks"] == 4
        assert details["compiled_block_bytes"] == 512
        assert details["output_block_bytes"] == 256
        assert details["device_budget_bytes"] == 1 << 30
        assert details["host_available_bytes"] == 1 << 31
    waves = pipeline.provenance()["waves"]
    required = {"in_flight_blocks", "effective_batch_draws", "seconds", "kernel_lanes_per_second",
                "capacity_blocks", "compiled_block_bytes", "output_block_bytes"}
    assert all(required <= row.keys() for row in waves)
    assert [row["in_flight_blocks"] for row in waves] == [1, 2]
    assert len(waves) == len({row["in_flight_blocks"] for row in waves}) < len(events)
    assert sum(row["wave_count"] for row in waves) == len(events)
    assert {row["in_flight_blocks"]: row["wave_count"] for row in waves} == {1: 2, 2: 2}
    assert read_keys == launched == list(range(6))
    assert values == list(range(6))
