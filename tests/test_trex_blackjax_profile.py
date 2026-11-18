"""One short approved dual-device native-slice profile, not estimator calibration.

Uses the real synthetic fixed-host TP integrand. Measures the identical 32-chain,
12-step deletion-batch workload with scalar dispatch versus the production
batched-chains entry point. Initialization/compilation and warmed evaluation
are reported separately. Same-point likelihoods use the existing 1e-10-nat
gate; cross-mode trajectory/counter equality is reported, not an acceptance bar.
Does not assert a speedup or fabricate physical counts.
This is a constrained-batch microprofile, not full integrate or dynesty timing.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from threading import Event, Thread
import time

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_kernel import NativeNestedKernel, _SLICE_CALL_BOUND
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _profile(device):
    jax, jnp = models.jax, models.jnp
    started = time.monotonic()
    with jax.default_device(device):
        scene = synthetic_trex_tp_integrand_case()
        reference = nested_evidence._TransitingPlanetIntegrand(**scene, host_mass=1.0, host_radius=1.0,
            bin_integration_days=0.0, nsamples=3, use_flat_priors=False, population=None, n_dimensions=4)
        kernel = NativeNestedKernel(reference.jax_unit_integrand(device=device), n_dimensions=4,
            n_live_points=128, num_delete=32, num_inner_steps=12, batched=True)
        positions = np.random.default_rng(20261014).uniform(size=(128, 4))
        key = jax.random.PRNGKey(20261014)
        state = kernel.initialize(positions, key)
        logl = np.asarray(state.particles.loglikelihood)
        threshold = float(np.sort(logl)[31])
        survivors = np.flatnonzero(logl > threshold)
        assert survivors.size, "profile input has no strict contour survivors"
        parents = np.resize(survivors, 32)
        starts = jax.tree.map(lambda leaf, idx=parents: leaf[idx], state.particles)
        chain_keys = jax.random.split(key, 32)

        def scalar():
            finals, infos = [], []
            for lane, chain_key in enumerate(chain_keys):
                particle = jax.tree.map(lambda leaf, idx=lane: leaf[idx], starts)
                steps = []
                for slice_key in jax.random.split(chain_key, 12):
                    particle, info = jax.block_until_ready(kernel.budgeted_slice(slice_key, particle,
                        jnp.asarray(threshold), jnp.asarray(_SLICE_CALL_BOUND, dtype=jnp.int64)))
                    steps.append(info)
                finals.append(particle)
                infos.append(jax.tree.map(lambda *leaves: jnp.stack(leaves), *steps))
            return jax.block_until_ready((jax.tree.map(lambda *leaves: jnp.stack(leaves), *finals),
                                          jax.tree.map(lambda *leaves: jnp.stack(leaves), *infos)))

        def batched():
            return jax.block_until_ready(kernel.batched_chains(chain_keys, starts, jnp.asarray(threshold)))

        def grouped(width):
            # Profiling only: invoke the actual production batch entry point
            # on disjoint sub-batches with unchanged parents/keys/inner steps.
            parts = []
            for lo in range(0, 32, width):
                subset = jax.tree.map(lambda leaf, lo=lo: leaf[lo:lo + width], starts)
                parts.append(jax.block_until_ready(kernel.batched_chains(
                    chain_keys[lo:lo + width], subset, jnp.asarray(threshold))))
            return jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *parts)

        scalar_start = time.monotonic()
        expected, expected_info = scalar()
        scalar_cold = time.monotonic() - scalar_start
        batch_start = time.monotonic()
        actual, actual_info = batched()
        batch_cold = time.monotonic() - batch_start
        timings = {"scalar": [], "batched": []}
        windows = []
        for repetition in range(3):
            # Alternate order; neither side is always given the warmest cache.
            order = (("scalar", scalar), ("batched", batched))
            if repetition % 2:
                order = order[::-1]
            for name, function in order:
                before = time.monotonic()
                function()
                after = time.monotonic()
                timings[name].append(after - before)
                windows.append({"mode": name, "start": before, "end": after})
        exact = all(np.array_equal(left, right) for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected)))
        logical_equal = np.array_equal(actual_info.likelihood_calls, expected_info.likelihood_calls)
        group_profiles, group_infos = [], []
        for width in (8, 16):
            before = time.monotonic()
            grouped_result = grouped(width)
            cold = time.monotonic() - before
            before = time.monotonic()
            grouped_result = grouped(width)
            warmed = time.monotonic() - before
            _, grouped_info = grouped_result
            group_infos.append((f"grouped{width}", grouped_info))
            assert np.asarray(grouped_info.numerically_valid).all()
            assert np.asarray(grouped_info.slice_info.is_accepted).all()
            counts = np.asarray(grouped_info.likelihood_calls)
            # Structural synchronized-while slots, NOT measured physical native
            # calls: compiler lowering and inner integrand masking are unknown.
            slots = sum(int(counts[lo:lo + width].max(axis=0).sum()) * width
                        for lo in range(0, 32, width))
            group_profiles.append({"chains_per_dispatch": width, "dispatches": 32 // width,
                "cold_seconds": cold, "warm_seconds": warmed,
                "logical_calls": int(counts.sum()), "inferred_synchronized_proposal_slots": slots,
                "physical_native_evaluations": None,
                "limit": "one warmed development timing, no statistically established speedup"})
        valid = bool(np.asarray(actual_info.numerically_valid).all()
                     and np.asarray(expected_info.numerically_valid).all())
        accepted = bool(np.asarray(actual_info.slice_info.is_accepted).all()
                         and np.asarray(expected_info.slice_info.is_accepted).all())
        point_checks = {}
        evaluate = jax.jit(kernel.integrand.evaluate)
        for mode, info in (("scalar", expected_info), ("batched", actual_info), *group_infos):
            point_values = [jax.block_until_ready(evaluate(point))
                            for point in info.last_position.reshape(-1, 4)]
            reference_logl = np.asarray([float(value.log_integrand) for value in point_values])
            reference_valid = np.asarray([bool(value.numerically_valid) for value in point_values])
            cached = np.asarray(info.last_log_integrand).reshape(-1)
            finite = np.isfinite(reference_logl) & np.isfinite(cached)
            counts = np.asarray(info.likelihood_calls).sum(axis=-1)
            point_checks[mode] = {
                "max_abs_difference_nats": float(np.max(np.abs(cached[finite] - reference_logl[finite])))
                    if finite.any() else None,
                "classes_equal": bool(np.array_equal(np.isfinite(cached), np.isfinite(reference_logl))
                                      and np.array_equal(np.isneginf(cached), np.isneginf(reference_logl))
                                      and np.array_equal(np.asarray(info.numerically_valid).reshape(-1), reference_valid)),
                "finite_gate_passed": bool(np.allclose(cached[finite], reference_logl[finite], rtol=0.0, atol=1e-10)),
                "counts_consistent": bool(np.array_equal(counts,
                    2 + np.asarray(info.slice_info.num_expansions) + np.asarray(info.slice_info.num_shrink))
                    and np.array_equal(info.remaining_budget, _SLICE_CALL_BOUND - counts)),
            }
        record = {"device": str(device), "device_id": device.id, "chains": 32, "steps": 12,
            "dimensions": 4, "native_integrand": "real-fixed-host-TP", "survivors": int(survivors.size),
            "scalar_cold_seconds": scalar_cold, "batched_cold_seconds": batch_cold,
            "warm_seconds": timings, "timing_windows": windows,
            "warm_scalar_over_batched": float(np.median(timings["scalar"]) / np.median(timings["batched"])),
            "device_profile_wall_seconds": time.monotonic() - started,
            "replacements_byte_equal": exact, "logical_counts_equal": logical_equal,
            "numerically_valid": valid, "all_slices_accepted": accepted,
            "scalar_logical_calls": int(np.asarray(expected_info.likelihood_calls).sum()),
            "batched_logical_calls": int(np.asarray(actual_info.likelihood_calls).sum()),
            "batched_physical_calls": None,
            "inferred_full_batch_synchronized_proposal_slots":
                int(np.asarray(actual_info.likelihood_calls).max(axis=0).sum()) * 32,
            "grouped_dispatch_profiles": group_profiles,
            "slot_inference_limit": "max proposal count per synchronized step/phase times active batch width; not a hardware evaluation counter or likelihood-internal work count",
            "same_point_checks": point_checks, "absolute_gate_nats": 1e-10,
            "cross_mode_bitwise_equality_is_acceptance": False,
            "allocator_counters": device.memory_stats(),
            "claim_eligible": False, "evidence_calibrated": False}
        return record


def test_short_dual_gpu_real_tp_deletion_batch_profile(record_property):
    try:
        devices = models.jax.devices("gpu")
    except RuntimeError:
        devices = []
    if len(devices) < 2:
        pytest.skip("short physical dual-GPU profile requires two available GPU devices")
    done = Event()
    samples = []
    started = time.monotonic()

    def sample():
        while not done.is_set():
            stamp = time.monotonic()
            try:
                result = subprocess.run(["nvidia-smi",
                    "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free",
                    "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=2, check=False)
                samples.append({"monotonic": stamp, "returncode": result.returncode,
                                "stdout": result.stdout, "stderr": result.stderr})
            except (OSError, subprocess.TimeoutExpired) as error:
                samples.append({"monotonic": stamp, "unavailable": f"{type(error).__name__}: {error}"})
            done.wait(0.5)

    worker = Thread(target=sample, daemon=True)
    worker.start()
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(_profile, devices[:2]))
    finally:
        done.set()
        worker.join(3)
        record = {"scope": "Constrained-batch microprofile; not full integration speed or calibration",
            "whole_profile_wall_seconds": time.monotonic() - started,
            "devices": rows, "gpu_samples": samples, "telemetry_thread_stopped": not worker.is_alive()}
        payload = json.dumps(record, default=str)
        record_property("dual_gpu_profile", payload)
        output = Path("/kaggle/working")
        if output.is_dir():
            (output / "blackjax_dual_gpu_profile.json").write_text(payload + "\n")
    assert len(rows) == 2
    assert all(row["numerically_valid"] and row["all_slices_accepted"] for row in rows)
    assert all(check["classes_equal"] and check["finite_gate_passed"] and check["counts_consistent"]
               for row in rows for check in row["same_point_checks"].values())
