"""Same-point native TP float64/support agreement and device throughput.

Uses the existing absolute 1e-10-nat gate, not trajectory byte equality.
Sequential_vmap is a test-only execution reference around the actual native
model. Fixed unit points and boundary probes are inputs, not expected science.
Cold and three alternating warmed repetitions include synchronization, exclude
upload/host readback, and do not establish nested-evidence calibration.
"""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


@pytest.mark.parametrize("placement", ["cpu:0", "gpu:0", "gpu:1"])
def test_same_point_tp_vectorization_gate_and_throughput(placement, tmp_path, record_property):
    platform, index = placement.split(":")
    jax, jnp = models.jax, models.jnp
    try:
        devices = jax.devices(platform)
    except RuntimeError:
        devices = []
    if len(devices) <= int(index):
        pytest.skip(f"{placement} unavailable; no accelerator throughput claim")
    device = devices[int(index)]
    with jax.default_device(device):
        reference = nested_evidence._TransitingPlanetIntegrand(
            **synthetic_trex_tp_integrand_case(), host_mass=1.0, host_radius=1.0,
            bin_integration_days=0.0, nsamples=3, use_flat_priors=False,
            population=None, n_dimensions=4)
        native = reference.jax_unit_integrand(device=device)
        # Diagnostic input population; the first 128 draws reproduce the prior
        # population of the retained NSS red. Each mode gets identical points.
        uniform = np.random.default_rng(20261014).uniform(size=(1024, 4))
        probes = []
        for dimension in range(4):
            for value in (-np.finfo(float).eps, 0.0, np.nextafter(1.0, 0.0), 1.0, np.nan):
                point = np.full(4, 0.5)
                point[dimension] = value
                probes.append(point)
        points = jax.device_put(np.concatenate((uniform, probes)), device)
        functions = {
            "vectorized": jax.jit(jax.vmap(native.evaluate)),
            "sequential_vmap_reference": jax.jit(jax.vmap(
                jax.custom_batching.sequential_vmap(native.evaluate))),
            "scalar_map_reference": jax.jit(lambda p: jax.lax.map(native.evaluate, p)),
        }
        single = jax.jit(native.evaluate)
        scalar_values = [jax.block_until_ready(single(point)) for point in points]
        scalar_values = jax.tree.map(lambda *leaves: jnp.stack(leaves), *scalar_values)
        values, cold = {}, {}
        for name, function in functions.items():
            before = time.monotonic()
            values[name] = jax.block_until_ready(function(points))
            cold[name] = time.monotonic() - before
        warm = {name: [] for name in functions}
        for repetition in range(3):
            order = list(functions) if repetition % 2 == 0 else list(reversed(functions))
            for name in order:
                before = time.monotonic()
                received = jax.block_until_ready(functions[name](points))
                warm[name].append(time.monotonic() - before)
                # Same executable/input replay, including NaNs at invalid probes.
                for left, right in zip(jax.tree.leaves(received), jax.tree.leaves(values[name])):
                    np.testing.assert_array_equal(left, right)
        expected = np.asarray(scalar_values.log_integrand)
        validity = np.asarray(scalar_values.numerically_valid)
        comparisons = {}
        for name in functions:
            actual = np.asarray(values[name].log_integrand)
            actual_valid = np.asarray(values[name].numerically_valid)
            finite = np.isfinite(expected) & np.isfinite(actual)
            delta = np.abs(actual[finite] - expected[finite])
            comparisons[name] = {
                "max_abs_difference_nats": float(delta.max()) if delta.size else None,
                "rms_difference_nats": float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                "unequal_finite_values": int(np.count_nonzero(actual[finite] != expected[finite])),
                "support_mismatches": int(np.count_nonzero(np.isneginf(actual) != np.isneginf(expected))),
                "validity_mismatches": int(np.count_nonzero(actual_valid != validity)),
                "nan_mismatches": int(np.count_nonzero(np.isnan(actual) != np.isnan(expected))),
                "finite_class_mismatches": int(np.count_nonzero(np.isfinite(actual) != np.isfinite(expected))),
                "outside_cube_valid_zero": bool(np.all(actual[1024:][[0, 3, 5, 8, 10, 13, 15, 18]] == -np.inf)),
                "nonfinite_input_invalid": bool(np.all(~actual_valid[1024:][[4, 9, 14, 19]])),
            }
        medians = {name: float(np.median(samples)) for name, samples in warm.items()}
        record = {
            "scope": "Native TP same-point likelihood, not trajectory/evidence calibration",
            "device": str(device), "seed": 20261014, "points": len(points), "uniform_points": 1024,
            "boundary_probes": 20, "finite_reference_points": int(np.isfinite(expected).sum()),
            "physical_zero_reference_points": int(np.isneginf(expected).sum()),
            "invalid_reference_points": int((~validity).sum()),
            "absolute_gate_nats": 1e-10, "relative_gate": 0.0,
            "comparison_to_individually_jitted_scalar": comparisons,
            "cold_seconds": cold, "warm_seconds": warm, "median_warm_seconds": medians,
            "warm_points_per_second": {name: len(points) / seconds for name, seconds in medians.items()},
            "vectorized_speedup_over_sequential_vmap": medians["sequential_vmap_reference"] / medians["vectorized"],
            "timing_scope": "Device-resident identical points; synchronize complete model output; excludes upload/readback",
            "allocator_counters": device.memory_stats(), "claim_eligible": False, "evidence_calibrated": False,
        }
        output = Path("/kaggle/working")
        if not output.is_dir():
            output = tmp_path
        stem = f"tp_batch_precision_{platform}{index}"
        (output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        np.savez_compressed(output / f"{stem}.npz", points=np.asarray(points),
            scalar_log_integrand=expected, scalar_numerically_valid=validity,
            **{name + "_log_integrand": np.asarray(v.log_integrand) for name, v in values.items()},
            **{name + "_numerically_valid": np.asarray(v.numerically_valid) for name, v in values.items()})
        record_property("tp_same_point_precision_and_throughput", json.dumps(record))
        assert np.isfinite(expected).any()
        for name, value in values.items():
            row = comparisons[name]
            assert row["support_mismatches"] == row["validity_mismatches"] == row["nan_mismatches"] == 0, record
            assert row["finite_class_mismatches"] == 0, record
            assert row["outside_cube_valid_zero"] and row["nonfinite_input_invalid"], record
            actual = np.asarray(value.log_integrand)
            assert actual.dtype == expected.dtype == np.dtype("float64")
            finite = np.isfinite(expected)
            np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.0, atol=1e-10)
