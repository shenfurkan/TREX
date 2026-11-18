"""Per-point float64 acceptance and retained cross-mode transition diagnostics.

The original seed, 128 live points, 32 chains and 12 slices are unchanged.
Execute the production slice/chains entry points, retain every returned leaf
and independently reevaluate the first unequal point. Under the operator's
revised contract, cross-mode trajectories are diagnostics, not bitwise truth.
Each cached value is tested at its own identical point against scalar native
evaluation at the existing 1e-10-nat gate. Calibration is a separate requirement.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_kernel import NativeNestedKernel, _SLICE_CALL_BOUND
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _differences(expected, actual):
    """Locate unequal returned values without discarding finite/support flags."""
    rows = {}
    jax = models.jax
    for (path, left), (_, right) in zip(jax.tree.flatten_with_path(expected)[0],
                                       jax.tree.flatten_with_path(actual)[0]):
        left, right = np.asarray(left), np.asarray(right)
        unequal = left != right
        indices = np.argwhere(unequal)
        if not unequal.any():
            continue
        first = tuple(indices[0])
        finite = unequal & np.isfinite(left) & np.isfinite(right)
        rows[jax.tree_util.keystr(path)] = {
            "unequal_values": int(unequal.sum()), "shape": list(left.shape),
            "first_index": list(first), "expected": str(left[first]), "actual": str(right[first]),
            "max_abs_finite_difference": float(np.max(np.abs(
                left[finite].astype(float) - right[finite].astype(float))))
                if finite.any() else None,
        }
    return rows


def _arrays(prefix, tree):
    return {prefix + models.jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in models.jax.tree.flatten_with_path(tree)[0]}


@pytest.mark.parametrize("placement", ["cpu:0", "gpu:0", "gpu:1"])
def test_real_tp_transition_same_point_float64_gate(placement, tmp_path, record_property):
    platform, index = placement.split(":")
    jax, jnp = models.jax, models.jnp
    try:
        devices = jax.devices(platform)
    except RuntimeError:
        devices = []
    if len(devices) <= int(index):
        pytest.skip(f"{placement} unavailable; not physical-device acceptance")
    device = devices[int(index)]
    with jax.default_device(device):
        reference = nested_evidence._TransitingPlanetIntegrand(
            **synthetic_trex_tp_integrand_case(), host_mass=1.0, host_radius=1.0,
            bin_integration_days=0.0, nsamples=3, use_flat_priors=False,
            population=None, n_dimensions=4)
        native = reference.jax_unit_integrand(device=device)
        kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=128,
                                    num_delete=32, num_inner_steps=12, batched=True)
        positions = np.random.default_rng(20261014).uniform(size=(128, 4))
        key = jax.random.PRNGKey(20261014)
        state = kernel.initialize(positions, key)
        logl = np.asarray(state.particles.loglikelihood)
        threshold = jnp.asarray(np.sort(logl)[31])
        survivors = np.flatnonzero(logl > float(threshold))
        assert survivors.size, "original real-TP case has no contour survivors"
        parents = np.resize(survivors, 32)
        starts = jax.tree.map(lambda leaf: leaf[parents], state.particles)
        chain_keys = jax.random.split(key, 32)
        keys = jax.vmap(lambda k: jax.random.split(k, 12))(chain_keys)
        scalar_finals, scalar_infos, scalar_states = [], [], []
        for lane in range(32):
            particle = jax.tree.map(lambda leaf, lane=lane: leaf[lane], starts)
            infos, states = [], []
            for slice_key in keys[lane]:
                particle, info = jax.block_until_ready(kernel.budgeted_slice(
                    slice_key, particle, threshold, jnp.asarray(_SLICE_CALL_BOUND, dtype=jnp.int64)))
                infos.append(info)
                states.append(particle)
            scalar_finals.append(particle)
            scalar_infos.append(jax.tree.map(lambda *x: jnp.stack(x), *infos))
            scalar_states.append(jax.tree.map(lambda *x: jnp.stack(x), *states))
        expected = jax.tree.map(lambda *x: jnp.stack(x), *scalar_finals)
        expected_info = jax.tree.map(lambda *x: jnp.stack(x), *scalar_infos)
        expected_states = jax.tree.map(lambda *x: jnp.stack(x), *scalar_states)
        actual, actual_info = jax.block_until_ready(kernel.batched_chains(chain_keys, starts, threshold))
        differences = _differences(expected_info, actual_info)
        unequal_slices = np.zeros((32, 12), dtype=bool)
        for left, right in zip(jax.tree.leaves(expected_info), jax.tree.leaves(actual_info)):
            mask = np.asarray(left) != np.asarray(right)
            if mask.ndim > 2:
                mask = np.any(mask, axis=tuple(range(2, mask.ndim)))
            unequal_slices |= mask
        first = min((tuple(pair) for pair in np.argwhere(unequal_slices)),
                    key=lambda pair: (pair[1], pair[0]), default=None)
        record = {"scope": "Same-point float64/support acceptance; cross-mode replay is diagnostic, not calibration",
            "device": str(device), "seed": 20261014, "live": 128, "chains": 32, "steps": 12,
            "threshold": float(threshold), "info_differences": differences,
            "final_differences": _differences(expected, actual),
            "first_unequal_slice": list(first) if first is not None else None,
            "first_unequal_order": "earliest step, then chain",
            "claim_eligible": False, "evidence_calibrated": False}
        arrays = {"positions": positions, "parents": parents, "chain_keys": np.asarray(chain_keys),
                  "slice_keys": np.asarray(keys), "threshold": np.asarray(threshold),
                  **_arrays("starts", starts), **_arrays("scalar_states", expected_states),
                  **_arrays("scalar_info", expected_info), **_arrays("batch_info", actual_info),
                  **_arrays("scalar_final", expected), **_arrays("batch_final", actual)}
        if first is not None:
            lane, step = first
            # Accepted slices end at last_position; verify acceptance below before
            # treating it as a replacement. Reevaluate both returned points using
            # the actual native model, not a copied sampler or stand-in likelihood.
            points = jnp.stack((expected_info.last_position[lane, step],
                                actual_info.last_position[lane, step]))
            scalar_eval = [jax.block_until_ready(jax.jit(native.evaluate)(p)) for p in points]
            scalar_eval = jax.tree.map(lambda *x: jnp.stack(x), *scalar_eval)
            vector_eval = jax.block_until_ready(jax.jit(jax.vmap(native.evaluate))(points))
            mapped_eval = jax.block_until_ready(jax.jit(lambda p: jax.lax.map(native.evaluate, p))(points))
            scalar_prior = jnp.stack([jax.jit(native.prior_transform)(p) for p in points])
            vector_prior = jax.block_until_ready(jax.jit(jax.vmap(native.prior_transform))(points))
            independent_prior = np.asarray([reference.prior_transform(np.asarray(p)) for p in points])
            independent_logl = [reference.log_likelihood(p) for p in independent_prior]
            previous = jax.tree.map(lambda leaf: leaf[lane] if step == 0 else leaf[lane, step - 1],
                                    starts if step == 0 else expected_states)
            replay, replay_info = jax.block_until_ready(kernel.budgeted_slice(
                keys[lane, step], previous, threshold, jnp.asarray(_SLICE_CALL_BOUND, dtype=jnp.int64)))
            record["first_slice"] = {
                "key": np.asarray(keys[lane, step]).tolist(),
                "same_returned_point": bool(np.array_equal(points[0], points[1])),
                "cached_log_integrands": [float(expected_info.last_log_integrand[lane, step]),
                                          float(actual_info.last_log_integrand[lane, step])],
                "native_scalar_log_integrands": np.asarray(scalar_eval.log_integrand).tolist(),
                "native_vector_log_integrands": np.asarray(vector_eval.log_integrand).tolist(),
                "native_mapped_log_integrands": np.asarray(mapped_eval.log_integrand).tolist(),
                "native_scalar_valid": np.asarray(scalar_eval.numerically_valid).tolist(),
                "native_vector_valid": np.asarray(vector_eval.numerically_valid).tolist(),
                "prior_scalar_vector_differences": _differences(scalar_prior, vector_prior),
                "independent_prior": independent_prior.tolist(),
                "independent_log_integrands": independent_logl,
                "independent_reference_limit": "SciPy/batman forward model; existing discretization tolerances apply",
                "scalar_replay_differences": _differences(
                    (jax.tree.map(lambda leaf: leaf[lane, step], expected_states),
                     jax.tree.map(lambda leaf: leaf[lane, step], expected_info)), (replay, replay_info)),
            }
            arrays.update({"first_points": np.asarray(points), "first_native_prior_scalar": np.asarray(scalar_prior),
                           "first_native_prior_vector": np.asarray(vector_prior),
                           "first_independent_prior": independent_prior,
                           **_arrays("first_native_scalar", scalar_eval),
                           **_arrays("first_native_vector", vector_eval),
                           **_arrays("first_native_mapped", mapped_eval)})
        same_point_comparisons = {}
        scalar_evaluate = jax.jit(native.evaluate)
        for mode, info in (("scalar", expected_info), ("vectorized", actual_info)):
            points = info.last_position.reshape(-1, 4)
            evaluations = [jax.block_until_ready(scalar_evaluate(point)) for point in points]
            evaluations = jax.tree.map(lambda *leaves: jnp.stack(leaves), *evaluations)
            cached = np.asarray(info.last_log_integrand).reshape(-1)
            reference_values = np.asarray(evaluations.log_integrand)
            same_point_comparisons[mode] = {
                "max_abs_cached_difference_nats": float(np.max(np.abs(cached - reference_values))),
                "validity_equal": bool(np.array_equal(np.asarray(info.numerically_valid).reshape(-1),
                                                       np.asarray(evaluations.numerically_valid))),
                "support_equal": bool(np.array_equal(np.isneginf(cached), np.isneginf(reference_values))),
            }
            arrays[mode + "_same_point_reference_log_integrand"] = reference_values
        record["same_point_comparisons"] = same_point_comparisons
        record["absolute_gate_nats"] = 1e-10
        record["cross_mode_bitwise_equality_is_acceptance"] = False
        output = Path("/kaggle/working")
        if not output.is_dir():
            output = tmp_path
        stem = f"blackjax_transition_replay_{platform}{index}"
        np.savez_compressed(output / f"{stem}.npz", **arrays)
        payload = json.dumps(record, default=str, indent=2)
        (output / f"{stem}.json").write_text(payload + "\n")
        record_property("transition_replay", payload)
        assert np.asarray(expected_info.numerically_valid).all() and np.asarray(actual_info.numerically_valid).all()
        assert np.asarray(expected_info.slice_info.is_accepted).all() and np.asarray(actual_info.slice_info.is_accepted).all()
        # Exact bookkeeping within each execution, not a demand that stochastic
        # cross-mode accept/reject paths consume identical numbers of proposals.
        for info in (expected_info, actual_info):
            counts = np.asarray(info.likelihood_calls).sum(axis=-1)
            np.testing.assert_array_equal(counts,
                2 + np.asarray(info.slice_info.num_expansions) + np.asarray(info.slice_info.num_shrink))
            np.testing.assert_array_equal(info.remaining_budget, _SLICE_CALL_BOUND - counts)
            assert not np.asarray(info.budget_exhausted).any()
            assert (np.asarray(info.last_log_integrand) > float(threshold)).all()
        for mode, row in same_point_comparisons.items():
            assert row["validity_equal"] and row["support_equal"], payload
            cached = np.asarray(expected_info.last_log_integrand if mode == "scalar"
                                else actual_info.last_log_integrand).reshape(-1)
            np.testing.assert_allclose(cached, arrays[mode + "_same_point_reference_log_integrand"],
                                       rtol=0.0, atol=1e-10)
        if "first_slice" in record:
            assert not record["first_slice"]["scalar_replay_differences"], payload
