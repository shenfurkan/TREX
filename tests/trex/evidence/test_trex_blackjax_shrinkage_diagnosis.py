"""Retain and regress the first interrupted 3200-live assessment seed.

7022 follows the 22 retained completed seeds 7000..7021. Save the unresolved
production audit and replay failed lanes from their original parents/keys.
Old-source failure receipts remain under tools/kaggle_runs/. A repaired replay
is software regression coverage, not successful independent calibration.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import NativeNestedKernel
from trex.evidence._blackjax_types import NativeIntegrationError, SLICE_CALL_BOUND
from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand


def _leaves(prefix, value):
    return {prefix + models.jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in models.jax.tree.flatten_with_path(value)[0]}


def test_retain_and_replay_seed7022_shrinkage_failure(tmp_path, record_property):
    jax, jnp = models.jax, models.jnp
    device = jax.devices("cpu")[0]
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = tmp_path
    with jax.default_device(device):
        native = GaussianIntegrand(device)
        kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=3200,
                                    num_delete=1600, num_inner_steps=12, batched=True)
        try:
            result = kernel.integrate(np.random.default_rng(7022).uniform(size=(3200, 4)),
                jax.random.PRNGKey(7022), evidence_tolerance=0.05,
                max_likelihood_calls=4_000_000, random_seed=7022)
        except NativeIntegrationError as error:
            result = error.result
        else:
            # The original failure is retained in immutable remote receipts.
            # After repair, require full convergence and eligibility at every
            # actual dispatched batch, not merely absence of the old exception.
            _assert_strict_parents(result)
            assert result.stop_reason == "evidence-tolerance"
            assert result.n_likelihood_calls <= 4_000_000
            assert result.claim_eligible is result.evidence_calibrated is False
            payload = json.dumps({"seed": 7022, "role": "development regression, not calibration",
                "stop_reason": result.stop_reason, "logical_calls": result.n_likelihood_calls,
                "batches": result.n_iterations, "strict_parent_audit": "passed"}, indent=2)
            (output / "blackjax_parent_repaired_seed7022.json").write_text(payload + "\n")
            record_property("repaired_parent_replay", payload)
            return
        record = {"seed": 7022, "failure": result.failure_reason, "stop_reason": result.stop_reason,
            "failed_operation": result.failed_operation, "completed_batches": result.n_iterations,
            "logical_calls": result.n_likelihood_calls, "exact_logical_calls": result.likelihood_calls_exact,
            "physical_calls": result.physical_likelihood_calls, "max_logical_calls": result.max_likelihood_calls,
            "live": result.n_live_points, "delete": result.num_delete, "inner_steps": result.num_inner_steps,
            "claim_eligible": False, "evidence_calibrated": False, "replay_slices": []}
        arrays = {"initial_positions": result.initial_positions,
                  **_leaves("initial_state", result.initial_state), **_leaves("last_consistent_state", result.final_state)}
        batch = result.batches[-1]
        record.update(batch_threshold=batch.threshold, failed_batch_completed=batch.completed,
                      calls_before_failed_batch=batch.calls_before, calls_after_failed_batch=batch.calls_after)
        if batch.batched_slices is not None:
            chain_keys, parents, infos = batch.batched_slices
            live_logl = np.asarray(result.final_state.particles.loglikelihood)
            _, inner_key = jax.random.split(batch.key)
            choice_key, sample_key = jax.random.split(inner_key)
            weights = jnp.asarray(live_logl > batch.threshold, dtype=jnp.float32)
            probabilities = weights / weights.sum()
            repeated_parents = jax.random.choice(choice_key, 3200, (len(parents),),
                                                 p=probabilities, replace=True)
            cumul = jnp.cumsum(probabilities)
            draws = cumul[-1] * (1 - jax.random.uniform(choice_key, (len(parents),), dtype=cumul.dtype))
            parent_values = live_logl[np.asarray(parents)]
            invalid_parents = np.flatnonzero(parent_values <= batch.threshold)
            record["parent_selection"] = {
                "invalid_lanes": invalid_parents.tolist(),
                "invalid_parent_indices": np.asarray(parents)[invalid_parents].tolist(),
                "invalid_parent_loglikelihoods": parent_values[invalid_parents].tolist(),
                "parent_weights": np.asarray(weights)[np.asarray(parents)[invalid_parents]].tolist(),
                "choice_key": np.asarray(choice_key).tolist(),
                "repeated_choice_equal": bool(np.array_equal(parents, repeated_parents)),
                "chain_keys_equal": bool(np.array_equal(chain_keys, jax.random.split(sample_key, len(parents)))),
                "invalid_lane_uniform_cdf_draws": np.asarray(draws)[invalid_parents].tolist(),
                "invalid_parent_cdf_values": np.asarray(cumul)[np.asarray(parents)[invalid_parents]].tolist(),
            }
            arrays.update(parent_weights=np.asarray(weights), parent_probabilities=np.asarray(probabilities),
                          parent_cdf=np.asarray(cumul), parent_cdf_draws=np.asarray(draws),
                          choice_key=np.asarray(choice_key))
            failed = np.argwhere(~np.asarray(infos.slice_info.is_accepted))
            arrays.update({"chain_keys": np.asarray(chain_keys), "parents": np.asarray(parents),
                           "failed_indices": failed, **_leaves("batch_info", infos),
                           **_leaves("batch_replacements", batch.replacement_particles)})
            record["failed_slices"] = failed.tolist()
            record["validity_all"] = bool(np.asarray(infos.numerically_valid).all())
            record["budget_exhausted_any"] = bool(np.asarray(infos.budget_exhausted).any())
            # Replay only affected chains through the actual scalar production
            # entry point, preserving each of their original inner keys.
            for lane in np.unique(failed[:, 0]):
                current = jax.tree.map(lambda leaf, parent=int(parents[lane]): leaf[parent],
                                       result.final_state.particles)
                keys = jax.random.split(chain_keys[lane], 12)
                for step, key in enumerate(keys):
                    start = current
                    checked_start = jax.block_until_ready(kernel.integration_evaluate(
                        start.position, start.loglikelihood_birth))[0]
                    current, info = jax.block_until_ready(kernel.budgeted_slice(key, start,
                        jnp.asarray(batch.threshold), jnp.asarray(SLICE_CALL_BOUND, dtype=jnp.int64)))
                    record["replay_slices"].append({"lane": int(lane), "step": step,
                        "key": np.asarray(key).tolist(), "start": np.asarray(start.position).tolist(),
                        "start_cached_logl": float(start.loglikelihood),
                        "start_recomputed_logl": float(checked_start.loglikelihood),
                        "cached_above_contour": bool(start.loglikelihood > batch.threshold),
                        "recomputed_above_contour": bool(checked_start.loglikelihood > batch.threshold),
                        "scalar_accepted": bool(info.slice_info.is_accepted),
                        "batch_accepted": bool(infos.slice_info.is_accepted[lane, step]),
                        "scalar_calls": np.asarray(info.likelihood_calls).tolist(),
                        "batch_calls": np.asarray(infos.likelihood_calls[lane, step]).tolist(),
                        "scalar_last_logl": float(info.last_log_integrand),
                        "batch_last_logl": float(infos.last_log_integrand[lane, step]),
                        "same_last_position": bool(np.array_equal(info.last_position, infos.last_position[lane, step])),
                        "scalar_bracket": [float(info.slice_info.bracket_left), float(info.slice_info.bracket_right)],
                        "batch_bracket": [float(infos.slice_info.bracket_left[lane, step]),
                                          float(infos.slice_info.bracket_right[lane, step])],
                    })
                    arrays.update(_leaves(f"lane{lane}_step{step}_start", start))
                    arrays.update(_leaves(f"lane{lane}_step{step}_scalar_info", info))
                    arrays.update(_leaves(f"lane{lane}_step{step}_scalar_proposal", current))
        else:
            record["scalar_failed_attempts"] = len(batch.slices)
            for index, attempt in enumerate(batch.slices):
                arrays.update(_leaves(f"scalar_slice{index}_start", attempt.start))
                arrays.update(_leaves(f"scalar_slice{index}_info", attempt.info))
        np.savez_compressed(output / "blackjax_shrinkage_seed7022.npz", **arrays)
        payload = json.dumps(record, indent=2)
        (output / "blackjax_shrinkage_seed7022.json").write_text(payload + "\n")
        record_property("shrinkage_replay", payload)
        assert result.failure_reason == "native slice shrinkage exhausted", payload
        assert result.stop_reason == "unresolved" and not batch.completed
        assert result.posterior_unit_samples is result.posterior_weights is None
        assert batch.batched_slices is not None, payload
        assert record["validity_all"] and not record["budget_exhausted_any"], payload


def _assert_strict_parents(result):
    live = np.asarray(result.initial_state.particles.loglikelihood).copy()
    for index, batch in enumerate(result.batches):
        if batch.batched_slices is not None:
            _, parents, _ = batch.batched_slices
        else:
            parents = [attempt.particle_index for attempt in batch.slices]
        assert (live[parents] > batch.threshold).all(), f"batch {index} selected a non-survivor parent"
        if batch.completed:
            if batch.batched_slices is not None:
                live[batch.dead_indices] = np.asarray(batch.replacement_particles.loglikelihood)
            else:
                live[batch.dead_indices] = [float(point.loglikelihood) for point in batch.replacement_particles]


@pytest.mark.parametrize("budget", [3600, 4_000_000])
def test_all_selected_parents_are_strict_survivors_before_slice_dispatch(budget):
    """Fail at the actual production boundary, independent of shrinkage caps."""
    jax = models.jax
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        kernel = NativeNestedKernel(GaussianIntegrand(device), n_dimensions=4, n_live_points=3200,
                                    num_delete=1600, num_inner_steps=12, batched=True)
        try:
            result = kernel.integrate(np.random.default_rng(7022).uniform(size=(3200, 4)),
                jax.random.PRNGKey(7022), evidence_tolerance=0.05,
                max_likelihood_calls=budget, random_seed=7022)
        except NativeIntegrationError as error:
            result = error.result
        # Replay the real committed live sets, not an engine stand-in.
        _assert_strict_parents(result)
        assert result.stop_reason == ("truncated" if budget == 3600 else "evidence-tolerance")
        assert result.n_likelihood_calls <= budget
        assert result.claim_eligible is result.evidence_calibrated is False
