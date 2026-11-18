"""Predeclared causal diagnostics of the unchanged known-Z development reds.

Development seeds 17/29/43, 128 live points, deletion=1, eight inner steps;
compare stopping tolerances 0.1 and 0.001 without changing existing tests.
The deeper diagnostic ceiling is 200,000 calls after seed 17 exhausted the
original 100,000-call diagnostic allowance (retained 20261014-130130 receipt).
The analytic contour volume for L(u)=exp(-4u) is X=u. It is used ONLY as
an independent diagnostic oracle, never as input to the production driver.
Stationarity controls use 4096 independent starts at contours 1/0.1/0.01,
seed 20261014, and report KS statistics and correlation without selecting
favorable seeds or claiming calibration from a non-rejection.
Precision experiment declared after causal decomposition, before execution:
1024 live points, eight inner steps, unchanged 0.1-nat accuracy and stopping
bar; development seeds 17/29/43 and fresh assessment seeds 3000/3001/3002.
This six-run experiment is not a 95% coverage or production adoption claim.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kstest

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import NativeNestedKernel
from tests.trex.evidence.test_blackjax_evidence import AnalyticIntegrand


def _retain(name, record, record_property):
    payload = json.dumps(record, sort_keys=True, allow_nan=False)
    record_property("known_z_diagnosis", payload)
    output = Path("/kaggle/working")
    if output.is_dir():
        (output / f"blackjax_known_z_diagnosis_{name}.json").write_text(payload + "\n")


def _analytic_decomposition(result):
    """Partition contour, quadrature and live-tail effects using exact calculus."""
    completed = [batch for batch in result.batches if batch.completed]
    dead_u = np.concatenate([np.asarray(batch.dead_particles.position)[:, 0] for batch in completed])
    assert dead_u.size == result.effective_live_counts.size
    assert np.all(np.diff(dead_u) < 0.0)
    previous_u = np.concatenate(([1.0], dead_u[:-1]))
    previous_assumed = np.concatenate(([1.0], np.exp(result.dead_log_volumes[:-1])))
    assumed = np.exp(result.dead_log_volumes)
    actual_tail_volume = float(dead_u[-1])
    assumed_tail_volume = float(assumed[-1])
    live_u = np.asarray(result.final_state.particles.position)[:, 0]
    assert np.all((live_u >= 0.0) & (live_u < actual_tail_volume))
    truth_z = -math.expm1(-4.0) / 4.0
    exact_tail = -math.expm1(-4.0 * actual_tail_volume) / 4.0
    exact_dead = truth_z - exact_tail
    # Upper/lower Riemann sums bound the exact integral over each observed
    # interval for this monotonic function; no nested sampler is reimplemented.
    upper = float(np.sum((previous_u - dead_u) * np.exp(-4.0 * dead_u)))
    lower = float(np.sum((previous_u - dead_u) * np.exp(-4.0 * previous_u)))
    assert lower <= exact_dead <= upper
    native_dead = float(np.sum((previous_assumed - assumed) * np.exp(-4.0 * dead_u)))
    live_mean = float(np.mean(np.exp(-4.0 * live_u)))
    reconstructed = native_dead + assumed_tail_volume * live_mean
    # Independent float64 exp/sum versus the production log-sum reductions.
    np.testing.assert_allclose(math.log(reconstructed), result.log_evidence, rtol=0.0, atol=2e-13)
    components = {
        "realized_quadrature_z": upper - exact_dead,
        "assumed_minus_realized_dead_volume_z": native_dead - upper,
        "live_mean_sampling_z": actual_tail_volume * live_mean - exact_tail,
        "assumed_minus_realized_tail_volume_z": (assumed_tail_volume - actual_tail_volume) * live_mean,
    }
    np.testing.assert_allclose(sum(components.values()), reconstructed - truth_z, rtol=0.0, atol=2e-14)
    log_shrink = -np.diff(np.concatenate(([0.0], np.log(dead_u))))
    scaled_shrink = result.effective_live_counts * log_shrink
    shrink_ks = kstest(scaled_shrink, "expon")
    replacements = np.concatenate([np.asarray(batch.replacement_particles[0].position)
                                   / (-batch.threshold / 4.0) for batch in completed])
    replacement_ks = kstest(replacements, "uniform")
    return {
        "logz": result.log_evidence,
        "error_nats": result.log_evidence_error,
        "truth_logz": math.log(truth_z),
        "delta_logz": result.log_evidence - math.log(truth_z),
        "standardized_error": (result.log_evidence - math.log(truth_z)) / result.log_evidence_error,
        "calls": result.n_likelihood_calls,
        "dead_points": int(dead_u.size),
        "stop_reason": result.stop_reason,
        "delta_logz_stop": result.base_run_delta_logz,
        "actual_tail_volume": actual_tail_volume,
        "assumed_tail_volume": assumed_tail_volume,
        "known_volume_upper_logz": math.log(upper + exact_tail),
        "known_volume_lower_logz": math.log(lower + exact_tail),
        "z_error_decomposition": components,
        "scaled_realized_shrink_mean": float(scaled_shrink.mean()),
        "shrink_exponential_ks_statistic": float(shrink_ks.statistic),
        "shrink_exponential_ks_pvalue": float(shrink_ks.pvalue),
        "replacement_uniform_ks_statistic": float(replacement_ks.statistic),
        "replacement_uniform_ks_pvalue": float(replacement_ks.pvalue),
        "ks_caveat": "Nested histories are dependent; p-values are exploratory, not independent calibration",
    }


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_known_z_error_decomposition_and_tighter_stop(seed, record_property):
    device = models.jax.devices("cpu")[0]
    kernel = NativeNestedKernel(AnalyticIntegrand("exponential", device), n_dimensions=1,
        n_live_points=128, num_delete=1, num_inner_steps=8)
    positions = np.random.default_rng(seed).uniform(size=(128, 1))
    key = models.jax.random.PRNGKey(seed)
    results = []
    summaries = []
    for tolerance in (0.1, 0.001):
        result = kernel.integrate(positions, key, evidence_tolerance=tolerance,
                                  max_likelihood_calls=200_000, random_seed=seed)
        summary = {"tolerance": tolerance, **_analytic_decomposition(result)}
        _retain(f"seed{seed}_tolerance{tolerance}", {"seed": seed, "run": summary}, record_property)
        assert result.stop_reason == "evidence-tolerance"
        assert result.n_likelihood_calls <= 200_000
        assert result.claim_eligible is result.evidence_calibrated is False
        summaries.append(summary)
        results.append(result)
    baseline, deeper = results
    # Same RNG/history prefix isolates stopping; no more-live or new-seed tuning.
    np.testing.assert_array_equal(deeper.posterior_unit_samples[:baseline.effective_live_counts.size],
                                  baseline.posterior_unit_samples[:baseline.effective_live_counts.size])
    _retain(f"seed{seed}", {"seed": seed, "runs": summaries,
        "tighter_stop_logz_change": deeper.log_evidence - baseline.log_evidence}, record_property)


@pytest.mark.parametrize("contour", [1.0, 0.1, 0.01])
def test_real_constrained_kernel_uniform_stationarity_and_mixing(contour, record_property):
    from blackjax.ns import base

    jax, jnp = models.jax, models.jnp
    device = jax.devices("cpu")[0]
    kernel = NativeNestedKernel(AnalyticIntegrand("exponential", device), n_dimensions=1,
        n_live_points=128, num_delete=1, num_inner_steps=8)
    rng = np.random.default_rng(20261014)
    positions = rng.uniform(0.0, contour, (4096, 1))
    with jax.default_device(device):
        state = base.StateWithLogLikelihood(jnp.asarray(positions), jnp.zeros(4096, dtype=jnp.float64),
            jnp.asarray(-4.0 * positions[:, 0]), jnp.full(4096, -jnp.inf, dtype=jnp.float64))
        keys = jax.random.split(jax.random.PRNGKey(20261014), 4096)
        final, info = kernel.batched_chains(keys, state, jnp.asarray(-4.0 * contour))
    assert np.asarray(info.numerically_valid).all()
    assert np.asarray(info.slice_info.is_accepted).all()
    normalized = np.asarray(final.position)[:, 0] / contour
    assert np.all((normalized >= 0.0) & (normalized < 1.0))
    ks = kstest(normalized, "uniform")
    _retain(f"stationarity_{contour}", {
        "seed": 20261014, "contour": contour, "independent_chains": 4096, "inner_steps": 8,
        "uniform_ks_statistic": float(ks.statistic), "uniform_ks_pvalue": float(ks.pvalue),
        "mean": float(normalized.mean()), "variance": float(normalized.var(ddof=1)),
        "start_final_correlation": float(np.corrcoef(positions[:, 0] / contour, normalized)[0, 1]),
        "interpretation": "Non-rejection is not calibration; inspect independent uniform stationary starts only",
    }, record_property)


@pytest.fixture(scope="module")
def precision_kernel():
    device = models.jax.devices("cpu")[0]
    return NativeNestedKernel(AnalyticIntegrand("exponential", device), n_dimensions=1,
        n_live_points=1024, num_delete=1, num_inner_steps=8, batched=True)


@pytest.mark.parametrize("seed", [17, 29, 43, 3000, 3001, 3002])
def test_known_z_more_live_points_at_unchanged_accuracy_bar(precision_kernel, seed, record_property):
    result = precision_kernel.integrate(np.random.default_rng(seed).uniform(size=(1024, 1)),
        models.jax.random.PRNGKey(seed), evidence_tolerance=0.1,
        max_likelihood_calls=2_000_000, random_seed=seed)
    truth = math.log(-math.expm1(-4.0) / 4.0)
    _retain(f"precision_seed{seed}", {
        "seed": seed, "role": "development" if seed in (17, 29, 43) else "fresh-assessment",
        "live_points": result.n_live_points, "inner_steps": result.num_inner_steps,
        "logz": result.log_evidence, "truth_logz": truth, "delta_logz": result.log_evidence - truth,
        "reported_error_nats": result.log_evidence_error, "calls": result.n_likelihood_calls,
        "stop_reason": result.stop_reason, "claim_eligible": result.claim_eligible,
        "evidence_calibrated": result.evidence_calibrated,
        "scope": "More-information experiment; not a repair to or relaxation of the original 128-live regression",
    }, record_property)
    assert abs(result.log_evidence - truth) < 0.1
    assert result.n_live_points == 1024
    assert result.stop_reason == "evidence-tolerance" and not result.budget_exhausted
    assert result.base_run_delta_logz < 0.1
    assert result.claim_eligible is result.evidence_calibrated is False
