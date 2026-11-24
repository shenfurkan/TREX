"""Development-only Gaussian diagnosis with independent constrained draws.

The exact-draw arm substitutes only the transition in the real evidence driver;
its synthetic slice counters are not native performance measurements. The
previously observed 2000..2009 seeds are development data, never acceptance.
"""

import json
import math
from types import SimpleNamespace

import numpy as np

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import NativeNestedKernel
from trex.evidence._blackjax_types import SLICE_CALL_BOUND
from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand, _GAUSSIAN_SD, _GAUSSIAN_TRUTH
from tests.trex.evidence.test_trex_blackjax_gaussian_diagnosis import _contour_diagnosis


def _exact_chains(seed, calls):
    """IID cube-intersection or four-ball draws, not a native slice transition."""
    from blackjax.ns.base import StateWithLogLikelihood

    jnp = models.jnp
    rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
    normalizer = -4.0 * math.log(_GAUSSIAN_SD * math.sqrt(2.0 * math.pi))

    def chains(keys, starts, threshold):
        count = len(keys)
        radius = math.sqrt(2.0 * _GAUSSIAN_SD**2 * (normalizer - float(threshold)))
        if radius <= 0.5:
            directions = rng.normal(size=(count, 4))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            points = 0.5 + radius * rng.random(count)[:, None] ** 0.25 * directions
            calls[0] += count
        else:
            # Prior rejection is exact even when the sphere intersects the cube.
            accepted = []
            remaining = count
            while remaining:
                candidates = rng.uniform(size=(remaining * 4, 4))
                calls[0] += len(candidates)
                selected = candidates[np.sum((candidates - 0.5) ** 2, axis=1) < radius**2]
                accepted.append(selected[:remaining])
                remaining -= len(selected[:remaining])
            points = np.concatenate(accepted)
        logl = normalizer - np.sum((points - 0.5) ** 2, axis=1) / (2.0 * _GAUSSIAN_SD**2)
        assert np.all(logl > float(threshold))
        result = StateWithLogLikelihood(jnp.asarray(points), starts.logdensity, jnp.asarray(logl),
                                        starts.loglikelihood_birth)
        # The driver's bookkeeping requires one valid entry per synthetic step.
        # These are explicitly NOT counts of the exact rejection work above.
        shape = (count, 12)
        info = SimpleNamespace(numerically_valid=np.ones(shape, dtype=bool),
            budget_exhausted=np.zeros(shape, dtype=bool),
            slice_info=SimpleNamespace(is_accepted=np.ones(shape, dtype=bool)),
            likelihood_calls=np.broadcast_to(np.array([0, 0, 1]), (*shape, 3)),
            remaining_budget=np.full(shape, SLICE_CALL_BOUND - 1))
        return result, info

    return chains


def _row(result, mode, seed, exact_calls):
    batch_radii = np.asarray([
        np.linalg.norm(np.asarray(batch.dead_particles.position)[-1] - 0.5)
        for batch in result.batches if batch.completed
    ])
    volumes = (math.pi**2 / 2.0) * batch_radii**4
    valid = (batch_radii[1:] <= 0.5) & (batch_radii[:-1] <= 0.5)
    observed_steps = np.log(volumes[1:][valid] / volumes[:-1][valid])
    counts = np.arange(201, 401, dtype=np.float64)
    expected_step = -np.sum(1.0 / counts)
    expected_sd = math.sqrt(np.sum(1.0 / counts**2))
    return {"mode": mode, "seed": seed, "logz": result.log_evidence,
        "delta": result.log_evidence - _GAUSSIAN_TRUTH, "error": result.log_evidence_error,
        "z": (result.log_evidence - _GAUSSIAN_TRUTH) / result.log_evidence_error,
        "stop": result.stop_reason, "driver_calls": result.n_likelihood_calls,
        "exact_proposals": exact_calls, "compression_z": ((observed_steps - expected_step) / expected_sd).tolist(),
        **_contour_diagnosis(result)}


def test_exact_constrained_draws_separate_mixing_from_compression(record_property):
    """Compare matching seeds/settings; report all outcomes, not a calibration gate."""
    jax = models.jax
    device = jax.devices("cpu")[0]
    kernel = NativeNestedKernel(GaussianIntegrand(device), n_dimensions=4, n_live_points=400,
                                num_delete=200, num_inner_steps=12, batched=True)
    native_chains = kernel.batched_chains
    rows = []
    for seed in range(2000, 2010):
        for mode in ("slice", "exact"):
            exact_calls = [0]
            kernel.batched_chains = native_chains if mode == "slice" else _exact_chains(seed, exact_calls)
            result = kernel.integrate(np.random.default_rng(seed).uniform(size=(400, 4)),
                jax.random.PRNGKey(seed), evidence_tolerance=0.05,
                max_likelihood_calls=2_000_000, random_seed=seed)
            assert result.stop_reason == "evidence-tolerance"
            rows.append(_row(result, mode, seed, exact_calls[0] if mode == "exact" else None))
    record_property("paired_gaussian_diagnosis", json.dumps(rows, allow_nan=False))


def test_exact_constrained_draws_on_historical_cohort(record_property):
    """All 50 already-seen R9 seeds; exact draws are a development control."""
    jax = models.jax
    device = jax.devices("cpu")[0]
    kernel = NativeNestedKernel(GaussianIntegrand(device), n_dimensions=4, n_live_points=400,
                                num_delete=200, num_inner_steps=12, batched=True)
    rows = []
    for seed in range(2000, 2050):
        exact_calls = [0]
        kernel.batched_chains = _exact_chains(seed, exact_calls)
        result = kernel.integrate(np.random.default_rng(seed).uniform(size=(400, 4)),
            jax.random.PRNGKey(seed), evidence_tolerance=0.05,
            max_likelihood_calls=2_000_000, random_seed=seed)
        assert result.stop_reason == "evidence-tolerance"
        rows.append(_row(result, "exact", seed, exact_calls[0]))
    record_property("exact_historical_gaussian_diagnosis", json.dumps(rows, allow_nan=False))
