"""Causal development diagnostics of the retained fresh Gaussian C4 failure.

Seeds 6000/6001/6002 are now development data, never held-out validation.
Compare stopping, slice mixing and finite live population one axis at a time.
The ball lies inside the unit cube for radius <= 1/2: exact prior volume is
pi**2*r**4/2 and Gaussian integral is chi2.cdf((r/SD)**2, 4). These analytic
oracles are never provided to the sampler. Riemann bounds distinguish known
contour quadrature from assumed nested volumes. Stationary starts are drawn
independently with NumPy from a uniform four-ball, not from sampler output.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, kstest

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import NativeNestedKernel
from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand, _GAUSSIAN_TRUTH, _GAUSSIAN_SD


def _retain(name, record, record_property, tmp_path):
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = tmp_path
    payload = json.dumps(record, indent=2, allow_nan=False)
    (output / f"blackjax_gaussian_diagnosis_{name}.json").write_text(payload + "\n")
    record_property("gaussian_diagnosis", payload)


def _contour_diagnosis(result):
    """Independent exact ball geometry, not another nested sampler."""
    dead = np.concatenate([np.asarray(batch.dead_particles.position)
                           for batch in result.batches if batch.completed])
    radius = np.sqrt(np.sum((dead - 0.5)**2, axis=1))
    indices = np.flatnonzero(radius <= 0.5)
    assert indices.size > 1
    r = radius[indices]
    assert np.all(np.diff(r) <= 0.0)
    volume = math.pi**2 * r**4 / 2.0
    previous = np.r_[math.pi**2 * 0.5**4 / 2.0, volume[:-1]]
    norm = (_GAUSSIAN_SD * math.sqrt(2.0 * math.pi))**-4
    likelihood = norm * np.exp(-r**2 / (2.0 * _GAUSSIAN_SD**2))
    previous_likelihood = np.r_[norm * math.exp(-0.5**2 / (2.0 * _GAUSSIAN_SD**2)), likelihood[:-1]]
    upper = float(np.sum((previous - volume) * likelihood))
    lower = float(np.sum((previous - volume) * previous_likelihood))
    sphere_z = float(chi2.cdf((0.5 / _GAUSSIAN_SD)**2, 4))
    tail_z = float(chi2.cdf((r[-1] / _GAUSSIAN_SD)**2, 4))
    exact_dead = sphere_z - tail_z
    # Independent quadrature uses closed-form CDF geometry; floating-point
    # comparison allowance covers subtraction of near-unit probabilities.
    assert lower - 2e-14 <= exact_dead <= upper + 2e-14
    outside_sphere_z = math.exp(_GAUSSIAN_TRUTH) - sphere_z
    known_volume_logz = math.log(upper + tail_z + outside_sphere_z)
    assumed = np.exp(result.dead_log_volumes[indices])
    live = np.asarray(result.final_state.particles.position)
    live_r = np.sqrt(np.sum((live - 0.5)**2, axis=1))
    assert np.all(live_r < r[-1])
    normalized_live_volume = (live_r / r[-1])**4
    return {
        "spherical_dead_points": int(indices.size),
        "native_logz": result.log_evidence, "truth_logz": _GAUSSIAN_TRUTH,
        "delta_logz": result.log_evidence - _GAUSSIAN_TRUTH,
        "known_volume_upper_logz": known_volume_logz,
        "known_volume_lower_logz": math.log(lower + tail_z + outside_sphere_z),
        "known_volume_quadrature_error_nats": known_volume_logz - _GAUSSIAN_TRUTH,
        "native_minus_known_volume_upper_nats": result.log_evidence - known_volume_logz,
        "assumed_minus_exact_logvolume_at_sphere_entry": float(math.log(assumed[0] / volume[0])),
        "assumed_minus_exact_logvolume_at_final_death": float(math.log(assumed[-1] / volume[-1])),
        "final_exact_volume": float(volume[-1]), "final_assumed_volume": float(assumed[-1]),
        "exact_outside_sphere_evidence": outside_sphere_z,
        "final_live_volume_uniform_ks": float(kstest(normalized_live_volume, "uniform").statistic),
        "final_live_volume_mean": float(normalized_live_volume.mean()),
        "diagnostic_limit": "Live histories are dependent; known-volume oracle removes realized compression uncertainty, not a production estimator or proof of mixing",
    }


@pytest.mark.parametrize("seed", [6000, 6001, 6002])
def test_gaussian_stopping_mixing_and_live_count_decomposition(seed, tmp_path, record_property):
    jax = models.jax
    device = jax.devices("cpu")[0]
    rows = []
    with jax.default_device(device):
        # Explicit bounded development settings, not a new production default.
        for live, inner, tolerance in ((400, 12, 0.05), (400, 12, 0.001),
                                       (400, 48, 0.05), (1600, 12, 0.05)):
            kernel = NativeNestedKernel(GaussianIntegrand(device), n_dimensions=4,
                n_live_points=live, num_delete=live // 2, num_inner_steps=inner, batched=True)
            result = kernel.integrate(np.random.default_rng(seed).uniform(size=(live, 4)),
                jax.random.PRNGKey(seed), evidence_tolerance=tolerance,
                max_likelihood_calls=2_000_000, random_seed=seed)
            row = {"seed": seed, "live_points": live, "inner_steps": inner,
                "stop_tolerance_nats": tolerance, "stop_reason": result.stop_reason,
                "calls": result.n_likelihood_calls, "reported_error_nats": result.log_evidence_error,
                "actual_stop_delta_nats": result.base_run_delta_logz, **_contour_diagnosis(result)}
            rows.append(row)
            _retain(f"seed{seed}_live{live}_inner{inner}_tol{tolerance}", row, record_property, tmp_path)
            assert result.stop_reason == "evidence-tolerance"
            assert result.n_likelihood_calls <= 2_000_000
            assert result.claim_eligible is result.evidence_calibrated is False
        _retain(f"seed{seed}", {"role": "development; not independent calibration", "runs": rows},
                record_property, tmp_path)


@pytest.mark.parametrize("radius", [0.1, 0.3])
def test_gaussian_constrained_slice_stationarity_on_independent_ball_starts(radius, tmp_path, record_property):
    from blackjax.ns import base

    jax, jnp = models.jax, models.jnp
    device = jax.devices("cpu")[0]
    rng = np.random.default_rng(20261014)
    directions = rng.normal(size=(4096, 4))
    directions /= np.sqrt(np.sum(directions**2, axis=1))[:, None]
    points = 0.5 + radius * rng.uniform(size=4096)[:, None]**0.25 * directions
    norm_log = -4.0 * math.log(_GAUSSIAN_SD * math.sqrt(2.0 * math.pi))
    threshold = norm_log - radius**2 / (2.0 * _GAUSSIAN_SD**2)
    rows = []
    with jax.default_device(device):
        native = GaussianIntegrand(device)
        logl = jax.jit(jax.vmap(native.evaluate))(jnp.asarray(points)).log_integrand
        starts = base.StateWithLogLikelihood(jnp.asarray(points), jnp.zeros(4096, dtype=jnp.float64),
                                            logl, jnp.full(4096, -jnp.inf, dtype=jnp.float64))
        keys = jax.random.split(jax.random.PRNGKey(20261014), 4096)
        for inner in (12, 48):
            kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=400,
                num_delete=200, num_inner_steps=inner, batched=True)
            final, info = jax.block_until_ready(kernel.batched_chains(keys, starts, jnp.asarray(threshold)))
            assert np.asarray(info.numerically_valid).all() and np.asarray(info.slice_info.is_accepted).all()
            final_points = np.asarray(final.position)
            start_volume = np.sum((points - 0.5)**2, axis=1)**2 / radius**4
            end_volume = np.sum((final_points - 0.5)**2, axis=1)**2 / radius**4
            assert np.all((end_volume >= 0.0) & (end_volume < 1.0))
            ks = kstest(end_volume, "uniform")
            row = {"radius": radius, "independent_chains": 4096, "inner_steps": inner,
                "volume_uniform_ks_statistic": float(ks.statistic), "volume_uniform_ks_pvalue": float(ks.pvalue),
                "volume_mean": float(end_volume.mean()), "volume_variance": float(end_volume.var(ddof=1)),
                "start_final_volume_correlation": float(np.corrcoef(start_volume, end_volume)[0, 1]),
                "coordinate_mean": final_points.mean(axis=0).tolist(),
                "coordinate_variance": final_points.var(axis=0, ddof=1).tolist(),
                "uniform_ball_coordinate_variance_truth": radius**2 / 6.0,
                "scope": "Independent stationary-start mixing development diagnostic, not calibrated evidence"}
            rows.append(row)
            _retain(f"stationarity_radius{radius}_inner{inner}", row, record_property, tmp_path)
        _retain(f"stationarity_radius{radius}", {"runs": rows}, record_property, tmp_path)
