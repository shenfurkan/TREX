"""Held-out known-Z coverage of the experimental BlackJAX NSS evidence driver.

Pre-registered in scratch/trex_gpu_migration_2026-10-13/overnight_2026-10-14/
CALIBRATION_PREREGISTRATION.md before any of these seeds ran. Synthetic controls
only: they test bias and reported-error coverage, not TREX scene calibration.

The calibrated configuration is 3,200 live points with the derived call ceiling
(4,000,000 at 4-D and 12 inner steps), the registered 0.05-nat stopping
tolerance and the unchanged 0.1-nat/48-of-50 C4 bar; it passed at 48-49/50 in
four independent 50-seed cohorts (docs/decisions.md, 2026-11-10). The historical
400-live control is *not* the calibrated configuration: it failed C4 at 29/50
on seeds 2000-2049, 29/50 on seeds 6000-6049 and 33/50 on the reassessment
cohort 421110000-421110049. There sd(delta) was 0.106-0.111 nat, at or above the
0.1-nat bar, and the ~0.114-nat reported error matched that scatter: the failure
is finite-live scatter, not a wrong error estimate. The 3,200-live sd(delta) is
0.040-0.047 nat against a ~0.040-nat reported error. The 400-live setting is
retained below as a non-production configuration regression, never as an
acceptance bar.
"""

import json
import math
import os
import resource
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import beta, binom

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_kernel import NativeNestedKernel
from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling
from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation
from tests.trex.evidence.test_blackjax_evidence import AnalyticIntegrand

pytestmark = pytest.mark.skipif(os.environ.get("EXONYM_RUN_SLOW_VALIDATIONS") != "1",
                                reason="held-out calibration runs only in approved slow-validation sessions")

_EXPONENTIAL_TRUTH = math.log(-math.expm1(-4.0) / 4.0)
_GAUSSIAN_SD = 0.1
_GAUSSIAN_DIMENSIONS = 4
_GAUSSIAN_TRUTH = _GAUSSIAN_DIMENSIONS * math.log(math.erf(0.5 / (_GAUSSIAN_SD * math.sqrt(2.0))))


class GaussianIntegrand:
    """Normalized isotropic Gaussian on the unit cube; its integral is known in closed form."""

    def __init__(self, device):
        self.device = device

    def evaluate(self, point):
        jnp = models.jnp
        inside = jnp.all((point >= 0.0) & (point < 1.0))
        value = (-0.5 * jnp.sum((point - 0.5) ** 2) / _GAUSSIAN_SD**2
                 - _GAUSSIAN_DIMENSIONS * math.log(_GAUSSIAN_SD * math.sqrt(2.0 * math.pi)))
        return UnitIntegrandEvaluation(jnp.where(inside, value, -jnp.inf), jnp.all(jnp.isfinite(point)))


def _cpu():
    return models.jax.devices("cpu")[0]


def _summary(result):
    # Keep only the scalars: each native result retains its full audit history,
    # and holding 50 production-route histories exhausted 32 GB (cal a1, 2026-10-14).
    return SimpleNamespace(log_evidence=result.log_evidence, log_evidence_error=result.log_evidence_error,
                           stop_reason=result.stop_reason, n_likelihood_calls=result.n_likelihood_calls,
                           random_seed=result.random_seed, n_live_points=result.n_live_points,
                           max_likelihood_calls=result.max_likelihood_calls,
                           configuration_calibrated=result.configuration_calibrated)


def _exponential_kernel(deletion):
    return NativeNestedKernel(AnalyticIntegrand("exponential", _cpu()), n_dimensions=1,
                              n_live_points=128, num_delete=deletion, num_inner_steps=8)


@pytest.fixture(scope="module", params=[1, 4])
def exponential_kernel(request):
    # Share compiled functions only; integrate initializes fresh particles and RNG state.
    return _exponential_kernel(request.param)


def _exponential_run(seed, kernel):
    positions = np.random.default_rng(seed).uniform(size=(128, 1))
    return _summary(kernel.integrate(positions, models.jax.random.PRNGKey(seed), evidence_tolerance=0.1,
                                     max_likelihood_calls=100_000, random_seed=seed))


def _production_route_run(seed, live_points=BLACKJAX_CALIBRATED_LIVE_POINTS,
                          max_likelihood_calls=None):
    """Unpatched native route at the calibrated 3,200-live configuration."""
    if max_likelihood_calls is None:
        max_likelihood_calls = blackjax_call_ceiling(
            live_points, live_points // 2, 3 * _GAUSSIAN_DIMENSIONS)
    integrand = SimpleNamespace(population=None, target_feh=None,
                                jax_unit_integrand=lambda *, device: GaussianIntegrand(device))
    controls = nested_evidence._SamplerControls(n_live_points=live_points, evidence_tolerance=0.05,
                                                max_likelihood_calls=max_likelihood_calls,
                                                random_seed=seed, n_jobs=1)
    with models.jax.default_device(_cpu()):
        return _summary(nested_evidence._run_native_nested_sampler(integrand, controls, _GAUSSIAN_DIMENSIONS))


def _assess(name, results, truth):
    delta = np.array([r.log_evidence - truth for r in results])
    error = np.array([r.log_evidence_error for r in results])
    z = delta / error
    n = len(results)
    precision_successes = int((abs(delta) < 0.1).sum())
    # Exact two-sided 95% binomial interval reports trial-count uncertainty;
    # it does not replace or loosen the registered empirical C4 count gate.
    precision_interval = [
        float(beta.ppf(0.025, precision_successes, n - precision_successes + 1))
            if precision_successes else 0.0,
        float(beta.ppf(0.975, precision_successes + 1, n - precision_successes))
            if precision_successes < n else 1.0,
    ]
    record = {
        "experiment": name, "truth": truth, "n": n,
        "mean_delta": float(delta.mean()), "sd_delta": float(delta.std(ddof=1)),
        "mean_reported_error": float(error.mean()),
        "within_1_sigma": int((abs(z) < 1).sum()), "within_2_sigma": int((abs(z) < 2).sum()),
        "within_0p1_nat": precision_successes,
        "within_0p1_nat_fraction_95pct_exact_binomial_interval": precision_interval,
        "stop_reasons": sorted({r.stop_reason for r in results}),
        "mean_likelihood_calls": float(np.mean([r.n_likelihood_calls for r in results])),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runs": [{"seed": r.random_seed, "log_evidence": r.log_evidence, "error": r.log_evidence_error,
                  "stop_reason": r.stop_reason, "calls": r.n_likelihood_calls} for r in results],
    }
    receipt_dir = Path("/kaggle/working")
    if receipt_dir.is_dir():
        (receipt_dir / f"blackjax_calibration_{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "runs"}))
    # Pre-registered criteria C1-C3.
    assert abs(record["mean_delta"]) <= 3.0 * record["sd_delta"] / math.sqrt(n), record
    assert binom.ppf(0.005, n, 0.6827) <= record["within_1_sigma"] <= binom.ppf(0.995, n, 0.6827), record
    assert record["within_2_sigma"] >= binom.ppf(0.005, n, 0.9545), record
    return record


def test_exponential_known_z_coverage_on_held_out_seeds(exponential_kernel):
    deletion = exponential_kernel.num_delete
    _assess(f"E{deletion // 4 + 1}_exponential_deletion{deletion}",
            [_exponential_run(seed, exponential_kernel) for seed in range(1000, 1100)], _EXPONENTIAL_TRUTH)


def test_exponential_kernel_reuse_keeps_seed_results_independent(exponential_kernel):
    expected = _exponential_run(1729, _exponential_kernel(exponential_kernel.num_delete))
    _exponential_run(1730, exponential_kernel)
    assert vars(_exponential_run(1729, exponential_kernel)) == vars(expected)


def test_production_route_gaussian_known_z_meets_the_evidence_bar():
    """Registered C4 at the calibrated 3,200-live configuration (seeds 2000-2049).

    The archived 400-live failure of these same seeds (29/50) is the historical
    uncalibrated control, not this acceptance bar.
    """
    results = [_production_route_run(seed) for seed in range(2000, 2050)]
    record = _assess("E3_gaussian4d_production_route_3200live", results, _GAUSSIAN_TRUTH)
    assert all(result.configuration_calibrated for result in results), record
    # Pre-registered C4: the 0.1-nat bar on >= 95% of runs and no truncated run.
    assert record["within_0p1_nat"] >= math.ceil(0.95 * record["n"]), record
    assert set(record["stop_reasons"]) <= {"evidence-tolerance", "likelihood-plateau"}, record


def test_four_hundred_live_points_is_not_the_calibrated_configuration():
    """The historical 400-live control stays an explicit non-production regression.

    It needs its explicit 2,000,000-call ceiling because the calibrated derived
    ceiling rounds to zero at this deletion geometry (200 deleted points x 12
    steps x 111 calls reserves less than one whole million). Its sd(delta) of
    0.106-0.111 nat sits at or above the 0.1-nat bar while its ~0.114-nat
    reported error matches that scatter, so this live count cannot meet the bar
    and no assertion here asks it to.
    """
    result = _production_route_run(9000, live_points=400, max_likelihood_calls=2_000_000)
    assert result.n_live_points == 400 and result.max_likelihood_calls == 2_000_000
    assert result.configuration_calibrated is False
    assert result.stop_reason == "evidence-tolerance"
