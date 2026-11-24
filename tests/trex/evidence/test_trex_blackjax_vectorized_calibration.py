"""Fresh known-evidence assessment of the vectorized production NSS route.

The calibrated configuration is 3,200 live points with the derived call ceiling
(4,000,000 at 4-D, 12 inner steps), deletion=1600, 0.05-nat stopping and the
unchanged 0.1-nat/48-of-50 C4 bar. It passed at 48-49/50 in four independent
50-seed cohorts (2026-11-10).

Development evidence: the historical 400-live control (explicit 2,000,000-call
ceiling) failed C4 at 29/50 on seeds 2000-2049 and 6000-6049, and an earlier
separately retained 400-live cohort reached 36/50; the reassessment cohort was
33/50. Its sd(delta) is 0.106-0.111 nat, at or above the 0.1-nat bar, and its
~0.114-nat reported error matches that scatter, so the failure is finite-live
scatter rather than a wrong error estimate. The 3,200-live sd(delta) is
0.040-0.047 nat. The 400-live setting is the historical failing control, not a
production configuration; test_trex_blackjax_calibration keeps it as an
explicit non-production regression.
The 1e-10 per-point arithmetic gate does not certify evidence precision.
"""

import math
import os
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_types import NativeIntegrationError
from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling
from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand, _assess, _GAUSSIAN_TRUTH, _summary


pytestmark = pytest.mark.skipif(os.environ.get("EXONYM_RUN_SLOW_VALIDATIONS") != "1",
                                reason="fresh calibration requires approved slow-validation compute")

_CALIBRATED_CEILING = blackjax_call_ceiling(
    BLACKJAX_CALIBRATED_LIVE_POINTS, BLACKJAX_CALIBRATED_LIVE_POINTS // 2, 12)


def _assess_production_route(*, first_seed, live_points, call_ceiling, experiment):
    """Shared real caller and unchanged C1-C4, not a sampler substitute."""
    device = models.jax.devices("cpu")[0]
    native = GaussianIntegrand(device)
    integrand = SimpleNamespace(population=None, target_feh=None,
                               jax_unit_integrand=lambda *, device: native)
    results = []
    receipt_dir = os.environ.get("TREX_BLACKJAX_RECEIPT_DIR")
    output = Path(receipt_dir) if receipt_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for seed in range(first_seed, first_seed + 50):
        controls = nested_evidence._SamplerControls(n_live_points=live_points, evidence_tolerance=0.05,
            max_likelihood_calls=call_ceiling, random_seed=seed, n_jobs=1)
        try:
            with models.jax.default_device(device):
                result = nested_evidence._run_native_nested_sampler(integrand, controls, 4)
        except NativeIntegrationError as error:
            if output is not None:
                partial = error.result
                with (output / f"blackjax_{experiment}_seed_outcomes.jsonl").open("a") as stream:
                    stream.write(json.dumps({"random_seed": seed, "status": "failed",
                        "stop_reason": partial.stop_reason, "failure_reason": partial.failure_reason,
                        "n_likelihood_calls": partial.n_likelihood_calls,
                        "likelihood_calls_exact": partial.likelihood_calls_exact}) + "\n")
            raise
        # Preserve completed seeds even if a later trial fails or is interrupted;
        # do not count an incomplete cohort as a 50-trial calibration result.
        summary = _summary(result)
        if output is not None:
            with (output / f"blackjax_{experiment}_seed_outcomes.jsonl").open("a") as stream:
                stream.write(json.dumps(vars(summary)) + "\n")
        assert result.sampler == "blackjax-nss-experimental"
        assert result.slice_execution == "vmapped-chains"
        assert result.n_live_points == live_points and result.num_delete == live_points // 2
        assert result.num_inner_steps == 12
        assert result.n_likelihood_calls <= call_ceiling
        assert result.claim_eligible is result.evidence_calibrated is False
        assert result.configuration_calibrated is (live_points == BLACKJAX_CALIBRATED_LIVE_POINTS)
        assert np.isfinite(result.log_evidence) and np.isfinite(result.log_evidence_error)
        results.append(summary)
    # Reuse the unchanged registered C1/C2/C3 oracle; retain every seed outcome
    # before the C4 assertion, even when the unchanged precision bar fails.
    record = _assess(experiment, results, _GAUSSIAN_TRUTH)
    assert record["within_0p1_nat"] >= math.ceil(0.95 * record["n"]), record
    assert set(record["stop_reasons"]) <= {"evidence-tolerance", "likelihood-plateau"}, record


def test_fresh_vectorized_production_known_evidence_calibration():
    # Fresh seeds 6000-6049 at the calibrated configuration. The same seeds
    # failed the 400-live control (29/50); that configuration is retained as a
    # non-production regression in test_trex_blackjax_calibration, not here.
    _assess_production_route(first_seed=6000, live_points=BLACKJAX_CALIBRATED_LIVE_POINTS,
        call_ceiling=_CALIBRATED_CEILING,
        experiment="E3_fresh_vectorized_gaussian4d_3200live_seeds6000_6049")


def test_higher_information_budget_on_fresh_held_out_gaussian_seeds():
    # New independent assessment, not a threshold/seed repair of the old red.
    _assess_production_route(first_seed=7000, live_points=BLACKJAX_CALIBRATED_LIVE_POINTS,
        call_ceiling=_CALIBRATED_CEILING,
        experiment="E3_precision3200_gaussian4d_seeds7000_7049")


def test_strict_parent_repair_on_fresh_held_out_gaussian_seeds():
    # Predeclared after the weighted-CDF failure, before these seeds execute.
    _assess_production_route(first_seed=8000, live_points=BLACKJAX_CALIBRATED_LIVE_POINTS,
        call_ceiling=_CALIBRATED_CEILING,
        experiment="E3_strict_parent3200_gaussian4d_seeds8000_8049")
