"""Addendum-registered 1-D exponential controls at the calibrated 3200-live route.

Registered in docs/preregistration/trex-blackjax-known-z-20261110.md before
these seeds ran. The original E1/E2 controls used 128 live points; this keeps
their deletion fractions, slice steps, tolerance and C1-C3 criteria and changes
only the live count to the new BlackJAX default.

Only E2 (deletion 4) runs, as the 2026-11-10 deviation in that record states: one
3200-live deletion-1 seed did not finish in 15 min, so 100 seeds exceed any session,
and the production route never deletes a single live point. E2 took 12,126 s.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import NativeNestedKernel
from tests.trex.evidence.test_trex_blackjax_calibration import _EXPONENTIAL_TRUTH, _assess
from tests.trex.evidence.test_blackjax_evidence import AnalyticIntegrand

pytestmark = pytest.mark.skipif(
    os.environ.get("EXONYM_RUN_SLOW_VALIDATIONS") != "1",
    reason="registered numerical assessment",
)

_LIVE_POINTS = 3200
# One full slice sweep of the live population with the actual 8 steps.
_CEILING = _LIVE_POINTS * 8 * 111
_SEEDS = range(421130000, 421130100)


def _run(seed, kernel):
    positions = np.random.default_rng(seed).uniform(size=(_LIVE_POINTS, 1))
    result = kernel.integrate(
        positions,
        models.jax.random.PRNGKey(seed),
        evidence_tolerance=0.1,
        max_likelihood_calls=_CEILING,
        random_seed=seed,
    )
    return SimpleNamespace(
        log_evidence=result.log_evidence,
        log_evidence_error=result.log_evidence_error,
        stop_reason=result.stop_reason,
        random_seed=result.random_seed,
        n_likelihood_calls=result.n_likelihood_calls,
    )


@pytest.mark.parametrize("deletion", [4])
def test_exponential_known_z_at_the_calibrated_live_count(deletion, record_property):
    kernel = NativeNestedKernel(
        AnalyticIntegrand("exponential", models.jax.devices("cpu")[0]),
        n_dimensions=1,
        n_live_points=_LIVE_POINTS,
        num_delete=deletion,
        num_inner_steps=8,
    )
    receipt_dir = Path(os.environ.get(
        "TREX_BLACKJAX_RECEIPT_DIR",
        "/kaggle/working" if Path("/kaggle/working").is_dir() else "scratch/current-lanes",
    ))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    output = receipt_dir / "blackjax_exponential_live3200_del{0}.jsonl".format(deletion)
    results = []
    with output.open("w", encoding="utf-8") as stream:
        for seed in _SEEDS:
            row = _run(seed, kernel)
            results.append(row)
            stream.write(json.dumps(vars(row), allow_nan=False) + "\n")
            stream.flush()
    record = _assess(
        "E{0}_exponential_deletion{1}_live3200".format(deletion // 4 + 1, deletion),
        results,
        _EXPONENTIAL_TRUTH,
    )
    record_property(
        "blackjax_exponential",
        json.dumps({"deletion": deletion, "summary": record, "runs": [vars(row) for row in results]}, allow_nan=False),
    )
    assert set(record["stop_reasons"]) <= {"evidence-tolerance", "likelihood-plateau"}, record
    assert all(run.n_likelihood_calls < _CEILING for run in results), record
