"""Pre-registered CPU Gaussian controls; no scene or MC-budget calibration.

The acceptance cohorts below run the calibrated 3,200-live configuration with
its derived call ceiling. The 400-live setting is the historical failing
control (33/50 within 0.1 nat on the reassessment seeds) and is kept only as a
descriptive subset, never as an acceptance bar; the 2026-11-10 record covers
both configurations.
"""

import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import beta, binom

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence
from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling
from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand, _GAUSSIAN_TRUTH

pytestmark = pytest.mark.skipif(os.environ.get("EXONYM_RUN_SLOW_VALIDATIONS") != "1",
                                reason="pre-registered numerical assessment")

_CALIBRATED_CEILING = blackjax_call_ceiling(
    BLACKJAX_CALIBRATED_LIVE_POINTS, BLACKJAX_CALIBRATED_LIVE_POINTS // 2, 12)


def _call_ceiling(live):
    # The calibrated derived ceiling rounds to zero at the 400-live deletion
    # geometry (200 deleted points reserve less than one whole million), so the
    # historical control keeps its explicit two-million-call ceiling.
    return 2_000_000 if live != BLACKJAX_CALIBRATED_LIVE_POINTS else _CALIBRATED_CEILING


def _trial(seed, live):
    jax = models.jax
    device = jax.devices("cpu")[0]
    integrand = SimpleNamespace(population=None, target_feh=None,
                                jax_unit_integrand=lambda *, device: GaussianIntegrand(device))
    controls = nested_evidence._SamplerControls(n_live_points=live, evidence_tolerance=0.05,
        max_likelihood_calls=_call_ceiling(live), random_seed=seed, n_jobs=1)
    start = time.monotonic()
    row = {"seed": seed, "live": live}
    try:
        with jax.default_device(device):
            result = nested_evidence._run_native_nested_sampler(integrand, controls, 4)
        row.update(logz=result.log_evidence, error=result.log_evidence_error,
                   stop=result.stop_reason, calls=result.n_likelihood_calls,
                   sampler=result.sampler, deletion=result.num_delete, steps=result.num_inner_steps,
                   execution=result.slice_execution,
                   calibrated=result.configuration_calibrated)
        row["valid"] = (math.isfinite(row["logz"]) and math.isfinite(row["error"])
            and row["error"] > 0 and row["sampler"] == "blackjax-nss-experimental"
            and row["deletion"] == live // 2 and row["steps"] == 12
            and row["calls"] <= controls.max_likelihood_calls)
        if not math.isfinite(row["logz"]):
            row["logz"] = None
        if not math.isfinite(row["error"]):
            row["error"] = None
    except Exception:
        # Retain the original exception and count the failed seed; do not replace it.
        row.update(valid=False, traceback=traceback.format_exc())
    row["seconds"] = time.monotonic() - start
    return row


def _summarize(rows):
    complete = [r for r in rows if r["valid"]]
    delta = np.array([r["logz"] - _GAUSSIAN_TRUTH for r in complete])
    errors = np.array([r["error"] for r in complete])
    z = delta / errors
    n, successes = len(rows), int(np.count_nonzero(abs(delta) < 0.1))
    interval = [float(beta.ppf(0.025, successes, n - successes + 1)) if successes else 0.0,
                float(beta.ppf(0.975, successes + 1, n - successes)) if successes < n else 1.0]
    return {"n": n, "complete": len(complete), "truth": _GAUSSIAN_TRUTH,
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else None,
        "mean_error": float(errors.mean()) if len(errors) else None,
        "mean_z": float(z.mean()) if len(z) else None,
        "sd_z": float(z.std(ddof=1)) if len(z) > 1 else None,
        "z_quantiles": np.quantile(z, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(z) else [],
        "within_1_sigma": int(np.count_nonzero(abs(z) < 1)),
        "within_2_sigma": int(np.count_nonzero(abs(z) < 2)),
        "within_0p1_nat": successes, "success_fraction_95pct_interval": interval,
        "mean_calls": float(np.mean([r["calls"] for r in complete])) if complete else None,
        "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
        "all_stops_valid": len(complete) == n and all(
            r["stop"] in {"evidence-tolerance", "likelihood-plateau"} for r in complete),
        "all_calibrated": len(complete) == n and all(r["calibrated"] for r in complete)}


def _run(live, seeds, record_property, name):
    rows = []
    receipt_dir = os.environ.get("TREX_BLACKJAX_RECEIPT_DIR")
    output = Path(receipt_dir) if receipt_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        row = _trial(seed, live)
        rows.append(row)
        if output is not None:
            # Preserve completed and failed outcomes even if a later seed fails.
            with (output / f"blackjax_{name}_live{live}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = _summarize(rows)
    record_property("blackjax_reassessment", json.dumps({"summary": summary, "runs": rows}, allow_nan=False))
    return summary


@pytest.mark.parametrize("live", [400, BLACKJAX_CALIBRATED_LIVE_POINTS])
def test_known_z_local_subset(live, record_property):
    """Ten independent seeds: descriptive precision and cost, not C1-C4 acceptance.

    The 400-live arm is the historical uncalibrated control: it is recorded for
    scatter/cost comparison and must not be labelled calibrated.
    """
    summary = _run(live, range(421109000, 421109010), record_property, "local_subset")
    assert summary["all_stops_valid"], summary
    assert summary["all_calibrated"] is (live == BLACKJAX_CALIBRATED_LIVE_POINTS), summary


def _assert_fifty(record):
    assert record["complete"] == 50, record
    assert abs(record["mean_delta"]) <= 3 * record["sd_delta"] / math.sqrt(50), record
    assert binom.ppf(0.005, 50, 0.6827) <= record["within_1_sigma"] <= binom.ppf(0.995, 50, 0.6827), record
    assert record["within_2_sigma"] >= binom.ppf(0.005, 50, 0.9545), record
    assert record["within_0p1_nat"] >= 48 and record["all_stops_valid"], record
    assert record["all_calibrated"], record


def test_known_z_local_fresh_cohort(record_property):
    """An independently pre-registered 50-seed local assessment at the calibrated configuration."""
    _assert_fifty(_run(BLACKJAX_CALIBRATED_LIVE_POINTS, range(421120000, 421120050),
                       record_property, "local_fresh_cohort"))


def test_known_z_fresh_cohort(record_property):
    """Apply the original registered 50-run bias, coverage and precision bars at the calibrated configuration."""
    _assert_fifty(_run(BLACKJAX_CALIBRATED_LIVE_POINTS, range(421110000, 421110050),
                       record_property, "fresh_cohort"))
