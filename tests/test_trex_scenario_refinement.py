"""Per-scenario refinement state in the run report; synthetic inputs only."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trex.app.engine import scenario_evidence_refinement
from trex.app.target import target
from trex.evidence.diagnostics import generate_diagnostics


def _result():
    lnZ = np.full(15, -100.0)
    lnZ[[0, 1, 2]] = -1.0, -5.0, -6.0
    fits = {
        0: {"scenario": "TP", "diagnostic_key": "TP", "mc_ln_evidence": -1.0,
            "mc_effective_sample_size": 5000.0, "n_draws": 1_000_000,
            # Synthetic phase split: this fixture checks transport, not a measurement.
            "phase_seconds": {"draw": 1.5, "wait": 0.25, "evaluate": 10.0},
            "transiting_draws": 12, "finite_likelihood_draws": 900,
            "proposal_support": {"state": "usable", "draws_considered": 1_000_000}},
        1: {"scenario": "EB", "diagnostic_key": "EB", "mc_ln_evidence": -150.0,
            "mc_effective_sample_size": 2.0, "n_draws": 1_000_000},
        2: {"scenario": "EBx2P", "diagnostic_key": "EBx2P", "mc_ln_evidence": -6.0,
            "mc_effective_sample_size": 3.0, "n_draws": 1_000_000},
    }
    result = generate_diagnostics(
        lnZ, effective_sample_sizes={0: 5000.0, 1: 500.0, 2: 1500.0}, scenario_fits=fits,
    )
    result.supplemental_nested_evidence = {
        "EB": {"scenario": "EB", "estimator": "mc-pilot-defensive-importance",
               "log_evidence": -5.0, "effective_sample_size": 500.0},
        "EBx2P": {"scenario": "EBx2P", "estimator": "nested-sampling", "acceptance": "importance-refined",
                   "log_evidence": -6.0, "importance_sampling": {"effective_sample_size": 1500.0}},
    }
    result.nested_evidence_escalation = {"mc_importance": {"attempts": {
        "EB": {"status": "accepted", "effective_sample_size": 500.0, "log_evidence": -5.0},
    }}}
    return result


def test_refinement_reports_both_ess_values_and_a_status_per_scenario():
    rows = {row["scenario"]: row for row in scenario_evidence_refinement(_result())}
    # An unrefined scenario keeps its Monte Carlo ESS and is labelled, not blank.
    assert rows["TP"]["mc_effective_sample_size"] == 5000.0
    assert rows["TP"]["refined_effective_sample_size"] is None
    assert rows["TP"]["refinement_status"] == "not-refined"
    assert "floor" in rows["TP"]["refinement_reason"]
    # The MC-pilot stage replaces the sub-floor Monte Carlo entry (2 draws -> 500).
    assert rows["EB"]["mc_effective_sample_size"] == 2.0
    assert rows["EB"]["refined_effective_sample_size"] == 500.0
    assert rows["EB"]["refinement_status"] == "refined"
    assert rows["EB"]["refinement_estimator"] == "mc-pilot-defensive-importance"
    # The nested router's substitution is reported with the same two numbers.
    assert rows["EBx2P"]["refined_effective_sample_size"] == 1500.0
    assert rows["EBx2P"]["refinement_estimator"] == "nested-sampling"
    assert all(row["refinement_status"] == "refined" for row in rows.values() if row["scenario"] in ("EB", "EBx2P"))


def test_a_stage_that_kept_the_monte_carlo_entry_says_why():
    result = _result()
    result.supplemental_nested_evidence = {}
    result.nested_evidence_escalation = {"mc_importance": {"attempts": {
        "EB": {"status": "unresolved", "reason": "MC-pilot importance effective sample size is below the registered floor"},
        "EBx2P": {"status": "unavailable", "reason": "no finite MC pilot centres"},
    }}}
    rows = {row["scenario"]: row for row in scenario_evidence_refinement(result)}
    assert rows["EB"]["refinement_status"] == "unresolved"
    assert "below the registered floor" in rows["EB"]["refinement_reason"]
    assert rows["EBx2P"]["refinement_status"] == "unavailable"
    assert rows["EB"]["refined_effective_sample_size"] is None


def test_an_ensemble_substitution_reports_no_kish_ess_but_is_refined():
    result = _result()
    result.supplemental_nested_evidence["PTP"] = {
        "scenario": "PTP", "estimator": "nested-sampling", "acceptance": "ensemble", "log_evidence": -3.0,
    }
    rows = {row["scenario"]: row for row in scenario_evidence_refinement(result)}
    assert rows["PTP"]["refinement_status"] == "refined"
    assert rows["PTP"]["refined_effective_sample_size"] is None
    assert "Kish" in rows["PTP"]["refinement_reason"]


def test_run_report_writes_the_per_scenario_refinement_table(tmp_path):
    result = _result()
    namespace = SimpleNamespace(
        failure_record=None, probs=pd.DataFrame({"scenario": [], "probability": []}),
        FPP=result.fpp, NFPP=result.nfpp, trex_result=result,
        calculation_provenance={"limb_darkening": []},
    )
    target.write_outputs(namespace, tmp_path / "result")
    payload = json.loads((tmp_path / "result" / "summary.json").read_text())
    rows = {row["scenario"]: row for row in payload["scenario_evidence_refinement"]}
    assert rows["EB"]["mc_effective_sample_size"] == 2.0
    assert rows["EB"]["refined_effective_sample_size"] == 500.0
    assert rows["EB"]["refinement_status"] == "refined"
    assert rows["TP"]["refinement_status"] == "not-refined"
    # The existing first-stage table still carries the same Monte Carlo entry.
    mc_rows = {row["scenario"]: row for row in payload["scenario_monte_carlo_evidence"]}
    assert mc_rows["EB"]["mc_effective_sample_size"] == 2.0
    # A scenario's measured wall split reaches summary.json; an older record
    # without one stays without one rather than gaining a fabricated zero.
    assert mc_rows["TP"]["phase_seconds"] == {"draw": 1.5, "wait": 0.25, "evaluate": 10.0}
    assert "phase_seconds" not in mc_rows["EB"]
    # The sampling evidence behind a sub-floor ESS is published with the row.
    assert mc_rows["TP"]["transiting_draws"] == 12
    assert mc_rows["TP"]["finite_likelihood_draws"] == 900
    assert mc_rows["TP"]["proposal_support"]["state"] == "usable"
