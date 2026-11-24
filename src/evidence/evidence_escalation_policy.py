"""Acceptance and routing policy for evidence escalation."""

from __future__ import annotations


import math
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from trex.foundation._numerics import _normalize_probabilities
from trex.evidence.diagnostics import residual_could_change_decision
from trex.evidence.evidence_escalation_calls import (
    _MAXIMUM_NESTED_LIKELIHOOD_CALLS,
    _failing_indices,
    _scenario_name,
)
from trex.evidence.mc_scenario import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
from trex.evidence.mc_target import _period_weighted_evidences
from trex.evidence.nested_controls import (
    NestedEvidenceUnavailableError,
    _STOP_REASON_EVIDENCE_TOLERANCE,
    _STOP_REASON_LIKELIHOOD_PLATEAU,
)


# NUMERICAL_POLICY: the per-scenario log-evidence standard error the
# probability-ratio derivation in ``marginal_likelihoods`` holds to ten
# percent. It is the same 0.1-nat bar the effective-sample-size floor is
# derived from, applied to the ensemble of nested runs instead of to Kish ESS.
_MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS: float = 0.1

# NUMERICAL_POLICY: an ensemble that misses that bar is refined by importance
# sampling over its own pooled nested posteriors. The
# refinement must meet the same bar and the Monte Carlo effective-sample-size
# floor, and lie within this many combined standard errors of the ensemble
# mean -- the usual three-sigma consistency bound, so an ensemble whose runs
# never visited the refinement's evidence cannot be overruled by it.
_IMPORTANCE_AGREEMENT_SIGMA: float = 3.0

# NUMERICAL_POLICY: sub-floor scenarios this close to the best evidence [nats] get the
# MC-pilot stage. An ESS 1 Monte Carlo TP sat 42 nats under nested on the low-scatter control, so
# its small probability did not mean small mass. Twice that, rounded up; not a bound.
_STAGE_MARGIN_NATS: float = 100.0


def _ensemble_summary(values: List[float]) -> Tuple[float, float]:
    """Mean and standard error of independent nested log-evidence runs, in nats."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise NestedEvidenceUnavailableError("nested evidence ensemble is non-finite")
    if array.size == 1:
        return float(array[0]), float("nan")
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(array.size))


def _importance_refusal(refined: Any, ensemble_mean: float, ensemble_sem: float) -> Optional[str]:
    """Why an importance refinement cannot stand in for its nested ensemble, or ``None``.

    It must meet the same 0.1-nat bar and the Monte Carlo effective-sample-size
    floor the production estimator is held to, and agree with the ensemble it
    refines: a refinement far from every nested run means the proposal and the
    runs disagree about where the evidence is, which no error bar resolves.
    """
    if not (math.isfinite(refined.log_evidence) and math.isfinite(refined.log_evidence_error)):
        return "importance refinement returned a non-finite evidence or error"
    if not math.isfinite(refined.effective_sample_size) or refined.effective_sample_size < MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE:
        return "importance refinement effective sample size {0:.4g} is below the {1:g} floor".format(
            refined.effective_sample_size, MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
        )
    if refined.log_evidence_error < 0 or refined.log_evidence_error > _MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS:
        return "importance refinement standard error {0:.3g} nat exceeds the {1:g}-nat bar".format(
            refined.log_evidence_error, _MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS
        )
    tolerance = _IMPORTANCE_AGREEMENT_SIGMA * math.hypot(refined.log_evidence_error, ensemble_sem)
    if not abs(refined.log_evidence - ensemble_mean) <= tolerance:
        return (
            "importance refinement {0:.4f} disagrees with the nested ensemble mean {1:.4f} "
            "beyond {2:g} combined standard errors".format(
                refined.log_evidence, ensemble_mean, _IMPORTANCE_AGREEMENT_SIGMA
            )
        )
    return None


def _mc_importance_refusal(refined: Any, log_evidence: float, mc_log_evidence: float, mc_ess: float) -> Optional[str]:
    """Registered ESS and error bars, then a one-sided check against the independent first stage.

    ESS and error see only the mass the kernels cover, so a pilot that missed a mode passes
    both, and then sits below Monte Carlo. A sub-floor Monte Carlo estimate can also
    err low, so agreement with it does not prove coverage.
    Its relative error is 1/sqrt(ESS) for Kish ESS.
    """
    if not (math.isfinite(refined.log_evidence) and math.isfinite(refined.log_evidence_error)):
        return "MC-pilot importance returned non-finite evidence or error"
    if not math.isfinite(refined.effective_sample_size) or refined.effective_sample_size < MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE:
        return "MC-pilot importance effective sample size is below the registered floor"
    if refined.log_evidence_error < 0 or refined.log_evidence_error > _MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS:
        return "MC-pilot importance standard error exceeds the registered bar"
    mc_error = 1.0 / math.sqrt(mc_ess) if math.isfinite(mc_ess) and mc_ess > 0.0 else math.inf
    tolerance = _IMPORTANCE_AGREEMENT_SIGMA * math.hypot(refined.log_evidence_error, mc_error)
    if mc_log_evidence - log_evidence > tolerance:
        return "MC-pilot importance {0:.4f} sits below Monte Carlo {1:.4f} beyond {2:g} combined standard errors".format(
            log_evidence, mc_log_evidence, _IMPORTANCE_AGREEMENT_SIGMA)
    return None


def _format_nats(value: float) -> str:
    """A log-evidence ratio in nats for a failure message, or its absence."""
    return "{0:.3g}".format(value) if math.isfinite(value) else "unmeasured"


def _nested_likelihood_call_cap(run: Any) -> int:
    """The run's own likelihood ceiling, or the dynesty-era default.

    Only the BlackJAX result records ``max_likelihood_calls``; a dynesty record
    without it keeps the shared 2,000,000-call limit unchanged.
    """
    cap = getattr(run, "max_likelihood_calls", None)
    return _MAXIMUM_NESTED_LIKELIHOOD_CALLS if cap is None else int(cap)


def _is_unresolved_nested_run(run: Any) -> bool:
    """Whether a nested result's recorded provenance says it did not integrate.

    Only evidence-tolerance and likelihood-plateau stops certify completion.
    Missing or unknown stop provenance cannot establish a completed integral.
    """
    return bool(
        run.stopped_short
        or run.budget_exhausted
        or str(run.stop_reason) not in (_STOP_REASON_EVIDENCE_TOLERANCE, _STOP_REASON_LIKELIHOOD_PLATEAU)
        or int(run.n_likelihood_calls) >= _nested_likelihood_call_cap(run)
    )


def residual_could_flip_decision(
    lnZ: np.ndarray, ess: Mapping[int, float], *, period_days: Optional[float] = None
) -> bool:
    """Whether sub-floor scenarios can flip C1/C2; zero sampled mass is unbounded."""
    values = _period_weighted_evidences(lnZ, period_days)
    probs, status = _normalize_probabilities(values)
    failures = _failing_indices(lnZ, ess)
    if not failures:
        return False
    if status != "ok":
        return status == "all_neginf" and any(np.isneginf(lnZ[index]) for index in failures)
    scenarios = {_scenario_name(index): float(value) for index, value in enumerate(probs)}
    residual = [_scenario_name(index) for index in failures]
    zero = [_scenario_name(index) for index in failures if np.isneginf(lnZ[index])]
    return residual_could_change_decision(
        scenarios, residual, [scenarios], **({"unbounded_residual": zero} if zero else {})
    )


def evidence_needs_escalation(
    lnZ: np.ndarray, ess: Mapping[int, float], *, degenerate: bool, at_production_budget: bool,
    period_days: Optional[float] = None,
) -> bool:
    """Shared escalation trigger for the engine and checkpoint loader."""
    return degenerate or (at_production_budget and (
        residual_could_flip_decision(lnZ, ess, period_days=period_days)
        or bool(stage_scenarios(lnZ, ess, period_days=period_days))
    ))


def stage_scenarios(
    lnZ: np.ndarray, ess: Mapping[int, float], *, period_days: Optional[float] = None
) -> list:
    """Sub-floor scenarios close enough to the best evidence for an ESS 1 error to matter."""
    values = _period_weighted_evidences(lnZ, period_days)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [index for index in _failing_indices(values, ess) if np.isneginf(values[index])]
    floor = float(np.max(finite)) - _STAGE_MARGIN_NATS
    return [index for index in _failing_indices(values, ess)
            if np.isneginf(values[index]) or values[index] >= floor]
