"""Gate-conditional nested-sampling escalation for sub-floor TREX scenarios.

One-draw-dominated importance estimates can vary with seed without resolving
the integral. Proposal-support labels do not establish convergence or empty
likelihood support.

At the production budget, eligible sub-floor scenarios are nested-escalated
when aggregate convergence fails or their estimate times the registered residual
factor could flip C1/C2. Support labels are retained, not routing vetoes.
The nested integrator evaluates the same scenario model.
Log evidences and their errors are in nats, probabilities are dimensionless,
and physical inputs retain the units required by :mod:`nested_evidence`.
Substitution requires the registered acceptance checks, which do not certify
global likelihood exploration or an empty feasible set.

Scope and boundaries
--------------------
* Escalation is attempted only at or above the registered production draw
  budget (``run_trex_vetting`` passes it as ``escalation_budget_floor``).
  Smaller test budgets defer if convergence is unmet.
* Eligibility follows the available scenario integrators, not the proposal
  support label. Missing inputs remain unresolved. Accepted records retain
  the support classification as diagnostic provenance.
* A substituted scenario must either land below the registered bound
  acceptance fraction of the unconverged allowance (single run, recorded as a
  bounded contribution) or be confirmed by an ensemble whose own standard
  error meets the project's 0.1-nat integration bar.  Nothing is accepted on
  ``dynesty``'s reported error alone: run-to-run scatter divided by reported
  error was 0.55-3.66 across scenarios and 3-5 for hosted EBx2P.
  An ensemble that misses the bar is refined by defensive
  importance sampling over its pooled nested posteriors at the registered draw
  budget, accepted only if that estimate meets the same bar, the Monte Carlo
  effective-sample-size floor and three-sigma agreement with the ensemble.
* An escalation that cannot run (dynesty absent, sampler truncated by its call
  cap, ensemble not converged, per-run scenario cap reached) leaves the
  scenario in the Monte Carlo evidence set with the failure reason recorded.
  The caller must enforce the unresolved-evidence publication boundary.
* Nested runner seeds are derived from the run's ``random_seed`` through
  ``numpy.random.SeedSequence`` per scenario index, so a repeat run with the
  same inputs and seed reproduces the same estimates.

The owning TREX decision retains ``claim_eligible`` false.
"""

from __future__ import annotations

import sys as sys

import math as math
import json as json
import os as os
import time as _time  # noqa: F401 - historical re-export/patch surface
from contextvars import ContextVar as ContextVar
from functools import wraps as wraps
from pathlib import Path as Path
from typing import Any as Any, Dict as Dict, List as List, Mapping as Mapping, Optional as Optional, Tuple as Tuple

import numpy as np  # noqa: F401 - historical re-export/patch surface

from trex.foundation.physical_constants import FRACTION_PER_PART_PER_MILLION as FRACTION_PER_PART_PER_MILLION
from trex.foundation._numerics import _normalize_probabilities as _normalize_probabilities
from trex.evidence.diagnostics import residual_could_change_decision as residual_could_change_decision
from trex.evidence.marginal_likelihoods import MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY as MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY, MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE as MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, N_NEIGHBOR_SCENARIOS as N_NEIGHBOR_SCENARIOS, N_TARGET_SCENARIOS as N_TARGET_SCENARIOS, TARGET_SCENARIO_SPECS as TARGET_SCENARIO_SPECS, _SCENARIO_NAMES as _SCENARIO_NAMES, _period_weighted_evidences as _period_weighted_evidences, resolved_neighbor_host as resolved_neighbor_host, unconverged_probability_mass as unconverged_probability_mass
from trex.evidence.nested_evidence import NestedCheckpointPaused as NestedCheckpointPaused, _checkpoint_identity as _checkpoint_identity, _load_completed as _load_completed, _save_completed as _save_completed, _selected_nested_estimator as _selected_nested_estimator, _DEFAULT_EVIDENCE_TOLERANCE as _DEFAULT_EVIDENCE_TOLERANCE, _STOP_REASON_TRUNCATED as _STOP_REASON_TRUNCATED, _STOP_REASON_UNRESOLVED as _STOP_REASON_UNRESOLVED, background_star_population as background_star_population, bound_companion_population as bound_companion_population, eb_scenario_nested_evidence as eb_scenario_nested_evidence, nested_checkpoint as nested_checkpoint, tp_scenario_nested_evidence as tp_scenario_nested_evidence
from trex.evidence.nested_controls import _HOSTED_N_DIMENSIONS as _HOSTED_N_DIMENSIONS, _NATIVE_INNER_STEPS_PER_DIMENSION as _NATIVE_INNER_STEPS_PER_DIMENSION, _native_kernel_policy_identity as _native_kernel_policy_identity, _selected_importance_proposal as _selected_importance_proposal
from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS as _BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling as _blackjax_call_ceiling

from trex.evidence.evidence_escalation_calls import _bounded_escalation as _bounded_escalation
from trex.evidence.evidence_escalation_calls import _failing_indices as _failing_indices
from trex.evidence.evidence_escalation_calls import _importance_seed as _importance_seed
from trex.evidence.evidence_escalation_calls import _is_support_unreachable as _is_support_unreachable
from trex.evidence.evidence_escalation_calls import _nested_call as _nested_call
from trex.evidence.evidence_escalation_calls import _nested_dimensions as _nested_dimensions
from trex.evidence.evidence_escalation_calls import _population_blocks as _population_blocks
from trex.evidence.evidence_escalation_calls import _scenario_name as _scenario_name
from trex.evidence.evidence_escalation_calls import _scenario_seeds as _scenario_seeds
from trex.evidence.evidence_escalation_calls import _target_plan_entry as _target_plan_entry
from trex.evidence.evidence_escalation_policy import _ensemble_summary as _ensemble_summary
from trex.evidence.evidence_escalation_policy import _format_nats as _format_nats
from trex.evidence.evidence_escalation_policy import _importance_refusal as _importance_refusal
from trex.evidence.evidence_escalation_policy import _mc_importance_refusal as _mc_importance_refusal
from trex.evidence.evidence_escalation_policy import _is_unresolved_nested_run as _is_unresolved_nested_run
from trex.evidence.evidence_escalation_policy import _nested_likelihood_call_cap as _nested_likelihood_call_cap
from trex.evidence.evidence_escalation_policy import evidence_needs_escalation as evidence_needs_escalation
from trex.evidence.evidence_escalation_policy import residual_could_flip_decision as residual_could_flip_decision
from trex.evidence.evidence_escalation_policy import stage_scenarios as stage_scenarios


# NUMERICAL_POLICY: a substituted scenario can be accepted from one nested run
# when its estimated probability against the initially converged estimates is
# at or below this fraction of the unconverged allowance. Omitting the other
# failing scenarios gives a conditional normalization bound, not a certified
# upper bound on the true evidence or its integration error. A scenario above
# this policy threshold requires the ensemble's own standard error.
_BOUND_ACCEPTANCE_FRACTION: float = 0.1

# NUMERICAL_POLICY: registered independent-run count for a substituted scenario
# that is not decisively below the conditional bound. This fixed ensemble size
# does not guarantee that its standard error meets the integration bar; the
# realized ensemble must pass that separate acceptance check.
_ENSEMBLE_RUNS: int = 3

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

# NUMERICAL_POLICY: ceiling on nested scenarios escalated in one evidence run.
# Measured cost is roughly 3-12 minutes per scenario on six CPU workers,
# so a run that needed this many would already
# be the pathological case; the cap bounds the added wall clock and any
# scenario left over keeps its Monte Carlo entry so the gate defers honestly.
_MAXIMUM_ESCALATED_SCENARIOS_PER_RUN: int = 18

# NUMERICAL_POLICY: hard ceiling on one nested run's likelihood evaluations.
# Natural stopping was 0.34-0.85 million likelihood calls on the reference scenes;
# a run that exceeds twice that has left its
# measured regime, so it is truncated and reported as unconverged rather than
# allowed to run unbounded.
_MAXIMUM_NESTED_LIKELIHOOD_CALLS: int = 2_000_000

# NUMERICAL_POLICY: the opt-in BlackJAX route runs at the 3,200-live configuration
# that passed the preregistered known-Z Gaussian control (48/50 within 0.1 nat;
# the 400-live setting was 29/50 local/r9, 33/50 remote). Dynesty keeps its own
# default and cap above.
# A batched deletion reserves (N/2) * 3d * 111 logical calls before dispatch; at
# N=3,200, d=4 that is 2,131,200, so the old 2,000,000 cap forced the scalar path.
# Twice that reservation, rounded down to whole millions, is 4,000,000 here:
# enough for the measured ~1.53M-call runs and a worst-case batch.
_BLACKJAX_SAMPLER_NAME: str = "blackjax-nss-experimental"

def _blackjax_live_points(n_live_points: Optional[int] = None) -> int:
    """The calibrated BlackJAX live count; a lower explicit request is refused."""
    if n_live_points is None:
        return _BLACKJAX_CALIBRATED_LIVE_POINTS
    if isinstance(n_live_points, bool) or not isinstance(n_live_points, (int, np.integer)):
        raise ValueError("BlackJAX n_live_points must be an integer")
    if int(n_live_points) < _BLACKJAX_CALIBRATED_LIVE_POINTS:
        raise ValueError(
            "BlackJAX nested evidence is calibrated at {0} live points; "
            "explicit n_live_points={1!r} is refused".format(
                _BLACKJAX_CALIBRATED_LIVE_POINTS, n_live_points
            )
        )
    return int(n_live_points)

def _blackjax_likelihood_call_cap(
    n_live_points: int, n_dimensions: int = _HOSTED_N_DIMENSIONS
) -> int:
    """Derived ceiling for this live/dimension geometry, in whole millions."""
    return _blackjax_call_ceiling(
        int(n_live_points),
        int(n_live_points) // 2,
        _NATIVE_INNER_STEPS_PER_DIMENSION * int(n_dimensions),
    )

# Operational ceiling, not an evidence stopping rule. The caller supplies an
# earlier remaining host deadline when setup/earlier phases used that allowance.
_DEFAULT_ESCALATION_WALL_SECONDS = 540 * 60.0
_escalation_runtime = ContextVar("trex_escalation_runtime", default=None)


# Proposal-support diagnostic labels retained for callers. They describe
# requested proposal windows, not the support of the full likelihood, and
# must not determine nested-integrator eligibility.
_SUPPORT_UNREACHABLE_STATES: frozenset = frozenset(
    {"below-support", "above-support"}
)

# NUMERICAL_POLICY: nested stop-provenance values that mark a run as not
# integrated to the registered evidence tolerance. A run reporting one of
# these keeps its Monte Carlo entry and the gate defers. Unknown provenance
# also cannot establish completion. A likelihood plateau is a completed integral.
_UNRESOLVED_STOP_REASONS: frozenset = frozenset(
    {_STOP_REASON_TRUNCATED, _STOP_REASON_UNRESOLVED, "unknown"}
)


class NestedEvidenceUnavailableError(RuntimeError):
    """The nested estimator could not be used for an escalated scenario."""



from trex.evidence.evidence_escalation_run import run_nested_escalation as run_nested_escalation  # noqa: E402 - driver defaults need the facade controls above

__all__ = [
    "NestedEvidenceUnavailableError",
    "evidence_needs_escalation",
    "residual_could_flip_decision",
    "run_nested_escalation",
    "stage_scenarios",
]
