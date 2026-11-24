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
from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS as _BLACKJAX_CALIBRATED_LIVE_POINTS  # noqa: F401 - historical re-export/patch surface
from trex.evidence.diagnostics import residual_could_change_decision as residual_could_change_decision
from trex.evidence.evidence_escalation_calls import (
    NestedEvidenceUnavailableError as NestedEvidenceUnavailableError,
    _BLACKJAX_SAMPLER_NAME as _BLACKJAX_SAMPLER_NAME,
    _BOUND_ACCEPTANCE_FRACTION as _BOUND_ACCEPTANCE_FRACTION,
    _DEFAULT_ESCALATION_WALL_SECONDS as _DEFAULT_ESCALATION_WALL_SECONDS,
    _ENSEMBLE_RUNS as _ENSEMBLE_RUNS,
    _MAXIMUM_ESCALATED_SCENARIOS_PER_RUN as _MAXIMUM_ESCALATED_SCENARIOS_PER_RUN,
    _MAXIMUM_NESTED_LIKELIHOOD_CALLS as _MAXIMUM_NESTED_LIKELIHOOD_CALLS,
    _SUPPORT_UNREACHABLE_STATES as _SUPPORT_UNREACHABLE_STATES,
    _blackjax_likelihood_call_cap as _blackjax_likelihood_call_cap,
    _blackjax_live_points as _blackjax_live_points,
    _bounded_escalation as _bounded_escalation,
    _escalation_runtime as _escalation_runtime,
    _failing_indices as _failing_indices,
    _importance_seed as _importance_seed,
    _is_support_unreachable as _is_support_unreachable,
    _nested_call as _nested_call,
    _nested_dimensions as _nested_dimensions,
    _population_blocks as _population_blocks,
    _scenario_name as _scenario_name,
    _scenario_seeds as _scenario_seeds,
    _target_plan_entry as _target_plan_entry,
)
from trex.evidence.evidence_escalation_policy import (
    _IMPORTANCE_AGREEMENT_SIGMA as _IMPORTANCE_AGREEMENT_SIGMA,
    _MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS as _MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS,
    _STAGE_MARGIN_NATS as _STAGE_MARGIN_NATS,
    _ensemble_summary as _ensemble_summary,
    _format_nats as _format_nats,
    _importance_refusal as _importance_refusal,
    _is_unresolved_nested_run as _is_unresolved_nested_run,
    _mc_importance_refusal as _mc_importance_refusal,
    _nested_likelihood_call_cap as _nested_likelihood_call_cap,
    evidence_needs_escalation as evidence_needs_escalation,
    residual_could_flip_decision as residual_could_flip_decision,
    stage_scenarios as stage_scenarios,
)
from trex.evidence.evidence_escalation_run import run_nested_escalation as run_nested_escalation
from trex.evidence.marginal_likelihoods import (
    MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY as MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY,
    _period_weighted_evidences as _period_weighted_evidences,
    unconverged_probability_mass as unconverged_probability_mass,
)
from trex.evidence.mc_scenario import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE as MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
from trex.evidence.mc_target import (
    N_NEIGHBOR_SCENARIOS as N_NEIGHBOR_SCENARIOS,
    N_TARGET_SCENARIOS as N_TARGET_SCENARIOS,
    TARGET_SCENARIO_SPECS as TARGET_SCENARIO_SPECS,
    _SCENARIO_NAMES as _SCENARIO_NAMES,
    resolved_neighbor_host as resolved_neighbor_host,
)
from trex.evidence.nested_checkpoint import (
    NestedCheckpointPaused as NestedCheckpointPaused,
    checkpoint_identity as _checkpoint_identity,  # noqa: F401 - historical re-export/patch surface
    load_completed as _load_completed,  # noqa: F401 - historical re-export/patch surface
    nested_checkpoint as nested_checkpoint,
    save_completed as _save_completed,  # noqa: F401 - historical re-export/patch surface
)
from trex.evidence.nested_controls import (
    _DEFAULT_EVIDENCE_TOLERANCE as _DEFAULT_EVIDENCE_TOLERANCE,
    _HOSTED_N_DIMENSIONS as _HOSTED_N_DIMENSIONS,
    _STOP_REASON_TRUNCATED as _STOP_REASON_TRUNCATED,
    _STOP_REASON_UNRESOLVED as _STOP_REASON_UNRESOLVED,
    _native_kernel_policy_identity as _native_kernel_policy_identity,
    _selected_importance_proposal as _selected_importance_proposal,
    _selected_nested_estimator as _selected_nested_estimator,
)
from trex.evidence.nested_evidence import (
    background_star_population as background_star_population,
    bound_companion_population as bound_companion_population,
    eb_scenario_nested_evidence as eb_scenario_nested_evidence,
    tp_scenario_nested_evidence as tp_scenario_nested_evidence,
)


# NUMERICAL_POLICY: nested stop-provenance values that mark a run as not
# integrated to the registered evidence tolerance. A run reporting one of
# these keeps its Monte Carlo entry and the gate defers. Unknown provenance
# also cannot establish completion. A likelihood plateau is a completed integral.
_UNRESOLVED_STOP_REASONS: frozenset = frozenset(
    {_STOP_REASON_TRUNCATED, _STOP_REASON_UNRESOLVED, "unknown"}
)

__all__ = [
    "NestedEvidenceUnavailableError",
    "evidence_needs_escalation",
    "residual_could_flip_decision",
    "run_nested_escalation",
    "stage_scenarios",
]
