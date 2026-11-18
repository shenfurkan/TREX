"""BlackJAX nested-sampling result and diagnostic types."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import operator
from typing import NamedTuple

import numpy as np


# The only BlackJAX release whose slice and NSS internals were checked.
BLACKJAX_VERSION: str = "1.7.1"

# Slice computational guards (BlackJAX ``max_expansions``/``max_shrinkage``),
# not scientific budgets; shared by the BlackJAX kernel and the budgeted copy.
SLICE_MAX_EXPANSIONS: int = 10
SLICE_MAX_SHRINKAGE: int = 100

# Most logical evaluations one budgeted slice can make: two initial bracket
# ends, at most j + k = max_expansions - 1 step-outs (j is drawn in
# 0..max_expansions-1) and at most max_shrinkage shrinks. Verified against
# blackjax/1.7.1/blackjax/mcmc/slice.py stepping_out and _shrink. This bounds
# logical calls, not padded vmap work.
SLICE_CALL_BOUND: int = 2 + (SLICE_MAX_EXPANSIONS - 1) + SLICE_MAX_SHRINKAGE

# The live count that passed the preregistered 4-D Gaussian known-Z control
# (48/50 seeds within 0.1 nat; 400 live points failed at 29/50). The README's
# "Opt-in BlackJAX" section records both cohorts.
BLACKJAX_CALIBRATED_LIVE_POINTS: int = 3_200


def _positive_count(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    value = operator.index(value)
    if value < 1:
        raise ValueError("BlackJAX call ceiling requires positive live, deletion and slice counts")
    return value


def blackjax_call_ceiling(n_live_points: int, num_delete: int, num_inner_steps: int) -> int:
    """Logical-call ceiling derived from a run's live/deletion geometry.

    A batched deletion reserves ``num_delete * num_inner_steps * SLICE_CALL_BOUND``
    calls before dispatch. The ceiling is twice that reservation's whole
    millions (4,000,000 at 3,200 live and 4-D, room for a full run plus one
    worst-case batch); a reservation under one million gets exactly twice
    itself, so the ceiling always admits at least one batch. ``n_live_points``
    is validated with the others but does not enter the formula.
    """
    _positive_count(n_live_points, "n_live_points")
    reservation = (_positive_count(num_delete, "num_delete")
                   * _positive_count(num_inner_steps, "num_inner_steps") * SLICE_CALL_BOUND)
    if reservation < 1_000_000:
        return 2 * reservation
    return 2 * (reservation // 1_000_000) * 1_000_000


class CheckedSliceInfo(NamedTuple):
    """Every slice's diagnostics and functional error, including rejected draws."""

    slice_info: object
    errors: object

    @property
    def is_accepted(self):
        return self.slice_info.is_accepted


class BudgetSliceInfo(NamedTuple):
    """Functional validity/budget and exact scalar evaluation diagnostics."""

    slice_info: object
    numerically_valid: object
    budget_exhausted: object
    likelihood_calls: object  # [left endpoint/expansion, right, shrinkage]
    remaining_budget: object
    last_position: object
    last_log_integrand: object
    blocked_phase: object  # -1: none; 0: left; 1: right; 2: shrinkage
    blocked_position: object  # not evaluated; meaningful only if blocked_phase >= 0


class _SliceBudget(NamedTuple):
    remaining: object
    counts: object
    valid: object
    exhausted: object
    last_position: object
    last_log_integrand: object
    blocked_phase: object
    blocked_position: object


@dataclass(frozen=True)
class NativeSliceAttempt:
    """One inner slice, including an interrupted/uncommitted proposal."""

    particle_index: int
    inner_index: int
    key: object
    start: object
    proposed: object
    info: BudgetSliceInfo


@dataclass(frozen=True)
class NativeBatchHistory:
    """Rank deletion and replacement attempt; only completed batches have weights."""

    key: object
    threshold: float
    dead_indices: np.ndarray
    dead_particles: object
    replacement_particles: tuple
    slices: tuple[NativeSliceAttempt, ...]
    completed: bool
    effective_live_counts: np.ndarray
    calls_before: int
    calls_after: int
    # Batched execution: stacked (chain_keys, parent_indices, per-step BudgetSliceInfo)
    # instead of per-slice attempts; replacement_particles is then one stacked pytree.
    batched_slices: object = None


@dataclass(frozen=True)
class NativeStoppingCheck:
    """Actual pre-finalization remaining-evidence check, all logarithms in nats."""

    log_dead_evidence: float
    log_remaining_volume: float
    max_live_loglikelihood: float
    delta_logz: float
    evidence_tolerance: float
    n_likelihood_calls: int
    reason: str
    log_likelihood_offset: float
    log_dead_evidence_relative: float
    max_live_loglikelihood_relative: float


@dataclass(frozen=True)
class NativeEvidenceResult:
    """NestedEvidenceResult-compatible diagnostics plus auditable raw history.

    Compression uncertainty is conditional on the sampled likelihood history;
    it omits mixing, mode loss, finite-live mean error and quadrature bias.
    Truncated estimates are descriptive partial-run quadrature, not convergence.
    Positions are dimensionless unit quantiles; log quantities are natural logs.
    Absolute log fields are rounded float64 views. Audit normalization using the
    relative fields and their explicit offset, especially for large log scales.
    If ``likelihood_calls_exact`` is false after a device failure or unclean recovery, integer call
    fields are completed-call lower bounds and physical calls are unavailable.
    """

    log_evidence: float
    log_evidence_error: float
    log_evidence_relative: float
    log_evidence_offset: float
    log_likelihood_offset: float
    n_live_points: int
    n_likelihood_calls: int
    n_iterations: int  # completed outer replacement batches, not dead points
    sampler: str
    sampler_version: str
    random_seed: int
    stopped_short: bool
    monitored_likelihood_calls: int
    budget_exhausted: bool
    stopped_on_evidence_tolerance: bool
    stop_reason: str
    base_run_delta_logz: float
    posterior_unit_samples: object
    posterior_weights: object
    initial_key: object
    continuation_key: object
    compression_key: object
    initial_positions: np.ndarray
    initial_particles: tuple
    initial_state: object
    final_state: object
    batches: tuple[NativeBatchHistory, ...]
    stopping_checks: tuple[NativeStoppingCheck, ...]
    effective_live_counts: np.ndarray
    dead_log_volumes: np.ndarray
    dead_log_volume_weights: np.ndarray
    final_live_log_volume_weights: np.ndarray
    dead_log_evidence_weights: np.ndarray
    final_live_log_evidence_weights: np.ndarray
    dead_log_evidence_weights_relative: np.ndarray
    final_live_log_evidence_weights_relative: np.ndarray
    compression_log_evidences: np.ndarray
    compression_log_evidences_relative: np.ndarray
    physical_likelihood_calls: int | None
    likelihood_calls_exact: bool
    likelihood_call_breakdown: dict[str, int]
    evidence_tolerance: float
    max_likelihood_calls: int
    num_delete: int
    num_inner_steps: int
    failure_reason: str | None
    failed_operation: str | None
    failed_attempt: dict | None
    external_log_evidence_shift: float = 0.0
    quadrature: str = "geometric-volume-riemann-plus-equal-live"
    slice_execution: str = "scalar-device-loop"
    direction_metric: str = "fixed-isotropic"
    speculative_expansions: int = 0
    batched_initialization: bool = False
    claim_eligible: bool = False
    evidence_calibrated: bool = False

    @property
    def configuration_calibrated(self) -> bool:
        """Whether this uses the assessed default policy and known-Z live/cap controls."""
        if (self.direction_metric != "fixed-isotropic" or self.speculative_expansions
                or self.batched_initialization):
            return False
        if np.ndim(self.initial_positions) != 2:
            return False
        n_dimensions = int(np.shape(self.initial_positions)[1])
        return (self.n_live_points == BLACKJAX_CALIBRATED_LIVE_POINTS
                and self.num_delete == self.n_live_points // 2
                and self.num_inner_steps == 3 * n_dimensions
                and self.max_likelihood_calls
                == blackjax_call_ceiling(self.n_live_points, self.num_delete, self.num_inner_steps))

    def _replace(self, **changes):
        """Support existing result consumers, preserving constant-weight audits.

        In particular, ``nested_evidence._with_occurrence_weight`` changes logZ
        by a constant outside the conditional integrand. Apply that same shift
        to absolute contribution/simulation fields, not to normalized posterior,
        conditional stopping checks or compression dispersion. Relative fields
        retain their offset-separated numerical precision.
        """
        if "log_evidence" in changes:
            if isinstance(changes["log_evidence"], bool):
                raise ValueError("log_evidence replacement must be a real number")
            requested = float(changes["log_evidence"])
            if not math.isfinite(requested) or not math.isfinite(self.log_evidence):
                raise ValueError("log_evidence replacement requires finite evidence")
            shift = requested - self.log_evidence
            offset = self.log_evidence_offset + shift
            external_shift = self.external_log_evidence_shift + shift
            if not all(math.isfinite(value) for value in (shift, offset, external_shift)):
                raise ValueError("log_evidence replacement exceeds float64 shift arithmetic")
            changes.update(
                log_evidence_offset=offset,
                external_log_evidence_shift=external_shift,
                dead_log_evidence_weights=self.dead_log_evidence_weights_relative + offset,
                final_live_log_evidence_weights=self.final_live_log_evidence_weights_relative + offset,
                compression_log_evidences=self.compression_log_evidences_relative + offset)
        return replace(self, **changes)


# Prefix of every failure message for a NaN/+inf integrand evaluation.
INVALID_INTEGRAND_FAILURE = "invalid numerical native integrand evaluation"


class NativeIntegrationError(RuntimeError):
    """Scientific/numerical failure carrying its unresolved partial audit record.

    ``numerical_failure_count`` counts the invalid integrand evaluations behind
    a message containing ``INVALID_INTEGRAND_FAILURE``: one at initialization, otherwise
    the invalid slices of the last batch. Other failures carry zero.
    """

    def __init__(self, message: str, result: "NativeEvidenceResult") -> None:
        super().__init__(message)
        self.result = result
        self.numerical_failure_count = 0
        # A device-side checkify failure arrives wrapped as "<operation>: <type>: <text>".
        if INVALID_INTEGRAND_FAILURE in str(message):
            if result.failed_operation == "initialization":
                self.numerical_failure_count = 1
            elif result.batches:
                batch = result.batches[-1]
                if batch.batched_slices is not None:
                    # batched_slices = (chain_keys, parent_indices, per-step BudgetSliceInfo).
                    _, _, step_info = batch.batched_slices
                    valid = np.asarray(step_info.numerically_valid, dtype=bool)
                    self.numerical_failure_count = int(np.count_nonzero(~valid))
                else:
                    self.numerical_failure_count = sum(not bool(item.info.numerically_valid) for item in batch.slices)
