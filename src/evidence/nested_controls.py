"""Controls and result types for nested evidence estimation."""

from __future__ import annotations

import math
import os
from typing import NamedTuple, Optional, Tuple

import numpy as np




def _facade(name):
    from trex.evidence import nested_evidence

    return getattr(nested_evidence, name)

# The isotropic inclination prior's support boundary, in the degrees
# convention ``sample_inc`` uses; a proposal past it lies outside the
# scenario prior and carries no support.
_INCLINATION_MAX_DEGREES: float = 90.0

# NUMERICAL_POLICY: the hosted scenarios' parameter space is four unit-cube
# variates -- occultor-size quantile (planet radius or binary mass ratio),
# eccentricity quantile, argument-of-periastron quantile, and the isotropic
# cosine-inclination variate.  It is a fixed structural count, not a knob.
_HOSTED_N_DIMENSIONS: int = 4

# Background populations use two variates; bound populations also draw an
# independent circular-orbit sky projection.
_POPULATION_N_DIMENSIONS: int = 2
_BOUND_POPULATION_N_DIMENSIONS: int = 3

# NUMERICAL_POLICY: initial live points for the 4-dimensional integration.
# Skilling (2006) requires nlite >> ndim for a stable evidence estimate;
# Monte Carlo reached an effective sample size of only about 3 in 4,000 draws;
# 400 live points keeps each scenario's call count at a few hundred thousand.
_DEFAULT_LIVE_POINTS: int = 400

# NUMERICAL_POLICY: dynesty's remaining-log-evidence-error stopping tolerance,
# in nats.  Half of the 0.1 nat error the scenario-probability derivation in
# ``marginal_likelihoods`` bounds a ten-percent probability-ratio error with,
# so the sampler stops with margin against the comparison tolerance.
_DEFAULT_EVIDENCE_TOLERANCE: float = 0.05

_SAMPLER_NAME: str = "dynesty.DynamicNestedSampler"

# NUMERICAL_POLICY: opt-in BlackJAX nested slice sampling, selected only by
# TREX_NESTED_ESTIMATOR=blackjax; dynesty stays the default until the
# known-Z/coverage calibration passes. Yallup et al. (2026, arXiv 2601.23252v2,
# appendix) delete k = m/2 per batch and use p = d inner slice steps, tripled
# to 3d on harder problems; the driver's fixed isotropic directions mix worse
# than their adaptive ones, so the tripled setting is used.
_NESTED_ESTIMATOR_ENV: str = "TREX_NESTED_ESTIMATOR"
_NATIVE_INNER_STEPS_PER_DIMENSION: int = 3

# Opt-in execution policies. Unset means the assessed default: isotropic
# directions, sequential initialization, no speculation. Selecting a
# non-default value changes execution only; it never inherits the default
# route's calibration, and it enters checkpoint/result identity.
_BLACKJAX_DIRECTION_ENV: str = "TREX_BLACKJAX_PRECOND"
_BLACKJAX_BATCH_INIT_ENV: str = "TREX_BLACKJAX_BATCH_INIT"
_BLACKJAX_W_MARCH_ENV: str = "TREX_BLACKJAX_W_MARCH"
_IMPORTANCE_PROPOSAL_ENV: str = "TREX_IMPORTANCE_PROPOSAL"
_BLACKJAX_INITIALIZATION_BLOCK: int = 64
_BLACKJAX_DIRECTIONS = {"eye": "fixed-isotropic", "pilot": "fixed-pilot", "split": "split-ensemble"}
_IMPORTANCE_PROPOSALS = ("kernel-mixture", "probit-gaussian")


def _selected_importance_proposal() -> str:
    """Validated defensive-proposal family; kernel mixture remains the default."""
    proposal = os.environ.get(_IMPORTANCE_PROPOSAL_ENV, "").strip() or _IMPORTANCE_PROPOSALS[0]
    if proposal not in _IMPORTANCE_PROPOSALS:
        raise ValueError("unknown {0}={1!r}; select kernel-mixture or probit-gaussian".format(
            _IMPORTANCE_PROPOSAL_ENV, proposal))
    return proposal


def _native_kernel_policy() -> dict:
    """Validated BlackJAX execution policy; all defaults match the assessed route.

    ``TREX_BLACKJAX_PRECOND`` selects eye/pilot/split directions,
    ``TREX_BLACKJAX_BATCH_INIT`` enables padded sequential-equivalent
    initialization, and ``TREX_BLACKJAX_W_MARCH`` selects 4/8 speculative
    stepping-out probes. An invalid or unsupported value raises before any
    evaluation; it is never silently ignored.
    """
    direction = os.environ.get(_BLACKJAX_DIRECTION_ENV, "").strip().lower() or "eye"
    if direction not in _BLACKJAX_DIRECTIONS:
        raise ValueError("unknown {0}={1!r}; select eye, pilot or split".format(
            _BLACKJAX_DIRECTION_ENV, direction))
    batch_init = os.environ.get(_BLACKJAX_BATCH_INIT_ENV, "").strip() or "0"
    if batch_init not in ("0", "1"):
        raise ValueError("{0} must be 0 or 1".format(_BLACKJAX_BATCH_INIT_ENV))
    speculation = os.environ.get(_BLACKJAX_W_MARCH_ENV, "").strip() or "0"
    if speculation not in ("0", "4", "8"):
        raise ValueError("{0} must be 0, 4 or 8".format(_BLACKJAX_W_MARCH_ENV))
    return {"direction_metric": _BLACKJAX_DIRECTIONS[direction],
            "batched_initialization": batch_init == "1",
            "initialization_block": _BLACKJAX_INITIALIZATION_BLOCK,
            "speculative_expansions": int(speculation)}


def _native_kernel_policy_identity(policy: Optional[dict] = None) -> tuple:
    """Checkpoint entries for non-default policy fields; empty for the default."""
    policy = _native_kernel_policy() if policy is None else policy
    identity = []
    if policy["direction_metric"] != "fixed-isotropic":
        identity.append(policy["direction_metric"])
    if policy["batched_initialization"]:
        identity.append(("batched-initialization", policy["initialization_block"]))
    if policy["speculative_expansions"]:
        identity.append(("speculative-expansions", policy["speculative_expansions"]))
    return tuple(identity)

# NUMERICAL_POLICY: random slice sampling.  These posteriors are thin curved
# shells in the unit cube, where the default random-walk proposals produced
# bootstrapped ellipsoids large enough that dynesty warned about inefficient
# uniform sampling; slice sampling reflects off the cube faces and needs no
# enlargement factor.
_SAMPLING_METHOD: str = "rslice"

_DYNESTY_STOPPED_SHORT_WARNING: str = "sampling was stopped short"

# NUMERICAL_POLICY: stop-provenance vocabulary shared with the escalation
# acceptance policy in ``evidence_escalation``. A returned run states why its
# base integration ended: the registered evidence tolerance was reached, every
# final live point shared one likelihood (a plateau, whose remaining mass the
# appended live set accounts for exactly), the run was truncated by a budget,
# or none of these could be certified. ``unknown`` is the structural default of
# a hand-constructed record and is never produced by the production runner.
_STOP_REASON_EVIDENCE_TOLERANCE: str = "evidence-tolerance"
_STOP_REASON_LIKELIHOOD_PLATEAU: str = "likelihood-plateau"
_STOP_REASON_TRUNCATED: str = "truncated"
_STOP_REASON_UNRESOLVED: str = "unresolved"
_STOP_REASON_UNKNOWN: str = "unknown"

# NUMERICAL_POLICY: worker processes are started with "spawn", not the
# Linux default "fork".  Importing this module pulls in the JAX accelerator
# through ``marginal_likelihoods``, and forking an interpreter that has
# initialized a CUDA context is not safe.  Spawn pays one interpreter
# startup per worker and is correct.
_MULTIPROCESSING_START_METHOD: str = "spawn"

# Worker processes come from the shared hardware profile
# (``hardware.get_default_n_jobs``): every CPU-bound path in this project
# uses the machine's derived count, and ``n_jobs=None`` means exactly
# that rather than one core.

class NestedEvidenceResult(NamedTuple):
    """A nested-sampling log evidence and its convergence diagnostics."""

    log_evidence: float
    log_evidence_error: float
    n_live_points: int
    n_likelihood_calls: int
    n_iterations: int
    sampler: str
    sampler_version: str
    random_seed: int
    stopped_short: bool = False
    # Provenance added with the monitored-budget runner. ``n_likelihood_calls``
    # remains the sampler's own bookkeeping (including its pseudo-calls);
    # ``monitored_likelihood_calls`` is the exact count taken at evaluation
    # entry through the shared counter, so it is the number the registered
    # budget is enforced against. ``base_run_delta_logz`` is the base
    # integration's remaining-evidence ratio at its final stopping check, in
    # nats, or NaN when it could not be recovered.
    monitored_likelihood_calls: int = 0
    budget_exhausted: bool = False
    stopped_on_evidence_tolerance: bool = False
    stop_reason: str = _STOP_REASON_UNKNOWN
    base_run_delta_logz: float = float("nan")
    # The run's weighted posterior on the unit cube (dynesty ``samples_u`` and
    # normalized importance weights), kept so an ensemble can be refined by
    # importance sampling (``_run_importance_sampler``); never written to a record.
    posterior_unit_samples: Optional[np.ndarray] = None
    posterior_weights: Optional[np.ndarray] = None


# NUMERICAL_POLICY: importance-sampling refinement of a nested ensemble.
# For a hosted EBx2P scene, independent nested runs scattered 0.35-0.62 nat while each reported
# ~0.12 and their simulated compression error was ~0.17: the slice proposals
# are not independent draws there, which no reported error captures. A
# defensive mixture of the unit-cube prior and a kernel density over the
# ensemble's pooled posterior is an exact importance proposal (support covers
# the prior, every weight is at most L / fraction; Hesterberg 1995,
# Technometrics 37, 185, DOI 10.1080/00401706.1995.10484303), so its estimate is
# unbiased in linear evidence and its standard error is the sample error of
# the weights; the logarithm of the estimate is not generally unbiased.
# The draw count is the registered production Monte Carlo budget; the kernel
# count, Scott's bandwidth n^(-1/(d+4)) (Scott 1992, Multivariate Density
# Estimation) and the defensive fraction change only the variance, never the
# value converged to. Measured: 1e6 draws gave 0.035 nat at Kish ESS 836, two
# independent replicates agreeing, against the ensembles' 0.2-0.6 nat.
_IMPORTANCE_DRAWS: int = 1_000_000
_IMPORTANCE_CENTRES: int = 2000
_IMPORTANCE_DEFENSIVE_FRACTION: float = 0.1
# NUMERICAL_POLICY: the stage's adaptive passes. A pilot under this Kish ESS is tempered so
# its centres spread; each pass draws this many points; passes end once the plain weights reach
# the ESS below, or at the round limit. They shape the proposal and never enter the estimate.
_IMPORTANCE_PILOT_ESS: float = 200.0
_IMPORTANCE_ADAPTATION_DRAWS: int = 100_000
_IMPORTANCE_ADAPTATION_ROUNDS: int = 6
_IMPORTANCE_ADAPTATION_ESS: float = 1000.0
_IMPORTANCE_ESTIMATOR: str = "nested-importance-sampling"
# Points per worker task and per kernel-density block: memory/throughput only.
_IMPORTANCE_EVALUATION_CHUNK: int = 1000
_IMPORTANCE_DENSITY_CHUNK: int = 2000
_IMPORTANCE_CHECKPOINT_SECONDS: float = 60.0


class ImportanceEvidenceResult(NamedTuple):
    """A defensive importance estimate with total logical evaluation cost.

    ``n_draws`` counts only fresh estimation draws, including zero-weight
    out-of-cube draws. ``n_likelihood_calls`` also includes adaptive evaluations
    inside the cube, but excludes device pilots, padding and retries.
    """

    log_evidence: float
    log_evidence_error: float
    n_draws: int
    n_likelihood_calls: int
    effective_sample_size: float
    maximum_weight_share: float
    n_centres: int
    defensive_fraction: float
    random_seed: int
    estimator: str = _IMPORTANCE_ESTIMATOR
    execution_provenance: Optional[dict] = None


class _SamplerControls(NamedTuple):
    n_live_points: int
    evidence_tolerance: float
    max_likelihood_calls: Optional[int]
    random_seed: int
    n_jobs: int


def _validate_scalar(value: float, label: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("{0} must be a finite number".format(label))
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        raise ValueError("{0} must be finite{1}".format(label, " and positive" if positive else ""))
    return parsed


def _validated_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    aperture_flux_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, "np.ndarray | float"]:
    """Admit real finite photometry and renormalize it onto the scenario host.

    Time and flux are matching non-empty vectors; sigma is positive and scalar
    or per-point. Boolean/text inputs raise ValueError before numeric coercion.
    """
    time_values = np.asarray(time)
    flux_values = np.asarray(flux)
    if time_values.dtype.kind not in "iuf":
        raise ValueError("time must contain finite real numbers, not boolean or text")
    time_values = time_values.astype(float, copy=False)
    if time_values.ndim != 1 or flux_values.shape != time_values.shape or time_values.size == 0:
        raise ValueError("time and flux must be matching non-empty one-dimensional arrays")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("time must be finite")
    # Preserve raw dtypes until the shared flux/uncertainty validator admits them.
    host_flux, host_sigma = _facade("renorm_flux_to_host")(
        flux_values, sigma, aperture_flux_fraction
    )
    return time_values, host_flux, host_sigma


def _validated_controls(
    n_live_points: int,
    evidence_tolerance: float,
    max_likelihood_calls: Optional[int],
    random_seed: int,
    n_dimensions: int = _HOSTED_N_DIMENSIONS,
    n_jobs: Optional[int] = None,
) -> _SamplerControls:
    if (isinstance(n_dimensions, (bool, np.bool_)) or not isinstance(n_dimensions, (int, np.integer))
            or n_dimensions <= 0):
        raise ValueError("parameter count must be a positive integer")
    if (isinstance(n_live_points, (bool, np.bool_)) or not isinstance(n_live_points, (int, np.integer))
            or n_live_points <= n_dimensions):
        raise ValueError("n_live_points must exceed the parameter count")
    tolerance = _validate_scalar(evidence_tolerance, "evidence_tolerance", positive=True)
    calls: Optional[int] = None
    if max_likelihood_calls is not None:
        if (isinstance(max_likelihood_calls, (bool, np.bool_)) or not isinstance(max_likelihood_calls, (int, np.integer))
                or max_likelihood_calls <= 0):
            raise ValueError("max_likelihood_calls must be a positive integer")
        calls = int(max_likelihood_calls)
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    workers = _facade("resolve_n_jobs")(n_jobs)
    return _SamplerControls(
        n_live_points=int(n_live_points),
        evidence_tolerance=tolerance,
        max_likelihood_calls=calls,
        random_seed=int(random_seed),
        n_jobs=workers,
    )
