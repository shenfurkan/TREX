"""Execute nested evidence escalation for selected scenarios."""

from __future__ import annotations

from trex.runtime.logging import host_logger


import math
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from trex.foundation.errors import NumericalLikelihoodError

from trex.evidence.evidence_escalation import _DEFAULT_ESCALATION_WALL_SECONDS, _bounded_escalation
from trex.evidence.evidence_escalation_calls import _neighbor_plan_entry


def _facade(name):
    from trex.evidence import evidence_escalation

    return getattr(evidence_escalation, name)

def _record_numerical_failure(outcome, name, exc):
    """Record a scenario's failed supported evaluations, whichever error type carries them.

    Every failure leaves the scenario unmet and the run continuing, so each
    caller logs it at warning level with the ``trex.nested.failed`` event.
    """
    count = exc.count if isinstance(exc, NumericalLikelihoodError) else getattr(exc, "numerical_failure_count", 0)
    if count:
        outcome["numerical_failures"][name] = {"count": int(count), "reason": str(exc)}


def _record_unmet(outcome, name, reason):
    outcome["unmet_scenarios"].append(name)
    outcome["unmet_reasons"][name] = reason
    if outcome["unmet_reason"] is None:
        outcome["unmet_reason"] = reason


@_bounded_escalation
def run_nested_escalation(
    *,
    lnZ: np.ndarray,
    ess: Mapping[int, float],
    support_state: Mapping[int, Mapping[str, Any]],
    escalation_budget_floor: int,
    n_draws: int,
    random_seed: Optional[int],
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    depth_ppm: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    exptime_days: float,
    bin_integration_days: float,
    target_aperture_flux_fraction: float,
    contrast_separations: Optional[np.ndarray],
    contrast_values: Optional[np.ndarray],
    plx: Optional[float],
    target_tess_mag: float,
    background_population: Optional[Mapping[str, np.ndarray]],
    background_area_sq_deg: Optional[float],
    nearby_stars: Optional[List[Mapping[str, Any]]],
    target_feh: Optional[float],
    resolution_limit_arcsec: Optional[float],
    nsamples: int,
    n_live_points: Optional[int] = None,
    n_jobs: Optional[int] = None,
    checkpoint_directory: Optional[Path] = None,
    wall_time_seconds: float = _DEFAULT_ESCALATION_WALL_SECONDS,
    deadline_monotonic: Optional[float] = None,
    target_teff_k: Optional[float] = None,
    baseline=None,
    stellar_light=None,
    joint_observations=None,
) -> dict:
    """Re-integrate finite sub-floor scenarios with nested sampling.

    ``support_state`` records proposal-support diagnostics for this run.
    Eligibility depends on a sub-floor finite evidence and an available
    scenario integrator, never on this classification. Accepted records retain
    it as provenance without interpreting it as a zero-support certificate.
    Log evidences and errors are in nats; probabilities are dimensionless.
    This is integration-policy orchestration, not a new physical model, and
    never changes the owning diagnostic's ``claim_eligible: false`` invariant.

    ``n_live_points`` is an optional explicit live count; BlackJAX uses its
    calibrated 3,200 default when it is omitted and refuses a lower value.

    ``checkpoint_directory`` retains each seed's live sampler, completed result
    and refinement blocks. ``wall_time_seconds`` (capped at 540 minutes) and an
    optional earlier process-monotonic ``deadline_monotonic`` pause at a safe
    checkpoint boundary with ``NestedCheckpointPaused``. Repeating the same
    scientific call/directory restores it; neither limit changes the priors,
    support, evidence stopping rule or likelihood-evaluation budget. Per-attempt
    cost includes failed/paused work and survives the next process.
    An unresolved scenario retains its own reason while independent scenarios
    continue within the same cap and deadline. Any unresolved attempt still
    defers the combined result; later substitutions cannot erase it.

    Returns a mapping with the substituted evidence vector, the remaining
    Monte Carlo effective-sample-size mapping, per-scenario nested provenance
    records, and, when the waiver could not be brought within the allowance,
    the reason and the scenarios left unconverged. ``escalated`` is the list of
    substituted scenario indices; an empty list means the caller keeps the
    Monte Carlo evidence set unchanged.
    """
    outcome: dict = {
        "escalated": [],
        "lnZ": np.array(lnZ, dtype=float),
        "remaining_ess": dict(ess),
        "records": {},
        "unmet_reason": None,
        "unmet_scenarios": [],
        "unmet_reasons": {},
        "unsubstituted_reason": None,
        "unsubstituted_scenarios": [],
        "attempted_scenarios": [],
        "numerical_failures": {},
        "cap_binding": False,
        "seeds": [],
        "attempt_costs": _facade("_escalation_runtime").get()[2],
    }
    if int(n_draws) < int(escalation_budget_floor):
        outcome["unmet_reason"] = (
            "escalation requires the registered production draw budget of "
            "{0} draws".format(escalation_budget_floor)
        )
        return outcome

    failures = _facade("_failing_indices")(lnZ, ess)
    if not failures:
        return outcome
    probs, _ = _facade("_normalize_probabilities")(_facade("_period_weighted_evidences")(lnZ, period_days))
    # Sub-floor mass above the allowance gates the run. Within it, a residual is still
    # re-integrated while its own estimate could flip C1/C2 under C5's registered factor.
    # Missing inputs or a failed attempt stay unmet even within that allowance.
    could_flip = _facade("residual_could_flip_decision")(lnZ, ess, period_days=period_days)
    gating = (
        _facade("unconverged_probability_mass")(probs, ess, lnZ)
        > _facade("MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY")
        or (any(np.isneginf(lnZ[index]) for index in failures)
             and could_flip)
    )
    if not gating and not could_flip:
        return outcome

    # Build the per-scenario plan, resolving populations and neighbor hosts
    # with the same population functions the Monte Carlo path uses.
    observed_depth_fraction = depth_ppm * _facade("FRACTION_PER_PART_PER_MILLION")
    blocks = _facade("_population_blocks")(
        M_s_Msun=M_s_Msun,
        R_s_Rsun=R_s_Rsun,
        plx=plx,
        contrast_separations=contrast_separations,
        contrast_values=contrast_values,
        resolution_limit_arcsec=resolution_limit_arcsec,
        background_population=background_population,
        background_area_sq_deg=background_area_sq_deg,
        target_tess_mag=target_tess_mag,
        target_teff_k=target_teff_k,
        **({"stellar_light": stellar_light, "target_feh": target_feh} if stellar_light is not None else {}),
        **({"joint_observations": joint_observations} if joint_observations is not None else {}),
    )
    plans: Dict[int, dict] = {}
    for index in failures:
        if index < _facade("N_TARGET_SCENARIOS"):
            plan = _facade("_target_plan_entry")(index)
            if plan is None:
                continue
            plans[index] = plan
            continue
        plan = _neighbor_plan_entry(
            index, nearby_stars, target_feh, observed_depth_fraction, stellar_light,
            **({"joint_observations": joint_observations} if joint_observations is not None else {}),
        )
        if plan is not None:
            plans[index] = plan

    # Zero sampled weight is unbounded, so handle it before finite realized masses.
    # A failed attempt defers the combined result, not independent scenario work.
    ordered = sorted(failures, key=lambda index: (np.isneginf(lnZ[index]), float(probs[index])), reverse=True)
    if n_live_points is not None and _facade("_selected_nested_estimator")() == "blackjax":
        _facade("_blackjax_live_points")(n_live_points)
    seeds, recorded_seed = _facade("_scenario_seeds")(random_seed, len(lnZ))
    outcome["seeds"] = seeds
    outcome["random_seed"] = recorded_seed
    substituted = np.array(lnZ, dtype=float)
    converged_reference = np.array(lnZ, dtype=float)
    converged_reference[failures] = float("-inf")
    remaining = dict(ess)
    call_inputs = dict(
        blocks=blocks, time=time, flux=flux, sigma=sigma, period_days=period_days,
        M_s_Msun=M_s_Msun, R_s_Rsun=R_s_Rsun, u1=u1, u2=u2, exptime_days=exptime_days,
        bin_integration_days=bin_integration_days, target_aperture_flux_fraction=target_aperture_flux_fraction,
        target_feh=target_feh, target_teff_k=target_teff_k, nsamples=nsamples, n_jobs=n_jobs,
        **({"baseline": baseline} if baseline is not None else {}),
        **({"stellar_light": stellar_light} if stellar_light is not None else {}),
        **({"joint_observations": joint_observations} if joint_observations is not None else {}),
    )
    attempted = 0
    for index in ordered:
        plan = plans.get(index)
        name = _facade("_scenario_name")(index)
        if plan is None:
            _record_unmet(outcome, name, "nested integration inputs are unavailable for {0}".format(name))
            continue
        if attempted >= _facade("_MAXIMUM_ESCALATED_SCENARIOS_PER_RUN"):
            outcome["cap_binding"] = True
            reason = "per-run nested escalation cap of {0} scenarios reached".format(
                _facade("_MAXIMUM_ESCALATED_SCENARIOS_PER_RUN")
            )
            if gating:
                _record_unmet(outcome, name, reason)
            else:
                # A capped, unattempted residual keeps the registered MC waiver, not a failed-integral label.
                outcome["unsubstituted_scenarios"].append(name)
                if outcome["unsubstituted_reason"] is None:
                    outcome["unsubstituted_reason"] = reason
            continue
        attempted += 1
        outcome["attempted_scenarios"].append(name)
        scenario_seed = seeds[index * _facade("_ENSEMBLE_RUNS")]
        started = _time.monotonic()
        try:
            first = _facade("_nested_call")(
                plan, **call_inputs, random_seed=scenario_seed, n_live_points=n_live_points,
            )
        except _facade("NestedCheckpointPaused"):
            raise
        except NumericalLikelihoodError as exc:
            _record_numerical_failure(outcome, name, exc)
            host_logger("trex").warning("TREX nested escalation of %s failed: %s: %s",
                                       name, type(exc).__name__, exc,
                                       extra={"event": "trex.nested.failed", "scenario": name})
            _record_unmet(outcome, name, "{0}: {1}".format(type(exc).__name__, exc))
            continue
        except _facade("NestedEvidenceUnavailableError") as exc:
            _record_unmet(outcome, name, str(exc))
            continue
        except Exception as exc:  # dynesty missing or sampler failure
            _record_numerical_failure(outcome, name, exc)
            # Keep the traceback: an infrastructure error (e.g. a worker pool that
            # cannot start) must not be indistinguishable from numerical failure.
            host_logger("trex").warning("TREX nested escalation of %s failed: %s: %s",
                                       name, type(exc).__name__, exc, exc_info=exc,
                                       extra={"event": "trex.nested.failed", "scenario": name})
            _record_unmet(outcome, name, "{0}: {1}".format(type(exc).__name__, exc))
            continue
        first_cap = _facade("_nested_likelihood_call_cap")(first)
        if _facade("_is_unresolved_nested_run")(first):
            if first.stopped_short or first.budget_exhausted or (
                first.n_likelihood_calls >= first_cap
            ):
                outcome["cap_binding"] = True
                reason = (
                    "{0} nested run reached its {1}-call ceiling before the "
                    "{2:g}-nat tolerance".format(
                        name,
                        first_cap,
                        _facade("_DEFAULT_EVIDENCE_TOLERANCE"),
                    )
                )
            else:
                reason = (
                    "{0} nested run did not reach the {1:g}-nat evidence "
                    "tolerance (base-run remaining evidence {2} nat, stop "
                    "reason {3})".format(
                        name,
                        _facade("_DEFAULT_EVIDENCE_TOLERANCE"),
                        _facade("_format_nats")(first.base_run_delta_logz),
                        first.stop_reason,
                    )
                )
            _record_unmet(outcome, name, reason)
            continue

        if not all(
            math.isfinite(value)
            for value in (first.log_evidence, first.log_evidence_error, plan["log_prior"])
        ):
            _record_unmet(outcome, name,
                "{0} nested run returned non-finite evidence, error, or prior; "
                "this is not a zero-support certificate".format(name)
            )
            continue
        trial_log_evidence = first.log_evidence + float(plan["log_prior"])
        if not math.isfinite(trial_log_evidence):
            _record_unmet(outcome, name,
                "{0} nested evidence with its scenario prior is non-finite".format(name)
            )
            continue
        bound_vector = np.array(converged_reference, dtype=float)
        bound_vector[index] = trial_log_evidence
        bound_probs, bound_status = _facade("_normalize_probabilities")(
            _facade("_period_weighted_evidences")(bound_vector, period_days)
        )
        if bound_status != "ok" or not math.isfinite(float(bound_probs[index])):
            _record_unmet(outcome, name,
                "{0} nested probability bound has invalid normalization: {1}".format(
                    name, bound_status
                )
            )
            continue
        single_run_probability_bound = float(bound_probs[index])
        evidence_values = [first.log_evidence]
        nested_runs = [first]
        refined = None
        ensemble_sem = float("nan")
        mean_evidence = first.log_evidence
        acceptance = "single-run-below-bound"
        if single_run_probability_bound > (
            _facade("MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY") * _facade("_BOUND_ACCEPTANCE_FRACTION")
        ):
            acceptance = "ensemble"
            try:
                for extra in range(1, _facade("_ENSEMBLE_RUNS")):
                    more = _facade("_nested_call")(
                        plan, **call_inputs,
                        random_seed=seeds[index * _facade("_ENSEMBLE_RUNS") + extra],
                        n_live_points=n_live_points,
                    )
                    more_cap = _facade("_nested_likelihood_call_cap")(more)
                    if _facade("_is_unresolved_nested_run")(more):
                        if more.stopped_short or more.budget_exhausted or (
                            more.n_likelihood_calls >= more_cap
                        ):
                            outcome["cap_binding"] = True
                            raise _facade("NestedEvidenceUnavailableError")(
                                "{0} ensemble run was truncated by its "
                                "likelihood-call ceiling".format(name)
                            )
                        raise _facade("NestedEvidenceUnavailableError")(
                            "{0} ensemble run did not reach the {1:g}-nat "
                            "evidence tolerance (stop reason {2})".format(
                                name,
                                _facade("_DEFAULT_EVIDENCE_TOLERANCE"),
                                more.stop_reason,
                            )
                        )
                    if not all(
                        math.isfinite(value)
                        for value in (more.log_evidence, more.log_evidence_error)
                    ):
                        raise _facade("NestedEvidenceUnavailableError")(
                            "{0} nested ensemble returned non-finite evidence or error; "
                            "this is not a zero-support certificate".format(name)
                        )
                    evidence_values.append(more.log_evidence)
                    nested_runs.append(more)
                mean_evidence, ensemble_sem = _facade("_ensemble_summary")(evidence_values)
            except _facade("NestedCheckpointPaused"):
                raise
            except Exception as exc:
                _record_numerical_failure(outcome, name, exc)
                host_logger("trex").warning(
                    "nested ensemble for %s failed", name, exc_info=exc,
                    extra={"event": "trex.nested.failed", "scenario": name},
                )
                _record_unmet(outcome, name, "{0}: {1}".format(type(exc).__name__, exc))
                continue
            if not math.isfinite(ensemble_sem) or (
                ensemble_sem > _facade("_MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS")
            ):
                failure = (
                    "{0} nested ensemble standard error {1:.3g} nat exceeds the "
                    "{2:g}-nat bar".format(
                        name,
                        ensemble_sem,
                        _facade("_MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS"),
                    )
                )
                posteriors = [
                    (run.posterior_unit_samples, run.posterior_weights) for run in nested_runs
                ]
                if any(u is None or w is None for u, w in posteriors):
                    _record_unmet(outcome, name,
                        failure + " and no nested posterior is available to refine it"
                    )
                    continue
                try:
                    refined = _facade("_nested_call")(
                        plan, **call_inputs,
                        random_seed=_facade("_importance_seed")(recorded_seed, len(lnZ), index),
                        importance_posterior=posteriors,
                    )
                except _facade("NestedCheckpointPaused"):
                    raise
                except Exception as exc:
                    _record_numerical_failure(outcome, name, exc)
                    host_logger("trex").warning(
                        "TREX importance refinement of %s failed: %s: %s",
                        name, type(exc).__name__, exc, exc_info=exc,
                        extra={"event": "trex.nested.failed", "scenario": name},
                    )
                    _record_unmet(outcome, name, "{0}; importance refinement failed: {1}: {2}".format(
                        failure, type(exc).__name__, exc
                    ))
                    continue
                refusal = _facade("_importance_refusal")(refined, mean_evidence, ensemble_sem)
                if refusal is not None:
                    _record_unmet(outcome, name, "{0}; {1}".format(failure, refusal))
                    continue
                acceptance = "importance-refined"
            trial_log_evidence = (
                refined.log_evidence if refined is not None else mean_evidence
            ) + float(plan["log_prior"])
            if not math.isfinite(trial_log_evidence):
                _record_unmet(outcome, name,
                    "{0} nested ensemble evidence with its scenario prior is non-finite".format(
                        name
                    )
                )
                continue

        # Commit the substitution only after every acceptance check succeeds.
        # Rejected trials never modify the evidence vector or its MC ESS entry.
        substituted[index] = trial_log_evidence
        remaining.pop(index, None)
        outcome["escalated"].append(index)
        calibrated = None
        if first.sampler == _facade("_BLACKJAX_SAMPLER_NAME"):
            calibrated = getattr(first, "configuration_calibrated", None)
            if calibrated is None:
                calibrated = (
                    int(first.n_live_points) == int(_facade("_BLACKJAX_CALIBRATED_LIVE_POINTS"))
                    and first_cap == _facade("_blackjax_likelihood_call_cap")(
                        int(first.n_live_points), _facade("_nested_dimensions")(plan, blocks)
                    )
                )
            calibrated = bool(calibrated)
        selected_light = plan.get("stellar_light") or stellar_light
        total_calls = sum(int(run.n_likelihood_calls) for run in nested_runs)
        total_monitored = sum(int(run.monitored_likelihood_calls) for run in nested_runs)
        if refined is not None:
            total_calls += int(refined.n_likelihood_calls)
            total_monitored += int(refined.n_likelihood_calls)
        total_iterations = sum(int(run.n_iterations) for run in nested_runs)
        outcome["records"][name] = {
            **({"stellar_light": selected_light.model_configuration()} if selected_light is not None else {}),
            "scenario": name,
            "index": index,
            "kind": plan["kind"],
            "use_2xP": bool(plan["use_2xP"]),
            "proposal_support": (
                dict(support_state[index])
                if isinstance(support_state.get(index), Mapping)
                else None
            ),
            "estimator": "nested-sampling" if refined is None else refined.estimator,
            "acceptance": acceptance,
            "log_evidence": float(
                refined.log_evidence
                if refined is not None
                else mean_evidence if acceptance == "ensemble" else first.log_evidence
            ),
            "importance_sampling": (
                None
                if refined is None
                else {
                    "log_evidence": float(refined.log_evidence),
                    "log_evidence_error_nats": float(refined.log_evidence_error),
                    "n_draws": int(refined.n_draws),
                    "n_likelihood_calls": int(refined.n_likelihood_calls),
                    "effective_sample_size": float(refined.effective_sample_size),
                    "maximum_weight_share": float(refined.maximum_weight_share),
                    "n_centres": int(refined.n_centres),
                    "defensive_fraction": float(refined.defensive_fraction),
                    "random_seed": int(refined.random_seed),
                    "execution_provenance": getattr(refined, "execution_provenance", None),
                    "agreement_standard_errors": _facade("_IMPORTANCE_AGREEMENT_SIGMA"),
                }
            ),
            "log_evidence_values": [float(value) for value in evidence_values],
            "ensemble_standard_error_nats": (
                None if not math.isfinite(ensemble_sem) else float(ensemble_sem)
            ),
            "substituted_probability": None,
            "single_run_probability_upper_bound": (
                single_run_probability_bound
                if acceptance == "single-run-below-bound"
                else None
            ),
            "seeds": [
                int(seeds[index * _facade("_ENSEMBLE_RUNS") + k])
                for k in range(len(evidence_values))
            ],
            "n_live_points": int(first.n_live_points),
            "max_likelihood_calls": first_cap,
            # None marks a non-BlackJAX estimator; this marker describes that route.
            "calibrated_configuration": calibrated,
            "n_likelihood_calls": total_calls,
            "n_iterations": total_iterations,
            "sampler": first.sampler,
            "sampler_version": first.sampler_version,
            "reported_log_evidence_error_nats": float(
                refined.log_evidence_error if refined is not None
                else ensemble_sem if acceptance == "ensemble" else first.log_evidence_error),
            "runs": [
                {
                    "random_seed": int(run.random_seed),
                    "log_evidence": float(run.log_evidence),
                    "reported_log_evidence_error_nats": float(run.log_evidence_error),
                    "n_live_points": int(run.n_live_points),
                    "n_likelihood_calls": int(run.n_likelihood_calls),
                    "n_iterations": int(run.n_iterations),
                    "sampler": run.sampler,
                    "sampler_version": run.sampler_version,
                    "stopped_short": bool(run.stopped_short),
                    "monitored_likelihood_calls": int(run.monitored_likelihood_calls),
                    "budget_exhausted": bool(run.budget_exhausted),
                    "stopped_on_evidence_tolerance": bool(
                        run.stopped_on_evidence_tolerance
                    ),
                    "stop_reason": str(run.stop_reason),
                    "base_run_delta_logz": (
                        float(run.base_run_delta_logz)
                        if math.isfinite(run.base_run_delta_logz)
                        else None
                    ),
                }
                for run in nested_runs
            ],
            "total_n_likelihood_calls": total_calls,
            "total_monitored_likelihood_calls": total_monitored,
            "total_n_iterations": total_iterations,
            "wall_seconds": _time.monotonic() - started,
        }
        post_probs, _ = _facade("_normalize_probabilities")(_facade("_period_weighted_evidences")(substituted, period_days))
        if _facade("unconverged_probability_mass")(
            post_probs, remaining, substituted
        ) <= _facade("MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY"):
            could_flip = _facade("residual_could_flip_decision")(substituted, remaining, period_days=period_days)
            if not could_flip and not outcome["unmet_scenarios"]:
                break

    final_probs, _ = _facade("_normalize_probabilities")(_facade("_period_weighted_evidences")(substituted, period_days))
    for record in outcome["records"].values():
        record["substituted_probability"] = float(final_probs[record["index"]])
    outcome["cumulative_wall_seconds"] = float(
        sum(float(cost.get("wall_seconds", 0.0)) for cost in outcome["attempt_costs"])
    )
    outcome["lnZ"] = substituted
    outcome["remaining_ess"] = remaining
    if _facade("unconverged_probability_mass")(
        final_probs, remaining, substituted
    ) > _facade("MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY") and outcome["unmet_reason"] is None:
        outcome["unmet_reason"] = (
            "the unconverged remainder still exceeds the {0:g} allowance "
            "after nested escalation".format(_facade("MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY"))
        )
    return outcome
