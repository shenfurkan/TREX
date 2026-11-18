"""Scenario calls and seeds for evidence escalation."""

from __future__ import annotations

from trex.runtime.logging import host_logger


import math
import json
import os
import time as _time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from trex.foundation.errors import NumericalLikelihoodError
from trex.evidence.host_measure import HostMeasure
from trex.evidence.nested_controls import _validate_scalar



def _facade(name):
    from trex.evidence import evidence_escalation

    return getattr(evidence_escalation, name)

def _bounded_escalation(function):
    @wraps(function)
    def run(**kwargs):
        wall = _validate_scalar(kwargs.get("wall_time_seconds", _facade("_DEFAULT_ESCALATION_WALL_SECONDS")),
                                "escalation wall_time_seconds", positive=True)
        deadline = _time.monotonic() + min(wall, _facade("_DEFAULT_ESCALATION_WALL_SECONDS"))
        requested = kwargs.get("deadline_monotonic")
        if requested is not None:
            deadline = min(deadline, _validate_scalar(requested, "escalation deadline", positive=False))
        directory = kwargs.get("checkpoint_directory")
        if not directory:
            raise ValueError("escalation requires a trusted checkpoint_directory")
        directory = Path(directory)
        costs = []
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        journal = directory / "attempts.jsonl"
        if journal.exists():
            try:
                costs = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError("escalation attempt journal is unreadable or corrupt") from exc
            if any(not isinstance(entry, dict) for entry in costs):
                raise ValueError("escalation attempt journal must contain objects")
        token = _facade("_escalation_runtime").set((directory, deadline, costs))
        try:
            return function(**kwargs)
        except _facade("NestedCheckpointPaused") as exc:
            exc.attempt_costs = list(costs)
            raise
        finally:
            _facade("_escalation_runtime").reset(token)
    return run

def _scenario_seeds(random_seed: Optional[int], count: int) -> Tuple[List[int], Optional[int]]:
    """Per-scenario integer seeds derived from one run seed.

    Spawning keeps a scenario's seed independent of the draw budget and of
    which scenarios escalated, so a repeat run reproduces the same nested
    estimates. A missing run seed draws fresh entropy and records it.
    """
    if random_seed is None:
        root = np.random.SeedSequence()
        recorded = int(root.entropy)
    else:
        root = np.random.SeedSequence(int(random_seed))
        recorded = int(random_seed)
    children = root.spawn(count * _facade("_ENSEMBLE_RUNS"))
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
    return seeds, recorded


def _importance_seed(recorded_seed: int, count: int, index: int) -> int:
    """The importance-refinement seed of one scenario.

    Spawned after every nested child (``SeedSequence`` children depend only on
    the root and their own spawn index), so adding it leaves each nested seed
    of ``_scenario_seeds`` unchanged.
    """
    child = np.random.SeedSequence(int(recorded_seed), spawn_key=(count * _facade("_ENSEMBLE_RUNS") + index,))
    return int(child.generate_state(1, dtype=np.uint32)[0])


def _failing_indices(lnZ: np.ndarray, ess: Mapping[int, float]) -> List[int]:
    """Evaluated scenarios below the ESS floor, including zero sampled weight."""
    failing = []
    for index, value in ess.items():
        if index < 0 or index >= len(lnZ):
            continue
        if np.isneginf(lnZ[index]) or not math.isfinite(value) or value < _facade("MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE"):
            failing.append(int(index))
    return failing


def _is_support_unreachable(
    support_state: Mapping[int, Mapping[str, Any]], index: int
) -> bool:
    """Whether this scenario's proposal window could not reach its prior support."""
    state = support_state.get(index)
    if not isinstance(state, Mapping):
        return False
    return str(state.get("state")) in _facade("_SUPPORT_UNREACHABLE_STATES")


def _scenario_name(index: int) -> str:
    return _facade("_SCENARIO_NAMES").get(index, "N{0}".format(index))


def _target_plan_entry(index: int) -> Optional[dict]:
    """Nested call shape for one target-hosted scenario index, or ``None``."""
    if index >= _facade("N_TARGET_SCENARIOS"):
        return None
    _, is_eb, use_2xP, companion_is_host, _, use_flat_priors, _, bound, background = (
        _facade("TARGET_SCENARIO_SPECS")[index]
    )
    hosted = None
    # Background-hosted rows also set companion_is_host, so test them first.
    if background == "background":
        hosted = "background"
    elif companion_is_host:
        hosted = "bound"
    population = None
    if bound == "primary":
        population = "bound"
    elif background == "diluted":
        population = "background"
    return {
        "index": index,
        "kind": "eb" if is_eb else "tp",
        "use_2xP": bool(use_2xP),
        "use_flat_priors": bool(use_flat_priors),
        "population": population,
        "hosted": hosted,
        "log_prior": 0.0,
    }


def _neighbor_plan_entry(index, neighbors, target_feh, observed_depth_fraction, stellar_light=None,
                         joint_observations=None):
    """Resolve the same source-owned host for MC-pilot and nested refinement."""
    neighbor_index, local = divmod(index - _facade("N_TARGET_SCENARIOS"), _facade("N_NEIGHBOR_SCENARIOS"))
    if neighbors is None or not 0 <= neighbor_index < len(neighbors):
        return None
    host = _facade("resolved_neighbor_host")(
        neighbors[neighbor_index], target_feh, neighbor_index, observed_depth_fraction,
        **({"stellar_light": stellar_light} if stellar_light is not None else {}),
        **({"joint_observations": joint_observations} if joint_observations is not None else {}),
    )
    if host is None:
        return None
    return {
        "index": index, "kind": "tp" if local == 0 else "eb", "use_2xP": local == 2,
        "use_flat_priors": False, "population": None, "hosted": None, "log_prior": host["log_prior"],
        "host_mass": host["M_s"], "host_radius": host["R_s"], "host_u1": host["u1"], "host_u2": host["u2"],
        "target_feh": host["target_feh"], "aperture_flux_fraction": host["aperture_flux_fraction"],
        **({"source_id": str(host["source_id"])} if joint_observations is not None else {}),
        **({"host_distribution": host["stellar_parameter_distribution"]}
           if "stellar_parameter_distribution" in host else {}),
        **({key: host[key] for key in ("stellar_light", "target_teff_k")} if stellar_light is not None else {}),
    }


def _population_blocks(
    *,
    M_s_Msun: float,
    R_s_Rsun: float,
    plx: Optional[float],
    contrast_separations: Optional[np.ndarray],
    contrast_values: Optional[np.ndarray],
    resolution_limit_arcsec: Optional[float],
    background_population: Optional[Mapping[str, np.ndarray]],
    background_area_sq_deg: Optional[float],
    target_tess_mag: float,
    target_teff_k: Optional[float] = None,
    stellar_light=None,
    target_feh=None,
    joint_observations=None,
) -> Dict[str, Any]:
    """Build the two unresolved-population prior blocks once per escalation."""
    blocks: Dict[str, Any] = {"bound": None, "background": None}
    if plx is not None and math.isfinite(plx) and plx > 0.0:
        blocks["bound"] = _facade("bound_companion_population")(
            M_s_Msun,
            R_s_Rsun,
            plx,
            contrast_separations=contrast_separations,
            contrast_values=contrast_values,
            resolution_limit_arcsec=resolution_limit_arcsec,
            **({"host_teff_k": target_teff_k} if target_teff_k is not None else {}),
            **({"stellar_light": stellar_light, "target_feh": target_feh} if stellar_light is not None else {}),
            **({"datasets": joint_observations.population_datasets, "band_light": joint_observations.band_light}
               if joint_observations is not None else {}),
        )
    if background_population is not None and background_area_sq_deg is not None:
        from trex.populations.background import declared_background_selection

        datasets = () if joint_observations is None else joint_observations.population_datasets
        # The scene's declared band and depth decide availability; the first
        # dataset's band is never an implicit fallback, and a native scene does
        # not need a finite TESS magnitude. Several undeclared bands refuse.
        declared = declared_background_selection(
            background_population, datasets=datasets, target_tess_mag=target_tess_mag)
        if declared.depth_mag is not None and math.isfinite(declared.depth_mag):
            blocks["background"] = _facade("background_star_population")(
                background_population,
                background_area_sq_deg,
                target_tess_mag,
                contrast_separations=contrast_separations,
                contrast_values=contrast_values,
                resolution_limit_arcsec=resolution_limit_arcsec,
                **({"datasets": datasets} if joint_observations is not None else {}),
                selection_band=declared.band,
                selection_magnitude=declared.depth_mag,
                completeness=declared.completeness,
            )
    return blocks


def _nested_dimensions(plan: Mapping[str, Any], blocks: Mapping[str, Any]) -> int:
    """Unit-cube dimension consumed by the scenario's nested integrand."""
    population = None
    if plan.get("hosted") is not None:
        population = blocks.get(plan["hosted"])
    elif plan.get("population") is not None:
        population = blocks.get(plan["population"])
    dimensions = int(_facade("_HOSTED_N_DIMENSIONS"))
    if population is not None:
        dimensions += int(population.n_dimensions)
    distribution = plan.get("host_distribution")
    if distribution is not None:
        dimensions += HostMeasure(distribution, plan.get("target_feh")).n_dimensions
    return dimensions


def _nested_call(
    plan: Mapping[str, Any],
    *,
    blocks: Mapping[str, Any],
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    exptime_days: float,
    bin_integration_days: float,
    target_aperture_flux_fraction: float,
    target_feh: Optional[float],
    nsamples: int,
    random_seed: int,
    n_jobs: Optional[int],
    importance_posterior: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    target_teff_k: Optional[float] = None,
    n_live_points: Optional[int] = None,
    baseline=None,
    stellar_light=None,
    joint_observations=None,
) -> Any:
    """Run one nested scenario estimate for a planned slot.

    With ``importance_posterior`` the same slot is integrated by the importance
    refinement over those nested posteriors instead. BlackJAX defaults to the
    calibrated live count and derives its call ceiling from that count; an
    explicit lower live count is refused rather than silently run uncalibrated.
    """
    estimator = _facade("_selected_nested_estimator")()
    common = dict(
        time=time,
        flux=flux,
        sigma=sigma,
        period_days=period_days,
        exptime_days=exptime_days,
        bin_integration_days=bin_integration_days,
        nsamples=nsamples,
        random_seed=random_seed,
        n_jobs=n_jobs,
        evidence_tolerance=_facade("_DEFAULT_EVIDENCE_TOLERANCE"),
        max_likelihood_calls=_facade("_MAXIMUM_NESTED_LIKELIHOOD_CALLS"),
        # Preserve the retained scene metallicity in diluted TP slots too.
        # Native factories resolve atmosphere coefficients only where required;
        # dropping an available value here loses the owning scene's context.
        target_feh=target_feh,
        **({"baseline": baseline} if baseline is not None else {}),
        **({"stellar_light": stellar_light} if stellar_light is not None else {}),
        **({"joint_observations": joint_observations} if joint_observations is not None else {}),
    )
    if estimator == "blackjax" and importance_posterior is None:
        live = _facade("_blackjax_live_points")(n_live_points)
        common["n_live_points"] = live
        common["max_likelihood_calls"] = _facade("_blackjax_likelihood_call_cap")(
            live, _nested_dimensions(plan, blocks)
        )
    elif n_live_points is not None:
        common["n_live_points"] = n_live_points
    if "host_mass" in plan:
        common.update(
            M_s_Msun=plan["host_mass"],
            R_s_Rsun=plan["host_radius"],
            u1=plan["host_u1"],
            u2=plan["host_u2"],
            target_aperture_flux_fraction=plan["aperture_flux_fraction"],
            target_feh=plan.get("target_feh", target_feh),
        )
        distribution = plan.get("host_distribution")
        if distribution is not None:
            common["host_distribution"] = distribution
            common["host_teff_k"] = HostMeasure(distribution, common["target_feh"]).marginal_median("teff_k")
        if stellar_light is not None:
            if plan.get("stellar_light") is None or plan.get("target_feh") is None or plan.get("target_teff_k") is None:
                raise _facade("NestedEvidenceUnavailableError")("resolved host is missing its stellar-light atmosphere")
            # A resolved light model uses this host's measured atmosphere, not the target's.
            common.update(stellar_light=plan["stellar_light"], target_feh=plan["target_feh"], host_teff_k=plan["target_teff_k"])
        if joint_observations is not None:
            common["joint_observations"] = joint_observations.for_host(
                plan["source_id"], av_mag=plan["stellar_light"].target_av_mag)
    else:
        common.update(
            M_s_Msun=M_s_Msun,
            R_s_Rsun=R_s_Rsun,
            u1=u1,
            u2=u2,
            target_aperture_flux_fraction=target_aperture_flux_fraction,
        )
    if plan["hosted"] is not None:
        block = blocks.get(plan["hosted"])
        if block is None:
            raise _facade("NestedEvidenceUnavailableError")(
                "the {0} population needed by the {1} scenario is unavailable".format(
                    plan["hosted"], _facade("_scenario_name")(plan["index"])
                )
            )
        common["hosted_population"] = block
    else:
        block = blocks.get(plan["population"]) if plan["population"] else None
        if plan["population"] is not None and block is None:
            raise _facade("NestedEvidenceUnavailableError")(
                "the {0} population needed by the {1} scenario is unavailable".format(
                    plan["population"], _facade("_scenario_name")(plan["index"])
                )
            )
        common["population"] = block
    if importance_posterior is not None:
        common["importance_posterior"] = importance_posterior
    if (plan["kind"] == "eb" or stellar_light is not None) and "host_mass" not in plan and target_teff_k is not None:
        common["host_teff_k"] = target_teff_k
    runtime = _facade("_escalation_runtime").get()
    directory, deadline, costs = runtime if runtime is not None else (None, float("inf"), [])
    if _time.monotonic() >= deadline:
        raise _facade("NestedCheckpointPaused")("escalation wall deadline reached; retained runs are resumable")
    cost = {"scenario": _facade("_scenario_name")(plan["index"]), "random_seed": int(random_seed),
            "backend": plan.get("estimator", "importance-refinement") if importance_posterior is not None else
                       "dynesty-scalar-cpu" if estimator == "dynesty" else estimator,
            "status": "interrupted", "wall_seconds": 0.0, "reused": False}
    started = _time.monotonic()
    prefix = None if directory is None else directory / "{0}-{1}-{2}".format(
        _facade("_scenario_name")(plan["index"]), random_seed, "importance" if importance_posterior is not None else "nested")
    # CPU allocation is operational. A restored sampler retains its proposal
    # queue/RNG; a completed result can be reused on a differently sized host.
    identity_inputs = (dict(plan), {k: v for k, v in common.items() if k != "n_jobs"}, estimator)
    if importance_posterior is not None:
        proposal = _facade("_selected_importance_proposal")()
        if proposal != "kernel-mixture":
            identity_inputs += (proposal, "unclipped-ndtr-v1")
    elif estimator == "blackjax":
        identity_inputs += _facade("_native_kernel_policy_identity")()
    identity = _facade("_checkpoint_identity")(identity_inputs) if prefix is not None else None
    for previous in costs:
        failure = previous.get("numerical_failure")
        if identity is not None and previous.get("input_identity") == identity and failure:
            raise NumericalLikelihoodError(
                "checkpoint retains unresolved numerical failure: " + failure["message"], failure["count"]
            )
    try:
        cached = None if prefix is None else _facade("_load_completed")(prefix.with_suffix(".result.npz"), identity)
        if cached is not None:
            result = cached
            cost["reused"] = True
        else:
            with _facade("nested_checkpoint")(prefix, deadline=deadline):
                if plan["kind"] == "tp":
                    result = _facade("tp_scenario_nested_evidence")(
                        use_flat_priors=bool(plan["use_flat_priors"]), **common
                    )
                else:
                    result = _facade("eb_scenario_nested_evidence")(use_2xP=plan["use_2xP"], **common)
            if prefix is not None:
                _facade("_save_completed")(prefix.with_suffix(".result.npz"), identity, result)
        cost["status"] = "completed"
        cost["n_likelihood_calls"] = int(result.n_likelihood_calls)
    except _facade("NestedCheckpointPaused"):
        cost["status"] = "paused"
        raise
    except Exception as exc:
        cost["status"] = "failed"
        count = exc.count if isinstance(exc, NumericalLikelihoodError) else getattr(exc, "numerical_failure_count", 0)
        if count:
            cost["input_identity"] = identity
            cost["numerical_failure"] = {
                "count": int(count), "message": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
        raise
    finally:
        cost["wall_seconds"] = _time.monotonic() - started
        costs.append(cost)
        if directory is not None:
            temporary = directory / "attempts.jsonl.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                for entry in costs:
                    stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "attempts.jsonl")
    wall_seconds = _time.monotonic() - started
    name = _facade("_scenario_name")(plan["index"])
    stop = result.stop_reason if importance_posterior is None else result.estimator
    host_logger("trex").info(
        "TREX nested %s: seed=%d, calls=%d, lnZ=%r, stop=%s, %.0f s",
        name, random_seed, result.n_likelihood_calls, result.log_evidence, stop, wall_seconds,
        extra={"event": "trex.nested", "scenario": name, "random_seed": int(random_seed),
               "n_likelihood_calls": int(result.n_likelihood_calls),
               "ln_evidence": float(result.log_evidence), "stop_reason": str(stop),
               "wall_seconds": wall_seconds},
    )
    return result
