"""MC-pilot defensive importance, before the existing nested escalation."""
from contextvars import ContextVar
import hashlib
import json
import math
import time

import numpy as np

from trex.evidence import evidence_escalation as escalation
from trex.evidence import nested_controls as controls
from trex.evidence.evidence_escalation_calls import _neighbor_plan_entry
from trex.evidence.nested_checkpoint import CheckpointInputMismatch, NestedCheckpointPaused, checkpoint_identity


VERSION = "mc-pilot-defensive-v3"
ESTIMATOR = "mc-pilot-defensive-importance"
pilot_seed = ContextVar("trex_mc_importance_pilot_seed", default=None)
native_stage = ContextVar("trex_mc_importance_native_stage", default=None)


def stage_controls():
    return {
        "version": VERSION,
        "draws": controls._IMPORTANCE_DRAWS,
        "centres": controls._IMPORTANCE_CENTRES,
        "defensive_fraction": controls._IMPORTANCE_DEFENSIVE_FRACTION,
        "density_chunk": controls._IMPORTANCE_DENSITY_CHUNK,
        "evaluation_chunk": controls._IMPORTANCE_EVALUATION_CHUNK,
        "pilot_ess": controls._IMPORTANCE_PILOT_ESS,
        "adaptation_draws": controls._IMPORTANCE_ADAPTATION_DRAWS,
        "adaptation_rounds": controls._IMPORTANCE_ADAPTATION_ROUNDS,
        "adaptation_ess": controls._IMPORTANCE_ADAPTATION_ESS,
        "ess_floor": escalation.MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE,
        "maximum_error_nats": escalation._MAXIMUM_ENSEMBLE_STANDARD_ERROR_NATS,
    }


def tempering_exponent(log_weights, target_ess):
    """Largest weight power in [0, 1] reaching the requested Kish ESS.

    An infeasible target (more than the available draws) leaves the weights
    unchanged, rather than erasing their information with a uniform pilot.
    """
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("tempering requires a nonempty vector of finite log weights")
    target_ess = controls._validate_scalar(target_ess, "tempering target ESS", positive=True)
    if target_ess > values.size:
        return 1.0
    shifted = values - np.max(values)

    def kish(power):
        weights = np.exp(power * shifted)
        return weights.sum() ** 2 / np.sum(weights ** 2)

    if kish(1.0) >= target_ess:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if kish(middle) >= target_ess else (low, middle)
    return low


def make_centres(logw, occultor, eccentricity, argument, inclination, band_low,
                 population_variates, background_population, *, seed, scenario, host_quantile=None):
    """Resample complete MC evidence terms into the nested unit-cube coordinates."""
    logw = np.asarray(logw, dtype=float)
    finite = np.flatnonzero(np.isfinite(logw))
    if finite.size == 0:
        return None
    # At pilot ESS near 1 the plain weights name one draw, which leaves a singular kernel covariance.
    power = tempering_exponent(logw[finite], controls._IMPORTANCE_PILOT_ESS)
    weights = np.exp(power * (logw[finite] - np.max(logw[finite])))
    weights /= weights.sum()
    tag = int.from_bytes(hashlib.sha256(repr(scenario).encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), tag, 0]))
    positions = (rng.random() + np.arange(controls._IMPORTANCE_CENTRES)) / controls._IMPORTANCE_CENTRES
    selected = finite[np.minimum(np.searchsorted(np.cumsum(weights), positions), finite.size - 1)]
    # Invert nested_integrands' transit-conditioned inclination transform; all values are unitless.
    band = np.asarray(band_low)[selected]
    if not np.isfinite(band).all() or np.any(band < 0.0) or np.any(band >= 1.0):
        raise ValueError("MC importance pilot requires a positive-width inclination band")
    unit_inc = (inclination[selected] - band) / (1.0 - band)
    columns = [occultor[selected], eccentricity[selected], argument[selected], unit_inc]
    if host_quantile is not None:
        extra = np.asarray(host_quantile)[selected]
        columns.extend(extra.T if extra.ndim == 2 else [extra])
    if population_variates:
        first, *remaining = population_variates[0]
        if background_population is not None:
            from trex.populations.background import (
                declared_background_selection,
                select_background_population,
            )

            declared = declared_background_selection(background_population)
            if declared.depth_mag is None and declared.band is None:
                # Bare legacy TESS table whose loader already applied its cut;
                # the pilot's raw row indices are ranks in that same table.
                tess = np.asarray(background_population.get("tess_mag"), dtype=float)
                if tess.ndim != 1 or not tess.size or not np.isfinite(tess).all():
                    raise ValueError(
                        "MC importance pilot requires a finite retained tess_mag for the legacy path")
                selection = None
            elif declared.depth_mag is None:
                raise ValueError(
                    "MC importance pilot needs the declared background selection depth")
            else:
                selection = select_background_population(
                    background_population, band=declared.band,
                    depth_mag=declared.depth_mag,
                    target_tess_mag=declared.depth_mag if declared.legacy_tess else None)
            if selection is None:
                order = np.argsort(tess, kind="stable")
                size = tess.size
            else:
                # Same pool and ordering as background_star_population's draw;
                # using the raw table's TESS order for a native band would
                # silently mis-map the unit coordinate.
                order = selection.selected_indices[
                    np.argsort(selection.magnitudes[selection.selected_indices], kind="stable")]
                size = selection.magnitudes.size
            rank = np.full(size, -1, dtype=int)
            rank[order] = np.arange(order.size)
            if np.any(rank[first[selected]] < 0):
                raise ValueError("MC importance pilot drew a star outside the declared background pool")
            # The midpoint selects the same row in the magnitude-sorted prior.
            first = (rank[first[selected]] + 0.5) / order.size
        else:
            first = first[selected]
        columns.append(first)
        columns.extend(value[selected] for value in remaining)
    centres = np.column_stack(columns)
    if not np.isfinite(centres).all() or np.any(centres < 0) or np.any(centres > 1):
        raise ValueError("MC importance pilot has invalid unit-cube coordinates")
    return centres


def make_state(seed, centres, input_identity):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("MC importance handoff seed must be a non-negative integer")
    if not isinstance(centres, dict):
        raise ValueError("MC importance handoff centres must be a mapping")
    normalized = {}
    for key, value in centres.items():
        if (isinstance(key, (bool, np.bool_)) or not isinstance(key, (str, int, np.integer))
                or str(int(key)) != str(key) or int(key) < 0):
            raise ValueError("MC importance scenario index must be a non-negative integer")
        name = str(key)
        if name in normalized:
            raise ValueError("MC importance handoff has duplicate scenario indices")
        normalized[name] = np.asarray(value, dtype=float).tolist()
    payload = {"seed": int(seed), "centres": normalized, "controls": stage_controls()}
    return {**payload, "identity": checkpoint_identity((input_identity, json.dumps(payload, sort_keys=True, allow_nan=False)))}


def validate_state(state, input_identity, scenario_count, target_mass=None):
    if not isinstance(state, dict) or state.get("controls") != stage_controls():
        raise ValueError("MC importance handoff controls or version changed")
    normalized = make_state(state.get("seed"), state.get("centres"), input_identity)
    if state.get("identity") != normalized["identity"]:
        raise ValueError("MC importance handoff inputs, centres or seed changed")
    for key, value in normalized["centres"].items():
        try:
            index = int(key)
            points = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("MC importance centres are malformed") from exc
        if (str(index) != key or not 0 <= index < scenario_count or points.ndim != 2
                or points.shape[0] != controls._IMPORTANCE_CENTRES or points.shape[1] not in (4, 5, 6, 7)
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1)):
            raise ValueError("MC importance centres are outside the scenario unit cube")
        if target_mass is not None and index < escalation.N_TARGET_SCENARIOS:
            plan = escalation._target_plan_entry(index)
            population = plan["hosted"] or plan["population"]
            expected = 4 + (3 if population == "bound" else 2 if population else 0)
            if points.shape[1] != expected:
                raise ValueError(f"MC importance scenario {index} requires {expected} unit dimensions, got {points.shape[1]}")
    return normalized


@escalation._bounded_escalation
def run_mc_importance(*, lnZ, ess, state, n_draws, input_identity, n_jobs=None,
                      checkpoint_directory=None, wall_time_seconds=escalation._DEFAULT_ESCALATION_WALL_SECONDS,
                      deadline_monotonic=None, **inputs):
    """Refine every finite sub-floor MC scenario; failures retain their MC entry.

    The estimator is the same on both hosts: an admitted accelerator evaluates
    the scenario's JAX unit-cube callback, and a CPU host runs that same
    callback as a batched float64 JAX kernel, or the declared float64 scalar
    integrand where no native callback exists. Only likelihood execution
    changes; proposal draws, densities, RNG, weights and reductions are shared.
    """
    outcome = {"lnZ": np.array(lnZ, dtype=float), "ess": dict(ess), "records": {}, "attempts": {}}
    failures = escalation._failing_indices(lnZ, ess)
    if not failures or n_draws < controls._IMPORTANCE_DRAWS:
        return outcome
    if state is None:
        outcome["attempts"] = {escalation._scenario_name(i): {"status": "unavailable", "reason": "no MC pilot in handoff"} for i in failures}
        return outcome
    state = validate_state(state, input_identity, len(lnZ), inputs["M_s_Msun"])
    try:
        blocks = escalation._population_blocks(
            M_s_Msun=inputs["M_s_Msun"], R_s_Rsun=inputs["R_s_Rsun"], plx=inputs["plx"],
            contrast_separations=inputs["contrast_separations"], contrast_values=inputs["contrast_values"],
            resolution_limit_arcsec=inputs["resolution_limit_arcsec"], background_population=inputs["background_population"],
            background_area_sq_deg=inputs["background_area_sq_deg"], target_tess_mag=inputs["target_tess_mag"],
            target_teff_k=inputs.get("target_teff_k"),
            **({"stellar_light": inputs["stellar_light"], "target_feh": inputs["target_feh"]} if inputs.get("stellar_light") is not None else {}),
            **({"joint_observations": inputs["joint_observations"]} if inputs.get("joint_observations") is not None else {}),
        )
    except Exception as exc:
        outcome["attempts"] = {escalation._scenario_name(i): {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"} for i in failures}
        return outcome
    # Descending probability, as the nested router: stop once the rest is outside the margin
    # and cannot gate or flip the run.
    period_days = inputs["period_days"]
    weighted = escalation._period_weighted_evidences(lnZ, period_days)
    ordered = sorted(failures, key=lambda index: float(weighted[index]), reverse=True)
    required = set(escalation.stage_scenarios(lnZ, ess, period_days=period_days))
    probs, _ = escalation._normalize_probabilities(weighted)
    for position, index in enumerate(ordered):
        if (index not in required
                and sum(float(probs[key]) for key in ordered[position:]) <= escalation.MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY
                and not escalation.residual_could_flip_decision(outcome["lnZ"], outcome["ess"], period_days=period_days)):
            break
        name = escalation._scenario_name(index)
        points = state["centres"].get(str(index))
        if points is None:
            outcome["attempts"][name] = {"status": "unavailable", "reason": "no finite MC pilot centres"}
            continue
        if index < escalation.N_TARGET_SCENARIOS:
            plan = escalation._target_plan_entry(index)
        else:
            plan = _neighbor_plan_entry(
                index, inputs.get("nearby_stars"), inputs["target_feh"],
                inputs["depth_ppm"] * escalation.FRACTION_PER_PART_PER_MILLION,
                inputs.get("stellar_light"),
                **({"joint_observations": inputs["joint_observations"]} if inputs.get("joint_observations") is not None else {}),
            )
        if plan is None:
            outcome["attempts"][name] = {"status": "unavailable", "reason": "scenario plan unavailable"}
            continue
        seed = int(np.random.SeedSequence([state["seed"], index, 1]).generate_state(1, dtype=np.uint32)[0])
        plan["mc_importance_identity"] = state["identity"]
        plan["mc_importance_controls"] = stage_controls()
        plan["estimator"] = ESTIMATOR
        started = time.monotonic()
        token = native_stage.set((VERSION, state["identity"], stage_controls()))
        try:
            result = escalation._nested_call(
                plan, blocks=blocks, time=inputs["time"], flux=inputs["flux"], sigma=inputs["sigma"],
                period_days=inputs["period_days"], M_s_Msun=inputs["M_s_Msun"], R_s_Rsun=inputs["R_s_Rsun"],
                u1=inputs["u1"], u2=inputs["u2"], exptime_days=inputs["exptime_days"],
                bin_integration_days=inputs["bin_integration_days"],
                target_aperture_flux_fraction=inputs["target_aperture_flux_fraction"],
                target_feh=inputs["target_feh"], target_teff_k=inputs.get("target_teff_k"),
                nsamples=inputs["nsamples"], random_seed=seed, n_jobs=n_jobs,
                importance_posterior=[(np.asarray(points), np.full(len(points), 1.0 / len(points)))],
                **({"baseline": inputs["baseline"]} if inputs.get("baseline") is not None else {}),
                **({"stellar_light": inputs["stellar_light"]} if inputs.get("stellar_light") is not None else {}),
                **({"joint_observations": inputs["joint_observations"]} if inputs.get("joint_observations") is not None else {}),
            )
        except (NestedCheckpointPaused, CheckpointInputMismatch):
            raise
        except Exception as exc:
            from trex.foundation.errors import NumericalLikelihoodError

            outcome["attempts"][name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}",
                                          "random_seed": seed, "wall_seconds": time.monotonic() - started,
                                          **({"unresolved_numerical_failures": exc.count}
                                             if isinstance(exc, NumericalLikelihoodError) else {})}
            continue
        finally:
            native_stage.reset(token)
        record = {
            "scenario": name, "index": index, "estimator": ESTIMATOR, "random_seed": seed,
            "log_evidence": float(result.log_evidence + plan["log_prior"]) if math.isfinite(result.log_evidence + plan["log_prior"]) else None,
            "log_evidence_error_nats": float(result.log_evidence_error) if math.isfinite(result.log_evidence_error) else None,
            "effective_sample_size": float(result.effective_sample_size) if math.isfinite(result.effective_sample_size) else None,
            "n_draws": int(result.n_draws),
            "n_likelihood_calls": int(result.n_likelihood_calls), "n_centres": int(result.n_centres),
            "defensive_fraction": float(result.defensive_fraction), "maximum_weight_share": float(result.maximum_weight_share),
            "execution_provenance": result.execution_provenance, "wall_seconds": time.monotonic() - started,
            "execution_backend": (result.execution_provenance or {}).get("backend"),
        }
        if record["log_evidence"] is None or result.n_draws != controls._IMPORTANCE_DRAWS:
            reason = "MC-pilot importance did not return finite evidence at the registered budget"
        else:
            reason = escalation._mc_importance_refusal(result, record["log_evidence"], float(lnZ[index]), ess[index])
        record.update(status="accepted" if reason is None else "unresolved", reason=reason)
        outcome["attempts"][name] = record
        if reason is None:
            outcome["lnZ"][index] = record["log_evidence"]
            outcome["ess"][index] = record["effective_sample_size"]
            outcome["records"][name] = record
            probs, _ = escalation._normalize_probabilities(
                escalation._period_weighted_evidences(outcome["lnZ"], period_days)
            )
    return outcome
