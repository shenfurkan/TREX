"""Checked Monte Carlo checkpoint serialization for escalation."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Optional

import numpy as np

from trex.stellar_light.host_limb_darkening import MC_LDC_ROLE as MC_LDC_ROLE

MC_LDC_UNAVAILABLE = "provenance unavailable (state written before limb-darkening provenance was recorded)"


def _summarize_mc_ldc_events(events: list) -> dict:
    """Validate and copy the existing per-call summaries, never per-draw arrays."""
    from trex.stellar_light.host_limb_darkening import TESS_LOGG_GRID_MAX_DEX, TESS_LOGG_GRID_MIN_DEX, TESS_LOGG_GRID_STEP_DEX, TESS_TEFF_GRID_MAX_K, TESS_TEFF_GRID_MIN_K, TESS_TEFF_GRID_STEP_K, load_triceratops_tess_grid

    if not isinstance(events, list):
        raise ValueError("Monte Carlo limb-darkening events must be a list")

    def number(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Monte Carlo limb-darkening provenance contains a non-finite number")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("Monte Carlo limb-darkening provenance contains a non-finite number") from exc
        if not math.isfinite(value):
            raise ValueError("Monte Carlo limb-darkening provenance contains a non-finite number")
        return value

    range_fields = ("requested_teff_k_min", "requested_teff_k_max",
                    "requested_logg_cgs_min", "requested_logg_cgs_max")
    fields = {"scenario", "clamped_draws", "interpolated_draws", "total_draws",
              "used_metallicity_dex", "used_nodes", *range_fields}
    split_fields = {"reference_draws", "atlas_draws", "phoenix_draws"}
    planes = {row["Z"] for row in load_triceratops_tess_grid().rows} if events else set()
    summaries = []
    for event in events:
        if not isinstance(event, Mapping) or set(event) not in (fields, fields | split_fields):
            raise ValueError("Monte Carlo limb-darkening event is malformed")
        scenario = event["scenario"]
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("Monte Carlo limb-darkening event lacks a scenario")
        counts = [event[name] for name in ("clamped_draws", "interpolated_draws", "total_draws")]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or value < 0 for value in counts):
            raise ValueError("Monte Carlo limb-darkening event has invalid counts")
        clamped, interpolated, total = (int(value) for value in counts)
        if not total or clamped + interpolated > total:
            raise ValueError("Monte Carlo limb-darkening event counts are inconsistent")
        split = {}
        if split_fields <= event.keys():
            if any(isinstance(event[name], (bool, np.bool_)) or not isinstance(event[name], (int, np.integer))
                   or event[name] < 0 for name in split_fields):
                raise ValueError("Monte Carlo limb-darkening source counts are invalid")
            split = {name: int(event[name]) for name in split_fields}
            if (split["atlas_draws"] + split["phoenix_draws"] != interpolated
                    or sum(split.values()) + clamped != total):
                raise ValueError("Monte Carlo limb-darkening source counts are inconsistent")
        ranges = {name: number(event[name]) for name in range_fields}
        if (ranges["requested_teff_k_min"] <= 0
                or ranges["requested_teff_k_min"] > ranges["requested_teff_k_max"]
                or ranges["requested_logg_cgs_min"] > ranges["requested_logg_cgs_max"]):
            raise ValueError("Monte Carlo limb-darkening event ranges are inconsistent")
        metals, nodes = event["used_metallicity_dex"], event["used_nodes"]
        if not isinstance(metals, list) or not metals or not isinstance(nodes, list):
            raise ValueError("Monte Carlo limb-darkening event nodes are malformed")
        if bool(nodes) != bool(clamped):
            raise ValueError("Monte Carlo limb-darkening clamp nodes are inconsistent")
        if any(not isinstance(node, (list, tuple)) or len(node) != 2 for node in nodes):
            raise ValueError("Monte Carlo limb-darkening event nodes are malformed")
        metals = [number(value) for value in metals]
        nodes = [[number(value) for value in node] for node in nodes]
        if (len(set(metals)) != len(metals) or len(metals) > total
                or any(value not in planes for value in metals)
                or len({tuple(node) for node in nodes}) != len(nodes) or len(nodes) > clamped
                or any(not (TESS_TEFF_GRID_MIN_K <= teff <= TESS_TEFF_GRID_MAX_K
                            and TESS_LOGG_GRID_MIN_DEX <= logg <= TESS_LOGG_GRID_MAX_DEX)
                       or not math.isclose(teff / TESS_TEFF_GRID_STEP_K, round(teff / TESS_TEFF_GRID_STEP_K), rel_tol=0.0, abs_tol=1e-9)
                       or not math.isclose(logg / TESS_LOGG_GRID_STEP_DEX, round(logg / TESS_LOGG_GRID_STEP_DEX), rel_tol=0.0, abs_tol=1e-9)
                       for teff, logg in nodes)):
            raise ValueError("Monte Carlo limb-darkening event nodes are inconsistent")
        summaries.append({
            "scenario": scenario, "clamped_draws": clamped, "interpolated_draws": interpolated,
            "total_draws": total, **ranges,
            **split,
            "used_metallicity_dex": metals, "used_nodes": nodes,
        })
    return {"status": "available", "events": summaries,
            "source_counts_complete": all(split_fields <= event.keys() for event in events)}


def _read_mc_ldc_provenance(state: Mapping) -> dict:
    """Missing old records remain unknown; a present malformed record refuses."""
    if "mc_ldc_provenance" not in state:
        return {"status": MC_LDC_UNAVAILABLE, "events": None}
    record = state["mc_ldc_provenance"]
    if (not isinstance(record, Mapping)
            or set(record) not in ({"status", "events"}, {"status", "events", "source_counts_complete"})
            or record["status"] != "available"):
        raise ValueError("escalation state Monte Carlo limb-darkening provenance is malformed")
    result = _summarize_mc_ldc_events(record["events"])
    complete = record.get("source_counts_complete", False)
    if not isinstance(complete, bool) or (complete and not result["source_counts_complete"]):
        raise ValueError("Monte Carlo limb-darkening source completeness is malformed")
    result["source_counts_complete"] = complete
    return result


def _publish_mc_ldc_provenance(events: list, record: Mapping) -> None:
    """Replace this run's MC entries on same-process resume; leave later events alone."""
    events[:] = [event for event in events
                 if event.get("role") != MC_LDC_ROLE and event.get("stage") != "monte-carlo"]
    events.append({"role": MC_LDC_ROLE, "status": record["status"],
                   "source_counts_complete": record.get("source_counts_complete", False)})
    if record["events"] is not None:
        events.extend({**event, "stage": "monte-carlo"} for event in record["events"])


def _portable_importance(outcome: Mapping[str, Any]) -> dict:
    """The MC-pilot outcome in checkpoint values."""
    return {
        "lnZ": [float(value) for value in outcome["lnZ"]],
        "ess": {str(int(index)): float(value) for index, value in outcome["ess"].items()},
        "records": dict(outcome["records"]),
        "attempts": dict(outcome["attempts"]),
    }


def _scenario_index(value, count: int) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer))
            or str(int(value)) != str(value) or not 0 <= int(value) < count):
        raise ValueError("escalation state carries an invalid scenario index")
    return int(value)


def _restore_evidence(values, effective_sample_sizes, count: int) -> tuple:
    """Admit finite or zero (-inf) evidence and finite non-negative ESS."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("escalation state evidence must be real numeric values")
    lnZ = np.asarray(raw, dtype=float)
    if lnZ.shape != (count,):
        raise ValueError("escalation state carries evidence for another scenario set")
    if np.any(np.isnan(lnZ)) or np.any(np.isposinf(lnZ)):
        raise ValueError("escalation state evidence must be finite or -inf")
    if not isinstance(effective_sample_sizes, Mapping):
        raise ValueError("escalation state ESS must be a mapping")
    ess = {}
    for key, value in effective_sample_sizes.items():
        index = _scenario_index(key, count)
        if index in ess:
            raise ValueError("escalation state carries duplicate scenario indices")
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0.0):
            raise ValueError("escalation state ESS must be finite and non-negative")
        ess[index] = float(value)
    return lnZ, ess


def _restore_importance(state: Mapping, expected_scenarios: int) -> Optional[dict]:
    """Restore a retained MC-pilot outcome for resumption."""
    outcome = state.get("mc_importance_outcome")
    if outcome is None:
        return None
    try:
        lnZ, ess = _restore_evidence(outcome["lnZ"], outcome["ess"], expected_scenarios)
        return {"lnZ": lnZ, "ess": ess,
                "records": dict(outcome["records"]), "attempts": dict(outcome["attempts"])}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("escalation state MC importance outcome is malformed") from exc


def _monte_carlo_state_record(
    *,
    lnZ: np.ndarray,
    ess: Mapping[int, float],
    support_state: Mapping[int, Mapping[str, Any]],
    jax_fallbacks: Mapping[int, Mapping[str, Any]],
    scenario_fits: Mapping[int, Mapping[str, Any]],
    n_draws: int,
    random_seed: Optional[int],
    monte_carlo_seconds: float,
    input_identity: Optional[str] = None,
    importance_state: Optional[dict] = None,
    importance_outcome: Optional[dict] = None,
    mc_ldc_provenance: Optional[dict] = None,
) -> dict:
    """The checked Monte Carlo state for a resumed nested escalation.

    It carries the evaluated log-evidences, estimator convergence and support
    diagnostics, accelerator fallback records and representative fits -- the
    complete input set the escalation and its report need -- plus the draw
    budget, the run seed and the measured Monte Carlo wall time. Evidence values
    are converted to Python floats and scenario indices to strings; nested
    fit and provenance mappings retain their supplied values.
    """
    return {
        **({"input_identity": input_identity} if input_identity is not None else {}),
        **({"mc_importance": importance_state} if importance_state is not None else {}),
        **({"mc_importance_outcome": _portable_importance(importance_outcome)}
           if importance_outcome is not None else {}),
        **({"mc_ldc_provenance": _read_mc_ldc_provenance({"mc_ldc_provenance": mc_ldc_provenance})}
           if mc_ldc_provenance is not None else {}),
        "schema_version": 1,
        "escalation_required": True,
        "n_draws": int(n_draws),
        "random_seed": random_seed,
        "monte_carlo_seconds": float(monte_carlo_seconds),
        "lnZ": [float(value) for value in lnZ],
        "effective_sample_sizes": {
            str(int(index)): float(value) for index, value in ess.items()
        },
        "support_states": {
            str(int(index)): dict(value) for index, value in support_state.items()
        },
        "jax_batch_fallbacks": {
            str(int(index)): dict(value) for index, value in jax_fallbacks.items()
        },
        "scenario_fits": {
            str(int(index)): dict(value) for index, value in scenario_fits.items()
        },
    }


def _restore_monte_carlo_state(
    state: Mapping,
    *,
    n_draws: int,
    random_seed: Optional[int],
    expected_scenarios: int,
    input_identity: Optional[str] = None,
    ldc_provenance_out: Optional[list] = None,
) -> tuple:
    """Validate a Monte Carlo checkpoint and rebuild its escalation inputs.

    A state that does not describe this same run -- a different draw budget,
    seed, scenario count, or a checkpoint without a pending escalation -- is
    refused, so a stale or mismatched checkpoint can never be escalated.
    """
    if not isinstance(state, Mapping):
        raise ValueError("escalation state must be a mapping")
    if input_identity is not None and state.get("input_identity") != input_identity:
        raise ValueError("escalation state physical/noise model, prior or numerical inputs changed")
    version = state.get("schema_version")
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)) or version != 1:
        raise ValueError("unsupported escalation state schema version")
    if state.get("escalation_required") is not True:
        raise ValueError("escalation state does not carry a pending escalation")
    try:
        recorded_draws = state["n_draws"]
        recorded_seed = state.get("random_seed")
        if (isinstance(recorded_draws, (bool, np.bool_)) or not isinstance(recorded_draws, (int, np.integer))
                or recorded_draws <= 0):
            raise ValueError("draw budget must be a positive integer")
        if recorded_seed is not None and (
            isinstance(recorded_seed, (bool, np.bool_)) or not isinstance(recorded_seed, (int, np.integer))
            or recorded_seed < 0
        ):
            raise ValueError("random seed must be a non-negative integer or None")
        lnZ, ess = _restore_evidence(state["lnZ"], state["effective_sample_sizes"], expected_scenarios)
        support_state = {
            _scenario_index(index, expected_scenarios): dict(value)
            for index, value in dict(state["support_states"]).items()
        }
        jax_fallbacks = {
            _scenario_index(index, expected_scenarios): dict(value)
            for index, value in dict(state["jax_batch_fallbacks"]).items()
        }
        scenario_fits = {
            _scenario_index(index, expected_scenarios): dict(value)
            for index, value in dict(state["scenario_fits"]).items()
        }
        monte_carlo_seconds = state["monte_carlo_seconds"]
        if isinstance(monte_carlo_seconds, (bool, np.bool_)) or not isinstance(monte_carlo_seconds, Real):
            raise ValueError("Monte Carlo wall time must be real numeric seconds")
        monte_carlo_seconds = float(monte_carlo_seconds)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "escalation state is incomplete or malformed: {0}".format(exc)
        ) from exc
    if recorded_draws != int(n_draws):
        raise ValueError(
            "escalation state draw budget {0} does not match n_draws={1}".format(
                recorded_draws, int(n_draws)
            )
        )
    if recorded_seed != random_seed:
        raise ValueError(
            "escalation state random seed {0!r} does not match random_seed={1!r}".format(
                recorded_seed, random_seed
            )
        )
    if lnZ.ndim != 1 or lnZ.size != int(expected_scenarios):
        raise ValueError(
            "escalation state carries {0} scenarios; the scene requires {1}".format(
                lnZ.size, int(expected_scenarios)
            )
        )
    if not math.isfinite(monte_carlo_seconds) or monte_carlo_seconds < 0.0:
        raise ValueError(
            "escalation state Monte Carlo wall time must be finite and non-negative"
        )
    mc_ldc_provenance = _read_mc_ldc_provenance(state)
    if ldc_provenance_out is not None:
        ldc_provenance_out.append(mc_ldc_provenance)
    return lnZ, ess, support_state, jax_fallbacks, scenario_fits, monte_carlo_seconds
