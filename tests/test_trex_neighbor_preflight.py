"""Observe real TREX draws; missing capable hosts must refuse before any draw."""

from contextlib import contextmanager
from copy import deepcopy
import math
import sys

import numpy as np
import pytest

from trex.app.engine import run_trex_vetting
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.evidence.marginal_likelihoods import N_NEIGHBOR_SCENARIOS, N_TARGET_SCENARIOS, _scenario_stages, calc_target_evidences
from tests.fixtures.synthetic_observations import synthetic_neighbor_preflight_case


@contextmanager
def _observe_draws():
    """Profile without replacing the engine, sampler, likelihood or RNG."""
    events = []
    states = {}
    previous = sys.getprofile()

    # The scenario overlap drives _scenario_stages directly; preparation and RNG draws stay on this thread.
    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _scenario_stages.__code__:
            events.append("scenario")
            states[id(frame)] = deepcopy(frame.f_locals["rng"].bit_generator.state)
        elif event == "return" and frame.f_code is _scenario_stages.__code__:
            if states.pop(id(frame)) != frame.f_locals["rng"].bit_generator.state:
                events.append("rng-advanced")

    sys.setprofile(observe)
    try:
        yield events
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("fraction", [1 / 512, 1 / 8], ids=["entire-flux-boundary", "capable"])
@pytest.mark.parametrize("prepend", [False, True], ids=["first-neighbor", "later-neighbor"])
def test_engine_refuses_missing_capable_host_before_scenario_or_rng_draw(tmp_path, fraction, prepend):
    scene, inputs = synthetic_neighbor_preflight_case(
        tmp_path, neighbor_fraction=fraction, characterized=True, prepend_characterized=prepend,
    )
    # Build a valid characterized prefix before making only the final host unavailable.
    payload = scene.to_dict()
    payload["resolved_neighbors"][-1].update(
        M_s=None, R_s=None, stellar_parameter_unavailable_reason="synthetic missing stellar data")
    scene = type(scene).from_dict(payload)
    before = scene.to_dict()
    with _observe_draws() as events:
        with pytest.raises(TrexEvidenceUnavailableError, match="synthetic missing stellar data"):
            run_trex_vetting(scene, **inputs)
    assert events == [], "missing host data was discovered after actual scenario/RNG work"
    assert scene.to_dict() == before


@pytest.mark.parametrize("fraction", [0.0, 1 / 1024], ids=["zero-flux", "too-faint"])
def test_engine_preserves_impossible_neighbor_slots_without_inventing_stellar_data(tmp_path, fraction):
    scene, inputs = synthetic_neighbor_preflight_case(tmp_path, neighbor_fraction=fraction)
    before = scene.to_dict()
    state = {}
    with _observe_draws() as events:
        result = run_trex_vetting(scene, **inputs, escalation_state_out=state)
    assert "scenario" in events and any(event != "scenario" for event in events)
    assert scene.to_dict() == before
    assert result.claim_eligible is False
    slots = result.scenario_fits[N_TARGET_SCENARIOS:]
    assert len(slots) == N_NEIGHBOR_SCENARIOS
    assert all(slot["status"] == "host-excluded" and slot["ln_evidence"] == -math.inf for slot in slots)
    # An unconverged test-budget result has no normalized probabilities.
    assert all(math.isnan(slot["prob"]) for slot in slots)
    assert all(slot["source_id"] == "synthetic-uncharacterized" for slot in slots)
    assert all(slot["representative_draw"] is None for slot in slots)


def test_admitted_scene_preserves_draws_slots_and_limb_darkening_provenance(tmp_path):
    """Caller equivalence, not an independent physical-accuracy benchmark."""
    scene, inputs = synthetic_neighbor_preflight_case(tmp_path, neighbor_fraction=1 / 8, characterized=True)
    state, provenance = {}, []
    run_trex_vetting(scene, **inputs, u1=0.4, u2=0.2,
                     escalation_state_out=state, ldc_provenance=provenance)
    assert state, "the declared tiny budget must leave an escalation handoff"
    direct = {key: value for key, value in inputs.items() if key not in {"sector_weights", "escalation_checkpoint_directory"}}
    fits, direct_provenance = {}, []
    lnz, _ = calc_target_evidences(
        **direct, M_s_Msun=scene.M_s_Msun, R_s_Rsun=scene.R_s_Rsun, u1=0.4, u2=0.2,
        target_teff_k=scene.Teff_K,
        target_aperture_flux_fraction=scene.target_aperture_flux_fraction(inputs["sector_weights"]),
        nearby_stars=scene.neighbor_dicts_for_evidence(inputs["sector_weights"]),
        contrast_separations=scene.contrast_separations, contrast_values=scene.contrast_values,
        plx=scene.plx_mas,
        target_tess_mag=scene.Tmag, background_population=scene.background_population(),
        background_area_sq_deg=scene.background_area_sq_deg, target_feh=scene.feh_dex,
        resolution_limit_arcsec=scene.unresolved_resolution_limit_arcsec,
        target_source_id=str(scene.tic_id), representative_fit_out=fits,
        ldc_provenance=direct_provenance,
    )
    np.testing.assert_array_equal(state["lnZ"], lnz)
    assert len(fits) == N_TARGET_SCENARIOS + N_NEIGHBOR_SCENARIOS
    # A single host-setup lookup is separate from the MC draws.
    setup = [p for p in provenance if p.get("stage") == "scene-preflight"]
    assert len(setup) == 1 and setup[0]["scenario"].endswith("-host-setup")
    assert (setup[0]["total_draws"], setup[0]["reference_draws"], setup[0]["atlas_draws"],
            setup[0]["phoenix_draws"], setup[0]["clamped_draws"]) == (1, 1, 0, 0, 0)
    def events(records):  # the engine tags the stage and stores nodes as JSON pairs
        return [{**{k: v for k, v in p.items() if k != "stage"},
                 "used_nodes": [[float(t), float(g)] for t, g in p["used_nodes"]]} for p in records]

    assert events(p for p in provenance if p.get("stage") == "monte-carlo") == events(direct_provenance)
    for key, record in fits.items():
        retained = state["scenario_fits"][str(key)]
        assert retained["representative_draw"] == record["representative_draw"]


def test_resumed_engine_also_refuses_uncharacterized_capable_host(tmp_path):
    scene, inputs = synthetic_neighbor_preflight_case(tmp_path, neighbor_fraction=1 / 8, characterized=True)
    state = {}
    run_trex_vetting(scene, **inputs, escalation_state_out=state)
    assert state
    payload = scene.to_dict()
    neighbor = payload["resolved_neighbors"][0]
    neighbor.update(M_s=None, R_s=None, stellar_parameter_unavailable_reason="synthetic missing stellar data")
    resumed_scene = type(scene).from_dict(payload)
    with _observe_draws() as events:
        with pytest.raises(TrexEvidenceUnavailableError, match="synthetic missing stellar data"):
            run_trex_vetting(resumed_scene, **inputs, escalation_state_in=state)
    assert events == []
