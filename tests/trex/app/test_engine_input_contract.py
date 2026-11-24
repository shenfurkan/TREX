"""Reject malformed observations and execution budgets at the public engine."""

import stat

import numpy as np
import pytest

from trex.app.engine import run_trex_vetting
from tests.fixtures.synthetic_observations import synthetic_portable_trex_inputs


@pytest.fixture
def engine_inputs(tmp_path):
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path)
    return dict(
        scene=scene, time=time, flux=flux, sigma=sigma, period_days=period,
        depth_ppm=binding["depth_ppm"], exptime_days=binding["exptime_days"],
        sector_weights=binding["sector_weights"], random_seed=binding["random_seed"],
        n_draws=8, nsamples=4,
    )


@pytest.mark.parametrize("field,value", [
    ("period_days", 0.0), ("period_days", -1.0), ("period_days", np.nan),
    ("period_days", np.inf), ("period_days", True),
    ("exptime_days", True), ("exptime_days", np.bool_(True)),
    ("bin_integration_days", True), ("bin_integration_days", -1.0),
    ("n_draws", True), ("n_draws", 0), ("n_draws", 1.5),
    ("nsamples", True), ("nsamples", 0), ("nsamples", 1.5),
    ("random_seed", True), ("random_seed", np.bool_(True)),
    ("random_seed", -1), ("random_seed", 1.5),
])
def test_public_engine_rejects_invalid_scientific_or_execution_scalars(engine_inputs, field, value):
    with pytest.raises(ValueError, match=field):
        run_trex_vetting(**{**engine_inputs, field: value})


@pytest.mark.parametrize("case", ["empty", "matrix", "misaligned", "nonfinite-time", "nonfinite-flux", "scalar-flux"])
def test_public_engine_rejects_invalid_photometry_before_integration(engine_inputs, case):
    time, flux = engine_inputs["time"].copy(), engine_inputs["flux"].copy()
    if case == "empty":
        time, flux = np.empty(0), np.empty(0)
    elif case == "matrix":
        time, flux = time[None, :], flux[None, :]
    elif case == "misaligned":
        time = time[:-1]
    elif case == "nonfinite-time":
        time[0] = np.nan
    elif case == "nonfinite-flux":
        flux[0] = np.inf
    else:
        flux = 1.0
    with pytest.raises(ValueError, match="photometry"):
        run_trex_vetting(**{**engine_inputs, "time": time, "flux": flux})


@pytest.mark.parametrize("recorded_seed", [None, True, -1, 1.5])
def test_seedless_engine_resume_requires_a_valid_retained_seed(engine_inputs, recorded_seed):
    with pytest.raises(ValueError, match="seed"):
        run_trex_vetting(**{**engine_inputs, "random_seed": None},
                         escalation_state_in={"random_seed": recorded_seed})


@pytest.mark.parametrize("composition", ["measured", "conditioned"])
def test_target_grid_changes_bound_evidence_without_changing_background_hosts(engine_inputs, tmp_path, composition):
    from trex.app.scene import TargetScene

    fixed = engine_inputs["scene"]
    payload = fixed.to_dict()
    payload.update(M_s_Msun=None, R_s_Rsun=None, Teff_K=None)
    rows = {"method": "synthetic paired primary properties", "mass_solar": [.45, 1.15],
            "radius_solar": [.43, 1.4], "teff_k": [3600., 6000.]}
    if composition == "conditioned":
        rows["initial_feh_dex"] = [-.4, .3]
        payload.update(feh_dex=None, metallicity_provenance=None)
        # Every capable neighbor still requires its own composition, rather
        # than receiving one of the target's sampled atmosphere labels.
        payload["resolved_neighbors"][0]["metallicity_provenance"] = fixed.metallicity_provenance
    options = {**engine_inputs, "n_draws": 1024, "nsamples": 2, "n_jobs": 1,
               "escalation_checkpoint_directory": str(tmp_path / "checkpoint")}

    def evaluate(weights):
        scene = TargetScene.from_dict({**payload, "target_distribution": {**rows, "weights": weights}})
        result = run_trex_vetting(**{**options, "scene": scene}, escalation_state_out={})
        return {row["scenario"]: row["ln_evidence"] for row in result.scenario_fits}

    cool = evaluate([.8, .2])
    warm = evaluate([.2, .8])
    assert cool["PTP"] is not None and warm["PTP"] is not None
    assert cool["PTP"] != warm["PTP"]
    assert cool["BTP"] is not None and cool["BTP"] == warm["BTP"]


def test_none_and_unmatched_gaia_constraints_preserve_all_evidence_bits(engine_inputs, tmp_path):
    import struct

    from trex.acquisition.gaia_constraints import GaiaDetectionConstraint

    options = {**engine_inputs, "n_draws": 4096, "nsamples": 2, "n_jobs": 1,
               "escalation_checkpoint_directory": str(tmp_path / "checkpoint")}
    scope = options["scene"].target_source_id
    reference = run_trex_vetting(**options)
    assert reference.status == "ok" and reference.fpp is not None and reference.nfpp is not None
    for constraints in (None, [GaiaDetectionConstraint(scope, False, p_detect=0.9),
                               GaiaDetectionConstraint(scope, True, period_match=False)]):
        result = run_trex_vetting(**options, release_constraints=constraints)
        assert result.input_identity == reference.input_identity
        assert struct.pack(">dd", result.fpp, result.nfpp) == struct.pack(">dd", reference.fpp, reference.nfpp)
        fields = ("diagnostic_key", "ln_evidence", "finite_likelihood_draws")
        assert [tuple(row.get(key) for key in fields) for row in result.scenario_fits] == [
            tuple(row.get(key) for key in fields) for row in reference.scenario_fits
        ]
    with pytest.raises(ValueError, match="no approved Gaia release detection likelihood"):
        run_trex_vetting(**options, release_constraints=[GaiaDetectionConstraint(scope, True, period_match=True)])


def test_seedless_public_engine_reuses_completed_nested_work_after_pause(engine_inputs, tmp_path, monkeypatch):
    """Real handoff/cache routing with synthetic MC evidence and a stub nested likelihood boundary."""
    from trex.app import engine
    from trex.evidence import evidence_escalation_calls as escalation_calls
    from trex.evidence.nested_checkpoint import NestedCheckpointPaused
    from tests.trex.evidence.escalation_support import _result

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "dynesty")
    mc_seeds, nested_calls = [], []
    count = engine.N_TARGET_SCENARIOS + len(engine_inputs["scene"].resolved_neighbors) * engine.N_NEIGHBOR_SCENARIOS

    def monte_carlo(**kwargs):
        mc_seeds.append(kwargs["random_seed"])
        values = np.full(count, -np.inf)
        values[:3] = [-100.0, -100.5, -101.0]
        kwargs["ess_out"].update({0: 500.0, 1: 1.0, 2: 1.0})
        return values, "ok"

    def integrate(**kwargs):
        nested_calls.append((kwargs["use_2xP"], kwargs["random_seed"]))
        if len(nested_calls) == 2:
            raise NestedCheckpointPaused("synthetic pause after one completed scenario")
        return _result(-300.0)

    monkeypatch.setattr(engine, "calc_target_evidences", monte_carlo)
    monkeypatch.setattr(escalation_calls, "eb_scenario_nested_evidence", integrate)
    inputs = {**engine_inputs, "random_seed": None, "n_draws": engine.TREX_PRODUCTION_N_DRAWS,
              "escalation_checkpoint_directory": str(tmp_path / "nested"), "n_jobs": 1}
    state = {}
    run_trex_vetting(**inputs, escalation_state_out=state)
    assert type(state["random_seed"]) is int and mc_seeds == [state["random_seed"]]
    assert stat.S_IMODE((tmp_path / "nested").stat().st_mode) == 0o700
    with pytest.raises(NestedCheckpointPaused, match="synthetic pause"):
        run_trex_vetting(**inputs, escalation_state_in=state)
    completed_call = nested_calls[0]
    result = run_trex_vetting(**inputs, escalation_state_in=state)
    assert result.status == "ok"
    assert mc_seeds == [state["random_seed"]], "resume repeated Monte Carlo"
    assert nested_calls.count(completed_call) == 1, "resume repeated a completed nested scenario"
    assert nested_calls[1] == nested_calls[2], "the interrupted scenario's seed changed"
