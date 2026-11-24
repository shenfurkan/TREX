"""Public real-engine and portable writer/loader controls; budgets are software-only."""

from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from trex.app.aij_loader import read_aij_observation
from trex.app.engine import run_observation_vetting
from trex.app.observation_io import export_observation_input
from trex.app.run import load_input
from trex.app.scene import TargetScene
from trex.evidence.marginal_likelihoods import TARGET_SCENARIO_SPECS, NEIGHBOR_SCENARIO_NAMES
from trex.evidence.nested_checkpoint import CheckpointInputMismatch, NestedCheckpointPaused
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import joint_case


def test_public_real_engine_keeps_both_datasets_and_all_scenario_records(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    space, ground = observations.observations
    table = tmp_path / "ground.tbl"
    table.write_text(
        "BJD_TDB\trel_flux_T1\trel_flux_err_T1\tAIRMASS\tFWHM\n"
        "2459000.055\t1\t.1\t1.8\t4\n"
        "2459000.105\t.999\t.1\t1.4\t3.8\n"
        "2459000.155\t1\t.1\t1.2\t3.5\n"
        "2459001.605\t1.002\t.1\t1.15\t3.6\n"
        "2459001.635\t1\t.1\t1.1\t3.7\n",
        encoding="utf-8",
    )
    ground = read_aij_observation(
        table, dataset_id=ground.dataset_id, segment_id=ground.segment_id,
        band_id=ground.band_id, response_id=ground.response_id, reference=observations.reference,
        observer="synthetic observatory", target="synthetic target", stamp="mid",
        exposure_days=.001, exposure_source="synthetic original exposure", nsamples=4,
        aperture_id=ground.aperture_id, source_shares=ground.source_shares, lineage=ground.lineage,
        flux_scale=1., uncertainty_source="synthetic normalized differential flux",
        prior_mean=[0., 0.], prior_covariance=[[1e-4, 0.], [0., 1e-4]],
        baseline_mode="additive", baseline_source="independent synthetic calibration, constant per exposure",
    )
    observations = ObservationSet((space, ground), observations.reference)
    pending = {}
    result = run_observation_vetting(
        scene, observations, period_days=3., depth_ppm=1000., epoch_days=0.,
        band_light=lights, n_draws=128, random_seed=7, n_jobs=1, escalation_state_out=pending,
        escalation_checkpoint_directory=str(tmp_path / "checkpoint"))
    assert len(result.scenario_fits) == len(TARGET_SCENARIO_SPECS) + len(NEIGHBOR_SCENARIO_NAMES)
    assert {item["dataset_id"] for item in result.fit_evaluation["joint_observations"]["datasets"]} == {"space", "ground"}
    assert all(record["status"] != "unavailable" for record in result.scenario_fits)


def test_portable_round_trip_binds_every_band_asset_and_spawned_density(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    output = export_observation_input(scene, observations, tmp_path / "input",
                                     period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights,
                                     random_seed=7)
    bound, arrays, period, exposure = load_input(output)
    assert period == 3. and exposure is None
    binding = bound._require_scene()
    joint = binding["joint_observations"]
    assert joint.likelihood.observations.to_dict() == observations.to_dict()
    assert len(binding["input_artifacts"]) == 5
    physical = dict(host_mass=1., host_radius=1., host_teff=5700., host_metallicity=0.,
                    period_days=3., inc_deg=89., a_cm=8e11, ecc=0., argp_deg=90., planet_radius=3.)
    expected = joint.evaluate(**physical)
    assert pickle.loads(pickle.dumps(joint)).evaluate(**physical) == expected
    np.testing.assert_array_equal(arrays["flux"], observations.observations[0].flux)
    health = bound.doctor(arrays["time"], arrays["flux"], arrays["sigma"], period, N=128)
    assert health["status"] == "ready", health
    assert health["evidence_draws"] == 0
    assert health["neighbors"] == 1
    metadata = json.loads((output / "trex-input.json").read_text())
    asset = output / metadata["band_light"]["synthetic/ground"]["file"]
    with asset.open("ab") as handle:
        handle.write(b"changed asset")
    with pytest.raises(ValueError, match="changed"):
        load_input(output)
    with pytest.raises(ValueError, match="changed"):
        bound._require_scene()


def test_portable_non_tic_target_does_not_invent_tess_magnitudes(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    scene = TargetScene.from_dict({
        **scene.to_dict(), "tic_id": None, "Tmag": None,
        "target_identity": {"namespace": "gaia-dr3", "native_id": "123"},
    })
    output = export_observation_input(scene, observations, tmp_path / "input",
                                     period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights)
    bound, _, _, _ = load_input(output)
    assert bound.stars["ID"].tolist() == ["target", "neighbor"]
    assert bound.stars["Tmag"].isna().tolist() == [True, True]


def test_public_unknown_host_share_refuses_before_any_evidence(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    first, second = observations.observations
    second = replace(second, source_shares={"target": 1., "neighbor": None})
    with pytest.raises(TrexEvidenceUnavailableError, match="unknown light"):
        run_observation_vetting(scene, ObservationSet((first, second), observations.reference),
                               period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights,
                                n_draws=4, random_seed=7)


def test_flux_only_neighbor_survives_scene_round_trip_without_host_scenarios(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    payload = scene.to_dict()
    payload["resolved_neighbors"][0]["eclipse_host_eligible"] = False
    flux_only = TargetScene.from_dict(payload)
    result = run_observation_vetting(
        flux_only, observations, period_days=3., depth_ppm=1000., epoch_days=0.,
        band_light=lights, n_draws=128, random_seed=7, n_jobs=1, escalation_state_out={},
        escalation_checkpoint_directory=str(tmp_path / "checkpoint"),
    )
    assert flux_only.resolved_neighbors[0]["source_id"] == "neighbor"
    assert len(result.scenario_fits) == len(TARGET_SCENARIO_SPECS)
    assert {record["source_id"] for record in result.scenario_fits} == {"target"}


@pytest.mark.parametrize("pause_before_first", [False, True], ids=["completed-prefix", "empty-ledger"])
def test_initial_monte_carlo_resume_matches_uninterrupted_scenarios(tmp_path, monkeypatch, pause_before_first):
    from trex.evidence import monte_carlo_state

    scene, observations, lights = joint_case(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    file = checkpoint / monte_carlo_state.INITIAL_MC_CHECKPOINT
    clock = [1.0 if pause_before_first else 0.0]
    monkeypatch.setattr(monte_carlo_state, "monotonic", lambda: clock[0])
    real_save = monte_carlo_state.InitialMonteCarloRecovery.save
    saved = []

    def save(recovery, index, **scenario):
        real_save(recovery, index, **scenario)
        saved.append(index)
        clock[0] = 1.0

    monkeypatch.setattr(monte_carlo_state.InitialMonteCarloRecovery, "save", save)
    options = dict(period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights,
                   n_draws=128, random_seed=7, n_jobs=1, escalation_checkpoint_directory=str(checkpoint))
    reason = "no scenarios completed" if pause_before_first else "completed Monte Carlo scenarios retained"
    with pytest.raises(NestedCheckpointPaused, match=reason):
        run_observation_vetting(scene, observations, **options, escalation_state_out={},
                                escalation_deadline_monotonic=1.0)
    with np.load(file, allow_pickle=False) as archive:
        stored = json.loads(str(archive["metadata"].item()))
    prefix = {int(name) for name in stored["scenarios"]}
    assert prefix == set(saved)
    assert bool(prefix) is not pause_before_first
    with pytest.raises(CheckpointInputMismatch, match="draw budget"):
        run_observation_vetting(scene, observations, **{**options, "n_draws": 129},
                                escalation_state_out={}, resume_initial_monte_carlo=True)
    saved.clear()
    resumed = run_observation_vetting(scene, observations, **options, escalation_state_out={},
                                      resume_initial_monte_carlo=True)
    assert not prefix.intersection(saved)
    reference = run_observation_vetting(
        scene, observations, **{**options, "escalation_checkpoint_directory": str(tmp_path / "reference")},
        escalation_state_out={},
    )
    fields = ("scenario_index", "diagnostic_key", "ln_evidence", "finite_likelihood_draws")
    assert [tuple(row.get(key) for key in fields) for row in resumed.scenario_fits] == [
        tuple(row.get(key) for key in fields) for row in reference.scenario_fits
    ]


def test_initial_monte_carlo_ledger_survives_sigkill_before_first_result(tmp_path):
    import os
    import signal
    import subprocess
    import sys

    from trex.evidence.monte_carlo_state import INITIAL_MC_CHECKPOINT

    if os.name != "posix":
        pytest.skip("SIGKILL test requires a POSIX child")
    script = """
import os, signal, sys
from pathlib import Path
from time import monotonic
from tests.fixtures.joint_observations import joint_case
from trex.app.engine import run_observation_vetting

def die_at_first_scenario(step, **update):
    if step == "Scenario TP":
        os.kill(os.getpid(), signal.SIGKILL)

root = Path(sys.argv[1])
scene, observations, lights = joint_case(root)
run_observation_vetting(
    scene, observations, period_days=3., depth_ppm=1000., epoch_days=0.,
    band_light=lights, n_draws=128, random_seed=7, n_jobs=1, escalation_state_out={},
    escalation_checkpoint_directory=str(root / "checkpoint"),
    escalation_deadline_monotonic=monotonic() + 300, progress_callback=die_at_first_scenario)
"""
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=300, check=False)
    assert child.returncode == -signal.SIGKILL, child.stderr.decode(errors="replace")
    checkpoint = tmp_path / "checkpoint"
    with np.load(checkpoint / INITIAL_MC_CHECKPOINT, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata"].item()))["scenarios"] == {}
    scene, observations, lights = joint_case(tmp_path)
    options = dict(period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights,
                   n_draws=128, random_seed=7, n_jobs=1)
    resumed = run_observation_vetting(
        scene, observations, **options, escalation_state_out={}, resume_initial_monte_carlo=True,
        escalation_checkpoint_directory=str(checkpoint))
    reference = run_observation_vetting(
        scene, observations, **options, escalation_state_out={},
        escalation_checkpoint_directory=str(tmp_path / "reference"))
    fields = ("diagnostic_key", "ln_evidence", "finite_likelihood_draws")
    assert [tuple(row.get(key) for key in fields) for row in resumed.scenario_fits] == [
        tuple(row.get(key) for key in fields) for row in reference.scenario_fits
    ]


def test_export_never_overwrites_an_existing_directory(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-data"
    marker.write_text("retain")
    with pytest.raises(FileExistsError):
        export_observation_input(scene, observations, output, period_days=3.,
                                 depth_ppm=1000., epoch_days=0., band_light=lights)
    assert marker.read_text() == "retain"


def test_zero_event_density_survives_portable_model_round_trip(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    output = export_observation_input(scene, observations, tmp_path / "input", period_days=3.,
                                     depth_ppm=1000., epoch_days=0., band_light=lights)
    bound, _, _, _ = load_input(output)
    before = JointLikelihood(observations)
    after = bound._require_scene()["joint_observations"].likelihood
    models = {row.dataset_id: np.ones(row.flux.shape) for row in observations.observations}
    assert before.density(models) == after.density(models)
