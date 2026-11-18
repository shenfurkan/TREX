"""Public real-engine and portable writer/loader controls; budgets are software-only."""

from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from trex.app.engine import run_observation_vetting
from trex.app.observation_io import export_observation_input
from trex.app.run import load_input
from trex.evidence.marginal_likelihoods import TARGET_SCENARIO_SPECS, NEIGHBOR_SCENARIO_NAMES
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import joint_case


def test_public_real_engine_keeps_both_datasets_and_all_scenario_records(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
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
    metadata = json.loads((output / "trex-input.json").read_text())
    asset = output / metadata["band_light"]["synthetic/ground"]["file"]
    with asset.open("ab") as handle:
        handle.write(b"changed asset")
    with pytest.raises(ValueError, match="changed"):
        load_input(output)
    with pytest.raises(ValueError, match="changed"):
        bound._require_scene()


def test_public_unknown_host_share_refuses_before_any_evidence(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    first, second = observations.observations
    second = replace(second, source_shares={"target": 1., "neighbor": None})
    with pytest.raises(TrexEvidenceUnavailableError, match="unknown light"):
        run_observation_vetting(scene, ObservationSet((first, second), observations.reference),
                               period_days=3., depth_ppm=1000., epoch_days=0., band_light=lights,
                               n_draws=4, random_seed=7)


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
