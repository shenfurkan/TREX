"""Real portable target inference/replay; no likelihood or ESS substitutions.

The low-information fixture tests API plumbing, convergence enforcement and
input fidelity, not independent astrophysical calibration or planet recovery.
"""

import copy
import json

import numpy as np
import pytest

from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.foundation.physical_constants import PARTS_PER_MILLION
from trex.app.scene import TargetScene
from trex.app.target import target
from tests.fixtures.synthetic_observations import synthetic_portable_trex_inputs


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(
        tmp_path_factory.mktemp("portable-inference"),
    )
    obj = target.from_scene(scene, **binding)
    assert obj.calc_probs(time, flux, sigma, period, N=4096, nsamples=4,
                          exptime=binding["exptime_days"]) is None
    return obj, binding, time, flux, sigma, period


def test_real_engine_returns_aligned_source_rows_without_fabricated_hosts(completed):
    obj, binding, *_ = completed
    assert obj.trex_result.status == "ok"
    assert obj.trex_result.claim_eligible is obj.claim_eligible is False
    assert len(obj.probs) == 21  # 15 target slots plus 3 slots per retained neighbor.
    assert obj.probs["ID"].tolist() == ["100001"] * 15 + ["100002"] * 3 + ["100003"] * 3
    assert all(len(getattr(obj, name)) == 21 for name in ("star_num", "u1", "u2", "fluxratio_EB", "fluxratio_comp"))
    assert obj.probs["prob"].sum() == pytest.approx(1.0)
    assert obj.FPP == obj.trex_result.fpp
    assert obj.NFPP == obj.trex_result.nfpp
    assert obj.probs.iloc[-3:]["M_s"].isna().all()
    assert obj.probs.iloc[-3:]["prob"].eq(0).all()
    assert obj.calculation_provenance["n_draws"] == 4096
    assert obj.calculation_provenance["random_seed"] == binding["random_seed"]
    assert obj.calculation_provenance["limb_darkening"]
    assert obj.calculation_provenance["input_artifacts"] == binding["input_artifacts"]


def test_measured_transit_values_reach_the_engine(tmp_path, monkeypatch):
    # They centre the Monte Carlo draws, so the public path must match the production chain.
    import trex.engine

    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path)
    measured = {"depth_sigma_ppm": 150.0, "duration_days": .1, "duration_sigma_days": .01}
    seen = {}
    real = trex.engine.run_trex_vetting

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(trex.engine, "run_trex_vetting", spy)
    obj = target.from_scene(scene, **binding, **measured)
    obj.calc_probs(time, flux, sigma, period, N=8192, nsamples=4, exptime=binding["exptime_days"])
    assert {name: seen[name] for name in measured} == measured
    assert obj.calculation_provenance["duration_days"] == .1
    with pytest.raises(ValueError, match="duration_days must be positive"):
        target.from_scene(scene, **binding, duration_days=0.0)


def test_actual_draw_replay_reproduces_likelihood_in_observed_aperture_frame(completed):
    obj, binding, time, flux, sigma, _ = completed
    models = obj.trex_result.representative_models(time)
    assert len(models) == len(obj.probs)
    families = set()
    for entry, model in zip(obj.trex_result.scenario_fits, models, strict=True):
        if entry["has_representative_draw"]:
            draw = entry["representative_draw"]
            assert np.isfinite(model).all() and model.shape == time.shape
            assert draw["a_cm"] > 0
            reconstructed = -.5 * np.sum(((flux - model) / sigma)**2)
            # Existing independent forward-model parity budget is 5 ppm. Its
            # exact chi-square perturbation bound includes both residual terms.
            delta = 5.0 / PARTS_PER_MILLION
            bound = np.sum((np.abs(flux - model) * delta + .5 * delta**2) / sigma**2)
            roundoff = np.finfo(float).eps * time.size * max(1., abs(draw["max_lnL"])) * 16
            assert abs(reconstructed - draw["max_lnL"]) <= bound + roundoff
            families.add((entry["scenario"], draw["period_multiplier"]))
            expected_fraction = .8375 if entry["neighbor_index"] is None else .1624
            assert entry["aperture_flux_fraction"] == pytest.approx(expected_fraction)
        else:
            assert model is None
    assert {("TP", 1), ("EB", 1), ("EBx2P", 2), ("NTP", 1)} <= families
    assert obj.trex_result.fit_evaluation["exptime_days"] == binding["exptime_days"]
    assert obj.trex_result.fit_evaluation["bin_integration_days"] == binding["bin_integration_days"] > 0


def test_fit_pdf_uses_retained_observations_and_rejects_changed_inputs(completed, tmp_path):
    obj, _, time, flux, sigma, _ = completed
    name = tmp_path / "retained-fits"
    assert obj.plot_fits(time, flux, sigma, save=True, fname=str(name)) is None
    assert name.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="differ"):
        obj.plot_fits(time + .001, flux, sigma)


def test_failed_run_clears_outputs_and_preserves_real_convergence_diagnostics(completed):
    original, binding, time, flux, sigma, period = completed
    obj = copy.deepcopy(original)
    with pytest.raises(TrexEvidenceUnavailableError):
        obj.calc_probs(time, flux, sigma, period, N=1, nsamples=4, exptime=binding["exptime_days"])
    assert obj.probs is obj.FPP is obj.NFPP is obj.trex_result is None
    assert obj.failure_record["n_draws"] == 1
    assert obj.failure_record["diagnostics"]["effective_sample_sizes"]
    with pytest.raises(ValueError, match="no synthetic fit"):
        obj.plot_fits(time, flux, sigma)


@pytest.mark.parametrize("option", [
    {"filt": "J"}, {"flatpriors": True}, {"molusc_file": "unsupported.csv"},
    {"contrast_curve_file": "unbound.csv"}, {"drop_scenario": ["EB"]},
])
def test_unsupported_options_do_not_silently_change_scenario_support(completed, option):
    original, binding, time, flux, sigma, period = completed
    obj = copy.deepcopy(original)
    with pytest.raises(NotImplementedError):
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"], **option)
    assert obj.probs is obj.FPP is obj.NFPP is None


def test_legacy_exposure_default_cannot_replace_measured_exposure(completed):
    obj, _, time, flux, sigma, period = completed
    obj = copy.deepcopy(obj)
    with pytest.raises(ValueError, match="native exposure"):
        obj.calc_probs(time, flux, sigma, period)
    assert obj.FPP is None


@pytest.mark.parametrize("option", [
    {"N": 0}, {"N": True}, {"N": 1.5}, {"nsamples": 0}, {"nsamples": True},
    {"parallel": "yes"}, {"verbose": 1.0}, {"exptime": np.nan},
])
def test_invalid_execution_options_fail_before_inference(completed, option):
    original, binding, time, flux, sigma, period = completed
    obj = copy.deepcopy(original)
    options = {"N": 64, "nsamples": 4, "exptime": binding["exptime_days"], **option}
    with pytest.raises(ValueError):
        obj.calc_probs(time, flux, sigma, period, **options)
    assert obj.FPP is None and obj.trex_result is None


def test_period_interval_and_misaligned_observations_are_not_coerced(completed):
    original, binding, time, flux, sigma, period = completed
    obj = copy.deepcopy(original)
    with pytest.raises(NotImplementedError, match="period intervals"):
        obj.calc_probs(time, flux, sigma, [period, period + .1], exptime=binding["exptime_days"])
    for times, values, errors in ((time[:-1], flux, sigma), (time, flux[:-1], sigma),
                                   (time, flux, np.full_like(sigma, np.nan))):
        with pytest.raises(ValueError):
            obj.calc_probs(times, values, errors, period, exptime=binding["exptime_days"])


def test_missing_replay_parameters_do_not_get_reconstructed_or_guessed(completed):
    original, _, time, *_ = completed
    result = copy.deepcopy(original.trex_result)
    entry = next(row for row in result.scenario_fits if row["has_representative_draw"])
    del entry["representative_draw"]["a_cm"]
    with pytest.raises(ValueError, match="missing finite physical"):
        result.representative_models(time)
    result = copy.deepcopy(original.trex_result)
    entry = next(row for row in result.scenario_fits
                 if row["has_representative_draw"] and row["representative_draw"]["is_EB"])
    entry["representative_draw"]["u1_EB"] = None
    with pytest.raises(ValueError, match="secondary"):
        result.representative_models(time)


def test_scene_roundtrip_preserves_neighbors_and_revalidates_retained_background(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    payload = json.loads(json.dumps(scene.to_dict(), allow_nan=False))
    restored = TargetScene.from_dict(payload)
    assert restored.to_dict() == scene.to_dict()
    restored.trilegal_cache.write_text("changed background\n")
    with pytest.raises(ValueError, match="background_sha256"):
        TargetScene.from_dict(payload)
    with pytest.raises(ValueError, match="changed"):
        target.from_scene(scene, **binding)


def test_mutation_or_changed_calibration_cannot_reuse_old_probabilities(tmp_path):
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path)
    obj = target.from_scene(scene, **binding)
    obj.stars.loc[0, "mass"] = 2.0  # Direct DataFrame edits must be detected too.
    with pytest.raises(ValueError, match="table changed"):
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"])
    assert obj._scene_binding is None
    obj = target.from_scene(scene, **binding)
    separations = obj.stars["sep (arcsec)"].copy()
    obj.update_star(obj.ID, "mass", 1.1)
    assert obj._scene_binding is None
    np.testing.assert_array_equal(obj.stars["sep (arcsec)"], separations)
    obj = target.from_scene(scene, **binding)
    next(path for path in tmp_path.glob("*-input.json")).write_text("changed calibration\n")
    with pytest.raises(ValueError, match="artifact"):
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"])
    assert obj.probs is None and obj._scene_binding is None


def test_offline_scene_does_not_manufacture_images_or_gaussian_throughput(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    obj = target.from_scene(scene, **binding)
    with pytest.raises(ValueError, match="image unavailable"):
        obj.plot_field(sector=1)
    with pytest.raises(ValueError, match="acquired pixel"):
        obj.calc_depths(.008, [np.array([[0, 0]])] * 2)


@pytest.mark.parametrize("weights", [{1: 1.0}, {1: 1.0, 2: 0.0},
                                    {1: np.nan, 2: 1.0}, {True: 1.0, 2: 1.0}])
def test_scene_binding_requires_real_complete_sector_weights(tmp_path, weights):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    with pytest.raises(ValueError, match="weight|sector"):
        target.from_scene(scene, **{**binding, "sector_weights": weights})


def test_scene_binding_cannot_drop_neighbors_or_double_count_aperture_light(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    obj = target.from_scene(scene, **binding)
    foreign = TargetScene.from_dict(scene.to_dict())
    foreign.tic_id += 1
    with pytest.raises(ValueError, match="target/sectors"):
        obj.bind_scene(foreign, **binding)
    incomplete = TargetScene.from_dict(scene.to_dict())
    incomplete.resolved_neighbors.pop()
    with pytest.raises(ValueError, match="every catalog source"):
        obj.bind_scene(incomplete, **binding)
    inconsistent = TargetScene.from_dict(scene.to_dict())
    inconsistent.target_aperture_flux_fractions = {1: 1.0, 2: 1.0}
    with pytest.raises(ValueError, match="overcounted"):
        target.from_scene(inconsistent, **binding)


def test_per_bin_widths_reach_inference_and_replay_like_an_equal_scalar(tmp_path):
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path)
    runs = []
    for width in (.01, [.01]*time.size):
        obj = target.from_scene(scene, **{**binding, "bin_integration_days": width})
        obj.calc_probs(time, flux, sigma, period, N=4096, nsamples=4, exptime=binding["exptime_days"])
        runs.append(float(obj.FPP))
    assert runs[0] == pytest.approx(runs[1], abs=1e-12)
    with pytest.raises(ValueError, match="bin width"):
        target.from_scene(scene, **{**binding, "bin_integration_days": [.01, 0.]})


def test_packaged_example_leaves_real_diagnostics_or_only_the_failure(tmp_path, completed):
    from trex.app.example import main

    main([str(tmp_path / "run")])
    summary = json.loads((tmp_path / "run/result/summary.json").read_text())
    assert summary["claim_eligible"] is False and summary["effective_sample_sizes"]
    assert (tmp_path / "run/result/scenarios.csv").read_text().count("\n") == 22
    original, binding, time, flux, sigma, period = completed
    failed = copy.deepcopy(original)
    with pytest.raises(TrexEvidenceUnavailableError):
        failed.calc_probs(time, flux, sigma, period, N=1, nsamples=4, exptime=binding["exptime_days"])
    failed.write_outputs(tmp_path / "failed")
    assert [path.name for path in (tmp_path / "failed").iterdir()] == ["failure.json"]
    record = json.loads((tmp_path / "failed/failure.json").read_text())
    assert record["error_type"] == "TrexEvidenceUnavailableError" and record["diagnostics"]
