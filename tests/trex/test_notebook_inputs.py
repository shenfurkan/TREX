"""Notebook usability of the public TREX target: a returned figure and optional SHA-256 inputs.

The digests below come from ``hashlib`` over the written bytes, independent of TREX's own
helper, so a wrong digest in the implementation cannot also be the expected value.
"""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from tests.fixtures.synthetic_observations import synthetic_portable_trex_inputs  # noqa: E402
from trex.scene import TargetScene  # noqa: E402
from trex.target import target  # noqa: E402


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(
        tmp_path_factory.mktemp("notebook-plot"))
    obj = target.from_scene(scene, **binding)
    obj.calc_probs(time, flux, sigma, period, N=4096, nsamples=4, exptime=binding["exptime_days"])
    return obj, time, flux, sigma


def test_plot_fits_returns_open_figure_with_one_panel_pair_per_scenario(completed):
    obj, time, flux, sigma = completed
    fig = obj.plot_fits(time, flux, sigma)
    try:
        assert plt.fignum_exists(fig.number)  # not closed: a notebook can still display it
        scenarios = len(obj.trex_result.scenario_fits)
        assert scenarios > 1
        # Each scenario owns a data panel and a residual panel.
        assert len(fig.axes) == 2 * scenarios
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert len(titles) == scenarios
    finally:
        plt.close(fig)


def test_plot_fits_with_save_still_writes_pdf_closes_and_returns_nothing(completed, tmp_path):
    obj, time, flux, sigma = completed
    before = set(plt.get_fignums())
    assert obj.plot_fits(time, flux, sigma, save=True, fname=str(tmp_path / "fits")) is None
    assert (tmp_path / "fits.pdf").read_bytes().startswith(b"%PDF")
    assert set(plt.get_fignums()) == before


def _scene_without_hash(scene):
    payload = scene.to_dict()
    del payload["background_sha256"]
    return TargetScene.from_dict(payload)


def test_omitted_background_sha256_is_computed_and_matches_a_supplied_one(tmp_path):
    scene, *_ = synthetic_portable_trex_inputs(tmp_path)
    omitted = _scene_without_hash(scene)
    expected = _digest(scene.trilegal_cache)
    assert omitted.background_sha256 == expected == scene.background_sha256
    assert omitted.to_dict() == scene.to_dict()  # recorded exactly as a supplied digest is


def test_supplied_background_sha256_is_still_verified(tmp_path):
    scene, *_ = synthetic_portable_trex_inputs(tmp_path)
    payload = scene.to_dict()
    payload["background_sha256"] = "0" * 64  # well-formed but wrong
    with pytest.raises(ValueError, match="does not match background_sha256"):
        TargetScene.from_dict(payload)
    payload["background_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="must be a SHA-256 digest"):
        TargetScene.from_dict(payload)


def test_missing_background_area_still_fails_closed(tmp_path):
    scene, *_ = synthetic_portable_trex_inputs(tmp_path)
    payload = scene.to_dict()
    del payload["background_area_sq_deg"]
    with pytest.raises(ValueError, match="background_area_sq_deg"):
        TargetScene.from_dict(payload)


def test_omitted_input_artifacts_hash_the_background_file_and_record_it(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    binding.pop("input_artifacts")
    obj = target.from_scene(_scene_without_hash(scene), **binding)
    recorded = obj._require_scene()["input_artifacts"]
    background = Path(scene.trilegal_cache).absolute()
    assert recorded == {str(background): _digest(background)}


def test_matching_input_artifacts_are_recorded_unchanged(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    obj = target.from_scene(scene, **binding)
    recorded = obj._require_scene()["input_artifacts"]
    assert recorded == {str(Path(name).absolute()): _digest(name) for name in binding["input_artifacts"]}
    assert len(recorded) == 2  # calibration file and background, exactly as supplied


def test_mismatching_input_artifacts_are_rejected(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    name = next(iter(binding["input_artifacts"]))
    wrong = {**binding, "input_artifacts": {**binding["input_artifacts"], name: "0" * 64}}
    with pytest.raises(ValueError, match="missing or changed"):
        target.from_scene(scene, **wrong)
    with pytest.raises(ValueError, match="input_artifacts"):
        target.from_scene(scene, **{**binding, "input_artifacts": ["not", "a", "mapping"]})


def test_omitted_input_artifacts_reject_a_background_changed_after_construction(tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    binding.pop("input_artifacts")
    scene.trilegal_cache.write_text("changed after the scene was built\n")
    with pytest.raises(ValueError, match="changed"):
        target.from_scene(scene, **binding)
