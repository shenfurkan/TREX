"""Bind a measured TESS contrast curve through ``from_scene``, without evidence runs.

The curve direction is tested against the definition, not against the engine:
the imaging weight compares a companion's delta magnitude,
``-2.5 log10(F_companion/F_target)`` [mag], with the curve limit interpolated at
its separation [arcsec] (Giacalone et al. 2021, AJ 161, 24, DOI
10.3847/1538-3881/abc6af, Sect. 2.4.3). All inputs are synthetic.
"""

import numpy as np
import pytest

from trex.app.scene import MEASURED_CONTRAST_CURVE_KIND, TargetScene
from trex.app.target import target
from trex.foundation.physical_constants import SECONDS_PER_DAY
from trex.populations.companion_imaging import imaging_log_prior

CURVE_SEPARATIONS = np.array([0.5, 2.0])
CURVE_MAGNITUDES = np.array([3.0, 5.0])


def _scene(tmp_path):
    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    return TargetScene(
        tic_id=100001, ra_deg=120.0, dec_deg=-30.0,
        M_s_Msun=1.02, R_s_Rsun=0.98, Teff_K=5778.0, Tmag=10.0, plx_mas=10.0,
        sectors=[1], contrast_separations=None, contrast_values=None,
        unresolved_resolution_limit_arcsec=2.2,
        target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.90}],
        resolved_neighbors=[{
            "source_id": "100002", "M_s": 0.8, "R_s": 0.8, "delta_mag": 1.5,
            "separation_arcsec": 4.0,
            "aperture_flux_fractions": [{"sector": 1, "fraction": 0.10}],
        }],
        N_background=1, trilegal_cache=background, background_area_sq_deg=0.1,
    )


def _binding():
    return dict(sector_weights={1: 1.0}, depth_ppm=8000.0,
                exptime_days=120.0 / SECONDS_PER_DAY, random_seed=7)


def _bound(tmp_path, **curve):
    return target.from_scene(_scene(tmp_path), **_binding(), **curve)


def test_from_scene_curve_changes_the_companion_imaging_weight(tmp_path):
    bound = _bound(tmp_path, contrast_curve=(CURVE_SEPARATIONS, CURVE_MAGNITUDES),
                   contrast_band="TESS")
    scene = bound._scene_binding["scene"]
    np.testing.assert_array_equal(scene.contrast_separations, CURVE_SEPARATIONS)
    np.testing.assert_array_equal(scene.contrast_values, CURVE_MAGNITUDES)

    separation = np.array([1.0])
    limit = float(np.interp(separation[0], CURVE_SEPARATIONS, CURVE_MAGNITUDES))
    bright_ratio = 10.0 ** (-0.4 * 1.0)   # delta mag = 1.0, brighter than the limit
    faint_ratio = 10.0 ** (-0.4 * 4.0)    # delta mag = 4.0, fainter than the limit
    assert 1.0 < limit < 4.0

    with_curve = imaging_log_prior(np.array([bright_ratio]), separation,
                                   scene.contrast_separations, scene.contrast_values, np.zeros(1))
    without_curve = imaging_log_prior(np.array([bright_ratio]), separation, None, None, np.zeros(1))
    faint_with_curve = imaging_log_prior(np.array([faint_ratio]), separation,
                                         scene.contrast_separations, scene.contrast_values, np.zeros(1))

    assert np.isneginf(with_curve).all()
    assert without_curve[0] == 0.0
    assert faint_with_curve[0] == 0.0


def test_curve_file_records_source_band_and_limitation(tmp_path):
    curve_path = tmp_path / "contrast.csv"
    curve_path.write_text(
        "# synthetic ExoFOP-style table\nsep_arcsec,dmag\n0.5,3.0\n2.0,5.0\n",
        encoding="utf-8",
    )
    scene = _bound(tmp_path, contrast_curve=curve_path, contrast_band="tess")._scene_binding["scene"]
    provenance = scene.contrast_provenance
    assert provenance["kind"] == MEASURED_CONTRAST_CURVE_KIND
    assert provenance["source"] == str(curve_path)
    assert provenance["band"] == "TESS"
    assert "measured TESS-band contrast curve" in provenance["limitation"]
    assert "remains the independent" in provenance["limitation"]
    assert "bounded by the Gaia DR2 close-pair limit" not in provenance["limitation"]

    restored = TargetScene.from_dict(scene.to_dict())
    assert restored.contrast_provenance == provenance
    np.testing.assert_array_equal(restored.contrast_separations, CURVE_SEPARATIONS)
    np.testing.assert_array_equal(restored.contrast_values, CURVE_MAGNITUDES)


@pytest.mark.parametrize("curve", [
    ([1.0], [3.0]),
    ([0.5, 2.0], [3.0]),
    ([2.0, 0.5], [3.0, 5.0]),
    ([0.5, 0.5], [3.0, 5.0]),
    ([0.0, 1.0], [3.0, 5.0]),
    ([0.5, np.nan], [3.0, 5.0]),
    ([0.5, 1.0], [3.0, np.inf]),
])
def test_malformed_curve_arrays_are_refused(tmp_path, curve):
    with pytest.raises(ValueError):
        _bound(tmp_path, contrast_curve=curve, contrast_band="TESS")


@pytest.mark.parametrize("text", ["", "1.0\n", "sep dmag\n", "# comment only\n"])
def test_malformed_curve_files_are_refused(tmp_path, text):
    curve_path = tmp_path / "contrast.csv"
    curve_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        _bound(tmp_path, contrast_curve=curve_path, contrast_band="TESS")


def test_missing_curve_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="existing regular file"):
        _bound(tmp_path, contrast_curve=tmp_path / "missing.csv", contrast_band="TESS")


@pytest.mark.parametrize("band", ["J", "Kepler", "R"])
def test_non_tess_band_is_refused(tmp_path, band):
    with pytest.raises(ValueError, match="TESS band only"):
        _bound(tmp_path, contrast_curve=(CURVE_SEPARATIONS, CURVE_MAGNITUDES), contrast_band=band)


def test_contrast_band_is_required_with_a_curve(tmp_path):
    with pytest.raises(ValueError, match="contrast_band is required"):
        _bound(tmp_path, contrast_curve=(CURVE_SEPARATIONS, CURVE_MAGNITUDES))
    with pytest.raises(ValueError, match="contrast_band requires contrast_curve"):
        _bound(tmp_path, contrast_band="TESS")


def test_calc_probs_file_argument_points_to_the_scene_binding(tmp_path):
    bound = _bound(tmp_path)
    with pytest.raises(NotImplementedError, match="from_tic.*from_scene"):
        bound.calc_probs(np.array([-0.1, 0.1]), np.array([1.0, 0.99]), 1e-3, 3.0,
                         contrast_curve_file=str(tmp_path / "curve.csv"))
