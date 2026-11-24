"""Dataset-indexed scene context: native bands, apertures and source shares.

The legacy TESS scene is a single sector-weighted view. These checks pin the
multi-observation contract: one common source identity, one share per dataset's
own aperture total, explicit zero shares, and unknown native-band light that is
never replaced by TESS or a solar estimate.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from trex.app.scene import TargetScene

_TRILEGAL = (
    "TESS,Mact,logg,logTe,[M/H],Av,m-M0,J,H,Ks\n"
    "13.0,0.8,4.5,3.70,0.0,0.03,10.0,12.0,11.5,11.0\n"
    "14.0,0.7,4.6,3.65,-0.2,0.05,10.0,13.0,12.5,12.0\n"
    "#TRILEGAL\n"
)


def _cache(tmp_path):
    path = tmp_path / "trilegal.csv"
    path.write_text(_TRILEGAL, encoding="utf-8")
    return path


def _kwargs(tmp_path, **overrides):
    background = _cache(tmp_path)
    kwargs = {
        "tic_id": 123,
        "ra_deg": 90.0,
        "dec_deg": -60.0,
        "M_s_Msun": 1.0,
        "R_s_Rsun": 1.0,
        "Teff_K": 5700.0,
        "Tmag": 10.0,
        "plx_mas": 2.0,
        "sectors": [1],
        "contrast_separations": np.array([0.1, 1.0]),
        "contrast_values": np.array([2.0, 5.0]),
        "target_aperture_flux_fractions": [{"sector": 1, "fraction": 1.0}],
        "resolved_neighbors": [
            {
                "source_id": "gaia-1",
                "M_s": 0.5,
                "R_s": 0.45,
                "delta_mag": 2.0,
                "separation_arcsec": 1.0,
                "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}],
            }
        ],
        "N_background": 1,
        "trilegal_cache": background,
        "background_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
        "background_area_sq_deg": 0.1,
        "target_source_id": "target-1",
    }
    kwargs.update(overrides)
    return kwargs


def _dataset(dataset_id, target_fraction, neighbor_fraction, **overrides):
    record = {
        "dataset_id": dataset_id,
        "band": {"id": "TESS", "unit": "mag", "system": "TESS"},
        "aperture": {"id": "spoc-sector-1", "kind": "pixel-mask", "denominator": "aperture-total"},
        "shares": {"target-1": target_fraction, "gaia-1": neighbor_fraction},
        "magnitudes": {
            "target-1": {"value": 10.0, "uncertainty": 0.02},
            "gaia-1": {"value": 12.0, "uncertainty": 0.05},
        },
    }
    record.update(overrides)
    return record


def test_legacy_scene_round_trip_is_unchanged(tmp_path):
    kwargs = _kwargs(tmp_path)
    kwargs.pop("target_source_id")
    scene = TargetScene(**kwargs)
    payload = scene.to_dict()
    assert "datasets" not in payload and "target_identity" not in payload
    rebuilt = TargetScene.from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert rebuilt.target_aperture_flux_fraction({1: 1.0}) == 1.0


def test_datasets_keep_common_identity_and_their_own_aperture_denominator(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[
            _dataset("tess-s1", 0.90, 0.10),
            _dataset(
                "lco-h",
                0.60,
                0.0,
                band={"id": "H", "unit": "mag", "system": "WIRC"},
                aperture={"id": "lco-aperture", "kind": "aperture-photometry",
                          "denominator": "aperture-total", "denominator_value": 1.25e6},
            ),
        ],
    ))
    assert scene.dataset_ids == ("tess-s1", "lco-h")
    assert scene.dataset_target_flux_fraction("tess-s1") == 0.90
    assert scene.dataset_target_flux_fraction("lco-h") == 0.60
    # The H dataset gives the neighbour an exact zero share; the host remains.
    assert scene.dataset_flux_fraction("lco-h", "gaia-1") == 0.0
    assert scene.n_neighbors == 1
    entries = scene.neighbor_dicts_for_evidence_by_dataset("lco-h")
    assert len(entries) == 1 and entries[0]["aperture_flux_fraction"] == 0.0
    assert entries[0]["aperture_flux_fraction_in_aperture"] is False
    assert entries[0]["band"]["id"] == "H"


def test_zero_share_is_not_a_removed_source(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[_dataset("flat", 1.0, 0.0)],
    ))
    assert scene.dataset_flux_fraction("flat", "gaia-1") == 0.0
    assert scene.n_neighbors == 1


def test_per_cadence_shares_survive_round_trip(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[_dataset(
            "changing",
            [0.80, 0.70, 0.65],
            [0.10, 0.20, 0.25],
            cadence_ids=["c0", "c1", "c2"],
        )],
    ))
    np.testing.assert_allclose(scene.dataset_flux_fraction("changing", "gaia-1"),
                               [0.10, 0.20, 0.25])
    assert scene.dataset_flux_fraction("changing", "gaia-1", cadence=2) == 0.25
    with pytest.raises(ValueError, match="cadence"):
        scene.neighbor_dicts_for_evidence_by_dataset("changing")
    assert scene.neighbor_dicts_for_evidence_by_dataset("changing", cadence=1)[0][
        "aperture_flux_fraction"] == 0.20
    rebuilt = TargetScene.from_dict(scene.to_dict())
    np.testing.assert_allclose(rebuilt.dataset_flux_fraction("changing", "gaia-1"),
                               [0.10, 0.20, 0.25])


def test_outside_aperture_is_an_explicit_zero(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[{
            "dataset_id": "h",
            "band": {"id": "H", "unit": "mag"},
            "aperture": {"id": "a", "kind": "pixel-mask", "denominator": "aperture-total"},
            "shares": {"target-1": 1.0},
            "outside_aperture": ["gaia-1"],
        }],
    ))
    assert scene.dataset_flux_fraction("h", "gaia-1") == 0.0


def test_unstated_source_share_is_refused_not_zero(tmp_path):
    with pytest.raises(ValueError, match="does not state the aperture share"):
        TargetScene(**_kwargs(
            tmp_path,
            datasets=[{
                "dataset_id": "h",
                "band": {"id": "H", "unit": "mag"},
                "aperture": {"id": "a", "kind": "pixel-mask", "denominator": "aperture-total"},
                "shares": {"target-1": 1.0},
            }],
        ))


def test_inconsistent_denominator_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sum above one"):
        TargetScene(**_kwargs(
            tmp_path,
            datasets=[_dataset("bad", 0.95, 0.10)],
        ))


def test_unknown_native_band_light_stays_unknown(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[{
            "dataset_id": "h",
            "band": {"id": "H", "unit": "mag", "system": "WIRC"},
            "aperture": {"id": "a", "kind": "pixel-mask", "denominator": "aperture-total"},
            "shares": {"target-1": 0.8, "gaia-1": 0.0},
            "magnitudes": {
                "target-1": {"value": 9.5, "uncertainty": 0.03},
                "gaia-1": {"value": None, "unavailable_reason": "no WIRC H detection"},
            },
        }],
    ))
    assert scene.dataset_magnitude("h", "target-1") == (9.5, 0.03)
    with pytest.raises(ValueError, match="unavailable"):
        scene.dataset_magnitude("h", "gaia-1")
    assert scene.dataset_magnitude_status("h", "gaia-1")["status"] == "unavailable"


def test_not_reported_magnitude_is_distinct_from_unavailable(tmp_path):
    scene = TargetScene(**_kwargs(tmp_path, datasets=[_dataset("h", 0.9, 0.0)]))
    assert scene.dataset_magnitude_status("h", "gaia-1")["status"] == "measured"
    with pytest.raises(ValueError, match="does not report"):
        scene.dataset_magnitude("h", "unknown-source")


def test_background_population_retains_dataset_bands(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        datasets=[_dataset(
            "h", 0.9, 0.0,
            band={"id": "H", "unit": "mag", "system": "2MASS",
                  "population_column": "H"},
        )],
    ))
    population = scene.background_population()
    np.testing.assert_allclose(population["band_magnitudes"]["H"], [11.5, 12.5])
    assert population["bands"]["H"]["unit"] == "mag"
    assert "missing_bands" in population
    assert scene.population_dataset_bands() == [
        {"dataset_id": "h", "band": "H", "target_magnitude": 10.0}
    ]


def test_non_tic_scene_keeps_identity_and_missing_tess_light(tmp_path):
    kwargs = _kwargs(tmp_path)
    kwargs.update(
        tic_id=None,
        Tmag=None,
        plx_mas=None,
        target_identity={
            "namespace": "plato-pic",
            "native_id": "PIC 123.45",
            "frame": "ICRS",
            "epoch_jyear": None,
            "cross_identifiers": [],
        },
    )
    kwargs.pop("target_source_id")
    scene = TargetScene(**kwargs)
    assert scene.tic_id is None
    assert scene.Tmag is None and scene.plx_mas is None
    assert scene.target_source_id == "PIC 123.45"
    assert scene.target_identity["namespace"] == "plato-pic"
    # The retained TRILEGAL selection is TESS-specific: no target Tmag means
    # no substitute, and the background consumer refuses.
    with pytest.raises(ValueError, match="TESS-specific"):
        scene.background_population()


def test_target_identity_must_match_a_supplied_tic_id(tmp_path):
    kwargs = _kwargs(tmp_path)
    kwargs["target_identity"] = {
        "namespace": "tic", "native_id": "999", "frame": "ICRS",
        "epoch_jyear": None, "cross_identifiers": [],
    }
    with pytest.raises(ValueError, match="does not match"):
        TargetScene(**kwargs)


def test_target_source_id_cannot_collide_with_a_neighbour(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        TargetScene(**_kwargs(tmp_path, target_source_id="gaia-1",
                              datasets=[_dataset("h", 0.9, 0.0)]))


def test_native_band_scene_loads_without_tmag_and_declares_completeness(tmp_path):
    scene = TargetScene(**_kwargs(
        tmp_path,
        Tmag=None,
        datasets=[_dataset(
            "h", 0.9, 0.0,
            band={"id": "H", "unit": "mag", "system": "2MASS",
                  "population_column": "H"},
        )],
        background_completeness="not modeled: TEST completeness statement",
    ))
    population = scene.background_population()
    assert population["selection_band"] == "H"
    assert population["selection_depth_mag"] == 10.0
    assert population["selection_completeness"] == "not modeled: TEST completeness statement"
    np.testing.assert_allclose(population["band_magnitudes"]["H"], [11.5, 12.5])
    # The cache's TESS column stays a convenience axis; the H depth is the
    # selection, not a TESS magnitude substituted for it.
    assert population["tess_mag"].size == 2
    assert np.isfinite(population["tess_mag"]).all()

    payload = scene.to_dict()
    assert payload["background_completeness"] == "not modeled: TEST completeness statement"
    rebuilt = TargetScene.from_dict(payload)
    assert rebuilt.background_population()["selection_band"] == "H"
    assert rebuilt.background_completeness == "not modeled: TEST completeness statement"


def test_multi_band_scene_requires_one_declared_background_band(tmp_path):
    tess = _dataset("tess-s1", 0.9, 0.1)
    h = _dataset(
        "lco-h", 0.6, 0.0,
        band={"id": "H", "unit": "mag", "system": "WIRC", "population_column": "H"},
    )
    with pytest.raises(ValueError, match="ambiguous"):
        TargetScene(**_kwargs(tmp_path, Tmag=None, datasets=[tess, h])).background_population()

    scene = TargetScene(**_kwargs(
        tmp_path, Tmag=None, datasets=[h, tess],
        background_selection_band="H",
        background_completeness="not modeled: TEST completeness statement",
    ))
    population = scene.background_population()
    assert population["selection_band"] == "H"
    assert population["selection_depth_mag"] == 10.0
    assert scene.to_dict()["background_selection_band"] == "H"

    with pytest.raises(ValueError, match="must name one of the scene's dataset bands"):
        TargetScene(**_kwargs(
            tmp_path, Tmag=None, datasets=[tess, h], background_selection_band="Y"))


def test_legacy_tess_dataset_depth_must_match_the_scene_tmag(tmp_path):
    mismatched = _dataset("tess-s1", 0.9, 0.1, magnitudes={
        "target-1": {"value": 11.5, "uncertainty": 0.02},
        "gaia-1": {"value": 12.0, "uncertainty": 0.05},
    })
    with pytest.raises(ValueError, match="TESS magnitude differs"):
        TargetScene(**_kwargs(tmp_path, datasets=[mismatched])).background_population()
