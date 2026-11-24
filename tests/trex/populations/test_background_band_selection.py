"""One native-band background selection shared by every observation dataset.

The background prior must not inherit a TESS magnitude cut for a non-TESS
observation. These checks pin the exact band/depth hand calculation, refusal of
an absent band, the producer's resolved-duplicate flag, and the invariant that
two datasets change only their aperture share of the same selected draw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trex.acquisition.lsst_sim import BULGE_POPULATION_HEADER
from trex.evidence.nested_population import PopulationDataset, background_star_population
from trex.populations.background import (
    declared_background_selection,
    load_trilegal_population,
    select_background_population,
)
from trex.populations.field_population import (
    load_bulge_field_population,
    unresolved_population,
)
from trex.populations.priors import background_population_expectation

_TRILEGAL = (
    "TESS,Mact,logg,logTe,[M/H],Av,m-M0,J,H,Ks\n"
    "9.0,0.8,4.5,3.70,0.0,0.03,10.0,10.0,11.5,11.0\n"
    "13.0,0.7,4.6,3.65,-0.2,0.05,10.0,12.0,12.5,12.0\n"
    "14.0,0.6,4.7,3.60,0.1,0.07,10.0,13.0,13.5,13.0\n"
    "15.0,0.5,4.8,3.55,0.2,0.09,10.0,,14.5,14.0\n"
    "#TRILEGAL\n"
)


def _population(tmp_path, target_tess_mag=None):
    path = tmp_path / "trilegal.csv"
    path.write_text(_TRILEGAL, encoding="utf-8")
    return load_trilegal_population(
        path, target_tess_mag, extra_bands={"J": "J", "H": "H", "Ks": "Ks"}
    )


def _fraction(delta_mag):
    return 10.0 ** (-0.4 * delta_mag) / (1.0 + 10.0 ** (-0.4 * delta_mag))


def test_native_selection_is_hand_counted_and_excludes_unknown_cells(tmp_path):
    population = _population(tmp_path)
    selection = select_background_population(population, band="J", depth_mag=11.5)

    np.testing.assert_array_equal(selection.selected_indices, [1, 2])
    assert selection.n_selected == 2
    assert selection.n_invalid == 1  # row 3 has an absent J cell, retained but not selectable
    assert selection.band == "J" and selection.depth_mag == 11.5
    # Row 0 is brighter than TESS 12 but still a valid J-band background star;
    # it exists only because the loader was not given a TESS cut.
    assert population["tess_mag"].size == 4
    assert select_background_population(population, band="J", depth_mag=9.5).n_selected == 3
    # Hand expectation: two selected stars over the retained 0.1 deg^2 field.
    expected = 2 / (0.1 * 3600.0**2) * math.pi * 10.0**2
    assert background_population_expectation(2, 0.1, 10.0) == pytest.approx(expected)


def test_absent_band_refuses_instead_of_falling_back_to_tess(tmp_path):
    population = _population(tmp_path)
    with pytest.raises(ValueError, match="does not retain band"):
        select_background_population(population, band="Y", depth_mag=11.5)
    with pytest.raises(ValueError, match="does not retain band"):
        background_star_population(
            population, 0.1, 12.0, resolution_limit_arcsec=10.0,
            datasets=[PopulationDataset("y-band", "Y", target_magnitude=11.5)],
        )


def test_selection_band_must_be_an_observed_dataset_band(tmp_path):
    population = _population(tmp_path)
    with pytest.raises(ValueError, match="not declared by any observation dataset"):
        select_background_population(
            population, band="H", depth_mag=11.0,
            datasets=[PopulationDataset("j-band", "J", target_magnitude=11.5)],
        )
    with pytest.raises(ValueError, match="requires the dataset target magnitude"):
        select_background_population(
            population, datasets=[PopulationDataset("j-band", "J", target_magnitude=None)],
        )


def test_resolved_duplicate_is_retained_but_not_selected(tmp_path):
    population = dict(_population(tmp_path))
    population["resolved_match"] = np.array([False, True, False, False])
    population["unresolved_mask"] = ~population["resolved_match"]
    selection = select_background_population(population, band="J", depth_mag=11.5)
    np.testing.assert_array_equal(selection.selected_indices, [2])
    assert selection.resolved_excluded == 1
    assert population["tess_mag"].size == 4  # no row was deleted


def test_two_datasets_share_one_selected_draw_and_only_change_shares(tmp_path):
    population = _population(tmp_path)
    block = background_star_population(
        population, 0.1, 12.0, resolution_limit_arcsec=10.0,
        selection_band="J",
        datasets=[
            PopulationDataset("j-band", "J", target_magnitude=11.5),
            PopulationDataset("h-band", "H", target_magnitude=11.0),
        ],
        completeness="not modeled: TEST completeness statement",
    )
    assert block.selection_band == "J"
    assert block.selection_depth_mag == 11.5
    assert block.completeness == "not modeled: TEST completeness statement"
    expected_weight = 2 / (0.1 * 3600.0**2) * math.pi * 10.0**2
    assert block.log_occurrence_weight == pytest.approx(math.log(expected_weight), rel=0, abs=1e-15)
    # A native-band pool withholds the legacy TESS column so a TESS-only
    # consumer refuses instead of silently selecting on it.
    assert np.isnan(block.tess_mag).all()
    np.testing.assert_array_equal(block.selection_mag, [12.0, 13.0])

    first = block.resolve(np.array([0.0, 0.5]))   # selected index 0: J=12
    second = block.resolve(np.array([0.0, 0.5]))
    assert dict(first.dataset_dilutions)["j-band"] == pytest.approx(_fraction(12.0 - 11.5))
    assert dict(first.dataset_dilutions)["h-band"] == pytest.approx(_fraction(12.5 - 11.0))
    assert first.dataset_dilutions == second.dataset_dilutions
    # The primary legacy flux fraction now follows the declared selection band.
    assert first.flux_fraction == pytest.approx(_fraction(12.0 - 11.5))


def test_mc_shared_draw_is_identical_when_only_the_second_band_depth_changes(tmp_path):
    from trex.evidence.mc_population import _draw_background_star

    population = _population(tmp_path)

    def draw(second_depth):
        dilutions, variates = [], []
        result = _draw_background_star(
            np.random.default_rng(7), 24, population, 0.1, None, None, None,
            resolution_limit_arcsec=10.0,
            datasets=[
                PopulationDataset("j-band", "J", target_magnitude=11.5),
                PopulationDataset("h-band", "H", target_magnitude=second_depth),
            ],
            dataset_dilutions_out=dilutions, variates_out=variates,
            selection_band="J",
        )
        return result, dilutions[0], variates[0]

    first, first_dilutions, first_variates = draw(11.0)
    second, second_dilutions, second_variates = draw(12.0)

    np.testing.assert_array_equal(first_variates[0], second_variates[0])
    np.testing.assert_array_equal(first_variates[1], second_variates[1])
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first_dilutions["j-band"], second_dilutions["j-band"])
    assert np.any(first_dilutions["h-band"] != second_dilutions["h-band"])
    assert set(np.unique(first_variates[0])) == {1, 2}  # only J-selected stars


def test_legacy_tess_draw_is_bit_for_bit_the_old_uniform_table(tmp_path):
    from trex.evidence.mc_population import _draw_background_star

    population = _population(tmp_path, target_tess_mag=12.0)
    n_draws = 32
    variates = []
    actual = _draw_background_star(
        np.random.default_rng(5), n_draws, population, 0.1, 12.0, None, None,
        resolution_limit_arcsec=10.0, variates_out=variates,
    )

    rng = np.random.default_rng(5)
    index = rng.integers(0, 3, n_draws)
    x_separation = rng.random(n_draws)
    delta = population["tess_mag"][index] - 12.0
    ratio = 10.0 ** (-0.4 * delta)
    expected_fraction = ratio / (1.0 + ratio)
    expectation = background_population_expectation(3, 0.1, 10.0)

    np.testing.assert_array_equal(variates[0][0], index)
    np.testing.assert_array_equal(variates[0][1], x_separation)
    np.testing.assert_array_equal(actual[0], expected_fraction)
    np.testing.assert_array_equal(actual[1], np.full(n_draws, math.log(expectation)))
    np.testing.assert_array_equal(actual[2], population["mass_solar"][index])


def test_bulge_field_population_selects_on_its_native_gaia_g_depth(tmp_path):
    rows = [
        ["10.0", "-60.0", "16.0", "17.0", "15.5", "0.8", "4.5", "3.70",
         "0.0", "0.1", "14.0", "1", "1", "16.0", "", "1"],   # resolved duplicate
        ["10.1", "-60.1", "22.5", "24.0", "21.0", "0.7", "1.0", "3.65",
         "-0.2", "0.2", "15.0", "1", "1", "", "", "0"],
        ["10.2", "-60.2", "19.0", "19.5", "18.2", "0.6", "4.7", "3.60",
         "0.1", "0.3", "14.5", "1", "1", "19.0", "relation", "0"],
    ]
    path = tmp_path / "bulge.csv"
    path.write_text(
        ",".join(BULGE_POPULATION_HEADER) + "\n"
        + "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    population = load_bulge_field_population(path)
    np.testing.assert_array_equal(population["band_magnitudes"]["Gaia/G"], [16.0, 22.5, 19.0])
    assert population["bands"]["Gaia/G"]["status"] == "retained"
    assert population["radius_solar"][1] > 2.0  # simulated logg decides size, not a dwarf relation

    selection = select_background_population(
        population, band="Gaia/G", depth_mag=21.0,
        completeness="not modeled: Gaia G query limit is not a correction",
    )
    np.testing.assert_array_equal(selection.selected_indices, [1])
    assert selection.resolved_excluded == 1
    assert selection.completeness.startswith("not modeled")

    unresolved = unresolved_population(population)
    np.testing.assert_array_equal(unresolved["band_magnitudes"]["Gaia/G"], [22.5, 19.0])
    with pytest.raises(ValueError, match="does not retain band"):
        select_background_population(population, band="Roman/WFI07.F146", depth_mag=24.0)

    block = background_star_population(
        population, 0.1, None, resolution_limit_arcsec=10.0,
        selection_band="Gaia/G", completeness="not modeled: Gaia G query limit",
        datasets=[PopulationDataset("gaia-g", "Gaia/G", target_magnitude=21.0)],
    )
    assert np.isnan(block.tess_mag).all()
    assert block.resolved_excluded == 1
    draw = block.resolve(np.array([0.0, 0.5]))
    assert draw.supported
    assert draw.flux_fraction == pytest.approx(_fraction(22.5 - 21.0))
    assert draw.mh_dex == pytest.approx(-0.2) and draw.av_mag == pytest.approx(0.2)


def test_scene_population_binding_keeps_the_native_band_declaration(tmp_path):
    from tests.fixtures.joint_observations import joint_case

    scene, _, _ = joint_case(tmp_path)
    assert scene.population_dataset_bands()[0] == {
        "dataset_id": "space", "band": "synthetic/space", "target_magnitude": 10.0,
    }
    assert scene.dataset("space")["band"]["population_column"] == "TESS"
    assert scene.dataset("ground")["band"]["population_column"] == "RED"


def test_canonical_tess_band_id_takes_the_legacy_cut_not_a_native_column(tmp_path):
    population = _population(tmp_path)
    selection = select_background_population(
        population, band="TESS/TESS.T", target_tess_mag=12.5)
    assert selection.band == "TESS" and selection.source == "legacy-tess"
    np.testing.assert_array_equal(selection.selected_indices, [1, 2, 3])

    dataset = PopulationDataset("tess", "TESS/TESS.T", target_magnitude=12.5)
    from_dataset = select_background_population(population, datasets=[dataset])
    np.testing.assert_array_equal(from_dataset.selected_indices, selection.selected_indices)
    assert from_dataset.source == "legacy-tess"


def test_undeclared_multi_band_datasets_refuse_independently_of_order(tmp_path):
    population = _population(tmp_path)
    j = PopulationDataset("j-band", "J", target_magnitude=11.5)
    h = PopulationDataset("h-band", "H", target_magnitude=11.0)
    for datasets in ([j, h], [h, j]):
        with pytest.raises(ValueError, match="ambiguous"):
            select_background_population(population, datasets=datasets)
        with pytest.raises(ValueError, match="ambiguous"):
            declared_background_selection(population, datasets=datasets)


def test_population_declared_band_and_depth_ignore_dataset_order(tmp_path):
    base = _population(tmp_path)
    population = dict(base)
    population["selection_band"] = "J"
    population["selection_depth_mag"] = 11.5
    population["selection_completeness"] = "not modeled: TEST statement"
    j = PopulationDataset("j-band", "J", target_magnitude=11.5)
    h = PopulationDataset("h-band", "H", target_magnitude=11.0)

    blocks = []
    for datasets in ([j, h], [h, j]):
        declared = declared_background_selection(population, datasets=datasets)
        assert (declared.band, declared.depth_mag) == ("J", 11.5)
        assert declared.completeness == "not modeled: TEST statement"
        block = background_star_population(
            population, 0.1, None, resolution_limit_arcsec=10.0,
            datasets=datasets, selection_band=declared.band,
            selection_magnitude=declared.depth_mag,
            completeness=declared.completeness,
        )
        blocks.append(block)
    for attribute in ("selection_band", "selection_depth_mag", "completeness",
                      "log_occurrence_weight"):
        assert getattr(blocks[0], attribute) == getattr(blocks[1], attribute)
    np.testing.assert_array_equal(blocks[0].selection_mag, blocks[1].selection_mag)
    np.testing.assert_array_equal(blocks[0].selection_mag, [12.0, 13.0])

    with pytest.raises(ValueError, match="differs from the population's declared"):
        select_background_population(population, band="H", datasets=[j, h])


def test_native_only_roman_cache_needs_a_native_selection_and_never_tess(tmp_path):
    path = tmp_path / "roman2024.csv"
    path.write_text(
        "F062,F087,F106,F129,F146,F158,F184,F213,Mact,logg,logTe,[M/H],Av\n"
        "20.0,19.5,19.0,18.5,18.0,17.5,17.0,16.5,0.8,4.5,3.70,0.0,0.03\n"
        "21.0,20.5,20.0,19.5,19.0,18.5,18.0,17.5,0.7,4.6,3.65,-0.2,0.05\n"
        "#TRILEGAL\n",
        encoding="utf-8",
    )
    population = load_trilegal_population(
        path, None, extra_bands={"Roman/WFI07.F146": "F146"})
    assert np.isnan(population["tess_mag"]).all()
    assert population["bands"]["Roman/WFI07.F146"]["column"] == "F146"
    selection = select_background_population(
        population, band="Roman/WFI07.F146", depth_mag=18.5)
    np.testing.assert_array_equal(selection.selected_indices, [1])
    assert selection.band == "Roman/WFI07.F146"

    # A TESS cut cannot be invented for a cache that has no TESS column, and a
    # TESS-only consumer refuses the blocked legacy path rather than falling back.
    with pytest.raises(ValueError, match="no TESS column"):
        load_trilegal_population(path, 20.0)
    with pytest.raises(ValueError, match="finite retained tess_mag"):
        select_background_population(population, target_tess_mag=20.0)
    with pytest.raises(ValueError, match="finite retained tess_mag"):
        background_star_population(
            population, 0.1, 20.0, resolution_limit_arcsec=10.0)


def test_mc_shared_draw_ignores_dataset_order_with_a_declared_scene_band(tmp_path):
    from trex.evidence.mc_population import _draw_background_star

    population = dict(_population(tmp_path))
    population["selection_band"] = "J"
    population["selection_depth_mag"] = 11.5
    j = PopulationDataset("j-band", "J", target_magnitude=11.5)
    h = PopulationDataset("h-band", "H", target_magnitude=11.0)

    def draw(datasets):
        variates = []
        result = _draw_background_star(
            np.random.default_rng(11), 16, population, 0.1, None, None, None,
            resolution_limit_arcsec=10.0, datasets=datasets,
            variates_out=variates,
        )
        return result, variates[0]

    first, first_variates = draw([j, h])
    second, second_variates = draw([h, j])
    np.testing.assert_array_equal(first_variates[0], second_variates[0])
    np.testing.assert_array_equal(first_variates[1], second_variates[1])
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
