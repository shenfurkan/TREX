"""The escalation gate binds one declared background band, never dataset order.

Production builders once omitted ``selection_band``, so a joint scene followed
its first dataset's band and a canonical ``TESS/TESS.T`` id
missed the legacy path. These checks exercise the real gate with a native
population, a non-finite TESS magnitude and reversed dataset order.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trex.evidence.evidence_escalation_calls import _population_blocks


def _population(tmp_path, *, declared=True):
    path = tmp_path / "trilegal.csv"
    path.write_text(
        "TESS,Mact,logg,logTe,[M/H],Av,J,H\n"
        "13.0,0.8,4.5,3.70,0.0,0.03,12.0,11.5\n"
        "14.0,0.7,4.6,3.65,-0.2,0.05,13.0,12.5\n"
        "#TRILEGAL\n",
        encoding="utf-8",
    )
    from trex.populations.background import load_trilegal_population

    population = load_trilegal_population(path, None, extra_bands={"J": "J", "H": "H"})
    if declared:
        population["selection_band"] = "J"
        population["selection_depth_mag"] = 11.5
        population["selection_completeness"] = "not modeled: TEST statement"
    return population


def _joint(datasets):
    return SimpleNamespace(population_datasets=tuple(datasets))


def _blocks(population, datasets, target_tess_mag=None):
    return _population_blocks(
        M_s_Msun=1.0, R_s_Rsun=1.0, plx=None,
        contrast_separations=None, contrast_values=None,
        resolution_limit_arcsec=10.0,
        background_population=population, background_area_sq_deg=0.1,
        target_tess_mag=target_tess_mag,
        # Joint scenes must carry band light; only the bound block (skipped: plx=None) reads it.
        stellar_light=object(),
        joint_observations=_joint(datasets),
    )["background"]


def test_native_scene_needs_no_tess_magnitude_and_order_cannot_change_the_block(tmp_path):
    population = _population(tmp_path)
    j = {"dataset_id": "j", "band": "J", "target_magnitude": 11.5}
    h = {"dataset_id": "h", "band": "H", "target_magnitude": 11.0}

    forward = _blocks(population, [j, h])
    reverse = _blocks(population, [h, j])

    assert forward is not None and reverse is not None
    assert forward.selection_band == reverse.selection_band == "J"
    assert forward.selection_depth_mag == reverse.selection_depth_mag == 11.5
    assert forward.completeness == reverse.completeness == "not modeled: TEST statement"
    np.testing.assert_array_equal(forward.selection_mag, reverse.selection_mag)
    assert forward.log_occurrence_weight == reverse.log_occurrence_weight


def test_undeclared_multi_band_joint_scene_refuses_instead_of_following_order(tmp_path):
    population = _population(tmp_path, declared=False)
    datasets = [
        {"dataset_id": "j", "band": "J", "target_magnitude": 11.5},
        {"dataset_id": "h", "band": "H", "target_magnitude": 11.0},
    ]
    for order in (datasets, list(reversed(datasets))):
        with pytest.raises(ValueError, match="ambiguous"):
            _blocks(population, order)


def test_no_declared_depth_leaves_the_scenario_unavailable_without_a_tess_substitute(tmp_path):
    population = _population(tmp_path, declared=False)
    assert _blocks(population, [], target_tess_mag=None) is None
    assert _blocks(population, [], target_tess_mag=12.0) is not None
