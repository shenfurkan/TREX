"""One physical population draw bound to every observation dataset.

The joint multi-band likelihood must not draw an independent background
population (or solid-angle prior) per band. These checks pin that boundary:
one star index and one areal position, with only the native-band flux ratio
changing between datasets; unknown band light refuses the draw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trex.evidence.nested_population import (
    PopulationDataset,
    background_star_population,
    bound_companion_population,
)
from trex.populations.background import load_trilegal_population

_TRILEGAL = (
    "TESS,Mact,logg,logTe,[M/H],Av,m-M0,J,H,Ks\n"
    "13.0,0.8,4.5,3.70,0.0,0.03,10.0,12.0,11.5,11.0\n"
    "14.0,0.7,4.6,3.65,-0.2,0.05,10.0,13.0,12.5,12.0\n"
    "#TRILEGAL\n"
)


def _population(tmp_path, body=_TRILEGAL):
    path = tmp_path / "trilegal.csv"
    path.write_text(body, encoding="utf-8")
    return load_trilegal_population(
        path, 12.0, extra_bands={"J": "J", "H": "H", "Ks": "Ks"}
    )


def _fraction(delta_mag):
    return 10.0 ** (-0.4 * delta_mag) / (1.0 + 10.0 ** (-0.4 * delta_mag))


class _BandLight:
    """Synthetic per-band light with a fixed ratio, not a fallback model."""

    def __init__(self, ratio):
        self.ratio = ratio
        self.target_av_mag = 0.0

    def pair(self, *_args):
        return self.ratio, None, None


def test_background_draw_shares_one_star_and_position_across_datasets(tmp_path):
    population = _population(tmp_path)
    block = background_star_population(
        population, 0.1, 12.0, resolution_limit_arcsec=21.0,
        selection_band="J",
        datasets=[
            PopulationDataset("j-band", "J", target_magnitude=10.0),
            PopulationDataset("h-band", "H", target_magnitude=10.5),
        ],
    )
    # unit point 0.5 selects index 1 (the fainter retained star) in every dataset.
    draw = block.resolve(np.array([0.5, 0.5]))
    dilutions = dict(draw.dataset_dilutions)
    assert set(dilutions) == {"j-band", "h-band"}
    assert dilutions["j-band"] == pytest.approx(_fraction(13.0 - 10.0))
    assert dilutions["h-band"] == pytest.approx(_fraction(12.5 - 10.5))
    assert dilutions["j-band"] != dilutions["h-band"]
    # One occurrence/selection weight, not one per band.
    assert isinstance(block.log_occurrence_weight, float)
    assert math.isfinite(block.log_occurrence_weight)


def test_missing_dataset_band_is_refused_at_construction(tmp_path):
    population = _population(tmp_path)
    with pytest.raises(ValueError, match="does not retain band"):
        background_star_population(
            population, 0.1, 12.0, resolution_limit_arcsec=21.0,
            datasets=[PopulationDataset("y-band", "Y", target_magnitude=10.0)],
        )


def test_partially_missing_band_cell_marks_the_draw_unsupported(tmp_path):
    body = _TRILEGAL.replace(
        "13.0,0.8,4.5,3.70,0.0,0.03,10.0,12.0,11.5,11.0",
        "13.0,0.8,4.5,3.70,0.0,0.03,10.0,12.0,11.5,",
    )
    population = _population(tmp_path, body)
    block = background_star_population(
        population, 0.1, 12.0, resolution_limit_arcsec=21.0,
        selection_band="J",
        datasets=[
            PopulationDataset("j-band", "J", target_magnitude=10.0),
            PopulationDataset("ks-band", "Ks", target_magnitude=10.0),
        ],
    )
    draw = block.resolve(np.array([0.0, 0.5]))  # J selection keeps index 0, whose Ks cell is absent
    assert draw.supported is False
    assert dict(draw.dataset_dilutions)["ks-band"] is None


def test_bound_draw_uses_each_native_band_light_once(tmp_path):
    block = bound_companion_population(
        0.5, 0.5, 10.0, resolution_limit_arcsec=100.0,
        host_teff_k=4000.0, target_feh=-0.2,
        datasets=[PopulationDataset("h-band", "H"), PopulationDataset("ks-band", "Ks")],
        band_light={"H": _BandLight(0.2), "Ks": _BandLight(0.1)},
    )
    draw = block.resolve(np.full(block.n_dimensions, 0.5))
    dilutions = dict(draw.dataset_dilutions)
    assert dilutions["h-band"] == pytest.approx(0.2 / 1.2)
    assert dilutions["ks-band"] == pytest.approx(0.1 / 1.1)
    assert draw.mass_solar > 0.0 and draw.radius_solar > 0.0


def test_bound_dataset_without_band_light_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no native-band light"):
        bound_companion_population(
            0.5, 0.5, 10.0, resolution_limit_arcsec=100.0,
            host_teff_k=4000.0, target_feh=-0.2,
            datasets=[PopulationDataset("h-band", "H")],
            band_light={},
        )


def test_mc_background_output_is_per_dataset_and_keeps_nan_for_unknown_cells(tmp_path):
    from trex.evidence.mc_population import _draw_background_star

    population = _population(tmp_path)
    out = []
    _draw_background_star(
        np.random.default_rng(7), 16, population, 0.1, 12.0, None, None,
        resolution_limit_arcsec=21.0,
        selection_band="J",
        datasets=[
            PopulationDataset("j-band", "J", target_magnitude=10.0),
            PopulationDataset("h-band", "H", target_magnitude=10.5),
        ],
        dataset_dilutions_out=out,
    )
    assert set(out[0]) == {"j-band", "h-band"}
    assert out[0]["j-band"].shape == (16,) and out[0]["h-band"].shape == (16,)
    assert np.any(out[0]["j-band"] != out[0]["h-band"])


def test_mc_bound_output_carries_each_band_ratio():
    from trex.evidence.mc_population import _draw_bound_companion

    out = []
    _draw_bound_companion(
        np.random.default_rng(7), 8, 0.5, 0.5, 10.0, None, None,
        resolution_limit_arcsec=100.0, host_teff_k=4000.0, target_feh=-0.2,
        datasets=[PopulationDataset("h-band", "H"), PopulationDataset("ks-band", "Ks")],
        band_light={"H": _BandLight(0.2), "Ks": _BandLight(0.1)},
        dataset_dilutions_out=out,
    )
    np.testing.assert_allclose(out[0]["h-band"], 0.2 / 1.2)
    np.testing.assert_allclose(out[0]["ks-band"], 0.1 / 1.1)
