"""TREX TargetScene input contract for synthetic population context."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from trex.app.scene import TargetScene


def _target_scene_kwargs(tmp_path):
    background = tmp_path / "synthetic_trilegal.csv"
    background.write_text("synthetic background population\n", encoding="utf-8")

    return {
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
        "resolved_neighbors": [],
        "N_background": 1,
        "trilegal_cache": background,
        "background_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
        "background_area_sq_deg": 0.1,
    }


def test_target_scene_basic(tmp_path):
    scene = TargetScene(**_target_scene_kwargs(tmp_path))
    assert scene.tic_id == 123
    assert scene.n_neighbors == 0


def test_target_scene_neighbors(tmp_path):
    kwargs = _target_scene_kwargs(tmp_path)
    kwargs["resolved_neighbors"] = [
        {
            "source_id": "synthetic-neighbor",
            "M_s": 0.5,
            "R_s": 0.45,
            "delta_mag": 2.0,
            "separation_arcsec": 1.0,
            "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}],
        }
    ]
    scene = TargetScene(**kwargs)
    assert scene.n_neighbors == 1
    m, r, _ = scene.neighbor_masses_radii()
    assert m[0] == 0.5
    assert r[0] == 0.45


def test_target_scene_rejects_changed_background(tmp_path):
    scene = TargetScene(**_target_scene_kwargs(tmp_path))
    scene.trilegal_cache.write_text("changed background population\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after scene construction"):
        scene.verify_background()
