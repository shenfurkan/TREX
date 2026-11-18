"""Public modules and packaged reference data remain usable without EXONYM."""

import ast
import importlib
import math
from importlib import resources
from pathlib import Path

import pytest

import trex


def test_portable_package_has_no_exonym_imports():
    package = Path(trex.__file__).resolve().parent
    assert (package / "stellar_light" / "host_limb_darkening.py").is_file(), "canonical portable component is missing"
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "exonym" for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not node.module or node.module.split(".")[0] != "exonym", path


def test_public_modules_share_their_canonical_implementations():
    for name in ("scene", "engine", "target", "run", "example"):
        assert importlib.import_module("trex." + name) is importlib.import_module("trex.app." + name)
    from trex import engine, scene, target
    from trex.models import likelihoods

    assert engine.TargetScene is scene.TargetScene
    assert target.target.__module__ == "trex.app.target"
    assert callable(engine.run_trex_vetting)
    assert callable(likelihoods.simulate_TP)
    assert callable(likelihoods.simulate_EB)


def test_reference_grid_and_licenses_are_packaged():
    data = resources.files("trex").joinpath("_data")
    assert data.joinpath("triceratops-1.0.20-ldc_tess.csv").is_file()
    assert data.joinpath("TRICERATOPS-MIT-LICENSE").is_file()
    license_text = data.joinpath("TRICERATOPS-MIT-LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2020 stevengiacalone" in license_text
    assert "Permission is hereby granted, free of charge" in license_text


def test_portable_scene_retains_neighbor_refusal_evidence(tmp_path):
    from trex.app.scene import TargetScene

    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    evaluation = {"123456": [{"source": "catalog", "status": "unavailable", "reason": "missing mass"}]}
    neighbor = {"source_id": "123456", "M_s": None, "R_s": None,
                "stellar_parameter_unavailable_reason": "no stellar measurement",
                "delta_mag": 1., "separation_arcsec": 4.,
                "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}]}
    scene = TargetScene(tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1.,
                        Teff_K=5700., Tmag=10., plx_mas=2., sectors=[1],
                        contrast_separations=[0.1, 1.], contrast_values=[2., 5.],
                        target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.9}],
                        resolved_neighbors=[neighbor], N_background=1, trilegal_cache=background,
                        background_area_sq_deg=0.1, neighbor_parameter_evaluations=evaluation)
    portable = scene.to_dict()
    restored = TargetScene.from_dict(portable)
    assert restored.neighbor_parameter_evaluations == evaluation
    assert restored.neighbor_dicts_for_evidence({1: 1.}) == scene.neighbor_dicts_for_evidence({1: 1.})
    portable["neighbor_parameter_evaluations"]["123456"][0]["reason"] = "changed"
    assert scene.neighbor_parameter_evaluations == evaluation


def test_portable_scene_accepts_a_missing_magnitude_contrast_with_its_reason(tmp_path):
    from trex.app.scene import TargetScene

    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    neighbor = {
        "source_id": "123456", "M_s": 0.8, "R_s": 0.8,
        "delta_mag": None,
        "delta_mag_unavailable_reason": "no measured G; the pixel fit supplies an aperture fraction",
        "separation_arcsec": 4.,
        "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}],
        "aperture_flux_fractions_upper_95": [{"sector": 1, "fraction": 0.12}],
        "aperture_flux_fraction_limitation": "marginal source limit; not additive",
    }
    scene = TargetScene(tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1.,
                        Teff_K=5700., Tmag=10., plx_mas=2., sectors=[1],
                        contrast_separations=[0.1, 1.], contrast_values=[2., 5.],
                        target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.85}],
                        resolved_neighbors=[neighbor], N_background=1, trilegal_cache=background,
                        background_area_sq_deg=0.1)
    restored = TargetScene.from_dict(scene.to_dict())
    assert restored.resolved_neighbors[0]["delta_mag"] is None
    assert restored.resolved_neighbors[0]["delta_mag_unavailable_reason"]
    assert math.isnan(restored.companion_delta_mags()[0])
    assert restored.companion_delta_mags()[0] != restored.companion_delta_mags()[0]
    entry = restored.neighbor_dicts_for_evidence({1: 1.})
    assert entry[0]["aperture_flux_fraction"] == 0.12
    assert entry[0]["aperture_flux_fraction_joint_95"] == 0.1
    assert entry[0]["aperture_flux_fraction_upper_95"] == 0.12
    assert entry[0]["aperture_flux_fraction_limitation"]


def test_portable_scene_rejects_a_missing_magnitude_contrast_without_a_reason(tmp_path):
    import pytest

    from trex.app.scene import TargetScene

    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    neighbor = {
        "source_id": "123456", "M_s": 0.8, "R_s": 0.8, "delta_mag": None,
        "separation_arcsec": 4.,
        "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}],
    }
    with pytest.raises(ValueError, match="without delta_mag must state why"):
        TargetScene(tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1.,
                    Teff_K=5700., Tmag=10., plx_mas=2., sectors=[1],
                    contrast_separations=[0.1, 1.], contrast_values=[2., 5.],
                    target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.9}],
                    resolved_neighbors=[neighbor], N_background=1, trilegal_cache=background,
                    background_area_sq_deg=0.1)


def test_portable_scene_rejects_a_marginal_limit_without_its_limitation(tmp_path):
    import pytest

    from trex.app.scene import TargetScene

    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    neighbor = {
        "source_id": "123456", "M_s": 0.8, "R_s": 0.8, "delta_mag": 1.,
        "separation_arcsec": 4.,
        "aperture_flux_fractions": [{"sector": 1, "fraction": 0.1}],
        "aperture_flux_fractions_upper_95": [{"sector": 1, "fraction": 0.12}],
    }
    with pytest.raises(ValueError, match="marginal aperture limit must state its limitation"):
        TargetScene(tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1.,
                    Teff_K=5700., Tmag=10., plx_mas=2., sectors=[1],
                    contrast_separations=[0.1, 1.], contrast_values=[2., 5.],
                    target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.9}],
                    resolved_neighbors=[neighbor], N_background=1, trilegal_cache=background,
                    background_area_sq_deg=0.1)


def test_scene_binding_checks_the_joint_scene_not_non_additive_margins(tmp_path):
    """A per-host marginal limit may overcount the joint scene without breaking binding."""
    import hashlib

    from trex.app.scene import TargetScene
    from trex.app.target import target as trex_target

    background = tmp_path / "background.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    neighbor = {
        "source_id": "123456", "M_s": 0.8, "R_s": 0.8, "delta_mag": 1.,
        "separation_arcsec": 4.,
        "aperture_flux_fractions": [{"sector": 1, "fraction": 0.2}],
        "aperture_flux_fractions_upper_95": [{"sector": 1, "fraction": 0.25}],
        "aperture_flux_fraction_limitation": "marginal source limit; not additive",
    }
    scene = TargetScene(tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1.,
                        Teff_K=5700., Tmag=10., plx_mas=2., sectors=[1],
                        contrast_separations=[0.1, 1.], contrast_values=[2., 5.],
                        target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.8}],
                        resolved_neighbors=[neighbor], N_background=1, trilegal_cache=background,
                        background_area_sq_deg=0.1)
    bound = trex_target.from_scene(
        scene,
        sector_weights={1: 1.0},
        depth_ppm=1000.0,
        exptime_days=120.0 / 86400.0,
        input_artifacts={str(background): hashlib.sha256(background.read_bytes()).hexdigest()},
    )
    binding = bound._scene_binding

    # Target 0.8 + neighbour joint 0.2 = 1.0 binds; the 0.25 marginal limit would
    # overcount (1.05) and must be retained, not summed.
    assert binding["depth_ppm"] == pytest.approx(1000.0)
    assert binding["sector_weights"] == {1: 1.0}
    # TargetScene normalizes per-sector rows to {sector: fraction}.
    assert binding["scene"].target_aperture_flux_fractions == {1: 0.8}
    retained = binding["scene"].resolved_neighbors[0]
    assert retained["aperture_flux_fractions"] == {1: 0.2}
    assert retained["aperture_flux_fractions_upper_95"] == {1: 0.25}
    assert {str(star_id) for star_id in binding["stars"]["ID"]} == {"123", "123456"}
