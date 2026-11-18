"""Host-specific limb-darkening routing through TREX scenario likelihoods."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from trex.foundation.constants import G, Msun, Rsun
from trex.stellar_light.host_limb_darkening import TARGET_HOST_LDC_ROLE, load_triceratops_tess_grid, summarize_ldc_grid_provenance


# The solar-analog fallback that used to be the production default for every
# target-hosted scenario (Claret 2017 TESS solar values). A non-solar target
# must not receive these.
_SOLAR_FALLBACK_U1 = 0.4
_SOLAR_FALLBACK_U2 = 0.2


def _non_solar_scene(tmp_path):
    """A validated scene whose atmosphere is nowhere near the solar analog."""
    from trex.app.scene import TargetScene
    from tests.fixtures.trex_scene import caller_provenance

    background = tmp_path / "synthetic_trilegal.csv"
    background.write_text(
        ",TESS,logTe,logg,Mact,[M/H]\n"
        "0,14.0,3.70,4.5,0.80,0.0\n"
        "1,16.0,3.60,4.7,0.50,0.0\n"
        "2,#TRILEGAL,normally,terminated,\n",
        encoding="utf-8",
    )
    return TargetScene(
        tic_id=123,
        ra_deg=90.0,
        dec_deg=-60.0,
        M_s_Msun=0.55,
        R_s_Rsun=0.52,
        Teff_K=3800.0,
        Tmag=10.0,
        plx_mas=2.0,
        feh_dex=-0.3,
        metallicity_provenance=caller_provenance(-0.3),
        sectors=[1],
        contrast_separations=np.array([0.1, 1.0]),
        contrast_values=np.array([2.0, 5.0]),
        target_aperture_flux_fractions=[{"sector": 1, "fraction": 0.5}],
        resolved_neighbors=[],
        N_background=1,
        trilegal_cache=background,
        background_sha256=hashlib.sha256(background.read_bytes()).hexdigest(),
        background_area_sq_deg=0.1,
    )


def _run_capturing_evidences(monkeypatch, scene, **overrides):
    """Run the production vet path with the evidence integral stubbed out."""
    from trex.app import engine as trex

    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return np.full(15, -10.0), "ok"

    monkeypatch.setattr(trex, "calc_target_evidences", capture)
    provenance: list = []
    time = np.linspace(-0.2, 0.2, 41)
    trex.run_trex_vetting(
        scene,
        time=time,
        flux=np.ones_like(time),
        sigma=1e-4,
        period_days=5.0,
        depth_ppm=1000.0,
        exptime_days=0.001,
        sector_weights={1: 1.0},
        ldc_provenance=provenance,
        **overrides,
    )
    return captured, provenance


def test_target_hosted_scenarios_use_the_target_atmosphere_not_the_solar_fallback(
    tmp_path, monkeypatch
):
    """A non-solar target must drive the target-hosted scenario coefficients.

    The target-hosted scenarios (TP/EB/EBx2P and the diluted P*/D* families)
    model the transit on the target star itself, so a solar analog there is
    simply the wrong atmosphere. The expected values come from the packaged
    grid at the scene's own effective temperature, its surface gravity derived
    from the scene mass and radius, and its metallicity.
    """
    scene = _non_solar_scene(tmp_path)
    captured, provenance = _run_capturing_evidences(monkeypatch, scene)

    logg_cgs = math.log10(G * scene.M_s_Msun * Msun / (scene.R_s_Rsun * Rsun) ** 2)
    node = load_triceratops_tess_grid().select(scene.Teff_K, logg_cgs, scene.feh_dex)

    assert captured["u1"] == pytest.approx(node.u1)
    assert captured["u2"] == pytest.approx(node.u2)
    # The fallback must be genuinely different here, or this test proves nothing.
    assert abs(node.u1 - _SOLAR_FALLBACK_U1) > 1.0e-3
    assert abs(node.u2 - _SOLAR_FALLBACK_U2) > 1.0e-3

    summary = summarize_ldc_grid_provenance(provenance)
    target = summary["target_host"]
    assert target is not None
    assert target["role"] == TARGET_HOST_LDC_ROLE
    assert target["source"] == "triceratops-tess-grid"
    assert target["used_teff_k"] == pytest.approx(node.used_teff_k)
    assert target["used_logg_cgs"] == pytest.approx(node.used_logg_cgs)
    assert target["used_metallicity_dex"] == pytest.approx(node.used_metallicity_dex)
    assert target["u1"] == pytest.approx(node.u1)
    assert target["u2"] == pytest.approx(node.u2)
    assert target["boundary_clamped"] == node.boundary_clamped


def test_explicitly_supplied_target_coefficients_are_used_and_recorded_as_such(
    tmp_path, monkeypatch
):
    """An override is honoured, and never reported as a grid lookup."""
    scene = _non_solar_scene(tmp_path)
    captured, provenance = _run_capturing_evidences(
        monkeypatch, scene, u1=0.31, u2=0.27
    )

    assert captured["u1"] == pytest.approx(0.31)
    assert captured["u2"] == pytest.approx(0.27)
    target = summarize_ldc_grid_provenance(provenance)["target_host"]
    assert target["source"] == "caller-supplied"
    assert target["u1"] == pytest.approx(0.31)
    assert "used_teff_k" not in target


def test_half_supplied_target_coefficients_are_refused(tmp_path, monkeypatch):
    """One coefficient without the other is an incoherent request, not a default."""
    scene = _non_solar_scene(tmp_path)
    with pytest.raises(ValueError, match="supplied together"):
        _run_capturing_evidences(monkeypatch, scene, u1=0.31)


def test_background_host_uses_its_own_grid_coefficients(monkeypatch):
    """A B* host must not receive the target star's quadratic coefficients."""
    from trex.evidence import marginal_likelihoods

    n_draws = 4
    captured = []
    grid = load_triceratops_tess_grid()
    expected = grid.coefficients(3500.0, math.log10(G * 0.5 * Msun / (0.5 * Rsun)**2), -0.2)
    original_likelihood = marginal_likelihoods.lnL_TP

    class Grid:
        def select_many(self, teff_k, logg_cgs, metallicity_dex):
            assert np.all(teff_k == 3500.0)
            assert np.all(metallicity_dex == -0.2)
            assert np.all(np.isfinite(logg_cgs))
            return grid.select_many(teff_k, logg_cgs, metallicity_dex)

    def background_draw(*_args, **_kwargs):
        _kwargs["metallicity_out"].append(np.full(n_draws, -0.2))
        return (
            np.full(n_draws, 0.1),
            np.zeros(n_draws),
            np.full(n_draws, 0.5),
            np.full(n_draws, 0.5),
            np.full(n_draws, 3500.0),
        )

    def likelihood(*args, **kwargs):
        captured.append((args[8], args[9]))
        return original_likelihood(*args, **kwargs)

    monkeypatch.setattr(marginal_likelihoods, "_draw_background_star", background_draw)
    monkeypatch.setattr(marginal_likelihoods, "load_triceratops_tess_grid", lambda: Grid())
    monkeypatch.setattr(marginal_likelihoods, "sample_inc", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "sample_ecc", lambda draws, **_kwargs: np.zeros(draws.size))
    monkeypatch.setattr(marginal_likelihoods, "sample_w", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "lnL_TP", likelihood)

    result = marginal_likelihoods._eval_scenario(
        time=np.array([0.0]),
        flux=np.array([1.0]),
        sigma=1e-4,
        period_days=3.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.3,
        u2=0.2,
        n_draws=n_draws,
        is_planet=True,
        is_EB=False,
        use_2xP=False,
        exptime_days=0.001,
        background_star="background",
        background_population={"unused": np.array([])},
        background_area_sq_deg=1.0,
        target_tess_mag=10.0,
        contrast_separations=np.array([0.1, 1.0]),
        contrast_values=np.array([2.0, 5.0]),
        target_feh=-0.2,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(result)
    assert captured
    assert all(pair == expected for pair in captured)
    assert expected != (0.3, 0.2)


def test_bound_companion_secondary_host_uses_its_own_grid_coefficients(monkeypatch):
    """An S* (companion-hosted) scenario must not receive the target's coefficients."""
    from trex.evidence import marginal_likelihoods

    n_draws = 4
    captured = []
    grid = load_triceratops_tess_grid()
    expected = grid.coefficients(3200.0, math.log10(G * 0.3 * Msun / (0.3 * Rsun)**2), -0.2)
    original_likelihood = marginal_likelihoods.lnL_TP

    class Grid:
        def select_many(self, teff_k, logg_cgs, metallicity_dex):
            assert np.all(teff_k == 3200.0)
            assert np.all(metallicity_dex == -0.2)
            assert np.all(np.isfinite(logg_cgs))
            return grid.select_many(teff_k, logg_cgs, metallicity_dex)

    def bound_draw(*_args, **_kwargs):
        return (
            np.full(n_draws, 0.1),
            np.zeros(n_draws),
            np.full(n_draws, 0.3),
            np.full(n_draws, 0.3),
            np.full(n_draws, 3200.0),
        )

    def likelihood(*args, **kwargs):
        captured.append((args[8], args[9]))
        return original_likelihood(*args, **kwargs)

    monkeypatch.setattr(marginal_likelihoods, "_draw_bound_companion", bound_draw)
    monkeypatch.setattr(marginal_likelihoods, "load_triceratops_tess_grid", lambda: Grid())
    monkeypatch.setattr(marginal_likelihoods, "sample_inc", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "sample_ecc", lambda draws, **_kwargs: np.zeros(draws.size))
    monkeypatch.setattr(marginal_likelihoods, "sample_w", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "lnL_TP", likelihood)

    result = marginal_likelihoods._eval_scenario(
        time=np.array([0.0]),
        flux=np.array([1.0]),
        sigma=1e-4,
        period_days=3.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.3,
        u2=0.2,
        n_draws=n_draws,
        is_planet=True,
        is_EB=False,
        use_2xP=False,
        exptime_days=0.001,
        bound_companion="secondary",
        plx_mas=10.0,
        contrast_separations=np.array([0.1, 1.0]),
        contrast_values=np.array([2.0, 5.0]),
        target_feh=-0.2,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(result)
    assert captured
    assert all(pair == expected for pair in captured)
    assert expected != (0.3, 0.2)


def test_resolved_neighbor_host_uses_its_own_grid_coefficients(monkeypatch):
    """A resolved-neighbor-hosted scenario must not receive the target's coefficients."""
    from trex.evidence import marginal_likelihoods

    captured_u1u2 = []
    from trex.stellar.stellar_relations import stellar_relations

    grid = load_triceratops_tess_grid()
    temperature = stellar_relations(np.array([0.8]))[1][0]
    expected = grid.coefficients(temperature, math.log10(G * 0.8 * Msun / (0.8 * Rsun)**2), -0.3)
    original_scenario = marginal_likelihoods._eval_scenario

    class Grid:
        def select_many(self, teff_k, logg_cgs, metallicity_dex):
            assert np.all(metallicity_dex == -0.3)
            assert np.all(np.isfinite(teff_k)) and np.all(np.isfinite(logg_cgs))
            return grid.select_many(teff_k, logg_cgs, metallicity_dex)

    def observed_scenario(time, flux, sigma, period_days, M_s_Msun, R_s_Rsun, u1, u2, *args, **kwargs):
        captured_u1u2.append((u1, u2))
        return original_scenario(time, flux, sigma, period_days, M_s_Msun, R_s_Rsun, u1, u2, *args, **kwargs)

    monkeypatch.setattr(marginal_likelihoods, "load_triceratops_tess_grid", lambda: Grid())
    monkeypatch.setattr(marginal_likelihoods, "_eval_scenario", observed_scenario)

    time = np.linspace(-0.1, 0.1, 21)
    flux = np.ones_like(time)
    lnZ, status = marginal_likelihoods.calc_target_evidences(
        time=time,
        flux=flux,
        sigma=1e-4,
        period_days=5.0,
        depth_ppm=500.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.4,
        u2=0.2,
        exptime_days=0.001,
        n_draws=4,
        target_aperture_flux_fraction=1.0,
        nearby_stars=[{"M_s": 0.8, "R_s": 0.8, "aperture_flux_fraction": 0.9}],
        random_seed=1,
        target_feh=-0.3,
        # Bound populations need parallax; both unresolved populations need an angular bound.
        plx=10.0,
        resolution_limit_arcsec=1.0,
        # Background scenarios need their retained population and sky area;
        # without them the engine correctly reports the set unavailable.
        background_population={
            "tess_mag": np.array([14.0]),
            "mass_solar": np.array([0.9]),
            "radius_solar": np.array([0.9]),
            "teff_k": np.array([5500.0]),
            "mh_dex": np.array([-0.3]),
        },
        background_area_sq_deg=0.1,
        target_tess_mag=10.0,
    )

    assert status == "ok"
    assert captured_u1u2
    # Neighbor scenarios are always evaluated strictly after every target
    # scenario, so the trailing N_NEIGHBOR_SCENARIOS calls are the neighbor's,
    # regardless of how many target-hosted scenarios were available here.
    neighbor_calls = captured_u1u2[-marginal_likelihoods.N_NEIGHBOR_SCENARIOS:]
    assert len(neighbor_calls) == marginal_likelihoods.N_NEIGHBOR_SCENARIOS
    assert all(pair == expected for pair in neighbor_calls)
    # And it must differ from the target's own (0.4, 0.2) coefficients.
    assert (0.4, 0.2) not in neighbor_calls
