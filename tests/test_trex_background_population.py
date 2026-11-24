"""TRILEGAL background-star population for the TREX D*/B* scenarios.

Giacalone et al. (2021), AJ 161, 24, evaluate the unresolved *background* star
scenarios against a simulated TRILEGAL population: each Monte Carlo draw picks a
background star fainter than the target, that star's TESS magnitude sets the
dilution, and the prior is the expected number of such stars close enough to
remain unresolved.

The reference implementation hardcodes the TRILEGAL simulated sky area as
``0.1`` deg^2 inside its prior; TREX does not: the area is a property of the
query that produced a particular candidate's cache, is not recorded in the
TRILEGAL output file itself, and therefore has to be retained as evidence.

Expected values here are derived from the source-file contents and the areal
arithmetic directly, never by running the production reduction under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


_TRILEGAL_HEADER = ",#Gc,logAge,[M/H],m_ini,logL,logTe,logg,m-M0,Av,m2/m1,mbol,TESS,J,H,Ks,Mact"


def _trilegal_rows(rows):
    """Build a TRILEGAL v1.6 CSV body with the real column layout and footer."""
    lines = [_TRILEGAL_HEADER]
    for index, (tess, logte, logg, mact) in enumerate(rows):
        lines.append(
            "{0},1,9.0,0.0,0.5,-0.5,{1},{2},10.0,0.03,0.0,12.0,{3},13.0,12.5,12.3,{4}".format(
                index, logte, logg, tess, mact
            )
        )
    lines.append("{0},#TRILEGAL,normally,terminated,,,,,,,,,,,,,".format(len(rows)))
    return "\n".join(lines) + "\n"


def _write_cache(tmp_path, rows):
    path = tmp_path / "trilegal_cache.csv"
    path.write_text(_trilegal_rows(rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing the retained TRILEGAL cache
# ---------------------------------------------------------------------------

def test_background_table_keeps_only_stars_fainter_than_the_target(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_cache(
        tmp_path,
        [
            (10.0, 3.75, 4.4, 0.9),   # brighter than the target: excluded
            (12.0, 3.70, 4.5, 0.8),   # equal to the target: retained
            (15.0, 3.60, 4.7, 0.5),   # fainter: retained
        ],
    )
    table = load_trilegal_population(path, target_tess_mag=12.0)

    np.testing.assert_allclose(table["tess_mag"], [12.0, 15.0])
    np.testing.assert_allclose(table["mass_solar"], [0.8, 0.5])
    np.testing.assert_allclose(table["logg_cgs"], [4.5, 4.7])
    np.testing.assert_allclose(table["teff_k"], [10.0 ** 3.70, 10.0 ** 3.60])


def test_background_table_drops_the_trilegal_footer(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_cache(tmp_path, [(13.0, 3.70, 4.5, 0.8)])
    table = load_trilegal_population(path, target_tess_mag=12.0)

    assert table["tess_mag"].size == 1


def test_background_table_fails_closed_on_missing_columns(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = tmp_path / "trilegal_cache.csv"
    path.write_text("source_id\nsynthetic-background\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TRILEGAL"):
        load_trilegal_population(path, target_tess_mag=12.0)


def test_background_table_fails_closed_when_nothing_is_fainter(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_cache(tmp_path, [(10.0, 3.75, 4.4, 0.9)])

    with pytest.raises(ValueError, match="fainter"):
        load_trilegal_population(path, target_tess_mag=12.0)


def test_background_radius_uses_the_catalog_surface_gravity(tmp_path):
    """R follows from the simulated mass and logg, not a main-sequence relation.

    TRILEGAL reports evolved stars too, so overriding its logg with a
    mass-radius relation would silently turn a giant into a dwarf.
    """
    from trex.populations.background import load_trilegal_population
    from trex.foundation.constants import G, Msun, Rsun

    mass, logg = 1.2, 2.5
    path = _write_cache(tmp_path, [(15.0, 3.70, logg, mass)])
    table = load_trilegal_population(path, target_tess_mag=12.0)

    expected = math.sqrt(G * mass * Msun / 10.0 ** logg) / Rsun
    assert float(table["radius_solar"][0]) == pytest.approx(expected)
    assert expected > 1.0   # the giant stays a giant


# ---------------------------------------------------------------------------
# The areal background prior
# ---------------------------------------------------------------------------

def test_background_surface_density_uses_the_retained_query_area():
    """The expected count must scale with the real simulated area, not 0.1."""
    from trex.populations.priors import background_population_expectation

    within_one = background_population_expectation(
        star_count=400, simulated_area_sq_deg=0.1, field_radius_arcsec=1.0
    )
    within_two = background_population_expectation(
        star_count=400, simulated_area_sq_deg=0.1, field_radius_arcsec=2.0
    )
    wider_survey = background_population_expectation(
        star_count=400, simulated_area_sq_deg=0.2, field_radius_arcsec=1.0
    )

    # Expected count = surface density * pi * radius^2.
    density_per_sq_arcsec = 400.0 / (0.1 * 3600.0 ** 2)
    assert within_one == pytest.approx(density_per_sq_arcsec * math.pi)
    assert within_two == pytest.approx(4.0 * within_one)
    assert wider_survey == pytest.approx(0.5 * within_one)


@pytest.mark.parametrize(
    "star_count,area,radius",
    [(0, 0.1, 1.0), (400, 0.0, 1.0), (400, 0.1, 0.0), (-1, 0.1, 1.0)],
)
def test_background_expectation_rejects_unusable_inputs(star_count, area, radius):
    from trex.populations.priors import background_population_expectation

    with pytest.raises(ValueError):
        background_population_expectation(
            star_count=star_count,
            simulated_area_sq_deg=area,
            field_radius_arcsec=radius,
        )


# ---------------------------------------------------------------------------
# The D*/B* scenarios must actually model an unresolved background star
# ---------------------------------------------------------------------------

_DILUTED_SCENARIOS = slice(9, 12)     # DTP, DEB, DEBx2P
_BACKGROUND_SCENARIOS = slice(12, 15)  # BTP, BEB, BEBx2P


def _population():
    return {
        "tess_mag": np.array([14.0, 16.0, 18.0]),
        "mass_solar": np.array([0.9, 0.6, 0.3]),
        "logg_cgs": np.array([4.4, 4.6, 4.9]),
        "teff_k": np.array([5400.0, 4200.0, 3400.0]),
        "radius_solar": np.array([0.95, 0.60, 0.30]),
        "mh_dex": np.array([0.0, -0.5, 0.5]),
    }


def _nearby_stars(aperture_flux_fraction):
    """One resolvable synthetic neighbour, or none for the absent case."""
    if aperture_flux_fraction is None:
        return None
    return [{
        "source_id": "synthetic-neighbor",
        "M_s": 0.6,
        "R_s": 0.6,
        "aperture_flux_fraction": aperture_flux_fraction,
    }]


def _evidences(**overrides):
    from trex.evidence.marginal_likelihoods import calc_target_evidences

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    kwargs = dict(
        time=time,
        flux=flux,
        sigma=1e-4,
        period_days=4.0,
        depth_ppm=1000.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.4,
        u2=0.2,
        exptime_days=0.001,
        target_aperture_flux_fraction=1.0,
        n_draws=600,
        contrast_separations=np.array([0.2, 1.0, 4.0]),
        contrast_values=np.array([2.0, 5.0, 8.0]),
        plx=10.0,
        target_tess_mag=12.0,
        background_population=_population(),
        background_area_sq_deg=0.1,
        random_seed=3,
        nsamples=5,
        target_feh=-0.2,
    )
    kwargs.update(overrides)
    lnZ, _ = calc_target_evidences(**kwargs)
    return lnZ


def test_background_scenarios_dilute_by_a_real_simulated_star(monkeypatch):
    """D*/B* must carry a background flux ratio, and B* must host the transit."""
    from trex.evidence import mc_scenario

    seen = {}
    original = mc_scenario._eval_scenario

    def record(*args, **kwargs):
        mode = kwargs.get("background_star")
        if mode is not None:
            seen.setdefault(mode, []).append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mc_scenario, "_eval_scenario", record)
    _evidences()

    assert sorted(seen) == ["background", "diluted"]
    assert len(seen["diluted"]) == 3 and len(seen["background"]) == 3
    # The diluting star is on the target's side for D*, and is the host for B*.
    assert all(entry["companion_is_host"] is False for entry in seen["diluted"])
    assert all(entry["companion_is_host"] is True for entry in seen["background"])


def test_background_scenarios_are_no_longer_degenerate_with_flat_prior_transits():
    """Before this model D*/B* passed companion_fluxratio = 0.0 and matched TP/EB."""
    lnZ = _evidences()

    assert np.all(np.isfinite(lnZ[_DILUTED_SCENARIOS]))
    assert np.all(np.isfinite(lnZ[_BACKGROUND_SCENARIOS]))
    # A background star dilutes the transit, so the diluted-target scenarios
    # cannot reproduce the undiluted target-only evidences.
    assert not np.allclose(lnZ[_DILUTED_SCENARIOS], lnZ[0:3])
    assert not np.allclose(lnZ[_BACKGROUND_SCENARIOS], lnZ[_DILUTED_SCENARIOS])


def test_background_evidences_ignore_resolved_neighbour_contrasts():
    """The background population is unrelated to the resolved Gaia neighbours."""
    faint = _evidences(nearby_stars=_nearby_stars(0.01))
    bright = _evidences(nearby_stars=_nearby_stars(0.05))
    absent = _evidences(nearby_stars=None)

    for scenarios in (_DILUTED_SCENARIOS, _BACKGROUND_SCENARIOS):
        np.testing.assert_allclose(faint[scenarios], bright[scenarios])
        np.testing.assert_allclose(faint[scenarios], absent[scenarios])


def test_background_evidences_scale_with_the_retained_simulated_area():
    """Halving the simulated area doubles the surface density, hence the weight."""
    base = _evidences(background_area_sq_deg=0.1)
    denser = _evidences(background_area_sq_deg=0.05)

    for scenarios in (_DILUTED_SCENARIOS, _BACKGROUND_SCENARIOS):
        np.testing.assert_allclose(
            denser[scenarios] - base[scenarios], np.log(2.0), rtol=1e-9
        )


def test_background_draws_are_not_a_degenerate_estimator():
    """Enough simulated stars must survive the imaging cut to marginalize over."""
    from trex.evidence.marginal_likelihoods import _draw_background_star

    n_draws = 20000
    _, lnprior, _, _, _ = _draw_background_star(
        np.random.default_rng(23),
        n_draws,
        _population(),
        0.1,
        12.0,
        np.array([0.2, 1.0, 4.0]),
        np.array([2.0, 5.0, 8.0]),
    )
    retained = float(np.mean(np.isfinite(lnprior)))

    assert 0.01 < retained < 1.0
    finite = lnprior[np.isfinite(lnprior)]
    assert float(np.ptp(finite)) == 0.0


def test_background_field_uses_catalog_bound_beyond_the_curve_outer_edge():
    """An imaging curve is not a population truncation or Gaia-resolved catalog."""
    from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC
    from trex.evidence.marginal_likelihoods import _draw_background_star, background_population_expectation

    _, log_prior, _, _, _ = _draw_background_star(
        np.random.default_rng(31),
        8,
        _population(),
        0.1,
        12.0,
        np.array([0.1, 1.0]),
        np.array([0.0, 0.0]),
    )
    expected = background_population_expectation(
        star_count=len(_population()["tess_mag"]),
        simulated_area_sq_deg=0.1,
        field_radius_arcsec=GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC,
    )

    np.testing.assert_allclose(log_prior, math.log(expected), rtol=0.0, atol=1e-14)


def test_background_scenarios_fail_closed_without_retained_evidence():
    without_population = _evidences(background_population=None)
    without_area = _evidences(background_area_sq_deg=None)

    for lnZ in (without_population, without_area):
        assert np.all(np.isneginf(lnZ[_DILUTED_SCENARIOS]))
        assert np.all(np.isneginf(lnZ[_BACKGROUND_SCENARIOS]))


def test_background_hosted_scenarios_fail_closed_without_target_feh():
    """A complete scene still needs target metallicity for its target-hosted EBs.

    DTP retains the supplied target atmosphere; BTP instead uses the simulated
    host's own retained [M/H]. Neither TP slot needs an unrelated target [Fe/H].
    """
    from trex.evidence.marginal_likelihoods import _eval_scenario

    with_feh = _evidences(target_feh=-0.2)
    assert np.all(np.isfinite(with_feh[_BACKGROUND_SCENARIOS]))
    with pytest.raises(ValueError, match="target_feh"):
        _evidences(target_feh=None)

    # Source-hosted B* uses its own simulated metallicity, while missing target EBs defer.
    from trex.evidence.diagnostics import generate_diagnostics
    fits = {}
    incomplete = _evidences(target_feh=None, n_draws=64, representative_fit_out=fits)
    assert fits[1]["status"] == "unavailable" and "metallicity" in fits[1]["reason"]
    assert all(fits[index]["status"] != "unavailable" for index in range(12, 15))
    result = generate_diagnostics(incomplete, scenario_fits=fits, period_days=4.0)
    assert result.degenerate and result.fpp is None and result.nfpp is None
    assert any(d.name == "scenario-input-unavailable" for d in result.diagnostics)

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    common_kwargs = dict(
        time=time, flux=flux, sigma=1e-4, period_days=4.0,
        M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.4, u2=0.2, n_draws=600,
        is_planet=True, is_EB=False, use_2xP=False, exptime_days=0.001,
        background_star="diluted", background_population=_population(),
        background_area_sq_deg=0.1, contrast_separations=np.array([0.2, 1.0, 4.0]),
        contrast_values=np.array([2.0, 5.0, 8.0]), target_tess_mag=12.0,
        nsamples=5,
    )
    diluted_with_feh = _eval_scenario(rng=np.random.default_rng(11), target_feh=-0.2, **common_kwargs)
    diluted_without_feh = _eval_scenario(rng=np.random.default_rng(11), target_feh=None, **common_kwargs)

    assert math.isfinite(diluted_with_feh)
    assert diluted_with_feh == pytest.approx(diluted_without_feh, rel=1e-12)

    # An unrelated BTP atmosphere uses the table's [M/H], so it needs no target [Fe/H].
    hosted_kwargs = dict(common_kwargs, background_star="background")
    assert _eval_scenario(rng=np.random.default_rng(11), target_feh=None, **hosted_kwargs) == pytest.approx(
        _eval_scenario(rng=np.random.default_rng(11), target_feh=-0.2, **hosted_kwargs), rel=1e-12)


def test_background_hosted_eb_skips_hosts_below_the_stellar_relation_domain():
    """A TRILEGAL host too light for any stellar secondary is impossible, not a crash."""
    from trex.evidence.marginal_likelihoods import _eval_scenario

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    population = {
        "tess_mag": np.array([14.0, 18.0]),
        "mass_solar": np.array([0.9, 0.078]),
        "logg_cgs": np.array([4.4, 5.3]),
        "teff_k": np.array([5400.0, 2700.0]),
        "radius_solar": np.array([0.95, 0.10]),
        "mh_dex": np.array([0.0, -0.5]),
    }
    kwargs = dict(
        time=time, flux=flux, sigma=1e-4, period_days=4.0,
        M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.4, u2=0.2, n_draws=400,
        is_planet=False, is_EB=True, use_2xP=False, exptime_days=0.001,
        background_star="background", background_area_sq_deg=0.1,
        contrast_separations=np.array([0.2, 1.0, 4.0]), contrast_values=np.array([2.0, 5.0, 8.0]),
        target_tess_mag=12.0, target_feh=0.0, nsamples=5,
    )
    mixed = _eval_scenario(rng=np.random.default_rng(3), background_population=population, **kwargs)
    only_light = {key: value[1:] for key, value in population.items()}
    light = _eval_scenario(rng=np.random.default_rng(3), background_population=only_light, **kwargs)

    assert not math.isnan(mixed)
    assert np.isneginf(light)


def test_background_hosted_scenarios_reject_a_non_finite_target_feh():
    import pytest

    with pytest.raises(ValueError, match="target_feh"):
        _evidences(target_feh=float("nan"))


def test_overlapped_scenarios_match_the_sequential_loop_bit_for_bit(monkeypatch):
    """Preparing scenario k+1 during scenario k's likelihood changes no draw or output."""
    import threading

    from trex.evidence import marginal_likelihoods

    class MainThreadOnly:
        """The run's generator; a draw from the likelihood worker would race the next scenario."""

        def __init__(self, generator):
            self._generator = generator

        def __getattr__(self, name):
            assert threading.current_thread() is threading.main_thread(), name
            return getattr(self._generator, name)

    default_rng = np.random.default_rng
    monkeypatch.setattr(marginal_likelihoods.np.random, "default_rng",
                        lambda *args: MainThreadOnly(default_rng(*args)))

    def run():
        outputs = {"ess_out": {}, "support_state_out": {}, "jax_fallback_out": {}, "representative_fit_out": {}}
        lnZ = _evidences(nearby_stars=[{"M_s": 0.5, "R_s": 0.5, "aperture_flux_fraction": 0.25}],
                         target_aperture_flux_fraction=0.75, **outputs)
        return lnZ, outputs

    overlapped = run()
    started = []
    original_init = marginal_likelihoods._ScenarioOverlap.__init__

    def sequential(self):
        original_init(self)
        started.append(self._enabled)
        self._enabled = False

    monkeypatch.setattr(marginal_likelihoods._ScenarioOverlap, "__init__", sequential)
    reference = run()

    assert started == [True]
    np.testing.assert_array_equal(overlapped[0], reference[0])
    assert np.isfinite(overlapped[0]).sum() > marginal_likelihoods.N_NEIGHBOR_SCENARIOS
    # `phase_seconds` is measured wall time, never reproducible; every other field stays bit-for-bit.
    def untimed(value):
        if isinstance(value, dict):
            return {key: untimed(item) for key, item in value.items() if key != "phase_seconds"}
        return value

    assert repr(untimed(overlapped[1])) == repr(untimed(reference[1]))


# ---------------------------------------------------------------------------
# Requested native bands (MM-D0)
# ---------------------------------------------------------------------------

def _write_band_cache(tmp_path, rows):
    """TRILEGAL rows carrying distinct J/H/Ks per star; ``rows`` are (tess, j, h, ks)."""
    lines = [",#Gc,logAge,[M/H],m_ini,logL,logTe,logg,m-M0,Av,m2/m1,mbol,TESS,J,H,Ks,Mact"]
    for index, (tess, j, h, ks) in enumerate(rows):
        lines.append(
            "{0},1,9.0,0.0,0.5,-0.5,3.70,4.5,10.0,0.03,0.0,12.0,{1},{2},{3},{4},0.8".format(
                index, tess, j, h, ks
            )
        )
    lines.append("{0},#TRILEGAL,normally,terminated,,,,,,,,,,,,,".format(len(rows)))
    path = tmp_path / "trilegal_bands.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_requested_bands_keep_their_native_identity_and_row_alignment(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_band_cache(tmp_path, [
        (10.0, 9.0, 8.0, 7.0),     # brighter than the target: no row in any band
        (12.0, 11.0, 10.5, 10.0),  # retained
        (15.0, 14.5, 13.7, 13.1),  # retained
    ])
    table = load_trilegal_population(path, target_tess_mag=12.0,
                                     extra_bands={"J": "J", "H": "H", "Ks": "Ks"})

    np.testing.assert_allclose(table["band_magnitudes"]["J"], [11.0, 14.5])
    np.testing.assert_allclose(table["band_magnitudes"]["H"], [10.5, 13.7])
    np.testing.assert_allclose(table["band_magnitudes"]["Ks"], [10.0, 13.1])
    # H and Ks are distinct arrays, not one shared 2MASS-like column.
    assert not np.array_equal(table["band_magnitudes"]["H"], table["band_magnitudes"]["Ks"])
    assert table["bands"]["H"] == {"column": "H", "unit": "mag", "status": "retained"}
    assert table["missing_bands"] == ()
    assert table["band_magnitudes"]["J"].shape == table["tess_mag"].shape


def test_absent_requested_band_is_reported_not_substituted(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_band_cache(tmp_path, [(13.0, 12.0, 11.5, 11.0)])
    table = load_trilegal_population(path, target_tess_mag=12.0, extra_bands=("Y",))

    assert table["missing_bands"] == ("Y",)
    assert "Y" not in table["band_magnitudes"]
    assert table["bands"]["Y"]["status"] == "missing-column"
    # The star is retained: a missing optional band never deletes a row.
    assert table["tess_mag"].size == 1


def test_partially_missing_band_keeps_the_row_and_marks_the_cell(tmp_path):
    from trex.populations.background import background_band_magnitudes, load_trilegal_population

    path = tmp_path / "trilegal_partial.csv"
    path.write_text(
        ",#Gc,logAge,[M/H],m_ini,logL,logTe,logg,m-M0,Av,m2/m1,mbol,TESS,J,H,Ks,Mact\n"
        "0,1,9.0,0.0,0.5,-0.5,3.70,4.5,10.0,0.03,0.0,12.0,13.0,12.0,12.5,,0.8\n"
        "1,1,9.0,0.0,0.5,-0.5,3.70,4.5,10.0,0.03,0.0,12.0,14.0,13.0,13.5,13.0,0.8\n"
        "2,#TRILEGAL,normally,terminated,,,,,,,,,,,,,\n",
        encoding="utf-8",
    )
    table = load_trilegal_population(path, target_tess_mag=12.0, extra_bands={"Ks": "Ks"})

    values, valid = background_band_magnitudes(table, "Ks")
    assert values.size == 2
    assert math.isnan(values[0]) and not valid[0]
    assert values[1] == 13.0 and valid[1]
    assert table["bands"]["Ks"]["status"] == "partially-missing"
    np.testing.assert_allclose(table["tess_mag"], [13.0, 14.0])


def test_legacy_load_does_not_gain_band_keys(tmp_path):
    from trex.populations.background import load_trilegal_population

    path = _write_band_cache(tmp_path, [(13.0, 12.0, 11.5, 11.0)])
    table = load_trilegal_population(path, target_tess_mag=12.0)

    assert "band_magnitudes" not in table and "missing_bands" not in table


def test_converter_handoff_retains_extra_band_columns(tmp_path):
    """The real TRILEGAL text -> CSV converter feeds the band-aware loader losslessly."""
    from trex.acquisition.trilegal_output import trilegal_output_to_csv
    from trex.populations.background import load_trilegal_population

    text = (
        "TESS Mact logg logTe [M/H] J H Ks\n"
        "13.0 0.8 4.5 3.70 0.0 12.4 11.9 11.3\n"
        "#TRILEGAL normally terminated\n"
    )
    csv_text, count = trilegal_output_to_csv(text)
    assert count == 1
    path = tmp_path / "converted.csv"
    path.write_text(csv_text, encoding="utf-8")
    table = load_trilegal_population(path, target_tess_mag=12.0, extra_bands=("J", "H", "Ks"))

    assert table["band_magnitudes"]["J"][0] == 12.4
    assert table["band_magnitudes"]["H"][0] == 11.9
    assert table["band_magnitudes"]["Ks"][0] == 11.3
