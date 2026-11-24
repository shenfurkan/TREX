"""D22: retain unrelated-star [M/H], never substitute the target's iron abundance."""

import csv
from importlib.resources import files

import numpy as np
import pytest

from trex.populations.background import load_trilegal_population
from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid


def cache(tmp_path, metallicities):
    path = tmp_path / "population.csv"
    path.write_text("TESS,Mact,logg,logTe,[M/H]\n" + "\n".join(
        f"{mag},1,4.5,{np.log10(5750)},{mh}"
        for mag, mh in zip([9, 13, 12], metallicities, strict=True)
    ))
    return path


def independent_coefficients(teff, logg, mh):
    # Read the released table directly, without the selection implementation.
    with files("trex").joinpath("_data/triceratops-1.0.20-ldc_tess.csv").open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    plane = min(sorted({r["Z"] for r in rows}), key=lambda z: abs(z - mh))
    row = next(r for r in rows if (r["Z"], r["Teff"], r["logg"]) == (plane, teff, logg))
    return row["aLSM"], row["bLSM"]


def test_loader_and_population_keep_selected_row_metallicity(tmp_path):
    from trex.evidence.nested_evidence import background_star_population
    from trex.evidence.marginal_likelihoods import _draw_background_star

    table = load_trilegal_population(cache(tmp_path, [0, -1, 0.5]), 10)
    np.testing.assert_array_equal(table["mh_dex"], [-1, 0.5])
    population = background_star_population(table, 0.1, 10, resolution_limit_arcsec=1)
    assert population.resolve(np.array([0, 0.5])).mh_dex == 0.5
    assert population.resolve(np.array([0.99, 0.5])).mh_dex == -1
    metallicities = []
    draws = _draw_background_star(np.random.default_rng(5), 20, table, 0.1, 10,
                                 None, None, resolution_limit_arcsec=1,
                                 metallicity_out=metallicities)
    expected = np.where(draws[0] > 0.1, 0.5, -1)
    np.testing.assert_array_equal(metallicities[0], expected)


@pytest.mark.parametrize("bad", ["nan", "inf", "", "unknown"])
def test_loader_refuses_invalid_metallicity(tmp_path, bad):
    with pytest.raises(ValueError, match="TRILEGAL"):
        load_trilegal_population(cache(tmp_path, [0, bad, 0.5]), 10)


def test_loader_refuses_missing_metallicity(tmp_path):
    path = cache(tmp_path, [0, -1, 0.5])
    path.write_text(path.read_text().replace(",[M/H]", ",Z_mass_fraction"))
    with pytest.raises(ValueError, match=r"\[M/H\]"):
        load_trilegal_population(path, 10)


def test_vector_atmospheres_use_each_source_metallicity():
    mh = np.array([-1.0, 0.5, -1.0])
    selected = load_triceratops_tess_grid().select_many(np.full(3, 5750), np.full(3, 4.5), mh)
    expected = np.array([independent_coefficients(5750, 4.5, z) for z in mh])
    np.testing.assert_array_equal(np.stack((selected.u1, selected.u2), axis=-1), expected)
    assert not np.array_equal(expected[0], expected[1])


def test_mc_preparation_routes_source_composition_to_both_binary_stars(monkeypatch):
    from trex.evidence import marginal_likelihoods as ml
    from trex.evidence import mc_scenario
    from trex.evidence.mc_scenario import _scenario_stages

    table = dict(tess_mag=np.array([12.0, 13.0]), mass_solar=np.ones(2),
                 radius_solar=np.ones(2), teff_k=np.full(2, 5750), mh_dex=np.array([-1, 0.5]))
    expected_mh = table["mh_dex"][np.random.default_rng(5).integers(0, 2, 16)]
    calls = []
    original = ml._host_grid_coefficients

    def record(t, m, r, z, **kwargs):
        result = original(t, m, r, z, **kwargs)
        calls.append((np.asarray(z), result))
        return result

    monkeypatch.setattr(mc_scenario, "_host_grid_coefficients", record)
    monkeypatch.setattr(mc_scenario, "HAS_JAX_TREX", False)
    stages = _scenario_stages(time=np.array([0.0]), flux=np.ones(1), sigma=0.001,
        period_days=3, M_s_Msun=1, R_s_Rsun=1, u1=0.3, u2=0.2, n_draws=16,
        is_planet=False, is_EB=True, use_2xP=False, exptime_days=0.001,
        background_star="background", background_population=table, background_area_sq_deg=0.1,
        target_tess_mag=10, target_feh=0.0, resolution_limit_arcsec=1,
        rng=np.random.default_rng(5))
    next(stages)  # Production population + atmosphere preparation only; no evidence calculation.
    stages.close()
    np.testing.assert_array_equal(calls[0][0], expected_mh)
    assert len(calls) == 2 and set(calls[1][0]) == {-1, 0.5}
    from trex.foundation.constants import G, Msun, Rsun
    gravity = round(np.log10(G * Msun / Rsun**2) / 0.5) * 0.5
    expected = np.array([independent_coefficients(5750, gravity, mh) for mh in expected_mh])
    np.testing.assert_array_equal(np.stack(calls[0][1], axis=-1), expected)


@pytest.mark.parametrize("kind", ["tp", "eb"])
def test_scalar_background_integrand_uses_grid_not_target_metallicity(kind):
    from trex.evidence import nested_evidence as ne
    from trex.foundation.constants import G, Msun, Rsun
    from tests.fixtures.synthetic_observations import (
        synthetic_trex_tp_integrand_case, synthetic_trex_eb_integrand_case,
    )

    table = dict(tess_mag=np.array([12.0]), mass_solar=np.ones(1),
                 radius_solar=np.ones(1), teff_k=np.full(1, 5750), mh_dex=np.array([-1.0]))
    population = ne.background_star_population(table, 0.1, 10, resolution_limit_arcsec=1)
    case = synthetic_trex_tp_integrand_case() if kind == "tp" else synthetic_trex_eb_integrand_case(2.0)
    cls = ne._TransitingPlanetIntegrand if kind == "tp" else ne._EclipsingPairIntegrand
    extra = {"use_flat_priors": False} if kind == "tp" else {}
    reference = cls(**case, **extra, host_mass=1, host_radius=1, nsamples=3,
                    bin_integration_days=0, population=population, n_dimensions=6,
                    hosted=True, target_feh=0.5)
    params = np.array([3.0 if kind == "tp" else 0.6, 0, 90, 90, 0.5, 0.5])
    seen = []
    # Independent table selection supplies the expected primary and secondary LDC.
    def independent_grid(t, mass, radius, z, **kwargs):
        seen.append(np.asarray(z))
        gravity = np.log10(G * np.asarray(mass) * Msun / (np.asarray(radius) * Rsun)**2)
        t = np.clip(np.round(np.asarray(t) / 250) * 250, 3500, 10000)
        gravity = np.clip(np.round(gravity / 0.5) * 0.5, 3.5, 5)
        values = np.array([independent_coefficients(ti, gi, -1) for ti, gi in zip(t, gravity)])
        return values[:, 0], values[:, 1]
    actual = reference.log_likelihood(params)
    reference.target_feh = None
    assert reference.log_likelihood(params) == actual
    from unittest.mock import patch
    with patch.object(ne, "_host_grid_coefficients", independent_grid):
        expected = reference.log_likelihood(params)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    assert all(np.all(z == -1) for z in seen)
