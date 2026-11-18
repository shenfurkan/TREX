"""Independent distance and solar-gravity checks for the MIST likelihood."""

import io
import math
import tarfile

import numpy as np
import pytest
from astropy import constants as const

from trex.acquisition.mist import (
    MIST_COLUMN_NAMES, MIST_COLUMNS, _gaia_edr3_k_g, mist_g_distribution, parallax_zero_point_mas,
    read_mist_grid,
)


def test_zero_point_uses_official_gaia_recipe_and_rejects_missing_inputs():
    from zero_point import zpt

    values = {"phot_g_mean_mag": 9.67, "nu_eff_used_in_astrometry_per_um": 1.55,
              "pseudocolour_per_um": None, "ecl_lat_deg": 30.0, "astrometric_params_solved": 31}
    zpt.load_tables()
    expected = zpt.get_zpt(np.array([9.67]), np.array([1.55]), np.array([np.nan]),
                           np.array([30.0]), np.array([31]), _warnings=False).item()
    assert parallax_zero_point_mas(values) == expected
    with pytest.raises(ValueError, match="incomplete"):
        parallax_zero_point_mas({})


def test_mist_selects_the_node_at_the_measured_distance():
    solar_logg = math.log10((const.G * const.M_sun / const.R_sun**2).cgs.value)
    # G=9.67 at 100 pc implies M_G=4.67; the competing giant is much farther.
    grid = np.array([[9.6, 1.0, 1.0, math.log10(5772), solar_logg, 0.0, 4.67, 1.0],
                     [9.9, 1.3, 1.3, math.log10(4700), 2.5, 0.0, 0.0, 1.0]])
    neighbor = {"phot_g_mean_mag": 9.67, "phot_g_mean_mag_error": 0.003,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02}
    measure = mist_g_distribution(neighbor, grid, 0.1, zero_point_fn=lambda _: 0.0)
    index = int(np.argmax(measure["weights"]))
    assert measure["mass_solar"][index] == 1.0 and measure["weights"][index] > 1 - 1e-12
    assert measure["radius_solar"][index] == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError, match="parallax S/N>=20"):
        mist_g_distribution({**neighbor, "parallax_error_mas": 1.0}, grid, 0.1, zero_point_fn=lambda _: 0.0)


def test_gaia_edr3_g_band_coefficient_matches_the_published_teffnorm_fit():
    """Expected values are hand sums of the DPAC Fitz19 kG TeffNorm rows.

    Source: https://www.cosmos.esa.int/web/gaia/edr3-extinction-law (retrieved
    2026-11-14), Fitz19_EDR3_MainSequence.csv and Fitz19_EDR3_HRDTop.csv,
    rows Xname=TeffNorm, Kname=kG. X=5040 K gives the dimensionless X=1.
    """
    assert float(_gaia_edr3_k_g(5040.0, 4.5, 0.0)) == pytest.approx(0.8366873257957672, abs=1e-15)
    assert float(_gaia_edr3_k_g(5040.0, 4.5, 0.5)) == pytest.approx(0.8240903122301792, abs=1e-15)
    assert float(_gaia_edr3_k_g(5040.0, 4.5, 0.9)) == pytest.approx(0.8144319826039989, abs=1e-15)
    # log g=0 selects the HRD-top fit; at X=1 its published sum is distinct.
    assert float(_gaia_edr3_k_g(5040.0, 0.0, 0.0)) == pytest.approx(0.836715951309955, abs=1e-15)
    # The fit is evaluated at its 3500 K and 10000 K boundaries outside them.
    assert float(_gaia_edr3_k_g(2500.0, 4.5, 0.2)) == float(_gaia_edr3_k_g(3500.0, 4.5, 0.2))
    assert float(_gaia_edr3_k_g(20000.0, 4.5, 0.2)) == float(_gaia_edr3_k_g(10000.0, 4.5, 0.2))


def _two_temperature_grid():
    solar_logg = math.log10((const.G * const.M_sun / const.R_sun**2).cgs.value)
    return np.array([[9.6, 1.0, 1.0, math.log10(5040.0), solar_logg, 0.0, 5.0, 1.0],
                     [9.6, 1.0, 1.0, math.log10(3500.0), solar_logg, 0.0, 5.0, 1.0]])


def test_nonzero_extinction_uses_the_node_g_band_coefficient_not_a_v():
    """At A_V=0.9 the 5040 K node has A_G=0.9*0.8144319826039989 mag, so
    G=5+5+A_G places it at 100 pc; the equal-gravity 3500 K node has a
    different k_G and must receive less weight. A_V subtraction would give
    both nodes the same attenuation and equal weights."""
    grid = _two_temperature_grid()
    neighbor = {"phot_g_mean_mag": 10.732988784343599, "phot_g_mean_mag_error": 0.001,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02}
    measure = mist_g_distribution(neighbor, grid, 0.9, zero_point_fn=lambda _: 0.0)
    weights = np.asarray(measure["weights"])
    assert measure["teff_k"][int(np.argmax(weights))] == pytest.approx(5040.0)
    assert weights[0] > 0.75
    assert "Gaia (E)DR3" in measure["limitations"] and "3500-10000 K" in measure["limitations"]


def test_zero_extinction_leaves_equal_node_weights_regardless_of_teff():
    grid = _two_temperature_grid()
    neighbor = {"phot_g_mean_mag": 10.0, "phot_g_mean_mag_error": 0.001,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02}
    measure = mist_g_distribution(neighbor, grid, 0.0, zero_point_fn=lambda _: 0.0)
    np.testing.assert_allclose(measure["weights"], [0.5, 0.5], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("change", [
    {"phot_g_mean_mag": None}, {"phot_g_mean_mag": True},
    {"phot_g_mean_mag_error": None}, {"phot_g_mean_mag_error": False},
])
def test_missing_or_boolean_photometry_is_a_value_error_before_the_zero_point(change):
    calls = []

    def zero_point(_neighbor):
        calls.append(1)
        raise AssertionError("zero point must not be evaluated for invalid photometry")

    neighbor = {"phot_g_mean_mag": 9.67, "phot_g_mean_mag_error": 0.003,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02, **change}
    with pytest.raises(ValueError):
        mist_g_distribution(neighbor, np.ones((1, 8)), 0.1, zero_point_fn=zero_point)
    assert calls == []


@pytest.mark.parametrize("extinction", [None, True, -0.1, float("nan")])
def test_an_invalid_attenuation_envelope_is_a_value_error(extinction):
    neighbor = {"phot_g_mean_mag": 9.67, "phot_g_mean_mag_error": 0.003,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02}
    with pytest.raises(ValueError):
        mist_g_distribution(neighbor, np.ones((1, 8)), extinction, zero_point_fn=lambda _: 0.0)


def test_an_overflowing_distance_node_gets_zero_weight_not_nan():
    solar_logg = math.log10((const.G * const.M_sun / const.R_sun**2).cgs.value)
    grid = np.array([[9.6, 1.0, 1.0, math.log10(5772), solar_logg, 0.0, 5.0, 1.0],
                     [9.6, 1.1, 1.1, math.log10(5000), solar_logg, 0.0, -2000.0, 1.0]])
    neighbor = {"phot_g_mean_mag": 10.0, "phot_g_mean_mag_error": 0.003,
                "parallax_mas": 10.0, "parallax_error_mas": 0.02}
    measure = mist_g_distribution(neighbor, grid, 0.1, zero_point_fn=lambda _: 0.0)
    assert len(measure["weights"]) == 1 and measure["mass_solar"] == [1.0]
    assert np.all(np.isfinite(measure["weights"])) and sum(measure["weights"]) == pytest.approx(1.0)


def _mist_archive(tmp_path, rows=None, header_columns=None, extinction="0.000", members=None):
    columns = list(header_columns) if header_columns is not None else ["EEP"] + [f"c{index}" for index in range(30)]
    if header_columns is None:
        for index, name in zip(MIST_COLUMNS, MIST_COLUMN_NAMES):
            columns[index] = name
    declaration = "" if extinction is None else f"# CCM89 extinction: Av =  {extinction}\n"
    header = declaration + "# " + " ".join(columns) + "\n"
    path = tmp_path / "mist.tar.xz"
    groups = members if members is not None else [("grid.iso.cmd", rows)]
    with tarfile.open(path, "w:xz") as archive:
        for name, group_rows in groups:
            content = (header + "".join(
                " ".join(str(value) for value in row) + "\n" for row in group_rows)).encode("ascii")
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def _mist_row(log_age, mass, feh, gaia_g=4.67):
    row = [0.0] * 31
    row[1] = log_age
    row[2], row[3] = mass, mass
    row[4], row[5] = math.log10(5772.0), 4.4
    row[7] = feh
    row[30] = gaia_g
    return row


def test_read_mist_grid_keeps_every_cell_and_replaces_its_cache(tmp_path):
    rows = [_mist_row(age, mass, feh)
            for age in (9.0, 9.1) for mass in (1.0, 1.2) for feh in (-0.2, 0.2)]
    path = _mist_archive(tmp_path, rows)
    grid = read_mist_grid(path)
    assert grid.shape == (8, 8) and np.all(grid[:, 7] > 0)
    assert set(grid[:, 5]) == {-0.2, 0.2}
    assert read_mist_grid(path) is grid


@pytest.mark.parametrize("mutation, message", [
    ({"header": ["EEP"] + [f"c{index}" for index in range(30)]}, "columns/units"),
    ({"rows": [_mist_row(9.0, mass, feh) for mass in (1.0, 1.2) for feh in (-0.2, 0.2)]},
     "at least two admitted ages"),
    ({"rows": [_mist_row(age, 1.0, feh) for age in (9.0, 9.1) for feh in (-0.2, 0.2)]},
     "mass coordinate"),
    ({"rows": [_mist_row(age, mass, 0.0) for age in (9.0, 9.1) for mass in (1.0, 1.2)]},
     "at least two \\[Fe/H\\]"),
])
def test_read_mist_grid_refuses_a_grid_without_adopted_cells(tmp_path, mutation, message):
    rows = mutation.get("rows", [_mist_row(age, mass, feh)
                                 for age in (9.0, 9.1) for mass in (1.0, 1.2) for feh in (-0.2, 0.2)])
    path = _mist_archive(tmp_path, rows, mutation.get("header"))
    with pytest.raises(ValueError, match=message):
        read_mist_grid(path)


def test_read_mist_grid_refuses_a_nonpositive_initial_mass(tmp_path):
    rows = [_mist_row(age, mass, feh)
            for age in (9.0, 9.1) for mass in (0.0, 1.2) for feh in (-0.2, 0.2)]
    with pytest.raises(ValueError, match="mass coordinate"):
        read_mist_grid(_mist_archive(tmp_path, rows))


def _valid_rows():
    return [_mist_row(age, mass, feh)
            for age in (9.0, 9.1) for mass in (1.0, 1.2) for feh in (-0.2, 0.2)]


def test_read_mist_grid_refuses_a_reddened_isochrone(tmp_path):
    with pytest.raises(ValueError, match="zero-extinction"):
        read_mist_grid(_mist_archive(tmp_path, _valid_rows(), extinction="0.500"))


def test_read_mist_grid_requires_the_declared_zero_extinction(tmp_path):
    with pytest.raises(ValueError, match="lacks its zero-extinction"):
        read_mist_grid(_mist_archive(tmp_path, _valid_rows(), extinction=None))


def test_read_mist_grid_cell_weights_are_global_across_member_layouts(tmp_path):
    rows = _valid_rows()
    by_age = [[row for row in rows if row[1] == age] for age in (9.0, 9.1)]
    by_feh = [[row for row in rows if row[7] == feh] for feh in (-0.2, 0.2)]
    by_mass = [[row for row in rows if row[2] == mass] for mass in (1.0, 1.2)]
    layouts = {"one": [("all.iso.cmd", rows)],
                "age": [("age_9.0.iso.cmd", by_age[0]), ("age_9.1.iso.cmd", by_age[1])],
               "feh": [("feh_-0.2.iso.cmd", by_feh[0]), ("feh_0.2.iso.cmd", by_feh[1])],
               "mass": [("mass_1.0.iso.cmd", by_mass[0]), ("mass_1.2.iso.cmd", by_mass[1])]}
    weights = {}
    for name, members in layouts.items():
        (tmp_path / name).mkdir()
        grid = read_mist_grid(_mist_archive(tmp_path / name, members=members))
        weights[name] = {(row[0], row[1], row[5]): row[7] for row in grid}
    assert weights["age"] == weights["feh"] == weights["mass"] == weights["one"]
    for name in ("age", "feh", "mass"):
        assert sum(weights[name].values()) == pytest.approx(sum(weights["one"].values()), rel=1e-15)


def test_repeated_eep_mass_shares_one_cell_across_archive_members(tmp_path):
    rows = _valid_rows()
    original = read_mist_grid(_mist_archive(tmp_path, rows))
    (tmp_path / "split").mkdir()
    repeated = read_mist_grid(_mist_archive(tmp_path / "split", members=[
        ("first.iso.cmd", rows), ("repeated.iso.cmd", [rows[0]]),
    ]))
    coordinate = (rows[0][1], rows[0][2], rows[0][7])
    match = ((repeated[:, 0] == coordinate[0]) & (repeated[:, 1] == coordinate[1])
             & (repeated[:, 5] == coordinate[2]))
    expected = original[(original[:, 0] == coordinate[0]) & (original[:, 1] == coordinate[1])
                        & (original[:, 5] == coordinate[2]), 7].item()
    assert repeated.shape[0] == original.shape[0] + 1 and np.count_nonzero(match) == 2
    np.testing.assert_allclose(repeated[match, 7], [expected / 2, expected / 2], rtol=1e-15)
    assert repeated[:, 7].sum() == pytest.approx(original[:, 7].sum(), rel=1e-15)


def test_read_mist_grid_cell_weight_matches_the_2x2x2_closed_form(tmp_path):
    grid = read_mist_grid(_mist_archive(tmp_path, _valid_rows()))
    dt = 10.0**9.1 - 10.0**9.0
    # Each (age, [Fe/H]) isochrone has one EEP row per mass, so dm = 0.2.
    dm = 1.2 - 1.0
    expected = {(age, mass, feh): dm * dt * 0.4 * mass**-2.35
                for age in (9.0, 9.1) for mass in (1.0, 1.2) for feh in (-0.2, 0.2)}
    actual = {(row[0], row[1], row[5]): row[7] for row in grid}
    assert actual.keys() == expected.keys()
    for key, weight in expected.items():
        assert actual[key] == pytest.approx(weight, rel=1e-15)
    assert grid[:, 7].sum() == pytest.approx(sum(expected.values()), rel=1e-15)


def test_read_mist_grid_accepts_a_single_age_per_file(tmp_path):
    rows = _valid_rows()
    split = read_mist_grid(_mist_archive(
        tmp_path, members=[("age_9.0.iso.cmd", rows[:4]), ("age_9.1.iso.cmd", rows[4:])]))
    assert split.shape == (8, 8) and np.all(split[:, 7] > 0)
    (tmp_path / "single").mkdir()
    single = read_mist_grid(_mist_archive(tmp_path / "single", rows))
    split_weights = {(row[0], row[1], row[5]): row[7] for row in split}
    single_weights = {(row[0], row[1], row[5]): row[7] for row in single}
    assert split_weights.keys() == single_weights.keys()
    for key, weight in split_weights.items():
        assert weight == single_weights[key]
    (tmp_path / "one_age").mkdir()
    with pytest.raises(ValueError, match="at least two admitted ages"):
        read_mist_grid(_mist_archive(tmp_path / "one_age", members=[
            ("a.iso.cmd", [_mist_row(9.0, mass, -0.2) for mass in (1.0, 1.2)]),
            ("b.iso.cmd", [_mist_row(9.0, mass, 0.2) for mass in (1.0, 1.2)]),
        ]))
