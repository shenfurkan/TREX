"""TICv8 G -> T conversion follows Stassun et al. (2019) Sect. 2.3.1 domain by domain."""

import math

import pytest

from trex.acquisition.photometry import STASSUN_2019_G_TO_T_COEFFS, tess_magnitude_from_gaia_photometry


def test_zero_color_gives_the_polynomial_constant():
    tess, relation = tess_magnitude_from_gaia_photometry(10.0, 12.0, 12.0)
    assert relation == "stassun2019-eq1" and tess == pytest.approx(10.0 + STASSUN_2019_G_TO_T_COEFFS.c0, abs=1e-12)


@pytest.mark.parametrize("bp, rp, relation", [
    # 0.4 is exactly twice 0.2 in binary floating point, so this BP - RP is the
    # literal bound -0.2. Rebuilding the color as 10.0 + (-0.2) - 10.0 gives
    # -0.1999999999999993, i.e. not the bound being tested.
    (0.2, 0.4, "stassun2019-eq1-tic-extrapolated"),
    (13.5, 10.0, "stassun2019-eq1-tic-extrapolated"),
    (15.9, 10.0, "stassun2019-eq1-tic-extrapolated"),
    (9.0, 10.0, "stassun2019-eq2-no-valid-color"),
    (16.0, 10.0, "stassun2019-eq2-no-valid-color"),
    (17.0, 10.0, "stassun2019-eq2-no-valid-color"),
])
def test_interval_bounds_are_exclusive(bp, rp, relation):
    tess, label = tess_magnitude_from_gaia_photometry(10.0, bp, rp)
    assert label == relation
    if relation.endswith("no-valid-color"):
        assert tess == pytest.approx(10.0 - 0.430, abs=1e-12)


@pytest.mark.parametrize("bp, rp", [(None, 9.0), (9.0, float("nan")), (True, 9.0), ("x", 9.0)])
def test_missing_color_uses_the_solar_color_offset(bp, rp):
    assert tess_magnitude_from_gaia_photometry("10.5", bp, rp) == (pytest.approx(10.07), "stassun2019-eq2-no-valid-color")


@pytest.mark.parametrize("g", [None, True, False, math.inf, "nan", ""])
def test_missing_g_gives_no_magnitude(g):
    assert tess_magnitude_from_gaia_photometry(g, 10.0, 9.0) == (None, None)
