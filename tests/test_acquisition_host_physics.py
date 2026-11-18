"""Host stellar-physics precedence and the catalog flux-ratio aperture shares."""

import math

import numpy as np
import pytest

from trex.acquisition.aperture import catalog_source_contributions, gaussian_psf_aperture_throughput
from trex.acquisition.host_physics import select_host_physics

GSPPHOT = {"teff_gspphot_k": 5772.0, "radius_gspphot_solar": 1.0, "logg_gspphot_cgs": 4.438}
TIC = {"Teff": 3300.0, "rad": 0.3, "mass": 0.28, "logg": 4.9, "e_mass": 0.02}


def test_flame_mass_wins_then_gravity_then_tic():
    flame = select_host_physics({**GSPPHOT, "mass_flame": 1.05, "radius_flame": 1.0,
                                 "mass_flame_lower_solar": 0.95, "mass_flame_upper_solar": 1.15,
                                 "radius_flame_lower_solar": 0.9, "radius_flame_upper_solar": 1.1,
                                 "flags_flame": "00"}, TIC)
    assert flame["mass_solar"] == 1.05 and flame["field_sources"]["mass_solar"] == "gaia-dr3-flame"
    # Without FLAME, M = g R^2 / G: the IAU nominal Sun has log g = 4.438 (cgs), so M ~ 1 Msun.
    gravity = select_host_physics(GSPPHOT, TIC)
    assert gravity["field_sources"]["mass_solar"] == "derived-gravitational-logg-radius"
    assert math.isclose(gravity["mass_solar"], 1.0, rel_tol=0.01) and gravity["teff_k"] == 5772.0
    # No GSP-Phot solution: every field comes from the TIC v8 row.
    tic = select_host_physics({}, TIC)
    assert (tic["teff_k"], tic["radius_solar"], tic["mass_solar"]) == (3300.0, 0.3, 0.28)
    assert set(tic["field_sources"].values()) == {"tic-v8-stassun2019"}


def test_cool_dwarf_list_overrides_gspphot_and_missing_sources_raise():
    cool = select_host_physics({**GSPPHOT, "mass_flame": 1.05, "radius_flame": 1.0,
                                "flags_flame": "00"}, {**TIC, "splists": "cooldwarfs_v8"})
    assert cool["mass_solar"] == 0.28 and cool["field_sources"]["teff_k"] == "tic-v8-cool-dwarf-list-stassun2019"
    with pytest.raises(ValueError, match="host lacks"):
        select_host_physics({}, None)


def test_a_flame_radius_without_an_admissible_mass_falls_back_to_tic():
    from trex.acquisition.host_physics import resolve_target_mass

    # FLAME applicability needs its mass as well as its radius, even with good quality flags.
    target = {"logg_gspphot_cgs": 4.0, "radius_flame": 1.0, "flags_flame": "00",
              "radius_flame_lower_solar": 0.9, "radius_flame_upper_solar": 1.1}
    assert resolve_target_mass(target, lambda: (TIC["mass"], TIC["e_mass"])) == (
        0.28, 0.02, "tic-v8-stassun2019")
    assert resolve_target_mass(target, lambda: None) == (None, None, None)


def test_gravity_mass_error_uses_gspphot_bounds_not_unvalidated_flame_bounds():
    from trex.acquisition.host_physics import resolve_target_mass

    target = {"logg_gspphot_cgs": 4.0, "radius_gspphot_solar": 1.0, "radius_flame": 1.0,
              "radius_gspphot_lower_solar": 0.9, "radius_gspphot_upper_solar": 1.1,
              "radius_flame_lower_solar": 0.5, "radius_flame_upper_solar": 1.5,
              "logg_gspphot_lower_cgs": 3.9, "logg_gspphot_upper_cgs": 4.1}
    mass, error, source = resolve_target_mass(target, lambda: pytest.fail("TIC mass must not be read"))
    assert source == "derived-gravitational-logg-radius"
    logg_sigma = 10.0 * (4.1 - 3.9) / 2.0
    radius_sigma = (1.1 - 0.9) / 2.0
    expected = mass * math.hypot(math.log(10.0) * logg_sigma, 2.0 * radius_sigma)
    assert error == pytest.approx(expected, rel=1e-15)
    assert error != pytest.approx(mass * math.hypot(math.log(10.0) * logg_sigma,
                                                    2.0 * (1.5 - 0.5) / 2.0))


def test_single_pixel_throughput_and_symmetric_flux_shares():
    # A unit Gaussian centred on one pixel keeps erf(0.5 / (sigma sqrt 2))^2 of its flux there.
    sigma = 0.75
    one_pixel = np.zeros((5, 5), bool)
    one_pixel[2, 2] = True
    expected = math.erf(0.5 / (sigma * math.sqrt(2.0))) ** 2
    assert math.isclose(gaussian_psf_aperture_throughput(one_pixel, 2.0, 2.0, sigma), expected, rel_tol=1e-12)
    # Two equal-flux sources mirrored about the aperture centre share its flux equally.
    aperture = np.zeros((7, 7), bool)
    aperture[3, 2:5] = True
    positions = [{"source_id": "a", "is_target": True, "x_pixel": 2.5, "y_pixel": 3.0},
                 {"source_id": "b", "is_target": False, "x_pixel": 3.5, "y_pixel": 3.0}]
    contributions, total = catalog_source_contributions(
        aperture, positions, [(10.0, "stassun2019-eq1")] * 2, [1.0, 1.0], 1)
    assert [c["aperture_flux_fraction"] for c in contributions] == pytest.approx([0.5, 0.5], abs=1e-15)
    assert math.isclose(total, 2 * contributions[0]["aperture_throughput"], rel_tol=1e-15)


def test_aperture_reach_is_farthest_corner_plus_four_sigma_in_arcsec():
    from astropy.wcs import WCS
    from trex.acquisition.aperture import aperture_reach_arcsec

    # TAN WCS at 21 arcsec per pixel with the target on (0-based) pixel (3, 3).
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [100.0, -30.0]
    wcs.wcs.crpix = [4.0, 4.0]
    wcs.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0]
    aperture = np.zeros((7, 7), dtype=np.int32)
    aperture[3, 3:6] = 2  # SPOC pipeline-aperture bit; farthest pixel centre 2 px east of the target
    cube = {"aperture_header": wcs.to_header(), "aperture": aperture, "flux": np.zeros((1, 7, 7))}
    expected = (2.0 + 0.5 * math.sqrt(2.0) + 4.0 * 0.75) * 21.0
    assert math.isclose(aperture_reach_arcsec(cube), expected, rel_tol=1e-9)
