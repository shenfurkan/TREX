"""Scenario priors against expectations derived outside the implementation.

Each expected value comes from the cited paper's formula, evaluated here by an
independent route (numerical quadrature or the retained TESS response read
directly), never from the production helpers under test.
"""

from importlib import resources

import numpy as np
import pytest
from astropy import constants
from scipy.integrate import quad

from trex.stellar_light import passband as trex_passband
from trex.populations import priors


def _g21_density(period, planet):
    """G21 Sect. 2.4.1 unnormalized broken power laws, continuous at the break [P in days]."""
    if planet:  # dN/dP ~ P^1.5 below 10 d, flat above
        return period ** 1.5 if period <= 10.0 else 10.0 ** 1.5
    return 0.3 ** -4.5 * period ** 5 if period <= 0.3 else period ** 0.5  # EB: P^5 below 0.3 d, P^0.5 above


def _g21_prior_by_quadrature(period, planet):
    def integral(low, high):
        return quad(_g21_density, low, high, args=(planet,), points=[0.3, 10.0], limit=200, epsabs=0, epsrel=1e-13)[0]

    return integral(period - 0.1, period + 0.1) / integral(0.1, 50.0)


@pytest.mark.parametrize("period", [1.0, 5.0, 10.0, 20.0])
def test_g21_period_priors_match_quadrature_of_the_paper_power_laws(period):
    assert priors.planet_period_prior(period) == pytest.approx(_g21_prior_by_quadrature(period, True), rel=1e-11)
    assert priors.eb_period_prior(period) == pytest.approx(_g21_prior_by_quadrature(period, False), rel=1e-11)


def test_moe2017_eq17_eccentricity_inverts_its_power_law_above_ten_days():
    eta = 0.6 - 0.7 / (np.log10(100.0) - 0.5)  # MDS17 Eq. 17 at P = 100 d
    x = np.linspace(0.0, 0.999, 101)
    e = priors.sample_ecc(x, planet=False, P_orb=100.0)
    np.testing.assert_allclose(e ** (1.0 + eta), x, rtol=1e-13, atol=1e-15)
    # Short periods keep the e^-0.8 (nu = 0.2) law: CDF e^0.2.
    np.testing.assert_allclose(priors.sample_ecc(x, planet=False, P_orb=5.0) ** 0.2, x, rtol=1e-12, atol=1e-15)


def test_radius_prior_switches_to_the_fgk_branch_only_above_the_m_dwarf_boundary():
    boundary = priors.RADIUS_PRIOR_M_DWARF_MAXIMUM_MASS_SOLAR
    assert boundary == 0.63
    radius = np.array([6.0])
    m_dwarf = priors.radius_quantile(radius, np.array([0.45]))
    assert priors.radius_quantile(radius, np.array([boundary])) == pytest.approx(m_dwarf, rel=0, abs=0)
    assert priors.radius_quantile(radius, np.array([0.64])) != pytest.approx(m_dwarf, rel=1e-3)


def _direct_tess_photon_flux(teff):
    path = resources.files("trex").joinpath("_data", "tess-response-function-v2.0.csv")
    wavelength_nm, response = np.loadtxt(path, delimiter=",", skiprows=7, unpack=True)
    wavelength_m = wavelength_nm * 1e-9
    h, c, k = constants.h.si.value, constants.c.si.value, constants.k_B.si.value
    planck = 2 * h * c ** 2 / wavelength_m ** 5 / np.expm1(h * c / (wavelength_m * k * teff))
    return np.trapezoid(response * planck * wavelength_m / (h * c), wavelength_m)  # photon-counting band


def test_band_surface_brightness_ratio_matches_a_direct_response_integral():
    expected = _direct_tess_photon_flux(4000.0) / _direct_tess_photon_flux(6000.0)
    assert trex_passband.tess_surface_brightness_ratio(4000.0, 6000.0) == pytest.approx(expected, rel=1e-4)
    assert trex_passband.tess_surface_brightness_ratio(5000.0, 5000.0) == 1.0
    with pytest.raises(ValueError):
        trex_passband.tess_surface_brightness_ratio(1500.0, 5000.0)
