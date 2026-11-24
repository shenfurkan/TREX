"""TREX stellar, photometric and orbital model primitives."""

from __future__ import annotations

import numpy as np
import pytest

from trex.foundation.constants import au
from trex.stellar.stellar_relations import stellar_relations
from trex.photometry.photometric_relations import J_Ks_to_Tmag
from trex.models.flux_calibration import companion_flux_ratio
from trex.models.flux_calibration import dilute_flux
from trex.photometry.photometric_relations import delta_mag_to_flux_ratio
from trex.models.orbital_mechanics import semi_major_axis_cgs
from trex.models.orbital_mechanics import a_over_Rs
from trex.models.orbital_mechanics import impact_parameter
from trex.stellar_light.passband import tess_surface_brightness_ratio
from trex.photometry.photometric_relations import STASSUN_2018_FIRST_REL_COEFFS
from trex.photometry.photometric_relations import STASSUN_2018_SECOND_REL_COEFFS


# ============================================================================
# Funcs
# ============================================================================

def test_stellar_relations_solar():
    r, t = stellar_relations(np.array([1.0]))
    assert 1.0 < r[0] < 1.2
    assert 5500 < t[0] < 6000


def test_stellar_relations_vectorized():
    masses = np.array([0.3, 1.0, 2.0])
    r, _ = stellar_relations(masses)
    assert len(r) == 3
    assert r[0] < r[1] < r[2]


def test_stellar_relations_clamp():
    r, _ = stellar_relations(np.array([1.0]), max_radii=np.array([0.5]))
    assert r[0] == pytest.approx(0.5)


def test_tmag_conversion():
    tmag = J_Ks_to_Tmag(np.array([4.5]), np.array([4.0]))
    expected = 4.5 + 1.22163 * 0.125 - 1.74299 * 0.25 + 1.89115 * 0.5 + 0.0563
    assert tmag[0] == pytest.approx(expected)


def test_tmag_piecewise_boundaries_use_published_polynomials_and_reject_extrapolation():
    """Stassun (2018) defines two direct cubics; it specifies no join offset.

    The accompanying erratum (AJ 156, 183; DOI 10.3847/1538-3881/aade86)
    corrects CTL dwarf/subgiant selection, not this 2MASS-to-TESS transform.
    Upstream TRICERATOPS applies the same direct branches.  Therefore a test
    must neither require artificial continuity at J-Ks=0.70 nor extrapolate
    beyond the published [-0.1, 1.0] calibration domain.
    """
    first_colour = 0.70
    second_colour = np.nextafter(first_colour, np.inf)
    upper_colour = 1.0
    first_expected = first_colour + np.polyval(STASSUN_2018_FIRST_REL_COEFFS, first_colour)
    second_expected = second_colour + np.polyval(
        STASSUN_2018_SECOND_REL_COEFFS, second_colour
    )
    upper_expected = upper_colour + np.polyval(STASSUN_2018_SECOND_REL_COEFFS, upper_colour)

    converted = J_Ks_to_Tmag(
        np.array([-0.1, first_colour, second_colour, upper_colour]),
        np.zeros(4),
    )

    assert converted[0] == pytest.approx(
        -0.1 + np.polyval(STASSUN_2018_FIRST_REL_COEFFS, -0.1)
    )
    assert converted[1] == pytest.approx(first_expected)
    assert converted[2] == pytest.approx(second_expected)
    assert converted[3] == pytest.approx(upper_expected)
    with pytest.raises(ValueError, match="outside"):
        J_Ks_to_Tmag(np.array([np.nextafter(-0.1, -np.inf)]), np.array([0.0]))
    with pytest.raises(ValueError, match="outside"):
        J_Ks_to_Tmag(np.array([np.nextafter(1.0, np.inf)]), np.array([0.0]))


def test_tess_surface_brightness_uses_temperature():
    ratio = tess_surface_brightness_ratio(np.array([4000.0]), np.array([6000.0]))
    assert 0.0 < ratio[0] < 1.0


def test_flux_helpers():
    assert companion_flux_ratio(np.array([0.5]))[0] == pytest.approx(1.0)
    diluted = dilute_flux(np.array([0.99, 0.98, 1.0]), 0.0)
    np.testing.assert_array_equal(diluted, np.array([0.99, 0.98, 1.0]))


def test_contrast_roundtrip():
    dm = np.array([1.0, 2.5, 5.0])
    fr = delta_mag_to_flux_ratio(dm)
    from trex.photometry.photometric_relations import flux_ratio_to_delta_mag
    dm_back = flux_ratio_to_delta_mag(fr)
    np.testing.assert_allclose(dm, dm_back, rtol=1e-12)


def test_keplerian():
    a = semi_major_axis_cgs(365.25, 1.989e33)
    assert a / au == pytest.approx(1.0, rel=0.01)
    a_rs = a_over_Rs(365.25, 1.989e33, 6.957e10)
    assert a_rs == pytest.approx(215.0, rel=0.01)
    b = impact_parameter(90.0, 10.0)
    assert b == pytest.approx(0.0, abs=1e-10)
