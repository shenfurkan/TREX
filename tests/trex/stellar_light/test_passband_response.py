"""Analytic photon/energy and area controls for the response boundary."""

import numpy as np
import pytest
from astropy import constants as const

from trex.stellar_light.passband import PhotonPassband, integrate_photon_intensity
from trex.stellar_light.release import AngularSpectrum
from trex.stellar_light.response import (
    SpectralResponse,
    ZeroPoint,
    is_qualified_response_id,
    require_common_zero_point_system,
    require_response_zero_point_edition,
)


def _spectrum(wavelength_cm):
    mu = np.linspace(0.01, 1.0, 100)
    return AngularSpectrum(
        4000.0, 4.5, 0.0, wavelength_cm, mu,
        np.ones((mu.size, wavelength_cm.size)), "synthetic flat radiance",
    )


def _response(response, *, quantity="dimensionless_transmission", detector_type=1,
              zero_point=None, edition="synthetic-v1"):
    return SpectralResponse(
        "SYNTHETIC/TEST.BAND",
        np.array([1.0e-4, 2.0e-4]),
        np.asarray(response, dtype=float),
        quantity,
        detector_type,
        ("Filter", "Instrument"),
        "synthetic",
        "test",
        "band",
        edition,
        "unit test response, not a published calibration",
        "vacuum",
        "synthetic test table (retrieved 2026-11-16)",
        zero_point,
    )


def _disc_moment(mu):
    # The integrator inserts the zero-path tangent at mu=0; a uniform profile
    # leaves exactly the missing triangular sliver below the first ray.
    return 0.5 - mu[0] ** 2 / 6


def _integrated(response, *, av_mag=0.0, normalized=False, extinction=None,
                extinction_source="synthetic zero extinction"):
    passband = response.to_photon_passband(
        a_lambda_over_av=extinction,
        extinction_source=extinction_source,
        normalized=normalized,
    )
    spectrum = _spectrum(passband.wavelength_cm)
    flux = integrate_photon_intensity(spectrum, passband, av_mag)[0]
    return flux, passband, spectrum


def test_photon_counter_flat_box_matches_the_analytic_photon_integral():
    response = _response([1.0, 1.0])
    flux, passband, spectrum = _integrated(response)
    wave = passband.wavelength_cm
    expected = (
        2.0 * np.pi * (wave[-1] ** 2 - wave[0] ** 2)
        / (2.0 * const.h.cgs.value * const.c.cgs.value)
        * _disc_moment(spectrum.mu)
    )
    assert flux == pytest.approx(expected, rel=2e-14)


def test_energy_counter_uses_inverse_wavelength_not_another_lambda_factor():
    """An energy signal through the photon kernel is integral(I_lambda R dlambda)."""
    response = _response([1.0, 1.0], detector_type=0)
    flux, passband, spectrum = _integrated(response)
    wave = passband.wavelength_cm
    expected = 2.0 * np.pi * (wave[-1] - wave[0]) * _disc_moment(spectrum.mu)
    assert flux == pytest.approx(expected, rel=2e-14)
    # A photon count of the same tabulated curve would instead scale with
    # (lambda2^2 - lambda1^2)/(2hc); the two conventions cannot be interchanged.
    photon = _integrated(_response([1.0, 1.0], detector_type=1))[0]
    assert flux != pytest.approx(photon, rel=1e-6)
    assert flux / photon == pytest.approx(
        2.0 * const.h.cgs.value * const.c.cgs.value / (wave[0] + wave[-1]), rel=2e-14
    )


def test_effective_area_keeps_absolute_area_while_normalized_shape_is_unchanged():
    area = _response([1.0e4, 2.0e4], quantity="effective_area_m2")
    scaled = _response([2.0e4, 4.0e4], quantity="effective_area_m2")
    area_flux, area_passband, _ = _integrated(area, normalized=False)
    scaled_flux, _, _ = _integrated(scaled, normalized=False)
    # m^2 -> cm^2 collecting area; a common scale factor stays absolute.
    np.testing.assert_array_equal(
        area_passband.response, np.asarray(area.response) * 1.0e4)
    assert scaled_flux == pytest.approx(2.0 * area_flux, rel=2e-14)
    np.testing.assert_allclose(
        np.asarray(scaled.photon_kernel_response(normalized=True)),
        np.asarray(area.photon_kernel_response(normalized=True)),
        rtol=2e-16,
    )
    assert _integrated(area, normalized=True)[0] == pytest.approx(
        area_flux / np.max(area_passband.response), rel=2e-14)


def test_extinction_is_applied_exactly_once_in_the_integrated_band():
    response = _response([1.0, 1.0])
    clear = _integrated(response)[0]
    reddened, _, _ = _integrated(
        response, av_mag=1.0, extinction=[1.0, 1.0],
        extinction_source="synthetic grey A_lambda/A_V=1",
    )
    assert reddened / clear == pytest.approx(10.0 ** -0.4, rel=2e-15)


def test_response_identity_rejects_bare_band_names_and_tracks_exact_bytes():
    assert is_qualified_response_id("P200/WIRC.J_ori")
    assert not is_qualified_response_id("J")
    with pytest.raises(ValueError, match="instrument-qualified"):
        SpectralResponse(
            "J", np.array([1e-4, 2e-4]), np.ones(2),
            "dimensionless_transmission", 0, ("Filter",),
            "Palomar", "WIRC", "J", "e", "context", "vacuum", "source",
        )
    first = _response([1.0, 2.0])
    second = _response([1.0, 2.0000001])
    assert first.input_id != second.input_id
    assert first.input_id == _response([1.0, 2.0]).input_id


def test_zero_point_systems_are_never_mixed_silently():
    vega = ZeroPoint("Vega", 1546.7, "Jy", "synthetic Vega zero point")
    ab = ZeroPoint("AB", 26.5, "mag", "synthetic AB zero point")
    shared = require_common_zero_point_system(
        [_response([1.0, 1.0], zero_point=vega),
         _response([1.0, 1.0], zero_point=ZeroPoint("Vega", 1600.0, "Jy", "synthetic"))]
    )
    assert shared == "Vega"
    with pytest.raises(ValueError, match="incompatible zero-point systems"):
        require_common_zero_point_system(
            [_response([1.0, 1.0], zero_point=vega),
             _response([1.0, 1.0], zero_point=ab)]
        )
    with pytest.raises(ValueError, match="declares no zero point"):
        require_common_zero_point_system([_response([1.0, 1.0])])


def test_response_and_zero_point_must_come_from_the_same_release_edition():
    vega = ZeroPoint(
        "Vega", 26.4, "mag", "synthetic GSFC release", "GSFC-v8-20240301")
    response = _response([1.0, 1.0], edition="GSFC-v8-20240301; retrieved 2026-11-16")
    assert require_response_zero_point_edition(response, vega) is vega
    other = _response([1.0, 1.0], edition="GSFC-v7-20220101; retrieved 2026-11-16")
    with pytest.raises(ValueError, match="does not match zero-point release"):
        require_response_zero_point_edition(other, vega)
    with pytest.raises(ValueError, match="declares no release edition"):
        require_response_zero_point_edition(
            response, ZeroPoint("Vega", 26.4, "mag", "synthetic, no edition"))
    attached = ZeroPoint(
        "AB", 26.5, "mag", "synthetic GSFC release", "GSFC-v8-20240301")
    with pytest.raises(ValueError, match="carries a different zero point"):
        require_response_zero_point_edition(
            _response([1.0, 1.0], edition="GSFC-v8-20240301", zero_point=attached),
            vega,
        )


def test_existing_tess_passband_algebra_and_identity_are_preserved():
    from trex.stellar_light.passband import _tess_photon_passband

    band = _tess_photon_passband()
    assert band.response_id == "TESS/TESS.T"
    assert isinstance(band.input_id, str) and len(band.input_id) == 64
    legacy = PhotonPassband(
        band.wavelength_cm, band.response, band.a_lambda_over_av,
        band.response_source, band.extinction_source,
    )
    assert legacy.response_id == ""
    spectrum = _spectrum(band.wavelength_cm)
    scipy_flux = integrate_photon_intensity(spectrum, band, 0.0)[0]
    legacy_flux = integrate_photon_intensity(spectrum, legacy, 0.0)[0]
    assert scipy_flux == legacy_flux
