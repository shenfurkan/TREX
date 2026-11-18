"""Synthetic controls for the tabulated A_lambda/A_V attenuation owner."""

import numpy as np
import pytest

from trex.stellar_light.extinction import ExtinctionCurve, ExtinctionUnsupportedError
from trex.stellar_light.passband import integrate_photon_intensity
from trex.stellar_light.release import AngularSpectrum
from trex.stellar_light.response import SpectralResponse

SOURCE = "synthetic attenuation table (retrieved 2026-11-16)"
APPLICABILITY = "synthetic sightline; unit-test domain only, no dust model"


def _curve(kappa=(0.0, 0.5, 1.0), wavelength_cm=(1.0e-4, 2.0e-4, 3.0e-4), **fields):
    parameters = {"r_v": 3.1, "source": SOURCE, "applicability": APPLICABILITY}
    parameters.update(fields)
    return ExtinctionCurve(
        np.asarray(wavelength_cm, dtype=float),
        np.asarray(kappa, dtype=float),
        **parameters,
    )


def test_explicit_zero_extinction_is_exactly_one_at_zero_and_positive_a_v():
    curve = ExtinctionCurve(
        np.array([1.0e-4, 2.0e-4]),
        np.zeros(2),
        None,
        "explicit zero-extinction model (retrieved 2026-11-16)",
        "all sightlines; no dust column is applied",
    )
    for av_mag in (0.0, 2.5):
        np.testing.assert_array_equal(curve.attenuation(av_mag), np.ones(2))
        assert curve.attenuation(av_mag, 1.5e-4) == 1.0
    # The A_V=0 limit is exact even where the table has no support.
    assert curve.attenuation(0.0, 9.0e-4) == 1.0


def test_nonzero_curve_requires_positive_finite_r_v():
    with pytest.raises(ValueError, match="R_V"):
        _curve(r_v=None)
    for r_v in (0.0, -2.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="R_V"):
            _curve(r_v=r_v)


def test_attenuation_matches_the_synthetic_linear_table_and_scales_with_a_v():
    curve = _curve(kappa=(0.0, 0.5, 1.0))
    query = np.array([1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4])
    expected_kappa = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        curve.at(query), expected_kappa, rtol=1e-14, atol=1e-15
    )
    av_mag = 0.7
    np.testing.assert_allclose(
        curve.attenuation(av_mag, query),
        10.0 ** (-0.4 * av_mag * expected_kappa),
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.log10(curve.attenuation(2.0 * av_mag, query)),
        2.0 * np.log10(curve.attenuation(av_mag, query)),
        rtol=1e-13,
        atol=1e-15,
    )


def test_at_reports_unsupported_wavelengths_and_refuses_bad_queries():
    curve = _curve()
    values = curve.at(np.array([0.5e-4, 1.0e-4, 3.0e-4, 4.0e-4]))
    assert np.isnan(values[[0, 3]]).all()
    np.testing.assert_array_equal(values[[1, 2]], [0.0, 1.0])
    assert np.isnan(curve.attenuation(1.0, 4.0e-4))
    assert curve.attenuation(0.0, 4.0e-4) == 1.0
    with pytest.raises(ValueError, match="finite nonnegative"):
        curve.at(-1.0e-4)
    with pytest.raises(ValueError, match="finite nonnegative"):
        curve.at(np.array([1.0e-4, np.nan]))


def test_on_grid_refuses_a_passband_that_extends_past_the_table():
    curve = _curve()
    np.testing.assert_array_equal(curve.on_grid([1.0e-4, 2.0e-4]), [0.0, 0.5])
    assert issubclass(ExtinctionUnsupportedError, ValueError)
    with pytest.raises(ExtinctionUnsupportedError, match="not covered"):
        curve.on_grid([1.0e-4, 2.0e-4, 4.0e-4])


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"wavelength_cm": (2.0e-4, 1.0e-4, 3.0e-4)}, "strictly increasing"),
        ({"kappa": (0.0, -0.1, 1.0)}, "nonnegative"),
        ({"source": "  "}, "extinction source"),
        ({"applicability": ""}, "extinction applicability"),
        ({"normalization": "A_V=1"}, "normalization"),
    ],
)
def test_invalid_curve_declarations_are_refused(fields, message):
    parameters = {
        "kappa": (0.0, 0.5, 1.0),
        "wavelength_cm": (1.0e-4, 2.0e-4, 3.0e-4),
    }
    parameters.update(fields)
    with pytest.raises(ValueError, match=message):
        _curve(**parameters)


def _response():
    return SpectralResponse(
        "SYNTHETIC/TEST.BAND",
        np.array([1.0e-4, 2.0e-4]),
        np.ones(2),
        "dimensionless_transmission",
        1,
        ("Filter", "Instrument"),
        "synthetic",
        "test",
        "band",
        "synthetic-v1",
        "unit test response, not a published calibration",
        "vacuum",
        "synthetic test table (retrieved 2026-11-16)",
    )


def _flat_spectrum(wavelength_cm):
    mu = np.linspace(0.01, 1.0, 100)
    return AngularSpectrum(
        4000.0, 4.5, 0.0, wavelength_cm, mu,
        np.ones((mu.size, wavelength_cm.size)), "synthetic flat radiance",
    )


def _integrated_flux(response, *, av_mag, a_lambda_over_av, provenance):
    passband = response.to_photon_passband(
        a_lambda_over_av=a_lambda_over_av,
        extinction_source=provenance,
    )
    return integrate_photon_intensity(
        _flat_spectrum(passband.wavelength_cm), passband, av_mag
    )[0]


def test_passband_boundary_returns_exact_kappa_and_provenance():
    response = _response()
    curve = _curve(kappa=(0.2, 0.6), wavelength_cm=(1.0e-4, 2.0e-4))
    kappa, provenance = curve.as_passband_extinction(response)
    np.testing.assert_array_equal(kappa, [0.2, 0.6])
    for text in (curve.source, curve.applicability, "normalization: A_V", "R_V=3.1"):
        assert text in provenance
    clear = _integrated_flux(
        response, av_mag=0.0, a_lambda_over_av=None,
        provenance="synthetic zero extinction",
    )
    equivalent = _integrated_flux(
        response, av_mag=0.0, a_lambda_over_av=kappa, provenance=provenance,
    )
    assert equivalent == pytest.approx(clear, rel=2e-15)


def test_grey_unit_kappa_equals_the_standard_magnitude_factor():
    response = _response()
    curve = ExtinctionCurve(
        np.array([0.5e-4, 2.5e-4]),
        np.ones(2),
        3.1,
        SOURCE,
        "synthetic grey sightline; no wavelength dependence",
    )
    kappa, provenance = curve.as_passband_extinction(response)
    np.testing.assert_array_equal(kappa, np.ones(2))
    clear = _integrated_flux(
        response, av_mag=0.0, a_lambda_over_av=None,
        provenance="synthetic zero extinction",
    )
    grey = _integrated_flux(
        response, av_mag=1.0, a_lambda_over_av=kappa, provenance=provenance,
    )
    assert grey / clear == pytest.approx(10.0 ** -0.4, rel=2e-15)
