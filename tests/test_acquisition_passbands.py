"""Synthetic file-boundary checks for the published response-table readers."""

import numpy as np
import pytest
from astropy import constants as const

from trex.acquisition.passbands import (
    read_plato_transmission,
    read_roman_effective_area,
    read_roman_zero_points,
    read_svo_response,
)
from trex.stellar_light.passband import integrate_photon_intensity
from trex.stellar_light.release import AngularSpectrum
from trex.stellar_light.response import require_common_zero_point_system

SOURCE_URL = "https://example.org/synthetic-response-table"
RETRIEVED_UTC = "2026-11-16"
ROMAN_RELEASE = "GSFC-v8-20240301"
ROMAN_AREA_FILENAME = "Roman_effarea_v8_SCA03_20240301.ecsv"
SVO_WAVELENGTHS_ANGSTROM = (4000.0, 5000.0, 6000.0)


def _svo_file(tmp_path, name, *, transmission=(0.1, 0.2, 0.3), query_status="OK",
              declared=None, transmission_unit="dimensionless"):
    parameters = {
        "filterID": "P200/WIRC.J_ori",
        "WavelengthUnit": "Angstrom",
        "DetectorType": "0",
        "components": "Filter",
        "Facility": "Palomar",
        "Instrument": "WIRC",
        "MagSys": "Vega",
        "ZeroPoint": "1600.0",
        "ZeroPointUnit": "Jy",
        "ZeroPointType": "Pogson",
        "PhotCalID": "P200/WIRC.J_ori/WIRC.J_ori",
    }
    if declared:
        parameters.update(declared)
    param_xml = "".join(
        '<PARAM name="{0}" datatype="char" arraysize="*" value="{1}"/>'.format(key, value)
        for key, value in parameters.items() if value is not None)
    rows = "".join(
        "<TR><TD>{0}</TD><TD>{1}</TD></TR>".format(wave, value)
        for wave, value in zip(SVO_WAVELENGTHS_ANGSTROM, transmission))
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<VOTABLE version="1.3"><RESOURCE type="results">'
        '<INFO name="QUERY_STATUS" value="{0}"/>'.format(query_status)
        + param_xml
        + '<TABLE><FIELD name="Wavelength" datatype="double" unit="Angstrom"/>'
        '<FIELD name="Transmission" datatype="double" unit="{0}"/>'.format(transmission_unit)
        + "<DATA><TABLEDATA>" + rows + "</TABLEDATA></DATA></TABLE>"
        "</RESOURCE></VOTABLE>",
        encoding="utf-8")
    return path


def _read_svo(path):
    return read_svo_response(
        path,
        source_url=SOURCE_URL,
        retrieved_utc=RETRIEVED_UTC,
        wavelength_convention="vacuum",
    )


def _flat_photon_flux(response, *, normalized=True):
    """Integrate a synthetic flat I_lambda=1 spectrum over the response grid."""
    passband = response.to_photon_passband(
        extinction_source="synthetic zero extinction", normalized=normalized)
    mu = np.linspace(0.01, 1.0, 100)
    spectrum = AngularSpectrum(
        4000.0, 4.5, 0.0, passband.wavelength_cm, mu,
        np.ones((mu.size, passband.wavelength_cm.size)), "synthetic flat radiance")
    return integrate_photon_intensity(spectrum, passband, 0.0)[0]


def _roman_area_file(directory, *, scale=1.0,
                     declaration="# Effective area for Roman (m^2)"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ROMAN_AREA_FILENAME
    waves = (0.62, 0.75, 0.90)
    areas = tuple(scale * value for value in (0.9, 1.2, 1.1))
    rows = "".join(
        "{0},{1},{2}\n".format(wave, 0.8 * area, area)
        for wave, area in zip(waves, areas))
    path.write_text(
        declaration + "\nWave,F062,F146\n" + rows, encoding="utf-8")
    return path


def _read_roman_area(path, *, band="F146", sca=3, zero_point=None):
    return read_roman_effective_area(
        path,
        band=band,
        sca=sca,
        wavelength_unit="um",
        detector_type=1,
        source_url=SOURCE_URL,
        retrieved_utc=RETRIEVED_UTC,
        release=ROMAN_RELEASE,
        zero_point=zero_point,
    )


def _roman_zero_point_file(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "Roman_zero_points_v8_20240301.ecsv"
    path.write_text(
        "# Roman/WFI zero points, GSFC-v8-20240301\n"
        "detector,element,VegaMag,ABMag,STMag\n"
        "WFI03,F062,24.5,25.6,27.4\n"
        "WFI03,F146,24.1,25.2,26.1\n",
        encoding="utf-8")
    return path


_PLATO_COMMENTS = (
    "# PLATO DLR-TN-0113 v2 N-CAM transmission\n"
    "# column 1 wavelength in nm\n")
_PLATO_HEADER = (
    "wavelength,expected transmission,as designed camera effective transmission,"
    "required bol transmission")


def _plato_file(directory, *, selected=("0.20", "0.21", "0.22"),
                comments=_PLATO_COMMENTS):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plato-ncam.csv"
    rows = "".join(
        "{0},0.1,{1},0.3\n".format(wave, value)
        for wave, value in zip((500, 600, 700), selected))
    path.write_text(comments + _PLATO_HEADER + "\n" + rows, encoding="utf-8")
    return path


def _read_plato(path, *, channel="NCAM",
                column="as designed camera effective transmission",
                response_id="PLATO/NCAM.AS_DESIGNED"):
    return read_plato_transmission(
        path,
        channel=channel,
        column=column,
        response_id=response_id,
        source_url=SOURCE_URL,
        retrieved_utc=RETRIEVED_UTC,
        release="DLR-TN-0113-v2",
        detector_type=1,
    )


def test_svo_empty_transmission_unit_is_the_declared_dimensionless_convention(tmp_path):
    response = _read_svo(_svo_file(tmp_path, "wirc.xml", transmission_unit=""))
    assert response.response_quantity == "dimensionless_transmission"
    assert response.detector_type == 0


def test_svo_records_keep_their_declared_identity_and_convention(tmp_path):
    wirc = _read_svo(_svo_file(tmp_path, "wirc.xml"))
    sdss = _read_svo(_svo_file(
        tmp_path, "sdss_z.xml", declared={
            "filterID": "SLOAN/SDSS.z",
            "DetectorType": "1",
            "components": "Filter + Instrument + Atmosphere",
            "Facility": "SDSS",
            "Instrument": "SDSS",
            "MagSys": "AB",
            "ZeroPoint": "3631.0",
            "PhotCalID": "SLOAN/SDSS.z/SDSS.z",
        }))

    assert wirc.response_id == "P200/WIRC.J_ori"
    assert wirc.detector_type == 0
    assert wirc.components == ("Filter",)
    np.testing.assert_allclose(
        wirc.wavelength_cm, [4.0e-5, 5.0e-5, 6.0e-5], rtol=1e-15)
    assert (wirc.zero_point.system, wirc.zero_point.unit) == ("Vega", "Jy")

    assert sdss.response_id == "SLOAN/SDSS.z"
    assert sdss.detector_type == 1
    assert sdss.components == ("Filter", "Instrument", "Atmosphere")
    assert sdss.response_id != wirc.response_id
    assert sdss.detector_type != wirc.detector_type
    assert sdss.components != wirc.components


def test_energy_and_photon_svo_conventions_differ_by_the_inverse_wavelength(tmp_path):
    energy = _read_svo(_svo_file(
        tmp_path, "energy.xml", transmission=(0.5, 0.5, 0.5)))
    photon = _read_svo(_svo_file(
        tmp_path, "photon.xml", transmission=(0.5, 0.5, 0.5),
        declared={"DetectorType": "1"}))

    energy_flux = _flat_photon_flux(energy, normalized=False)
    photon_flux = _flat_photon_flux(photon, normalized=False)
    wave = 1.0e-8 * np.asarray(SVO_WAVELENGTHS_ANGSTROM)
    # Absolute signals: S_E = integral I R dlambda and S_P = integral
    # (lambda/hc) I R dlambda. A flat R then gives S_E/S_P = 2hc/(lambda1+lambda2);
    # this inverse-wavelength factor is the energy-counter convention.
    inverse_wavelength = (
        2.0 * const.h.cgs.value * const.c.cgs.value / (wave[0] + wave[-1]))
    assert energy_flux / photon_flux == pytest.approx(inverse_wavelength, rel=2e-14)


@pytest.mark.parametrize("name, transmission, declared, query_status, match", [
    ("negative.xml", (-0.01, 0.2, 0.3), None, "OK", "negative entries"),
    ("bare.xml", (0.1, 0.2, 0.3), {"filterID": "J"}, "OK", "instrument-qualified"),
    ("no_zero_point.xml", (0.1, 0.2, 0.3), {"ZeroPoint": None}, "OK", "without a zero point"),
    ("no_components.xml", (0.1, 0.2, 0.3), {"components": None}, "OK", "SVO components"),
    ("bad_status.xml", (0.1, 0.2, 0.3), None, "ERROR", "successful query"),
])
def test_svo_reader_refuses_undeclared_or_invalid_provenance(
        tmp_path, name, transmission, declared, query_status, match):
    path = _svo_file(tmp_path, name, transmission=transmission,
                     query_status=query_status, declared=declared)
    with pytest.raises(ValueError, match=match):
        _read_svo(path)


def test_roman_effective_area_keeps_m2_values_and_per_sca_identity(tmp_path):
    base_path = _roman_area_file(tmp_path / "base")
    base = _read_roman_area(base_path)
    assert base.response_id == "Roman/WFI03.F146"
    assert base.response_quantity == "effective_area_m2"
    np.testing.assert_array_equal(base.response, [0.9, 1.2, 1.1])
    np.testing.assert_allclose(
        base.wavelength_cm, [6.2e-5, 7.5e-5, 9.0e-5], rtol=1e-15)

    scaled = _read_roman_area(_roman_area_file(tmp_path / "scaled", scale=4.0))
    assert _flat_photon_flux(scaled, normalized=False) == pytest.approx(
        4.0 * _flat_photon_flux(base, normalized=False), rel=2e-14)
    np.testing.assert_allclose(
        scaled.photon_kernel_response(normalized=True),
        base.photon_kernel_response(normalized=True), rtol=2e-16)

    with pytest.raises(ValueError, match="does not identify the selected SCA"):
        _read_roman_area(base_path, sca=4)

    undeclared = _roman_area_file(
        tmp_path / "undeclared", declaration="# Roman WFI effective area")
    with pytest.raises(ValueError, match=r"m\^2 quantity"):
        _read_roman_area(undeclared)


def test_roman_zero_points_keep_their_systems_and_never_mix_silently(tmp_path):
    points = read_roman_zero_points(
        path=_roman_zero_point_file(tmp_path / "zero-points"),
        source_url=SOURCE_URL,
        retrieved_utc=RETRIEVED_UTC,
        release=ROMAN_RELEASE,
    )
    assert set(points) == {(3, "F062"), (3, "F146")}
    zero = points[(3, "F146")]
    assert set(zero) == {"Vega", "AB", "ST"}
    assert (zero["Vega"].system, zero["Vega"].unit) == ("Vega", "mag")
    assert zero["AB"].system == "AB"
    assert zero["ST"].system == "ST"
    assert zero["Vega"].value == pytest.approx(24.1)

    area_path = _roman_area_file(tmp_path / "area")
    vega_f062 = _read_roman_area(
        area_path, band="F062", zero_point=points[(3, "F062")]["Vega"])
    vega_f146 = _read_roman_area(area_path, zero_point=zero["Vega"])
    ab_f146 = _read_roman_area(area_path, zero_point=zero["AB"])
    assert require_common_zero_point_system([vega_f062, vega_f146]) == "Vega"
    with pytest.raises(ValueError, match="incompatible zero-point systems"):
        require_common_zero_point_system([vega_f146, ab_f146])


def test_plato_transmission_selects_the_declared_as_designed_model(tmp_path):
    response = _read_plato(_plato_file(tmp_path / "ncam"))
    assert response.response_id == "PLATO/NCAM.AS_DESIGNED"
    assert response.response_quantity == "dimensionless_transmission"
    assert "DLR-TN-0113-v2" in response.edition
    assert "as-designed" in response.calibration_context
    assert "pre-launch" in response.calibration_context
    np.testing.assert_array_equal(response.response, [0.2, 0.21, 0.22])
    np.testing.assert_allclose(
        response.wavelength_cm, [5.0e-5, 6.0e-5, 7.0e-5], rtol=1e-15)


def test_plato_reader_refuses_requirements_and_undeclared_tables(tmp_path):
    path = _plato_file(tmp_path / "ncam")
    with pytest.raises(ValueError, match="not a published transmission model"):
        _read_plato(path, column="required bol transmission")
    with pytest.raises(ValueError, match="does not match the selected camera channel"):
        _read_plato(path, channel="FCAM_BLUE",
                    column="expected transmission blue camera")
    with pytest.raises(ValueError, match="does not name the selected NCAM channel"):
        _read_plato(path, response_id="PLATO/FCAM_BLUE.AS_DESIGNED")
    with pytest.raises(ValueError, match="non-finite"):
        _read_plato(_plato_file(
            tmp_path / "nan", selected=("0.20", "NaN", "0.22")))
    undeclared = _plato_file(
        tmp_path / "no-nm",
        comments="# PLATO DLR-TN-0113 v2 N-CAM transmission\n")
    with pytest.raises(ValueError, match="declare wavelength in nm"):
        _read_plato(undeclared)


def test_retrieved_utc_must_be_an_iso_utc_date(tmp_path):
    path = _roman_zero_point_file(tmp_path / "zero-points")
    with pytest.raises(ValueError, match="ISO 8601"):
        read_roman_zero_points(
            path=path,
            source_url=SOURCE_URL,
            retrieved_utc="16/11/2026",
            release=ROMAN_RELEASE,
        )


def test_historical_wfirst_names_are_not_modern_roman_throughputs(tmp_path):
    path = _roman_area_file(tmp_path / "roman")
    with pytest.raises(ValueError, match="historical WFIRST band"):
        _read_roman_area(path, band="W149")
    with pytest.raises(ValueError, match="historical WFIRST band"):
        _read_roman_area(path, band="Z087")
