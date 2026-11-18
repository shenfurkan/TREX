"""Packaged Roman/PLATO response assets load through the validating readers.

Expected values are read independently from the raw packaged bytes with plain
string splitting, so a reader that mis-parses a delimiter, unit or column fails.
"""

from importlib import resources

import numpy as np
import pytest

from trex.acquisition.passbands import (
    packaged_plato_transmission,
    packaged_roman_response,
)

_ROMAN = ("_data", "responses", "roman")
_PLATO = ("_data", "responses", "plato")


def _raw(parts, name):
    return resources.files("trex").joinpath(*parts, name).read_text(encoding="utf-8-sig")


def _data_lines(text):
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


@pytest.mark.parametrize("sca", range(1, 19))
def test_every_packaged_roman_sca_loads_its_own_f146_curve(sca):
    response = packaged_roman_response("F146", sca)

    lines = _data_lines(_raw(_ROMAN, f"Roman_effarea_v8_SCA{sca:02d}_20240301.ecsv"))
    header = [cell.strip() for cell in lines[0].split(",")]
    rows = [[float(cell) for cell in line.split(",")] for line in lines[1:]]
    column = header.index("F146")
    assert response.response_id == f"Roman/WFI{sca:02d}.F146"
    assert response.response_quantity == "effective_area_m2"
    assert response.components == ("Filter", "Instrument", "Detector")
    # Micrometres -> cm, unchanged effective area.
    np.testing.assert_array_equal(response.wavelength_cm, np.array([row[0] for row in rows]) * 1.0e-4)
    np.testing.assert_array_equal(response.response, np.array([row[column] for row in rows]))
    assert "2026-11-15" in response.source


def test_packaged_roman_zero_point_is_the_same_sca_and_filter_row():
    response = packaged_roman_response("F146", 7, zero_point_system="AB")

    rows = [line.split() for line in _data_lines(_raw(_ROMAN, "Roman_zeropoints_20240301.ecsv"))]
    header = rows[0]
    expected = next(float(row[header.index("ABMag")]) for row in rows[1:]
                    if row[0] == "WFI07" and row[1] == "F146")
    assert response.zero_point.system == "AB"
    assert response.zero_point.value == expected
    with pytest.raises(ValueError, match="zero point"):
        packaged_roman_response("F146", 7, zero_point_system="PLATO_P")


def test_historical_wfirst_label_is_refused_for_the_packaged_table():
    with pytest.raises(ValueError, match="historical WFIRST"):
        packaged_roman_response("W146", 1)


def test_packaged_plato_ncam_columns_stay_distinct_and_exclude_detector_qe():
    expected = packaged_plato_transmission("NCAM", "expected transmission")
    designed = packaged_plato_transmission("NCAM", "as designed camera effective transmission")

    lines = _data_lines(_raw(_PLATO, "optical_transmission_ncam_v2.csv"))
    header = [cell.strip().lower() for cell in lines[0].split(",")]
    first = [cell.strip() for cell in lines[1].split(",")]
    assert float(first[0]) == pytest.approx(500.0)
    assert expected.response[0] == float(first[header.index("expected transmission")])
    assert expected.wavelength_cm[0] == pytest.approx(500.0e-7, rel=0, abs=1e-18)
    assert expected.response_id != designed.response_id
    assert expected.components == ("Filter", "Instrument")
    assert "Detector" not in designed.components


def test_packaged_plato_requirement_column_is_not_a_response():
    with pytest.raises(ValueError, match="not a published transmission model"):
        packaged_plato_transmission("NCAM", "required bol transmission")


def test_packaged_plato_fcam_channels_read_their_own_semicolon_table():
    blue = packaged_plato_transmission("FCAM_BLUE", "blue filter mission")
    red = packaged_plato_transmission("FCAM_RED", "red filter mission")
    assert blue.instrument == "FCAM_BLUE" and red.instrument == "FCAM_RED"
    assert not np.array_equal(blue.response, red.response)


def test_roman2024_trilegal_columns_are_the_packaged_wfi_bands():
    from trex.acquisition.trilegal import TRILEGAL_ROMAN_COLUMNS

    lines = _data_lines(_raw(_ROMAN, "Roman_effarea_v8_SCA07_20240301.ecsv"))
    header = [cell.strip() for cell in lines[0].split(",")]
    # Same filter set; the STScI table lists F146 after F184, and no consumer reads order.
    assert sorted(name for name in header if name.startswith("F")) == sorted(TRILEGAL_ROMAN_COLUMNS)
