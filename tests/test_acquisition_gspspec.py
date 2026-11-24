"""GSP-Spec calibration on synthetic rows."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from trex.acquisition.gspspec import GSP_SPEC_COLUMNS, calibrate_gspspec_mh, fetch_gspspec, parse_gspspec


def _payload(*rows):
    fields = "".join(f'<FIELD name="{name}" datatype="char" arraysize="*"/>' for name in GSP_SPEC_COLUMNS)
    body = "".join("<TR>" + "".join(f"<TD>{row[name]}</TD>" for name in GSP_SPEC_COLUMNS) + "</TR>"
                   for row in rows)
    return (f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
            f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>{fields}'
            f'<DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE>'
            f'</RESOURCE></VOTABLE>').encode()


def test_gspspec_parser_preserves_identity_and_rejects_foreign_rows():
    fields = "".join(f'<FIELD name="{name}" datatype="char" arraysize="*"/>' for name in GSP_SPEC_COLUMNS)
    cells = "".join(f"<TD>{_row()[name]}</TD>" for name in GSP_SPEC_COLUMNS)
    payload = (f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
               f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>{fields}'
               f'<DATA><TABLEDATA><TR>{cells}</TR></TABLEDATA></DATA></TABLE>'
               f'</RESOURCE></VOTABLE>').encode()
    assert parse_gspspec(payload, ["123"])[0]["source_id"] == "123"
    with pytest.raises(ValueError, match="unrequested"):
        parse_gspspec(payload, ["456"])
    with pytest.raises(ValueError, match="not a complete VOTable") as raised:
        parse_gspspec(payload.replace(b'value="OK"', b'value="OVERFLOW"'), ["123"])
    assert "OVERFLOW" in str(raised.value)


def test_gspspec_parser_refuses_a_repeated_source_id():
    with pytest.raises(ValueError, match="duplicate"):
        parse_gspspec(_payload(_row(), _row()), ["123"])


def test_gspspec_query_retains_the_raw_response(tmp_path):
    fields = "".join(f'<FIELD name="{name}" datatype="char" arraysize="*"/>' for name in GSP_SPEC_COLUMNS)
    cells = "".join(f"<TD>{_row()[name]}</TD>" for name in GSP_SPEC_COLUMNS)
    payload = (f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
               f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>{fields}'
               f'<DATA><TABLEDATA><TR>{cells}</TR></TABLEDATA></DATA></TABLE>'
               f'</RESOURCE></VOTABLE>').encode()

    def retrieve(url, accept):
        assert accept == "application/xml"
        assert "123" in parse_qs(urlsplit(url).query)["QUERY"][0]
        return payload

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=retrieve,
                              ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap")
    record, rows, failure = fetch_gspspec(tmp_path, service, ["123"])
    assert failure is None and rows[0]["source_id"] == "123"
    artifact = record["responses"][0]
    assert (tmp_path / artifact["path"]).read_bytes() == payload
    assert artifact["uri"].startswith("https://") and artifact["retrieved_utc"].endswith("Z")


def _row(**overrides):
    row = {
        "source_id": "123", "teff_gspspec": 5800.0, "teff_gspspec_lower": 5700.0,
        "teff_gspspec_upper": 5900.0, "logg_gspspec": 4.0, "logg_gspspec_lower": 3.9,
        "logg_gspspec_upper": 4.1, "mh_gspspec": 0.1, "mh_gspspec_lower": 0.05,
        "mh_gspspec_upper": 0.15, "flags_gspspec": "00100000000000199000000999900029999999999",
    }
    row.update(overrides)
    return row


def test_gspspec_calibration_uses_the_published_polynomial_and_half_width():
    # p(4.0) = 0.274 - 0.1373*4 - 0.0050*16 + 0.0048*64 = -0.048; [M/H] 0.1 -> 0.052.
    assert calibrate_gspspec_mh(_row()) == pytest.approx((0.052, 0.05))
    assert calibrate_gspspec_mh(_row(logg_gspspec=2.0)) == pytest.approx((0.1 + 0.274 - 0.2746 - 0.02 + 0.0384, 0.05))


@pytest.mark.parametrize("change", [
    {"teff_gspspec": 3400.0}, {"logg_gspspec": 5.1}, {"teff_gspspec_upper": 7000.0},
    {"logg_gspspec_lower": 2.0}, {"mh_gspspec_lower": -0.5},
    {"teff_gspspec_lower": 6000.0}, {"logg_gspspec_upper": 3.8},
    {"flags_gspspec": "00200000000000199000000999900029999999999"},
    {"flags_gspspec": "0010000000000019900000099990002999999999X"},
    {"mh_gspspec_upper": 0.05},
])
def test_gspspec_rejects_rows_outside_the_published_selection(change):
    assert calibrate_gspspec_mh(_row(**change)) is None


def test_the_km_giant_quality_flag_is_checked_by_its_named_position():
    base = _row()["flags_gspspec"]
    assert base[12] in "01"
    assert calibrate_gspspec_mh(_row(flags_gspspec=base[:12] + "3" + base[13:])) is None
