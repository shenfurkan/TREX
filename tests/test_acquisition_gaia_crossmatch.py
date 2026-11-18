"""Counterparts must be unique inside the documented angular limit."""

from types import SimpleNamespace

import pytest

from trex.acquisition.gaia_crossmatch import DR2_NEIGHBOURHOOD_COLUMNS, dr2_counterparts, query_gaia_neighborhood


def _votable(dr3_id=123, dr2_id=456, distance=50.0):
    fields = "".join(
        '<FIELD name="{}" datatype="{}"/>'.format(
            name, "long" if name.endswith("_source_id") else
            "boolean" if name == "proper_motion_propagation" else "double")
        for name in DR2_NEIGHBOURHOOD_COLUMNS)
    cells = "<TR><TD>{}</TD><TD>{}</TD><TD>{}</TD><TD></TD><TD>true</TD></TR>".format(
        dr3_id, dr2_id, distance)
    return ('<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
            '<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>'
            f'{fields}<DATA><TABLEDATA>{cells}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>').encode()


def _service(bodies):
    calls = []

    def retrieve(url, accept):
        calls.append((url, accept))
        return bodies.pop(0)

    return SimpleNamespace(ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap",
                           transport_failures=[], _http_get_bytes=retrieve), calls


def test_dr2_counterpart_requires_one_close_valid_row():
    row = {"dr3_source_id": "123", "dr2_source_id": "456", "angular_distance": 99.0,
           "magnitude_difference": None, "proper_motion_propagation": True}
    assert dr2_counterparts([row], ["123"])["123"]["dr2_source_id"] == "456"
    assert dr2_counterparts([row], [123])["123"]["dr2_source_id"] == "456"
    assert dr2_counterparts([{**row, "angular_distance": 100.0}], ["123"])["123"]["status"] == "unavailable"
    assert dr2_counterparts([row, {**row, "dr2_source_id": "789"}], ["123"])["123"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="outside"):
        dr2_counterparts([row], ["124"])


def test_neighbourhood_query_rejects_an_unknown_match_field(tmp_path):
    service, calls = _service([])
    with pytest.raises(ValueError, match="dr3_source_id or dr2_source_id"):
        query_gaia_neighborhood(tmp_path, service, "source_id", [123], "rid")
    assert calls == []


def test_neighbourhood_query_accepts_a_string_output_dir_and_retains_the_body(tmp_path):
    service, calls = _service([_votable()])
    record, rows, failure = query_gaia_neighborhood(str(tmp_path), service, "dr3_source_id", [123], "rid")
    assert failure is None and rows[0]["dr2_source_id"] == 456
    assert record["status"] == "available" and record["source_ids"] == ["123"]
    name = "gaia_dr2_neighbourhood_rid_dr3_source_id_part00_mirror0.xml"
    assert (tmp_path / name).read_bytes() == _votable()
    assert calls[0][0].startswith("https://esa.example/tap?") and calls[0][1] == "application/xml"


def test_a_malformed_first_body_tries_the_mirror(tmp_path):
    service, calls = _service([b"<html>truncated</html>", _votable(dr2_id=789)])
    record, rows, failure = query_gaia_neighborhood(tmp_path, service, "dr3_source_id", [123], "rid")
    assert failure is None and rows[0]["dr2_source_id"] == 789
    assert len(calls) == 2 and len(record["responses"]) == 2
    assert record["responses"][0]["path"].endswith("_mirror0.xml")
    assert record["responses"][1]["path"].endswith("_mirror1.xml")
    assert [item["status"] for item in record["responses"]] == ["unavailable", "available"]
    assert (tmp_path / record["responses"][0]["path"]).read_bytes() == b"<html>truncated</html>"
    assert record["responses"][0]["reason"]


def test_two_malformed_bodies_are_a_retryable_provider_failure(tmp_path):
    service, calls = _service([b"<html>truncated</html>", b"<html>also truncated</html>"])
    record, rows, failure = query_gaia_neighborhood(tmp_path, service, "dr3_source_id", [123], "rid")
    assert rows is None and record["status"] == "unavailable"
    assert len(calls) == 2 and len(record["responses"]) == 2
    assert all(item["status"] == "unavailable" for item in record["responses"])
    assert failure.category == "provider" and failure.retryable
    assert "incomplete" in failure.reason.lower() or "unexpected" in failure.reason.lower()
