"""The standalone PLATO PIC provider retains complete cone evidence."""

import hashlib
import json
from urllib.parse import parse_qs

import pytest

from trex.acquisition import plato_pic
from trex.acquisition.transport import AcquisitionError, AcquisitionFailure, NetworkPolicy

TARGET_FIELDS = (("PICname", "char"), ("RAdeg", "double"), ("DEdeg", "double"))
CONTAMINANT_FIELDS = (
    ("PICname", "char"), ("RAdeg", "double"), ("DEdeg", "double"),
    ("PlatoMagNCAM", "double"), ("Gmag", "double"), ("GaiaDR3", "long"),
)
TARGET = {"PICname": "PIC-0000", "RAdeg": 10.0, "DEdeg": -20.0}


def _field(name, datatype):
    arraysize = ' arraysize="*"' if datatype == "char" else ""
    return f'<FIELD name="{name}" datatype="{datatype}"{arraysize}/>'


def votable(fields, rows, *, query_status="OK", document_info=""):
    """A minimal synthetic single-table VOTable; no catalogue data is implied."""
    field_xml = "".join(_field(name, datatype) for name, datatype in fields)
    body = "".join("<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">{document_info}'
            f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="{query_status}"/>'
            f'<TABLE>{field_xml}<DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE>'
            f'</RESOURCE></VOTABLE>').encode("utf-8")


def target_page(names):
    return votable(TARGET_FIELDS, [(name, 10.0, -20.0) for name in names])


def contaminant_page(count):
    return votable(CONTAMINANT_FIELDS, [
        (f"PIC-{index:04d}", 10.0 + index * 0.001, -20.0, 12.0 + index * 0.1, 13.0,
         1234567890123456789 + index) for index in range(count)])


class FakeTransport:
    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.calls = []
        self.transport_failures = []

    def post_form(self, url, data, accept="application/json"):
        self.calls.append((url, data, accept))
        body = self.bodies.pop(0)
        if isinstance(body, BaseException):
            raise body
        return body


def posted_queries(transport):
    return [parse_qs(data.decode("ascii"))["QUERY"][0] for _url, data, _accept in transport.calls]


def test_parse_pic_votable_returns_the_header_and_typed_rows():
    columns, rows = plato_pic.parse_pic_votable(target_page(["PIC-0001"]))
    assert columns == ("PICname", "RAdeg", "DEdeg")
    assert rows == [{"PICname": "PIC-0001", "RAdeg": 10.0, "DEdeg": -20.0}]


def test_target_identity_has_no_invented_epoch_and_keeps_exact_gaia():
    identity = plato_pic.pic_target_identity({
        **TARGET, "GaiaDR3": "1234567890123456789"})
    assert identity == {
        "namespace": "plato-pic", "native_identifier": "PIC-0000", "frame": "ICRS",
        "epoch": None, "ra_deg": 10.0, "dec_deg": -20.0,
        "gaia_dr3_id": "1234567890123456789", "gaia_dr3_column": "GaiaDR3",
    }


def test_target_identity_keeps_only_an_epoch_the_row_declares():
    assert plato_pic.pic_target_identity({**TARGET, "epoch": 2016.0})["epoch"] == 2016.0
    assert plato_pic.pic_target_identity(TARGET)["epoch"] is None


def test_a_malformed_present_gaia_value_is_recorded_not_silently_none():
    identity = plato_pic.pic_target_identity({**TARGET, "GaiaDR3": "12.5"})
    assert identity["gaia_dr3_id"] is None
    assert identity["gaia_dr3_malformed"]["raw_value"] == "12.5"
    assert "reason" in identity["gaia_dr3_malformed"]


def test_pic_identity_is_a_canonical_catalog_identity():
    from trex.acquisition.identity import CatalogIdentity

    identity = plato_pic.pic_catalog_identity({**TARGET, "GaiaDR3": "1234567890123456789"})
    assert identity.namespace == "plato-pic"
    assert identity.native_id == "PIC-0000"
    assert identity.epoch_jyear is None
    assert identity.cross_identifiers == (("gaia-dr3", "1234567890123456789"),)
    assert CatalogIdentity.from_record(identity.to_record()) == identity


def test_keyset_pagination_retains_every_page(tmp_path):
    transport = FakeTransport(target_page(["PIC-0001", "PIC-0002"]), target_page(["PIC-0003"]))
    record, rows = plato_pic.acquire_pic_targets(
        10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True), page_size=2, transport=transport)

    assert [row["PICname"] for row in rows] == ["PIC-0001", "PIC-0002", "PIC-0003"]
    assert record["status"] == "available"
    assert (record["page_count"], record["row_count"], record["page_size"]) == (2, 3, 2)
    assert all(url == plato_pic.PLATO_TAP_URL for url, _data, _accept in transport.calls)
    queries = posted_queries(transport)
    assert queries[0].endswith("ORDER BY PICname") and "PICname >" not in queries[0]
    assert "PICname > 'PIC-0002'" in queries[1]
    assert [item["path"] for item in record["raw_files"]] == [
        "pic_targets_raw_00.xml", "pic_targets_raw_01.xml"]
    assert [item["rows"] for item in record["raw_files"]] == [2, 1]
    for item in record["raw_files"]:
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    written = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert written["status"] == "available" and written["row_count"] == 3
    assert record["tap_url"] == plato_pic.PLATO_TAP_URL and record["table"] == plato_pic.PIC_TARGET_TABLE
    assert record["search_radius_deg"] == 0.2 and record["retrieved_utc"].endswith("Z")
    assert "aperture footprint" in record["limitation"]


def test_target_query_escapes_the_keyset_literal():
    query = plato_pic.build_pic_target_query(10.0, -20.0, 0.2, after_pic_name="PIC'O", page_size=5)
    assert "SELECT TOP 5" in query and "PICname > 'PIC''O'" in query
    assert "CIRCLE('ICRS', 10.0, -20.0, 0.2)" in query


def test_empty_target_table_is_empty_not_unavailable(tmp_path):
    transport = FakeTransport(votable(TARGET_FIELDS, []))
    record, rows = plato_pic.acquire_pic_targets(
        10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True), transport=transport)
    assert rows == [] and record["status"] == "empty" and record["row_count"] == 0
    assert record["page_count"] == 1 and record["raw_header"] == ["PICname", "RAdeg", "DEdeg"]


@pytest.mark.parametrize("document_info", ["", '<INFO name="QUERY_STATUS" value="OVERFLOW"/>'])
def test_overflow_is_a_distinct_pagination_failure(document_info, tmp_path):
    body = votable(TARGET_FIELDS, [], query_status="OK" if document_info else "OVERFLOW",
                   document_info=document_info)
    transport = FakeTransport(body)
    with pytest.raises(plato_pic.PicPaginationError, match="OVERFLOW"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "unavailable" and record["error_kind"] == "overflow"
    assert (tmp_path / record["raw_files"][0]["path"]).is_file()


def test_an_error_document_is_malformed_not_empty(tmp_path):
    transport = FakeTransport(votable(TARGET_FIELDS, [], query_status="ERROR"))
    with pytest.raises(AcquisitionError, match="malformed"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "malformed" and record["error_kind"] == "response"


def test_malformed_xml_is_malformed_not_unavailable(tmp_path):
    transport = FakeTransport(b"<VOTABLE><RESOURCE>")
    with pytest.raises(AcquisitionError, match="malformed") as raised:
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    assert not isinstance(raised.value, plato_pic.PicPaginationError)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "malformed" and record["error_kind"] == "response"


def test_transport_failure_is_unavailable_and_raises(tmp_path):
    transport = FakeTransport(None)
    transport.transport_failures.append(
        AcquisitionFailure("pax.esac.esa.int", "connect_timeout", True, 3, "timed out"))
    with pytest.raises(AcquisitionError, match="timed out") as raised:
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    assert raised.value.retryable
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "unavailable" and record["failure"]["category"] == "connect_timeout"


def test_a_transport_exception_propagates_after_the_unavailable_record(tmp_path):
    transport = FakeTransport(AcquisitionError("response exceeds the configured size limit"))
    with pytest.raises(AcquisitionError, match="exceeds"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "unavailable" and record["error_kind"] == "transport"
    assert record["error"]["exception_class"] == "AcquisitionError"


def test_a_repeated_page_is_a_pagination_failure(tmp_path):
    transport = FakeTransport(target_page(["PIC-0001", "PIC-0002"]),
                              target_page(["PIC-0001", "PIC-0002"]))
    with pytest.raises(plato_pic.PicPaginationError, match="keyset"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      page_size=2, transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "unavailable" and record["error_kind"] == "repeated_page"


def test_the_page_budget_refuses_unbounded_pagination(monkeypatch, tmp_path):
    monkeypatch.setattr(plato_pic, "MAX_PAGES", 2)
    transport = FakeTransport(target_page(["PIC-0001", "PIC-0002"]),
                              target_page(["PIC-0003", "PIC-0004"]))
    with pytest.raises(plato_pic.PicPaginationError, match="page budget"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      page_size=2, transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "unavailable" and record["error_kind"] == "page_limit"


def test_a_missing_target_column_records_the_raw_header(tmp_path):
    fields = (("PICname", "char"), ("RAdeg", "double"))
    transport = FakeTransport(votable(fields, [("PIC-0001", 10.0)]))
    with pytest.raises(AcquisitionError, match="missing required columns"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(True),
                                      transport=transport)
    record = json.loads((tmp_path / "pic_targets_query.json").read_text())
    assert record["status"] == "malformed"
    assert record["raw_header"] == ["PICname", "RAdeg"]
    assert (tmp_path / record["raw_files"][0]["path"]).is_file()


def test_offline_policy_refuses_network(tmp_path):
    with pytest.raises(AcquisitionError, match="disabled"):
        plato_pic.acquire_pic_targets(10.0, -20.0, 0.2, tmp_path, NetworkPolicy(False))
    with pytest.raises(AcquisitionError, match="disabled"):
        plato_pic.acquire_pic_contaminants(TARGET, tmp_path, NetworkPolicy(False))


def test_all_contaminants_are_retained_without_a_fifteen_row_cap(tmp_path):
    transport = FakeTransport(contaminant_page(20))
    target = plato_pic.pic_target_identity(TARGET)
    record, rows = plato_pic.acquire_pic_contaminants(
        target, tmp_path, NetworkPolicy(True), transport=transport)

    assert record["status"] == "available" and record["row_count"] == 20 and len(rows) == 20
    assert record["host_pic_name"] == "PIC-0000"
    assert all(row["host_pic_name"] == "PIC-0000" for row in rows)
    assert rows[0]["magnitudes"]["PlatoMagNCAM"] == {"value": 12.0, "unit": "mag"}
    assert rows[0]["magnitudes"]["Gmag"] == {"value": 13.0, "unit": "mag"}
    assert rows[0]["gaia_dr3_id"] == "1234567890123456789"
    assert rows[19]["gaia_dr3_id"] == "1234567890123456808"
    assert record["raw_header"] == [name for name, _datatype in CONTAMINANT_FIELDS]
    assert record["association"].startswith("Rows are returned by a cone query")
    query = posted_queries(transport)[0]
    assert "SELECT * FROM pic_go.pic_contaminant_go" in query
    assert "CIRCLE('ICRS', 10.0, -20.0" in query
    assert parse_qs(transport.calls[0][1].decode("ascii"))["MAXREC"] == ["20000"]
    assert (tmp_path / "pic_contaminants_raw.xml").is_file()


def test_empty_contaminant_table_is_a_successful_empty_result(tmp_path):
    transport = FakeTransport(votable(CONTAMINANT_FIELDS, []))
    record, rows = plato_pic.acquire_pic_contaminants(
        TARGET, tmp_path, NetworkPolicy(True), transport=transport)
    assert record["status"] == "empty" and record["row_count"] == 0 and rows == []
    assert record["host_pic_name"] == "PIC-0000"


def test_a_contaminant_without_radeg_is_malformed_and_keeps_the_header(tmp_path):
    fields = (("PICname", "char"), ("DEdeg", "double"), ("Gmag", "double"))
    transport = FakeTransport(votable(fields, [("PIC-0001", -20.0, 13.0)]))
    with pytest.raises(AcquisitionError, match="missing required columns"):
        plato_pic.acquire_pic_contaminants(TARGET, tmp_path, NetworkPolicy(True), transport=transport)
    record = json.loads((tmp_path / "pic_contaminants_query.json").read_text())
    assert record["status"] == "malformed"
    assert record["raw_header"] == ["PICname", "DEdeg", "Gmag"]


def test_a_contaminant_table_without_a_magnitude_is_malformed(tmp_path):
    fields = (("PICname", "char"), ("RAdeg", "double"), ("DEdeg", "double"))
    transport = FakeTransport(votable(fields, [("PIC-0001", 10.0, -20.0)]))
    with pytest.raises(AcquisitionError, match="no native magnitude column"):
        plato_pic.acquire_pic_contaminants(TARGET, tmp_path, NetworkPolicy(True), transport=transport)
    record = json.loads((tmp_path / "pic_contaminants_query.json").read_text())
    assert record["status"] == "malformed"


def test_contaminant_transport_failure_is_unavailable(tmp_path):
    transport = FakeTransport(None)
    transport.transport_failures.append(
        AcquisitionFailure("pax.esac.esa.int", "http_status", False, 1, "HTTP 404: not found", 404))
    with pytest.raises(AcquisitionError, match="HTTP 404"):
        plato_pic.acquire_pic_contaminants(TARGET, tmp_path, NetworkPolicy(True), transport=transport)
    record = json.loads((tmp_path / "pic_contaminants_query.json").read_text())
    assert record["status"] == "unavailable" and record["failure"]["http_status"] == 404


def test_contaminant_rows_record_malformed_and_absent_gaia_values():
    rows = [
        {"RAdeg": 1.0, "DEdeg": 2.0, "Gmag": 15.0},
        {"RAdeg": 1.1, "DEdeg": 2.1, "Gmag": 16.0, "GaiaDR3": "12.5"},
    ]
    normalized = plato_pic.pic_contaminant_rows(rows, host_pic_name="PIC-0001")
    assert normalized[0]["gaia_dr3_id"] is None and "gaia_dr3_malformed" not in normalized[0]
    assert normalized[1]["gaia_dr3_id"] is None
    assert normalized[1]["gaia_dr3_malformed"]["raw_value"] == "12.5"
    assert normalized[1]["raw"]["GaiaDR3"] == "12.5"
