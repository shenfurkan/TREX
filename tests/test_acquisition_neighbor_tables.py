"""Synthetic complete and failed archive-table transport contracts."""

from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from trex.acquisition import transport
from trex.acquisition.neighbor_tables import fetch_neighbor_table


def test_neighbor_table_retains_complete_votable(tmp_path):
    body = b'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
    <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>
    <FIELD name="source_id" datatype="long"/><FIELD name="value" datatype="double"/>
    <DATA><TABLEDATA><TR><TD>123</TD><TD>0.25</TD></TR></TABLEDATA></DATA>
    </TABLE></RESOURCE></VOTABLE>'''
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=lambda _url, _accept: body)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"), source_ids=["123"],
        response_name="response.xml", backends=(("esa", "https://example.org/tap"),))
    assert failure is None and rows == [{"source_id": 123, "value": 0.25}]
    response = record["source_queries"]["123"]
    assert (tmp_path / response["response"]).read_bytes() == body
    assert response["uri"].startswith("https://") and response["retrieved_utc"].endswith("Z")


def test_neighbor_table_access_denial_never_falls_through(tmp_path, monkeypatch):
    calls = []

    def denied(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(transport, "urlopen", denied)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id",), source_ids=["123"],
        response_name="response.xml", backends=(("esa", "https://esa.example/tap"),
                                                  ("aip", "https://aip.example/tap")))
    assert len(calls) == 1 and record["status"] == "unavailable" and rows is None
    assert failure.http_status == 403 and not failure.retryable


def test_standalone_host_tables_preserve_missing_rows_and_raw_queries(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from trex.acquisition import neighbor_tables

    def fetch(_output, _service, *, table, **kwargs):
        assert kwargs["source_ids"] == ["123", "456"]
        rows = [{"source_id": "123", "mass_flame": 1.0}] if table == neighbor_tables.GAIA_DR3_AP_TABLE else []
        return {"status": "available", "response": {"path": table + ".xml"}}, rows, None

    monkeypatch.setattr(neighbor_tables, "fetch_neighbor_table", fetch)
    service = SimpleNamespace(ESA_GAIA_TAP_URL="https://esa.example", AIP_GAIA_TAP_URL="https://aip.example")
    result = neighbor_tables.acquire_host_tables(tmp_path, service, ["123", "456"])
    assert result["sources"]["123"]["gaia"]["mass_flame_solar"] == 1.0
    assert result["sources"]["456"] == {"gaia": None, "starhorse": None, "shboost": None}
    assert json.loads((tmp_path / result["record_path"]).read_text()) == result


def test_standalone_host_tables_stop_on_access_denial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from trex.acquisition import neighbor_tables
    from trex.acquisition.transport import AcquisitionError, AcquisitionFailure

    calls = []
    failure = AcquisitionFailure("esa.example", "http_status", False, 1, "HTTP 403", 403, "HTTPError")

    def denied(*_args, **kwargs):
        calls.append(kwargs["table"])
        return {"status": "unavailable"}, None, failure

    monkeypatch.setattr(neighbor_tables, "fetch_neighbor_table", denied)
    service = SimpleNamespace(ESA_GAIA_TAP_URL="https://esa.example", AIP_GAIA_TAP_URL="https://aip.example")
    with pytest.raises(AcquisitionError, match="gaia host table unavailable"):
        neighbor_tables.acquire_host_tables(tmp_path, service, ["123"])
    assert calls == [neighbor_tables.GAIA_DR3_AP_TABLE]


def test_neighbor_table_primary_timeout_uses_existing_mirror_tuple(tmp_path):
    from trex.acquisition.transport import AcquisitionFailure

    body = _votable(("source_id", "value"), {"source_id": 123, "value": 0.25})
    calls = []

    def fetch(url, _accept):
        calls.append(url)
        if url.startswith("https://primary.example/"):
            service.transport_failures.append(AcquisitionFailure("primary.example", "connect_timeout", True, 1, "timeout"))
            return None
        return body

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=fetch)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"), source_ids=["123"],
        response_name="response.xml", backends=(("primary", "https://primary.example/tap"),
                                                ("mirror", "https://mirror.example/tap")))
    assert len(calls) == 2 and rows == [{"source_id": 123, "value": 0.25}] and failure is None
    assert record["backend"] == record["source_queries"]["123"]["backend"] == "mirror"
    assert record["failed_backends"] == ["primary"]
    assert [b["status"] for b in record["backends"]] == ["unavailable", "available"]
    assert (tmp_path / "response.xml").read_bytes() == body


def test_neighbor_table_primary_success_never_queries_mirror(tmp_path):
    body = _votable(("source_id", "value"), {"source_id": 123, "value": 0.25})
    calls = []

    def fetch(url, _accept):
        calls.append(url)
        assert url.startswith("https://primary.example/")
        return body

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=fetch)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"), source_ids=["123"],
        response_name="response.xml", backends=(("primary", "https://primary.example/tap"),
                                                ("mirror", "https://mirror.example/tap")))
    assert len(calls) == 1 and failure is None
    assert rows == [{"source_id": 123, "value": 0.25}]
    assert record["backend"] == "primary" and (tmp_path / "response.xml").read_bytes() == body


def _votable(columns, row):
    from xml.etree.ElementTree import Element, SubElement, tostring

    root = Element("VOTABLE", version="1.3", xmlns="http://www.ivoa.net/xml/VOTable/v1.3")
    resource = SubElement(root, "RESOURCE", type="results")
    SubElement(resource, "INFO", name="QUERY_STATUS", value="OK")
    table = SubElement(resource, "TABLE")
    for column in columns:
        field = SubElement(table, "FIELD", name=column, datatype="long" if column == "source_id" else "double")
        if "flag" in column:
            field.set("datatype", "char")
            field.set("arraysize", "*")
    data = SubElement(SubElement(table, "DATA"), "TABLEDATA")
    tr = SubElement(data, "TR")
    for column in columns:
        SubElement(tr, "TD").text = str(row[column]) if column in row else ""
    return tostring(root)


def _host_service(monkeypatch, unavailable=None):
    from urllib.parse import parse_qs, urlsplit
    from trex.acquisition import neighbor_tables as nt

    rows = {
        nt.GAIA_DR3_AP_TABLE: {"source_id": 123, "mass_flame": 1.25, "teff_gspphot": 6000.0},
        nt.STARHORSE_EDR3_TABLE: {
            "source_id": 123, "mass16": 1.0, "mass50": 1.2, "mass84": 1.4,
            "logg16": 3.9, "logg50": 4.0, "logg84": 4.1, "teff50": 6000.0,
            "fidelity": 0.9, "sh_photoflag": "GBPRPJHKs", "sh_outflag": "0000"},
        nt.SHBOOST_TABLE: {
            "source_id": 123, "xgbdist_mass_mean": 1.2, "xgbdist_mass_std": 0.1,
            "xgbdist_logg_mean": 4.0, "xgbdist_logg_std": 0.1,
            "xgbdist_logteff_mean": 3.75, "xgbdist_logteff_std": 0.02,
            "xgbdist_met_mean": -0.2, "xgbdist_met_std": 0.1, "xgbdist_av_mean": 0.05,
            "xgb_inputflag": "XP5pGBPRPJHKs", "xgb_mass_outputflag": "0", "xgb_logg_outputflag": "0",
            "xgb_logteff_outputflag": "0", "xgb_met_outputflag": "0"},
    }
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1, backoff_seconds=0))
    calls, bodies = [], {}

    def fetch(url, _accept):
        query = parse_qs(urlsplit(url).query)["QUERY"][0]
        columns, rest = query.removeprefix("SELECT ").split(" FROM ")
        table = rest.split(" WHERE ")[0]
        calls.append(table)
        if table == unavailable:
            service.transport_failures.append(transport.AcquisitionFailure(
                "gaia.aip.de", "connect_timeout", True, 1, "synthetic AIP timeout"))
            return None
        body = _votable(columns.replace(" ", "").split(","), rows[table])
        bodies[table] = body
        return body

    monkeypatch.setattr(service, "_http_get_bytes", fetch)
    return service, calls, bodies


@pytest.mark.parametrize("failed_label", ["starhorse", "shboost"])
def test_host_tables_resume_completed_responses_without_refetch(tmp_path, monkeypatch, failed_label):
    import json
    from trex.acquisition import neighbor_tables as nt

    failed_table = nt.STARHORSE_EDR3_TABLE if failed_label == "starhorse" else nt.SHBOOST_TABLE
    service, calls, bodies = _host_service(monkeypatch, failed_table)
    with pytest.raises(transport.AcquisitionError, match=rf"{failed_label} host table unavailable.*gaia.aip.de") as error:
        nt.acquire_host_tables(tmp_path, service, ["123"])
    assert error.value.retryable and error.value.network_failure.endpoint == "gaia.aip.de"
    before_path, = tmp_path.glob("host_tables_*.json")
    before_bytes = before_path.read_bytes()
    before = json.loads(before_bytes)
    completed = {k: v for k, v in before["queries"].items() if v["status"] == "available"}
    retained_bytes = {q["response"]["path"]: (tmp_path / q["response"]["path"]).read_bytes() for q in completed.values()}

    service, calls, _ = _host_service(monkeypatch)
    result = nt.acquire_host_tables(tmp_path, service, ["123"])
    assert calls == ([nt.STARHORSE_EDR3_TABLE, nt.SHBOOST_TABLE] if failed_label == "starhorse" else [nt.SHBOOST_TABLE])
    assert result["status"] == "available" and before_path.read_bytes() == before_bytes
    for label, query in completed.items():
        assert result["queries"][label] == query
        assert result["sources"]["123"][label] == before["sources"]["123"][label]
    assert all((tmp_path / name).read_bytes() == body for name, body in retained_bytes.items())
    assert result["sources"]["123"]["gaia"]["mass_flame_solar"] == 1.25
    assert result["sources"]["123"]["starhorse"]["mass_solar"] == 1.2
    assert result["sources"]["123"]["shboost"]["distribution"]["marginals"]["mass_mean"] == 1.2


@pytest.mark.parametrize("damage", ["differing-mirror-value", "release", "identity", "symlink", "missing-response"])
def test_host_resume_refuses_changed_or_misbound_retained_evidence(tmp_path, monkeypatch, damage):
    import json
    from trex.acquisition import neighbor_tables as nt

    service, _, _ = _host_service(monkeypatch, nt.SHBOOST_TABLE)
    with pytest.raises(transport.AcquisitionError):
        nt.acquire_host_tables(tmp_path, service, ["123"])
    record_path, = tmp_path.glob("host_tables_*.json")
    record = json.loads(record_path.read_text())
    query = record["queries"]["starhorse"]
    path = tmp_path / query["response"]["path"]
    if damage == "differing-mirror-value":
        # A rounded mirror row must not replace the retained native row, even for the same ID.
        path.write_bytes(path.read_bytes().replace(b"<TD>1.2</TD>", b"<TD>1.200001</TD>"))
    elif damage == "release":
        query["table"] = "gaiaedr3_contrib.another_release"
    elif damage == "identity":
        query["source_queries"]["456"] = query["source_queries"].pop("123")
    elif damage == "symlink":
        moved = path.with_suffix(".moved")
        path.rename(moved)
        path.symlink_to(moved)
    else:
        query["responses"] = []
    record_path.write_text(json.dumps(record))
    service, calls, _ = _host_service(monkeypatch)
    with pytest.raises(ValueError, match="retained host"):
        nt.acquire_host_tables(tmp_path, service, ["123"])
    assert calls == []


def test_unverified_starhorse_mirror_is_not_used_on_outage(tmp_path, monkeypatch):
    from trex.acquisition import neighbor_tables as nt

    service, calls, _ = _host_service(monkeypatch, nt.STARHORSE_EDR3_TABLE)
    with pytest.raises(transport.AcquisitionError, match="gaia.aip.de"):
        nt.acquire_host_tables(tmp_path, service, ["123"])
    # VizieR's rounded values / changed colour release are not an admitted fallback.
    assert calls == [nt.GAIA_DR3_AP_TABLE, nt.STARHORSE_EDR3_TABLE]


def test_host_resume_reparses_raw_values_and_keeps_missing_rows_missing(tmp_path, monkeypatch):
    import json
    from trex.acquisition import neighbor_tables as nt

    service, _, _ = _host_service(monkeypatch, nt.STARHORSE_EDR3_TABLE)
    with pytest.raises(transport.AcquisitionError):
        nt.acquire_host_tables(tmp_path, service, ["123", "456"])
    path, = tmp_path.glob("host_tables_*.json")
    before = json.loads(path.read_text())
    before["sources"]["123"]["gaia"]["mass_flame_solar"] = 999.0
    path.write_text(json.dumps(before))
    service, calls, _ = _host_service(monkeypatch, nt.STARHORSE_EDR3_TABLE)
    with pytest.raises(transport.AcquisitionError):
        nt.acquire_host_tables(tmp_path, service, ["123", "456"])
    assert calls == [nt.STARHORSE_EDR3_TABLE]
    after_path, = [p for p in tmp_path.glob("host_tables_*.json") if p != path]
    after = json.loads(after_path.read_text())
    assert after["queries"]["gaia"] == before["queries"]["gaia"]
    assert after["sources"]["123"]["gaia"]["mass_flame_solar"] == 1.25
    assert after["sources"]["456"]["gaia"] is None


def _votable_rows(columns, rows):
    from xml.etree.ElementTree import Element, SubElement, tostring

    root = Element("VOTABLE", version="1.3", xmlns="http://www.ivoa.net/xml/VOTable/v1.3")
    resource = SubElement(root, "RESOURCE", type="results")
    SubElement(resource, "INFO", name="QUERY_STATUS", value="OK")
    table = SubElement(resource, "TABLE")
    for column in columns:
        SubElement(table, "FIELD", name=column, datatype="long" if column == "source_id" else "double")
    data = SubElement(SubElement(table, "DATA"), "TABLEDATA")
    for row in rows:
        tr = SubElement(data, "TR")
        for column in columns:
            SubElement(tr, "TD").text = str(row[column])
    return tostring(root)


def test_shboost_batch_preserves_per_source_binding_and_empty_matches(tmp_path):
    from urllib.parse import parse_qs, urlsplit

    from trex.acquisition.neighbor_tables import SHBOOST_TABLE

    columns = ("source_id", "value")
    body = _votable_rows(columns, [{"source_id": 123, "value": 0.25}, {"source_id": 125, "value": 0.75}])
    queries = []

    def get(url, _accept):
        queries.append(parse_qs(urlsplit(url).query)["QUERY"][0])
        return body

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=get)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table=SHBOOST_TABLE, columns=columns, source_ids=["123", "124", "125"],
        response_name="shboost.xml", backends=(("gaia-aip", "https://example.org/tap"),),
    )
    assert failure is None and rows == [{"source_id": 123, "value": 0.25}, {"source_id": 125, "value": 0.75}]
    assert len(queries) == 1 and queries[0].endswith("WHERE source_id IN (123, 124, 125)")
    assert set(record["source_queries"]) == {"123", "124", "125"}
    assert {binding["response"] for binding in record["source_queries"].values()} == {"shboost.xml"}
    assert (tmp_path / "shboost.xml").read_bytes() == body


def test_shboost_timed_out_batch_recovers_by_exact_source_without_losing_empty_rows(tmp_path):
    from urllib.parse import parse_qs, urlsplit

    from trex.acquisition.neighbor_tables import SHBOOST_TABLE

    columns = ("source_id", "value")
    row = {"source_id": 123, "value": 0.25}
    bodies = {"123": _votable_rows(columns, [row]), "124": _votable_rows(columns, [])}
    service = SimpleNamespace(transport_failures=[])

    def get(url, _accept):
        scope = parse_qs(urlsplit(url).query)["QUERY"][0].split(" WHERE ")[-1]
        if scope == "source_id IN (123, 124)":
            service.transport_failures.append(
                transport.AcquisitionFailure("example.org", "read_timeout", True, 1, "batch timed out")
            )
            return None
        return bodies[scope.removeprefix("source_id=")]

    service._http_get_bytes = get
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table=SHBOOST_TABLE, columns=columns, source_ids=["123", "124"],
        response_name="shboost.xml", backends=(("gaia-aip", "https://example.org/tap"),),
    )
    assert failure is None and rows == [row]
    assert record["backends"][0]["recovery"] == "retry-exact-source"
    assert {
        identifier: (tmp_path / binding["response"]).read_bytes()
        for identifier, binding in record["source_queries"].items()
    } == bodies


def _retained_query(output, columns, entries):
    """Build a retained query whose responses are written under ``output``."""
    import hashlib

    responses, source_queries = [], {}
    for index, (source_id, rows) in enumerate(entries.items()):
        body = _votable_rows(columns, rows)
        name = "response_part{0:02d}.xml".format(index)
        (output / name).write_bytes(body)
        responses.append({"path": name, "sha256": hashlib.sha256(body).hexdigest()})
        source_queries[source_id] = {
            "uri": "https://archive.example/query/{0}".format(source_id),
            "backend": "retained-backend", "chunk": index,
            "retrieved_utc": "2026-11-10T00:00:00Z", "response": name,
        }
    return {
        "table": "gaia.table", "columns": list(columns), "query": "SELECT retained",
        "uri": "https://archive.example/query", "backend": "retained-backend",
        "status": "unavailable", "reason": "no backend returned a response for chunk 2 (aip)",
        "backends": [{"backend": "aip", "chunk": 2, "status": "unavailable"}],
        "failed_backends": ["aip"], "responses": responses, "source_queries": source_queries,
    }


def _retained_fixture(tmp_path, entries):
    return _retained_query(tmp_path, ("source_id", "value"), entries)


def _no_network(_url, _accept):
    raise AssertionError("a complete retained response must not be requested again")


def test_neighbor_table_resume_reuses_complete_responses_without_refetch(tmp_path):
    import json

    retained = _retained_fixture(tmp_path, {"123": [{"source_id": 123, "value": 0.25}],
                                            "456": [{"source_id": 456, "value": 0.5}]})
    before = json.dumps(retained, sort_keys=True)
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=_no_network)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"),
        source_ids=["123", "456"], response_name="response.xml",
        backends=(("esa", "https://example.org/tap"),), retained_query=retained)
    assert failure is None
    assert rows == [{"source_id": 123, "value": 0.25}, {"source_id": 456, "value": 0.5}]
    assert record["status"] == "available" and record["row_count"] == 2
    assert record["responses"] == retained["responses"]
    assert record["source_queries"] == retained["source_queries"]
    assert record["retained_from"]["status"] == "unavailable"
    assert record["retained_from"]["reason"] == retained["reason"]
    assert record["retained_from"]["backends"] == retained["backends"]
    assert record["retained_from"]["reused_sources"] == 2
    assert json.dumps(retained, sort_keys=True) == before


def test_neighbor_table_resume_fetches_only_the_missing_source_chunk(tmp_path):
    retained = _retained_fixture(tmp_path, {"123": [{"source_id": 123, "value": 0.25}]})
    calls = []

    def fetch(url, _accept):
        calls.append(url)
        return _votable_rows(("source_id", "value"), [{"source_id": 456, "value": 0.5}])

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=fetch)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"),
        source_ids=["123", "456"], response_name="response.xml",
        backends=(("esa", "https://example.org/tap"),), retained_query=retained)
    assert len(calls) == 1 and failure is None
    from urllib.parse import parse_qs, urlsplit

    assert "IN (456)" in parse_qs(urlsplit(calls[0]).query)["QUERY"][0]
    assert rows == [{"source_id": 123, "value": 0.25}, {"source_id": 456, "value": 0.5}]
    assert [response["path"] for response in record["responses"]] == ["response_part00.xml", "response_part01.xml"]
    assert set(record["source_queries"]) == {"123", "456"}
    assert record["source_queries"]["123"] == retained["source_queries"]["123"]
    assert record["source_queries"]["456"]["chunk"] == 1
    assert record["retained_from"]["reused_sources"] == 1
    assert record["status"] == "available"


@pytest.mark.parametrize("damage", ["release", "columns", "identity", "symlink", "missing-response", "sha256"])
def test_neighbor_table_resume_refuses_changed_or_misbound_retained_evidence(tmp_path, damage):
    import hashlib

    retained = _retained_fixture(tmp_path, {"123": [{"source_id": 123, "value": 0.25}]})
    response = retained["responses"][0]
    path = tmp_path / response["path"]
    if damage == "release":
        retained["table"] = "gaia.another_release"
    elif damage == "columns":
        retained["columns"] = ["source_id"]
    elif damage == "identity":
        body = _votable_rows(("source_id", "value"), [{"source_id": 999, "value": 0.25}])
        path.write_bytes(body)
        response["sha256"] = hashlib.sha256(body).hexdigest()
    elif damage == "symlink":
        moved = path.with_suffix(".moved")
        path.rename(moved)
        path.symlink_to(moved)
    elif damage == "missing-response":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=_no_network)
    with pytest.raises(ValueError, match="retained neighbour"):
        fetch_neighbor_table(
            tmp_path, service, table="gaia.table", columns=("source_id", "value"),
            source_ids=["123"], response_name="response.xml",
            backends=(("esa", "https://example.org/tap"),), retained_query=retained)


def test_neighbor_table_failed_resume_keeps_retained_responses_and_exact_failure(tmp_path):
    from trex.acquisition.transport import AcquisitionFailure

    retained = _retained_fixture(tmp_path, {"123": [{"source_id": 123, "value": 0.25}]})
    retained_bytes = (tmp_path / retained["responses"][0]["path"]).read_bytes()
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=None)

    def fetch(_url, _accept):
        service.transport_failures.append(
            AcquisitionFailure("gaia.aip.de", "read_timeout", True, 1, "the provider did not answer in time"))
        return None

    service._http_get_bytes = fetch
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"),
        source_ids=["123", "456"], response_name="response.xml",
        backends=(("aip", "https://gaia.aip.de/tap"),), retained_query=retained)
    assert rows is None and failure is not None and failure.retryable
    assert record["status"] == "unavailable"
    assert record["reason"] == "no backend returned a response for chunk 1 (aip)"
    assert record["responses"] == retained["responses"]
    assert record["source_queries"] == retained["source_queries"]
    assert record["retained_from"]["reason"] == retained["reason"]
    assert (tmp_path / retained["responses"][0]["path"]).read_bytes() == retained_bytes


def test_shboost_read_timeout_falls_back_to_the_async_job(tmp_path):
    from trex.acquisition.neighbor_tables import SHBOOST_TABLE
    from trex.acquisition.transport import AcquisitionFailure

    body = _votable_rows(("source_id", "value"), [{"source_id": 123, "value": 0.25}])
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=None)

    def timed_out(_url, _accept):
        service.transport_failures.append(
            AcquisitionFailure("gaia.aip.de", "read_timeout", True, 1, "the provider did not answer in time"))
        return None

    service._http_get_bytes = timed_out
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table=SHBOOST_TABLE, columns=("source_id", "value"), source_ids=["123"],
        response_name="shboost.xml", backends=(("gaia-aip", "https://gaia.aip.de/tap/sync"),),
        async_fetch=lambda _service, _query: (body, "https://gaia.aip.de/uws/results/1"))
    assert failure is None and rows == [{"source_id": 123, "value": 0.25}]
    assert record["status"] == "available" and record["backend"] == "gaia-aip-async"
    assert [item["backend"] for item in record["backends"]] == ["gaia-aip", "gaia-aip-async"]
    assert record["source_queries"]["123"]["backend"] == "gaia-aip-async"


def test_neighbor_table_resume_reuses_a_valid_empty_response(tmp_path):
    retained = _retained_fixture(tmp_path, {"123": []})
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=_no_network)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"),
        source_ids=["123"], response_name="response.xml",
        backends=(("esa", "https://example.org/tap"),), retained_query=retained)
    assert failure is None and rows == []
    assert record["status"] == "available" and record["row_count"] == 0
    assert record["retained_from"]["reused_sources"] == 1


def test_a_multi_source_response_is_reused_for_a_requested_subset(tmp_path):
    import hashlib

    columns = ("source_id", "value")
    body = _votable_rows(columns, [{"source_id": 123, "value": 0.25},
                                   {"source_id": 456, "value": 0.5}])
    name = "response_part00.xml"
    (tmp_path / name).write_bytes(body)
    source_queries = {str(source_id): {"uri": "https://archive.example/query", "backend": "retained",
                                       "chunk": 0, "retrieved_utc": "2026-11-10T00:00:00Z", "response": name}
                      for source_id in (123, 456)}
    retained = {"table": "gaia.table", "columns": list(columns), "query": "SELECT retained",
                "uri": "https://archive.example/query", "backend": "retained", "status": "available",
                "responses": [{"path": name, "sha256": hashlib.sha256(body).hexdigest()}],
                "source_queries": source_queries}
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=_no_network)
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=columns, source_ids=["123"],
        response_name="response.xml", backends=(("esa", "https://example.org/tap"),), retained_query=retained)
    assert failure is None and rows == [{"source_id": 123, "value": 0.25}]
    assert set(record["source_queries"]) == {"123"} and record["responses"] == retained["responses"]


def test_a_malformed_new_body_is_retained_but_not_bound(tmp_path):
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=lambda *_args: b"<html>bad</html>")
    record, rows, failure = fetch_neighbor_table(
        tmp_path, service, table="gaia.table", columns=("source_id", "value"), source_ids=["123"],
        response_name="response.xml", backends=(("esa", "https://example.org/tap"),))
    assert rows is None and record["status"] == "unavailable" and failure is None
    rejected = record["rejected_responses"][0]
    assert rejected["source_ids"] == ["123"] and (tmp_path / rejected["path"]).is_file()
    assert record["responses"] == [] and record["source_queries"] == {}


@pytest.mark.parametrize("stored_path", [None, "../host_tables_foreign.json"])
def test_an_incomplete_host_table_resume_preserves_the_failed_record(tmp_path, monkeypatch, stored_path):
    import json
    from trex.acquisition import neighbor_tables as nt

    service, calls, _ = _host_service(monkeypatch, nt.STARHORSE_EDR3_TABLE)
    with pytest.raises(transport.AcquisitionError):
        nt.acquire_host_tables(tmp_path, service, ["123"])
    record_path, = tmp_path.glob("host_tables_*.json")
    if stored_path is not None:
        retained = json.loads(record_path.read_text())
        record_path.write_text(json.dumps({**retained, "record_path": stored_path}))
    failed_bytes = record_path.read_bytes()
    service, calls, _ = _host_service(monkeypatch)
    result = nt.acquire_host_tables(tmp_path, service, ["123"])
    assert result["record_path"] != record_path.name and result["status"] == "available"
    assert result["resumed_from"] == record_path.name
    assert calls == [nt.STARHORSE_EDR3_TABLE, nt.SHBOOST_TABLE]
    assert record_path.read_bytes() == failed_bytes
    assert {path.name for path in tmp_path.glob("host_tables_*.json")} == {record_path.name, result["record_path"]}
    assert json.loads((tmp_path / result["record_path"]).read_text())["status"] == "available"


def test_neighbor_table_refuses_duplicate_ids_an_empty_backend_list_or_a_non_xml_name(tmp_path):
    service = SimpleNamespace(transport_failures=[], _http_get_bytes=_no_network)
    common = dict(table="gaia.table", columns=("source_id",), response_name="response.xml",
                  backends=(("esa", "https://example.org/tap"),))
    with pytest.raises(ValueError, match="unique"):
        fetch_neighbor_table(tmp_path, service, source_ids=["123", "000123"], **common)
    with pytest.raises(ValueError, match="at least one backend"):
        fetch_neighbor_table(tmp_path, service, source_ids=["123"], **{**common, "backends": ()})
    with pytest.raises(ValueError, match=r"plain \.xml"):
        fetch_neighbor_table(tmp_path, service, source_ids=["123"], **{**common, "response_name": "response.txt"})


def test_a_corrupt_host_record_is_skipped_not_reused(tmp_path, monkeypatch):
    from trex.acquisition import neighbor_tables as nt

    (tmp_path / "host_tables_corrupt.json").write_text("{not json", encoding="utf-8")
    service, _, _ = _host_service(monkeypatch)
    result = nt.acquire_host_tables(tmp_path, service, ["123"])
    assert result["status"] == "available" and (tmp_path / "host_tables_corrupt.json").is_file()


def test_async_shboost_uses_the_policy_user_agent_and_timeout(monkeypatch):
    import functools
    import requests
    import pyvo
    from trex.acquisition import neighbor_tables as nt

    class Session:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def request(self, *_args, **_kwargs):
            raise AssertionError("the test TAP service does not issue HTTP requests")

    sessions = []

    def session_factory():
        session = Session()
        sessions.append(session)
        return session

    class Job:
        phase = "COMPLETED"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self):
            return None

        def _update(self, **_kwargs):
            raise AssertionError("a completed job must not be polled")

        def raise_if_error(self):
            return None

        result_uri = "https://aip.example/results/1.xml"

    class TAP:
        base = None

        def __init__(self, base, session=None):
            TAP.base = base
            assert session.headers["User-Agent"] == "client-x/1"

        def submit_job(self, query):
            assert query.startswith("SELECT ")
            return Job()

    monkeypatch.setattr(requests, "Session", session_factory)
    monkeypatch.setattr(pyvo.dal, "TAPService", TAP)
    service = SimpleNamespace(AIP_GAIA_TAP_URL="https://aip.example/tap/sync", policy=transport.NetworkPolicy(
        True, timeout_seconds=12.0, user_agent=lambda: "client-x/1"), _http_get_bytes=lambda url, accept: b"<xml/>")
    body, url = nt.fetch_shboost_async(service, "SELECT source_id FROM table")
    assert body == b"<xml/>" and url.endswith("1.xml") and TAP.base == "https://aip.example/tap"
    assert isinstance(sessions[0].request, functools.partial)
    assert sessions[0].request.keywords["timeout"] == 12.0
