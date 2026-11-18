"""Official-GALAH release and targeted TAP reading with synthetic rows and identifiers."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from trex.acquisition.galah_dr3 import (
    GALAH_DR3_SERVICE_RELEASE, GALAH_DR3_TAP_BUDGET_SECONDS, GALAH_DR3_TAP_TABLE,
    GALAH_DR3_TAP_URL, GALAH_DR3_TARGETED_FIELDS, GALAH_DR3_URL, GalahTargetedRelease,
    fetch_galah_gaia_neighborhood, galah_dr3_associations, galah_dr3_rows, galah_shared_release,
    retain_galah_rows,
)


@pytest.fixture(autouse=True)
def _zeroed_galah_backoff(monkeypatch):
    """Record every GALAH retry wait through the shared seam; never sleep."""
    from trex.acquisition import transport

    waits = []
    monkeypatch.setattr(transport, "sleep_before_retry", waits.append)
    return waits


def test_galah_release_uses_explicit_cache_and_retains_retrieval(tmp_path):
    calls = []

    def download(url, path):
        calls.append((url, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        _release(path)
        return path

    service = SimpleNamespace(download=download, transport_failures=[])
    path, record = galah_shared_release(tmp_path, service, download=True)
    assert calls == [(GALAH_DR3_URL, path)] and record["bytes"] == path.stat().st_size
    assert record["retrieved_utc"].endswith("Z") and record["url"] == GALAH_DR3_URL
    assert galah_shared_release(tmp_path, service) == (path, record) and len(calls) == 1


def test_cold_galah_cache_selects_the_documented_targeted_service_without_download(tmp_path):
    service = SimpleNamespace(download=lambda *_: pytest.fail("implicit bulk download"), transport_failures=[])
    release, identity = galah_shared_release(tmp_path, service)
    assert isinstance(release, GalahTargetedRelease)
    assert identity["url"] == GALAH_DR3_TAP_URL and identity["table"] == GALAH_DR3_TAP_TABLE
    assert identity["release"] == GALAH_DR3_SERVICE_RELEASE and identity["row_space"] == "votable_response"
    assert identity["bytes"] is None and identity["retrieved_utc"].endswith("Z")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("new_failure", [False, True])
def test_explicit_download_failure_preserves_only_its_own_transport_diagnostic(tmp_path, new_failure):
    from trex.acquisition.transport import AcquisitionError, AcquisitionFailure

    stale = AcquisitionFailure("other.example", "http_status", False, 1, "old HTTP 403", 403, "HTTPError")
    failure = AcquisitionFailure("cloud.datacentral.org.au", "connect_timeout", True, 1,
                                 "synthetic timeout", None, "TimeoutError")
    failures = [stale]

    def download(_url, _path):
        if new_failure:
            failures.append(failure)
        return None

    with pytest.raises(AcquisitionError, match="no completed release") as exc:
        galah_shared_release(tmp_path, SimpleNamespace(download=download, transport_failures=failures), download=True)
    assert exc.value.network_failure is (failure if new_failure else None)
    assert exc.value.retryable is new_failure
    assert GALAH_DR3_URL in str(exc.value) and str(tmp_path) in str(exc.value)
    assert "old HTTP 403" not in str(exc.value)
    if new_failure:
        assert "connect_timeout: synthetic timeout" in str(exc.value)
    assert not list(tmp_path.iterdir())


def test_targeted_service_failure_stops_the_real_metallicity_consumer_without_fallback(
        tmp_path, _zeroed_galah_backoff):
    from trex.acquisition.apogee_dr17 import APOGEE_DR17_COLUMNS
    from trex.acquisition.metallicity import acquire_metallicity
    from trex.acquisition.transport import AcquisitionError, AcquisitionFailure

    failure = AcquisitionFailure("datacentral.org.au", "connect_timeout", True, 1,
                                 "synthetic timeout", None, "TimeoutError")
    calls = []

    def retrieve(url, accept):
        calls.append((url, accept))
        if accept == "text/csv":
            query = parse_qs(urlsplit(url).query)["cmd"][0]
            if query.startswith("SELECT COUNT"):
                return b"#Table1\nrow_count\n0\n"
            return ("#Table1\n" + ",".join(APOGEE_DR17_COLUMNS) + "\n").encode()
        assert accept == "application/xml"
        if url.startswith(GALAH_DR3_TAP_URL):
            service.transport_failures.append(failure)
            return None
        return _crossmatch_response([(888, 999, 0.1, 0.01, True)])

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[],
                              ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap",
                              download=lambda *_: pytest.fail("implicit bulk download"))
    with pytest.raises(AcquisitionError, match="higher-priority.*connect_timeout") as exc:
        acquire_metallicity(tmp_path / "out", tmp_path / "cache", service, ["888"])
    galah_calls = [url for url, _accept in calls if url.startswith(GALAH_DR3_TAP_URL)]
    assert len(galah_calls) == 3 and len(calls) == 7 and exc.value.retryable
    assert _zeroed_galah_backoff == [2.0, 4.0]
    assert exc.value.network_failure is failure
    [record_path] = (tmp_path / "out").glob("metallicity_query_*.json")
    record = json.loads(record_path.read_text())
    galah = record["queries"]["galah"]
    assert record["status"] == galah["status"] == "unavailable" and "connect_timeout" in galah["reason"]
    transport_record = galah["transport"]
    assert transport_record["classification"] == "outage"
    assert transport_record["attempts"] == 3 and transport_record["bounded_by"] == "attempts"
    assert transport_record["budget_seconds"] == GALAH_DR3_TAP_BUDGET_SECONDS
    assert transport_record["last_failure"]["category"] == "connect_timeout"
    assert record["queries"]["apogee"]["row_count"] == 0
    # The outage defers the ladder; it is never answered by a lower rung.
    assert "gspspec" not in record["queries"] and "shboost" not in record
    assert not (tmp_path / "cache/GALAH_DR3_main_allstar_v2.fits").exists()


def test_completed_empty_targeted_response_advances_the_ladder_without_deferral():
    from trex.acquisition.metallicity import select_high_rungs

    calls = []
    link = _link("888", "999")

    def fetch_gspspec(_ids):
        calls.append("gspspec")
        return {"status": "available"}, [], None

    queries, measured, failure = select_high_rungs(
        ["888"],
        fetch_apogee=lambda _ids: (calls.append("apogee") or {"status": "available", "row_count": 0}, [], None),
        open_galah=lambda: (object(), {"url": GALAH_DR3_TAP_URL, "retrieved_utc": "2026-01-01T00:00:00Z",
                                       "bytes": 0}),
        fetch_crossmatch=lambda _ids: (calls.append("galah") or {"status": "available"}, [link], [link], None),
        read_galah=lambda *_args: [],
        retain_galah=lambda *_args: {"path": "empty.json", "release": {}},
        fetch_gspspec=fetch_gspspec,
    )
    assert failure is None and measured == {}
    assert calls == ["apogee", "galah", "gspspec"]
    assert queries["galah"]["status"] == "available" and queries["galah"]["row_count"] == 0
    assert "no exact or uniquely close GALAH row" in queries["galah"]["source_reasons"]["888"]


@pytest.mark.parametrize("valid_schema", [True, False])
def test_staged_release_reaches_the_real_consumer_or_records_schema_failure(tmp_path, valid_schema):
    from trex.acquisition.apogee_dr17 import APOGEE_DR17_COLUMNS
    from trex.acquisition.metallicity import acquire_metallicity
    from trex.acquisition.transport import AcquisitionError

    cache, output = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    path = cache / "GALAH_DR3_main_allstar_v2.fits"
    if valid_schema:
        _release(path)
    else:
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([])]).writeto(path)
    identity = {"url": GALAH_DR3_URL, "retrieved_utc": "2026-01-01T00:00:00Z", "bytes": path.stat().st_size}
    path.with_suffix(".retrieval.json").write_text(json.dumps(identity))
    calls = []

    def retrieve(url, accept):
        calls.append(url)
        if accept == "text/csv":
            query = parse_qs(urlsplit(url).query)["cmd"][0]
            if query.startswith("SELECT COUNT"):
                return b"#Table1\nrow_count\n0\n"
            return ("#Table1\n" + ",".join(APOGEE_DR17_COLUMNS) + "\n").encode()
        assert accept == "application/xml"
        return b'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
        <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>
        <FIELD name="dr3_source_id" datatype="long"/><FIELD name="dr2_source_id" datatype="long"/>
        <FIELD name="angular_distance" datatype="double"/><FIELD name="magnitude_difference" datatype="double"/>
        <FIELD name="proper_motion_propagation" datatype="boolean"/><DATA><TABLEDATA>
        <TR><TD>888</TD><TD>999</TD><TD>0.1</TD><TD>0.01</TD><TD>true</TD></TR>
        </TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'''

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[],
                              ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap",
                              download=lambda *_: pytest.fail("staged release downloaded again"))
    if not valid_schema:
        with pytest.raises(AcquisitionError, match="lacks required"):
            acquire_metallicity(output, cache, service, ["888"])
    else:
        result = acquire_metallicity(output, cache, service, ["888"])
        provenance = result["sources"]["888"]["provenance"]
        assert provenance["source"] == "galah-dr3" and provenance["value_dex"] == 0.25
        assert provenance["dr2_source_id"] == "999" and provenance["binding"]["row_index_zero_based"] == 3
        artifact = result["queries"]["galah"]["artifact"]
        payload = json.loads((output / artifact["path"]).read_text())
        assert payload["release"] == identity
        assert payload["rows"][0]["row_index_zero_based"] == 3
    assert len(calls) == 4
    [record_path] = output.glob("metallicity_query_*.json")
    record = json.loads(record_path.read_text())
    galah = record["queries"]["galah"]
    assert galah["status"] == ("available" if valid_schema else "unavailable")
    assert galah["release"] == identity
    for role in ("forward", "reverse"):
        response = galah[role]["responses"][0]
        assert (output / response["path"]).read_bytes().startswith(b"<VOTABLE")
    assert record.get("shboost", {}).get("status", "not-needed") == "not-needed"


def test_targeted_service_returns_exact_rows_and_retains_the_raw_response(tmp_path):
    dr3, dr2 = "9007199254740993", "456"
    body = _targeted_response([(17000000000000001, dr2, 0, 0, -0.25, 0.1, 42.0)])
    calls = []

    def retrieve(url, accept):
        calls.append((url, accept))
        return body

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[])
    release, identity = galah_shared_release(tmp_path / "cache", service)
    rows = galah_dr3_rows(release, [dr3], [dr2])
    assert rows == [{
        "sobject_id": 17000000000000001, "dr2_source_id": dr2, "dr3_source_id": None,
        "flag_sp": 0, "flag_fe_h": 0, "fe_h": -0.25, "e_fe_h": 0.1, "snr_c3_iraf": 42.0,
        "row_index_zero_based": 0,
    }]
    assert len(calls) == 1 and calls[0][1] == "application/xml"
    query = parse_qs(urlsplit(calls[0][0]).query)
    assert calls[0][0].startswith(GALAH_DR3_TAP_URL) and query["FORMAT"] == ["votable"]
    assert query["QUERY"][0].startswith(
        f"SELECT {','.join(GALAH_DR3_TARGETED_FIELDS)} FROM {GALAH_DR3_TAP_TABLE} WHERE source_id IN (")
    assert f"source_id IN ({dr2})" in query["QUERY"][0]
    assert identity["bytes"] == len(body) and identity["table"] == GALAH_DR3_TAP_TABLE
    assert identity["url"].startswith(GALAH_DR3_TAP_URL + "?") and identity["retrieved_utc"].endswith("Z")
    assert Path(identity["response_path"]).read_bytes() == body


def test_targeted_rows_resolve_dr3_identity_only_through_both_neighbourhood_answers(tmp_path):
    dr3, dr2 = "9007199254740993", "456"
    body = _targeted_response([(17000000000000001, dr2, 0, 0, -0.25, 0.1, 42.0)])
    service = SimpleNamespace(_http_get_bytes=lambda *_: body, transport_failures=[])
    release, _identity = galah_shared_release(tmp_path / "cache", service)
    [row] = galah_dr3_rows(release, [dr3], [dr2])
    link = _link(dr3, dr2)
    assert galah_dr3_associations([dr3], [row], [link], [link])[dr3] == {
        "status": "available", "row": row,
    }
    wrong_reverse = [link, _link("9007199254740995", dr2)]
    assert galah_dr3_associations([dr3], [row], [link], wrong_reverse)[dr3]["status"] == "unavailable"


def test_targeted_service_completed_empty_is_not_a_transport_failure(tmp_path):
    body = _targeted_response([])
    calls = []

    def retrieve(_url, _accept):
        calls.append(_url)
        return body

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[])
    release, identity = galah_shared_release(tmp_path, service)
    assert galah_dr3_rows(release, [], ["456"]) == []
    assert identity["bytes"] == len(body) and Path(identity["response_path"]).read_bytes() == body
    record = identity["transport"]
    assert len(calls) == 1 and record["classification"] == "empty-answer"
    assert record["attempts"] == 1 and record["bounded_by"] is None


def test_crowded_targeted_galah_query_uses_post_and_retains_every_identifier(tmp_path, monkeypatch):
    from io import BytesIO
    from trex.acquisition import transport

    dr2_ids = [str(9000000000000000000 + index) for index in range(868)]
    body = _targeted_response([
        (17000000000000001, dr2_ids[0], 0, 0, -0.25, 0.1, 42.0),
        (17000000000000002, dr2_ids[-1], 0, 0, 0.15, 0.1, 45.0),
    ])
    requests = []

    class Response(BytesIO):
        headers = {"Content-Length": str(len(body))}

    def respond(request, timeout):
        requests.append(request)
        assert request.get_method() == "POST" and request.full_url == GALAH_DR3_TAP_URL
        query = parse_qs(request.data.decode("ascii"))["QUERY"][0]
        assert query.endswith("IN (" + ",".join(dr2_ids) + ")")
        return Response(body)

    monkeypatch.setattr(transport, "urlopen", respond)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    release, identity = galah_shared_release(tmp_path / "cache", service)
    rows = galah_dr3_rows(release, [], dr2_ids)
    assert [row["dr2_source_id"] for row in rows] == [dr2_ids[0], dr2_ids[-1]]
    artifact = retain_galah_rows(tmp_path / "output", rows, identity, {})
    payload = json.loads((tmp_path / "output" / artifact["path"]).read_text())
    assert payload["rows"] == rows
    assert (tmp_path / "output" / payload["response"]["path"]).read_bytes() == body
    assert len(requests) == 1 and not service.transport_failures


@pytest.mark.parametrize("category,status", [("read_timeout", None), ("http_status", 503)])
def test_transient_outage_is_retried_within_the_recorded_budget(
        tmp_path, category, status, _zeroed_galah_backoff):
    from trex.acquisition.transport import AcquisitionFailure

    dr2 = "456"
    body = _targeted_response([(17000000000000001, dr2, 0, 0, -0.25, 0.1, 42.0)])
    failure = AcquisitionFailure("datacentral.org.au", category, True, 1,
                                 "synthetic transient failure", status, "SyntheticError")
    outcomes = [None, body]

    def retrieve(_url, _accept):
        result = outcomes.pop(0)
        if result is None:
            service.transport_failures.append(failure)
        return result

    service = SimpleNamespace(transport_failures=[])
    service._http_get_bytes = retrieve
    release, identity = galah_shared_release(tmp_path, service)
    assert len(galah_dr3_rows(release, [], [dr2])) == 1
    assert identity["bytes"] == len(body)
    record = identity["transport"]
    assert record["attempts"] == 2 and record["bounded_by"] is None
    assert record["classification"] == "answered" and record["last_failure"]["category"] == category
    assert record["budget_seconds"] == GALAH_DR3_TAP_BUDGET_SECONDS
    assert record["elapsed_seconds"] >= 0.0
    assert _zeroed_galah_backoff == [2.0]


def test_slow_outage_stops_at_the_wall_budget_before_the_attempt_bound(tmp_path, monkeypatch):
    from trex.acquisition import galah_dr3
    from trex.acquisition.transport import AcquisitionFailure

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(galah_dr3, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(galah_dr3, "GALAH_DR3_TAP_BUDGET_SECONDS", 10.0)
    failure = AcquisitionFailure("datacentral.org.au", "read_timeout", True, 1,
                                 "synthetic read timeout", None, "TimeoutError")
    calls = []

    def retrieve(_url, _accept):
        calls.append(clock.now)
        clock.now += 6.0
        service.transport_failures.append(failure)
        return None

    service = SimpleNamespace(transport_failures=[])
    service._http_get_bytes = retrieve
    release, identity = galah_shared_release(tmp_path, service)
    with pytest.raises(ValueError, match="outage.*bounded by budget") as exc:
        galah_dr3_rows(release, [], ["456"])
    record = identity["transport"]
    assert calls == [0.0, 6.0] and record["attempts"] == 2
    assert record["bounded_by"] == "budget" and record["classification"] == "outage"
    assert record["elapsed_seconds"] == 12.0 and record["budget_seconds"] == 10.0
    assert exc.value.network_failure is failure


@pytest.mark.parametrize("category,status,expected", [
    ("http_status", 400, "malformed-request"),
    ("http_status", 403, "malformed-request"),
    ("tls_verification", None, "tls_verification"),
])
def test_rejected_or_untrusted_request_is_not_retried(
        tmp_path, category, status, expected, _zeroed_galah_backoff):
    from trex.acquisition.transport import AcquisitionFailure

    failure = AcquisitionFailure("datacentral.org.au", category, False, 1,
                                 f"synthetic {expected}", status, "SyntheticError")
    calls = []

    def retrieve(_url, _accept):
        calls.append(_url)
        service.transport_failures.append(failure)
        return None

    service = SimpleNamespace(transport_failures=[])
    service._http_get_bytes = retrieve
    release, identity = galah_shared_release(tmp_path, service)
    with pytest.raises(ValueError, match=expected) as exc:
        galah_dr3_rows(release, [], ["456"])
    record = identity["transport"]
    assert len(calls) == 1 and record["classification"] == expected
    assert record["attempts"] == 1 and record["bounded_by"] == "classification"
    assert record["last_failure"]["retryable"] is False
    assert exc.value.network_failure is failure and _zeroed_galah_backoff == []


def test_unexpected_transport_error_stays_an_operational_defect(tmp_path):
    def retrieve(_url, _accept):
        raise RuntimeError("unexpected transport defect")

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[])
    release, _identity = galah_shared_release(tmp_path, service)
    with pytest.raises(RuntimeError, match="unexpected transport defect"):
        galah_dr3_rows(release, [], ["456"])


def test_targeted_service_failure_formats_an_embedding_diagnostic(tmp_path):
    class EmbeddingFailure:
        category = "connect_timeout"
        retryable = True

        def message(self):
            return "datacentral.org.au: timed out"

    failure = EmbeddingFailure()
    service = SimpleNamespace(transport_failures=[])

    def retrieve(_url, _accept):
        service.transport_failures.append(failure)
        return None

    service._http_get_bytes = retrieve
    release, _ = galah_shared_release(tmp_path, service)
    with pytest.raises(ValueError, match="connect_timeout: datacentral.org.au: timed out") as exc:
        galah_dr3_rows(release, [], ["456"])
    assert exc.value.network_failure is failure


def test_targeted_service_failure_carries_its_transport_diagnostic(tmp_path):
    from trex.acquisition.transport import AcquisitionFailure

    failure = AcquisitionFailure("datacentral.org.au", "connect_timeout", True, 1,
                                 "synthetic timeout", None, "TimeoutError")
    service = SimpleNamespace(transport_failures=[])

    def retrieve(_url, _accept):
        service.transport_failures.append(failure)
        return None

    service._http_get_bytes = retrieve
    release, _ = galah_shared_release(tmp_path, service)
    with pytest.raises(ValueError, match="connect_timeout: synthetic timeout") as exc:
        galah_dr3_rows(release, [], ["456"])
    assert exc.value.network_failure is failure
    assert not list(tmp_path.iterdir())


_FIELDS = (
    '<FIELD name="sobject_id" datatype="long"/><FIELD name="source_id" datatype="long"/>'
    '<FIELD name="flag_sp" datatype="int"/><FIELD name="flag_fe_h" datatype="int"/>'
    '<FIELD name="fe_h" datatype="double"/><FIELD name="e_fe_h" datatype="double"/>'
    '<FIELD name="snr_c3_iraf" datatype="double"/>'
)
_FIELDS_WITHOUT_SOBJECT = (
    '<FIELD name="source_id" datatype="long"/>'
    '<FIELD name="flag_sp" datatype="int"/><FIELD name="flag_fe_h" datatype="int"/>'
    '<FIELD name="fe_h" datatype="double"/><FIELD name="e_fe_h" datatype="double"/>'
    '<FIELD name="snr_c3_iraf" datatype="double"/>'
)
_FIELDS_FLOAT_SOURCE = _FIELDS.replace('<FIELD name="source_id" datatype="long"/>',
                                       '<FIELD name="source_id" datatype="double"/>')


def _targeted_response(rows, fields=_FIELDS):
    records = "".join("<TR>" + "".join(f"<TD>{value}</TD>" for value in row) + "</TR>" for row in rows)
    return (f'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
    <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>{fields}
    <DATA><TABLEDATA>{records}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>''').encode()


@pytest.mark.parametrize("body, message", [
    (b"<html>not a votable</html>", "incomplete"),
    (b'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
    <RESOURCE type="results"><INFO name="QUERY_STATUS" value="ERROR">table not found</INFO>
    </RESOURCE></VOTABLE>''', "incomplete"),
    (_targeted_response([(2, 0, 0, 0.1, 0.1, 40.0)], fields=_FIELDS_WITHOUT_SOBJECT),
     "required columns"),
    (_targeted_response([(1, "123.0", 0, 0, 0.1, 0.1, 40.0)], fields=_FIELDS_FLOAT_SOURCE),
     "64-bit"),
    (_targeted_response([(1, 999, 0, 0, 0.1, 0.1, 40.0)]), "unrequested"),
    (_targeted_response([]).replace(b"</RESOURCE>", b'<INFO name="QUERY_STATUS" value="OVERFLOW"/></RESOURCE>'),
     "incomplete"),
])
def test_targeted_service_refuses_malformed_or_unrequested_responses(tmp_path, body, message):
    service = SimpleNamespace(_http_get_bytes=lambda *_: body, transport_failures=[])
    release, identity = galah_shared_release(tmp_path, service)
    with pytest.raises(ValueError, match=message) as exc:
        galah_dr3_rows(release, ["123"], ["456"])
    if message == "unrequested":
        assert not hasattr(exc.value, "network_failure")
    else:
        assert exc.value.network_failure.retryable
    assert identity["bytes"] == len(body) and Path(identity["response_path"]).read_bytes() == body


def test_targeted_service_reaches_the_real_consumer_with_a_bound_raw_response(tmp_path):
    from trex.acquisition.apogee_dr17 import APOGEE_DR17_COLUMNS
    from trex.acquisition.metallicity import acquire_metallicity

    row_body = _targeted_response([(17000000000000001, "999", 0, 0, 0.25, 0.1, 40.0)])
    calls = []

    def retrieve(url, accept):
        calls.append((url, accept))
        if accept == "text/csv":
            query = parse_qs(urlsplit(url).query)["cmd"][0]
            if query.startswith("SELECT COUNT"):
                return b"#Table1\nrow_count\n0\n"
            return ("#Table1\n" + ",".join(APOGEE_DR17_COLUMNS) + "\n").encode()
        assert accept == "application/xml"
        if url.startswith(GALAH_DR3_TAP_URL):
            return row_body
        return _crossmatch_response([(888, 999, 0.1, 0.01, True)])

    service = SimpleNamespace(_http_get_bytes=retrieve, transport_failures=[],
                              ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap",
                              download=lambda *_: pytest.fail("implicit bulk download"))
    result = acquire_metallicity(tmp_path / "out", tmp_path / "cache", service, ["888"])
    assert len(calls) == 5
    provenance = result["sources"]["888"]["provenance"]
    assert provenance["source"] == "galah-dr3" and provenance["value_dex"] == 0.25
    assert provenance["binding"]["row_index_zero_based"] == 0
    artifact = result["queries"]["galah"]["artifact"]
    payload = json.loads((tmp_path / "out" / artifact["path"]).read_text())
    assert payload["release"]["table"] == GALAH_DR3_TAP_TABLE
    assert payload["release"]["row_space"] == "votable_response"
    assert payload["release"]["bytes"] == len(row_body) and payload["rows"][0]["dr3_source_id"] is None
    assert payload["rows"][0]["dr2_source_id"] == "999"
    assert (tmp_path / "out" / payload["response"]["path"]).read_bytes() == row_body
    assert (tmp_path / "out" / payload["response"]["path"]).name == artifact["response"]["path"]
    assert not (tmp_path / "cache/GALAH_DR3_main_allstar_v2.fits").exists()
    assert not list((tmp_path / "cache").glob("galah_dr3_tap_*"))


def test_galah_matched_rows_bind_to_the_retained_release(tmp_path):
    row = {"row_index_zero_based": 3, "fe_h": -0.25}
    metadata = {"url": GALAH_DR3_URL, "retrieved_utc": "2026-01-01T00:00:00Z", "bytes": 1}
    artifact = retain_galah_rows(tmp_path, [row], metadata, {"123": {"status": "available", "row": row}})
    payload = json.loads((tmp_path / artifact["path"]).read_text())
    assert payload["release"] == metadata and payload["rows"] == [row]
    assert payload["associations"]["123"]["row_index_zero_based"] == 3


def test_an_unreadable_receipt_is_refused_rather_than_treated_as_a_web_release(tmp_path):
    path = tmp_path / "GALAH_DR3_main_allstar_v2.fits"
    _release(path)
    path.with_suffix(".retrieval.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        galah_shared_release(tmp_path, SimpleNamespace(download=None, transport_failures=[]), download=False)


def test_no_requested_identities_scan_no_table_column(tmp_path):
    path = tmp_path / "allstar.fits"
    _release(path)
    assert galah_dr3_rows(path, [], []) == []


def test_galah_crossmatch_retains_both_real_boundary_responses(tmp_path):
    def retrieve(url, accept):
        assert accept == "application/xml"
        query = parse_qs(urlsplit(url).query)["QUERY"][0]
        assert "dr3_source_id" in query or "dr2_source_id" in query
        return b'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
        <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>
        <FIELD name="dr3_source_id" datatype="long"/><FIELD name="dr2_source_id" datatype="long"/>
        <FIELD name="angular_distance" datatype="double"/><FIELD name="magnitude_difference" datatype="double"/>
        <FIELD name="proper_motion_propagation" datatype="boolean"/><DATA><TABLEDATA>
        <TR><TD>123</TD><TD>456</TD><TD>0.1</TD><TD>0.01</TD><TD>true</TD></TR>
        </TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'''

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=retrieve,
                              ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap")
    record, forward, reverse, failure = fetch_galah_gaia_neighborhood(tmp_path, service, ["123"])
    assert failure is None and len(forward) == len(reverse) == 1
    for role in ("forward", "reverse"):
        artifact = record[role]["responses"][0]
        assert (tmp_path / artifact["path"]).is_file()
        assert artifact["uri"].startswith("https://") and artifact["retrieved_utc"].endswith("Z")


def _release(path):
    columns = [
        fits.Column(name="sobject_id", format="K", array=np.array([10, 11, 12, 13], dtype=np.int64)),
        fits.Column(name="dr2_source_id", format="K", array=np.array([456, 456, 777, 999], dtype=np.int64)),
        fits.Column(name="dr3_source_id", format="K", array=np.array([123, 789, 123, 888], dtype=np.int64)),
        fits.Column(name="flag_sp", format="K", array=np.array([0, 0, 1, 0], dtype=np.int64)),
        fits.Column(name="flag_fe_h", format="K", array=np.array([0, 0, 0, 0], dtype=np.int64)),
        fits.Column(name="fe_h", format="E", array=np.array([-0.25, -0.5, np.nan, 0.25], dtype=np.float32)),
        fits.Column(name="e_fe_h", format="D", array=np.array([0.1, 0.2, 0.3, 0.1])),
        fits.Column(name="snr_c3_iraf", format="D", array=np.array([40.0, 50.0, 60.0, 70.0])),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)


def test_galah_reader_retains_all_dr2_and_dr3_candidates(tmp_path):
    path = tmp_path / "allstar.fits"
    _release(path)
    rows = galah_dr3_rows(path, ["123"], ["456"])
    assert [(row["dr3_source_id"], row["dr2_source_id"]) for row in rows] == [
        ("123", "456"), ("789", "456"), ("123", "777"),
    ]
    assert [row["row_index_zero_based"] for row in rows] == [0, 1, 2]
    assert rows[0]["fe_h"] == -0.25 and rows[0]["e_fe_h"] == 0.1
    assert rows[2]["fe_h"] is None and rows[2]["flag_sp"] == 1
    assert galah_dr3_rows(path, ["9999"], []) == []


def test_galah_reader_rejects_float_id_and_truncated_release(tmp_path):
    path = tmp_path / "allstar.fits"
    _release(path)
    with pytest.raises(ValueError, match="exact integer"):
        galah_dr3_rows(path, [123.0], [])
    data = path.read_bytes()
    path.write_bytes(data[:-2880])
    with pytest.raises(ValueError, match="truncated|completely"):
        galah_dr3_rows(path, ["123"], [])


def test_galah_reader_and_association_preserve_ids_above_float_precision(tmp_path):
    path = tmp_path / "allstar.fits"
    _release(path)
    dr3, dr2 = "9007199254740993", "9007199254740995"
    with fits.open(path, mode="update") as hdus:
        hdus[1].data["dr3_source_id"][3] = int(dr3)
        hdus[1].data["dr2_source_id"][3] = int(dr2)
    [row] = galah_dr3_rows(path, [dr3], [dr2])
    assert row["dr3_source_id"] == dr3 and row["dr2_source_id"] == dr2
    link = _link(dr3, dr2)
    assert galah_dr3_associations([dr3], [row], [link], [link])[dr3] == {"status": "available", "row": row}


def _link(dr3, dr2, distance=0.1):
    return {"dr3_source_id": dr3, "dr2_source_id": dr2, "angular_distance": distance,
            "magnitude_difference": 0.01, "proper_motion_propagation": True}


def test_galah_requires_bidirectional_identity_and_keeps_ambiguities(tmp_path):
    path = tmp_path / "allstar.fits"
    _release(path)
    good, foreign, _ = galah_dr3_rows(path, ["123"], ["456"])
    link = _link("123", "456")
    assert galah_dr3_associations(["123"], [good], [link], [link])["123"] == {
        "status": "available", "row": good,
    }
    assert galah_dr3_associations(["123"], [], [link], [link])["123"]["status"] == "unavailable"
    assert "ambiguous" in galah_dr3_associations(["123"], [good, foreign], [link], [link])["123"]["reason"]
    reverse = [link, _link("789", "456")]
    assert "ambiguous" in galah_dr3_associations(["123"], [good], [link], reverse)["123"]["reason"]
    with pytest.raises(ValueError, match="incomplete"):
        galah_dr3_associations(["123"], [good], [link], [])
    with pytest.raises(ValueError, match="disagree"):
        galah_dr3_associations(["123"], [good], [link], [_link("123", "456", 2.0)])


@pytest.mark.parametrize("change", [
    {"flag_sp": 1}, {"flag_fe_h": 1}, {"snr_c3_iraf": 30.0}, {"e_fe_h": 0}, {"fe_h": None},
])
def test_galah_rejects_unsupported_iron_measurements(tmp_path, change):
    path = tmp_path / "allstar.fits"
    _release(path)
    row = galah_dr3_rows(path, ["123"], [])[0]
    link = _link("123", "456")
    result = galah_dr3_associations(["123"], [{**row, **change}], [link], [link])["123"]
    assert result["status"] == "unavailable" and "quality" in result["reason"]


def _crossmatch_response(rows):
    records = "".join(
        "<TR>" + "".join(f"<TD>{str(value).lower() if isinstance(value, bool) else value}</TD>"
                         for value in row) + "</TR>" for row in rows)
    return (f'''<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
    <RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>
    <FIELD name="dr3_source_id" datatype="long"/><FIELD name="dr2_source_id" datatype="long"/>
    <FIELD name="angular_distance" datatype="double"/><FIELD name="magnitude_difference" datatype="double"/>
    <FIELD name="proper_motion_propagation" datatype="boolean"/><DATA><TABLEDATA>{records}</TABLEDATA></DATA>
    </TABLE></RESOURCE></VOTABLE>''').encode()
