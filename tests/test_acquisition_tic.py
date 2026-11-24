"""TIC responses preserve identities, completeness, and raw evidence."""

import json
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from trex.acquisition import transport as network
from trex.acquisition.tic import fetch_tic_by_gaia, fetch_tic_by_gaia_batch, fetch_tic_cone, fetch_tic_target, tic_neighbor_entry
from trex.acquisition.transport import AcquisitionError


def test_tic_unique_usable_row_and_ambiguous_rows():
    good = {"GAIA": "123", "ID": 10, "mass": 0.8, "rad": 0.7, "disposition": "NONE"}
    duplicate = {**good, "ID": 11, "disposition": "DUPLICATE"}
    selected = tic_neighbor_entry([good, duplicate], "123")
    assert selected["status"] == "available" and selected["duplicate_resolution"]["selected_tic_id"] == "10"
    ambiguous = tic_neighbor_entry([good, {**good, "ID": 12}], "123")
    assert ambiguous["status"] == "unavailable" and len(ambiguous["tic_rows"]) == 2
    assert tic_neighbor_entry([{**good, "mass": None}], "123")["status"] == "unavailable"


def test_target_lookup_retains_exact_response_and_query(tmp_path):
    payload = {"status": "COMPLETE", "data": [{"ID": "123", "ra": 1.0, "dec": 2.0}]}
    body = json.dumps(payload).encode()

    def post(url, request):
        query = json.loads(parse_qs(request.decode())["request"][0])
        assert url.startswith("https://mast.stsci.edu/")
        assert query["params"]["filters"][0]["values"] == ["123"]
        return body

    row, record = fetch_tic_target("123", tmp_path, SimpleNamespace(transport_failures=[], post_mast=post))
    assert row == payload["data"][0]
    assert (tmp_path / record["response"]).read_bytes() == body
    assert json.loads((tmp_path / "tic_target_query.json").read_text())["retrieved_utc"].endswith("Z")


@pytest.mark.parametrize("payload,reason", [
    ({"status": "EXECUTING", "data": []}, "did not complete"),
    ({"status": "COMPLETE", "data": []}, "exactly one"),
    ({"status": "COMPLETE", "data": [{"ID": "456"}]}, "different ID"),
    ({"status": "COMPLETE", "data": [{"ID": "123"}], "paging": {"pagesFiltered": 2}}, "paginated"),
])
def test_target_refuses_incomplete_or_foreign_results_and_retains_response(tmp_path, payload, reason):
    body = json.dumps(payload).encode()
    service = SimpleNamespace(transport_failures=[], post_mast=lambda *_args: body)
    with pytest.raises(AcquisitionError, match=reason):
        fetch_tic_target("123", tmp_path, service)
    assert (tmp_path / "tic_target.json").read_bytes() == body


def test_neighbor_lookup_retains_all_duplicate_rows_and_query(tmp_path):
    rows = [{"ID": 11, "GAIA": "456", "disposition": "DUPLICATE"},
            {"ID": 12, "GAIA": "456", "mass": 1.0, "rad": 1.0}]
    body = json.dumps({"status": "COMPLETE", "data": rows, "paging": {"pagesFiltered": 1, "rowsFiltered": 2}}).encode()

    def post(_url, request):
        query = json.loads(parse_qs(request.decode())["request"][0])
        assert query["params"]["filters"][0] == {"paramName": "GAIA", "values": ["456"]}
        return body

    result, record = fetch_tic_by_gaia("456", tmp_path, SimpleNamespace(transport_failures=[], post_mast=post))
    assert result == rows and (tmp_path / record["response"]).read_bytes() == body
    assert len(list(tmp_path.glob("*.query.json"))) == 1
    assert tic_neighbor_entry(result, "456")["tic_id"] == "12"


def test_neighbor_lookup_refuses_an_incomplete_page_but_keeps_raw_bytes(tmp_path):
    body = json.dumps({"status": "COMPLETE", "data": [], "paging": {"pagesFiltered": 1, "rowsFiltered": 2}}).encode()
    with pytest.raises(AcquisitionError, match="incomplete or paginated"):
        fetch_tic_by_gaia("456", tmp_path, SimpleNamespace(transport_failures=[], post_mast=lambda *_: body))
    assert any(path.read_bytes() == body for path in tmp_path.glob("*.json"))


def test_batched_tic_lookup_keeps_absent_rows_absent_and_refuses_foreign_rows(tmp_path):
    def query(rows):
        body = json.dumps({"status": "COMPLETE", "data": rows}).encode()

        def post(_url, request):
            payload = json.loads(parse_qs(request.decode())["request"][0])
            assert payload["params"]["filters"] == [{"paramName": "GAIA", "values": ["456", "789"]}]
            return body

        return fetch_tic_by_gaia_batch("456 789".split(), tmp_path, SimpleNamespace(transport_failures=[], post_mast=post))

    row = {"ID": 12, "GAIA": "456", "mass": 1.0, "rad": 1.0}
    rows, receipt = query([row])
    assert rows == [row]
    assert tic_neighbor_entry(rows, "789")["status"] == "unavailable"
    assert json.loads((tmp_path / receipt["response"]).read_text())["data"] == [row]
    with pytest.raises(AcquisitionError, match="unrequested Gaia identity"):
        query([row, {**row, "GAIA": "790"}])


def _install_replies(monkeypatch, replies):
    pending = iter(replies)
    calls = []

    class Response(BytesIO):
        headers = {}

    def respond(request, timeout):
        calls.append(timeout)
        reply = next(pending)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply)

    monkeypatch.setattr(network, "urlopen", respond)
    monkeypatch.setattr(network, "sleep_before_retry", lambda _delay: None)
    return calls


def test_target_provider_error_attempt_is_retained_before_recovery(tmp_path, monkeypatch):
    error = b'{"status":"ERROR","msg":"TIC service busy"}'
    success = json.dumps({"status": "COMPLETE", "data": [{"ID": "123", "ra": 1.0, "dec": 2.0}]}).encode()
    calls = _install_replies(monkeypatch, [error, success])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=2, timeout_seconds=4))

    row, record = fetch_tic_target("123", tmp_path, service)

    assert row["ID"] == "123" and len(calls) == 2 and calls == [4, 4]
    history = record["attempt_history"]
    assert history[0]["failure"]["category"] == "provider_error"
    assert history[0]["failure"]["message"] == "TIC service busy"
    assert (tmp_path / history[0]["response"]).read_bytes() == error
    assert (tmp_path / history[1]["response"]).read_bytes() == success
    assert json.loads((tmp_path / "tic_target_query.json").read_text())["status"] == "available"


def test_target_exhaustion_keeps_a_failure_record_and_raw_attempts(tmp_path, monkeypatch):
    first, second = b'{"status":"ERROR","msg":"busy"}', TimeoutError("second attempt timed out")
    calls = _install_replies(monkeypatch, [first, second])
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=2))

    with pytest.raises(AcquisitionError, match="second attempt timed out") as caught:
        fetch_tic_target("123", tmp_path, service)

    assert len(calls) == 2 and caught.value.retryable
    assert "TIC target query" in str(caught.value)
    assert caught.value.network_failure.category == "connect_timeout"
    failure_record = json.loads((tmp_path / caught.value.failure_record).read_text())
    assert failure_record["status"] == "unavailable" and len(failure_record["attempt_history"]) == 2
    assert failure_record["category"] == "connect_timeout"
    retained = failure_record["attempt_history"][0]
    assert (tmp_path / retained["response"]).read_bytes() == first
    assert not (tmp_path / "tic_target_query.json").exists()


def test_a_transport_acquisition_error_still_retains_attempts_and_a_failure(tmp_path, monkeypatch):
    class Response(BytesIO):
        headers = {}

    monkeypatch.setattr(network, "urlopen", lambda *_args, **_kwargs: Response(b"too-large"))
    service = network.AcquisitionTransport(network.NetworkPolicy(True, attempts=1, max_response_bytes=3))
    with pytest.raises(AcquisitionError, match="size limit") as caught:
        fetch_tic_target("123", tmp_path, service)
    failure_record = json.loads((tmp_path / caught.value.failure_record).read_text())
    assert failure_record["status"] == "unavailable"
    assert failure_record["attempt_history"][0]["body_issue"] == "response exceeded the configured size limit"
    attempt_body = (tmp_path / failure_record["attempt_history"][0]["response"]).read_bytes()
    assert attempt_body == b"too-"
    assert not (tmp_path / "tic_target_query.json").exists()


def test_cone_query_converts_arcseconds_to_degrees(tmp_path):
    def post(_url, request):
        query = json.loads(parse_qs(request.decode())["request"][0])
        assert query["service"] == "Mast.Catalogs.Tic.Cone"
        assert query["params"] == {"ra": 10.0, "dec": 20.0, "radius": 1.0 / 60.0}
        return b'{"status":"COMPLETE","data":[]}'

    rows, _ = fetch_tic_cone(10.0, 20.0, 60.0, tmp_path, SimpleNamespace(transport_failures=[], post_mast=post))
    assert rows == []
