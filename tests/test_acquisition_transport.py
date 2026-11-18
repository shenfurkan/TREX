"""Transport outages defer; access denials do not retry."""

from http.client import RemoteDisconnected
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from trex.acquisition import transport
from trex.acquisition.gspspec import fetch_gspspec


@pytest.mark.parametrize("status,expected_calls,retryable", [(403, 1, False), (401, 1, False), (503, 3, True)])
def test_http_failures_obey_retry_policy(monkeypatch, status, expected_calls, retryable):
    calls, waits = [], []

    def fail(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "unavailable", {}, None)

    monkeypatch.setattr(transport, "urlopen", fail)
    monkeypatch.setattr(transport, "sleep_before_retry", waits.append)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    assert service._http_get_bytes("https://example.org/query", "text/csv") is None
    failure = service.transport_failures[-1]
    assert len(calls) == expected_calls and failure.retryable is retryable
    assert len(waits) == expected_calls - 1 and failure.http_status == status


def test_gspspec_does_not_fall_through_after_access_denial(tmp_path, monkeypatch):
    calls = []

    def fail(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(transport, "urlopen", fail)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    record, rows, failure = fetch_gspspec(tmp_path, service, ["123"])
    assert len(calls) == 1 and rows is None and record["status"] == "unavailable"
    assert failure.http_status == 403 and not failure.retryable


def test_offline_and_dns_failure_are_explicit(monkeypatch):
    service = transport.AcquisitionTransport(transport.NetworkPolicy(False))
    with pytest.raises(transport.AcquisitionError, match="disabled"):
        service._http_get_bytes("https://example.org/", "application/xml")
    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")))
    monkeypatch.setattr(transport, "sleep_before_retry", lambda _delay: None)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=2))
    assert service._http_get_bytes("https://example.org/", "application/xml") is None
    assert service.transport_failures[-1].retryable


def test_streamed_download_is_complete_and_never_overwrites(tmp_path, monkeypatch):
    from io import BytesIO

    class Response(BytesIO):
        headers = {"Content-Length": "3"}

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"abc"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    path = tmp_path / "release"
    assert service.download("https://example.org/release", path) == path
    assert path.read_bytes() == b"abc"
    with pytest.raises(transport.AcquisitionError, match="already exists"):
        service.download("https://example.org/release", path)
    assert path.read_bytes() == b"abc"


def test_truncated_stream_is_a_retryable_failure(tmp_path, monkeypatch):
    from io import BytesIO

    class Response(BytesIO):
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"abc"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    assert service.download("https://example.org/release", tmp_path / "release") is None
    assert service.transport_failures[-1].retryable
    assert not (tmp_path / "release").exists()


def test_form_post_uses_explicit_policy_and_content_type(monkeypatch):
    from io import BytesIO

    calls = []

    class Response(BytesIO):
        headers = {"Content-Length": "2"}

    def respond(request, timeout):
        calls.append(request)
        return Response(b"{}")

    monkeypatch.setattr(transport, "urlopen", respond)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    assert service.post_form("https://example.org/invoke", b"request=value") == b"{}"
    assert calls[0].get_method() == "POST" and calls[0].data == b"request=value"
    assert calls[0].get_header("Content-type") == "application/x-www-form-urlencoded"


def test_truncated_in_memory_response_is_not_accepted(monkeypatch):
    from io import BytesIO

    class Response(BytesIO):
        headers = {"Content-Length": "10"}

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"{}"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    assert service.post_form("https://example.org/invoke", b"request=value") is None
    assert service.transport_failures[-1].retryable


def test_server_closing_without_response_is_a_retryable_outage(monkeypatch):
    calls = []

    def disconnect(request, timeout):
        calls.append(request.full_url)
        raise RemoteDisconnected("Remote end closed connection without response")

    monkeypatch.setattr(transport, "urlopen", disconnect)
    monkeypatch.setattr(transport, "sleep_before_retry", lambda _delay: None)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=3))
    assert service._http_get_bytes("https://example.org/tap", "application/xml") is None
    failure = service.transport_failures[-1]
    assert len(calls) == 3 and failure.retryable and failure.exception_class == "RemoteDisconnected"


def test_mast_error_status_is_retried_then_left_for_the_caller(monkeypatch):
    replies = [b'{"status" : "ERROR","msg" : "Request failed."}', b'{"status": "COMPLETE", "data": []}']
    calls = []

    class Response:
        def __init__(self, body):
            self.body, self.headers = body, {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _limit):
            return self.body

    def respond(request, timeout):
        calls.append(request.full_url)
        return Response(replies[min(len(calls) - 1, len(replies) - 1)])

    monkeypatch.setattr(transport, "urlopen", respond)
    monkeypatch.setattr(transport, "sleep_before_retry", lambda _delay: None)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=3))
    assert service.post_mast("https://example.org/invoke", b"request=x") == replies[1] and len(calls) == 2
    replies[1] = replies[0]
    calls.clear()
    assert service.post_mast("https://example.org/invoke", b"request=x") == replies[0] and len(calls) == 3
    failure = service.mast_attempts[-1]["failure"]
    assert failure.category == "provider_error" and failure.http_status == 200
    assert failure.attempts == 3 and failure.reason == "Request failed."
    assert not service.transport_failures


def test_mast_provider_and_network_errors_share_one_retry_budget(monkeypatch):
    error_body = b'{"status":"ERROR","msg":"MAST busy"}'
    outcomes = iter([error_body, TimeoutError("timeout one"), TimeoutError("timeout two")])
    calls, waits = [], []

    class Response(BytesIO):
        headers = {}

    def respond(request, timeout):
        calls.append(timeout)
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return Response(result)

    monkeypatch.setattr(transport, "urlopen", respond)
    monkeypatch.setattr(transport, "sleep_before_retry", waits.append)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=3, timeout_seconds=7, backoff_seconds=2))
    assert service.post_mast("https://example.org/invoke", b"request=x") is None
    assert calls == [7, 7, 7] and waits == [2, 4]
    failure = service.transport_failures[-1]
    assert failure.category == "connect_timeout" and failure.attempts == 3 and "timeout two" in failure.reason
    assert service.mast_attempts[0]["body"] == error_body
    assert [item["failure"].category for item in service.mast_attempts] == [
        "provider_error", "connect_timeout", "connect_timeout"]


def test_mast_recovers_from_mixed_errors_without_losing_attempt_history(monkeypatch):
    error_body = b'{"status":"ERROR","msg":"try later"}'
    result_body = b'{"status":"COMPLETE","data":[]}'
    outcomes = iter([TimeoutError("first"), error_body, result_body])

    class Response(BytesIO):
        headers = {}

    def respond(_request, timeout):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    monkeypatch.setattr(transport, "urlopen", respond)
    monkeypatch.setattr(transport, "sleep_before_retry", lambda _delay: None)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=3))
    assert service.post_mast("https://example.org/invoke", b"request=x") == result_body
    assert service.mast_attempts[0]["failure"].category == "connect_timeout"
    assert service.mast_attempts[1]["body"] == error_body
    assert service.mast_attempts[1]["failure"].reason == "try later"
    assert service.mast_attempts[2]["body"] == result_body
    assert not service.transport_failures


def test_mast_http_error_body_is_kept_and_nonretryable_failure_stops(monkeypatch):
    calls = []
    body = b"access denied by archive"

    def refuse(request, timeout):
        calls.append(timeout)
        raise HTTPError(request.full_url, 403, "denied", {}, BytesIO(body))

    monkeypatch.setattr(transport, "urlopen", refuse)
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=3))
    assert service.post_mast("https://example.org/invoke", b"request=x") is None
    assert len(calls) == 1 and service.mast_attempts[0]["body"] == body
    failure = service.transport_failures[-1]
    assert failure.category == "http_status" and failure.http_status == 403 and not failure.retryable


def test_mast_incomplete_response_keeps_partial_bytes(monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"partial"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    assert service.post_mast("https://example.org/invoke", b"request=x") is None
    attempt = service.mast_attempts[0]
    assert attempt["body"] == b"partial" and "IncompleteRead" in attempt["body_issue"]
    assert attempt["failure"].category == "incomplete_transfer"


def _captured_user_agent(monkeypatch, policy):
    seen = []

    def refuse(request, timeout):
        seen.append(request.get_header("User-agent"))
        raise HTTPError(request.full_url, 403, "denied", {}, None)

    monkeypatch.setattr(transport, "urlopen", refuse)
    transport.AcquisitionTransport(policy)._http_get_bytes("https://example.org/query", "text/csv")
    return seen


def test_requests_carry_trex_identity_unless_the_caller_supplies_one(monkeypatch):
    assert _captured_user_agent(monkeypatch, transport.NetworkPolicy(True)) == ["trex-acquisition/1"]
    agents = iter(["client-a/1", "client-b/2"])
    policy = transport.NetworkPolicy(True, user_agent=lambda: next(agents))
    assert _captured_user_agent(monkeypatch, policy) == ["client-a/1"]


@pytest.mark.parametrize("value", ["", "two\r\nlines", None])
def test_a_malformed_caller_user_agent_is_refused(value):
    with pytest.raises(ValueError, match="single-line"):
        transport.request_user_agent(transport.NetworkPolicy(True, user_agent=lambda: value))
    with pytest.raises(ValueError, match="callable"):
        transport.NetworkPolicy(True, user_agent="client/1")


def test_trilegal_session_uses_the_caller_user_agent():
    from trex.acquisition.trilegal_tls import create_session

    session = create_session(transport.NetworkPolicy(True, user_agent=lambda: "client-c/3"))
    assert session.headers["User-Agent"] == "client-c/3"
    assert create_session(transport.NetworkPolicy(True)).headers["User-Agent"] == "trex-acquisition/1"


@pytest.mark.parametrize("field, value", [
    ("timeout_seconds", True), ("timeout_seconds", 0), ("timeout_seconds", float("nan")),
    ("backoff_seconds", False), ("backoff_seconds", -1.0), ("backoff_seconds", "1"),
    ("attempts", True), ("max_response_bytes", 0), ("max_download_bytes", 1.5),
])
def test_network_policy_refuses_non_numeric_or_out_of_range_limits(field, value):
    with pytest.raises(ValueError):
        transport.NetworkPolicy(True, **{field: value})


def test_public_get_bytes_uses_the_http_get_seam(monkeypatch):
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    calls = []

    def seam(url, accept):
        calls.append((url, accept))
        return b"body"

    monkeypatch.setattr(service, "_http_get_bytes", seam)
    assert service.get_bytes("https://example.org/image", "application/fits") == b"body"
    assert calls == [("https://example.org/image", "application/fits")]


def test_malformed_content_length_is_a_retryable_incomplete_transfer(monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": "three"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"abc"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1))
    assert service._http_get_bytes("https://example.org/query", "text/csv") is None
    assert service.transport_failures[-1].category == "incomplete_transfer"
    assert service.transport_failures[-1].retryable


def test_oversized_response_is_refused(monkeypatch):
    class Response(BytesIO):
        headers = {}

    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: Response(b"abcd"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True, attempts=1, max_response_bytes=3))
    with pytest.raises(transport.AcquisitionError, match="size limit"):
        service._http_get_bytes("https://example.org/query", "text/csv")


def test_existing_or_linked_download_destination_is_refused_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "urlopen", lambda *_args, **_kwargs: pytest.fail("network was opened"))
    service = transport.AcquisitionTransport(transport.NetworkPolicy(True))
    existing = tmp_path / "release"
    existing.write_bytes(b"cached")
    with pytest.raises(transport.AcquisitionError, match="already exists"):
        service.download("https://example.org/release", existing)
    link = tmp_path / "link"
    link.symlink_to(existing)
    with pytest.raises(ValueError, match="symlink"):
        service.download("https://example.org/release", link)
    assert existing.read_bytes() == b"cached"
