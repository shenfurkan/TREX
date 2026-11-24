"""The standalone TRILEGAL producer retains its complete response."""

import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from trex.acquisition import trilegal
from trex.acquisition.transport import NetworkPolicy

# The provider form offers both coordinate systems; these are placeholder defaults.
FORM_DEFAULTS = {"gal_coord": "1", "gc_l": "0", "gc_b": "90", "eq_alpha": "0", "eq_delta": "0",
                 "field": "1", "photsys_file": "tab_mag_odfnew/tab_mag_2mass.dat", "icm_lim": "1",
                 "mag_lim": "20", "binary_kind": "1", "extinction_infty": "0.0378"}
FORM = '<form action="/cgi-bin/trilegal_1.6">' + ''.join(
    f'<input name="{key}" value="{value}">' for key, value in FORM_DEFAULTS.items()) + '</form>'
def expected_fields(field_extinction):
    return {**FORM_DEFAULTS, **trilegal.form_overrides(1.0, 2.0, field_extinction["av_infinity_mag"])}
# ICRS definition of the Galactic system (ESA 1997, The Hipparcos and Tycho
# Catalogues, Vol. 1, Sec. 1.5.3): north Galactic pole and the l = b = 0 direction.
NORTH_GALACTIC_POLE = (192.85948, 27.12825)
GALACTIC_CENTRE = (266.40499, -28.93617)
SUBMITTED_PAGE = '<a href="/tmp/output.dat">output</a>'
RAW = "TESS Mact logg logTe [M/H]\n12 1 4.4 3.76 0.0\n#TRILEGAL normally terminated\n"
RETAINED_URL = "https://stev.oapd.inaf.it/tmp/running.dat"
RETAINED_FORM = [["gal_coord", "1"], ["gc_l", "96.3"], ["gc_b", "-58.8"], ["extinction_infty", "0.5000"]]


def response(text, status=200):
    reply = requests.Response()
    reply.status_code = status
    reply.encoding = "utf-8"
    reply.raw = BytesIO(text.encode("utf-8"))
    reply.close = Mock(wraps=reply.close)
    return reply


def retained_request(tmp_path, field_extinction):
    """Write the submitted job a rerun resumes for this field."""
    path = tmp_path / trilegal.REQUEST_FILE_NAME
    path.write_text(json.dumps({
        "status": "submitted", "ra_deg": 1.0, "dec_deg": 2.0, "field_submission": trilegal.FIELD_SUBMISSION,
        "field_extinction": field_extinction, "output_url": RETAINED_URL, "submitted_form": RETAINED_FORM}))
    return path


@pytest.mark.parametrize(("map_av", "form_av"), [
    (0.5, "0.5000"),
    (0.01575, "0.0158"),
    (1.00105, "1.0010"),
])
def test_standalone_trilegal_retains_population_and_request(tmp_path, monkeypatch, field_extinction, map_av, form_av):
    field_extinction["av_infinity_mag"] = map_av
    replies = [response(FORM), response(SUBMITTED_PAGE), response(RAW)]

    class Session:
        def get(self, url, **kwargs):
            if url == trilegal.FORM_URL:
                return replies[0]
            return replies[2]

        def post(self, url, **kwargs):
            assert dict(kwargs["data"]) == expected_fields(field_extinction)
            return replies[1]

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Session())
    population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    for reply in replies:
        reply.close.assert_called_once()
    assert population.read_text() == "TESS,Mact,logg,logTe,[M/H]\n12,1,4.4,3.76,0.0\n"
    record = json.loads(query.read_text())
    assert record["star_count"] == 1 and record["ra_deg"] == 1.0
    assert record["output_url"].endswith("/tmp/output.dat") and record["retrieved_utc"].endswith("Z")
    # The form's extinction is the map value, and the query keeps the map record.
    assert dict(record["submitted_form"])["extinction_infty"] == form_av
    assert record["field_extinction"] == field_extinction
    assert trilegal.is_galactic_field_record(record)


def test_offline_policy_refuses_network(tmp_path):
    with pytest.raises(trilegal.TrilegalUnavailableError, match="disabled"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(False))
    with pytest.raises(trilegal.TrilegalUnavailableError, match="disabled"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(False),
                                  query_runner=lambda *_args: pytest.fail("unexpected network query"))


@pytest.mark.parametrize("page,match", [
    ("<p>server busy</p>", "server busy"),
    ('<a href="https://foreign.example/tmp/output">output</a>', "unexpected result link"),
    ('<a href="http://stev.oapd.inaf.it/tmp/output">output</a>', "unexpected result link"),
])
def test_unusable_result_links_are_refused(page, match):
    with pytest.raises(trilegal.TrilegalUnavailableError, match=match):
        trilegal.result_url(trilegal.FORM_URL, page)


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (502, True), (503, True),
                                              (400, False), (403, False), (404, False)])
def test_http_status_errors_carry_the_status_and_map_retryability(status, retryable):
    from types import SimpleNamespace

    error = trilegal._http_unavailable(SimpleNamespace(status_code=status), "result download")
    manifest = error.to_manifest()
    assert error.retryable is retryable
    assert manifest["category"] == "http_status" and manifest["http_status"] == status
    assert manifest["status"] == "unavailable"


def test_polling_does_not_sleep_after_the_last_attempt():
    replies = [response("", 404) for _ in range(3)]
    pending = iter(replies)
    waits = []
    session = SimpleNamespace(get=lambda *_args, **_kwargs: next(pending))
    with pytest.raises(trilegal.TrilegalUnavailableError, match="polling budget"):
        trilegal.poll_result(session, trilegal.FORM_URL, timeout=1, attempts=3, interval=10,
                             sleep=waits.append, check_shutdown=lambda: None, progress_callback=None)
    assert waits == [10, 10]
    for reply in replies:
        reply.close.assert_called_once()


def test_records_refuse_non_finite_json(tmp_path):
    from trex.acquisition.storage import write_bytes

    with pytest.raises(ValueError):
        trilegal._write_record(tmp_path / "record.json", write_bytes, {"value": float("nan")})
    assert not (tmp_path / "record.json").exists()


@pytest.mark.parametrize("status,text,match", [
    (403, "forbidden", "HTTP 403"),
    (200, "#TRILEGAL failed", "abnormally"),
])
def test_poll_refuses_http_denial_and_abnormal_termination(status, text, match):
    reply = response(text, status)
    session = SimpleNamespace(get=lambda *_args, **_kwargs: reply)
    with pytest.raises(trilegal.TrilegalUnavailableError, match=match):
        trilegal.poll_result(session, trilegal.FORM_URL, timeout=1, attempts=1, interval=0,
                             sleep=lambda _: None, check_shutdown=lambda: None, progress_callback=None)
    reply.close.assert_called_once()


def test_query_propagates_retryable_transport_failure_and_closes_session():
    class Session:
        closed = False

        def get(self, *_args, **_kwargs):
            raise requests.ConnectTimeout("provider is slow")

        def close(self):
            self.closed = True

    session = Session()
    with pytest.raises(requests.ConnectTimeout, match="provider is slow"):
        trilegal.query_trilegal(session, {}, timeout=1, poll=lambda *_: None)
    assert session.closed


def test_tls_transport_failure_never_attempts_aia(tmp_path, monkeypatch):
    class Session:
        def get(self, *_args, **_kwargs):
            raise requests.exceptions.SSLError("connection interrupted")

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _: Session())
    monkeypatch.setattr(trilegal, "create_aia_completed_session", lambda *_: pytest.fail("unexpected AIA"))
    with pytest.raises(trilegal.TrilegalUnavailableError, match="connection interrupted") as raised:
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    # A transport outage is typed and retryable for standalone callers; its cause is kept.
    assert raised.value.retryable and isinstance(raised.value.__cause__, requests.exceptions.SSLError)
    assert raised.value.to_manifest()["exception_class"] == "SSLError"
    # The stage is recorded as unavailable with the exact error, never as a population.
    record = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert record["status"] == "unavailable" and record["stage"] == "form-page"
    assert "connection interrupted" in record["error"]["message"]
    assert any(item.startswith("SSLError:") for item in record["error"]["diagnostic_chain"])
    assert not (tmp_path / "trilegal_cache.csv").exists() and not (tmp_path / "trilegal_output.dat").exists()


class _SlowSubmitter:
    """Serve the form, then let the submission time out."""

    def __init__(self):
        self.timeouts = []

    def get(self, url, **kwargs):
        return response(FORM)

    def post(self, url, timeout, **kwargs):
        self.timeouts.append(timeout)
        raise requests.ReadTimeout("submission held open by the provider")

    def close(self):
        pass


def test_a_slow_submission_uses_the_measured_timeout_and_retains_the_exact_error(tmp_path, monkeypatch):
    session = _SlowSubmitter()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: session)
    with pytest.raises(trilegal.TrilegalUnavailableError, match="submission held open") as raised:
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))

    # Sized from 2026-11-08: unanswered at 120 s and 300 s, so twice the longest non-answer.
    assert session.timeouts == [trilegal.SUBMISSION_TIMEOUT_SECONDS] and trilegal.SUBMISSION_TIMEOUT_SECONDS >= 600.0
    assert isinstance(raised.value.__cause__, requests.ReadTimeout) and raised.value.retryable
    record = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert record["status"] == "unavailable" and record["stage"] == "submission"
    assert "submission held open by the provider" in record["error"]["message"]
    assert record["error"]["retryable"] is True
    assert record["error"]["diagnostic_chain"][0].startswith("TrilegalUnavailableError:")
    assert record["error"]["diagnostic_chain"][1].startswith("ReadTimeout:")
    assert not (tmp_path / "trilegal_request.json").exists()  # no result URL was ever received


def test_a_submission_is_recorded_before_polling(tmp_path):
    class Session:
        def get(self, url, **kwargs):
            return response(FORM)

        def post(self, url, **kwargs):
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    request_path = tmp_path / trilegal.REQUEST_FILE_NAME
    seen = []

    def poll(_session, url):
        seen.append((url, json.loads(request_path.read_text())))
        return RAW

    submitted, url, text, _aia = trilegal.query_trilegal(
        Session(), trilegal.form_overrides(1.0, 2.0, 0.5), timeout=1, poll=poll,
        request_path=request_path, coordinates=(1.0, 2.0))

    assert len(seen) == 1 and seen[0][1]["status"] == "submitted"
    assert seen[0][0] == url and url.endswith("/tmp/output.dat") and seen[0][1]["ra_deg"] == 1.0
    assert [tuple(pair) for pair in seen[0][1]["submitted_form"]] == submitted and text == RAW


def test_a_polling_failure_is_resumed_from_its_result_url_without_resubmitting(tmp_path, monkeypatch):
    class Session:
        def __init__(self):
            self.gets, self.posts = [], []

        def get(self, url, **kwargs):
            self.gets.append(url)
            return response(FORM)

        def post(self, url, **kwargs):
            self.posts.append(url)
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Session())
    polled, stopped = [], []

    def poll(_session, url, **_kwargs):
        polled.append(url)
        if not stopped:
            stopped.append(url)
            raise trilegal.TrilegalUnavailableError("simulation did not terminate")
        return RAW

    monkeypatch.setattr(trilegal, "poll_result", poll)
    with pytest.raises(trilegal.TrilegalUnavailableError, match="did not terminate"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))

    request = json.loads((tmp_path / "trilegal_request.json").read_text())
    assert request["status"] == "submitted" and request["output_url"] == polled[0]
    unavailable = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert unavailable["status"] == "unavailable" and unavailable["stage"] == "polling"
    assert not (tmp_path / "trilegal_cache.csv").exists()

    resumed = Session()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: resumed)
    population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    # The rerun spent no form request and no second simulation, only the recorded result link.
    assert resumed.gets == [] and resumed.posts == [] and polled[-1] == request["output_url"]
    assert population.read_text() == "TESS,Mact,logg,logTe,[M/H]\n12,1,4.4,3.76,0.0\n"
    assert json.loads(query.read_text())["status"] == "available"
    assert json.loads((tmp_path / "trilegal_request.json").read_text())["status"] == "completed"


def test_a_timed_out_retained_job_keeps_its_identity_and_polls_again(tmp_path, monkeypatch, field_extinction):
    request_path = retained_request(tmp_path, field_extinction)
    # The provider answers 404 until the real polling budget runs out; only its
    # waits are removed, not the budget, the failure tag or the stage record.
    monkeypatch.setattr(trilegal.time, "sleep", lambda _seconds: None)

    class TimedOut:
        def __init__(self):
            self.gets = []

        def get(self, url, **kwargs):
            self.gets.append(url)
            return response("", 404)

        def post(self, url, **kwargs):
            pytest.fail("a retained job must be polled, not resubmitted")

        def close(self):
            pass

    timed_out = TimedOut()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: timed_out)
    with pytest.raises(trilegal.TrilegalUnavailableError, match="polling budget"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))

    assert set(timed_out.gets) == {RETAINED_URL}
    stage = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert stage["stage"] == "polling"
    assert stage["output_url"] == RETAINED_URL and stage["submitted_form"] == RETAINED_FORM
    assert json.loads(request_path.read_text())["status"] == "submitted"

    class Resumed:
        def __init__(self):
            self.gets, self.posts = [], 0

        def get(self, url, **kwargs):
            self.gets.append(url)
            return response(RAW)

        def post(self, url, **kwargs):
            self.posts += 1
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    resumed = Resumed()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: resumed)
    population, _query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    # The next attempt polls the recorded link and spends no form request or simulation.
    assert resumed.gets == [RETAINED_URL] and resumed.posts == 0
    assert population.read_text() == "TESS,Mact,logg,logTe,[M/H]\n12,1,4.4,3.76,0.0\n"


def test_a_failed_attempt_never_replaces_an_available_query_record(tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            return response(FORM if url == trilegal.FORM_URL else RAW)

        def post(self, url, **kwargs):
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Session())
    _population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    available = query.read_text()
    assert json.loads(available)["status"] == "available"
    assert not (tmp_path / trilegal.STAGE_RECORD_FILE_NAME).exists()

    def failed_poll(*_args, **_kwargs):
        raise trilegal.TrilegalUnavailableError("simulation did not terminate")

    monkeypatch.setattr(trilegal, "poll_result", failed_poll)
    with pytest.raises(trilegal.TrilegalUnavailableError, match="did not terminate"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))

    # The last good population keeps its own available, hash-bound record; the
    # failed attempt is an explicit unavailable stage record beside it.
    assert query.read_text() == available
    stage = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert stage["status"] == "unavailable" and stage["stage"] == "polling"


def test_a_recorded_result_url_from_another_host_is_not_resumed(tmp_path):
    request_path = tmp_path / trilegal.REQUEST_FILE_NAME
    request_path.write_text(json.dumps({
        "status": "submitted", "ra_deg": 1.0, "dec_deg": 2.0,
        "output_url": "https://foreign.example/tmp/output.dat", "field_submission": trilegal.FIELD_SUBMISSION,
        "submitted_form": [["gc_l", "96.3"]]}))

    class Session:
        posts = 0

        def get(self, url, **kwargs):
            return response(FORM)

        def post(self, url, **kwargs):
            self.posts += 1
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    session = Session()
    trilegal.query_trilegal(session, trilegal.form_overrides(1.0, 2.0, 0.5), timeout=1,
                            poll=lambda *_: RAW, request_path=request_path, coordinates=(1.0, 2.0))
    # A record that points elsewhere is not a resume state; the form is submitted normally.
    assert session.posts == 1


def test_an_abnormally_terminated_job_is_not_resumed(tmp_path, monkeypatch):
    class Session:
        def get(self, url, **kwargs):
            return response(FORM if url == trilegal.FORM_URL else "#TRILEGAL failed\n")

        def post(self, url, **kwargs):
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Session())
    with pytest.raises(trilegal.TrilegalUnavailableError, match="abnormally"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    # The provider's job is over; the next attempt submits again instead of polling a dead link.
    assert json.loads((tmp_path / "trilegal_request.json").read_text())["status"] == "completed"
    assert json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())["stage"] == "polling"


def test_an_abnormally_terminated_retained_job_keeps_its_identity_and_frees_a_new_submission(
        tmp_path, monkeypatch, field_extinction):
    request_path = retained_request(tmp_path, field_extinction)

    class Terminated:
        def get(self, url, **kwargs):
            assert url == RETAINED_URL
            return response("#TRILEGAL failed\n")

        def post(self, url, **kwargs):
            pytest.fail("a retained job must be polled, not resubmitted")

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Terminated())
    with pytest.raises(trilegal.TrilegalUnavailableError, match="abnormally"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))

    stage = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert stage["stage"] == "polling" and stage["output_url"] == RETAINED_URL
    assert stage["submitted_form"] == RETAINED_FORM
    # The finished job keeps its identity while the completed status frees the next attempt.
    completed = json.loads(request_path.read_text())
    assert completed["status"] == "completed" and completed["output_url"] == RETAINED_URL
    assert completed["submitted_form"] == RETAINED_FORM

    class Fresh:
        def __init__(self):
            self.posts = 0

        def get(self, url, **kwargs):
            return response(FORM if url == trilegal.FORM_URL else RAW)

        def post(self, url, **kwargs):
            self.posts += 1
            assert dict(kwargs["data"]) == expected_fields(field_extinction)
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    fresh = Fresh()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: fresh)
    _population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    assert fresh.posts == 1
    assert json.loads(query.read_text())["output_url"].endswith("/tmp/output.dat")


def test_malformed_output_leaves_an_unavailable_population_stage(tmp_path):
    def query_runner(_overrides, _field_extinction):
        return ([("eq_alpha", "1.0")], "https://stev.oapd.inaf.it/tmp/output.dat",
                "TESS Mact logg logTe [M/H]\n12 1 4 3 0\n", None)

    with pytest.raises(trilegal.TrilegalUnavailableError, match="termination"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True), query_runner=query_runner)

    record = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert record["status"] == "unavailable" and record["stage"] == "population"
    assert record["output_url"].endswith("/tmp/output.dat")
    # The provider's job finished; a rerun submits anew instead of polling malformed bytes.
    assert json.loads((tmp_path / trilegal.REQUEST_FILE_NAME).read_text())["status"] == "completed"
    assert not (tmp_path / "trilegal_cache.csv").exists() and not (tmp_path / "trilegal_output.dat").exists()


def test_malformed_population_writes_nothing(tmp_path, field_extinction):
    with pytest.raises(trilegal.TrilegalUnavailableError, match="termination"):
        trilegal.retain_trilegal(tmp_path, 1.0, 2.0, [], "https://example.org",
                                 "TESS Mact logg logTe [M/H]\n12 1 4 3 0\n", None, field_extinction=field_extinction)
    assert list(tmp_path.iterdir()) == []


def test_the_field_is_submitted_in_galactic_degrees_not_as_an_hour_angle():
    # TRILEGAL reads eq_alpha in hours; a right ascension in degrees there
    # simulates the field at (RA mod 24) x 15 deg. Galactic l, b are degrees by
    # definition, so the form uses gal_coord=1 and leaves eq_alpha untouched.
    pole = trilegal.form_overrides(*NORTH_GALACTIC_POLE, 0.5)
    assert pole["gal_coord"] == "1" and "eq_alpha" not in pole and "eq_delta" not in pole
    assert float(pole["gc_b"]) == pytest.approx(90.0, abs=1e-3)
    centre = trilegal.form_overrides(*GALACTIC_CENTRE, 0.5)
    assert float(centre["gc_b"]) == pytest.approx(0.0, abs=1e-3)
    assert min(float(centre["gc_l"]), 360.0 - float(centre["gc_l"])) == pytest.approx(0.0, abs=1e-3)
    # A high-latitude target whose RA misread as hours would land near the plane.
    _longitude, b = trilegal.galactic_field(352.172, -34.491)
    assert b == pytest.approx(-70.8, abs=0.1)
    assert trilegal.galactic_field((352.172 % 24) * 15, -34.491)[1] == pytest.approx(12.5, abs=0.1)


def test_records_carry_the_galactic_field_and_an_old_submission_is_not_resumed(tmp_path, monkeypatch,
                                                                               field_extinction):
    request_path = tmp_path / trilegal.REQUEST_FILE_NAME
    # A job submitted with the RA-in-hours form values simulates another field.
    request_path.write_text(json.dumps({
        "status": "submitted", "ra_deg": 1.0, "dec_deg": 2.0,
        "output_url": "https://stev.oapd.inaf.it/tmp/old.dat",
        "submitted_form": [["gal_coord", "2"], ["eq_alpha", "1.0"], ["eq_delta", "2.0"]]}))
    polled = []

    class Session:
        posts = 0

        def get(self, url, **kwargs):
            polled.append(url)
            return response(FORM if url == trilegal.FORM_URL else RAW)

        def post(self, url, **kwargs):
            self.posts += 1
            assert dict(kwargs["data"]) == expected_fields(field_extinction)
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    session = Session()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: session)
    _population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    assert session.posts == 1 and not any(url.endswith("/tmp/old.dat") for url in polled)
    record = json.loads(query.read_text())
    expected_l, expected_b = trilegal.galactic_field(1.0, 2.0)
    assert trilegal.is_galactic_field_record(record)
    assert (record["gc_l_deg"], record["gc_b_deg"]) == (expected_l, expected_b)
    assert trilegal.is_galactic_field_record(json.loads(request_path.read_text()))


def test_a_galactic_field_submission_is_resumed_without_resubmitting(tmp_path, field_extinction):
    request_path = tmp_path / trilegal.REQUEST_FILE_NAME
    request_path.write_text(json.dumps({
        "status": "submitted", "ra_deg": 1.0, "dec_deg": 2.0, "field_submission": trilegal.FIELD_SUBMISSION,
        "field_extinction": field_extinction,
        "output_url": "https://stev.oapd.inaf.it/tmp/running.dat",
        "submitted_form": [["gal_coord", "1"], ["gc_l", "96.3"], ["gc_b", "-58.8"],
                           ["extinction_infty", "0.5000"]]}))

    class Session:
        def get(self, url, **kwargs):
            raise AssertionError("a resumed job must not reload the form")

        def post(self, url, **kwargs):
            raise AssertionError("a resumed job must not be resubmitted")

        def close(self):
            pass

    submitted, url, raw, _ = trilegal.query_trilegal(
        Session(), trilegal.form_overrides(1.0, 2.0, 0.5), timeout=1, poll=lambda _s, _u: RAW,
        request_path=request_path, coordinates=(1.0, 2.0))
    assert url.endswith("/tmp/running.dat") and raw == RAW and ("gal_coord", "1") in submitted


def test_an_unavailable_field_extinction_defers_without_submitting(tmp_path, monkeypatch):
    def unreachable(*_args, **_kwargs):
        error = trilegal.TrilegalUnavailableError("field extinction map query failed: service down")
        error.trilegal_stage = "field-extinction"
        raise error

    monkeypatch.setattr(trilegal, "field_extinction_from_map", unreachable)

    class Session:
        def get(self, *_args, **_kwargs):
            raise AssertionError("no form page before the field extinction")

        def post(self, *_args, **_kwargs):
            raise AssertionError("no submission without the field extinction")

        def close(self):
            pass

    monkeypatch.setattr(trilegal, "create_session", lambda _policy: Session())
    with pytest.raises(trilegal.TrilegalUnavailableError, match="service down"):
        trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    record = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert record["status"] == "unavailable" and record["stage"] == "field-extinction"
    assert record["field_extinction"] is None
    assert not (tmp_path / "trilegal_cache.csv").exists()
    assert not (tmp_path / trilegal.REQUEST_FILE_NAME).exists()


def test_a_record_without_the_field_extinction_is_never_resumed_or_bound(tmp_path, monkeypatch, field_extinction):
    # A Galactic-field record written before this amendment submitted the
    # server's 0.0378 mag default for every field; it states no field value.
    request_path = tmp_path / trilegal.REQUEST_FILE_NAME
    request_path.write_text(json.dumps({
        "status": "submitted", "ra_deg": 1.0, "dec_deg": 2.0, "field_submission": trilegal.FIELD_SUBMISSION,
        "output_url": "https://stev.oapd.inaf.it/tmp/old.dat",
        "submitted_form": [["gal_coord", "1"], ["extinction_infty", "0.0378"]]}))
    assert not trilegal.is_galactic_field_record(json.loads(request_path.read_text()))

    polled = []

    class Session:
        posts = 0

        def get(self, url, **kwargs):
            polled.append(url)
            return response(FORM if url == trilegal.FORM_URL else RAW)

        def post(self, url, **kwargs):
            self.posts += 1
            assert dict(kwargs["data"])["extinction_infty"] == "0.5000"
            return response(SUBMITTED_PAGE)

        def close(self):
            pass

    session = Session()
    monkeypatch.setattr(trilegal, "create_session", lambda _policy: session)
    _population, query = trilegal.acquire_trilegal(1.0, 2.0, tmp_path, NetworkPolicy(True))
    assert session.posts == 1 and not any(url.endswith("/tmp/old.dat") for url in polled)
    assert trilegal.is_galactic_field_record(json.loads(query.read_text()))

    # A record whose form and stated field value disagree is not bindable either.
    record = json.loads(query.read_text())
    record["submitted_form"] = [[name, "0.5001" if name == "extinction_infty" else value]
                                for name, value in record["submitted_form"]]
    assert not trilegal.is_galactic_field_record(record)


def test_the_field_attenuation_envelope_reads_both_record_kinds(field_extinction):
    from trex.acquisition.background_population import field_attenuation_envelope

    web = {"model": "trilegal", "field_extinction": field_extinction,
           "submitted_form": [["extinction_infty", "0.5000"]]}
    assert field_attenuation_envelope(web) == 0.5
    assert field_attenuation_envelope({"model": "trilegal-lsst-sim-dr2", "field_av_max": 4.755}) == 4.755
    with pytest.raises(ValueError, match="map-derived"):
        field_attenuation_envelope({"model": "trilegal", "submitted_form": [["extinction_infty", "0.0378"]]})


def test_roman_form_submits_the_documented_roman2024_system(field_extinction):
    values = trilegal.roman_form_overrides(
        1.0, 2.0, field_extinction["av_infinity_mag"], magnitude_limit=24.0)
    assert values["photsys_file"] == "tab_mag_odfnew/tab_mag_Roman2024.dat"
    assert values["mag_lim"] == "24"
    assert values["field"] == "0.1"
    assert values["gc_l"] == trilegal.form_overrides(
        1.0, 2.0, field_extinction["av_infinity_mag"])["gc_l"]
    with pytest.raises(ValueError, match="magnitude limit"):
        trilegal.roman_form_overrides(1.0, 2.0, 0.5, magnitude_limit=0.0)
    with pytest.raises(ValueError, match="magnitude limit"):
        trilegal.roman_form_overrides(1.0, 2.0, 0.5, magnitude_limit=33.0)


# Parser and form rules owned by trex.acquisition.
PARSER_FORM_PAGE = """<html><body>
<form method="post" action="trilegal_1.6">
<input type="text" name="gal_coord" value="1">
<input type="submit" name="submit_form" value="Submit" > <input type="submit" name="reset_form" value="Reset" >
<textarea name="notes">kept default</textarea>
<input type="submit" name="submit_form" value="Submit" > <input type="submit" name="reset_form" value="Reset" >
</form></body></html>"""
PARSER_OUTPUT = (
    "#Gc logAge [M/H] m_ini   logL   logTe  logg  m-M0   Av    m2/m1 mbol   J      H      Ks     TESS   Mact\n"
    " 1  9.10  -0.10  0.812  -0.310  3.720  4.550  9.50  0.040  0.00  5.525  14.210 13.800 13.720 14.650 0.811\n"
    " 2  9.80   0.05  1.150   1.420  3.650  2.900 11.20  0.050  0.00  1.200  12.100 11.450 11.300 12.950 1.149\n"
    "#TRILEGAL normally terminated\n"
)


def test_only_the_first_submit_control_is_sent_like_mechanicalsoup():
    from trex.acquisition.trilegal_forms import submitted_form

    _action, submitted, enctype = submitted_form(PARSER_FORM_PAGE, {"gal_coord": "2"})
    assert enctype == "application/x-www-form-urlencoded"
    names = [name for name, _value in submitted]
    assert names.count("submit_form") == 1
    assert "reset_form" not in names
    assert dict(submitted)["gal_coord"] == "2"


@pytest.mark.parametrize(
    "text,message",
    [
        (PARSER_OUTPUT.replace("#TRILEGAL normally terminated\n", ""), "normal termination footer"),
        (PARSER_OUTPUT.replace(" TESS ", " T_mag "), "lacks required column"),
        (PARSER_OUTPUT.replace(" 0.811\n", "\n"), "different column count"),
        (PARSER_OUTPUT.replace("0.811", "nan"), "non-finite Mact"),
        (PARSER_OUTPUT.replace("14.650", "bright"), "non-numeric TESS"),
        (PARSER_OUTPUT.splitlines()[0] + "\n#TRILEGAL normally terminated\n", "no simulated stars"),
        (PARSER_OUTPUT + " 3 9.0 0.0 1.0 0.0 3.7 4.5 9.0 0.0 0.0 5.0 14.0 13.0 13.0 14.0 1.0\n", "after its termination"),
        (PARSER_OUTPUT.replace("#TRILEGAL normally terminated", "#TRILEGAL synthetic abnormal footer"),
         "terminated abnormally"),
    ],
)
def test_malformed_output_never_becomes_a_population(text, message):
    from trex.acquisition.trilegal_output import trilegal_output_to_csv

    with pytest.raises(trilegal.TrilegalUnavailableError, match=message):
        trilegal_output_to_csv(text)
