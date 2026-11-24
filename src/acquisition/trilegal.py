"""Acquire and retain a TRILEGAL 1.6 background population."""

import hashlib
import json
import math
import os
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from trex.runtime.logging import host_logger

from .dust_map import field_extinction_from_map, validate_icrs_field
from .storage import write_bytes
from .transport import AcquisitionFailure
from .trilegal_errors import TrilegalUnavailableError
from .trilegal_forms import LinkParser, submitted_form
from .trilegal_output import TRILEGAL_NORMAL_TERMINATION, trilegal_output_to_csv
from .trilegal_tls import create_aia_completed_session, create_session

FORM_URL = "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6"
# TRILEGAL holds the submission open while the simulation queues. Live
# submissions have stayed unanswered past both 120 s and 300 s, so the POST
# waits twice the longest observed non-answer; page reads keep the caller's
# short socket timeout.
SUBMISSION_TIMEOUT_SECONDS = 600.0
REQUEST_FILE_NAME = "trilegal_request.json"
# A failed attempt is recorded beside, never over, a retained successful query:
# the stage is explicitly unavailable while the last good population keeps its
# own hash-bound record and provenance.
STAGE_RECORD_FILE_NAME = "trilegal_stage.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _http_unavailable(response, what):
    """A TRILEGAL HTTP error; 429 and 5xx are retryable provider states, other codes are not."""
    status = int(response.status_code)
    retryable = status == 429 or 500 <= status < 600
    reason = "{0} returned HTTP {1}".format(what, status)
    failure = AcquisitionFailure(urlparse(FORM_URL).hostname, "http_status", retryable, 1, reason, status,
                                 "HTTPError")
    return TrilegalUnavailableError("TRILEGAL " + reason, failure)


def _response_text(response, limit, what):
    """Read one bounded streamed provider reply and always release its connection."""
    try:
        if 300 <= response.status_code < 400:
            raise TrilegalUnavailableError("TRILEGAL redirects are not permitted")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise _http_unavailable(response, what) from exc
        payload = bytearray()
        for block in response.iter_content(65536):
            if len(payload) + len(block) > limit:
                raise TrilegalUnavailableError("TRILEGAL response exceeds the network policy size bound")
            payload.extend(block)
        return payload.decode(response.encoding or "utf-8")
    except UnicodeError as exc:
        raise TrilegalUnavailableError("TRILEGAL response has invalid text encoding") from exc
    finally:
        response.close()


def _certificate_failure(exc):
    """Whether a TLS error is a certificate trust failure AIA completion may repair.

    Hostname mismatches and self-signed chains are included so they reach the
    AIA verifier, which then refuses them explicitly instead of the caller
    guessing from an unverified chain.
    """
    for _ in range(8):
        if exc is None:
            break
        if isinstance(exc, ssl.SSLCertVerificationError) or any(
            marker in str(exc).lower() for marker in
            ("certificate", "hostname mismatch", "unknown ca", "self signed", "self-signed")
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# The field is submitted in Galactic coordinates. TRICERATOPS 1.0.20
# (funcs.query_TRILEGAL) sends ``gal_coord=2`` with the right ascension in
# degrees as ``eq_alpha``, which TRILEGAL reads in hours (modulo 24): the
# simulated field lies at RA' = (RA mod 24) x 15 deg instead of the target.
# Star counts of populations submitted that way match Gaia DR3 counts of the
# shifted field, not of the target (10-40x apart). Galactic longitude and
# latitude are degrees by definition, so ``gal_coord=1`` with ``gc_l``/``gc_b``
# leaves no unit choice; VESPA submits the field the same way (Morton et al.
# 2016, ApJ 822, 86).
# Records carry FIELD_SUBMISSION, and a population without it is never reused.
FIELD_SUBMISSION = "galactic-lb-deg"
_FIELD_EXTINCTION_FORM_FORMAT = ".4f"


def galactic_field(ra_deg, dec_deg):
    """Return the Galactic (l, b) in degrees of an ICRS position given in degrees."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    galactic = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    return float(galactic.l.deg), float(galactic.b.deg)


def is_galactic_field_record(record):
    """Return whether a TRILEGAL record was submitted at the target's field with its extinction.

    A record written before the field-extinction amendment has no
    ``field_extinction``: its form carried the server's 0.0378 mag default for
    every field, so it is never resumed, reused or bound.  The submitted form's
    ``extinction_infty`` must equal the recorded map value rounded as it was
    for submission. Subtracting the two floats can wrongly reject the exact
    form value at a decimal midpoint.
    """
    if not isinstance(record, dict) or record.get("field_submission") != FIELD_SUBMISSION:
        return False
    extinction = record.get("field_extinction")
    if not isinstance(extinction, dict) or extinction.get("status") != "available":
        return False
    try:
        stated = float(extinction["av_infinity_mag"])
        submitted = float(dict(record["submitted_form"])["extinction_infty"])
    except (KeyError, TypeError, ValueError):
        return False
    return (math.isfinite(stated) and stated >= 0.0
            and submitted == float(format(stated, _FIELD_EXTINCTION_FORM_FORMAT)))


def _field_fields(ra_deg, dec_deg):
    if ra_deg is None or dec_deg is None:
        return {}
    gc_l, gc_b = galactic_field(ra_deg, dec_deg)
    return {"field_submission": FIELD_SUBMISSION, "gc_l_deg": gc_l, "gc_b_deg": gc_b}


def form_overrides(ra_deg, dec_deg, extinction_infty):
    """Return the form values for the 0.1 square degree field centred on the target.

    Field size, photometric system, magnitude limit and binary setting follow
    TRICERATOPS 1.0.20; the centre is submitted in Galactic degrees (see
    FIELD_SUBMISSION).  ``extinction_infty`` is the field's total V-band
    extinction at infinity from the dust map (``trex.acquisition.dust_map``),
    the value TRILEGAL uses to normalize its disc extinction layer; the
    server's 0.0378 mag default is never submitted.
    """
    from trex.populations._validation import real_scalar

    value = real_scalar(extinction_infty, "field extinction")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("field extinction must be finite and nonnegative")
    gc_l, gc_b = galactic_field(ra_deg, dec_deg)
    return {"gal_coord": "1", "gc_l": "{0:.6f}".format(gc_l), "gc_b": "{0:.6f}".format(gc_b),
            "field": "0.1", "photsys_file": TRILEGAL_TESS_PHOTSYS_FILE,
            "icm_lim": "1", "mag_lim": "21", "binary_kind": "0",
            "extinction_infty": format(value, _FIELD_EXTINCTION_FORM_FORMAT)}


# TRILEGAL 1.6 (https://stev.oapd.inaf.it/trilegal, retrieved 2026-11-16)
# offers the documented Roman2024 photometric system: all WFI filters, Vega,
# mean of all detectors from the Roman technical information effective areas
# (March 2024), per the provider's systems page
# https://stev.oapd.inaf.it/cmd_3.9/photsys.html (retrieved 2026-11-16).
# The system nickname follows the provider's tab_mag_odfnew convention.
# TRILEGAL_ROMAN_COLUMNS is the expected WFI filter set (wavelength order); no
# acquisition path checks a returned header against it yet, so the first live
# Roman population must compare its returned columns before any use.
TRILEGAL_TESS_PHOTSYS_FILE = "tab_mag_odfnew/tab_mag_TESS_2mass.dat"
TRILEGAL_ROMAN_SYSTEM = "Roman2024"
TRILEGAL_ROMAN_PHOTSYS_FILE = "tab_mag_odfnew/tab_mag_Roman2024.dat"
TRILEGAL_ROMAN_COLUMNS = ("F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213")


def roman_form_overrides(ra_deg, dec_deg, extinction_infty, *, magnitude_limit):
    """TRILEGAL form values for a Roman/WFI-native background population.

    ``magnitude_limit`` is the simulation's own faint limit in the Roman
    system [mag], declared by the caller because it is not the target's
    selection depth and must be measured or chosen for the field. The field,
    extinction calibration and binary setting match ``form_overrides``.
    """
    from trex.populations._validation import real_scalar

    limit = real_scalar(magnitude_limit, "TRILEGAL Roman magnitude limit")
    if not 0.0 < limit <= 32.0:
        raise ValueError("TRILEGAL Roman magnitude limit must be in (0, 32] mag")
    values = form_overrides(ra_deg, dec_deg, extinction_infty)
    values["photsys_file"] = TRILEGAL_ROMAN_PHOTSYS_FILE
    values["mag_lim"] = "{0:g}".format(limit)
    return values


def result_url(form_url, page):
    """Select a result link from the provider, never a foreign host."""
    parser = LinkParser()
    parser.feed(page)
    parser.close()
    if not parser.hrefs:
        detail = "response reports server busy" if "server busy" in page.lower() else "cause unknown"
        raise TrilegalUnavailableError("TRILEGAL did not return a result link ({0})".format(detail))
    host = urlparse(form_url).netloc
    for href in parser.hrefs:
        url = urljoin(form_url, href)
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.netloc == host and "/tmp/" in parsed.path:
            return url
    raise TrilegalUnavailableError(
        "TRILEGAL returned an unexpected result link: {0} (page links: {1})".format(
            urljoin(form_url, parser.hrefs[0]), [urljoin(form_url, href) for href in parser.hrefs]))


def poll_result(session, url, *, timeout, attempts, interval, sleep, check_shutdown, progress_callback,
                max_response_bytes=100_000_000):
    """Poll a result with bounded attempts, reporting progress without inventing completion."""
    from trex.populations.background import TRILEGAL_FOOTER_MARKER
    from trex.runtime.progress import emit_progress

    started = time.monotonic()
    for attempt in range(attempts):
        check_shutdown()
        emit_progress(progress_callback, step="TRILEGAL simulation", done=attempt, total=attempts,
                      sub_phase="polling", note="attempt {0}/{1}, elapsed {2:.0f}s".format(
                          attempt + 1, attempts, time.monotonic() - started))
        response = session.get(url, timeout=timeout, stream=True, allow_redirects=False)
        if response.status_code != 404:
            text = _response_text(response, max_response_bytes, "result download")
            last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
            if last.startswith(TRILEGAL_NORMAL_TERMINATION):
                emit_progress(progress_callback, step="TRILEGAL simulation", done=attempts, total=attempts,
                              sub_phase="complete", note="terminated normally after {0:.0f}s".format(
                                  time.monotonic() - started))
                return text
            if last.startswith(TRILEGAL_FOOTER_MARKER):
                error = TrilegalUnavailableError("TRILEGAL terminated abnormally: {0}".format(last))
                # The provider's job is over, so a rerun must submit anew rather
                # than poll this terminated result link.
                error.trilegal_job_finished = True
                raise error
        else:
            response.close()
        if attempt + 1 < attempts:
            sleep(interval)
    raise TrilegalUnavailableError(
        "TRILEGAL simulation did not terminate within the polling budget of {0} attempts ({1:.0f}s elapsed)".format(
            attempts, time.monotonic() - started), retryable=True)


def _write_record(path, writer, payload):
    writer(Path(path), (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def _submission_record(ra_deg, dec_deg, submitted, output_url, *, status="submitted", field_extinction=None):
    """Describe a submitted job so a rerun can poll it instead of resubmitting."""
    return {"status": status, "schema_version": 1, "model": "trilegal", "trilegal_version": "1.6",
            "form_url": FORM_URL, "output_url": output_url, "recorded_utc": _utc_now(),
            "ra_deg": ra_deg, "dec_deg": dec_deg, "simulated_area_sq_deg": 0.1,
            **_field_fields(ra_deg, dec_deg), "field_extinction": field_extinction,
            "submitted_form": [[name, value] for name, value in submitted]}


def _retained_submission(path, ra_deg, dec_deg):
    """Return a recorded submission for this field, when its result link is still pollable."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("status") != "submitted":
        return None
    if not is_galactic_field_record(record):
        # Submitted before the Galactic-field or field-extinction amendment: the
        # simulated field or its extinction is not this field's.
        return None
    if record.get("ra_deg") != ra_deg or record.get("dec_deg") != dec_deg:
        return None
    url = record.get("output_url")
    if not isinstance(url, str):
        return None
    form = record.get("submitted_form")
    if not isinstance(form, list) or not all(isinstance(item, list) and len(item) == 2 for item in form):
        return None
    url = urljoin(FORM_URL, url)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != urlparse(FORM_URL).netloc or "/tmp/" not in parsed.path:
        return None
    return {**record, "output_url": url}


def _diagnostic_chain(exc, limit=8):
    """Return the exact exception chain, outermost first, without interpreting it."""
    chain = []
    for _ in range(limit):
        if exc is None:
            break
        chain.append("{0}: {1}".format(type(exc).__name__, exc))
        exc = exc.__cause__ or exc.__context__
    return chain


def _failure_record(ra_deg, dec_deg, exc, *, stage, submitted=(), output_url=None, candidate_id=None,
                    field_extinction=None):
    """Describe an unavailable TRILEGAL stage with the exact error that stopped it."""
    record = {"status": "unavailable", "schema_version": 1, "model": "trilegal", "trilegal_version": "1.6",
              "form_url": FORM_URL, "stage": stage, "recorded_utc": _utc_now(),
              "ra_deg": ra_deg, "dec_deg": dec_deg, "simulated_area_sq_deg": 0.1,
              **_field_fields(ra_deg, dec_deg), "field_extinction": field_extinction,
              "output_url": output_url, "submitted_form": [[name, value] for name, value in submitted],
              "error": {"exception_class": type(exc).__name__, "message": str(exc).strip(),
                        "diagnostic_chain": _diagnostic_chain(exc),
                        "retryable": bool(getattr(exc, "retryable", isinstance(exc, requests.RequestException)))}}
    if isinstance(exc, TrilegalUnavailableError):
        record["error"]["failure"] = exc.to_manifest()
    elif getattr(exc, "failure", None) is not None:
        record["error"]["failure"] = exc.failure.to_manifest()
    if candidate_id is not None:
        record.update(candidate_id=candidate_id, source="candidate-data")
    return record


def _tag_failure(exc, stage, submitted, output_url):
    """Attach the query phase to a failure so the caller's retained record names it."""
    exc.trilegal_stage = stage
    exc.trilegal_submitted = list(submitted)
    exc.trilegal_output_url = output_url


def query_trilegal(session, overrides, *, timeout, poll, post_timeout=SUBMISSION_TIMEOUT_SECONDS,
                   request_path=None, writer=write_bytes, coordinates=None, field_extinction=None,
                   max_response_bytes=100_000_000):
    """Submit the provider's checked form and poll with the same session.

    A submission is recorded with its result URL before polling, so a rerun for
    the same field polls the running job instead of spending a second one.
    """
    ra_deg, dec_deg = coordinates if coordinates is not None else (None, None)
    submitted, output_url, stage = [], None, "form-page"
    try:
        retained = None if request_path is None else _retained_submission(request_path, ra_deg, dec_deg)
        if retained is not None:
            # Name the recorded job before polling: a failure here must keep its
            # exact result URL and form in the stage record and the completed
            # request record, not look like an unidentified query.
            submitted = [tuple(item) for item in retained["submitted_form"]]
            output_url = retained["output_url"]
            stage = "polling"
            text = poll(session, output_url)
            return submitted, output_url, text, getattr(session, "exonym_aia_intermediate", None)
        page = session.get(FORM_URL, timeout=timeout, stream=True, allow_redirects=False)
        page_text = _response_text(page, max_response_bytes, "form page")
        stage = "form"
        action, submitted, encoding = submitted_form(page_text, overrides)
        url = urljoin(FORM_URL, action) if action else FORM_URL
        if urlparse(url).scheme != "https" or urlparse(url).netloc != urlparse(FORM_URL).netloc:
            raise TrilegalUnavailableError("TRILEGAL form has an unexpected submission URL")
        kwargs = {"files": [(name, (None, value)) for name, value in submitted]} if encoding == "multipart/form-data" else {"data": submitted}
        stage = "submission"
        response = session.post(url, timeout=post_timeout, stream=True, allow_redirects=False, **kwargs)
        response_text = _response_text(response, max_response_bytes, "form submission")
        stage = "result-link"
        output_url = result_url(url, response_text)
        if request_path is not None:
            _write_record(request_path, writer, _submission_record(
                ra_deg, dec_deg, submitted, output_url, field_extinction=field_extinction))
        stage = "polling"
        host_logger(__name__).info("TRILEGAL job submitted; polling its result")
        text = poll(session, output_url)
        return submitted, output_url, text, getattr(session, "exonym_aia_intermediate", None)
    except Exception as exc:
        _tag_failure(exc, stage, submitted, output_url)
        raise
    finally:
        session.close()


def retain_trilegal(output_dir, ra_deg, dec_deg, submitted, output_url, raw, aia, *,
                    field_extinction, candidate_id=None, record_prefix=Path(), writer=write_bytes):
    """Write validated raw data, population and request record at the caller's boundary.

    ``field_extinction`` is the dust-map record the submitted
    ``extinction_infty`` came from; a population without it is not retainable
    because its field extinction would be unknown.
    """
    if not isinstance(field_extinction, dict) or field_extinction.get("status") != "available":
        raise ValueError("a TRILEGAL population requires its field extinction record")
    csv, count = trilegal_output_to_csv(raw)
    output = Path(output_dir)
    raw_bytes, csv_bytes = raw.encode("utf-8"), csv.encode("utf-8")
    record = {"status": "available", "schema_version": 1, "model": "trilegal", "trilegal_version": "1.6", "form_url": FORM_URL,
              "output_url": output_url, "retrieved_utc": _utc_now(),
              "ra_deg": ra_deg, "dec_deg": dec_deg, "simulated_area_sq_deg": 0.1,
              **_field_fields(ra_deg, dec_deg), "field_extinction": field_extinction,
              "simulated_area_reference": "Giacalone et al. (2021), AJ 161, 24, Sec. 2.3; DOI 10.3847/1538-3881/abc6af",
              "query_parameter_reference": (
                  "TRICERATOPS 1.0.20 funcs.query_TRILEGAL field size, photometric system, magnitude limit and "
                  "binary setting (Giacalone et al. 2021 software); field centre in Galactic degrees as in VESPA "
                  "(Morton et al. 2016) because TRILEGAL reads eq_alpha in hours; extinction_infty from the "
                  "field's SFD98 map value instead of the server default"),
              "submitted_form": [[name, value] for name, value in submitted],
              "population_file": (record_prefix / "trilegal_cache.csv").as_posix(),
              "population_sha256": hashlib.sha256(csv_bytes).hexdigest(),
              "raw_output_file": (record_prefix / "trilegal_output.dat").as_posix(),
              "raw_output_sha256": hashlib.sha256(raw_bytes).hexdigest(), "star_count": count}
    if not is_galactic_field_record(record):
        raise ValueError("the submitted form does not carry the retained field extinction")
    if candidate_id is not None:
        record.update(candidate_id=candidate_id, source="candidate-data")
    if aia is not None:
        record["tls_trust"] = {"ca_source": "certifi+aia-verified-intermediate", "aia_intermediate": aia}
    writer(output / "trilegal_output.dat", raw_bytes)
    writer(output / "trilegal_cache.csv", csv_bytes)
    _write_record(output / "trilegal_query.json", writer, record)
    return output / "trilegal_cache.csv", output / "trilegal_query.json"


def acquire_trilegal(ra_deg, dec_deg, output_dir, policy, *, query_runner=None, field_extinction=None,
                     candidate_id=None, record_prefix=Path(), writer=write_bytes):
    """Retain raw output, CSV and request provenance, or raise on incomplete evidence.

    The submitted ``extinction_infty`` comes from the field's dust-map value
    (``trex.acquisition.dust_map``) and the same record is retained with the
    query.  ``field_extinction`` supplies it directly for a caller that already
    holds the record; otherwise a resumable submission's own record is reused,
    and a new field queries the map.  When the map is unavailable the stage
    stays explicitly unavailable: the form is never submitted at the server's
    0.0378 mag default.
    """
    if not policy.allow_network:
        raise TrilegalUnavailableError("network access is disabled by policy")
    validate_icrs_field(ra_deg, dec_deg)
    output = Path(output_dir)
    request_path = output / REQUEST_FILE_NAME
    overrides = None

    def query(session):
        return query_trilegal(
            session, overrides, timeout=policy.timeout_seconds,
            poll=lambda client, url: poll_result(client, url, timeout=policy.timeout_seconds,
                                                  attempts=90, interval=10, sleep=time.sleep,
                                                  check_shutdown=lambda: None, progress_callback=None,
                                                  max_response_bytes=policy.max_response_bytes),
            request_path=request_path, writer=writer, coordinates=(ra_deg, dec_deg),
            field_extinction=field_extinction, max_response_bytes=policy.max_response_bytes)

    def retain_failure(exc, stage, submitted, output_url):
        host_logger(__name__).warning("TRILEGAL stage %s failed: %s (retryable=%s)", stage, type(exc).__name__,
                                      bool(getattr(exc, "retryable", False)))
        _write_record(output / STAGE_RECORD_FILE_NAME, writer,
                      _failure_record(ra_deg, dec_deg, exc, stage=stage, submitted=submitted,
                                      output_url=output_url, candidate_id=candidate_id,
                                      field_extinction=field_extinction))

    try:
        if field_extinction is None:
            retained = _retained_submission(request_path, ra_deg, dec_deg)
            field_extinction = (retained["field_extinction"] if retained is not None else
                                field_extinction_from_map(ra_deg, dec_deg, policy, writer=writer,
                                                          output_dir=output_dir, record_prefix=record_prefix))
        overrides = form_overrides(ra_deg, dec_deg, field_extinction["av_infinity_mag"])
        if query_runner is not None:
            submitted, url, raw, aia = query_runner(overrides, field_extinction)
        else:
            try:
                try:
                    submitted, url, raw, aia = query(create_session(policy))
                except requests.exceptions.SSLError as exc:
                    if not _certificate_failure(exc):
                        raise
                    session = create_aia_completed_session(urlparse(FORM_URL).hostname, policy)
                    try:
                        submitted, url, raw, aia = query(session)
                    finally:
                        os.unlink(session.exonym_ca_bundle_path)
            except requests.RequestException as exc:
                # Standalone callers get a typed, retryable outage rather than a raw transport exception.
                reason = "{0}: {1}".format(type(exc).__name__, exc)
                error = TrilegalUnavailableError("TRILEGAL request failed: " + reason, AcquisitionFailure(
                    urlparse(FORM_URL).hostname, "transport", True, 1, reason, None, type(exc).__name__))
                _tag_failure(error, getattr(exc, "trilegal_stage", "query"),
                             getattr(exc, "trilegal_submitted", ()), getattr(exc, "trilegal_output_url", None))
                raise error from exc
    except Exception as exc:
        retain_failure(exc, getattr(exc, "trilegal_stage", "query"),
                       getattr(exc, "trilegal_submitted", ()), getattr(exc, "trilegal_output_url", None))
        if getattr(exc, "trilegal_job_finished", False) and request_path.is_file():
            # The job is over; a completed request record keeps a rerun from polling it again.
            _write_record(request_path, writer, _submission_record(
                ra_deg, dec_deg, getattr(exc, "trilegal_submitted", ()),
                getattr(exc, "trilegal_output_url", None), status="completed",
                field_extinction=field_extinction))
        raise
    _write_record(request_path, writer, _submission_record(ra_deg, dec_deg, submitted, url, status="completed",
                                                           field_extinction=field_extinction))
    try:
        population, query = retain_trilegal(output_dir, ra_deg, dec_deg, submitted, url, raw, aia,
                                            field_extinction=field_extinction,
                                            candidate_id=candidate_id, record_prefix=record_prefix, writer=writer)
    except Exception as exc:
        retain_failure(exc, "population", submitted, url)
        raise
    stage_path = output / STAGE_RECORD_FILE_NAME
    if stage_path.is_file():
        stage_path.unlink()
    return population, query
