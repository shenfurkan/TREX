"""Bounded, verified acquisition transport with explicit retry policy."""

import errno
import json
import math
import os
import socket
import ssl
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from trex.runtime.logging import host_logger
from .storage import _sync_directory, write_bytes


def _warn(message, *args):
    # Resolved per call: an embedding package sets its log namespace after import.
    host_logger(__name__).warning(message, *args)


@dataclass(frozen=True)
class NetworkPolicy:
    """Network access, timeout and bounded backoff for archive requests."""

    allow_network: bool
    attempts: int = 3
    timeout_seconds: float = 30.0
    backoff_seconds: float = 1.0
    max_response_bytes: int = 100_000_000
    max_download_bytes: int = 2_000_000_000
    # Caller-supplied User-Agent source, called once per request (or session),
    # which must return a non-empty single-line string. Leaving the field None
    # keeps TREX's own client identifier; an embedding application can pass
    # its own outbound policy without TREX importing it.
    user_agent: Optional[Callable[[], str]] = None

    def __post_init__(self):
        if not isinstance(self.allow_network, bool):
            raise ValueError("allow_network must be boolean")
        if self.user_agent is not None and not callable(self.user_agent):
            raise ValueError("user_agent must be a callable returning a User-Agent string or None")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 1:
            raise ValueError("attempts must be a positive integer")
        for name in ("timeout_seconds", "backoff_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number of seconds")
        if not self.timeout_seconds > 0 or not self.backoff_seconds >= 0:
            raise ValueError("timeout must be positive and backoff nonnegative, both finite")
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(self.max_download_bytes, bool) or not isinstance(self.max_download_bytes, int) or self.max_download_bytes < 1:
            raise ValueError("max_download_bytes must be a positive integer")


DEFAULT_USER_AGENT = "trex-acquisition/1"


def request_user_agent(policy):
    """Return the User-Agent for one request under ``policy``."""
    if policy.user_agent is None:
        return DEFAULT_USER_AGENT
    value = policy.user_agent()
    if not isinstance(value, str) or not value.strip() or any(ch in value for ch in "\r\n"):
        raise ValueError("user_agent must return a non-empty single-line string")
    return value


def archive_request(url, *, headers, data=None):
    """Build a request; long TAP-sync queries use an equivalent form POST.

    TAP 1.1 (2019), section 2.1, supports POST for synchronous queries.
    The 2 KiB switch avoids intermediary URI limits, not a scientific row
    limit. Preserve the encoded parameters verbatim and leave other GETs
    and explicit POST bodies unchanged.
    """
    if data is None and len(url) > 2048:
        parsed = urlsplit(url)
        parameters = {key.upper(): value for key, value in parse_qs(parsed.query).items()}
        if (parsed.path.rstrip("/").endswith("/sync")
                and parameters.get("REQUEST", [""])[0].lower() == "doquery"
                and "QUERY" in parameters):
            data = parsed.query.encode("ascii")
            url = parsed._replace(query="", fragment="").geturl()
            headers = {**headers, "Content-Type": "application/x-www-form-urlencoded"}
    return Request(url, data=data, headers=headers)


def fetch_tap_async(service, sync_url, query, *, deadline, maxrec, user_agent=None):
    """Run the same read-only TAP query through UWS within the caller's deadline."""
    import pyvo
    import requests

    policy = getattr(service, "policy", None)
    if policy is not None and not policy.allow_network:
        raise AcquisitionError("network access is disabled by policy")
    timeout = policy.timeout_seconds if policy is not None else service.timeout
    max_bytes = policy.max_response_bytes if policy is not None else service.MAX_JSON_RESPONSE_BYTES
    try:
        with requests.Session() as session:
            request = session.request

            def remaining():
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise requests.Timeout("TAP asynchronous query reached its deadline")
                return seconds

            def bounded_request(*args, **kwargs):
                kwargs["timeout"] = min(timeout, remaining(), kwargs.get("timeout") or timeout)
                return request(*args, **kwargs)

            session.request = bounded_request
            if policy is not None:
                session.headers["User-Agent"] = request_user_agent(policy)
            elif user_agent is not None:
                session.headers["User-Agent"] = user_agent
            tap = pyvo.dal.TAPService(sync_url.removesuffix("/sync"), session=session)
            with tap.submit_job(query, maxrec=maxrec, FORMAT="votable") as job:
                job.run()
                phase = job.phase
                while phase not in {"COMPLETED", "ERROR", "ABORTED"}:
                    # Short status reads avoid the sync query's long wait for data.
                    sleep_before_retry(min(1.0, remaining()))
                    phase = job.phase
                if phase != "COMPLETED":
                    job.raise_if_error()
                    raise AcquisitionError(f"TAP asynchronous query ended in phase {phase}")
                uri = job.result_uri
                if not uri or urlsplit(uri).scheme != "https":
                    raise AcquisitionError("TAP asynchronous query has no HTTPS result URI")
                with session.get(uri, stream=True, headers={"Accept": "application/xml"}) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=65536):
                        remaining()
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise AcquisitionError("TAP asynchronous response exceeds the configured size limit")
                remaining()
                return bytes(body), uri
    except (requests.RequestException, pyvo.dal.DALAccessError, ValueError) as exc:
        raise AcquisitionError(f"TAP asynchronous query failed: {type(exc).__name__}: {exc}") from exc


@dataclass(frozen=True)
class AcquisitionFailure:
    """Reason an archive did not supply a complete response."""

    endpoint: str
    category: str
    retryable: bool
    attempts: int
    reason: str
    http_status: int | None = None
    exception_class: str = ""

    def to_manifest(self):
        result = {"status": "deferred" if self.category == "rate_limit" else "unavailable",
                  "category": self.category, "event": "network." + self.category, "endpoint": self.endpoint,
                  "exception_class": self.exception_class, "retryable": self.retryable,
                  "attempts": self.attempts, "message": self.reason, "diagnostic_chain": [self.reason]}
        if self.http_status is not None:
            result["http_status"] = self.http_status
        return result


class AcquisitionError(RuntimeError):
    """A scene could not be acquired without inventing missing evidence."""

    def __init__(self, reason, *, retryable=False):
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


def sleep_before_retry(seconds):
    """Wait through the single acquisition backoff seam."""
    time.sleep(seconds)


def mast_error_reason(payload):
    """The provider's reason when a MAST JSON payload reports ``status: ERROR``, else None."""
    if not isinstance(payload, dict) or payload.get("status") != "ERROR":
        return None
    for key in ("msg", "message", "reason", "detail", "error"):
        value = payload.get(key)
        if value is not None and value != "":
            return str(value)
    return "MAST reported ERROR without a reason"


_mast_error_reason = mast_error_reason


def _declared_length(response):
    """Content-Length as an integer, or None when absent.

    A malformed header is treated as an incomplete transfer (retryable), not
    as an unexpected exception.
    """
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal():
        raise IncompleteRead(b"", None)
    return int(text)


def _mast_attempt(attempt, **details):
    return {"attempt": attempt, "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **details}


class AcquisitionTransport:
    """Bounded archive transport retaining typed failures for the caller.

    ``transport_failures`` accumulates one ``AcquisitionFailure`` per failed
    request; callers detect their own failures by its growth, so one instance
    serves one thread at a time. ``_http_get_bytes(url, accept)`` keeps the
    historical retrieval seam (embedding applications supply their own);
    long TAP-sync queries use a form POST without changing their parameters.
    """

    ESA_GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
    AIP_GAIA_TAP_URL = "https://gaia.aip.de/tap/sync"
    fetch_tap_async = fetch_tap_async

    def __init__(self, policy: NetworkPolicy):
        if not isinstance(policy, NetworkPolicy):
            raise TypeError("policy must be a NetworkPolicy")
        self.policy = policy
        self.transport_failures = []
        self._mast_attempts = []

    def _http_get_bytes(self, url, accept):
        """Retrieve bounded bytes, using form POST for long TAP-sync queries."""
        return self._request(url, accept)

    def get_bytes(self, url, accept):
        """Retrieve a bounded body under the policy; None after a recorded failure.

        Delegates to ``_http_get_bytes``, so a seam patched there applies here too.
        """
        return self._http_get_bytes(url, accept)

    @property
    def mast_attempts(self):
        """Attempt evidence of the most recent ``post_mast`` call, for retention."""
        return self._mast_attempts

    def post_form(self, url, data, accept="application/json"):
        """Post a form under the same network, timeout and retry policy as GET."""
        return self._request(url, accept, data=data)

    def post_mast(self, url, data):
        """Post a MAST request with one shared budget for HTTP and provider failures.

        The last ERROR response remains the return value for existing callers.
        Attempt evidence is kept until the next MAST request for the product
        acquisition owner to retain, including responses preceding an outage.
        """
        self._mast_attempts = []
        return self._request(url, "application/json", data=data, mast_attempts=self._mast_attempts)

    def download(self, url, destination):
        """Stream a complete file atomically without buffering the release in memory.

        An existing destination is refused before any network request: a
        completed cached release is never replaced.
        """
        destination = Path(destination)
        if destination.is_symlink():
            raise ValueError("download destination may not be a symlink")
        if destination.exists():
            raise AcquisitionError("download destination already exists")
        return self._request(url, "application/octet-stream", destination)

    def _request(self, url, accept, destination=None, *, data=None, mast_attempts=None):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("acquisition requires an HTTPS URL without credentials")
        if not self.policy.allow_network:
            raise AcquisitionError("network access is disabled by policy")
        for attempt in range(1, self.policy.attempts + 1):
            failed_body = None
            body_issue = None
            reading_response = False
            try:
                headers = {"User-Agent": request_user_agent(self.policy), "Accept": accept}
                if data is not None:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                request = archive_request(url, data=data, headers=headers)
                with urlopen(request, timeout=self.policy.timeout_seconds) as response:
                    reading_response = True
                    if destination is not None:
                        return self._download_response(response, destination)
                    body = response.read(self.policy.max_response_bytes + 1)
                    if len(body) > self.policy.max_response_bytes:
                        if mast_attempts is not None:
                            mast_attempts.append(_mast_attempt(attempt, body=body,
                                                               body_issue="response exceeded the configured size limit"))
                        raise AcquisitionError("archive response exceeds the configured size limit")
                    length = _declared_length(response)
                    if length is not None and length != len(body):
                        raise IncompleteRead(body, length - len(body))
                    if mast_attempts is None:
                        return body
                    try:
                        reason = mast_error_reason(json.loads(body.decode("utf-8")))
                    except (UnicodeDecodeError, ValueError, RecursionError):
                        reason = None
                    if reason is None:
                        mast_attempts.append(_mast_attempt(attempt, body=body))
                        return body
                    failure = AcquisitionFailure(parsed.hostname, "provider_error", True, attempt, reason, 200)
            except HTTPError as exc:
                exception_class = type(exc).__name__
                status = int(exc.code)
                retryable = status in (429, 500, 502, 503, 504)
                category = "rate_limit" if status == 429 else "http_status"
                reason = f"HTTP {status}: {exc.reason}"
                if mast_attempts is not None and exc.fp is not None:
                    try:
                        failed_body = exc.read(self.policy.max_response_bytes + 1)
                        if len(failed_body) > self.policy.max_response_bytes:
                            body_issue = "HTTP error response exceeded the configured size limit"
                    except IncompleteRead as read_error:
                        failed_body = read_error.partial
                        body_issue = str(read_error)
                    except OSError as read_error:
                        body_issue = f"{type(read_error).__name__}: {read_error}"
            except (URLError, TimeoutError, OSError, IncompleteRead) as exc:
                # A ConnectionError (e.g. http.client.RemoteDisconnected) carries no errno but is an outage.
                if (isinstance(exc, OSError) and not isinstance(exc, (URLError, TimeoutError, ssl.SSLError, ConnectionError))
                        and exc.errno not in {errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
                                              errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.EHOSTDOWN,
                                              errno.ENETUNREACH, errno.ENETDOWN, errno.ENETRESET, errno.EPIPE}):
                    raise
                cause = exc.reason if isinstance(exc, URLError) else exc
                exception_class = type(exc).__name__
                retryable = not isinstance(cause, ssl.SSLCertVerificationError)
                category = ("tls_verification" if not retryable else
                            "dns" if isinstance(cause, socket.gaierror) else
                            "tls_transport" if isinstance(cause, ssl.SSLError) else
                            ("read_timeout" if reading_response else "connect_timeout")
                            if isinstance(cause, TimeoutError) else "incomplete_transfer")
                status = None
                reason = f"{type(exc).__name__}: {exc}"
                if mast_attempts is not None and isinstance(exc, IncompleteRead):
                    failed_body = exc.partial
                    body_issue = reason
            else:
                # The only non-returning success is an HTTP-200 MAST ERROR.
                if mast_attempts is not None:
                    mast_attempts.append(_mast_attempt(attempt, body=body, failure=failure))
                if attempt == self.policy.attempts:
                    _warn("MAST provider request failed: %s", reason)
                    return body
                sleep_before_retry(min(64.0, self.policy.backoff_seconds * 2 ** (attempt - 1)))
                continue
            failure = AcquisitionFailure(parsed.hostname, category, retryable, attempt, reason, status, exception_class)
            if mast_attempts is not None:
                entry = _mast_attempt(attempt, failure=failure)
                if failed_body is not None:
                    entry["body"] = failed_body
                if body_issue is not None:
                    entry["body_issue"] = body_issue
                mast_attempts.append(entry)
            if not retryable or attempt == self.policy.attempts:
                self.transport_failures.append(failure)
                _warn("archive request failed: %s", reason)
                return None
            sleep_before_retry(min(64.0, self.policy.backoff_seconds * 2 ** (attempt - 1)))
        raise AssertionError("unreachable retry loop")

    def _download_response(self, response, destination):
        if destination.is_symlink():
            raise ValueError("download destination may not be a symlink")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.policy.max_download_bytes:
                        raise AcquisitionError("archive download exceeds the configured size limit")
                    handle.write(chunk)
                length = _declared_length(response)
                if length is not None and length != total:
                    raise IncompleteRead(b"", length - total)
                handle.flush()
                os.fsync(handle.fileno())
            # A concurrent writer never replaces a completed cached release.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise AcquisitionError("download destination already exists") from None
            _sync_directory(destination.parent)
            return destination
        finally:
            # The hard link keeps the data; the temporary name always goes.
            temporary.unlink(missing_ok=True)


def write_mast_attempts(output_dir, stem, attempts, final_name=None):
    """Retain ``post_mast`` attempt evidence; return the attempt history entries.

    Each attempt that carried a body is written to ``<stem>_attempt_<n>.json``,
    except the last one when ``final_name`` is given: that body is the accepted
    response and is written once, under ``final_name``.
    """
    history = []
    for index, item in enumerate(attempts):
        entry = {"attempt": item["attempt"], "recorded_utc": item["recorded_utc"]}
        if "failure" in item:
            entry["failure"] = item["failure"].to_manifest()
            if item["failure"].category == "provider_error":
                entry["failure"]["event"] = "mast.provider_error"
        if "body_issue" in item:
            entry["body_issue"] = item["body_issue"]
        if "body" in item:
            name = (final_name if index == len(attempts) - 1 and final_name is not None
                    else f"{stem}_attempt_{item['attempt']}.json")
            write_bytes(Path(output_dir) / name, item["body"])
            entry["response"] = name
        history.append(entry)
    return history
