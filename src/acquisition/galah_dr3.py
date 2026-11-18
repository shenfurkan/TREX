"""Read GALAH DR3 abundance rows and validate Gaia associations.

Cold acquisition uses Data Central's ADQL main-star table instead of streaming
the 833 MB official allstar FITS.  The live TAP service (read 2026-11-15,
https://datacentral.org.au/vo/tap/sync) exposes ``galah_dr3.main_star``; its
``TAP_SCHEMA.columns`` description names ``source_id`` as the Gaia DR2
source_id, so the targeted query is keyed by the DR2 counterparts already
obtained from ``gaiaedr3.dr2_neighbourhood``.  The GALAH DR3 page (read
2026-11-14, https://www.galah-survey.org/dr3/the_catalogues/) names Data
Release 3.2 and the ``_v2`` allstar FITS; Data Central's GALAH update page (read
2026-11-14, https://docs.datacentral.org.au/galah/dr3/updates/) states that DR3.2
did not change the main-star stellar parameters or abundances, only Gaia
identifiers/astrometry and radial velocities.  The TAP table therefore carries
the same [Fe/H] values while its identity join stays DR2; the DR3 identity is
verified against the bidirectional neighbourhood answer before a row is
admitted.  This is a documented service adaptation, not a claim that the TAP
table is the DR3.2 ``_v2`` schema.  Buder et al. (2021), MNRAS 506, 150, "Main
catalogues", retained at
``literature/buder_2021_galah_dr3/arxiv_source/mnras_v3.tex:1008-1029``,
establishes the one-row-per-star allstar semantics and TAP access.
"""

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import numpy as np
from astropy.io import fits

from . import transport
from .gaia_crossmatch import DR2_COUNTERPART_RADIUS_MAS, dr2_counterparts, query_gaia_neighborhood
from .identity import gaia_identifier as _gaia_identifier
from .storage import write_bytes, write_json
from .transport import AcquisitionError, AcquisitionFailure
from .votable import parse_votable_bytes

# Canonical retained-row fields, shared by FITS and targeted-service rows.
GALAH_DR3_FIELDS = (
    "sobject_id", "dr2_source_id", "dr3_source_id", "flag_sp", "flag_fe_h",
    "fe_h", "e_fe_h", "snr_c3_iraf",
)
# Data Central's live main-star table columns; its source_id is Gaia DR2.
GALAH_DR3_TARGETED_FIELDS = (
    "sobject_id", "source_id", "flag_sp", "flag_fe_h", "fe_h", "e_fe_h", "snr_c3_iraf",
)
GALAH_DR3_URL = (
    "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/"
    "GALAH_DR3_main_allstar_v2.fits"
)
GALAH_DR3_TAP_URL = "https://datacentral.org.au/vo/tap/sync"
GALAH_DR3_TAP_TABLE = "galah_dr3.main_star"
GALAH_DR3_SERVICE_RELEASE = "GALAH DR3 (Data Central TAP main_star)"
# One targeted TAP query is bounded in wall time, not only in attempts: the
# 2026-11-16 auto-vet run spent 308 s in tic-neighbors and then deferred with
# no draws after a single DataCentral read timeout.  300 s is the established
# per-archive-path budget and stays below the runner's 900 s idle limit.
# Three attempts absorb two transient timeouts; the capped doubling wait
# matches the shared acquisition transport's backoff shape.
GALAH_DR3_TAP_BUDGET_SECONDS = 300.0
GALAH_DR3_TAP_ATTEMPTS = 3
GALAH_DR3_TAP_BACKOFF_SECONDS = 2.0
GALAH_DR3_TAP_BACKOFF_MAX_SECONDS = 64.0


def _failure_detail(failure):
    """One-line diagnostic for either repository's transport-failure object."""
    if failure is None:
        return "no transport diagnostic"
    reason = getattr(failure, "reason", None)
    if not isinstance(reason, str) or not reason:
        message = getattr(failure, "message", None)
        if callable(message):
            try:
                produced = message()
            except Exception:
                produced = None
            if isinstance(produced, str) and produced:
                reason = produced
    if not isinstance(reason, str) or not reason:
        manifest = getattr(failure, "to_manifest", None)
        if callable(manifest):
            try:
                produced = manifest().get("message")
            except Exception:
                produced = None
            if isinstance(produced, str) and produced:
                reason = produced
    category = getattr(failure, "category", None)
    if isinstance(reason, str) and reason:
        return f"{category}: {reason}" if category else reason
    return type(failure).__name__


# Retryable provider/transport categories, which D-55 treats as an outage.
_GALAH_DR3_OUTAGE_CATEGORIES = frozenset({
    "connect_timeout", "read_timeout", "incomplete_transfer", "dns",
    "tls_transport", "rate_limit", "service_error", "provider", "provider_error",
})


def _galah_dr3_classification(failure):
    """Return the D-55 class of one failed GALAH TAP request.

    ``outage`` is a retryable transport, provider or 5xx failure.
    ``malformed-request`` is a 4xx rejection: the archive refused this ADQL or
    this caller, so the same request cannot succeed unchanged (401/403 are
    refusals, not weather). ``tls_verification`` keeps D-55's separate
    non-retryable class because a trust failure is never fixed by waiting.
    Anything else stays ``unclassified``; an operational defect is not
    relabelled as an outage.
    """
    category = getattr(failure, "category", None)
    if category in _GALAH_DR3_OUTAGE_CATEGORIES:
        return "outage"
    if category == "http_status":
        status = getattr(failure, "http_status", None)
        if isinstance(status, int) and 400 <= status < 500:
            return "malformed-request"
        if getattr(failure, "retryable", False):
            return "outage"
        return "unclassified"
    if category == "tls_verification":
        return "tls_verification"
    return "unclassified"


def _galah_dr3_tap_response(release, url):
    """Fetch one targeted query under a recorded wall budget.

    Returns ``(body, record, failure)``.  ``record`` states the budget, the
    attempts used, the elapsed monotonic seconds and which bound stopped the
    loop: ``None`` when a body arrived, else ``budget``, ``attempts`` or
    ``classification``.  ``classification`` is ``answered`` when a body arrived
    (the caller adds ``empty-answer`` when the parsed table has no rows), else
    the D-55 class of the last failure.  One attempt is one ``_http_get_bytes``
    call, so an embedding service's own per-request retry policy stays inside an
    attempt.
    """
    started = time.monotonic()
    attempts, body, failure, bounded_by = 0, None, None, None
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= GALAH_DR3_TAP_BUDGET_SECONDS:
            bounded_by = "budget"
            break
        attempts += 1
        before = len(release.service.transport_failures)
        body = release.service._http_get_bytes(url, "application/xml")
        if body is not None:
            break
        failure = (release.service.transport_failures[-1]
                   if len(release.service.transport_failures) > before else None)
        elapsed = time.monotonic() - started
        if (_galah_dr3_classification(failure) != "outage"
                or not getattr(failure, "retryable", False)):
            bounded_by = "classification"
            break
        if elapsed >= GALAH_DR3_TAP_BUDGET_SECONDS:
            bounded_by = "budget"
            break
        if attempts >= GALAH_DR3_TAP_ATTEMPTS:
            bounded_by = "attempts"
            break
        delay = min(GALAH_DR3_TAP_BACKOFF_MAX_SECONDS,
                    GALAH_DR3_TAP_BACKOFF_SECONDS * 2 ** (attempts - 1))
        transport.sleep_before_retry(min(delay, GALAH_DR3_TAP_BUDGET_SECONDS - elapsed))
    record = {
        "budget_seconds": GALAH_DR3_TAP_BUDGET_SECONDS,
        "attempts": attempts,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "bounded_by": bounded_by,
        "classification": "answered" if body is not None else _galah_dr3_classification(failure),
    }
    if failure is not None:
        last = {"retryable": bool(getattr(failure, "retryable", False)),
                "reason": _failure_detail(failure)}
        category = getattr(failure, "category", None)
        if category is not None:
            last["category"] = category
        status = getattr(failure, "http_status", None)
        if isinstance(status, int):
            last["http_status"] = status
        record["last_failure"] = last
    return body, record, failure


@dataclass(frozen=True)
class GalahTargetedRelease:
    """A cold-cache GALAH handle that resolves rows through the documented TAP table."""

    cache_dir: Path
    service: Any
    identity: dict


def galah_shared_release(cache_dir, service, *, download=False):
    """Return the retained official release or a documented targeted-service handle.

    A retained ``GALAH_DR3_main_allstar_v2.fits`` remains the preferred source:
    its sibling ``.retrieval.json`` records URL, UTC retrieval date and size, and
    a size mismatch (an interrupted download) is refused. With no retained
    release the default returns ``GalahTargetedRelease``: ``galah_dr3_rows`` then
    queries only the requested Gaia DR2 counterparts in
    ``galah_dr3.main_star`` and retains the raw VOTable. No implicit bulk
    download occurs. Explicit ``download=True`` still stages the official FITS
    through ``service.download(url, path)`` for callers that deliberately keep
    that option.

    The targeted identity before its row query names the documented service,
    table and release, with ``bytes=None`` because no response exists yet;
    ``galah_dr3_rows`` fills the response URL, UTC time and byte count in the
    same mapping and binds them in the matched-row artifact.
    """
    root = Path(cache_dir)
    path = root / "GALAH_DR3_main_allstar_v2.fits"
    receipt = path.with_suffix(".retrieval.json")
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("GALAH shared release must be a regular file inside the catalogue cache")
    if not path.exists() and not download:
        identity = {
            "url": GALAH_DR3_TAP_URL,
            "table": GALAH_DR3_TAP_TABLE,
            "release": GALAH_DR3_SERVICE_RELEASE,
            "row_space": "votable_response",
            "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "bytes": None,
        }
        return GalahTargetedRelease(root, service, identity), identity
    if not path.exists():
        before = len(service.transport_failures)
        result = service.download(GALAH_DR3_URL, path)
        if result is None or not path.is_file():
            failure = service.transport_failures[-1] if len(service.transport_failures) > before else None
            detail = "" if failure is None else f"; {_failure_detail(failure)}"
            error = AcquisitionError(
                f"GALAH download returned no completed release from {GALAH_DR3_URL} to {path}{detail}",
                retryable=bool(failure and failure.retryable))
            error.network_failure = failure
            raise error
        write_json(receipt, {
            "url": GALAH_DR3_URL, "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "bytes": path.stat().st_size,
        })
    if not path.is_file() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("GALAH release requires its URL/retrieval-date record")
    try:
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("GALAH release URL/retrieval-date record is unreadable") from exc
    if (not isinstance(metadata, dict) or metadata.get("url") != GALAH_DR3_URL
            or not isinstance(metadata.get("retrieved_utc"), str) or not metadata["retrieved_utc"]
            or metadata.get("bytes") != path.stat().st_size):
        raise ValueError("GALAH release URL/retrieval-date record does not match the retained release")
    try:
        retrieved = datetime.fromisoformat(metadata["retrieved_utc"])
    except ValueError as exc:
        raise ValueError("GALAH release retrieval date must be UTC") from exc
    if retrieved.utcoffset() != timedelta(0):
        raise ValueError("GALAH release retrieval date must be UTC")
    return path, metadata


def retain_galah_rows(output_dir, rows, metadata, associations):
    """Retain matched rows, release identity and any targeted raw response."""
    identity = {key: metadata[key] for key in ("url", "retrieved_utc", "bytes")}
    for key in ("table", "release", "row_space"):
        if key in metadata:
            identity[key] = metadata[key]
    output_dir = Path(output_dir)
    raw_response = None
    response_path = metadata.get("response_path")
    if response_path is not None:
        source = Path(response_path)
        if source.is_symlink() or not source.is_file() or source.stat().st_size != metadata["bytes"]:
            raise ValueError("GALAH targeted response is missing or does not match its byte record")
        response_name = f"galah_service_{uuid4().hex}.xml"
        write_bytes(output_dir / response_name, source.read_bytes())
        try:
            source.unlink()
        except OSError:
            pass
        raw_response = {
            "path": response_name, "uri": identity["url"],
            "retrieved_utc": identity["retrieved_utc"], "bytes": identity["bytes"],
        }
    payload = {
        "schema_version": 1, "release": identity, "fields": list(GALAH_DR3_FIELDS),
        "rows": rows,
        "associations": {
            source_id: {"status": entry["status"], "row_index_zero_based": entry.get("row", {}).get(
                "row_index_zero_based")} for source_id, entry in associations.items()
        },
    }
    if raw_response is not None:
        payload["response"] = raw_response
    name = f"galah_rows_{uuid4().hex}.json"
    write_json(output_dir / name, payload)
    artifact = {"path": name, "release": identity}
    if raw_response is not None:
        artifact["response"] = raw_response
    return artifact


def _galah_targeted_rows(release, _dr3_ids, dr2_ids):
    """Query Data Central's DR2-keyed main-star table and retain its raw VOTable.

    A retryable transport failure is retried inside ``GALAH_DR3_TAP_BUDGET_SECONDS``;
    the budget, attempts, elapsed seconds, stopping bound and D-55 class are left
    in ``release.identity["transport"]`` for the caller's query record. A refusal
    (4xx) or a trust failure is not retried, and an exhausted outage raises
    ``ValueError`` with the classified failure attached. A complete zero-row
    answer returns ``[]``: that is a result, never an outage.
    """
    dr2 = list(dict.fromkeys(_gaia_identifier(value) for value in dr2_ids))
    if not dr2:
        return []
    query = (f"SELECT {','.join(GALAH_DR3_TARGETED_FIELDS)} FROM {GALAH_DR3_TAP_TABLE} "
             "WHERE source_id IN (" + ",".join(dr2) + ")")
    url = GALAH_DR3_TAP_URL + "?" + urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "MAXREC": "10000", "QUERY": query,
    })
    body, transport_record, failure = _galah_dr3_tap_response(release, url)
    if body is None:
        detail = _failure_detail(failure)
        error = ValueError(
            f"GALAH targeted query did not complete: {transport_record['classification']} after "
            f"{transport_record['attempts']} attempt(s) in {transport_record['elapsed_seconds']} s "
            f"(bounded by {transport_record['bounded_by']}; {detail}); stage the official local release "
            f"from {GALAH_DR3_URL} with its retrieval receipt, or retry the service"
        )
        error.network_failure = failure
        release.identity["transport"] = transport_record
        raise error
    descriptor, name = tempfile.mkstemp(prefix="galah_dr3_tap_", suffix=".xml")
    os.close(descriptor)
    response_path = Path(name)
    write_bytes(response_path, body)
    release.identity.update({
        "url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bytes": len(body),
        "response_path": str(response_path),
        "transport": transport_record,
    })
    try:
        rows = parse_galah_response(body)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        failure = AcquisitionFailure(
            urlsplit(GALAH_DR3_TAP_URL).hostname or "datacentral.org.au",
            "provider", True, 1, reason, None, type(exc).__name__)
        error = ValueError(
            f"GALAH targeted response is incomplete: {reason}; raw response: {response_path}"
        )
        error.network_failure = failure
        raise error from exc
    if not rows:
        transport_record["classification"] = "empty-answer"
    if any(row["dr2_source_id"] not in dr2 for row in rows):
        raise ValueError("GALAH targeted response contains an unrequested Gaia source")
    return rows


def parse_galah_response(body):
    """Validate Data Central's native TAP columns and DR2 identities.

    Returns rows in response order with zero-based response indices. The
    service's ``source_id`` is Gaia DR2; ``dr2_source_id`` carries it and
    ``dr3_source_id`` stays None until the caller's bidirectional neighbourhood
    check resolves the DR3 identity. Missing abundances remain None; malformed
    columns and incomplete TAP responses raise ValueError. No network or cache
    is used during retained replay.
    """
    parsed = parse_votable_bytes(body)
    table = parsed["table"]
    if set(table.colnames) != set(GALAH_DR3_TARGETED_FIELDS):
        raise ValueError("GALAH targeted response does not contain exactly the required columns")
    for key in ("sobject_id", "source_id"):
        dtype = table[key].dtype
        if dtype.kind != "i" or dtype.itemsize != 8:
            raise ValueError("GALAH targeted identity columns must be signed 64-bit integers")
    for key in ("flag_sp", "flag_fe_h"):
        if table[key].dtype.kind not in "iu":
            raise ValueError("GALAH targeted quality flags must be integers")
    for key in ("fe_h", "e_fe_h", "snr_c3_iraf"):
        if table[key].dtype.kind != "f":
            raise ValueError("GALAH targeted abundance and S/N columns must be floating point")
    result = []
    for index, values in enumerate(parsed["data"]):
        item = dict(zip(table.colnames, values))
        if any(item[key] is None or isinstance(item[key], bool) or not isinstance(item[key], int)
               for key in ("sobject_id", "source_id", "flag_sp", "flag_fe_h")):
            raise ValueError("GALAH targeted response has a missing, noninteger identity or quality flag")
        for key in ("sobject_id", "source_id", "flag_sp", "flag_fe_h"):
            item[key] = int(item[key])
        for key in ("fe_h", "e_fe_h", "snr_c3_iraf"):
            value = item[key]
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("GALAH targeted response has a nonnumeric abundance or S/N")
                value = float(value)
            item[key] = value if value is not None and math.isfinite(value) else None
        item["dr2_source_id"] = str(item.pop("source_id"))
        item["dr3_source_id"] = None
        item["row_index_zero_based"] = index
        result.append(item)
    return result


def galah_dr3_rows(release, dr3_ids: list[str], dr2_ids: list[str]) -> list[dict[str, Any]]:
    """Return rows binding requested DR3 or candidate DR2 identities, retaining ambiguity.

    ``release`` is either a retained official allstar FITS path (read in place)
    or a ``GalahTargetedRelease`` from ``galah_shared_release``. The live TAP
    table is keyed by Gaia DR2 ``source_id``; its rows carry ``dr3_source_id``
    as None and the caller's bidirectional neighbourhood check resolves it.
    Targeted rows carry the position of the row in the retained VOTable
    response; the artifact's ``release.row_space`` distinguishes that from an
    allstar release index. Exact DR3/DR2 identifiers are preserved as decimal
    strings and the selection/quality rules stay in ``galah_dr3_associations``.
    """
    if isinstance(release, GalahTargetedRelease):
        return _galah_targeted_rows(release, dr3_ids, dr2_ids)
    path = Path(release)
    dr3 = np.array([int(_gaia_identifier(value)) for value in dr3_ids], dtype=np.int64)
    dr2 = np.array([int(_gaia_identifier(value)) for value in dr2_ids], dtype=np.int64)
    if path.is_symlink() or not path.is_file():
        raise ValueError("GALAH release is not an available regular file")
    try:
        with fits.open(path, memmap=True) as hdus:
            if len(hdus) < 2 or not isinstance(hdus[1], fits.BinTableHDU):
                raise ValueError("GALAH release has no binary results table")
            hdu = hdus[1]
            info = hdu.fileinfo()
            if info is None or path.stat().st_size < info["datLoc"] + info["datSpan"]:
                raise ValueError("GALAH release results table is truncated")
            columns = hdu.columns
            if any(name not in columns.names for name in GALAH_DR3_FIELDS):
                raise ValueError("GALAH release lacks required identity or abundance columns")
            if any(columns[name].format != "K" for name in GALAH_DR3_FIELDS[:5]):
                raise ValueError("GALAH release identity and flag columns are not signed 64-bit integers")
            if any(columns[name].format not in ("E", "D") for name in GALAH_DR3_FIELDS[5:]):
                raise ValueError("GALAH release abundance or S/N columns are not floating point")
            table = hdu.data
            wanted = np.zeros(len(table), dtype=bool)
            if dr3.size:
                wanted |= np.isin(table["dr3_source_id"], dr3)
            if dr2.size:
                wanted |= np.isin(table["dr2_source_id"], dr2)
            selected = np.flatnonzero(wanted)
            result = []
            for index in selected:
                row = table[index]
                item = {key: int(row[key]) for key in GALAH_DR3_FIELDS[:5]}
                for key in GALAH_DR3_FIELDS[5:]:
                    value = float(row[key])
                    item[key] = value if math.isfinite(value) else None
                item["dr2_source_id"] = str(item["dr2_source_id"])
                item["dr3_source_id"] = str(item["dr3_source_id"])
                item["row_index_zero_based"] = int(index)
                result.append(item)
            return result
    except (OSError, IndexError, KeyError, TypeError, fits.VerifyError) as exc:
        raise ValueError(f"GALAH release could not be read completely: {type(exc).__name__}: {exc}") from exc


def validated_neighbourhood_pair(source_ids, forward, reverse):
    """Check that the forward and reverse Gaia neighbourhood answers agree.

    Every forward DR3->DR2 pair must reappear, identical, in the reverse
    DR2->DR3 answer, which may not name a DR2 source the forward answer did not
    give. Returns ``(forward_map, reverse_rows)``; raises ``ValueError`` when
    an answer is duplicated, incomplete or inconsistent.
    """
    forward_map = dr2_counterparts(forward, source_ids)
    all_forward = [row for entry in forward_map.values() for row in entry["counterparts"]]
    dr2_ids = {row["dr2_source_id"] for row in all_forward}
    reverse_ids = [_gaia_identifier(row["dr3_source_id"]) for row in reverse]
    reverse_map = dr2_counterparts(reverse, list(dict.fromkeys(reverse_ids)))
    all_reverse = [row for entry in reverse_map.values() for row in entry["counterparts"]]
    if any(row["dr2_source_id"] not in dr2_ids for row in all_reverse):
        raise ValueError("Gaia reverse neighbourhood returned an unrequested DR2 source")
    forward_pairs = [(row["dr3_source_id"], row["dr2_source_id"]) for row in all_forward]
    reverse_pairs = [(row["dr3_source_id"], row["dr2_source_id"]) for row in all_reverse]
    if (len(set(forward_pairs)) != len(forward_pairs) or len(set(reverse_pairs)) != len(reverse_pairs)
            or not set(forward_pairs).issubset(reverse_pairs)):
        raise ValueError("Gaia neighbourhood responses are duplicated or incomplete")
    reverse_by_pair = dict(zip(reverse_pairs, all_reverse))
    if any(reverse_by_pair[pair] != row for pair, row in zip(forward_pairs, all_forward)):
        raise ValueError("Gaia forward and reverse neighbourhood responses disagree")
    return forward_map, all_reverse


def galah_dr3_associations(
    source_ids: list[str], rows: list[dict[str, Any]], forward: list[dict[str, Any]], reverse: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Select only a unique, bidirectionally associated, quality-approved GALAH row.

    Both Gaia neighbourhood queries must have completed before this function is
    called; missing reverse links are errors, not evidence that GALAH is empty.
    """
    ids = [_gaia_identifier(value) for value in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Gaia source identities must be unique")
    forward_map, all_reverse = validated_neighbourhood_pair(ids, forward, reverse)
    rows_by_dr3, rows_by_dr2, close_reverse = {}, {}, {}
    for row in rows:
        if row.get("dr3_source_id") is not None:
            rows_by_dr3.setdefault(row["dr3_source_id"], []).append(row)
        rows_by_dr2.setdefault(row["dr2_source_id"], []).append(row)
    for item in all_reverse:
        if item["angular_distance"] < DR2_COUNTERPART_RADIUS_MAS:
            close_reverse.setdefault(item["dr2_source_id"], []).append(item)

    results: dict[str, dict[str, Any]] = {}
    for source_id in ids:
        counterparts = forward_map[source_id]["counterparts"]
        close = [row for row in counterparts if row["angular_distance"] < DR2_COUNTERPART_RADIUS_MAS]
        candidates = {id(row): row for row in rows_by_dr3.get(source_id, [])}
        for dr2_id in {row["dr2_source_id"] for row in close}:
            candidates.update((id(row), row) for row in rows_by_dr2.get(dr2_id, []))
        candidates = list(candidates.values())
        if not candidates:
            results[source_id] = {"status": "unavailable", "reason": "no exact or uniquely close GALAH row"}
            continue
        if len(candidates) != 1 or len(close) != 1:
            results[source_id] = {"status": "unavailable", "reason": "ambiguous GALAH or Gaia DR2 association"}
            continue
        row, dr2_id = candidates[0], close[0]["dr2_source_id"]
        reverse_close = close_reverse.get(dr2_id, [])
        # The live Data Central table carries only the Gaia DR2 source_id; a
        # missing DR3 identity is resolved by the bidirectional neighbourhood
        # check below, not by trusting either side alone.
        if (row.get("dr3_source_id") not in (None, source_id) or row["dr2_source_id"] != dr2_id
                or len(reverse_close) != 1 or reverse_close[0]["dr3_source_id"] != source_id):
            results[source_id] = {"status": "unavailable", "reason": "ambiguous GALAH DR2-to-DR3 association"}
            continue
        # GALAH DR3 best practices, Buder et al. (2021), DOI 10.1093/mnras/stab1242.
        value, error, snr = (row.get(key) for key in ("fe_h", "e_fe_h", "snr_c3_iraf"))
        if (row.get("flag_sp") != 0 or row.get("flag_fe_h") != 0
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (value, error, snr)) or error <= 0 or snr <= 30):
            results[source_id] = {"status": "unavailable", "reason": "GALAH iron quality, uncertainty or S/N cut"}
            continue
        results[source_id] = {"status": "available", "row": row}
    return results


def fetch_galah_gaia_neighborhood(output_dir, service, source_ids):
    """Retain completed forward and reverse Gaia cross-matches for GALAH DR3."""
    output_dir = Path(output_dir)
    ids = [_gaia_identifier(value) for value in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Gaia source identities must be unique")
    retrieval_id = uuid4().hex
    forward_record, forward, failure = query_gaia_neighborhood(output_dir, service, "dr3_source_id", ids, retrieval_id)
    record = {"status": "unavailable", "forward": forward_record}
    if forward is None:
        return record, None, None, failure
    try:
        dr2_ids = list(dict.fromkeys(_gaia_identifier(row["dr2_source_id"]) for row in forward))
    except (ValueError, KeyError, TypeError) as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record, None, None, None
    reverse_record, reverse, failure = query_gaia_neighborhood(output_dir, service, "dr2_source_id", dr2_ids, retrieval_id)
    record["reverse"] = reverse_record
    if reverse is None:
        return record, None, None, failure
    try:
        validated_neighbourhood_pair(ids, forward, reverse)
    except (ValueError, KeyError, TypeError) as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record, None, None, None
    record["status"] = "available" if ids else "not-needed"
    return record, forward, reverse, None
