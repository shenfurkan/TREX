"""Normalize source-associated TIC v8 rows without inventing stellar values."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode
from uuid import uuid4

from .identity import exact_identifier, gaia_identifier
from .storage import write_bytes, write_json
from trex.runtime.logging import host_logger

from .transport import AcquisitionError, mast_error_reason, write_mast_attempts

MAST_INVOKE_URI = "https://mast.stsci.edu/api/v0/invoke"
TIC_SERVICE = "Mast.Catalogs.Filtered.Tic"
TIC_NEIGHBOR_METHOD = "tic-v8-mast"
TIC_CONE_SERVICE = "Mast.Catalogs.Tic.Cone"
TIC_TARGET_QUERY_FILE = "tic_target_query.json"

TIC_REJECTED_DISPOSITIONS = frozenset(("SPLIT", "DUPLICATE", "ARTIFACT"))
DUPLICATE_TIC_ROW_RULE = (
    "when several TIC rows share one Gaia source_id, exactly one row whose "
    "disposition is not DUPLICATE/SPLIT/ARTIFACT is usable; zero or several "
    "such rows are refused rather than taking the first"
)


def _retrieved_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rows_or_error(body, *, single):
    try:
        data = json.loads(body.decode("utf-8"))
        reason = mast_error_reason(data)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"response is not valid UTF-8 JSON: {exc}") from exc
    if reason is not None:
        error = AcquisitionError(f"MAST TIC query failed at the provider: {reason}", retryable=True)
        error.category = "provider_error"
        raise error
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("response has no data table")
    if data.get("status") not in (None, "COMPLETE"):
        raise ValueError(f"response did not complete: {data.get('status')}")
    paging = data.get("paging")
    if paging is not None and not isinstance(paging, dict):
        raise ValueError("response has malformed paging")
    if isinstance(paging, dict):
        pages, count = paging.get("pagesFiltered", 1), paging.get("rowsFiltered")
        invalid_count = count is not None and (isinstance(count, bool) or not isinstance(count, int)
                                               or count != len(data["data"]))
        if (isinstance(pages, bool) or not isinstance(pages, int) or pages not in (0, 1)
                or (pages == 0 and data["data"]) or invalid_count):
            raise ValueError("response is incomplete or paginated")
    if any(not isinstance(row, dict) for row in data["data"]):
        raise ValueError("response contains a malformed row")
    if single and len(data["data"]) != 1:
        raise ValueError("query did not return exactly one row")
    return data["data"]


def _write_failure(output, stem, error, history):
    """Keep a transport exhaustion addressable without replacing a good query record."""
    record = {"status": "unavailable", "category": getattr(error, "category", "transport_failure"),
              "reason": str(error), "retrieved_utc": _retrieved_utc()}
    if history:
        record["attempt_history"] = history
    failure = getattr(error, "network_failure", None)
    if failure is not None:
        record["network_failure"] = failure.to_manifest()
    path = output / f"{stem}_failure_{uuid4().hex}.json"
    write_json(path, record)
    return path.name


def _post_tic(payload, output, transport, stem, response_name, label):
    """POST one TIC request; retain its attempts and accepted body, or raise after a failure record.

    ``stem`` names the attempt files (unique per call, so a rerun never
    overwrites an earlier attempt's raw response); the accepted body is
    written once, as ``response_name``. Returns ``(body, history)``.
    """
    request = urlencode({"request": json.dumps(payload, sort_keys=True)}).encode("ascii")
    before = len(transport.transport_failures)
    request_error = None
    try:
        body = transport.post_mast(MAST_INVOKE_URI, request)
    except AcquisitionError as exc:
        # Network disabled by policy, size limit: still retain the attempts made.
        body, request_error = None, exc
    attempts = getattr(transport, "mast_attempts", [])
    history = write_mast_attempts(output, stem, attempts, None if body is None else response_name)
    failure = transport.transport_failures[-1] if len(transport.transport_failures) > before else None
    if body is None:
        detail = str(request_error) if request_error is not None else failure.reason if failure else None
        error = request_error or AcquisitionError(f"MAST {label} failed" + (f": {detail}" if detail else ""),
                                                  retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        error.category = failure.category if failure is not None else getattr(error, "category", "transport_failure")
        error.attempt_history = history
        error.failure_record = _write_failure(output, stem, error, history)
        host_logger(__name__).warning("MAST %s failed after %d attempt(s): %s", label, len(history),
                                      error.category)
        raise error
    if not history or history[-1].get("response") != response_name:
        write_bytes(output / response_name, body)
    return body, history


def _parsed_rows(body, record, query_path, *, single, label, check=None):
    """Rows of an accepted TIC body; the query record is written once with the outcome.

    ``check(rows)`` may raise ValueError to refuse rows that parse but do not
    answer the request.
    """
    try:
        rows = _rows_or_error(body, single=single)
        if check is not None:
            check(rows)
    except AcquisitionError as exc:
        record.update(status="unavailable", category=getattr(exc, "category", "provider_error"), reason=str(exc))
        write_json(query_path, record)
        raise
    except ValueError as exc:
        record.update(status="unavailable", category="malformed_response", reason=str(exc))
        write_json(query_path, record)
        raise AcquisitionError(f"MAST {label} response is incomplete or invalid: {exc}") from exc
    return rows


def fetch_tic_target(tic_id, output_dir, transport):
    """Retain the exact-ID TIC response and reject ambiguous or incomplete matches.

    The accepted response is ``tic_target.json`` and its record
    ``tic_target_query.json`` (fixed names, so a scene resumes from them);
    attempt bodies keep unique names.
    """
    identifier = exact_identifier(tic_id, "TIC ID")
    payload = {"service": TIC_SERVICE, "params": {"columns": "*", "filters": [
        {"paramName": "ID", "values": [identifier]}]}, "format": "json"}
    output = Path(output_dir)
    response = output / "tic_target.json"
    if response.is_symlink():
        raise ValueError("TIC response must be a regular file inside the output directory")
    body, history = _post_tic(payload, output, transport, f"tic_target_response_{uuid4().hex}",
                              response.name, "TIC target query")
    record = {"uri": MAST_INVOKE_URI, "service": TIC_SERVICE, "request": payload,
              "retrieved_utc": _retrieved_utc(), "response": response.name}
    if history:
        record["attempt_history"] = history
    query_path = output / TIC_TARGET_QUERY_FILE

    def same_target(rows):
        if exact_identifier(rows[0].get("ID"), "TIC ID") != identifier:
            raise ValueError("MAST TIC target response has a different ID")

    rows = _parsed_rows(body, record, query_path, single=True, label="TIC target", check=same_target)
    record["status"] = "available"
    write_json(query_path, record)
    return rows[0], record


def fetch_tic_by_gaia(dr2_id, output_dir, transport):
    """Retain every TIC row matching one exact Gaia DR2 identity."""
    return fetch_tic_by_gaia_batch([dr2_id], output_dir, transport)


def fetch_tic_by_gaia_batch(dr2_ids, output_dir, transport):
    """Retain full, scope-checked rows for one bounded set of exact Gaia DR2 IDs."""
    identifiers = [gaia_identifier(value) for value in dr2_ids]
    if not 1 <= len(identifiers) <= 8 or len(set(identifiers)) != len(identifiers):
        raise ValueError("filtered TIC query requires one to eight unique Gaia DR2 identities")
    payload = {"service": TIC_SERVICE, "params": {"columns": "*", "filters": [
        {"paramName": "GAIA", "values": identifiers}]}, "format": "json", "pagesize": 50000}
    return _fetch_tic_rows(payload, output_dir, transport, f"tic_gaia_{identifiers[0]}", "TIC Gaia-identity query",
                           check=lambda rows: _check_gaia_scope(rows, identifiers))


def _check_gaia_scope(rows, identifiers):
    if any(str(row.get("GAIA")) not in identifiers for row in rows):
        raise ValueError("filtered TIC response contains an unrequested Gaia identity")


def fetch_tic_cone(ra_deg, dec_deg, radius_arcsec, output_dir, transport):
    """Retain a complete TIC cone, using degrees for MAST's radius parameter."""
    values = (ra_deg, dec_deg, radius_arcsec)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values) or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90 or radius_arcsec <= 0):
        raise ValueError("TIC cone requires finite ICRS coordinates and a positive radius in arcsec")
    payload = {"service": TIC_CONE_SERVICE,
               "params": {"ra": ra_deg, "dec": dec_deg, "radius": radius_arcsec / 3600.0},
               "format": "json", "pagesize": 50000}
    return _fetch_tic_rows(payload, output_dir, transport, "tic_cone", "TIC cone query")


def _fetch_tic_rows(payload, output_dir, transport, name, label, *, check=None):
    output = Path(output_dir)
    stem = f"{name}_{uuid4().hex}"
    body, history = _post_tic(payload, output, transport, stem, f"{stem}.json", label)
    record = {"uri": MAST_INVOKE_URI, "service": payload["service"], "request": payload,
              "retrieved_utc": _retrieved_utc(), "response": f"{stem}.json"}
    if history:
        record["attempt_history"] = history
    query_path = output / f"{stem}.query.json"
    rows = _parsed_rows(body, record, query_path, single=False, label=label, check=check)
    record["status"] = "available"
    write_json(query_path, record)
    return rows, record


def _finite_positive(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _tic_row_descriptions(matches: List[Mapping[str, Any]]) -> str:
    return ", ".join(
        "TIC {0} disposition {1!r}".format(row.get("ID"), row.get("disposition"))
        for row in matches
    )


def tic_neighbor_entry(rows: List[Mapping[str, Any]], source_id: str) -> Dict[str, Any]:
    """Normalize one source, retaining duplicate-resolution and refusal evidence."""
    matches = [row for row in rows if str(row.get("GAIA")) == source_id]
    if not matches:
        return {"status": "unavailable", "reason": "no TIC row carries this Gaia source_id"}
    if len(matches) > 1:
        usable = [
            row for row in matches
            if not (isinstance(row.get("disposition"), str)
                    and row.get("disposition").strip().upper() in TIC_REJECTED_DISPOSITIONS)
        ]
        if len(usable) != 1:
            return {
                "status": "unavailable",
                "reason": "more than one TIC row carries this Gaia source_id and {0}; refused: {1}".format(
                    DUPLICATE_TIC_ROW_RULE, _tic_row_descriptions(matches)
                ),
                "tic_rows": [
                    {"tic_id": str(row.get("ID")), "disposition": row.get("disposition")}
                    for row in matches
                ],
            }
        row = usable[0]
        duplicate_resolution = {
            "rule": DUPLICATE_TIC_ROW_RULE,
            "selected_tic_id": str(row.get("ID")),
            "rows": [
                {"tic_id": str(item.get("ID")), "disposition": item.get("disposition")}
                for item in matches
            ],
        }
    else:
        row = matches[0]
        duplicate_resolution = None
    entry: Dict[str, Any] = {
        "tic_id": str(row.get("ID")),
        "mass_solar": _finite_positive(row.get("mass")),
        "mass_solar_err": _finite_positive(row.get("e_mass")),
        "radius_solar": _finite_positive(row.get("rad")),
        "radius_solar_err": _finite_positive(row.get("e_rad")),
        "teff_k": _finite_positive(row.get("Teff")),
        "teff_k_err": _finite_positive(row.get("e_Teff")),
        "lumclass": row.get("lumclass"),
        "obj_type": row.get("objType"),
        "disposition": row.get("disposition"),
        "duplicate_id": row.get("duplicate_id"),
        **({"duplicate_resolution": duplicate_resolution} if duplicate_resolution is not None else {}),
    }
    disposition = row.get("disposition")
    if isinstance(disposition, str) and disposition.strip().upper() in TIC_REJECTED_DISPOSITIONS:
        entry.update(status="unavailable", reason="TIC disposition {0!r} is not a usable star".format(disposition))
    elif entry["mass_solar"] is None or entry["radius_solar"] is None:
        entry.update(status="unavailable", reason="TIC row has no finite positive mass and radius")
    else:
        entry["status"] = "available"
    return entry
