"""Parse and select exact-Gaia-ID APOGEE DR17 abundance responses."""

import csv
import io
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from .identity import gaia_identifier as _gaia_identifier
from .storage import write_bytes

APOGEE_DR17_URI = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
APOGEE_DR17_COLUMNS = (
    "gaiaedr3_source_id", "apogee_id", "starflag", "fe_h", "fe_h_err", "fe_h_flag", "aspcapflag",
)
# Gaia IDs per SkyServer GET query: bounds the URL length (about 20 characters per ID).
APOGEE_IDS_PER_QUERY = 40


def parse_apogee_count(body: bytes) -> int:
    """Read the single aggregate row needed to detect truncated CSV output."""
    try:
        rows = [row for row in csv.reader(io.StringIO(body.decode("utf-8-sig"), newline=""), strict=True) if row]
        if (len(rows) != 3 or rows[:2] != [["#Table1"], ["row_count"]] or len(rows[2]) != 1
                or not rows[2][0].isascii() or not rows[2][0].isdecimal()):
            raise ValueError("APOGEE response has no complete count result")
        return int(rows[2][0])
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("APOGEE count response is not a complete CSV table") from exc


def parse_apogee_table(body: bytes) -> list[dict[str, str]]:
    """Return every row of a complete SkyServer CSV table with exact identifiers."""
    try:
        rows = csv.reader(io.StringIO(body.decode("utf-8-sig"), newline=""), strict=True)
        if next(rows, None) != ["#Table1"] or next(rows, None) != list(APOGEE_DR17_COLUMNS):
            raise ValueError("APOGEE response has no expected SkyServer CSV table")
        result = []
        for row in rows:
            if not row:
                continue
            if len(row) != len(APOGEE_DR17_COLUMNS):
                raise ValueError("APOGEE response has a malformed or incomplete row")
            entry = dict(zip(APOGEE_DR17_COLUMNS, row))
            entry["gaiaedr3_source_id"] = _gaia_identifier(entry["gaiaedr3_source_id"])
            result.append(entry)
        return result
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("APOGEE response is not a complete CSV table") from exc


def parse_apogee_dr17(body: bytes, source_ids: list[str], expected_count: int) -> list[dict[str, str]]:
    """Accept a complete SkyServer CSV result, including an empty result, without rounding IDs.

    Every row must belong to a requested Gaia source and the row total must
    equal the separate COUNT(*) answer, so a truncated CSV is refused.
    """
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
        raise ValueError("APOGEE row count must be a nonnegative integer")
    requested = {_gaia_identifier(value) for value in source_ids}
    rows = parse_apogee_table(body)
    if any(row["gaiaedr3_source_id"] not in requested for row in rows):
        raise ValueError("APOGEE response contains an unrequested Gaia source")
    if len(rows) != expected_count:
        raise ValueError("APOGEE row response does not match its completed count query")
    return rows


def apogee_dr17_associations(source_ids: list[str], rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Select an unambiguous, quality-approved DR17 named iron abundance per Gaia source."""
    ids = [_gaia_identifier(value) for value in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Gaia source identities must be unique")
    grouped: dict[str, list[dict[str, str]]] = {source_id: [] for source_id in ids}
    for row in rows:
        source_id = _gaia_identifier(row["gaiaedr3_source_id"])
        if source_id not in grouped:
            raise ValueError("APOGEE rows contain an unrequested Gaia source")
        grouped[source_id].append(row)
    result: dict[str, dict[str, Any]] = {}
    for source_id, matches in grouped.items():
        if not matches:
            result[source_id] = {"status": "unavailable", "reason": "no APOGEE DR17 abundance row"}
            continue
        # Several APOGEE stars/visits on one Gaia ID give no defensible single abundance.
        if len(matches) != 1:
            result[source_id] = {"status": "unavailable", "reason": "ambiguous APOGEE DR17 rows"}
            continue
        row = matches[0]
        try:
            value, error = float(row["fe_h"]), float(row["fe_h_err"])
            flags = [int(row[key]) for key in ("starflag", "fe_h_flag", "aspcapflag")]
        except (KeyError, ValueError, TypeError, OverflowError):
            result[source_id] = {"status": "unavailable", "reason": "APOGEE iron value, uncertainty or flags missing"}
            continue
        # SDSS DR17 recommends named FE_H and documents -9999-class missing values and quality bitmasks.
        if (not row.get("apogee_id", "").strip() or not math.isfinite(value) or value <= -6
                or not math.isfinite(error) or error <= 0 or flags != [0, 0, 0]):
            result[source_id] = {"status": "unavailable", "reason": "APOGEE iron quality, uncertainty or flags"}
            continue
        result[source_id] = {"status": "available", "row": row, "value_dex": value, "uncertainty_dex": error}
    return result


def fetch_apogee_dr17(output_dir: Path, service: Any, source_ids: list[str]):
    """Query and retain batches; an unavailable response is never an empty result.

    Each batch sends a COUNT(*) query and a row query: SkyServer CSV has no
    completeness marker, so the count is what detects a truncated row answer.
    A failed or malformed batch makes the whole result unavailable (the rows
    of earlier batches are not returned as a partial answer); the responses
    already retained stay listed in the record. ``service`` provides
    ``_http_get_bytes`` and ``transport_failures`` (AcquisitionTransport or a
    caller's archive service with the same seam).
    """
    output_dir = Path(output_dir)
    source_ids = [_gaia_identifier(value) for value in source_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("APOGEE source identities must be unique")
    record = {"status": "available" if source_ids else "not-needed", "columns": list(APOGEE_DR17_COLUMNS),
              "responses": [], "source_queries": {}}
    if not source_ids:
        return record, [], None
    all_rows = []
    retrieval_id = uuid4().hex
    for index, start in enumerate(range(0, len(source_ids), APOGEE_IDS_PER_QUERY)):
        chunk = source_ids[start:start + APOGEE_IDS_PER_QUERY]
        source_query = (
            " FROM apogeeStar AS s JOIN aspcapStar AS a "
            "ON a.apstar_id=s.apstar_id WHERE s.gaiaedr3_source_id IN ("
            + ",".join(chunk) + ")"
        )
        queries = {
            "count": "SELECT COUNT(*) AS row_count" + source_query,
            "rows": ("SELECT s.gaiaedr3_source_id,s.apogee_id,s.starflag,a.fe_h,a.fe_h_err,"
                     "a.fe_h_flag,a.aspcapflag" + source_query),
        }
        answers = {}
        for role, query in queries.items():
            url = APOGEE_DR17_URI + "?" + urlencode({"cmd": query, "format": "csv"})
            failure_count = len(service.transport_failures)
            body = service._http_get_bytes(url, "text/csv")
            if body is None:
                failure = service.transport_failures[-1] if len(service.transport_failures) > failure_count else None
                record.update(status="unavailable", reason=f"APOGEE {role} request failed for batch {index}", uri=url)
                if failure is not None:
                    record["network_failure"] = failure.to_manifest()
                return record, None, failure
            retrieved_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            name = f"apogee_dr17_{retrieval_id}_part{index:02d}_{role}.csv"
            write_bytes(output_dir / name, body)
            artifact = {"path": name, "uri": url,
                        "retrieved_utc": retrieved_utc, "role": role, "chunk": index}
            record["responses"].append(artifact)
            answers[role] = (body, artifact)
        try:
            count = parse_apogee_count(answers["count"][0])
            all_rows.extend(parse_apogee_dr17(answers["rows"][0], chunk, count))
        except ValueError as exc:
            record.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            return record, None, None
        for source_id in chunk:
            record["source_queries"][source_id] = {
                "uri": answers["rows"][1]["uri"], "retrieved_utc": answers["rows"][1]["retrieved_utc"],
                "chunk": index, "response": answers["rows"][1]["path"],
            }
    record["row_count"] = len(all_rows)
    return record, all_rows, None
