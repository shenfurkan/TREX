"""Validate Gaia DR3-to-DR2 neighbourhood associations."""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping
from urllib.parse import urlencode, urlparse

from .identity import gaia_identifier as _gaia_identifier
from .storage import write_bytes
from .transport import AcquisitionFailure
from .votable import parse_votable_bytes

# Gaia DR3 documentation 1.3, section 16.3: true counterparts fall within ~100 mas,
# spurious ones dominate at a few hundred; a resolved pair ~1 arcsec apart lists both.
DR2_COUNTERPART_RADIUS_MAS = 100.0
DR2_NEIGHBOURHOOD_COLUMNS = (
    "dr3_source_id", "dr2_source_id", "angular_distance", "magnitude_difference", "proper_motion_propagation"
)
# Identifiers per TAP GET query: bounds the URL length (about 20 characters per ID).
GAIA_IDS_PER_QUERY = 40


def dr2_counterparts(rows: List[Mapping[str, Any]], source_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Bind a DR3 source to its only DR2 counterpart within 100 mas; retain every row.

    ``source_ids`` are DR3 identifiers (int or decimal string); the mapping is
    keyed by their canonical decimal strings.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {_gaia_identifier(source_id): [] for source_id in source_ids}
    for row in rows:
        dr3_id, dr2_id = (_gaia_identifier(row[name]) for name in DR2_NEIGHBOURHOOD_COLUMNS[:2])
        if dr3_id not in groups:
            raise ValueError("TAP returned a DR3 source outside the requested neighbourhoods")
        distance = row["angular_distance"]
        magnitude = row["magnitude_difference"]
        propagated = row["proper_motion_propagation"]
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not math.isfinite(distance) or distance < 0.0):
            raise ValueError("TAP angular_distance must be finite nonnegative mas")
        if magnitude is not None and (
            isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)) or not math.isfinite(magnitude)
        ):
            raise ValueError("TAP magnitude_difference must be finite mag or null")
        if not isinstance(propagated, bool):
            raise ValueError("TAP proper_motion_propagation must be boolean")
        groups[dr3_id].append({**row, "dr3_source_id": dr3_id, "dr2_source_id": dr2_id})
    mapping = {}
    for source_id, counterparts in groups.items():
        entry: Dict[str, Any] = {"counterparts": counterparts}
        close = [row for row in counterparts if row["angular_distance"] < DR2_COUNTERPART_RADIUS_MAS]
        if len(close) == 1:
            entry.update(status="unique", dr2_source_id=close[0]["dr2_source_id"])
        else:
            entry.update(
                status="unavailable", dr2_source_id=None,
                reason="no DR2 counterpart within 100 mas in gaiadr3.dr2_neighbourhood" if not close
                else "ambiguous DR3-to-DR2 association: several counterparts within 100 mas; none selected",
            )
        mapping[source_id] = entry
    return mapping


def dr4_to_dr2_identity_chain(dr4_id, dr3_candidates, dr2_rows):
    """Compose a release-qualified link without reusing DR3's separation cut for DR4.

    The caller must supply DR3 candidates from a complete DR4 association
    query for this exact DR4 source, after that release's own admission;
    this function does not claim an unreleased ADQL schema or separation rule.
    Neither missing nor ambiguous links are replaced by identifier equality.
    """
    dr4_id = _gaia_identifier(dr4_id)
    candidates = [_gaia_identifier(value) for value in dr3_candidates]
    chain = {"DR4": dr4_id, "DR3": None, "DR2": None, "dr3_candidates": candidates}
    if len(candidates) != 1:
        return {**chain, "status": "unavailable",
                "reason": "ambiguous DR4-to-DR3 association" if candidates else "no DR4-to-DR3 association"}
    dr3_id, = candidates
    link = dr2_counterparts(dr2_rows, [dr3_id])[dr3_id]
    return {**chain, "DR3": dr3_id, "DR2": link["dr2_source_id"], **{
        key: value for key, value in link.items() if key != "dr2_source_id"
    }}


def _parsed_rows(body, field, chunk):
    """Rows of one complete neighbourhood VOTable; ValueError when it is not."""
    try:
        parsed = parse_votable_bytes(body)
    except Exception as exc:
        raise ValueError(f"Gaia neighbourhood response is incomplete: {exc}") from exc
    if list(parsed["table"].colnames) != list(DR2_NEIGHBOURHOOD_COLUMNS):
        raise ValueError("Gaia neighbourhood response is incomplete or has unexpected columns")
    for column in DR2_NEIGHBOURHOOD_COLUMNS[:2]:
        dtype = parsed["table"][column].dtype
        if dtype.kind != "i" or dtype.itemsize != 8:
            raise ValueError("Gaia neighbourhood identifiers must be signed 64-bit integers")
    rows = [dict(zip(DR2_NEIGHBOURHOOD_COLUMNS, row)) for row in parsed["data"]]
    if any(_gaia_identifier(row[field]) not in chunk for row in rows):
        raise ValueError("Gaia neighbourhood returned an unrequested source")
    return rows


def query_gaia_neighborhood(output_dir, service, field, source_ids, retrieval_id):
    """Query and retain complete Gaia DR3/DR2 cross-match responses.

    ``field`` is the column the identifiers are matched on (``dr3_source_id`` or
    ``dr2_source_id``). Each batch tries the ESA archive, then the AIP mirror;
    a mirror is also tried when the first answers HTTP 200 with an incomplete
    or malformed VOTable, and every received body is retained. Each response's
    status distinguishes accepted rows from failed bodies kept for diagnosis,
    so replay and measurement provenance select the successful mirror. Returns
    ``(record, rows, failure)``; ``rows`` is None when no mirror gave a complete
    answer, with ``failure`` describing the last one (a malformed answer is a
    retryable provider failure). ``service`` provides ``_http_get_bytes``,
    ``transport_failures`` and the two TAP URLs.
    """
    if field not in DR2_NEIGHBOURHOOD_COLUMNS[:2]:
        raise ValueError("Gaia neighbourhood queries match dr3_source_id or dr2_source_id")
    output_dir = Path(output_dir)
    source_ids = [_gaia_identifier(value) for value in source_ids]
    record = {"status": "available" if source_ids else "not-needed", "responses": [], "source_ids": source_ids}
    if not source_ids:
        return record, [], None
    rows = []
    for index, start in enumerate(range(0, len(source_ids), GAIA_IDS_PER_QUERY)):
        chunk = source_ids[start:start + GAIA_IDS_PER_QUERY]
        query = "SELECT {0} FROM gaiadr3.dr2_neighbourhood WHERE {1} IN ({2})".format(
            ", ".join(DR2_NEIGHBOURHOOD_COLUMNS), field, ", ".join(chunk))
        arguments = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query})
        chunk_rows, failure, reason = None, None, f"Gaia {field} query failed for batch {index}"
        for mirror, backend in enumerate((service.ESA_GAIA_TAP_URL, service.AIP_GAIA_TAP_URL)):
            url = backend + "?" + arguments
            before = len(service.transport_failures)
            body = service._http_get_bytes(url, "application/xml")
            if body is None:
                if len(service.transport_failures) > before:
                    failure = service.transport_failures[-1]
                    if not failure.retryable:
                        break
                continue
            name = f"gaia_dr2_neighbourhood_{retrieval_id}_{field}_part{index:02d}_mirror{mirror}.xml"
            write_bytes(output_dir / name, body)
            record["responses"].append({
                "path": name, "uri": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source_ids": chunk,
            })
            try:
                chunk_rows = _parsed_rows(body, field, chunk)
                record["responses"][-1]["status"] = "available"
                break
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                record["responses"][-1].update(status="unavailable", reason=reason)
                failure = AcquisitionFailure(urlparse(backend).hostname or backend, "provider", True, 1,
                                             reason, None, type(exc).__name__)
        if chunk_rows is None:
            record.update(status="unavailable", reason=reason)
            if failure is not None:
                record["network_failure"] = failure.to_manifest()
            return record, None, failure
        rows.extend(chunk_rows)
    return record, rows, None
