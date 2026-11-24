"""Calibrated Gaia DR3 GSP-Spec metallicity selection."""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from trex.runtime.logging import host_logger

from .identity import gaia_identifier
from .storage import write_bytes
from .votable import parse_votable_bytes

GSP_SPEC_TABLE = "gaiadr3.astrophysical_parameters"
GSP_SPEC_COLUMNS = (
    "source_id", "teff_gspspec", "teff_gspspec_lower", "teff_gspspec_upper",
    "logg_gspspec", "logg_gspspec_lower", "logg_gspspec_upper",
    "mh_gspspec", "mh_gspspec_lower", "mh_gspspec_upper", "flags_gspspec",
)

# Recio-Blanco et al. (2023), A&A 674, A29, Eq. 9 and Table 3,
# DOI 10.1051/0004-6361/202243750: third-order [M/H] calibration.
GSP_SPEC_MH_COEFFICIENTS = (0.274, -0.1373, -0.0050, 0.0048)
GSP_SPEC_MIN_TEFF_K = 3500.0
GSP_SPEC_MAX_LOGG = 5.0
GSP_SPEC_MAX_TEFF_WIDTH_K = 750.0
GSP_SPEC_MAX_LOGG_WIDTH = 1.0
GSP_SPEC_MAX_MH_WIDTH_DEX = 0.5
# Gaia source IDs per TAP GET query: bounds the URL length (about 20 characters per ID).
GSP_SPEC_IDS_PER_QUERY = 40
# Zero-based positions in ``flags_gspspec`` (Recio-Blanco et al. 2023, Table 2,
# characters 1-13) and the values the Appendix G selection accepts.
GSP_SPEC_FLAG_LIMITS = (
    (0, "vbroadT", "01"), (1, "vbroadG", "01"), (2, "vbroadM", "01"),
    (3, "vradT", "01"), (4, "vradG", "01"), (5, "vradM", "01"),
    (6, "fluxNoise", "0123"), (7, "extrapol", "012"), (12, "KMgiantPar", "01"),
)


def parse_gspspec_table(body: bytes) -> list[dict[str, Any]]:
    """Read every GSP-Spec row from a complete VOTable."""
    try:
        parsed = parse_votable_bytes(body)
    except Exception as exc:
        host_logger(__name__).warning("GSP-Spec response was not a complete VOTable: %s", exc)
        raise ValueError(f"GSP-Spec response is not a complete VOTable: {type(exc).__name__}: {exc}") from exc
    if list(parsed["table"].colnames) != list(GSP_SPEC_COLUMNS):
        raise ValueError("GSP-Spec response is incomplete or has unexpected columns")
    rows = []
    for row in parsed["data"]:
        entry = dict(zip(GSP_SPEC_COLUMNS, row))
        entry["source_id"] = gaia_identifier(entry["source_id"])
        rows.append(entry)
    return rows


def parse_gspspec(body: bytes, source_ids: list[str]) -> list[dict[str, Any]]:
    """Read exact-source GSP-Spec rows; reject unrequested or repeated identities."""
    requested = {gaia_identifier(value) for value in source_ids}
    rows = parse_gspspec_table(body)
    if any(row["source_id"] not in requested for row in rows):
        raise ValueError("GSP-Spec response contains an unrequested Gaia source")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("GSP-Spec returned duplicate source identities")
    return rows


def fetch_gspspec(output_dir: Path, service: Any, source_ids: list[str]):
    """Retain every completed exact-source response; failure is never an empty result."""
    output_dir = Path(output_dir)
    ids = [gaia_identifier(value) for value in source_ids]
    record = {"table": GSP_SPEC_TABLE, "columns": list(GSP_SPEC_COLUMNS),
              "status": "available" if ids else "not-needed", "responses": [], "source_queries": {}}
    if not ids:
        return record, [], None
    retrieval_id = uuid4().hex
    rows = []
    for index, start in enumerate(range(0, len(ids), GSP_SPEC_IDS_PER_QUERY)):
        chunk = ids[start:start + GSP_SPEC_IDS_PER_QUERY]
        query = "SELECT {0} FROM {1} WHERE source_id IN ({2})".format(
            ", ".join(GSP_SPEC_COLUMNS), GSP_SPEC_TABLE, ", ".join(chunk)
        )
        arguments = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query})
        chunk_rows, failure = None, None
        for mirror, backend in enumerate((service.ESA_GAIA_TAP_URL, service.AIP_GAIA_TAP_URL)):
            url = backend + "?" + arguments
            before = len(service.transport_failures)
            body = service._http_get_bytes(url, "application/xml")
            if body is not None:
                retrieved_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                suffix = "" if mirror == 0 else f"_mirror{mirror}"
                name = f"gspspec_{retrieval_id}_part{index:02d}{suffix}.xml"
                write_bytes(output_dir / name, body)
                artifact = {"path": name, "uri": url, "retrieved_utc": retrieved_utc, "chunk": index}
                record["responses"].append(artifact)
                try:
                    chunk_rows = parse_gspspec(body, chunk)
                except (ValueError, KeyError, TypeError) as exc:
                    record.setdefault("backend_failures", []).append({
                        **artifact, "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                break
            if len(service.transport_failures) > before:
                failure = service.transport_failures[-1]
                if not failure.retryable:
                    break
        if chunk_rows is None:
            record.update(status="unavailable", reason=f"GSP-Spec query failed for batch {index}")
            if failure is not None:
                record["network_failure"] = failure.to_manifest()
            return record, None, failure
        rows.extend(chunk_rows)
        for source_id in chunk:
            record["source_queries"][source_id] = {
                "uri": url, "retrieved_utc": retrieved_utc, "chunk": index,
                "response": artifact["path"],
            }
    record["row_count"] = len(rows)
    return record, rows, None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def calibrate_gspspec_mh(row: dict[str, Any]) -> tuple[float, float] | None:
    """Return calibrated [M/H] and half the 16-84 width after the published selection."""
    values = {name: _number(row.get(name)) for name in (
        "teff_gspspec", "teff_gspspec_lower", "teff_gspspec_upper",
        "logg_gspspec", "logg_gspspec_lower", "logg_gspspec_upper",
        "mh_gspspec", "mh_gspspec_lower", "mh_gspspec_upper",
    )}
    if any(value is None for value in values.values()):
        return None
    flags = row.get("flags_gspspec")
    if not isinstance(flags, str) or len(flags) < 13 or any(char not in "0123456789" for char in flags):
        return None
    teff, teff_lower, teff_upper = (values[f"teff_gspspec{s}"] for s in ("", "_lower", "_upper"))
    logg, logg_lower, logg_upper = (values[f"logg_gspspec{s}"] for s in ("", "_lower", "_upper"))
    mh, mh_lower, mh_upper = (values[f"mh_gspspec{s}"] for s in ("", "_lower", "_upper"))
    # An inverted 16-84 interval has a negative width and would pass the width cuts.
    # Only [M/H] must also bracket its value (the calibrated quantity); the Teff
    # and logg values are not required to lie inside their own intervals.
    if teff_upper < teff_lower or logg_upper < logg_lower or not mh_lower <= mh <= mh_upper:
        return None
    # Recio-Blanco et al. (2023), A&A 674, A29, Appendix G query.
    selected = (
        teff > GSP_SPEC_MIN_TEFF_K and 0.0 < logg < GSP_SPEC_MAX_LOGG
        and teff_upper - teff_lower < GSP_SPEC_MAX_TEFF_WIDTH_K
        and logg_upper - logg_lower < GSP_SPEC_MAX_LOGG_WIDTH
        and mh_upper - mh_lower < GSP_SPEC_MAX_MH_WIDTH_DEX
        and (teff >= 3800.0 or logg <= 3.5)
        and (teff >= 4150.0 or logg <= 2.4 or logg >= 3.6)
        and all(flags[index] in accepted for index, _name, accepted in GSP_SPEC_FLAG_LIMITS)
    )
    if not selected:
        return None
    p0, p1, p2, p3 = GSP_SPEC_MH_COEFFICIENTS
    calibrated = mh + p0 + p1 * logg + p2 * logg**2 + p3 * logg**3
    uncertainty = 0.5 * (mh_upper - mh_lower)
    if not math.isfinite(calibrated) or not math.isfinite(uncertainty) or uncertainty <= 0.0:
        return None
    return calibrated, uncertainty
