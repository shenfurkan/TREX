"""Source-associated composition selection and raw-response provenance."""

import math
from typing import Any, Mapping

from .gspspec import calibrate_gspspec_mh
from .identity import gaia_identifier
from .transport import AcquisitionError

ANDRAE_LIMITATION = (
    "Andrae et al. (2023) XP is refused: its documented full table publishes no per-star "
    "uncertainty, and population scatter is not a per-star measurement"
)
LADDER_LIMITATION = "[M/H] is used as the legacy atmosphere coordinate; its uncertainty is not marginalized"
# SHBoost XP selection, Khalatyan et al. (2024), A&A 691, A98, Appendix A: the
# complete input set (XP spectra, 5-parameter astrometry, G/BP/RP, 2MASS JHKs,
# WISE W1-W4); metallicity output flag 0, which the paper defines as an
# uncertainty below 0.3 dex (enforced on the value too); log g > 7 is the
# paper's white-dwarf region, outside the stellar models used here.
SHBOOST_COMPLETE_INPUT_FLAG = "XP5pGBPRPJHKsW1W2W3W4"
SHBOOST_MET_OUTPUT_FLAG_OK = "0"
SHBOOST_MAX_MET_STD_DEX = 0.3
SHBOOST_WHITE_DWARF_MIN_LOGG = 7.0


def _real(value):
    if isinstance(value, bool) or value is None:
        raise TypeError("not a number")
    return float(value)


def apogee_provenance(source_id: str, selected: Mapping[str, Any], query: Mapping[str, Any]):
    row = selected["row"]
    source_query = query["source_queries"][source_id]
    return {
        "source": "apogee-dr17", "quantity": "[Fe/H]", "value_dex": selected["value_dex"],
        "uncertainty_dex": selected["uncertainty_dex"], "source_id": source_id,
        "retrieved_utc": source_query["retrieved_utc"], "uri": source_query["uri"],
        "apogee_id": row["apogee_id"], "starflag": int(row["starflag"]),
        "fe_h_flag": int(row["fe_h_flag"]), "aspcapflag": int(row["aspcapflag"]),
        "limitation": LADDER_LIMITATION + "; APOGEE [Fe/H] is a calibrated named-tag abundance",
        "binding": {"response": source_query["response"], "row": dict(row)},
    }


def galah_provenance(source_id: str, selected: Mapping[str, Any], query: Mapping[str, Any],
                     artifact: Mapping[str, Any]):
    row = selected["row"]
    release = artifact["release"]
    if release.get("row_space") in ("votable_response", "votable_batches"):
        response = next(item for item in query["forward"]["responses"]
                        if source_id in item["source_ids"] and item.get("status", "available") == "available")
        uri, retrieved = response["uri"], response["retrieved_utc"]
    else:
        uri, retrieved = release["url"], release["retrieved_utc"]
    return {
        "source": "galah-dr3", "quantity": "[Fe/H]", "value_dex": row["fe_h"],
        "uncertainty_dex": row["e_fe_h"], "source_id": source_id, "retrieved_utc": retrieved, "uri": uri,
        "sobject_id": row["sobject_id"], "dr2_source_id": row["dr2_source_id"],
        "flag_sp": row["flag_sp"], "flag_fe_h": row["flag_fe_h"], "snr_c3_iraf": row["snr_c3_iraf"],
        "limitation": LADDER_LIMITATION + "; GALAH DR3 Fe I abundance (Buder et al. 2021)",
        "binding": {"artifact": artifact["path"], "row_index_zero_based": row["row_index_zero_based"]},
    }


def gspspec_provenance(source_id: str, row: Mapping[str, Any], query: Mapping[str, Any]):
    calibrated = calibrate_gspspec_mh(dict(row))
    if calibrated is None:
        return None
    value, uncertainty = calibrated
    source_query = query["source_queries"][source_id]
    return {
        "source": "gaia-gspspec-calibrated", "quantity": "[M/H]", "value_dex": value,
        "uncertainty_dex": uncertainty, "source_id": source_id,
        "retrieved_utc": source_query["retrieved_utc"], "uri": source_query["uri"],
        "flags_gspspec": row["flags_gspspec"], "logg_gspspec": row["logg_gspspec"],
        "calibration": "Recio-Blanco et al. (2023) Eq. 9, third-order [M/H] coefficients",
        "limitation": LADDER_LIMITATION + "; calibrated GSP-Spec [M/H], half the published 16-84 interval",
        "binding": {"response": source_query["response"], "row": dict(row)},
    }


def shboost_provenance(source_id: str, values: Mapping[str, Any], query: Mapping[str, Any],
                       response: str | None, columns: str = "neighbour"):
    try:
        mean = _real(values["xgbdist_met_mean"])
        sigma = _real(values["xgbdist_met_std"])
        logg = _real(values["xgbdist_logg_mean"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (values.get("xgb_inputflag") != SHBOOST_COMPLETE_INPUT_FLAG
            or values.get("xgb_met_outputflag") != SHBOOST_MET_OUTPUT_FLAG_OK
            or not all(math.isfinite(value) for value in (mean, sigma, logg))
            or not 0 < sigma < SHBOOST_MAX_MET_STD_DEX or logg > SHBOOST_WHITE_DWARF_MIN_LOGG
            or not isinstance(response, str) or not response.strip()):
        return None
    return {
        "source": "shboost-xp", "quantity": "[M/H]", "value_dex": mean, "uncertainty_dex": sigma,
        "source_id": source_id, "retrieved_utc": query.get("retrieved_utc"), "uri": query.get("uri"),
        "input_flag": values["xgb_inputflag"], "metallicity_output_flag": values["xgb_met_outputflag"],
        "limitation": LADDER_LIMITATION + "; [M/H], not spectroscopic [Fe/H]; CMD-edge applicability is not numerically classified",
        "binding": {"response": response, "values": dict(values), "columns": columns},
    }


def select_high_rungs(source_ids, *, fetch_apogee, open_galah, fetch_crossmatch,
                      read_galah, retain_galah, fetch_gspspec):
    """Run ordered high-rung lookups using the caller's artifact boundary.

    Each completed rung's record carries ``source_reasons``: why each source it
    examined has no admitted measurement there. A rung that cannot complete stops
    the ladder (never an empty catalogue answer). The GALAH rung also carries the
    targeted query's ``transport`` record (budget, attempts, elapsed seconds,
    stopping bound and failure classification) when it resolved through the live service.
    Returns ``(queries, measured, failure)``.
    """
    from .apogee_dr17 import apogee_dr17_associations
    from .galah_dr3 import galah_dr3_associations

    source_ids = [gaia_identifier(value) for value in source_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("metallicity source identities must be unique")
    queries: dict[str, Any] = {}
    measured: dict[str, dict[str, Any]] = {}
    apogee_record, apogee_rows, failure = fetch_apogee(source_ids)
    queries["apogee"] = apogee_record
    if apogee_rows is None:
        return queries, measured, failure
    apogee_record["source_reasons"] = {}
    for source_id, selected in apogee_dr17_associations(source_ids, apogee_rows).items():
        if selected["status"] == "available":
            measured[source_id] = {"provenance": apogee_provenance(source_id, selected, apogee_record)}
        else:
            apogee_record["source_reasons"][source_id] = selected["reason"]

    remaining = [value for value in source_ids if value not in measured]
    queries["galah"] = {"status": "not-needed"}
    if remaining:
        try:
            release, release_identity = open_galah()
        except (AcquisitionError, OSError, ValueError) as exc:  # download/provider/cache failures defer
            queries["galah"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            return queries, measured, getattr(exc, "network_failure", None)
        neighborhood, forward, reverse, failure = fetch_crossmatch(remaining)
        galah_record = {
            "status": neighborhood["status"],
            "release": {key: release_identity[key] for key in ("url", "retrieved_utc", "bytes")},
            **({"reason": neighborhood["reason"]} if "reason" in neighborhood else {}),
            **{key: neighborhood[key] for key in ("forward", "reverse") if key in neighborhood},
        }
        queries["galah"] = galah_record
        if forward is None or reverse is None:
            return queries, measured, failure
        try:
            rows = read_galah(release, remaining, sorted({row["dr2_source_id"] for row in forward}))
            associations = galah_dr3_associations(remaining, rows, forward, reverse)
        except ValueError as exc:
            galah_record.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            if (transport_record := release_identity.get("transport")) is not None:
                galah_record["transport"] = transport_record
            if release_identity.get("response_batches") or release_identity.get("response_path"):
                galah_record.update(retain_galah(None, release_identity, {}))
            return queries, measured, getattr(exc, "network_failure", None)
        artifact = retain_galah(rows, release_identity, associations)
        if (transport_record := release_identity.get("transport")) is not None:
            galah_record["transport"] = transport_record
        galah_record["release"] = artifact["release"]
        galah_record["row_count"] = len(rows)
        galah_record["artifact"] = artifact
        galah_record["source_reasons"] = {}
        for source_id, selected in associations.items():
            if selected["status"] == "available":
                measured[source_id] = {
                    "provenance": galah_provenance(source_id, selected, galah_record, artifact)}
            else:
                galah_record["source_reasons"][source_id] = selected["reason"]
        remaining = [value for value in source_ids if value not in measured]

    queries["gspspec"] = {"status": "not-needed"}
    if remaining:
        gspspec_record, rows, failure = fetch_gspspec(remaining)
        queries["gspspec"] = gspspec_record
        if rows is None:
            return queries, measured, failure
        # fetch_gspspec refuses repeated source ids, so one row per source.
        by_id = {row["source_id"]: row for row in rows}
        gspspec_record["source_reasons"] = {}
        for source_id in remaining:
            row = by_id.get(source_id)
            provenance = None if row is None else gspspec_provenance(source_id, row, gspspec_record)
            if provenance is not None:
                measured[source_id] = {"provenance": provenance}
            else:
                gspspec_record["source_reasons"][source_id] = (
                    "no GSP-Spec row" if row is None else "outside the published GSP-Spec selection")
    return queries, measured, None


def acquire_high_rungs(output_dir, cache_dir, service, source_ids):
    """Acquire APOGEE, GALAH and GSP-Spec without a pipeline workspace."""
    from .apogee_dr17 import fetch_apogee_dr17
    from .galah_dr3 import fetch_galah_gaia_neighborhood, galah_dr3_rows, galah_shared_release, retain_galah_rows
    from .gspspec import fetch_gspspec

    return select_high_rungs(
        source_ids,
        fetch_apogee=lambda ids: fetch_apogee_dr17(output_dir, service, ids),
        open_galah=lambda: galah_shared_release(cache_dir, service),
        fetch_crossmatch=lambda ids: fetch_galah_gaia_neighborhood(output_dir, service, ids),
        read_galah=galah_dr3_rows,
        retain_galah=lambda rows, identity, associations: retain_galah_rows(output_dir, rows, identity, associations),
        fetch_gspspec=lambda ids: fetch_gspspec(output_dir, service, ids),
    )


def acquire_metallicity(output_dir, cache_dir, service, source_ids):
    """Acquire ordered raw-bound APOGEE, GALAH, GSP-Spec and SHBoost composition."""
    from pathlib import Path
    from uuid import uuid4

    from .host_distributions import shboost_target_values
    from .neighbor_tables import SHBOOST_TABLE, SHBOOST_TARGET_COLUMNS, fetch_neighbor_table
    from .storage import write_json

    output = Path(output_dir)
    ids = [gaia_identifier(value) for value in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("metallicity source identities must be unique")
    record_name = f"metallicity_query_{uuid4().hex}.json"
    queries, measured, failure = acquire_high_rungs(output, Path(cache_dir), service, ids)
    result = {"status": "in-progress", "source_ids": ids, "queries": queries, "record_path": record_name}
    if failure is not None or any(record.get("status") == "unavailable" for record in queries.values()):
        result["status"] = "unavailable"
        if failure is not None:
            result["network_failure"] = failure.to_manifest()
        write_json(output / record_name, result)
        reasons = "; ".join(f"{name}: {record.get('reason', record.get('network_failure', 'unavailable'))}"
                            for name, record in queries.items() if record.get("status") == "unavailable")
        error = AcquisitionError(
            f"higher-priority metallicity query unavailable ({reasons}); record: {output / record_name}",
            retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        raise error
    remaining = [source_id for source_id in ids if source_id not in measured]
    if not remaining:
        result.update(status="available", shboost={"status": "not-needed"},
                      sources=compose_metallicity(ids, queries, measured, {}))
        write_json(output / record_name, result)
        return result
    shboost_query, rows, failure = fetch_neighbor_table(
        output, service, table=SHBOOST_TABLE, columns=SHBOOST_TARGET_COLUMNS, source_ids=remaining,
        response_name=f"metallicity_shboost_{uuid4().hex}.xml", backends=(("gaia-aip", service.AIP_GAIA_TAP_URL),))
    result["shboost"] = shboost_query
    if rows is None:
        result["status"] = "unavailable"
        if failure is not None:
            result["network_failure"] = failure.to_manifest()
        write_json(output / record_name, result)
        error = AcquisitionError("SHBoost metallicity query unavailable", retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        raise error
    by_id = {gaia_identifier(row["source_id"]): row for row in rows}
    inputs = {}
    for source_id in remaining:
        source_query = shboost_query["source_queries"].get(source_id, {})
        row = by_id.get(source_id)
        inputs[source_id] = (shboost_target_values(row), source_query, source_query.get("response"), "target")
    result["sources"] = compose_metallicity(ids, queries, measured, inputs)
    result["status"] = "available"
    write_json(output / record_name, result)
    return result


def compose_metallicity(source_ids, queries, measured, shboost):
    """Return each source's admitted provenance or its explicit unavailable reason.

    ``shboost`` maps a source id to ``(values, query, response, columns)``. The
    reason names, per rung, why this source was not admitted there. An incomplete
    higher rung prevents lower-rung selection, but does not erase earlier measurements.
    """
    results: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        if source_id in measured:
            results[source_id] = {"status": "measured", **measured[source_id]}
            continue
        incomplete = [
            f"{name}: {record.get('reason') or 'query unavailable'}"
            for name in ("apogee", "galah", "gspspec")
            if (record := queries.get(name) or {}).get("status") == "unavailable"
        ]
        if incomplete:
            results[source_id] = {
                "status": "unavailable",
                "reason": "ordered metallicity acquisition incomplete; " + "; ".join(incomplete),
            }
            continue
        values, query, response, columns = shboost.get(source_id, (None, None, None, "neighbour"))
        provenance = None if values is None else shboost_provenance(
            source_id, values, query or {}, response, columns)
        if provenance is not None:
            results[source_id] = {"status": "measured", "provenance": provenance}
            continue
        rejected = []
        for name in ("apogee", "galah", "gspspec"):
            record = queries.get(name) or {}
            if record.get("status") != "available":
                continue
            reason = (record.get("source_reasons") or {}).get(source_id)
            if reason is not None:
                rejected.append(f"{name}: {reason}")
            elif not record.get("row_count"):
                rejected.append(name + ": completed query with no exact association")
            else:
                rejected.append(name + ": no admitted measurement (quality/association)")
        if values is not None:
            rejected.append("shboost-xp: quality flags, uncertainty or response binding")
        results[source_id] = {"status": "unavailable", "reason": "; ".join(rejected + [ANDRAE_LIMITATION])}
    return results
