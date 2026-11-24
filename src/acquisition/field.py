"""Acquire TIC and Gaia field context for a standalone scene."""

from pathlib import Path
from uuid import uuid4

from .gaia_crossmatch import dr2_counterparts, query_gaia_neighborhood
from .gaia_sources import GAIA_CONE_QUERY_FILE, fetch_gaia_sources, normalize_and_match_gaia
from .identity import gaia_identifier
from .storage import write_json
from .tic import TIC_TARGET_QUERY_FILE, fetch_tic_by_gaia_batch, fetch_tic_cone, fetch_tic_target, tic_neighbor_entry
from .transport import AcquisitionError, AcquisitionTransport, NetworkPolicy

FIELD_CONTEXT_FILE = "field_context.json"
TIC_NEIGHBORS_QUERY_FILE = "tic_neighbors_query.json"
# TIC v8 positions are J2000 and Gaia DR3 neighbours' at J2016; the TIC cone
# reaches this far beyond the farthest neighbour so proper motion between the
# two epochs cannot push a neighbour's TIC row out of it [arcsec].
TIC_CONE_EPOCH_MARGIN_ARCSEC = 60.0


def _require_policy(policy):
    if not isinstance(policy, NetworkPolicy):
        raise TypeError("policy must be a NetworkPolicy")


def acquire_field_context(tic_id, output_dir, *, search_radius_arcsec, policy):
    """Retain TIC and Gaia responses and their unique epoch-matched association.

    ``search_radius_arcsec`` must cover the observed aperture footprint; it is
    supplied explicitly rather than treating an arbitrary cone as complete.
    TIC v8 coordinates are at epoch J2000 (Stassun et al. 2019, Sect. 2).
    """
    _require_policy(policy)
    output = Path(output_dir)
    transport = AcquisitionTransport(policy)
    tic, tic_query = fetch_tic_target(tic_id, output, transport)
    try:
        ra, dec = tic["ra"], tic["dec"]
    except KeyError as exc:
        # The TIC response itself was retained as available; the unusable row is recorded here.
        write_json(output / FIELD_CONTEXT_FILE, {"status": "unavailable", "tic_id": str(tic.get("ID")),
                                                 "reason": "TIC target row lacks ICRS coordinates"})
        raise AcquisitionError("TIC target row lacks ICRS coordinates") from exc
    rows, gaia_query = fetch_gaia_sources(ra, dec, search_radius_arcsec, output, transport)
    target, neighbors = normalize_and_match_gaia(rows, ra, dec, 2000.0, match_radius_arcsec=2.0)
    tic_neighbors = acquire_tic_neighbors(target, neighbors, output, policy=policy, centre=(ra, dec),
                                          transport=transport)
    association = {"status": "available", "tic_id": str(tic["ID"]), "target_coordinate_epoch_jyear": 2000.0,
                   "target_coordinate_source": "TIC v8 J2000 (Stassun et al. 2019, Sect. 2)",
                   "search_radius_arcsec": search_radius_arcsec, "target": target, "neighbors": neighbors,
                   "tic_query": TIC_TARGET_QUERY_FILE, "gaia_query": GAIA_CONE_QUERY_FILE,
                   "tic_neighbors_query": TIC_NEIGHBORS_QUERY_FILE}
    write_json(output / FIELD_CONTEXT_FILE, association)
    return tic, target, neighbors, {"tic": tic_query, "gaia": gaia_query, "tic_neighbors": tic_neighbors}


def acquire_tic_neighbors(target, neighbors, output_dir, *, policy, centre=None, transport=None):
    """Retain verified DR3-to-DR2 and TIC rows for each Gaia neighbour.

    TAP outages defer rather than being interpreted as an empty catalogue; an
    ambiguous Gaia association is retained as unavailable and never guessed.

    ``centre`` is the ICRS (ra, dec) [deg] the neighbours' ``separation_arcsec``
    is measured from (the TIC J2000 target position in ``acquire_field_context``);
    the TIC cone uses the same centre. Without it the target row's position is
    used. ``transport`` reuses the caller's ``AcquisitionTransport``.
    """
    _require_policy(policy)
    output = Path(output_dir)
    record_path = output / TIC_NEIGHBORS_QUERY_FILE
    ids = [gaia_identifier(neighbor["source_id"]) for neighbor in neighbors]
    if len(ids) != len(set(ids)):
        raise ValueError("Gaia source identities must be unique")
    transport = transport if transport is not None else AcquisitionTransport(policy)
    query, rows, failure = query_gaia_neighborhood(output, transport, "dr3_source_id", ids, uuid4().hex)
    if rows is None:
        write_json(record_path, {"status": "unavailable", "cross_identification": query})
        reason = failure.reason if failure is not None else query.get("reason", "no response")
        error = AcquisitionError(f"Gaia DR3-to-DR2 neighbourhood query failed: {reason}",
                                 retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        raise error
    try:
        mapping = dr2_counterparts(rows, ids)
    except (ValueError, KeyError, TypeError) as exc:
        write_json(record_path, {"status": "unavailable", "cross_identification": query,
                                 "reason": f"{type(exc).__name__}: {exc}"})
        raise AcquisitionError("Gaia DR3-to-DR2 neighbourhood response is invalid") from exc
    record = {"status": "in-progress", "cross_identification": query, "mapping": mapping, "neighbors": {}}
    cone_rows, cone_query = None, None
    mapped = [neighbor for source_id, neighbor in zip(ids, neighbors) if mapping[source_id]["dr2_source_id"]]
    if mapped:
        # Mapped rows absent from the cone are fetched by exact DR2 identity below.
        ra_deg, dec_deg = centre if centre is not None else (target["ra_deg"], target["dec_deg"])
        try:
            radius = max(float(neighbor["separation_arcsec"]) for neighbor in mapped) + TIC_CONE_EPOCH_MARGIN_ARCSEC
            cone_rows, cone_query = fetch_tic_cone(ra_deg, dec_deg, radius, output, transport)
        except AcquisitionError as exc:
            failure = getattr(exc, "network_failure", None)
            history = getattr(exc, "attempt_history", None)
            if history:
                record["cone_attempt_history"] = history
            if failure is not None and not failure.retryable:
                record.update(status="unavailable", reason=str(exc))
                record["cone_failure"] = failure.to_manifest()
                write_json(record_path, record)
                raise
            record["cone_error"] = str(exc)
            if failure is not None:
                record["cone_failure"] = failure.to_manifest()
    if cone_query is not None:
        record["cone"] = cone_query
    cone_by_gaia = {}
    for row in cone_rows or ():
        cone_by_gaia.setdefault(str(row.get("GAIA")), []).append(row)
    fetched = {}
    missing = list(dict.fromkeys(mapping[sid]["dr2_source_id"] for sid in ids
                                 if mapping[sid]["dr2_source_id"] and mapping[sid]["dr2_source_id"] not in cone_by_gaia))
    try:
        for start in range(0, len(missing), 8):
            chunk = missing[start:start + 8]
            rows, tic_query = fetch_tic_by_gaia_batch(chunk, output, transport)
            for dr2_id in chunk:
                fetched[dr2_id] = (rows, tic_query)
    except AcquisitionError as exc:
        record.update(status="unavailable", reason=str(exc))
        failure = getattr(exc, "network_failure", None)
        if failure is not None:
            record["network_failure"] = failure.to_manifest()
        history = getattr(exc, "attempt_history", None)
        if history:
            record["attempt_history"] = history
        write_json(record_path, record)
        raise
    for source_id in ids:
        association = mapping[source_id]
        dr2_id = association["dr2_source_id"]
        if dr2_id is None:
            record["neighbors"][source_id] = {"status": "unavailable", "reason": association["reason"],
                                              "dr2_source_id": None}
            continue
        if dr2_id in cone_by_gaia:
            tic_rows, tic_query = cone_by_gaia[dr2_id], cone_query
        else:
            tic_rows, tic_query = fetched[dr2_id]
        record["neighbors"][source_id] = {**tic_neighbor_entry(tic_rows, dr2_id),
                                          "dr2_source_id": dr2_id, "query": tic_query}
    # Written once: only an available record is ever resumed, and every failure
    # above writes its own unavailable record before raising.
    record["status"] = "available"
    write_json(record_path, record)
    return record
