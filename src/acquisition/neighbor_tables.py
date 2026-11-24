"""Retained, exact-source Gaia and SHBoost table acquisition."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from .host_distributions import gaia_ap_entry, shboost_entry, starhorse_entry
from .identity import gaia_identifier
from .storage import write_bytes, write_json
from .transport import AcquisitionError, request_user_agent
from .votable import parse_votable_bytes

SHBOOST_TABLE = "gaiadr3_contrib.shboost2024"
GAIA_DR3_AP_TABLE = "gaiadr3.astrophysical_parameters"
STARHORSE_EDR3_TABLE = "gaiaedr3_contrib.starhorse_1_1"
SHBOOST_COLUMNS = (
    "source_id", "xgbdist_mass_mean", "xgbdist_mass_std", "xgbdist_logg_mean",
    "xgbdist_logg_std", "xgbdist_logteff_mean", "xgbdist_logteff_std",
    "xgbdist_met_mean", "xgbdist_met_std", "xgbdist_av_mean", "xgb_inputflag",
    "xgb_mass_outputflag", "xgb_logg_outputflag", "xgb_logteff_outputflag", "xgb_met_outputflag",
)
SHBOOST_TARGET_COLUMNS = (
    "source_id", "xgbdist_met_mean", "xgbdist_met_std", "xgb_inputflag",
    "xgb_met_outputflag", "xgbdist_logteff_mean", "xgbdist_logg_mean", "bprp0", "mg0",
)
GAIA_AP_COLUMNS = (
    "source_id", "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper",
    "radius_gspphot", "radius_gspphot_lower", "radius_gspphot_upper",
    "distance_gspphot", "mass_flame", "mass_flame_lower", "mass_flame_upper",
    "radius_flame", "radius_flame_lower", "radius_flame_upper",
    "flags_flame", "evolstage_flame", "age_flame",
)
STARHORSE_COLUMNS = (
    "source_id", "mass16", "mass50", "mass84", "teff16", "teff50", "teff84",
    "logg16", "logg50", "logg84", "dist16", "dist50", "dist84", "av50",
    "fidelity", "sh_photoflag", "sh_outflag", "bprp0", "mg0",
)
# Gaia IDs per TAP GET query: bounds the URL length (about 20 characters per ID).
# SHBoost is queried one source at a time: the AIP service rejects its IN-list
# queries with HTTP 400 (then the UWS asynchronous endpoint is tried).
NEIGHBOR_QUERY_CHUNK_SIZE = 40


def _neighbor_rows(body, columns, source_ids):
    """Rows of one complete VOTable whose ids are unique and all in ``source_ids``; ValueError otherwise."""
    try:
        parsed = parse_votable_bytes(body)
    except Exception as exc:
        raise ValueError(f"neighbour parameter response failed complete-VOTable validation: {exc}") from exc
    if list(parsed["table"].colnames) != list(columns):
        raise ValueError("neighbour parameter response has unexpected columns: {0}".format(list(parsed["table"].colnames)))
    rows = [dict(zip(columns, row)) for row in parsed["data"]]
    returned_ids = [gaia_identifier(row["source_id"]) for row in rows]
    if len(set(returned_ids)) != len(returned_ids) or not set(returned_ids).issubset(source_ids):
        raise ValueError("neighbour table returned duplicate or unrequested source identities")
    return rows


def _retained_neighbor_responses(output, query, table, columns, source_ids):
    """Validate a retained query's complete responses and reparse their bytes.

    Only responses bound to a requested source are returned; a response whose
    source is not requested stays untouched in the old record. Every bound
    response is still checked, so damaged retained evidence is refused rather
    than silently replaced. A source with a valid response but no row is
    covered: absence from the table is evidence, not missing work.
    """
    if query is None:
        return [], [], {}, None
    if not isinstance(query, dict):
        raise ValueError("retained neighbour table record is malformed")
    if query.get("table") != table or query.get("columns") != list(columns):
        raise ValueError("retained neighbour table release or columns differ from the request")
    responses = query.get("responses")
    source_queries = query.get("source_queries")
    if not isinstance(responses, list) or not isinstance(source_queries, dict):
        raise ValueError("retained neighbour table has no per-source response record")
    if not responses:
        if source_queries:
            raise ValueError("retained neighbour table has source queries without responses")
        return [], [], {}, None
    requested = set(source_ids)
    by_name = {}
    for response in responses:
        if (not isinstance(response, dict) or not isinstance(response.get("path"), str)
                or not isinstance(response.get("sha256"), str)):
            raise ValueError("retained neighbour response binding is malformed")
        name = response["path"]
        if Path(name).name != name:
            raise ValueError("retained neighbour responses must be plain filenames inside the output directory")
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError("retained neighbour responses must be regular files inside the output directory")
        if name in by_name:
            raise ValueError("retained neighbour response is duplicated")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != response["sha256"]:
            raise ValueError(f"retained neighbour response changed: {name}")
        by_name[name] = (response, body)
    chunks, bound = {}, {}
    for source_id, binding in source_queries.items():
        if not isinstance(binding, dict) or binding.get("response") not in by_name:
            raise ValueError("retained neighbour source query does not bind a retained response")
        # A response is validated against every source it was requested for,
        # then only the currently requested sources' rows are reused.
        bound.setdefault(binding["response"], []).append(source_id)
        if source_id in requested:
            chunks.setdefault(binding["response"], []).append(source_id)
    retained_responses, retained_source_queries, rows = [], {}, []
    for name, (response, body) in by_name.items():
        chunk = chunks.get(name)
        if not chunk:
            continue
        retained_responses.append(response)
        retained_source_queries.update({source_id: source_queries[source_id] for source_id in chunk})
        try:
            parsed_rows = _neighbor_rows(body, columns, bound[name])
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise ValueError(f"retained neighbour response {name} is not reusable: {exc}") from exc
        wanted = set(chunk)
        rows.extend(row for row in parsed_rows if gaia_identifier(row["source_id"]) in wanted)
    if not retained_responses:
        return [], [], {}, None
    previous = {
        "status": query.get("status"),
        "reason": query.get("reason"),
        "query": query.get("query"),
        "uri": query.get("uri"),
        "backend": query.get("backend"),
        "backends": list(query.get("backends") or []),
        "failed_backends": list(query.get("failed_backends") or []),
        "responses": len(responses),
        "source_queries": len(source_queries),
        "reused_responses": len(retained_responses),
        "reused_sources": len(retained_source_queries),
    }
    return rows, retained_responses, retained_source_queries, previous


def fetch_neighbor_table(output_dir, service, *, table, columns, source_ids, response_name, backends,
                         async_fetch=None, retained_query=None):
    """Retain complete source-associated table responses or explicitly defer.

    ``retained_query`` is a previous record for this exact table and column
    list, with response paths relative to ``output_dir``. Its complete
    responses are reparsed from their original VOTable bytes and reused, so a
    retry requests only the source chunks those responses do not cover. The
    new record keeps the retained bindings and their provenance beside a
    ``retained_from`` block describing the previous attempt.

    Every received body is written to ``output_dir``; only a body that
    validates is bound in ``responses``/``source_queries`` (a malformed one is
    listed in ``rejected_responses``), so a later retry can reuse the record.
    ``backends`` is a non-empty sequence of ``(name, TAP sync URL)``.
    """
    output_dir = Path(output_dir)
    if async_fetch is None:
        async_fetch = fetch_shboost_async
    if Path(response_name).name != response_name or not response_name.endswith(".xml"):
        raise ValueError("response name must be a plain .xml filename")
    if not backends:
        raise ValueError("neighbour table acquisition needs at least one backend")
    record = {"table": table, "query": None, "columns": list(columns), "uri": backends[0][1],
              "status": "not-needed" if not source_ids else "available", "backend": None, "backends": [],
              "failed_backends": [], "response": None, "responses": [], "chunk_size": NEIGHBOR_QUERY_CHUNK_SIZE,
              "source_queries": {}}
    if not source_ids:
        return record, [], None
    source_ids = [gaia_identifier(value) for value in source_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("neighbour table source identities must be unique")
    rows, last_failure, unavailable_backends = [], None, []
    retained_rows, retained_responses, retained_source_queries, previous = _retained_neighbor_responses(
        output_dir, retained_query, table, columns, source_ids)
    if previous is not None:
        record.update(query=previous["query"], uri=previous["uri"] or record["uri"],
                      backend=previous["backend"], retained_from=previous)
    record["responses"] = retained_responses
    record["response"] = retained_responses[0] if retained_responses else None
    record["source_queries"] = retained_source_queries
    rows.extend(retained_rows)
    missing = [source_id for source_id in source_ids if source_id not in retained_source_queries]
    # A retry must not overwrite a retained response, so its chunk index
    # continues after the highest retained one.
    chunk_offset = max(
        (binding.get("chunk") for binding in retained_source_queries.values()
         if isinstance(binding.get("chunk"), int) and not isinstance(binding.get("chunk"), bool)),
        default=-1,
    ) + 1
    used_names = {response["path"] for response in retained_responses}
    # Large AIP SHBoost IN lists timed out. Keep the batching bounded and
    # retain the known exact-ID fallback for a service that rejects a group.
    chunk_size = 8 if table == SHBOOST_TABLE else NEIGHBOR_QUERY_CHUNK_SIZE
    record["chunk_size"] = chunk_size
    chunks = [missing[start:start + chunk_size] for start in range(0, len(missing), chunk_size)]
    for position, chunk in enumerate(chunks):
        chunk_index = chunk_offset + position
        if table == SHBOOST_TABLE and len(chunk) == 1:
            query = f"SELECT {','.join(columns)} FROM {table} WHERE source_id={chunk[0]}"
        else:
            query = f"SELECT {', '.join(columns)} FROM {table} WHERE source_id IN ({', '.join(chunk)})"
        arguments = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query})
        record["query"] = query
        body = None
        split_chunk = False
        for backend_name, base_url in backends:
            if backend_name in unavailable_backends:
                continue
            url = base_url + "?" + arguments
            accepted_backend = backend_name
            before = len(service.transport_failures)
            body = service._http_get_bytes(url, "application/xml")
            record["uri"] = url
            backend = {"backend": backend_name, "chunk": chunk_index, "uri": url,
                       "status": "available" if body is not None else "unavailable"}
            # Decisions below use this attempt's failure only, never one left by an earlier chunk.
            attempt_failure = None
            if body is None and len(service.transport_failures) > before:
                attempt_failure = last_failure = service.transport_failures[-1]
                backend.update(attempt_failure.to_manifest())
                unavailable_backends.append(backend_name)
                if backend_name not in record["failed_backends"]:
                    record["failed_backends"].append(backend_name)
            record["backends"].append(backend)
            if (body is None and table == SHBOOST_TABLE and len(chunk) > 1 and attempt_failure is not None
                    and (attempt_failure.category == "read_timeout"
                         or (attempt_failure.category == "http_status" and attempt_failure.http_status == 400))):
                unavailable_backends.remove(backend_name)
                chunks.extend([source_id] for source_id in chunk)
                backend["recovery"] = "retry-exact-source"
                split_chunk = True
                break
            # A read timeout and an HTTP 400 refusal both fail the sync connection, not the query.
            if (body is None and table == SHBOOST_TABLE and attempt_failure is not None
                    and (attempt_failure.category == "read_timeout"
                         or (attempt_failure.category == "http_status" and attempt_failure.http_status == 400))
                    and async_fetch):
                async_record = {"backend": "gaia-aip-async", "chunk": chunk_index, "uri": base_url}
                try:
                    body, async_url = async_fetch(service, query)
                except AcquisitionError:
                    raise
                except (RuntimeError, ValueError, TimeoutError) as exc:
                    async_record.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
                else:
                    record["uri"] = async_url
                    accepted_backend = "gaia-aip-async"
                    if backend_name in unavailable_backends:
                        unavailable_backends.remove(backend_name)
                    async_record.update(status="available", uri=async_url)
                record["backends"].append(async_record)
            if body is not None:
                record["backend"] = accepted_backend
                break
            if attempt_failure is not None and not attempt_failure.retryable:
                break
        if split_chunk:
            continue
        if body is None:
            record["status"] = "unavailable"
            record["reason"] = "no backend returned a response for chunk {0} ({1})".format(
                chunk_index, ", ".join(record["failed_backends"]) or "none")
            return record, None, last_failure
        source_query = {"uri": record["uri"], "backend": record["backend"], "chunk": chunk_index,
                        "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        name = (response_name if len(chunks) == 1 and not retained_responses
                else response_name.replace(".xml", "_part{0:02d}.xml".format(chunk_index)))
        if name in used_names:
            raise ValueError("neighbour response name collides with a retained response")
        used_names.add(name)
        write_bytes(output_dir / name, body)
        response = {"path": name, "sha256": hashlib.sha256(body).hexdigest()}
        try:
            rows.extend(_neighbor_rows(body, columns, chunk))
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            record["status"] = "unavailable"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("rejected_responses", []).append({**response, "source_ids": chunk})
            return record, None, None
        for source_id in chunk:
            record["source_queries"][source_id] = {**source_query, "response": name}
        record["responses"].append(response)
        if len(record["responses"]) == 1:
            record["response"] = record["responses"][0]
    record["status"] = "available"
    record["row_count"] = len(rows)
    return record, rows, None


def _retained_host_record(output, source_ids):
    """Resume only the newest record for this exact request, including old failed attempts.

    An unreadable or malformed record cannot be resumed and is skipped (left
    in place); a symlinked record is refused.
    """
    for path in sorted(output.glob("host_tables_*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if path.is_symlink():
            raise ValueError("retained host table records may not be symlinks")
        try:
            record = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(record, dict) and record.get("source_ids") == source_ids:
            record["record_path"] = path.name
            return record
    return None


def _retained_host_rows(output, query, table, columns, source_ids):
    """Reparse the original responses; never trust cached normalized stellar values."""
    if query["table"] != table or query["columns"] != list(columns):
        raise ValueError("retained host table release or columns differ from the request")
    if set(query["source_queries"]) != set(source_ids):
        raise ValueError("retained host table does not cover the requested source identities")
    rows, seen = [], set()
    for response in query["responses"]:
        name = response["path"]
        path = output / name
        if Path(name).name != name or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("retained host responses must be regular files inside the output directory")
        if name in seen:
            raise ValueError("retained host response is duplicated")
        seen.add(name)
        chunk = [sid for sid in source_ids if query["source_queries"][sid]["response"] == name]
        if not chunk:
            raise ValueError("retained host response has no associated source query")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != response["sha256"]:
            raise ValueError(f"retained host response changed: {name}")
        rows.extend(_neighbor_rows(body, columns, chunk))
    if seen != {q["response"] for q in query["source_queries"].values()}:
        raise ValueError("retained host table is missing a response")
    return rows


def acquire_host_tables(output_dir, service, source_ids):
    """Fetch complete source-associated Gaia, StarHorse and SHBoost host records.

    Resumes the newest ``host_tables_*.json`` for the same source list: complete
    tables are reparsed from their retained responses, and an incomplete one
    reuses its completed chunks (``retained_query``). Each invocation writes a
    new record linked by ``resumed_from``, so retries never erase an earlier
    failure. The new record is updated after each table, so an interrupted run
    resumes from the last completed table without refetching retained responses.
    """
    output = Path(output_dir)
    ids = [gaia_identifier(value) for value in source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("host source identities must be unique")
    retained = _retained_host_record(output, ids)
    retrieval_id = uuid4().hex
    record_name = f"host_tables_{retrieval_id}.json"
    result = {"status": "in-progress", "source_ids": ids, "queries": {},
              "sources": {sid: {} for sid in ids}, "record_path": record_name}
    if retained is not None:
        result["resumed_from"] = retained["record_path"]
    tables = (
        ("gaia", GAIA_DR3_AP_TABLE, GAIA_AP_COLUMNS, f"gaia_ap_{retrieval_id}.xml",
         (("esa-tap", service.ESA_GAIA_TAP_URL), ("gaia-aip", service.AIP_GAIA_TAP_URL)), gaia_ap_entry),
        ("starhorse", STARHORSE_EDR3_TABLE, STARHORSE_COLUMNS, f"starhorse_{retrieval_id}.xml",
         (("gaia-aip", service.AIP_GAIA_TAP_URL),), starhorse_entry),
        ("shboost", SHBOOST_TABLE, SHBOOST_COLUMNS, f"host_shboost_{retrieval_id}.xml",
         (("gaia-aip", service.AIP_GAIA_TAP_URL),), shboost_entry),
    )
    for label, table, columns, name, backends, normalize in tables:
        query = (retained.get("queries") or {}).get(label) if retained is not None else None
        if isinstance(query, dict) and query.get("status") in {"available", "not-needed"}:
            rows = _retained_host_rows(output, query, table, columns, ids)
            failure = None
        else:
            query, rows, failure = fetch_neighbor_table(output, service, table=table, columns=columns,
                                                       source_ids=ids, response_name=name, backends=backends,
                                                       retained_query=query if isinstance(query, dict) else None)
        result["queries"][label] = query
        if rows is None:
            endpoints = ", ".join(url for _, url in backends)
            detail = query.get("reason") or (failure.reason if failure is not None else "no complete response")
            result.update(status="unavailable", reason=f"{label} host table unavailable at {endpoints}: {detail}")
            if failure is not None:
                result["network_failure"] = failure.to_manifest()
            write_json(output / record_name, result)
            error = AcquisitionError(result["reason"], retryable=bool(failure and failure.retryable))
            error.network_failure = failure
            raise error
        by_id = {gaia_identifier(row["source_id"]): row for row in rows}
        for sid in ids:
            row = by_id.get(sid)
            try:
                result["sources"][sid][label] = normalize(row)
            except (ValueError, TypeError, KeyError) as exc:
                result["sources"][sid][label] = None
                result["sources"][sid][label + "_refusal"] = f"{type(exc).__name__}: {exc}"
        write_json(output / record_name, result)
    result["status"] = "available"
    write_json(output / record_name, result)
    return result


def fetch_shboost_async(service, query):
    """Run a bounded read-only TAP UWS query and retain its result bytes.

    Used when the AIP sync endpoint refuses a SHBoost query with HTTP 400 or its
    read times out. The job runs at most 90 s; requests carry the service policy's User-Agent and
    timeout. ``_update`` is pyvo's polling call (its public ``wait`` differs
    across supported pyvo versions). The result is fetched
    through ``service`` so its download bounds apply.
    """
    import time
    from functools import partial
    import requests
    import pyvo

    policy = getattr(service, "policy", None)
    if policy is not None and not policy.allow_network:
        raise AcquisitionError("network access is disabled by policy")
    timeout = 30.0 if policy is None else min(30.0, float(policy.timeout_seconds))
    try:
        with requests.Session() as session:
            session.request = partial(session.request, timeout=timeout)
            if policy is not None:
                session.headers["User-Agent"] = request_user_agent(policy)
            if not service.AIP_GAIA_TAP_URL.endswith("/sync"):
                raise RuntimeError("AIP TAP URL is not a /sync endpoint; its UWS base is unknown")
            tap = pyvo.dal.TAPService(service.AIP_GAIA_TAP_URL.removesuffix("/sync"), session=session)
            with tap.submit_job(query) as job:
                job.run()
                deadline = time.monotonic() + 90
                while job.phase not in {"COMPLETED", "ERROR", "ABORTED"}:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("SHBoost asynchronous query exceeded 90 seconds")
                    job._update(wait_for_statechange=True, timeout=min(timeout, remaining))
                job.raise_if_error()
                url = job.result_uri
                body = service._http_get_bytes(url, "application/xml")
                if body is None:
                    raise RuntimeError("SHBoost asynchronous result download failed")
                return body, url
    except (requests.RequestException, pyvo.dal.DALAccessError) as exc:
        raise RuntimeError(f"SHBoost async archive query failed: {type(exc).__name__}: {exc}") from exc
