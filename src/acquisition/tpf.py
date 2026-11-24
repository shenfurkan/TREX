"""Retain MAST observation and product responses for TESS SPOC pixel files."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from uuid import uuid4

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.time import Time

from .identity import exact_identifier
from .storage import write_bytes, write_json
from trex.photometry.time_axis import rebase_bjd_tdb
from trex.runtime.logging import host_logger

from .transport import AcquisitionError, mast_error_reason, write_mast_attempts

MAST_INVOKE_URI = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD_URI = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
BTJD_REFERENCE_BJD = 2457000.0
MINIMUM_PRODUCT_CADENCES = 50


def _hdu_checksum_valid(path, hdu):
    """Check raw HDU records, including padding, without requiring DATASUM.

    Seaman, Pence & Rots, arXiv:1201.1345, CHECKSUM convention: the big-endian
    32-bit one's-complement sum of the entire HDU must be negative zero.
    Astropy 6.0.1 verify_checksum substitutes zero data sum when DATASUM is absent,
    although the convention explicitly permits CHECKSUM without DATASUM.
    """
    info = hdu.fileinfo()
    remaining = info["datLoc"] + info["datSpan"] - info["hdrLoc"]
    total = 0
    with Path(path).open("rb") as stream:
        stream.seek(info["hdrLoc"])
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block or len(block) % 4:
                return False
            total += int(np.frombuffer(block, dtype=">u4").sum(dtype=np.uint64))
            while total >> 32:
                total = (total & 0xFFFFFFFF) + (total >> 32)
            remaining -= len(block)
    return total == 0xFFFFFFFF


def load_tpf(path, tic_id, sector):
    """Read a SPOC TPF with declared TDB time and quality-zero pixel cadences.

    Output time is BTJD_TDB days; flux stays in the FITS-declared electron/s.
    This narrowly accepts BJDREF-relative TESS FITS rather than guessing an epoch:
    BJDREF, or BJDREFI with optional BJDREFF (never BJDREFF alone). Split
    references stay split through rebasing; invalid raw time rows are retained
    for the caller's quality/mask accounting, not silently removed. Every
    refusal, including an empty or mis-shaped SPOC aperture mask, raises
    ``AcquisitionError``.
    """
    identifier = exact_identifier(tic_id, "TIC ID")
    try:
        with fits.open(path, memmap=False) as hdul:
            primary, pixels, aperture = hdul[0], hdul[1], hdul[2]
            for hdu in hdul:
                if "CHECKSUM" in hdu.header and not _hdu_checksum_valid(path, hdu):
                    raise ValueError("TPF FITS CHECKSUM failed")
                if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
                    raise ValueError("TPF FITS DATASUM failed")
            if str(primary.header.get("TELESCOP", "")).strip().upper() != "TESS":
                raise ValueError("TPF is not a declared TESS product")
            if exact_identifier(primary.header.get("TICID"), "TPF TICID") != identifier:
                raise ValueError("TPF TIC identity does not match the requested target")
            if (isinstance(sector, bool) or not isinstance(sector, int) or sector <= 0
                    or isinstance(primary.header["SECTOR"], bool) or primary.header["SECTOR"] != sector):
                raise ValueError("TPF sector does not match the requested sector")
            for key in ("CAMERA", "CCD"):
                value = primary.header[key]
                if isinstance(value, bool) or value not in (1, 2, 3, 4):
                    raise ValueError(f"TPF {key} is invalid")
            header = dict(primary.header)
            for key in ("TIMESYS", "TIMEUNIT", "BJDREF", "BJDREFI", "BJDREFF", "MJDREF", "MJDREFI", "MJDREFF"):
                if key in primary.header and key in pixels.header and primary.header[key] != pixels.header[key]:
                    raise ValueError(f"TPF primary and pixel headers disagree on {key}")
                if key in pixels.header:
                    header[key] = pixels.header[key]
            if str(header.get("TIMESYS", "")).strip().upper() not in ("TDB", "BJD_TDB"):
                raise ValueError("TPF TIMESYS must declare TDB")
            if any(key in header for key in ("MJDREF", "MJDREFI", "MJDREFF")):
                raise ValueError("TPF MJDREF is not supported by the BJD-relative loader")
            unit = u.Unit(str(header["TIMEUNIT"]).strip())
            if not unit.is_equivalent(u.day):
                raise ValueError("TPF TIMEUNIT is not a time unit")
            direct = header.get("BJDREF")
            integer, fraction = header.get("BJDREFI"), header.get("BJDREFF")
            if any(isinstance(value, bool) for value in (direct, integer, fraction)):
                raise ValueError("TPF BJD reference keywords must be numbers")
            if integer is not None and (not math.isfinite(float(integer))
                                        or float(integer) != int(float(integer))):
                raise ValueError("TPF BJDREFI must be a finite integer")
            if fraction is not None and integer is None:
                raise ValueError("TPF BJDREFF without BJDREFI does not define a reference epoch")
            split = None if integer is None else Time(float(integer), float(fraction or 0), format="jd", scale="tdb")
            if direct is not None and split is not None and float(direct) != split.jd:
                raise ValueError("TPF BJDREF conflicts with split reference")
            if split is None and direct is None:
                raise ValueError("TPF BJD reference is missing")
            reference = split if split is not None else Time(float(direct), format="jd", scale="tdb")
            if not np.isfinite([reference.jd1, reference.jd2]).all():
                raise ValueError("TPF BJD reference is not finite")
            if str(pixels.columns["FLUX"].unit).strip().lower() not in ("e-/s", "electron/s", "electrons/s"):
                raise ValueError("TPF FLUX must declare electrons per second")
            time = np.asarray(pixels.data["TIME"], dtype=float) * unit.to(u.day)
            if np.any(np.isfinite(time)) and np.median(time[np.isfinite(time)]) > 2400000:
                raise ValueError("TPF TIME contains absolute dates with a BJD reference")
            finite_time = np.isfinite(time)
            if np.any(finite_time):
                time[finite_time] = rebase_bjd_tdb(
                    time[finite_time], reference, Time(BTJD_REFERENCE_BJD, format="jd", scale="tdb"))
            raw_quality = np.asarray(pixels.data["QUALITY"])
            if raw_quality.ndim != 1 or not np.issubdtype(raw_quality.dtype, np.integer) or np.any(raw_quality < 0):
                raise ValueError("TPF QUALITY must contain nonnegative integer flags")
            if np.issubdtype(raw_quality.dtype, np.unsignedinteger) and np.any(raw_quality > np.iinfo(np.int64).max):
                raise ValueError("TPF QUALITY exceeds supported integer flags")
            flux = np.asarray(pixels.data["FLUX"], dtype=float)
            if (time.ndim != 1 or flux.ndim != 3 or flux.shape[0] != time.size
                    or raw_quality.size != time.size or time.size < MINIMUM_PRODUCT_CADENCES):
                raise ValueError("TPF cadence and pixel arrays have incompatible shapes")
            if not np.any(np.isfinite(time)):
                raise ValueError("TPF has no finite observation times")
            mask = np.asarray(aperture.data)
            # Imported here: aperture -> gaia_sources -> tpf would be circular at module level.
            from .aperture import ApertureSceneError, pipeline_aperture_mask
            try:
                pipeline_aperture_mask(mask, flux.shape[1:])
            except ApertureSceneError as exc:
                raise ValueError(f"TPF SPOC aperture mask is unusable: {exc}") from exc
            return {"path": Path(path), "sector": sector, "time": time, "quality": raw_quality.astype(np.int64),
                    "quality_retained": raw_quality == 0, "flux": flux, "aperture": mask,
                    "header": header, "aperture_header": aperture.header.copy(), "flux_unit": "e-/s",
                    "time_system": "BTJD_TDB"}
    except (OSError, KeyError, IndexError, TypeError, ValueError, VerifyError, u.UnitsError) as exc:
        raise AcquisitionError(f"invalid TESS TPF: {exc}") from exc


def _response_rows(body):
    try:
        data = json.loads(body.decode("utf-8"))
        reason = mast_error_reason(data)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"response is not valid UTF-8 JSON: {exc}") from exc
    if reason is not None:
        raise AcquisitionError(f"MAST query failed at the provider: {reason}", retryable=True)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("response has no data table")
    if data.get("status") not in (None, "COMPLETE"):
        raise ValueError("response did not complete")
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
    return data["data"]


def _query(service, params, output, transport, retained=None):
    payload = {"service": service, "params": params, "format": "json", "pagesize": 50000}
    if (isinstance(retained, dict) and retained.get("status") == "available"
            and retained.get("uri") == MAST_INVOKE_URI and retained.get("request") == payload
            and retained.get("service") == service):
        name = retained.get("response")
        if isinstance(name, str) and name and Path(name).name == name:
            path = output / name
            if not path.is_symlink() and path.is_file():
                try:
                    rows = _response_rows(path.read_bytes())
                    if rows:
                        return rows, retained
                except (AcquisitionError, ValueError, FileNotFoundError):
                    pass
    identifier = uuid4().hex
    record = {"service": service, "uri": MAST_INVOKE_URI, "request": payload, "status": "in-progress"}
    manifest = output / f"mast_query_{identifier}.json"
    write_json(manifest, record)
    before = len(transport.transport_failures)
    request_error = None
    try:
        body = transport.post_mast(MAST_INVOKE_URI, urlencode({"request": json.dumps(payload, sort_keys=True)}).encode())
    except AcquisitionError as exc:
        body, request_error = None, exc

    history = write_mast_attempts(output, f"mast_response_{identifier}", getattr(transport, "mast_attempts", []),
                                  None if body is None else f"mast_response_{identifier}.json")
    if history:
        record["attempt_history"] = history
    if body is None:
        failure = transport.transport_failures[-1] if len(transport.transport_failures) > before else None
        reason = str(request_error) if request_error is not None else failure.reason if failure else "no response"
        record.update(status="unavailable", category="transport_failure", reason=f"MAST product query failed: {reason}")
        if failure is not None:
            record["network_failure"] = failure.to_manifest()
        write_json(manifest, record)
        error = request_error or AcquisitionError(record["reason"], retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        error.query_record = record
        raise error
    response = output / f"mast_response_{identifier}.json"
    if not history or history[-1].get("response") != response.name:
        write_bytes(response, body)
    record.update(response=response.name, retrieved_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    try:
        rows = _response_rows(body)
    except AcquisitionError as exc:
        record.update(status="unavailable", category="provider_error", reason=str(exc), retryable=True)
        write_json(manifest, record)
        exc.query_record = record
        raise
    except ValueError as exc:
        record.update(status="unavailable", category="malformed_response", reason=str(exc))
        write_json(manifest, record)
        error = AcquisitionError(f"MAST product response is incomplete or invalid: {exc}")
        error.query_record = record
        raise error from exc
    record["status"] = "available"
    write_json(manifest, record)
    return rows, record


def acquire_tpf_products(tic_id, sectors, output_dir, transport):
    """Retrieve requested SPOC TPFs, retaining complete MAST responses and FITS bytes.

    Select the shortest declared cadence per sector, refusing ambiguous product
    ties. Retained queries and files pass the same selection and FITS checks
    before reuse; a replacement is written separately from the retained bytes.
    """
    tic_id = exact_identifier(tic_id, "TIC ID")
    if not isinstance(sectors, (list, tuple)) or not sectors or any(
        isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in sectors
    ) or len(set(sectors)) != len(sectors):
        raise ValueError("sectors must be unique positive integers")
    output = Path(output_dir)
    manifest = output / "tpf_products.json"
    previous_bytes = None
    previous = {}
    recovery = []
    if manifest.is_symlink():
        raise AcquisitionError("retained TPF record may not be a symlink")
    if manifest.exists():
        previous_bytes = manifest.read_bytes()
        try:
            previous = json.loads(previous_bytes.decode("utf-8"))
            if not isinstance(previous, dict):
                raise ValueError("record is not an object")
            retained_id = exact_identifier(previous.get("tic_id"), "retained TIC ID")
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            recovery.append({"event": "rejected_record", "reason": str(exc)})
            previous = {}
        else:
            if retained_id != tic_id:
                raise AcquisitionError("retained TPF record belongs to a different target")
    previous_products = previous.get("products", [])
    if not isinstance(previous_products, list):
        recovery.append({"event": "rejected_products", "reason": "retained products are not a list"})
        previous_products = []
    records = []
    obs_query = product_query = failed_product = None
    previous_record = None

    def publish(status, *, error=None):
        nonlocal previous_record
        if previous_bytes is not None and previous_record is None:
            previous_record = f"tpf_products_history_{uuid4().hex}.json"
            write_bytes(output / previous_record, previous_bytes)
        result = {"status": status, "tic_id": tic_id, "products": records}
        if obs_query is not None:
            result["observations"] = obs_query
        if product_query is not None:
            result["product_query"] = product_query
        if previous_record is not None:
            result["previous_record"] = previous_record
        if recovery:
            result["recovery_history"] = recovery
        if failed_product is not None:
            result["failed_product"] = failed_product
        if error is not None:
            result["reason"] = str(error)
            query_record = getattr(error, "query_record", None)
            if query_record is not None:
                result["failed_query"] = query_record
                result["category"] = query_record.get("category")
            elif getattr(error, "category", None) is not None:
                result["category"] = error.category
            failure = getattr(error, "network_failure", None)
            if failure is not None:
                result["network_failure"] = failure.to_manifest()
        write_json(manifest, result)

    filters = [{"paramName": "target_name", "values": [tic_id]},
               {"paramName": "project", "values": ["TESS"]},
               {"paramName": "provenance_name", "values": ["SPOC"]},
               {"paramName": "dataproduct_type", "values": ["timeseries"]},
               {"paramName": "sequence_number", "values": list(sectors)}]
    try:
        observations, obs_query = _query("Mast.Caom.Filtered", {"columns": "*", "filters": filters}, output,
                                         transport, previous.get("observations"))
        if previous.get("observations") and obs_query != previous["observations"]:
            recovery.append({"event": "reacquired_observations"})
        if not observations:
            error = AcquisitionError("MAST returned no SPOC observations for the requested target and sectors")
            error.category = "empty_result"
            raise error
        ids = []
        for row in observations:
            if str(row.get("target_name")) != tic_id or row.get("project") != "TESS" or row.get("provenance_name") != "SPOC":
                raise AcquisitionError("MAST returned an observation outside the requested target or provenance")
            try:
                raw_sector = row["sequence_number"]
                if (isinstance(raw_sector, bool) or not isinstance(raw_sector, (int, str))
                        or (isinstance(raw_sector, str) and not raw_sector.isdecimal())):
                    raise ValueError("sector is not an integer")
                sector = int(raw_sector)
                obsid = exact_identifier(row["obsid"], "MAST obsid")
            except (KeyError, TypeError, ValueError) as exc:
                raise AcquisitionError("MAST observation lacks sector or product-group identity") from exc
            if sector not in sectors or obsid in ids:
                raise AcquisitionError("MAST returned an unexpected or duplicated observation")
            ids.append(obsid)
        products, product_query = _query("Mast.Caom.Products", {"obsid": ",".join(ids)}, output, transport,
                                         previous.get("product_query"))
        if previous.get("product_query") and product_query != previous["product_query"]:
            recovery.append({"event": "reacquired_product_query"})
        by_id = {str(row["obsid"]): row for row in observations}
        choices = {sector: [] for sector in sectors}
        for product in products:
            name, obsid = product.get("productFilename"), str(product.get("parent_obsid"))
            if not isinstance(name, str) or not name.lower().endswith(("_tp.fits", "_fast-tp.fits")):
                continue
            if obsid not in by_id:
                raise AcquisitionError("MAST product has an unrequested parent observation")
            row = by_id[obsid]
            try:
                if isinstance(row["t_exptime"], bool):
                    raise ValueError("cadence is not numeric")
                exposure = float(row["t_exptime"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AcquisitionError("SPOC TPF observation lacks a declared cadence") from exc
            if not math.isfinite(exposure) or exposure <= 0:
                raise AcquisitionError("SPOC TPF observation has invalid cadence")
            uri = product.get("dataURI")
            if (not isinstance(uri, str) or uri != "mast:TESS/product/" + name
                    or Path(name).name != name):
                raise AcquisitionError("SPOC TPF product has an invalid archive identity")
            choices[int(row["sequence_number"])].append((exposure, uri, name))

        selected = {}
        for sector in sectors:
            candidates = sorted(set(choices[sector]))
            if not candidates:
                error = AcquisitionError(f"MAST returned no SPOC TPF for sector {sector}")
                error.category = "empty_result"
                raise error
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                raise AcquisitionError(f"MAST returned ambiguous shortest-cadence TPFs for sector {sector}")
            selected[sector] = candidates[0]

        resolved = {}
        for sector, (exposure, uri, name) in selected.items():
            matches = [item for item in previous_products if isinstance(item, dict) and item.get("sector") == sector]
            if not matches:
                continue
            item = matches[0]
            path_name = item.get("path")
            url = MAST_DOWNLOAD_URI + quote(uri, safe=":/")
            try:
                if (len(matches) != 1 or isinstance(item.get("sector"), bool)
                        or not isinstance(item.get("sector"), int) or isinstance(item.get("cadence_seconds"), bool)
                        or item.get("cadence_seconds") != exposure or item.get("uri") != url
                        or not isinstance(path_name, str) or Path(path_name).name != path_name
                        or path_name in (".", "..") or not path_name.endswith("_" + name)
                        or not isinstance(item.get("retrieved_utc"), str)
                        or not item["retrieved_utc"]):
                    raise AcquisitionError("retained TPF record does not match the selected archive product")
                path = output / path_name
                if path.is_symlink() or not path.is_file():
                    raise AcquisitionError("retained TPF file is missing or not a regular file")
                load_tpf(path, tic_id, sector)
            except AcquisitionError as exc:
                recovery.append({"event": "rejected_product", "sector": sector, "path": path_name, "reason": str(exc)})
            else:
                resolved[sector] = item

        records = [resolved[sector] for sector in sectors if sector in resolved]
        if (previous.get("status") == "available" and records == previous_products
                and len(records) == len(sectors) and obs_query == previous.get("observations")
                and product_query == previous.get("product_query") and not recovery):
            return records
        for sector, (exposure, uri, name) in selected.items():
            if sector in resolved:
                recovery.append({"event": "reused_product", "sector": sector, "path": resolved[sector]["path"]})
                continue
            url = MAST_DOWNLOAD_URI + quote(uri, safe=":/")
            path = output / f"tpf_{uuid4().hex}_{name}"
            failed_product = {"sector": sector, "uri": url}
            publish("in-progress")
            host_logger(__name__).info("downloading SPOC TPF for TIC %s sector %s", tic_id, sector)
            before = len(transport.transport_failures)
            result = transport.download(url, path)
            if result is None:
                failure = transport.transport_failures[-1] if len(transport.transport_failures) > before else None
                error = AcquisitionError("MAST TPF download failed" + (f": {failure.reason}" if failure else ""),
                                         retryable=bool(failure and failure.retryable))
                error.network_failure = failure
                host_logger(__name__).warning("TPF download failed for TIC %s sector %s (retryable=%s)",
                                              tic_id, sector, error.retryable)
                raise error
            failed_product.update(path=path.name, retrieved_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            load_tpf(path, tic_id, sector)
            resolved[sector] = {"sector": sector, "cadence_seconds": exposure, "path": path.name, "uri": url,
                                "retrieved_utc": failed_product["retrieved_utc"]}
            records = [resolved[item] for item in sectors if item in resolved]
            if previous_bytes is not None:
                recovery.append({"event": "retrieved_product", "sector": sector, "path": path.name})
            failed_product = None
            publish("in-progress")
        publish("available")
        return records
    except AcquisitionError as exc:
        publish("unavailable", error=exc)
        raise
