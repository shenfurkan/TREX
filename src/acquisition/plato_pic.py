"""ESA PLATO Archive (PAX) PIC targets and contaminants for a declared cone.

The target table, cone predicate and keyset columns follow the astroquery PLATO
module and the ESA PAX TAP documentation read 2026-11-16:
https://docs.astropy.org/projects/astroquery/en/latest/esa/plato/plato.html
The contaminant table's columns are not documented there, so that query selects
``*`` and this module validates only the required position columns and the
presence of a finite native magnitude; the complete raw header is retained.

A cone is the requested aperture footprint, never a claim that the catalogue is
complete outside it.  Target pages use ``TOP`` plus a ``PICname`` keyset so a
large result is retrieved page by page instead of as one bulk download.
"""

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from xml.etree import ElementTree

from .dust_map import validate_icrs_field
from .identity import gaia_identifier
from .storage import write_bytes, write_json
from .transport import AcquisitionError, AcquisitionTransport
from .votable import parse_votable_bytes

PLATO_TAP_URL = "https://pax.esac.esa.int/tap/tap"
PIC_TARGET_TABLE = "pic_go.pic_target_go"
PIC_CONTAMINANT_TABLE = "pic_go.pic_contaminant_go"
PIC_TARGETS_RECORD_FILE = "pic_targets_query.json"
PIC_CONTAMINANTS_RECORD_FILE = "pic_contaminants_query.json"
# A 2000-row page stays well inside the transport's response-byte bound while a
# dense PIC cone still needs only a few requests.
PAGE_SIZE = 2000
# 50 x 2000 target rows is far above any PIC aperture footprint; reaching the
# bound means the provider did not honour the keyset, so the result is refused.
MAX_PAGES = 50
# The contaminant table is queried inside the same 0.1 deg^2 footprint used by
# the population and extinction stages.  It is the requested aperture, not a
# completeness radius for the catalogue.
CONE_AREA_SQ_DEG = 0.1
CONTAMINANT_RADIUS_DEG = math.sqrt(CONE_AREA_SQ_DEG / math.pi)
# Far above the few dozen contaminants a PIC target has; TAP marks a reply that
# exceeds it OVERFLOW, which this module raises instead of accepting truncated.
CONTAMINANT_MAXREC = 20000
PIC_TARGET_COLUMNS = (
    "PICname", "RAdeg", "DEdeg", "PlatoMagNCAM", "AG", "AKs", "Teff",
    "targetStatusFlag", "tPICplanetFlag", "tPICscientificRanking",
    "active_start_date", "active_end_date", "VmagCalculated",
)
PIC_TARGET_REQUIRED_COLUMNS = ("PICname", "RAdeg", "DEdeg")
CONTAMINANT_REQUIRED_COLUMNS = ("RAdeg", "DEdeg")
CONTAMINANT_MAGNITUDE_COLUMNS = ("PlatoMagNCAM", "Gmag", "VmagCalculated", "Jmag", "Hmag", "Ksmag")
CONTAMINANT_IDENTITY_COLUMNS = ("PICname", "source_id", "GaiaDR3", "GaiaEDR3", "contaminantID")
GAIA_DR3_COLUMNS = ("GaiaDR3", "gaia_dr3", "gaia_dr3_id")
SEARCH_LIMITATION = (
    "The cone covers only the requested aperture footprint; it is not a claim that the "
    "catalogue is complete outside that footprint."
)
CONTAMINANT_ASSOCIATION = (
    "Rows are returned by a cone query centred on the resolved host position; no "
    "nearest-match association to another catalogue is made."
)


class PicPaginationError(AcquisitionError):
    """A TAP result page overflowed, or keyset pagination did not complete."""


class PicResponseError(ValueError):
    """The provider's response is not one usable VOTable result table."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _validate_radius(radius_deg):
    if (isinstance(radius_deg, bool) or not isinstance(radius_deg, (int, float))
            or not math.isfinite(radius_deg) or not 0 < radius_deg <= 180):
        raise ValueError("cone radius must be a positive finite number of degrees no greater than 180")
    return float(radius_deg)


def _validate_page_size(page_size):
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 0 < page_size <= PAGE_SIZE:
        raise ValueError(f"page_size must be an integer in 1..{PAGE_SIZE}")
    return page_size


def _adql_literal(value):
    return "'" + value.replace("'", "''") + "'"


def _tap_overflow(body):
    """Whether any INFO element declares a TAP overflow, at document or RESOURCE level."""
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return False
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].upper() != "INFO":
            continue
        name = element.get("name") or ""
        value = element.get("value") or ""
        if (name.upper() == "QUERY_STATUS" and value.upper() == "OVERFLOW") or name.lower() == "overflow":
            return True
    return False


def _post_tap(transport, query, *, maxrec=None):
    fields = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query}
    if maxrec is not None:
        fields["MAXREC"] = str(maxrec)
    return transport.post_form(PLATO_TAP_URL, urlencode(fields).encode("ascii"),
                               accept="application/x-votable+xml")


def _network_transport(policy, transport):
    if not policy.allow_network:
        raise AcquisitionError("network access is disabled by policy")
    return AcquisitionTransport(policy) if transport is None else transport


def build_pic_target_query(ra_deg, dec_deg, radius_deg, *, after_pic_name=None, page_size=PAGE_SIZE):
    """ADQL text for one keyset page of PIC targets inside an ICRS cone.

    ``after_pic_name`` selects rows with ``PICname`` strictly greater than the
    previous page's last value; the single quote in the literal is doubled for
    ADQL.  The caller owns the page loop (``acquire_pic_targets``).
    """
    validate_icrs_field(ra_deg, dec_deg)
    radius_deg = _validate_radius(radius_deg)
    page_size = _validate_page_size(page_size)
    if after_pic_name is not None and (not isinstance(after_pic_name, str) or not after_pic_name.strip()):
        raise ValueError("after_pic_name must be a non-empty PICname string")
    ra_deg, dec_deg = float(ra_deg), float(dec_deg)
    where = ("1=CONTAINS(POINT('ICRS', RAdeg, DEdeg), "
             f"CIRCLE('ICRS', {ra_deg!r}, {dec_deg!r}, {radius_deg!r}))")
    if after_pic_name is not None:
        where += " AND PICname > " + _adql_literal(after_pic_name)
    return (f"SELECT TOP {page_size} {', '.join(PIC_TARGET_COLUMNS)} FROM {PIC_TARGET_TABLE} "
            f"WHERE {where} ORDER BY PICname")


def build_pic_contaminant_query(ra_deg, dec_deg, radius_deg):
    """ADQL text for the PIC contaminants inside an ICRS cone."""
    validate_icrs_field(ra_deg, dec_deg)
    radius_deg = _validate_radius(radius_deg)
    ra_deg, dec_deg = float(ra_deg), float(dec_deg)
    return (f"SELECT * FROM {PIC_CONTAMINANT_TABLE} WHERE 1=CONTAINS("
            f"POINT('ICRS', RAdeg, DEdeg), CIRCLE('ICRS', {ra_deg!r}, {dec_deg!r}, {radius_deg!r}))")


def parse_pic_votable(body):
    """Return ``(columns, rows)`` of one complete PLATO TAP result page.

    Values keep their VOTable scalar types and masked cells become ``None``, so
    64-bit identifiers stay exact.  An empty ``TABLEDATA`` is a valid result:
    its header is returned with no rows.  Raises ``PicPaginationError`` for a
    TAP ``QUERY_STATUS=OVERFLOW`` (INFO at document or RESOURCE level) and
    ``PicResponseError`` for a non-VOTable, error document or malformed table.
    """
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError("VOTable body must be bytes")
    body = bytes(body)
    if _tap_overflow(body):
        raise PicPaginationError("PLATO TAP response reports QUERY_STATUS=OVERFLOW (truncated result)")
    try:
        parsed = parse_votable_bytes(body)
    except Exception as exc:  # trust boundary: any parser failure is a malformed response
        raise PicResponseError(
            f"PLATO TAP response is not a complete VOTable: {type(exc).__name__}: {exc}") from exc
    columns = tuple(parsed["table"].colnames)
    rows = [dict(zip(columns, row)) for row in parsed["data"]]
    return columns, rows


def _first_present(row, columns):
    for column in columns:
        value = row.get(column)
        if value is not None:
            return value
    return None


def _first_nonempty(row, columns):
    for column in columns:
        value = row.get(column)
        if value is not None and str(value).strip():
            return column, value
    return None, None


def pic_target_identity(row):
    """Identity of one PIC target row, with no invented coordinate epoch.

    PAX target rows declare no coordinate epoch, so ``epoch`` is ``None`` unless
    the row itself carries an ``epoch``/``ref_epoch`` value; J2000 is never
    assumed.  A present ``GaiaDR3`` value that does not parse as an exact
    signed-64-bit identifier is retained under ``gaia_dr3_malformed`` instead of
    being silently dropped.
    """
    if not isinstance(row, dict):
        raise TypeError("PIC target row must be a mapping")
    pic_name = row.get("PICname")
    if not isinstance(pic_name, str) or not pic_name.strip():
        raise ValueError("PIC target row has no exact PICname string")
    ra_deg = _finite_float(row.get("RAdeg"))
    dec_deg = _finite_float(row.get("DEdeg"))
    if ra_deg is None or dec_deg is None or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90:
        raise ValueError("PIC target row has no finite ICRS RAdeg/DEdeg")
    epoch = _finite_float(row.get("epoch")) if row.get("epoch") is not None else None
    if epoch is None and row.get("ref_epoch") is not None:
        epoch = _finite_float(row.get("ref_epoch"))
    identity = {
        "namespace": "plato-pic",
        "native_identifier": pic_name,
        "frame": "ICRS",
        "epoch": epoch,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "gaia_dr3_id": None,
    }
    gaia_column, raw_gaia = _first_nonempty(row, GAIA_DR3_COLUMNS)
    if raw_gaia is not None:
        try:
            identity["gaia_dr3_id"] = gaia_identifier(raw_gaia)
        except ValueError as exc:
            identity["gaia_dr3_malformed"] = {"raw_value": str(raw_gaia), "reason": str(exc)}
        else:
            identity["gaia_dr3_column"] = gaia_column
    return identity


def pic_catalog_identity(row):
    """Canonical :class:`trex.acquisition.identity.CatalogIdentity` for a PIC row.

    The native ``PICname`` is the exact identity. A valid Gaia DR3 column is a
    cross-identifier, not a replacement; a malformed one remains recorded in
    :func:`pic_target_identity` and is never used to resolve the target.
    """
    from .identity import catalog_identity

    identity = pic_target_identity(row)
    cross = []
    if identity.get("gaia_dr3_id") is not None:
        cross.append(("gaia-dr3", identity["gaia_dr3_id"]))
    return catalog_identity(
        "plato-pic",
        identity["native_identifier"],
        frame=identity["frame"],
        epoch_jyear=identity["epoch"],
        cross_identifiers=cross,
    )


def _magnitude_columns(row):
    return [name for name in row
            if name in CONTAMINANT_MAGNITUDE_COLUMNS
            or (name.lower().endswith("mag") and not name.lower().startswith("e_"))]


def pic_contaminant_rows(rows, *, host_pic_name):
    """Normalize contaminant rows without a magnitude cut or duplicate removal.

    Each row keeps its native ICRS position in degrees, every native magnitude
    column under its own name with the ``mag`` unit (``None`` when missing),
    the host ``PICname`` that the cone was centred on, and the verbatim raw row.
    A malformed present ``GaiaDR3`` value is recorded, not replaced by ``None``.
    """
    if not isinstance(host_pic_name, str) or not host_pic_name.strip():
        raise ValueError("host PICname must be a non-empty string")
    normalized = []
    for row in rows:
        ra_deg = _finite_float(row.get("RAdeg"))
        dec_deg = _finite_float(row.get("DEdeg"))
        if ra_deg is None or dec_deg is None or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90:
            raise ValueError("PIC contaminant row has no finite ICRS RAdeg/DEdeg")
        magnitudes = {name: {"value": _finite_float(row.get(name)), "unit": "mag"}
                      for name in _magnitude_columns(row)}
        _identity_column, native_identifier = _first_nonempty(row, CONTAMINANT_IDENTITY_COLUMNS)
        contaminant = {
            "host_pic_name": host_pic_name,
            "native_identifier": native_identifier,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "magnitudes": magnitudes,
            "gaia_dr3_id": None,
            "raw": dict(row),
        }
        _gaia_column, raw_gaia = _first_nonempty(row, GAIA_DR3_COLUMNS)
        if raw_gaia is not None:
            try:
                contaminant["gaia_dr3_id"] = gaia_identifier(raw_gaia)
            except ValueError as exc:
                contaminant["gaia_dr3_malformed"] = {"raw_value": str(raw_gaia), "reason": str(exc)}
        normalized.append(contaminant)
    return normalized


def _validate_target_table(columns, rows):
    missing = [name for name in PIC_TARGET_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"PLATO PIC target table is missing required columns {missing}; "
                         f"raw header: {list(columns)}")
    for row in rows:
        pic_name = row.get("PICname")
        if not isinstance(pic_name, str) or not pic_name.strip():
            raise ValueError("PLATO PIC target row has no exact PICname")
        ra_deg = _finite_float(row.get("RAdeg"))
        dec_deg = _finite_float(row.get("DEdeg"))
        if ra_deg is None or dec_deg is None or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90:
            raise ValueError("PLATO PIC target row has no finite ICRS RAdeg/DEdeg")


def _validate_contaminant_table(columns, rows):
    missing = [name for name in CONTAMINANT_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"PLATO PIC contaminant table is missing required columns {missing}; "
                         f"raw header: {list(columns)}")
    for row in rows:
        ra_deg = _finite_float(row.get("RAdeg"))
        dec_deg = _finite_float(row.get("DEdeg"))
        if ra_deg is None or dec_deg is None or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90:
            raise ValueError("PLATO PIC contaminant row has no finite ICRS RAdeg/DEdeg")
    if rows:
        magnitude_columns = [name for name in CONTAMINANT_MAGNITUDE_COLUMNS if name in columns]
        if not magnitude_columns:
            raise ValueError("PLATO PIC contaminant table has no native magnitude column; "
                             f"raw header: {list(columns)}")
        if not any(_finite_float(row.get(name)) is not None
                   for row in rows for name in magnitude_columns):
            raise ValueError("PLATO PIC contaminant table has no finite native magnitude; "
                             f"raw header: {list(columns)}")


def _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns):
    record = {
        "schema_version": 1,
        "catalog": "plato-pic",
        "provider": "ESA PLATO Archive (PAX)",
        "tap_url": PLATO_TAP_URL,
        "table": PIC_TARGET_TABLE,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "search_radius_deg": radius_deg,
        "page_size": page_size,
        "page_count": len(queries),
        "row_count": len(rows),
        "queries": list(queries),
        "raw_files": list(raw_files),
        "pagination": "keyset: PICname > last returned PICname, ORDER BY PICname",
        "retrieved_utc": _utc_now(),
        "limitation": SEARCH_LIMITATION,
    }
    if columns:
        record["raw_header"] = list(columns)
    return record


def _contaminant_record(host_pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns):
    record = {
        "schema_version": 1,
        "catalog": "plato-pic",
        "provider": "ESA PLATO Archive (PAX)",
        "tap_url": PLATO_TAP_URL,
        "table": PIC_CONTAMINANT_TABLE,
        "host_pic_name": host_pic_name,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "search_radius_deg": radius_deg,
        "maxrec": CONTAMINANT_MAXREC,
        "query_adql": query,
        "row_count": len(rows),
        "raw_files": list(raw_files),
        "association": CONTAMINANT_ASSOCIATION,
        "retrieved_utc": _utc_now(),
        "limitation": SEARCH_LIMITATION,
    }
    if columns:
        record["raw_header"] = list(columns)
    return record


def _write_failure(path, record, status, reason, *, kind, failure=None, exception=None):
    record.update(status=status, reason=reason)
    if kind is not None:
        record["error_kind"] = kind
    if failure is not None:
        record["failure"] = failure.to_manifest()
    if exception is not None:
        record["error"] = {
            "exception_class": type(exception).__name__,
            "message": str(exception),
            "retryable": bool(getattr(exception, "retryable", False)),
        }
    write_json(path, record)


def _transport_failure(transport):
    failures = getattr(transport, "transport_failures", [])
    return failures[-1] if failures else None


def _missing_response(what, transport):
    failure = _transport_failure(transport)
    reason = f"PLATO TAP {what} query failed: {failure.reason if failure is not None else 'no response'}"
    error = AcquisitionError(reason, retryable=bool(failure is not None and failure.retryable))
    error.network_failure = failure
    return error, failure


def acquire_pic_targets(ra_deg, dec_deg, radius_deg, output_dir, policy, *, page_size=PAGE_SIZE,
                        transport=None):
    """Retain every PIC target in the cone, or raise on incomplete evidence.

    Pages are requested with a ``PICname`` keyset until one returns fewer than
    ``page_size`` rows.  Each page's exact bytes are written under
    ``pic_targets_raw_NN.xml`` and their SHA-256 is recorded.  The returned
    ``record`` is one of ``available`` (rows found), ``empty`` (a complete table
    with no rows), ``unavailable`` (no complete response: transport failure,
    TAP overflow or a keyset that did not advance) or ``malformed`` (a response
    that is not the expected target table).  A disabled network policy raises
    before any request and writes no record.
    """
    validate_icrs_field(ra_deg, dec_deg)
    radius_deg = _validate_radius(radius_deg)
    page_size = _validate_page_size(page_size)
    transport = _network_transport(policy, transport)
    output = Path(output_dir)
    record_path = output / PIC_TARGETS_RECORD_FILE
    queries, raw_files, rows, columns = [], [], [], ()
    after_pic_name = None

    while len(queries) < MAX_PAGES:
        query = build_pic_target_query(ra_deg, dec_deg, radius_deg,
                                       after_pic_name=after_pic_name, page_size=page_size)
        queries.append(query)
        try:
            body = _post_tap(transport, query)
        except AcquisitionError as exc:
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns)
            _write_failure(record_path, record, "unavailable", str(exc), kind="transport", exception=exc)
            raise
        if body is None:
            error, failure = _missing_response("target", transport)
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns)
            _write_failure(record_path, record, "unavailable", str(error), kind="transport", failure=failure)
            raise error
        name = f"pic_targets_raw_{len(raw_files):02d}.xml"
        write_bytes(output / name, body)
        raw_files.append({"path": name, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
        page_columns = ()
        try:
            page_columns, page_rows = parse_pic_votable(body)
            _validate_target_table(page_columns, page_rows)
        except PicPaginationError as exc:
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows,
                                    page_columns or columns)
            _write_failure(record_path, record, "unavailable", str(exc), kind="overflow", exception=exc)
            raise
        except (PicResponseError, ValueError, TypeError) as exc:
            message = f"PLATO PIC target response is malformed: {exc}"
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows,
                                    page_columns or columns)
            _write_failure(record_path, record, "malformed", message, kind="response", exception=exc)
            raise AcquisitionError(message) from exc
        columns = page_columns
        raw_files[-1]["rows"] = len(page_rows)
        rows.extend(page_rows)
        if len(page_rows) < page_size:
            status = "available" if rows else "empty"
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns)
            record["status"] = status
            write_json(record_path, record)
            return record, rows
        last_pic_name = page_rows[-1]["PICname"]
        if after_pic_name is not None and last_pic_name <= after_pic_name:
            reason = (f"PLATO PIC target keyset did not advance: provider repeated or regressed "
                      f"PICname {last_pic_name!r} after {after_pic_name!r}")
            record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns)
            _write_failure(record_path, record, "unavailable", reason, kind="repeated_page")
            raise PicPaginationError(reason)
        after_pic_name = last_pic_name

    reason = f"PLATO PIC target pagination reached the maximum page budget (MAX_PAGES={MAX_PAGES})"
    record = _target_record(ra_deg, dec_deg, radius_deg, page_size, queries, raw_files, rows, columns)
    _write_failure(record_path, record, "unavailable", reason, kind="page_limit")
    raise PicPaginationError(reason)


def _target_position(target):
    if not isinstance(target, dict):
        raise TypeError("target must be a PIC target row or identity mapping")
    pic_name = _first_present(target, ("PICname", "native_identifier"))
    ra_deg = _first_present(target, ("RAdeg", "ra_deg"))
    dec_deg = _first_present(target, ("DEdeg", "dec_deg"))
    if not isinstance(pic_name, str) or not pic_name.strip():
        raise ValueError("target has no exact PICname string")
    try:
        validate_icrs_field(ra_deg, dec_deg)
    except ValueError:
        raise ValueError("target has no finite ICRS RAdeg/DEdeg") from None
    return pic_name, float(ra_deg), float(dec_deg)


def acquire_pic_contaminants(target, output_dir, policy, *, radius_deg=CONTAMINANT_RADIUS_DEG,
                             transport=None):
    """Retain the PIC contaminants around a resolved target, tagged with its PICname.

    ``target`` is a target row or a ``pic_target_identity`` mapping.  Every row
    returned by the cone is retained (no 15-row cap, magnitude cut or duplicate
    removal) and tagged with the host ``PICname``.  An empty contaminant table is
    a successful ``empty`` result; a transport failure is ``unavailable`` and
    raises ``AcquisitionError`` after its record is written.
    """
    pic_name, ra_deg, dec_deg = _target_position(target)
    radius_deg = _validate_radius(radius_deg)
    transport = _network_transport(policy, transport)
    output = Path(output_dir)
    record_path = output / PIC_CONTAMINANTS_RECORD_FILE
    query = build_pic_contaminant_query(ra_deg, dec_deg, radius_deg)
    raw_files, rows, columns = [], [], ()

    try:
        body = _post_tap(transport, query, maxrec=CONTAMINANT_MAXREC)
    except AcquisitionError as exc:
        record = _contaminant_record(pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns)
        _write_failure(record_path, record, "unavailable", str(exc), kind="transport", exception=exc)
        raise
    if body is None:
        error, failure = _missing_response("contaminant", transport)
        record = _contaminant_record(pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns)
        _write_failure(record_path, record, "unavailable", str(error), kind="transport", failure=failure)
        raise error
    name = "pic_contaminants_raw.xml"
    write_bytes(output / name, body)
    raw_files.append({"path": name, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
    try:
        columns, rows = parse_pic_votable(body)
        _validate_contaminant_table(columns, rows)
        normalized = pic_contaminant_rows(rows, host_pic_name=pic_name)
    except PicPaginationError as exc:
        record = _contaminant_record(pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns)
        _write_failure(record_path, record, "unavailable", str(exc), kind="overflow", exception=exc)
        raise
    except (PicResponseError, ValueError, TypeError) as exc:
        message = f"PLATO PIC contaminant response is malformed: {exc}"
        record = _contaminant_record(pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns)
        _write_failure(record_path, record, "malformed", message, kind="response", exception=exc)
        raise AcquisitionError(message) from exc
    raw_files[-1]["rows"] = len(rows)
    record = _contaminant_record(pic_name, ra_deg, dec_deg, radius_deg, query, raw_files, rows, columns)
    record["status"] = "available" if rows else "empty"
    write_json(record_path, record)
    return record, normalized
