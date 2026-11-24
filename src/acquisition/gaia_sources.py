"""Retain a complete Gaia DR3 astrometric cone before scene construction."""

import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from .identity import gaia_identifier
from .storage import write_bytes, write_json
from .transport import AcquisitionError
from .votable import parse_votable_bytes
from .tpf import BTJD_REFERENCE_BJD

# Positions, proper motions and epoch for target matching; magnitudes and fluxes
# for flux ratios; the Lindegren et al. (2021) parallax zero-point inputs
# (nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved);
# RUWE and the non_single_star bitmask as measured multiplicity context.
# A missing value stays null rather than being given a numerical default.
GAIA_CONE_QUERY_FILE = "gaia_cone_query.json"
GAIA_SOURCE_COLUMNS = (
    "source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "ruwe", "non_single_star", "pmra", "pmdec", "ref_epoch", "sep_arcsec", "phot_g_mean_flux",
    "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_bp_mean_flux_error",
    "phot_rp_mean_flux", "phot_rp_mean_flux_error", "parallax", "parallax_error",
    "nu_eff_used_in_astrometry", "pseudocolour", "ecl_lat", "astrometric_params_solved",
    "ra_error", "dec_error", "pmra_error", "pmdec_error", "parallax_over_error",
    "phot_g_mean_flux_over_error", "phot_bp_mean_flux_over_error", "phot_rp_mean_flux_over_error",
)


def _finite_float(value):
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def observation_epoch_jyear(time_btjd):
    """Julian-year epoch of the median finite BTJD_TDB cadence, or None.

    The epoch ``gaia_position_at_epoch`` propagates a source to for one TPF sector.
    """
    values = np.asarray(time_btjd, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    observation_time = Time(BTJD_REFERENCE_BJD, format="jd", scale="tdb") + float(np.median(finite)) * u.day
    return float(observation_time.jyear)


def gaia_position_at_epoch(source, epoch_jyear):
    """Propagate an ICRS Gaia position using its own proper motion and epoch.

    Angles are degrees, proper motions mas/yr, and epochs Julian years. A
    missing proper-motion component or reference epoch cannot be guessed.
    """
    values = [_finite_float(source.get(key)) for key in (
        "ra_deg", "dec_deg", "pmra_mas_per_year", "pmdec_mas_per_year", "reference_epoch_jyear")]
    if any(value is None for value in values) or _finite_float(epoch_jyear) is None:
        return None
    ra_deg, dec_deg, pmra, pmdec, reference_epoch = values
    try:
        propagated = SkyCoord(
            ra=ra_deg * u.deg, dec=dec_deg * u.deg,
            pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
            obstime=Time(reference_epoch, format="jyear"), frame="icrs",
        ).apply_space_motion(new_obstime=Time(float(epoch_jyear), format="jyear"))
    except (TypeError, ValueError, u.UnitsError):
        return None
    return float(propagated.ra.to_value(u.deg)), float(propagated.dec.to_value(u.deg))


_MOTION_KEYS = ("ra_deg", "dec_deg", "pmra_mas_per_year", "pmdec_mas_per_year", "reference_epoch_jyear")


def _target_epoch_separations(sources, target, epoch_jyear):
    """Separation [arcsec] from ``target`` of each source moved to ``epoch_jyear``, or None.

    All sources with complete motion are propagated in one vectorized
    ``apply_space_motion``, the same calculation as ``gaia_position_at_epoch``;
    if that array call fails, each source is propagated on its own so one bad
    row cannot remove the others' positions.
    """
    separations = [None] * len(sources)
    moving = [index for index, source in enumerate(sources)
              if all(source.get(key) is not None for key in _MOTION_KEYS)]
    if not moving:
        return separations
    values = np.array([[sources[index][key] for key in _MOTION_KEYS] for index in moving], dtype=float)
    try:
        moved = SkyCoord(
            ra=values[:, 0] * u.deg, dec=values[:, 1] * u.deg,
            pm_ra_cosdec=values[:, 2] * u.mas / u.yr, pm_dec=values[:, 3] * u.mas / u.yr,
            obstime=Time(values[:, 4], format="jyear"), frame="icrs",
        ).apply_space_motion(new_obstime=Time(float(epoch_jyear), format="jyear"))
        arcsec = target.separation(SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")).arcsec
    except (TypeError, ValueError, u.UnitsError):
        arcsec = []
        for index in moving:
            position = gaia_position_at_epoch(sources[index], epoch_jyear)
            arcsec.append(math.nan if position is None else float(target.separation(
                SkyCoord(ra=position[0] * u.deg, dec=position[1] * u.deg, frame="icrs")).arcsec))
    for index, value in zip(moving, np.asarray(arcsec, dtype=float)):
        if math.isfinite(value):
            separations[index] = 0.0 if value <= 1e-9 else float(value)
    return separations


def _candidate_summary(sources):
    """Name each compatible candidate and its own epoch evidence, without ranking them."""
    return "; ".join(
        "{0} ({1}, epoch separation {2!r} arcsec)".format(
            source["source_id"], source["target_match_method"], source["target_match_separation_arcsec"])
        for source in sources)


def _ambiguous_target_error(reason, matches):
    """AcquisitionError whose ``candidates`` retain every compatible identity and its evidence."""
    error = AcquisitionError(reason)
    error.candidates = [
        {"source_id": source["source_id"], "target_match_method": source["target_match_method"],
         "target_match_separation_arcsec": source["target_match_separation_arcsec"]}
        for source in matches]
    return error


def normalize_and_match_gaia(rows, ra_deg, dec_deg, epoch_jyear, *, match_radius_arcsec, expected_source_id=None):
    """Normalize Gaia rows and require one source compatible at the target epoch.

    Positions are ICRS degrees, epochs Julian years, and the matching radius
    arcseconds. A row with both proper motions is propagated; one without them
    uses its catalogue separation, because its Gaia-epoch position is then the
    only position the catalogue gives. Rows are validated here as well as in
    ``fetch_gaia_sources``: this function also accepts rows from other callers.

    ``expected_source_id`` is an optional caller-known Gaia DR3 identifier
    (for example from an exact-identity cross-match; TIC v8 ``GAIA`` values are
    DR2 and must be resolved first). It resolves several epoch-compatible rows
    only when exactly one carries that identity. Otherwise the association
    fails closed and names every candidate; no nearest or brightest row is
    chosen. RUWE and ``non_single_star`` are retained as measured context, not
    thresholded into a multiplicity decision.
    """
    target_values = (ra_deg, dec_deg, epoch_jyear, match_radius_arcsec)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in target_values) or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90
            or match_radius_arcsec <= 0):
        raise ValueError("Gaia matching requires finite ICRS position, epoch and positive radius")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    sources = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Gaia row must be an object")
        source_id = gaia_identifier(row.get("source_id"))
        source = {
            "source_id": source_id, "ra_deg": _finite_float(row.get("ra")),
            "dec_deg": _finite_float(row.get("dec")), "separation_arcsec": _finite_float(row.get("sep_arcsec")),
            "pmra_mas_per_year": _finite_float(row.get("pmra")),
            "pmdec_mas_per_year": _finite_float(row.get("pmdec")),
            "reference_epoch_jyear": _finite_float(row.get("ref_epoch")),
            "parallax_mas": _finite_float(row.get("parallax")),
            "parallax_error_mas": _finite_float(row.get("parallax_error")),
            "nu_eff_used_in_astrometry_per_um": _finite_float(row.get("nu_eff_used_in_astrometry")),
            "pseudocolour_per_um": _finite_float(row.get("pseudocolour")),
            "ecl_lat_deg": _finite_float(row.get("ecl_lat")),
            "astrometric_params_solved": _finite_float(row.get("astrometric_params_solved")),
            "ra_error_mas": _finite_float(row.get("ra_error")), "dec_error_mas": _finite_float(row.get("dec_error")),
            "pmra_error_mas_per_year": _finite_float(row.get("pmra_error")),
            "pmdec_error_mas_per_year": _finite_float(row.get("pmdec_error")),
            "ruwe": _finite_float(row.get("ruwe")),
            "non_single_star": _finite_float(row.get("non_single_star")),
        }
        for band in ("g", "bp", "rp"):
            source[f"phot_{band}_mean_mag"] = _finite_float(row.get(f"phot_{band}_mean_mag"))
            flux = _finite_float(row.get(f"phot_{band}_mean_flux"))
            error = _finite_float(row.get(f"phot_{band}_mean_flux_error"))
            source[f"phot_{band}_mean_mag_error"] = (
                (2.5 / math.log(10.0)) * (error / flux) if flux is not None and flux > 0
                and error is not None and error > 0 else None)
            source[f"phot_{band}_mean_flux_over_error"] = _finite_float(row.get(f"phot_{band}_mean_flux_over_error"))
        source["parallax_over_error"] = _finite_float(row.get("parallax_over_error"))
        solution = source["astrometric_params_solved"]
        source["astrometric_params_solved"] = int(solution) if solution is not None and solution.is_integer() else None
        # Gaia DR3 documentation 1.3, Sec. 20.1.1: a short bitmask whose bits 1/2/3
        # mark astrometric/spectroscopic/eclipsing non-single-star processing.
        nss = source["non_single_star"]
        source["non_single_star"] = int(nss) if nss is not None and nss.is_integer() else None
        if (source["ra_deg"] is None or source["dec_deg"] is None or source["separation_arcsec"] is None
                or not 0 <= source["ra_deg"] < 360 or not -90 <= source["dec_deg"] <= 90
                or source["separation_arcsec"] < 0):
            raise ValueError("Gaia response contains an incomplete astrometric row")
        source["catalog_row_status"] = "complete"
        sources.append(source)
    if len({row["source_id"] for row in sources}) != len(sources):
        raise ValueError("Gaia response contains duplicate source identities")
    for source, separation in zip(sources, _target_epoch_separations(sources, target, epoch_jyear)):
        source["target_epoch_separation_arcsec"] = separation
        source["target_match_method"] = None
        source["target_match_separation_arcsec"] = None
        source["target_match_evidence"] = None
        has_motion = source["pmra_mas_per_year"] is not None and source["pmdec_mas_per_year"] is not None
        if has_motion and separation is not None and separation <= match_radius_arcsec:
            source["target_match_method"] = "proper_motion_to_target_epoch"
            source["target_match_separation_arcsec"] = separation
        elif not has_motion and source["separation_arcsec"] <= match_radius_arcsec:
            source["target_match_method"] = "catalog_position_without_proper_motion"
            source["target_match_separation_arcsec"] = source["separation_arcsec"]
    matches = [source for source in sources if source["target_match_method"] is not None]
    if expected_source_id is not None:
        expected = gaia_identifier(expected_source_id)
        selected = [source for source in matches if source["source_id"] == expected]
        if len(selected) == 1:
            selected[0]["target_match_evidence"] = "caller_supplied_gaia_dr3_source_id"
            return selected[0], [source for source in sources if source is not selected[0]]
        if not matches:
            raise AcquisitionError(
                "Gaia target association expected source {0} but no source is compatible at the target epoch".format(
                    expected))
        raise _ambiguous_target_error(
            "Gaia target association expected source {0} but the compatible candidates are: {1}".format(
                expected, _candidate_summary(matches)), matches)
    if len(matches) != 1:
        raise _ambiguous_target_error(
            "Gaia target association requires exactly one compatible source; found {0}{1}".format(
                len(matches), ": " + _candidate_summary(matches) if matches else ""), matches)
    matches[0]["target_match_evidence"] = "unique_epoch_compatible_match"
    return matches[0], [source for source in sources if source is not matches[0]]


def fetch_gaia_sources(ra_deg, dec_deg, radius_arcsec, output_dir, transport):
    """Fetch all Gaia DR3 rows in a declared ICRS cone, retaining response and query.

    A complete TAP response is not yet a target association: matching requires
    a declared coordinate epoch and proper-motion propagation by the caller.
    """
    values = (ra_deg, dec_deg, radius_arcsec)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values) or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90 or radius_arcsec <= 0):
        raise ValueError("Gaia cone requires finite ICRS coordinates and a positive radius in arcsec")
    columns = ["gs." + name for name in GAIA_SOURCE_COLUMNS if name != "sep_arcsec"]
    columns.insert(GAIA_SOURCE_COLUMNS.index("sep_arcsec"),
                   "DISTANCE(POINT('ICRS', gs.ra, gs.dec), "
                   f"POINT('ICRS', {ra_deg}, {dec_deg}))*3600 AS sep_arcsec")
    query = (f"SELECT {', '.join(columns)} FROM gaiadr3.gaia_source AS gs "
             f"WHERE 1=CONTAINS(POINT('ICRS', gs.ra, gs.dec), "
             f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec / 3600.0})) ORDER BY sep_arcsec ASC")
    arguments = urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": query})
    output = Path(output_dir)
    retrieval_id = uuid4().hex
    record = {"table": "gaiadr3.gaia_source", "query": query, "radius_arcsec": radius_arcsec,
              "responses": [], "failures": []}
    parse_error = None
    for backend in (transport.ESA_GAIA_TAP_URL, transport.AIP_GAIA_TAP_URL):
        url = backend + "?" + arguments
        before = len(transport.transport_failures)
        body = transport._http_get_bytes(url, "application/xml")
        if body is None:
            if len(transport.transport_failures) > before:
                failure = transport.transport_failures[-1]
                record["failures"].append(failure.to_manifest())
                if not failure.retryable:
                    break
            continue
        name = f"gaia_cone_{retrieval_id}_{len(record['responses']):02d}.xml"
        # Retained even when parsing or row validation fails.
        write_bytes(output / name, body)
        response = {"path": name, "uri": url,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        record["responses"].append(response)
        try:
            rows = _complete_rows(body)
        except Exception as exc:
            # An HTTP 200 with a truncated or malformed table: the other mirror may answer completely.
            parse_error = exc
            response["reason"] = f"{type(exc).__name__}: {exc}"
            continue
        record.update(status="available", row_count=len(rows))
        write_json(output / GAIA_CONE_QUERY_FILE, record)
        return rows, record
    if parse_error is not None and not any(not item["retryable"] for item in record["failures"]):
        record.update(status="unavailable", reason=f"{type(parse_error).__name__}: {parse_error}")
        write_json(output / GAIA_CONE_QUERY_FILE, record)
        raise AcquisitionError("Gaia cone response is incomplete or invalid", retryable=True) from parse_error
    record.update(status="unavailable", reason="no Gaia backend returned a complete response")
    write_json(output / GAIA_CONE_QUERY_FILE, record)
    failure = transport.transport_failures[-1] if transport.transport_failures else None
    error = AcquisitionError(record["reason"], retryable=bool(failure and failure.retryable))
    error.network_failure = failure
    raise error


def _complete_rows(body):
    """Rows of a complete cone VOTable with canonical source IDs; raises when incomplete."""
    parsed = parse_votable_bytes(body)
    table = parsed["table"]
    if list(table.colnames) != list(GAIA_SOURCE_COLUMNS):
        raise ValueError("Gaia response has unexpected columns")
    dtype = table["source_id"].dtype
    if dtype.kind != "i" or dtype.itemsize != 8:
        raise ValueError("Gaia source identifiers must be signed 64-bit integers")
    rows = [dict(zip(GAIA_SOURCE_COLUMNS, row)) for row in parsed["data"]]
    identities = [gaia_identifier(row["source_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Gaia response contains duplicate source identities")
    for row, source_id in zip(rows, identities):
        row["source_id"] = source_id
        for key in ("ra", "dec", "sep_arcsec"):
            if (isinstance(row[key], bool) or not isinstance(row[key], (int, float))
                    or not math.isfinite(row[key])):
                raise ValueError("Gaia response contains an incomplete astrometric row")
        if not 0 <= row["ra"] < 360 or not -90 <= row["dec"] <= 90 or row["sep_arcsec"] < 0:
            raise ValueError("Gaia response contains an invalid astrometric row")
    return rows
