"""TRILEGAL LSST simulation (Dal Tio et al. 2022) as the background-population fallback.

When the TRILEGAL web service cannot supply a population, the same TRILEGAL
code's precomputed all-sky simulation for the LSST footprint
(Dal Tio et al. 2022, ApJS 262, 22, DOI 10.3847/1538-4365/ac7be6; table
``lsst_sim.simdr2`` at the NOIRLab Astro Data Lab, read 2026-11-09) supplies
the stars of the same 0.1 deg^2 field. It is a fallback, never a substitute:
the record names the provider and the TRILEGAL failure that triggered it, and
a later acquisition tries TRILEGAL first again.

Differences from a TRILEGAL 1.6 web population, recorded with each result:
the simulation uses PARSEC v1.2S + COLIBRI tracks and the Galaxy calibration
of Dal Tio et al. (2022, Sec. 2), per-star extinction from the Planck 2014
dust map (the web form applies one asymptotic value), and photometry in the
LSST and Gaia (Maiz Apellaniz & Weiler 2018) systems only. The TESS magnitude
is derived from Gaia G, BP and RP with the TICv8 relation of Stassun et al.
(2019, AJ 158, 138; ``trex.acquisition.photometry``), and the relation used
for each star is counted in the record.

Coverage (Dal Tio et al. 2022, Sec. 2.1): declination below 5 deg, ecliptic
latitude within 10 deg, and the Galactic plane (|b| < 10 deg) below
declination 35 deg. A field must lie inside that union by at least the cone
radius and the table must return stars for it; otherwise no fallback exists.
"""

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlencode, urlparse

from .dust_map import validate_icrs_field
from .photometry import tess_magnitude_from_gaia_photometry
from .storage import write_bytes
from .transport import AcquisitionTransport
from .trilegal_errors import TrilegalUnavailableError

MODEL = "trilegal-lsst-sim-dr2"
FIELD_SUBMISSION = "icrs-cone-deg"
TAP_URL = "https://datalab.noirlab.edu/tap/sync"
TABLE = "lsst_sim.simdr2"
REFERENCE = "Dal Tio et al. (2022), ApJS 262, 22, DOI 10.3847/1538-4365/ac7be6"
FIELD_AREA_SQ_DEG = 0.1
CONE_RADIUS_DEG = math.sqrt(FIELD_AREA_SQ_DEG / math.pi)
# The web population stops at TESS = 21 (TRICERATOPS mag_lim=21 on the first
# filter of the TESS+2MASS system). G - T reaches 1.69 mag at BP - RP = 6, the
# red end of the TICv8 relation, so G < 23 keeps every star that can be
# brighter than T = 21; the TESS cut is applied after the conversion.
TESS_LIMIT = 21.0
GAIA_G_QUERY_LIMIT = 23.0
COLUMNS = ("gaia_gmag", "g_bpmag", "g_rpmag", "mass", "logg", "logte", "m_h", "av", "mu0", "gc", "label")
POPULATION_HEADER = ("TESS", "Mact", "logg", "logTe", "[M/H]", "Av", "m-M0", "Gc", "label")
RAW_FILE_NAME = "lsst_sim_output.csv"
POPULATION_FILE_NAME = "trilegal_cache.csv"
RECORD_FILE_NAME = "trilegal_query.json"

# Bulge/field population: every simulated row is retained with its native
# coordinates, Gaia photometry, distance modulus and extinction. The TESS
# magnitude is a derived convenience column, never a row cut.
BULGE_FIELD_COLUMNS = ("ra", "dec", *COLUMNS)
BULGE_POPULATION_HEADER = (
    "ra", "dec", "gaia_gmag", "g_bpmag", "g_rpmag", "mass", "logg", "logte",
    "m_h", "av", "mu0", "gc", "label", "tess_mag", "tess_relation", "resolved_match",
)
BULGE_POPULATION_FILE_NAME = "bulge_field_population.csv"
BULGE_RECORD_FILE_NAME = "bulge_field_query.json"
BULGE_RAW_FILE_NAME = "bulge_field_raw.csv"
# A resolved scene source and a simulated field star at the same position are
# one object counted twice. This is a duplicate-removal radius, not a
# completeness or blending scale; the caller states it with the population.
BULGE_DUPLICATE_RADIUS_ARCSEC = 2.0


def _unavailable(message, failure=None):
    """A fallback outage attributed to the Astro Data Lab endpoint, not TRILEGAL."""
    return TrilegalUnavailableError(message, failure, endpoint=urlparse(TAP_URL).hostname)


def _field_coordinates(ra_deg, dec_deg):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    position = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return (float(position.galactic.b.deg),
            float(position.barycentrictrueecliptic.lat.deg))


def footprint_reason(ra_deg, dec_deg):
    """Return why a 0.1 deg^2 cone is not fully inside the simulated area, or None.

    ``ValueError`` for a centre that is not a finite ICRS position.
    """
    validate_icrs_field(ra_deg, dec_deg)
    gal_b, ecl_beta = _field_coordinates(ra_deg, dec_deg)
    margin = CONE_RADIUS_DEG
    if dec_deg < 5.0 - margin or abs(ecl_beta) < 10.0 - margin or (abs(gal_b) < 10.0 - margin and dec_deg < 35.0 - margin):
        return None
    return ("field (Dec {0:.2f}, b {1:.2f}, ecliptic latitude {2:.2f} deg) is not inside the "
            "LSST simulation footprint by the cone radius").format(dec_deg, gal_b, ecl_beta)


def _adql(select, ra_deg, dec_deg):
    return ("SELECT {0} FROM {1} WHERE q3c_radial_query(ra, dec, {2!r}, {3!r}, {4!r}) = 't' "
            "AND gaia_gmag < {5!r}").format(select, TABLE, float(ra_deg), float(dec_deg),
                                            CONE_RADIUS_DEG, GAIA_G_QUERY_LIMIT)


def _tap_csv(transport, query):
    # CSV carries no completeness marker, so each table query is paired with a
    # COUNT(*) of the same selection; MAXREC is far above any 0.1 deg^2 field.
    body = transport.post_form(TAP_URL, urlencode({
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "MAXREC": "2000000", "QUERY": query,
    }).encode("ascii"), accept="text/csv")
    if body is None:
        failure = transport.transport_failures[-1] if transport.transport_failures else None
        raise _unavailable(
            "LSST simulation query failed: {0}".format(failure.reason if failure else "no response"), failure)
    text = body.decode("utf-8")
    if text.lstrip().startswith("<"):
        raise _unavailable("LSST simulation query returned an error document: {0}".format(text[:300]))
    return body, text


def population_from_rows(text):
    """Convert the TAP CSV to the TRILEGAL population contract; return (csv, count, relations)."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or [name.strip() for name in reader.fieldnames] != list(COLUMNS):
        raise _unavailable("LSST simulation response has unexpected columns: {0}".format(reader.fieldnames))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(POPULATION_HEADER)
    count, relations, av_max = 0, {}, 0.0
    for row in reader:
        values = {}
        for name in ("mass", "logg", "logte", "m_h", "av", "mu0", "gc", "label"):
            try:
                values[name] = float(row[name])
            except (TypeError, ValueError):
                raise _unavailable("LSST simulation row has a non-numeric {0}: {1!r}".format(
                    name, row[name])) from None
            if not math.isfinite(values[name]):
                raise _unavailable("LSST simulation row has a non-finite {0}".format(name))
        # Every simulated star bounds the field attenuation, including those fainter than the TESS cut.
        av_max = max(av_max, values["av"])
        tess, relation = tess_magnitude_from_gaia_photometry(row["gaia_gmag"], row["g_bpmag"], row["g_rpmag"])
        if tess is None:
            raise _unavailable("LSST simulation row has no Gaia G magnitude")
        if not tess < TESS_LIMIT:
            continue
        relations[relation] = relations.get(relation, 0) + 1
        if not (values["gc"].is_integer() and values["label"].is_integer()):
            raise _unavailable("LSST simulation row has a non-integer Galactic component or stage label")
        writer.writerow(["{0:.4f}".format(tess), row["mass"], row["logg"], row["logte"], row["m_h"],
                         row["av"], row["mu0"], int(values["gc"]), int(values["label"])])
        count += 1
    if count == 0:
        raise _unavailable("LSST simulation returned no star brighter than TESS {0}".format(TESS_LIMIT))
    return out.getvalue(), count, relations, av_max


def acquire_lsst_sim_population(ra_deg, dec_deg, output_dir, policy, *, trigger=None, candidate_id=None,
                                record_prefix=Path(), writer=write_bytes):
    """Retain the LSST-simulation stars of the 0.1 deg^2 field as a fallback population.

    ``trigger`` is the TRILEGAL stage record (or message) that made the
    fallback necessary; it is kept in the population record. ``candidate_id``
    is the caller's identifier, recorded with ``source: candidate-data``.
    """
    reason = footprint_reason(ra_deg, dec_deg)
    if reason is not None:
        raise _unavailable("no LSST simulation fallback: " + reason)
    transport = AcquisitionTransport(policy)
    _count_body, count_text = _tap_csv(transport, _adql("COUNT(*) AS n", ra_deg, dec_deg))
    try:
        expected = int(count_text.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise _unavailable("LSST simulation count query is unreadable: {0!r}".format(count_text[:200])) from None
    if expected == 0:
        raise _unavailable("no LSST simulation fallback: the table holds no star in this field")
    query = _adql(", ".join(COLUMNS), ra_deg, dec_deg)
    raw, text = _tap_csv(transport, query)
    returned = max(0, len(text.strip().splitlines()) - 1)
    if returned != expected:
        raise _unavailable(
            "LSST simulation response is incomplete: {0} of {1} rows".format(returned, expected))
    population, star_count, relations, av_max = population_from_rows(text)
    population_bytes = population.encode("utf-8")
    record = {
        "status": "available", "schema_version": 1, "model": MODEL, "fallback": True,
        "fallback_trigger": trigger, "provider": "NOIRLab Astro Data Lab", "tap_url": TAP_URL, "table": TABLE,
        "reference": REFERENCE, "query_adql": query,
        "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ra_deg": ra_deg, "dec_deg": dec_deg, "field_submission": FIELD_SUBMISSION,
        "cone_radius_deg": CONE_RADIUS_DEG, "simulated_area_sq_deg": FIELD_AREA_SQ_DEG,
        "gaia_g_query_limit": GAIA_G_QUERY_LIMIT, "tess_limit": TESS_LIMIT, "rows_returned": returned,
        "tess_relation_counts": relations,
        # Upper end of the field attenuation envelope for neighbour inference,
        # the role extinction_infty has in a TRILEGAL web record.
        "field_av_max": av_max,
        "field_av_source": ("largest per-star V-band extinction among the simulated stars of the field "
                            "with Gaia G < {0:g} (the query limit), Planck 2014 dust map as applied by "
                            "Dal Tio et al. 2022").format(GAIA_G_QUERY_LIMIT),
        "tess_relation_reference": "Stassun et al. (2019), AJ 158, 138, Eqs. 1-2 (TICv8)",
        "model_differences": (
            "PARSEC v1.2S + COLIBRI tracks and the Dal Tio et al. (2022) Galaxy calibration; per-star "
            "Planck 2014 extinction; TESS magnitude derived from Gaia G/BP/RP"),
        "population_file": (record_prefix / POPULATION_FILE_NAME).as_posix(),
        "population_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "raw_output_file": (record_prefix / RAW_FILE_NAME).as_posix(), "star_count": star_count,
    }
    if candidate_id is not None:
        record.update(candidate_id=candidate_id, source="candidate-data")
    output = Path(output_dir)
    writer(output / RAW_FILE_NAME, raw)
    writer(output / POPULATION_FILE_NAME, population_bytes)
    writer(output / RECORD_FILE_NAME, (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return output / POPULATION_FILE_NAME, output / RECORD_FILE_NAME


def _separation_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation [arcsec] between two ICRS positions [deg]."""
    first, second = math.radians(dec1), math.radians(dec2)
    delta_ra = math.radians(ra1 - ra2)
    cosine = (math.sin(first) * math.sin(second)
              + math.cos(first) * math.cos(second) * math.cos(delta_ra))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _resolved_positions(resolved_sources):
    """Normalize resolved scene sources to ``(source_id, ra_deg, dec_deg)``."""
    positions = []
    for source in resolved_sources or ():
        if isinstance(source, Mapping):
            source_id = source.get("source_id")
            ra, dec = source.get("ra_deg"), source.get("dec_deg")
        else:
            try:
                source_id, ra, dec = source
            except (TypeError, ValueError) as exc:
                raise ValueError("resolved sources must be (source_id, ra_deg, dec_deg)") from exc
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("resolved source requires a nonempty identity")
        ra, dec = float(ra), float(dec)
        if not (math.isfinite(ra) and math.isfinite(dec)) or not 0.0 <= ra < 360.0 or not -90.0 <= dec <= 90.0:
            raise ValueError("resolved source position must be finite ICRS degrees")
        positions.append((source_id, ra, dec))
    return tuple(positions)


def bulge_field_from_rows(text, *, resolved_sources=(), duplicate_radius_arcsec=BULGE_DUPLICATE_RADIUS_ARCSEC):
    """Convert the TAP CSV into a lossless field population; return (csv, record).

    Every row the provider returned is written, including stars fainter than
    TESS 21 and stars whose Gaia colour cannot derive a TESS magnitude. The
    TESS column is therefore partly unknown, not a selection cut. A simulated
    star within ``duplicate_radius_arcsec`` of a resolved scene source is
    flagged ``resolved_match=True`` so the unresolved population can exclude it
    without deleting the row or reclassifying the resolved source.
    """
    radius = float(duplicate_radius_arcsec)
    if not math.isfinite(radius) or radius <= 0.0:
        raise _unavailable("bulge duplicate radius must be finite and positive")
    resolved = _resolved_positions(resolved_sources)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise _unavailable("LSST simulation response has no header")
    header = [name.strip() for name in reader.fieldnames]
    missing = [name for name in BULGE_FIELD_COLUMNS if name not in header]
    if missing:
        raise _unavailable("LSST simulation response lacks column(s): {0}".format(", ".join(missing)))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BULGE_POPULATION_HEADER)
    relations, av_max, retained, resolved_matches = {}, 0.0, 0, 0
    tess_unavailable = 0
    for row in reader:
        values = {}
        for name in ("ra", "dec", "gaia_gmag", "g_bpmag", "g_rpmag", "mass", "logg", "logte",
                     "m_h", "av", "mu0", "gc", "label"):
            try:
                values[name] = float(row[name])
            except (TypeError, ValueError):
                raise _unavailable("LSST simulation row has a non-numeric {0}: {1!r}".format(
                    name, row[name])) from None
            if not math.isfinite(values[name]):
                raise _unavailable("LSST simulation row has a non-finite {0}".format(name))
        if not (0.0 <= values["ra"] < 360.0 and -90.0 <= values["dec"] <= 90.0):
            raise _unavailable("LSST simulation row has an invalid ICRS position")
        if values["mass"] <= 0.0 or values["logte"] <= 0.0 or values["av"] < 0.0:
            raise _unavailable("LSST simulation row has a non-physical mass, temperature or extinction")
        av_max = max(av_max, values["av"])
        tess, relation = tess_magnitude_from_gaia_photometry(row["gaia_gmag"], row["g_bpmag"], row["g_rpmag"])
        if tess is None:
            tess_unavailable += 1
            tess_text, relation_text = "", ""
        else:
            tess_text, relation_text = "{0:.6f}".format(tess), relation
            relations[relation] = relations.get(relation, 0) + 1
        if not (values["gc"].is_integer() and values["label"].is_integer()):
            raise _unavailable("LSST simulation row has a non-integer Galactic component or stage label")
        matched = any(
            _separation_arcsec(values["ra"], values["dec"], ra, dec) <= radius
            for _source_id, ra, dec in resolved
        )
        resolved_matches += int(matched)
        writer.writerow([
            "{0:.8f}".format(values["ra"]), "{0:.8f}".format(values["dec"]),
            row["gaia_gmag"], row["g_bpmag"], row["g_rpmag"], row["mass"], row["logg"],
            row["logte"], row["m_h"], row["av"], row["mu0"], int(values["gc"]), int(values["label"]),
            tess_text, relation_text, int(matched),
        ])
        retained += 1
    if retained == 0:
        raise _unavailable("LSST simulation returned no star in this field")
    record = {
        "rows_returned": retained,
        "rows_retained": retained,
        "resolved_matches": resolved_matches,
        "unresolved_rows": retained - resolved_matches,
        "tess_unavailable_rows": tess_unavailable,
        "tess_relation_counts": relations,
        "field_av_max": av_max,
        "duplicate_radius_arcsec": radius,
        "magnitude_systems": {
            "gaia": "Gaia DR2 G/BP/RP as stored by lsst_sim.simdr2 (Maiz Apellaniz & Weiler 2018)",
            "tess": "derived from Gaia G/BP/RP with Stassun et al. (2019) TICv8 Eqs. 1-2; empty where unavailable",
            "distance": "m-M0 [mag], a distance modulus, not a parallax",
            "extinction": "per-star A_V [mag] from the Planck 2014 dust map as applied by Dal Tio et al. (2022)",
        },
        "selection": {
            "gaia_g_limit": GAIA_G_QUERY_LIMIT,
            "tess_limit_applied": None,
            "note": ("Gaia G < {0:g} is the provider query depth; no TESS magnitude cut is applied and "
                     "the population is not corrected for catalog incompleteness").format(GAIA_G_QUERY_LIMIT),
        },
        "completeness": ("not modeled: the Gaia G query limit is a selection depth, not a completeness "
                         "correction; a Bulge FPP from this population must state that limit"),
        "solid_angle_sq_deg": FIELD_AREA_SQ_DEG,
    }
    return out.getvalue(), record


def acquire_bulge_field_population(ra_deg, dec_deg, output_dir, policy, *, resolved_sources=(),
                                   duplicate_radius_arcsec=BULGE_DUPLICATE_RADIUS_ARCSEC,
                                   candidate_id=None, record_prefix=Path(), writer=write_bytes):
    """Retain a justified Bulge/field population with every simulated row.

    The provider is the TRILEGAL LSST simulation (Dal Tio et al. 2022) already
    used as the TESS fallback. This path differs from the fallback: it does not
    cut on a derived TESS magnitude, keeps native coordinates and distance,
    and flags simulated stars that duplicate a resolved scene source. The
    query depth, area and unmodeled completeness travel with the record.
    """
    reason = footprint_reason(ra_deg, dec_deg)
    if reason is not None:
        raise _unavailable("no Bulge field population: " + reason)
    transport = AcquisitionTransport(policy)
    _count_body, count_text = _tap_csv(transport, _adql("COUNT(*) AS n", ra_deg, dec_deg))
    try:
        expected = int(count_text.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise _unavailable("LSST simulation count query is unreadable: {0!r}".format(count_text[:200])) from None
    if expected == 0:
        raise _unavailable("no Bulge field population: the table holds no star in this field")
    query = _adql(", ".join(BULGE_FIELD_COLUMNS), ra_deg, dec_deg)
    raw, text = _tap_csv(transport, query)
    returned = max(0, len(text.strip().splitlines()) - 1)
    if returned != expected:
        raise _unavailable(
            "LSST simulation response is incomplete: {0} of {1} rows".format(returned, expected))
    population, statistics = bulge_field_from_rows(
        text, resolved_sources=resolved_sources, duplicate_radius_arcsec=duplicate_radius_arcsec
    )
    population_bytes = population.encode("utf-8")
    record = {
        "status": "available", "schema_version": 1, "model": MODEL, "provider": "NOIRLab Astro Data Lab",
        "tap_url": TAP_URL, "table": TABLE, "reference": REFERENCE, "query_adql": query,
        "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ra_deg": ra_deg, "dec_deg": dec_deg, "field_submission": FIELD_SUBMISSION,
        "cone_radius_deg": CONE_RADIUS_DEG, **statistics,
        "population_file": (record_prefix / BULGE_POPULATION_FILE_NAME).as_posix(),
        "population_sha256": hashlib.sha256(population_bytes).hexdigest(),
        "raw_output_file": (record_prefix / BULGE_RAW_FILE_NAME).as_posix(),
    }
    if candidate_id is not None:
        record.update(candidate_id=candidate_id, source="candidate-data")
    output = Path(output_dir)
    writer(output / BULGE_RAW_FILE_NAME, raw)
    writer(output / BULGE_POPULATION_FILE_NAME, population_bytes)
    writer(output / BULGE_RECORD_FILE_NAME,
           (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return output / BULGE_POPULATION_FILE_NAME, output / BULGE_RECORD_FILE_NAME
