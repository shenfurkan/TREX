"""Acquire a calibrated ``TargetScene`` from a TIC identifier, without a pipeline workspace.

Every value comes from a retained archive response under ``output_dir``:
TIC v8 and Gaia DR3 field context, Gaia DR3 astrophysical parameters,
StarHorse EDR3 and SHBoost host tables, the ordered APOGEE -> GALAH ->
GSP-Spec -> SHBoost metallicity ladder, SPOC target-pixel files whose
apertures and WCS place every Gaia source for the Giacalone et al. (2021)
catalog flux-ratio scene (no pixel fit), and a TRILEGAL background
population. A missing required observation raises instead of being defaulted.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from functools import cache as memoize
from trex.populations.background import load_trilegal_population

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from trex.app.scene import TargetScene, measured_contrast_curve, with_measured_contrast_curve
from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC

from .aperture import acquire_catalog_aperture, aperture_reach_arcsec, colourless_aperture_upper_fraction
from .tpf import acquire_tpf_products, load_tpf
from .field import FIELD_CONTEXT_FILE, TIC_NEIGHBORS_QUERY_FILE, acquire_field_context
from .gaia_sources import GAIA_CONE_QUERY_FILE
from .tic import TIC_TARGET_QUERY_FILE
from .host_physics import select_host_physics
from .metallicity import acquire_metallicity
from .host_distributions import neighbor_parameter_entry
from .neighbor_tables import acquire_host_tables
from .scene_values import resolve_neighbor_properties
from .storage import write_json
from .transport import AcquisitionError, AcquisitionTransport, NetworkPolicy
from .background_population import (acquire_background_population, field_attenuation_envelope, is_fallback_record,
                                    record_simulates_target_field)

# Minimum Gaia cone radius [arcsec]. The cone actually queried is the larger of
# this and ``aperture_reach_arcsec`` over the sectors' SPOC apertures, which
# exceeds 60 arcsec for any aperture wider than about 2 TESS pixels.
DEFAULT_SEARCH_RADIUS_ARCSEC = 60.0
# A neighbour is characterized beyond FLAME/TIC/GSP-Phot only when its largest
# sector aperture flux fraction reaches 0.9 times the measured depth: a fully
# eclipsed source with a smaller share cannot produce the observed depth, so
# its unavailable reason is retained instead. The 0.9 leaves margin for the
# depth's own uncertainty.
SUPPLEMENTAL_DEPTH_SAFETY_FACTOR = 0.9
MIST_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.0_UBVRIplus.txz"

# Gaia DR3 astrophysical-parameter entry key -> archival row key used by the
# neighbour and host hierarchies (FLAME, GSP-Phot).
_AP_FIELDS = {
    "mass_flame_solar": "mass_flame", "radius_flame_solar": "radius_flame",
    "mass_flame_lower_solar": "mass_flame_lower_solar", "mass_flame_upper_solar": "mass_flame_upper_solar",
    "radius_flame_lower_solar": "radius_flame_lower_solar", "radius_flame_upper_solar": "radius_flame_upper_solar",
    "flags_flame": "flags_flame", "evolstage_flame": "evolstage_flame", "age_flame_gyr": "age_flame_gyr",
    "teff_gspphot_k": "teff_gspphot_k", "teff_gspphot_lower_k": "teff_gspphot_lower_k",
    "teff_gspphot_upper_k": "teff_gspphot_upper_k", "logg_gspphot_cgs": "logg_gspphot_cgs",
    "logg_gspphot_lower_cgs": "logg_gspphot_lower_cgs", "logg_gspphot_upper_cgs": "logg_gspphot_upper_cgs",
    "radius_gspphot_solar": "radius_gspphot_solar", "radius_gspphot_lower_solar": "radius_gspphot_lower_solar",
    "radius_gspphot_upper_solar": "radius_gspphot_upper_solar", "distance_gspphot_pc": "distance_gspphot_pc",
}


def with_astrophysical_parameters(source, ap_entry):
    """Return the Gaia source with its own DR3 AP values under archival keys (missing stays None).

    The host-physics and neighbour hierarchies read FLAME/GSP-Phot under these
    keys; ``scene_values`` applies its own FLAME domain and evolved-log g rules
    to them in both stages.
    """
    merged = dict(source)
    for key, name in _AP_FIELDS.items():
        merged[name] = None if ap_entry is None else ap_entry.get(key)
    return merged


def _latest_retrieval(queries):
    """Latest UTC retrieval time recorded in the host-table queries, or None."""
    times = [binding.get("retrieved_utc") for query in queries.values() if isinstance(query, dict)
             for binding in (query.get("source_queries") or {}).values() if isinstance(binding, dict)]
    times = [value for value in times if isinstance(value, str) and value]
    return max(times) if times else None


def _gaia_neighbor_record(host, neighbor_ids):
    """Shape the host tables like the Gaia-neighbour record the hierarchy consumes.

    ``retrieved_utc`` is the latest archive retrieval among the host tables'
    responses (a resumed scene keeps its original dates).
    """
    queries = host["queries"]
    entries = {source_id: neighbor_parameter_entry(
        host["sources"][source_id].get("gaia"), host["sources"][source_id].get("starhorse"),
        host["sources"][source_id].get("shboost"), host["sources"][source_id].get("shboost_refusal"),
        queries["shboost"]["source_queries"].get(source_id, {}),
        queries["gaia"], queries["starhorse"]) for source_id in neighbor_ids}
    return {"neighbors": entries,
            "queries": {"astrophysical_parameters": queries["gaia"], "starhorse": queries["starhorse"],
                        "shboost": queries["shboost"]},
            "retrieved_utc": _latest_retrieval(queries)}


def _mist_archive(cache_dir, transport):
    """Return the official MIST v1.2 release, retained once with its URL and retrieval date."""
    path = Path(cache_dir) / "MIST_v1.2_vvcrit0.0_UBVRIplus.txz"
    receipt = path.with_suffix(".retrieval.json")
    if path.is_symlink() or receipt.is_symlink():
        raise AcquisitionError("MIST release and its retrieval record must be regular files")
    if not path.is_file():
        if transport.download(MIST_URL, path) is None or not path.is_file():
            raise AcquisitionError("MIST release download did not complete")
        write_json(receipt, {"url": MIST_URL,
                             "retrieved_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
    record = _load_record(receipt)
    if record is None or record.get("url") != MIST_URL or not record.get("retrieved_utc"):
        raise AcquisitionError("MIST release lacks its original URL/retrieval-date record")
    return path, record


def _digest_bound_population(query_path, record):
    """The population CSV beside ``query_path`` if it still has the recorded digest, else None.

    The record's ``population_file`` may carry the writer's record prefix; the
    file itself is always beside its query record, so only its name is used.
    """
    population = query_path.parent / Path(record["population_file"]).name
    if query_path.is_symlink() or population.is_symlink() or not population.is_file():
        return None
    if hashlib.sha256(population.read_bytes()).hexdigest() != record.get("population_sha256"):
        return None
    return population


def _retained_trilegal(query_path, target, max_offset_arcsec):
    """Admit a retained TRILEGAL population only for this field and only with its recorded bytes."""
    record = _load_record(query_path)
    if record is None:
        raise ValueError(f"retained TRILEGAL query record {query_path} is unreadable")
    if not record_simulates_target_field(record):
        raise ValueError(
            "retained TRILEGAL population predates the Galactic field submission: its form sent the "
            "RA in degrees as hours, so it simulates another field; query TRILEGAL anew")
    population = _digest_bound_population(query_path, record)
    if population is None:
        raise ValueError("retained TRILEGAL population must be a regular file beside its query record, "
                         "unchanged since the record was written")
    offset = SkyCoord(record["ra_deg"] * u.deg, record["dec_deg"] * u.deg).separation(
        SkyCoord(target["ra_deg"] * u.deg, target["dec_deg"] * u.deg)).arcsec
    if not offset <= max_offset_arcsec:
        raise ValueError(f"retained TRILEGAL population was simulated {offset:.1f} arcsec from this target")
    return population, query_path


def _load_record(path):
    """Return a JSON object record, or ``None`` when unreadable/not an object."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _available_record(path):
    """Return the record only when its status is exactly ``"available"``."""
    payload = _load_record(path)
    return payload if payload is not None and payload.get("status") == "available" else None


def _find_available(directory, pattern, source_ids):
    """Return the newest available record of a stage for exactly ``source_ids``.

    Symlinked records are ignored; a record for another source list (an
    earlier field) is never reused.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob(pattern) if not path.is_symlink() and path.is_file()]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        payload = _available_record(path)
        if payload is not None and payload.get("source_ids") == source_ids:
            return payload
    return None


def _load_field(directory, tic_id):
    """Reconstruct retained field context, or return ``None`` when not available."""
    directory = Path(directory)
    context = _available_record(directory / FIELD_CONTEXT_FILE)
    tic_neighbors = _available_record(directory / TIC_NEIGHBORS_QUERY_FILE)
    if context is None or tic_neighbors is None or str(context.get("tic_id")) != str(tic_id):
        return None
    tic_payload = _load_record(directory / "tic_target.json")
    rows = tic_payload.get("data") if isinstance(tic_payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    if not isinstance(context.get("target"), dict) or not isinstance(context.get("neighbors"), list):
        return None
    return rows[0], context.get("target"), context.get("neighbors"), {
        "tic": _load_record(directory / TIC_TARGET_QUERY_FILE),
        "gaia": _load_record(directory / GAIA_CONE_QUERY_FILE),
        "tic_neighbors": tic_neighbors,
    }


def _load_trilegal(directory):
    """Return retained background when its record is available and digest-bound."""
    query_path = Path(directory) / "trilegal_query.json"
    record = _available_record(query_path)
    if record is None:
        return None
    if not record_simulates_target_field(record):
        raise ValueError(
            "the retained TRILEGAL population in {0} predates the Galactic field submission and simulates "
            "another field; acquire this scene in a new output_dir".format(query_path.parent))
    if is_fallback_record(record):
        # A fallback never stands in for TRILEGAL on a rerun: try TRILEGAL again.
        return None
    population = _digest_bound_population(query_path, record)
    return None if population is None else (population, query_path, record)


def _separations_from_target(target, neighbors):
    """Return neighbours with ``separation_arcsec`` from the target at the Gaia catalogue epoch."""
    if not neighbors:
        return []
    centre = SkyCoord(target["ra_deg"] * u.deg, target["dec_deg"] * u.deg)
    positions = SkyCoord(np.array([n["ra_deg"] for n in neighbors], dtype=float) * u.deg,
                         np.array([n["dec_deg"] for n in neighbors], dtype=float) * u.deg)
    separations = centre.separation(positions).arcsec
    return [{**neighbor, "separation_arcsec": float(value)} for neighbor, value in zip(neighbors, separations)]


def acquire_target_scene(tic_id, sectors, output_dir, *, depth_ppm, cache_dir=None, policy=None,
                         search_radius_arcsec=DEFAULT_SEARCH_RADIUS_ARCSEC, trilegal_population=None,
                         contrast_curve=None, contrast_band=None):
    """Acquire every scene input for ``tic_id`` and return the validated ``TargetScene``.

    ``depth_ppm`` is the measured transit depth that decides which faint
    neighbours need characterization. Aperture flux shares follow the
    Giacalone et al. (2021) catalog flux-ratio method on each sector's SPOC
    aperture. ``output_dir`` receives every raw response, query record
    and ``scene_acquisition.json``; ``cache_dir`` holds shared releases (GALAH
    DR3, MIST) and defaults to ``output_dir / "cache"``. The request
    (``tic_id``, ``sectors``, ``depth_ppm``, ``radius``,
    ``trilegal_population``) is written first to ``scene_request.json``; a
    rerun in the same directory must repeat it exactly, reusing every stage
    whose retained record is ``"available"`` (TPF products are revalidated by
    ``acquire_tpf_products`` on every rerun), and a different request raises.
    No high-resolution contrast curve is acquired. A measured one may be
    supplied (``contrast_curve``: two-column file or ``(separations [arcsec],
    delta_magnitudes)`` pair, with its ``contrast_band``; see
    ``trex.app.scene.with_measured_contrast_curve``); it is recorded in the
    request, bound in the scene and stated in ``scene_acquisition.json``.
    Without one, unresolved companions are bounded by the Gaia DR2 close-pair
    resolution (Giacalone et al. 2021, AJ 161, 24, Sect. 1).
    ``trilegal_population`` optionally names a retained TRILEGAL query record
    (``trilegal_query.json`` beside its population CSV, as TREX writes it) to
    use instead of a new simulation, like TRICERATOPS's ``trilegal_fname``; it
    must belong to this field and match its recorded digest.
    """
    if (isinstance(depth_ppm, bool) or not isinstance(depth_ppm, (int, float)) or not math.isfinite(depth_ppm)
            or not 0 < depth_ppm < 1e6):
        raise ValueError("depth_ppm must be finite in (0, 1e6)")
    sectors = [int(s) for s in sectors]
    if not sectors or len(set(sectors)) != len(sectors):
        raise ValueError("sectors must name at least one TESS sector, each once")
    if (isinstance(search_radius_arcsec, bool) or not isinstance(search_radius_arcsec, (int, float))
            or not math.isfinite(search_radius_arcsec) or search_radius_arcsec <= 0):
        raise ValueError("search_radius_arcsec must be a positive finite number")
    output = Path(output_dir)
    request = {"tic_id": int(tic_id), "sectors": sectors, "depth_ppm": float(depth_ppm),
               "radius": float(search_radius_arcsec),
               "trilegal_population": None if trilegal_population is None else str(trilegal_population)}
    contrast = None
    if contrast_curve is not None or contrast_band is not None:
        # Validated before any request is written; the measured values join the request so a
        # resumed output_dir can neither drop nor switch the curve. Requests without a curve
        # keep their earlier form and still resume.
        if contrast_curve is None:
            raise ValueError("contrast_band requires contrast_curve")
        contrast = measured_contrast_curve(contrast_curve, contrast_band)
        request["contrast_curve"] = contrast
    request_path = output / "scene_request.json"
    if output.exists() and any(output.iterdir()):
        existing = _load_record(request_path)
        if existing is None:
            raise ValueError(
                "output_dir is non-empty without a scene_request.json; it cannot be resumed "
                "and every retained file must belong to one scene")
        if existing != request:
            raise ValueError(
                "scene request differs from the retained scene_request.json; "
                "use a new output_dir for a different request")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(request_path, request)
    cache = output / "cache" if cache_dir is None else Path(cache_dir)
    for directory in (output, cache):
        directory.mkdir(parents=True, exist_ok=True)
    policy = NetworkPolicy(allow_network=True) if policy is None else policy
    transport = AcquisitionTransport(policy)

    # The Gaia cone must reach every star whose PSF can light a sector's SPOC aperture,
    # so the TPFs come first and widen the requested radius when needed. Retained
    # products pass the acquisition owner's identity, completeness and sector checks.
    products = acquire_tpf_products(tic_id, sectors, output / "aperture", transport)
    reach = max(aperture_reach_arcsec(load_tpf(output / "aperture" / product["path"], tic_id, product["sector"]))
                for product in products)
    cone_radius = max(float(search_radius_arcsec), reach)
    field = _load_field(output / "field", tic_id)
    if field is not None:
        tic, target, neighbors, field_queries = field
    else:
        tic, target, neighbors, field_queries = acquire_field_context(
            tic_id, output / "field", search_radius_arcsec=cone_radius, policy=policy)
    tic_record = field_queries["tic_neighbors"]
    # The cone is centred on the TIC J2000 position; a neighbour's separation is from the
    # target's own Gaia DR3 position at the same catalogue epoch (target proper motion excluded).
    neighbors = _separations_from_target(target, neighbors)
    target_id = target["source_id"]
    neighbor_ids = [neighbor["source_id"] for neighbor in neighbors]
    source_ids = [target_id, *neighbor_ids]
    host = _find_available(output / "host", "host_tables_*.json", source_ids)
    if host is None:
        host = acquire_host_tables(output / "host", transport, source_ids)
    target = with_astrophysical_parameters(target, host["sources"][target_id].get("gaia"))
    neighbors = [with_astrophysical_parameters(n, host["sources"][n["source_id"]].get("gaia")) for n in neighbors]
    physics = select_host_physics(target, tic)
    if target.get("parallax_mas") is None:
        raise AcquisitionError("Gaia DR3 host lacks the parallax a scene requires")

    aperture = _available_record(output / "aperture" / "aperture_scene.json")
    if aperture is None:
        aperture = acquire_catalog_aperture(tic_id, sectors, target, neighbors, output / "aperture", transport,
                                            products=products)
    fractions = {source_id: [] for source_id in source_ids}
    upper_fractions = {}
    tess_photometry = {}
    for sector in aperture["sectors"]:
        for contribution in sector["source_contributions"]:
            source_id = contribution["source_id"]
            if source_id not in fractions:
                raise AcquisitionError("aperture scene names a source outside this Gaia field")
            magnitude, relation = contribution.get("tess_mag"), contribution.get("tess_mag_relation")
            if (isinstance(magnitude, bool) or not isinstance(magnitude, (int, float))
                    or not math.isfinite(magnitude) or not isinstance(relation, str) or not relation.strip()):
                raise AcquisitionError(f"aperture scene lacks TESS magnitude provenance for Gaia source {source_id}")
            # The aperture changes by sector, but the catalog magnitude used to predict flux does not.
            if tess_photometry.setdefault(source_id, (magnitude, relation)) != (magnitude, relation):
                raise AcquisitionError(f"aperture scene disagrees on the TESS magnitude of Gaia source {source_id}")
            fraction = contribution["aperture_flux_fraction"]
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
                raise AcquisitionError(f"aperture scene has an invalid flux fraction for Gaia source {source_id}")
            fractions[source_id].append({"sector": sector["sector"], "fraction": fraction})
            if source_id != target_id and relation == "stassun2019-eq2-no-valid-color":
                # Older retained apertures have the same nominal share but no derived bound.
                upper = contribution.get("aperture_flux_fraction_upper_95", colourless_aperture_upper_fraction(fraction))
                if isinstance(upper, bool) or not isinstance(upper, (int, float)) or not fraction <= upper <= 1.0:
                    raise AcquisitionError(f"aperture scene has an invalid bright-end fraction for Gaia source {source_id}")
                upper_fractions.setdefault(source_id, []).append({"sector": sector["sector"], "fraction": upper})
    if any(sorted(item["sector"] for item in items) != sorted(sectors) for items in fractions.values()):
        raise AcquisitionError("aperture scene does not cover every Gaia source in every requested sector once")
    tess_mag, tess_relation = tess_photometry[target_id]

    metallicity = _find_available(output / "metallicity", "metallicity_query_*.json", source_ids)
    if metallicity is None:
        metallicity = acquire_metallicity(output / "metallicity", cache, transport, source_ids)
    target_metallicity = metallicity["sources"][target_id]
    if target_metallicity["status"] != "measured":
        raise AcquisitionError("target metallicity unavailable: " + str(target_metallicity.get("reason")))

    if trilegal_population is None:
        loaded_trilegal = _load_trilegal(output / "background")
        if loaded_trilegal is not None:
            population, query_path, background = loaded_trilegal
        else:
            population, query_path = acquire_background_population(
                target["ra_deg"], target["dec_deg"], output / "background", policy)
            background = json.loads(query_path.read_text())
    else:
        population, query_path = _retained_trilegal(Path(trilegal_population), target, search_radius_arcsec)
        background = json.loads(query_path.read_text())

    mist_release = []

    @memoize
    def host_population():
        return load_trilegal_population(population, None)

    def mist():
        # Opened only if a neighbour reaches the MIST stage, then reused.
        if not mist_release:
            mist_release.append(_mist_archive(cache, transport) + (field_attenuation_envelope(background),))
        return mist_release[0]

    gaia_record = _gaia_neighbor_record(host, neighbor_ids)
    threshold = SUPPLEMENTAL_DEPTH_SAFETY_FACTOR * depth_ppm * 1e-6
    resolved, evaluations = [], {}
    for neighbor in neighbors:
        source_id = neighbor["source_id"]
        upper_rows = upper_fractions.get(source_id)
        properties = resolve_neighbor_properties(
            {**neighbor, "aperture_flux_fractions_by_sector": [
                item["fraction"] for item in (upper_rows if upper_rows is not None else fractions[source_id])]},
            source_id, tic_record, gaia_record, threshold, mist_loader=mist, population_loader=host_population)
        if "stellar_parameter_evaluations" in properties:
            evaluations[source_id] = properties["stellar_parameter_evaluations"]
        entry = metallicity["sources"][source_id]
        composition = ({"metallicity_provenance": entry["provenance"]} if entry["status"] == "measured"
                       else {"metallicity_unavailable_reason": entry["reason"]})
        resolved.append({
            "source_id": source_id,
            "M_s": properties.get("mass_solar"), "R_s": properties.get("radius_solar"),
            "stellar_parameter_unavailable_reason": properties.get("stellar_parameter_unavailable_reason"),
            "delta_mag": tess_photometry[source_id][0] - tess_mag,
            "separation_arcsec": neighbor["separation_arcsec"],
            "aperture_flux_fractions": fractions[source_id],
            **({"aperture_flux_fractions_upper_95": upper_rows,
                "aperture_flux_fraction_limitation": (
                    "Stassun et al. (2019) Eq. (1) bright end of the TIC v8 colour domain; "
                    "other sources held fixed, not a statistical 95% limit.")}
               if upper_rows is not None else {}),
            **{key: properties[key] for key in (
                "mass_solar_err_lower", "mass_solar_err_upper", "radius_solar_err_lower", "radius_solar_err_upper",
                "stellar_parameter_provenance", "stellar_parameter_distribution",
                "stellar_relations_calibration_status") if key in properties},
            **composition,
        })
    scene = TargetScene(
        tic_id=int(tic["ID"]), ra_deg=target["ra_deg"], dec_deg=target["dec_deg"],
        M_s_Msun=physics["mass_solar"], R_s_Rsun=physics["radius_solar"], Teff_K=physics["teff_k"],
        Tmag=tess_mag, plx_mas=target["parallax_mas"], sectors=sectors,
        contrast_separations=None, contrast_values=None, resolved_neighbors=resolved,
        target_aperture_flux_fractions=fractions[target_id], N_background=int(background["star_count"]),
        trilegal_cache=population, background_area_sq_deg=float(background["simulated_area_sq_deg"]),
        feh_dex=target_metallicity["provenance"]["value_dex"],
        metallicity_provenance=target_metallicity["provenance"],
        unresolved_resolution_limit_arcsec=GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC,
        neighbor_parameter_evaluations=evaluations,
    )
    unresolved_limitation = ("No high-resolution contrast curve; unresolved companions are bounded by the "
                             "Gaia DR2 close-pair limit.")
    if contrast is not None:
        scene = with_measured_contrast_curve(
            scene, (contrast["separations_arcsec"], contrast["delta_magnitudes"]), contrast["band"],
            source=contrast["source"])
        unresolved_limitation = scene.contrast_provenance["limitation"]
    write_json(output / "scene_acquisition.json", {
        "tic_id": int(tic["ID"]), "sectors": sectors, "gaia_source_id": target_id,
        "depth_ppm": depth_ppm, "aperture_method": aperture["method"],
        "host_field_sources": {**physics["field_sources"], "tess_mag": tess_relation,
                               "parallax_mas": "gaia-dr3-astrometry"},
        "minimum_capable_aperture_flux_fraction": threshold,
        "gaia_cone_radius_arcsec": cone_radius, "aperture_reach_arcsec": reach,
        "limitations": [
            aperture["limitation"],
            unresolved_limitation,
            "Only Gaia DR3 sources are modelled; a TIC v8 source without a Gaia DR3 row has no aperture share.",
            "No difference-image localization evidence is bound.",
        ],
        "scene": scene.to_dict(),
    })
    return scene
