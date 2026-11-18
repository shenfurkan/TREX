"""Shared fail-closed scene field and Gaia FLAME applicability rules."""

import math

GAIA_DR3_FLAME_METHOD = "gaia-dr3-flame"
STARHORSE_EDR3_METHOD = "starhorse-edr3"
GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD = "gaia-dr3-gspphot-teff-main-sequence-relation"
FLAME_RECOMMENDED_FLAG_SECOND_CHARACTER = "0"
FLAME_GIANT_FLAG_FIRST_CHARACTER = "1"
FLAME_MODEL_MASS_SOLAR_MIN = 0.5
FLAME_MODEL_MASS_SOLAR_MAX = 10.0
FLAME_GIANT_MASS_SOLAR_MIN = 1.0
FLAME_GIANT_MASS_SOLAR_MAX = 2.0
FLAME_GIANT_AGE_GYR_MIN = 1.0
# Below this GSP-Phot log g [cgs] a star is evolved and the main-sequence
# mass-Teff relation does not apply.
MAIN_SEQUENCE_MIN_LOGG_CGS = 3.5
GSPPHOT_MAIN_SEQUENCE_LIMITATION = (
    "Assumes a main-sequence star; GSP-Phot temperatures carry the systematics documented by Andrae et al. (2023).")
FLAME_GIANT_DWARF_REFUSAL = (
    "FLAME flags classify this source as a giant; a temperature-only dwarf replacement is not supported")
GSPPHOT_MISSING_LOGG_AFTER_FLAME_REFUSAL = (
    "rejected FLAME values leave the evolutionary state unresolved and the retained GSP-Phot log g is "
    "missing, so a temperature-only dwarf replacement is not supported")


class SceneBuildError(RuntimeError):
    """Retained evidence cannot produce a scientifically supported scene."""


def finite_number(value, name, *, positive=False):
    """``value`` as a finite (optionally positive) float; ``SceneBuildError`` naming ``name`` otherwise."""
    if isinstance(value, bool):
        raise SceneBuildError("{0} must be a finite number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneBuildError("{0} must be a finite number".format(name)) from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise SceneBuildError("{0} must be a {1} number".format(name, qualifier))
    return number


def _real(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def uncertainty_envelope(mass_solar, radius_solar, mass_lower, mass_upper, radius_lower, radius_upper):
    """Retain asymmetric 16-84 distances, not a marginalized error bar.

    A quantity gets ``*_err_lower``/``*_err_upper`` only when its value and both
    bounds are finite numbers with 0 < lower <= value <= upper.
    """
    fields = {}
    for name, value, lower, upper in (("mass_solar", mass_solar, mass_lower, mass_upper),
                                      ("radius_solar", radius_solar, radius_lower, radius_upper)):
        if all(_real(item) for item in (value, lower, upper)) and 0.0 < lower <= value <= upper:
            fields[f"{name}_err_lower"] = value - lower
            fields[f"{name}_err_upper"] = upper - value
    return fields


def _gspphot_main_sequence(teff_k, logg_cgs):
    """GSP-Phot Teff on the main-sequence relation: ``(properties, None)`` or ``(None, reason)``.

    Shared by the archival and supplemental stages so both refuse an evolved
    log g the same way.
    """
    import numpy as np
    from trex.stellar.stellar_relations import (
        main_sequence_mass_from_teff, stellar_radii_in_domain, STELLAR_RELATIONS_CALIBRATION_STATUS,
    )

    if _real(logg_cgs) and logg_cgs < MAIN_SEQUENCE_MIN_LOGG_CGS:
        return None, "GSP-Phot log g {0:.2f} is evolved; the main-sequence relation does not apply".format(logg_cgs)
    try:
        teff_k = finite_number(teff_k, "neighbor teff_gspphot_k", positive=True)
        mass_solar = main_sequence_mass_from_teff(teff_k)
        radius_solar = float(stellar_radii_in_domain(np.array([mass_solar]))[0])
    except (SceneBuildError, ValueError) as exc:
        return None, "temperature cannot be placed on the main-sequence relation: {0}".format(exc)
    return {"mass_solar": mass_solar, "radius_solar": radius_solar,
            "stellar_parameter_source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "teff_gspphot_k": teff_k,
            "stellar_parameter_limitations": GSPPHOT_MAIN_SEQUENCE_LIMITATION,
            "stellar_relations_calibration_status": STELLAR_RELATIONS_CALIBRATION_STATUS}, None


def _flame_classified_giant(flags_flame):
    """Whether FLAME's first flag character marks a giant.

    Gaia DR3 documentation release 1.3, Sec. 20.2.1, ``flags_flame``:
    A=1 marks a giant; A=2 means unavailable mass/age (read 2026-11-14).
    A giant classification is evolutionary evidence that a temperature-only
    main-sequence substitution would contradict, whether or not a usable FLAME
    mass and radius were retained.
    """
    return isinstance(flags_flame, str) and flags_flame[:1] == FLAME_GIANT_FLAG_FIRST_CHARACTER


def flame_record_values(ap_record):
    """Project a retained Gaia AP entry to the FLAME applicability inputs."""
    return {"mass_flame": ap_record.get("mass_flame_solar"),
             "radius_flame": ap_record.get("radius_flame_solar"),
             **{key: ap_record.get(key) for key in ("mass_flame_lower_solar", "mass_flame_upper_solar",
                                                  "radius_flame_lower_solar", "radius_flame_upper_solar")},
            "flags_flame": ap_record.get("flags_flame"),
            "evolstage_flame": ap_record.get("evolstage_flame"),
            "age_flame_gyr": ap_record.get("age_flame_gyr")}


def starhorse_neighbor_values(starhorse, gaia_record, source_id):
    """Retain StarHorse medians, asymmetric envelope and source-specific limitations."""
    mass = float(starhorse["mass_solar"])
    radius = float(starhorse["radius_solar"])
    query = gaia_record["queries"]["starhorse"]
    provenance = {
        "source": STARHORSE_EDR3_METHOD, "table": query.get("table"),
        "method": (
            "StarHorse EDR3 Bayesian isochrone fit to Gaia EDR3 astrometry and photometry; "
            "radius from R/Rsun = sqrt((M/Msun) * g_sun / 10**logg), a unit conversion of the "
            "posterior medians, not a second stellar model"
        ),
        "retrieval_url": query.get("uri"), "retrieved_utc": gaia_record.get("retrieved_utc"),
        "backend": query.get("backend"),
        "quality": {"astrometric_fidelity": starhorse.get("astrometric_fidelity"),
                    "sh_outflag": starhorse.get("sh_outflag"), "sh_photoflag": starhorse.get("sh_photoflag"),
                    "logg_cgs": starhorse.get("logg_cgs"), "distance_kpc": starhorse.get("distance_kpc"),
                    "av_mag": starhorse.get("av_mag")},
        "domain_checks": [
            "astrometric fidelity > 0.5 (Rybizki et al. 2022; Anders et al. 2022, Sec. 3.2)",
            "sh_outflag digits 1, 2 and 4 are 0; a set third digit was checked against the paper's mass and log g half-range thresholds",
            "finite mass, log g and photometric-input flags",
        ],
        "limitations": (
            "StarHorse posterior medians; the 16-84 envelope is retained but the evidence "
            "calculation keeps host mass and radius fixed. Photometric inputs: {0}."
        ).format(starhorse.get("sh_photoflag")),
    }
    if str(starhorse.get("sh_photoflag")) == "G":
        provenance["limitations"] += " Only Gaia G contributed, so the mass is prior-informed rather than colour-constrained."
    logg = starhorse.get("logg_cgs")
    if logg is not None and logg < 3.5:
        provenance["limitations"] += (
            " The retained host atmosphere/LDC path uses the main-sequence mass-Teff relation; "
            "log g {0:.2f} is evolved, so that relation is outside its intended domain and the "
            "host scenarios carry this stated model limitation.".format(logg)
        )
    return {
        "mass_solar": mass, "radius_solar": radius, "stellar_parameter_source": STARHORSE_EDR3_METHOD,
        "stellar_parameter_provenance": provenance, "stellar_parameter_limitations": provenance["limitations"],
        **uncertainty_envelope(mass, radius, starhorse.get("mass_lower_solar"), starhorse.get("mass_upper_solar"),
                               starhorse.get("radius_lower_solar"), starhorse.get("radius_upper_solar")),
    }


def supplemental_neighbor_properties(gaia_record, source_id, *, neighbor_flags_flame=None, flame_refused=False):
    """Evaluate supplemental FLAME, StarHorse and GSP-Phot in that order.

    ``gaia_record`` is the retained host-table record (``neighbors`` entries with
    Gaia DR3 AP, StarHorse and SHBoost values). Returns ``(properties, evaluations)``
    with ``properties`` None when every source is unavailable or refused;
    each evaluation records a source's status and reason. ``neighbor_flags_flame``
    and ``flame_refused`` carry the archival stage's FLAME context so both stages
    apply the same rule: a FLAME-classified giant, or a refused FLAME value
    without a retained GSP-Phot log g, blocks the temperature-only main-sequence
    substitution; otherwise GSP-Phot is evaluated on its own domain checks.
    """
    from .host_distributions import starhorse_domain_refusal

    evaluations = []
    entry = gaia_record["neighbors"].get(source_id) if isinstance(gaia_record, dict) else None
    if not isinstance(entry, dict):
        return None, [{"source": "gaia-neighbor-parameters", "status": "unavailable",
                       "reason": "no retained Gaia/StarHorse entry for this neighbour"}]
    ap_record = entry.get("gaia_dr3_astrophysical_parameters")
    if isinstance(ap_record, dict) and (
        ap_record.get("mass_flame_solar") is not None or ap_record.get("radius_flame_solar") is not None
    ):
        try:
            properties = flame_neighbor_properties(flame_record_values(ap_record), source_id)
        except SceneBuildError as exc:
            evaluations.append({"source": GAIA_DR3_FLAME_METHOD, "status": "refused", "reason": str(exc)})
        else:
            query = gaia_record["queries"]["astrophysical_parameters"]
            provenance = {
                "source": GAIA_DR3_FLAME_METHOD, "table": query.get("table"),
                "method": "FLAME mass and radius from GSP-Phot temperature and luminosity with the Gaia DR3 stellar models",
                "retrieval_url": query.get("uri"), "retrieved_utc": gaia_record.get("retrieved_utc"),
                "backend": query.get("backend"),
                "quality": {"flags_flame": properties.get("flame_flags"),
                            "evolstage_flame": properties.get("flame_evolutionary_stage")},
                "domain_checks": ["flags_flame second character '0' (recommended quality subset)",
                                  "mass within the documented 0.5-10 Msun FLAME model domain",
                                  "giant branch satisfies its 1-2 Msun and age > 1 Gyr constraint"],
                "limitations": "Gaia DR3 FLAME model values; host mass and radius stay fixed scenario inputs.",
            }
            properties.update({"stellar_parameter_provenance": provenance,
                               "stellar_parameter_limitations": provenance["limitations"],
                               **uncertainty_envelope(
                                   properties["mass_solar"], properties["radius_solar"],
                                   ap_record.get("mass_flame_lower_solar"), ap_record.get("mass_flame_upper_solar"),
                                   ap_record.get("radius_flame_lower_solar"), ap_record.get("radius_flame_upper_solar"))})
            evaluations.append({"source": GAIA_DR3_FLAME_METHOD, "status": "accepted", "reason": "inside its documented domain"})
            return properties, evaluations
    else:
        evaluations.append({"source": GAIA_DR3_FLAME_METHOD, "status": "unavailable",
                            "reason": "no FLAME mass or radius in the retained Gaia DR3 table"})
    starhorse = entry.get("starhorse_edr3")
    if isinstance(starhorse, dict):
        refusal = starhorse_domain_refusal(starhorse)
        if refusal is not None:
            evaluations.append({"source": "starhorse-edr3", "status": "refused", "reason": refusal})
        else:
            evaluations.append({"source": "starhorse-edr3", "status": "accepted",
                                "reason": "fidelity and output-flag domain checks passed"})
            return starhorse_neighbor_values(starhorse, gaia_record, source_id), evaluations
    else:
        evaluations.append({"source": "starhorse-edr3", "status": "unavailable",
                             "reason": "no StarHorse EDR3 row for this Gaia source"})
    flame_refused = flame_refused or any(
        item["source"] == GAIA_DR3_FLAME_METHOD and item["status"] == "refused" for item in evaluations)
    giant = (_flame_classified_giant(neighbor_flags_flame)
             or _flame_classified_giant(ap_record.get("flags_flame") if isinstance(ap_record, dict) else None))
    if giant:
        evaluations.append({"source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "status": "refused",
                            "reason": FLAME_GIANT_DWARF_REFUSAL})
        return None, evaluations
    teff_k = ap_record.get("teff_gspphot_k") if isinstance(ap_record, dict) else None
    logg = ap_record.get("logg_gspphot_cgs") if isinstance(ap_record, dict) else None
    if teff_k is None:
        evaluations.append({"source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "status": "unavailable",
                            "reason": "no GSP-Phot temperature in the retained Gaia DR3 table"})
        return None, evaluations
    if flame_refused and not _real(logg):
        evaluations.append({"source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "status": "refused",
                            "reason": GSPPHOT_MISSING_LOGG_AFTER_FLAME_REFUSAL})
        return None, evaluations
    properties, refusal = _gspphot_main_sequence(teff_k, logg)
    if properties is None:
        evaluations.append({"source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "status": "refused", "reason": refusal})
        return None, evaluations
    evaluations.append({"source": GAIA_DR3_GSPPHOT_MAIN_SEQUENCE_METHOD, "status": "accepted",
                        "reason": "GSP-Phot temperature inside the retained relation domain"})
    return properties, evaluations


def flame_neighbor_properties(neighbor, source_id):
    """Admit Gaia DR3 FLAME mass/radius only within the documented model domain.

    Mass and radius are solar units; giant ages are Gyr. Gaia DR3 FLAME quality
    and applicability follow Creevey et al. (2023), A&A 674 A26.
    """
    mass_solar = finite_number(neighbor.get("mass_flame"), "neighbor mass_flame", positive=True)
    radius_solar = finite_number(neighbor.get("radius_flame"), "neighbor radius_flame", positive=True)
    flags = neighbor.get("flags_flame")
    if not isinstance(flags, str) or len(flags) < 2:
        raise SceneBuildError("neighbor {0} lacks Gaia DR3 FLAME quality flags".format(source_id))
    if flags[1] != FLAME_RECOMMENDED_FLAG_SECOND_CHARACTER:
        raise SceneBuildError("neighbor {0} Gaia DR3 FLAME flag {1!r} is outside the recommended quality subset".format(source_id, flags))
    if not FLAME_MODEL_MASS_SOLAR_MIN <= mass_solar <= FLAME_MODEL_MASS_SOLAR_MAX:
        raise SceneBuildError("neighbor {0} Gaia DR3 FLAME mass is outside its documented model domain".format(source_id))
    age_gyr = None
    if flags[0] == FLAME_GIANT_FLAG_FIRST_CHARACTER:
        age_gyr = finite_number(neighbor.get("age_flame_gyr"), "neighbor age_flame_gyr", positive=True)
        if not (FLAME_GIANT_MASS_SOLAR_MIN <= mass_solar <= FLAME_GIANT_MASS_SOLAR_MAX and age_gyr > FLAME_GIANT_AGE_GYR_MIN):
            raise SceneBuildError("neighbor {0} Gaia DR3 FLAME giant does not meet its documented mass and age applicability constraint".format(source_id))
    # Gaia DR3 flags_flame first digit marks giants (1) and missing mass/age (2); the
    # documentation gives no null condition for evolstage_flame, so it is recorded, not required.
    evolutionary_stage = neighbor.get("evolstage_flame")
    if isinstance(evolutionary_stage, bool) or not isinstance(evolutionary_stage, int):
        evolutionary_stage = None
    return {"mass_solar": mass_solar, "radius_solar": radius_solar,
             "stellar_parameter_source": GAIA_DR3_FLAME_METHOD, "flame_flags": flags,
             "flame_evolutionary_stage": evolutionary_stage, "flame_age_gyr": age_gyr,
             **uncertainty_envelope(
                 mass_solar, radius_solar, neighbor.get("mass_flame_lower_solar"),
                 neighbor.get("mass_flame_upper_solar"), neighbor.get("radius_flame_lower_solar"),
                 neighbor.get("radius_flame_upper_solar"))}


def tic_neighbor_values(tic_record, source_id):
    """Admit retained TIC v8 mass/radius only within the stellar-relation domain."""
    import numpy as np
    from trex.stellar.stellar_relations import validate_stellar_masses
    from .tic import TIC_NEIGHBOR_METHOD

    if tic_record is None:
        return None
    entry = tic_record["neighbors"].get(source_id)
    if not isinstance(entry, dict) or entry.get("status") != "available":
        return None
    try:
        mass_solar = finite_number(entry.get("mass_solar"), "neighbor TIC mass", positive=True)
        radius_solar = finite_number(entry.get("radius_solar"), "neighbor TIC radius", positive=True)
        validate_stellar_masses(np.array([mass_solar]))
    except (SceneBuildError, ValueError):
        return None
    return {
        "mass_solar": mass_solar, "radius_solar": radius_solar,
        "stellar_parameter_source": TIC_NEIGHBOR_METHOD, "tic_id": str(entry.get("tic_id")),
        "stellar_parameter_limitations": (
            "TIC v8 catalog mass and radius as used by the reference TRICERATOPS "
            "for nearby stars; {0}".format(tic_record.get("cross_identification_limitation", ""))
        ),
    }


def unavailable_neighbor_properties(reason):
    """Preserve a resolved source whose stellar parameters remain unavailable."""
    return {"mass_solar": None, "radius_solar": None, "stellar_parameter_source": "unavailable",
            "stellar_parameter_unavailable_reason": reason}


def tic_neighbor_refusal(tic_record, source_id, gaia_reason, *, missing_tic_guidance="no current TIC neighbour record"):
    """Retain why no TIC v8 mass/radius applies to a resolved source."""
    if tic_record is None:
        return unavailable_neighbor_properties("{0}; {1}".format(gaia_reason, missing_tic_guidance))
    entry = tic_record["neighbors"].get(source_id)
    if not isinstance(entry, dict):
        return unavailable_neighbor_properties("{0}; the TIC neighbour record does not cover this source".format(gaia_reason))
    if entry.get("status") != "available":
        return unavailable_neighbor_properties("{0}; TIC: {1}".format(gaia_reason, entry.get("reason", "unavailable")))
    return unavailable_neighbor_properties("{0}; TIC mass/radius cannot be used".format(gaia_reason))


def _initial_neighbor_selection(neighbor, source_id, tic_record, *, missing_tic_guidance):
    """FLAME -> TIC -> GSP-Phot with each refusal retained as an evaluation.

    A refused FLAME value does not reject an independently applicable TIC v8
    mass/radius or a GSP-Phot temperature that carries finite main-sequence
    log-g evidence. FLAME's first flag character still marks a giant, and a
    refused FLAME solution without a retained GSP-Phot log g leaves the
    evolutionary state unresolved; neither is replaced by a temperature-only
    dwarf relation.
    """
    evaluations = []
    flame_reason = None
    if neighbor.get("mass_flame") is not None or neighbor.get("radius_flame") is not None:
        try:
            return flame_neighbor_properties(neighbor, source_id), None, evaluations
        except SceneBuildError as exc:
            flame_reason = "Gaia DR3 FLAME values are outside their documented applicability: {0}".format(exc)
            evaluations.append({"source": GAIA_DR3_FLAME_METHOD,
                                "status": "refused", "reason": flame_reason})
    tic_values = tic_neighbor_values(tic_record, source_id)
    if tic_values is not None:
        return tic_values, None, evaluations
    if _flame_classified_giant(neighbor.get("flags_flame")):
        reason = FLAME_GIANT_DWARF_REFUSAL
        if flame_reason is not None:
            reason = "{0}; {1}".format(flame_reason, reason)
        return None, reason, evaluations
    teff_k = neighbor.get("teff_gspphot_k")
    if teff_k is None:
        gaia_reason = flame_reason or "neighbor has neither Gaia DR3 FLAME values nor a GSP-Phot temperature"
        reason = tic_neighbor_refusal(tic_record, source_id, gaia_reason,
                                      missing_tic_guidance=missing_tic_guidance)
        return None, reason["stellar_parameter_unavailable_reason"], evaluations
    if flame_reason is not None and not _real(neighbor.get("logg_gspphot_cgs")):
        gaia_reason = "{0}; {1}".format(flame_reason, GSPPHOT_MISSING_LOGG_AFTER_FLAME_REFUSAL)
        reason = tic_neighbor_refusal(tic_record, source_id, gaia_reason,
                                      missing_tic_guidance=missing_tic_guidance)
        return None, reason["stellar_parameter_unavailable_reason"], evaluations
    properties, refusal = _gspphot_main_sequence(teff_k, neighbor.get("logg_gspphot_cgs"))
    if properties is None:
        gaia_reason = "{0}; {1}".format(flame_reason, refusal) if flame_reason is not None else refusal
        reason = tic_neighbor_refusal(tic_record, source_id, gaia_reason,
                                      missing_tic_guidance=missing_tic_guidance)
        return None, reason["stellar_parameter_unavailable_reason"], evaluations
    return properties, None, evaluations


def initial_neighbor_properties(neighbor, source_id, tic_record, *, missing_tic_guidance="no current TIC neighbour record"):
    """Try archival FLAME, TIC and archival GSP-Phot without stellar defaults.

    Returns (properties, unresolved_reason). A refused FLAME value still allows
    an independently applicable TIC v8 mass/radius, or a GSP-Phot temperature
    with finite main-sequence log-g evidence. A FLAME-classified giant, or a
    refused FLAME value without a retained GSP-Phot log g, keeps the
    temperature-only dwarf relation unavailable. GSP-Phot log g below
    ``MAIN_SEQUENCE_MIN_LOGG_CGS`` is refused here as in the supplemental
    stage, so an evolved star continues to StarHorse/SHBoost/MIST instead.
    """
    properties, reason, _evaluations = _initial_neighbor_selection(
        neighbor, source_id, tic_record, missing_tic_guidance=missing_tic_guidance)
    return properties, reason


def select_neighbor_properties(neighbor, source_id, tic_record=None, gaia_record=None,
                               minimum_capable_fraction=None, *, missing_tic_guidance="no current TIC neighbour record"):
    """Apply retained host hierarchy through SHBoost; return unresolved MIST inputs separately."""
    from .host_distributions import distribution_values

    initial, unresolved_reason, evaluations = _initial_neighbor_selection(
        neighbor, source_id, tic_record, missing_tic_guidance=missing_tic_guidance)
    if initial is not None:
        return initial, None, evaluations
    sectors = neighbor.get("aperture_flux_fractions_by_sector")
    largest_fraction = max(sectors) if isinstance(sectors, list) and sectors else 0.0
    if gaia_record is None or minimum_capable_fraction is None or largest_fraction < minimum_capable_fraction:
        return unavailable_neighbor_properties(unresolved_reason), None, evaluations
    flame_refused = any(item["source"] == GAIA_DR3_FLAME_METHOD and item["status"] == "refused"
                        for item in evaluations)
    supplemental, supplemental_evaluations = supplemental_neighbor_properties(
        gaia_record, source_id, neighbor_flags_flame=neighbor.get("flags_flame"),
        flame_refused=flame_refused)
    evaluations.extend(supplemental_evaluations)
    if supplemental is not None:
        return supplemental, None, evaluations
    entry = gaia_record.get("neighbors", {}).get(source_id, {})
    shboost = entry.get("shboost")
    if shboost is not None:
        distribution = shboost.get("distribution")
        if not isinstance(distribution, dict) or not all(key in distribution for key in ("method", "limitations")):
            evaluations.append({"source": "shboost-xp-marginals", "status": "refused",
                                "reason": "retained SHBoost marginal distribution is missing or malformed"})
            return None, unresolved_reason, evaluations
        query = shboost.get("query") or {}
        table_query = gaia_record.get("queries", {}).get("shboost") or {}
        if not query.get("uri"):
            query = table_query
        provenance = {"source": "shboost-xp-marginals", "table": table_query.get("table"),
                      "method": distribution["method"], "retrieval_url": query.get("uri"),
                      "retrieved_utc": gaia_record.get("retrieved_utc"), "backend": query.get("backend"),
                       "quality": shboost.get("values"),
                      "domain_checks": ["XP spectrum and positive finite marginal uncertainties"],
                      "limitations": distribution["limitations"]}
        return distribution_values(distribution, provenance), None, evaluations
    return None, unresolved_reason, evaluations


def resolve_neighbor_properties(neighbor, source_id, tic_record=None, gaia_record=None,
                                minimum_capable_fraction=None, *, missing_tic_guidance="no current TIC neighbour record",
                                mist_loader=None):
    """Single owner for FLAME -> TIC -> GSP-Phot -> supplements -> SHBoost -> MIST.

    Plain inputs only: ``mist_loader`` is a zero-argument callable returning
    ``(archive, receipt, extinction_max)`` or ``(archive, receipt,
    extinction_max, quality_extra)`` for a retained official MIST release,
    where ``receipt`` carries ``url``/``retrieved_utc`` and ``quality_extra``
    carries caller artifact bindings (e.g. a workspace grid/extinction
    artifact record). The caller owns artifact currency; a loader or MIST
    refusal becomes a recorded evaluation, never a silent default.
    """
    from .host_distributions import distribution_values
    from .mist import mist_g_distribution, read_mist_grid

    selected, unresolved_reason, evaluations = select_neighbor_properties(
        neighbor, source_id, tic_record, gaia_record, minimum_capable_fraction,
        missing_tic_guidance=missing_tic_guidance)
    if selected is not None:
        if evaluations:
            selected["stellar_parameter_evaluations"] = evaluations
        return selected
    # The G-only MIST route is for sources without a usable BP magnitude.
    if mist_loader is not None and not _real(neighbor.get("phot_bp_mean_mag")):
        try:
            loaded = mist_loader()
            archive, receipt, extinction = loaded[:3]
            quality_extra = loaded[3] if len(loaded) > 3 else {}
            distribution = mist_g_distribution(neighbor, read_mist_grid(archive), extinction)
        except (ValueError, KeyError) as exc:
            evaluations.append({"source": "mist-g-parallax", "status": "refused", "reason": str(exc)})
        else:
            provenance = {
                "source": "mist-g-parallax", "table": archive.name,
                "method": distribution["method"], "retrieval_url": receipt["url"],
                "retrieved_utc": receipt["retrieved_utc"],
                "backend": "official-MIST-release",
                "quality": {"parallax_snr": float(neighbor["parallax_mas"]) / float(neighbor["parallax_error_mas"]),
                            **dict(quality_extra)},
                "domain_checks": ["parallax S/N >= 20; low-extinction field; finite released grid"],
                "limitations": distribution["limitations"],
            }
            properties = distribution_values(distribution, provenance)
            properties["stellar_parameter_evaluations"] = [
                *evaluations, {"source": "mist-g-parallax", "status": "accepted",
                               "reason": "retained G/parallax inputs and released grid passed domain checks"}]
            return properties
    properties = unavailable_neighbor_properties(unresolved_reason)
    if evaluations:
        properties["stellar_parameter_evaluations"] = evaluations
    return properties
