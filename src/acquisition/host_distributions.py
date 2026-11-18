"""Catalogue stellar parameters for scene hosts and neighbours, never solar defaults.

Normalizes Gaia DR3 astrophysical parameters (GSP-Phot, FLAME), applies the
StarHorse EDR3 domain rules of Anders et al. (2022), builds SHBoost marginal
pseudo-posteriors from Khalatyan et al. (2024), A&A 691, A98, Appendix A, and
assembles the per-neighbour parameter record the scene retains.
"""

import math

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.special import ndtr, ndtri

# Solar surface gravity G M_sun / R_sun^2 in cm s^-2, from Astropy's IAU constants.
_SOLAR_GRAVITY_CGS = float((const.G * const.M_sun / const.R_sun**2).to_value(u.cm / u.s**2))


def _number(value, *, text=False):
    """A finite float, or None for missing, boolean or non-finite values.

    Archive VOTable rows carry numbers; SHBoost rows may carry numeric text,
    which ``text=True`` admits.
    """
    if isinstance(value, bool) or value is None:
        return None
    if not text and not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value):
    return _number(value)


def _text(value):
    """Stripped text of a catalogue string field, or None when absent or blank."""
    return (str(value).strip() or None) if value is not None else None


def _finite_positive(value):
    number = _finite_float(value)
    return number if number is not None and number > 0.0 else None


def gaia_ap_entry(row):
    """Normalize Gaia DR3 astrophysical parameters without inventing values."""
    if row is None:
        return None
    entry = {
        "teff_gspphot_k": _finite_positive(row.get("teff_gspphot")),
        "teff_gspphot_lower_k": _finite_positive(row.get("teff_gspphot_lower")),
        "teff_gspphot_upper_k": _finite_positive(row.get("teff_gspphot_upper")),
        "logg_gspphot_cgs": _finite_float(row.get("logg_gspphot")),
        "logg_gspphot_lower_cgs": _finite_float(row.get("logg_gspphot_lower")),
        "logg_gspphot_upper_cgs": _finite_float(row.get("logg_gspphot_upper")),
        "radius_gspphot_solar": _finite_positive(row.get("radius_gspphot")),
        "radius_gspphot_lower_solar": _finite_positive(row.get("radius_gspphot_lower")),
        "radius_gspphot_upper_solar": _finite_positive(row.get("radius_gspphot_upper")),
        "distance_gspphot_pc": _finite_positive(row.get("distance_gspphot")),
        "mass_flame_solar": _finite_positive(row.get("mass_flame")),
        "mass_flame_lower_solar": _finite_positive(row.get("mass_flame_lower")),
        "mass_flame_upper_solar": _finite_positive(row.get("mass_flame_upper")),
        "radius_flame_solar": _finite_positive(row.get("radius_flame")),
        "radius_flame_lower_solar": _finite_positive(row.get("radius_flame_lower")),
        "radius_flame_upper_solar": _finite_positive(row.get("radius_flame_upper")),
        "flags_flame": _text(row.get("flags_flame")),
        "evolstage_flame": None,
        "age_flame_gyr": _finite_positive(row.get("age_flame")),
    }
    stage = _finite_float(row.get("evolstage_flame"))
    if stage is not None and stage.is_integer():
        entry["evolstage_flame"] = int(stage)
    return entry if any(value is not None for value in entry.values()) else None


def starhorse_radius_solar(mass_solar, logg_cgs):
    """R/R_sun from g = G M / R^2 with M in M_sun and log g in cgs (any finite log g)."""
    return math.sqrt(float(mass_solar) * _SOLAR_GRAVITY_CGS / 10.0**float(logg_cgs))


def starhorse_entry(row):
    """Normalize StarHorse medians and retain its un-clipped 16–84 radius envelope."""
    if row is None:
        return None
    mass = _finite_positive(row.get("mass50"))
    logg = _finite_float(row.get("logg50"))
    entry = {
        "mass_solar": mass, "mass_lower_solar": _finite_positive(row.get("mass16")),
        "mass_upper_solar": _finite_positive(row.get("mass84")), "teff_k": _finite_positive(row.get("teff50")),
        "logg_cgs": logg, "logg_lower_cgs": _finite_float(row.get("logg16")),
        "logg_upper_cgs": _finite_float(row.get("logg84")),
        "distance_lower_kpc": _finite_positive(row.get("dist16")),
        "distance_kpc": _finite_positive(row.get("dist50")),
        "distance_upper_kpc": _finite_positive(row.get("dist84")),
        "av_mag": _finite_float(row.get("av50")), "astrometric_fidelity": _finite_float(row.get("fidelity")),
        "sh_outflag": _text(row.get("sh_outflag")), "sh_photoflag": _text(row.get("sh_photoflag")),
        "radius_solar": None, "radius_lower_solar": None, "radius_upper_solar": None,
    }
    # log g is a logarithm: zero or negative values (luminous giants) still give
    # a positive gravity 10**logg, so only finiteness is required.
    if mass is not None and logg is not None:
        entry["radius_solar"] = starhorse_radius_solar(mass, logg)
        mass_low, mass_high = entry["mass_lower_solar"], entry["mass_upper_solar"]
        logg_low, logg_high = entry["logg_lower_cgs"], entry["logg_upper_cgs"]
        if None not in (mass_low, mass_high, logg_low, logg_high):
            entry["radius_lower_solar"] = starhorse_radius_solar(mass_low, logg_high)
            entry["radius_upper_solar"] = starhorse_radius_solar(mass_high, logg_low)
    return entry if any(value is not None for value in entry.values()) else None


def starhorse_domain_refusal(starhorse):
    """Apply Anders et al. (2022) Sec. 3.2 fidelity and output-flag criteria.

    The large-uncertainty flag aggregates quantities, so use the paper's mass
    and log-g spread limits rather than rejecting a good mass for another label.
    """
    mass = starhorse.get("mass_solar")
    radius = starhorse.get("radius_solar")
    if mass is None or radius is None:
        return "StarHorse row has no finite mass or log g from which to derive a radius"
    fidelity = starhorse.get("astrometric_fidelity")
    if fidelity is None:
        return "StarHorse astrometric fidelity is missing"
    if fidelity <= 0.5:
        return "StarHorse astrometric fidelity {0:.3f} <= 0.5".format(fidelity)
    outflag = starhorse.get("sh_outflag")
    if not isinstance(outflag, str) or len(outflag) != 4 or not outflag.isdigit():
        return "StarHorse output flag is missing or malformed: {0!r}".format(outflag)
    if outflag[0] in "12":
        return "StarHorse output flag {0!r} marks too few consistent models".format(outflag)
    if outflag[1] != "0":
        return "StarHorse output flag {0!r} marks a significantly negative extinction".format(outflag)
    if outflag[3] != "0":
        return "StarHorse output flag {0!r} marks suspiciously small uncertainties".format(outflag)
    if outflag[2] != "0":
        mass_lower, mass_upper = starhorse.get("mass_lower_solar"), starhorse.get("mass_upper_solar")
        logg_lower, logg_upper = starhorse.get("logg_lower_cgs"), starhorse.get("logg_upper_cgs")
        if not all(value is not None for value in (mass_lower, mass_upper, logg_lower, logg_upper)):
            return "StarHorse output flag {0!r} marks large uncertainties and its spread is unavailable".format(outflag)
        mass_spread = 0.5 * (mass_upper - mass_lower) / mass
        logg_spread = 0.5 * (logg_upper - logg_lower)
        if mass_spread >= 1.0 or logg_spread >= 1.0:
            return (
                "StarHorse output flag {0!r} marks large uncertainties and the used quantities "
                "exceed the paper's threshold (mass half-range {1:.2f}, log g half-range {2:.2f})".format(
                    outflag, mass_spread, logg_spread)
            )
    photoflag = starhorse.get("sh_photoflag")
    if not isinstance(photoflag, str) or not photoflag.strip():
        return "StarHorse photometric-input flag is missing"
    return None


def finite(row, key):
    """``row[key]`` as a finite number (numeric text allowed); ValueError otherwise."""
    value = _number(row.get(key), text=True)
    if value is None:
        raise ValueError(f"missing or non-finite numeric {key}")
    return value


def positive(row, key):
    """``row[key]`` as a finite positive number; ValueError otherwise."""
    value = finite(row, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return value


def radius(mass, logg):
    """Vectorized R/R_sun from mass (M_sun) and cgs log g."""
    return np.sqrt(mass * _SOLAR_GRAVITY_CGS / 10.0**logg)


def positive_normal_quantile(mean, sigma, q):
    """Inverse CDF of a normal conditioned on a positive value (mass > 0)."""
    lower = ndtr(-mean / sigma)
    return mean + sigma * ndtri(lower + (1 - lower) * np.asarray(q))


def shboost_distribution(row):
    """Continuous product marginals and display-only quadrature.

    Gauss-Legendre summarizes the full inverse CDF, not a clipped sigma range.
    Mass is a normal conditioned on positivity; log g and log10(T/K) are normal.
    These are catalogue pseudo-posteriors, not a claimed joint stellar posterior.
    """
    means = [positive(row, k) for k in ("xgbdist_mass_mean", "xgbdist_logteff_mean")]
    logg = finite(row, "xgbdist_logg_mean")
    sigmas = [positive(row, k) for k in (
        "xgbdist_mass_std", "xgbdist_logg_std", "xgbdist_logteff_std")]
    if not str(row.get("xgb_inputflag") or "").startswith("XP"):
        raise ValueError("SHBoost requires an XP input flag")
    x, w = np.polynomial.legendre.leggauss(12)
    u, w = (x + 1) / 2, w / 2
    mass = positive_normal_quantile(means[0], sigmas[0], u)
    gravity = logg + sigmas[1] * ndtri(u)
    temperature = 10.0 ** (means[1] + sigmas[2] * ndtri(u))
    m, g, t = np.meshgrid(mass, gravity, temperature, indexing="ij")
    weights = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return {
        "mass_solar": m.ravel().tolist(), "radius_solar": radius(m, g).ravel().tolist(),
        "teff_k": t.ravel().tolist(), "weights": weights.tolist(),
        "marginals": {"mass_mean": means[0], "mass_std": sigmas[0],
                      "logg_mean": logg, "logg_std": sigmas[1],
                      "logteff_mean": means[1], "logteff_std": sigmas[2]},
        "method": "SHBoost continuous product-marginal inverse CDF; 12-node quadrature for display only",
        "limitations": (
            "SHBoost marginal pseudo-posteriors; label covariance is unavailable. "
            "Mass normal conditioned on M>0; output flags retained, not used to erase broad errors. "
            "Unresolved multiplicity/XP blending is not modeled. Label-covariance/model sensitivity "
            "requires remote assessment; not a calibrated joint posterior. This measure does not adjudicate "
            "conflicting external catalog solutions; retained alternatives must be bracketed. Display quantiles "
            "use quadrature; evidence integrates the continuous marginals without a sigma cut."
        ),
    }


def weighted_quantiles(values, weights, q):
    """Midpoint weighted quantiles avoid biasing the 12-node SHBoost median low."""
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) - weights[order] / 2
    return np.interp(q, cdf / weights.sum(), values[order])


def distribution_values(distribution, provenance):
    """Use marginal medians for display; evidence consumes the entire measure.

    ``mass_solar`` and ``radius_solar`` here are independent marginal medians,
    not a claimed joint stellar sample.  The scene retains the full joint
    measure and the TREX consumer integrates it. The returned distribution is
    a copy carrying the marginal 16th/84th percentiles (``*_p16_p84``); the
    same percentiles give the report-only asymmetric ``*_err_lower/upper``
    fields that other neighbour sources carry. The input is not modified.
    """
    distribution = dict(distribution)
    result = {"stellar_parameter_distribution": distribution,
              "stellar_parameter_source": provenance["source"],
              "stellar_parameter_provenance": provenance,
              "stellar_parameter_limitations": distribution["limitations"]}
    marginals = distribution.get("marginals")
    for name in ("mass_solar", "radius_solar"):
        if name == "mass_solar" and marginals is not None:
            # The continuous mass marginal is known; do not read it off 12 quadrature nodes.
            quantiles = positive_normal_quantile(marginals["mass_mean"], marginals["mass_std"], [.16, .5, .84])
        else:
            quantiles = weighted_quantiles(distribution[name], distribution["weights"], [.16, .5, .84])
        low, median, high = (float(value) for value in quantiles)
        result[name] = median
        distribution[name + "_p16_p84"] = [low, high]
        if 0.0 < low <= median <= high:
            result[name + "_err_lower"] = median - low
            result[name + "_err_upper"] = high - median
    return result


def _shboost_values(row, numeric_keys=()):
    """SHBoost columns without ``source_id``: ``xgbdist_*`` and ``numeric_keys`` as numbers, the rest as text."""
    return {key: (_number(value, text=True) if key.startswith("xgbdist_") or key in numeric_keys else _text(value))
            for key, value in row.items() if key != "source_id"}


def shboost_target_values(row):
    """The target's SHBoost row, including its de-reddened colour and absolute magnitude."""
    return None if row is None else _shboost_values(row, ("bprp0", "mg0"))


def shboost_entry(row):
    """Normalize one exact-ID XP row, retaining quality flags and broad errors."""
    if row is None:
        return None
    normalized = _shboost_values(row)
    for flag in ("xgb_mass_outputflag", "xgb_logg_outputflag", "xgb_logteff_outputflag"):
        if normalized.get(flag) not in ("0", "1"):
            raise ValueError("SHBoost output flag missing or malformed: " + flag)
    return {"values": normalized, "distribution": shboost_distribution(normalized)}


def neighbor_parameter_entry(ap_entry, starhorse, shboost, shboost_refusal, shboost_query, ap_record, starhorse_record):
    """One Gaia-neighbour record entry: retained source values, status and why values are missing.

    ``ap_entry``/``starhorse``/``shboost`` are normalized rows (or None),
    ``shboost_refusal`` why a returned SHBoost row was not admitted, and the two
    records the table queries, whose status distinguishes "no values" from an
    unavailable query.
    """
    entry = {"gaia_dr3_astrophysical_parameters": ap_entry, "starhorse_edr3": starhorse}
    if shboost is not None:
        entry["shboost"] = {**shboost, "query": shboost_query or {}}
    elif shboost_refusal is not None:
        entry["shboost_refusal"] = shboost_refusal
    reasons = []
    for value, name, record in ((ap_entry, "gaia-dr3-astrophysical-parameters", ap_record),
                                (starhorse, "starhorse-edr3", starhorse_record)):
        if value is None:
            reasons.append(f"{name}: no values" if record.get("status") == "available"
                           else f"{name}: query unavailable ({record.get('reason')})")
    if shboost is None and shboost_refusal is not None:
        reasons.append(f"shboost-xp: {shboost_refusal}")
    entry["status"] = "available" if ap_entry is not None or starhorse is not None or entry.get("shboost") else "unavailable"
    entry["reason"] = "; ".join(reasons) if reasons else None
    return entry

