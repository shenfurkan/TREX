"""Replay retained TREX likelihood draws, never substitute fitted parameters.

Replay calls TREX's scalar light-curve model. The quadratic-disc kernel
follows Mandel & Agol (2002), ApJ 580, L171, DOI ``10.1086/345520``.

Time, period, exposure and bin width are relative days, semimajor axis is cm,
radii use the draw's solar/Earth units, and angles are degrees. Models are
returned in normalized observed-aperture flux, undoing the same host-frame
renormalization used for the likelihood. A maximum-likelihood Monte Carlo
draw is not a posterior estimate; nested evidence does not supply a new fit.
"""

import math

import numpy as np

from trex.models.likelihoods import simulate_EB, simulate_TP
from trex.foundation._numerics import _require_positive_width, phase_bin_widths


def replay_representative_model(time, record, *, exptime_days, bin_integration_days, nsamples):
    """Evaluate an actual retained draw or reject incomplete/nonphysical state."""
    values = np.asarray(time)
    if values.dtype.kind not in "iuf":
        raise ValueError("fit replay times must be real, not boolean or text")
    values = values.astype(float, copy=False)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("fit replay requires finite phase-relative times [days]")
    if not record.get("has_representative_draw") or record.get("representative_draw") is None:
        raise ValueError("no retained representative likelihood draw")
    draw = record["representative_draw"]
    if draw.get("origin") != "monte-carlo-max-lnL":
        raise ValueError("fit replay requires the actual Monte Carlo likelihood draw")
    required = ("a_cm", "R_s", "P_orb", "inc", "ecc", "w", "u1", "u2", "fluxratio_comp")
    if any(not isinstance(draw.get(key), (int, float, np.integer, np.floating))
           or isinstance(draw[key], (bool, np.bool_)) or not math.isfinite(draw[key]) for key in required):
        raise ValueError("representative draw is missing finite physical parameters")
    if draw["a_cm"] <= 0 or draw["R_s"] <= 0 or draw["P_orb"] <= 0 or not 0 <= draw["ecc"] < 1:
        raise ValueError("representative draw has nonphysical orbit or host parameters")
    upper_supported = (draw["fluxratio_comp"] <= 1 if "secondary_to_host_flux_ratio" in draw
                       else draw["fluxratio_comp"] < 1)
    if draw["fluxratio_comp"] < 0 or not upper_supported or type(draw.get("companion_is_host")) is not bool:
        raise ValueError("representative draw has invalid companion dilution")
    _require_positive_width(exptime_days, "fit replay exposure")
    try:
        bin_integration_days = phase_bin_widths(bin_integration_days, values.size)
    except ValueError as exc:
        raise ValueError("fit replay requires a retained non-negative bin width [days], "
                         "or per-bin widths replayed on the observed bins") from exc
    if isinstance(nsamples, (bool, np.bool_)) or not isinstance(nsamples, (int, np.integer)) or nsamples <= 0:
        raise ValueError("fit replay requires positive integer supersampling")
    fraction = record.get("aperture_flux_fraction")
    if (not isinstance(fraction, (int, float, np.integer, np.floating))
            or isinstance(fraction, (bool, np.bool_)) or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("fit replay requires the retained calibrated aperture flux fraction")
    options = dict(
        time=values, P_orb=draw["P_orb"], inc_deg=draw["inc"], a_cm=draw["a_cm"],
        R_s_solar=draw["R_s"], u1=draw["u1"], u2=draw["u2"], exptime_days=exptime_days,
        ecc=draw["ecc"], argp_deg=draw["w"], companion_fluxratio=draw["fluxratio_comp"],
        companion_is_host=draw["companion_is_host"], nsamples=nsamples,
        bin_integration_days=bin_integration_days,
    )
    if draw.get("stellar_light_model") is not None:
        profiles = draw.get("angular_profiles", {})
        if not isinstance(profiles, dict) or profiles.keys() - {"angular_profile", "secondary_angular_profile"}:
            raise ValueError("retained angular profiles contain unknown model options")
        if "tabulated-angular" in draw["stellar_light_model"] and "angular_profile" not in profiles:
            raise ValueError("selected angular model draw lacks its actual original rays")
        options.update({key: tuple(np.asarray(part, dtype=float) for part in value)
                        for key, value in profiles.items()})
    if draw.get("is_EB") is True and draw.get("is_planet") is False:
        ratio_key = ("secondary_to_host_flux_ratio" if "secondary_to_host_flux_ratio" in draw else "fluxratio_EB")
        required_eb = ("R_EB", ratio_key, "u1_EB", "u2_EB")
        if any(not isinstance(draw.get(key), (int, float, np.integer, np.floating))
               or isinstance(draw[key], (bool, np.bool_)) or not math.isfinite(draw[key]) for key in required_eb):
            raise ValueError("representative EB draw lacks its secondary parameters/atmosphere")
        if draw["R_EB"] <= 0 or draw[ratio_key] <= 0 or (ratio_key == "fluxratio_EB" and draw[ratio_key] >= 1):
            raise ValueError("representative EB draw has invalid secondary radius or flux")
        if ratio_key == "secondary_to_host_flux_ratio":
            options[ratio_key] = draw[ratio_key]
        model, _ = simulate_EB(
            R_EB_solar=draw["R_EB"], EB_fluxratio=draw.get("fluxratio_EB"),
            u1_EB=draw["u1_EB"], u2_EB=draw["u2_EB"], **options,
        )
    elif draw.get("is_planet") is True and draw.get("is_EB") is False:
        _require_positive_width(draw.get("R_p"), "representative planet radius")
        model = simulate_TP(R_p_earth=draw["R_p"], **options)
    else:
        raise ValueError("representative draw must identify exactly one likelihood family")
    if model.shape != values.shape or not np.isfinite(model).all():
        raise ValueError("representative likelihood model is not finite/aligned")
    return 1.0 + fraction * (model - 1.0)
