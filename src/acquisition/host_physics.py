"""Gaia DR3 / TIC v8 host stellar physics: parsing, mass derivation and source precedence."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from astropy import units as u
from astropy.constants import G, M_sun, R_sun

TIC_V8_TARGET_MASS_SOURCE = "tic-v8-stassun2019"
TIC_V8_STELLAR_SOURCE = TIC_V8_TARGET_MASS_SOURCE
TIC_V8_COOL_DWARF_SOURCE = "tic-v8-cool-dwarf-list-stassun2019"
TIC_V8_COOL_DWARF_LIST_TAG = "cooldwarfs_v8"
GAIA_FLAME_MASS_SOURCE = "gaia-dr3-flame"
GAIA_GSPPHOT_SOURCE = "gaia-dr3-gspphot"
DERIVED_GRAVITATIONAL_MASS_SOURCE = "derived-gravitational-logg-radius"
ANDRAE_2023_GSPPHOT_LOGG_ERROR_INFLATION = 10.0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_value(value: Any) -> Optional[float]:
    """``value`` as a finite float, else None (booleans and non-numbers included)."""
    if not _is_number(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_positive(value: Any) -> Optional[float]:
    """``value`` as a positive finite float, else None."""
    number = _finite_value(value)
    return number if number is not None and number > 0.0 else None


def _symmetric_error(lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    """Half the width of Gaia's 16th-84th percentile interval as a 1-sigma, or None.

    None when either bound is missing or non-finite, or the bounds are inverted.
    """
    lower_value = _finite_value(lower)
    upper_value = _finite_value(upper)
    if lower_value is None or upper_value is None or upper_value < lower_value:
        return None
    return (upper_value - lower_value) / 2.0


def _derived_mass_from_logg_radius(logg_cgs: Any, radius_solar: Any) -> Optional[float]:
    """``M = g R**2 / G`` in solar masses, with ``g = 10**logg_cgs`` in cm s^-2 and R in R_sun.

    Astropy (CODATA/IAU) ``G``, ``M_sun`` and ``R_sun``. ``logg_cgs`` needs only
    to be finite (any finite log gravity gives a positive g); ``radius_solar``
    must be finite and positive. Returns None otherwise or when g overflows.
    """
    logg_value = _finite_value(logg_cgs)
    radius = _finite_positive(radius_solar)
    if logg_value is None or radius is None:
        return None
    try:
        gravity_cgs = math.pow(10.0, logg_value)
    except OverflowError:
        return None
    if not math.isfinite(gravity_cgs) or gravity_cgs <= 0.0:
        return None
    mass_solar = (gravity_cgs * u.cm / u.s**2 * (radius * R_sun) ** 2 / G).to_value(M_sun)
    return mass_solar if math.isfinite(mass_solar) and mass_solar > 0.0 else None


def _derived_mass_error_from_logg_radius(
    mass_solar: float,
    logg_sigma: Optional[float],
    radius_solar: float,
    radius_sigma: Optional[float],
) -> Optional[float]:
    """First-order ``sigma_M`` for ``M = g R**2 / G``.

    ``sigma_g / g = ln(10) sigma_logg`` and M is quadratic in R, so
    ``sigma_M / M = sqrt((ln(10) sigma_logg)**2 + (2 sigma_R / R)**2)``. None
    when an uncertainty is missing, non-finite or negative, or the result is zero.
    """
    if logg_sigma is None or radius_sigma is None:
        return None
    if not (math.isfinite(logg_sigma) and logg_sigma >= 0.0):
        return None
    if not (math.isfinite(radius_sigma) and radius_sigma >= 0.0):
        return None
    relative = math.hypot(math.log(10.0) * logg_sigma, 2.0 * radius_sigma / radius_solar)
    mass_err = mass_solar * relative
    return mass_err if math.isfinite(mass_err) and mass_err > 0.0 else None


def _is_tic_cool_dwarf(row: Dict[str, Any]) -> bool:
    """Return whether a TIC v8 row carries the Cool Dwarf List tag in ``splists``."""
    tags = row.get("splists")
    if not isinstance(tags, str):
        return False
    return TIC_V8_COOL_DWARF_LIST_TAG in {tag.strip() for tag in tags.split(",")}


def _tic_stellar_physics(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract TIC v8 Teff, log g, radius, and mass with their catalog errors.

    Returns ``None`` unless Teff, radius, and mass are finite and positive and
    log g is finite; the errors are kept as reported (``None`` when absent) and
    metallicity stays ``None`` when TIC v8 has no value, never a default.
    """
    teff = _finite_positive(row.get("Teff"))
    radius = _finite_positive(row.get("rad"))
    mass = _finite_positive(row.get("mass"))
    logg = _finite_value(row.get("logg"))
    if teff is None or radius is None or mass is None or logg is None:
        return None
    return {
        "teff_k": teff,
        "teff_k_err": _finite_positive(row.get("e_Teff")),
        "logg_cgs": logg,
        "logg_err_cgs": _finite_positive(row.get("e_logg")),
        "radius_solar": radius,
        "radius_solar_err": _finite_positive(row.get("e_rad")),
        "mass_solar": mass,
        "mass_solar_err": _finite_positive(row.get("e_mass")),
        "feh": _finite_value(row.get("MH")),
        "feh_err": _finite_positive(row.get("e_MH")),
    }


def _calibrated_gspphot_logg_error(formal_error: Optional[float]) -> Optional[float]:
    """Return the Andrae (2023)-inflated GSP-Phot log g error, or None."""
    if formal_error is None:
        return None
    return float(formal_error) * ANDRAE_2023_GSPPHOT_LOGG_ERROR_INFLATION


def resolve_target_mass(target: Dict[str, Any], tic_mass) -> tuple:
    """Apply the three permitted target-mass sources in precedence order.

    FLAME must pass the same retained quality/applicability owner used for
    neighbours, including its own radius and giant-age constraints.
    ``target`` carries normalized Gaia DR3 fields (``mass_flame``,
    ``logg_gspphot_cgs``, ``radius_gspphot_solar`` and their bounds);
    ``tic_mass`` is called only when FLAME and ``M = g R**2 / G`` both fail and
    returns ``(mass, error)`` or ``None``. Returns ``(mass_solar,
    mass_solar_err, field_source)``; ``field_source`` is ``None`` exactly when
    no permitted source applies.
    """
    from .scene_values import SceneBuildError, flame_neighbor_properties

    try:
        flame = flame_neighbor_properties(target, str(target.get("source_id", "target")))
    except SceneBuildError:
        flame = None
    if flame is not None:
        return (
            flame["mass_solar"],
            _symmetric_error(target.get("mass_flame_lower_solar"), target.get("mass_flame_upper_solar")),
            GAIA_FLAME_MASS_SOURCE,
        )
    radius_value = target.get("radius_gspphot_solar")
    radius_bounds = ("radius_gspphot_lower_solar", "radius_gspphot_upper_solar")
    if _finite_positive(radius_value) is None:
        radius_value = None if flame is None else flame["radius_solar"]
        radius_bounds = ("radius_flame_lower_solar", "radius_flame_upper_solar")
    derived_mass = _derived_mass_from_logg_radius(target.get("logg_gspphot_cgs"), radius_value)
    if derived_mass is not None:
        derived_err = _derived_mass_error_from_logg_radius(
            derived_mass,
            _calibrated_gspphot_logg_error(
                _symmetric_error(target.get("logg_gspphot_lower_cgs"), target.get("logg_gspphot_upper_cgs"))
            ),
            float(radius_value),
            _symmetric_error(target.get(radius_bounds[0]), target.get(radius_bounds[1])),
        )
        return derived_mass, derived_err, DERIVED_GRAVITATIONAL_MASS_SOURCE
    fallback = tic_mass()
    if fallback is not None:
        return fallback[0], fallback[1], TIC_V8_TARGET_MASS_SOURCE
    return None, None, None


def select_host_physics(target: Dict[str, Any], tic_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the scene host's Teff [K], radius and mass [solar] with field sources.

    Precedence: a TIC v8 Cool Dwarf List row (empirical M-dwarf calibration,
    Stassun et al. 2019 Sect. 3.3) -> Gaia DR3 GSP-Phot Teff/radius with the
    FLAME -> g R^2/G -> TIC v8 mass chain -> TIC v8 Teff/radius/mass. GSP-Phot
    is used only when its Teff, log g and radius are all present, so the host
    Teff, radius and the g R^2/G mass come from one GSP-Phot solution; a row
    lacking log g takes the complete TIC v8 set instead. Raises ``ValueError``
    when no source supplies all three; nothing is defaulted.
    """
    tic_physics = None if tic_row is None else _tic_stellar_physics(tic_row)
    teff = _finite_positive(target.get("teff_gspphot_k"))
    radius = _finite_positive(target.get("radius_gspphot_solar"))
    if tic_physics is not None and _is_tic_cool_dwarf(tic_row):
        source = TIC_V8_COOL_DWARF_SOURCE
    elif teff is not None and radius is not None and _finite_value(target.get("logg_gspphot_cgs")) is not None:
        def tic_mass():
            mass = None if tic_row is None else _finite_positive(tic_row.get("mass"))
            return None if mass is None else (mass, _finite_positive(tic_row.get("e_mass")))

        mass, mass_err, mass_source = resolve_target_mass(target, tic_mass)
        if mass is None:
            raise ValueError("Gaia DR3 host has no permitted mass source (FLAME, g R^2/G, TIC v8)")
        return {"teff_k": teff, "radius_solar": radius, "mass_solar": mass, "mass_solar_err": mass_err,
                "field_sources": {"teff_k": GAIA_GSPPHOT_SOURCE, "radius_solar": GAIA_GSPPHOT_SOURCE,
                                  "mass_solar": mass_source}}
    elif tic_physics is not None:
        source = TIC_V8_STELLAR_SOURCE
    else:
        raise ValueError("host lacks GSP-Phot Teff/log g/radius and a complete TIC v8 Teff/radius/mass")
    return {"teff_k": tic_physics["teff_k"], "radius_solar": tic_physics["radius_solar"],
            "mass_solar": tic_physics["mass_solar"], "mass_solar_err": tic_physics["mass_solar_err"],
            "field_sources": dict.fromkeys(("teff_k", "radius_solar", "mass_solar"), source)}
