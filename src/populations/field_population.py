"""Lossless Bulge/field population retained from the TRILEGAL LSST simulation.

The provider is the same precomputed TRILEGAL all-sky simulation used as the
TESS background fallback (Dal Tio et al. 2022, ApJS 262, 22, DOI
``10.3847/1538-4365/ac7be6``; table ``lsst_sim.simdr2``). This reader is the
Bulge/field contract, not the fallback contract: it keeps every simulated row,
including stars fainter than TESS 21 and stars whose Gaia colour cannot derive
a TESS magnitude, and it retains native position, Gaia photometry, distance
modulus, composition and extinction.

Units and selection
-------------------
``ra_deg``/``dec_deg`` are ICRS degrees; ``gaia_g_mag``/``bp_rp_mag`` are mag;
``mass_solar`` is ``M_sun``; ``logg_cgs`` is ``log10(cm/s^2)``; ``teff_k`` is K;
``radius_solar`` is ``R_sun`` derived from the simulated mass and surface
gravity (not a main-sequence relation, so an evolved star stays evolved);
``mh_dex`` is the simulation's [M/H]=log10(Z/Z_sun) [dex]; ``av_mag`` is the
per-star V-band extinction [mag]; ``distance_modulus_mag`` is m-M0 [mag] and
``distance_pc`` is its inverse ``10**((m-M0+5)/5)``, not a parallax. ``tess_mag``
is NaN where the retained Gaia colour cannot derive one; it is a convenience
column, never the population's selection depth.

The native Gaia G/BP/RP magnitudes are exposed through ``band_magnitudes``,
``band_valid`` and ``bands`` under their exact ``Gaia/G``, ``Gaia/BP`` and
``Gaia/RP`` ids. No other band is invented: a consumer observing another
instrument must refuse rather than select this population through Gaia or TESS.

``resolved_match`` marks a simulated star within the caller's duplicate-removal
radius of a resolved scene source. Those rows are kept so the record is
lossless, but ``unresolved_mask`` excludes them from the unresolved population
so the same object is not counted twice. Completeness is not modeled: the Gaia
G query limit is a selection depth, not a correction, and a Bulge FPP must
state that limit.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from trex.foundation.constants import G, Msun, Rsun

BULGE_FIELD_REQUIRED_COLUMNS = (
    "ra", "dec", "gaia_gmag", "g_bpmag", "g_rpmag", "mass", "logg", "logte",
    "m_h", "av", "mu0", "gc", "label", "tess_mag", "resolved_match",
)
BULGE_FIELD_RECORD_REQUIRED = (
    "status", "model", "provider", "table", "reference", "query_adql",
    "retrieved_utc", "ra_deg", "dec_deg", "cone_radius_deg", "solid_angle_sq_deg",
    "rows_returned", "rows_retained", "resolved_matches", "unresolved_rows",
    "selection", "completeness", "magnitude_systems", "population_sha256",
)


def _finite(value, name: str, path: Path, row: int, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bulge population has a non-numeric {name} at {path}:{row}") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"Bulge population has an invalid {name} at {path}:{row}")
    return number


def _optional_finite(value, name: str, path: Path, row: int) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bulge population has a non-numeric {name} at {path}:{row}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Bulge population has an invalid {name} at {path}:{row}")
    return number


def _integer(value, name: str, path: Path, row: int) -> int:
    """Require an exact integer; ``int(1.5)`` would silently relabel the row."""
    number = _finite(value, name, path, row)
    if not number.is_integer():
        raise ValueError(f"Bulge population has a non-integer {name} at {path}:{row}")
    return int(number)


def load_bulge_field_population(path: Path) -> Dict[str, Any]:
    """Read every retained Bulge/field row into aligned arrays.

    No row is dropped and no band is substituted. ``resolved_match`` is kept as
    a per-row flag; ``unresolved_mask`` is its complement for inference.
    """
    path = Path(path)
    ra, dec, g, bp, rp, mass, logg, logte, mh, av, mu0, gc, label, tess, resolved = (
        [] for _ in range(15)
    )
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("Bulge population has no header")
            header = [name.strip() for name in reader.fieldnames]
            missing = [name for name in BULGE_FIELD_REQUIRED_COLUMNS if name not in header]
            if missing:
                raise ValueError("Bulge population lacks required columns: " + ", ".join(missing))
            for row_number, row in enumerate(reader, 2):
                ra.append(_finite(row["ra"], "ra", path, row_number))
                dec.append(_finite(row["dec"], "dec", path, row_number))
                if not 0.0 <= ra[-1] < 360.0 or not -90.0 <= dec[-1] <= 90.0:
                    raise ValueError(f"Bulge population has an invalid ICRS position at {path}:{row_number}")
                g.append(_finite(row["gaia_gmag"], "gaia_gmag", path, row_number))
                bp.append(_finite(row["g_bpmag"], "g_bpmag", path, row_number))
                rp.append(_finite(row["g_rpmag"], "g_rpmag", path, row_number))
                mass.append(_finite(row["mass"], "mass", path, row_number, positive=True))
                logg.append(_finite(row["logg"], "logg", path, row_number))
                logte.append(_finite(row["logte"], "logte", path, row_number, positive=True))
                mh.append(_finite(row["m_h"], "m_h", path, row_number))
                av.append(_finite(row["av"], "av", path, row_number))
                if av[-1] < 0.0:
                    raise ValueError(f"Bulge population has negative extinction at {path}:{row_number}")
                mu0.append(_finite(row["mu0"], "mu0", path, row_number))
                gc.append(_integer(row["gc"], "gc", path, row_number))
                label.append(_integer(row["label"], "label", path, row_number))
                tess.append(_optional_finite(row.get("tess_mag"), "tess_mag", path, row_number))
                flag = str(row.get("resolved_match", "")).strip().lower()
                if flag not in ("0", "1", "true", "false", ""):
                    raise ValueError(f"Bulge population has an invalid resolved_match at {path}:{row_number}")
                resolved.append(flag in ("1", "true"))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Bulge population is unreadable: {path}") from exc
    if not ra:
        raise ValueError("Bulge population has no simulated stars")
    mass_solar = np.asarray(mass, dtype=float)
    logg_cgs = np.asarray(logg, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radius_solar = np.power(10.0, (
            math.log10(G * Msun) + np.log10(mass_solar) - logg_cgs
        ) / 2.0 - math.log10(Rsun))
    if not np.isfinite(radius_solar).all() or np.any(radius_solar <= 0.0):
        raise ValueError("Bulge population has an unrepresentable stellar radius")
    g_mag = np.asarray(g, dtype=float)
    bp_mag = np.asarray(bp, dtype=float)
    rp_mag = np.asarray(rp, dtype=float)
    band_values = {"Gaia/G": g_mag, "Gaia/BP": bp_mag, "Gaia/RP": rp_mag}
    arrays = {
        "ra_deg": np.asarray(ra, dtype=float),
        "dec_deg": np.asarray(dec, dtype=float),
        "gaia_g_mag": g_mag,
        "bp_rp_mag": bp_mag - rp_mag,
        "mass_solar": mass_solar,
        "logg_cgs": logg_cgs,
        "radius_solar": radius_solar,
        "teff_k": np.power(10.0, np.asarray(logte, dtype=float)),
        "mh_dex": np.asarray(mh, dtype=float),
        "av_mag": np.asarray(av, dtype=float),
        "distance_modulus_mag": np.asarray(mu0, dtype=float),
        "distance_pc": np.power(10.0, (np.asarray(mu0, dtype=float) + 5.0) / 5.0),
        "galactic_component": np.asarray(gc, dtype=int),
        "stage_label": np.asarray(label, dtype=int),
        "tess_mag": np.asarray(tess, dtype=float),
        "resolved_match": np.asarray(resolved, dtype=bool),
        "band_magnitudes": band_values,
        "band_valid": {band: np.isfinite(values) for band, values in band_values.items()},
        "bands": {
            band: {
                "column": column, "unit": "mag", "system": "Gaia DR2",
                "status": "retained" if bool(np.isfinite(values).all()) else "partially-missing",
            }
            for band, column, values in zip(
                band_values, ("gaia_gmag", "g_bpmag", "g_rpmag"), band_values.values(), strict=True
            )
        },
        "missing_bands": (),
    }
    arrays["unresolved_mask"] = ~arrays["resolved_match"]
    return arrays


def read_bulge_field_record(path: Path) -> Dict[str, Any]:
    """Read and validate the acquisition record that accompanies the population.

    A successful catalog query is not a calibrated population: the record must
    state its selection depth, solid angle, completeness limitation and the
    population digest it describes.
    """
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Bulge population record is unreadable: {path}") from exc
    if not isinstance(record, Mapping):
        raise ValueError("Bulge population record must be a JSON object")
    missing = [name for name in BULGE_FIELD_RECORD_REQUIRED if name not in record]
    if missing:
        raise ValueError("Bulge population record lacks: " + ", ".join(missing))
    if record.get("status") != "available":
        raise ValueError("Bulge population record is not an available retrieval")
    if not isinstance(record.get("selection"), Mapping) or not isinstance(record.get("magnitude_systems"), Mapping):
        raise ValueError("Bulge population record must state its selection and magnitude systems")
    if record["selection"].get("tess_limit_applied") is not None:
        raise ValueError("Bulge population record must not claim a TESS magnitude selection")
    solid_angle = float(record["solid_angle_sq_deg"])
    if not math.isfinite(solid_angle) or solid_angle <= 0.0:
        raise ValueError("Bulge population record has an invalid solid angle")
    return dict(record)


def unresolved_population(population: Mapping[str, Any]) -> Dict[str, Any]:
    """The unresolved subset of a retained field population, without deleting rows.

    ``resolved_match`` stars remain in the retained record but are excluded
    here because their resolved counterpart is already a scene source.
    """
    if not isinstance(population, Mapping) or "unresolved_mask" not in population:
        raise ValueError("field population requires its unresolved_mask")
    mask = np.asarray(population["unresolved_mask"], dtype=bool)
    if mask.ndim != 1 or not mask.size or not mask.any():
        raise ValueError("field population has no unresolved star")
    excluded = {"unresolved_mask", "resolved_match"}
    subset = {}
    for key, value in population.items():
        if key in excluded:
            continue
        if key in ("band_magnitudes", "band_valid"):
            subset[key] = {band: np.asarray(array)[mask] for band, array in value.items()}
            continue
        if key in ("bands", "missing_bands"):
            subset[key] = value
            continue
        array = np.asarray(value)
        if array.shape != mask.shape:
            raise ValueError("field population arrays must align with unresolved_mask")
        subset[key] = array[mask]
    subset["resolved_match"] = np.zeros(int(mask.sum()), dtype=bool)
    return subset


__all__ = [
    "BULGE_FIELD_RECORD_REQUIRED",
    "BULGE_FIELD_REQUIRED_COLUMNS",
    "load_bulge_field_population",
    "read_bulge_field_record",
    "unresolved_population",
]
