"""Portable TargetScene -- stellar population context for TREX vetting.

Manages aperture stars, resolved neighbour catalogs, TRILEGAL background
star simulations, and contrast-curve data for a single TESS target.

The class is candidate-neutral: all target-specific data is passed via
constructor arguments or loaded from candidate-owned files.

Units, provenance, and fail-closed contract
-------------------------------------------
The scene contract implements the input side of Giacalone et al. (2021), AJ 161,
24, DOI ``10.3847/1538-3881/abc6af``, with the background
population from Girardi et al. (2005), A&A 436, 895, DOI
``10.1051/0004-6361:20042352``.  Coordinates are ICRS degrees; mass/radius are
``M_sun``/``R_sun``; temperature is K; TESS magnitude and contrast are mag and
delta-mag; parallax is mas; contrast separation is arcsec; and background count
is an integer count.  The retained TRILEGAL cache and SHA-256 binding are
required.  Missing/nonfinite scene fields, nonmonotonic contrast curves,
duplicate neighbors, or a hash mismatch raises instead of supplying a fallback.
The scene is conditional scenario input, not source attribution or a claim.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC

MEASURED_CONTRAST_CURVE_KIND = "measured-contrast-curve"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    from trex.populations._validation import real_scalar

    return real_scalar(value, name, positive=positive)


def _metallicity_record(record: object, value: object, *, source_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate the shared target/neighbour composition contract."""
    if not isinstance(record, Mapping):
        raise ValueError("metallicity_provenance must be a mapping")
    provenance = dict(record)
    if provenance.get("source") == "caller-supplied":
        required = {"source", "quantity", "value_dex", "uncertainty_dex"}
        references = [provenance[key] for key in ("citation", "uri") if key in provenance]
        if (not required <= set(provenance) or set(provenance) - required - {"citation", "uri"}
                or provenance["quantity"] not in ("[Fe/H]", "[M/H]")
                or not references or any(not isinstance(ref, str) or not ref.strip() for ref in references)
                or isinstance(provenance["value_dex"], (bool, np.bool_))
                or not isinstance(provenance["value_dex"], (int, float, np.integer, np.floating))
                or isinstance(provenance["uncertainty_dex"], (bool, np.bool_))
                or not isinstance(provenance["uncertainty_dex"], (int, float, np.integer, np.floating))
                or _finite_number(provenance["value_dex"], "metallicity value") != value):
            raise ValueError("caller metallicity provenance is incomplete or invalid")
        _finite_number(provenance["uncertainty_dex"], "metallicity uncertainty", positive=True)
        return copy.deepcopy(provenance)

    common = {"source", "quantity", "value_dex", "uncertainty_dex", "source_id",
              "retrieved_utc", "uri", "limitation"}
    extras = {
        "apogee-dr17": ({"apogee_id", "starflag", "fe_h_flag", "aspcapflag"}, "[Fe/H]"),
        "galah-dr3": ({"sobject_id", "dr2_source_id", "flag_sp", "flag_fe_h", "snr_c3_iraf"}, "[Fe/H]"),
        "gaia-gspspec-calibrated": ({"flags_gspspec", "logg_gspspec", "calibration"}, "[M/H]"),
        "shboost-xp": ({"input_flag", "metallicity_output_flag"}, "[M/H]"),
    }
    source = provenance.get("source")
    if source not in extras:
        raise ValueError("metallicity_provenance does not name an admitted catalogue source")
    extra, quantity = extras[source]
    # Every catalogue source carries a raw-row binding; legacy SHBoost records may omit it.
    if (not (common | extra) <= set(provenance) <= (common | extra | {"binding"})
            or provenance.get("quantity") != quantity
            or isinstance(provenance["value_dex"], (bool, np.bool_))
            or not isinstance(provenance["value_dex"], (int, float, np.integer, np.floating))
            or isinstance(provenance["uncertainty_dex"], (bool, np.bool_))
            or not isinstance(provenance["uncertainty_dex"], (int, float, np.integer, np.floating))
            or _finite_number(provenance["value_dex"], "metallicity value") != value
            or (source_id is not None and provenance["source_id"] != source_id)):
        raise ValueError("metallicity_provenance does not match the scene source or metallicity")
    uncertainty = _finite_number(provenance["uncertainty_dex"], "metallicity uncertainty", positive=True)
    patterns = {
        "source_id": r"^[1-9][0-9]{0,18}$",
        "retrieved_utc": (r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
                          r"T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]+)?(Z|\+00:00)$"),
        "uri": r"^https://[^/\s]+(/[^\s]*)?$",
        "limitation": r"\S",
    }
    if any(not isinstance(provenance[key], str) or not re.search(pattern, provenance[key])
           for key, pattern in patterns.items()):
        raise ValueError("metallicity provenance is incomplete or invalid")
    binding = provenance.get("binding")
    if binding is None and source != "shboost-xp":
        raise ValueError("metallicity provenance requires its raw-row binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise ValueError("metallicity provenance requires its raw-row binding")
    if source == "galah-dr3":
        if (set(binding) != {"artifact", "row_index_zero_based"}
                or not isinstance(binding["artifact"], str) or not binding["artifact"]
                or isinstance(binding["row_index_zero_based"], bool)
                or not isinstance(binding["row_index_zero_based"], int)
                or binding["row_index_zero_based"] < 0
                or provenance["flag_sp"] != 0 or provenance["flag_fe_h"] != 0
                or not isinstance(provenance["sobject_id"], int)
                or not isinstance(provenance["dr2_source_id"], str)
                or not re.search(patterns["source_id"], provenance["dr2_source_id"])
                or _finite_number(provenance["snr_c3_iraf"], "GALAH S/N") <= 30.0):
            raise ValueError("GALAH metallicity provenance is incomplete or invalid")
    elif source == "shboost-xp":
        binding_invalid = binding is not None and (
            set(binding) - {"columns"} != {"response", "values"}
            or not isinstance(binding.get("response"), str) or not binding.get("response")
            or not isinstance(binding.get("values"), Mapping)
        )
        if (binding_invalid or provenance["input_flag"] != "XP5pGBPRPJHKsW1W2W3W4"
                or provenance["metallicity_output_flag"] != "0"
                or uncertainty >= 0.3):
            raise ValueError("SHBoost flag 0 requires metallicity uncertainty below 0.3 dex")
    elif source == "apogee-dr17":
        if (set(binding) != {"response", "row"}
                or not isinstance(binding["response"], str) or not binding["response"]
                or not isinstance(binding["row"], Mapping)
                or not isinstance(provenance["apogee_id"], str) or not provenance["apogee_id"]
                or any(provenance[key] != 0 for key in ("starflag", "fe_h_flag", "aspcapflag"))):
            raise ValueError("APOGEE metallicity provenance is incomplete or invalid")
    else:
        if (set(binding) != {"response", "row"}
                or not isinstance(binding["response"], str) or not binding["response"]
                or not isinstance(binding["row"], Mapping)
                or not isinstance(provenance["flags_gspspec"], str)
                or len(provenance["flags_gspspec"]) < 13
                or "Recio-Blanco" not in str(provenance["calibration"])):
            raise ValueError("GSP-Spec metallicity provenance is incomplete or invalid")
    return copy.deepcopy(provenance)


def _require_tess_band(band):
    if band.strip().lower() != "tess":
        raise ValueError(
            "TREX evaluates companion contrasts in the TESS band only; "
            "no verified bandpass conversion is implemented, so a non-TESS "
            "contrast curve is refused"
        )


def _target_identity(record, tic_id) -> Optional[Dict[str, Any]]:
    """Validate an optional exact target identity; never synthesize a TIC id."""
    if record is None:
        return None
    from trex.acquisition.identity import CatalogIdentity

    try:
        identity = CatalogIdentity.from_record(record)
    except ValueError as exc:
        raise ValueError("target_identity is not a valid exact catalog identity") from exc
    if identity.namespace == "tic" and tic_id is not None and identity.native_id != str(int(tic_id)):
        raise ValueError("target identity does not match the scene TIC id")
    return identity.to_record()


def _target_source_id(value, target_identity, resolved_source_ids) -> Optional[str]:
    """Resolve the target's scene source id without inventing a catalogue id.

    With datasets, the target must be named by the same identity text used in
    every observation. An exact target identity can supply it; otherwise the
    caller states it. A value that collides with a resolved neighbour is
    refused because the two sources would become indistinguishable.
    """
    if value is None and target_identity is not None:
        value = target_identity["native_id"]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_source_id must be nonempty text")
    if value in resolved_source_ids:
        raise ValueError("target_source_id collides with a resolved neighbour identity")
    return value


def _dataset_band(value) -> Dict[str, Any]:
    """Validate a dataset's native band declaration.

    ``unit`` is the magnitude unit; no band conversion is performed here, so a
    band id that is not TESS is preserved with its own native system label.
    """
    if isinstance(value, str):
        value = {"id": value}
    if not isinstance(value, Mapping):
        raise ValueError("dataset band must be a mapping or band-id string")
    unknown = set(value) - {"id", "unit", "system", "reference", "population_column"}
    if unknown or not isinstance(value.get("id"), str) or not value["id"].strip():
        raise ValueError("dataset band requires an id and no unknown fields")
    band = {"id": value["id"], "unit": value.get("unit", "mag")}
    if band["unit"] != "mag":
        raise ValueError("dataset band magnitudes must be in magnitudes (unit 'mag')")
    for key in ("system", "reference", "population_column"):
        if value.get(key) is not None:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError("dataset band " + key + " must be nonempty text")
            band[key] = value[key]
    return band


def _dataset_aperture(value) -> Dict[str, Any]:
    """Validate the aperture/frame whose total is the share denominator."""
    if not isinstance(value, Mapping):
        raise ValueError("dataset aperture must be a mapping")
    unknown = set(value) - {"id", "kind", "denominator", "denominator_value", "frame"}
    if unknown:
        raise ValueError("dataset aperture has unknown fields")
    for key in ("id", "kind", "denominator"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError("dataset aperture requires " + key)
    aperture = {key: value[key] for key in ("id", "kind", "denominator")}
    if value.get("denominator_value") is not None:
        aperture["denominator_value"] = _finite_number(
            value["denominator_value"], "dataset aperture denominator", positive=True
        )
    if value.get("frame") is not None:
        if not isinstance(value["frame"], str) or not value["frame"].strip():
            raise ValueError("dataset aperture frame must be nonempty text")
        aperture["frame"] = value["frame"]
    return aperture


def _dataset_fraction(value, label: str):
    """Return a scalar or tuple of per-cadence fractions in [0, 1]."""
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(label + " must be a nonempty finite per-cadence sequence")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError(label + " is outside [0, 1]")
        return tuple(float(item) for item in array)
    scalar = _finite_number(value, label)
    if scalar < 0.0 or scalar > 1.0:
        raise ValueError(label + " is outside [0, 1]")
    return scalar


def _dataset_shares(value, source_ids, target_source_id, label: str) -> Dict[str, Dict[str, Any]]:
    """Normalize dataset shares onto every scene source.

    Accepts either ``{source_id: fraction}`` (construction) or the serialized
    ``[{"source_id", "fraction", "in_aperture"}]`` form. In the mapping
    shorthand an exact zero is the caller's outside-aperture declaration; the
    serialized form can state an in-aperture zero explicitly. A source declared
    outside the aperture keeps an exact zero share, not deletion. A source
    named neither here nor in ``outside_aperture`` is refused: "not modelled
    in this dataset" cannot be silently read as a zero.
    """
    flags: Dict[str, bool] = {}
    unset = object()
    if isinstance(value, Mapping) and value:
        rows = dict(value)
    elif isinstance(value, (list, tuple)) and value:
        rows = {}
        for row in value:
            if (not isinstance(row, Mapping) or set(row) - {"source_id", "fraction", "in_aperture"}
                    or not isinstance(row.get("source_id"), str)):
                raise ValueError(label + " share row is malformed")
            if row["source_id"] in rows:
                raise ValueError(label + " shares repeat a source")
            rows[row["source_id"]] = row["fraction"]
            flags[row["source_id"]] = row["in_aperture"] if "in_aperture" in row else True
    else:
        raise ValueError(label + " shares must be a non-empty mapping or list")
    unknown = set(rows) - set(source_ids)
    if unknown:
        raise ValueError(label + " shares name sources outside the scene")
    shares = {}
    for source_id, fraction in rows.items():
        normalized = _dataset_fraction(fraction, label + " fraction")
        explicit = flags.get(source_id, unset)
        if explicit is unset:
            # The mapping shorthand cannot state geometry; an exact zero is
            # the caller's outside-aperture declaration. The serialized list
            # form keeps an explicit in-aperture zero distinct.
            in_aperture = (
                any(value != 0.0 for value in normalized)
                if isinstance(normalized, tuple) else normalized != 0.0
            )
        else:
            in_aperture = explicit
        if not isinstance(in_aperture, (bool, np.bool_)):
            raise ValueError(label + " in_aperture must be boolean")
        if not in_aperture:
            if isinstance(normalized, tuple):
                if any(value != 0.0 for value in normalized):
                    raise ValueError(label + " outside-aperture source must have exactly zero share")
            elif normalized != 0.0:
                raise ValueError(label + " outside-aperture source must have exactly zero share")
        shares[source_id] = {"fraction": normalized, "in_aperture": bool(in_aperture)}
    return shares


def _dataset_cadence_ids(value, n_cadences: int) -> Optional[tuple]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != n_cadences:
        raise ValueError("dataset cadence_ids must match its per-cadence share length")
    normalized = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (str, int, np.integer)) or not str(item).strip():
            raise ValueError("dataset cadence ids must be nonempty text or integers")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError("dataset cadence ids must be unique")
    return tuple(normalized)


def _validated_dataset(record, source_ids, target_source_id, resolved_source_ids) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("dataset must be a mapping")
    unknown = set(record) - {
        "dataset_id", "band", "aperture", "shares", "outside_aperture",
        "magnitudes", "cadence_ids",
    }
    if unknown:
        raise ValueError("dataset has unknown fields: " + ", ".join(sorted(unknown)))
    dataset_id = record.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset requires a nonempty dataset_id")
    shares = _dataset_shares(record.get("shares"), source_ids, target_source_id, dataset_id)
    outside = record.get("outside_aperture", ())
    if isinstance(outside, str) or not isinstance(outside, (list, tuple)):
        raise ValueError(dataset_id + " outside_aperture must be a list of source ids")
    for source_id in outside:
        if source_id not in resolved_source_ids:
            raise ValueError(dataset_id + " outside_aperture names a source outside the resolved scene")
        if source_id in shares:
            raise ValueError(dataset_id + " source is both in and outside the aperture")
        shares[source_id] = {"fraction": 0.0, "in_aperture": False}
    missing = set(source_ids) - set(shares)
    if missing:
        raise ValueError(
            dataset_id + " does not state the aperture share of: " + ", ".join(sorted(missing))
        )
    lengths = {len(item["fraction"]) for item in shares.values() if isinstance(item["fraction"], tuple)}
    if len(lengths) > 1:
        raise ValueError(dataset_id + " per-cadence shares have inconsistent lengths")
    n_cadences = lengths.pop() if lengths else 1
    # A source inside the aperture cannot take more than the aperture's total;
    # a sum above one is an inconsistent denominator, not a modelling choice.
    for index in range(n_cadences):
        total = sum(
            item["fraction"][index] if isinstance(item["fraction"], tuple) else item["fraction"]
            for item in shares.values() if item["in_aperture"]
        )
        if total > 1.0 + 1e-9:
            raise ValueError(dataset_id + " aperture shares sum above one; denominator is inconsistent")

    magnitudes = {}
    raw_magnitudes = record.get("magnitudes", {})
    if not isinstance(raw_magnitudes, Mapping):
        raise ValueError(dataset_id + " magnitudes must be a mapping")
    unknown = set(raw_magnitudes) - set(source_ids)
    if unknown:
        raise ValueError(dataset_id + " magnitudes name sources outside the scene")
    for source_id, entry in raw_magnitudes.items():
        if isinstance(entry, Mapping):
            unknown = set(entry) - {"value", "uncertainty", "unavailable_reason"}
            if unknown:
                raise ValueError(dataset_id + " magnitude entry has unknown fields")
            value = entry.get("value")
            uncertainty = entry.get("uncertainty")
            reason = entry.get("unavailable_reason")
        else:
            value, uncertainty, reason = entry, None, None
        if value is None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(dataset_id + " unknown magnitude must state why")
            magnitudes[source_id] = {
                "value": None, "uncertainty": None, "unavailable_reason": reason,
            }
            continue
        if reason is not None:
            raise ValueError(dataset_id + " magnitude cannot be both measured and unavailable")
        magnitudes[source_id] = {
            "value": _finite_number(value, dataset_id + " magnitude"),
            "uncertainty": (None if uncertainty is None else _finite_number(
                uncertainty, dataset_id + " magnitude uncertainty")),
            "unavailable_reason": None,
        }
        if magnitudes[source_id]["uncertainty"] is not None and magnitudes[source_id]["uncertainty"] < 0.0:
            raise ValueError(dataset_id + " magnitude uncertainty must be non-negative")
    return {
        "dataset_id": dataset_id,
        "band": _dataset_band(record.get("band")),
        "aperture": _dataset_aperture(record.get("aperture")),
        "shares": shares,
        "magnitudes": magnitudes,
        "cadence_ids": _dataset_cadence_ids(record.get("cadence_ids"), n_cadences),
    }


def validated_contrast_curve(separations, values):
    """Return a contrast curve as float arrays: separation [arcsec], limiting delta-mag [mag].

    The curve is read in its measured direction (``companion_imaging``), so it
    needs at least two finite points at strictly increasing positive separation.
    """
    from trex.populations.companion_imaging import ContrastCurve

    curve = ContrastCurve(separations, values)
    return curve.separations, curve.contrasts


class TargetScene:
    """Stellar population context for one TESS target.

    Attributes:
        tic_id: TIC identifier.
        ra_deg, dec_deg: Target coordinates [degrees].
        M_s_Msun: Target mass [Msun].
        R_s_Rsun: Target radius [Rsun].
        Teff_K: Target effective temperature [K].
        Tmag: TESS magnitude.
        plx_mas: Gaia parallax [mas].
        feh_dex: Target metallicity [dex], or ``None`` when unavailable.
            A supplied value requires a matching measured-source provenance.
        sectors: Observed TESS sectors.
        contrast_separations: [arcsec] from contrast curve.
        contrast_values: [delta_mag] from contrast curve.
        contrast_provenance: Optional record of a measured curve's source and
            native band. Supplying ``contrast_curve`` explicitly to
            ``target.from_tic``/``target.from_scene`` creates this record;
            binding a scene with pre-existing curve arrays does not invent
            provenance when none was supplied.
        resolved_neighbors: List of dicts with M_s, R_s, delta_mag, separation.
            A source whose magnitude contrast was not measured may supply
            ``delta_mag=None`` only with a nonempty
            ``delta_mag_unavailable_reason``; its aperture flux fraction
            remains the evidence for its host scenario. A residual-scene
            source may also supply ``aperture_flux_fractions_upper_95``, the
            source-specific marginal limit used for its host scenario, and
            ``aperture_flux_fraction_limitation``; ``aperture_flux_fractions``
            remains the coherent joint scene fraction.
        N_background: Number of background stars from TRILEGAL.
        trilegal_cache: Path to cached TRILEGAL CSV.
        background_sha256: Digest binding the retained background population.
            Optional: when omitted it is computed from ``trilegal_cache`` and
            recorded exactly as a supplied one; when supplied it is verified.
        background_area_sq_deg: Simulated field area [deg^2]; required (a missing
            value raises ``ValueError``).
        background_population_provenance: The TRILEGAL LSST-simulation fallback
            projection bound to this population's digest, or ``None`` for a
            TRILEGAL web result. Its limitation states the measured same-cone
            count deficit and that the effect on a particular FPP is not measured.
        background_selection_band: One dataset band whose own native magnitude
            and target depth define the shared background faint limit. Required
            when several dataset bands exist and no TESS cut applies; omitted
            for a single-band or legacy TESS scene.
        background_completeness: Verbatim statement of the retained population's
            completeness limit (measurement or the fact it is not modeled).
            ``None`` means not stated, never a completeness of one.
    """

    def __init__(
        self,
        tic_id: Optional[int],
        ra_deg: float,
        dec_deg: float,
        M_s_Msun: float,
        R_s_Rsun: float,
        Teff_K: float,
        Tmag: Optional[float],
        plx_mas: Optional[float],
        sectors: List[int],
        contrast_separations: Optional[np.ndarray],
        contrast_values: Optional[np.ndarray],
        resolved_neighbors: List[Dict[str, Any]],
        target_aperture_flux_fractions: List[Dict[str, Any]],
        N_background: int,
        trilegal_cache: str | Path,
        background_sha256: Optional[str] = None,
        background_area_sq_deg: Optional[float] = None,
        feh_dex: Optional[float] = None,
        unresolved_resolution_limit_arcsec: Optional[float] = None,
        localization_evidence: Optional[Dict[str, Any]] = None,
        neighbor_parameter_evaluations: Optional[Mapping[str, Any]] = None,
        metallicity_provenance: Optional[Mapping[str, Any]] = None,
        contrast_provenance: Optional[Mapping[str, Any]] = None,
        background_population_provenance: Optional[Mapping[str, Any]] = None,
        datasets: Optional[List[Mapping[str, Any]]] = None,
        target_source_id: Optional[str] = None,
        target_identity: Optional[Mapping[str, Any]] = None,
        background_selection_band: Optional[str] = None,
        background_completeness: Optional[str] = None,
    ) -> None:
        self.target_identity = _target_identity(target_identity, tic_id)
        if tic_id is None:
            if self.target_identity is None:
                raise ValueError("a scene requires a TIC id or an exact target identity")
        elif isinstance(tic_id, bool) or not isinstance(tic_id, int) or tic_id <= 0:
            raise ValueError("tic_id must be a positive integer")
        if (self.target_identity is not None and self.target_identity["namespace"] == "tic"
                and str(self.target_identity["native_id"]) != str(tic_id)):
            raise ValueError("target identity does not match the scene TIC id")
        self.tic_id = tic_id
        self.ra_deg = _finite_number(ra_deg, "ra_deg")
        self.dec_deg = _finite_number(dec_deg, "dec_deg")
        if not 0.0 <= self.ra_deg < 360.0:
            raise ValueError("ra_deg must be in [0, 360)")
        if not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError("dec_deg must be in [-90, 90]")
        self.M_s_Msun = _finite_number(M_s_Msun, "M_s_Msun", positive=True)
        self.R_s_Rsun = _finite_number(R_s_Rsun, "R_s_Rsun", positive=True)
        self.Teff_K = _finite_number(Teff_K, "Teff_K", positive=True)
        # A non-TIC scene may genuinely lack a TESS magnitude or a Gaia
        # parallax. Missing stays missing: it is never replaced by a solar
        # estimate or 1/parallax, and the consumers that need it must defer.
        self.Tmag = None if Tmag is None else _finite_number(Tmag, "Tmag")
        self.plx_mas = None if plx_mas is None else _finite_number(plx_mas, "plx_mas", positive=True)
        self.feh_dex = None if feh_dex is None else _finite_number(feh_dex, "feh_dex")

        if not isinstance(sectors, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sectors
        ):
            raise ValueError("sectors must be a list of positive integers")
        self.sectors = list(sectors)

        self.target_aperture_flux_fractions = self._aperture_flux_fractions(
            ([] if target_aperture_flux_fractions is None and datasets is not None
             else target_aperture_flux_fractions),
            "target",
            allow_empty=datasets is not None,
        )
        if (contrast_separations is None) != (contrast_values is None):
            raise ValueError("contrast separations and values must be supplied together")
        measured_curve = contrast_separations is not None or contrast_values is not None
        if measured_curve:
            self.contrast_separations, self.contrast_values = validated_contrast_curve(
                contrast_separations, contrast_values)
            if unresolved_resolution_limit_arcsec is None:
                unresolved_resolution_limit_arcsec = GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC
            self.unresolved_resolution_limit_arcsec = _finite_number(
                unresolved_resolution_limit_arcsec,
                "unresolved_resolution_limit_arcsec",
                positive=True,
            )
        else:
            if unresolved_resolution_limit_arcsec is None:
                raise ValueError(
                    "a scene requires a measured contrast curve or the catalog resolution limit"
                )
            self.contrast_separations = None
            self.contrast_values = None
            self.unresolved_resolution_limit_arcsec = _finite_number(
                unresolved_resolution_limit_arcsec,
                "unresolved_resolution_limit_arcsec",
                positive=True,
            )

        # TREX computes companion contrasts in the TESS band only; a curve in
        # another filter needs a verified colour conversion, which TREX does not
        # implement. The record therefore also states which limit is binding.
        self.contrast_provenance = None
        if contrast_provenance is not None:
            if not measured_curve:
                raise ValueError("contrast provenance requires a measured contrast curve")
            if not isinstance(contrast_provenance, Mapping):
                raise ValueError("contrast provenance must be a mapping")
            record = dict(contrast_provenance)
            if (
                record.get("kind") != MEASURED_CONTRAST_CURVE_KIND
                or any(
                    not isinstance(record.get(key), str) or not record[key].strip()
                    for key in ("source", "band", "limitation")
                )
            ):
                raise ValueError(
                    "contrast provenance must state the curve kind, source, band and limitation"
                )
            _require_tess_band(record["band"])
            self.contrast_provenance = copy.deepcopy(record)
        if not isinstance(resolved_neighbors, list):
            raise ValueError("resolved_neighbors must be a list")
        self.resolved_neighbors = []
        source_ids = set()
        for index, neighbor in enumerate(resolved_neighbors):
            if not isinstance(neighbor, Mapping):
                raise ValueError("resolved neighbor {0} must be an object".format(index))
            source_id = neighbor.get("source_id")
            if not isinstance(source_id, str) or not source_id or source_id in source_ids:
                raise ValueError("resolved neighbors must have unique source_id values")
            source_ids.add(source_id)
            unavailable_reason = None
            if neighbor.get("M_s") is None and neighbor.get("R_s") is None:
                unavailable_reason = neighbor.get("stellar_parameter_unavailable_reason")
                if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
                    raise ValueError(
                        "resolved neighbor {0} without stellar parameters must state why".format(index)
                    )
                mass_value: Optional[float] = None
                radius_value: Optional[float] = None
            else:
                if neighbor.get("stellar_parameter_unavailable_reason") is not None:
                    raise ValueError("resolved stellar parameters cannot also be marked unavailable")
                mass_value = _finite_number(neighbor.get("M_s"), "neighbor M_s", positive=True)
                radius_value = _finite_number(neighbor.get("R_s"), "neighbor R_s", positive=True)
            provenance = neighbor.get("stellar_parameter_provenance")
            if provenance is not None:
                if not isinstance(provenance, Mapping) or not isinstance(provenance.get("source"), str):
                    raise ValueError(
                        "resolved neighbor {0} stellar-parameter provenance must name its source".format(index)
                    )
                provenance = copy.deepcopy(dict(provenance))
            calibration = {}
            if "stellar_relations_calibration_status" in neighbor:
                status = neighbor["stellar_relations_calibration_status"]
                if status != "uncalibrated":
                    raise ValueError("retained stellar relations must disclose their uncalibrated status")
                calibration["stellar_relations_calibration_status"] = status
            errors: Dict[str, float] = {}
            for name in (
                "mass_solar_err_lower", "mass_solar_err_upper",
                "radius_solar_err_lower", "radius_solar_err_upper",
            ):
                value = neighbor.get(name)
                if value is None:
                    continue
                number = _finite_number(value, "neighbor " + name)
                if number < 0.0:
                    raise ValueError("resolved neighbor {0} {1} must be non-negative".format(index, name))
                errors[name] = number
            delta_mag_value = neighbor.get("delta_mag")
            delta_mag_reason = neighbor.get("delta_mag_unavailable_reason")
            if delta_mag_value is None:
                if not isinstance(delta_mag_reason, str) or not delta_mag_reason.strip():
                    raise ValueError(
                        "resolved neighbor {0} without delta_mag must state why".format(index)
                    )
                delta_mag_value = None
            else:
                if delta_mag_reason is not None:
                    raise ValueError(
                        "resolved neighbor {0} carries both delta_mag and an unavailable reason".format(
                            index
                        )
                    )
                delta_mag_value = _finite_number(delta_mag_value, "neighbor delta_mag")
            upper_rows = neighbor.get("aperture_flux_fractions_upper_95")
            upper_fractions = None
            upper_limitation = None
            if upper_rows is not None:
                upper_fractions = self._aperture_flux_fractions(upper_rows, "neighbor upper")
                upper_limitation = neighbor.get("aperture_flux_fraction_limitation")
                if not isinstance(upper_limitation, str) or not upper_limitation.strip():
                    raise ValueError(
                        "resolved neighbor {0} marginal aperture limit must state its limitation".format(
                            index
                        )
                    )
            elif neighbor.get("aperture_flux_fraction_limitation") is not None:
                raise ValueError(
                    "resolved neighbor {0} states a marginal-limit limitation without the limit".format(
                        index
                    )
                )
            self.resolved_neighbors.append(
                {
                    "source_id": source_id,
                    "M_s": mass_value,
                    "R_s": radius_value,
                    "stellar_parameter_unavailable_reason": unavailable_reason,
                    "delta_mag": delta_mag_value,
                    **({"delta_mag_unavailable_reason": delta_mag_reason}
                       if delta_mag_reason is not None else {}),
                    "separation_arcsec": _finite_number(
                        neighbor.get("separation_arcsec"),
                        "neighbor separation_arcsec",
                        positive=True,
                    ),
                    "aperture_flux_fractions": self._aperture_flux_fractions(
                        ([] if neighbor.get("aperture_flux_fractions") is None and datasets is not None
                         else neighbor.get("aperture_flux_fractions")),
                        "neighbor",
                        allow_empty=datasets is not None,
                    ),
                    **({"aperture_flux_fractions_upper_95": upper_fractions,
                        "aperture_flux_fraction_limitation": upper_limitation}
                       if upper_fractions is not None else {}),
                    **errors,
                    **calibration,
                    **({"stellar_parameter_provenance": provenance} if provenance is not None else {}),
                }
            )
            if "metallicity_provenance" in neighbor and "metallicity_unavailable_reason" in neighbor:
                raise ValueError("neighbor metallicity cannot be both measured and unavailable")
            if "metallicity_provenance" in neighbor:
                composition = neighbor["metallicity_provenance"]
                value = composition.get("value_dex") if isinstance(composition, Mapping) else None
                self.resolved_neighbors[-1]["metallicity_provenance"] = _metallicity_record(
                    composition, value, source_id=source_id
                )
            if "metallicity_unavailable_reason" in neighbor:
                reason = neighbor["metallicity_unavailable_reason"]
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("neighbor metallicity unavailable reason must be nonempty")
                self.resolved_neighbors[-1]["metallicity_unavailable_reason"] = reason
            if neighbor.get("stellar_atmosphere") is not None:
                atmosphere = neighbor["stellar_atmosphere"]
                if not isinstance(atmosphere, Mapping) or not isinstance(atmosphere.get("source"), str) or not atmosphere["source"].strip():
                    raise ValueError("resolved atmosphere requires its source-associated provenance")
                values = {name: _finite_number(atmosphere.get(name), "neighbor " + name, positive=name == "teff_k")
                          for name in ("teff_k", "mh_dex", "av_mag")}
                if values["av_mag"] < 0:
                    raise ValueError("resolved atmosphere extinction must be nonnegative [mag]")
                self.resolved_neighbors[-1]["stellar_atmosphere"] = {**values, "source": atmosphere["source"]}
            if neighbor.get("stellar_parameter_distribution") is not None:
                from trex.evidence.host_measure import HostMeasure
                distribution = neighbor["stellar_parameter_distribution"]
                HostMeasure(distribution, self.feh_dex)
                self.resolved_neighbors[-1]["stellar_parameter_distribution"] = copy.deepcopy(distribution)

        if isinstance(N_background, bool) or not isinstance(N_background, int) or N_background < 0:
            raise ValueError("N_background must be a non-negative integer")
        self.N_background = N_background
        self.trilegal_cache = Path(trilegal_cache)
        if self.trilegal_cache.is_symlink() or not self.trilegal_cache.is_file():
            raise ValueError("trilegal_cache must be an available regular file")
        if background_sha256 is None:
            background_sha256 = _sha256(self.trilegal_cache)
        elif not isinstance(background_sha256, str) or len(background_sha256) != 64:
            raise ValueError("background_sha256 must be a SHA-256 digest")
        elif _sha256(self.trilegal_cache) != background_sha256:
            raise ValueError("trilegal_cache does not match background_sha256")
        self.background_sha256 = background_sha256
        self.background_area_sq_deg = _finite_number(
            background_area_sq_deg, "background_area_sq_deg", positive=True
        )
        # The TRILEGAL LSST-simulation fallback is stated with the population it
        # produced; a TRILEGAL web population carries nothing here, so a web
        # scene and its inference records stay unchanged.
        from trex.acquisition.background_population import fallback_provenance, retained_fallback_record

        if background_population_provenance is not None:
            # A portable scene supplies the projection, so the digest it carries
            # must still name the file the scene was built from.
            provenance = fallback_provenance(background_population_provenance)
            if provenance is None:
                raise ValueError(
                    "background_population_provenance must name the TRILEGAL LSST-simulation fallback"
                )
            if provenance.get("population_sha256") != self.background_sha256:
                raise ValueError(
                    "background_population_provenance does not name the retained background population"
                )
            self.background_population_provenance = provenance
        else:
            record = retained_fallback_record(self.trilegal_cache, self.background_sha256)
            self.background_population_provenance = None if record is None else fallback_provenance(record)
        self.metallicity_provenance = None
        if self.feh_dex is not None and metallicity_provenance is None:
            raise ValueError("target metallicity requires measured-source provenance")
        if metallicity_provenance is not None:
            if self.feh_dex is None:
                raise ValueError("metallicity provenance requires the target metallicity value")
            self.metallicity_provenance = _metallicity_record(metallicity_provenance, self.feh_dex)
        # Retained centroid diagnostic (legacy ``trex-spatial-prior-v1``).
        # This record is not a source-assignment likelihood; centroid-derived
        # neighbour weights remain neutral even when the diagnostic is present.
        self.localization_evidence: Optional[Dict[str, Any]] = (
            None if localization_evidence is None else copy.deepcopy(dict(localization_evidence))
        )
        # Portable diagnostics explain why a neighbour remains missing, never replace its values.
        self.neighbor_parameter_evaluations: Dict[str, Any] = (
            {} if neighbor_parameter_evaluations is None
            else copy.deepcopy(dict(neighbor_parameter_evaluations))
        )
        unknown = set(self.neighbor_parameter_evaluations) - {n["source_id"] for n in self.resolved_neighbors}
        if unknown:
            raise ValueError(
                "neighbour parameter evaluations name sources outside the scene: {0}".format(sorted(unknown))
            )
        for evaluations in self.neighbor_parameter_evaluations.values():
            if not isinstance(evaluations, (list, tuple)) or any(
                not isinstance(item, Mapping) or any(
                    not isinstance(item.get(key), str) or not item[key].strip()
                    for key in ("source", "status", "reason"))
                for item in evaluations
            ):
                raise ValueError("neighbour parameter evaluations require source, status and reason records")
        self.target_source_id = _target_source_id(
            target_source_id, self.target_identity, [n["source_id"] for n in self.resolved_neighbors]
        )
        self.datasets: Dict[str, Dict[str, Any]] = {}
        if datasets is not None:
            if not isinstance(datasets, list) or not datasets:
                raise ValueError("datasets must be a non-empty list")
            if self.target_source_id is None:
                raise ValueError("datasets require the target's exact target_source_id")
            source_ids = [self.target_source_id, *[n["source_id"] for n in self.resolved_neighbors]]
            resolved = [n["source_id"] for n in self.resolved_neighbors]
            for record in datasets:
                dataset = _validated_dataset(record, source_ids, self.target_source_id, resolved)
                if dataset["dataset_id"] in self.datasets:
                    raise ValueError("dataset_id values must be unique")
                self.datasets[dataset["dataset_id"]] = dataset
        # One declared faint-limit band for the whole scene (D-79). None keeps
        # the legacy TESS cut when there are no datasets, and the single
        # dataset band when only one exists; several bands then refuse.
        if background_selection_band is not None:
            if (not isinstance(background_selection_band, str)
                    or not background_selection_band.strip()):
                raise ValueError("background_selection_band must be nonempty text or None")
            if background_selection_band not in {
                    dataset["band"]["id"] for dataset in self.datasets.values()}:
                raise ValueError(
                    "background_selection_band must name one of the scene's dataset bands")
        self.background_selection_band = background_selection_band
        if background_completeness is not None and (
                not isinstance(background_completeness, str) or not background_completeness.strip()):
            raise ValueError("background_completeness must be nonempty text or None")
        self.background_completeness = background_completeness

    def background_population(self) -> Dict[str, Any]:
        """Return the retained simulated background stars for the declared selection.

        The population cache is re-verified against its recorded digest before
        it is parsed, so a background scenario can never be evaluated against a
        file that changed after the scene was built. The selection band is the
        scene's ``background_selection_band`` or, when only one dataset band
        exists, that band; several undeclared bands refuse instead of following
        dataset order. Its target magnitude from that dataset sets the faint
        limit and the loader retains every native row (no TESS substitution).
        ``selection_band``, ``selection_depth_mag`` and
        ``selection_completeness`` are attached for the evidence gate and do
        not alter the arrays. ``background_completeness`` is carried verbatim;
        None means not stated, never one. The legacy no-dataset TESS path still
        requires Tmag and keeps its loader-applied cut unchanged.
        """
        from trex.populations.background import is_legacy_tess_band, load_trilegal_population

        self.verify_background()
        dataset_bands = list(dict.fromkeys(
            dataset["band"]["id"] for dataset in self.datasets.values()))
        band = self.background_selection_band
        if band is None:
            if len(dataset_bands) > 1:
                raise ValueError(
                    "background selection band is ambiguous across the scene datasets "
                    "({0}); declare background_selection_band".format(
                        ", ".join(repr(value) for value in dataset_bands))
                )
            band = dataset_bands[0] if dataset_bands else "TESS"
        if dataset_bands and band not in dataset_bands:
            raise ValueError("background selection band is not one of the scene datasets")
        legacy = is_legacy_tess_band(band)
        depth = None
        if dataset_bands:
            magnitudes = [
                dataset["magnitudes"].get(self.target_source_id)
                for dataset in self.datasets.values()
                if dataset["band"]["id"] == band
            ]
            if (not magnitudes or any(entry is None or entry["value"] is None
                                      for entry in magnitudes)):
                raise ValueError(
                    "background selection band {0!r} has no measured target magnitude; "
                    "a TESS or other-band value is not substituted".format(band))
            depth = float(magnitudes[0]["value"])
            if any(float(entry["value"]) != depth for entry in magnitudes[1:]):
                raise ValueError(
                    "datasets sharing the background selection band disagree on the target magnitude")
        if legacy:
            if self.Tmag is not None and depth is not None and float(self.Tmag) != depth:
                raise ValueError(
                    "the scene's TESS magnitude differs from the dataset target magnitude")
            depth = self.Tmag if depth is None else depth
            if depth is None:
                raise ValueError(
                    "the retained TRILEGAL selection is TESS-specific and requires the target TESS magnitude; "
                    "a non-TESS scene must not substitute one"
                )
        requested = {
            dataset["band"]["id"]: dataset["band"].get(
                "population_column", dataset["band"]["id"])
            for dataset in self.datasets.values()
            if not is_legacy_tess_band(dataset["band"]["id"])
        }
        population = load_trilegal_population(
            self.trilegal_cache, depth if legacy else None,
            extra_bands=requested or None,
        )
        population["selection_band"] = band
        population["selection_depth_mag"] = depth
        population["selection_completeness"] = self.background_completeness
        return population

    def to_dict(self) -> Dict[str, Any]:
        """Export explicit constructor inputs for a portable, revalidated transfer.

        No files are copied, resolved neighbors are not truncated, and the
        background keeps its original SHA-256. Paths are explicit caller-owned
        paths; transferring an observation to another host requires separate
        authorization. This record is conditional input, not calibration proof.
        """
        def fractions(rows):
            return [{"sector": sector, "fraction": fraction} for sector, fraction in rows.items()]

        def dataset_records():
            records = []
            for dataset in self.datasets.values():
                shares = []
                for source_id, row in dataset["shares"].items():
                    fraction = row["fraction"]
                    shares.append({
                        "source_id": source_id,
                        "fraction": list(fraction) if isinstance(fraction, tuple) else fraction,
                        "in_aperture": row["in_aperture"],
                    })
                records.append({
                    "dataset_id": dataset["dataset_id"],
                    "band": copy.deepcopy(dataset["band"]),
                    "aperture": copy.deepcopy(dataset["aperture"]),
                    "shares": shares,
                    "magnitudes": copy.deepcopy(dataset["magnitudes"]),
                    **({"cadence_ids": list(dataset["cadence_ids"])}
                       if dataset["cadence_ids"] is not None else {}),
                })
            return records

        return {
            "tic_id": self.tic_id, "ra_deg": self.ra_deg, "dec_deg": self.dec_deg,
            "M_s_Msun": self.M_s_Msun, "R_s_Rsun": self.R_s_Rsun,
            "Teff_K": self.Teff_K, "Tmag": self.Tmag, "plx_mas": self.plx_mas,
            "feh_dex": self.feh_dex, "sectors": list(self.sectors),
            **({"target_identity": copy.deepcopy(self.target_identity)}
               if self.target_identity is not None else {}),
            **({"target_source_id": self.target_source_id}
               if self.target_source_id is not None else {}),
            **({"datasets": dataset_records()} if self.datasets else {}),
            **({"metallicity_provenance": copy.deepcopy(self.metallicity_provenance)}
               if self.metallicity_provenance is not None else {}),
            "contrast_separations": None if self.contrast_separations is None else self.contrast_separations.tolist(),
            "contrast_values": None if self.contrast_values is None else self.contrast_values.tolist(),
            "contrast_provenance": copy.deepcopy(self.contrast_provenance),
            "resolved_neighbors": [
                {**copy.deepcopy(neighbor),
                 "aperture_flux_fractions": fractions(neighbor["aperture_flux_fractions"]),
                 **({"aperture_flux_fractions_upper_95": fractions(
                     neighbor["aperture_flux_fractions_upper_95"])}
                    if "aperture_flux_fractions_upper_95" in neighbor else {})}
                for neighbor in self.resolved_neighbors
            ],
            "target_aperture_flux_fractions": fractions(self.target_aperture_flux_fractions),
            "N_background": self.N_background, "trilegal_cache": str(self.trilegal_cache),
            "background_sha256": self.background_sha256,
            "background_area_sq_deg": self.background_area_sq_deg,
            **({"background_selection_band": self.background_selection_band}
               if self.background_selection_band is not None else {}),
            **({"background_completeness": self.background_completeness}
               if self.background_completeness is not None else {}),
            **({"background_population_provenance": copy.deepcopy(self.background_population_provenance)}
               if self.background_population_provenance is not None else {}),
            "unresolved_resolution_limit_arcsec": self.unresolved_resolution_limit_arcsec,
            "localization_evidence": copy.deepcopy(self.localization_evidence),
            "neighbor_parameter_evaluations": copy.deepcopy(self.neighbor_parameter_evaluations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetScene":
        """Revalidate every supplied field and the retained background on import."""
        if not isinstance(payload, Mapping):
            raise ValueError("portable scene input must be a mapping")
        return cls(**payload)

    @staticmethod
    def _aperture_flux_fractions(value: object, label: str, *, allow_empty: bool = False) -> Dict[int, float]:
        if not isinstance(value, list) or (not value and not allow_empty):
            raise ValueError("{0} aperture flux fractions must be a non-empty list".format(label))
        fractions: Dict[int, float] = {}
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("{0} aperture flux fraction is malformed".format(label))
            sector = row.get("sector")
            if isinstance(sector, bool) or not isinstance(sector, int) or sector <= 0 or sector in fractions:
                raise ValueError("{0} aperture flux fraction sector is invalid".format(label))
            fraction = _finite_number(row.get("fraction"), "{0} aperture flux fraction".format(label))
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("{0} aperture flux fraction is outside [0, 1]".format(label))
            fractions[sector] = fraction
        return fractions

    @property
    def n_neighbors(self) -> int:
        return len(self.resolved_neighbors)

    @property
    def has_contrast_data(self) -> bool:
        return self.contrast_separations is not None

    def verify_background(self) -> None:
        """Reject a background population that changed after scene construction."""
        if self.trilegal_cache.is_symlink() or not self.trilegal_cache.is_file():
            raise ValueError("trilegal_cache is no longer an available regular file")
        if _sha256(self.trilegal_cache) != self.background_sha256:
            raise ValueError("trilegal_cache changed after scene construction")

    def neighbor_masses_radii(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (masses, radii, delta_mags) for resolved neighbors.

        All neighbour properties were explicitly supplied in the scene manifest.
        """
        n = len(self.resolved_neighbors)
        masses = np.full(n, np.nan)
        radii = np.full(n, np.nan)
        delta_mags = np.full(n, np.nan)

        for i, nb in enumerate(self.resolved_neighbors):
            delta = nb["delta_mag"]
            delta_mags[i] = np.nan if delta is None else float(delta)
            masses[i] = np.nan if nb["M_s"] is None else float(nb["M_s"])
            radii[i] = np.nan if nb["R_s"] is None else float(nb["R_s"])

        return masses, radii, delta_mags

    def companion_delta_mags(self) -> np.ndarray:
        """Delta magnitudes of all resolved neighbours, NaN where unmeasured."""
        if not self.resolved_neighbors:
            return np.array([])
        return np.array(
            [np.nan if nb["delta_mag"] is None else float(nb["delta_mag"])
             for nb in self.resolved_neighbors],
            dtype=float,
        )

    @staticmethod
    def _reduce_aperture_flux_fraction(
        fractions: Dict[int, float], sector_weights: Mapping[int, float], label: str
    ) -> float:
        """Reduce per-sector aperture fractions onto the observed photometry.

        The phase-folded curve handed to the evidence calculation is an
        inverse-variance combination across sectors, so the dilution that curve
        actually carries is the same inverse-variance-weighted mean of the
        per-sector aperture flux fractions.  ``sector_weights`` must therefore
        be the per-sector inverse-variance weights of the retained cadences.
        """
        if not isinstance(sector_weights, Mapping) or not sector_weights:
            raise ValueError("observed sector weights are required to reduce aperture flux fractions")
        weighted = 0.0
        total = 0.0
        for sector, weight in sector_weights.items():
            if isinstance(sector, (bool, np.bool_)) or not isinstance(sector, (int, np.integer)):
                raise ValueError("observed sector IDs must be integers")
            sector_index = int(sector)
            weight_value = _finite_number(weight, "observed sector weight", positive=True)
            if not math.isfinite(weight_value) or weight_value <= 0.0:
                raise ValueError("observed sector weight for sector {0} is invalid".format(sector_index))
            if sector_index not in fractions:
                raise ValueError(
                    "{0} aperture scene does not cover observed sector {1}".format(label, sector_index)
                )
            weighted += weight_value * fractions[sector_index]
            total += weight_value
        if total <= 0.0:
            raise ValueError("observed sector weights sum to zero")
        return weighted / total

    def target_aperture_flux_fraction(self, sector_weights: Mapping[int, float]) -> float:
        """Target share of in-aperture flux over the observed sectors."""
        if not self.target_aperture_flux_fractions:
            raise ValueError(
                "scene has no TESS sector shares; use dataset_target_flux_fraction for this dataset"
            )
        return self._reduce_aperture_flux_fraction(
            self.target_aperture_flux_fractions, sector_weights, "target"
        )

    def neighbor_dicts_for_evidence(
        self, sector_weights: Optional[Mapping[int, float]] = None
    ) -> List[Dict[str, Any]]:
        """Return per-neighbor host properties for the evidence calculation.

        Every resolved neighbour is returned, including one without stellar
        properties: dropping it would silently remove its scenarios. The
        evidence calculation decides whether such a neighbour can host the
        observed depth and fails closed if it can. When ``sector_weights`` is
        supplied each entry also carries that neighbor's aperture share over
        the observed sectors. For a residual-scene source with a marginal
        limit, ``aperture_flux_fraction`` is that source-specific marginal 95%
        fraction (the joint scenario is not an upper bound for one source) and
        the coherent joint value is retained as
        ``aperture_flux_fraction_joint_95``; otherwise the two are equal. Every
        entry also carries a neutral ``log_prior``: the retained difference-image
        centroid diagnostic does not implement a calibrated scenario likelihood
        and must not reweight or delete a neighbour. Measured aperture dilution
        remains independently applied.
        """
        from trex.populations.spatial_prior import neighbor_log_priors

        log_priors = neighbor_log_priors(
            [neighbor["separation_arcsec"] for neighbor in self.resolved_neighbors],
            self.localization_evidence,
        )
        entries: List[Dict[str, Any]] = []
        for index, neighbor in enumerate(self.resolved_neighbors):
            entry: Dict[str, Any] = {
                "source_id": neighbor["source_id"],
                "M_s": neighbor["M_s"],
                "R_s": neighbor["R_s"],
                "log_prior": log_priors[index],
                **({key: copy.deepcopy(neighbor[key]) for key in (
                    "mass_solar_err_lower", "mass_solar_err_upper",
                    "radius_solar_err_lower", "radius_solar_err_upper",
                    "stellar_parameter_provenance",
                    "stellar_parameter_distribution",
                    "stellar_relations_calibration_status",
                    "metallicity_provenance", "metallicity_unavailable_reason",
                    "delta_mag_unavailable_reason",
                ) if key in neighbor}),
                **({"stellar_atmosphere": copy.deepcopy(neighbor["stellar_atmosphere"])}
                   if "stellar_atmosphere" in neighbor else {}),
            }
            if neighbor.get("stellar_parameter_unavailable_reason"):
                reason = neighbor["stellar_parameter_unavailable_reason"]
                evaluations = self.neighbor_parameter_evaluations.get(neighbor["source_id"])
                if evaluations:
                    reason += "; tried: " + "; ".join(
                        "{0}: {1} ({2})".format(item["source"], item["status"], item["reason"])
                        for item in evaluations
                    )
                entry["stellar_parameter_unavailable_reason"] = reason
            if sector_weights is not None:
                joint_fraction = self._reduce_aperture_flux_fraction(
                    neighbor["aperture_flux_fractions"], sector_weights, "neighbor"
                )
                entry["aperture_flux_fraction"] = joint_fraction
                entry["aperture_flux_fraction_joint_95"] = joint_fraction
                if "aperture_flux_fractions_upper_95" in neighbor:
                    upper_fraction = self._reduce_aperture_flux_fraction(
                        neighbor["aperture_flux_fractions_upper_95"],
                        sector_weights,
                        "neighbor upper",
                    )
                    entry["aperture_flux_fraction"] = upper_fraction
                    entry["aperture_flux_fraction_upper_95"] = upper_fraction
                    entry["aperture_flux_fraction_limitation"] = neighbor[
                        "aperture_flux_fraction_limitation"
                    ]
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Dataset-indexed observations (MM-S0)
    # ------------------------------------------------------------------

    @property
    def dataset_ids(self) -> tuple:
        """Identifiers of the declared observation datasets, in declaration order."""
        return tuple(self.datasets)

    def _dataset(self, dataset_id: str) -> Dict[str, Any]:
        if dataset_id not in self.datasets:
            raise ValueError("scene has no dataset {0!r}".format(dataset_id))
        return self.datasets[dataset_id]

    def dataset(self, dataset_id: str) -> Dict[str, Any]:
        """A copy of one validated dataset record."""
        return copy.deepcopy(self._dataset(dataset_id))

    def dataset_flux_fraction(self, dataset_id: str, source_id: str, cadence: Optional[int] = None):
        """One source's share of that dataset's own aperture total.

        A scalar share is returned for every cadence. A per-cadence share
        returns the full float array when ``cadence`` is omitted and one value
        when it is supplied. An exact zero is a real flat prediction for that
        dataset; it does not remove the source or its host scenario.
        """
        dataset = self._dataset(dataset_id)
        row = dataset["shares"].get(source_id)
        if row is None:
            raise ValueError("dataset {0!r} does not state source {1!r}".format(dataset_id, source_id))
        fraction = row["fraction"]
        if not isinstance(fraction, tuple):
            return float(fraction)
        if cadence is None:
            return np.asarray(fraction, dtype=float)
        if isinstance(cadence, (bool, np.bool_)) or not isinstance(cadence, (int, np.integer)):
            raise ValueError("cadence index must be an integer")
        index = int(cadence)
        if not 0 <= index < len(fraction):
            raise ValueError("cadence index is outside the dataset")
        return float(fraction[index])

    def dataset_target_flux_fraction(self, dataset_id: str, cadence: Optional[int] = None):
        """The target's share of one dataset's aperture total."""
        if self.target_source_id is None:
            raise ValueError("scene does not declare the target's exact source identity")
        return self.dataset_flux_fraction(dataset_id, self.target_source_id, cadence)

    def dataset_magnitude_status(self, dataset_id: str, source_id: str) -> Dict[str, Any]:
        """Whether one dataset reports a source's native-band magnitude.

        ``not-reported`` and ``unavailable`` are both unknown light. Neither is
        zero and neither authorizes a solar/TESS substitute; a consumer that
        needs the band must defer or refuse the dataset.
        """
        dataset = self._dataset(dataset_id)
        entry = dataset["magnitudes"].get(source_id)
        record = {
            "dataset_id": dataset_id,
            "source_id": source_id,
            "band": copy.deepcopy(dataset["band"]),
        }
        if entry is None:
            return {**record, "status": "not-reported", "value": None, "uncertainty": None}
        if entry["value"] is None:
            return {
                **record, "status": "unavailable", "value": None,
                "uncertainty": None, "reason": entry["unavailable_reason"],
            }
        return {
            **record, "status": "measured", "value": entry["value"],
            "uncertainty": entry["uncertainty"],
        }

    def dataset_magnitude(self, dataset_id: str, source_id: str) -> Tuple[float, Optional[float]]:
        """Measured native-band magnitude and uncertainty, or an explicit refusal."""
        status = self.dataset_magnitude_status(dataset_id, source_id)
        if status["status"] == "unavailable":
            raise ValueError(
                "dataset {0!r} magnitude for {1!r} is unavailable: {2}".format(
                    dataset_id, source_id, status["reason"]
                )
            )
        if status["status"] != "measured":
            raise ValueError(
                "dataset {0!r} does not report a magnitude for {1!r}; the band light is unknown".format(
                    dataset_id, source_id
                )
            )
        return float(status["value"]), status["uncertainty"]

    def neighbor_dicts_for_evidence_by_dataset(
        self, dataset_id: str, cadence: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Per-neighbour evidence entries carrying this dataset's aperture share.

        Every resolved neighbour is returned, including one with an exact zero
        share: a zero-share dataset predicts a flat curve for that host, while
        the host remains a scenario. The share is the source's fraction of this
        dataset's own aperture total, never renormalized against another
        dataset. Per-cadence datasets require the cadence index.
        """
        dataset = self._dataset(dataset_id)
        if dataset["cadence_ids"] is not None and cadence is None:
            raise ValueError("per-cadence dataset requires a cadence index")
        entries = self.neighbor_dicts_for_evidence()
        for entry in entries:
            fraction = self.dataset_flux_fraction(dataset_id, entry["source_id"], cadence)
            if isinstance(fraction, np.ndarray):
                raise ValueError("per-cadence dataset requires a cadence index")
            entry["dataset_id"] = dataset_id
            entry["band"] = copy.deepcopy(dataset["band"])
            entry["aperture_id"] = dataset["aperture"]["id"]
            entry["aperture_denominator"] = dataset["aperture"]["denominator"]
            entry["aperture_flux_fraction"] = float(fraction)
            entry["aperture_flux_fraction_in_aperture"] = bool(
                dataset["shares"][entry["source_id"]]["in_aperture"]
            )
            magnitude = self.dataset_magnitude_status(dataset_id, entry["source_id"])
            if magnitude["status"] == "measured":
                entry["band_magnitude"] = magnitude["value"]
                entry["band_magnitude_uncertainty"] = magnitude["uncertainty"]
        return entries

    def population_datasets(self) -> List[Dict[str, Any]]:
        """Dataset records for a joint population likelihood.

        Each entry binds one dataset's native band and aperture denominator to
        the same source identities. The target and every resolved neighbour are
        present with their own share; per-cadence shares stay arrays so a
        changing aperture is not collapsed into one sector-weighted scalar.
        """
        records = []
        for dataset in self.datasets.values():
            source_fractions = {
                source_id: (list(row["fraction"]) if isinstance(row["fraction"], tuple) else row["fraction"])
                for source_id, row in dataset["shares"].items()
            }
            records.append({
                "dataset_id": dataset["dataset_id"],
                "band": copy.deepcopy(dataset["band"]),
                "aperture": copy.deepcopy(dataset["aperture"]),
                "target_source_id": self.target_source_id,
                "target_fraction": source_fractions.get(self.target_source_id),
                "source_fractions": source_fractions,
                "in_aperture": {
                    source_id: row["in_aperture"] for source_id, row in dataset["shares"].items()
                },
                "magnitudes": copy.deepcopy(dataset["magnitudes"]),
                "cadence_ids": (None if dataset["cadence_ids"] is None
                                else list(dataset["cadence_ids"])),
            })
        return records

    def population_dataset_bands(self) -> List[Dict[str, Any]]:
        """Dataset bindings for one physical population draw (MM-D1 handoff).

        Every dataset is returned; ``target_magnitude`` is ``None`` when the
        target's native-band magnitude is not measured. A background-star
        population then refuses that binding instead of substituting TESS,
        while a bound companion needs only its per-band light. The band id is
        the same one ``background_population`` requested from the retained
        cache, so the population's own magnitude arrays align by construction.
        """
        records = []
        for dataset in self.datasets.values():
            magnitude = dataset["magnitudes"].get(self.target_source_id)
            records.append({
                "dataset_id": dataset["dataset_id"],
                "band": dataset["band"]["id"],
                "target_magnitude": (None if magnitude is None else magnitude["value"]),
            })
        return records


__all__ = ["TargetScene"]


_CONTRAST_TOKEN_SPLIT = re.compile(r"[,\t;]+|\s+")


def _parse_contrast_curve_text(text):
    """Parse a two-column separation [arcsec], delta-magnitude [mag] table.

    Named headers and comment lines are skipped. Numeric rows require exactly
    two finite columns; prose containing numbers is never converted into data.
    File order is retained so the scene can refuse an unsorted curve.
    """
    separations, magnitudes = [], []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        stripped = stripped.split("#", 1)[0].split("//", 1)[0].strip()
        tokens = [token for token in _CONTRAST_TOKEN_SPLIT.split(stripped) if token]
        try:
            float(tokens[0])
        except (ValueError, IndexError):
            continue
        if len(tokens) != 2:
            raise ValueError(f"contrast curve row is misaligned: {stripped!r}")
        try:
            numbers = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("contrast curve rows must contain two finite numbers") from exc
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("contrast curve rows must contain two finite numbers")
        separations.append(numbers[0])
        magnitudes.append(numbers[1])
    if not separations:
        raise ValueError("contrast curve contains no aligned two-column rows")
    return np.asarray(separations, dtype=float), np.asarray(magnitudes, dtype=float)


def _contrast_curve_arrays(contrast_curve):
    """Return (separations [arcsec], delta magnitudes [mag], source) for a curve."""
    if isinstance(contrast_curve, (str, os.PathLike)):
        path = Path(contrast_curve)
        if path.is_symlink() or not path.is_file():
            raise ValueError("contrast_curve must be an existing regular file")
        return (*_parse_contrast_curve_text(path.read_text(encoding="utf-8")), str(path))
    if isinstance(contrast_curve, np.ndarray):
        if contrast_curve.ndim != 2 or contrast_curve.shape[0] != 2:
            raise ValueError("contrast_curve array must have shape (2, N)")
        separations, magnitudes = contrast_curve
    elif isinstance(contrast_curve, (tuple, list)) and len(contrast_curve) == 2:
        separations, magnitudes = contrast_curve
    else:
        raise ValueError(
            "contrast_curve must be a two-column file or a (separations, delta_magnitudes) pair"
        )
    return (np.asarray(separations, dtype=float), np.asarray(magnitudes, dtype=float),
            "caller-supplied arrays")



def measured_contrast_curve(contrast_curve, contrast_band, *, source=None):
    """Parse and validate a measured curve; return its JSON record (arcsec, delta-mag, band, source).

    ``source`` overrides the recorded origin (a retained request names the file
    it was read from). The band must be stated; the scene refuses non-TESS bands.
    """
    if not isinstance(contrast_band, str) or not contrast_band.strip():
        raise ValueError(
            "contrast_band is required with contrast_curve and must name the band "
            "the curve was measured in"
        )
    _require_tess_band(contrast_band)
    separations, magnitudes, origin = _contrast_curve_arrays(contrast_curve)
    separations, magnitudes = validated_contrast_curve(separations, magnitudes)
    return {"separations_arcsec": separations.tolist(), "delta_magnitudes": magnitudes.tolist(),
            "band": contrast_band.strip().upper(), "source": origin if source is None else str(source)}

def with_measured_contrast_curve(scene, contrast_curve, contrast_band, *, source=None):
    """Rebuild ``scene`` with a measured curve and the provenance that binds it.

    ``TargetScene.from_dict`` runs the existing scene checks on the curve, so
    curve validation is not duplicated here.
    """
    if contrast_curve is None:
        if contrast_band is not None:
            raise ValueError("contrast_band requires contrast_curve")
        return scene
    record = measured_contrast_curve(contrast_curve, contrast_band, source=source)
    separations = np.asarray(record["separations_arcsec"])
    magnitudes = np.asarray(record["delta_magnitudes"])
    band, source = record["band"], record["source"]
    payload = scene.to_dict()
    payload.update(
        contrast_separations=separations.tolist(),
        contrast_values=magnitudes.tolist(),
        contrast_provenance={
            "kind": MEASURED_CONTRAST_CURVE_KIND,
            "source": source,
            "band": band,
            "limitation": (
                "Unresolved companions are bounded by the measured {0}-band contrast curve "
                "from {1} where that curve covers their separation; the Gaia DR2 close-pair "
                "limit (Giacalone et al. 2021, AJ 161, 24, Sect. 1) remains the independent "
                "unresolved-scene bound outside the curve's measured range."
            ).format(band, source),
        },
    )
    return TargetScene.from_dict(payload)
