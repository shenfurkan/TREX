"""Retained centroid diagnostics, not a resolved-neighbour scenario likelihood.

The legacy ``trex-spatial-prior-v1`` record keeps candidate-owned offsets and
uncertainties in arcseconds, and dimensionless source-competition ratios.
Bryson et al. (2013), PASP 125, 889, DOI ``10.1086/671767``,
Sec. 3.4.1 Eq. 13 describes offset significance, not a Bayesian spatial prior.

A null-tail probability is not a source-assignment likelihood. In particular,
``|separation - offset_distance|`` discards the direction of both vectors and
the full covariance, and a significance threshold supplies no principled
likelihood floor. No calibrated source-location likelihood is implemented by
this contract. Its former Rayleigh-tail reweighting is therefore disabled:
every neighbour receives an exactly neutral log weight, including when payload
flags assert calibration. Retained records remain available for review. A future
likelihood needs vector geometry, noise/selection model and calibration evidence; an
effective sigma reverse-engineered from a p-value does not provide these.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Caller-owned localization output: schema_version=1.0/work_package=PRF_SOURCE_LOCALIZATION.
# The derived trex-spatial-prior-v1 record is a different contract, not an input schema.
LOCALIZATION_RESULT_RELATIVE_PATH = Path("outputs") / "prf_localization_results.json"
SPATIAL_PRIOR_CONTRACT = "trex-spatial-prior-v1"
SPATIAL_PRIOR_REFERENCE = (
    "Bryson et al. (2013), PASP 125, 889, Sec. 3.4.1 Eq. 13 "
    "(DOI 10.1086/671767)"
)

def _finite_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _unavailable(reason: str) -> Dict[str, Any]:
    """Adapter/scene consumers read status/reason; no numerical replacement."""
    return {"status": "unavailable", "reason": reason}


def spatial_evidence_from_localization_payload(
    payload: object, candidate_id: str
) -> Dict[str, Any]:
    """Return the retained difference-image evidence, or a stated unavailability.

    The payload is the parsed ``prf_localization_results.json`` object; it is
    validated here (candidate identity, characterized summary, characterized
    multi-transit offset) rather than trusted, so a stub, a foreign candidate's
    result, or an unresolved offset resolves to an explicit ``unavailable``
    record instead of an inferred number.
    """
    if not isinstance(payload, dict):
        return _unavailable("localization result is not a JSON object")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return _unavailable("localization requires a nonempty candidate identity")
    if payload.get("schema_version") != "1.0" or payload.get("work_package") != "PRF_SOURCE_LOCALIZATION":
        return _unavailable("localization result has an unsupported input schema")
    if payload.get("candidate_id") != candidate_id:
        return _unavailable("localization result belongs to a different candidate")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip() or source.strip().startswith("not-run"):
        return _unavailable(
            "no characterized localization run is retained "
            "(source {0!r})".format(source)
        )
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return _unavailable("localization result lacks a summary")
    ratio = _finite_float(summary.get("median_target_to_other_difference_ratio"))
    if ratio is None or ratio < 0:
        return _unavailable(
            "localization result has no finite nonnegative target-to-strongest-other "
            "difference-image amplitude ratio"
        )
    per_transit = payload.get("per_transit_centroid_offsets")
    combination = per_transit.get("bryson_2013_multi_transit_average") if isinstance(per_transit, dict) else None
    if not isinstance(combination, dict):
        return _unavailable("localization has no multi-transit difference-image offset combination")
    if combination.get("decision_state") != "characterized":
        return _unavailable(
            "localization has no characterized multi-transit difference-image "
            "offset combination"
        )
    offset = _finite_float(combination.get("offset_distance_arcsec"))
    sigma = _finite_float(combination.get("offset_distance_sigma_arcsec"))
    if offset is None or offset < 0 or sigma is None or sigma <= 0.0:
        return _unavailable(
            "localization requires a finite nonnegative offset and positive uncertainty"
        )
    significance = offset / sigma
    if not math.isfinite(significance):
        return _unavailable("localization offset significance is unrepresentable")
    return {
        "status": "available",
        "schema": SPATIAL_PRIOR_CONTRACT,
        "reference": SPATIAL_PRIOR_REFERENCE,
        # Keep asserted diagnostic flags for review; neither enables reweighting.
        "calibrated": bool(payload.get("calibrated") is True),
        "source_assignment_interpretable": bool(
            payload.get("source_assignment_interpretable") is True
        ),
        "target_to_max_other_difference_ratio": ratio,
        "difference_image_offset_arcsec": offset,
        "difference_image_offset_sigma_arcsec": sigma,
        "offset_significance_sigma": significance,
    }


def localization_spatial_evidence(workspace: Any) -> Dict[str, Any]:
    """Read caller-owned localization evidence without following outputs/file symlinks."""
    candidate_id = getattr(workspace, "candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return _unavailable("localization requires a nonempty candidate identity")
    root = Path(workspace.path)
    path = root / LOCALIZATION_RESULT_RELATIVE_PATH
    if root.is_symlink() or path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        return _unavailable(
            "localization result {0} is absent or not a regular file".format(
                LOCALIZATION_RESULT_RELATIVE_PATH.as_posix()
            )
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _unavailable("localization result is not readable JSON")
    return spatial_evidence_from_localization_payload(
        payload, candidate_id
    )


def neighbor_log_priors(
    separations_arcsec: Sequence[float], evidence: Optional[Mapping[str, Any]]
) -> List[float]:
    """Return neutral dimensionless log weights for every resolved neighbour.

    Separations are in arcseconds. ``evidence`` is retained for interface
    compatibility, but the legacy centroid-tail contract supplies no valid
    scenario likelihood, regardless of its flags. See the module's statistical
    boundary. This does not alter measured aperture dilution or drop neighbours.
    """
    separations = list(separations_arcsec)
    if any(_finite_float(value) is None or value < 0 for value in separations):
        raise ValueError("neighbor separations must be finite nonnegative real arcseconds")
    return [0.0] * len(separations)


__all__ = [
    "LOCALIZATION_RESULT_RELATIVE_PATH",
    "SPATIAL_PRIOR_CONTRACT",
    "SPATIAL_PRIOR_REFERENCE",
    "localization_spatial_evidence",
    "neighbor_log_priors",
    "spatial_evidence_from_localization_payload",
]
