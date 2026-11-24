"""Scenario-independent, native-time TREX noise calibration.

Pont, Zucker & Queloz (2006), MNRAS 373, 231, DOI
``10.1111/j.1365-2966.2006.11012.x``, Sects. 2.3 and 2.6,
motivate the transit-timescale binned-RMS estimator in :mod:`trex.models.red_noise`.
Winn et al. (2008), ApJ 683, 1076, DOI ``10.1086/589737``,
Sect. 3, use out-of-transit single-point scatter times beta >= 1 as an
approximate effective flux error.

Time/period/duration/cadence are days (native BTJD_TDB); flux and reported
sigma are normalized relative flux. Residuals are standardized by each
reported sigma, preserving heteroscedastic weights. Calibration is performed
separately within each retained sector, before folding destroys chronological
adjacency. Bins never bridge transit masks or gaps. The largest sector factor
is a conservative *policy envelope*, not a fitted hyperparameter or a GP.
It is fixed once for every planet, binary, and neighboring-host scenario.

Event-phase support is explicit. With no ``event_windows`` the calibration
excludes the primary and a circular secondary at phase 0.5 with the primary
T14; that half-phase assumption is recorded, not silently generalized to an
eccentric orbit. A caller that declares an event phase and duration (for
example an eccentric secondary from a retained fit) passes ``event_windows``
and those windows replace the circular assumption.

This diagonal beta approximation assumes stationary noise represented by the
out-of-transit data, and independent events separated beyond the correlation
time. It does not model a full covariance, marginalize estimator uncertainty,
or establish calibrated FPP coverage. Missing calibration defers, never uses
beta = 1 as a fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from trex.models.red_noise import (
    GAP_TOLERANCE_CADENCES, MIN_BIN_POINTS, PONT_ETAL_2006, WINN_ETAL_2008,
    out_of_transit_mask, pont_2006_beta, transit_bin_size_points,
)
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.flux_calibration import validate_flux_uncertainty


def _validated_event_windows(
    event_windows: Sequence[Mapping[str, Any]] | None,
    *,
    period_days: float,
    duration_days: float,
) -> list[dict[str, Any]]:
    """Validate caller-declared eclipse windows; None means the circular default.

    Each window is ``{"phase": fraction in [0, 1), "duration_days": days,
    "role": label}``. ``duration_days`` defaults to the primary T14. A window
    replaces the calibration's circular half-phase assumption; the caller owns
    its scientific provenance. An explicitly empty sequence is an invalid
    declaration; invalid values raise ValueError.
    """
    if event_windows is None:
        return []
    if isinstance(event_windows, (Mapping, str, bytes)):
        raise ValueError("event_windows must be a sequence of phase/duration mappings")
    windows: list[dict[str, Any]] = []
    for item in event_windows:
        if not isinstance(item, Mapping):
            raise ValueError("each event window must be a mapping with phase and duration_days")
        phase = item.get("phase")
        duration = item.get("duration_days", duration_days)
        role = item.get("role", "declared-event")
        if isinstance(phase, (bool, np.bool_)) or not isinstance(phase, Real) or not math.isfinite(phase):
            raise ValueError("event window phase must be a finite real fraction of the period")
        if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real) or not math.isfinite(duration):
            raise ValueError("event window duration must be finite real days")
        if not 0.0 <= float(phase) < 1.0:
            raise ValueError("event window phase must lie in [0, 1)")
        if not 0.0 < float(duration) < period_days / 2.0:
            raise ValueError("event window duration must be positive and shorter than P/2")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("event window role must be a non-empty string")
        windows.append({"phase": float(phase), "duration_days": float(duration), "role": role.strip()})
    if not windows:
        raise ValueError("event_windows must contain at least one declared window or be omitted")
    return windows


def calibrate_trex_noise(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    sectors: np.ndarray,
    *,
    period_days: float,
    epoch_btjd: float,
    duration_days: float,
    cadence_days_by_sector: Mapping[int, float],
    event_windows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure a shared effective-error factor from native OOT photometry.

    Adopt max(1, std(z)) * max(1, beta) per sector, with
    z = (flux - OOT median) / reported sigma and
    beta = std(mean_N(z)) / (std(z) / sqrt(N)), at N = T14/cadence.
    ``cadence_days_by_sector`` is supplied from the FITS-header-resolved
    exposures of the retained sectors; missing headers never fall back to a
    guessed sampling interval.
    The existing estimator uses sample standard deviations (ddof=1); an
    additional finite-bin RMS correction must not be applied twice.

    The primary window always includes half a cadence of padding. By default a
    circular secondary at phase 0.5 with the primary duration is excluded too;
    that default is recorded as an assumption and does not support an eccentric
    secondary elsewhere. ``event_windows`` replaces that circular window with
    caller-declared phases and durations (e.g. an eccentric secondary from a
    retained fit, or a longer eclipse), each padded by half a cadence. Event
    exclusion applies only to calibration: every point remains in the
    likelihood, so a declared eclipse is not reclassified as noise and not
    removed from the model comparison.

    ``selection`` records the windows actually used; ``secondary_mask`` and
    ``event_phase_support`` distinguish the circular assumption from declared
    phases. ``cadence_source`` identifies caller-owned FITS evidence.
    ``sector_combination`` is a conservative maximum policy envelope, not an
    inferred noise hyperparameter. ``likelihood_scope`` requires that same
    fixed factor across scenarios. Point counts use nearest integer rounding
    with ties to even. Every retained sector is assessed before an aggregate
    deferral is raised; passed sectors do not clear a failed one.
    """
    time = np.asarray(time_btjd)
    if not isinstance(cadence_days_by_sector, Mapping):
        raise ValueError("cadence_days_by_sector must map sector IDs to FITS-header cadences")
    values = np.asarray(flux)
    sector_values = np.asarray(sectors)
    if time.ndim != 1 or not time.size or values.shape != time.shape or sector_values.shape != time.shape:
        raise ValueError("TREX noise calibration requires aligned native photometry")
    if time.dtype.kind not in "iuf" or values.dtype.kind not in "iuf":
        raise ValueError("TREX noise calibration requires real, nonboolean native photometry")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError("TREX noise calibration requires finite native photometry")
    time = time.astype(float, copy=False)
    values = values.astype(float, copy=False)
    if not np.issubdtype(sector_values.dtype, np.integer) or np.any(sector_values <= 0):
        raise ValueError("TREX noise calibration requires positive integer sectors")
    errors = np.broadcast_to(validate_flux_uncertainty(sigma, values), time.shape)
    for value in (period_days, epoch_btjd, duration_days):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("TREX noise calibration requires finite real ephemeris values")
    if not 0 < duration_days < period_days / 2:
        raise ValueError("TREX noise calibration requires T14 shorter than half the period")
    declared_windows = _validated_event_windows(
        event_windows, period_days=float(period_days), duration_days=float(duration_days)
    )
    # Each window repeats with the candidate period. The primary is always masked;
    # explicit declarations replace the circular secondary assumption.
    windows = [{"phase": 0.0, "duration_days": float(duration_days), "role": "primary"}]
    if declared_windows:
        windows.extend(declared_windows)
        secondary_mask = "caller-declared-event-windows"
        selection_rule = "exclude primary and caller-declared event windows with half-cadence padding"
        event_phase_support = "declared event phase and duration supplied by the caller"
    else:
        windows.append({"phase": 0.5, "duration_days": float(duration_days), "role": "circular-half-phase"})
        secondary_mask = "circular-half-phase-window"
        selection_rule = "exclude primary and half-phase windows within (T14 + cadence)/2"
        event_phase_support = (
            "circular assumption: only a phase-0.5 secondary with the primary T14 is excluded; "
            "an eccentric secondary elsewhere remains in the calibration sample"
        )

    record: dict[str, Any] = {
        "status": "passed",
        "method": "pont-2006-transit-scale-standardized-residuals",
        "calibration_time_domain": "native-BTJD_TDB-before-phase-folding",
        "cadence_source": "FITS-header-resolved exposure of each retained sector",
        "period_days": float(period_days),
        "epoch_btjd": float(epoch_btjd),
        "duration_days": float(duration_days),
        "selection": selection_rule,
        "event_windows": [
            {
                "phase": float(window["phase"]),
                "epoch_btjd": float(epoch_btjd) + float(window["phase"]) * float(period_days),
                "duration_days": float(window["duration_days"]),
                "role": str(window["role"]),
            }
            for window in windows
        ],
        "secondary_mask": secondary_mask,
        "event_phase_support": event_phase_support,
        "sector_combination": "maximum effective-error factor over retained sectors",
        "sectors": {},
        "reference": dict(PONT_ETAL_2006),
        "inflation_reference": dict(WINN_ETAL_2008),
        "likelihood_scope": "same fixed sigma for all MC/JAX/nested scenarios; no second beta inflation",
        "limitations": (
            "diagonal beta approximation, not a GP or calibrated FPP coverage; "
            "assumes OOT stationarity and independent transit events; "
            "eccentric secondaries are excluded only when their phase and duration are declared"
        ),
        "claim_eligible": False,
    }
    for sector in np.unique(sector_values):
        selection = sector_values == sector
        native_time = time[selection]
        cadence = cadence_days_by_sector.get(int(sector))
        reason = None
        scale = None
        bin_points = 0
        oot_count = 0
        if isinstance(cadence, (bool, np.bool_)) or not isinstance(cadence, Real) or not math.isfinite(cadence) or cadence <= 0:
            reason = "missing or invalid FITS-header cadence"
            cadence = None  # deferred diagnostics must also remain finite JSON
        elif duration_days < cadence * MIN_BIN_POINTS:
            reason = "insufficient native cadence resolution at T14"
        else:
            oot = np.ones(native_time.shape, dtype=bool)
            for window in windows:
                window_factor = (window["duration_days"] + cadence) / window["duration_days"]
                oot &= out_of_transit_mask(
                    native_time, period_days=period_days,
                    epoch_btjd=epoch_btjd + window["phase"] * period_days,
                    duration_days=window["duration_days"], exclusion_factor=window_factor,
                )
            oot_count = int(np.count_nonzero(oot))
            bin_points = transit_bin_size_points(duration_days, cadence)
            if oot_count:
                with np.errstate(over="ignore", invalid="ignore"):
                    selected = np.flatnonzero(selection)[oot]
                    oot_flux = values[selected]
                    residual = (oot_flux - np.median(oot_flux)) / errors[selected]
                if not np.all(np.isfinite(residual)):
                    reason = "nonfinite standardized OOT residuals"
                else:
                    scale = pont_2006_beta(
                        residual, native_time[oot], bin_size_points=bin_points,
                        cadence_days=cadence,
                    )
            if scale is None and reason is None:
                reason = "insufficient complete OOT bins or undefined residual scatter at T14"
        sector_record = {
            "status": "deferred" if reason else "passed",
            "cadence_days": None if cadence is None else float(cadence),
            "bin_size_points": bin_points,
            "out_of_transit_sample_count": oot_count,
            "gap_tolerance_cadences": float(GAP_TOLERANCE_CADENCES),
        }
        record["sectors"][str(int(sector))] = sector_record
        if reason:
            sector_record["reason"] = reason
            continue
        white_scale = max(1.0, scale.unbinned_rms)
        sector_record.update(
            beta_raw=scale.beta_raw,
            beta=scale.beta,
            bin_size_days=scale.bin_size_days,
            bin_count=scale.bin_count,
            sample_count=scale.sample_count,
            dropped_sample_count=scale.dropped_sample_count,
            scatter_units="dimensionless standardized residuals",
            standardized_unbinned_scatter=scale.unbinned_rms,
            standardized_binned_scatter=scale.binned_rms,
            white_scatter_factor=white_scale,
            effective_error_factor=white_scale * scale.beta,
        )
    deferred = [sector for sector, item in record["sectors"].items() if item["status"] == "deferred"]
    if deferred:
        reason = "; ".join(
            f"sector {sector}: {record['sectors'][sector]['reason']}" for sector in deferred
        )
        record.update(status="deferred", reason=reason, native_uncertainty_factor=None, deferred_sectors=deferred)
        raise TrexEvidenceUnavailableError(
            f"TREX noise calibration deferred: {reason}",
            diagnostics={"noise_model": record},
        )
    record["native_uncertainty_factor"] = max(
        item["effective_error_factor"] for item in record["sectors"].values()
    )
    return record
