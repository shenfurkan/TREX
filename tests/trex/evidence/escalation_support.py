"""Synthetic escalation arguments shared by TREX's nested-sampling tests.

The values describe routing and checkpoint behavior; none of them is a
measurement or a production draw budget. ``n_draws`` is TREX's production
floor because the escalation policy reads that floor, while the estimator is
stubbed in every test using these helpers.
"""

from __future__ import annotations

import numpy as np

from trex.app.engine import _TREX_DEFAULT_N_DRAWS
from trex.evidence.marginal_likelihoods import (
    MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE,
    N_NEIGHBOR_SCENARIOS,
    N_TARGET_SCENARIOS,
)
from trex.evidence.nested_evidence import NestedEvidenceResult
from trex.evidence.nested_controls import _DEFAULT_EVIDENCE_TOLERANCE
from trex.foundation.physical_constants import SECONDS_PER_DAY

_SYNTHETIC_CADENCE_COUNT: int = 21
_SYNTHETIC_BASELINE_DAYS: float = 0.1
_SYNTHETIC_FLUX_SIGMA: float = 1.0e-3
_SYNTHETIC_PERIOD_DAYS: float = 3.5
_SYNTHETIC_EXPOSURE_DAYS: float = 120.0 / SECONDS_PER_DAY
_SYNTHETIC_HOST_MASS_SOLAR: float = 1.0
_SYNTHETIC_HOST_RADIUS_SOLAR: float = 1.0
_SYNTHETIC_U1: float = 0.3
_SYNTHETIC_U2: float = 0.27
_SYNTHETIC_DEPTH_PPM: float = 1200.0
_CONVERGED_LOG_EVIDENCE: float = -100.0
_BOUNDARY_LOG_EVIDENCE: float = -100.5
_NON_NEGLIGIBLE_NESTED_LOG_EVIDENCE: float = -100.2
_CONVERGED_EFFECTIVE_SAMPLE_SIZE: float = 5.0 * MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
_SINGLE_DRAW_EFFECTIVE_SAMPLE_SIZE: float = 1.0


def _synthetic_time():
    return np.linspace(
        -_SYNTHETIC_BASELINE_DAYS, _SYNTHETIC_BASELINE_DAYS, _SYNTHETIC_CADENCE_COUNT
    )


def _boundary_support(*indices):
    """Routing evidence marking these scenario windows as outside their support."""
    return {
        int(index): {
            "state": "below-support",
            "draws_considered": 1,
            "below_fraction": 1.0,
            "above_fraction": 0.0,
        }
        for index in indices
    }


def _evidence_vector(values):
    """A full scenario evidence vector with only the named slots evaluated."""
    lnZ = np.full(N_TARGET_SCENARIOS + N_NEIGHBOR_SCENARIOS, -np.inf)
    for index, value in values.items():
        lnZ[index] = value
    return lnZ


def _binding_evidence():
    """A two-scenario set whose single boundary scenario holds real probability."""
    lnZ = _evidence_vector({0: _CONVERGED_LOG_EVIDENCE, 1: _BOUNDARY_LOG_EVIDENCE})
    ess = {
        0: _CONVERGED_EFFECTIVE_SAMPLE_SIZE,
        1: _SINGLE_DRAW_EFFECTIVE_SAMPLE_SIZE,
    }
    return lnZ, ess


def _base_kwargs(**overrides):
    """Escalation arguments for a target-hosted scene with no neighbors."""
    kwargs = dict(
        support_state=_boundary_support(1, 2),
        escalation_budget_floor=_TREX_DEFAULT_N_DRAWS,
        n_draws=_TREX_DEFAULT_N_DRAWS,
        random_seed=7,
        time=_synthetic_time(),
        flux=np.ones(_SYNTHETIC_CADENCE_COUNT),
        sigma=_SYNTHETIC_FLUX_SIGMA,
        period_days=_SYNTHETIC_PERIOD_DAYS,
        depth_ppm=_SYNTHETIC_DEPTH_PPM,
        M_s_Msun=_SYNTHETIC_HOST_MASS_SOLAR,
        R_s_Rsun=_SYNTHETIC_HOST_RADIUS_SOLAR,
        u1=_SYNTHETIC_U1,
        u2=_SYNTHETIC_U2,
        exptime_days=_SYNTHETIC_EXPOSURE_DAYS,
        bin_integration_days=0.0,
        target_aperture_flux_fraction=1.0,
        contrast_separations=None,
        contrast_values=None,
        plx=None,
        target_tess_mag=10.0,
        background_population=None,
        background_area_sq_deg=None,
        nearby_stars=None,
        target_feh=0.0,
        resolution_limit_arcsec=0.1,
        nsamples=20,
        n_jobs=1,
    )
    kwargs.update(overrides)
    return kwargs


def _result(log_evidence, *, stopped_short=False):
    """A routing double with explicit completion or truncation provenance."""
    return NestedEvidenceResult(
        log_evidence=float(log_evidence),
        log_evidence_error=0.05,
        n_live_points=400,
        n_likelihood_calls=1000,
        n_iterations=100,
        sampler="stub",
        sampler_version="0",
        random_seed=1,
        stopped_short=stopped_short,
        stopped_on_evidence_tolerance=not stopped_short,
        stop_reason="truncated" if stopped_short else "evidence-tolerance",
        base_run_delta_logz=float("inf") if stopped_short else _DEFAULT_EVIDENCE_TOLERANCE / 2,
    )
