"""Vectorized Monte Carlo integration for the TREX statistical vetting engine.

Implements log-evidence estimation for all 18 astrophysical scenarios
defined in Giacalone et al. (2021, AJ, 161, 24).

Architecture:
    Each scenario-dedicated function performs the same pattern:
    1. Draw ``n_draws`` samples from astrophysical priors.
    2. Compute geometric transit probability (b, a_Rs, rp_rs).
    3. For transiting draws, evaluate the Mandel-Agol likelihood.
    4. Compute lnZ = log(mean(exp(lnL + lnprior))) via _log_mean_exp.

Units, verified source, and failure boundary
---------------------------------------------
The scenario framework is Giacalone et al. (2021), AJ 161, 24,
DOI ``10.3847/1538-3881/abc6af``. Time/period/exposure
are days; flux and sigma are normalized relative flux; depth is ppm; masses and
radii are solar/Earth units as named; semimajor axis is cm; contrast separation
is arcsec; parallax is mas; and log evidence, priors, FPP/NFPP, and scenario
probabilities are dimensionless.  Geometrically impossible draws contribute
zero weight; numerical anomalies are surfaced by the normalization status.
Missing/invalid scene values or exposure is rejected.
"""

from __future__ import annotations

import math
from typing import (
    Any as Any,
    Dict as Dict,
    Mapping as Mapping,
    Optional as Optional,
    Tuple as Tuple,
)

import numpy as np

from trex.foundation.errors import TrexEvidenceUnavailableError as TrexEvidenceUnavailableError

from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC as GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC, Msun as Msun, Rsun as Rsun, Rearth as Rearth, G as G
from trex.foundation._numerics import _log_mean_exp as _log_mean_exp, _log_weight_ess as _log_weight_ess, phase_bin_widths as phase_bin_widths
from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid as load_triceratops_tess_grid
from trex.populations.companion_imaging import imaging_log_prior as imaging_log_prior, binary_imaging_log_prior as binary_imaging_log_prior
from trex.foundation.physical_constants import FRACTION_PER_PART_PER_MILLION as FRACTION_PER_PART_PER_MILLION, MILLIARCSECONDS_PER_ARCSECOND as MILLIARCSECONDS_PER_ARCSECOND
from trex.runtime.progress import emit_progress as emit_progress
from trex.stellar.stellar_relations import stellar_relations as stellar_relations
from trex.stellar.stellar_relations import stellar_masses_of_radii as stellar_masses_of_radii
from trex.stellar.stellar_relations import stellar_radii_in_domain as stellar_radii_in_domain
from trex.models.orbital_mechanics import semi_major_axis_cgs as semi_major_axis_cgs
from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT as POGSON_FLUX_EXPONENT
from trex.models.flux_calibration import renorm_flux_to_host as renorm_flux_to_host
from trex.stellar_light.passband import tess_surface_brightness_ratio as tess_surface_brightness_ratio
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR as STELLAR_RELATIONS_MASS_MIN_SOLAR
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MAX_SOLAR as STELLAR_RELATIONS_MASS_MAX_SOLAR
from trex.populations.priors import eb_mass_ratio_support as eb_mass_ratio_support, radius_quantile as radius_quantile, sample_rp as sample_rp, sample_inc as sample_inc, sample_ecc as sample_ecc, sample_w as sample_w, scenario_period_log_prior as scenario_period_log_prior, sample_q_host_mass_ratio as sample_q_host_mass_ratio, host_mass_ratio_inverse_cdf as host_mass_ratio_inverse_cdf, MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR as MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR, MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS as MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS, MOE2017_LOGPERIOD_MAX_DAYS as MOE2017_LOGPERIOD_MAX_DAYS, companion_log_separation_au as companion_log_separation_au, companion_population_frequency as companion_population_frequency, background_population_expectation as background_population_expectation, sample_companion_log_period as sample_companion_log_period, sample_companion_log_projected_separation as sample_companion_log_projected_separation, sample_companion_log_semimajor_axis as sample_companion_log_semimajor_axis, _companion_log_period_from_log_separation as _companion_log_period_from_log_separation, companion_log_period_from_log_projected_separation as companion_log_period_from_log_projected_separation, companion_hierarchical_stability_supported as companion_hierarchical_stability_supported, sample_q_companion as sample_q_companion, WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR as WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR
from trex.models.likelihoods import lnL_TP as lnL_TP, lnL_EB as lnL_EB
from trex.accelerators._jax_models import HAS_JAX_TREX as HAS_JAX_TREX, batch_lnL_TP as batch_lnL_TP, batch_lnL_EB as batch_lnL_EB

from trex.evidence.mc_proposals import _window_quantile as _window_quantile
from trex.evidence.mc_proposals import _defensive_window_draws as _defensive_window_draws
from trex.evidence.mc_proposals import _per_draw_values as _per_draw_values
from trex.evidence.mc_proposals import _proposal_support_state as _proposal_support_state
from trex.evidence.mc_population import _host_grid_coefficients as _host_grid_coefficients
from trex.evidence.mc_population import _draw_bound_companion as _draw_bound_companion
from trex.evidence.mc_population import _draw_background_star as _draw_background_star
from trex.evidence.mc_scenario import _scenario_stages as _scenario_stages
from trex.evidence.mc_scenario import _eval_scenario as _eval_scenario
from trex.evidence.mc_scenario import _geometric_transit_mask as _geometric_transit_mask
from trex.evidence.mc_scenario import _ScenarioOverlap as _ScenarioOverlap
from trex.evidence.mc_scenario import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE as MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
# Historical re-exports: these controls moved to the Monte Carlo owner modules,
# and callers that imported them from here keep that path.
from trex.evidence.mc_scenario import _JAX_MAX_SUPPORTED_ECCENTRICITY as _JAX_MAX_SUPPORTED_ECCENTRICITY
from trex.evidence.mc_scenario import _PROPOSAL_COVERAGE_SIGMA as _PROPOSAL_COVERAGE_SIGMA
from trex.evidence.mc_scenario import _PROPOSAL_GEOMETRY_SPAN as _PROPOSAL_GEOMETRY_SPAN
from trex.evidence.mc_scenario import _record_jax_batch_fallback as _record_jax_batch_fallback
from trex.evidence.mc_scenario import _warn_jax_batch_fallback as _warn_jax_batch_fallback
from trex.evidence.mc_proposals import _DEFENSIVE_PRIOR_FRACTION as _DEFENSIVE_PRIOR_FRACTION
from trex.evidence.mc_proposals import _QUANTILE_BISECTION_ITERATIONS as _QUANTILE_BISECTION_ITERATIONS
from trex.evidence.mc_target import _SCENARIO_NAMES as _SCENARIO_NAMES
from trex.evidence.mc_target import N_NEIGHBOR_SCENARIOS as N_NEIGHBOR_SCENARIOS
from trex.evidence.mc_target import N_TARGET_SCENARIOS as N_TARGET_SCENARIOS
from trex.evidence.mc_target import NEIGHBOR_SCENARIO_NAMES as NEIGHBOR_SCENARIO_NAMES
from trex.evidence.mc_target import TARGET_SCENARIO_SPECS as TARGET_SCENARIO_SPECS
from trex.evidence.mc_target import _period_weighted_evidences as _period_weighted_evidences
from trex.evidence.mc_target import scenario_period_log_priors as scenario_period_log_priors
from trex.evidence.mc_target import resolved_neighbor_host as resolved_neighbor_host
from trex.evidence.mc_target import calc_target_evidences as calc_target_evidences


# Scenario identifiers

SCENARIO_TP: int = 0
SCENARIO_EB: int = 1
SCENARIO_EBX2P: int = 2
SCENARIO_PTP: int = 3
SCENARIO_PEB: int = 4
SCENARIO_PEBX2P: int = 5
SCENARIO_STP: int = 6
SCENARIO_SEB: int = 7
SCENARIO_SEBX2P: int = 8
SCENARIO_DTP: int = 9
SCENARIO_DEB: int = 10
SCENARIO_DEBX2P: int = 11
SCENARIO_BTP: int = 12
SCENARIO_BEB: int = 13
SCENARIO_BEBX2P: int = 14
SCENARIO_NTP_OFFSET: int = 15
SCENARIO_NEB_OFFSET: int = 16
SCENARIO_NEBX2P_OFFSET: int = 17

TARGET_SCENARIO_NAMES: Tuple[str, ...] = tuple(
    _SCENARIO_NAMES[index] for index in range(N_TARGET_SCENARIOS)
)

# NUMERICAL_POLICY: routing allowance on the realized normalized mass of
# sub-floor scenarios, not a bound on their true evidence or on FPP error.
# ESS alone cannot distinguish missed support from genuinely negligible support.
# An arbitrary underestimate of evidence can turn a tiny realized probability
# into a dominant true probability after renormalization.
# The registered value remains one tenth of FPP_THRESHOLD in diagnostics;
# this convention does not guarantee that an FPP threshold cannot be crossed.
# Production escalation replaces accepted scenario estimates, but any remaining
# sub-floor MC evidence lacks a convergence certificate. Conditional diagnostic
# estimates may still be retained under this heuristic; the claim C5 reader
# must defer an unbounded residual rather than treat its small mass as proof.
MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY: float = 1.0e-3


def degenerate_evidence_scenarios(
    lnZ: np.ndarray, effective_sample_sizes: Mapping[int, float]
) -> Dict[str, float]:
    """Return the evaluated scenarios whose evidence is not a converged integral.

    The ESS mapping identifies scenarios actually sampled. An evaluated ``-inf``
    means no finite sampled weight, not a certificate that the prior is empty.
    Structurally excluded slots have no ESS entry. Return names and failing ESS.
    """
    failures: Dict[str, float] = {}
    for index, ess in effective_sample_sizes.items():
        if index < 0 or index >= len(lnZ):
            continue
        if not np.isfinite(lnZ[index]) or not math.isfinite(ess) or ess < MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE:
            failures[_SCENARIO_NAMES.get(index, "N{0}".format(index))] = float(ess)
    return failures


def unconverged_probability_mass(
    probs: np.ndarray, effective_sample_sizes: Mapping[int, float], lnZ: np.ndarray
) -> float:
    """Sum realized dimensionless masses of finite-evidence, sub-floor scenarios.

    This direct sum over ``probs`` is an estimator-routing statistic, not a bound
    on true scenario evidence or FPP error: missed support may be arbitrarily
    underestimated. ``lnZ`` is dimensionless natural-log evidence and the ESS
    values are effective draw counts. This statistic never grants claim eligibility.
    """
    failing = degenerate_evidence_scenarios(lnZ, effective_sample_sizes)
    if not failing:
        return 0.0
    # Sorted: the sum must not depend on ESS dict order, which differs between a GPU run and a resume.
    indices = sorted(
        index
        for index in effective_sample_sizes
        if _SCENARIO_NAMES.get(index, "N{0}".format(index)) in failing and index < len(probs)
    )
    return float(np.sum(probs[indices])) if indices else 0.0


def compute_fpp_nfpp(
    lnZ: np.ndarray,
    effective_sample_sizes: Optional[Mapping[int, float]] = None,
    *,
    period_days: Optional[float] = None,
) -> Tuple[float, float, np.ndarray, str]:
    """Compute dimensionless conditional FPP/NFPP by normalizing log-evidences.

    Args:
        lnZ: Dimensionless natural-log evidences per scenario (index order above).
            These are prior-free stored evidences; the G21 period scenario prior
            is applied here, exactly once, and never written back into ``lnZ``.
        effective_sample_sizes: Optional per-scenario Kish effective sample
            sizes from :func:`calc_target_evidences`. When supplied, realized
            sub-floor mass above the registered routing allowance returns
            ``degenerate_evidence`` and no FPP/NFPP. A smaller residual can retain
            conditional diagnostic estimates, not an evidence-error bound or a
            C5 convergence certificate. No claim eligibility is granted here.
        period_days: Observed transit period [days].  Supplying it applies the
            G21 Sect. 2.4.1 scenario prior to every scenario before
            normalization; omitting it reproduces the historical prior-free
            normalization for callers that do not own a period.

    Returns:
        (FPP, NFPP, probs, status): Dimensionless FPP, NFPP and normalized
        probability vector, plus the diagnostic routing status.
    """
    from trex.foundation._numerics import _normalize_probabilities

    values = _period_weighted_evidences(lnZ, period_days)
    if len(values) < N_TARGET_SCENARIOS:
        raise ValueError(f"scenario vector must contain at least {N_TARGET_SCENARIOS} target scenarios")
    probs, norm_status = _normalize_probabilities(values)

    zero_sampled = bool(effective_sample_sizes is not None and any(
        0 <= index < len(values) and np.isneginf(values[index])
        for index in effective_sample_sizes
    ))
    if norm_status == "all_neginf" and zero_sampled:
        return float("nan"), float("nan"), probs, "degenerate_evidence"
    if norm_status != "ok":
        fpp = 1.0 if norm_status == "all_neginf" else float("nan")
        return fpp, float("nan"), probs, norm_status

    if effective_sample_sizes is not None:
        unconverged = degenerate_evidence_scenarios(values, effective_sample_sizes)
        # Zero observed weight supplies no upper bound; the residual test must
        # allow arbitrary positive mass instead of multiplying zero by 1e4.
        from trex.evidence.evidence_escalation_policy import residual_could_flip_decision

        if unconverged and (
            unconverged_probability_mass(probs, effective_sample_sizes, values)
            > MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY
            or (zero_sampled and residual_could_flip_decision(values, effective_sample_sizes))
        ):
            # The probability vector is returned unchanged so a reviewer can see
            # the collapse, but no FPP/NFPP is derived from it.
            return float("nan"), float("nan"), probs, "degenerate_evidence"

    # FPP is the normalized mass of every retained non-planet-hosted scenario.
    # Summing that mass directly is algebraically equivalent to
    # ``1 - (P_TP + P_PTP + P_DTP)`` but preserves finite low-probability false
    # scenario mass that binary64 subtraction from unity would erase. Do not
    # clip this result: a normalization anomaly must remain observable.
    n_total = len(probs)
    planet_scenario_indices = (SCENARIO_TP, SCENARIO_PTP, SCENARIO_DTP)
    false_scenario_mask = np.ones(n_total, dtype=bool)
    false_scenario_mask[list(planet_scenario_indices)] = False
    fpp = float(np.sum(probs[false_scenario_mask]))
    nfpp = float(np.sum(probs[N_TARGET_SCENARIOS:])) if n_total > N_TARGET_SCENARIOS else 0.0

    return fpp, nfpp, probs, norm_status


__all__ = [
    "SCENARIO_TP",
    "SCENARIO_EB",
    "SCENARIO_EBX2P",
    "SCENARIO_PTP",
    "SCENARIO_PEB",
    "SCENARIO_PEBX2P",
    "SCENARIO_STP",
    "SCENARIO_SEB",
    "SCENARIO_SEBX2P",
    "SCENARIO_DTP",
    "SCENARIO_DEB",
    "SCENARIO_DEBX2P",
    "SCENARIO_BTP",
    "SCENARIO_BEB",
    "SCENARIO_BEBX2P",
    "N_TARGET_SCENARIOS",
    "N_NEIGHBOR_SCENARIOS",
    "TARGET_SCENARIO_NAMES",
    "NEIGHBOR_SCENARIO_NAMES",
    "MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE",
    "MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY",
    "calc_target_evidences",
    "compute_fpp_nfpp",
    "scenario_period_log_priors",
    "degenerate_evidence_scenarios",
    "unconverged_probability_mass",
    "_eval_scenario",
    "_geometric_transit_mask",
]
