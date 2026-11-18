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

from trex.runtime.logging import host_logger

import contextvars

import math
import warnings
from concurrent.futures import (
    ThreadPoolExecutor as ThreadPoolExecutor,
)
from functools import (
    partial as partial,
    wraps as wraps,
)
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
from trex.evidence.mc_target import resolved_neighbor_host as resolved_neighbor_host
from trex.evidence.mc_target import calc_target_evidences as calc_target_evidences

_ORIGINAL_LNL_TP = lnL_TP
_ORIGINAL_LNL_EB = lnL_EB
_ORIGINAL_BATCH_TP = batch_lnL_TP
_ORIGINAL_BATCH_EB = batch_lnL_EB


def _retry_jax_batch(batch, *args, _prepared=None, **kwargs):
    """Retry the same draws/devices once with halved runtime chunks.

    Only a second failure reaches the existing warned, recorded batman fallback.
    No random draws, likelihood inputs, or scientific settings are regenerated.
    """
    if _prepared is not None:
        batch, args, kwargs = _prepared, (), {}
    kwargs["_chunk_plan"] = kwargs.get("_chunk_plan") or {}
    try:
        return batch(*args, **kwargs)
    except Exception as exc:
        host_logger("trex").warning(
            "JAX batch failed, retrying with halved chunks: %s: %s", type(exc).__name__, exc,
            extra={"event": "trex.batch.retry"})
    # Leave the exception scope first so its traceback cannot retain the failed
    # chunk's device buffers while the retry tries to allocate a smaller batch.
    kwargs["chunk_size_divisor"] = 2 * kwargs.get("chunk_size_divisor", 1)
    return batch(*args, **kwargs)


def _warn_jax_batch_fallback(kind: str, exc: Exception) -> None:
    """Report a failed JAX batch before its draws use the batman reference path.

    The batman likelihood evaluates the identical prior draws, so the evidence
    stays defined, but the failed accelerator must remain observable.
    """
    warnings.warn(
        "JAX batched {0} likelihood failed ({1}: {2}); evaluating these draws "
        "with the batman reference likelihood instead".format(kind, type(exc).__name__, exc),
        RuntimeWarning,
        stacklevel=3,
    )


def _record_jax_batch_fallback(
    out: "Optional[list]", kind: str, draws: int, exc: Exception
) -> None:
    """Append the reference-path provenance for one failed JAX batch.

    The warning states that the fallback happened; this record lets the
    vetting report state it afterwards.  It carries no likelihood value: the
    named draws were evaluated by the batman reference likelihood, and that
    evidence is the one the scenario returns.  ``draws`` is the number of
    transiting draws the failed batch would have covered, so a reviewer can
    weigh how much of the integral took the slow path.
    """
    if out is None:
        return
    out.append(
        {
            "kind": kind,
            "draws": int(draws),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    )



def _geometric_transit_mask(
    b: np.ndarray, rp_rs: np.ndarray, ecc: np.ndarray, argp: np.ndarray, a_rs: np.ndarray,
) -> np.ndarray:
    """Boolean mask for draws that geometrically transit.

    ``b`` is the circular-orbit impact parameter ``a_rs * |cos i|``. The
    eccentricity correction belongs on the impact parameter itself, because it
    scales the star-planet separation at conjunction:

    .. math::

        b_{\\mathrm{tra}} = \\frac{a \\cos i}{R_\\star}
                            \\frac{1 - e^2}{1 + e \\sin \\omega}

    Winn (2010), *Exoplanets*, pp. 55–77, *Transits and Occultations*,
    preprint DOI ``10.48550/arXiv.1001.2010``, eq. 7. Upstream TRICERATOPS applies it the same way
    (``marginal_likelihoods.lnZ_TTP``: ``r = a(1-e**2)/(1+e sin w)`` then
    ``b = r cos i / R_s``, with the reciprocal appearing in its ``e_corr``
    transit probability). Applying it to the ``1 + rp_rs`` threshold instead
    inverts the correction: the two agree only at ``e = 0`` and diverge with
    eccentricity in a direction set by the argument of periastron.

    A draw whose periastron separation ``a (1 - e)`` is below the summed
    host and occultor radii describes overlapping bodies, not a physical
    orbit, so it has zero support whatever its inclination. Upstream
    TRICERATOPS 1.0.20 excludes the same collisions in every eccentric
    scenario (``coll = (R_occ + R_s) > a*(1-e)``). Without this boundary the
    near-unity eccentricity tail of a short-period scene was integrated as
    support, and sent to the slow reference likelihood.
    """
    factor = (1.0 - ecc ** 2) / (1.0 + ecc * np.sin(np.radians(argp)))
    detached = a_rs * (1.0 - ecc) >= 1.0 + rp_rs
    return (np.abs(b) * factor <= 1.0 + rp_rs) & detached

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

_SCENARIO_NAMES: Dict[int, str] = {
    0: "TP", 1: "EB", 2: "EBx2P", 3: "PTP", 4: "PEB", 5: "PEBx2P",
    6: "STP", 7: "SEB", 8: "SEBx2P", 9: "DTP", 10: "DEB", 11: "DEBx2P",
    12: "BTP", 13: "BEB", 14: "BEBx2P",
}

N_TARGET_SCENARIOS: int = 15
N_NEIGHBOR_SCENARIOS: int = 3
TARGET_SCENARIO_NAMES: Tuple[str, ...] = tuple(
    _SCENARIO_NAMES[index] for index in range(N_TARGET_SCENARIOS)
)
NEIGHBOR_SCENARIO_NAMES: Tuple[str, ...] = ("NTP", "NEB", "NEBx2P")

# The target-hosted scenario table, in scenario-index order. Each row is
# ``(is_planet, is_EB, use_2xP, companion_is_host, companion_fluxratio,
# use_flat_priors, lnprior_comp, bound_companion, background_star)``, exactly
# the arguments ``_eval_scenario`` receives from ``calc_target_evidences``.
# It is module-level so the nested-evidence escalation path can route the same
# scenario slots without duplicating (and drifting from) this table.
TARGET_SCENARIO_SPECS: Tuple[tuple, ...] = (
    (True, False, False, False, 0.0, False, 0.0, None, None),     # 0 TP
    (False, True, False, False, 0.0, False, 0.0, None, None),     # 1 EB
    (False, True, True, False, 0.0, False, 0.0, None, None),      # 2 EBx2P
    (True, False, False, False, 0.0, False, 0.0, "primary", None),     # 3 PTP
    (False, True, False, False, 0.0, False, 0.0, "primary", None),     # 4 PEB
    (False, True, True, False, 0.0, False, 0.0, "primary", None),      # 5 PEBx2P
    (True, False, False, True, 0.0, False, 0.0, "secondary", None),    # 6 STP
    (False, True, False, True, 0.0, False, 0.0, "secondary", None),    # 7 SEB
    (False, True, True, True, 0.0, False, 0.0, "secondary", None),     # 8 SEBx2P
    (True, False, False, False, 0.0, False, 0.0, None, "diluted"),     # 9 DTP
    (False, True, False, False, 0.0, True, 0.0, None, "diluted"),      # 10 DEB
    (False, True, True, False, 0.0, True, 0.0, None, "diluted"),       # 11 DEBx2P
    (True, False, False, True, 0.0, False, 0.0, None, "background"),   # 12 BTP
    (False, True, False, True, 0.0, True, 0.0, None, "background"),    # 13 BEB
    (False, True, True, True, 0.0, True, 0.0, None, "background"),     # 14 BEBx2P
)
if any(is_planet and is_EB for is_planet, is_EB, *_ in TARGET_SCENARIO_SPECS):
    raise ValueError("target scenario cannot be both a planet and an eclipsing binary")

# NUMERICAL_POLICY: estimator-routing floor, not a physical or empirical
# quantity or proof of convergence. With representative sampled support, the
# large-N uncertainty proxy for a Monte Carlo log-evidence is about 1/sqrt(ESS)
# nats for Kish effective sample size (Kish 1965, Survey Sampling, section 11.7).
# The registered 0.1-nat proxy corresponds to 1 / 0.1**2 effective draws.
# Realized ESS cannot establish that support or modes were not missed.
MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE: float = 100.0

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


# Importance-sampling proposal controls
#
# Sampling every parameter from its astrophysical prior makes the scenario
# evidence an extreme-value statistic rather than an integral: the observed
# transit pins the occultor radius to a fraction of a percent, so prior draws
# essentially never land where the likelihood has support, and the reported
# lnZ becomes the largest single draw (measured Kish ESS of 1.000 at both
# 2,000 and 50,000 draws). The controls below
# define a proposal distribution that samples where the likelihood lives.
#
# Importance sampling is exact for ANY proposal whose support covers the
# prior's: the draws carry a compensating weight ln(p/q), so these numbers
# change the estimator's variance and can never change the value it converges
# to. They are numerical-method controls, not physical or empirical
# quantities, and none of them is a fitted or tuned scientific parameter.

# NUMERICAL_POLICY: dyadic depth for locating a proposal window on an inverse-CDF
# prior. 2^-24 = 6e-8 sits far inside any window, and the log weights use
# the window actually drawn from, so this bound moves efficiency, never the evidence.
_QUANTILE_BISECTION_ITERATIONS: int = 24

# NUMERICAL_POLICY: Hesterberg (1995, Technometrics 37, 185, DOI
# 10.1080/00401706.1995.10484303) defensive mixture fraction. A share of every
# scenario's draws is always taken from the unmodified prior, which keeps the
# proposal's support identical to the prior's and bounds every importance
# weight by 1/fraction per mixture. One half is the maximally conservative
# choice: each mixture at most doubles a draw's weight relative to a prior draw
# (the occultor and inclination mixtures are independent, so together at most
# four times), so a poor proposal degrades the estimator towards plain prior
# sampling instead of inflating its variance without bound.
_DEFENSIVE_PRIOR_FRACTION: float = 0.5

# NUMERICAL_POLICY: size the proposal window from measurement uncertainty
# rather than a fixed multiplier; five Gaussian standard deviations cover
# the bulk of the posterior while retaining the defensive prior component.
_PROPOSAL_COVERAGE_SIGMA: float = 5.0

# NUMERICAL_POLICY: multiplicative slack on the depth-to-radius-ratio and
# duration-to-impact-parameter conversions, which are exact only for a uniform
# disc at zero impact parameter. Quadratic limb darkening makes the central
# depth exceed the radius ratio squared, so the true radius ratio is smaller
# than the naive square root; grazing geometry pushes it the other way; and an
# eclipsing pair's own secondary light dilutes its eclipse. A factor of 1.5
# widens the proposal; the defensive component retains prior support even when
# this window misses part of the likelihood.
_PROPOSAL_GEOMETRY_SPAN: float = 1.5

# NUMERICAL_POLICY: inclusive upper bound on the eccentricity a Monte Carlo draw
# may carry into the device batch. The bound is the largest representable double
# below one, so every eccentricity the Kipping (2013) and Moe & Di Stefano (2017)
# priors can admit is evaluated by the device solver. Batman cannot adjudicate
# the near-unity tail it used to own: its eccentric-anomaly solver returns no
# finite value for most arguments of periastron at e >= 0.98, and those draws
# became unresolved numerical failures rather than evidence. The settled
# reference is not batman agreement but an independent bisection solution of
# E - e sin E = M over [0, 0.999]
# (tests/trex/accelerators/test_jax_kepler_domain.py), which is the same solver
# ``evidence._jax_unit_integrand`` already applies to the whole admitted domain.
# The geometric mask removes colliding periastra before this router, so no draw
# reaches the solver with a separation below the orbit's detachment bound.
# Batman remains the scalar/CPU reference and the batch fallback; its agreement
# with the device batch below e = 0.97 is still checked in
# tests/trex/accelerators/test_engine_likelihoods.py.
_JAX_MAX_SUPPORTED_ECCENTRICITY: float = float(np.nextafter(1.0, 0.0))




# NUMERICAL_POLICY: the upper edge of an inverse-CDF prior's support is reached
# only at the unit variate's supremum, which binary64 cannot represent. The next
# float below one is the closest representable interior point and is exact
# enough for a support-endpoint comparison; it is a numerical boundary, not a
# physical quantity.
_SUPPORT_ENDPOINT_QUANTILE: float = float(np.nextafter(1.0, 0.0))




def _finish_stages(stages) -> float:
    """Run a :func:`_scenario_stages` generator to its returned lnZ."""
    while True:
        try:
            next(stages)
        except StopIteration as stop:
            return stop.value


@wraps(_scenario_stages)
def _eval_scenario(*args, **kwargs) -> float:
    return _finish_stages(_scenario_stages(*args, **kwargs))


_ORIGINAL_EVAL_SCENARIO = _eval_scenario


class _ScenarioOverlap:
    """Run one MC while the next scenario prepares inputs, compiles and runs its CPU pilot.

    Preparation stays on the calling thread in scenario order and the likelihood
    stage draws no random numbers, so every draw and lnZ is bit-identical to the
    sequential loop; the GPU no longer idles through each EB scenario's CPU-side
    window bisection. Results are recorded in submission order.
    """

    def __init__(self) -> None:
        self._enabled = _eval_scenario is _ORIGINAL_EVAL_SCENARIO
        self._executor = ThreadPoolExecutor(max_workers=1) if self._enabled else None
        self._pending = None

    def submit(self, call: partial, record) -> None:
        if not self._enabled:
            record(call())
            return
        previous = ((self._pending[0], self._pending[2], self._pending[3])
                    if self._pending is not None else None)
        stages = _scenario_stages(*call.args, **call.keywords, _previous_scenario=previous)
        try:
            prepared = next(stages)
        except StopIteration as stop:
            self.wait()
            record(stop.value)
            return
        except BaseException:
            self.wait()
            raise
        self.wait()
        kernel = getattr(getattr(prepared, "evaluate", None), "kernel", None)
        # The device lease lives in context variables, so the worker runs in a copy.
        future = self._executor.submit(contextvars.copy_context().run, _finish_stages, stages)
        self._pending = (future, record, call.keywords.get("ldc_scenario"),
                         kernel)

    def wait(self) -> None:
        if self._pending is not None:
            future, record, _, _ = self._pending
            self._pending = None
            record(future.result())

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)


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


def scenario_period_log_priors(period_days: float, count: int) -> np.ndarray:
    """Per-scenario log P_orb prior for the first ``count`` scenario slots.

    G21 Sect. 2.4.1 applies the planet prior to transiting-planet scenarios and
    the eclipsing-binary prior to every EB scenario; ``x2P`` scenarios apply
    ``2 P_orb`` to the model and therefore take the prior at that physical
    period.  The assignment follows :data:`TARGET_SCENARIO_SPECS` and the
    neighbour slots (NTP planet; NEB and NEBx2P EB), not the upstream code's
    unused function names.
    """
    from trex.evidence.nested_controls import _validate_scalar

    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("scenario count must be a non-negative integer")
    period_days = _validate_scalar(period_days, "period_days", positive=True)
    values = np.zeros(count, dtype=float)
    categories = {}
    for index in range(values.size):
        if index < N_TARGET_SCENARIOS:
            is_planet = bool(TARGET_SCENARIO_SPECS[index][0])
            use_2xP = bool(TARGET_SCENARIO_SPECS[index][2])
        else:
            offset = (index - N_TARGET_SCENARIOS) % N_NEIGHBOR_SCENARIOS
            is_planet = offset == 0
            use_2xP = offset == 2
        key = (is_planet, use_2xP)
        if key not in categories:
            categories[key] = scenario_period_log_prior(period_days, planet=is_planet, use_2xP=use_2xP)
        values[index] = categories[key]
    return values


def _period_weighted_evidences(lnZ: np.ndarray, period_days: Optional[float]) -> np.ndarray:
    """Add the observed-period scenario factors without changing stored evidence."""
    values = np.asarray(lnZ, dtype=float)
    return values if period_days is None else values + scenario_period_log_priors(period_days, values.size)


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
