"""Nested-sampling evidence estimator for the hosted TREX scenarios.

Nested sampling estimates the same evidence integral as production Monte
Carlo when importance weights fail the registered convergence floor:

.. math::

    Z = \\int L(\\theta) \\, \\pi(\\theta) \\, d\\theta

Reuse the scenario, inverse-CDF priors, forward model, and likelihood from
:mod:`marginal_likelihoods`, :mod:`priors`, and :mod:`likelihoods`. The isotropic
inclination variate is uniform within the geometric transit band; its
``ln(1 - band_low)`` likelihood factor restores the excluded prior mass.
The excluded band has zero likelihood, so this change of variables preserves
the integral.

Covered scenarios, and the host convention
-------------------------------------------
* ``tp_scenario_nested_evidence``: the transiting-planet scenario.  The host
  parameters are explicit, so the same call covers a resolved neighbor host
  by passing that neighbor's mass, radius, limb-darkening coefficients, and
  aperture flux fraction; the default broken planet-radius law is shared
  with the production neighbor path.
* ``eb_scenario_nested_evidence``: the eclipsing-pair scenario, with
  ``use_2xP`` selecting the ``EBx2P`` slot.  Its second parameter is the Moe
  & Di Stefano (2017) mass-ratio variate in place of the planet radius, and
  the semimajor axis follows each draw's own total mass, exactly as
  ``_eval_scenario`` computes it.

* ``bound_companion_population`` and ``background_star_population``: the two
  unresolved populations, as extra prior dimensions either scenario function
  accepts through ``population``.  They cover the *diluted* slots -- ``PTP``,
  ``PEB``, ``PEBx2P`` (unresolved bound companion) and ``DTP``, ``DEB``,
  ``DEBx2P`` (unresolved background star) -- where the frame's own host still
  carries the event.  Each block reuses the production inverse CDFs, the
  production retention mask (imaging detection, resolvability, and the
  measured mass-ratio domain), and the production occurrence rate.

The population-*hosted* slots (``STP``/``SEB``/``SEBx2P`` on the unresolved
companion, ``BTP``/``BEB``/``BEBx2P`` on the background star) are covered
through the ``hosted_population`` argument: the drawn source is the host, so
its mass, radius, temperature, orbit geometry, and limb darkening are
resolved inside the likelihood per point, and the target's flux dilutes the
event exactly as the production ``companion_is_host=True`` branch does.
Resolved-neighbor slots are covered by the explicit-host arguments
(``M_s_Msun``/``R_s_Rsun``/``u1``/``u2`` and the neighbor's aperture flux
fraction, with the same default broken radius prior for the neighbor planet slot).

Verified sources, units, and failure boundary
---------------------------------------------
Scenario framework: Giacalone et al. (2021), AJ 161, 24,
DOI ``10.3847/1538-3881/abc6af``.  Nested sampling: Skilling (2006), Bayesian
Analysis 1, 833, DOI ``10.1214/06-BA127``. Sampler: dynesty, Speagle (2020),
MNRAS 493, 3132, DOI ``10.1093/mnras/staa278``. Time/period/
exposure are days, radius is Earth radii, mass is solar masses, flux and
sigma are normalized relative flux, and log evidence is dimensionless.
Results are log evidences, not scenario probabilities. Non-finite or incomplete
sampler output raises; structural scenario impossibilities
(an out-of-domain host) raise at entry
rather than fabricating an estimate.
"""

from __future__ import annotations

import math as math
import multiprocessing as multiprocessing
import os as os
from time import monotonic as _monotonic  # noqa: F401 - historical re-export/patch surface
import warnings as warnings
from typing import Mapping as Mapping, NamedTuple as NamedTuple, Optional as Optional, Sequence as Sequence, Tuple as Tuple, Union as Union

import numpy as np  # noqa: F401 - historical re-export/patch surface
from scipy.special import logsumexp as logsumexp

from trex.foundation.physical_constants import FULL_TURN_DEGREES as FULL_TURN_DEGREES, MILLIARCSECONDS_PER_ARCSECOND as MILLIARCSECONDS_PER_ARCSECOND
from trex.runtime.hardware import resolve_n_jobs as resolve_n_jobs
from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC as GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC, Rearth as Rearth, Msun as Msun, Rsun as Rsun
from trex.populations.companion_imaging import ImagingConstraint as ImagingConstraint
from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT as POGSON_FLUX_EXPONENT
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MAX_SOLAR as STELLAR_RELATIONS_MASS_MAX_SOLAR
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR as STELLAR_RELATIONS_MASS_MIN_SOLAR
from trex.populations.companion_imaging import contrast_limit_at_separation as contrast_limit_at_separation
from trex.models.flux_calibration import renorm_flux_to_host as renorm_flux_to_host
from trex.models.orbital_mechanics import semi_major_axis_cgs as semi_major_axis_cgs
from trex.stellar.stellar_relations import stellar_relations as stellar_relations
from trex.stellar_light.passband import tess_surface_brightness_ratio as tess_surface_brightness_ratio
from trex.foundation._numerics import phase_bin_widths as phase_bin_widths
from trex.evidence.host_measure import HostMeasure
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.evidence.nested_checkpoint import NestedCheckpointPaused as NestedCheckpointPaused, SamplerCheckpoint as _SamplerCheckpoint, checkpoint_context as _checkpoint_context, checkpoint_identity as _checkpoint_identity, load_completed as _load_completed, load_importance_progress as _load_importance_progress, nested_checkpoint as nested_checkpoint, save_completed as _save_completed, save_importance_progress as _save_importance_progress  # noqa: F401 - resolved through the nested facade
from trex.models.likelihoods import lnL_EB as lnL_EB, lnL_TP as lnL_TP
from trex.evidence.marginal_likelihoods import _geometric_transit_mask as _geometric_transit_mask, _host_grid_coefficients as _host_grid_coefficients
from trex.populations.priors import MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS as MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS, MOE2017_LOGPERIOD_MAX_DAYS as MOE2017_LOGPERIOD_MAX_DAYS, MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR as MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR, background_population_expectation as background_population_expectation, companion_log_separation_au as companion_log_separation_au, companion_population_frequency as companion_population_frequency, eb_mass_ratio_support as eb_mass_ratio_support, sample_companion_log_period as sample_companion_log_period, sample_companion_log_projected_separation as sample_companion_log_projected_separation, sample_companion_log_semimajor_axis as sample_companion_log_semimajor_axis, _companion_log_period_from_log_separation as _companion_log_period_from_log_separation, companion_log_period_from_log_projected_separation as companion_log_period_from_log_projected_separation, companion_hierarchical_stability_supported as companion_hierarchical_stability_supported, sample_ecc as sample_ecc, sample_inc as sample_inc, sample_q_companion as sample_q_companion, sample_q_host_mass_ratio as sample_q_host_mass_ratio, sample_rp as sample_rp, sample_w as sample_w, WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR as WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR


from trex.evidence._blackjax_types import BLACKJAX_CALIBRATED_LIVE_POINTS as _BLACKJAX_CALIBRATED_LIVE_POINTS, blackjax_call_ceiling as _blackjax_call_ceiling
from trex.evidence.nested_controls import ImportanceEvidenceResult as ImportanceEvidenceResult
from trex.evidence.nested_controls import NestedEvidenceResult as NestedEvidenceResult
from trex.evidence.nested_controls import _DEFAULT_EVIDENCE_TOLERANCE as _DEFAULT_EVIDENCE_TOLERANCE
from trex.evidence.nested_controls import _DEFAULT_LIVE_POINTS as _DEFAULT_LIVE_POINTS
from trex.evidence.nested_controls import _DYNESTY_STOPPED_SHORT_WARNING as _DYNESTY_STOPPED_SHORT_WARNING
from trex.evidence.nested_controls import _HOSTED_N_DIMENSIONS as _HOSTED_N_DIMENSIONS
from trex.evidence.nested_controls import _IMPORTANCE_CENTRES as _IMPORTANCE_CENTRES
from trex.evidence.nested_controls import _IMPORTANCE_CHECKPOINT_SECONDS as _IMPORTANCE_CHECKPOINT_SECONDS
from trex.evidence.nested_controls import _IMPORTANCE_DEFENSIVE_FRACTION as _IMPORTANCE_DEFENSIVE_FRACTION
from trex.evidence.nested_controls import _IMPORTANCE_DENSITY_CHUNK as _IMPORTANCE_DENSITY_CHUNK
from trex.evidence.nested_controls import _IMPORTANCE_DRAWS as _IMPORTANCE_DRAWS
from trex.evidence.nested_controls import _IMPORTANCE_ESTIMATOR as _IMPORTANCE_ESTIMATOR
from trex.evidence.nested_controls import _IMPORTANCE_EVALUATION_CHUNK as _IMPORTANCE_EVALUATION_CHUNK
from trex.evidence.nested_controls import _INCLINATION_MAX_DEGREES as _INCLINATION_MAX_DEGREES
from trex.evidence.nested_controls import _MULTIPROCESSING_START_METHOD as _MULTIPROCESSING_START_METHOD
from trex.evidence.nested_controls import _NATIVE_INNER_STEPS_PER_DIMENSION as _NATIVE_INNER_STEPS_PER_DIMENSION
from trex.evidence.nested_controls import _native_kernel_policy as _native_kernel_policy
from trex.evidence.nested_controls import _native_kernel_policy_identity as _native_kernel_policy_identity
from trex.evidence.nested_controls import _selected_importance_proposal as _selected_importance_proposal
from trex.evidence.nested_controls import _NESTED_ESTIMATOR_ENV as _NESTED_ESTIMATOR_ENV
from trex.evidence.nested_controls import _POPULATION_N_DIMENSIONS as _POPULATION_N_DIMENSIONS
from trex.evidence.nested_controls import _SAMPLER_NAME as _SAMPLER_NAME
from trex.evidence.nested_controls import _SAMPLING_METHOD as _SAMPLING_METHOD
from trex.evidence.nested_controls import _STOP_REASON_EVIDENCE_TOLERANCE as _STOP_REASON_EVIDENCE_TOLERANCE
from trex.evidence.nested_controls import _STOP_REASON_LIKELIHOOD_PLATEAU as _STOP_REASON_LIKELIHOOD_PLATEAU
from trex.evidence.nested_controls import _STOP_REASON_TRUNCATED as _STOP_REASON_TRUNCATED
from trex.evidence.nested_controls import _STOP_REASON_UNKNOWN as _STOP_REASON_UNKNOWN
from trex.evidence.nested_controls import _STOP_REASON_UNRESOLVED as _STOP_REASON_UNRESOLVED
from trex.evidence.nested_controls import _SamplerControls as _SamplerControls
from trex.evidence.nested_controls import _validate_scalar as _validate_scalar
from trex.evidence.nested_controls import _validated_controls as _validated_controls
from trex.evidence.nested_controls import _validated_light_curve as _validated_light_curve
from trex.evidence.nested_sampling import _LikelihoodBudgetExhausted as _LikelihoodBudgetExhausted
from trex.evidence.nested_sampling import _MonitoredLikelihood as _MonitoredLikelihood
from trex.evidence.nested_sampling import _base_run_ended_on_plateau as _base_run_ended_on_plateau
from trex.evidence.nested_sampling import _base_run_remaining_evidence as _base_run_remaining_evidence
from trex.evidence.nested_sampling import _monitored_likelihood_count as _monitored_likelihood_count
from trex.evidence.nested_sampling import _run_nested_sampler as _run_nested_sampler
from trex.evidence.nested_sampling import _worker_initializer as _worker_initializer
from trex.evidence.nested_sampling import _worker_pool as _worker_pool
from trex.evidence.nested_sampling import _cached_importance_block as _cached_importance_block
from trex.evidence.nested_importance import _LogLikelihoodBlock as _LogLikelihoodBlock
from trex.evidence.nested_importance import _run_importance_sampler as _run_importance_sampler
from trex.evidence.nested_population import UnresolvedPopulation as UnresolvedPopulation
from trex.evidence.nested_population import UnresolvedPopulationDraw as UnresolvedPopulationDraw
from trex.evidence.nested_population import _BackgroundStarPopulation as _BackgroundStarPopulation
from trex.evidence.nested_population import _BoundCompanionPopulation as _BoundCompanionPopulation
from trex.evidence.nested_population import _angular_constraint as _angular_constraint
from trex.evidence.nested_population import background_star_population as background_star_population
from trex.evidence.nested_population import bound_companion_population as bound_companion_population
from trex.evidence.nested_integrands import _EclipsingPairIntegrand as _EclipsingPairIntegrand
from trex.evidence.nested_integrands import _TransitingPlanetIntegrand as _TransitingPlanetIntegrand
from trex.evidence.nested_integrands import _hosted_draw as _hosted_draw
from trex.evidence.nested_integrands import _reject_out_of_prior as _reject_out_of_prior
from trex.evidence.nested_integrands import _resolved_dilution as _resolved_dilution
from trex.evidence.nested_integrands import _transit_band_low as _transit_band_low

# The shared evaluation counter of the run in progress.  Workers receive it as
# a pool initializer argument; a serial run sets it directly.  Runs are
# sequential in this project, so one module-level slot is sufficient and keeps
# the pickled likelihood wrapper free of multiprocessing objects.
_monitored_likelihood_counter = None


def _set_likelihood_counter(counter) -> None:
    global _monitored_likelihood_counter
    _monitored_likelihood_counter = counter


def _selected_nested_estimator():
    """Validate the opt-in before dispatch or completed-result reuse."""
    estimator = os.environ.get(_NESTED_ESTIMATOR_ENV, "dynesty")
    if estimator not in {"dynesty", "blackjax"}:
        raise ValueError("unknown TREX nested estimator {0!r}; select dynesty or blackjax".format(estimator))
    return estimator


# A private sentinel distinguishes an omitted live count or call ceiling from
# one the caller supplied, so only the default follows the selected estimator.
_UNSET = object()


def _default_nested_controls(n_live_points, max_likelihood_calls, n_dimensions):
    """Resolve omitted controls for the selected estimator; explicit values pass through.

    BlackJAX's only validated configuration is the calibrated 3,200-live
    setting, with its call ceiling derived from this run's live count and
    dimension.  Dynesty keeps its 400-live default and no explicit call cap.
    An explicit live count is a caller-owned budget: it is passed through
    unchanged, so an omitted ceiling keeps its existing refusal.
    """
    if n_live_points is not _UNSET:
        return n_live_points, None if max_likelihood_calls is _UNSET else max_likelihood_calls
    if _selected_nested_estimator() == "blackjax":
        if max_likelihood_calls is _UNSET:
            max_likelihood_calls = _blackjax_call_ceiling(
                _BLACKJAX_CALIBRATED_LIVE_POINTS,
                _BLACKJAX_CALIBRATED_LIVE_POINTS // 2,
                _NATIVE_INNER_STEPS_PER_DIMENSION * int(n_dimensions),
            )
        return _BLACKJAX_CALIBRATED_LIVE_POINTS, max_likelihood_calls
    return _DEFAULT_LIVE_POINTS, None if max_likelihood_calls is _UNSET else max_likelihood_calls


def _integrate(integrand, controls, scenario, n_dimensions, importance_posterior):
    """One nested run, or the importance refinement over ``importance_posterior``.

    Both integrate the identical unit-cube integrand, so a scenario has exactly
    one prior, likelihood and support whichever estimator is asked for.
    """
    estimator = _selected_nested_estimator()
    if importance_posterior is None:
        if estimator == "blackjax":
            if not callable(getattr(integrand, "jax_unit_integrand", None)):
                raise TrexEvidenceUnavailableError("this observation model has no native BlackJAX integrand; use dynesty")
            return _run_native_nested_sampler(integrand, controls, n_dimensions)
        return _run_nested_sampler(
            integrand.prior_transform, integrand.log_likelihood, controls, scenario, n_dimensions
        )
    acceleration = _native_refinement_integrand(integrand)
    return _run_importance_sampler(
        integrand.prior_transform,
        integrand.log_likelihood,
        importance_posterior,
        n_dimensions=n_dimensions,
        random_seed=controls.random_seed,
        n_jobs=controls.n_jobs,
        native_integrand=acceleration,
        cpu_native_integrand=None if acceleration is not None else _cpu_refinement_integrand(integrand),
        proposal=_selected_importance_proposal(),
    )


def _run_native_nested_sampler(integrand, controls, n_dimensions, *, batched=True):
    """One explicitly selected BlackJAX NSS run of the identical integrand.

    Unsupported native contracts raise rather than changing estimator or
    silently selecting CPU execution. Supplied host properties must be
    propagated by scenario callers, not fabricated here. The
    native result is ``evidence_calibrated=False`` until known-Z calibration.
    An active recovery context requires an owned directory and checks its
    deadline before native setup, as the dynesty path does.
    """
    runtime = _checkpoint_context.get()
    prefix, deadline = runtime if runtime is not None else (None, math.inf)
    if runtime is not None:
        if prefix is None:
            raise RuntimeError("resumable escalation requires a candidate-owned checkpoint directory")
        if _monotonic() >= deadline:
            raise NestedCheckpointPaused("nested wall deadline reached before sampling")
    from trex.accelerators import _jax_models as models

    policy = _native_kernel_policy()
    if not models.HAS_JAX_TREX:
        raise RuntimeError("the BlackJAX nested estimator requires JAX float64 support")
    if controls.max_likelihood_calls is None:
        raise ValueError("the BlackJAX nested estimator requires an explicit likelihood ceiling")
    # Hosted atmospheres and eclipsed secondaries require scene metallicity.
    # PTP/DTP leave the target as host and consume its supplied u1/u2; their
    # companion populations only dilute, so metallicity is not a dependency.
    fixed_host_population_tp = (isinstance(integrand, _TransitingPlanetIntegrand)
                               and isinstance(integrand.population, (_BoundCompanionPopulation, _BackgroundStarPopulation))
                               and not integrand.hosted)
    if (integrand.population is not None and integrand.target_feh is None
            and not fixed_host_population_tp
            and not (integrand.hosted and isinstance(integrand.population, _BackgroundStarPopulation))):
        raise ValueError("the BlackJAX population-host contract requires target_feh")
    configured = models.jax.config.jax_default_device
    if configured is not None:
        device = configured
    else:
        accelerators = models._usable_jax_devices()
        if accelerators:
            device = accelerators[0]
        elif os.environ.get("JAX_PLATFORMS") == "cpu":
            device = models.jax.devices("cpu")[0]
        else:
            raise RuntimeError("BlackJAX has no admitted accelerator; CPU execution requires explicit CPU placement")
    native = integrand.jax_unit_integrand(device=device)
    kernel = native_nested_kernel(
        native, n_dimensions=int(n_dimensions), n_live_points=controls.n_live_points,
        num_delete=controls.n_live_points // 2,
        num_inner_steps=_NATIVE_INNER_STEPS_PER_DIMENSION * int(n_dimensions),
        batched=batched, **policy,
    )
    # Prior positions and the slice key are separate streams of the one run seed.
    positions = np.random.default_rng(controls.random_seed).uniform(size=kernel.shape)
    recovery = ({} if prefix is None else {
        "checkpoint": prefix.with_suffix(".blackjax.npz"),
        "checkpoint_identity": _checkpoint_identity((integrand, controls._replace(n_jobs=1), batched, "blackjax-1.7.1")),
        "deadline": deadline,
    })
    return kernel.integrate(
        positions, models.jax.random.PRNGKey(controls.random_seed),
        evidence_tolerance=controls.evidence_tolerance,
        max_likelihood_calls=int(controls.max_likelihood_calls),
        random_seed=controls.random_seed,
        **recovery,
    )


def _refinement_native_capable(integrand):
    """Whether the scenario declares a float64 JAX unit-cube callback."""
    from trex.accelerators import _jax_models as models
    return models.HAS_JAX_TREX and callable(getattr(integrand, "jax_unit_integrand", None))


def _native_refinement_integrand(integrand):
    """Use native callbacks on admitted accelerators, honoring explicit CPU placement.

    This selects only likelihood execution for the unchanged defensive
    estimator; it does not select BlackJAX evidence or bypass its calibration.
    Missing optional acceleration retains the declared scalar CPU path. A
    selected native callback's numerical/loader failure must propagate.
    """
    from trex.accelerators import _jax_models as models

    if not _refinement_native_capable(integrand):
        return None
    configured = models.jax.config.jax_default_device
    if configured is not None:
        return integrand if configured.platform not in ("cpu", "interpreter") else None
    return integrand if models._usable_jax_devices() else None


def _cpu_refinement_integrand(integrand):
    """The same native callback for batched float64 execution on the CPU device.

    One integrand model then serves an accelerator host and a CPU host, so the
    refinement estimate is comparable across them; scenarios without a native
    callback keep the scalar reference path.
    """
    return integrand if _refinement_native_capable(integrand) else None


def _with_occurrence_weight(
    result: NestedEvidenceResult, population: "Optional[UnresolvedPopulation]"
) -> NestedEvidenceResult:
    """Add the population's constant occurrence rate to a converged evidence.

    The production estimator carries the rate inside its expectation; it is
    constant on the retained set and ``-inf`` off it, so the shift is exact and
    leaves the sampler's own ``log_evidence_error`` unchanged.
    """
    if population is None:
        return result
    return result._replace(
        log_evidence=result.log_evidence + float(population.log_occurrence_weight)
    )


def tp_scenario_nested_evidence(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    exptime_days: float,
    *,
    target_aperture_flux_fraction: float,
    use_flat_priors: bool = False,
    population: Optional[UnresolvedPopulation] = None,
    hosted_population: Optional[UnresolvedPopulation] = None,
    target_feh: Optional[float] = None,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    n_live_points: "int | object" = _UNSET,
    evidence_tolerance: float = _DEFAULT_EVIDENCE_TOLERANCE,
    random_seed: int,
    max_likelihood_calls: "int | None | object" = _UNSET,
    n_jobs: int | None = None,
    importance_posterior: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    host_teff_k: Optional[float] = None,
    stellar_light=None,
    baseline=None,
    host_distribution=None,
    joint_observations=None,
) -> "NestedEvidenceResult | ImportanceEvidenceResult":
    """Estimate a transiting-planet scenario's log evidence by nested sampling.

    The likelihood is the production :func:`lnL_TP` on the production
    scenario priors, evaluated on the light curve renormalized onto the given
    host's share of the aperture flux (the same ``renorm_flux_to_host`` step
    ``calc_target_evidences`` performs), so the scenario integral is the one
    the Monte Carlo path estimates. ``use_flat_priors`` explicitly overrides
    the shared default broken planet-radius prior.
    ``nsamples`` and ``bin_integration_days`` select the same
    exposure-integration model.

    ``population`` attaches an unresolved diluting source -- the
    bound-companion (``PTP``) or background-star (``DTP``) slot -- as
    extra prior dimensions.  The target remains the host; the drawn
    source only dilutes, exactly as ``_eval_scenario`` dilutes with
    ``companion_is_host=False``, and its occurrence rate is added to the
    returned log evidence.

    ``hosted_population`` selects the population-*hosted* slot instead
    (``STP``/``BTP``): the drawn source is the host star, the target's flux
    dilutes the event, and the radius prior, orbit geometry, and host
    atmosphere are resolved per draw exactly as ``_eval_scenario`` resolves
    the ``S*``/``B*`` slots. S* requires ``target_feh``; B* uses each simulated
    star's retained TRILEGAL [M/H] in the grid's scaled-solar abundance coordinate and is
    mutually exclusive with ``population``.

    ``max_likelihood_calls``, when supplied, bounds the run's likelihood
    evaluations; if it binds before the tolerance is reached, the run is
    truncated or fails and cannot be accepted as converged.

    When omitted, ``n_live_points`` and ``max_likelihood_calls`` follow the
    selected estimator: the calibrated 3,200-live BlackJAX setting with the
    ceiling derived from this run's live count and dimension, or dynesty's
    400-live default with no explicit call cap.  Supplying either value
    overrides only that control.

    ``n_jobs`` evaluates the sampler's point proposals in that many worker
    processes; ``None`` takes the shared hardware profile's count
    (``hardware.get_default_n_jobs``), which is what every CPU-bound path
    in this project uses.  Above one,  The integral is unchanged, but the sampler
    consumes its random stream in a different order, so a parallel run is
    not bit-identical to a serial one at the same seed -- it is another
    draw from the same estimator, and its agreement with the serial run is
    bounded by the same run-to-run scatter as any other pair of seeds.

    ``importance_posterior`` -- ``(unit_samples, weights)`` of independent
    nested runs of this same scenario -- replaces the nested run by the
    importance refinement of :func:`_run_importance_sampler` over the identical
    integrand, seeded by ``random_seed``, and returns an
    :class:`ImportanceEvidenceResult`.
    """
    if population is not None and hosted_population is not None:
        raise ValueError(
            "a TP scenario integrates either a diluting population or a "
            "hosted population, not both"
        )
    if joint_observations is not None and baseline is not None:
        raise ValueError("joint observations already own their baseline priors")
    active_population = population if hosted_population is None else hosted_population
    hosted = hosted_population is not None
    if target_feh is not None:
        target_feh = _validate_scalar(target_feh, "target_feh", positive=False)
    if hosted and not isinstance(active_population, _BackgroundStarPopulation) and target_feh is None:
        raise ValueError("population-hosted TP scenarios require candidate target_feh")
    period_days = _validate_scalar(period_days, "period_days", positive=True)
    host_mass = _validate_scalar(M_s_Msun, "M_s_Msun", positive=True)
    host_radius = _validate_scalar(R_s_Rsun, "R_s_Rsun", positive=True)
    if joint_observations is None:
        exptime = _validate_scalar(exptime_days, "exptime_days", positive=True)
    elif exptime_days is not None or np.any(np.asarray(bin_integration_days) != 0):
        raise ValueError("joint observations own their exposure; no legacy integration is allowed")
    else:
        exptime = None
    bin_days = phase_bin_widths(bin_integration_days, np.asarray(time).size)
    u1_value = _validate_scalar(u1, "u1", positive=False)
    u2_value = _validate_scalar(u2, "u2", positive=False)
    n_dimensions = _HOSTED_N_DIMENSIONS + (
        0 if active_population is None else int(active_population.n_dimensions)
    )
    sampler_dimensions = n_dimensions + (
        0 if host_distribution is None else HostMeasure(host_distribution, target_feh).n_dimensions
    )
    n_live_points, max_likelihood_calls = _default_nested_controls(
        n_live_points, max_likelihood_calls, sampler_dimensions
    )
    controls = _validated_controls(
        n_live_points, evidence_tolerance, max_likelihood_calls, random_seed,
        sampler_dimensions, n_jobs,
    )
    time_values, host_flux, host_sigma = _validated_light_curve(
        time, flux, sigma, target_aperture_flux_fraction
    )

    integrand = _TransitingPlanetIntegrand(
        time=time_values,
        flux=host_flux,
        sigma=host_sigma,
        period_days=period_days,
        host_mass=host_mass,
        host_radius=host_radius,
        u1=u1_value,
        u2=u2_value,
        exptime_days=exptime,
        bin_integration_days=bin_days,
        nsamples=nsamples,
        use_flat_priors=use_flat_priors,
        population=active_population,
        n_dimensions=n_dimensions,
        hosted=hosted,
        target_feh=target_feh,
        host_teff_k=host_teff_k,
        stellar_light=stellar_light,
        baseline_likelihood=(None if baseline is None else baseline.prepare(time, flux, sigma, bin_integration_days).for_host(target_aperture_flux_fraction)),
        joint_observations=joint_observations,
    )
    if joint_observations is not None:
        # Joint data own a per-dataset band/share/exposure likelihood that a
        # single-band callback cannot represent. The device twin is attached
        # only for the fixed-host family; population/characterized-host
        # configurations keep the scalar joint integrand explicitly.
        from trex.accelerators._jax_joint_unit_integrand import joint_planet_callback

        integrand.jax_unit_integrand = joint_planet_callback(integrand)
    if host_distribution is not None:
        from trex.evidence.host_measure import CharacterizedHostIntegrand
        integrand = CharacterizedHostIntegrand(integrand, host_distribution)
        n_dimensions = integrand.n_dimensions
        if joint_observations is not None:
            # A characterized host changes the per-band atmosphere per draw;
            # no native joint callback represents that yet.
            integrand.jax_unit_integrand = None
    return _with_occurrence_weight(
        _integrate(integrand, controls, "TP", n_dimensions, importance_posterior),
        active_population,
    )


def eb_scenario_nested_evidence(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    exptime_days: float,
    *,
    target_aperture_flux_fraction: float,
    use_2xP: bool = False,
    population: Optional[UnresolvedPopulation] = None,
    hosted_population: Optional[UnresolvedPopulation] = None,
    target_feh: Optional[float] = None,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    n_live_points: "int | object" = _UNSET,
    evidence_tolerance: float = _DEFAULT_EVIDENCE_TOLERANCE,
    random_seed: int,
    max_likelihood_calls: "int | None | object" = _UNSET,
    n_jobs: int | None = None,
    importance_posterior: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    host_teff_k: Optional[float] = None,
    stellar_light=None,
    baseline=None,
    host_distribution=None,
    joint_observations=None,
) -> "NestedEvidenceResult | ImportanceEvidenceResult":
    """Estimate an eclipsing-pair scenario's log evidence by nested sampling.

    The second parameter is the Moe & Di Stefano (2017) mass-ratio variate
    that ``sample_q_host_mass_ratio`` draws for this host; the secondary
    radius, its TESS-band flux ratio, and the pair's total mass follow per
    draw from the retained stellar relations, exactly as ``_eval_scenario``
    computes them.  The likelihood is the production :func:`lnL_EB` (primary
    and secondary eclipses) on the host-renormalized light curve.
    ``target_feh`` is required except for background-hosted EB slots: the eclipsed secondary's
    own temperature and gravity select its TESS grid limb darkening, using
    the same composition as its binary host (TRILEGAL [M/H] for B*, target [Fe/H] otherwise).
    ``use_2xP`` selects the ``EBx2P`` slot and its twin mass-ratio support;
    the full-q prior is not renormalized within either branch (G21 §2.4.4).

    ``population`` attaches an unresolved diluting source -- the
    bound-companion (``PEB``/``PEBx2P``) or background-star
    (``DEB``/``DEBx2P``) slot -- as extra prior dimensions.  The pair
    still orbits the given host; the drawn source only dilutes, and its
    occurrence rate is added to the returned log evidence.

    ``hosted_population`` selects the population-*hosted* slot instead
    (``SEB``/``SEBx2P``, ``BEB``/``BEBx2P``): the drawn source is the
    eclipsing pair's primary, and its mass, radius, temperature, mass-ratio
    support floor, orbit geometry, secondary stellar relations, and host
    atmosphere are resolved per draw exactly as ``_eval_scenario`` resolves
    the ``S*``/``B*`` slots, with the target's flux diluting the event.  It
    is mutually exclusive with ``population``.

    ``n_live_points`` and ``max_likelihood_calls`` follow the selected
    estimator exactly as in :func:`tp_scenario_nested_evidence`.

    ``importance_posterior`` selects the importance refinement exactly as in
    :func:`tp_scenario_nested_evidence`.
    """
    # Supplied fixed-host Teff [K] belongs to component fluxes; hosted draws retain their own atmosphere.
    if joint_observations is not None and baseline is not None:
        raise ValueError("joint observations already own their baseline priors")
    if population is not None and hosted_population is not None:
        raise ValueError(
            "an EB scenario integrates either a diluting population or a "
            "hosted population, not both"
        )
    active_population = population if hosted_population is None else hosted_population
    hosted = hosted_population is not None
    if target_feh is not None:
        target_feh = _validate_scalar(target_feh, "target_feh", positive=False)
    if not (hosted and isinstance(active_population, _BackgroundStarPopulation)) and target_feh is None:
        raise ValueError("EB secondary limb darkening requires candidate target_feh")
    period_days = _validate_scalar(period_days, "period_days", positive=True)
    host_mass = _validate_scalar(M_s_Msun, "M_s_Msun", positive=True)
    host_radius = _validate_scalar(R_s_Rsun, "R_s_Rsun", positive=True)
    if joint_observations is None:
        exptime = _validate_scalar(exptime_days, "exptime_days", positive=True)
    elif exptime_days is not None or np.any(np.asarray(bin_integration_days) != 0):
        raise ValueError("joint observations own their exposure; no legacy integration is allowed")
    else:
        exptime = None
    bin_days = phase_bin_widths(bin_integration_days, np.asarray(time).size)
    u1_value = _validate_scalar(u1, "u1", positive=False)
    u2_value = _validate_scalar(u2, "u2", positive=False)
    effective_period = period_days * 2.0 if use_2xP else period_days
    if not hosted and host_mass < MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
        raise ValueError(
            "Moe & Di Stefano mass-ratio prior has no valid q support for this primary mass"
        )
    n_dimensions = _HOSTED_N_DIMENSIONS + (
        0 if active_population is None else int(active_population.n_dimensions)
    )
    sampler_dimensions = n_dimensions + (
        0 if host_distribution is None else HostMeasure(host_distribution, target_feh).n_dimensions
    )
    n_live_points, max_likelihood_calls = _default_nested_controls(
        n_live_points, max_likelihood_calls, sampler_dimensions
    )
    controls = _validated_controls(
        n_live_points, evidence_tolerance, max_likelihood_calls, random_seed,
        sampler_dimensions, n_jobs,
    )
    time_values, host_flux, host_sigma = _validated_light_curve(
        time, flux, sigma, target_aperture_flux_fraction
    )

    integrand = _EclipsingPairIntegrand(
        time=time_values,
        flux=host_flux,
        sigma=host_sigma,
        effective_period=effective_period,
        use_2xP=use_2xP,
        host_mass=host_mass,
        host_radius=host_radius,
        u1=u1_value,
        u2=u2_value,
        exptime_days=exptime,
        bin_integration_days=bin_days,
        nsamples=nsamples,
        population=active_population,
        n_dimensions=n_dimensions,
        hosted=hosted,
        target_feh=target_feh,
        host_teff_k=host_teff_k,
        stellar_light=stellar_light,
        baseline_likelihood=(None if baseline is None else baseline.prepare(time, flux, sigma, bin_integration_days).for_host(target_aperture_flux_fraction)),
        joint_observations=joint_observations,
    )
    if joint_observations is not None:
        # Same joint routing rule as the TP path: fixed-host joint scenarios
        # use the device twin, population/hosted ones stay on the scalar
        # joint integrand instead of a single-band callback.
        from trex.accelerators._jax_joint_unit_integrand import joint_eclipsing_pair_callback

        integrand.jax_unit_integrand = joint_eclipsing_pair_callback(integrand)
    if host_distribution is not None:
        from trex.evidence.host_measure import CharacterizedHostIntegrand
        integrand = CharacterizedHostIntegrand(integrand, host_distribution)
        n_dimensions = integrand.n_dimensions
        if joint_observations is not None:
            integrand.jax_unit_integrand = None
    scenario = "EBx2P" if use_2xP else "EB"
    return _with_occurrence_weight(
        _integrate(integrand, controls, scenario, n_dimensions, importance_posterior),
        active_population,
    )


def native_nested_kernel(integrand, *, n_dimensions, n_live_points, num_delete, num_inner_steps, batched=False,
                         batched_initialization=False, initialization_block=64,
                         direction_metric="fixed-isotropic", speculative_expansions=0):
    """Build an experimental checked NSS kernel with optional evidence integration.

    Unit positions are dimensionless and log contours are nats. The adapter
    exposes raw dead/live states and optional evidence estimates; calibration and
    RNG ownership remain explicit caller obligations. Default routing stays dynesty.
    Optional execution policies are validated by ``NativeNestedKernel`` and are
    never a calibration claim.
    """
    from trex.evidence._blackjax_kernel import NativeNestedKernel

    return NativeNestedKernel(integrand, n_dimensions=n_dimensions, n_live_points=n_live_points,
                              num_delete=num_delete, num_inner_steps=num_inner_steps, batched=batched,
                              batched_initialization=batched_initialization,
                              initialization_block=initialization_block,
                              direction_metric=direction_metric,
                              speculative_expansions=speculative_expansions)
