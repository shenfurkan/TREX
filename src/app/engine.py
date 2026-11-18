"""TREX conditional false-positive evidence engine.

Bayesian false-positive probabilities informed by Giacalone et al. (2021, AJ,
161, 24), without a runtime dependency on ``triceratops``. Candidate-bound
aperture, bound-companion and background models differ from that implementation.

Public API
----------
``run_trex_vetting`` consumes a ``TargetScene`` and observed photometry and
returns a ``TrexResult`` with FPP, NFPP, scenario probabilities and diagnostics.

Scientific Guardrail
--------------------
Results retain ``claim_eligible=False``; conditional scenario probabilities
do not supply empirical scene-model calibration.

Units, primary literature, and fail-closed boundary
----------------------------------------------------
The method source is Giacalone et al. (2021), AJ 161, 24,
DOI ``10.3847/1538-3881/abc6af``. ``time`` is
phase-relative days, ``flux`` and ``sigma`` are normalized relative flux,
``period_days`` and ``exptime_days`` are days, ``depth_ppm`` is ppm, stellar
mass/radius are ``M_sun``/``R_sun``, contrast separations are arcsec, contrast
is delta-mag, and FPP/NFPP/scenario probabilities are dimensionless.  A
validated scene, complete retained neighbor/contrast/TRILEGAL inputs, finite
dynamic exposure, and finite observed photometry are mandatory.  Any absent or
invalid boundary is unresolved rather than a false-positive conclusion.
"""

from __future__ import annotations

import copy
import math
import os
import time as _time
from typing import Any as Any, Callable, Mapping, Optional

import numpy as np

from trex.foundation.constants import G, Msun, Rsun
from trex.foundation.environment import read_environment
from trex.models.flux_calibration import validate_flux_uncertainty
from trex.stellar_light.host_limb_darkening import TARGET_HOST_LDC_ROLE, LimbDarkeningNode, load_triceratops_tess_grid
from trex.app.scene import TargetScene
from trex.evidence.diagnostics import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, TrexDiagnostic, TrexResult, generate_diagnostics
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.app.observation_vetting import run_observation_vetting as run_observation_vetting
from trex.evidence.evidence_escalation import evidence_needs_escalation, residual_could_flip_decision, run_nested_escalation
from trex.evidence.nested_evidence import NestedCheckpointPaused, _selected_nested_estimator
from trex.evidence.marginal_likelihoods import _SCENARIO_NAMES, N_NEIGHBOR_SCENARIOS, N_TARGET_SCENARIOS, calc_target_evidences, resolved_neighbor_host
from trex.foundation.physical_constants import FRACTION_PER_PART_PER_MILLION
from trex.foundation._numerics import phase_bin_widths
from trex.evidence.monte_carlo_state import _monte_carlo_state_record as _monte_carlo_state_record, _publish_mc_ldc_provenance, _restore_importance as _restore_importance, _restore_monte_carlo_state as _restore_monte_carlo_state, _summarize_mc_ldc_events
from trex.evidence.mc_importance import make_state as _make_mc_importance_state
from trex.evidence.mc_importance import pilot_seed as _mc_importance_pilot_seed
from trex.evidence.mc_importance import run_mc_importance as _run_mc_importance
from trex.evidence.mc_importance import validate_state as _validate_mc_importance_state
from trex.app.doctor import INPUT_ADMISSION, record_engine_execution
from trex.runtime.logging import host_logger
from trex.populations._validation import real_scalar


# Historical solar-analog quadratic coefficients; automatic target selection
# uses the target's grid node instead. Claret (2017), A&A 600, A30, Table 1,
# DOI 10.1051/0004-6361/201629705, supplies the TESS limb-darkening context.
_TREX_DEFAULT_U1: float = 0.4   # Claret 2017 TESS solar u1 (Kipping q-space safe)
_TREX_DEFAULT_U2: float = 0.2   # Claret 2017 TESS solar u2
# TIER 1 LITERATURE — per-scenario Monte Carlo draws for the marginal
# likelihood p(D | S_j) ~ (1/N) sum_n p(D | theta_j^(n), S_j).
# Giacalone et al. (2021, AJ, 161, 24;
# DOI 10.3847/1538-3881/abc6af), Sec. 2 "Marginal Likelihoods": N is chosen
# "high enough to produce results that are consistent between consecutive
# calculations (which we determine to be N=10^6)". The reference TRICERATOPS
# implementation uses the same N in every ``lnZ_*`` and in
# ``triceratops.calc_probs``. This production budget is not reduced for memory
# pressure; working sets adapt to the host. The engine's
# Kish effective-sample-size gate (``diagnostics``,
# ``MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE``) stays an independent check: a
# scene that is still unconverged at this N defers rather than publishing a
# seed-dependent FPP, and the remedy is a larger budget, never a smaller one.
_TREX_DEFAULT_N_DRAWS: int = 1_000_000
TREX_PRODUCTION_N_DRAWS: int = _TREX_DEFAULT_N_DRAWS  # Public name for callers outside TREX.
_TREX_DEFAULT_NSAMPLES: int = 20    # sub-exposure midpoint nodes per exposure


def _target_surface_gravity_log10_cgs(scene: TargetScene) -> float:
    """Target log surface gravity [cgs] from the scene mass and radius."""
    return math.log10(
        G * scene.M_s_Msun * Msun / (scene.R_s_Rsun * Rsun) ** 2
    )


def target_limb_darkening_node(scene: TargetScene) -> LimbDarkeningNode:
    """Select the target star's own TESS limb-darkening grid node."""
    return load_triceratops_tess_grid().select(
        scene.Teff_K,
        _target_surface_gravity_log10_cgs(scene),
        scene.feh_dex,
    )


def _resolved_target_limb_darkening(
    scene: TargetScene,
    u1: Optional[float],
    u2: Optional[float],
    ldc_provenance: Optional[list],
) -> tuple:
    """Return the target coefficients and record where they came from."""
    if (u1 is None) != (u2 is None):
        raise ValueError(
            "u1 and u2 must be supplied together or left to the target "
            "atmosphere lookup"
        )
    if u1 is not None:
        if ldc_provenance is not None:
            ldc_provenance.append(
                {
                    "role": TARGET_HOST_LDC_ROLE,
                    "source": "caller-supplied",
                    "u1": float(u1),
                    "u2": float(u2),
                    "boundary_clamped": False,
                }
            )
        return float(u1), float(u2)
    logg_cgs = _target_surface_gravity_log10_cgs(scene)
    node = load_triceratops_tess_grid().select(
        scene.Teff_K, logg_cgs, scene.feh_dex
    )
    if ldc_provenance is not None:
        ldc_provenance.append(
            {
                "role": TARGET_HOST_LDC_ROLE,
                "source": node.source,
                "requested_teff_k": float(scene.Teff_K),
                "requested_logg_cgs": float(logg_cgs),
                "requested_metallicity_dex": float(scene.feh_dex),
                "used_teff_k": float(node.used_teff_k),
                "used_logg_cgs": float(node.used_logg_cgs),
                "used_metallicity_dex": float(node.used_metallicity_dex),
                "boundary_clamped": bool(node.boundary_clamped),
                "interpolated": bool(node.interpolated),
                "u1": float(node.u1),
                "u2": float(node.u2),
            }
        )
    return float(node.u1), float(node.u2)


def _input_identity(scene, evidence_inputs, *settings):
    """Bind an MC checkpoint to its inputs, TREX source bytes and atmosphere tables.

    The scene's `background_sha256` binds the population file. Its parsed radii and
    temperatures differ in the last bits between NumPy SIMD builds, and its path is a
    host detail, so neither enters the digest. Identical kits and inputs remain
    comparable across hosts; a source change requires fresh Monte Carlo.
    """
    from trex.evidence.nested_checkpoint import checkpoint_identity

    bound_scene = copy.copy(scene)
    bound_scene.trilegal_cache = None
    if bound_scene.contrast_provenance is not None:
        bound_scene.contrast_provenance = {
            key: value for key, value in bound_scene.contrast_provenance.items() if key not in {"source", "limitation"}
        }
    return checkpoint_identity((
        bound_scene, {key: value for key, value in evidence_inputs.items() if key != "background_population"}, *settings,
    ))


def admitted_photometry(time, flux, sigma):
    """Normalize and validate photometry once for every admission caller.

    Times are phase-relative days, flux is normalized relative flux and sigma is
    a scalar or one positive per-point uncertainty. This is the engine's own
    check; ``target.calc_probs`` and the doctor call it instead of restating it.
    """
    try:
        time, flux = np.asarray(time), np.asarray(flux)
        if time.dtype.kind not in "iuf" or flux.dtype.kind not in "iuf":
            raise ValueError("photometry must contain real numbers, not boolean or text")
        time, flux = time.astype(float, copy=False), flux.astype(float, copy=False)
        if (time.ndim != 1 or time.size < 2 or flux.shape != time.shape
                or not np.isfinite(time).all() or not np.isfinite(flux).all()):
            raise ValueError(
                "photometry requires finite aligned one-dimensional phase-relative times [days] and normalized flux"
            )
        return time, flux, validate_flux_uncertainty(sigma, flux)
    except Exception as error:
        error.trex_boundary = INPUT_ADMISSION
        raise


def _resolved_random_seed(random_seed, escalation_state_in=None) -> int:
    """Resolve one run seed; resumed draws must use the retained generated seed."""
    if random_seed is None:
        if escalation_state_in is None:
            return int(np.random.SeedSequence().entropy)
        if (not isinstance(escalation_state_in, Mapping)
                or type(escalation_state_in.get("random_seed")) is not int):
            raise ValueError("resumed escalation state must retain its generated random seed")
        random_seed = escalation_state_in["random_seed"]
    if (isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer))
            or random_seed < 0):
        raise ValueError("random_seed must be a non-negative integer or None")
    return int(random_seed)


def _joint_native_covers_scene(scene, population, nearby_stars=None,
                               joint_observations=None) -> bool:
    """Whether every joint scenario this scene can produce has a native callback.

    The native joint twins cover the fixed-host TP/EB families and the
    unresolved population slots (bound-companion ``P*``/``S*`` and
    background-star ``D*``/``B*`` through
    ``trex.accelerators._jax_joint_population_integrand``), so a retained
    bound or background population no longer forces the scalar joint route.
    Two configurations still keep it: a resolved neighbour whose retained
    record carries a stellar-parameter distribution is a characterized host
    with no joint device twin, and a background population without a usable
    declared selection would defer its ``D*``/``B*`` slots per scenario
    instead of evaluating them. Admitting a native-only estimator for either
    would defer the run per scenario, so those scenes keep the scalar
    dynesty requirement instead.
    """
    if any(neighbor.get("stellar_parameter_distribution") is not None
           for neighbor in (nearby_stars or ())):
        return False
    if (population is None or scene.background_area_sq_deg is None
            or joint_observations is None):
        return True
    from trex.populations.background import declared_background_selection

    try:
        declared = declared_background_selection(
            population, datasets=joint_observations.population_datasets,
            target_tess_mag=scene.Tmag)
    except ValueError as exc:
        raise TrexEvidenceUnavailableError(
            "joint native admission could not resolve the declared background "
            "selection: " + str(exc)) from exc
    return declared.depth_mag is not None and math.isfinite(declared.depth_mag)


def prepare_vetting_inputs(
    scene, time, flux, sigma, period_days, depth_ppm, exptime_days,
    bin_integration_days=0.0, sector_weights=None, u1=None, u2=None,
    n_draws=_TREX_DEFAULT_N_DRAWS, nsamples=_TREX_DEFAULT_NSAMPLES,
    ldc_provenance=None, escalation_state_in=None, escalation_state_out=None,
    baseline=None, stellar_light=None, joint_observations=None,
):
    """The engine's canonical pre-draw admission, shared by target.calc_probs and the doctor.

    Times/exposure are days, flux/errors relative, depth ppm; no RNG or sampler
    is started here. Returned arrays and resolved scene values stay in memory.
    """
    if not isinstance(scene, TargetScene):
        raise TypeError("scene must be a validated TargetScene")
    if joint_observations is not None:
        from trex.evidence.joint_observations import ScenarioObservations

        if not isinstance(joint_observations, ScenarioObservations):
            raise ValueError("joint_observations must bind a prepared observation likelihood")
        if (exptime_days is not None or baseline is not None or sector_weights
                or np.any(np.asarray(bin_integration_days) != 0)):
            raise ValueError("joint observations own their exposures, apertures and baseline priors")
        if stellar_light is None:
            raise ValueError("joint observations require exact physical band light")
        _selected_nested_estimator()
        joint_observations.bind_scene(scene)
    for name, value in (("n_draws", n_draws), ("nsamples", nsamples)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value, zero_allowed in (("period_days", period_days, False),
                                      ("depth_ppm", depth_ppm, False),
                                      ("exptime_days", exptime_days, False)):
        if name == "exptime_days" and joint_observations is not None:
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value) or value < 0.0 or (value == 0.0 and not zero_allowed)):
            domain = "non-negative" if zero_allowed else "positive"
            raise ValueError(f"{name} must be finite and {domain}")
    time, flux, sigma = admitted_photometry(time, flux, sigma)
    bin_integration_days = phase_bin_widths(bin_integration_days, time.size)
    if not sector_weights and joint_observations is None:
        raise ValueError(
            "observed per-sector inverse-variance weights are required to apply "
            "the calibrated aperture scene"
        )
    if baseline is not None:
        if _selected_nested_estimator() != "dynesty":
            raise TrexEvidenceUnavailableError(
                "native Gaussian baseline requires dynesty; BlackJAX baseline escalation is unsupported"
            )
        if len(sector_weights or {}) != 1:
            raise TrexEvidenceUnavailableError("native baseline needs single-sector scene dilution; no sectors are pruned")
        if not np.array_equal(baseline.time_days, time) or np.any(np.asarray(bin_integration_days) != 0):
            raise ValueError("joint baseline requires its exact unbinned native-time observations")
    if scene.feh_dex is None:
        # Every host atmosphere needs composition; omitting scenarios would bias FPP low.
        raise ValueError("scene.feh_dex is required to evaluate every false-positive scenario")
    # Target-hosted TP/EB and diluted P*/D* need the target atmosphere, not a solar analog.
    if stellar_light is None:
        u1, u2 = _resolved_target_limb_darkening(scene, u1, u2, ldc_provenance)
    else:
        if u1 is not None or u2 is not None:
            raise ValueError("selected physical stellar light cannot be mixed with overridden target coefficients")
        if stellar_light.target_av_mag is None:
            raise TrexEvidenceUnavailableError("selected physical stellar light requires the target's own A_V [mag]")
        component = stellar_light.prepare_component(
            scene.Teff_K, scene.M_s_Msun, scene.R_s_Rsun, scene.feh_dex, stellar_light.target_av_mag)
        u1, u2 = float(component.u1), float(component.u2)
        if ldc_provenance is not None:
            ldc_provenance.append(dict(role=TARGET_HOST_LDC_ROLE, source=stellar_light.model_id,
                                       u1=u1, u2=u2, boundary_clamped=False))
    if escalation_state_in is not None and escalation_state_out is not None:
        raise ValueError("escalation_state_in and escalation_state_out are mutually exclusive")
    if joint_observations is None:
        nearby_stars = scene.neighbor_dicts_for_evidence(sector_weights)
        target_aperture_flux_fraction = scene.target_aperture_flux_fraction(sector_weights)
    else:
        nearby_stars = scene.neighbor_dicts_for_evidence()
        for neighbor in nearby_stars:
            neighbor["aperture_flux_fraction"] = joint_observations.proposal_fraction(neighbor["source_id"])
        target_aperture_flux_fraction = joint_observations.proposal_fraction(scene.target_source_id)
    # Missing capable hosts must refuse before any target-family draws or replay.
    preflight_ldc = []
    if ldc_provenance is not None:
        ldc_provenance[:] = [event for event in ldc_provenance if event.get("stage") != "scene-preflight"]
    try:
        for index, neighbor in enumerate(nearby_stars):
            resolved_neighbor_host(
                neighbor, scene.feh_dex, index, depth_ppm * FRACTION_PER_PART_PER_MILLION,
                ldc_provenance=preflight_ldc if ldc_provenance is not None else None,
                **({"stellar_light": stellar_light} if stellar_light is not None else {}),
                **({"joint_observations": joint_observations} if joint_observations is not None else {}),
            )
    finally:
        if ldc_provenance is not None:
            ldc_provenance.extend({**event, "stage": "scene-preflight"} for event in preflight_ldc)
    population = scene.background_population()
    if stellar_light is not None:
        from trex.stellar_light.inputs import background_extinction
        try:
            background_extinction(population)
        except ValueError as exc:
            raise TrexEvidenceUnavailableError(str(exc)) from exc
    if joint_observations is not None and _selected_nested_estimator() != "dynesty":
        # Only the scenario families the joint parity covers may use a
        # native-only estimator; characterized hosts and an undeclared
        # background selection keep the scalar joint route.
        if not _joint_native_covers_scene(scene, population, nearby_stars=nearby_stars,
                                          joint_observations=joint_observations):
            raise TrexEvidenceUnavailableError(
                "joint observations admit the native nested estimator only when every "
                "scenario this scene can produce has a native joint callback; this scene "
                "retains a characterized resolved host or an undeclared background "
                "selection whose scenarios keep the scalar joint likelihood")
    return (time, flux, sigma, bin_integration_days, u1, u2,
            nearby_stars, target_aperture_flux_fraction, population)


@record_engine_execution
def run_trex_vetting(
    scene: TargetScene,
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    depth_ppm: float,
    exptime_days: float,
    bin_integration_days: float = 0.0,
    sector_weights: Optional[Mapping[int, float]] = None,
    u1: Optional[float] = None,
    u2: Optional[float] = None,
    n_draws: int = _TREX_DEFAULT_N_DRAWS,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    nsamples: int = _TREX_DEFAULT_NSAMPLES,
    ldc_provenance: Optional[list] = None,
    depth_sigma_ppm: Optional[float] = None,
    duration_days: Optional[float] = None,
    duration_sigma_days: Optional[float] = None,
    n_jobs: Optional[int] = None,
    escalation_state_in: Optional[Mapping] = None,
    escalation_state_out: Optional[dict] = None,
    escalation_checkpoint_directory: Optional[str] = None,
    escalation_wall_time_seconds: Optional[float] = None,
    escalation_deadline_monotonic: Optional[float] = None,
    baseline=None,
    stellar_light=None,
    joint_observations=None,
) -> TrexResult:
    """Run a complete TREX false-positive probability calculation.

    Args:
        scene: TargetScene with source-owned stellar and neighbour properties;
            background hosts use their own simulated composition, not target's.
        time: Days from conjunction (chronological native time for baseline).
        flux: Normalised flux.
        sigma: Flux uncertainty: a scalar shared by every point (the
            TRICERATOPS convention) or one propagated value per point.
            Production callers calibrate it once from native OOT photometry
            with ``noise.calibrate_trex_noise`` before phase folding. Every
            MC/JAX/nested scenario uses that same correction; never estimate
            beta on folded times or fit a separate noise scale per scenario.
        period_days: Orbital period [days].
        depth_ppm: Transit depth [ppm].
        u1, u2: Target quadratic coefficients. Unset selects its temperature,
            CGS gravity and metallicity in the packaged TRICERATOPS TESS grid;
            both explicit values override and are recorded. No solar fallback.
            With stellar_light, its own physical products are mandatory instead.
        n_draws: Draws per scenario: Giacalone et al. 2021 N=10^6 in production;
            smaller budgets are for tests only. Working sets never reduce N.
        random_seed: Nonnegative RNG seed. None generates one seed before MC;
            escalation handoffs retain it and resume restores it.
        progress_callback: Optional callable(step, done, total).
        exptime_days: TESS exposure time [days].
        bin_integration_days: Separate phase-bin averaging width [days].
        baseline: Explicit GaussianBaseline on native chronological TDB days
            from conjunction, reported aperture errors and a proper external
            nuisance prior. Raw single-sector input only; no phase-bin box.
        sector_weights: Per-sector inverse-variance weights of the retained
            cadences behind ``flux``. Required: they reduce the scene's
            calibrated per-sector aperture flux fractions onto the same
            multi-sector combination the observed curve carries.
        nsamples: Supersampling factor.
        n_jobs: Worker processes used only when nested escalation is needed.
        stellar_light: Explicit verified-domain StellarLight; None preserves
            historical physics. Own target A_V [mag] is mandatory, unsupported
            atmospheres refuse without a fallback or population-prior cut.
        escalation_state_out: Return pending escalation in this mapping using
            _monte_carlo_state_record; the caller owns durable storage.
        escalation_state_in: Restore that state without repeating MC, checking
            budget, seed, observations and full physical/noise inputs.
        escalation_checkpoint_directory: Trusted caller-owned recovery
            directory; required by the real resumable sampler. Alternatively
            ``TREX_ESCALATION_CHECKPOINT_DIRECTORY`` supplies it.
        escalation_wall_time_seconds: Operational seconds, at most 540 minutes;
            TREX_ESCALATION_WALL_SECONDS or that ceiling by default.
            Never reduces evidence budgets.
        escalation_deadline_monotonic: Earlier host-monotonic deadline [s],
            never a portable checkpoint time.

    Returns:
        TrexResult with FPP, NFPP, per-scenario probabilities, diagnostics and
        the engine's ``input_identity`` for the evaluated inputs.
    """
    admission_started = _time.monotonic()
    try:
        wall_seconds = real_scalar(
            float(read_environment("TREX_ESCALATION_WALL_SECONDS",
                                   "EXONYM_TREX_ESCALATION_WALL_SECONDS") or 540 * 60)
            if escalation_wall_time_seconds is None else escalation_wall_time_seconds,
            "escalation_wall_time_seconds", positive=True,
        )
        if escalation_deadline_monotonic is not None:
            escalation_deadline_monotonic = real_scalar(
                escalation_deadline_monotonic, "escalation_deadline_monotonic")
        prepared = prepare_vetting_inputs(
            scene, time, flux, sigma, period_days, depth_ppm, exptime_days,
            bin_integration_days, sector_weights, u1, u2, n_draws, nsamples,
            ldc_provenance, escalation_state_in, escalation_state_out, baseline, stellar_light,
            **({"joint_observations": joint_observations} if joint_observations is not None else {}))
        random_seed = _resolved_random_seed(random_seed, escalation_state_in)
    except Exception as error:
        error.trex_boundary = INPUT_ADMISSION
        raise
    (time, flux, sigma, bin_integration_days, u1, u2,
     nearby_stars, target_aperture_flux_fraction, population) = prepared
    host_logger("trex.engine").info(
        "TREX inputs admitted: %d cadences, %d neighbors, %d draws per scenario",
        int(time.size), len(nearby_stars), int(n_draws),
        extra={"event": "trex.engine.admission",
               "stage": "resume" if escalation_state_in is not None else "full",
               "phase": INPUT_ADMISSION, "n_draws": int(n_draws),
               "nsamples": int(nsamples),
               "wall_seconds": _time.monotonic() - admission_started})

    # One-draw concentration is an extreme-value statistic, not a converged integral.
    scenario_ess: dict = {}
    scenario_support: dict = {}
    scenario_jax_fallbacks: dict = {}
    scenario_fits: dict = {}

    evidence_inputs = dict(
        time=time,
        flux=flux,
        sigma=sigma,
        period_days=period_days,
        depth_ppm=depth_ppm,
        M_s_Msun=scene.M_s_Msun,
        R_s_Rsun=scene.R_s_Rsun,
        target_teff_k=scene.Teff_K,
        u1=u1,
        u2=u2,
        exptime_days=exptime_days,
        bin_integration_days=bin_integration_days,
        target_aperture_flux_fraction=target_aperture_flux_fraction,
        contrast_separations=scene.contrast_separations,
        contrast_values=scene.contrast_values,
        plx=scene.plx_mas,
        target_tess_mag=scene.Tmag,
        background_population=population,
        background_area_sq_deg=scene.background_area_sq_deg,
        nearby_stars=nearby_stars or None,
        nsamples=nsamples,
        target_feh=scene.feh_dex,
        resolution_limit_arcsec=scene.unresolved_resolution_limit_arcsec,
        **({"baseline": baseline} if baseline is not None else {}),
        **({"stellar_light": stellar_light} if stellar_light is not None else {}),
        **({"joint_observations": joint_observations} if joint_observations is not None else {}),
    )
    input_identity = _input_identity(scene, evidence_inputs, depth_sigma_ppm, duration_days, duration_sigma_days)
    # Refuse another scene's MC checkpoint before building escalation inputs.
    expected_scenarios = N_TARGET_SCENARIOS + (
        len(nearby_stars) * N_NEIGHBOR_SCENARIOS if nearby_stars else 0
    )

    checkpoint_directory = escalation_checkpoint_directory or read_environment(
        "TREX_ESCALATION_CHECKPOINT_DIRECTORY",
        "EXONYM_TREX_ESCALATION_CHECKPOINT_DIRECTORY")
    # After input validation, before Monte Carlo: an escalation that may run for hours must be resumable.
    if not checkpoint_directory:
        raise ValueError(
            "TREX requires escalation_checkpoint_directory or "
            "TREX_ESCALATION_CHECKPOINT_DIRECTORY before Monte Carlo"
        )
    deferred_escalation = False
    importance_state = None
    if escalation_state_in is not None:
        restored_ldc = []
        (
            lnZ,
            scenario_ess,
            scenario_support,
            scenario_jax_fallbacks,
            scenario_fits,
            monte_carlo_seconds,
        ) = _restore_monte_carlo_state(
            escalation_state_in,
            n_draws=n_draws,
            random_seed=random_seed,
            expected_scenarios=expected_scenarios,
            input_identity=input_identity,
            ldc_provenance_out=restored_ldc,
        )
        mc_ldc_provenance = restored_ldc[0]
        importance_state = escalation_state_in.get("mc_importance")
        if importance_state is not None:
            _validate_mc_importance_state(importance_state, input_identity, expected_scenarios, scene.M_s_Msun)
    else:
        mc_ldc_events = []
        monte_carlo_started = _time.monotonic()
        stage_seed = int(np.random.SeedSequence(random_seed, spawn_key=(0x4D434953,)).generate_state(2, dtype=np.uint64)[0])
        centres = {}
        token = _mc_importance_pilot_seed.set(stage_seed if n_draws >= _TREX_DEFAULT_N_DRAWS else None)
        try:
            lnZ, _ = calc_target_evidences(
                **evidence_inputs,
                n_draws=n_draws,
                random_seed=random_seed,
                progress_callback=progress_callback,
                ldc_provenance=mc_ldc_events,
                ess_out=scenario_ess,
                support_state_out=scenario_support,
                jax_fallback_out=scenario_jax_fallbacks,
                representative_fit_out=scenario_fits,
                importance_centres_out=centres,
                target_source_id=(scene.target_source_id if joint_observations is not None else str(scene.tic_id)),
                depth_sigma_ppm=depth_sigma_ppm,
                duration_days=duration_days,
                duration_sigma_days=duration_sigma_days,
            )
        finally:
            _mc_importance_pilot_seed.reset(token)
        monte_carlo_seconds = _time.monotonic() - monte_carlo_started
        mc_ldc_provenance = _summarize_mc_ldc_events(mc_ldc_events)
        if n_draws >= _TREX_DEFAULT_N_DRAWS:
            importance_state = _make_mc_importance_state(stage_seed, centres, input_identity)

    host_logger("trex.engine").info(
        "TREX Monte Carlo phase %s: %d draws per scenario, %.2f s",
        "restored" if escalation_state_in is not None else "finished", int(n_draws),
        float(monte_carlo_seconds),
        extra={"event": "trex.engine.phase", "phase": "monte-carlo",
               "status": "restored" if escalation_state_in is not None else "finished",
               "n_draws": int(n_draws), "wall_seconds": float(monte_carlo_seconds)})

    if ldc_provenance is not None:
        _publish_mc_ldc_provenance(ldc_provenance, mc_ldc_provenance)

    result = generate_diagnostics(
        lnZ,
        verbose=True,
        effective_sample_sizes=scenario_ess,
        scenario_fits=scenario_fits,
        period_days=period_days,
    )
    result.input_identity = input_identity
    # At the production budget a residual whose own estimate could flip C1/C2 under C5's
    # registered factor is escalated too, not only a set failing the aggregate gate.
    gated = result.status == "degenerate_evidence" or (
        n_draws >= _TREX_DEFAULT_N_DRAWS
        and residual_could_flip_decision(lnZ, scenario_ess, period_days=period_days)
    )
    needs_escalation = evidence_needs_escalation(
        lnZ, scenario_ess,
        degenerate=result.status == "degenerate_evidence",
        at_production_budget=n_draws >= _TREX_DEFAULT_N_DRAWS,
        period_days=period_days,
    )
    if escalation_state_in is not None and not needs_escalation:
        # A checkpoint is only produced for a set needing escalation; a
        # state that no longer reproduces one is inconsistent, not escalatable.
        raise ValueError(
            "escalation state does not reproduce a Monte Carlo evidence set "
            "that needs escalation"
        )
    escalation: Optional[dict] = None
    nested_escalation_seconds: Optional[float] = None
    importance_seconds: Optional[float] = None
    if needs_escalation:
        # Create the private run root before the importance subdirectory can create it with default parent permissions.
        os.makedirs(checkpoint_directory, mode=0o700, exist_ok=True)
        escalation_started = _time.monotonic()
        deadline = escalation_started + min(wall_seconds, 540 * 60.0)
        if escalation_deadline_monotonic is not None:
            deadline = min(deadline, float(escalation_deadline_monotonic))
        importance_directory = (os.path.join(checkpoint_directory, "mc-pilot-importance")
                                if checkpoint_directory is not None else None)
        importance = (None if escalation_state_in is None
                      else _restore_importance(escalation_state_in, expected_scenarios))

        def run_stage():
            return _run_mc_importance(
                lnZ=lnZ, ess=scenario_ess, state=importance_state, n_draws=n_draws,
                input_identity=input_identity, n_jobs=n_jobs, checkpoint_directory=importance_directory,
                wall_time_seconds=wall_seconds, deadline_monotonic=deadline, **evidence_inputs,
            )

        if escalation_state_out is not None:
            # The Monte Carlo/escalation split point. The MC-pilot stage uses
            # whatever float64 backend this host admits -- the scalar CPU
            # integrand or an accelerator -- so it runs before the checkpoint;
            # nested sampling follows on the resume host. A pause leaves the
            # stage to that host.
            importance_paused = False
            try:
                importance = run_stage()
            except NestedCheckpointPaused:
                importance = None
                importance_paused = True
            importance_seconds = _time.monotonic() - escalation_started
            escalation_state_out.update(
                _monte_carlo_state_record(
                    lnZ=lnZ,
                    ess=scenario_ess,
                    support_state=scenario_support,
                    jax_fallbacks=scenario_jax_fallbacks,
                    scenario_fits=scenario_fits,
                    n_draws=n_draws,
                    random_seed=random_seed,
                    monte_carlo_seconds=monte_carlo_seconds,
                    input_identity=input_identity,
                    importance_state=importance_state,
                    importance_outcome=importance,
                    mc_ldc_provenance=mc_ldc_provenance,
                )
            )
            if importance_paused:
                escalation_state_out["mc_importance_paused"] = True
            escalation_state_out["session_remaining_seconds"] = max(0.0, deadline - _time.monotonic())
            deferred_escalation = True
        else:
            # At the production budget, re-integrate sub-floor scenarios with an
            # available nested integrator. Proposal-support labels are diagnostic,
            # not eligibility requirements. The substituted set must clear the
            # same aggregate gate without an unresolved nested attempt.
            try:
                if importance is None:
                    importance = run_stage()
                    importance_seconds = _time.monotonic() - escalation_started
                lnZ = importance["lnZ"]
                scenario_ess = importance["ess"]
                nested_started = _time.monotonic()
                escalation = run_nested_escalation(
                    lnZ=lnZ,
                    ess=scenario_ess,
                    support_state=scenario_support,
                    escalation_budget_floor=_TREX_DEFAULT_N_DRAWS,
                    n_draws=n_draws,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    checkpoint_directory=checkpoint_directory,
                    wall_time_seconds=wall_seconds,
                    deadline_monotonic=deadline,
                    **evidence_inputs,
                )
            except NestedCheckpointPaused as exc:
                elapsed = _time.monotonic() - escalation_started
                phase = "mc_importance" if importance is None else "nested_escalation"
                phase_seconds = elapsed if importance is None else _time.monotonic() - nested_started
                exc.diagnostics = {
                    "n_draws": int(n_draws),
                    "effective_sample_sizes": dict(result.effective_sample_sizes),
                    phase: {"paused": True, "wall_seconds": phase_seconds,
                            "checkpoint_directory": str(importance_directory if importance is None else checkpoint_directory),
                            "attempt_costs": getattr(exc, "attempt_costs", [])},
                    "phase_wall_seconds": {"monte_carlo": monte_carlo_seconds, phase: phase_seconds,
                                            **({"mc_importance": importance_seconds}
                                               if importance_seconds is not None else {})},
                }
                raise
            nested_escalation_seconds = _time.monotonic() - nested_started
            if importance_seconds is not None:
                host_logger("trex.engine").info(
                    "TREX MC-pilot importance phase finished: %.2f s", importance_seconds,
                    extra={"event": "trex.engine.phase", "phase": "mc-importance",
                           "status": "finished", "wall_seconds": float(importance_seconds)})
            host_logger("trex.engine").info(
                "TREX nested escalation phase finished: %.2f s", nested_escalation_seconds,
                extra={"event": "trex.engine.phase", "phase": "nested-escalation",
                       "status": "finished", "wall_seconds": float(nested_escalation_seconds)})
            if escalation["escalated"] or importance["records"]:
                lnZ = escalation["lnZ"]
                scenario_ess = escalation["remaining_ess"]
                result = generate_diagnostics(
                    lnZ,
                    verbose=True,
                    effective_sample_sizes=scenario_ess,
                    nested_provenance=escalation["records"],
                    importance_provenance=importance["records"],
                    scenario_fits=scenario_fits,
                    period_days=period_days,
                )
                result.input_identity = input_identity
            # A stage that ran but substituted nothing still carries its per-scenario
            # outcome: a stage that could not refine a scenario must not look like a
            # run where refinement was unnecessary.
            if gated or escalation["escalated"] or importance["records"] or importance["attempts"]:
                result.nested_evidence_escalation = {
                    "attempted_scenarios": list(escalation["attempted_scenarios"]),
                    "escalated": [
                        _SCENARIO_NAMES.get(index, "N{0}".format(index))
                        for index in escalation["escalated"]
                    ],
                    "unmet_reason": escalation["unmet_reason"],
                    "unmet_scenarios": list(escalation["unmet_scenarios"]),
                    "unmet_reasons": dict(escalation.get("unmet_reasons", {})),
                    "numerical_failures": dict(escalation.get("numerical_failures", {})),
                    "unsubstituted_scenarios": list(escalation["unsubstituted_scenarios"]),
                    "unsubstituted_reason": escalation["unsubstituted_reason"],
                    "cap_binding": bool(escalation["cap_binding"]),
                    "random_seed": escalation.get("random_seed"),
                    "seeds": escalation.get("seeds"),
                    "attempt_costs": escalation.get("attempt_costs", []),
                    "cumulative_wall_seconds": escalation.get("cumulative_wall_seconds"),
                    "mc_importance": {"estimator": "mc-pilot-defensive-importance",
                                      "attempts": importance["attempts"]},
                }
    importance_failures = {
        name: attempt["unresolved_numerical_failures"]
        for name, attempt in ((importance or {}).get("attempts", {}) if needs_escalation else {}).items()
        if attempt.get("unresolved_numerical_failures", 0)
    }
    if importance_failures:
        result.status, result.degenerate = "degenerate_evidence", True
        result.fpp = result.nfpp = None
        for entry in result.scenario_fits:
            name = entry.get("diagnostic_key")
            if name in importance_failures:
                entry.setdefault("numerical_failures_by_stage", {})["mc-pilot-importance"] = importance_failures[name]
        result.diagnostics.insert(0, TrexDiagnostic(
            name="scenario-numerical-failure", status="fail",
            message="MC-pilot importance failed numerically: "
                    + ", ".join(f"{name}={count}" for name, count in importance_failures.items()),
        ))

    # The JAX-batch fallback records are diagnostics, not a gate input: they
    # attach whether the run published or deferred, so a report can never imply
    # the accelerator path evaluated draws it did not.
    result.jax_batch_fallbacks = {
        _SCENARIO_NAMES.get(index, "N{0}".format(index)): dict(record)
        for index, record in scenario_jax_fallbacks.items()
    }
    # Wall time per engine phase, in seconds. A split run carries the Monte
    # Carlo measurement in its checkpoint, so the accelerator share of a
    # resumed escalation stays measurable on the host that runs it.
    result.phase_wall_seconds = {
        "monte_carlo": (
            float(monte_carlo_seconds) if monte_carlo_seconds is not None else None
        ),
        "nested_escalation": (
            float(nested_escalation_seconds)
            if nested_escalation_seconds is not None
            else None
        ),
        "nested_escalation_cumulative": (
            result.nested_evidence_escalation.get("cumulative_wall_seconds")
            if result.nested_evidence_escalation else None
        ),
    }
    if importance_seconds is not None:
        result.phase_wall_seconds["mc_importance"] = float(importance_seconds)
    result.fit_evaluation = {
        **({"stellar_light": stellar_light.model_configuration()} if stellar_light is not None else {}),
        **({"noise_model": baseline.provenance()} if baseline is not None else {}),
        "exptime_days": None if exptime_days is None else float(exptime_days),
        "bin_integration_days": (bin_integration_days.tolist() if isinstance(bin_integration_days, np.ndarray)
                                 else float(bin_integration_days)),
        "nsamples": int(nsamples),
        "time_domain": "native-chronological-TDB-days-from-conjunction" if baseline is not None else "phase-relative-days-from-transit-midpoint",
        "flux_unit": "normalized-relative-aperture-flux",
        "representative_origin": "monte-carlo-max-lnL; not a posterior or nested best fit",
        **({"joint_observations": joint_observations.provenance()} if joint_observations is not None else {}),
    }
    if deferred_escalation:
        return result
    if result.status == "degenerate_evidence" or (
        escalation is not None and escalation["unmet_scenarios"]
    ):
        # An unresolved nested attempt cannot be excused by later substitutions
        # diluting its estimated probability below the aggregate allowance.
        # Honest inability rather than a number: the evidence set failed its
        # aggregate gate or an attempted nested integration remains unresolved.
        # Raising here routes the stage through the existing unresolved-evidence
        # boundary, which records the failure and defers, instead of writing a
        # seed-dependent FPP as a measurement.
        detail = next(
            (
                diagnostic.message
                for diagnostic in result.diagnostics
                if diagnostic.name in ("scenario-input-unavailable", "scenario-numerical-failure",
                                       "evidence-effective-sample-size")
            ),
            "the scenario evidences did not reach the registered effective-sample-size floor",
        )
        if escalation is not None and escalation["unmet_scenarios"]:
            detail = escalation["unmet_reason"] or detail
        # The convergence record travels with the failure. A prose message
        # alone loses the per-scenario table, which is the measurement every
        # estimator and future budget decisions need; the caller persists it
        # into the report on deferral.
        unconverged = {
            name: ess
            for name, ess in result.effective_sample_sizes.items()
            if not math.isfinite(ess)
            or ess < MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
        }
        unconverged_mass = next(
            (
                diagnostic.value
                for diagnostic in result.diagnostics
                if diagnostic.name == "evidence-effective-sample-size"
            ),
            None,
        )
        raise TrexEvidenceUnavailableError(
            "TREX scenario evidence did not converge at n_draws={0}: {1}".format(n_draws, detail),
            diagnostics={
                "n_draws": int(n_draws),
                "effective_sample_sizes": dict(result.effective_sample_sizes),
                "minimum_floor": MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE,
                "unconverged_scenarios": dict(
                    sorted(unconverged.items(), key=lambda item: item[1])
                ),
                "unconverged_probability_mass": unconverged_mass,
                "numerical_failures": {
                    entry["diagnostic_key"]: {
                        "count": entry.get("unresolved_numerical_failures", 0)
                                 + sum(entry.get("numerical_failures_by_stage", {}).values()),
                        "stages": {"monte-carlo": entry.get("unresolved_numerical_failures", 0),
                                   **entry.get("numerical_failures_by_stage", {})},
                        "examples": entry.get("numerical_failure_examples", []),
                    }
                    for entry in result.scenario_fits
                    if entry.get("unresolved_numerical_failures", 0) or entry.get("numerical_failures_by_stage")
                },
                "nested_escalation": result.nested_evidence_escalation,
                "supplemental_nested_evidence": result.supplemental_nested_evidence,
                "jax_batch_fallbacks": dict(result.jax_batch_fallbacks),
                "phase_wall_seconds": dict(result.phase_wall_seconds),
            },
        )
    return result


def scenario_monte_carlo_evidence(result: TrexResult) -> list[dict[str, Any]]:
    """Return each scenario's Monte Carlo log evidence, its error and its ESS.

    The evidence is the first-stage Monte Carlo value the engine normalized
    (the neighbour occurrence prior is added where the engine added it), so a
    scenario later refined by the nested or defensive-importance stage still
    retains the Monte Carlo entry it replaced. The error is ``1/sqrt(ESS)``,
    the Monte Carlo standard error the escalation policy already uses to compare a refined
    estimate with the first-stage integral. ``result.effective_sample_sizes``
    holds the final ESS, so the first-stage ESS is read from the retained fit.
    An older result without it can use the final map only when that scenario is
    explicitly still Monte Carlo; a refined or unknown origin stays missing.

    A scenario the engine did not evaluate has no effective sample size, so its
    error stays ``None``; its log evidence stays the engine's ``-inf`` and is
    serialized by the writer as a visible string, never zero. Scenario order and
    names come from the engine's own record, not a second mapping.

    A fit record's availability ``status`` and its ``reason`` (an unsupported
    bound-companion scenario, for example) are carried verbatim on the row when
    the record carries them; their absence means the record carried none.

    ``phase_seconds`` is the scenario's measured ``draw``/``wait``/``evaluate``
    wall split, carried when the producing record has one. It says where the
    scenario's time went, not that any phase was efficient.

    ``transiting_draws``, ``finite_likelihood_draws`` and ``proposal_support``
    are the sampling evidence behind a sub-floor ESS: whether the likelihood
    occupied a tiny part of the prior volume, and whether the proposal window
    could reach that prior's support. They describe the sample; they do not
    certify that unsampled support is absent.
    """
    rows: list[dict[str, Any]] = []
    effective_sample_sizes = result.effective_sample_sizes or {}
    for entry in result.scenario_fits or []:
        name = entry.get("diagnostic_key") or entry.get("scenario")
        if not isinstance(name, str) or not name:
            continue
        evidence = entry.get("mc_ln_evidence")
        prior = entry.get("neighbor_log_prior")
        if (
            isinstance(evidence, (int, float))
            and not isinstance(evidence, bool)
            and isinstance(prior, (int, float))
            and not isinstance(prior, bool)
        ):
            evidence = float(evidence) + float(prior)
        ess = entry.get("mc_effective_sample_size")
        if (ess is None and entry.get("evidence_origin") == "monte-carlo"
                and name not in (result.supplemental_nested_evidence or {})):
            ess = effective_sample_sizes.get(name)
        finite_ess = (
            isinstance(ess, (int, float))
            and not isinstance(ess, bool)
            and math.isfinite(float(ess))
            and float(ess) > 0.0
        )
        n_draws = entry.get("n_draws")
        row: dict[str, Any] = {
            "scenario": name,
            "scenario_label": entry.get("scenario"),
            "scenario_index": entry.get("scenario_index"),
            "mc_ln_evidence": (
                float(evidence)
                if isinstance(evidence, (int, float)) and not isinstance(evidence, bool)
                else None
            ),
            "mc_ln_evidence_error": 1.0 / math.sqrt(float(ess)) if finite_ess else None,
            "mc_effective_sample_size": float(ess) if finite_ess else None,
            "n_draws": int(n_draws) if isinstance(n_draws, int) and not isinstance(n_draws, bool) else None,
            "evidence_origin": entry.get("evidence_origin"),
        }
        for field in ("status", "reason", "unresolved_numerical_failures", "numerical_failure_examples",
                      "metallicity_provenance", "metallicity_unavailable_reason",
                      "host_metallicity_dex", "metallicity_treatment", "phase_seconds",
                      "transiting_draws", "finite_likelihood_draws", "proposal_support"):
            if field in entry:
                row[field] = entry[field]
        rows.append(row)
    return rows


def scenario_evidence_refinement(result: TrexResult) -> list[dict[str, Any]]:
    """Return each scenario's Monte Carlo and refined evidence state.

    The Monte Carlo entry is the first-stage estimate the engine normalized
    (see :func:`scenario_monte_carlo_evidence`). ``refined_effective_sample_size``
    is the substituted estimate's ESS when a refinement stage replaced it -- the
    MC-pilot defensive importance stage or the nested router -- and ``None``
    otherwise. A nested ensemble substitution reports no Kish ESS, so its
    refined ESS stays ``None`` with that reason; its evidence and estimator are
    retained in ``supplemental_nested_evidence``. The refined log evidence is not
    repeated here because the nested records store it before their scenario prior,
    while the Monte Carlo rows include it.
    ``refinement_status`` is ``refined`` for a substituted scenario,
    ``unresolved``/``unavailable`` for a stage attempt that kept the Monte
    Carlo entry, and ``not-refined`` when no stage replaced the estimate.
    A deferred run writes its own diagnostics instead of this table; this
    record never turns a missing refinement into a zero.

    ``refinement_reason`` carries the stage's own refusal or the reason the
    scenario was not selected, so a reader can tell a scenario above the
    evidence floor from one the router left outside the decision margin.
    """
    rows: list[dict[str, Any]] = []
    effective_sample_sizes = result.effective_sample_sizes or {}
    attempts = {}
    if result.nested_evidence_escalation:
        attempts = (result.nested_evidence_escalation.get("mc_importance") or {}).get("attempts") or {}
    records = result.supplemental_nested_evidence or {}
    for row in scenario_monte_carlo_evidence(result):
        name = row["scenario"]
        refined = records.get(name)
        attempt = attempts.get(name) or {}
        refined_ess = None
        estimator = None
        if refined is not None:
            status = "refined"
            estimator = refined.get("estimator")
            if estimator == "mc-pilot-defensive-importance":
                refined_ess = refined.get("effective_sample_size")
            else:
                importance = refined.get("importance_sampling") or {}
                refined_ess = importance.get("effective_sample_size", effective_sample_sizes.get(name))
            reason = None
            if refined_ess is None:
                reason = "the substituted estimator does not report a Kish effective sample size"
        elif row.get("status") == "unavailable":
            status, reason = "unavailable", row.get("reason")
        elif attempt.get("status") == "accepted":
            # An accepted attempt is always mirrored in the records; keep the
            # attempt's own values if that record is absent.
            status = "refined"
            estimator = attempt.get("estimator")
            refined_ess = attempt.get("effective_sample_size")
            reason = None
        elif attempt:
            status = attempt.get("status") or "unresolved"
            reason = attempt.get("reason")
        else:
            status = "not-refined"
            final_ess = effective_sample_sizes.get(name)
            if isinstance(final_ess, (int, float)) and not isinstance(final_ess, bool) and math.isfinite(float(final_ess)) \
                    and float(final_ess) >= MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE:
                reason = "Monte Carlo effective sample size is at or above the evidence floor"
            elif result.nested_evidence_escalation is None:
                reason = "no evidence refinement stage ran"
            else:
                reason = "not selected by the evidence refinement router"
        rows.append({
            "scenario": name,
            "mc_ln_evidence": row["mc_ln_evidence"],
            "mc_effective_sample_size": row["mc_effective_sample_size"],
            "refined_effective_sample_size": refined_ess,
            "refinement_status": status,
            "refinement_estimator": estimator,
            "refinement_reason": reason,
        })
    return rows


def scenario_status_records(result: TrexResult) -> dict[str, dict[str, Any]]:
    """Per-scenario availability status retained from the fit records.

    Returns one entry per scenario whose fit record carries a ``status`` or
    ``reason`` (the engine records both for an unavailable scenario); ``reason``
    is included only when the record carries one. A scenario's absence here
    means no status was recorded, which evaluated scenarios and older reports
    share; it is not a claim about availability.
    """
    records: dict[str, dict[str, Any]] = {}
    for entry in result.scenario_fits or []:
        name = entry.get("diagnostic_key") or entry.get("scenario")
        if not isinstance(name, str) or not name:
            continue
        present = {key: entry[key] for key in ("status", "reason") if key in entry}
        if present:
            records[name] = present
    return records


def unavailable_scenario_records(result: TrexResult) -> list[dict[str, Any]]:
    """The scenarios a report's FPP excludes because their model is unavailable.

    An excluded scenario keeps its computed probability (zero for the stored
    ``-inf`` evidence) and the reason is the producing prior's own message.
    This is incomplete-model disclosure: an unmodeled population is excluded
    support, never evidence that it is absent.
    """
    return [
        {"scenario": name, "reason": record.get("reason")}
        for name, record in scenario_status_records(result).items()
        if record.get("status") == "unavailable"
    ]


__all__ = [
    "TREX_PRODUCTION_N_DRAWS",
    "run_trex_vetting",
    "run_observation_vetting",
    "scenario_evidence_refinement",
    "scenario_monte_carlo_evidence",
    "scenario_status_records",
    "unavailable_scenario_records",
    "TargetScene",
    "TrexResult",
    "TrexEvidenceUnavailableError",
]
