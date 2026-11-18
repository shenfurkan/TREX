"""Monte Carlo evaluation of target and neighbour scenarios."""
from __future__ import annotations
import math
from functools import partial
from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np
from trex.evidence.nested_controls import _validate_scalar


def _facade(name):
    """Look the name up on ``marginal_likelihoods``, the module tests patch."""
    from trex.evidence import marginal_likelihoods

    return getattr(marginal_likelihoods, name)


def _unevaluated_scenario_record(index, scenario, diagnostic_key, source_id, neighbor_index,
                                 status, is_planet, is_EB, use_2xP, companion_is_host, period_days):
    return {
        "scenario_index": index,
        "scenario": scenario,
        "diagnostic_key": diagnostic_key,
        "source_id": source_id,
        "neighbor_index": neighbor_index,
        "status": status,
        "origin": None,
        "evidence_origin": status,
        "has_representative_draw": False,
        "representative_draw": None,
        "is_planet": bool(is_planet),
        "is_EB": bool(is_EB),
        "use_2xP": bool(use_2xP),
        "period_multiplier": 2 if use_2xP else 1,
        "companion_is_host": companion_is_host,
        "star_num": 2 if companion_is_host else 1,
        "P_orb": float(period_days * 2.0 if use_2xP else period_days),
        "mc_ln_evidence": -math.inf,
        "ln_evidence": -math.inf,
    }


class _NeighborLdcEvents:
    """Attribute events without changing the scenario name used to seed the pilot."""

    def __init__(self, events, index, source_id, scenario):
        self.events = events
        self.original = f"Neighbor {index}"
        self.label = f"neighbor-{index}-source-{source_id}-{scenario}"
        self.scenario = self.label + "-characterized-host"

    def append(self, event):
        name = event["scenario"]
        if name.startswith(self.original):
            event = {**event, "scenario": self.label + name[len(self.original):]}
        self.events.append(event)


def neighbor_metallicity_context(neighbor, target_feh, *, physical_light=False):
    """Describe the actual composition used, including the measured-target fallback."""
    composition = neighbor.get("metallicity_provenance")
    if composition is not None:
        if not isinstance(composition, Mapping):
            raise ValueError("neighbor metallicity_provenance must be a mapping")
        value = composition.get("value_dex")
        value = _validate_scalar(value, "neighbor metallicity", positive=False)
    if physical_light:
        atmosphere = neighbor.get("stellar_atmosphere")
        if not isinstance(atmosphere, Mapping) or not all(
            atmosphere.get(key) is not None for key in ("teff_k", "mh_dex", "av_mag", "source")
        ):
            raise _facade("TrexEvidenceUnavailableError")(
                "resolved source requires its own Teff, [M/H], A_V and provenance; target borrowing is forbidden")
        _validate_scalar(atmosphere["teff_k"], "neighbor atmosphere temperature", positive=True)
        mh = _validate_scalar(atmosphere["mh_dex"], "neighbor atmosphere metallicity", positive=False)
        av = _validate_scalar(atmosphere["av_mag"], "neighbor atmosphere extinction", positive=False)
        if av < 0.0 or not atmosphere["source"]:
            raise ValueError("neighbor atmosphere requires non-negative extinction and source provenance")
        value = float(composition["value_dex"]) if composition is not None else mh
        return {"host_metallicity_dex": value,
                "metallicity_treatment": ("source-associated" if composition is not None
                                          else "source-associated physical atmosphere"),
                "metallicity_provenance": composition,
                "metallicity_unavailable_reason": neighbor.get("metallicity_unavailable_reason")}
    if target_feh is not None:
        target_feh = _validate_scalar(target_feh, "target_feh", positive=False)
    return {
        "host_metallicity_dex": float(composition["value_dex"]) if composition is not None else target_feh,
        "metallicity_treatment": ("source-associated" if composition is not None else
                                  "target-metallicity fallback" if target_feh is not None else
                                  "unavailable (no target metallicity)"),
        "metallicity_provenance": composition,
        "metallicity_unavailable_reason": None if composition is not None else (
            neighbor.get("metallicity_unavailable_reason") or "legacy input did not supply a reason"),
    }


def resolved_neighbor_host(
    neighbor: Mapping[str, Any],
    target_feh: Optional[float],
    index: int,
    observed_depth_fraction: float,
    ldc_provenance: Optional[list] = None,
    *, stellar_light=None, joint_observations=None,
) -> Optional[dict]:
    """Resolve one resolved-neighbor scenario host, or ``None`` when it cannot host.

    The resolved-neighbor evidence path and the nested-evidence escalation path
    must agree on which neighbors can host the observed event and with what
    stellar parameters; this function makes that decision. ``None`` leaves the
    neighbor's three scenario slots at ``-inf``: zero in-aperture flux or an
    eclipse deeper than the host's flux are physical exclusions, while missing
    usable metallicity on the reference-grid path is not. A neighbor that can
    host the depth but has no retained mass and radius raises instead.

    Returns a mapping with per-host mass [Msun], radius [Rsun], limb-darkening
    coefficients, aperture flux fraction, and the bounded spatial log-prior.
    Selected physical light additionally requires this source's own
    stellar_atmosphere (teff_k [K], mh_dex [dex], av_mag [mag], source);
    the historical target-composition approximation is never used in that path.
    """
    if "aperture_flux_fraction" not in neighbor:
        raise ValueError(
            "resolved neighbor {0} has no calibrated aperture flux fraction".format(index)
        )
    neighbor_fraction = _validate_scalar(neighbor["aperture_flux_fraction"], "neighbor aperture flux fraction", positive=False)
    if (
        not math.isfinite(neighbor_fraction)
        or neighbor_fraction < 0.0
        or neighbor_fraction > 1.0
    ):
        raise ValueError(
            "resolved neighbor {0} has an invalid aperture flux fraction".format(index)
        )
    if joint_observations is None:
        if neighbor_fraction == 0.0 or observed_depth_fraction > neighbor_fraction:
            return None
    else:
        neighbor_fraction = joint_observations.proposal_fraction(neighbor.get("source_id", index))
    composition_context = neighbor_metallicity_context(neighbor, target_feh, physical_light=stellar_light is not None)
    neighbor_feh = composition_context["host_metallicity_dex"]
    if stellar_light is None and (neighbor_feh is None or not math.isfinite(float(neighbor_feh))):
        return None
    if neighbor.get("M_s") is None or neighbor.get("R_s") is None:
        raise _facade("TrexEvidenceUnavailableError")(
            "resolved neighbor {0} can host the observed depth but has no stellar "
            "mass and radius: {1}. Supply source-associated mass [solMass] and radius "
            "[solRad] through an applicable validated stellar-data source; do not remove "
            "this possible host or substitute solar values.".format(
                neighbor.get("source_id", index),
                neighbor.get("stellar_parameter_unavailable_reason", "not supplied"),
            ),
            diagnostics={"input_requirements": [{
                "module": "trex.resolved-neighbor",
                "source_id": str(neighbor.get("source_id", index)),
                "reason": neighbor.get("stellar_parameter_unavailable_reason") or "not supplied",
                "required_quantities": [
                    {"name": name, "unit": unit} for key, name, unit in (
                        ("M_s", "mass_solar", "solMass"), ("R_s", "radius_solar", "solRad"),
                    ) if neighbor.get(key) is None
                ],
                "observed_depth_fraction": float(observed_depth_fraction),
                "aperture_flux_fraction": neighbor_fraction,
                "uncertainty_treatment": (
                    "Host mass and radius are fixed scenario inputs; their uncertainties are not marginalized."
                ),
            }]},
        )
    M_n = _validate_scalar(neighbor["M_s"], "neighbor mass", positive=True)
    R_n = _validate_scalar(neighbor["R_s"], "neighbor radius", positive=True)
    physical = {}
    if stellar_light is None:
        distribution = neighbor.get("stellar_parameter_distribution")
        if distribution is None:
            neighbor_teff_k = _facade("stellar_relations")(np.array([M_n]))[1]
        else:
            from trex.evidence.host_measure import HostMeasure
            measure = HostMeasure(distribution, neighbor_feh)
            neighbor_teff_k = np.array([measure.marginal_median("teff_k")])
        host_u1_n, host_u2_n = _facade("_host_grid_coefficients")(
            neighbor_teff_k, np.array([M_n]), np.array([R_n]), neighbor_feh,
            scenario=f"neighbor-{index}-source-{neighbor.get('source_id', index)}-host-setup", provenance=ldc_provenance)
    else:
        from dataclasses import replace
        atmosphere = neighbor.get("stellar_atmosphere")
        if atmosphere is None:
            raise _facade("TrexEvidenceUnavailableError")(
                "resolved source requires its own Teff, [M/H], A_V and provenance; target borrowing is forbidden")
        source_light = replace(stellar_light, target_av_mag=atmosphere["av_mag"])
        component = source_light.prepare_component(atmosphere["teff_k"], M_n, R_n, neighbor_feh, atmosphere["av_mag"])
        host_u1_n, host_u2_n = np.atleast_1d(component.u1), np.atleast_1d(component.u2)
        physical = dict(stellar_light=source_light, target_teff_k=atmosphere["teff_k"])
    neighbor_log_prior = _validate_scalar(neighbor.get("log_prior", 0.0), "neighbor spatial log-prior", positive=False)
    if neighbor_log_prior > 0.0:
        raise ValueError(
            "resolved neighbor {0} has an invalid spatial log-prior".format(index)
        )
    return {
        "source_id": neighbor.get("source_id", index),
        "M_s": M_n,
        "R_s": R_n,
        "u1": float(host_u1_n[0]),
        "u2": float(host_u2_n[0]),
        "aperture_flux_fraction": neighbor_fraction,
        "log_prior": float(neighbor_log_prior),
        "target_feh": neighbor_feh,
        **composition_context,
        # Carried for reporting only: the evidence calculation keeps this
        # host's mass and radius fixed, so the retained 16-84 envelope is
        # never used as a sampling prior or silently clipped.
        **({key: neighbor[key] for key in (
            "mass_solar_err_lower", "mass_solar_err_upper",
            "radius_solar_err_lower", "radius_solar_err_upper",
            "stellar_parameter_provenance",
            "stellar_parameter_distribution",
        ) if neighbor.get(key) is not None}),
        **physical,
    }


def calc_target_evidences(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    depth_ppm: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    exptime_days: float,
    bin_integration_days: float = 0.0,
    target_aperture_flux_fraction: Optional[float] = None,
    n_draws: int = 2000,
    contrast_separations: Optional[np.ndarray] = None,
    contrast_values: Optional[np.ndarray] = None,
    plx: Optional[float] = None,
    target_tess_mag: float = float("nan"),
    background_population: Optional[Mapping[str, np.ndarray]] = None,
    background_area_sq_deg: Optional[float] = None,
    nearby_stars: Optional[list] = None,
    random_seed: Optional[int] = None,
    progress_callback: Optional[callable] = None,
    nsamples: int = 20,
    target_feh: Optional[float] = None,
    resolution_limit_arcsec: Optional[float] = None,
    ldc_provenance: Optional[list] = None,
    ess_out: Optional[Dict[int, float]] = None,
    support_state_out: Optional[Dict[int, dict]] = None,
    jax_fallback_out: Optional[Dict[int, dict]] = None,
    depth_sigma_ppm: Optional[float] = None,
    duration_days: Optional[float] = None,
    duration_sigma_days: Optional[float] = None,
    representative_fit_out: Optional[Dict[int, dict]] = None,
    target_source_id: Optional[Any] = None,
    target_teff_k: Optional[float] = None,
    stellar_light=None,
    baseline=None,
    importance_centres_out: Optional[Dict[int, list]] = None,
    joint_observations=None,
) -> Tuple[np.ndarray, str]:
    """Compute log-evidences for all target-star and neighbour scenarios.

    ``ess_out`` records Kish ESS by evaluated scenario index; skipped scenarios
    retain ``-inf`` evidence and no entry. The caller supplies it to
    :func:`compute_fpp_nfpp` to refuse degenerate evidence normalization.

    ``importance_centres_out`` receives optional unit-cube continuation pilots
    by scenario index, independently of the representative model records.

    ``support_state_out`` records :func:`_proposal_support_state` for each
    evaluated occultor-radius proposal window without changing the evidence.

    ``jax_fallback_out`` records each failed JAX batch re-evaluated by batman:
    scenario index, likelihood family, reference-path draw count, exception
    type/message. Successful batches leave no entry.

    ``resolution_limit_arcsec`` bounds the unresolved population independently
    of any measured contrast curve. The curve excludes detected companions only
    over its measured separation range; it never truncates the prior at its
    inner or outer edge. If a curve is supplied without an explicit catalog
    bound, the published Gaia close-pair resolution is used.

    Every candidate host is evaluated on the light curve renormalized onto
    that host's calibrated share of the in-aperture flux, following the
    per-host ``renorm_flux`` step of Giacalone et al. (2021, AJ, 161, 24).
    ``target_aperture_flux_fraction`` and each entry of ``nearby_stars`` must
    therefore carry an ``aperture_flux_fraction`` measured from calibrated
    aperture-scene evidence: a catalog magnitude ratio is not aperture
    throughput, so no default can supply a missing measurement.

    ``target_feh`` is the candidate's own metallicity [dex]; it is required
    to select host-specific limb darkening for the bound-companion-
    hosted (S*) scenarios (host atmosphere from that draw's companion mass
    via the Torres et al. 2010 relation). Resolved neighbors use their own
    source-associated metallicity when supplied, otherwise the target's with
    the reason retained. No bound-companion population carries an independent
    metallicity. GSP-Phot metallicity is not used as a substitute.
    B* instead uses each retained TRILEGAL row's simulated [M/H] [dex],
    including for its eclipsed secondary. Without ``target_feh`` S* slots and
    neighbors lacking their own metallicity remain unresolved. Every EB slot,
    including target-hosted and diluted D*/P*
    pairs, also requires the host's composition for the eclipsed secondary's own
    grid atmosphere: its temperature and gravity follow that draw's secondary
    mass and radius. Missing metallicity cannot be replaced by primary LDC;
    such scenarios are recorded unavailable before evaluation. Callers that
    cannot retain those scenario records must supply ``target_feh`` or fail.

    ``target_teff_k`` retains the supplied target temperature [K] in component
    fluxes; when omitted, legacy callers keep the mass-temperature relation.
    Population-hosted events use the drawn host's own temperature instead.
    With explicit ``stellar_light``, resolved hosts also need source-owned
    temperature/composition/extinction; missing inputs raise, not -inf evidence.
    ``baseline`` explicitly retains a normalized native-time Gaussian density
    in aperture coordinates for all hosts. It is prepared once and never
    receives the host-renormalized data as independent conditioning evidence.

    Returns:
        (lnZ, status): lnZ array of shape (n_scenarios,). Status is ``unavailable``
        for missing scenario inputs, ``unresolved-numerical`` for supported
        likelihood failures, or ``ok`` for completed sampling. ``ok`` does not
        certify estimator convergence.
    """
    if joint_observations is None:
        exptime_days = _validate_scalar(exptime_days, "exptime_days", positive=True)
    elif exptime_days is not None or baseline is not None or np.any(np.asarray(bin_integration_days) != 0):
        raise ValueError("joint observations own their exposure and baseline; no legacy integration is allowed")
    bin_integration_days = _facade("phase_bin_widths")(bin_integration_days, np.asarray(time).size)
    if (
        isinstance(depth_ppm, (bool, np.bool_))
        or not isinstance(depth_ppm, (int, float, np.number))
        or not math.isfinite(depth_ppm)
        or depth_ppm <= 0.0
    ):
        raise ValueError("depth_ppm must be finite and positive")
    if (contrast_separations is None) != (contrast_values is None):
        raise ValueError("contrast_separations and contrast_values must be supplied together")
    curve_available = (
        contrast_separations is not None
        and contrast_values is not None
        and np.asarray(contrast_separations).ndim == 1
        and np.asarray(contrast_values).ndim == 1
        and len(contrast_separations) >= 2
        and len(contrast_values) >= 2
    )
    if (
        resolution_limit_arcsec is None
        and curve_available
    ):
        resolution_limit_arcsec = _facade("GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC")
    if target_aperture_flux_fraction is None:
        raise ValueError(
            "target aperture flux fraction is required; calibrated aperture-scene "
            "evidence must supply every host's share of the in-aperture flux"
        )
    target_aperture_flux_fraction = _validate_scalar(
        target_aperture_flux_fraction, "target aperture flux fraction", positive=True
    )
    if target_aperture_flux_fraction > 1.0:
        raise ValueError("target aperture flux fraction must not exceed one")
    if target_feh is not None:
        target_feh = _validate_scalar(target_feh, "target_feh", positive=False)
    if target_feh is None and representative_fit_out is None:
        raise ValueError("target_feh is required when unavailable scenario records cannot be retained")
    if target_teff_k is not None:
        target_teff_k = _validate_scalar(target_teff_k, "target_teff_k", positive=True)
    target_flux, target_sigma = _facade("renorm_flux_to_host")(
        flux, sigma, target_aperture_flux_fraction
    )
    # The depth each host would have to produce alone, in its own renormalized
    # frame. It positions the occultor-radius importance proposal only; every
    # draw carries the compensating weight, so this never enters a likelihood.
    target_depth_fraction = (
        depth_ppm * _facade("FRACTION_PER_PART_PER_MILLION") / float(target_aperture_flux_fraction)
    )
    if depth_sigma_ppm is not None:
        depth_sigma_ppm = _validate_scalar(depth_sigma_ppm, "depth_sigma_ppm", positive=True)
    if resolution_limit_arcsec is not None:
        resolution_limit_arcsec = _validate_scalar(resolution_limit_arcsec, "resolution_limit_arcsec", positive=True)
    if plx is not None:
        plx = _validate_scalar(plx, "plx", positive=True)
    target_depth_sigma = (
        depth_sigma_ppm * _facade("FRACTION_PER_PART_PER_MILLION") / float(target_aperture_flux_fraction)
        if depth_sigma_ppm is not None and math.isfinite(depth_sigma_ppm) and depth_sigma_ppm > 0.0
        else None
    )
    baseline_density = None if baseline is None else baseline.prepare(time, flux, sigma, bin_integration_days)
    n_scenarios = _facade("N_TARGET_SCENARIOS")
    if nearby_stars:
        n_scenarios += len(nearby_stars) * _facade("N_NEIGHBOR_SCENARIOS")
    # A missing-input gate must not shift another scenario's random draws.
    scenario_seeds = np.random.SeedSequence(random_seed).spawn(n_scenarios)

    lnZ = np.full(n_scenarios, -np.inf)
    unavailable = False
    numerical_failure = False

    def _commit_scenario_result(index, value, ess, state, fallback, fit, context,
                                neighbor_log_prior=None, centres=None):
        nonlocal numerical_failure
        lnZ[index] = value
        for output, records in ((ess_out, ess), (support_state_out, state), (jax_fallback_out, fallback)):
            if output is not None and records:
                output[index] = records[-1]
        if neighbor_log_prior is not None:
            lnZ[index] += neighbor_log_prior
        if importance_centres_out is not None and centres:
            importance_centres_out[index] = centres[-1]
        if representative_fit_out is not None and fit:
            fit_record = dict(fit[-1])
            numerical_failure |= bool(fit_record.get("unresolved_numerical_failures"))
            fit_record.update(context(), ln_evidence=float(lnZ[index]))
            # The proposal's own support classification travels with the
            # evidence it produced: a sub-floor ESS is not interpretable
            # without knowing whether the window could reach the prior at all.
            if state:
                fit_record["proposal_support"] = dict(state[-1])
            representative_fit_out[index] = fit_record

    # Unresolved populations need an angular constraint; a resolved neighbour's
    # catalog contrast is not a constraint on an unresolved companion.
    unresolved_constraint_available = curve_available or (
        resolution_limit_arcsec is not None
        and math.isfinite(float(resolution_limit_arcsec))
        and float(resolution_limit_arcsec) > 0.0
    )
    bound_companion_available = (
        unresolved_constraint_available
        and plx is not None
        and math.isfinite(plx)
        and plx > 0.0
    )
    bound_companion_unavailable_reason = None
    if bound_companion_available:
        try:
            _facade("companion_population_frequency")(M_s_Msun)
        except ValueError as exc:
            if not math.isfinite(M_s_Msun) or M_s_Msun <= 0.0:
                raise
            bound_companion_available = False
            bound_companion_unavailable_reason = str(exc)

    # The bound-companion *hosted* (S*) scenarios additionally need the
    # target's own metallicity to select the drawn companion's atmosphere
    # from the TRICERATOPS grid (see ``_host_grid_coefficients``). The
    # diluted P* planet scenarios keep the target as host; P* EB scenarios
    # separately require the metallicity for the eclipsed secondary.
    bound_companion_hosted_available = (
        bound_companion_available
        and target_feh is not None
        and math.isfinite(float(target_feh))
    )

    # A background star is an unrelated line-of-sight source: its scenario
    # needs the retained population, that population's simulated sky area,
    # the target magnitude that defines "fainter", and an angular bound on
    # the field; a contrast curve is optional. Target parallax only enters
    # the *bound*-companion branch to convert an orbital period into a
    # projected separation.  Requiring it here retired every background
    # scenario to -inf whenever the parallax was missing, biasing FPP low.
    datasets = () if joint_observations is None else joint_observations.population_datasets
    declared = None
    if background_population is not None:
        from trex.populations.background import declared_background_selection

        declared = declared_background_selection(
            background_population, datasets=datasets, target_tess_mag=target_tess_mag)
    background_available = (
        declared is not None
        and background_area_sq_deg is not None
        and declared.depth_mag is not None
        and math.isfinite(declared.depth_mag)
        and unresolved_constraint_available
    )

    # B* uses the selected TRILEGAL row's [M/H], not the unrelated target's [Fe/H].
    background_hosted_available = background_available

    # (is_planet, is_EB, use_2xP, comp_is_host, comp_frac, flat, lnprior_comp,
    #  bound_companion, background_star)
    target_id_str = str(target_source_id) if target_source_id is not None else "target"
    target_baseline = None if baseline_density is None else baseline_density.for_host(target_aperture_flux_fraction)
    overlap = _facade("_ScenarioOverlap")()
    try:
        for idx, (is_pl, is_eb, x2p, cih, cfr, flat, lnp_c, bound, background) in enumerate(_facade("TARGET_SCENARIO_SPECS")):
            scenario_name = _facade("_SCENARIO_NAMES").get(idx, str(idx))
            host_is_secondary = bool(cih or bound == "secondary" or background == "background")
            _facade("emit_progress")(
                progress_callback,
                f"Scenario {scenario_name}",
                done=idx,
                total=n_scenarios,
            )
            if (
                (target_feh is None and is_eb and background != "background")
                or
                (bound == "secondary" and not bound_companion_hosted_available)
                or (bound is not None and not bound_companion_available)
                or (background == "background" and not background_hosted_available)
                or (background is not None and not background_available)
                or (bound is None and background is None and not math.isfinite(lnp_c) and lnp_c < -1e300)
            ):
                unavailable |= not (
                    bound is None and background is None and not math.isfinite(lnp_c) and lnp_c < -1e300
                )
                if representative_fit_out is not None:
                    missing = []
                    if target_feh is None and (is_eb and background != "background" or bound == "secondary"):
                        missing.append("target metallicity [Fe/H]")
                    if bound is not None or background is not None:
                        if not unresolved_constraint_available:
                            missing.append("unresolved angular constraint (contrast curve or resolution limit)")
                    if bound is not None:
                        if plx is None or not math.isfinite(plx) or plx <= 0.0:
                            missing.append("finite positive target parallax")
                        if bound_companion_unavailable_reason:
                            missing.append(bound_companion_unavailable_reason)
                    if background is not None:
                        if background_population is None:
                            missing.append("background population")
                        if background_area_sq_deg is None:
                            missing.append("background population sky area")
                        if declared is None or declared.depth_mag is None:
                            missing.append("declared background selection depth")
                    status = "unavailable" if missing else "not-evaluated"
                    representative_fit_out[idx] = _unevaluated_scenario_record(
                        idx, scenario_name, scenario_name, target_id_str, None, status,
                        is_pl, is_eb, x2p, host_is_secondary, period_days,
                    )
                    representative_fit_out[idx]["reason"] = (
                        "; ".join(missing) if missing else "scenario prior is impossible (lnp_c < -1e300)"
                    )
                continue
            scenario_ess: list = []
            scenario_state: list = []
            scenario_fallback: list = []
            scenario_fit: list = []
            scenario_centres: list = []
            call = partial(
                _facade("_eval_scenario"),
                time, target_flux, target_sigma, period_days, M_s_Msun, R_s_Rsun,
                u1, u2, n_draws, is_pl, is_eb, x2p, exptime_days,
                bin_integration_days,
                companion_is_host=cih, companion_fluxratio=cfr,
                lnprior_comp=lnp_c, use_flat_priors=flat,
                rng=np.random.default_rng(scenario_seeds[idx]), nsamples=nsamples,
                bound_companion=bound,
                plx_mas=plx,
                contrast_separations=contrast_separations,
                contrast_values=contrast_values,
                background_star=background,
                background_population=background_population,
                background_area_sq_deg=background_area_sq_deg,
                target_feh=target_feh,
                target_teff_k=target_teff_k,
                **({"stellar_light": stellar_light} if stellar_light is not None else {}),
                **({"baseline_likelihood": target_baseline} if target_baseline is not None else {}),
                **({"joint_observations": joint_observations.for_host(target_id_str)}
                   if joint_observations is not None else {}),
                target_tess_mag=target_tess_mag,
                resolution_limit_arcsec=resolution_limit_arcsec,
                ldc_provenance=ldc_provenance,
                ldc_scenario=scenario_name,
                ess_out=scenario_ess,
                support_state_out=scenario_state,
                jax_fallback_out=scenario_fallback,
                host_depth_fraction=target_depth_fraction,
                host_depth_sigma=target_depth_sigma,
                host_duration_days=duration_days,
                host_duration_sigma_days=duration_sigma_days,
                representative_fit_out=scenario_fit if representative_fit_out is not None else None,
                **({"importance_centres_out": scenario_centres} if importance_centres_out is not None else {}),
            )

            def record(value, idx=idx, scenario_name=scenario_name, scenario_ess=scenario_ess,
                       scenario_state=scenario_state, scenario_fallback=scenario_fallback,
                       scenario_fit=scenario_fit, scenario_centres=scenario_centres):
                _commit_scenario_result(
                    idx, value, scenario_ess, scenario_state, scenario_fallback, scenario_fit,
                    lambda: dict(
                        scenario_index=idx,
                        scenario=scenario_name,
                        diagnostic_key=scenario_name,
                        source_id=target_id_str,
                        neighbor_index=None,
                        aperture_flux_fraction=float(target_aperture_flux_fraction),
                        evidence_origin="monte-carlo",
                    ),
                    centres=scenario_centres,
                )

            overlap.submit(call, record)

        # Giacalone et al. (2021), Section II.1: a resolved host must supply the
        # observed depth within its own fraction of the total aperture flux.
        if nearby_stars:
            observed_depth_fraction = depth_ppm * _facade("FRACTION_PER_PART_PER_MILLION")
            for ni, neighbor in enumerate(nearby_stars):
                base = _facade("N_TARGET_SCENARIOS") + ni * _facade("N_NEIGHBOR_SCENARIOS")
                neighbor_source_id = str(neighbor.get("source_id", ni))
                host = _facade("resolved_neighbor_host")(
                    neighbor, target_feh, ni, observed_depth_fraction, ldc_provenance,
                    **({"stellar_light": stellar_light} if stellar_light is not None else {}),
                    **({"joint_observations": joint_observations} if joint_observations is not None else {}),
                )
                if host is None:
                    missing_metallicity = (
                        stellar_light is None and target_feh is None
                        and float(neighbor["aperture_flux_fraction"]) > 0.0
                        and observed_depth_fraction <= float(neighbor["aperture_flux_fraction"])
                    )
                    unavailable |= missing_metallicity
                    if representative_fit_out is not None:
                        for local in range(_facade("N_NEIGHBOR_SCENARIOS")):
                            is_pl = (local == 0)
                            is_eb = not is_pl
                            x2p = (local == 2)
                            si = base + local
                            representative_fit_out[si] = _unevaluated_scenario_record(
                                si, _facade("NEIGHBOR_SCENARIO_NAMES")[local],
                                _facade("_SCENARIO_NAMES").get(si, "N{0}".format(si)),
                                neighbor_source_id, ni, "unavailable" if missing_metallicity else "host-excluded",
                                is_pl, is_eb, x2p, False, period_days,
                            )
                            if missing_metallicity:
                                representative_fit_out[si]["reason"] = "target metallicity [Fe/H]"
                    continue
                neighbor_fraction = host["aperture_flux_fraction"]
                neighbor_flux, neighbor_sigma = _facade("renorm_flux_to_host")(
                    flux, sigma, neighbor_fraction
                )
                neighbor_log_prior = host["log_prior"]
                neighbor_baseline = None if baseline_density is None else baseline_density.for_host(neighbor_fraction)
                for local in range(_facade("N_NEIGHBOR_SCENARIOS")):
                    is_pl = (local == 0)
                    is_eb = not is_pl
                    x2p = (local == 2)
                    si = base + local
                    _facade("emit_progress")(
                        progress_callback, f"Neighbor {ni}", done=si, total=n_scenarios
                    )
                    neighbor_ess: list = []
                    neighbor_state: list = []
                    neighbor_fallback: list = []
                    neighbor_fit: list = []
                    neighbor_centres: list = []
                    neighbor_ldc = (_NeighborLdcEvents(ldc_provenance, ni, neighbor_source_id,
                                                     _facade("NEIGHBOR_SCENARIO_NAMES")[local])
                                    if ldc_provenance is not None else None)
                    call = partial(
                        _facade("_eval_scenario"),
                        time, neighbor_flux, neighbor_sigma, period_days, M_s_Msun, R_s_Rsun,
                        host["u1"], host["u2"], n_draws, is_pl, is_eb, x2p, exptime_days,
                        bin_integration_days,
                        use_flat_priors=False,
                        M_s_neighbor=host["M_s"], R_s_neighbor=host["R_s"],
                        **({"host_distribution": host["stellar_parameter_distribution"]}
                           if "stellar_parameter_distribution" in host else {}),
                        rng=np.random.default_rng(scenario_seeds[si]), nsamples=nsamples,
                        **({"baseline_likelihood": neighbor_baseline} if neighbor_baseline is not None else {}),
                        **({"joint_observations": joint_observations.for_host(
                            neighbor_source_id, av_mag=host["stellar_light"].target_av_mag)}
                           if joint_observations is not None else {}),
                        target_feh=host.get("target_feh", target_feh),
                        **({"stellar_light": host["stellar_light"], "target_teff_k": host["target_teff_k"]} if stellar_light is not None else {}),
                        ldc_scenario="Neighbor {0}".format(ni),
                        scenario_label=_facade("NEIGHBOR_SCENARIO_NAMES")[local],
                        ldc_provenance=neighbor_ldc,
                        ess_out=neighbor_ess,
                        support_state_out=neighbor_state,
                        jax_fallback_out=neighbor_fallback,
                        host_depth_fraction=observed_depth_fraction / neighbor_fraction,
                        host_depth_sigma=(
                            None if target_depth_sigma is None
                            else target_depth_sigma * float(target_aperture_flux_fraction) / neighbor_fraction
                        ),
                        host_duration_days=duration_days,
                        host_duration_sigma_days=duration_sigma_days,
                        representative_fit_out=neighbor_fit if representative_fit_out is not None else None,
                        **({"importance_centres_out": neighbor_centres} if importance_centres_out is not None else {}),
                    )

                    def record(value, si=si, local=local, ni=ni, host=host,
                               neighbor_source_id=neighbor_source_id, neighbor_fraction=neighbor_fraction,
                               neighbor_log_prior=neighbor_log_prior, neighbor_ess=neighbor_ess,
                               neighbor_state=neighbor_state, neighbor_fallback=neighbor_fallback,
                               neighbor_fit=neighbor_fit, neighbor_centres=neighbor_centres):
                        _commit_scenario_result(
                            si, value, neighbor_ess, neighbor_state, neighbor_fallback, neighbor_fit,
                            lambda: dict(
                                scenario_index=si,
                                scenario=_facade("NEIGHBOR_SCENARIO_NAMES")[local],
                                diagnostic_key=_facade("_SCENARIO_NAMES").get(si, "N{0}".format(si)),
                                source_id=str(host.get("source_id", neighbor_source_id)),
                                neighbor_index=ni,
                                aperture_flux_fraction=float(neighbor_fraction),
                                neighbor_log_prior=float(neighbor_log_prior),
                                metallicity_provenance=host["metallicity_provenance"],
                                metallicity_unavailable_reason=host["metallicity_unavailable_reason"],
                                host_metallicity_dex=host["target_feh"],
                                metallicity_treatment=host["metallicity_treatment"],
                                evidence_origin="monte-carlo",
                            ), neighbor_log_prior, centres=neighbor_centres,
                        )

                    overlap.submit(call, record)

    finally:
        try:
            overlap.wait()
        finally:
            overlap.close()

    # Every scenario index has now been decided, so report the real terminal
    # count: without this the last update would sit one short of the total and
    # a finished evaluation would look stalled.
    _facade("emit_progress")(
        progress_callback, "Scenario evaluation complete", done=n_scenarios, total=n_scenarios
    )
    status = "unresolved-numerical" if numerical_failure else "unavailable" if unavailable else "ok"
    return lnZ, status
