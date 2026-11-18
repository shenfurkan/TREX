"""Monte Carlo scenario likelihood evaluation."""
from __future__ import annotations

from trex.runtime.logging import host_logger
import math
import traceback
from functools import partial
from time import perf_counter
from typing import Any, Dict, Mapping, Optional
import numpy as np
from trex.foundation.errors import NumericalLikelihoodError
from trex.evidence.mc_occultors import draw_eb_occultor, draw_planet_occultor
from trex.evidence.nested_controls import _validate_scalar


# NUMERICAL_POLICY: skipped draws sit this far below the best evaluated
# evidence term [nats], so together they move Z by under N*exp(-30), 1e-7 at 1e6 draws.
_SCREEN_CUT_NATS = 30.0
# Share of grid points in the partial chi-square, and draws per full-grid call.
_SCREEN_POINT_FRACTION = 0.1
_SCREEN_STEP_DRAWS = 1 << 14


def _facade(name):
    """Look the name up on ``marginal_likelihoods``, the module tests patch."""
    from trex.evidence import marginal_likelihoods

    return getattr(marginal_likelihoods, name)


def _screened_lnL(bound, log_weight, evaluate):
    """Full-grid lnL for the draws that can reach the evidence, -inf for the rest.

    ``bound`` is lnL over part of the grid. Gaussian residual terms are never positive,
    so it bounds the full lnL from above whatever the model. Draws run in descending
    bound + log_weight order and stop once no remaining draw can come within
    ``_SCREEN_CUT_NATS`` of the best evaluated term. ``evaluate(draws)`` returns their lnL.
    """
    potential = np.nan_to_num(bound+log_weight, nan=np.inf)  # NaN says nothing, so keep the draw.
    order = np.argsort(-potential, kind="stable")
    lnL = np.full(bound.size, -np.inf)
    best = -np.inf
    for start in range(0, order.size, _SCREEN_STEP_DRAWS):
        if potential[order[start]] < best-_SCREEN_CUT_NATS:
            break
        draws = order[start:start+_SCREEN_STEP_DRAWS]
        lnL[draws] = evaluate(draws)
        best = np.fmax(best, np.fmax.reduce(lnL[draws]+log_weight[draws]))
    return lnL


def _scenario_stages(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: "np.ndarray | float",
    period_days: float,
    M_s_Msun: float,
    R_s_Rsun: float,
    u1: float,
    u2: float,
    n_draws: int,
    is_planet: bool,
    is_EB: bool,
    use_2xP: bool,
    exptime_days: float,
    bin_integration_days: float = 0.0,
    companion_is_host: bool = False,
    companion_fluxratio: Any = 0.0,
    lnprior_comp: Any = 0.0,
    use_flat_priors: bool = False,
    M_s_neighbor: Optional[Any] = None,
    R_s_neighbor: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
    nsamples: int = 20,
    bound_companion: Optional[str] = None,
    plx_mas: float = 0.0,
    contrast_separations: Optional[np.ndarray] = None,
    contrast_values: Optional[np.ndarray] = None,
    background_star: Optional[str] = None,
    background_population: Optional[Mapping[str, np.ndarray]] = None,
    background_area_sq_deg: Optional[float] = None,
    target_tess_mag: float = float("nan"),
    target_feh: Optional[float] = None,
    resolution_limit_arcsec: Optional[float] = None,
    ldc_provenance: Optional[list] = None,
    ldc_scenario: str = "host",
    scenario_label: Optional[str] = None,
    ess_out: Optional[list] = None,
    support_state_out: Optional[list] = None,
    jax_fallback_out: Optional[list] = None,
    host_depth_fraction: Optional[float] = None,
    host_depth_sigma: Optional[float] = None,
    host_duration_days: Optional[float] = None,
    host_duration_sigma_days: Optional[float] = None,
    representative_fit_out: Optional[list] = None,
    _previous_scenario=None,
    target_teff_k: Optional[float] = None,
    stellar_light=None,
    baseline_likelihood=None,
    host_distribution=None,
    importance_centres_out: Optional[list] = None,
    joint_observations=None,
):
    """Estimate lnZ for one astrophysical scenario via Monte Carlo integration.

    A generator: it yields once after drawing every random variate, and returns
    lnZ when the likelihood stage finishes (see :func:`_eval_scenario`).

    ``companion_fluxratio``, ``lnprior_comp``, ``M_s_neighbor``, and
    ``R_s_neighbor`` each accept a scalar shared by every draw or a per-draw
    array of length ``n_draws``.  The unresolved-companion scenarios of
    Giacalone et al. (2021) marginalize over a companion *population*, so the
    diluting companion, the host star it orbits, and that draw's occurrence
    prior all vary draw by draw.  The prior is added inside the expectation --
    ``lnZ = log mean_i exp(lnL_i + lnprior_i)`` -- which reduces exactly to the
    historical "add a scalar afterwards" form when the prior is constant.

    ``host_depth_fraction`` is the observed transit depth as a fraction of the
    flux array passed in, i.e. already renormalized onto this scenario's
    resolved host.  It positions the occultor-radius importance proposal and
    nothing else: the draws carry a compensating ``ln(p/q)`` weight, so the
    depth never enters the likelihood and cannot pull the evidence toward the
    observation.  Omitting it falls the scenario back to prior sampling.

    ``ess_out``, when supplied, receives one appended float: the Kish effective
    sample size of this scenario's log-weights.  It is the estimator's own
    convergence evidence -- an ``lnZ`` returned with ``ESS`` near one is the
    largest single draw, not an integral -- and the caller records it so a
    degenerate evidence set is never normalized into a probability.

    ``support_state_out``, when supplied, receives one appended mapping from
    :func:`_proposal_support_state`: whether the occultor-radius proposal
    window intersects the scenario prior's hard support.  It is routing
    evidence for an integrator that handles boundary-supported likelihoods; it
    never alters a draw, a weight, or the returned evidence.

    ``jax_fallback_out``, when supplied, receives one appended record per JAX
    batch that raised and was re-evaluated with the batman reference
    likelihood: the likelihood family, the number of draws routed to the
    reference path, and the exception type and message.  A reference
    evaluation that never attempted a JAX batch records nothing, so an empty
    record list states that the accelerator path carried the whole batch.

    ``importance_centres_out`` receives the optional unit-cube pilot separately
    from representative fit records; it is continuation input, not a fitted model.

    Returns:
        Log-evidence lnZ (float; -inf if no draws transit).

    ``scenario_label`` is the canonical scenario code (``NTP``, ``NEB``,
    ``NEBx2P``) of a neighbour scenario, whose ``ldc_scenario`` is the shared
    ``Neighbor <i>`` name that also seeds the importance pilot; logs use both
    so the three neighbour scenarios stay distinct.
    """
    log_name = ldc_scenario if scenario_label is None else f"{ldc_scenario} {scenario_label}"
    started = perf_counter()
    if joint_observations is None:
        exptime_days = _validate_scalar(exptime_days, "exptime_days", positive=True)
    elif exptime_days is not None:
        raise ValueError("joint observations own their exposure responses; exptime_days must be None")
    period_days = _validate_scalar(period_days, "period_days", positive=True)
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(n_draws, (int, np.integer)) or n_draws <= 0:
        raise ValueError("n_draws must be a positive integer")
    for name, value in (("host_depth_fraction", host_depth_fraction), ("host_depth_sigma", host_depth_sigma),
                        ("host_duration_days", host_duration_days), ("host_duration_sigma_days", host_duration_sigma_days)):
        if value is not None:
            _validate_scalar(value, name, positive=True)
    if rng is None:
        rng = np.random.default_rng()

    eff_period = period_days * 2.0 if use_2xP else period_days

    M_host = M_s_neighbor if M_s_neighbor is not None else M_s_Msun
    R_host = _facade("_per_draw_values")(
        R_s_neighbor if R_s_neighbor is not None else R_s_Rsun, n_draws, "host radius"
    )
    M_host_arr = _facade("_per_draw_values")(M_host, n_draws, "host mass")

    # Component light must use this host's temperature [K], not a target surrogate.
    host_teff_k = (None if target_teff_k is None else
                   _facade("_per_draw_values")(target_teff_k, n_draws, "target temperature"))
    host_metallicity = target_feh
    host_av_mag = None if stellar_light is None else stellar_light.target_av_mag
    host_u1 = _facade("_per_draw_values")(u1, n_draws, "u1")
    host_u2 = _facade("_per_draw_values")(u2, n_draws, "u2")
    host_quantile = None
    if host_distribution is not None:
        if bound_companion is not None or background_star is not None or stellar_light is not None:
            raise ValueError("characterized resolved host requires the reference light model and no unresolved population")
        from trex.evidence.host_measure import HostMeasure
        measure = HostMeasure(host_distribution, target_feh, ldc_provenance)
        host_quantile = rng.random(n_draws if measure.n_dimensions == 1 else (n_draws, measure.n_dimensions))
        M_host_arr, R_host, host_teff_k, host_u1, host_u2 = measure.resolve(host_quantile)

    # A measured contrast curve keeps the historical draw call unchanged; the
    # catalog resolution bound is forwarded only when it is the constraint.
    resolution_options = (
        {} if resolution_limit_arcsec is None else {"resolution_limit_arcsec": resolution_limit_arcsec}
    )
    imaging_constraints = []
    dataset_dilutions = []
    companion_stability_out = None
    from trex.evidence.mc_importance import pilot_seed
    capture_pilot = pilot_seed.get() is not None and importance_centres_out is not None
    population_variates = [] if capture_pilot else None
    occultor_quantile = [] if capture_pilot else None
    if capture_pilot and (bound_companion is not None or background_star is not None):
        resolution_options["variates_out"] = population_variates
    if is_EB:
        resolution_options["imaging_state_out"] = imaging_constraints
    if joint_observations is not None and (bound_companion is not None or background_star is not None):
        resolution_options.update(datasets=joint_observations.population_datasets,
                                  dataset_dilutions_out=dataset_dilutions)
        if bound_companion is not None:
            resolution_options["band_light"] = joint_observations.band_light
    if bound_companion is not None:
        companion_is_host = bound_companion == "secondary"
        if stellar_light is not None:
            resolution_options.update(stellar_light=stellar_light, target_feh=target_feh)
        if target_teff_k is not None:
            resolution_options["host_teff_k"] = target_teff_k
        # Hierarchical stability (Mardling & Aarseth 2001 Eq. 90) masks the
        # drawn companion period against the observed inner orbit.  A planet
        # inner binary is the host alone, so the mask resolves inside the
        # draw; an eclipsing pair draws its mass ratio after this population,
        # so the draw defers its (log period, companion mass) columns and the
        # shared rule is applied once the EB secondary exists below.
        resolution_options["observed_period_days"] = eff_period
        if is_EB:
            companion_stability_out: list = []
            resolution_options["stability_out"] = companion_stability_out
        else:
            resolution_options["stability_hosted"] = companion_is_host
            companion_stability_out = None
        companion_fluxratio, lnprior_comp, M_comp, R_comp, teff_comp = _facade("_draw_bound_companion")(
            rng, n_draws, M_s_Msun, R_s_Rsun, plx_mas,
            contrast_separations, contrast_values,
            **resolution_options,
        )
        if companion_is_host:
            # The transit is hosted by the unresolved companion, so the
            # companion's own mass, radius, and (Torres et al. 2010)
            # main-sequence temperature set the geometry and atmosphere.
            M_host_arr = M_comp
            R_host = R_comp
            host_teff_k = teff_comp
            if target_feh is None:
                raise ValueError("secondary-hosted scenarios require candidate target_feh")
            if stellar_light is None:
                host_u1, host_u2 = _facade("_host_grid_coefficients")(
                    teff_comp, M_comp, R_comp, target_feh,
                    scenario=ldc_scenario, provenance=ldc_provenance,
                )
    elif background_star is not None:
        companion_stability_out = None
        metallicity_out = []
        extinction_out = []
        if background_star == "background":
            resolution_options["metallicity_out"] = metallicity_out
            if stellar_light is not None:
                resolution_options["extinction_out"] = extinction_out
        (
            companion_fluxratio,
            lnprior_comp,
            M_background,
            R_background,
            teff_background,
        ) = _facade("_draw_background_star")(
            rng, n_draws, background_population, background_area_sq_deg,
            target_tess_mag, contrast_separations, contrast_values,
            **resolution_options,
        )
        companion_is_host = background_star == "background"
        if companion_is_host:
            # The simulated background star hosts the event, so its catalog
            # mass, surface-gravity radius, and temperature set the geometry.
            M_host_arr = M_background
            R_host = R_background
            host_teff_k = teff_background
            host_metallicity = metallicity_out[0]
            if stellar_light is not None:
                host_av_mag = extinction_out[0]
            if stellar_light is None:
                host_u1, host_u2 = _facade("_host_grid_coefficients")(
                    teff_background, M_background, R_background, host_metallicity,
                    scenario=ldc_scenario, provenance=ldc_provenance,
                )

    fixed_host_component = None
    if stellar_light is not None:
        if host_teff_k is None:
            raise ValueError("physical stellar light requires the actual host temperature")
        if host_metallicity is None:
            raise ValueError("physical stellar light requires the host metallicity")
        host_metallicity = _facade("_per_draw_values")(host_metallicity, n_draws, "host metallicity")
        host_av_mag = _facade("_per_draw_values")(host_av_mag, n_draws, "host extinction")
        if not all(np.isfinite(value).all() for value in (host_metallicity, host_av_mag)):
            raise ValueError("physical stellar light requires finite host metallicity and extinction")
        if not companion_is_host and all(np.all(value == value[0]) for value in (
            host_teff_k, M_host_arr, R_host, host_metallicity, host_av_mag
        )):
            # Fixed scenario inputs are prepared once; no draw-dependent atmosphere is cached.
            fixed_host_component = stellar_light.prepare_component(
                host_teff_k[0], M_host_arr[0], R_host[0], host_metallicity[0], host_av_mag[0])
            host_u1, host_u2 = (np.full(n_draws, value) for value in (fixed_host_component.u1, fixed_host_component.u2))
        else:
            _, host_u1, host_u2 = stellar_light.surface_light(
                host_teff_k, M_host_arr, R_host, host_metallicity, host_av_mag)

    companion_fluxratio = _facade("_per_draw_values")(
        companion_fluxratio, n_draws, "companion flux ratio"
    )
    lnprior_arr = _facade("_per_draw_values")(lnprior_comp, n_draws, "companion prior")
    # These weights are occurrence rates -- an expected number of companions or
    # of background stars -- so a positive log is legitimate in a crowded field
    # or around a massive primary. Only a non-finite positive weight is invalid.
    if np.any(np.isposinf(lnprior_arr)) or np.any(np.isnan(lnprior_arr)):
        raise ValueError("companion prior must be a finite occurrence weight or -inf")

    # Orbital shape is drawn from its prior: the observed single transit does
    # not pin eccentricity or the argument of periastron the way it pins the
    # occultor radius and the inclination, so re-proposing them would buy no
    # concentration.
    x_ecc = rng.random(n_draws)
    x_w = rng.random(n_draws)
    ecc = _facade("sample_ecc")(x_ecc, planet=is_planet, P_orb=eff_period)
    w = _facade("sample_w")(x_w)

    # The occultor-radius proposal is centred on the radius ratio the observed
    # depth implies for THIS draw's host. ``flux`` is already renormalized onto
    # the resolved host, and the forward model dilutes the event by any
    # unresolved companion, so the depth the model must produce before that
    # dilution is the frame depth times the dilution factor below -- the exact
    # inverse of the ``(flux + F_dilute) / (1 + F_dilute)`` step in
    # ``simulate_TP``. Without a measured depth the window is left empty and
    # every draw falls back to the prior.
    if host_depth_fraction is not None and math.isfinite(host_depth_fraction) and host_depth_fraction > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            companion_flux = np.where(
                companion_fluxratio > 0.0,
                companion_fluxratio / (1.0 - companion_fluxratio),
                0.0,
            )
            dilution = np.where(
                companion_fluxratio > 0.0,
                1.0 + np.where(companion_is_host, 1.0 / companion_flux, companion_flux),
                1.0,
            )
        dilution = np.where(np.isfinite(dilution) & (dilution > 0.0), dilution, 1.0)
        # The window spans the measured depth uncertainty, not a fixed factor:
        # a proposal far wider than the posterior leaves the estimator at one
        # effective draw however many draws are taken.
        depth_sigma = (
            float(host_depth_sigma)
            if host_depth_sigma is not None
            and math.isfinite(host_depth_sigma)
            and host_depth_sigma > 0.0
            else host_depth_fraction
        )
        depth_low = max(host_depth_fraction - _facade("_PROPOSAL_COVERAGE_SIGMA") * depth_sigma, 0.0)
        depth_high = host_depth_fraction + _facade("_PROPOSAL_COVERAGE_SIGMA") * depth_sigma
        occultor_rs_center = np.sqrt(host_depth_fraction * dilution)
        occultor_rs_low = np.sqrt(depth_low * dilution) / _facade("_PROPOSAL_GEOMETRY_SPAN")
        occultor_rs_high = np.sqrt(depth_high * dilution) * _facade("_PROPOSAL_GEOMETRY_SPAN")
    else:
        occultor_rs_center = None
        occultor_rs_low = None
        occultor_rs_high = None

    if is_EB:
        (eb_supported, M_total, R_EB, secondary_to_host_ratio, eb_u1, eb_u2, lnprior_arr, R_p, M_EB, occultor_log_weight) = draw_eb_occultor(
            M_host_arr,
            R_host,
            companion_fluxratio,
            companion_is_host,
            host_teff_k,
            imaging_constraints,
            ldc_provenance,
            ldc_scenario,
            lnprior_arr,
            n_draws,
            occultor_rs_center,
            occultor_rs_high,
            occultor_rs_low,
            rng,
            support_state_out,
            host_metallicity,
            use_2xP,
            stellar_light=stellar_light, host_av_mag=host_av_mag,
            **({"quantile_out": occultor_quantile} if capture_pilot else {}),
        )
        if companion_stability_out is not None:
            # The inner binary total mass is now known per draw: the eclipsing
            # pair primary (the companion for PEB, the target for SEB) plus its
            # secondary. The shared Mardling & Aarseth (2001) Eq. 90 rule masks
            # the drawn companion period against the observed (effective)
            # period, exactly as the planet slots mask inside the draw.
            log_period_out, companion_mass_out = companion_stability_out[0]
            inner_binary_mass = M_host_arr + M_EB
            outer_mass = (
                np.full(n_draws, M_s_Msun, dtype=float)
                if companion_is_host
                else companion_mass_out
            )
            finite = np.isfinite(log_period_out)
            if np.all(finite):
                companion_stability_ok = _facade("companion_hierarchical_stability_supported")(
                    log_period_out, eff_period, inner_binary_mass, outer_mass
                )
            else:
                # Zero-measure axis endpoints remain unsupported, not fabricated periods.
                companion_stability_ok = np.zeros(n_draws, dtype=bool)
                companion_stability_ok[finite] = _facade("companion_hierarchical_stability_supported")(
                    log_period_out[finite], eff_period, inner_binary_mass[finite], outer_mass[finite]
                )
        else:
            companion_stability_ok = None
    else:
        companion_stability_ok = None
        (eb_supported, M_total, R_EB, EB_fluxratio_arr, R_p, occultor_log_weight) = draw_planet_occultor(
            M_host_arr,
            R_host,
            n_draws,
            occultor_rs_center,
            occultor_rs_high,
            occultor_rs_low,
            rng,
            support_state_out,
            use_flat_priors,
            **({"quantile_out": occultor_quantile} if capture_pilot else {}),
        )

    a_cm = _facade("semi_major_axis_cgs")(eff_period, M_total)
    a_rs = a_cm / (R_host * _facade("Rsun"))
    occultor_rs = (
        R_EB / R_host if is_EB else R_p * _facade("Rearth") / (R_host * _facade("Rsun"))
    )

    # Inclination is drawn only from the band that transits. ``sample_inc``
    # with its default bounds is the isotropic inverse CDF ``cos i = 1 - x``,
    # and ``_geometric_transit_mask`` requires
    # ``a_rs * |cos i| * eccentric_factor <= 1 + occultor_rs``. The excluded
    # band contributes zero transit likelihood; the broad defensive proposal
    # covers the transiting band and its weights retain the original prior mass.
    eccentric_factor = (1.0 - ecc ** 2) / (1.0 + ecc * np.sin(np.radians(w)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_inc_max = (1.0 + occultor_rs) / (eccentric_factor * a_rs)
    cos_inc_max = np.where(np.isfinite(cos_inc_max), cos_inc_max, 0.0)
    transit_quantile = np.clip(cos_inc_max, 0.0, 1.0)
    band_low = 1.0 - transit_quantile

    # The observed duration pins the impact parameter as hard as the depth pins
    # the radius ratio, so the transiting band alone is still far wider than the
    # posterior. For a circular orbit,
    # ``sin(pi T14 / P) = sqrt((1 + k)**2 - b**2) / (a_rs sin i)``
    # (Seager & Mallen-Ornelas 2003, ApJ 585, 1038, Eq. 3,
    # DOI 10.1086/346105). The proposal approximates sin i as one and widens
    # the resulting chord bounds; ``eccentric_factor`` converts them to
    # inclination, not an exact eccentric duration.
    narrow_low = band_low
    narrow_high = np.ones(n_draws, dtype=float)
    if (
        host_duration_days is not None
        and math.isfinite(host_duration_days)
        and host_duration_days > 0.0
    ):
        duration_sigma = (
            float(host_duration_sigma_days)
            if host_duration_sigma_days is not None
            and math.isfinite(host_duration_sigma_days)
            and host_duration_sigma_days > 0.0
            else host_duration_days
        )
        span = _facade("_PROPOSAL_COVERAGE_SIGMA") * duration_sigma
        with np.errstate(divide="ignore", invalid="ignore"):
            # A longer duration means a smaller impact parameter, so the short
            # duration bound gives the upper b bound and vice versa.
            chord_short = a_rs * np.sin(
                np.pi * max(host_duration_days - span, 0.0) / eff_period
            ) * _facade("_PROPOSAL_GEOMETRY_SPAN")
            chord_long = a_rs * np.sin(
                np.pi * (host_duration_days + span) / eff_period
            ) / _facade("_PROPOSAL_GEOMETRY_SPAN")
            b_high = np.sqrt(np.maximum((1.0 + occultor_rs) ** 2 - chord_short ** 2, 0.0))
            b_low = np.sqrt(np.maximum((1.0 + occultor_rs) ** 2 - chord_long ** 2, 0.0))
            cos_low = np.clip(b_low / (eccentric_factor * a_rs), 0.0, 1.0)
            cos_high = np.clip(b_high / (eccentric_factor * a_rs), 0.0, 1.0)
        cos_low = np.where(np.isfinite(cos_low), cos_low, 0.0)
        cos_high = np.where(np.isfinite(cos_high), cos_high, 0.0)
        # cos i = 1 - x, so the larger impact parameter is the smaller quantile.
        narrow_low = np.clip(1.0 - np.maximum(cos_low, cos_high), band_low, 1.0)
        narrow_high = np.clip(1.0 - np.minimum(cos_low, cos_high), band_low, 1.0)

    x_inc, inclination_log_weight = _facade("_defensive_window_draws")(
        rng, n_draws, narrow_low, narrow_high,
        broad_low=band_low, broad_high=np.ones(n_draws, dtype=float),
    )
    inc = _facade("sample_inc")(x_inc)

    b = np.abs(a_rs * np.cos(np.radians(inc)))
    transit_mask = _facade("_geometric_transit_mask")(b, occultor_rs, ecc, w, a_rs)
    if eb_supported is not None:
        transit_mask = transit_mask & eb_supported
    if companion_stability_ok is not None:
        transit_mask = transit_mask & companion_stability_ok
    transit_mask = transit_mask & np.isfinite(lnprior_arr)
    lnL = np.full(n_draws, -np.inf)
    joint_best = None
    numerical_failures = np.zeros(n_draws, dtype=bool)
    numerical_examples = []

    def record_numerical_failure(indices, values, source, exc=None):
        bad = ~np.isfinite(values)
        numerical_failures[np.asarray(indices)] = bad
        numerical_examples[:] = [
            example for example in numerical_examples if numerical_failures[example["draw_index"]]
        ]
        if np.any(bad):
            failed = np.asarray(indices)[bad]
            if not numerical_examples:
                if exc is not None:
                    frames = traceback.extract_tb(exc.__traceback__)
                    if frames:
                        source = f"{frames[-1].filename}:{frames[-1].lineno}"
                numerical_examples.append({"draw_index": int(failed[0]), "source": source,
                                           "reason": (f"{type(exc).__name__}: {exc}" if exc is not None else
                                                      "non-finite likelihood on supported geometry")})

    log_weight = lnprior_arr + occultor_log_weight + inclination_log_weight
    prepared = None
    jax_call = None
    angular_model = stellar_light is not None and stellar_light.grid.angular is not None
    if np.any(transit_mask):
        idx = np.where(transit_mask)[0]
        # Every admitted draw goes to the tested device solver, including the
        # near-unity tail batman cannot solve.
        jax_eligible = ecc[idx] <= _facade("_JAX_MAX_SUPPORTED_ECCENTRICITY")
        idx_fast = idx[jax_eligible]
        idx_reference = idx[~jax_eligible]
        batch = _facade("batch_lnL_EB") if is_EB else _facade("batch_lnL_TP")
        original_batch = _facade("_ORIGINAL_BATCH_EB") if is_EB else _facade("_ORIGINAL_BATCH_TP")
        reference_likelihood = _facade("lnL_EB") if is_EB else _facade("lnL_TP")
        original_likelihood = _facade("_ORIGINAL_LNL_EB") if is_EB else _facade("_ORIGINAL_LNL_TP")
        if is_EB:
            if R_EB is None or secondary_to_host_ratio is None:
                raise RuntimeError("EB scenario requires evaluated radius and secondary-to-host flux from stellar relations")
        if (_facade("HAS_JAX_TREX") and joint_observations is None and not angular_model
                and baseline_likelihood is None and reference_likelihood is original_likelihood and idx_fast.size):
            def likelihood_call(points, draws):
                def at_points(value):
                    return np.asarray(value)[points] if np.ndim(value) else value

                occultor_args = ((R_EB[draws], None) if is_EB else (R_p[draws],))
                return partial(
                    batch, at_points(time), at_points(flux), at_points(sigma), *occultor_args, eff_period, inc[draws],
                    a_cm[draws] if isinstance(a_cm, np.ndarray) else a_cm,
                    R_host[draws], host_u1[draws], host_u2[draws], exptime_days,
                    ecc_arr=ecc[draws], w_arr=w[draws],
                    companion_fluxratio=companion_fluxratio[draws],
                    companion_is_host=companion_is_host, nsamples=nsamples,
                    bin_integration_days=at_points(bin_integration_days),
                    **(dict(u1_EB=eb_u1[draws], u2_EB=eb_u2[draws],
                            secondary_to_host_flux_ratio=secondary_to_host_ratio[draws]) if is_EB else {}),
                )

            screening = batch is original_batch
            points = slice(None)
            if screening:
                # The points a flat model fits worst carry the signal, so they rule out most draws.
                misfit = np.square((np.asarray(flux)-1.0)/np.asarray(sigma))
                points = np.sort(np.argsort(-misfit)[:max(4, int(_SCREEN_POINT_FRACTION*misfit.size))])
            batch_call = likelihood_call(points, idx_fast)
            if screening:
                try:
                    prepared = batch_call(_prepare_only=True)
                    prepared.prepare(_previous_scenario)
                except Exception as exc:
                    # Speculative setup cannot bypass the normal retry/fallback.
                    host_logger("trex").warning(
                        "TREX %s: overlapped setup failed; retrying through the batch path (%s)",
                        log_name, type(exc).__name__,
                    )
                    prepared = None
            jax_call = partial(_facade("_retry_jax_batch"), batch, *batch_call.args,
                               **batch_call.keywords, _prepared=prepared)

    # RNG, masking, slicing, quadrature, upload and compilation stay on the
    # caller. Accelerator timing and MC evaluation follow the yield.
    drawn = perf_counter()
    yield prepared
    resumed = perf_counter()

    def angular_options(i):
        noise_options = {} if baseline_likelihood is None else {"baseline_likelihood": baseline_likelihood}
        if not angular_model:
            return noise_options
        mh = np.broadcast_to(host_metallicity, (n_draws,))[i]
        av = np.broadcast_to(host_av_mag, (n_draws,))[i]
        first = fixed_host_component if fixed_host_component is not None else stellar_light.prepare_component(
            host_teff_k[i], M_host_arr[i], R_host[i], mh, av)
        options = {"angular_profile": first.angular_profile}
        if is_EB:
            _, temperature = _facade("stellar_relations")(np.array([M_EB[i]]))
            second = stellar_light.prepare_component(temperature[0], M_EB[i], R_EB[i], mh, av)
            options["secondary_angular_profile"] = second.angular_profile
        return {**options, **noise_options}

    if np.any(transit_mask):
        evaluated = False
        if jax_call is not None:
            try:
                values = jax_call()
                if not screening:
                    record_numerical_failure(idx_fast, values, "jax")
                if screening:
                    def full_grid(draws):
                        call = likelihood_call(slice(None), idx_fast[draws])
                        full = _facade("_retry_jax_batch")(batch, *call.args, **call.keywords)
                        record_numerical_failure(idx_fast[draws], full, "jax-full-grid")
                        return full

                    values = _screened_lnL(values, log_weight[idx_fast], full_grid)
                lnL[idx_fast] = values
                evaluated = True
            except Exception as exc:
                kind = "EB" if is_EB else "TP"
                _facade("_warn_jax_batch_fallback")(kind, exc)
                _facade("_record_jax_batch_fallback")(jax_fallback_out, kind, idx_fast.size, exc)
        for i in (idx_reference if evaluated else idx):
            a_cm_val = float(a_cm[i]) if isinstance(a_cm, np.ndarray) else float(a_cm)
            kind = "EB" if is_EB else "TP"
            error = None
            try:
                if joint_observations is not None:
                    def take(value, index=i):
                        return None if value is None else float(np.broadcast_to(value, (n_draws,))[index])

                    if (bound_companion is not None or background_star is not None) and not dataset_dilutions:
                        raise _facade("TrexEvidenceUnavailableError")("joint population did not retain per-dataset dilution")
                    value, diagnostics = joint_observations.evaluate(
                        host_mass=float(M_host_arr[i]), host_radius=float(R_host[i]),
                        host_teff=take(host_teff_k), host_metallicity=take(host_metallicity),
                        host_av=take(host_av_mag), period_days=eff_period, inc_deg=float(inc[i]),
                        a_cm=a_cm_val, ecc=float(ecc[i]), argp_deg=float(w[i]),
                        planet_radius=None if is_EB else float(R_p[i]),
                        secondary_mass=float(M_EB[i]) if is_EB else None,
                        secondary_radius=float(R_EB[i]) if is_EB else None,
                        secondary_teff=(float(_facade("stellar_relations")(np.array([M_EB[i]]))[1][0])
                                        if is_EB else None),
                        companion_is_host=companion_is_host,
                        return_diagnostics=True,
                        dataset_dilutions=({key: take(value) for key, value in dataset_dilutions[0].items()}
                                           if dataset_dilutions else None))
                    lnL[i] = value
                    if joint_best is None or value > joint_best[0]:
                        joint_best = (value, diagnostics)
                    record_numerical_failure([i], [lnL[i]], f"joint-scalar-{kind}")
                    continue
                occultor_args = ((float(R_EB[i]), None) if is_EB else (float(R_p[i]),))
                lnL[i] = _facade(f"lnL_{kind}")(
                    time, flux, sigma, *occultor_args,
                    eff_period, float(inc[i]), a_cm_val,
                    float(R_host[i]), float(host_u1[i]), float(host_u2[i]), exptime_days,
                    ecc=float(ecc[i]), argp_deg=float(w[i]),
                    companion_fluxratio=float(companion_fluxratio[i]),
                    companion_is_host=companion_is_host,
                    nsamples=nsamples,
                    bin_integration_days=bin_integration_days,
                    **(dict(u1_EB=float(eb_u1[i]), u2_EB=float(eb_u2[i]),
                            secondary_to_host_flux_ratio=float(secondary_to_host_ratio[i])) if is_EB else {}),
                    **angular_options(i),
                )
            except NumericalLikelihoodError as exc:
                error = exc
                lnL[i] = np.nan
            record_numerical_failure([i], [lnL[i]], f"scalar-{kind}", error)

    # Evidence = log(mean_i(exp(lnL_i + lnprior_i))). The companion occurrence
    # prior varies draw by draw for the unresolved-companion scenarios, so it
    # belongs inside the expectation; for a constant prior this reduces exactly
    # to adding the scalar afterwards.
    # The proposal weights complete the change of measure: the draws came from
    # q, not from the prior, so each carries ln(p/q) alongside its likelihood
    # and occurrence prior. With both terms zero this reduces exactly to the
    # prior-sampling estimator, which is what a scenario with no usable
    # proposal window falls back to.
    failure_count = int(np.count_nonzero(numerical_failures))
    if failure_count and representative_fit_out is None:
        raise NumericalLikelihoodError(f"{ldc_scenario}: {failure_count} supported likelihood evaluations failed", failure_count)
    logw = lnL + log_weight
    observed_logw = logw[~numerical_failures] if failure_count else logw
    if ess_out is not None:
        ess_out.append(_facade("_log_weight_ess")(observed_logw))
    log_evidence = float(_facade("_log_mean_exp")(observed_logw, N_total=observed_logw.size))
    if failure_count and observed_logw.size and math.isfinite(log_evidence):
        # Sum only computed terms but retain the original draw denominator; this is not a completed evidence.
        log_evidence += math.log(observed_logw.size / n_draws)
    if representative_fit_out is not None:
        finite_indices = np.flatnonzero(np.isfinite(lnL))
        if finite_indices.size > 0:
            best_idx = int(finite_indices[int(np.argmax(lnL[finite_indices]))])
            b_conjunction = float(b[best_idx] * eccentric_factor[best_idx])
            representative_draw: Optional[Dict[str, Any]] = {
                "origin": "monte-carlo-max-lnL",
                "draw_index": best_idx,
                "max_lnL": float(lnL[best_idx]),
                "draw_log_weight": float(logw[best_idx]),
                "M_s": float(M_host_arr[best_idx]),
                **({"joint_dataset_diagnostics": joint_best[1]} if joint_best is not None else {}),
                "R_s": float(R_host[best_idx]),
                "P_orb": float(eff_period),
                "a_cm": float(a_cm[best_idx]) if isinstance(a_cm, np.ndarray) else float(a_cm),
                "inc": float(inc[best_idx]),
                "b": b_conjunction,
                "ecc": float(ecc[best_idx]),
                "w": float(w[best_idx]),
                "R_p": 0.0 if is_EB else float(R_p[best_idx]),
                "M_EB": float(M_EB[best_idx]) if is_EB else 0.0,
                "R_EB": (
                    float(R_EB[best_idx])
                    if is_EB
                    else 0.0
                ),
                "u1": float(host_u1[best_idx]),
                "u2": float(host_u2[best_idx]),
                "u1_EB": float(eb_u1[best_idx]) if is_EB else None,
                "u2_EB": float(eb_u2[best_idx]) if is_EB else None,
                **({"secondary_to_host_flux_ratio": float(secondary_to_host_ratio[best_idx])} if is_EB else {}),
                "fluxratio_comp": float(companion_fluxratio[best_idx]),
                "star_num": 2 if companion_is_host else 1,
                "companion_is_host": bool(companion_is_host),
                "is_planet": bool(is_planet),
                "is_EB": bool(is_EB),
                "use_2xP": bool(use_2xP),
                "period_multiplier": 2 if use_2xP else 1,
            }
            if stellar_light is not None:
                representative_draw["stellar_light_model"] = stellar_light.model_id
                representative_draw["host_atmosphere"] = {
                    "teff_k": float(host_teff_k[best_idx]),
                    "mh_dex": float(np.broadcast_to(host_metallicity, (n_draws,))[best_idx]),
                    "av_mag": float(np.broadcast_to(host_av_mag, (n_draws,))[best_idx]),
                }
                if angular_model:
                    profiles = angular_options(best_idx)
                    representative_draw["angular_profiles"] = {
                        name: [np.asarray(part).tolist() for part in profile]
                        for name, profile in profiles.items() if name.endswith("angular_profile")}
        else:
            representative_draw = None
        fit_record = {
                "status": ("unresolved-numerical" if failure_count else
                           "ok" if representative_draw is not None else "no-finite-likelihood-draws"),
                **({"reason": f"{failure_count} supported likelihood evaluations failed numerically"}
                   if failure_count else
                   {"reason": "zero finite sampled weight; prior support is not certified empty"}
                   if not math.isfinite(log_evidence) else {}),
                "unresolved_numerical_failures": failure_count,
                "numerical_failure_examples": numerical_examples,
                "origin": "monte-carlo-max-lnL" if representative_draw is not None else None,
                "has_representative_draw": representative_draw is not None,
                "representative_draw": representative_draw,
                "is_planet": bool(is_planet),
                "is_EB": bool(is_EB),
                "use_2xP": bool(use_2xP),
                "period_multiplier": 2 if use_2xP else 1,
                "companion_is_host": bool(companion_is_host),
                "star_num": 2 if companion_is_host else 1,
                "P_orb": float(eff_period),
                "mc_ln_evidence": float(log_evidence),
                # First-stage ESS stays with the first-stage evidence: a later stage
                # replaces both in the engine's arrays, and the acceptance check
                # compares against this value, not the refined one.
                "mc_effective_sample_size": float(ess_out[-1]) if ess_out else None,
                "n_draws": int(n_draws),
                "transiting_draws": int(np.count_nonzero(transit_mask)),
                "finite_likelihood_draws": int(finite_indices.size),
            }
        if host_distribution is not None:
            fit_record["stellar_parameter_treatment"] = {
                "method": host_distribution["method"], "marginalized": True,
                "representation": "continuous-marginals" if measure.marginals is not None else "released-grid",
                "stellar_dimensions": measure.n_dimensions,
                "display_or_grid_nodes": len(host_distribution["weights"]),
                "limitations": host_distribution["limitations"],
            }
        representative_fit_out.append(fit_record)
    if capture_pilot and not failure_count and math.isfinite(log_evidence) and ess_out and (
        not math.isfinite(ess_out[-1]) or ess_out[-1] < _facade("MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE")
    ):
        from trex.evidence.mc_importance import make_centres

        centres = make_centres(
            logw, occultor_quantile[0], x_ecc, x_w, x_inc, band_low,
            population_variates, background_population if background_star is not None else None,
            seed=pilot_seed.get(), scenario=(ldc_scenario, is_planet, use_2xP),
            **({"host_quantile": host_quantile} if host_quantile is not None else {}),
        )
        if centres is not None:
            importance_centres_out.append(centres.tolist())
    # One measurement feeds both the log line and the retained per-scenario
    # record, so a cost claim never cites two differently stopped clocks.
    phase_seconds = {
        "draw": drawn - started,
        "wait": resumed - drawn,
        "evaluate": perf_counter() - resumed,
    }
    if representative_fit_out is not None:
        fit_record["phase_seconds"] = phase_seconds
    ess = float(ess_out[-1]) if ess_out else None
    host_logger("trex").info(
        "TREX scenario %s: draws=%d, ESS=%s, lnZ=%r, draw=%.2fs, wait=%.2fs, evaluate=%.2fs",
        log_name, n_draws, ess, log_evidence,
        phase_seconds["draw"], phase_seconds["wait"], phase_seconds["evaluate"],
        extra={"event": "trex.scenario", "scenario": log_name, "n_draws": int(n_draws),
               "ess": ess, "ln_evidence": log_evidence, "draw_seconds": phase_seconds["draw"],
               "wait_seconds": phase_seconds["wait"],
               "evaluate_seconds": phase_seconds["evaluate"]},
    )
    return log_evidence
