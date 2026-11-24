"""Monte Carlo draws from unresolved stellar populations."""
from __future__ import annotations
import math
import warnings
from typing import Mapping, Optional, Tuple
import numpy as np
from trex.foundation.constants import G, Msun, Rsun
from trex.foundation.physical_constants import MILLIARCSECONDS_PER_ARCSECOND, POGSON_FLUX_EXPONENT
from trex.populations._parameters import (
    MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS,
    MOE2017_LOGPERIOD_MAX_DAYS,
    WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR,
)
from trex.populations.companion_imaging import imaging_log_prior
from trex.populations.companion_priors import (
    _companion_log_period_from_log_separation,
    companion_hierarchical_stability_supported,
    companion_log_separation_au,
    companion_population_frequency,
    sample_companion_log_period,
    sample_companion_log_semimajor_axis,
    sample_q_companion,
)
from trex.populations.priors import background_population_expectation
from trex.stellar.stellar_relations import stellar_relations
from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid
from trex.stellar_light.passband import tess_surface_brightness_ratio
from trex.evidence.nested_controls import _validate_scalar
from trex.evidence.nested_population import _angular_constraint
from trex.evidence.mc_proposals import _per_draw_values
from trex.populations.mass_ratio_priors import _stellar_mass_ratio_floor


def _host_grid_coefficients(
    teff_k: np.ndarray,
    mass_solar: np.ndarray,
    radius_solar: np.ndarray,
    metallicity_dex,
    scenario: str = "host",
    provenance: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-draw TESS limb darkening from the published reference model.

    ``teff_k``, mass, and radius belong to the actual host draw. Surface gravity
    is derived from those retained physical values with the same IAU/CODATA CGS
    constants used by TREX orbital calculations. Coefficients follow the vendored
    TRICERATOPS 1.0.20 reference inside its lookup bounds; a draw outside them
    that lands in a complete published Claret (2017) ATLAS or eligible cool
    solar PHOENIX-COND cell takes interpolated coefficients instead of a
    main-sequence-like nearest node. Everything else keeps the reference clamp. An explicit
    stellar-light model never comes here. When ``provenance`` is supplied, one
    auditable event per scenario separates the interpolated draw count from the
    draws that still needed the nearest-node clamp, and names the nodes and
    metallicity plane actually used.
    """
    temperature = np.asarray(teff_k, dtype=float)
    mass = np.asarray(mass_solar, dtype=float)
    radius = np.asarray(radius_solar, dtype=float)
    if temperature.shape != mass.shape or mass.shape != radius.shape:
        raise ValueError("host atmosphere arrays must have matching per-draw shapes")
    if (
        not np.all(np.isfinite(temperature))
        or not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(radius))
        or np.any(temperature <= 0.0)
        or np.any(mass <= 0.0)
        or np.any(radius <= 0.0)
    ):
        raise ValueError("host atmosphere arrays must be finite and positive")
    if np.asarray(metallicity_dex).dtype.kind == "b":
        raise ValueError("host metallicity must be finite numeric values, not booleans")
    metallicity = np.asarray(metallicity_dex, dtype=float)
    if metallicity.ndim != 0 and metallicity.shape != temperature.shape:
        raise ValueError("host metallicity must be scalar or match the per-draw shape")
    if not np.all(np.isfinite(metallicity)):
        raise ValueError("host metallicity must be finite for host-specific limb darkening")
    logg_cgs = np.log10(G * mass * Msun / np.square(radius * Rsun))
    grid = load_triceratops_tess_grid()
    cache = getattr(grid, "_cached_single_coefficients", None)
    if provenance is None and temperature.shape == (1,) and metallicity.ndim == 0 and cache is not None:
        u1, u2, clamped_value = cache(float(temperature[0]), float(logg_cgs[0]), float(metallicity))
        clamped = np.array([clamped_value])
        selections = None
    else:
        selections = grid.select_many(temperature, logg_cgs, metallicity)
        clamped = selections.boundary_clamped
    if np.any(clamped):
        # One constant line per process, not one per draw. The per-draw
        # numbers (used teff, used logg, counts) are appended to ``provenance``
        # below and reach the candidate's JSON artifact in full; varying the
        # console text would defeat the stdlib warning de-duplication and
        # flood a multi-seed run with dozens of identical blocks.
        warnings.warn(
            "TREX host limb-darkening requests fell outside the TRICERATOPS TESS "
            "grid bounds; the nearest grid nodes were used. Per-draw detail is "
            "recorded in the vetting artifact's limb-darkening provenance.",
            RuntimeWarning,
            stacklevel=2,
        )
    if provenance is not None and temperature.size:
        # Lookup inputs are counted even when every one used the reference grid.
        interpolated = selections.interpolated
        provenance.append(
            {
                "scenario": scenario,
                "clamped_draws": int(np.count_nonzero(clamped)),
                "interpolated_draws": int(np.count_nonzero(interpolated)),
                "reference_draws": int(np.count_nonzero(~clamped & ~interpolated)),
                "atlas_draws": int(np.count_nonzero(selections.atlas_interpolated)),
                "phoenix_draws": int(np.count_nonzero(selections.phoenix_interpolated)),
                "total_draws": int(temperature.size),
                "requested_teff_k_min": float(np.min(temperature)),
                "requested_teff_k_max": float(np.max(temperature)),
                "requested_logg_cgs_min": float(np.min(logg_cgs)),
                "requested_logg_cgs_max": float(np.max(logg_cgs)),
                "used_metallicity_dex": [
                    float(value) for value in np.unique(selections.used_metallicity_dex)
                ],
                "used_nodes": [tuple(node) for node in np.unique(np.column_stack((
                    selections.used_teff_k[clamped], selections.used_logg_cgs[clamped],
                )), axis=0)],
            }
        )
    return (np.array([u1]), np.array([u2])) if selections is None else (selections.u1, selections.u2)


def _bound_orbit(x_period, x_q, primary_mass):
    """Resolve the existing bound prior at one primary mass [Msun]."""
    if primary_mass <= WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR:
        interior = (x_period > 0) & (x_period < 1)
        # Zero-measure Gaussian endpoints remain rejected, never clipped into the prior.
        admitted_period = np.where(interior, x_period, 0.5) if not interior.all() else x_period
        log_axis = sample_companion_log_semimajor_axis(admitted_period, primary_mass)
        companion_mass = sample_q_companion(x_q, primary_mass, None) * primary_mass
        log_period = _companion_log_period_from_log_separation(primary_mass + companion_mass, log_axis)
        supported = interior & np.isfinite(log_period)
    else:
        log_period = sample_companion_log_period(x_period, primary_mass)
        companion_mass = sample_q_companion(x_q, primary_mass, log_period) * primary_mass
        log_axis = companion_log_separation_au(primary_mass + companion_mass, log_period)
        supported = (log_period >= MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS) & (
            log_period <= MOE2017_LOGPERIOD_MAX_DAYS
        )
    return log_axis, log_period, companion_mass, supported, math.log(companion_population_frequency(primary_mass))


def _bound_orbits_by_mass(x_period, x_q, primary_mass):
    if np.ndim(primary_mass) == 0:
        return _bound_orbit(x_period, x_q, primary_mass)
    # Group identical MIST masses without rounding or reshuffling the paired
    # radius and temperature. Each group uses the original fixed-mass prior.
    order = np.argsort(primary_mass, kind="stable")
    sorted_mass = primary_mass[order]
    cuts = np.flatnonzero(sorted_mass[1:] != sorted_mass[:-1]) + 1
    outputs = tuple(np.empty_like(x_period, dtype=dtype) for dtype in (float, float, float, bool, float))
    for rows in np.split(order, cuts):
        mass = float(primary_mass[rows[0]])
        if _stellar_mass_ratio_floor(mass) >= 1.0:
            # A zero-measure stellar-q interval has no companion. Keep these
            # primary draws in the denominator with zero evidence.
            resolved = (-np.inf, -np.inf, mass, False, -np.inf)
        else:
            resolved = _bound_orbit(x_period[rows], x_q[rows], mass)
        for output, values in zip(outputs, resolved, strict=True):
            output[rows] = values
    return outputs


def _positive_bound_input(value, n_draws, name):
    if np.ndim(value) == 0:
        return _validate_scalar(value, name, positive=True)
    values = _per_draw_values(value, n_draws, name)
    if not np.all(np.isfinite(values) & (values > 0.0)):
        raise ValueError(f"{name} must be finite and positive")
    return values


def _draw_bound_companion(
    rng: np.random.Generator,
    n_draws: int,
    M_s_Msun: float | np.ndarray,
    R_s_Rsun: float | np.ndarray,
    plx_mas: float,
    contrast_separations: Optional[np.ndarray],
    contrast_values: Optional[np.ndarray],
    resolution_limit_arcsec: Optional[float] = None,
    imaging_state_out: Optional[list] = None,
    host_teff_k: float | np.ndarray | None = None,
    stellar_light=None,
    target_feh=None,
    variates_out=None,
    observed_period_days: Optional[float] = None,
    stability_hosted: Optional[bool] = None,
    stability_out: Optional[list] = None,
    datasets=None,
    band_light=None,
    dataset_dilutions_out: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw an unresolved bound-companion population for one scenario.

    ``resolution_limit_arcsec`` is the unresolved-scene outer bound
    (Giacalone et al. 2021, Sect. 1). A measured contrast curve is an additional
    detectability constraint only where it has measured support; it does not
    truncate companions inside an uncovered inner or outer interval.

    Follows the generative order of the population itself: the M-dwarf branch
    draws a moment-matched log-normal axis and independent circular-orbit
    projection, and derives the physical period from both masses; the
    higher-mass branch draws the Moe & Di Stefano (2017) Eq. 23 period and
    uses the same independent circular, isotropic sky projection. The
    mass-ratio distribution belonging to that draw supplies the companion
    mass, and the resulting contrast decides whether the imaging would have
    detected it. A draw the curve detects, one wider than the unresolved-scene
    bound, or one Mardling & Aarseth (2001) Eq. 90 marks dynamically unstable
    gets zero prior weight rather than being dropped from the sample.

    Primary M [Msun], R [Rsun], and supplied Teff [K] may be scalars or
    paired length-N arrays. Each draw uses that primary for the occurrence
    weight, orbit, stability and contrast. Parallax [mas] is fixed here.

    ``observed_period_days`` enables the hierarchical-stability mask.  For a
    planet scenario pass ``stability_hosted``: False for the target-hosted
    slot (STP; the companion is the outer body) and True for the
    companion-hosted slot (PTP).  For an eclipsing-pair scenario the inner
    binary's secondary mass is drawn after this population, so the caller
    passes ``stability_out`` instead: it receives ``(log_period, M_comp)``
    and applies the same shared mask once the EB mass ratio exists.

    Returns:
        ``(companion_fluxratio, lnprior, companion_mass, companion_radius,
        companion_teff_k)``, each of shape ``(n_draws,)``. ``companion_teff_k``
        is the same Torres et al. (2010) main-sequence relation value already
        used to form the companion's TESS-band flux ratio here; it is reused
        rather than re-derived when the companion itself hosts the transit.
    """
    _, contrast_separations, contrast_values, unresolved_limit = _angular_constraint(
        contrast_separations, contrast_values, resolution_limit_arcsec, "bound-companion"
    )
    M_s_Msun = _positive_bound_input(M_s_Msun, n_draws, "M_s_Msun")
    R_s_Rsun = _positive_bound_input(R_s_Rsun, n_draws, "R_s_Rsun")
    plx_mas = _validate_scalar(plx_mas, "plx_mas", positive=True)
    if host_teff_k is not None:
        host_teff_k = _positive_bound_input(host_teff_k, n_draws, "host_teff_k")
    if observed_period_days is not None:
        observed_period_days = _validate_scalar(observed_period_days, "observed_period_days", positive=True)

    x_period = rng.random(n_draws)
    x_q = rng.random(n_draws)
    x_projection = rng.random(n_draws)
    if variates_out is not None:
        variates_out.append((x_period, x_q, x_projection))
    log_axis, log_period, M_comp, q_supported, log_occurrence = _bound_orbits_by_mass(x_period, x_q, M_s_Msun)
    # Circular orbit with isotropic orientation/phase: s/a = sqrt(1-u²).
    separation_arcsec = (
        np.power(10.0, log_axis) * np.sqrt((1.0 - x_projection) * (1.0 + x_projection))
        * plx_mas / MILLIARCSECONDS_PER_ARCSECOND
    )
    no_companion = np.isneginf(log_occurrence)
    if np.ndim(log_occurrence) and np.any(no_companion):
        R_comp = _per_draw_values(R_s_Rsun, n_draws, "primary radius").copy()
        teff_comp = _per_draw_values(host_teff_k, n_draws, "primary temperature").copy()
        R_comp[~no_companion], teff_comp[~no_companion] = stellar_relations(M_comp[~no_companion])
    else:
        R_comp, teff_comp = stellar_relations(M_comp)
    # Keep a supplied host temperature [K]; only legacy callers without it use the scenario relation.
    if host_teff_k is None:
        if stellar_light is not None:
            raise ValueError("physical bound light requires the actual target temperature")
        _, teff_primary = stellar_relations(np.broadcast_to(M_s_Msun, (n_draws,)))
    else:
        teff_primary = host_teff_k

    # TESS-band flux of the companion relative to the primary, from the same
    # radius-and-surface-brightness construction the eclipsing-binary branch
    # uses, rather than a separate mass-luminosity power law.
    if stellar_light is None:
        flux_ratio = (R_comp / R_s_Rsun) ** 2 * tess_surface_brightness_ratio(
            teff_comp, teff_primary
        )
    else:
        flux_ratio, _, _ = stellar_light.pair(
            (teff_primary, M_s_Msun, R_s_Rsun), (teff_comp, M_comp, R_comp),
            target_feh, stellar_light.target_av_mag,
        )
    with np.errstate(invalid="ignore"):
        companion_fluxratio = np.where(np.isposinf(flux_ratio), 1.0, flux_ratio / (1.0 + flux_ratio))
    # A companion brighter than the primary would not be an unresolved
    # secondary; an equal-brightness twin is retained, since the excess twin
    # population is part of the modelled distribution. The imaging constraint
    # applies this test to the combined sources once the EB secondary exists.
    retained = q_supported & (separation_arcsec <= unresolved_limit)
    if observed_period_days is not None:
        if stability_hosted is None:
            if stability_out is None:
                raise ValueError(
                    "an eclipsing-pair bound companion must defer its stability "
                    "mask through stability_out"
                )
            stability_out.append((log_period, M_comp))
        else:
            # Planet scenarios: the inner binary is the planet host alone (the
            # planet's mass is not part of any G21 prior and is negligible),
            # so the drawn companion mass decides both branches of the mask.
            inner_binary = M_comp if stability_hosted else M_s_Msun
            outer_mass = M_s_Msun if stability_hosted else M_comp
            if np.all(np.isfinite(log_period)):
                stable = companion_hierarchical_stability_supported(
                    log_period, observed_period_days, inner_binary, outer_mass
                )
            else:
                finite = np.isfinite(log_period)
                stable = np.zeros(n_draws, dtype=bool)
                stable[finite] = companion_hierarchical_stability_supported(
                    log_period[finite], observed_period_days,
                    np.broadcast_to(inner_binary, (n_draws,))[finite],
                    np.broadcast_to(outer_mass, (n_draws,))[finite],
                )
            retained = retained & stable

    lnprior = np.where(retained, log_occurrence, -np.inf)
    lnprior = imaging_log_prior(
        flux_ratio, separation_arcsec, contrast_separations, contrast_values, lnprior,
        require_fainter=True, deferred=imaging_state_out,
    )
    if dataset_dilutions_out is not None:
        # A missing native band is retained as ``None`` for that dataset; the
        # joint consumer refuses the draw instead of substituting TESS.
        per_dataset = {}
        for dataset in datasets or ():
            band = dataset.band if hasattr(dataset, "band") else dataset["band"]
            dataset_id = dataset.dataset_id if hasattr(dataset, "dataset_id") else dataset["dataset_id"]
            light = None if band_light is None else band_light.get(band)
            if light is None and band.strip().lower() == "tess":
                ratio = flux_ratio
            elif light is None:
                per_dataset[dataset_id] = None
                continue
            else:
                target_av = (stellar_light.target_av_mag if stellar_light is not None
                             else getattr(light, "target_av_mag", None))
                ratio, _, _ = light.pair(
                    (teff_primary, M_s_Msun, R_s_Rsun),
                    (teff_comp, M_comp, R_comp),
                    target_feh, target_av,
                )
            ratio = np.asarray(ratio, dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                per_dataset[dataset_id] = np.where(
                    np.isposinf(ratio), 1.0, ratio / (1.0 + ratio)
                )
        dataset_dilutions_out.append(per_dataset)
    return companion_fluxratio, lnprior, M_comp, R_comp, teff_comp


def _draw_background_star(
    rng: np.random.Generator,
    n_draws: int,
    population: Mapping[str, np.ndarray],
    simulated_area_sq_deg: float,
    target_tess_mag,
    contrast_separations: Optional[np.ndarray],
    contrast_values: Optional[np.ndarray],
    resolution_limit_arcsec: Optional[float] = None,
    imaging_state_out: Optional[list] = None,
    metallicity_out: Optional[list] = None,
    extinction_out: Optional[list] = None,
    variates_out=None,
    datasets=None,
    dataset_dilutions_out: Optional[list] = None,
    selection_band: Optional[str] = None,
    selection_magnitude=None,
    completeness: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw unresolved background stars for one scenario.

    Without a measured contrast curve the field is bounded by
    ``resolution_limit_arcsec``, the catalog close-pair resolution beyond which
    a star is a resolved neighbour already in the aperture scene; no imaging
    detection removes any draw inside it.

    Each draw takes a simulated star from the candidate's own retained
    population and places it at a separation drawn uniformly in area out to the
    catalog's unresolved-scene boundary. The star pool is selected once in the
    declared observing band (``selection_band``, defaulting to the first
    dataset's band) at that dataset's target magnitude; an absent band refuses
    rather than reverting to a TESS magnitude. With no datasets or band, the
    legacy TESS path uses ``tess_mag``/``target_tess_mag`` unchanged. The
    contrast curve only excludes detections over its measured separation range;
    outside that range the star remains in the scene prior.
    A draw the imaging would have detected carries no weight; the rest carry the
    expected number of such selected stars inside the field, which is why the
    population query's simulated sky area has to be retained evidence rather
    than assumed. ``completeness`` is recorded by the caller elsewhere; this
    path applies no completeness correction and never assumes one.

    LIMITATION: retained population stars do not carry sky positions in the
    TRILEGAL cache, so an individual cross-match against Gaia neighbors inside
    the unresolved-scene boundary is unavailable. Any overlap is therefore left
    visible as a model limitation rather than hidden by treating the imaging
    edge as a catalog resolution boundary.

    The detection mask is deliberately fail-open inside the imaging's inner
    working angle: ``contrast_limit_at_separation`` returns nan there because
    the curve constrains nothing, and a draw with no limit is retained rather
    than counted as a detection.  Separation is uniform in area, so that
    uncovered disc holds prior measure ``(inner / field radius) ** 2`` which a
    finite draw budget can miss. A ``-inf`` background evidence at a small
    draw budget is therefore a
    statement about the covered separations, not a proof of empty support: an
    estimator that samples the whole prior can find that
    sliver and may return a finite value. Retaining is the conservative
    direction -- it adds false-positive weight rather than removing it -- so
    the mask is not tightened here.

    Returns:
        ``(fluxratio, lnprior, mass_solar, radius_solar, teff_k)``, each of
        shape ``(n_draws,)``. Optional ``metallicity_out`` receives the aligned
        population [M/H] array [dex], without changing this historical tuple or RNG.
    """
    _, contrast_separations, contrast_values, unresolved_limit = _angular_constraint(
        contrast_separations, contrast_values, resolution_limit_arcsec, "background-star"
    )
    simulated_area_sq_deg = _validate_scalar(simulated_area_sq_deg, "simulated_area_sq_deg", positive=True)
    datasets = tuple(datasets or ())
    from trex.populations.background import (
        background_dataset_magnitudes,
        background_metallicity_dex,
        select_background_population,
    )

    selection = select_background_population(
        population,
        datasets=datasets or (),
        band=selection_band,
        depth_mag=selection_magnitude,
        target_tess_mag=target_tess_mag,
        completeness=completeness,
    )
    indices = selection.selected_indices
    star_count = int(indices.size)
    for name in ("mass_solar", "radius_solar", "teff_k"):
        column = np.asarray(population[name], dtype=float)
        if (column.shape != selection.magnitudes.shape
                or not np.isfinite(column[indices]).all()
                or np.any(column[indices] <= 0.0)):
            raise ValueError(
                f"background {name} must be positive, finite and aligned with the selected stars"
            )
    field_radius_arcsec = unresolved_limit
    expectation = background_population_expectation(
        star_count=star_count,
        simulated_area_sq_deg=simulated_area_sq_deg,
        field_radius_arcsec=field_radius_arcsec,
    )

    index = indices[rng.integers(0, star_count, n_draws)]
    if metallicity_out is not None:
        metallicity_out.append(background_metallicity_dex(population)[index])
    if extinction_out is not None:
        from trex.stellar_light.inputs import background_extinction

        extinction_out.append(background_extinction(population)[index])
    delta_mag = selection.magnitudes[index] - selection.depth_mag
    # Pogson: the simulated star's flux relative to the target's.
    with np.errstate(over="ignore", invalid="ignore"):
        flux_ratio = np.power(10.0, POGSON_FLUX_EXPONENT * delta_mag)
        fluxratio = np.where(np.isposinf(flux_ratio), 1.0, flux_ratio / (1.0 + flux_ratio))

    # Uniform in area: the cumulative of p(rho) ∝ rho is (rho / rho_max) ** 2.
    x_separation = rng.random(n_draws)
    separation_arcsec = field_radius_arcsec * np.sqrt(x_separation)
    if variates_out is not None:
        variates_out.append((index, x_separation))
    lnprior = imaging_log_prior(
        flux_ratio, separation_arcsec, contrast_separations, contrast_values,
        np.full(n_draws, math.log(expectation)), deferred=imaging_state_out,
    )
    if dataset_dilutions_out is not None:
        # The same star index and areal position serve every dataset. A native
        # band cell the cache left unusable stays NaN (no support), and a band
        # the population does not retain stays None (no substitution).
        per_dataset = {}
        for dataset in datasets or ():
            band = dataset.band if hasattr(dataset, "band") else dataset["band"]
            dataset_id = dataset.dataset_id if hasattr(dataset, "dataset_id") else dataset["dataset_id"]
            target = (dataset.target_magnitude if hasattr(dataset, "target_magnitude")
                      else dataset.get("target_magnitude"))
            try:
                magnitudes, valid = background_dataset_magnitudes(population, band)
            except ValueError:
                per_dataset[dataset_id] = None
                continue
            if target is None:
                per_dataset[dataset_id] = None
                continue
            delta = magnitudes[index] - float(target)
            with np.errstate(over="ignore", invalid="ignore"):
                ratio = np.power(10.0, POGSON_FLUX_EXPONENT * delta)
                fractions = np.where(np.isposinf(ratio), 1.0, ratio / (1.0 + ratio))
            per_dataset[dataset_id] = np.where(valid[index], fractions, np.nan)
        dataset_dilutions_out.append(per_dataset)
    return (
        fluxratio,
        lnprior,
        np.asarray(population["mass_solar"], dtype=float)[index],
        np.asarray(population["radius_solar"], dtype=float)[index],
        np.asarray(population["teff_k"], dtype=float)[index],
    )
