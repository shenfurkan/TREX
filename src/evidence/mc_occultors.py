"""Occultor-prior draw stages; geometry and likelihood stay in mc_scenario."""
from __future__ import annotations
import numpy as np

from trex.populations.priors import MOE2017_Q_TWIN_THRESHOLD


def _facade(name):
    """Look the name up on ``marginal_likelihoods``, the module tests patch."""
    from trex.evidence import marginal_likelihoods

    return getattr(marginal_likelihoods, name)


def _distinct_rows(*columns):
    """Distinct rows and each draw's row: np.unique(axis=0) without sorting constant columns."""
    values = np.column_stack(columns)
    if not np.all(np.isfinite(values)) or np.any((values == 0.0) & np.signbit(values)):
        return np.unique(values, axis=0, return_inverse=True)
    keys = [values[:, i] for i in range(values.shape[1]-1, -1, -1) if not np.all(values[:, i] == values[0, i])]
    if not keys:
        return values[:1], np.zeros(len(values), dtype=np.intp)
    order = np.argsort(keys[0]) if len(keys) == 1 else np.lexsort(keys)
    ordered = values[order]
    first = np.empty(len(values), dtype=bool)
    first[0] = True
    first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(values), dtype=np.intp)
    inverse[order] = np.cumsum(first) - 1
    return ordered[first], inverse


def draw_eb_occultor(M_host_arr, R_host, companion_fluxratio, companion_is_host, host_teff_k, imaging_constraints, ldc_provenance, ldc_scenario, lnprior_arr, n_draws, occultor_rs_center, occultor_rs_high, occultor_rs_low, rng, support_state_out, target_feh, use_2xP, *, stellar_light=None, host_av_mag=None, quantile_out=None):
    """Draw secondaries within the EB/twin branch, retaining its full-prior mass.

    The fourth return is F_secondary/F_primary, not a fraction. Occultor log
    weights include the branch's original probability, not a renormalized q
    prior; pilot quantiles remain coordinates of the original full-q prior.
    """
    if target_feh is not None:
        target_feh = _facade("_per_draw_values")(target_feh, n_draws, "EB host metallicity")
    # The eclipsing secondary must itself remain a star, so the mass-ratio
    # support floor follows each draw's own host mass. That matters when
    # the pair orbits an unresolved companion, whose mass varies per draw.
    # A host too light to carry any stellar secondary hosts no eclipsing
    # pair at all: those draws are marked impossible rather than being
    # given a fabricated secondary or dropped from the sample.
    eb_supported = M_host_arr >= _facade("MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR")

    def _eb_radius_ratio_sampler(supported, host_mass, host_radius, mass_ratio):
        """Secondary-to-host radius ratio for a mass-ratio quantile, per draw.

        This evaluates the prior endpoints for support diagnostics. Window
        bounds instead use the running-maximum radius-to-mass inverse and the
        same per-host mass-ratio law. It is not a likelihood evaluation.
        The mass-ratio prior's own floor sits on the mass-radius relation's
        domain edge, so a probe quantile can land a hair below it in
        binary64. Such a secondary has no retained radius, and for ordering
        purposes it is smaller than every secondary that does have one, so
        it takes ratio zero rather than extrapolating the relation or
        failing the search. No draw's likelihood is evaluated here.
        """
        supported_index = np.flatnonzero(supported)
        supported_mass = host_mass[supported]

        def ratio_of(variates: np.ndarray) -> np.ndarray:
            ratio = np.zeros(supported.size, dtype=float)
            if mass_ratio is not None:
                secondary_mass = mass_ratio(variates[supported]) * supported_mass
                in_domain = (
                    (secondary_mass >= _facade("STELLAR_RELATIONS_MASS_MIN_SOLAR"))
                    & (secondary_mass <= _facade("STELLAR_RELATIONS_MASS_MAX_SOLAR"))
                )
                if np.any(in_domain):
                    index = supported_index[in_domain]
                    ratio[index] = (
                        _facade("stellar_radii_in_domain")(secondary_mass[in_domain]) / host_radius[index]
                    )
            return ratio

        return ratio_of

    window_requested = occultor_rs_center is not None and np.any(eb_supported)
    if window_requested:
        rows, inverse = _distinct_rows(M_host_arr, R_host, occultor_rs_low, occultor_rs_high)
    else:
        rows, inverse = _distinct_rows(M_host_arr, R_host)
    inverse = inverse.ravel()
    rows_supported = rows[:, 0] >= _facade("MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR")
    mass_ratio = _facade("host_mass_ratio_inverse_cdf")(rows[rows_supported, 0]) if np.any(rows_supported) else None
    branch_low, branch_high = np.zeros(len(rows)), np.zeros(len(rows))
    if mass_ratio is not None:
        # Giacalone et al. (2021), AJ 161, 24, §2.4.4, DOI 10.3847/1538-3881/abc6af:
        # the other q branch has identically zero likelihood, but its prior mass must not be reassigned.
        split = mass_ratio.cdf(np.full(np.count_nonzero(rows_supported), MOE2017_Q_TWIN_THRESHOLD))
        if use_2xP:
            branch_low[rows_supported], branch_high[rows_supported] = split, 1.0
        else:
            branch_high[rows_supported] = split
    broad_low, broad_high = branch_low[inverse], branch_high[inverse]

    if window_requested:
        def window_quantile(ratio):
            if np.any(np.isnan(ratio)):
                raise ValueError("EB proposal radius bounds must not be NaN")
            quantile = np.where(ratio > 0.0, 1.0, 0.0)
            if mass_ratio is not None:
                supported_ratio = ratio[rows_supported]
                positive = (supported_ratio > 0.0) & np.isfinite(supported_ratio)
                # An unbounded upper radius has CDF=1; evaluate its finite q=1 endpoint.
                secondary = rows[rows_supported, 0].copy()
                secondary[positive] = _facade("stellar_masses_of_radii")(
                    supported_ratio[positive] * rows[rows_supported, 1][positive]
                )
                if not np.all(np.isfinite(secondary[positive])):
                    raise ValueError("EB proposal radius inversion returned nonfinite masses")
                supported_quantile = mass_ratio.cdf(secondary / rows[rows_supported, 0])
                quantile[rows_supported] = np.where(supported_ratio > 0.0, supported_quantile, 0.0)
            return _facade("_window_quantile")(quantile)[inverse]

        bounds = [window_quantile(rows[:, column]) for column in (2, 3)]
        x_q, occultor_log_weight = _facade("_defensive_window_draws")(
            rng, n_draws, *bounds, broad_low=broad_low, broad_high=broad_high)
    else:
        width = broad_high - broad_low
        x_q = broad_low + width * rng.random(n_draws)
        with np.errstate(divide="ignore"):
            occultor_log_weight = np.log(width)

    if quantile_out is not None:
        quantile_out.append(x_q)

    if support_state_out is not None:
        if occultor_rs_center is not None and np.any(eb_supported):
            distinct = _eb_radius_ratio_sampler(rows_supported, rows[:, 0], rows[:, 1], mass_ratio)
            support_state_out.append(
                _facade("_proposal_support_state")(
                    occultor_rs_low,
                    occultor_rs_high,
                    distinct(branch_low)[inverse],
                    distinct(np.minimum(branch_high, _facade("_SUPPORT_ENDPOINT_QUANTILE")))[inverse],
                    eb_supported & (broad_high > broad_low),
                )
            )
        else:
            support_state_out.append(
                {"state": "not-requested", "draws_considered": 0}
            )

    q = np.zeros(n_draws, dtype=float)
    if np.any(eb_supported):
        q[eb_supported] = _facade("sample_q_host_mass_ratio")(
            x_q[eb_supported], M_host_arr[eb_supported]
        )
    # G21 §2.4.4: branch masks partition the original q prior; N is unchanged.
    eb_supported[eb_supported] = _facade("eb_mass_ratio_support")(q[eb_supported], use_2xP)
    M_EB = np.where(eb_supported, q * M_host_arr, M_host_arr)
    M_total = (M_host_arr + M_EB) * _facade("Msun")
    if host_teff_k is None:
        _, teff_host = _facade("stellar_relations")(M_host_arr)
    else:
        # A simulated background host carries its own catalog temperature;
        # a main-sequence relation would silently overwrite it.
        teff_host = np.asarray(host_teff_k, dtype=float)
    # Stellar radii and temperatures determine the physical TESS-band
    # component fluxes, rather than assuming equal surface brightness.
    # They are evaluated only for supported draws: an unsupported host can
    # lie below the relation's mass domain (e.g. a TRILEGAL background
    # star), and its slots hold the host's own values purely to stay
    # finite -- those draws carry -inf prior and never enter the likelihood.
    R_EB = np.array(np.broadcast_to(R_host, (n_draws,)), dtype=float)
    teff_EB = np.array(np.broadcast_to(teff_host, (n_draws,)), dtype=float)
    if np.any(eb_supported):
        R_EB[eb_supported], teff_EB[eb_supported] = _facade("stellar_relations")(M_EB[eb_supported])
    # The occultation removes secondary light, so use its own atmosphere.
    # Unsupported slots never reach either forward-model backend.
    eb_u1 = np.full(n_draws, np.nan, dtype=float)
    eb_u2 = np.full(n_draws, np.nan, dtype=float)
    if np.any(eb_supported) and stellar_light is None:
        if target_feh is None:
            raise ValueError("EB secondary limb darkening requires candidate target_feh")
        # A physical binary shares its host's composition; unrelated background pairs do not share the target's.
        secondary_metallicity = target_feh[eb_supported]
        eb_u1[eb_supported], eb_u2[eb_supported] = _facade("_host_grid_coefficients")(
            teff_EB[eb_supported], M_EB[eb_supported], R_EB[eb_supported], secondary_metallicity,
            scenario=ldc_scenario + "-eclipsed-secondary", provenance=ldc_provenance,
        )
    if stellar_light is None:
        EB_flux_ratio = (
            (R_EB / R_host) ** 2
            * _facade("tess_surface_brightness_ratio")(teff_EB, teff_host)
        )
    else:
        EB_flux_ratio = np.full(n_draws, np.nan)
    if stellar_light is not None and np.any(eb_supported):
        from trex.stellar_light.adapters import binary_light

        ratio, _, secondary = binary_light(
            stellar_light, teff_host, M_host_arr, R_host, teff_EB, M_EB, R_EB,
            target_feh, host_av_mag, eb_supported,
        )
        EB_flux_ratio[eb_supported] = ratio
        eb_u1[eb_supported], eb_u2[eb_supported] = secondary
    # Unsupported slots have no physical secondary (NaN with stellar_light) and get -inf just below.
    lnprior_arr = _facade("binary_imaging_log_prior")(
        lnprior_arr, imaging_constraints, np.where(eb_supported, EB_flux_ratio, 1.0), companion_is_host,
        secondary_to_host=True,
    )
    lnprior_arr = np.where(eb_supported, lnprior_arr, -np.inf)
    R_p = np.zeros(n_draws, dtype=float)
    return eb_supported, M_total, R_EB, EB_flux_ratio, eb_u1, eb_u2, lnprior_arr, R_p, M_EB, occultor_log_weight


def draw_planet_occultor(M_host_arr, R_host, n_draws, occultor_rs_center, occultor_rs_high, occultor_rs_low, rng, support_state_out, use_flat_priors, *, quantile_out=None):
    eb_supported = None
    M_total = M_host_arr * _facade("Msun")
    R_EB = None
    EB_fluxratio_arr = None

    def _planet_radius(variates: np.ndarray) -> np.ndarray:
        return _facade("sample_rp")(variates, M_host_arr, flatpriors=use_flat_priors)

    if occultor_rs_center is not None:
        radius_per_ratio = R_host * _facade("Rsun") / _facade("Rearth")
        # Draws sharing a host and window share their inverse bounds.
        rows, inverse = _distinct_rows(M_host_arr, occultor_rs_low * radius_per_ratio, occultor_rs_high * radius_per_ratio)
        bounds = [
            _facade("_window_quantile")(_facade("radius_quantile")(rows[:, column], rows[:, 0], use_flat_priors))[inverse.ravel()]
            for column in (1, 2)
        ]
        x_rp, occultor_log_weight = _facade("_defensive_window_draws")(rng, n_draws, *bounds)
    else:
        x_rp = rng.random(n_draws)
        occultor_log_weight = np.zeros(n_draws, dtype=float)
    if quantile_out is not None:
        quantile_out.append(x_rp)
    R_p = _planet_radius(x_rp)

    if support_state_out is not None:
        if occultor_rs_center is not None:
            def support_radius(variate):
                draws = np.full(len(rows), variate, dtype=float)
                return _facade("sample_rp")(draws, rows[:, 0], flatpriors=use_flat_priors)[inverse.ravel()]

            support_state_out.append(
                _facade("_proposal_support_state")(
                    occultor_rs_low,
                    occultor_rs_high,
                    support_radius(0.0) / radius_per_ratio,
                    support_radius(_facade("_SUPPORT_ENDPOINT_QUANTILE")) / radius_per_ratio,
                    np.ones(n_draws, dtype=bool),
                )
            )
        else:
            support_state_out.append(
                {"state": "not-requested", "draws_considered": 0}
            )
    return eb_supported, M_total, R_EB, EB_fluxratio_arr, R_p, occultor_log_weight
