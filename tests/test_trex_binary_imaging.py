"""Pin the bound-binary imaging constraint against G21's combined-light imaging law, without evidence runs.

Source: Giacalone et al. 2021, AJ 161, 24, DOI 10.3847/1538-3881/abc6af,
Sect. 2.4.3. TRICERATOPS 1.0.20 (``marginal_likelihoods.py``, SEB and BEB
branches) adds component and target fluxes before taking the magnitude. The
independent oracle below adds physical fluxes in the two resolved image sources.
All stars, separations [arcsec], and constant contrast curves [mag] are synthetic.
"""

import math

import numpy as np
import pytest

from trex.evidence import marginal_likelihoods as ml
from trex.evidence import nested_evidence as ne
from trex.stellar.stellar_relations import stellar_relations
from trex.stellar_light.passband import tess_surface_brightness_ratio


@pytest.mark.parametrize("use_2xp", [False, True])
@pytest.mark.parametrize("family", ["SEB", "BEB", "DEB", "PEB"])
@pytest.mark.parametrize("estimator", ["mc", "nested"])
def test_eb_imaging_weights_use_the_two_image_sources(monkeypatch, family, use_2xp, estimator):
    """A branch-supported EB crosses the contrast threshold in the right direction."""
    hosted = family in {"SEB", "BEB"}
    radius, temperature = stellar_relations(np.array([0.5, 1.0]))
    host_ratio = float(radius[0] ** 2 * tess_surface_brightness_ratio(temperature[0], temperature[1]))
    eb_q = 1.0 if use_2xp else 0.9
    secondary_radius, secondary_teff = stellar_relations(np.array([eb_q * (0.5 if hosted else 1.0)]))
    eb_ratio = float(secondary_radius[0] ** 2 * tess_surface_brightness_ratio(secondary_teff[0], temperature[1]))
    # G21 §2.4.3: image-source contrast is -2.5 log10(F_other/F_target) [mag].
    combined_ratio = host_ratio + eb_ratio if hosted else host_ratio / (1.0 + eb_ratio)
    host_only_mag = -2.5 * math.log10(host_ratio)
    combined_mag = -2.5 * math.log10(combined_ratio)
    curve_mag = (host_only_mag + combined_mag) / 2.0

    monkeypatch.setattr(ml, "sample_companion_log_period", lambda x, _mass: np.full(x.size, 3.0))
    monkeypatch.setattr(ml, "sample_q_companion", lambda x, *_args: np.full(x.size, 0.5))
    monkeypatch.setattr(ml, "companion_log_separation_au", lambda _mass, x: np.full(x.size, 0.0))
    monkeypatch.setattr(ml, "sample_q_host_mass_ratio", lambda x, _mass: np.full(x.size, eb_q))
    monkeypatch.setattr(ml, "sample_ecc", lambda x, **_kwargs: np.zeros(x.size))
    monkeypatch.setattr(ml, "sample_inc", lambda x: np.full(x.size, 90.0))
    monkeypatch.setattr(ml, "_host_grid_coefficients", lambda *args, **kwargs: (np.full(1, 0.3), np.full(1, 0.2)))
    monkeypatch.setattr(ml, "lnL_EB", lambda *args, **kwargs: -1.0)
    for name in ("sample_companion_log_period", "sample_q_companion", "companion_log_separation_au",
                 "sample_q_host_mass_ratio", "sample_ecc", "sample_inc", "_host_grid_coefficients", "lnL_EB"):
        monkeypatch.setattr(ne, name, getattr(ml, name))
    options = (
        {"bound_companion": "secondary" if hosted else "primary", "plx_mas": 1000.0}
        if family in {"SEB", "PEB"} else {
            "background_star": "background" if hosted else "target",
            "background_area_sq_deg": 0.1,
            "target_tess_mag": 10.0,
            "background_population": {
                "tess_mag": np.array([10.0 + host_only_mag]),
                "mass_solar": np.array([0.5]),
                "radius_solar": radius[:1],
                "teff_k": temperature[:1],
                "mh_dex": np.zeros(1),
            },
        }
    )
    if estimator == "nested":
        curve = dict(contrast_separations=np.array([0.0001, 2.2]), contrast_values=np.full(2, curve_mag))
        population = (ne.bound_companion_population(1.0, 1.0, 1000.0, **curve)
                      if family in {"SEB", "PEB"} else ne.background_star_population(
                          options["background_population"], 0.1, 10.0, **curve))
        integrand = ne._EclipsingPairIntegrand(
            time=np.array([0.0]), flux=np.array([1.0]), sigma=0.001,
            effective_period=6.0 if use_2xp else 3.0, host_mass=1.0, host_radius=1.0,
            use_2xP=use_2xp,
            u1=0.3, u2=0.2, exptime_days=0.001, bin_integration_days=0.0,
            nsamples=20, population=population, n_dimensions=4 + population.n_dimensions, hosted=hosted, target_feh=0.0,
        )
        result = integrand.log_likelihood(integrand.prior_transform(np.full(integrand.n_dimensions, 0.5)))
    else:
        result = ml._eval_scenario(
            time=np.array([0.0]), flux=np.array([1.0]), sigma=0.001,
            period_days=3.0, M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.3, u2=0.2,
            n_draws=1, is_planet=False, is_EB=True, use_2xP=use_2xp,
            exptime_days=0.001, target_feh=0.0, rng=np.random.default_rng(17),
            contrast_separations=np.array([0.0001, 2.2]),
            contrast_values=np.full(2, curve_mag), **options,
        )

    assert bool(np.isneginf(result)) == (combined_mag < curve_mag)


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_nested_population_imaging_matches_mc_on_identical_unit_draws(bound, hosted):
    """Float64 imaging weights agree exactly, including uncovered and twin draws."""
    from trex.populations.companion_imaging import binary_imaging_log_prior

    count, seed = 256, 20261027
    curve = dict(contrast_separations=np.array([0.05, 0.5, 2.2]), contrast_values=np.array([0.0, 3.0, 6.0]))
    rng = np.random.default_rng(seed)
    if bound:
        block = ne.bound_companion_population(1.0, 1.0, 50.0, **curve)
        points = np.column_stack((rng.random(count), rng.random(count), rng.random(count)))
        constraints = []
        fraction, prior, masses, radii, temperatures = ml._draw_bound_companion(
            np.random.default_rng(seed), count, 1.0, 1.0, 50.0, **curve, imaging_state_out=constraints)
    else:
        radii, temperatures = stellar_relations(np.array([0.3, 0.5, 1.0]))
        population = dict(tess_mag=np.array([12.0, 14.0, 16.0]), mass_solar=np.array([0.3, 0.5, 1.0]),
                          radius_solar=radii, teff_k=temperatures, mh_dex=np.zeros(3))
        block = ne.background_star_population(population, 0.1, 10.0, **curve)
        indices, separations = rng.integers(0, 3, count), rng.random(count)
        points = np.column_stack(((indices + 0.5) / 3, separations))
        constraints = []
        fraction, prior, masses, radii, temperatures = ml._draw_background_star(
            np.random.default_rng(seed), count, population, 0.1, 10.0, **curve, imaging_state_out=constraints)
    host_mass = masses if hosted else np.ones(count)
    host_radius = radii if hosted else np.ones(count)
    host_teff = temperatures if hosted else stellar_relations(np.ones(count))[1]
    supported = host_mass >= ne.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
    q_unit = rng.random(count)
    q = ne.sample_q_host_mass_ratio(q_unit[supported], host_mass[supported])
    secondary_radius, secondary_teff = stellar_relations(q * host_mass[supported])
    # Production EB component flux (G21 §2.4.3), converted to target-flux units.
    secondary = np.zeros(count)
    secondary[supported] = (secondary_radius / host_radius[supported]) ** 2 * tess_surface_brightness_ratio(
        secondary_teff, host_teff[supported])
    if hosted:
        secondary *= fraction / (1.0 - fraction)
    expected = binary_imaging_log_prior(prior, constraints, secondary, hosted)
    expected[~supported] = -np.inf
    actual = []
    for point, secondary_flux, host_supported in zip(points, secondary, supported, strict=True):
        draw = block.resolve(point, defer_imaging=True)
        actual.append(float(draw.imaging.apply(block.log_occurrence_weight, secondary_flux, hosted)[0])
                      if draw.supported and host_supported else -np.inf)
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float64), expected)
