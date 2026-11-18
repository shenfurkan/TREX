"""The supplied resolved target temperature must survive stellar-flux plumbing."""

import numpy as np
import pytest

from trex.stellar import stellar_relations as trex_stellar
from trex.stellar_light import passband as trex_passband
from trex.evidence import marginal_likelihoods as ml
from trex.evidence import nested_evidence as ne


def test_bound_flux_uses_supplied_host_temperature_without_changing_population():
    args = (32, 1., 1., 50., None, None)
    old = ml._draw_bound_companion(np.random.default_rng(17), *args, resolution_limit_arcsec=1.)
    new = ml._draw_bound_companion(np.random.default_rng(17), *args,
                                  resolution_limit_arcsec=1., host_teff_k=4500.)
    for index in (2, 3, 4):
        np.testing.assert_array_equal(old[index], new[index])
    # Retained Planck law at one wavelength; tests routing, not atmosphere applicability.
    ratio = new[3]**2 * trex_passband.tess_surface_brightness_ratio(new[4], 4500.)
    np.testing.assert_allclose(new[0], ratio / (1. + ratio), rtol=2e-15)
    assert np.any(new[0] != old[0])


def test_mc_preparation_keeps_target_temperature_but_not_for_unrelated_host(monkeypatch):
    from trex.evidence.mc_scenario import _scenario_stages

    monkeypatch.setattr(ml, "HAS_JAX_TREX", False)
    population = dict(tess_mag=np.array([12.]), mass_solar=np.array([1.]),
        radius_solar=np.array([1.]), teff_k=np.array([6000.]), mh_dex=np.array([0.]))
    for hosted in (False, True):
        stages = _scenario_stages(time=np.array([0.]), flux=np.ones(1), sigma=.001,
            period_days=3., M_s_Msun=1., R_s_Rsun=1., u1=.3, u2=.2,
            n_draws=32, is_planet=False, is_EB=True, use_2xP=False, exptime_days=.001,
            target_teff_k=4500., target_feh=0., rng=np.random.default_rng(17),
            **(dict(background_star="background", background_population=population,
                background_area_sq_deg=.1, target_tess_mag=10., resolution_limit_arcsec=1.) if hosted else {}))
        next(stages)  # Draw/prepare only: zero evidence evaluations.
        state = stages.gi_frame.f_locals
        expected_t = 6000. if hosted else 4500.
        ratio = (state["R_EB"] / state["R_host"])**2 * trex_passband.tess_surface_brightness_ratio(
            trex_stellar.stellar_relations(state["M_EB"])[1], expected_t)
        support = state["eb_supported"]
        assert support.any()
        np.testing.assert_allclose(state["secondary_to_host_ratio"][support], ratio[support], rtol=2e-15)
        stages.close()


def test_bound_nested_population_keeps_supplied_host_temperature():
    population = ne.bound_companion_population(1., 1., 50., resolution_limit_arcsec=1.,
                                              host_teff_k=4500.)
    draw = population.resolve(np.array([.3, .3, .5]), defer_imaging=True)
    expected = draw.radius_solar**2 * float(trex_passband.tess_surface_brightness_ratio(draw.teff_k, 4500.))
    assert population.host_teff == 4500.
    assert draw.flux_fraction == pytest.approx(expected / (1. + expected), rel=2e-15)


@pytest.mark.parametrize("population_kind,hosted", [
    ("fixed", False), ("bound", False), ("background", False),
    ("bound", True), ("background", True),
])
def test_scalar_native_eb_keep_supplied_target_temperature(population_kind, hosted):
    from trex.accelerators import _jax_models as models

    population = None
    if population_kind == "bound":
        population = ne.bound_companion_population(1., 1., 50., resolution_limit_arcsec=1.,
                                                  host_teff_k=4500.)
    if population_kind == "background":
        population = ne.background_star_population(dict(tess_mag=np.array([12.]),
            mass_solar=np.array([1.]), radius_solar=np.array([1.]), teff_k=np.array([6000.]),
            mh_dex=np.array([0.])), .1, 10., resolution_limit_arcsec=1.)
    time = np.linspace(-.06, .06, 7)
    reference = ne._EclipsingPairIntegrand(time=time, flux=np.ones(7), sigma=.01,
        effective_period=3., host_mass=1., host_radius=1., host_teff_k=4500.,
        u1=.3, u2=.2, exptime_days=.001, bin_integration_days=0., nsamples=3,
        population=population, n_dimensions=4 if population is None else 4 + population.n_dimensions, target_feh=0., hosted=hosted)
    assert np.isnan(reference.host_teff) if hosted else reference.host_teff == 4500.
    native = reference.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    if population_kind == "fixed":
        assert native.host_teff == 4500.
    elif population_kind == "bound" and not hosted:
        assert native.base.host_teff == 4500.
    point = np.array([.4, .02, .25, .5] + ([] if population is None else [.3] * population.n_dimensions))
    scalar = reference.log_likelihood(reference.prior_transform(point))
    actual = native.evaluate(point)
    assert bool(actual.numerically_valid) and np.isfinite(scalar)
    assert float(actual.log_integrand) == pytest.approx(scalar, rel=1e-4, abs=1e-6)
    if hosted:
        legacy = ne._EclipsingPairIntegrand(time=time, flux=np.ones(7), sigma=.01,
            effective_period=3., host_mass=1., host_radius=1.,
            u1=.3, u2=.2, exptime_days=.001, bin_integration_days=0., nsamples=3,
            population=population, n_dimensions=4 + population.n_dimensions, target_feh=0., hosted=True)
        assert legacy.log_likelihood(legacy.prior_transform(point)) == scalar


def test_escalation_forwards_temperature_through_population_and_each_ensemble_call(tmp_path, monkeypatch):
    from tests.trex.evidence.escalation_support import (
        _NON_NEGLIGIBLE_NESTED_LOG_EVIDENCE,
        _base_kwargs,
        _binding_evidence,
        _result,
    )
    from trex.evidence import evidence_escalation as es

    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return _result(_NON_NEGLIGIBLE_NESTED_LOG_EVIDENCE)

    monkeypatch.setattr(es, "eb_scenario_nested_evidence", capture)
    lnz, ess = _binding_evidence()
    es.run_nested_escalation(lnZ=lnz, ess=ess,
                            **_base_kwargs(target_teff_k=4500., plx=50., checkpoint_directory=tmp_path))
    assert len(calls) == es._ENSEMBLE_RUNS
    assert all(call["host_teff_k"] == 4500. for call in calls)
    kwargs = _base_kwargs(target_teff_k=4500., plx=50.)
    blocks = es._population_blocks(**{key: kwargs[key] for key in (
        "M_s_Msun", "R_s_Rsun", "plx", "contrast_separations", "contrast_values",
        "resolution_limit_arcsec", "background_population", "background_area_sq_deg",
        "target_tess_mag", "target_teff_k")})
    assert blocks["bound"].host_teff == 4500.


def test_engine_forwards_existing_scene_temperature(tmp_path, monkeypatch):
    from tests.test_trex_host_limb_darkening_routing import _non_solar_scene, _run_capturing_evidences

    scene = _non_solar_scene(tmp_path)
    captured, _ = _run_capturing_evidences(monkeypatch, scene)
    assert captured["target_teff_k"] == scene.Teff_K
