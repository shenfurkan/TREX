"""Bounded production-vector controls for coupled flux/atmosphere routing."""

import numpy as np
import pytest

from trex.accelerators import _jax_models as jm
from trex.stellar import stellar_relations as trex_stellar
from trex.evidence import nested_evidence as ne
from trex.evidence.mc_population import _draw_bound_companion
from trex.evidence.mc_scenario import _scenario_stages
from trex.stellar_light import PhotonAtmosphereGrid, StellarLight, UnsupportedAtmosphereError
from trex.stellar_light.adapters import binary_light
from trex.stellar_light.native import NativeStellarLight


def _light(*, angular=False, steep=False):
    # Deliberately non-Planck, target-neutral synthetic angular field, not a fallback.
    axes = tuple(np.array(v) for v in ([2300., 8000.], [3., 6.], [-4., 1.], [0., 1.]))
    centre = axes[0][:, None, None, None]*np.ones((2, 2, 2, 2))
    table = np.stack((np.pi*centre*(1-.2/3-.3/6), centre, -.2*centre, -.3*centre), axis=-1)
    angular_data = None
    if angular:
        from trex.stellar_light.passband import angular_moment, quadratic_reduction
        mu = np.array([0., .04, .3, .75, 1.])
        intensity = np.array([0., .02, .8, 1.1, 1.2])
        if steep:
            # Everywhere nonnegative rays whose flux-conserving quadratic is negative inside.
            mu = np.linspace(0., 1., 79)
            intensity = mu**4
        polynomial, _ = quadratic_reduction(mu, intensity, require_nonnegative=not steep)
        table = centre[..., None]*np.r_[2*np.pi*angular_moment(mu, intensity), polynomial]
        angular_data = (np.broadcast_to(mu, (2, 2, 2, mu.size)), centre[..., None]*intensity)
    return StellarLight(PhotonAtmosphereGrid(axes, table, {"test": "synthetic"}, angular=angular_data), 0.)


def _bound_ratio(mass, radius, teff):
    """Synthetic field: surface photons scale with Teff, so F2/F1=(R2/R1)²*T2/T1."""
    ratio = (radius/.5)**2*teff/4000.
    return ratio/(1+ratio)


def test_hosted_mc_preparation_keeps_the_scenario_where_the_quadratic_is_negative():
    """One authentic-ray host with an unusable reduction must not refuse every draw."""
    light = _light(angular=True, steep=True)
    with pytest.raises(ValueError, match="no nonnegative quadratic reduction"):
        light.atmosphere(4000., .5, .5, -.2, 0.)
    population = dict(tess_mag=np.array([13., 14.]), mass_solar=np.array([.5, .8]),
        radius_solar=np.array([.5, .8]), teff_k=np.array([4000., 5000.]),
        mh_dex=np.array([-3.27, .1]), av_mag=np.array([0., 0.]))
    stages = _scenario_stages(np.array([0.]), np.ones(1), .01, 3., .5, .5, .1, .1, 8, True, False, False,
        .001, target_teff_k=4000., target_feh=-.2, stellar_light=light, rng=np.random.default_rng(7),
        background_star="background", background_population=population, background_area_sq_deg=.1,
        target_tess_mag=12., resolution_limit_arcsec=21.)
    next(stages)
    state = stages.gi_frame.f_locals
    assert state["jax_call"] is None  # Rays stay the forward model; u1/u2 only fill the diagnostic slots.
    assert np.all(np.isfinite(state["host_u1"]))
    stages.close()


def test_bound_flux_ratio_needs_no_nonnegative_quadratic_in_mc_or_nested():
    light = _light(angular=True, steep=True)
    fractions, _, mass, radius, teff = _draw_bound_companion(np.random.default_rng(7), 8, .5, .5, 10.,
        None, None, resolution_limit_arcsec=100., host_teff_k=4000., stellar_light=light, target_feh=-.2)
    np.testing.assert_allclose(fractions, _bound_ratio(mass, radius, teff), rtol=2e-15)
    block = ne.bound_companion_population(.5, .5, 10., resolution_limit_arcsec=100.,
        host_teff_k=4000., stellar_light=light, target_feh=-.2)
    draw = block.resolve(np.full(block.n_dimensions, .5))  # Bound blocks add the projection.
    assert draw.flux_fraction == pytest.approx(_bound_ratio(draw.mass_solar, draw.radius_solar, draw.teff_k), rel=2e-15)


def test_mc_binary_light_assembles_no_per_draw_ray_arrays(monkeypatch):
    """Rays are prepared per transiting draw; a 1e6-draw array would need about 36 GB."""
    light = _light(angular=True)
    selected = np.array([True, False, True])
    host = (np.array([3300., 1., 4300.]), np.array([.3, 1., .7]), np.array([.3, 1., .7]))
    secondary = (np.array([3000., 1., 3300.]), np.array([.2, 1., .3]), np.array([.2, 1., .3]))
    expected = light.prepare_pair(tuple(v[selected] for v in host), tuple(v[selected] for v in secondary), -.2, .1)

    def refuse(*_args, **_kwargs):
        raise AssertionError("whole-array rays were assembled")

    monkeypatch.setattr(PhotonAtmosphereGrid, "angular_profile", refuse)
    ratio, first, second = binary_light(light, *host, *secondary, -.2, .1, selected)
    np.testing.assert_array_equal(ratio, expected[0])
    np.testing.assert_array_equal(first, expected[1][1:3])
    np.testing.assert_array_equal(second, expected[2][1:3])


def test_mc_preparation_uses_same_atmosphere_for_both_lights():
    light = _light()
    stages = _scenario_stages(np.array([0.]), np.array([1.]), .01, 3., .5, .5,
        .1, .1, 8, False, True, False, .001, target_teff_k=4000.,
        target_feh=-.2, stellar_light=light, rng=np.random.default_rng(7))
    next(stages)  # ZERO likelihood evaluations and no evidence estimate.
    state = stages.gi_frame.f_locals
    selected = state["eb_supported"]
    r, t = trex_stellar.stellar_relations(state["M_EB"][selected])
    expected = (r/.5)**2*t/4000.
    np.testing.assert_allclose(state["secondary_to_host_ratio"][selected], expected, rtol=2e-15)
    np.testing.assert_allclose(state["host_u1"], .2, rtol=1e-14)
    np.testing.assert_allclose(state["eb_u2"][selected], .3, rtol=1e-14)
    stages.close()


def test_surface_photons_scalar_vector_and_native_agree():
    light = _light()
    host = (np.array([3300., 4300.]), np.array([.3, .7]), np.array([.3, .7]))
    secondary = (np.array([3000., 3300.]), np.array([.2, .3]), np.array([.2, .3]))
    ratio, _, _ = light.pair(host, secondary, [-3.27, -.2], [0., 0.])
    native = NativeStellarLight(light, jm.jax.devices("cpu")[0])
    actual = native.pair_ratio(*host, *secondary, np.array([-3.27, -.2]), np.array([0., 0.]))
    np.testing.assert_allclose(actual, ratio, rtol=2e-15)
    for index in range(2):
        one, _, _ = light.pair(tuple(v[index] for v in host), tuple(v[index] for v in secondary),
                               [-3.27, -.2][index], 0.)
        assert one == pytest.approx(ratio[index], rel=2e-15)
    identity, _, _ = light.pair(host, host, -.2, 0.)
    np.testing.assert_array_equal(identity, 1.)


@pytest.mark.parametrize("angular", [False, True])
def test_prepared_component_pair_preserve_scalar_vector_and_native_products(angular):
    light = _light(angular=angular)
    primary = (np.array([3300., 4300.]), np.array([.3, .7]), np.array([.3, .7]))
    secondary = (np.array([3000., 3300.]), np.array([.2, .3]), np.array([.2, .3]))
    mh, av = np.array([-3.27, -.2]), np.array([0., .1])
    prepared = light.prepare_component(*primary, mh, av)
    np.testing.assert_array_equal(np.asarray(prepared[:3]), light.atmosphere(*primary, mh, av))
    ratio, first, second = light.prepare_pair(primary, secondary, mh, av)
    old_ratio, old_first, old_second = light.pair(primary, secondary, mh, av)
    np.testing.assert_array_equal(ratio, old_ratio)
    np.testing.assert_array_equal(first[1:3], old_first)
    np.testing.assert_array_equal(second[1:3], old_second)
    native = NativeStellarLight(light, jm.jax.devices("cpu")[0])
    device, valid = native.prepare_component(*primary, mh, av)
    assert np.all(valid)
    np.testing.assert_allclose(np.asarray(device[:3]), prepared[:3], rtol=2e-14)
    if angular:
        for actual, expected in zip(device.angular_profile, prepared.angular_profile):
            np.testing.assert_allclose(actual, expected, rtol=2e-14)
    else:
        assert device.angular_profile is prepared.angular_profile is None


@pytest.mark.parametrize("angular", [False, True])
def test_fixed_scalar_native_eb_consume_coupled_photon_model(angular):
    light = _light(angular=angular)
    # Circular geometry places the secondary exactly P/2 after the primary.
    time = np.array([-.04, 0., .04, 1.46, 1.5, 1.54])
    reference = ne._EclipsingPairIntegrand(time=time,
        flux=np.ones(time.size), sigma=.01, effective_period=3., host_mass=.5, host_radius=.5,
        host_teff_k=4000., u1=.1, u2=.1, exptime_days=.001, bin_integration_days=0.,
        nsamples=20, population=None, n_dimensions=4, target_feh=-.2, stellar_light=light)
    native = reference.jax_unit_integrand(device=jm.jax.devices("cpu")[0])
    point = np.array([.3, 0., .25, .8])
    physical = np.asarray(native.prior_transform(point))
    scalar = reference.log_likelihood(physical)
    result = native.evaluate(point)
    assert bool(result.numerically_valid)
    assert float(result.log_integrand) == pytest.approx(scalar, rel=1e-4, abs=1e-6)
    q = physical[0]
    r, t = reference.secondary_geometry(q)
    assert float(native.secondary_flux_ratio(q, r, t)) == pytest.approx((r/.5)**2*t/4000., rel=2e-15)


@pytest.mark.parametrize("family,hosted,planet", [
    ("bound", False, False), ("bound", True, False),
    ("background", False, False), ("background", True, False),
    ("bound", True, True), ("background", True, True),
])
@pytest.mark.parametrize("angular", [False, True])
def test_population_scalar_native_consume_source_atmosphere(family, hosted, planet, angular):
    light = _light(angular=angular)
    if family == "bound":
        block = ne.bound_companion_population(.5, .5, 10., resolution_limit_arcsec=100.,
            host_teff_k=4000., stellar_light=light, target_feh=-.2)
    else:
        block = ne.background_star_population(dict(tess_mag=np.array([13., 14.]),
            mass_solar=np.array([.5, .8]), radius_solar=np.array([.5, .8]),
            teff_k=np.array([4000., 5000.]), mh_dex=np.array([-3.27, .1]),
            av_mag=np.array([0., 0.])), 1., 12., resolution_limit_arcsec=100.)
    cls = ne._TransitingPlanetIntegrand if planet else ne._EclipsingPairIntegrand
    extra = dict(period_days=3., use_flat_priors=False) if planet else dict(effective_period=3.)
    reference = cls(time=np.array([-.04, 0., .04]), flux=np.ones(3), sigma=.01,
        host_mass=.5, host_radius=.5, host_teff_k=4000., u1=.1, u2=.1,
        exptime_days=.001, bin_integration_days=0., nsamples=20, population=block,
        n_dimensions=4 + block.n_dimensions, hosted=hosted, target_feh=-.2, stellar_light=light, **extra)
    native = reference.jax_unit_integrand(device=jm.jax.devices("cpu")[0])
    point = np.r_[.3, .01, .25, .8, np.full(block.n_dimensions, .5)]
    parameters = reference.prior_transform(point)
    np.testing.assert_allclose(native.prior_transform(point), parameters, rtol=1e-10, atol=1e-8)
    scalar = reference.log_likelihood(parameters)
    result = native.evaluate(point)
    assert bool(result.numerically_valid)
    assert np.isfinite(scalar)
    assert float(result.log_integrand) == pytest.approx(scalar, rel=1e-4, abs=1e-6)


def test_missing_secondary_is_failure_not_zero_prior_or_reference_fallback():
    full = _light()
    # Same synthetic field with no cool-atmosphere support; keep the binary prior unchanged.
    axes = (np.array([3500., 8000.]), *full.grid.axes[1:])
    light = StellarLight(PhotonAtmosphereGrid(axes, full.grid.table, {"test": "synthetic narrow support"}), 0.)
    ref = ne._EclipsingPairIntegrand(time=np.array([0.]), flux=np.ones(1), sigma=.01,
        effective_period=3., host_mass=.5, host_radius=.5, host_teff_k=4000., u1=.1, u2=.1,
        exptime_days=.001, bin_integration_days=0., nsamples=20, population=None,
        n_dimensions=4, target_feh=-.2, stellar_light=light)
    native = ref.jax_unit_integrand(device=jm.jax.devices("cpu")[0])
    point = np.array([.1, .01, .25, .8])
    with pytest.raises(UnsupportedAtmosphereError):
        ref.log_likelihood(np.asarray(native.prior_transform(point)))
    result = native.evaluate(point)
    assert not bool(result.numerically_valid)
    assert np.isnan(float(result.log_integrand))


def test_bound_missing_light_does_not_change_structural_binary_branch_support():
    full = _light()
    axes = (np.array([3500., 8000.]), *full.grid.axes[1:])
    light = StellarLight(PhotonAtmosphereGrid(axes, full.grid.table, {"test": "narrow"}), 0.)
    block = ne.bound_companion_population(.5, .5, 10., resolution_limit_arcsec=100.,
        host_teff_k=4000., stellar_light=light, target_feh=-.2)
    ref = ne._EclipsingPairIntegrand(time=np.array([0.]), flux=np.ones(1), sigma=.01,
        effective_period=3., host_mass=.5, host_radius=.5, host_teff_k=4000., u1=.1, u2=.1,
        exptime_days=.001, bin_integration_days=0., nsamples=20, population=block,
        n_dimensions=4 + block.n_dimensions, target_feh=-.2, stellar_light=light)
    native = ref.jax_unit_integrand(device=jm.jax.devices("cpu")[0])
    point = np.r_[.1, .01, .25, .8, np.full(block.n_dimensions, .5)]
    result = native.evaluate(point)
    assert not bool(result.numerically_valid)
    assert np.isnan(float(result.log_integrand))
    # A twin q is outside this ordinary-EB prior independently of the missing light.
    point[0] = .99
    parameters = ref.prior_transform(point)
    assert ref.log_likelihood(parameters) == -np.inf
    result = native.evaluate(point)
    assert bool(result.numerically_valid)
    assert float(result.log_integrand) == -np.inf


@pytest.mark.slow
def test_real_nested_run_on_negative_quadratic_rays_resumes_exactly(tmp_path, monkeypatch):
    """Actual dynesty over authentic-style rays; a pause cannot change the numbers."""
    light = _light(angular=True, steep=True)
    block = ne.background_star_population(dict(tess_mag=np.array([13., 14.]), mass_solar=np.array([.5, .8]),
        radius_solar=np.array([.5, .8]), teff_k=np.array([4000., 5000.]), mh_dex=np.array([-3.27, .1]),
        av_mag=np.array([0., 0.])), .1, 12., resolution_limit_arcsec=21.)
    time = np.linspace(-.05, .05, 9)
    call = dict(time=time, flux=np.ones(9), sigma=.01, period_days=3., M_s_Msun=.5, R_s_Rsun=.5, u1=.1, u2=.1,
        exptime_days=.001, target_aperture_flux_fraction=1., hosted_population=block, target_feh=-.2,
        host_teff_k=4000., stellar_light=light, nsamples=3, n_live_points=20, evidence_tolerance=10.,
        random_seed=7, max_likelihood_calls=30_000, n_jobs=1)
    monkeypatch.setattr(ne, "_monotonic", lambda: 0.)
    with ne.nested_checkpoint(tmp_path/"complete", deadline=50.):
        complete = ne.tp_scenario_nested_evidence(**call)

    clock = iter([0.]*12+[100.]*100_000)
    monkeypatch.setattr(ne, "_monotonic", lambda: next(clock))
    with pytest.raises(ne.NestedCheckpointPaused):
        with ne.nested_checkpoint(tmp_path/"resumed", deadline=50.):
            ne.tp_scenario_nested_evidence(**call)
    monkeypatch.setattr(ne, "_monotonic", lambda: 0.)
    with ne.nested_checkpoint(tmp_path/"resumed", deadline=50.):
        resumed = ne.tp_scenario_nested_evidence(**call)

    assert np.isfinite(complete.log_evidence)
    assert resumed.log_evidence == complete.log_evidence
    np.testing.assert_array_equal(resumed.posterior_unit_samples, complete.posterior_unit_samples)
    np.testing.assert_array_equal(resumed.posterior_weights, complete.posterior_weights)


@pytest.mark.slow
@pytest.mark.parametrize("planet", [False, True])
def test_mc_angular_likelihoods_use_original_profiles_not_quadratic_batch(planet):
    """Finite synthetic producer/consumer parity, NOT a production-budget estimate."""
    from scipy.special import logsumexp
    from trex.models import likelihoods as lk
    light = _light(angular=True)
    time = np.array([-.03, 0., .03, 1.5])
    stages = _scenario_stages(time, np.ones(4), .01, 3., .5, .5, .1, .1,
        8, planet, not planet, False, .001, target_teff_k=4000.,
        target_feh=-.2, stellar_light=light, rng=np.random.default_rng(7))
    next(stages)
    state = stages.gi_frame.f_locals.copy()
    assert state["jax_call"] is None  # Old bulk evaluator supports only a quadratic law.
    with pytest.raises(StopIteration) as result:
        next(stages)
    expected_likelihood = np.full(8, -np.inf)
    primary_profile = light.angular_profile(4000., .5, .5, -.2, 0.)
    for i in np.flatnonzero(state["transit_mask"]):
        common = dict(ecc=state["ecc"][i], argp_deg=state["w"][i], angular_profile=primary_profile)
        args = (time, np.ones(4), .01)
        orbit = (state["eff_period"], state["inc"][i], state["a_cm"][i], .5,
                 state["host_u1"][i], state["host_u2"][i], .001)
        if planet:
            expected_likelihood[i] = lk.lnL_TP(*args, state["R_p"][i], *orbit, **common)
        else:
            radius, temperature = trex_stellar.stellar_relations([state["M_EB"][i]])
            secondary_profile = light.angular_profile(temperature[0], state["M_EB"][i], radius[0], -.2, 0.)
            expected_likelihood[i] = lk.lnL_EB(*args, radius[0], None,
                *orbit, **common, u1_EB=state["eb_u1"][i], u2_EB=state["eb_u2"][i],
                secondary_angular_profile=secondary_profile,
                secondary_to_host_flux_ratio=state["secondary_to_host_ratio"][i])
    assert np.any(np.isfinite(expected_likelihood))
    weights = expected_likelihood+state["lnprior_arr"]+state["occultor_log_weight"]+state["inclination_log_weight"]
    assert result.value.value == pytest.approx(logsumexp(weights)-np.log(8), abs=1e-11)
