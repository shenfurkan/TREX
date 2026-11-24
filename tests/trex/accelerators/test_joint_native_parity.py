"""Native joint multi-band likelihood versus the scalar joint owner.

The reference is ``JointLikelihood``/``ScenarioObservations`` evaluated on the
same independently constructed aperture streams. Tolerances: the finite
likelihood comparison uses the established TREX batman/JAX kernel agreement
(``rtol=1e-4``, ``atol=1e-10``); the joint log density adds ``S`` stream values
in dataset order, so its float64 summation bound is ``(S + 1) * eps * sum|v|``
with ``eps = 2.22e-16``, i.e. below ``1e-10`` for the O(10^2) log densities
here. The shared-baseline marginalization is additionally checked against an
independent dense multivariate-normal formula that never uses the owning
Cholesky-completed-square algebra; the two are algebraically identical, so for
these O(10-10^2)-nat densities their float64 route difference is ~1e-12
relative and `abs=1e-9` leaves two orders of margin.
"""

import types
from dataclasses import replace

import numpy as np
import pytest
import scipy.stats

from trex.accelerators import _jax_models as models
from trex.accelerators._jax_joint_unit_integrand import (
    JointEclipsingPairUnitIntegrand,
    JointPlanetUnitIntegrand,
    joint_eclipsing_pair_callback,
    joint_planet_callback,
)
from trex.app.engine import _joint_native_covers_scene, prepare_vetting_inputs
from trex.evidence import nested_controls
from trex.evidence import nested_evidence as ne
from trex.evidence.joint_observations import ScenarioObservations
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.baseline import GaussianBaseline
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import REFERENCE, joint_case, light, observation

JAX = pytest.mark.skipif(not models.HAS_JAX_TREX,
                         reason="native joint likelihood needs JAX/jaxoplanet float64")
HOST = dict(teff=5700., mass=1., radius=1., feh=0., period=3., epoch=.02)


def _scenario(*, shared_baseline=None, zero_share=False):
    """Three ragged native-time streams with a different atmosphere per band."""
    rows = (
        observation(shares={"target": .85, "neighbor": .15}),
        observation("ground", time=np.array([-.11, -.03, .05, .07, .9]), sigma=.05,
                    shares={"target": .6, "neighbor": .4}),
        observation("aux", time=np.array([-.2, .21, 1.2, 1.24]), sigma=.2,
                    shares=({"target": 0., "neighbor": 1.} if zero_share
                            else {"target": .3, "neighbor": .7})),
    )
    set_ = ObservationSet(rows, REFERENCE)
    lights = {
        "synthetic/space": light("space", u1=.22, u2=.31),
        "synthetic/ground": light("ground", u1=.41, u2=.18),
        "synthetic/aux": light("aux", u1=.1, u2=.42),
    }
    joint = ScenarioObservations(JointLikelihood(set_, shared_baseline=shared_baseline),
                                 host_id="target", band_light=lights, epoch_days=HOST["epoch"])
    return set_, lights, joint


def _light_by_dataset(set_, lights):
    """The scalar owner's per-dataset light for the fixed host."""
    values = {}
    for row in set_.observations:
        light_ = lights[row.band_id]
        component = light_.prepare_component(HOST["teff"], HOST["mass"], HOST["radius"],
                                             HOST["feh"], light_.target_av_mag)
        values[row.dataset_id] = {"u1": float(component.u1), "u2": float(component.u2),
                                  "angular_profile": component.angular_profile}
    return values


def _tp_integrand(joint, *, population=None):
    first = joint.likelihood.observations.observations[0]
    return ne._TransitingPlanetIntegrand(
        time=joint.likelihood.times[0] - joint.epoch_days, flux=first.flux, sigma=first.sigma,
        period_days=HOST["period"], host_mass=HOST["mass"], host_radius=HOST["radius"],
        u1=.2, u2=.3, exptime_days=None, bin_integration_days=0., nsamples=4,
        use_flat_priors=False, population=population, n_dimensions=4,
        target_feh=HOST["feh"], host_teff_k=HOST["teff"],
        stellar_light=joint.band_light[first.band_id], joint_observations=joint)


def _eb_integrand(joint, *, use_2xP=False, population=None):
    first = joint.likelihood.observations.observations[0]
    return ne._EclipsingPairIntegrand(
        time=joint.likelihood.times[0] - joint.epoch_days, flux=first.flux, sigma=first.sigma,
        effective_period=HOST["period"] * (2. if use_2xP else 1.), use_2xP=use_2xP,
        host_mass=HOST["mass"], host_radius=HOST["radius"], u1=.2, u2=.3,
        exptime_days=None, bin_integration_days=0., nsamples=4,
        population=population, n_dimensions=4, target_feh=HOST["feh"],
        host_teff_k=HOST["teff"], stellar_light=joint.band_light[first.band_id],
        joint_observations=joint)


def _device():
    return models.jax.devices("cpu")[0]


@JAX
def test_native_joint_tp_matches_the_scalar_joint_density_on_ragged_bands():
    _, _, joint = _scenario()
    reference = _tp_integrand(joint)
    factory = joint_planet_callback(reference)
    assert callable(factory), "the fixed-host joint TP scenario has no device twin"
    native = factory(device=_device())
    assert isinstance(native, JointPlanetUnitIntegrand)
    points = np.random.default_rng(20261116).uniform(size=(24, 4))
    points[:4, 1] = (1e-9, 1e-4, .35, .9)
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    assert np.asarray(evaluated.numerically_valid).all()
    actual = np.asarray(evaluated.log_integrand, dtype=float)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.sum() >= 8
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)


@JAX
@pytest.mark.parametrize("binary", [False, True])
def test_native_joint_eb_matches_the_scalar_joint_density_on_ragged_bands(binary):
    _, _, joint = _scenario()
    reference = _eb_integrand(joint, use_2xP=binary)
    factory = joint_eclipsing_pair_callback(reference)
    assert callable(factory), "the fixed-host joint EB scenario has no device twin"
    native = factory(device=_device())
    assert isinstance(native, JointEclipsingPairUnitIntegrand)
    points = np.random.default_rng(20261117).uniform(size=(24, 4))
    # The disjoint EB/twin q support starts where the twin branch carries its
    # prior mass; keep the drawn quantiles inside the selected branch so both
    # branches are actually compared.
    points[:, 0] = np.linspace(.9, .999, 24) if binary else np.linspace(.2, .8, 24)
    points[:4, 1] = (1e-9, 1e-3, .2, .4)
    points[:4, 2] = (.1, .8, .45, .2)
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    assert np.asarray(evaluated.numerically_valid).all()
    actual = np.asarray(evaluated.log_integrand, dtype=float)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.sum() >= 8
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)


def _shared_baseline(set_):
    times = np.concatenate([row.time_axis.relative_midpoints(set_.reference)
                            for row in set_.observations])
    streams = tuple(row.dataset_id for row in set_.observations for _ in row.flux)
    centre = times - times.mean()
    return GaussianBaseline(
        time_days=times, design=np.column_stack((np.ones(times.size), centre)),
        prior_mean=np.zeros(2), prior_covariance=np.asarray([[4e-4, 1e-5], [1e-5, 2e-5]]),
        mode="additive", source="synthetic test prior", stream_ids=streams)


@JAX
def test_native_joint_shared_baseline_is_one_marginalization_over_all_streams():
    set_, lights, _ = _scenario()
    baseline = _shared_baseline(set_)
    joint = ScenarioObservations(JointLikelihood(set_, shared_baseline=baseline),
                                 host_id="target", band_light=lights, epoch_days=HOST["epoch"])
    reference = _tp_integrand(joint)
    native = joint_planet_callback(reference)(device=_device())
    point = np.random.default_rng(20261118).uniform(size=4)
    point[1] = .2
    radius, ecc, argp, inc = reference.prior_transform(point)
    prediction = joint.likelihood.models(
        host_id="target", light_by_dataset=_light_by_dataset(set_, lights),
        period_days=HOST["period"], epoch_days=HOST["epoch"], inc_deg=float(inc),
        a_cm=reference.a_cm, host_radius_solar=HOST["radius"],
        planet_radius_earth=float(radius), ecc=float(ecc), argp_deg=float(argp))
    scalar = joint.evaluate(
        host_mass=HOST["mass"], host_radius=HOST["radius"], host_teff=HOST["teff"],
        host_metallicity=HOST["feh"], period_days=HOST["period"], inc_deg=float(inc),
        a_cm=reference.a_cm, ecc=float(ecc), argp_deg=float(argp),
        planet_radius=float(radius))
    # Independent marginalization y ~ N(m + A mu, diag(sigma^2) + A Lambda A^T)
    # through a dense formula that never calls the owning Cholesky algebra.
    rows = set_.observations
    observed = np.concatenate([row.flux for row in rows])
    sigma = np.concatenate([row.sigma for row in rows])
    predicted = np.concatenate([prediction[row.dataset_id] for row in rows])
    shifted = (observed - predicted - baseline.design @ baseline.prior_mean) / sigma
    whitened = baseline.design / sigma[:, None]
    covariance = np.eye(observed.size) + whitened @ baseline.prior_covariance @ whitened.T
    independent = float(
        scipy.stats.multivariate_normal(mean=np.zeros(observed.size), cov=covariance).logpdf(shifted)
        - np.sum(np.log(sigma)))
    assert scalar == pytest.approx(independent, abs=1e-9)
    # The unit-cube integrand adds the scalar integrand's own prior-support term
    # (log band factor) to this likelihood, so the native value is compared with
    # the scalar integrand on the same point, as in the per-dataset parity tests.
    expected = float(reference.log_likelihood(reference.prior_transform(point)))
    assert np.isfinite(expected) and expected != pytest.approx(scalar, abs=1e-6)
    values = np.asarray(models.jax.jit(native.evaluate)(point).log_integrand, dtype=float)
    assert np.isfinite(values)
    np.testing.assert_allclose(values, expected, rtol=1e-4, atol=1e-10)


def _row_baseline(time):
    """One per-dataset proper Gaussian prior, the other admitted noise form."""
    centre = time - time.mean()
    return GaussianBaseline(
        time_days=time, design=np.column_stack((np.ones(time.size), centre)),
        prior_mean=np.zeros(2), prior_covariance=np.asarray([[1e-4, 0.], [0., 1e-6]]),
        mode="additive", source="synthetic per-dataset prior")


@JAX
def test_native_joint_per_dataset_baseline_is_summed_in_dataset_order():
    rows = (observation(shares={"target": .8, "neighbor": .2}),
            observation("ground", time=np.array([-.09, -.01, .05, .5]), sigma=.08,
                        shares={"target": .5, "neighbor": .5}))
    rows = tuple(replace(row, baseline=_row_baseline(row.time_axis.relative_midpoints(REFERENCE)))
                 for row in rows)
    set_ = ObservationSet(rows, REFERENCE)
    lights = {row.band_id: light(row.dataset_id) for row in rows}
    joint = ScenarioObservations(JointLikelihood(set_), host_id="target",
                                 band_light=lights, epoch_days=HOST["epoch"])
    assert all(prepared is not None for prepared in joint.likelihood.baselines)
    reference = _tp_integrand(joint)
    native = joint_planet_callback(reference)(device=_device())
    points = np.random.default_rng(20261122).uniform(size=(6, 4))
    points[:, 1] = .15
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    actual = np.asarray(models.jax.vmap(native.evaluate)(points).log_integrand, dtype=float)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-10)


@JAX
def test_native_joint_zero_share_stream_predicts_flat_without_deleting_scenario():
    set_, lights, joint = _scenario(zero_share=True)
    reference = _tp_integrand(joint)
    native = joint_planet_callback(reference)(device=_device())
    kept = ObservationSet(set_.observations[:2], REFERENCE)
    reduced = ScenarioObservations(JointLikelihood(kept), host_id="target",
                                   band_light=lights, epoch_days=HOST["epoch"])
    reduced_native = joint_planet_callback(_tp_integrand(reduced))(device=_device())
    points = np.random.default_rng(20261119).uniform(size=(8, 4))
    points[:, 1] = .1
    with_aux = np.asarray(models.jax.vmap(native.evaluate)(points).log_integrand, dtype=float)
    without = np.asarray(models.jax.vmap(reduced_native.evaluate)(points).log_integrand, dtype=float)
    auxiliary = set_.observations[2]
    flat = scipy.stats.norm.logpdf(auxiliary.flux, loc=1., scale=auxiliary.sigma).sum()
    assert np.isfinite(with_aux).all()
    np.testing.assert_allclose(with_aux - without, flat, rtol=0., atol=1e-9)


@JAX
def test_joint_importance_refinement_uses_the_device_twin_and_keeps_the_estimate():
    _, _, joint = _scenario()
    reference = _tp_integrand(joint)
    reference.jax_unit_integrand = joint_planet_callback(reference)
    samples = np.random.default_rng(20261120).uniform(size=(256, 4))
    posterior = [(samples, np.full(samples.shape[0], 1. / samples.shape[0]))]
    kwargs = dict(n_dimensions=4, random_seed=20261121, n_jobs=1, n_draws=1024)
    scalar = ne._run_importance_sampler(
        reference.prior_transform, reference.log_likelihood, posterior, **kwargs)
    native = ne._run_importance_sampler(
        reference.prior_transform, reference.log_likelihood, posterior,
        cpu_native_integrand=reference, **kwargs)
    assert scalar.execution_provenance["backend"] == "scalar-cpu"
    assert native.execution_provenance["backend"] == "jax-native-cpu"
    assert native.n_likelihood_calls == scalar.n_likelihood_calls
    assert np.isfinite(native.log_evidence)
    assert native.log_evidence == pytest.approx(scalar.log_evidence, abs=1e-6)


def _captured_integrand(monkeypatch):
    captured = {}

    class _Captured(Exception):
        pass

    def integrate(integrand, controls, scenario, n_dimensions, importance_posterior):
        captured["integrand"] = integrand
        raise _Captured

    monkeypatch.setattr(ne, "_integrate", integrate)
    return captured, _Captured


@JAX
def test_joint_scenario_routing_attaches_only_the_joint_device_twin(monkeypatch):
    set_, lights, joint = _scenario()
    captured, captured_error = _captured_integrand(monkeypatch)
    options = dict(
        time=joint.likelihood.times[0] - joint.epoch_days,
        flux=set_.observations[0].flux, sigma=set_.observations[0].sigma,
        period_days=HOST["period"], M_s_Msun=HOST["mass"], R_s_Rsun=HOST["radius"],
        u1=.2, u2=.3, exptime_days=None, bin_integration_days=0., nsamples=4,
        target_aperture_flux_fraction=joint.proposal_fraction("target"),
        target_feh=HOST["feh"], random_seed=5, n_jobs=1, max_likelihood_calls=None,
        importance_posterior=[(np.full((4, 4), .5), np.full(4, .25))],
        stellar_light=lights[set_.observations[0].band_id], host_teff_k=HOST["teff"],
        joint_observations=joint)
    with pytest.raises(captured_error):
        ne.tp_scenario_nested_evidence(**options)
    fixed = captured["integrand"].jax_unit_integrand
    assert callable(fixed), "a fixed-host joint TP scenario lost its device twin"
    assert type(fixed(device=_device())) is JointPlanetUnitIntegrand

    population = ne.bound_companion_population(
        HOST["mass"], HOST["radius"], 10., resolution_limit_arcsec=10.,
        host_teff_k=HOST["teff"], stellar_light=lights["synthetic/space"],
        target_feh=HOST["feh"], datasets=joint.population_datasets,
        band_light=joint.band_light)
    with pytest.raises(captured_error):
        ne.tp_scenario_nested_evidence(population=population, **options)
    population_twin = captured["integrand"].jax_unit_integrand
    assert callable(population_twin), (
        "a population-diluted joint scenario must keep a joint device twin, "
        "not a single-band callback or the scalar path")
    from trex.accelerators._jax_joint_population_integrand import (
        JointBoundPopulationUnitIntegrand,
    )

    assert type(population_twin(device=_device())) is JointBoundPopulationUnitIntegrand


def _raised_message(call):
    try:
        call()
    except TrexEvidenceUnavailableError as error:
        return str(error)
    raise AssertionError("joint admission did not refuse the uncovered combination")


def test_native_joint_estimator_admission_is_scoped_to_covered_scenes(tmp_path, monkeypatch):
    scene, observations, lights = joint_case(tmp_path)
    first = observations.observations[0]
    likelihood = JointLikelihood(observations)
    # The production admission binds the scene's dataset magnitudes onto the
    # joint before the coverage check; the direct check needs them too.
    joint = ScenarioObservations(likelihood, host_id="target", band_light=lights,
                                 epoch_days=0., target_magnitudes={"space": 10., "ground": 9.})
    covered = types.SimpleNamespace(plx_mas=None, background_area_sq_deg=None)
    assert _joint_native_covers_scene(covered, None)
    # Retained bound and background populations now have joint device twins,
    # so they no longer force the scalar route by themselves.
    assert _joint_native_covers_scene(scene, scene.background_population(),
                                      nearby_stars=[], joint_observations=joint)
    monkeypatch.setenv(nested_controls._NESTED_ESTIMATOR_ENV, "blackjax")
    prepared = prepare_vetting_inputs(
        scene, likelihood.times[0], first.flux, first.sigma, 3., 1000., None,
        stellar_light=lights[first.band_id], joint_observations=joint)
    assert prepared is not None, "a fully covered joint scene must admit the native estimator"
    # A characterized resolved host has no joint device twin: the scene keeps
    # the scalar dynesty requirement instead of deferring per scenario.
    scene.resolved_neighbors[0]["stellar_parameter_distribution"] = {
        "teff_k": (5000., 5100.), "mass_solar": (.79, .81), "radius_solar": (.79, .81)}
    nearby = scene.neighbor_dicts_for_evidence()
    assert any(neighbor.get("stellar_parameter_distribution") is not None
               for neighbor in nearby)
    assert not _joint_native_covers_scene(scene, scene.background_population(),
                                          nearby_stars=nearby, joint_observations=joint)
    message = _raised_message(lambda: prepare_vetting_inputs(
        scene, likelihood.times[0], first.flux, first.sigma, 3., 1000., None,
        stellar_light=lights[first.band_id], joint_observations=joint))
    assert "characterized resolved host" in message


@JAX
def test_native_joint_callback_refuses_boxcar_and_phase_bin_inputs():
    _, _, joint = _scenario()
    reference = _tp_integrand(joint)
    reference.bin_integration_days = 1e-3
    with pytest.raises(ValueError, match="exposure responses"):
        JointPlanetUnitIntegrand(reference, device=_device())
    reference = _tp_integrand(joint)
    reference.exptime_days = 1e-3
    with pytest.raises(ValueError, match="exposure responses"):
        JointPlanetUnitIntegrand(reference, device=_device())
