"""Native joint population likelihood versus the scalar joint integrands.

The reference is the joint branch of the actual scalar population
integrands (``_TransitingPlanetIntegrand``/``_EclipsingPairIntegrand``
through ``ScenarioObservations``) on independently constructed ragged
aperture streams. Tolerances follow the established joint and population
twins: the likelihood comparison uses ``rtol=1e-4``, ``atol=1e-10`` (the
batman/JAX kernel agreement) and the prior transforms ``rtol=1e-12``.
Physical GPU acceptance runs only with ``EXONYM_REQUIRE_GPU=1``, exactly
like the single-band population parity modules.
"""

import os
import time

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.accelerators._jax_joint_population_integrand import (
    JointBackgroundPopulationUnitIntegrand,
    JointBoundPopulationUnitIntegrand,
)
from trex.accelerators._jax_joint_unit_integrand import (
    joint_eclipsing_pair_callback,
    joint_planet_callback,
)
from trex.evidence import nested_evidence as ne
from trex.evidence.joint_observations import ScenarioObservations
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import REFERENCE, light, observation
from tests.fixtures.synthetic_observations import synthetic_trex_background_integrand_population

JAX = pytest.mark.skipif(not models.HAS_JAX_TREX,
                         reason="native joint likelihood needs JAX/jaxoplanet float64")
HOST = dict(teff=5700., mass=1., radius=1., feh=0., period=3., epoch=.02)
GPU = pytest.param("gpu:0", marks=pytest.mark.skipif(
    os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="physical GPU acceptance requires an approved GPU host"))
GPU1 = pytest.param("gpu:1", marks=pytest.mark.skipif(
    os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="physical GPU acceptance requires an approved GPU host"))


def _joint_scenario():
    """Three ragged native-time streams; per-dataset magnitudes for dilution."""
    rows = (
        observation(shares={"target": .85, "neighbor": .15}),
        observation("ground", time=np.array([-.11, -.03, .05, .07, .9]), sigma=.05,
                    shares={"target": .6, "neighbor": .4}),
        observation("aux", time=np.array([-.2, .21, 1.2, 1.24]), sigma=.2,
                    shares={"target": .3, "neighbor": .7}),
    )
    set_ = ObservationSet(rows, REFERENCE)
    lights = {
        "synthetic/space": light("space", u1=.22, u2=.31),
        "synthetic/ground": light("ground", u1=.41, u2=.18),
        "synthetic/aux": light("aux", u1=.1, u2=.42),
    }
    joint = ScenarioObservations(
        JointLikelihood(ObservationSet(rows, REFERENCE)), host_id="target",
        band_light=lights, epoch_days=HOST["epoch"],
        target_magnitudes={"space": 10., "ground": 9., "aux": 11.})
    return set_, lights, joint


def _bound_population(lights, joint, primary_mass, measured_curve):
    return ne.bound_companion_population(
        primary_mass, primary_mass, 50., resolution_limit_arcsec=1.0,
        contrast_separations=np.asarray([.1, .5, 1.]) if measured_curve else None,
        contrast_values=np.asarray([1., 3., 5.]) if measured_curve else None,
        host_teff_k=HOST["teff"], stellar_light=lights["synthetic/space"],
        target_feh=HOST["feh"], datasets=joint.population_datasets,
        band_light=joint.band_light)


def _background_population(joint, measured_curve):
    population = dict(synthetic_trex_background_integrand_population())
    population["av_mag"] = np.asarray([.02, .05, .1])
    population["band_magnitudes"] = {
        "synthetic/space": np.asarray([13., 12., 14.]),
        "synthetic/ground": np.asarray([12.5, 11.5, 13.8]),
        "synthetic/aux": np.asarray([13.9, 12.9, 15.2]),
    }
    population["band_valid"] = {
        "synthetic/space": np.array([True, True, True]),
        "synthetic/ground": np.array([True, True, True]),
        "synthetic/aux": np.array([True, True, True]),
    }
    return ne.background_star_population(
        population, simulated_area_sq_deg=1.0, target_tess_mag=10.0,
        contrast_separations=np.asarray([.25, .5, 1.]) if measured_curve else None,
        contrast_values=np.asarray([1., 3., 5.]) if measured_curve else None,
        resolution_limit_arcsec=1.0, datasets=joint.population_datasets,
        selection_band="synthetic/space", selection_magnitude=10.0,
        completeness="not modeled: synthetic fixture population")


def _tp_integrand(joint, *, population=None, hosted=False, host_mass=1.0):
    first = joint.likelihood.observations.observations[0]
    return ne._TransitingPlanetIntegrand(
        time=joint.likelihood.times[0] - joint.epoch_days, flux=first.flux,
        sigma=first.sigma, period_days=HOST["period"], host_mass=host_mass,
        host_radius=host_mass, u1=.2, u2=.3, exptime_days=None,
        bin_integration_days=0., nsamples=4, use_flat_priors=False,
        population=population, n_dimensions=4 if population is None else 4 + population.n_dimensions,
        hosted=hosted, target_feh=HOST["feh"], host_teff_k=HOST["teff"],
        stellar_light=joint.band_light[first.band_id], joint_observations=joint)


def _eb_integrand(joint, *, population=None, hosted=False, use_2xP=False, host_mass=1.0):
    first = joint.likelihood.observations.observations[0]
    return ne._EclipsingPairIntegrand(
        time=joint.likelihood.times[0] - joint.epoch_days, flux=first.flux,
        sigma=first.sigma, effective_period=HOST["period"] * (2. if use_2xP else 1.),
        use_2xP=use_2xP, host_mass=host_mass, host_radius=host_mass, u1=.2, u2=.3,
        exptime_days=None, bin_integration_days=0., nsamples=4,
        population=population, n_dimensions=4 if population is None else 4 + population.n_dimensions,
        hosted=hosted, target_feh=HOST["feh"], host_teff_k=HOST["teff"],
        stellar_light=joint.band_light[first.band_id], joint_observations=joint)


def _device(execution_device):
    platform, _, index = execution_device.partition(":")
    return models.jax.devices(platform)[int(index or 0)]


def _orbit_points(reference, kind, count, seed):
    points = np.random.default_rng(seed).uniform(size=(count, reference.n_dimensions))
    if kind == "tp":
        points[:4, 1] = (1e-9, 1e-4, .35, .9)
    else:
        points[:, 0] = np.linspace(.9, .999, count) if kind == "ebx2p" else np.linspace(.2, .8, count)
        points[:4, 1] = (1e-9, 1e-3, .2, .4)
        points[:4, 2] = (.1, .8, .45, .2)
    if reference.population is not None:
        points[:4, 4] = (0.0, .5, np.nextafter(1.0, 0.0), .999)
    return points


def _assert_population_parity(reference, native, points, device):
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    placed = models.jax.device_put(points, device)
    transformed = models.jax.jit(models.jax.vmap(native.prior_transform))(placed)
    np.testing.assert_allclose(transformed, parameters, rtol=1e-12, atol=1e-14)
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(placed)
    models.jax.block_until_ready(evaluated.log_integrand)
    assert evaluated.log_integrand.devices() == {device}
    assert evaluated.log_integrand.dtype == np.float64
    assert np.asarray(evaluated.numerically_valid).all()
    np.testing.assert_array_equal(np.isneginf(evaluated.log_integrand), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.sum() >= 4
    np.testing.assert_allclose(
        np.asarray(evaluated.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)
    assert native.claim_eligible is False
    program = str(models.jax.make_jaxpr(models.jax.vmap(native.evaluate))(points))
    assert "pure_callback" not in program and "io_callback" not in program
    return expected, np.asarray(evaluated.log_integrand, dtype=float)


@JAX
@pytest.mark.parametrize("kind", ["tp", "eb", "ebx2p"])
@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("measured_curve", [False, True])
@pytest.mark.parametrize("primary_mass", [1.0, 0.3])
@pytest.mark.parametrize("execution_device", ["cpu", GPU, GPU1])
def test_joint_bound_population_matches_scalar(
        kind, hosted, measured_curve, primary_mass, execution_device):
    _set, lights, joint = _joint_scenario()
    population = _bound_population(lights, joint, primary_mass, measured_curve)
    if kind == "tp":
        reference = _tp_integrand(joint, population=population, hosted=hosted, host_mass=primary_mass)
        factory = joint_planet_callback(reference)
    else:
        reference = _eb_integrand(joint, population=population, hosted=hosted,
                                  use_2xP=(kind == "ebx2p"), host_mass=primary_mass)
        factory = joint_eclipsing_pair_callback(reference)
    assert callable(factory), "a population joint scenario lost its device twin"
    native = factory(device=_device(execution_device))
    assert type(native) is JointBoundPopulationUnitIntegrand
    points = _orbit_points(reference, kind, 48, 20261118)
    _assert_population_parity(reference, native, points, _device(execution_device))


@JAX
@pytest.mark.parametrize("kind", ["tp", "eb", "ebx2p"])
@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("measured_curve", [False, True])
@pytest.mark.parametrize("execution_device", ["cpu", GPU, GPU1])
def test_joint_background_population_matches_scalar(
        kind, hosted, measured_curve, execution_device):
    _set, _lights, joint = _joint_scenario()
    population = _background_population(joint, measured_curve)
    if kind == "tp":
        reference = _tp_integrand(joint, population=population, hosted=hosted)
        factory = joint_planet_callback(reference)
    else:
        reference = _eb_integrand(joint, population=population, hosted=hosted,
                                  use_2xP=(kind == "ebx2p"))
        factory = joint_eclipsing_pair_callback(reference)
    assert callable(factory), "a population joint scenario lost its device twin"
    native = factory(device=_device(execution_device))
    assert type(native) is JointBackgroundPopulationUnitIntegrand
    points = _orbit_points(reference, kind, 48, 20261119)
    _assert_population_parity(reference, native, points, _device(execution_device))


@JAX
def test_joint_population_parity_detects_perturbed_per_band_inputs():
    """The parity above is sensitivity, not coincidence.

    A wrong per-dataset share, a wrong per-band host-light calibration and a
    wrong per-draw extinction each break the agreement, so the parity tests
    fail when the per-draw per-band calculation is perturbed.
    """
    _set, lights, joint = _joint_scenario()
    population = _bound_population(lights, joint, 1.0, False)
    hosted_reference = _tp_integrand(joint, population=population, hosted=True)
    hosted_native = joint_planet_callback(hosted_reference)(device=_device("cpu"))
    points = _orbit_points(hosted_reference, "tp", 32, 20261123)
    expected = _assert_population_parity(
        hosted_reference, hosted_native, points, _device("cpu"))[0]
    finite = np.isfinite(expected)
    parity = dict(rtol=1e-4, atol=1e-10)

    original_shares = hosted_native.streams.joint.shares
    perturbed_share = original_shares[1] * 1.05
    hosted_native.streams.joint.shares = (
        original_shares[:1] + (perturbed_share,) + original_shares[2:])
    mutated = np.asarray(models.jax.jit(models.jax.vmap(hosted_native.evaluate))(
        points).log_integrand, dtype=float)
    hosted_native.streams.joint.shares = original_shares
    assert not np.allclose(mutated[finite], expected[finite], **parity), (
        "a five percent per-dataset share error must break the joint parity")

    original_table = hosted_native.streams.band_lights[0].table
    hosted_native.streams.band_lights[0].table = original_table * 1.01
    mutated = np.asarray(models.jax.jit(models.jax.vmap(hosted_native.evaluate))(
        points).log_integrand, dtype=float)
    hosted_native.streams.band_lights[0].table = original_table
    assert not np.allclose(mutated[finite], expected[finite], **parity), (
        "a one percent per-band host-light error must break the joint parity")

    original_av = hosted_native.field_av_mag
    hosted_native.field_av_mag = 0.7
    mutated = np.asarray(models.jax.jit(models.jax.vmap(hosted_native.evaluate))(
        points).log_integrand, dtype=float)
    hosted_native.field_av_mag = original_av
    assert not np.allclose(mutated[finite], expected[finite], **parity), (
        "a wrong per-draw extinction must break the joint parity")


@JAX
def test_joint_population_callback_refuses_boxcar_and_phase_bin_inputs():
    """Joint observations own their exposures; no legacy integration is admitted."""
    _set, lights, joint = _joint_scenario()
    population = _bound_population(lights, joint, 1.0, False)
    reference = _tp_integrand(joint, population=population)
    reference.bin_integration_days = 1e-3
    with pytest.raises(ValueError, match="exposure responses"):
        joint_planet_callback(reference)(device=_device("cpu"))
    reference = _tp_integrand(joint, population=population)
    reference.exptime_days = 1e-3
    with pytest.raises(ValueError, match="exposure responses"):
        joint_eclipsing_pair_callback(reference)(device=_device("cpu"))


@JAX
@pytest.mark.parametrize("execution_device", ["cpu", GPU, GPU1])
def test_joint_population_matched_host_timing_is_recorded(execution_device):
    """Wall seconds of the scalar and native joint likelihood on matched points.

    The measured split is the matched-host timing record for the widened
    joint coverage; no speed ratio is asserted from it here.
    """
    device = _device(execution_device)
    _set, lights, joint = _joint_scenario()
    population = _bound_population(lights, joint, 1.0, False)
    reference = _tp_integrand(joint, population=population, hosted=True)
    native = joint_planet_callback(reference)(device=device)
    points = _orbit_points(reference, "tp", 96, 20261124)
    parameters = [reference.prior_transform(point) for point in points]
    scalar_started = time.perf_counter()
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    scalar_seconds = time.perf_counter() - scalar_started
    placed = models.jax.device_put(points, device)
    compiled = models.jax.jit(models.jax.vmap(native.evaluate))
    compiled(placed)
    native_started = time.perf_counter()
    evaluated = compiled(placed)
    models.jax.block_until_ready(evaluated.log_integrand)
    native_seconds = time.perf_counter() - native_started
    assert scalar_seconds > 0.0 and native_seconds > 0.0
    actual = np.asarray(evaluated.log_integrand, dtype=float)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)
    print("matched-host joint population timing on {0}: scalar={1:.3f}s native={2:.3f}s "
          "for {3} draws".format(execution_device, scalar_seconds, native_seconds, points.shape[0]))
