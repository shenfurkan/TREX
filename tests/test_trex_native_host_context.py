"""Native execution preserves supplied stellar context at real caller boundaries.

All stellar/population values are explicit target-neutral test fixtures. Missing
observations are never replaced. Small likelihood ceilings are computational
test controls only; no production draw budget or scientific engine is mocked.
"""

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import evidence_escalation as escalation
from trex.evidence import evidence_escalation_policy as escalation_policy
from trex.evidence import nested_evidence as nested
from tests.fixtures.synthetic_observations import (
    synthetic_trex_tp_integrand_case, synthetic_trex_eb_integrand_case,
    synthetic_trex_background_integrand_population,
)


@pytest.fixture(params=["cpu", "gpu:0", "gpu:1"])
def execution_device(request):
    platform, _, index = request.param.partition(":")
    try:
        devices = models.jax.devices(platform)
    except RuntimeError:
        devices = []
    index = int(index or 0)
    if index >= len(devices):
        pytest.skip(f"no physical {request.param} JAX device on approved host")
    with models.jax.default_device(devices[index]):
        yield devices[index]


def _population(kind):
    if kind == "bound":
        return nested.bound_companion_population(1.0, 1.0, 50.0, resolution_limit_arcsec=1.0)
    return nested.background_star_population(synthetic_trex_background_integrand_population(),
        simulated_area_sq_deg=1.0, target_tess_mag=10.0, resolution_limit_arcsec=1.0)


def _reference(kind, population, *, hosted=False, feh=-0.35):
    scene = (synthetic_trex_tp_integrand_case() if kind == "tp"
             else synthetic_trex_eb_integrand_case(2.0))
    kwargs = dict(scene, host_mass=1.0, host_radius=1.0, nsamples=3, bin_integration_days=0.0,
        population=population, hosted=hosted, target_feh=feh, n_dimensions=4 + population.n_dimensions)
    if kind == "tp":
        return nested._TransitingPlanetIntegrand(**kwargs, use_flat_priors=False)
    return nested._EclipsingPairIntegrand(**kwargs)


@pytest.mark.parametrize("kind", ["tp", "eb"])
@pytest.mark.parametrize("hosted", [False, True])
def test_bound_population_native_nested_retains_explicit_execution_device(execution_device, kind, hosted):
    reference = _reference(kind, _population("bound"), hosted=hosted)
    native = reference.jax_unit_integrand(device=execution_device)
    assert native.device == execution_device
    controls = nested._SamplerControls(32, 0.05, 32, 20261025, 1)
    result = nested._run_native_nested_sampler(reference, controls, reference.n_dimensions)
    assert result is not None and result.sampler == "blackjax-nss-experimental"
    assert result.n_likelihood_calls == 32
    assert result.initial_state.particles.position.devices() == {execution_device}
    assert result.initial_state.particles.loglikelihood.devices() == {execution_device}
    assert result.claim_eligible is result.evidence_calibrated is False


@pytest.mark.parametrize("index,population_kind", [(3, "bound"), (9, "background")])
def test_escalation_forwards_retained_feh_to_real_diluted_tp_integrand(tmp_path, monkeypatch, index, population_kind):
    # Use a real resumable dynesty call: the ceiling stops it after initialization,
    # which must defer while preserving the integrand's supplied metallicity.
    monkeypatch.setattr(escalation_policy, "_MAXIMUM_NESTED_LIKELIHOOD_CALLS", 401)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "dynesty")
    built = []
    construct = nested._TransitingPlanetIntegrand.__init__

    def observe(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(nested._TransitingPlanetIntegrand, "__init__", observe)
    population = _population(population_kind)
    token = escalation._escalation_runtime.set((tmp_path, float("inf"), []))
    try:
        with pytest.raises(RuntimeError, match="exhausted its registered 401-evaluation likelihood budget"):
            escalation._nested_call(escalation._target_plan_entry(index),
                blocks={population_kind: population}, **synthetic_trex_tp_integrand_case(), M_s_Msun=1.0,
                R_s_Rsun=1.0, bin_integration_days=0.0, target_aperture_flux_fraction=1.0, target_feh=-0.35,
                nsamples=3, random_seed=20261025, n_jobs=1)
    finally:
        escalation._escalation_runtime.reset(token)
    # These scenes' likelihood does not vary with [Fe/H], so the forwarded value is checked directly.
    assert len(built) == 1 and built[0].target_feh == -0.35
    reference = _reference("tp", population, feh=-0.35)
    reference.use_flat_priors = bool(escalation._target_plan_entry(index)["use_flat_priors"])
    points = np.random.default_rng(20261027).random((64, reference.n_dimensions))
    expected = np.asarray([reference.log_likelihood(reference.prior_transform(point)) for point in points])
    actual = np.asarray([built[0].log_likelihood(built[0].prior_transform(point)) for point in points])
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any()
    np.testing.assert_array_equal(actual[finite], expected[finite])


@pytest.mark.parametrize("population_kind", ["bound", "background"])
def test_diluting_tp_does_not_require_unused_host_metallicity(execution_device, population_kind):
    reference = _reference("tp", _population(population_kind), feh=None)
    controls = nested._SamplerControls(32, 0.05, 32, 20261025, 1)
    result = nested._run_native_nested_sampler(reference, controls, reference.n_dimensions)
    assert result is not None, "diluted TP must not fall back solely because unused host metallicity is absent"
    assert result.initial_state.particles.loglikelihood.devices() == {execution_device}
    native = reference.jax_unit_integrand(device=execution_device)
    assert native.hosted is False
    if population_kind == "background":
        np.testing.assert_array_equal(np.asarray(native.rows[3]), np.tile([reference.u1, reference.u2],
                                                                       (native.star_count, 1)))
    else:
        np.testing.assert_array_equal(np.asarray(native.base.u_coeffs), [reference.u1, reference.u2])
        assert native.stellar is None  # no unused eclipsed atmosphere constructed
    expected = np.asarray([reference.log_likelihood(reference.prior_transform(point))
                           for point in result.initial_positions])
    actual = np.asarray(result.initial_state.particles.loglikelihood)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any()
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("population_kind", ["bound", "background"])
@pytest.mark.parametrize("kind", ["tp", "eb"])
def test_truly_missing_population_host_metallicity_is_not_fabricated(execution_device, population_kind, kind):
    population = _population(population_kind)
    if population_kind == "background":
        # Source [M/H], not target [Fe/H], defines a background pair's atmosphere.
        reference = _reference(kind, population, hosted=True, feh=None)
        assert reference.jax_unit_integrand(device=execution_device) is not None
        population = population._replace(mh_dex=np.full_like(population.mh_dex, np.nan))
        reference = _reference(kind, population, hosted=True, feh=-0.35)
        with pytest.raises(ValueError, match="metallicity"):
            reference.jax_unit_integrand(device=execution_device)
    else:
        reference = _reference(kind, population, hosted=True, feh=None)
        with pytest.raises(ValueError, match="target_feh"):
            reference.jax_unit_integrand(device=execution_device)


def test_missing_host_mass_is_not_replaced_to_enable_native_execution(execution_device):
    reference = _reference("tp", _population("bound"))
    assert np.isfinite(reference.period_days) and reference.period_days > 0.0
    reference.host_mass = float("nan")
    # Bound TP setup recomputes Kepler's axis before the fixed-host adapter validates its fields.
    with pytest.raises(ValueError, match="orbital period and total mass must be finite positive real numbers"):
        reference.jax_unit_integrand(device=execution_device)


def test_default_nested_estimator_does_not_become_blackjax(monkeypatch):
    monkeypatch.delenv("TREX_NESTED_ESTIMATOR", raising=False)
    # Inspect the production opt-in's default without replacing an engine.
    assert nested._SAMPLER_NAME == "dynesty.DynamicNestedSampler"
