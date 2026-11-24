"""Actual EB integrand: native priors/splines/host grids/primary and secondary.

These are numerical/software development comparisons, not evidence calibration
or a held-out P6/P7/P8 result. All observations are analytic target-neutral data.
"""

import os

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence.nested_evidence import _EclipsingPairIntegrand
from tests.fixtures.synthetic_observations import synthetic_trex_eb_integrand_case


def _pair(mass, period, boxed, *, use_2xP=False, device=None):
    scene = synthetic_trex_eb_integrand_case(period)
    scalar = _EclipsingPairIntegrand(**scene, host_mass=mass, host_radius=mass,
                                    bin_integration_days=scene["exptime_days"] if boxed else 0.0,
                                    nsamples=3, population=None, n_dimensions=4, target_feh=0.0,
                                    use_2xP=use_2xP)
    factory = getattr(scalar, "jax_unit_integrand", None)
    assert callable(factory), "the production EB integrand has no pure-JAX unit-cube callback"
    return scalar, factory(device=models.jax.devices("cpu")[0] if device is None else device)


@pytest.mark.parametrize("mass", [0.12, 0.3, 0.63, 1.0, 3.0])
@pytest.mark.parametrize("period,boxed", [(2.0, False), (20.0, True)])
def test_complete_eb_integrand_matches_scalar_reference(mass, period, boxed):
    scalar, native = _pair(mass, period, boxed)
    points = np.random.default_rng(20261015).uniform(size=(32, 4))
    points[:2, 0] = (0.0, np.nextafter(1.0, 0.0))
    points[:2, 1] = (0.0, 1 - 1e-12)
    expected_parameters = np.asarray([scalar.prior_transform(point) for point in points])
    actual_parameters = np.asarray(models.jax.jit(models.jax.vmap(native.prior_transform))(points))
    np.testing.assert_allclose(actual_parameters, expected_parameters, rtol=1e-12, atol=1e-14)
    result = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    expected = np.asarray([scalar.log_likelihood(value) for value in expected_parameters])
    assert np.asarray(result.numerically_valid).all()
    np.testing.assert_array_equal(np.isneginf(result.log_integrand), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any(), "the fixture must exercise supported EB draws"
    np.testing.assert_allclose(np.asarray(result.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)
    assert native.claim_eligible is False
    program = str(models.jax.make_jaxpr(models.jax.vmap(native.evaluate))(points))
    assert "pure_callback" not in program and "io_callback" not in program


def test_eb_population_factory_never_silently_falls_back_to_cpu():
    scalar, _native = _pair(1.0, 2.0, False)
    scalar.hosted = True
    with pytest.raises(NotImplementedError, match="population"):
        scalar.jax_unit_integrand(device=models.jax.devices("cpu")[0])


def test_native_eb_uses_published_extensions_before_reference_nodes(monkeypatch):
    from dataclasses import replace
    from trex.stellar_light import host_limb_darkening as ldc

    grid = ldc.load_triceratops_tess_grid()
    temperatures = np.array([2949., 11000., 2000., 5750.])
    gravities = np.full(temperatures.size, 4.5)
    nodes = [grid.select(t, g, 0.) for t, g in zip(temperatures, gravities, strict=True)]
    assert [node.interpolated for node in nodes] == [True, True, False, False]
    assert [node.boundary_clamped for node in nodes] == [False, False, True, False]
    sparse = replace(grid, rows=tuple(row for row in grid.rows if row["Teff"] != 10000.))
    with monkeypatch.context() as patch:
        patch.setattr(ldc, "load_triceratops_tess_grid", lambda: sparse)
        _, native = _pair(1., 3., False)
    coefficients, valid, clamped = models.jax.jit(models.jax.vmap(native._grid_coefficients))(
        temperatures, gravities)
    np.testing.assert_allclose(coefficients, [(node.u1, node.u2) for node in nodes], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(valid, [True, True, True, True])
    np.testing.assert_array_equal(clamped, [False, False, True, False])


@pytest.mark.parametrize("twice", [False, True])
def test_native_eb_partition_matches_scalar(twice):
    """G21 §2.4.4 branch zeros must survive JIT without conditioning the q CDF."""
    from trex.populations.priors import eb_mass_ratio_support

    scalar, native = _pair(1.0, 3.0, False, use_2xP=twice)
    points = np.array([[0.2, 0.0, 0.25, 0.5], [0.99, 0.0, 0.25, 0.5]])
    parameters = np.array([scalar.prior_transform(point) for point in points])
    expected_support = eb_mass_ratio_support(parameters[:, 0], twice)
    assert expected_support.sum() == 1
    result = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    expected = np.array([scalar.log_likelihood(value) for value in parameters])
    np.testing.assert_array_equal(np.isfinite(result.log_integrand), expected_support)
    np.testing.assert_array_equal(np.isfinite(result.log_integrand), np.isfinite(expected))
    assert np.asarray(result.numerically_valid).all()


def test_eb_invalid_unit_inputs_remain_distinct_from_structural_zero():
    _scalar, native = _pair(1.0, 20.0, False)
    points = np.asarray([[0.5, 0.5, 0.5, 0.5], [-0.1, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5, np.nan]])
    result = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    assert np.isfinite(result.log_integrand[0])
    assert np.isneginf(result.log_integrand[1])
    assert np.isnan(result.log_integrand[2])
    assert np.asarray(result.numerically_valid).tolist() == [True, True, False]


def test_companion_outside_stellar_relations_is_zero_support_not_failure(monkeypatch):
    """The scalar EB returns -inf before evaluating an out-of-domain relation.

    The relation's NaN there must become a valid -inf, not the invalid NaN the
    native sampler treats as a numerical failure (NativeIntegrationError). The
    upper domain edge is lowered only to place a supported unit point outside
    it; the relation and the routing read the same module constant. Evaluation
    is eager so no compiled trace keeps the unpatched constant.
    """
    from trex.stellar import stellar_relations

    _scalar, native = _pair(1.0, 2.0, False)
    point = np.array([0.5, 0.0, 0.25, 0.5])
    inside = native.evaluate(point)
    assert bool(inside.numerically_valid)

    monkeypatch.setattr(stellar_relations, "STELLAR_RELATIONS_MASS_MAX_SOLAR",
                        stellar_relations.STELLAR_RELATIONS_MASS_MIN_SOLAR * 1.0001)
    outside = native.evaluate(point)
    assert bool(outside.numerically_valid)
    assert float(outside.log_integrand) == -np.inf


@pytest.fixture(params=["cpu", pytest.param("gpu:0", marks=pytest.mark.skipif(
    os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="requires an approved physical GPU host")),
    pytest.param("gpu:1", marks=pytest.mark.skipif(
        os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="requires an approved physical GPU host"))])
def low_mass_device(request):
    platform, _, index = request.param.partition(":")
    return models.jax.devices(platform)[int(index or 0)]


@pytest.mark.parametrize("mass", [0.105, 0.12, 0.3])
@pytest.mark.parametrize("twice", [False, True])
def test_m_dwarf_stellar_floor_and_branch_support_on_device(mass, twice, low_mass_device):
    """Real low-mass priors, no substituted stellar-domain constants.

    At 0.105 Msun the 0.1-Msun secondary floor forces q>0.95, so the
    ordinary EB slot is identically zero while EBx2P retains support.
    """
    scalar, native = _pair(mass, 2., False, use_2xP=twice, device=low_mass_device)
    quantiles = np.array([0., 1e-12, 0.1, 0.5, 0.9, np.nextafter(1., 0.)])
    points = np.column_stack((quantiles, np.zeros(6), np.full(6, .25), np.full(6, .5)))
    placed = models.jax.device_put(points, low_mass_device)
    transformed = models.jax.jit(models.jax.vmap(native.prior_transform))(placed)
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(placed)
    scalar_parameters = np.asarray([scalar.prior_transform(point) for point in points])
    expected = np.asarray([scalar.log_likelihood(p) for p in scalar_parameters])
    assert transformed.devices() == evaluated.log_integrand.devices() == {low_mass_device}
    assert evaluated.log_integrand.dtype == np.float64
    assert np.asarray(evaluated.numerically_valid).all()
    assert (np.asarray(transformed)[:, 0] * mass >= 0.1).all()
    np.testing.assert_allclose(transformed, scalar_parameters, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(np.isneginf(evaluated.log_integrand), np.isneginf(expected))
    finite = np.isfinite(expected)
    if mass == 0.105 and not twice:
        assert not finite.any()
    else:
        assert finite.any()
    np.testing.assert_allclose(np.asarray(evaluated.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)


def test_m_dwarf_structural_zeros_survive_importance_and_blackjax_entry(low_mass_device):
    """Actual ordered importance evaluator and checked sampler accept -inf lanes."""
    from trex.accelerators._native_unit_batch import NativeUnitCubeBatch
    from trex.evidence._blackjax_kernel import NativeNestedKernel

    scalar, native = _pair(.12, 2., False, device=low_mass_device)
    points = np.random.default_rng(20261116).uniform(size=(32, 4))
    points[:2] = [[0., 0., .25, .5], [np.nextafter(1., 0.), 0., .25, .5]]
    expected = np.array([scalar.log_likelihood(scalar.prior_transform(point)) for point in points])
    assert np.isfinite(expected).any() and np.isneginf(expected).any()
    batch = NativeUnitCubeBatch(scalar, points)
    with models.jax.default_device(low_mass_device):
        actual = batch(slice(0, len(points)))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)
    kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=32, num_delete=4, num_inner_steps=2)
    key = models.jax.device_put(models.jax.random.PRNGKey(20261116), low_mass_device)
    state = kernel.initialize(points, key)
    assert state.particles.loglikelihood.devices() == {low_mass_device}
    np.testing.assert_array_equal(np.isneginf(state.particles.loglikelihood), np.isneginf(expected))
    np.testing.assert_allclose(np.asarray(state.particles.loglikelihood)[finite], expected[finite],
                               rtol=1e-4, atol=1e-10)
    # A real constrained step also evaluates rejected proposals; it must not
    # confuse physical zeros with the sampler's numerical-failure channel.
    next_state, info = kernel.step(models.jax.random.fold_in(key, 1), state)
    assert np.asarray(info.update_info.is_accepted).all()
    assert not np.isnan(next_state.particles.loglikelihood).any()


def test_m_dwarf_numerical_failure_is_not_hidden_as_zero_support(low_mass_device):
    from trex.accelerators._native_unit_batch import NativeUnitCubeBatch
    from trex.evidence._blackjax_kernel import NativeNestedKernel
    from trex.foundation.errors import NumericalLikelihoodError

    scene = synthetic_trex_eb_integrand_case(2.)
    scene["sigma"] = np.finfo(np.float64).tiny
    scalar = _EclipsingPairIntegrand(**scene, host_mass=.12, host_radius=.12,
        bin_integration_days=0., nsamples=3, population=None, n_dimensions=4, target_feh=0.)
    native = scalar.jax_unit_integrand(device=low_mass_device)
    points = np.tile([0., 0., .25, .5], (8, 1))
    placed = models.jax.device_put(points, low_mass_device)
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(placed)
    assert not np.asarray(evaluated.numerically_valid).any()
    assert np.isnan(evaluated.log_integrand).all()
    with models.jax.default_device(low_mass_device), pytest.raises(NumericalLikelihoodError, match="numerical"):
        NativeUnitCubeBatch(scalar, points)(slice(0, len(points)))
    kernel = NativeNestedKernel(native, n_dimensions=4, n_live_points=8, num_delete=2, num_inner_steps=2)
    with pytest.raises(ValueError, match="numerical"):
        kernel.initialize(points, models.jax.device_put(models.jax.random.PRNGKey(20261116), low_mass_device))
