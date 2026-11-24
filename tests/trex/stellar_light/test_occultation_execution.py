"""Sparse execution must reproduce the original dense 64-node angular integral."""

import numpy as np
import pytest

from trex.stellar_light import occultation as occ


def _dense_reference(separation, k, mu, intensity):
    """Frozen pre-optimization v2 arithmetic, including every zero contribution.

    This is an execution-parity oracle, not independent atmospheric calibration.
    The independent analytic/adaptive physical oracles remain in test_occultation.
    """
    intensity = intensity/intensity.max()
    a, b = mu[:, :-1], mu[:, 1:]
    ia, ib = intensity[:, :-1], intensity[:, 1:]
    moment = np.sum((b-a)*((2*a+b)*ia+(a+2*b)*ib)/6)
    d = np.asarray(separation)[..., None, None]
    lo = np.maximum(a, np.sqrt(np.maximum(1-(d+k)**2, 0.)))
    hi = np.minimum(b, np.sqrt(np.maximum(1-(d-k)**2, 0.)))
    width = np.maximum(hi-lo, 0.)
    points, weights = np.polynomial.legendre.leggauss(64)
    angle = (points+1)*np.pi/2
    sample = lo[..., None]+width[..., None]*(1-np.cos(angle))/2
    rule = weights*np.pi/4*np.sin(angle)*width[..., None]
    value = ia[..., None]+(ib-ia)[..., None]*(sample-a[..., None])/(b-a)[..., None]
    r = np.sqrt(np.maximum(1-sample**2, 0.))
    divisor = 2*d[..., None]*r
    cosine = (r*r+d[..., None]**2-k*k)/np.where(divisor > 0, divisor, 1.)
    partial = np.sum(value*sample*2*np.arccos(np.clip(cosine, -1., 1.))*rule, axis=(-3, -2, -1))
    start = np.minimum(np.maximum(a, np.sqrt(np.maximum(1-np.maximum(k-d, 0.)**2, 0.))), b)
    initial = ia+(ib-ia)*(start-a)/(b-a)
    full = 2*np.pi*np.sum((b-start)*((2*start+b)*initial+(start+2*b)*ib)/6, axis=(-2, -1))
    return (partial+full)/(2*np.pi*moment)


@pytest.mark.parametrize("k", [.01, .3, .9, 1., 1.2, 2.])
def test_sparse_rings_and_exact_disc_limits_match_dense_rule_for_every_original_ray(k):
    mu = np.array([[0., .01, .3, .7, 1.], [0., .05, .2, .8, 1.], [0., .2, .4, .9, 1.]])
    intensity = np.array([[0., .2, .6, .9, 1.], [0., 0., 0., 0., 0.], [0., 0., .2, .8, 1.2]])
    d = np.r_[np.linspace(0., 1+k, 31), 1+k+1e-12].reshape(4, 8)
    expected = _dense_reference(d, k, mu, intensity)
    np.testing.assert_allclose(occ.occulted_fraction(d, k, mu, intensity), expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(occ.disc_flux(d, k, (mu, intensity)), 1-expected, atol=2e-14, rtol=0)
    # Reuse must not retain the previous radius ratio or intensity field.
    intensity[0] *= 3
    np.testing.assert_allclose(occ.disc_flux(d, k/2, (mu, intensity)),
                               1-_dense_reference(d, k/2, mu, intensity), atol=2e-14, rtol=0)


def test_exact_no_overlap_and_total_eclipse_need_no_partial_ring_work(monkeypatch):
    calls = []
    original = occ.occulted_fraction
    def measured(d, *args, **kwargs):
        calls.append(np.asarray(d).copy())
        return original(d, *args, **kwargs)
    monkeypatch.setattr(occ, "occulted_fraction", measured)
    profile = np.array([0., .3, 1.]), np.array([0., .7, 1.])
    np.testing.assert_array_equal(occ.disc_flux([0., .1, 2.2, 3.], 1.2, profile), [0., 0., 1., 1.])
    assert not calls
    result = occ.disc_flux([.5, 3., 1., 0.], 1.2, profile)
    np.testing.assert_array_equal(calls[0], [.5, 1.])
    assert result[1] == 1. and result[3] == 0.


@pytest.mark.parametrize("compiled", [False, True])
def test_native_compaction_and_dense_fallback_match_the_frozen_dense_rule(monkeypatch, compiled):
    from trex.accelerators import _jax_models as jm

    if not jm.HAS_JAX_TREX:
        pytest.skip("JAX float64 backend is unavailable")
    mu = np.array([[0., .01, .3, .7, 1.], [0., .05, .2, .8, 1.], [0., .2, .4, .9, 1.]])
    intensity = np.array([[0., .2, .6, .9, 1.], [0., 0., 0., 0., 0.], [0., 0., .2, .8, 1.2]])
    k = .9
    mu_j, intensity_j = jm.jnp.asarray(mu), jm.jnp.asarray(intensity)

    def evaluate(d, radius_ratio):
        return occ.occulted_fraction(d, radius_ratio, mu_j, intensity_j, xp=jm.jnp)

    operation = jm.jax.jit(evaluate) if compiled else evaluate

    def native(d, radius_ratio=k):
        return np.asarray(operation(jm.jnp.asarray(d), jm.jnp.asarray(radius_ratio)))

    scalars = np.array([0., .1, .5, 1., 1.2, 1.9])
    for d in scalars:
        assert native(d).item() == pytest.approx(_dense_reference(d, k, mu, intensity).item(), abs=2e-14, rel=0)
    batched = np.array([[0., .1, .5, 1.], [1.2, 1.9, 2.2, 3.]])
    np.testing.assert_allclose(native(batched), _dense_reference(batched, k, mu, intensity),
                               atol=2e-14, rtol=0)
    assert native(0., 1.2).item() == pytest.approx(1., abs=2e-15, rel=0)
    # Fewer gather slots than blocking intervals must select the dense rule, not a truncated sum.
    monkeypatch.setattr(occ, "_RAY_INTERVAL_CAPACITY", 2)
    if compiled:
        # A fresh trace must read the changed static capacity, not reuse the first executable.
        operation = jm.jax.jit(lambda d, radius_ratio: evaluate(d, radius_ratio))
    for d in scalars:
        assert native(d).item() == pytest.approx(_dense_reference(d, k, mu, intensity).item(), abs=2e-14, rel=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("k", [.4, 1., 1.4])
def test_native_disc_flux_keeps_contacts_front_mask_and_dense_parity(k, compiled):
    from trex.accelerators import _jax_models as jm
    from trex.evidence._jax_unit_integrand import _disc_flux

    if not jm.HAS_JAX_TREX:
        pytest.skip("JAX float64 backend is unavailable")
    mu = np.array([[0., .05, .3, .8, 1.]])
    intensity = np.array([[0., .2, .7, 1., 1.2]])
    contacts = np.array([max(k-1, 0.), 1+k])
    d = np.r_[0., contacts, np.nextafter(contacts, np.inf),
              np.maximum(np.nextafter(contacts, -np.inf), 0.), .5, 1., 3., 4.]
    # More than one map batch exercises both the 32-lane body and its remainder.
    d = np.tile(d, 3)
    front = np.arange(d.size) % 3 != 0
    profile = tuple(jm.jnp.asarray(value) for value in (mu, intensity))

    def evaluate(distance, in_front, rays):
        return _disc_flux(distance, k, jm.jnp.array([.2, .1]), in_front, angular_profile=rays)

    operation = jm.jax.jit(evaluate) if compiled else evaluate
    actual = np.asarray(operation(jm.jnp.asarray(d), jm.jnp.asarray(front), profile))
    expected = np.where(front, occ.disc_flux(d, k, (mu, intensity)), 1.)
    dense = np.where(front, 1-_dense_reference(d, k, mu, intensity), 1.)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual, dense, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(actual[~front | (d >= 1+k)], 1.)
    np.testing.assert_array_equal(actual[front & (d <= k-1)], 0.)

    invalid_d = jm.jnp.asarray([np.nan, np.inf, -1.])
    assert np.isnan(operation(invalid_d, jm.jnp.ones(3, dtype=bool), profile)).all()
    invalid_profile = profile[0], jm.jnp.full_like(profile[1], jm.jnp.nan)
    # Contact limits must not disguise an unavailable atmosphere as valid light.
    assert np.isnan(operation(jm.jnp.asarray(d), jm.jnp.ones(d.size, dtype=bool), invalid_profile)).all()


def test_only_parameter_independent_quadrature_is_reused():
    occ._ring_rule.cache_clear()
    first = occ._ring_rule(64)
    assert occ._ring_rule(64) is first
    assert occ._ring_rule(32)[0].size == 32
    assert all(not values.flags.writeable for values in first)
