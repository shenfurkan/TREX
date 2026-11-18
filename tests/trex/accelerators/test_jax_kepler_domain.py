"""Device Kepler solve and sky geometry over the eccentricity domain the priors admit.

Reference: a vectorized bisection solution of ``E - e sin E = M``. Bisection
shares no starting value, derivative, iteration count or convergence test with
the production solver, so a shared iteration defect cannot pass. Each tolerance
below is derived from the conditioning of ``f(E)`` or from an established model
tolerance, not fitted to the returned values.
"""

from __future__ import annotations

import numpy as np
import pytest

from trex.accelerators import _jax_models

pytestmark = pytest.mark.skipif(
    not _jax_models.HAS_JAX_TREX, reason="JAX/jaxoplanet unavailable"
)

# Circular, the replaced 0.93 routing point, the range where batman's solver
# stops returning finite values, the detached tail production can reach
# (1 - e near 1e-3), and 0.999 as margin beyond the geometric mask.
ECCENTRICITIES = (0.0, 1e-12, 0.3, 0.5, 0.8, 0.93, 0.95, 0.97, 0.99, 0.999)


def _bisection_true_anomaly(mean_anomaly, ecc, iterations=200):
    """sin(f), cos(f) from a bisection solution of Kepler's equation.

    ``|E - M| = e |sin E| <= e`` brackets the root in ``[M - e, M + e]`` for
    every e < 1, so no starting guess or derivative is needed. 200 halvings
    exceed the 53-bit mantissa; the midpoint then stalls on the adjacent
    representable values, which is the float64 root.
    """
    M = np.asarray(mean_anomaly, dtype=float)
    e = np.asarray(ecc, dtype=float)
    M, e = np.broadcast_arrays(M, e)
    low, high = M - e, M + e
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        below = middle - e * np.sin(middle) - M < 0.0
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    eccentric = 0.5 * (low + high)
    # Half-angle form: stable for E near 0 and pi, and independent of the
    # solver's own tan(f/2) transformation.
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * eccentric),
        np.sqrt(1.0 - e) * np.cos(0.5 * eccentric),
    )
    return np.sin(true_anomaly), np.cos(true_anomaly)


def _device_true_anomaly(mean_anomaly, ecc):
    """The production float64 JAX solve, as arrays."""
    import jax.numpy as jnp
    from trex.accelerators._jax_orbits import _kepler_true_anomaly

    sin_f, cos_f = _kepler_true_anomaly(
        jnp.asarray(mean_anomaly, dtype=jnp.float64), jnp.asarray(ecc, dtype=jnp.float64)
    )
    return np.asarray(sin_f), np.asarray(cos_f)


def _reference_sky_separation(time, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days=0.0):
    """Sky separation and front/behind flag from the bisection orbit.

    Same convention as the device path: ``t0`` is inferior conjunction, where
    the argument of latitude is pi/2, and the separation follows Winn (2010)
    Eq. 5 with the orbit evaluated independently.
    """
    omega = np.radians(argp_deg)
    conjunction = 0.5 * np.pi - omega
    eccentric = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(0.5 * conjunction),
        np.sqrt(1.0 + ecc) * np.cos(0.5 * conjunction),
    )
    mean_anomaly = eccentric - ecc * np.sin(eccentric) + 2.0 * np.pi * (time - t0_days) / period_days
    sin_f, cos_f = _bisection_true_anomaly(mean_anomaly, ecc)
    sin_theta = np.sin(omega) * cos_f + np.cos(omega) * sin_f
    cos_theta = np.cos(omega) * cos_f - np.sin(omega) * sin_f
    separation = a_rs * (1.0 - ecc ** 2) / (1.0 + ecc * cos_f)
    distance = separation * np.sqrt(cos_theta ** 2 + (sin_theta * np.cos(np.radians(inc_deg))) ** 2)
    return distance, sin_theta


def test_device_solver_agrees_with_an_independent_bisection_solution(record_property):
    """sin f and cos f match a bisection solve across the admitted domain.

    A float64 root of the same equation differs from another float64 root only
    by the conditioning of f(E), whose largest derivative is
    ``sqrt(2/(1-e))`` at periastron; a few ulp of |E| <= 2pi times that
    derivative, with one rounding from the half-angle transform, is the
    derived bound. The 1e-10 floor is orders below the 3.7e-6 batman agreement
    already measured at e <= 0.93, so it bounds the observable separation far
    more tightly than any evidence comparison needs.
    """
    mean_anomaly = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 4001)
    spacing = np.finfo(float).eps * np.max(np.abs(mean_anomaly))
    for ecc in ECCENTRICITIES:
        sin_f, cos_f = _device_true_anomaly(mean_anomaly, ecc)
        reference_sin, reference_cos = _bisection_true_anomaly(mean_anomaly, ecc)
        assert np.all(np.isfinite(sin_f)) and np.all(np.isfinite(cos_f)), ecc
        assert np.max(np.abs(sin_f ** 2 + cos_f ** 2 - 1.0)) <= 1e-14, ecc
        bound = max(1e-10, 8.0 * np.sqrt(2.0 / (1.0 - ecc)) * spacing)
        worst = float(max(np.max(np.abs(sin_f - reference_sin)),
                          np.max(np.abs(cos_f - reference_cos))))
        record_property(f"worst_e{ecc:g}", f"{worst:.3e}")
        record_property(f"bound_e{ecc:g}", f"{bound:.3e}")
        assert worst <= bound, (ecc, worst, bound)


def test_circular_limit_is_the_mean_anomaly():
    """e = 0 has f = M exactly; a circular orbit must not perturb the phase."""
    mean_anomaly = np.linspace(-6.0 * np.pi, 6.0 * np.pi, 4001)
    sin_f, cos_f = _device_true_anomaly(mean_anomaly, 0.0)
    assert np.max(np.abs(sin_f - np.sin(mean_anomaly))) <= 1e-15
    assert np.max(np.abs(cos_f - np.cos(mean_anomaly))) <= 1e-15


@pytest.mark.parametrize("ecc", ECCENTRICITIES)
def test_periastron_and_apoastron_return_unit_vectors(ecc):
    """M = 0 is periastron (f = 0) and M = pi is apoastron (f = pi) for any e."""
    sin_f, cos_f = _device_true_anomaly(np.array([0.0, np.pi]), ecc)
    assert abs(sin_f[0]) <= 1e-15 and abs(cos_f[0] - 1.0) <= 1e-15
    assert abs(sin_f[1]) <= 1e-15 and abs(cos_f[1] + 1.0) <= 1e-15


def test_wrapped_mean_anomalies_give_the_same_true_anomaly():
    """sin f and cos f are periodic in the mean anomaly, not linear in it.

    The solver reduces modulo one turn; a draw offset by whole orbits must give
    the same pair, not a different phase or a different branch.
    """
    mean_anomaly = np.linspace(-1.5 * np.pi, 1.5 * np.pi, 701)
    for ecc in (0.0, 0.5, 0.93, 0.999):
        base = np.stack(_device_true_anomaly(mean_anomaly, ecc))
        for turns in (-2, 1, 5):
            shifted = np.stack(_device_true_anomaly(mean_anomaly + turns * 2.0 * np.pi, ecc))
            assert np.max(np.abs(shifted - base)) <= 1e-12, (ecc, turns)


def test_mean_anomaly_round_trip_agrees_with_the_production_inverse():
    """f(M) fed back through the production inverse returns M.

    ``_mean_anomaly`` is the same convention the sky geometry starts from, so
    this checks the composed orbit step, not only the forward solve. The
    inverse is well conditioned (dM/df amplification is at most
    ``2 sqrt(2/(1-e))`` times the f-conditioning), hence the 1e-9 bound; a
    solver that stalls near periastron leaves errors many orders larger.
    """
    import jax.numpy as jnp
    from trex.accelerators._jax_orbits import _mean_anomaly

    mean_anomaly = np.linspace(-np.pi, np.pi, 4001)
    wrapped = (mean_anomaly + np.pi) % (2.0 * np.pi) - np.pi
    for ecc in ECCENTRICITIES:
        sin_f, cos_f = _device_true_anomaly(mean_anomaly, ecc)
        recovered = np.asarray(_mean_anomaly(
            jnp.asarray(np.arctan2(sin_f, cos_f), dtype=jnp.float64), jnp.asarray(ecc)))
        difference = np.abs((recovered - wrapped + np.pi) % (2.0 * np.pi) - np.pi)
        assert np.max(difference) <= 1e-9, (ecc, float(np.max(difference)))


@pytest.mark.parametrize("ecc", (0.9, 0.97, 0.99, 0.999))
def test_sky_separation_matches_independent_orbital_geometry(ecc):
    """The JAX sky separation follows the independently solved orbit.

    Tolerances scale with the separation: at the transit-relevant d ~ 1 ratio a
    1e-10 relative separation error is far below the 3.7e-6 batman agreement
    already measured at e <= 0.93. Times at conjunction are compared in
    separation only; the front/behind test is a step function there and is
    checked away from |sin(theta)| = 0.
    """
    import jax.numpy as jnp
    from trex.accelerators._jax_orbits import _jax_batman_sky_separation

    period, a_rs, inc_deg = 40.0, 300.0, 89.4
    time = np.linspace(-0.6 * period, 0.6 * period, 601)
    for argp in (0.0, 90.0, 180.0, 270.0):
        reference, sin_theta = _reference_sky_separation(
            time, period, a_rs, inc_deg, ecc, argp)
        device = np.asarray(_jax_batman_sky_separation(
            jnp.asarray(time, dtype=jnp.float64), jnp.asarray(period), jnp.asarray(a_rs),
            jnp.asarray(inc_deg), jnp.asarray(ecc), jnp.asarray(argp), jnp.asarray(0.0)))
        away = np.abs(sin_theta) > 1e-9
        front = sin_theta > 0.0
        assert np.array_equal(np.isfinite(device)[away], front[away]), (ecc, argp)
        assert np.max(np.abs(device[away & front] - reference[away & front])
                      / np.maximum(1.0, reference[away & front])) <= 1e-10, (ecc, argp)
        assert np.all(np.isposinf(device[away & ~front])), (ecc, argp)


@pytest.mark.parametrize("ecc", (0.0, 0.5, 0.9, 0.97, 0.99, 0.999))
def test_transit_flux_follows_an_independently_solved_orbit(ecc):
    """The full device flux, not just the separation, follows the reference orbit.

    The limb-darkened kernel is the production one; feeding it the
    bisection-solved separation isolates the Kepler solve. The occultation
    profile's derivative near contact is O(1) per stellar radius, so the 1e-8
    flux bound is well above the 1e-10 separation agreement and still far
    below the 3.7e-6 batman agreement measured at e <= 0.93.
    """
    import jax.numpy as jnp
    from trex.accelerators._jax_models import core_limb_dark
    from trex.accelerators._jax_orbits import _jax_transit

    period, a_rs, inc_deg, rp_rs = 40.0, 300.0, 89.4, 0.1
    u1, u2 = 0.4, 0.2
    time = np.linspace(-0.04 * period, 0.04 * period, 401)
    for argp in (0.0, 90.0, 180.0, 270.0):
        reference, sin_theta = _reference_sky_separation(
            time, period, a_rs, inc_deg, ecc, argp)
        separation = jnp.asarray(reference, dtype=jnp.float64)
        # Far-side sentinel exactly as the device geometry applies it.
        instantaneous = 1.0 + core_limb_dark(
            jnp.array([u1, u2], dtype=jnp.float64),
            jnp.where(jnp.asarray(sin_theta > 0.0), separation, jnp.inf),
            jnp.asarray(rp_rs),
        )
        device = np.asarray(_jax_transit(
            time, period, rp_rs, a_rs, inc_deg, u1, u2, 0.0, ecc=ecc, argp_deg=argp,
            nsamples=1,
        ))
        assert np.all(np.isfinite(device)), (ecc, argp)
        assert np.max(np.abs(device - np.asarray(instantaneous))) <= 1e-8, (ecc, argp)
