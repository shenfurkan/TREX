"""Conservative contact windows must contain every real eclipse sample.

Oracles here are independent of the window construction: the production
quadratic disc flux (``_quadratic_disc_flux``) for the primary and the shared
pair separation with ``core_limb_dark`` for the secondary decide which samples
are eclipsed; the window must contain all of them. The analytic circular window
and the period/offset scaling are checked in closed form. Containment is
one-sided: a wider window is allowed (and is the documented conservative
contract), a missed eclipse is not. The explicit companion-frame orbit parity
lives in ``test_shared_occultation_geometry.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trex.accelerators import _jax_models

pytestmark = pytest.mark.skipif(
    not _jax_models.HAS_JAX_TREX, reason="JAX/jaxoplanet unavailable"
)

PERIOD_DAYS = 10.0
U_HOST = (0.40, 0.20)
U_COMPANION = (0.25, 0.15)


def _fluxes(time, ecc, argp, a_rs, k, inc, u_coeffs):
    from trex.evidence._jax_unit_integrand import _quadratic_disc_flux

    flux, valid = _quadratic_disc_flux(
        _jax_models.jnp.asarray(time), PERIOD_DAYS, k, a_rs, ecc, argp, inc,
        _jax_models.jnp.asarray(u_coeffs),
    )
    assert bool(_jax_models.jnp.all(valid))
    return np.asarray(flux)


def _eclipsed(flux):
    """Strictly reduced flux, away from the exact contact boundary and rounding."""
    return flux < 1.0 - 1e-12


def _mean_anomaly(time, ecc, argp):
    from trex.accelerators import _jax_orbits

    return np.asarray(_jax_orbits._mean_anomaly_at(
        _jax_models.jnp.asarray(time), PERIOD_DAYS, ecc, argp, 0.0))


def test_circular_window_matches_the_analytic_half_angle():
    from trex.accelerators import _kepler_culling as culling

    a_rs, k, offset = 20.0, 0.1, 0.0
    lo_p, width_p, lo_s, width_s = culling.contact_windows(0.0, 0.0, a_rs, k, PERIOD_DAYS, offset)
    delta = math.asin((1.0 + k) / a_rs)
    np.testing.assert_allclose(float(lo_p), 0.5 * math.pi - delta, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(float(width_p), 2.0 * delta, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(float(lo_s), 1.5 * math.pi - delta, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(float(width_s), 2.0 * delta, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "ecc,argp,a_rs,k,inc,span",
    [
        (0.3, 40.0, 15.0, 0.12, 88.7, 2.0 * PERIOD_DAYS),
        # Near-unity eccentricity: detached at periastron (200 * 0.05 >> 1.1)
        # but the conjunction sits near apastron, where the mean-anomaly window
        # is far narrower than the true-anomaly window and cancellation matters.
        (0.95, 40.0, 200.0, 0.1, 89.5, 0.1),
    ],
    ids=["moderate", "high-eccentricity"],
)
def test_primary_eclipse_samples_lie_inside_the_primary_window(ecc, argp, a_rs, k, inc, span):
    from trex.accelerators import _kepler_culling as culling

    time = np.linspace(-span, span, 4001)
    flux = _fluxes(time, ecc, argp, a_rs, k, inc, U_HOST)
    eclipse = _eclipsed(flux)
    assert eclipse.sum() > 10, "fixture must contain a real transit, not an empty set"

    lo, width, _lo_s, _width_s = culling.contact_windows(ecc, argp, a_rs, k, PERIOD_DAYS, 0.0)
    inside = culling.in_contact_window(_mean_anomaly(time, ecc, argp), lo, width)
    assert bool(np.asarray(inside)[eclipse].all())
    # The window must actually cull, otherwise containment is vacuous.
    assert not bool(np.asarray(inside).all())


def test_secondary_eclipse_samples_lie_inside_the_secondary_window():
    from trex.accelerators import _jax_orbits, _kepler_culling as culling
    from trex.accelerators._jax_models import core_limb_dark, jnp

    ecc, argp, a_rs, k, inc = 0.3, 40.0, 15.0, 0.2, 88.7
    time = np.linspace(-2.0 * PERIOD_DAYS, 2.0 * PERIOD_DAYS, 4001)
    d, front = _jax_orbits._sky_geometry(
        jnp.asarray(time), PERIOD_DAYS, a_rs, inc, ecc, argp, 0.0)
    secondary = np.asarray(1.0 + core_limb_dark(
        jnp.asarray(U_COMPANION), jnp.where(front, jnp.inf, d / k), 1.0 / k))
    eclipse = _eclipsed(secondary)
    assert eclipse.sum() > 10

    mean_anomaly = _mean_anomaly(time, ecc, argp)
    lo_p, width_p, lo_s, width_s = culling.contact_windows(
        ecc, argp, a_rs, k, PERIOD_DAYS, 0.0)
    inside = culling.in_contact_window(mean_anomaly, lo_s, width_s)
    assert bool(np.asarray(inside)[eclipse].all())
    primary_only = culling.in_contact_window(mean_anomaly, lo_p, width_p)
    assert not bool(np.asarray(primary_only)[eclipse].any())
    assert not bool(np.asarray(inside).all())


def test_per_bin_quadrature_offsets_are_covered_by_the_window_margin():
    from trex.accelerators import _kepler_culling as culling
    from trex.foundation._numerics import _per_bin_two_box_quadrature

    ecc, argp, a_rs, k, inc = 0.2, 30.0, 12.0, 0.1, 88.9
    exposure_days = 0.002
    centres = np.linspace(-0.35, 0.35, 24)
    widths = np.resize(np.array([0.008, 0.021, 0.011, 0.016]), centres.size)
    offsets, weights = _per_bin_two_box_quadrature(widths, exposure_days)
    assert offsets.shape == weights.shape == (centres.size, offsets.shape[1])

    node_time = centres[:, None] + offsets
    node_flux = _fluxes(node_time.reshape(-1), ecc, argp, a_rs, k, inc, U_HOST).reshape(node_time.shape)
    contaminated = np.any(node_flux < 1.0 - 1e-12, axis=1)
    assert contaminated.sum() > 3

    max_offset = float(np.max(np.abs(offsets)))
    lo, width, _lo_s, _width_s = culling.contact_windows(
        ecc, argp, a_rs, k, PERIOD_DAYS, max_offset)
    inside = culling.in_contact_window(_mean_anomaly(centres, ecc, argp), lo, width)
    assert bool(np.asarray(inside)[contaminated].all())


def test_exposure_only_offsets_are_covered_by_the_window_margin():
    from trex.accelerators import _kepler_culling as culling

    ecc, argp, a_rs, k, inc = 0.2, 30.0, 12.0, 0.1, 88.9
    exposure_days = 0.004
    centres = np.linspace(-0.3, 0.3, 601)
    node_time = centres[:, None] + np.array([-0.5, 0.5]) * exposure_days
    node_flux = _fluxes(node_time.reshape(-1), ecc, argp, a_rs, k, inc, U_HOST).reshape(node_time.shape)
    contaminated = np.any(node_flux < 1.0 - 1e-12, axis=1)
    assert contaminated.sum() > 3

    lo, width, _lo_s, _width_s = culling.contact_windows(
        ecc, argp, a_rs, k, PERIOD_DAYS, 0.5 * exposure_days)
    inside = culling.in_contact_window(_mean_anomaly(centres, ecc, argp), lo, width)
    assert bool(np.asarray(inside)[contaminated].all())


def test_window_wraps_at_the_signed_pi_cut():
    """omega = 270 deg puts the primary conjunction exactly on M = -pi."""
    from trex.accelerators import _kepler_culling as culling

    ecc, argp, a_rs, k, inc = 0.2, 270.0, 15.0, 0.1, 88.5
    time = np.linspace(-0.3, 0.3, 1201)
    flux = _fluxes(time, ecc, argp, a_rs, k, inc, U_HOST)
    eclipse = _eclipsed(flux)
    assert eclipse.sum() > 3

    lo, width, _lo_s, _width_s = culling.contact_windows(ecc, argp, a_rs, k, PERIOD_DAYS, 0.0)
    mean_anomaly = _mean_anomaly(time, ecc, argp)
    assert np.min(mean_anomaly) < -math.pi < np.max(mean_anomaly)
    inside = culling.in_contact_window(mean_anomaly, lo, width)
    assert bool(np.asarray(inside)[eclipse].all())


def test_doubled_period_only_halves_the_offset_margin():
    from trex.accelerators import _kepler_culling as culling

    ecc, argp, a_rs, k, offset = 0.3, 40.0, 15.0, 0.1, 0.01
    _lo_p1, width_p1, _lo_s1, width_s1 = culling.contact_windows(
        ecc, argp, a_rs, k, PERIOD_DAYS, offset)
    _lo_p2, width_p2, _lo_s2, width_s2 = culling.contact_windows(
        ecc, argp, a_rs, k, 2.0 * PERIOD_DAYS, offset)
    margin = 2.0 * (2.0 * math.pi / PERIOD_DAYS) * offset
    np.testing.assert_allclose(float(width_p1) - float(width_p2), 0.5 * margin, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(float(width_s1) - float(width_s2), 0.5 * margin, rtol=0.0, atol=1e-13)


def test_non_detached_orbit_cannot_be_culled_anywhere():
    from trex.accelerators import _kepler_culling as culling

    ecc, argp, a_rs, k = 0.5, 17.0, 2.0, 0.1  # a (1 - e) = 1.0 <= 1 + k
    lo_p, width_p, lo_s, width_s = culling.contact_windows(ecc, argp, a_rs, k, PERIOD_DAYS, 0.0)
    # A single-window caller must also keep everything: the bound is vacuous.
    assert float(width_p) == pytest.approx(2.0 * math.pi, abs=1e-12)
    assert float(width_s) == pytest.approx(2.0 * math.pi, abs=1e-12)
    mean_anomaly = np.linspace(-math.pi, math.pi, 721)
    inside = (np.asarray(culling.in_contact_window(mean_anomaly, lo_p, width_p))
              | np.asarray(culling.in_contact_window(mean_anomaly, lo_s, width_s)))
    assert inside.all()


def test_long_period_window_culls_most_of_the_orbit():
    from trex.accelerators import _kepler_culling as culling

    lo_p, width_p, lo_s, width_s = culling.contact_windows(
        0.1, 20.0, 200.0, 0.01, 100.0, 0.0)
    mean_anomaly = np.linspace(-math.pi, math.pi, 20001, endpoint=False)
    inside = (np.asarray(culling.in_contact_window(mean_anomaly, lo_p, width_p))
              | np.asarray(culling.in_contact_window(mean_anomaly, lo_s, width_s)))
    assert 0.0 < inside.mean() < 0.02


@pytest.mark.parametrize("compiled", [False, True])
def test_invalid_inputs_fall_back_to_a_full_turn(compiled):
    from trex.accelerators import _kepler_culling as culling

    cases = [
        (math.nan, 40.0, 15.0, 0.1, PERIOD_DAYS, 0.0),
        (0.3, math.inf, 15.0, 0.1, PERIOD_DAYS, 0.0),
        (0.3, 40.0, math.nan, 0.1, PERIOD_DAYS, 0.0),
        (0.3, 40.0, 0.0, 0.1, PERIOD_DAYS, 0.0),
        (0.3, 40.0, 15.0, 0.0, PERIOD_DAYS, 0.0),
        (1.0, 40.0, 15.0, 0.1, PERIOD_DAYS, 0.0),
        (0.3, 40.0, 15.0, 0.1, 0.0, 0.0),
        (0.3, 40.0, 15.0, 0.1, PERIOD_DAYS, -1.0),
    ]
    operation = _jax_models.jax.jit(culling.contact_windows) if compiled else culling.contact_windows
    for case in cases:
        lo_p, width_p, lo_s, width_s = operation(*case)
        assert np.isfinite([lo_p, width_p, lo_s, width_s]).all()
        assert float(width_p) == pytest.approx(2.0 * math.pi, abs=1e-12)
        assert float(width_s) == pytest.approx(2.0 * math.pi, abs=1e-12)

    nan = math.nan
    assert not bool(culling.in_contact_window(nan, 0.0, 2.0 * math.pi))
    assert bool(culling.in_contact_window(0.0, 0.0, 2.0 * math.pi))
