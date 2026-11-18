"""One Kepler geometry for the EB primary and secondary, without a second orbit.

The oracle for the pair split is the CPU reference's explicit companion-frame
orbit (``a/k``, ``omega + 180``, mid-eclipse at the secondary phase); the
oracle for the composed model is an explicitly built primary plus secondary
from ``core_limb_dark`` and the shared ``binary_flux_weights``. The far-side
case reproduces the regression where a finite batman sentinel let an occultor
of radius 150 cover the disc from behind. None of these checks establishes
physical accuracy; they pin the shared-geometry contract.
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
ECC = 0.3
ARGP = 40.0
A_RS = 15.0
INC = 88.7
K = 0.2
U_HOST = (0.55, 0.15)
U_COMPANION = (0.20, 0.30)


def _time_grid():
    return np.linspace(-1.5 * PERIOD_DAYS, 1.5 * PERIOD_DAYS, 3001)


def test_pair_separations_split_one_geometry_into_two_frames():
    from trex.accelerators import _jax_orbits
    from trex.accelerators._jax_models import jnp

    time = _time_grid()
    d, front = _jax_orbits._sky_geometry(
        jnp.asarray(time), PERIOD_DAYS, A_RS, INC, ECC, ARGP, 0.0)
    primary, secondary = _jax_orbits._pair_occultation_separations(d, front, K)

    front = np.asarray(front)
    d = np.asarray(d)
    primary, secondary = np.asarray(primary), np.asarray(secondary)
    assert front.any() and (~front).any(), "fixture must cross both sides"
    np.testing.assert_allclose(primary[front], d[front], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(secondary[~front], d[~front] / K, rtol=0.0, atol=1e-13)
    assert np.isinf(primary[~front]).all()
    assert np.isinf(secondary[front]).all()
    assert np.isfinite(d).all()


def test_explicit_companion_frame_orbit_matches_the_pair_separation():
    from trex.accelerators import _jax_orbits
    from trex.accelerators._jax_models import core_limb_dark, jnp
    from trex.evidence._jax_unit_integrand import _quadratic_disc_flux

    time = _time_grid()
    d, front = _jax_orbits._sky_geometry(
        jnp.asarray(time), PERIOD_DAYS, A_RS, INC, ECC, ARGP, 0.0)
    _primary, secondary = _jax_orbits._pair_occultation_separations(d, front, K)
    shared = np.asarray(1.0 + core_limb_dark(
        jnp.asarray(U_COMPANION), secondary, 1.0 / K))

    phase = float(_jax_orbits._jax_secondary_eclipse_phase(
        jnp.asarray(ECC), jnp.asarray(ARGP)))
    explicit, valid = _quadratic_disc_flux(
        jnp.asarray(time), PERIOD_DAYS, 1.0 / K, A_RS / K, ECC, ARGP + 180.0, INC,
        jnp.asarray(U_COMPANION), phase * PERIOD_DAYS)
    assert bool(jnp.all(valid))
    # Two Kepler solves of the same orbit; near a contact boundary the disc
    # flux derivative is singular, so the separation-level rounding is allowed
    # to grow to 1e-10 in flux. A phase or frame mistake is O(0.01-1).
    np.testing.assert_allclose(shared, np.asarray(explicit), rtol=0.0, atol=1e-10)


def test_primary_and_secondary_use_their_own_limb_darkening_coefficients():
    from trex.accelerators._jax_models import core_limb_dark, jnp
    from trex.evidence._jax_unit_integrand import _disc_flux

    # Conjunction separations inside contact, with the front/back flags chosen
    # so each member is the occultor in its own frame. The secondary separation
    # 5.5 companion radii is inside the host's 5-radius disc but not a complete
    # occultation, where every limb-darkening law would give the same flux.
    primary_separation = 0.8
    secondary_separation = 5.5
    primary = _disc_flux(primary_separation, K, jnp.asarray(U_HOST), True)
    secondary = _disc_flux(secondary_separation, 1.0 / K, jnp.asarray(U_COMPANION), True)
    np.testing.assert_allclose(
        float(primary), float(1.0 + core_limb_dark(jnp.asarray(U_HOST), primary_separation, K)),
        rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        float(secondary), float(1.0 + core_limb_dark(
            jnp.asarray(U_COMPANION), secondary_separation, 1.0 / K)),
        rtol=0.0, atol=1e-15)
    # Swapping the coefficient sets changes each partial-overlap deficit, so the
    # per-member assignment above is observable rather than assumed.
    swapped_primary = _disc_flux(primary_separation, K, jnp.asarray(U_COMPANION), True)
    swapped_secondary = _disc_flux(secondary_separation, 1.0 / K, jnp.asarray(U_HOST), True)
    assert abs(float(primary) - float(swapped_primary)) > 1e-6
    assert abs(float(secondary) - float(swapped_secondary)) > 1e-6

    # The back-side member is exactly unocculted regardless of its radius.
    assert float(_disc_flux(0.0, 150.0, jnp.asarray(U_HOST), False)) == 1.0
    assert float(_disc_flux(primary_separation, K, jnp.asarray(U_HOST), False)) == 1.0
    assert float(_disc_flux(1e-6, 1e6, jnp.asarray(U_COMPANION), False)) == 1.0


def test_simulate_eb_matches_explicit_primary_plus_secondary():
    from trex.accelerators import _jax_orbits
    from trex.accelerators._jax_models import core_limb_dark, jnp
    from trex.foundation.constants import Rsun
    from trex.models.flux_calibration import binary_flux_weights

    time = _time_grid()
    exposure_days = 0.002
    ratio = 0.35
    companion_fraction = 0.1
    companion_is_host = False
    model = _jax_orbits._simulate_eb_jax(
        jnp.asarray(time), K, 0.2, PERIOD_DAYS, INC, A_RS * Rsun, 1.0,
        U_HOST[0], U_HOST[1], exposure_days, ECC, ARGP, companion_fraction,
        companion_is_host, 1, u1_EB=U_COMPANION[0], u2_EB=U_COMPANION[1],
        secondary_to_host_flux_ratio=ratio)

    d, front = _jax_orbits._sky_geometry(
        jnp.asarray(time), PERIOD_DAYS, A_RS, INC, ECC, ARGP, 0.0)
    primary = 1.0 + core_limb_dark(
        jnp.asarray(U_HOST), jnp.where(front, d, jnp.inf), K)
    secondary = 1.0 + core_limb_dark(
        jnp.asarray(U_COMPANION), jnp.where(front, jnp.inf, d / K), 1.0 / K)
    weights = binary_flux_weights(ratio, companion_fraction, companion_is_host, array_module=jnp)
    expected = weights[0] * primary + weights[1] * secondary + weights[2]
    np.testing.assert_allclose(np.asarray(model), np.asarray(expected), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("compiled", [False, True])
def test_nonpositive_radius_ratio_is_a_numerical_failure_not_a_flat_curve(compiled):
    from trex.accelerators import _jax_orbits
    from trex.accelerators._jax_models import jnp
    from trex.foundation.constants import Rsun

    time = np.linspace(-1.0, 1.0, 21)
    def evaluate(radius_ratio):
        return _jax_orbits._simulate_eb_jax(
            jnp.asarray(time), radius_ratio, 0.2, PERIOD_DAYS, INC, A_RS * Rsun, 1.0,
            0.3, 0.1, 0.002, ECC, ARGP, 0.0, False, 1,
            u1_EB=0.2, u2_EB=0.1, secondary_to_host_flux_ratio=0.3)

    operation = _jax_models.jax.jit(evaluate) if compiled else evaluate
    for radius_ratio in (0.0, -0.1, np.nan, np.inf):
        assert bool(jnp.isnan(operation(radius_ratio)).all())
    assert bool(jnp.isfinite(operation(0.2)).all())


@pytest.mark.parametrize("ecc", [0.0, 1e-9, 0.1, 0.3, 0.7, 0.93])
@pytest.mark.parametrize("argp_deg", [0.0, 23.5, 90.0, 180.0, 250.0, 359.0])
def test_device_secondary_phase_matches_cpu_orbital_mechanics(ecc, argp_deg):
    from trex.accelerators import _jax_orbits
    from trex.models import orbital_mechanics

    device = float(_jax_orbits._jax_secondary_eclipse_phase(
        _jax_models.jnp.asarray(ecc), _jax_models.jnp.asarray(argp_deg)))
    cpu = orbital_mechanics.secondary_eclipse_phase(ecc, argp_deg)
    assert math.isfinite(device) and 0.0 <= device < 1.0
    np.testing.assert_allclose(device, cpu, rtol=0.0, atol=1e-13)
