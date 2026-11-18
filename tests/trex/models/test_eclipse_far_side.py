"""An EB secondary's large occultor must not cover the disc from behind.

The secondary eclipse passes ``rp = 1/k``. With k = 1/150 (a 0.1 R_sun star
eclipsed by a 15 R_sun host) contact needs a separation below 151 radii, so
batman's far-side sentinel of 100 radii used to produce a full occultation half
an orbit after the real eclipse. Circular, edge-on orbit with omega = 90 deg:
the occultor is centred on the disc at t = 0 (exact full occultation, flux 0),
at quadrature at P/4 (separation 600 > 151, flux 1) and directly behind the
disc at P/2, where the flux must also be exactly 1.
"""
import numpy as np
import pytest

OCCULTOR_RADIUS = 150.0
AXIS = 600.0
PERIOD_DAYS = 10.0
TIMES = np.array([0.0, 0.25 * PERIOD_DAYS, 0.5 * PERIOD_DAYS])
EXPECTED = np.array([0.0, 1.0, 1.0])


def test_cpu_secondary_occultor_does_not_cover_the_disc_from_behind():
    from trex.models.likelihoods import _batman_transit

    flux = _batman_transit(TIMES, PERIOD_DAYS, OCCULTOR_RADIUS, AXIS, 90.0, 0.4, 0.2, 1e-4,
                           argp_deg=90.0, nsamples=1)
    np.testing.assert_allclose(flux, EXPECTED, rtol=0.0, atol=1e-12)


def test_device_secondary_occultor_does_not_cover_the_disc_from_behind():
    pytest.importorskip("jax")
    from trex.accelerators import _jax_models

    if not _jax_models.HAS_JAX_TREX:
        pytest.skip("JAX transit kernels unavailable")
    flux = np.asarray(_jax_models._jax_transit(TIMES, PERIOD_DAYS, OCCULTOR_RADIUS, AXIS, 90.0, 0.4, 0.2,
                                               1e-4, argp_deg=90.0, nsamples=1))
    np.testing.assert_allclose(flux, EXPECTED, rtol=0.0, atol=1e-12)


# Full EB model, uniform discs (u = 0) so both eclipses have exact values: a
# 0.1 R_sun EB star around a 15 R_sun host (k = 1/150) on a circular, edge-on
# orbit. At t = 0 the EB star crosses the host centre (host flux 1 - k^2) and
# must stay fully visible; at P/2 the host covers it completely.
EB_RADIUS, HOST_RADIUS, EB_FLUXRATIO, AXIS_HOST_RADII = 0.1, 15.0, 0.01, 40.0


def _eb_expected():
    k = EB_RADIUS / HOST_RADIUS
    weight = EB_FLUXRATIO / (1.0 - EB_FLUXRATIO)
    return np.array([(1.0 - k * k + weight) / (1.0 + weight), 1.0, 1.0 / (1.0 + weight)])


def _eb_arguments():
    from trex.foundation.constants import Rsun

    return (TIMES, EB_RADIUS, EB_FLUXRATIO, PERIOD_DAYS, 90.0, AXIS_HOST_RADII * HOST_RADIUS * Rsun,
            HOST_RADIUS, 0.0, 0.0, 1e-4)


def test_cpu_eb_with_small_radius_ratio_has_only_its_two_eclipses():
    from trex.models.likelihoods import simulate_EB

    flux, _ = simulate_EB(*_eb_arguments(), argp_deg=90.0, nsamples=1, u1_EB=0.0, u2_EB=0.0)
    np.testing.assert_allclose(flux, _eb_expected(), rtol=0.0, atol=1e-12)


def test_device_eb_with_small_radius_ratio_has_only_its_two_eclipses():
    pytest.importorskip("jax")
    from trex.accelerators import _jax_models

    if not _jax_models.HAS_JAX_TREX:
        pytest.skip("JAX transit kernels unavailable")
    flux = np.asarray(_jax_models._simulate_eb_jax(*_eb_arguments(), argp_deg=90.0, nsamples=1,
                                                   u1_EB=0.0, u2_EB=0.0))
    np.testing.assert_allclose(flux, _eb_expected(), rtol=0.0, atol=1e-12)
