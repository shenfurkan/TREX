"""TREX accepts per-point propagated flux uncertainties (M1).

A constant array must reproduce the scalar TRICERATOPS convention exactly, and a
varying array must give the same weighted likelihood in the NumPy model and
the independent JAX batch implementation.
"""

import math

import numpy as np
import pytest

from trex.foundation.constants import SECONDS_PER_DAY, au
from trex.models.flux_calibration import renorm_flux_to_host
from trex.models.flux_calibration import validate_flux_uncertainty
from trex.models.likelihoods import lnL_EB, lnL_TP, simulate_TP

# The test target is per-point sigma, not the LDC value; any valid pair works.
EB_LDC = {"u1_EB": 0.25, "u2_EB": 0.2}


def _light_curve():
    time = np.linspace(-0.1, 0.1, 80)
    flux = np.ones_like(time)
    flux[35:45] = 0.99
    sigma = np.linspace(5e-4, 2e-3, time.size)
    return time, flux, sigma


def test_flux_uncertainty_validation_accepts_scalars_and_matching_arrays():
    flux = np.ones(4)
    assert validate_flux_uncertainty(1e-3, flux) == 1e-3
    np.testing.assert_array_equal(validate_flux_uncertainty(np.full(4, 2e-3), flux), np.full(4, 2e-3))
    for bad in (0.0, math.nan, np.full(3, 1e-3), np.array([1e-3, 1e-3, -1e-3, 1e-3])):
        with pytest.raises(ValueError):
            validate_flux_uncertainty(bad, flux)


def test_renormalization_scales_each_point_uncertainty():
    flux = np.array([1.0, 0.99, 1.0])
    sigma = np.array([1e-3, 2e-3, 3e-3])
    host_flux, host_sigma = renorm_flux_to_host(flux, sigma, 0.25)
    np.testing.assert_allclose(host_sigma, sigma / 0.25)
    np.testing.assert_allclose(host_flux, (flux - 0.75) / 0.25)


def test_numpy_likelihood_with_an_array_is_the_weighted_residual_sum():
    time, flux, sigma = _light_curve()
    exptime = 120.0 / SECONDS_PER_DAY
    arguments = (2.5, 3.5, 89.0, 0.05 * au, 1.0, 0.3, 0.2, exptime)
    model = simulate_TP(time, *arguments, 0.0, 90.0, 0.0, False, 10, 0.0)
    expected = -0.5 * float(np.sum(np.square(flux - model) / np.square(sigma)))

    assert lnL_TP(time, flux, sigma, *arguments, nsamples=10) == pytest.approx(expected, rel=1e-12)
    constant = np.full(time.size, 1e-3)
    assert lnL_TP(time, flux, constant, *arguments, nsamples=10) == lnL_TP(time, flux, 1e-3, *arguments, nsamples=10)


def test_jax_batch_likelihood_matches_numpy_for_per_point_sigma():
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_EB, batch_lnL_TP

    if not HAS_JAX_TREX:
        pytest.skip("JAX not available in current environment")
    time, flux, sigma = _light_curve()
    exptime = 120.0 / SECONDS_PER_DAY
    rp_arr = np.array([2.0, 3.0])
    inc_arr = np.array([88.0, 89.5])
    a_cm_arr = np.array([0.04, 0.06]) * au
    ecc_arr = np.array([0.0, 0.1])
    w_arr = np.array([90.0, 45.0])

    batch_tp = batch_lnL_TP(time, flux, sigma, rp_arr, 3.5, inc_arr, a_cm_arr, 1.0, 0.3, 0.2, exptime, ecc_arr, w_arr, nsamples=10)
    scalar_tp = batch_lnL_TP(time, flux, 1e-3, rp_arr, 3.5, inc_arr, a_cm_arr, 1.0, 0.3, 0.2, exptime, ecc_arr, w_arr, nsamples=10)
    constant_tp = batch_lnL_TP(
        time, flux, np.full(time.size, 1e-3), rp_arr, 3.5, inc_arr, a_cm_arr, 1.0, 0.3, 0.2, exptime, ecc_arr, w_arr, nsamples=10
    )
    np.testing.assert_allclose(constant_tp, scalar_tp, rtol=1e-12)
    for k in range(rp_arr.size):
        single = lnL_TP(
            time, flux, sigma, rp_arr[k], 3.5, inc_arr[k], a_cm_arr[k], 1.0, 0.3, 0.2, exptime,
            ecc=ecc_arr[k], argp_deg=w_arr[k], nsamples=10,
        )
        assert math.isclose(batch_tp[k], single, rel_tol=1e-4, abs_tol=1e-4)

    reb_arr = np.array([0.4, 0.6])
    feb_arr = np.array([0.1, 0.2])
    batch_eb = batch_lnL_EB(time, flux, sigma, reb_arr, feb_arr, 4.0, inc_arr, a_cm_arr, 1.0, 0.3, 0.2, exptime, ecc_arr, w_arr, nsamples=10, **EB_LDC)
    for k in range(reb_arr.size):
        single = lnL_EB(
            time, flux, sigma, reb_arr[k], feb_arr[k], 4.0, inc_arr[k], a_cm_arr[k], 1.0, 0.3, 0.2, exptime,
            ecc=ecc_arr[k], argp_deg=w_arr[k], nsamples=10,
            **EB_LDC,
        )
        assert math.isclose(batch_eb[k], single, rel_tol=1e-4, abs_tol=1e-4)
