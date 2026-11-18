"""Temporal response controls: analytic moments, gaps and actual TP/EB consumers."""

import numpy as np
import pytest

from trex.foundation.constants import Rsun
from trex.models.likelihoods import (
    _batman_transit, lnL_EB, lnL_TP, simulate_EB, simulate_TP,
)
from trex.photometry.exposure import ExposureResponse


def test_boxcar_delta_limit_and_independent_quadratic_moment():
    delta = ExposureResponse.boxcar([0., 0.], nsamples=1, source="instantaneous limit")
    np.testing.assert_array_equal(delta.evaluation_times([1., 2.]), [[1.], [2.]])
    rule = ExposureResponse.boxcar([.125, .25], nsamples=128, source="declared boxcars")
    time = np.array([1., 2.])
    actual = np.sum(rule.evaluation_times(time)**2 * rule.weights, axis=1)
    # Composite midpoint quadrature has a known quadratic error h^2/(12 n^2).
    expected = time**2 + np.array([.125, .25])**2 / 12 * (1 - 1 / 128**2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def test_stack_weights_live_time_not_elapsed_span_or_window_count():
    rule = ExposureResponse.stacked([-.5, .25], [-.25, .75], nsamples=64,
                                    source="two unequal live windows")
    offsets = rule.evaluation_times([0.])[0]
    assert not np.any((offsets > -.25) & (offsets < .25))
    # Integral t dt over the windows, divided by their total live duration.
    expected = ((.25**2 - .5**2) + (.75**2 - .25**2)) / (2 * .75)
    assert np.dot(offsets, rule.weights) == pytest.approx(expected, abs=2e-16)
    np.testing.assert_allclose(rule.weights[:64].sum(), 1 / 3, rtol=0, atol=2e-16)
    np.testing.assert_allclose(rule.weights[64:].sum(), 2 / 3, rtol=0, atol=2e-16)
    per_row = ExposureResponse.stacked([[-.5, .25], [0., 1.]], [[-.25, .75], [.5, 1.5]],
                                       nsamples=4, source="row-specific windows")
    assert per_row.evaluation_times([0., 3.]).shape == (2, 8)
    np.testing.assert_allclose(per_row.weights.sum(axis=1), 1., rtol=0, atol=2e-16)


@pytest.mark.parametrize("offsets, weights", [
    ([], []), ([[0.]], [1.]), ([0.], [-1.]), ([0., 1.], [.3, .3]),
    ([np.inf], [1.]), ([0.], [np.nan]), ([False], [1.]), ([0.], [True]),
    (np.ma.array([0.], mask=True), [1.]), ([[[0.]]], [[[1.]]]),
])
def test_bad_quadrature_is_not_normalized_or_silently_masked(offsets, weights):
    with pytest.raises(ValueError):
        ExposureResponse(offsets, weights, "declared response")


@pytest.mark.parametrize("width, samples", [(-.1, 2), ([], 2), ([[.1]], 2), (.1, 0),
                                          (.1, True), (.1, 1.5), (np.nan, 2)])
def test_bad_boxcar_refuses(width, samples):
    with pytest.raises(ValueError):
        ExposureResponse.boxcar(width, nsamples=samples, source="bad")


@pytest.mark.parametrize("start, end", [([0.], [0.]), ([1.], [0.]), ([0., .1], [.2, .3]),
                                      ([0., 1.], [.5]), ([np.nan], [1.])])
def test_bad_or_overlapping_windows_refuse(start, end):
    with pytest.raises(ValueError):
        ExposureResponse.stacked(start, end, nsamples=2, source="bad")


def test_response_is_owned_and_checked_against_actual_consumer_shape():
    offsets, weights = np.array([-.1, .1]), np.array([.5, .5])
    rule = ExposureResponse(offsets, weights, "declared response")
    offsets[:] = 0
    weights[:] = 0
    np.testing.assert_array_equal(rule.offset_days, [-.1, .1])
    assert not rule.offset_days.flags.writeable and not rule.weights.flags.writeable
    per_row = ExposureResponse.boxcar([.1, .2], nsamples=2, source="two rows")
    with pytest.raises(ValueError, match="match"):
        per_row.evaluation_times([0.])
    with pytest.raises(ValueError, match="source"):
        ExposureResponse([0.], [1.], "")
    with pytest.raises(ValueError, match="finite"):
        ExposureResponse([np.finfo(float).max], [1.], "large").evaluation_times([np.finfo(float).max])


def _tp(time, *, response=None, exposure=.01, samples=16):
    return simulate_TP(time, 8., 3., 89., 8 * Rsun, 1., .3, .2, exposure,
                       nsamples=samples, exposure_response=response)


def _eb(time, *, response=None, exposure=.01, samples=16):
    return simulate_EB(time, .3, .1, 3., 89., 8 * Rsun, 1., .3, .2, exposure,
                       u1_EB=.1, u2_EB=.4, nsamples=samples, exposure_response=response)[0]


@pytest.mark.parametrize("forward", [_tp, _eb], ids=["planet", "binary"])
def test_variable_boxcars_match_individual_legacy_forward_models(forward):
    time = np.r_[np.linspace(-.075, .075, 9), np.linspace(1.425, 1.575, 9)]
    widths = np.linspace(.002, .025, time.size)
    rule = ExposureResponse.boxcar(widths, nsamples=16, source="variable boxcars")
    expected = np.array([forward(np.array([t]), exposure=w)[0] for t, w in zip(time, widths)])
    actual = forward(time, response=rule, exposure=None)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-16)
    assert np.any(actual < .999), "the control must actually cover an eclipse"


@pytest.mark.parametrize("forward", [_tp, _eb], ids=["planet", "binary"])
def test_disjoint_windows_use_actual_orbit_for_primary_and_secondary(forward):
    time = np.r_[np.linspace(-.08, .08, 11), np.linspace(1.42, 1.58, 11)]
    rule = ExposureResponse.stacked([-.03, .01], [-.01, .02], nsamples=16,
                                    source="unequal disjoint integrations")
    sampled = forward(rule.evaluation_times(time).reshape(-1), exposure=1e-12, samples=1)
    expected = sampled.reshape(time.size, -1) @ rule.weights
    actual = forward(time, response=rule, exposure=None)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-16)
    filled_gap = forward(time + .005, exposure=.05, samples=48)
    assert np.max(np.abs(actual - filled_gap)) > 1e-5


def test_ground_boxcar_convergence_against_independent_dense_time_integral():
    time = np.array([-.065, -.055, -.045, 0., .045, .055, .065])
    width = .025
    dense_offsets = np.linspace(-width / 2, width / 2, 8193)
    instantaneous = _tp((time[:, None] + dense_offsets).reshape(-1), exposure=1e-12, samples=1)
    expected = np.trapezoid(instantaneous.reshape(time.size, -1), dense_offsets, axis=1) / width
    coarse = _tp(time, response=ExposureResponse.boxcar(width, nsamples=8, source="coarse"),
                 exposure=None)
    fine = _tp(time, response=ExposureResponse.boxcar(width, nsamples=256, source="fine"),
               exposure=None)
    # Sub-ppm flux error is the requirement for this specified ground-box case,
    # not a universal sample count for all radii, durations or contact geometry.
    assert np.max(np.abs(fine - expected)) < 1e-6
    assert np.max(np.abs(fine - expected)) < np.max(np.abs(coarse - expected))


@pytest.mark.parametrize("kind", ["tp", "eb"])
def test_temporal_response_reaches_real_density_without_changing_errors(kind):
    time = np.linspace(-.08, .08, 13)
    rule = ExposureResponse.boxcar(np.linspace(.002, .02, time.size), nsamples=16,
                                  source="heterogeneous ground integrations")
    model = (_tp if kind == "tp" else _eb)(time, response=rule, exposure=None)
    sigma = np.linspace(.001, .002, time.size)
    residual = np.linspace(-.001, .001, time.size)
    flux = model + residual
    expected = -.5 * np.sum(((flux - model) / sigma)**2)
    common = dict(P_orb=3., inc_deg=89., a_cm=8 * Rsun, R_s_solar=1., u1=.3, u2=.2,
                  exptime_days=None, exposure_response=rule, nsamples=16)
    actual = (lnL_TP(time, flux, sigma, R_p_earth=8., **common) if kind == "tp" else
              lnL_EB(time, flux, sigma, R_EB_solar=.3, EB_fluxratio=.1,
                     u1_EB=.1, u2_EB=.4, **common))
    assert actual == pytest.approx(expected, abs=2e-14)


def test_explicit_response_refuses_double_exposure_or_bin_integration():
    rule = ExposureResponse.boxcar(.01, nsamples=8, source="declared box")
    options = dict(time=[0.], period_days=3., rp_rs=.1, a_rs=8., inc_deg=89., u1=.3, u2=.2)
    with pytest.raises(ValueError, match="exptime_days=None"):
        _batman_transit(**options, exptime_days=.01, exposure_response=rule)
    with pytest.raises(ValueError, match="phase-bin"):
        _batman_transit(**options, exptime_days=None, bin_integration_days=.02, exposure_response=rule)
    with pytest.raises(ValueError, match="exptime_days=None"):
        _batman_transit(**options, exptime_days=None, exposure_response="ramp")
