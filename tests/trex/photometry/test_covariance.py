"""Independent sample-covariance identities, white limits and known correlated noise."""

import json

import numpy as np
import pytest

from trex.photometry.covariance import folded_covariance


def _control(rho=0.):
    # Independent Gaussian four-bin blocks, with analytic C_ab=rho**abs(a-b).
    # Four independent orbital repeats are folded; disjoint blocks stay independent.
    q, windows, repeats = 4, 128, 4
    exposure = 1/32
    duration, period = q*exposure, windows*q*exposure
    covariance = rho**abs(np.subtract.outer(np.arange(q), np.arange(q)))
    noise = np.random.default_rng(261030).multivariate_normal(
        np.zeros(q), covariance, size=(repeats, windows),
    ).reshape(-1)
    time = (np.arange(noise.size)+.5)*exposure-period/2
    sigma = np.full(time.size, .001)
    return dict(time_btjd=time, flux=1+sigma*noise, sigma=sigma, period_days=period,
                epoch_btjd=0., duration_days=duration, exposure_days=exposure,
                transit_bin_count=q, adopted_uncertainty_factor=1.), covariance


@pytest.mark.parametrize("rho", [0., .8])
def test_shifted_fold_covariance_matches_independent_gaussian_block_oracle(rho):
    inputs, expected = _control(rho)
    result = folded_covariance(**inputs)
    assert result["status"] == "measured"
    assert result["expected_window_count"] == 128
    assert result["cadence_reuse_count"] == 0
    assert result["independent_realization_count"] is None
    samples = np.array([row["standardized_residuals"] for row in result["windows"] if row["complete"]])
    # Independent NumPy full-matrix oracle checks the production lag reductions.
    oracle = np.cov(samples, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["standardized_bin_variances"], np.diag(oracle), rtol=1e-14)
    np.testing.assert_allclose(result["adjacent_standardized_covariances"], np.diag(oracle, 1), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(result["standardized_mean_variance_over_envelope"], np.sum(oracle)/4, rtol=1e-14)
    # Gaussian sample covariance has Var(C_ab)=(C_aa*C_bb+C_ab²)/(M-1).
    # Four sampling SEs, not an adopted production significance threshold.
    se = np.sqrt((1+rho**2)/(len(samples)-1))
    np.testing.assert_allclose(np.diag(oracle), 1., atol=4*np.sqrt(2/(len(samples)-1)))
    np.testing.assert_allclose(np.diag(oracle, 1), rho, atol=4*se)
    expected_mean_ratio = np.sum(expected)/4
    assert abs(result["standardized_mean_variance_over_envelope"]/expected_mean_ratio-1) < 4*np.sqrt(2/(len(samples)-1))
    if rho:
        assert result["standardized_mean_variance_over_envelope"] > 2
    json.dumps(result, allow_nan=False)


def test_known_primary_secondary_and_additional_events_never_enter_the_covariance():
    inputs, _ = _control(.8)
    event = dict(period_days=inputs["period_days"], epoch_btjd=inputs["period_days"]/4,
                 duration_days=inputs["duration_days"])
    inputs["known_events"] = [event]
    expected = folded_covariance(**inputs)
    changed = inputs["flux"].copy()
    for epoch in (0., inputs["period_days"]/2, event["epoch_btjd"]):
        phase = (inputs["time_btjd"]-epoch+inputs["period_days"]/2) % inputs["period_days"]-inputs["period_days"]/2
        changed[abs(phase) <= (inputs["duration_days"]+inputs["exposure_days"])/2] -= .3
    assert folded_covariance(**{**inputs, "flux": changed}) == expected
    # Enlarging the already adopted envelope scales variance ratios, not measured C.
    scaled = folded_covariance(**{**inputs, "adopted_uncertainty_factor": 2.})
    assert scaled["adjacent_standardized_covariances"] == expected["adjacent_standardized_covariances"]
    assert scaled["weighted_transit_mean_variance_over_envelope"] == expected["weighted_transit_mean_variance_over_envelope"]/4


def test_missing_phase_coverage_retains_every_window_and_does_not_invent_zero_covariance():
    inputs, _ = _control()
    keep = np.arange(len(inputs["time_btjd"])) % 4 == 0
    for key in ("time_btjd", "flux", "sigma"):
        inputs[key] = inputs[key][keep]
    result = folded_covariance(**inputs)
    assert result["status"] == "deferred"
    assert result["complete_window_count"] == 0
    assert len(result["windows"]) == result["expected_window_count"]
    assert "adjacent_standardized_covariances" not in result
    assert all(None in row["standardized_residuals"] for row in result["windows"])
    json.dumps(result, allow_nan=False)
