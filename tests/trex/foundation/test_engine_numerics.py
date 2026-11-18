"""TREX foundation constants and log-space numerical primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trex.foundation.constants import Msun, Rsun, Rearth, G
from trex.foundation._numerics import _log_mean_exp, _log_weight_ess, _normalize_probabilities


# ============================================================================
# Constants
# ============================================================================

def test_physical_constants_are_finite():
    assert math.isfinite(Msun) and Msun > 0
    assert math.isfinite(Rsun) and Rsun > 0
    assert math.isfinite(Rearth) and Rearth > 0
    assert math.isfinite(G) and G > 0
    from trex.foundation.constants import SOLAR_MEAN_DENSITY_G_CM3
    assert 1.0 < SOLAR_MEAN_DENSITY_G_CM3 < 2.0


# ============================================================================
# Numerics
# ============================================================================

def test_log_mean_exp_all_finite():
    lnw = np.full(10, -5.0)
    result = _log_mean_exp(lnw, N_total=10)
    assert result == pytest.approx(-5.0, rel=1e-12)


def test_log_mean_exp_with_neginf():
    lnw = np.array([-1.0, -2.0, -np.inf, -3.0])
    expected = math.log((math.exp(-1) + math.exp(-2) + math.exp(-3)) / 4.0)
    result = _log_mean_exp(lnw, N_total=4)
    assert result == pytest.approx(expected, rel=1e-12)


def test_log_mean_exp_all_neginf():
    result = _log_mean_exp(np.full(5, -np.inf), N_total=5)
    assert result == -np.inf


def test_log_mean_exp_posinf():
    result = _log_mean_exp(np.array([-1.0, np.inf, -2.0]), N_total=3)
    assert result == np.inf


def test_log_mean_exp_size_mismatch():
    with pytest.raises(ValueError, match="N_total"):
        _log_mean_exp(np.array([1.0, 2.0]), N_total=5)


def test_normalize_probabilities_ok():
    lnZ = np.array([0.0, -2.0, -10.0])
    probs, status = _normalize_probabilities(lnZ)
    assert status == "ok"
    assert np.sum(probs) == pytest.approx(1.0)


@pytest.mark.parametrize("offset", [0.0, 1e300, -1e300])
def test_equal_evidences_normalize_even_with_a_large_common_offset(offset):
    probs, status = _normalize_probabilities(np.full(18, offset))
    assert status == "ok"
    np.testing.assert_allclose(probs, np.full(18, 1 / 18), rtol=0, atol=2e-16)
    assert probs.sum() == pytest.approx(1.0, rel=0, abs=4e-16)


@pytest.mark.parametrize("offset", [0.0, 2.0**50, -(2.0**50)])
def test_normalization_preserves_representable_evidence_gaps(offset):
    # Integer gaps remain exact at these offsets; any drift is normalization error.
    probs, status = _normalize_probabilities(np.array([0.0, -2.0, -4.0, -np.inf]) + offset)
    denominator = 1 + math.exp(-2) + math.exp(-4)
    expected = np.array([1, math.exp(-2), math.exp(-4), 0]) / denominator
    assert status == "ok"
    np.testing.assert_allclose(probs, expected, rtol=0, atol=2e-16)


def test_normalization_accepts_extreme_finite_evidence_gaps():
    limit = np.finfo(np.float64).max
    with np.errstate(over="raise"):
        probs, status = _normalize_probabilities(np.array([limit, -limit, -np.inf]))
    assert status == "ok"
    np.testing.assert_array_equal(probs, [1, 0, 0])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_normalize_probabilities_anomaly(invalid):
    probs, status = _normalize_probabilities(np.array([-1.0, invalid]))
    assert status == "anomaly"
    np.testing.assert_array_equal(probs, [0, 0])


def test_normalize_probabilities_all_neginf():
    probs, status = _normalize_probabilities(np.full(3, -np.inf))
    assert status == "all_neginf"
    np.testing.assert_array_equal(probs, [0, 0, 0])


@pytest.mark.parametrize("offset", [0.0, 1e300, -1e300])
def test_equal_weight_ess_counts_samples_regardless_of_common_log_scale(offset):
    assert _log_weight_ess(np.r_[np.full(64, offset), -np.inf]) == pytest.approx(64, rel=0, abs=2e-14)


@pytest.mark.parametrize("offset", [0.0, 2.0**50, -(2.0**50)])
def test_weight_ess_preserves_representable_gaps(offset):
    expected = (1 + math.exp(-2) + math.exp(-4)) ** 2 / (1 + math.exp(-4) + math.exp(-8))
    assert _log_weight_ess(np.array([0., -2., -4., -np.inf]) + offset) == pytest.approx(
        expected, rel=0, abs=1e-15)


def test_weight_ess_retains_extreme_and_undefined_weight_contracts():
    limit = np.finfo(np.float64).max
    with np.errstate(over="raise"):
        assert _log_weight_ess([limit, -limit, -np.inf]) == 1.0
    assert _log_weight_ess([-np.inf, -np.inf]) == 0.0
    assert math.isnan(_log_weight_ess([0., np.inf]))
    with pytest.raises(ValueError, match="nonfinite log weight"):
        _log_weight_ess([0., np.nan])
