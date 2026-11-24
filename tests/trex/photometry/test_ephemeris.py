"""Shared (t0, P) uncertainty propagation against explicit linear algebra."""

import numpy as np
import pytest

from trex.photometry.ephemeris import LinearEphemeris


def _ephemeris(cov=((4e-8, -1e-9), (-1e-9, 9e-10))):
    return LinearEphemeris(1.25, 3.5, np.asarray(cov), "BTJD reference", "synthetic fit")


def test_epoch_covariance_equals_explicit_jacobian_propagation():
    ephemeris = _ephemeris()
    epochs = np.array([-3, 0, 2, 17])
    times, covariance = ephemeris.epochs(epochs)

    np.testing.assert_allclose(times, 1.25 + epochs * 3.5, rtol=0, atol=1e-15)
    var_t0, cov, var_p = 4e-8, -1e-9, 9e-10
    for i, e in enumerate(epochs):
        assert ephemeris.epoch_variance(int(e)) == pytest.approx(var_t0 + e * e * var_p + 2 * e * cov, rel=1e-14)
        for j, f in enumerate(epochs):
            expected = var_t0 + (e + f) * cov + e * f * var_p
            assert covariance[i, j] == pytest.approx(expected, rel=1e-12, abs=1e-24)


def test_shared_ephemeris_couples_epochs_unlike_independent_shifts():
    _, covariance = _ephemeris(cov=((1e-8, 0.0), (0.0, 1e-9))).epochs(np.array([0, 10]))
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    # Shared t0 alone correlates every epoch: 1e-8 / sqrt(1e-8 * (1e-8 + 100e-9)).
    assert correlation == pytest.approx(1e-8 / np.sqrt(1e-8 * 1.1e-7), rel=1e-12)
    assert covariance[0, 1] > 0


def test_unbounded_gaussian_tails_are_not_truncated_at_five_sigma():
    ephemeris = _ephemeris()
    var_t0, cov, var_p = 4e-8, -1e-9, 9e-10
    for epoch in (5, 10**3, 10**6):
        expected = var_t0 + epoch * epoch * var_p + 2 * epoch * cov
        assert ephemeris.epoch_variance(epoch) == pytest.approx(expected, rel=1e-14)
        assert np.isfinite(expected)


def test_invalid_measures_are_refused():
    with pytest.raises(ValueError, match="positive semidefinite"):
        _ephemeris(cov=((1e-8, 1e-6), (1e-6, 1e-8)))
    with pytest.raises(ValueError, match="symmetric"):
        _ephemeris(cov=((1e-8, 1e-9), (2e-9, 1e-8)))
    with pytest.raises(ValueError, match="positive finite period"):
        LinearEphemeris(0.0, -1.0, np.eye(2), "r", "s")
    with pytest.raises(ValueError, match="integer"):
        _ephemeris().epochs(np.array([0.5]))
    with pytest.raises(ValueError, match="integer"):
        _ephemeris().epoch_variance(True)
