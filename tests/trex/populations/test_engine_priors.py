"""TREX population prior samplers."""

from __future__ import annotations

import numpy as np
import pytest

from trex.populations.priors import MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR, sample_rp, sample_inc, sample_ecc, sample_q


# ============================================================================
# Priors
# ============================================================================

def test_sample_rp_range():
    rng = np.random.default_rng(42)
    rp = sample_rp(rng.random(2000), np.ones(2000))
    assert np.all(rp >= 0.5)
    assert np.all(rp <= 20.0)
    assert np.median(rp) < 5.0  # Small planets favored


def test_sample_inc_isotropic():
    rng = np.random.default_rng(42)
    inc = sample_inc(rng.random(2000))
    assert 55.0 < np.median(inc) < 65.0


def test_sample_ecc_planet():
    rng = np.random.default_rng(42)
    ecc = sample_ecc(rng.random(500), planet=True, P_orb=5.0)
    assert np.all((ecc >= 0) & (ecc < 1))
    assert 0.1 < np.median(ecc) < 0.4


@pytest.mark.parametrize("planet, period_days", [(True, 5.0), (False, 5.0), (False, 20.0)])
def test_sample_ecc_is_deterministic_for_supplied_draws(planet, period_days):
    draws = np.array([0.01, 0.2, 0.5, 0.9])

    first = sample_ecc(draws, planet=planet, P_orb=period_days)
    second = sample_ecc(draws, planet=planet, P_orb=period_days)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(("draws", "message"), [
    (np.array([-0.1]), r"must lie in \[0, 1\]"),
    (np.array([1.0]), r"must be finite values in \[0, 1\)"),
    (np.array([np.nan]), "finite real numbers"),
])
def test_sample_ecc_rejects_invalid_draws(draws, message):
    with pytest.raises(ValueError, match=message):
        sample_ecc(draws, planet=True, P_orb=5.0)


def test_sample_q_solar():
    rng = np.random.default_rng(42)
    q = sample_q(rng.random(500), M_s=1.0)
    assert np.all((q >= 0.1) & (q <= 1.0))


def test_sample_q_rejects_primary_mass_without_valid_moe_domain():
    with pytest.raises(ValueError, match="no valid q support"):
        sample_q(
            np.array([0.5]),
            M_s=MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR - 1e-6,
        )
