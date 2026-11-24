"""Partial chi-square screening must leave the Monte Carlo evidence unchanged."""
import math

import numpy as np
import pytest

from trex.foundation.constants import SECONDS_PER_DAY
from tests.fixtures.synthetic_observations import SYNTHETIC_SOLAR_METALLICITY_DEX


def test_screen_skips_only_draws_that_cannot_reach_the_evidence(monkeypatch):
    from trex.evidence import mc_scenario

    monkeypatch.setattr(mc_scenario, "_SCREEN_STEP_DRAWS", 8)
    rng = np.random.default_rng(3)
    n = 400
    weight = rng.normal(0.0, 3.0, n)
    # A wide bound spread, a true lnL below it, and one bound that says nothing.
    bound = -rng.exponential(60.0, n)
    true = bound-rng.exponential(5.0, n)
    bound[17], true[17] = np.nan, 2.0
    seen = []

    def evaluate(draws):
        seen.extend(draws.tolist())
        return true[draws]

    lnL = mc_scenario._screened_lnL(bound, weight, evaluate)
    skipped = np.setdiff1d(np.arange(n), seen)
    assert skipped.size > n/2 and 17 in seen and len(seen) == len(set(seen))
    np.testing.assert_array_equal(lnL[seen], true[seen])
    assert np.all(np.isneginf(lnL[skipped]))
    best = np.max(true[seen]+weight[seen])
    assert np.all(bound[skipped]+weight[skipped] < best-mc_scenario._SCREEN_CUT_NATS)
    # The stated error bound: what the skipped draws could contribute, against the evidence.
    assert np.exp(np.max(true[skipped]+weight[skipped])-best)*n < n*math.exp(-mc_scenario._SCREEN_CUT_NATS)


@pytest.mark.parametrize("scenario", ("TP", "EB"))
def test_screened_evidence_matches_the_full_grid(monkeypatch, scenario):
    from trex.evidence import marginal_likelihoods as ml
    from trex.evidence import mc_scenario

    if not ml.HAS_JAX_TREX:
        pytest.skip("JAX is unavailable")
    is_eb = scenario == "EB"
    rng = np.random.default_rng(11)
    time = np.linspace(-0.2, 0.2, 120)
    flux = 1.0+rng.normal(0.0, 3e-4, time.size)
    flux[50:70] -= 0.05 if is_eb else 0.005
    kwargs = dict(
        n_draws=3000, is_planet=not is_eb, is_EB=is_eb, use_2xP=False,
        exptime_days=120.0/SECONDS_PER_DAY, nsamples=4,
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )
    original = mc_scenario._screened_lnL
    evaluated = []

    def counted(bound, log_weight, evaluate):
        def tally(draws):
            evaluated.append(draws.size)
            return evaluate(draws)
        return original(bound, log_weight, tally)

    monkeypatch.setattr(mc_scenario, "_screened_lnL", counted)
    monkeypatch.setattr(mc_scenario, "_SCREEN_STEP_DRAWS", 64)
    screened_ess: list = []
    screened = ml._eval_scenario(time, flux, 3e-4, 4.0, 1.0, 1.0, 0.3, 0.2, ess_out=screened_ess,
                                 rng=np.random.default_rng(2468), **kwargs)
    screened_draws = sum(evaluated)
    monkeypatch.setattr(mc_scenario, "_SCREEN_CUT_NATS", math.inf)
    full_ess: list = []
    full = ml._eval_scenario(time, flux, 3e-4, 4.0, 1.0, 1.0, 0.3, 0.2, ess_out=full_ess,
                             rng=np.random.default_rng(2468), **kwargs)
    assert sum(evaluated) > 2*screened_draws
    assert screened == pytest.approx(full, abs=1e-6)
    assert screened_ess[0] == pytest.approx(full_ess[0], rel=1e-6)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_partial_likelihood_cannot_be_screened_as_zero(nonfinite, monkeypatch):
    from trex.evidence import mc_scenario

    monkeypatch.setattr(mc_scenario, "_SCREEN_STEP_DRAWS", 1)
    full_grid = np.array([nonfinite, -40.0, -41.0, -90.0])
    weights = np.zeros_like(full_grid)
    bounds = np.array([nonfinite, -40.0, -41.0, -90.0])
    actual = mc_scenario._screened_lnL(bounds, weights, full_grid.__getitem__)
    np.testing.assert_array_equal(actual, [nonfinite, -40.0, -41.0, -np.inf])
