"""Regressions for the EB/twin support, twin-period and planet-radius prior routing.

Giacalone et al. 2021, AJ 161 24, DOI 10.3847/1538-3881/abc6af:
§2.4.4 disjoint EB/twin support; §2.4.2 common planet-radius law. The official
1.0.20 path has no 100-day twin cutoff. Synthetic controls isolate these rules;
they do not tune or calibrate any confirmed planet's FPP.
"""

import inspect

import numpy as np
import pytest

from trex.evidence import marginal_likelihoods as ml
from trex.evidence import mc_occultors, mc_scenario
from trex.foundation.constants import G, Msun, Rsun, SECONDS_PER_DAY
from trex.stellar.stellar_relations import stellar_relations


@pytest.mark.parametrize("twice", [False, True])
@pytest.mark.parametrize("period", [3.0, 60.0])
def test_mc_eb_slots_partition_q_without_an_extra_period_cap(monkeypatch, twice, period):
    """Synthetic uniform q tests branch weights and geometry, not the astrophysical q law."""
    class UniformMassRatio:
        # q ~ Uniform[1/2, 1]: Pr(q < 0.95) = 9/10, Pr(q >= 0.95) = 1/10.
        def __call__(self, quantiles):
            return 0.5 + 0.5 * quantiles

        def cdf(self, q):
            return np.clip((q - 0.5) / 0.5, 0.0, 1.0)

    prior = UniformMassRatio()
    sampled_q = []
    evaluated = []
    # Zero-contribution draws must stay in N even though they never reach the likelihood.
    log_prior = np.array([0.0, -np.inf, 0.0, -np.inf])

    def sample_q(quantiles, host):
        assert np.all(host == 1.0)
        q = prior(quantiles)
        sampled_q.extend(q)
        return q

    monkeypatch.setattr(mc_scenario, "HAS_JAX_TREX", False)
    monkeypatch.setattr(mc_scenario, "sample_ecc", lambda x, **kwargs: np.zeros_like(x))
    monkeypatch.setattr(mc_scenario, "sample_w", lambda x: np.full_like(x, 90.0))
    monkeypatch.setattr(mc_scenario, "sample_inc", lambda x: np.full_like(x, 90.0))
    monkeypatch.setattr(mc_occultors, "host_mass_ratio_inverse_cdf", lambda host: prior)
    monkeypatch.setattr(mc_occultors, "sample_q_host_mass_ratio", sample_q)

    def likelihood(time, flux, sigma, radius, *args, **kwargs):
        evaluated.append(radius)
        return 0.0

    monkeypatch.setattr(mc_scenario, "lnL_EB", likelihood)
    fits = []
    value = ml._eval_scenario(
        time=np.array([-0.01, 0.0, 0.01]), flux=np.ones(3), sigma=0.001,
        period_days=period, M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.4, u2=0.2,
        n_draws=4, is_planet=False, is_EB=True, use_2xP=twice, exptime_days=0.001,
        lnprior_comp=log_prior,
        target_feh=0.0, rng=np.random.default_rng(3), representative_fit_out=fits)
    assert np.isfinite(value)
    q = np.asarray(sampled_q)
    assert q.size == log_prior.size == 4
    assert np.all((q >= 0.95) & (q <= 1.0)) if twice else np.all((q >= 0.5) & (q < 0.95))
    expected_q = q[np.isfinite(log_prior)]
    expected_radius = stellar_relations(expected_q)[0]
    np.testing.assert_array_equal(evaluated, expected_radius)
    # Kepler's third law [CGS→Rsun]; circular isotropic transit probability
    # is (R_host+R_secondary)/a. Conditional draws retain the original branch
    # mass, and both zero-contribution draws remain in the denominator.
    a_rs = np.cbrt(G * Msun * (1 + expected_q) * (period * (2 if twice else 1) * SECONDS_PER_DAY / (2 * np.pi)) ** 2) / Rsun
    branch_mass = 1 / 10 if twice else 9 / 10
    expected_evidence = branch_mass * np.sum((1 + expected_radius) / a_rs) / q.size
    assert value == pytest.approx(np.log(expected_evidence), abs=1e-12)
    assert fits[0]["n_draws"] == 4
    assert fits[0]["finite_likelihood_draws"] == 2


def test_every_planet_family_routes_the_same_default_radius_prior(monkeypatch):
    assert all(not row[5] for row in ml.TARGET_SCENARIO_SPECS if row[0])
    planet_calls = []
    signature = inspect.signature(mc_scenario._eval_scenario)

    def evaluate(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if arguments["is_planet"]:
            planet_calls.append(arguments)
        return -1.0

    monkeypatch.setattr(mc_scenario, "_eval_scenario", evaluate)
    ml.calc_target_evidences(
        time=np.array([-0.01, 0.0, 0.01]), flux=np.array([1.0, 0.999, 1.0]), sigma=0.001,
        period_days=3.0, depth_ppm=1000, M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.4, u2=0.2,
        exptime_days=0.001, n_draws=4, target_feh=0.0, target_aperture_flux_fraction=0.5,
        nearby_stars=[{"source_id": "synthetic", "M_s": 1.0, "R_s": 1.0, "aperture_flux_fraction": 0.5}])
    assert len(planet_calls) >= 2  # TP and capable resolved NTP, at minimum
    assert all(not call["use_flat_priors"] for call in planet_calls)
