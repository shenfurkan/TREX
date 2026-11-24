"""Pointwise physical TP/EB likelihoods, not evidence runs or calibration."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from trex.models import likelihoods
from trex.foundation.constants import Rsun
from trex.photometry.baseline import GaussianBaseline


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
def test_physical_primary_secondary_and_repeated_events_use_aperture_density(binary, mode):
    time = np.linspace(-.2, 6.2, 321)
    common = dict(P_orb=3., inc_deg=90., a_cm=10*Rsun, R_s_solar=1.,
                  u1=.2, u2=.3, exptime_days=.002, ecc=.2, argp_deg=40., nsamples=20)
    extra = dict(R_EB_solar=.4, EB_fluxratio=.1, u1_EB=.3, u2_EB=.1) if binary else dict(R_p_earth=8.)
    if binary:
        model, secondary_phase = likelihoods.simulate_EB(time, **extra, **common)
        assert abs(secondary_phase-.5) > .05
        assert model[np.argmin(abs(time-secondary_phase*3))] < .99
    else:
        model = likelihoods.simulate_TP(time, **extra, **common)
    assert model[np.argmin(abs(time))] < .995
    assert model[np.argmin(abs(time-3))] < .995
    fraction = .13
    aperture = 1+fraction*(model-1)
    sigma = np.full(time.size, .0007)
    flux = aperture+.0001*np.sin(time*20)
    design = np.column_stack((np.ones(time.size), time/6))
    mu, covariance = np.array([.0003, -.0002]), np.diag([1e-6, 4e-7])
    prior = GaussianBaseline(time, design, mu, covariance, mode, "synthetic external prior")
    density = prior.prepare(time, flux, sigma).for_host(fraction)
    host_flux, host_sigma = (flux-(1-fraction))/fraction, sigma/fraction
    call = likelihoods.lnL_EB if binary else likelihoods.lnL_TP
    actual = call(time, host_flux, host_sigma, **extra, **common, baseline_likelihood=density)
    basis = design if mode == "additive" else aperture[:, None]*design
    expected = multivariate_normal.logpdf(flux, mean=aperture+basis@mu,
        cov=np.diag(sigma**2)+basis@covariance@basis.T)
    assert actual == pytest.approx(expected, abs=2e-9)
    # The unselected historical branch still returns its original unnormalized white residual [nat].
    assert call(time, host_flux, host_sigma, **extra, **common) == pytest.approx(
        -.5*np.sum(((host_flux-model)/host_sigma)**2), abs=2e-11)


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
def test_zero_baseline_shifts_real_mc_likelihood_and_pilot_once(monkeypatch, mode, binary):
    from trex.evidence import marginal_likelihoods as ml
    from trex.evidence import mc_importance, mc_scenario

    # A zero design has exactly the white aperture Gaussian density, regardless
    # of the proper coefficient prior or host fraction.
    time = np.linspace(-.1, .1, 17)
    flux = 1 + .002*np.sin(40*time)
    sigma = np.full(time.size, .02)
    fraction = .23
    host_flux, host_sigma = (flux - (1-fraction))/fraction, sigma/fraction
    baseline = GaussianBaseline(time, np.zeros((time.size, 1)), np.array([.3]),
                                np.array([[.5]]), mode, "synthetic zero design")
    density = baseline.prepare(time, flux, sigma).for_host(fraction)
    monkeypatch.setattr(mc_scenario, "HAS_JAX_TREX", False)

    def calculate(selected):
        ess, centres = [], []
        value = ml._eval_scenario(
            time, host_flux, host_sigma, 3., 1., 1., .4, .2, 64,
            not binary, binary, False, 120/86400,
            rng=np.random.default_rng(19), nsamples=1, target_feh=0.,
            ess_out=ess, importance_centres_out=centres, baseline_likelihood=selected,
        )
        return value, ess[0], centres[0]

    token = mc_importance.pilot_seed.set(27)
    try:
        white, white_ess, white_centres = calculate(None)
        normalized, normalized_ess, normalized_centres = calculate(density)
    finally:
        mc_importance.pilot_seed.reset(token)
    offset = -.5*np.sum(np.log(2*np.pi*sigma**2))
    assert np.isfinite(white)
    assert normalized - white == pytest.approx(offset, abs=2e-10)
    assert normalized_ess == pytest.approx(white_ess, abs=1e-10)
    np.testing.assert_array_equal(normalized_centres, white_centres)
