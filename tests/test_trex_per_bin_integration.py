"""Per-bin phase integration: broad baseline bins integrate the model over their own width."""

import math

import numpy as np
import pytest

from trex.models.likelihoods import _batman_transit, lnL_EB, simulate_EB


EXPOSURE_DAYS = 120.0 / 86400.0
EB = dict(u1=0.4, u2=0.2, u1_EB=0.3, u2_EB=0.3, ecc=0.0, argp_deg=90.0)


def _production_like_bins(period, duration, n_points):
    """The transit-centred non-uniform edges of ``_prepare_observed_transit_input``."""
    transit_bins = max(1, math.ceil(duration / EXPOSURE_DAYS))
    broad = (min(100, n_points) - 6) // 2
    limit = min(3.0 * duration, 0.25 * period)
    return np.concatenate((
        np.linspace(-period / 2, -limit, broad + 1), np.linspace(-limit, -duration / 2, 4)[1:],
        np.linspace(-duration / 2, duration / 2, transit_bins + 1)[1:], np.linspace(duration / 2, limit, 4)[1:],
        np.linspace(limit, period / 2, broad + 1)[1:],
    )), duration / transit_bins


def test_equal_per_bin_widths_reproduce_the_scalar_kernel():
    time = np.linspace(-0.12, 0.12, 241)
    scalar = _batman_transit(time, 3.836, 0.1, 12.0, 89.2, 0.4, 0.2, 0.000231, bin_integration_days=0.0009)
    per_bin = _batman_transit(time, 3.836, 0.1, 12.0, 89.2, 0.4, 0.2, 0.000231,
                              bin_integration_days=np.full(time.size, 0.0009))
    np.testing.assert_allclose(per_bin, scalar, rtol=0, atol=1e-14)


def test_per_bin_widths_score_an_eb_secondary_that_falls_in_broad_bins():
    period, mass = 10.0, 1.3 * 1.989e33
    a_cm = (6.674e-8 * mass * (period * 86400.0) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    args = (0.5, 0.02, period, 87.0, a_cm, 1.0)
    time = np.arange(0.0, 54.8, EXPOSURE_DAYS)
    truth, _ = simulate_EB(time, *args, exptime_days=EXPOSURE_DAYS, bin_integration_days=0.0, **EB)
    phase = np.remainder(time + period / 2, period) - period / 2
    eclipsed = phase[(truth < 1 - 1e-9) & (np.abs(phase) < period / 4)]
    edges, transit_width = _production_like_bins(period, np.ptp(eclipsed) + EXPOSURE_DAYS, time.size)
    index = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, edges.size - 2)
    populated = np.unique(index)
    centres = np.array([phase[index == i].mean() for i in populated])
    flux = np.array([truth[index == i].mean() for i in populated])
    sigma = np.array([1e-3 / math.sqrt(np.count_nonzero(index == i)) for i in populated])
    widths = np.diff(edges)[populated]
    assert 1 - truth[np.abs(np.abs(phase) - period / 2) < 0.01].min() > 1e-3  # a real secondary

    def score(width):
        return lnL_EB(centres, flux, sigma, *args, EB["u1"], EB["u2"], EXPOSURE_DAYS, 0.0, 90.0,
                      bin_integration_days=width, u1_EB=0.3, u2_EB=0.3)

    assert score(widths) > -1.0
    assert score(transit_width) < -10.0


def test_numpy_float32_scalar_width_is_passed_through_unchanged():
    from trex.foundation._numerics import phase_bin_widths

    width = np.float32(0.0009)
    assert phase_bin_widths(width, 5) is width
    assert type(phase_bin_widths(1, 5)) is float


@pytest.mark.parametrize("widths", [np.full(4, -1.0), np.full(3, 0.001), np.array([0.001, np.nan, 0.001, 0.001])])
def test_invalid_per_bin_widths_are_refused(widths):
    with pytest.raises(ValueError, match="bin_integration_days"):
        _batman_transit(np.linspace(-0.1, 0.1, 4), 3.0, 0.1, 10.0, 89.0, 0.4, 0.2, 0.001,
                        bin_integration_days=widths)


def test_jax_batch_matches_batman_with_per_bin_widths():
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_EB

    if not HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed in this environment")
    time = np.concatenate((np.linspace(-1.5, -0.2, 20), np.linspace(-0.05, 0.05, 41), np.linspace(0.2, 1.5, 20)))
    widths = np.concatenate((np.full(20, 0.065), np.full(41, 0.0025), np.full(20, 0.065)))
    rng = np.random.default_rng(5)
    flux = 1.0 - 0.02 * np.exp(-0.5 * (time / 0.02) ** 4) + rng.normal(0, 3e-4, time.size)
    sigma = np.full(time.size, 3e-4)
    r_eb, f_eb, inc = np.array([0.15, 0.3]), np.array([0.01, 0.12]), np.array([89.2, 88.6])
    a_cm, r_host, u1, u2 = np.full(2, 1.4e12), np.full(2, 0.83), np.full(2, 0.4), np.full(2, 0.2)
    accelerated = batch_lnL_EB(
        time, flux, sigma, r_eb, f_eb, 3.0, inc, a_cm, r_host, u1, u2, 0.000231,
        ecc_arr=np.zeros(2), w_arr=np.full(2, 90.0), companion_fluxratio=np.zeros(2), nsamples=20,
        bin_integration_days=widths, u1_EB=u1, u2_EB=u2,
    )
    reference = np.array([
        lnL_EB(time, flux, sigma, float(r_eb[i]), float(f_eb[i]), 3.0, float(inc[i]), 1.4e12, 0.83, 0.4, 0.2,
               0.000231, ecc=0.0, argp_deg=90.0, nsamples=20, bin_integration_days=widths, u1_EB=0.4, u2_EB=0.2)
        for i in range(2)
    ])
    assert np.max(np.abs((accelerated - reference) / reference)) < 1e-5


def test_native_jax_integrand_matches_reference_with_per_bin_widths():
    from trex.accelerators import _jax_models as models
    from trex.evidence.nested_evidence import _TransitingPlanetIntegrand
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed in this environment")
    scene = synthetic_trex_tp_integrand_case()
    time = np.asarray(scene["time"], dtype=float)
    # Wider boxes away from the transit, as the production baseline bins are.
    widths = np.where(np.abs(time) > 0.1, 20.0 * scene["exptime_days"], scene["exptime_days"])
    assert np.unique(widths).size == 2
    reference = _TransitingPlanetIntegrand(
        **scene, host_mass=1.0, host_radius=1.0, bin_integration_days=widths,
        nsamples=3, use_flat_priors=False, population=None, n_dimensions=4,
    )
    native = reference.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    points = np.random.default_rng(20261019).uniform(size=(16, 4))
    parameters = np.asarray([reference.prior_transform(point) for point in points])
    expected = np.asarray([reference.log_likelihood(value) for value in parameters])
    actual = np.asarray(models.jax.jit(models.jax.vmap(native.evaluate))(points).log_integrand)
    finite = np.isfinite(expected)
    assert finite.any()
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-4, atol=1e-10)
