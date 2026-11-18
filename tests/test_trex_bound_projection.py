"""Independent circular-orbit sky-projection checks for higher-mass companions."""

import math

from astropy import constants, units
import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import marginal_likelihoods as ml
from trex.evidence import nested_evidence as ne
from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case


def _fixed_orbit():
    period_quantile, mass_quantile = 0.5, 0.5
    log_period = ml.sample_companion_log_period(np.array([period_quantile]), 1.0)[0]
    q = ml.sample_q_companion(np.array([mass_quantile]), 1.0, np.array([log_period]))[0]
    # Independent Kepler law with Astropy units: a^3 = G(M1+M2)(P/2pi)^2.
    axis = (constants.G * (1.0 + q) * constants.M_sun * (10**log_period * units.day / (2 * math.pi))**2)**(1/3)
    return period_quantile, mass_quantile, axis.to_value(units.au) * 0.05 / 2.0


class _FixedVariates:
    def __init__(self, count, period, mass, projection):
        self.values = iter((np.full(count, period), np.full(count, mass), projection))

    def random(self, count):
        result = next(self.values)
        assert len(result) == count
        return result


def test_circular_isotropic_projection_matches_analytic_fraction_in_mc_and_nested():
    count = 2000
    period, mass, limit = _fixed_orbit()
    projection = (np.arange(count) + 0.5) / count
    retained = []
    rng = _FixedVariates(count, period, mass, projection)
    _, prior, _, _, _ = ml._draw_bound_companion(
        rng, count, 1.0, 1.0, 50.0, None, None, resolution_limit_arcsec=limit,
        variates_out=retained,
    )
    # Isotropic cos(theta): P(s < a/2) = P(|cos(theta)| > sqrt(3)/2).
    expected = projection >= math.sqrt(3.0) / 2.0
    np.testing.assert_array_equal(np.isfinite(prior), expected)
    assert np.mean(np.isfinite(prior)) == pytest.approx(1 - math.sqrt(3.0) / 2.0, abs=1 / count)
    np.testing.assert_array_equal(retained[0][2], projection)

    block = ne.bound_companion_population(1.0, 1.0, 50.0, resolution_limit_arcsec=limit)
    assert block.n_dimensions == 3
    for index in (0, 1000, 1731, 1732, 1733, 1999):
        assert block.resolve(np.array([period, mass, projection[index]])).supported == expected[index]
    with pytest.raises(ValueError, match="old-dimensional checkpoint"):
        block.resolve(np.array([period, mass]))


@pytest.mark.skipif(not models.HAS_JAX_TREX, reason="JAX is unavailable")
def test_circular_isotropic_projection_matches_analytic_fraction_in_native_jax():
    period, mass, limit = _fixed_orbit()
    block = ne.bound_companion_population(1.0, 1.0, 50.0, resolution_limit_arcsec=limit)
    reference = ne._TransitingPlanetIntegrand(
        **synthetic_trex_tp_integrand_case(), host_mass=1.0, host_radius=1.0,
        nsamples=3, bin_integration_days=0.0, population=block,
        n_dimensions=4 + block.n_dimensions, hosted=False, target_feh=0.0, use_flat_priors=False,
    )
    native = reference.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    points = np.full((2000, reference.n_dimensions), 0.5)
    points[:, 4], points[:, 5] = period, mass
    points[:, 6] = (np.arange(len(points)) + 0.5) / len(points)
    # The independent circular-orbit CDF predicts support precisely from the last coordinate.
    supports = models.jax.jit(models.jax.vmap(native._resolve))(points)[4]
    np.testing.assert_array_equal(np.asarray(supports), points[:, 6] >= math.sqrt(3.0) / 2.0)
