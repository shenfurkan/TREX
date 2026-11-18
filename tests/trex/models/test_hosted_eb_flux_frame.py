"""Independent equal-star conservation control at the actual hosted-EB boundary."""

import math

import numpy as np

from trex.evidence.nested_evidence import _EclipsingPairIntegrand, background_star_population
from trex.foundation.constants import G, Msun, Rsun, SECONDS_PER_DAY
from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT
from trex.stellar.stellar_relations import stellar_relations


def _equal_star_background_binary():
    """Analytic two-eclipse control: identical stars behind a constant target.

    TEST_DESIGN: the retained population star has one-half the target's
    apparent flux, and q = 1 draws that same star as the secondary. Total
    light is two target-flux units; every total eclipse removes one-half
    unit, so both normalized eclipse centers are exactly 3/4. The retained
    stellar relation only chooses matching radii/temperatures for this
    synthetic scene, not the expected light curve.
    """
    mass = 1.0
    radius, teff = stellar_relations(np.asarray([mass]))
    apparent_ratio = 0.5
    target_mag = 10.0
    table = {
        "tess_mag": np.asarray(
            [target_mag + math.log10(apparent_ratio) / POGSON_FLUX_EXPONENT]
        ),
        "mass_solar": np.asarray([mass]),
        "radius_solar": radius,
        "teff_k": teff,
        "mh_dex": np.asarray([0.0]),
    }
    period = 2.0
    return {
        "time": np.asarray([0.0, period / 2]),
        "flux": np.full(2, 0.75),
        "sigma": 1e-3,
        "effective_period": period,
        "host_mass": mass,
        "host_radius": float(radius[0]),
        "u1": 0.0,
        "u2": 0.0,
        "exptime_days": 1e-6,
        "bin_integration_days": 0.0,
        "nsamples": 1,
        "n_dimensions": 6,
        "hosted": True,
        "target_feh": 0.0,
    }, table, target_mag


def test_identical_background_binary_has_equal_physical_primary_and_secondary_depths():
    scene, table, target_mag = _equal_star_background_binary()
    population = background_star_population(table, 1.0, target_mag, resolution_limit_arcsec=1.0)
    # D-35: equal-mass pairs belong to twin support; effective_period is already supplied.
    scalar = _EclipsingPairIntegrand(**scene, population=population, use_2xP=True)
    # Actual production integrand: equal stellar masses, circular edge-on orbit,
    # one retained background source. No likelihood/forward-model mock.
    parameters = np.asarray([1.0, 0.0, 90.0, 90.0, 0.5, 0.0])
    actual = scalar.log_likelihood(parameters)
    # Analytic data make chi^2 exactly zero. Only the isotropic geometric-band
    # factor remains; use independent cube-root evaluation of Kepler's law.
    axis_cm = np.cbrt(G * (2.0 * scene["host_mass"] * Msun)
                     * (scene["effective_period"] * SECONDS_PER_DAY) ** 2 / (4.0 * math.pi ** 2))
    expected = math.log(2.0 * scene["host_radius"] * Rsun / axis_cm)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8,
                               err_msg="a hosted pair's secondary flux must share its primary's target-frame scaling")
