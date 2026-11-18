"""Angular occultation controls: independent integrals plus production wiring parity.

The central/grazing and ray-mixture cases build their expected values from
independent radial/adaptive integrals. The ensemble and both-body cases below
are wiring/parity controls over the same production reduction, not second
physical oracles.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from trex.stellar_light.occultation import occulted_fraction
from trex.stellar_light.passband import angular_moment


@pytest.mark.parametrize("k", [.01, .3, .9, 1., 1.2])
def test_central_tabulated_occultation_matches_exact_radial_moment(k):
    mu = np.array([0., .05, .2, .6, 1.])
    intensity = np.array([0., .1, .8, 1.1, 1.2])
    boundary = np.sqrt(max(1-k*k, 0.))
    knots = np.r_[boundary, mu[mu > boundary]]
    expected = (angular_moment(knots, np.interp(knots, mu, intensity))
                / angular_moment(mu, intensity)) if k < 1 else 1.
    assert occulted_fraction(0., k, mu, intensity) == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("d,k", [(0.1, .2), (.7, .9), (1., .3), (1.2999, .3), (1.3, .3), (.2, 1.3)])
def test_contacts_and_grazing_match_independent_adaptive_radial_integral(d, k):
    mu = np.array([0., .05, .2, .6, 1.])
    intensity = np.array([0., .1, .8, 1.1, 1.2])
    def ring(r):
        angle = 2*np.arccos(np.clip((r*r+d*d-k*k)/(2*d*r), -1., 1.))
        return r*angle*np.interp(np.sqrt(1-r*r), mu, intensity)
    boundaries = sorted(set([0., 1., *np.sqrt(1-mu**2),
                             *[r for r in (abs(d-k), d+k) if 0 < r < 1]]))
    blocked = sum(quad(ring, a, b, epsabs=1e-12)[0] for a, b in zip(boundaries[:-1], boundaries[1:]))
    expected = blocked/(2*np.pi*angular_moment(mu, intensity))
    low = occulted_fraction(d, k, mu, intensity, order=16)
    high = occulted_fraction(d, k, mu, intensity, order=32)
    assert high == pytest.approx(expected, abs=2e-10)
    assert low == pytest.approx(high, abs=2e-8)


def test_extinction_corner_collapse_equals_the_sixteen_corner_sum():
    """The A_V pair is summed before quadrature, not interpolated as a new ray set."""
    from trex.stellar_light.angular import interpolate_angular

    axes = (np.array([3000., 4000.]), np.array([4., 5.]), np.array([-.5, .5]), np.array([0., 1.]))
    weights = (np.array([.25, .75]), np.array([.5, .5]), np.array([.75, .25]), np.array([.75, .25]))
    coordinate = (3000.+.75*1000., 4.5, -.25, .25)
    mu = np.array([[[[0., .1+.02*(2*i+j+k), .6, 1.] for k in range(2)]
                    for j in range(2)] for i in range(2)])
    intensity = np.empty((2, 2, 2, 2, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for a in range(2):
                    intensity[i, j, k, a] = (1.+.2*i+.1*j+.05*k)*(1.+.3*a)*np.array([0., .2, .8, 1.])
    rays, fields, valid = interpolate_angular(axes, (mu, intensity), coordinate)
    assert valid and rays.shape == (8, 4) and fields.shape == (8, 4)

    d, k = np.array([0., .7, 1.1, 2.]), .8
    numerator = denominator = 0.
    for i in range(2):
        for j in range(2):
            for kk in range(2):
                for a in range(2):
                    weight = weights[0][i]*weights[1][j]*weights[2][kk]*weights[3][a]
                    moment = angular_moment(mu[i, j, kk], intensity[i, j, kk, a])
                    numerator += weight*moment*occulted_fraction(
                        d, k, mu[i, j, kk], intensity[i, j, kk, a])
                    denominator += weight*moment
    expected = numerator/denominator
    np.testing.assert_allclose(occulted_fraction(d, k, rays, fields), expected, atol=2e-15, rtol=0)


def test_zero_weight_missing_corner_never_invalidates_exact_rays():
    from trex.stellar_light.angular import interpolate_angular

    axes = (np.array([3000., 4000.]), np.array([5.]), np.array([0.]), np.array([0., 1.]))
    mu = np.array([[[[0., .1, .6, 1.]]], [[[0., .2, .7, 1.]]]])
    intensity = np.full((2, 1, 1, 2, 4), np.nan)
    intensity[0, 0, 0, 0] = np.array([0., .2, .8, 1.])
    intensity[0, 0, 0, 1] = np.array([0., .4, 1., 1.2])
    rays, fields, valid = interpolate_angular(axes, (mu, intensity), (3000., 5., 0., 0.))
    assert valid and np.isfinite(fields).all()
    assert np.array_equal(rays[0], mu[0, 0, 0]) and np.array_equal(fields[0], intensity[0, 0, 0, 0])

    _, _, valid = interpolate_angular(axes, (mu, intensity), (3250., 5., 0., .5))
    assert not valid
    mu[1, 0, 0] = np.nan
    _, _, valid = interpolate_angular(axes, (mu, intensity), (3000., 5., 0., 0.))
    assert valid


def test_original_ray_mixture_keeps_different_node_knots_and_photon_flux():
    from trex.stellar_light.angular import interpolate_angular
    axes = tuple(np.array(v) for v in ([3000., 4000.], [5.], [0.], [0.]))
    mu = np.array([[[[0., .1, .6, 1.]]], [[[0., .2, .7, 1.]]]])
    intensity = np.array([[[[[0., .2, .8, 1.]]]], [[[[0., .7, 1.2, 2.]]]]])
    rays, fields, valid = interpolate_angular(axes, (mu, intensity), (3250., 5., 0., 0.))
    assert valid
    d = np.array([0., .7, 1.1])
    moments = [angular_moment(mu[i, 0, 0], intensity[i, 0, 0, 0]) for i in range(2)]
    blocked = [occulted_fraction(d, .8, mu[i, 0, 0], intensity[i, 0, 0, 0]) for i in range(2)]
    expected = (.75*moments[0]*blocked[0]+.25*moments[1]*blocked[1])/(.75*moments[0]+.25*moments[1])
    np.testing.assert_allclose(occulted_fraction(d, .8, rays, fields), expected, atol=2e-15)


@pytest.mark.parametrize("widths", [0., .003, np.array([.001, .002, .01, .003, .02])])
def test_angular_orbit_exposure_and_bin_boxes_keep_uniform_disc_limit(widths):
    from trex.models.likelihoods import _batman_transit
    time = np.array([-.04, -.02, 0., .02, .04])
    args = (time, 3., .3, 10., 89., 0., 0., .001)
    old = _batman_transit(*args, ecc=.2, argp_deg=30., bin_integration_days=widths)
    angular = _batman_transit(*args, ecc=.2, argp_deg=30., bin_integration_days=widths,
                              angular_profile=(np.array([0., 1.]), np.array([1., 1.])))
    np.testing.assert_allclose(angular, old, atol=2e-8, rtol=0)


def test_both_eclipsed_bodies_consume_their_own_angular_profiles():
    # Wiring check: the expected curve reuses production occulted_fraction.
    from trex.foundation.constants import Rsun
    from trex.models.likelihoods import simulate_EB
    mu = np.array([0., .05, .2, .6, 1.])
    primary = np.array([0., .1, .8, 1.1, 1.2])
    secondary = np.array([0., .02, .4, .9, 1.])
    k, ratio = .9, .4
    curve, _ = simulate_EB(np.array([0., 1.5]), k, ratio/(1+ratio), 3., 90.,
        10*Rsun, 1., .2, .3, 1e-12, nsamples=1, u1_EB=.1, u2_EB=.1,
        angular_profile=(mu, primary), secondary_angular_profile=(mu, secondary))
    expected = [1-occulted_fraction(0., k, mu, primary)/(1+ratio), 1-ratio/(1+ratio)]
    np.testing.assert_allclose(curve, expected, atol=2e-14, rtol=0)
