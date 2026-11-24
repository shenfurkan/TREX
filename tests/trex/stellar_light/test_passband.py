"""Analytic controls for the coupled angular/total photon normalization defect."""

import numpy as np
import pytest
from astropy import constants as c

from trex.stellar_light.passband import PhotonPassband, integrate_photon_intensity
from trex.stellar_light.release import AngularSpectrum
from trex.stellar_light.components import apparent_photon_flux


def test_nonnegative_original_rays_survive_a_negative_auxiliary_quadratic():
    """The angular compiler must not impose suitability of an unused polynomial."""
    from trex.foundation.constants import G, Msun, Rsun
    from trex.stellar_light import PhotonAtmosphereGrid, StellarLight
    from trex.stellar_light.passband import angular_moment
    from trex.stellar_light.passband import reanchor_r_method_rays

    wave = np.array([6e-5, 8e-5, 1e-4])
    mu = np.linspace(0., 1., 79)[1:]
    # A steep, everywhere nonnegative synthetic law exposes the same reduction
    # defect. The bright limb shelf collapsing over a narrow outer annulus is
    # the spherical drop-off the r-method edge requires (a bare power of mu has
    # no limb and is refused at compilation). The shelf stays small (0.02) so the
    # mu**4 quadratic keeps its negative interior minimum (about -0.1); its drop over
    # dr ~ 0.001 per ray (slope ~ 7) still exceeds the mu**4 maximum slope 4r(1-r^2) ~ 1.5.
    shelf = np.clip((mu - 0.05)/0.04, 0., 1.)
    spectrum = AngularSpectrum(4000., 4.5, 0., wave, mu,
        np.broadcast_to((mu[:, None]**4 + 0.02*shelf[:, None]), (78, 3)),
        "synthetic steep angular law")
    band = PhotonPassband(wave, np.ones(3), np.zeros(3), "synthetic flat", "synthetic zero")
    with pytest.raises(ValueError, match="negative intensity"):
        integrate_photon_intensity(spectrum, band, 0.)
    grid = PhotonAtmosphereGrid.from_spectra([spectrum], band, [0.])
    light = StellarLight(grid, 0.)
    # Invert g=GM/R² [CGS] to supply the exact synthetic atmosphere coordinate.
    radius = np.sqrt(G*Msun/10**4.5)/Rsun
    component = light.prepare_component(4000., 1., radius, 0., 0.)
    rays, intensity = component.angular_profile
    assert np.all(intensity >= 0.)
    # The compiled rays ARE the production r-method anchoring of the raw
    # integration: the stored disc ends at the transit-visible limb.
    _, _, raw_mu, raw_intensity, _ = integrate_photon_intensity(
        spectrum, band, 0., require_nonnegative_quadratic=False)
    anchored_expected, edge = reanchor_r_method_rays(raw_mu, raw_intensity)
    np.testing.assert_array_equal(rays[0], raw_mu)
    np.testing.assert_allclose(intensity[0], anchored_expected, rtol=0, atol=0)
    assert edge["radius"] < 1.
    # This synthetic polynomial is negative at its interior minimum, not at the limb.
    vertex = -component.u1/(2*component.u2)
    assert 0. < vertex < 1.
    assert 1-component.u1*vertex-component.u2*vertex**2 < 0.
    moment = sum(angular_moment(m, p) for m, p in zip(rays, intensity))
    assert 2*np.pi*moment == pytest.approx(component.surface_photon_flux, rel=2e-14)
    assert grid.source["quadratic_errors"][0]["quadratic_nonnegative"] is False
    with pytest.raises(ValueError, match="no nonnegative quadratic reduction"):
        light.atmosphere(4000., 1., radius, 0., 0.)


def test_absolute_photons_angular_moment_and_extinction_once():
    # Synthetic uniform radiance: F=pi*I and integral lambda dlambda are analytic.
    wave = np.array([6e-5, 8e-5, 1e-4])
    mu = np.linspace(0., 1., 101)[1:]
    spectrum = AngularSpectrum(4000, 4.5, 0., wave, mu, np.ones((100, 3)), "synthetic")
    band = PhotonPassband(wave, np.ones(3), np.ones(3), "synthetic flat", "synthetic grey")
    flux, polynomial, rays, profile, errors = integrate_photon_intensity(spectrum, band, 2.5)
    photon_integral = (wave[-1]**2-wave[0]**2)/(2*c.h.cgs.value*c.c.cgs.value)*.1
    # The explicit zero-path tangent triangular sliver is included, not discarded.
    expected_moment = .5-mu[0]**2/6
    assert flux == pytest.approx(2*np.pi*photon_integral*expected_moment, rel=2e-15)
    assert np.pi*(polynomial[0]+polynomial[1]/3+polynomial[2]/6) == pytest.approx(flux, rel=2e-15)
    assert profile[-1] == pytest.approx(photon_integral)
    assert abs(errors["disk_flux_relative_error"]) < 1e-14
    observed = apparent_photon_flux(flux, 1., 10.)
    assert apparent_photon_flux(flux, 2., 10.)/observed == 4
    assert apparent_photon_flux(flux, 1., 20.)/observed == .25


def test_quadratic_reduction_refuses_a_partial_disc():
    from trex.stellar_light.passband import quadratic_reduction

    mu = np.array([.01, .5, 1.])
    with pytest.raises(ValueError, match="whole disc"):
        quadratic_reduction(mu, .4+.6*mu)


def test_r_method_edge_locates_and_reanchors_a_spherical_limb():
    """Analytic piecewise-linear limb: edge interval, anchor identity, refusals.

    Rays mu=[0, .01, .05, .15, .4, .7, 1], I=[0, .005, .02, .3, .8, .95, 1]:
    the steepest radial drop is the (0.05, 0.15) interval (drop 27.8 per unit
    radius; the next steepest is 12.5), so the edge radius is the midpoint of
    its bracketing radii sqrt(1-0.05^2)=0.9987493 and sqrt(1-0.15^2)=0.9886860.
    """
    from trex.stellar_light.passband import r_method_edge, reanchor_r_method_rays

    mu = np.array([0., .01, .05, .15, .4, .7, 1.])
    intensity = np.array([0., .005, .02, .3, .8, .95, 1.])
    edge = r_method_edge(mu, intensity)
    assert edge["radius"] == pytest.approx(0.5*(np.sqrt(1-.05**2)+np.sqrt(1-.15**2)), abs=1e-12)
    np.testing.assert_allclose(
        edge["radius_interval"], sorted([np.sqrt(1-.05**2), np.sqrt(1-.15**2)]), atol=1e-12, rtol=0)
    assert edge["mu_interval"] == pytest.approx([.05, .15], abs=0)
    assert 0 < edge["beyond_edge_photon_fraction"] < 0.02
    anchored, again = reanchor_r_method_rays(mu, intensity)
    assert again is edge or again == edge
    # Same mu grid and ray count; the centre is exact and mu=0 carries the
    # edge intensity, not the artificial tangent zero.
    assert anchored.shape == intensity.shape
    assert anchored[-1] == intensity[-1]
    assert anchored[0] == pytest.approx(
        np.interp(np.sqrt(1-edge["radius"]**2), mu, intensity), abs=1e-12)
    assert np.all(anchored >= 0)
    # Interior sample: the anchored value is the raw profile at the mapped
    # interior position (radius scaling r -> r/R_edge is exact and uniform).
    position = np.sqrt(1-edge["radius"]**2*(1-mu**2))
    np.testing.assert_allclose(anchored, np.interp(position, mu, intensity), atol=1e-12, rtol=0)


def test_r_method_edge_refuses_profiles_without_a_spherical_limb():
    """Uniform, centre-peaked and mid-disc-steep laws have no limb to anchor."""
    from trex.stellar_light.passband import r_method_edge

    mu = np.r_[0., np.linspace(0.01, 1., 40)]
    with pytest.raises(ValueError, match="outward radial intensity drop"):
        r_method_edge(mu, np.ones_like(mu))
    # Only a drop concentrated in the innermost interval is centre-peaked; a smooth
    # mu**n law is steepest mid-disc and is refused by the photon-fraction guard below.
    with pytest.raises(ValueError, match="not the disc centre"):
        r_method_edge(mu, np.where(mu == 1., 1., 0.5))
    with pytest.raises(ValueError, match="no resolved spherical drop-off"):
        r_method_edge(mu, (mu**4))
    with pytest.raises(ValueError, match="ordered complete nonnegative rays"):
        r_method_edge(np.array([.01, .5, 1.]), np.array([.2, .6, 1.]))
