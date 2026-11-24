"""The actual MC and nested consumers retain the same physical observations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from trex.evidence import nested_evidence as ne
from trex.evidence.evidence_escalation import run_nested_escalation
from trex.evidence.joint_observations import ScenarioObservations
from trex.evidence.marginal_likelihoods import N_NEIGHBOR_SCENARIOS, N_TARGET_SCENARIOS
from trex.evidence.mc_target import calc_target_evidences, resolved_neighbor_host
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import REFERENCE, joint_case, light, observation


def test_zero_share_auxiliary_adds_the_same_normalized_density_to_every_mc_scenario(tmp_path):
    scene, observations, lights = joint_case(tmp_path, auxiliary_zero=True)
    primary, auxiliary = observations.observations
    one = ScenarioObservations(JointLikelihood(ObservationSet((primary,), REFERENCE)),
                               host_id="target", band_light={primary.band_id: lights[primary.band_id]},
                               epoch_days=0., target_magnitudes={"space": 10.})
    two = ScenarioObservations(JointLikelihood(observations), host_id="target",
                               band_light=lights, epoch_days=0., target_magnitudes={"space": 10., "ground": 9.})
    results = []
    for joint in (one, two):
        records = {}
        neighbors = scene.neighbor_dicts_for_evidence({1: 1.})
        values, status = calc_target_evidences(
            primary.time_axis.time_days, primary.flux, primary.sigma, 3., 1000., 1., 1., .2, .3, None,
            joint_observations=joint, stellar_light=lights[primary.band_id], target_aperture_flux_fraction=.9,
            target_source_id="target", n_draws=128, random_seed=7, target_feh=0., target_teff_k=5700.,
            target_tess_mag=10., plx=10., background_population=scene.background_population(),
            background_area_sq_deg=.1, resolution_limit_arcsec=10., nearby_stars=neighbors,
            representative_fit_out=records,
        )
        assert status == "ok"
        assert all(record["status"] != "unavailable" for record in records.values())
        results.append(values)
    shift = norm.logpdf(auxiliary.flux, loc=1., scale=auxiliary.sigma).sum()
    assert np.isfinite(results).all()
    np.testing.assert_allclose(results[1] - results[0], shift, rtol=0., atol=2e-9)


@pytest.mark.parametrize("binary", [False, True])
def test_nested_uses_the_joint_normalized_density_and_preserves_geometric_prior(binary):
    row = observation()
    joint = ScenarioObservations(JointLikelihood(ObservationSet((row,), REFERENCE)),
                                 host_id="target", band_light={row.band_id: light()}, epoch_days=0.)
    options = dict(time=row.time_axis.time_days, flux=row.flux, sigma=row.sigma, host_mass=1.,
                   host_radius=1., host_teff_k=5700., u1=.2, u2=.3, exptime_days=None,
                   bin_integration_days=0., nsamples=4, population=None, n_dimensions=4,
                   target_feh=0., stellar_light=light(), joint_observations=joint)
    if binary:
        integrand = ne._EclipsingPairIntegrand(effective_period=3., **options)
        values = np.array([.5, .1, 70., 89.])
        radius, temperature = integrand.secondary_geometry(values[0])
        a_cm, a_rs = integrand.draw_axes(values[0])
        physical = dict(secondary_mass=.5, secondary_radius=radius, secondary_teff=temperature)
        ratio = radius
    else:
        integrand = ne._TransitingPlanetIntegrand(period_days=3., use_flat_priors=False, **options)
        values = np.array([3., .1, 70., 89.])
        a_cm, a_rs = integrand.a_cm, integrand.a_rs
        physical, ratio = {"planet_radius": 3.}, integrand.occultor_radius_ratio(3.)
    expected = joint.evaluate(host_mass=1., host_radius=1., host_teff=5700., host_metallicity=0.,
                              period_days=3., inc_deg=89., a_cm=a_cm, ecc=.1, argp_deg=70., **physical)
    geometric = 1-ne._transit_band_low(ratio, .1, 70., a_rs)
    assert integrand.log_likelihood(values) == pytest.approx(expected + np.log(geometric), abs=1e-10)
    assert not ne._refinement_native_capable(integrand)


def test_tess_zero_share_does_not_exclude_a_ground_host(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    first, second = observations.observations
    rows = (replace(first, source_shares={"target": 1., "neighbor": 0.}),
            replace(second, source_shares={"target": .8, "neighbor": .2}))
    joint = ScenarioObservations(JointLikelihood(ObservationSet(rows, REFERENCE)),
                                 host_id="target", band_light=lights, epoch_days=0.)
    neighbor = scene.neighbor_dicts_for_evidence({1: 1.})[0]
    neighbor["aperture_flux_fraction"] = 0.
    neighbor["stellar_atmosphere"]["av_mag"] = .25
    host = resolved_neighbor_host(neighbor, 0., 0, .01, stellar_light=lights[first.band_id],
                                  joint_observations=joint)
    assert host["source_id"] == "neighbor"
    assert host["aperture_flux_fraction"] == .2
    assert host["stellar_light"].target_av_mag == .25


def test_joint_evidence_entrypoints_reject_missing_primary_band_light(tmp_path):
    scene, observations, lights = joint_case(tmp_path)
    primary = observations.observations[0]
    joint = ScenarioObservations(JointLikelihood(observations), host_id="target",
                                 band_light=lights, epoch_days=0.)
    common = dict(
        time=primary.time_axis.time_days, flux=primary.flux, sigma=primary.sigma,
        period_days=3., depth_ppm=1000., M_s_Msun=1., R_s_Rsun=1., u1=.2, u2=.3,
        exptime_days=None, bin_integration_days=0., joint_observations=joint,
        target_aperture_flux_fraction=.9, target_feh=0., target_teff_k=5700.,
        contrast_separations=None, contrast_values=None, plx=10., target_tess_mag=10.,
        background_population=None, background_area_sq_deg=None, resolution_limit_arcsec=10.,
        nearby_stars=scene.neighbor_dicts_for_evidence(), nsamples=4, random_seed=7,
    )
    with pytest.raises(ValueError, match="require exact physical band light"):
        calc_target_evidences(**common, n_draws=4)
    logz = np.full(N_TARGET_SCENARIOS + N_NEIGHBOR_SCENARIOS, -np.inf)
    neighbor_tp = N_TARGET_SCENARIOS
    # A sole finite, low-ESS scenario carries all probability, so nested planning must run.
    logz[neighbor_tp] = 1.
    with pytest.raises(ValueError, match="require exact physical band light"):
        run_nested_escalation(
            **common, lnZ=logz, ess={neighbor_tp: 1.}, support_state={}, n_draws=4, escalation_budget_floor=4,
            checkpoint_directory=tmp_path / "nested",
        )
