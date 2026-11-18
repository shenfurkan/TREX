"""Multi-band preparation, identity binding and per-host routing controls."""

import numpy as np
import pytest

from trex.stellar_light.components import BandStellarLight
from trex.stellar_light.host_limb_darkening import (
    LimbDarkeningUnavailableError,
    resolve_host_ldc,
)
from trex.stellar_light.interpolation import (
    PhotonAtmosphereGrid,
    UnsupportedAtmosphereError,
)
from trex.stellar_light.passband import PhotonPassband
from trex.stellar_light.release import AngularSpectrum, V3_RELEASE

WAVE = np.array([0.5e-4, 1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4])
MU = np.linspace(0.01, 1.0, 20)


def _spectrum(teff=3000.0, logg=4.5, mh=0.0):
    intensity = (
        (1.0 + MU)[:, None] * (WAVE / WAVE[0])[None, :] ** -1.0
        * 10.0 ** (0.1 * mh) * 1.0e6
    )
    return AngularSpectrum(
        teff, logg, mh, WAVE, MU, intensity,
        "synthetic official-node spectrum", V3_RELEASE,
    )


def _band(response_id, low, high):
    grid = np.array([low, high])
    return PhotonPassband(
        grid, np.ones(2), np.zeros(2),
        "synthetic flat band {0}".format(response_id),
        "synthetic zero extinction",
        response_id,
    )


BAND_A = _band("SYNTHETIC/A.BAND", 0.6e-4, 1.4e-4)
BAND_B = _band("SYNTHETIC/B.BAND", 1.5e-4, 2.5e-4)


def _anchored_surface_flux(spectrum, band, av_mag):
    """Production-convention expectation: the r-method-anchored disc moment."""
    from trex.stellar_light.passband import (
        angular_moment, integrate_photon_intensity, reanchor_r_method_rays)

    _, _, mu, intensity, _ = integrate_photon_intensity(
        spectrum, band, av_mag, require_nonnegative_quadratic=False)
    anchored, _ = reanchor_r_method_rays(mu, intensity)
    return 2.0*np.pi*angular_moment(mu, anchored)


def test_multiband_compilation_matches_separate_band_integration_exactly():
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        [_spectrum()], {"SYNTHETIC/A.BAND": BAND_A, "SYNTHETIC/B.BAND": BAND_B}, [0.0])
    spectrum = _spectrum()
    for band_id, band in (("SYNTHETIC/A.BAND", BAND_A), ("SYNTHETIC/B.BAND", BAND_B)):
        grid = grids[band_id]
        separate = PhotonAtmosphereGrid.from_spectra([spectrum], band, [0.0])
        node = tuple(int(np.searchsorted(axis, value)) for axis, value in
                     zip(grid.axes[:3], (3000.0, 4.5, 0.0)))
        cell = node + (0,)
        assert np.array_equal(grid.table[cell], separate.table[cell])
        expected = _anchored_surface_flux(spectrum, band, 0.0)
        assert grid.table[cell][0] == pytest.approx(expected, rel=2e-14)
        assert grid.source["response_id"] == band_id
        assert grid.source["passband_input_id"] == band.input_id
        assert grid.angular is not None
        mu = grid.angular[0][node]
        intensity = grid.angular[1][node + (0,)]
        profile_flux = 2.0*np.pi*np.sum(
            np.diff(mu)*((2*mu[:-1] + mu[1:])*intensity[:-1]
                         + (mu[:-1] + 2*mu[1:])*intensity[1:])/6.0)
        assert profile_flux == pytest.approx(expected, rel=2e-14)
    a = tuple(int(np.searchsorted(axis, value)) for axis, value in
              zip(grids["SYNTHETIC/A.BAND"].axes[:3], (3000.0, 4.5, 0.0))) + (0,)
    b = tuple(int(np.searchsorted(axis, value)) for axis, value in
              zip(grids["SYNTHETIC/B.BAND"].axes[:3], (3000.0, 4.5, 0.0))) + (0,)
    assert grids["SYNTHETIC/A.BAND"].table[a][0] != pytest.approx(
        grids["SYNTHETIC/B.BAND"].table[b][0])


def test_band_swap_cannot_reuse_a_compiled_preparation(tmp_path):
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        [_spectrum()], {"SYNTHETIC/A.BAND": BAND_A}, [0.0])
    path = tmp_path/"prepared.npz"
    grids["SYNTHETIC/A.BAND"].to_file(path)
    model_id = grids["SYNTHETIC/A.BAND"].model_id
    with pytest.raises(ValueError, match="prepared for response"):
        PhotonAtmosphereGrid.from_file(
            path, expected_model_id=model_id, expected_response_id="SYNTHETIC/B.BAND")
    with pytest.raises(ValueError, match="different response table"):
        PhotonAtmosphereGrid.from_file(
            path, expected_model_id=model_id,
            expected_passband_input_id=BAND_B.input_id)
    loaded = PhotonAtmosphereGrid.from_file(
        path, expected_model_id=model_id, expected_response_id="SYNTHETIC/A.BAND",
        expected_passband_input_id=BAND_A.input_id)
    # Unprepared nodes stay NaN; round-trip equality must compare those too.
    assert np.array_equal(
        loaded.table, grids["SYNTHETIC/A.BAND"].table, equal_nan=True)


def test_band_keys_must_be_canonical_and_unique():
    with pytest.raises(ValueError, match="differs from its response identity"):
        PhotonAtmosphereGrid.from_spectra_multi(
            [_spectrum()], {"SYNTHETIC/OTHER.BAND": BAND_A}, [0.0])
    with pytest.raises(ValueError, match="instrument-qualified"):
        PhotonAtmosphereGrid.from_spectra_multi(
            [_spectrum()],
            {"a": PhotonPassband(BAND_A.wavelength_cm, BAND_A.response, BAND_A.a_lambda_over_av,
                                 "synthetic", "synthetic zero"),
             "b": BAND_B},
            [0.0])


def test_absent_band_support_is_a_band_specific_failure_not_a_solar_substitute():
    too_wide = _band("SYNTHETIC/WIDE.BAND", 0.4e-4, 9.0e-4)
    failures = []
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        [_spectrum()], {"SYNTHETIC/A.BAND": BAND_A, "SYNTHETIC/WIDE.BAND": too_wide},
        [0.0], node_failures=failures)
    assert grids["SYNTHETIC/A.BAND"] is not None
    assert grids["SYNTHETIC/WIDE.BAND"] is None
    assert {entry["band"] for entry in failures} == {"SYNTHETIC/WIDE.BAND"}
    with pytest.raises(UnsupportedAtmosphereError):
        grids["SYNTHETIC/A.BAND"].evaluate(5777.0, 4.44, 0.0, 0.0)


def _compositional_grids():
    # Bracketing log g nodes keep the radius-derived gravity usable even when
    # floating-point roundoff lands just off the intended 4.5-dex node.
    spectra = [_spectrum(logg=logg, mh=mh)
               for mh in (0.0, -0.5) for logg in (4.0, 4.5, 5.0)]
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        spectra, {"SYNTHETIC/A.BAND": BAND_A}, [0.0, 1.0])
    return BandStellarLight(grids, target_av_mag=0.0)


def test_own_atmosphere_pair_keeps_radius_scaling_and_applies_extinction_once():
    grey = PhotonPassband(
        np.array([0.6e-4, 1.4e-4]), np.ones(2), np.ones(2),
        "synthetic grey band", "synthetic grey A_lambda/A_V=1", "SYNTHETIC/GREY.BAND")
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        [_spectrum(logg=logg, mh=mh) for mh in (0.0, -0.5)
         for logg in (4.0, 4.5, 5.0)],
        {"SYNTHETIC/GREY.BAND": grey}, [0.0, 1.0])
    light = BandStellarLight(grids)
    from trex.foundation.constants import G, Msun, Rsun
    mass, radius = 1.0, np.sqrt(G*Msun/10**4.5)/Rsun
    primary = (3000.0, mass, radius)
    # Same surface gravity, twice the radius: mass scales as R^2.
    secondary = (3000.0, 4.0*mass, 2.0*radius)
    ratio, first, second = light.pair_own_atmospheres(
        "SYNTHETIC/GREY.BAND", primary, secondary, 0.0, 0.0, 0.0, 0.0)
    # Equal SED and radius-squared scaling: a 2x radius companion has 4x flux.
    assert ratio == pytest.approx(4.0, rel=2e-14)
    assert second.surface_photon_flux == pytest.approx(first.surface_photon_flux, rel=2e-15)
    # A different companion composition changes only the companion surface light.
    metal_poor, _, poor_second = light.pair_own_atmospheres(
        "SYNTHETIC/GREY.BAND", primary, secondary, 0.0, -0.5, 0.0, 0.0)
    assert metal_poor != pytest.approx(4.0)
    assert poor_second.surface_photon_flux != pytest.approx(first.surface_photon_flux)
    # One body reddened by A_V=1 with grey kappa=1: exactly one 10^-0.4 factor.
    reddened_secondary, _, _ = light.pair_own_atmospheres(
        "SYNTHETIC/GREY.BAND", primary, secondary, 0.0, 0.0, 0.0, 1.0)
    assert reddened_secondary / ratio == pytest.approx(10.0 ** -0.4, rel=2e-14)


def test_each_star_routes_its_own_composition_and_extinction_per_band():
    light = _compositional_grids()
    # 3000 K, 4.5 dex, radius chosen to reproduce g=10^4.5 CGS.
    from trex.foundation.constants import G, Msun, Rsun
    mass, radius = 1.0, np.sqrt(G*Msun/10**4.5)/Rsun
    solar = light.prepare_component("SYNTHETIC/A.BAND", 3000.0, mass, radius, 0.0, 0.0)
    metal_poor = light.prepare_component("SYNTHETIC/A.BAND", 3000.0, mass, radius, -0.5, 0.0)
    reddened = light.prepare_component("SYNTHETIC/A.BAND", 3000.0, mass, radius, 0.0, 1.0)
    assert solar.surface_photon_flux != pytest.approx(metal_poor.surface_photon_flux)
    assert reddened.surface_photon_flux == pytest.approx(
        solar.surface_photon_flux, rel=1e-15)  # synthetic zero extinction curve
    with pytest.raises(UnsupportedAtmosphereError):
        light.prepare_component("SYNTHETIC/A.BAND", 3000.0, mass, radius, 0.5, 0.0)


def test_band_light_refuses_an_unprepared_band_and_records_identities():
    light = _compositional_grids()
    with pytest.raises(ValueError, match="was not prepared"):
        light.light("SYNTHETIC/B.BAND")
    configuration = light.model_configuration()
    assert set(configuration) == {"SYNTHETIC/A.BAND"}
    assert configuration["SYNTHETIC/A.BAND"]["response_id"] == "SYNTHETIC/A.BAND"
    assert configuration["SYNTHETIC/A.BAND"]["passband_input_id"] == BAND_A.input_id


def test_limb_darkening_routes_tess_exactly_and_refuses_other_bands():
    node = resolve_host_ldc("TESS/TESS.T", 5777.0, 4.44, 0.0)
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid
    reference = load_triceratops_tess_grid().select(5777.0, 4.44, 0.0)
    assert (node.u1, node.u2, node.used_teff_k, node.used_metallicity_dex) == (
        reference.u1, reference.u2, reference.used_teff_k, reference.used_metallicity_dex)
    with pytest.raises(LimbDarkeningUnavailableError, match="prepared PHOENIX"):
        resolve_host_ldc("SYNTHETIC/A.BAND", 5777.0, 4.44, 0.0)
    with pytest.raises(LimbDarkeningUnavailableError, match="Zenodo 10.5281/zenodo.7339706"):
        resolve_host_ldc("PLATO/NCAM.V2", 5777.0, 4.44, 0.0)
    with pytest.raises(LimbDarkeningUnavailableError, match="Sing et al. 2026"):
        resolve_host_ldc("Roman/WFI01.F146", 3000.0, 4.5, 0.0)
    with pytest.raises(ValueError, match="canonical response id"):
        resolve_host_ldc("", 5777.0, 4.44, 0.0)


def test_cool_and_evolved_hosts_keep_their_own_published_atmosphere_domains():
    from dataclasses import replace
    from trex.foundation.constants import G, Msun, Rsun
    from trex.stellar_light.host_limb_darkening import CLARET_SOURCE_ID, PHOENIX_SOURCE_ID

    cool_spectra = [_spectrum(teff=3200.0, logg=logg, mh=0.0)
                    for logg in (4.0, 4.5, 5.0)]
    evolved_spectra = []
    for logg in (2.0, 2.5, 3.0):
        spectrum = _spectrum(teff=5000.0, logg=logg, mh=0.0)
        evolved_spectra.append(replace(spectrum, intensity=2.0*spectrum.intensity))
    grids = PhotonAtmosphereGrid.from_spectra_multi(
        cool_spectra + evolved_spectra, {"SYNTHETIC/A.BAND": BAND_A}, [0.0])
    grid = grids["SYNTHETIC/A.BAND"]
    light = BandStellarLight(grids)
    for teff, logg in ((3200.0, 4.5), (5000.0, 2.5)):
        node = tuple(int(np.searchsorted(axis, value)) for axis, value in
                     zip(grid.axes[:3], (teff, logg, 0.0)))
        assert (grid.axes[0][node[0]], grid.axes[1][node[1]]) == (teff, logg)
        assert np.isfinite(grid.table[node + (0,)]).all()
        # Matched-radius geometry: the original-ray profile integrates to the
        # same surface photon flux as the tabulated total at this node.
        mu = grid.angular[0][node]
        intensity = grid.angular[1][node + (0,)]
        profile_flux = 2.0*np.pi*np.sum(
            np.diff(mu)*((2*mu[:-1] + mu[1:])*intensity[:-1]
                         + (mu[:-1] + 2*mu[1:])*intensity[1:])/6.0)
        assert profile_flux == pytest.approx(grid.table[node + (0,)][0], rel=2e-14)
    cool = light.prepare_component(
        "SYNTHETIC/A.BAND", 3200.0, 1.0,
        np.sqrt(G*Msun/10**4.5)/Rsun, 0.0, 0.0)
    evolved = light.prepare_component(
        "SYNTHETIC/A.BAND", 5000.0, 1.0,
        np.sqrt(G*Msun/10**2.5)/Rsun, 0.0, 0.0)
    assert cool.surface_photon_flux > 0.0 and evolved.surface_photon_flux > 0.0
    # The fixture's evolved spectrum is exactly twice the cool one: separate
    # node compilation must preserve that ratio, not substitute one atmosphere.
    assert evolved.surface_photon_flux == pytest.approx(2.0*cool.surface_photon_flux, rel=2e-14)

    # Published routing: the cool solar-metallicity host takes PHOENIX-COND
    # table 15 below 3500 K, and the evolved low-gravity host takes the Claret
    # (2017) ATLAS table 25 extension instead of the nearest dwarf node.
    cool_ldc = resolve_host_ldc("TESS/TESS.T", 2949.0, 4.5, 0.0)
    assert cool_ldc.source == PHOENIX_SOURCE_ID and cool_ldc.interpolated
    evolved_ldc = resolve_host_ldc("TESS/TESS.T", 5000.0, 2.5, 0.0)
    assert evolved_ldc.source == CLARET_SOURCE_ID and evolved_ldc.interpolated
    assert (evolved_ldc.used_teff_k, evolved_ldc.used_logg_cgs) == (5000.0, 2.5)
    assert (evolved_ldc.u1, evolved_ldc.u2) == pytest.approx((0.4055, 0.2281), abs=1e-12)
