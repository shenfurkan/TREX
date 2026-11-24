"""Independent NCAM intensity-profile check against the official PLATO release.

Sources: Morello et al. (2022), RNAAS 6, 248, "First release of PLATO
consortium stellar limb-darkening coefficients", Zenodo
10.5281/zenodo.7339706 v1.1 (CC-BY-4.0), file
``PLATO_WP122400_release_21nov2022.zip``; the ASCII ``*_ints.dat`` members
give I(mu)/I0 on 12 mu nodes for 1D MARCS models. TREX side: its own prepared
PHOENIX SPECINT angular profile integrated through the packaged PLATO NCAM
"as designed camera effective transmission", at PHOENIX (Teff, log g)
nodes with [M/H] = 0. The release's 1997 MARCS models are randomly placed, so
each PHOENIX node is compared with the nearest MARCS model within 10 K and
0.02 dex (declared 2026-11-16, before any profile comparison was seen; the
first run showed only the release layout).

Predeclared expectation and tolerance (declared before any result is seen):
PHOENIX and MARCS 1D intensity profiles are different model atmospheres, so a
few-percent profile difference is expected; agreement is not required to be
exact. The test passes only when, over the compared nodes and 12 mu values,
the normalized-intensity RMS difference is <= 0.02 and the maximum absolute
difference is <= 0.05, and the resulting transit depth for Rp/Rs = 0.1 differs
by <= 1% relative at impact parameters 0.0 and 0.6. The packaged NCAM curve
excludes detector QE; if a residual exceeds this tolerance, model and response
differences must be separated before assigning a cause.

This module downloads data, so it is only meaningful as an explicitly targeted
network crosscheck; it is not part of the default unit-test path. The two
downloading tests carry the ``network`` marker and skip themselves when the
hermetic suite's socket guard is installed: run them only as declared
``--network-targets`` crosschecks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from dataclasses import replace
from datetime import datetime, timezone
from time import perf_counter
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

ZENODO_MD5 = "8ff8c7d40a5c0f3fc05da0d5c9a65c2c"
ZENODO_URL = ("https://zenodo.org/api/records/7339706/files/"
              "PLATO_WP122400_release_21nov2022.zip/content")
MAX_NODES = 6
RX = RY = 512
RADIUS_RATIO = 0.1
IMPACT_PARAMETERS = (0.0, 0.6)


def _hermetic_socket_guard_installed():
    """The hermetic suite's sitecustomize socket guard, not the provider."""
    return getattr(socket.socket.connect, "__module__", "") == "sitecustomize"


def NETWORK_CROSSCHECK(test):
    """Apply the ``network`` marker and the guard-activated skip to one test.

    ``pytest.mark.network(pytest.mark.skipif(...))`` stores the skipif mark as
    inert argument data of the ``network`` mark instead of applying it, so the
    skip never fired: under the hermetic guard (2026-11-18) both
    downloading tests ran, attempted the Zenodo fetch and failed with the
    socket guard's RuntimeError. Applying each mark to the test function is the
    composition pytest actually evaluates.
    """
    test = pytest.mark.skipif(
        _hermetic_socket_guard_installed(),
        reason="downloads the official Zenodo release; run only as an explicitly "
               "declared --network-targets network crosscheck")(test)
    return pytest.mark.network(test)


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "trex-crosscheck"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed provider URL
        destination.write_bytes(response.read())
    return destination


NCAM_INTENSITY_MEMBER = "PLATO_WP122400_release_21nov2022/ascii/PLATO_NCAM_ints.dat"
# The 1997 MARCS models are randomly placed in (Teff, log g) (Morello et al. 2022,
# section 2), so none sits exactly on a PHOENIX node. A MARCS model stands in for
# a PHOENIX node only within these offsets; adjacent release rows 8 K apart
# differ by about 1e-3 in normalized intensity, well inside the 0.02 RMS tolerance.
MATCH_TEFF_K = 10.0
MATCH_LOGG_DEX = 0.02


def _marcs_profiles(archive):
    """Rows of the NCAM intensity table as {(Teff, log g): (mu, I/I(mu=1))}.

    Layout from the release's own header (``#Format Units Label Explanation``
    lines, first read 2026-11-16): ``Teff`` [K], ``logg`` [cgs], ``I0``, then
    ``nI_12`` ... ``nI_01`` normalized to the mu=1 intensity, each header line
    giving its mu node. The mu values are taken from those lines.
    """
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read(NCAM_INTENSITY_MEMBER).decode("utf-8", errors="strict")
    mu_nodes, labels, profiles = [], [], {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            fields = stripped[1:].split()
            # Column rows start with a format code (F.0, E.6, ...); the
            # description and "#Format Units Label" lines do not.
            # Units may contain spaces ("electron / (cm2 s sr)"), so the label
            # is the first field naming a declared column.
            if fields and re.fullmatch(r"[A-Z]\.\d+", fields[0]):
                label = next((field for field in fields[1:]
                              if re.fullmatch(r"Teff|logg|I0|nI_\d+", field)), None)
                if label is None:
                    raise AssertionError("unlabelled NCAM column row: {0!r}".format(stripped))
                labels.append(label)
            match = re.search(r"at mu=([0-9.]+)", stripped)
            if match is not None:
                mu_nodes.append(float(match.group(1)))
            continue
        if not stripped:
            continue
        values = [float(field) for field in stripped.split()]
        if len(values) != 3 + len(mu_nodes):
            raise AssertionError("row has {0} values, header declares {1}".format(len(values), 3 + len(mu_nodes)))
        profiles[(values[0], values[1])] = values[3:]
    if labels[:3] != ["Teff", "logg", "I0"] or len(mu_nodes) != 12:
        raise AssertionError("unexpected NCAM table header: {0} / {1} mu nodes".format(labels[:4], len(mu_nodes)))
    mu = np.asarray(mu_nodes, dtype=float)
    order = np.argsort(mu)
    if not (np.isclose(mu[order][-1], 1.0) and len(set(mu.tolist())) == 12):
        raise AssertionError("NCAM mu nodes do not end at the disc centre")
    return {key: (mu[order], np.asarray(value, dtype=float)[order]) for key, value in profiles.items()}


def _nearest_marcs(marcs, teff, logg):
    """The MARCS model standing in for one PHOENIX node, or None."""
    best = None
    for key in marcs:
        dt, dg = abs(key[0] - teff), abs(key[1] - logg)
        if dt <= MATCH_TEFF_K and dg <= MATCH_LOGG_DEX:
            score = (dt / MATCH_TEFF_K) ** 2 + (dg / MATCH_LOGG_DEX) ** 2
            if best is None or score < best[0]:
                best = (score, key)
    return None if best is None else best[1]


def _phoenix_profile(teff, logg, directory):
    from trex.acquisition.passbands import packaged_plato_transmission
    from trex.stellar_light.release import acquire_phoenix_intensity
    from trex.stellar_light.passband import integrate_photon_intensity

    spectrum = acquire_phoenix_intensity(teff, logg, 0.0, directory)
    response = packaged_plato_transmission(
        "NCAM", "as designed camera effective transmission")
    passband = response.to_photon_passband(
        extinction_source="zero extinction: shape-only NCAM profile crosscheck")
    _, _, mu, intensity, _ = integrate_photon_intensity(
        spectrum, passband, 0.0, require_nonnegative_quadratic=False)
    order = np.argsort(mu)
    mu, intensity = mu[order], intensity[order]
    return mu, intensity / np.interp(1.0, mu, intensity), spectrum, passband


def _r_method_edge(mu, profile):
    """Unfitted steepest radial drop, using native rays only (dimensionless r).

    Claret (2017), A&A 600, A30, DOI 10.1051/0004-6361/201629705,
    Sec. 3, describes the r-method of Espinoza & Jordan (2015).
    Here finite differences locate an interval, not a precisely known inflection:
    its midpoint is the diagnostic edge; both endpoints are sensitivity cases.
    The artificial zero-intensity mu=0 ray must not define the edge.
    """
    mu, profile = np.asarray(mu), np.asarray(profile)
    if (mu.ndim != 1 or mu.size < 3 or profile.shape != mu.shape
            or not np.isfinite(mu).all() or not np.isfinite(profile).all()
            or mu[0] <= 0 or mu[-1] != 1 or np.any(np.diff(mu) <= 0)
            or np.any(profile < 0) or profile[-1] <= 0):
        raise ValueError("r-method requires ordered native positive-mu rays")
    radius = np.sqrt(1 - mu[::-1]**2)
    drop = -np.diff(profile[::-1]) / np.diff(radius)
    index = int(np.argmax(drop))
    if drop[index] <= 0:
        raise ValueError("native profile has no radial intensity drop")
    inner, outer = radius[index:index + 2]
    return {"radius": float((inner + outer) / 2),
            "radius_interval": [float(inner), float(outer)],
            "native_mu_interval": [float(mu[::-1][index + 1]), float(mu[::-1][index])],
            "peak_at_boundary": index in (0, len(drop) - 1)}


def _at_radius(mu, profile, target_mu, radius):
    """I/Ic at r_original=radius*r_comparison; no intensity renormalization."""
    if not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError("comparison radius must be in (0, 1]")
    original_mu = np.sqrt(1 - radius**2 * (1 - np.asarray(target_mu)**2))
    if np.any(original_mu < mu[0]) or np.any(original_mu > mu[-1]):
        raise ValueError("radius mapping would extrapolate native rays")
    return np.interp(original_mu, mu, profile)


def _polar_depth(mu, profile, *, impact_parameter, order=64):
    """Production ring integral on the same truncated disc as the raster check.

    Removing mu<mu_min changes the disc radius to sqrt(1-mu_min^2).
    Both the occultor radius and separation must change to those units too.
    """
    from trex.stellar_light.occultation import occulted_fraction

    cut = mu[0]**2
    radius = np.sqrt(1 - cut)
    scaled_mu = np.sqrt((mu**2 - cut) / (1 - cut))
    return float(occulted_fraction(
        impact_parameter / radius, RADIUS_RATIO / radius,
        scaled_mu, profile, order=order))


def _profile_metrics(mu, profile, reference):
    residual = profile - reference
    return {"rms": float(np.sqrt(np.mean(residual**2))),
            "max_abs": float(np.max(np.abs(residual))),
            "rms_mu_ge_0_1": float(np.sqrt(np.mean(residual[mu >= .1]**2))),
            "max_abs_mu_ge_0_1": float(np.max(np.abs(residual[mu >= .1]))),
            "residual_by_mu": residual.tolist(),
            "depth_relative": {
                str(b): (_transit_depth(mu, profile, impact_parameter=b)
                         / _transit_depth(mu, reference, impact_parameter=b) - 1)
                for b in IMPACT_PARAMETERS}}


def _spectral_shape_envelope(spectrum, passband, query_mu, reference):
    """Conservative bound for ANY nonnegative response on this wavelength support.

    I_band(mu)/I_band(1) is a weighted mean of I_lambda(mu)/I_lambda(1),
    with nonnegative weights R(lambda)*lambda*I_lambda(1). It must lie between
    the spectral extrema. Include bracketing wavelengths: linear spectral
    interpolation cannot escape their ratio envelope. This bounds the supplied
    sampled spectrum, not unobserved high-resolution lines or out-of-band light.
    """
    from scipy.interpolate import interp1d

    if np.min(query_mu) < spectrum.mu[0] or np.max(query_mu) > spectrum.mu[-1]:
        return {"status": "unavailable; query outside native angular support"}
    wave = spectrum.wavelength_cm
    lo = max(0, np.searchsorted(wave, passband.wavelength_cm[0], side="right") - 1)
    hi = min(wave.size, np.searchsorted(wave, passband.wavelength_cm[-1], side="left") + 1)
    centre = spectrum.intensity[-1, lo:hi]
    if np.any(centre <= 0):
        return {"status": "unavailable; nonpositive monochromatic centre intensity",
                "nonpositive_samples": int(np.sum(centre <= 0))}
    intensity = interp1d(spectrum.mu, spectrum.intensity[:, lo:hi], axis=0,
                         bounds_error=True)(query_mu)
    ratio = intensity / centre
    lower, upper = np.min(ratio, axis=-1), np.max(ratio, axis=-1)
    gap = np.maximum(np.maximum(lower - reference, reference - upper), 0.)
    return {"status": "bounded_for_sampled_spectrum",
            "lower": lower.tolist(), "upper": upper.tolist(),
            "marcs_distance_outside_envelope": gap.tolist(),
            "max_distance_outside_envelope": float(np.max(gap)),
            "bracketing_wavelength_cm": wave[[lo, hi - 1]].tolist()}


def _cause_diagnostics(spectrum, passband, mu, profile, mu_marcs, marc, interpolated_marcs):
    """One-variable controls; none changes the predeclared comparison or production."""
    from scipy.interpolate import PchipInterpolator
    from trex.stellar_light.passband import integrate_photon_intensity

    native_mu, native_profile = mu[1:], profile[1:]
    edge = _r_method_edge(native_mu, native_profile)
    corrected = _at_radius(native_mu, native_profile, mu_marcs, edge["radius"])
    diagnostic = {
        "phoenix_release": spectrum.release,
        "phoenix_metadata": spectrum.metadata,
        "phoenix_mh_dex": spectrum.mh_dex,
        "native_mu": native_mu.tolist(), "native_profile": native_profile.tolist(),
        "spectrum_wavelength_cm": spectrum.wavelength_cm[[0, -1]].tolist(),
        "passband_wavelength_cm": passband.wavelength_cm[[0, -1]].tolist(),
        "qe_control": {"status": "unavailable",
                       "reason": "no packaged primary PLATO QE or Morello laboratory response"},
        "r_method": {**edge, **_profile_metrics(mu_marcs, corrected, marc)},
        "edge_interval_sensitivity": [
            {"radius": r, **_profile_metrics(
                mu_marcs, _at_radius(native_mu, native_profile, mu_marcs, r), marc)}
            for r in edge["radius_interval"]],
        "tangent_insertion_max_effect": float(np.max(np.abs(
            np.interp(mu_marcs, mu, profile)
            - np.interp(mu_marcs, native_mu, native_profile)))),
        "pchip_minus_linear": _profile_metrics(
            mu_marcs, PchipInterpolator(mu, profile)(mu_marcs),
            np.interp(mu_marcs, mu, profile)),
        "any_positive_response_envelope_raw": _spectral_shape_envelope(
            spectrum, passband, mu_marcs, marc),
        "any_positive_response_envelope_same_edge": _spectral_shape_envelope(
            spectrum, passband,
            np.sqrt(1 - edge["radius"]**2 * (1 - mu_marcs**2)), marc),
    }
    wave = np.union1d(
        spectrum.wavelength_cm[(spectrum.wavelength_cm >= passband.wavelength_cm[0])
                               & (spectrum.wavelength_cm <= passband.wavelength_cm[-1])],
        passband.wavelength_cm)
    energy = replace(
        passband, wavelength_cm=wave,
        response=np.interp(wave, passband.wavelength_cm, passband.response)
        * passband.wavelength_cm[-1] / wave,
        a_lambda_over_av=np.zeros_like(wave),
        response_source="diagnostic energy weighting of the same optical curve")
    _, _, variant_mu, intensity, _ = integrate_photon_intensity(
        spectrum, energy, 0., require_nonnegative_quadratic=False)
    variant = intensity / intensity[-1]
    diagnostic["energy_weighting"] = {
        "raw": _profile_metrics(mu_marcs, np.interp(mu_marcs, variant_mu, variant), marc),
        # Hold the baseline edge fixed to isolate the response effect.
        "same_edge": _profile_metrics(
            mu_marcs, _at_radius(variant_mu[1:], variant[1:], mu_marcs, edge["radius"]), marc),
        "new_edge": _r_method_edge(variant_mu[1:], variant[1:]),
    }
    if np.isfinite(interpolated_marcs).all():
        diagnostic["marcs_parameter_interpolation"] = {
            "status": "inside_convex_hull",
            "interpolated_minus_nearest": _profile_metrics(mu_marcs, interpolated_marcs, marc),
            "raw_against_interpolated": _profile_metrics(
                mu_marcs, np.interp(mu_marcs, mu, profile), interpolated_marcs),
            "r_method_against_interpolated": _profile_metrics(mu_marcs, corrected, interpolated_marcs),
        }
    else:
        diagnostic["marcs_parameter_interpolation"] = {"status": "outside_convex_hull; no extrapolation"}
    # The 12-node raster path can smear a sharp native limb over an entire interval.
    native_cut_mu = np.r_[mu_marcs[0], mu[mu > mu_marcs[0]]]
    native_cut_profile = np.interp(native_cut_mu, mu, profile)
    edge_mu = np.sqrt(1 - edge["radius"]**2)
    inner_mu = native_mu[native_mu > edge_mu]
    corrected_mu = np.r_[0., np.sqrt((inner_mu**2 - edge_mu**2) / edge["radius"]**2)]
    corrected_mu[-1] = 1.  # The disc centre is exact, independent of roundoff in the radius map.
    corrected_profile = _at_radius(
        native_mu, native_profile, corrected_mu, edge["radius"])
    corrected_cut_mu = np.r_[mu_marcs[0], corrected_mu[corrected_mu > mu_marcs[0]]]
    corrected_cut_profile = np.interp(corrected_cut_mu, corrected_mu, corrected_profile)
    depth_controls = {}
    for b in IMPACT_PARAMETERS:
        profiles = {
            "marcs_12": (mu_marcs, marc),
            "raw_12": (mu_marcs, np.interp(mu_marcs, mu, profile)),
            "raw_native": (native_cut_mu, native_cut_profile),
            "r_method_12": (mu_marcs, corrected),
            "r_method_native": (corrected_cut_mu, corrected_cut_profile),
        }
        row = {name: _polar_depth(m, p, impact_parameter=b) for name, (m, p) in profiles.items()}
        row["native_order128_relative_change"] = (
            _polar_depth(native_cut_mu, native_cut_profile, impact_parameter=b, order=128)
            / row["raw_native"] - 1)
        row["r_method_order128_relative_change"] = (
            _polar_depth(corrected_cut_mu, corrected_cut_profile, impact_parameter=b, order=128)
            / row["r_method_native"] - 1)
        row["raw_native_relative_to_marcs"] = row["raw_native"] / row["marcs_12"] - 1
        row["r_method_native_relative_to_marcs"] = row["r_method_native"] / row["marcs_12"] - 1
        row["raw_12_relative_to_marcs"] = row["raw_12"] / row["marcs_12"] - 1
        row["r_method_12_relative_to_marcs"] = row["r_method_12"] / row["marcs_12"] - 1
        row["raw_raster_relative_to_polar"] = (
            _transit_depth(mu_marcs, profiles["raw_12"][1], impact_parameter=b) / row["raw_12"] - 1)
        depth_controls[str(b)] = row
    diagnostic["depth_controls"] = depth_controls
    return diagnostic


@pytest.mark.parametrize("coefficients", [(1., 0., 0., 0.), (.4, .6, .2, .05)])
def test_ncam_integration_has_analytic_photon_and_angular_normalization(coefficients):
    """Known chromatic linear law through the actual packaged NCAM loader/kernel.

    I_lambda=(a+b*mu)/lambda+(c+d*mu)/lambda_ref, no extinction.
    For piecewise-linear R, integral R and integral lambda*R are analytic.
    The latter's trapezoid error decreases quadratically; 2e-7 relative in
    flux/profile is a numerical check, not the unchanged MARCS tolerance.
    """
    from astropy import constants as const
    from trex.acquisition.passbands import packaged_plato_transmission
    from trex.stellar_light.passband import integrate_photon_intensity
    from trex.stellar_light.release import AngularSpectrum

    band = packaged_plato_transmission(
        "NCAM", "as designed camera effective transmission").to_photon_passband(
            extinction_source="zero extinction: analytic NCAM control")
    a, b, c, d = coefficients
    left, right = band.wavelength_cm[:-1], band.wavelength_cm[1:]
    rl, rr = band.response[:-1], band.response[1:]
    width = right - left
    s0 = np.sum(width * (rl + rr) / 2)
    s1 = np.sum(width * ((2*left + right)*rl + (left + 2*right)*rr) / 6)
    hc = const.h.cgs.value * const.c.cgs.value
    reference_wave = band.wavelength_cm[-1]
    centre_a = (a*s0 + c*s1/reference_wave) / hc
    slope_b = (b*s0 + d*s1/reference_wave) / hc
    native_mu = np.array([1e-4, .01, .05, .1, .3, .6, 1.])
    expected = centre_a + slope_b*native_mu
    envelope_spectrum = AngularSpectrum(
        5500., 4.5, 0., band.wavelength_cm, native_mu,
        ((a + b*native_mu[:, None]) / band.wavelength_cm
         + (c + d*native_mu[:, None]) / reference_wave), "analytic envelope control")
    envelope = _spectral_shape_envelope(envelope_spectrum, band, native_mu, expected / expected[-1])
    assert envelope["max_distance_outside_envelope"] < 2e-15
    # I=0 is inserted at mu=0: the missing constant-radiance sliver is A*mu_min²/6.
    expected_flux = 2*np.pi*(centre_a/2 + slope_b/3 - centre_a*native_mu[0]**2/6)
    errors = []
    for subdivisions in (128, 256):
        wave = np.unique(np.concatenate([
            np.linspace(lo, hi, subdivisions + 1) for lo, hi in zip(left, right)]))
        intensity = ((a + b*native_mu[:, None]) / wave
                     + (c + d*native_mu[:, None]) / reference_wave)
        spectrum = AngularSpectrum(5500., 4.5, 0., wave, native_mu, intensity, "analytic control")
        flux, _, rays, profile, _ = integrate_photon_intensity(
            spectrum, band, 0., require_nonnegative_quadratic=False)
        error = abs(flux / expected_flux - 1)
        errors.append(error)
        np.testing.assert_array_equal(rays[1:], native_mu)
        np.testing.assert_allclose(profile[1:], expected, rtol=2e-7, atol=0)
        np.testing.assert_allclose(
            profile[1:] / profile[-1], expected / expected[-1], rtol=2e-7, atol=0)
        assert error < 2e-7
    assert errors[1] <= .26*errors[0] + 1e-13
    # Preserve energy while doubling wavelengths: twice as many half-energy photons.
    doubled = replace(spectrum, wavelength_cm=2*wave, intensity=intensity/2)
    doubled_band = replace(band, wavelength_cm=2*band.wavelength_cm)
    scaled_flux, _, _, scaled_profile, _ = integrate_photon_intensity(
        doubled, doubled_band, 0., require_nonnegative_quadratic=False)
    assert scaled_flux / flux == pytest.approx(2., rel=2e-13)
    np.testing.assert_allclose(scaled_profile[1:] / profile[1:], 2., rtol=2e-13)
    report = {
        "coefficients": coefficients, "relative_flux_errors_128_256": errors,
        "max_relative_intensity_error": float(np.max(np.abs(profile[1:] / expected - 1))),
        "wavelength_rescaling_flux_ratio": float(scaled_flux / flux)}
    print("NCAM-ANALYTIC " + json.dumps(report))
    if os.environ.get("EXONYM_TEST_REPORT_DIR"):
        Path(os.environ["EXONYM_TEST_REPORT_DIR"], f"ncam-analytic-{a}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")


def test_ncam_radius_diagnostics_recover_known_mapping_and_uniform_depth():
    """Independent affine-profile mapping and uniform-disc area controls."""
    radial_nodes = np.array([0., .2, .5, .9, .96, .97, .99, .995, .99999])
    mu = np.sqrt(1 - radial_nodes[::-1]**2)
    profile = np.array([1., .95, .8, .6, .5, .5, 0., 0., 0.])[::-1]
    edge = _r_method_edge(mu, profile)
    assert edge["radius"] == pytest.approx(.98, abs=1e-12)
    np.testing.assert_allclose(edge["radius_interval"], [.97, .99], atol=1e-12, rtol=0)
    assert not edge["peak_at_boundary"]
    target = np.array([.01, .2, .5, 1.])
    mapped = _at_radius(mu, .25 + .75*mu, target, .98)
    expected = .25 + .75*np.sqrt(1 - .98**2*(1 - target**2))
    np.testing.assert_allclose(mapped, expected, atol=2e-15, rtol=0)
    with pytest.raises(ValueError, match="native positive-mu"):
        _r_method_edge(np.r_[0., mu], np.r_[0., profile])
    with pytest.raises(ValueError, match="no radial intensity drop"):
        _r_method_edge(mu, np.ones_like(mu))
    for impact in IMPACT_PARAMETERS:
        # Planet wholly inside the mu>=mu_min disc: blocked/total = k²/(1-mu_min²).
        uniform = np.ones_like(target)
        depth = _polar_depth(target, uniform, impact_parameter=impact)
        assert depth == pytest.approx(RADIUS_RATIO**2 / (1 - target[0]**2), rel=2e-7)


def _transit_depth(mu_nodes, profile, *, impact_parameter):
    """Relative blocked flux on a uniform Cartesian grid, excluding the unmodelled limb.

    Both profiles are evaluated on the same truncated stellar disc
    (mu >= the smallest published node); the very limb is not extrapolated.
    """
    axis = np.linspace(-1.0, 1.0, RX)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    mu = np.sqrt(np.clip(1.0 - x*x - y*y, 0.0, None))
    disc = (x*x + y*y <= 1.0) & (mu >= mu_nodes[0])
    light = np.interp(mu[disc], mu_nodes, profile)
    planet = ((x[disc] - impact_parameter)**2 + y[disc]**2) <= RADIUS_RATIO**2
    return float(light[planet].sum() / light.sum())


def _report_release_structure(archive):
    """Record the release layout before any parsing assumption is applied.

    The first execution (2026-11-16) showed one ``PLATO_NCAM_ints.dat`` for the
    whole grid rather than per-model files, and the release note describes a
    randomly distributed 1997-model grid (Morello et al. 2022, section 2). The
    report lets the parser follow the provider's documented layout.
    """
    directory = os.environ.get("EXONYM_TEST_REPORT_DIR")
    if not directory:
        return
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        heads = {}
        for info in infos:
            name = info.filename
            lowered = name.lower()
            if lowered.endswith(("readme", "readme.txt", "readme.md")) or "ncam" in lowered:
                text = bundle.read(name)[:6000].decode("utf-8", errors="replace")
                heads[name] = text.splitlines()[:40]
    Path(directory, "morello-release-structure.json").write_text(json.dumps({
        "source": ZENODO_URL,
        "members": [{"name": info.filename, "bytes": info.file_size} for info in infos][:200],
        "heads": heads,
    }, indent=2), encoding="utf-8")


@NETWORK_CROSSCHECK
def test_morello_ncam_profiles_match_the_prepared_phoenix_profile(tmp_path):
    from scipy.interpolate import LinearNDInterpolator
    from trex.stellar_light.release import LOGG_AXIS, TEFF_AXIS

    started = perf_counter()
    archive = _download(ZENODO_URL, tmp_path / "PLATO_WP122400_release_21nov2022.zip")
    assert hashlib.md5(archive.read_bytes()).hexdigest() == ZENODO_MD5
    _report_release_structure(archive)
    marcs = _marcs_profiles(archive)
    coordinates = np.asarray(list(marcs))
    marcs_interpolator = LinearNDInterpolator(
        coordinates / [MATCH_TEFF_K, MATCH_LOGG_DEX],
        np.asarray([value[1] for value in marcs.values()]))
    matched = []
    for teff in TEFF_AXIS:
        for logg in LOGG_AXIS:
            if 4500.0 <= teff <= 7000.0 and 3.0 <= logg <= 5.0:
                key = _nearest_marcs(marcs, float(teff), float(logg))
                if key is not None:
                    matched.append((float(teff), float(logg), key))
    if not matched:
        raise AssertionError(
            "no MARCS model within {0} K / {1} dex of a PHOENIX node in 4500-7000 K, log g 3.0-5.0".format(
                MATCH_TEFF_K, MATCH_LOGG_DEX))
    step = max(1, math.ceil(len(matched) / MAX_NODES))
    selected = matched[::step][:MAX_NODES]

    records, residuals = [], []
    for teff, logg, marcs_key in selected:
        node_started = perf_counter()
        mu_marcs, marc = marcs[marcs_key]
        mu_phoenix, phoenix, spectrum, passband = _phoenix_profile(teff, logg, tmp_path / "phoenix")
        at_nodes = np.interp(mu_marcs, mu_phoenix, phoenix)
        residual = at_nodes - marc
        residuals.append(residual)
        depth = {}
        for impact in IMPACT_PARAMETERS:
            marc_depth = _transit_depth(mu_marcs, marc, impact_parameter=impact)
            phoenix_depth = _transit_depth(mu_marcs, at_nodes, impact_parameter=impact)
            depth[str(impact)] = {
                "marcs": marc_depth,
                "phoenix": phoenix_depth,
                "relative": (phoenix_depth - marc_depth) / marc_depth,
            }
        records.append({"teff_k": teff, "logg_cgs": logg,
                        "marcs_teff_k": marcs_key[0], "marcs_logg_cgs": marcs_key[1],
                        "max_abs": float(np.max(np.abs(residual))),
                        "rms": float(np.sqrt(np.mean(residual**2))),
                        # Diagnostic only (no limit): where on the disc the
                        # difference sits, and the same metric without the limb
                        # nodes, to separate a spherical-PHOENIX vs plane-parallel
                        # MARCS limb from a passband or integration difference.
                        "mu": mu_marcs.tolist(),
                        "residual_by_mu": residual.tolist(),
                        "rms_mu_ge_0_1": float(np.sqrt(np.mean(residual[mu_marcs >= 0.1]**2))),
                        "max_abs_mu_ge_0_1": float(np.max(np.abs(residual[mu_marcs >= 0.1]))),
                         "depth": depth,
                         "diagnostics": _cause_diagnostics(
                             spectrum, passband, mu_phoenix, phoenix, mu_marcs, marc,
                             np.asarray(marcs_interpolator(
                                 [[teff / MATCH_TEFF_K, logg / MATCH_LOGG_DEX]]))[0]),
                         "seconds": perf_counter() - node_started})

    stacked = np.concatenate(residuals)
    rms = float(np.sqrt(np.mean(stacked**2)))
    peak = float(np.max(np.abs(stacked)))
    worst_depth = max(abs(row["depth"][key]["relative"])
                      for row in records for key in row["depth"])
    report = {"source": "Zenodo 10.5281/zenodo.7339706 v1.1",
              "source_url": ZENODO_URL,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "passband": "PLATO NCAM as designed camera effective transmission",
              "diagnostic_status": "exploratory; original predeclared assertions remain authoritative",
              "seconds": perf_counter() - started,
              "nodes": records, "rms": rms, "max_abs": peak,
              "max_abs_depth_relative": worst_depth,
              "tolerance": {"rms": 0.02, "max_abs": 0.05, "depth_relative": 0.01}}
    print("MORELLO-NCAM-CROSSCHECK " + json.dumps(report, sort_keys=True))
    if os.environ.get("EXONYM_TEST_REPORT_DIR"):
        Path(os.environ["EXONYM_TEST_REPORT_DIR"], "morello-ncam-crosscheck.json").write_text(
            json.dumps({**report, "matched_phoenix_nodes": len(matched),
                        "match_window": {"teff_k": MATCH_TEFF_K, "logg_dex": MATCH_LOGG_DEX}}, indent=2),
            encoding="utf-8")
    assert rms <= 0.02, report
    assert peak <= 0.05, report
    assert worst_depth <= 0.01, report


@NETWORK_CROSSCHECK
def test_r_method_anchor_removes_the_ncam_limb_mis_anchoring_signature(tmp_path):
    """Regression check that production r-method anchoring removes the limb mis-anchoring.

    This is NOT a PLATO limb-darkening agreement or validation check. Its bounds
    were chosen after the exploratory r-method residuals were seen, to separate the
    mis-anchoring signature (node RMS >= 0.09, max 0.474) from the residual
    cross-model difference; agreement is judged only by the original test above
    at RMS 0.02 / max 0.05 / 1% depth, and any PLATO LDC claim stays deferred
    until that test passes.

    Declared 2026-11-17, before any result of THIS comparison is seen. The
    numbers quoted in this docstring come from the retained exploratory
    diagnosis, which used an equivalent test-local radius
    mapping, not the production function checked here; the original
    predeclared test above and its recorded failure stay authoritative.

    Compared objects are two different atmosphere models through two different
    N-CAM responses: TREX PHOENIX-ACES-AGSS-COND-2011 LTE (Asplund 2009
    abundances) through the packaged PlatoSim3 DLR-TN-0113 v2 as-designed
    camera effective transmission (optics only, no detector QE),
    versus Morello et al. (2022) MARCS-NLTE (Magg 2022 abundances) through
    the unpublished PLATO laboratory responses including QE. The Zenodo
    release carries only intensity/LDC tables and the public ExoTETHyS
    ships no PLATO passband, so no official N-CAM curve with QE exists and
    the response and atmosphere differences cannot be separated.

    TREX side applies the production convention end to end: the profile of
    integrate_photon_intensity re-anchored by
    trex.stellar_light.passband.reanchor_r_method_rays, the same function
    grid compilation applies, removing the spherical-tangent
    limb/radius anchoring that produced the original failure (full-disc RMS
    0.137, max 0.474).

    Predeclared tolerances, over the same node selection rule (nearest MARCS
    model within 10 K / 0.02 dex of a PHOENIX node in 4500-7000 K, log g
    3.0-5.0, first of every ceil(n/6)th matched node, at most 6 nodes) and
    the same 12 release mu nodes:
    - normalized-intensity RMS <= 0.04 over all compared nodes and mu values;
    - maximum absolute difference <= 0.15;
    - transit depth for Rp/Rs = 0.1 differing by <= 2.5% relative at impact
      parameters 0.0 and 0.6.
    These are cross-model bounds: the retained diagnosis attributes a smooth
    +0.02-0.03 mid-mu excess and -1.3% to -1.8% depth difference to the
    atmosphere/response pair, and the bounds admit that irreducible
    difference (RMS 0.017-0.025, max up to 0.071, depth up to 1.8% were
    observed under the equivalent mapping) with bounded margin, while any
    remaining limb/radius mis-anchoring still fails them (it produced
    node RMS >= 0.09 and max 0.474). If the residual exceeds these
    tolerances, the remaining cause is reported; the bounds are not moved.
    """
    from trex.stellar_light.passband import reanchor_r_method_rays
    from trex.stellar_light.release import LOGG_AXIS, TEFF_AXIS

    started = perf_counter()
    archive = _download(ZENODO_URL, tmp_path / "PLATO_WP122400_release_21nov2022.zip")
    assert hashlib.md5(archive.read_bytes()).hexdigest() == ZENODO_MD5
    marcs = _marcs_profiles(archive)
    matched = []
    for teff in TEFF_AXIS:
        for logg in LOGG_AXIS:
            if 4500.0 <= teff <= 7000.0 and 3.0 <= logg <= 5.0:
                key = _nearest_marcs(marcs, float(teff), float(logg))
                if key is not None:
                    matched.append((float(teff), float(logg), key))
    if not matched:
        raise AssertionError(
            "no MARCS model within {0} K / {1} dex of a PHOENIX node in 4500-7000 K, log g 3.0-5.0".format(
                MATCH_TEFF_K, MATCH_LOGG_DEX))
    step = max(1, math.ceil(len(matched) / MAX_NODES))
    selected = matched[::step][:MAX_NODES]

    records, residuals = [], []
    for teff, logg, marcs_key in selected:
        mu_marcs, marc = marcs[marcs_key]
        mu_phoenix, phoenix, _spectrum, _passband = _phoenix_profile(teff, logg, tmp_path / "phoenix")
        anchored, edge = reanchor_r_method_rays(mu_phoenix, phoenix)
        at_nodes = np.interp(mu_marcs, mu_phoenix, anchored)
        residual = at_nodes - marc
        residuals.append(residual)
        depth = {}
        for impact in IMPACT_PARAMETERS:
            marc_depth = _transit_depth(mu_marcs, marc, impact_parameter=impact)
            phoenix_depth = _transit_depth(mu_marcs, at_nodes, impact_parameter=impact)
            depth[str(impact)] = {
                "marcs": marc_depth,
                "phoenix": phoenix_depth,
                "relative": (phoenix_depth - marc_depth) / marc_depth,
            }
        records.append({"teff_k": teff, "logg_cgs": logg,
                        "marcs_teff_k": marcs_key[0], "marcs_logg_cgs": marcs_key[1],
                        "r_method_edge": edge,
                        "raw_max_abs": float(np.max(np.abs(
                            np.interp(mu_marcs, mu_phoenix, phoenix) - marc))),
                        "max_abs": float(np.max(np.abs(residual))),
                        "rms": float(np.sqrt(np.mean(residual**2))),
                        "mu": mu_marcs.tolist(),
                        "residual_by_mu": residual.tolist(),
                        "depth": depth})

    stacked = np.concatenate(residuals)
    rms = float(np.sqrt(np.mean(stacked**2)))
    peak = float(np.max(np.abs(stacked)))
    worst_depth = max(abs(row["depth"][key]["relative"])
                      for row in records for key in row["depth"])
    report = {"source": "Zenodo 10.5281/zenodo.7339706 v1.1",
              "source_url": ZENODO_URL,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "passband": "PLATO NCAM as designed camera effective transmission",
              "convention": "production r-method re-anchoring",
              "seconds": perf_counter() - started,
              "nodes": records, "rms": rms, "max_abs": peak,
              "max_abs_depth_relative": worst_depth,
              "tolerance": {"rms": 0.04, "max_abs": 0.15, "depth_relative": 0.025}}
    print("MORELLO-NCAM-RMETHOD-CROSSCHECK " + json.dumps(report, sort_keys=True))
    if os.environ.get("EXONYM_TEST_REPORT_DIR"):
        Path(os.environ["EXONYM_TEST_REPORT_DIR"], "morello-ncam-rmethod-crosscheck.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    assert rms <= 0.04, report
    assert peak <= 0.15, report
    assert worst_depth <= 0.025, report
