"""Photon weighting and flux-conserving quadratic reduction of angular spectra.

Chen et al. (2019), A&A 632, A105, DOI 10.1051/0004-6361/201936612,
photon-weighted magnitude and bolometric correction (``eq_Mag`` and
``eq_BC_photon`` in the source); Parviainen & Aigrain (2015),
MNRAS 453, 3821, DOI 10.1093/mnras/stv1857.
No spectral extinction law is guessed: a tabulated A_lambda/A_V and its source
are required. The same attenuated photon profile determines F and u1/u2.

Transit-radius convention: spherical SPECINT rays are anchored at the model
reference circle, not at the transit-visible stellar radius, so grid
compilation re-anchors them to the r-method limb before any consumer sees
them (``reanchor_r_method_rays``); this function set only integrates whatever
ray set it is given.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import resources
import math
from typing import Tuple

from astropy import constants as const
import numpy as np

from trex.foundation.physical_constants import SECOND_RADIATION_CONSTANT_M_K


@dataclass(frozen=True)
class PhotonPassband:
    """Vacuum wavelength [cm], photon response, A_lambda/A_V [dimensionless]."""

    wavelength_cm: np.ndarray
    response: np.ndarray
    a_lambda_over_av: np.ndarray
    response_source: str
    extinction_source: str
    # Instrument-qualified response identity (facility/instrument.filter). The
    # historical empty default keeps the TESS constructed passbands readable;
    # multi-band preparation always supplies it (see stellar_light.response).
    response_id: str = ""

    def __post_init__(self):
        arrays = [np.asarray(v) for v in (
            self.wavelength_cm, self.response, self.a_lambda_over_av)]
        if (arrays[0].ndim != 1 or arrays[0].size < 2
                or any(a.dtype.kind not in "iuf" or a.shape != arrays[0].shape
                       or not np.isfinite(a).all() for a in arrays)
                or np.any(arrays[0][1:] <= arrays[0][:-1]) or np.any(arrays[0] <= 0)
                or any(np.any(a < 0) for a in arrays[1:]) or not np.any(arrays[1] > 0)
                or any(not isinstance(s, str) or not s.strip()
                       for s in (self.response_source, self.extinction_source))
                or not isinstance(self.response_id, str)):
            raise ValueError("invalid photon passband/extinction tabulation or missing source")
        for name, value in zip(("wavelength_cm", "response", "a_lambda_over_av"), arrays):
            value = value.astype(float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def input_id(self) -> str:
        """Digest of the exact kernel and extinction tables for wrong-input binding.

        Prepared grids and resumes compare this value; it detects a substituted
        band or law, not scientific validity (AGENTS.md: a digest identifies bytes).
        """
        digest = hashlib.sha256()
        for array in (self.wavelength_cm, self.response, self.a_lambda_over_av):
            digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        for text in (self.response_id, self.response_source, self.extinction_source):
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()


def angular_moment(mu, intensity):
    """Integral I(mu)*mu*dmu, exact for the piecewise-linear ray profile."""
    a, b = mu[:-1], mu[1:]
    # Integrate the two linear hat functions times mu analytically [sr-free moment].
    return np.sum((b-a) * ((2*a+b)*intensity[:-1] + (a+2*b)*intensity[1:]) / 6)


# Real spherical PHOENIX nodes of the PLATO domain emit ~3e-4 of their photons
# beyond the r-method edge (measured on the retained 2026-11-16 crosscheck
# records); a profile whose "steepest drop" leaves more than this bound of the
# light outside has no resolved limb at all (smooth plane-parallel-like laws,
# e.g. a power of mu, peak mid-disc) and is refused rather than re-anchored.
R_METHOD_MAX_BEYOND_EDGE_PHOTON_FRACTION = 0.02


def r_method_edge(mu, intensity):
    """Steepest-radial-drop limb radius of one complete native ray profile.

    Espinoza & Jordan (2015) r-method, as described by Claret (2017), A&A 600,
    A30, DOI 10.1051/0004-6361/201629705, Sect. 3, and adopted for direct
    transit-profile use by Parviainen & Aigrain (2015), MNRAS 453, 3821,
    DOI 10.1093/mnras/stv1857 (renormalize in z). Spherical SPECINT mu values
    are anchored at the model reference circle (Husser et al. 2013 spherical
    64-layer atmospheres), which lies outside the transit-visible edge; the
    radius-ratio and separation consumers of the occultation kernel are given
    in stellar-radii units, so the visible edge is the radius their unit disc
    must end at.

    Input rays span mu=0..1 with the artificial zero-intensity tangent at
    mu=0; that ray is excluded from the edge search but contributes to the
    beyond-edge photon fraction. A native interval is the discrete edge (the
    derivative maximum lies inside it); its radius midpoint is returned with
    both endpoints. An edge in the outermost native interval is accepted
    (a razor-thin drop-off can be narrower than one interval); an edge
    adjacent to the disc centre, a profile with no outward intensity drop, or
    one leaving more than
    ``R_METHOD_MAX_BEYOND_EDGE_PHOTON_FRACTION`` of the photons beyond the
    edge has no spherical limb and raises ``ValueError``.

    Returns a dict with ``radius``, ``radius_interval`` (ascending),
    ``mu_interval``, ``edge_intensity_over_centre`` and
    ``beyond_edge_photon_fraction``; all dimensionless.
    """
    mu, intensity = np.asarray(mu), np.asarray(intensity)
    if (mu.ndim != 1 or mu.size < 4 or intensity.shape != mu.shape
            or mu.dtype.kind not in "iuf" or intensity.dtype.kind not in "iuf"
            or not np.isfinite(mu).all() or not np.isfinite(intensity).all()
            or np.any(mu[1:] <= mu[:-1]) or mu[0] != 0. or mu[-1] != 1.
            or np.any(intensity < 0) or intensity[-1] <= 0):
        raise ValueError("r-method edge requires ordered complete nonnegative rays")
    radius = np.sqrt(1.0 - mu[1:]**2)
    drop = -np.diff(intensity[1:]) / np.diff(radius)
    if not np.any(drop > 0):
        raise ValueError("r-method edge requires an outward radial intensity drop")
    index = int(np.argmax(drop))
    if index == drop.size - 1:
        # Steepest change adjacent to the disc centre: a smooth law, no limb.
        raise ValueError("r-method edge requires the steepest drop at the limb, not the disc centre")
    edge_radius = 0.5 * (radius[index] + radius[index + 1])
    mu_edge = float(np.sqrt(1.0 - edge_radius**2))
    # Beyond the edge: the artificial tangent ramp plus the outer native rays.
    beyond_mu = np.r_[0.0, mu[1:1 + index + 1], mu_edge]
    beyond_intensity = np.r_[0.0, intensity[1:1 + index + 1],
                             np.interp(mu_edge, mu, intensity)]
    total = angular_moment(mu, intensity)
    beyond = angular_moment(beyond_mu, beyond_intensity)
    if total <= 0 or not 0 <= beyond <= total:
        raise ValueError("r-method edge requires a positive disc moment")
    if beyond / total > R_METHOD_MAX_BEYOND_EDGE_PHOTON_FRACTION:
        raise ValueError(
            "r-method edge leaves {0:.3f} of the photons beyond the limb; the "
            "profile has no resolved spherical drop-off".format(beyond / total))
    return {
        "radius": float(edge_radius),
        "radius_interval": [float(min(radius[index], radius[index + 1])),
                            float(max(radius[index], radius[index + 1]))],
        "mu_interval": [float(mu[1 + index]), float(mu[2 + index])],
        "edge_intensity_over_centre": float(np.interp(mu_edge, mu, intensity) / intensity[-1]),
        "beyond_edge_photon_fraction": float(beyond / total),
    }


def reanchor_r_method_rays(mu, intensity):
    """Re-anchor native spherical rays so the unit disc ends at the r-method edge.

    The returned intensity keeps the input mu grid and ray count exactly (grid
    storage, cross-node ray compatibility and extinction-knot mu sharing are
    unchanged); each value is the input profile linearly interpolated at the
    interior position ``sqrt(1 - R_edge^2 (1 - mu_i^2))``, and the mu=0 ray
    carries the edge intensity instead of the artificial tangent zero. The
    centre ray mu=1 is exact. Occultation consumers then apply their
    stellar-radii radius ratio and separation to the transit-visible disc;
    the photon fraction emitted beyond the edge (<= ~3e-4 for retained
    spherical nodes) is excluded from the re-anchored disc moment.

    Returns ``(anchored_intensity, edge_record)``; raises ``ValueError``
    through :func:`r_method_edge` when no spherical limb is resolved.
    """
    edge = r_method_edge(mu, intensity)
    position = np.sqrt(1.0 - edge["radius"]**2 * (1.0 - np.asarray(mu)**2))
    return np.interp(position, mu, intensity), edge


def quadratic_nonnegative(values, *, xp=np):
    """Minimum of absolute I=a+b*x+c*x² on x=1-mu in [0,1]."""
    a, b, c = values[..., 1], values[..., 2], values[..., 3]
    vertex = -b/xp.where(c != 0, 2*c, 1.)
    minimum = xp.minimum(a, a+b+c)
    minimum = xp.where((c > 0) & (vertex > 0) & (vertex < 1),
                       xp.minimum(minimum, a+b*vertex+c*vertex**2), minimum)
    return minimum >= 0


def quadratic_reduction(mu, intensity, *, require_nonnegative=True):
    """Return absolute polynomial (a0,a1,a2), errors; I=a0+a1*x+a2*x².

    x=1-mu. Centre intensity and the exact disk moment are constrained, not
    merely fitted. Errors include the spherical limb. This function never
    discards rays or renormalizes them itself: the transit-radius convention
    (r-method re-anchoring to the visible limb) is applied by the caller that
    owns the compiled rays, before this reduction; a profile on the raw
    spherical tangent grid may still be passed for whole-atmosphere moments.
    This is an explicitly approximate angular law, not an exact representation
    of PHOENIX.
    """
    mu, intensity = np.asarray(mu), np.asarray(intensity)
    if (mu.dtype.kind not in "iuf" or intensity.dtype.kind not in "iuf"
            or mu.ndim != 1 or mu.size < 2 or intensity.shape != mu.shape
            or not np.isfinite(mu).all() or not np.isfinite(intensity).all()
            or np.any(mu[1:] <= mu[:-1]) or np.any(intensity < 0)):
        raise ValueError("quadratic reduction requires finite ordered rays and aligned nonnegative intensities")
    if mu[0] != 0. or mu[-1] != 1.:
        raise ValueError("quadratic reduction integrates the whole disc: rays must span mu=0..1")
    centre = float(intensity[-1])
    moment = angular_moment(mu, intensity)
    if centre <= 0 or moment <= 0:
        raise ValueError("atmosphere has nonpositive photon intensity/flux")
    # Two Gauss points per ray interval integrate x^n*(1-I/Ic) exactly (degree <=3).
    a, b = mu[:-1], mu[1:]
    sample = (a+b)[:, None]/2+(b-a)[:, None]/2*np.array([-1., 1.])/np.sqrt(3.)
    profile = np.interp(sample, mu, intensity) / centre
    x = 1-sample
    rhs = np.array([np.sum((b-a)/2*np.sum(x**n*(1-profile), axis=1)) for n in (1, 2)])
    # Analytic disk moment: integral mu*(1-u1*x-u2*x²) = 1/2-u1/6-u2/12.
    constraint = np.array([1/6, 1/12])
    # Exact continuous least-squares Gram matrix, integral x^(i+j) dmu [dimensionless].
    system = np.block([[np.array([[1/3, 1/4], [1/4, 1/5]]), constraint[:, None]],
                       [constraint[None, :], np.zeros((1, 1))]])
    u = np.linalg.solve(system, np.r_[rhs, .5-moment/centre])[:2]
    # Nonnegative quadratic on the entire interval, including its interior extremum.
    extrema = [0., 1.]
    if u[1] != 0 and 0 < -u[0]/(2*u[1]) < 1:
        extrema.append(-u[0]/(2*u[1]))
    nonnegative = bool(np.min(1-u[0]*np.array(extrema)-u[1]*np.square(extrema)) >= 0)
    if require_nonnegative and not nonnegative:
        raise ValueError("flux-conserving quadratic law has negative intensity")
    # Three-point Gaussian quadrature integrates the squared quadratic residual exactly.
    points, weights = np.polynomial.legendre.leggauss(3)
    sample = (a+b)[:, None]/2+(b-a)[:, None]/2*points
    profile = np.interp(sample, mu, intensity)/centre
    x = 1-sample
    residual = 1-u[0]*x-u[1]*x*x-profile
    rms = np.sqrt(np.sum((b-a)/2*np.sum(residual**2*weights, axis=1)))
    check = mu
    if u[1] != 0:
        slope = np.diff(intensity/centre)/(b-a)
        stationary = 1+(u[0]-slope)/(2*u[1])
        check = np.r_[mu, stationary[(stationary > a) & (stationary < b)]]
    x = 1-check
    maximum = np.max(abs(1-u[0]*x-u[1]*x*x-np.interp(check, mu, intensity)/centre))
    errors = {
        "quadratic_nonnegative": nonnegative,
        "profile_rms": float(rms),
        "profile_max_abs": float(maximum),
        "disk_flux_relative_error": float((.5-u @ constraint)/(moment/centre)-1),
    }
    return np.r_[centre, -centre*u], errors


def integrate_photon_intensity(spectrum, bandpass, av_mag, *, require_nonnegative_quadratic=True):
    """Return flux [photons s^-1 cm^-2], quadratic, raw profile and fit errors.

    A_V [mag] attenuates the spectrum once. The returned surface factor is
    already attenuated; it must not be re-extinguished after geometric scaling.
    The returned rays are the native spherical SPECINT grid; grid compilation
    re-anchors them to the r-method limb before storage (see
    :func:`reanchor_r_method_rays`), so a caller needing the transit-radius
    convention must apply that anchoring to this output itself.
    """
    av = np.asarray(av_mag)
    if av.ndim != 0 or av.dtype.kind not in "iuf" or not np.isfinite(av) or av < 0:
        raise ValueError("A_V must be finite and nonnegative")
    wave_input, mu_input, intensity_input = (np.asarray(v) for v in (
        spectrum.wavelength_cm, spectrum.mu, spectrum.intensity))
    if (any(a.dtype.kind not in "iuf" or not np.isfinite(a).all()
            for a in (wave_input, mu_input, intensity_input))
            or wave_input.ndim != 1 or wave_input.size < 2 or np.any(wave_input <= 0)
            or np.any(wave_input[1:] <= wave_input[:-1]) or mu_input.ndim != 1 or not mu_input.size
            or mu_input[0] <= 0 or mu_input[-1] != 1 or np.any(mu_input[1:] <= mu_input[:-1])
            or intensity_input.shape != (mu_input.size, wave_input.size) or np.any(intensity_input < 0)):
        raise ValueError("photon integration requires a finite ordered angular spectrum")
    if (spectrum.wavelength_cm[0] > bandpass.wavelength_cm[0]
            or spectrum.wavelength_cm[-1] < bandpass.wavelength_cm[-1]):
        raise ValueError("spectrum does not cover the complete passband")
    # Keep every native spectral sample and insert response knots/boundaries [cm].
    lo, hi = bandpass.wavelength_cm[[0, -1]]
    wave = np.union1d(spectrum.wavelength_cm[(spectrum.wavelength_cm >= lo)
                                          & (spectrum.wavelength_cm <= hi)], bandpass.wavelength_cm)
    response = np.interp(wave, bandpass.wavelength_cm, bandpass.response)
    extinction = np.interp(wave, bandpass.wavelength_cm, bandpass.a_lambda_over_av)
    # Chen eq_Mag photon response: lambda/(hc) * R * 10^(-0.4 A_lambda), CGS.
    weight = wave/(const.h.cgs.value*const.c.cgs.value)*response*10**(-.4*av_mag*extinction)
    # Identical per-ray trapezoids share the SAME wavelength differences; retain every sample/ray.
    sampled = np.array([np.interp(wave, spectrum.wavelength_cm, ray) for ray in spectrum.intensity])
    profile = np.trapezoid(sampled*weight, wave, axis=-1)
    # The spherical tangent has zero path length at mu=0; the release starts just inside it.
    mu, profile = np.r_[0., spectrum.mu], np.r_[0., profile]
    flux = 2*np.pi*angular_moment(mu, profile)
    polynomial, errors = quadratic_reduction(mu, profile, require_nonnegative=require_nonnegative_quadratic)
    return flux, polynomial, mu, profile, errors


_TESS_RESPONSE_FILENAME = "tess-response-function-v2.0.csv"
_TESS_RESPONSE_SOURCE = (
    "TESS Instrument Response Function v2.0, R. Vanderspek (MIT/POC), 2020-07-30; "
    "https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv, "
    "retrieved 2026-10-29"
)
TESS_BAND_TEMPERATURE_MIN_K = 2000.0
# Core table edge. Above it the table continues with the same log step, so every
# interpolation inside [2000, 50000] K reads the same two nodes as before.
_TESS_BAND_TABLE_CORE_MAX_K = 50000.0
# TRILEGAL background populations contain hot white dwarfs and post-AGB stars
# (retained populations reach 1.15e5 K); 1e6 K covers its hottest tracks.
TESS_BAND_TEMPERATURE_MAX_K = 1.0e6
# NUMERICAL_POLICY: 481 log-spaced temperatures make the tabulated log blackbody
# band flux linear-interpolation error <= 3e-5 relative over the stellar Teff
# range; the table is a one-time 0.4 s build, not a per-draw atmosphere call.
_TESS_BAND_TABLE_POINTS = 481


@lru_cache(maxsize=1)
def _tess_photon_passband():
    """Official TESS photon response, loaded once; no extinction is applied."""
    wavelengths_nm = []
    response = []
    with resources.files("trex").joinpath("_data", _TESS_RESPONSE_FILENAME).open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(",")
            if len(parts) != 2:
                raise ValueError("TESS photon response rows must contain wavelength and response")
            wave, value = (float(part) for part in parts)
            if not 0 <= value <= 1:
                raise ValueError("TESS photon response must be in [0,1]")
            wavelengths_nm.append(wave)
            response.append(value)
    wavelength_cm = np.asarray(wavelengths_nm, dtype=float) * 1.0e-7
    return PhotonPassband(
        wavelength_cm,
        np.asarray(response, dtype=float),
        np.zeros_like(wavelength_cm),
        _TESS_RESPONSE_SOURCE,
        "zero extinction: the default companion ratio is a blackbody reference",
        "TESS/TESS.T",
    )


@lru_cache(maxsize=1)
def _tess_band_log_blackbody_flux() -> Tuple[np.ndarray, np.ndarray]:
    """log Teff [K] and log TESS-band photon flux for a Planck reference table."""
    from trex.stellar_light.release import AngularSpectrum

    band = _tess_photon_passband()
    core = np.linspace(
        math.log(TESS_BAND_TEMPERATURE_MIN_K),
        math.log(_TESS_BAND_TABLE_CORE_MAX_K),
        _TESS_BAND_TABLE_POINTS,
    )
    step = core[1] - core[0]
    extra = math.ceil((math.log(TESS_BAND_TEMPERATURE_MAX_K) - core[-1]) / step)
    temperatures = np.exp(np.concatenate((core, core[-1] + step * np.arange(1, extra + 1))))
    wavelength_m = band.wavelength_cm * 1.0e-2
    photon_flux = np.empty(temperatures.size, dtype=float)
    for index, teff in enumerate(temperatures):
        # Planck I_lambda [erg s^-1 cm^-2 cm^-1 sr^-1]; the passband owner
        # applies its lambda/(hc) photon weight and the disc moment.
        intensity = (
            2.0 * const.h.cgs.value * const.c.cgs.value ** 2
            / band.wavelength_cm ** 5
            / np.expm1(SECOND_RADIATION_CONSTANT_M_K / (wavelength_m * teff))
        )
        spectrum = AngularSpectrum(
            float(teff),
            4.5,
            0.0,
            band.wavelength_cm,
            np.array([1.0]),
            intensity[None, :],
            "Planck blackbody reference",
        )
        photon_flux[index] = integrate_photon_intensity(
            spectrum, band, 0.0, require_nonnegative_quadratic=False
        )[0]
    return np.log(temperatures), np.log(photon_flux)


def tess_surface_brightness_ratio(
    numerator_teff_k: np.ndarray,
    denominator_teff_k: np.ndarray,
) -> np.ndarray:
    """TESS-band-integrated blackbody photon surface-brightness ratio.

    The historical default evaluated both Planck functions at the 786.5-nm
    pivot wavelength.  This function instead integrates each Planck spectrum
    over the official TESS response through the existing
    :mod:`trex.stellar_light` passband owner.  It remains a blackbody
    reference, not a PHOENIX atmosphere: explicit ``StellarLight`` evidence
    inputs still supply the physical atmosphere and never reach this default.
    """
    numerator = np.asarray(numerator_teff_k)
    denominator = np.asarray(denominator_teff_k)
    if (
        numerator.dtype.kind not in "iuf" or denominator.dtype.kind not in "iuf"
        or not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
        or np.any(numerator < TESS_BAND_TEMPERATURE_MIN_K)
        or np.any(numerator > TESS_BAND_TEMPERATURE_MAX_K)
        or np.any(denominator < TESS_BAND_TEMPERATURE_MIN_K)
        or np.any(denominator > TESS_BAND_TEMPERATURE_MAX_K)
    ):
        raise ValueError(
            "effective temperatures must be finite and inside the blackbody "
            f"reference table [{TESS_BAND_TEMPERATURE_MIN_K:g}, {TESS_BAND_TEMPERATURE_MAX_K:g}] K"
        )
    log_temperature, log_flux = _tess_band_log_blackbody_flux()
    return np.exp(
        np.interp(np.log(numerator), log_temperature, log_flux)
        - np.interp(np.log(denominator), log_temperature, log_flux)
    )
