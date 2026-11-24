"""Binary/companion photon-light adapter in TREX's target-flux frame."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from astropy import units as u

from trex.foundation.constants import G, Msun, Rsun

_PARSEC_CM = u.pc.to(u.cm)

class ComponentLight(NamedTuple):
    """Prepared component: attenuated surface photons, quadratic and original rays.

    ``surface_photon_flux`` is photons s^-1 cm^-2; ``angular_profile`` is
    (mu, photon intensity per sr), or None ONLY for the explicit v1 model.
    When rays are present u1/u2 are diagnostic approximations, not a fallback.
    No radius/distance factor or additional extinction has been applied.
    """

    surface_photon_flux: object
    u1: object
    u2: object
    angular_profile: tuple | None


@dataclass(frozen=True)
class StellarLight:
    """Explicit physical-model input; None at old APIs means historical reference.

    The atmosphere grid is a declared scaled-solar model assumption, not an
    inferred stellar alpha enhancement. Target A_V [mag] is required, never
    guessed from background extinction. Background binaries use their OWN A_V;
    an unused target A_V may remain None in a background-hosted calculation.
    """

    grid: object
    target_av_mag: float | None = None

    def __post_init__(self):
        if self.target_av_mag is not None:
            av = np.asarray(self.target_av_mag)
            if av.ndim != 0 or av.dtype.kind not in "iuf" or not np.isfinite(av) or av < 0:
                raise ValueError("physical stellar light requires target A_V [mag]")
            object.__setattr__(self, "target_av_mag", float(av))

    @property
    def model_id(self):
        return self.grid.model_id

    def model_configuration(self):
        """JSON-ready physical selection for result context, not a numerical binding.

        Resume consumers must ALSO bind the actual grid axes/table/angular arrays
        in their existing input identity. A shared release label alone cannot
        distinguish changed retained cells, passband or extinction calculations.
        """
        source = self.grid.source
        return dict(model_id=self.model_id, release=source.get("release"),
                    abundance=source.get("abundance"), response=source.get("response"),
                    response_id=source.get("response_id"),
                    passband_input_id=source.get("passband_input_id"),
                    extinction=source.get("extinction"), target_av_mag=self.target_av_mag,
                    coordinates=["teff_K", "logg_cgs", "scaled_solar_mh_dex", "av_mag"],
                    photon_surface_flux_unit="photon s^-1 cm^-2",
                    angular_representation=("original-piecewise-linear-rays" if self.grid.angular is not None
                                            else "flux-conserving-quadratic"),
                    quadratic_role=("diagnostic-only; may be negative; use supplied original rays"
                                    if self.grid.angular is not None else "physical forward law"),
                    radius_convention="supplied stellar radius as declared atmosphere-disc radius",
                    interpolation="multilinear in absolute photon light; all positive-weight corners required")

    def _coordinates(self, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """Validate supplied stellar coordinates; gravity is log10 CGS, not SI."""
        if mh_dex is None or av_mag is None:
            raise ValueError("physical atmosphere requires its own composition and extinction")
        values = [np.asarray(v) for v in (teff_k, mass_solar, radius_solar, mh_dex, av_mag)]
        if any(v.dtype.kind not in "iuf" or not np.isfinite(v).all() for v in values):
            raise ValueError("physical atmosphere coordinates must be finite real numbers")
        temperature, mass, radius, mh_dex, av_mag = np.broadcast_arrays(*values)
        if (np.any(temperature <= 0) or np.any(mass <= 0) or np.any(radius <= 0) or np.any(av_mag < 0)):
            raise ValueError("physical atmosphere needs finite positive host temperature, mass and radius")
        # g=GM/R² [cm s^-2], same IAU/CODATA constants as the orbit calculation.
        gravity = np.log10(G*mass*Msun/(radius*Rsun)**2)
        return temperature, gravity, mh_dex, av_mag

    def atmosphere(self, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """Return a physically usable quadratic reduction; not the tabulated path.

        Some authentic angular profiles have no nonnegative reduction under
        the declared centre/flux constraints. Those remain usable through
        prepare_component's original rays, never through a negative quadratic.
        """
        from .passband import quadratic_nonnegative

        values = self.grid.evaluate(*self._coordinates(
            teff_k, mass_solar, radius_solar, mh_dex, av_mag))
        if not np.all(quadratic_nonnegative(values)):
            raise ValueError("atmosphere has no nonnegative quadratic reduction; consume its prepared original rays")
        return values[..., 0], -values[..., 2]/values[..., 1], -values[..., 3]/values[..., 1]

    def surface_light(self, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """Surface photons [s^-1 cm^-2] and (u1,u2) for per-draw arrays, without rays.

        Forward consumers take original rays per draw from prepare_component: a
        whole array would hold 2x8x79 float64 per draw and per body. On an
        angular grid u1/u2 are diagnostic and may be negative-valued; a v1 grid
        refuses negative nodes at construction and interpolates convexly.
        """
        values = self.grid.evaluate(*self._coordinates(
            teff_k, mass_solar, radius_solar, mh_dex, av_mag))
        return values[..., 0], -values[..., 2]/values[..., 1], -values[..., 3]/values[..., 1]

    def prepare_component(self, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """One component query for both photon light and angular occultation.

        Inputs: K, solar mass/radius, scaled-solar [M/H] dex, own A_V mag.
        Missing support raises; it never selects the reference model or a
        neighbouring composition. Prepared products are explicit values, not a
        parameter-dependent cache. Use separately for target and both EB bodies.
        On an angular grid u1/u2 are diagnostic polynomial coefficients only;
        ALWAYS consume angular_profile, never substitute that approximation.
        """
        coordinates = self._coordinates(teff_k, mass_solar, radius_solar, mh_dex, av_mag)
        values = self.grid.evaluate(*coordinates)
        profile = None if self.grid.angular is None else self.grid.angular_profile(*coordinates)
        return ComponentLight(values[..., 0], -values[..., 2]/values[..., 1],
                              -values[..., 3]/values[..., 1], profile)

    def angular_profile(self, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """None denotes the explicit v1 quadratic model, never a missing-ray fallback."""
        if self.grid.angular is None:
            return None
        return self.grid.angular_profile(*self._coordinates(teff_k, mass_solar, radius_solar, mh_dex, av_mag))

    def prepare_pair(self, primary, secondary, mh_dex, av_mag):
        """Return F2/F1 and prepared primary/secondary; same distance and own A_V.

        Each body is (T[K], M[Msun], R[Rsun]); the apparent primary magnitude
        remains the external anchor, already including distance and extinction.
        """
        first = self.prepare_component(*primary, mh_dex, av_mag)
        second = self.prepare_component(*secondary, mh_dex, av_mag)
        # Chen 2019 eq_Mag: same-distance photon ratio; attenuation is already in both fluxes.
        ratio = ((np.asarray(secondary[2])/np.asarray(primary[2]))**2
                 * second.surface_photon_flux/first.surface_photon_flux)
        return ratio, first, second

    def pair_own_atmospheres(self, primary, secondary, primary_mh_dex, secondary_mh_dex,
                             primary_av_mag, secondary_av_mag):
        """Same-distance F2/F1 with each body's own composition and extinction.

        ``primary``/``secondary`` are (T[K], M[Msun], R[Rsun]). A companion may
        have a different measured [M/H] and A_V; sharing the target's values is
        not implied. Attenuation enters each surface spectrum once, and the
        radius-squared scaling is applied after the surface fluxes are prepared.
        """
        first = self.prepare_component(*primary, primary_mh_dex, primary_av_mag)
        second = self.prepare_component(*secondary, secondary_mh_dex, secondary_av_mag)
        ratio = ((np.asarray(secondary[2])/np.asarray(primary[2]))**2
                 * second.surface_photon_flux/first.surface_photon_flux)
        return ratio, first, second

    def pair(self, primary, secondary, mh_dex, av_mag):
        """Same-distance binary: (T[K],M[Msun],R[Rsun]) -> F2/F1 and both (u1,u2).

        Ray-free for per-draw arrays; surface_light says when (u1,u2) are diagnostic.
        The apparent primary flux comes from the retained magnitude separately.
        Extinction enters BOTH spectra here; do not attenuate that magnitude or
        apply another distance factor when converting this ratio to target units.
        """
        first = self.surface_light(*primary, mh_dex, av_mag)
        second = self.surface_light(*secondary, mh_dex, av_mag)
        # Chen 2019 eq_Mag: same-distance component ratio, radius² times surface photons.
        ratio = (np.asarray(secondary[2])/np.asarray(primary[2]))**2*second[0]/first[0]
        return ratio, first[1:], second[1:]


def apparent_photon_flux(surface_flux, radius_solar, distance_pc):
    """Attenuated surface -> observed photons s^-1 cm^-2, broadcast in solar radii/pc.

    Surface flux must be finite and nonnegative; radius/distance finite positive
    real numbers. No second extinction factor or missing-input substitution.
    """
    values = [np.asarray(v) for v in (surface_flux, radius_solar, distance_pc)]
    if any(v.dtype.kind not in "iuf" or not np.isfinite(v).all() for v in values):
        raise ValueError("photon flux, radius and distance must be finite real numbers")
    surface_flux, radius_solar, distance_pc = np.broadcast_arrays(*values)
    if np.any(surface_flux < 0) or np.any(distance_pc <= 0) or np.any(radius_solar <= 0):
        raise ValueError("surface photon flux must be nonnegative; radius and distance must be positive")
    # Chen 2019 eq_Mag, inverse square dilution; no further extinction factor.
    return surface_flux*(radius_solar*Rsun/(distance_pc*_PARSEC_CM))**2


class BandStellarLight:
    """One prepared atmosphere grid per canonical response id.

    A band is selected explicitly and never inferred from a nearest wavelength.
    Each call routes the star's own temperature, mass, radius, composition and
    extinction into that band's grid, so a secondary, resolved neighbour or
    background star cannot silently reuse the target's atmosphere. Unavailable
    bands are absent rather than represented by another filter's grid.
    """

    def __init__(self, grids, target_av_mag=None):
        if not hasattr(grids, "items") or not grids:
            raise ValueError("multi-band stellar light requires at least one prepared grid")
        lights = {}
        for band_id, grid in grids.items():
            if not isinstance(band_id, str) or not band_id.strip():
                raise ValueError("prepared band keys must be nonempty response ids")
            if grid is None:
                raise ValueError("cannot expose an unavailable prepared band as stellar light")
            declared = grid.source.get("response_id")
            if declared and declared != band_id:
                raise ValueError("prepared grid response identity differs from its band key")
            lights[band_id] = StellarLight(grid, target_av_mag)
        self._lights = MappingProxyType(lights)
        self.target_av_mag = target_av_mag

    @property
    def bands(self):
        """Canonical prepared response ids, sorted for deterministic records."""
        return tuple(sorted(self._lights))

    def light(self, band_id):
        """Return the selected band's light; refuse an unselected band."""
        try:
            return self._lights[band_id]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "band {0!r} was not prepared; available bands are {1}".format(
                    band_id, list(self.bands))
            ) from exc

    def __getitem__(self, band_id):
        return self.light(band_id)

    def prepare_component(self, band_id, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        """Per-band component light for one star's own atmosphere coordinates."""
        return self.light(band_id).prepare_component(
            teff_k, mass_solar, radius_solar, mh_dex, av_mag)

    def prepare_pair(self, band_id, primary, secondary, mh_dex, av_mag):
        """Per-band same-distance pair ratio; both bodies keep their own coordinates."""
        return self.light(band_id).prepare_pair(primary, secondary, mh_dex, av_mag)

    def pair_own_atmospheres(self, band_id, primary, secondary, primary_mh_dex,
                             secondary_mh_dex, primary_av_mag, secondary_av_mag):
        """Per-band pair ratio using each body's own composition and extinction."""
        return self.light(band_id).pair_own_atmospheres(
            primary, secondary, primary_mh_dex, secondary_mh_dex,
            primary_av_mag, secondary_av_mag)

    def angular_profile(self, band_id, teff_k, mass_solar, radius_solar, mh_dex, av_mag):
        return self.light(band_id).angular_profile(
            teff_k, mass_solar, radius_solar, mh_dex, av_mag)

    def model_configuration(self):
        """Per-band physical selection for result context; not a sampling cue."""
        return {band_id: light.model_configuration()
                for band_id, light in sorted(self._lights.items())}

