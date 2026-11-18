"""Evidence-bound TESS target interface with TRICERATOPS 1.0.20 signatures.

Acquisition and source-table mutation are implemented for TESS. ``calc_depths``
is only the reference software's approximate Gaussian aperture diagnostic, not
a calibrated PRF measurement. ``bind_scene`` / ``from_scene`` require the real
calibrated scene, observed weights/exposure and retained input hashes before
``calc_probs`` can run the portable engine. No missing observations,
coefficients, or fits are replaced with defaults. Results are claim-ineligible.
API compatibility is not numerical equivalence to the upstream population model.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import math
import os
from collections.abc import Mapping
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.special import ndtr

from trex.models.flux_calibration import validate_flux_uncertainty
from trex.app.scene import TargetScene, _sha256, with_measured_contrast_curve
from trex.foundation.environment import read_environment
from trex.foundation.physical_constants import PARTS_PER_MILLION
from trex.app.doctor import (
    INPUT_ADMISSION, TARGET_BINDING, failed_input_report, inspect_inputs, record_target_failure,
)

STARS_COLUMNS = (
    "ID", "Tmag", "Jmag", "Hmag", "Kmag", "ra", "dec", "mass", "rad",
    "Teff", "plx", "disposition", "duplicate_id", "sep (arcsec)", "PA (E of N)",
)
# Reference software assumptions, not instrument measurements. Retained API
# source: TRICERATOPS 1.0.20, triceratops.py __init__ and calc_depths.
_REFERENCE_PIXEL_SCALE_ARCSEC = 20.25
_REFERENCE_SIGMA_PIXELS = 0.75


def _finite(value, label):
    from trex.populations._validation import real_scalar

    return real_scalar(value, label)


def _at_boundary(error, boundary):
    """Attach the failed boundary without replacing the original exception."""
    error.trex_boundary = boundary
    return error


def _json_safe(value):
    """Plain JSON with non-finite numbers and unknown objects kept visible as strings."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, str):
        return value if not isinstance(value, np.bool_) else bool(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def _query_tic_catalog(tic_id, radius_arcsec):
    import astropy.units as u
    from astroquery.mast import Catalogs

    return Catalogs.query_object(f"TIC{tic_id}", radius=radius_arcsec * u.arcsec, catalog="TIC")


def _search_tesscut(target, sector):
    import lightkurve

    return lightkurve.search_tesscut(target=target, sector=sector)


def _open_spoc_lc_hdulist(tic_id, sector):
    import lightkurve

    search = lightkurve.search_lightcurve(f"TIC {tic_id}", sector=sector, author="SPOC")
    product = search.download()
    if product is None:
        raise RuntimeError("No SPOC light-curve product returned")
    return product.hdu


def _gaussian_pixel_integral(cols, rows, mu_col, mu_row, sigma, amplitude):
    """Integrate a declared circular Gaussian over disjoint unit pixels.

    This is exact for that approximate PSF, not an observed TESS PRF. Survival
    functions in the positive tail avoid cancellation of two values near one.
    """
    def interval(lo, hi):
        return np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))

    dx = interval((cols - 0.5 - mu_col) / sigma, (cols + 0.5 - mu_col) / sigma)
    dy = interval((rows - 0.5 - mu_row) / sigma, (rows + 0.5 - mu_row) / sigma)
    return float(amplitude * np.sum(dx * dy))


def _stars_from_scene(scene):
    """Project only supplied scene fields; unavailable catalog fields stay NaN."""
    rows = [{
        "ID": scene.target_source_id if scene.tic_id is None else str(scene.tic_id),
        "Tmag": scene.Tmag, "ra": scene.ra_deg, "dec": scene.dec_deg,
        "mass": scene.M_s_Msun, "rad": scene.R_s_Rsun, "Teff": scene.Teff_K,
        "plx": scene.plx_mas, "sep (arcsec)": 0.0,
    }]
    rows.extend({
        "ID": neighbor["source_id"],
        "Tmag": np.nan if neighbor["delta_mag"] is None else scene.Tmag + neighbor["delta_mag"],
        "mass": neighbor["M_s"], "rad": neighbor["R_s"],
        "sep (arcsec)": neighbor["separation_arcsec"],
    } for neighbor in scene.resolved_neighbors)
    return pd.DataFrame(rows).reindex(columns=STARS_COLUMNS)


def _verify_artifacts(artifacts):
    if not artifacts:
        raise ValueError("hash-bound calibrated scene and photometry input artifacts are required")
    for name, digest in artifacts.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"bound input artifact is missing or changed: {path}")




class target:
    """TESS acquisition and conditional inference, with explicit evidence binding."""

    claim_eligible = False

    def __init__(self, ID: int, sectors: np.ndarray, search_radius: int = 10,
                 mission: str = "TESS", lightkurve_cache_dir=None, trilegal_fname=None,
                 ra: float = None, dec: float = None, verify_ssl: bool = True):
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS

        if mission != "TESS":
            raise NotImplementedError("Only TESS acquisition is implemented")
        if ra is not None or dec is not None:
            raise NotImplementedError("Alternate-coordinate acquisition is not implemented")
        if verify_ssl is not True:
            raise ValueError("Disabling TLS verification is not supported")
        if isinstance(ID, (bool, np.bool_)) or not isinstance(ID, (int, np.integer)) or ID <= 0:
            raise ValueError("ID must be a positive integer")
        if isinstance(search_radius, (bool, np.bool_)) or not isinstance(search_radius, (int, np.integer)) or search_radius <= 0:
            raise ValueError("search_radius must be a positive integer")
        values = np.asarray(sectors)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iu" or np.any(values <= 0):
            raise ValueError("sectors must be a non-empty 1D array of positive integers")
        if len(np.unique(values)) != len(values):
            raise ValueError("duplicate sectors are not supported")
        self.ID, self.sectors, self.search_radius = int(ID), values.copy(), int(search_radius)
        self._target_id_str = str(self.ID)
        self.N_pix = 2 * self.search_radius + 2
        self.mission = mission
        table = _query_tic_catalog(self.ID, self.search_radius * _REFERENCE_PIXEL_SCALE_ARCSEC)
        selected = table[list(STARS_COLUMNS[:13])]
        stars = selected.to_pandas() if hasattr(selected, "to_pandas") else pd.DataFrame(selected)
        stars["ID"] = stars["ID"].astype(str)
        if stars["ID"].duplicated().any():
            raise ValueError("duplicate catalog source ID")
        matches = stars.index[stars["ID"] == self._target_id_str].tolist()
        if len(matches) != 1:
            raise ValueError("Catalog must contain the owning target")
        stars = stars.loc[[matches[0]] + [i for i in stars.index if i != matches[0]]].reset_index(drop=True)
        for col in ("ra", "dec", "Tmag"):
            if not np.isfinite(stars[col].to_numpy(dtype=float)).all():
                raise ValueError(f"Every catalog source requires finite {col}")
        if ((stars["ra"] < 0) | (stars["ra"] >= 360) | (stars["dec"].abs() > 90)).any():
            raise ValueError("Catalog coordinates are outside their angular domain")
        self.stars = stars
        self._refresh_separations()
        self.TESS_images, self.col0s, self.row0s, self.pix_coords, self._sector_wcs = [], [], [], [], []
        coord = SkyCoord(stars.loc[0, "ra"], stars.loc[0, "dec"], unit="deg")
        for sector in self.sectors:
            try:
                options = {"cutout_size": (self.N_pix, self.N_pix)}
                if lightkurve_cache_dir is not None:
                    options["download_dir"] = str(lightkurve_cache_dir)
                products = _search_tesscut(target=coord, sector=int(sector)).download_all(**options)
                if products is None or len(products) == 0:
                    raise ValueError("No cutout returned")
                hdus = products[0].hdu
                cube = np.asarray(hdus[1].data["FLUX"], dtype=float)
                if cube.ndim != 3 or not np.isfinite(cube).any():
                    raise ValueError("Malformed cutout flux cube")
                image = np.nanmean(cube, axis=0)
                wcs = WCS(hdus[2].header)
                if not wcs.has_celestial:
                    raise ValueError("WCS lacks celestial axes")
                col0, row0 = (_finite(hdus[1].header[k], k) for k in ("1CRV4P", "2CRV4P"))
                self.TESS_images.append(image.copy())
                self.col0s.append(col0)
                self.row0s.append(row0)
                self._sector_wcs.append(wcs)
            except Exception as exc:
                raise RuntimeError(f"Requested sector {int(sector)} acquisition/WCS failed: {exc}") from exc
        self._refresh_pixels()
        self.trilegal_fname = None if trilegal_fname is None else str(Path(trilegal_fname))
        self.trilegal_sha256 = None
        if self.trilegal_fname is not None:
            path = Path(self.trilegal_fname)
            if path.is_symlink() or not path.is_file():
                raise ValueError("trilegal_fname must be an existing regular file")
            self.trilegal_sha256 = _sha256(path)
        self._invalidate()

    def _refresh_separations(self):
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(self.stars["ra"].to_numpy(float), self.stars["dec"].to_numpy(float), unit="deg")
        self.stars["sep (arcsec)"] = coords[0].separation(coords).arcsec
        self.stars["PA (E of N)"] = coords[0].position_angle(coords).deg

    def _refresh_pixels(self):
        self.pix_coords = []
        for wcs, col0, row0 in zip(self._sector_wcs, self.col0s, self.row0s, strict=True):
            xy = np.asarray(wcs.all_world2pix(self.stars[["ra", "dec"]].to_numpy(float), 0))
            if not np.isfinite(xy).all():
                raise ValueError("Source coordinates cannot be projected through cutout WCS")
            self.pix_coords.append(xy + [col0, row0])

    def _invalidate(self):
        self._scene_binding = None
        self._clear_results()
        if isinstance(self.stars, pd.DataFrame):
            self.stars = self.stars.drop(columns=[c for c in ("fluxratio", "tdepth") if c in self.stars])
        self.excluded_impossible_hosts = []
        self.depth_provenance = None

    def _clear_results(self):
        for name in ("probs", "FPP", "NFPP", "star_num", "u1", "u2", "fluxratio_EB", "fluxratio_comp", "trex_result"):
            setattr(self, name, None)
        self.calculation_provenance = None
        self.failure_record = None
        self._fit_observations = None

    @classmethod
    def from_scene(cls, scene, *, contrast_curve=None, contrast_band=None, **binding):
        """Create an offline target from a supplied calibrated ``TargetScene``.

        ``binding`` is the explicit ``bind_scene`` contract. No catalog, image,
        aperture, or stellar parameter is synthesized or downloaded. Field
        plotting requires acquired images and remains unavailable for this path.
        ``contrast_curve`` is an optional two-column file (separation [arcsec],
        delta magnitude [mag]; the TRICERATOPS/ExoFOP layout) or a
        ``(separations, delta_magnitudes)`` pair; ``contrast_band`` is then
        required and names the band the curve was measured in. TREX evaluates
        companion contrasts in the TESS band, so a non-TESS band is refused
        rather than converted.
        """
        if not isinstance(scene, TargetScene):
            raise TypeError("scene must be a validated TargetScene")
        scene = with_measured_contrast_curve(scene, contrast_curve, contrast_band)
        obj = cls.__new__(cls)
        obj.ID, obj._target_id_str = scene.tic_id, scene.target_source_id if scene.tic_id is None else str(scene.tic_id)
        obj.sectors = np.array(scene.sectors, dtype=int)
        obj.search_radius = obj.N_pix = None
        obj.mission = "multi-observation" if binding.get("joint_observations") is not None else "TESS"
        obj.stars = _stars_from_scene(scene)
        obj.TESS_images, obj.col0s, obj.row0s, obj.pix_coords, obj._sector_wcs = [], [], [], [], []
        obj.trilegal_fname, obj.trilegal_sha256 = str(scene.trilegal_cache), scene.background_sha256
        obj._invalidate()
        obj.bind_scene(scene, **binding)
        return obj

    @classmethod
    def from_tic(cls, ID, sectors, *, output_dir, depth_ppm, cache_dir=None, policy=None,
                 search_radius_arcsec=60.0, trilegal_population=None,
                 contrast_curve=None, contrast_band=None, **binding):
        """Acquire the scene for TIC ``ID`` from public archives, then bind it.

        Every archive response is retained under the new ``output_dir`` (see
        ``trex.acquisition.scene``); ``depth_ppm`` is the measured transit
        depth [ppm], used for neighbour characterization and bound for
        inference. ``binding`` is the remaining ``bind_scene`` contract
        (``sector_weights``, ``exptime_days``, ...) measured from the caller's
        light curve. ``trilegal_population`` reuses a retained TRILEGAL query
        record for this field instead of a new simulation.
        ``contrast_curve`` and ``contrast_band`` bind an optional measured
        high-resolution contrast curve exactly as in ``from_scene``; when
        supplied, the returned scene carries the curve and records which limit
        binds unresolved companions.
        """
        from trex.acquisition.scene import acquire_target_scene

        scene = acquire_target_scene(ID, np.asarray(sectors).tolist(), output_dir, depth_ppm=depth_ppm,
                                     cache_dir=cache_dir, policy=policy, search_radius_arcsec=search_radius_arcsec,
                                     trilegal_population=trilegal_population,
                                     contrast_curve=contrast_curve, contrast_band=contrast_band)
        # The acquired scene already carries the curve, and scene_acquisition.json states it.
        return cls.from_scene(scene, depth_ppm=depth_ppm, **binding)

    def bind_scene(self, scene, *, sector_weights, depth_ppm, exptime_days, input_artifacts=None,
                    bin_integration_days=0.0, random_seed=None, baseline=None, stellar_light=None,
                    depth_sigma_ppm=None, duration_days=None, duration_sigma_days=None,
                    joint_observations=None):
        """Bind calibrated inputs; never derive aperture evidence from magnitudes.

        ``scene`` follows Giacalone et al. (2021), AJ 161, 24,
        DOI 10.3847/1538-3881/abc6af. It must come from an externally calibrated
        aperture model (a supervisor's validated scene adapter), not calc_depths.
        ``sector_weights`` are sums of inverse variances of the retained native
        cadences; exposure and separate bin width are days; observed depth is ppm.
        ``input_artifacts`` maps retained calibration/photometry paths to SHA-256
        and is verified when supplied. When omitted, TREX hashes the one file it
        was given, the scene's background population, and records that.
        Hashes prove identity, not physical calibration. The caller owns the
        calibration's applicability and supplies one fixed calibrated sigma to
        calc_probs; this interface never estimates red noise on folded times.
        ``stellar_light`` explicitly binds the canonical atmosphere grid/model
        and target A_V [mag]; None retains historical physics. Component inputs
        are K, solar mass/radius and scaled-solar [M/H] dex. Resolved sources
        require their own atmosphere inputs rather than inheriting the target's.
        ``depth_sigma_ppm``, ``duration_days`` and ``duration_sigma_days`` are the
        measured transit values that centre the engine's Monte Carlo draws; leave
        them unset only when they were not measured.
        """
        self._invalidate()
        if not isinstance(scene, TargetScene):
            raise TypeError("scene must be a validated TargetScene")
        if scene.tic_id != self.ID or list(scene.sectors) != self.sectors.tolist():
            raise ValueError("calibrated scene target/sectors do not match this target")
        scene_stars = _stars_from_scene(scene)
        if scene_stars["ID"].duplicated().any() or set(scene_stars["ID"]) != set(self.stars["ID"]):
            raise ValueError("calibrated scene must retain every catalog source exactly once")
        if joint_observations is not None:
            joint_observations.bind_scene(scene)
            if sector_weights or exptime_days is not None or baseline is not None or np.any(np.asarray(bin_integration_days) != 0):
                raise ValueError("joint observations own their aperture, integration and baseline models")
        elif (not scene.sectors or len(set(scene.sectors)) != len(scene.sectors)
                or not isinstance(sector_weights, Mapping) or set(sector_weights) != set(scene.sectors)):
            raise ValueError("observed sector weights must cover exactly the calibrated sectors")
        weights = {}
        for sector, weight in (sector_weights or {}).items():
            if isinstance(sector, (bool, np.bool_)) or not isinstance(sector, (int, np.integer)):
                raise ValueError("observed sector weights require integer sector IDs")
            value = _finite(weight, "observed sector weight")
            if value <= 0:
                raise ValueError("observed sector weights must be positive")
            weights[int(sector)] = value
        depth = _finite(depth_ppm, "depth_ppm")
        exposure = None if joint_observations is not None else _finite(exptime_days, "exptime_days")
        # A zero scalar means no phase binning; per-point widths are positive [d], with length checked by the engine.
        width = (_finite(bin_integration_days, "bin_integration_days") if np.ndim(bin_integration_days) == 0
                 else tuple(_finite(value, "bin_integration_days") for value in bin_integration_days))
        if not 0 < depth < PARTS_PER_MILLION or (exposure is not None and exposure <= 0) or (
                 width < 0 if isinstance(width, float) else not width or min(width) <= 0):
            raise ValueError("positive depth/exposure, non-negative scalar bin width and positive per-bin widths are required")
        if random_seed is not None and (isinstance(random_seed, (bool, np.bool_))
                or not isinstance(random_seed, (int, np.integer)) or random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer or None")
        measured = {}
        for name, value in (("depth_sigma_ppm", depth_sigma_ppm), ("duration_days", duration_days),
                            ("duration_sigma_days", duration_sigma_days)):
            if value is not None:
                measured[name] = _finite(value, name)
                if measured[name] <= 0:
                    raise ValueError(f"{name} must be positive")
        if input_artifacts is None:
            scene.verify_background()  # ValueError if the file vanished or changed since construction
            input_artifacts = {str(scene.trilegal_cache): _sha256(scene.trilegal_cache)}
        if not isinstance(input_artifacts, Mapping):
            raise ValueError("input_artifacts must map retained input paths to SHA-256 digests")
        _verify_artifacts(input_artifacts)
        artifacts = {str(Path(name).absolute()): digest for name, digest in input_artifacts.items()}
        scene = TargetScene.from_dict(scene.to_dict())
        scene.verify_background()
        target_fraction = (joint_observations.proposal_fraction(scene.target_source_id) if joint_observations is not None
                           else scene.target_aperture_flux_fraction(weights))
        # The overcount guard applies to the coherent joint scene, not to the
        # per-host marginal limits: a residual G-less source's host scenario
        # uses its own marginal 95% fraction, and per-source limits are not
        # additive with the joint target fraction.
        neighbor_joint_fractions = [] if joint_observations is not None else [
            scene._reduce_aperture_flux_fraction(
                neighbor["aperture_flux_fractions"], weights, "neighbor"
            )
            for neighbor in scene.resolved_neighbors
        ]
        fractions = [target_fraction, *neighbor_joint_fractions]
        # Sum-roundoff grows with the number of supplied binary64 flux shares.
        roundoff = np.finfo(float).eps * len(fractions)
        if (not np.isfinite(fractions).all() or target_fraction <= 0
                or sum(fractions) > 1.0 + roundoff):
            raise ValueError("calibrated aperture flux shares must be finite, non-overcounted and include the target")
        if self.trilegal_sha256 is not None and self.trilegal_sha256 != scene.background_sha256:
            raise ValueError("calibrated scene background differs from the acquired target")
        # The calibrated scene, not preliminary TIC fields, owns inference inputs.
        adopted = self.stars.set_index("ID")
        for _, row in scene_stars.iterrows():
            for key in STARS_COLUMNS[1:]:
                if pd.notna(row[key]):
                    adopted.loc[row["ID"], key] = row[key]
        self.stars = adopted.reset_index()[list(STARS_COLUMNS)]
        self._refresh_pixels()
        self._scene_binding = dict(
            scene=scene, sector_weights=weights, depth_ppm=depth,
            exptime_days=exposure, bin_integration_days=width,
            random_seed=None if random_seed is None else int(random_seed),
            input_artifacts=artifacts, stars=self.stars.copy(deep=True), measured=measured,
            **({"baseline": baseline} if baseline is not None else {}),
            **({"stellar_light": stellar_light} if stellar_light is not None else {}),
            **({"joint_observations": joint_observations} if joint_observations is not None else {}),
        )

    def _require_scene(self, *, invalidate=True):
        binding = self._scene_binding
        if binding is None:
            raise ValueError("bind a calibrated TargetScene before calculating probabilities or replaying fits")
        scene = binding["scene"]
        expected_source = scene.target_source_id if scene.tic_id is None else str(scene.tic_id)
        if (self.ID != scene.tic_id or self._target_id_str != expected_source
                or not np.array_equal(self.sectors, scene.sectors)):
            if invalidate:
                self._invalidate()
            raise ValueError("target identity/sectors changed after calibrated scene binding; rebind the scene")
        if not isinstance(self.stars, pd.DataFrame) or not self.stars.equals(binding["stars"]):
            if invalidate:
                self._invalidate()
            raise ValueError("star table changed after calibrated scene binding; rebind the scene")
        try:
            _verify_artifacts(binding["input_artifacts"])
            binding["scene"].verify_background()
            if binding.get("joint_observations") is not None:
                binding["joint_observations"].bind_scene(scene)
        except (OSError, ValueError):
            if invalidate:
                self._invalidate()
            raise
        return binding

    def add_star(self, ID: int, Tmag: float, bound: bool):
        from trex.acquisition.identity import exact_identifier

        source_id = exact_identifier(ID, "source ID")
        if not source_id or source_id in set(self.stars["ID"]):
            raise ValueError("Star ID is empty or already exists")
        magnitude = _finite(Tmag, "Tmag")
        if not isinstance(bound, (bool, np.bool_)):
            raise ValueError("bound must be boolean")
        row = {key: np.nan for key in STARS_COLUMNS}
        row.update(ID=source_id, Tmag=magnitude, ra=self.stars.loc[0, "ra"], dec=self.stars.loc[0, "dec"])
        row["plx"] = self.stars.loc[0, "plx"] if bound else np.nan
        self._invalidate()
        self.stars = pd.concat([self.stars, pd.DataFrame([row])], ignore_index=True)
        self._refresh_separations()
        self._refresh_pixels()

    def remove_star(self, drop_stars: np.ndarray):
        ids = {str(value) for value in np.atleast_1d(drop_stars).ravel()}
        if self._target_id_str in ids:
            raise ValueError("Cannot remove owning target")
        self._invalidate()
        self.stars = self.stars.loc[~self.stars["ID"].isin(ids)].reset_index(drop=True)
        self._refresh_pixels()

    def update_star(self, ID: int, param: str, value: float):
        matches = self.stars.index[self.stars["ID"] == str(ID)].tolist()
        if not matches:
            raise ValueError("Unknown source ID")
        if param not in STARS_COLUMNS[1:11]:
            raise ValueError("Only measured photometric, astrometric and stellar fields can be updated")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{param} must be real, not boolean or text")
        numeric = float(value)
        if param in ("Tmag", "ra", "dec"):
            numeric = _finite(value, param)
        elif not math.isnan(numeric) and not math.isfinite(numeric):
            raise ValueError(f"{param} must be finite or missing (NaN)")
        if param in ("mass", "rad", "Teff", "plx") and not math.isnan(numeric) and numeric <= 0:
            raise ValueError(f"{param} must be positive")
        if (param == "ra" and not 0 <= numeric < 360) or (param == "dec" and not -90 <= numeric <= 90):
            raise ValueError("Coordinate outside angular domain")
        self._invalidate()
        self.stars.loc[matches, param] = numeric
        if param in ("ra", "dec"):
            self._refresh_separations()
            self._refresh_pixels()

    def get_spoc_apertures(self):
        result = []
        for sector in self.sectors:
            try:
                with _open_spoc_lc_hdulist(self._target_id_str, int(sector)) as hdus:
                    mask = np.asarray(hdus[2].data)
                    if mask.ndim != 2 or mask.dtype.kind not in "iu":
                        raise ValueError("Invalid aperture bitmask")
                    # SPOC bit 2 marks aperture membership, independently of other bits.
                    rows, cols = np.where((mask & 2) != 0)
                    if not len(rows):
                        raise ValueError("Empty SPOC aperture")
                    pixels = np.column_stack((cols + hdus[2].header["CRVAL1P"], rows + hdus[2].header["CRVAL2P"]))
                result.append(pixels.astype(float))
            except Exception as exc:
                raise RuntimeError(f"SPOC aperture unavailable for sector {int(sector)}: {exc}") from exc
        return result

    def calc_depths(self, tdepth: float, all_ap_pixels=None):
        """Reference Gaussian dilution diagnostic; not calibrated aperture evidence.

        ``tdepth`` is fractional, pixels are detector (column, row). Equal sector
        means reproduce the reference approximation, not observational weights.
        This result cannot be consumed by ``calc_probs`` as calibrated evidence.
        """
        depth = _finite(tdepth, "fractional tdepth")
        if not 0 < depth < 1:
            raise ValueError("fractional tdepth must be in (0, 1)")
        if all_ap_pixels is None or len(all_ap_pixels) != len(self.sectors):
            raise ValueError("One explicit aperture per sector is required")
        if len(self.pix_coords) != len(self.sectors):
            raise ValueError("acquired pixel coordinates are required for the Gaussian diagnostic")
        magnitudes = self.stars["Tmag"].to_numpy(float)
        ratios = []
        for aperture, coordinates in zip(all_ap_pixels, self.pix_coords, strict=True):
            pixels = np.asarray(aperture, dtype=float)
            if pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels) or not np.isfinite(pixels).all():
                raise ValueError("aperture must be finite (N, 2) detector pixels")
            if not np.equal(pixels, np.floor(pixels)).all() or len(np.unique(pixels, axis=0)) != len(pixels):
                raise ValueError("aperture pixels must be unique integer centers")
            amplitudes = 10 ** (-0.4 * (magnitudes - magnitudes.min()))
            integrals = np.array([_gaussian_pixel_integral(pixels[:, 0], pixels[:, 1], *xy,
                                                         _REFERENCE_SIGMA_PIXELS, amp)
                                  for xy, amp in zip(coordinates, amplitudes, strict=True)])
            if not np.isfinite(integrals).all() or integrals.sum() <= 0:
                raise ValueError("No finite positive aperture flux")
            ratios.append(integrals / integrals.sum())
        fractions = np.mean(ratios, axis=0)
        self._invalidate()
        self.stars["fluxratio"] = fractions
        possible = fractions >= depth
        self.stars["tdepth"] = np.divide(depth, fractions, out=np.full_like(fractions, np.nan), where=possible)
        self.excluded_impossible_hosts = self.stars.loc[~possible, "ID"].tolist()
        self.depth_provenance = {
            "model": "triceratops-1.0.20-reference-circular-gaussian-approximation",
            "sigma_pixels": _REFERENCE_SIGMA_PIXELS, "sector_weighting": "reference-equal-sector-mean",
            "calibrated": False, "claim_eligible": False,
            "excluded_impossible_hosts": list(self.excluded_impossible_hosts),
        }

    def doctor(self, time, flux, sigma, period_days, *, exptime=None, N=1_000_000, nsamples=20):
        """Check bound inputs with the engine's own admission; run no evidence draws.

        Days and normalized flux/errors use the same conventions as calc_probs.
        Existing probabilities and failure records are never changed by this check.
        """
        started = monotonic()
        try:
            binding = self._require_scene(invalidate=False)
            if exptime is not None and _finite(exptime, "exptime [days]") != binding["exptime_days"]:
                raise ValueError("exptime must equal the bound native exposure [days]; do not use a guessed default")
            return inspect_inputs(
                binding["scene"], time, flux, sigma, period_days, binding["depth_ppm"], binding["exptime_days"],
                sector_weights=binding["sector_weights"], bin_integration_days=binding["bin_integration_days"],
                n_draws=N, nsamples=nsamples,
                **({"baseline": binding["baseline"]} if binding.get("baseline") is not None else {}),
                **({"stellar_light": binding["stellar_light"]} if binding.get("stellar_light") is not None else {}))
        except Exception as error:
            return failed_input_report(
                error, stage=getattr(error, "trex_boundary", TARGET_BINDING), started=started)

    @record_target_failure
    def calc_probs(self, time: np.ndarray, flux_0: np.ndarray, flux_err_0: float, P_orb,
                   contrast_curve_file: str = None, filt: str = "TESS", N: int = 1000000,
                   parallel: bool = False, drop_scenario: list = [],  # noqa: B006 - upstream API
                   verbose: int = 1, flatpriors: bool = False, exptime: float = 0.00139,
                   nsamples: int = 20, molusc_file: str = None,
                   escalation_checkpoint_directory=None, escalation_state_in=None,
                   escalation_state_out=None, escalation_deadline_monotonic=None):
        """Run every supported scenario, using the bound calibrated scene.

        ``time`` is phase-relative days from the transit midpoint; flux/error
        are normalized aperture flux and its fixed, noise-calibrated uncertainty.
        N defaults to 1,000,000 per scenario; smaller budgets are for tests only.
        ``exptime`` must equal the bound native exposure: the legacy default is
        not evidence of a measured cadence. ``parallel`` selects nested-worker
        concurrency; the MC engine retains its measured device/resource policy.
        Unsupported upstream options raise rather than silently altering priors.
        With ``escalation_state_out``, pending state is returned in that
        mapping without publishing incomplete probability or fit fields.
        """
        from trex.app.engine import _resolved_random_seed, admitted_photometry, run_trex_vetting
        from trex.foundation.errors import TrexEvidenceUnavailableError
        from trex.runtime.hardware import resolve_n_jobs

        self._clear_results()
        try:
            binding = self._require_scene()
        except Exception as error:
            _at_boundary(error, TARGET_BINDING)
            raise
        if contrast_curve_file is not None:
            raise NotImplementedError(
                "calc_probs does not read a contrast-curve file; pass contrast_curve= "
                "and contrast_band= to target.from_tic or target.from_scene"
            )
        if filt != "TESS" or flatpriors or molusc_file is not None:
            raise NotImplementedError("alternate filters, flatpriors and MOLUSC are unsupported")
        if drop_scenario is not None and len(drop_scenario):
            raise NotImplementedError("drop_scenario is unsupported; every false-positive scenario must be retained")
        if (not isinstance(parallel, bool) or not isinstance(flatpriors, bool)
                or isinstance(verbose, bool) or not isinstance(verbose, (int, np.integer)) or verbose not in (0, 1, 2)):
            raise ValueError("parallel must be boolean and verbose must be 0, 1 or 2")
        for value, label in ((N, "N"), (nsamples, "nsamples")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise _at_boundary(ValueError(f"{label} must be a positive integer"), INPUT_ADMISSION)
        if np.ndim(P_orb) != 0:
            raise _at_boundary(
                NotImplementedError("period intervals are unsupported; supply a measured scalar period [days]"),
                INPUT_ADMISSION)
        try:
            period = _finite(P_orb, "P_orb [days]")
        except Exception as error:
            _at_boundary(error, INPUT_ADMISSION)
            raise
        if period <= 0:
            raise _at_boundary(ValueError("P_orb must be positive"), INPUT_ADMISSION)
        try:
            exposure = None if binding.get("joint_observations") is not None and exptime is None else _finite(exptime, "exptime [days]")
        except Exception as error:
            _at_boundary(error, TARGET_BINDING)
            raise
        if exposure != binding["exptime_days"]:
            raise _at_boundary(
                ValueError("exptime must equal the bound native exposure [days]; do not use a guessed default"),
                TARGET_BINDING)
        times, values, admitted_sigma = admitted_photometry(time, flux_0, flux_err_0)
        errors = np.broadcast_to(admitted_sigma, values.shape).copy()
        joint = binding.get("joint_observations")
        if joint is not None:
            first = joint.likelihood.observations.observations[0]
            if any(not np.array_equal(given, expected) for given, expected in zip(
                    (times, values, errors), (joint.likelihood.times[0] - joint.epoch_days, first.flux, first.sigma),
                    strict=True)):
                raise ValueError("proposal photometry must equal the first bound observation")
        seed = _resolved_random_seed(binding["random_seed"], escalation_state_in)
        if escalation_checkpoint_directory is None:
            escalation_checkpoint_directory = read_environment(
                "TREX_ESCALATION_CHECKPOINT_DIRECTORY",
                "EXONYM_TREX_ESCALATION_CHECKPOINT_DIRECTORY")
        if escalation_checkpoint_directory is None:
            from uuid import uuid4

            # Select a resumable path now; the escalation owner creates it only if needed.
            state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
            escalation_checkpoint_directory = str(state_root / "trex" / "checkpoints" / uuid4().hex)
        self.escalation_checkpoint_directory = escalation_checkpoint_directory
        from trex.evidence.mc_target import neighbor_metallicity_context

        provenance = {
            "claim_eligible": False, "n_draws": int(N), "random_seed": seed,
            "input_artifacts": dict(binding["input_artifacts"]),
            "sector_inverse_variance_weights": dict(binding["sector_weights"]),
            "exptime_days": binding["exptime_days"], "bin_integration_days": binding["bin_integration_days"],
            "depth_ppm": binding["depth_ppm"], "period_days": period, "nsamples": int(nsamples),
            **binding["measured"],
            "limb_darkening": [], "noise_scope": "caller-calibrated fixed sigma shared by every scenario",
            "neighbor_metallicity": [
                {"source_id": neighbor["source_id"], **neighbor_metallicity_context(
                    neighbor, binding["scene"].feh_dex, physical_light=binding.get("stellar_light") is not None)}
                for neighbor in binding["scene"].resolved_neighbors
            ],
            **({"noise_model": binding["baseline"].provenance()} if binding.get("baseline") is not None else {}),
            **({"stellar_light": binding["stellar_light"].model_configuration()} if binding.get("stellar_light") is not None else {}),
            **({"joint_observations": joint.provenance()} if joint is not None else {}),
            **({"background_population": copy.deepcopy(binding["scene"].background_population_provenance)}
               if binding["scene"].background_population_provenance is not None else {}),
        }
        try:
            result = run_trex_vetting(
                binding["scene"], times, values, errors, period, binding["depth_ppm"],
                exptime_days=binding["exptime_days"], bin_integration_days=binding["bin_integration_days"],
                sector_weights=binding["sector_weights"], n_draws=int(N), random_seed=seed,
                nsamples=int(nsamples), ldc_provenance=provenance["limb_darkening"],
                n_jobs=resolve_n_jobs(None) if parallel else 1, **binding["measured"],
                escalation_checkpoint_directory=escalation_checkpoint_directory,
                escalation_state_in=escalation_state_in, escalation_state_out=escalation_state_out,
                escalation_deadline_monotonic=escalation_deadline_monotonic,
                **({"baseline": binding["baseline"]} if binding.get("baseline") is not None else {}),
                **({"stellar_light": binding["stellar_light"]} if binding.get("stellar_light") is not None else {}),
                **({"joint_observations": joint} if joint is not None else {}),
            )
            if escalation_state_out:
                self._require_scene()
                return
            table = result.to_probability_dataframe()
            arrays = result.aligned_fit_arrays()
            self._require_scene()  # Do not publish results if bound inputs changed during inference.
        except Exception as error:
            self.failure_record = {**provenance, "error_type": type(error).__name__, "reason": str(error)}
            if isinstance(error, TrexEvidenceUnavailableError):
                self.failure_record["diagnostics"] = copy.deepcopy(error.diagnostics)
            raise
        self.trex_result, self.probs, self.FPP, self.NFPP = result, table, result.fpp, result.nfpp
        for name, array in arrays.items():
            setattr(self, name, array)
        self.calculation_provenance = provenance
        self._fit_observations = (times.copy(), values.copy(), errors)

    def write_outputs(self, directory, *, existing_empty=False):
        """Leave probabilities and diagnostics, or the failure, without overwriting output.

        A failed calculation writes only ``failure.json``, so no FPP/NFPP can be read
        from it. Non-finite numbers stay visible as JSON strings, never zero.
        """
        output = Path(directory)
        if output.is_symlink():
            raise FileExistsError(f"output path must not be a symlink: {output}")
        output.mkdir(exist_ok=existing_empty)
        if existing_empty and (output.is_symlink() or any(output.iterdir())):
            raise FileExistsError(f"output directory is not empty: {output}")
        from trex.stellar_light.host_limb_darkening import summarize_ldc_grid_provenance

        if self.failure_record is not None:
            payload, name = {**self.failure_record, "limb_darkening_grid": summarize_ldc_grid_provenance(
                self.failure_record.get("limb_darkening", []))}, "failure.json"
        else:
            result = self.trex_result
            if result is None:
                raise ValueError("no calculation has been attempted")
            from trex.acquisition.storage import write_bytes

            write_bytes(output / "scenarios.csv", self.probs.to_csv(index=False).encode("utf-8"))
            from trex.app.engine import (
                scenario_evidence_refinement, scenario_monte_carlo_evidence, unavailable_scenario_records,
            )

            payload, name = dict(
                FPP=self.FPP, NFPP=self.NFPP, status=result.status, claim_eligible=result.claim_eligible,
                input_identity=result.input_identity,
                effective_sample_sizes=result.effective_sample_sizes,
                scenario_monte_carlo_evidence=scenario_monte_carlo_evidence(result),
                scenario_evidence_refinement=scenario_evidence_refinement(result),
                unavailable_scenarios=unavailable_scenario_records(result),
                phase_wall_seconds=result.phase_wall_seconds,
                nested_evidence_escalation=result.nested_evidence_escalation,
                supplemental_nested_evidence=result.supplemental_nested_evidence,
                jax_batch_fallbacks=result.jax_batch_fallbacks, diagnostics=result.diagnostics,
                limb_darkening_grid=summarize_ldc_grid_provenance((self.calculation_provenance or {}).get("limb_darkening", [])),
                calculation_provenance=self.calculation_provenance,
            ), "summary.json"
        from trex.acquisition.storage import write_json

        write_json(output / name, _json_safe(payload))

    def plot_fits(self, time: np.ndarray, flux_0: np.ndarray, flux_err_0: float,
                   save: bool = False, fname: str = None):
        """Plot actual MC representative draws and residuals; never synthetic fits.

        Inputs must equal the likelihood's retained relative-day/normalized-flux
        arrays and fixed uncertainties. Labels distinguish MC representative
        draws from substituted nested evidences and disclose unavailable draws.
        With ``save`` false the open matplotlib figure is returned (a notebook
        displays it); with ``save`` true the PDF is written and None is returned.
        """
        from trex.app.target_plot import plot_representative_fits

        self._require_scene()
        if self._scene_binding.get("joint_observations") is not None:
            raise NotImplementedError("single-curve fit plotting cannot represent joint aperture observations")
        if self.trex_result is None or self._fit_observations is None:
            raise ValueError("no completed inference with retained draws; no synthetic fit is drawn")
        times, values = np.asarray(time, dtype=float), np.asarray(flux_0, dtype=float)
        errors = np.broadcast_to(validate_flux_uncertainty(flux_err_0, values), values.shape)
        if any(not np.array_equal(given, retained) for given, retained in zip(
                (times, values, errors), self._fit_observations, strict=True)):
            raise ValueError("fit plot inputs differ from the retained likelihood observations")
        return plot_representative_fits(self.trex_result, times, values, errors,
                                        save=save, fname=fname or f"TIC{self.ID}_fits")

    def plot_field(self, sector: int = None, ap_pixels=None, ap_color: str = "red",
                   save: bool = False, fname: str = None):
        """Return an open field figure, or save a PDF and close it when save=True."""
        import matplotlib.pyplot as plt

        if sector is None and len(self.sectors) != 1:
            raise ValueError("Specify sector for a multi-sector field")
        selected = self.sectors[0] if sector is None else sector
        if isinstance(selected, (bool, np.bool_)) or not isinstance(selected, (int, np.integer)):
            raise ValueError("sector must be an integer")
        indices = np.flatnonzero(self.sectors == selected)
        if len(indices) != 1:
            raise ValueError("Requested sector was not acquired")
        i = int(indices[0])
        if not self.TESS_images:
            raise ValueError("field image unavailable: from_scene does not synthesize acquired pixels")
        image = self.TESS_images[i]
        fig, ax = plt.subplots()
        keep_open = False
        try:
            ax.imshow(image, origin="lower", extent=(self.col0s[i] - 0.5, self.col0s[i] + image.shape[1] - 0.5,
                                                     self.row0s[i] - 0.5, self.row0s[i] + image.shape[0] - 0.5))
            ax.scatter(*self.pix_coords[i].T, facecolors="none", edgecolors="white")
            if ap_pixels is not None:
                pixels = np.asarray(ap_pixels, dtype=float)
                if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
                    raise ValueError("ap_pixels must be finite (N, 2) detector coordinates")
                ax.scatter(*pixels.T, marker="s", facecolors="none", edgecolors=ap_color)
            ax.set(xlabel="Detector column [pixel]", ylabel="Detector row [pixel]")
            if save:
                path = Path(fname or f"TIC{self.ID}_sector{selected}")
                fig.savefig(path if path.suffix.lower() == ".pdf" else Path(str(path) + ".pdf"))
            keep_open = not save
        finally:
            if not keep_open:
                plt.close(fig)
        return None if save else fig


__all__ = ["target", "STARS_COLUMNS"]
