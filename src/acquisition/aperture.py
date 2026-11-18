"""Calibrated detector-response operations for target-pixel scenes."""

import math
from functools import lru_cache
import re
from io import BytesIO
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import numpy as np

from .gaia_sources import gaia_position_at_epoch
from .storage import write_bytes, write_json
from .transport import AcquisitionError
from trex.runtime.logging import host_logger

# Official SPOC PRF product geometry, from PRF_BASE_URL + start_s0001/00README.txt and
# start_s0004/00README.txt (STScI, retrieved 2026-10-27).
TESS_PRF_PHYSICAL_PIXELS_PER_AXIS = 13
TESS_PRF_SAMPLES_PER_PIXEL = 9
TESS_PRF_ARRAY_PIXELS_PER_AXIS = TESS_PRF_PHYSICAL_PIXELS_PER_AXIS * TESS_PRF_SAMPLES_PER_PIXEL
PRF_BASE_URL = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/"
_GRID_FILE_PATTERN = re.compile(r'href="(tess\d+-prf-(\d)-(\d)-row(\d{4})-col(\d{4})\.fits)"')


class ApertureSceneError(RuntimeError):
    """Required aperture-scene evidence is absent, invalid, or inapplicable."""


class MissingCatalogPhotometry(ApertureSceneError):
    """The catalogue lacks a required measured flux, not a failed calculation."""


CATALOG_METHOD_ID = "giacalone2021-gaussian-psf-catalog-flux-ratio-v1"
CATALOG_METHOD_REFERENCE = "10.3847/1538-3881/abc6af"
# Giacalone et al. (2021), Sec. 2.1: "we adopt a standard deviation of 0.75
# pixels for all stars, regardless of CCD location".
GIACALONE_2021_PSF_SIGMA_PIXELS = Fraction(3, 4)
# Stassun et al. (2019), TIC v8 Gaia-to-TESS magnitude relation.
TESS_MAGNITUDE_RELATION_REFERENCE = "10.3847/1538-3881/ab3467"


def gaussian_psf_aperture_throughput(aperture_mask, x_pixel, y_pixel, sigma_pixels):
    """Return the fraction of a unit-flux circular Gaussian PSF inside the aperture.

    Pixel ``(row j, column i)`` spans ``[i - 1/2, i + 1/2] x [j - 1/2, j + 1/2]``
    in the zero-based astropy convention, so the integral of the separable
    Gaussian over each pixel is a product of two error-function differences.
    """
    from scipy.special import erf

    sigma = float(sigma_pixels)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ApertureSceneError("Gaussian PSF width must be positive and finite")
    if not math.isfinite(float(x_pixel)) or not math.isfinite(float(y_pixel)):
        raise ApertureSceneError("Gaussian PSF centre must be finite")
    rows, columns = np.nonzero(np.asarray(aperture_mask, dtype=bool))
    half = float(Fraction(1, 2))
    scale = sigma * math.sqrt(2.0)
    along_x = half * (erf((columns + half - x_pixel) / scale) - erf((columns - half - x_pixel) / scale))
    along_y = half * (erf((rows + half - y_pixel) / scale) - erf((rows - half - y_pixel) / scale))
    return float(math.fsum((along_x * along_y).tolist()))


def catalog_tess_magnitudes(sources):
    """Return each source's (TESS magnitude, Stassun et al. relation).

    The first source is the target: its G magnitude sets every relative flux
    and must be measured.  A neighbour without a finite G magnitude is refused,
    named, because no retained epoch-matched observation bounds its TESS flux.
    """
    from .photometry import tess_magnitude_from_gaia_photometry

    magnitudes, unavailable = [], []
    for index, row in enumerate(sources):
        source_id = str(row.get("source_id") or "unknown-source")
        magnitude, relation = tess_magnitude_from_gaia_photometry(
            row.get("phot_g_mean_mag"), row.get("phot_bp_mean_mag"), row.get("phot_rp_mean_mag"))
        if magnitude is not None and relation is not None and math.isfinite(float(magnitude)):
            magnitudes.append((float(magnitude), relation))
            continue
        if index == 0:
            unavailable.append("{0} (target lacks a measured G magnitude)".format(source_id))
        else:
            unavailable.append(
                "{0} (no measured G magnitude and no epoch-matched TESS-band flux ceiling)".format(source_id))
    if unavailable:
        raise MissingCatalogPhotometry(
            "archival Gaia source(s) {0} have no measured G magnitude in the retained Gaia DR3 "
            "rows. Gaia omits a mean G magnitude for sources with fewer than 10 valid G transits "
            "and after the poor-SSC photometric filter (Gaia Collaboration et al. 2021, A&A 649, "
            "A1, Sec. 7.1.2; Riello et al. 2021, A&A 649, A3, Sec. 8.2; Gaia EDR3/DR3 known "
            "issues). A Gaia DR2 catalogue G from the TIC is a measurement at the 2014-2016 "
            "epoch, not an upper bound on the source's TESS-epoch flux: a variable source can "
            "brighten by more than the 0.1 mag DR2-to-DR3 G-band scatter, and no retained "
            "TESS-band measurement or background model bounds that flux. The catalog flux-ratio "
            "scene cannot model these sources".format(", ".join(unavailable)))
    return magnitudes


def catalog_source_contributions(aperture, positions, magnitudes, relative_fluxes, sector):
    """Giacalone et al. (2021) Sec. 2.1 flux shares: catalog flux times Gaussian-PSF aperture throughput.

    Returns ``(contributions, total)``; ``relative_fluxes`` are linear TESS
    fluxes relative to the target, ``total`` their summed aperture flux.
    """
    sigma = float(GIACALONE_2021_PSF_SIGMA_PIXELS)
    throughputs = [gaussian_psf_aperture_throughput(aperture, source["x_pixel"], source["y_pixel"], sigma)
                   for source in positions]
    aperture_fluxes = [flux * throughput for flux, throughput in zip(relative_fluxes, throughputs)]
    total = math.fsum(aperture_fluxes)
    if not math.isfinite(total) or total <= 0.0:
        raise ApertureSceneError("no modeled catalog source flux falls inside the sector {0} SPOC aperture".format(sector))
    contributions = []
    for source, (magnitude, relation), flux, throughput, aperture_flux in zip(
            positions, magnitudes, relative_fluxes, throughputs, aperture_fluxes):
        contributions.append({
            "source_id": source["source_id"], "is_target": bool(source["is_target"]),
            "x_pixel": float(source["x_pixel"]), "y_pixel": float(source["y_pixel"]),
            "tess_mag": magnitude, "tess_mag_relation": relation, "relative_total_flux": flux,
            "aperture_throughput": throughput, "relative_aperture_flux": aperture_flux,
            "aperture_flux_fraction": aperture_flux / total,
        })
    return contributions, total


def load_empirical_prf_template(path):
    """Load and normalize a finite, nonnegative empirical PRF FITS image."""
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("astropy is required to read the calibrated PRF FITS template") from exc
    try:
        with fits.open(path, memmap=False) as hdul:
            template = np.asarray(hdul[0].data, dtype=float)
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("calibrated PRF template FITS is unreadable") from exc
    if template.ndim != 2 or template.size == 0 or not np.all(np.isfinite(template)) or np.any(template < 0.0):
        raise RuntimeError("calibrated PRF template must be a finite non-negative 2-D image")
    total = float(np.sum(template))
    if total <= 0.0:
        raise RuntimeError("calibrated PRF template has zero total response")
    return template / total


def validate_official_prf_fits(path):
    """Validate documented STScI PRF arrays and metadata without normalizing."""
    from astropy.io import fits

    try:
        with fits.open(path, memmap=False) as hdul:
            primary = hdul[0]
            template = np.asarray(primary.data, dtype=float)
            uncertainty = np.asarray(hdul[1].data, dtype=float)
            samples_per_pixel = int(primary.header["NSAMP"])
            datatype = str(primary.header["DATATYPE"]).strip()
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise RuntimeError("official TESS PRF FITS structure is unreadable") from exc
    expected_shape = (TESS_PRF_ARRAY_PIXELS_PER_AXIS, TESS_PRF_ARRAY_PIXELS_PER_AXIS)
    if template.shape != expected_shape or uncertainty.shape != expected_shape:
        raise RuntimeError("official TESS PRF FITS arrays do not match the documented dimensions")
    if samples_per_pixel != TESS_PRF_SAMPLES_PER_PIXEL or datatype != "PRF":
        raise RuntimeError("official TESS PRF FITS metadata does not match the documented product")
    if not np.all(np.isfinite(template)) or np.any(template < 0.0) or not np.any(template > 0.0):
        raise RuntimeError("official TESS PRF image must contain finite non-negative response values")
    if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
        raise RuntimeError("official TESS PRF uncertainty image must be finite and non-negative")
    return {"array_shape": [int(template.shape[0]), int(template.shape[1])],
            "physical_pixels_per_axis": TESS_PRF_PHYSICAL_PIXELS_PER_AXIS,
            "samples_per_pixel": samples_per_pixel}


def target_detector_position(cube):
    """Resolve target CCD position from the TPF aperture WCS and CRVALnP [pixels].

    Uses the loaded cube's ``aperture_header`` (TPF HDU 2); only a cube without
    it reopens ``cube["path"]``.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    header = cube["header"]
    try:
        camera = int(header["CAMERA"])
        ccd = int(header["CCD"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("candidate TPF header lacks a usable CAMERA/CCD assignment") from exc
    aperture_header = cube.get("aperture_header")
    if aperture_header is None:
        with fits.open(cube["path"], memmap=False) as hdul:
            aperture_header = hdul[2].header.copy()
    try:
        corner_column = float(aperture_header["CRVAL1P"])
        corner_row = float(aperture_header["CRVAL2P"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("TPF aperture header lacks the physical-WCS CRVAL1P/CRVAL2P keywords") from exc
    wcs = equatorial_celestial_wcs(WCS(aperture_header))
    if wcs is None:
        raise RuntimeError("TPF aperture WCS is not a two-axis equatorial celestial WCS")
    # CRPIX is the target's observation-epoch WCS reference, not RA_OBJ/DEC_OBJ.
    local_x, local_y = (float(value) - 1.0 for value in wcs.wcs.crpix)
    ny, nx = cube["flux"].shape[1:]
    if not (0.0 <= local_x < nx and 0.0 <= local_y < ny):
        raise RuntimeError("target sky position projects outside the ingested TPF cutout")
    if not (math.isfinite(local_x) and math.isfinite(local_y)):
        raise RuntimeError("target pixel position is not finite")
    return {"local_x": local_x, "local_y": local_y, "absolute_column": corner_column + local_x,
            "absolute_row": corner_row + local_y, "camera": camera, "ccd": ccd}


def aperture_reach_arcsec(cube):
    """Radius [arcsec] around the target that holds every star able to light the SPOC aperture.

    The farthest aperture-pixel corner from the target (the WCS reference pixel,
    its observation-epoch position) plus four standard deviations of the
    0.75-pixel Gaussian PSF, times the WCS pixel scale. A star beyond it puts
    less than the 4-sigma one-dimensional Gaussian tail (3e-5) of its flux into
    the aperture.
    """
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    wcs = equatorial_celestial_wcs(WCS(cube["aperture_header"]))
    if wcs is None:
        raise ApertureSceneError("TPF aperture WCS is not a two-axis equatorial celestial WCS")
    x_target, y_target = (float(value) - 1.0 for value in wcs.wcs.crpix)
    rows, columns = np.nonzero(pipeline_aperture_mask(cube["aperture"], np.asarray(cube["flux"]).shape[1:]))
    if rows.size == 0:
        raise ApertureSceneError("SPOC pipeline aperture is empty")
    farthest = float(np.max(np.hypot(columns - x_target, rows - y_target))) + 0.5 * math.sqrt(2.0)
    scale_arcsec = float(np.max(proj_plane_pixel_scales(wcs))) * 3600.0  # degrees per pixel -> arcsec
    return (farthest + 4.0 * float(GIACALONE_2021_PSF_SIGMA_PIXELS)) * scale_arcsec


def acquire_official_prf(cube, output_dir, transport):
    """Retain the STScI listing and nearest validated detector PRF for a TPF."""
    position = target_detector_position(cube)
    sector = cube["sector"]
    if isinstance(sector, bool) or not isinstance(sector, int) or sector < 1:
        raise ValueError("sector must be a positive integer")
    camera, ccd = position["camera"], position["ccd"]
    if camera not in range(1, 5) or ccd not in range(1, 5):
        raise ValueError("TESS camera and CCD must be between 1 and 4")
    directory = "start_s0001" if sector <= 3 else "start_s0004"
    url = f"{PRF_BASE_URL}{directory}/cam{camera}_ccd{ccd}/"
    body = transport._http_get_bytes(url, "text/html")
    if body is None:
        failure = transport.transport_failures[-1] if transport.transport_failures else None
        error = AcquisitionError("STScI PRF listing request failed", retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        raise error
    output = Path(output_dir)
    identifier = uuid4().hex
    listing = output / f"prf_grid_{identifier}.html"
    write_bytes(listing, body)
    retrieved = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {"status": "in-progress", "sector": sector, "camera": camera, "ccd": ccd,
              "detector_position": position, "listing": {"path": listing.name, "uri": url, "retrieved_utc": retrieved}}
    manifest = output / f"prf_query_{identifier}.json"
    write_json(manifest, record)
    grid = [(int(row), int(col), name) for name, cam, chip, row, col in
            _GRID_FILE_PATTERN.findall(body.decode("utf-8", errors="replace"))
            if int(cam) == camera and int(chip) == ccd]
    if not grid:
        record.update(status="unavailable", reason="STScI PRF listing has no matching detector grid")
        write_json(manifest, record)
        raise AcquisitionError(record["reason"])
    row, column, name = min(grid, key=lambda item: (item[0] - position["absolute_row"]) ** 2
                            + (item[1] - position["absolute_column"]) ** 2)
    product = output / f"prf_{identifier}.fits"
    downloaded = transport.download(url + name, product)
    if downloaded is None:
        failure = transport.transport_failures[-1] if transport.transport_failures else None
        record.update(status="unavailable", reason="STScI PRF FITS download failed")
        if failure is not None:
            record["network_failure"] = failure.to_manifest()
        write_json(manifest, record)
        error = AcquisitionError(record["reason"], retryable=bool(failure and failure.retryable))
        error.network_failure = failure
        raise error
    record["product"] = {"path": product.name, "uri": url + name, "retrieved_utc":
                         datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                         "field_position": {"row": row, "column": column}}
    try:
        record["product_structure"] = validate_official_prf_fits(product)
    except RuntimeError as exc:
        record.update(status="unavailable", reason=str(exc))
        write_json(manifest, record)
        raise AcquisitionError("STScI PRF FITS validation failed") from exc
    record["status"] = "available"
    write_json(manifest, record)
    return product, record


def pipeline_aperture_mask(aperture, shape):
    """Select SPOC pipeline-aperture bit 1 and require a nonempty matching image."""
    mask = (np.asarray(aperture) & 2) != 0
    if mask.shape != shape or not np.any(mask):
        raise ApertureSceneError("TPF has no declared SPOC pipeline aperture mask")
    return mask


def out_of_transit_image(cube, in_transit):
    """Median native electron/s TPF pixels outside the supplied transit mask."""
    time = np.asarray(cube.get("time"), dtype=float)
    quality = np.asarray(cube.get("quality_retained"), dtype=bool)
    flux = np.asarray(cube.get("flux"), dtype=float)
    in_transit = np.asarray(in_transit, dtype=bool)
    if (flux.ndim != 3 or time.ndim != 1 or quality.shape != time.shape
            or in_transit.shape != time.shape or flux.shape[0] != time.size):
        raise ApertureSceneError("TPF cube has incompatible time, quality, or pixel-flux dimensions")
    valid = quality & ~in_transit & np.isfinite(time) & np.all(np.isfinite(flux), axis=(1, 2))
    if not np.any(valid):
        raise ApertureSceneError("TPF has no finite quality-zero out-of-transit cadences")
    image = np.median(flux[valid], axis=0)
    if not np.all(np.isfinite(image)):
        raise ApertureSceneError("out-of-transit TPF image is non-finite")
    return image


def equatorial_celestial_wcs(wcs):
    """Return a two-axis equatorial RA/Dec WCS, refusing swapped or non-equatorial axes."""
    try:
        if not bool(wcs.has_celestial):
            return None
        celestial = wcs.celestial
        if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
            return None
        if (celestial.wcs.lng, celestial.wcs.lat) != (0, 1):
            return None
        if (celestial.wcs.lngtyp, celestial.wcs.lattyp) != ("RA", "DEC"):
            return None
    except (AttributeError, TypeError, ValueError):
        return None
    return celestial


def source_positions(wcs_header, target, neighbors, epoch_jyear=None, *, wcs_factory=None):
    """Project source ICRS coordinates into the TPF at its observation epoch."""
    if wcs_factory is None:
        from astropy.wcs import WCS
        wcs_factory = WCS
    try:
        wcs = equatorial_celestial_wcs(wcs_factory(wcs_header))
    except Exception as exc:
        raise ApertureSceneError("TPF WCS is unavailable") from exc
    if wcs is None:
        raise ApertureSceneError("TPF WCS is not a two-axis equatorial celestial WCS")
    positions = []
    source_ids = set()
    for index, row in enumerate([target, *neighbors]):
        source_id = str(row.get("source_id", ""))
        if not source_id or source_id in source_ids:
            raise ApertureSceneError("archival Gaia scene has missing or duplicate source identities")
        source_ids.add(source_id)
        try:
            moved = gaia_position_at_epoch(dict(row), epoch_jyear) if epoch_jyear is not None else None
            ra_deg, dec_deg = moved if moved is not None else (float(row["ra_deg"]), float(row["dec_deg"]))
            x_value, y_value = wcs.world_to_pixel_values(ra_deg, dec_deg)
            x_pixel = float(np.asarray(x_value))
            y_pixel = float(np.asarray(y_value))
        except Exception as exc:
            raise ApertureSceneError("archival Gaia source {0} cannot be projected into the TPF".format(source_id)) from exc
        if not math.isfinite(x_pixel) or not math.isfinite(y_pixel):
            raise ApertureSceneError("archival Gaia source {0} has non-finite TPF coordinates".format(source_id))
        positions.append({
            "source_id": source_id, "x_pixel": x_pixel, "y_pixel": y_pixel,
            "is_target": bool(row.get("is_target", index == 0)),
            "catalog_position_source": str(row.get("catalog_position_source", "archival-gaia")),
            "tic_id": row.get("tic_id"),
        })
    return positions


@lru_cache(maxsize=8)
def _pixel_grid(shape):
    # A phase search evaluates thousands of responses on one TPF shape.
    grid = np.indices(shape, dtype=float)
    grid.setflags(write=False)
    return grid[0], grid[1]


def detector_prf_image(shape, x_pixel, y_pixel, template, oversampling):
    """Sample the already pixel-integrated SPOC PRF without renormalization.

    STScI ``PRF_BASE_URL`` + start_s0001/00README.txt (retrieved 2026-10-27)
    defines 9x9 intra-pixel samples of the pixel response, not an optical PSF. Averaging
    neighbouring samples again convolves a second detector pixel into the PRF.
    """
    from scipy.ndimage import map_coordinates

    if template.ndim != 2 or template.shape[0] != template.shape[1]:
        raise ApertureSceneError("calibrated PRF template must be square")
    grid_y, grid_x = _pixel_grid(tuple(int(n) for n in shape))
    centre = (template.shape[0] - 1.0) / 2.0
    # SPOC's tabulated detector response is evaluated once at the source phase [native response/amplitude].
    response = map_coordinates(template, [(grid_y - y_pixel) * oversampling + centre,
                               (grid_x - x_pixel) * oversampling + centre],
                               order=1, mode="constant", cval=0.0)
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ApertureSceneError("calibrated PRF evaluation is non-finite or negative")
    return response


def solve_nonnegative(matrix, data):
    """Solve unit-column-scaled nonnegative amplitudes with SciPy's active-set NNLS.

    SciPy 1.15.3 documents the 3 * n_columns iteration budget. An exhausted
    solve stays unsuccessful; its unreported iteration count is not invented.
    Reference: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.nnls.html
    """
    from scipy.optimize import OptimizeResult, nnls

    maxiter = 3 * matrix.shape[1]
    try:
        coefficients, _ = nnls(matrix, data, maxiter=maxiter)
    except RuntimeError as exc:
        return OptimizeResult(success=False, status=0, nit="unreported",
                              message="{0} (maxiter={1})".format(exc, maxiter))
    return OptimizeResult(x=coefficients, success=True)


def fit_prf_scene(image, aperture, positions, template, oversampling, *, target_offset=(0.0, 0.0),
                  background=True, source_amplitudes="nonnegative", pixel_response=None,
                  diagnostics=False, solver=solve_nonnegative):
    """Fit all finite TPF pixels with nonnegative sources and a background.

    All finite pixels constrain the fit, not only the often-small pipeline
    aperture. A nonnegative pedestal absorbs residual background, and the
    target offset absorbs catalogue-to-observation phase error. Unit-column
    scaling z_j = ||A_j|| x_j preserves the native constrained objective while
    avoiding disparate source/background scales. Scaling does not establish
    identifiability or attribution accuracy. The residual is the Euclidean
    norm in native flux units, as is the fitted background.

    Pixel responses are already detector-integrated; they are never convolved
    with another pixel kernel. A template normalized to sum one at oversampling
    k has full detector response approximately 1/k**2. Signed amplitudes are
    diagnostic only; optional diagnostics expose the native design without
    inferring uncertainty from residuals. Production uses neither option.
    """
    from scipy.optimize import lsq_linear

    if not positions:
        raise ApertureSceneError("PRF fit requires at least one source position")
    if positions[0].get("is_target") is not True or any(source.get("is_target") for source in positions[1:]):
        raise ApertureSceneError("PRF fit requires the unique target as its first source")
    if source_amplitudes not in ("nonnegative", "signed-diagnostic"):
        raise ApertureSceneError("unknown source-amplitude constraint")
    mask = np.isfinite(image)
    if not np.any(mask):
        raise ApertureSceneError("out-of-transit image has no finite pixels")
    response_images = []
    for index, source in enumerate(positions):
        dx, dy = target_offset if index == 0 else (0.0, 0.0)
        x = float(source["x_pixel"]) + float(dx)
        y = float(source["y_pixel"]) + float(dy)
        response = (detector_prf_image(image.shape, x, y, template, oversampling)
                    if pixel_response is None else np.asarray(pixel_response(image.shape, x, y), dtype=float))
        if response.shape != image.shape or not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ApertureSceneError("source pixel-integrated response is invalid")
        response_images.append(response)
    design = np.column_stack([response[mask] for response in response_images])
    if not np.all(np.isfinite(design)):
        raise ApertureSceneError("source PRF design is non-finite")
    column_norms = np.linalg.norm(design, axis=0)
    if not np.any(column_norms > 0.0):
        raise ApertureSceneError("source PRF design has no finite response")
    matrix = design
    lower = np.full(design.shape[1], 0.0 if source_amplitudes == "nonnegative" else -np.inf)
    upper = np.full(design.shape[1], np.inf, dtype=float)
    if background:
        matrix = np.column_stack([design, np.ones((int(mask.sum()), 1))])
        # The sky pedestal has nonnegative flux support, not signed residual support.
        lower = np.concatenate([lower, [0.0]])
        upper = np.concatenate([upper, [np.inf]])
    data = np.asarray(image[mask], dtype=float)
    scales = np.linalg.norm(matrix, axis=0)
    # An exactly invisible source has zero support, not an invented amplitude.
    scales = np.where(scales > 0.0, scales, 1.0)
    try:
        if source_amplitudes == "nonnegative":
            result = solver(matrix / scales, data)
        else:
            result = lsq_linear(matrix / scales, data, bounds=(lower, upper), method="trf",
                                tol=1e-12, lsmr_tol="auto", max_iter=2000)
    except (RuntimeError, ValueError) as exc:
        raise ApertureSceneError("out-of-transit multi-source PRF fit failed: {0}".format(exc)) from exc
    if not result.success:
        raise ApertureSceneError(
            "out-of-transit multi-source PRF fit did not converge "
            "(status={0}, iterations={1}): {2}".format(result.status, result.nit, result.message))
    native_coefficients = result.x / scales
    amplitudes = np.asarray(native_coefficients[: design.shape[1]], dtype=float)
    background_native = float(native_coefficients[-1]) if background else 0.0
    residual = float(np.linalg.norm(matrix @ native_coefficients - data))
    output = {"amplitudes": amplitudes, "background_native": background_native, "residual": residual,
              "fit_pixel_count": int(mask.sum()),
              "target_offset_pixels": {"dx": float(target_offset[0]), "dy": float(target_offset[1])},
              "response_images": response_images, "mask": mask}
    if diagnostics:
        output.update(
            design_native=matrix, coefficients_native=native_coefficients,
            residual_pixels=matrix @ native_coefficients - data,
            source_amplitude_constraint=source_amplitudes,
            background_at_lower_bound=bool(background and background_native <= 1e-8 * max(float(np.max(np.abs(data))), 1.0)),
            source_at_lower_bound=(
                amplitudes <= 1e-8 * max(float(np.max(np.abs(amplitudes))), 1.0)
                if source_amplitudes == "nonnegative" else np.zeros(len(amplitudes), dtype=bool)),
        )
    return output


def fit_target_offset(image, aperture, positions, template, oversampling, *, fitter=fit_prf_scene):
    """Fit target phase against the full catalogue and nonnegative sky [pixels].

    The aperture initializes a bounded full-image profile; neighbours and sky
    remain in every final solve. Model mismatch can still bias attribution.
    """
    from scipy.optimize import least_squares, minimize

    if not positions:
        raise ApertureSceneError("target offset requires a target position")
    if positions[0].get("is_target") is not True or any(source.get("is_target") for source in positions[1:]):
        raise ApertureSceneError("target offset requires the unique target as its first source")
    x0 = float(positions[0]["x_pixel"])
    y0 = float(positions[0]["y_pixel"])
    yy, xx = np.indices(image.shape, dtype=float)
    cutout = ((np.abs(xx - x0) <= 2.0) & (np.abs(yy - y0) <= 2.0)
              & np.asarray(aperture, dtype=bool) & np.isfinite(image))
    if int(np.count_nonzero(cutout)) < 4:
        return 0.0, 0.0
    data = np.asarray(image[cutout], dtype=float)

    def residual(dx, dy):
        response = detector_prf_image(image.shape, x0 + dx, y0 + dy, template, oversampling)[cutout]
        denominator = float(np.dot(response, response))
        if denominator <= 0.0:
            return np.inf
        amplitude = max(float(np.dot(response, data)) / denominator, 0.0)
        return float(np.sum(np.square(amplitude * response - data)))

    # Deterministic initialization avoids phase-dependent local minima.
    best = None
    for dx in np.arange(-0.4, 0.401, 0.02):
        for dy in np.arange(-0.4, 0.401, 0.02):
            key = (residual(float(dx), float(dy)), abs(float(dx)) + abs(float(dy)), float(dx), float(dy))
            if best is None or key < best[0]:
                best = (key, float(dx), float(dy))
    assert best is not None
    _, best_dx, best_dy = best
    for dx in np.arange(best_dx - 0.02, best_dx + 0.0201, 0.0025):
        for dy in np.arange(best_dy - 0.02, best_dy + 0.0201, 0.0025):
            key = (residual(float(dx), float(dy)), abs(float(dx)) + abs(float(dy)), float(dx), float(dy))
            if key < best[0]:
                best = (key, float(dx), float(dy))
    # Variable projection: for each trial target phase [pixels] the nonnegative
    # source and sky amplitudes are re-solved, so only the phase is searched.
    # Only the target moves; every neighbour response is computed once.
    neighbour_responses = {}

    def response(shape, x, y):
        if (x, y) == target_xy[0]:
            return detector_prf_image(shape, x, y, template, oversampling)
        if (x, y) not in neighbour_responses:
            neighbour_responses[(x, y)] = detector_prf_image(shape, x, y, template, oversampling)
        return neighbour_responses[(x, y)]

    target_xy = [None]

    def profiled(offset, **options):
        target_xy[0] = (x0 + float(offset[0]), y0 + float(offset[1]))
        return fitter(image, aperture, positions, template, oversampling,
                      target_offset=(float(offset[0]), float(offset[1])), background=True,
                      pixel_response=response, **options)

    tolerance = math.sqrt(np.finfo(float).eps)
    fitted = minimize(lambda offset: profiled(offset)["residual"] ** 2,
        [best[1], best[2]], method="Powell", bounds=[(-0.5, 0.5), (-0.5, 0.5)],
        options={"xtol": tolerance, "ftol": tolerance})
    if (not fitted.success and fitted.status == 1 and np.all(np.isfinite(fitted.x))
            and math.isfinite(fitted.fun)):
        # Aravkin & van Leeuwen 2012, arXiv:1206.6532v1, Eqs. (3)-(4): profile each phase.
        host_logger(__name__).warning(
            "target phase Powell exhausted (status=%s, nfev=%s): %s; refining same image",
            fitted.status, fitted.nfev, fitted.message,
        )

        def pixel_residual(offset):
            return profiled(offset, diagnostics=True)["residual_pixels"]

        # TRF linear loss preserves the same squared native residual [e-/s].
        refined = least_squares(pixel_residual, fitted.x, bounds=(-0.5, 0.5),
                                ftol=1e-12, xtol=1e-12, gtol=1e-8)
        if (not refined.success or not np.all(np.isfinite(refined.x))
                or not np.all(np.isfinite(refined.fun)) or 2 * refined.cost > fitted.fun):
            raise ApertureSceneError(
                "target phase joint scene fit did not converge: {0}; same-image refinement: {1}".format(
                    fitted.message, refined.message))
        return float(refined.x[0]), float(refined.x[1])
    if not fitted.success or not np.all(np.isfinite(fitted.x)) or not math.isfinite(fitted.fun):
        raise ApertureSceneError("target phase joint scene fit did not converge: {0}".format(fitted.message))
    return float(fitted.x[0]), float(fitted.x[1])


def fit_sector_image(image, aperture, positions, template, oversampling, *,
                     offset_fitter=fit_target_offset, fitter=fit_prf_scene):
    """Fit native sector pixels and divide each modeled aperture flux by the total.

    ``positions[0]`` is the target (as ``source_positions`` orders them); the
    fitted phase offset applies to it alone.
    """
    if not positions or positions[0].get("is_target") is not True or any(
            source.get("is_target") is True for source in positions[1:]):
        raise ApertureSceneError("PRF scene needs the target first and only once")
    target_offset = offset_fitter(image, aperture, positions, template, oversampling)
    fit = fitter(image, aperture, positions, template, oversampling,
                 target_offset=target_offset, background=True)
    amplitudes = fit["amplitudes"]
    response_images = fit["response_images"]
    aperture_contributions = np.asarray(
        [amplitude * float(np.sum(response[aperture]))
         for amplitude, response in zip(amplitudes, response_images)], dtype=float)
    total_aperture_flux = float(np.sum(aperture_contributions))
    if not math.isfinite(total_aperture_flux) or total_aperture_flux <= 0.0:
        raise ApertureSceneError("out-of-transit PRF fit has no positive modeled aperture flux")
    source_contributions = []
    catalog_source_contributions = []
    for index, (source, amplitude, response, contribution) in enumerate(zip(
        positions, amplitudes, response_images, aperture_contributions
    )):
        # fit_prf_scene applies the phase offset to positions[0] (the target).
        dx, dy = target_offset if index == 0 else (0.0, 0.0)
        record = {
            "source_id": source["source_id"], "is_target": bool(source["is_target"]),
            "x_pixel": float(source["x_pixel"]) + float(dx),
            "y_pixel": float(source["y_pixel"]) + float(dy),
            "aperture_throughput": float(np.sum(response[aperture])),
            "oot_fitted_total_flux_native": float(amplitude),
            "oot_aperture_flux_native": float(contribution),
            "aperture_flux_fraction": float(contribution / total_aperture_flux),
            "catalog_position_source": str(source.get("catalog_position_source", "archival-gaia")),
        }
        if source.get("catalog_position_source") == "tic-v8-cone":
            catalog_source_contributions.append(record)
        else:
            source_contributions.append(record)
    return {
        "pipeline_aperture_pixel_count": int(np.count_nonzero(aperture)),
        "oot_fit_residual_native": float(fit["residual"]),
        "modeled_aperture_flux_native": total_aperture_flux,
        "fit_pixel_count": int(fit["fit_pixel_count"]), "fit_region": "full-tpf",
        "background_native": float(fit["background_native"]),
        "target_offset_pixels": fit["target_offset_pixels"],
        "source_contributions": source_contributions,
        "catalog_source_contributions": catalog_source_contributions,
    }


def acquire_aperture_images(tic_id, sectors, target, neighbors, transit_masks, output_dir, transport):
    """Fit retained SPOC sectors with explicit cadence-aligned transit masks.

    Retains each supplied mask and BTJD_TDB cadence array. The caller remains
    responsible for recording its ephemeris; official PRF provenance alone does
    not establish injection/recovery coverage.
    """
    from .gaia_sources import observation_epoch_jyear
    from .tpf import acquire_tpf_products, load_tpf

    output = Path(output_dir)
    records = acquire_tpf_products(tic_id, sectors, output, transport)
    if not records:
        raise AcquisitionError("no SPOC target-pixel sector is available to fit")
    sectors_out = []
    for product in records:
        sector = product["sector"]
        if sector not in transit_masks:
            raise AcquisitionError(f"sector {sector} lacks a recorded transit mask")
        cube = load_tpf(output / product["path"], tic_id, sector)
        image = out_of_transit_image(cube, transit_masks[sector])
        mask_buffer = BytesIO()
        np.savez_compressed(mask_buffer, time_btjd_tdb=cube["time"], in_transit=np.asarray(transit_masks[sector], dtype=bool))
        mask_name = f"transit_mask_s{sector:04d}_{uuid4().hex}.npz"
        write_bytes(output / mask_name, mask_buffer.getvalue())
        aperture = pipeline_aperture_mask(cube["aperture"], image.shape)
        epoch = observation_epoch_jyear(cube["time"])
        if epoch is None:
            raise AcquisitionError(f"sector {sector} has no finite observation epoch")
        positions = source_positions(cube["aperture_header"], target, neighbors, epoch)
        prf_path, prf_record = acquire_official_prf(cube, output, transport)
        template = load_empirical_prf_template(prf_path)
        fit = fit_sector_image(image, aperture, positions, template,
                               prf_record["product_structure"]["samples_per_pixel"])
        sectors_out.append({"sector": sector, "raw_tpf": product, "transit_mask": mask_name, "prf": prf_record, **fit})
        write_json(output / "aperture_images.json", {"status": "in-progress", "sectors": sectors_out})
    result = {"status": "available", "sectors": sectors_out,
              "limitation": "Official PRF provenance does not establish injection/recovery coverage."}
    write_json(output / "aperture_images.json", result)
    return result


def acquire_catalog_aperture(tic_id, sectors, target, neighbors, output_dir, transport, products=None):
    """Giacalone et al. (2021) catalog flux-ratio scene on retained SPOC TPF apertures and WCS.

    Every Gaia source is projected at the sector's observation epoch through the
    TPF aperture WCS; its catalog TESS flux (Stassun et al. 2019) passes through
    the published 0.75-pixel Gaussian PSF. No pixel is fitted, so no transit
    ephemeris is needed and the scene is uncalibrated: no injection/recovery
    calibration supports it.
    """
    from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT
    from .gaia_sources import observation_epoch_jyear
    from .tpf import acquire_tpf_products, load_tpf

    output = Path(output_dir)
    sources = [target, *neighbors]
    magnitudes = catalog_tess_magnitudes(sources)
    target_magnitude = magnitudes[0][0]
    relative_fluxes = [10.0 ** (POGSON_FLUX_EXPONENT * (item[0] - target_magnitude)) for item in magnitudes]
    sectors_out = []
    # ``products`` reuses TPFs already retained in ``output_dir`` by acquire_tpf_products.
    products = products if products is not None else acquire_tpf_products(tic_id, sectors, output, transport)
    if not products:
        raise AcquisitionError("no SPOC target-pixel sector is available for the catalog aperture scene")
    for product in products:
        cube = load_tpf(output / product["path"], tic_id, product["sector"])
        aperture = pipeline_aperture_mask(cube["aperture"], np.asarray(cube["flux"]).shape[1:])
        epoch = observation_epoch_jyear(cube["time"])
        if epoch is None:
            raise AcquisitionError(f"sector {product['sector']} has no finite observation epoch")
        positions = source_positions(cube["aperture_header"], target, neighbors, epoch)
        contributions, total = catalog_source_contributions(
            aperture, positions, magnitudes, relative_fluxes, product["sector"])
        sector_record = {"sector": product["sector"], "raw_tpf": product,
                         "pipeline_aperture_pixel_count": int(np.count_nonzero(aperture)),
                         "modeled_relative_aperture_flux": total, "source_contributions": contributions}
        sectors_out.append(sector_record)
    result = {"status": "available", "calibrated": False, "method": CATALOG_METHOD_ID,
              "method_reference": CATALOG_METHOD_REFERENCE,
              "psf_model": {"kind": "circular-gaussian", "sigma_pixels": float(GIACALONE_2021_PSF_SIGMA_PIXELS),
                            "pixel_integration": "exact-error-function"},
              "tess_magnitude_relation_reference": TESS_MAGNITUDE_RELATION_REFERENCE, "sectors": sectors_out,
              "limitation": ("Catalog TESS fluxes through a fixed 0.75-pixel Gaussian PSF (Giacalone et al. 2021, "
                             "Sec. 2.1), not fitted to the TPF pixels; the true TESS PRF varies with camera, CCD "
                             "and field position.")}
    write_json(output / "aperture_scene.json", result)
    return result
