"""Field V-band extinction at infinity from the SFD98 dust map for TRILEGAL.

TRILEGAL's disc extinction layer is normalized by the total absorption at
infinity :math:`A_V^\\infty`, which the paper's default takes from the
Schlegel, Finkbeiner & Davis (1998) maps: "either adopting a local absorption
density ... or adopting a total absorption at infinity :math:`A_V^\\infty` as
given by Schlegel et al. (1998) maps.  We set the second option to be the
default one" (Girardi et al. 2005, A&A 436, 895, Sec. 3.5).  The web form's
``extinction_infty`` is that value in magnitudes of Johnson V; leaving it at
the server's 0.0378 mag gives every field the same asymptotic extinction,
which is wrong by orders of magnitude near the Galactic plane.

The map is sampled over the same 0.1 deg^2 cone the population is simulated
in.  The service returns the SFD map on a 0.025 deg grid; the field value is
the largest E(B-V) inside the cone, i.e. the upper end of the field's
attenuation envelope (the role ``field_av_max`` has in the LSST-simulation
fallback).  E(B-V) is the raw SFD value; the V-band extinction uses the
Schlafly & Finkbeiner (2011) conversion for Landolt V with an R_V = 3.1 F99
reddening law, A_V = 2.742 E(B-V)_SFD (their Table 6, which carries the 14%
SFD recalibration).  That recalibration is measured on SDSS stars at high
Galactic latitude with E(B-V)_SFD < 1; inside the plane it is an
extrapolation, and the Planck 2014 map used by the LSST simulation gives
larger extinctions there.
"""

from __future__ import annotations

import io
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from .storage import write_bytes
from .transport import AcquisitionTransport
from .trilegal_errors import TrilegalUnavailableError

FIELD_AREA_SQ_DEG = 0.1
CONE_RADIUS_DEG = math.sqrt(FIELD_AREA_SQ_DEG / math.pi)
MAP = "sfd98-ebv"
MAP_FILE_NAME = "trilegal_field_extinction.fits"
SERVICE_URL = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"
MAP_HOST = "irsa.ipac.caltech.edu"
# Schlafly & Finkbeiner (2011), ApJ 737, 103, Table 6: extinction per unit
# SFD E(B-V) for Landolt V, R_V = 3.1 F99 law, N = 0.78 normalization.
AV_PER_EBV_SFD = 2.742
REFERENCE = "Schlegel, Finkbeiner & Davis (1998), ApJ 500, 525, DOI 10.1086/305772"
RECALIBRATION = "Schlafly & Finkbeiner (2011), ApJ 737, 103, DOI 10.1088/0004-637X/737/2/103"
CONVERSION = (
    "A_V = {0:.3f} E(B-V)_SFD (Schlafly & Finkbeiner 2011, Table 6: Landolt V, "
    "R_V = 3.1 F99 law, N = 0.78)".format(AV_PER_EBV_SFD)
)
VALIDITY = (
    "SFD98 E(B-V) with the Schlafly & Finkbeiner (2011) recalibration; the 14% "
    "recalibration is measured for E(B-V)_SFD < 1 at high Galactic latitude "
    "(Schlafly & Finkbeiner 2011, Sec. 4.3 and 6), so the plane is an extrapolation, and "
    "the Planck 2014 map used by the LSST simulation gives larger values there"
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _unavailable(message, failure=None):
    """Return a typed outage tagged with the phase that failed."""
    error = TrilegalUnavailableError(message, failure, endpoint=MAP_HOST)
    error.trilegal_stage = "field-extinction"
    return error


def _reddening_image(body):
    """Return the E(B-V) map image URL and reference-pixel value of the reply."""
    try:
        root = ElementTree.fromstring(body.decode("utf-8"))
    except (UnicodeDecodeError, ElementTree.ParseError) as exc:
        raise _unavailable("field extinction map reply is unreadable: {0}".format(exc)) from None
    for result in root.iter("result"):
        if "E(B-V)" not in (result.findtext("desc") or ""):
            continue
        url = (result.findtext("data/image") or "").strip()
        reference = (result.findtext("statistics/refPixelValueSFD") or "").split()
        if url:
            try:
                return url, float(reference[0])
            except (IndexError, ValueError):
                return url, None
    raise _unavailable("field extinction map reply carries no E(B-V) image")


def field_extinction_from_image(header, data, ra_deg, dec_deg):
    """Return the E(B-V) statistics of the 0.1 deg^2 cone inside a map image.

    ``data`` is the service's SFD E(B-V) cutout in magnitudes; the returned
    ``max`` is the value submitted as the field's asymptotic extinction, the
    ``centre`` value is the map at the field centre for provenance.

    Pixel positions come from the image WCS in its own celestial frame and axis
    order (equatorial or Galactic), compared with the ICRS field centre. The
    cutout must contain the whole cone and every pixel inside it must be
    finite; anything else raises ``TrilegalUnavailableError``, never a partial
    maximum.
    """
    try:
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise _unavailable("field extinction map image is empty")
        wcs = WCS(header)
        if not wcs.has_celestial or wcs.celestial.pixel_n_dim != 2:
            raise _unavailable("field extinction map image has no two-axis celestial world coordinate system")
        celestial = wcs.celestial
        rows, columns = values.shape
        yy, xx = np.mgrid[0:rows, 0:columns]
        centre = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
        separation = pixel_to_skycoord(xx, yy, celestial, origin=0).separation(centre).deg
        if not np.all(np.isfinite(separation)):
            raise _unavailable("field extinction map image has pixels without sky coordinates")
        border = np.ones(values.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        if float(separation[border].min()) < CONE_RADIUS_DEG:
            raise _unavailable("field extinction map image does not cover the whole 0.1 deg^2 cone")
        inside = separation <= CONE_RADIUS_DEG
        selected = values[inside]
        if selected.size == 0:
            raise _unavailable("field extinction map has no finite pixel inside the 0.1 deg^2 cone")
        if not np.all(np.isfinite(selected)):
            raise _unavailable("field extinction map has missing pixels inside the 0.1 deg^2 cone")
        column, row = (int(round(float(coordinate))) for coordinate in skycoord_to_pixel(centre, celestial, origin=0))
    except TrilegalUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad map image must not escape untyped
        raise _unavailable("field extinction map image is unusable: {0}".format(exc)) from None
    centre_value = None
    if 0 <= row < rows and 0 <= column < columns:
        centre_value = float(values[row, column])
    return {"max": float(selected.max()), "mean": float(selected.mean()), "min": float(selected.min()),
            "centre": centre_value, "pixels": int(selected.size)}


def validate_icrs_field(ra_deg, dec_deg):
    """Refuse a field centre that is not a finite ICRS (RA in [0, 360), Dec in [-90, 90]) degree pair."""
    if (isinstance(ra_deg, bool) or isinstance(dec_deg, bool)
            or not isinstance(ra_deg, (int, float)) or not isinstance(dec_deg, (int, float))
            or not math.isfinite(ra_deg) or not math.isfinite(dec_deg)
            or not 0 <= ra_deg < 360 or not -90 <= dec_deg <= 90):
        raise ValueError("target coordinates are outside their valid domain")


def _response_or_unavailable(transport, body, what):
    if body is None:
        failure = transport.transport_failures[-1] if transport.transport_failures else None
        raise _unavailable("field extinction map {0} failed: {1}".format(
            what, failure.reason if failure else "no response"), failure)
    return body


def field_extinction_from_map(ra_deg, dec_deg, policy, *, writer=write_bytes, output_dir=Path(),
                              record_prefix=Path()):
    """Query the SFD map over the 0.1 deg^2 field and return its retained record.

    The reply and its image are fetched under ``policy`` (its User-Agent source
    is the caller's), the image is written into ``output_dir``, and the record
    names it relative to ``record_prefix`` with the map, retrieval date,
    conversion and raw E(B-V) values.
    """
    if not policy.allow_network:
        raise _unavailable("field extinction requires network access")
    validate_icrs_field(ra_deg, dec_deg)
    transport = AcquisitionTransport(policy)
    body = _response_or_unavailable(transport, transport.post_form(
        SERVICE_URL, urlencode({"locstr": "{0:.6f} {1:.6f}".format(ra_deg, dec_deg)}).encode("ascii"),
        accept="text/xml"), "query")
    image_url, service_reference = _reddening_image(body)
    parsed = urlparse(image_url)
    if parsed.scheme != "https" or parsed.hostname != MAP_HOST:
        raise _unavailable("field extinction map reply names an unexpected image host: {0}".format(image_url))
    image = _response_or_unavailable(transport, transport.get_bytes(image_url, "application/fits"),
                                     "image download")
    try:
        with fits.open(io.BytesIO(image)) as hdus:
            header, data = hdus[0].header, hdus[0].data
    except (OSError, ValueError) as exc:
        raise _unavailable("field extinction map image is not a readable FITS file: {0}".format(exc)) from None
    unit = str(header.get("BUNIT", "")).upper()
    if "E(B-V)" not in unit:
        raise _unavailable("field extinction map image is not E(B-V) (BUNIT {0!r})".format(unit))
    statistics = field_extinction_from_image(header, data, ra_deg, dec_deg)
    av_infinity = max(0.0, statistics["max"] * AV_PER_EBV_SFD)
    map_path = record_prefix / MAP_FILE_NAME
    writer(Path(output_dir) / MAP_FILE_NAME, image)
    return {
        "status": "available",
        "map": MAP,
        "reference": REFERENCE,
        "recalibration": RECALIBRATION,
        "service_url": SERVICE_URL,
        "query": "{0:.6f} {1:.6f}".format(ra_deg, dec_deg),
        "map_file": map_path.as_posix(),
        "retrieved_utc": _utc_now(),
        "field_area_sq_deg": FIELD_AREA_SQ_DEG,
        "cone_radius_deg": CONE_RADIUS_DEG,
        "sampling": "largest E(B-V) among the map pixels inside the 0.1 deg^2 cone (service grid 0.025 deg)",
        "service_reference_ebv_sfd": service_reference,
        "ebv_sfd": statistics,
        "av_per_ebv_sfd": AV_PER_EBV_SFD,
        "conversion": CONVERSION,
        "validity": VALIDITY,
        "av_infinity_mag": av_infinity,
    }
