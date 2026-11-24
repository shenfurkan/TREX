"""Read official PHOENIX angular radiation fields, never LDTk profiles.

Husser et al. (2013), A&A 553, A6, DOI 10.1051/0004-6361/201219058;
official /specint/ and /spectra/ documentation. The SPECINT-2011 release has
78 rays and vacuum wavelengths in Angstrom. Its raw I_lambda scale is CGS per
cm per sr (independently checked against the separate BUNIT-bearing HiRes
surface flux). It omits BUNIT/PHXALPHA; only named scaled-solar releases are
accepted, not arbitrary FITS arrays with guessed units or composition. Official
v3.0/SpecIntFITS/README.txt extends the same atmosphere
family to 5.5 microns. GRIDVER=3 is a separate model: its actual ray grids and
line-microturbulence metadata differ, so it cannot fill old-release holes.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from astropy.io import fits
import numpy as np

from trex.acquisition.storage import _sync_directory, write_json
from trex.acquisition.transport import AcquisitionError, AcquisitionTransport, NetworkPolicy
from trex.runtime.logging import host_logger

RELEASE = "PHOENIX-ACES-AGSS-COND-SPECINT-2011"
V3_RELEASE = "PHOENIX-ACES-AGSS-COND-2011-SpecInt-v3"
V3_SUFFIX = "PHOENIX-ACES-AGSS-COND-2011-SpecInt"
# Official SPECINT release coordinates (/specint/); these are not a support mask.
TEFF_AXIS = np.r_[np.arange(2300., 7001., 100.), np.arange(7200., 12001., 200.)]
LOGG_AXIS = np.arange(0., 6.01, .5)
METALLICITY_AXIS = np.array([-4., -3., -2., -1.5, -1., -.5, 0., .5, 1.])
V3_METALLICITY_AXIS = np.array([-4., -3., -2.5, -2., -1.5, -1., -.5, 0., .5, 1.])


@dataclass(frozen=True)
class AngularSpectrum:
    """Raw I_lambda [erg s^-1 cm^-2 cm^-1 sr^-1], vacuum wavelength [cm]."""

    teff_k: float
    logg_cgs: float
    mh_dex: float
    wavelength_cm: np.ndarray
    mu: np.ndarray
    intensity: np.ndarray
    source: str
    release: str = RELEASE
    metadata: dict | None = None


def read_phoenix_intensity(path):
    """Read one actual scaled-solar node; malformed required FITS fields raise ValueError."""
    path = Path(path)
    if path.name.endswith("." + RELEASE + ".fits"):
        release = RELEASE
    elif path.name.endswith("." + V3_SUFFIX + ".fits"):
        release = V3_RELEASE
    else:
        raise ValueError("requires an official scaled-solar PHOENIX specific-intensity release")
    try:
        with fits.open(path, memmap=False) as hdus:
            return _phoenix_spectrum(hdus, path, release)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("PHOENIX angular spectrum lacks valid required FITS fields") from exc


def _phoenix_spectrum(hdus, path, release):
    """Validate the official 78-ray release before it is published as usable input."""
    # Both selected releases contain 78 native rays; their mu positions need not agree.
    if not hdus:
        raise ValueError("PHOENIX angular spectrum lacks valid required FITS fields")
    header = hdus[0].header
    if (header.get("CTYPE1") != "WAVE" or header.get("CUNIT1") != "Angstrom"
            or header.get("CTYPE2") != "MU" or header.get("PHXEOS") != "ACES"
            or header.get("PHXDUST") is not False or header.get("INTERPOL", False)):
        raise ValueError("unsupported PHOENIX radiation-field metadata")
    if header.get("PHXALPHA", 0.0) != 0.0:
        raise ValueError("SPECINT-2011 adapter supports only the scaled-solar release")
    if release == V3_RELEASE and header.get("GRIDVER") != 3:
        raise ValueError("v3 specific intensities require GRIDVER=3, not a renamed older spectrum")
    intensity = np.asarray(hdus[0].data, dtype=float)
    mu = np.asarray(hdus["MU"].data, dtype=float)
    # Official linear FITS WCS, vacuum Angstrom -> cm; no air conversion.
    wave = (header["CRVAL1"] + (np.arange(header["NAXIS1"]) + 1
                               - header["CRPIX1"]) * header["CDELT1"]) * 1e-8
    parameters = [float(header[key]) for key in ("PHXTEFF", "PHXLOGG", "PHXM_H")]
    metadata = {key: header[key] for key in (
        "PHXVER", "PHXREFF", "PHXXI_L", "PHXXI_M", "PHXXI_N") if key in header}
    metadata["wavelength_step_Angstrom"] = float(header["CDELT1"])
    if (mu.ndim != 1 or wave.ndim != 1 or intensity.shape != (mu.size, wave.size) or mu.size != 78
            or not np.isfinite(intensity).all() or np.any(intensity < 0)
            or not np.isfinite(mu).all() or np.any(np.diff(mu) <= 0)
            or mu[0] <= 0 or mu[-1] != 1 or not np.isfinite(wave).all()
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)
            or not np.isfinite(parameters).all() or parameters[0] <= 0):
        raise ValueError("invalid PHOENIX angular spectrum")
    return AngularSpectrum(*parameters, wave, mu, intensity, str(path), release, metadata)


def phoenix_intensity_location(teff_k, logg_cgs, mh_dex, *, release=V3_RELEASE, sampling="50A"):
    """Official filename/URL for an explicitly sampled release node, not availability."""
    mh_axis = V3_METALLICITY_AXIS if release == V3_RELEASE else METALLICITY_AXIS
    if release not in (RELEASE, V3_RELEASE) or any(
            np.ndim(value) != 0 or np.asarray(value).dtype.kind not in "iuf" or value not in axis for value, axis in zip(
            (teff_k, logg_cgs, mh_dex), (TEFF_AXIS, LOGG_AXIS, mh_axis))):
        raise ValueError("unsupported PHOENIX release or published atmosphere coordinate")
    if sampling not in ("50A", "full") or (sampling == "full" and release != V3_RELEASE):
        raise ValueError("explicit full-resolution acquisition is supported for the selected v3 release only")
    # Official filenames use -0.0 for solar metallicity and - for nonnegative logg.
    metallicity = "-0.0" if mh_dex == 0 else f"{mh_dex:+.1f}"
    suffix = V3_SUFFIX if release == V3_RELEASE else RELEASE
    name = f"lte{teff_k:05.0f}-{logg_cgs:.2f}{metallicity}.{suffix}.fits"
    collection = "SpecInt50FITS" if sampling == "50A" else "SpecIntFITS"
    base = (f"https://ftp.gwdg.de/pub/misc/phoenix/v3.0/{collection}/" if release == V3_RELEASE
            else "https://ftp.gwdg.de/pub/misc/phoenix/SpecInt50FITS/" + RELEASE + "/")
    url = base + f"Z{metallicity}/" + name
    return name, url


def acquire_phoenix_intensity(teff_k, logg_cgs, mh_dex, directory, *, release=V3_RELEASE, sampling="50A",
                             transport=None):
    """Acquire one explicitly sampled official node, retaining its URL and UTC date.

    Coordinates are K, log10(cm s^-2), scaled-solar [M/H] dex. Missing nodes
    raise, never retry a different family/node. Releases are kept separate:
    v3 changes ray grids and line microturbulence, not only spectral extent.
    Husser 2013 and official v3.0/SpecIntFITS/README.txt define the sources.
    The shared transport bounds and retries downloads. Full FITS validation
    precedes publication; retained invalid or unreceipted input is refused,
    never silently replaced. Receipts record URL/date, not a scientific hash.
    """
    name, url = phoenix_intensity_location(teff_k, logg_cgs, mh_dex, release=release, sampling=sampling)
    path = phoenix_intensity_cache_path(teff_k, logg_cgs, mh_dex, directory, release=release, sampling=sampling)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".retrieval.json")
    if path.is_symlink() or receipt.is_symlink():
        raise ValueError("atmosphere cache files must not be symlinks")
    if path.exists():
        if not receipt.exists():
            raise ValueError("retained atmosphere has no acquisition URL/date receipt")
        try:
            retained = json.loads(receipt.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("retained atmosphere acquisition receipt is unreadable") from exc
        if (not isinstance(retained, dict) or retained.get("url") != url or retained.get("release") != release
                or not isinstance(retained.get("retrieved_utc"), str) or not retained["retrieved_utc"].strip()):
            raise ValueError("retained atmosphere acquisition receipt differs from the selected release/node")
        spectrum = read_phoenix_intensity(path)
    else:
        transport = transport if transport is not None else AcquisitionTransport(NetworkPolicy(allow_network=True))
        host_logger(__name__).info("PHOENIX atmosphere download started", extra={"event": "stellar_light.download"})
        # Stage on the destination filesystem; no partial FITS or receipt becomes a cache hit.
        with tempfile.TemporaryDirectory(prefix=".phoenix-", dir=directory) as staging:
            temporary = Path(staging)/name
            if transport.download(url, temporary) is None:
                failure = transport.transport_failures[-1]
                error = AcquisitionError(failure.reason, retryable=failure.retryable)
                error.failure = failure
                raise error
            spectrum = read_phoenix_intensity(temporary)
            if (spectrum.teff_k, spectrum.logg_cgs, spectrum.mh_dex) != (teff_k, logg_cgs, mh_dex):
                raise ValueError("downloaded atmosphere coordinates do not match the requested node")
            if path.exists() or receipt.exists() or path.is_symlink() or receipt.is_symlink():
                raise ValueError("atmosphere cache changed during acquisition; refusing replacement")
            staged_receipt = Path(staging)/receipt.name
            write_json(staged_receipt, dict(
                url=url, release=release, retrieved_utc=datetime.now(timezone.utc).isoformat()))
            published = []
            try:
                # The receipt is ready before the FITS becomes visible; neither link replaces data.
                os.link(staged_receipt, receipt)
                published.append(receipt)
                os.link(temporary, path)
                published.append(path)
                _sync_directory(directory)
            except BaseException:
                for destination in reversed(published):
                    destination.unlink()
                _sync_directory(directory)
                raise
        spectrum = replace(spectrum, source=str(path))
        host_logger(__name__).info("PHOENIX atmosphere download completed", extra={"event": "stellar_light.download"})
    if (spectrum.teff_k, spectrum.logg_cgs, spectrum.mh_dex) != (teff_k, logg_cgs, mh_dex):
        raise ValueError("retained atmosphere coordinates do not match the requested node")
    return spectrum


def phoenix_intensity_cache_path(teff_k, logg_cgs, mh_dex, directory, *, release=V3_RELEASE, sampling="50A"):
    """Official node cache path, shared by preparation and acquisition."""
    name, _ = phoenix_intensity_location(teff_k, logg_cgs, mh_dex, release=release, sampling=sampling)
    root = Path(directory)/release
    return root/("SpecIntFITS" if sampling == "full" else "")/name
