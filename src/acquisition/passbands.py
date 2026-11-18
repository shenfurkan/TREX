"""Read published response tables without guessing units or detector conventions.

This module is the file I/O boundary for :mod:`trex.stellar_light.response`.
It accepts only tables whose own metadata declares the wavelength unit, detector
type and included components. A missing or contradictory declaration raises;
nothing is inferred from a filename, a filter letter or a plausible-looking
number range.

Supported editions:

* SVO Filter Profile Service VOTable XML (ground/space filter profiles). The
  ``DetectorType`` flag and ``components`` string are read from each record, not
  from a summary table. A filter-only WIRC curve therefore stays distinct from a
  ``Filter + Instrument + Atmosphere`` SDSS curve.
* Roman/GSFC per-SCA effective-area ECSV. The file has no wavelength-unit
  column, so the caller must supply the unit it verified; the leading comment
  declares m^2 effective area and that declaration is checked.
* PLATO DLR-TN-0113 v2 N-CAM and F-CAM transmission CSV, selecting a named
  published column. These are pre-launch as-designed/expected curves, not an
  in-flight calibration.

Every reader returns :class:`~trex.stellar_light.response.SpectralResponse`
with the source URL, retrieval UTC date and edition; no SHA-256 is recorded as
scientific provenance (it identifies bytes only).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re
from xml.etree import ElementTree

import numpy as np

from trex.stellar_light.response import SpectralResponse, ZeroPoint

_WAVELENGTH_TO_CM = {
    "angstrom": 1.0e-8,
    "angstroms": 1.0e-8,
    "a": 1.0e-8,
    "nm": 1.0e-7,
    "um": 1.0e-4,
    "micron": 1.0e-4,
    "microns": 1.0e-4,
    "cm": 1.0,
    "m": 100.0,
}
_AREA_TO_M2 = {"m2": 1.0, "m^2": 1.0, "cm2": 1.0e-4, "cm^2": 1.0e-4}

# PLATO DLR-TN-0113 v2 columns that are actual transmission models. Requirement
# columns (BOL/EOL limits) are not responses and are deliberately absent.
PLATO_TRANSMISSION_COLUMNS = {
    "NCAM": (
        "optical transmission",
        "expected transmission",
        "as designed camera effective transmission",
    ),
    "FCAM_BLUE": (
        "blue filter mission",
        "expected transmission blue camera",
        "as designed blue camera effective transmission",
    ),
    "FCAM_RED": (
        "red filter mission",
        "expected transmission red camera",
        "as designed red camera effective transmission",
    ),
}

# Historical WFIRST names are not modern Roman throughputs. A rename is not a
# response conversion, so these labels are refused rather than aliased.
ROMAN_HISTORICAL_BAND_NAMES = frozenset({
    "W146", "W149", "R062", "Z087", "Y106", "J129", "H158", "K213",
})


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{0} must be a non-empty string".format(label))
    return value.strip()


def _utc(value):
    text = _text(value, "retrieval UTC date")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(T[0-9:.+Z-]+)?", text):
        raise ValueError("retrieval date must be an ISO 8601 UTC date")
    return text


def _source(source_url, retrieved_utc):
    return "{0} (retrieved {1})".format(_text(source_url, "source URL"), _utc(retrieved_utc))


def _tag(tag):
    return tag.rsplit("}", 1)[-1]


def _read_votable(path):
    """Return (params, field_units, rows) from one SVO VOTable record."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise ValueError("unreadable VOTable response file") from exc
    for info in root.iter():
        if _tag(info.tag) == "INFO" and info.get("name") == "QUERY_STATUS":
            if info.get("value") != "OK":
                raise ValueError("SVO record does not declare a successful query")
    params = {}
    fields = {}
    rows = []
    for element in root.iter():
        tag = _tag(element.tag)
        if tag == "PARAM":
            name = element.get("name")
            if name is None:
                continue
            value = element.get("value")
            if name in params and params[name] != value:
                raise ValueError("SVO record has contradictory duplicate PARAM {0}".format(name))
            params[name] = value
        elif tag == "FIELD":
            name = element.get("name")
            if name is not None:
                fields[name] = element.get("unit") or ""
        elif tag == "TABLEDATA":
            for row in element:
                if _tag(row.tag) != "TR":
                    continue
                values = [cell.text for cell in row if _tag(cell.tag) == "TD"]
                if values:
                    rows.append(values)
    if not params or not fields or not rows:
        raise ValueError("VOTable response lacks parameters, fields or tabulated data")
    return params, fields, rows


def _float(value, label):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid {0} entry {1!r}".format(label, value)) from exc
    if not math.isfinite(number):
        raise ValueError("{0} contains a non-finite value".format(label))
    return number


def _strict_wavelength_cm(values, unit):
    factor = _WAVELENGTH_TO_CM.get(_text(unit, "wavelength unit").lower())
    if factor is None:
        raise ValueError(
            "unsupported wavelength unit {0!r}; no unit conversion is guessed".format(unit)
        )
    wavelength = np.asarray([_float(value, "wavelength") for value in values], dtype=float)*factor
    if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("response wavelengths must be positive and strictly increasing")
    return wavelength


def _response_quantity(unit):
    normalized = (unit or "").strip().replace(" ", "").lower()
    if normalized in ("", "dimensionless", "1"):
        return "dimensionless_transmission", 1.0
    factor = _AREA_TO_M2.get(normalized)
    if factor is None:
        raise ValueError(
            "unsupported response unit {0!r}; use a dimensionless transmission or an "
            "explicit m2/cm2 effective area".format(unit)
        )
    return "effective_area_m2", factor


def _zero_point(params, source_url, retrieved_utc, release):
    """One SVO zero point (Pogson flux density) or None when undeclared."""
    value = params.get("ZeroPoint")
    if value is None:
        if params.get("MagSys") or params.get("ZeroPointUnit"):
            raise ValueError("SVO record declares a magnitude system without a zero point")
        return None
    unit = _text(params.get("ZeroPointUnit"), "SVO zero-point unit")
    if unit != "Jy":
        raise ValueError("only an explicitly declared Jy SVO zero point is supported")
    if params.get("ZeroPointType", "Pogson") != "Pogson":
        raise ValueError("unsupported SVO zero-point type")
    system = _text(params.get("MagSys"), "SVO magnitude system")
    aliases = {"Vega": "Vega", "AB": "AB", "ST": "ST"}
    if system not in aliases:
        raise ValueError("unsupported SVO magnitude system {0!r}".format(system))
    number = _float(value, "SVO zero point")
    return ZeroPoint(
        aliases[system], number, "Jy",
        "{0}; {1}; retrieved {2}".format(
            params.get("PhotCalID") or params.get("filterID"),
            _text(source_url, "source URL"), _utc(retrieved_utc),
        ),
        _text(release, "release"),
    )


def read_svo_response(
    path,
    *,
    source_url,
    retrieved_utc,
    wavelength_convention,
    release="SVO-FPS",
):
    """Load one SVO filter profile with its declared convention and components.

    ``wavelength_convention`` is required because an SVO ``em.wl`` UCD does not
    distinguish air from vacuum; the caller records the convention it verified.
    """
    params, fields, rows = _read_votable(path)
    response_id = _text(params.get("filterID"), "SVO filterID")
    if "Wavelength" not in fields or "Transmission" not in fields:
        raise ValueError("SVO record must expose Wavelength and Transmission fields")
    detector_type = _text(params.get("DetectorType"), "SVO detector type")
    if detector_type not in ("0", "1"):
        raise ValueError("SVO detector type must be 0 or 1")
    components = tuple(
        part.strip() for part in _text(params.get("components"), "SVO components").split("+")
        if part.strip()
    )
    if not components:
        raise ValueError("SVO record declares no response components")
    quantity, area_factor = _response_quantity(fields["Transmission"])
    wavelength = _strict_wavelength_cm(
        [row[0] for row in rows], params.get("WavelengthUnit"))
    response = np.asarray(
        [_float(row[1], "transmission") for row in rows], dtype=float)*area_factor
    if np.any(response < 0):
        raise ValueError(
            "SVO response has negative entries; repair them explicitly instead of "
            "silently clipping a published curve"
        )
    filter_id = response_id.split("/", 1)[-1]
    filter_name = params.get("Band") or filter_id.rsplit(".", 1)[-1]
    facility = params.get("Facility") or params.get("Instrument") or response_id.split("/", 1)[0]
    instrument = params.get("Instrument") or facility
    profile_reference = params.get("ProfileReference") or "not declared"
    calibration_reference = params.get("CalibrationReference") or "not declared"
    return SpectralResponse(
        response_id,
        wavelength,
        response,
        quantity,
        int(detector_type),
        components,
        facility,
        instrument,
        filter_name,
        "{0}; {1}; retrieved {2}".format(_text(release, "release"), response_id, _utc(retrieved_utc)),
        (
            "SVO published profile; components={0}; detector convention from the "
            "record; profile reference={1}; calibration reference={2}; a filter-only "
            "curve does not certify an end-to-end instrument calibration"
        ).format("+".join(components), profile_reference, calibration_reference),
        _text(wavelength_convention, "wavelength convention"),
        _source(source_url, retrieved_utc),
        _zero_point(params, source_url, retrieved_utc, release),
    )


def _read_delimited(path):
    """Return (declared unit text, header, rows) from a comment-headed table."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError("unreadable response table") from exc
    declaration = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
    data = [line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(data) < 3:
        raise ValueError("response table has no header and data rows")
    if "," not in data[0] and ";" not in data[0]:
        # ECSV's default delimiter is whitespace (the released Roman zero points).
        header, *cells = (line.split() for line in data)
        rows = [row for row in cells if row]
        return declaration, header, rows
    delimiter = ";" if data[0].count(";") > data[0].count(",") else ","
    reader = csv.reader(data, delimiter=delimiter)
    header = [name.strip() for name in next(reader)]
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("response table has no data rows")
    return declaration, header, rows


def read_roman_effective_area(
    path,
    *,
    band,
    sca,
    wavelength_unit,
    detector_type,
    source_url,
    retrieved_utc,
    release,
    zero_point=None,
):
    """Read one GSFC per-SCA Roman/WFI effective-area column [m^2].

    The file's own comment must declare m^2 area. Wavelength unit is required
    because the ECSV has no unit column. ``sca`` (1-18) must match the filename
    so a per-SCA curve cannot be relabelled.
    """
    path = Path(path)
    if isinstance(sca, (bool, np.bool_)) or not isinstance(sca, (int, np.integer)) or not 1 <= sca <= 18:
        raise ValueError("Roman SCA index must be an integer in 1..18")
    match = re.search(r"SCA(\d{2})", path.name)
    if match is None or int(match.group(1)) != int(sca):
        raise ValueError("Roman effective-area filename does not identify the selected SCA")
    declaration, header, rows = _read_delimited(path)
    if "m^2" not in declaration.lower() and "m2" not in declaration.lower():
        raise ValueError("Roman effective-area file does not declare an m^2 quantity")
    lowercase = [name.lower() for name in header]
    if "wave" not in lowercase:
        raise ValueError("Roman effective-area file lacks the Wave column")
    band = _text(band, "Roman band").upper()
    if band in ROMAN_HISTORICAL_BAND_NAMES:
        raise ValueError(
            "historical WFIRST band {0!r} is not a modern Roman F146/F087/F213 "
            "throughput; supply that response's own curve and identity".format(band)
        )
    if band.lower() not in lowercase:
        raise ValueError("Roman effective-area file lacks band {0}".format(band))
    wave_index, band_index = lowercase.index("wave"), lowercase.index(band.lower())
    wavelength = _strict_wavelength_cm(
        [row[wave_index] for row in rows], wavelength_unit)
    response = np.asarray([_float(row[band_index], band) for row in rows], dtype=float)
    if np.any(response < 0):
        raise ValueError("Roman effective area has negative entries")
    filter_name = _text(band, "Roman band")
    return SpectralResponse(
        "Roman/WFI{0:02d}.{1}".format(int(sca), filter_name),
        wavelength,
        response,
        "effective_area_m2",
        detector_type,
        ("Filter", "Instrument", "Detector"),
        "Roman",
        "WFI",
        filter_name,
        "{0}; retrieved {1}".format(_text(release, "release"), _utc(retrieved_utc)),
        (
            "GSFC Roman/WFI per-SCA effective area as of 2024-03-01, including "
            "detector QE; pre-launch released parameters, not an in-flight "
            "calibration; zero points must come from the same edition"
        ),
        "vacuum",
        _source(source_url, retrieved_utc),
        zero_point,
    )


def read_roman_zero_points(*, path, source_url, retrieved_utc, release):
    """Read the released per-SCA Vega/AB/ST magnitudes as ZeroPoint records."""
    _declaration, header, rows = _read_delimited(path)
    lowercase = [name.lower() for name in header]
    required = ("detector", "element", "vegamag", "abmag", "stmag")
    if any(name not in lowercase for name in required):
        raise ValueError("Roman zero-point table lacks a required magnitude column")
    indices = {name: lowercase.index(name) for name in required}
    result = {}
    for row in rows:
        detector = _text(row[indices["detector"]], "Roman detector")
        match = re.fullmatch(r"WFI(\d{2})", detector)
        if match is None:
            raise ValueError("Roman zero-point detector is not a WFI SCA")
        sca = int(match.group(1))
        element = _text(row[indices["element"]], "Roman element")
        key = (sca, element)
        if key in result:
            raise ValueError("Roman zero-point table has a duplicate detector/filter row")
        source = "{0}; {1}; retrieved {2}".format(
            _text(source_url, "source URL"), _text(release, "release"), _utc(retrieved_utc))
        result[key] = {
            "Vega": ZeroPoint("Vega", _float(row[indices["vegamag"]], "VegaMag"), "mag",
                              source, _text(release, "release")),
            "AB": ZeroPoint("AB", _float(row[indices["abmag"]], "ABMag"), "mag",
                            source, _text(release, "release")),
            "ST": ZeroPoint("ST", _float(row[indices["stmag"]], "STMag"), "mag",
                            source, _text(release, "release")),
        }
    if not result:
        raise ValueError("Roman zero-point table has no rows")
    return result


def read_plato_transmission(
    path,
    *,
    channel,
    column,
    response_id,
    source_url,
    retrieved_utc,
    release,
    detector_type,
    zero_point=None,
):
    """Read one declared PLATO DLR-TN-0113 v2 transmission column.

    N-CAM and F-CAM editions are kept separate; the selected column name is
    embedded in the calibration context so ``as designed``, ``expected`` and
    ``required`` values cannot be confused. Required-transmission limit columns
    are refused because a requirement is not a measured response.
    """
    path = Path(path)
    channel = _text(channel, "PLATO channel").upper()
    if channel not in PLATO_TRANSMISSION_COLUMNS:
        raise ValueError("unsupported PLATO channel {0!r}".format(channel))
    declaration, header, rows = _read_delimited(path)
    if not re.search(r"wavelength\s+in\s+nm", declaration, flags=re.IGNORECASE):
        raise ValueError("PLATO transmission file does not declare wavelength in nm")
    is_ncam = "N-CAM" in declaration or bool(re.search(r"\bNCAM", declaration))
    if (channel == "NCAM") != is_ncam:
        raise ValueError("PLATO transmission file does not match the selected camera channel")
    response_id = _text(response_id, "PLATO response id")
    identifier = (response_id.upper().replace("_", "").replace("-", "")
                  .replace(".", "").replace(" ", ""))
    tokens = (("FCAM", "BLUE") if channel == "FCAM_BLUE"
              else ("FCAM", "RED") if channel == "FCAM_RED" else ("NCAM",))
    if not identifier.startswith("PLATO/") or not all(token in identifier for token in tokens):
        raise ValueError(
            "PLATO response id does not name the selected {0} channel".format(channel)
        )
    normalized = [" ".join(name.lower().split()) for name in header]
    selected = " ".join(_text(column, "PLATO column").lower().split())
    if selected not in PLATO_TRANSMISSION_COLUMNS[channel]:
        raise ValueError(
            "PLATO column {0!r} is not a published transmission model for {1}".format(
                column, channel
            )
        )
    if selected not in normalized:
        raise ValueError("PLATO transmission file lacks column {0!r}".format(column))
    wave_index = next((index for index, name in enumerate(normalized)
                       if name.startswith("wavelength")), None)
    if wave_index is None:
        raise ValueError("PLATO transmission file lacks a wavelength column")
    response_index = normalized.index(selected)
    wavelength = _strict_wavelength_cm(
        [row[wave_index] for row in rows], "nm")
    response = np.asarray([_float(row[response_index], selected) for row in rows], dtype=float)
    if np.any(response < 0):
        raise ValueError("PLATO transmission has negative entries")
    if "as designed" in selected:
        kind = "as-designed"
    elif "expected" in selected:
        kind = "expected-with-contamination"
    else:
        kind = "without-contamination"
    return SpectralResponse(
        response_id,
        wavelength,
        response,
        "dimensionless_transmission",
        detector_type,
        ("Filter", "Instrument"),
        "PLATO",
        channel,
        selected,
        "{0}; retrieved {1}".format(_text(release, "release"), _utc(retrieved_utc)),
        (
            "PLATO DLR-TN-0113 v2 {0} channel, column {1!r} ({2}); pre-launch "
            "published model, not an in-flight calibration and not per-camera "
            "as-built performance".format(channel, selected, kind)
        ),
        "vacuum",
        _source(source_url, retrieved_utc),
        zero_point,
    )


# Packaged editions (raw bytes; see trex/_data/responses/README.md).
_ROMAN_RESOURCE = ("_data", "responses", "roman")
_ROMAN_EFFAREA_FILE = "Roman_effarea_v8_SCA{0:02d}_20240301.ecsv"
_ROMAN_ZERO_POINT_FILE = "Roman_zeropoints_20240301.ecsv"
_ROMAN_SOURCE_URL = (
    "https://github.com/RomanSpaceTelescope/roman-technical-information/tree/"
    "14a9b0c98b0da723d0f2e4c6277956183c1fdd06/roman_technical_information/data/"
    "WideFieldInstrument/Imaging"
)
_ROMAN_RELEASE = "GSFC Roman/WFI effective area v8, parameters as of 2024-03-01"
_PLATO_RESOURCE = ("_data", "responses", "plato")
_PLATO_FILES = {"NCAM": "optical_transmission_ncam_v2.csv",
                "FCAM_BLUE": "optical_transmission_fcam_v2.csv",
                "FCAM_RED": "optical_transmission_fcam_v2.csv"}
_PLATO_SOURCE_URL = (
    "https://github.com/IvS-KULeuven/PlatoSim3/tree/"
    "4793088fc6fbb47cd134de1fbce2030da37d1421/docs/technical/DLR-TN-0113"
)
_PLATO_RELEASE = "PLATO DLR-TN-0113 v2 (PlatoSim3 docs)"
_PACKAGED_RETRIEVED_UTC = "2026-11-15"


def _packaged_path(parts, name):
    from importlib import resources

    resource = resources.files("trex").joinpath(*parts, name)
    if not resource.is_file():
        raise FileNotFoundError("packaged response {0} is not installed".format(name))
    # The readers take a filesystem path; installed wheels/sources are unpacked.
    return Path(str(resource))


def packaged_roman_response(band, sca, *, zero_point_system=None):
    """Packaged GSFC Roman/WFI per-SCA effective area [m^2] for one filter.

    ``Wave`` is micrometres: the ECSV declares no unit, and its 0.400-2.599
    range against the documented 0.48-2.3 um WFI coverage was checked when the
    table was packaged. The tables include filter, optics and detector QE for
    photon-counting H4RG detectors, so the photon kernel uses them directly
    (detector type 1). ``zero_point_system`` (``"AB"``, ``"Vega"`` or ``"ST"``)
    attaches that same-edition per-SCA zero point; None attaches none.
    """
    if isinstance(sca, (bool, np.bool_)) or not isinstance(sca, (int, np.integer)) or not 1 <= sca <= 18:
        raise ValueError("Roman SCA index must be an integer in 1..18")
    zero_point = None
    if zero_point_system is not None:
        table = read_roman_zero_points(
            path=_packaged_path(_ROMAN_RESOURCE, _ROMAN_ZERO_POINT_FILE),
            source_url=_ROMAN_SOURCE_URL, retrieved_utc=_PACKAGED_RETRIEVED_UTC,
            release=_ROMAN_RELEASE)
        key = (int(sca), _text(band, "Roman band").upper())
        if key not in table or zero_point_system not in table[key]:
            raise ValueError("no packaged Roman {0} zero point for SCA{1:02d} {2}".format(
                zero_point_system, int(sca), key[1]))
        zero_point = table[key][zero_point_system]
    return read_roman_effective_area(
        _packaged_path(_ROMAN_RESOURCE, _ROMAN_EFFAREA_FILE.format(int(sca))),
        band=band, sca=int(sca), wavelength_unit="um", detector_type=1,
        source_url=_ROMAN_SOURCE_URL, retrieved_utc=_PACKAGED_RETRIEVED_UTC,
        release=_ROMAN_RELEASE, zero_point=zero_point)


def packaged_plato_transmission(channel, column):
    """Packaged PLATO DLR-TN-0113 v2 optical transmission for one channel/column.

    These curves exclude detector quantum efficiency (``components`` stays
    ``Filter+Instrument``); they are not a complete PLATO photon response.
    The response id is ``PLATO/<channel>.<column>`` so different published
    columns can never share an id. No PLATO zero point is attached.
    """
    channel = _text(channel, "PLATO channel").upper()
    if channel not in _PLATO_FILES:
        raise ValueError("unsupported PLATO channel {0!r}".format(channel))
    label = "-".join(_text(column, "PLATO column").lower().split())
    return read_plato_transmission(
        _packaged_path(_PLATO_RESOURCE, _PLATO_FILES[channel]),
        channel=channel, column=column,
        response_id="PLATO/{0}.{1}".format(channel.replace("_", "-"), label),
        source_url=_PLATO_SOURCE_URL, retrieved_utc=_PACKAGED_RETRIEVED_UTC,
        release=_PLATO_RELEASE, detector_type=1)
