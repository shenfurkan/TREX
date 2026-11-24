"""The TRILEGAL field extinction comes from the SFD98 map at the target's field.

Expected values are computed by hand from the cited conversion: the map's
E(B-V) is the raw SFD98 value, and A_V = 2.742 E(B-V)_SFD (Schlafly &
Finkbeiner 2011, ApJ 737, 103, Table 6: Landolt V, R_V = 3.1 F99 law).
"""

import io
from io import BytesIO
from urllib.parse import parse_qs
from urllib.error import URLError

import numpy as np
import pytest
from astropy.io import fits

from trex.acquisition import dust_map, transport
from trex.acquisition.transport import NetworkPolicy
from trex.acquisition.trilegal_errors import TrilegalUnavailableError

# 0.05 deg pixels: the 0.1 deg^2 cone (r = 0.1784 deg) holds offsets with
# dx^2 + dy^2 <= 12.73, i.e. 37 pixels, and reaches x = centre +- 3.
# The extra border at +-0.20 deg keeps the whole cone inside the image.
HEADER = fits.Header({
    "NAXIS": 2, "NAXIS1": 9, "NAXIS2": 9, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRVAL1": 150.0, "CRVAL2": 20.0, "CRPIX1": 5.0, "CRPIX2": 5.0,
    "CDELT1": -0.05, "CDELT2": 0.05, "BUNIT": "mag E(B-V)",
})
DATA = np.pad(np.tile(np.arange(7.0) * 0.05, (7, 1)), 1, mode="edge")
FIELD = (150.0, 20.0)
XML = """<?xml version="1.0"?>
<results status="ok">
  <result>
    <desc>E(B-V) Reddening</desc>
    <data><image>https://irsa.ipac.caltech.edu:443//workspace/x/p059Dust.fits</image></data>
    <statistics><refPixelValueSFD>0.1500 (mag)</refPixelValueSFD></statistics>
  </result>
  <result>
    <desc>100 Micron Emission</desc>
    <data><image>https://irsa.ipac.caltech.edu:443//workspace/x/p100Dust.fits</image></data>
  </result>
</results>"""


def _fits_bytes(header=HEADER, data=DATA):
    buffer = io.BytesIO()
    fits.PrimaryHDU(np.asarray(data, dtype=float), header=header).writeto(buffer)
    return buffer.getvalue()


def test_the_field_value_is_the_map_maximum_over_the_cone():
    statistics = dust_map.field_extinction_from_image(HEADER, DATA, *FIELD)
    # Hand-counted cone pixels: dx=0 -> 7, dx=+-1 -> 7 each, dx=+-2 -> 5 each,
    # dx=+-3 -> 3 each; the x values sum to 111 over 37 pixels.
    assert statistics["max"] == pytest.approx(0.3) and statistics["min"] == pytest.approx(0.0)
    assert statistics["mean"] == pytest.approx(0.15) and statistics["centre"] == pytest.approx(0.15)
    assert statistics["pixels"] == 37
    assert dust_map.field_extinction_from_image(HEADER, DATA, *FIELD)["max"] * dust_map.AV_PER_EBV_SFD == \
        pytest.approx(0.8226)


def test_an_image_without_the_field_is_refused():
    header = HEADER.copy()
    header["CRVAL1"] = 40.0
    with pytest.raises(TrilegalUnavailableError, match="no finite pixel"):
        dust_map.field_extinction_from_image(header, DATA, *FIELD)


def _serve(monkeypatch, replies, seen):
    class Response(BytesIO):
        headers = {}

    def urlopen(request, timeout):
        seen.append((request.get_header("User-agent"), request.get_method(), request.data))
        return Response(replies.pop(0))

    monkeypatch.setattr(transport, "urlopen", urlopen)


def test_the_map_query_retains_the_file_and_the_conversion(tmp_path, monkeypatch):
    seen = []
    _serve(monkeypatch, [XML.encode(), _fits_bytes()], seen)
    policy = NetworkPolicy(True, attempts=1, user_agent=lambda: "client/9")
    record = dust_map.field_extinction_from_map(*FIELD, policy, output_dir=tmp_path, record_prefix=tmp_path)

    # The submitted value is the cone maximum times the cited coefficient:
    # 0.3 mag E(B-V)_SFD * 2.742 = 0.8226 mag A_V.
    assert record["av_infinity_mag"] == pytest.approx(0.8226)
    assert record["ebv_sfd"]["max"] == pytest.approx(0.3) and record["ebv_sfd"]["pixels"] == 37
    assert record["service_reference_ebv_sfd"] == pytest.approx(0.15)
    assert record["map"] == "sfd98-ebv" and "Schlegel" in record["reference"]
    assert "Schlafly" in record["recalibration"] and "2.742" in record["conversion"]
    assert record["retrieved_utc"].endswith("Z") and record["status"] == "available"
    assert record["map_file"] == (tmp_path / dust_map.MAP_FILE_NAME).as_posix()
    assert (tmp_path / dust_map.MAP_FILE_NAME).read_bytes() == _fits_bytes()

    agent, method, body = seen[0]
    assert agent == "client/9" and method == "POST"
    assert parse_qs(body.decode()) == {"locstr": ["150.000000 20.000000"]}
    assert seen[1][1] == "GET"


def test_a_map_outage_is_a_typed_field_extinction_failure(tmp_path, monkeypatch):
    def unreachable(request, timeout):
        raise URLError("service down")

    monkeypatch.setattr(transport, "urlopen", unreachable)
    with pytest.raises(TrilegalUnavailableError, match="map query failed") as raised:
        dust_map.field_extinction_from_map(*FIELD, NetworkPolicy(True, attempts=1), output_dir=tmp_path, record_prefix=tmp_path)
    assert raised.value.trilegal_stage == "field-extinction" and raised.value.retryable
    assert not (tmp_path / dust_map.MAP_FILE_NAME).exists()


def test_an_image_in_another_unit_is_refused(tmp_path, monkeypatch):
    header = HEADER.copy()
    header["BUNIT"] = "MJy/sr"
    _serve(monkeypatch, [XML.encode(), _fits_bytes(header=header)], [])
    with pytest.raises(TrilegalUnavailableError, match=r"not E\(B-V\)"):
        dust_map.field_extinction_from_map(*FIELD, NetworkPolicy(True, attempts=1), output_dir=tmp_path, record_prefix=tmp_path)


def test_a_galactic_axis_map_gives_the_same_cone_as_an_equatorial_one():
    from astropy.coordinates import SkyCoord

    galactic = SkyCoord(*FIELD, unit="deg", frame="icrs").galactic
    header = HEADER.copy()
    header["CTYPE1"], header["CTYPE2"] = "GLON-TAN", "GLAT-TAN"
    header["CRVAL1"], header["CRVAL2"] = galactic.l.deg, galactic.b.deg
    equatorial = dust_map.field_extinction_from_image(HEADER, DATA, *FIELD)
    rotated = dust_map.field_extinction_from_image(header, DATA, *FIELD)
    assert rotated["pixels"] == equatorial["pixels"]
    for key in ("max", "min", "mean", "centre"):
        assert rotated[key] == pytest.approx(equatorial[key], abs=1e-10)


def test_a_cutout_that_does_not_contain_the_whole_cone_is_refused():
    header = HEADER.copy()
    # Centre the field on the corner pixel: the cone extends outside the image.
    header["CRPIX1"], header["CRPIX2"] = 0.5, 0.5
    with pytest.raises(TrilegalUnavailableError, match="whole 0.1 deg\\^2 cone"):
        dust_map.field_extinction_from_image(header, DATA, *FIELD)


def test_any_missing_pixel_inside_the_cone_is_refused():
    data = DATA.copy()
    data[4, 4] = np.nan
    with pytest.raises(TrilegalUnavailableError, match="missing pixels inside"):
        dust_map.field_extinction_from_image(HEADER, data, *FIELD)


def test_non_numeric_data_or_a_non_celestial_wcs_is_a_typed_failure():
    with pytest.raises(TrilegalUnavailableError, match="unusable"):
        dust_map.field_extinction_from_image(HEADER, [["not-a-number"]], *FIELD)
    header = HEADER.copy()
    header["CTYPE1"], header["CTYPE2"] = "LINEAR", "LINEAR"
    with pytest.raises(TrilegalUnavailableError, match="celestial"):
        dust_map.field_extinction_from_image(header, DATA, *FIELD)


@pytest.mark.parametrize("ra, dec", [(150.0, 95.0), (float("nan"), 20.0), (-1.0, 20.0),
                                     (True, 20.0), (360.0, 20.0), (150.0, float("inf"))])
def test_the_field_centre_must_be_a_finite_icrs_degree_pair(ra, dec):
    with pytest.raises(ValueError, match="valid domain"):
        dust_map.validate_icrs_field(ra, dec)


def test_a_map_failure_names_its_endpoint(tmp_path):
    with pytest.raises(TrilegalUnavailableError) as raised:
        dust_map.field_extinction_from_map(*FIELD, NetworkPolicy(False))
    assert raised.value.endpoint == "irsa.ipac.caltech.edu"
    assert raised.value.trilegal_stage == "field-extinction"
