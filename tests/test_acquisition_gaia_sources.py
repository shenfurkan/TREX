"""Gaia cone responses retain complete rows and fail closed on transport or overflow."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trex.acquisition.gaia_sources import (
    GAIA_SOURCE_COLUMNS, fetch_gaia_sources, gaia_position_at_epoch, normalize_and_match_gaia,
)
from trex.acquisition.transport import AcquisitionError, AcquisitionFailure


def _response(status="OK", source_id="9000000000000000001"):
    fields = "".join('<FIELD name="{}" datatype="{}"/>'.format(
        name, "long" if name == "source_id" else "int" if name == "non_single_star" else "double")
        for name in GAIA_SOURCE_COLUMNS)
    values = {"source_id": source_id, "ra": "1.0", "dec": "2.0", "sep_arcsec": "0.2",
              "ruwe": "1.05", "non_single_star": "3"}
    cells = "".join(f"<TD>{values.get(name, '')}</TD>" for name in GAIA_SOURCE_COLUMNS)
    return (f'<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
            f'<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/><TABLE>{fields}'
            f'<DATA><TABLEDATA><TR>{cells}</TR></TABLEDATA></DATA></TABLE>'
            f'<INFO name="QUERY_STATUS" value="{status}"/></RESOURCE></VOTABLE>').encode()


def _transport(get):
    return SimpleNamespace(ESA_GAIA_TAP_URL="https://esa.example/tap", AIP_GAIA_TAP_URL="https://aip.example/tap",
                           transport_failures=[], _http_get_bytes=get)


def test_cone_retains_exact_source_and_raw_response(tmp_path):
    body = _response()
    rows, record = fetch_gaia_sources(1.0, 2.0, 60.0, tmp_path, _transport(lambda *_: body))
    assert rows[0]["source_id"] == "9000000000000000001"
    assert rows[0]["parallax"] is None and rows[0]["sep_arcsec"] == pytest.approx(0.2)
    assert rows[0]["ruwe"] == pytest.approx(1.05) and rows[0]["non_single_star"] == 3
    assert "gs.ruwe" in record["query"] and "gs.non_single_star" in record["query"]
    assert (tmp_path / record["responses"][0]["path"]).read_bytes() == body
    assert record["responses"][0]["retrieved_utc"].endswith("Z")


def test_overflow_refuses_the_result_but_retains_response(tmp_path):
    body = _response(status="OVERFLOW")
    with pytest.raises(AcquisitionError, match="incomplete or invalid"):
        fetch_gaia_sources(1.0, 2.0, 60.0, tmp_path, _transport(lambda *_: body))
    assert next(tmp_path.glob("gaia_cone_*.xml")).read_bytes() == body


def test_access_denial_does_not_fall_through_to_another_backend(tmp_path):
    calls = []
    service = _transport(None)

    def denied(url, _accept):
        calls.append(url)
        service.transport_failures.append(AcquisitionFailure("esa.example", "http_status", False, 1,
                                                              "HTTP 403", 403, "HTTPError"))
        return None

    service._http_get_bytes = denied
    with pytest.raises(AcquisitionError) as exc:
        fetch_gaia_sources(1.0, 2.0, 60.0, tmp_path, service)
    assert len(calls) == 1 and not exc.value.retryable and exc.value.network_failure.http_status == 403


def test_proper_motion_moves_at_declared_epoch_and_requires_both_components():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    source = {"ra_deg": 10.0, "dec_deg": 0.0, "pmra_mas_per_year": 1000.0,
              "pmdec_mas_per_year": 0.0, "reference_epoch_jyear": 2016.0}
    moved = gaia_position_at_epoch(source, 2000.0)
    distance = SkyCoord(10 * u.deg, 0 * u.deg).separation(SkyCoord(moved[0] * u.deg, moved[1] * u.deg))
    assert distance.to_value(u.arcsec) == pytest.approx(16.0, rel=1e-7)
    assert gaia_position_at_epoch({**source, "pmdec_mas_per_year": None}, 2000.0) is None
    assert gaia_position_at_epoch(source, None) is None


def test_target_match_propagates_high_proper_motion_and_preserves_neighbors():
    # At 1 arcsec/year a Gaia position at 2016 is 16 arcsec from its J2000 position.
    target = {"source_id": "123", "ra": 10.0 + 16.0 / 3600.0, "dec": 0.0, "sep_arcsec": 16.0,
              "pmra": 1000.0, "pmdec": 0.0, "ref_epoch": 2016.0,
              "phot_g_mean_flux": 1000.0, "phot_g_mean_flux_error": 10.0}
    neighbor = {"source_id": "456", "ra": 10.01, "dec": 0.0, "sep_arcsec": 36.0}
    matched, others = normalize_and_match_gaia([target, neighbor], 10.0, 0.0, 2000.0, match_radius_arcsec=2.0)
    assert matched["source_id"] == "123" and matched["target_match_method"] == "proper_motion_to_target_epoch"
    assert matched["target_match_evidence"] == "unique_epoch_compatible_match"
    assert matched["phot_g_mean_mag_error"] == pytest.approx((2.5 / __import__("math").log(10)) * 0.01)
    assert [row["source_id"] for row in others] == ["456"]
    with pytest.raises(AcquisitionError, match="exactly one compatible source; found 2"):
        normalize_and_match_gaia([target, {**target, "source_id": "789"}],
                                 10.0, 0.0, 2000.0, match_radius_arcsec=2.0)
    with pytest.raises(AcquisitionError, match="found 0"):
        normalize_and_match_gaia([{**target, "ref_epoch": None}], 10.0, 0.0, 2000.0, match_radius_arcsec=2.0)


def test_astrometric_context_is_retained_as_measured_values():
    row = {"source_id": "123", "ra": 10.0, "dec": 0.0, "sep_arcsec": 0.1,
           "pmra": 0.0, "pmdec": 0.0, "ref_epoch": 2016.0, "ruwe": 3.68, "non_single_star": 3}
    matched, _ = normalize_and_match_gaia([row], 10.0, 0.0, 2016.0, match_radius_arcsec=2.0)
    assert matched["ruwe"] == pytest.approx(3.68) and matched["non_single_star"] == 3
    assert "ruwe_binary_probability" not in matched and "multiplicity_probability" not in matched
    missing, _ = normalize_and_match_gaia([{**row, "ruwe": None, "non_single_star": None}],
                                          10.0, 0.0, 2016.0, match_radius_arcsec=2.0)
    assert missing["ruwe"] is None and missing["non_single_star"] is None
    unflagged, _ = normalize_and_match_gaia([{**row, "non_single_star": 0}],
                                            10.0, 0.0, 2016.0, match_radius_arcsec=2.0)
    assert unflagged["non_single_star"] == 0


def test_close_matches_resolve_only_by_exact_gaia_dr3_identity():
    # Both rows are epoch-compatible; the nearer row must not win without identity evidence.
    near = {"source_id": "123", "ra": 10.0, "dec": 0.0, "sep_arcsec": 0.1,
            "pmra": 0.0, "pmdec": 0.0, "ref_epoch": 2016.0}
    far = {**near, "source_id": "789", "ra": 10.0 + 1.5 / 3600.0, "sep_arcsec": 1.5}
    with pytest.raises(AcquisitionError) as ambiguous:
        normalize_and_match_gaia([near, far], 10.0, 0.0, 2016.0, match_radius_arcsec=2.0)
    assert "found 2" in str(ambiguous.value)
    assert "123" in str(ambiguous.value) and "789" in str(ambiguous.value)
    assert [item["source_id"] for item in ambiguous.value.candidates] == ["123", "789"]
    matched, others = normalize_and_match_gaia(
        [near, far], 10.0, 0.0, 2016.0, match_radius_arcsec=2.0, expected_source_id=789)
    assert matched["source_id"] == "789"
    assert matched["target_match_evidence"] == "caller_supplied_gaia_dr3_source_id"
    assert [row["source_id"] for row in others] == ["123"]
    with pytest.raises(AcquisitionError, match="expected source 999"):
        normalize_and_match_gaia([near, far], 10.0, 0.0, 2016.0, match_radius_arcsec=2.0,
                                 expected_source_id=999)


def test_vectorized_epoch_separations_equal_the_per_source_propagation():
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from trex.acquisition.gaia_sources import gaia_position_at_epoch

    epoch = 2020.0
    moved_target = gaia_position_at_epoch(
        {"ra_deg": 10.0, "dec_deg": 0.0, "pmra_mas_per_year": 1000.0,
         "pmdec_mas_per_year": 0.0, "reference_epoch_jyear": 2016.0}, epoch)
    target = SkyCoord(moved_target[0] * u.deg, moved_target[1] * u.deg, frame="icrs")
    target_row = {"source_id": "1", "ra": 10.0, "dec": 0.0, "pmra": 1000.0, "pmdec": 0.0,
                  "ref_epoch": 2016.0, "sep_arcsec": 4.0}
    neighbour_row = {"source_id": "2", "ra": 10.01, "dec": 0.0, "pmra": 500.0, "pmdec": -250.0,
                     "ref_epoch": 2016.0, "sep_arcsec": 36.0}
    matched, others = normalize_and_match_gaia(
        [target_row, neighbour_row], moved_target[0], moved_target[1], epoch, match_radius_arcsec=1.0)
    assert matched["target_epoch_separation_arcsec"] == pytest.approx(0.0, abs=1e-9)
    neighbour_position = gaia_position_at_epoch(
        {"ra_deg": 10.01, "dec_deg": 0.0, "pmra_mas_per_year": 500.0,
         "pmdec_mas_per_year": -250.0, "reference_epoch_jyear": 2016.0}, epoch)
    expected = target.separation(SkyCoord(*neighbour_position, unit="deg", frame="icrs")).arcsec
    assert others[0]["target_epoch_separation_arcsec"] == pytest.approx(float(expected), rel=1e-12)


def test_boolean_gaia_values_are_not_numeric_measurements():
    from trex.acquisition.gaia_sources import _finite_float

    assert _finite_float(True) is None
    assert _finite_float(np.bool_(False)) is None
    assert _finite_float("1.5") == 1.5


def test_a_malformed_first_body_tries_the_second_mirror(tmp_path):
    bodies = [b"<html>truncated</html>", _response()]
    calls = []

    def retrieve(url, accept):
        calls.append(url)
        return bodies.pop(0)

    service = _transport(retrieve)
    rows, record = fetch_gaia_sources(1.0, 2.0, 60.0, tmp_path, service)
    assert len(calls) == 2 and calls[0].startswith("https://esa.example/") and calls[1].startswith("https://aip.example/")
    assert rows[0]["source_id"] == "9000000000000000001" and record["status"] == "available"
    names = [item["path"] for item in record["responses"]]
    assert names[0].endswith("_00.xml") and names[1].endswith("_01.xml")
    assert names[0][:-len("_00.xml")] == names[1][:-len("_01.xml")]
    assert (tmp_path / "gaia_cone_query.json").is_file()


def test_two_malformed_bodies_are_a_retryable_provider_failure(tmp_path):
    service = _transport(lambda *_: b"<html>truncated</html>")
    with pytest.raises(AcquisitionError, match="incomplete or invalid") as raised:
        fetch_gaia_sources(1.0, 2.0, 60.0, tmp_path, service)
    assert raised.value.retryable
    record = json.loads((tmp_path / "gaia_cone_query.json").read_text())
    assert record["status"] == "unavailable" and len(record["responses"]) == 2
