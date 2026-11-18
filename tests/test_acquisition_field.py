"""TIC and Gaia acquisition connect through a unique epoch-aware match."""

import json

import pytest

from trex.acquisition import field
from trex.acquisition.transport import AcquisitionError, AcquisitionFailure, NetworkPolicy


def test_field_context_retains_association_to_both_raw_queries(tmp_path, monkeypatch):
    tic = {"ID": "123", "ra": 10.0, "dec": 0.0}
    rows = [{"source_id": "456", "ra": 10.0, "dec": 0.0, "sep_arcsec": 0.0,
             "pmra": 0.0, "pmdec": 0.0, "ref_epoch": 2016.0}]
    monkeypatch.setattr(field, "fetch_tic_target", lambda *_args: (tic, {"response": "tic.json"}))
    monkeypatch.setattr(field, "fetch_gaia_sources", lambda *_args: (rows, {"response": "gaia.xml"}))
    monkeypatch.setattr(field, "acquire_tic_neighbors", lambda *_args, **_kwargs: {"neighbors": {}})
    got_tic, target, neighbors, queries = field.acquire_field_context(
        "123", tmp_path, search_radius_arcsec=60.0, policy=NetworkPolicy(False))
    assert got_tic == tic and target["source_id"] == "456" and neighbors == []
    assert queries["tic"]["response"] == "tic.json" and queries["gaia"]["response"] == "gaia.xml"
    record = json.loads((tmp_path / "field_context.json").read_text())
    assert record["target_coordinate_epoch_jyear"] == 2000.0 and record["target"]["source_id"] == "456"


def test_ambiguous_gaia_matches_never_write_a_field_context(tmp_path, monkeypatch):
    monkeypatch.setattr(field, "fetch_tic_target", lambda *_args: ({"ID": "123", "ra": 10.0, "dec": 0.0}, {}))
    row = {"source_id": "456", "ra": 10.0, "dec": 0.0, "sep_arcsec": 0.0}
    monkeypatch.setattr(field, "fetch_gaia_sources", lambda *_args: ([row, {**row, "source_id": "789"}], {}))
    with pytest.raises(AcquisitionError, match="found 2"):
        field.acquire_field_context("123", tmp_path, search_radius_arcsec=60.0, policy=NetworkPolicy(False))
    assert not (tmp_path / "field_context.json").exists()


def test_neighbor_context_keeps_unique_and_ambiguous_crossmatches(tmp_path, monkeypatch):
    def link(dr3, dr2):
        return {"dr3_source_id": dr3, "dr2_source_id": dr2, "angular_distance": 1.0,
                "magnitude_difference": None, "proper_motion_propagation": True}

    rows = [link("123", "456"), link("789", "900"), link("789", "901")]
    monkeypatch.setattr(field, "query_gaia_neighborhood", lambda *_args: ({"status": "available"}, rows, None))
    calls = []

    def tic(dr2, _output, _transport):
        calls.append(dr2)
        return [{"ID": 10, "GAIA": dr2, "mass": 0.8, "rad": 0.7}], {"response": "raw.json"}

    monkeypatch.setattr(field, "fetch_tic_by_gaia", tic)
    monkeypatch.setattr(field, "fetch_tic_cone", lambda *_args: ([], {"response": "cone.json"}))
    target = {"ra_deg": 10.0, "dec_deg": 0.0}
    neighbors = [{"source_id": "123", "separation_arcsec": 3.0},
                 {"source_id": "789", "separation_arcsec": 4.0}]
    result = field.acquire_tic_neighbors(target, neighbors, tmp_path, policy=NetworkPolicy(False))
    assert calls == ["456"] and result["neighbors"]["123"]["status"] == "available"
    assert result["neighbors"]["789"]["status"] == "unavailable"
    assert "ambiguous" in result["neighbors"]["789"]["reason"]
    assert json.loads((tmp_path / "tic_neighbors_query.json").read_text()) == result


def test_neighbor_crossmatch_outage_is_not_treated_as_empty(tmp_path, monkeypatch):
    failure = type("Failure", (), {"retryable": True, "reason": "synthetic transport outage"})()
    monkeypatch.setattr(field, "query_gaia_neighborhood", lambda *_args: ({"status": "unavailable"}, None, failure))
    with pytest.raises(AcquisitionError, match="neighbourhood query failed.*synthetic transport outage") as exc:
        field.acquire_tic_neighbors({}, [{"source_id": "123"}], tmp_path, policy=NetworkPolicy(False))
    assert exc.value.retryable and json.loads((tmp_path / "tic_neighbors_query.json").read_text())["status"] == "unavailable"


def test_a_tic_row_without_coordinates_is_recorded_before_the_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(field, "fetch_tic_target", lambda *_args: ({"ID": "123"}, {}))
    monkeypatch.setattr(field, "fetch_gaia_sources", lambda *_args: pytest.fail("no cone without a TIC position"))
    with pytest.raises(AcquisitionError, match="lacks ICRS coordinates"):
        field.acquire_field_context("123", tmp_path, search_radius_arcsec=60.0, policy=NetworkPolicy(False))
    record = json.loads((tmp_path / "field_context.json").read_text())
    assert record == {"status": "unavailable", "tic_id": "123",
                      "reason": "TIC target row lacks ICRS coordinates"}


def test_neighbor_context_uses_the_given_centre_and_transport(tmp_path, monkeypatch):
    row = {"dr3_source_id": "123", "dr2_source_id": "456", "angular_distance": 1.0,
           "magnitude_difference": None, "proper_motion_propagation": True}
    sentinel = object()
    seen = {}

    def query(output_dir, transport, field_name, ids, retrieval_id):
        assert transport is sentinel
        assert field_name == "dr3_source_id" and ids == ["123"]
        return {"status": "available"}, [row], None

    monkeypatch.setattr(field, "query_gaia_neighborhood", query)

    def cone(ra_deg, dec_deg, radius, _output, transport):
        seen.update(ra=ra_deg, dec=dec_deg, radius=radius, transport=transport)
        return [], {"response": "cone.json"}

    monkeypatch.setattr(field, "fetch_tic_cone", cone)
    monkeypatch.setattr(field, "fetch_tic_by_gaia", lambda *_args: (
        [{"ID": 10, "GAIA": "456", "mass": 0.8, "rad": 0.7}], {"response": "raw.json"}))
    result = field.acquire_tic_neighbors({"ra_deg": 10.0, "dec_deg": 0.0},
                                         [{"source_id": "123", "separation_arcsec": 3.0}],
                                         tmp_path, policy=NetworkPolicy(False), centre=(1.0, 2.0),
                                         transport=sentinel)
    assert seen == {"ra": 1.0, "dec": 2.0, "radius": 63.0, "transport": sentinel}
    assert result["neighbors"]["123"]["status"] == "available"


def test_neighbor_cone_rows_avoid_slow_per_source_lookup(tmp_path, monkeypatch):
    row = {"dr3_source_id": "123", "dr2_source_id": "456", "angular_distance": 1.0,
           "magnitude_difference": None, "proper_motion_propagation": True}
    monkeypatch.setattr(field, "query_gaia_neighborhood", lambda *_args: ({"status": "available"}, [row], None))
    monkeypatch.setattr(field, "fetch_tic_cone", lambda *args: (
        [{"ID": 10, "GAIA": "456", "mass": 0.8, "rad": 0.7}], {"response": "cone.json"}))
    monkeypatch.setattr(field, "fetch_tic_by_gaia", lambda *_args: pytest.fail("unexpected slow lookup"))
    result = field.acquire_tic_neighbors({"ra_deg": 10.0, "dec_deg": 0.0},
                                         [{"source_id": "123", "separation_arcsec": 3.0}],
                                         tmp_path, policy=NetworkPolicy(False))
    assert result["neighbors"]["123"]["query"]["response"] == "cone.json"


def test_neighbor_cone_access_denial_never_falls_through(tmp_path, monkeypatch):
    row = {"dr3_source_id": "123", "dr2_source_id": "456", "angular_distance": 1.0,
           "magnitude_difference": None, "proper_motion_propagation": True}
    monkeypatch.setattr(field, "query_gaia_neighborhood", lambda *_args: ({"status": "available"}, [row], None))
    failure = AcquisitionFailure("mast.stsci.edu", "http_status", False, 1, "HTTP 403", 403, "HTTPError")
    history = [{"attempt": 1, "recorded_utc": "2026-11-12T00:00:00Z",
                "failure": failure.to_manifest(), "response": "tic_cone_attempt_1.json"}]

    def denied(*_args):
        error = AcquisitionError("MAST TIC neighbour query failed")
        error.network_failure = failure
        error.attempt_history = history
        raise error

    monkeypatch.setattr(field, "fetch_tic_cone", denied)
    monkeypatch.setattr(field, "fetch_tic_by_gaia", lambda *_args: pytest.fail("access denial must not fall through"))
    with pytest.raises(AcquisitionError, match="query failed"):
        field.acquire_tic_neighbors({"ra_deg": 10.0, "dec_deg": 0.0},
                                    [{"source_id": "123", "separation_arcsec": 3.0}],
                                    tmp_path, policy=NetworkPolicy(False))
    record = json.loads((tmp_path / "tic_neighbors_query.json").read_text())
    assert record["status"] == "unavailable" and record["cone_failure"]["http_status"] == 403
    assert record["cone_attempt_history"] == history
