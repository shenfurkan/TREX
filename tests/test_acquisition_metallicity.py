"""Metallicity composition retains source precedence and raw bindings."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from trex.acquisition.apogee_dr17 import APOGEE_DR17_COLUMNS
from trex.acquisition.metallicity import acquire_high_rungs, acquire_metallicity, compose_metallicity


def test_shboost_is_used_only_without_a_higher_measurement_and_with_binding():
    values = {"xgbdist_met_mean": 0.2, "xgbdist_met_std": 0.1,
              "xgbdist_logg_mean": 4.0, "xgb_inputflag": "XP5pGBPRPJHKsW1W2W3W4",
              "xgb_met_outputflag": "0"}
    shboost = {"123": (values, {"uri": "https://example.org", "retrieved_utc": "2026-01-01T00:00:00Z"},
                        "response.xml", "neighbour")}
    higher = {"123": {"provenance": {"source": "apogee-dr17"}}}
    assert compose_metallicity(["123"], {}, higher, shboost)["123"]["provenance"]["source"] == "apogee-dr17"
    measured = compose_metallicity(["123"], {}, {}, shboost)["123"]["provenance"]
    assert measured["source"] == "shboost-xp" and measured["binding"]["response"] == "response.xml"
    shboost["123"] = (values, {}, None, "neighbour")
    result = compose_metallicity(["123"], {}, {}, shboost)["123"]
    assert result["status"] == "unavailable" and "Andrae" in result["reason"]


def test_standalone_high_rungs_keep_apogee_and_skip_lower_rungs(tmp_path):
    calls = []

    def retrieve(url, accept):
        calls.append(url)
        assert accept == "text/csv"
        query = parse_qs(urlsplit(url).query)["cmd"][0]
        if query.startswith("SELECT COUNT"):
            return b"#Table1\nrow_count\n1\n"
        return ("#Table1\n" + ",".join(APOGEE_DR17_COLUMNS) +
                "\n123,synthetic-star,0,-0.25,0.1,0,0\n").encode()

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=retrieve)
    queries, measured, failure = acquire_high_rungs(tmp_path / "output", tmp_path / "cache", service, ["123"])
    assert failure is None and len(calls) == 2 and queries["galah"]["status"] == "not-needed"
    provenance = measured["123"]["provenance"]
    assert provenance["source"] == "apogee-dr17" and provenance["value_dex"] == -0.25
    assert (tmp_path / "output" / provenance["binding"]["response"]).is_file()


def test_standalone_metallicity_queries_shboost_only_for_unresolved_sources(tmp_path, monkeypatch):
    import json
    from trex.acquisition import metallicity, neighbor_tables

    higher = {"123": {"provenance": {"source": "apogee-dr17"}}}
    monkeypatch.setattr(metallicity, "acquire_high_rungs", lambda *_: ({}, higher, None))
    row = {"source_id": "456", "xgbdist_met_mean": 0.2, "xgbdist_met_std": 0.1,
           "xgbdist_logg_mean": 4.0, "xgb_inputflag": "XP5pGBPRPJHKsW1W2W3W4", "xgb_met_outputflag": "0"}

    def fetch(*_args, **kwargs):
        assert kwargs["source_ids"] == ["456"]
        return ({"status": "available", "source_queries": {"456": {
            "response": "raw.xml", "uri": "https://example.org/", "retrieved_utc": "2026-01-01T00:00:00Z"}}},
            [row], None)

    monkeypatch.setattr(neighbor_tables, "fetch_neighbor_table", fetch)
    result = acquire_metallicity(tmp_path / "out", tmp_path / "cache", SimpleNamespace(AIP_GAIA_TAP_URL="https://example.org/"), ["123", "456"])
    assert result["sources"]["123"]["provenance"]["source"] == "apogee-dr17"
    assert result["sources"]["456"]["provenance"]["source"] == "shboost-xp"
    assert json.loads((tmp_path / "out" / result["record_path"]).read_text()) == result


def test_high_priority_outage_never_falls_through_to_shboost(tmp_path, monkeypatch):
    from trex.acquisition import metallicity, neighbor_tables
    from trex.acquisition.transport import AcquisitionError, AcquisitionFailure

    failure = AcquisitionFailure("example.org", "http_status", False, 1, "HTTP 403", 403, "HTTPError")
    monkeypatch.setattr(metallicity, "acquire_high_rungs", lambda *_: ({"apogee": {"status": "unavailable"}}, {}, failure))
    monkeypatch.setattr(neighbor_tables, "fetch_neighbor_table", lambda *_args, **_kwargs: pytest.fail("unexpected fallback"))
    with pytest.raises(AcquisitionError, match="higher-priority") as exc:
        acquire_metallicity(tmp_path / "out", tmp_path / "cache", object(), ["123"])
    assert exc.value.network_failure.http_status == 403 and not exc.value.retryable


def test_select_high_rungs_canonicalizes_ids_before_any_rung():
    from trex.acquisition.metallicity import select_high_rungs

    seen = []

    def fetch_apogee(ids):
        seen.extend(ids)
        return {"status": "unavailable"}, None, None

    queries, measured, failure = select_high_rungs(
        ["000123"], fetch_apogee=fetch_apogee, open_galah=pytest.fail, fetch_crossmatch=pytest.fail,
        read_galah=pytest.fail, retain_galah=pytest.fail, fetch_gspspec=pytest.fail)
    assert seen == ["123"] and measured == {} and failure is None


def test_a_programming_error_while_opening_galah_is_not_a_catalogue_absence():
    from trex.acquisition.metallicity import select_high_rungs

    def fetch_apogee(_ids):
        return {"status": "available", "row_count": 0}, [], None

    with pytest.raises(RuntimeError, match="programming error"):
        select_high_rungs(["123"], fetch_apogee=fetch_apogee,
                          open_galah=lambda: (_ for _ in ()).throw(RuntimeError("programming error")),
                          fetch_crossmatch=pytest.fail, read_galah=pytest.fail,
                          retain_galah=pytest.fail, fetch_gspspec=pytest.fail)


def test_compose_reports_each_source_s_own_rung_reason():
    from trex.acquisition.metallicity import compose_metallicity

    queries = {
        "apogee": {"status": "available", "row_count": 1,
                   "source_reasons": {"123": "no APOGEE DR17 abundance row"}},
        "galah": {"status": "not-needed"},
        "gspspec": {"status": "not-needed"},
    }
    results = compose_metallicity(["123", "456"], queries, {}, {})
    assert results["123"]["reason"].startswith("apogee: no APOGEE DR17 abundance row")
    assert results["456"]["reason"].startswith("apogee: no admitted measurement (quality/association)")
    assert "Andrae" in results["456"]["reason"]


@pytest.mark.parametrize("galah_status", ["available", "unavailable"])
def test_incomplete_high_rung_blocks_shboost_but_keeps_completed_measurements(galah_status):
    higher = {"123": {"provenance": {"source": "apogee-dr17"}}}
    values = {"xgbdist_met_mean": 0.2, "xgbdist_met_std": 0.1,
              "xgbdist_logg_mean": 4.0, "xgb_inputflag": "XP5pGBPRPJHKsW1W2W3W4",
              "xgb_met_outputflag": "0"}
    lower = {"456": (values, {"uri": "https://example.org/", "retrieved_utc": "2026-01-01T00:00:00Z"},
                      "response.xml", "neighbour")}
    queries = {"apogee": {"status": "available"},
               "galah": {"status": galah_status, "reason": "synthetic malformed release"}}
    result = compose_metallicity(["123", "456"], queries, higher, lower)
    assert result["123"] == {"status": "measured", **higher["123"]}
    if galah_status == "available":
        assert result["456"]["provenance"]["source"] == "shboost-xp"
    else:
        assert result["456"]["status"] == "unavailable"
        assert "galah: synthetic malformed release" in result["456"]["reason"]
        assert "provenance" not in result["456"]


def test_shboost_refuses_boolean_values_and_an_empty_response_name():
    from trex.acquisition.metallicity import (
        SHBOOST_COMPLETE_INPUT_FLAG, SHBOOST_MAX_MET_STD_DEX, shboost_provenance,
    )

    values = {"xgbdist_met_mean": 0.2, "xgbdist_met_std": 0.1,
              "xgbdist_logg_mean": 4.0, "xgb_inputflag": SHBOOST_COMPLETE_INPUT_FLAG,
              "xgb_met_outputflag": "0"}
    query = {"uri": "https://example.org", "retrieved_utc": "2026-01-01T00:00:00Z"}
    assert shboost_provenance("123", values, query, "response.xml")["value_dex"] == 0.2
    assert shboost_provenance("123", {**values, "xgbdist_met_mean": True}, query, "response.xml") is None
    assert shboost_provenance("123", values, query, "") is None
    assert shboost_provenance("123", values, query, None) is None
    assert shboost_provenance("123", {**values, "xgbdist_met_std": SHBOOST_MAX_MET_STD_DEX},
                              query, "response.xml") is None
    assert shboost_provenance("123", {**values, "xgb_met_outputflag": "1"}, query, "response.xml") is None


def test_every_measured_source_needs_no_shboost_query(tmp_path, monkeypatch):
    from trex.acquisition import metallicity, neighbor_tables

    measured = {"123": {"provenance": {"source": "apogee-dr17"}}}
    monkeypatch.setattr(metallicity, "acquire_high_rungs", lambda *_: ({}, measured, None))
    monkeypatch.setattr(neighbor_tables, "fetch_neighbor_table", lambda *_args, **_kwargs: pytest.fail("unexpected SHBoost query"))
    result = acquire_metallicity(tmp_path / "out", tmp_path / "cache", object(), ["123"])
    assert result["status"] == "available" and result["shboost"] == {"status": "not-needed"}
