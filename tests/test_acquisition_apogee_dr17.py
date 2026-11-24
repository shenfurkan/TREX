"""Exact identity and quality checks on synthetic APOGEE responses."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from trex.acquisition.apogee_dr17 import (
    APOGEE_DR17_COLUMNS, apogee_dr17_associations, fetch_apogee_dr17, parse_apogee_dr17,
)

HEADER = ("#Table1\n" + ",".join(APOGEE_DR17_COLUMNS) + "\n").encode()


def _row(source_id):
    return f"{source_id},synthetic-star,0,-0.25,0.1,0,0\n".encode()


def test_apogee_csv_requires_the_complete_expected_rows_and_exact_ids():
    source_id = "6000000000000000001"
    rows = parse_apogee_dr17(HEADER + _row(source_id) * 2, [source_id], 2)
    assert [row["gaiaedr3_source_id"] for row in rows] == [source_id, source_id]
    assert parse_apogee_dr17(HEADER, [source_id], 0) == []
    with pytest.raises(ValueError, match="count query"):
        parse_apogee_dr17(HEADER, [source_id], 1)
    with pytest.raises(ValueError, match="unrequested"):
        parse_apogee_dr17(HEADER + _row("6000000000000000002"), [source_id], 1)
    with pytest.raises(ValueError, match="incomplete row"):
        parse_apogee_dr17(HEADER + b"6000000000000000001,synthetic", [source_id], 1)
    with pytest.raises(ValueError, match="expected SkyServer"):
        parse_apogee_dr17(b"<html>failed</html>", [source_id], 0)


def test_apogee_associations_require_a_single_usable_row():
    row = parse_apogee_dr17(HEADER + _row("123"), ["123"], 1)[0]
    assert apogee_dr17_associations(["123"], [row])["123"]["value_dex"] == -0.25
    assert apogee_dr17_associations(["123"], [])["123"]["status"] == "unavailable"
    assert "ambiguous" in apogee_dr17_associations(["123"], [row, row])["123"]["reason"]
    for change in ({"fe_h": "-9999"}, {"fe_h": "nan"}, {"fe_h_err": "0"},
                   {"starflag": "1"}, {"fe_h_flag": "1"}, {"aspcapflag": "1"}):
        assert apogee_dr17_associations(["123"], [{**row, **change}])["123"]["status"] == "unavailable"
    with pytest.raises(ValueError, match="unrequested"):
        apogee_dr17_associations(["124"], [row])


def test_apogee_batches_retain_complete_raw_responses(tmp_path):
    ids = [str(6000000000000000000 + index) for index in range(41)]
    calls = []

    def retrieve(url, accept):
        assert accept == "text/csv"
        query = parse_qs(urlsplit(url).query)["cmd"][0]
        calls.append(query)
        first_batch = ids[0] in query
        if query.startswith("SELECT COUNT"):
            return b"#Table1\nrow_count\n" + (b"1\n" if first_batch else b"0\n")
        return HEADER + (_row(ids[0]) if first_batch else b"")

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=retrieve)
    record, rows, failure = fetch_apogee_dr17(tmp_path, service, ids)
    assert failure is None and record["status"] == "available"
    assert len(calls) == 4 and len(record["responses"]) == 4 and len(rows) == 1
    assert set(record["source_queries"]) == set(ids)
    assert all((tmp_path / item["path"]).is_file() and item["uri"].startswith("https://")
               and item["retrieved_utc"].endswith("Z") for item in record["responses"])


@pytest.mark.parametrize("failure", ["transport", "truncated-count", "truncated-rows"])
def test_apogee_incomplete_responses_are_not_empty(tmp_path, failure):
    def retrieve(url, _accept):
        query = parse_qs(urlsplit(url).query)["cmd"][0]
        if query.startswith("SELECT COUNT"):
            return b"#Table1\nrow_count\n" if failure == "truncated-count" else b"#Table1\nrow_count\n1\n"
        return None if failure == "transport" else HEADER

    service = SimpleNamespace(transport_failures=[], _http_get_bytes=retrieve)
    record, rows, _ = fetch_apogee_dr17(tmp_path, service, ["123"])
    assert record["status"] == "unavailable" and rows is None
    assert record["responses"] and all((tmp_path / item["path"]).is_file() for item in record["responses"])
