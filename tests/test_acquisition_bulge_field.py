"""Bulge/field population: every simulated row, native distance and selection.

The field population is not a renamed TESS cache. These checks pin the
lossless contract: no TESS magnitude cut, distance modulus retained, resolved
duplicates flagged rather than deleted, and a record that states its selection
depth and unmodeled completeness.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from trex.acquisition import lsst_sim
from trex.acquisition.lsst_sim import (
    BULGE_FIELD_COLUMNS,
    BULGE_POPULATION_FILE_NAME,
    BULGE_RECORD_FILE_NAME,
    acquire_bulge_field_population,
    bulge_field_from_rows,
)
from trex.populations.field_population import (
    load_bulge_field_population,
    read_bulge_field_record,
    unresolved_population,
)
from trex.acquisition.transport import NetworkPolicy

def _tap_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(BULGE_FIELD_COLUMNS)
    for row in rows:
        writer.writerow(row)
    return out.getvalue()


def _row(ra, dec, g, bp, rp, mass=0.8, logg=4.5, logte=3.70, mh=0.0, av=0.1, mu0=14.0, gc=1, label=1):
    return [ra, dec, g, bp, rp, mass, logg, logte, mh, av, mu0, gc, label]


def test_every_row_is_retained_without_a_tess_magnitude_cut():
    text = _tap_csv([
        _row(10.0, -60.0, 15.0, 16.0, 14.5),          # bright
        _row(10.001, -60.001, 22.5, 24.0, 21.0),      # fainter than TESS 21
        _row(10.002, -60.002, 19.0, 19.5, 18.2),
    ])
    csv_text, record = bulge_field_from_rows(text)
    assert record["rows_returned"] == 3
    assert record["rows_retained"] == 3
    assert record["selection"]["tess_limit_applied"] is None
    assert "not corrected" in record["selection"]["note"]
    assert record["solid_angle_sq_deg"] > 0.0

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 3
    assert float(rows[1]["gaia_gmag"]) == 22.5
    assert rows[1]["tess_mag"] != ""  # derived convenience value, not a cut


def test_distance_modulus_is_retained_and_inverted_without_parallax(tmp_path):
    text = _tap_csv([_row(10.0, -60.0, 16.0, 17.0, 15.5, mu0=14.0)])
    csv_text, _ = bulge_field_from_rows(text)
    path = tmp_path / BULGE_POPULATION_FILE_NAME
    path.write_text(csv_text, encoding="utf-8")
    population = load_bulge_field_population(path)

    assert population["distance_modulus_mag"][0] == 14.0
    assert population["distance_pc"][0] == pytest.approx(10.0 ** ((14.0 + 5.0) / 5.0))
    assert population["mh_dex"][0] == 0.0
    assert population["av_mag"][0] == 0.1


def test_resolved_source_is_flagged_not_deleted(tmp_path):
    text = _tap_csv([
        _row(10.0, -60.0, 16.0, 17.0, 15.5),     # duplicates the resolved source
        _row(10.1, -60.1, 16.0, 17.0, 15.5),     # unrelated field star
    ])
    csv_text, record = bulge_field_from_rows(
        text, resolved_sources=[("gaia-1", 10.0, -60.0)], duplicate_radius_arcsec=2.0
    )
    assert record["resolved_matches"] == 1
    assert record["unresolved_rows"] == 1
    path = tmp_path / BULGE_POPULATION_FILE_NAME
    path.write_text(csv_text, encoding="utf-8")
    population = load_bulge_field_population(path)

    # Both rows remain in the retained population.
    assert population["ra_deg"].size == 2
    assert population["resolved_match"].tolist() == [True, False]
    unresolved = unresolved_population(population)
    assert unresolved["ra_deg"].size == 1
    assert unresolved["ra_deg"][0] == pytest.approx(10.1)


def test_record_requires_selection_and_refuses_a_tess_cut(tmp_path):
    record = {
        "status": "available", "model": lsst_sim.MODEL, "provider": "NOIRLab Astro Data Lab",
        "table": lsst_sim.TABLE, "reference": lsst_sim.REFERENCE, "query_adql": "SELECT ...",
        "retrieved_utc": "2026-11-16T00:00:00Z", "ra_deg": 10.0, "dec_deg": -60.0,
        "cone_radius_deg": 0.1, "solid_angle_sq_deg": 0.1, "rows_returned": 1,
        "rows_retained": 1, "resolved_matches": 0, "unresolved_rows": 1,
        "selection": {"gaia_g_limit": 23.0, "tess_limit_applied": None, "note": "Gaia depth only"},
        "completeness": "not modeled", "magnitude_systems": {"gaia": "Gaia DR2"},
        "population_sha256": "0" * 64,
    }
    path = tmp_path / BULGE_RECORD_FILE_NAME
    path.write_text(json.dumps(record), encoding="utf-8")
    assert read_bulge_field_record(path)["rows_retained"] == 1

    record["selection"]["tess_limit_applied"] = 21.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="TESS magnitude selection"):
        read_bulge_field_record(path)


def test_acquisition_handoff_writes_every_row_and_its_record(tmp_path, monkeypatch):
    data = _tap_csv([_row(10.0, -60.0, 16.0, 17.0, 15.5), _row(10.1, -60.1, 20.0, 21.0, 19.0)])

    class _Transport:
        def __init__(self, _policy):
            self.calls = 0

        def post_form(self, *_args, **_kwargs):
            self.calls += 1
            return (b"n\n2\n" if self.calls == 1 else data.encode("utf-8"))

    monkeypatch.setattr(lsst_sim, "AcquisitionTransport", _Transport)
    population_path, record_path = acquire_bulge_field_population(
        10.0, -60.0, tmp_path, NetworkPolicy(allow_network=False),
        resolved_sources=[("gaia-1", 10.0, -60.0)],
    )
    record = read_bulge_field_record(record_path)
    assert record["rows_retained"] == 2
    assert record["resolved_matches"] == 1
    population = load_bulge_field_population(population_path)
    assert population["ra_deg"].size == 2
    assert unresolved_population(population)["ra_deg"].size == 1


def test_nonphysical_row_is_refused_not_silently_dropped():
    text = _tap_csv([_row(10.0, -60.0, 16.0, 17.0, 15.5, av=-0.1)])
    with pytest.raises(lsst_sim.TrilegalUnavailableError, match="non-physical"):
        bulge_field_from_rows(text)


def test_reader_refuses_a_non_integer_component_in_a_hand_edited_file(tmp_path):
    """The reader is a trust boundary: ``int(1.5)`` must not relabel a row."""
    from trex.acquisition.lsst_sim import BULGE_POPULATION_HEADER

    row = ["10.0", "-60.0", "16.0", "17.0", "15.5", "0.8", "4.5", "3.70",
           "0.0", "0.1", "14.0", "1.5", "1", "16.0", "", "0"]
    path = tmp_path / "population.csv"
    path.write_text(
        ",".join(BULGE_POPULATION_HEADER) + "\n" + ",".join(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integer gc"):
        load_bulge_field_population(path)
