"""The TRILEGAL LSST simulation is a recorded fallback for the background population."""

import hashlib
import json
from io import BytesIO
from urllib.parse import parse_qs

import pytest

from trex.acquisition import background_population, lsst_sim, transport, trilegal
from trex.acquisition.transport import NetworkPolicy
from trex.acquisition.trilegal_errors import TrilegalUnavailableError

HEADER = ",".join(lsst_sim.COLUMNS)
# G, BP, RP, mass, logg, logte, m_h, av, mu0, gc, label
BRIGHT = "15.0,15.5,14.5,0.9,4.5,3.75,-0.2,0.3,10.0,1,1"
FAINT_DUSTY = "23.5,24.5,22.4,0.3,5.0,3.50,0.0,1.7,14.0,1,1"  # T > 21: dropped, yet bounds Av
ROWS = HEADER + "\n" + BRIGHT + "\n" + FAINT_DUSTY + "\n"
SOUTH = (238.1082, -48.145)


def test_rows_become_the_trilegal_population_contract_with_tess_from_gaia():
    population, count, relations, av_max = lsst_sim.population_from_rows(ROWS)
    header, row = population.splitlines()
    assert header == "TESS,Mact,logg,logTe,[M/H],Av,m-M0,Gc,label"
    # Stassun et al. (2019) Eq. 1 by hand, BP - RP = 1.0:
    # 15 - 0.00522555 + 0.0891337 - 0.633923 + 0.0324473 = 14.4824325
    assert float(row.split(",")[0]) == pytest.approx(14.4824325, abs=1e-4)
    assert row.split(",")[1:5] == ["0.9", "4.5", "3.75", "-0.2"]
    assert count == 1 and relations == {"stassun2019-eq1": 1} and av_max == 1.7


def test_malformed_rows_are_refused_not_repaired():
    with pytest.raises(TrilegalUnavailableError, match="unexpected columns"):
        lsst_sim.population_from_rows("gaia_gmag,mass\n15,1\n")
    with pytest.raises(TrilegalUnavailableError, match="non-numeric mass"):
        lsst_sim.population_from_rows(HEADER + "\n" + BRIGHT.replace("0.9", "", 1) + "\n")


def test_footprint_follows_dal_tio_2022_with_a_cone_margin():
    assert lsst_sim.footprint_reason(*SOUTH) is None                      # Dec < 5
    assert lsst_sim.footprint_reason(83.8659, 21.294) is None              # plane, Dec < 35
    assert lsst_sim.footprint_reason(50.5097, 17.240) is None              # ecliptic band
    assert "footprint" in lsst_sim.footprint_reason(325.0146, 48.407)      # north, off-plane
    assert "footprint" in lsst_sim.footprint_reason(188.9226, 27.214)      # north Galactic pole region


@pytest.mark.parametrize("ra, dec", [(float("nan"), 0.0), (10.0, float("inf")), (True, 0.0),
                                     (10.0, False), (-1.0, 0.0), (10.0, 91.0)])
def test_footprint_refuses_an_invalid_field_centre(ra, dec):
    with pytest.raises(ValueError, match="valid domain"):
        lsst_sim.footprint_reason(ra, dec)


@pytest.mark.parametrize("index, value", [(9, "1.5"), (10, "2.5")])
def test_a_fractional_component_or_stage_label_is_not_truncated(index, value):
    fields = BRIGHT.split(",")
    fields[index] = value
    with pytest.raises(TrilegalUnavailableError, match="non-integer Galactic component or stage label") as raised:
        lsst_sim.population_from_rows(HEADER + "\n" + ",".join(fields) + "\n")
    assert raised.value.endpoint == "datalab.noirlab.edu"


def test_a_field_with_no_star_brighter_than_the_tess_limit_is_not_a_population():
    with pytest.raises(TrilegalUnavailableError, match="no star brighter than TESS"):
        lsst_sim.population_from_rows(HEADER + "\n" + FAINT_DUSTY + "\n")
    fields = BRIGHT.split(",")
    fields[0] = ""
    with pytest.raises(TrilegalUnavailableError, match="no Gaia G magnitude"):
        lsst_sim.population_from_rows(HEADER + "\n" + ",".join(fields) + "\n")


def _serve(monkeypatch, replies, seen):
    class Response(BytesIO):
        headers = {}

    def urlopen(request, timeout):
        seen.append((request.get_header("User-agent"), parse_qs(request.data.decode())))
        return Response(replies.pop(0).encode())

    monkeypatch.setattr(transport, "urlopen", urlopen)


def test_fallback_retains_rows_record_and_trigger(tmp_path, monkeypatch):
    seen = []
    _serve(monkeypatch, ["n\n2\n", ROWS], seen)
    policy = NetworkPolicy(True, attempts=1, user_agent=lambda: "client/9")
    trigger = {"status": "unavailable", "stage": "form-page", "error": {"message": "TLS EOF"}}
    population, record_path = lsst_sim.acquire_lsst_sim_population(*SOUTH, tmp_path, policy, trigger=trigger)
    record = json.loads(record_path.read_text())
    assert record["model"] == lsst_sim.MODEL and record["fallback"] is True
    assert record["fallback_trigger"] == trigger and record["field_av_max"] == 1.7
    assert record["star_count"] == 1 and record["rows_returned"] == 2
    assert record["simulated_area_sq_deg"] == 0.1
    assert "Gaia G < 23" in record["field_av_source"] and "raw_output_sha256" not in record
    assert background_population.record_simulates_target_field(record)
    assert not trilegal.is_galactic_field_record(record)
    assert background_population.field_attenuation_envelope(record) == 1.7
    assert (tmp_path / lsst_sim.RAW_FILE_NAME).read_text() == ROWS
    assert population.read_text().startswith("TESS,Mact")
    agents = {agent for agent, _form in seen}
    queries = [form["QUERY"][0] for _agent, form in seen]
    assert agents == {"client/9"} and "COUNT(*)" in queries[0] and "q3c_radial_query" in queries[1]


def test_an_incomplete_or_empty_answer_is_not_a_population(tmp_path, monkeypatch):
    _serve(monkeypatch, ["n\n3\n", ROWS], [])
    with pytest.raises(TrilegalUnavailableError, match="incomplete: 2 of 3"):
        lsst_sim.acquire_lsst_sim_population(*SOUTH, tmp_path, NetworkPolicy(True, attempts=1))
    _serve(monkeypatch, ["n\n0\n"], [])
    with pytest.raises(TrilegalUnavailableError, match="no star in this field"):
        lsst_sim.acquire_lsst_sim_population(*SOUTH, tmp_path, NetworkPolicy(True, attempts=1))
    assert not (tmp_path / "trilegal_query.json").exists()
    with pytest.raises(TrilegalUnavailableError, match="footprint"):
        lsst_sim.acquire_lsst_sim_population(325.0146, 48.407, tmp_path, NetworkPolicy(True, attempts=1))


def _trilegal_down(*_args):
    raise TrilegalUnavailableError("TRILEGAL refused the connection")


def test_chain_uses_trilegal_first_and_the_fallback_only_when_it_fails(tmp_path, monkeypatch):
    calls = []

    def fallback(ra_deg, dec_deg, output_dir, policy, **kwargs):
        calls.append(kwargs["trigger"])
        return "fallback-population", "fallback-record"

    monkeypatch.setattr(background_population, "acquire_lsst_sim_population", fallback)

    def good(_overrides, field_extinction):
        form = [("gal_coord", "1"),
                ("extinction_infty", "{0:.4f}".format(field_extinction["av_infinity_mag"]))]
        return (form, "https://stev.oapd.inaf.it/tmp/x.dat",
                "TESS Mact logg logTe [M/H]\n12 1 4.4 3.76 0\n#TRILEGAL normally terminated\n", None)

    population, _record = background_population.acquire_background_population(
        *SOUTH, tmp_path / "up", NetworkPolicy(True), query_runner=good)
    assert population.name == "trilegal_cache.csv" and calls == []

    assert background_population.acquire_background_population(
        *SOUTH, tmp_path / "down", NetworkPolicy(True), query_runner=_trilegal_down) == (
        "fallback-population", "fallback-record")
    assert calls[0]["status"] == "unavailable" and "refused" in calls[0]["error"]["message"]


def test_when_both_fail_the_trilegal_error_is_raised_unchanged(tmp_path):
    # conftest makes the fallback unreachable, as when the Data Lab is down.
    with pytest.raises(TrilegalUnavailableError, match="refused the connection") as raised:
        background_population.acquire_background_population(
            *SOUTH, tmp_path, NetworkPolicy(True), query_runner=_trilegal_down)
    assert "Astro Data Lab" in raised.value.fallback_failure
    stage = json.loads((tmp_path / trilegal.STAGE_RECORD_FILE_NAME).read_text())
    assert stage["fallback"]["status"] == "unavailable" and stage["stage"] == "query"


def test_any_fallback_acquisition_error_still_raises_the_original_trilegal_error(tmp_path, monkeypatch):
    from trex.acquisition.transport import AcquisitionError

    def fallback_fails(*_args, **_kwargs):
        raise AcquisitionError("fallback transport refused")

    monkeypatch.setattr(background_population, "acquire_lsst_sim_population", fallback_fails)
    with pytest.raises(TrilegalUnavailableError, match="refused the connection") as raised:
        background_population.acquire_background_population(
            *SOUTH, tmp_path, NetworkPolicy(True), query_runner=_trilegal_down)
    assert raised.value.fallback_failure == "fallback transport refused"


def test_a_stale_stage_record_does_not_describe_a_new_failure(tmp_path):
    from trex.acquisition.background_population import _trigger

    (tmp_path / trilegal.STAGE_RECORD_FILE_NAME).write_text(json.dumps({
        "status": "unavailable", "stage": "submit",
        "error": {"exception_class": "TrilegalUnavailableError", "message": "an older failure"}}))
    current = RuntimeError("this attempt's failure")
    record = _trigger(tmp_path, current)
    assert record["error"]["exception_class"] == "RuntimeError"
    assert record["error"]["message"] == "this attempt's failure"


def test_a_missing_fallback_record_is_none_but_a_broken_one_is_refused(tmp_path):
    from trex.acquisition.background_population import retained_fallback_record

    population = tmp_path / "trilegal_cache.csv"
    population.write_text("TESS\n12\n", encoding="utf-8")
    digest = hashlib.sha256(population.read_bytes()).hexdigest()
    assert retained_fallback_record(population, digest) is None

    record_path = tmp_path / "trilegal_query.json"
    record_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        retained_fallback_record(population, digest)

    record_path.write_text(json.dumps({
        "model": lsst_sim.MODEL, "fallback": True, "field_submission": lsst_sim.FIELD_SUBMISSION,
        "population_sha256": digest}), encoding="utf-8")
    assert retained_fallback_record(population, digest)["model"] == lsst_sim.MODEL

    record_path.write_text(json.dumps({
        "model": lsst_sim.MODEL, "fallback": True, "field_submission": lsst_sim.FIELD_SUBMISSION}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="does not name the retained population"):
        retained_fallback_record(population, digest)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, None])
def test_the_field_attenuation_envelope_refuses_non_finite_or_negative_values(value):
    from trex.acquisition.background_population import field_attenuation_envelope

    with pytest.raises(ValueError):
        field_attenuation_envelope({"model": lsst_sim.MODEL, "field_av_max": value})
