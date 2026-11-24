"""Offline end-to-end target-scene assembly with fake stage acquirers.

Synthetic records: one target and two neighbours. Expected scene values are
derived by hand from the inputs: host physics uses the GSP-Phot branch with
the FLAME mass taking precedence over ``g R**2 / G`` and TIC v8, the target
TESS magnitude is the Stassun et al. (2019) Eq. (2) ``G - 0.430`` for no
valid colour, neighbours take their FLAME mass/radius, and aperture shares
are the fake contributions. A rerun after a TRILEGAL failure must revalidate
the retained TPF products, reuse the other ``"available"`` stages and call
only TRILEGAL.
"""

import hashlib
import json
from pathlib import Path

import pytest

from trex.acquisition import scene as scene_module
from trex.acquisition.aperture import catalog_source_contributions, catalog_tess_magnitudes
from trex.acquisition.scene import acquire_target_scene
from trex.acquisition.storage import write_json
from trex.acquisition.transport import AcquisitionError
from trex.acquisition.trilegal import FIELD_SUBMISSION
from trex.acquisition.trilegal_errors import TrilegalUnavailableError
from trex.target import target

TIC_ID = 999999001
TARGET_ID = "1111111111111111111"
NEIGHBOR1_ID = "2222222222222222222"
NEIGHBOR2_ID = "3333333333333333333"

TARGET = {
    "source_id": TARGET_ID, "ra_deg": 10.0, "dec_deg": 20.0,
    "phot_g_mean_mag": 10.0, "phot_bp_mean_mag": None, "phot_rp_mean_mag": None,
    "parallax_mas": 20.0,
}
NEIGHBOR1 = {
    "source_id": NEIGHBOR1_ID, "ra_deg": 10.0, "dec_deg": 20.0 + 10.0 / 3600.0,
    "phot_g_mean_mag": 12.0, "phot_bp_mean_mag": 12.5, "phot_rp_mean_mag": 11.5,
    "parallax_mas": 15.0, "separation_arcsec": 11.0,  # from the cone centre, not the target
}
NEIGHBOR2 = {
    "source_id": NEIGHBOR2_ID, "ra_deg": 10.0, "dec_deg": 20.0 + 20.0 / 3600.0,
    "phot_g_mean_mag": 13.5, "phot_bp_mean_mag": 14.0, "phot_rp_mean_mag": 13.0,
    "parallax_mas": 12.0, "separation_arcsec": 21.0,
}
# Hand expectations: GSP-Phot Teff/radius with FLAME mass (0.95 Msun).
TARGET_AP = {
    "teff_gspphot_k": 5600.0, "logg_gspphot_cgs": 4.3,
    "radius_gspphot_solar": 1.1, "mass_flame_solar": 0.95,
    "radius_flame_solar": 1.1, "flags_flame": "00",
}
NEIGHBOR1_AP = {
    "mass_flame_solar": 0.8, "radius_flame_solar": 0.75,
    "flags_flame": "00", "evolstage_flame": 2,
}
NEIGHBOR2_AP = {
    "mass_flame_solar": 0.6, "radius_flame_solar": 0.55,
    "flags_flame": "00", "evolstage_flame": 2,
}


def _provenance(value):
    return {"source": "caller-supplied", "quantity": "[M/H]", "value_dex": value,
            "uncertainty_dex": 0.2, "citation": "synthetic test source",
            "uri": "https://example.org/synthetic"}


def _field_record():
    return {"status": "available", "tic_id": str(TIC_ID),
            "target": dict(TARGET), "neighbors": [dict(NEIGHBOR1), dict(NEIGHBOR2)]}


def fake_setup(tmp_path, calls, trilegal_fails=False, reach_arcsec=45.0, field_submission=True):
    """Install fake stage acquirers writing real-format retained records."""
    def fake_tpf(tic_id, sectors, output_dir, transport):
        calls.append("tpf")
        products = [{"sector": sector, "path": f"tpf_s{sector}.fits"} for sector in sectors]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(output_dir) / "tpf_products.json", {"status": "available", "products": products})
        return products

    def fake_field(tic_id, output_dir, *, search_radius_arcsec, policy):
        calls.append(("field", search_radius_arcsec))
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tic_row = {"ID": TIC_ID, "Teff": 5000.0, "rad": 0.9, "mass": 0.8, "logg": 4.5}
        write_json(output / "tic_target_query.json", {"status": "available"})
        write_json(output / "gaia_cone_query.json", {"status": "available"})
        write_json(output / "tic_target.json", {"status": "COMPLETE", "data": [tic_row]})
        write_json(output / "tic_neighbors_query.json", {"status": "available", "neighbors": {}})
        write_json(output / "field_context.json", _field_record())
        return tic_row, dict(TARGET), [dict(NEIGHBOR1), dict(NEIGHBOR2)], {
            "tic": {}, "gaia": {}, "tic_neighbors": {"status": "available", "neighbors": {}}}

    def fake_host(output_dir, service, source_ids):
        calls.append("host")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        queries = {label: {"status": "available", "table": "synthetic", "uri": "https://example.org/",
                           "backend": "test", "source_queries": {}}
                   for label in ("gaia", "starhorse", "shboost")}
        record = {"status": "available", "source_ids": [TARGET_ID, NEIGHBOR1_ID, NEIGHBOR2_ID],
                  "queries": queries, "sources": {
            TARGET_ID: {"gaia": dict(TARGET_AP), "starhorse": None, "shboost": None},
            NEIGHBOR1_ID: {"gaia": dict(NEIGHBOR1_AP), "starhorse": None, "shboost": None},
            NEIGHBOR2_ID: {"gaia": dict(NEIGHBOR2_AP), "starhorse": None, "shboost": None}}}
        write_json(output / "host_tables_fake.json", record)
        return record

    def fake_aperture(tic_id, sectors, target, neighbors, output_dir, transport, products=None):
        assert products is not None, "the scene must reuse the TPFs it already retained"
        calls.append("aperture")
        sources = [target, *neighbors]
        contributions = [
            {"source_id": row["source_id"], "aperture_flux_fraction": fraction,
             "tess_mag": magnitude, "tess_mag_relation": relation}
            for row, (magnitude, relation), fraction in zip(
                sources, catalog_tess_magnitudes(sources), (0.80, 0.15, 0.05), strict=True)]
        record = {"status": "available", "method": "synthetic-catalog",
                  "limitation": "synthetic limitation",
                  "sectors": [{"sector": sector, "source_contributions": contributions}
                              for sector in sectors]}
        write_json(Path(output_dir) / "aperture_scene.json", record)
        return record

    def fake_metallicity(output_dir, cache_dir, service, source_ids):
        calls.append("metallicity")
        record = {"status": "available", "source_ids": list(source_ids), "sources": {
            TARGET_ID: {"status": "measured", "provenance": _provenance(-0.10)},
            NEIGHBOR1_ID: {"status": "measured", "provenance": _provenance(-0.20)},
            NEIGHBOR2_ID: {"status": "measured", "provenance": _provenance(-0.30)}}}
        write_json(Path(output_dir) / "metallicity_query_fake.json", record)
        return record

    def fake_trilegal(ra_deg, dec_deg, output_dir, policy):
        calls.append("trilegal")
        if trilegal_fails:
            raise TrilegalUnavailableError("synthetic TRILEGAL outage")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        csv = ("TESS,Mact,logg,logTe,[M/H]\n12.0,0.5,4.5,3.7,-0.1\n"
               "12.5,0.4,4.6,3.6,-0.2\n13.0,0.3,4.7,3.5,-0.3\n")
        population = output / "trilegal_cache.csv"
        population.write_text(csv)
        record = {"status": "available", "population_file": "trilegal_cache.csv",
                  "population_sha256": hashlib.sha256(csv.encode()).hexdigest(),
                  "submitted_form": [["extinction_infty", "0.02"]],
                  "star_count": 3, "simulated_area_sq_deg": 0.1}
        if field_submission:
            record["field_submission"] = FIELD_SUBMISSION
        write_json(output / "trilegal_query.json", record)
        return population, output / "trilegal_query.json"

    return {"acquire_tpf_products": fake_tpf, "load_tpf": lambda path, tic_id, sector: {"sector": sector},
            "aperture_reach_arcsec": lambda cube: reach_arcsec,
            "acquire_field_context": fake_field, "acquire_host_tables": fake_host,
            "acquire_catalog_aperture": fake_aperture, "acquire_metallicity": fake_metallicity,
            "acquire_background_population": fake_trilegal}


def test_scene_resume_revalidates_tpfs_and_calls_only_trilegal(tmp_path, monkeypatch):
    output = tmp_path / "scene"
    calls = []
    for name, fake in fake_setup(tmp_path, calls, trilegal_fails=True).items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(TrilegalUnavailableError):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    request = json.loads((output / "scene_request.json").read_text())
    assert request == {"tic_id": TIC_ID, "sectors": [1], "depth_ppm": 10000.0,
                       "radius": 60.0, "trilegal_population": None}
    assert calls == ["tpf", ("field", 60.0), "host", "aperture", "metallicity", "trilegal"]

    calls.clear()
    fakes = fake_setup(tmp_path, calls, trilegal_fails=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("available stage must be reused, not reacquired")

    for name in ("acquire_field_context", "acquire_host_tables",
                 "acquire_catalog_aperture", "acquire_metallicity"):
        monkeypatch.setattr(scene_module, name, forbidden)
    monkeypatch.setattr(scene_module, "acquire_tpf_products", fakes["acquire_tpf_products"])
    monkeypatch.setattr(scene_module, "acquire_background_population", fakes["acquire_background_population"])
    scene = acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    # The TPF stage is always rerun through the acquisition owner so retained
    # products are revalidated rather than trusted from tpf_products.json.
    assert calls == ["tpf", "trilegal"]

    assert scene.tic_id == TIC_ID and scene.sectors == [1]
    assert scene.M_s_Msun == 0.95 and scene.R_s_Rsun == 1.1 and scene.Teff_K == 5600.0
    assert scene.Tmag == 10.0 - 0.430  # Stassun Eq. (2), no valid colour
    assert scene.plx_mas == 20.0 and scene.feh_dex == -0.10
    assert scene.target_aperture_flux_fractions == {1: 0.80}
    assert scene.N_background == 3 and scene.background_area_sq_deg == 0.1
    by_id = {item["source_id"]: item for item in scene.resolved_neighbors}
    assert by_id[NEIGHBOR1_ID]["M_s"] == 0.8 and by_id[NEIGHBOR1_ID]["R_s"] == 0.75
    # Eq. (1) at BP-RP=1 gives T=G-0.51756755, unlike the raw Gaia G contrast.
    assert by_id[NEIGHBOR1_ID]["delta_mag"] == pytest.approx(1.91243245, abs=1e-12)
    assert by_id[NEIGHBOR1_ID]["separation_arcsec"] == pytest.approx(10.0, abs=1e-6)  # pure declination offset
    assert by_id[NEIGHBOR1_ID]["aperture_flux_fractions"] == {1: 0.15}
    assert by_id[NEIGHBOR1_ID]["metallicity_provenance"]["value_dex"] == -0.20
    assert by_id[NEIGHBOR2_ID]["M_s"] == 0.6 and by_id[NEIGHBOR2_ID]["R_s"] == 0.55
    assert by_id[NEIGHBOR2_ID]["delta_mag"] == pytest.approx(3.41243245, abs=1e-12)
    assert by_id[NEIGHBOR2_ID]["separation_arcsec"] == pytest.approx(20.0, abs=1e-6)
    assert by_id[NEIGHBOR2_ID]["metallicity_provenance"]["value_dex"] == -0.30
    bound = target.from_scene(scene, sector_weights={1: 1.0}, depth_ppm=10000.0, exptime_days=0.001)
    assert bound.stars.set_index("ID").loc[NEIGHBOR1_ID, "Tmag"] == pytest.approx(11.48243245, abs=1e-12)


def test_scene_resume_fails_before_downstream_stages_when_retained_tpf_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "scene"
    calls = []
    for name, fake in fake_setup(tmp_path, calls).items():
        monkeypatch.setattr(scene_module, name, fake)
    acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    assert calls == ["tpf", ("field", 60.0), "host", "aperture", "metallicity", "trilegal"]

    calls.clear()

    def reject(tic_id, sectors, output_dir, transport):
        calls.append(("tpf-validation", tic_id, tuple(sectors)))
        assert (Path(output_dir) / "tpf_products.json").is_file()
        raise AcquisitionError("retained TPF failed identity and sector validation")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("downstream stages must not reuse products that failed validation")

    monkeypatch.setattr(scene_module, "acquire_tpf_products", reject)
    for name in ("acquire_field_context", "acquire_host_tables", "acquire_catalog_aperture",
                 "acquire_metallicity", "acquire_background_population"):
        monkeypatch.setattr(scene_module, name, forbidden)
    with pytest.raises(AcquisitionError, match="identity and sector validation"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    assert calls == [("tpf-validation", TIC_ID, (1,))]


def test_scene_resume_rejects_different_request(tmp_path, monkeypatch):
    output = tmp_path / "scene"
    calls = []
    for name, fake in fake_setup(tmp_path, calls, trilegal_fails=True).items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(TrilegalUnavailableError):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    with pytest.raises(ValueError, match="scene request differs"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=5000.0)


def test_gaia_cone_widens_to_reach_every_star_that_can_light_the_aperture(tmp_path, monkeypatch):
    # A 17-pixel SPOC aperture can need 133 arcsec (2026-11-08, TOI-181 b); a fixed 60 arcsec cone
    # would drop stars whose PSF still reaches it.
    calls = []
    for name, fake in fake_setup(tmp_path, calls, reach_arcsec=133.0).items():
        monkeypatch.setattr(scene_module, name, fake)
    acquire_target_scene(TIC_ID, [1], tmp_path / "scene", depth_ppm=10000.0)
    assert ("field", 133.0) in calls
    record = json.loads((tmp_path / "scene" / "scene_acquisition.json").read_text())
    assert record["gaia_cone_radius_arcsec"] == 133.0 and record["aperture_reach_arcsec"] == 133.0


def test_a_population_simulated_with_the_hour_angle_form_values_is_never_bound(tmp_path, monkeypatch):
    # Before the Galactic field submission the form's eq_alpha received the RA
    # in degrees, which TRILEGAL reads in hours: that population is another field.
    calls = []
    old = fake_setup(tmp_path, calls, field_submission=False)
    population, query = old["acquire_background_population"](TARGET["ra_deg"], TARGET["dec_deg"], tmp_path / "old", None)
    for name, fake in fake_setup(tmp_path, calls).items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(ValueError, match="predates the Galactic field submission"):
        acquire_target_scene(TIC_ID, [1], tmp_path / "given", depth_ppm=10000.0, trilegal_population=query)

    # A scene directory that already holds such a population cannot be resumed.
    resumed = tmp_path / "resumed"
    for name, fake in fake_setup(tmp_path, calls, trilegal_fails=True).items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(TrilegalUnavailableError):
        acquire_target_scene(TIC_ID, [1], resumed, depth_ppm=10000.0)
    old["acquire_background_population"](TARGET["ra_deg"], TARGET["dec_deg"], resumed / "background", None)
    with pytest.raises(ValueError, match="new output_dir"):
        acquire_target_scene(TIC_ID, [1], resumed, depth_ppm=10000.0)


def test_a_fallback_population_is_not_reused_when_the_scene_is_rerun(tmp_path, monkeypatch):
    # The fallback stands in for one attempt only: a rerun asks TRILEGAL again.
    from trex.acquisition import lsst_sim

    output = tmp_path / "scene"
    calls = []
    for name, fake in fake_setup(tmp_path, calls, trilegal_fails=True).items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(TrilegalUnavailableError):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    background = output / "background"
    background.mkdir(parents=True, exist_ok=True)
    population = background / "trilegal_cache.csv"
    population.write_text("TESS,Mact,logg,logTe,[M/H]\n12.0,0.5,4.5,3.7,-0.1\n")
    write_json(background / "trilegal_query.json", {
        "status": "available", "model": lsst_sim.MODEL, "fallback": True,
        "field_submission": lsst_sim.FIELD_SUBMISSION, "population_file": "trilegal_cache.csv",
        "population_sha256": hashlib.sha256(population.read_bytes()).hexdigest(),
        "simulated_area_sq_deg": 0.1, "field_av_max": 0.4, "star_count": 1})
    calls.clear()
    for name, fake in fake_setup(tmp_path, calls).items():
        monkeypatch.setattr(scene_module, name, fake)
    scene = acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)
    assert calls[-1] == "trilegal" and scene.N_background == 3


def test_a_measured_contrast_curve_is_requested_bound_and_stated_in_the_acquisition_record(tmp_path, monkeypatch):
    # The curve must reach the retained records, not only the returned scene: a resumed directory
    # that silently dropped or switched it would bound unresolved companions by a different limit.
    curve = tmp_path / "speckle.dat"
    curve.write_text("# sep_arcsec dmag\n0.1 2.0\n0.5 4.5\n1.2 6.0\n", encoding="utf-8")
    output = tmp_path / "scene"
    calls = []
    for name, fake in fake_setup(tmp_path, calls).items():
        monkeypatch.setattr(scene_module, name, fake)
    scene = acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0, contrast_curve=curve, contrast_band="tess")
    assert scene.contrast_separations.tolist() == [0.1, 0.5, 1.2]
    assert scene.contrast_values.tolist() == [2.0, 4.5, 6.0]
    request = json.loads((output / "scene_request.json").read_text())
    assert request["contrast_curve"]["band"] == "TESS" and request["contrast_curve"]["source"] == str(curve)
    record = json.loads((output / "scene_acquisition.json").read_text())
    assert record["scene"]["contrast_values"] == [2.0, 4.5, 6.0]
    assert not any("No high-resolution contrast curve" in line for line in record["limitations"])
    assert any("measured TESS-band contrast curve" in line for line in record["limitations"])
    curve.write_text("0.1 1.0\n0.5 3.0\n1.2 5.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scene request differs"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0, contrast_curve=curve, contrast_band="TESS")
    with pytest.raises(ValueError, match="scene request differs"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0)


def test_a_malformed_contrast_curve_is_refused_before_any_request_is_written(tmp_path):
    output = tmp_path / "scene"
    with pytest.raises(ValueError, match="separations must be positive and strictly increasing"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0,
                             contrast_curve=([0.5, 0.1], [3.0, 2.0]), contrast_band="TESS")
    with pytest.raises(ValueError, match="contrast_band is required"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0, contrast_curve=([0.1, 0.5], [2.0, 3.0]))
    with pytest.raises(ValueError, match="TESS band only"):
        acquire_target_scene(TIC_ID, [1], output, depth_ppm=10000.0,
                             contrast_curve=([0.1, 0.5], [2.0, 3.0]), contrast_band="K")
    assert not (output / "scene_request.json").exists()


@pytest.mark.parametrize("sectors, radius", [([], 60.0), ([1, 1], 60.0), ([1], 0.0),
                                             ([1], -1.0), ([1], float("nan")), ([1], True)])
def test_an_invalid_request_is_refused_before_any_record_is_written(tmp_path, sectors, radius):
    output = tmp_path / "scene"
    with pytest.raises(ValueError):
        acquire_target_scene(TIC_ID, sectors, output, depth_ppm=10000.0, search_radius_arcsec=radius)
    assert not (output / "scene_request.json").exists()


@pytest.mark.parametrize("gaia", [{}, {"mass_flame_solar": None, "radius_flame_solar": 3.0, "flags_flame": "10"}])
@pytest.mark.parametrize("retained_without_upper", [False, True])
def test_colourless_neighbour_bound_reaches_characterization_and_evidence(tmp_path, monkeypatch, gaia,
                                                                         retained_without_upper):
    output = tmp_path / "scene"
    calls = []
    fakes = fake_setup(tmp_path, calls, trilegal_fails=retained_without_upper)
    monkeypatch.setitem(NEIGHBOR2, "phot_bp_mean_mag", None)
    original_host = fakes["acquire_host_tables"]
    original_aperture = fakes["acquire_catalog_aperture"]

    def host_with_starhorse(output_dir, service, source_ids):
        record = original_host(output_dir, service, source_ids)
        record["sources"][NEIGHBOR2_ID] = {
            "gaia": gaia,
            "starhorse": {"mass_solar": 1.4, "radius_solar": 3.0, "astrometric_fidelity": 0.9,
                          "sh_outflag": "0000", "sh_photoflag": "GBPRP"}}
        write_json(Path(output_dir) / "host_tables_fake.json", record)
        return record

    def aperture_with_bound(tic_id, sectors, target, neighbors, output_dir, transport, products=None):
        record = original_aperture(tic_id, sectors, target, neighbors, output_dir, transport, products=products)
        rows = record["sectors"][0]["source_contributions"]
        positions = [
            {"source_id": row["source_id"], "x_pixel": 0.0, "y_pixel": 0.0, "is_target": index == 0}
            for index, row in enumerate(rows)
        ]
        magnitudes = [(row["tess_mag"], row["tess_mag_relation"]) for row in rows]
        fluxes = [row["aperture_flux_fraction"] for row in rows]
        record["sectors"][0]["source_contributions"], _ = catalog_source_contributions(
            [[True]], positions, magnitudes, fluxes, sectors[0])
        write_json(Path(output_dir) / "aperture_scene.json", record)
        return record

    fakes["acquire_host_tables"] = host_with_starhorse
    fakes["acquire_catalog_aperture"] = aperture_with_bound
    for name, fake in fakes.items():
        monkeypatch.setattr(scene_module, name, fake)
    if retained_without_upper:
        with pytest.raises(TrilegalUnavailableError, match="synthetic TRILEGAL outage"):
            acquire_target_scene(TIC_ID, [1], output, depth_ppm=100000.0)
        path = output / "aperture" / "aperture_scene.json"
        record = json.loads(path.read_text())
        del record["sectors"][0]["source_contributions"][-1]["aperture_flux_fraction_upper_95"]
        write_json(path, record)
        calls.clear()
        monkeypatch.setattr(scene_module, "acquire_background_population",
                            fake_setup(tmp_path, calls)["acquire_background_population"])
    scene = acquire_target_scene(TIC_ID, [1], output, depth_ppm=100000.0)
    if retained_without_upper:
        assert calls == ["tpf", "trilegal"]

    by_id = {item["source_id"]: item for item in scene.resolved_neighbors}
    assert by_id[NEIGHBOR2_ID]["M_s"] == 1.4
    # Equal throughputs make the nominal share 0.05; brighten only this source by 3.19.
    expected_upper = (3.19 * 0.05) / (0.95 + 3.19 * 0.05)
    evidence = {row["source_id"]: row for row in scene.neighbor_dicts_for_evidence({1: 1.0})}
    assert evidence[NEIGHBOR2_ID]["aperture_flux_fraction"] == pytest.approx(expected_upper, rel=0.002)
    assert evidence[NEIGHBOR2_ID]["aperture_flux_fraction_joint_95"] == pytest.approx(0.05)
    assert evidence[NEIGHBOR1_ID]["aperture_flux_fraction"] == pytest.approx(0.15)
    evaluations = scene.neighbor_parameter_evaluations[NEIGHBOR2_ID]
    assert any(item["source"] == "starhorse-edr3" and item["status"] == "accepted" for item in evaluations)
    if gaia:
        assert any(item["source"] == "gaia-dr3-flame" and item["status"] == "refused"
                   and "mass_flame" in item["reason"] for item in evaluations)


def test_aperture_flux_fractions_must_cover_every_requested_sector(tmp_path, monkeypatch):
    output = tmp_path / "scene"
    calls = []
    fakes = fake_setup(tmp_path, calls)
    original_aperture = fakes["acquire_catalog_aperture"]

    def one_sector_only(tic_id, sectors, target, neighbors, output_dir, transport, products=None):
        record = original_aperture(tic_id, sectors, target, neighbors, output_dir, transport, products=products)
        record["sectors"] = record["sectors"][:1]
        return record

    fakes["acquire_catalog_aperture"] = one_sector_only
    for name, fake in fakes.items():
        monkeypatch.setattr(scene_module, name, fake)
    with pytest.raises(AcquisitionError, match="cover every Gaia source in every requested sector"):
        acquire_target_scene(TIC_ID, [1, 2], output, depth_ppm=10000.0)


def test_scene_resume_refuses_inconsistent_aperture_magnitude(tmp_path, monkeypatch):
    for name, fake in fake_setup(tmp_path, [], trilegal_fails=True).items():
        monkeypatch.setattr(scene_module, name, fake)
    output = tmp_path / "scene"
    with pytest.raises(TrilegalUnavailableError):
        acquire_target_scene(TIC_ID, [1, 2], output, depth_ppm=10000.0)
    path = output / "aperture/aperture_scene.json"
    record = json.loads(path.read_text())
    record["sectors"][1]["source_contributions"][1]["tess_mag"] += 0.1
    write_json(path, record)

    with pytest.raises(AcquisitionError, match=f"TESS magnitude of Gaia source {NEIGHBOR1_ID}"):
        acquire_target_scene(TIC_ID, [1, 2], output, depth_ppm=10000.0)


def test_target_separations_match_the_per_neighbour_projection():
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from trex.acquisition.scene import _separations_from_target

    target = {"ra_deg": 10.0, "dec_deg": 20.0}
    neighbours = [{"ra_deg": 10.0, "dec_deg": 20.0 + 10.0 / 3600.0},
                  {"ra_deg": 10.0 + 20.0 / 3600.0, "dec_deg": 20.0}]
    resolved = _separations_from_target(target, neighbours)
    centre = SkyCoord(10.0 * u.deg, 20.0 * u.deg)
    for neighbour, expected in zip(neighbours, resolved):
        independent = centre.separation(SkyCoord(neighbour["ra_deg"] * u.deg, neighbour["dec_deg"] * u.deg)).arcsec
        assert expected["separation_arcsec"] == pytest.approx(float(independent), rel=1e-12)


def test_mist_release_refuses_symlinks_and_an_unreadable_receipt(tmp_path):
    from types import SimpleNamespace
    from trex.acquisition.scene import _mist_archive

    name = "MIST_v1.2_vvcrit0.0_UBVRIplus.txz"
    real = tmp_path / "real.txz"
    real.write_bytes(b"synthetic")
    (tmp_path / name).symlink_to(real)
    with pytest.raises(AcquisitionError, match="regular files"):
        _mist_archive(tmp_path, SimpleNamespace(download=lambda *_: pytest.fail("no download")))
    (tmp_path / name).unlink()
    (tmp_path / name).write_bytes(b"synthetic")
    (tmp_path / name).with_suffix(".retrieval.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="URL/retrieval-date"):
        _mist_archive(tmp_path, SimpleNamespace(download=lambda *_: pytest.fail("no download")))


def test_mist_release_downloads_once_and_records_a_utc_retrieval(tmp_path):
    from types import SimpleNamespace
    from trex.acquisition.scene import MIST_URL, _mist_archive

    name = "MIST_v1.2_vvcrit0.0_UBVRIplus.txz"
    downloads = []

    def download(url, path):
        downloads.append(url)
        path.write_bytes(b"synthetic")
        return path

    path, record = _mist_archive(tmp_path, SimpleNamespace(download=download))
    assert downloads == [MIST_URL] and path.name == name
    assert record["url"] == MIST_URL and record["retrieved_utc"].endswith("Z")
    assert _mist_archive(tmp_path, SimpleNamespace(download=lambda *_: pytest.fail("no second download"))) == (path, record)


def test_a_retained_scene_stage_with_other_source_ids_is_not_reused(tmp_path, monkeypatch):
    from trex.acquisition.scene import _find_available

    write_json(tmp_path / "host_tables_old.json", {"status": "available", "source_ids": ["999"]})
    write_json(tmp_path / "host_tables_new.json", {"status": "available", "source_ids": ["123"]})
    assert _find_available(tmp_path, "host_tables_*.json", ["123"])["source_ids"] == ["123"]
    assert _find_available(tmp_path, "host_tables_*.json", ["456"]) is None


def test_a_symlinked_scene_stage_record_is_not_reused(tmp_path):
    from trex.acquisition.scene import _find_available

    write_json(tmp_path / "real.json", {"status": "available", "source_ids": ["123"]})
    (tmp_path / "host_tables_link.json").symlink_to(tmp_path / "real.json")
    assert _find_available(tmp_path, "host_tables_*.json", ["123"]) is None


def test_neighbour_retrieval_time_is_the_latest_host_table_date():
    from trex.acquisition.scene import _latest_retrieval

    queries = {
        "astrophysical_parameters": {"source_queries": {
            "1": {"retrieved_utc": "2026-01-01T00:00:00Z"},
            "2": {"retrieved_utc": "2026-03-01T00:00:00Z"}}},
        "starhorse": {"source_queries": {"1": {"retrieved_utc": "2026-02-01T00:00:00Z"}}},
        "shboost": {"source_queries": {"1": {}}},
    }
    assert _latest_retrieval(queries) == "2026-03-01T00:00:00Z"
