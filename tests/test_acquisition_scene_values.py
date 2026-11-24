"""Gaia FLAME applicability is shared across standalone and workspace scenes."""

import pytest

from trex.acquisition.scene_values import (
    SceneBuildError, flame_neighbor_properties, initial_neighbor_properties, select_neighbor_properties,
    starhorse_neighbor_values, supplemental_neighbor_properties,
    tic_neighbor_values, uncertainty_envelope,
)


def test_flame_domain_and_giant_age_are_not_guessed():
    source = {"mass_flame": 1.0, "radius_flame": 1.5, "flags_flame": "00", "evolstage_flame": 2}
    assert flame_neighbor_properties(source, "123")["mass_solar"] == 1.0
    with pytest.raises(SceneBuildError, match="quality subset"):
        flame_neighbor_properties({**source, "flags_flame": "01"}, "123")
    with pytest.raises(SceneBuildError, match="age_flame"):
        flame_neighbor_properties({**source, "flags_flame": "10"}, "123")
    with pytest.raises(SceneBuildError, match="giant"):
        flame_neighbor_properties({**source, "flags_flame": "10", "age_flame_gyr": 1.0}, "123")


def test_tic_neighbor_values_refuse_missing_and_out_of_domain_mass():
    row = {"status": "available", "mass_solar": 0.8, "radius_solar": 0.7, "tic_id": "456"}
    record = {"neighbors": {"123": row}, "cross_identification_limitation": "synthetic"}
    assert tic_neighbor_values(record, "123")["mass_solar"] == 0.8
    assert tic_neighbor_values(record, "999") is None
    assert tic_neighbor_values({"neighbors": {"123": {**row, "mass_solar": 1e6}}}, "123") is None


def test_uncertainty_envelope_preserves_asymmetry_and_omits_invalid_bounds():
    result = uncertainty_envelope(2., 3., 1., 4., 2., 6.)
    assert result == {"mass_solar_err_lower": 1., "mass_solar_err_upper": 2.,
                      "radius_solar_err_lower": 1., "radius_solar_err_upper": 3.}
    assert uncertainty_envelope(2., 3., 3., 4., None, 6.) == {}


def test_starhorse_scene_keeps_evolved_and_single_band_limitations():
    row = {"mass_solar": 1., "radius_solar": 2., "sh_photoflag": "G", "logg_cgs": 3.,
           "mass_lower_solar": 0.8, "mass_upper_solar": 1.2,
           "radius_lower_solar": 1.5, "radius_upper_solar": 2.5}
    record = {"queries": {"starhorse": {"table": "synthetic", "uri": "https://example.org/"}},
              "retrieved_utc": "2026-01-01T00:00:00Z"}
    result = starhorse_neighbor_values(row, record, "123")
    assert result["mass_solar_err_lower"] == pytest.approx(0.2)
    assert result["radius_solar_err_upper"] == pytest.approx(0.5)
    assert "prior-informed" in result["stellar_parameter_limitations"]
    assert "evolved" in result["stellar_parameter_limitations"]


def test_supplemental_neighbor_prefers_valid_flame_over_starhorse():
    ap = {"mass_flame_solar": 1., "radius_flame_solar": 1.5, "flags_flame": "00", "evolstage_flame": 2}
    record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": ap, "starhorse_edr3": {
        "mass_solar": 0.7, "radius_solar": 0.6, "astrometric_fidelity": 0.9,
        "sh_outflag": "0000", "sh_photoflag": "valid"}}},
        "queries": {"astrophysical_parameters": {"table": "synthetic", "uri": "https://example.org/"}},
        "retrieved_utc": "2026-01-01T00:00:00Z"}
    result, evaluations = supplemental_neighbor_properties(record, "123")
    assert result["stellar_parameter_source"] == "gaia-dr3-flame"
    assert result["mass_solar"] == 1. and evaluations[-1]["status"] == "accepted"


def test_gspphot_main_sequence_relation_refuses_an_evolved_log_g():
    from trex.acquisition.scene_values import _gspphot_main_sequence

    main_sequence, reason = _gspphot_main_sequence(5800.0, 3.5)
    assert reason is None and main_sequence["mass_solar"] > 0
    refused, reason = _gspphot_main_sequence(5000.0, 3.0)
    assert refused is None and "evolved" in reason


def test_an_evolved_archival_neighbour_continues_to_starhorse():
    neighbor = {"teff_gspphot_k": 5000.0, "logg_gspphot_cgs": 3.0,
                "aperture_flux_fractions_by_sector": [0.5]}
    ap = {"teff_gspphot_k": 5000.0, "logg_gspphot_cgs": 3.0,
          "mass_flame_solar": None, "radius_flame_solar": None}
    starhorse = {"mass_solar": 1.4, "radius_solar": 3.0, "astrometric_fidelity": 0.9,
                 "sh_outflag": "0000", "sh_photoflag": "GBPRP"}
    record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": ap,
                                    "starhorse_edr3": starhorse}},
              "queries": {"astrophysical_parameters": {"table": "synthetic", "uri": "https://example.org/"},
                          "starhorse": {"table": "synthetic", "uri": "https://example.org/"}},
              "retrieved_utc": "2026-01-01T00:00:00Z"}
    result, reason, evaluations = select_neighbor_properties(
        neighbor, "123", gaia_record=record, minimum_capable_fraction=0.01)
    assert reason is None and result["stellar_parameter_source"] == "starhorse-edr3"
    assert result["mass_solar"] == 1.4


def test_uncertainty_envelope_also_requires_a_finite_radius_and_bounds():
    assert uncertainty_envelope(2., float("nan"), 1., 3., 2., 4.) == {
        "mass_solar_err_lower": 1., "mass_solar_err_upper": 1.}
    assert uncertainty_envelope(2., 3., 1., 3., float("inf"), 4.) == {
        "mass_solar_err_lower": 1., "mass_solar_err_upper": 1.}


def test_a_missing_or_non_finite_bp_magnitude_uses_the_mist_route(tmp_path, monkeypatch):
    from trex.acquisition import scene_values
    from trex.acquisition.scene_values import resolve_neighbor_properties

    calls = []

    def distribution(neighbor, grid, extinction, **_kwargs):
        calls.append((neighbor["phot_bp_mean_mag"], grid, extinction))
        return {"mass_solar": [1.0], "radius_solar": [1.0], "teff_k": [5800.0], "weights": [1.0],
                "method": "synthetic MIST", "limitations": "synthetic"}

    monkeypatch.setattr(scene_values, "read_mist_grid", lambda _archive: "grid")
    monkeypatch.setattr(scene_values, "mist_g_distribution", distribution)
    archive = tmp_path / "mist.tar.xz"
    archive.write_bytes(b"synthetic")

    def loader():
        return (archive, {"url": "https://example.org/mist", "retrieved_utc": "2026-01-01T00:00:00Z"}, 0.1)

    for bp in (None, float("nan")):
        neighbor = {"phot_g_mean_mag": 10.0, "phot_g_mean_mag_error": 0.01,
                    "parallax_mas": 10.0, "parallax_error_mas": 0.02, "phot_bp_mean_mag": bp,
                    "ecl_lat_deg": 0.0, "astrometric_params_solved": 31,
                    "nu_eff_used_in_astrometry_per_um": 1.5,
                    "aperture_flux_fractions_by_sector": [0.5]}
        gaia_record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": {}}},
                       "queries": {"astrophysical_parameters": {"status": "available"}}}
        result = resolve_neighbor_properties(
            neighbor, "123", gaia_record=gaia_record, minimum_capable_fraction=0.01, mist_loader=loader)
        assert result["stellar_parameter_source"] == "mist-g-parallax"
        assert result["mass_solar"] == pytest.approx(1.0)
    assert len(calls) == 2 and calls[0][1:] == ("grid", 0.1)


def test_initial_neighbor_uses_tic_after_a_refused_flame_value():
    neighbor = {"mass_flame": 1., "radius_flame": 1., "flags_flame": "01", "evolstage_flame": 2,
                "aperture_flux_fractions_by_sector": [0.5]}
    tic = {"neighbors": {"123": {"status": "available", "mass_solar": 0.8,
                                   "radius_solar": 0.7, "tic_id": "456"}}}
    result, reason = initial_neighbor_properties(neighbor, "123", tic)
    assert result["stellar_parameter_source"] == "tic-v8-mast" and reason is None
    result, reason, evaluations = select_neighbor_properties(neighbor, "123", tic)
    assert result["stellar_parameter_source"] == "tic-v8-mast" and reason is None
    assert any(item["source"] == "gaia-dr3-flame" and item["status"] == "refused"
               and "quality subset" in item["reason"] for item in evaluations)
    result, reason = initial_neighbor_properties({}, "123", tic)
    assert result["stellar_parameter_source"] == "tic-v8-mast" and reason is None


def test_initial_neighbor_uses_eligible_gsp_phot_after_a_refused_flame_value():
    neighbor = {"mass_flame": 0.4, "radius_flame": 0.5, "flags_flame": "00",
                "teff_gspphot_k": 3400.0, "logg_gspphot_cgs": 4.5,
                "aperture_flux_fractions_by_sector": [0.5]}
    result, reason, evaluations = select_neighbor_properties(neighbor, "123")
    assert reason is None
    assert result["stellar_parameter_source"] == "gaia-dr3-gspphot-teff-main-sequence-relation"
    assert result["mass_solar"] > 0 and result["radius_solar"] > 0
    assert any(item["source"] == "gaia-dr3-flame" and item["status"] == "refused"
               and "model domain" in item["reason"] for item in evaluations)


def test_refused_flame_without_a_gspphot_log_g_is_not_a_dwarf_substitution():
    from trex.acquisition.scene_values import resolve_neighbor_properties

    neighbor = {"mass_flame": 0.4, "radius_flame": 0.5, "flags_flame": "00",
                "teff_gspphot_k": 5000.0, "aperture_flux_fractions_by_sector": [0.5]}
    record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": {
        "mass_flame_solar": 0.4, "radius_flame_solar": 0.5, "flags_flame": "00",
        "teff_gspphot_k": 5000.0}}}}
    result = resolve_neighbor_properties(
        neighbor, "123", gaia_record=record, minimum_capable_fraction=0.01)
    assert result["mass_solar"] is None and result["radius_solar"] is None
    assert "log g is missing" in result["stellar_parameter_unavailable_reason"]
    assert any(item["source"] == "gaia-dr3-gspphot-teff-main-sequence-relation"
               and item["status"] == "refused" and "log g is missing" in item["reason"]
               for item in result["stellar_parameter_evaluations"])


def test_refused_flame_without_a_supported_fallback_stays_unavailable():
    from trex.acquisition.scene_values import resolve_neighbor_properties

    neighbor = {"mass_flame": 1.4, "radius_flame": 3.0, "flags_flame": "10",
                "aperture_flux_fractions_by_sector": [0.5]}
    result = resolve_neighbor_properties(
        neighbor, "123", gaia_record={"neighbors": {}}, minimum_capable_fraction=0.01)
    assert result["mass_solar"] is None and result["radius_solar"] is None
    assert result["stellar_parameter_source"] == "unavailable"
    assert "age_flame_gyr" in result["stellar_parameter_unavailable_reason"]
    assert any(item["source"] == "gaia-dr3-flame" and item["status"] == "refused"
               for item in result["stellar_parameter_evaluations"])


@pytest.mark.parametrize("flame", [
    {"mass_flame": None, "radius_flame": 1.5, "flags_flame": "00"},
    {"mass_flame": 0.4, "radius_flame": 0.5, "flags_flame": "00"},
    {"mass_flame": 1.4, "radius_flame": 3.0, "flags_flame": "10"},
])
def test_rejected_archival_flame_does_not_block_independent_starhorse(flame):
    from trex.acquisition.scene_values import resolve_neighbor_properties

    neighbor = {**flame, "aperture_flux_fractions_by_sector": [0.5]}
    record = {
        "neighbors": {"123": {"gaia_dr3_astrophysical_parameters": {
            "mass_flame_solar": flame["mass_flame"], "radius_flame_solar": flame["radius_flame"],
            "flags_flame": flame["flags_flame"]},
            "starhorse_edr3": {"mass_solar": 1.4, "radius_solar": 3.0, "logg_cgs": 3.0,
                              "astrometric_fidelity": 0.9, "sh_outflag": "0000", "sh_photoflag": "GBPRP"}}},
        "queries": {"starhorse": {"table": "synthetic", "uri": "https://example.org/"}},
        "retrieved_utc": "2026-01-01T00:00:00Z",
    }
    result = resolve_neighbor_properties(neighbor, "123", gaia_record=record, minimum_capable_fraction=0.01)
    assert result["stellar_parameter_source"] == "starhorse-edr3"
    assert (result["mass_solar"], result["radius_solar"]) == (1.4, 3.0)
    evaluations = result["stellar_parameter_evaluations"]
    assert any(item["source"] == "gaia-dr3-flame" and item["status"] == "refused"
               and item["reason"] for item in evaluations)
    assert any(item["source"] == "starhorse-edr3" and item["status"] == "accepted" for item in evaluations)


def test_rejected_flame_is_not_replaced_by_a_temperature_only_dwarf():
    from trex.acquisition.scene_values import resolve_neighbor_properties

    neighbor = {"mass_flame": 1.4, "radius_flame": 3.0, "flags_flame": "10",
                "teff_gspphot_k": 5000.0, "aperture_flux_fractions_by_sector": [0.5],
                "phot_bp_mean_mag": 12.0}
    record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": {
        "mass_flame_solar": 1.4, "radius_flame_solar": 3.0, "flags_flame": "10",
        "teff_gspphot_k": 5000.0}}}}
    result = resolve_neighbor_properties(neighbor, "123", gaia_record=record, minimum_capable_fraction=0.01)
    assert result["mass_solar"] is None and result["radius_solar"] is None
    assert "age_flame_gyr" in result["stellar_parameter_unavailable_reason"]
    assert any(item["status"] == "refused" and "dwarf replacement" in item["reason"]
               for item in result["stellar_parameter_evaluations"])


def test_giant_flags_from_either_stage_block_the_dwarf_substitution():
    from trex.acquisition.scene_values import resolve_neighbor_properties

    neighbor = {"mass_flame": 1.4, "radius_flame": 3.0, "flags_flame": "10",
                "teff_gspphot_k": 5000.0, "logg_gspphot_cgs": 4.5,
                "aperture_flux_fractions_by_sector": [0.5]}
    record = {"neighbors": {"123": {"gaia_dr3_astrophysical_parameters": {
        "teff_gspphot_k": 5000.0, "logg_gspphot_cgs": 4.5}}}}
    result = resolve_neighbor_properties(
        neighbor, "123", gaia_record=record, minimum_capable_fraction=0.01)
    assert result["mass_solar"] is None and result["radius_solar"] is None
    assert any(item["source"] == "gaia-dr3-gspphot-teff-main-sequence-relation"
               and item["status"] == "refused" and "giant" in item["reason"]
               for item in result["stellar_parameter_evaluations"])


def test_neighbor_selection_retains_unavailable_faint_source_without_inventing_mass():
    source = {"aperture_flux_fractions_by_sector": [0.001]}
    result, reason, evaluations = select_neighbor_properties(source, "123", gaia_record={"neighbors": {}},
                                                              minimum_capable_fraction=0.01)
    assert result["mass_solar"] is None and "no current TIC" in result["stellar_parameter_unavailable_reason"]
    assert reason is None and evaluations == []


def test_flame_without_evolutionary_stage_is_admitted_by_its_documented_flags():
    # Gaia DR3: flags_flame "00" = nothing to report, parallax used; evolstage_flame may be null
    # (live TOI-181 b neighbour 6552346104699487872, 2026-11-08: mass 1.016, radius 1.629, age 10.4 Gyr).
    from trex.acquisition.scene_values import flame_neighbor_properties

    values = flame_neighbor_properties({"mass_flame": 1.016, "radius_flame": 1.629, "flags_flame": "00",
                                        "evolstage_flame": None, "age_flame_gyr": 10.4}, "n")
    assert (values["mass_solar"], values["radius_solar"]) == (1.016, 1.629)
    assert values["flame_evolutionary_stage"] is None
