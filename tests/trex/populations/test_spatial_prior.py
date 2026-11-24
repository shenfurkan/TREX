"""Retained difference-image diagnostics must not reweight TREX neighbours.

Covers the legacy record ``trex-spatial-prior-v1``, its diagnostic-only
interpretation, and the end-to-end absence of centroid-tail suppression.
Explicit synthetic scenario priors still exercise the engine's likelihood/prior
algebra independently of localization.
"""

from __future__ import annotations

import numpy as np
import pytest

from trex.populations.spatial_prior import neighbor_log_priors, spatial_evidence_from_localization_payload

# Declared synthetic log weight used only to exercise scenario-weight algebra;
# it is not a Bryson spatial prior or an inferred source-location likelihood.
_SYNTHETIC_LOG_PRIOR = -4.5


def _localization_payload(
    candidate_id,
    *,
    ratio=4.2,
    offset=0.05,
    sigma=0.02,
    source="official-prf-difference-imaging",
    decision_state="characterized",
):
    return {
        "schema_version": "1.0",
        "work_package": "PRF_SOURCE_LOCALIZATION",
        "candidate_id": candidate_id,
        "source": source,
        "calibrated": False,
        "source_assignment_interpretable": False,
        "summary": {"median_target_to_other_difference_ratio": ratio},
        "per_transit_centroid_offsets": {
            "bryson_2013_multi_transit_average": {
                "decision_state": decision_state,
                "offset_distance_arcsec": offset,
                "offset_distance_sigma_arcsec": sigma,
            }
        },
    }


# ---------------------------------------------------------------------------
# Contract: neutral when the deciding evidence is absent or unsupportive
# ---------------------------------------------------------------------------


def test_neighbor_log_priors_are_neutral_without_any_evidence():
    assert neighbor_log_priors([1.0, 33.0], None) == [0.0, 0.0]
    assert neighbor_log_priors([1.0, 33.0], {"status": "unavailable", "reason": "x"}) == [0.0, 0.0]


@pytest.mark.parametrize("calibrated,interpretable", [(False, False), (True, True)])
def test_centroid_tail_is_not_a_scenario_likelihood_even_with_status_flags(calibrated, interpretable):
    payload = _localization_payload("c")
    payload.update(calibrated=calibrated, source_assignment_interpretable=interpretable)
    evidence = spatial_evidence_from_localization_payload(payload, "c")
    assert evidence["status"] == "available"  # Retain the diagnostic, not a prior.
    assert neighbor_log_priors([0.05, 33.0], evidence) == [0.0, 0.0]


def test_neighbor_log_priors_are_neutral_when_the_ratio_is_not_target_favoured():
    evidence = spatial_evidence_from_localization_payload(
        _localization_payload("c", ratio=0.5), "c"
    )
    assert evidence["status"] == "available"
    assert neighbor_log_priors([33.0], evidence) == [0.0]


def test_neighbor_log_priors_are_neutral_when_the_target_centroid_is_itself_offset():
    # offset significance 50 > Bryson 3-sigma: no target-favoured location.
    evidence = spatial_evidence_from_localization_payload(
        _localization_payload("c", offset=1.0, sigma=0.02), "c"
    )
    assert neighbor_log_priors([33.0], evidence) == [0.0]


def test_spatial_evidence_rejects_stub_foreign_and_unresolved_results():
    stub = _localization_payload("c", source="not-run-official-prf-required")
    assert spatial_evidence_from_localization_payload(stub, "c")["status"] == "unavailable"
    assert (
        spatial_evidence_from_localization_payload(_localization_payload("other"), "c")["status"]
        == "unavailable"
    )
    unresolved = _localization_payload("c", decision_state="deferred")
    assert (
        spatial_evidence_from_localization_payload(unresolved, "c")["status"] == "unavailable"
    )


# ---------------------------------------------------------------------------
# Effect: centroid diagnostics never suppress a neighbour
# ---------------------------------------------------------------------------


def test_neighbor_log_priors_do_not_suppress_an_inconsistent_neighbour():
    evidence = spatial_evidence_from_localization_payload(_localization_payload("c"), "c")
    priors = neighbor_log_priors([0.05, 33.0], evidence)
    assert priors == [0.0, 0.0]


def test_neighbor_log_prior_does_not_use_the_rayleigh_tail():
    evidence = spatial_evidence_from_localization_payload(
        _localization_payload("c", offset=0.0, sigma=0.1), "c"
    )
    assert neighbor_log_priors([0.1], evidence) == [0.0]


def _target_scene_kwargs(tmp_path):
    background = tmp_path / "synthetic_trilegal.csv"
    background.write_text("synthetic background population\n", encoding="utf-8")
    import hashlib

    return {
        "tic_id": 123,
        "ra_deg": 90.0,
        "dec_deg": -60.0,
        "M_s_Msun": 1.0,
        "R_s_Rsun": 1.0,
        "Teff_K": 5700.0,
        "Tmag": 10.0,
        "plx_mas": 2.0,
        "sectors": [1],
        "contrast_separations": np.array([0.1, 1.0]),
        "contrast_values": np.array([2.0, 5.0]),
        "target_aperture_flux_fractions": [{"sector": 1, "fraction": 1.0}],
        "resolved_neighbors": [],
        "N_background": 1,
        "trilegal_cache": background,
        "background_sha256": hashlib.sha256(background.read_bytes()).hexdigest(),
        "background_area_sq_deg": 0.1,
    }


def test_target_scene_neighbour_entries_keep_diagnostic_centroids_neutral(tmp_path):
    from trex.app.scene import TargetScene

    neighbor = {
        "source_id": "synthetic-neighbor",
        "M_s": 0.7,
        "R_s": 0.68,
        "delta_mag": 2.0,
        "separation_arcsec": 33.0,
        "aperture_flux_fractions": [{"sector": 1, "fraction": 0.05}],
    }
    evidence = spatial_evidence_from_localization_payload(_localization_payload("c"), "c")
    kwargs = _target_scene_kwargs(tmp_path)
    kwargs["resolved_neighbors"] = [neighbor]
    kwargs["localization_evidence"] = evidence
    scene = TargetScene(**kwargs)
    entries = scene.neighbor_dicts_for_evidence({1: 1.0})
    assert entries[0]["log_prior"] == 0.0
    assert entries[0]["aperture_flux_fraction"] == 0.05
    assert entries[0]["source_id"] == neighbor["source_id"]

    baseline_kwargs = _target_scene_kwargs(tmp_path)
    baseline_kwargs["resolved_neighbors"] = [neighbor]
    baseline = TargetScene(**baseline_kwargs)
    assert baseline.neighbor_dicts_for_evidence({1: 1.0})[0]["log_prior"] == 0.0


def test_centroid_evidence_preserves_scenario_probabilities_and_explicit_prior_algebra():
    from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC, SECONDS_PER_DAY
    from trex.evidence.marginal_likelihoods import (
        calc_target_evidences,
        compute_fpp_nfpp,
    )

    evidence = spatial_evidence_from_localization_payload(_localization_payload("c"), "c")
    log_prior = neighbor_log_priors([33.0], evidence)[0]
    assert log_prior == 0.0

    time = np.linspace(-0.1, 0.1, 100)
    flux = np.ones_like(time)
    flux[45:55] = 0.995
    common = dict(
        time=time,
        flux=flux,
        sigma=0.001,
        period_days=3.5,
        depth_ppm=5000.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.3,
        u2=0.2,
        exptime_days=120.0 / SECONDS_PER_DAY,
        target_aperture_flux_fraction=1.0,
        n_draws=100,
        resolution_limit_arcsec=GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC,
        random_seed=7,
        nsamples=10,
        target_feh=0.0,
        plx=10.0,
        # Background scenarios need their retained population and sky area;
        # without them the engine correctly reports the set unavailable.
        background_population={
            "tess_mag": np.array([14.0]),
            "mass_solar": np.array([0.9]),
            "radius_solar": np.array([0.9]),
            "teff_k": np.array([5500.0]),
            "mh_dex": np.array([0.0]),
        },
        background_area_sq_deg=0.1,
        target_tess_mag=10.0,
    )

    def run(prior):
        neighbor = {
            "source_id": "synthetic-neighbor",
            "M_s": 0.7,
            "R_s": 0.68,
            "aperture_flux_fraction": 0.5,
            "log_prior": prior,
        }
        return calc_target_evidences(nearby_stars=[neighbor], **common)

    lnZ_base, base_status = run(0.0)
    lnZ_centroid, centroid_status = run(log_prior)
    assert base_status == "ok" and centroid_status == "ok"
    assert np.array_equal(lnZ_centroid, lnZ_base)
    _, nfpp_base, probs_base, status_base = compute_fpp_nfpp(lnZ_base)
    _, nfpp_centroid, probs_centroid, centroid_probability_status = compute_fpp_nfpp(lnZ_centroid)
    assert status_base == "ok" and centroid_probability_status == "ok"
    assert np.array_equal(probs_centroid, probs_base)
    assert nfpp_centroid == nfpp_base

    # The engine must still apply an explicit prior; disabling a faulty
    # centroid consumer must not disable general scenario-weight algebra.
    lnZ_prior, prior_status = run(_SYNTHETIC_LOG_PRIOR)
    assert base_status == "ok" and prior_status == "ok"
    assert np.all(np.isfinite(lnZ_prior[15:18]))
    # The prior is applied to exactly the three neighbour scenario slots.
    assert np.allclose(lnZ_base[15:18] - lnZ_prior[15:18], -_SYNTHETIC_LOG_PRIOR)
    assert np.array_equal(lnZ_base[:15], lnZ_prior[:15])

    _, nfpp_base, probs_base, status_base = compute_fpp_nfpp(lnZ_base)
    _, nfpp_prior, probs_prior, status_prior = compute_fpp_nfpp(lnZ_prior)
    assert status_base == "ok" and status_prior == "ok"
    assert probs_prior[15:18].sum() < probs_base[15:18].sum()
    assert nfpp_prior < nfpp_base
