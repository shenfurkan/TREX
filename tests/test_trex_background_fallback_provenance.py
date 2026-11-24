"""A fallback background population says so in the inference result it produced.

The TRILEGAL LSST-simulation fallback (Dal Tio et al. 2022, ApJS 262, 22)
replaces an unavailable TRILEGAL web population, and its same-cone star counts
can fall below Gaia DR3 (0.17-0.82 in the R4 fields). The count deficit is
measured; its effect on a particular FPP is not, because the background-hosted
scenarios include both false-positive and planet-hosted terms while other
scenarios do not use the population. A fallback FPP must therefore not read
like a TRILEGAL web result. All inputs here are synthetic.
"""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trex.app import engine as engine_module
from trex.app.scene import TargetScene
from trex.app.target import target
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.foundation.physical_constants import SECONDS_PER_DAY

FALLBACK_TRIGGER = {
    "status": "unavailable", "stage": "submit",
    "error": {"exception_class": "TrilegalUnavailableError", "message": "synthetic TRILEGAL timeout"},
}


def _fallback_record(digest):
    return {
        "status": "available", "model": "trilegal-lsst-sim-dr2", "fallback": True,
        "fallback_trigger": FALLBACK_TRIGGER, "field_submission": "icrs-cone-deg",
        "population_sha256": digest,
    }


def _web_record(digest):
    return {"status": "available", "model": "trilegal", "population_sha256": digest}


def _scene(tmp_path, query_record=None):
    background = tmp_path / "trilegal_cache.csv"
    background.write_text("synthetic population\n", encoding="utf-8")
    digest = hashlib.sha256(background.read_bytes()).hexdigest()
    if query_record is not None:
        (tmp_path / "trilegal_query.json").write_text(
            json.dumps(query_record(digest)), encoding="utf-8")
    return TargetScene(
        tic_id=100001, ra_deg=120.0, dec_deg=-30.0,
        M_s_Msun=1.02, R_s_Rsun=0.98, Teff_K=5778.0, Tmag=10.0, plx_mas=10.0,
        sectors=[1], contrast_separations=None, contrast_values=None,
        unresolved_resolution_limit_arcsec=2.2,
        target_aperture_flux_fractions=[{"sector": 1, "fraction": 1.0}],
        resolved_neighbors=[], N_background=1, trilegal_cache=background,
        background_area_sq_deg=0.1,
    )


def _bound(tmp_path, query_record=None):
    return target.from_scene(
        _scene(tmp_path, query_record), sector_weights={1: 1.0}, depth_ppm=8000.0,
        exptime_days=120.0 / SECONDS_PER_DAY, random_seed=7)


def _fake_result():
    return SimpleNamespace(
        to_probability_dataframe=lambda: pd.DataFrame({"scenario": [], "probability": []}),
        aligned_fit_arrays=dict,
        fpp=0.2, nfpp=0.1, status="ok", claim_eligible=False, input_identity="synthetic",
        effective_sample_sizes={}, scenario_fits=[], supplemental_nested_evidence={},
        nested_evidence_escalation=None, jax_batch_fallbacks={}, diagnostics=[],
        phase_wall_seconds={},
    )


def _calculate(bound, monkeypatch):
    monkeypatch.setattr(engine_module, "run_trex_vetting", lambda *_args, **_kwargs: _fake_result())
    bound.calc_probs(np.array([-0.1, 0.1]), np.array([1.0, 0.99]), 1e-3, 3.0,
                     exptime=120.0 / SECONDS_PER_DAY, N=8, parallel=False)


def test_fallback_record_is_carried_by_the_scene_and_its_export(tmp_path):
    scene = _scene(tmp_path, _fallback_record)
    provenance = scene.background_population_provenance
    assert provenance["model"] == "trilegal-lsst-sim-dr2"
    assert provenance["fallback"] is True
    assert provenance["fallback_trigger"] == FALLBACK_TRIGGER
    assert "same-cone star counts can fall below Gaia DR3" in provenance["limitation"]
    assert "effect of that deficit on a particular FPP is not measured" in provenance["limitation"]
    assert "biased low" not in provenance["limitation"] and "lower bound" not in provenance["limitation"]
    assert "TRILEGAL LSST simulation fallback" in provenance["limitation"]
    # The portable export carries it to `python -m trex.run`, which has no query record.
    payload = scene.to_dict()
    assert payload["background_population_provenance"] == provenance
    assert TargetScene.from_dict(payload).background_population_provenance == provenance


def test_fallback_provenance_requires_the_retained_population(tmp_path):
    with pytest.raises(ValueError, match="does not name the retained population"):
        _scene(tmp_path, lambda digest: {**_fallback_record("0" * 64)})
    with pytest.raises(ValueError, match="does not simulate the target field"):
        _scene(tmp_path, lambda digest: {
            **_fallback_record(digest), "field_submission": "galactic-lb-deg"
        })
    scene = _scene(tmp_path, _fallback_record)
    with pytest.raises(ValueError, match="must name the TRILEGAL LSST-simulation fallback"):
        TargetScene.from_dict({**scene.to_dict(), "background_population_provenance": _web_record(scene.background_sha256)})
    with pytest.raises(ValueError, match="does not state its model"):
        TargetScene.from_dict({**scene.to_dict(), "background_population_provenance": {"fallback": True}})
    with pytest.raises(ValueError, match="does not name the retained background population"):
        TargetScene.from_dict({
            **scene.to_dict(),
            "background_population_provenance": {
                **scene.background_population_provenance,
                "population_sha256": "0" * 64,
            },
        })


def test_a_fallback_model_without_the_flag_is_still_marked(tmp_path):
    scene = _scene(tmp_path, lambda digest: {**_fallback_record(digest), "fallback": None})
    provenance = scene.background_population_provenance
    assert provenance["fallback"] is True
    assert provenance["population_sha256"] == scene.background_sha256


def test_a_web_record_that_claims_the_fallback_is_refused():
    from trex.acquisition.background_population import fallback_provenance, record_simulates_target_field

    record = {**_web_record("a" * 64), "fallback": True}
    assert not record_simulates_target_field(record)
    with pytest.raises(ValueError, match="does not name the fallback model"):
        fallback_provenance(record)


def test_a_web_population_carries_no_fallback_provenance(tmp_path):
    scene = _scene(tmp_path, _web_record)
    assert scene.background_population_provenance is None
    assert "background_population_provenance" not in scene.to_dict()


def test_fallback_scene_writes_its_provenance_and_limitation(tmp_path, monkeypatch):
    bound = _bound(tmp_path, _fallback_record)
    _calculate(bound, monkeypatch)
    bound.write_outputs(tmp_path / "result")
    payload = json.loads((tmp_path / "result" / "summary.json").read_text())
    provenance = payload["calculation_provenance"]["background_population"]
    assert provenance["model"] == "trilegal-lsst-sim-dr2"
    assert provenance["fallback"] is True
    assert provenance["fallback_trigger"] == FALLBACK_TRIGGER
    assert "same-cone star counts can fall below Gaia DR3" in provenance["limitation"]
    assert "effect of that deficit on a particular FPP is not measured" in provenance["limitation"]


def test_scaling_background_evidence_can_move_fpp_either_way():
    """The fallback count deficit alone does not determine the FPP direction.

    ``_draw_background_star`` multiplies every D*/B* evidence by the expected
    number of simulated stars in the unresolved field, so a population with
    fewer stars scales all background-hosted scenarios by one factor. These
    two synthetic mixes share that scaling: one FPP falls and one rises. The
    G21 period factors are constant under that scaling, so the prior-free
    normalization here preserves the sign argument. The limitation text
    therefore reports the measured count deficit, not a bias direction.
    """
    from trex.evidence.marginal_likelihoods import N_TARGET_SCENARIOS, compute_fpp_nfpp

    def fpp(tp, eb, dtp, deb):
        evidence = np.full(N_TARGET_SCENARIOS, -np.inf)
        evidence[0] = math.log(tp)    # TP, planet-hosted, no population
        evidence[1] = math.log(eb)    # EB, no population
        evidence[9] = math.log(dtp)   # DTP, planet scenario, population
        evidence[10] = math.log(deb)  # DEB, false-positive, population
        value, _nfpp, _probs, status = compute_fpp_nfpp(evidence)
        assert status == "ok"
        return value

    # False background mass dominates: FPP = 11/112 falls to 6/106.5.
    assert fpp(100.0, 1.0, 1.0, 10.0) == pytest.approx(11.0 / 112.0)
    assert fpp(100.0, 1.0, 0.5, 5.0) == pytest.approx(6.0 / 106.5)
    assert fpp(100.0, 1.0, 0.5, 5.0) < fpp(100.0, 1.0, 1.0, 10.0)

    # DTP dominates the planet mass: FPP = 1.1/102.1 rises to 0.6/51.6.
    assert fpp(1.0, 0.1, 100.0, 1.0) == pytest.approx(1.1 / 102.1)
    assert fpp(1.0, 0.1, 50.0, 0.5) == pytest.approx(0.6 / 51.6)
    assert fpp(1.0, 0.1, 50.0, 0.5) > fpp(1.0, 0.1, 100.0, 1.0)


def test_fallback_failure_record_carries_the_limitation(tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise TrexEvidenceUnavailableError("synthetic engine refusal")

    monkeypatch.setattr(engine_module, "run_trex_vetting", refuse)
    bound = _bound(tmp_path, _fallback_record)
    with pytest.raises(TrexEvidenceUnavailableError):
        bound.calc_probs(np.array([-0.1, 0.1]), np.array([1.0, 0.99]), 1e-3, 3.0,
                         exptime=120.0 / SECONDS_PER_DAY, N=8, parallel=False)
    bound.write_outputs(tmp_path / "result")
    payload = json.loads((tmp_path / "result" / "failure.json").read_text())
    assert payload["background_population"]["fallback"] is True
    assert "effect of that deficit on a particular FPP is not measured" in payload["background_population"]["limitation"]


@pytest.mark.parametrize("query_record", [None, _web_record])
def test_a_result_without_a_fallback_population_is_unchanged(tmp_path, monkeypatch, query_record):
    bound = _bound(tmp_path, query_record)
    _calculate(bound, monkeypatch)
    bound.write_outputs(tmp_path / "result")
    payload = json.loads((tmp_path / "result" / "summary.json").read_text())
    assert "background_population" not in payload["calculation_provenance"]
