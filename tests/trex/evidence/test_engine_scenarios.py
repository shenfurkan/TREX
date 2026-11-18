"""TREX scenario evidence, diagnostics and convergence tests."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from trex.foundation.constants import SECONDS_PER_DAY
from trex.stellar.stellar_relations import stellar_relations
from trex.models.orbital_mechanics import secondary_eclipse_phase
from trex.app.scene import TargetScene
from trex.evidence.diagnostics import TrexResult, generate_diagnostics
from trex.evidence.marginal_likelihoods import _eval_scenario, calc_target_evidences
from trex.models.likelihoods import simulate_EB
from trex.foundation.errors import NumericalLikelihoodError
from trex.evidence.marginal_likelihoods import N_TARGET_SCENARIOS
from trex.evidence.marginal_likelihoods import SCENARIO_DTP
from trex.evidence.marginal_likelihoods import SCENARIO_PTP
from trex.evidence.marginal_likelihoods import SCENARIO_TP
from trex.evidence.marginal_likelihoods import compute_fpp_nfpp
from trex.evidence.marginal_likelihoods import MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY
from tests.fixtures.synthetic_observations import SYNTHETIC_SOLAR_METALLICITY_DEX


def synthetic_eb_ldc():
    """Representative cool-secondary coefficients from the retained TESS grid.

    Local to this module so the TREX-only tests do not import an EXONYM fixture.
    """
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    # TEST_DESIGN: select a retained cool-dwarf grid node, not invented LDC.
    u1, u2 = load_triceratops_tess_grid().coefficients(3500, 5, SYNTHETIC_SOLAR_METALLICITY_DEX)
    return {"u1_EB": u1, "u2_EB": u2}


def _target_scene_kwargs(tmp_path):
    background = tmp_path / "synthetic_trilegal.csv"
    background.write_text("synthetic background population\n", encoding="utf-8")

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


def test_eb_scenarios_use_empirical_companion_radii(monkeypatch):
    import trex.evidence.marginal_likelihoods as marginal_likelihoods

    captured = []
    monkeypatch.setattr(
        marginal_likelihoods,
        "lnL_EB",
        lambda *_args, **kwargs: captured.append((_args[3], kwargs["secondary_to_host_flux_ratio"])) or 0.0,
    )
    monkeypatch.setattr(
        marginal_likelihoods,
        "sample_q_host_mass_ratio",
        lambda draws, _host_masses: np.full(draws.size, 0.2),
    )
    monkeypatch.setattr(marginal_likelihoods, "sample_inc", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "sample_ecc", lambda draws, **_kwargs: np.zeros(draws.size))
    monkeypatch.setattr(marginal_likelihoods, "sample_w", lambda draws: np.full(draws.size, 90.0))

    _eval_scenario(
        np.array([0.0]), np.array([1.0]), 0.01, 5.0, 1.0, 1.0, 0.4, 0.2,
        2, is_planet=False, is_EB=True, use_2xP=False, exptime_days=0.01,
        rng=np.random.default_rng(1),
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )

    expected_radius, _ = stellar_relations(np.array([0.2]))
    assert [item[0] for item in captured] == pytest.approx([float(expected_radius[0])] * 2)
    assert captured[0][0] != pytest.approx(0.2)
    assert 0.0 < captured[0][1] < 1.0


def test_scenario_records_its_measured_wall_split(monkeypatch):
    """A per-scenario cost claim needs the split on the retained record.

    The scenario log line is disposable, so the same measurement is attached
    to the fit record that reaches the report. The values are measured, not
    asserted: this checks the record carries a complete, non-negative split
    that fits inside the call's own wall time.
    """
    import time as wall_clock

    import trex.evidence.marginal_likelihoods as marginal_likelihoods

    monkeypatch.setattr(marginal_likelihoods, "HAS_JAX_TREX", False)
    monkeypatch.setattr(marginal_likelihoods, "lnL_TP", lambda *_args, **_kwargs: 0.0)
    fits: list = []
    started = wall_clock.perf_counter()
    _eval_scenario(
        np.array([-0.01, 0.0, 0.01]), np.ones(3), 0.01, 1.0, 1.0, 1.0, 0.4, 0.2,
        16, is_planet=True, is_EB=False, use_2xP=False, exptime_days=0.01,
        rng=np.random.default_rng(3), representative_fit_out=fits,
    )
    elapsed = wall_clock.perf_counter() - started

    phases = fits[0]["phase_seconds"]
    assert set(phases) == {"draw", "wait", "evaluate"}
    assert all(math.isfinite(value) and value >= 0.0 for value in phases.values())
    assert sum(phases.values()) <= elapsed


@pytest.mark.parametrize("exptime_days", [0.0, -0.01, np.nan, np.inf])
def test_evidence_requires_finite_positive_exptime(exptime_days):
    with pytest.raises(ValueError, match="exptime_days"):
        calc_target_evidences(
            np.array([0.0]), np.array([1.0]), 0.01, 5.0, 1000.0, 1.0, 1.0,
            0.4, 0.2, exptime_days=exptime_days,
        )


def test_secondary_eclipse_phase_uses_keplerian_mean_anomalies():
    assert secondary_eclipse_phase(0.0, 0.0) == pytest.approx(0.5)
    assert secondary_eclipse_phase(0.3, 0.0) != pytest.approx(0.5)


def test_eb_model_evaluates_secondary_at_observed_times(monkeypatch):
    import trex.models.likelihoods as likelihoods

    calls = []
    observed_time = np.array([-0.2, 0.0, 0.2])
    monkeypatch.setattr(
        likelihoods,
        "_batman_transit",
        lambda time, *_args, **kwargs: calls.append((time, kwargs.get("t0_days", 0.0))) or np.full_like(time, 0.8),
    )

    _, phase = simulate_EB(
        observed_time, 0.6, 0.25, 5.0, 89.0, 1.0e12, 1.0, 0.4, 0.2,
        0.01, ecc=0.3, argp_deg=0.0,
        **synthetic_eb_ldc(),
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1][0], observed_time)
    assert calls[1][1] == pytest.approx(phase * 5.0)


def test_eb_model_preserves_an_exact_equal_radius_ratio():
    """Batman must receive the physical equal-radius EB geometry unchanged."""
    pytest.importorskip("batman")

    flux, _ = simulate_EB(
        np.array([-0.01, 0.0, 0.01]), 1.0, 0.25, 5.0, 89.0, 1.0e12,
        1.0, 0.4, 0.2, 0.01,
        **synthetic_eb_ldc(),
    )

    assert np.all(np.isfinite(flux))


def test_unevaluable_eccentric_draw_remains_a_numerical_failure():
    """A draw whose batman Kepler solver diverges has unresolved likelihood.

    Near-unity eccentricities are permitted by the Kipping (2013) planet and
    Moe & Di Stefano (2017) binary priors, but Batman's eccentric-anomaly
    solver cannot converge for them. Failure to compute does not exclude the
    allowed draw from the physical prior (referee report AU3).
    """
    pytest.importorskip("batman")
    from trex.models.likelihoods import DrawUnsupportedError, lnL_EB, lnL_TP, simulate_EB

    time = np.linspace(-0.15, 0.15, 641)
    flux = np.ones_like(time)
    sigma = 1.0e-3
    unsupported = dict(
        P_orb=5.0, inc_deg=89.0, a_cm=1.0e12, R_s_solar=1.0, u1=0.4, u2=0.2,
        exptime_days=0.01, ecc=0.98, argp_deg=90.0, companion_fluxratio=0.1,
        nsamples=20, bin_integration_days=0.01,
    )

    with pytest.raises(DrawUnsupportedError):
        simulate_EB(time, 0.6, 0.25, **unsupported, u1_EB=unsupported["u1"], u2_EB=unsupported["u2"])
    with pytest.raises(DrawUnsupportedError):
        lnL_EB(time, flux, sigma, 0.6, 0.25, **unsupported, u1_EB=unsupported["u1"], u2_EB=unsupported["u2"])
    with pytest.raises(DrawUnsupportedError):
        lnL_TP(time, flux, sigma, 1.0, **unsupported)


# ============================================================================
# Diagnostics
# ============================================================================

def test_generate_diagnostics_full():
    lnZ = np.full(15, -100.0)
    lnZ[0] = -10.0
    result = generate_diagnostics(lnZ)
    assert result.fpp is not None
    assert 0 <= result.fpp <= 1
    assert not result.degenerate
    assert not result.claim_eligible
    assert len(result.diagnostics) > 0


def test_generate_diagnostics_rejects_a_vector_without_every_target_scenario():
    # FPP is the sum over fixed scenario indices; a shorter vector would misassign them.
    with pytest.raises(ValueError, match="at least 15 target scenarios"):
        generate_diagnostics(np.array([-10.0, -12.0, -8.0, -15.0]))


def test_trex_result_top_scenarios():
    result = TrexResult()
    result.probs = np.array([0.8, 0.15, 0.03, 0.02])
    top = result.top_scenarios(2)
    assert len(top) == 2
    assert top[0][1] > top[1][1]


def test_subthreshold_overall_diagnostic_is_explicitly_nonvalidating():
    from trex.evidence.diagnostics import FPP_THRESHOLD, generate_diagnostics
    from trex.evidence.marginal_likelihoods import N_TARGET_SCENARIOS, SCENARIO_EB, SCENARIO_TP

    target_probability = FPP_THRESHOLD / (1.0 + FPP_THRESHOLD)
    log_evidence = np.full(N_TARGET_SCENARIOS, -np.inf)
    log_evidence[SCENARIO_TP] = 0.0
    log_evidence[SCENARIO_EB] = math.log(target_probability / (1.0 - target_probability))

    result = generate_diagnostics(log_evidence)
    overall = next(item for item in result.diagnostics if item.name == "overall")

    assert result.fpp is not None and result.fpp < FPP_THRESHOLD
    assert result.claim_eligible is False
    assert overall.status == "pass"
    assert "diagnostic only" in overall.message
    assert "validation" not in overall.message.lower()


def test_scenario_rejects_nonfinite_likelihood_draws():
    """A numerical likelihood failure must not be converted into zero evidence."""
    with pytest.raises(NumericalLikelihoodError, match="supported likelihood evaluations failed"):
        _eval_scenario(
            time=np.array([-0.01, 0.0, 0.01]),
            flux=np.array([1.0, np.nan, 1.0]),
            sigma=0.01,
            period_days=1.0,
            M_s_Msun=1.0,
            R_s_Rsun=1.0,
            u1=0.4,
            u2=0.2,
            n_draws=128,
            is_planet=True,
            is_EB=False,
            use_2xP=False,
            exptime_days=0.01,
            rng=np.random.default_rng(7),
        )


@pytest.mark.parametrize("offset", [0.0, 2.0**50, -(2.0**50)])
def test_fpp_directly_sums_retained_false_scenario_probability_mass(offset):
    """Retain tiny finite FPP mass instead of subtracting it from unity.

    With one unit-relative TP evidence and equal false-scenario log-evidences
    50 natural-log units lower, the independently evaluated normalized false
    probability is ``12 * exp(-50) / (1 + 14 * exp(-50))``.  The two additional
    target-planet scenarios are deliberately retained in the denominator but
    excluded from FPP, matching the published FPP definition implemented by
    the scenario map.  Computing ``1 - P(planet scenarios)`` rounds this
    physically finite mass to zero in IEEE-754 binary64; summing the retained
    false-scenario probabilities does not.
    """
    log_evidence_gap = 50.0
    lnz = np.full(N_TARGET_SCENARIOS, -log_evidence_gap)
    lnz[SCENARIO_TP] = 0.0

    fpp, nfpp, probabilities, status = compute_fpp_nfpp(lnz + offset)

    retained_planet_scenarios = 3
    expected_false_scenarios = N_TARGET_SCENARIOS - retained_planet_scenarios
    expected_fpp = expected_false_scenarios * math.exp(-log_evidence_gap) / (
        1.0 + (N_TARGET_SCENARIOS - 1) * math.exp(-log_evidence_gap)
    )

    assert status == "ok"
    assert fpp > 0.0
    assert fpp == pytest.approx(expected_fpp, rel=1.0e-12)
    assert nfpp == 0.0
    assert probabilities[SCENARIO_PTP] > 0.0
    assert probabilities[SCENARIO_DTP] > 0.0


@pytest.mark.parametrize("offset", [0.0, 1e300, -1e300])
def test_equal_scenario_evidences_keep_fpp_nfpp_normalized(offset):
    from trex.foundation._numerics import _log_weight_ess

    # Fifteen target and three neighbor slots: only three target slots are planets.
    effective_count = _log_weight_ess(np.full(128, offset))
    fpp, nfpp, probabilities, status = compute_fpp_nfpp(
        np.full(18, offset), {index: effective_count for index in range(18)})
    assert status == "ok"
    np.testing.assert_allclose(probabilities, np.full(18, 1 / 18), rtol=0, atol=2e-16)
    assert fpp == pytest.approx(15 / 18, rel=0, abs=4e-16)
    assert nfpp == pytest.approx(3 / 18, rel=0, abs=2e-16)


# ============================================================================
# Evidence-estimator convergence (fixscientific.md F1-a, F4)
# ============================================================================

def test_log_weight_ess_counts_equal_weights():
    """Equally weighted draws have an effective sample size equal to their count."""
    from trex.foundation._numerics import _log_weight_ess

    assert _log_weight_ess(np.full(64, -3.0)) == pytest.approx(64.0, rel=1e-12)


def test_log_weight_ess_ignores_zero_weight_draws():
    """``-inf`` draws carry no weight, so they do not raise the effective count."""
    from trex.foundation._numerics import _log_weight_ess

    logw = np.concatenate([np.full(8, -3.0), np.full(500, -np.inf)])
    assert _log_weight_ess(logw) == pytest.approx(8.0, rel=1e-12)
    assert _log_weight_ess(np.full(5, -np.inf)) == 0.0


def test_log_weight_ess_is_one_when_a_single_draw_dominates():
    """One draw many nats above the rest is the degenerate estimator this guards."""
    from trex.foundation._numerics import _log_weight_ess

    logw = np.full(10_000, -500.0)
    logw[0] = 0.0
    assert _log_weight_ess(logw) == pytest.approx(1.0, abs=1e-9)


def test_compute_fpp_nfpp_defers_a_single_draw_dominated_evidence_set():
    """An evidence set below the effective-sample-size floor yields no FPP.

    Normalizing log-evidences that are each the largest single draw produces a
    probability vector determined by the sampler seed, so the estimator must
    report that it did not converge rather than publish a number.
    """
    from trex.evidence.marginal_likelihoods import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE, compute_fpp_nfpp

    lnZ = np.full(15, -100.0)
    lnZ[0] = -10.0
    converged = {index: MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE for index in range(15)}

    fpp, nfpp, probs, status = compute_fpp_nfpp(lnZ, converged)
    assert status == "ok"
    assert math.isfinite(fpp) and math.isfinite(nfpp)

    degenerate = dict(converged)
    degenerate[0] = 1.0
    fpp, nfpp, probs, status = compute_fpp_nfpp(lnZ, degenerate)
    assert status == "degenerate_evidence"
    assert math.isnan(fpp) and math.isnan(nfpp)
    # The collapsed vector is still returned so a reviewer can see the collapse.
    assert np.sum(probs) == pytest.approx(1.0)


def test_zero_sampled_evidence_does_not_certify_an_empty_prior():
    """A finite sample can miss positive support; an unevaluated excluded slot has no ESS."""
    from trex.evidence.marginal_likelihoods import compute_fpp_nfpp

    lnZ = np.full(15, -100.0)
    lnZ[3] = -np.inf
    sizes = {index: 250.0 for index in range(15)}
    sizes[3] = 0.0

    _, _, _, status = compute_fpp_nfpp(lnZ, sizes)
    assert status == "degenerate_evidence"

    # D-48: a structurally excluded slot has no evaluated-scenario ESS entry.
    sizes.pop(3)
    _, _, _, status = compute_fpp_nfpp(lnZ, sizes)
    assert status == "ok"


def test_generate_diagnostics_reports_the_degenerate_evidence_set():
    """The degeneracy is named, counted, and carries its floor in the record."""
    lnZ = np.full(15, -100.0)
    lnZ[0] = -10.0
    sizes = {index: 1.0 for index in range(15)}

    result = generate_diagnostics(lnZ, effective_sample_sizes=sizes)
    assert result.status == "degenerate_evidence"
    assert result.degenerate
    assert result.fpp is None and result.nfpp is None
    assert result.effective_sample_sizes["TP"] == pytest.approx(1.0)
    failure = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.name == "evidence-effective-sample-size"
    )
    assert failure.status == "fail"
    assert failure.threshold == MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY


def test_unconverged_scenarios_carrying_no_probability_do_not_defer_the_run():
    """Small realized residuals retain diagnostics, not a convergence certificate."""
    from trex.evidence.marginal_likelihoods import unconverged_probability_mass

    lnZ = np.full(15, -300.0)
    lnZ[0] = -10.0
    lnZ[9] = -11.0
    sizes = {index: 5000.0 for index in range(15)}
    for index in (1, 2, 4, 5, 7, 8):  # every eclipsing-pair scenario
        sizes[index] = 1.0

    _, _, probs, status = compute_fpp_nfpp(lnZ, sizes)
    assert unconverged_probability_mass(probs, sizes, lnZ) < (
        MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY
    )
    assert status == "ok"

    result = generate_diagnostics(lnZ, effective_sample_sizes=sizes)
    assert not result.degenerate
    assert result.fpp is not None
    waiver = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.name == "evidence-effective-sample-size"
    )
    assert waiver.status == "info"
    # The waived scenarios stay named in the record.
    assert "EB" in waiver.message
    assert "not an evidence-error bound" in waiver.message
    assert "does not establish C5 convergence" in waiver.message


def test_unconverged_scenarios_holding_real_probability_still_defer():
    """Diagnostic routing uses realized probability mass, not scenario identity."""
    lnZ = np.full(15, -300.0)
    lnZ[0] = -10.0
    lnZ[1] = -10.5
    sizes = {index: 5000.0 for index in range(15)}
    sizes[1] = 1.0

    _, _, _, status = compute_fpp_nfpp(lnZ, sizes)
    assert status == "degenerate_evidence"


def test_generate_diagnostics_retains_effective_sample_sizes_when_converged():
    """The convergence evidence is recorded on a passing run too, not only a failing one."""
    lnZ = np.full(15, -100.0)
    lnZ[0] = -10.0
    sizes = {index: 4000.0 for index in range(15)}

    result = generate_diagnostics(lnZ, effective_sample_sizes=sizes)
    assert result.status == "ok"
    assert not result.degenerate
    assert result.effective_sample_sizes["TP"] == pytest.approx(4000.0)


def test_trex_evidence_unavailable_error_carries_the_convergence_record(monkeypatch, tmp_path):
    """A deferred run must carry the per-scenario table that explains it.

    Prose alone lost the only measurement that decides between a larger draw
    budget and a different estimator (fixscientific.md F12-e): the 1e5-draw
    run left no record of which scenarios were near the floor. The exception
    now transports the table to the caller, which persists it.
    """
    from trex.app import engine as trex
    from trex.foundation.errors import TrexEvidenceUnavailableError
    from trex.evidence.marginal_likelihoods import MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE

    kwargs = _target_scene_kwargs(tmp_path)
    kwargs["feh_dex"] = 0.0
    kwargs["metallicity_provenance"] = {
        "source": "caller-supplied",
        "quantity": "[Fe/H]",
        "value_dex": 0.0,
        "uncertainty_dex": 0.1,
        "citation": "Synthetic test value; not a measurement of a real source",
    }
    scene = TargetScene(**kwargs)
    # The placeholder TRILEGAL file in this fixture is not a parseable
    # population; the faked evidence call never reads it.
    monkeypatch.setattr(scene, "background_population", dict)

    def fake_calc_target_evidences(**call_kwargs):
        ess_out = call_kwargs["ess_out"]
        ess_out.update({index: 1.0 for index in range(N_TARGET_SCENARIOS)})
        lnZ = np.full(N_TARGET_SCENARIOS, -100.0)
        lnZ[0] = -10.0
        return lnZ, "ok"

    monkeypatch.setattr(trex, "calc_target_evidences", fake_calc_target_evidences)

    with pytest.raises(TrexEvidenceUnavailableError) as excinfo:
        trex.run_trex_vetting(
            scene,
            time=np.linspace(-0.1, 0.1, 21),
            flux=np.ones(21),
            sigma=1.0e-3,
            period_days=3.5,
            depth_ppm=5000.0,
            exptime_days=120.0 / SECONDS_PER_DAY,
            sector_weights={1: 1.0},
            n_draws=2000,
        )

    diagnostics = excinfo.value.diagnostics
    assert diagnostics["n_draws"] == 2000
    assert diagnostics["minimum_floor"] == MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE
    assert diagnostics["effective_sample_sizes"]["TP"] == pytest.approx(1.0)
    assert diagnostics["unconverged_scenarios"]["TP"] == pytest.approx(1.0)
    assert diagnostics["unconverged_probability_mass"] > 0.5


def test_a_partial_escalation_defers_instead_of_publishing_an_fpp(monkeypatch, tmp_path):
    """One unresolved attempt poisons the set even when the remainder normalizes.

    The accepted scenario is retained in the failure record so a continuation
    does not repeat it; the unresolved one keeps the run deferred, because a
    normalization over the accepted subset would publish a probability that the
    unmeasured scenario could still move.
    """
    from trex.app import engine as trex
    from trex.foundation.errors import TrexEvidenceUnavailableError

    kwargs = _target_scene_kwargs(tmp_path)
    kwargs["feh_dex"] = 0.0
    kwargs["metallicity_provenance"] = {
        "source": "caller-supplied",
        "quantity": "[Fe/H]",
        "value_dex": 0.0,
        "uncertainty_dex": 0.1,
        "citation": "Synthetic test value; not a measurement of a real source",
    }
    scene = TargetScene(**kwargs)
    monkeypatch.setattr(scene, "background_population", dict)

    def fake_calc_target_evidences(**call_kwargs):
        ess_out = call_kwargs["ess_out"]
        ess_out.update({index: 5000.0 for index in range(N_TARGET_SCENARIOS)})
        ess_out[0] = 1.0  # TP: one draw dominates its own evidence
        ess_out[1] = 1.0
        lnZ = np.full(N_TARGET_SCENARIOS, -100.0)
        lnZ[0] = -10.0
        return lnZ, "ok"

    escalation_calls = []

    def fake_run_nested_escalation(**call_kwargs):
        escalation_calls.append(call_kwargs)
        lnZ = np.array(call_kwargs["lnZ"], dtype=float)
        lnZ[0] = -10.2  # the accepted nested re-integration
        return {
            "attempted_scenarios": ["TP", "EB"],
            "escalated": [0],
            "lnZ": lnZ,
            "remaining_ess": {1: 1.0},  # EB stays unresolved
            "records": {"TP": {"scenario": "TP", "estimator": "nested-sampling",
                               "log_evidence": -10.2}},
            "unmet_reason": "EB: nested integration returned no completed run",
            "unmet_scenarios": ["EB"],
            "unmet_reasons": {"EB": "nested integration returned no completed run"},
            "unsubstituted_scenarios": [],
            "unsubstituted_reason": None,
            "cap_binding": False,
            "random_seed": 0,
            "seeds": {},
            "attempt_costs": [],
            "cumulative_wall_seconds": 0.0,
        }

    monkeypatch.setattr(trex, "calc_target_evidences", fake_calc_target_evidences)
    monkeypatch.setattr(trex, "run_nested_escalation", fake_run_nested_escalation)

    with pytest.raises(TrexEvidenceUnavailableError) as excinfo:
        trex.run_trex_vetting(
            scene,
            time=np.linspace(-0.1, 0.1, 21),
            flux=np.ones(21),
            sigma=1.0e-3,
            period_days=3.5,
            depth_ppm=5000.0,
            exptime_days=120.0 / SECONDS_PER_DAY,
            sector_weights={1: 1.0},
            n_draws=2000,
        )

    assert escalation_calls, "the engine must attempt the escalation path this test exercises"
    diagnostics = excinfo.value.diagnostics
    escalation = diagnostics["nested_escalation"]
    assert escalation["escalated"] == ["TP"]
    assert escalation["unmet_scenarios"] == ["EB"]
    assert "EB" in str(excinfo.value)
    # The completed substitution travels with the deferral, so a continuation
    # can reuse it; the unresolved scenario is not silently dropped.
    assert diagnostics["supplemental_nested_evidence"]["TP"]["estimator"] == "nested-sampling"
    assert diagnostics["effective_sample_sizes"]["EB"] == pytest.approx(1.0)


def test_unconverged_mass_does_not_depend_on_ess_dict_order():
    """GPU accept 2026-10-20: a GPU run and its resume built the ESS dict in different orders and the
    unconverged mass differed in the last bit (1.0 vs 1.0000000000000009)."""
    from trex.evidence.marginal_likelihoods import unconverged_probability_mass

    probs = np.random.default_rng(0).random(30) ** 8
    probs /= probs.sum()
    lnZ = np.log(probs)
    forward = {index: 1.0 for index in range(30)}
    backward = {index: 1.0 for index in reversed(range(30))}
    assert unconverged_probability_mass(probs, forward, lnZ) == unconverged_probability_mass(probs, backward, lnZ)
