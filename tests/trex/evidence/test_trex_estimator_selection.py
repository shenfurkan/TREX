"""Explicit estimator refusal and declared placement; mocks test dispatch only."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence as nested


def _unsupported():
    def native(*, device):
        raise NotImplementedError("unsupported synthetic contract")

    return SimpleNamespace(population=None, target_feh=None, hosted=False, baseline_likelihood=None,
                           prior_transform=lambda x: x, log_likelihood=lambda x: 0.0,
                           jax_unit_integrand=native)


@pytest.mark.parametrize("selection", ["typo", "", "auto"])
def test_unknown_selection_never_starts_an_estimator(monkeypatch, selection):
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", selection)
    monkeypatch.setattr(nested, "_run_nested_sampler", lambda *a: pytest.fail("wrong estimator ran"))
    with pytest.raises(ValueError, match="estimator"):
        nested._integrate(_unsupported(), nested._SamplerControls(32, 0.05, 32, 7, 1), "TP", 4, None)


def test_explicit_native_contract_failure_never_falls_back(monkeypatch):
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    monkeypatch.setattr(nested, "_run_nested_sampler", lambda *a: pytest.fail("silent dynesty fallback"))
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        with pytest.raises(NotImplementedError, match="unsupported synthetic"):
            nested._integrate(_unsupported(), nested._SamplerControls(32, 0.05, 32, 7, 1), "TP", 4, None)


def test_explicit_native_does_not_silently_select_cpu(monkeypatch):
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(models, "_usable_jax_devices", lambda: ())
    with models.jax.default_device(None):
        with pytest.raises(RuntimeError, match="explicit.*CPU|CPU.*explicit"):
            nested._integrate(_unsupported(), nested._SamplerControls(32, 0.05, 32, 7, 1), "TP", 4, None)


def test_missing_population_context_is_not_a_different_estimator(monkeypatch):
    integrand = _unsupported()
    integrand.population = object()
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        with pytest.raises(ValueError, match="target_feh"):
            nested._integrate(integrand, nested._SamplerControls(32, 0.05, 32, 7, 1), "TP", 4, None)


def test_explicit_selection_is_checked_before_completed_escalation_reuse(tmp_path, monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "typo")
    monkeypatch.setattr(escalation, "_load_completed", lambda *a: pytest.fail("incompatible completed cache was read"))
    costs = []
    token = escalation._escalation_runtime.set((tmp_path, float("inf"), costs))
    try:
        with pytest.raises(ValueError, match="estimator"):
            escalation._nested_call(escalation._target_plan_entry(0), blocks={},
                **synthetic_trex_tp_integrand_case(), M_s_Msun=1.0, R_s_Rsun=1.0,
                bin_integration_days=0.0, target_aperture_flux_fraction=1.0, target_feh=0.0,
                nsamples=3, random_seed=7, n_jobs=1)
    finally:
        escalation._escalation_runtime.reset(token)


def test_completed_escalation_reuse_is_bound_to_estimator(tmp_path, monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from trex.evidence.nested_checkpoint import CheckpointInputMismatch
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    calls = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return nested.NestedEvidenceResult(-10.0, 0.05, 400, 1000, 100, "stub", "0", kwargs["random_seed"])

    monkeypatch.setattr(escalation, "tp_scenario_nested_evidence", estimate)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "dynesty")
    costs = []
    token = escalation._escalation_runtime.set((tmp_path, math.inf, costs))
    inputs = dict(synthetic_trex_tp_integrand_case(), blocks={}, M_s_Msun=1.0, R_s_Rsun=1.0,
                  bin_integration_days=0.0, target_aperture_flux_fraction=1.0, target_feh=0.0,
                  nsamples=3, random_seed=7, n_jobs=1)
    try:
        first = escalation._nested_call(escalation._target_plan_entry(0), **inputs)
        reused = escalation._nested_call(escalation._target_plan_entry(0), **inputs)
        np.testing.assert_equal(reused._asdict(), first._asdict())
        assert len(calls) == 1 and costs[-1]["reused"] is True
        monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
        with pytest.raises(CheckpointInputMismatch, match="inputs do not match"):
            escalation._nested_call(escalation._target_plan_entry(0), **inputs)
        assert len(calls) == 1 and costs[-1]["status"] == "failed"
    finally:
        escalation._escalation_runtime.reset(token)


def _captured_nested_call(monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    captured = []

    def capture(**kwargs):
        captured.append(kwargs)
        return nested.NestedEvidenceResult(
            log_evidence=-10.0, log_evidence_error=0.05, n_live_points=kwargs.get("n_live_points", 400),
            n_likelihood_calls=1000, n_iterations=100, sampler="stub", sampler_version="0",
            random_seed=kwargs["random_seed"],
        )

    monkeypatch.setattr(escalation, "tp_scenario_nested_evidence", capture)
    case = synthetic_trex_tp_integrand_case()
    escalation._nested_call(
        escalation._target_plan_entry(0), blocks={}, **case,
        M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
        target_aperture_flux_fraction=1.0, target_feh=0.0,
        nsamples=3, random_seed=7, n_jobs=1,
    )
    return escalation, captured[0]


def test_blackjax_route_default_matches_the_calibrated_known_z_configuration(monkeypatch):
    """The escalation route passes the calibrated 3,200-live controls and the
    derived 4,000,000-call ceiling from the production contract."""
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    escalation, kwargs = _captured_nested_call(monkeypatch)

    assert kwargs["n_live_points"] == escalation._BLACKJAX_CALIBRATED_LIVE_POINTS == 3200
    assert kwargs["max_likelihood_calls"] == 4_000_000
    assert kwargs["max_likelihood_calls"] == escalation._blackjax_likelihood_call_cap(3200)
    assert kwargs["evidence_tolerance"] == 0.05


@pytest.mark.parametrize("live,cap", [(3200, 4_000_000), (5000, 6_000_000)])
def test_blackjax_explicit_live_count_is_honoured(monkeypatch, live, cap):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    captured = []
    monkeypatch.setattr(
        escalation, "tp_scenario_nested_evidence",
        lambda **kwargs: captured.append(kwargs) or nested.NestedEvidenceResult(
            -10.0, 0.05, kwargs["n_live_points"], 1000, 100, "stub", "0", kwargs["random_seed"]),
    )
    escalation._nested_call(
        escalation._target_plan_entry(0), blocks={}, **synthetic_trex_tp_integrand_case(),
        M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
        target_aperture_flux_fraction=1.0, target_feh=0.0,
        nsamples=3, random_seed=7, n_jobs=1, n_live_points=live,
    )
    assert captured[0]["n_live_points"] == live
    assert captured[0]["max_likelihood_calls"] == cap


def test_blackjax_cap_uses_the_scenario_dimension(monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    captured = []
    monkeypatch.setattr(
        escalation, "tp_scenario_nested_evidence",
        lambda **kwargs: captured.append(kwargs) or nested.NestedEvidenceResult(
            -10.0, 0.05, kwargs["n_live_points"], 1000, 100, "stub", "0", kwargs["random_seed"]),
    )
    plan = dict(escalation._target_plan_entry(0))
    plan["population"] = "bound"
    escalation._nested_call(
        plan, blocks={"bound": SimpleNamespace(n_dimensions=3)},
        **synthetic_trex_tp_integrand_case(), M_s_Msun=1.0, R_s_Rsun=1.0,
        bin_integration_days=0.0, target_aperture_flux_fraction=1.0, target_feh=0.0,
        nsamples=3, random_seed=7, n_jobs=1,
    )
    assert captured[0]["max_likelihood_calls"] == escalation._blackjax_likelihood_call_cap(3200, 7)


@pytest.mark.skipif(os.environ.get("EXONYM_RUN_SLOW_VALIDATIONS") != "1",
                    reason="one native BlackJAX assessment run")
def test_default_blackjax_escalation_runs_the_native_batched_path(monkeypatch, tmp_path):
    """The default route executes 3,200 live under its derived cap, batched."""
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case
    from tests.trex.evidence.test_trex_blackjax_calibration import GaussianIntegrand, _GAUSSIAN_TRUTH

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")

    def run_gaussian(**kwargs):
        integrand = SimpleNamespace(
            population=None, target_feh=None,
            jax_unit_integrand=GaussianIntegrand,
        )
        controls = nested._SamplerControls(
            n_live_points=kwargs["n_live_points"],
            evidence_tolerance=kwargs["evidence_tolerance"],
            max_likelihood_calls=kwargs["max_likelihood_calls"],
            random_seed=kwargs["random_seed"],
            n_jobs=1,
        )
        with models.jax.default_device(models.jax.devices("cpu")[0]):
            return nested._run_native_nested_sampler(integrand, controls, 4)

    monkeypatch.setattr(escalation, "tp_scenario_nested_evidence", run_gaussian)
    token = escalation._escalation_runtime.set((tmp_path, math.inf, []))
    try:
        result = escalation._nested_call(
            escalation._target_plan_entry(0), blocks={}, **synthetic_trex_tp_integrand_case(),
            M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
            target_aperture_flux_fraction=1.0, target_feh=0.0,
            nsamples=3, random_seed=421140000, n_jobs=1,
        )
    finally:
        escalation._escalation_runtime.reset(token)

    print(json.dumps({
        "n_live_points": int(result.n_live_points),
        "max_likelihood_calls": int(result.max_likelihood_calls),
        "n_likelihood_calls": int(result.n_likelihood_calls),
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
        "stop_reason": result.stop_reason,
        "slice_execution": result.slice_execution,
        "configuration_calibrated": bool(result.configuration_calibrated),
    }))
    assert result.n_live_points == 3200
    assert result.max_likelihood_calls == escalation._blackjax_likelihood_call_cap(3200)
    assert result.slice_execution == "vmapped-chains"
    assert result.stop_reason == "evidence-tolerance"
    assert result.n_likelihood_calls < result.max_likelihood_calls
    assert result.configuration_calibrated
    assert math.isfinite(result.log_evidence) and abs(result.log_evidence - _GAUSSIAN_TRUTH) < 0.1


def test_blackjax_refuses_an_explicit_live_count_below_the_calibrated_value(monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    monkeypatch.setattr(escalation, "tp_scenario_nested_evidence", lambda **_: pytest.fail("uncalibrated run started"))
    with pytest.raises(ValueError, match="calibrated at 3200"):
        escalation._nested_call(
            escalation._target_plan_entry(0), blocks={}, **synthetic_trex_tp_integrand_case(),
            M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
            target_aperture_flux_fraction=1.0, target_feh=0.0,
            nsamples=3, random_seed=7, n_jobs=1, n_live_points=400,
        )


def test_dynesty_controls_keep_their_defaults_and_honour_an_explicit_live_count(monkeypatch):
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.delenv("TREX_NESTED_ESTIMATOR", raising=False)
    escalation, kwargs = _captured_nested_call(monkeypatch)
    assert "n_live_points" not in kwargs  # nested_evidence's unchanged 400 default
    assert kwargs["max_likelihood_calls"] == escalation._MAXIMUM_NESTED_LIKELIHOOD_CALLS == 2_000_000

    captured = []
    monkeypatch.setattr(
        escalation, "tp_scenario_nested_evidence",
        lambda **values: captured.append(values) or nested.NestedEvidenceResult(
            -10.0, 0.05, values["n_live_points"], 1000, 100, "stub", "0", values["random_seed"]),
    )
    escalation._nested_call(
        escalation._target_plan_entry(0), blocks={}, **synthetic_trex_tp_integrand_case(),
        M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
        target_aperture_flux_fraction=1.0, target_feh=0.0,
        nsamples=3, random_seed=7, n_jobs=1, n_live_points=200,
    )
    assert captured[0]["n_live_points"] == 200
    assert captured[0]["max_likelihood_calls"] == 2_000_000


def _low_level_entries():
    from tests.fixtures.synthetic_observations import (
        synthetic_trex_eb_integrand_case,
        synthetic_trex_tp_integrand_case,
    )

    tp = synthetic_trex_tp_integrand_case()
    eb = synthetic_trex_eb_integrand_case(2.0)
    eb["period_days"] = eb.pop("effective_period")
    return (
        pytest.param(nested.tp_scenario_nested_evidence, tp, id="tp"),
        pytest.param(nested.eb_scenario_nested_evidence, eb, id="eb"),
    )


_LOW_LEVEL_SCENE = dict(
    M_s_Msun=1.0, R_s_Rsun=1.0, target_aperture_flux_fraction=1.0,
    target_feh=0.0, nsamples=3, random_seed=7, n_jobs=1,
)


def _entry_controls(monkeypatch, entry, case, **controls):
    captured = []

    def capture(integrand, sampler_controls, scenario, n_dimensions, importance_posterior):
        captured.append(sampler_controls)
        return "stub"

    monkeypatch.setattr(nested, "_integrate", capture)
    entry(**case, **_LOW_LEVEL_SCENE, **controls)
    return captured[0]


@pytest.mark.parametrize("entry,case", _low_level_entries())
def test_blackjax_low_level_default_is_the_calibrated_configuration(monkeypatch, entry, case):
    """A direct entry call cannot silently run BlackJAX at 400 live points."""
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    controls = _entry_controls(monkeypatch, entry, case)
    assert controls.n_live_points == 3200
    assert controls.max_likelihood_calls == 4_000_000
    assert controls.max_likelihood_calls == nested._blackjax_call_ceiling(3200, 1600, 12)


@pytest.mark.parametrize("entry,case", _low_level_entries())
def test_dynesty_low_level_default_is_unchanged(monkeypatch, entry, case):
    monkeypatch.delenv("TREX_NESTED_ESTIMATOR", raising=False)
    controls = _entry_controls(monkeypatch, entry, case)
    assert controls.n_live_points == 400
    assert controls.max_likelihood_calls is None


@pytest.mark.parametrize("entry,case", _low_level_entries())
def test_low_level_explicit_controls_keep_their_meaning(monkeypatch, entry, case):
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    controls = _entry_controls(
        monkeypatch, entry, case, n_live_points=5000, max_likelihood_calls=6_000_000
    )
    assert (controls.n_live_points, controls.max_likelihood_calls) == (5000, 6_000_000)


def test_low_level_explicit_ceiling_none_is_still_refused(monkeypatch):
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    with pytest.raises(ValueError, match="explicit likelihood ceiling"):
        nested.tp_scenario_nested_evidence(
            **synthetic_trex_tp_integrand_case(), max_likelihood_calls=None, **_LOW_LEVEL_SCENE
        )


def test_low_level_explicit_live_count_keeps_control_validation_order(monkeypatch):
    """An explicit live count is validated as before, without an estimator lookup first."""
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "typo")
    with pytest.raises(ValueError, match="must exceed the parameter count"):
        nested.tp_scenario_nested_evidence(
            **synthetic_trex_tp_integrand_case(),
            n_live_points=3, max_likelihood_calls=1000, **_LOW_LEVEL_SCENE
        )
