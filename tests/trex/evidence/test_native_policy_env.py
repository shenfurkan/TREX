"""Opt-in inference policy parsing and wiring; no JAX or likelihood execution."""

from types import SimpleNamespace

import pytest

from trex.evidence import nested_controls as controls
from trex.evidence import nested_evidence as nested


_ENVIRONMENT = (
    controls._BLACKJAX_DIRECTION_ENV,
    controls._BLACKJAX_BATCH_INIT_ENV,
    controls._BLACKJAX_W_MARCH_ENV,
    controls._IMPORTANCE_PROPOSAL_ENV,
)


def _clear(monkeypatch):
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_defaults_preserve_the_assessed_policy(monkeypatch):
    _clear(monkeypatch)
    assert controls._selected_importance_proposal() == "kernel-mixture"
    assert controls._native_kernel_policy() == {
        "direction_metric": "fixed-isotropic",
        "batched_initialization": False,
        "initialization_block": 64,
        "speculative_expansions": 0,
    }
    assert controls._native_kernel_policy_identity() == ()


@pytest.mark.parametrize("name,value,expected", [
    (controls._BLACKJAX_DIRECTION_ENV, "pilot", "fixed-pilot"),
    (controls._BLACKJAX_DIRECTION_ENV, "split", "split-ensemble"),
    (controls._BLACKJAX_BATCH_INIT_ENV, "1", True),
    (controls._BLACKJAX_W_MARCH_ENV, "4", 4),
    (controls._BLACKJAX_W_MARCH_ENV, "8", 8),
])
def test_each_documented_value_selects_its_policy(monkeypatch, name, value, expected):
    _clear(monkeypatch)
    monkeypatch.setenv(name, value)
    policy = controls._native_kernel_policy()
    key = {controls._BLACKJAX_DIRECTION_ENV: "direction_metric",
           controls._BLACKJAX_BATCH_INIT_ENV: "batched_initialization",
           controls._BLACKJAX_W_MARCH_ENV: "speculative_expansions"}[name]
    assert policy[key] == expected
    assert controls._native_kernel_policy_identity()


@pytest.mark.parametrize("name,value", [
    (controls._BLACKJAX_DIRECTION_ENV, "adaptive"),
    (controls._BLACKJAX_BATCH_INIT_ENV, "yes"),
    (controls._BLACKJAX_W_MARCH_ENV, "1"),
    (controls._IMPORTANCE_PROPOSAL_ENV, "kde"),
])
def test_invalid_or_unsupported_values_are_refused(monkeypatch, name, value):
    _clear(monkeypatch)
    monkeypatch.setenv(name, value)
    resolver = (controls._selected_importance_proposal if name == controls._IMPORTANCE_PROPOSAL_ENV
                else controls._native_kernel_policy)
    with pytest.raises(ValueError, match=name):
        resolver()


def test_policy_identity_contains_only_non_default_fields(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(controls._BLACKJAX_DIRECTION_ENV, "split")
    monkeypatch.setenv(controls._BLACKJAX_BATCH_INIT_ENV, "1")
    monkeypatch.setenv(controls._BLACKJAX_W_MARCH_ENV, "4")
    assert controls._native_kernel_policy_identity() == (
        "split-ensemble", ("batched-initialization", 64), ("speculative-expansions", 4))


def test_importance_route_uses_the_selected_proposal(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(controls._IMPORTANCE_PROPOSAL_ENV, "probit-gaussian")
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return "refined"

    monkeypatch.setattr(nested, "_run_importance_sampler", run)
    integrand = SimpleNamespace(prior_transform=lambda x: x, log_likelihood=lambda x: 0.0)
    posterior = [(SimpleNamespace(), SimpleNamespace())]
    sampler_controls = SimpleNamespace(random_seed=5, n_jobs=1)
    assert nested._integrate(integrand, sampler_controls, "TP", 1, posterior) == "refined"
    assert captured["proposal"] == "probit-gaussian"
    assert captured["random_seed"] == 5 and captured["n_jobs"] == 1


def test_completed_result_identity_binds_the_native_policy(tmp_path, monkeypatch):
    from trex.evidence import evidence_escalation as escalation
    from tests.fixtures.synthetic_observations import synthetic_trex_tp_integrand_case

    _clear(monkeypatch)
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "blackjax")
    identities = []

    def load(_path, identity):
        identities.append(identity)
        return nested.NestedEvidenceResult(
            -10.0, 0.05, 3200, 1, 1, "stub", "0", 7)

    monkeypatch.setattr(escalation, "_load_completed", load)
    monkeypatch.setattr(escalation, "tp_scenario_nested_evidence",
                        lambda **_: pytest.fail("a retained result with its identity was not reused"))
    case = synthetic_trex_tp_integrand_case()
    costs = []
    token = escalation._escalation_runtime.set((tmp_path, float("inf"), costs))
    try:
        for policy in (None, "split", None):
            if policy is None:
                monkeypatch.delenv(controls._BLACKJAX_DIRECTION_ENV, raising=False)
            else:
                monkeypatch.setenv(controls._BLACKJAX_DIRECTION_ENV, policy)
            escalation._nested_call(
                escalation._target_plan_entry(0), blocks={}, **case,
                M_s_Msun=1.0, R_s_Rsun=1.0, bin_integration_days=0.0,
                target_aperture_flux_fraction=1.0, target_feh=0.0,
                nsamples=3, random_seed=7, n_jobs=1)
    finally:
        escalation._escalation_runtime.reset(token)
    assert identities[0] == identities[2] != identities[1]


def test_unbatched_speculation_is_refused(monkeypatch):
    from trex.evidence._blackjax_kernel import NativeNestedKernel

    with pytest.raises(ValueError, match="requires batched=True"):
        NativeNestedKernel(
            object(), n_dimensions=1, n_live_points=2, num_delete=1, num_inner_steps=1,
            batched=False, speculative_expansions=4)


def test_native_route_forwards_the_selected_policy_and_controls(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(controls._BLACKJAX_DIRECTION_ENV, "split")
    monkeypatch.setenv(controls._BLACKJAX_BATCH_INIT_ENV, "1")
    monkeypatch.setenv(controls._BLACKJAX_W_MARCH_ENV, "8")
    from trex.accelerators import _jax_models as models

    device = SimpleNamespace(platform="cpu", device_kind="cpu")
    fake_jax = SimpleNamespace(
        config=SimpleNamespace(jax_default_device=device),
        random=SimpleNamespace(PRNGKey=lambda seed: ("key", seed)),
    )
    monkeypatch.setattr(models, "HAS_JAX_TREX", True)
    monkeypatch.setattr(models, "jax", fake_jax)
    captured = {}

    class Kernel:
        shape = (8, 2)

        def integrate(self, positions, key, **kwargs):
            captured["positions"] = positions.shape
            captured["key"] = key
            captured.update(kwargs)
            return "evidence"

    def build(native, **kwargs):
        captured["native"] = native
        captured.update(kwargs)
        return Kernel()

    monkeypatch.setattr(nested, "native_nested_kernel", build)
    integrand = SimpleNamespace(population=None, target_feh=None,
                                jax_unit_integrand=lambda device: "native")
    sampler_controls = SimpleNamespace(
        n_live_points=8, evidence_tolerance=0.1, max_likelihood_calls=100, random_seed=3)
    assert nested._run_native_nested_sampler(integrand, sampler_controls, 2) == "evidence"
    assert captured["native"] == "native"
    assert captured["n_live_points"] == 8 and captured["num_delete"] == 4
    assert captured["num_inner_steps"] == 6 and captured["batched"] is True
    assert captured["direction_metric"] == "split-ensemble"
    assert captured["batched_initialization"] is True and captured["initialization_block"] == 64
    assert captured["speculative_expansions"] == 8
    assert captured["positions"] == (8, 2) and captured["key"] == ("key", 3)
    assert captured["evidence_tolerance"] == 0.1
    assert captured["max_likelihood_calls"] == 100 and captured["random_seed"] == 3
