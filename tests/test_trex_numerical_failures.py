"""Numerical failures are not physical exclusions (peer review AU3)."""
import json

import numpy as np
import pytest

from trex.evidence import nested_evidence as nested
from trex.foundation.errors import NumericalLikelihoodError


@pytest.mark.parametrize('kind', ['tp', 'eb'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_scalar_nested_distinguishes_model_failure_from_excluded_geometry(monkeypatch, kind, value):
    # A central, circular, detached synthetic orbit is supported; face-on is not.
    time = np.linspace(-0.1, 0.1, 11)
    common = dict(time=time, flux=np.ones_like(time), sigma=1e-3,
                  host_mass=1.0, host_radius=1.0, u1=0.4, u2=0.2,
                  exptime_days=0.001, bin_integration_days=0.0, nsamples=1,
                  population=None, n_dimensions=4, target_feh=0.0)
    if kind == 'tp':
        integrand = nested._TransitingPlanetIntegrand(period_days=5.0, use_flat_priors=False, **common)
        point = np.array([2.0, 0.0, 90.0, 90.0])
        name = 'lnL_TP'
    else:
        integrand = nested._EclipsingPairIntegrand(effective_period=5.0, **common)
        point = np.array([0.5, 0.0, 90.0, 90.0])
        name = 'lnL_EB'
    monkeypatch.setattr(nested, name, lambda *_a, **_k: value)
    with pytest.raises(NumericalLikelihoodError, match='non-finite value on supported geometry'):
        integrand.log_likelihood(point)
    point[-1] = 0.0
    assert integrand.log_likelihood(point) == -np.inf


@pytest.mark.parametrize('backend', ['scalar', 'jax'])
def test_mc_counts_supported_numerical_failures_and_defers(monkeypatch, backend):
    from trex.evidence import marginal_likelihoods as ml
    from trex.evidence.diagnostics import generate_diagnostics
    from trex.evidence.monte_carlo_state import _monte_carlo_state_record, _restore_monte_carlo_state

    # Circular central transits are supported; inject three distinct solver/float failures.
    monkeypatch.setattr(ml, 'sample_ecc', lambda x, **_k: np.zeros_like(x))
    if backend == 'scalar':
        values = iter([np.nan, np.inf, NumericalLikelihoodError('synthetic solver failure')])

        def likelihood(*_a, **_k):
            value = next(values, -1.0)
            if isinstance(value, Exception):
                raise value
            return value

        monkeypatch.setattr(ml, 'lnL_TP', likelihood)
    else:
        monkeypatch.setattr(ml, 'HAS_JAX_TREX', True)
        monkeypatch.setattr(ml, 'batch_lnL_TP', lambda *_a, **_k: None)

        def batch(_batch, *args, **_kwargs):
            values = np.full(len(args[3]), -1.0)
            values[:3] = [np.nan, np.inf, -np.inf]
            return values

        monkeypatch.setattr(ml, '_retry_jax_batch', batch)

    time = np.linspace(-0.1, 0.1, 11)
    sizes, fits = [], []
    lnz = ml._eval_scenario(time, np.ones_like(time), 1e-3, 5.0, 1.0, 1.0,
                            0.4, 0.2, 24, True, False, False, 0.001,
                            rng=np.random.default_rng(7), ess_out=sizes,
                            representative_fit_out=fits)
    assert fits[0]['transiting_draws'] == 24
    assert fits[0]['unresolved_numerical_failures'] == 3
    assert fits[0]['status'] == 'unresolved-numerical'
    assert fits[0]['numerical_failure_examples']
    vector = np.full(15, -np.inf)
    vector[0] = lnz
    # Isolate numerical gating from the independently failing small-budget ESS.
    result = generate_diagnostics(vector, effective_sample_sizes={0: 1000.0}, scenario_fits={0: fits[0]})
    assert result.status == 'degenerate_evidence' and result.fpp is None and result.nfpp is None
    assert any(d.name == 'scenario-numerical-failure' and 'TP=3' in d.message for d in result.diagnostics)

    state = _monte_carlo_state_record(lnZ=vector, ess={0: sizes[0]}, support_state={},
        jax_fallbacks={}, scenario_fits={0: fits[0]}, n_draws=24, random_seed=7, monte_carlo_seconds=1.0)
    restored = _restore_monte_carlo_state(state, n_draws=24, random_seed=7, expected_scenarios=15)
    assert restored[4][0]['unresolved_numerical_failures'] == 3


def test_scalar_importance_counts_nonfinite_outputs_without_turning_them_into_zeros():
    from trex.evidence.nested_importance import _LogLikelihoodBlock

    # One of three supported values is NaN; the others remain ordinary evaluations.
    block = _LogLikelihoodBlock(lambda point: point, lambda point: np.nan if point[0] == 0.5 else -1.0)
    with pytest.raises(NumericalLikelihoodError) as failure:
        block(np.array([[0.25], [0.5], [0.75]]))
    assert failure.value.count == 1


def test_importance_pilot_counts_numerical_failures(monkeypatch):
    from trex.evidence import nested_controls as controls
    from trex.evidence.nested_importance import _adapt_proposal

    monkeypatch.setattr(controls, '_IMPORTANCE_ADAPTATION_DRAWS', 24)
    rng = np.random.default_rng(7)
    points = rng.uniform(size=(32, 2))
    with pytest.raises(NumericalLikelihoodError, match='adaptive importance pass') as failure:
        _adapt_proposal(points, np.full(32, 1 / 32), lambda x: np.full(len(x), np.nan), rng)
    assert 0 < failure.value.count <= 24


def test_native_failure_count_excludes_padding_and_other_slices():
    from trex.accelerators import _jax_models as models
    from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation
    from trex.accelerators._native_unit_batch import NativeUnitCubeBatch

    class Reference:
        n_dimensions = 1
        time = np.array([0.0])
        nsamples = 1

        def jax_unit_integrand(self, *, device):
            class Native:
                def evaluate(self, point):
                    valid = point[0] != 0.5
                    return UnitIntegrandEvaluation(models.jnp.where(valid, 0.0, models.jnp.nan), valid)
            return Native()

    points = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    with models.jax.default_device(models.jax.devices('cpu')[0]):
        evaluator = NativeUnitCubeBatch(Reference(), points)
        evaluator.block = 4  # the last logical point is padded into four device lanes
        with pytest.raises(NumericalLikelihoodError) as failure:
            evaluator(slice(0, 5))
        assert failure.value.count == 1
        evaluator = NativeUnitCubeBatch(Reference(), points[::-1])
        evaluator.block = 4
        np.testing.assert_array_equal(evaluator(slice(1, 4)), np.zeros(3))


def test_baseline_overflow_is_counted_by_real_monte_carlo():
    from trex.evidence import marginal_likelihoods as ml
    from trex.evidence.diagnostics import generate_diagnostics
    from trex.photometry.baseline import GaussianBaseline

    time = np.linspace(-.05, .05, 11)
    flux, sigma = np.ones(time.size), np.full(time.size, 1e-200)
    density = GaussianBaseline(time, np.zeros((time.size, 1)), np.zeros(1), np.ones((1, 1)),
                               "additive", "synthetic zero design").prepare(time, flux, sigma)
    fits, ess = [], []
    with np.errstate(over="ignore", invalid="ignore"):
        value = ml._eval_scenario(
            time, flux, sigma, 3., 1., 1., .4, .2, 32, True, False, False, 120/86400,
            rng=np.random.default_rng(8), nsamples=1, target_feh=0.,
            baseline_likelihood=density, representative_fit_out=fits, ess_out=ess,
        )
    assert fits[0]["status"] == "unresolved-numerical"
    assert fits[0]["unresolved_numerical_failures"] > 0
    assert fits[0]["numerical_failure_examples"]
    lnz = np.full(15, -np.inf)
    lnz[0] = value
    result = generate_diagnostics(lnz, effective_sample_sizes={0: 1000.}, scenario_fits={0: fits[0]})
    assert result.status == "degenerate_evidence" and result.fpp is None and result.nfpp is None


def test_baseline_overflow_remains_counted_through_mc_pilot_and_resume(monkeypatch, tmp_path):
    from trex.accelerators import _jax_models as models
    from trex.evidence import mc_importance as stage
    from trex.evidence import nested_controls as controls
    from trex.photometry.baseline import GaussianBaseline

    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "dynesty")
    monkeypatch.setattr(models, "HAS_JAX_TREX", False)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 64)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 1)
    time = np.linspace(-.05, .05, 11)
    flux = np.ones(time.size)
    sigma = np.full(time.size, 1e-200)
    baseline = GaussianBaseline(time, np.zeros((time.size, 1)), np.zeros(1),
                                np.ones((1, 1)), "additive", "synthetic zero design")
    points = np.random.default_rng(8).uniform(.05, .95, (controls._IMPORTANCE_CENTRES, 4))
    state = stage.make_state(7, {"0": points.tolist()}, "input")
    lnz = np.full(15, -np.inf)
    lnz[0] = -1.
    directory = tmp_path / "stage"
    inputs = dict(
        time=time, flux=flux, sigma=sigma, period_days=3., depth_ppm=1000.,
        M_s_Msun=1., R_s_Rsun=1., u1=.4, u2=.2, exptime_days=120/86400,
        bin_integration_days=0., target_aperture_flux_fraction=.6,
        contrast_separations=None, contrast_values=None, resolution_limit_arcsec=None,
        plx=None, target_tess_mag=10., background_population=None, background_area_sq_deg=None,
        target_feh=0., target_teff_k=5772., nsamples=1, baseline=baseline,
    )
    kwargs = dict(lnZ=lnz, ess={0: 1.}, state=state, n_draws=controls._IMPORTANCE_DRAWS,
                  input_identity="input", checkpoint_directory=directory, n_jobs=1, **inputs)
    with np.errstate(over="ignore", invalid="ignore"):
        first = stage.run_mc_importance(**kwargs)
        resumed = stage.run_mc_importance(**kwargs)
    attempt = first["attempts"]["TP"]
    assert attempt["status"] == "unavailable"
    assert attempt["unresolved_numerical_failures"] > 0
    assert resumed["attempts"]["TP"]["unresolved_numerical_failures"] == attempt["unresolved_numerical_failures"]
    assert resumed["lnZ"][0] == lnz[0] and resumed["ess"][0] == 1.
    journal = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    assert journal[0]["numerical_failure"]["count"] == attempt["unresolved_numerical_failures"]
    assert "baseline" in journal[0]["numerical_failure"]["traceback"]
