"""Every unresolved population must consume the same normalized baseline density."""
from contextlib import nullcontext

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence import nested_evidence as nested
from trex.photometry.baseline import GaussianBaseline


@pytest.mark.parametrize("kind", ["bound", "background"])
@pytest.mark.parametrize("hosted", [False, True], ids=["diluting", "hosting"])
@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
def test_population_native_uses_normalized_aperture_baseline(monkeypatch, kind, hosted, binary, mode):
    time = np.linspace(-.15, .15, 17)
    flux, sigma = np.ones(time.size), np.full(time.size, .01)
    baseline = GaussianBaseline(time, np.column_stack((np.ones(time.size), time)), np.array([.001, -.002]),
                                np.diag([1e-6, 2e-6]), mode, "synthetic independent baseline")
    if kind == "bound":
        population = nested.bound_companion_population(1., 1., 10., resolution_limit_arcsec=1., host_teff_k=5772.)
    else:
        population = nested.background_star_population(
            {"tess_mag": np.array([12.]), "mass_solar": np.array([.8]), "radius_solar": np.array([.8]),
             "teff_k": np.array([5000.]), "mh_dex": np.array([0.])}, .1, 10., resolution_limit_arcsec=1.)
    captured = []
    def capture(integrand, controls, scenario, dimensions, posterior):
        captured.append(integrand)
        return nested.NestedEvidenceResult(0., .01, 400, 0, 0, "construction-only", "0", 7)
    monkeypatch.setattr(nested, "_integrate", capture)
    call = nested.eb_scenario_nested_evidence if binary else nested.tp_scenario_nested_evidence
    call(time, flux, sigma, 3., 1., 1., .4, .2, 120/86400,
         target_aperture_flux_fraction=.2, target_feh=0., host_teff_k=5772., baseline=baseline,
         nsamples=3, random_seed=7, n_jobs=1,
         **{"hosted_population" if hosted else "population": population})
    scalar, = captured
    assert nested._cpu_refinement_integrand(scalar) is scalar
    native = scalar.jax_unit_integrand(device=models.jax.devices("cpu")[0])
    points = np.random.default_rng(371).uniform(.02, .98, size=(32, scalar.n_dimensions))
    expected = np.array([scalar.log_likelihood(scalar.prior_transform(point)) for point in points])
    actual = models.jax.jit(models.jax.vmap(native.evaluate))(points)
    assert np.asarray(actual.numerically_valid).all()
    values = np.asarray(actual.log_integrand)
    np.testing.assert_array_equal(np.isfinite(values), np.isfinite(expected))
    assert np.isfinite(expected).any(), "the control needs retained physical support"
    np.testing.assert_allclose(values, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
@pytest.mark.parametrize("native", [False, True], ids=["scalar", "native-cpu"])
def test_zero_baseline_reaches_real_defensive_refinement(monkeypatch, mode, binary, native):
    """The full Gaussian normalization shifts the real estimator, not its weights."""
    if native and not models.HAS_JAX_TREX:
        pytest.skip("JAX is unavailable")
    time = np.linspace(-.1, .1, 17)
    flux = 1 + .002*np.sin(40*time)
    sigma = np.full(time.size, .02)
    baseline = GaussianBaseline(time, np.zeros((time.size, 1)), np.array([.3]),
                                np.array([[.5]]), mode, "synthetic zero design")
    rng = np.random.default_rng(11)
    points = rng.uniform(.05, .95, (64, 4))
    posterior = [(points, np.full(64, 1/64))]
    sampler = nested._run_importance_sampler
    monkeypatch.setattr(nested, "_IMPORTANCE_CENTRES", 32)
    monkeypatch.setattr(nested, "_run_importance_sampler",
                        lambda *args, **kwargs: sampler(*args, n_draws=512, **kwargs))
    if not native:
        monkeypatch.setattr(models, "HAS_JAX_TREX", False)
    call = nested.eb_scenario_nested_evidence if binary else nested.tp_scenario_nested_evidence
    context = models.jax.default_device(models.jax.devices("cpu")[0]) if native else nullcontext()
    with context:
        kwargs = dict(target_aperture_flux_fraction=.23, target_feh=0., host_teff_k=5772.,
                      nsamples=1, random_seed=7, n_jobs=1, importance_posterior=posterior)
        white = call(time, flux, sigma, 3., 1., 1., .4, .2, 120/86400, **kwargs)
        normalized = call(time, flux, sigma, 3., 1., 1., .4, .2, 120/86400, baseline=baseline, **kwargs)
    offset = -.5*np.sum(np.log(2*np.pi*sigma**2))
    assert normalized.log_evidence - white.log_evidence == pytest.approx(offset, abs=2e-9)
    assert normalized.effective_sample_size == pytest.approx(white.effective_sample_size, rel=1e-10)
    assert normalized.n_draws == white.n_draws == 512
    assert normalized.execution_provenance["backend"] == ("jax-native-cpu" if native else "scalar-cpu")


@pytest.mark.parametrize("native", [False, True], ids=["scalar", "native-cpu"])
def test_population_occurrence_is_added_once_after_real_refinement(monkeypatch, native):
    if native and not models.HAS_JAX_TREX:
        pytest.skip("JAX is unavailable")
    time = np.linspace(-.1, .1, 17)
    flux, sigma = np.ones(time.size), np.full(time.size, .02)
    baseline = GaussianBaseline(time, np.zeros((time.size, 1)), np.zeros(1),
                                np.ones((1, 1)), "additive", "synthetic zero design")
    population = nested.background_star_population(
        {"tess_mag": np.array([12.]), "mass_solar": np.array([.8]), "radius_solar": np.array([.8]),
         "teff_k": np.array([5000.]), "mh_dex": np.array([0.])}, .1, 10., resolution_limit_arcsec=1.)
    points = np.random.default_rng(12).uniform(.05, .95, (64, 6))
    posterior = [(points, np.full(64, 1/64))]
    sampler, integrate = nested._run_importance_sampler, nested._integrate
    raw = []

    def capture(*args, **kwargs):
        result = integrate(*args, **kwargs)
        raw.append(result)
        return result

    monkeypatch.setattr(nested, "_IMPORTANCE_CENTRES", 32)
    monkeypatch.setattr(nested, "_run_importance_sampler",
                        lambda *args, **kwargs: sampler(*args, n_draws=512, **kwargs))
    monkeypatch.setattr(nested, "_integrate", capture)
    if not native:
        monkeypatch.setattr(models, "HAS_JAX_TREX", False)
    context = models.jax.default_device(models.jax.devices("cpu")[0]) if native else nullcontext()
    with context:
        def evaluate(block):
            return nested.tp_scenario_nested_evidence(
                time, flux, sigma, 3., 1., 1., .4, .2, 120/86400,
                target_aperture_flux_fraction=.23, target_feh=0., nsamples=1, random_seed=7,
                n_jobs=1, population=block, baseline=baseline, importance_posterior=posterior,
            )
        first = evaluate(population)
        shifted = evaluate(population._replace(log_occurrence_weight=population.log_occurrence_weight + .7))
    assert first.log_evidence == pytest.approx(raw[0].log_evidence + population.log_occurrence_weight, abs=1e-12)
    assert shifted.log_evidence - first.log_evidence == pytest.approx(.7, abs=1e-12)
    assert shifted.effective_sample_size == pytest.approx(first.effective_sample_size, abs=1e-10)
