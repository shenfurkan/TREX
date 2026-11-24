"""Full diluted/background-hosted TP/EB callbacks versus actual scalar integrands."""

import os

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence.nested_evidence import _TransitingPlanetIntegrand, _EclipsingPairIntegrand, background_star_population
from tests.fixtures.synthetic_observations import (
    synthetic_trex_tp_integrand_case, synthetic_trex_eb_integrand_case,
    synthetic_trex_background_integrand_population,
)


@pytest.mark.parametrize("kind", ["tp", "eb"])
@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("measured_curve", [False, True])
@pytest.mark.parametrize("execution_device", ["cpu", pytest.param("gpu:0", marks=pytest.mark.skipif(
    os.environ.get("EXONYM_REQUIRE_GPU") != "1", reason="physical GPU acceptance requires an approved GPU host")),
    pytest.param("gpu:1", marks=pytest.mark.skipif(os.environ.get("EXONYM_REQUIRE_GPU") != "1",
                                                reason="physical GPU acceptance requires an approved GPU host"))])
def test_complete_background_integrand_matches_scalar(kind, hosted, measured_curve, execution_device):
    population = background_star_population(
        synthetic_trex_background_integrand_population(), simulated_area_sq_deg=1.0,
        target_tess_mag=10.0, resolution_limit_arcsec=1.0,
        contrast_separations=np.asarray([0.25, 0.5, 1.0]) if measured_curve else None,
        contrast_values=np.asarray([1.0, 3.0, 5.0]) if measured_curve else None,
    )
    scene = synthetic_trex_tp_integrand_case() if kind == "tp" else synthetic_trex_eb_integrand_case(2.0)
    constructor = _TransitingPlanetIntegrand if kind == "tp" else _EclipsingPairIntegrand
    kwargs = dict(scene, host_mass=1.0, host_radius=1.0, nsamples=3,
                  bin_integration_days=scene["exptime_days"], population=population,
                  n_dimensions=6, hosted=hosted, target_feh=0.0)
    if kind == "tp":
        kwargs["use_flat_priors"] = False
    scalar = constructor(**kwargs)
    platform, _, index = execution_device.partition(":")
    device = models.jax.devices(platform)[int(index or 0)]
    native = scalar.jax_unit_integrand(device=device)
    points = np.random.default_rng(20261016).uniform(size=(48, 6))
    # Include every discrete host and both sides of the imaging table range;
    # this is declared development coverage, not tuning on a held-out scene.
    points[:6, 4] = np.tile([0.0, 0.5, np.nextafter(1.0, 0.0)], 2)
    points[:3, 5], points[3:6, 5] = 0.0, np.nextafter(1.0, 0.0)
    parameters = np.asarray([scalar.prior_transform(point) for point in points])
    placed = models.jax.device_put(points, device)
    transformed = models.jax.jit(models.jax.vmap(native.prior_transform))(placed)
    np.testing.assert_allclose(transformed, parameters, rtol=1e-12, atol=1e-14)
    expected = np.asarray([scalar.log_likelihood(value) for value in parameters])
    evaluated = models.jax.jit(models.jax.vmap(native.evaluate))(placed)
    assert evaluated.log_integrand.devices() == {device}
    assert evaluated.log_integrand.dtype == np.float64
    assert np.asarray(evaluated.numerically_valid).all()
    np.testing.assert_array_equal(np.isneginf(evaluated.log_integrand), np.isneginf(expected))
    finite = np.isfinite(expected)
    assert finite.any()
    np.testing.assert_allclose(np.asarray(evaluated.log_integrand)[finite], expected[finite], rtol=1e-4, atol=1e-10)
    assert native.claim_eligible is False
    program = str(models.jax.make_jaxpr(models.jax.vmap(native.evaluate))(points))
    assert "pure_callback" not in program and "io_callback" not in program
