"""Nested entry points and real prior/likelihood admission, without sampler stubs."""

import numpy as np
import pytest

from trex.evidence import nested_evidence as nested
from trex.evidence.nested_integrands import _EclipsingPairIntegrand, _TransitingPlanetIntegrand


def _integrand(binary, *, metallicity=0.0, host_mass=1.0):
    inputs = dict(
        time=np.array([-0.01, 0.0, 0.01]), flux=np.ones(3), sigma=0.02,
        host_mass=host_mass, host_radius=1.0, u1=0.4, u2=0.2,
        exptime_days=0.001, bin_integration_days=0.0, nsamples=1,
        population=None, n_dimensions=4, target_feh=metallicity, host_teff_k=5750.0,
    )
    if binary:
        return _EclipsingPairIntegrand(effective_period=3.0, **inputs)
    return _TransitingPlanetIntegrand(period_days=3.0, use_flat_priors=False, **inputs)


@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
@pytest.mark.parametrize("metallicity", [True, np.bool_(False), "0.0", np.nan])
def test_atmosphere_metallicity_is_not_coerced_from_boolean_or_text(binary, metallicity):
    with pytest.raises(ValueError, match="target_feh"):
        _integrand(binary, metallicity=metallicity)


@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
@pytest.mark.parametrize("point", [[], [0.5, 0.5], [[0.5] * 4], [True] * 4, [-0.1] * 4, [np.nan] * 4])
def test_prior_transform_refuses_invalid_unit_vectors_before_indexing(binary, point):
    with pytest.raises(ValueError, match="unit point"):
        _integrand(binary).prior_transform(point)


@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
def test_valid_prior_point_reaches_the_actual_likelihood(binary):
    integrand = _integrand(binary)
    parameters = integrand.prior_transform(np.array([0.5, 0.0, 0.25, 1.0]))
    assert parameters.shape == (4,) and np.isfinite(parameters).all()
    assert np.isfinite(integrand.log_likelihood(parameters))


def test_fixed_binary_refuses_an_unrepresentable_secondary_instead_of_raising():
    integrand = _integrand(True, host_mass=0.1)
    # 0.5 * 0.1 Msun is below the retained stellar grid's 0.1 Msun endpoint.
    assert integrand.log_likelihood(np.array([0.5, 0.0, 90.0, 90.0])) == -np.inf


@pytest.mark.parametrize("binary", [False, True], ids=["TP", "EB"])
def test_null_continuous_marginals_use_the_actual_discrete_host_dimension(binary, monkeypatch):
    monkeypatch.setenv("TREX_NESTED_ESTIMATOR", "dynesty")
    host = {
        "method": "synthetic-discrete", "mass_solar": [1.0], "radius_solar": [1.0],
        "teff_k": [5750.0], "weights": [1.0], "marginals": None,
    }
    call = nested.eb_scenario_nested_evidence if binary else nested.tp_scenario_nested_evidence
    # Dynesty's dynamic run targets >=10000 posterior ESS; a smaller call cap tests truncation, not dimensions.
    result = call(
        np.array([-0.01, 0.0, 0.01]), np.ones(3), 0.02, 3.0, 1.0, 1.0, 0.4, 0.2, 0.001,
        target_aperture_flux_fraction=1.0, target_feh=0.0, nsamples=1,
        random_seed=19, n_jobs=1, n_live_points=6, host_distribution=host,
    )
    assert result.n_live_points == 6
    assert result.posterior_unit_samples.shape[1] == 5
    assert np.isfinite(result.posterior_weights).all()
    assert result.posterior_weights.sum() == pytest.approx(1.0)
