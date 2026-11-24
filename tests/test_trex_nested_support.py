"""Pin nested EB support defects against G21 Sect. 2.4.4.

Giacalone et al. 2021, AJ 161, 24, DOI 10.3847/1538-3881/abc6af. The full-q prior
is unchanged: a wrong branch has zero likelihood, not a conditional-q prior.
These tests never run an evidence sampler.
"""

import numpy as np
import pytest

from trex.evidence import nested_evidence as nested
from trex.evidence import nested_integrands
from trex.populations import priors


def test_public_eb_support_partitions_the_exact_twin_boundary():
    boundary = priors.MOE2017_Q_TWIN_THRESHOLD
    q = np.array([np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)])
    np.testing.assert_array_equal(priors.eb_mass_ratio_support(q, False), [True, False, False])
    np.testing.assert_array_equal(priors.eb_mass_ratio_support(q, True), [False, True, True])


def _capture_eb(monkeypatch, *, twice, period=3.0, population=None, hosted=False):
    captured = {}

    def integrate(integrand, controls, scenario, n_dimensions, posterior):
        captured["integrand"] = integrand
        return nested.NestedEvidenceResult(0.0, 0.01, 400, 0, 0, "stub", "0", 1)

    monkeypatch.setattr(nested, "_integrate", integrate)
    nested.eb_scenario_nested_evidence(
        np.array([-0.01, 0.0, 0.01]), np.ones(3), 0.001, period, 1.0, 1.0, 0.4, 0.2,
        0.001, target_aperture_flux_fraction=1.0, target_feh=0.0, use_2xP=twice,
        population=None if hosted else population, hosted_population=population if hosted else None,
        random_seed=1, n_jobs=1)
    return captured["integrand"]


@pytest.mark.parametrize("twice", [False, True])
@pytest.mark.parametrize("population_kind", ["none", "diluting", "hosted"])
def test_nested_eb_rejects_only_the_other_q_branch(monkeypatch, twice, population_kind):
    population = None
    if population_kind != "none":
        population = nested.background_star_population(
            {"tess_mag": np.array([12.0]), "mass_solar": np.array([1.0]),
             "radius_solar": np.array([1.0]), "teff_k": np.array([5800.0]), "mh_dex": np.array([0.0])},
            0.1, 10.0, resolution_limit_arcsec=1.0)
    integrand = _capture_eb(monkeypatch, twice=twice, population=population,
                            hosted=population_kind == "hosted")
    monkeypatch.setattr(nested_integrands, "lnL_EB", lambda *args, **kwargs: 0.0)
    q_values = [0.5, np.nextafter(0.95, -np.inf), 0.95, 1.0]
    for q in q_values:
        parameters = np.array([q, 0.0, 90.0, 90.0] + ([] if population is None else [0.5, 0.5]))
        result = integrand.log_likelihood(parameters)
        assert np.isfinite(result) == (q >= 0.95 if twice else q < 0.95), (q, result)
    # Both slots retain the same normalized full-q inverse CDF; no branch
    # rescaling of u[0] or per-branch evidence factor may be introduced.
    for u in (0.1, 0.5, 0.99):
        unit = np.array([u, 0.5, 0.25, 0.5] + ([] if population is None else [0.5, 0.5]))
        assert integrand.prior_transform(unit)[0] == priors.sample_q_host_mass_ratio(
            np.array([u]), np.array([1.0]))[0]


def test_nested_twins_do_not_acquire_an_extra_100_day_cap(monkeypatch):
    integrand = _capture_eb(monkeypatch, twice=True, period=60.0)
    assert integrand.effective_period == 120.0
