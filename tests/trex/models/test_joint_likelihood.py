"""Independent Gaussian oracles and connected multi-aperture forward models."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from trex.foundation.constants import Rsun
from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.baseline import GaussianBaseline
from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import REFERENCE, observation


def test_ragged_heteroscedastic_density_is_sum_of_normalized_reported_errors():
    rows = (observation(flux=[1., .98, 1.01], sigma=[.01, .03, .02]),
            observation("ground", time=[0., .01], flux=[1., .99], sigma=.1))
    likelihood = JointLikelihood(ObservationSet(rows, REFERENCE))
    models = {row.dataset_id: np.ones(row.flux.size) for row in rows}
    actual, details = likelihood.density(models)
    expected = sum(norm.logpdf(row.flux, loc=1., scale=row.sigma).sum() for row in rows)
    assert actual == pytest.approx(expected, abs=2e-12)
    assert actual == pytest.approx(sum(item["log_density"] for item in details.values()), abs=2e-12)


def test_shared_nuisance_is_one_gaussian_convolution_not_two_independent_priors():
    rows = (observation(flux=[1.02, .99, 1.01]), observation("ground", flux=[1.03, 1., 1.01]))
    times = np.concatenate([row.time_axis.time_days for row in rows])
    prior = GaussianBaseline(times, np.ones((6, 1)), np.array([0.]),
                             np.array([[.04**2]]), "additive", "synthetic external prior",
                             stream_ids=("space",)*3 + ("ground",)*3)
    likelihood = JointLikelihood(ObservationSet(rows, REFERENCE), shared_baseline=prior)
    actual, details = likelihood.density({row.dataset_id: np.ones(3) for row in rows})
    flux = np.concatenate([row.flux for row in rows])
    covariance = np.eye(6)*.02**2 + np.ones((6, 6))*.04**2
    expected = multivariate_normal.logpdf(flux, mean=np.ones(6), cov=covariance)
    assert actual == pytest.approx(expected, abs=3e-11)
    assert all(item["log_density"] is None for item in details.values())
    with pytest.raises(ValueError, match="joint prior"):
        JointLikelihood(ObservationSet((replace(rows[0], baseline=GaussianBaseline(
            rows[0].time_axis.time_days, np.ones((3, 1)), np.array([0.]),
            np.array([[.04**2]]), "additive", "private prior")), rows[1]), REFERENCE), shared_baseline=prior)


def test_zero_share_predicts_flat_without_removing_the_host_in_the_other_aperture():
    rows = (observation(), observation("ground", shares={"target": 0.}))
    likelihood = JointLikelihood(ObservationSet(rows, REFERENCE))
    models = likelihood.models(
        host_id="target", light_by_dataset={row.dataset_id: {"u1": 0., "u2": 0.} for row in rows},
        period_days=3., epoch_days=0., inc_deg=90., a_cm=10*Rsun, host_radius_solar=1.,
        planet_radius_earth=10.)
    assert models["space"][1] < 1.
    np.testing.assert_array_equal(models["ground"], 1.)
    unknown = replace(rows[1], source_shares={"target": None})
    with pytest.raises(TrexEvidenceUnavailableError, match="unknown aperture"):
        JointLikelihood(ObservationSet((unknown,), REFERENCE)).models(
            host_id="target", light_by_dataset={"ground": {}}, period_days=3., epoch_days=0.,
            inc_deg=90., a_cm=10*Rsun, host_radius_solar=1., planet_radius_earth=10.)


def test_dense_heteroscedastic_shared_draw_matches_direct_numpy_at_1e_12():
    """Independent dense density oracle for one shared physical transit draw.

    Both datasets see the same orbit/epoch/occultor; only their cadence and
    reported sigma differ. The expected value is the docstring formula
    ``sum_d [-0.5 (chi2_d + n_d log 2pi + 2 sum log sigma_d)]`` computed with
    plain numpy from the engine's model arrays. Tolerance: abs 2e-12 nat,
    about 10x the observed float64 summation error of the existing flat case.
    """
    space_time = np.linspace(-0.09, 0.09, 220)
    ground_time = np.linspace(-0.06, 0.06, 80)
    space_sigma = np.linspace(8e-4, 2.5e-3, space_time.size)
    ground_sigma = np.full(ground_time.size, 6e-3)
    rows = (
        observation(time=space_time, flux=1.0 + 2e-4*np.sin(40*space_time),
                    sigma=space_sigma),
        observation("ground", time=ground_time,
                    flux=1.0 + 5e-4*np.cos(25*ground_time), sigma=ground_sigma),
    )
    likelihood = JointLikelihood(ObservationSet(rows, REFERENCE))
    models = likelihood.models(
        host_id="target",
        light_by_dataset={row.dataset_id: {"u1": .25, "u2": .3} for row in rows},
        period_days=3.2, epoch_days=0.004, inc_deg=88.5, a_cm=9.5*Rsun,
        host_radius_solar=1.0, planet_radius_earth=11.0,
    )
    # The shared draw must actually move both heteroscedastic streams.
    assert not np.allclose(models["space"], 1.0) and not np.allclose(models["ground"], 1.0)

    expected = 0.0
    for row in rows:
        model = models[row.dataset_id]
        assert model.shape == row.flux.shape
        residual = (row.flux - model) / row.sigma
        expected -= 0.5 * (float(residual @ residual)
                           + row.flux.size * np.log(2*np.pi)
                           + 2*np.log(row.sigma).sum())
    actual, details = likelihood.density(models)
    assert actual == pytest.approx(expected, abs=2e-12, rel=0)
    for row in rows:
        residual = (row.flux - models[row.dataset_id]) / row.sigma
        assert details[row.dataset_id]["reported_error_chi2"] == pytest.approx(
            float(residual @ residual), abs=2e-12, rel=0)
