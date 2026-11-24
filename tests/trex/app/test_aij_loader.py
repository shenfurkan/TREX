import numpy as np
import pytest
from astropy.time import Time

from trex.app.aij_loader import read_aij_observation
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.observations import ObservationSet


@pytest.fixture
def aij_table(tmp_path):
    path = tmp_path / "Measurements.tbl"
    header = "\tLabel\tBJD_TDB\trel_flux_T1\trel_flux_err_T1\tAIRMASS\tFWHM\n"
    late = "1\tlate.fits\t2459000.1350000003\t.51\t.002\t1.2\t4\n"
    early = "2\tearly.fits\t2459000.1250000001\t.49\t.001\t1.4\t3\n"
    path.write_text(header + late + early, encoding="utf-8")
    kwargs = dict(
        dataset_id="ground", segment_id="visit", band_id="test/camera.g", response_id="test-g-asset",
        reference=Time(2459000, .125, format="jd", scale="tdb"),
        observer="test site", target="test target", stamp="mid", exposure_days=[.002, .001],
        exposure_source="test exposure headers", nsamples=2, aperture_id="test aperture",
        source_shares={"target": [.8, .9], "neighbor": None}, lineage=[["late-id"], ["early-id"]],
        flux_scale=.5, uncertainty_source="synthetic independent normalization and table errors",
        prior_mean=[0., 0.], prior_covariance=[[1e-4, 0.], [0., 4e-4]],
        baseline_mode="additive", baseline_source="synthetic independent prior, constant per exposure",
    )
    return path, kwargs


def test_aij_observation_keeps_aligned_rows_and_reaches_joint_likelihood(aij_table):
    path, kwargs = aij_table
    row = read_aij_observation(path, **kwargs)
    np.testing.assert_allclose(row.time_axis.time_days, [1e-10, .0100000003], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(row.raw_rows, [1, 0])
    assert row.raw_count == 2
    assert row.lineage == (frozenset({"early-id"}), frozenset({"late-id"}))
    np.testing.assert_array_equal(row.flux, [.98, 1.02])
    np.testing.assert_array_equal(row.sigma, [.002, .004])
    np.testing.assert_array_equal(row.source_shares["target"], [.9, .8])
    assert row.source_shares["neighbor"] is None
    np.testing.assert_array_equal(row.exposure_response.offset_days, [[-.00025, .00025], [-.0005, .0005]])
    np.testing.assert_array_equal(row.baseline.design, [[1.4, 3], [1.2, 4]])

    likelihood = JointLikelihood(ObservationSet((row,), kwargs["reference"]))
    actual, diagnostics = likelihood.density({"ground": np.ones(2)})
    design = np.array([[1.4, 3.], [1.2, 4.]])
    covariance = np.diag([.002**2, .004**2]) + design @ np.diag([1e-4, 4e-4]) @ design.T
    residual = np.array([-.02, .02])
    expected = -.5 * (residual @ np.linalg.solve(covariance, residual)
                      + np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi))
    assert actual == pytest.approx(expected, abs=1e-12)
    assert diagnostics["ground"]["band_id"] == "test/camera.g"


@pytest.mark.parametrize(("before", "after", "message"), [
    ("AIRMASS", "altitude", "column"),
    ("2459000.1250000001", "59000.125", "full BJD_TDB"),
    ("2459000.1250000001", "2459000.1350000003", "strictly increasing"),
    ("\t.001\t1.4", "\t0\t1.4", "positive"),
    ("\t1.4\t3", "\tNaN\t3", "finite measurements"),
])
def test_aij_rejects_invalid_measurement_rows(aij_table, before, after, message):
    path, kwargs = aij_table
    path.write_text(path.read_text(encoding="utf-8").replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_aij_observation(path, **kwargs)


def test_aij_csv_matches_native_tab_export(aij_table):
    path, kwargs = aij_table
    csv_path = path.with_suffix(".csv")
    csv_path.write_text(path.read_text(encoding="utf-8").replace("\t", ","), encoding="utf-8")
    csv_row = read_aij_observation(csv_path, **kwargs).to_dict()
    tab_row = read_aij_observation(path, **kwargs).to_dict()
    csv_row["time_axis"]["source"] = tab_row["time_axis"]["source"]
    assert csv_row == tab_row
