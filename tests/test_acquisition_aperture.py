"""Calibrated PRFs sample a pixel-integrated detector response once."""

import numpy as np
import pytest
from scipy.special import ndtr

from trex.acquisition import aperture as acquisition
from trex.acquisition.aperture import (
    ApertureSceneError, acquire_aperture_images, acquire_catalog_aperture, detector_prf_image,
    fit_prf_scene, fit_sector_image, fit_target_offset, load_empirical_prf_template,
    out_of_transit_image, pipeline_aperture_mask, source_positions, target_detector_position,
)
from trex.acquisition.transport import AcquisitionError


def test_pipeline_aperture_mask_requires_spoc_bit_and_matching_shape():
    np.testing.assert_array_equal(pipeline_aperture_mask([[1, 2, 3]], (1, 3)), [[False, True, True]])
    with pytest.raises(ApertureSceneError, match="no declared"):
        pipeline_aperture_mask([[0, 1]], (1, 2))


def test_out_of_transit_image_excludes_bad_quality_and_transits():
    cube = {"time": [1., 2., 3., 4.], "quality_retained": [True, True, False, True],
            "flux": np.array([[[2.]], [[100.]], [[200.]], [[4.]]])}
    np.testing.assert_array_equal(out_of_transit_image(cube, [False, True, False, False]), [[3.]])
    with pytest.raises(ApertureSceneError, match="no finite quality-zero"):
        out_of_transit_image(cube, [True, True, True, True])


def test_prf_fits_normalization_and_invalid_input(tmp_path):
    from astropy.io import fits

    path = tmp_path / "prf.fits"
    fits.writeto(path, np.array([[1., 3.], [0., 0.]]))
    np.testing.assert_array_equal(load_empirical_prf_template(path), [[0.25, 0.75], [0., 0.]])
    fits.writeto(path, np.array([[1., -1.]]), overwrite=True)
    with pytest.raises(RuntimeError, match="finite non-negative"):
        load_empirical_prf_template(path)


def test_source_projection_uses_equatorial_wcs_and_rejects_duplicate_ids():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10., 20.]
    wcs.wcs.crpix = [5., 6.]
    wcs.wcs.cdelt = [-0.01, 0.01]
    target = {"source_id": "1", "ra_deg": 10., "dec_deg": 20.}
    positions = source_positions(wcs.to_header(), target, [])
    assert positions[0]["x_pixel"] == pytest.approx(4.)
    assert positions[0]["y_pixel"] == pytest.approx(5.)
    with pytest.raises(ApertureSceneError, match="duplicate"):
        source_positions(wcs.to_header(), target, [target])


def test_official_prf_acquisition_retains_nearest_grid_and_validates(tmp_path, monkeypatch):
    from astropy.io import fits

    position = {"camera": 1, "ccd": 2, "absolute_row": 200., "absolute_column": 300.}
    monkeypatch.setattr(acquisition, "target_detector_position", lambda _: position)
    listing = (b'<a href="tess123-prf-1-2-row0100-col0100.fits">a</a>'
               b'<a href="tess123-prf-1-2-row0200-col0300.fits">b</a>')

    class Transport:
        transport_failures = []

        def _http_get_bytes(self, url, accept):
            assert url.endswith("start_s0001/cam1_ccd2/") and accept == "text/html"
            return listing

        def download(self, url, path):
            assert url.endswith("row0200-col0300.fits")
            primary = fits.PrimaryHDU(np.ones((117, 117)))
            primary.header["NSAMP"] = 9
            primary.header["DATATYPE"] = "PRF"
            fits.HDUList([primary, fits.ImageHDU(np.zeros((117, 117)))]).writeto(path)
            return path

    path, record = acquisition.acquire_official_prf({"sector": 1}, tmp_path, Transport())
    assert path.is_file() and record["status"] == "available"
    assert (tmp_path / record["listing"]["path"]).read_bytes() == listing
    assert record["product"]["field_position"] == {"row": 200, "column": 300}


def test_aperture_acquisition_binds_cadence_mask_and_raw_inputs(tmp_path, monkeypatch):
    from trex.acquisition import tpf

    cube = {"time": np.array([1., 2.]), "quality_retained": [True, True],
            "flux": np.array([[[1.]], [[4.]]]), "aperture": np.array([[2]]), "aperture_header": {}}
    monkeypatch.setattr(tpf, "acquire_tpf_products", lambda *_: [{"sector": 1, "path": "tpf.fits"}])
    monkeypatch.setattr(tpf, "load_tpf", lambda *_: cube)
    monkeypatch.setattr(acquisition, "source_positions", lambda *_: [{"source_id": "1"}])
    monkeypatch.setattr(acquisition, "acquire_official_prf", lambda *_: (tmp_path / "prf.fits", {
        "product_structure": {"samples_per_pixel": 9}, "product": {"path": "prf.fits"}}))
    monkeypatch.setattr(acquisition, "load_empirical_prf_template", lambda *_: np.ones((3, 3)))
    monkeypatch.setattr(acquisition, "fit_sector_image", lambda image, *_: {"modeled_aperture_flux_native": float(image[0, 0])})
    result = acquisition.acquire_aperture_images("123", [1], {}, [], {1: [False, True]}, tmp_path, None)
    sector = result["sectors"][0]
    assert sector["modeled_aperture_flux_native"] == 1.
    with np.load(tmp_path / sector["transit_mask"]) as saved:
        np.testing.assert_array_equal(saved["time_btjd_tdb"], [1., 2.])
        np.testing.assert_array_equal(saved["in_transit"], [False, True])
    assert sector["raw_tpf"]["path"] == "tpf.fits" and sector["prf"]["product"]["path"] == "prf.fits"


def test_detector_prf_matches_independent_pixel_integral():
    samples, sigma = 9, 0.7
    axis = np.arange(-58, 59, dtype=float) / samples
    integrated = ndtr((axis + 0.5) / sigma) - ndtr((axis - 0.5) / sigma)
    template = np.outer(integrated, integrated)
    normalization = template.sum()
    template /= normalization
    yy, xx = np.indices((11, 11), dtype=float)
    x, y = 5 + 2 / samples, 5 - 1 / samples
    expected = ((ndtr((xx - x + 0.5) / sigma) - ndtr((xx - x - 0.5) / sigma))
                * (ndtr((yy - y + 0.5) / sigma) - ndtr((yy - y - 0.5) / sigma)) / normalization)
    np.testing.assert_allclose(detector_prf_image(expected.shape, x, y, template, samples), expected, rtol=0, atol=2e-17)
    with pytest.raises(ApertureSceneError, match="square"):
        detector_prf_image((3, 3), 1, 1, np.ones((3, 4)), samples)


def test_prf_fit_recovers_native_source_flux_and_rejects_exhausted_solver():
    from scipy.optimize import OptimizeResult

    template = np.zeros((3, 3))
    template[1, 1] = 1.0
    image = np.full((5, 5), 10.0)
    image[2, 2] += 100.0
    positions = [{"source_id": "123", "is_target": True, "x_pixel": 2.0, "y_pixel": 2.0}]
    result = fit_prf_scene(image, np.ones((5, 5), dtype=bool), positions, template, 1)
    assert result["amplitudes"][0] == pytest.approx(100.0, abs=1e-10)
    assert result["background_native"] == pytest.approx(10.0, abs=1e-10)
    with pytest.raises(ApertureSceneError, match="did not converge.*status=0"):
        fit_prf_scene(image, np.ones((5, 5), dtype=bool), positions, template, 1,
                      solver=lambda *_: OptimizeResult(success=False, status=0, nit="unreported", message="exhausted"))


def test_target_phase_recovers_offset_with_background():
    axis = np.arange(-40, 41, dtype=float) / 9
    template = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 0.7 ** 2))
    positions = [{"source_id": "123", "is_target": True, "x_pixel": 4.0, "y_pixel": 4.0}]
    image = 1000 * detector_prf_image((9, 9), 4.2, 3.9, template, 9) + 12
    offset = fit_target_offset(image, np.ones((9, 9), dtype=bool), positions, template, 9)
    assert offset == pytest.approx((0.2, -0.1), abs=1e-5)


def test_sector_flux_fractions_follow_native_aperture_flux():
    positions = [{"source_id": "1", "x_pixel": 0, "y_pixel": 0, "is_target": True},
                 {"source_id": "2", "x_pixel": 1, "y_pixel": 0, "is_target": False}]
    response = [np.array([[0.5, 0.5]]), np.array([[0.25, 0.75]])]
    fit = {"amplitudes": np.array([2., 4.]), "response_images": response, "residual": 0.,
           "fit_pixel_count": 2, "background_native": 0., "target_offset_pixels": {"dx": 0., "dy": 0.}}
    result = fit_sector_image(np.ones((1, 2)), np.array([[True, False]]), positions, None, 1,
                              offset_fitter=lambda *_: (0., 0.), fitter=lambda *_, **__: fit)
    assert [row["aperture_flux_fraction"] for row in result["source_contributions"]] == [0.5, 0.5]


def test_catalog_tess_magnitudes_refuses_a_gless_source_whatever_catalog_epoch_is_offered():
    from trex.acquisition.aperture import catalog_tess_magnitudes

    target = {"source_id": "target", "phot_g_mean_mag": 10.0}
    # TIC v8 Gaia DR2 G and DR3 Bronze G are measurements at the 2014-2016 and
    # 2014-2017 epochs.  They do not bracket TESS sector 65 in 2023 and cannot
    # bound a source that brightens afterwards, so even a supplied G bound is
    # not accepted as an upper flux limit.
    gless = {
        "source_id": "gless",
        "phot_g_mean_mag": None,
        "phot_bp_mean_mag": 21.0353,
        "phot_rp_mean_mag": 18.7181,
        "phot_g_mean_mag_upper_limit": 20.4565,
        "phot_g_mean_mag_upper_limit_reason": "tic-v8-gaia-dr2-g",
        "phot_g_mean_mag_upper_limit_source": "TIC v8 GAIAmag (Gaia DR2)",
    }
    with pytest.raises(ApertureSceneError, match="gless \\(no measured G magnitude") as exc_info:
        catalog_tess_magnitudes([target, gless])
    message = str(exc_info.value)
    assert "fewer than 10 valid G transits" in message
    assert "Riello et al. 2021" in message
    assert "2014-2016" in message
    assert "not an upper bound" in message

    with pytest.raises(ApertureSceneError, match="target \\(target lacks a measured G"):
        catalog_tess_magnitudes([{**target, "phot_g_mean_mag": None}, {"source_id": "ok", "phot_g_mean_mag": 12.0}])

    assert catalog_tess_magnitudes([target, {"source_id": "ok", "phot_g_mean_mag": 12.0}]) == [
        (9.57, "stassun2019-eq2-no-valid-color"),
        (11.57, "stassun2019-eq2-no-valid-color"),
    ]


def _tpk_header():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [3.0, 4.0]
    wcs.wcs.cdelt = [-0.01, 0.01]
    header = wcs.to_header()
    header["CRVAL1P"], header["CRVAL2P"] = 100.0, 200.0
    return header


def test_target_detector_position_uses_the_loaded_aperture_header(tmp_path):
    cube = {"header": {"CAMERA": 1, "CCD": 2}, "aperture_header": _tpk_header(),
            "flux": np.zeros((1, 8, 8)), "path": tmp_path / "missing.fits"}
    position = target_detector_position(cube)
    assert position["local_x"] == pytest.approx(2.0)
    assert position["local_y"] == pytest.approx(3.0)
    assert position["absolute_column"] == pytest.approx(102.0)
    assert position["absolute_row"] == pytest.approx(203.0)
    assert position["camera"] == 1 and position["ccd"] == 2


def test_target_detector_position_reopens_the_tpf_only_without_a_loaded_header(tmp_path):
    from astropy.io import fits

    path = tmp_path / "tpf.fits"
    primary = fits.PrimaryHDU()
    primary.header.update(CAMERA=1, CCD=2)
    fits.HDUList([primary, fits.ImageHDU(), fits.ImageHDU(header=_tpk_header())]).writeto(path)
    cube = {"header": {"CAMERA": 1, "CCD": 2}, "flux": np.zeros((1, 8, 8)), "path": path}
    assert target_detector_position(cube) == target_detector_position(
        {**cube, "aperture_header": _tpk_header()})


def test_sector_fit_requires_the_target_first_and_once():
    positions = [{"source_id": "1", "is_target": False, "x_pixel": 0.0, "y_pixel": 0.0}]
    with pytest.raises(ApertureSceneError, match="target first and only once"):
        fit_sector_image(np.ones((3, 3)), np.ones((3, 3), dtype=bool), positions, None, 1,
                         offset_fitter=lambda *_: (0.0, 0.0), fitter=lambda *_, **__: {})
    with pytest.raises(ApertureSceneError, match="target first and only once"):
        fit_sector_image(np.ones((3, 3)), np.ones((3, 3), dtype=bool),
                         [{"source_id": "1", "is_target": True, "x_pixel": 0.0, "y_pixel": 0.0},
                          {"source_id": "2", "is_target": True, "x_pixel": 1.0, "y_pixel": 1.0}],
                         None, 1, offset_fitter=lambda *_: (0.0, 0.0), fitter=lambda *_, **__: {})


def test_the_fitted_offset_is_reported_on_the_target_row_only():
    positions = [{"source_id": "1", "is_target": True, "x_pixel": 2.0, "y_pixel": 2.0},
                 {"source_id": "2", "is_target": False, "x_pixel": 3.0, "y_pixel": 2.0}]
    response = [np.full((3, 3), 0.5), np.full((3, 3), 0.5)]
    fit = {"amplitudes": np.array([2.0, 1.0]), "response_images": response, "residual": 0.0,
           "fit_pixel_count": 9, "background_native": 0.0,
           "target_offset_pixels": {"dx": 0.2, "dy": -0.1}}
    result = fit_sector_image(np.ones((3, 3)), np.ones((3, 3), dtype=bool), positions, None, 1,
                              offset_fitter=lambda *_: (0.2, -0.1), fitter=lambda *_, **__: fit)
    target, neighbour = result["source_contributions"]
    assert (target["x_pixel"], target["y_pixel"]) == (pytest.approx(2.2), pytest.approx(1.9))
    assert (neighbour["x_pixel"], neighbour["y_pixel"]) == (3.0, 2.0)


def test_neighbour_phase_responses_are_cached_across_the_target_search(monkeypatch):
    original = acquisition.detector_prf_image
    calls = []

    def counted(shape, x, y, template, oversampling):
        calls.append((x, y))
        return original(shape, x, y, template, oversampling)

    monkeypatch.setattr(acquisition, "detector_prf_image", counted)
    axis = np.arange(-40, 41, dtype=float) / 9
    template = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 0.7 ** 2))
    positions = [{"source_id": "1", "is_target": True, "x_pixel": 4.0, "y_pixel": 4.0},
                 {"source_id": "2", "is_target": False, "x_pixel": 5.0, "y_pixel": 4.0}]
    image = (1000 * original((9, 9), 4.2, 3.9, template, 9)
             + 300 * original((9, 9), 5.0, 4.0, template, 9) + 5)
    offset = fit_target_offset(image, np.ones((9, 9), dtype=bool), positions, template, 9)
    assert offset == pytest.approx((0.2, -0.1), abs=1e-3)
    assert calls.count((5.0, 4.0)) == 1

    calls.clear()

    def uncached_fitter(image, aperture, positions, template, oversampling, *, target_offset=(0.0, 0.0),
                        background=True, pixel_response=None, **kwargs):
        return fit_prf_scene(image, aperture, positions, template, oversampling,
                             target_offset=target_offset, background=background, **kwargs)

    uncached = fit_target_offset(image, np.ones((9, 9), dtype=bool), positions, template, 9,
                                 fitter=uncached_fitter)
    assert uncached == offset
    assert calls.count((5.0, 4.0)) > 1


@pytest.mark.parametrize("function, match", [
    (lambda output: acquire_aperture_images("123", [1], {}, [], {1: [False]}, output, None),
     "no SPOC target-pixel sector"),
    (lambda output: acquire_catalog_aperture(
        "123", [1], {"source_id": "1", "phot_g_mean_mag": 10.0}, [], output, None),
     "no SPOC target-pixel sector"),
])
def test_no_retained_tpf_product_is_an_acquisition_error(tmp_path, monkeypatch, function, match):
    from trex.acquisition import tpf

    monkeypatch.setattr(tpf, "acquire_tpf_products", lambda *_args: [])
    with pytest.raises(AcquisitionError, match=match):
        function(tmp_path)
