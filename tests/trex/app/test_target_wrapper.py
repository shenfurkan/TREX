"""Honest acquisition and fail-closed coverage of the portable wrapper.

The abandoned success fixture overwrote computed ESS and assumed calibrated
evidence from an approximate Gaussian. It is retained in pre-shortcut-repair.zip.
Only external acquisition is mocked here; no scientific output is overwritten.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.integrate import dblquad

import trex.app.target as module
from tests.fixtures.synthetic_observations import synthetic_portable_trex_inputs


def _catalog():
    return Table({
        "ID": ["100001", "100002", "100003"], "Tmag": [10., 11.5, 20.],
        "Jmag": [9., 10., 19.], "Hmag": [9., 10., 19.], "Kmag": [9., 10., 19.],
        "ra": [120., 120.004, 120.], "dec": [-30., -30., -29.996],
        "mass": [1.02, .82, .55], "rad": [.98, .79, .52],
        "Teff": [5778., 5100., 3900.], "plx": [10., 9.5, 4.],
        "disposition": ["", "", ""], "duplicate_id": ["", "", ""],
    })


def _cutout(valid_wcs=True):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [11, 11]
    wcs.wcs.cdelt = [-20.25 / 3600, 20.25 / 3600]
    wcs.wcs.crval = [120, -30]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    flux = fits.Column(name="FLUX", format="484D", dim="(22,22)", array=np.full((4, 22, 22), 150.))
    table = fits.BinTableHDU.from_columns([flux])
    table.header["1CRV4P"], table.header["2CRV4P"] = 100, 200
    image = fits.ImageHDU(np.zeros((22, 22)), header=wcs.to_header() if valid_wcs else fits.Header())
    return fits.HDUList([fits.PrimaryHDU(), table, image])


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setattr(module, "_query_tic_catalog", lambda *_a, **_k: _catalog())

    def install(sectors=(1, 2), valid_wcs=True):
        def search(target, sector):
            products = [SimpleNamespace(hdu=_cutout(valid_wcs))] if sector in sectors else []
            return SimpleNamespace(download_all=lambda **_k: products)
        monkeypatch.setattr(module, "_search_tesscut", search)
    install()
    return install


def test_public_method_parameter_contract():
    # Parameter order from retained official TRICERATOPS 1.0.20 source, not a
    # runtime dependency on an ignored tools/ receipt or the official package.
    expected = {
        "__init__": "self ID sectors search_radius mission lightkurve_cache_dir trilegal_fname ra dec verify_ssl",
        "add_star": "self ID Tmag bound", "remove_star": "self drop_stars",
        "update_star": "self ID param value", "get_spoc_apertures": "self",
        "plot_field": "self sector ap_pixels ap_color save fname",
        "calc_depths": "self tdepth all_ap_pixels",
        "calc_probs": "self time flux_0 flux_err_0 P_orb contrast_curve_file filt N parallel drop_scenario verbose flatpriors exptime nsamples molusc_file",
        "plot_fits": "self time flux_0 flux_err_0 save fname",
    }
    for name, names in expected.items():
        parameters = inspect.signature(getattr(module.target, name)).parameters
        # Upstream positional order is preserved; TREX-only recovery parameters may follow, defaulting to None.
        assert list(parameters)[:len(names.split())] == names.split()
        extra = list(parameters.values())[len(names.split()):]
        assert all(p.name.startswith("escalation_") and p.default is None for p in extra)
    assert inspect.signature(module.target.calc_probs).parameters["N"].default == 1_000_000


def test_table_mutations_preserve_source_coordinate_alignment(transport):
    obj = module.target(100001, np.array([1, 2]))
    assert list(obj.stars.columns) == list(module.STARS_COLUMNS)
    assert obj.claim_eligible is False
    saved = [pixels[2].copy() for pixels in obj.pix_coords]
    with pytest.raises(ValueError, match="target"):
        obj.remove_star(100001)
    assert obj.remove_star(100002) is None
    assert obj.stars["ID"].tolist() == ["100001", "100003"]
    for xy, expected in zip(obj.pix_coords, saved, strict=True):
        np.testing.assert_array_equal(xy[1], expected)
    assert obj.add_star(999001, 13.2, True) is None
    assert obj.add_star(999002, 14.1, False) is None
    assert np.isnan(obj.stars.loc[2, "mass"])
    assert obj.stars.loc[2, "plx"] == 10
    assert np.isnan(obj.stars.loc[3, "plx"])
    with pytest.raises(ValueError, match="already exists"):
        obj.add_star(999001, 13, False)
    old = obj.pix_coords[0][1].copy()
    assert obj.update_star(100003, "ra", 120.008) is None
    assert not np.array_equal(old, obj.pix_coords[0][1])


def test_acquired_target_accepts_the_same_calibrated_scene_contract(transport, tmp_path):
    scene, binding, *_ = synthetic_portable_trex_inputs(tmp_path)
    obj = module.target(100001, np.array([1, 2]))
    scene.M_s_Msun = .95  # Explicit synthetic calibrated value, not a mass estimate.
    assert obj.bind_scene(scene, **binding) is None
    assert obj.stars.loc[0, "mass"] == .95
    assert obj._scene_binding["scene"].M_s_Msun == .95
    assert obj.depth_provenance is None  # No Gaussian is promoted to calibration.
    assert all(pixels.shape == (3, 2) for pixels in obj.pix_coords)
    scene.M_s_Msun = 2.0
    assert obj._scene_binding["scene"].M_s_Msun == .95


@pytest.mark.parametrize("sectors", [[1.5], [True], [1, 1], [], [[1]]])
def test_no_sector_coercion_or_duplicate_weighting(transport, sectors):
    with pytest.raises(ValueError, match="sector"):
        module.target(100001, np.array(sectors))


def test_missing_sector_and_malformed_wcs_fail_closed(transport):
    transport(sectors=(1,))
    with pytest.raises(RuntimeError, match="sector 2"):
        module.target(100001, np.array([1, 2]))
    transport(valid_wcs=False)
    with pytest.raises(RuntimeError, match="WCS"):
        module.target(100001, np.array([1]))


def test_spoc_uses_aperture_bit_not_maximum_mask_value(transport, monkeypatch):
    def open_hdus(*_args):
        image = fits.ImageHDU(np.array([[2, 3], [7, 4]], dtype=np.int32))
        image.header["CRVAL1P"], image.header["CRVAL2P"] = 100, 200
        return fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(), image])
    monkeypatch.setattr(module, "_open_spoc_lc_hdulist", open_hdus)
    obj = module.target(100001, np.array([1]))
    np.testing.assert_array_equal(obj.get_spoc_apertures()[0], [[100, 200], [101, 200], [100, 201]])
    def missing(*_a):
        raise OSError("missing FITS")
    monkeypatch.setattr(module, "_open_spoc_lc_hdulist", missing)
    with pytest.raises(RuntimeError, match="SPOC"):
        obj.get_spoc_apertures()


def test_reference_depths_have_independent_integral_and_explicit_limits(transport):
    obj = module.target(100001, np.array([1]))
    aperture = np.array([[x, y] for x in range(109, 113) for y in range(209, 213)])
    with pytest.raises(ValueError, match="aperture"):
        obj.calc_depths(.008)
    with pytest.raises(ValueError, match="fractional"):
        obj.calc_depths(8000, [aperture])
    assert obj.calc_depths(.008, [aperture]) is None
    integrals = []
    for (mu_x, mu_y), magnitude in zip(obj.pix_coords[0], obj.stars["Tmag"], strict=True):
        amp = 10 ** (-.4 * (magnitude - 10))
        def density(y, x, mx, my, a):
            return a / (2 * np.pi * .75**2) * np.exp(-((x - mx)**2 + (y - my)**2) / (2 * .75**2))
        integrals.append(sum(dblquad(density, x - .5, x + .5, y - .5, y + .5,
                                     args=(mu_x, mu_y, amp))[0] for x, y in aperture))
    expected_fractions = np.array(integrals) / sum(integrals)
    np.testing.assert_allclose(obj.stars["fluxratio"], expected_fractions, rtol=1e-10, atol=1e-12)
    # The independent integral, not a guessed catalog brightness cutoff,
    # establishes that the declared faint control cannot host this depth.
    assert expected_fractions[2] < .008
    assert obj.depth_provenance["calibrated"] is False
    assert "100003" in obj.excluded_impossible_hosts
    assert np.isnan(obj.stars.loc[2, "tdepth"])  # unavailable, not a fabricated fitted zero
    obj.update_star(100002, "mass", np.nan)
    assert "tdepth" not in obj.stars
    assert obj.depth_provenance is None


def test_unbound_probability_integration_and_fit_replay_never_fabricate_outputs(transport, tmp_path):
    obj = module.target(100001, np.array([1]))
    t, f = np.linspace(-.08, .08, 80), np.ones(80)
    with pytest.raises(ValueError, match="calibrated"):
        obj.calc_probs(t, f, .002, 3.2, N=64)
    assert obj.probs is obj.FPP is obj.NFPP is obj.trex_result is None
    with pytest.raises(ValueError, match="calibrated"):
        obj.plot_fits(t, f, .002, save=True, fname=str(tmp_path / "invalid"))
    assert not (tmp_path / "invalid.pdf").exists()
    obj.plot_field(sector=1, save=True, fname=str(tmp_path / "field"))
    assert (tmp_path / "field.pdf").is_file()


def test_upstream_calc_probs_selects_a_recovery_directory_without_creating_it_before_inference(
        tmp_path, monkeypatch):
    """The wrapper selects the path; the escalation owner creates it only when recovery work is needed."""
    from pathlib import Path

    import trex.engine

    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path)
    obj = module.target.from_scene(scene, **binding)
    monkeypatch.delenv("TREX_ESCALATION_CHECKPOINT_DIRECTORY", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    seen = {}

    def stop(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before Monte Carlo")

    monkeypatch.setattr(trex.engine, "run_trex_vetting", stop)
    with pytest.raises(RuntimeError, match="stop before Monte Carlo"):
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"], N=8, nsamples=4)
    directory = seen["escalation_checkpoint_directory"]
    assert directory == obj.escalation_checkpoint_directory
    assert Path(directory).parent == tmp_path / "state" / "trex" / "checkpoints"
    assert not Path(directory).exists()
    explicit = tmp_path / "chosen"
    with pytest.raises(RuntimeError):
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"], N=8, nsamples=4,
                       escalation_checkpoint_directory=str(explicit))
    assert seen["escalation_checkpoint_directory"] == str(explicit)
