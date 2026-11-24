"""Native-band measured magnitude ratios; no cross-filter transformation."""

import numpy as np
import pytest

from trex.photometry.photometric_relations import (
    delta_mag_to_flux_ratio,
    measured_same_band_flux_ratio,
)


def test_measured_magnitude_difference_is_a_flux_ratio_only_in_its_own_band():
    ratio = measured_same_band_flux_ratio(
        2.5, band_id="P200/WIRC.J_ori", reference_band_id="P200/WIRC.J_ori")
    assert ratio == pytest.approx(10.0 ** -1.0, rel=2e-16)
    assert np.asarray(ratio).ndim == 0
    np.testing.assert_allclose(
        measured_same_band_flux_ratio(
            np.array([0.0, 5.0]), band_id="SLOAN/SDSS.z",
            reference_band_id="SLOAN/SDSS.z"),
        np.array([1.0, 10.0 ** -2.0]), rtol=2e-16,
    )


def test_cross_band_and_bare_band_names_are_refused():
    with pytest.raises(ValueError, match="only within one native band"):
        measured_same_band_flux_ratio(
            1.0, band_id="SLOAN/SDSS.z", reference_band_id="PAN-STARRS/PS1.z")
    for bare in ("J", "z", "Ks", ""):
        with pytest.raises(ValueError, match="instrument-qualified"):
            measured_same_band_flux_ratio(
                1.0, band_id=bare, reference_band_id=bare)
    # Distinct instrument responses are never aliased even when the letter agrees.
    with pytest.raises(ValueError, match="only within one native band"):
        measured_same_band_flux_ratio(
            1.0, band_id="2MASS/2MASS.J", reference_band_id="P200/WIRC.J_ori")


def test_pure_pogson_conversions_are_unchanged():
    assert delta_mag_to_flux_ratio(0.0) == 1.0
    assert delta_mag_to_flux_ratio(-2.5) == pytest.approx(10.0, rel=2e-16)
    with pytest.raises(ValueError, match="finite"):
        delta_mag_to_flux_ratio(np.nan)
