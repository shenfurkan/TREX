"""Native-band measured imaging: independent nondetections and per-band limits.

Synthetic curves only; separations are arcsec and contrasts/limits delta-mag
(Giacalone et al. 2021, AJ 161, 24, DOI 10.3847/1538-3881/abc6af, Sect. 2.4.3).
The expected values follow the definition, not the engine: a companion at
delta magnitude ``-2.5 log10(F_companion/F_host)`` survives only where it is at
least as faint as the tightest limit whose measured support covers its
separation.  A separation outside a curve's support places no bound and is not
a detection, and a curve never bounds a filter it was not measured in.
"""

import numpy as np
import pytest

from trex.populations.companion_imaging import (
    ContrastCurve,
    MeasuredContrastSet,
    imaging_log_prior,
)


def _curve(separations, contrasts, *, band="TESS", model="step limit", source="synthetic"):
    return ContrastCurve(
        np.asarray(separations, dtype=float),
        np.asarray(contrasts, dtype=float),
        band_id=band,
        detection_model=model,
        source=source,
    )


def _flux_ratio(delta_mag):
    """F_companion/F_host for a delta magnitude, via the Pogson definition."""
    return 10.0 ** (-0.4 * delta_mag)


def test_same_band_curves_combine_as_independent_nondetections():
    broad = _curve([0.5, 1.0, 2.0], [1.0, 2.0, 5.0])
    narrow = _curve([0.5, 2.0], [3.0, 4.0])
    curves = MeasuredContrastSet((broad, narrow))

    limits = curves.at("TESS", np.array([0.5, 1.0, 2.0, 0.2, 2.5]))
    # Overlapping support: the surviving limit is the maximum (tightest) one.
    np.testing.assert_allclose(limits[:3], [3.0, 10.0 / 3.0, 5.0])
    # Inside both inner working angles and beyond both outer edges: no bound.
    assert np.isnan(limits[3:]).all()


def test_disjoint_same_band_supports_do_not_bridge_the_gap():
    left = _curve([0.5, 1.0], [1.0, 2.0])
    right = _curve([1.5, 2.0], [3.0, 4.0])
    curves = MeasuredContrastSet((left, right))

    limits = curves.at("TESS", np.array([0.75, 1.25, 1.75]))
    np.testing.assert_allclose(limits[[0, 2]], [1.5, 3.5])
    assert np.isnan(limits[1])


def test_inner_working_angle_places_no_constraint():
    curves = MeasuredContrastSet((_curve([1.0, 2.0], [2.0, 4.0]),))
    assert np.isnan(float(curves.at("TESS", 0.5)))

    twin = _flux_ratio(0.0)
    uncovered = curves.apply("TESS", np.zeros(1), [twin], [0.5])
    covered = curves.apply("TESS", np.zeros(1), [twin], [1.5])
    assert uncovered[0] == 0.0
    assert np.isneginf(covered[0])


def test_bands_are_never_combined_and_an_unknown_band_is_refused():
    tess = _curve([0.5, 2.0], [2.0, 3.0], band="TESS")
    johnson = _curve([0.5, 2.0], [6.0, 7.0], band="J")
    curves = MeasuredContrastSet((tess, johnson))

    assert curves.band_ids == ("TESS", "J")
    # Linear interpolation between (0.5, 2.0) and (2.0, 3.0) at 1.0 arcsec;
    # the same rule is pinned by the independent-nondetection test above.
    assert float(curves.at("TESS", 1.0)) == pytest.approx(7.0/3.0)
    assert float(curves.at("J", 1.0)) == pytest.approx(19.0/3.0)

    # 4 mag is fainter than TESS's 2.333 mag limit but brighter than J's
    # 6.333 mag limit: each band keeps its own curve instead of borrowing another.
    companion = _flux_ratio(4.0)
    assert curves.apply("TESS", np.zeros(1), [companion], [1.0])[0] == 0.0
    assert np.isneginf(curves.apply("J", np.zeros(1), [companion], [1.0])[0])

    with pytest.raises(ValueError, match="no measured contrast curve"):
        curves.at("Kepler", 1.0)
    with pytest.raises(ValueError, match="no measured contrast curve"):
        curves.apply("Kepler", np.zeros(1), [companion], [1.0])


@pytest.mark.parametrize("field", ["band_id", "detection_model", "source"])
def test_set_refuses_a_curve_without_declared_provenance(field):
    metadata = {"band_id": "TESS", "detection_model": "step limit", "source": "synthetic"}
    metadata[field] = ""
    curve = ContrastCurve(np.array([0.5, 1.5]), np.array([2.0, 3.0]), **metadata)
    with pytest.raises(ValueError, match=field):
        MeasuredContrastSet((curve,))


@pytest.mark.parametrize("field", ["band_id", "detection_model", "source"])
def test_declared_provenance_must_be_a_nonempty_string(field):
    arrays = (np.array([0.5, 1.5]), np.array([2.0, 3.0]))
    with pytest.raises(ValueError, match=field):
        ContrastCurve(*arrays, **{field: " "})
    with pytest.raises(ValueError, match=field):
        ContrastCurve(*arrays, **{field: 7})


def test_empty_measured_set_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        MeasuredContrastSet(())


def test_apply_rejects_only_companions_brighter_than_the_band_limit():
    curves = MeasuredContrastSet((_curve([0.5, 2.0], [2.5, 2.5]),))
    assert float(curves.at("TESS", 1.0)) == pytest.approx(2.5)

    prior = curves.apply(
        "TESS", np.zeros(2), np.array([_flux_ratio(1.5), _flux_ratio(3.5)]), np.ones(2)
    )
    assert np.isneginf(prior[0])
    assert prior[1] == 0.0

    # A companion exactly at the limit is retained: the rule is delta >= limit.
    at_limit = MeasuredContrastSet((_curve([0.5, 2.0], [0.0, 0.0]),))
    assert at_limit.apply(
        "TESS", np.zeros(1), [_flux_ratio(0.0)], np.ones(1), require_fainter=True
    )[0] == 0.0


def test_require_fainter_keeps_an_equal_brightness_twin_off_curve():
    curves = MeasuredContrastSet((_curve([1.0, 2.0], [2.0, 4.0]),))
    twin = _flux_ratio(0.0)
    brighter_than_host = _flux_ratio(-0.5)

    assert curves.apply("TESS", np.zeros(1), [twin], [0.5], require_fainter=True)[0] == 0.0
    assert np.isneginf(
        curves.apply("TESS", np.zeros(1), [brighter_than_host], [0.5], require_fainter=True)[0]
    )
    # Without the rule the uncovered separation is a complete no-op.
    assert curves.apply("TESS", np.zeros(1), [brighter_than_host], [0.5])[0] == 0.0


def test_apply_refuses_invalid_priors_and_boolean_or_negative_inputs():
    curves = MeasuredContrastSet((_curve([0.5, 2.0], [2.0, 3.0]),))
    valid = dict(lnprior=np.zeros(1), flux_ratio=np.ones(1), separation_arcsec=np.ones(1))

    with pytest.raises(ValueError):
        curves.apply("TESS", **{**valid, "lnprior": np.array([np.nan])})
    with pytest.raises(ValueError):
        curves.apply("TESS", **{**valid, "lnprior": True})
    with pytest.raises(ValueError):
        curves.apply("TESS", **{**valid, "flux_ratio": np.array([True])})
    with pytest.raises(ValueError):
        curves.apply("TESS", **{**valid, "separation_arcsec": -1.0})
    with pytest.raises(ValueError):
        curves.apply("TESS", **valid, require_fainter=1)


def test_legacy_two_argument_curve_matches_the_single_band_path():
    separations = np.array([0.5, 2.0])
    contrasts = np.array([2.5, 4.0])
    legacy_curve = ContrastCurve(separations, contrasts)
    assert (legacy_curve.band_id, legacy_curve.detection_model, legacy_curve.source) == ("", "", "")

    prior = np.array([0.0, -np.inf, 0.0, 0.0])
    flux = np.array([_flux_ratio(0.0), _flux_ratio(0.0), _flux_ratio(0.0), _flux_ratio(2.5)])
    separation = np.array([1.0, 1.0, 2.5, 1.0])

    legacy = imaging_log_prior(flux, separation, separations, contrasts, prior.copy())
    measured = MeasuredContrastSet((_curve(separations, contrasts),)).apply(
        "TESS", prior.copy(), flux, separation
    )
    np.testing.assert_array_equal(measured, legacy)


def test_curve_support_is_not_extrapolated_past_its_outer_edge():
    curve = _curve([0.5, 1.5], [2.0, 3.0])
    curves = MeasuredContrastSet((curve,))

    assert float(curves.at("TESS", 1.5)) == pytest.approx(3.0)
    assert np.isnan(float(curve.at(1.5 + 1e-9)))
    assert np.isnan(float(curves.at("TESS", 1.5 + 1e-9)))
    assert np.isnan(curves.at("TESS", np.array([1.5 + 1e-9, 4.0]))).all()
