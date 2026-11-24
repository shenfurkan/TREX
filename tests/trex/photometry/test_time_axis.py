"""Relative time controls use exact fractions or independent decimal arithmetic."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from trex.photometry.time_axis import TimeAxis, rebase_bjd_tdb


def _reference(jd=2457000.0, fraction=0.0, scale="tdb"):
    return Time(jd, fraction, format="jd", scale=scale)


def _axis(**changes):
    inputs = dict(time_days=[0.0, 0.25, 0.5], reference=_reference(),
                  exposure_days=0.125, stamp="mid", correction="barycentric",
                  observer="declared spacecraft", target="synthetic direction", source="synthetic native stamps")
    return TimeAxis(**(inputs | changes))


def test_standard_tess_offsets_are_unchanged_and_not_aliased():
    offsets = np.array([1325.25, 1325.25 + 2**-30, 1326.0])
    actual = rebase_bjd_tdb(offsets, _reference(), _reference())
    np.testing.assert_array_equal(actual, offsets)
    actual[0] = 0
    assert offsets[0] == 1325.25


def test_split_reference_preserves_a_fraction_below_a_full_jd_ulp():
    fraction = 2**-34
    offsets = np.array([0.0, 2**-35, 0.25])
    actual = rebase_bjd_tdb(offsets, _reference(fraction=fraction), _reference())
    np.testing.assert_array_equal(actual, offsets + fraction)


def test_rebasing_matches_decimal_not_a_full_jd_round_trip():
    source, target = _reference(2460000.0, 0.123456789), _reference(2459999.0, 0.987654321)
    offsets = np.array([0.0, 0.0123456789, 0.5])
    with localcontext() as ctx:
        ctx.prec = 50
        shift = Decimal(1) + Decimal(0.123456789) - Decimal(0.987654321)
        expected = np.array([float(shift + Decimal(float(t))) for t in offsets])
    # 1e-15 day = 86.4 ps: below timing relevance, above split-JD roundoff.
    np.testing.assert_allclose(rebase_bjd_tdb(offsets, source, target), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("offsets", [[], [[0., 1.]], [False, True], ["0", "1"], [0., np.nan],
                                         [0., np.inf], [0j, 1j], np.ma.array([0., 1.], mask=[0, 1]),
                                         np.array([0., 1.]) * u.second])
def test_invalid_offsets_are_not_coerced_or_silently_removed(offsets):
    with pytest.raises(ValueError, match="time offsets"):
        rebase_bjd_tdb(offsets, _reference(), _reference())


@pytest.mark.parametrize("reference", [2457000., _reference(scale="utc"), _reference(scale="tcb"),
                                             Time([2457000., 2457001.], format="jd", scale="tdb"),
                                             Time(np.ma.array(2457000., mask=True), format="jd", scale="tdb")])
def test_references_must_explicitly_be_scalar_tdb(reference):
    with pytest.raises(ValueError, match="finite, unmasked TDB Time"):
        rebase_bjd_tdb([0., 1.], reference, _reference())
    with pytest.raises(ValueError, match="finite, unmasked TDB Time"):
        rebase_bjd_tdb([0., 1.], _reference(), reference)


@pytest.mark.parametrize(("stamp", "offset"), [("start", -0.0625), ("mid", 0.), ("end", 0.0625)])
def test_equivalent_boxcar_anchors_produce_identical_midpoints(stamp, offset):
    centers = np.array([0., 0.25, 0.5])
    axis = _axis(time_days=centers + offset, stamp=stamp)
    np.testing.assert_array_equal(axis.relative_midpoints(_reference()), centers)
    np.testing.assert_array_equal(axis.time_days, centers + offset)


def test_variable_boxcars_and_simultaneous_distinct_streams_are_legal():
    axis = _axis(time_days=[0., 0.25, 0.5], exposure_days=[0., 0.125, 0.25], stamp="start")
    np.testing.assert_array_equal(axis.relative_midpoints(_reference()), [0., 0.3125, 0.625])
    np.testing.assert_array_equal(_axis().relative_midpoints(_reference()),
                                  _axis(observer="another spacecraft").relative_midpoints(_reference()))


@pytest.mark.parametrize("changes", [
    {"time_days": [0., 0., 1.]}, {"time_days": [1., 0.]}, {"time_days": []},
    {"exposure_days": -0.1}, {"exposure_days": [0.1, 0.2]}, {"exposure_days": [[0.1]]},
    {"exposure_days": np.nan}, {"exposure_days": True},
    {"stamp": "unknown"}, {"correction": "heliocentric"}, {"correction": "none"},
    {"observer": ""}, {"target": None}, {"source": " "},
    {"time_days": [0., 0.25], "exposure_days": [1., 0.], "stamp": "start"},
])
def test_ambiguous_conventions_and_invalid_chronology_refuse(changes):
    with pytest.raises(ValueError):
        _axis(**changes)


def test_axis_copies_arrays_and_reference_before_normalizing():
    offsets, exposure, reference = np.array([0., 0.25]), np.array([0.125, 0.25]), _reference()
    axis = _axis(time_days=offsets, exposure_days=exposure, reference=reference, stamp="start")
    offsets[0], exposure[0] = 9., 9.
    reference[()] = _reference(2458000.)
    np.testing.assert_array_equal(axis.relative_midpoints(_reference()), [0.0625, 0.375])
    assert not axis.time_days.flags.writeable and not axis.exposure_days.flags.writeable
    assert not axis.reference.writeable


def test_midpoints_feed_the_actual_native_gaussian_baseline():
    from trex.photometry.baseline import GaussianBaseline

    time = _axis(stamp="start").relative_midpoints(_reference())
    prior = GaussianBaseline(time, np.ones((3, 1)), [0.], [[0.01]], "additive", "synthetic independent prior")
    density = prior.prepare(time, [1., 1., 1.], 0.1)
    # C=0.01 I + 0.01 11.T: eigenvalues 0.04, 0.01, 0.01.
    expected = -0.5 * (3 * np.log(2 * np.pi) + np.log(0.04) + 2 * np.log(0.01))
    assert density([1., 1., 1.]) == pytest.approx(expected, rel=0, abs=2e-14)
