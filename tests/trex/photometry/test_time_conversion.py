"""Clock-label controls and independent geometric limits at the real boundary."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_body_barycentric, get_body_barycentric_posvel
from astropy.time import Time
from astropy.utils import iers

from trex.photometry.time_axis import TimeAxis
from trex.photometry.time_conversion import barycentric_tdb


@pytest.fixture(autouse=True)
def offline_earth_orientation():
    with iers.conf.set_temp("auto_download", False):
        yield


def _arrival(scale="utc"):
    return Time([2457754., 2457754.25], [.5, .5], format="jd", scale=scale,
                location=EarthLocation.from_geocentric(0, 0, 0, unit=u.m))


def test_barycentric_tdb_is_unchanged_and_tcb_only_changes_clock_scale():
    stamps = Time(2457754., [.5, .5 + 2**-34], format="jd", scale="tdb")
    same = barycentric_tdb(stamps, correction="barycentric")
    np.testing.assert_array_equal(same.jd1, stamps.jd1)
    np.testing.assert_array_equal(same.jd2, stamps.jd2)
    converted = barycentric_tdb(stamps.tcb, correction="barycentric")
    np.testing.assert_allclose((converted - stamps).to_value(u.s), 0., rtol=0, atol=1e-10)
    same[0] = stamps[1]
    assert stamps[0].jd2 != stamps[1].jd2


def test_utc_and_explicit_tt_clock_labels_describe_the_same_arrival():
    # At 2017-01-01 TAI-UTC is 37 s and TT-TAI is 32.184 s.
    utc = _arrival()
    tt = Time(["2017-01-01T00:01:09.184", "2017-01-01T06:01:09.184"],
              format="isot", scale="tt", location=utc.location)
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    actual = barycentric_tdb(utc, correction="none", target=target)
    expected = barycentric_tdb(tt, correction="none", target=target)
    np.testing.assert_allclose((actual - expected).to_value(u.s), 0., rtol=0, atol=1e-8)
    # Scale conversion alone omits a direction-dependent minutes-sized delay.
    assert np.all(np.abs((actual - utc.tdb).to_value(u.s)) > 1.)


def test_geocentric_delay_matches_independent_position_dot_direction_and_sign():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    opposite = SkyCoord(ra=210 * u.deg, dec=20 * u.deg, frame="icrs")
    actual = barycentric_tdb(arrival, correction="none", target=target)
    reverse = barycentric_tdb(arrival, correction="none", target=opposite)
    earth = get_body_barycentric("earth", arrival.tdb, ephemeris="builtin").xyz
    direction = target.cartesian.xyz
    expected_seconds = (np.sum(earth * direction[:, None], axis=0) / const.c).to_value(u.s)
    actual_seconds = (actual - arrival.tdb).to_value(u.s)
    np.testing.assert_allclose(actual_seconds, expected_seconds, rtol=0, atol=1e-8)
    np.testing.assert_allclose((reverse - arrival.tdb).to_value(u.s), -actual_seconds,
                               rtol=0, atol=1e-8)


def test_geometric_conversion_preserves_split_nodes_and_feeds_time_axis():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    nodes = arrival[:, None] + np.array([0., 2**-30]) * u.day
    converted = barycentric_tdb(nodes, correction="none", target=target)
    assert converted.shape == (2, 2)
    # A ~80-us separation remains resolved, unlike a full-JD float round trip.
    np.testing.assert_allclose((converted[:, 1] - converted[:, 0]).to_value(u.day),
                               2**-30, rtol=2e-4, atol=0)
    reference = Time(2457754., format="jd", scale="tdb")
    axis = TimeAxis((converted[:, 0] - reference).to_value(u.day), reference,
                    0., "mid", "barycentric", "geocentric control",
                    "synthetic ICRS direction", "declared UTC nodes converted once")
    np.testing.assert_array_equal(axis.relative_midpoints(reference),
                                   (converted[:, 0] - reference).to_value(u.day))


def test_surface_observer_is_not_silently_replaced_with_geocentre():
    arrival = _arrival()
    surface = Time(arrival.jd1, arrival.jd2, format="jd", scale=arrival.scale,
                   location=EarthLocation.from_geodetic(20 * u.deg, 40 * u.deg, 0 * u.m))
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    difference = (barycentric_tdb(surface, correction="none", target=target)
                  - barycentric_tdb(arrival, correction="none", target=target)).to_value(u.s)
    assert np.any(np.abs(difference) > 1e-4)
    assert np.all(np.abs(difference) < .03)


@pytest.mark.parametrize("correction", ["heliocentric", "", None, "unknown"])
def test_ambiguous_correction_cannot_be_guessed(correction):
    with pytest.raises(ValueError, match="correction"):
        barycentric_tdb(_arrival(), correction=correction)


def test_missing_observer_direction_and_clock_conventions_refuse():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    with pytest.raises(ValueError, match="EarthLocation"):
        barycentric_tdb(Time(arrival.jd1, arrival.jd2, format="jd", scale="utc"),
                         correction="none", target=target)
    for invalid in (None, SkyCoord([30, 31], [-20, -20], unit=u.deg), SkyCoord(l=0*u.deg, b=0*u.deg, frame="galactic")):
        with pytest.raises(ValueError, match="ICRS"):
            barycentric_tdb(arrival, correction="none", target=invalid)
    with pytest.raises(ValueError, match="already propagated"):
        barycentric_tdb(arrival, correction="none",
                         target=SkyCoord(ra=30*u.deg, dec=-20*u.deg, pm_ra_cosdec=1*u.mas/u.yr,
                                         pm_dec=0*u.mas/u.yr, obstime=Time("2000-01-01")))
    with pytest.raises(ValueError, match="TDB or TCB"):
        barycentric_tdb(arrival, correction="barycentric")
    with pytest.raises(ValueError, match="another target correction"):
        barycentric_tdb(arrival.tdb, correction="barycentric", target=target)
    with pytest.raises(ValueError, match="observer clock"):
        barycentric_tdb(Time("2017-01-01", scale="local"), correction="none", target=target)
    masked = arrival.copy()
    masked[0] = np.ma.masked
    for invalid in (masked, [2457754.], Time([], format="jd", scale="tdb")):
        with pytest.raises(ValueError, match="unmasked Time"):
            barycentric_tdb(invalid, correction="barycentric")


def _hjd(arrival, target):
    """Heliocentric TDB label without the ground location that produced it."""
    shifted = arrival.tdb + arrival.light_travel_time(
        target, kind="heliocentric", ephemeris="builtin"
    )
    return Time(shifted.jd1, shifted.jd2, format="jd", scale="tdb")


def test_qualified_hjd_inversion_matches_astropy_heliocentric_difference():
    """BJD = HJD + (r_sun.n)/c, checked through Astropy's own light-travel times."""
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    barycentric = barycentric_tdb(arrival, correction="none", target=target)
    heliocentric = _hjd(arrival, target)
    actual = barycentric_tdb(heliocentric, correction="heliocentric", target=target)
    assert actual.shape == heliocentric.shape
    scalar = Time(heliocentric.jd1[0], heliocentric.jd2[0], format="jd", scale="tdb")
    assert barycentric_tdb(scalar, correction="heliocentric", target=target).isscalar
    # The inversion evaluates the Sun at the HJD stamp instead of the (undeclared)
    # arrival time; its first-order error is |v_sun| |HJD - arrival| / c.
    _, velocity = get_body_barycentric_posvel("sun", arrival.tdb, ephemeris="builtin")
    speed = velocity.norm().to_value(u.m / u.s)
    lag = np.abs((heliocentric - arrival.tdb).to_value(u.s))
    bound = speed * lag / const.c.to_value(u.m / u.s)
    error = np.abs((actual - barycentric).to_value(u.s))
    assert np.all(error <= 1.01 * bound + 1e-9)
    assert np.all(bound < 3e-5)


def test_hjd_shift_is_the_sun_barycentric_projection_with_its_sign():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    heliocentric = _hjd(arrival, target)
    actual = barycentric_tdb(heliocentric, correction="heliocentric", target=target)
    sun = get_body_barycentric("sun", heliocentric.tdb, ephemeris="builtin").xyz
    direction = target.cartesian.xyz
    expected = (np.sum(sun * direction[:, None], axis=0) / const.c).to_value(u.s)
    np.testing.assert_allclose((actual - heliocentric.tdb).to_value(u.s), expected,
                               rtol=0, atol=1e-9)
    opposite = SkyCoord(ra=210 * u.deg, dec=20 * u.deg, frame="icrs")
    reverse = barycentric_tdb(heliocentric, correction="heliocentric", target=opposite)
    np.testing.assert_allclose((reverse - heliocentric.tdb).to_value(u.s), -expected,
                               rtol=0, atol=1e-9)


def test_hjd_convention_without_scale_observer_or_target_refuses():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    with pytest.raises(ValueError, match="correction"):
        barycentric_tdb(arrival, correction="heliocentric")
    with pytest.raises(ValueError, match="heliocentric observer"):
        barycentric_tdb(arrival, correction="heliocentric", target=target)
    with pytest.raises(ValueError, match="clock scale"):
        barycentric_tdb(Time("2017-01-01", scale="local"), correction="heliocentric", target=target)
    for invalid in (SkyCoord(l=0*u.deg, b=0*u.deg, frame="galactic"),
                    SkyCoord(ra=30*u.deg, dec=-20*u.deg, pm_ra_cosdec=1*u.mas/u.yr,
                             pm_dec=0*u.mas/u.yr, obstime=Time("2000-01-01"))):
        with pytest.raises(ValueError, match="ICRS|already propagated"):
            barycentric_tdb(Time(arrival.jd1, arrival.jd2, format="jd", scale="tdb"),
                            correction="heliocentric", target=invalid)


def test_qualified_hjd_inversion_feeds_time_axis():
    arrival = _arrival()
    target = SkyCoord(ra=30 * u.deg, dec=-20 * u.deg, frame="icrs")
    converted = barycentric_tdb(_hjd(arrival, target), correction="heliocentric", target=target)
    reference = Time(2457754., format="jd", scale="tdb")
    axis = TimeAxis((converted - reference).to_value(u.day), reference,
                    0., "mid", "barycentric", "heliocentric control",
                    "synthetic ICRS direction", "qualified HJD inversion")
    np.testing.assert_array_equal(axis.relative_midpoints(reference),
                                  (converted - reference).to_value(u.day))



# --- spacecraft state-vector boundary (MM-T1) -------------------------------

from astropy.time import TimeDelta  # noqa: E402

from trex.photometry.time_conversion import (  # noqa: E402
    SpacecraftState,
    spacecraft_barycentric_tdb,
)

_TARGET = SkyCoord(ra=30.0 * u.deg, dec=-20.0 * u.deg, frame="icrs")


def _unit(target):
    vector = target.cartesian.xyz.value
    return vector / np.linalg.norm(vector)


def _linear_state(origin="barycentre", n=9, velocity=(12.0, -7.0, 3.0), start=(1.5e8, 2.0e7, -4.0e6)):
    epochs = Time(2461000.0, np.linspace(0.0, 2.0, n), format="jd", scale="tdb")
    seconds = (epochs - epochs[0]).to_value(u.s)
    velocity = np.asarray(velocity, dtype=float)
    position = np.asarray(start, dtype=float) + seconds[:, None] * velocity
    return SpacecraftState(epochs, position, np.tile(velocity, (n, 1)), origin, "synthetic straight line")


def test_straight_line_state_gives_the_exact_dot_product_delay():
    state = _linear_state()
    stamps = Time(2461000.0, [0.1, 0.77, 1.9], format="jd", scale="tdb")
    converted = spacecraft_barycentric_tdb(stamps, state=state, target=_TARGET)

    seconds = (stamps - state.times[0]).to_value(u.s)
    position = np.asarray((1.5e8, 2.0e7, -4.0e6)) + seconds[:, None] * np.asarray((12.0, -7.0, 3.0))
    expected = position @ _unit(_TARGET) / const.c.to_value(u.km / u.s)
    # Cubic Hermite reproduces a straight line exactly up to float64 roundoff (~1e-16 relative).
    np.testing.assert_allclose(converted.roemer_delay_s, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose((converted.times - stamps).to_value(u.s), expected, rtol=0, atol=1e-9)
    # A straight line has zero fourth derivative, so the estimated bound is zero.
    assert converted.interpolation_error_s == pytest.approx(0.0, abs=1e-15)


def test_reversed_position_reverses_the_delay_sign():
    forward = _linear_state()
    backward = SpacecraftState(forward.times, -forward.position_km, -forward.velocity_km_s,
                               "barycentre", "mirrored")
    stamps = Time(2461000.0, [0.5], format="jd", scale="tdb")
    a = spacecraft_barycentric_tdb(stamps, state=forward, target=_TARGET).roemer_delay_s
    b = spacecraft_barycentric_tdb(stamps, state=backward, target=_TARGET).roemer_delay_s
    np.testing.assert_allclose(a, -b, rtol=1e-14, atol=0)


def test_geocentric_zero_offset_matches_astropy_geocentre_light_travel_time():
    epochs = Time(2457754.0, np.linspace(0.0, 1.0, 6), format="jd", scale="tdb")
    zero = np.zeros((6, 3))
    state = SpacecraftState(epochs, zero, zero, "geocentre", "geocentre itself")
    stamps = Time(2457754.0, [0.2, 0.6], format="jd", scale="utc")
    converted = spacecraft_barycentric_tdb(stamps, state=state, target=_TARGET)

    at_geocentre = Time(stamps.jd1, stamps.jd2, format="jd", scale="utc",
                        location=EarthLocation.from_geocentric(0, 0, 0, unit=u.m))
    reference = barycentric_tdb(at_geocentre, correction="none", target=_TARGET)
    # Same ERFA builtin Earth ephemeris and distant-source formula; 1 ns is far below transit timing.
    np.testing.assert_allclose((converted.times - reference).to_value(u.s), 0.0, rtol=0, atol=1e-9)


def test_curved_state_reports_a_bound_that_covers_the_interpolation_error():
    # Circular orbit sampled coarsely: the Hermite error is real and must be bounded.
    epochs = Time(2461000.0, np.linspace(0.0, 4.0, 17), format="jd", scale="tdb")
    seconds = (epochs - epochs[0]).to_value(u.s)
    radius, omega = 1.5e6, 2 * np.pi / (30 * 86400.0)
    position = radius * np.stack([np.cos(omega * seconds), np.sin(omega * seconds), 0 * seconds], axis=1)
    velocity = radius * omega * np.stack([-np.sin(omega * seconds), np.cos(omega * seconds), 0 * seconds], axis=1)
    state = SpacecraftState(epochs, position, velocity, "barycentre", "synthetic circle")
    stamps = Time(2461000.0, np.linspace(0.05, 3.95, 41), format="jd", scale="tdb")
    converted = spacecraft_barycentric_tdb(stamps, state=state, target=_TARGET)

    t = (stamps - epochs[0]).to_value(u.s)
    exact = radius * np.stack([np.cos(omega * t), np.sin(omega * t), 0 * t], axis=1) @ _unit(_TARGET)
    error = np.max(np.abs(converted.roemer_delay_s - exact / const.c.to_value(u.km / u.s)))
    assert converted.interpolation_error_s is not None
    assert error <= converted.interpolation_error_s * 1.01 + 1e-12


def test_stamps_outside_state_coverage_are_refused_not_extrapolated():
    state = _linear_state()
    late = state.times[-1] + TimeDelta(1.0, format="sec")
    with pytest.raises(ValueError, match="outside the spacecraft state-table coverage"):
        spacecraft_barycentric_tdb(Time([late.jd1], [late.jd2], format="jd", scale="tdb"),
                                   state=state, target=_TARGET)


def test_undeclared_origin_ground_location_and_barycentric_input_are_refused():
    state = _linear_state()
    with pytest.raises(ValueError, match="origin must be declared"):
        SpacecraftState(state.times, state.position_km, state.velocity_km_s, "L2", "x")
    with pytest.raises(ValueError, match="ephemeris source"):
        SpacecraftState(state.times, state.position_km, state.velocity_km_s, "barycentre", "")
    with pytest.raises(ValueError, match="ground EarthLocation"):
        spacecraft_barycentric_tdb(_arrival(), state=state, target=_TARGET)
    tcb = Time(2461000.0, [0.5], format="jd", scale="tcb")
    with pytest.raises(ValueError, match="not already barycentric"):
        spacecraft_barycentric_tdb(tcb, state=state, target=_TARGET)


def test_few_state_epochs_state_the_missing_error_estimate():
    state = _linear_state(n=3)
    converted = spacecraft_barycentric_tdb(Time(2461000.0, [1.0], format="jd", scale="tdb"),
                                           state=state, target=_TARGET)
    assert converted.interpolation_error_s is None
    assert any("not estimated" in limit for limit in converted.limits)


def test_spacecraft_conversion_feeds_a_barycentric_time_axis():
    state = _linear_state()
    stamps = Time(2461000.0, [0.25, 0.5, 0.75], format="jd", scale="tdb")
    converted = spacecraft_barycentric_tdb(stamps, state=state, target=_TARGET)
    reference = Time(2461000.0, 0.5, format="jd", scale="tdb")
    axis = converted.time_axis(reference=reference, exposure_days=60.0 / 86400.0, stamp="mid",
                               observer="synthetic spacecraft", target="synthetic", source="test")
    expected = (stamps - reference).to_value(u.day) + converted.roemer_delay_s / 86400.0
    np.testing.assert_allclose(axis.time_days, expected, rtol=0, atol=1e-14)
    assert axis.correction == "barycentric"
