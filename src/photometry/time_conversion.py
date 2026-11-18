"""Explicit clock and observer conversion before relative-TDB reference arithmetic."""

from dataclasses import dataclass

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, get_body_barycentric
from astropy.time import Time, TimeDelta


def barycentric_tdb(times: Time, *, correction: str, target: SkyCoord | None = None) -> Time:
    """Return precise TDB barycentric stamps with the input shape preserved.

    ``correction='barycentric'`` admits already-corrected TDB or TCB and only
    converts the scale. It forbids a new target direction to make accidental
    double correction visible. ``correction='none'`` admits observer arrival
    stamps in UTC/TAI/TT/TDB/TCB/TCG, with a finite scalar EarthLocation on
    ``times`` and a scalar, finite ICRS direction in ``target``. Adapters must
    propagate astrometry to the observation epoch before passing that direction;
    unpropagated coordinate differentials are refused.

    ``correction='heliocentric'`` is the qualified HJD inversion: the input is
    the arrival time at the heliocentre (HJD) on an explicit standard clock
    scale, the target direction is required, and the observer is the
    heliocentre (an attached EarthLocation is contradictory and refused). The
    barycentric time is ``HJD_TDB + (r_sun . n)/c`` with the Sun's barycentric
    position from the same builtin ERFA ephemeris. No HJD_UTC/HJD/local
    convention is inferred when the scale or target is undeclared. The Sun is
    evaluated at the HJD stamp, not at the unknown arrival time
    ``HJD - (r_obs,sun . n)/c`` (the observer is not declared), so the shift
    carries an error bounded by ``|v_sun| |HJD - t_arrival| / c``: about
    13 m/s x 500 s / c, i.e. at most ~25 us for an Earth-like heliocentric
    distance. That is far below transit-timing precision, but it is not exact.

    Uses Astropy 6.1.7 Time.light_travel_time's distant-source approximation,
    builtin ERFA ephemeris, and ``times.tdb + light_travel_time``. The inspected
    software contract is https://docs.astropy.org/en/v6.1.7/api/astropy.time.Time.html
    (light_travel_time); finite-distance light propagation and Shapiro delay
    are not modeled here. IERS/leap-second availability and warnings remain
    visible under the caller's Astropy configuration; no service is queried
    for a target name, site name or ephemeris kernel.

    Non-Time, empty/masked/nonfinite inputs, local/UT1 clock labels, unknown
    correction states and ambiguous HJD/BJD_UTC conventions raise ValueError.
    No spacecraft frame/origin is guessed from an EarthLocation; mission
    stamps use ``spacecraft_barycentric_tdb`` with a declared state table.
    This function converts individual stamps (including quadrature nodes),
    NOT an exposure duration or a start/mid/end anchor.
    """
    if (not isinstance(times, Time) or not times.size or np.any(times.mask)
            or not np.isfinite(times.jd1).all() or not np.isfinite(times.jd2).all()):
        raise ValueError("arrival stamps must be finite, nonempty, unmasked Time values")
    if correction == "barycentric":
        if times.scale not in ("tdb", "tcb"):
            raise ValueError("already-barycentric stamps require explicit TDB or TCB, not BJD_UTC")
        if target is not None:
            raise ValueError("already-barycentric stamps cannot receive another target correction")
        return times.tdb.copy()
    if correction == "heliocentric":
        if times.scale not in ("utc", "tai", "tt", "tdb", "tcb", "tcg"):
            raise ValueError("heliocentric correction requires an explicit standard clock scale")
        if times.location is not None:
            raise ValueError(
                "heliocentric correction has a heliocentric observer, not an EarthLocation"
            )
        if target is None:
            raise ValueError("heliocentric correction requires a declared target direction")
        if (not isinstance(target, SkyCoord) or not target.isscalar or target.frame.name != "icrs"
                or bool(np.any(getattr(target, "mask", False)))
                or not np.isfinite([target.ra.to_value(u.rad), target.dec.to_value(u.rad)]).all()
                or target.data.differentials):
            raise ValueError("target must be a finite scalar ICRS direction with astrometry already propagated")
        direction = target.cartesian.xyz.value
        direction = direction / np.linalg.norm(direction)
        sun = get_body_barycentric("sun", times.tdb, ephemeris="builtin")
        # HJD is arrival at the Sun; BJD adds the Sun-to-barycentre projection.
        direction = direction.reshape((3,) + (1,) * (sun.xyz.ndim - 1))
        shift = (np.sum(sun.xyz * direction, axis=0) / const.c).to_value(u.s)
        result = times.tdb + TimeDelta(shift, format="sec")
        if not np.isfinite(result.jd1).all() or not np.isfinite(result.jd2).all():
            raise ValueError("barycentric TDB stamps must remain finite")
        return result
    if correction != "none":
        raise ValueError(
            "correction must be none, barycentric or a qualified heliocentric HJD; "
            "ambiguous HJD/BJD_UTC input is unsupported"
        )
    if times.scale not in ("utc", "tai", "tt", "tdb", "tcb", "tcg"):
        raise ValueError("observer clock must explicitly be UTC, TAI, TT, TDB, TCB or TCG")
    location = times.location
    if (not isinstance(location, EarthLocation) or not location.isscalar
            or not np.isfinite([v.to_value(u.m) for v in location.geocentric]).all()):
        raise ValueError("observer arrival stamps require a finite scalar EarthLocation")
    if (not isinstance(target, SkyCoord) or not target.isscalar or target.frame.name != "icrs"
            or bool(np.any(getattr(target, "mask", False)))
            or not np.isfinite([target.ra.to_value(u.rad), target.dec.to_value(u.rad)]).all()
            or target.data.differentials):
        raise ValueError("target must be a finite scalar ICRS direction with astrometry already propagated")
    result = times.tdb + times.light_travel_time(target, kind="barycentric", ephemeris="builtin")
    if not np.isfinite(result.jd1).all() or not np.isfinite(result.jd2).all():
        raise ValueError("barycentric TDB stamps must remain finite")
    return result


_STATE_ORIGINS = ("barycentre", "geocentre")


@dataclass(frozen=True, eq=False)
class SpacecraftState:
    """Declared spacecraft state-vector table for observer light-travel time.

    ``times`` are the state epochs (any Astropy scale; compared in TDB), strictly
    increasing, N>=2. ``position_km``/``velocity_km_s`` are (N, 3) ICRS-axis
    vectors relative to ``origin``: ``"barycentre"`` (solar-system barycentre,
    BCRS axes) or ``"geocentre"`` (GCRS axes; the Earth's barycentric position
    from Astropy's builtin ERFA ephemeris is added). ``source`` names the
    ephemeris product/edition (e.g. an OEM/SPICE file and its release). No
    origin, frame or orbit is presumed; mission adapters must supply this table.
    """

    times: Time
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    origin: str
    source: str

    def __post_init__(self):
        times = self.times
        if (not isinstance(times, Time) or times.isscalar or times.size < 2 or np.any(times.mask)
                or not np.isfinite(times.jd1).all() or not np.isfinite(times.jd2).all()):
            raise ValueError("spacecraft state needs at least two finite, unmasked epochs")
        tdb = times.tdb
        seconds = (tdb - tdb[0]).to_value(u.s)
        if np.any(np.diff(seconds) <= 0):
            raise ValueError("spacecraft state epochs must be strictly increasing")
        arrays = []
        for name in ("position_km", "velocity_km_s"):
            value = getattr(self, name)
            if isinstance(value, u.Quantity) or np.ma.isMaskedArray(value):
                raise ValueError(f"{name} must be plain real numbers in its stated unit")
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != (times.size, 3) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape (N, 3)")
            array.flags.writeable = False
            arrays.append(array)
        if self.origin not in _STATE_ORIGINS:
            raise ValueError("spacecraft state origin must be declared as barycentre or geocentre")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("spacecraft state requires its ephemeris source")
        epochs = tdb.copy()
        epochs.writeable = False
        object.__setattr__(self, "times", epochs)
        object.__setattr__(self, "position_km", arrays[0])
        object.__setattr__(self, "velocity_km_s", arrays[1])


@dataclass(frozen=True, eq=False)
class SpacecraftConversion:
    """Barycentric TDB stamps plus the geometric term and its numerical bound.

    ``roemer_delay_s`` is n.r/c per stamp [s]. ``interpolation_error_s`` is the
    cubic-Hermite bound h^4/384 sum_k max|r_k''''|/c with r'''' estimated by
    finite differences of the tabulated velocities (an estimate, not a proof); None when fewer than five state
    epochs make that estimate unavailable (then ``limits`` says so).
    """

    times: Time
    roemer_delay_s: np.ndarray
    interpolation_error_s: float | None
    limits: tuple

    def time_axis(self, *, reference: Time, exposure_days, stamp: str, observer: str, target: str, source: str):
        """Relative barycentric TimeAxis at a split-JD TDB ``reference``.

        Offsets are formed from the two-part Time difference, so full-JD floats
        are never built. ``exposure_days``/``stamp`` describe the native boxcar
        anchor exactly as for any other TimeAxis; this does not infer a ramp.
        """
        from .time_axis import TimeAxis, _tdb_reference

        reference = _tdb_reference(reference)
        offsets = (self.times.ravel() - reference).to_value(u.day)
        return TimeAxis(time_days=np.asarray(offsets, dtype=np.float64), reference=reference,
                        exposure_days=exposure_days, stamp=stamp, correction="barycentric",
                        observer=observer, target=target, source=source)


def _hermite(seconds, nodes, position, velocity):
    index = np.clip(np.searchsorted(nodes, seconds, side="right") - 1, 0, nodes.size - 2)
    h = (nodes[index + 1] - nodes[index])[:, None]
    s = ((seconds - nodes[index])[:, None]) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return (h00 * position[index] + h10 * h * velocity[index]
            + h01 * position[index + 1] + h11 * h * velocity[index + 1])


def _fourth_derivative_bound(nodes, velocity):
    if nodes.size < 5:
        return None
    # Third finite difference of velocity estimates r'''' on the local spacing.
    derivative, grid = velocity, nodes
    for _ in range(3):
        derivative = np.diff(derivative, axis=0) / np.diff(grid)[:, None]
        grid = 0.5 * (grid[1:] + grid[:-1])
    # Hermite's bound holds per component; |n.e| <= sum_k |e_k| for a unit n.
    return float(np.sum(np.max(np.abs(derivative), axis=0)))


def spacecraft_barycentric_tdb(times: Time, *, state: SpacecraftState, target: SkyCoord) -> SpacecraftConversion:
    """Convert spacecraft arrival stamps to barycentric TDB via n.r/c.

    ``times`` are arrival stamps already on a standard Astropy scale (the
    mission product owns on-board clock correlation); they are converted to TDB
    with Astropy's geocentric TT->TDB relation. The neglected location term at
    the spacecraft is of order v_earth.r/c^2 (about 0.5 us at 1.5e6 km), stated
    in ``limits``. ``target`` is a scalar ICRS direction with astrometry already
    propagated; the distant-source approximation is used (no finite-distance
    parallax or Shapiro delay), matching ``barycentric_tdb`` for ground sites.
    Stamps outside the state-table coverage are refused, never extrapolated.
    Cubic Hermite interpolation of the declared positions/velocities supplies r.
    """
    if (not isinstance(times, Time) or not times.size or np.any(times.mask)
            or not np.isfinite(times.jd1).all() or not np.isfinite(times.jd2).all()):
        raise ValueError("arrival stamps must be finite, nonempty, unmasked Time values")
    if times.scale not in ("utc", "tai", "tt", "tdb", "tcg"):
        raise ValueError("spacecraft arrival stamps must be UTC, TAI, TT, TDB or TCG, not already barycentric")
    if times.location is not None:
        raise ValueError("spacecraft stamps must not carry a ground EarthLocation")
    if not isinstance(state, SpacecraftState):
        raise ValueError("spacecraft conversion requires a declared SpacecraftState table")
    if (not isinstance(target, SkyCoord) or not target.isscalar or target.frame.name != "icrs"
            or not np.isfinite([target.ra.to_value(u.rad), target.dec.to_value(u.rad)]).all()
            or target.data.differentials):
        raise ValueError("target must be a finite scalar ICRS direction with astrometry already propagated")
    tdb = times.tdb
    flat = tdb.ravel()
    nodes = (state.times - state.times[0]).to_value(u.s)
    seconds = (flat - state.times[0]).to_value(u.s)
    if np.any(seconds < nodes[0]) or np.any(seconds > nodes[-1]):
        raise ValueError("arrival stamps fall outside the spacecraft state-table coverage")
    position = _hermite(seconds, nodes, state.position_km, state.velocity_km_s)
    if state.origin == "geocentre":
        earth = get_body_barycentric("earth", flat, ephemeris="builtin")
        position = position + earth.xyz.to_value(u.km).T
    direction = target.cartesian.xyz.value
    direction = direction / np.linalg.norm(direction)
    speed = const.c.to_value(u.km / u.s)
    delay = (position @ direction) / speed
    fourth = _fourth_derivative_bound(nodes, state.velocity_km_s)
    limits = [
        "geocentric TT->TDB; spacecraft-location clock term (~v_earth.r/c^2) not applied",
        "distant-source direction; no finite-distance parallax or Shapiro delay",
    ]
    if fourth is None:
        bound = None
        limits.append("fewer than five state epochs: interpolation error is not estimated")
    else:
        bound = float(np.max(np.diff(nodes)) ** 4 / 384.0 * fourth / speed)
    result = flat + TimeDelta(delay, format="sec")
    if not np.isfinite(result.jd1).all() or not np.isfinite(result.jd2).all():
        raise ValueError("barycentric TDB stamps must remain finite")
    return SpacecraftConversion(result.reshape(times.shape), delay.reshape(times.shape), bound, tuple(limits))
