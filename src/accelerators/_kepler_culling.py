"""Conservative eclipse-contact windows in mean anomaly for opt-in culling.

The functions answer "could the occultor be in contact with the occulted body
at this mean anomaly?" without evaluating a limb-darkened disc, so a caller
can skip samples far from either conjunction.

The bound is conservative. For argument of latitude theta = omega + f, sky
separation d = r sqrt(cos^2 theta + sin^2 theta cos^2 i) (Winn 2010,
*Exoplanets*, Eq. 5; arXiv:1001.2010) with orbital radius r. Since
d >= r |cos theta| and r >= a (1 - e), contact d <= 1 + k requires

    |cos theta| <= s = min(1, (1 + k) / (a (1 - e))),

so every contact lies inside ``|theta - pi/2| <= arcsin(s)`` or
``|theta - 3 pi/2| <= arcsin(s)``. One condition serves primary and secondary
eclipse: the secondary is the same sky separation measured in companion radii,
``d / k <= 1 + 1/k``. The window is mapped to mean anomaly through the same
Kepler owner the production paths use, taking the larger of the two endpoint
offsets; a wrapped endpoint difference flags a span beyond pi and falls back
to a half turn. A non-detached orbit ``a (1 - e) <= 1 + k`` makes the bound
vacuous, so both windows are the full turn. Invalid inputs fall back to a full
turn, so culling cannot remove a finite sample that the geometry might place
in contact.

Keplerian orbit and mean anomaly: Murray & Correia (2010), *Exoplanets*.
Sky projection and conjunction convention: Winn (2010), Eq. 5.

Time convention: ``_jax_orbits._mean_anomaly_at`` puts ``t0_days`` at inferior
conjunction, ``M = mean_anomaly(pi/2 - omega) + 2 pi (t - t0) / P``. The
windows use that absolute mean anomaly, so the primary is centred on
``M(t0)`` and the secondary at true anomaly advanced by pi, not generally
half an orbital period later. The mean
anomaly is linear in time, so an eccentric orbit's contact windows are wide
or narrow in time exactly as their mean-anomaly width says; ``period_days``
enters through the mean motion. ``max_offset_days`` is the largest absolute
time offset (exposure or phase-bin quadrature node) applied to an observation
and extends each window half-width by ``2 pi max_offset_days / P``, so a
sample is kept when a neighbouring exposure can reduce the observed flux.

All arithmetic is float64 JAX, angles are degrees, times are TDB days, and
radii are in units of the occulted body's radius (``radius_ratio`` is the
occultor radius in those units).
"""

from __future__ import annotations

import math

from trex.accelerators._jax_runtime import jnp

_TWO_PI = 2.0 * math.pi


def _mean_anomaly(true_anomaly, ecc):
    """Mean anomaly from true anomaly, with the same branch convention as the forward orbit."""
    eccentric_anomaly = 2.0 * jnp.atan2(
        jnp.sqrt(jnp.maximum(1.0 - ecc, 0.0)) * jnp.sin(0.5 * true_anomaly),
        jnp.sqrt(jnp.maximum(1.0 + ecc, 0.0)) * jnp.cos(0.5 * true_anomaly),
    )
    return eccentric_anomaly - ecc * jnp.sin(eccentric_anomaly)


def _wrap_to_pi(angle):
    """Signed remainder of ``angle`` in [-pi, pi)."""
    return jnp.mod(angle + math.pi, _TWO_PI) - math.pi


def _conjunction_window(true_anomaly, ecc, delta, period_days, max_offset_days):
    """Conjunction mean anomaly and the half-width containing its contact window.

    ``true_anomaly`` is the conjunction true anomaly, ``delta`` the contact
    half-width in true anomaly. The mean anomaly is monotone in true anomaly,
    so the larger endpoint offset bounds the whole window. A negative wrapped
    offset means the true span crossed pi, where an endpoint no longer bounds
    it; a half turn is then the safe width.
    """
    centre = _mean_anomaly(true_anomaly, ecc)
    forward = _wrap_to_pi(_mean_anomaly(true_anomaly + delta, ecc) - centre)
    backward = _wrap_to_pi(centre - _mean_anomaly(true_anomaly - delta, ecc))
    bounded = (forward >= 0.0) & (backward >= 0.0)
    half = jnp.maximum(forward, backward) + _TWO_PI * jnp.maximum(max_offset_days, 0.0) / period_days
    return centre, jnp.where(bounded, jnp.minimum(half, math.pi), math.pi)


def contact_windows(ecc, argp_deg, a_rs, radius_ratio, period_days, max_offset_days):
    """Mean-anomaly windows that contain every possible contact.

    Args:
        ecc: Eccentricity, in [0, 1).
        argp_deg: Argument of periastron [degrees].
        a_rs: Semimajor axis [occulted body radii].
        radius_ratio: Occultor radius [occulted body radii], positive.
        period_days: Orbital period [days], positive.
        max_offset_days: Largest absolute quadrature/exposure offset [days],
            nonnegative; keeps a sample whose neighbouring exposure can be
            eclipsed even when its central time is not.

    Returns:
        ``(primary_lo, primary_width, secondary_lo, secondary_width)`` in mean
        anomaly rad, using the ``_mean_anomaly_at`` convention (``t0`` at
        inferior conjunction). The window is ``[lo, lo + width]`` modulo one
        turn. Widths are at most a full turn; a full turn means "cannot cull".

    A non-detached orbit (``a (1 - e) <= 1 + k``) returns a full turn for both
    windows: the contact bound carries no information there, so a caller using
    only the primary or only the secondary window must keep every sample. A
    non-finite or non-physical input also returns a full turn rather than a
    narrower or NaN window, because a culling predicate must never discard a
    finite sample on invalid geometry. The separate caller validity flags
    remain responsible for NaN/+inf failures.
    """
    values = tuple(jnp.asarray(value, dtype=jnp.float64) for value in
                   (ecc, argp_deg, a_rs, radius_ratio, period_days, max_offset_days))
    ecc, argp_deg, a_rs, radius_ratio, period_days, max_offset_days = values
    finite = jnp.isfinite(values[0])
    for value in values[1:]:
        finite = finite & jnp.isfinite(value)
    valid = (finite & (ecc >= 0.0) & (ecc < 1.0) & (a_rs > 0.0)
             & (radius_ratio > 0.0) & (period_days > 0.0) & (max_offset_days >= 0.0))

    detached = a_rs * (1.0 - ecc)
    ratio = jnp.where(detached > 0.0, (1.0 + radius_ratio) / detached, 1.0)
    delta = jnp.arcsin(jnp.minimum(jnp.maximum(ratio, 0.0), 1.0))
    # ratio >= 1 is exactly the vacuous-bound case; arcsin(1) would otherwise
    # leave a half-turn shape that a single-window caller could cull against.
    undetached = ratio >= 1.0

    argp = jnp.radians(argp_deg)
    primary_centre, primary_half = _conjunction_window(
        0.5 * math.pi - argp, ecc, delta, period_days, max_offset_days)
    secondary_centre, secondary_half = _conjunction_window(
        1.5 * math.pi - argp, ecc, delta, period_days, max_offset_days)

    full_turn = math.pi
    informative = valid & ~undetached
    primary_half = jnp.where(informative, primary_half, full_turn)
    secondary_half = jnp.where(informative, secondary_half, full_turn)
    primary_lo = jnp.where(valid, primary_centre - primary_half, 0.0)
    secondary_lo = jnp.where(valid, secondary_centre - secondary_half, 0.0)
    return primary_lo, 2.0 * primary_half, secondary_lo, 2.0 * secondary_half


def in_contact_window(mean_anomaly, lo, width):
    """Whether a mean anomaly is inside ``[lo, lo + width]`` modulo one turn.

    A width of a full turn is always inside for finite input. Non-finite input
    is reported outside the window; a culling caller must keep its own
    validity/failure accounting and must not treat this predicate as a
    numerical-failure test.
    """
    return jnp.mod(mean_anomaly - lo, _TWO_PI) <= width
