"""Float64 CPU occultations by an opaque circular disc."""

import math

import numpy as np

from trex.models.orbital_mechanics import kepler_sky_geometry


def _cel(kc, p, a, b):
    """Evaluate cel(kc, p, a, b) for 0 < kc <= 1 and p >= 1.

    Integral definition: Agol, Luger & Foreman-Mackey (2020), AJ 159, 123,
    DOI 10.3847/1538-3881/ab4fee, TeX label `eq:cel`. This duplication is
    derived by writing x = tan(phi), pairing x with 1/(kc*x), then changing
    variables to y = (kc*x - 1/x)/(1+kc) and inverting positive y. The paired
    rational weight has the same form at kc' = 2*sqrt(kc)/(1+kc).
    At kc = 1 the remaining tangent integral is elementary. The complement
    approaches one quadratically, and an unresolved result is not returned.
    """
    for _ in range(18):
        if np.all(1 - kc <= np.finfo(float).eps):
            root = np.sqrt(p)
            return (math.pi / 2) * (a + b / root) / (1 + root)
        scale = 1 + kc
        ratio = kc / p
        a, b = (a + b / p) / scale, 2 * (1 + ratio) * (a * kc + b) / scale**3
        p = p * ((1 + ratio) / scale)**2
        kc = 2 * np.sqrt(kc) / scale
    raise FloatingPointError("complete elliptic integral did not converge in float64")


def _elliptic_moments(z, p, limb):
    """Table 1 lambda_1 / lambda_2, away from their singular contact rows."""
    difference = z - p
    a = difference**2
    span = 4 * z * p
    b = (z + p)**2
    b_minus_one = (z - (1 - p)) * (z + p + 1)
    one_minus_a = ((p + 1) - z) * (z - (p - 1))
    q = -difference * (p + z)
    kc2 = np.empty_like(z)
    kc2[limb] = b_minus_one[limb] / span[limb]
    kc2[~limb] = -b_minus_one[~limb] / one_minus_a[~limb]
    kc = np.sqrt(np.clip(kc2, 0, 1))
    moment = np.empty_like(z)
    if np.any(limb):
        lkc, gap, h = kc[limb], difference[limb], one_minus_a[limb]
        k = _cel(lkc, 1.0, 1.0, 1.0)
        j = _cel(lkc, 1.0, 1.0, 0.0)
        pi_cos = _cel(lkc, 1 / a[limb], 1.0, 0.0)
        # Combine the divergent K and Pi terms and cancel the O(p^4)
        # terms algebraically (Agol et al. 2020, TeX `eq:cel_identities`).
        moment[limb] = h * (
            b_minus_one[limb] * (2 * j - k)
            + (3 * q[limb] - 2 * h) * j + 3 * (z[limb] + p) / gap * pi_cos
        ) / (9 * math.pi * np.sqrt(z[limb] * p))
    body = ~limb
    if np.any(body):
        bk, h, bz, bq = kc[body], one_minus_a[body], z[body], q[body]
        k = _cel(bk, 1.0, 1.0, 1.0)
        e = _cel(bk, 1.0, 1.0, bk**2)
        pi = _cel(bk, b[body] / a[body], 1.0, 1.0)
        moment[body] = (2 / (9 * math.pi * np.sqrt(h))) * (
            (1 - 5 * bz**2 + p**2 + bq**2) * k
            + h * (bz**2 + 7 * p**2 - 4) * e - 3 * (bq / a[body]) * pi
        )
    return moment + np.where(z < p, 2 / 3, 0)


def quadratic_limb_dark_flux(separation: np.ndarray, rp_rs: float, u1: float, u2: float) -> np.ndarray:
    """Return normalized quadratic limb-darkened flux for float64 separations.

    Mandel & Agol (2002), ApJ 580, L171, arXiv:astro-ph/0210099, Eq. (1),
    Sec. 4 and Table 1. Mirrors TREX's JAX ``_analytic_limb_dark`` moments.
    Uses the finite lambda_5 plus step limit (including sqrt(p*(1-p))),
    the corrected 2p and 1/(2p) moduli, and exact full coverage; see also
    Agol et al. (2020), DOI 10.3847/1538-3881/ab4fee, Appendix A.

    Distances and radius ratio are relative to the source radius; separation
    uses its magnitude. A zero ratio or infinite separation leaves unit flux.
    A negative ratio reflects the flux about unity. NaN separations and invalid
    normalization return NaN.
    """
    z = np.abs(np.asarray(separation, dtype=np.float64))
    p = abs(rp_rs)
    norm = 1 - u1 / 3 - u2 / 6
    if not all(map(math.isfinite, (p, u1, u2, norm))) or norm == 0:
        return np.full_like(z, np.nan)
    flux = np.ones_like(z)
    flux[np.isnan(z)] = np.nan
    if p == 0:
        return flux
    complete = (p >= 1) & (z <= p - 1)
    flux[complete] = 0
    active = (z < 1 + p) & ~complete
    if not np.any(active):
        return 2 - flux if rp_rs < 0 else flux
    d = z[active]
    p2 = p**2
    limb = d > 1 - p
    area = np.full_like(d, p2)
    radial = p2 / 2 * (p2 + 2 * d**2)
    if np.any(limb):
        lz = d[limb]
        gap = lz - p
        lens = np.sqrt(np.maximum(
            0, ((p + 1) - lz) * (lz - (p - 1)) * (lz - (1 - p)) * (lz + p + 1)
        ))
        # atan2 uses both parts of each angle and does not round a nonzero
        # contact angle to zero through an arccos argument rounded to one.
        delta = gap * (lz + p)
        d0 = 2 * p2 + delta - 1
        k0, k1 = np.arctan2(lens, d0), np.arctan2(lens, 1 + delta)
        shape = 1 + 5 * p2 + lz**2
        if p >= 8:
            # For a large occultor, write the lens in terms of x-atan(x)
            # before the O(p^3) terms in the radial moment cancel. Here
            # x = tan(k0) <= 1/sqrt(p^2-1); the next series term is < 1e-19.
            x = lens / d0
            x2 = x**2
            series = 0.0
            for n in range(9, 0, -1):
                series = (-1)**(n + 1) / (2 * n + 1) + x2 * series
            remainder = d0 * x * x2 * series
            area[limb] = (k1 + ((1 - delta) * k0 - remainder) / 2) / math.pi
            radial[limb] = (k1 + ((4 * p2 + 1 - delta**2) * k0 - shape * remainder) / 4) / (2 * math.pi)
        else:
            area[limb] = (p2 * k0 + k1 - lens / 2) / math.pi
            radial[limb] = (k1 + 2 * radial[limb] * k0 - shape * lens / 4) / (2 * math.pi)
    mu = np.zeros_like(d)
    if u1 + 2 * u2 != 0:
        # Snap only within the former 1e-14 float64 geometric contact guard:
        # evaluating the uncancelled terms at a zero complementary modulus is singular.
        at_limb = (p < 1) & (abs(d - (1 - p)) < 1e-14)
        at_center = (abs(d - p) < 1e-14) & ~at_limb
        concentric = (d == 0) & ~at_limb & ~at_center
        generic = ~(at_limb | at_center | concentric)
        if np.any(at_limb):
            mu[at_limb] = (2 * math.acos(1 - 2 * p) / (3 * math.pi)
                           - 4 * math.sqrt(p * (1 - p)) * (3 + 2 * p - 8 * p2) / (9 * math.pi))
        if np.any(at_center):
            if p == 0.5:
                moment = 1 / 3 - 4 / (9 * math.pi)
            else:
                modulus = 2 * p if p < 0.5 else 0.5 / p
                kc = math.sqrt((1 - modulus) * (1 + modulus))
                m = modulus**2
                # Agol et al. (2020), TeX `eq:biglam_stable`, b=r limits.
                if p < 0.5:
                    moment = 1 / 3 + 2 * _cel(kc, 1.0, m - 3, (1 - m) * (2 * m - 3)) / (9 * math.pi)
                else:
                    moment = 1 / 3 + 4 * p * _cel(kc, 1.0, 1 - 3 * m, m - 1) / (9 * math.pi)
            mu[at_center] = moment
        if np.any(concentric):
            mu[concentric] = -(2 / 3) * math.expm1(1.5 * math.log1p(-p2))
        if np.any(generic):
            mu[generic] = _elliptic_moments(d[generic], p, limb[generic])
    c2 = u1 + 2 * u2
    flux[active] = 1 - ((1 - c2) * area + c2 * mu + u2 * radial) / norm
    return 2 - flux if rp_rs < 0 else flux


def quadratic_transit_flux(
    time_days: np.ndarray, period_days: float, rp_rs: float, a_rs: float,
    inc_deg: float, u1: float, u2: float, ecc: float = 0.0,
    argp_deg: float = 90.0, t0_days: float = 0.0,
) -> np.ndarray:
    """Instantaneous normalized flux at times in days (no exposure averaging)."""
    separation, front = kepler_sky_geometry(time_days, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days)
    return quadratic_limb_dark_flux(np.where(front, separation, np.inf), rp_rs, u1, u2)
