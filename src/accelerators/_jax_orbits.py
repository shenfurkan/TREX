"""JAX Keplerian orbit and transit geometry.

Batched device forms of the CPU batman reference in ``trex.models.likelihoods``.
Times are days, angles degrees, radii and separations in units of the
occulted body's radius, fluxes normalized to unity; all arithmetic is float64.

Sources:

- Mandel & Agol (2002), ApJ 580, L171, DOI 10.1086/345520 (quadratic limb-darkened occultation).
- Agol et al. (2020), AJ 159, 123, DOI 10.3847/1538-3881/ab4fee (jaxoplanet's occultation kernel).
- Kreidberg (2015), PASP 127, 1161, DOI 10.1086/683602 (batman, the CPU reference matched here).
- Winn (2010), in *Exoplanets*, ed. S. Seager, Univ. Arizona Press, pp. 55-77, Eq. 5,
  arXiv:1001.2010 (sky-projected separation).
- Murray & Correia (2010), in *Exoplanets*, ed. S. Seager, Univ. Arizona Press (Keplerian orbits).
"""
from __future__ import annotations

from typing import Tuple

from trex.accelerators._jax_runtime import jax, jnp, core_limb_dark
from trex.accelerators._kepler_culling import _mean_anomaly, contact_windows, in_contact_window
from trex.foundation.constants import Rearth, Rsun
from trex.models.flux_calibration import binary_flux_weights
from jaxoplanet.core.kepler import kepler as _device_kepler


def _kepler_true_anomaly(M: jnp.ndarray, ecc: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """sin(f), cos(f) from the mean anomaly; tracer-safe, branch-free, float64.

    Kepler's equation E - e sin E = M is solved by jaxoplanet's device solver,
    the same one ``trex.evidence._jax_unit_integrand`` already uses for the
    nested native path: a cubic starting approximation refined by one
    fourth-order Householder step, with no convergence loop and no iteration
    count to justify. The replaced local form seeded four Halley steps from a
    second-order series in e, whose start is far below the periastron root as
    e -> 1, so its accuracy over the near-unity tail the Kipping (2013) and
    Moe & Di Stefano (2017) priors admit was not established.

    ``M`` is any finite real: sin f and cos f are periodic in E and the solver
    reduces the mean anomaly modulo one turn before solving. ``ecc`` is in
    [0, 1); colliding periastra are removed before either native path sees a
    draw (``marginal_likelihoods._geometric_transit_mask``). Accuracy in
    ``sin f``/``cos f`` and in the resulting sky separation is measured against
    an independent bisection solution in
    ``tests/trex/accelerators/test_jax_kepler_domain.py``.
    """
    return _device_kepler(M, ecc)


def _mean_anomaly_at(time, period_days, ecc, argp_deg, t0_days=0.0):
    """Mean anomaly at a time, with ``t0_days`` at inferior conjunction.

    The same convention as :func:`_sky_geometry`: at ``t0_days`` the argument
    of latitude theta = omega + f is pi/2 (mid-transit), Winn (2010) Eq. 5.
    """
    return (_mean_anomaly(0.5 * jnp.pi - jnp.radians(argp_deg), ecc)
            + (2.0 * jnp.pi / period_days) * (time - t0_days))


def _sky_geometry(time, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days):
    """Sky separation in host radii and whether the orbiting body is in front.

    ``t0_days`` is mid-transit (inferior conjunction), where the argument of
    latitude theta = omega + f is pi/2. Batman's convention, Winn (2010) Eq. 5.
    """
    M = _mean_anomaly_at(time, period_days, ecc, argp_deg, t0_days)
    return _geometry_from_mean_anomaly(M, a_rs, inc_deg, ecc, argp_deg)


def _geometry_from_mean_anomaly(M, a_rs, inc_deg, ecc, argp_deg):
    argp_rad = jnp.radians(argp_deg)
    inc_rad = jnp.radians(inc_deg)
    sin_f, cos_f = _kepler_true_anomaly(M, ecc)
    sin_argp, cos_argp = jnp.sin(argp_rad), jnp.cos(argp_rad)
    sin_theta = sin_argp * cos_f + cos_argp * sin_f
    cos_theta = cos_argp * cos_f - sin_argp * sin_f
    r = a_rs * (1.0 - jnp.square(ecc)) / (1.0 + ecc * cos_f)
    d = r * jnp.sqrt(jnp.square(cos_theta) + jnp.square(sin_theta) * jnp.square(jnp.cos(inc_rad)))
    # +Z points to the observer; Z > 0 puts the orbiting body in front of the host.
    return d, r * sin_theta * jnp.sin(inc_rad) > 0.0


def _contact_geometry(time, period, a_rs, inc_deg, ecc, argp_deg, t0, ratio, *, secondary):
    """Solve only necessary geometry on the actual quadrature nodes, not bin centres."""
    M = _mean_anomaly_at(time, period, ecc, argp_deg, t0)
    p0, pw, s0, sw = contact_windows(ecc, argp_deg, a_rs, ratio, period, 0.0)
    # A small angular roundoff allowance protects an endpoint computed by a
    # different sequence of arithmetic than the Kepler solution.
    margin = 64 * jnp.finfo(jnp.float64).eps * (1.0 + jnp.abs(M))
    active = in_contact_window(M, p0 - margin, pw + 2 * margin)
    if secondary:
        active |= in_contact_window(M, s0 - margin, sw + 2 * margin)
    physical = (jnp.isfinite(inc_deg) & jnp.isfinite(ratio) & (ratio > 0)
                & jnp.isfinite(a_rs) & (a_rs > 0) & jnp.isfinite(ecc) & (ecc >= 0) & (ecc < 1)
                & jnp.isfinite(period) & (period > 0) & jnp.isfinite(argp_deg))
    valid = physical & jnp.isfinite(M)

    def solve(_):
        return _geometry_from_mean_anomaly(jnp.where(active, M, 0.0), a_rs, inc_deg, ecc, argp_deg)

    d, front = jax.lax.cond(jnp.any(active & valid), solve,
                            lambda _: (jnp.zeros_like(M), jnp.zeros_like(M, dtype=bool)), None)
    valid &= ~active | jnp.isfinite(d)
    return jnp.where(active, d, jnp.inf), front, valid


def _jax_batman_sky_separation(
    time: jnp.ndarray,
    period_days: jnp.ndarray,
    a_rs: jnp.ndarray,
    inc_deg: jnp.ndarray,
    ecc: jnp.ndarray,
    argp_deg: jnp.ndarray,
    t0_days: jnp.ndarray,
) -> jnp.ndarray:
    """Occultation separation in host radii; infinite when the body is behind the host.

    Batman's finite far-side sentinel (100) is not enough for an EB secondary,
    whose occultor radius 1/k exceeds 99 when k < 0.0101; jaxoplanet's kernel
    treats an infinite separation as no overlap.
    """
    d, front = _sky_geometry(time, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days)
    return jnp.where(front | ~jnp.isfinite(d), d, jnp.inf)


def _pair_occultation_separations(d, front, radius_ratio):
    """Primary and secondary separations from one host-frame geometry.

    ``d`` is the sky separation in host radii from :func:`_sky_geometry`, and
    ``front`` marks the companion in front of the host. The primary occultation
    uses ``d`` with the companion radius ratio ``k``; the secondary eclipse is
    the same orbit seen from the companion, where every separation is ``d / k``
    and the host's radius ratio is ``1 / k``. The host is in front exactly when
    the companion is not. Quadratic limb darkening evaluates an infinite
    separation as no overlap; fractional pixels are never returned for the
    back-side pair member.

    ``radius_ratio`` must be positive; callers reject a collided/zero-radius
    configuration explicitly rather than letting the division hide it.
    """
    finite_geometry = jnp.isfinite(d)
    primary = jnp.where(front | ~finite_geometry, d, jnp.inf)
    secondary = jnp.where(front & finite_geometry, jnp.inf, d / radius_ratio)
    return primary, secondary


def _exposure_grid(time, exptime_days, nsamples):
    """Midpoint sub-exposure times, shape (n_time, nsamples), or the times themselves."""
    if nsamples <= 1:
        return time
    sub_offsets = exptime_days * ((jnp.arange(nsamples) + 0.5) / nsamples - 0.5)
    return time[:, None] + sub_offsets[None, :]


def _exposure_mean(delta, nsamples):
    return jnp.mean(delta, axis=-1) if nsamples > 1 else delta


def _jax_transit(
    time: jnp.ndarray,
    period_days: float,
    rp_rs: float,
    a_rs: float,
    inc_deg: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    nsamples: int = 20,
    t0_days: float = 0.0,
) -> jnp.ndarray:
    """Evaluate quadratic Mandel-Agol flux with numerical Keplerian supersampling."""
    f64 = jnp.float64
    t_grid = _exposure_grid(jnp.asarray(time, dtype=f64), exptime_days, nsamples)
    radius = jnp.asarray(rp_rs, dtype=f64)
    d, front, valid = _contact_geometry(
        t_grid, period_days, a_rs, inc_deg, ecc, argp_deg, t0_days, radius, secondary=False,
    )
    delta = core_limb_dark(jnp.array([u1, u2], dtype=f64), jnp.where(front, d, jnp.inf), radius)
    return 1.0 + _exposure_mean(jnp.where(valid, delta, jnp.nan), nsamples)


def _jax_secondary_eclipse_phase(ecc: jnp.ndarray, argp_deg: jnp.ndarray) -> jnp.ndarray:
    """Secondary-eclipse phase, the device form of ``orbital_mechanics.secondary_eclipse_phase``."""
    argp = jnp.radians(argp_deg)
    transit = _mean_anomaly(0.5 * jnp.pi - argp, ecc)
    occultation = _mean_anomaly(1.5 * jnp.pi - argp, ecc)
    return (occultation - transit) % (2.0 * jnp.pi) / (2.0 * jnp.pi)


def _simulate_tp_jax(
    time: jnp.ndarray,
    R_p_earth: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
) -> jnp.ndarray:
    """Simulate a transiting planet light curve using JAX with third-light dilution."""
    rp_rs = R_p_earth * Rearth / (R_s_solar * Rsun)
    a_rs = a_cm / (R_s_solar * Rsun)
    flux = _jax_transit(time, P_orb, rp_rs, a_rs, inc_deg, u1, u2, exptime_days,
                        ecc, argp_deg, nsamples, t0_days=0.0)
    host = jnp.where(companion_is_host, companion_fluxratio, 1.0 - companion_fluxratio)
    return jnp.where(companion_fluxratio > 0.0, host * flux + (1.0 - host), flux)


def _simulate_eb_jax(
    time: jnp.ndarray,
    R_EB_solar: float,
    EB_fluxratio: float,
    P_orb: float,
    inc_deg: float,
    a_cm: float,
    R_s_solar: float,
    u1: float,
    u2: float,
    exptime_days: float,
    ecc: float = 0.0,
    argp_deg: float = 90.0,
    companion_fluxratio: float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
    *,
    u1_EB: float,
    u2_EB: float,
    secondary_to_host_flux_ratio=None,
) -> jnp.ndarray:
    """Simulate an eclipsing binary light curve using JAX (primary + secondary eclipse).

    The primary eclipse is the EB star (radius ratio k = R_EB / R_s) crossing
    the host. The secondary eclipse is the same orbit seen from the EB star: in
    EB radii the host's radius is 1/k and every separation is the primary one
    divided by k, and the host is in front exactly when the EB star is behind.
    The CPU reference gets the same geometry from a second batman orbit with
    a/k, omega + 180 deg and mid-eclipse at the secondary phase; here the Kepler
    solution is shared. ``k`` must be positive: zero gives a non-finite model,
    which the batch scheduler records as a numerical failure.
    """
    f64 = jnp.float64
    k = jnp.asarray(R_EB_solar / R_s_solar, dtype=f64)
    a_rs = a_cm / (R_s_solar * Rsun)
    t_grid = _exposure_grid(jnp.asarray(time, dtype=f64), exptime_days, nsamples)
    d, eb_in_front, valid = _contact_geometry(
        t_grid, P_orb, a_rs, inc_deg, ecc, argp_deg, 0.0, k, secondary=True,
    )
    primary_separation, secondary_separation = _pair_occultation_separations(d, eb_in_front, k)
    primary_flux = 1.0 + _exposure_mean(
        jnp.where(valid, core_limb_dark(jnp.array([u1, u2], dtype=f64), primary_separation, k), jnp.nan), nsamples)
    secondary_flux = 1.0 + _exposure_mean(
        jnp.where(valid, core_limb_dark(jnp.array([u1_EB, u2_EB], dtype=f64), secondary_separation, 1.0 / k), jnp.nan),
        nsamples)

    if secondary_to_host_flux_ratio is None:
        companion = companion_fluxratio / (1.0 - companion_fluxratio)
        secondary = EB_fluxratio / (1.0 - EB_fluxratio)
        host = jnp.where(companion_is_host, companion, 1.0)
        other = jnp.where(companion_is_host, 1.0, companion)
        total = host + secondary + other
        weights = host / total, secondary / total, other / total
    else:
        weights = binary_flux_weights(secondary_to_host_flux_ratio, companion_fluxratio,
                                      companion_is_host, array_module=jnp)
    model = weights[0] * primary_flux + weights[1] * secondary_flux + weights[2]
    # Invalid radius ratios must remain failures, not diluted flat curves.
    return jnp.where((k > 0.0) & jnp.isfinite(k), model, jnp.nan)
