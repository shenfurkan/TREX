"""Opt-in analytic quadratic limb-darkening occultation flux in float64 JAX.

Candidate for the accelerators layer. Nothing in TREX imports it, no
environment variable selects it and no default changes, so every existing
result stands until adoption is decided on numerical evidence.

Conventions
-----------
``u1`` and ``u2`` are ``gamma1`` and ``gamma2`` of Mandel & Agol (2002) in the
radiative law ``I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`` with
``mu = sqrt(1 - r^2)``. Separations and radii are in units of the occulted
star's radius. ``quadratic_disc_flux`` returns the flux deficit ``F - 1`` for
the flux normalized to the unobscured star, so a transit is negative; this is
the sign contract of ``jaxoplanet.core.limb_dark.light_curve``, the kernel this
module is a candidate to replace.

Sources
-------
- Mandel, K. & Agol, E. (2002), ApJ 580, L171, DOI 10.1086/345520,
  Section 4 "Quadratic Limb Darkening" and Table 1.
- Bulirsch, R. (1969), Numer. Math. 13, 266, DOI 10.1007/BF02165405,
  Sections 1-2 (the general complete elliptic integral ``cel``).
"""

from __future__ import annotations

from typing import Tuple

import jax

# TREX evaluates every numerical path in float64. The flag must precede the
# array-API import so nothing is traced in float32 first.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

__all__ = ["cel", "quadratic_disc_flux"]

# Unrolled AGM budget of ``cel``. See its docstring for the a-priori argument:
# nine steps escape the smallest positive float64 complementary modulus, six
# further quadratic steps take the relative gap below 1e-16, and three extra
# steps absorb the rounding those inequalities neglect.
_CEL_STEPS = 18
_CEL_RELATIVE_GAP = 1.0e-15


def _cel_with_steps(kc, p, a, b, n_steps: int):
    """Bulirsch's ``cel`` with an explicit AGM budget; see :func:`cel`.

    ``n_steps`` is the number of unrolled AGM steps actually taken. The public
    entry point fixes it at ``_CEL_STEPS``; the tests double it to measure
    self-convergence. The first step whose relative gap meets
    ``_CEL_RELATIVE_GAP`` is kept, so extra steps cost arithmetic only.
    """
    dtype = jnp.float64
    kc = jnp.asarray(kc, dtype=dtype)
    p = jnp.asarray(p, dtype=dtype)
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)

    # The p <= 0 Cauchy-principal-value branch of Bulirsch's algorithm is not
    # implemented, so p == 0 (logarithmic singularity) and every non-finite or
    # out-of-domain argument are reported as NaN rather than a finite value.
    valid = (
        jnp.isfinite(kc) & jnp.isfinite(p) & jnp.isfinite(a) & jnp.isfinite(b)
        & (kc > 0.0) & (kc <= 1.0) & (p > 0.0)
    )

    scale = jnp.sqrt(jnp.where(p > 0.0, p, 1.0))
    current = a
    weight = b / scale
    arithmetic = jnp.ones_like(kc)
    product = jnp.where(valid, kc, 1.0)
    modulus = jnp.where(valid, kc, 1.0)
    result = jnp.full_like(kc, jnp.nan)
    done = jnp.zeros_like(kc, dtype=bool)

    for _ in range(n_steps):
        previous = current
        current = weight / scale + current
        ratio = product / scale
        weight = 2.0 * (previous * ratio + weight)
        scale = ratio + scale
        previous_mean = arithmetic
        arithmetic = modulus + arithmetic
        converged = jnp.abs(1.0 - modulus / previous_mean) <= _CEL_RELATIVE_GAP
        candidate = 0.5 * jnp.pi * (current * arithmetic + weight) / (
            arithmetic * (arithmetic + scale)
        )
        result = jnp.where(converged & ~done, candidate, result)
        done = done | converged
        modulus = 2.0 * jnp.sqrt(product)
        product = modulus * arithmetic

    return jnp.where(valid, result, jnp.nan)


def cel(kc, p, a, b):
    """Bulirsch's general complete elliptic integral, float64 and branch-free.

    ``cel(kc, p, a, b)`` evaluates

    .. math::

        \\int_0^{\\pi/2}
        \\frac{a\\cos^2 t + b\\sin^2 t}
             {(\\cos^2 t + p\\sin^2 t)\\sqrt{\\cos^2 t + kc^2\\sin^2 t}}\\,dt

    with ``kc`` the complementary modulus and ``p`` the integral's parameter
    (Bulirsch 1969, Numer. Math. 13, 266, DOI 10.1007/BF02165405, Sections 1-2;
    the recurrences below are his published arithmetic-geometric-mean
    algorithm, transcribed branch-free with ``jnp.where`` and a fixed unrolled
    step count, no trigonometric function and no ``while_loop``).

    Special cases used by callers: ``cel(kc, 1, 1, 1) = K(k)`` and
    ``cel(kc, 1, 1, kc^2) = E(k)`` with ``k^2 = 1 - kc^2``, and
    ``cel(kc, 1 - n, 1, 1)`` is the complete third-kind integral
    ``Pi(n, k) = int dt / ((1 - n sin^2 t) sqrt(1 - k^2 sin^2 t))``, the sign
    convention in which Mandel & Agol (2002) Table 1 states its ``Pi``
    arguments.

    Domain: ``kc`` in (0, 1] and ``p >= 0``. ``p == 0`` is the integrand's
    logarithmic singularity and is reported as NaN, never as a silently finite
    value; ``p < 0`` needs the Cauchy principal value, which this module does
    not use and does not implement. Any non-finite argument, ``kc <= 0`` or
    ``kc > 1`` also returns NaN. Inside the domain the intermediate products
    overflow float64 only for parameters below roughly 1e-300, where the
    integral itself is ~1e161; ``quadratic_disc_flux`` always calls ``cel``
    with ``p >= 1``.

    Cost and accuracy: the AGM is run for at most ``_CEL_STEPS`` = 18
    unrolled steps, which is fixed rather than data-dependent so the kernel
    traces and compiles once. Over the whole positive float64 domain the
    complementary modulus obeys ``kc' = 2 sqrt(kc * em) >= 2 sqrt(kc)`` because
    the mean ``em`` starts at 1 and only grows, i.e.
    ``log2 kc' >= 1 + log2 kc / 2``. From the smallest positive float64
    ``kc = 5e-324`` (``log2 kc = -1074``) that bound reaches ``kc >= 1`` after
    nine steps; once modulus and mean are within a factor two of each other
    each step contracts the relative gap quadratically (``e' ~ e^2 / 4``), so
    six further steps take a gap of 1 below 1e-16. Fifteen steps therefore
    bound the relative gap over the complete float64 ``kc`` domain and the
    budget keeps three spare steps. If a pathological input defeats the
    budget, the returned value stays NaN instead of a finite wrong number.
    """
    return _cel_with_steps(kc, p, a, b, _CEL_STEPS)


def quadratic_disc_flux(u1, u2, separation, radius_ratio) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Quadratic limb-darkened occultation of a star by an opaque disk.

    Evaluates the Mandel & Agol (2002) analytic light curve (ApJ 580, L171,
    DOI 10.1086/345520, Section 4 and Table 1) for the radiative law
    ``I(mu) = 1 - u1 (1 - mu) - u2 (1 - mu)^2`` with
    ``mu = sqrt(1 - r^2)``, ``gamma1 = u1`` and ``gamma2 = u2``.

    Parameters
    ----------
    u1, u2 : array_like
        Limb-darkening coefficients. They must be finite; ``u1 + u2 <= 1`` and
        ``u1, u2 >= 0`` are the caller's physical contract (it makes
        ``I(mu) >= 0``, hence ``Omega = 1 - u1/3 - u2/6 >= 2/3``). The
        coefficients are not clipped: outside that contract the algebraic
        continuation is returned and may be unphysical.
    separation : array_like
        Sky-projected center-to-center distance in stellar radii; its absolute
        value is used, as in the batman and jaxoplanet kernels.
    radius_ratio : array_like
        Occulting radius in stellar radii; values above one are the
        eclipsing-binary secondary case and are fully supported.

    Returns
    -------
    (delta_flux, valid) : tuple of arrays
        ``delta_flux = F - 1`` for the normalized flux ``F`` (negative during
        occultation), float64, and a boolean mask. ``valid`` is False and
        ``delta_flux`` NaN when any argument is non-finite,
        ``radius_ratio <= 0``, ``Omega == 0`` (no unocculted flux to
        normalize by, e.g. ``u1 = 3, u2 = 0``) or the evaluated flux is not
        finite; a NaN is never returned with ``valid`` True. The geometric
        no-overlap and complete-coverage results are *valid* exact zeros and
        exact minus ones, never NaN.

    Exact limits (returned by short circuit, not by the elliptic forms):
    ``separation >= 1 + radius_ratio`` gives exactly ``0.0``;
    ``radius_ratio >= 1`` and ``separation <= radius_ratio - 1`` gives exactly
    ``-1.0``. The boundary rows ``separation == 1 + radius_ratio`` and
    ``separation == |1 - radius_ratio|`` are finite and equal the one-sided
    limits: the outer one is the zero above; the inner one is ``lambda5`` for
    ``radius_ratio < 1`` (with the ``p -> 1`` limit ``-1.0``), and the exact
    full-coverage ``-1.0`` for ``radius_ratio >= 1``.

    Adopted relations
    -----------------
    ``F = 1 - [(1 - c2) lambda_e + c2 (lambda_d + (2/3) Theta(p - z))
    - c4 eta_d] / Omega`` with ``c2 = u1 + 2 u2``, ``c4 = -u2``,
    ``Theta`` the Heaviside step, ``z = |separation|``, ``p = radius_ratio``
    and ``Omega = 1 - u1/3 - u2/6`` (Section 4, immediately after the
    intensity law). Section 4 prints the denominator as ``(4 Omega)^{-1}`` with
    ``Omega = sum c_n/(n + 4)``; for the quadratic law that sum equals
    ``(1 - u1/3 - u2/6)/4``, so ``(4 Omega)^{-1}`` is ``1/Omega`` as written
    here, and the direct area integral in the tests reproduces this normalised
    form.

    ``lambda_e = [p^2 kappa0 + kappa1 - sqrt(4 z^2 - (1 + z^2 - p^2)^2)/2] / pi``
    with ``kappa0 = arccos[(p^2 + z^2 - 1)/(2 p z)]`` and
    ``kappa1 = arccos[(1 - p^2 + z^2)/(2 z)]`` (Equation 1).

    Table 1, with ``a = (z - p)^2``, ``b = (z + p)^2``, ``q = p^2 - z^2`` and
    ``Pi`` evaluated as ``cel(kc, 1 - n, 1, 1)``:

    - ``lambda1 = [((1 - b)(2b + a - 3) - 3 q (b - 2)) K(k)
      + 4 p z (z^2 + 7 p^2 - 4) E(k) - 3 (q/a) Pi((a - 1)/a, k)] /
      (9 pi sqrt(p z))`` with ``k^2 = (1 - a)/(b - a)`` (cases II and VIII).
    - ``lambda2 = 2 [ (1 - 5z^2 + p^2 + q^2) K(1/k)
      + (1 - a)(z^2 + 7p^2 - 4) E(1/k) - 3 (q/a) Pi((a - b)/a, 1/k)] /
      (9 pi sqrt(1 - a))`` (cases III and IX).
    - ``lambda3 = 1/3 + 16p (2p^2 - 1) E(m) / (9 pi)
      - (1 - 4p^2)(3 - 8p^2) K(m) / (9 pi p)`` with ``m = 1/(2p)`` (case VII).
    - ``lambda4 = 1/3 + 2 [4 (2p^2 - 1) E(m) + (1 - 4p^2) K(m)] / (9 pi)``
      with ``m = 2p`` (case V).
    - ``lambda5 = 2 arccos(1 - 2p) / (3 pi)
      - 4 sqrt(p (1 - p)) (3 + 2p - 8p^2) / (9 pi)`` (case IV).
    - ``lambda6 = -(2/3)(1 - p^2)^(3/2)`` (case X).
    - ``eta1 = [kappa1 + 2 eta2 kappa0
      - (1 + 5p^2 + z^2) sqrt((1 - a)(b - 1))/4] / (2 pi)`` (cases II, VII,
      VIII) and ``eta2 = p^2 (p^2 + 2 z^2)/2`` (cases III, IV, V, IX, X).

    Three readings of the retained text are stated because the printed rows are
    not self-consistent there. (i) ``lambda5`` is printed without the
    ``sqrt(p (1 - p))`` factor, which would give ``-4/(3 pi)`` at ``p -> 0``
    and contradict the case VI value ``1/3 - 4/(9 pi)`` at ``p = 1/2``; the
    factor is restored, which is the continuous limit of the case III/IV row
    and gives ``lambda5(1) = 2/3``, the ``p -> 1`` case X limit
    ``lambda6 + 2/3``. (ii) The modulus arguments printed as ``E(2k)`` in ``lambda4`` and
    ``E(1/(2k))`` in ``lambda3`` are read as the moduli ``2p`` and ``1/(2p)``:
    at ``z = p`` the printed ``k`` equals ``1/(2p)``, and only ``2p < 1`` for
    ``p < 1/2`` (respectively ``1/(2p) < 1`` for ``p > 1/2``) keeps the row
    real, which is exactly the case V / case VII domain; both rows then
    reproduce ``lambda2`` and ``lambda1`` at ``z = p`` term by term, including
    the ``0 * K(1)`` limit at ``p = 1/2`` that reproduces the case VI value.
    (iii) The ``Pi`` terms are evaluated as ``cel(kc, 1 - n, 1, 1)``; this is
    the mapping under which both ``z -> p`` limits of ``lambda1`` and
    ``lambda2`` differ from the case V/VII rows by exactly the ``2/3`` of the
    Heaviside term, i.e. under which the flux is continuous across ``z = p``.
    The case XI row ``(lambda, eta) = (1, 1)`` is not used: the exact ``-1.0``
    of full coverage comes from the short circuit, and the closed form itself
    reaches ``-1.0`` at that boundary with ``eta = 1/2``.

    The tangent rows are evaluated in closed form rather than as the limiting
    elliptic expressions: at ``z = 1 - p`` with ``p < 1`` the sum
    ``lambda_d + (2/3) Theta`` equals ``lambda5`` for every ``p < 1`` (the
    table prints that row only for ``p < 1/2``, where the Heaviside term
    vanishes; the ``p > 1/2`` continuation is fixed by continuity at
    ``p = 1/2`` and ``p = 1``), and at ``z = p`` the sum equals ``lambda4``,
    the case VI constant or ``lambda3`` for ``p < 1/2``, ``p == 1/2`` and
    ``p > 1/2`` respectively, with ``eta2``, ``3/32`` or ``eta1``. Points within
    one float64 step of those rows are still evaluated by the neighbouring
    closed form, whose log-divergent terms cancel there: the value stays finite
    and its accuracy degrades smoothly towards the row value rather than
    jumping or returning NaN.

    No custom JVP is registered: TREX has no gradient use for this kernel.
    """
    dtype = jnp.float64
    u1 = jnp.asarray(u1, dtype=dtype)
    u2 = jnp.asarray(u2, dtype=dtype)
    separation = jnp.asarray(separation, dtype=dtype)
    radius_ratio = jnp.asarray(radius_ratio, dtype=dtype)

    valid = (
        jnp.isfinite(u1) & jnp.isfinite(u2) & jnp.isfinite(separation)
        & jnp.isfinite(radius_ratio) & (radius_ratio > 0.0)
    )
    # Absolute separation, as in the batman and jaxoplanet kernels; the invalid
    # placeholders keep the unselected branch finite instead of propagating inf.
    z = jnp.abs(separation)
    p = jnp.where(valid, radius_ratio, 1.0)

    a = jnp.square(z - p)
    b = jnp.square(z + p)
    q = jnp.square(p) - jnp.square(z)
    one_minus_a = 1.0 - a

    outside = z >= 1.0 + p
    # ``b <= 1`` with ``p >= 1`` is full coverage within rounding of the
    # ``z = p - 1`` row (e.g. ``p = 1``, ``z < 1e-16``); there both elliptic
    # moduli below round to zero and ``cel`` would return NaN.
    complete = (p >= 1.0) & ((z <= p - 1.0) | (b <= 1.0))
    concentric = (z == 0.0) & (p < 1.0)

    # Safe denominators: every quantity below is discarded where these vanish
    # (z = 0, z = p, z = 1 +- p), but keeping them finite keeps the traced
    # unselected branches free of inf/NaN.
    span = jnp.where(b > a, b - a, 1.0)
    body_radius2 = jnp.where(a > 0.0, a, 1.0)
    one_minus_a_safe = jnp.where(one_minus_a > 0.0, one_minus_a, 1.0)
    z_safe = jnp.where(z > 0.0, z, 1.0)
    zp_safe = jnp.where(z * p > 0.0, z * p, 1.0)

    kc_limb = jnp.sqrt(jnp.clip((b - 1.0) / span, 0.0, 1.0))
    kc_body = jnp.sqrt(jnp.clip((1.0 - b) / one_minus_a_safe, 0.0, 1.0))

    k_limb = cel(kc_limb, 1.0, 1.0, 1.0)
    e_limb = cel(kc_limb, 1.0, 1.0, kc_limb * kc_limb)
    pi_limb = cel(kc_limb, 1.0 / body_radius2, 1.0, 1.0)

    k_body = cel(kc_body, 1.0, 1.0, 1.0)
    e_body = cel(kc_body, 1.0, 1.0, kc_body * kc_body)
    pi_body = cel(kc_body, b / body_radius2, 1.0, 1.0)

    lambda1 = (
        ((1.0 - b) * (2.0 * b + a - 3.0) - 3.0 * q * (b - 2.0)) * k_limb
        + 4.0 * z * p * (jnp.square(z) + 7.0 * jnp.square(p) - 4.0) * e_limb
        - 3.0 * (q / body_radius2) * pi_limb
    ) / (9.0 * jnp.pi * jnp.sqrt(zp_safe))

    lambda2 = 2.0 / (9.0 * jnp.pi * jnp.sqrt(one_minus_a_safe)) * (
        (1.0 - 5.0 * jnp.square(z) + jnp.square(p) + jnp.square(q)) * k_body
        + one_minus_a * (jnp.square(z) + 7.0 * jnp.square(p) - 4.0) * e_body
        - 3.0 * (q / body_radius2) * pi_body
    )

    # Tangent rows: z = p (cases V, VI, VII) and z = 1 - p (case IV).
    modulus_body = 2.0 * p
    kc_tangent_body = jnp.sqrt(jnp.clip(1.0 - modulus_body * modulus_body, 0.0, 1.0))
    lambda4 = 1.0 / 3.0 + 2.0 / (9.0 * jnp.pi) * (
        4.0 * (2.0 * jnp.square(p) - 1.0)
        * cel(kc_tangent_body, 1.0, 1.0, kc_tangent_body * kc_tangent_body)
        + (1.0 - 4.0 * jnp.square(p)) * cel(kc_tangent_body, 1.0, 1.0, 1.0)
    )

    modulus_limb = 1.0 / (2.0 * p)
    kc_tangent_limb = jnp.sqrt(jnp.clip(1.0 - modulus_limb * modulus_limb, 0.0, 1.0))
    lambda3 = (
        1.0 / 3.0
        + 16.0 * p / (9.0 * jnp.pi) * (2.0 * jnp.square(p) - 1.0)
        * cel(kc_tangent_limb, 1.0, 1.0, kc_tangent_limb * kc_tangent_limb)
        - (1.0 - 4.0 * jnp.square(p)) * (3.0 - 8.0 * jnp.square(p))
        * cel(kc_tangent_limb, 1.0, 1.0, 1.0) / (9.0 * jnp.pi * p)
    )

    lambda5 = (
        2.0 / (3.0 * jnp.pi) * jnp.arccos(jnp.clip(1.0 - 2.0 * p, -1.0, 1.0))
        - 4.0 / (9.0 * jnp.pi) * jnp.sqrt(jnp.maximum(p * (1.0 - p), 0.0))
        * (3.0 + 2.0 * p - 8.0 * jnp.square(p))
    )
    lambda6 = -(2.0 / 3.0) * jnp.power(jnp.maximum(1.0 - jnp.square(p), 0.0), 1.5)

    kappa0 = jnp.arccos(
        jnp.clip((jnp.square(p) + jnp.square(z) - 1.0) / (2.0 * zp_safe), -1.0, 1.0)
    )
    kappa1 = jnp.arccos(
        jnp.clip((1.0 - jnp.square(p) + jnp.square(z)) / (2.0 * z_safe), -1.0, 1.0)
    )
    lambda_e = (
        jnp.square(p) * kappa0 + kappa1
        - 0.5 * jnp.sqrt(jnp.maximum(4.0 * jnp.square(z)
                                     - jnp.square(1.0 + jnp.square(z) - jnp.square(p)), 0.0))
    ) / jnp.pi
    # A contained occultor covers exactly pi*p^2; the lens formula loses
    # precision at internal tangency through cancelling arccos/sqrt terms.
    lambda_e = jnp.where((p < 1.0) & (z <= 1.0 - p), jnp.square(p), lambda_e)
    eta2 = 0.5 * jnp.square(p) * (jnp.square(p) + 2.0 * jnp.square(z))
    eta1 = (
        kappa1 + 2.0 * eta2 * kappa0
        - 0.25 * (1.0 + 5.0 * jnp.square(p) + jnp.square(z))
        * jnp.sqrt(jnp.maximum(one_minus_a * (b - 1.0), 0.0))
    ) / (2.0 * jnp.pi)

    # Cases II, VIII and the z >= 1 - p part of VII use lambda1/eta1; cases III
    # and IX use lambda2/eta2. At z = 1 - p the two branches meet at lambda5.
    # The branch is chosen from ``b`` itself, the quantity inside both elliptic
    # moduli: ``z > 1 - p`` can hold in float64 while ``(z + p)^2`` rounds to
    # ``<= 1``, which would select lambda1 with ``kc_limb = 0`` and give NaN.
    # For ``p >= 1`` every non-complete point has ``b > 1``.
    limb = b > 1.0
    lambda_d = jnp.where(limb, lambda1, lambda2)
    eta_d = jnp.where(limb, eta1, eta2)
    lambda_sum = lambda_d + jnp.where(z < p, 2.0 / 3.0, 0.0)

    at_body_tangent = z == p
    lambda_sum = jnp.where(
        at_body_tangent,
        jnp.where(p < 0.5, lambda4, jnp.where(p > 0.5, lambda3, lambda5)),
        lambda_sum,
    )
    eta_d = jnp.where(at_body_tangent, jnp.where(p < 0.5, eta2, eta1), eta_d)

    # ``b == 1`` is the same tangent row within rounding; both moduli vanish.
    at_limb_tangent = (p < 1.0) & ((z == 1.0 - p) | (b == 1.0))
    lambda_sum = jnp.where(at_limb_tangent, lambda5, lambda_sum)
    eta_d = jnp.where(at_limb_tangent, eta2, eta_d)

    lambda_sum = jnp.where(concentric, lambda6 + 2.0 / 3.0, lambda_sum)

    c2 = u1 + 2.0 * u2
    c4 = -u2
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    # Omega is the unocculted disc flux; Omega == 0 (e.g. u1 = 3, u2 = 0) has
    # no normalized flux at all, so it is invalid rather than a division by 0.
    valid = valid & (omega != 0.0)
    omega_safe = jnp.where(omega != 0.0, omega, 1.0)
    blocked = ((1.0 - c2) * lambda_e + c2 * lambda_sum - c4 * eta_d) / omega_safe

    delta_flux = jnp.where(outside, 0.0, jnp.where(complete, -1.0, -blocked))
    # Final guard so no residual non-finite value is ever reported as valid.
    valid = valid & jnp.isfinite(delta_flux)
    return jnp.where(valid, delta_flux, jnp.nan), valid
