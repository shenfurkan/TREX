"""Occult exact piecewise-linear angular rays, without fitting a limb-darkening law.

Mandel & Agol (2002), ApJ 580, L171, DOI 10.1086/345520:
the blocked disc integral is evaluated in polar rings rather than a quadratic
basis. The ring's covered angle follows directly from the cosine rule. Radii
and separation are in units of the eclipsed body's declared atmosphere radius;
compiled angular profiles are r-method-anchored, so that radius is the
transit-visible limb, matching the stellar-radii units of the caller's radius
ratio and separation.
"""

from functools import lru_cache
from numbers import Integral

import numpy as np

# Fixed gather width for the native (JAX) active-interval rule. A ring interval
# outside the covered annulus contributes exactly zero, so gathering only the
# blocking intervals keeps the 64-node rule per contributing interval. The
# excess is a dense fallback, never a truncated sum.
_RAY_INTERVAL_CAPACITY = 64


@lru_cache(maxsize=8, typed=True)
def _ring_rule(order):
    """Reuse only parameter-independent Gauss nodes; profiles/orbits are never cached."""
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("angular quadrature order must be a positive integer")
    points, weights = np.polynomial.legendre.leggauss(order)
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


def _occultation_inputs(separation, radius_ratio, mu, intensity, *, xp):
    d, k, mu, intensity = (xp.asarray(v) for v in (separation, radius_ratio, mu, intensity))
    if any(v.dtype.kind not in "iuf" for v in (d, k, mu, intensity)):
        raise ValueError("angular occultation inputs must be real numeric values, not booleans")
    if k.ndim != 0:
        raise ValueError("occultor radius ratio must be a scalar")
    if mu.shape != intensity.shape or mu.ndim not in (1, 2) or mu.shape[-1] < 2:
        raise ValueError("invalid complete angular photon profile")
    profile_valid = (xp.all(xp.isfinite(mu)) & xp.all(xp.isfinite(intensity))
                     & xp.all(mu[..., 1:] > mu[..., :-1]) & xp.all(mu[..., 0] == 0)
                     & xp.all(mu[..., -1] == 1) & xp.all(intensity >= 0) & xp.any(intensity > 0))
    if xp is np:
        if not np.isfinite(k) or k <= 0:
            raise ValueError("occultor radius ratio must be finite and positive")
        if np.any(d < 0):
            raise ValueError("occultation separation must be nonnegative")
        if not profile_valid:
            raise ValueError("invalid complete angular photon profile")
    valid = profile_valid & xp.isfinite(k) & (k > 0) & xp.isfinite(d) & (d >= 0)
    return d, k, mu, intensity, valid


def occulted_fraction(separation, radius_ratio, mu, intensity, *, xp=np, order=64):
    """Blocked/total photons for one or a weighted mixture of ray profiles.

    Last axes are (component, ray); a single 1-D profile is also accepted.
    Intensities are absolute photon intensities, with interpolation weights
    already applied to each component. Radius ratio is a positive scalar;
    separation is nonnegative in source-radius units. Invalid host profiles raise;
    nonfinite separations and invalid native traced values return NaN, never zero
    prior. No profile/radius rescaling, limb cut or output clipping is applied.
    The existing 64-point rule is unchanged; its accuracy needs independent controls.

    Only ring intervals with a nonzero covered width and nonzero endpoint
    intensity can block photons, so both backends evaluate just those. NumPy
    gathers them; native (JAX) code gathers at most ``_RAY_INTERVAL_CAPACITY``
    of them and a ``lax.cond`` over that count selects the unchanged dense rule
    whenever a grazing/full-disc crossing needs more, so no interval is dropped.
    ``lax.cond`` executes one branch per call for scalar separations; under an
    outer ``vmap`` JAX lowers it to a select that evaluates both branches, and a
    batched separation keeps the dense rule directly.
    """
    separation, k, mu, intensity, valid = _occultation_inputs(
        separation, radius_ratio, mu, intensity, xp=xp)
    mu, intensity = xp.atleast_2d(mu), xp.atleast_2d(intensity)
    # An overall intensity scale cancels; normalizing avoids large CGS intermediates.
    scale = xp.max(intensity)
    intensity = intensity/xp.where(scale > 0, scale, 1.)
    a, b = mu[..., :-1], mu[..., 1:]
    ia, ib = intensity[..., :-1], intensity[..., 1:]
    moment = xp.sum((b-a)*((2*a+b)*ia+(a+2*b)*ib)/6)
    d = separation[..., None, None]
    # Partial rings have |d-k| < r < d+k, with mu=sqrt(1-r²).
    lo = xp.maximum(a, xp.sqrt(xp.maximum(1-(d+k)**2, 0.)))
    hi = xp.minimum(b, xp.sqrt(xp.maximum(1-(d-k)**2, 0.)))
    width = xp.maximum(hi-lo, 0.)
    points, weights = _ring_rule(order)
    angle = xp.asarray((points+1)*np.pi/2)
    rule_angle = xp.asarray(weights)*np.pi/4*xp.sin(angle)

    def rings(low, span, left, right, value_left, value_right, distance):
        """Shared partial-ring samples; zero-width rows stay exactly zero."""
        # Sinusoidal remapping regularizes the square-root contact edges [mu, dimensionless].
        sample = low[..., None]+span[..., None]*(1-xp.cos(angle))/2
        rule = rule_angle*span[..., None]
        value = value_left[..., None]+(value_right-value_left)[..., None]*(sample-left[..., None])/(right-left)[..., None]
        r = xp.sqrt(xp.maximum(1-sample**2, 0.))
        divisor = 2*distance[..., None]*r
        cosine = (r*r+distance[..., None]**2-k*k)/xp.where(divisor > 0, divisor, 1.)
        covered_angle = 2*xp.arccos(xp.clip(cosine, -1., 1.))
        return value*sample*covered_angle*rule

    if xp is np:
        # Mandel & Agol disc geometry: zero-width/zero-intensity intervals block exactly zero photons.
        active = (width > 0) & ((ia != 0) | (ib != 0))
        rows, components, rays = np.nonzero(active.reshape(-1, *a.shape))
        integral = rings(lo.reshape(-1, *a.shape)[rows, components, rays],
                         width.reshape(-1, *a.shape)[rows, components, rays],
                         a[components, rays], b[components, rays],
                         ia[components, rays], ib[components, rays],
                         d.reshape(-1)[rows])
        contributions = np.sum(integral, axis=-1)
        partial = np.bincount(rows, weights=contributions, minlength=d.size).reshape(d.shape[:-2])
    else:
        import jax
        jnp = jax.numpy
        active = ((width > 0) & ((ia != 0) | (ib != 0))).reshape(-1)
        capacity = min(_RAY_INTERVAL_CAPACITY, active.size)
        count = jnp.sum(active)

        def compact(_operand):
            index = jnp.nonzero(active, size=capacity, fill_value=0)[0]
            keep = jnp.arange(capacity) < count
            flat_lo, flat_width = lo.reshape(-1), width.reshape(-1)
            flat_a, flat_b, flat_ia, flat_ib = a.reshape(-1), b.reshape(-1), ia.reshape(-1), ib.reshape(-1)
            integral = rings(flat_lo[index], flat_width[index], flat_a[index], flat_b[index],
                             flat_ia[index], flat_ib[index], d.reshape(-1)[0])
            return jnp.sum(jnp.where(keep[:, None], integral, 0.), axis=(-2, -1))

        def dense(_operand):
            return jnp.sum(rings(lo, width, a, b, ia, ib, d), axis=(-3, -2, -1))

        partial = (jax.lax.cond(count <= capacity, compact, dense, None)
                   if separation.ndim == 0 else dense(None))
    # Rings wholly inside the occultor have r <= k-d; integrate their linear hats exactly.
    start = xp.maximum(a, xp.sqrt(xp.maximum(1-xp.maximum(k-d, 0.)**2, 0.)))
    start = xp.minimum(start, b)
    initial = ia+(ib-ia)*(start-a)/(b-a)
    full = 2*np.pi*xp.sum((b-start)*((2*start+b)*initial+(start+2*b)*ib)/6,
                         axis=(-2, -1))
    return xp.where(valid, (partial+full)/(2*np.pi*xp.where(moment > 0, moment, 1.)), xp.nan)


def disc_flux(separation, radius_ratio, profile):
    """Unit-normalized flux; 32-time chunks bound memory, not the quadrature order.

    Radius ratio is scalar; negative separations raise and nonfinite ones retain
    NaN. Profiles are complete nonnegative (component,ray) mixtures or 1-D rays.
    """
    d, radius_ratio, mu, intensity, _ = _occultation_inputs(separation, radius_ratio, *profile, xp=np)
    d = d.astype(float, copy=False)
    flat = d.reshape(-1)
    result = np.ones_like(flat)
    result[~np.isfinite(flat)] = np.nan
    # Exact disjoint/fully covered disc limits [radius units]; no quadrature samples are removed.
    result[np.isfinite(flat) & (flat <= radius_ratio-1)] = 0.
    partial = np.flatnonzero((flat < 1+radius_ratio) & (flat > radius_ratio-1))
    for start in range(0, len(partial), 32):
        indices = partial[start:start+32]
        result[indices] -= occulted_fraction(flat[indices], radius_ratio, mu, intensity)
    return result.reshape(d.shape)
