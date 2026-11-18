"""Scenario mass/radius/temperature splines, not calibrated stellar posteriors.

The exact cool-dwarf node vector is unpublished in the retained source.
Both radius and temperature may jump at the 0.63-Msun branch boundary.
"""

from functools import lru_cache
import math
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, PPoly


# Torres (2010) + Chabrier CDWRF mass-radius-Teff splines

_Mass_nodes_Torres = np.array(
    [0.26, 0.47, 0.59, 0.69, 0.87, 0.98, 1.085, 1.4, 1.65, 2.0, 2.5, 3.0, 4.4, 15.0, 40.0],
    dtype=float,
)
_Teff_nodes_Torres = np.array(
    [3170, 3520, 3840, 4410, 5150, 5560, 5940, 6650, 7300, 8180, 9790, 11400, 15200, 30000, 42000],
    dtype=float,
)
_Rad_nodes_Torres = np.array(
    [0.28, 0.47, 0.60, 0.72, 0.9, 1.05, 1.2, 1.55, 1.8, 2.1, 2.4, 2.6, 3.0, 6.2, 11.0],
    dtype=float,
)
_Teff_spline_Torres = InterpolatedUnivariateSpline(_Mass_nodes_Torres, _Teff_nodes_Torres)
_Rad_spline_Torres = InterpolatedUnivariateSpline(_Mass_nodes_Torres, _Rad_nodes_Torres)
_Rad_poly_Torres = PPoly.from_spline(_Rad_spline_Torres._eval_args)

_Mass_nodes_cdwrf = np.array([0.1, 0.135, 0.2, 0.35, 0.48, 0.58, 0.63], dtype=float)
_Teff_nodes_cdwrf = np.array([2800, 3000, 3200, 3400, 3600, 3800, 4000], dtype=float)
_Rad_nodes_cdwrf = np.array([0.12, 0.165, 0.23, 0.36, 0.48, 0.585, 0.6], dtype=float)
_Teff_spline_cdwrf = InterpolatedUnivariateSpline(_Mass_nodes_cdwrf, _Teff_nodes_cdwrf)
_Rad_spline_cdwrf = InterpolatedUnivariateSpline(_Mass_nodes_cdwrf, _Rad_nodes_cdwrf)
_Rad_poly_cdwrf = PPoly.from_spline(_Rad_spline_cdwrf._eval_args)
_Teff_poly_cdwrf = PPoly.from_spline(_Teff_spline_cdwrf._eval_args)
_Teff_poly_Torres = PPoly.from_spline(_Teff_spline_Torres._eval_args)
for _poly in (_Rad_poly_cdwrf, _Teff_poly_cdwrf, _Rad_poly_Torres, _Teff_poly_Torres):
    _poly.x.setflags(write=False)
    _poly.c.setflags(write=False)

# Named stellar mass domain boundaries — derived from spline node arrays.
# Both splines are evaluated only within their retained grid extents; evaluating
# outside triggers scipy extrapolation and produces physically unreliable
# radii and temperatures.  stellar_relations() enforces these limits
# with a ValueError (fail-closed), so callers receive an explicit signal
# rather than silently incorrect stellar parameters.
#
# Torres, Andersen & Giménez (2010), A&ARv 18, 67,
# DOI 10.1007/s00159-009-0025-1: grid spans 0.26–40.0 Msun.
# The cool-dwarf nodes are retained as an uncalibrated scenario grid.  The
# source literature establishes the TIC cool-dwarf catalogue and its use of
# literature relations, but does not publish this exact node vector. Until an
# independently calibrated grid is registered, these values must not be presented as
# calibrated stellar characterization (Muirhead et al. 2018, AJ 155, 180,
# DOI 10.3847/1538-3881/aab710).
STELLAR_RELATIONS_CALIBRATION_STATUS = "uncalibrated"

# Minimum mass supported by the retained CDWRF scenario node [Solar masses].
STELLAR_RELATIONS_MASS_MIN_SOLAR: float = float(_Mass_nodes_cdwrf[0])
# Maximum mass supported by the Torres upper node [Solar masses].
STELLAR_RELATIONS_MASS_MAX_SOLAR: float = float(_Mass_nodes_Torres[-1])
# Mass crossover from CDWRF to Torres regime [Solar masses].
STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR: float = float(_Mass_nodes_cdwrf[-1])


def stellar_relations(
    masses: np.ndarray,
    max_radii: Optional[np.ndarray] = None,
    max_teffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate radii and effective temperatures from stellar masses.

    Uses Torres, Andersen & Giménez (2010) for M > 0.63 Msun and the retained
    cool-dwarf scenario grid for M <= 0.63 Msun.  Raises ``ValueError`` for any
    mass outside the combined retained-node domain
    [``STELLAR_RELATIONS_MASS_MIN_SOLAR``, ``STELLAR_RELATIONS_MASS_MAX_SOLAR``]
    (fail-closed; no extrapolation or silent clamp of physically invalid
    stellar masses). Radius and temperature need not be continuous at 0.63 Msun.

    Args:
        masses: Real numeric star masses [Solar masses], shape (N,), possibly empty.
        max_radii: Optional per-star *physical* maximum radii [Solar radii].
            Applied after spline evaluation to honour an externally supplied
            upper constraint (e.g. a host-star radius constraint).  Does not
            substitute for domain validation.
        max_teffs: Optional per-star *physical* maximum effective temperatures
            [K].  Same usage as ``max_radii``.

    Returns:
        (radii, teffs): Radii [Solar radii], Teffs [K].

    Raises:
        ValueError: If any mass is non-finite, non-positive, or outside the
            grid domain [0.1, 40.0] Msun, inputs have the wrong shape/type,
            or supplied upper bounds are not aligned, finite and positive.

    References:
        Torres, Andersen & Giménez (2010), A&ARv 18, 67,
        DOI ``10.1007/s00159-009-0025-1``.
        Cool-dwarf scenario grid: Muirhead et al. (2018), AJ 155, 180,
        DOI ``10.3847/1538-3881/aab710``. The exact
        node vector is not published in the retained source, so this function
        remains ``uncalibrated`` and cannot be used as a stellar posterior.
    """
    masses = validate_stellar_masses(masses)
    radii = stellar_radii_in_domain(masses)
    teffs = np.zeros_like(masses)
    hot = masses > STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR
    cool = ~hot
    if np.any(hot):
        teffs[hot] = _Teff_poly_Torres(masses[hot])
    if np.any(cool):
        teffs[cool] = _Teff_poly_cdwrf(masses[cool])
    for output, bound in ((radii, max_radii), (teffs, max_teffs)):
        if bound is not None:
            bound = np.asarray(bound)
            if (bound.dtype.kind not in "iuf" or bound.shape != masses.shape
                    or not np.isfinite(bound).all() or np.any(bound <= 0)):
                raise ValueError("stellar upper bounds must match masses and be finite positive numbers")
            np.minimum(output, bound, out=output)
    if not np.all(np.isfinite(radii) & (radii > 0.0)):
        raise ValueError("stellar-relation radii must be finite and positive")
    if not np.all(np.isfinite(teffs) & (teffs > 0.0)):
        raise ValueError("stellar-relation effective temperatures must be finite and positive")
    return radii, teffs


def validate_stellar_masses(masses: np.ndarray) -> np.ndarray:
    """Real 1-D masses in solar units; empty batches are valid, extrapolation is not."""
    masses = np.asarray(masses)
    if (masses.dtype.kind not in "iuf" or masses.ndim != 1
            or not np.isfinite(masses).all()):
        raise ValueError("stellar masses must be a finite real numeric 1-D array")
    if np.any((masses < STELLAR_RELATIONS_MASS_MIN_SOLAR) | (masses > STELLAR_RELATIONS_MASS_MAX_SOLAR)):
        raise ValueError("stellar masses outside retained spline-node domain [0.1, 40.0] Msun; no extrapolation")
    return masses.astype(float, copy=False)


def stellar_relation_polynomials() -> tuple:
    """Read-only (lower, upper, radius, Teff) PPoly branches in Msun/Rsun/K.

    Cool dwarfs own 0.63 Msun; the hot branch's lower endpoint is excluded.
    These are the original interpolants, not a new fit or a calibration.
    """
    return (
        (STELLAR_RELATIONS_MASS_MIN_SOLAR, STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR,
         _Rad_poly_cdwrf, _Teff_poly_cdwrf),
        (STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR, STELLAR_RELATIONS_MASS_MAX_SOLAR,
         _Rad_poly_Torres, _Teff_poly_Torres),
    )


def stellar_radii_in_domain(masses: np.ndarray) -> np.ndarray:
    """Validated 1-D masses [Msun] -> radii [Rsun], without evaluating temperatures."""
    masses = validate_stellar_masses(masses)
    radii = np.zeros_like(masses)
    hot = masses > STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR
    cool = ~hot
    # The same retained cubic splines in the piecewise-polynomial form used by the JAX path.
    if np.any(hot):
        radii[hot] = _Rad_poly_Torres(masses[hot])
    if np.any(cool):
        radii[cool] = _Rad_poly_cdwrf(masses[cool])
    return radii


@lru_cache(maxsize=1)
def _radius_mass_table() -> Tuple[np.ndarray, np.ndarray]:
    # Proposal-only resolution, not an exact physical inverse; keep both sides of the branch jump.
    boundary = STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR
    masses = np.union1d(np.geomspace(STELLAR_RELATIONS_MASS_MIN_SOLAR, STELLAR_RELATIONS_MASS_MAX_SOLAR, (1 << 18) + 1),
                        [boundary, np.nextafter(boundary, np.inf)])
    envelope = np.maximum.accumulate(stellar_radii_in_domain(masses))
    envelope.setflags(write=False)
    masses.setflags(write=False)
    return envelope, masses


def stellar_masses_of_radii(radii: np.ndarray) -> np.ndarray:
    """Approximate first-crossing mass [Msun] of the running radius maximum [Rsun].

    Proposal windows only: finite nonnegative radii outside the sampled envelope
    saturate at the mass endpoints. Plateaus use their first crossing; radii in
    the branch jump map to its boundary, not a star of that exact radius.
    This interpolation is not an inverse stellar characterization or posterior.
    """
    radii = np.asarray(radii)
    if radii.dtype.kind not in "iuf" or not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("proposal radii must be finite nonnegative real numbers")
    envelope, masses = _radius_mass_table()
    right = np.searchsorted(envelope, radii, side="left")
    right = np.clip(right, 1, len(envelope)-1)
    left = right-1
    width = envelope[right]-envelope[left]
    fraction = np.clip((radii-envelope[left])/np.where(width > 0, width, 1.), 0., 1.)
    result = masses[left]+fraction*(masses[right]-masses[left])
    return np.where(radii > envelope[-1], masses[-1], result)


@lru_cache(maxsize=4)
def _cubic_spline_interior_extrema(spline, lower: float, upper: float) -> tuple[float, ...]:
    """Return the exact interior extrema of a piecewise-cubic spline on [lower, upper].

    Derivative roots include stationary internal knots. Results are cached for
    the fixed scenario splines; zero-derivative intervals need only their edges.
    """
    polynomial = spline if isinstance(spline, PPoly) else PPoly.from_spline(spline._eval_args)
    roots = polynomial.derivative().roots(extrapolate=False)
    return tuple(sorted(set(float(root) for root in roots if np.isfinite(root) and lower < root < upper)))


def main_sequence_mass_from_teff(teff_k: float) -> float:
    """Invert the retained main-sequence ``Teff(M)`` relation for one star.

    :func:`stellar_relations` maps mass to temperature with the Torres,
    Andersen & Giménez (2010) nodes above
    ``STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR`` and the retained cool-dwarf
    nodes at or below it. This returns the unique mass on that same relation
    whose temperature equals ``teff_k``. A temperature outside the relation, in
    the gap between its two branches, or reached by more than one mass raises
    instead of choosing a value. Like the forward relation it assumes a
    main-sequence star and is scenario input only, never a stellar posterior.
    """
    from scipy.optimize import brentq

    numeric = np.asarray(teff_k)
    if numeric.ndim != 0 or numeric.dtype.kind not in "iuf" or not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("main-sequence inversion requires a finite positive Teff")
    target = float(numeric)
    # The cool-dwarf branch owns its upper boundary mass; the Torres branch
    # starts just above it, matching ``stellar_relations``.
    branches = (
        (_Teff_poly_cdwrf, STELLAR_RELATIONS_MASS_MIN_SOLAR, STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR),
        (_Teff_poly_Torres, np.nextafter(STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR, np.inf),
         STELLAR_RELATIONS_MASS_MAX_SOLAR),
    )
    masses = []
    for spline, lower, upper in branches:
        edges = [lower, *_cubic_spline_interior_extrema(spline, lower, upper), upper]
        for left, right in zip(edges[:-1], edges[1:]):
            residual_left = float(spline(left)) - target
            residual_right = float(spline(right)) - target
            if residual_left == 0.0:
                masses.append(left)
            elif residual_left * residual_right < 0.0:
                # ``spline`` is bound as a default rather than captured: brentq
                # consumes the lambda inside this iteration today, but a future
                # deferral of the solve would silently evaluate every root on
                # the last branch's spline.
                masses.append(
                    float(
                        brentq(
                            lambda mass, spline=spline: float(spline(mass)) - target,
                            left,
                            right,
                            xtol=np.finfo(float).tiny,
                            rtol=4*np.finfo(float).eps,
                        )
                    )
                )
            if residual_right == 0.0 and right == upper:
                masses.append(right)
    unique = sorted(set(masses))
    if not unique:
        raise ValueError(
            "Teff {0!r} K lies outside the retained main-sequence relation".format(target)
        )
    if len(unique) > 1:
        raise ValueError(
            "Teff {0!r} K is reached by more than one mass on the retained relation".format(target)
        )
    mass = unique[0]
    actual = stellar_relations(np.array([mass]))[1][0]
    if not math.isclose(actual, target, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("main-sequence inversion failed its forward temperature check")
    return mass


__all__ = [
    "STELLAR_RELATIONS_CALIBRATION_STATUS", "STELLAR_RELATIONS_MASS_MIN_SOLAR",
    "STELLAR_RELATIONS_MASS_MAX_SOLAR", "STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR",
    "validate_stellar_masses", "stellar_relation_polynomials", "stellar_relations",
    "stellar_radii_in_domain", "stellar_masses_of_radii", "main_sequence_mass_from_teff",
]
