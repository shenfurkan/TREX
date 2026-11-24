"""Independent numerical checks for the opt-in analytic quadratic kernel.

The reference is not the implementation in disguise: it integrates the
Mandel & Agol (2002) radiative law

    I(r) = (1 - u1 - u2) + (u1 + 2 u2) sqrt(1 - r^2) - u2 (1 - r^2)

directly over the occulted overlap as a two-dimensional area integral,
normalised by the star's total flux ``pi * Omega`` with
``Omega = 1 - u1/3 - u2/6``.  No elliptic integral, table row or closed form
of the implementation is reused, so a shared algebraic defect cannot pass.

At stellar radius ``r`` the ring element subtends
``2 arccos((r^2 + d^2 - p^2)/(2 r d))`` inside the occultor, so each area
integral is one radial quadrature.  The three transition radii ``|d - p|``,
``d + p`` and the limb ``r = 1`` are split, and every subinterval is mapped by
``r = a + (b - a) sin^2(pi v / 2)``; that map turns the square-root behaviour
at both ends into a power series, so the transformed integrand is analytic on
each closed subinterval and the Gauss-Legendre error falls far below double
precision well before the 128 nodes used here (the module's own 128-vs-256
comparison asserts that, rather than assuming it).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
scipy_special = pytest.importorskip("scipy.special")

from trex.accelerators._analytic_limb_dark import (  # noqa: E402
    _cel_with_steps,
    cel,
    quadratic_disc_flux,
)

# 128 nodes on a post-substitution analytic integrand; the 128-vs-256 check
# below measures the reference's own error before this tolerance is used.
_QUADRATURE_ORDER = 128
# The closed form's terms are O(1) and cancel to ~1e-13 near grazing
# configurations, while the reference is accurate to ~1e-14; 1e-10 is still a
# relative 1e-5 of the shallowest tested transit and four orders tighter than
# any formula-level error (a missing factor or sign moves the flux by >= 1e-3).
_FLUX_TOLERANCE = 1.0e-10

RADIUS_RATIOS = (0.01, 0.1, 0.5, 0.8, 1.0, 1.4)
COEFFICIENT_SETS = ((0.0, 0.0), (0.2, 0.3))


def _overlap_integrals(distance, radius, order):
    """(area, sqrt(1-r^2), 1-r^2) integrals over the overlap, in units of pi.

    ``distance`` and ``radius`` are in stellar radii; ``distance`` must be
    positive. Each return value is ``(1/pi) * integral over the overlap`` of
    the named weight, so the full disk gives ``(1, 2/3, 1/2)``.
    """
    if distance == 0.0:
        covered = min(radius, 1.0)
        area = covered * covered
        root = (2.0 / 3.0) * (1.0 - (1.0 - covered * covered) ** 1.5)
        square = covered * covered - 0.5 * covered**4
        return area, root, square

    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (nodes + 1.0)
    base = 0.5 * weights
    edges = sorted(
        {0.0, min(abs(distance - radius), 1.0), min(distance + radius, 1.0), 1.0}
    )
    totals = np.zeros(3)
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        angle = np.pi * unit
        radii = left + (right - left) * np.sin(0.5 * angle) ** 2
        jacobian = (right - left) * 0.5 * np.pi * np.sin(angle)
        cosine = (radii * radii + distance * distance - radius * radius) / (
            2.0 * radii * distance
        )
        azimuth = 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))
        # The covered ring element is r dr dphi over the arc ``azimuth``; the
        # factor is r, not 2r. The full disk check below (area -> 1) pins it.
        weight = base * jacobian * radii * azimuth
        totals[0] += np.sum(weight)
        totals[1] += np.sum(weight * np.sqrt(np.maximum(1.0 - radii * radii, 0.0)))
        totals[2] += np.sum(weight * (1.0 - radii * radii))
    return tuple(totals / np.pi)


def _reference_delta(u1, u2, distance, radius, order=_QUADRATURE_ORDER):
    """Blocked flux ``F - 1`` from the direct area integral of the intensity law."""
    area, root, square = _overlap_integrals(distance, radius, order)
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    return -(
        (1.0 - u1 - u2) * area + (u1 + 2.0 * u2) * root - u2 * square
    ) / omega


def _cel_definition_quadrature(kc, p, a, b, order=256):
    """Bulirsch's defining integral by Gauss-Legendre, analytic for p > 0."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = 0.25 * np.pi * (nodes + 1.0)
    cosine2 = np.cos(theta) ** 2
    sine2 = np.sin(theta) ** 2
    integrand = (a * cosine2 + b * sine2) / (
        (cosine2 + p * sine2) * np.sqrt(cosine2 + kc * kc * sine2)
    )
    return np.sum(0.25 * np.pi * weights * integrand)


def _geometry_cases():
    """Radius ratios with the concentric, interior, tangent, mid-limb and outer rows."""
    cases = []
    for radius_ratio in RADIUS_RATIOS:
        edge = abs(1.0 - radius_ratio)
        separations = [
            0.0,
            0.5 * radius_ratio,
            edge,
            0.5 * (edge + 1.0 + radius_ratio),
            1.0,
            1.0 + radius_ratio,
        ]
        if radius_ratio < 1.0:
            # Both lambda2 subregions: 0.5 (1 - p) lies strictly inside the
            # stellar disk, and 0.5 lies between p and 1 - p for p < 1/2
            # (case III) while 0.5 p < p covers case IX.
            separations.extend((0.5 * (1.0 - radius_ratio), 0.5))
        cases.extend(
            (radius_ratio, separation) for separation in sorted(set(separations))
        )
    return cases


def test_reference_quadrature_converges():
    """The reference itself: 128 and 256 nodes must agree far below the tolerance."""
    worst = 0.0
    for distance, radius in (
        (0.3, 0.1),
        (0.9, 0.1),
        (0.95, 0.1),
        (1.0, 0.5),
        (1.2, 0.5),
        (1.0, 1.4),
        (2.0, 1.4),
        (2.4, 2.5),
    ):
        coarse = np.array(_overlap_integrals(distance, radius, _QUADRATURE_ORDER))
        fine = np.array(_overlap_integrals(distance, radius, 2 * _QUADRATURE_ORDER))
        worst = max(worst, float(np.max(np.abs(coarse - fine))))
    assert worst < 1.0e-13


def test_cel_matches_scipy_complete_elliptic_integrals():
    """cel(kc, 1, 1, 1) = K(k) and cel(kc, 1, 1, kc^2) = E(k), k^2 = 1 - kc^2.

    The parameter is rounded once to float64 and ``kc`` is recovered from it, so
    the comparison is not limited by the ``1 - kc^2`` round trip; over
    ``kc`` in [1e-3, 1] the AGM gap bound of 1e-16 leaves rtol 1e-13 as a
    measured-in-test tolerance with two orders of margin.
    """
    for target in np.logspace(-3.0, 0.0, 25):
        parameter = 1.0 - target * target
        kc = float(np.sqrt(1.0 - parameter))
        assert float(cel(kc, 1.0, 1.0, 1.0)) == pytest.approx(
            float(scipy_special.ellipk(parameter)), rel=1.0e-13
        )
        assert float(cel(kc, 1.0, 1.0, kc * kc)) == pytest.approx(
            float(scipy_special.ellipe(parameter)), rel=1.0e-13
        )


@pytest.mark.parametrize(
    ("kc", "p", "a", "b"),
    (
        (0.9, 1.0, 1.0, 0.81),
        (0.6, 2.0, 1.0, 1.0),
        (0.75, 0.7, 1.0, 1.0),
        (0.4, 5.0, 1.0, 1.0),
        (0.99, 3.0, 1.0, 0.5),
    ),
)
def test_cel_general_definition_matches_quadrature(kc, p, a, b):
    """The general (p != 1, a != b) branch against its defining integral.

    The integrand is analytic on the closed interval for p > 0, so 256-node
    Gauss-Legendre is accurate to ~1e-15 relative and rel 1e-12 is safe.
    """
    assert float(cel(kc, p, a, b)) == pytest.approx(
        _cel_definition_quadrature(kc, p, a, b), rel=1.0e-12
    )


def test_cel_fixed_step_budget_is_self_convergent():
    """18 AGM steps reach the same stopping point as 36 over the kc domain.

    Both runs keep the first step that meets the gap criterion, so a different
    answer means the 18-step budget stopped before convergence (or returned
    NaN); rtol 1e-14 is the stated measured-in-test tolerance.
    """
    for target in np.logspace(-8.0, 0.0, 33):
        short = float(_cel_with_steps(target, 1.0, 1.0, 1.0, 18))
        long = float(_cel_with_steps(target, 1.0, 1.0, 1.0, 36))
        assert np.isfinite(short)
        assert short == pytest.approx(long, rel=1.0e-14)


@pytest.mark.parametrize(
    ("kc", "p", "a", "b"),
    (
        (0.5, 0.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (-0.5, 1.0, 1.0, 1.0),
        (1.5, 1.0, 1.0, 1.0),
        (np.nan, 1.0, 1.0, 1.0),
        (0.5, np.inf, 1.0, 1.0),
        (0.5, 1.0, np.nan, 1.0),
        (0.5, 1.0, 1.0, np.inf),
    ),
)
def test_cel_reports_undefined_inputs_as_nan(kc, p, a, b):
    """p == 0 is the logarithmic singularity and must never be silently finite."""
    assert np.isnan(float(cel(kc, p, a, b)))


@pytest.mark.parametrize(("radius_ratio", "separation"), _geometry_cases())
@pytest.mark.parametrize(("u1", "u2"), COEFFICIENT_SETS)
def test_quadratic_flux_matches_area_quadrature(u1, u2, radius_ratio, separation):
    """Interior, tangent, mid-limb and grazing rows against the direct integral."""
    delta, valid = quadratic_disc_flux(u1, u2, separation, radius_ratio)
    assert bool(valid)
    assert np.isfinite(float(delta))
    expected = _reference_delta(u1, u2, separation, radius_ratio)
    assert float(delta) == pytest.approx(expected, abs=_FLUX_TOLERANCE, rel=0)


@pytest.mark.parametrize("radius_ratio", (.01, .1, .5, .8))
def test_contained_uniform_disc_has_exact_occultor_area_eager_and_jit(radius_ratio):
    import jax

    for evaluate in (quadratic_disc_flux, jax.jit(quadratic_disc_flux)):
        for separation in (0., .5 * (1. - radius_ratio), 1. - radius_ratio):
            delta, valid = evaluate(0., 0., separation, radius_ratio)
            assert bool(valid)
            assert float(delta) == -(radius_ratio * radius_ratio)


@pytest.mark.parametrize("radius_ratio", (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.5, 2.5))
def test_tangent_rows_match_area_quadrature(radius_ratio):
    """z = p rows (lambda4, the case VI constant and lambda3) are exact closed forms."""
    delta, valid = quadratic_disc_flux(0.2, 0.3, radius_ratio, radius_ratio)
    assert bool(valid)
    assert np.isfinite(float(delta))
    expected = _reference_delta(0.2, 0.3, radius_ratio, radius_ratio)
    assert float(delta) == pytest.approx(expected, abs=_FLUX_TOLERANCE, rel=0)


@pytest.mark.parametrize("radius_ratio", (0.1, 0.5))
def test_near_grazing_external_contact_matches_area_quadrature(radius_ratio):
    """Just inside external contact, where the elliptic rows cancel O(1) terms.

    The exact ``z = 1 + p`` row is short-circuited, so it cannot expose a wrong
    elliptic branch. At ``1e-4`` inside it the deficit is ~1e-6 and the
    reference self-converges below 1e-13; ``1e-8`` still rejects a missing
    factor or modulus (which moves the flux by the full ~1e-6).
    """
    separation = 1.0 + radius_ratio - 1.0e-4
    delta, valid = quadratic_disc_flux(0.2, 0.3, separation, radius_ratio)
    assert bool(valid)
    assert np.isfinite(float(delta))
    expected = _reference_delta(0.2, 0.3, separation, radius_ratio)
    assert float(delta) == pytest.approx(expected, abs=1e-8, rel=0)


@pytest.mark.parametrize("radius_ratio", RADIUS_RATIOS)
@pytest.mark.parametrize(("u1", "u2"), COEFFICIENT_SETS)
def test_exact_geometric_limits_are_exact(u1, u2, radius_ratio):
    """No-overlap and complete-coverage rows are exact, not elliptic-limit values."""
    delta, valid = quadratic_disc_flux(u1, u2, 1.0 + radius_ratio, radius_ratio)
    assert bool(valid)
    assert float(delta) == 0.0

    if radius_ratio >= 1.0:
        for separation in (0.0, radius_ratio - 1.0):
            delta, valid = quadratic_disc_flux(u1, u2, separation, radius_ratio)
            assert bool(valid)
            assert float(delta) == -1.0


@pytest.mark.parametrize(
    ("u1", "u2", "separation", "radius_ratio"),
    (
        (0.2, 0.3, 0.5, 0.0),
        (0.2, 0.3, 0.5, -0.5),
        (0.2, 0.3, np.nan, 0.5),
        (0.2, 0.3, np.inf, 0.5),
        (0.2, 0.3, -np.inf, 0.5),
        (np.nan, 0.3, 0.5, 0.5),
        (0.2, np.inf, 0.5, 0.5),
        (0.2, 0.3, 0.5, np.nan),
        (0.2, 0.3, 0.5, np.inf),
    ),
)
def test_invalid_inputs_are_flagged_and_nan(u1, u2, separation, radius_ratio):
    delta, valid = quadratic_disc_flux(u1, u2, separation, radius_ratio)
    assert not bool(valid)
    assert np.isnan(float(delta))


def test_geometric_zero_support_stays_valid():
    """A finite far separation is a valid exact zero, distinct from invalid input."""
    delta, valid = quadratic_disc_flux(0.2, 0.3, 1.0e6, 0.5)
    assert bool(valid)
    assert float(delta) == 0.0


def test_eclipsing_binary_secondary_with_larger_occultor():
    """radius_ratio > 1: the host is the occulted body in a secondary eclipse.

    ``p = 2.5`` exercises full coverage (z <= p - 1 = 1.5), the limb branch
    (1.5 < z < 3.5), the z = p tangent row lambda3 and the outer zero.
    """
    radius_ratio = 2.5
    for separation in (0.0, 1.4, 1.6, 2.0, 2.5, 3.0, 3.5):
        delta, valid = quadratic_disc_flux(0.2, 0.3, separation, radius_ratio)
        assert bool(valid)
        assert np.isfinite(float(delta))
        if separation <= radius_ratio - 1.0:
            assert float(delta) == -1.0
        elif separation >= 1.0 + radius_ratio:
            assert float(delta) == 0.0
        else:
            expected = _reference_delta(0.2, 0.3, separation, radius_ratio)
            assert float(delta) == pytest.approx(expected, abs=_FLUX_TOLERANCE, rel=0)


def test_zero_disc_normalization_is_invalid_not_division_by_zero():
    """Omega = 1 - u1/3 - u2/6 == 0 leaves no flux to normalize by."""
    for separation in (0.0, 0.3, 2.0):
        delta, valid = quadratic_disc_flux(3.0, 0.0, separation, 0.1)
        assert not bool(valid)
        assert np.isnan(float(delta))


@pytest.mark.parametrize("radius_ratio", (0.1, 0.3, 0.5, 0.7, 0.9))
def test_one_ulp_around_internal_tangent_stays_finite_and_continuous(radius_ratio):
    """Within float64 steps of z = 1 - p both elliptic moduli round to zero.

    The branch choice must follow ``(z + p)^2`` so neither NaN branch is
    selected; the flux is continuous there, so the tangent-row quadrature
    value bounds every neighbouring float.
    """
    tangent = 1.0 - radius_ratio
    expected = _reference_delta(0.2, 0.3, tangent, radius_ratio)
    separations = [tangent]
    for _ in range(4):
        separations.append(np.nextafter(separations[-1], 2.0))
    lower = tangent
    for _ in range(4):
        lower = np.nextafter(lower, -1.0)
        separations.append(lower)
    for separation in separations:
        delta, valid = quadratic_disc_flux(0.2, 0.3, separation, radius_ratio)
        assert bool(valid)
        assert np.isfinite(float(delta))
        # The neighbouring closed forms cancel log-divergent terms here, so
        # their accuracy degrades towards the row (see the kernel docstring);
        # 1e-6 is still two orders below the best TESS per-point precision
        # and far below the O(p^2) jump a NaN-masked or dropped point implies.
        assert float(delta) == pytest.approx(expected, abs=1.0e-6, rel=0)


@pytest.mark.parametrize("radius_ratio", (0.05, 0.5, 0.99, 1.0, 2.5))
def test_concentric_and_near_concentric_never_return_valid_nan(radius_ratio):
    for separation in (0.0, 1.0e-300, 1.0e-17, 1.0e-12):
        delta, valid = quadratic_disc_flux(0.2, 0.3, separation, radius_ratio)
        assert bool(valid)
        assert np.isfinite(float(delta))
        if radius_ratio >= 1.0 and separation < 1.0e-15:
            # (z + p)^2 rounds to 1 for p = 1, i.e. full coverage within rounding.
            assert float(delta) == -1.0
