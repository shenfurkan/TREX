"""Independent polar-disc quadrature for TREX's synthetic EB light curves.

The expected flux in these fixtures is integrated here, not by TREX: a
quadratic limb-darkened disc is occulted on a polar grid, so a TREX forward
model that reuses one body's coefficients for the other cannot match it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

_TAU = 2.0 * math.pi


def covered_arc_radians(radius: float, separation: float, planet_radius: float) -> float:
    """Return the occulted angular arc of one stellar ring of the given radius."""
    if separation == 0.0:
        return 2.0 * math.pi if radius <= planet_radius else 0.0
    if separation + radius <= planet_radius:
        return 2.0 * math.pi
    if separation >= radius + planet_radius or radius >= separation + planet_radius:
        return 0.0
    cosine = (radius**2 + separation**2 - planet_radius**2) / (2.0 * radius * separation)
    return 2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0)))


def covered_angular_integral(arc_radians: float, nodes: np.ndarray, weights: np.ndarray) -> float:
    """Integrate one analytically bounded occulted angular interval."""
    if arc_radians == 0.0:
        return 0.0
    angles = 0.5 * arc_radians * nodes
    integrand = np.ones_like(angles)
    return 0.5 * arc_radians * float(np.sum(weights * integrand))


def instantaneous_flux(
    separation: float, radius_ratio: float, q1: float, q2: float, radial_nodes: int, angular_nodes: int
) -> float:
    """Relative flux of a quadratic-limb-darkened disc occulted at sky separation ``separation``."""
    root_q1 = math.sqrt(q1)
    u1 = 2.0 * root_q1 * q2
    u2 = root_q1 * (1.0 - 2.0 * q2)
    breaks = [0.0, 1.0]
    for boundary in (abs(separation - radius_ratio), separation + radius_ratio):
        if 0.0 < boundary < 1.0:
            breaks.append(boundary)
    breaks = sorted(set(breaks))
    nodes, weights = np.polynomial.legendre.leggauss(radial_nodes)
    angular, angular_weights = np.polynomial.legendre.leggauss(angular_nodes)
    blocked = 0.0
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        radii = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        mu = np.sqrt(1.0 - radii**2)
        intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        arcs = np.asarray(
            [covered_angular_integral(covered_arc_radians(float(r), separation, radius_ratio), angular, angular_weights) for r in radii]
        )
        blocked += 0.5 * (upper - lower) * float(np.sum(weights * intensity * arcs * radii))
    total = math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    return 1.0 - blocked / total


def sky_separation(phase_days: float, period_days: float, a_rs: float, impact_parameter: float) -> float:
    """Circular-orbit projected separation in stellar radii (Winn 2010, eqs. 1-7)."""
    inclination = math.acos(impact_parameter / a_rs)
    angle = _TAU * phase_days / period_days
    return math.hypot(a_rs * math.sin(angle), a_rs * math.cos(inclination) * math.cos(angle))


@lru_cache(maxsize=32)
def exposure_averaged_flux(
    phase_days: Tuple[float, ...],
    period_days: float,
    radius_ratio: float,
    a_rs: float,
    impact_parameter: float,
    q1: float,
    q2: float,
    exposure_seconds: float,
    subsamples: int = 31,
    radial_nodes: int = 96,
    angular_nodes: int = 24,
) -> np.ndarray:
    """Exposure-averaged model at each phase (days from mid-transit).

    Sub-exposures whose separation exceeds ``1 + k`` or that lie on the far
    side of the orbit contribute exactly unit flux, so the quadrature is only
    evaluated where the discs overlap.
    """
    exposure_days = exposure_seconds / 86400.0
    offsets = ((np.arange(subsamples, dtype=float) + 0.5) / subsamples - 0.5) * exposure_days
    flux = np.ones(len(phase_days), dtype=float)
    reach = 1.0 + radius_ratio
    for index, phase in enumerate(phase_days):
        values = []
        for offset in offsets:
            local = phase + offset
            if math.cos(_TAU * local / period_days) <= 0.0:
                values.append(1.0)
                continue
            separation = sky_separation(local, period_days, a_rs, impact_parameter)
            values.append(
                1.0 if separation >= reach else instantaneous_flux(separation, radius_ratio, q1, q2, radial_nodes, angular_nodes)
            )
        flux[index] = float(np.mean(values))
    return flux
