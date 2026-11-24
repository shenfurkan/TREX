"""Interpolate absolute angular photon fields without interpolating ray positions.

Each PHOENIX node has its own spherical-ray mu coordinates. A cell's angular
field is the weighted SUM of its original piecewise-linear profiles, not a
profile evaluated at averaged mu. Integration commutes with this linear sum.
"""

import numpy as np

from .interpolation import _interpolation_cell

TABULATED_MODEL_ID = "phoenix-specint-2011-tess-photon-tabulated-angular-rmethod-v3"
V3_TABULATED_MODEL_ID = "phoenix-specint-gridv3-tess-photon-tabulated-angular-rmethod-v2"
# The rmethod ids replaced the raw-ray ids (tabulated-angular-v2/-v1) when grid
# compilation began storing r-method-anchored intensities: compiled
# rays end at the transit-visible limb, not the spherical model reference
# circle. An older prepared grid carries the old id and is refused on load
# instead of silently mixing radius conventions; it must be re-prepared.


def interpolate_angular(axes, angular, coordinates, *, xp=np):
    """Return (...,8,ray) profiles; axes are (Teff, logg, M/H, A_V).

    Native axes have already passed the owning grid's validation. Absent
    zero-weight corners use harmless increasing mu to keep zero-light integrals
    free of divisions by zero, not to supply a missing atmosphere.

    The two extinction nodes of a cell share identical mu rays (the compiled
    grid stores mu without an A_V axis), so their absolute photon intensities
    are summed with their interpolation weights into one component before any
    quadrature. That is the same linear field as the sixteen-corner sum: no ray
    coordinate is averaged, no component is renormalized and the total photon
    moment is unchanged.
    """
    mu_table, intensity_table = angular
    valid, corners = _interpolation_cell(axes, coordinates, xp=xp)
    absent_mu = xp.linspace(0., 1., mu_table.shape[-1])
    rays, profiles = [], []
    # product((0,1), repeat=4) advances the extinction axis last.
    for (low, weight_low), (high, weight_high) in zip(corners[::2], corners[1::2]):
        mu = mu_table[low[:3]]
        rays_finite = xp.isfinite(mu).all(axis=-1)
        profile, total = None, None
        for address, weight in ((low, weight_low), (high, weight_high)):
            space = intensity_table[address]
            finite = xp.isfinite(space).all(axis=-1)
            valid = valid & ((weight == 0) | finite)
            term = weight[..., None]*xp.where(finite[..., None], space, 0.)
            profile = term if profile is None else profile+term
            total = weight if total is None else total+weight
        valid = valid & ((total == 0) | rays_finite)
        # Absent zero-weight corners contribute exactly zero, never invented light.
        rays.append(xp.where(rays_finite[..., None], mu, absent_mu))
        profiles.append(profile)
    return xp.stack(rays, axis=-2), xp.stack(profiles, axis=-2), valid
