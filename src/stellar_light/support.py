"""Release-cell demand from complete populations and continuous stellar relations."""

from itertools import product

import numpy as np
from numpy.polynomial import Polynomial

from trex.stellar import stellar_relations as funcs
from trex.populations import priors
from trex.populations.mass_ratio_priors import stellar_mass_ratio_floor, stellar_mass_ratio_floors
from trex.foundation.constants import G, Msun, Rsun
from .release import (LOGG_AXIS, METALLICITY_AXIS, RELEASE, TEFF_AXIS,
                      V3_METALLICITY_AXIS, V3_RELEASE)


def release_axes(release, av_axis):
    """Published atmosphere axes plus explicitly declared extinction knots [mag]."""
    av = np.asarray(av_axis)
    if (release not in (RELEASE, V3_RELEASE) or av.dtype.kind not in "iuf" or av.ndim != 1 or not av.size
            or not np.isfinite(av).all() or np.any(av < 0) or np.any(av[1:] <= av[:-1])):
        raise ValueError("requires one supported release and ordered nonnegative A_V knots")
    return TEFF_AXIS, LOGG_AXIS, V3_METALLICITY_AXIS if release == V3_RELEASE else METALLICITY_AXIS, av.astype(float, copy=False)


def positive_weight_nodes(coordinates, axes):
    """Unique physical nodes used by multilinear queries; retain invalid requests.

    Exact axis knots need only that knot, as in interpolate_photons. Extinction
    knots do not require separate spectra. No endpoint clipping or extrapolation.
    """
    points = np.asarray(coordinates, dtype=float).reshape(-1, 4)
    nodes, invalid = set(), []
    for index, point in enumerate(points):
        if any(not np.isfinite(v) or v < a[0] or v > a[-1] for v, a in zip(point, axes)):
            invalid.append(index)
            continue
        brackets = []
        for axis, value in zip(axes[:3], point[:3]):
            high = int(np.searchsorted(axis, value))
            brackets.append([float(axis[high])] if axis[high] == value
                            else [float(axis[high-1]), float(axis[high])])
        nodes.update(product(*brackets))
    return sorted(nodes), invalid


def _real_roots(polynomial, left, right):
    """Real interior roots of a retained spline polynomial in local mass [Msun]."""
    roots = polynomial.roots()
    tolerance = 64*np.finfo(float).eps
    return [float(r.real) for r in roots if abs(r.imag) <= tolerance*max(1., abs(r.real))
            and left < r.real < right]


def _relation_pieces(lower, upper):
    """Traverse every T/logg cell crossed by the unchanged secondary relation.

    Split each exact cubic at every atmosphere-axis crossing, not at chosen
    representative draws. T(m)=T_k is cubic; g=GM/R² gives
    R(m)²-G*Msun*m/(Rsun²*10**logg_k)=0 (degree six). Between consecutive
    roots the interpolation cell is constant. Endpoints and one interior point
    therefore enumerate its corners, including tangencies and the .63-Msun jump.
    The retained scenario relation is uncalibrated (funcs), not a stellar posterior.
    """
    pieces = []
    for branch_lower, branch_upper, rp, tp in funcs.stellar_relation_polynomials():
        lo, hi = max(lower, branch_lower), min(upper, branch_upper)
        if branch_lower == funcs.STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR:
            lo = max(lo, np.nextafter(branch_lower, np.inf))
        if hi <= lo:
            continue
        if not np.array_equal(rp.x, tp.x):
            raise ValueError("stellar radius and temperature pieces must share mass knots")
        for i, (left, right) in enumerate(zip(rp.x[:-1], rp.x[1:])):
            a, b = max(lo, left), min(hi, right)
            if b <= a:
                continue
            r = Polynomial(rp.c[:, i][::-1])
            t = Polynomial(tp.c[:, i][::-1])
            mass = Polynomial([left, 1.])
            cuts = [a-left, b-left]
            for knot in TEFF_AXIS:
                cuts.extend(_real_roots(t-knot, a-left, b-left))
            for knot in LOGG_AXIS:
                # CGS gravity crossing; same IAU/CODATA constants as the component producer.
                cuts.extend(_real_roots(r*r-G*Msun/Rsun**2/10**knot*mass, a-left, b-left))
            cuts = np.unique(cuts)
            pieces.append((left, r, t, cuts))
    return pieces


def _relation_cells(lower, upper, pieces):
    """Restrict precomputed exact spline/cell crossings to one full mass interval."""
    points = []
    for origin, radius, temperature, boundaries in pieces:
        left, right = max(lower-origin, boundaries[0]), min(upper-origin, boundaries[-1])
        if right <= left:
            continue
        cuts = np.r_[left, boundaries[(boundaries > left) & (boundaries < right)], right]
        samples = np.r_[cuts, (cuts[:-1]+cuts[1:])/2]
        radii, teffs = radius(samples), temperature(samples)
        if np.any(radii <= 0) or np.any(teffs <= 0):
            raise ValueError("secondary relation has nonpositive radius/temperature in its support")
        gravity = np.log10(G*Msun*(samples+origin)/(Rsun*radii)**2)
        points.extend(zip(teffs, gravity))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def population_atmosphere_demand(population, av_axis, *, release=V3_RELEASE, bound_host_indices=()):
    """Plan all primary cells AND full EB/twin secondary-prior cells, without cuts.

    Population fields are the actual consumed teff_k, mass_solar, radius_solar,
    mh_dex and av_mag arrays. All rows and failures remain indexed. Secondary
    q support is the union of the unchanged EB and twin branches, not a sampled
    set of q values. Caller may also use a row for a fixed target or neighbour.
    Explicit ``bound_host_indices`` identify fixed hosts that ALSO generate a
    continuous bound-companion population. Its whole companion interval and
    each possible companion's EB secondaries are included, with the SAME target
    composition/extinction assumption as the existing consumers. No imaging or
    orbital-geometry acceptance is inferred from atmosphere support.
    """
    axes = release_axes(release, av_axis)
    fields = ("teff_k", "mass_solar", "radius_solar", "mh_dex", "av_mag")
    if not all(k in population for k in fields):
        raise ValueError("atmosphere demand requires temperature, mass, radius, metallicity and extinction rows")
    arrays = [np.asarray(population[k]) for k in fields]
    if any(a.dtype.kind not in "iuf" or a.ndim != 1 or a.shape != arrays[0].shape
           or not np.isfinite(a).all() for a in arrays):
        raise ValueError("atmosphere demand needs every finite, aligned population row")
    if any(np.any(a <= 0) for a in arrays[:3]) or np.any(arrays[-1] < 0):
        raise ValueError("population temperatures/masses/radii must be positive and A_V nonnegative")
    teff, mass, radius, mh, av = arrays
    try:
        bound_hosts = set(bound_host_indices)
    except TypeError as exc:
        raise ValueError("bound host indices must be an iterable of retained row indices") from exc
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 <= i < len(mass) for i in bound_hosts):
        raise ValueError("bound host indices must refer to retained population rows")
    gravity = np.log10(G*Msun*mass/(Rsun*radius)**2)
    supported = ((mass >= priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
                 & (mass <= funcs.STELLAR_RELATIONS_MASS_MAX_SOLAR))
    # Axis crossings depend on the SAME retained splines, not population composition or host row.
    floors = stellar_mass_ratio_floors(mass[supported])*mass[supported]
    bound_lows = []
    for i in bound_hosts:
        if supported[i]:
            companion_low = stellar_mass_ratio_floor(mass[i])*mass[i]
            secondary_host_low = max(companion_low, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
            bound_lows.append(min(companion_low, stellar_mass_ratio_floor(secondary_host_low)*secondary_host_low))
    pieces = (_relation_pieces(float(np.min(np.r_[floors, bound_lows])),
                               float(np.max(mass[supported]))) if np.any(supported) else [])
    rows, all_nodes, paths = [], set(), {}
    for index, values in enumerate(zip(teff, gravity, mass, mh, av)):
        t, g, m, z, extinction = values
        primary = [[t, g, z, extinction]]
        nodes, invalid = positive_weight_nodes(primary, axes)
        row = dict(index=index, primary_coordinates=primary[0], primary_nodes=nodes,
                   primary_out_of_domain=bool(invalid), secondary_nodes=[], secondary_out_of_domain=[])
        all_nodes.update(nodes)
        if m < priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR:
            row["secondary_prior_status"] = "no-stellar-secondary-prior-support"
        else:
            low = stellar_mass_ratio_floor(m)*m
            row["secondary_mass_interval_solar"] = [float(low), float(m)]
            if low < funcs.STELLAR_RELATIONS_MASS_MIN_SOLAR or m > funcs.STELLAR_RELATIONS_MASS_MAX_SOLAR:
                row["secondary_prior_status"] = "outside-retained-stellar-relation"
            else:
                if m not in paths:
                    paths[m] = _relation_cells(low, m, pieces)
                path = paths[m]
                queries = np.column_stack([path, np.full(len(path), z), np.full(len(path), extinction)])
                nodes, invalid = positive_weight_nodes(queries, axes)
                row.update(secondary_nodes=nodes, secondary_out_of_domain=queries[invalid].tolist(),
                           secondary_prior_status="full-EB-and-twin-mass-support")
                all_nodes.update(nodes)
                if index in bound_hosts:
                    secondary_host_low = max(low, priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR)
                    nested_low = min(low, stellar_mass_ratio_floor(secondary_host_low)*secondary_host_low)
                    path = _relation_cells(nested_low, m, pieces)
                    queries = np.column_stack([path, np.full(len(path), z), np.full(len(path), extinction)])
                    nodes, invalid = positive_weight_nodes(queries, axes)
                    row.update(bound_companion_mass_interval_solar=[float(low), float(m)],
                               bound_and_nested_secondary_mass_union_solar=[float(nested_low), float(m)],
                               bound_nodes=nodes, bound_out_of_domain=queries[invalid].tolist())
                    all_nodes.update(nodes)
        rows.append(row)
    return dict(release=release, av_axis=axes[3].tolist(), required_nodes=sorted(all_nodes), rows=rows,
                stellar_relations_calibration_status=funcs.STELLAR_RELATIONS_CALIBRATION_STATUS,
                continuous_support="exact retained spline pieces split at every T/logg axis crossing",
                limits="atmosphere availability only; no population/prior pruning or scientific acceptance")
