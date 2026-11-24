"""Read the official MIST v1.2 grid without extracting its archive.

The released ``Gaia_G_EDR3`` column is an intrinsic magnitude at zero
extinction, so the G/parallax likelihood converts the retained field A_V
envelope to A_G with the official Gaia (E)DR3 k_G(Teff,A_V) coefficient
rather than subtracting A_V from G.  The grid reader enforces the release's
own ``CCM89 extinction: Av = 0.000`` declaration before that column is used.
"""

import math
import tarfile

import numpy as np
from scipy.special import logsumexp

from .host_distributions import positive, radius

MIST_COLUMNS = (1, 2, 3, 4, 5, 7, 30)
MIST_COLUMN_NAMES = ("log10_isochrone_age_yr", "initial_mass", "star_mass", "log_Teff", "log_g",
                     "[Fe/H]_init", "Gaia_G_EDR3")
# Positions in a ``read_mist_grid`` row: the MIST_COLUMNS values, then the cell weight.
LOG_AGE, INITIAL_MASS, STAR_MASS, LOG_TEFF, LOG_G, FEH_INIT, GAIA_G, CELL_WEIGHT = range(8)
# One grid (a few hundred MB) stays in memory; a different archive replaces it.
_MIST_CACHE = {}
_ZERO_POINT_TABLES_LOADED = False

# Gaia (E)DR3 G-band extinction-law coefficients for k_G = A_G/A_0 with
# X = Teff/5040 K.  The rows are "TeffNorm", "kG" from the DPAC
# Fitz19_EDR3_MainSequence.csv and Fitz19_EDR3_HRDTop.csv tables
# (https://www.cosmos.esa.int/web/gaia/edr3-extinction-law, retrieved
# 2026-11-14; Danielski et al. 2018, A&A 614, A19, DOI
# 10.1051/0004-6361/201732327; Riello et al. 2021, A&A 649, A3 passbands).
# Order is the published formula
#   a1 + a2 X + a3 X^2 + a4 X^3 + a5 A0 + a6 A0^2 + a7 A0^3
#      + a8 A0 X + a9 A0 X^2 + a10 X A0^2.
# The fit is for solar metallicity, 3500 < Teff < 10000 K and
# 0.01 < A0 < 20 mag, with a maximum 3.5% residual in G; the source states
# that extinction-law and reference-spectra systematics dominate.
_EDR3_KG_MAIN_SEQUENCE = (
    0.259021858973784, 0.93676111876298, -0.437446499585492, 0.0783508476444952,
    -0.00136127433235224, 0.000797222688660053, -2.52364537395233e-05,
    -0.029981504413869, 0.00555506017434798, 0.000402778419603886,
)
_EDR3_KG_HRD_TOP = (
    0.235975767531247, 0.974642001154732, -0.454849105677279, 0.0809472883012553,
    0.00143267173083805, 0.000711522982426832, -2.43558666819971e-05,
    -0.0330300302504699, 0.00626996185287723, 0.000448105647489681,
)
_EDR3_TEFF_MIN_K = 3500.0
_EDR3_TEFF_MAX_K = 10000.0


def _edr3_kg_polynomial(coefficients, x, a_v):
    """Evaluate one published Gaia (E)DR3 k_G polynomial."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = coefficients
    return (a1 + a2 * x + a3 * x**2 + a4 * x**3 + a5 * a_v + a6 * a_v**2
            + a7 * a_v**3 + a8 * a_v * x + a9 * a_v * x**2 + a10 * x * a_v**2)


def _gaia_edr3_k_g(teff_k, log_g, a_v):
    """Evaluate the published k_G=A_G/A_0 fit at the retained A_V envelope coordinate.

    ``teff_k`` is effective temperature in K, ``log_g`` is log10(g / cm s^-2)
    and ``a_v`` is the retained V-band attenuation in mag. Substituting A_V
    for the fit's monochromatic A_0 at 550 nm is an explicit approximation.
    The main-sequence
    fit assumes log g=4.5; the HRD-top fit assumes log g=4 above 5250 K and
    log g=-8.3+0.0023 Teff below it.  The fit whose assumed gravity is closer
    to the node's gravity is used.  The published domain is 3500-10000 K and
    0.01<A_0<20 mag; outside the temperature range the fit is evaluated at the
    nearest boundary so an unbounded polynomial extrapolation never enters the
    likelihood.  A_V=0 gives A_G=0 exactly at the use site.  The likelihood
    envelope includes A_V<0.01 mag, where the same polynomial is evaluated
    continuously as an uncalibrated low-extinction continuation. The
    coefficient is a model relation, not a measured per-star extinction
    uncertainty.
    """
    teff_k = np.clip(np.asarray(teff_k, dtype=float), _EDR3_TEFF_MIN_K, _EDR3_TEFF_MAX_K)
    log_g = np.asarray(log_g, dtype=float)
    a_v = np.asarray(a_v, dtype=float)
    x = teff_k / 5040.0
    hrd_top_log_g = np.where(teff_k >= 5250.0, 4.0, -8.3 + 0.0023 * teff_k)
    main_sequence = _edr3_kg_polynomial(_EDR3_KG_MAIN_SEQUENCE, x, a_v)
    hrd_top = _edr3_kg_polynomial(_EDR3_KG_HRD_TOP, x, a_v)
    return np.where(np.abs(log_g - 4.5) <= np.abs(log_g - hrd_top_log_g),
                    main_sequence, hrd_top)


def read_mist_grid(archive):
    """Read the actual v1.2 nonrotating release, with mass/age cell weights.

    Salpeter dN/dM proportional to M^-2.35 (Montet et al. Sec. 3.2), uniform
    linear age and [Fe/H]. EEP nodes are NOT given equal evolutionary weights.
    Mass cells are per (age, [Fe/H]) isochrone; age and [Fe/H] cell widths are
    global over the archive's unique admitted coordinates. Mass cells and repeated
    EEP counts also span all members, so splitting an isochrone between members
    keeps the same cell weights. No file from the
    untrusted archive is extracted or executed. The release's own ``CCM89
    extinction: Av = 0.000`` declaration is required, so a reddened isochrone
    is never treated as intrinsic and corrected twice.
    Raises ``ValueError`` for a header without the adopted columns, a
    nonzero/missing extinction declaration, non-finite or non-positive masses,
    fewer than two admitted ages or [Fe/H] values across the archive, or fewer
    than two masses for one isochrone (a cell width needs two nodes).
    """
    from astropy.cosmology import Planck18
    key = (str(archive.resolve()), archive.stat().st_mtime_ns, archive.stat().st_size)
    if key in _MIST_CACHE:
        return _MIST_CACHE[key]
    maximum_age = Planck18.age(0).to_value("yr")
    chunks = []
    with tarfile.open(archive, "r|xz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".iso.cmd"):
                continue
            stream = handle.extractfile(member)
            extinction_zero = False
            for line in stream:
                if line.startswith(b"# CCM89 extinction:"):
                    try:
                        extinction = float(line.decode("ascii").split("=")[-1])
                    except ValueError as exc:
                        raise ValueError("MIST release extinction declaration is malformed") from exc
                    if extinction != 0.0:
                        raise ValueError("MIST release is not a zero-extinction isochrone")
                    extinction_zero = True
                if line.startswith(b"# EEP "):
                    columns = line.decode("ascii").split()[1:]
                    if (len(columns) <= max(MIST_COLUMNS)
                            or tuple(columns[i] for i in MIST_COLUMNS) != MIST_COLUMN_NAMES):
                        raise ValueError("MIST release columns/units do not match the adopted model")
                    break
            else:
                raise ValueError("MIST release lacks its photometric/physical coordinate header")
            if not extinction_zero:
                raise ValueError("MIST release lacks its zero-extinction declaration")
            data = np.loadtxt(stream, usecols=MIST_COLUMNS, comments="#", ndmin=2)
            if not np.all(np.isfinite(data)):
                raise ValueError("nonfinite MIST release row")
            # Isochrones older than the Planck 2018 age of the Universe carry no prior mass.
            data = data[10.0**data[:, LOG_AGE] <= maximum_age]
            if data.shape[0] == 0:
                continue
            chunks.append(data)
    if not chunks:
        raise ValueError("MIST archive has no isochrones")
    grid = np.concatenate(chunks)
    chunks.clear()
    ages = np.unique(grid[:, LOG_AGE])
    if ages.size < 2:
        raise ValueError("MIST archive needs at least two admitted ages for an age cell width")
    dm = np.empty(len(grid))
    for age in ages:
        age_indices = np.flatnonzero(grid[:, LOG_AGE] == age)
        at_age = grid[age_indices]
        for feh in np.unique(at_age[:, FEH_INIT]):
            selected = at_age[:, FEH_INIT] == feh
            rows = at_age[selected]
            masses, inverse, counts = np.unique(rows[:, INITIAL_MASS], return_inverse=True, return_counts=True)
            if masses.size < 2 or masses[0] <= 0 or np.any(rows[:, STAR_MASS] <= 0):
                raise ValueError("MIST mass coordinate is not positive and ordered")
            # Repeated EEP masses share one physical cell, even across members.
            dm[age_indices[selected]] = np.gradient(masses)[inverse] / counts[inverse]
    if np.any(dm <= 0):
        raise ValueError("MIST mass coordinate is not positive and ordered")
    grid = np.column_stack((grid, dm))
    # Global age/[Fe/H] widths do not depend on the member split.
    grid[:, CELL_WEIGHT] *= np.gradient(10.0**ages)[np.searchsorted(ages, grid[:, LOG_AGE])]
    metals = np.unique(grid[:, FEH_INIT])
    if metals.size < 2:
        raise ValueError("MIST archive needs at least two [Fe/H] grids for a metallicity cell width")
    grid[:, CELL_WEIGHT] *= np.gradient(metals)[np.searchsorted(metals, grid[:, FEH_INIT])]
    grid[:, CELL_WEIGHT] *= grid[:, INITIAL_MASS]**-2.35
    _MIST_CACHE.clear()
    _MIST_CACHE[key] = grid
    return grid


def parallax_zero_point_mas(inputs):
    """Evaluate Lindegren et al. (2021) Gaia parallax zero point in mas."""
    global _ZERO_POINT_TABLES_LOADED
    release = inputs.get("parallax_release", "DR3")
    if release not in ("EDR3", "DR3"):
        raise ValueError(f"no approved Gaia {release} parallax zero-point model")
    from zero_point import zpt

    if not _ZERO_POINT_TABLES_LOADED:
        zpt.load_tables()
        _ZERO_POINT_TABLES_LOADED = True
    try:
        # The supported array API avoids the NumPy 2 scalar-casting incompatibility.
        values = zpt.get_zpt(
            np.array([float(inputs["phot_g_mean_mag"])]),
            np.array([float(inputs["nu_eff_used_in_astrometry_per_um"])
                      if inputs.get("nu_eff_used_in_astrometry_per_um") is not None else math.nan]),
            np.array([float(inputs["pseudocolour_per_um"])
                      if inputs.get("pseudocolour_per_um") is not None else math.nan]),
            np.array([float(inputs["ecl_lat_deg"])]),
            np.array([int(inputs["astrometric_params_solved"])]),
            _warnings=False,
        )
        value = np.asarray(values, dtype=float).item()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"parallax zero-point inputs are incomplete: {exc}") from exc
    if not math.isfinite(value):
        raise ValueError("the source lies outside the Lindegren et al. (2021) zero-point validity domain")
    return value


def _real(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def mist_g_observables(neighbor, zero_point_fn=parallax_zero_point_mas):
    """Return G, its error [mag], corrected parallax and its error [mas]."""
    g = _real(neighbor.get("phot_g_mean_mag"), "phot_g_mean_mag")
    gerr = positive(neighbor, "phot_g_mean_mag_error")
    parallax = positive(neighbor, "parallax_mas")
    perr = positive(neighbor, "parallax_error_mas")
    zero_point = zero_point_fn(neighbor)
    corrected = parallax - zero_point
    if corrected / perr < 20:
        raise ValueError("G-only MIST route needs corrected parallax S/N>=20")
    return g, gerr, corrected, perr, zero_point


def mist_g_distribution(neighbor, grid, extinction_max, *, zero_point_fn=parallax_zero_point_mas,
                        distance_scale_pc=1000.0, include_initial_feh=False, include_logg=False):
    """G-only, high-S/N parallax inference without a dwarf assumption.

    The MIST grid's ``Gaia_G_EDR3`` is an intrinsic absolute magnitude at
    A_V=0, so the observed G is modelled as M_G + mu + A_G, not M_G + mu + A_V.
    The retained V-band field envelope, [0,A_V(infinity)] mag,
    is integrated per grid node with the official Gaia (E)DR3 G-band
    coefficient k_G(Teff,A_V) = A_G/A_V above; A_V=0 therefore returns the
    exact zero-extinction likelihood.  The envelope is V-band A_V from the
    SFD98 E(B-V) map with A_V=2.742 E(B-V) (Schlafly & Finkbeiner 2011),
    while the DPAC fit's A_0 is the 550 nm monochromatic extinction; that
    definition difference is neither separately corrected nor quantified here.
    Bailer-Jones (2015),
    eqn:r2e-2prior, gives p(d) proportional to d^2 exp(-d/L), L=1000 pc
    (illustrative, not a fitted direction-dependent Galactic model).
    Integrating the narrow Gaussian G likelihood gives d^3 (the d ln d/dG
    Jacobian). Gaussian parallax variance includes measured G uncertainty by
    first-order propagation. Distances are pc, parallaxes mas and G errors mag.
    Every input problem raises ``ValueError``. All grid nodes are kept (no sigma
    cut), so the returned lists can hold every positive-weight node of the grid.
    """
    g, gerr, parallax, perr, zero_point = mist_g_observables(neighbor, zero_point_fn)
    extinction_max = _real(extinction_max, "field A_V(infinity)")
    distance_scale_pc = _real(distance_scale_pc, "distance scale [pc]")
    if distance_scale_pc <= 0:
        raise ValueError("distance scale must be positive and finite [pc]")
    if not 0 <= extinction_max < 1:
        raise ValueError("G-only MIST route needs parallax S/N>=20 and field A_V(infinity)<1 mag")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] <= CELL_WEIGHT:
        raise ValueError("MIST grid must be a two-dimensional array with the adopted columns")
    if not np.all(np.isfinite(grid[:, (LOG_TEFF, LOG_G, GAIA_G, CELL_WEIGHT)])):
        raise ValueError("MIST grid coordinates and cell weights must be finite")
    if np.any(grid[:, CELL_WEIGHT] <= 0):
        raise ValueError("MIST grid cell weights must be positive")
    teff_k = 10.0**grid[:, LOG_TEFF]
    if not np.all(np.isfinite(teff_k)) or np.any(teff_k <= 0):
        raise ValueError("MIST grid effective temperatures must be finite and positive")
    log_g = grid[:, LOG_G]
    x, weights = np.polynomial.legendre.leggauss(12)
    log_cell = np.log(grid[:, CELL_WEIGHT])
    log_weights = []
    for attenuation, weight in zip((x + 1) * extinction_max / 2, weights / 2):
        # attenuation is the V-band envelope coordinate; k_G converts it to
        # the Gaia G band per node, including its A_V dependence.
        a_g = attenuation * _gaia_edr3_k_g(teff_k, log_g, attenuation)
        with np.errstate(over="ignore", invalid="ignore"):
            distance_pc = 10.0**((g - grid[:, GAIA_G] - a_g) / 5 + 1)
            model_plx = 1000 / distance_pc
            variance = perr**2 + (model_plx * math.log(10) / 5 * gerr)**2
            term = (log_cell + 3 * np.log(distance_pc) - distance_pc / distance_scale_pc + math.log(weight)
                    - .5 * (parallax - model_plx)**2 / variance - .5 * np.log(variance))
        # A node whose distance overflows has zero prior density (exp(-d/L)), not NaN.
        log_weights.append(np.where(np.isfinite(term), term, -np.inf))
    logw = logsumexp(log_weights, axis=0)
    if not np.any(np.isfinite(logw)):
        raise ValueError("no finite MIST likelihood support for this source")
    # Retain all representable weights; no credible-interval or sigma cut.
    weights = np.exp(logw - logsumexp(logw))
    keep = weights > 0
    if not np.any(keep):
        raise ValueError("MIST source measure has no positive-weight nodes")
    return {
        "mass_solar": grid[keep, STAR_MASS].tolist(),
        "radius_solar": radius(grid[keep, STAR_MASS], grid[keep, LOG_G]).tolist(),
        "teff_k": (10.0**grid[keep, LOG_TEFF]).tolist(), "weights": weights[keep].tolist(),
        **({"logg_cgs": grid[keep, LOG_G].tolist()} if include_logg else {}),
        **({"initial_feh_dex": grid[keep, FEH_INIT].tolist()} if include_initial_feh else {}),
        "method": ("MIST v1.2 G/parallax likelihood with mass-age-metallicity cell integration "
                   "and Gaia (E)DR3 A_G/A_V extinction"),
        "parallax_zero_point_mas": zero_point,
        "limitations": (
            "G-only inference is prior-informed; single-star grid, uniform linear age/[Fe/H], "
            f"Salpeter IMF and EDSD distance prior L={distance_scale_pc} pc (scale sensitivity required). Field attenuation envelope "
            f"0..{extinction_max} mag A_V from the retained field record; not a measured source extinction. "
            "A_G/A_V uses the official Gaia (E)DR3 k_G(Teff,A_V) fit (Danielski et al. 2018 method; "
            "Riello et al. 2021 passbands; solar metallicity, 3500-10000 K, 0.01<A_0<20 mag; "
            "3.5% maximum G-band fit residual, extinction-law/reference-spectra systematics dominate); "
            "MIST nodes outside that temperature range use the nearest-boundary coefficient. "
            "The MS/HRD-top fit is selected by nearest assumed gravity, an adopted approximation. "
            "The fit's monochromatic A_0 at 550 nm is approximated by the retained V-band A_V; "
            "A_V<0.01 uses a continuous uncalibrated extension. These and metallicity/temperature "
            "extrapolation are not sampled uncertainties. "
            f"Lindegren et al. (2021) zero point {zero_point} mas subtracted; high-S/N linearized G errors. "
            "Grid, extinction, binary and parallax-systematic sensitivity remain required."
        ),
    }
