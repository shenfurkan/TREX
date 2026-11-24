"""Portable host-specific TESS limb darkening from published reference tables.

TREX retains its distinct aperture and population model, but host-specific
limb-darkening selection follows the packaged grid and nearest-node procedure
of ``triceratops==1.0.20`` inside that package's lookup bounds.  The versioned
reference grid is vendored as TREX package data with its MIT license notice, so
native TREX execution does not depend on a separately installed optional engine.

The grid columns are ``logg``, ``Teff``, ``Z``, ``xi``, ``aLSM``, and ``bLSM``.
``aLSM`` and ``bLSM`` are passed to the quadratic transit law as dimensionless
``u1`` and ``u2``. The adapter selects the nearest metallicity plane, then follows TRICERATOPS'
published-code discrete TESS lookup convention: 250-K and 0.5-dex grid nodes
within the package's 3500--10000 K and 3.5--5.0 dex lookup bounds. Its 49 absent
metal-poor nodes have a separate MPS-ATLAS quadratic supplement, never a
metallicity-plane substitution or an alteration of existing nodes.

That packaged cut is a subset of Claret (2017, A&A 600, A30) ATLAS table 25,
which the paper's Table 1 describes as 3500--50000 K, ``0.0 <= log g <= 5.0``
dex, ``-5.0 <= [M/H] <= +1.0``, and microturbulence 0--8 km/s. Outside the
packaged bounds an evolved host now takes bilinearly interpolated ``aLSM``/
``bLSM`` from the same published table instead of the nearest main-sequence-like
node. Bilinear interpolation in (Teff [K], log10 g) needs a complete published
four-node cell at the nearest metallicity plane; a hole, a domain edge, or a
missing plane keeps the reference nearest-node rule and its warning. No value is
extrapolated and metallicity is never interpolated.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib import resources
from numbers import Integral
from typing import Dict, List, Tuple

import numpy as np


TRICERATOPS_SOURCE_VERSION = "1.0.20"
MC_LDC_ROLE = "monte-carlo-limb-darkening-provenance"
TESS_GRID_RESOURCE = "_data/triceratops-1.0.20-ldc_tess.csv"
GRID_COLUMNS = ("logg", "Teff", "Z", "xi", "aLSM", "bLSM")
TESS_SUPPLEMENT_RESOURCE = "_data/mps-atlas-tess-quadratic-supplement.csv"
TESS_SUPPLEMENT_COLUMNS = ("logg", "Teff", "Z", "u1", "u2")
# TRICERATOPS 1.0.20 marginal_likelihoods.py discretizes companion hosts onto
# these packaged TESS grid coordinates before indexing ldc_tess.csv. They are
# reference-engine lookup rules, not newly fitted empirical coefficients.
TESS_TEFF_GRID_STEP_K = 250.0
TESS_LOGG_GRID_STEP_DEX = 0.5
TESS_TEFF_GRID_MIN_K = 3500.0
TESS_TEFF_GRID_MAX_K = 10000.0
TESS_LOGG_GRID_MIN_DEX = 3.5
TESS_LOGG_GRID_MAX_DEX = 5.0

# Claret (2017, A&A 600, A30, DOI 10.1051/0004-6361/201629705) ATLAS table 25
# restricted to the series' V_xi = 2 km/s reference
# plane. It is the published source of every row in the packaged cut above, plus
# the evolved-star support (log g 0.0--3.5, Teff 10000--50000 K) that
# TRICERATOPS 1.0.20 does not ship. Values are the paper's LSM coefficients of
# Eq. 2; nothing here is derived or fitted.
CLARET_TESS_GRID_RESOURCE = "_data/claret-2017-tess-table25-atlas.csv"
CLARET_SOURCE_ID = "claret-2017-tess-atlas-table25"
CLARET_SOURCE_REFERENCE = "Claret (2017), A&A 600, A30, DOI 10.1051/0004-6361/201629705"
CLARET_SOURCE_CATALOGUE = "CDS/VizieR J/A+A/600/A30/table25"
CLARET_SOURCE_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    "?-source=J%2FA%2BA%2F600%2FA30%2Ftable25&-out.all=&-out.max=unlimited"
)
CLARET_SOURCE_RETRIEVED_UTC = "2026-10-07T22:24:58.529115+00:00"
CLARET_SOURCE_METHOD = (
    "ATLAS plane-parallel model atmospheres, TESS passband, quadratic law Eq. 2, "
    "least-squares fit (aLSM, bLSM), microturbulence 2 km/s plane"
)
CLARET_SOURCE_INTERPOLATION = (
    "bilinear in (Teff [K], log10 g [cgs]) inside a complete published four-node cell "
    "at the nearest metallicity plane [dex]; no extrapolation, no metallicity "
    "interpolation, missing nodes fall back to the reference nearest node"
)
CLARET_TEFF_MIN_K = 3500.0
CLARET_TEFF_MAX_K = 50000.0
CLARET_LOGG_MIN_DEX = 0.0
CLARET_LOGG_MAX_DEX = 5.0
PHOENIX_TESS_GRID_RESOURCE = "_data/claret-2017-tess-table15-phoenix-cond.csv"
PHOENIX_SOURCE_ID = "claret-2017-tess-phoenix-cond-table15"
PHOENIX_SOURCE_URL = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
                      "-source=J%2FA%2BA%2F600%2FA30%2Ftableab&-out.max=unlimited")
PHOENIX_SOURCE_RETRIEVED_UTC = "2026-11-04T19:40:00Z"

# Provenance contract written into the TREX report when a modeled host's
# atmosphere is discretized onto the packaged grid's boundary nodes.
LDC_GRID_CLAMP_CONTRACT = "triceratops-tess-ldc-grid-clamp-v1"
TRICERATOPS_GRID_SOURCE = "triceratops-tess-grid"
CANONICAL_TESS_BAND_ID = "TESS/TESS.T"
# Official PLATO Consortium deliverable and its inspected domain. The table is
# not vendored in this TREX release; naming the source/domain keeps the refusal
# actionable without inventing coefficients or treating the MPS-ATLAS
# cross-check as an official flight calibration.
PLATO_LDC_REFERENCE = (
    "Morello (2022), RNAAS 6, 248, Zenodo 10.5281/zenodo.7339706; "
    "MARCS+NLTE ExoTETHyS models, 4500-7000 K, log g 3.0-5.0 dex"
)
ROMAN_NIR_LDC_REFERENCE = (
    "Roman wide-band spherical limb-darkening tables (Sing et al. 2026, "
    "arXiv:2609.00263); not installed in this release"
)


class LimbDarkeningUnavailableError(ValueError):
    """No published coefficient table covers the requested band/star."""



def _finite(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError("{0} must be finite".format(label))
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{0} must be finite".format(label)) from exc
    if not math.isfinite(numeric):
        raise ValueError("{0} must be finite".format(label))
    return numeric


@dataclass(frozen=True)
class LimbDarkeningNode:
    """The coefficient source actually used for one host-atmosphere request.

    ``used_teff_k``/``used_logg_cgs`` are the reference grid node inside the
    packaged bounds; an interpolated request keeps its own continuous
    coordinates because no node was substituted.
    """

    u1: float
    u2: float
    used_teff_k: float
    used_logg_cgs: float
    used_metallicity_dex: float
    boundary_clamped: bool
    source: str = TRICERATOPS_GRID_SOURCE
    interpolated: bool = False


@dataclass(frozen=True)
class LimbDarkeningPopulation:
    """Per-draw reference nodes with the same rounding and audit as ``select``."""

    u1: np.ndarray
    u2: np.ndarray
    used_teff_k: np.ndarray
    used_logg_cgs: np.ndarray
    used_metallicity_dex: float | np.ndarray
    boundary_clamped: np.ndarray
    interpolated: np.ndarray
    phoenix_interpolated: np.ndarray

    @property
    def atlas_interpolated(self):
        return self.interpolated & ~self.phoenix_interpolated


@dataclass(frozen=True)
class TriceratopsTessLimbDarkeningGrid:
    """Reference rows retain their identity; source-derived missing nodes stay separate."""

    package_version: str
    sha256: str
    rows: Tuple[Dict[str, float], ...]
    supplementary_rows: Tuple[Dict[str, float], ...] = ()
    supplement_sha256: str = ""
    extension_rows: Tuple[Dict[str, float], ...] = ()
    extension_sha256: str = ""
    phoenix_rows: Tuple[Dict[str, float], ...] = ()
    phoenix_sha256: str = ""

    @cached_property
    def _metallicity_planes(self) -> Tuple[float, ...]:
        """Distinct metallicity planes, ascending, built once per grid."""
        return tuple(sorted({row["Z"] for row in self.rows}))

    @cached_property
    def _node_index(self) -> Dict[Tuple[float, float, float], Dict[str, float]]:
        """Exact ``(Z, Teff, logg)`` node lookup, built once per grid."""
        index = {(row["Z"], row["Teff"], row["logg"]): row for row in self.rows}
        for row in self.supplementary_rows:
            key = row["Z"], row["Teff"], row["logg"]
            if key in index:
                raise ValueError("limb-darkening supplement must not replace an existing node")
            index[key] = dict(row, aLSM=row["u1"], bLSM=row["u2"])
        return index

    def _reference_node(self, teff: float, logg: float, metallicity: float) -> LimbDarkeningNode:
        """Reference-engine nearest node, including its auditable boundary clamp."""
        selected_metallicity = min(
            self._metallicity_planes, key=lambda value: abs(value - metallicity)
        )
        rounded_teff = np_round_reference(teff / TESS_TEFF_GRID_STEP_K) * TESS_TEFF_GRID_STEP_K
        selected_teff = min(max(rounded_teff, TESS_TEFF_GRID_MIN_K), TESS_TEFF_GRID_MAX_K)
        rounded_logg = np_round_reference(logg / TESS_LOGG_GRID_STEP_DEX) * TESS_LOGG_GRID_STEP_DEX
        selected_logg = min(max(rounded_logg, TESS_LOGG_GRID_MIN_DEX), TESS_LOGG_GRID_MAX_DEX)
        row = self._node_index.get((selected_metallicity, selected_teff, selected_logg))
        if row is None:
            raise RuntimeError("TRICERATOPS TESS grid lacks an exact host coefficient node")
        return LimbDarkeningNode(
            u1=row["aLSM"],
            u2=row["bLSM"],
            used_teff_k=selected_teff,
            used_logg_cgs=selected_logg,
            used_metallicity_dex=selected_metallicity,
            boundary_clamped=not (
                TESS_TEFF_GRID_MIN_K <= teff <= TESS_TEFF_GRID_MAX_K
                and TESS_LOGG_GRID_MIN_DEX <= logg <= TESS_LOGG_GRID_MAX_DEX
            ),
        )

    def select(self, teff_k: float, logg_cgs: float, metallicity_dex: float) -> LimbDarkeningNode:
        """Return the coefficient source for one host atmosphere.

        Inside the packaged ``3500--10000 K`` and ``3.5--5.0 dex`` lookup bounds
        this is the reference engine's 250-K/0.5-dex nearest node, exactly as
        TRICERATOPS 1.0.20 selects it; a boundary clamp sets
        ``boundary_clamped`` and is auditable rather than hidden. Outside those
        bounds, a request inside a complete published Claret (2017) ATLAS
        four-node cell is bilinearly interpolated at its own continuous
        coordinates (``source`` = ``claret-2017-tess-atlas-table25``).
        Below 3500 K, complete PHOENIX-COND solar-plane cells are used where
        applicable. Both extensions set ``interpolated``; everything else keeps
        the nearest-node rule with its clamp, never an extrapolation.
        """
        teff = _finite(teff_k, "teff_k")
        logg = _finite(logg_cgs, "logg_cgs")
        metallicity = _finite(metallicity_dex, "metallicity_dex")
        if teff <= 0.0:
            raise ValueError("teff_k must be positive")
        if not (TESS_TEFF_GRID_MIN_K <= teff <= TESS_TEFF_GRID_MAX_K
                and TESS_LOGG_GRID_MIN_DEX <= logg <= TESS_LOGG_GRID_MAX_DEX):
            selected_metallicity = min(
                self._metallicity_planes, key=lambda value: abs(value - metallicity)
            )
            if teff < TESS_TEFF_GRID_MIN_K and selected_metallicity == 0.0:
                u1, u2, inside = self._extended_values(
                    np.asarray([teff]), np.asarray([logg]), 0.0, tables=self._phoenix_tables
                )
                if inside[0]:
                    return LimbDarkeningNode(float(u1[0]), float(u2[0]), teff, logg,
                                             0.0, False, PHOENIX_SOURCE_ID, True)
            u1, u2, inside = self._extended_values(
                np.asarray([teff]), np.asarray([logg]), metallicity
            )
            if inside[0]:
                selected_metallicity = min(
                    self._metallicity_planes, key=lambda value: abs(value - metallicity)
                )
                return LimbDarkeningNode(
                    u1=float(u1[0]),
                    u2=float(u2[0]),
                    used_teff_k=teff,
                    used_logg_cgs=logg,
                    used_metallicity_dex=selected_metallicity,
                    boundary_clamped=False,
                    source=CLARET_SOURCE_ID,
                    interpolated=True,
                )
        return self._reference_node(teff, logg, metallicity)

    def coefficients(self, teff_k: float, logg_cgs: float, metallicity_dex: float) -> Tuple[float, float]:
        """Return host-specific ``(u1, u2)`` from the selected published source."""
        node = self.select(teff_k, logg_cgs, metallicity_dex)
        return node.u1, node.u2

    @cached_property
    def _phoenix_tables(self) -> tuple:
        """Claret (2017) table 15, quasi-spherical solar PHOENIX-COND LSM nodes."""
        if not self.phoenix_rows:
            return (np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0, 0)), np.empty((0, 0, 0)))
        temperatures = np.array(sorted({row["Teff"] for row in self.phoenix_rows}))
        gravities = np.array(sorted({row["logg"] for row in self.phoenix_rows}))
        planes = np.array([0.0])
        u1, u2 = (np.full((1, gravities.size, temperatures.size), np.nan) for _ in range(2))
        for row in self.phoenix_rows:
            index = (0, np.searchsorted(gravities, row["logg"]), np.searchsorted(temperatures, row["Teff"]))
            u1[index], u2[index] = row["aLSM"], row["bLSM"]
        return temperatures, gravities, planes, u1, u2

    @cached_property
    def _extension_tables(self) -> tuple:
        """Dense published Claret nodes; absent (Z, log g, Teff) entries stay NaN.

        A dense lookup is cheap because the published grid is small and fixed;
        building it never scales with the Monte Carlo draw count.
        """
        if not self.extension_rows:
            return (np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0, 0)), np.empty((0, 0, 0)))
        temperatures = np.array(sorted({row["Teff"] for row in self.extension_rows}))
        gravities = np.array(sorted({row["logg"] for row in self.extension_rows}))
        planes = np.array(sorted({row["Z"] for row in self.extension_rows}))
        if tuple(planes) != self._metallicity_planes:
            raise ValueError("Claret extension metallicity planes differ from the reference grid")
        shape = (planes.size, gravities.size, temperatures.size)
        u1, u2 = np.full(shape, np.nan), np.full(shape, np.nan)
        for row in self.extension_rows:
            index = (
                int(np.searchsorted(planes, row["Z"])),
                int(np.searchsorted(gravities, row["logg"])),
                int(np.searchsorted(temperatures, row["Teff"])),
            )
            u1[index], u2[index] = row["aLSM"], row["bLSM"]
        return temperatures, gravities, planes, u1, u2

    def _extended_values(
        self, teff_k: np.ndarray, logg_cgs: np.ndarray, metallicity_dex, *, tables=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear Claret coefficients and an ``inside`` mask for each request.

        The nearest published metallicity plane is reused unchanged. Teff and
        log g must bracket published nodes; an exact node collapses to a
        zero-width interval, a hole or a domain edge sets ``inside`` False so
        the caller keeps the reference nearest-node rule. Weights of zero never
        require the opposite corner to exist, which is why an exact node on a
        published edge remains usable.
        """
        temperatures, gravities, planes, u1_table, u2_table = self._extension_tables if tables is None else tables
        teff = np.asarray(teff_k, dtype=float)
        logg = np.asarray(logg_cgs, dtype=float)
        if teff.shape != logg.shape:
            raise ValueError("host atmosphere arrays must have matching per-draw shapes")
        inside = np.zeros(teff.shape, dtype=bool)
        u1 = np.zeros(teff.shape, dtype=float)
        u2 = np.zeros(teff.shape, dtype=float)
        if temperatures.size == 0:
            return u1, u2, inside
        metallicity = np.asarray(metallicity_dex, dtype=float)
        if metallicity.ndim == 0:
            zi = int(np.argmin(np.abs(planes - metallicity)))
        else:
            zi = np.argmin(np.abs(metallicity[..., None] - planes), axis=-1)
        n_t, n_g = temperatures.size, gravities.size
        ti = np.searchsorted(temperatures, teff, side="left")
        gi = np.searchsorted(gravities, logg, side="left")
        exact_t = (ti < n_t) & (temperatures[np.minimum(ti, n_t - 1)] == teff)
        exact_g = (gi < n_g) & (gravities[np.minimum(gi, n_g - 1)] == logg)
        t0, t1 = np.where(exact_t, ti, ti - 1), np.where(exact_t, ti, ti)
        g0, g1 = np.where(exact_g, gi, gi - 1), np.where(exact_g, gi, gi)
        bracketed = (t0 >= 0) & (t1 < n_t) & (g0 >= 0) & (g1 < n_g)
        t0c, t1c = np.clip(t0, 0, n_t - 1), np.clip(t1, 0, n_t - 1)
        g0c, g1c = np.clip(g0, 0, n_g - 1), np.clip(g1, 0, n_g - 1)
        dt = temperatures[t1c] - temperatures[t0c]
        dg = gravities[g1c] - gravities[g0c]
        wt = np.where(dt > 0.0, (teff - temperatures[t0c]) / np.where(dt > 0.0, dt, 1.0), 0.0)
        wg = np.where(dg > 0.0, (logg - gravities[g0c]) / np.where(dg > 0.0, dg, 1.0), 0.0)
        w00, w01, w10, w11 = (1.0 - wg) * (1.0 - wt), (1.0 - wg) * wt, wg * (1.0 - wt), wg * wt
        a00, a01 = u1_table[zi, g0c, t0c], u1_table[zi, g0c, t1c]
        a10, a11 = u1_table[zi, g1c, t0c], u1_table[zi, g1c, t1c]
        b00, b01 = u2_table[zi, g0c, t0c], u2_table[zi, g0c, t1c]
        b10, b11 = u2_table[zi, g1c, t0c], u2_table[zi, g1c, t1c]
        complete = bracketed
        for values, weight in (
            (a00, w00), (a01, w01), (a10, w10), (a11, w11),
            (b00, w00), (b01, w01), (b10, w10), (b11, w11),
        ):
            complete = complete & (np.isfinite(values) | (weight == 0.0))
        inside = complete
        if not inside.any():
            return u1, u2, inside
        # Zero-weight NaN corners are replaced only after the completeness mask
        # has decided; they cannot leak a value into a supported request.
        def _combine(*terms) -> np.ndarray:
            return sum(np.where(np.isfinite(value), value, 0.0) * weight
                       for value, weight in terms)

        u1 = np.where(inside, _combine((a00, w00), (a01, w01), (a10, w10), (a11, w11)), 0.0)
        u2 = np.where(inside, _combine((b00, w00), (b01, w01), (b10, w10), (b11, w11)), 0.0)
        return u1, u2, inside

    @cached_property
    def _population_index(self) -> tuple:
        """Dense indices of actual retained coordinates; missing nodes stay NaN.

        This is a lookup table, not interpolation. Building it scales with the
        small reference grid, never with the Monte Carlo draw count.
        """
        temperatures = np.array(sorted({key[1] for key in self._node_index}))
        gravities = np.array(sorted({key[2] for key in self._node_index}))
        shape = (len(self._metallicity_planes), temperatures.size, gravities.size)
        u1, u2 = np.full(shape, np.nan), np.full(shape, np.nan)
        for (metallicity, teff, logg), row in self._node_index.items():
            if metallicity not in self._metallicity_planes:
                raise ValueError("limb-darkening supplement must use a reference metallicity plane")
            index = (
                self._metallicity_planes.index(metallicity),
                np.searchsorted(temperatures, teff), np.searchsorted(gravities, logg),
            )
            u1[index], u2[index] = row["aLSM"], row["bLSM"]
        return temperatures, gravities, u1, u2

    @cached_property
    def _cached_single_coefficients(self):
        # Nested proposals repeatedly select the same discrete background stars.
        @lru_cache(maxsize=4096)
        def lookup(teff, logg, metallicity):
            selected = self.select_many(np.array([teff]), np.array([logg]), metallicity)
            return float(selected.u1[0]), float(selected.u2[0]), bool(selected.boundary_clamped[0])

        return lookup

    def select_many(self, teff_k, logg_cgs, metallicity_dex) -> LimbDarkeningPopulation:
        """Vectorized source selection, preserving every draw and clamp.

        Temperatures are K, gravities log10(cgs), metallicity dex (scalar or per-star).
        Matched finite arrays and positive temperatures are required. Each draw
        follows the same rule as scalar ``select``: reference nearest node inside
        the packaged bounds, interpolated Claret coefficients outside them, and
        the reference clamp for everything the published table cannot bracket.
        A sparse/missing reference coefficient node raises exactly as scalar
        ``select`` does.
        """
        if any(np.asarray(v).dtype.kind not in "iuf" for v in (teff_k, logg_cgs, metallicity_dex)):
            raise ValueError("host atmosphere coordinates must be real numeric values, not booleans")
        teff, logg = np.asarray(teff_k, dtype=float), np.asarray(logg_cgs, dtype=float)
        metallicity = np.asarray(metallicity_dex, dtype=float)
        if metallicity.ndim != 0 and metallicity.shape != teff.shape:
            raise ValueError("metallicity_dex must be scalar or match the host atmosphere shape")
        if not np.all(np.isfinite(metallicity)):
            raise ValueError("metallicity_dex must be finite")
        if teff.shape != logg.shape:
            raise ValueError("host atmosphere arrays must have matching per-draw shapes")
        if not np.all(np.isfinite(teff)) or not np.all(np.isfinite(logg)):
            raise ValueError("host atmosphere arrays must be finite")
        if np.any(teff <= 0.0):
            raise ValueError("teff_k must be positive")
        planes = np.asarray(self._metallicity_planes)
        # Nearest scaled-solar abundance plane [dex]; ascending argmin preserves scalar tie handling.
        zi = (int(np.argmin(abs(planes-metallicity))) if metallicity.ndim == 0 else
              np.argmin(abs(metallicity[..., None]-planes), axis=-1))
        selected_metallicity = planes[zi]
        rounded_teff = np.round(teff / TESS_TEFF_GRID_STEP_K) * TESS_TEFF_GRID_STEP_K
        rounded_logg = np.round(logg / TESS_LOGG_GRID_STEP_DEX) * TESS_LOGG_GRID_STEP_DEX
        selected_teff = np.clip(rounded_teff, TESS_TEFF_GRID_MIN_K, TESS_TEFF_GRID_MAX_K)
        selected_logg = np.clip(rounded_logg, TESS_LOGG_GRID_MIN_DEX, TESS_LOGG_GRID_MAX_DEX)
        u1, u2 = np.full(teff.shape, np.nan), np.full(teff.shape, np.nan)
        interpolated = np.zeros(teff.shape, dtype=bool)
        phoenix = np.zeros(teff.shape, dtype=bool)
        outside = ~((teff >= TESS_TEFF_GRID_MIN_K) & (teff <= TESS_TEFF_GRID_MAX_K)
                    & (logg >= TESS_LOGG_GRID_MIN_DEX) & (logg <= TESS_LOGG_GRID_MAX_DEX))
        if np.any(outside):
            extended_u1, extended_u2, inside = self._extended_values(
                teff[outside], logg[outside],
                metallicity if metallicity.ndim == 0 else metallicity[outside],
            )
            interpolated[outside] = inside
            u1[interpolated], u2[interpolated] = extended_u1[inside], extended_u2[inside]
            # Table 15 has only solar composition; never assign it to another plane.
            cool = (teff < TESS_TEFF_GRID_MIN_K) & (selected_metallicity == 0.0)
            if np.any(cool):
                a, b, supported = self._extended_values(teff[cool], logg[cool], 0.0, tables=self._phoenix_tables)
                use = np.zeros(teff.shape, dtype=bool)
                use[cool] = supported
                u1[use], u2[use] = a[supported], b[supported]
                interpolated |= use
                phoenix = use
        reference = ~interpolated
        if np.any(reference):
            temperatures, gravities, u1_table, u2_table = self._population_index
            ti = np.searchsorted(temperatures, selected_teff[reference])
            gi = np.searchsorted(gravities, selected_logg[reference])
            if np.any(ti >= temperatures.size) or np.any(gi >= gravities.size):
                raise RuntimeError("TRICERATOPS TESS grid lacks an exact host coefficient node")
            if np.any(temperatures[ti] != selected_teff[reference]) or np.any(gravities[gi] != selected_logg[reference]):
                raise RuntimeError("TRICERATOPS TESS grid lacks an exact host coefficient node")
            index = (zi if metallicity.ndim == 0 else zi[reference], ti, gi)
            u1[reference], u2[reference] = u1_table[index], u2_table[index]
            if not np.all(np.isfinite(u1[reference])) or not np.all(np.isfinite(u2[reference])):
                raise RuntimeError("TRICERATOPS TESS grid lacks an exact host coefficient node")
        return LimbDarkeningPopulation(
            u1, u2,
            np.where(interpolated, teff, selected_teff),
            np.where(interpolated, logg, selected_logg),
            float(selected_metallicity) if metallicity.ndim == 0 else selected_metallicity,
            outside & ~interpolated,
            interpolated,
            phoenix,
        )


@lru_cache(maxsize=1)
def load_triceratops_tess_grid() -> TriceratopsTessLimbDarkeningGrid:
    """Load and validate the vendored version-pinned reference TESS grid.

    The grid is packaged data whose bytes are hash-recorded on the returned
    object, so it is immutable and safe to share; the result is cached per
    process because the per-likelihood host-atmosphere lookups of the nested
    evidence estimator would otherwise re-parse the CSV on every call. The
    published Claret (2017) ATLAS extension is loaded and cross-checked against
    every packaged row in the same call, so a mismatched pair is refused rather
    than silently mixing two tables.

    Raises:
        RuntimeError: A vendored grid resource is unavailable.
        ValueError: A vendored grid is malformed, has non-finite rows, or the
            extension disagrees with a packaged reference node.
    """
    try:
        resource = resources.files("trex").joinpath(TESS_GRID_RESOURCE)
        raw = resource.read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        raise RuntimeError("vendored TRICERATOPS TESS limb-darkening grid is unavailable") from exc
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames != list(GRID_COLUMNS):
            raise ValueError("vendored TRICERATOPS TESS grid has unexpected columns")
        rows: List[Dict[str, float]] = []
        for raw_row in reader:
            if raw_row is None:
                raise ValueError("vendored TRICERATOPS TESS grid has an invalid row")
            row = {name: _finite(raw_row.get(name), "grid {0}".format(name)) for name in GRID_COLUMNS}
            rows.append(row)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("vendored TRICERATOPS TESS grid is unreadable") from exc
    if not rows:
        raise ValueError("vendored TRICERATOPS TESS grid has no coefficient rows")
    # Kostogryz et al. 2022 (A&A 666, A60, DOI 10.1051/0004-6361/202243722,
    # CDS tablee3) TESS centre-to-limb variations,
    # absolute I(mu) linear in Teff at fixed M/H and logg, held flat from mu=0.01 to 0, then
    # passband.quadratic_reduction (dimensionless u). Flux-conserving quadratics, NOT Claret LSM coefficients.
    try:
        supplement_raw = resources.files("trex").joinpath(TESS_SUPPLEMENT_RESOURCE).read_bytes()
    except (ModuleNotFoundError, OSError) as exc:
        raise RuntimeError("TESS limb-darkening supplement is unavailable") from exc
    try:
        supplement = csv.DictReader(supplement_raw.decode("utf-8").splitlines())
        if supplement.fieldnames != list(TESS_SUPPLEMENT_COLUMNS):
            raise ValueError("TESS limb-darkening supplement has unexpected columns")
        supplementary_rows = tuple({name: _finite(row.get(name), "supplement " + name)
                                   for name in TESS_SUPPLEMENT_COLUMNS} for row in supplement)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("TESS limb-darkening supplement is unreadable") from exc
    keys = {(row["Z"], row["Teff"], row["logg"]) for row in supplementary_rows}
    if not supplementary_rows or len(keys) != len(supplementary_rows):
        raise ValueError("TESS limb-darkening supplement has no nodes or repeats a node")
    extension_raw, extension_rows = _load_claret_extension()
    phoenix_raw, phoenix_rows = _load_claret_extension(PHOENIX_TESS_GRID_RESOURCE)
    if any(row["Z"] != 0.0 or not 2300.0 <= row["Teff"] <= 12000.0
           or not 2.5 <= row["logg"] <= 6.0 for row in phoenix_rows):
        raise ValueError("packaged PHOENIX-COND table has invalid source coordinates")
    _require_reference_parity(rows, extension_rows)
    from trex.stellar_light.passband import quadratic_nonnegative

    # Claret 2017 Eq. 2: normalized polynomial (unused flux, I(1), -u1, -u2), dimensionless.
    polynomial = np.array([[1., 1., -row["u1"], -row["u2"]] for row in supplementary_rows])
    if not quadratic_nonnegative(polynomial).all():
        raise ValueError("TESS limb-darkening supplement has negative quadratic intensity")
    return TriceratopsTessLimbDarkeningGrid(
        package_version=TRICERATOPS_SOURCE_VERSION,
        sha256=hashlib.sha256(raw).hexdigest(),
        rows=tuple(rows),
        supplementary_rows=supplementary_rows,
        supplement_sha256=hashlib.sha256(supplement_raw).hexdigest(),
        extension_rows=extension_rows,
        extension_sha256=hashlib.sha256(extension_raw).hexdigest(),
        phoenix_rows=phoenix_rows,
        phoenix_sha256=hashlib.sha256(phoenix_raw).hexdigest(),
    )


def _load_claret_extension(resource=CLARET_TESS_GRID_RESOURCE) -> Tuple[bytes, Tuple[Dict[str, float], ...]]:
    """Read and validate a packaged Claret (2017) atmosphere table."""
    try:
        raw = resources.files("trex").joinpath(resource).read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
        raise RuntimeError("packaged Claret (2017) TESS limb-darkening table is unavailable") from exc
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
        phoenix = resource == PHOENIX_TESS_GRID_RESOURCE
        columns = ("logg", "Teff", "Z", "L/HP", "aLSM", "bLSM") if phoenix else GRID_COLUMNS
        if reader.fieldnames != list(columns):
            raise ValueError("packaged Claret (2017) grid has unexpected columns")
        rows = tuple(
            {name: _finite(raw_row.get(name), "claret {0}".format(name)) for name in columns}
            for raw_row in reader
        )
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("packaged Claret (2017) grid is unreadable") from exc
    if not rows:
        raise ValueError("packaged Claret (2017) grid has no coefficient rows")
    coordinate = "L/HP" if phoenix else "xi"
    if any(row[coordinate] != 2.0 for row in rows):
        raise ValueError("packaged Claret (2017) grid has an unexpected model parameter")
    keys = {(row["Z"], row["Teff"], row["logg"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("packaged Claret (2017) grid repeats an atmosphere node")
    return raw, rows


def _require_reference_parity(
    rows: List[Dict[str, float]], extension_rows: Tuple[Dict[str, float], ...]
) -> None:
    """Refuse an extension that disagrees with any packaged reference node.

    The packaged TRICERATOPS cut is a strict subset of the published table, so
    the two files must agree node for node inside that cut. A swapped or
    truncated extension would otherwise silently change host coefficients.
    """
    extension = {(row["Z"], row["Teff"], row["logg"]): row for row in extension_rows}
    packaged = {(row["Z"], row["Teff"], row["logg"]): row for row in rows}
    for key, reference in packaged.items():
        published = extension.get(key)
        if published is None or (published["aLSM"], published["bLSM"]) != (
            reference["aLSM"], reference["bLSM"]
        ):
            raise ValueError("Claret (2017) extension disagrees with a packaged reference node")
    in_cut = {key for key in extension if
              TESS_TEFF_GRID_MIN_K <= key[1] <= TESS_TEFF_GRID_MAX_K
              and TESS_LOGG_GRID_MIN_DEX <= key[2] <= TESS_LOGG_GRID_MAX_DEX}
    if in_cut != set(packaged):
        raise ValueError("Claret (2017) extension does not match the packaged reference coverage")


def np_round_reference(value: float) -> float:
    """Use NumPy's tie-to-even rounding used by the reference implementation."""
    return float(np.round(value))


# Discriminates the one record describing the *target* star's own
# atmosphere from the per-host clamp events, so the report can disclose
# which node the target-hosted scenarios actually used without that
# disclosure being read as a boundary clamp.
TARGET_HOST_LDC_ROLE: str = "target-host-atmosphere"


def summarize_ldc_grid_provenance(events: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate host limb-darkening provenance for the TREX report.

    ``events`` is the list accumulated by the evidence calculation: one entry per
    modeled host whose draws fell outside the packaged reference bounds, plus at
    most one ``TARGET_HOST_LDC_ROLE`` record naming the coefficients the target
    star's own scenarios used. Each outside-bounds entry separates draws that
    took interpolated Claret (2017) coefficients from draws that still
    needed the reference nearest-node clamp, and the block below names the
    extension source, URL, retrieval date and interpolation rule. Returning the
    grid bounds and the exact nodes used makes both the clamp and the target
    selection auditable instead of silent. Unavailable MC state leaves
    unknown flags as ``None``; a known clamped *target* still sets the flag.
    """
    marker, target_host = None, None
    mc_events, clamp_events, interpolation_events = [], [], []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("limb-darkening provenance entries must be mappings")
        role = event.get("role")
        if role == MC_LDC_ROLE and marker is None:
            marker = event
        if event.get("stage") == "monte-carlo":
            mc_events.append(event)
        if role == TARGET_HOST_LDC_ROLE:
            if target_host is None:
                target_host = event
        else:
            if event.get("clamped_draws", 0) or event.get("used_nodes"):
                clamp_events.append(event)
            if event.get("interpolated_draws", 0):
                interpolation_events.append(event)
    mc_status = None if marker is None else marker.get("status")
    source_fields = ("reference_draws", "atlas_draws", "phoenix_draws", "clamped_draws", "total_draws")
    complete = bool(marker and marker.get("status") == "available" and marker.get("source_counts_complete"))
    complete = complete and all(
        all(isinstance(event.get(name), Integral) and not isinstance(event[name], (bool, np.bool_))
            and event[name] >= 0 for name in source_fields)
        and sum(event[name] for name in source_fields[:-1]) == event["total_draws"] for event in mc_events)
    source_counts = {name: sum(event[name] for event in mc_events) if complete else None for name in source_fields}
    source_counts["status"] = ("available" if complete else
                               "unknown (legacy source counts incomplete)" if mc_status == "available" else
                               "unavailable")
    summary = {
        "contract": LDC_GRID_CLAMP_CONTRACT,
        "package_version": TRICERATOPS_SOURCE_VERSION,
        "coefficient_supplement": {
            "resource": TESS_SUPPLEMENT_RESOURCE,
            "source": "Kostogryz et al. 2022, A&A 666, A60, CDS J/A+A/666/A60/tablee3",
            "method": "MPS-ATLAS TESS CLV; flux-conserving quadratic; missing nodes only",
            "reference": "flux-conserving quadratic fit, not Claret LSM coefficients",
        },
        "coefficient_extension": {
            "resource": CLARET_TESS_GRID_RESOURCE,
            "source": CLARET_SOURCE_ID,
            "citation": CLARET_SOURCE_REFERENCE,
            "catalogue": CLARET_SOURCE_CATALOGUE,
            "url": CLARET_SOURCE_URL,
            "retrieved_utc": CLARET_SOURCE_RETRIEVED_UTC,
            "method": CLARET_SOURCE_METHOD,
            "interpolation": CLARET_SOURCE_INTERPOLATION,
            "teff_k_domain": [CLARET_TEFF_MIN_K, CLARET_TEFF_MAX_K],
            "logg_cgs_domain": [CLARET_LOGG_MIN_DEX, CLARET_LOGG_MAX_DEX],
            "decision": "interpolated only outside the packaged bounds, never extrapolated",
        },
        "cool_coefficient_extension": {
            "resource": PHOENIX_TESS_GRID_RESOURCE,
            "source": PHOENIX_SOURCE_ID,
            "citation": CLARET_SOURCE_REFERENCE,
            "url": PHOENIX_SOURCE_URL,
            "retrieved_utc": PHOENIX_SOURCE_RETRIEVED_UTC,
            "method": "PHOENIX-COND quasi-spherical TESS quadratic LSM, solar metallicity only",
            "selection": "below 3500 K, only if nearest reference metallicity is solar and all weighted nodes exist",
        },
        "teff_k_lookup_bounds": [TESS_TEFF_GRID_MIN_K, TESS_TEFF_GRID_MAX_K],
        "logg_cgs_lookup_bounds": [TESS_LOGG_GRID_MIN_DEX, TESS_LOGG_GRID_MAX_DEX],
        "clamped": bool(clamp_events)
        or bool(target_host and target_host.get("boundary_clamped")),
        "events": clamp_events,
        "interpolated": bool(interpolation_events)
        or bool(target_host and target_host.get("interpolated")),
        "interpolation_events": interpolation_events,
        "target_host": target_host,
        "monte_carlo_source_counts": source_counts,
    }
    summary["monte_carlo_provenance_status"] = mc_status if mc_status is not None else "unavailable"
    if not complete:
        for flag in ("clamped", "interpolated"):
            if not summary[flag]:
                summary[flag] = None
        for field in ("events", "interpolation_events"):
            if not summary[field]:
                summary[field] = None
    return summary


def resolve_host_ldc(band_id: str, teff_k: float, logg_cgs: float, metallicity_dex: float) -> LimbDarkeningNode:
    """Route one host's own atmosphere to the published coefficient table for a band.

    Only the canonical TESS response has a packaged coefficient table in this
    release; the exact historical selection is reused unchanged. A different
    band raises ``LimbDarkeningUnavailableError`` instead of substituting TESS
    coefficients: the prepared PHOENIX angular profile is the supported route
    for other bands, and it keeps each star's own composition and extinction.
    """
    if not isinstance(band_id, str) or not band_id.strip():
        raise ValueError("a limb-darkening band must be a canonical response id")
    if band_id != CANONICAL_TESS_BAND_ID:
        if band_id.startswith("PLATO/"):
            raise LimbDarkeningUnavailableError(
                "the official PLATO limb-darkening table is not installed ({0}); "
                "load it only inside its inspected domain, and do not describe the "
                "MPS-ATLAS supplement as an official flight calibration; use the "
                "prepared PHOENIX angular profile meanwhile".format(PLATO_LDC_REFERENCE)
            )
        if band_id.startswith("Roman/"):
            raise LimbDarkeningUnavailableError(
                "no Roman wide-band limb-darkening table is installed ({0}); use the "
                "prepared PHOENIX angular profile for this response".format(
                    ROMAN_NIR_LDC_REFERENCE)
            )
        raise LimbDarkeningUnavailableError(
            "no published limb-darkening coefficient table for band {0!r}; use the "
            "prepared PHOENIX angular profile for that response".format(band_id)
        )
    return load_triceratops_tess_grid().select(teff_k, logg_cgs, metallicity_dex)


__all__ = [
    "CANONICAL_TESS_BAND_ID",
    "LDC_GRID_CLAMP_CONTRACT",
    "LimbDarkeningUnavailableError",
    "PLATO_LDC_REFERENCE",
    "ROMAN_NIR_LDC_REFERENCE",
    "TARGET_HOST_LDC_ROLE",
    "LimbDarkeningNode",
    "LimbDarkeningPopulation",
    "TESS_LOGG_GRID_MAX_DEX",
    "TESS_LOGG_GRID_MIN_DEX",
    "TESS_TEFF_GRID_MAX_K",
    "TESS_TEFF_GRID_MIN_K",
    "TriceratopsTessLimbDarkeningGrid",
    "load_triceratops_tess_grid",
    "resolve_host_ldc",
    "summarize_ldc_grid_provenance",
]
