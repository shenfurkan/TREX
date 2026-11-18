"""Multilinear interpolation inside actual same-family atmosphere cells only."""

from dataclasses import dataclass
from itertools import product
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from trex.acquisition.storage import _sync_directory


class UnsupportedAtmosphereError(ValueError):
    """An atmosphere is unavailable, not zero astrophysical prior probability."""

    def __init__(self, model_id, coordinates):
        self.coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 4)
        # Keep the supplied failed-request array, not another million-draw snapshot.
        first = None if not len(self.coordinates) else self.coordinates[0].tolist()
        super().__init__(
            f"{model_id}: {len(self.coordinates)} unsupported atmosphere request(s) "
            f"(K,logg,[M/H],A_V); first={first}; "
            "all requests retained in exception.coordinates; no star removed or fallback used")


def _affine_step(axis):
    """Step of a uniformly spaced static axis, or None.

    A direct bracket is only exact when every knot shares one step up to
    floating rounding; nonuniform and traced axes keep ``searchsorted``. The
    caller corrects the affine index against the real knots, so rounding at a
    knot cannot select a wrong cell.
    """
    try:
        values = np.asarray(axis, dtype=float)
    except (TypeError, ValueError):  # traced axis under jit: use the search bracket
        return None
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        return None
    differences = np.diff(values)
    if not differences[0] > 0 or not np.allclose(differences, differences[0], rtol=1e-12, atol=0):
        return None
    return float(differences[0])


def _interpolation_cell(axes, coordinates, *, xp=np):
    """Shared corners in (Teff, logg, M/H, A_V) order; native axes are prevalidated."""
    if len(axes) != 4 or len(coordinates) != 4:
        raise ValueError("atmosphere interpolation requires four axes and coordinates")
    axes = tuple(xp.asarray(a) for a in axes)
    if xp is np and any(a.dtype.kind not in "iuf" or a.ndim != 1 or not a.size
                        or not np.isfinite(a).all() or np.any(a[1:] <= a[:-1]) for a in axes):
        raise ValueError("atmosphere interpolation axes must be finite and strictly increasing")
    arrays = [xp.asarray(c) for c in coordinates]
    if any(c.dtype.kind not in "iuf" for c in arrays):
        raise ValueError("atmosphere coordinates must be real numeric values, not booleans")
    point = xp.stack(xp.broadcast_arrays(*[c.astype(float) for c in arrays]), axis=-1)
    valid = xp.all(xp.isfinite(point), axis=-1)
    indices, fractions = [], []
    for dim, axis in enumerate(axes):
        value = point[..., dim]
        valid = valid & (value >= axis[0]) & (value <= axis[-1])
        if len(axis) == 1:
            indices.append(xp.zeros_like(value, dtype=int))
            fractions.append(xp.zeros_like(value))
        else:
            step = _affine_step(axis)
            if step is None:
                index = xp.clip(xp.searchsorted(axis, value, side="right")-1, 0, len(axis)-2)
            else:
                quotient = (value-axis[0])/step
                index = xp.floor(xp.where(xp.isfinite(quotient), quotient, 0.)).astype(int)
                index = xp.clip(index, 0, len(axis)-2)
                # Quotient rounding can land one cell off exactly at a knot.
                index = xp.where((index < len(axis)-2) & (axis[index+1] <= value), index+1, index)
                index = xp.where((index > 0) & (axis[index] > value), index-1, index)
            indices.append(index)
            fractions.append((value-axis[index])/(axis[index+1]-axis[index]))
    corners = []
    for corner in product((0, 1), repeat=4):
        weight = xp.ones(point.shape[:-1])
        address = []
        for dim, high in enumerate(corner):
            weight = weight*(fractions[dim] if high else 1-fractions[dim])
            address.append(indices[dim]+(high if len(axes[dim]) > 1 else 0))
        corners.append((tuple(address), weight))
    return valid, corners


def interpolate_photons(axes, table, coordinates, *, xp=np):
    """Shared NumPy/JAX arithmetic; coordinates (K, logg CGS, [M/H] dex, A_V mag).

    Table columns: photon surface flux and three absolute quadratic intensities.
    Missing nodes stay NaN. Zero-weight corners need not exist (exact nodes are
    valid); every positive-weight corner must exist. No clamp or extrapolation.
    """
    valid, corners = _interpolation_cell(axes, coordinates, xp=xp)
    result = xp.zeros(valid.shape+(4,))
    for address, weight in corners:
        values = table[address]
        finite = xp.all(xp.isfinite(values), axis=-1)
        valid = valid & ((weight == 0) | finite)
        result = result+weight[..., None]*xp.where(finite[..., None], values, 0.)
    valid = valid & (result[..., 0] > 0) & (result[..., 1] > 0)
    return xp.where(valid[..., None], result, xp.nan), valid


@dataclass(frozen=True)
class PhotonAtmosphereGrid:
    """One explicitly scaled-solar PHOENIX family, never a rectangular support claim."""

    axes: tuple
    table: np.ndarray
    source: dict
    model_id: str = "phoenix-specint-2011-tess-photon-flux-conserving-quadratic-v1"
    angular: tuple | None = None

    def __post_init__(self):
        if any(np.asarray(v).dtype.kind not in "iuf" for v in (*self.axes, self.table)):
            raise ValueError("atmosphere grids require real numeric arrays, not booleans")
        axes = tuple(np.asarray(a, dtype=float).copy() for a in self.axes)
        table = np.asarray(self.table, dtype=float).copy()
        if (len(axes) != 4 or any(a.ndim != 1 or not a.size or not np.isfinite(a).all()
                                 or np.any(np.diff(a) <= 0) for a in axes)
                or table.shape != tuple(len(a) for a in axes)+(4,)
                or not isinstance(self.source, dict) or not self.source
                or np.any(axes[0] <= 0) or np.any(axes[3] < 0)):
            raise ValueError("invalid atmosphere grid or absent release provenance")
        from .release import V3_RELEASE
        if self.source.get("release") == V3_RELEASE and self.angular is None:
            raise ValueError("v3 photon atmospheres require their original angular profiles")
        finite = np.isfinite(table).all(axis=-1)
        if not np.any(finite) or np.any(~finite & ~np.isnan(table).all(axis=-1)):
            raise ValueError("atmosphere grid has no nodes or a corrupt partial/nonfinite node")
        if np.any(finite & ((table[..., 0] <= 0) | (table[..., 1] <= 0))):
            raise ValueError("nonpositive atmosphere photon light")
        from .passband import quadratic_nonnegative
        # A diagnostic polynomial cannot veto the independently checked tabulated angular model.
        if self.angular is None and np.any(finite & ~quadratic_nonnegative(table)):
            raise ValueError("negative quadratic angular intensity")
        # The quadratic and independently stored disk flux must conserve identical photons.
        reconstructed = np.pi*(table[..., 1]+table[..., 2]/3+table[..., 3]/6)
        if not np.allclose(reconstructed[finite], table[..., 0][finite], rtol=1e-12, atol=0):
            raise ValueError("quadratic intensity and surface photon flux are inconsistent")
        for a in (*axes, table):
            a.setflags(write=False)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "table", table)
        if self.angular is not None:
            from .angular import TABULATED_MODEL_ID, V3_TABULATED_MODEL_ID
            if len(self.angular) != 2 or any(np.asarray(v).dtype.kind not in "iuf" for v in self.angular):
                raise ValueError("angular ray tables require real numeric mu and intensity arrays")
            mu, intensity = (np.asarray(v, dtype=float).copy() for v in self.angular)
            if (mu.shape[:-1] != table.shape[:3] or intensity.shape[:-1] != table.shape[:4]
                    or mu.shape[-1] != intensity.shape[-1] or mu.shape[-1] < 2):
                raise ValueError("angular ray tables do not match the atmosphere grid")
            present = np.isfinite(intensity).all(axis=-1)
            rays_present = np.isfinite(mu).all(axis=-1)
            if (not np.array_equal(present, finite) or np.any(~present & ~np.isnan(intensity).all(axis=-1))
                    or np.any(present & ~rays_present[..., None]) or np.any(intensity[present] < 0)
                    or np.any(np.diff(mu[rays_present], axis=-1) <= 0)
                    or np.any(mu[rays_present, 0] != 0) or np.any(mu[rays_present, -1] != 1)):
                raise ValueError("incomplete/corrupt angular photon profile")
            a, b = mu[..., :-1, None], mu[..., 1:, None]
            values = np.moveaxis(intensity, -1, -2)
            flux = 2*np.pi*np.sum((b-a)*((2*a+b)*values[..., :-1, :]
                                         +(a+2*b)*values[..., 1:, :])/6, axis=-2)
            if not np.allclose(flux[present], table[..., 0][present], rtol=1e-12, atol=0):
                raise ValueError("angular intensity and surface photon flux are inconsistent")
            mu.setflags(write=False)
            intensity.setflags(write=False)
            object.__setattr__(self, "angular", (mu, intensity))
            object.__setattr__(self, "model_id", V3_TABULATED_MODEL_ID
                               if self.source.get("release") == V3_RELEASE else TABULATED_MODEL_ID)

    def evaluate(self, teff_k, logg_cgs, mh_dex, av_mag):
        values, valid = interpolate_photons(self.axes, self.table, (teff_k, logg_cgs, mh_dex, av_mag))
        if not np.all(valid):
            points = np.stack(np.broadcast_arrays(teff_k, logg_cgs, mh_dex, av_mag), axis=-1)
            raise UnsupportedAtmosphereError(self.model_id, points[~valid])
        return values

    def angular_profile(self, teff_k, logg_cgs, mh_dex, av_mag):
        """(...,8,ray) original rays; each cell's extinction pair is summed first."""
        from .angular import interpolate_angular
        if self.angular is None:
            raise ValueError("this compiled atmosphere has no authentic angular profiles")
        mu, intensity, valid = interpolate_angular(
            self.axes, self.angular, (teff_k, logg_cgs, mh_dex, av_mag))
        if not np.all(valid):
            points = np.stack(np.broadcast_arrays(teff_k, logg_cgs, mh_dex, av_mag), axis=-1)
            raise UnsupportedAtmosphereError(self.model_id, points[~valid])
        return mu, intensity

    def to_file(self, path):
        """Atomically publish compiled arrays; interrupted writes preserve the old grid."""
        path = Path(path)
        if path.suffix != ".npz":
            path = Path(str(path)+".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **{f"axis{i}": a for i, a in enumerate(self.axes)},
                                    table=self.table, source=json.dumps(self.source, allow_nan=False),
                                    **({"mu": self.angular[0], "intensity": self.angular[1]}
                                       if self.angular is not None else {}))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def from_file(cls, path, *, expected_model_id, expected_response_id=None,
                  expected_passband_input_id=None):
        """Load explicit compiled photon inputs without silently losing angular rays.

        NPZ contains axis0..axis3, table, JSON source, and mu/intensity together
        for an angular model. The selected identity must match after all physical
        consistency checks; a quadratic-only cache is never upgraded to angular.
        When a caller needs one exact band, the prepared response identity/digest
        must match too: a spectrally different band cannot reuse this grid even
        if the historical model-id label is unchanged.
        """
        from .release import RELEASE, V3_RELEASE

        with np.load(path, allow_pickle=False) as data:
            if not {*(f"axis{i}" for i in range(4)), "table", "source"} <= set(data.files):
                raise ValueError("compiled atmosphere lacks required axes, table or source")
            has_mu, has_intensity = "mu" in data, "intensity" in data
            if has_mu != has_intensity:
                raise ValueError("compiled atmosphere has an incomplete angular representation")
            if data["source"].ndim != 0 or data["source"].dtype.kind not in "US":
                raise ValueError("compiled atmosphere source must be scalar JSON text")
            source = json.loads(data["source"].item())
            if not isinstance(source, dict) or source.get("release") not in (RELEASE, V3_RELEASE):
                raise ValueError("compiled atmosphere has an unsupported physical release")
            if (expected_response_id is not None
                    and source.get("response_id") != expected_response_id):
                raise ValueError(
                    "compiled atmosphere was prepared for response {0!r}, not {1!r}".format(
                        source.get("response_id"), expected_response_id
                    )
                )
            if (expected_passband_input_id is not None
                    and source.get("passband_input_id") != expected_passband_input_id):
                raise ValueError("compiled atmosphere was prepared from a different response table")
            grid = cls(tuple(data[f"axis{i}"] for i in range(4)), data["table"],
                       source,
                       angular=(data["mu"], data["intensity"]) if has_mu else None)
        if grid.model_id != expected_model_id:
            raise ValueError("compiled atmosphere does not match the explicitly selected stellar-light model")
        return grid

    @classmethod
    def from_spectra(cls, spectra, bandpass, av_axis, *, node_failures=None):
        """Stream actual nodes; missing planes stay NaN, never partial AV nodes.

        By default invalid nodes raise. Explicit preparation may retain every
        failed node in ``node_failures`` and continue preparing the others;
        the failed cell stays unavailable, never a population/prior exclusion.
        With that collector, no usable node returns None rather than an empty grid.
        This single-band entry point delegates to the multi-band compiler so the
        two paths share one integration/identity implementation.
        """
        band_id = bandpass.response_id or "band"
        grids = cls.from_spectra_multi(
            spectra, {band_id: bandpass}, av_axis, node_failures=node_failures)
        return grids[band_id]

    @classmethod
    def from_spectra_multi(cls, spectra, bandpasses, av_axis, *, node_failures=None):
        """Compile every selected band over one streamed atmosphere demand.

        Each spectrum is read once and integrated through each band's own photon
        kernel and extinction curve; no band reuses another band's prepared
        table. Keys must be the canonical instrument-qualified response ids when
        more than one band is requested, so a prepared grid cannot be silently
        relabelled. A node that fails in one band is retained as a band-specific
        failure while the other bands keep their usable cells. Missing cells
        stay NaN; nothing is extrapolated or substituted.

        Transit-radius convention (D-77): the stored rays, flux and quadratic
        are those of the r-method-anchored profile, whose unit disc ends at the
        transit-visible limb (the steepest radial intensity drop) instead of
        the spherical model reference circle. Occultation consumers supply
        radius ratios and separations in stellar-radii units; the anchoring is
        what makes those units true for the stored disc. The per-node edge
        record is kept in ``quadratic_errors`` under ``r_method_edge``. A node
        whose profile resolves no spherical limb is a band-specific
        ``photon-compilation-failed`` node failure, not a silently raw ray set.
        """
        from .passband import (PhotonPassband, angular_moment, integrate_photon_intensity,
                               quadratic_reduction, reanchor_r_method_rays)
        from .release import RELEASE, V3_RELEASE
        from .support import release_axes
        from itertools import chain

        if not isinstance(bandpasses, dict) or not bandpasses:
            raise ValueError("multi-band preparation requires a nonempty band mapping")
        for band_id, bandpass in bandpasses.items():
            if not isinstance(band_id, str) or not band_id.strip():
                raise ValueError("prepared band keys must be nonempty response ids")
            if not isinstance(bandpass, PhotonPassband):
                raise ValueError("prepared bands must be PhotonPassband inputs")
            if bandpass.response_id and bandpass.response_id != band_id:
                raise ValueError("prepared band key differs from its response identity")
            if len(bandpasses) > 1 and not bandpass.response_id:
                raise ValueError(
                    "multi-band preparation requires instrument-qualified response ids; "
                    "a bare band label cannot bind the prepared grid")
        spectra = iter(spectra)
        first = next(spectra, None)
        if first is None:
            if node_failures is not None:
                return {band_id: None for band_id in bandpasses}
            raise ValueError("requires actual nodes from one supported PHOENIX release")
        if first.release not in (RELEASE, V3_RELEASE):
            raise ValueError("requires actual nodes from one supported PHOENIX release")
        axes = release_axes(first.release, av_axis)
        state = {
            band_id: {
                "table": np.full(tuple(len(a) for a in axes) + (4,), np.nan),
                "angular": None,
                "diagnostics": [],
            }
            for band_id in bandpasses
        }
        sources, metadata, seen = [], [], set()
        shared_spectra = chain((first,), spectra)
        for s in shared_spectra:
            if s.release != first.release:
                raise ValueError("different PHOENIX releases cannot fill one another's missing nodes")
            index = tuple(int(np.searchsorted(a, v)) for a, v in zip(axes, (s.teff_k, s.logg_cgs, s.mh_dex)))
            if any(i >= len(a) or a[i] != v for a, i, v in zip(
                    axes, index, (s.teff_k, s.logg_cgs, s.mh_dex))):
                raise ValueError("spectrum is not an official SPECINT node")
            if index in seen:
                raise ValueError("duplicate atmosphere node")
            seen.add(index)
            node_used = False
            for band_id, bandpass in bandpasses.items():
                band_state = state[band_id]
                try:
                    # Absolute original rays define this angular model; quadratic suitability is diagnostic only.
                    prepared = [integrate_photon_intensity(
                        s, bandpass, av, require_nonnegative_quadratic=False) for av in axes[3]]
                    if any(mu.shape != prepared[0][2].shape or not np.array_equal(mu, prepared[0][2])
                           for _, _, mu, _, _ in prepared):
                        raise ValueError("one atmosphere node must share its original rays across extinction knots")
                    if band_state["angular"] is not None and prepared[0][2].shape != band_state["angular"][0].shape[-1:]:
                        raise ValueError("atmosphere nodes have incompatible ray counts")
                    # Transit-radius convention (D-77): re-anchor every extinction
                    # knot's rays to its own r-method limb, then derive the stored
                    # flux and quadratic from the same anchored profile so the
                    # moment/quadratic/flux consistency contracts stay exact.
                    anchored_prepared = []
                    for _flux, _coefficients, mu, intensity, _errors in prepared:
                        anchored, edge = reanchor_r_method_rays(mu, intensity)
                        coefficients, errors = quadratic_reduction(
                            mu, anchored, require_nonnegative=False)
                        anchored_prepared.append((
                            2*np.pi*angular_moment(mu, anchored), coefficients,
                            mu, anchored, dict(errors, r_method_edge=edge)))
                    prepared = anchored_prepared
                except ValueError as exc:
                    if node_failures is None:
                        raise
                    node_failures.append(dict(
                        node=[s.teff_k, s.logg_cgs, s.mh_dex], source=s.source,
                        band=band_id, status="photon-compilation-failed", error=str(exc)))
                    continue
                table = band_state["table"]
                for ai, (av, (flux, coefficients, mu, intensity, errors)) in enumerate(zip(axes[3], prepared)):
                    if band_state["angular"] is None:
                        band_state["angular"] = (
                            np.full(table.shape[:3] + mu.shape, np.nan),
                            np.full(table.shape[:4] + mu.shape, np.nan))
                    band_state["angular"][0][index] = mu
                    band_state["angular"][1][index + (ai,)] = intensity
                    table[index + (ai,)] = np.r_[flux, coefficients]
                    band_state["diagnostics"].append(
                        dict(node=[s.teff_k, s.logg_cgs, s.mh_dex, float(av)], **errors))
                node_used = True
            if node_used:
                sources.append(s.source)
                metadata.append(dict(source=s.source, **(s.metadata or {})))
        grids = {}
        for band_id, bandpass in bandpasses.items():
            band_state = state[band_id]
            if not band_state["diagnostics"]:
                if node_failures is None:
                    raise ValueError("no usable atmosphere node for band {0}".format(band_id))
                grids[band_id] = None
                continue
            grids[band_id] = cls(axes, band_state["table"], {
                "release": first.release, "abundance": "scaled-solar AGSS2009",
                "response": bandpass.response_source,
                "response_id": bandpass.response_id,
                "passband_input_id": bandpass.input_id,
                "extinction": bandpass.extinction_source,
                "spectra": sources, "quadratic_errors": band_state["diagnostics"],
                "node_metadata": metadata,
            }, angular=band_state["angular"])
        return grids
