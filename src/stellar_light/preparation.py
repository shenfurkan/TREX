"""Demand-driven official-node acquisition and one coherent photon-grid preparation.

One atmosphere demand can prepare several selected bands. The spectral nodes are
acquired once and streamed through each band's own photon kernel and extinction
curve; no band reuses another band's compiled cells. The preparation record keeps
each band's response identity and input digest, so a later band swap cannot
silently reuse an incompatible grid.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError

import numpy as np

from .interpolation import PhotonAtmosphereGrid
from .passband import PhotonPassband
from .release import acquire_phoenix_intensity, phoenix_intensity_location, phoenix_intensity_cache_path
from .support import release_axes
from trex.acquisition.storage import write_json
from trex.acquisition.transport import AcquisitionError
from trex.runtime.logging import host_logger


class AtmospherePreparation(NamedTuple):
    """Per-band prepared grids; a partial grid remains usable only on its real support."""

    grids: dict
    demand: dict
    node_failures: list

    @property
    def grid(self):
        """The sole grid of a single-band preparation; refuse an implicit band choice."""
        if len(self.grids) != 1:
            raise ValueError(
                "this preparation holds {0} bands; select one explicitly from .grids".format(
                    len(self.grids)
                )
            )
        return next(iter(self.grids.values()))


def _band_map(bands):
    """Normalize a single passband or a canonical band mapping."""
    if isinstance(bands, PhotonPassband):
        band_id = bands.response_id or "band"
        return {band_id: bands}, True
    if not hasattr(bands, "items") or not bands:
        raise ValueError("atmosphere preparation requires a passband or a nonempty band mapping")
    mapping = dict(bands)
    for band_id, bandpass in mapping.items():
        if not isinstance(band_id, str) or not band_id.strip():
            raise ValueError("prepared band keys must be nonempty response ids")
        if not isinstance(bandpass, PhotonPassband):
            raise ValueError("prepared bands must be PhotonPassband inputs")
        if bandpass.response_id and bandpass.response_id != band_id:
            raise ValueError("prepared band key differs from its response identity")
        if len(mapping) > 1 and not bandpass.response_id:
            raise ValueError("multi-band preparation requires instrument-qualified response ids")
    return mapping, len(mapping) == 1


def _available_nodes(grid):
    available = set()
    if grid is not None:
        # Every declared A_V knot must be compiled for the same physical spectrum.
        for index in np.argwhere(np.isfinite(grid.table[..., 0]).all(axis=-1)):
            available.add(tuple(float(grid.axes[i][index[i]]) for i in range(3)))
    return available


def _population_coverage(demand, available):
    coverage = []
    for row in demand["rows"]:
        primary_missing = [n for n in row["primary_nodes"] if tuple(n) not in available]
        secondary_missing = [n for n in row["secondary_nodes"] if tuple(n) not in available]
        bound_missing = [n for n in row.get("bound_nodes", []) if tuple(n) not in available]
        secondary_status = row["secondary_prior_status"]
        coverage.append(dict(index=row["index"], primary_status=(
            "physically-out-of-domain" if row["primary_out_of_domain"] else
            "unavailable-cells" if primary_missing else "available"),
            secondary_status=(secondary_status if secondary_status != "full-EB-and-twin-mass-support" else
                              "physically-out-of-domain" if row["secondary_out_of_domain"] else
                              "unavailable-cells" if secondary_missing else "available-full-mass-support"),
            primary_missing_nodes=primary_missing, secondary_missing_nodes=secondary_missing,
            **({"bound_status": ("physically-out-of-domain" if row["bound_out_of_domain"] else
                                 "unavailable-cells" if bound_missing else "available-full-mass-support"),
                "bound_missing_nodes": bound_missing} if "bound_nodes" in row else {})))
    return coverage


def prepare_atmospheres(demand, bands, directory, *, transport=None):
    """Acquire only declared positive-weight cells and compile each selected band.

    Use population_atmosphere_demand first. Each raw spectrum is read, integrated
    through every selected band and released before the next one; no full spectral
    library is held in RAM. HTTP 404 is distinguished from absent LOCAL cache,
    transport and malformed inputs. Expected node failures survive in
    preparation.json and the returned object; unexpected errors are recorded as
    aborted and re-raised. A node that fails in one band remains usable in the
    others, and its failure names the band.
    Acquired nodes describe retained spectra, not successful photon compilation;
    each receives a separate compiled status per band after grid preparation.
    Grid support is never enlarged by substituting, extrapolating or dropping a
    population row. Preparation does not run any likelihood or evidence sampler.
    """
    if not isinstance(demand, dict) or not {"release", "av_axis", "required_nodes", "rows"} <= demand.keys():
        raise ValueError("atmosphere preparation requires a complete demand record")
    release = demand["release"]
    axes = release_axes(release, demand["av_axis"])
    bandpasses, single_band = _band_map(bands)
    if not isinstance(demand["required_nodes"], (list, tuple)) or not isinstance(demand["rows"], (list, tuple)):
        raise ValueError("atmosphere demand nodes and rows must be sequences")
    for node in demand["required_nodes"]:
        if not isinstance(node, (list, tuple)) or len(node) != 3:
            raise ValueError("atmosphere demand nodes must have three official coordinates")
        phoenix_intensity_location(*node, release=release)
    required_fields = {"index", "primary_nodes", "secondary_nodes", "primary_out_of_domain",
                       "secondary_out_of_domain", "secondary_prior_status"}
    for row in demand["rows"]:
        if (not isinstance(row, dict) or not required_fields <= row.keys()
                or ("bound_nodes" in row and "bound_out_of_domain" not in row)):
            raise ValueError("atmosphere demand has an incomplete population support row")
        for field in ("primary_nodes", "secondary_nodes", "bound_nodes"):
            nodes = row.get(field, [])
            if not isinstance(nodes, (list, tuple)):
                raise ValueError("atmosphere demand row nodes must be sequences")
            for node in nodes:
                if not isinstance(node, (list, tuple)) or len(node) != 3:
                    raise ValueError("atmosphere demand nodes must have three official coordinates")
                phoenix_intensity_location(*node, release=release)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    failures, acquired = [], []
    report = dict(release=release, started_utc=datetime.now(timezone.utc).isoformat(),
                  required_nodes=demand["required_nodes"], acquired_nodes=acquired,
                  node_failures=failures, bands={})
    sampling_rows = demand.get("node_sampling", [])
    if not isinstance(sampling_rows, (list, tuple)):
        raise ValueError("atmosphere demand has an invalid sampling declaration")
    sampling = {}
    for row in sampling_rows:
        if (not isinstance(row, dict) or not {"node", "sampling"} <= row.keys()
                or not isinstance(row["node"], (list, tuple)) or len(row["node"]) != 3):
            raise ValueError("atmosphere demand has an invalid sampling declaration")
        phoenix_intensity_location(*row["node"], release=release, sampling=row["sampling"])
        sampling[tuple(row["node"])] = row["sampling"]

    def save():
        write_json(directory/"preparation.json", report)

    def spectra():
        for node in demand["required_nodes"]:
            selected_sampling = sampling.get(tuple(node), "50A")
            _, url = phoenix_intensity_location(*node, release=release, sampling=selected_sampling)
            path = phoenix_intensity_cache_path(*node, directory, release=release, sampling=selected_sampling)
            cached = path.is_file() and path.with_suffix(".retrieval.json").is_file()
            try:
                spectrum = acquire_phoenix_intensity(*node, directory, release=release,
                                                    sampling=selected_sampling, transport=transport)
            except (HTTPError, URLError, OSError, ValueError, AcquisitionError) as exc:
                failure = getattr(exc, "failure", None)
                status = ("official-file-not-found" if (isinstance(exc, HTTPError) and exc.code == 404
                                                       or failure is not None and failure.http_status == 404)
                          else "transport-failed" if isinstance(exc, (HTTPError, URLError, AcquisitionError))
                          else "local-io-failed" if isinstance(exc, OSError)
                          else "invalid-retained-or-official-node")
                failures.append(dict(node=list(node), url=url, status=status,
                    sampling=selected_sampling, local_cache_present=cached,
                    attempted_utc=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(exc).__name__}: {exc}",
                    **({"transport_failure": failure.to_manifest()} if failure is not None else {})))
                host_logger(__name__).warning("PHOENIX atmosphere node unavailable", exc_info=True,
                                             extra={"event": "stellar_light.preparation"})
                save()
                if status == "local-io-failed":
                    raise
            else:
                acquired.append(dict(node=list(node), sampling=selected_sampling, local_cache_present=cached, url=url))
                yield spectrum

    save()
    host_logger(__name__).info("PHOENIX atmosphere preparation started", extra={"event": "stellar_light.preparation"})
    stream = spectra()
    try:
        grids = PhotonAtmosphereGrid.from_spectra_multi(
            stream, bandpasses, axes[3], node_failures=failures)
    except BaseException as exc:
        report.update(status="aborted", error_type=type(exc).__name__,
                      stopped_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise
    available = {band_id: _available_nodes(grid) for band_id, grid in grids.items()}
    for node in acquired:
        compiled = {band_id: tuple(node["node"]) in nodes
                    for band_id, nodes in available.items()}
        node["compiled"] = next(iter(compiled.values())) if single_band else compiled
    overall = []
    for band_id, grid in grids.items():
        bandpass = bandpasses[band_id]
        coverage = _population_coverage(demand, available[band_id])
        # Acquisition failures carry no band and leave every band incomplete;
        # a compilation failure names its band and must not mark the others partial.
        band_failures = [failure for failure in failures
                         if failure.get("band") in (None, band_id)]
        status = ("unavailable" if grid is None
                  else "partial" if band_failures else "completed")
        overall.append(status)
        report["bands"][band_id] = dict(
            response_id=bandpass.response_id,
            response_source=bandpass.response_source,
            passband_input_id=bandpass.input_id,
            model_id=None if grid is None else grid.model_id,
            status=status,
            compiled_nodes=len(available[band_id]),
            population_support=coverage,
        )
        if single_band:
            # Historical single-band readers keep their top-level fields; the
            # per-band record above is authoritative for every preparation.
            report.update(grid_model_id=None if grid is None else grid.model_id,
                          band_id=band_id, population_support=coverage)
    if all(status == "unavailable" for status in overall):
        status = "unavailable"
    elif all(status == "completed" for status in overall):
        status = "completed"
    else:
        status = "partial"
    report.update(status=status,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    save()
    host_logger(__name__).info("PHOENIX atmosphere preparation completed",
                              extra={"event": "stellar_light.preparation",
                                     "count": sum(len(v) for v in available.values()),
                                     "bands": sorted(grids)})
    return AtmospherePreparation(grids, demand, failures)
