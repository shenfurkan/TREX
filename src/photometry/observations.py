"""Retained aperture observations; no catalog, band conversion or noise estimation."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np
from astropy.time import Time

from trex.models.flux_calibration import validate_flux_uncertainty
from trex.photometry.baseline import GaussianBaseline
from trex.photometry.exposure import ExposureResponse
from trex.photometry.time_axis import TimeAxis, _days, _tdb_reference


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty string")
    return value


def exposure_lineage(values, row_count):
    """Validate canonical original-exposure sets without inventing or coercing IDs.

    One nonempty collection of nonempty string identities is required per row.
    Dataset names, filenames and derived-row indices are not substitutes for
    original identities. ObservationSet checks repeated exposures across rows.
    """
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise ValueError("lineage requires original-exposure identities for every row")
    try:
        rows = tuple(values)
        if len(rows) != row_count:
            raise ValueError("lineage must describe exactly every observation row")
        result = []
        for row in rows:
            if isinstance(row, (str, bytes, Mapping)):
                raise ValueError("lineage rows must be nonempty collections of original-exposure identities")
            origins = tuple(row)
            if not origins:
                raise ValueError("lineage rows must contain original-exposure identities")
            for origin in origins:
                _label(origin, "original exposure identity")
            result.append(frozenset(origins))
    except TypeError as exc:
        raise ValueError("lineage must contain collections of original-exposure identities") from exc
    return tuple(result)


@dataclass(frozen=True, eq=False)
class Observation:
    """One retained chronological stream in normalized aperture flux.

    ``band_id`` is the exact instrument-qualified response identity, not a
    bare filter letter; ``response_id`` binds its actual response/model asset
    edition. No passband registry is imported or guessed here.
    ``source_shares`` maps scene source IDs to scalar/per-row quiescent
    aperture fractions. Missing/None stays unknown, never zero. For an
    unresolved binary the source denotes the combined quiescent pair.

    Each ``lineage`` row is a nonempty set of canonical original exposure
    identities, independent of derived filenames/dataset IDs. Derived averages
    list all their constituent exposures. Overlap is refused by ObservationSet
    until an explicit shared-photon likelihood is supported.
    ``raw_rows`` maps retained rows to the original product's ``raw_count``;
    rejection/masking belongs to the producer and is never repeated here.

    TimeAxis supplies midpoints; response offsets [TDB days] are relative to
    those midpoints. Non-box responses require native midpoint stamps, since
    this boundary does not infer a stack/ramp's anchor from elapsed duration.
    Reported scalar/per-row sigma is fractional aperture flux under the
    caller's explicitly sourced fixed Gaussian uncertainty model, not
    automatically recalibrated from a short auxiliary curve.
    """

    dataset_id: str
    segment_id: str
    band_id: str
    response_id: str
    time_axis: TimeAxis
    flux: np.ndarray
    sigma: np.ndarray
    exposure_response: ExposureResponse
    aperture_id: str
    source_shares: Mapping
    lineage: tuple
    raw_rows: np.ndarray
    raw_count: int
    uncertainty_source: str
    baseline: GaussianBaseline | None = None

    def __post_init__(self):
        for name in ("dataset_id", "segment_id", "band_id", "response_id", "aperture_id", "uncertainty_source"):
            _label(getattr(self, name), name)
        if "/" not in self.band_id or any(c.isspace() for c in self.band_id):
            raise ValueError("band_id must identify the instrument-qualified response, not a bare filter")
        if not isinstance(self.time_axis, TimeAxis) or not isinstance(self.exposure_response, ExposureResponse):
            raise ValueError("observation requires a TimeAxis and explicit ExposureResponse")
        time = self.time_axis.relative_midpoints(self.time_axis.reference)
        self.exposure_response.evaluation_times(time)
        if self.time_axis.stamp != "mid":
            expected = ExposureResponse.boxcar(self.time_axis.exposure_days,
                                               nsamples=self.exposure_response.weights.shape[-1],
                                               source="anchor check")
            if (not np.array_equal(expected.offset_days, self.exposure_response.offset_days)
                    or not np.array_equal(expected.weights, self.exposure_response.weights)):
                raise ValueError("non-box temporal responses require explicit native midpoint stamps")
        flux = _days(self.flux, "fractional aperture flux")
        if flux.shape != time.shape:
            raise ValueError("flux must align with retained time stamps")
        sigma = np.broadcast_to(validate_flux_uncertainty(_days(self.sigma, "fractional aperture uncertainty"), flux),
                                time.shape).copy()
        if (isinstance(self.raw_count, (bool, np.bool_)) or not isinstance(self.raw_count, Integral)
                or self.raw_count < time.size):
            raise ValueError("raw_count must be an integer covering every retained row")
        rows = np.asarray(self.raw_rows)
        if (np.ma.isMaskedArray(self.raw_rows) or rows.dtype.kind not in "iu" or rows.shape != time.shape
                or np.any(rows < 0) or np.any(rows >= self.raw_count) or np.unique(rows).size != rows.size):
            raise ValueError("raw_rows must be unique aligned integer indices into the original product")
        rows = rows.copy()
        if not isinstance(self.source_shares, Mapping):
            raise ValueError("source_shares must map exact source IDs to known fractions or None")
        shares = {}
        total = np.zeros(time.size)
        for source, value in self.source_shares.items():
            _label(source, "source ID")
            if value is None:
                shares[source] = None
                continue
            fraction = _days(value, "aperture source share")
            if fraction.ndim > 1 or (fraction.ndim == 1 and fraction.shape != time.shape):
                raise ValueError("source share must be scalar or aligned to every retained row")
            fraction = np.broadcast_to(fraction, time.shape).copy()
            if np.any((fraction < 0) | (fraction > 1)):
                raise ValueError("aperture source share must lie in [0,1]")
            total += fraction
            fraction.flags.writeable = False
            shares[source] = fraction
        if np.any(total > 1 + 16 * np.finfo(float).eps):
            raise ValueError("known source shares exceed their common aperture denominator")
        lineage = exposure_lineage(self.lineage, time.size)
        if self.baseline is not None and not isinstance(self.baseline, GaussianBaseline):
            raise ValueError("baseline must be an explicit proper GaussianBaseline prior")
        flux.flags.writeable = sigma.flags.writeable = rows.flags.writeable = False
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "raw_rows", rows)
        object.__setattr__(self, "source_shares", MappingProxyType(shares))
        object.__setattr__(self, "lineage", lineage)

    def __reduce__(self):
        # Spawned inference workers cannot pickle mappingproxy directly.
        return self.from_dict, (self.to_dict(),)

    def to_dict(self):
        """JSON-compatible physical inputs; native reference remains split."""
        axis = self.time_axis
        baseline = None if self.baseline is None else self.baseline.to_dict()
        return {
            **{name: getattr(self, name) for name in (
                "dataset_id", "segment_id", "band_id", "response_id", "aperture_id",
                "uncertainty_source")},
            "raw_count": int(self.raw_count),
            "time_axis": {"time_days": axis.time_days.tolist(), "reference_jd1": float(axis.reference.jd1),
                          "reference_jd2": float(axis.reference.jd2), "exposure_days": axis.exposure_days.tolist(),
                          **{name: getattr(axis, name) for name in ("stamp", "correction", "observer", "target", "source")}},
            "flux": self.flux.tolist(), "sigma": self.sigma.tolist(),
            "exposure_response": {"offset_days": self.exposure_response.offset_days.tolist(),
                                  "weights": self.exposure_response.weights.tolist(),
                                  "source": self.exposure_response.source},
            "source_shares": {key: None if value is None else value.tolist()
                              for key, value in self.source_shares.items()},
            "lineage": [sorted(row) for row in self.lineage], "raw_rows": self.raw_rows.tolist(),
            "baseline": baseline,
        }

    @classmethod
    def from_dict(cls, payload):
        """Revalidate an exported observation rather than trusting stored arrays."""
        values = dict(payload)
        native = dict(values.pop("time_axis"))
        reference = Time(native.pop("reference_jd1"), native.pop("reference_jd2"), format="jd", scale="tdb")
        values["time_axis"] = TimeAxis(reference=reference, **native)
        values["exposure_response"] = ExposureResponse(**values["exposure_response"])
        if values.get("baseline") is not None:
            values["baseline"] = GaussianBaseline(**values["baseline"])
        return cls(**values)


@dataclass(frozen=True, eq=False)
class ObservationSet:
    """Independent retained streams at one split-TDB reference.

    Ragged/non-simultaneous coverage and repeated timestamps in different
    streams are legal. Reused original photons, even under new filenames or
    IDs, are not independent data and raise ValueError. A shared instrumental
    Gaussian prior is a separate likelihood choice, not permission to reuse
    overlapping photon measurements.
    """

    observations: tuple[Observation, ...]
    reference: Time

    def __post_init__(self):
        reference = _tdb_reference(self.reference).copy()
        reference.writeable = False
        observations = tuple(self.observations)
        if not observations or any(not isinstance(row, Observation) for row in observations):
            raise ValueError("observation set needs at least one retained Observation")
        identities, seen = set(), set()
        for row in observations:
            if row.dataset_id in identities:
                raise ValueError("dataset IDs must be unique within the observation set")
            identities.add(row.dataset_id)
            row.time_axis.relative_midpoints(reference)
            for origins in row.lineage:
                if seen.intersection(origins):
                    raise ValueError("overlapping original exposure lineage cannot count as independent evidence")
                seen.update(origins)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "observations", observations)

    def to_dict(self):
        return {"version": 1, "reference_jd1": float(self.reference.jd1),
                "reference_jd2": float(self.reference.jd2),
                "observations": [row.to_dict() for row in self.observations]}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("version") != 1:
            raise ValueError("unsupported observation-set version")
        return cls(tuple(Observation.from_dict(row) for row in payload["observations"]),
                   Time(payload["reference_jd1"], payload["reference_jd2"], format="jd", scale="tdb"))
