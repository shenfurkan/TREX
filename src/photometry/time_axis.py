"""Explicit already-barycentric TDB offsets without collapsing split Julian dates.

Reference rebasing is arithmetic, not a UTC/TDB conversion or a light-travel
correction. Adapters must establish those physical conventions first.
"""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta


def _tdb_reference(value):
    if (not isinstance(value, Time) or not value.isscalar or value.scale != "tdb"
            or bool(value.mask) or not np.isfinite([value.jd1, value.jd2]).all()):
        raise ValueError("time reference must be one finite, unmasked TDB Time")
    return value


def _days(value, label):
    if isinstance(value, u.Quantity) or np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must be unmasked real numbers explicitly in days")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be real numbers, not boolean or text")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite days")
    return array


def rebase_bjd_tdb(time_days, reference: Time, output_reference: Time) -> np.ndarray:
    """Express already-barycentric TDB offsets at another TDB JD reference.

    ``time_days`` is a nonempty finite 1-D real array [days]. Both references
    are scalar, finite, unmasked ``Time`` objects explicitly in TDB. Retain
    their two-part JD representation: never form full-JD floats before
    subtracting. Identical references preserve the original float64 offsets
    bit for bit. Otherwise only the final relative days are rounded to float64;
    precision already lost in input full-JD floats cannot be recovered.

    No time-scale, observer, target or barycentric correction is performed.
    Invalid arrays/references or unrepresentable output raise ValueError.
    Raw-product readers must retain rejected rows separately before calling.
    """
    values = _days(time_days, "time offsets")
    if values.ndim != 1 or not values.size:
        raise ValueError("time offsets must be a nonempty 1-D array")
    reference, output_reference = map(_tdb_reference, (reference, output_reference))
    if reference.jd1 == output_reference.jd1 and reference.jd2 == output_reference.jd2:
        return values
    result = (reference + TimeDelta(values, format="jd") - output_reference).to_value(u.day)
    if not np.isfinite(result).all():
        raise ValueError("rebased time offsets cannot be represented finitely")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class TimeAxis:
    """One retained chronological stream of already-barycentric TDB stamps.

    ``time_days`` [days] has N>=1 finite, strictly increasing native stamps;
    ``reference`` is their scalar split-JD TDB origin. ``exposure_days`` is
    finite nonnegative scalar/per-row boxcar elapsed duration, NOT photon
    accumulation, revisit cadence, stacked live windows or ramp response.
    Zero explicitly describes the instantaneous mathematical limit.
    ``stamp`` declares ``start``, ``mid`` or ``end`` of that boxcar.

    ``correction`` must explicitly be ``barycentric``. Observer/target/source
    are required provenance labels, not coordinates sufficient for a new
    correction. No UTC, HJD or uncorrected input is converted here. Missing
    conventions, masked/nonfinite data and invalid chronology raise ValueError;
    adapters own row masking and must not silently drop rows in this contract.
    Arrays and the copied reference are read-only. Distinct streams may have
    identical stamps; chronology is checked within this stream only.
    """

    time_days: np.ndarray
    reference: Time
    exposure_days: np.ndarray
    stamp: str
    correction: str
    observer: str
    target: str
    source: str

    def __post_init__(self):
        reference = _tdb_reference(self.reference).copy()
        reference.writeable = False
        time = _days(self.time_days, "time offsets")
        exposure = _days(self.exposure_days, "exposure duration")
        if time.ndim != 1 or not time.size or np.any(time[1:] <= time[:-1]):
            raise ValueError("time axis needs strictly increasing nonempty native stamps")
        if exposure.ndim > 1 or (exposure.ndim == 1 and exposure.shape != time.shape) or np.any(exposure < 0):
            raise ValueError("exposure duration must be nonnegative scalar or one value per stamp")
        if not isinstance(self.stamp, str) or self.stamp not in ("start", "mid", "end"):
            raise ValueError("time stamp must explicitly be start, mid or end")
        if not isinstance(self.correction, str) or self.correction != "barycentric":
            raise ValueError("time axis requires an explicit already-barycentric correction")
        for name in ("observer", "target", "source"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"time axis requires an explicit {name}")
        exposure = np.broadcast_to(exposure, time.shape).copy()
        time.flags.writeable = exposure.flags.writeable = False
        object.__setattr__(self, "time_days", time)
        object.__setattr__(self, "exposure_days", exposure)
        object.__setattr__(self, "reference", reference)
        self.relative_midpoints(reference)

    def relative_midpoints(self, reference: Time) -> np.ndarray:
        """Return increasing float64 midpoint offsets [days] at a TDB origin.

        Native stamps remain unchanged. Reject variable-duration stamp shifts
        or an excessively remote origin that destroys midpoint chronology.
        Only the declared boxcar anchor is shifted; no geometric correction.
        """
        values = rebase_bjd_tdb(self.time_days, self.reference, reference)
        with np.errstate(over="ignore", invalid="ignore"):
            values += {"start": 0.5, "mid": 0.0, "end": -0.5}[self.stamp] * self.exposure_days
        if not np.isfinite(values).all() or np.any(values[1:] <= values[:-1]):
            raise ValueError("midpoint offsets must remain finite and strictly increasing")
        return values
