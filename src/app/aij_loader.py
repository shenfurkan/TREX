"""Import AstroImageJ differential aperture photometry without detrending it."""

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from astropy.time import Time
import numpy as np

from trex.photometry.baseline import GaussianBaseline
from trex.photometry.exposure import ExposureResponse
from trex.photometry.observations import Observation, exposure_lineage
from trex.photometry.time_axis import TimeAxis, _days, _tdb_reference

_COLUMNS = ("bjd_tdb", "rel_flux_t1", "rel_flux_err_t1", "airmass", "fwhm")


def read_aij_observation(
    path, *, dataset_id, segment_id, band_id, response_id, reference,
    observer, target, stamp, exposure_days, exposure_source, nsamples,
    aperture_id, source_shares, lineage, flux_scale, uncertainty_source,
    prior_mean, prior_covariance, baseline_mode, baseline_source,
):
    """Return an Observation with a Gaussian prior on raw (airmass, FWHM).

    Read a tab-delimited AstroImageJ table or a comma-delimited `.csv`. Columns
    are matched case-insensitively. BJD_TDB must be full, already barycentric
    Julian days; explicitly give its exposure anchor and the shared scalar TDB
    reference used by the ObservationSet. Flux and errors are divided by the
    caller's fixed quiescent differential-flux level (1 for normalized data).
    Record its provenance in ``uncertainty_source``. No rows are masked, and
    neither the scale nor its uncertainty is inferred from these measurements.

    Boxcar durations [days], lineage and any per-row aperture shares are in
    file order; all follow the chronological sort. The independently sourced
    prior is on the two uncentered columns in that order (airmass dimensionless,
    FWHM in the export's units), held fixed within each exposure. No intercept
    is added and no prior is estimated from the table.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="," if path.suffix.lower() == ".csv" else "\t", strict=True)
        header = [name.strip().lower() for name in next(reader, ())]
        if any(header.count(name) != 1 for name in _COLUMNS):
            raise ValueError(f"{path}: expected exactly one of each AstroImageJ column {_COLUMNS}")
        columns = [header.index(name) for name in _COLUMNS]
        dates, measurements = [], []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}:{reader.line_num}: AstroImageJ row does not match header")
            date, *values = (row[index].strip() for index in columns)
            try:
                numeric = [float(value) for value in (date, *values)]
            except ValueError as exc:
                raise ValueError(f"{path}:{reader.line_num}: invalid AstroImageJ measurement") from exc
            if not all(map(math.isfinite, numeric)) or numeric[0] < 2_400_000:
                raise ValueError(f"{path}:{reader.line_num}: expected finite measurements and full BJD_TDB")
            dates.append(date)
            measurements.append(numeric[1:])
    if len(dates) < 2:
        raise ValueError(f"{path}: Gaussian baseline requires at least two measurements")

    scale = _days(flux_scale, "quiescent differential flux")
    if scale.ndim != 0 or scale <= 0:
        raise ValueError("flux_scale must be one positive quiescent differential flux")
    if not isinstance(source_shares, Mapping):
        raise ValueError("source_shares must map exact source IDs to known fractions or None")
    reference = _tdb_reference(reference)
    # Reading decimal strings avoids losing cadence precision to full-JD floats.
    offsets = (Time(dates, format="jd", scale="tdb") - reference).to_value("jd")
    order = np.argsort(offsets)

    def ordered(value, name):
        array = _days(value, name)
        if array.ndim == 0:
            return array
        if array.shape != order.shape:
            raise ValueError(f"{name} must be scalar or aligned to AstroImageJ file rows")
        return array[order]

    axis = TimeAxis(offsets[order], reference, ordered(exposure_days, "exposure duration"),
                    stamp, "barycentric", observer, target, f"AstroImageJ BJD_TDB: {path.name}")
    flux, sigma, *basis = np.asarray(measurements)[order].T
    baseline = GaussianBaseline(axis.relative_midpoints(reference), np.column_stack(basis),
                                prior_mean, prior_covariance, baseline_mode, baseline_source)
    origins = exposure_lineage(lineage, len(dates))
    return Observation(
        dataset_id=dataset_id, segment_id=segment_id, band_id=band_id, response_id=response_id,
        time_axis=axis, flux=flux / scale, sigma=sigma / scale,
        exposure_response=ExposureResponse.boxcar(axis.exposure_days, nsamples=nsamples, source=exposure_source),
        aperture_id=aperture_id,
        source_shares={key: None if value is None else ordered(value, "aperture source share")
                       for key, value in source_shares.items()},
        lineage=tuple(origins[index] for index in order), raw_rows=order, raw_count=len(dates),
        uncertainty_source=uncertainty_source, baseline=baseline,
    )
