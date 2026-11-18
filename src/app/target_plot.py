"""Diagnostic PDF of retained MC likelihood draws, not adopted best-fit curves."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from trex.foundation.physical_constants import PARTS_PER_MILLION


def plot_representative_fits(result, time, flux, sigma, *, save, fname):
    """Show all scenarios in aperture flux, residuals in ppm, and missing draws.

    Times are relative days from transit midpoint. No display binning, smoothing,
    refitting or replacement of unavailable representative parameters is done.
    Supplied uncertainties are exactly those used by the common likelihood.

    Returns the figure when ``save`` is false, leaving it open so a notebook can
    display it. With ``save`` true the PDF is written, the figure is closed and
    ``None`` is returned. A .pdf suffix is appended only when absent.
    """
    import matplotlib.pyplot as plt

    from trex.models.flux_calibration import validate_flux_uncertainty

    time, flux = np.asarray(time), np.asarray(flux)
    sigma = validate_flux_uncertainty(sigma, flux)
    if time.dtype.kind not in "iuf" or time.shape != flux.shape or not np.isfinite(time).all():
        raise ValueError("plot times must be finite real values aligned with the flux")
    models = result.representative_models(time)
    count = len(models)
    if not count:
        raise ValueError("no retained scenarios are available to plot")
    columns = min(3, count)
    rows = math.ceil(count / columns)
    order = np.argsort(time)
    with plt.rc_context({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(3.5 * columns, 3.2 * rows), layout="constrained")
        keep_open = False
        try:
            grid = fig.add_gridspec(2 * rows, columns, height_ratios=[3, 1] * rows)
            for index, (entry, model) in enumerate(zip(result.scenario_fits, models, strict=True)):
                row, col = divmod(index, columns)
                ax = fig.add_subplot(grid[2 * row, col])
                residual = fig.add_subplot(grid[2 * row + 1, col], sharex=ax)
                ax.errorbar(time, flux, yerr=sigma, fmt=".", color="0.4", markersize=2, linewidth=.5)
                ax.set_title(f"{entry['source_id']} / {entry['scenario']}   P={entry['prob']:.3g}")
                ax.set_ylabel("Normalized flux")
                if model is None:
                    ax.text(.03, .05, "No retained MC likelihood draw", transform=ax.transAxes, fontsize=8)
                    residual.text(.03, .5, "Residual unavailable", transform=residual.transAxes, fontsize=8)
                else:
                    ax.plot(time[order], model[order], color="#4477AA", linestyle="-", linewidth=1)
                    residual.errorbar(time, (flux - model) * PARTS_PER_MILLION,
                                      yerr=sigma * PARTS_PER_MILLION, fmt=".", color="0.4",
                                      markersize=2, linewidth=.5)
                    residual.axhline(0, color="#4477AA", linestyle="--", linewidth=.7)
                ax.text(.03, .9, f"Evidence: {entry['evidence_origin']}", transform=ax.transAxes, fontsize=8)
                residual.set(xlabel="Time from transit midpoint [d]", ylabel="Residual [ppm]")
                ax.tick_params(labelbottom=False)
            fig.suptitle("TREX diagnostic only — retained max-lnL MC draws, not posterior fits", fontsize=10)
            if save:
                path = Path(fname)
                fig.savefig(path if path.suffix.lower() == ".pdf" else Path(str(path) + ".pdf"))
            keep_open = not save
        finally:
            if not keep_open:
                plt.close(fig)
    return None if save else fig
