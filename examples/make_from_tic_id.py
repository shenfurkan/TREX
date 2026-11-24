"""Write examples/from_tic_id.ipynb (unexecuted); execute it with nbclient to record outputs.

    python examples/make_from_tic_id.py
    python -m nbclient --execute examples/from_tic_id.ipynb   # or: jupyter execute --inplace ...

The notebook acquires a real TESS target from its TIC identifier, prepares its SPOC light
curve, calibrates the flux uncertainty on the native time series, and runs TREX.
The generated notebook has no executed outputs; a reduced-budget run checks only the software path.
"""

import json
from pathlib import Path

CELLS = [
    ("markdown", """# TREX from a TIC identifier: TOI-181 b

TREX starts from a TIC ID and a list of sectors, the way TRICERATOPS does
(`target(ID, sectors)` -> `calc_probs` -> `FPP`). `target.from_tic` queries the public
archives itself and keeps every response under `output_dir`:

- TIC v8 target row and a Gaia DR3 cone, matched at the TIC J2000 epoch;
- Gaia DR3 astrophysical parameters, StarHorse EDR3 and SHBoost rows for the host and every neighbour;
- the host metallicity from the first available of APOGEE DR17, GALAH DR3, Gaia GSP-Spec and SHBoost;
- the SPOC target-pixel file of each sector (pipeline aperture and WCS), with each source's share of
  the aperture flux from its catalog TESS magnitude through a 0.75-pixel Gaussian PSF
  (Giacalone et al. 2021, AJ 161, 24, Sec. 2.1);
- a TRILEGAL simulation of the background field.

The ephemeris and depth below are the TOI catalogue values for TOI-181.01 (NASA Exoplanet Archive
TOI table, retrieved 2026-11-08).

TREX 1.0 is not on PyPI. Install it from the GitHub source with
`pip install "git+https://github.com/shenfurkan/TREX"`, or run `pip install .` in a clone; the
import is `import trex`. Set `TREX_EXAMPLE_CACHE` to reuse an existing GALAH DR3/MIST cache;
otherwise the first acquisition downloads them under `trex-toi-181/cache`.

This notebook defaults to 1,000,000 draws per scenario and records the actual budget in its
outputs. A reduced `TREX_TIC_DRAWS` override is a software-path check, not production-budget
evidence. This illustrates one target, not a validated real-data recipe. TREX does not validate
or confirm planets."""),
    ("code", """from pathlib import Path
import json
import os

import lightkurve as lk
import numpy as np

from trex.models.noise import calibrate_trex_noise
from trex.photometry.binning import weighted_phase_bins
from trex.target import target

TIC_ID = 76923707
SECTORS = [2]
PERIOD_DAYS = 4.5320502
EPOCH_BTJD = 2458358.1182630 - 2457000.0   # BJD_TDB -> BTJD_TDB
DURATION_DAYS = 2.4927338 / 24.0
DEPTH_PPM = 7809.0577339
N_DRAWS = int(os.environ.get("TREX_TIC_DRAWS", 1000000))
work = Path("trex-toi-181")
cache = Path(os.environ.get("TREX_EXAMPLE_CACHE", work / "cache"))"""),
    ("markdown", """## Native light curve and noise calibration

SPOC 2-minute PDCSAP flux for each requested sector, with the default quality mask. Flux and its
reported error are normalized per sector. The sectors stay on their native BTJD times here:
`calibrate_trex_noise` measures the effective error factor from the out-of-transit residuals before
folding, at the transit timescale set by `DURATION_DAYS` and each sector's FITS-header cadence. It
reports `max(1, standardized scatter) * max(1, beta)` per sector (Pont et al. 2006, MNRAS 373, 231),
and the largest sector factor is applied to every point. That inflated per-point uncertainty is the
single fixed `sigma` every scenario uses. Calibrating on folded points or fitting a separate noise
scale per scenario is not supported.

The default noise mask excludes the primary and a **circular** half-phase window; that assumption is
returned as `secondary_mask` and `event_phase_support`. An eccentric secondary at another phase stays
in the native noise sample unless its measured phase and duration are supplied through
`event_windows`. The half-phase region is still retained in the folded likelihood either way.

The fold keeps the **whole orbit**, not a primary-only or fixed half-phase window. TREX scores
eclipsing-binary and doubled-period scenarios, and an eccentric secondary can fall at any phase, so
the observation must retain every phase the declared models can use. Each phase bin carries its own
width; TREX convolves that width with the exposure, so a broad baseline bin is not treated as a
single 2-minute cadence. Empty phase bins are absent data, not zero flux."""),
    ("code", """curves = lk.search_lightcurve(f"TIC {TIC_ID}", mission="TESS", author="SPOC", exptime=120,
                              sector=SECTORS).download_all(flux_column="pdcsap_flux")
times, fluxes, errors, sector_ids, cadence_days, weights = [], [], [], [], {}, {}
for lc in curves:
    lc = lc.remove_nans().normalize()
    sector = int(lc.meta["SECTOR"])
    cadence_days[sector] = float(lc.meta["TIMEDEL"])
    weights[sector] = float(np.sum(1.0 / np.asarray(lc.flux_err.value) ** 2))
    times.append(np.asarray(lc.time.value))
    fluxes.append(np.asarray(lc.flux.value))
    errors.append(np.asarray(lc.flux_err.value))
    sector_ids.append(np.full(lc.time.value.size, sector, dtype=int))
time_btjd = np.concatenate(times)
flux = np.concatenate(fluxes)
sigma_native = np.concatenate(errors)
sectors = np.concatenate(sector_ids)

noise = calibrate_trex_noise(
    time_btjd, flux, sigma_native, sectors,
    period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD, duration_days=DURATION_DAYS,
    cadence_days_by_sector=cadence_days,
)
sigma = sigma_native * noise["native_uncertainty_factor"]
for sector, record in noise["sectors"].items():
    print(f"sector {sector}: beta={record['beta']:.2f}, white={record['white_scatter_factor']:.2f}, "
          f"cadence={record['cadence_days'] * 86400:.0f} s, bin points={record['bin_size_points']}")
print(f"native effective-error factor {noise['native_uncertainty_factor']:.2f}")
print("secondary mask:", noise["secondary_mask"])

# Keep the whole phase cycle. A primary-only window hides the flat or eclipsed
# half phase that separates EB from EBx2P, and a fixed half-phase window misses
# an eccentric secondary; both discard support the declared models use.
phase = ((time_btjd - EPOCH_BTJD + 0.5 * PERIOD_DAYS) % PERIOD_DAYS) - 0.5 * PERIOD_DAYS
norm = np.median(flux)  # full-orbit baseline, not a window around the transit
flux, sigma = flux / norm, sigma / norm
exptime_days = min(cadence_days.values())
exposure_days = exptime_days
transit_bins = max(2, int(np.ceil(DURATION_DAYS / exposure_days)))
baseline_bins = max(1, int(np.ceil((0.5 * PERIOD_DAYS - 0.5 * DURATION_DAYS) / DURATION_DAYS)))
edges = np.unique(np.concatenate((
    np.linspace(-0.5 * PERIOD_DAYS, -0.5 * DURATION_DAYS, baseline_bins + 1),
    np.linspace(-0.5 * DURATION_DAYS, 0.5 * DURATION_DAYS, transit_bins + 1),
    np.linspace(0.5 * DURATION_DAYS, 0.5 * PERIOD_DAYS, baseline_bins + 1),
)))
binned = weighted_phase_bins(phase, flux, sigma, edges)
time, flux, sigma = (binned[name] for name in ("time_days", "flux", "flux_err"))
bin_integration_days = binned["widths_days"].tolist()
order = np.argsort(time)
time, flux, sigma = time[order], flux[order], sigma[order]
bin_integration_days = [bin_integration_days[index] for index in order]
in_transit = np.abs(time) <= 0.5 * DURATION_DAYS
depth_ppm = (np.median(flux[~in_transit]) - np.median(flux[in_transit])) * 1e6
print(f"{time.size} folded phase bins from {time_btjd.size} native cadences, "
      f"phase span {time.min():.2f}..{time.max():.2f} d, median bin sigma {np.median(sigma) * 1e6:.0f} ppm")
print(f"folded primary depth {depth_ppm:.0f} ppm from the {DEPTH_PPM:.0f} ppm input; "
      f"half-phase bins retained: {int(np.count_nonzero(np.abs(np.abs(time) - 0.5 * PERIOD_DAYS) <= DURATION_DAYS))}")"""),
    ("markdown", """## Background population

`from_tic` submits the field to TRILEGAL in Galactic degrees (`gal_coord=1`, `gc_l_deg`/`gc_b_deg`).
Open `scene/background/trilegal_query.json` to see which population a run used: a web result records
`field_submission: galactic-lb-deg`; when TRILEGAL cannot answer, the same 0.1 deg2 cone is taken
from the TRILEGAL LSST simulation inside its footprint, and the record says
`model: trilegal-lsst-sim-dr2`, `fallback: true` and `fallback_trigger` with the TRILEGAL failure. A
fallback is not treated as current: the next acquisition tries TRILEGAL again. TOI-181 b lies inside
the fallback footprint, so this notebook still acquires a background when the web service is down."""),
    ("code", """work.mkdir(parents=True, exist_ok=True)
(work / "noise.json").write_text(json.dumps(noise, allow_nan=False))
star = target.from_tic(TIC_ID, SECTORS, output_dir=work / "scene", depth_ppm=DEPTH_PPM,
                       cache_dir=cache, sector_weights=weights, exptime_days=exptime_days,
                       bin_integration_days=bin_integration_days, random_seed=181)
star.stars"""),
    ("markdown", """## Probabilities

`calc_probs` runs every scenario at `N_DRAWS` draws per scenario here; the default is
1,000,000, and any `TREX_TIC_DRAWS` override is printed in the output. The per-point `sigma`
already includes the native-time effective-error factor, and the same fixed uncertainty is used by
every scenario.

A real target at this budget can reach TREX's effective-sample-size floor and defer. Such a
deferral refuses to publish an unconverged FPP; it is not a failure of the data. The
full-budget calculation is not guaranteed to converge. A recorded numerical failure is saved
and raised, not relabeled as a scientific deferral."""),
    ("code", """from trex.foundation.errors import TrexEvidenceUnavailableError

probs = None
try:
    star.calc_probs(time, flux, sigma, PERIOD_DAYS, exptime=exptime_days, N=N_DRAWS,
                    escalation_checkpoint_directory=str(work / "escalation"))
except TrexEvidenceUnavailableError as error:
    diagnosis = star.failure_record["operational_diagnosis"]
    if diagnosis["status"] == "failed":
        star.write_outputs(work / "result")
        raise
    print(f"TREX deferred at {N_DRAWS:,} draws per scenario ({diagnosis['category']}).")
    print(diagnosis["next_action"])
    print(error)
else:
    print("draws per scenario:", N_DRAWS)
    print("status:", star.trex_result.status)
    print("FPP =", star.FPP, " NFPP =", star.NFPP)
    probs = star.probs

print(probs if probs is not None else "No probability table: failure.json is under the result directory.")
"""),
    ("code", """star.write_outputs(work / "result")
sorted(path.name for path in (work / "result").iterdir())"""),
]


def main():
    notebook = {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
             **({"execution_count": None, "outputs": []} if kind == "code" else {})}
            for kind, text in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    path = Path(__file__).with_name("from_tic_id.ipynb")
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
