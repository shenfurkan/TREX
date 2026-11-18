# TREX tutorial

This tutorial assumes basic Python and no prior experience with TRICERATOPS. It
has two stages: run the offline synthetic example to see what a TREX run leaves,
then prepare and acquire a real TESS target from its TIC ID. Nothing here
establishes the existence of a planet; every probability is conditional on the
supplied light curve, scene, priors and scenarios.

Install TREX first as described in [Installation](installation.md). The
background vocabulary is in [Concepts](concepts.md).

## 1. What the calculation is

A **light curve** is a series of brightness measurements. A **transit** is a dip
when a body passes in front of a star. A planet can cause a transit, but so can
an **eclipsing binary**: two stars that pass in front of each other. A nearby
eclipsing binary can also put its light into the same photometric aperture, the
pixels summed to make the light curve.

A **false positive** is a signal that resembles a planet transiting the target
star but has another cause. TREX evaluates a set of **scenarios**: specified
explanations for the signal and its stellar surroundings. Examples are a planet
on the target, an eclipsing binary on the target, or an eclipse on a resolved
nearby star. Scenarios missing from the model are not tested.

A **prior** expresses assumptions about scenario weights or their physical
parameters before the light curve is considered. A scenario's **evidence** is
the probability of the data averaged over that scenario's possible parameter
values. The scenario probabilities depend on both the evidence and the prior
weights. TREX estimates the evidence from simulated **draws** of the parameters.
An **effective sample size (ESS)** summarises how many equally weighted draws
would have a similar concentration of weight. Low ESS can mean a few draws
dominate the estimate; high ESS cannot prove that no important region was
missed.

TREX sums the normalised scenario probabilities in two ways:

- **FPP** (false-positive probability) sums every modelled scenario except the
  planet-on-target scenarios `TP`, `PTP` and `DTP`.
- **NFPP** (nearby false-positive probability) is the part of FPP from
  scenarios on resolved nearby stars. It is not the probability of every
  possible blend.

Both are conditional on the supplied light curve, stellar scene, population
models, priors and scenarios. They do not cover every instrumental or
astrophysical explanation, and a small FPP is not a model-independent
probability that a planet exists. TREX does not validate or confirm planets;
its `claim_eligible` field is always `false`. The scenario framework is
described by [Giacalone et al. (2021)](https://doi.org/10.3847/1538-3881/abc6af).

## 2. Run the synthetic example

The offline example uses invented data and the real engine:

```bash
python -m trex.example trex-demo
```

It refuses an existing `trex-demo` directory, so use a new name for a second
run. It creates:

```text
trex-demo/inputs/synthetic-trilegal.csv
trex-demo/inputs/synthetic-aperture-input.json
trex-demo/result/summary.json      # a completed run
trex-demo/result/scenarios.csv     # a completed run
```

On success it prints a status line. If the calculation raises, it retains the
failure record under `result/` and re-raises the exception; a refused run leaves
`failure.json` instead of a summary.

The inputs are in
[the example source](https://github.com/shenfurkan/TREX/blob/main/src/app/example.py).
Its scene is invented:

| Input | Value |
|---|---|
| Target | TIC 100001 (invented), RA 120°, Dec −30°, 1.02 M☉, 0.98 R☉, 5778 K, TESS mag 10, parallax 10 mas, [Fe/H] 0 |
| Sectors | 1 and 2 |
| Neighbours | One at 12 arcsec with mass and radius; one at 14.4 arcsec with no stellar parameters and a negligible aperture share |
| Background | Two synthetic rows over 0.1 deg² |
| Light curve | 80 points over ±0.08 d; flux 0.992 within 0.03 d of the midpoint, 1.0 outside |
| Period and depth | 3.2 d and 8000 ppm |
| Flux uncertainty | `1.0` in normalised flux, deliberately as large as the out-of-transit flux |
| Sector weights | `{1: 3.0, 2: 1.0}` |
| Exposure and bin width | 120 s and 60 s |
| Seed and budget | 424242, 4096 draws per scenario, 4 sub-exposure samples |

The large uncertainty and the low draw count make this a check of the software
path, not a scientific result. Its probabilities are not meaningful for any
star.

The example calls `target.from_scene` with a supplied `TargetScene`. It does not
access an archive and does not exercise `from_tic`.

### Inspect the result

Open `trex-demo/result/summary.json`. The fields that matter first are:

| Field | Meaning |
|---|---|
| `status` | `ok` only when normalisation returned finite FPP and NFPP; `ok` is not a certificate that the scene or model is complete |
| `FPP`, `NFPP` | The two conditional sums above |
| `effective_sample_sizes` | Final ESS per scenario |
| `scenario_monte_carlo_evidence` | First-stage evidence, its error, ESS and status per scenario |
| `scenario_evidence_refinement` | Whether refinement replaced the first-stage estimate, and why |
| `unavailable_scenarios` | Scenarios excluded for a missing input, with reasons |
| `diagnostics` | Named checks and numerical warnings |

A missing refined ESS is not zero. An unavailable scenario is a missing input,
not evidence that the explanation is impossible. A host-excluded neighbour
scenario is a separate physical exclusion with zero probability; it is not an
unavailable input. Read [Outputs](outputs.md) for every field.

`scenarios.csv` has one row per scenario and host: 15 target scenarios plus
three for each resolved neighbour. The `prob` column is the normalised scenario
probability, not FPP. The other columns describe the **representative draw**,
the Monte Carlo draw with the highest likelihood for that scenario. It is a
single sample chosen to show the shape of the model, not a posterior mean,
median or best fit. Do not read one row as the full inference.

If the calculation fails, inspect the exception and `failure.json` when it is
present. Library calls such as `calc_probs` raise and retain failure information
on the target; they do not automatically write a file. Call
`target.write_outputs` explicitly to publish the retained record.

## 3. Prepare a real light curve

TREX does not search for transits. Start from a signal you already have: a
period, an epoch, a depth and a duration measured from a TESS light curve or
taken from a catalogue. The remaining preparation is yours:

| Step | What you provide | TREX role |
|---|---|---|
| Transit search and ephemeris | Period, epoch, depth, duration | None |
| Photometry and quality filtering | Native cadence, sector, quality mask, normalisation | None |
| Noise calibration | One fixed flux uncertainty per point | `calibrate_trex_noise` on native cadence |
| Phase folding | Times in days from the transit midpoint | None |
| Source localisation | Centroid or difference-image evidence, if you have it | No calibrated likelihood; resolved neighbours keep neutral spatial weights |
| High-resolution imaging | Optional measured contrast curve | No imaging is acquired |

The public TOI-181 workflow in
[`examples/from_tic_id.ipynb`](https://github.com/shenfurkan/TREX/blob/main/examples/from_tic_id.ipynb)
is the worked example for the rest of this page. It uses the NASA Exoplanet
Archive TOI catalogue values for TOI-181.01: TIC 76923707, sector 2, period
4.5320502 d, epoch BTJD 18358.1182630, duration 2.4927338 h and depth
7809.0577339 ppm. The notebook is a worked example for the acquisition
sequence; execution budgets and pending outputs are discussed in section 6.

The constants used below are:

```python
TIC_ID = 76923707
SECTORS = [2]
PERIOD_DAYS = 4.5320502
EPOCH_BTJD = 2458358.1182630 - 2457000.0   # BJD_TDB -> BTJD_TDB
DURATION_DAYS = 2.4927338 / 24.0
DEPTH_PPM = 7809.0577339

cadence_days = {2: 120.0 / 86400.0}        # each sector's FITS-header exposure
```

The notebook contains the download and concatenation loop that fills the native
`time_btjd`, `flux`, `sigma_native` and `sectors` arrays and the per-sector
`weights`.

### Noise calibration

Calibrate the uncertainty on the **native, unfolded** time series. Folding
destroys the time adjacency that the binned-RMS statistic needs, so a noise
estimate from folded points is a different measurement and is not supported.

```python
from trex.models.noise import calibrate_trex_noise

noise = calibrate_trex_noise(
    time_btjd, flux, sigma_native, sectors,
    period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD, duration_days=DURATION_DAYS,
    cadence_days_by_sector=cadence_days,
)
sigma = sigma_native * noise["native_uncertainty_factor"]
```

`calibrate_trex_noise` measures a shared effective-error factor from the
out-of-transit residuals at the transit timescale, using the Pont et al. (2006)
binned-RMS statistic. It reports one record per sector with the cadence, bin
size, raw and adopted `beta`, and the effective-error factor. It adopts
`max(1, standardised scatter) * max(1, beta)` per sector and uses the largest
sector factor. A sector that cannot support the estimate raises
`TrexEvidenceUnavailableError`; it does not silently use `beta = 1`.

The resulting fixed `sigma` is used by every scenario. Do not fit a separate
noise scale per scenario. The example saves `noise.json` before scene acquisition,
so a later archive failure does not discard the completed calibration.

For a measured eccentric secondary, add
`event_windows=[{"phase": secondary_phase, "duration_days": secondary_duration,
"role": "eccentric-secondary"}]` to the calibration call. Declared windows
replace the circular phase-0.5 secondary, not the primary. The record labels
which rule was used; all points still enter the likelihood.

### Fold and pass the light curve

```python
from trex.photometry.binning import weighted_phase_bins

phase = ((time_btjd - EPOCH_BTJD + 0.5 * PERIOD_DAYS) % PERIOD_DAYS) - 0.5 * PERIOD_DAYS
norm = np.median(flux)
exptime_days = min(cadence_days.values())
transit_bins = max(2, int(np.ceil(DURATION_DAYS / exptime_days)))
baseline_bins = max(1, int(np.ceil((0.5 * PERIOD_DAYS - 0.5 * DURATION_DAYS) / DURATION_DAYS)))
edges = np.unique(np.concatenate((
    np.linspace(-0.5 * PERIOD_DAYS, -0.5 * DURATION_DAYS, baseline_bins + 1),
    np.linspace(-0.5 * DURATION_DAYS, 0.5 * DURATION_DAYS, transit_bins + 1),
    np.linspace(0.5 * DURATION_DAYS, 0.5 * PERIOD_DAYS, baseline_bins + 1),
)))
binned = weighted_phase_bins(phase, flux / norm, sigma / norm, edges)
time, flux, sigma = binned["time_days"], binned["flux"], binned["flux_err"]
bin_integration_days = binned["widths_days"]
order = np.argsort(time)
time, flux, sigma = time[order], flux[order], sigma[order]
bin_integration_days = bin_integration_days[order]
```

Keep the **whole orbit**: EB and EBx2P scenarios can have eccentric secondaries
away from phase 0.5. Empty bins are omitted, never filled with zero flux.
Pass the actual per-bin `bin_integration_days` together with `exptime_days`;
TREX convolves both boxes instead of treating broad bins as single cadences.

`time` is in **days relative to the transit midpoint**, using the same time
standard as the epoch. `BTJD = BJD − 2457000`; the physical standard (TDB) does
not change. Mixing a BJD epoch with BTJD times shifts the fold.

Exposure and bin width are separate inputs. `exptime_days` is the native
exposure from the FITS header, not a guessed constant. `bin_integration_days` is
the width of a phase-averaged bin and is `0` for individual cadences. If you
binned the folded curve, supply that bin width; TREX then convolves the exposure
and bin windows.

The per-sector inverse-variance sum is also yours to compute, after each
sector's normalization:

```python
weights[sector] = float(np.sum(1.0 / np.asarray(lc.flux_err.value) ** 2))
```

`sector_weights` must cover exactly the sectors in the scene, with finite
positive weights. If the weights do not match the photometry, the dilution
applied to the model will not match the dilution in the data.

## 4. Acquire the scene from a TIC ID

For the public TOI-181 example:

```python
from pathlib import Path
from trex.target import target

work = Path("trex-toi-181")
star = target.from_tic(
    TIC_ID, SECTORS,
    output_dir=work / "scene",
    depth_ppm=DEPTH_PPM,
    cache_dir=work / "cache",
    sector_weights={2: weights[2]},
    exptime_days=cadence_days[2],
    bin_integration_days=bin_integration_days,
    random_seed=181,
)
```

`target.from_tic` queries the public archives, retains the acquisition record
under `output_dir` (shared catalogue releases under `cache_dir`) and returns a
target bound to a validated `TargetScene`. The acquisition needs internet access
and may download large catalogues. It does not search for the signal or calibrate
the noise; those are the steps above.

`depth_ppm` is the measured depth. It decides which faint neighbours need
stellar parameters, and it is bound for the calculation; do not substitute a
visual estimate. `sector_weights` must cover exactly the observed sectors.
`exptime_days` must be the FITS-header exposure, and the same value must be
passed to `calc_probs`. `random_seed` is recorded with the result; setting it
makes a rerun reproducible.

A missing required archive input raises instead of being defaulted. A resolved
neighbour without a stellar parameter is retained with its reason. At calculation
time the outcomes differ: a neighbour whose aperture share cannot produce the
observed depth is `host-excluded` (its three slots stay at `-inf` evidence with
zero probability, and the run can still publish); a neighbour that could host
the depth but lacks mass and radius refuses the whole calculation at preflight
(`TrexEvidenceUnavailableError`). A neighbour without its own metallicity uses
the recorded target-metallicity fallback on the reference-grid path; with
explicit physical stellar light it must carry its own `stellar_atmosphere`. The
archive order, fallbacks and retained records are on [From a TIC ID](from-tic.md).

If you already have a calibrated `TargetScene`, use `target.from_scene` instead
and see [From a scene](from-scene.md). Both paths require the same binding:
sector weights, depth and exposure.

A measured high-resolution contrast curve can be supplied to either path with
`contrast_curve` and `contrast_band="TESS"`. TREX does not acquire imaging
itself; without a curve, unresolved companions are bounded by the Gaia DR2
close-pair limit of 2.2 arcsec.

## 5. Calculate the probabilities

```python
star.calc_probs(
    time, flux, sigma, PERIOD_DAYS,
    exptime=cadence_days[2],
    N=1_000_000,
    escalation_checkpoint_directory=str(work / "escalation"),
)
star.write_outputs(work / "result")
```

The time array is the folded relative-day array from section 3; `flux` and
`sigma` are the normalised values. `sigma` already includes the native-time
noise calibration and is the same fixed uncertainty for every scenario.
`P_orb` is a scalar measured period; period intervals are refused. `exptime`
must equal the exposure bound at `from_tic`; the legacy default `0.00139` days
is a placeholder, not a measurement.

`N` is the number of draws per scenario. The production budget is `1_000_000`.
The default `nsamples=20` sets sub-exposure samples per exposure. A smaller `N`
is a software check, not a scientific result.

`calc_probs` raises `TrexEvidenceUnavailableError` when required evidence is
absent or when a scenario cannot reach the registered ESS floor after
refinement. The target retains `failure_record`; call `write_outputs` to publish
`failure.json`. A deferred run is not an FPP of zero.

### Prepared-run command

`python -m trex.run INPUT_DIRECTORY OUTPUT_DIRECTORY` runs a scene and light
curve that have already been prepared; its input files and checkpoint rules are
on [Prepared runs](prepared-runs.md). Its exit status is a lifecycle contract
for unattended callers. It does not apply to `calc_probs`, `write_outputs` or
the synthetic example.

A prepared run can pause during nested refinement. Keep the checkpoint
directory and the exact inputs; rerunning the same command with the same inputs
and budget continues it. The default dynesty route and the experimental BlackJAX
route both retain native checkpoint state; a resume is bound to the same inputs,
controls and estimator, and BlackJAX restore parity away from the writing host
has not been established.

| Exit | Meaning |
|---:|---|
| `0` | Completed with a reported FPP and NFPP. |
| `1` | Execution failure, including publication failure; the original exception propagates. |
| `2` | Input, checkpoint or output refused before a result existed. A new output retains `failure.json` when publication succeeds; an existing output is left untouched. |
| `3` | Unresolved evidence: `failure.json` or a non-ok `summary.json` retains the reason, and no FPP/NFPP is reported as success. |
| `75` | Checkpoint pause; no new output is created. Rerun the same command with the same inputs and budget. |

A non-zero exit is not an FPP of zero. For exit `75`, keep the checkpoint and
the exact input files. For exit `1`, inspect the raised exception and any
retained failure record.

## 6. Read and interpret the result

Read `summary.json` in this order:

1. `status`. If it is not `ok`, stop and read the failure or the relevant
   diagnostics.
2. `unavailable_scenarios`. Every entry is a missing input, not evidence of
   absence.
3. `effective_sample_sizes` and `scenario_evidence_refinement`. A low final ESS
   or an unresolved refinement means the probability is not trustworthy.
4. `FPP` and `NFPP`, interpreted with the scenario table.
5. `scenarios.csv`, to see which explanations carry the remaining probability.
6. `calculation_provenance`, to check that the scene, seed, exposure, bin width
   and sector weights are the ones you meant.

A useful habit is to ask which scenario dominates. If `EB` dominates, the light
curve is better explained by a stellar eclipse than by a planet. If a
resolved-neighbour eclipse dominates, a nearby star is a better explanation. If
`TP` dominates but `NFPP` is not small, blend explanations are still
competitive. No single row is the full inference.

A completed example run writes `summary.json` and `scenarios.csv`; a refused
calculation writes `failure.json` only, so a failed run cannot be read as an FPP
of zero. Inputs rejected before the calculation may leave no file; the Python
exception carries the validation message. `write_outputs` does not overwrite an
existing non-empty directory.

### Reduced budgets, deferrals and production runs

| Run | Draws per scenario | What it establishes |
|---|---|---|
| `python -m trex.example` | 4,096 | The software path and the files a run leaves |
| Current `from_tic_id.ipynb` | configurable | Regenerated full-orbit example; execution and saved outputs pending |
| Production | 1,000,000 | The intended scientific budget; convergence is still checked |

The current TOI-181 notebook has no saved executed outputs. A reduced-budget
smoke run checks the software path, not scientific convergence; it may defer
with `failure.json` instead of an FPP. No full-budget real-target result is
included in this documentation.

For scale, one full-budget synthetic run (the project's quick-start notebook)
took 18,312 s of wall time on four Kaggle CPUs in the 2026-11-10 record. That is
one host and one code snapshot, not a real-target benchmark; a real target can
take longer or shorter depending on the scene, the number of scenarios and the
backend.

A full-budget run can still be incomplete or unreliable. Check its diagnostics
and assumptions, and do not treat `status: ok` as a planet claim.

## 7. What to record

Keep, alongside the result:

- the TREX version or commit and the installed package versions;
- the light-curve file, its source, sectors, quality mask and cadence;
- the noise-calibration record, including the adopted factor per sector;
- the scene directory, `scene_request.json` and every retained archive response;
- the period, epoch, depth, duration, exposure and bin width you passed;
- the seed, draw budget and `nsamples`;
- the checkpoint directory, if the run paused.

`summary.json` carries the input identity, seed, hash-bound input files, sector
weights, exposure, bin width and depth. It does not replace keeping the original
files.

## 8. Where to go next

- [From a TIC ID](from-tic.md): the acquisition order, archives and limits used
  in section 4.
- [From a scene](from-scene.md): the offline path and measured contrast curves.
- [Inputs](inputs.md) and [Outputs](outputs.md): the exact interface and every
  field.
- [Scenarios](scenarios.md): what each row of `scenarios.csv` means physically.
- [Python API](api.md): every public call with its signature.
- [FAQ](faq.md): concrete failure messages and what to check for each.
- [Documentation overview](README.md): every page in one index.
