# Prepared runs: `python -m trex.run`

This page answers the question "how do I run a target from the command line,
without writing a Python script around TREX?" It documents the input files,
output and checkpoint rules, resume behaviour and exit codes.

This path is for a scene and light curve that have already been prepared and
validated. It does not acquire anything from a TIC ID and does not calibrate the
light-curve noise.

## Command

```bash
python -m trex.run /path/to/prepared-input /path/to/result --draws 1000000
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--draws` | `1000000` | Draws per scenario. Below 1,000,000 is for software tests only |
| `--nsamples` | `20` | Exposure supersampling |
| `--escalation-checkpoint-directory` | `OUTPUT_DIRECTORY.escalation` | Recovery directory for a paused refinement |

The command prints `status=... FPP=... NFPP=...; see OUTPUT_DIRECTORY` on
success, or a pause message and the checkpoint path. A completed non-`ok`
result prints `status=...; no complete FPP/NFPP pair was reported`. A refusal
and a scientific deferral are reported on standard error. A completed result is
not a validation or confirmation; every result carries `claim_eligible: false`.

The engine also reads these environment variables:

| Variable | Effect |
|---|---|
| `TREX_NESTED_ESTIMATOR` | `dynesty` (default) or the experimental `blackjax`; an unknown value is refused |
| `TREX_BLACKJAX_PRECOND` | Experimental blackjax directions: `eye` (default), `pilot` or `split` |
| `TREX_BLACKJAX_BATCH_INIT` | `1` enables padded, sequential-equivalent initialization; `0` (default) keeps the sequential loop |
| `TREX_BLACKJAX_W_MARCH` | `4` or `8` enable speculative batched stepping-out probes; `0` (default) disables them |
| `TREX_IMPORTANCE_PROPOSAL` | `kernel-mixture` (default) or experimental `probit-gaussian` |
| `TREX_ESCALATION_WALL_SECONDS` | Per-session refinement ceiling; default 540 minutes, capped at 540 minutes |
| `TREX_ESCALATION_DEADLINE_MONOTONIC` | An earlier host monotonic deadline; never persisted |

These limits change when a pause happens, not the scientific draw budget.

## The input directory

Both the legacy single-curve format (version 1) and the joint-observation
format ([version 2](#version-2-joint-observations)) require these three files:

```text
INPUT_DIRECTORY/
├── trex-input.json
├── arrays.npz
└── background.csv
```

The input path must be a real directory, not a symlink, and each of the three
named paths must be a regular, non-symlink file. The names are literal; the
loader does not search for alternatives. Version 2 also requires the band
assets named by its manifest. When the scene JSON does not
carry `background_population_provenance`, the scene may read a
`trilegal_query.json` beside `background.csv` to recover the TRILEGAL
LSST-simulation fallback record.

### `trex-input.json`

The version-1 JSON object (`version` omitted or set to `1`) has three required keys:

| Key | Required | Content |
|---|---|---|
| `scene` | yes | the `TargetScene` constructor payload; all fields are listed in [From a scene](from-scene.md#a-valid-scene) |
| `binding` | yes | the `target.from_scene` keyword payload; `sector_weights`, `depth_ppm` and `exptime_days` are required |
| `period_days` | yes | scalar orbital period [days], finite and positive |

Other top-level keys in version 1 are ignored. Inside `scene` and `binding`, unknown keys are
passed to `TargetScene` or `bind_scene` and are refused rather than ignored.

A complete minimal payload:

```json
{
  "scene": {
    "tic_id": 100001,
    "ra_deg": 120.0,
    "dec_deg": -30.0,
    "M_s_Msun": 1.02,
    "R_s_Rsun": 0.98,
    "Teff_K": 5778.0,
    "Tmag": 10.0,
    "plx_mas": 10.0,
    "feh_dex": 0.0,
    "metallicity_provenance": {
      "source": "caller-supplied",
      "quantity": "[Fe/H]",
      "value_dex": 0.0,
      "uncertainty_dex": 0.1,
      "citation": "the paper or catalogue this value came from"
    },
    "sectors": [1, 2],
    "contrast_separations": [0.1, 1.0, 2.0],
    "contrast_values": [1.0, 3.0, 5.0],
    "contrast_provenance": null,
    "resolved_neighbors": [
      {
        "source_id": "100002",
        "M_s": 0.82,
        "R_s": 0.79,
        "delta_mag": 1.5,
        "separation_arcsec": 12.0,
        "aperture_flux_fractions": [
          {"sector": 1, "fraction": 0.1499},
          {"sector": 2, "fraction": 0.1999}
        ]
      }
    ],
    "target_aperture_flux_fractions": [
      {"sector": 1, "fraction": 0.85},
      {"sector": 2, "fraction": 0.80}
    ],
    "N_background": 2,
    "trilegal_cache": "background.csv",
    "background_sha256": null,
    "background_area_sq_deg": 0.1,
    "unresolved_resolution_limit_arcsec": 2.2,
    "localization_evidence": null,
    "neighbor_parameter_evaluations": {}
  },
  "binding": {
    "sector_weights": {"1": 3.0, "2": 1.0},
    "depth_ppm": 8000.0,
    "exptime_days": 0.001388888888888889,
    "random_seed": 424242
  },
  "period_days": 3.2
}
```

`scene` accepts every `TargetScene` constructor field from
[From a scene](from-scene.md#a-valid-scene). The loader replaces
`trilegal_cache` with `INPUT_DIRECTORY/background.csv`, so the path written in
the JSON is not used and the key may be omitted. `background_sha256` is not
rewritten: `null` or an omitted key computes the digest from the file; a
supplied digest is verified. `N_background` is a recorded count and is not
compared with the file.

`binding` is the `target.from_scene` payload, not only the `bind_scene`
arguments. It accepts the `bind_scene` arguments
(`sector_weights`, `depth_ppm`, `exptime_days`, `random_seed`,
`depth_sigma_ppm`, `duration_days`, `duration_sigma_days`) plus the
`from_scene` curve arguments `contrast_curve` and `contrast_band`. Two keys are
overwritten before binding: `input_artifacts` is replaced by the three file
digests, and `bin_integration_days` is replaced by the value in `arrays.npz`.
Sector keys in `sector_weights` are JSON strings and are converted to integers;
use keys such as `"1"`, not `"1.0"`.

`baseline` and `stellar_light` are Python objects, not JSON values. They are not
supported by this file contract; use the Python API for those advanced paths. A
`contrast_curve` path in `binding` is read once, relative to the process working
directory, and is not one of the three retained inputs; prefer putting the
curve in `scene` as `contrast_separations` and `contrast_values` so its values
travel in the JSON. The path is not digested, but the curve values it produces
are part of the bound scene and of the restored-state identity described under
[Checkpoint and resume](#checkpoint-and-resume): a resume whose curve file now
parses to different values is refused, not reused.

`period_days` is the orbital period of the signal, in days. It must be a scalar;
an interval is refused. The example above assumes `background.csv` contains the
population and at least one row with `TESS >= 10.0`.

### `arrays.npz`

A NumPy archive with four required arrays:

| Key | Shape | Unit | Meaning |
|---|---|---|---|
| `time` | `(n,)`, `n >= 2` | days from transit midpoint | phase-folded times; finite real values |
| `flux` | same shape as `time` | normalized flux | finite real values; out of transit approximately 1 |
| `sigma` | scalar or `(n,)` | normalized flux | finite and positive; one calibrated value or one per point |
| `bin_integration_days` | scalar or `(n,)` | days | scalar `0` for exposure-only supersampling; per-point widths must be positive |

The arrays must be real numeric arrays. The loader uses `numpy.load` with its
default `allow_pickle=False`, so object arrays and pickled objects are refused;
booleans and strings are refused by the photometry admission. The four keys are
required. Extra arrays are read but not used by the calculation.

The loader reads every array in the archive and replaces
`binding["bin_integration_days"]` with the archive value. A value written in
`trex-input.json` is ignored. A scalar saved with `numpy.savez` as a zero-
dimensional array becomes a float; per-point widths become a list and must be
finite, positive and the same length as `time`.

All four keys are checked during input loading. A missing key is refused
with exit 2 and an `input-loading` failure record before inference or
checkpoint creation.

### `background.csv`

The TRILEGAL population for the field, in the format described in
[Background population](background-population.md). It must contain the required
columns `TESS`, `Mact`, `logg`, `logTe` and `[M/H]`; `Av` and `m-M0` are
optional. It must contain at least one row with `TESS >= Tmag`; a population
with no star at least as faint as the target is refused.

The file starts with a column header. A legacy unnamed leading index column is
allowed. A `#TRILEGAL ... normally terminated` footer is accepted; data after
the footer is rejected. The CSV is parsed at calculation admission, not by
`load_input`, so malformed content is refused when the calculation starts
rather than when the file is hashed.

### Input hashes

The loader computes SHA-256 for the three common files, checks any version-2
band-light assets against their manifest digests, and binds the complete
mapping as `input_artifacts`, replacing any mapping in the JSON. The bound
target verifies the artifacts at binding and again after inference before
accepting the result. The scene also verifies the background independently.

Relative and absolute input directories are accepted. The loader and input
binding use absolute artifact keys, while the handoff compares the digests
by filename, without the enclosing directory. Directory symlinks and symlinked input files are
still refused.

A digest proves file identity, not physical calibration. It does not tell you
whether the aperture model, the stellar parameters or the light-curve
uncertainty are correct.

## Output directory rules

The output directory must be either

- absent, or
- an empty directory that belongs to a resumable run, meaning the checkpoint's
  `monte-carlo.json` exists and the output directory is empty.

A directory that exists and is a symlink, is not a directory, is non-empty, or
is empty without a resumable handoff is refused before any calculation. A
previous `failure.json` makes the directory non-empty, so a failed or deferred
run cannot be retried in place; choose a new output directory. The output's
parent directory must already exist; the command does not create a missing
parent chain.

The output directory is checked before the input files are loaded. A bad output
is therefore reported even when the inputs are also bad.

## Checkpoint and resume

The checkpoint defaults to `OUTPUT_DIRECTORY.escalation` and can be set with
`--escalation-checkpoint-directory`. It must:

- be outside the output directory,
- not be a symlink,
- be owned by the current user,
- have no group or other permission bits.

The command creates a missing checkpoint with mode `0700`; missing parent
directories are created with the platform default permissions. An existing
checkpoint directory's mode is checked, not changed: any group or other bit
causes a refusal. A checkpoint path that exists as a regular file is also
refused.

The checkpoint holds the Monte Carlo handoff in `monte-carlo.json` and the
sampler's recovery files, including `.npz` state, dynesty `.dynesty` state or
the BlackJAX `.blackjax.npz` state, and the `mc-pilot-importance` directory when
that stage ran. The directory is execution state, not scientific evidence.
Native sampler state contains Python pickle; restore only a checkpoint created
by your own run with the same pinned TREX runtime, never an untrusted uploaded
checkpoint.

`monte-carlo.json` records the binding (`inputs`, `draws`, `nsamples`,
`estimator`) and the Monte Carlo state. A new handoff is published atomically
and is never replaced. On resume the command compares the retained binding with
the current one and refuses if anything changed:

```text
ValueError: checkpoint inputs or calculation controls changed (draws) at
CHECKPOINT; use a new checkpoint directory
```

If the checkpoint is non-empty but has no `monte-carlo.json`, the command
refuses rather than guessing what the files are. If the handoff is malformed or
its binding is not a mapping, the command refuses and leaves the damaged file
in place for inspection.

The default estimator is dynesty. `TREX_NESTED_ESTIMATOR=blackjax` selects the
experimental BlackJAX route. Both retain resumable state: dynesty in its native
`.dynesty` and `.npz` files, BlackJAX in `.blackjax.npz`. The estimator name is
part of the handoff binding, so a checkpoint made by one estimator is refused
by the other. The nested sampler additionally binds the TREX source bytes, the
atmosphere tables and its integration controls; a mismatch is refused rather
than reused. There is no promise that a checkpoint resumes across TREX versions,
tables, devices or changed controls. BlackJAX remains experimental.

When refinement pauses, the command writes no output; it prints the checkpoint
path and exits with status **75**. Rerun the same command with the same inputs
and options to resume. The handoff comparison uses the three file digests,
`--draws`, `--nsamples` and the selected estimator. Relative and absolute input
paths follow the same binding rules described above.

That handoff comparison is the first check, not the only input check. When the
retained Monte Carlo state is restored, the engine rebuilds the effective inputs
of this run and compares their identity with the one retained in the handoff:
the scene (including contrast-curve values whether they arrived in
`trex-input.json` or through an external `contrast_curve` file, the aperture and
neighbor records, and, when the scene does not carry it, the provenance
projected from a `trilegal_query.json` record read beside `background.csv`), the
photometry, period, depth, stellar parameters and limb-darkening choice, the
calculation controls, and the TREX source and atmosphere-table bytes. A
mismatch refuses the run with exit **2** and
`ValueError: escalation state physical/noise model, prior or numerical inputs
changed` before any retained draw is reused; the handoff stays in place for
inspection. A change to `trex-input.json`, `arrays.npz` or `background.csv` is
normally reported first as a changed handoff binding, because those three
digests are compared before the scene is restored.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Completed with `status="ok"` and a reported `FPP`/`NFPP` pair |
| `1` | Execution or publication failure, including a numerical likelihood failure and an uncaught exception |
| `2` | Refused before a result: argument, output, checkpoint, input-loading, input-binding or invalid-input admission |
| `3` | The calculation ran but evidence stayed unresolved, or the returned status was not `ok` |
| `75` | Paused at a retained checkpoint; no output was created |

The constants are `EXIT_COMPLETED = 0`, `EXIT_FAILED = 1`, `EXIT_REFUSED = 2`,
`EXIT_DEFERRED = 3` and `EXIT_PAUSED = 75`. An execution failure is raised, not
returned by `main()`; status 1 is the process exit status produced by the
uncaught exception.

The distinction matters for unattended callers:

- `0` is a completed calculation, not a validated planet. `claim_eligible` is
  always `false`.
- `1` is a real execution or publication failure. The original traceback is
  printed; a `failure.json` may exist, but is not guaranteed when publication
  itself failed.
- `2` covers an argument error, a refused output or checkpoint, a missing or
  malformed input file, a scene/binding validation error, and an invalid-input
  exception during admission. For a new output, the command publishes
  `failure.json` when it can.
- `3` is a scientific deferral: a convergence or missing-evidence outcome, or a
  result whose `status` is not `ok`. It is not a software crash and it is not a
  successful `FPP`/`NFPP` result.
- `75` means another session is needed at the retained checkpoint.

## What is written

| Outcome | Exit | Output directory |
|---|---:|---|
| Success, `status="ok"` | 0 | `summary.json` and `scenarios.csv`; no `failure.json` |
| Handled refusal, new output | 2 | `failure.json` only, when publication succeeds |
| Handled refusal, empty resumable output | 2 | untouched; the handoff is retained |
| Handled refusal, existing output | 2 | nothing; the existing output is left untouched |
| Argument-parser error | 2 | nothing; the output path is not considered |
| Refusal of the output path itself | 2 | nothing |
| Execution failure | 1 | `failure.json` only, unless the empty resumable output is protected or publication fails |
| Scientific deferral, unresolved evidence | 3 | `failure.json` only, unless the empty resumable output is protected |
| Scientific deferral, completed non-`ok` status | 3 | `summary.json` and `scenarios.csv`, with the non-`ok` status; no `failure.json` |
| Checkpoint pause | 75 | nothing; `monte-carlo.json` and sampler state remain in the checkpoint |

`failure.json` contains the error type, reason, operational diagnosis and, when
available, diagnostics and the limb-darkening grid summary. It has no `FPP` or
`NFPP` field, so a failure cannot be read as a probability of zero. See
[Outputs](outputs.md) for the full record.

The "no output file" behaviour belongs to a library call that raises before
`write_outputs`, to an argument-parser error, or to a refusal caused by the
output path itself. It is not universal to the command line: a handled input or
validation refusal writes `failure.json` into a new output when publication is
possible. A publication failure propagates as exit 1 and may leave an empty
directory or one of the success files, so a record is not guaranteed.

## What the command does internally

1. Resolve the checkpoint path and refuse it if it lies inside the output.
2. Check the output directory's admissibility.
3. Load and validate the common files and any version-2 band-light assets;
   rebuild the `TargetScene` and the bound target, with input digests.
4. Create or validate the checkpoint directory's ownership and mode.
5. Compare the retained handoff binding; when it matches, the engine re-checks
   the effective input identity and restores its Monte Carlo state.
6. Call `calc_probs` with `parallel=True`, the requested draws and
   supersampling, the bound exposure and the checkpoint.
7. If the first call returns pending Monte Carlo state, publish
   `monte-carlo.json`; continue nested refinement in the same process when the
   session deadline permits.
8. Write the result, or the failure, and return the exit code.

The command is the same engine as the Python API. It adds resumability and the
file-based input contract, not a different calculation.

## Version 2: joint observations

Use `trex.app.observation_io.export_observation_input` to write a calibrated
scene and `ObservationSet` for the same `python -m trex.run` command.
The exporter takes the measured `period_days` and `depth_ppm`, the epoch in
days relative to the set's shared TDB reference, and `band_light` keyed by
exact band ID. It checks that photometry, the scene and the light models agree
before writing a new directory.

In addition to the three files above, keep every `light-*.npz` asset named
by the manifest. `trex-input.json` identifies version 2, records observations,
the shared reference and optional baseline, and gives each band's compiled
grid filename, digest, model/response identifiers and target extinction.
`arrays.npz` repeats the first observation as proposal photometry with zero
phase-bin width; the loader checks it against the full observation set. Each
observation supplies its own exposure integration, so the CLI's `--nsamples`
does not replace that integration. AstroImageJ files are read first with the
[Python loader](api.md#astroimagej-photometry), not directly by the command.

## A minimal synthetic example

This example writes a version-1 input with synthetic data. Use a fresh
directory for the inputs and a different new directory for the outputs.

```python
import json
from pathlib import Path

import numpy as np

from trex.app import example

inputs = Path("demo-input")
scene, binding, time, flux, sigma, period = example.synthetic_inputs(inputs)
(inputs / "background.csv").write_bytes(
    (inputs / "synthetic-trilegal.csv").read_bytes()
)

payload = {
    "scene": scene.to_dict(),
    "binding": {
        key: value for key, value in binding.items()
        if key not in {"input_artifacts", "bin_integration_days"}
    },
    "period_days": period,
}
(inputs / "trex-input.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
)
np.savez(
    inputs / "arrays.npz",
    time=time,
    flux=flux,
    sigma=sigma,
    bin_integration_days=np.asarray(binding["bin_integration_days"]),
)
```

The synthetic helper also leaves `synthetic-trilegal.csv` and
`synthetic-aperture-input.json` in the directory; they are demonstration
leftovers and are not part of the contract. The JSON still names
`synthetic-trilegal.csv` in `trilegal_cache`; the loader rewrites it to
`background.csv`. The copied background has the same bytes, so the scene's
retained digest still verifies. The archive's `bin_integration_days` overrides
the binding value, as described above.

Run it with a reduced budget:

```bash
python -m trex.run "$PWD/demo-input" "$PWD/demo-output" --draws 4096
```

The result checks that the file contract, the engine and the output writer work
on your machine. It is not a scientific probability. Relative input and output
paths are also supported.

## Related pages

- [From a scene](from-scene.md): the scene fields and the Python binding.
- [Inputs](inputs.md): field-level rules for the scene, arrays and units.
- [Outputs](outputs.md): what `summary.json`, `scenarios.csv` and
  `failure.json` contain.
- [Python API](api.md): the same calls from Python.
- [From a TIC ID](from-tic.md): how to acquire a scene before preparing it.
- [Scenarios](scenarios.md): what the scenario rows mean.
- [FAQ](faq.md): paused runs, refusals and missing outputs.
