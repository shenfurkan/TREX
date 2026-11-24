# FAQ and troubleshooting

This page answers the questions that come up while installing, running and
reading TREX, and maps the common error messages to a cause. Short answers first;
the linked pages carry the detail.

## Scientific meaning

### Can TREX validate or confirm a planet?

No. TREX reports probabilities conditional on its inputs, modelled scenarios and
priors. It cannot rule out explanations it does not model. Its `claim_eligible`
field is always `false`. Other observations and independent scientific
assessment are needed for any planet claim.

### Is a small FPP a detection?

No. A small FPP means the modelled false-positive scenarios carry little weight
for the light curve and scene you supplied. It does not mean the planet
explanation is correct, that the scene is complete, or that the priors apply.
Read [Concepts](concepts.md#7-what-the-probabilities-do-not-mean).

### What is the difference between FPP and NFPP?

FPP sums every modelled explanation except a planet on the target. NFPP sums the
explanations hosted by resolved nearby stars. NFPP is part of FPP, so it cannot
be larger than FPP for the same normalization. [Concepts](concepts.md#4-from-evidence-to-fpp-and-nfpp).

### Why does TREX use a fixed scenario list?

Because a probability needs a denominator. The scenario framework of Giacalone
et al. (2021) defines the explanations being compared. An explanation outside
the list receives no probability mass. That is a limitation, not a statement
that the explanation is impossible. [Scenarios](scenarios.md).

### Does TREX search for transits?

No. It is a vetting calculation. You supply the period, epoch, depth and
duration, or take them from a catalogue. Use a transit search for the search
step.

### Why is the synthetic example's FPP meaningless?

The target, neighbours, light curve and background are invented, the flux
uncertainty is deliberately 1.0 in normalized units, and the budget is 4,096
draws. It exercises the software path. It is not a result about a star.

### Why do two runs of the same target differ slightly?

Monte Carlo estimates have sampling noise. A different seed or draw budget
changes the realized draws; a different exposure supersampling or backend
changes the computed likelihood. With the same inputs and seed the scenario
streams are reproducible, but bitwise identity across backends is not
guaranteed. Compare the ESS and the diagnostics, not only the last digits of
FPP.

### Why is my probability different from a TRICERATOPS result?

The two codes share the scenario framework but differ in the evidence estimator,
the likelihood details, the population priors and the scene acquisition. Even
matched inputs do not make their probabilities automatically identical. The
[source-level comparison](differences-from-triceratops.md) records the
differences. Do not treat either number as ground truth.

## Installing and running

### Which version is this documentation for?

TREX `1.0.0`, the version declared in `pyproject.toml`. The Git tag
`v1.0.0` identifies the source for the rebuilt wheel.

### Is TREX on PyPI?

Yes. Use `python -m pip install trex==1.0.0` with Python 3.14. The unrelated
pre-1.0 releases on PyPI have been yanked. There is no `trex` console script.

### What Python and platform do I need?

Python 3.14.x. Linux x86-64 is the primary platform; macOS arm64 has no recorded native run. Native
Windows is not supported; use WSL. Intel macOS lacks matching wheels for the
required `jaxlib` release. [Installation](installation.md).

### Do I need a GPU?

No. The default install uses CPU JAX. A full-budget run is not short: the
recorded synthetic quick-start took 18,312 s on four x86-64 CPUs, and an earlier
T4 GPU record of the same synthetic input took 2,273 s. Those are different code
snapshots and are not a controlled speed benchmark; neither is a real-target
timing. [Measurement](measurements/quickstart-cpu-2026-11-10.md).

### How do I check whether JAX sees my GPU?

```bash
python -c "import jax; print(jax.devices())"
```

CPU-only output is a supported configuration. Installing the `cuda12` extra is
not a GPU test; the driver and device must also be visible to JAX.

### Why will the example not run a second time?

It refuses to overwrite its output directory: the example creates the directory
with `mkdir()`, so a second run raises `FileExistsError`. Give it a new
directory name, or inspect the existing files. Do not delete data you need.

## Inputs and the scene

### Why does `from_tic` need a metallicity?

The target's own scenarios use its limb darkening, every eclipsing-binary
scenario needs the host composition for the eclipsed secondary's atmosphere, and
every bound-companion-hosted scenario needs it for the drawn companion. A missing
value cannot be replaced by solar without assigning probability to a model that
was never evaluated. Acquisition raises if no catalogue in the ladder supplies a
measured value.

### Which sources supply the metallicity?

In order: APOGEE DR17, GALAH DR3, calibrated Gaia GSP-Spec, SHBoost. The
selection and its provenance are retained with the acquisition outputs.
[From a TIC ID](from-tic.md#3-stellar-parameters).

### Why does a neighbour have no scenarios?

Possible reasons, all recorded in the output:

- its aperture flux fraction is zero,
- the observed depth is larger than its entire aperture flux fraction.

Those are physical `host-excluded` outcomes: the three slots keep `-inf`
evidence and the run can still publish. A neighbour with no finite metallicity
and no usable target fallback is `unavailable`, and the run defers rather than
normalizing over a reduced inventory. A neighbour that could host the depth but
has neither fixed mass and radius nor a usable stellar-parameter distribution
is different again: the run raises instead of dropping it.
A scene whose aperture table lacks an observed sector is refused, not
treated as a neighbour with no scenarios. An unavailable neighbour is not
evidence that it is absent.

### Can I supply my own aperture fractions and contrast curve?

Yes. Build a `TargetScene` and use `target.from_scene`. The aperture fraction is
a share of in-aperture flux, not a magnitude ratio. The curve must be in the
TESS band. [From a scene](from-scene.md).

### Why is a non-TESS contrast curve refused?

TREX evaluates companion contrasts in the TESS band and implements no verified
bandpass conversion. Converting a curve would add an unvalidated step to every
companion decision, so the input is refused instead.

### Which file does `background.csv` refer to?

The TRILEGAL-format population CSV used by the prepared-run command. In the
Python API you pass its path as `trilegal_cache`. The format is in
[Background population](background-population.md#the-population-file).

### Why is `depth_ppm` needed if the light curve has the depth?

It is an input used to decide which faint neighbours need stellar parameters and
to centre the proposal windows. It does not enter the likelihood as data. Supply
the measured depth, not a guess.

### Why must `exptime` match the bound exposure?

The forward model integrates the transit over the actual exposure. A guessed
exposure changes the predicted shape. The binding records the real cadence; the
calculation must use it.

## Convergence and refinement

### What does a low effective sample size mean?

The ESS summarises how concentrated the draw weights are. Low ESS can mean the
estimate depends on a few draws. TREX may refine a scenario, defer the result, or
report why a scenario was unavailable. A high ESS is not proof that all
important possibilities were sampled.

### Why did my run defer instead of reporting an FPP?

The engine publishes a result only when the evidence set passes its convergence
routing. Common causes:

- a scenario stayed below 100 effective draws and held more than 10⁻³ of the
  normalized probability after any attempted refinement,
- a sampled scenario had no finite weight and could flip the decision,
- a supported likelihood failed numerically,
- a required input was missing.

A deferral is a refusal to publish an unconverged number. It is not a low FPP.
A wall-time pause is a separate outcome: it leaves a checkpoint and exit 75,
not a deferred result.

### Why is there no summary or FPP?

Read the exception and, if present, `failure.json`. Missing observations, archive
outages and numerical failures are not zero probabilities. A check that fails
before the calculation starts may not create an output file.

### Why does refinement not run at 4,096 draws?

Nested refinement is attempted only at or above the production budget of
1,000,000 draws. Smaller budgets are for software tests; if convergence fails
there, the run defers.

### Which nested estimator is used?

dynesty by default, with 400 live points and an evidence tolerance of 0.05
nats. The opt-in BlackJAX route is experimental. Set
`TREX_NESTED_ESTIMATOR=blackjax` to select it; any other value raises. The
selected estimator is recorded in the prepared-run binding.

### Is the BlackJAX route validated?

Only against synthetic tests with a known answer. A four-parameter Gaussian
stayed within 0.1 nat of the true log evidence for 48 of 50 seeds in each of two
separate 50-seed runs, and a one-parameter exponential for 100 of 100. A nat is
the natural-log unit of evidence
([Concepts](concepts.md#3-priors-likelihood-and-evidence)). The route is not
calibrated for real scenes, multimodal likelihoods or GPU execution, and the
native Gaussian baseline still requires dynesty. dynesty remains the default.
[How TREX works](how-trex-works.md#the-opt-in-blackjax-estimator).

### Why does a paused run exit with 75?

Status 75 is the prepared-run command's "paused, resume me" convention. It means
the refinement ran out of wall time and wrote a checkpoint; no `summary.json` or
`failure.json` is written. Run the same command again with the same inputs; do
not delete the checkpoint.

### Can I resume a run that used BlackJAX?

Yes, from a retained BlackJAX checkpoint. The route keeps its own native
`.blackjax.npz` state and continues at a complete boundary under the same input
and control identity. The prepared-run binding records the estimator, so resume
with the same estimator; do not switch it. BlackJAX remains experimental. See
[Checkpoints and resume](outputs.md#4-checkpoints-and-resume).

## Reading the outputs

### What is `N15` in `summary.json`?

The scenario index for the first neighbour's `NTP` slot. Neighbour scenarios are
keyed by index (`N15`, `N16`, `N17`, then `N18`, ...), while `scenarios.csv` uses
`NTP`, `NEB`, `NEBx2P` and the neighbour's `ID`. [Outputs](outputs.md#scenario-keys).

### Why is a scenario probability zero but the status `ok`?

A scenario can have zero probability because its evidence is many orders of
magnitude smaller than the dominant scenarios, or because a neighbour was
physically excluded (zero aperture share, or a depth larger than its entire flux
share). Those are computed results. They are different from an unavailable
scenario, which is excluded for a missing input and listed in
`unavailable_scenarios`.

### Why is a number written as a string such as `"-inf"`?

Non-finite values are serialized as strings so the JSON remains valid and the
failure stays visible. A `"-inf"` evidence means the scenario contributed zero
weight: no sampled draw had finite likelihood, or the scenario was excluded as
physically impossible or unavailable. It is not a small computed probability;
read the scenario's status and reason.

### What does `claim_eligible: false` mean?

TREX does not issue planet claims. The field is always false by design. A
threshold pass in the diagnostics is a display convention, not a claim.

## Troubleshooting by message

| Message | Cause | What to do |
|---|---|---|
| `TREX scenario evidence did not converge at n_draws=...` | A sub-floor scenario held too much probability, or a sampled scenario had no finite weight | Run at the production budget if you are below it; keep the checkpoint; read `unconverged_scenarios` and the diagnostics. Do not lower the budget |
| `resolved neighbor 0 can host the observed depth but has no stellar mass and radius` | A possible host has neither parameters nor a `stellar_parameter_distribution` | Supply its mass and radius, or a conditioned distribution, through a scene. Do not remove the neighbour |
| `scene.feh_dex is required to evaluate every false-positive scenario` | No target metallicity | Supply a measured value with provenance |
| `a scene requires a measured contrast curve or the catalog resolution limit` | No unresolved-scene bound | Add a curve or `unresolved_resolution_limit_arcsec` |
| `calibrated aperture flux shares must be finite, non-overcounted and include the target` | Fractions sum above 1, or the target has none | Recheck the aperture model and the per-sector shares |
| `observed sector weights must cover exactly the calibrated sectors` | Missing or extra sector key | Use the scene's exact sector list |
| `exptime must equal the bound native exposure [days]; do not use a guessed default` | `calc_probs` exposure differs from the binding | Pass the same exposure in both places |
| `bound input artifact is missing or changed` | An `input_artifacts` file moved or was edited | Restore the file or rebind with the current paths and digests |
| `TREX evaluates companion contrasts in the TESS band only` | Non-TESS `contrast_band` | Supply a TESS-band curve; no conversion is implemented |
| `contrast_band is required with contrast_curve and must name the band the curve was measured in` | Curve supplied without its band | Name the band the curve was measured in |
| `contrast curve separations must be positive and strictly increasing` | Malformed curve | Fix the table; separations must be positive and increase, with at least two finite points |
| `TRILEGAL population has no simulated star fainter than the target` | Population and target are inconsistent | Re-acquire the population for this field and target |
| `scene request differs from the retained scene_request.json` | A new acquisition reused an old output directory with different settings | Use a new output directory |
| `checkpoint inputs or calculation controls changed` | A resume used different draws, inputs or estimator | Use the original settings or start a new checkpoint |
| `unknown TREX nested estimator '...'; select dynesty or blackjax` | Invalid `TREX_NESTED_ESTIMATOR` | Use `dynesty` or `blackjax` |
| `native Gaussian baseline requires dynesty; BlackJAX baseline escalation is unsupported` | A `baseline` was supplied with BlackJAX selected | Select dynesty or omit the baseline |
| `FileExistsError` (`[Errno 17] File exists` on Linux) | `write_outputs` was pointed at an existing directory | Choose a new directory; only an existing empty directory is reusable with `existing_empty=True` |
| `FileExistsError: output directory is not empty: ...` | `existing_empty=True` but the directory contains files | Do not reuse a non-empty output |
| `ValueError: no calculation has been attempted` | `write_outputs` called before `calc_probs` | Run the calculation first |
| `ValueError: fit plot inputs differ from the retained likelihood observations` | `plot_fits` got edited arrays | Pass the exact arrays used in `calc_probs` |

### A warning appeared about limb darkening

```text
TREX host limb-darkening requests fell outside the TRICERATOPS TESS grid bounds;
the nearest grid nodes were used.
```

This is not an error. The requested atmosphere fell outside the published grid,
so TREX used the nearest node and recorded the lookup in
`limb_darkening_grid`. Check the provenance if the host is far outside the grid
(for example a very cool or very evolved star).

### A warning appeared about a JAX batch fallback

The batch failed and those draws were re-evaluated with the NumPy reference
likelihood. The evidence remains defined, and `jax_batch_fallbacks` records the
draw count and exception. A single fallback is not a wrong result, but repeated
fallbacks are a performance signal and may indicate an accelerator problem.

## Still stuck?

If the answer is not here:

1. Read `failure.json` or the exception's `diagnostics`.
2. Check `calculation_provenance` against the inputs you intended.
3. Re-run the synthetic example to confirm the installation is intact.
4. Reduce the problem: one sector, one neighbour, and a small budget, to see
   which input the engine rejects.

Do not publish or act on a probability from a run whose diagnostics you have not
read.
