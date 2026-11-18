# Outputs

This page explains what a run writes and what each field means. It covers the
files produced by `write_outputs`, the columns of the scenario table, the
failure record, checkpoints, plots and logs.

A successful run writes two files:

```text
result/
├── summary.json
└── scenarios.csv
```

A refused, deferred or failed calculation may write one file instead:

```text
result/
└── failure.json
```

The command-line path and its input directory are documented in
[Prepared runs](prepared-runs.md). Inputs rejected before a calculation starts
may leave no file at all; the Python exception carries the reason. Existing
output directories are never overwritten.

## Outcomes

| Outcome | Record | Command exit |
|---|---|---|
| Completed calculation | `summary.json` and `scenarios.csv`, `status: ok` | 0 |
| Insufficient, missing or unconverged evidence | `failure.json` with `operational_diagnosis.status: deferred`; no FPP/NFPP | 3 |
| Execution failure | `failure.json` with `operational_diagnosis.status: failed` | 1 |
| Refusal before a result | `failure.json` when a new output can be published, otherwise no record | 2 |
| Paused refinement | no output file from the command; a library `write_outputs` record is a deferred `failure.json` with category `paused` | 75 |

Publication itself can fail; the original exception then propagates and a
record is not guaranteed. A pause is a recovery state, not a deferral or an
execution failure. A run with fewer than 1,000,000 draws per scenario is a
software check: an `ok` status at such a budget is not production-budget
scientific evidence.

## 1. `summary.json`

The file is strict JSON with two-space indentation and sorted keys. Non-finite
numbers are written as strings (`"-inf"`, `"inf"`, `"nan"`) rather than `NaN` or
`Infinity`, and unknown objects are written as strings, so a numerical failure
stays visible instead of becoming zero. A JSON `null` means the value was not
computed or not available.

The top-level keys are exactly:

| Field | Type | Meaning |
|---|---|---|
| `FPP` | number or null | False-positive probability: the sum over every target scenario except `TP`, `PTP` and `DTP`, plus every neighbour scenario |
| `NFPP` | number or null | Nearby false-positive probability: the part of `FPP` supplied by resolved-neighbour scenarios |
| `status` | string | Normalization and evidence status; only `ok` is a completed result |
| `claim_eligible` | boolean | Always `false` |
| `input_identity` | string | Digest of the bound scene and the scientific inputs the engine evaluated |
| `effective_sample_sizes` | object | Final Kish effective sample size (ESS) by scenario, where the retained estimator reports one |
| `scenario_monte_carlo_evidence` | array | First-stage Monte Carlo evidence, its error, ESS and availability per scenario |
| `scenario_evidence_refinement` | array | Whether a refinement stage replaced each Monte Carlo estimate |
| `unavailable_scenarios` | array | Scenarios whose required model input was absent, with reasons; a run containing one normally defers |
| `phase_wall_seconds` | object | Measured wall time per engine phase |
| `nested_evidence_escalation` | object or null | What the refinement router attempted |
| `supplemental_nested_evidence` | object | Estimator-labelled records for substituted scenarios |
| `jax_batch_fallbacks` | object | JAX batches re-evaluated by the `batman` reference path |
| `diagnostics` | array | Named checks, their status and messages |
| `limb_darkening_grid` | object | Summary of the packaged atmosphere grid and the lookups that occurred |
| `calculation_provenance` | object | Seed, inputs, exposure, weights and configured models |

A non-empty `unavailable_scenarios` prevents an `ok` normalization, so a
published `summary.json` has an empty list. The missing entry is not evidence
that the missing explanation has probability zero; it is a reason the run
deferred.

### Scenario keys

The 15 target scenarios are keyed by name: `TP`, `EB`, `EBx2P`, `PTP`, `PEB`,
`PEBx2P`, `STP`, `SEB`, `SEBx2P`, `DTP`, `DEB`, `DEBx2P`, `BTP`, `BEB`,
`BEBx2P`.

Neighbour slots are keyed by scenario index: `N15`, `N16`, `N17` for the first
neighbour; `N18`, `N19`, `N20` for the second; and so on. Within each group the
physical labels repeat in the order `NTP`, `NEB`, `NEBx2P`. In
`scenario_monte_carlo_evidence`, `scenario` is the index key (`N15`) and
`scenario_label` is the physical label (`NTP`). The host star is identified by
`ID` in `scenarios.csv`; the Monte Carlo rows do not carry a `source_id` field.

### `effective_sample_sizes`

A map from scenario key to the final Kish effective sample size. A value near
the draw count means the importance weights are spread nearly evenly; a value
near 1 means one draw dominates. The production floor is 100. An MC-pilot
defensive-importance substitution records its own ESS here. A nested
substitution that reports no Kish ESS may be absent from the map; its estimate
is then described in `scenario_evidence_refinement` and
`supplemental_nested_evidence`. The map is evidence about the realized weights,
not a convergence certificate.

### `scenario_monte_carlo_evidence`

One row per scenario slot, including scenarios that were not evaluated. The
first-stage value is retained even when a later stage replaced it.

| Field | Meaning |
|---|---|
| `scenario` | Scenario key (`TP`, `N15`, ...) |
| `scenario_label` | Physical label (`TP`, `NTP`, ...) |
| `scenario_index` | Position in the evidence vector |
| `mc_ln_evidence` | First-stage log evidence in nats (natural-log units; 1 nat is a likelihood ratio of `e`), including the resolved neighbour's spatial log-prior where applicable; `"-inf"` if no finite sampled weight |
| `mc_ln_evidence_error` | `1/√ESS` in nats, or null when no finite ESS |
| `mc_effective_sample_size` | First-stage ESS, or null |
| `n_draws` | Draws used, or null when the record does not carry an integer count |
| `evidence_origin` | `monte-carlo`, `nested-escalation`, `mc-pilot-defensive-importance`, `unknown`, or a per-scenario availability label (`unavailable`, `not-evaluated`, `host-excluded`) |
| `status` | Optional; the producing record's status, such as `ok`, `unavailable`, `not-evaluated`, `host-excluded`, `no-finite-likelihood-draws` or `record-unavailable` |
| `reason` | Optional; the producing prior's message |
| `unresolved_numerical_failures` | Optional count of supported draws that failed |
| `numerical_failure_examples` | Optional list of example failures |
| `metallicity_provenance`, `metallicity_unavailable_reason`, `host_metallicity_dex`, `metallicity_treatment` | Optional composition record for neighbour and background hosts |
| `transiting_draws` | Optional; sampled draws on transiting geometry |
| `finite_likelihood_draws` | Optional; draws with a finite likelihood |
| `proposal_support` | Optional; the occultor-radius proposal's support state (`usable`, `below-support`, ...), its own classification record |
| `phase_seconds` | Optional; measured `draw`/`wait`/`evaluate` wall split for this scenario |

The last four fields explain a sub-floor ESS: how much of the prior volume
produced a finite likelihood, whether the proposal window could reach that
prior's support, and where the scenario's time went. They describe the sample;
they do not certify that unsampled support is absent.

A scenario with `"-inf"` evidence or `status: "unavailable"` is not a measured
zero. It had no finite sampled weight or no model to attribute mass to it, so
its zero contribution is not evidence that the scenario is absent.

### `scenario_evidence_refinement`

One row per scenario slot:

| Field | Meaning |
|---|---|
| `scenario` | Scenario key |
| `mc_ln_evidence` | First-stage log evidence |
| `mc_effective_sample_size` | First-stage ESS |
| `refined_effective_sample_size` | ESS reported by the substituted estimator, or null |
| `refinement_status` | `refined`, `not-refined`, `unresolved` or `unavailable` |
| `refinement_estimator` | `mc-pilot-defensive-importance`, `nested-sampling`, `nested-importance-sampling`, or null |
| `refinement_reason` | Why refinement did not replace the estimate, or why a substituted estimate has no reported ESS; null otherwise |

A refined estimate reports an ESS when its estimator supplies one. A plain
nested ensemble does not report a Kish ESS, so `refined_effective_sample_size`
stays null with that reason; an importance-refined nested estimate can report
its ESS through its `importance_sampling` record. A missing refined ESS is not
zero. The refined log evidence is not repeated in this row; it is in
`supplemental_nested_evidence`. A nested record stores that value before its
scenario prior, while the MC-pilot defensive-importance record includes its
prior.

### `unavailable_scenarios`

A list of `{"scenario": ..., "reason": ...}` entries for scenarios whose status
is `unavailable`. A missing required input or model makes the affected scenario
unmodelled, which is not the same as zero probability. Physically
`host-excluded` scenarios and `not-evaluated` impossible-prior slots are not
included here. The FPP is conditional on the modeled set; a missing scenario can
make it biased low and cannot be read as evidence that the scenario is absent.

### `phase_wall_seconds`

Measured wall time in seconds per engine phase:

| Key | Meaning |
|---|---|
| `monte_carlo` | Monte Carlo stage, including a restored value on a resumed run |
| `nested_escalation` | Nested refinement in this process, or null when it did not run |
| `nested_escalation_cumulative` | Wall time summed across retained refinement attempts, or null |
| `mc_importance` | Defensive-importance stage, present only when that stage ran |

### `nested_evidence_escalation`

Null when no refinement stage ran. Otherwise it contains:

| Field | Meaning |
|---|---|
| `attempted_scenarios` | Scenario names the router attempted |
| `escalated` | Scenario names successfully substituted |
| `unmet_reason`, `unmet_scenarios` | Why an attempted integration did not meet the acceptance bar, and the affected names |
| `numerical_failures` | Per-scenario numerical failure records |
| `unsubstituted_scenarios`, `unsubstituted_reason` | Scenarios left on the Monte Carlo estimate and why |
| `cap_binding` | Whether a registered cap limited the stage |
| `random_seed`, `seeds` | Seeds used by the stage |
| `attempt_costs` | Per-attempt cost records, including failed and paused work |
| `cumulative_wall_seconds` | Wall time summed across attempts |
| `mc_importance` | Object with `estimator: "mc-pilot-defensive-importance"` and `attempts`, keyed by scenario |

### `supplemental_nested_evidence`

A map from scenario key to the substituted estimator's record. A scenario has
one final estimator and cannot appear in both families.

An MC-pilot defensive-importance record carries:

| Field | Meaning |
|---|---|
| `scenario`, `index` | Scenario identity |
| `estimator` | `mc-pilot-defensive-importance` |
| `random_seed` | Importance-stage seed |
| `log_evidence` | Log evidence including the scenario prior, or null |
| `log_evidence_error_nats` | Reported error in nats, or null |
| `effective_sample_size` | Kish ESS, or null |
| `n_draws`, `n_likelihood_calls`, `n_centres` | Draw and evaluation counts |
| `defensive_fraction`, `maximum_weight_share` | Proposal and weight diagnostics |
| `execution_provenance`, `execution_backend` | Execution route |
| `wall_seconds` | Stage wall time |
| `status`, `reason` | `accepted` with a null reason; unresolved attempts stay in `nested_evidence_escalation.mc_importance.attempts` |

A nested-sampling record carries:

| Field | Meaning |
|---|---|
| `scenario`, `index` | Scenario identity |
| `kind`, `use_2xP` | Scenario family and period multiplier |
| `proposal_support` | Proposal-support diagnostic, or null |
| `estimator` | `nested-sampling` or `nested-importance-sampling` |
| `acceptance` | `single-run-below-bound`, `ensemble` or `importance-refined` |
| `log_evidence` | Log evidence before the scenario prior; add that prior to compare with `substituted_probability` |
| `importance_sampling` | Importance-refinement record, or null |
| `log_evidence_values` | Per-run evidence values |
| `ensemble_standard_error_nats` | Ensemble standard error, or null |
| `substituted_probability` | Normalized probability after substitution |
| `single_run_probability_upper_bound` | Bound used by a single-run acceptance, or null |
| `seeds` | Per-run seeds |
| `n_live_points`, `max_likelihood_calls` | Sampler controls |
| `calibrated_configuration` | BlackJAX control marker (`true` only for the default assessed policy with the 3,200-live and derived-cap controls), or null for a non-BlackJAX estimator |
| `n_likelihood_calls`, `n_iterations` | Work counts |
| `sampler`, `sampler_version` | Estimator identity |
| `reported_log_evidence_error_nats` | Error reported by the accepted estimate |
| `runs` | Per-run records |
| `total_n_likelihood_calls`, `total_monitored_likelihood_calls`, `total_n_iterations` | Totals |
| `wall_seconds` | Stage wall time |
| `stellar_light` | Physical-light configuration, present only when bound |

When present, `importance_sampling` carries `log_evidence`,
`log_evidence_error_nats`, `n_draws`, `n_likelihood_calls`,
`effective_sample_size`, `maximum_weight_share`, `n_centres`,
`defensive_fraction`, `random_seed`, `execution_provenance` and
`agreement_standard_errors`. Each per-run record in `runs` carries
`random_seed`, `log_evidence`, `reported_log_evidence_error_nats`,
`n_live_points`, `n_likelihood_calls`, `n_iterations`, `sampler`,
`sampler_version`, `stopped_short`, `monitored_likelihood_calls`,
`budget_exhausted`, `stopped_on_evidence_tolerance`, `stop_reason` and
`base_run_delta_logz`.

### `jax_batch_fallbacks`

A map from scenario key to the last recorded fallback for that scenario. Each
record has `kind` (the likelihood family), `draws` (transiting draws that would
have been covered), `exception_type` and `message`. These records disclose that
the accelerator path failed and the draws were evaluated by the `batman`
reference likelihood; they are not a gate input and do not imply the evidence
was regenerated.

### `diagnostics`

Each entry is an object with `name`, `status`, `message`, `value` and
`threshold`. A diagnostic `status` is `pass`, `fail` or `info`; `value` and
`threshold` may be null. The `fpp`, `nfpp` and `overall` entries are a display
threshold screen using `FPP < 0.01` and `NFPP < 0.001`. A pass makes no planet
claim, and a fail does not reject the candidate; both remain conditional on the
inputs and scenarios. When the result is degenerate, the `fpp`, `nfpp` and
`overall` entries may be absent.

The diagnostic names that indicate a problem are:

| Name | Meaning |
|---|---|
| `scenario-input-unavailable` | A required input was absent; the affected scenarios are unmodelled, which is not the same as zero probability |
| `scenario-numerical-failure` | Supported likelihood evaluations failed numerically |
| `evidence-effective-sample-size` | Sub-floor scenarios hold more mass than the routing allowance (status `fail`), or less but remain a caveat (status `info`) |
| `numerical-stability` | `NaN` or `+inf` appeared in the evidences |
| `all-neginf` | Every scenario had zero sampled weight; FPP would be a failed computation |
| `mc-pilot-importance-substitution` | A defensive-importance estimate replaced the Monte Carlo one |
| `evidence-estimator-substitution` | Nested estimates replaced Monte Carlo estimates |

`scenario-*` entries report the top five scenarios by probability and are
informational.

### `limb_darkening_grid`

The summary has these top-level keys:

| Key | Meaning |
|---|---|
| `contract` | The registered clamp/selection contract text |
| `package_version` | TRICERATOPS reference package version |
| `coefficient_supplement` | Packaged coefficient supplement, source and method |
| `coefficient_extension` | Claret extension source, citation, URL, retrieval date, domains and interpolation rule |
| `cool_coefficient_extension` | Cool-star PHOENIX extension source and selection rule |
| `teff_k_lookup_bounds`, `logg_cgs_lookup_bounds` | Lookup domains |
| `clamped` | Whether a draw or target lookup used the nearest published node; `null` when unknown |
| `events` | Per-host clamp/node events, or null |
| `interpolated` | Whether a draw or target lookup used interpolated coefficients; `null` when unknown |
| `interpolation_events` | Interpolation events, or null |
| `target_host` | The coefficients used by the target's own scenarios, or null |
| `monte_carlo_source_counts` | Draw counts by coefficient source, or null counts when incomplete |
| `monte_carlo_provenance_status` | `available`, `unavailable`, or an incomplete-legacy status |

A clamp means the requested atmosphere fell outside the published grid and the
nearest node was used. It is not an error.

### `calculation_provenance`

The record a reader needs to reproduce the calculation:

| Field | Meaning |
|---|---|
| `claim_eligible` | `false` |
| `n_draws` | Draws per scenario |
| `random_seed` | Seed used for the run, including a generated seed when none was supplied |
| `input_artifacts` | Absolute paths and SHA-256 digests verified at binding |
| `sector_inverse_variance_weights` | Per-sector inverse-variance weights supplied |
| `exptime_days` | Native exposure in days |
| `bin_integration_days` | Scalar or per-point phase-bin widths in days; scalar `0` selects exposure-only integration |
| `depth_ppm` | Measured depth supplied |
| `period_days` | Orbital period in days |
| `nsamples` | Exposure supersampling |
| `limb_darkening` | Per-lookup atmosphere provenance events |
| `noise_scope` | States that one caller-calibrated `sigma` is shared by every scenario |
| `neighbor_metallicity` | Per-neighbour composition and treatment |
| `noise_model` | Present when an explicit native-time baseline was bound |
| `stellar_light` | Present when an explicit physical stellar-light model was bound |
| `background_population` | Present when the population came from the TRILEGAL LSST fallback |
| `depth_sigma_ppm`, `duration_days`, `duration_sigma_days` | Present only when those measured transit values were supplied |

`neighbor_metallicity` entries carry `source_id`, `host_metallicity_dex`,
`metallicity_treatment`, `metallicity_provenance` and
`metallicity_unavailable_reason`. The `noise_model` record is the bound
baseline's provenance (model, mode, source, coefficient count, time domain,
normalization, execution and domain). The `stellar_light` record is the model
configuration (model id, release, abundance, response, extinction, target A_V,
coordinates and interpolation). The `background_population` record carries the
fallback model, population SHA-256, trigger and limitation. The unmeasured
optional records are absent rather than filled with defaults.

## 2. `scenarios.csv`

One row per scenario and host, in scenario-index order. The columns are exactly:

| Column | Unit | Meaning |
|---|---|---|
| `ID` | none | Host star identifier; the target's TIC ID or the neighbour's source ID |
| `scenario` | none | `TP`, `EB`, `NTP`, ... |
| `M_s` | M☉ | Host mass of the representative draw |
| `R_s` | R☉ | Host radius of the representative draw |
| `P_orb` | days | Model period (`2P` for `x2P`) |
| `inc` | degrees | Inclination of the representative draw |
| `b` | none | Impact parameter at conjunction |
| `ecc` | none | Eccentricity |
| `w` | degrees | Argument of periastron |
| `R_p` | R⊕ | Planet radius for planet scenarios; `0` for binary scenarios |
| `M_EB` | M☉ | Eclipsing secondary mass for binary scenarios; `0` otherwise |
| `R_EB` | R☉ | Eclipsing secondary radius for binary scenarios; `0` otherwise |
| `prob` | none | Normalized scenario probability |

The **representative draw** is the Monte Carlo draw with the highest
log-likelihood for that scenario. It is not a posterior mean, median or best
fit. When a scenario has no representative draw, the draw-derived columns are
left empty in the CSV; `P_orb` still carries the model period from the scenario
record. A `host-excluded` scenario has probability zero; a missing-input
scenario blocks publication instead. A scenario substituted by
a refinement stage can have a nonzero probability without a Monte Carlo draw;
read its row in `scenario_evidence_refinement` before interpreting the empty
columns. When a draw exists, a component the scenario does not use is written as
zero: `R_p` for a binary, `M_EB` and `R_EB` for a planet.

The probabilities include the observed-period scenario prior and any accepted
refinement substitution. They are conditional on the modeled scenarios and are
not a posterior sample. `prob` is not FPP: `FPP` sums the non-planet target
scenarios and all neighbour scenarios.

Reading the table:

- A large `prob` with a plausible radius and impact parameter is a scenario that
  explains the light curve well.
- A binary row with `M_EB` near `M_s` describes a near-twin pair; a small
  `M_EB` describes a low-mass companion.
- `b` above `1` for the representative draw can occur for a grazing or
  eccentric configuration; the likelihood still evaluates it.
- `P_orb` is the model period, so `EBx2P` rows show twice the signal period.

No single row carries the full inference, which is the evidence integral. Treat
the representative draw as a visual and numerical summary only.

## 3. Failures and status

### Status values

A published `summary.json` records a calculation outcome. Read `status` before
reading `FPP` or `NFPP`; only `ok` supplies a usable pair.

| Status | Meaning and reader action |
|---|---|
| `ok` | Normalization produced finite FPP and NFPP. Read the diagnostics and the scenario table; this is not a planet claim |
| `degenerate_evidence` | A missing-input, numerical or effective-sample-size gate prevented a usable normalization. Treat the run as deferred, not as a probability of zero |
| `all_neginf` | Every sampled scenario had zero finite weight. `FPP` may show the normalization fallback `1.0`; that is a failed computation, not a measured FPP |
| `anomaly` | `NaN` or `+inf` appeared in the evidences; no FPP or NFPP is usable |

`not-run` is the default status of an in-memory `TrexResult` before any
calculation and is not a normal output file. "Deferred" and "paused" are
lifecycle outcomes, not additional `status` strings.

The normal calculation path publishes a completed result only for `ok`; a
non-`ok` result raised by `calc_probs` is recorded as `failure.json` through the
target's failure record, with convergence and missing-evidence cases normally
arriving as `TrexEvidenceUnavailableError`. A returned non-`ok` result is a
deferral (exit 3) with no usable probability, and neither form is a probability
of zero.

### `failure.json`

There are two writers, and their records differ.

| Writer | When | Record |
|---|---|---|
| `target.write_outputs` after a failed `calc_probs` | The library or command reached the target API | The target's failure record; the calculation provenance fields are present when the engine built them |
| `trex.app.run._write_failure` | The command refused before a target failure record existed | A compact record without provenance |

Both writers normally include:

| Field | Meaning |
|---|---|
| `claim_eligible` | `false` when the record was built by the engine; an early library refusal that never built provenance can omit it |
| `error_type` | Exception class, usually `TrexEvidenceUnavailableError` |
| `reason` | The message the exception raised |
| `operational_diagnosis` | Operational category and next action |
| `limb_darkening_grid` | Atmosphere-grid summary; empty when no lookups were retained, including the compact command path and early library refusals |
| `diagnostics` | Present when the writer copied a non-empty diagnostics mapping from the exception |

When the engine built provenance, the target-written record also carries its
top-level fields: `n_draws`, `random_seed`, `input_artifacts`,
`sector_inverse_variance_weights`, `exptime_days`, `bin_integration_days`,
`depth_ppm`, `period_days`, `nsamples`, `limb_darkening`, `noise_scope`,
`neighbor_metallicity`, and the optional `noise_model`, `stellar_light`,
`background_population` and measured transit fields. They are top-level fields,
not a nested `calculation_provenance` object. The compact command record does
not contain them.

`operational_diagnosis` has:

| Field | Meaning |
|---|---|
| `status` | `deferred` or `failed` |
| `stage` | Boundary that stopped, such as `input-admission`, `target-binding`, `input-loading`, `checkpoint`, `evidence`, `execution` or `target-api` |
| `category` | `paused`, `convergence`, `missing-evidence`, `numerical-failure`, `invalid-input` or `execution-error` |
| `error_type`, `reason` | Exception identity and message |
| `next_action` | What to inspect or supply next |
| `phase` | Present for a pause when the paused phase is known (`mc_importance` or `nested_escalation`) |

The evidence diagnostics carried by `TrexEvidenceUnavailableError` include the
draw count, per-scenario ESS, the ESS floor, unconverged scenarios and their
probability mass, numerical-failure records, nested records, JAX fallbacks and
phase wall times when those stages ran.

There is no `FPP` or `NFPP` field. A failed calculation therefore cannot be read
as a probability of zero. Non-finite values remain strings such as `"-inf"` or
`"nan"`, never zero; a missing value is `null`.

Common reasons include:

- a scenario stayed below the ESS floor and held more than 10⁻³ of the
  normalized probability after refinement;
- a sampled scenario had no finite weight and could change the decision;
- a supported likelihood failed numerically;
- a required input was missing, so a scenario could not be evaluated;
- the calculation paused at its wall-time deadline. A command-line pause exits
  with code 75 and writes no `failure.json`; a library caller receives
  `NestedCheckpointPaused`.

A recorded numerical likelihood failure is an execution failure, not an
under-sampled scientific deferral. The operational category and exit code
distinguish them.

### Existing outputs are protected

A completed, failed or deferred output is never overwritten. A new run requires
an absent output directory, or an empty directory that belongs to a resumable
run (the checkpoint's `monte-carlo.json` exists). A non-empty directory, a
symlink, a non-directory, or an empty directory with no resumable handoff is
refused. When a refusal is published, it creates `failure.json` only in a new
output directory; publication failure propagates, so a record is not
guaranteed.

Inside an accepted output, each file is published atomically. A publication
failure can leave an incomplete output directory, but it cannot silently replace
a completed file.

## 4. Checkpoints and resume

`calc_probs` accepts `escalation_checkpoint_directory`. When it is omitted, the
path is selected in this order:

1. `TREX_ESCALATION_CHECKPOINT_DIRECTORY`, when set;
2. `$XDG_STATE_HOME/trex/checkpoints/<random-id>`, or
   `~/.local/state/trex/checkpoints/<random-id>` when `XDG_STATE_HOME` is unset.

The selected path is recorded on the target as
`star.escalation_checkpoint_directory`. In the API path the directory is
created only when refinement needs it; the command creates its checkpoint
directory before the calculation.

`python -m trex.run` defaults to `<OUTPUT_DIRECTORY>.escalation` and accepts
`--escalation-checkpoint-directory`. The directory must:

- be outside the output directory,
- not be a symlink,
- be owned by the current user,
- have no group or other permission bits;
- be created with mode `0700` when the command creates it.

When refinement pauses, the checkpoint retains the Monte Carlo state in
`monte-carlo.json` and, once nested refinement has started, its sampler state in
the refinement files. The command writes no `summary.json` or `failure.json`.
The command prints the checkpoint path and exits with status 75. A library call
raises `NestedCheckpointPaused` and leaves the caller to retain the state; if the
caller then calls `write_outputs`, it publishes the paused failure record with
category `paused`.

Resume requires:

- the same checkpoint directory and a matching `monte-carlo.json`, including
  its recorded integer run seed;
- the same three input file digests, draws, supersampling and estimator;
- the same output state: absent, or an empty directory with that handoff;
- the same TREX source/tables and integration inputs for the nested sampler
  state.

A pause is a recovery state, not a deferral or an execution failure: it retains
the seed, budget and input identity and produces no probability. The handoff
binding compares the input digests of `trex-input.json`,
`arrays.npz` and `background.csv`, plus `draws`, `nsamples` and the selected
estimator. The nested checkpoints additionally bind TREX source bytes,
atmosphere table digests and their integration controls. A changed input or
control is refused rather than silently reused. Do not delete the checkpoint of
a paused run unless you intend to restart refinement from the beginning.

The default estimator is dynesty. `TREX_NESTED_ESTIMATOR=blackjax` selects the
experimental BlackJAX nested-slice route, which has its own native checkpoint
(`.blackjax.npz`); it can continue from a retained boundary under the same
identity rules. BlackJAX remains experimental and requires its own supported
configuration. A non-default BlackJAX policy (`TREX_BLACKJAX_PRECOND`,
`TREX_BLACKJAX_BATCH_INIT` or `TREX_BLACKJAX_W_MARCH`) is part of the native
checkpoint identity, so it cannot resume a default-policy checkpoint. A
non-default `TREX_IMPORTANCE_PROPOSAL` is part of the command handoff and result
identity. Neither inherits the default configuration's calibration marker.
There is no promise that a checkpoint resumes across TREX versions, atmosphere
tables, estimators, devices or changed controls; such a change either fails
identity comparison or starts a new refinement. Keep the checkpoint with the
inputs that produced it.

## 5. Plots

`star.plot_fits(time, flux, err)` returns a matplotlib figure showing the light
curve and a representative draw for each scenario, with residuals. It requires:

- a completed, non-degenerate result;
- inputs exactly equal to the retained likelihood observations, including the
  same uncertainty array.

With `save=True` it writes a PDF and returns `None`; with `save=False` it
returns the open figure. The default filename is `TIC<ID>_fits`, and `.pdf` is
appended when absent. The plot labels distinguish Monte Carlo representative
draws from substituted evidences and disclose unavailable draws. It is a visual
check, not a posterior predictive check.

`star.plot_field(sector=None, ap_pixels=None, save=False)` shows the acquired
TESS image for one sector with the catalog sources and, optionally, an aperture.
It is available only for an acquired scene; `from_scene` does not synthesize
pixel data. With `save=True` it writes a PDF (default
`TIC<ID>_sector<SECTOR>`) and returns `None`. Specify `sector` when more than
one sector was acquired.

## 6. Logging

The standalone commands (`python -m trex.run`, `python -m trex.doctor`,
`python -m trex.example`) write structured JSONL diagnostics to stderr and, by
default, to a file. Each command invocation gets its own directory:

```text
$XDG_STATE_HOME/trex/log/<UTC timestamp>-<invocation id>/
```

When `XDG_STATE_HOME` is unset or not absolute, the root is
`~/.local/state/trex/log/`. Each process in the invocation writes its own
`trex-<pid>-<uuid>.jsonl` file in that directory; workers do not share a file.
A checkout may select `scratch/logs` with `--log-dir`, but that is never the
default.

| Option | Effect |
|---|---|
| `--log-dir DIR` | Write per-process logs under `DIR` |
| `--no-log-file` | Use stderr only, including workers; mutually exclusive with `--log-dir` |

If the default log root is unavailable, the command continues on stderr. If an
explicit `--log-dir` is unavailable, the command does not start. Logs are
disposable diagnostics: gates, resume and scientific provenance never read
them, and `--no-log-file` does not change a result. The owning result record,
not the log, retains the facts needed to interpret a calculation.

## 7. Exit codes

These codes belong to the command-line layer. Library calls (`calc_probs`,
`run_trex_vetting`, `write_outputs`) raise exceptions and do not return exit
codes.

| Code | Meaning |
|---:|---|
| `0` | Completed with `status: ok` and a reported FPP/NFPP pair |
| `1` | Execution failed, including a publication failure; the original exception is retained |
| `2` | Input, checkpoint or output refused before a result existed; a new output may retain `failure.json` |
| `3` | The calculation ran but required evidence stayed unresolved; `failure.json` (or a returned non-`ok` result) retains the reason, and no FPP/NFPP is reported as a success |
| `75` | Refinement paused at a retained checkpoint; the command creates no output file, and rerunning the same command with the same inputs and controls continues it |

## 8. A reading checklist

Read in this order:

1. `status == "ok"`. For any other status, stop treating the numbers as a
   measurement and read the failure record or diagnostics.
2. `unavailable_scenarios` is empty. A missing-input scenario blocks a
   published normalization rather than silently contributing zero; a
   `host-excluded` row is a separate physical exclusion.
3. `effective_sample_sizes`, `scenario_evidence_refinement` and
   `supplemental_nested_evidence` agree: every scenario has a finite ESS or an
   explicit substitution, sub-floor scenarios are refined or below the routing
   allowance, and every substituted estimate names its estimator.
4. `FPP` and `NFPP` are finite and consistent with the scenario table. The
   `fpp`, `nfpp` and `overall` diagnostics are a threshold screen, not a claim.
5. `scenarios.csv` tells a physically plausible story: the dominant scenarios,
   host masses and radii, impact parameters and periods make sense for the star
   and the light curve.
6. `calculation_provenance` matches the inputs you intended: period, depth,
   exposure, bin width, seed, sector weights, input hashes and the optional
   noise, stellar-light or background models.
7. `input_identity`, the random seed, draw count and inputs are retained, so the
   same calculation can be replayed with the same software, estimator, device and
   execution path. A resumed run restores the recorded seed rather than drawing a
   new one. Repeatability does not make the model correct.

Cross-check the interpretation against [How TREX works](how-trex-works.md) and
[Concepts](concepts.md), and use [FAQ and troubleshooting](faq.md) when a field
signals a problem.

## 9. Limits

A low ESS or a missing scenario makes the result unreliable. The ESS measures
how concentrated the realized weights are; it does not repair missing support
or a missing model. A run that cannot resolve its evidence should defer rather
than publish a seed-dependent FPP.

A high ESS does not exclude an unsampled explanation. It says the draws that
were evaluated carry spread-out weights; it cannot prove that every scenario,
mode or population was represented. The FPP is conditional on the modeled
scenarios, priors and calibrated inputs, and `claim_eligible` is always `false`.

A scientific deferral is not automatically a software defect. A convergence or
missing-evidence deferral can be the honest result of limited observations or
an unsupported model; a numerical-failure or execution-error category points to
a failing implementation path. Read `operational_diagnosis` before deciding
which remedy applies.

## Next

- [Prepared runs](prepared-runs.md): the command-line path and its input files.
- [Inputs](inputs.md): the scene, photometry, weights and noise calibration.
- [How TREX works](how-trex-works.md): what the evidences and refinements mean.
- [Concepts](concepts.md): FPP, NFPP and ESS definitions.
- [FAQ and troubleshooting](faq.md): what to do when a field signals a problem.
- [API reference](api.md): the in-memory result and failure objects.
