# Module and function reference

This page maps the code that carries a run from inputs to outputs: which module
owns which responsibility, and what that module must not be used for. Exact
signatures and defaults are in the [Python API](api.md) and [Inputs](inputs.md);
the mathematics is in [How TREX works](how-trex-works.md).

Names starting with `_` are private helpers with no compatibility promise, and
the `_jax_*`, `_blackjax_*` and other underscore-prefixed modules are
implementation details. A public name listed here is one you can import from
the stated module and use without relying on private call order. Some tables
also name internal stage owners (`prepare_vetting_inputs`, `load_input`,
`synthetic_inputs`, runtime helpers) because they explain the implementation;
unless [Python API](api.md) documents a stable contract for them, they are not
part of the compatibility promise.

The short compatibility modules `trex.target`, `trex.scene`, `trex.engine`,
`trex.run` and `trex.example` replace themselves with their `trex.app` owner;
`trex.doctor` re-exports its supported names. Their only extra code is the
`__main__` command dispatch. See
[Import paths](api.md#import-paths-and-compatibility-identities).

## How the package is layered

| Layer | Module group | Owns | Not for |
|---|---|---|---|
| User workflow | `trex.app`, `trex.acquisition` | Scene construction or acceptance, binding, engine invocation, result files, commands | Likelihood mathematics or reinterpreting a result |
| Inference | `trex.evidence` | Scenario draws, likelihood integration, normalization, refinement, checkpoints, replay | Acquiring catalog data or inventing missing scenario support |
| Physics | `trex.models` | Forward models, likelihoods, noise and flux calibration, orbital geometry | Population frequencies or priors |
| Populations | `trex.populations` | Prior densities, companion and background populations, imaging constraints, spatial diagnostics | Observed photometry or likelihood evaluation |
| Photometry | `trex.photometry` | Phase binning, covariance, Gaussian baseline, magnitude–flux conversions | Evidence normalization or scenario weights |
| Stellar | `trex.stellar` | Mass–radius–temperature relations | Atmosphere grids or limb darkening |
| Stellar light | `trex.stellar_light` | Limb darkening, passbands, atmosphere grids, physical light models | Free-floating fits or unverified band conversions |
| Accelerators | `trex.accelerators` | Batched JAX kernels and device dispatch | A second scientific model; the CPU path remains the reference |
| Runtime | `trex.runtime` | Hardware/resource policy, worker count, logging, progress | Scientific budgets, thresholds or convergence policy |
| Foundation | `trex.foundation` | Constants, exceptions, small numerical helpers | Science policy or acquisition |

## One run in call order

| Stage | Typical calls | What exists afterwards |
|---|---|---|
| Acquire and bind | `target.from_tic` or `target.from_scene`, both ending in `bind_scene` | A validated `TargetScene` and a bound `target` |
| Calibrate noise | `calibrate_trex_noise` | A per-sector uncertainty record; the caller fixes `sigma` from `native_uncertainty_factor` |
| Admit inputs | `prepare_vetting_inputs` (or `target.doctor`) | Arrays and resolved scene values, without draws |
| Evaluate | `run_trex_vetting` → `calc_target_evidences` → `simulate_TP`/`simulate_EB` and `lnL_TP`/`lnL_EB` | Per-scenario log evidences, ESS and fit records |
| Normalize | `scenario_period_log_priors` → `compute_fpp_nfpp` (inside `generate_diagnostics`) | `FPP`, `NFPP`, the probability vector and diagnostics |
| Refine if needed | `evidence_needs_escalation` → `run_mc_importance` → `run_nested_escalation` | Replacement evidences and a checkpoint |
| Report | `write_outputs`, optionally `plot_fits` | `summary.json` and `scenarios.csv`, or `failure.json` |

The arrows are dependencies, not a claim that each function calls the next
directly. `calc_probs` is the compatibility wrapper; the engine selects the
scenario-specific model and likelihood internally.

## `trex.app`: user workflow

This layer owns the objects a user touches: a scene, a bound target, the engine
call, result files and the command-line entry points. It validates and passes
inputs; it does not contain the probability mathematics. Use it for a complete
vetting run, not for calling a sampler directly.

### `target`

`target` binds a scene to observations, owns the result attributes and provides
the reference-compatible acquisition path. The class methods are the documented
workflow; direct construction is the TRICERATOPS-compatible acquisition path.

| Member | Owns |
|---|---|
| `target.from_tic(...)` | Acquire a scene for a TIC ID through `trex.acquisition`, then bind it |
| `target.from_scene(...)` | Bind a validated `TargetScene` without network access |
| `star.bind_scene(...)` | Attach sector weights, depth, exposure, seed, baseline/light model and input hashes; verify the scene |
| `star.calc_probs(...)` | Run the scenario evidence calculation and set the result attributes |
| `star.doctor(...)` | Run the engine's pre-draw admission and return a JSON-safe report |
| `star.write_outputs(...)` | Write the result or the failure record to a new directory |
| `star.plot_fits(...)` | Plot the retained representative model of each scenario |
| `star.plot_field(...)` | Plot the acquired TESS image with catalog sources, when pixel data exist |
| `star.add_star(...)`, `star.remove_star(...)`, `star.update_star(...)` | Edit the acquisition star table before binding |
| `star.get_spoc_apertures()` | Return the SPOC aperture pixel arrays per sector |
| `star.calc_depths(...)` | Compute the reference Gaussian dilution diagnostic; not a calibrated input |

Do not use `calc_depths` to supply aperture evidence to `calc_probs`; the
calibrated `TargetScene` owns inference inputs. Do not edit the star table after
binding and expect the old result to remain valid: the next calculation refuses
until the scene is rebound.

### `TargetScene`

`TargetScene` validates and stores the stellar scene: target parameters,
aperture fractions, resolved neighbours, an unresolved-companion bound and a
background population. It is the input contract for the engine and for a
portable prepared run.

| Member | Owns |
|---|---|
| `TargetScene(...)` | Validate and store every scene field |
| `to_dict()` | Export constructor inputs for transfer or a prepared run |
| `from_dict(payload)` | Revalidate an exported scene and its background file |
| `verify_background()` | Reject a background file whose bytes changed |
| `background_population()` | Load the retained background stars as arrays |
| `target_aperture_flux_fraction(weights)` | Reduce the target's per-sector fractions with the observations |
| `neighbor_dicts_for_evidence(weights)` | Prepare per-neighbour host records for the engine |
| `neighbor_masses_radii()` | Return neighbour mass, radius and brightness-difference arrays |
| `companion_delta_mags()` | Return the brightness difference of every resolved neighbour |
| `n_neighbors`, `has_contrast_data` | Count neighbours; report whether a measured contrast curve is stored |

Do not use `localization_evidence` as a source-location likelihood: it is a
retained diagnostic and the neighbour spatial log priors are neutral. Do not
treat `to_dict()` as a copy of the background file; the path and digest are
recorded and the file must travel separately.

### Engine, commands and diagnostics

| Name | Module | Owns |
|---|---|---|
| `run_trex_vetting(...)` | `trex.app.engine` | The engine entry point: scene plus observations to `TrexResult` |
| `prepare_vetting_inputs(...)` | `trex.app.engine` | The canonical pre-draw admission shared by `calc_probs` and the doctor |
| `TREX_PRODUCTION_N_DRAWS` | `trex.app.engine` | The 1,000,000-draw production budget |
| `scenario_monte_carlo_evidence(result)` | `trex.app.engine` | First-stage log evidence, its `1/√ESS` error and ESS per scenario |
| `scenario_evidence_refinement(result)` | `trex.app.engine` | Per-scenario record of whether a later stage replaced the Monte Carlo estimate |
| `scenario_status_records(result)` | `trex.app.engine` | Availability status and reason per scenario |
| `unavailable_scenario_records(result)` | `trex.app.engine` | Scenarios whose model input was unavailable; a run containing one does not publish FPP/NFPP |
| `load_input(directory)` | `trex.app.run` | Rebuild a prepared run's scene, arrays and binding |
| `main(arguments=None)` | `trex.app.run` | The `python -m trex.run` command |
| `synthetic_inputs(directory)` | `trex.app.example` | Build the invented scene used by the offline demonstration |
| `read_aij_observation(...)` | `trex.app.aij_loader` | Read AstroImageJ differential photometry into a joint-observation input without detrending |
| `run_observation_vetting(...)` | `trex.app.observation_vetting` | Run the scenario engine on native-time joint photometry |
| `export_observation_input(...)` | `trex.app.observation_io` | Write a portable joint observation directory with its compiled light assets |
| `inspect_inputs(...)`, `probe_mast(...)`, `group_failures(...)` | `trex.app.doctor` | The no-draw input check, bounded MAST probe and failure-record grouping |

The engine's limb-darkening node selection and photometry admission are internal
helpers (`target_limb_darkening_node`, `admitted_photometry`) used by the
admission path and by `calc_probs`; they are not a separate public contract.
`inspect_inputs` runs the real admission; it does not establish convergence or
a valid FPP. `synthetic_inputs` and the example budget are software checks, not
scientific results.

## `trex.evidence`: from draws to FPP

This layer draws simulated systems for each scenario, evaluates how well each
draw matches the light curve, integrates those likelihoods into evidences, and
normalizes them into probabilities. Refinement re-estimates the same integral
when a Monte Carlo estimate is dominated by too few draws. Use it through
`run_trex_vetting`; call its pieces directly only when you own the inputs and
the convergence decision.

### Scenario evaluation

| Name | Module | Owns |
|---|---|---|
| `calc_target_evidences(...)` | `trex.evidence.mc_target` | Compute log evidences for the target and every capable neighbour |
| `resolved_neighbor_host(...)` | `trex.evidence.mc_target` | Decide whether one neighbour can host the event and with which parameters |
| `neighbor_metallicity_context(...)` | `trex.evidence.mc_target` | Record the composition actually used, including the target-value fallback |
| `HostMeasure`, `CharacterizedHostIntegrand` | `trex.evidence.host_measure` | Represent and integrate a resolved host's measured parameter distribution |

Do not call `calc_target_evidences` with a scene that has not been validated or
without `scene.feh_dex`; the admission exists to prevent omitted scenarios from
biasing FPP low.

### Normalization and diagnostics

| Name | Module | Owns |
|---|---|---|
| `compute_fpp_nfpp(...)` | `trex.evidence.marginal_likelihoods` | Apply the period prior, normalize log evidences, and sum FPP and NFPP |
| `scenario_period_log_priors(period_days, count)` | `trex.evidence.marginal_likelihoods` | Per-scenario log period prior for the first `count` slots |
| `degenerate_evidence_scenarios(...)` | `trex.evidence.marginal_likelihoods` | Scenarios whose evidence is not a converged integral, with their ESS |
| `unconverged_probability_mass(...)` | `trex.evidence.marginal_likelihoods` | Probability still held by finite-evidence scenarios that did not converge |
| `generate_diagnostics(...)` | `trex.evidence.diagnostics` | Build the named threshold, availability and numerical checks |
| `residual_could_change_decision(...)` | `trex.evidence.diagnostics` | Whether residual sub-floor scenarios could flip the joint FPP/NFPP threshold decision |
| `TrexDiagnostic` | `trex.evidence.diagnostics` | One check: name, status, message, value and threshold |
| `TrexResult` | `trex.evidence.diagnostics` | The complete result object, including probabilities, ESS and diagnostics |

`compute_fpp_nfpp` is a normalization/diagnostic function: it cannot recover a
scenario that was not evaluated, and its status is not a validation claim.

### Refinement and nested evidence

| Name | Module | Owns |
|---|---|---|
| `evidence_needs_escalation(...)` | `trex.evidence.evidence_escalation` | Shared trigger for whether refinement should start |
| `stage_scenarios(...)` | `trex.evidence.evidence_escalation` | Scenarios below the ESS floor close enough to the best evidence to matter |
| `residual_could_flip_decision(...)` | `trex.evidence.evidence_escalation` | Whether unresolved scenarios can flip the decision |
| `run_mc_importance(...)` | `trex.evidence.mc_importance` | Defensive importance-sampling refinement of finite-evidence sub-floor scenarios |
| `run_nested_escalation(...)` | `trex.evidence.evidence_escalation` | Nested-sampling refinement of scenarios that still fail the gate |
| `tp_scenario_nested_evidence(...)` | `trex.evidence.nested_evidence` | Nested evidence for a transiting-planet scenario |
| `eb_scenario_nested_evidence(...)` | `trex.evidence.nested_evidence` | Nested evidence for an eclipsing-binary scenario |
| `native_nested_kernel(...)` | `trex.evidence.nested_evidence` | Build the experimental nested-slice sampling kernel; default runs use dynesty |
| `bound_companion_population(...)`, `background_star_population(...)` | `trex.evidence.nested_population` | Nested-sampling population blocks for bound and background hosts |
| `NestedEvidenceUnavailableError` | `trex.evidence.evidence_escalation` | The nested estimator could not be used for this scenario |

`run_mc_importance` and `run_nested_escalation` change the estimator, not the
scenario definitions or the scientific budget. Do not use them to turn an
under-sampled result into a passing one without retaining the replacement
record and its convergence evidence.

### Checkpoints and replay

| Name | Module | Owns |
|---|---|---|
| `nested_checkpoint(path, *, deadline)` | `trex.evidence.nested_checkpoint` | Bind a recovery path and monotonic deadline for a resumable refinement |
| `checkpoint_identity(value)` | `trex.evidence.nested_checkpoint` | Tie a checkpoint to its inputs, calculation controls and source bytes |
| `save_completed(...)`, `load_completed(...)` | `trex.evidence.nested_checkpoint` | Persist or restore a completed nested run |
| `load_importance_progress(...)`, `save_importance_progress(...)` | `trex.evidence.nested_checkpoint` | Resume the defensive importance stage |
| `load_importance_adaptation(...)`, `save_importance_adaptation(...)` | `trex.evidence.nested_checkpoint` | Retain the importance proposal's adaptation state |
| `SamplerCheckpoint` | `trex.evidence.nested_checkpoint` | Save and restore the sampler's own state and evaluation count |
| `NestedCheckpointPaused` | `trex.evidence.nested_checkpoint` | Raised when the time limit stops refinement but state is kept |
| `CheckpointInputMismatch` | `trex.evidence.nested_checkpoint` | Retained state belongs to different inputs |
| `replay_representative_model(...)` | `trex.evidence.fit_replay` | Rebuild one retained draw's model curve for inspection |

A checkpoint is execution state, not evidence of estimator accuracy. Native
dynesty state contains pickle data; restore only files created by the same
pinned TREX runtime, never an untrusted upload.

## `trex.models`: forward models and noise

A forward model turns one draw into the light curve the telescope would record;
the likelihood compares that prediction with the data. The noise functions
decide how much to trust each point, and the orbital functions supply the
geometry the model needs. This layer does not choose priors or scenario
frequencies.

### Transit, eclipse and likelihood

| Name | Module | Owns |
|---|---|---|
| `simulate_TP(...)` | `trex.models.likelihoods` | Predict a transiting-planet light curve for one draw |
| `simulate_EB(...)` | `trex.models.likelihoods` | Predict an eclipsing-binary light curve, including the secondary |
| `quadratic_transit_flux(...)` | `trex.models.transit_flux` | Compute instantaneous NumPy transit flux; does not average over exposures |
| `lnL_TP(...)` | `trex.models.likelihoods` | Gaussian log likelihood of a planet draw against the observations |
| `lnL_EB(...)` | `trex.models.likelihoods` | Gaussian log likelihood of a binary draw against the observations |
| `DrawUnsupportedError` | `trex.models.likelihoods` | A draw's forward model could not be evaluated numerically |

The likelihoods share one fixed, caller-calibrated `sigma`. Do not fit a
scenario-specific noise scale with these unnormalized expressions.

### Noise and correlated-noise diagnostics

| Name | Module | Owns |
|---|---|---|
| `calibrate_trex_noise(...)` | `trex.models.noise` | Estimate the shared effective-error factor from native out-of-transit photometry |
| `out_of_transit_mask(...)` | `trex.models.red_noise` | Build the out-of-transit mask for a candidate ephemeris |
| `median_cadence_days(...)` | `trex.models.red_noise` | Median positive sampling interval in days |
| `transit_bin_size_points(...)` | `trex.models.red_noise` | Number of cadences spanning the transit duration |
| `pont_2006_beta(...)` | `trex.models.red_noise` | Pont et al. (2006) binned-RMS factor for one bin size |
| `winn_2008_multiscale_beta(...)` | `trex.models.red_noise` | Median of that factor over several averaging times |
| `compute_red_noise_diagnostic(...)` | `trex.models.red_noise` | Measure correlated noise in out-of-transit residuals and record it |
| `RedNoiseScale` | `trex.models.red_noise` | One retained binned-RMS scale estimate |

`calibrate_trex_noise` is a one-time, per-sector calibration on native times.
Do not estimate beta on folded times, and do not add a second per-scenario
inflation.

### Orbital mechanics and flux calibration

| Name | Module | Owns |
|---|---|---|
| `semi_major_axis_cgs(...)` | `trex.models.orbital_mechanics` | Semi-major axis in cm from Kepler's third law |
| `a_over_Rs(...)` | `trex.models.orbital_mechanics` | Scaled semi-major axis `a/R_*` |
| `impact_parameter(...)` | `trex.models.orbital_mechanics` | Impact parameter for circular or eccentric orbits |
| `secondary_eclipse_phase(...)` | `trex.models.orbital_mechanics` | Secondary-eclipse phase from Keplerian mean anomalies |
| `companion_flux_ratio(...)` | `trex.models.flux_calibration` | Convert a companion's share of combined flux to a flux ratio |
| `validate_flux_uncertainty(...)` | `trex.models.flux_calibration` | Validate and expand a scalar or per-point uncertainty |
| `renorm_flux_to_host(...)` | `trex.models.flux_calibration` | Renormalize aperture photometry onto one candidate host |
| `dilute_flux(...)` | `trex.models.flux_calibration` | Apply dilution by a companion or background source |

## `trex.populations`: priors and populations

These functions describe what kinds of systems are plausible before the light
curve is considered: planet occurrence, binary companions, background stars and
which companions the imaging could have detected. A prior is an assumption made
before looking at this target's data.

The public sampler API lives in `trex.populations.priors`, which also re-exports
the companion and mass-ratio samplers. Kernels live in
`trex.populations.companion_priors` and `trex.populations.mass_ratio_priors`.

### Planets, periods and mass ratios

| Name | Module | Owns |
|---|---|---|
| `sample_rp(...)`, `radius_quantile(...)` | `trex.populations.priors` | Draw planet radii from the broken power law, and invert it |
| `sample_inc(...)`, `sample_ecc(...)`, `sample_w(...)` | `trex.populations.priors` | Draw inclination, eccentricity and argument of periastron |
| `planet_period_prior(...)`, `eb_period_prior(...)` | `trex.populations.priors` | Period factor over `P ± 0.1 d` |
| `scenario_period_prior(...)`, `scenario_period_log_prior(...)` | `trex.populations.priors` | Period factor for one scenario's applied model period, and its natural log |
| `sample_q(...)`, `sample_q_host_mass_ratio(...)` | `trex.populations.priors` | Draw a binary mass ratio; the host-mass variant uses a per-draw host mass |
| `eb_mass_ratio_support(...)` | `trex.populations.priors` | Allowed `q` range for the EB and twin scenarios |
| `host_mass_ratio_inverse_cdf(...)` | `trex.populations.priors` | Inverse CDF of the host-mass-dependent mass-ratio law |
| `moe2017_binary_eccentricity_eta(...)` | `trex.populations.priors` | Period-dependent eccentricity exponent |

### Companions

| Name | Module | Owns |
|---|---|---|
| `moe2017_cumulative_companion_frequency(...)` | `trex.populations.companion_priors` | Companions per primary below a period |
| `moe2017_largeq_slope(...)`, `moe2017_twin_fraction(...)` | `trex.populations.companion_priors` | Large-mass-ratio slope and near-equal-mass twin excess |
| `companion_log_separation_au(...)` | `trex.populations.companion_priors` | Convert companion period to semi-major axis |
| `companion_population_frequency(...)` | `trex.populations.companion_priors` | Bound-companion weight before imaging constraints |
| `sample_companion_log_period(...)` | `trex.populations.companion_priors` | Draw companion orbital periods |
| `sample_companion_log_projected_separation(...)` | `trex.populations.companion_priors` | Draw Winters et al. (2019) projected separations |
| `sample_companion_log_semimajor_axis(...)` | `trex.populations.companion_priors` | Draw moment-matched semi-major axes |
| `companion_log_period_from_log_projected_separation(...)` | `trex.populations.companion_priors` | Convert projected separation to period |
| `companion_hierarchical_stability_supported(...)` | `trex.populations.companion_priors` | Apply the Mardling & Aarseth stability criterion |
| `sample_q_companion(...)` | `trex.populations.companion_priors` | Draw bound-companion mass ratios |

### Background, imaging and spatial weights

| Name | Module | Owns |
|---|---|---|
| `background_population_expectation(...)` | `trex.populations.priors` | Expected number of simulated background stars inside a circular field |
| `load_trilegal_population(...)` | `trex.populations.background` | Read a TRILEGAL cache into per-star arrays |
| `background_metallicity_dex(...)` | `trex.populations.background` | Return the background stars' [M/H], requiring one finite value each |
| `contrast_limit_at_separation(...)` | `trex.populations.companion_imaging` | Faintest detectable Δmag at a separation |
| `image_source_flux_ratio(...)` | `trex.populations.companion_imaging` | Flux ratio of the image source containing the eclipsing pair |
| `imaging_log_prior(...)`, `binary_imaging_log_prior(...)` | `trex.populations.companion_imaging` | Apply the contrast-curve detection limit to a source or a drawn binary |
| `ContrastCurve`, `ImagingConstraint` | `trex.populations.companion_imaging` | A validated curve; a sampled source with its measured contrast limit |
| `localization_spatial_evidence(...)` | `trex.populations.spatial_prior` | Read retained difference-image (centroid) evidence |
| `spatial_evidence_from_localization_payload(...)` | `trex.populations.spatial_prior` | Validate that evidence, or state why it is unavailable |
| `neighbor_log_priors(...)` | `trex.populations.spatial_prior` | Zero log spatial weight per resolved neighbour (no spatial preference) |

The retained centroid diagnostic does not implement a calibrated scenario
likelihood. Do not use it to reweight, add or delete a neighbour.

## `trex.photometry`: arrays and baselines

These helpers prepare the numerical arrays the likelihood consumes: phase bins,
their covariance, an optional baseline fit on the unbinned light curve, and
conversions between magnitudes and flux ratios.

| Name | Module | Owns |
|---|---|---|
| `TimeAxis`, `rebase_bjd_tdb(...)` | `trex.photometry.time_axis` | Retained already-barycentric TDB streams, precise reference/boxcar-anchor arithmetic |
| `barycentric_tdb(...)` | `trex.photometry.time_conversion` | Explicit clock conversion and distant-source ground observer correction; not ambiguous HJD or spacecraft frames |
| `ExposureResponse` | `trex.photometry.exposure` | Shared/per-row temporal quadrature and disjoint uniform live windows for scalar forward models |
| `weighted_phase_bins(...)` | `trex.photometry.binning` | Bin phase-folded flux and errors, retaining empty bins |
| `folded_covariance(...)` | `trex.photometry.covariance` | Measure adjacent-bin covariance on shifted folds (descriptive) |
| `GaussianBaseline` | `trex.photometry.baseline` | Gaussian prior on the coefficients of a fixed baseline model |
| `BaselineLikelihood` | `trex.photometry.baseline` | Prepared aperture likelihood with an integrated baseline nuisance prior |
| `J_Ks_to_Tmag(...)` | `trex.photometry.photometric_relations` | Convert 2MASS colours to a TESS magnitude |
| `delta_mag_to_flux_ratio(...)`, `flux_ratio_to_delta_mag(...)` | `trex.photometry.photometric_relations` | Convert between magnitude difference and flux ratio |

`folded_covariance` is a diagnostic, not a noise likelihood. `GaussianBaseline`
is an advanced single-sector, unbinned-native-time option; do not combine it
with phase-bin averaging.

## `trex.acquisition`: how a TIC scene is built

`acquire_target_scene(...)` is the orchestrator behind `target.from_tic`. The
other entry points are useful when you need one retained product rather than a
whole scene. This layer retrieves and validates evidence; it does not run
inference.

| Name | Module | Owns |
|---|---|---|
| `acquire_target_scene(...)` | `trex.acquisition.scene` | Acquire every scene input for a TIC ID and return a validated scene |
| `acquire_tpf_products(...)` | `trex.acquisition.tpf` | Download and retain the SPOC target-pixel files |
| `acquire_aperture_images(...)` | `trex.acquisition.aperture` | Fit retained pixel images with explicit transit masks |
| `acquire_catalog_aperture(...)` | `trex.acquisition.aperture` | Build the catalog flux-ratio aperture model from SPOC apertures and WCS |
| `acquire_field_context(...)` | `trex.acquisition.field` | Retain TIC and Gaia responses and their association |
| `acquire_tic_neighbors(...)` | `trex.acquisition.field` | Retain TIC and Gaia rows for each resolved neighbour |
| `acquire_host_tables(...)` | `trex.acquisition.neighbor_tables` | Retain Gaia, StarHorse and SHBoost host records |
| `acquire_metallicity(...)` | `trex.acquisition.metallicity` | Acquire the ordered APOGEE, GALAH, GSP-Spec and SHBoost composition |
| `acquire_background_population(...)` | `trex.acquisition.background_population` | Get a TRILEGAL population, or the recorded fallback |
| `acquire_trilegal(...)` | `trex.acquisition.trilegal` | Submit, poll and retain a TRILEGAL population |
| `tess_magnitude_from_gaia_photometry(...)` | `trex.acquisition.photometry` | Convert Gaia photometry with the TIC v8 relation |
| `GaiaDetectionConstraint` | `trex.acquisition.gaia_constraints` | A source-scoped record; TREX 1.0 accepts only evidence-neutral records |
| `NetworkPolicy` | `trex.acquisition.transport` | Network permission, timeout, retries and response bounds |
| `AcquisitionTransport` | `trex.acquisition.transport` | Bounded HTTP transport with typed failures |
| `AcquisitionError` | `trex.acquisition.transport` | A scene input could not be acquired without inventing data |
| `TrilegalUnavailableError` | `trex.acquisition.trilegal_errors` | The background provider failed or is unavailable |

Supporting modules, each responsible for one archive or one scene rule:

| Module | Owns |
|---|---|
| `apogee_dr17` | APOGEE DR17 iron abundances and associations |
| `background_population` | TRILEGAL/fallback record selection and provenance |
| `dust_map` | SFD98 field extinction for the TRILEGAL request |
| `field` | TIC and Gaia field context |
| `gaia_crossmatch` | Gaia DR3-to-DR2 counterparts |
| `gaia_sources` | Gaia cone query and epoch propagation |
| `galah_dr3` | GALAH DR3 release, exact-ID TAP batches and associations |
| `gspspec` | Gaia GSP-Spec spectroscopic rows and calibration |
| `host_distributions` | StarHorse and SHBoost catalog-parameter distributions |
| `host_physics` | Target mass, radius and temperature precedence |
| `identity` | Exact catalogue identifiers |
| `lsst_sim` | Recorded LSST-simulation background fallback |
| `metallicity` | Ordered composition ladder and provenance |
| `mist` | MIST stellar-evolution grid, parallax zero point and G-band inference |
| `neighbor_tables` | Gaia, StarHorse and SHBoost table retrieval |
| `photometry` | Gaia-to-TESS magnitude relation |
| `scene` | Scene orchestration and validation |
| `scene_values` | Neighbour property resolution and refusal reasons |
| `storage` | Durable, atomic retention of responses |
| `tic` | TIC target, cone and neighbour rows |
| `tpf` | SPOC target-pixel file loading |
| `transport` | Network policy, transport and acquisition errors |
| `trilegal` | TRILEGAL form submission, polling and retention |
| `trilegal_tls` | TLS certificate chain checks for the TRILEGAL host |
| `votable` | VOTable (XML table) parsing |

Do not construct a scene by filling missing archive values with defaults. A
stage that cannot supply its record must remain visibly unavailable and the
scene must refuse or defer.

## `trex.stellar` and `trex.stellar_light`

These layers describe the host star: its mass, radius and temperature, and the
light it emits. The first turns a mass into a radius and an effective
temperature; the second turns an atmosphere, a passband and an extinction into
photons.

### Stellar relations

| Name | Module | Owns |
|---|---|---|
| `stellar_relations(...)` | `trex.stellar.stellar_relations` | Estimate radius and effective temperature from mass |
| `stellar_radii_in_domain(...)` | `trex.stellar.stellar_relations` | Radii without the domain checks or temperatures |
| `stellar_masses_of_radii(...)` | `trex.stellar.stellar_relations` | Smallest mass reaching each radius in the relation |
| `main_sequence_mass_from_teff(...)` | `trex.stellar.stellar_relations` | Invert the main-sequence `Teff(M)` relation |
| `validate_stellar_masses(...)` | `trex.stellar.stellar_relations` | Enforce the relation's mass domain |
| `STELLAR_RELATIONS_CALIBRATION_STATUS` | `trex.stellar.stellar_relations` | The retained status string, `"uncalibrated"` |

The cool-dwarf node vector remains uncalibrated; `STELLAR_RELATIONS_CALIBRATION_STATUS`
states that, so these relations are scenario inputs, not calibrated stellar
posteriors. Do not use them for atmosphere models or limb darkening.

### Limb darkening

| Name | Module | Owns |
|---|---|---|
| `load_triceratops_tess_grid()` | `trex.stellar_light.host_limb_darkening` | Load the vendored TESS limb-darkening table |
| `TriceratopsTessLimbDarkeningGrid` | `trex.stellar_light.host_limb_darkening` | Select and audit coefficients for a host atmosphere |
| `LimbDarkeningNode` | `trex.stellar_light.host_limb_darkening` | The coefficient source actually used for one request |
| `LimbDarkeningPopulation` | `trex.stellar_light.host_limb_darkening` | Per-draw nodes with the same rounding and audit |
| `summarize_ldc_grid_provenance(...)` | `trex.stellar_light.host_limb_darkening` | Aggregate the lookups for the result |

### Physical stellar light

| Name | Module | Owns |
|---|---|---|
| `StellarLight`, `ComponentLight` | `trex.stellar_light.components` | An explicit physical-light model and one prepared component |
| `PhotonAtmosphereGrid` | `trex.stellar_light.interpolation` | One scaled-solar atmosphere family |
| `PhotonPassband` | `trex.stellar_light.passband` | Photon response and extinction for a passband |
| `integrate_photon_intensity(...)` | `trex.stellar_light.passband` | Integrate a spectrum through a passband into a photon flux |
| `tess_surface_brightness_ratio(...)` | `trex.stellar_light.passband` | TESS-band blackbody surface-brightness ratio |
| `prepare_atmospheres(...)`, `AtmospherePreparation` | `trex.stellar_light.preparation` | Acquire and compile the atmosphere cells a run needs |
| `read_phoenix_intensity(...)` | `trex.stellar_light.release` | Read one actual PHOENIX intensity node |
| `acquire_phoenix_intensity(...)` | `trex.stellar_light.release` | Download one declared atmosphere node |
| `population_atmosphere_demand(...)` | `trex.stellar_light.support` | Plan the atmosphere cells a population will request |
| `UnsupportedAtmosphereError` | `trex.stellar_light.interpolation` | The requested atmosphere is unavailable, which is not zero probability |

A non-TESS contrast band is refused rather than converted; no verified bandpass
conversion is implemented. Physical light requires the target's own `A_V` and
does not fall back to a solar analog.

## `trex.accelerators` and `trex.runtime`

The accelerators evaluate scenarios that have a native batched kernel on CPU or
GPU. Scenarios without one use the scalar CPU path. The runtime functions report
what hardware exists, enforce the memory policy, and choose how many workers a
run may use. Switching device changes speed, not the statistical model.

The batched kernels live in the private `trex.accelerators._jax_models` and
`trex.accelerators._jax_dispatch` modules, with shared float64 setup in
`trex.accelerators._jax_runtime`; they are not a stable public API. Use
`run_trex_vetting`; it selects and falls back automatically and records a
fallback in the result.

| Name | Module | Owns |
|---|---|---|
| `resolve_n_jobs(...)` | `trex.runtime.hardware` | Validate an explicit worker count or supply the hardware default |
| `get_default_n_jobs()` | `trex.runtime.hardware` | Worker default for CPU-bound runs |
| `available_cpu_count()` | `trex.runtime.hardware` | Affinity- and cgroup-bounded logical CPU count |
| `accelerator_memory_floor_bytes()` | `trex.runtime.hardware` | Minimum device memory for accelerator use |
| `accelerator_meets_memory_floor(...)` | `trex.runtime.hardware` | Whether a device passes that policy |
| `configure_jax_memory(...)` | `trex.runtime.hardware` | Ask JAX to allocate memory on demand before its backend starts |
| `device_memory_profile(...)` | `trex.runtime.hardware` | Sample allocator and driver memory counters |
| `device_memory_available_bytes(...)` | `trex.runtime.hardware` | Free memory for the next allocation, or `None` when unknown |
| `device_memory_limit_bytes(...)` | `trex.runtime.hardware` | Addressable JAX allocator capacity |
| `cpu_resource_profile(...)`, `memory_resource_profile(...)` | `trex.runtime.hardware` | Effective CPU and RAM under affinity and container limits |
| `is_memory_exhaustion(exc)` | `trex.runtime.hardware_resources` | Classify an allocation failure before reducing a working set and retrying |
| `AcceleratorLease` | `trex.runtime.hardware` | Exclusive claim on one physical device, recoverable after a crash |
| `accelerator_execution_scope(...)` | `trex.runtime.accelerator_pool` | Hold exclusive device leases for one computation |
| `execution_devices(...)` | `trex.runtime.accelerator_pool` | Restrict a device list to the current allocation |
| `record_accelerator_event(...)` | `trex.runtime.accelerator_pool` | Retain one accelerator decision in the run record |
| `observe_accelerator_admissions(...)` | `trex.runtime.accelerator_pool` | Install a caller-owned observer for device admission records |
| `driver_memory_profile(...)` | `trex.runtime.hardware_drivers` | Fresh driver free/total VRAM and device identity |
| `lease_metadata_lock(...)` | `trex.runtime.hardware_lock` | Lock one device-lease metadata file while it is updated |
| `prepare_jax_cuda_libraries(...)` | `trex.runtime.jax_cuda_runtime` | Prefer the runtime's CUDA linker when a host linker shadows it |
| `emit_progress(...)` | `trex.runtime.progress` | Deliver one progress update to a callback |
| `get_logger(...)`, `host_logger(...)`, `set_host_namespace(...)` | `trex.runtime.logging` | Named loggers and host namespace selection |
| `standalone_cli(...)` | `trex.runtime.logging` | Wrap a module command's logging lifecycle |

In an embedded run, logging uses the host's namespace. Standalone logging
checks TREX source names and live exception classes structurally; it still
filters unrecognized message text and arbitrary payloads.

Runtime overrides are environment variables, not function arguments:
`TREX_N_JOBS` sets the worker count, `TREX_GPU_MIN_MEMORY_BYTES` sets the
device memory floor, `TREX_ACCELERATOR_LEASE_PATH` moves the lease file, and
`TREX_ACCELERATOR_MAX_DEVICES` caps the devices one computation may lease.
The defaults are described in the [Python API](api.md#environment-variables).

Do not use the runtime layer to change draw counts, evidence budgets,
convergence thresholds or claim eligibility. It reports and enforces execution
resources only.

## `trex.foundation`

Shared constants, exceptions and small numerical helpers. Import standards from
`trex.foundation.physical_constants` (canonical owner), the short CGS aliases
from `trex.foundation.constants`, and the two main exception types from
`trex.foundation.errors`.

| Name | Module | Owns |
|---|---|---|
| `TrexEvidenceUnavailableError` | `trex.foundation.errors` | Required evidence is absent or did not converge; `diagnostics` carries the reason |
| `NumericalLikelihoodError` | `trex.foundation.errors` | Supported likelihood evaluations failed numerically; `count` carries how many |
| `GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC` | `trex.foundation.constants` | Default unresolved-source limit, 2.2 arcsec |
| `GRAVITATIONAL_CONSTANT_CGS`, `SPEED_OF_LIGHT_CGS`, `SECONDS_PER_DAY` | `trex.foundation.constants` | Physical constants in the units the engine uses |
| `Msun`, `Rsun`, `Rearth`, `au` | `trex.foundation.constants` | Solar, Earth and astronomical-unit constants |
| `PARTS_PER_MILLION`, `FRACTION_PER_PART_PER_MILLION` | `trex.foundation.physical_constants` | ppm representation factors |

`trex.foundation._numerics` is an internal numerical module; its helpers
(including `phase_bin_widths`) are not a public numerical library.

## Reading a traceback

A traceback usually lands in one of these places:

- **Acquisition:** `trex.acquisition.*`, `AcquisitionError`,
  `TrilegalUnavailableError`, `ApertureSceneError`.
- **Binding and validation:** `TargetScene`, `target.bind_scene`, messages about
  sectors, aperture fractions, metallicity or input hashes.
- **Inference:** `trex.evidence.*`, `TrexEvidenceUnavailableError`,
  `NumericalLikelihoodError`, `DrawUnsupportedError`.
- **Refinement:** `trex.evidence.nested_*`, `NestedCheckpointPaused`,
  `NestedEvidenceUnavailableError`, `CheckpointInputMismatch`.
- **Platform:** `trex.runtime.*`, JAX device or memory messages.

The [FAQ](faq.md) maps user-facing messages to a cause. For what each stage
computes, read [How TREX works](how-trex-works.md); for acquisition, read
[From a TIC ID](from-tic.md); for written fields, read [Outputs](outputs.md).
