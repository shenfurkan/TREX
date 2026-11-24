# Python API

This page lists the supported import paths and the exact call signatures,
defaults, units, return values and errors of the public TREX interface. Field
rules behind the parameters are in [Inputs](inputs.md), and the files a run
writes are in [Outputs](outputs.md).

The code blocks on this page are source-level examples of the API shape. They
were not executed to produce this page, and no output, probability or timing
shown here is a measured result.

Importing `trex` performs no device or network work.

```python
from trex.target import target
from trex.scene import TargetScene
from trex.engine import run_trex_vetting
from trex.models.noise import calibrate_trex_noise
from trex.foundation.errors import TrexEvidenceUnavailableError
```

## Import paths and compatibility identities

`trex.target`, `trex.scene`, `trex.engine`, `trex.run` and `trex.example` are
compatibility identities. They replace themselves in `sys.modules` with the
corresponding `trex.app` module, so both paths reach the same module object and
the same classes and functions. `trex.doctor` is a thin module that re-exports
the supported doctor names from `trex.app.doctor`; it is not the same module
object, but the names are the same objects.

| Public path | Canonical module | What it provides |
|---|---|---|
| `trex.target` | `trex.app.target` | `target`, `STARS_COLUMNS` |
| `trex.scene` | `trex.app.scene` | `TargetScene` |
| `trex.engine` | `trex.app.engine` | `run_trex_vetting`, `TREX_PRODUCTION_N_DRAWS` and the result helpers |
| `trex.run` | `trex.app.run` | `python -m trex.run`; `load_input` is the command's internal loader |
| `trex.doctor` | `trex.app.doctor` | `python -m trex.doctor`, `inspect_inputs`, `probe_mast`, `group_failures` |
| `trex.example` | `trex.app.example` | `python -m trex.example`; `synthetic_inputs` is the example's internal scene builder |
| `trex.app.aij_loader` | same | `read_aij_observation` imports AstroImageJ differential photometry |
| `trex.app.observation_vetting` | same | `run_observation_vetting` fits scenarios jointly on native-time observations |
| `trex.app.observation_io` | same | `export_observation_input` writes a version-2 joint observation directory |
| `trex.acquisition.gaia_constraints` | same | `GaiaDetectionConstraint` for source-scoped detection records |

Prefer the short public paths above. The `trex.app` modules are implementation
owners; importing them directly is supported for the entry points listed in
this table, not as a general second API. Names starting with `_` are private in
either path and carry no compatibility promise.

## `target`

`target` binds a validated scene to observations and runs the inference.
Constructing it directly acquires TESS data for a TIC ID:

```python
target(
    ID, sectors, search_radius=10, mission="TESS",
    lightkurve_cache_dir=None, trilegal_fname=None,
    ra=None, dec=None, verify_ssl=True,
)
```

| Argument | Unit | Meaning |
|---|---|---|
| `ID` | none | Positive TIC identifier |
| `sectors` | none | Non-empty 1-D array of positive integer TESS sectors, each once |
| `search_radius` | pixels | Positive integer cutout search radius; the cutout side is `2·search_radius + 2` pixels |
| `mission` | none | Only `"TESS"` is implemented |
| `lightkurve_cache_dir` | path | Optional download directory for the cutouts |
| `trilegal_fname` | path | Optional existing background-population CSV; its SHA-256 is recorded |
| `ra`, `dec` | deg | Alternate-coordinate acquisition; not implemented |
| `verify_ssl` | none | Must remain `True` |

Direct construction queries the TIC, downloads one TESScut cutout per sector and
stores the star table and detector pixel coordinates. It requires network
access. It raises:

- `NotImplementedError` for a non-TESS mission or any non-`None` `ra`/`dec`.
- `ValueError` for a non-positive or boolean `ID`, a non-positive `search_radius`,
  an empty, non-integer or duplicate `sectors` array, disabled TLS verification,
  a duplicate catalog source ID, a catalog without exactly one owning target, a
  non-finite catalog `ra`, `dec` or `Tmag`, coordinates outside their angular
  domain, or a missing `trilegal_fname` file.
- `RuntimeError` when a requested sector's cutout or WCS cannot be used.

Public state after direct construction:

| Attribute | Type | Meaning |
|---|---|---|
| `ID`, `search_radius`, `N_pix` | int or `None` | TIC ID; cutout half-size; cutout side in pixels |
| `sectors` | `numpy.ndarray` | Acquired sectors |
| `mission` | str | `"TESS"` |
| `stars` | `pandas.DataFrame` | Acquisition table with `STARS_COLUMNS` |
| `TESS_images` | list | Mean cutout image per sector |
| `col0s`, `row0s` | list | Detector reference pixel per sector |
| `pix_coords` | list | Per sector, an `(N, 2)` detector-pixel array for the star table |
| `trilegal_fname`, `trilegal_sha256` | path, str or `None` | Background file and its digest |

### `target.from_scene`

```python
target.from_scene(scene, *, contrast_curve=None, contrast_band=None, **binding)
```

Create an offline target from a validated `TargetScene`. It downloads nothing
and synthesizes no stellar parameters. `**binding` is the `bind_scene`
contract below; `sector_weights`, `depth_ppm` and `exptime_days` are required.

`contrast_curve` accepts a two-column file, a `(separations, delta_magnitudes)`
pair, or a `(2, N)` array. `contrast_band` is then required and must be `TESS`;
supplying `contrast_band` without `contrast_curve` raises `ValueError`. A curve
replaces the scene's previous curve; without one, the scene's existing curve or
resolution limit applies.

Returns a `target` with no acquired pixel data. `search_radius` and `N_pix` are
`None`, `TESS_images`, `col0s`, `row0s` and `pix_coords` are empty, and `stars`
is assembled from the scene fields that exist. `plot_field` raises for this
path, and `calc_depths` raises because it needs pixel coordinates.

### `target.from_tic`

```python
target.from_tic(
    ID, sectors, *, output_dir, depth_ppm, cache_dir=None, policy=None,
    search_radius_arcsec=60.0, trilegal_population=None,
    contrast_curve=None, contrast_band=None, **binding,
)
```

Acquire the scene for TIC `ID` from public archives, then bind it.

| Argument | Unit | Meaning |
|---|---|---|
| `ID` | none | TIC identifier |
| `sectors` | none | Observed TESS sectors |
| `output_dir` | path | Directory for every archive response and query record; empty, or an exact repeat of the same request |
| `depth_ppm` | ppm | Measured transit depth; finite in `(0, 10⁶)` |
| `cache_dir` | path | Shared releases (GALAH DR3, MIST); defaults to `output_dir/cache` |
| `policy` | `trex.acquisition.transport.NetworkPolicy` | Transport policy; defaults to network allowed |
| `search_radius_arcsec` | arcsec | Minimum Gaia cone radius; the aperture reach can widen it |
| `trilegal_population` | path | A retained `trilegal_query.json` for this field, reused instead of a new query |
| `contrast_curve`, `contrast_band` | none | Optional measured curve, as in `from_scene` |
| `**binding` | none | Remaining `bind_scene` arguments; `depth_ppm` is already bound by the named argument |

`output_dir` must be empty or an exact repeat of the same `scene_request.json`.
A different request raises, so two scenes never mix in one directory. Every
stage whose retained record is `"available"` is reused on a rerun; target-pixel
products are revalidated on every rerun.

Returns a bound `target`. See [From a TIC ID](from-tic.md).

### `bind_scene`

```python
star.bind_scene(
    scene, *,
    sector_weights,
    depth_ppm,
    exptime_days,
    input_artifacts=None,
    bin_integration_days=0.0,
    random_seed=None,
    baseline=None,
    stellar_light=None,
    depth_sigma_ppm=None,
    duration_days=None,
    duration_sigma_days=None,
    joint_observations=None,
)
```

Attach a calibrated scene and observations. It invalidates any previous
results, verifies the scene, verifies the input artifacts, and reduces the
per-sector aperture fractions onto the observed photometry.

| Argument | Unit | Required |
|---|---|---|
| `sector_weights` | inverse variance | yes; positive weights covering exactly the scene sectors |
| `depth_ppm` | ppm | yes; `0 < depth < 10⁶` |
| `exptime_days` | days | yes; positive |
| `input_artifacts` | path → SHA-256 | no; when omitted, the scene background file is hashed |
| `bin_integration_days` | days | no; scalar `≥ 0` (`0` selects exposure-only integration), or one strictly positive width per photometry point |
| `random_seed` | none | no; non-negative integer or `None`; `None` lets the run generate one seed and record it with the result or failure |
| `baseline` | `trex.photometry.baseline.GaussianBaseline` | no; advanced, dynesty only, single sector, unbinned native times |
| `stellar_light` | `trex.stellar_light.StellarLight` | no; requires the target's own `A_V` |
| `depth_sigma_ppm`, `duration_days`, `duration_sigma_days` | ppm, days, days | no; measured values that centre proposal windows |
| `joint_observations` | `ScenarioObservations` | no; for joint data, pass `sector_weights=None`, `exptime_days=None`, zero bin width and no separate baseline; the observations own these inputs |

Returns `None`. Raises `TypeError` for a non-`TargetScene`, and `ValueError`
when the scene target/sectors or source table differ, when the weights do not
cover the scene sectors, when the depth, exposure or width is outside its
domain, when an artifact is missing or changed, or when the calibrated
aperture shares overcount the aperture.

A `target` can be rebound. Rebinding clears previous probabilities and fit
arrays. If the star table, target identity or sectors change after binding, the
next calculation refuses until the scene is bound again.

### AstroImageJ photometry

For the joint-observation Python API, use:

```python
from trex.app.aij_loader import read_aij_observation

observation = read_aij_observation(
    path, dataset_id=dataset_id, segment_id=segment_id,
    band_id=band_id, response_id=response_id, reference=reference,
    observer=observer, target=target_name, stamp=stamp,
    exposure_days=exposure_days, exposure_source=exposure_source,
    nsamples=nsamples, aperture_id=aperture_id, source_shares=source_shares,
    lineage=lineage, flux_scale=flux_scale, uncertainty_source=uncertainty_source,
    prior_mean=prior_mean, prior_covariance=prior_covariance,
    baseline_mode=baseline_mode, baseline_source=baseline_source,
)
```

Supply a tab-delimited AstroImageJ table (such as `Measurements.tbl`) or a
comma-delimited `.csv` with `BJD_TDB`, `rel_flux_T1`, `rel_flux_err_T1`,
`AIRMASS` and `FWHM` (case-insensitive). The dates must already be full
barycentric Julian dates in TDB. Pass the shared scalar TDB `astropy.time.Time`
reference, the exposure stamp (`"start"`, `"mid"` or `"end"`), and the boxcar
duration in days. Keep per-exposure durations, aperture shares and lineage in
file order; TREX sorts them with the measurements.

The loader returns an `Observation` for an `ObservationSet`. It divides flux
and errors by the supplied quiescent flux (`1` for already normalized data),
without estimating that scale, masking data or detrending. Supply a Gaussian
baseline prior in column order `(AIRMASS, FWHM)`, on the raw columns in the
export's units, with no intercept. Joint inference also needs a calibrated
scene with matching dataset/aperture/source shares and physical light models
for the exact passbands. This is not an input format for `python -m trex.run`.

See [native-time observation vetting](#native-time-observation-vetting) to run
an assembled set.

### Binding inputs versus calculation arguments

`bind_scene` owns every evidence input: the calibrated scene, per-sector
weights, measured depth, native exposure, phase-bin width, run-seed choice, the
optional measured transit values that centre proposals, and the optional
baseline or physical-light model. `calc_probs` takes the folded photometry and
measured period, plus per-call controls such as `N`, `nsamples`, `parallel`,
`verbose` and escalation state. It has no argument for a new depth, weight,
width, seed or model; rebind the scene to change one. Its `exptime` only repeats
the bound native exposure and must match it.

### `calc_probs`

```python
star.calc_probs(
    time, flux_0, flux_err_0, P_orb,
    contrast_curve_file=None, filt="TESS", N=1_000_000,
    parallel=False, drop_scenario=[], verbose=1, flatpriors=False,
    exptime=0.00139, nsamples=20, molusc_file=None,
    escalation_checkpoint_directory=None, escalation_state_in=None,
    escalation_state_out=None, escalation_deadline_monotonic=None,
)
```

Run every supported scenario on the bound scene. Returns `None` and sets the
result attributes on the object.

| Argument | Unit | Meaning |
|---|---|---|
| `time` | days from transit midpoint | 1-D, at least 2 finite points |
| `flux_0` | normalized flux | Same shape as `time` |
| `flux_err_0` | normalized flux | Scalar or one per point, finite positive |
| `P_orb` | days | Scalar only; a non-scalar period raises |
| `contrast_curve_file` | none | Not supported; raises. Pass the curve at binding |
| `filt` | none | Only `"TESS"` is supported |
| `N` | draws | Draws per scenario; production is 1,000,000 |
| `parallel` | none | Use nested-worker concurrency |
| `drop_scenario` | none | Not supported; every false-positive scenario is retained |
| `verbose` | none | `0`, `1` or `2` |
| `flatpriors` | none | Not supported; raises |
| `exptime` | days | Must equal the bound native exposure; the signature's `0.00139` default is legacy, not a measured cadence |
| `nsamples` | none | Exposure supersampling; positive integer |
| `molusc_file` | none | Not supported; raises |
| `escalation_checkpoint_directory` | path | Recovery directory; `TREX_ESCALATION_CHECKPOINT_DIRECTORY` is used when set, otherwise a unique path under the user state directory, created only if refinement needs it |
| `escalation_state_in` | mapping | Resume a split run from a saved state |
| `escalation_state_out` | dict | Return pending state instead of publishing incomplete results |
| `escalation_deadline_monotonic` | seconds | Host monotonic deadline for refinement |

`N` smaller than 1,000,000 is for software tests; an `ok` status at such a
budget is not production-budget scientific evidence. `parallel=True` passes the
hardware default worker count to the engine; the nested estimator itself is
unchanged. With a bound `GaussianBaseline`, `time` must be the baseline's exact
native chronological TDB days from conjunction and the bin width must be zero.

### Seeds and reproducibility

`bind_scene(random_seed=...)` fixes the seed used by `calc_probs`;
`run_trex_vetting` also accepts `random_seed` directly. A non-negative integer is
used as given. When it is `None`, TREX generates one seed before the Monte Carlo
stage and derives the scenario, importance and nested seeds from it; no stage,
worker or scenario draws a replacement. `calc_probs` records the resolved seed in
`calculation_provenance` on a completed run, and a pending escalation state,
failure record or checkpoint retains it otherwise. A direct `run_trex_vetting`
caller must retain the seed it passed, or the checkpoint or returned refinement
record when refinement ran, to replay or resume the run; a generated seed with no
refinement is not returned, so pass a value when the run must be replayable. A
compatible resume restores the recorded integer seed and refuses state whose
seed differs. A new call with `random_seed=None` is a new run with a new seed.
Same inputs, seed, software, estimator, device and execution path replay the
same random stream; a different seed is an independent estimate, not a
guaranteed equal answer.

After a successful call:

| Attribute | Type | Meaning |
|---|---|---|
| `star.trex_result` | `TrexResult` | Full result object |
| `star.probs` | `pandas.DataFrame` | `scenarios.csv` content |
| `star.FPP` | float | False-positive probability |
| `star.NFPP` | float | Nearby false-positive probability |
| `star.star_num`, `star.u1`, `star.u2`, `star.fluxratio_EB`, `star.fluxratio_comp` | arrays | Aligned representative quantities, one per scenario |
| `star.calculation_provenance` | dict | Provenance record |
| `star.failure_record` | dict or `None` | Set when a calculation raised |
| `star.escalation_checkpoint_directory` | path | Where refinement state is stored |

On failure the result attributes are cleared, `failure_record` is set by the
failure wrapper, and the exception propagates. When the engine built the record,
it carries a `TrexEvidenceUnavailableError`'s `diagnostics`.

`star.claim_eligible` is a class attribute and is always `False`.

### `doctor`

```python
star.doctor(time, flux, sigma, period_days, *, exptime=None, N=1_000_000, nsamples=20)
```

Run the engine's own pre-draw admission against the bound inputs and return a
JSON-safe report. It starts no sampler, takes no evidence draws and does not
change an existing result or failure record. `time`, `flux` and `sigma` use the
`calc_probs` conventions. When `exptime` is supplied it must equal the bound
exposure.

### `write_outputs`

```python
star.write_outputs(directory, *, existing_empty=False)
```

Create `directory` and write the result. After a successful `calc_probs` it
writes `scenarios.csv` and `summary.json`; a non-`ok` result raised by
`calc_probs` is retained as `failure.json` instead. With a failure record it
writes only `failure.json`, so no FPP/NFPP can be read from it. With the default
`existing_empty=False`, it raises `FileExistsError` if the directory already
exists. With `existing_empty=True`, it may create a missing directory, but an
existing one must be empty and must not be a symlink.

Returns `None`. Raises `ValueError` if no calculation was attempted and no
failure was recorded. Non-finite numbers remain visible in JSON as strings.

### Plotting

```python
star.plot_fits(time, flux_0, flux_err_0, save=False, fname=None)
star.plot_field(sector=None, ap_pixels=None, ap_color="red", save=False, fname=None)
```

`plot_fits` draws the retained representative Monte Carlo models and residuals
for each scenario. Its inputs must equal the retained likelihood observations
exactly. With `save=False` it returns an open matplotlib figure; with
`save=True` it writes a PDF and returns `None`.

`plot_field` draws the acquired TESS image for one sector and overlays the
catalog sources. It requires an acquired image; the `from_scene` path has none.
`sector=None` is accepted only for a single-sector target. With `save=True` it
writes a PDF and returns `None`.

### Star-table editing

These methods edit the acquisition star table before binding. They do not
replace a calibrated `TargetScene`; the scene still owns the inference inputs.

```python
star.add_star(ID, Tmag, bound)
star.remove_star(drop_stars)
star.update_star(ID, param, value)
star.get_spoc_apertures()
star.calc_depths(tdepth, all_ap_pixels=None)
```

| Method | Returns | Notes |
|---|---|---|
| `add_star` | `None` | Adds a source at the target's coordinates with `Tmag`; `bound` is boolean and controls whether it inherits the target's parallax |
| `remove_star` | `None` | Removes the listed source IDs; refuses to remove the target |
| `update_star` | `None` | Accepts `Tmag`, `Jmag`, `Hmag`, `Kmag`, `ra`, `dec`, `mass`, `rad`, `Teff` and `plx`; other columns are refused |
| `get_spoc_apertures` | list of `(N, 2)` arrays | One detector-pixel array per sector, using SPOC aperture bit 2 |
| `calc_depths` | `None` | Reference Gaussian dilution diagnostic in fractional depth |

`calc_depths` requires `all_ap_pixels` with one explicit `(N, 2)` detector-pixel
aperture per sector. It sets `stars["fluxratio"]`, `stars["tdepth"]`,
`excluded_impossible_hosts` and `depth_provenance`. Equal sector weighting
reproduces the reference approximation; it is not calibrated aperture evidence
and its result cannot be supplied to `calc_probs`.

## `TargetScene`

```python
TargetScene(
    tic_id, ra_deg, dec_deg, M_s_Msun, R_s_Rsun, Teff_K, Tmag, plx_mas,
    sectors, contrast_separations, contrast_values, resolved_neighbors,
    target_aperture_flux_fractions, N_background, trilegal_cache,
    background_sha256=None, background_area_sq_deg=None, feh_dex=None,
    unresolved_resolution_limit_arcsec=None, localization_evidence=None,
    neighbor_parameter_evaluations=None, metallicity_provenance=None,
    contrast_provenance=None, background_population_provenance=None,
)
```

Validates and stores the scene. Units: ICRS degrees, `M_sun`, `R_sun`, K, TESS
magnitude, mas, days/sectors as named. Field rules are in
[Inputs](inputs.md#3-the-scene).

Validation responsibilities owned by the constructor:

- Finite coordinates in `[0, 360)` and `[-90, 90]`; positive mass, radius,
  temperature and parallax.
- A positive-integer `tic_id`; a list of positive-integer `sectors`.
- Paired contrast arrays, strictly increasing positive separations, and a
  measured-curve or resolution-limit bound. A measured curve defaults the
  unresolved limit to `GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC` (2.2 arcsec).
- A TESS-band `contrast_provenance` record when one is supplied; a non-TESS band
  is refused because no band conversion is implemented.
- Unique nonempty neighbour `source_id` values; missing stellar parameters or
  `delta_mag` must carry a nonempty reason; negative parameter errors are
  refused.
- A retained regular `trilegal_cache` file whose SHA-256 matches
  `background_sha256` (computed when omitted) and a positive
  `background_area_sq_deg`. The signature default `None` for the area is not a
  usable value: omitting it raises `ValueError`.
- A target `feh_dex` requires measured-source `metallicity_provenance`;
  neighbour composition records are validated against their admitted source
  record.
- `localization_evidence` is retained as a diagnostic only;
  `neighbor_parameter_evaluations` may name only scene neighbours and must carry
  source, status and reason strings.

Useful members:

| Member | Returns | Meaning |
|---|---|---|
| `to_dict()` | dict | Explicit constructor inputs for portable transfer |
| `from_dict(payload)` | `TargetScene` | Revalidate a dict and the background file |
| `verify_background()` | `None` | Reject a changed or missing background file |
| `background_population()` | dict of arrays | Load the retained stars as arrays |
| `target_aperture_flux_fraction(sector_weights)` | float | Weighted target share |
| `neighbor_dicts_for_evidence(sector_weights=None)` | list of dicts | Per-neighbour host properties |
| `neighbor_masses_radii()` | `(masses, radii, delta_mags)` | Arrays, `NaN` where unmeasured |
| `companion_delta_mags()` | array | Δmag for all neighbours, `NaN` where unmeasured |
| `n_neighbors` | int | Number of resolved neighbours |
| `has_contrast_data` | bool | `True` exactly when a measured contrast curve is stored |
| `unresolved_resolution_limit_arcsec` | float | Independent unresolved-source bound, including outside a measured curve's range |
| `contrast_separations`, `contrast_values` | array or `None` | Stored measured curve |

`has_contrast_data` is `True` only when a measured curve is stored. A scene
without one still carries `unresolved_resolution_limit_arcsec`, which bounds
unresolved companions; `has_contrast_data` is then `False`.

`to_dict` records file paths and no file contents. Transferring a scene to
another host requires moving the background file and preserving its digest.
`from_dict` is `TargetScene(**payload)`, so it runs the same validation and
re-verifies the background.

## `run_trex_vetting`

```python
run_trex_vetting(
    scene, time, flux, sigma, period_days, depth_ppm, exptime_days,
    bin_integration_days=0.0, sector_weights=None, u1=None, u2=None,
    n_draws=1_000_000, random_seed=None, progress_callback=None,
    nsamples=20, ldc_provenance=None, depth_sigma_ppm=None,
    duration_days=None, duration_sigma_days=None, n_jobs=None,
    escalation_state_in=None, escalation_state_out=None,
    escalation_checkpoint_directory=None, escalation_wall_time_seconds=None,
    escalation_deadline_monotonic=None, resume_initial_monte_carlo=False,
    baseline=None, stellar_light=None, joint_observations=None,
    release_constraints=None,
)
```

The engine entry point behind `calc_probs`. It consumes a `TargetScene` and the
observations and returns a `TrexResult`. `calc_probs` is the higher-level
wrapper that adds the compatibility attributes.

| Argument | Unit | Meaning |
|---|---|---|
| `scene` | none | Validated `TargetScene`; `scene.feh_dex` is required |
| `time` | days | Phase-relative days from transit midpoint, finite 1-D |
| `flux` | normalized flux | Same shape as `time` |
| `sigma` | normalized flux | Scalar shared by every point or one positive value per point |
| `period_days` | days | Positive scalar |
| `depth_ppm` | ppm | Positive scalar |
| `exptime_days` | days | Positive scalar |
| `bin_integration_days` | days | Scalar `≥ 0` (`0` selects exposure-only integration), or one strictly positive width per photometry point |
| `sector_weights` | inverse variance | Required: positive weights covering the observed sectors |
| `u1`, `u2` | none | Explicit target coefficients; both or neither. Unset uses the target atmosphere lookup |
| `n_draws` | draws | Draws per scenario; production is 1,000,000 |
| `random_seed` | none | Non-negative run seed; `None` generates one before Monte Carlo (retained by the escalation checkpoint when refinement runs) |
| `progress_callback` | callable | Called as `callback(step, done=..., total=...)` |
| `nsamples` | none | Exposure supersampling; positive integer |
| `ldc_provenance` | list or `None` | Optional list that receives limb-darkening provenance events |
| `depth_sigma_ppm`, `duration_days`, `duration_sigma_days` | ppm, days, days | Optional measured values used to centre proposal windows |
| `n_jobs` | count | Worker processes used only when nested escalation is needed |
| `escalation_state_in`, `escalation_state_out` | mapping, dict | Mutually exclusive resume and handoff controls |
| `escalation_checkpoint_directory` | path | Required recovery directory (see below) |
| `escalation_wall_time_seconds` | seconds | Operational wall budget; at most 540 minutes; a larger value is clamped |
| `escalation_deadline_monotonic` | seconds | Earlier host monotonic deadline |
| `resume_initial_monte_carlo` | boolean | Reuse completed per-scenario Monte Carlo work with the original explicit seed and budget; incompatible with `escalation_state_in` |
| `baseline` | `GaussianBaseline` | Explicit joint baseline; dynesty only, one sector, exact unbinned native times |
| `stellar_light` | `StellarLight` | Explicit physical-light model; requires the target's own `A_V` |
| `joint_observations` | `ScenarioObservations` | Joint aperture likelihood and exact band assets; do not also pass sector weights, a legacy exposure, bin width or baseline |
| `release_constraints` | list or tuple of `GaiaDetectionConstraint`, or `None` | Optional evidence-neutral records; see below |

`run_trex_vetting` requires `escalation_checkpoint_directory` or the
`TREX_ESCALATION_CHECKPOINT_DIRECTORY` environment variable before Monte
Carlo. An escalation that may run for hours must be resumable, and this is a
hard refusal, not a fallback. `calc_probs` selects a unique path under the
user state directory when the caller names none; the directory is created only
if refinement needs it.

### Gaia detection records

Pass `release_constraints` to `run_trex_vetting`, not to `target.calc_probs`
or the prepared-run CLI. Each `GaiaDetectionConstraint(target_host_scope,
detected, p_detect=None, period_match=None, mass_ratio_limit=None)` must name
the exact source ID of the target or a resolved neighbour in the scene.
`detected` and `period_match` are booleans or `None`; supplied detection
probabilities and mass-ratio limits must be in `[0, 1]`.

TREX 1.0 does not use these records to change scenario weights. Missing
information, non-detections and detections without a confirmed period match
are neutral. A record with both `detected=True` and `period_match=True` is
refused with `TrexEvidenceUnavailableError` before sampling: this release has
no approved Gaia detection likelihood. Do not treat catalog absence or RUWE
as a non-detection constraint.

### `prepare_vetting_inputs` (advanced, not a stability promise)

`prepare_vetting_inputs` is an engine admission helper used by `calc_probs` and
`python -m trex.doctor`. It is importable from `trex.engine`, but it is not part
of the small stable public surface and its return tuple may change. Most callers
should use `run_trex_vetting` or `target.doctor`.

```python
prepare_vetting_inputs(
    scene, time, flux, sigma, period_days, depth_ppm, exptime_days,
    bin_integration_days=0.0, sector_weights=None, u1=None, u2=None,
    n_draws=1_000_000, nsamples=20, ldc_provenance=None,
    escalation_state_in=None, escalation_state_out=None,
    baseline=None, stellar_light=None, joint_observations=None,
)
```

The engine's canonical pre-draw admission, shared by `target.calc_probs` and
`python -m trex.doctor`. It starts no RNG or sampler and performs no network
request. It returns a nine-item tuple:

```text
(time, flux, sigma, bin_integration_days, u1, u2,
 nearby_stars, target_aperture_flux_fraction, population)
```

It rejects a non-`TargetScene` (`TypeError`), a non-positive `n_draws` or
`nsamples`, a non-positive or non-finite period/depth/exposure, malformed
photometry, empty `sector_weights`, a missing `scene.feh_dex`, a baseline that
is not dynesty, multi-sector or on exact unbinned native times, and
`escalation_state_in` together with `escalation_state_out`. It raises
`TrexEvidenceUnavailableError` when a baseline/physical-light configuration is
unsupported or a capable host or background composition cannot be represented.

## Native-time observation vetting

```python
from trex.app.observation_vetting import run_observation_vetting

result = run_observation_vetting(
    scene, observations, period_days=period_days, depth_ppm=depth_ppm,
    epoch_days=epoch_days, band_light=band_light, random_seed=1729,
)
```

The keyword-only inputs are `period_days`, `depth_ppm`, `epoch_days`,
`band_light`, and optional `shared_baseline=None`, followed by keyword
engine controls such as `n_draws`, `random_seed`, and `n_jobs`. The function
returns a `TrexResult`. Pass an `ObservationSet`, with epoch in days relative
to its shared TDB reference and `band_light` mapping each exact band ID to
its `StellarLight` model. `shared_baseline`, if supplied, is a
`GaussianBaseline` prior across the streams. The first stream supplies
proposal coordinates; the likelihood uses every stream on its own times,
with its calibrated shares and integration. Use dynesty for joint scenes
with population-hosted or diluted scenarios; unsupported native-only
combinations are refused before draws. To run the same input format as a
command, [export a version-2 directory](prepared-runs.md#version-2-joint-observations).

## `TrexResult`

A dataclass with these fields:

| Field | Type | Meaning |
|---|---|---|
| `fpp`, `nfpp` | float or `None` | False-positive probability and nearby false-positive probability; `None` when normalization did not produce them |
| `probs` | `numpy.ndarray` or `None` | Normalized scenario probabilities |
| `status` | str | Normalization status; default `"not-run"` |
| `effective_sample_sizes` | dict | Final Kish ESS by scenario name |
| `supplemental_nested_evidence` | dict | Substituted estimator records |
| `nested_evidence_escalation` | dict or `None` | Refinement routing record |
| `jax_batch_fallbacks` | dict | Per-scenario accelerator fallbacks |
| `input_identity` | str or `None` | Digest of evaluated inputs |
| `period_prior_period_days` | float or `None` | Period at which the scenario prior was evaluated |
| `period_log_priors` | dict | Per-scenario log period factor |
| `scenario_fits` | list | Per-scenario representative draws and availability |
| `fit_evaluation` | dict | Exposure, bin width, time domain and representative-draw origin used for replay |
| `phase_wall_seconds` | dict | Wall time per phase; `None` for a phase that did not run |
| `diagnostics` | list of `TrexDiagnostic` | Named checks |
| `claim_eligible` | bool | Always `False` |
| `claim_block_reason` | str | Why the result is not a claim |
| `degenerate` | bool | Numerical degeneracy flag |

Methods:

| Method | Returns | Requirement |
|---|---|---|
| `top_scenarios(n=5)` | list of `(name, probability)`, descending | Returns `[]` when `probs` is `None` or empty |
| `to_probability_dataframe()` | `pandas.DataFrame` with `PUBLIC_PROBABILITY_COLUMNS` | Complete, non-degenerate result with retained fits |
| `aligned_fit_arrays()` | dict of `star_num`, `u1`, `u2`, `fluxratio_EB`, `fluxratio_comp` | Same |
| `representative_models(time)` | list of model arrays or `None` per scenario | Same |

The last three raise `ValueError` when the result is incomplete, degenerate, or
was not produced with retained representative fits. `representative_models`
returns `None` for a scenario without a retained draw; a nested evidence
substitution is not a fitted profile.

`TrexDiagnostic` carries `name`, `status`, `message`, `value=None` and
`threshold=None`.

## Low-level probability functions

```python
from trex.evidence.marginal_likelihoods import compute_fpp_nfpp

fpp, nfpp, probs, status = compute_fpp_nfpp(
    lnZ, effective_sample_sizes=None, *, period_days=None
)
```

| Argument | Meaning |
|---|---|
| `lnZ` | 1-D array of prior-free log evidences in scenario-index order, with at least 15 entries |
| `effective_sample_sizes` | Optional mapping of scenario index to ESS; supplying it enables degeneracy routing |
| `period_days` | Keyword-only optional observed period [days]; applies the Giacalone et al. period prior once before normalization |

Returns dimensionless `FPP`, `NFPP`, the normalized probability vector and the
status string. `FPP` is the summed mass of every retained scenario that is not
planet-hosted on the target (`TP`, `PTP`, `DTP`); `NFPP` is the summed mass of
the neighbour slots and is `0.0` when the vector has only target slots.
Statuses are:

| Status | Meaning |
|---|---|
| `ok` | Normal log-sum-exp normalization |
| `all_neginf` | Every scenario has zero sampled evidence; `FPP = 1.0` and `NFPP = nan`. If the optional ESS map reports a sampled `-inf`, the status is `degenerate_evidence` instead |
| `anomaly` | A NaN or `+inf` is present; `FPP` and `NFPP` are `nan` |
| `degenerate_evidence` | A sampled sub-floor scenario retains more than the registered routing allowance, or zero sampled mass could flip the decision; `FPP` and `NFPP` are `nan` |

Raises `ValueError` when `lnZ` is not a non-empty 1-D real array or has fewer
than 15 entries. This function performs no likelihood evaluation; it cannot
produce a result whose scenarios were not already computed.

## Noise calibration

```python
from trex.models.noise import calibrate_trex_noise

record = calibrate_trex_noise(
    time_btjd, flux, sigma, sectors,
    *,
    period_days=...,
    epoch_btjd=...,
    duration_days=...,
    cadence_days_by_sector={1: 120.0 / 86400.0},
    event_windows=None,
)
```

All four ephemeris/cadence keywords are required. `time_btjd` is native
`BTJD_TDB` days; `flux` and reported `sigma` are normalized relative flux; the
period, epoch and duration are days. `sectors` is one positive integer per
cadence. `cadence_days_by_sector` maps each retained sector to the cadence read
from its FITS header; there is no guessed sampling interval.

Optional `event_windows` is a sequence of mappings with `phase` in [0, 1),
`duration_days` (default T14) and a non-empty `role`. Declared windows replace
the default circular phase-0.5 secondary; the primary is always excluded.
Invalid windows raise `ValueError`. This is calibration-only: all supplied
points remain in the likelihood.

The returned dict contains `status`, `method`,
`calibration_time_domain`, `cadence_source`, `period_days`, `epoch_btjd`,
`duration_days`, `selection`, `event_windows`, `secondary_mask`,
`event_phase_support`, `sector_combination`, `sectors`, `reference`,
`inflation_reference`, `likelihood_scope`, `limitations`, `claim_eligible`
(always `False`) and `native_uncertainty_factor` (the largest per-sector
effective factor).

`secondary_mask` is `"circular-half-phase-window"` or
`"caller-declared-event-windows"`; `event_phase_support` distinguishes an assumed
circular secondary from caller-declared eclipse geometry.

Each `sectors[str(sector)]` record contains `status`, `cadence_days`,
`bin_size_points`, `out_of_transit_sample_count`, `gap_tolerance_cadences` and,
for a passed sector, `beta_raw`, `beta`, `bin_size_days`, `bin_count`,
`sample_count`, `dropped_sample_count`, `scatter_units`,
`standardized_unbinned_scatter`, `standardized_binned_scatter`,
`white_scatter_factor` and `effective_error_factor`. The two scatter values are
dimensionless RMS values of standardized residuals.

Raises `ValueError` for misaligned, non-finite or non-real photometry, a
non-integer/`≤ 0` sector array, or a duration outside
`0 < duration_days < period_days / 2`; a missing required keyword raises
`TypeError`. When a retained sector cannot support the estimate, raises
`TrexEvidenceUnavailableError` whose `diagnostics["noise_model"]` carries the
partial record with the deferred sector and its reason.

## Constants

| Constant | Module | Value |
|---|---|---|
| `TREX_PRODUCTION_N_DRAWS` | `trex.app.engine` (`trex.engine`) | `1_000_000` |
| `MINIMUM_EVIDENCE_EFFECTIVE_SAMPLE_SIZE` | `trex.evidence.marginal_likelihoods` | `100.0` |
| `MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY` | `trex.evidence.marginal_likelihoods` | `1e-3` |
| `FPP_THRESHOLD` | `trex.evidence.diagnostics` | `0.01` |
| `NFPP_THRESHOLD` | `trex.evidence.diagnostics` | `0.001` |
| `N_TARGET_SCENARIOS` | `trex.evidence.marginal_likelihoods` | `15` |
| `N_NEIGHBOR_SCENARIOS` | `trex.evidence.marginal_likelihoods` | `3` |
| `TARGET_SCENARIO_NAMES` | `trex.evidence.marginal_likelihoods` | `("TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P", "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P")` |
| `NEIGHBOR_SCENARIO_NAMES` | `trex.evidence.marginal_likelihoods` | `("NTP", "NEB", "NEBx2P")` |
| `GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC` | `trex.foundation.constants` | `2.2` |

`FPP_THRESHOLD` and `NFPP_THRESHOLD` are routing thresholds for reports, not
validation or confirmation criteria. `MAXIMUM_UNCONVERGED_SCENARIO_PROBABILITY`
is one tenth of `FPP_THRESHOLD` by registered convention, not an FPP error
bound.

## Exceptions

| Exception | Module | Parent | Meaning |
|---|---|---|---|
| `TrexEvidenceUnavailableError` | `trex.foundation.errors` | `ValueError` | Required evidence is absent or did not converge; `diagnostics` carries the structured reason |
| `NumericalLikelihoodError` | `trex.foundation.errors` | `RuntimeError` | Supported likelihood evaluations failed numerically; `count` carries how many |
| `DrawUnsupportedError` | `trex.models.likelihoods` | `NumericalLikelihoodError` | A draw's orbit or model flux could not be evaluated numerically |
| `NestedCheckpointPaused` | `trex.evidence.nested_checkpoint` | `TrexEvidenceUnavailableError` | Refinement paused at its deadline; state is retained for a rerun |
| `CheckpointInputMismatch` | `trex.evidence.nested_checkpoint` | `ValueError` | Retained state belongs to different integration inputs |
| `NestedEvidenceUnavailableError` | `trex.evidence.evidence_escalation` | `RuntimeError` | The nested estimator could not be used for a scenario |
| `AcquisitionError` | `trex.acquisition.transport` | `RuntimeError` | A scene input could not be acquired without inventing data |
| `TrilegalUnavailableError` | `trex.acquisition.trilegal_errors` | `AcquisitionError` | The background population provider failed or is unavailable |
| `ApertureSceneError` | `trex.acquisition.aperture` | `RuntimeError` | The retained aperture evidence is unusable |
| `UnsupportedAtmosphereError` | `trex.stellar_light.interpolation` | `ValueError` | The requested atmosphere is unavailable; this is not zero probability |

`TrexEvidenceUnavailableError.diagnostics` is a deep-copied mapping with
string keys. `str(error)` is only the message; a consumer that records the
failure must read `diagnostics` explicitly. `NumericalLikelihoodError.count` is
a positive integer; booleans, fractions and text are refused.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TREX_NESTED_ESTIMATOR` | `dynesty` | Selects `dynesty` or the experimental `blackjax` nested estimator. Any other value raises `ValueError` before dispatch or completed-result reuse |
| `TREX_BLACKJAX_PRECOND` | `eye` | Experimental BlackJAX slice directions: `eye` (fixed isotropic), `pilot` (one fixed pilot metric) or `split` (per-batch split-ensemble); any other value raises |
| `TREX_BLACKJAX_BATCH_INIT` | `0` | `1` evaluates prior points in sequential-equivalent padded blocks; `0` keeps the sequential loop |
| `TREX_BLACKJAX_W_MARCH` | `0` | `4` or `8` enable batched speculative stepping-out probes; `0` disables them |
| `TREX_IMPORTANCE_PROPOSAL` | `kernel-mixture` | Defensive-importance family for the nested-ensemble refinement: `kernel-mixture` or the experimental `probit-gaussian`; any other value raises |
| `TREX_ESCALATION_CHECKPOINT_DIRECTORY` | none | Default recovery directory when none is passed to `calc_probs`/`run_trex_vetting` |
| `TREX_ESCALATION_WALL_SECONDS` | `32400` (540 minutes) | Operational refinement wall budget in seconds, parsed as a positive finite number; the engine never uses more than 540 minutes |
| `TREX_ESCALATION_DEADLINE_MONOTONIC` | none | Host monotonic deadline read by `python -m trex.run`; must be finite when set |

Unset or default values select the assessed route: `dynesty`, fixed-isotropic
BlackJAX slice directions, sequential (unbatched) initialization, zero
speculative expansions and the `kernel-mixture` proposal for nested-ensemble
importance refinement. Every other accepted value of these inference controls is
an experimental opt-in. A non-default BlackJAX policy enters the native
checkpoint and completed-result identity; a non-default
`TREX_IMPORTANCE_PROPOSAL` enters the nested-importance and completed-result
identity and the CLI handoff. No opt-in inherits the default configuration's
calibrated marker (`configuration_calibrated` / `calibrated_configuration`);
padded or speculative physical work is outside the logical-call count and no
speedup is claimed. Default-route evidence does not establish the calibration,
acceleration or scientific accuracy of an opt-in route.

Runtime overrides read by `trex.runtime.hardware`:

| Variable | Default | Effect |
|---|---|---|
| `TREX_N_JOBS` | affinity-visible CPUs minus 2, floored at 1 | Worker-process count; must be a positive integer |
| `TREX_GPU_MIN_MEMORY_BYTES` | `0` (no floor) | Minimum addressable device memory before a GPU is used |
| `TREX_ACCELERATOR_LEASE_PATH` | host-keyed file in the system temporary directory | Location of the accelerator lease |
| `TREX_ACCELERATOR_MAX_DEVICES` | all visible devices in the scope | Maximum devices one computation may lease; positive integer |

Every `TREX_*` variable in these tables still accepts its retired `EXONYM_*`
spelling for one release and reports a `FutureWarning` once per process; the
`TREX_*` name wins when both are set.

## Command-line entry points

### `python -m trex.run`

```text
python -m trex.run INPUT_DIRECTORY OUTPUT_DIRECTORY
    [--draws N] [--nsamples N] [--escalation-checkpoint-directory PATH]
```

| Option | Default | Meaning |
|---|---|---|
| `--draws` | `1000000` | Draws per scenario; below 1,000,000 is for software tests |
| `--nsamples` | `20` | Exposure supersampling |
| `--escalation-checkpoint-directory` | `OUTPUT_DIRECTORY.escalation` | Recovery directory; must be outside the output directory |

Exit status:

| Code | Meaning |
|---|---|
| `0` | Calculation completed with a reported FPP and NFPP |
| `1` | Execution failed, including failure-record publication |
| `2` | Input, checkpoint or output refused before a result existed |
| `3` | Calculation ran but required evidence stayed unresolved; no FPP/NFPP is reported as success |
| `75` | Escalation paused at a retained checkpoint; the command created no output; rerun the same command |

See [Prepared runs](prepared-runs.md) for both input formats and resume rules.

### `python -m trex.doctor`

```text
python -m trex.doctor [INPUT_DIRECTORY]
    [--mast-tic TIC] [--timeout SECONDS] [--failure PATH ...]
```

At least one of `INPUT_DIRECTORY`, `--mast-tic` or `--failure` is required.
`INPUT_DIRECTORY` runs the engine's pre-draw admission through `target.doctor`.
`--mast-tic` performs one bounded MAST metadata query on the current host;
`--timeout` is a positive wall-clock limit in seconds, at most 60, default
`15.0`. `--failure` is repeatable and groups retained `failure.json` records by
their recorded signature. The command prints a JSON object and exits `0` only
when every input/MAST report is `ready` and no failure group was found;
otherwise it exits `1`.

### `python -m trex.example`

```text
python -m trex.example NEW_DIRECTORY
```

Runs the installed public path on invented inputs with 4,096 draws per scenario
and 4 exposure samples, and writes the same output files a real run writes.
`NEW_DIRECTORY` must not exist. The probabilities mean nothing; this is a
software demonstration, not a scientific result. It exits `0` when the result
status is `ok`, `3` when the demonstration ran but its status is not `ok`, and
`1` on an execution failure.

All three commands also accept the logging wrapper's `--log-dir DIR` or
`--no-log-file` option; these control per-process logs, not the calculation.

## Next

- [Module and function reference](modules.md): the responsibilities behind these
  calls, grouped by stage.
- [Prepared runs](prepared-runs.md): the command-line interface and its input
  files.
- [Inputs](inputs.md) and [Outputs](outputs.md): the formats behind these calls.
- [From a scene](from-scene.md) and [From a TIC ID](from-tic.md): the two scene
  paths.
- [Scenarios](scenarios.md): the scenario names and what each one models.
