# From a scene

This page answers the question "how do I run TREX on a scene I prepared
myself?" It covers the offline path, the scene contract, the contrast curve,
the provenance rules that make a scene valid, and what this path cannot do.
Field-level definitions are in [Inputs](inputs.md); the file-based runner for
the same scene is in [Prepared runs](prepared-runs.md).

Use this path when:

- you already have a calibrated aperture model, for example from a point-response
  function fit or a dedicated photometry pipeline,
- the target is not in the TIC, or the acquisition metadata is incomplete,
- you need to run the same scene repeatedly without network access,
- you are testing a specific change to the stellar or neighbour parameters.

Do not use it to avoid supplying a measurement. A scene with guessed values
still runs, but the probabilities describe the guesses, not the star.

## The call

```python
from trex.scene import TargetScene
from trex.target import target

scene = TargetScene(...)  # fields are listed below

star = target.from_scene(
    scene,
    sector_weights={1: 3.0, 2: 1.0},
    depth_ppm=8000.0,
    exptime_days=120.0 / 86400.0,
    bin_integration_days=0.0,
    random_seed=424242,
)
```

`from_scene` does not query a catalogue, download an image, fit an aperture
mask or synthesize a stellar parameter. It reads the scene's background file
to verify it, verifies any files named in `input_artifacts`, validates the
scene, optionally replaces its contrast curve, and binds the observations. It
returns a `target` ready for `calc_probs`.

The two signatures are

```python
target.from_scene(scene, *, contrast_curve=None, contrast_band=None, **binding)
star.bind_scene(scene, *, sector_weights, depth_ppm, exptime_days,
                input_artifacts=None, bin_integration_days=0.0,
                random_seed=None, baseline=None, stellar_light=None,
                depth_sigma_ppm=None, duration_days=None,
                duration_sigma_days=None)
```

`scene` must be a validated `TargetScene`; another object raises
`TypeError("scene must be a validated TargetScene")`. `from_scene` requires the
three binding values `sector_weights`, `depth_ppm` and `exptime_days`; the
remaining arguments are optional. `bind_scene` returns `None` and invalidates
any earlier result on that target.

`bind_scene` owns every evidence input: the scene, sector weights, depth, native
exposure, bin width, run-seed choice and optional models. `calc_probs` receives
only the folded photometry, the measured period and per-call controls; it cannot
replace a bound value, so changing one requires rebinding the scene.

| Argument | Unit | Required | Rule |
|---|---|---|---|
| `sector_weights` | inverse-variance weight | yes | finite and positive; keys exactly the scene's sectors; integer sector IDs |
| `depth_ppm` | ppm | yes | finite, `0 < depth < 10⁶`; the value is ppm, not a fraction (`0.01` for 1% must be supplied as `10000.0`) |
| `exptime_days` | days | yes | finite and positive |
| `input_artifacts` | path → SHA-256 | no | non-empty mapping; each file must be a regular, non-symlink file whose digest matches; when omitted, TREX hashes the background population |
| `bin_integration_days` | days | no | scalar `>= 0` (`0` selects exposure-only integration), or one strictly positive width in days per photometry point |
| `random_seed` | none | no | non-negative integer or `None`; `None` lets the run generate one seed and record it with the result or failure (see [Python API](api.md#seeds-and-reproducibility)) |
| `depth_sigma_ppm` | ppm | no | positive; centres the depth proposal; not likelihood data |
| `duration_days`, `duration_sigma_days` | days | no | positive; centre the duration proposal; not likelihood data |
| `baseline` | `GaussianBaseline` | no | advanced; dynesty only, single-sector scene, exact unbinned native-time observations |
| `stellar_light` | `StellarLight` | no | advanced; requires the target's own `A_V` and, for each neighbour that can host the observed depth, its own `stellar_atmosphere` |

`bind_scene` requires the scene's TIC ID and sector list to match the target,
every scene source to appear exactly once in the target star table, the sector
weights to cover exactly the scene sectors, and, when the target already
retains a background digest, that digest to match the scene's background. It
rebuilds the scene from `scene.to_dict()` and re-verifies the background before
reducing the aperture fractions. A scene whose target, sectors or source table
does not match is refused. Relative `input_artifacts` paths are resolved against
the process working directory and recorded as absolute paths.

The three required values are measurements from the caller's light curve.
`depth_ppm` is the measured transit depth, not a fitted model parameter;
`sector_weights` are the sums of inverse variances of the retained native
cadences; `exptime_days` is the native exposure. TREX does not estimate red
noise from folded times and does not derive a depth from the scene.

After `calc_probs`, `star.write_outputs(directory)` writes `summary.json` and
`scenarios.csv`, or `failure.json` when the calculation recorded a failure. It
refuses a symlink and any existing output unless `existing_empty=True` is used
for an empty directory; that flag is for a resumable output the command
manages. See [Outputs](outputs.md).

A completed call has `trex_result.status == "ok"`. A convergence or
missing-evidence refusal raises and leaves a failure record: that is a deferral,
not a zero probability. A refinement paused at its deadline writes no output
file from the command and is resumed from its checkpoint, not restarted with a
new seed.

## A valid scene

`TargetScene` is fail-closed: missing or non-finite fields, duplicate neighbour
IDs, a non-monotonic contrast curve, a missing resolution bound or a background
hash mismatch raise instead of supplying a fallback. A scene is conditional
scenario input, not a source attribution or a claim.

### Target and observation fields

| Field | Type and unit | Required | Rule |
|---|---|---|---|
| `tic_id` | integer | yes | positive; booleans are refused |
| `ra_deg` | float, degrees | yes | finite, `[0, 360)` |
| `dec_deg` | float, degrees | yes | finite, `[-90, 90]` |
| `M_s_Msun` | float, solar masses | yes | finite and positive |
| `R_s_Rsun` | float, solar radii | yes | finite and positive |
| `Teff_K` | float, kelvin | yes | finite and positive |
| `Tmag` | float, TESS magnitude | yes | finite |
| `plx_mas` | float, milliarcseconds | yes | finite and positive |
| `feh_dex` | float, dex | no | finite; `None` is allowed by the constructor but every calculation refuses; when supplied, `metallicity_provenance` is required and its `value_dex` must equal this value |
| `metallicity_provenance` | mapping | with `feh_dex` | a caller-supplied record or an admitted catalogue record; see below |
| `sectors` | list of integers | yes | positive integers; a JSON list, not a tuple or array; duplicates are refused when binding |

### Aperture fractions and unresolved companions

| Field | Type and unit | Required | Rule |
|---|---|---|---|
| `target_aperture_flux_fractions` | list of records | yes | non-empty; each record is `{"sector": integer > 0, "fraction": float in [0, 1]}`; sectors unique; the weighted target fraction must be positive |
| `contrast_separations` | 1-D float, arcseconds | with the curve | at least two points; finite; positive; strictly increasing |
| `contrast_values` | 1-D float, magnitudes | with the curve | same shape as the separations; finite limiting Δmag; no monotonicity or sorting is imposed |
| `contrast_provenance` | mapping | no | requires a measured curve; `kind` must be `measured-contrast-curve`; `source`, `band` and `limitation` must be non-empty strings; `band` must be TESS; extra keys are retained |
| `unresolved_resolution_limit_arcsec` | float, arcseconds | if no curve | positive; with a curve and omitted, it defaults to the Gaia DR2 close-pair limit of 2.2 arcsec |

A curve and `unresolved_resolution_limit_arcsec` are independent bounds. The
curve applies where it has support; the resolution limit remains the
unresolved-scene bound outside the curve's measured range. The scene does not
cross-check the limit against the curve's coverage.

### Resolved neighbours

`resolved_neighbors` is a list of records; it may be empty. Each record has:

| Field | Type and unit | Required | Rule |
|---|---|---|---|
| `source_id` | string | yes | non-empty and unique across the scene |
| `M_s`, `R_s` | float, solar masses and radii | pair | both finite and positive, or both `None` with a reason |
| `stellar_parameter_unavailable_reason` | string | if `M_s`/`R_s` are `None` | non-empty; must be absent when both values are supplied |
| `delta_mag` | float, magnitudes | value or reason | finite Δmag relative to the target, or `None` with a reason |
| `delta_mag_unavailable_reason` | string | if `delta_mag` is `None` | non-empty; must be absent when `delta_mag` is supplied |
| `separation_arcsec` | float, arcseconds | yes | finite and positive |
| `aperture_flux_fractions` | list of records | yes | non-empty; same record rules as the target; a row is required for every sector in `sector_weights` (binding requires that set to equal `scene.sectors`); extra rows are stored but unused |
| `aperture_flux_fractions_upper_95` | list of records | no | same record rules; requires `aperture_flux_fraction_limitation`; used as this source's marginal 95% fraction for its host scenario |
| `aperture_flux_fraction_limitation` | string | with the upper limit | non-empty; must be absent when no upper limit is supplied |
| `mass_solar_err_lower`, `mass_solar_err_upper`, `radius_solar_err_lower`, `radius_solar_err_upper` | float, solar masses and radii | no | finite and non-negative; recorded but not marginalized |
| `stellar_parameter_provenance` | mapping | no | must have a non-empty `source` string; other keys are retained |
| `stellar_relations_calibration_status` | string | no | only `"uncalibrated"` is accepted; it is a disclosure, not a calibration |
| `metallicity_provenance` | mapping | no | validated by the same composition contract as the target; for a catalogue source the `source_id` must match the neighbour; cannot coexist with `metallicity_unavailable_reason` |
| `metallicity_unavailable_reason` | string | no | non-empty; cannot coexist with `metallicity_provenance` |
| `stellar_atmosphere` | mapping | no | `source` non-empty; `teff_k` positive; `mh_dex` finite; `av_mag` finite and non-negative; extra keys are dropped |
| `stellar_parameter_distribution` | mapping | no | an advanced characterized-host distribution validated by `trex.evidence.host_measure.HostMeasure` against the target metallicity |

The joint scene fraction is always `aperture_flux_fractions`. A residual-scene
source may add `aperture_flux_fractions_upper_95` and its limitation; the
evidence calculation then uses the source-specific marginal limit for that
source's host scenario, while the coherent joint value remains the scene
fraction. The overcount check uses the joint values, not the marginal limits.

A neighbour without stellar parameters is retained. If it could host the
observed depth, the calculation fails closed with
`TrexEvidenceUnavailableError`; otherwise its three slots are `host-excluded`
with `-inf` evidence and zero probability, and the run can still publish.
Dropping the neighbour would bias FPP low. In the default physics
path, a neighbour without `metallicity_provenance` uses the target metallicity
and the evidence record states that fallback; with `stellar_light`, its
`stellar_atmosphere.mh_dex` supplies the composition instead. A non-empty
`metallicity_unavailable_reason` states why no independent value exists.

### Background and diagnostics

| Field | Type and unit | Required | Rule |
|---|---|---|---|
| `N_background` | integer | yes | non-negative; a recorded count, not compared with the file or re-counted |
| `trilegal_cache` | path | yes | an existing regular, non-symlink file; a relative path is resolved against the process working directory |
| `background_sha256` | 64-character hexadecimal digest string | no | if supplied, it must equal the file's SHA-256; when omitted, the digest is computed and recorded |
| `background_area_sq_deg` | float, square degrees | yes | finite and positive; there is no default |
| `background_population_provenance` | mapping | no | only the TRILEGAL LSST-simulation fallback projection; its `population_sha256` must equal `background_sha256`; when omitted, a `trilegal_query.json` beside the file may supply it |
| `localization_evidence` | mapping | no | copied as a retained centroid diagnostic; current evidence weights are neutral and the record does not reweight neighbours |
| `neighbor_parameter_evaluations` | mapping | no | keys must be resolved-neighbour `source_id` values; each value is a list of records with non-empty `source`, `status` and `reason`; appended to an unavailable-parameter reason |

The background file is a TRILEGAL-format CSV with the columns `TESS`, `Mact`,
`logg`, `logTe` and `[M/H]`; `Av` and `m-M0` are optional. It must contain at
least one row with `TESS >= Tmag`. See
[Background population](background-population.md) for the parser rules.

### Provenance records

For an external scene, the normal target record is

```python
metallicity_provenance={
    "source": "caller-supplied",
    "quantity": "[Fe/H]",          # or "[M/H]"
    "value_dex": 0.02,
    "uncertainty_dex": 0.08,
    "citation": "Smith et al. (2020), Table 3",
}
```

The caller-supplied record accepts exactly the keys `source`, `quantity`,
`value_dex`, `uncertainty_dex`, plus `citation` and/or `uri`. At least one of
`citation` and `uri` must be a non-empty string. `value_dex` must equal
`feh_dex` exactly and `uncertainty_dex` must be positive. A missing value cannot
be replaced by solar; the calculation refuses with
`scene.feh_dex is required to evaluate every false-positive scenario`.

The other admitted source names are `apogee-dr17`, `galah-dr3`,
`gaia-gspspec-calibrated` and `shboost-xp`. Each requires its catalogue flags
and retrieval record; all except a legacy `shboost-xp` record also require a
raw-row binding. The source-specific rules are in
[Inputs](inputs.md#3-the-scene). A record that names a source but omits its
required flags or binding is refused.

`contrast_provenance` is optional in the scene validator but required in
practice to state which band the stored curve was measured in. If you pass
`contrast_curve` to `from_scene`, TREX builds the record for you; if the curve
is already in the scene, supply the record so the band and source travel with
the result. The validator cannot check the band of an unlabelled stored curve.

`background_population_provenance` is needed only when the population came from
the LSST-simulation fallback. It records the fallback model and the population
digest and states the measured same-cone count deficit; it does not assert a
bias direction. A TRILEGAL web population carries no such record.

`to_dict()` carries all of these records. `from_dict()` revalidates them; it
does not re-derive a missing provenance record from the values.

### Cross-field rules

- `contrast_separations` and `contrast_values` are supplied together. If both
  are `None`, `unresolved_resolution_limit_arcsec` is required. If a curve is
  supplied and the limit is omitted, the Gaia DR2 close-pair limit of 2.2 arcsec
  is used.
- `contrast_provenance` requires a measured curve. A record without a curve is
  refused.
- `feh_dex` and `metallicity_provenance` are supplied together. `feh_dex=None`
  with a provenance record is refused.
- A neighbour's `M_s` and `R_s` are both supplied or both `None`. A neighbour
  with neither must state why; a neighbour with either must not carry the
  unavailable reason.
- A neighbour's `delta_mag` and `delta_mag_unavailable_reason` are mutually
  exclusive. The same applies to `metallicity_provenance` and
  `metallicity_unavailable_reason`, and to
  `aperture_flux_fractions_upper_95` and
  `aperture_flux_fraction_limitation`.
- At binding, `sectors` must be non-empty and duplicate-free; every weighted
  sector must appear in the target fraction list and in every neighbour's joint
  fraction list. `bind_scene` requires `sector_weights` to equal the scene's
  sectors, so the weighted set is the scene sector set. Extra fraction sectors
  are unused.
- The target's weighted fraction must be positive. The sum of the target and
  all neighbour joint fractions must not exceed 1 beyond `eps * N`, where `N`
  is the number of supplied shares.
- Neighbour `source_id` values must be unique. A source ID outside the scene is
  refused in `neighbor_parameter_evaluations`.

## Building the aperture fractions

The fraction `f` is the source's share of the total in-aperture flux, not its
share of the catalogue brightness. A correct measurement accounts for:

- the shape of the aperture mask in each sector,
- the position of each source in that mask,
- the pixel response function at that position,
- the relative brightness of the sources in the TESS band.

If you have a point-response-function model, integrate each source's model over
the aperture pixels, sum the integrals, and divide each by the total. Do this
per sector, because the aperture and the source positions change between
sectors.

Rules the scene enforces:

- each fraction list is non-empty; each sector is a positive integer and each
  fraction lies in `[0, 1]`,
- a sector cannot appear twice in one list,
- every weighted sector must appear in the target fraction list and in every
  neighbour's fraction list; extra fraction sectors are stored but unused
  (`bind_scene` requires the weights to equal the scene's sectors, so the
  weighted set is the scene sector set),
- the target fraction must be positive after sector weighting,
- the joint target and neighbour fractions must not exceed 1 beyond
  floating-point roundoff.

At binding, the per-sector fractions are reduced with the observed
inverse-variance sector weights. That reduced value is what enters the
renormalization and dilution. If the weights do not describe the folded light
curve, the dilution will not match the data. The reduction also refuses a
neighbour fraction list that does not cover an observed sector, with
`neighbor aperture scene does not cover observed sector N`.

## Contrast curves

A measured curve bounds unresolved companions where it has support. Accepted
forms are:

- a path to a two-column text file: separation in arcsec, limiting Δmag in mag;
  the path must name an existing regular, non-symlink file,
- a `(separations, delta_magnitudes)` pair of lists or arrays,
- a NumPy array of shape `(2, N)`.

The file parser skips blank lines, comment lines and non-numeric header rows;
a numeric row must contain exactly two finite columns, and file order is
retained. A header row whose first token is not numeric is skipped, but a
misaligned numeric row raises.

Requirements:

- at least two points,
- finite, positive, strictly increasing separations,
- finite contrast values; contrast monotonicity is not imposed,
- `contrast_band="TESS"` when the curve is passed to `from_scene`.

TREX evaluates companion contrasts in the TESS band. It has no verified
bandpass conversion, so a curve measured in another filter is refused rather
than converted:

```text
ValueError: TREX evaluates companion contrasts in the TESS band only; no
verified bandpass conversion is implemented, so a non-TESS contrast curve is
refused
```

`contrast_band` is required when you pass `contrast_curve` to `from_scene`.
Passing a band without a curve raises
`ValueError("contrast_band requires contrast_curve")`. The band name is matched
case-insensitively; when `from_scene` builds the provenance record it stores the
band in upper case.

The curve's coverage matters. At a separation inside the inner working angle or
beyond the last measured point, the curve provides no limit, and the companion
stays in the prior. That is the conservative direction for false positives: it
keeps possible blends rather than excluding them. A small draw budget can
return `-inf` evidence when no sampled draw survives the curve's limit. That is
a sampling outcome, not proof that the scenario has empty support.

`contrast_provenance` records the curve kind, source, band and a limitation
string. When you pass a curve through `from_scene`, TREX builds that record from
the file name or the array origin and rebuilds the scene. The curve is then part
of the scene. `calc_probs(contrast_curve_file=...)` raises
`NotImplementedError`; pass the curve at binding instead.

## Retained input hashes

`input_artifacts` maps retained calibration or photometry files to SHA-256
digests. Each named path must be a regular, non-symlink file. TREX verifies the
mapping:

- when the scene is bound,
- again after inference, before the result is accepted,
- and when a resumed run compares the retained handoff.

An `input_artifacts` file that changes before one of those checks is refused
with `bound input artifact is missing or changed: PATH`. The background
population is always verified separately by `scene.verify_background()`, even
when it is not included in `input_artifacts`; a changed background raises
`trilegal_cache changed after scene construction` (or `trilegal_cache is no
longer an available regular file`), and the mapping adds other files to the
checked and recorded set.

If you omit `input_artifacts`, TREX hashes the background population file and
records that. It does not silently claim to have verified files it never saw.
Relative artifact paths are resolved against the process working directory at
binding and recorded as absolute paths.

The hash proves identity, not physical calibration. It does not tell you
whether the aperture model or the stellar parameters are correct.

## Reusing a scene on another machine

`scene.to_dict()` exports the constructor inputs, including the background
path and digest. It records paths, not file contents, and it does not truncate
the neighbour list. `TargetScene.from_dict(payload)` revalidates every field
and the background bytes. To transfer a scene:

1. copy the background CSV,
2. update `trilegal_cache` to its new path,
3. keep `background_sha256` unchanged,
4. if the population came from the LSST-simulation fallback, keep
   `background_population_provenance` in the payload,
5. revalidate with `TargetScene.from_dict`, then call `verify_background()`.

If the payload carries `background_sha256` and the copied bytes differ from it,
`from_dict` raises `trilegal_cache does not match background_sha256`. Supply a
new digest only when you deliberately intend to bind a different population;
that is a new input, not a transfer of the old one.

Paths in the payload are strings. A relative `trilegal_cache` is resolved
against the process working directory, so use an absolute path when the working
directory can change. If an advanced neighbour record contains NumPy arrays,
convert them to lists before JSON encoding; the loader accepts lists and
revalidates them.

The exported dict is conditional input, not a calibration certificate. It
carries no file contents and no proof that the aperture model or stellar
parameters apply to the observation.

## What this path cannot do

- `plot_field` raises `ValueError("field image unavailable: from_scene does not
  synthesize acquired pixels")`: there are no acquired pixel images.
- `calc_depths` raises `ValueError("acquired pixel coordinates are required for
  the Gaussian diagnostic")`: it needs acquired pixel coordinates, not a scene.
- `get_spoc_apertures` is not an offline substitute. It still queries and
  downloads the SPOC aperture by TIC ID; `from_scene` does not call it.
- `target.from_scene` does not query the TIC, Gaia, TRILEGAL or any other
  service.
- It does not synthesize catalogue values. The target star table contains only
  the supplied scene fields; unavailable neighbour colours, temperatures,
  parallaxes and coordinates stay `NaN`.
- It does not check that your stellar parameters are physically consistent or
  that your aperture model is calibrated. It checks structure, ranges,
  provenance completeness and the background digest.
- It does not create a contrast curve. Without one, the resolution limit is the
  only unresolved-scene bound.
- It does not calibrate the light-curve noise. `calc_probs` uses the fixed
  `sigma` you supply; `calibrate_trex_noise` is a separate native-photometry
  step.
- An omitted seed is not fixed in advance: `random_seed=None` generates one run
  seed and records it, so two separate runs differ unless you supply the same
  seed or reuse the recorded one.

## When to prefer `from_tic`

Use `from_tic` when:

- the target is in the TIC and Gaia DR3,
- the catalogue aperture approximation (a fixed 0.75-pixel Gaussian) is
  acceptable for your purpose,
- you want the archive provenance and the fallback behaviour recorded for you.

Use `from_scene` when:

- you have a better aperture model,
- you are combining inputs from several sources that the acquisition ladder does
  not support,
- you need a reproducible offline run,
- or you are varying one input at a time in a controlled comparison.

Either way, the caller owns the judgement about whether the measured inputs and
approximations apply to the observation.

## Related pages

- [Inputs](inputs.md): every scene field, unit and rule.
- [Prepared runs](prepared-runs.md): the same scene and light curve from three
  files on the command line.
- [Outputs](outputs.md): `summary.json`, `scenarios.csv` and `failure.json`.
- [Python API](api.md): signatures, return values and the bound attributes.
- [From a TIC ID](from-tic.md): the acquisition path.
- [Background population](background-population.md): the CSV format and the
  fallback.
- [Scenarios](scenarios.md): what the host scenarios in the result mean.
- [FAQ](faq.md): common refusal messages and paused runs.
- [Tutorial](tutorial.md): a complete synthetic scene you can copy.
