# Background population

This page answers the question "what are the background stars in the model, and
where do they come from?" It documents the TRILEGAL query, the population file,
the recorded fallback provider, the cache rules, and what a change of population
does and does not let you conclude. It applies to the `DTP`, `DEB`, `DEBx2P`,
`BTP`, `BEB` and `BEBx2P` scenarios.

## Why TREX needs a background population

An aperture can contain an unrelated star behind or near the target. If that
star is eclipsed by an unseen companion, or if a planet transits it, the combined
aperture light dips even when the target itself is constant. A background star
also adds constant light and dilutes a transit on the target. The star can be
too faint or too close to appear as a separate catalogue source, so it is not in
the resolved-neighbour list.

TREX represents this unresolved population as a simulated list of stars for the
target's field. Each background-scenario draw selects one star and a projected
separation, then evaluates the event on the resulting flux ratio and geometry:

- `DTP`, `DEB`, `DEBx2P`: the event is on the target; the background star
  dilutes the light.
- `BTP`, `BEB`, `BEBx2P`: the event is on the background star itself.

The population is mandatory scene input. `TargetScene` requires a retained
population CSV, a SHA-256 digest (supplied or computed from the file), and a
positive simulated area; `run_trex_vetting` loads the population before Monte
Carlo. A missing or unreadable population stops the run. TREX does not drop
those six scenarios and continue with the rest, and it does not assign them zero
probability. [Scenarios](scenarios.md) gives the physical definitions.

## The TRILEGAL query

The primary source is the TRILEGAL 1.6 web service (Girardi et al. 2005, *A&A*
436, 895, https://doi.org/10.1051/0004-6361:20042352). TREX submits a 0.1 deg²
field centred on the target with:

| Form setting | Value | Reason |
|---|---|---|
| Coordinates | Galactic longitude and latitude in degrees (`gal_coord=1`, `gc_l`, `gc_b`) | TRILEGAL reads its equatorial right-ascension field in hours; Galactic degrees leave no hour/degree choice |
| Photometric system | TESS + 2MASS (`tab_mag_TESS_2mass.dat`) | Provides the TESS magnitude the evidence uses, plus 2MASS context |
| Limiting magnitude | 21 in the first filter of that system (TESS) | Matches the reference workflow's limit |
| Binaries | Off (`binary_kind=0`) | The population entries are single stars; an eclipsing companion of a background star is a `BEB`/`BEBx2P` scenario, not a population row |
| Extinction at infinity | Field value from the SFD98 dust map with `A_V = 2.742 E(B−V)` | The form default of 0.0378 mag is not this field's extinction |
| Field size | 0.1 deg² | The simulated area recorded with the population |

### Field extinction

TREX queries the IRSA SFD98 E(B−V) map over the same 0.1 deg² cone, whose radius
is 0.1784 deg. The value submitted as `extinction_infty` is

```text
A_V = 2.742 × max(E(B−V))
```

over the map pixels inside the cone, clamped at zero. The record keeps the map's
maximum, mean, minimum, centre value and pixel count, the map file, the retrieval
date and the conversion. The conversion is Schlafly & Finkbeiner (2011), *ApJ*
737, 103, Table 6, for Landolt V with the `R_V = 3.1` F99 reddening law; the map
is Schlegel, Finkbeiner & Davis (1998), *ApJ* 500, 525. The 2.742 factor includes
the Schlafly & Finkbeiner (2011) 14 per cent recalibration of the SFD98 scale.

That recalibration is measured on high-latitude SDSS stars with
`E(B−V)_SFD < 1`. Inside the Galactic plane it is an extrapolation, and the
Planck 2014 map used by the fallback gives larger extinctions there. The map
query fails rather than returning a partial maximum if the cutout does not cover
the whole cone or if any pixel inside it is missing.

The map record is required. A web population is not resumed, reused or bound
unless it states `field_submission: "galactic-lb-deg"`, carries an available
`field_extinction` record, and its submitted `extinction_infty` agrees with that
record's `av_infinity_mag` to 5×10⁻⁵ mag. The tolerance matches the form's four
decimal places. A record written before this field-extinction amendment carried
the server's 0.0378 mag default for every field; it is refused rather than
silently reused.

Leaving `extinction_infty` at the default would give every field the same
asymptotic extinction. TRILEGAL uses that value to normalize its disc extinction
layer, so a fixed default is not a measurement of the field and is wrong by
orders of magnitude near the Galactic plane.

The reason for Galactic coordinates is a unit problem, not a preference:
TRILEGAL's `eq_alpha` field is read in hours, while a right ascension supplied in
degrees is a different number. Galactic longitude and latitude are degrees by
definition. [Differences from TRICERATOPS](differences-from-triceratops.md)
records the reference implementation behaviour this avoids.

### Submission, polling and retention

TREX reads the form, checks every override against the controls the form
actually offers, submits it, and polls the result URL. The form POST has a 600 s
timeout because live submissions have stayed unanswered past 300 s. The
acquisition polls every 10 s for up to 90 attempts. A result link is accepted
only on the same HTTPS host and under `/tmp/`; redirects are refused. A result is
complete only when its last non-empty line begins with
`#TRILEGAL normally terminated`.

An abnormal termination footer is retained as a failure and marks the job
finished, so a rerun submits a new job instead of polling the dead result URL. If
the result never terminates within the polling budget, the stage is recorded as
a retryable outage.

TLS verification is never disabled. The session verifies against certifi (or the
system trust store if the certifi bundle is unusable). If the server presents a
leaf certificate without its intermediate, TREX attempts a narrowly scoped AIA
completion: it fetches the intermediate named by the leaf's `caIssuers` URL,
accepts it only when it verifies against the certifi roots and directly issued
the leaf, and uses a temporary PEM bundle for that session. An AIA-completed
query record carries a `tls_trust` entry naming the source.

The TRILEGAL session retries GET requests (HTTP 429 and 5xx). The form POST is
not retried, because a resent form would start a second TRILEGAL simulation.

The acquisition writes these files under `background/`:

| File | Written when | Contents |
|---|---|---|
| `trilegal_request.json` | after the result link, then at completion | coordinates, field, area, submitted form, result URL, field extinction |
| `trilegal_stage.json` | on failure | provider stage, exact exception, retryability, submitted form, field record |
| `trilegal_field_extinction.fits` | when the dust map answers | the SFD98 E(B−V) cutout used for `extinction_infty` |
| `trilegal_output.dat` | on success | raw provider table |
| `trilegal_cache.csv` | on success | normalized population, the file the scene reads |
| `trilegal_query.json` | on success | provider and model, field, area, submitted form, field extinction, population and raw-output digests, star count |

A successful TRILEGAL acquisition removes any stale `trilegal_stage.json`. A
`status: "submitted"` request record lets a rerun poll the running job for the
same ICRS field and field extinction instead of spending a second simulation. An
abnormally terminated job is marked `completed`, so it is not polled again. A
failed attempt writes a stage record; it does not overwrite a retained
`trilegal_query.json` or `trilegal_cache.csv`.

### What triggers the fallback

`acquire_background_population` tries TRILEGAL first. If `acquire_trilegal` raises
`TrilegalUnavailableError` or `requests.RequestException`, it tries the recorded
LSST-simulation fallback described below. The trigger can be the form page, the
submission, the result link, polling, TRILEGAL output validation, or the
field-extinction map stage; the stage record names which one failed. A
`ValueError` from the caller's own inputs (for example, an invalid target
coordinate) is not converted into a fallback attempt.

If the fallback also fails, the original TRILEGAL exception is raised unchanged
in meaning, with the fallback failure attached as `fallback_failure` and written
into the stage record when the stage record names a stage.

## The population file

`trilegal_cache.csv` has one header row and one row per simulated star. The
header names the columns; TREX reads the required and retained optional columns
by name and ignores the rest. A web population keeps every column TRILEGAL
returned. A fallback population also carries the integer `Gc` (Galactic
component) and `label` (evolutionary-stage label) columns, which TREX checks but
does not use in the evidence.

| Column | Unit | Meaning |
|---|---|---|
| `TESS` | mag | Apparent TESS magnitude, including distance and extinction |
| `Mact` | M☉ | Current stellar mass |
| `logg` | dex | `log10(g / cm s⁻²)` |
| `logTe` | dex | `log10(T_eff / K)` |
| `[M/H]` | dex | `log10(Z/Z_sun)`, not a mass fraction and not a measured iron abundance |
| `Av` | mag | Optional: the row's total V-band extinction |
| `m-M0` | mag | Optional: the row's distance modulus |

`Av` and `m-M0` are retained as `av_mag` and `distance_modulus_mag` when present.
An absent value is not replaced by zero. `Av` is used when a physical
stellar-light model is bound; `m-M0` is provenance and does not enter the
evidence.

### Parsing and row selection

- The first non-empty line is the header. A single leading unnamed index column
  (a legacy pandas index) is tolerated; any other empty or duplicate column name
  raises.
- Empty lines are skipped. Only the recognized `#TRILEGAL` termination footer is
  accepted as a non-data row; any other comment or text row raises, and data
  after the footer raises.
- A missing required column raises. A truncated row, or a non-numeric or
  non-finite value in a required or retained optional column, raises. A negative
  `Av` raises, and so do a non-positive or non-finite mass, temperature or
  derived radius.
- The loader keeps only rows with `TESS >= Tmag`: a brighter unresolved star
  would dominate the aperture and is not the modelled configuration. The file
  still contains the brighter simulated stars.
- If no row is kept, `TargetScene.background_population()` raises and the run
  does not proceed. An empty retained list is not evidence that the background
  is empty.

The radius is derived from the row's mass and surface gravity:

```text
R = sqrt(G M / g)
```

so an evolved simulated star keeps its simulated size instead of being forced
onto a main-sequence relation.

### Counts and area

Three different counts appear in the record and the scene:

| Quantity | Where | What it counts |
|---|---|---|
| `star_count` | `trilegal_query.json` | Rows written to the population CSV. For a web population, every simulated star TRILEGAL returned. For the fallback, every converted star with `TESS < 21`. |
| `N_background` | `TargetScene` | The recorded `star_count`, as a non-negative integer. It is provenance; nothing in the evidence reads it. |
| retained rows | `load_trilegal_population` | Rows with `TESS >= Tmag`. This count is `N_stars` in the occurrence weight below. |

`N_background` is therefore not the number of stars a draw can select. The file
is authoritative for that: the loader applies the target-magnitude cut every
time the population is loaded.

The simulated area is `simulated_area_sq_deg` in the query record, 0.1 deg² for
both providers. `TargetScene.background_area_sq_deg` is the positive area the
scene binds; `target.from_tic` copies it from the record. The area is not
inferred from the CSV, which carries no area. TREX does not require the area to
be 0.1: a portable scene may state another positive simulated area, and the
occurrence weight uses that value. The occurrence-weight code takes the area as
a required argument instead of hardcoding the reference query's 0.1 deg².

Row order carries no meaning. The draw is uniform over the retained rows. The
nested sampler sorts a copy by TESS magnitude for its own grid, which does not
change the population.

The CSV has no per-star coordinates. The record's field is the ICRS pair
`ra_deg`/`dec_deg`, with the submitted Galactic `gc_l_deg` and `gc_b_deg` for a
web population. `field_submission` states which frame was used:
`galactic-lb-deg` for the web query, `icrs-cone-deg` for the fallback.

## How the population enters the evidence

For each background-scenario draw, TREX:

1. selects a row uniformly from the retained rows;
2. draws a separation uniformly in area out to the unresolved limit,
   `separation = r_limit · √u`;
3. computes the TESS flux ratio from the magnitude difference,
   `F_bg/F_target = 10^(−0.4 (Tmag_bg − Tmag_target))`, and the in-aperture
   fraction `F_bg/(1 + F_bg)`;
4. reads the row's mass, radius, temperature, `[M/H]` and, when present, `Av`;
5. evaluates the scenario on that star and geometry.

If a measured contrast curve is bound, a draw whose contrast is brighter than
the curve at its separation gets zero weight. The curve can therefore reduce the
number of admissible draws; it does not change the population file.

The expected number of such stars inside the unresolved field is

```text
λ = N_stars / (A_sim · 1.296×10⁷ arcsec² deg⁻²) · π r_limit²
```

with `N_stars` the retained rows, `A_sim` the scene's `background_area_sq_deg`
in square degrees, the conversion factor turning it into square arcseconds, and
`r_limit` the unresolved limit in arcseconds. The base log weight of the draw is
`log λ`; the imaging constraint can set it to −∞. This is an occurrence weight,
not a probability that a background star exists.

For the population, the Monte Carlo integrates over the star row and the
separation. Everything else in the population model is conditioned on, not
marginalized:

| Quantity | Treatment |
|---|---|
| Field centre, 0.1 deg² area, TESS = 21 limit | Fixed by the query and recorded |
| Population realization | Fixed; TREX does not resample the file or marginalize over TRILEGAL model uncertainty |
| Star row | Drawn uniformly from the retained rows |
| Projected separation | Drawn uniformly in area to `r_limit`; a measured contrast curve can reject a draw |
| Flux ratio | Determined by the drawn TESS magnitude |
| Mass, radius, temperature | The drawn row's simulated values |
| `[M/H]` | The drawn row's value; used when the star hosts the event |
| `Av` | The drawn row's value; used with a bound stellar-light model |
| Distance | Not drawn; it is already implicit in the apparent TESS magnitude. `m-M0` is provenance only |
| `λ` | Deterministic from the retained count and the stated area; not a fitted parameter |

The retained file is one realization of the simulated field. Its sampling noise,
the choice of TRILEGAL model, and the difference between the simulation and the
actual stars behind the target are not sampled or marginalized. The count `λ`
has no attached uncertainty in the evidence.

The star's own `[M/H]` sets its limb darkening when it hosts the event; it does
not inherit the target's composition. For the `D*` scenarios the event is on the
target, so the target's composition sets the event and the background star only
dilutes. Per-star `Av` is a row property: the web query's submitted
`extinction_infty` fixes the field normalization, while the fallback uses its own
Planck 2014 per-star extinctions. The target's own extinction is separate.

Two consequences follow:

- A larger retained count at fixed area raises `λ` and the weight of the
  background scenarios. It does not by itself say whether the normalized FPP
  rises or falls, because the denominator also contains scenarios that do not
  use the population.
- The star is chosen uniformly from the retained list, not weighted by flux. The
  magnitude cut and the flux ratio do the brightness weighting.

## The recorded LSST-simulation fallback

If the web-population stage cannot produce a population, TREX uses a recorded
population from the TRILEGAL LSST simulation (Dal Tio et al. 2022, *ApJS* 262,
22, https://doi.org/10.3847/1538-4365/ac7be6) for the same 0.1 deg² cone. It is
the same TRILEGAL code's precomputed simulation for the LSST footprint, read from
`lsst_sim.simdr2` at the NOIRLab Astro Data Lab. It is a fallback, not a
substitute for a web result.

The footprint covers fields with declination below 5°, ecliptic latitude within
10°, or the Galactic plane (`|b| < 10°`) below declination 35°. A field must be
inside that union by at least the cone radius (0.1784 deg), and the cone must
contain at least one star that converts to `TESS < 21`. Otherwise no fallback
exists and the original TRILEGAL failure is raised.

A fallback record identifies itself:

```json
{
  "model": "trilegal-lsst-sim-dr2",
  "fallback": true,
  "field_submission": "icrs-cone-deg",
  "fallback_trigger": {
    "status": "unavailable",
    "stage": "...",
    "error": {"exception_class": "...", "message": "..."}
  },
  "provider": "NOIRLab Astro Data Lab",
  "table": "lsst_sim.simdr2",
  "population_sha256": "..."
}
```

The fallback writes the raw TAP rows to `lsst_sim_output.csv`, the normalized
population to `trilegal_cache.csv`, and the fallback record to
`trilegal_query.json`. A failed TRILEGAL stage record may remain beside them.
The record also keeps the TAP URL and table, the ADQL query, `rows_returned`,
`tess_relation_counts`, the Gaia and TESS limits, and `field_av_max`.

The fallback record carries these differences from a TRILEGAL 1.6 web
population:

- the tracks and Galaxy calibration are PARSEC v1.2S + COLIBRI with the Dal Tio
  et al. (2022) calibration;
- extinction is per star from the Planck 2014 dust map, not one asymptotic field
  value;
- photometry is LSST and Gaia only; the TESS magnitude is derived from Gaia G,
  BP and RP with the TIC v8 relation of Stassun et al. (2019, *AJ* 158, 138),
  and the relation used for each star is counted in the record;
- the TAP query selects Gaia `G < 23` and writes only `TESS < 21`; the returned
  row count is checked against a `COUNT(*)` of the same selection before the
  population is accepted;
- the field attenuation envelope is the largest per-star `Av` among the
  simulated stars with Gaia `G < 23`, not the SFD98 value.

The record's limitation states that same-cone star counts can fall below Gaia DR3
counts, and that the effect of that deficit on a particular FPP is not measured.
A fallback-derived FPP is not interchangeable with one from the TRILEGAL web
service for a quantitative comparison; the record carries that limitation and
the trigger. Do not state that a fallback makes a given FPP higher or lower.
The normalized FPP also contains scenarios that do not use the population, so
the direction is not fixed.

An automatic rerun in the same `output_dir` does not reuse a fallback record:
`background/trilegal_query.json` with the fallback model is skipped and TRILEGAL
is tried again. If you explicitly pass that fallback record to
`trilegal_population`, it is admitted as a fallback with its provenance and
limitation, not as a current web result.

## Reusing a retained population

`target.from_tic(..., trilegal_population=path)` accepts a retained
`trilegal_query.json` beside its population CSV instead of running a new query.
TREX admits it only when:

- the query record is readable and states a field this acquisition may use: a
  web record must carry `field_submission: "galactic-lb-deg"` and an available
  `field_extinction` whose `av_infinity_mag` agrees with the submitted form; a
  fallback record must name the fallback model and
  `field_submission: "icrs-cone-deg"`;
- the population file is a regular file beside the record, not a symlink, and
  its SHA-256 matches the record's `population_sha256`;
- the record's field is within the requested `search_radius_arcsec` of the
  target. The default is 60 arcsec. This check uses the requested radius, not the
  wider Gaia cone that the aperture reach may produce.

A record from a different field is rejected. For example:

```text
ValueError: retained TRILEGAL population was simulated 123.4 arcsec from this target
```

A population from a different sky position can have a different density,
extinction and metallicity distribution. Reusing it would silently change the
background scenarios. A web record that does not carry the Galactic field
submission and the map extinction is also refused; the message names the
pre-amendment field bug, and a web record written before the field-extinction
amendment is unusable for the same reason. Acquire a new population for that
field.

The acquisition records the supplied path in the scene; it does not copy the
files into the new `output_dir`. Keep them where the scene can read them.

The same digest and file checks apply when a scene is loaded from a dict:
`background_sha256` binds the file, and `verify_background()` re-checks it before
binding and again when the population is loaded. A scene that carries a fallback
provenance projection must name the same digest. These checks prove file
identity; the acquisition-time field check is what proves the population was
simulated at this target.

### Cache freshness and refusal

Within one `output_dir`, a rerun reuses an `available` web record when the
population bytes still match its digest. A fallback record is skipped so TRILEGAL
is tried again. A record that is not in the required field form raises and tells
you to acquire the scene in a new `output_dir`; it is not silently replaced. A
missing or changed population file on this path causes a new acquisition rather
than a hard error. The `scene_request.json` must still match the original
request exactly.

There is no age-based expiry. `retrieved_utc` is provenance, not a time to live.
A population is reused as long as the request and record match and the bytes
still match their digest. Re-query when you want a new realization, when the
retained record predates the Galactic field or field-extinction amendment, when
the record is for another field, or when the bytes are missing or changed. The
next automatic acquisition tries TRILEGAL again; the fallback is used only if
that fails.

A population becomes unusable when:

- the CSV is missing, is a symlink, or no longer matches the recorded SHA-256;
- the query record is missing, unreadable, or not `available`;
- a web record lacks the Galactic field submission or the map extinction, or its
  submitted `extinction_infty` disagrees with the recorded value;
- a fallback record is inconsistent or names another population digest;
- the field is outside the search radius on an explicit reuse, or the same
  `output_dir` is asked for a different request;
- no row is at least as faint as the target.

The last case is a population/target mismatch, not a corrupt cache. The TRILEGAL
query stops at TESS 21, so a target fainter than that limit can leave no row with
`TESS >= Tmag`. Re-querying at the same limit will not necessarily help; check
the target magnitude against the limit before treating the failure as a service
problem.

## What the population does not provide

- **Sky positions of individual stars.** The retained cache carries no
  positions, so TREX cannot cross-match a simulated star against the Gaia
  neighbours inside the unresolved boundary. That overlap remains a model
  limitation.
- **A measured luminosity function or empirical calibration.** TRILEGAL is a
  simulation, not a catalogue of the actual stars behind the target. TREX applies
  no empirical calibration to either provider.
- **Binary stars in the list.** The query runs with binaries off; an unresolved
  binary is represented by a scenario family, not by a population row.
- **Completeness.** The magnitude limit and the simulated area define the list.
  Fainter stars are absent. A missing row is not evidence that no blend exists;
  it means the simulation did not produce that star within its limit and area.
- **Population-level uncertainty.** The file is one realization. TREX does not
  sample its sampling noise, TRILEGAL model uncertainty, or cosmic variance, and
  `λ` carries no uncertainty into the evidence.
- **Full-sky coverage.** Each query covers 0.1 deg², and the fallback is
  restricted to the LSST simulation footprint. The population is a field
  realization, not a sky survey.
- **Measured membership, proper motions or variability.** The rows are simulated
  stars; they are not tied to observed sources.

## Practical guidance

Supply a retained population when you already have a digest-bound
`trilegal_query.json` and its CSV for this field, when you want to avoid a new
simulation on a host where the TRILEGAL service is slow or unreliable, or when
you need the same bytes across runs. Keep the query record and the CSV together;
the population file is found beside the query record, and it must keep the
basename recorded in `population_file`. Do not hand-edit either file: the digest
protects the CSV bytes, not the record.

Re-query when the retained record is for another field, predates the Galactic
field or field-extinction amendment, or no longer matches its file. Re-query when
you want a new realization or when a fallback was used and the web service is
available again. Check a target fainter than TESS 21 before re-querying; the
21 mag limit is a real boundary.

Record the provider and model, the field, the simulated area, the star count, the
retrieval date, and whether the population is a fallback. A `TargetScene`
records the population path, its SHA-256, the area, and the fallback provenance
when one applies; `TargetScene.to_dict()` is the portable form. A fallback
result's `background_population` record carries its limitation and trigger.

## Related pages

- [From a TIC ID](from-tic.md): when the query runs and what is retained.
- [Inputs](inputs.md): the population fields and their validation.
- [Scenarios](scenarios.md): what `D*` and `B*` mean physically.
- [How TREX works](how-trex-works.md): how the occurrence weight enters the
  evidence.
- [Outputs](outputs.md): where the background provenance is written.
- [FAQ](faq.md): which file `background.csv` refers to.
