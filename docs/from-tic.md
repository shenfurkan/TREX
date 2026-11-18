# From a TIC ID

`target.from_tic` acquires the stellar scene for a TESS target from public
archives and binds it. It is the shortest path from a TIC ID to a scene that
`calc_probs` can use, but it is not a transit search. It does not find the
signal, measure its depth, calibrate the light-curve noise, localise the source,
or acquire high-resolution imaging. Those remain your steps.

The complete light-curve-to-probability example is the [tutorial](tutorial.md);
this page explains what the call queries, in what order, what it retains and
what it approximates. The scene fields are in [Inputs](inputs.md).

## The call

```python
from pathlib import Path
from trex.target import target

star = target.from_tic(
    TIC_ID, SECTORS,
    output_dir=Path("scene"),
    depth_ppm=DEPTH_PPM,
    sector_weights=SECTOR_WEIGHTS,
    exptime_days=EXPTIME_DAYS,
    random_seed=1234,
)
```

The full signature is:

```python
target.from_tic(
    ID, sectors, *, output_dir, depth_ppm, cache_dir=None, policy=None,
    search_radius_arcsec=60.0, trilegal_population=None,
    contrast_curve=None, contrast_band=None, **binding,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `ID` | required | TIC v8 identifier, a positive integer |
| `sectors` | required | observed TESS sectors; non-empty, no duplicates |
| `output_dir` | required | directory for the request and every archive response |
| `depth_ppm` | required | measured transit depth in ppm, `0 < depth_ppm < 10^6`; decides which neighbours need stellar parameters and is bound for the calculation |
| `cache_dir` | `output_dir/cache` | shared GALAH DR3 and MIST releases |
| `policy` | network allowed | `NetworkPolicy` for archive requests |
| `search_radius_arcsec` | `60.0` | minimum Gaia cone radius; the aperture reach can widen it |
| `trilegal_population` | `None` | a retained `trilegal_query.json` for this field, reused instead of a new simulation |
| `contrast_curve`, `contrast_band` | `None` | optional measured TESS-band contrast curve; the band is required with the curve |

`**binding` is the rest of the `bind_scene` contract. The important entries are:

| Binding entry | Default | Meaning |
|---|---|---|
| `sector_weights` | required | sum of `1 / sigma_i^2` over the retained native cadences, one positive weight per observed sector |
| `exptime_days` | required | native exposure in days; `calc_probs` must receive the same value |
| `bin_integration_days` | `0.0` | phase-bin width in days; `0` for unbinned cadences |
| `random_seed` | `None` | non-negative integer recorded with the result; if omitted, `calc_probs` draws and records one |
| `input_artifacts` | `None` | mapping of retained calibration or photometry paths to SHA-256; when omitted, only the background population is hashed and bound |
| `depth_sigma_ppm`, `duration_days`, `duration_sigma_days` | `None` | measured values that centre the draws; leave unset only when they were not measured |

`sector_weights` must cover exactly the scene's sectors. If binding fails, the
acquired scene is still retained in `output_dir`; correcting the arguments and
running again reuses it.

A successful call returns a bound `target`. It does not calculate
probabilities: `calc_probs` is a separate call, and the draw budget is chosen
there. Acquisition does not depend on `N`.

## What is acquired, and from where

| Product | Source | Use in the scene |
|---|---|---|
| Target-pixel files | MAST, one shortest-cadence SPOC TPF per sector | Pipeline aperture mask, WCS, cadence and observation epoch |
| TIC v8 rows | MAST TIC services | Target coordinates and identifiers; TIC neighbour mass, radius and temperature fallbacks |
| Field sources | Gaia DR3 cone, ESA Gaia TAP (AIP mirror as fallback) | Resolved neighbours and their astrometry |
| Astrophysical parameters | Gaia DR3 `astrophysical_parameters`: GSP-Phot and FLAME | Target/neighbour temperatures, gravities, radii, masses and extinctions |
| Host tables | Gaia AIP TAP: StarHorse EDR3 and SHBoost | Alternative mass, radius, distance, extinction and marginal distributions |
| Metallicity | APOGEE DR17 (SDSS SkyServer), GALAH DR3 (Data Central release), calibrated GSP-Spec, SHBoost | Target and neighbour composition |
| Background population | TRILEGAL 1.6, or the recorded TRILEGAL LSST-simulation fallback | Unresolved background stars |
| MIST release | Official MIST v1.2 archive, retained with URL and retrieval date | A G-band plus parallax mass/radius measure for a neighbour without BP photometry |

Every archive response is written under `output_dir`; shared releases go under
`cache_dir`. The retained record is the provenance: you can inspect what the
archive returned, not only what TREX used.

## The order

### 1. Target-pixel files first

TREX retrieves each sector's SPOC target-pixel file from MAST before fixing the
Gaia cone radius. The pipeline aperture and its WCS determine how far from the
target a star can still contribute light. A retained product is revalidated for
target identity, completeness and sector before it is reused; the product for
each requested sector must be unique.

The aperture reach is the farthest aperture-pixel corner plus four standard
deviations of the declared 0.75-pixel Gaussian, converted with the WCS pixel
scale. The Gaia cone radius is the larger of the requested
`search_radius_arcsec` (60 arcsec by default) and that reach. A bright target
with a large pipeline aperture therefore gets a cone wider than 60 arcsec.

### 2. TIC and Gaia field

The TIC v8 target row supplies the position used as the cone centre. TIC v8
coordinates are J2000; the Gaia DR3 cone is queried around that position. The
target is associated with exactly one Gaia DR3 source within 2 arcsec, using
proper motion to move candidates to the target epoch when the needed astrometry
is present. A cone that gives zero or several compatible targets raises; the
match is not guessed.

Every other Gaia DR3 source in the cone becomes a resolved neighbour. Neighbour
separations are measured from the target's Gaia DR3 catalogue position at the
same catalogue epoch, not from the target's proper-motion-corrected position.
The aperture positions are separately propagated to each sector's observation
epoch when the pixel model is built.

TIC v8 neighbour rows are attached through a Gaia DR3-to-DR2 cross-match and
then the TIC cone, with a per-identity query when the cone does not cover a
source. An ambiguous or unavailable cross-identification is recorded as
unavailable rather than assigned to the nearest row.

A TIC neighbour row without a Gaia DR3 counterpart has no aperture share and is
not modelled. `scene_acquisition.json` states this limitation.

### 3. Stellar parameters

**Host.** The target's temperature, radius and mass follow this order:

1. a TIC v8 row tagged in the Cool Dwarf List (the empirical M-dwarf
   calibration);
2. Gaia DR3 GSP-Phot temperature and radius, with mass from FLAME, else from
   `M = g R^2 / G` using the GSP-Phot log g and radius, else from the TIC v8
   mass;
3. a complete TIC v8 temperature, radius and mass row.

GSP-Phot is used only when its temperature, log g and radius are all present.
The temperature and radius then come from that solution; the mass comes from
FLAME, or from the GSP-Phot log g and radius, or from the TIC v8 mass. A row
lacking log g falls to the complete TIC v8 set. Acquisition raises if no source
supplies all three quantities; nothing is defaulted.

**Metallicity.** After the aperture shares are built, TREX selects the target
composition in this order:

1. APOGEE DR17,
2. GALAH DR3,
3. calibrated Gaia GSP-Spec,
4. SHBoost.

TREX takes the first rung that supplies a measured value for a source. A
catalogue query that is unavailable stops the ladder rather than silently
dropping to the next rung; a completed query with no admitted row continues.
The target's metallicity must be measured. If none of the four sources supplies
one, acquisition raises.

For GALAH, an existing official `GALAH_DR3_main_allstar_v2.fits` cache is
read in place. Otherwise TREX queries Data Central's live
`galah_dr3.main_star` table for the requested Gaia DR2 counterparts and retains
the raw VOTable; that table's `source_id` is Gaia DR2, so the existing
bidirectional DR2 neighbourhood check establishes the DR3 identity. Data
Central's update page records that DR3.2 did not change the main-star stellar
parameters or abundances. TREX does not implicitly download the 833 MB
catalogue. The targeted query is retried with backoff inside a 300 s wall
budget; its attempts, elapsed time, stopping bound (budget, attempts or
classification) and D-55 class (`outage`, `empty-answer`, `malformed-request`,
`tls_verification`) are recorded in the metallicity query record. A 4xx
refusal or a certificate failure is not retried, and a complete zero-row
answer is a result, not an outage. TAP errors, overflow and schema/identity
mismatches stop acquisition rather than selecting a lower rung. Live service
availability is not guaranteed.

The requirement is scientific, not procedural. Every eclipsing-binary scenario
needs the host composition for the eclipsed secondary's atmosphere, and every
bound-companion-hosted scenario needs it for the drawn companion. Treating a
missing value as solar would assign false-positive scenarios a probability they
did not earn.

A neighbour without its own metallicity keeps a recorded reason. On the
reference-grid path, its scenarios may use the target's measured metallicity and
record `metallicity_treatment = "target-metallicity fallback"`; that is not a
measurement of the neighbour. When `stellar_light` is bound, each resolved
source must carry its own atmosphere instead, and the target fallback is
refused.

**Neighbours.** A resolved neighbour is characterised from:

1. Gaia DR3 FLAME, only inside its documented quality, mass and giant-age
   domain;
2. TIC v8 mass and radius, when the cross-identification supplies them;
3. Gaia DR3 GSP-Phot temperature on the main-sequence mass–temperature
   relation, refused below `log g = 3.5`.

A FLAME refusal still permits independent TIC mass/radius measurements.
GSP-Phot substitution after that refusal requires finite main-sequence log g
and no FLAME giant flag. Independent StarHorse, SHBoost and MIST estimates
remain eligible; a flagged giant is never assigned temperature-only dwarf
properties merely to complete the scene. Beyond these three, a source is
characterised only when its largest sector aperture flux fraction reaches 0.9
times the measured depth: a fainter source cannot produce the observed dip even
if fully eclipsed. That supplemental stage uses StarHorse EDR3 (Bayesian
isochrone medians with domain checks), then SHBoost marginal distributions, and
finally a MIST v1.2 G-band plus parallax likelihood when BP photometry is
absent, the parallax signal-to-noise is at least 20, and the field attenuation
is below 1 mag. Supplemental attempts and the original FLAME refusal are
retained in the scene, including when an independent estimate succeeds.

StarHorse medians are used as fixed host inputs; their retained 16–84 envelopes
are not marginalised. SHBoost supplies a marginal distribution that the
evidence calculation integrates, not a single fixed value. MIST is not a
general substitute for missing photometry: it is a prior-informed G-only
measure whose single-star grid, age prior, distance prior and
field-attenuation envelope remain stated limitations.

At calculation time, the neighbour outcomes are not interchangeable. A
neighbour whose aperture share over the observed sectors is below the observed
depth, or is zero, is `host-excluded`: its three slots stay at `-inf` evidence
with zero probability, and the run can still publish. A neighbour whose share
reaches the depth but has no mass and radius is not dropped: the engine's
preflight refuses the whole calculation with `TrexEvidenceUnavailableError`.
Missing neighbour metallicity follows the separate rule above.

### 4. Aperture flux shares

The aperture model is the Giacalone et al. (2021, AJ 161, 24,
<https://doi.org/10.3847/1538-3881/abc6af>) Section 2.1 catalog method:

- take each sector's SPOC pipeline aperture mask;
- convert each Gaia source's photometry to a TESS magnitude;
- place a circular Gaussian with standard deviation 0.75 pixels at the source's
  position in that sector;
- integrate the Gaussian over the aperture pixels exactly, using error-function
  differences per pixel;
- normalise the integrals to get each source's share of the in-aperture flux.

The TESS magnitude comes from Gaia G, BP and RP through the Stassun et al.
(2019, AJ 158, 138, <https://doi.org/10.3847/1538-3881/ab3467>) TIC v8 relation.
For `-0.2 < G_BP - G_RP < 3.5` TREX uses Eq. (1). Inside TIC v8's wider
`[-1.0, 6.0]` extrapolation interval it uses the same polynomial and labels the
value `stassun2019-eq1-tic-extrapolated`. With no colour in that interval it
uses the solar-colour offset `T = G - 0.430` and labels it
`stassun2019-eq2-no-valid-color`. The per-source relation label is retained in
`aperture_scene.json`.

The relation has about 0.1 mag catalogue-level accuracy for sources with a
measured G, and the paper assigns a conservative 0.6 mag error to its no-colour
branch. TREX uses the central magnitude only; it does not propagate either error
into the flux ratio. The relation was fitted to Gaia DR2 passbands, so applying
it to DR3 photometry assumes the passband change is negligible at that level.
Neither the 0.1 mag nor the 0.6 mag figure is a bound on an individual source.

The catalog stage refuses the whole scene if any Gaia source in the cone has no
measured G magnitude, naming the source: no retained epoch-matched measurement
bounds its TESS flux, so it cannot be assigned an aperture share. A TIC
neighbour row without a Gaia DR3 counterpart is ignored instead, and the
limitation is recorded.

Each sector's fractions are recorded separately. At binding time, TREX reduces
them with the observed inverse-variance sector weights, so the dilution in the
model matches the dilution in the folded light curve.

The approximation is one of the most important limits of this path. The real
TESS pixel response function varies with camera, CCD and position in the focal
plane. A fixed circular Gaussian with a 0.75-pixel width is a catalogue
approximation, not a measurement of the scene's light distribution, and the
model is explicitly uncalibrated. A scene built from a measured
point-response-function fit, or from a dedicated aperture model, is preferable
when one is available.

### 5. Background population

TREX submits a 0.1 deg² field centred on the target to TRILEGAL 1.6 in Galactic
longitude and latitude in degrees. The form uses the TESS + 2MASS photometric
system, a TESS magnitude limit of 21 and no simulated binaries. It sets
`extinction_infty` from the SFD98 dust map instead of leaving the form's
0.0378 mag default. The value is the largest SFD E(B−V) among the map pixels
inside the cone, converted with `A_V = 2.742 E(B−V)` (Schlafly & Finkbeiner
2011, ApJ 737, 103, Table 6). That conversion is calibrated at high Galactic
latitude and is an extrapolation inside the plane.

The acquisition retains the population CSV, the submitted form, the query
record and the raw output. A failed attempt also leaves a stage record. The
field-extinction map and its record are retained with them.
[Background population](background-population.md) documents the file format and
the form.

If TRILEGAL cannot answer, `target.from_tic` uses the recorded TRILEGAL LSST
simulation (Dal Tio et al. 2022, ApJS 262, 22) for the same 0.1 deg² cone when
the field lies inside that simulation's footprint: declination below 5°,
ecliptic latitude within 10°, or Galactic latitude within 10° at declination
below 35°, with the cone fully inside. Outside the footprint, the TRILEGAL
failure is raised.

A fallback record states its provider and the failure that triggered it:

```json
{
  "model": "trilegal-lsst-sim-dr2",
  "fallback": true,
  "fallback_trigger": {"...": "the TRILEGAL failure"}
}
```

The next acquisition tries TRILEGAL again; a fallback is never treated as a
current web result. A retained population from another field is rejected. The
fallback uses different stellar tracks, extinction and photometric systems from
the web service, so a fallback-derived FPP is not interchangeable with a web
result; the record says so.

### 6. Contrast curve

`from_tic` does not acquire high-resolution imaging. Without a measured curve,
unresolved companions are bounded by the Gaia DR2 close-pair resolution of 2.2
arcsec (Giacalone et al. 2021, Section 1). You can supply a measured curve:

```python
star = target.from_tic(
    TIC_ID, sectors, output_dir=..., depth_ppm=...,
    contrast_curve="contrast.txt", contrast_band="TESS",
    sector_weights=..., exptime_days=...,
)
```

The curve is validated before any request is written and becomes part of the
scene request. A resumed acquisition cannot drop or switch it. A non-TESS band
is refused rather than converted: TREX evaluates companion contrasts in the
TESS band only, and no verified bandpass conversion is implemented.

### 7. Diffuse priors that are not acquired

The scene does not contain a source-localisation likelihood, a centroid
posterior, a pixel-level difference image, or a measured aperture correction.
`from_tic` binds no `localization_evidence`. A supplied `TargetScene` may carry
one as a retained diagnostic, but the implemented spatial prior gives every
resolved neighbour an exactly neutral weight: it does not reweight or delete a
neighbour from a centroid offset. Read `scene_acquisition.json` limitations and
do not describe the run as a source-localisation measurement.

## Output directory

`output_dir` contains the request and every retained response. Names you will
see include:

```text
scene_request.json
scene_acquisition.json
aperture/tpf_products.json
aperture/aperture_scene.json
aperture/tpf_*.fits
aperture/mast_*.json
field/field_context.json
field/tic_target.json
field/tic_target_query.json
field/tic_neighbors_query.json
field/gaia_cone_query.json
field/gaia_cone_*.xml
host/host_tables_*.json
host/*.xml
metallicity/metallicity_query_*.json
metallicity/*.xml
metallicity/*.csv
background/trilegal_field_extinction.fits
background/trilegal_request.json
background/trilegal_stage.json
background/trilegal_output.dat
background/trilegal_cache.csv
background/trilegal_query.json
cache/MIST_v1.2_vvcrit0.0_UBVRIplus.txz
cache/MIST_v1.2_vvcrit0.0_UBVRIplus.retrieval.json
cache/GALAH_DR3_main_allstar_v2.fits
cache/GALAH_DR3_main_allstar_v2.retrieval.json
```

The exact set depends on which stages ran and which sources answered.
`trilegal_stage.json` records the last failed TRILEGAL attempt; it is removed
when TRILEGAL itself later succeeds, and kept beside a fallback record.
Host-table and metallicity records are updated or added per attempt, and the
newest `"available"` record for exactly the same source list is the one reused.

`scene_acquisition.json` states the TIC ID, sectors, Gaia source ID, depth,
aperture method, host field sources, TESS magnitude relation, parallax source,
the minimum capable aperture flux fraction, cone radius, aperture reach, the
scene itself and a list of limitations. Read the limitations before using the
scene.

Acquisition and binding write no probability file. `write_outputs` writes
`summary.json` or `failure.json` to a separate result directory after
`calc_probs`.

## Re-running and resuming

A rerun in the same `output_dir` must repeat the same request exactly. For the
example call, `scene_request.json` is:

```json
{
  "tic_id": 76923707,
  "sectors": [2],
  "depth_ppm": 7809.0577339,
  "radius": 60.0,
  "trilegal_population": null
}
```

When a contrast curve was supplied, its separations, magnitudes, band and
source are part of the request as well. A different request raises:

```text
ValueError: scene request differs from the retained scene_request.json; use a new output_dir for a different request
```

A non-empty directory without a `scene_request.json` cannot be resumed. This
prevents two scenes from sharing one output tree.

Every stage whose retained record is `"available"` for the same source list is
reused. In detail:

- TPF products are revalidated on every rerun for target identity,
  completeness and sector; a replacement is written beside a history record.
- Field context resumes when its target association and TIC-neighbour records
  are available for this TIC ID.
- Host tables resume from completed table chunks; an interrupted sequence
  continues at the last completed table.
- The catalog aperture scene is reused when its record is available.
- The newest available metallicity record for exactly these sources is reused.
- A TRILEGAL web population is reused only when its field, extinction and
  population digest still match. A fallback population is never reused: the
  next acquisition tries TRILEGAL again.

If acquisition succeeds but binding fails, the scene remains valid; rerunning
the same request with corrected binding arguments reuses the retained stages.

## What can fail, and why

| Stage | Message or condition | Cause | What to do |
|---|---|---|---|
| Directory | `output_dir is non-empty without a scene_request.json` | unrelated files or a partial scene | use a new directory; do not mix scenes |
| Directory | `scene request differs from the retained scene_request.json` | same directory, different TIC, sectors, depth, radius, population or curve | use a new directory |
| TPF | `MAST returned no SPOC TPF for sector N` or `... ambiguous shortest-cadence TPFs ...` | no unique SPOC product for the sector | check the sector and archive status |
| TPF | `retained TPF record belongs to a different target` | directory reused for another TIC | use a new directory |
| Field | `Gaia target association requires exactly one compatible source` | zero or several Gaia sources at the TIC position | check the TIC and Gaia position; do not force a match |
| Host | `Gaia DR3 host lacks the G photometry or parallax a scene requires` | incomplete Gaia row | use a different target or a supplied scene |
| Host | `host lacks GSP-Phot Teff/log g/radius and a complete TIC v8 Teff/radius/mass` | no complete host-parameter set | use a different target or `from_scene` |
| Aperture | `archival Gaia source(s) ... have no measured G magnitude ...` | a cone source cannot be assigned a TESS flux | inspect the retained Gaia response; the catalog scene cannot model it |
| Metallicity | `higher-priority metallicity query unavailable` or `SHBoost metallicity query unavailable` | archive or service outage, not “no value” | retry; the ladder does not silently drop a failed query |
| Metallicity | `target metallicity unavailable: ...` | all four catalogues completed without a measured target value | use `from_scene` with a measured value, or a target with one |
| Background | `TRILEGAL ...` raised | service or footprint failure | retry; inside the fallback footprint the run uses the recorded fallback, outside it the failure is final for that attempt |
| Binding | `observed sector weights must cover exactly the calibrated sectors` | missing or extra sector key | pass exactly the scene's sectors, with positive weights |
| `calc_probs` | `resolved neighbor ... can host the observed depth but has no stellar mass and radius` | a possible host lacks parameters | supply them through `from_scene`; do not remove the neighbour |

A missing neighbour parameter is not a reason to delete the neighbour. If the
source cannot host the depth, its slots are host-excluded; if it can, the
calculation refuses until mass and radius are supplied. Removing a possible host
removes its false-positive scenarios and biases FPP low.

## Network, time and cache

Acquisition needs internet access and can download large catalogues. The
default `NetworkPolicy` allows network access with bounded retries; a custom
policy can restrict it, but the stages that need the network then cannot run.
Requests require HTTPS, and TLS verification cannot be disabled.

MIST is fetched once into `cache_dir` and retained with its URL and retrieval
date. GALAH DR3 is cached the same way. A rerun reuses cached releases and any
stage whose retained record is available. Acquisition time depends on the
archives, the field and the number of sectors; no fixed estimate is promised
here.

## Related pages

- [Tutorial](tutorial.md): the light curve, noise calibration and probability
  call around this acquisition.
- [From a scene](from-scene.md): the offline alternative and measured contrast
  curves.
- [Background population](background-population.md): TRILEGAL and the fallback.
- [Inputs](inputs.md): what the acquired values must satisfy.
- [Scenarios](scenarios.md): which scenarios depend on which acquired values.
- [FAQ](faq.md): error messages and next steps.
