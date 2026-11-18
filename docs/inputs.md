# Inputs and units

This page lists what you have to supply: the phase-folded light curve and
measured transit parameters, the per-sector weights, the scene (calibrated
target, resolved neighbours, aperture flux shares and background population),
the contrast curve and the TRILEGAL file, with their units, accepted values and
the validation TREX applies. It is a reference; the [tutorial](tutorial.md)
shows the calls in order and [From a scene](from-scene.md) shows how to build a
scene by hand.

TREX never replaces a missing measurement with a default. The run either
refuses the input or records the affected scenarios as unavailable with the
reason. A plausible-looking substitute (solar metallicity, a guessed parallax,
a fabricated background) would change the probabilities without telling the
reader.

## 1. The light curve and measured parameters

`calc_probs` and the low-level `run_trex_vetting` take the same observations
under slightly different argument names:

| `calc_probs` / `run_trex_vetting` | Shape | Unit | Enforced rule |
|---|---|---|---|
| `time` | 1-D, at least 2 points | days from the transit midpoint | real, finite; same length as the flux array |
| `flux_0` / `flux` | same shape as `time` | normalized relative flux | real, finite |
| `flux_err_0` / `sigma` | scalar, or same shape as `flux` | normalized relative flux | real, finite, strictly positive |
| `P_orb` / `period_days` | scalar | days | finite, positive; a period interval is refused |
| `depth_ppm` | scalar | ppm | finite, positive; binding also requires `< 10⁶` |
| `exptime` / `exptime_days` | scalar | days | finite, positive; `calc_probs` also requires equality with the bound exposure |
| `bin_integration_days` | scalar, or one value per point | days | scalar `>= 0`; per-point values strictly positive |
| `N` / `n_draws` | integer | draws per scenario | positive; default 1,000,000 |
| `nsamples` | integer | dimensionless | positive; default 20 |

Boolean and text arrays are refused rather than converted. `admitted_photometry`
performs this check once and is shared by `calc_probs`, the doctor and the
engine.

`calc_probs` takes only the time series and the period directly. `sector_weights`,
`depth_ppm` and `bin_integration_days` are fixed at `bind_scene`, and `exptime`
must repeat the bound exposure; the low-level `run_trex_vetting` takes all of
them explicitly.

### Time

`time` is phase-relative: days measured from the transit midpoint, with the
conjunction at zero. A typical fold is

```python
phase = ((time_btjd - epoch_btjd + 0.5 * period_days) % period_days) - 0.5 * period_days
```

TREX does not test that the array is folded, centred, sorted, unique or limited
to a transit window, and it applies no outlier rejection: every point you pass
enters every scenario's likelihood. An absolute BTJD or BJD array is accepted
and used as if those numbers were phase offsets, so a run can finish with a
meaningless likelihood. Fold first and confirm that the dip sits at zero.

BJD and BTJD are offsets on the same physical timescale: `BTJD = BJD − 2457000`,
both counting in TDB. The offset changes no rate and cancels in a phase
difference computed from two times in the same convention. Mixing conventions
between the light curve and the epoch shifts the fold by the offset, modulo the
period, so use one convention consistently.

The only paths that use chronological times are the native-time noise calibration
(section 5) and the optional Gaussian baseline; with a baseline, `time` itself
must be the native TDB days.

### Flux and its uncertainty

`flux` is normalized relative flux; out of transit it should be approximately
1. TREX does not renormalize the curve, so any out-of-transit offset you leave
in is seen by every scenario.

`sigma` is one fixed uncertainty for the whole run: a scalar shared by every
point (the TRICERATOPS convention) or one propagated value per point. Calibrate
it once on the native, unfolded cadences (section 5) and pass that calibrated
value before folding. A per-point array preserves heteroscedastic weights; with
equal values it reproduces the scalar likelihood.

The likelihood is

```text
ln L = −0.5 Σ_i (F_obs,i − F_model,i)² / σ_i²
```

The Gaussian normalization term is omitted. In host coordinates the omitted
`N log(σ/f)` term cancels the `f**N` Jacobian of the flux renormalization, so
this reduced form is invariant under the per-host rescaling used by the
neighbour scenarios. It is not invariant under a change of the supplied `σ`:
multiplying every `σ` by a constant `c` divides every log-likelihood by `c²`
and therefore moves the normalized probabilities. The value is part of the
model, not a cosmetic choice, and there is no per-scenario noise fit.

### Depth and duration

`depth_ppm` is a measured input that TREX does not derive. It is used to

- decide which resolved neighbours could host the observed dip at all,
- centre the occultor-radius proposal window,
- set the depth that a resolved neighbour would have to produce.

It does not enter the likelihood as a data term; the light curve carries the
depth information. The same applies to the optional `depth_sigma_ppm`,
`duration_days` and `duration_sigma_days`: they shape the Monte Carlo proposal
windows (radius ratio and impact parameter) and change sampling efficiency, not
the target distribution or the prior.

`bind_scene` requires `0 < depth_ppm < 10⁶` and a positive exposure. Acquisition
(`target.from_tic`) applies the same bound with its own message. The engine's
pre-draw admission checks only that `depth_ppm` is finite and positive.

The scenario period prior is a broken power law normalized over 0.1–50 d and
evaluated at the applied model period (`P` or `2P`). A period outside that range
is not refused; the prior is clamped to the nearest edge bin, so such a signal is
modelled with the edge-bin prior rather than with a prior centred on its period.
See [Scenarios](scenarios.md).

### Exposure and bin width

`exptime` is the native detector exposure. `calc_probs` compares it to the
exposure bound by `from_scene`/`from_tic` and refuses a mismatch, so a default
value cannot silently stand in for a measured cadence.

`bin_integration_days` is the separate phase-bin averaging width. With the
scalar `0`, the model is averaged over `nsamples` exposure midpoints only. With
a positive bin width, the phase-bin box and the exposure box are convolved
analytically and `nsamples` does not change the model. Per-point widths let each
binned point integrate over its own span; a per-point width of zero is refused,
only the scalar form may be zero.

#### Explicit temporal response in scalar forward models

The lower-level `simulate_TP`, `simulate_EB`, `lnL_TP` and `lnL_EB` functions
also accept `exposure_response=trex.photometry.exposure.ExposureResponse(...)`.
This is not yet an option on the full `calc_probs`/engine workflow.

Supply `exptime_days=None` and `bin_integration_days=0` with this option:
neither the legacy exposure box nor another phase-bin box may be applied
again. `ExposureResponse.boxcar` prepares scalar or per-row widths, including
an explicitly instantaneous zero-width limit. `ExposureResponse.stacked`
prepares disjoint uniform live windows weighted by their durations; gaps
are not filled with a continuous cadence box. Choose and check its explicit
`nsamples` for the actual ingress/egress and precision requirement. The
forward model's legacy `nsamples` does not override an explicit response.

Raw response offsets are in TDB days relative to the supplied model stamps;
weights are finite, nonnegative and sum to one for each observation. A
detector ramp or nonuniform coadd needs its own qualified response. An
effective exposure or revisit cadence does not establish that response.

#### Clock conversion before reference rebasing

`trex.photometry.time_conversion.barycentric_tdb` accepts precise Astropy
`Time` values and an explicit correction state. Already-barycentric TDB/TCB
input receives only a clock-scale conversion, never another geometric
correction. Uncorrected ground arrival times require a finite scalar
`EarthLocation` and an ICRS direction already propagated to the observation
epoch. The distant-source correction uses Astropy's builtin ERFA ephemeris;
IERS/leap-second availability and warnings remain visible.

Convert individual temporal quadrature nodes when converting their
observer frame. A geometric correction is not a fixed offset or a change
to an effective exposure keyword. Ambiguous HJD/BJD_UTC conventions and
unqualified spacecraft state-vector origins/frames are unsupported.
`trex.photometry.time_axis` then performs split-JD reference rebasing without
collapsing absolute Julian dates to one float.

## 2. Sector weights

`sector_weights` maps each observed TESS sector to a positive weight:

```python
sector_weights={1: 3.0, 2: 1.0}
```

The weights are the sums of the inverse variances of the retained native
cadences in each sector:

```text
w_s = Σ_i 1 / σ_i²
```

TREX uses them to reduce every per-sector aperture flux fraction onto the same
multi-sector combination the folded light curve carries:

```text
f̄ = Σ_s w_s f_s / Σ_s w_s
```

Rules enforced at binding:

- the keys must be exactly the scene's sectors, compared as a set,
- each key must be an integer sector ID (a bool is refused),
- each weight must be finite and strictly positive.

The engine applies the same per-key checks, but only requires a non-empty mapping
and a fraction row for every sector you weight; `bind_scene` is what enforces
exact coverage. TREX cannot check that the weights describe the light curve you
pass. If they do not, the dilution applied to the model will not match the
dilution in the data.

## 3. The scene

A scene is a `trex.scene.TargetScene`. It can come from `target.from_tic`
(acquired), from `target.from_scene` (supplied), or from `TargetScene.from_dict`
(a validated export). The constructor validates every supplied field; no stellar,
aperture or background value is inferred.

### Target fields

All floating-point scene fields go through one finite-real check: a bool,
string, NaN or infinity is refused with `<field> must be a finite real scalar`
(plus `greater than zero` where the field must be positive).

| Field | Unit | Rule |
|---|---|---|
| `tic_id` | integer | positive; a bool is refused |
| `ra_deg` | degrees | finite, `[0, 360)` |
| `dec_deg` | degrees | finite, `[-90, 90]` |
| `M_s_Msun` | solar masses | finite, positive |
| `R_s_Rsun` | solar radii | finite, positive |
| `Teff_K` | kelvin | finite, positive |
| `Tmag` | TESS magnitude | finite |
| `plx_mas` | milliarcseconds | finite, positive |
| `feh_dex` | dex | finite, or `None`; a supplied value requires `metallicity_provenance` |
| `sectors` | list of sector numbers | positive integers; non-empty and duplicate-free at binding |
| `target_aperture_flux_fractions` | list of sector/fraction rows | non-empty; see below |

`TargetScene` accepts a `sectors` list of positive integers but does not reject
duplicates at construction; `bind_scene` refuses an empty or duplicated sector
list, because the weights could not be matched to it unambiguously. A scene
without `feh_dex` is constructible but every calculation refuses:

```text
scene.feh_dex is required to evaluate every false-positive scenario
```

The target metallicity sets the target's limb darkening and the atmospheres of
the eclipsing and companion-hosted scenarios; solar is not substituted.

The remaining constructor fields are the contrast arrays and their provenance
(section 4), `unresolved_resolution_limit_arcsec`, the background population
(below), `metallicity_provenance`, and two diagnostic records:
`localization_evidence` and `neighbor_parameter_evaluations`. The diagnostics are
retained for reporting only; the centroid record does not reweight or remove a
neighbour, and the neighbour log-prior is neutral.

### Metallicity provenance

A caller-supplied value must look like this:

```python
metallicity_provenance={
    "source": "caller-supplied",
    "quantity": "[Fe/H]",          # or "[M/H]"
    "value_dex": 0.02,
    "uncertainty_dex": 0.08,
    "citation": "the paper or catalogue this value came from",
}
```

Rules for this record:

- exactly these keys, plus `citation`, `uri`, or both; any other key is refused,
- `quantity` must be `[Fe/H]` or `[M/H]`,
- at least one of `citation` and `uri` must be a non-empty string,
- `value_dex` must equal `feh_dex` exactly,
- `uncertainty_dex` must be finite and positive.

The acquisition path writes richer records for four admitted catalogue sources:
`apogee-dr17`, `galah-dr3`, `gaia-gspspec-calibrated` and `shboost-xp`. Each
requires its own flags, retrieval time and https URI, and all but legacy SHBoost
records carry a raw-row binding; each source is validated against its own rules.
[From a TIC ID](from-tic.md) describes the selection ladder.

### Aperture flux fractions

```python
target_aperture_flux_fractions=[
    {"sector": 1, "fraction": 0.85},
    {"sector": 2, "fraction": 0.80},
]
```

Each fraction is the source's share of the total in-aperture flux in that
sector, in `[0, 1]`. Rows are `{"sector": int, "fraction": float}`; the sector
must be a positive integer, repeated sectors in the list are refused, and the
list must be non-empty. Extra rows for sectors you do not weight are accepted
and ignored; a missing row for a weighted sector is refused when the fractions
are reduced:

```text
target aperture scene does not cover observed sector 2
```

These are throughput measurements. A catalogue brightness ratio ignores the
pixel response function, the aperture shape and the position of the source in
the aperture, so it cannot stand in for them. `target.from_tic` derives them
from the pipeline aperture with a documented approximation; a custom scene must
supply its own measured or modelled values. [From a scene](from-scene.md) shows
the integration.

At binding, each source's per-sector rows are reduced with the observed sector
weights (section 2). The reduced target fraction must be positive, and the sum
of the reduced target and all reduced neighbour fractions must not exceed 1
beyond a roundoff allowance of `eps × (number of reduced fractions)`:

```text
calibrated aperture flux shares must be finite, non-overcounted and include the target
```

The overcount guard applies to the coherent joint scene, not to a
source-specific marginal limit.

### Resolved neighbours

Each entry is a dictionary:

```python
{
    "source_id": "100002",              # unique non-empty string
    "M_s": 0.82,                        # solar masses, or None
    "R_s": 0.79,                        # solar radii, or None
    "delta_mag": 1.5,                   # magnitudes relative to the target, or None
    "separation_arcsec": 12.0,          # finite, positive
    "aperture_flux_fractions": [
        {"sector": 1, "fraction": 0.1499},
        {"sector": 2, "fraction": 0.1999},
    ],
    # optional:
    "mass_solar_err_lower": 0.02,
    "mass_solar_err_upper": 0.03,
    "radius_solar_err_lower": 0.01,
    "radius_solar_err_upper": 0.02,
    "stellar_parameter_provenance": {"source": "..."},
    "metallicity_provenance": {...},
    "metallicity_unavailable_reason": "...",
    "stellar_atmosphere": {"source": "...", "teff_k": 5200.0, "mh_dex": 0.0, "av_mag": 0.1},
    "stellar_parameter_distribution": {...},
}
```

Rules:

- `source_id` values must be unique non-empty strings.
- `M_s` and `R_s` are supplied together, both finite and positive, or both
  `None`. The `None` form requires `stellar_parameter_unavailable_reason`.
- `delta_mag` is the neighbour's TESS magnitude relative to the target. It may
  be `None` only with a non-empty `delta_mag_unavailable_reason`; carrying both
  is refused. A neighbour's brightness enters the evidence through its measured
  aperture flux fraction, not through `delta_mag`, so an unmeasured magnitude
  contrast does not remove its host scenarios.
- `separation_arcsec` is required, finite and positive.
- `aperture_flux_fractions` must be a non-empty list following the rules above.
  A zero fraction keeps the entry but excludes its scenarios.
- The mass and radius uncertainties, when supplied, must be finite and
  non-negative. They are recorded but not marginalized: the evidence
  calculation uses fixed host values.
- `stellar_parameter_provenance` is optional and must name its source.
- A neighbour may carry its own `metallicity_provenance`, or a
  `metallicity_unavailable_reason`, never both. With neither, the target's
  `feh_dex` is used and the record states `target-metallicity fallback`. A
  neighbour with its own record must satisfy the same schema as the target,
  except that its value is the record's `value_dex`.
- `stellar_atmosphere` requires a non-empty `source` and finite `teff_k`
  (positive), `mh_dex` and `av_mag >= 0`. It is required for every resolved
  source when an explicit `stellar_light` model is bound.
- `stellar_parameter_distribution` is an optional characterized-host
  distribution, validated by `trex.evidence.host_measure.HostMeasure`.
- If `stellar_relations_calibration_status` is present it must be exactly
  `"uncalibrated"`.
- A residual-scene source may also carry
  `aperture_flux_fractions_upper_95` and `aperture_flux_fraction_limitation`:
  a source-specific marginal 95% fraction and the statement of its limitation.
  The host scenario then uses that marginal value while the coherent joint
  fraction stays in the scene.

A neighbour that can host the observed depth (positive aperture fraction and
`depth_ppm <= fraction × 10⁶`) but lacks mass or radius refuses the calculation
with `TrexEvidenceUnavailableError`; dropping a possible host would bias FPP
low. A neighbour that cannot host the depth keeps its three scenario slots with
`-inf` evidence and a recorded status (`host-excluded`); see
[Scenarios](scenarios.md).

### Unresolved companions

A scene must state how unresolved companions are bounded. Supply either

- a measured contrast curve (section 4), or
- `unresolved_resolution_limit_arcsec`: a finite positive separation in
  arcseconds. Sources inside it are treated as unresolved; sources beyond it are
  resolved neighbours of the aperture scene.

If a curve is supplied and no limit is given, the Gaia DR2 close-pair limit of
2.2 arcsec is used as the independent unresolved-scene bound. If neither is
supplied, `TargetScene` raises:

```text
a scene requires a measured contrast curve or the catalog resolution limit
```

When both are supplied, both apply: the curve constrains companions over its
measured separation range, and the limit bounds the population everywhere. The
limit is also the outer radius of the unresolved background-star field used by
the background-hosted scenarios. TREX does not invent a curve.

### Background population

| Field | Unit | Rule |
|---|---|---|
| `trilegal_cache` | path | existing regular file, not a symlink |
| `background_sha256` | hex string | optional; when supplied it must be 64 characters and match the file |
| `background_area_sq_deg` | square degrees | required, finite, positive |
| `N_background` | integer | non-negative |
| `background_population_provenance` | mapping | optional; a TRILEGAL LSST-simulation fallback record whose digest matches the file |

`trilegal_cache` points to a CSV with one row per simulated star. TREX requires
these columns, matched by name:

| Column | Meaning |
|---|---|
| `TESS` | apparent TESS magnitude, including distance and extinction |
| `Mact` | current mass, solar masses |
| `logg` | surface gravity, `log10(cm/s²)` |
| `logTe` | `log10(T_eff/K)` |
| `[M/H]` | simulated metallicity, `log10(Z/Z_sun)` dex; not a measured iron abundance |

Optional columns `Av` and `m-M0` are retained as `av_mag` and
`distance_modulus_mag`; `Av` must be non-negative. A leading unnamed index
column and a UTF-8 byte-order mark are accepted.

The parser skips empty lines and the accepted `#TRILEGAL` footer forms (a bare
marker or `normally terminated`); any other row must be numeric. It rejects
empty or duplicate column names, missing required columns, truncated rows,
non-finite cells and data after the footer. Column order does not matter.

Only stars at least as faint as the target are retained (`TESS >= Tmag`); a
brighter simulated star would dominate the aperture and is not the modelled
configuration. The radius is derived from mass and surface gravity,
`R = sqrt(GM/g)`, so an evolved simulated star keeps its size instead of being
forced onto a main sequence.

If no simulated star is at least as faint as the target, loading the population
raises:

```text
TRILEGAL population has no simulated star fainter than the target
```

That error surfaces when the run loads the population, not when the scene object
is constructed. A population with no valid background stars is not evidence that
the background is empty.

`background_sha256` is computed from the file when omitted and recorded exactly
as a supplied one would be. It is verified when the scene is constructed, again
whenever the population is parsed, at binding, and before results are
published, so a cache that changed after the scene was built is refused:

```text
trilegal_cache changed after scene construction
```

`background_area_sq_deg` is the simulated solid angle of the query that produced
the cache, not a search radius. It sets the surface density used by the
background-hosted scenarios. `N_background` is the recorded query `star_count`
(web: rows returned; fallback: converted stars with `TESS < 21`), before the
scene's `TESS >= Tmag` draw cut. It is provenance only: the parsed file supplies
the stars the evidence uses, and the two are not cross-checked.

If a `trilegal_query.json` sits beside the file and records the LSST-simulation
fallback, the scene records that projection and its limitation automatically.
[Background population](background-population.md) covers the file and the
fallback.

## 4. Contrast curves

A measured contrast curve is a two-column table:

```text
# separation [arcsec]   limiting delta magnitude [mag]
0.10                    1.0
0.50                    2.5
1.00                    3.0
2.00                    5.0
```

Accepted forms:

- a path to a two-column text file (the TRICERATOPS/ExoFOP layout),
- a `(separations, delta_magnitudes)` pair of arrays,
- a NumPy array of shape `(2, N)`.

A file must be an existing regular file, not a symlink. Blank lines, `#` and
`//` comments, and named header rows are skipped; a numeric row must contain
exactly two finite numbers.

Rules:

- at least two points,
- finite, strictly increasing, positive separations,
- no monotonicity requirement on the delta magnitudes; a non-monotonic measured
  curve is used as measured,
- the band must be the TESS band; the name is matched case-insensitively,
- a non-TESS curve is refused, because TREX evaluates companion contrasts in the
  TESS band and implements no verified bandpass conversion:

```text
TREX evaluates companion contrasts in the TESS band only; no verified bandpass conversion is implemented, so a non-TESS contrast curve is refused
```

The curve is read in its measured direction: for a separation, it gives the
faintest detectable delta magnitude. Below its smallest and above its largest
measured separation it provides no constraint, and those draws remain in the
prior. Where the curve provides no constraint, the unresolved population is
still bounded by `unresolved_resolution_limit_arcsec`; the curve adds a
detectability constraint only over its measured range. A `-inf` background
evidence at a small draw budget can therefore describe only the covered
separations; it does not show that the support is empty.

Pass the curve when binding:

```python
star = target.from_scene(
    scene, contrast_curve="curve.txt", contrast_band="TESS",
    sector_weights=..., depth_ppm=..., exptime_days=..., ...)

star = target.from_tic(
    tic_id, sectors, output_dir=..., depth_ppm=...,
    contrast_curve=(sep, dmag), contrast_band="TESS",
    sector_weights=..., exptime_days=..., ...)
```

`calc_probs(contrast_curve_file=...)` raises: the curve is part of the scene so
that its provenance and band travel with the result.

Through `from_scene`/`from_tic`, TREX writes `contrast_provenance` with
`kind: "measured-contrast-curve"`, the source, the TESS band and a limitation
string. A scene built directly may omit that record; when it is present, all
four fields are required.

## 5. What the uncertainty calibration needs

`calibrate_trex_noise` measures one shared effective-error factor from native,
unfolded photometry. The library does not call it for you: you calibrate, then
multiply `sigma` by the recorded factor before folding.

```python
from trex.models.noise import calibrate_trex_noise

noise = calibrate_trex_noise(
    time_btjd, flux, sigma, sectors,
    period_days=P, epoch_btjd=t0, duration_days=T14,
    cadence_days_by_sector={1: 120.0 / 86400.0, 2: 120.0 / 86400.0},
)
sigma_calibrated = sigma * noise["native_uncertainty_factor"]
```

| Argument | Unit | Rule |
|---|---|---|
| `time_btjd` | BTJD_TDB | 1-D, non-empty, finite |
| `flux` | normalized | same shape as `time`, finite |
| `sigma` | normalized | scalar or same shape, finite positive |
| `sectors` | integer array | same shape as `time`, positive |
| `period_days` | days | finite, positive |
| `epoch_btjd` | BTJD | finite |
| `duration_days` | days | finite, `0 < T14 < P/2` |
| `cadence_days_by_sector` | mapping sector to days | one finite positive cadence per retained sector, from the FITS headers |
| `event_windows` | mapping sequence or `None` | optional declared eclipse phases/durations; replaces the default circular secondary |

The 2,457,000-day BJD/BTJD offset cancels in the phase arithmetic, so the array
and epoch may be BJD instead, as long as both use the same convention. The
retained record states `native-BTJD_TDB-before-phase-folding`.

Optional `event_windows` is a sequence of mappings
`{"phase": fraction in [0, 1), "duration_days": days, "role": label}`.
An omitted duration defaults to T14; the role must be non-empty. Declared windows
replace the default circular secondary, never the primary. Invalid windows raise
`ValueError`.

By default, each sector excludes the primary and a **circular** phase-0.5
secondary of the primary duration, each padded by half a cadence:

```text
selection = exclude primary and circular half-phase windows within (T14 + cadence) / 2
```

An eccentric secondary elsewhere remains in calibration unless the caller
declares its measured phase and duration. The record's `event_windows`,
`secondary_mask` and `event_phase_support` identify the windows and assumption
used. This exclusion is calibration-only: all supplied points remain in the
likelihood.

The estimator bins the standardized out-of-transit
residuals at `N = T14 / cadence` points, rounded to the nearest integer with
ties to even, and breaks bins at gaps larger than three cadences. It computes

```text
beta_raw = σ_binned / (σ_unbinned / √N)
beta     = max(1, beta_raw)
factor   = max(1, standardized unbinned scatter) × beta
```

The run uses the largest sector factor, so a sector noisier than its reported
`sigma` is inflated even when `beta` is 1. This is a fixed policy envelope
applied once, not a fitted hyperparameter and not a per-scenario scale.

Calibration defers rather than falling back to `beta = 1`. A sector is deferred
when its cadence is missing or invalid, when `T14 < 2 cadences`, when there are
fewer than 20 out-of-transit samples, when fewer than four complete bins remain,
or when the residual scatter is undefined or non-finite. The call then raises:

```text
TREX noise calibration deferred; inspect per-sector diagnostics
```

and the diagnostics retain each sector's status, cadence, bin size, counts and
reason. [API](api.md#noise-calibration) lists the signature.

## 6. Optional advanced inputs

`bind_scene` and the engine also accept:

- `random_seed`: non-negative integer or `None`. When `None`, `calc_probs`
  generates a seed and stores it in the provenance; a resumed escalation must
  reuse the recorded seed.
- `input_artifacts`: mapping of retained input paths to SHA-256 digests. TREX
  verifies them at binding and again before publishing results. When omitted, it
  hashes the background population and records that. A hash proves identity, not
  physical calibration.
- `baseline`: an explicit `trex.photometry.baseline.GaussianBaseline` on native
  chronological TDB days from a single conjunction. The times must be strictly
  increasing with at least two cadences, the design must be evaluated at every
  retained cadence, and the coefficient prior must be a proper Gaussian
  supplied by the caller. It requires dynesty, a single-sector scene, and the
  exact unbinned native-time observations; it is unavailable with the BlackJAX
  estimator.
- `stellar_light`: an explicit physical stellar-light model with the target's
  own `A_V`. It replaces the default blackbody surface-brightness construction
  and requires per-source atmosphere inputs (`teff_k`, `mh_dex`, `av_mag`,
  `source`) for resolved neighbours. It cannot be mixed with explicit `u1`/`u2`
  overrides, and a missing target `A_V` refuses rather than defaulting. Without
  it, the historical physics is retained.
- `depth_sigma_ppm`, `duration_days`, `duration_sigma_days`: measured values
  that shape the proposal windows. They are optional and do not enter the
  likelihood. Bind them only when measured.

`run_trex_vetting` (and `prepare_vetting_inputs`) additionally take optional
`u1` and `u2` quadratic limb-darkening coefficients for the target. Supply both
or neither. When omitted, TREX selects the target's
temperature–gravity–metallicity node from the packaged TRICERATOPS TESS grid and
records the requested and used values, including whether the request was clamped
to a grid boundary. There is no solar fallback. Explicit coefficients cannot be
mixed with `stellar_light`.

These paths are validated strictly and are easy to misconfigure.
[API](api.md) lists the signatures; the source docstrings state the required
provenance.

## 7. Missing inputs

TREX distinguishes three input outcomes:

- **Modelled.** The scenario is evaluated; its log-evidence and effective sample
  size are recorded.
- **Unavailable.** A required model input is missing, so the scenario has no
  model. Its slot stays at `-inf` evidence with status `unavailable` and the
  reason. The engine treats the evidence set as incomplete: it raises
  `TrexEvidenceUnavailableError` and publishes no FPP/NFPP or `scenarios.csv`.
  It is not silently assigned a zero probability; a missing scenario is not an
  impossible one.
- **Refused.** The input cannot support the run at all, and the calculation
  raises before any draw. This is the case for missing target metallicity,
  missing sector weights, missing background area, a malformed scene field, a
  changed input artifact, an unreadable population, or a neighbour that can host
  the depth but has no mass and radius.

A separate physical exclusion is the neighbour that cannot host the observed
depth given its measured aperture share: its slots are recorded `host-excluded`
with `-inf` evidence. That is a statement about the host's light, not proof that
the neighbour does not exist. [Outputs](outputs.md) lists where each status is
written, and [Concepts](concepts.md) explains why unavailable scenarios block a
normalized result rather than becoming zero-probability evidence.

`target.doctor(...)` runs the same pre-draw admission without any evidence draw
and returns a report instead of raising, so these refusals can be diagnosed
before the expensive run.

## 8. Validation errors you will see

| Message | Raised when |
|---|---|
| `photometry requires finite aligned one-dimensional phase-relative times [days] and normalized flux` | `time`/`flux` not 1-D, fewer than 2 points, mismatched shapes, NaN or infinity |
| `flux uncertainty must be finite and positive` | scalar or per-point `sigma` is zero, negative, NaN or infinite |
| `per-point flux uncertainty must match the flux array shape` | array-valued `sigma` with the wrong shape |
| `n_draws must be a positive integer` | also `nsamples`; a bool is refused |
| `period_days must be finite and positive` | engine admission; also `depth_ppm` and `exptime_days` |
| `observed per-sector inverse-variance weights are required to apply the calibrated aperture scene` | missing or empty `sector_weights` |
| `observed sector weights must cover exactly the calibrated sectors` | binding: keys differ from the scene sectors, or the scene sector list is empty or duplicated |
| `calibrated aperture flux shares must be finite, non-overcounted and include the target` | binding: reduced target fraction is not positive, or the reduced sum exceeds 1 beyond roundoff |
| `scene.feh_dex is required to evaluate every false-positive scenario` | target metallicity is `None` |
| `target metallicity requires measured-source provenance` | `feh_dex` supplied without `metallicity_provenance` |
| `caller metallicity provenance is incomplete or invalid` | wrong keys, quantity, missing citation/URI, or `value_dex` not equal to `feh_dex` |
| `a scene requires a measured contrast curve or the catalog resolution limit` | neither a curve nor `unresolved_resolution_limit_arcsec` |
| `TREX evaluates companion contrasts in the TESS band only; ...` | `contrast_band` is not TESS |
| `contrast curve requires matching separation and contrast arrays` | fewer than two points or mismatched vectors |
| `contrast curve separations must be positive and strictly increasing` | separations out of order, duplicated, or non-positive |
| `exptime must equal the bound native exposure [days]; do not use a guessed default` | `calc_probs` exposure differs from the binding |
| `positive depth/exposure and non-negative bin width are required` | binding: depth outside `(0, 10⁶)`, exposure `<= 0`, or a negative bin width |
| `depth_ppm must be finite in (0, 1e6)` | `target.from_tic` acquisition domain |
| `bound input artifact is missing or changed: <path>` | an `input_artifacts` file moved or was edited |
| `TRILEGAL population cache lacks required columns: <names>` | missing `TESS`, `Mact`, `logg`, `logTe` or `[M/H]` |
| `resolved neighbors must have unique source_id values` | duplicated or empty neighbour ID |
| `resolved neighbor N without stellar parameters must state why` | both `M_s` and `R_s` are `None` without a reason |
| `resolved neighbor N can host the observed depth but has no stellar mass and radius: ...` | a possible host lacks `M_s`/`R_s` (`TrexEvidenceUnavailableError`) |
| `TREX noise calibration deferred; inspect per-sector diagnostics` | missing cadence, insufficient native resolution or too few out-of-transit bins |
| `trilegal_cache changed after scene construction` | re-verification: the population no longer matches `background_sha256` (construction uses `trilegal_cache does not match background_sha256`) |

Errors raised by the engine's pre-draw admission are tagged `input-admission`;
an exposure mismatch and a changed binding are tagged `target-binding`. Errors
raised by `TargetScene` or `bind_scene` themselves carry no tag. A refused input
is not a scientific result and not a convergence failure.
[FAQ and troubleshooting](faq.md) gives the remedy for each case.

## 9. Next

- [Outputs](outputs.md): the files these inputs produce.
- [From a TIC ID](from-tic.md): how the acquisition fills the scene fields.
- [From a scene](from-scene.md): the offline path.
- [Prepared runs](prepared-runs.md): the same contract as three input files.
- [Background population](background-population.md): the TRILEGAL file and its
  fallback.
- [Concepts](concepts.md): the units table and what the probabilities mean.
- [Tutorial](tutorial.md): the calls in order.
