# How TREX works

This page answers the question "what is the program actually computing?" It
follows one run from the inputs to the reported probabilities, explains why each
step exists, and states the assumptions each step makes. Read
[Concepts](concepts.md) first if false-positive probability (FPP), evidence or a
prior is still unfamiliar.

The method is the scenario framework of Giacalone et al. (2021), *AJ* 161, 24,
<https://doi.org/10.3847/1538-3881/abc6af>. The implementation is TREX's own:
its priors, forward models, evidence estimator, convergence routing and scene
acquisition differ in documented places from the reference implementation. This
page describes what TREX does, not what another code does. The deliberate
differences from TRICERATOPS are listed in
[How TREX differs from TRICERATOPS](differences-from-triceratops.md).

## 1. The shape of the calculation

```text
folded light curve + fixed sigma + stellar scene + measured period, depth, exposure
        │
        ├─ per scenario:
        │     draw N systems from the scenario priors
        │       (or from a defensive proposal centred on the measurement)
        │     predict each draw's light curve through the aperture
        │     compute each draw's log-likelihood
        │     add the draw's occurrence and proposal weights
        │     log-mean-exp over draws → Monte Carlo lnZ and ESS
        │
        ├─ if the production-budget Monte Carlo set fails the convergence routing:
        │     stage 1: MC-pilot defensive importance sampling
        │     stage 2: nested sampling (dynesty default; BlackJAX opt-in)
        │
        ├─ add the period factor once to every scenario
        ├─ normalize in log space → P_i
        ├─ FPP = Σ P_i over every scenario except TP, PTP, DTP
        ├─ NFPP = Σ P_i over the resolved-neighbour slots
        └─ route: publish only if the convergence gate passes; otherwise defer
```

The order matters. The likelihood knows nothing about the period factor; the
period factor is applied once at normalization. Population weights such as the
companion occurrence rate or the expected number of background stars enter
inside the evidence, because they are part of each draw's prior weight (and a
draw excluded by an imaging or stability mask carries zero weight). This keeps
each stored evidence prior-free with respect to the period factor, so a refined
or resumed run cannot apply that factor twice.

| Stage | What it does | Estimator and control | Output |
|---|---|---|---|
| Monte Carlo | prior draws, forward model, likelihood, occurrence weights | log-mean-exp; Kish ESS | `lnZ_MC`, ESS per scenario |
| MC-pilot importance | reweight the high-weight region with a kernel proposal | 1,000,000 draws; 10% defensive prior; ESS floor and 0.1-nat bar | refined `lnZ` for accepted scenarios |
| Nested sampling | integrate the same prior and likelihood with live points | dynesty 400 live, 0.05-nat tolerance; ensemble of three and 0.1-nat bar | substituted `lnZ` and error |
| Normalization | add the period factor once, normalize in log space | log-sum-exp | `P_i`, `FPP`, `NFPP` |
| Routing | check the convergence gate; the FPP/NFPP thresholds are display only | ESS floor and 10⁻³ mass allowance | publish, or defer with diagnostics |

## 2. What one draw contains

A draw is one simulated system for one scenario. Its parameters depend on what
the scenario claims is happening; the distributions below supply the shape
parameters for the relevant family. The scenario index order, event host and
availability conditions are in [Scenarios](scenarios.md).

### Shape priors used by the scenario families

| Quantity | Prior or value | Unit and source |
|---|---|---|
| Planet radius `R_p` | `dN/dR ∝ R^0` for `0.5 ≤ R ≤ 3`, `R^-4` for `3 < R ≤ 6`, `R^-0.5` for `6 < R ≤ 20` when `M_* > 0.63 M_sun`; `R^0`, `R^-7`, `R^-0.5` for the same ranges when `M_* ≤ 0.63 M_sun` | `R_earth`; Giacalone et al. (2021) Section 2.4.2, Eqs. 10–11; normalized over 0.5–20 |
| Inclination `i` | uniform in `cos i` inside the band that can transit | degrees; isotropic |
| Planet eccentricity `e` | Kipping (2013) Beta(0.867, 3.030) | dimensionless |
| Binary eccentricity `e` | `p(e) ∝ e^-0.8` for `P ≤ 10 d`; for `P > 10 d`, `p(e) ∝ e^η` with `η = 0.6 − 0.7/(log10(P/day) − 0.5)`, capped at `log10(P/day) = 6` | Moe & Di Stefano (2017) Eq. 17 |
| Argument of periastron `ω` | uniform 0–360 | degrees |
| Eclipsing-pair mass ratio `q = M_2/M_1` | `p(q) ∝ q^0.3` below `q = 0.3`, `q^-0.5` above, plus a twin excess above `q = 0.95` with `F_twin = 0.30` | Moe & Di Stefano (2017); solar-type short-period calibration |
| Bound-companion mass ratio `q` | the same three-segment form, with the large-q slope steepened by −0.3 per dex above `log10(P/day) = 5`, and `F_twin = 0.30 − 0.15 log10(M_1/M_sun)` tapering from `log10(P/day) = 1` to a mass-dependent cutoff | Moe & Di Stefano (2017) Eqs. 5–9 |
| Stellar mass-ratio floor | `max(0.1, 0.1/M_1)`, so the secondary is at least 0.1 `M_sun` | mass ratio; per draw |

The 0.63 `M_sun` radius-prior boundary is a TREX modelling choice. Giacalone et
al. use 0.63 `M_sun` for their stellar-property relations (Section 2.3), but the
paper gives no numeric mass boundary for the radius prior itself; adopting that
threshold is not a published occurrence cutoff.

The 0.60–0.80 `M_sun` bound-companion band uses the Moe & Di Stefano branch
below its 0.8 `M_sun` calibration limit. That is a documented extrapolation, not
a calibrated interpolation. Primaries above 40 `M_sun` are refused rather than
extrapolated.

The planet radius prior is applied to every planet scenario, including the
diluted and background-hosted ones, as the reference framework does. The
distribution is normalized over 0.5–20 Earth radii. Eclipsing-pair radii are
not drawn from this law; they follow the mass ratio and the stellar
mass–radius–temperature relation.

### Target-hosted planet (`TP`)

The planet radius follows the broken power law above. The orbital period is
fixed at the measured period. Eccentricity is Kipping (2013) Beta(0.867, 3.030),
`ω` is uniform, and inclination is isotropic inside the transiting band. The
target's mass, radius and limb darkening are fixed scene values.

Inclination is uniform in `cos i`, but only inside the band that can transit.
The condition is

```text
b_tra = (a/R_*) · cos i · (1 − e²)/(1 + e sin ω) ≤ 1 + k
```

where `a` is the semi-major axis, `R_*` the host radius, and `k = R_p/R_*` or
`R_EB/R_*` the radius ratio. The factor `(1 − e²)/(1 + e sin ω)` converts the
circular impact parameter to the star–occultor separation at conjunction (Winn
2010, *Transits and Occultations*, eq. 7). A draw whose periastron separation
`a(1 − e)` is smaller than the summed radii is rejected as an overlapping orbit,
not counted as a non-transit.

The semi-major axis follows Kepler's third law:

```text
a³ = G (M_* + M_body) P² / (4π²)
```

with `a` in cm, `P` in seconds, and masses in grams inside the code. For a
planet, the planet mass is neglected and `M_body = 0`; for a binary, the total
mass includes the secondary.

### Eclipsing binary on the target (`EB`, `EBx2P`)

The secondary mass comes from a mass ratio `q = M_2/M_1` drawn from the broken
power law and twin excess of Moe & Di Stefano (2017). The eclipsing pair uses
their short-period law with `p(q) ∝ q^0.3` below `q = 0.3`, `p(q) ∝ q^-0.5`
above, and a twin excess above `q = 0.95` with short-period fraction 0.30 at
1 `M_sun`. The support floor is resolved against each draw's own host mass, so
the secondary cannot fall below the stellar mass–radius domain (0.1 `M_sun`).
The secondary radius and temperature come from the same
mass–radius–temperature relations used for bound companions.

The eclipsing pair's TESS-band flux ratio is built from the radii and the
surface-brightness ratio:

```text
r = F_2/F_1 = (R_2/R_1)² · S(T_2)/S(T_1)
```

where `S(T)` is the TESS-band photon surface brightness of a blackbody at
temperature `T` (by default; an optional physical stellar-light model replaces
it with a band-integrated atmosphere). The code carries `r` as a
secondary-to-primary ratio; the equivalent fraction relative to the pair is
`f_EB = r/(1 + r)`. The pair's light is then combined with the target and any
third source according to the scenario.

`EBx2P` uses a model period of `2P` and evaluates both eclipses at the observed
times. Physically it represents a pair whose true period is twice the period of
the observed dips, so only one of the two eclipses appears in the folded signal.
The `x2P` label applies to every binary family: `EBx2P`, `PEBx2P`, `SEBx2P`,
`DEBx2P`, `BEBx2P`, `NEBx2P`.

The eccentricity density for binaries is `p(e) ∝ e^-0.8` for `P ≤ 10 d` (Moe &
Di Stefano 2017); TREX samples it through the cumulative `e^0.2`. For `P > 10 d`
it uses their Eq. 17 period-dependent power law. The `EB`/`EBx2P` split at
`q = 0.95` is retained: `EB` draws have `q < 0.95`, `EBx2P` draws have
`q ≥ 0.95`.

### Bound companions (`PTP`, `PEB`, `STP`, `SEB` and their `x2P` variants)

An unresolved bound companion is drawn from a stellar multiplicity model:

- For primaries above 0.60 `M_sun`, the companion period distribution is the
  Moe & Di Stefano (2017) Eq. 23 cumulative frequency over
  `0.2 ≤ log10(P/day) ≤ 8.0`, with period-dependent mass-ratio slope and twin
  fraction. The companion frequency is corrected for the drawn mass-ratio
  support and capped at one companion per primary.
- For M-dwarf primaries (`0.075 ≤ M_* ≤ 0.60 M_sun`), the semi-major axis is
  drawn from a log-normal whose mean and variance are moment-matched to the
  Winters et al. (2019) projected-separation fit (peak 20 AU,
  `σ_log s = 1.16 dex`), with the isotropic projection variance removed; the
  companion rate is 0.324 per primary before the stellar mass-ratio support
  correction. The mass ratio follows the Reggiani & Meyer (2013) `q^0.25`
  field-binary law. Orbits are taken as circular, and the sky-projected
  separation is then drawn from `s/a = √(1 − u²)` with `u` uniform in [0, 1).
- The drawn companion's projected separation is compared with the measured
  contrast curve, where the curve covers that separation, and with the
  unresolved-scene bound (2.2 arcsec, the Gaia DR2 close-pair limit, when the
  scene does not set a different limit).

A companion the imaging would have detected, one wider than the unresolved
limit, or one dynamically unstable against the observed inner orbit gets zero
prior weight. Stability uses the Mardling & Aarseth (2001) Eq. 90 criterion
instead of fixed period cuts. TREX assumes a circular, coplanar outer orbit
(`e_out = 0`, mutual inclination `i = 0`) in that criterion; the general form
also has an outer-eccentricity and inclination dependence, and G21 assumes
circular orbits but not zero mutual inclination. This is a documented modelling
choice, not the least restrictive form of the criterion.

The scenario families differ in which body hosts the event:

- `PTP`, `PEB`: the event is on the target; the bound companion only adds
  dilution to the aperture light.
- `STP`, `SEB`: the event is on the bound companion; the companion's own mass,
  radius and temperature set the geometry and the limb darkening.

### Background stars (`DTP`, `DEB`, `BTP`, `BEB` and `x2P` variants)

Background scenarios draw a star from the retained TRILEGAL population for the
field. Every retained star is at least as faint as the target; a brighter
unresolved star would dominate the aperture and is not the modelled
configuration. Each draw picks a star, computes its flux ratio from the TESS
magnitude difference,

```text
F_bg/F_target = 10^(-0.4 (Tmag_bg − Tmag_target))
f_bg = F_bg / (1 + F_bg)
```

and places it at a separation drawn uniformly in area out to the unresolved
limit:

```text
separation = r_limit · √u,    u uniform in [0, 1)
```

The expected number of such stars inside the field is the population's surface
density times the field area:

```text
λ_bg = N_stars / (A_sim · 1.296×10⁷ arcsec² deg⁻²) · π r_limit²
```

with `A_sim` the simulated sky area of the TRILEGAL query in square degrees, the
conversion factor turning it into square arcseconds, and `r_limit` in
arcseconds. That expected count is the draw's occurrence weight, inside the
evidence. See [Background population](background-population.md) for the query and
the population file.

`DTP` and `DEB` place the event on the target and let the background star dilute
the light. `BTP` and `BEB` place the event on the background star, using that
star's own mass, radius, temperature and TRILEGAL metallicity.

The background population does not carry sky positions in the retained cache.
TREX therefore cannot cross-match individual simulated stars against Gaia
neighbours inside the unresolved boundary. That overlap remains a model
limitation rather than being hidden by treating the imaging edge as a catalogue
resolution boundary.

### Resolved neighbours (`NTP`, `NEB`, `NEBx2P`)

For each resolved neighbour that can host the observed depth, TREX evaluates the
same three scenario types on that neighbour. A neighbour is skipped when:

- its aperture flux fraction is zero,
- the observed depth is larger than the neighbour's entire aperture flux
  fraction (an eclipse deeper than the star's contribution is impossible), or
- no finite metallicity is available for the reference-grid path.

A neighbour that could host the depth but has no mass and radius raises
`TrexEvidenceUnavailableError`. TREX does not drop it, because dropping a
possible host biases FPP low. A neighbour skipped for zero flux or an impossible
depth is recorded as `host-excluded`: its three slots stay at `-inf` evidence,
which is a physical exclusion and can still publish. A neighbour skipped for
missing metallicity is `unavailable`, which defers the run.

## 3. The forward model

The predicted light curve is a Mandel & Agol (2002) transit evaluated with
quadratic limb darkening, implemented by `batman` (Kreidberg 2015) on the CPU
reference path and by an equivalent batched JAX kernel where available.

Two integrations are applied:

- **Exposure time.** Each model point is averaged over the exposure, using
  `nsamples` sub-exposure midpoints at
  `exptime · ((i + 0.5)/n − 0.5)`. This is the PyTransit 2.2 midpoint rule,
  the engine TRICERATOPS uses; `nsamples` defaults to 20.
- **Phase bin.** `bin_integration_days` is a width in days. A scalar `0` keeps
  the exposure-only path above. A nonzero scalar, or one width per time point,
  convolves the exposure and bin boxes analytically with the exact trapezoid
  kernel and evaluates it at Gauss–Legendre nodes; per-point widths must be
  strictly positive and match the photometry length. A positive bin width
  therefore carries the exposure box inside the kernel and `nsamples` no longer
  changes the result; `nsamples` still sets the sub-exposure sampling of the
  unbinned path. A per-point bin width gets its own quadrature, so a broad
  baseline bin is integrated over its own span rather than assumed narrow.

For an eclipsing binary, both eclipses are evaluated at the observed times. The
secondary is placed at its eccentric phase, computed from Kepler's equation
under the edge-on conjunction approximation (true anomalies `π/2 − ω` and
`3π/2 − ω`), and it uses its own limb-darkening coefficients, not the primary's.
The model conserves absolute flux across the host, the eclipsing secondary, and
any third source. With `r = F_2/F_1`, `c` the unresolved companion's share
`F_comp/(F_comp + F_target)`, and `h = c` if the companion hosts the pair and
`h = 1 − c` otherwise,

```text
F_model = [ h·F_1 + r h·F_2 + (1 − h) ] / (1 + r h)
```

where `F_1` and `F_2` are each body's normalized shape (unity outside its own
eclipse). A bare pair is the `c = 0`, `h = 1` case; a diluted planet is the
`r = 0` case.

The transit shape depends on the impact parameter, the radius ratio and limb
darkening. A central transit of a small body is approximately a flat-bottomed
box of fractional depth `k²`; limb darkening rounds the bottom, and a grazing
transit (`b` close to `1 + k`) becomes V-shaped. TREX does not fit these shapes
by eye. It computes them for every draw and compares them point by point.

### Limb darkening

Quadratic limb darkening has the form

```text
I(μ)/I(1) = 1 − u1 (1 − μ) − u2 (1 − μ)²
```

where `μ` is the cosine of the angle between the line of sight and the surface
normal. TREX selects `u1`, `u2` from the host's effective temperature, surface
gravity and metallicity:

- Inside the packaged 3500–10000 K, 3.5–5.0 dex reference grid, it uses the
  nearest 250 K / 0.5 dex node of the vendored TRICERATOPS 1.0.20 TESS table.
  A request that falls back to the nearest node with a boundary clamp is
  recorded.
- Outside those bounds, a host inside a complete published four-node cell of
  Claret (2017) table 25 is interpolated bilinearly in `T_eff` and `log g` at
  the nearest metallicity plane. Below 3500 K at solar metallicity, complete
  PHOENIX-COND cells from Claret (2017) table 15 are used where available.
- Anything else keeps the nearest-node rule with its clamp. Nothing is
  extrapolated, and metallicity is not interpolated.
- Forty-nine metal-poor nodes absent from the packaged table take flux-conserving
  quadratic coefficients from the Kostogryz et al. (2022) MPS-ATLAS supplement
  (not Claret least-squares coefficients). No existing node is changed.

The provenance is aggregated per modelled host in the result: requested and
used temperature, gravity and metallicity, whether the value was clamped or
interpolated, and the source table. The eclipsed secondary uses its own
coefficients from the same selection rule.

These coefficient tables are TESS-band. TREX also bundles the PlatoSim3
N-CAM and F-CAM optical-transmission curves (filter and instrument only; no
detector quantum efficiency), but PLATO limb darkening is not validated: the
predeclared comparison of TREX PHOENIX intensities through the packaged N-CAM
curve with the official Morello et al. (2022, RNAAS 6, 248,
<https://doi.org/10.5281/zenodo.7339706>) release fails its declared
tolerances. TREX therefore claims no PLATO limb-darkening support; the same
comparison must pass before any PLATO coefficient is used or quoted.

## 4. Aperture flux, dilution and the renormalized light curve

A TESS aperture sums the pixels of several stars. Let `f` be the fraction of the
in-aperture flux that comes from one source. For a light curve normalized to
unity out of transit, the flux that source alone would produce is

```text
F_host = (F_observed − (1 − f)) / f
σ_host = σ_observed / f
```

This is the `renorm_flux_to_host` step of Giacalone et al. (2021). It removes
the constant light of every other resolved source and rescales the dip to the
host's own flux level. TREX evaluates each candidate host on its own
renormalized curve, which is why the aperture fractions are required inputs and
why a bare catalogue magnitude ratio is not acceptable: the fraction is an
aperture/PSF-weighted throughput, not a brightness ratio.

The Gaussian likelihood uses the reduced `−χ²/2`. Renormalizing to a host with
aperture fraction `f` changes two things: the errors become `σ_host = σ/f`, so
the Gaussian normalization term in the log density,
`−Σ_i log σ_host,i = −Σ_i log(σ_i/f)`, gains `N log f` relative to
`−Σ_i log σ_i`; and mapping a host-frame density back to the aperture data
contributes the inverse Jacobian `f^(−N)`. The two `N log f` terms cancel. The
reduced chi-square is therefore invariant under renormalization, including with
per-point uncertainties. The cancellation is per host; it is not an assumption
that the omitted constant is identical across hosts.

For an unresolved companion or background star, the observed dip is diluted:

```text
F_observed = (F_event + F_dilute) / (1 + F_dilute)
```

`F_dilute` is the companion's flux relative to the event host. With
`c = F_comp/(F_comp + F_target)` supplied as the companion's share of the
combined light,

```text
F_comp/F_target = c / (1 − c)
```

and the dilution is that ratio when the target hosts the event, or its
reciprocal when the companion hosts the event. The model therefore predicts a
shallower dip when the event is diluted, which is exactly what a blend
produces.

Sector combination: the phase-folded curve is a combination of the retained
sectors. TREX reduces each source's per-sector aperture fractions with the
observed inverse-variance sector weights, so the dilution applied to the model
is the same weighted mean the light curve carries:

```text
f̄ = Σ_s w_s f_s / Σ_s w_s
```

`w_s` is the sum of inverse variances of that sector's native cadences. This is
the aperture combination only; the flux uncertainty is a separate, fixed
native-time calibration (Section 5).

## 5. The likelihood

The likelihood consumes the measured inputs described in [Inputs](inputs.md).
For each draw `θ` and each scenario, TREX predicts a normalized flux
`F_model,i(θ)` at every observed phase and computes

```text
ln L(θ) = −0.5 Σ_i (F_obs,i − F_model,i(θ))² / σ_i²
```

`σ_i` is the fixed, calibrated uncertainty, a scalar or one value per point.
The production `σ` is calibrated once on the native, unfolded cadence:
`calibrate_trex_noise` measures, per retained sector,
`max(1, std(z)) × max(1, β)` with `z = (flux − out-of-transit median)/σ` and
`β` the Pont et al. (2006) binned-RMS factor at `T14/cadence`. The caller
multiplies the maximum of that per-sector factor over the retained sectors into
`σ` once, and every scenario uses the result. A sector that cannot support the
estimate defers; TREX never falls back to `β = 1`.
The default calibration mask assumes a circular phase-0.5 secondary; declare
a measured eccentric secondary through `event_windows`.

The sum runs over every point supplied. TREX does not clip or reweight points
inside the likelihood: out-of-transit selection belongs to the noise
calibration, and the Gaussian sum uses the folded data as given. A draw that
cannot transit is removed by prior support (log weight `−∞`), not by giving it
a zero-likelihood data point.

Every scenario uses the same input `σ` and the same photometry. A resolved host
is evaluated after `renorm_flux_to_host`, with `σ_host = σ/f`; the reduced
chi-square is invariant under that transformation, also for per-point errors.
The host-frame Gaussian normalization is not identical across hosts: it gains
`N log f` relative to the aperture frame, and the renormalization Jacobian
cancels that gain per host (Section 4). What remains is the same reduced
chi-square for every host. This is why a scenario-specific noise scale is not
supported: it would change the comparison between scenarios, not just the shape
of one model.

An optional **native-time baseline likelihood** can be bound to the scene. It
evaluates the exact normalized Gaussian density of the unbinned chronological
data after integrating a caller-supplied Gaussian prior on linear baseline
coefficients, additive or multiplicative (Gibson 2014). The prepared aperture
observations and errors are authoritative; the host-frame model is mapped back
to aperture coordinates once. It is one likelihood with a nuisance trend
marginalized, not a second fitted noise model, and it is available only on the
single-sector unbinned path.

Numerical failures are not converted into zero likelihood. A draw whose forward
model or likelihood cannot be evaluated is recorded; it is excluded from the
sum while its share of the draw mean is retained in the denominator, and the
scenario's evidence is marked unresolved. If any supported draw fails, the run
does not publish a probability.

## 6. The period prior

Giacalone et al. (2021) Section 2.4.1 give the only explicit scenario prior in
the framework: the probability that a transiting planet or eclipsing binary has
the period that produces the observed signal, integrated over `P ± 0.1 d`.
Both densities are normalized over 0.1–50 days:

```text
planet:  p(P) ∝ P^1.5   for 0.1 ≤ P ≤ 10 d
                P^0     for 10 < P ≤ 50 d

EB:      p(P) ∝ P^5.0   for 0.1 ≤ P ≤ 0.3 d
                P^0.5   for 0.3 < P ≤ 50 d
```

TREX evaluates the integral over `[P − 0.1, P + 0.1]`, clamping the centre so
the window stays inside the 0.1–50 d support and is never silently truncated.
It does not renormalize the result into a per-bin density. `x2P` scenarios apply
the prior at `2P`, because that is the model's physical period.

This factor is applied once, when probabilities are formed, and is recorded per
scenario in `period_log_priors`. It is not folded into the stored evidence and
is never applied a second time after refinement or resume. A caller that
normalizes evidences without a period reproduces the prior-free arithmetic; the
production engine always supplies the period.

The period prior is why two candidates with identical light-curve shapes but
different periods do not automatically receive identical FPPs. It is also a
population assumption, not a measurement of this particular system.

## 7. Estimating the evidence

### The basic estimator

The evidence is the prior-averaged likelihood:

```text
Z = ∫ L(θ) π(θ) dθ
```

TREX estimates it from `N` draws:

```text
ln Z ≈ ln [ (1/N) Σ_k exp(lnL_k + lnprior_k + ln(p/q)_k) ]
```

The terms are:

- `lnL_k`: the likelihood of draw `k`.
- `lnprior_k`: the draw's occurrence weight, such as the log companion
  frequency or the log expected background-star count. A draw excluded by an
  imaging or stability mask carries `−∞`. The weight is inside the average
  because it is part of the draw's prior weight; for a constant weight this
  reduces to adding it once.
- `ln(p/q)_k`: the importance weight when the draw came from a proposal `q`
  instead of the prior `p`.

The mean is computed with a log-sum-exp reduction, not by exponentiating raw
log-likelihoods. `−inf` entries contribute zero weight but still count in the
denominator, preserving the mean over all draws. `NaN` raises a numerical
failure; `+inf` propagates to the normalization status as an anomaly rather than
being converted to a finite weight.

One run seed is resolved before this stage. If the caller omitted it, TREX
generates it once for that run; `calc_probs` records it in the result provenance
or pending state, and a recovery checkpoint retains it. The scenario, importance
and nested seeds are derived from it, and a compatible resume restores it
instead of drawing another. The same seed, inputs, software, estimator, device
and execution path replay the same random stream; a different seed gives an
independent estimate, not a guaranteed identical one.

### Importance sampling and defensive proposals

Sampling every parameter from its astrophysical prior makes the integral an
extreme-value problem. The observed transit pins the occultor radius to a small
fraction of the prior range and the inclination to a narrow band, so prior draws
almost never land where the likelihood is appreciable. The reported evidence
would then be dominated by one lucky draw, with an ESS near 1.

TREX therefore uses importance sampling for two parameters in the Monte Carlo
stage:

- **Occultor radius**, centred on the radius ratio implied by the measured depth
  for that draw's host:
  `k_center = sqrt(depth × dilution)`. The window spans five times the measured
  depth uncertainty on each side (or the depth itself when no uncertainty was
  supplied) and is widened by a factor of 1.5 for geometry
  (`_PROPOSAL_GEOMETRY_SPAN`). Without a measured depth the window is empty and
  every draw falls back to the prior.
- **Inclination**, narrowed using the observed transit duration. For a circular
  orbit,
  `sin(π T14/P) = sqrt((1+k)² − b²)/(a/R_*)` (Seager & Mallen-Ornelas 2003,
  Eq. 3). TREX converts the duration bounds to impact-parameter bounds and then
  to an inclination window, widening them to cover the posterior bulk. The
  conversion approximates `sin i ≈ 1` and applies the eccentric factor only to
  translate `b` into inclination, so the duration window is approximate for
  eccentric orbits.

Each proposal is a **defensive mixture**: in this per-scenario stage, half the
draws come from the broad component and half from the narrowed window
(Hesterberg 1995). The broad component is the prior, or the transiting band of
the prior where the excluded region has identically zero likelihood. The
mixture keeps the proposal's support equal to the prior's and bounds every
importance weight: a poor proposal degrades the estimate toward plain prior
sampling instead of inflating its variance without bound. Importance sampling
is unbiased in linear evidence for any proposal whose support covers the
prior's, so these choices change the variance of the estimate, not the value it
converges to. They are numerical controls, not scientific parameters.

The depth and duration windows position proposals only. The measured depth does
not enter the likelihood, so it cannot pull the evidence toward the observation.

### Screening within a scenario

At large `N`, evaluating the full light curve for every transiting draw is
expensive. On the batched path TREX first evaluates a subset of the points (10%,
at least four, chosen as the points where a flat model fits worst) for all
candidate draws. Because Gaussian residual terms are never positive, that
partial chi-square bounds the full log-likelihood from above. Draws are then
processed in descending order of bound plus weight, and evaluation stops once no
remaining draw can come within 30 nats of the best evaluated term. At
`N = 10^6`, the skipped contribution is below `N e^-30 ≈ 10^-7`. The screening
changes cost, not the integral.

### CPU and accelerator paths

Where a scenario provides a native callback, TREX evaluates the same batched
float64 JAX kernel on CPU and GPU, and every float64-admissible draw (`e < 1`)
goes to that kernel. The two implementations agree to about `3.7 × 10^-6`
relative flux at `e = 0.93`, and `batman` stops returning finite values for
many orbits near `e = 0.98`; `batman` therefore remains the scalar CPU
reference rather than the high-eccentricity adjudicator. A JAX batch that
raises is retried with halved chunks, then re-evaluated with the `batman`
reference path. Every fallback is recorded in `jax_batch_fallbacks`, so a
report cannot imply the accelerator evaluated draws it did not.

Scenarios without a native callback use the scalar CPU integrand. This is a
performance distinction, not a different model: both paths evaluate the same
prior draws and the same forward model.

## 8. From evidence to FPP and NFPP

At normalization, TREX adds the period factor to every scenario and normalizes:

```text
P_i = exp(lnZ_i + lnΠ_i − logsumexp_j(lnZ_j + lnΠ_j))
```

where `Π_i` is the period factor. The normalization is done in log space. The
probability vector sums to one when the status is `ok`.

`FPP` is the direct sum over every scenario except `TP`, `PTP` and `DTP`:

```text
FPP = Σ_{i ∉ {TP, PTP, DTP}} P_i
```

`NFPP` is the direct sum over the neighbour scenarios:

```text
NFPP = Σ_{i ∈ {NTP, NEB, NEBx2P} for all neighbours} P_i
```

Both are dimensionless probabilities in `[0, 1]`. Because the neighbour
scenarios are a subset of the false-positive scenarios, `NFPP ≤ FPP` for the
same normalization. `PTP` and `DTP` are planet-on-the-target explanations with
additional dilution, so they count as the planet explanation for the target; a
planet on a bound companion, a background star or a resolved neighbour is a
false positive for the target.

Why the answer depends on the inputs:

- The likelihood changes with the light curve, the uncertainty, the exposure and
  the bin width. A deeper dip or a smaller `σ` favours models that reproduce it
  more sharply. The measured `depth_ppm` does not enter the likelihood; it
  centres the proposal window and decides which neighbours can host the depth.
- The aperture flux fractions change the dilution and the host-renormalized
  depth. A neighbour with a larger share of the aperture flux is a more viable
  host.
- Adding or removing a resolved neighbour adds or removes three scenario slots.
  A neighbour that can host the event contributes non-zero evidence to the
  normalization; a `host-excluded` neighbour contributes `-inf` and leaves the
  denominator unchanged.
- An unavailable scenario is not treated as a zero-probability explanation and
  is not silently dropped from a reduced inventory. If any scenario is
  unavailable for lack of an input, or a supported likelihood evaluation failed
  numerically, the run does not publish a normalized FPP/NFPP: it defers with
  diagnostics. A neighbour recorded as `host-excluded` is different: a zero
  aperture flux fraction or a depth larger than the neighbour's entire flux
  fraction is a physical exclusion, its three slots keep `-inf` evidence, and
  the run can still publish.
- The priors change the weights before the data: the period factor, the radius
  distribution, the companion frequency, the background density.
- The draw budget and estimator change the numerical estimate. A converged
  estimate should be stable across seeds; an unconverged one need not be.

No single direction of change is guaranteed. For example, changing the
background population changes the evidence of the `D*` and `B*` scenarios and
the total normalization, while `TP`, `PTP` and the `S*` scenarios do not use
that population. The resulting FPP can move either way. TREX records a fallback
population when it uses one, but it does not measure the size or sign of that
shift for a particular target.

## 9. Convergence, refinement and deferral

### ESS and the floor

TREX computes the Kish effective sample size of each scenario's weights:

```text
ESS = (Σ_k w_k)² / Σ_k w_k²
```

The reported value is a diagnostic of the estimator, not a physical quantity.
`ESS ≈ 1` means the evidence is effectively one draw. The registered floor is
100 effective draws, corresponding to a rough `1/√ESS = 0.1` nat Monte Carlo
error proxy with representative sampled support. A non-finite ESS is treated as
unconverged.

At the production budget, a scenario below the floor is eligible for refinement.
Below the production budget, no nested refinement is started; a failing run
defers instead.

### Stage 1: MC-pilot defensive importance sampling

The first refinement stage runs at the production budget for sub-floor scenarios
close enough to the best evidence that an ESS-1 error could matter (within 100
nats of the best period-weighted evidence), and continues to less promising
scenarios while their remaining mass could still clear the 10⁻³ allowance or
flip the decision. It builds a defensive importance proposal from the retained
Monte Carlo pilot draws: a kernel density concentrated on the high-weight
region in occultor radius, eccentricity, argument of periastron, the
transit-conditioned inclination, and the population/host variates. Period is
not a proposal dimension; it is fixed by the scenario. One tenth of the
estimation draws come from the prior and the rest from the kernel density. The
stage evaluates the same scenario model with the scenario's native float64
kernel where available, and a float64 scalar CPU integrand otherwise.

A refined estimate is accepted only if its ESS reaches the floor, its standard
error is at most 0.1 nat, and it agrees with the Monte Carlo estimate within
three combined standard errors (one-sided: a refined estimate below Monte Carlo
beyond that tolerance is refused). The agreement check catches a proposal that
missed a mode, which ESS and its own error estimate cannot see.

### Stage 2: nested sampling

If a scenario still fails the aggregate gate, TREX integrates it with nested
sampling. Nested sampling is a different algorithm: it maintains a set of live
points sampled from the prior, repeatedly replaces the lowest-likelihood point,
and accumulates the evidence from the shrinking prior volume (Skilling 2006).
It estimates the integral directly instead of reweighting draws from a proposal,
which is why it can help when importance sampling misses support. The nested
integrand samples the inclination variate only inside the transiting band and
carries the factor `ln(1 − band_low)` to restore the excluded prior mass; the
excluded band has zero likelihood, so the integral is unchanged.

The default estimator is **dynesty** (Speagle 2020), a dynamic nested sampler
using random slice sampling, with 400 live points and an evidence tolerance of
0.05 nat. A scenario's nested estimate is not accepted on a single run's
reported error. TREX runs an ensemble of three independent nested runs and
requires the ensemble's standard error to meet 0.1 nat. A scenario whose
estimated probability is at most one tenth of the unconverged allowance
(10⁻⁴, conditional on the already-converged scenarios) may be accepted from a
single bounded run instead. An ensemble that misses the bar is refined by
defensive importance sampling over its pooled nested posteriors; that refinement
must meet the same 0.1-nat bar and the ESS floor, and lie within three combined
standard errors of the ensemble mean.

Practical limits of this stage:

- At most 18 scenarios are escalated in one run.
- One dynesty run is capped at 2,000,000 likelihood calls. A run that reaches
  the cap is recorded as truncated and remains unresolved.
- The default escalation wall-time budget is 540 minutes. It can be lowered with
  `TREX_ESCALATION_WALL_SECONDS`, never raised past the ceiling. A host
  deadline can pause the run earlier.
- Escalation is attempted only at or above 1,000,000 draws. A smaller budget
  defers if convergence is unmet, and a converged-looking `ok` result at a
  smaller budget remains a software check, not production-budget scientific
  evidence.

### The opt-in BlackJAX estimator

Setting `TREX_NESTED_ESTIMATOR=blackjax` selects the experimental BlackJAX
nested-slice route instead of dynesty. Its status in this version:

- It runs at 3,200 live points, with a logical-call ceiling derived from the live
  count and dimension: 4,000,000 at 3,200 live in four dimensions. A lower
  explicit live count is refused; a higher one is honoured but is not labelled
  calibrated.
- It passed preregistered synthetic controls with a known answer: a
  four-parameter Gaussian met 48/50 within 0.1 nat in each of two independent
  50-seed runs, and a one-parameter exponential met 100/100 within 0.1 nat. The
  reported error in the pooled Gaussian control was about 10% below the observed
  run-to-run scatter, and the route used about eight times the logical likelihood
  calls of the lower-live control.
- It is **not calibrated** for real scenes, multimodal likelihoods or GPU
  execution. The default dynesty route's controls, results and timing are not
  evidence for this route's calibration, acceleration or scientific accuracy.
- It supports characterized resolved hosts through the native
  `CharacterizedHostIntegrand` and its `_jax_host_integrand` callback, the same
  stellar measure the Monte Carlo and dynesty paths integrate. Only the native
  Gaussian baseline requires dynesty; that combination is refused explicitly.
- BlackJAX native checkpoint recovery is implemented for trusted,
  owner-private local state at complete initialization and rank-deletion
  boundaries, so a hard kill can replay work since the last boundary. The live
  set, dead history, volume accounting, continuation key and cumulative logical
  calls are restored together. Remote restore parity has not been established.
- If the variable holds any value other than `dynesty` or `blackjax`, TREX
  raises instead of silently choosing one.
- Unset or default execution controls are the assessed policy: sequential
  (unbatched) initialization, fixed-isotropic slice directions, zero speculative
  expansions and the `kernel-mixture` proposal for nested-ensemble importance
  refinement. The experimental variants are selected by
  `TREX_BLACKJAX_PRECOND=eye|pilot|split` (directions),
  `TREX_BLACKJAX_BATCH_INIT=0|1` (padded initialization) and
  `TREX_BLACKJAX_W_MARCH=0|4|8` (speculative stepping-out probes); the
  refinement proposal is selected separately by
  `TREX_IMPORTANCE_PROPOSAL=kernel-mixture|probit-gaussian`. A non-default
  BlackJAX policy enters the native and completed-result identity, so it cannot
  resume another policy's state; a non-default importance proposal enters the
  refinement and completed-result identity and the CLI handoff. Neither inherits
  the default route's calibration marker. Padded or speculative physical work is
  not a known logical-call count and no speedup is claimed.

The known-evidence controls establish that the estimator can integrate those
synthetic distributions. They do not validate a real FPP or establish
acceleration.

### Publishing or deferring

A run publishes FPP and NFPP only when:

- every evaluated scenario has a finite evidence or a recorded physical
  exclusion (for example a `host-excluded` neighbour),
- no supported scenario is unavailable for lack of an input,
- no supported likelihood evaluation failed numerically,
- and scenarios still below the ESS floor hold at most 10⁻³ of the normalized
  probability, with no zero-sampled scenario that could flip the decision under
  the registered residual factor (10,000× the largest realized probability, used
  as a routing allowance).

If those conditions fail, the engine raises `TrexEvidenceUnavailableError` with
diagnostics instead of returning a number. `write_outputs` then writes
`failure.json`. A refusal is information: it says the model or the data did not
support a converged answer, not that the signal is or is not a planet.

The threshold screen is separate from this routing. The diagnostics mark
`FPP < 0.01` and `NFPP < 0.001` as pass or fail for display; the values do not
create a claim and do not change `claim_eligible`. Giacalone et al. (2021) used
the looser `FPP < 0.015` classification plane. Neither threshold is a calibrated
error bound.

## 10. What the diagnostics contain

On a successful run, `summary.json` and the result object carry:

- `status`: `ok` when normalization succeeded; otherwise a named failure mode
  such as `degenerate_evidence`, `all_neginf` or `anomaly`.
- `effective_sample_sizes`: per scenario, keyed by scenario name for the 15
  target scenarios and `N15`, `N16`, ... for neighbour slots.
- `scenario_monte_carlo_evidence`: each scenario's first-stage log evidence, its
  `1/√ESS` error, its ESS, draw count and availability status.
- `scenario_evidence_refinement`: whether a refined estimate replaced the Monte
  Carlo one, which estimator produced it, and why a scenario was not refined.
- `unavailable_scenarios`: any scenario that was excluded for a missing input,
  with the producing prior's reason. A publishable run has none; otherwise the
  run defers and the failure record names them.
- `nested_evidence_escalation`: the attempted and substituted scenarios, refusal
  reasons, numerical failures, seeds and attempt costs.
- `supplemental_nested_evidence`: the estimator-labelled records for substituted
  scenarios.
- `jax_batch_fallbacks`: draws re-evaluated by the `batman` path.
- `limb_darkening_grid`: an aggregate of the atmosphere lookups.
- `calculation_provenance`: seed, draws, input hashes, sector weights, exposure,
  bin width, depth, period, noise scope and neighbour metallicity treatment.
- `diagnostics`: the named checks, with `status`, `message`, `value` and
  `threshold` where applicable.

The diagnostic names a reader will see most often are `fpp`, `nfpp`,
`overall`, `scenario-*`, `evidence-effective-sample-size`,
`scenario-input-unavailable`, `scenario-numerical-failure`,
`numerical-stability`, `all-neginf`, `mc-pilot-importance-substitution` and
`evidence-estimator-substitution`. [Outputs](outputs.md) lists their meaning.

## 11. Assumptions and limits

These are properties of the implemented method, stated so a user can judge
whether the result applies to a given signal.

- **The scenario inventory is fixed.** Explanations outside it receive no
  probability mass. The net effect of an omitted population on FPP has no fixed
  sign.
- **Priors are population assumptions.** The radius, eccentricity, multiplicity
  and background distributions are not fitted to the target. The M-dwarf
  eclipsing-binary mass-ratio law is the solar-type Moe & Di Stefano law,
  extrapolated, and is not calibrated for M dwarfs. The M-dwarf bound-companion
  mass ratio is the field-binary Reggiani & Meyer law, extrapolated beyond its
  fitted separations and assumed independent of separation.
- **Stellar relations are not a characterization.** The mass–radius–temperature
  spline is used to build scenario models. The cool-dwarf node vector is
  retained as an uncalibrated scenario grid, not as a measured stellar
  posterior. The target's mass and radius are fixed scene inputs and their
  uncertainties are not marginalized. A resolved neighbour may supply a
  characterized stellar measure, in which case the evidence integrates over it;
  otherwise its mass and radius are fixed. Limb-darkening coefficient
  uncertainty is not marginalized.
- **The aperture model is approximate and uncalibrated.** `target.from_tic`
  derives aperture flux shares from catalogue TESS magnitudes through a fixed
  0.75-pixel circular Gaussian PSF on the SPOC aperture mask. The real TESS
  pixel response function varies with camera, CCD and field position, and no
  injection/recovery calibration supports the resulting scene. `calc_depths`
  reports the same approximation as a diagnostic, with equal-sector means and
  no retained per-sector aperture-scene provenance, and cannot be used as
  calibrated aperture evidence.
- **The unresolved-scene bound has edges.** Without a measured contrast curve,
  unresolved companions are bounded by the Gaia DR2 close-pair limit of
  2.2 arcsec (or an explicitly supplied scene limit). A measured curve
  constrains only the separations it covers; inside its inner working angle and
  outside its outer edge it provides no constraint. The imaging mask cannot
  exclude companions beyond the measured curve.
- **Background stars have no retained sky positions.** A simulated background
  star cannot be individually cross-matched against the Gaia neighbours in the
  aperture.
- **The centroid diagnostic is neutral.** A retained difference-image centroid
  record is reported for review but does not reweight or delete a resolved
  neighbour, because no calibrated source-assignment likelihood is implemented.
- **Noise is a diagonal approximation.** The shared `σ` from
  `calibrate_trex_noise` is a per-sector `max(1, std) × max(1, β)` envelope, with
  the maximum over retained sectors applied to every scenario. It assumes
  stationary out-of-transit noise and independent transit events. It is not a
  Gaussian process, does not marginalize estimator uncertainty, and does not
  establish calibrated FPP coverage. Eccentric secondary eclipses outside the
  half-phase mask are not removed unless their phase and duration are declared
  through `event_windows`. The default record labels its circular assumption.
- **Convergence checks are not proof.** ESS and nested-sampling agreement cannot
  rule out a mode missed by both. The residual-mass allowance is a routing
  heuristic, not a bound on FPP error.
- **The probabilities are not empirically calibrated and are claim-ineligible.**
  They are conditional on the inputs, the priors and the fixed scenario list,
  and they have not been calibrated against a sample of known planets and false
  positives. `claim_eligible` is always `false`. A planet claim needs other
  observations and independent scientific assessment.

## 12. References used by the implementation

| Topic | Reference |
|---|---|
| Scenario framework, FPP and NFPP definitions, planet radius prior, period prior | Giacalone et al. (2021), AJ 161, 24, <https://doi.org/10.3847/1538-3881/abc6af> |
| Transit and eclipse model | Mandel & Agol (2002), ApJ 580, L171, <https://doi.org/10.1086/345520>; Kreidberg (2015), PASP 127, 1161, <https://doi.org/10.1086/683602> |
| Transit geometry and duration | Winn (2010), <https://doi.org/10.48550/arXiv.1001.2010>; Seager & Mallen-Ornelas (2003), ApJ 585, 1038, <https://doi.org/10.1086/346105> |
| Planet eccentricity prior | Kipping (2013), MNRAS 434, L51, <https://doi.org/10.1093/mnrasl/slt075> |
| Binary periods, mass ratios, eccentricities, companion frequency | Moe & Di Stefano (2017), ApJS 230, 15, <https://doi.org/10.3847/1538-4365/aa6fb6> |
| M-dwarf companions | Winters et al. (2019), AJ 157, 216, <https://doi.org/10.3847/1538-3881/ab05dc>; Reggiani & Meyer (2013), A&A 553, A124, <https://doi.org/10.1051/0004-6361/201321631> |
| Hierarchical stability | Mardling & Aarseth (2001), MNRAS 321, 398, <https://doi.org/10.1046/j.1365-8711.2001.03974.x> |
| Stellar mass–radius–temperature relations | Torres, Andersen & Giménez (2010), A&ARv 18, 67, <https://doi.org/10.1007/s00159-009-0025-1> |
| Limb darkening | Claret (2017), A&A 600, A30, <https://doi.org/10.1051/0004-6361/201629705>; Kostogryz et al. (2022), A&A 666, A60, <https://doi.org/10.1051/0004-6361/202243722> |
| Background population | Girardi et al. (2005), A&A 436, 895, <https://doi.org/10.1051/0004-6361:20042352> |
| Field extinction | Schlegel, Finkbeiner & Davis (1998), ApJ 500, 525, <https://doi.org/10.1086/305772>; Schlafly & Finkbeiner (2011), ApJ 737, 103, <https://doi.org/10.1088/0004-637X/737/2/103> |
| TIC TESS magnitudes | Stassun et al. (2019), AJ 158, 138, <https://doi.org/10.3847/1538-3881/ab3467> |
| Noise calibration | Pont, Zucker & Queloz (2006), MNRAS 373, 231, <https://doi.org/10.1111/j.1365-2966.2006.11012.x>; Winn et al. (2008), ApJ 683, 1076, <https://doi.org/10.1086/589737> |
| Importance sampling | Hesterberg (1995), Technometrics 37, 185, <https://doi.org/10.1080/00401706.1995.10484303> |
| Nested sampling | Skilling (2006), Bayesian Analysis 1, 833, <https://doi.org/10.1214/06-BA127>; Speagle (2020), MNRAS 493, 3132, <https://doi.org/10.1093/mnras/staa278> |
| Gaussian linear-baseline marginalization | Gibson (2014), MNRAS 445, 3401, <https://doi.org/10.1093/mnras/stu1975> |
| Effective sample size | Kish (1965), *Survey Sampling*, Wiley, section 11.7 |
| Gaia close-pair resolution | Brown et al. (2018), A&A 616, A1, <https://doi.org/10.1051/0004-6361/201833051>, as adopted by Giacalone et al. (2021) Section 1 |

Bundled tables carry their own attribution notices in
`src/_data/` in the source tree, including the Claret 2017 CDS
attribution, the MPS-ATLAS attribution, and the TRICERATOPS MIT licence notice.
