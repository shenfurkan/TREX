# Concepts: what TREX calculates

This page defines the astronomy, the probability arithmetic and the diagnostics
TREX reports, in the order they appear in a run. Each term also appears in the
[glossary](glossary.md).

If you want the implementation details behind these definitions, read
[How TREX works](how-trex-works.md) next.

## 1. Transits, candidates and false positives

A **light curve** is a series of brightness measurements of a star over time.
TREX expects a **phase-folded** light curve: the measurements stacked on the
candidate period so that every orbit overlaps, with time measured in days from
the midpoint of the transit. The out-of-transit flux is normalized to about 1.

A **transit** is the dip produced when a body passes in front of a star. For a
planet, the fractional depth of a central transit is approximately

```text
depth ≈ (R_p / R_*)²
```

where `R_p` is the planet radius and `R_*` the star radius. A 1% dip, or
10,000 ppm, corresponds to a radius ratio of 0.1: a Jupiter-sized body around a
Sun-sized star. The word **ppm** means parts per million, so 1 ppm is 10⁻⁶ of
the out-of-transit flux.

A **transit candidate** is a signal that looks like such a dip but has not been
shown to be a planet. A dip on its own is not proof. Several physical systems
produce a similar shape:

- An **eclipsing binary (EB)**: two stars orbiting each other, with one passing
  in front of the other. A stellar eclipse can be far deeper than a planetary
  transit and can mimic a shallow one when the pair is blended with other light.
- A **background eclipsing binary**: an unrelated pair behind or near the
  target, whose light falls in the same photometric aperture. The aperture is
  the set of pixels summed to make the light curve. If the pair is faint and
  blended, its eclipses appear diluted in the target's light curve.
- A **resolved neighbour**: a star visible as a separate source in the
  catalogue or in images, but still inside the aperture or close enough that
  scattered light enters it. If that star varies or eclipses, the aperture
  light dips.
- An **unresolved bound companion**: a star in the same system, too close to
  separate in the available images, which can host the transit or dilute it.
- **Instrumental or astrophysical variability** that happens to be periodic.
  TREX does not model these as scenarios, so it cannot assign them a
  probability. A run assumes the user has already vetted the signal for obvious
  systematics and stellar variability.

A **false positive** is an explanation of the signal other than a planet
transiting the target star. The point of a false-positive calculation is to
compare the planet explanation with the alternatives that can be modelled, using
the actual light curve and the actual stellar scene rather than the dip alone.
In this arithmetic a planet transiting a resolved neighbour, an unresolved
bound companion or a background star is also a false positive for the target:
the question is whether the signal is a planet transiting *this* star, not
whether a planet exists somewhere in the scene.

A **TESS Object of Interest (TOI)** is a candidate that has received follow-up
and a catalogue entry. A TESS Input Catalog (**TIC**) ID identifies the star.
Neither designation is a validation.

## 2. Scenarios: the explanations TREX compares

A **scenario** is one specified explanation for the signal and its
surroundings. TREX uses the inventory of Giacalone et al. (2021), *AJ* 161, 24,
<https://doi.org/10.3847/1538-3881/abc6af>:

| Group | Scenario | Event host | Additional light |
|---|---|---|---|
| Target only | `TP` | target | none |
| Target only | `EB`, `EBx2P` | target | the eclipsing secondary |
| Unresolved bound companion, event on the target | `PTP` | target | bound companion dilutes |
| Unresolved bound companion, event on the target | `PEB`, `PEBx2P` | target | bound companion dilutes; the secondary also eclipses |
| Bound companion as host | `STP` | bound companion | target dilutes |
| Bound companion as host | `SEB`, `SEBx2P` | bound companion | target dilutes; the secondary also eclipses |
| Unresolved background star, event on the target | `DTP` | target | background star dilutes |
| Unresolved background star, event on the target | `DEB`, `DEBx2P` | target | background star dilutes; the secondary also eclipses |
| Background star as host | `BTP` | background star | target dilutes |
| Background star as host | `BEB`, `BEBx2P` | background star | target dilutes; the secondary also eclipses |
| Resolved neighbour as host | `NTP`, `NEB`, `NEBx2P` | that neighbour | the rest of the aperture dilutes |

`x2P` means the pair's true period is twice the observed signal period, so only
one of its two eclipses appears in the folded signal. It applies to every binary
family: `EBx2P`, `PEBx2P`, `SEBx2P`, `DEBx2P`, `BEBx2P` and `NEBx2P`.

The 15 target scenarios are three target-only scenarios, six with an unresolved
bound companion and six with an unresolved background star. They are evaluated
once. The three neighbour scenarios are evaluated once for each resolved
neighbour that can host the observed depth, so an evidence vector with `n`
resolved neighbours has `15 + 3n` entries. [Scenarios](scenarios.md) gives the
physical configuration, priors and availability conditions for each one.

Two limitations follow from using a fixed inventory:

- An explanation outside the inventory is not tested. An unavailable scenario
  is not a scenario with probability zero: its explanation cannot be evaluated
  with the supplied inputs, and a run that needs it defers instead of publishing
  a normalization over a reduced inventory.
- The net effect of an omission has no fixed sign. Adding a missing
  false-positive explanation can raise FPP; adding a missing
  planet-on-the-target explanation (`TP`, `PTP` or `DTP`) raises the
  normalization and adds a scenario excluded from FPP, which can lower it.

## 3. Priors, likelihood and evidence

A **prior** is an assumption about a quantity before the light curve is
considered. TREX uses priors at two levels.

**Population priors** describe how often a configuration occurs and how its
physical parameters are distributed: the planet radius distribution, the
eccentricity distribution, the binary mass-ratio distribution, the rate of
bound companions, the density of background stars. A **draw** is one simulated
system sampled from these priors: one mass ratio, one period, one radius, one
inclination, one impact parameter (the projected star–body separation at
conjunction, in stellar radii), and so on.

**The period factor** is the probability that a transiting planet or eclipsing
binary has the period that produces the observed signal, integrated over
`P ± 0.1 d` (Giacalone et al. 2021 Section 2.4.1). It is a scenario weight, not
a density over a sampled parameter. TREX evaluates it once, before
normalization, and records it separately; it is never stored inside the
evidence. [How TREX works](how-trex-works.md#6-the-period-prior) gives the
exact broken power laws.

The **likelihood** of one draw is the probability of the observed light curve
given that draw's predicted light curve. TREX uses a Gaussian likelihood with a
fixed uncertainty per point:

```text
ln L(θ) = −0.5 · Σ_i [ (F_obs,i − F_model,i(θ))² / σ_i² ]
```

Here `F_obs` is the observed normalized flux, `F_model` is the model flux,
`σ` is the flux uncertainty (one value per point or a scalar), and `i` runs
over every point supplied. `ln L` is a natural logarithm and its unit is the
**nat**. The leading `−0.5` and the omitted Gaussian constant do not change
comparisons between scenarios. A resolved host is evaluated in its own
renormalized frame with `σ_host = σ/f`, so the Gaussian normalization term
gains `N log f`; the renormalization Jacobian `f^(−N)` cancels that gain for
that same host. The reduced chi-square is therefore invariant, and the
cancellation is per host: it is not an assumption that the omitted constant is
identical across hosts. TREX does not clip or reweight points inside the
likelihood. A draw that cannot transit is removed by prior support (log weight
`−∞`), not by being assigned a zero-likelihood data point.

An optional **native-time baseline likelihood** replaces the reduced `−χ²/2`
with the exact normalized Gaussian density of the unbinned chronological data,
after integrating a caller-supplied Gaussian prior on linear baseline
coefficients (Gibson 2014). It is one likelihood with a nuisance trend
marginalized, not a second fitted noise model.

The **evidence** of a scenario is the likelihood averaged over the scenario's
prior:

```text
Z_i = ∫ L_i(θ) π_i(θ) dθ
```

`Z_i` measures how well this explanation accounts for the data, averaged over
all the parameter values it allows. TREX estimates the integral from `N` draws:

```text
Z_i ≈ (1/N) · Σ_k exp( lnL_i(θ_k) + lnprior_k + ln(p/q)_k )
```

where `lnprior_k` is the draw's occurrence weight (for example the companion
frequency or the expected background-star count) and `ln(p/q)_k` is the
importance weight when the draw came from a proposal `q` instead of the prior
`p`. The arithmetic is done in log space, so the reported quantities are
`ln Z_i`.

## 4. From evidence to FPP and NFPP

Write `Π_i` for the period factor. The normalized probability of scenario `i`
is

```text
P_i = Π_i · Z_i / Σ_j ( Π_j · Z_j )
```

`Π_i` is the period factor described above; `π_i` is reserved for the
parameter prior inside the evidence. The probabilities of all modelled
scenarios sum to one when the normalization status is `ok`. `Π_i` and `Z_i` are
not probabilities that a planet exists; they are weights inside this fixed
inventory.

TREX reports two sums:

```text
FPP  = Σ P_i   over every scenario except TP, PTP, DTP
NFPP = Σ P_i   over NTP, NEB, NEBx2P for every resolved neighbour
```

`FPP` is the total modelled probability of a false positive for the target.
`NFPP` is the part of it hosted by resolved nearby stars. Because the neighbour
scenarios are included in FPP, NFPP cannot exceed FPP for the same
normalization.

`PTP` and `DTP` are planet-on-the-target explanations with additional dilution,
so they are excluded from FPP together with `TP`. A planet on a bound companion
(`STP`), a background star (`BTP`) or a resolved neighbour (`NTP`) is a false
positive for the target.

FPP is computed as a direct sum over the false-positive scenarios, not as
`1 − (P_TP + P_PTP + P_DTP)`. The direct sum keeps a very small false-positive
probability visible instead of losing it to floating-point cancellation.

### An invented example of the arithmetic

Suppose an analysis leaves three scenarios with non-negligible weight. The
numbers below are made up to show the arithmetic; they are not a TREX result.

```text
scenario   Π_i · Z_i     P_i
TP            0.90       0.90 / 1.00 = 0.90
EB            0.08       0.08
NEB           0.02       0.02
sum           1.00       1.00

FPP  = 0.08 + 0.02 = 0.10
NFPP = 0.02
```

The planet explanation carries 90% of the modelled weight, so FPP is 10%. If
the same evidences were combined with a different period factor, or with an
extra scenario, both sums would change even though the light curve did not.

## 5. Draws, effective sample size and convergence

`N` is the number of Monte Carlo draws per scenario. The production default is
1,000,000, the value Giacalone et al. (2021) use. Smaller budgets exercise the
software; the 4,096-draw examples in this documentation are software checks,
not scientific results.

A raw draw count is not the same as the information actually used. If a few
draws carry almost all the weight, the estimate behaves like a much smaller
sample. The **effective sample size (ESS)** measures that concentration:

```text
ESS = (Σ_k w_k)² / Σ_k w_k²
```

where `w_k` is the weight of draw `k` in linear space. For a non-degenerate
weight set, ESS is a dimensionless count between 1 and `N`. `ESS ≈ N` means the
estimate uses the draws evenly. `ESS ≈ 1` means one draw dominates and the
"integral" is close to a single sample. All-zero weight gives ESS 0; a
non-finite weight gives NaN, which every floor treats as unconverged. Nested
refinement uses live points and reports its own error, not a Kish ESS.

TREX's ESS floor is 100. Scenarios below it can be refined in two stages:
defensive importance sampling (reweighting draws from a proposal that keeps a
fixed share on the prior, so the weights stay bounded) and, if needed, nested
sampling (integrating the evidence as the sampled prior volume shrinks). In the
per-scenario radius and inclination proposal that prior share is one half; in
the refinement stage it is one tenth. If scenarios still below the floor hold
more than 10⁻³ of the normalized probability, TREX does not report an FPP; it
**defers**. A sampled scenario with no finite weight can also force deferral.

| Control | Value | Role |
|---|---|---|
| Production draws | 1,000,000 per scenario | scientific budget; smaller is a software check |
| ESS floor | 100 | refinement eligibility; a rough `1/√ESS = 0.1` nat proxy |
| Unconverged mass allowance | 10⁻³ | routing heuristic on the realized probability |
| Nested ensemble standard error | 0.1 nat | acceptance bar for a substituted estimate |
| Single-run conditional bound | 10⁻⁴ | 0.1 × the unconverged allowance |
| `FPP` display threshold | 0.01 | diagnostic screen only |
| `NFPP` display threshold | 0.001 | diagnostic screen only |

The routing values are guards against a precise-looking number computed from a
handful of draws. The 10⁻³ allowance is not a bound on the true residual
evidence: missed support can be underestimated by an arbitrary factor, and a
high ESS cannot rule out a region that neither the proposal nor the nested
sampler visited. The display thresholds do not create a claim.

## 6. Units and conventions

| Quantity | Unit or convention |
|---|---|
| Phase-folded `time` | days from the transit midpoint, in the same time standard as the ephemeris |
| Native time for noise calibration and the baseline | BTJD_TDB days |
| Orbital period | days |
| Flux | normalized, out-of-transit ≈ 1, dimensionless |
| Flux uncertainty | same normalized units, one value per point or a scalar |
| Transit depth `depth_ppm` | ppm; observed depth of the aperture light curve |
| Exposure time | days |
| Phase-bin width | days; `bin_integration_days = 0` for unbinned points |
| `nsamples` | dimensionless count of sub-exposure midpoints (default 20) |
| Sector weight | dimensionless sum of inverse variances |
| Sky separation | arcseconds |
| Brightness difference | magnitudes (Δmag) |
| Extinction | magnitudes: `A_V`, `E(B−V)` |
| Background field position | Galactic longitude and latitude, degrees |
| Evidence | natural log (nats) internally |
| FPP, NFPP | dimensionless probabilities in [0, 1] |

Three conventions cause avoidable mistakes:

- **Time standards.** TDB is a barycentric timescale that runs uniformly apart
  from small relativistic corrections. BJD and BTJD both count in it and differ
  only by an offset: BTJD is BJD minus 2457000. Use one convention consistently
  between the light curve and the epoch. The native-time noise calibration and
  the optional baseline require BTJD_TDB.
- **Depth.** `depth_ppm` is a measured input. It centres the occultor-radius
  proposal and decides which neighbours can host the depth. It does not enter
  the likelihood; the light curve carries the depth information. TREX does not
  measure the depth for you.
- **Uncertainty.** Calibrate `σ` on the native, unfolded photometry, including
  the supported time-correlated-noise correction. `calibrate_trex_noise`
  measures, per retained sector, `max(1, std(z)) × max(1, β)` with
  `z = (flux − out-of-transit median)/σ` and `β` the binned-RMS factor at
  `T14/cadence`; the factor supplied to every scenario is the maximum over the
  retained sectors. A sector that cannot support the estimate defers rather
  than falling back to `β = 1`. Do not estimate a separate noise scale from
  folded points or per scenario. The default mask assumes a circular phase-0.5
  secondary; `event_windows` replaces that assumption with measured eclipse
  phases and durations. See [Inputs](inputs.md#5-what-the-uncertainty-calibration-needs).

## 7. What the probabilities do not mean

- They are **not** `P(planet exists)`. They are conditional on the supplied
  light curve, stellar scene, population models, priors and scenario list.
- A small FPP is **not** a model-independent detection. TREX does not validate
  or confirm planets, and `claim_eligible` is always `false`.
- Missing scenarios are **not** impossible scenarios.
- A low ESS, a deferred run, or a missing summary is not a low probability. Read
  `failure.json` and its diagnostics; a missing `summary.json` means no
  probability was published.
- A high ESS says the weights are well behaved for the scenarios that were
  sampled. It does not prove the model, the priors or the input scene are right.
- There is **no empirical FPP calibration**. The probabilities are not
  calibrated against a catalogue of confirmed planets and false positives.
- A threshold screen such as `FPP < 0.01` is a reporting convention, not a
  discovery criterion. TREX's diagnostics apply `FPP < 0.01` and
  `NFPP < 0.001` for display; Giacalone et al. (2021) used the looser
  `FPP < 0.015` classification plane. Neither value changes `claim_eligible`.

## 8. Next

- [Tutorial](tutorial.md): run the calculation once and read the files.
- [How TREX works](how-trex-works.md): the model and estimator in depth.
- [Scenarios](scenarios.md): every scenario, its host and its availability.
- [Inputs](inputs.md): the measured arrays and scene values, with units.
- [Outputs](outputs.md): where each number is written.
- [How TREX differs from TRICERATOPS](differences-from-triceratops.md): the
  deliberate implementation differences.
- [Background population](background-population.md): the TRILEGAL stars behind
  the `D*` and `B*` scenarios.
- [Glossary](glossary.md): every term in one place.
