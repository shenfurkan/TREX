# Scenarios

This page is a reference for what each row of `scenarios.csv` claims is
happening. It covers the scenario names, their physical configuration, the
population they draw from, the inputs that make them available, and which
scenarios enter FPP and NFPP. Read [Concepts](concepts.md) for FPP, evidence and
priors, and [How TREX works](how-trex-works.md) for the estimator that turns
scenario evidences into probabilities.

TREX uses the fixed scenario inventory of Giacalone et al. (2021), *AJ* 161, 24,
<https://doi.org/10.3847/1538-3881/abc6af>: 15 target scenarios and three
scenario types for each resolved neighbour. The 18 names are 15 target names
plus three neighbour names; the evaluated vector has `15 + 3n` entries when the
scene has `n` resolved neighbours.

## Naming

Read each name as an optional prefix, an event code (`TP` for a transiting
planet or `EB` for an eclipsing binary), and an optional `x2P` suffix. `x2P`
means the model period is twice the observed signal period. The prefix says
where the event is and what other light enters the aperture:

- No prefix (`TP`, `EB`, `EBx2P`): the event is on the target, with no
  unresolved companion.
- `P`: an unresolved **bound** companion is present. `PTP`/`PEB` keep the event
  on the target and let the companion dilute it. `STP`/`SEB` move the event to
  the companion; `S` marks that the companion is the host.
- `D`: an unresolved **background** star dilutes the light; the event stays on
  the target.
- `B`: the unresolved background star is the host; the target's light dilutes
  the event.
- `N`: a resolved neighbour is the host. The three neighbour scenarios are
  `NTP`, `NEB` and `NEBx2P`.

`x2P` means the pair's true period is `2P`, so eclipses occur every `P`; only
one of the two eclipses is assumed to appear in the folded signal. The label is
used only for eclipsing-binary scenarios. In every EB family it also selects the
near-twin mass-ratio branch (`q >= 0.95`; see the priors below).

## The 15 target scenarios

Scenario index order is fixed. It is the order used by the low-level `lnZ` array
and by `scenario_index` in the output records.

| Index | Scenario | Host star | Event | Diluted by | Planet event | Model period | FPP | NFPP |
|---:|---|---|---|---|---|---|---|---|
| 0 | `TP` | target | transit | — | yes | `P` | no | no |
| 1 | `EB` | target | eclipse | — | no | `P` | yes | no |
| 2 | `EBx2P` | target | eclipse | — | no | `2P` | yes | no |
| 3 | `PTP` | target | transit | bound companion | yes | `P` | no | no |
| 4 | `PEB` | target | eclipse | bound companion | no | `P` | yes | no |
| 5 | `PEBx2P` | target | eclipse | bound companion | no | `2P` | yes | no |
| 6 | `STP` | bound companion | transit | target | yes | `P` | yes | no |
| 7 | `SEB` | bound companion | eclipse | target | no | `P` | yes | no |
| 8 | `SEBx2P` | bound companion | eclipse | target | no | `2P` | yes | no |
| 9 | `DTP` | target | transit | background star | yes | `P` | no | no |
| 10 | `DEB` | target | eclipse | background star | no | `P` | yes | no |
| 11 | `DEBx2P` | target | eclipse | background star | no | `2P` | yes | no |
| 12 | `BTP` | background star | transit | target | yes | `P` | yes | no |
| 13 | `BEB` | background star | eclipse | target | no | `P` | yes | no |
| 14 | `BEBx2P` | background star | eclipse | target | no | `2P` | yes | no |

"Diluted by" names the additional light that shallows the event. For `S*` and
`B*` it is the target's light, because the event is on the unresolved companion
or background star. The "Planet event" column says whether the event itself is
a planet transit; it is not the same question as FPP membership.

The "Model period" column is also the period at which the scenario period prior
is evaluated. For `x2P` scenarios the factor is evaluated at `2P`, because that
is the model's physical period.

## The neighbour scenarios

For each resolved neighbour, TREX appends three slots. The first neighbour uses
indices 15, 16 and 17; the next uses 18, 19 and 20; and so on.

| Slot offset | Name | Host star | Event | Diluted by | Planet event | Model period | FPP | NFPP |
|---:|---|---|---|---|---|---|---|---|
| 0 | `NTP` | resolved neighbour | transit | — | yes | `P` | yes | yes |
| 1 | `NEB` | resolved neighbour | eclipse | — | no | `P` | yes | yes |
| 2 | `NEBx2P` | resolved neighbour | eclipse | — | no | `2P` | yes | yes |

With `n` resolved neighbours the evidence vector has `15 + 3n` entries. A
neighbour that cannot host the event keeps its three slots at `-inf` evidence
and is recorded with status `host-excluded`, so the vector length does not depend
on which neighbours were usable. When `n = 0`, the vector has 15 entries and
`NFPP = 0`.

Two different labels appear in the outputs:

- `scenarios.csv` uses `scenario` = `NTP`/`NEB`/`NEBx2P` and the neighbour's
  `ID`.
- The per-scenario maps and rows in `summary.json` use `N15`, `N16`, `N17` for
  the first neighbour, `N18`, `N19`, `N20` for the second, and so on, matching
  the scenario index. This applies to `effective_sample_sizes` and to the
  `scenario_monte_carlo_evidence` and `scenario_evidence_refinement` rows;
  `scenario_monte_carlo_evidence` rows carry both `scenario` (`N15`) and
  `scenario_label` (`NTP`). The result object's `period_log_priors` map uses the
  same `N15` keys. The neighbour order is the scene's `resolved_neighbors`
  order; `scenarios.csv` carries the `source_id` as `ID`.

## What each family models

### Target planet and binary

`TP` is the planet explanation: a planet of radius drawn from the radius prior
transits the target, with the target's measured mass, radius and limb darkening.
The planet's eccentricity is drawn from the Kipping (2013) prior and its
inclination from the transiting band. `PTP` and `DTP` are the same planet on the
target with, respectively, an unresolved bound companion or a background star
adding constant light. All three are planet-on-target scenarios for FPP.

`EB` and `EBx2P` replace the planet with a stellar secondary drawn from the
binary mass-ratio and eccentricity distributions. The eclipsing pair contributes
its own light, so the model includes both the eclipse shape and the constant
flux of the secondary outside eclipse; both eclipses are evaluated at the
observed times. `PEB`/`PEBx2P` add a bound companion, and `DEB`/`DEBx2P` add a
background star, as an extra constant flux term.

### Bound companions

`PTP`/`PEB` keep the event on the target and add an unresolved bound companion
drawn from the stellar multiplicity model. The companion's flux dilutes the
observed dip and changes the aperture flux budget. `STP`/`SEB` move the event to
the companion: the companion's drawn mass, radius and temperature define the
transit geometry and the stellar limb darkening, and the target's light dilutes
the event.

Every bound-companion draw is checked against the imaging constraint and the
catalog resolution limit, and against the Mardling & Aarseth (2001) Eq. 90
stability criterion for the observed inner orbit. A companion that would have
been detected, is wider than the unresolved limit, or is dynamically unstable
gets zero prior weight. The occurrence weight is the companion frequency for the
target mass, capped at one companion per primary.

### Background stars

`DTP`/`DEB` keep the event on the target and draw an unresolved background star
from the retained TRILEGAL population. The star adds constant light and dilutes
the dip. `BTP`/`BEB` move the event to the background star, using that star's
simulated mass, radius (derived from its mass and surface gravity), temperature
and TRILEGAL `[M/H]`; the target's light dilutes the event.

The occurrence weight is the expected number of such stars inside the circular
unresolved field: the population's surface density times `π r_limit²`. Every
retained star can be drawn, and the star is placed at a separation uniform in
area out to the unresolved limit. The contrast curve only excludes detections
over its measured separation range; inside its inner working angle and beyond
its outer edge it provides no constraint. The retained cache has no individual
sky positions, so a simulated star cannot be cross-matched one-to-one against
Gaia neighbours; that overlap remains a model limitation.

### Resolved neighbours

The three neighbour scenarios are the same planet/binary pair evaluated on the
neighbour's own renormalized light curve. The neighbour's aperture flux fraction
sets how much of the aperture its own flux represents. The depth the neighbour
would have to produce is `depth_observed / f_neighbour`. `NTP` is a planet
transit, and `NEB`/`NEBx2P` are eclipsing binaries; all three count as false
positives for the target because the event is not on the target.

The neighbour's mass and radius are scene inputs. When the scene supplies a
`stellar_parameter_distribution`, the three neighbour scenarios draw the host
mass, radius and temperature from it per draw; otherwise those values are fixed
and their uncertainties are not marginalized. Its limb darkening comes from its
own temperature and metallicity where available. A neighbour without its own
metallicity may use the target's measured value on the reference-grid path. The
retained centroid diagnostic supplies a neutral spatial weight (`0.0`), not a
calibrated source-assignment likelihood.

## Availability: what each family needs

The engine decides availability before evaluating a scenario. There are two
different failure levels: scene/input admission, which refuses the run before
draws, and a per-scenario slot that is unavailable or physically excluded.

### Admission before any draws

The engine refuses the run, rather than silently omitting scenarios, when:

- the scene lacks finite positive target mass, radius or temperature, a finite
  TESS magnitude, or a positive parallax;
- the target metallicity `feh_dex` is absent, or is supplied without measured
  provenance;
- neither a measured contrast curve nor a positive catalog resolution limit is
  supplied;
- the calibrated per-sector aperture fractions, or the observed per-sector
  inverse-variance weights, are missing or inconsistent;
- the retained background population is missing, contains no star at least as
  faint as the target, or its simulated sky area is missing;
- a resolved neighbour can host the observed depth but has no mass and radius.
  This raises `TrexEvidenceUnavailableError`; the neighbour is not dropped,
  because dropping a possible host biases FPP low.

The target metallicity is required at admission, before any scenario is
evaluated: scenarios hosted by the target need it for limb darkening, and the
target-hosted EB and companion-hosted scenarios also need a host atmosphere for
the eclipsed secondary or the drawn companion. Acquisition-side reasons are
listed in [From TIC](from-tic.md).

### Per-scenario requirements

| Requirement | Applies to | If missing |
|---|---|---|
| Finite target metallicity `feh_dex` with provenance | all scenarios at engine admission | run refused (`ValueError`) |
| Unresolved angular constraint: measured contrast curve, or positive resolution limit | bound-companion and background-star families; the scene requires one in every case | scene refused if neither is supplied |
| Positive target parallax | bound-companion families (converts period to projected separation) | scene admission refuses; target-hosted and background scenarios do not use it |
| Retained background population, its simulated sky area, and a finite target TESS magnitude | background-star families | engine admission fails; an empty population is not an empty sky |
| Target mass in the bound-companion domain: 0.075–40 M☉ and a stellar mass-ratio floor below one | bound-companion families | those slots are unavailable; the engine withholds FPP |
| Target aperture flux fraction in `(0, 1]` | all | input admission refuses |
| Neighbour aperture flux fraction `> 0` and `>= observed depth fraction` | that neighbour's three scenarios | `host-excluded`: `-inf` evidence, zero probability, but not a missing-input failure |
| Neighbour mass and radius, if it can host the depth | that neighbour's three scenarios | the whole run is refused at preflight (`TrexEvidenceUnavailableError`) |
| Neighbour's own metallicity, or the target-metallicity fallback | that neighbour's three scenarios on the reference-grid path | fallback is used and recorded; this is not an unavailable scenario |
| Neighbour's own `stellar_atmosphere` when physical stellar light is selected | that neighbour's three scenarios | input admission refuses; the target's composition is not borrowed on that path |
| Optional measured contrast curve in the TESS band: at least two finite points, positive strictly increasing separations | optional | without a curve, a positive catalog resolution limit must be supplied; the Gaia DR2 close-pair value is 2.2 arcsec |

The metallicity rule is asymmetric by design:

- A target metallicity must be measured and must carry provenance. TREX refuses
  a bare number.
- A neighbour without its own metallicity may use the target's measured value on
  the reference-grid path. The output records this as
  `metallicity_treatment = "target-metallicity fallback"` and does not present
  it as a measurement of the neighbour. With physical stellar light, the
  neighbour instead needs its own source-associated `stellar_atmosphere`.
- A background star uses its own TRILEGAL `[M/H]` instead of the target's.

### Unavailable, host-excluded, or zero

These outcomes are not interchangeable:

- **Unavailable.** A required input is missing, so the scenario has no model.
  Its slot stays at `-inf` evidence and carries status `unavailable` plus the
  reason. The engine treats any unavailable scenario as a failed evidence set:
  it raises `TrexEvidenceUnavailableError`, reports no FPP or NFPP, and writes
  no `scenarios.csv`. It does not silently assign the scenario probability zero.
- **Host-excluded.** A resolved neighbour cannot produce the observed depth
  given its aperture flux fraction, or has zero aperture flux fraction. Its
  three slots stay at `-inf` and carry status `host-excluded`. This is a
  physical exclusion for that neighbour, not a missing input; the run can still
  publish, with zero probability for those slots.
- **Zero evidence from sampling.** An evaluated scenario can return `-inf`
  because no sampled draw produced a finite weight. The status may be
  `no-finite-likelihood-draws`. That is not proof that the prior support is
  empty; the ESS and convergence diagnostics decide whether the evidence set can
  be normalized.
- **Numerical failure.** A supported likelihood evaluation that fails is
  recorded as unresolved, never converted to zero likelihood. The run may defer.

At the low-level evidence-vector API, an unavailable slot is `-inf`, which would
enter a hand-built normalization as zero. The application engine does not
publish that vector; it refuses first.

## Scenario-level priors

Giacalone et al. (2021) call the period factor the scenario prior. It is applied
once, at normalization. The other entries below are population priors inside the
evidence; they are not fitted to the target.

| Prior | Applies to | Form and source constants | Source |
|---|---|---|---|
| Period | every target and neighbour slot | Planet: `P^1.5` for 0.1–10 d, `P^0` for 10–50 d. EB: `P^5.0` for 0.1–0.3 d, `P^0.5` for 0.3–50 d. Integrated over `P ± 0.1 d`, normalized over 0.1–50 d, clamped at the support edges. Evaluated at `2P` for `x2P`. | Giacalone et al. (2021) §2.4.1; `PERIOD_PRIOR_MIN_DAYS = 0.1`, `PERIOD_PRIOR_MAX_DAYS = 50`, `PERIOD_PRIOR_HALF_WIDTH_DAYS = 0.1` |
| Planet radius | every planet event (`TP`, `PTP`, `STP`, `DTP`, `BTP`, `NTP`) | Broken power law over 0.5–20 R⊕ with breaks at 3 and 6 R⊕: slopes `(0, -4, -0.5)` for hosts above 0.63 M☉ and `(0, -7, -0.5)` at or below it. The 0.63 M☉ boundary is a TREX modelling choice, not a published occurrence cutoff. | Giacalone et al. (2021) §2.4.2, Eqs. 10–11 |
| Inclination | every transiting or eclipsing event | Isotropic, uniform in `cos i`, restricted to the band that can transit. | Geometry; no fitted parameter |
| Eccentricity | every event | Planets: Kipping Beta(0.867, 3.030). Binaries: `p(e) ∝ e^-0.8` for `P <= 10 d`, and the period-dependent Eq. 17 law above 10 d. Evaluated at the scenario's model period (`2P` for `x2P`). | Kipping (2013), MNRAS 434, L51, <https://doi.org/10.1093/mnrasl/slt075>; Moe & Di Stefano (2017), ApJS 230, 15, <https://doi.org/10.3847/1538-4365/aa6fb6> |
| Eclipsing-pair mass ratio | every EB family | Moe & Di Stefano broken power law with a twin excess. The support is split at `q = 0.95`: `EB` draws `q < 0.95`, `x2P` draws `q >= 0.95`. | Moe & Di Stefano (2017); Giacalone et al. (2021) §2.4.4 |
| Bound-companion frequency and orbit | `P*`/`S*` | Primaries 0.075–0.60 M☉: Winters et al. (2019) branch, 0.324 companions per primary, log-normal projected separation peaked at 20 AU with 1.16 dex scatter, Reggiani & Meyer (2013) `q^0.25` mass ratio. Primaries above 0.60 M☉: Moe & Di Stefano (2017) Eq. 23 period cumulative and broken-power-law mass ratio; 0.60–0.80 M☉ is an explicit extrapolation. Frequency capped at one companion per primary; above 40 M☉ is refused. | Winters et al. (2019), AJ 157, 216, <https://doi.org/10.3847/1538-3881/ab05dc>; Reggiani & Meyer (2013), A&A 553, A124, <https://doi.org/10.1051/0004-6361/201321631>; Moe & Di Stefano (2017) |
| Companion mass, radius and temperature | `P*`/`S*`, EB secondaries | Mass from `q` times the primary mass; radius and temperature from the stellar mass–radius–temperature relation. A drawn bound companion is masked by the Mardling & Aarseth (2001) Eq. 90 stability criterion. | Torres, Andersen & Giménez (2010), A&ARv 18, 67, <https://doi.org/10.1007/s00159-009-0025-1>; Mardling & Aarseth (2001), MNRAS 321, 398, <https://doi.org/10.1046/j.1365-8711.2001.03974.x> |
| Background-star occurrence and position | `D*`/`B*` | Retained TRILEGAL stars at least as faint as the target. Expected count `λ = N_stars / A_sim × π r_limit²`, with `A_sim` converted from square degrees to square arcseconds; separation uniform in area out to `r_limit`; contrast mask removes draws the imaging would detect. | Girardi et al. (2005), A&A 436, 895, <https://doi.org/10.1051/0004-6361:20042352> |
| Resolved-neighbour stellar parameters | `N*` | Scene values, or the scene-supplied `stellar_parameter_distribution` marginalized per draw. No separate prior is invented for a neighbour without a distribution. | Scene contract; `resolved_neighbor_host`, `HostMeasure` |
| Resolved-neighbour spatial weight | `N*` | Neutral: exactly `0.0` log weight for every neighbour. The retained centroid diagnostic is not a calibrated source-assignment likelihood. | `spatial_prior` contract |

The period factor is the only prior applied at normalization. The planet radius,
eccentricity, inclination, mass-ratio, companion-frequency and background terms
enter inside each scenario's evidence. [How TREX works](how-trex-works.md) §2 and
§6 give the full derivation and the forward-model consequences; this page does
not repeat it.

## Which scenarios enter FPP and NFPP

```text
FPP  = Σ P_i  for every i except 0 (TP), 3 (PTP) and 9 (DTP)
NFPP = Σ P_i  for the neighbour slots of every neighbour (i >= 15)
```

So the planet-on-target scenarios, with or without dilution, count as the planet
explanation. A planet on a bound companion, a planet on a background star, and
a planet on a resolved neighbour are all false positives for the target. An
`NTP` is still a planet; it counts as a false positive because it does not
transit the target. When both are defined, `NFPP <= FPP` because every neighbour
scenario is included in FPP.

The source constants are:

- `PLANET_SCENARIO_NAMES = ("TP", "PTP", "DTP")` in the diagnostics module.
- `N_TARGET_SCENARIOS = 15` and `N_NEIGHBOR_SCENARIOS = 3`.
- `compute_fpp_nfpp` sums every entry except the three planet-on-target indices
  for FPP and every entry from index 15 onward for NFPP.

This definition follows Giacalone et al. (2021). It is a modelling choice: a
planet on a neighbouring star is still a planet, just not a planet transiting
the target.

## Ordering and output keys

- `TARGET_SCENARIO_SPECS` fixes the first 15 slots in the order above, and
  `TARGET_SCENARIO_NAMES` is the corresponding name tuple:
  `TP`, `EB`, `EBx2P`, `PTP`, `PEB`, `PEBx2P`, `STP`, `SEB`, `SEBx2P`, `DTP`,
  `DEB`, `DEBx2P`, `BTP`, `BEB`, `BEBx2P`.
- `NEIGHBOR_SCENARIO_NAMES = ("NTP", "NEB", "NEBx2P")` fixes the three slots per
  neighbour. `SCENARIO_NTP_OFFSET = 15`, `SCENARIO_NEB_OFFSET = 16` and
  `SCENARIO_NEBX2P_OFFSET = 17` are the first neighbour's offsets; each later
  neighbour adds three.
- `_SCENARIO_NAMES` covers only indices 0–14. Neighbour keys are synthesized as
  `N{index}`, which is why `summary.json` uses `N15`, `N16`, ... rather than
  `NTP`, `NEB`, ... in `effective_sample_sizes` and the per-scenario evidence
  rows. `TrexResult.period_log_priors` uses the same keys.
- `compute_fpp_nfpp` requires at least 15 entries and raises if the vector is
  shorter. It does not validate names or ordering.
- `scenarios.csv` has one row per scenario and host, in scenario-index order.
  Its columns are `ID`, `scenario`, `M_s`, `R_s`, `P_orb`, `inc`, `b`, `ecc`,
  `w`, `R_p`, `M_EB`, `R_EB`, `prob`. `ID` is the target TIC ID or the
  neighbour's `source_id`; `P_orb` is the model period (`2P` for `x2P`); `prob`
  is the normalized probability. [Outputs](outputs.md) defines every column and
  the representative-draw caveat.
- `scenarios.csv` is written only for a published result. A run with an
  unavailable scenario raises before writing it, so a missing `scenarios.csv`
  is not a table of zeros.

If you build an `lnZ` array by hand for a diagnostic, keep this order. A
misordered array does not raise; it silently assigns probabilities to the wrong
physical explanations.

## Reading a scenario probability

After any refinement, TREX adds the period prior and normalizes:

```text
P_i = exp(lnZ_i + lnπ_i - logsumexp_j(lnZ_j + lnπ_j))
```

`lnZ_i` is the prior-averaged likelihood of scenario `i`; `π_i` is the period
prior for that scenario. Stored evidences are prior-free with respect to the
period factor, so the factor is applied exactly once, at normalization. A
caller that supplies no period reproduces the prior-free arithmetic; the
production engine always supplies the period and records the per-scenario
factors in the result object's `period_log_priors`.

A high probability means that, within this fixed inventory and these priors, the
scenario explains the light curve better than the alternatives. It is not a
detection, a validation or a confirmation. `claim_eligible` is always `false`.
Zero probability and unavailability are not proofs of impossibility: zero can
mean that no sampled draw reached the likelihood, and unavailable means the
model was not evaluated. A high-probability `TP` is therefore still only a
conditional statement about the target.

## Common confusions

| Apparent reading | Correct reading |
|---|---|
| `x2P` doubles the observed signal period | It doubles the **model** period: the pair's true period is `2P`, so only one eclipse appears at the signal period. It applies only to EB families and selects the `q >= 0.95` branch. |
| `DTP` is a planet on a background star | `DTP` is a planet on the **target**, diluted by a background star. `BTP` is the planet on the background star. |
| `DEB` is an EB on a background star | `DEB` is an EB on the **target**, diluted by a background star. `BEB` is the EB on the background star. |
| `PTP` and `STP` are the same geometry | `PTP` has the planet on the **target**, with a bound companion diluting it. `STP` has the planet on the **bound companion**. |
| `NTP` is a planet, so it belongs with `TP` | `NTP` is a planet, but not on the target; it enters FPP and NFPP. |
| `N15` is a sixteenth target scenario | `N15` is the first neighbour's `NTP` slot. Target scenarios are indices 0–14. |
| A scenario with probability zero is impossible | Zero can come from host exclusion, no finite sampled weight, or a failed normalization. `host-excluded` is a definite physical exclusion for that neighbour and depth; zero evidence from sampling is not proof that the prior support is empty. |
| `NFPP` means "not a false positive" | It is the **nearby** false-positive probability: the neighbour-hosted part of FPP. |

## Related pages

- [Concepts](concepts.md): FPP, NFPP, priors, evidence and probability.
- [How TREX works](how-trex-works.md): the full prior set, forward model,
  evidence estimator and convergence routing.
- [Inputs](inputs.md): the scene, aperture fractions, contrast curve and
  background population each family needs.
- [Outputs](outputs.md): `scenarios.csv`, `summary.json`, scenario keys and
  failure records.
- [From TIC](from-tic.md): how acquisition supplies neighbours, metallicity,
  aperture fractions and the background population.
- [Glossary](glossary.md): `EB`, `blend`, `dilution`, `x2P` and related terms.
