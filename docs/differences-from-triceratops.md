# How TREX differs from TRICERATOPS

TREX uses the scenario framework of TRICERATOPS 1.0.20 (Giacalone et al. 2021,
AJ 161, 24, DOI 10.3847/1538-3881/abc6af, hereafter G21): Bayesian evidences for
a fixed inventory of astrophysical scenarios that could produce a transit-like
signal, normalized into a false-positive probability (FPP) and a nearby
false-positive probability (NFPP). This page lists where TREX computes it
differently, why, and whether a matched-input comparison can remove the
difference. Each entry describes the current code.

Both probabilities are conditional on the inputs, assumptions and modelled
scenarios. TREX does not validate or confirm planets. Its `claim_eligible`
field is always `false` by design. For an introduction to FPP, NFPP, priors,
evidence, draws and effective sample size, see the [tutorial](tutorial.md).

"Matched input" means both engines receive the same light curve, uncertainty,
exposure, stellar parameters, neighbours and background population.

## What is unchanged

- **Scenario set.** The G21 scenarios for the target: TP, EB, EBx2P (target);
  PTP, PEB, PEBx2P (unresolved bound companion, planet or eclipse on the primary);
  STP, SEB, SEBx2P (unresolved bound companion as host); DTP, DEB, DEBx2P
  (unresolved background star dilutes the target; planet or eclipse on the
  target); BTP, BEB, BEBx2P (unresolved background star as host). For every
  resolved neighbour, the three scenarios NTP, NEB and NEBx2P.
- **Definitions.** FPP is the normalized probability of every scenario except the
  target-hosted planet scenarios TP, PTP and DTP. NFPP is the normalized
  probability of the resolved-neighbour scenarios.
- **Stellar relations.** TREX retains the reference mass-radius-temperature
  scenario spline nodes. Sharing the nodes does not independently calibrate
  them as stellar measurements. The related review is Torres, Andersen &
  Giménez (2010), A&ARv 18, 67, DOI 10.1007/s00159-009-0025-1.
- **Limb-darkening lookup** inside the TRICERATOPS grid bounds: the same packaged
  Claret (2017, A&A 600, A30, DOI 10.1051/0004-6361/201629705) table, the same
  nearest metallicity plane and the same nearest-node rule. The TRICERATOPS data
  file is vendored with its MIT licence notice; the Claret table carries its CDS
  attribution notice. The 49 metal-poor nodes absent from that table take
  coefficients from the Kostogryz et al. (2022, A&A 666, A60,
  DOI 10.1051/0004-6361/202243722) MPS-ATLAS supplement; no existing node is
  changed.

## Evidence estimator

| | TRICERATOPS | TREX |
|---|---|---|
| Monte Carlo draws | prior draws; non-transiting draws contribute zero weight | prior draws plus defensive importance proposals conditioned on the observed depth and duration, with importance weights (Hesterberg 1995, Technometrics 37, 185, DOI 10.1080/00401706.1995.10484303). The target integral is unchanged; the realized variance depends on the proposal. |
| Convergence check | none | the effective sample size (ESS) of evaluated scenarios' weights is reported. TREX may attempt refinement of eligible scenarios with ESS below 100, using nested sampling (Skilling 2006, Bayesian Analysis 1, 833, DOI 10.1214/06-BA127; dynesty, Speagle 2020, MNRAS 493, 3132, DOI 10.1093/mnras/staa278) or defensive importance sampling. The latter uses the scenario's native float64 JAX kernel on an admitted accelerator or on the CPU, and a float64 scalar CPU integrand when no native callback exists. Attempted refinement need not succeed. |
| Unconverged result | no equivalent ESS-based routing | if scenarios below the ESS floor still hold more than 10⁻³ of the normalized probability, TREX does not report an FPP. A sampled scenario with no finite weight can also force deferral. The realized residual is a routing heuristic, not a bound on FPP error. |
| Draw budget | user-chosen | 1,000,000 draws per scenario in production; smaller budgets are for software tests and do not start nested refinement. |

Opt-in experimental estimator: setting `TREX_NESTED_ESTIMATOR=blackjax`
selects the BlackJAX nested-slice route; dynesty remains the default. The
escalation route runs BlackJAX at 3,200 live points, with a logical-call
ceiling derived from the run's live count and dimension (4,000,000 at 3,200
live in four dimensions). The preregistered 4-D Gaussian known-Z control met
48/50 within 0.1 nat in each of two independent 50-seed cohorts (fresh local
and remote); the 1-D exponential control met 100/100 within 0.1 nat. The
reported error in the pooled Gaussian control was about 10% below the observed
run-to-run scatter, and the route used about eight times the logical likelihood
calls of the lower-live control. These synthetic controls do not calibrate real
scenes, multimodal likelihoods or GPU execution.

The adapter now implements native BlackJAX recovery at complete initialization
and rank-deletion boundaries. The live set, dead history, volume accounting,
continuation key and cumulative logical calls are restored together. Deadlines
are checked between native dispatches; an in-flight batch cannot be interrupted
by this API. A hard kill can replay work since the last durable boundary, never
another independent evidence draw. A retained numerical failure is not retried
as if it were a clean pause. Completed BlackJAX results use the same trusted,
input-bound recovery policy instead of a dynesty-shaped surrogate.
After an unclean recovery, unknown uncommitted physical calls are not guessed:
call counts are completed-work lower bounds and physical counts are unavailable.

Native callbacks also implement the characterized-host joint-row or continuous
stellar measure and the normalized Gaussian linear-baseline likelihood
(additive or multiplicative), including the determinant and aperture-coordinate
mapping. Discrete stellar rows retain their joint M/R/Teff association; continuous
mass/logg/logTeff labels assume independent supplied marginals, with radius
derived from the same mass/gravity draw. Interior tail quantiles are no longer
clamped to 1e-8. These are implementation changes awaiting remote numerical
parity and restore checks, not new calibration results or a default-estimator
change. Characterized hosts still exclude simultaneous unresolved populations
and the explicit photon-atmosphere model; use only the documented supported
model combinations, not a central-star substitute. The public engine still
refuses the BlackJAX + native-baseline combination; that path remains
dynesty-only.

Neither baseline marginalization nor host-label integration marginalizes
limb-darkening/model discrepancy or supplies missing stellar covariance.
A fixed Gaussian basis is not a general activity/red-noise model. Population
domain and field/extinction restrictions still apply; fallback count deficits
do not establish the direction of a normalized FPP change. No software check
establishes calibrated FPP or population coverage.

Why: a probability computed from a handful of effective draws can be wrong by
orders of magnitude while looking precise. A high ESS still cannot prove that a
mode missed by both the proposal and the nested sampler is absent.

Switchable for comparison: the production path cannot turn this off. The
low-level `compute_fpp_nfpp` can normalize without the ESS routing, for
diagnostics only.

## Numerics

- Evidences and probabilities are computed in log space (log-mean-exp,
  log-sum-exp). The released TRICERATOPS 1.0.20 code exponentiates its stored
  log-evidences and normalizes them in linear space; a very small evidence can
  underflow to zero. TREX reports NaN or +∞ as a numerical failure instead of
  turning it into a finite weight.
- FPP is the direct sum of the false-positive probabilities, not
  `1 − (P_TP + P_PTP + P_DTP)`, reducing cancellation for a very small FPP.
  TRICERATOPS 1.0.20 uses the subtraction form.
- Each scenario has its own random stream. Record the seed, inputs, software and
  hardware for reproducibility; the same seed does not promise bitwise identity
  across backends. The streams do not reproduce TRICERATOPS draws for that seed.

These choices are not switchable on the production path. Different random
streams also change a finite-draw estimate even without arithmetic failure.

## Likelihood

- **Secondary eclipse.** TREX evaluates both eclipses of an eclipsing binary at
  the observed times, with the secondary at its eccentric phase. TRICERATOPS
  constrains the EB models with a 1.5σ upper limit on the predicted secondary
  depth (G21 Sect. 2.4) instead of evaluating the secondary at the observed
  times. Consequence: if the secondary phase was never observed, TREX does not
  penalize the EB, so its FPP can be higher than TRICERATOPS' for the same data.
- **Per-body limb darkening.** The eclipsed secondary star uses its own
  atmosphere coefficients, not the primary's.
- **Uncertainties.** TREX accepts a scalar or one uncertainty per point. The
  caller must calibrate the uncertainties on native-time photometry before
  folding, including any supported time-correlated-noise correction. TREX does
  not automatically estimate β from a folded light curve. TRICERATOPS takes one
  scalar σ. Matched comparison: give both engines the same scalar σ.
- **Integration time.** TREX integrates the model over the native exposure and,
  for phase-folded binned data, over each bin's own width. TRICERATOPS integrates
  the exposure only. Matched comparison: unbinned data with
  `bin_integration_days = 0`.
- **Forward model.** The CPU Monte Carlo path uses batman (Kreidberg 2015,
  PASP 127, 1161, DOI 10.1086/683602; Mandel & Agol 2002, ApJ 580, L171,
  DOI 10.1086/345520); the native refinement kernel uses an equivalent JAX
  implementation on CPU and GPU. Argument of periastron is used with one
  convention throughout.
- **High eccentricity.** Where a scenario provides a native callback, every
  float64-admissible draw (`e < 1`) is evaluated by the batched JAX kernel;
  `batman` remains the scalar CPU reference and the fallback for a failed
  batch. The two paths agree to about `3.7e-6` relative flux at `e = 0.93`,
  and `batman` stops returning finite values for many orbits near
  `e = 0.98`, so the batched path owns the eccentric tail.

## Populations and priors

- **Orbital-period prior.** G21 Sect. 2.4.1 defines a period prior for each
  scenario. TREX applies it once, at normalization, and does not fold it into
  the stored evidences. The `x2P` scenarios evaluate the same prior at `2P`.
  Upstream implementation differences require a separate source-level
  comparison; this page does not claim an upstream defect.
- **Binary eccentricity.** Moe & Di Stefano (2017, ApJS 230, 15,
  DOI 10.3847/1538-4365/aa6fb6) Eq. 17 for P > 10 d instead of a fixed
  exponent; `e^-0.8` for P <= 10 d.
- **EB mass ratio.** The q < 0.95 (EB) and q ≥ 0.95 (EBx2P) partition of G21 is
  kept; q support follows the mass of each drawn host, not the target's.
- **Bound companions, solar-type and heavier primaries.** Moe & Di Stefano (2017)
  period distribution, period-dependent mass-ratio and twin fractions, and the
  Eq. 23 companion frequency corrected for the drawn q support, capped at one
  companion per primary as in G21. Primaries above 40 M☉ are refused rather than
  extrapolated.
- **Bound companions of M dwarfs.** A log-normal semi-major axis moment-matched
  to Winters et al. (2019, AJ 157, 216, DOI 10.3847/1538-3881/ab05dc), 0.324
  companions per primary, the field-binary q^0.25 law of Reggiani & Meyer (2013,
  A&A 553, A124, DOI 10.1051/0004-6361/201321631), and an isotropic projection
  of circular orbits for the imaging constraint.
- **Hierarchical stability.** Bound-companion draws are masked with the Mardling
  & Aarseth (2001, MNRAS 321, 398, DOI 10.1046/j.1365-8711.2001.03974.x) Eq. 90
  criterion instead of fixed period cuts.
- **Planet radius prior.** The G21 broken law for every planet scenario,
  including DTP, BTP and NTP; the M-dwarf branch uses M★ ≤ 0.63 M☉, the boundary
  G21 gives for its stellar relations.
- **Background stars.** The background prior is the natural log of the expected
  number of stars, density × πr² for the retained query radius. TRICERATOPS
  1.0.20 works from a fixed 0.1 deg² TRILEGAL field (G21 Sect. 2.4); TREX uses
  the retained query's own simulated area, so a query with a different area
  does not silently change the prior. Every retained TRILEGAL star (Girardi et
  al. 2005, A&A 436, 895, DOI 10.1051/0004-6361:20042352) can be drawn, and
  each star's own [M/H] sets its limb darkening.
- **Background field position and extinction (corrected upstream defects).**
  The released TRICERATOPS 1.0.20 source passes the target right ascension in
  degrees to the TRILEGAL 1.6 form's `eq_alpha` field, which TRILEGAL reads in
  hours (modulo 24). The simulated field can therefore sit away from the
  target: RA' = (RA mod 24) × 15°. TRICERATOPS 1.0.20 also leaves TRILEGAL's
  `extinction_infty` at the server's 0.0378 mag default for every field. TREX
  submits the target field in Galactic longitude and latitude in degrees and
  sets `extinction_infty` from the SFD98 map (Schlegel, Finkbeiner & Davis
  1998, ApJ 500, 525, DOI 10.1086/305772) with `A_V = 2.742 E(B-V)`
  (Schlafly & Finkbeiner 2011, ApJ 737, 103, DOI 10.1088/0004-637X/737/2/103,
  Table 6). That conversion is calibrated at high Galactic latitude with
  E(B-V)_SFD < 1; inside the plane it is an extrapolation. TREX refuses a
  TRILEGAL web record that lacks that field or extinction.

None of these is switchable to the TRICERATOPS form; that is why a
matched-input comparison reports joint differences rather than a single cause.

## Stellar light

- **Companion and background flux.** By default TREX uses radius² times the
  blackbody surface brightness integrated over the TESS photon response,
  instead of TRICERATOPS' empirical mass-to-flux spline. This is a blackbody
  surrogate, not a stellar atmosphere. A caller can explicitly bind a
  separate stellar-atmosphere model.
- **Eclipsing-binary dilution.** An eclipsing binary's light is assigned to the
  image it belongs to: the combined light of the pair for SEB and BEB, and the
  target or primary plus its secondary for DEB and PEB, with the bound companion
  or background star only diluting that image.
- **Limb darkening outside the TRICERATOPS grid.** A host inside a complete
  four-node cell of Claret (2017) table 25 is interpolated bilinearly in Teff
  and log g at its nearest metallicity plane. A host below 3500 K whose nearest
  plane is solar uses the Claret (2017) table 15 PHOENIX-COND coefficients where
  a complete cell exists. Anything else keeps the TRICERATOPS nearest-node rule
  and is flagged. Nothing is extrapolated and metallicity is not interpolated.

## Scene

- **Neighbours and aperture.** TREX takes a scene: Gaia-resolved neighbours and
  per-sector aperture flux fractions, with sector weights from the
  photometric uncertainties. TRICERATOPS queries the TIC and models the
  aperture with a Gaussian PSF. Matched comparison: supply both engines the same
  flux fractions.
- **Acquisition from a TIC ID.** Both engines start from a TIC ID and a list of
  sectors. TRICERATOPS 1.0.20 `target()` queries the TIC within `search_radius`
  pixels (10 pixels, 202.5 arcsec, by default), takes Tmag, mass, radius, Teff and
  parallax from the TIC, projects TIC J2000 positions through each sector's
  TESScut FFI WCS, and in `calc_depths` places a 0.75-pixel Gaussian on each star
  inside the caller's aperture or, without one, a 5x5-pixel box on the target;
  the flux ratios of the sectors are averaged with equal weight. TREX
  `target.from_tic` takes Gaia DR3 sources out to the radius at which a star's
  0.75-pixel Gaussian still reaches the pipeline aperture (farthest aperture
  corner plus four standard deviations, with a minimum search radius) propagated
  to each sector's observation epoch, the
  SPOC pipeline aperture and WCS from each sector's target-pixel file, Stassun et
  al. (2019) TESS magnitudes from Gaia photometry through the same 0.75-pixel
  Gaussian, and per-sector flux fractions weighted by each sector's measured
  inverse variance. Target parameters come from Gaia DR3 GSP-Phot and FLAME or the
  TIC v8 Cool Dwarf List, neighbour parameters from FLAME, the TIC, Gaia
  GSP-Phot, StarHorse, SHBoost or a MIST G+parallax measure, and each source's
  metallicity where available from APOGEE DR17, GALAH DR3, calibrated GSP-Spec
  or SHBoost. Every archive response is retained. Matched comparison: build the
  TREX scene, then supply the same flux fractions to TRICERATOPS.
- **Missing data and approximations.** A missing target metallicity, parallax
  or background population is not replaced by a solar default. An unavailable
  TRILEGAL query inside the LSST-simulation footprint can use that recorded
  fallback for the same field, with its limitation. On the default
  reference-grid path, a neighbour without its own metallicity can instead use
  the target's measured metallicity as an explicitly recorded approximation.
  This is not a measurement of the neighbour. With the optional physical
  stellar-light model, a resolved source needs its own atmosphere inputs.
  Without a measured contrast curve the scene must state the catalog resolution
  limit; no curve is invented. Missing required inputs can make a scenario
  unavailable or cause the run to be refused. An omitted scenario is not
  evidence that it is absent.
- **Contrast-curve band.** TRICERATOPS applies a supplied curve in its own
  filter (`calc_probs(filt=...)`); TREX evaluates companion contrasts in the
  TESS band, so `target.from_tic` and `target.from_scene` accept a measured
  TESS-band curve and refuse a non-TESS band rather than converting it.
  `calc_probs` does not read a contrast-curve file.
- **Metallicity provenance.** A target metallicity must come with its source,
  uncertainty and retrieval record. Values asserted by a caller are accepted
  only with a citation.

## Execution

- CPU and GPU execution, with batched JAX scenario evaluation where available.
  The full-budget synthetic quick-start produced FPP 1.082e-5 on four Kaggle
  CPUs and 1.08e-5 on an earlier T4 GPU record; that is one synthetic input,
  not a general backend-equality or accuracy claim.
- Checkpoints and resume: the prepared-input command can resume a run paused
  during refinement (exit code 75) with the same checkpoint, inputs and
  controls. dynesty and the opt-in BlackJAX route keep separate native
  checkpoints; the estimator is part of the resume binding.
- `python -m trex.run` distinguishes lifecycle exits: `0` completed with FPP
  and NFPP; `1` execution failure (including numerical or required-record
  publication failure); `2` input/checkpoint/output refusal; `3` unresolved
  evidence; `75` checkpoint pause. A returned non-ok result retains a non-ok
  summary, not a successful probability result.
- With a new output, the prepared-input command retains `failure.json` for
  handled input/checkpoint refusals and raised calculation failures or
  deferrals, including the original exception and diagnostics, when publication
  succeeds. Refusals leave existing outputs untouched. An existing empty output
  is resumable only with a retained Monte Carlo handoff; calculation failures
  and deferrals leave it empty for continuation. A pause creates no new output.
  Completed, failed and deferred outputs are never overwritten.
- Required-record creation, serialization or write errors propagate, chained
  to the original error, and exit `1`; there is no fallback output directory
  or guarantee of a file when publication fails. Direct library calls raise
  exceptions rather than returning CLI exit codes and do not automatically
  publish `failure.json`; callers explicitly use `target.write_outputs` for
  the target's retained failure record.

## Known limits

- Scenario populations not modelled (for example hierarchical triples beyond
  one companion) are absent from both numerator and denominator; the net
  effect on FPP has no fixed sign.
- The M-dwarf eclipsing-binary q law is the solar-type Moe & Di Stefano law;
  it is uncalibrated for M dwarfs.
- Limb-darkening coefficient uncertainty is not marginalized.
- The background-imaging mask cannot exclude companions inside the inner
  working angle or beyond the measured contrast curve.
- Background stars have no retained sky positions, so an individual simulated
  star cannot be cross-matched against a Gaia neighbour in the aperture.
- A recorded TRILEGAL-LSST fallback population can have same-cone counts below
  Gaia DR3 counts; its effect on a particular FPP is not measured.
- ESS and nested-sampling agreement cannot rule out a mode missed by both.
- From a TIC ID, aperture flux shares follow catalog magnitudes through a fixed
  0.75-pixel Gaussian; the TESS PRF varies with camera, CCD and field position.

## References

The citations above are the ones the implementation relies on. Additional
references for the estimator and priors appear in
[How TREX works](how-trex-works.md#12-references-used-by-the-implementation).
