# Glossary

Every term used in these pages, with the meaning TREX gives it.

**Aperture**: the set of pixels summed to produce a light curve. In TESS, the
pipeline defines it; its shape varies by sector.

**Aperture flux fraction**: a source's share of the total in-aperture flux in
one sector, in `[0, 1]`. TREX uses it to renormalize the light curve onto each
candidate host and to model dilution. It is a throughput measurement, not a
magnitude ratio. See [From a scene](from-scene.md#aperture-fractions-and-unresolved-companions).

**Background star**: an unrelated star along the line of sight whose light can
enter the aperture. TREX draws these from a simulated population for the field.
See [Background population](background-population.md).

**Baseline**: an optional linear nuisance model with a Gaussian prior on its
native-time basis coefficients, marginalized jointly with the transit. It
requires dynesty, a single-sector scene and unbinned times.

**Bayes' theorem**: the rule that turns a prior and a likelihood into a
posterior probability. TREX uses its evidence form, `P(S|D) ∝ P(S) P(D|S)`, to
compare scenarios.

**BJD / BTJD / TDB**: barycentric Julian date; the TESS offset form, equal to
BJD minus 2457000; and barycentric dynamical time, a barycentric timescale that
runs uniformly apart from small relativistic corrections. BJD and BTJD both count
in it and differ only by that offset.

**BlackJAX**: a JAX library for Bayesian inference. TREX can use its
nested-slice sampler as an experimental opt-in estimator. It keeps its own
native checkpoint, is not the default and is not calibrated on real scenes.

**Blend**: light from more than one source entering the same aperture. A blend
can dilute a transit and make it look shallower, or produce a dip from a source
other than the target.

**Candidate**: a transit-like signal that has not been shown to be a planet.
Also called a transit candidate.

**Checkpoint**: the directory where TREX stores the state needed to resume a
paused refinement. For `python -m trex.run` it holds `monte-carlo.json` plus the
nested-sampler state files. See
[Outputs](outputs.md#4-checkpoints-and-resume).

**Claim eligibility**: TREX's field stating whether a result may be used as a
planet claim. It is always `false` by design.

**Contrast curve**: a measured detection limit versus angular separation, given
as separation in arcsec and limiting Δmag in mag. TREX accepts TESS-band curves
only.

**Convergence**: the condition that a numerical estimate has enough effective
samples, and enough agreement between independent estimators, to be reported.
TREX routes on ESS and on nested-sampling acceptance checks. See
[How TREX works](how-trex-works.md#9-convergence-refinement-and-deferral).

**Deferral**: a refusal to publish FPP/NFPP because the evidence set did not
converge. A deferral is not a probability of zero. See
[How TREX works](how-trex-works.md#9-convergence-refinement-and-deferral).

**Dilution**: the reduction of an observed transit depth when the eclipsed
source is not the only source in the aperture. If a fraction of the light comes
from elsewhere, the fractional dip is reduced by that fraction.

**Draw**: one simulated parameter set used to estimate a scenario's evidence.
TREX mixes draws from the prior with draws from defensive proposals and weights
them accordingly; a draw is not a posterior sample.

**dynesty**: the default nested-sampling library TREX uses to refine low-ESS
scenarios (Speagle 2020). It runs at 400 live points with an evidence tolerance
of 0.05 nats by default.

**EB**: eclipsing binary, two stars passing in front of each other. `EBx2P`
means the pair's true period is twice the observed signal period.

**Eccentricity (`e`)**: the ellipticity of an orbit, from 0 (circular) to below
1. It changes the star–body separation at conjunction and the timing of a
secondary eclipse.

**Effective sample size (ESS)**: `(Σw)² / Σw²`, a count between 1 and `N` that
measures how evenly the draw weights are distributed. Low ESS means the estimate
depends on few draws. See
[Concepts](concepts.md#5-draws-effective-sample-size-and-convergence).

**Evidence (`Z`)**: the prior-averaged likelihood of a scenario,
`Z = ∫ L(θ) π(θ) dθ`. It measures how well a scenario explains the data across
its allowed parameters. TREX stores it as `ln Z` in nats.

**Exposure time**: the duration of one cadence, in days. The forward model
integrates the transit over it.

**False positive**: a signal that resembles a planet transiting the target but
has another cause.

**Flux**: the measured brightness of the target, normalized so that the
out-of-transit level is about 1. TREX works in relative flux.

**FPP**: false-positive probability. The sum of the normalized probabilities of
every modelled scenario except a planet on the target (`TP`, `PTP`, `DTP`). See
[Concepts](concepts.md#4-from-evidence-to-fpp-and-nfpp).

**Gaussian likelihood**: the likelihood form TREX uses,
`lnL = −0.5 Σ (F_obs − F_model)²/σ²`. It assumes independent Gaussian errors
with the supplied scalar or per-point `σ`.

**Impact parameter (`b`)**: the projected separation of the transiting body and
the stellar centre at conjunction, in units of the stellar radius. A central
transit has `b ≈ 0`; a grazing transit has `b` near `1 + k`.

**Inclination (`i`)**: the angle between the orbital plane and the plane of the
sky, in degrees. Transits require `i ≈ 90°`.

**Importance sampling**: estimating an integral by drawing from a proposal
distribution `q` instead of the prior `p` and weighting each draw by `p/q`.
TREX uses defensive mixtures for the occultor radius and inclination. See
[How TREX works](how-trex-works.md#7-estimating-the-evidence).

**Light curve**: a time series of brightness measurements.

**Likelihood (`L`)**: the probability of the observed data given a model and its
parameters.

**Limb darkening**: the dimming of a star's apparent brightness toward its edge,
which changes the shape of a transit. TREX uses a quadratic law with
coefficients `u1`, `u2`.

**Metallicity**: the abundance of elements heavier than helium, usually
`[Fe/H]` or `[M/H]` in dex. TREX requires a measured target value with
provenance.

**Nat**: natural-log unit of information. A log-evidence difference of 1 nat is
a likelihood ratio of `e`.

**Neighbour**: a resolved star near the target, usually a Gaia source, whose
light may enter the aperture.

**Nested sampling**: an integration method that maintains live points sampled
from the prior and accumulates evidence as the prior volume shrinks (Skilling
2006). See [How TREX works](how-trex-works.md#9-convergence-refinement-and-deferral).

**NFPP**: nearby false-positive probability. The sum of the normalized
probabilities of the resolved-neighbour scenarios (`NTP`, `NEB`, `NEBx2P`). See
[Concepts](concepts.md#4-from-evidence-to-fpp-and-nfpp).

**Nuisance parameter**: a parameter that is not of direct interest but affects
the data, such as an extra noise scale.

**Period prior (scenario prior)**: the scenario factor from Giacalone et al.
(2021) Section 2.4.1, describing how likely a transiting planet or eclipsing
binary is to have the period that produces the observed signal. TREX applies it
once at normalization; an `x2P` scenario evaluates it at `2P`.

**Posterior**: the probability distribution of parameters given the data.
TREX's Monte Carlo draws are not a posterior sample, and its representative
draws are not posterior summaries.

**ppm**: parts per million, `10⁻⁶` of the out-of-transit flux.

**Prior**: an assumption about a parameter or scenario weight before the data.
TREX has population priors (inside the evidence) and a period prior (applied at
normalization).

**Proposal**: the distribution importance sampling draws from. TREX's
radius/inclination proposals are defensive mixtures that keep half the draws on
the prior; the refinement stage uses a one-tenth prior share.

**Representative draw**: the single Monte Carlo draw with the highest
likelihood for a scenario, written to `scenarios.csv` to show the model shape.
It is not a posterior summary.

**Scenario**: one modelled explanation for the signal and its surroundings. See
[Scenarios](scenarios.md).

**Scenario name**: a prefix plus event code. A bare name (`TP`, `EB`, `EBx2P`)
is target-hosted. `P` marks an unresolved bound companion whose light dilutes
the target; `S` moves the event to that companion; `D` dilutes with an
unresolved background star; `B` makes the background star the host; `N` is a
resolved neighbour. `TP` is a transiting planet, `EB` an eclipsing binary, and
`x2P` doubles the model period. The full table is in
[Scenarios](scenarios.md#naming).

**Sector**: a TESS observation segment, roughly 27 days. Photometry from
different sectors has different cadence, aperture and noise.

**Sector weight**: the sum of inverse variances of a sector's native cadences.
TREX uses it to combine per-sector aperture fractions.

**TESS**: Transiting Exoplanet Survey Satellite.

**TIC**: TESS Input Catalog. A TIC ID identifies a star.

**TOI**: TESS Object of Interest, a TESS signal flagged for further review.

**Transit**: the dip in a star's light when a body passes in front of it.

**TRILEGAL**: a stellar-population simulation used to generate the background
star population for a field (Girardi et al. 2005). See
[Background population](background-population.md).

**Unavailable scenario**: a scenario that cannot be evaluated because a required
input or model is absent. It is listed in `unavailable_scenarios` and the engine
defers instead of publishing a normalization over a reduced inventory; this is
not the same as a computed probability of zero.

**x2P**: a scenario in which the modelled period is twice the observed signal
period, representing an eclipsing pair whose two eclipses are separated by the
signal period.

## Constants and thresholds

| Term | Value | Meaning |
|---|---:|---|
| Production draw budget | 1,000,000 | Draws per scenario |
| ESS floor | 100 | Below this a scenario is eligible for refinement |
| Sub-floor routing allowance | 10⁻³ | If sub-floor scenarios still hold more normalized probability after refinement, TREX defers |
| FPP diagnostic threshold | 0.01 | Routing and display screen only; does not enable a claim. Giacalone et al. (2021) used 0.015 for their published classification |
| NFPP diagnostic threshold | 0.001 | Routing and display screen only; does not enable a claim |
| Unresolved-scene default | 2.2 arcsec | Gaia DR2 close-pair limit used when a measured contrast curve is supplied without a resolution limit |
| Default exposure supersampling | 20 | Sub-exposure midpoints |
| Default refinement wall budget | 540 minutes | Operational ceiling; a host deadline can pause earlier |

## Related pages

- [Concepts](concepts.md): the definitions in context.
- [How TREX works](how-trex-works.md): the calculations.
- [Scenario reference](scenarios.md): every scenario name.
- [Outputs](outputs.md): the report fields and checkpoint files.
- [How TREX differs from TRICERATOPS](differences-from-triceratops.md): implementation differences.
