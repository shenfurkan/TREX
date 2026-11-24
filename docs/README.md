# TREX

TREX estimates how likely it is that a transit-like signal in TESS data comes
from something other than a planet orbiting the target star. For a signal you
have already found and measured, it reports a **false-positive probability
(FPP)** and a **nearby false-positive probability (NFPP)**.

## Why this question comes up

TESS records the brightness of stars for weeks at a time. When a planet passes
in front of its star, the star dims slightly at a regular interval: a transit.
A dip of that shape is not proof of a planet, because several other
arrangements of stars produce similar dips:

- the target is itself an eclipsing binary, two stars passing in front of each
  other;
- a fainter star whose light falls in the same TESS pixels hosts the eclipse or
  transit. It can be an unresolved companion bound to the target or an
  unrelated star far behind it, and the target's own light makes its dip look
  shallower;
- a nearby star resolved in Gaia hosts the eclipse or transit, and part of its
  light falls in the target's aperture.

TREX compares a fixed inventory of such explanations with the measured light
curve and reports how much probability each one carries. FPP is the total
probability of every explanation except a planet on the target; NFPP is the
part of it that comes from resolved neighbours.

The output is conditional on the light curve, the stellar scene, the population
models, the priors and the scenario list you supply. It is not a statement that
a planet exists. TREX does not validate or confirm planets, and its
`claim_eligible` field is always `false` by design.

## Scope and standalone boundary

TREX is a standalone Python package. It does not import a survey pipeline, and
it does not need one to run: the two entry paths acquire a public TESS scene
from archives or accept an externally calibrated `TargetScene`.

- **TREX does** estimate conditional FPP and NFPP for a signal you have already
  identified and measured, evaluate the implemented astrophysical scenarios,
  retain its inputs and diagnostics, and expose a Python API and a prepared-file
  command line.
- **TREX does not** search for transits, choose a period or ephemeris, calibrate
  the light-curve noise for you, measure source localization or a
  high-resolution contrast curve, or turn a probability into a validation or
  confirmation. A small FPP is not a detection.
- **The standalone boundary** is the `trex` package and its public imports
  (`trex.target`, `trex.scene`, `trex.engine`, and the documented helpers). A
  host application may call them, but TREX does not depend on the host's
  workspace layout, private records or candidate metadata.

This documentation describes TREX `1.0.0`, the version declared in the
package metadata. If a page and the source disagree, the source is
authoritative. TREX is licensed under GPL-3.0-or-later; bundled catalogue tables
keep their own attribution notices.

## What a run looks like

A run needs three things:

1. Prepared photometry with per-point flux uncertainties. The single-curve
   route takes a folded curve with times relative to transit midpoint; calibrate
   the uncertainties at native cadence before folding. The joint-observation
   route takes separate observations with a shared TDB time reference,
   exposure information and per-band light models, without folding them.
2. A stellar scene: the target's mass, radius, temperature, TESS magnitude,
   parallax and metallicity; every resolved neighbour with its brightness
   difference and share of the aperture flux; a bound on unresolved companions;
   and a simulated background-star population for the field.
3. The measured period in days and depth in ppm. Supply exposure and per-sector
   weights for a single curve, or the epoch in the shared time reference for
   joint observations. See [Prepared runs](prepared-runs.md) for the file layouts.

TREX draws simulated systems from astrophysical priors for each scenario and
computes how well each draw reproduces the light curve. It averages the
likelihood over the draws to get a scenario **evidence**, applies the scenario
period prior, and normalizes the evidences into probabilities. FPP sums every
modelled explanation except a planet on the target. NFPP sums the explanations
hosted by resolved neighbours.

## Install

Install TREX 1.0.0 from PyPI with Python 3.14:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install trex==1.0.0
```

A clone that contains `uv.lock` can use `uv sync` instead. The default install
runs JAX on the CPU. On Linux with an NVIDIA GPU and a compatible driver, the
`cuda12` extra adds CUDA dependencies; it does not guarantee that JAX detects
the GPU. See [Installation](installation.md) for requirements, platform limits
and verification.

## First run

This offline example uses invented data, a deliberately large flux uncertainty
and a reduced draw budget. It exercises the real engine and writes the same
kinds of files as a real run, but its numbers are not a result about a star. Run
it where `trex-demo` does not already exist; the example refuses to overwrite an
earlier run.

```bash
python -m trex.example trex-demo
python - <<'PY'
import json
from pathlib import Path
p = Path("trex-demo/result")
report = json.loads((p / "summary.json").read_text())
print("status:", report["status"])
print("FPP:", report["FPP"], "NFPP:", report["NFPP"])
print("scenarios:", p / "scenarios.csv")
PY
```

The example uses 4,096 draws per scenario. Production uses 1,000,000. At the
reduced budget the probabilities mean nothing; the files show what a run
produces on success or failure.

## How to read this documentation

Start with [Installation](installation.md) and the [Tutorial](tutorial.md),
or pick a task below. [Concepts](concepts.md) explains FPP and NFPP;
[How TREX works](how-trex-works.md) describes the calculation and its limits.

If you already know the field and need an interface:

- [Inputs](inputs.md): units, formats and validation rules.
- [Outputs](outputs.md): every field TREX writes.
- [Python API](api.md): signatures, returned objects and errors.
- [Prepared runs](prepared-runs.md): `python -m trex.run` for single-curve or joint observations.
- [Scenarios](scenarios.md): the 15 target and 3 per-neighbour scenarios.
- [Module and function reference](modules.md): what each module and public function does.
- [From a TIC ID](from-tic.md): the acquisition path and its sources.
- [From a scene](from-scene.md): the offline path and contrast curves.
- [Background population](background-population.md): TRILEGAL and the fallback.
- [FAQ and troubleshooting](faq.md): failure messages and what to do next.
- [Glossary](glossary.md): terms used across these pages.

[The documentation index](index.md) lists every page with a one-line
description. The Fezadan-hosted HTML viewer renders
these Markdown files for human readers; each Markdown file also stays available
at a fixed URL under `/docs/` for agents and plain-text readers.

## Common tasks

| I want to... | Start with |
|---|---|
| install TREX and check that it runs | [Installation](installation.md), then the first run above |
| analyse a TESS target from a TIC ID | [Tutorial](tutorial.md), then [From a TIC ID](from-tic.md) |
| use my own stellar scene | [From a scene](from-scene.md), with [Inputs](inputs.md) open |
| run from prepared files | [Prepared runs](prepared-runs.md) |
| read a result | [Outputs](outputs.md), then [Concepts](concepts.md) |
| understand how a probability was computed | [How TREX works](how-trex-works.md) |
| find what a function or module does | [Module and function reference](modules.md) |
| fix an error or a deferred run | [FAQ and troubleshooting](faq.md) |
| compare TREX with TRICERATOPS, the TESS false-positive tool whose scenario framework TREX follows | [Source notes](differences-from-triceratops.md) |

## Citation and provenance

The scenario framework is Giacalone et al. (2021), *AJ* 161, 24,
<https://doi.org/10.3847/1538-3881/abc6af>. That paper defines the 15 target
scenarios, the three scenarios per resolved neighbour, and the FPP and NFPP
definitions TREX uses. TREX computes them with its own estimator, population
models and scene handling; [Relationship to TRICERATOPS](differences-from-triceratops.md)
records the source-level differences and the adopted upstream corrections.

Please cite TREX as described in `CITATION.cff` in the TREX repository, and cite
the Giacalone et al. paper for the scenario framework. Individual model
components carry their own references, listed on the page that uses them.

## Reporting a TREX result

A report should let another reader reconstruct the calculation and judge its
limits. State:

- the TREX version or commit, the installed dependency versions and the
  hardware/backend;
- the light curve, its source, sectors, quality mask, cadence and the noise
  calibration actually applied;
- the scene, including the target parameters and provenance, every resolved
  neighbour, the aperture fractions, the unresolved bound and the background
  population (with its source and fallback status);
- the period, epoch, depth, duration, exposure, bin width, draw budget
  (`N = 1,000,000` for production), `nsamples` and seed;
- FPP, NFPP, the per-scenario probabilities and the ESS/refinement diagnostics;
- which scenarios dominate and which are unavailable, with their reasons;
- the conditional nature of the result and the `claim_eligible: false` field.

Do not describe a small FPP as a detection, validation or confirmation. A
deferral is a statement about convergence, not a probability of zero.

## Runtime

A full-budget synthetic run takes hours on a few CPU cores and tens of minutes
on a supported GPU. A real target can take longer, and a laptop may be too slow
for the production budget. Runtime depends on the light curve, the scene,
whether low-ESS scenarios need refinement, and the hardware. The 4,096-draw
example finishes in minutes; production uses 1,000,000 draws per scenario.

These are expectations, not a benchmark. One synthetic input does not establish
general accuracy or CPU/GPU equivalence. The retained records are the
[full-budget CPU completion](measurements/quickstart-cpu-2026-11-10.md) and the
[earlier CPU pause](measurements/quickstart-cpu-2026-11-09.md); both use invented
data.
