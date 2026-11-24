<p align="center">
  <img src="https://raw.githubusercontent.com/shenfurkan/TREX/main/docs/images/trex-header.svg" alt="TREX" width="640">
</p>

TREX estimates how likely it is that a transit-like signal in TESS data comes
from something other than a planet orbiting the target star. For a signal you
have already found and measured, it reports a **false-positive probability
(FPP)** and a **nearby false-positive probability (NFPP)**. TREX is a standalone
Python package. It does not search for transits, and it does not validate or
confirm planets.

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

Before a signal is treated as a planet candidate, these explanations have to be
weighed against the planet explanation. TREX does that calculation.

## What TREX computes

TREX follows the scenario framework of Giacalone et al. (2021), the framework of
the TRICERATOPS tool: 15 scenarios in which the signal comes from the target, an
unresolved bound companion or an unresolved background star, plus three
scenarios for every resolved neighbour. For each scenario, TREX draws simulated
systems from astrophysical priors (1,000,000 draws per scenario in production),
measures how well each one reproduces your light curve, and averages the result
into an evidence. The normalized evidences become scenario probabilities.

- **FPP** is the total probability of every explanation except a planet on the
  target.
- **NFPP** is the part of FPP that comes from resolved neighbours.

Every result is conditional on what you supply: the light curve, the stellar
scene, the population models and the priors. It is not a statement that a
planet exists, and the output field `claim_eligible` is always `false`.

## What you need

1. A light curve of the signal with per-point flux uncertainties. The
   single-curve route takes a folded curve with times relative to the transit
   midpoint; calibrate the uncertainties at native cadence before folding.
2. A stellar scene: the target's mass, radius, temperature, TESS magnitude,
   parallax and metallicity; every resolved neighbour with its brightness
   difference and share of the aperture flux; a bound on unresolved companions;
   and a simulated background-star population for the field.
   `target.from_tic` can acquire this scene from public archives.
3. The measured period in days and depth in ppm.

TREX does not find the signal, choose its period or ephemeris, calibrate the
noise, localise the source or measure a high-resolution contrast curve. Those
steps come before TREX.

## Install

TREX 1.0.0 needs Python 3.14:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install trex==1.0.0
```

The unrelated pre-1.0 releases on PyPI have been yanked. The default install
runs JAX on the CPU. On Linux with an NVIDIA GPU, the `cuda12` extra adds CUDA
dependencies. See
[Installation](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/installation.md)
for supported platforms, optional dependencies and how to verify an install.

## First run

```bash
python -m trex.example trex-demo
```

Use a directory name that does not already exist. This offline check uses
invented data and 4,096 draws per scenario instead of 1,000,000, so it finishes
in minutes and its numbers say nothing about a real star. On success it writes
`trex-demo/result/summary.json` (FPP, NFPP, status and diagnostics) and
`trex-demo/result/scenarios.csv` (one row per scenario).

## Running a real target

- **From a TIC ID:** `target.from_tic` acquires the target's stellar scene from
  public archives, including its Gaia DR3 neighbours and a TRILEGAL background
  population. You supply the measured depth, the sectors, the exposure time and
  the noise weights. Start with the
  [Tutorial](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/tutorial.md),
  then [From a TIC ID](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/from-tic.md).
- **From prepared files:** `python -m trex.run` runs a single folded curve or
  joint observations from an input directory, can resume a paused run and
  reports the outcome as an exit code. See
  [Prepared runs](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/prepared-runs.md).

With the production budget, the synthetic example takes hours on a few CPU
cores and tens of minutes on a supported GPU; a real target can take longer.
These are expectations, not a benchmark.

## Reading the result

A completed run has `status: ok` in `summary.json`. If the evidence is missing,
insufficient or has not converged, TREX writes `failure.json` with a deferred
status and no FPP or NFPP instead of reporting a number. A deferral is a
statement about convergence, not a probability of zero. A small FPP is not a
detection, a validation or a confirmation.
[Read the output](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/outputs.md)
explains every field.

## Documentation

- [Overview](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/README.md)
- [Tutorial](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/tutorial.md)
- [Concepts: FPP, NFPP, priors, evidence and draws](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/concepts.md)
- [How TREX works](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/how-trex-works.md)
- [Scenarios](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/scenarios.md)
- [Read the output](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/outputs.md)
- [FAQ and troubleshooting](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/faq.md)
- [Relationship to TRICERATOPS](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/differences-from-triceratops.md)
- [Full documentation index](https://github.com/shenfurkan/TREX/blob/v1.0.0/docs/index.md)

## Help, credits and licence

For problems, [open an issue](https://github.com/shenfurkan/TREX/issues) with
the TREX version, command and error. Do not attach credentials or private data.
LLMs assisted with writing and revising TREX code; TREX does not use LLMs at
runtime.

TREX uses the TRICERATOPS scenario framework and does not require
TRICERATOPS to run. Cite the scenario framework as Giacalone et al. (2021),
*AJ* 161, 24, <https://doi.org/10.3847/1538-3881/abc6af>, and TREX as described
in [CITATION.cff](https://github.com/shenfurkan/TREX/blob/v1.0.0/CITATION.cff).

Thanks to [berrysauce](https://github.com/berrysauce/) for transferring the
`trex` package name on PyPI.

Copyright (C) 2026 Furkan Şen. Licensed under GPL-3.0-or-later; see
[LICENSE](https://github.com/shenfurkan/TREX/blob/v1.0.0/LICENSE).
