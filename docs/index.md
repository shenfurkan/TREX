# TREX documentation index

TREX `1.0.0`. TREX estimates conditional false-positive probabilities for
TESS transit signals using the scenario framework of Giacalone et al. (2021).
It does not validate or confirm planets.

Every page is a plain Markdown file at a stable URL under `/docs/`, served as
UTF-8 text. The HTML viewer in `site/` (published at the site root) renders
those files for human readers; the Markdown files remain the source of truth
and need no build step.

If you are new to TREX, read the four pages under "Start here" in order: they
build from the astronomy and probability vocabulary to the full calculation.
The pages under "Using TREX" and "Reference" can be read in any order, and each
one links back to the concepts it uses.

## Start here

| Page | What it answers |
|---|---|
| [Overview](README.md) | What TREX computes, what a run looks like, install summary, first run, documentation map |
| [Concepts](concepts.md) | What a transit candidate is, why dips have other causes, FPP/NFPP, priors, evidence, draws, ESS, units |
| [Tutorial](tutorial.md) | A synthetic run end to end, then the steps for a real target |
| [How TREX works](how-trex-works.md) | The complete calculation: priors, forward model, likelihood, evidence, normalization, refinement, limits |

## Using TREX

| Page | What it answers |
|---|---|
| [Installation](installation.md) | Python and platform requirements, pip and uv installs, dependencies, verification, reproducibility |
| [Inputs](inputs.md) | Units, formats and validation for the light curve, scene, contrast curve and background population |
| [Outputs](outputs.md) | Every field of `summary.json`, `scenarios.csv` and `failure.json`, plus checkpoints and plots |
| [Python API](api.md) | `target`, `TargetScene`, `run_trex_vetting`, `TrexResult`, constants, exceptions, environment variables |
| [Prepared runs](prepared-runs.md) | `python -m trex.run`, its three input files, resume behaviour and exit codes |
| [From a TIC ID](from-tic.md) | Which archives are queried, in what order, what is retained, and what is approximated |
| [From a scene](from-scene.md) | Building and binding your own scene and contrast curve |
| [Background population](background-population.md) | TRILEGAL, the population file, the recorded fallback and its interpretation limits |

## Reference

| Page | What it answers |
|---|---|
| [Scenarios](scenarios.md) | The 15 target scenarios and 3 per-neighbour scenarios: configuration, priors, availability, FPP membership |
| [Module and function reference](modules.md) | What each module and public function does, and where it sits in a run |
| [Glossary](glossary.md) | Every term and threshold in one place |
| [FAQ and troubleshooting](faq.md) | Common questions, error messages and what to check |
| [Relationship to TRICERATOPS](differences-from-triceratops.md) | Source-level differences and the adopted corrections, for provenance |

## Measurement records

| Page | What it records |
|---|---|
| [Synthetic quick-start, CPU completion](measurements/quickstart-cpu-2026-11-10.md) | One full-budget synthetic run on four Kaggle CPUs, with its FPP, NFPP and limits |
| [Earlier CPU pause](measurements/quickstart-cpu-2026-11-09.md) | The superseded pause of an earlier frozen snapshot, retained as history |

These are execution records for one synthetic input, not real-target results or
general performance claims.

## For agents and plain-text readers

- [llms.txt](llms.txt): a compact index for agent discovery, pointing at the
  Markdown pages above.
- Every page is retrievable directly, for example
  `https://trex.fezadan.org/docs/concepts.md`.
- Pages use stable filenames; section anchors are ordinary Markdown headings.

The static `.md` files are the source of truth. No HTML rendering or content
negotiation is required to read them.

## Citation and licence

TREX is licensed under GPL-3.0-or-later; bundled catalogue tables keep their own
attribution notices. The scenario framework is Giacalone et al. (2021), *AJ*
161, 24, <https://doi.org/10.3847/1538-3881/abc6af>. Cite TREX as described in
`CITATION.cff` in the TREX repository.
