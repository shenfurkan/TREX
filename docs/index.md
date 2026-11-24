# TREX documentation index

TREX `1.0.0`. TREX estimates conditional false-positive probabilities for
TESS transit signals using the scenario framework of Giacalone et al. (2021).
It does not validate or confirm planets.

Start with installation and the offline tutorial. For a measured signal,
use the TIC route to acquire a scene, or the prepared-run route if you already
have calibrated inputs. The output guide explains the result files.

## Start here

| Page | What it answers |
|---|---|
| [Overview](README.md) | Why a transit-like dip may not be a planet, what TREX computes, what a run looks like, install summary, first run, documentation map |
| [Installation](installation.md) | Python and platform requirements, pip and uv installs, dependencies, verification, reproducibility |
| [Tutorial](tutorial.md) | A synthetic run end to end, then the steps for a real target |
| [From a TIC ID](from-tic.md) | Which archives are queried, what is retained, and what is approximated |
| [Prepared runs](prepared-runs.md) | `python -m trex.run` for single-curve or joint observations, resume and exit codes |
| [Outputs](outputs.md) | How to read `summary.json`, `scenarios.csv` and `failure.json` |

## Using TREX

| Page | What it answers |
|---|---|
| [Inputs](inputs.md) | Units, formats and validation for the light curve, scene, contrast curve and background population |
| [Python API](api.md) | `target`, `TargetScene`, `run_trex_vetting`, `TrexResult`, constants, exceptions, environment variables |
| [From a scene](from-scene.md) | Building and binding your own scene and contrast curve |
| [Concepts](concepts.md) | Transit candidates, FPP/NFPP, priors, evidence, draws and ESS |
| [How TREX works](how-trex-works.md) | The complete calculation, assumptions and limits |
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
| [Synthetic quick-start, CPU completion](measurements/quickstart-cpu-2026-11-10.md) | One full-budget CPU run on synthetic input, with its FPP, NFPP and limits |
| [Earlier CPU pause](measurements/quickstart-cpu-2026-11-09.md) | The superseded pause of an earlier frozen snapshot, retained as history |

These are execution records for one synthetic input, not real-target results or
general performance claims.

## For agents and plain-text readers

- [llms.txt](llms.txt): a compact index for agent discovery, pointing at the
  Markdown pages above.
- On Fezadan, each page is available as raw Markdown under the same release
  path, for example [concepts.md](concepts.md).
- Pages use stable filenames; section anchors are ordinary Markdown headings.

The static `.md` files are the source of truth. No HTML rendering or content
negotiation is required to read them.

## Citation and licence

TREX is licensed under GPL-3.0-or-later; bundled catalogue tables keep their own
attribution notices. The scenario framework is Giacalone et al. (2021), *AJ*
161, 24, <https://doi.org/10.3847/1538-3881/abc6af>. Cite TREX as described in
`CITATION.cff` in the TREX repository.
