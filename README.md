# TREX

TREX estimates the **false-positive probability (FPP)** and the **nearby
false-positive probability (NFPP)** of a TESS transit signal. It is a
standalone package: it either acquires a public TESS scene from archives or
accepts an externally calibrated scene, and it does not import a survey
pipeline or depend on a host workspace. It is not a transit search, a
noise-calibration service or a planet-validation tool.

Terms used below:

- A **transit** is a dip in a star's light when another body passes in front
  of it. A **false positive** is a signal that looks like a planet transiting
  the target star but has another cause, such as an eclipsing pair of stars.
- A **scenario** is one modelled explanation of the signal. FPP sums the
  probabilities of the scenarios that are not a planet on the target; NFPP
  sums those on resolved nearby stars.
- A **prior** describes assumed parameter or scenario weights before the
  light-curve evidence. The **evidence** measures how well a scenario,
  averaged over its parameter choices, accounts for the data; TREX estimates it
  from simulated **draws** of those parameters.
- **Effective sample size (ESS)** is how many equally weighted draws would
  have comparable weight concentration.

These probabilities depend on the inputs, scenarios and priors; they are not a
guarantee that a planet exists. TREX does not validate or confirm planets, and
its `claim_eligible` field is always `false` by design.

## Installation

TREX needs Python 3.12. The tested platform is Linux x86-64. macOS arm64
dependency resolution is recorded in `uv.lock` but no native install or run was
performed; treat it as expected rather than verified. Native Windows is out of
scope; use WSL. Intel macOS is unsupported because the required
`batman-package` and `jaxlib` releases publish no matching wheels.

TREX 1.0 is installed from GitHub, not PyPI. The distribution and import name
are both `trex`, but the existing PyPI project named `trex` is unrelated to
this package, so do not use `pip install trex`. Install the `v1.0.0` release
into a Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/shenfurkan/TREX@v1.0.0"
```

or from a clone:

```bash
git clone --branch v1.0.0 https://github.com/shenfurkan/TREX
cd TREX
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The default installation uses JAX on the CPU. On a Linux machine with an NVIDIA
GPU and compatible driver, install the optional CUDA 12 dependencies with
`python -m pip install '.[cuda12]'` from a clone, or
`python -m pip install "trex[cuda12] @ git+https://github.com/shenfurkan/TREX@v1.0.0"`
from the Git URL. Installing the extra does not guarantee that JAX will detect
the GPU; see the [FAQ](https://github.com/shenfurkan/TREX/blob/main/docs/faq.md).
For reproducible research, retain the versions actually installed along with
the inputs and run settings. Compatibility across every allowed dependency
version has not been established.

Source files live directly in `src/`; setuptools maps that directory to the
installed `trex` package. For checkout development, use `uv sync --locked`
before `uv run pytest`; adding `src/` to `PYTHONPATH` is not an installation.

## First run

This offline example uses invented data and a reduced draw budget. It exercises
the real engine; its numbers are **not** a result about a star. Run it from any
directory where `trex-demo` does not already exist:

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

## Read the output

`summary.json` contains the run status, FPP, NFPP, per-scenario ESS and
refinement records. `scenarios.csv` contains each scenario's probability and a
representative model draw. Read both, including unavailable-scenario reasons,
before interpreting a result. A low ESS or missing scenario can make a result
unreliable. A high ESS cannot rule out an explanation that was never sampled.
An error may leave `failure.json` instead of a summary. See the
[tutorial](https://github.com/shenfurkan/TREX/blob/main/docs/tutorial.md) for
the example's inputs, the probability definitions and the path from a TESS
Input Catalog (TIC) ID to a real analysis.

The nested-evidence refinement is dynesty by default. The opt-in BlackJAX
route is experimental; its calibration and limits are in the
[1.0.0 release notes](#release-notes).

## CPU and GPU

Both CPU and GPU paths run the same engine. At the production budget of
1,000,000 draws per scenario, measured full-budget runs of the synthetic
quick-start are:

| Run | Hardware | Wall time | FPP | Status |
|---|---|---:|---:|---|
| Full-budget synthetic quick-start | 4 Kaggle CPUs | 18,312 s | 1.082e-5 | ok |
| Full-budget synthetic quick-start, earlier record | Kaggle T4 GPU | 2,273 s | 1.08e-5 | ok |

The two records use different frozen code snapshots and are not a controlled
speed benchmark. Their FPPs agree to the recorded digits
(1.0819255685882508e-5 on CPU and 1.0819255685792538e-5 on the T4 record), but
this is one synthetic input, not a general accuracy claim. The CPU run is
recorded in [docs/measurements/quickstart-cpu-2026-11-10.md](https://github.com/shenfurkan/TREX/blob/main/docs/measurements/quickstart-cpu-2026-11-10.md).

Where a scenario provides the native callback, the refinement evaluates the
same batched float64 JAX kernel on CPU and GPU; scenarios without one use a
float64 scalar CPU integrand. Runtime depends on the light curve, the stellar
scene, the hardware and whether low-ESS scenarios need refinement. A real
target can take longer than the synthetic example.

The notebooks retained in `examples/` were last executed at a reduced budget.
No full-budget notebook re-execution is recorded in this repository, so their
saved outputs are evidence for the software path only, not for a real-target
result or a production timing.

## Release notes

### 1.0.0

First public release. It provides:

- **TIC-to-scene acquisition.** `target.from_tic` acquires the target and
  neighbour scene, background population and aperture information from public
  archives, retains every archive response, and returns a target bound to a
  validated `TargetScene`. `target.from_scene` binds an externally measured
  scene.
- **TRILEGAL field and extinction.** The background field is submitted in
  Galactic longitude and latitude in degrees. TREX sets the field's
  `extinction_infty` from the SFD98 map with `A_V = 2.742 E(B-V)` instead of
  the form's 0.0378 mag default. If TRILEGAL is unavailable, the recorded
  TRILEGAL LSST-simulation fallback serves the same 0.1 deg² field inside its
  footprint; the next acquisition retries TRILEGAL.
- **Measured contrast curves.** `target.from_tic` and `target.from_scene`
  accept a measured TESS-band contrast curve (a two-column file or an array
  pair) and refuse a curve in any other band.
- **CPU refinement.** On a host without an admitted accelerator, the MC-pilot
  defensive-importance stage runs the scenario's native callback as a batched
  float64 JAX kernel on the CPU, or a float64 scalar integrand where no native
  callback exists. Where that callback exists, the same native kernel is used
  on CPU and GPU.
- **Report.** `summary.json` carries `scenario_evidence_refinement`, one row
  per scenario with the Monte Carlo and refined ESS, status and reason.
- **Opt-in BlackJAX.** dynesty remains the default. Setting
  `TREX_NESTED_ESTIMATOR=blackjax` selects the experimental BlackJAX
  nested-slice route at 3,200 live points, with a logical-call ceiling derived
  from the run's live count and dimension (4,000,000 at 3,200 live in four
  dimensions). The preregistered 4-D Gaussian known-Z control met 48/50 within
  0.1 nat in each of two independent 50-seed cohorts (fresh local and remote);
  the 1-D exponential control met 100/100 within 0.1 nat. The reported error in
  the pooled Gaussian control was about 10% below the observed run-to-run
  scatter, and the route used about eight times the logical likelihood calls of
  the lower-live control. Real scenes, multimodal likelihoods and GPU execution
  are not calibrated.
- **Full-budget CPU completion.** The synthetic quick-start completes at
  1,000,000 draws per scenario on four CPUs (see [CPU and GPU](#cpu-and-gpu)).
- **Examples.** The synthetic quick-start uses `bin_integration_days = 0` for
  unbinned points and an explicit draw-budget switch; the TIC notebook
  calibrates the flux uncertainty on native cadences before phase folding.
- **Distribution.** TREX 1.0 is installed from GitHub; there is no PyPI release
  for 1.0. The package and import name are both `trex`.

Known limits: the BlackJAX route is calibrated only on the synthetic known-Z
controls, not on real scenes, multimodal likelihoods or GPU execution; each
full-budget timing is one synthetic input on one host; the retained notebook
outputs are reduced-budget and no full-budget notebook re-execution is recorded;
macOS arm64 dependency resolution was checked, but no native install or run is
recorded; probabilities remain conditional and `claim_eligible` is always
`false`.

## Reporting bugs and contributing

Report a reproducible bug or request a feature at
[github.com/shenfurkan/TREX/issues](https://github.com/shenfurkan/TREX/issues).
Include the TREX version or commit, the input kind, the draw budget, the
command or API call, and the relevant error or `failure.json`. Do not attach
credentials or proprietary observations. Small fixes and documentation
corrections are welcome as pull requests; keep a change's error handling,
convergence rules and draw budgets intact, and state which checks you ran.

## More information

- [Documentation overview](docs/README.md) and [page index](docs/index.md)
- [Installation details](docs/installation.md)
- [Concepts](docs/concepts.md) and [how TREX works](docs/how-trex-works.md)
- [Tutorial](docs/tutorial.md), [from a TIC ID](docs/from-tic.md), [from a scene](docs/from-scene.md) and [prepared runs](docs/prepared-runs.md)
- [Inputs](docs/inputs.md), [outputs](docs/outputs.md) and [Python API](docs/api.md)
- [Scenarios](docs/scenarios.md), [background population](docs/background-population.md) and [module reference](docs/modules.md)
- [FAQ and troubleshooting](docs/faq.md) and [glossary](docs/glossary.md)
- [Differences from TRICERATOPS](docs/differences-from-triceratops.md)
- [Source and examples](https://github.com/shenfurkan/TREX)
- [TRICERATOPS paper: Giacalone et al. (2021)](https://doi.org/10.3847/1538-3881/abc6af)

Please cite TREX as described in
[CITATION.cff](https://github.com/shenfurkan/TREX/blob/main/CITATION.cff) and
cite the TRICERATOPS paper for the scenario framework. TREX is licensed under
GPL-3.0-or-later; bundled tables have their own attribution notices.

Copyright (C) 2026 Furkan Şen. This program is free software: you can
redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. It is distributed WITHOUT ANY
WARRANTY; see [LICENSE](LICENSE).
