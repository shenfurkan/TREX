# Installation

This page covers installing TREX, the platforms it can run on, and the checks
that show an installation is intact. The package metadata declares `trex`
version `1.0.0` and Python `>=3.14,<3.15`.

## Requirements and support status

TREX itself is pure Python. The platform limits below come from its compiled
dependencies, including `jaxlib`.

| Component | Status | Basis |
|---|---|---|
| Python 3.14.x | Required | `requires-python = ">=3.14,<3.15"`; CI selects 3.14 |
| Other Python series | Rejected by package metadata | Outside the declared Python range |
| Linux x86-64 | Primary install path | The project README records the Linux install path; CI targets `ubuntu-latest` |
| macOS arm64 | No native run recorded | Check wheel availability for the selected dependency graph |
| WSL on x86-64 Windows | Linux path, not separately tested | WSL presents the Linux x86-64 environment; there is no WSL CI job |
| Native Windows | Unsupported | No Windows classifier; the `python -m trex.run` checkpoint check uses `os.geteuid`, `st_uid` and POSIX mode bits |
| Intel macOS | Unsupported | `jaxlib` has no Intel macOS wheel in the lock |
| Linux aarch64 | Not tested | No native install or run recorded for this path |

CI is not a full numerical or network test. It runs `ruff check src tests` and
`pytest -q -m 'not slow and not network'`; slow numerical tests and live-network
tests are outside that job. "Tested path" therefore means the install and a
subset of the test suite, not every scientific path.

Native Windows is not a supported run target even if some dependency wheels
exist for it. Use WSL for the Linux path. Intel macOS lacks the required
compiled wheels. On macOS arm64 the same POSIX commands apply, but only
dependency resolution has been checked; no native install or run is recorded.

`target.from_tic` needs network access to public archives (MAST, Gaia,
TRILEGAL, MIST). Every archive response is retained under the acquisition
output directory, so that tree holds catalogue and pixel-file downloads as well
as results. The synthetic example described below runs offline.

## Distribution

The distribution and import name are both `trex`. Install TREX 1.0.0 from PyPI:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install trex==1.0.0
```

The unrelated pre-1.0 releases on PyPI have been yanked. The pin selects
this exact release.

There are no console-script entry points. Run the installed modules with
`python -m`.

### From a clone

To install this release from source instead of PyPI:

```bash
git clone --branch v1.0.0 https://github.com/shenfurkan/TREX
cd TREX
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

### Editable or developer install

For development on `main`, use an editable install so Python imports from
the checkout:

```bash
git clone https://github.com/shenfurkan/TREX
cd TREX
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The `dev` group in `pyproject.toml` is a dependency group, not an optional
extra; `pip install -e '.[dev]'` does not install it. Install its pinned tools
directly when using pip:

```bash
python -m pip install pytest==9.1.1 pytest-xdist==3.8.0 ruff==0.16.9
```

### With uv

To install TREX 1.0.0 from PyPI without cloning the repository:

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install trex==1.0.0
```

If Python 3.14 is not available, uv downloads it automatically by default.

A clone that contains `uv.lock` can install the recorded resolution:

```bash
git clone https://github.com/shenfurkan/TREX
cd TREX
uv sync --locked --python 3.14
uv run python -m trex.example /tmp/trex-demo
```

`--locked` installs exactly the resolution recorded in `uv.lock` and refuses to
rewrite the lock. `uv sync` also installs the `dev` group by default, so
`uv run pytest` and `uv run ruff` use the project environment. Without
`--locked`, uv may update the resolution.

### From a Git URL

To install a fixed revision without keeping a clone:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install "git+https://github.com/shenfurkan/TREX@<tag>"
```

`<tag>` must be a tag that exists in the repository. This page does not assert
that any tag has been created; confirm the name with

```bash
git ls-remote --tags https://github.com/shenfurkan/TREX
```

before using it. A full commit hash also works in place of `<tag>`. A missing
or misspelled reference fails during clone or checkout. If no tag exists,
install from a clone and record the commit.

### CUDA 12 extra

The default installation uses JAX on the CPU. The only optional extra is
`cuda12`, declared as `jax[cuda12]>=0.11.2,<0.12`. On Linux with an NVIDIA GPU
and a compatible driver, install it from PyPI:

```bash
python -m pip install 'trex[cuda12]==1.0.0'
```

From a clone, use:

```bash
python -m pip install '.[cuda12]'
```

or with uv:

```bash
uv sync --locked --python 3.14 --extra cuda12
```

From a Git URL, name the extra in the requirement, using an existing tag or
commit reference as above:

```bash
python -m pip install "trex[cuda12] @ git+https://github.com/shenfurkan/TREX@<tag>"
```

Installing the extra is not a GPU test. JAX still needs a compatible NVIDIA
driver and a device it can see. Check with:

```bash
python -c "import jax; print(jax.devices())"
```

CPU-only output is a supported configuration. The CUDA 12 extra is a
Linux/NVIDIA path; macOS arm64 uses the default CPU install.

## Dependencies

These are the direct runtime requirements in `[project].dependencies`.
Transitive packages come from the resolver, or from `uv.lock` when using
`uv sync`.

| Package | Declared requirement | Role |
|---|---|---|
| `numpy` | `>=2.5.5` | Array arithmetic |
| `scipy` | `>=1.18.1` | Special functions, linear algebra, optimisation |
| `pandas` | `>=2.3.3` | Probability and result tables |
| `matplotlib` | `>=3.11.2` | `plot_fits` and `plot_field` |
| `astropy` | `>=8.0.3` | Coordinates, FITS/WCS, VOTable, constants |
| `astroquery` | `>=0.4.11` | Catalogue and archive queries |
| `lightkurve` | `>=2.6.0` | TESScut cutouts and SPOC light-curve products |
| `pyvo` | `>=1.9.1` | Gaia TAP and UWS queries |
| `jax` | `>=0.11.2,<0.12` | Batched likelihood kernels and accelerator path |
| `jaxoplanet` | `>=0.1.0` | Native JAX transit and limb-darkening models |
| `equinox` | `>=0.13.8` | JAX model component |
| `optax` | `>=0.2.8` | JAX optimisation component |
| `dynesty` | `>=3.0.0` | Default nested-sampling refinement for low-ESS scenarios |
| `blackjax` | `>=1.7.1` | Experimental opt-in nested-slice estimator |
| `gaiadr3-zeropoint` | `>=0.1.0` | Gaia DR3 parallax zero-point tables |
| `requests` | `>=2.34.2` | Archive HTTP transport |
| `urllib3` | `>=2.8.0` | HTTP retry policy |
| `certifi` | `>=2026.7.22` | TLS certificate bundle |
| `cryptography` | `>=50.0.1` | Archive transport support |

`setuptools>=83` is a build requirement, not a direct runtime dependency.

The optional `cuda12` extra is separate from this table:
`jax[cuda12]>=0.11.2,<0.12`. It is not installed by `pip install .` or a plain
`uv sync`.

The package data declaration includes `_data/*.csv` and `_data/*LICENSE`, so an
installed wheel carries the bundled limb-darkening and response tables under
`trex/_data/`. Those tables do not need a separate download.

## Verify the installation

Importing the `trex` namespace performs no device or network work:

```bash
python -c "import trex; print(trex.__file__)"
```

The path should be the environment or checkout you installed, not a different
Python installation. The operational entry points are modules:

```bash
python -m trex.doctor --help
python -c "import jax; print(jax.devices())"
python -m trex.example /tmp/trex-demo
```

- `python -m trex.doctor --help` lists the input check, the optional MAST probe
  and the failure-record grouping mode. With an input directory,
  `python -m trex.doctor INPUT_DIR` runs the engine's pre-draw admission checks
  with zero evidence draws; it does not run inference or establish convergence.
  `--mast-tic` performs a real, time-bounded MAST metadata query. The synthetic
  example's `inputs/` directory is not a prepared-run input directory.
- `jax.devices()` should report at least one device. CPU devices are a supported
  configuration.
- `python -m trex.example /tmp/trex-demo` builds an invented target, two
  invented neighbours and an artificial dip, then runs the real engine at 4,096
  draws per scenario with `nsamples=4`. It refuses an existing directory. A
  completed calculation leaves `result/summary.json` and `result/scenarios.csv`;
  a recorded failure leaves `result/failure.json` instead, with no FPP. The
  command prints a status line, returns 0 when the status is `ok` and 3 for a
  completed run with another status. Choose a new directory if
  `/tmp/trex-demo` already exists.

The example is a software check. Its inputs are invented, its flux uncertainty
is deliberately large, and its budget is far below the production budget of
1,000,000 draws per scenario. Its FPP is not a scientific result.

None of the commands on this page were executed while the page was written. The
Linux install statement comes from the project root `README.md`, the Python/CI
configuration from `.github/workflows/ci.yml`, the macOS and wheel statements
from that README and `uv.lock`, and the entry points, messages and package data
from `src/` and `pyproject.toml`. No output or timing is reproduced here.

## Reproducibility

For an analysis you intend to keep, record:

| Item | Where to record it |
|---|---|
| TREX version and commit | `uv pip show trex` or `python -m pip show trex`; `git rev-parse HEAD` in a clone |
| Installed packages | `uv pip freeze` or `python -m pip freeze` from the active environment |
| Input files and SHA-256 digests | `calculation_provenance.input_artifacts` in `summary.json`; `input_artifacts=` when binding a scene |
| Random seed | `calculation_provenance.random_seed` |
| Draw budget | `calculation_provenance.n_draws`; API argument `N`; prepared-run CLI `--draws` |
| Exposure supersampling | `calculation_provenance.nsamples`; prepared-run CLI `--nsamples` |
| Exposure and bin widths | `calculation_provenance.exptime_days` and `bin_integration_days`, in days |
| Period, depth and sector weights | `calculation_provenance.period_days`, `depth_ppm` and `sector_inverse_variance_weights` |
| CPU or GPU | The `jax.devices()` output and the host record |

`uv.lock` fixes the resolution for Python 3.14; pip installs from the declared
dependency ranges and may select different versions. Compatibility across
every allowed dependency version has not been established. A result reproduced
on a different backend is expected to agree statistically, not bit for bit.
The provenance fields above record the settings of a run; they do not make two
different dependency sets equivalent.

## Troubleshooting

| Symptom | Likely cause | Check and action |
|---|---|---|
| `pip show trex` reports a pre-1.0 version | An unrelated package predates this release on PyPI; those versions are yanked | Use a fresh Python 3.14 environment and `python -m pip install trex==1.0.0` |
| `ModuleNotFoundError: No module named 'trex'` | A different interpreter or an inactive environment | Run `python -c "import sys; print(sys.executable)"`; activate the environment and reinstall with the same `python` |
| `ERROR: Package 'trex' requires a different Python` | The interpreter is outside 3.14 | Create the environment with `python3.14` and use its `python -m pip` |
| `FileExistsError` from the example | The output directory already exists | Choose a new directory. The example will not overwrite an earlier run |
| JAX reports CPU devices only | The default CPU installation, or no visible GPU | This is supported. Use it as is unless a GPU was intended |
| The CUDA extra is installed but JAX still reports CPU | The driver or device is not visible, or CUDA setup fell back to the reference path | Check `jax.devices()` and the driver. Look for the warning `JAX setup failed; CPU reference execution selected`; check that the CUDA 12 JAX wheels and `nvidia-nvjitlink-cu12` are in `pip freeze` |
| `python -m trex.doctor` with no arguments exits with an argparse error | The command needs one of its documented modes | Run `python -m trex.doctor --help`; no-argument invocation is not an install check |
| The example runs but `target.from_tic` fails | Archive or network access is unavailable | `from_tic` needs network access; `python -m trex.doctor --mast-tic TIC_ID` is a bounded MAST metadata probe, not a full acquisition test |

On Linux, a host CUDA toolkit can shadow the CUDA linker wheel. When it detects
that shadow, TREX preloads the installed `nvidia-nvjitlink-cu12` library before
JAX initialises. If that preload fails, the JAX setup warning above records the
fallback rather than fabricating a GPU capability.

## Next steps

- [Documentation home](README.md)
- [Concepts](concepts.md) for FPP and the scenario framework.
- [Tutorial](tutorial.md) for a guided first analysis.
- [Python API](api.md) for the calls used above.
- [Outputs](outputs.md) for `summary.json`, `scenarios.csv` and failure records.
- [FAQ and troubleshooting](faq.md) for the message-level answers.
