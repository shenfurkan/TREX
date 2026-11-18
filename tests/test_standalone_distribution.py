"""Build/install TREX wheels and editable source and run its public API exonym-free.

Run only on the approved host. The isolated install uses the test
environment's own resolved dependency versions, each admitted only when it
satisfies this project's declared requirement: a hermetic runner has no
network, and this project's lock may pin newer point releases than the host
cache holds. Only the editable case imports the explicitly installed checkout. This is
CPU software acceptance, not empirical calibration.
"""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from packaging.markers import default_environment

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

from tests.fixtures.synthetic_observations import synthetic_portable_trex_inputs


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
PROBE = ROOT / "tests/fixtures/portable_trex_installed_check.py"

pytestmark = pytest.mark.slow


def _declared_requirements(project):
    """Return (extra, requirement) for runtime dependencies and every optional extra."""
    return [(extra, Requirement(value))
            for extra, values in (("", project["dependencies"]),
                                  *project.get("optional-dependencies", {}).items())
            for value in values]


@pytest.fixture(scope="module")
def standalone_distributions(tmp_path_factory):
    directory = tmp_path_factory.mktemp("standalone-dist")
    result = subprocess.run(
        ["uv", "build", "--python", sys.executable, "--sdist", "--wheel", "--out-dir", str(directory)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return directory


def test_standalone_wheel_and_sdist_are_exonym_free_and_rebuildable(standalone_distributions, tmp_path):
    wheel = next(standalone_distributions.glob("*.whl"))
    source_files = [path for path in SOURCE.rglob("*.py")
                    if "__pycache__" not in path.relative_to(SOURCE).parts]
    source_files.extend(path for path in (SOURCE / "_data").iterdir() if path.is_file())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {(canonicalize_name(item.name), tuple(sorted(item.extras)), str(item.specifier),
                 f'extra == "{extra}"' if extra else "")
                for extra, item in _declared_requirements(project)}
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith("exonym/") for name in archive.namelist())
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        built = {(canonicalize_name(item.name), tuple(sorted(item.extras)), str(item.specifier),
                  str(item.marker or ""))
                 for item in map(Requirement, (line.removeprefix("Requires-Dist: ")
                                               for line in archive.read(metadata).decode().splitlines()
                                               if line.startswith("Requires-Dist: ")))}
        assert built == declared, f"wheel metadata differs from pyproject: {built ^ declared}"
        for path in source_files:
            assert archive.read("trex/" + path.relative_to(SOURCE).as_posix()) == path.read_bytes()
    sdist = next(standalone_distributions.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path, filter="data")
    project = next(tmp_path.glob("trex-*"))
    result = subprocess.run(["uv", "build", "--python", sys.executable, "--wheel", "--out-dir", str(tmp_path / "rebuilt")],
                            cwd=project, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(next((tmp_path / "rebuilt").glob("*.whl"))) as archive:
        for path in source_files:
            assert archive.read("trex/" + path.relative_to(SOURCE).as_posix()) == path.read_bytes()


def _installed_closure(requirements):
    """Pin the requirements' installed dependency closure for an isolated install.

    Offline hosts install from the environment's wheel cache, so each selected
    version is the installed one and is admitted only when it satisfies the
    declared specifier. ``uv pip check`` verifies the resulting closure.
    """
    pending = [Requirement(value) for value in requirements]
    visited = set()
    selected = {}
    environment = default_environment()
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = name, tuple(sorted(requirement.extras))
        if key in visited:
            continue
        visited.add(key)
        distribution = importlib.metadata.distribution(name)
        assert requirement.specifier.contains(distribution.version, prereleases=True), (
            f"installed {name} {distribution.version} does not satisfy {requirement}")
        selected[name] = distribution.version
        for raw in distribution.requires or []:
            dependency = Requirement(raw)
            if dependency.marker is None or any(dependency.marker.evaluate({**environment, "extra": extra})
                                                 for extra in {"", *requirement.extras}):
                pending.append(dependency)
    assert "exonym" not in selected and "scipaperfinder" not in selected
    return [f"{name}=={version}" for name, version in sorted(selected.items())]


def test_declared_requirements_resolve_in_the_project_lock():
    """The lock is this project's pin authority; every declared range must resolve in it."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    locked = {canonicalize_name(package["name"]): package["version"]
              for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
              if "version" in package}
    for _extra, requirement in _declared_requirements(project):
        name = canonicalize_name(requirement.name)
        assert name in locked, f"lock has no resolution for declared requirement {requirement}"
        assert requirement.specifier.contains(locked[name], prereleases=True), (
            f"lock resolves {requirement} to {locked[name]}")


@pytest.mark.parametrize("editable", [False, True], ids=["wheel", "editable"])
def test_clean_installed_package_runs_real_target_inference_without_exonym(
    standalone_distributions, tmp_path, record_property, editable,
):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    pins = tmp_path / "runtime-lock.txt"
    pins.write_text("\n".join(_installed_closure(project["dependencies"])) + "\n")
    environment = tmp_path / "environment"
    result = subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    python = environment / "bin/python"
    wheel = next(standalone_distributions.glob("*.whl"))
    package = ["--editable", str(ROOT)] if editable else [str(wheel)]
    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "-r", str(pins), *package],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checked = subprocess.run(["uv", "pip", "check", "--python", str(python)],
                             capture_output=True, text=True, timeout=120)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    scene, binding, time, flux, sigma, period = synthetic_portable_trex_inputs(tmp_path / "inputs")
    inputs = tmp_path / "portable-inputs.json"
    inputs.write_text(json.dumps({
        "scene": scene.to_dict(), "binding": binding,
        "time": time.tolist(), "flux": flux.tolist(), "sigma": sigma.tolist(), "period_days": period,
        "source_root": str(SOURCE) if editable else None,
    }, allow_nan=False))
    result = subprocess.run([str(python), "-I", str(PROBE), str(inputs)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    record_property("independent_installed_inference_receipt", result.stdout.strip())
    record_property("independent_runtime_pins", pins.read_text())
