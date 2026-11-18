"""Portable linker preparation; mock loader cost, not the preparation policy."""

from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from trex.runtime import jax_cuda_runtime as runtime


@pytest.mark.parametrize("platforms", ["cpu", "cpu,tpu"])
def test_loader_does_not_change_cpu_library_paths(monkeypatch, platforms):
    monkeypatch.setenv("JAX_PLATFORMS", platforms)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/target-neutral/driver:/target-neutral/toolkit")
    result = runtime.prepare_jax_cuda_libraries()
    assert result["preloaded"] is False
    assert os.environ["LD_LIBRARY_PATH"] == "/target-neutral/driver:/target-neutral/toolkit"


def test_host_shadow_uses_installed_library_without_rewriting_search_path(tmp_path, monkeypatch):
    host = tmp_path / "host"
    host.mkdir()
    (host / "libnvJitLink.so.12").touch()
    installed = tmp_path / "installed"
    library = installed / "nvidia/nvjitlink/lib/libnvJitLink.so.12"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu,cuda")
    search = os.pathsep.join((str(host), str(host)))
    monkeypatch.setenv("LD_LIBRARY_PATH", search)
    monkeypatch.setattr(runtime.metadata, "distribution", lambda _name: SimpleNamespace(
        locate_file=lambda name: installed / name, version="synthetic",
    ))
    loaded = []
    monkeypatch.setattr(runtime, "_load_linker", loaded.append)
    policy = runtime.prepare_jax_cuda_libraries()
    assert policy["preloaded"]
    assert policy["host_shadow_count"] == 1
    assert policy["library"] == "libnvJitLink.so.12"
    assert loaded == [str(library.resolve())]
    assert os.environ["LD_LIBRARY_PATH"] == search


def test_installed_linker_is_preloaded_before_real_jax_initialization():
    if sys.platform != "linux":
        pytest.skip("installed Linux nvJitLink wheel is required")
    try:
        package = metadata.distribution("nvidia-nvjitlink-cu12")
    except metadata.PackageNotFoundError:
        pytest.skip("optional CUDA linker wheel is not installed")
    path = Path(package.locate_file("nvidia/nvjitlink/lib/libnvJitLink.so.12")).resolve()
    assert path.is_file(), "installed CUDA linker wheel is incomplete"
    environment = dict(os.environ, LD_LIBRARY_PATH=str(path.parent))
    environment.pop("JAX_PLATFORMS", None)
    source = """
import json
from trex.runtime.jax_cuda_runtime import prepare_jax_cuda_libraries
policy = prepare_jax_cuda_libraries()
import jax
assert policy['preloaded']
print(json.dumps(policy))
"""
    result = subprocess.run([sys.executable, "-c", source], env=environment,
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["library"] == path.name
