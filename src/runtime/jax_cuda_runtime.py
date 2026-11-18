"""Load the installed CUDA linker dependency before JAX's CUDA plugin.

ELF LD_LIBRARY_PATH takes precedence over a wheel's RUNPATH. A host CUDA
toolkit can therefore shadow the locked nvJitLink wheel needed by cuSPARSE,
even though cuSPARSE itself comes from the project environment. This adapter
loads only that installed dependency by absolute path; it does not rewrite
the environment, select a GPU, initialize JAX, or bypass version checks.
CPU-only processes and hosts without a shadowing library do no native work.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path
import sys


@lru_cache(maxsize=None)
def _load_linker(path: str):
    """Retain the native handle for the interpreter's lifetime."""
    library = ctypes.CDLL(path)
    getattr(library, "__nvJitLinkCreate_12_0")
    return library


def prepare_jax_cuda_libraries() -> dict:
    """Prefer the runtime's nvJitLink when a host ELF search path shadows it.

    Must run before the first CUDA plugin initialization. A broken installed
    wheel raises with a fixed diagnostic and chained loader error; missing wheels remain the
    dependency installer's responsibility, never a fabricated GPU capability.
    No global JAX device state or user library-path setting is changed.
    """
    policy = {"preloaded": False, "library": None, "host_shadow_count": 0, "reason": None}
    platforms = {part.strip().lower() for part in os.environ.get("JAX_PLATFORMS", "").split(",") if part.strip()}
    if sys.platform != "linux" or platforms and not platforms & {"cuda", "gpu"}:
        policy["reason"] = "CUDA linker preparation is not required"
        return policy
    soname = "libnvJitLink.so.12"
    # This adapter follows the CUDA 12 wheels in the selected dependency lock.
    paths = list(dict.fromkeys(
        str(Path(directory) / soname)
        for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if directory and (Path(directory) / soname).is_file()
    ))
    if not paths:
        policy["reason"] = "no host CUDA linker shadow detected"
        return policy
    policy["host_shadow_count"] = len(paths)
    try:
        distribution = metadata.distribution("nvidia-nvjitlink-cu12")
    except metadata.PackageNotFoundError:
        policy["reason"] = "CUDA 12 nvJitLink wheel is not installed"
        return policy
    path = Path(distribution.locate_file("nvidia/nvjitlink/lib/" + soname)).resolve()
    # Trust the installed distribution's manifest, including valid --user installs.
    if not path.is_file():
        raise RuntimeError("installed nvJitLink wheel is missing its linker library")
    try:
        _load_linker(str(path))
    except (OSError, AttributeError) as exc:
        raise RuntimeError("cannot preload installed CUDA linker dependency") from exc
    policy.update(preloaded=True, library=soname, version=distribution.version)
    return policy


__all__ = ["prepare_jax_cuda_libraries"]
