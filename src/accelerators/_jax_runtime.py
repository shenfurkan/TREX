"""Shared JAX setup for the double-precision TREX accelerator kernels."""

from trex.runtime.hardware import configure_jax_memory
from trex.runtime.jax_cuda_runtime import prepare_jax_cuda_libraries
from trex.runtime.logging import host_logger


JAX_ALLOCATION_POLICY = configure_jax_memory()
JAX_SETUP_FAILURE = None
JAX_CUDA_LIBRARY_POLICY = {"preloaded": False, "reason": "CUDA library preparation has not completed"}

try:
    JAX_CUDA_LIBRARY_POLICY = prepare_jax_cuda_libraries()
    import jax as jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: F401 - shared runtime export

    try:
        from jaxoplanet.core.limb_dark import light_curve as core_limb_dark  # noqa: F401 - shared runtime export
    except ImportError:
        from jaxoplanet.light_curves.limb_dark import _limb_dark_light_curve as core_limb_dark  # noqa: F401 - shared runtime export
    HAS_JAX_TREX = True
except (ImportError, OSError, RuntimeError, AttributeError) as exc:
    HAS_JAX_TREX = False
    JAX_SETUP_FAILURE = {
        "error_type": type(exc).__name__,
        "reason": "JAX setup failed; CPU reference execution selected",
    }
    host_logger("trex").warning(
        JAX_SETUP_FAILURE["reason"], exc_info=True,
        extra={"event": "trex.hardware.fallback", "error_type": type(exc).__name__},
    )
