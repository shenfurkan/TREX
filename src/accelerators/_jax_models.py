"""JAX-accelerated batched likelihood evaluators for TREX.

Provides vectorized GPU/CPU forward models for Mandel-Agol quadratic transit
and eclipsing binary scenarios with numerical Keplerian orbit propagation and
sub-cadence numerical supersampling. The orbit and light-curve physics, with
its sources, lives in ``trex.accelerators._jax_orbits``; this module owns
batching, device placement and the log-likelihood reduction.

All internal operations execute under full double precision (float64).
Time, period, and exposure duration use days; semimajor axis uses cm;
radii use R_earth / R_sun; angles use degrees; flux is normalized to unity.
Without a phase-bin box, sub-exposure integration averages ``nsamples``
midpoint nodes, the PyTransit 2.2 rule TRICERATOPS uses, matching the batman
reference path node for node.  With one, both boxes are carried by the exact
trapezoid kernel of ``_two_box_convolution_quadrature`` and the instantaneous
model is evaluated at its Gauss-Legendre nodes, matching the batman
reference path node for node.
"""

from __future__ import annotations

from trex.runtime.logging import host_logger

from functools import partial as partial
from fractions import Fraction as Fraction
from typing import Optional as Optional, Tuple as Tuple

import numpy as np

from trex.foundation._numerics import _per_bin_two_box_quadrature as _per_bin_two_box_quadrature, _two_box_convolution_quadrature as _two_box_convolution_quadrature, phase_bin_widths as phase_bin_widths
from trex.accelerators._jax_batch import BatchEvaluator as BatchEvaluator
from trex.foundation.constants import Rsun as Rsun, Rearth as Rearth
from trex.runtime.hardware import configure_jax_memory as configure_jax_memory
from trex.accelerators import _jax_runtime
from trex.accelerators._jax_runtime import (
    HAS_JAX_TREX as HAS_JAX_TREX,
    JAX_ALLOCATION_POLICY as JAX_ALLOCATION_POLICY,
    JAX_CUDA_LIBRARY_POLICY as JAX_CUDA_LIBRARY_POLICY,
    JAX_SETUP_FAILURE as JAX_SETUP_FAILURE,
)

# Dispatch helpers are re-exported here on purpose: ``_jax_dispatch._models()``
# looks them up on this module, so a test patches one name in one place.
from trex.accelerators._jax_dispatch import _jax_batch_device_memory_bytes as _jax_batch_device_memory_bytes
from trex.accelerators._jax_dispatch import _jax_batch_memory_budget_bytes as _jax_batch_memory_budget_bytes
from trex.accelerators._jax_dispatch import _jax_chunk_size as _jax_chunk_size
from trex.accelerators._jax_dispatch import _pilot_parity as _pilot_parity
from trex.accelerators._jax_dispatch import _compile_pilot as _compile_pilot
from trex.accelerators._jax_dispatch import _timed_pilot as _timed_pilot
from trex.accelerators._jax_dispatch import _device_pilot as _device_pilot
from trex.accelerators._jax_dispatch import _whole_blocks as _whole_blocks
from trex.accelerators._jax_dispatch import _PreparedBatch as _PreparedBatch
from trex.accelerators._jax_dispatch import _plan_batch_chunk as _plan_batch_chunk
from trex.accelerators._jax_dispatch import _run_jax_chunks as _run_jax_chunks

# Retain every eligible device. Placement belongs to each runtime batch, not
# process-global JAX configuration. Capacity is a policy, not a speed prediction.
JAX_DEVICE_SELECTION_REASON: Optional[str] = None
# The public tuple is populated on first access, not on import. Parser/help
# imports must not initialize a CUDA backend or reserve a device context.
JAX_USABLE_DEVICES: tuple


def _usable_jax_devices() -> tuple:
    """Resolve and cache the existing placement policy only when it is needed."""
    global JAX_DEVICE_SELECTION_REASON

    if "JAX_USABLE_DEVICES" in globals():
        cached = globals()["JAX_USABLE_DEVICES"]
        # A non-tuple (for example None from a mistaken override) would read as
        # "no accelerators" and silently move production work to the CPU.
        if not isinstance(cached, tuple):
            raise TypeError(f"JAX_USABLE_DEVICES must be a tuple of devices, got {type(cached).__name__}")
        return cached
    if not HAS_JAX_TREX:
        JAX_DEVICE_SELECTION_REASON = JAX_SETUP_FAILURE["reason"] if JAX_SETUP_FAILURE else "JAX is unavailable"
        globals()["JAX_USABLE_DEVICES"] = ()
        return ()
    from trex.runtime.hardware import accelerator_memory_floor_bytes

    accelerator_memory_floor_bytes()  # validate user policy outside driver handling
    try:
        from trex.runtime.hardware import accelerator_meets_memory_floor

        _accelerators = [
            _device
            for _device in jax.devices()
            if _device.platform not in ("cpu", "interpreter")
        ]
        devices = tuple(
            device for device in _accelerators if accelerator_meets_memory_floor(device)
        )
        if _accelerators and not devices:
            JAX_DEVICE_SELECTION_REASON = "no accelerator meets the memory policy floor; CPU execution selected"

            host_logger("trex").warning(JAX_DEVICE_SELECTION_REASON)
    except Exception as exc:  # noqa: BLE001 - placement is an optimisation, never a gate
        JAX_DEVICE_SELECTION_REASON = "JAX placement probe failed; CPU execution selected"

        host_logger("trex").warning(
            JAX_DEVICE_SELECTION_REASON, exc_info=True,
            extra={"event": "trex.hardware.fallback", "error_type": type(exc).__name__},
        )
        devices = ()
    globals()["JAX_USABLE_DEVICES"] = devices
    return devices


def __getattr__(name: str):
    """Preserve the public device-tuple facade without eager hardware probing."""
    if name == "JAX_USABLE_DEVICES":
        return _usable_jax_devices()
    if HAS_JAX_TREX and name in {
        "_kepler_true_anomaly", "_jax_batman_sky_separation", "_jax_transit",
        "_jax_secondary_eclipse_phase", "_simulate_tp_jax", "_simulate_eb_jax",
    }:
        return getattr(_jax_orbits, name)
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))


# JAX core models (only instantiated when JAX is present)

if HAS_JAX_TREX:
    from trex.accelerators import _jax_orbits

    jax, jnp, core_limb_dark = _jax_runtime.jax, _jax_runtime.jnp, _jax_runtime.core_limb_dark

    @partial(jax.jit, static_argnames=("companion_is_host", "nsamples"))
    def _batch_lnL_TP_jit(
        time: jnp.ndarray,
        flux_obs: jnp.ndarray,
        inv_sigma_sq: jnp.ndarray,
        rp_arr: jnp.ndarray,
        p_orb: float,
        inc_arr: jnp.ndarray,
        a_cm_arr: jnp.ndarray,
        R_host_arr: jnp.ndarray,
        u1_arr: jnp.ndarray,
        u2_arr: jnp.ndarray,
        exptime_days: float,
        ecc_arr: jnp.ndarray,
        w_arr: jnp.ndarray,
        cfr_arr: jnp.ndarray,
        companion_is_host: bool,
        nsamples: int,
        bin_weights: "jnp.ndarray | None" = None,
    ) -> jnp.ndarray:
        def single_eval(r_p, inc_val, a_val, r_host, u1_val, u2_val, ecc_val, w_val, cfr_val):
            model = _jax_orbits._simulate_tp_jax(
                time, r_p, p_orb, inc_val, a_val, r_host, u1_val, u2_val,
                exptime_days, ecc=ecc_val, argp_deg=w_val,
                companion_fluxratio=cfr_val,
                companion_is_host=companion_is_host,
                nsamples=nsamples,
            )
            if bin_weights is not None:
                # ``time`` arrived expanded to one row of quadrature nodes per
                # phase bin; the instantaneous model there is weighted by the
                # same trapezoid-kernel nodes and weights the batman reference
                # path uses, so the two paths convolve the two boxes identically.
                model = jnp.sum(
                    model.reshape(-1, bin_weights.shape[-1]) * bin_weights, axis=1
                )
            diff = flux_obs - model
            return -0.5 * jnp.sum(jnp.square(diff) * inv_sigma_sq)

        return jax.vmap(single_eval)(
            rp_arr, inc_arr, a_cm_arr, R_host_arr, u1_arr, u2_arr, ecc_arr, w_arr, cfr_arr
        )


    @partial(jax.jit, static_argnames=("companion_is_host", "nsamples", "secondary_to_host"))
    def _batch_lnL_EB_jit(
        time: jnp.ndarray,
        flux_obs: jnp.ndarray,
        inv_sigma_sq: jnp.ndarray,
        reb_arr: jnp.ndarray,
        feb_arr: jnp.ndarray,
        p_orb: float,
        inc_arr: jnp.ndarray,
        a_cm_arr: jnp.ndarray,
        R_host_arr: jnp.ndarray,
        u1_arr: jnp.ndarray,
        u2_arr: jnp.ndarray,
        eb_u1_arr: jnp.ndarray,
        eb_u2_arr: jnp.ndarray,
        exptime_days: float,
        ecc_arr: jnp.ndarray,
        w_arr: jnp.ndarray,
        cfr_arr: jnp.ndarray,
        companion_is_host: bool,
        nsamples: int,
        bin_weights: "jnp.ndarray | None" = None,
        secondary_to_host: bool = False,
    ) -> jnp.ndarray:
        def single_eval(r_eb, f_eb, inc_val, a_val, r_host, u1_val, u2_val, eb_u1, eb_u2, ecc_val, w_val, cfr_val):
            model = _jax_orbits._simulate_eb_jax(
                time, r_eb, None if secondary_to_host else f_eb, p_orb, inc_val, a_val, r_host, u1_val, u2_val,
                exptime_days, ecc=ecc_val, argp_deg=w_val,
                companion_fluxratio=cfr_val,
                companion_is_host=companion_is_host,
                nsamples=nsamples,
                u1_EB=eb_u1, u2_EB=eb_u2,
                secondary_to_host_flux_ratio=f_eb if secondary_to_host else None,
            )
            if bin_weights is not None:
                # Same trapezoid-kernel treatment as the TP path above.
                model = jnp.sum(
                    model.reshape(-1, bin_weights.shape[-1]) * bin_weights, axis=1
                )
            diff = flux_obs - model
            return -0.5 * jnp.sum(jnp.square(diff) * inv_sigma_sq)

        return jax.vmap(single_eval)(
            reb_arr, feb_arr, inc_arr, a_cm_arr, R_host_arr, u1_arr, u2_arr,
            eb_u1_arr, eb_u2_arr,
            ecc_arr, w_arr, cfr_arr,
        )


# NUMERICAL_POLICY: multiplier on the nominal model-grid size to account for the
# temporaries XLA holds per draw (Kepler solve, sky separation, limb-darkened
# flux). Historical requested/nominal ratio was about 7.4. This seeds a pilot
# compile only on GPUs; XLA memory analysis then replaces the grid estimate.
_JAX_BATCH_MEMORY_OVERHEAD: int = 8

# EXECUTION_POLICY: conservative CPU/unknown-device working-set share. GPU auto
# mode instead sizes the complete compiled working set with measured VRAM.
_JAX_BATCH_MEMORY_DEVICE_DIVISOR: int = 4

# EXECUTION_POLICY: keep ten percent of observed allocator headroom outside the
# compiled working set for executable/context allocations and concurrent use.
# XLA temporaries are already measured, unlike the old quarter-grid policy.
_JAX_COMPILED_MEMORY_FRACTION = Fraction(9, 10)

# Per-process-tree override for the resolved batch budget, in bytes. Set it to
# pin the chunk size on a host whose device misreports its addressable memory.
JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE: str = "TREX_JAX_BATCH_MEMORY_BYTES"


# Device choice is measured per kernel, never inferred from a model name, a
# device count or float32 throughput. Algorithmic, not empirical: the pilot is
# large enough to saturate a GPU batch. The winning device's own pilot values
# supply the production prefix; CPU reference values never replace accelerator
# draws. Cached decisions retain placement only, never a previous run's values.
DEVICE_PILOT_DRAWS = 1 << 14
# A uniform per-draw ln-likelihood offset d changes a scenario evidence by the
# factor exp(d), i.e. by relative |d| ~ 1e-6, three orders of magnitude below
# the 1/sqrt(N) = 1e-3 Monte Carlo scale of the N = 10^6 production budget.
DEVICE_PILOT_LNL_TOLERANCE = 1e-6
# Draws more than this far below the pilot maximum underflow exp() relative to
# it in float64 (smallest subnormal ~ exp(-745)) and cannot move an evidence.
_FLOAT64_LOG_UNDERFLOW = 745.0
_DEVICE_DECISIONS: dict = {}


def _phase_bin_quadrature(
    time: np.ndarray, bin_integration_days: float, exposure_days: float
) -> Tuple[np.ndarray, "np.ndarray | None"]:
    """Expand phase times onto the two-box trapezoid-kernel quadrature nodes.

    Returns the flattened evaluation grid and the kernel weights (``None``
    when no phase-bin box applies). The offsets, ordering, and row-major
    flattening are identical to the batman reference path, so weighting the
    instantaneous model back by these nodes reproduces the same two-box
    convolution the exact trapezoid kernel defines.
    """
    values = np.asarray(time, dtype=float)
    widths = phase_bin_widths(bin_integration_days, values.size)
    if isinstance(widths, np.ndarray):
        offsets, weights = _per_bin_two_box_quadrature(widths, float(exposure_days))
        return (values[:, None] + offsets).reshape(-1), weights
    if widths <= 0.0:
        return values, None
    offsets, weights = _two_box_convolution_quadrature(
        float(bin_integration_days), float(exposure_days)
    )
    return (values[:, None] + offsets[None, :]).reshape(-1), weights


def _per_draw(value: np.ndarray | float, n_items: int, label: str) -> np.ndarray:
    """One value per draw: a read-only broadcast view of a scalar, or a validated array.

    The view costs no memory for a scalar shared by a million draws; blocks are
    copied into contiguous device buffers only when they are launched.
    """
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.broadcast_to(array, (n_items,))
    if array.shape != (n_items,):
        raise ValueError(
            "{0} must be a scalar or an array of {1} per-draw values".format(label, n_items)
        )
    return array


def _shared_observation_inputs(time, flux_obs, sigma, bin_integration_days, exptime_days, nsamples):
    """Validated observation arrays and quadrature shared by every draw.

    ``sigma`` (relative flux) must be finite and positive: a zero or NaN
    uncertainty would silently become an infinite or NaN weight in every
    log-likelihood. Returns ``(inv_sigma_sq, eval_times, bin_weights,
    exposure_samples)``.
    """
    sigma_values = np.asarray(sigma, dtype=float)
    if not np.all(np.isfinite(sigma_values)) or np.any(sigma_values <= 0.0):
        raise ValueError("flux uncertainties must be finite and positive")
    inv_sigma_sq = np.broadcast_to(1.0 / np.square(sigma_values), np.shape(time))
    eval_times, bin_weights = _phase_bin_quadrature(time, bin_integration_days, exptime_days)
    # With the phase-bin kernel active the exposure box is already inside the
    # weights, so the model must be evaluated instantaneously (one exposure
    # sample); without it, ``nsamples`` keeps its exposure-supersampling role.
    exposure_samples = 1 if bin_weights is not None else nsamples
    return inv_sigma_sq, eval_times, bin_weights, exposure_samples


def _evaluate_batch(kernel, shared, draws, n_items, eval_times, exposure_samples,
                    chunk_size_divisor, chunk_plan, prepare_only):
    evaluate = BatchEvaluator(
        jax, jnp, kernel, shared, draws, eval_times.size * exposure_samples,
        static_arguments=("companion_is_host", "nsamples", "secondary_to_host"),
    )
    if prepare_only:
        return _PreparedBatch(evaluate, eval_times.size, exposure_samples)
    return _run_jax_chunks(
        n_items, eval_times.size, exposure_samples, evaluate, chunk_size_divisor, chunk_plan,
    )


def batch_lnL_TP(
    time: np.ndarray,
    flux_obs: np.ndarray,
    sigma: np.ndarray | float,
    rp_arr: np.ndarray,
    p_orb: float,
    inc_arr: np.ndarray,
    a_cm_arr: np.ndarray | float,
    R_host: np.ndarray | float,
    u1: np.ndarray | float,
    u2: np.ndarray | float,
    exptime_days: float,
    ecc_arr: np.ndarray,
    w_arr: np.ndarray,
    companion_fluxratio: np.ndarray | float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *,
    chunk_size_divisor: int = 1,
    _chunk_plan: Optional[dict] = None,
    _prepare_only: bool = False,
) -> np.ndarray | _PreparedBatch:
    """Evaluate log-likelihoods for an array of transiting planet draws concurrently.

    ``a_cm_arr``, ``R_host``, ``u1``, ``u2``, and ``companion_fluxratio`` each
    accept either a scalar shared by every draw or a per-draw array: the
    unresolved-companion scenarios marginalize over a companion population, so
    the host star and the diluting companion both vary draw by draw.
    ``chunk_size_divisor`` reduces only the runtime working set for a retry.
    Private ``_prepare_only`` returns a deferred batch for scenario overlap.
    """
    if not HAS_JAX_TREX:
        raise RuntimeError("JAX/jaxoplanet is not available in the current environment")
    n_items = len(rp_arr)
    if n_items == 0:
        return np.empty(0, dtype=float)
    inv_sigma_sq, eval_times, bin_weights, exposure_samples = _shared_observation_inputs(
        time, flux_obs, sigma, bin_integration_days, exptime_days, nsamples)
    shared = dict(
        time=eval_times, flux_obs=np.asarray(flux_obs), inv_sigma_sq=inv_sigma_sq,
        p_orb=float(p_orb), exptime_days=float(exptime_days),
        companion_is_host=bool(companion_is_host), nsamples=int(exposure_samples), bin_weights=bin_weights,
    )
    draws = dict(
        rp_arr=rp_arr, inc_arr=inc_arr, a_cm_arr=_per_draw(a_cm_arr, n_items, "a_cm"),
        R_host_arr=_per_draw(R_host, n_items, "R_host"), u1_arr=_per_draw(u1, n_items, "u1"),
        u2_arr=_per_draw(u2, n_items, "u2"), ecc_arr=ecc_arr, w_arr=w_arr,
        cfr_arr=_per_draw(companion_fluxratio, n_items, "companion_fluxratio"),
    )
    return _evaluate_batch(_batch_lnL_TP_jit, shared, draws, n_items, eval_times, exposure_samples,
                           chunk_size_divisor, _chunk_plan, _prepare_only)


def batch_lnL_EB(
    time: np.ndarray,
    flux_obs: np.ndarray,
    sigma: np.ndarray | float,
    reb_arr: np.ndarray,
    feb_arr: np.ndarray,
    p_orb: float,
    inc_arr: np.ndarray,
    a_cm_arr: np.ndarray | float,
    R_host: np.ndarray | float,
    u1: np.ndarray | float,
    u2: np.ndarray | float,
    exptime_days: float,
    ecc_arr: np.ndarray,
    w_arr: np.ndarray,
    companion_fluxratio: np.ndarray | float = 0.0,
    companion_is_host: bool = False,
    nsamples: int = 20,
    bin_integration_days: float = 0.0,
    *,
    u1_EB: np.ndarray | float,
    u2_EB: np.ndarray | float,
    secondary_to_host_flux_ratio=None,
    chunk_size_divisor: int = 1,
    _chunk_plan: Optional[dict] = None,
    _prepare_only: bool = False,
) -> np.ndarray | _PreparedBatch:
    """Evaluate log-likelihoods for an array of eclipsing binary draws concurrently.

    ``a_cm_arr``, ``R_host``, ``u1``, ``u2``, and ``companion_fluxratio`` each
    accept a scalar shared by every draw or a per-draw array (see
    ``batch_lnL_TP``). Required ``u1_EB, u2_EB`` describe the eclipsed
    secondary and likewise accept scalars or per-draw arrays; nonfinite values
    are refused, as in the CPU ``simulate_EB``.
    ``secondary_to_host_flux_ratio`` accepts F_secondary/F_primary instead of
    the legacy target-referenced feb_arr fraction, which must then be None.
    ``chunk_size_divisor`` reduces only the runtime working set for a retry.
    Private ``_prepare_only`` returns a deferred batch for scenario overlap.
    """
    if not HAS_JAX_TREX:
        raise RuntimeError("JAX/jaxoplanet is not available in the current environment")
    n_items = len(reb_arr)
    if n_items == 0:
        return np.empty(0, dtype=float)
    if secondary_to_host_flux_ratio is not None:
        from trex.models.flux_calibration import binary_flux_weights

        if feb_arr is not None:
            raise ValueError("explicit secondary-to-host flux requires feb_arr=None")
        binary_flux_weights(secondary_to_host_flux_ratio, companion_fluxratio, companion_is_host)
        feb_arr = _per_draw(secondary_to_host_flux_ratio, n_items, "secondary-to-host flux ratio")
    if any(np.asarray(value).dtype.kind not in "iuf" for value in (u1_EB, u2_EB)):
        raise ValueError("secondary limb-darkening coefficients must be finite real numbers")
    if not np.all(np.isfinite(u1_EB)) or not np.all(np.isfinite(u2_EB)):
        raise ValueError("secondary limb-darkening coefficients must be finite")
    inv_sigma_sq, eval_times, bin_weights, exposure_samples = _shared_observation_inputs(
        time, flux_obs, sigma, bin_integration_days, exptime_days, nsamples)
    shared = dict(
        time=eval_times, flux_obs=np.asarray(flux_obs), inv_sigma_sq=inv_sigma_sq,
        p_orb=float(p_orb), exptime_days=float(exptime_days),
        companion_is_host=bool(companion_is_host), nsamples=int(exposure_samples), bin_weights=bin_weights,
        secondary_to_host=secondary_to_host_flux_ratio is not None,
    )
    draws = dict(
        reb_arr=reb_arr, feb_arr=feb_arr, inc_arr=inc_arr, a_cm_arr=_per_draw(a_cm_arr, n_items, "a_cm"),
        R_host_arr=_per_draw(R_host, n_items, "R_host"), u1_arr=_per_draw(u1, n_items, "u1"),
        u2_arr=_per_draw(u2, n_items, "u2"), eb_u1_arr=_per_draw(u1_EB, n_items, "u1_EB"),
        eb_u2_arr=_per_draw(u2_EB, n_items, "u2_EB"), ecc_arr=ecc_arr, w_arr=w_arr,
        cfr_arr=_per_draw(companion_fluxratio, n_items, "companion_fluxratio"),
    )
    return _evaluate_batch(_batch_lnL_EB_jit, shared, draws, n_items, eval_times, exposure_samples,
                           chunk_size_divisor, _chunk_plan, _prepare_only)


__all__ = [
    "HAS_JAX_TREX",
    "JAX_USABLE_DEVICES",
    "batch_lnL_TP",
    "batch_lnL_EB",
]
