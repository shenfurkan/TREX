"""BatchEvaluator compile-once dispatch, matched values and cold/warm timing.

The measured object is the real ``batch_lnL_TP`` path on the CPU reference
device. ``_compiled_bytes`` used to compile a second copy of the kernel only to
read ``memory_analysis()`` while ``kernel(...)`` compiled the same signature
again for execution. Counts observe the compile that is paid for; repeated
execution must preserve bits, and direct JIT comparison checks matched inputs.
Neither establishes physical model accuracy.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.fixtures.synthetic_observations import synthetic_batch_likelihood_case


def _tp_inputs(count=8):
    """A valid target-neutral TP batch; execution tests need finite arrays only."""
    time, flux, params = synthetic_batch_likelihood_case()
    return dict(
        time=time, flux_obs=flux, sigma=params["exptime_days"],
        rp_arr=np.full(count, 7.0), p_orb=params["P_orb"],
        inc_arr=np.full(count, params["inc_deg"]), a_cm_arr=params["a_cm"],
        R_host=params["R_s_solar"], u1=params["u1"], u2=params["u2"],
        exptime_days=params["exptime_days"], ecc_arr=np.zeros(count),
        w_arr=np.full(count, np.degrees(np.pi / 2)), nsamples=params["nsamples"],
    )


class _CountingKernel:
    """The real jitted kernel, with compiles and direct dispatches observed."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.compiles = 0
        self.direct_calls = 0

    def lower(self, *args, **arguments):
        self.compiles += 1
        return self.kernel.lower(*args, **arguments)

    def __call__(self, *args, **arguments):
        self.direct_calls += 1
        return self.kernel(*args, **arguments)


@pytest.mark.parametrize("memory_analysis_available", [True, False])
def test_a_block_compiles_once_and_executes_through_the_retained_executable(monkeypatch, memory_analysis_available):
    """One compile supplies both the memory measurement and the execution."""
    from trex.accelerators import _jax_models as models
    from trex.accelerators import _jax_batch as batch

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    kernel = models._batch_lnL_TP_jit
    counting = _CountingKernel(kernel)
    monkeypatch.setattr(models, "_batch_lnL_TP_jit", counting)
    if not memory_analysis_available:
        monkeypatch.setattr(batch, "_required_bytes", lambda executable: None)
    inputs = _tp_inputs()
    count = inputs["rp_arr"].size

    with models.jax.default_device(models.jax.devices("cpu")[0]):
        prepared = models.batch_lnL_TP(**inputs, _prepare_only=True)
        prepared.prepare()
        first = prepared()
        second = prepared()
        jnp = models.jnp
        direct = np.asarray(kernel(
            jnp.asarray(inputs["time"]), jnp.asarray(inputs["flux_obs"]),
            jnp.asarray(np.broadcast_to(1.0 / inputs["sigma"] ** 2, inputs["time"].shape)),
            jnp.asarray(inputs["rp_arr"]), inputs["p_orb"], jnp.asarray(inputs["inc_arr"]),
            jnp.asarray(np.full(count, inputs["a_cm_arr"])),
            jnp.asarray(np.full(count, inputs["R_host"])),
            jnp.asarray(np.full(count, inputs["u1"])),
            jnp.asarray(np.full(count, inputs["u2"])), inputs["exptime_days"],
            jnp.asarray(inputs["ecc_arr"]), jnp.asarray(inputs["w_arr"]),
            jnp.asarray(np.zeros(count)), False, inputs["nsamples"], None,
        ))

    assert counting.compiles == 1, "each immutable block shape compiles once"
    assert counting.direct_calls == 0, "execution must use the compiled artifact, not a second JIT call"
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    # The retained executable runs the same HLO the JIT dispatch would compile;
    # this pins the execution-path change to the same log-likelihood values on
    # identical inputs, not merely to "some finite number".
    assert np.allclose(first, direct, rtol=1e-12, atol=1e-12)


def test_cold_and_warm_evaluation_return_the_same_draws(record_property):
    """Measure the compile and warm evaluation; reuse must not change a value."""
    from trex.accelerators import _jax_models as models

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    with models.jax.default_device(models.jax.devices("cpu")[0]):
        prepared = models.batch_lnL_TP(**_tp_inputs(), _prepare_only=True)
        started = time.perf_counter()
        prepared.prepare()
        compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        cold = prepared()
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        warm = prepared()
        warm_seconds = time.perf_counter() - started

    assert np.array_equal(cold, warm)
    assert compile_seconds > 0.0 and warm_seconds > 0.0
    # JUnit properties survive the runner's compacted report; a print would not.
    record_property("compile_seconds", f"{compile_seconds:.3f}")
    record_property("first_eval_seconds", f"{cold_seconds:.4f}")
    record_property("warm_eval_seconds", f"{warm_seconds:.4f}")
    print(f"\n[BATCH DISPATCH] compile={compile_seconds:.3f}s "
          f"first_eval={cold_seconds:.4f}s warm_eval={warm_seconds:.4f}s")


@pytest.mark.parametrize("memory_analysis_available", [True, False])
def test_native_unit_blocks_reuse_the_measured_executable(monkeypatch, memory_analysis_available, record_property):
    from trex.accelerators import _jax_models as models
    from trex.accelerators import _native_unit_batch as native_batch
    from trex.evidence import nested_evidence
    from tests.fixtures.synthetic_observations import synthetic_trex_dense_tp_integrand_case

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")
    reference = nested_evidence._TransitingPlanetIntegrand(
        **synthetic_trex_dense_tp_integrand_case(), host_mass=1.0, host_radius=1.0,
        nsamples=3, bin_integration_days=0.0, use_flat_priors=False, population=None, n_dimensions=4,
    )
    points = np.random.default_rng(312).uniform(size=(9, 4))
    device = models.jax.devices("cpu")[0]
    evaluator = native_batch.NativeUnitCubeBatch(reference, points, max_in_flight_blocks=1)
    real_kernel = evaluator._kernel(device)
    counting = _CountingKernel(real_kernel)
    evaluator._kernels[device] = counting
    if not memory_analysis_available:
        monkeypatch.setattr(native_batch, "_required_bytes", lambda executable: None)
    with models.jax.default_device(device):
        required = evaluator._compiled_bytes(device, evaluator.block)
        first = evaluator(slice(0, len(points)))
        second = evaluator(slice(0, len(points)))
        subset = evaluator(slice(2, 7))
        direct = real_kernel(models.jax.device_put(points, device))
    assert (required is not None) == memory_analysis_available
    assert counting.compiles == 1 and counting.direct_calls == 0
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(subset, first[2:7])
    assert np.any(np.isfinite(first)) and np.all(np.asarray(direct.numerically_valid))
    np.testing.assert_allclose(first, np.asarray(direct.log_integrand), rtol=1e-12, atol=1e-12)
    record_property("native_compiles", counting.compiles)
    record_property("native_direct_jit_calls", counting.direct_calls)
