"""Ordered native TREX unit-cube evaluation for defensive refinement.

Execution only: proposal generation, densities, RNG and the evidence/ESS
reductions stay in ``nested_evidence``. Values are log integrands in nats at
prior-quantile positions, from the real scenario callback with its own
photometry, priors and numerical-validity flags.

Runs through the shared measured-memory, device-pilot, lease and chunk
scheduler. Padding lanes repeat a retained point and never enter N or a
reduction; executed lanes (including pilots and retries) are counted per device.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

from trex.accelerators import _jax_models as models
from trex.accelerators._jax_batch import (
    OrderedBlockPipeline, _required_bytes, contiguous_range, evaluation_block, is_memory_exhaustion, pad_to_block,
    pipeline_capacity, validate_max_depth,
)
from trex.foundation.errors import NumericalLikelihoodError


class NativeUnitCubeBatch:
    """Device-owned scenario callbacks with bounded, ordered float64 outputs."""

    def __init__(self, reference, points, *, max_in_flight_blocks=None):
        self.max_in_flight_blocks = validate_max_depth(max_in_flight_blocks)
        self.reference = reference
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != reference.n_dimensions:
            raise ValueError("native unit positions must have shape (draws, scenario dimensions)")
        if not np.isfinite(self.points).all():
            raise ValueError("native unit positions must be finite")
        self.block = evaluation_block(len(self.points), len(reference.time) * reference.nsamples)
        self._kernels = {}
        self._dispatches = {}
        self._executables = {}
        self._pipelines = {}
        self._integrity_failures = []
        self._numerical_failure = None
        self._lock = Lock()
        # ``__call__`` reads ``self.points``, which a subset evaluation replaces
        # for its duration; evaluations of one instance therefore never overlap.
        self._evaluation_lock = Lock()
        # The scheduler caches device decisions by kernel identity. A unique
        # token keeps decisions per scenario object without the global cache
        # holding this instance's proposal array through a bound method.
        self._device_pilot_key = object()

    def _kernel(self, device):
        with self._lock:
            if device not in self._kernels:
                native = self.reference.jax_unit_integrand(device=device)
                self._kernels[device] = models.jax.jit(models.jax.vmap(native.evaluate))
                self._dispatches[device] = 0
            return self._kernels[device]

    def _record_failure(self, device, error):
        with self._lock:
            self._integrity_failures.append(f"{device}: {type(error).__name__}: {error}")
            if isinstance(error, NumericalLikelihoodError):
                self._numerical_failure = error

    def _raise_recorded_failure(self):
        if self._numerical_failure is not None:
            raise self._numerical_failure
        raise RuntimeError("native unit numerical/execution integrity failure: "
                           + "; ".join(self._integrity_failures))

    def _run_blocks(self, device, first, stop, block):
        jax, jnp = models.jax, models.jnp
        with self._lock:
            pipeline = self._pipelines.setdefault((device, block), OrderedBlockPipeline(self.max_in_flight_blocks))
        executable, required = self._executable(device, block)

        def launch(start, block):
            points = self.points[start:min(start + block, len(self.points))]
            return executable(jax.device_put(pad_to_block(points, block), device)), len(points)

        def read(result, count, block=block):
            jax.block_until_ready(result)
            with self._lock:
                self._dispatches[device] += block
            if result.log_integrand.dtype != jnp.float64 or result.log_integrand.devices() != {device}:
                raise RuntimeError("native likelihood escaped its float64 execution device")
            return (np.asarray(result.log_integrand, dtype=np.float64)[:count],
                    np.asarray(result.numerically_valid, dtype=bool)[:count])

        return pipeline.run(range(first, stop, block), launch, read,
                            lambda: pipeline_capacity(required, block * np.dtype(np.float64).itemsize, device),
                            device=device, block=block)

    def __call__(self, subset):
        start, stop = contiguous_range(subset, len(self.points))
        if stop <= start:
            return np.empty(0, dtype=np.float64)
        jax = models.jax
        device = jax.config.jax_default_device or jax.devices()[0]
        if self._integrity_failures:
            # A device already produced an integrity failure: the final result
            # will be refused, so do not spend the remaining work on another device.
            self._raise_recorded_failure()
        while True:
            block = self.block
            first = start - start % block
            try:
                blocks = self._run_blocks(device, first, stop, block)
            except Exception as error:
                if block <= 1 or not is_memory_exhaustion(error):
                    self._record_failure(device, error)
                    raise
                from trex.runtime.accelerator_pool import record_accelerator_event

                record_accelerator_event("native-unit-block-memory-retry", device, batch=block,
                                         reason="allocation exhausted memory", error_type=type(error).__name__)
                with self._lock:
                    self.block = min(self.block, max(1, block // 2))
                continue
            window = slice(start - first, stop - first)
            output = np.concatenate([values for values, _ in blocks])[window]
            valid = np.concatenate([flags for _, flags in blocks])[window]
            failed = ~valid | np.isnan(output) | np.isposinf(output)
            if np.any(failed):
                error = NumericalLikelihoodError("native likelihood returned supported numerical failures", int(failed.sum()))
                self._record_failure(device, error)
                raise error
            return output

    def _executable(self, device, block):
        """Retain the device/block executable even when memory analysis is unavailable."""
        key = (device, block)
        with self._lock:
            if key in self._executables:
                return self._executables[key]
        kernel = self._kernel(device)
        signature = models.jax.ShapeDtypeStruct((block, self.points.shape[1]), models.jnp.float64)
        executable = kernel.lower(signature).compile()
        entry = (executable, _required_bytes(executable))
        with self._lock:
            return self._executables.setdefault(key, entry)

    def _compiled_bytes(self, device, block):
        return self._executable(device, block)[1]

    def plan_chunk(self, initial, maximum, budget, device):
        """Measure this fixed kernel's working set; only allocation OOM shrinks it.

        Unknown headroom (``budget is None``) admits one immutable block, the
        same rule the scheduler applies to any unknown device counter.
        """
        from trex.runtime.accelerator_pool import record_accelerator_event

        with models.jax.default_device(device):
            while True:
                block = self.block
                try:
                    required = self._compiled_bytes(device, block)
                    if required is None:
                        record_accelerator_event("native-unit-compiler-memory-unavailable", device, batch=block)
                        return block
                except Exception as error:
                    if block <= 1 or not is_memory_exhaustion(error):
                        raise
                    record_accelerator_event("native-unit-compiler-memory-retry", device, batch=block,
                                             reason="compiler exhausted memory", error_type=type(error).__name__)
                    with self._lock:
                        self.block = min(self.block, max(1, block // 2))
                    continue
                if budget is None:
                    record_accelerator_event("native-unit-memory-headroom-unavailable", device, batch=block,
                                             bytes_required=required)
                    return block
                if required > budget:
                    if block <= 1:
                        raise MemoryError("native unit callback exceeds measured memory headroom")
                    record_accelerator_event("native-unit-measured-memory-plan", device, batch=block,
                                             bytes_required=required, available_budget=budget)
                    with self._lock:
                        self.block = min(self.block, max(1, block // 2))
                    continue
                record_accelerator_event("native-unit-compiled-memory", device, batch=block,
                                         bytes_required=required, available_budget=budget)
                # Ownership is separate from numerical granule and in-flight
                # depth. The pipeline re-probes capacity before each wave and
                # never treats this whole scheduler slice as one allocation.
                return max(block, min(int(maximum), max(int(initial), block)))

    def evaluate(self, *, subset=None):
        """Log integrands for every point, or for ``points[subset]`` in a recovery wave.

        A recovery wave keeps the compiled kernels, dispatch counters and
        scheduler state; it evaluates the subset as the current point set.
        """
        with self._evaluation_lock:
            if subset is None:
                return self._evaluate_points()
            points = self.points
            self.points = points[subset]
            try:
                return self._evaluate_points()
            finally:
                self.points = points

    def _evaluate_points(self):
        if len(self.points) == 0:
            return np.empty(0, dtype=np.float64)
        result = models._run_jax_chunks(len(self.points), len(self.reference.time), self.reference.nsamples,
                                        self, 1)
        # The shared auto-device pilot can demote a failed accelerator. A
        # supported numerical/dtype/callback failure is not merely a slower
        # device and must never become an adopted CPU fallback estimate.
        if self._integrity_failures:
            self._raise_recorded_failure()
        return result

    def provenance(self):
        """Observed execution counts, including padding, pilots and retries."""
        return {
            "backend": "jax-native-unit-cube",
            "logical_likelihood_calls": len(self.points),
            "reduction_backend": "numpy-cpu-original-order",
            "device_dispatches": [
                {"device": str(device), "platform": device.platform, "device_id": device.id,
                 "float_dtype": "float64", "executed_kernel_lanes": count}
                for device, count in self._dispatches.items()
            ],
            "final_evaluation_block": self.block,
            "adaptive_pipeline": [
                {"device": str(device), "evaluation_block": block, **pipeline.provenance()}
                for (device, block), pipeline in self._pipelines.items()
            ],
        }
