"""Device-resident TREX inputs and compiler-measured batch working sets.

This changes execution only: independent float64 likelihoods are returned in
their input order. XLA's memory analysis includes arguments, outputs and live
temporaries minus aliased buffers; it is a compiled requirement, not the old
model-grid multiplier. Unsupported counters stay explicitly unknown.
See https://docs.jax.dev/en/latest/notebooks/host-offloading.html#memory-analysis.
"""

from __future__ import annotations

from threading import Lock
import time

import numpy as np


# Algorithmic, not empirical: draws per kernel call. Each call evaluates one
# block aligned to the start of the draw array, so a draw is always computed by
# the same compiled shape at the same index whatever the chunk plan, memory
# headroom or device count. XLA promises no bitwise agreement across batch
# shapes (CPU log-likelihoods differed by 1.6e-16 for shapes 1 and 2), so without
# this a replay with different free memory could move an FPP. The element bound
# keeps one block's working set near 128 MiB at the 8x working-set overhead.
_BLOCK_ELEMENTS = 1 << 21
_BLOCK_MAX_DRAWS = 1 << 14


def evaluation_block(n_items: int, elements_per_draw: int) -> int:
    """Draws per kernel call, fixed by the data shape and the draw count.

    A batch that fits one block is that block (no padding); a larger one
    uses the power-of-two shape cap, so its compiled shape is shared.
    """
    cap = min(max(1, _BLOCK_ELEMENTS // max(1, int(elements_per_draw))), _BLOCK_MAX_DRAWS)
    return min(max(1, int(n_items)), 1 << (cap.bit_length() - 1))


def is_memory_exhaustion(exc: Exception) -> bool:
    """Only allocation exhaustion permits a smaller-working-set retry."""
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in (
        "resource_exhausted", "out of memory", "cuda_error_out_of_memory",
        "failed to allocate", "allocation failed",
    ))


def _required_bytes(executable):
    """Compiler-measured working set of one compiled executable, or ``None``.

    XLA's analysis includes arguments, outputs and live temporaries minus
    aliased buffers; it is a compiled requirement, not a model-grid estimate.
    A field a backend cannot report leaves the working set unknown rather than
    turning a guess into an admission decision.
    """
    analysis = executable.memory_analysis()
    if analysis is None:
        return None
    fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    values = [getattr(analysis, name, None) for name in fields]
    alias = getattr(analysis, "alias_size_in_bytes", None)
    if any(value is None for value in (*values, alias)):
        return None
    return sum(values) - alias


def pipeline_capacity(required_bytes, output_bytes, device, *, granule_floor=False, resident_bytes=0):
    """Live capacity for immutable kernel blocks, not an inferred GPU class.

    Compiler bytes include arguments/outputs/temporaries minus aliases. The
    shared inputs among those arguments (``resident_bytes``) are uploaded once
    per device and are already outside the measured headroom, so each further
    in-flight block costs only ``required_bytes - resident_bytes``. The
    existing device budget owns driver/allocator probes and user overrides;
    host headroom separately bounds pending readback. Unknown counters admit
    one block, not a guessed memory floor. Units are bytes throughout.
    """
    from trex.accelerators import _jax_models as models
    from trex.runtime.hardware import memory_resource_profile

    budget = models._jax_batch_memory_budget_bytes(device)
    host = memory_resource_profile()["available_bytes"]
    row = {"device_budget_bytes": budget, "host_available_bytes": host,
           "compiled_block_bytes": required_bytes, "output_block_bytes": output_bytes,
           "resident_input_bytes": resident_bytes}
    if required_bytes is None or budget is None or host is None:
        return {**row, "capacity_blocks": 1, "reason": "capacity counter unavailable; single block only"}
    if required_bytes <= 0 or output_bytes <= 0 or resident_bytes < 0:
        raise ValueError("compiled working set and output sizes must be positive")
    per_block = max(required_bytes - resident_bytes, output_bytes)
    capacity = min(budget // per_block, host // output_bytes)
    if capacity < 1:
        if granule_floor and host >= output_bytes:
            # BatchEvaluator's existing ABI rounds scheduler budgets smaller
            # than one immutable numerical granule UP to that granule (see
            # _whole_blocks). Preserve that explicitly recorded floor rather
            # than silently changing shape/rounding for a one-draw override.
            return {**row, "capacity_blocks": 1, "reason": "scheduler budget below immutable granule; existing single-block floor"}
        raise MemoryError("one immutable likelihood block exceeds measured headroom")
    return {**row, "capacity_blocks": capacity, "reason": "live device/compiler/host capacity"}


def validate_max_depth(max_depth):
    """``None`` or a positive integer ceiling on in-flight blocks."""
    if max_depth is not None and (isinstance(max_depth, (bool, np.bool_))
                                  or not isinstance(max_depth, (int, np.integer)) or max_depth < 1):
        raise ValueError("max_depth must be a positive integer or None")
    return int(max_depth) if max_depth is not None else None


def pad_to_block(values: np.ndarray, block: int) -> np.ndarray:
    """Fill unused kernel positions by repeating the last retained row.

    Padded lanes are discarded before any evidence reduction: they are not
    synthetic prior draws and never enter N/ESS/FPP.
    """
    missing = block - len(values)
    if missing <= 0:
        return values
    return np.concatenate((values, np.repeat(values[-1:], missing, axis=0)))


def contiguous_range(draws: slice, n_items: int):
    """``(start, stop)`` of a contiguous slice over ``n_items`` draws."""
    start, stop, step = draws.indices(n_items)
    if step != 1:
        raise ValueError("draw slices must be contiguous")
    return start, max(start, stop)


# One slower wave may be thermal/JIT-warmup noise; require a second miss to settle.
_TUNE_CONFIRMATIONS = 2


class OrderedBlockPipeline:
    """Capacity-bounded asynchronous waves, tuned on actual retained work.

    Powers-of-two concurrency exploration changes work in flight, never the
    compiled numerical granule. Compare completed-wave throughput and retain the
    best observed depth when growth stops helping, but only after
    ``_TUNE_CONFIRMATIONS`` consecutive non-improving full-width waves; an
    improving wave clears that evidence. A partial tail wave or a wave clamped
    by fresh counters never tunes, and every wave is clamped against those
    counters. No extra draws/RNG or synthetic workload are generated. This is
    bounded feedback, not a guarantee of optimal device saturation.
    """

    def __init__(self, max_depth=None):
        self.max_depth = validate_max_depth(max_depth)
        self.depth = 1
        self.best_depth = 1
        self.best_rate = 0.0
        self.settled = False
        self.oom_limit = None
        self.waves = {}
        self._misses = 0

    def _depth_limit(self, probe) -> int:
        limit = probe["capacity_blocks"]
        if self.max_depth is not None:
            limit = min(limit, self.max_depth)
        if self.oom_limit is not None:
            limit = min(limit, self.oom_limit)
        return limit

    @staticmethod
    def _drain(pending, read):
        """Finish outstanding asynchronous work; return a non-allocation failure.

        A numerical failure in any batch outranks allocation OOM and may not
        disappear into a smaller-working-set retry.
        """
        integrity_error = None
        for output in pending:
            if output is None:
                continue
            try:
                read(*output)
            except Exception as drain_error:
                if not is_memory_exhaustion(drain_error):
                    integrity_error = drain_error
        return integrity_error

    def _record_wave(self, probe, width, block, seconds, device):
        from trex.runtime.accelerator_pool import record_accelerator_event

        if not np.isfinite(seconds) or seconds <= 0:
            raise RuntimeError("pipeline throughput clock returned a nonpositive/nonfinite interval")
        rate = width * block / seconds
        row = {**probe, "in_flight_blocks": width, "effective_batch_draws": width * block,
               "seconds": seconds, "kernel_lanes_per_second": rate}
        # Bound in-memory provenance by observed depths, not N/block. The
        # event stream retains individual waves, including fresh counters.
        aggregate = self.waves.setdefault(width, {**row, "wave_count": 0,
                                                   "total_seconds": 0.0, "total_kernel_lanes": 0})
        aggregate.update(row)
        aggregate["wave_count"] += 1
        aggregate["total_seconds"] += seconds
        aggregate["total_kernel_lanes"] += width * block
        record_accelerator_event("adaptive-block-wave", device, **row)
        return rate

    def _tune(self, rate, width, depth, limit):
        # A partial (tail) wave or one clamped by fresh counters cannot show
        # that the requested depth became slower, so neither is evidence.
        if width != depth or depth != self.depth or self.settled:
            return
        if rate > self.best_rate:
            self.best_rate, self.best_depth = rate, depth
            self.depth = min(limit, depth * 2)
            self._misses = 0
        else:
            self._misses += 1
            if self._misses >= _TUNE_CONFIRMATIONS:
                self.depth, self.settled = self.best_depth, True

    def run(self, starts, launch, read, capacity, *, device, block):
        from trex.runtime.accelerator_pool import record_accelerator_event

        values, cursor = [], 0
        while cursor < len(starts):
            probe = capacity()
            limit = self._depth_limit(probe)
            depth = min(self.depth, limit)
            width = min(depth, len(starts) - cursor)
            pending, completed = [], []
            started = time.perf_counter()
            try:
                for start in starts[cursor:cursor + width]:
                    pending.append(launch(start, block))
                for index, output in enumerate(pending):
                    # read may update counters before raising; never consume that output twice.
                    pending[index] = None
                    completed.append(read(*output))
            except Exception as error:
                integrity_error = self._drain(pending, read)
                if integrity_error is not None:
                    raise integrity_error from error
                if not is_memory_exhaustion(error) or width <= 1:
                    raise
                self.oom_limit = max(1, width // 2)
                self.depth = self.best_depth = min(self.best_depth, self.oom_limit)
                self.settled = True
                record_accelerator_event("pipeline-memory-retry", device, in_flight_blocks=width,
                                         retry_blocks=self.depth, reason="pipeline exhausted memory",
                                         error_type=type(error).__name__)
                continue  # same keys/draws/order; partial wave is not committed
            rate = self._record_wave(probe, width, block, time.perf_counter() - started, device)
            values.extend(completed)
            cursor += width
            self._tune(rate, width, depth, limit)
        return values

    def provenance(self):
        return {"next_depth": self.depth, "best_observed_depth": self.best_depth,
                "operator_depth_ceiling": self.max_depth,
                "best_observed_kernel_lanes_per_second": self.best_rate,
                "oom_depth_limit": self.oom_limit, "waves": list(self.waves.values())}


class BatchEvaluator:
    """One real JIT kernel with immutable inputs uploaded once per device.

    Calling it with a slice of draw indices returns their float64
    log-likelihoods in order; an empty slice returns an empty array.
    """

    def __init__(self, jax, jnp, kernel, shared: dict, draws: dict, elements_per_draw: int) -> None:
        self.jax, self.jnp, self.kernel = jax, jnp, kernel
        self.shared, self.draws = shared, draws
        self.n_items = len(next(iter(draws.values())))
        self.block = evaluation_block(self.n_items, elements_per_draw)
        self._device_inputs: dict = {}
        self._executables: dict = {}
        self._compile_limits: dict = {}
        self._pipelines: dict = {}
        self._lock = Lock()

    def _device(self):
        """The device a kernel call made now would use; one key for every cache."""
        device = self.jax.config.jax_default_device
        return device if device is not None else self.jax.devices()[0]

    def _shared_arguments(self) -> dict:
        device = self._device()
        with self._lock:
            if device not in self._device_inputs:
                self._device_inputs[device] = {
                    name: self.jnp.asarray(value, dtype=self.jnp.float64)
                    if isinstance(value, np.ndarray) else value
                    for name, value in self.shared.items()
                }
            return self._device_inputs[device]

    def _resident_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._shared_arguments().values() if hasattr(value, "nbytes"))

    def __call__(self, draws: slice) -> np.ndarray:
        start, stop = contiguous_range(draws, self.n_items)
        if stop <= start:
            return np.empty(0, dtype=float)
        while True:
            block = self.block
            first = start - start % block
            try:
                values = self._pipelined(range(first, stop, block), block)
            except Exception as exc:
                if block <= 1 or not is_memory_exhaustion(exc):
                    raise
                from trex.runtime.accelerator_pool import record_accelerator_event

                # Only a real allocation failure may shrink the block; bitwise
                # plan invariance then no longer holds and the event says so.
                record_accelerator_event("evaluation-block-memory-retry", self._device(),
                                         batch=block, reason="allocation exhausted memory",
                                         error_type=type(exc).__name__)
                with self._lock:
                    self.block = min(self.block, block // 2)
                continue
            return np.concatenate(values)[start - first:stop - first]

    def _pipelined(self, starts: range, block: int) -> list:
        """Adapt in-flight blocks to actual capacity/throughput, retaining shape."""
        device = self._device()
        with self._lock:
            pipeline = self._pipelines.setdefault((device, block), OrderedBlockPipeline())
        required = self._compiled_bytes(block)
        resident = self._resident_bytes()
        return pipeline.run(starts, self._launch_block, self._read_block,
                            lambda: pipeline_capacity(required, block * np.dtype(np.float64).itemsize,
                                                      device, granule_floor=True, resident_bytes=resident),
                            device=device, block=block)

    @staticmethod
    def _read_block(output, count: int) -> np.ndarray:
        return np.asarray(output, dtype=float)[:count]

    def _launch_block(self, start: int, block: int) -> tuple:
        stop = min(start + block, self.n_items)
        arguments = dict(self._shared_arguments())
        for name, value in self.draws.items():
            # ascontiguousarray copies only a zero-stride scalar view; a contiguous
            # draw slice is uploaded as is. ``jnp.asarray`` is the upload, on the
            # current default device: a device allocation failure surfaces here
            # and shrinks the block.
            arguments[name] = self.jnp.asarray(
                pad_to_block(np.ascontiguousarray(value[start:stop], dtype=np.float64), block),
                dtype=self.jnp.float64)
        executable, _ = self._executable(block)
        # JAX binds static keywords at compile time; only dynamic inputs remain.
        return executable(**{name: arguments[name] for name in executable.in_avals[1]}), stop - start

    def _executable(self, size: int):
        """``(compiled, required_bytes_or_None)`` for one immutable block shape.

        One ``lower().compile()`` both measures the working set and supplies the
        executable that runs the block. Compiling a separate copy only to read
        ``memory_analysis()`` left the JIT cache with a second executable of the
        same signature, so every scenario compiled its kernel twice.
        """
        device = self._device()
        key = (device, size)
        with self._lock:
            limit = self._compile_limits.get(device)
            if limit is not None and size > limit:
                # An already observed compiler allocation failure is evidence, not
                # a shape to repropose every time a queue acquires another slice.
                raise MemoryError("requested block exceeds retained compiler allocation ceiling")
            cached = self._executables.get(key)
            if cached is not None:
                return cached
        arguments = dict(self._shared_arguments())
        arguments.update({
            name: self.jax.ShapeDtypeStruct((size,), self.jnp.float64)
            for name in self.draws
        })
        try:
            # Compile outside the lock: it is slow and other devices' callers
            # must not wait on it. A duplicate compile only repeats a measurement.
            executable = self.kernel.lower(**arguments).compile()
        except Exception as exc:
            if is_memory_exhaustion(exc):
                from trex.runtime.accelerator_pool import record_accelerator_event

                record_accelerator_event("compiler-memory-retry", device, batch=size,
                                         reason="compiler exhausted memory", error_type=type(exc).__name__)
                # Retain this evaluator's observed OOM ceiling. Fresh driver
                # counters cannot prove the compiler's larger shape now fits.
                with self._lock:
                    self._compile_limits[device] = min(
                        self._compile_limits.get(device, size), max(1, size // 2),
                    )
            raise
        entry = (executable, _required_bytes(executable))
        with self._lock:
            # A concurrent caller may have stored this key first; execute the
            # executable the evaluator retained so every block shares one copy.
            return self._executables.setdefault(key, entry)

    def _compiled_bytes(self, size: int):
        return self._executable(size)[1]
