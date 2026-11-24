"""Device selection and execution for JAX likelihood batches."""
from __future__ import annotations

from trex.foundation.environment import read_environment
from trex.runtime.hardware_resources import is_memory_exhaustion
from trex.runtime.logging import host_logger
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Event, Lock
from typing import Callable, Optional, Tuple
import numpy as np
from trex.accelerators._jax_batch import BatchEvaluator


def _models():
    """The ``_jax_models`` facade: tests patch names there, so look them up late."""
    from trex.accelerators import _jax_models

    return _jax_models


def _jax_batch_device_memory_bytes(device=None) -> Optional[int]:
    """Measured driver/allocator headroom on the actual execution device.

    An explicit device is probed directly, without allocating on the default
    device. Legacy no-argument calls probe actual default placement. The shared
    hardware probe bounds driver-free VRAM by allocator headroom when both
    counters exist; an allocator capacity alone is not available memory.
    ``None`` means unknown memory, never zero. CPU memory uses OS counters.
    """
    if not _models().HAS_JAX_TREX:
        return None
    try:
        from trex.runtime.hardware import device_memory_available_bytes

        devices = (device,) if device is not None else _models().jnp.zeros(()).devices()
        reported = [device_memory_available_bytes(item) for item in devices]
    except Exception:  # noqa: BLE001 - an unreadable device is "unknown", never zero
        return None
    known = [limit for limit in reported if isinstance(limit, int) and limit >= 0]
    return min(known) if known else None


def _jax_batch_memory_budget_bytes(device=None) -> Optional[int]:
    """Bytes one vmapped chunk may occupy on the device it will run on.

    ``TREX_JAX_BATCH_MEMORY_BYTES`` wins when set. Accelerators budget the
    compiled working set from measured driver/allocator headroom, leaving ten percent
    outside it. GPU temporaries are not capped by a host-RAM model-grid guess.
    CPU estimates use remaining host/cgroup RAM. Unknown accelerator headroom
    permits a one-draw probe, not a host-RAM guess; unknown CPU RAM raises a
    recorded capacity failure. There is no fabricated byte budget or floor.
    This is a working-set estimate, not a guarantee against concurrent OOM.

    Raises:
        ValueError: If the override is set to something that is not a positive
            integer. A misread override would silently serialize a batch or
            exhaust the device, so it fails instead of falling back.
    """
    override = read_environment(
        _models().JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, "EXONYM_JAX_BATCH_MEMORY_BYTES"
    )
    if override is not None and override.strip():
        try:
            parsed = int(override.strip())
        except ValueError as exc:
            raise ValueError(
                "{0} must be a positive integer, got {1!r}".format(
                    _models().JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, override
                )
            ) from exc
        if parsed < 1:
            raise ValueError(
                "{0} must be a positive integer, got {1!r}".format(
                    _models().JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, override
                )
            )
        return parsed
    from trex.runtime.hardware import memory_resource_profile

    reported = (
        _models()._jax_batch_device_memory_bytes() if device is None
        else _models()._jax_batch_device_memory_bytes(device)
    )
    if device is None:
        device = _models().jax.config.jax_default_device
        if device is None:
            device = next(iter(_models().jnp.zeros(()).devices()))
    if device.platform not in ("cpu", "interpreter"):
        if reported is None:
            from trex.runtime.accelerator_pool import record_accelerator_event

            record_accelerator_event("device-memory-unavailable", device, batch=1)
            host_logger("trex").warning("%s: device memory unavailable; probing with one draw, no guessed RAM/VRAM budget", device)
            return None
        # GPU model grids live in VRAM, not the notebook instance's host RAM.
        budget = reported * _models()._JAX_COMPILED_MEMORY_FRACTION.numerator // _models()._JAX_COMPILED_MEMORY_FRACTION.denominator
    else:
        available = memory_resource_profile()["available_bytes"]
        counters = [value for value in (reported, available) if value is not None]
        if not counters:
            raise MemoryError("memory counters unavailable for a TREX JAX batch; no budget can be inferred")
        budget = min(counters) // _models()._JAX_BATCH_MEMORY_DEVICE_DIVISOR
    if budget <= 0:
        raise MemoryError("no measured memory headroom remains for a TREX JAX batch")
    return budget


def _jax_chunk_size(n_points: int, nsamples: int, device=None) -> int:
    """Largest number of draws whose vmapped evaluation fits the memory budget."""
    per_draw = max(int(n_points) * max(int(nsamples), 1) * np.dtype(np.float64).itemsize * _models()._JAX_BATCH_MEMORY_OVERHEAD, 1)
    budget = _models()._jax_batch_memory_budget_bytes(device)
    return max(1, budget // per_draw) if budget is not None else 1


def _pilot_parity(reference: np.ndarray, values: np.ndarray) -> dict:
    """Float64 agreement of one device with the CPU reference on identical draws."""
    same_nonfinite = all(
        np.array_equal(test(reference), test(values)) for test in (np.isnan, np.isposinf, np.isneginf)
    )
    finite = np.isfinite(reference) & np.isfinite(values)
    if finite.any():
        relevant = finite & (reference >= reference[finite].max() - _models()._FLOAT64_LOG_UNDERFLOW)
        max_abs = float(np.max(np.abs(values[relevant] - reference[relevant])))
    else:
        max_abs = 0.0
    return {
        "same_nonfinite_pattern": same_nonfinite,
        "max_abs_lnL_difference": max_abs,
        "tolerance": _models().DEVICE_PILOT_LNL_TOLERANCE,
        "passed": bool(same_nonfinite and max_abs <= _models().DEVICE_PILOT_LNL_TOLERANCE),
    }


def _compile_pilot(evaluate: BatchEvaluator, device) -> tuple[int, int | None]:
    """Compile the device's admitted block and report its shape and working bytes."""
    import time as _time

    started = _time.perf_counter()
    with _models().jax.default_device(device):
        while True:
            block = evaluate._block_for(device)
            try:
                required = evaluate._compiled_bytes(block)
                _models().jax.block_until_ready(evaluate._shared_arguments())
                break
            except Exception as exc:
                if block <= 1 or not is_memory_exhaustion(exc):
                    raise
                from trex.runtime.accelerator_pool import record_accelerator_event

                record_accelerator_event("evaluation-block-memory-retry", device,
                                         batch=block, reason="allocation exhausted memory",
                                         error_type=type(exc).__name__)
                evaluate._retain_smaller_block(device, block)
    with evaluate._lock:
        if not hasattr(evaluate, "_pilot_compile_seconds"):
            evaluate._pilot_compile_seconds = {}
        evaluate._pilot_compile_seconds.setdefault(device, _time.perf_counter() - started)
    return block, required


# Placement decisions are shared by concurrent scenario workers.
_DECISIONS_LOCK = Lock()


def _decision_key(evaluate, devices, n_points: int, nsamples: int) -> tuple:
    """Kernel, per-draw work and device set that one measured placement applies to.

    Relative CPU/accelerator throughput depends on the light-curve size, so the
    same kernel evaluated on a different number of points gets its own pilot.
    """
    kernel = (getattr(evaluate, "_device_pilot_key", None) or getattr(evaluate, "kernel", None)
              or getattr(evaluate, "__qualname__", type(evaluate).__name__))
    return kernel, int(n_points) * max(int(nsamples), 1), tuple(str(device) for device in devices)


def _cached_decision(key):
    with _DECISIONS_LOCK:
        return _models()._DEVICE_DECISIONS.get(key)


def _terminal_exhaustion(evaluate) -> bool:
    """A BatchEvaluator only lets exhaustion escape at one draw and one block in flight.

    Smaller scheduler slices cannot shrink that working set further, so
    repeating the same draws would only repeat the same failure.
    """
    return isinstance(evaluate, _models().BatchEvaluator)


def _timed_pilot(evaluate: Callable[[slice], np.ndarray], device, draws: int) -> Tuple[np.ndarray, float, int]:
    """AOT compile, then time one real full pilot; halve only on OOM.

    BatchEvaluator always launches its immutable block shape, not the scheduler
    slice's shape. Other evaluator ABIs retain their existing warm-up contract.
    """
    import time as _time

    while True:
        try:
            with _models().jax.default_device(device):
                if isinstance(evaluate, _models().BatchEvaluator):
                    _models()._compile_pilot(evaluate, device)
                else:
                    evaluate(slice(0, draws))
                started = _time.perf_counter()
                values = evaluate(slice(0, draws))
                return values, _time.perf_counter() - started, draws
        except Exception as exc:
            if draws <= 1 or not is_memory_exhaustion(exc) or _terminal_exhaustion(evaluate):
                raise
        draws = max(1, draws // 2)


def _device_pilot(evaluate: Callable[[slice], np.ndarray], devices: tuple, n_items: int,
                  n_points: int, nsamples: int, result: np.ndarray) -> Tuple[tuple, int]:
    """Choose this kernel's devices from measured throughput and float64 parity.

    The CPU is the reference device. An accelerator is eligible only if it
    reproduces the reference on the same draws within the stated tolerance;
    eligible accelerators are used (concurrently) only when their summed
    throughput beats the CPU. Returns the chosen devices and the number of
    leading draws already evaluated on them (written to ``result``). A cached
    decision for the same kernel and device set runs no pilot. Virtual devices
    prove dispatch only, not physical performance.

    The chosen device's own pilot evaluation *is* its production evaluation of
    those draws: same executable, same inputs, so keeping it repeats no block
    and every kept value still comes from the chosen device. The CPU reference
    is therefore never substituted for an accelerator draw. The prefix retained
    from a winning accelerator is the part every chosen accelerator reproduced,
    so the parity evidence covers the values that enter the result.
    """
    from trex.runtime.accelerator_pool import record_accelerator_event

    key = _decision_key(evaluate, devices, n_points, nsamples)
    prepared = getattr(evaluate, "_cpu_pilot", None)
    if prepared is not None:
        del evaluate._cpu_pilot
    cached = _cached_decision(key)
    if cached is not None:
        return cached, 0
    cpu = _models().jax.devices("cpu")[0]
    if prepared is not None:
        # A measured CPU reference and CPU rate describe the same draws on the
        # same CPU device whatever set admission finally leases, so narrowing
        # or reordering the accelerator set must not repeat that pilot.
        reference, cpu_seconds, draws, overlap = prepared
        cpu_draws = draws
        # Never index past this call's result: a caller may dispatch a deferred
        # batch for fewer draws than its prepared predecessor measured.
        draws = min(draws, n_items)
    else:
        draws = _models()._whole_blocks(min(n_items, _models().DEVICE_PILOT_DRAWS, _models()._jax_chunk_size(n_points, nsamples, cpu)),
                               evaluate, n_items, cpu)
        reference, cpu_seconds, draws = _models()._timed_pilot(evaluate, cpu, draws)
        cpu_draws = draws
        overlap = None
    rows = [{"device": str(cpu), "draws": cpu_draws, "seconds": cpu_seconds,
             "draws_per_second": cpu_draws / cpu_seconds if cpu_seconds > 0 else None, "reference": True}]
    eligible, accelerator_rate = [], 0.0
    retained, retained_draws = reference, draws
    for device in devices:
        if device.platform in ("cpu", "interpreter"):
            continue
        row = {"device": str(device), "reference": False}
        try:
            values, seconds, used = _models()._timed_pilot(evaluate, device, draws)
        except Exception as exc:  # noqa: BLE001 - a failing pilot makes the device ineligible, recorded
            row.update(eligible=False, reason=f"pilot failed: {type(exc).__name__}: {exc}")
        else:
            parity = _models()._pilot_parity(reference[:used], values)
            rate = used / seconds if seconds > 0 else float("inf")
            row.update(draws=used, seconds=seconds, draws_per_second=rate, parity=parity,
                       eligible=parity["passed"],
                       reason=None if parity["passed"] else "float64 parity with the CPU reference failed")
            if parity["passed"]:
                eligible.append(device)
                accelerator_rate += rate
                if len(eligible) == 1:
                    retained, retained_draws = values, used
                else:
                    retained_draws = min(retained_draws, used)
        rows.append(row)
    cpu_rate = rows[0]["draws_per_second"] or float("inf")
    chosen = tuple(eligible) if eligible and accelerator_rate > cpu_rate else (cpu,)
    reason = (
        "eligible accelerators beat the CPU at float64 parity" if chosen != (cpu,)
        else "no accelerator passed float64 parity" if not eligible
        else "CPU throughput >= summed eligible accelerator throughput"
    )
    for row, device in zip(rows, (cpu, *(d for d in devices if d.platform not in ("cpu", "interpreter")))):
        row["compile_seconds"] = getattr(evaluate, "_pilot_compile_seconds", {}).get(device)
    decision = {"kernel": str(key[0]), "chosen": [str(device) for device in chosen], "reason": reason,
                 "cpu_draws_per_second": cpu_rate, "accelerator_draws_per_second": accelerator_rate,
                "pilot": rows, "cpu_reference_overlap": overlap,
                "cpu_timing_condition": "previous-scenario-in-flight" if overlap else "serial",
                "accelerator_timing_condition": "serial-after-previous-scenario"}
    record_accelerator_event("device-pilot", chosen[0], **decision)
    host_logger("trex").info(
        "TREX device pilot: chose %s (%s)", decision["chosen"], reason,
        extra={"event": "trex.device_pilot", **decision},
    )
    measured = chosen
    with _DECISIONS_LOCK:
        # A concurrent worker may have measured the same key first; keep one answer.
        chosen = _models()._DEVICE_DECISIONS.setdefault(key, chosen)
    if chosen == measured and chosen != (cpu,):
        # Report only the prefix every chosen accelerator reproduced, taken from
        # the device that will own the leading slice.
        result[:retained_draws] = retained[:retained_draws]
        return chosen, retained_draws
    if chosen != (cpu,):
        # A concurrent worker published a different accelerator set; only that
        # set's own pilot values may be reused, and this worker has none.
        return chosen, 0
    result[:draws] = reference[:draws]
    return chosen, draws


def _whole_blocks(draws: int, evaluate, n_items: int, device) -> int:
    """Round a chunk to whole evaluation blocks so no block is split across chunks.

    A block is the evaluator's numerical granule; a budget below one block
    still evaluates a whole block, whose working set is bounded by design.
    """
    block = evaluate._block_for(device) if isinstance(evaluate, _models().BatchEvaluator) else getattr(evaluate, "block", 1)
    return min(max(int(n_items), 1), max(block, int(draws) // block * block))


class _PreparedBatch:
    """Carry deterministic setup and an optional CPU pilot between scenarios."""

    def __init__(self, evaluate, n_points, nsamples):
        self.evaluate, self.n_points, self.nsamples = evaluate, n_points, nsamples

    def prepare(self, previous=None):
        from trex.runtime.accelerator_pool import execution_devices, record_accelerator_event

        evaluate = self.evaluate
        configured = _models().jax.config.jax_default_device
        devices = ((configured,) if configured is not None and configured.platform in ("cpu", "interpreter")
                   else execution_devices(_models()._usable_jax_devices()) or (_models().jax.devices("cpu")[0],))
        key = _decision_key(evaluate, devices, self.n_points, self.nsamples)
        decided = _cached_decision(key)
        # A preceding pilot of this same kernel will publish its decision before
        # this scenario runs; do not speculatively repeat its CPU reference.
        if (any(d.platform not in ("cpu", "interpreter") for d in devices)
                and decided is None
                and (previous is None or previous[2] is not evaluate.kernel)):
            cpu = _models().jax.devices("cpu")[0]
            draws = _models()._whole_blocks(min(evaluate.n_items, _models().DEVICE_PILOT_DRAWS,
                                      _models()._jax_chunk_size(self.n_points, self.nsamples, cpu)),
                                   evaluate, evaluate.n_items, cpu)
            _models()._compile_pilot(evaluate, cpu)
            overlap = previous[1] if previous is not None and not previous[0].done() else None
            reference, seconds, draws = _models()._timed_pilot(evaluate, cpu, draws)
            # The reference and CPU rate are device-set independent, so dispatch
            # may reuse them after admission leases a different accelerator set.
            evaluate._cpu_pilot = (reference, seconds, draws, overlap)
        for device in decided or devices:
            # Compilation/upload is allowed during the preceding MC; accelerator
            # evaluation/timing occurs only after _ScenarioOverlap.wait().
            if device.platform in ("cpu", "interpreter"):
                _models()._compile_pilot(evaluate, device)
            else:
                try:
                    _models()._compile_pilot(evaluate, device)
                except Exception as error:
                    # The serial pilot/planner owns retries; record the skipped
                    # device instead of letting it disappear from dispatch.
                    record_accelerator_event("device-compile-failed", device,
                                             error_type=type(error).__name__)
                    host_logger("hardware").warning(
                        "%s: device compile failed; skipping this device (%s)",
                        device, type(error).__name__,
                        extra={"event": "trex.hardware.compile_failed",
                               "error_type": type(error).__name__, "device": str(device)},
                    )
                    continue

    def __call__(self, *, chunk_size_divisor=1, _chunk_plan=None):
        return _models()._run_jax_chunks(self.evaluate.n_items, self.n_points, self.nsamples,
                               self.evaluate, chunk_size_divisor, _chunk_plan)


def _plan_batch_chunk(evaluate, initial, maximum, budget, device):
    """Plan ownership from the actual executable, never compile a scheduler-sized kernel."""

    if budget is None:
        return 1
    block, required = _models()._compile_pilot(evaluate, device)
    log = host_logger("trex")
    if required is None or required <= 0:
        from trex.runtime.accelerator_pool import record_accelerator_event

        record_accelerator_event("compiler-memory-unavailable", device, batch=block)
        log.warning("%s: compiler memory analysis unavailable; using one immutable block (%d draws)",
                    device, block)
        return block
    proposed = max(1, min(maximum, block * budget // required))
    proposed = min(maximum, 1 << (proposed.bit_length() - 1))
    log.info(
        "TREX %s: batch=%d, compiled block=%d draws/%d bytes, memory budget=%d bytes",
        device, proposed, block, required, budget,
        extra={"event": "trex.batch_plan"},
    )
    return proposed


def _run_jax_chunks(
    n_items: int,
    n_points: int,
    nsamples: int,
    evaluate: Callable[[slice], np.ndarray],
    chunk_size_divisor: int,
    chunk_plan: Optional[dict] = None,
    _allocated_devices: Optional[tuple] = None,
) -> np.ndarray:
    """Disjoint draw slices with one concurrent, work-taking worker per device.

    Each worker synchronizes its result to host memory before starting its next
    chunk, bounding in-flight device allocations. After each device's first
    chunk, ready devices take unstarted draws from the common queue instead of
    waiting for a slower peer's static assignment. Different-sized chunks retain
    original draw order; no reduction or scientific model changes here. Explicit
    CPU placement remains authoritative (including the accelerator lease path).
    """
    if isinstance(chunk_size_divisor, bool) or not isinstance(chunk_size_divisor, int) or chunk_size_divisor < 1:
        raise ValueError("chunk_size_divisor must be a positive integer")
    if n_items == 0:
        # No draw to probe: the pilot's whole-block rounding would size a block
        # that cannot be written into a zero-length result.
        return np.empty(0, dtype=float)
    if _allocated_devices is None:
        from trex.runtime.accelerator_pool import accelerator_execution_scope

        with accelerator_execution_scope(_requested_devices(chunk_plan)) as allocated:
            return _models()._run_jax_chunks(
                n_items, n_points, nsamples, evaluate, chunk_size_divisor, chunk_plan,
                _allocated_devices=allocated or (_models().jax.devices("cpu")[0],),
            )
    devices = _allocated_devices
    if chunk_plan and any(device not in chunk_plan for device in devices):
        # Admission can change between scenario retries (for example a lease
        # becomes contested). A new device needs its own measured plan.
        chunk_plan.clear()
    result = np.empty(n_items, dtype=float)
    first = 0
    if not chunk_plan and any(device.platform not in ("cpu", "interpreter") for device in devices):
        devices, first = _models()._device_pilot(evaluate, devices, n_items, n_points, nsamples, result)
        if first >= n_items:
            return result
    per_device_items = (n_items - first + len(devices) - 1) // len(devices)
    planner = (partial(_models()._plan_batch_chunk, evaluate) if isinstance(evaluate, _models().BatchEvaluator)
               else getattr(evaluate, "plan_chunk", None))
    chunks = _planned_chunks(evaluate, devices, n_items, n_points, nsamples, per_device_items,
                             planner, chunk_size_divisor, chunk_plan)
    _execute_chunks(evaluate, devices, chunks, first, n_items, result,
                    _ChunkReplanner(evaluate, planner, per_device_items, chunk_size_divisor, chunk_plan))
    return result


def _requested_devices(chunk_plan) -> tuple:
    configured = _models().jax.config.jax_default_device
    if chunk_plan:
        # A retry keeps its original devices even if default placement changes.
        return tuple(chunk_plan)
    if configured is not None and configured.platform in ("cpu", "interpreter"):
        return (configured,)
    return _models()._usable_jax_devices() or (_models().jax.devices("cpu")[0],)


def _auto_budget(chunk_size_divisor) -> bool:
    """Compiled-memory planning applies unless the user pinned a budget or a retry halved it."""
    override = read_environment(
        _models().JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, "EXONYM_JAX_BATCH_MEMORY_BYTES"
    )
    return not (override or "").strip() and chunk_size_divisor == 1


def _planned_chunks(evaluate, devices, n_items, n_points, nsamples, per_device_items,
                    planner, chunk_size_divisor, chunk_plan) -> list:
    """First chunk per device, in whole evaluation blocks."""
    chunks = [
        max(1, min(per_device_items, _models()._jax_chunk_size(n_points, nsamples, device)) // chunk_size_divisor)
        for device in devices
    ]
    # Pinned batches/CPU paths retain their conservative estimate. Accelerator
    # auto mode uses real compiled memory, not a fixed GiB working-set cap.
    if _auto_budget(chunk_size_divisor) and planner is not None and not chunk_plan:
        def plan(device, chunk):
            if device.platform in ("cpu", "interpreter"):
                return chunk
            return planner(chunk, per_device_items, _models()._jax_batch_memory_budget_bytes(device), device)

        if len(devices) == 1:
            chunks = [plan(devices[0], chunks[0])]
        else:
            from contextvars import copy_context

            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                futures = [pool.submit(copy_context().run, plan, device, chunk) for device, chunk in zip(devices, chunks)]
                chunks = [future.result() for future in futures]
    if chunk_plan is not None:
        if chunk_plan:
            # Current headroom can further shrink the retry, but newly freed
            # memory must never undo the requested halving of the first plan.
            chunks = [
                min(chunk, max(1, chunk_plan[device] // chunk_size_divisor))
                for device, chunk in zip(devices, chunks)
            ]
        else:
            chunk_plan.update(zip(devices, chunks))
    return [_models()._whole_blocks(chunk, evaluate, n_items, device) for device, chunk in zip(devices, chunks)]


class _ChunkReplanner:
    """Next chunk size for one device after it finishes a chunk."""

    def __init__(self, evaluate, planner, per_device_items, chunk_size_divisor, chunk_plan):
        self.evaluate, self.planner = evaluate, planner
        self.per_device_items, self.chunk_size_divisor, self.chunk_plan = (
            per_device_items, chunk_size_divisor, chunk_plan)

    def __call__(self, device, chunk, working_chunk, allocation_reduced, n_items) -> int:
        chunk = _models()._whole_blocks(min(chunk, working_chunk), self.evaluate, n_items, device)
        if self.chunk_plan is not None:
            self.chunk_plan[device] = min(self.chunk_plan[device], chunk)
        # Free memory can grow after another workload leaves. Auto
        # accelerator chunks replan from fresh counters; an OOM
        # retry ceiling and explicit user budgets never grow.
        if (self.planner is not None and not allocation_reduced
                and device.platform not in ("cpu", "interpreter") and _auto_budget(self.chunk_size_divisor)):
            chunk = _models()._whole_blocks(
                self.planner(chunk, self.per_device_items, _models()._jax_batch_memory_budget_bytes(device), device),
                self.evaluate, n_items, device,
            )
            if self.chunk_plan is not None:
                self.chunk_plan[device] = chunk
        return chunk


def _evaluate_span(evaluate, device, result, start, end, chunk, stop) -> Tuple[int, bool]:
    """Fill ``result[start:end]``, halving the slice only on allocation failure.

    Returns the last working slice size and whether it had to shrink.
    """
    cursor, working_chunk, reduced = start, chunk, False
    while cursor < end and not stop.is_set():
        stop_at = min(cursor + working_chunk, end)
        try:
            result[cursor:stop_at] = evaluate(slice(cursor, stop_at))
        except Exception as exc:
            if (working_chunk <= 1 or not is_memory_exhaustion(exc)
                    or _terminal_exhaustion(evaluate)):
                raise
            from trex.runtime.accelerator_pool import record_accelerator_event

            record_accelerator_event("allocation-memory-retry", device, batch=working_chunk,
                                     reason="allocation exhausted memory", error_type=type(exc).__name__)
            host_logger("trex").warning(
                "%s: allocation exhausted at batch=%d; retrying the same draws with a smaller working set (%s)",
                device, working_chunk, type(exc).__name__,
            )
        else:
            cursor = stop_at
            continue
        # Exit the exception scope to release failed buffers.
        working_chunk = max(1, working_chunk // 2)
        reduced = True
    return working_chunk, reduced


def _execute_chunks(evaluate, devices, chunks, first, n_items, result, replan) -> None:
    next_start = min(first + sum(chunks), n_items)
    queue_lock = Lock()
    stop = Event()

    def worker(device, offset, chunk):
        nonlocal next_start
        try:
            with _models().jax.default_device(device):
                start, assigned_chunk = offset, chunk
                # Sticky for this worker: once an allocation failed, its chunk never grows again.
                allocation_reduced = False
                while start < n_items and not stop.is_set():
                    end = min(start + assigned_chunk, n_items)
                    working_chunk, reduced = _evaluate_span(evaluate, device, result, start, end, chunk, stop)
                    allocation_reduced = allocation_reduced or reduced
                    chunk = replan(device, chunk, working_chunk, allocation_reduced, n_items)
                    with queue_lock:
                        start, assigned_chunk = next_start, chunk
                        next_start = min(start + chunk, n_items)
        except BaseException:
            # Other devices may finish their current kernel, but must not drain
            # all remaining draws before this batch can retry or be interrupted.
            stop.set()
            raise

    if len(devices) == 1:
        worker(devices[0], first, chunks[0])
        return
    from contextvars import copy_context

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = []
        offset = first
        try:
            for device, chunk in zip(devices, chunks):
                if offset < n_items:
                    futures.append(pool.submit(copy_context().run, worker, device, offset, chunk))
                offset += chunk
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            raise
