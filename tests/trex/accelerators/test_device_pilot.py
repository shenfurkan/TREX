"""Measured per-kernel device choice for TREX batches (throughput + float64 parity).

Synthetic devices and a substituted timing probe exercise the selection logic
only; they prove dispatch decisions, not physical GPU performance. The one
physical-accelerator test skips unless JAX exposes a real GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest

models = pytest.importorskip("trex.accelerators._jax_models")
if not models.HAS_JAX_TREX:
    pytest.skip("JAX/jaxoplanet is not installed", allow_module_level=True)


class _Device(SimpleNamespace):
    """Synthetic accelerator identity: platform and name only."""

    def __str__(self):
        return self.name

    __hash__ = object.__hash__


def _probe(table):
    """Hardware timing boundary: (values, seconds) per device name."""

    def timed(evaluate, device, draws):
        values, seconds = table[str(device)]
        return np.asarray(values[:draws], dtype=float), seconds, draws

    return timed


@pytest.fixture(autouse=True)
def _fresh_decisions(monkeypatch):
    monkeypatch.setattr(models, "_DEVICE_DECISIONS", {})


class _Evaluator:
    """Attribute-carrying evaluator identity; its values never leave the synthetic table."""

    kernel = "prepared-kernel"

    def __call__(self, draws):
        return -np.arange(draws.start, draws.stop, dtype=float)


def _evaluate(draws):
    return -np.arange(draws.start, draws.stop, dtype=float)


def _choose(monkeypatch, table, devices, n_items=8):
    monkeypatch.setattr(models, "_timed_pilot", _probe(table))
    result = np.full(n_items, np.nan)
    chosen, first = models._device_pilot(_evaluate, devices, n_items, 10, 1, result)
    return chosen, first, result


def test_faster_accelerator_that_fails_float64_parity_is_not_eligible(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    table = {str(cpu): (reference, 1.0), "gpu:0": (reference + 1e-3, 0.01)}
    chosen, first, result = _choose(monkeypatch, table, (gpu,))
    assert chosen == (cpu,)
    assert first == 8
    np.testing.assert_array_equal(result, reference)


def test_slower_parity_passing_accelerator_leaves_the_cpu_chosen(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    table = {str(cpu): (reference, 1.0), "gpu:0": (reference, 2.0)}
    chosen, _, _ = _choose(monkeypatch, table, (gpu,))
    assert chosen == (cpu,)


def test_parity_passing_accelerators_are_used_together_when_their_sum_beats_the_cpu(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpus = (_Device(platform="gpu", name="gpu:0"), _Device(platform="gpu", name="gpu:1"))
    reference = -np.arange(8.0)
    # Each GPU alone is slower than the CPU; together they are faster.
    table = {str(cpu): (reference, 1.0), "gpu:0": (reference + 5e-7, 1.5), "gpu:1": (reference, 1.5)}
    chosen, first, result = _choose(monkeypatch, table, gpus)
    assert chosen == gpus
    # The prefix keeps the leading accelerator's own pilot values, not the CPU
    # reference: no block is evaluated twice and no CPU value stands in for an
    # accelerator draw, so a seed's bits do not depend on a cached decision.
    assert first == 8
    np.testing.assert_array_equal(result, reference + 5e-7)


def test_retained_prefix_is_limited_to_what_every_chosen_accelerator_reproduced(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpus = (_Device(platform="gpu", name="gpu:0"), _Device(platform="gpu", name="gpu:1"))
    reference = -np.arange(8.0)
    used = {"gpu:0": 8, "gpu:1": 4}

    def timed(evaluate, device, draws):
        count = used.get(str(device), draws)
        return reference[:count].copy(), 1.0 if device is cpu else 0.1, count

    monkeypatch.setattr(models, "_timed_pilot", timed)
    result = np.full(8, np.nan)
    chosen, first = models._device_pilot(_evaluate, gpus, 8, 10, 1, result)
    assert chosen == gpus
    # Only the prefix both accelerators reproduced is kept; the rest re-enters
    # the ordinary queue rather than claiming parity that was never measured.
    assert first == 4
    np.testing.assert_array_equal(result[:4], reference[:4])
    assert np.isnan(result[4:]).all()


def test_prepared_cpu_reference_is_reused_when_admission_changes_the_device_set(monkeypatch):
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    calls = []

    def timed(evaluate, device, draws):
        calls.append(str(device))
        return reference + 1e-9, 0.1, draws

    monkeypatch.setattr(models, "_timed_pilot", timed)
    evaluate = _Evaluator()
    # Prepared for two visible accelerators; dispatch admitted only one.
    evaluate._cpu_pilot = (reference, 2.0, 8, "previous-scenario")
    result = np.full(8, np.nan)
    chosen, first = models._device_pilot(evaluate, (gpu,), 8, 10, 1, result)
    assert chosen == (gpu,)
    assert calls == ["gpu:0"], "a narrowed admission must not repeat the CPU pilot"
    assert first == 8
    np.testing.assert_array_equal(result, reference + 1e-9)
    assert not hasattr(evaluate, "_cpu_pilot")


def test_failed_accelerator_pilot_reuses_the_prepared_cpu_reference(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    calls = []

    def timed(evaluate, device, draws):
        calls.append(str(device))
        raise RuntimeError("CUDA initialization failed")

    monkeypatch.setattr(models, "_timed_pilot", timed)
    evaluate = _Evaluator()
    evaluate._cpu_pilot = (reference, 2.0, 8, None)
    result = np.full(8, np.nan)
    chosen, first = models._device_pilot(evaluate, (gpu,), 8, 10, 1, result)
    assert chosen == (cpu,)
    assert calls == ["gpu:0"]
    assert first == 8
    np.testing.assert_array_equal(result, reference)
    assert not hasattr(evaluate, "_cpu_pilot")


@pytest.mark.parametrize("gpu_seconds", [0.1, 1.0])
def test_prepared_reference_can_dispatch_a_shorter_batch(monkeypatch, gpu_seconds):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    evaluate = _Evaluator()
    evaluate._cpu_pilot = (reference, 2.0, 8, None)

    def timed(_evaluate, device, draws):
        assert device is gpu and draws == 3
        return reference[:draws].copy(), gpu_seconds, draws

    monkeypatch.setattr(models, "_timed_pilot", timed)
    result = np.full(3, np.nan)
    chosen, first = models._device_pilot(evaluate, (gpu,), 3, 10, 1, result)
    # CPU measured 8/2 draws/s, not 3/2: capping the retained prefix cannot
    # retroactively lower its measured throughput.
    assert chosen == ((gpu,) if gpu_seconds == 0.1 else (cpu,))
    assert first == 3
    np.testing.assert_array_equal(result, reference[:3])


def test_empty_dispatch_never_evaluates_a_pilot_or_batch(monkeypatch):
    monkeypatch.setattr(models, "_device_pilot", lambda *_a: pytest.fail("empty pilot"))
    result = models._run_jax_chunks(
        0, 10, 1, lambda *_a: pytest.fail("empty batch"), 1)
    assert result.shape == (0,) and result.dtype == np.dtype("float64")


def test_cached_accelerator_decision_keeps_no_previously_piloted_values(monkeypatch):
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)
    calls = []

    def timed(evaluate, device, draws):
        calls.append(str(device))
        return reference, 0.1 if device is gpu else 1.0, draws

    monkeypatch.setattr(models, "_timed_pilot", timed)
    first_result = np.full(8, np.nan)
    chosen, first = models._device_pilot(_evaluate, (gpu,), 8, 10, 1, first_result)
    assert chosen == (gpu,) and first == 8
    second_result = np.full(8, np.nan)
    chosen, first = models._device_pilot(_evaluate, (gpu,), 8, 10, 1, second_result)
    assert chosen == (gpu,) and first == 0
    assert np.isnan(second_result).all(), "pilot values are per-run, never a value cache"
    assert len(calls) == 2


def test_retained_pilot_prefix_leaves_the_remainder_evaluated_once_in_order(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    pilot_draws, n_items = 8, 14
    evaluated = []

    def evaluate(draws):
        evaluated.append((draws.start, draws.stop))
        return -np.arange(draws.start, draws.stop, dtype=float)

    def timed(evaluate_, device, draws):
        return evaluate_(slice(0, draws)), 1.0 if device is cpu else 0.1, draws

    monkeypatch.setattr(models, "DEVICE_PILOT_DRAWS", pilot_draws)
    monkeypatch.setattr(models, "_jax_chunk_size", lambda *args: pilot_draws)
    monkeypatch.setattr(models, "_timed_pilot", timed)
    result = models._run_jax_chunks(n_items, 10, 1, evaluate, 1, _allocated_devices=(gpu,))
    np.testing.assert_array_equal(result, -np.arange(n_items, dtype=float))
    assert evaluated[0] == (0, pilot_draws)           # CPU reference
    assert evaluated[1] == (0, pilot_draws)           # accelerator pilot, kept as the prefix
    assert evaluated[2:] == [(pilot_draws, n_items)]  # remainder, once, in order


def test_nonfinite_pattern_mismatch_fails_parity_even_with_tiny_finite_error():
    reference = np.array([0.0, -np.inf, -1.0])
    assert not models._pilot_parity(reference, np.array([0.0, np.nan, -1.0]))["passed"]
    assert models._pilot_parity(reference, reference.copy())["passed"]


def test_draws_that_underflow_relative_to_the_maximum_do_not_decide_parity():
    reference = np.array([0.0, -1.0, -1.0e4])
    values = reference + np.array([0.0, 0.0, 1.0])  # differs only where exp() underflows
    assert models._pilot_parity(reference, values)["passed"]


def test_pilot_failure_is_recorded_and_makes_the_device_ineligible(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    reference = -np.arange(8.0)

    def timed(evaluate, device, draws):
        if device is gpu:
            raise RuntimeError("CUDA initialization failed")
        return reference[:draws], 1.0, draws

    monkeypatch.setattr(models, "_timed_pilot", timed)
    chosen, _ = models._device_pilot(_evaluate, (gpu,), 8, 10, 1, np.empty(8))
    assert chosen == (cpu,)


def test_cached_decision_runs_no_second_pilot(monkeypatch):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    calls = []
    reference = -np.arange(8.0)

    def timed(evaluate, device, draws):
        calls.append(str(device))
        return reference[:draws], 1.0 if device is cpu else 2.0, draws

    monkeypatch.setattr(models, "_timed_pilot", timed)
    models._device_pilot(_evaluate, (gpu,), 8, 10, 1, np.empty(8))
    chosen, first = models._device_pilot(_evaluate, (gpu,), 8, 10, 1, np.empty(8))
    assert chosen == (cpu,) and first == 0
    assert len(calls) == 2


@pytest.mark.parametrize("workers", (1, 2))
def test_pilot_draws_are_production_draws_and_the_queue_resumes_in_order(monkeypatch, workers):
    cpu = models.jax.devices("cpu")[0]
    gpu = _Device(platform="gpu", name="gpu:0")
    n_items = 3 * models.DEVICE_PILOT_DRAWS // 2
    evaluated = []

    def evaluate(draws):
        evaluated.append((draws.start, draws.stop))
        return -np.arange(draws.start, draws.stop, dtype=float)

    def pilot(evaluate_, devices, n, n_points, nsamples, result):
        result[:models.DEVICE_PILOT_DRAWS] = evaluate_(slice(0, models.DEVICE_PILOT_DRAWS))
        return (cpu,) * workers, models.DEVICE_PILOT_DRAWS

    monkeypatch.setattr(models, "_device_pilot", pilot)
    result = models._run_jax_chunks(n_items, 10, 1, evaluate, 1, _allocated_devices=(gpu,))
    np.testing.assert_array_equal(result, -np.arange(n_items, dtype=float))
    covered = sorted(evaluated)
    assert covered[0] == (0, models.DEVICE_PILOT_DRAWS)
    assert sum(stop - start for start, stop in covered) == n_items  # each draw exactly once


def test_physical_accelerator_pilot_matches_the_cpu_reference():
    """Real EB kernel on every visible device; skipped without a physical GPU.

    Whatever the pilot chooses, the production result must reproduce explicit
    CPU placement within the parity tolerance. Throughput is recorded by the
    pilot, not asserted: it is a measurement, not a contract.
    """
    from tests.fixtures.synthetic_observations import synthetic_batch_likelihood_case

    jax = models.jax
    if not any(device.platform not in ("cpu", "interpreter") for device in jax.devices()):
        pytest.skip("no physical accelerator visible to JAX")
    time, flux, params = synthetic_batch_likelihood_case()
    count = models.DEVICE_PILOT_DRAWS + 7
    common = dict(
        time=time, flux_obs=flux, sigma=params["exptime_days"], p_orb=params["P_orb"],
        inc_arr=np.full(count, params["inc_deg"]), a_cm_arr=params["a_cm"], R_host=params["R_s_solar"],
        u1=params["u1"], u2=params["u2"], exptime_days=params["exptime_days"],
        ecc_arr=np.zeros(count), w_arr=np.full(count, 90.0), nsamples=params["nsamples"],
        reb_arr=params["R_EB_solar"] * np.linspace(1.0, 2.0, count),
        feb_arr=np.full(count, params["EB_fluxratio"]), u1_EB=params["u1_EB"], u2_EB=params["u2_EB"],
    )
    with jax.default_device(jax.devices("cpu")[0]):
        reference = models.batch_lnL_EB(**common)
    production = models.batch_lnL_EB(**common)
    assert models._DEVICE_DECISIONS, "auto placement with an accelerator must run the measured pilot"
    parity = models._pilot_parity(np.asarray(reference), np.asarray(production))
    assert parity["passed"], parity
