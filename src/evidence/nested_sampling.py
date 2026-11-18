"""Nested-sampling execution and stopping checks."""

from __future__ import annotations

import math
import multiprocessing
import os
import warnings
from typing import Optional

import numpy as np


from trex.runtime.logging import collect_worker_warnings, install_worker_warning_report, start_worker_warning_spool, worker_log_setup, worker_warning_report
from trex.evidence.nested_controls import NestedEvidenceResult, _HOSTED_N_DIMENSIONS, _SamplerControls


def _facade(name):
    from trex.evidence import nested_evidence

    return getattr(nested_evidence, name)

def _monitored_likelihood_count() -> int:
    counter = _facade("_monitored_likelihood_counter")
    return int(counter.value) if counter is not None else 0


class _LikelihoodBudgetExhausted(RuntimeError):
    """The monitored nested-sampling evaluation budget was exhausted."""


# Reserve one actual evaluation: larger private reservations can exhaust the
# shared budget while other workers still hold unused slots. A pool.map
# flush also cannot guarantee that every worker receives a flush task.
_MONITORED_COUNTER_BATCH: int = 1


class _MonitoredLikelihood:
    """Count every nested likelihood evaluation against the registered budget.

    ``dynesty`` stops on its own ``maxcall`` ceiling only between sampling
    iterations, and its call bookkeeping does not include every initialization
    evaluation, so that ceiling is not an execution-level guarantee.  This
    wrapper checks the shared counter before each evaluation: at most the
    registered number of evaluations is ever performed, across every worker,
    and the next request raises ``_LikelihoodBudgetExhausted`` so a truncated
    history can never be reported as a converged one.  The wrapper stores no
    multiprocessing object itself, so it pickles under the spawn start method;
    the counter reaches the workers through the pool initializer.

    The parallel reservation path currently reserves one evaluation at a time,
    preserving the full shared budget and exact recorded count. Larger private
    reservations are not adopted: one worker could otherwise exhaust admission
    before other workers executed their reserved calls. Serial checks are unchanged.
    """

    def __init__(self, log_likelihood, budget: int, *, batched: bool = False):
        self._log_likelihood = log_likelihood
        self._budget = int(budget)
        self._batched = bool(batched)
        self._allowed = 0

    def __call__(self, x):
        counter = _facade("_monitored_likelihood_counter")
        if counter is not None:
            if self._batched:
                self._reserve(counter)
            else:
                with counter.get_lock():
                    if counter.value >= self._budget:
                        raise _facade("_LikelihoodBudgetExhausted")(
                            "monitored nested likelihood budget of {0} evaluations "
                            "exhausted".format(self._budget)
                        )
                    counter.value += 1
        return self._log_likelihood(x)

    def _reserve(self, counter) -> None:
        if self._allowed > 0:
            self._allowed -= 1
            return
        with counter.get_lock():
            take = min(_MONITORED_COUNTER_BATCH, int(self._budget) - int(counter.value))
            if take <= 0:
                raise _facade("_LikelihoodBudgetExhausted")(
                    "monitored nested likelihood budget of {0} evaluations "
                    "exhausted".format(self._budget)
                )
            counter.value += take
            self._allowed = take - 1

    def _flush_monitored_local(self) -> None:
        counter = _facade("_monitored_likelihood_counter")
        if counter is None or not self._batched:
            return
        with counter.get_lock():
            counter.value -= self._allowed
            self._allowed = 0


def _flush_worker_monitored_counter(_index: int) -> None:
    """Withdraw one worker's unconsumed budget reservation after a run ends."""
    from dynesty.pool import FunctionCache

    wrapper = getattr(FunctionCache, "loglike", None)
    flush = getattr(wrapper, "_flush_monitored_local", None)
    if flush is not None:
        flush()


# Set by _worker_initializer: the picklable block callable shipped once per
# importance-refinement worker, instead of being re-pickled with every block
# task.  The module global is per-process, so each worker caches its own copy.
_importance_block_cache = None


def _cached_importance_block(points):
    block = _importance_block_cache
    if block is None:
        raise RuntimeError("importance worker initializer did not receive its block callable")
    return block(points)


def _worker_initializer(
    counter=None, functions=None, importance_block=None, log_setup=None, warning_report=None
) -> None:
    """Keep a nested-sampling worker off the accelerator and give it the counter.

    These dynesty likelihoods call scalar CPU Batman, not the JAX batch
    evaluator. Initializing an unused accelerator context per CPU child would
    consume VRAM without executing its likelihood there. This placement is a
    backend boundary, not a hardware-model/capacity restriction. Monte Carlo
    batches and NumPyro fits retain computation-scoped device admission.
    Existing user backend overrides are preserved.
    """
    global _importance_block_cache
    _importance_block_cache = importance_block
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if log_setup is not None:
        initialize, arguments = log_setup
        initialize(*arguments)
    # Report captured warnings when the pool closes; a killed worker leaves no
    # payload and is counted as lost by collect_worker_warnings.
    install_worker_warning_report(warning_report)
    _facade("_set_likelihood_counter")(counter)
    if functions is not None:
        from dynesty.pool import initializer

        initializer(*functions, (), {}, (), {})


def _worker_pool(
    n_jobs: int, counter=None, *, functions=None, importance_block=None, warning_report=None
):
    """A spawn-context pool for dynesty, or ``None`` when running serially."""
    if int(n_jobs) <= 1:
        return None
    context = multiprocessing.get_context(_facade("_MULTIPROCESSING_START_METHOD"))
    return context.Pool(
        int(n_jobs),
        initializer=_facade("_worker_initializer"),
        initargs=(counter, functions, importance_block, worker_log_setup(), warning_report),
    )


def _base_run_remaining_evidence(sampler) -> Optional[float]:
    """The base run's remaining-evidence ratio at its final stopping check.

    dynesty evaluates ``ln(1 + Z_est/Z) < dlogz`` at the top of every base
    iteration but stores the value only in the generator yield it consumes
    internally, so the number is recovered here from the base-run record: the
    last dead sample before the appended final live set carries the remaining
    volume and cumulative evidence, and the appended live set carries their
    likelihoods.  The expression is dynesty's own criterion (Speagle 2020,
    MNRAS 493, 3132, DOI ``10.1093/mnras/staa278``), evaluated on
    its saved state, in nats.

    The record must be ``sampler.base_run``, never ``saved_run``: every dynamic
    batch goes through ``combine_runs``, which rebuilds ``saved_run`` by
    interleaving base and batch samples in likelihood order and recomputing
    ``logvol``/``logz`` for the merged run.  Read there, the base samples are
    no longer contiguous and their volumes are the merged run's, so the value
    is not the one dynesty stopped on and can land on either side of the
    tolerance.  ``base_run`` is written only by the initial run.

    Returns ``None`` when the record does not contain an identifiable base
    run, which the caller treats as unresolved rather than as convergence.
    """
    base = getattr(sampler, "base_run", None)
    n_live = int(getattr(sampler, "nlive_init", 0))
    if base is None or n_live <= 0 or "logl" not in base.keys():
        return None
    logl = np.asarray(base["logl"], dtype=float)
    logvol = np.asarray(base["logvol"], dtype=float)
    logz = np.asarray(base["logz"], dtype=float)
    if logl.size <= n_live or logvol.size != logl.size or logz.size != logl.size:
        return None
    last_live = logl.size - 1
    last_dead = last_live - n_live
    if last_dead < 0 or not all(
        math.isfinite(float(value)) for value in (logvol[last_dead], logz[last_dead])
    ):
        return None
    live_logl = logl[last_dead + 1 : last_live + 1]
    if live_logl.size == 0 or not np.all(np.isfinite(live_logl)):
        return None
    return float(
        np.logaddexp(
            0.0,
            float(np.max(live_logl))
            + float(logvol[last_dead])
            - float(logz[last_dead]),
        )
    )


def _base_run_ended_on_plateau(sampler) -> bool:
    """Whether the base run's appended final live set shares one finite likelihood.

    dynesty ends the base run once every live point has the same likelihood
    and appends those points with the remaining volume, so the remaining
    evidence is accounted for exactly: Fowlie, Handley & Su (2021, MNRAS 503,
    1199, DOI 10.1093/mnras/stab590, sec. 5.1) state that such a
    run "converges if all live points share the same likelihood, and the
    remaining evidence is correctly accounted for". Like every nested stopping
    rule it cannot reveal a higher region the live points never found.
    """
    base = getattr(sampler, "base_run", None)
    n_live = int(getattr(sampler, "nlive_init", 0))
    if base is None or n_live <= 0 or "logl" not in base.keys():
        return False
    logl = np.asarray(base["logl"], dtype=float)
    if logl.size <= n_live:
        return False
    live_logl = logl[-n_live:]
    return bool(np.all(np.isfinite(live_logl)) and np.ptp(live_logl) == 0.0)


def _run_nested_sampler(
    prior_transform,
    log_likelihood,
    controls: _SamplerControls,
    scenario: str,
    n_dimensions: int = _HOSTED_N_DIMENSIONS,
) -> NestedEvidenceResult:
    """Run dynesty once and validate its returned evidence record.

    ``scenario`` only names the caller in error messages.  The returned
    number is taken as-is; the run's termination provenance is recorded so
    the caller can require that the base integration stopped on the
    registered evidence tolerance rather than on a budget. Evaluations are
    counted through a counter shared by every worker and a run that would
    exceed the registered ceiling raises instead of returning a partial
    history, so a number of unknown convergence can never be substituted.
    """
    try:
        import dynesty
    except ImportError as exc:
        raise RuntimeError(
            "dynesty is required for nested evidence estimation; reinstall "
            "TREX's pinned core dependencies"
        ) from exc

    runtime = _facade("_checkpoint_context").get()
    recovery = None
    if runtime is not None:
        prefix, deadline = runtime
        if prefix is None:
            raise RuntimeError("resumable escalation requires a candidate-owned checkpoint directory")
        identity = _facade("_checkpoint_identity")((
            prior_transform, log_likelihood, controls._replace(n_jobs=0), scenario, n_dimensions
        ))
        completed = _facade("_load_completed")(prefix.with_suffix(".npz"), identity)
        if completed is not None:
            return completed
        if _facade("_monotonic")() >= deadline:
            raise _facade("NestedCheckpointPaused")("nested wall deadline reached before sampling")
        recovery = _facade("_SamplerCheckpoint")(prefix.with_suffix(".dynesty"), identity, deadline)

    counter = None
    monitored_likelihood = log_likelihood
    if controls.max_likelihood_calls is not None:
        counter = multiprocessing.get_context(_facade("_MULTIPROCESSING_START_METHOD")).Value(
            "L", 0
        )
        # A pool exists exactly when n_jobs > 1, so the reservation batching
        # covers every parallel worker; the serial path keeps the per-call
        # exact check.  If the pool construction below fails, the run aborts
        # before any evaluation, so the two flags can never disagree.
        monitored_likelihood = _facade("_MonitoredLikelihood")(
            log_likelihood,
            controls.max_likelihood_calls,
            batched=int(controls.n_jobs) > 1,
        )
    run_kwargs = {
        "nlive_init": controls.n_live_points,
        "dlogz_init": controls.evidence_tolerance,
        "print_progress": False,
    }
    if controls.max_likelihood_calls is not None:
        run_kwargs["maxcall_init"] = controls.max_likelihood_calls
        run_kwargs["maxcall"] = controls.max_likelihood_calls
    functions = (monitored_likelihood, prior_transform)
    warning_spool = (
        start_worker_warning_spool() if int(controls.n_jobs) > 1 else None
    )
    try:
        pool = _facade("_worker_pool")(
            controls.n_jobs,
            counter,
            functions=functions,
            warning_report=worker_warning_report(warning_spool),
        )
    except BaseException:
        collect_worker_warnings(warning_spool, 0)  # Nothing ran; only remove the spool.
        raise
    pool_kwargs = (
        {} if pool is None else {"pool": pool, "queue_size": controls.n_jobs}
    )
    successful = False
    cache_initialized = False
    try:
        _facade("_set_likelihood_counter")(counter)
        if pool is not None or recovery is not None:
            from dynesty.pool import initializer, loglike_cache, prior_transform_cache

            # Ship fixed photometry/populations once per worker, not per proposal.
            # A restored parallel sampler still refers to these cache wrappers
            # when this host has only one worker and therefore no pool.
            initializer(*functions, (), {}, (), {})
            cache_initialized = True
            if pool is not None:
                monitored_likelihood, prior_transform = loglike_cache, prior_transform_cache
        sampler = recovery.restore(dynesty, pool, counter) if recovery is not None else None
        if sampler is None:
            sampler = dynesty.DynamicNestedSampler(
                monitored_likelihood,
                prior_transform,
                int(n_dimensions),
                sample=_facade("_SAMPLING_METHOD"),
                rstate=np.random.default_rng(controls.random_seed),
                **pool_kwargs,
            )
        else:
            run_kwargs["resume"] = True
        if recovery is not None:
            run_kwargs.update(print_progress=True, print_func=recovery.progress(
                sampler, clock=_facade("_monotonic"), count=_facade("_monitored_likelihood_count"),
            ))
        caught_warnings = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.filterwarnings(
                    "always",
                    message=r"The sampling was stopped short.*",
                    category=UserWarning,
                    module=r"dynesty\.sampler",
                )
                try:
                    sampler.run_nested(**run_kwargs)
                except _facade("_LikelihoodBudgetExhausted") as exc:
                    raise RuntimeError(
                        "the {0} nested run exhausted its registered {1}-evaluation "
                        "likelihood budget after {2} monitored evaluations; the "
                        "partial history is unresolved and must defer".format(
                            scenario,
                            controls.max_likelihood_calls,
                            _facade("_monitored_likelihood_count")(),
                        )
                    ) from exc
        finally:
            # A failed or paused sampler still owns the warnings emitted before it stopped.
            for item in caught_warnings:
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
        # Withdraw every worker's unused reservation so the recorded
        # monitored count is the exact number of executed evaluations.
        if counter is not None and pool is not None:
            pool.map(_flush_worker_monitored_counter, range(int(controls.n_jobs)), chunksize=1)
        monitored_calls = _facade("_monitored_likelihood_count")()
        results = sampler.results
        successful = True
    finally:
        try:
            if pool is not None:
                try:
                    if successful:
                        pool.close()
                    else:
                        pool.terminate()
                    pool.join()
                finally:
                    # Adopt the warnings workers reported on their way out, and name
                    # the workers a terminate or crash left unreported.
                    collect_worker_warnings(warning_spool, int(controls.n_jobs))
        finally:
            try:
                if cache_initialized:
                    from dynesty.pool import FunctionCache

                    del (FunctionCache.loglike, FunctionCache.prior_transform,
                         FunctionCache.logl_args, FunctionCache.logl_kwargs,
                         FunctionCache.ptform_args, FunctionCache.ptform_kwargs)
            finally:
                _facade("_set_likelihood_counter")(None)
    log_evidence = np.asarray(results.logz, dtype=float).reshape(-1)
    log_evidence_error = np.asarray(results.logzerr, dtype=float).reshape(-1)
    likelihood_calls = np.asarray(results.ncall, dtype=float).reshape(-1)
    iterations = np.asarray(results.niter, dtype=float).reshape(-1)
    if (
        log_evidence.size == 0
        or log_evidence_error.size == 0
        or likelihood_calls.size == 0
        or iterations.size == 0
        or not math.isfinite(float(log_evidence[-1]))
        or not math.isfinite(float(log_evidence_error[-1]))
        or not np.all(np.isfinite(likelihood_calls))
        or not math.isfinite(float(iterations[-1]))
    ):
        raise RuntimeError(
            "dynesty returned incomplete or non-finite nested-sampling "
            "results for the {0} scenario".format(scenario)
        )
    # dynesty records per-iteration call counts in its history, so the total
    # is their sum; its iteration record is cumulative. A run whose total
    # reaches the registered call cap was truncated by it, and its reported
    # remaining error is not a convergence statement.
    total_calls = int(np.sum(likelihood_calls))
    stopped_short = any(
        _facade("_DYNESTY_STOPPED_SHORT_WARNING") in str(item.message).lower()
        for item in caught_warnings
    ) or (
        controls.max_likelihood_calls is not None
        and total_calls >= controls.max_likelihood_calls
    )
    budget_exhausted = (
        controls.max_likelihood_calls is not None
        and monitored_calls >= int(controls.max_likelihood_calls)
    )
    base_run_delta_logz = _facade("_base_run_remaining_evidence")(sampler)
    tolerance_reached = (
        base_run_delta_logz is not None
        and base_run_delta_logz < controls.evidence_tolerance
    )
    if stopped_short or budget_exhausted:
        stop_reason = _facade("_STOP_REASON_TRUNCATED")
    elif tolerance_reached:
        stop_reason = _facade("_STOP_REASON_EVIDENCE_TOLERANCE")
    elif _facade("_base_run_ended_on_plateau")(sampler):
        stop_reason = _facade("_STOP_REASON_LIKELIHOOD_PLATEAU")
    else:
        stop_reason = _facade("_STOP_REASON_UNRESOLVED")
    log_weights = np.asarray(results.logwt, dtype=float)
    posterior_samples = np.asarray(results.samples_u, dtype=float)
    if (log_weights.ndim != 1 or not log_weights.size
            or posterior_samples.shape != (log_weights.size, n_dimensions)
            or not np.isfinite(posterior_samples).all()
            or np.any((posterior_samples < 0) | (posterior_samples > 1))
            or np.isnan(log_weights).any() or np.isposinf(log_weights).any()
            or not np.isfinite(log_weights).any()):
        raise RuntimeError("dynesty returned invalid posterior samples or weights for the {0} scenario".format(scenario))
    # The common offset cancels on normalization; the largest weight stays one, not underflowed zero.
    posterior_weights = np.exp(log_weights - np.max(log_weights))
    posterior_weights /= posterior_weights.sum()
    result = _facade("NestedEvidenceResult")(
        log_evidence=float(log_evidence[-1]),
        log_evidence_error=float(log_evidence_error[-1]),
        n_live_points=controls.n_live_points,
        n_likelihood_calls=total_calls,
        n_iterations=int(iterations[-1]),
        sampler=_facade("_SAMPLER_NAME"),
        sampler_version=str(dynesty.__version__),
        random_seed=controls.random_seed,
        stopped_short=stopped_short,
        monitored_likelihood_calls=int(monitored_calls),
        budget_exhausted=bool(budget_exhausted),
        stopped_on_evidence_tolerance=stop_reason == _facade("_STOP_REASON_EVIDENCE_TOLERANCE"),
        stop_reason=stop_reason,
        base_run_delta_logz=(
            float(base_run_delta_logz)
            if base_run_delta_logz is not None
            else float("nan")
        ),
        posterior_unit_samples=posterior_samples,
        posterior_weights=posterior_weights,
    )
    if recovery is not None:
        _facade("_save_completed")(prefix.with_suffix(".npz"), identity, result)
    return result
