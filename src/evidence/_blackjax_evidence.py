"""BlackJAX nested-sampling evidence and compression uncertainty."""

from dataclasses import replace
import math
from time import monotonic

import numpy as np

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_checkpoint import load_native, save_native
from trex.evidence._blackjax_types import (
    BLACKJAX_VERSION,
    SLICE_CALL_BOUND,
    NativeBatchHistory,
    NativeEvidenceResult,
    NativeIntegrationError,
    NativeSliceAttempt,
    NativeStoppingCheck,
)
from trex.evidence.nested_checkpoint import NestedCheckpointPaused

# Algorithmic Monte Carlo budget for conditional compression uncertainty only.
# Fowlie et al. (2021), section 4.2; not a TREX accuracy/coverage calibration.
_COMPRESSION_REPLICATES = 256

# Diagonal ridge for an opt-in direction metric's host Cholesky factor.
_PILOT_RIDGE_SCALE = 1e-10

# Persisted in ``calls`` as physically dispatched initialization lanes that were
# not logically committed (block padding and lanes dropped after an invalid one).
_INITIALIZATION_PADDING = "initialization_padding"


def _compression_log_evidences(key, dead_logl, live_counts, final_logl):
    """Fowlie (2021) section 4.2: joint-order-statistic compression simulations.

    Stream 256 paths without retaining a dead-points x paths working set.
    Same Riemann/equal-live rule as the point estimate; no final live deaths.
    ``-Exp(1)/n`` is ``log(Beta(n,1))`` under the declared IID idealization.
    Likelihood inputs and outputs are RELATIVE to one fixed log-likelihood
    offset. Do not add the offset before estimating dispersion or posterior.
    """
    jax, jnp = models.jax, models.jnp

    def body(carry, event):
        logx, logz = carry
        index, logl, count = event
        event_key = jax.random.fold_in(jax.random.fold_in(key, jnp.uint32(index >> 32)),
                                      jnp.uint32(index & np.iinfo(np.uint32).max))
        change = -jax.random.exponential(event_key,
                                         (_COMPRESSION_REPLICATES,), dtype=jnp.float64) / count
        logmass = logx + jnp.log(-jnp.expm1(change))
        return (logx + change, jnp.logaddexp(logz, logmass + logl)), None

    initial = (jnp.zeros(_COMPRESSION_REPLICATES, dtype=jnp.float64),
               jnp.full(_COMPRESSION_REPLICATES, -jnp.inf, dtype=jnp.float64))
    (logx, logz), _ = jax.lax.scan(body, initial,
        (jnp.arange(dead_logl.size, dtype=jnp.int64), dead_logl, live_counts))
    log_mean_live = jax.scipy.special.logsumexp(final_logl) - jnp.log(final_logl.size)
    return jnp.logaddexp(logz, logx + log_mean_live)


def _relative_log_likelihood(values, offset):
    """Separate a common log scale without changing support or clipping values.

    Reject finite inputs whose subtraction overflows: that is an unresolved
    arithmetic/domain failure, not an observational support zero. Absolute logZ
    fields remain rounded float64 views; relative fields are the precise audit.
    """
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            relative = values - offset
        except FloatingPointError as exc:
            raise ArithmeticError("native log-likelihood dynamic range exceeds offset arithmetic") from exc
    if np.any(np.isfinite(values) & ~np.isfinite(relative)):
        raise ArithmeticError("native log-likelihood dynamic range exceeds offset arithmetic")
    return relative


def _ridge_cholesky_factor(positions):
    """Host float64 Cholesky factor of the ridge-stabilized live covariance.

    Mirrors the kernel's ``prepare`` centering (displacements about the first
    point, divided by ``n - 1``) so the factor describes the same ensemble. The
    ridge ``1e-10 * max(trace / dimension, 1.0)`` on the diagonal keeps a
    collapsed ensemble factorable, where ``prepare`` deliberately fails instead.
    A covariance needs at least two points; anything else raises through the
    caller's failure history rather than returning an undeclared metric.
    """
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("a direction metric requires at least two ensemble points")
    displacements = points - points[0]
    centered = displacements - displacements.mean(axis=0)
    covariance = centered.T @ centered / (points.shape[0] - 1)
    dimension = points.shape[1]
    ridge = _PILOT_RIDGE_SCALE * max(float(np.trace(covariance)) / dimension, 1.0)
    factor = np.linalg.cholesky(covariance + ridge * np.eye(dimension))
    if not np.isfinite(factor).all() or not (np.diag(factor) > 0.0).all():
        raise ValueError("invalid numerical direction covariance factor")
    return factor


def _split_ensemble_chain_factors(positions, loglikelihood, threshold, parent_indices):
    """One fixed factor per chain, taken from the opposite strict-survivor half.

    Strict survivors ``loglikelihood > threshold`` in live-state order split into
    a deterministic first half and second half. A chain whose parent lies in the
    first half uses the factor of the second half and vice versa, so no chain's
    metric is built from its own parent's points. A half holding fewer than two
    points cannot form a covariance; then one factor from all survivors serves
    both halves. A single strict survivor has no ensemble spread at all, so that
    batch falls back to the isotropic identity (documented degenerate-ensemble
    fallbacks) instead of failing the last deletion.
    """
    points = np.asarray(positions, dtype=np.float64)
    parents = np.asarray(parent_indices, dtype=np.int64)
    survivors = np.flatnonzero(np.asarray(loglikelihood, dtype=np.float64) > threshold)
    if survivors.size == 0:
        raise ValueError("a direction metric requires a strict survivor")
    if survivors.size == 1:
        # The end-game batches leave one strict survivor, which has no ensemble
        # spread to shape a direction; the isotropic metric keeps the run finite
        # instead of failing the last deletion on a degenerate covariance.
        return np.repeat(np.eye(points.shape[1], dtype=np.float64)[None], parents.size, axis=0)
    first, second = np.split(survivors, [survivors.size // 2])
    if first.size < 2 or second.size < 2:
        shared = _ridge_cholesky_factor(points[survivors])
        return np.repeat(shared[None], parents.size, axis=0)
    from_second = _ridge_cholesky_factor(points[second])
    from_first = _ridge_cholesky_factor(points[first])
    return np.where(np.isin(parents, first)[:, None, None], from_second, from_first)


def _integrate_positions(self, values, key, tolerance, ceiling, seed, *, checkpoint=None, identity=None, deadline=math.inf):
    """Driver behind ``NativeNestedKernel.integrate`` (``self``), after its input validation."""
    from blackjax.ns import base

    jax, jnp = models.jax, models.jnp
    points = jax.device_put(jnp.asarray(values, dtype=jnp.float64), self.integrand.device)
    initial_key = jax.device_put(key, self.integrand.device)
    # Validate the RNG interface without drawing/reseeding the caller's stream.
    key_data = jax.random.key_data(initial_key)
    normalized_key = jax.random.wrap_key_data(key_data, impl=jax.random.key_impl(initial_key))
    if normalized_key.ndim != 0:
        raise ValueError("experimental integration requires a single JAX key")
    continuation_key = initial_key
    active_compression_key = None
    initial_particles = []
    initial_state = state = None
    batches = []
    checks = []
    calls = {"initialization": 0, "left": 0, "right": 0, "shrinkage": 0}
    total_calls = 0
    logx, logz_dead = 0.0, -math.inf
    failure = None
    stop_reason = "unresolved"
    likelihood_offset = 0.0
    failed_operation = None
    failed_attempt = None
    calls_exact = True
    padded = False  # Masked batch elements make physical call counts unknown.

    restored = None if checkpoint is None else load_native(checkpoint, identity, device=self.integrand.device)
    if restored is not None:
        if restored.get("failed") is not None:
            result = restored["failed"]
            raise NativeIntegrationError(
                "retained BlackJAX integration failed; inspect its history: " + str(result.failure_reason), result)
        (continuation_key, initial_particles, initial_state, state, batches, checks, calls,
         total_calls, logx, logz_dead, likelihood_offset, padded, calls_exact) = restored["state"]
        # A periodic snapshot recovered after an unclean stop cannot know how
        # much uncommitted physical work ran before the process was killed.
        calls_exact = calls_exact and restored["clean_pause"]
    last_saved = monotonic()

    def retain(*, failed=None):
        nonlocal last_saved
        now = monotonic()
        if checkpoint is not None and (failed is not None or now >= deadline or now - last_saved >= 60):
            save_native(checkpoint, identity, {"failed": failed, "clean_pause": now >= deadline, "state": (
                continuation_key, initial_particles, initial_state, state, batches, checks, calls,
                total_calls, logx, logz_dead, likelihood_offset, padded, calls_exact)})
            last_saved = now
        if failed is None and now >= deadline:
            raise NestedCheckpointPaused("nested wall deadline reached; sampler checkpoint retained")

    def stopping_check(reason):
        if state is None:
            maximum, maximum_relative, delta = math.nan, math.nan, math.nan
        else:
            maximum = float(np.max(np.asarray(state.particles.loglikelihood)))
            maximum_relative = float(_relative_log_likelihood(maximum, likelihood_offset))
            delta = (float(np.logaddexp(0.0, logx + maximum_relative - logz_dead))
                     if math.isfinite(maximum) else math.nan)
        absolute_dead = likelihood_offset + logz_dead
        record = NativeStoppingCheck(absolute_dead, logx, maximum, delta,
                                     tolerance, total_calls, reason, likelihood_offset,
                                     logz_dead, maximum_relative)
        checks.append(record)
        return record

    def finish(*, finalize=True):
        nonlocal continuation_key, active_compression_key, failure, stop_reason
        completed = [batch for batch in batches if batch.completed]
        empty = np.empty(0, dtype=np.float64)
        counts = (np.concatenate([batch.effective_live_counts for batch in completed])
                  if completed else np.empty(0, dtype=np.int64))
        dead_logl = (np.concatenate([np.asarray(batch.dead_particles.loglikelihood) for batch in completed])
                     if completed else empty.copy())
        # Fowlie Eq. 14: exact geometric expectation, NOT the -q/N approximation.
        logvol = -np.cumsum(1.0 / counts)
        previous = np.concatenate(([0.0], logvol[:-1])) if counts.size else empty.copy()
        # Eq. 6 one-sided Riemann rule. Stable difference without clipping.
        dead_logmass = previous + np.log(-np.expm1(-1.0 / counts))
        dead_logweights = dead_logmass + dead_logl
        dead_relative = empty.copy()
        live_logmass = live_logweights = empty.copy()
        live_relative = empty.copy()
        posterior_samples = posterior_weights = None
        simulated_logz = empty.copy()
        simulated_relative = empty.copy()
        uncertainty = math.nan
        logz = logz_relative = math.nan
        compression_key = active_compression_key
        if state is not None and np.isneginf(np.asarray(state.particles.loglikelihood)).all():
            logz = logz_relative = -math.inf
        if state is not None and finalize and stop_reason != "unresolved":
            last_logx = float(logvol[-1]) if counts.size else 0.0
            # Complete residual volume ONCE. Final live points are not another
            # shrinking rank batch; no extra interval/boundary trapezoid.
            live_logmass = np.full(self.shape[0], last_logx - math.log(self.shape[0]))
            live_logl = np.asarray(state.particles.loglikelihood)
            live_logweights = live_logmass + live_logl
            dead_relative = dead_logmass + _relative_log_likelihood(dead_logl, likelihood_offset)
            live_relative = live_logmass + _relative_log_likelihood(live_logl, likelihood_offset)
            combined_relative = np.concatenate((dead_relative, live_relative))
            logz_relative = float(np.logaddexp.reduce(combined_relative))
            logz = likelihood_offset + logz_relative
            if math.isfinite(logz_relative) and math.isfinite(logz):
                continuation_key, compression_key = jax.random.split(continuation_key)
                active_compression_key = compression_key
                simulated_relative = np.asarray(self.compression_log_evidences(
                    compression_key, jnp.asarray(_relative_log_likelihood(dead_logl, likelihood_offset)),
                    jnp.asarray(counts), jnp.asarray(_relative_log_likelihood(live_logl, likelihood_offset))))
                simulated_logz = simulated_relative + likelihood_offset
                # Dispersion and posterior are computed BEFORE restoring the
                # common offset. Absolute float64 views can lose small terms.
                uncertainty = float(np.std(simulated_relative - simulated_relative[0], ddof=1)) if counts.size else 0.0
                if not np.isfinite(simulated_relative).all() or not math.isfinite(uncertainty):
                    failure, stop_reason = "invalid numerical compression uncertainty", "unresolved"
                if stop_reason != "unresolved":
                    dead_positions = [np.asarray(batch.dead_particles.position) for batch in completed]
                    posterior_samples = np.concatenate(dead_positions + [np.asarray(state.particles.position)])
                    posterior_weights = np.exp(combined_relative - logz_relative)
                    posterior_weights /= posterior_weights.sum()  # floating-point normalization only
                    if (not np.isfinite(posterior_weights).all() or (posterior_weights < 0.0).any()
                            or not np.isfinite(posterior_samples).all()
                            or not ((posterior_samples >= 0.0) & (posterior_samples < 1.0)).all()):
                        failure, stop_reason = "invalid numerical experimental weighted posterior", "unresolved"
                        posterior_samples = posterior_weights = None
            else:
                failure = ("zero evidence from all-zero live support remains unresolved"
                           if np.isneginf(logz_relative) else "invalid numerical evidence arithmetic")
                stop_reason = "unresolved"
        if checks:
            checks[-1] = replace(checks[-1], reason=stop_reason)
        return NativeEvidenceResult(
            log_evidence=logz, log_evidence_error=uncertainty, log_evidence_relative=logz_relative,
            log_evidence_offset=likelihood_offset, log_likelihood_offset=likelihood_offset,
            n_live_points=self.shape[0],
            n_likelihood_calls=total_calls, n_iterations=len(completed),
            sampler="blackjax-nss-experimental", sampler_version=BLACKJAX_VERSION, random_seed=seed,
            stopped_short=stop_reason in {"truncated", "unresolved"},
            monitored_likelihood_calls=total_calls,
            budget_exhausted=total_calls >= ceiling or stop_reason == "truncated",
            stopped_on_evidence_tolerance=stop_reason == "evidence-tolerance", stop_reason=stop_reason,
            base_run_delta_logz=checks[-1].delta_logz if checks else math.nan,
            posterior_unit_samples=posterior_samples, posterior_weights=posterior_weights,
            initial_key=initial_key, continuation_key=continuation_key, compression_key=compression_key,
            initial_positions=values, initial_particles=tuple(initial_particles),
            initial_state=initial_state, final_state=state, batches=tuple(batches), stopping_checks=tuple(checks),
            effective_live_counts=counts, dead_log_volumes=logvol,
            dead_log_volume_weights=dead_logmass, final_live_log_volume_weights=live_logmass,
            dead_log_evidence_weights=dead_logweights, final_live_log_evidence_weights=live_logweights,
            dead_log_evidence_weights_relative=dead_relative, final_live_log_evidence_weights_relative=live_relative,
            compression_log_evidences=simulated_logz, compression_log_evidences_relative=simulated_relative,
            physical_likelihood_calls=(total_calls + calls.get(_INITIALIZATION_PADDING, 0)
                                       if calls_exact and not padded else None),
            likelihood_calls_exact=calls_exact,
            slice_execution="vmapped-chains" if padded else "scalar-device-loop",
            likelihood_call_breakdown={key: value for key, value in calls.items()
                                       if key != _INITIALIZATION_PADDING},
            evidence_tolerance=tolerance,
            max_likelihood_calls=ceiling, num_delete=self.num_delete, num_inner_steps=self.num_inner_steps,
            failure_reason=failure, failed_operation=failed_operation, failed_attempt=failed_attempt,
            direction_metric=self.direction_metric,
            batched_initialization=self.batched_initialization,
            speculative_expansions=self.speculative_expansions)

    def fail_with_history(exc, operation, *, unknown_calls=False, attempt=None):
        nonlocal failure, stop_reason, failed_operation, failed_attempt, calls_exact
        failure = f"{operation}: {type(exc).__name__}: {exc}"
        failed_operation, stop_reason = operation, "unresolved"
        failed_attempt = attempt
        calls_exact = calls_exact and not unknown_calls
        # Do not run compression or retry failed scientific work to build a
        # failure record. If a device kernel failed without returning its
        # functional counter, only the completed-call lower bound is known.
        if checks:
            checks[-1] = replace(checks[-1], reason=stop_reason, n_likelihood_calls=total_calls)
        result = finish(finalize=False)
        retain(failed=result)
        return NativeIntegrationError(failure, result)

    def checked_call(function, *args, operation, native=False, attempt=None, **kwargs):
        try:
            result = function(*args, **kwargs)
            # Surface asynchronous device failures inside the provenance
            # boundary, before treating the returned counter as exact.
            return jax.block_until_ready(result)
        except Exception as exc:
            raise fail_with_history(exc, operation, unknown_calls=native, attempt=attempt) from exc

    def final_result():
        try:
            return finish()
        except Exception as exc:
            raise fail_with_history(exc, "finalization", attempt={"compression_key": active_compression_key}) from exc

    # Opt-in fixed metric: one factor from the caller's initial prior positions,
    # unchanged for every slice of the whole run.
    pilot_factor = (checked_call(_ridge_cholesky_factor, values, operation="direction metric")
                    if self.direction_metric == "fixed-pilot" else None)
    # Sequential initialization has no unchecked/padded evaluations after an
    # invalid point or ceiling, and preserves an incomplete prior population.
    if self.batched_initialization:
        # Block-granular boundaries: one evaluation call per block, and the
        # logical state advances only through lanes validated in original order.
        while state is None:
            retain()
            remaining_points = points.shape[0] - len(initial_particles)
            if remaining_points <= 0:
                break
            count = min(self.initialization_block, remaining_points, ceiling - total_calls)
            if count <= 0:
                failure, stop_reason = "native likelihood budget exhausted during initialization", "truncated"
                stopping_check(stop_reason)
                return final_result()
            first, block_calls = len(initial_particles), total_calls
            selected = points[first:first + count]
            # Padding repeats lanes already dispatched in this block, never a
            # new out-of-batch corner, so it cannot introduce new support.
            lanes = selected[np.arange(self.initialization_block) % count]
            particles, valid = checked_call(self.batched_initialization_evaluate, lanes,
                jnp.asarray(-jnp.inf), operation="initialization", native=True,
                attempt={"position": selected[0], "remaining_budget": ceiling - total_calls})
            committed = 0
            for lane in range(count):
                initial_particles.append(jax.tree.map(lambda leaf, idx=lane: leaf[idx], particles))
                total_calls += 1
                calls["initialization"] += 1
                committed = lane + 1
                if not bool(valid[lane]):
                    discarded = self.initialization_block - committed
                    calls[_INITIALIZATION_PADDING] = calls.get(_INITIALIZATION_PADDING, 0) + discarded
                    failure = "invalid numerical native integrand evaluation during initialization"
                    stopping_check("unresolved")
                    failed_operation = "initialization"
                    failed_attempt = {"position": points[first + lane],
                                      "remaining_budget": ceiling - (block_calls + lane)}
                    result = finish(finalize=False)
                    retain(failed=result)
                    raise NativeIntegrationError(failure, result)
            if committed < self.initialization_block:
                calls[_INITIALIZATION_PADDING] = (
                    calls.get(_INITIALIZATION_PADDING, 0) + self.initialization_block - committed)
    else:
        for point in points[len(initial_particles):] if state is None else ():
            retain()
            if total_calls >= ceiling:
                failure, stop_reason = "native likelihood budget exhausted during initialization", "truncated"
                stopping_check(stop_reason)
                return final_result()
            particle, valid = checked_call(self.integration_evaluate, point, jnp.asarray(-jnp.inf),
                operation="initialization", native=True,
                attempt={"position": point, "remaining_budget": ceiling - total_calls})
            initial_particles.append(particle)
            total_calls += 1
            calls["initialization"] += 1
            if not bool(valid):
                failure = "invalid numerical native integrand evaluation during initialization"
                stopping_check("unresolved")
                failed_operation = "initialization"
                result = finish(finalize=False)
                retain(failed=result)
                raise NativeIntegrationError(failure, result)
    if state is None:
        initial_state = state = checked_call(
            lambda: base.NSState(jax.tree.map(lambda *leaves: jnp.stack(leaves), *initial_particles)),
            operation="initial live assembly")
        finite_initial = np.asarray(state.particles.loglikelihood)
        finite_initial = finite_initial[np.isfinite(finite_initial)]
        if finite_initial.size:
            likelihood_offset = float(np.max(finite_initial))

    while True:
        retain()
        record = checked_call(stopping_check, "continue", operation="stopping check")
        live_logl = np.asarray(state.particles.loglikelihood)
        if total_calls >= ceiling:
            failure, stop_reason = "native likelihood budget exhausted", "truncated"
            break
        if np.isneginf(live_logl).all():
            failure = "all-zero live support: zero evidence is unresolved, not a support-absence certificate"
            break
        # Plateau handling precedes the strict-survivor kernel, but does not
        # hide invalid numerics or a singular incoming covariance.
        uniform = bool(np.all(live_logl == live_logl[0]))
        order = np.argsort(live_logl, kind="stable")
        ordered = live_logl[order]
        q = self.num_delete
        # End at the first tied contour encountered in the proposed batch,
        # and evict the COMPLETE contour before any births (Fowlie Algorithm 2).
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], ordered.size]
        for start, end in zip(starts, ends):
            if start < q and end - start > 1:
                q = int(end if end < self.shape[0] else start)
                break
        threshold = float(ordered[q - 1]) if q else float(ordered[0])
        # A finite full-live plateau has no strict survivor by definition.
        # Check all other prepare guards using a strictly lower *guard-only*
        # threshold; no likelihood or contour is altered for proposals.
        guard_threshold = -math.inf if uniform else threshold
        error, _ = checked_call(self.integration_prepare, state, jnp.asarray(guard_threshold),
                                operation="live preparation")
        try:
            error.throw()
        except ValueError as exc:
            failure = str(exc)
            failed_operation = "live preparation"
            break
        if uniform:
            stop_reason = "likelihood-plateau"
            break
        if record.delta_logz < tolerance:
            stop_reason = "evidence-tolerance"
            break

        continuation_key, batch_key = checked_call(jax.random.split, continuation_key, operation="batch RNG split")
        _, inner_key = checked_call(jax.random.split, batch_key, operation="batch RNG split")
        choice_key, sample_key = checked_call(jax.random.split, inner_key, operation="parent RNG split")
        # All strict survivors have equal weight. Compact their support
        # before sampling: a floating weighted CDF can select a zero-weight
        # plateau when a finite-precision uniform lands on its boundary.
        survivors = jnp.asarray(np.flatnonzero(live_logl > threshold))
        parent_indices = np.asarray(checked_call(jax.random.choice, choice_key, survivors, (q,),
            replace=True, operation="parent selection"))
        if not (live_logl[parent_indices] > threshold).all():
            raise fail_with_history(ValueError("selected parent is not a strict contour survivor"),
                "parent selection", attempt={"parent_indices": parent_indices,
                    "choice_key": choice_key, "batch_key": batch_key, "threshold": threshold})
        chain_keys = checked_call(jax.random.split, sample_key, q, operation="chain RNG split")
        # Opt-in split metric: one host factor per chain for this batch, built
        # from the opposite strict-survivor half and fixed for all inner steps.
        chain_factors = (np.broadcast_to(pilot_factor, (q, *pilot_factor.shape))
                         if pilot_factor is not None else None)
        if self.direction_metric == "split-ensemble":
            chain_factors = checked_call(_split_ensemble_chain_factors,
                np.asarray(state.particles.position), live_logl, threshold, parent_indices,
                operation="direction metric")
        indices = order[:q].copy()
        dead_particles = jax.tree.map(lambda leaf, idx=indices: leaf[idx], state.particles)
        replacements, slices = [], []
        before = total_calls
        completed = True
        remaining = ceiling - total_calls
        # Batched only when the batch's worst case fits the hard ceiling; otherwise
        # the scalar loop below keeps exact interruption at the last evaluation.
        batched = self.batched and remaining >= q * self.num_inner_steps * SLICE_CALL_BOUND
        batched_record = replacement_stack = None
        if batched:
            padded = True
            parent_array = jnp.asarray(parent_indices)
            starts = jax.tree.map(lambda leaf, idx=parent_array: leaf[idx], state.particles)
            arguments = (chain_keys, starts, jnp.asarray(threshold))
            if chain_factors is not None:
                arguments += (chain_factors,)
            finals, infos = checked_call(self.batched_chains, *arguments,
                operation="batched constrained slices", native=True,
                attempt={"chain_keys": chain_keys, "parent_indices": parent_indices, "threshold": threshold,
                         "remaining_budget": remaining, "batch_key": batch_key, "dead_indices": indices})
            lane_counts = np.asarray(infos.likelihood_calls, dtype=np.int64)
            used = lane_counts.sum(axis=-1)
            if (lane_counts.shape != (q, self.num_inner_steps, 3) or (lane_counts < 0).any()
                    or (used > SLICE_CALL_BOUND).any()
                    or (np.asarray(infos.remaining_budget) != SLICE_CALL_BOUND - used).any()):
                raise fail_with_history(ArithmeticError("inconsistent functional slice likelihood counter"),
                    "batched constrained slice counter", unknown_calls=True,
                    attempt={"chain_keys": chain_keys, "threshold": threshold, "returned_info": infos,
                             "batch_key": batch_key})
            total_calls += int(used.sum())
            for phase, count in zip(("left", "right", "shrinkage"), lane_counts.sum(axis=(0, 1))):
                calls[phase] += int(count)
            batched_record = (chain_keys, parent_indices, infos)
            positions = np.asarray(finals.position)
            if not np.asarray(infos.numerically_valid).all():
                failure = "invalid numerical native integrand evaluation in constrained slice"
                completed = False
            elif np.asarray(infos.budget_exhausted).any():
                failure = "batched slice exceeded its proven evaluation bound"
                completed = False
            elif not np.asarray(infos.slice_info.is_accepted).all():
                failure = "native slice shrinkage exhausted"
                completed = False
            elif not (np.asarray(finals.loglikelihood) > threshold).all():
                failure = "replacement is not a strict contour survivor"
                completed = False
            elif ((np.asarray(finals.logdensity) != 0.0).any() or not np.isfinite(positions).all()
                  or not ((positions >= 0.0) & (positions < 1.0)).all()):
                failure = "invalid numerical replacement unit positions"
                completed = False
            replacement_stack = finals._replace(
                loglikelihood_birth=jnp.full(q, threshold, dtype=jnp.float64))
        else:
            for lane, (parent_index, chain_key) in enumerate(zip(parent_indices, chain_keys)):
                factor = chain_factors[lane] if chain_factors is not None else pilot_factor
                current = jax.tree.map(lambda leaf, idx=int(parent_index): leaf[idx], state.particles)
                inner_keys = checked_call(jax.random.split, chain_key, self.num_inner_steps, operation="slice RNG split")
                for inner_index, slice_key in enumerate(inner_keys):
                    start_particle = current
                    current, info = checked_call(self.budgeted_slice, slice_key, current, jnp.asarray(threshold),
                        jnp.asarray(ceiling - total_calls, dtype=jnp.int64),
                        **({"factor": factor} if factor is not None else {}),
                        operation="constrained slice", native=True,
                        attempt={"key": slice_key, "start": current, "threshold": threshold,
                                 "remaining_budget": ceiling - total_calls, "batch_key": batch_key,
                                 "dead_indices": indices, "dead_particles": dead_particles,
                                 "completed_slices": tuple(slices), "completed_replacements": tuple(replacements)})
                    counts = np.asarray(info.likelihood_calls, dtype=np.int64)
                    remaining_before = ceiling - total_calls
                    if (counts.shape != (3,) or (counts < 0).any()
                            or int(counts.sum()) > remaining_before
                            or int(info.remaining_budget) != remaining_before - int(counts.sum())):
                        exc = ArithmeticError("inconsistent functional slice likelihood counter")
                        raise fail_with_history(exc, "constrained slice counter", unknown_calls=True,
                            attempt={"key": slice_key, "start": start_particle, "threshold": threshold,
                                     "remaining_budget": remaining_before, "returned_info": info,
                                     "batch_key": batch_key, "completed_slices": tuple(slices)})
                    total_calls += int(counts.sum())
                    for phase, count in zip(("left", "right", "shrinkage"), counts):
                        calls[phase] += int(count)
                    slices.append(NativeSliceAttempt(int(parent_index), inner_index, slice_key,
                                                     start_particle, current, info))
                    if not bool(info.numerically_valid):
                        failure = "invalid numerical native integrand evaluation in constrained slice"
                        completed = False
                    elif bool(info.budget_exhausted):
                        failure, stop_reason = "native likelihood budget exhausted inside constrained slice", "truncated"
                        completed = False
                    elif not bool(info.slice_info.is_accepted):
                        failure = "native slice shrinkage exhausted"
                        completed = False
                    elif not float(current.loglikelihood) > threshold:
                        failure = "replacement is not a strict contour survivor"
                        completed = False
                    elif (float(current.logdensity) != 0.0
                          or not np.isfinite(np.asarray(current.position)).all()
                          or not ((np.asarray(current.position) >= 0.0)
                                  & (np.asarray(current.position) < 1.0)).all()):
                        failure = "invalid numerical replacement unit positions"
                        completed = False
                    if not completed:
                        break
                replacements.append(current._replace(loglikelihood_birth=jnp.asarray(threshold)))
                if not completed:
                    break
        counts = self.shape[0] - np.arange(q, dtype=np.int64)
        history = NativeBatchHistory(batch_key, threshold, indices, dead_particles,
                       replacement_stack if batched else tuple(replacements), tuple(slices),
                       completed, counts, before, total_calls, batched_record)
        if not completed:
            batches.append(history)
            failed_operation = "constrained slice"
            stopping_check(stop_reason)
            break
        attempt = {"batch": replace(history, completed=False)}
        new_particles = replacement_stack if batched else checked_call(
            lambda particles=tuple(replacements): jax.tree.map(lambda *leaves: jnp.stack(leaves), *particles),
            operation="replacement assembly", attempt=attempt)
        new_state = checked_call(
            lambda live_state=state, new_points=new_particles, indices=indices:
                base.NSState(jax.tree.map(lambda live, new: live.at[indices].set(new),
                                         live_state.particles, new_points)),
            operation="live commit preparation", attempt=attempt)
        relative_dead = checked_call(_relative_log_likelihood, np.asarray(dead_particles.loglikelihood),
                                    likelihood_offset, operation="volume-accounting", attempt=attempt)
        next_logx, next_logz = logx, logz_dead
        for logl, count in zip(relative_dead, counts):
            delta = -1.0 / count
            next_logz = float(np.logaddexp(next_logz, next_logx + math.log(-math.expm1(delta)) + logl))
            next_logx += delta
        # Atomic logical commit: the live state, death history and volume
        # accounting advance together only after all replacement work clears.
        state, logx, logz_dead = new_state, next_logx, next_logz
        batches.append(history)

    # Update the actual final check's reason, never recompute it using final
    # live evidence or convert a failed transition into convergence.
    checks[-1] = replace(checks[-1], n_likelihood_calls=total_calls, reason=stop_reason)
    result = final_result()
    if stop_reason == "unresolved":
        retain(failed=result)
        raise NativeIntegrationError(failure or "experimental integration unresolved", result)
    return result
