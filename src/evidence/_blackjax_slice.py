# Budget-aware slice portions adapted from BlackJAX 1.7.1:
# Copyright 2020- The Blackjax Authors, licensed under Apache License 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Provided AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modifications: functional budget/validity carry, scalar interruption history;
# the rest of this module follows the TREX license (GPL-3.0-or-later).
"""Budgeted BlackJAX slice transitions."""

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_types import (
    SLICE_CALL_BOUND,
    SLICE_MAX_EXPANSIONS,
    SLICE_MAX_SHRINKAGE,
    BudgetSliceInfo,
    _SliceBudget,
)


def _build_budgeted_slice(evaluate, dimension, speculation=0):
    """Scalar BlackJAX 1.7.1 stepping-out/shrinkage with a threaded hard budget.

    Adapted from blackjax/mcmc/slice.py (Apache-2.0, BlackJAX Authors),
    ``stepping_out``, ``_shrink``, ``_univariate_slice`` and ``build_kernel``.
    Retains their key splits, random bracket allocation and accepted-state cache.
    Only computational interruption and functional numerical validity are added.
    Budget-interrupted proposals are never committed by the outer driver.
    The default expansion uses no vmap: no masked or padded likelihood
    evaluations follow a limit; the opt-in ``speculation`` wave below is the
    single exception and never changes a logical counter.
    ``factor`` is this chain's fixed direction metric; ``None`` keeps the
    isotropic identity used by every default run.

    ``speculation`` (batched chains only; 0 keeps the sequential expansion) turns
    each expansion phase into waves of that many device-parallel probes. The wave
    emits the same iterated ``t + step`` positions the sequential loop would, so
    the first outside probe, bracket, accepted state and every logical counter
    are unchanged: later probes of a wave and padding repeats are evaluated but
    discarded and can never affect validity, the budget or the returned state.
    The only gain is fewer sequential while-loop waves under ``vmap``; no
    measured speedup is claimed.
    """
    from blackjax.mcmc.slice import SliceInfo
    from blackjax.ns.nss import sample_direction_from_covariance_factor

    jax, jnp = models.jax, models.jnp

    def kernel(key, state, threshold, remaining_budget, direction=None, factor=None):
        prop_key, slice_key = jax.random.split(key)
        # The metric is fixed for all inner steps: isotropic unless the caller
        # supplied an opt-in factor, so no uncorrected position law is fitted here.
        if direction is None:
            metric = jnp.eye(dimension, dtype=jnp.float64) if factor is None else factor
            direction = sample_direction_from_covariance_factor(prop_key, state.position, metric)
        level_key, interval_key, shrink_key = jax.random.split(slice_key, 3)
        level = state.logdensity + jnp.log(jax.random.uniform(level_key))
        u_key, jk_key = jax.random.split(interval_key)
        left = -jax.random.uniform(u_key)
        right = left + 1.0
        j = jnp.floor(SLICE_MAX_EXPANSIONS * jax.random.uniform(jk_key)).astype(jnp.int64)
        k = SLICE_MAX_EXPANSIONS - 1 - j
        initial = _SliceBudget(remaining_budget, jnp.zeros(3, dtype=jnp.int64),
                               jnp.asarray(True), jnp.asarray(False),
                               state.position, state.loglikelihood,
                               jnp.asarray(-1, dtype=jnp.int64), state.position)

        def proposal(t, phase, budget):
            position = state.position + t * direction

            def active(b):
                candidate, valid = evaluate(position, threshold)
                new_budget = _SliceBudget(b.remaining - 1, b.counts.at[phase].add(1),
                                         b.valid & valid, b.exhausted,
                                         position, candidate.loglikelihood, b.blocked_phase, b.blocked_position)
                inside = (candidate.logdensity >= level) & (candidate.loglikelihood > threshold) & valid
                return candidate, inside, new_budget

            def inactive(b):
                # Any non-positive remaining budget is exhaustion, never a
                # shrinkage failure.
                blocked = b.valid & (b.remaining <= 0) & ~b.exhausted
                return state, jnp.asarray(False), b._replace(
                    exhausted=b.exhausted | blocked,
                    blocked_phase=jnp.where(blocked, phase, b.blocked_phase),
                    blocked_position=jnp.where(blocked, position, b.blocked_position))

            return jax.lax.cond(budget.valid & (budget.remaining > 0), active, inactive, budget)

        _, left_inside, budget = proposal(left, 0, initial)
        initial_left_calls = budget.counts[0]

        def expand(phase, step):
            def body(carry):
                t, n, _, b = carry
                _, inside, next_budget = proposal(t + step, phase, b)
                return t + step, n - 1, inside, next_budget

            return body

        def expansion_condition(carry):
            _, n, inside, b = carry
            return inside & (n > 0) & b.valid & ~b.exhausted

        def speculative_expand(phase, step):
            """One wave of up to ``speculation`` probes of a single expansion phase.

            The probes reuse the sequential loop's iterated ``t + step`` values and
            its first out-of-slice test, so the bracket and the first outside probe
            are the ones the sequential loop would find. Probes after that one and
            padding repeats of the last wanted probe are physical only: they never
            extend ``n``, never enter ``counts``/``remaining``, never contribute to
            ``valid`` and never reach the returned state.
            """
            width = jnp.asarray(speculation, dtype=jnp.int64)
            offsets = jnp.arange(1, speculation + 1, dtype=jnp.int64)

            def body(carry):
                t, n, _, b = carry
                probes = []
                probe_t = t
                for _ in range(speculation):
                    probe_t = probe_t + step
                    probes.append(probe_t)
                probe_t = jnp.stack(probes)
                wanted = jnp.minimum(offsets, jnp.minimum(n, width))
                positions = state.position + probe_t[wanted - 1, None] * direction
                candidates, valids = jax.vmap(lambda point: evaluate(point, threshold))(positions)
                inside = ((candidates.logdensity >= level) & (candidates.loglikelihood > threshold)
                          & valids)
                attempts = jnp.minimum(jnp.minimum(n, width), jnp.maximum(b.remaining, 0))
                outside = (offsets <= attempts) & ~inside
                # Logical probes through and including the first outside one.
                used = jnp.where(outside.any(), jnp.asarray(jnp.argmax(outside), dtype=jnp.int64) + 1,
                                 attempts)
                last = jnp.maximum(used, jnp.asarray(1, jnp.int64)) - 1
                following = jnp.minimum(used, width - 1)
                started = used > 0
                # The sequential loop's next attempt would be inactive (no budget).
                # Continuing the same iterated addition keeps that position exact;
                # ``used < n`` holds whenever this is set, so one probe remains.
                next_t = jnp.where(used < width, probe_t[following], probe_t[width - 1] + step)
                blocked = (used < n) & ~outside.any() & (b.remaining - used <= 0)
                last_position = state.position + probe_t[last] * direction
                # The sequential loop's last executed body is the inactive attempt,
                # which returns its unevaluated next ``t``; keep that for exactness.
                new_t = jnp.where(blocked, next_t, jnp.where(started, probe_t[last], t))
                return (
                    new_t,
                    n - used,
                    jnp.where(started, inside[last], jnp.asarray(False)),
                    _SliceBudget(b.remaining - used, b.counts.at[phase].add(used),
                                 b.valid & jnp.all(jnp.where(offsets <= used, valids, True)),
                                 b.exhausted | blocked,
                                 jnp.where(started, last_position, b.last_position),
                                  jnp.where(started, candidates.loglikelihood[last], b.last_log_integrand),
                                 jnp.where(blocked, jnp.asarray(phase, dtype=jnp.int64), b.blocked_phase),
                                 jnp.where(blocked, state.position + next_t * direction,
                                           b.blocked_position)))

            return body

        expansion_body = speculative_expand if speculation else expand
        left, _, _, budget = jax.lax.while_loop(
            expansion_condition, expansion_body(0, -1.0), (left, j, left_inside, budget))
        _, right_inside, budget = proposal(right, 1, budget)
        initial_right_calls = budget.counts[1]
        right, _, _, budget = jax.lax.while_loop(
            expansion_condition, expansion_body(1, 1.0), (right, k, right_inside, budget))
        bracket_left, bracket_right = left, right

        def shrink_condition(carry):
            _, _, _, _, found, b = carry
            return ~found & (b.counts[2] < SLICE_MAX_SHRINKAGE) & b.valid & ~b.exhausted

        def shrink(carry):
            lo, hi, rng, current, _, b = carry
            rng, subkey = jax.random.split(rng)
            t = lo + jax.random.uniform(subkey) * (hi - lo)
            candidate, found, next_budget = proposal(t, 2, b)
            current = jax.tree.map(lambda new, old: jnp.where(found, new, old), candidate, current)
            lo = jnp.where(t < 0.0, t, lo)
            hi = jnp.where(t >= 0.0, t, hi)
            return lo, hi, rng, current, found, next_budget

        _, _, _, new_state, found, budget = jax.lax.while_loop(
            shrink_condition, shrink, (left, right, shrink_key, state, jnp.asarray(False), budget))
        info = SliceInfo(found, budget.counts[0] - initial_left_calls + budget.counts[1] - initial_right_calls,
                         budget.counts[2], bracket_left, bracket_right)
        return new_state, BudgetSliceInfo(info, budget.valid, budget.exhausted, budget.counts,
                                         budget.remaining, budget.last_position, budget.last_log_integrand,
                                         budget.blocked_phase, budget.blocked_position)

    return kernel


def _build_batched_chains(slice_kernel, num_inner_steps):
    """All chains of one deletion batch in one device call: vmap over chains, scan over steps.

    Each chain consumes exactly the scalar driver's key stream
    (``split(chain_key, num_inner_steps)``). Cross-execution arithmetic is tested
    on identical points within the native float64 gate, not by demanding byte
    equality of stochastic trajectories; estimator accuracy needs calibration.
    Every slice receives ``SLICE_CALL_BOUND`` as its budget, which it cannot
    exhaust, so the hard run ceiling is enforced by the caller before the call.
    Under vmap the budget ``cond`` becomes a select: masked elements are evaluated
    and discarded, so logical counts stay exact but physical counts are not known.
    Generate directions with device-side scalar map, not vmap: the pinned CPU
    compiler's scalar and SIMD rsqrt normalization round differently. BlackJAX's
    fixed-metric direction uses the position's structure only, so sampling it
    before the chains does not depend on their evolving positions. Likelihood
    work remains vmapped; scalar RNG/normalization has no host slice dispatch.
    ``factors`` holds one fixed (dimension, dimension) factor per chain; ``None``
    keeps the single isotropic identity every default run uses. A slice kernel
    built with ``speculation`` may be passed instead of the sequential one: its
    expansion waves probe several candidates per while-loop wave, but the first
    outside probe is the one the sequential loop selects and discarded probes
    never affect the state or a logical counter, so the transition law is
    unchanged. The gain is fewer sequential while-loop waves inside ``vmap``; no
    measured speedup is claimed.
    """
    from blackjax.ns.nss import sample_direction_from_covariance_factor

    jax, jnp = models.jax, models.jnp

    def chain(keys, start, threshold, directions):
        def step(particle, inputs):
            key, direction = inputs
            return slice_kernel(key, particle, threshold, jnp.asarray(SLICE_CALL_BOUND, dtype=jnp.int64),
                                direction=direction)

        return jax.lax.scan(step, start, (keys, directions))

    def chains(chain_keys, starts, threshold, factors=None):
        keys = jax.vmap(lambda key: jax.random.split(key, num_inner_steps))(chain_keys)
        prop_keys = jax.vmap(jax.vmap(lambda key: jax.random.split(key)[0]))(keys)
        flat_keys = prop_keys.reshape((-1, *prop_keys.shape[2:]))
        positions = jnp.repeat(starts.position, num_inner_steps, axis=0)
        # Vectorizing normalization can round differently and change keyed slice decisions.
        if factors is None:
            factor = jnp.eye(starts.position.shape[-1], dtype=jnp.float64)
            directions = jax.lax.map(lambda inputs: sample_direction_from_covariance_factor(
                inputs[0], inputs[1], factor), (flat_keys, positions))
        else:
            # Select each chain's fixed factor by its lane, one per inner step.
            lane_factors = jnp.repeat(factors, num_inner_steps, axis=0)
            directions = jax.lax.map(lambda inputs: sample_direction_from_covariance_factor(
                inputs[0], inputs[1], inputs[2]), (flat_keys, positions, lane_factors))
        directions = directions.reshape((chain_keys.shape[0], num_inner_steps, *starts.position.shape[1:]))
        return jax.vmap(chain, in_axes=(0, 0, None, 0))(keys, starts, threshold, directions)

    return jax.jit(chains)
