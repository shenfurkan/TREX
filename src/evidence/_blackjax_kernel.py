# Budget-aware slice portions adapted from BlackJAX 1.7.1:
# Copyright 2020- The Blackjax Authors, licensed under Apache License 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Provided AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# Modifications: functional budget/validity carry, scalar interruption history;
# the rest of this module follows the TREX license (GPL-3.0-or-later).
"""Experimental checked BlackJAX NSS transitions and opt-in evidence driver.

The prior is uniform on the unit cube; the supplied native integrand retains
the scientific transforms/support. Invalid numerical evaluations are errors,
including proposals the slice sampler would otherwise reject. Physical zeros
remain negative infinity. Raw dead/live states and caller-owned keys are kept.
``integrate`` is experimental; only TREX_NESTED_ESTIMATOR=blackjax selects it.
Transition correctness and compression error do not establish calibration.

Uses BlackJAX 1.7.1's NSS constrained-step and base-state APIs, not its adaptive
evidence utilities. Slice caps are computational guards, not scientific budgets.
Exhausting shrinkage fails rather than silently adopting a stalled replacement.
"""

from functools import partial
from importlib.metadata import version
import math

import numpy as np

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_evidence import _compression_log_evidences, _integrate_positions
from trex.evidence._blackjax_slice import _build_batched_chains, _build_budgeted_slice
from trex.evidence._blackjax_types import (
    BLACKJAX_VERSION,
    SLICE_CALL_BOUND,
    SLICE_MAX_EXPANSIONS,
    SLICE_MAX_SHRINKAGE,
    CheckedSliceInfo,
)

# Historical name imported by the standalone profile/replay tests.
_SLICE_CALL_BOUND = SLICE_CALL_BOUND


def _require_supported_runtime():
    if version("blackjax") != BLACKJAX_VERSION:
        raise RuntimeError(f"experimental integration requires verified BlackJAX {BLACKJAX_VERSION}")
    if not models.jax.config.jax_enable_x64:
        raise RuntimeError("experimental integration requires JAX float64 execution")


class NativeNestedKernel:
    """Unit-cube constrained transitions; diagnostic and evidence-uncalibrated.

    Positions are dimensionless quantiles; log integrand/contours are nats.
    This object owns no RNG stream: callers supply initialization and step keys
    and must retain them with raw dead/live outputs for reproducibility.
    """

    claim_eligible = False
    evidence_calibrated = False

    def __init__(self, integrand, *, n_dimensions, n_live_points, num_delete, num_inner_steps, batched=False,
                 batched_initialization=False, initialization_block=64, direction_metric="fixed-isotropic",
                 speculative_expansions=0):
        from blackjax.mcmc.slice import build_kernel as build_slice_kernel, stepping_out
        from blackjax.ns import base, nss
        from blackjax.ns.from_mcmc import update_with_mcmc_take_last
        from jax.experimental import checkify

        for name, value in (("n_dimensions", n_dimensions), ("n_live_points", n_live_points),
                            ("num_delete", num_delete), ("num_inner_steps", num_inner_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_delete >= n_live_points:
            raise ValueError("num_delete must leave at least one live survivor")
        if n_live_points <= n_dimensions:
            raise ValueError("live population must exceed the covariance dimension")
        if not isinstance(batched_initialization, (bool, np.bool_)):
            raise ValueError("batched_initialization must be a boolean")
        if (isinstance(initialization_block, bool) or not isinstance(initialization_block, int)
                or initialization_block < 1 or initialization_block & (initialization_block - 1)):
            raise ValueError("initialization_block must be a positive power of two")
        if direction_metric not in ("fixed-isotropic", "fixed-pilot", "split-ensemble"):
            raise ValueError("direction_metric must be fixed-isotropic, fixed-pilot or split-ensemble")
        if isinstance(speculative_expansions, bool) or not isinstance(speculative_expansions, int) \
                or speculative_expansions not in (0, 4, 8):
            raise ValueError("speculative_expansions must be 0, 4 or 8")
        if speculative_expansions and not batched:
            # Speculation exists only in the batched-chain slice; accepting it
            # here would enter checkpoint identity and result provenance
            # without changing a single scalar-path evaluation.
            raise ValueError("speculative_expansions requires batched=True")
        self.shape = (n_live_points, n_dimensions)
        self.batched = bool(batched)
        self.batched_initialization = bool(batched_initialization)
        self.initialization_block = initialization_block
        self.direction_metric = str(direction_metric)
        self.speculative_expansions = speculative_expansions
        self.num_delete = num_delete
        self.num_inner_steps = num_inner_steps
        self.integrand = integrand
        jax, jnp = models.jax, models.jnp

        def particle(position, loglikelihood_birth=-jnp.inf):
            result = integrand.evaluate(position)
            checkify.check(result.numerically_valid, "invalid numerical native integrand evaluation")
            checkify.check(~jnp.isnan(result.log_integrand) & ~jnp.isposinf(result.log_integrand),
                           "nonfinite numerical native log integrand")
            prior = jnp.where(jnp.all((position >= 0.0) & (position < 1.0)), 0.0, -jnp.inf)
            return base.StateWithLogLikelihood(position, prior, result.log_integrand,
                                               jnp.asarray(loglikelihood_birth, dtype=jnp.float64))

        # The slice implementation returns its explicit acceptance flag when
        # shrinkage exhausts. These caps do not reduce any scientific draw budget.
        slice_kernel = build_slice_kernel(interval=stepping_out, max_expansions=SLICE_MAX_EXPANSIONS,
                                          max_shrinkage=SLICE_MAX_SHRINKAGE)
        constrained_step = nss.slice_constrained_step(particle, slice_kernel, nss.covariance_proposal)
        # Checkify each scalar slice BEFORE BlackJAX scans and vmaps it. An
        # outer checkify-of-vmap-of-while is unsupported by JAX. Keep every
        # functional error as output data, not a callback or a discarded flag.
        checked_slice = checkify.checkify(constrained_step)

        def checked_constrained_step(key, state, loglikelihood_0, **params):
            error, (new_state, info) = checked_slice(key, state, loglikelihood_0, **params)
            return new_state, CheckedSliceInfo(info, error)

        update = update_with_mcmc_take_last(checked_constrained_step, num_inner_steps, num_delete)

        def initialize(positions, key):
            del key  # Input positions already own the caller's prior RNG draws.
            return base.NSState(jax.vmap(particle)(positions))

        def prepare(state, threshold=None):
            likelihoods = state.particles.loglikelihood
            checkify.check(jnp.all(~jnp.isnan(likelihoods) & ~jnp.isposinf(likelihoods)),
                           "invalid numerical incoming live likelihood")
            positions = state.particles.position
            checkify.check(jnp.all(jnp.isfinite(positions) & (positions >= 0.0) & (positions < 1.0)),
                           "invalid incoming live unit positions")
            if threshold is None:
                threshold = jnp.sort(likelihoods)[num_delete - 1]
            survivors = jnp.any(likelihoods > threshold)
            checkify.check(survivors, "no strict live survivor above deletion contour")
            # Center displacements, not large absolute coordinates. Identical
            # points then remain exactly zero even when a rounded mean of their
            # absolute coordinates would manufacture a tiny nonzero variance.
            displacements = positions - positions[0]
            centered = displacements - jnp.mean(displacements, axis=0)
            covariance = centered.T @ centered / (n_live_points - 1)
            factor = jnp.linalg.cholesky(covariance)
            factor_valid = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor) > 0.0)
            checkify.check(factor_valid, "invalid numerical live covariance factor")
            return factor, threshold

        def validate(state, new_state, info, threshold):
            checkify.check(jnp.all(info.update_info.is_accepted), "native slice shrinkage exhausted")
            checkify.check(jnp.all(new_state.particles.loglikelihood >= threshold),
                           "invalid replacement below deletion contour")
            replacements = new_state.particles.loglikelihood[base.delete_fn(state, num_delete)[1]]
            checkify.check(jnp.all(replacements > threshold), "replacement is not a strict contour survivor")

        checked_prepare = checkify.checkify(prepare)
        checked_validate = checkify.checkify(validate)

        def transition(key, state):
            preparation_error, (factor, threshold) = checked_prepare(state)

            def inner(inner_key, live_state, contour):
                return update(inner_key, live_state, contour, covariance_factor=factor)

            kernel = base.build_kernel(partial(base.delete_fn, num_delete=num_delete), inner)
            new_state, info = kernel(key, state)
            validation_error, _ = checked_validate(state, new_state, info, threshold)
            return (preparation_error, info.update_info.errors, validation_error), (new_state, info)

        self.checked_initialize = jax.jit(checkify.checkify(initialize))
        self.checked_prepare = jax.jit(checked_prepare)
        self.checked_step = jax.jit(transition)

        def integration_evaluate(position, birth):
            result = integrand.evaluate(position)
            if result.log_integrand.shape != () or result.log_integrand.dtype != jnp.float64:
                raise ValueError("experimental integration requires scalar float64 native log integrand")
            if result.numerically_valid.shape != () or result.numerically_valid.dtype != jnp.bool_:
                raise ValueError("experimental integration requires scalar boolean native validity")
            valid = (result.numerically_valid & jnp.all(jnp.isfinite(position))
                     & ~jnp.isnan(result.log_integrand) & ~jnp.isposinf(result.log_integrand))
            prior = jnp.where(jnp.all((position >= 0.0) & (position < 1.0)), 0.0, -jnp.inf)
            return base.StateWithLogLikelihood(position, prior, result.log_integrand,
                                               jnp.asarray(birth, dtype=jnp.float64)), valid

        self.integration_evaluate = jax.jit(integration_evaluate)
        self.integration_prepare = jax.jit(checkify.checkify(prepare))
        raw_slice = _build_budgeted_slice(integration_evaluate, n_dimensions)
        self.budgeted_slice = jax.jit(raw_slice)
        # Speculation is a batched-chain-only execution policy: the scalar path
        # keeps the exact sequential kernel object and budget interruption.
        chain_slice = raw_slice
        if self.speculative_expansions:
            chain_slice = _build_budgeted_slice(integration_evaluate, n_dimensions,
                                                speculation=self.speculative_expansions)
        self.batched_chains = _build_batched_chains(chain_slice, num_inner_steps)
        self.compression_log_evidences = jax.jit(_compression_log_evidences)
        if self.batched_initialization:
            # One device call per padded block; padding repeats dispatched lanes
            # so no evaluation leaves the block's own corners.
            self.batched_initialization_evaluate = jax.jit(
                jax.vmap(integration_evaluate, in_axes=(0, None)))

    def initialize(self, positions, key):
        """Validate supplied prior positions and evaluate each exactly once.

        ``key`` is accepted for interface symmetry and not consumed: the
        positions already carry the caller's prior draws.
        """
        _require_supported_runtime()
        values = np.asarray(positions)
        if values.dtype.kind not in "fiu":
            raise ValueError("initial live positions must be real numeric positions")
        if values.shape != self.shape:
            raise ValueError(f"initial live unit positions must have shape {self.shape}")
        if not np.isfinite(values).all() or not ((values >= 0.0) & (values < 1.0)).all():
            raise ValueError("invalid or nonfinite initial live unit positions")
        points = models.jax.device_put(models.jnp.asarray(values, dtype=models.jnp.float64), self.integrand.device)
        error, state = self.checked_initialize(points, key)
        error.throw()
        return state

    def step(self, key, state):
        """Return raw dead/live transition only after all checked errors clear."""
        _require_supported_runtime()
        if state.particles.position.shape != self.shape:
            raise ValueError(f"live unit positions must have shape {self.shape}")
        # Reject invalid incoming states before executing BlackJAX's survivor
        # selection; its no-survivor fallback is not permitted in this adapter.
        error, _ = self.checked_prepare(state)
        error.throw()
        errors, result = self.checked_step(key, state)
        for error in errors:
            error.throw()
        return result

    def integrate(self, positions, key, *, evidence_tolerance, max_likelihood_calls, random_seed,
                  checkpoint=None, checkpoint_identity=None, deadline=math.inf):
        """Experimental nested integration of caller-supplied IID prior points.

        Args:
            positions: Unit-cube prior draws of shape ``self.shape``.
            key: One caller-owned JAX key (typed or legacy); never reseeded.
            evidence_tolerance: Positive remaining-evidence stop, in nats.
            max_likelihood_calls: Hard positive ceiling on logical integrand
                evaluations, including rejected and outside-cube proposals.
            random_seed: Non-negative provenance label only.
            checkpoint, checkpoint_identity: Optional trusted recovery file and
                the integrand identity it is bound to.
            deadline: ``time.monotonic()`` value [s] at which to pause at the
                next durable boundary; ``inf`` never pauses.

        Returns a ``NativeEvidenceResult``. A budget stop is ``truncated``, never
        convergence; a finite all-live plateau stops conditionally on no unseen
        higher mode. Numerical or all-zero-support failures raise
        ``NativeIntegrationError`` carrying the unresolved partial record; invalid
        controls raise ``ValueError`` before any evaluation. If a device fault
        loses a call counter, counts are completed-call lower bounds and
        ``likelihood_calls_exact`` is false.

        Method: joint deletions with geometric order-statistic volumes (live
        counts N, N-1, ..., N-q+1) and whole tied contours evicted together
        (Fowlie et al. 2021, MNRAS 503, 1199, DOI 10.1093/mnras/stab590, Eqs. 6,
        14, 19, Algorithm 2). Weights are L_i (X_prev - X_i) plus X_remaining / N
        per final live point. The error is the dispersion of 256 compression
        simulations, not a mixing or coverage certificate. Sums are kept
        relative to the first live maximum and restored only in absolute views;
        the relative fields are the precise audit for large offsets.

        Unlike ``step``, slice directions use a fixed metric: isotropic by
        default, or the opt-in ``direction_metric`` (one ridge-stabilized pilot
        factor from the caller's initial positions, or per-batch factors from the
        opposite strict-survivor half). No metric is refitted inside an inner
        step, and neither option is a calibrated adaptation; the covariance
        guards still run. With ``batched_initialization`` prior points are
        evaluated in whole padded blocks, but only the prefix a sequential loop
        would have reached is logically counted and kept. Batched chains reserve
        the worst-case batch budget before dispatch and fall back to scalar
        slices near the ceiling; their masked physical evaluations are not
        counted. ``speculative_expansions`` only changes how many stepping-out
        probes one batched expansion wave evaluates: logical counts, brackets
        and states stay those of the sequential loop, and no speedup is claimed.
        Replay is deterministic for identical inputs, versions and
        device; recovery resumes from the last complete deletion boundary, so a
        killed batch is replayed, not redrawn.
        """
        for name, value in (("max_likelihood_calls", max_likelihood_calls), ("random_seed", random_seed)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (1 if name == "max_likelihood_calls" else 0):
                raise ValueError(f"{name} is outside its allowed integer domain")
        if max_likelihood_calls > np.iinfo(np.int64).max:
            raise ValueError("max_likelihood_calls exceeds the int64 counter domain")
        if (isinstance(evidence_tolerance, (bool, np.bool_, complex, np.complexfloating))
                or not isinstance(evidence_tolerance, (int, float, np.number))):
            raise ValueError("evidence_tolerance must be finite and positive in nats")
        try:
            evidence_tolerance = float(evidence_tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("evidence_tolerance must be finite and positive in nats") from exc
        if not math.isfinite(evidence_tolerance) or evidence_tolerance <= 0.0:
            raise ValueError("evidence_tolerance must be finite and positive in nats")
        supplied = np.asarray(positions)
        if supplied.dtype.kind not in "fiu":
            raise ValueError("initial live positions must be real numeric positions")
        values = np.asarray(supplied, dtype=np.float64)
        if values.shape != self.shape:
            raise ValueError(f"initial live unit positions must have shape {self.shape}")
        if not np.isfinite(values).all() or not ((values >= 0.0) & (values < 1.0)).all():
            raise ValueError("invalid or nonfinite initial live unit positions")
        if isinstance(deadline, (bool, np.bool_)) or not isinstance(deadline, (int, float, np.integer, np.floating)):
            raise ValueError("deadline must be a monotonic-clock time in seconds")
        if math.isnan(deadline):
            raise ValueError("deadline must be a monotonic-clock time in seconds, not NaN")
        _require_supported_runtime()
        jax = models.jax
        # Normalize typed and legacy raw keys alike before the identity binding reads them.
        try:
            normalized_key = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
        except (TypeError, ValueError) as exc:
            raise ValueError("experimental integration requires a single JAX key") from exc
        if normalized_key.ndim != 0:
            raise ValueError("experimental integration requires a single JAX key")
        if checkpoint is not None:
            if checkpoint_identity is None:
                raise ValueError("BlackJAX recovery requires an explicit integrand identity")
            from trex.evidence.nested_checkpoint import checkpoint_identity as bind_identity

            # Opt-in execution policies enter the identity only when they differ
            # from the defaults, so an all-default checkpoint stays compatible
            # while a policy change can never resume another policy's state.
            policy = [self.direction_metric] if self.direction_metric != "fixed-isotropic" else []
            if self.batched_initialization:
                policy.append(("batched-initialization", self.initialization_block))
            if self.speculative_expansions:
                policy.append(("speculative-expansions", self.speculative_expansions))
            checkpoint_identity = bind_identity((checkpoint_identity, values, models.jax.random.key_data(key),
                str(models.jax.random.key_impl(key)), evidence_tolerance, max_likelihood_calls, random_seed,
                self.shape, self.num_delete, self.num_inner_steps, self.batched,
                version("blackjax"), version("jax"), version("jaxlib"), self.integrand.device.platform,
                self.integrand.device.device_kind, *policy))
        # Explicit placement, including keys and budget scalars; no host-likelihood fallback.
        with models.jax.default_device(self.integrand.device):
            return _integrate_positions(self, values.copy(), key, float(evidence_tolerance),
                                        int(max_likelihood_calls), int(random_seed),
                                        checkpoint=checkpoint, identity=checkpoint_identity,
                                        deadline=float(deadline))
