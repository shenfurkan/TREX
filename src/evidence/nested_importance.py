"""Defensive importance refinement for nested evidence."""

from __future__ import annotations

import math
from time import monotonic as _monotonic
from typing import Optional

import numpy as np
from scipy.special import logsumexp
from trex.foundation.errors import NumericalLikelihoodError


from trex.runtime.hardware import resolve_n_jobs
from trex.runtime.logging import collect_worker_warnings, start_worker_warning_spool, worker_warning_report
from trex.evidence.nested_checkpoint import (
    NestedCheckpointPaused,
    checkpoint_context as _checkpoint_context,
    checkpoint_identity as _checkpoint_identity,
    load_importance_adaptation,
    load_importance_progress as _load_importance_progress,
    save_importance_adaptation,
    save_importance_progress as _save_importance_progress,
)
from trex.evidence.nested_controls import (
    ImportanceEvidenceResult,
    _IMPORTANCE_CENTRES,
    _IMPORTANCE_CHECKPOINT_SECONDS,
    _IMPORTANCE_DEFENSIVE_FRACTION,
    _IMPORTANCE_DENSITY_CHUNK,
    _IMPORTANCE_DRAWS,
    _IMPORTANCE_EVALUATION_CHUNK,
)
from trex.evidence.nested_sampling import _cached_importance_block, _worker_pool


_PROPOSAL_KERNEL_MIXTURE = "kernel-mixture"
_PROPOSAL_PROBIT_GAUSSIAN = "probit-gaussian"
_PROPOSAL_CHOICES = (_PROPOSAL_KERNEL_MIXTURE, _PROPOSAL_PROBIT_GAUSSIAN)


def _validate_proposal(proposal):
    if proposal not in _PROPOSAL_CHOICES:
        raise ValueError("proposal must be one of {0}".format(", ".join(_PROPOSAL_CHOICES)))
    return proposal


class _LogLikelihoodBlock:
    """``lnL(prior_transform(u))`` for a block of unit-cube points.

    The integrand dynesty integrates, as a picklable callable for a spawn pool.
    """

    def __init__(self, prior_transform, log_likelihood):
        self.prior_transform = prior_transform
        self.log_likelihood = log_likelihood

    def __call__(self, points: np.ndarray) -> np.ndarray:
        values = np.array(
            [self.log_likelihood(self.prior_transform(np.array(point, dtype=float))) for point in points],
            dtype=float,
        )
        invalid = np.isnan(values) | np.isposinf(values)
        if np.any(invalid):
            raise NumericalLikelihoodError("scalar importance likelihood returned NaN or +inf", int(invalid.sum()))
        return values


def _kernel_mixture(samples, weights, rng, *, regularize=False):
    """Equal-weight centres and Scott-bandwidth Gaussian kernels of the defensive proposal."""
    count = _IMPORTANCE_CENTRES
    ndim = samples.shape[1]
    positions = (rng.random() + np.arange(count)) / count
    centres = samples[np.minimum(np.searchsorted(np.cumsum(weights), positions), weights.size - 1)]
    # One weighted atom has zero empirical spread; np.cov's unbiased denominator is undefined.
    covariance = (np.zeros((ndim, ndim)) if regularize and np.count_nonzero(weights) == 1
                  else np.atleast_2d(np.cov(samples.T, aweights=weights)))
    bandwidth = count ** (-1.0 / (ndim + 4))
    if regularize:
        # A retained posterior can be nearly rank-deficient (a short or collapsed
        # run, or a coordinate the history never moved); a 1e-10 covariance ridge
        # keeps the kernels proper and is negligible for a well-conditioned one.
        covariance = covariance + 1e-10 * np.eye(ndim)
    chol = np.linalg.cholesky(bandwidth ** 2 * covariance)
    chol_inv = np.linalg.inv(chol)
    whitened = centres @ chol_inv.T
    return {"centres": centres, "chol": chol, "chol_inv": chol_inv, "whitened": whitened,
            "centre_sq": np.sum(whitened ** 2, axis=1),
            "log_norm": -0.5 * ndim * math.log(2.0 * math.pi) - float(np.sum(np.log(np.diag(chol))))}


def _mixture_draws(mixture, rng, n_draws, ndim):
    """Draw from the prior/kernel mixture; points outside the cube stay and count in the mean."""
    uniform = rng.random(n_draws) < _IMPORTANCE_DEFENSIVE_FRACTION
    draws = rng.random((n_draws, ndim))
    kernel = ~uniform
    picks = rng.integers(_IMPORTANCE_CENTRES, size=int(kernel.sum()))
    draws[kernel] = mixture["centres"][picks] + rng.standard_normal((int(kernel.sum()), ndim)) @ mixture["chol"].T
    return draws, np.flatnonzero(np.all((draws > 0.0) & (draws < 1.0), axis=1))


def _mixture_log_density(mixture, points):
    """Log density of the prior/kernel mixture at points inside the unit cube."""
    log_q = np.empty(len(points))
    chunk = _IMPORTANCE_DENSITY_CHUNK
    for lo in range(0, len(points), chunk):
        z = points[lo:lo + chunk] @ mixture["chol_inv"].T
        sq = np.sum(z ** 2, axis=1)[:, None] + mixture["centre_sq"][None, :] - 2.0 * z @ mixture["whitened"].T
        log_kernel = mixture["log_norm"] + logsumexp(-0.5 * sq, axis=1) - math.log(_IMPORTANCE_CENTRES)
        log_q[lo:lo + chunk] = np.logaddexp(
            math.log(_IMPORTANCE_DEFENSIVE_FRACTION),
            math.log1p(-_IMPORTANCE_DEFENSIVE_FRACTION) + log_kernel,
        )
    return log_q


def _probit_mixture(samples, weights, *, regularize=False):
    """One Gaussian component on the probit scale of the defensive proposal.

    ``z = Phi^{-1}(u)`` maps the open unit cube to R^d, so the weighted
    posterior mean and covariance there describe one Gaussian with complete
    support in the cube. A single weighted atom has no empirical spread --
    np.cov's unbiased denominator is undefined -- so it gets a zero
    covariance. The ``1e-10 * max(trace(C)/d, 1)`` diagonal ridge is negligible
    against the unit probit scale and keeps Cholesky defined for a collapsed
    posterior. ``regularize`` is accepted for call-site parity with
    ``_kernel_mixture``; the single-component fit always applies both
    safeguards.
    """
    from scipy.special import ndtri

    weights = np.asarray(weights, dtype=float)
    active = weights > 0.0
    # Zero-weight boundary samples cannot influence the fit; ndtri of an
    # exactly 0 or 1 weighted sample is infinite and is refused instead.
    z = ndtri(np.asarray(samples, dtype=float)[active])
    if not np.isfinite(z).all():
        raise ValueError("probit proposal needs weighted samples strictly inside (0, 1)")
    mean = np.average(z, axis=0, weights=weights[active])
    ndim = z.shape[1]
    covariance = (np.zeros((ndim, ndim)) if np.count_nonzero(weights) == 1
                  else np.atleast_2d(np.cov(z.T, aweights=weights[active])))
    covariance = covariance + 1e-10 * max(float(np.trace(covariance)) / ndim, 1.0) * np.eye(ndim)
    chol = np.linalg.cholesky(covariance)
    chol_inv = np.linalg.inv(chol)
    return {"mean": mean, "chol": chol, "chol_inv": chol_inv,
            "log_norm": -0.5 * ndim * math.log(2.0 * math.pi) - float(np.sum(np.log(np.diag(chol))))}


def _probit_draws(mixture, rng, n_draws, ndim):
    """Draw the probit-Gaussian/prior mixture; returns draws and the evaluated subset.

    Neither Gaussian coordinates nor unit-cube draws are clipped or redrawn:
    moving a draw inward would introduce atoms absent from the continuous
    proposal density used for weighting. ``ndtr`` rounds ``z > ~8.3`` to
    exactly 1.0 (and ``z < ~-38`` to 0.0), i.e. onto the boundary of the
    float64 cube although the exact transform lies inside it. Such a draw is
    handled like a kernel-mixture draw outside the cube: it stays in
    ``n_draws`` and counts as a zero contribution, but is not evaluated. The
    only mass this omits is the integrand over that unrepresentable sliver,
    whose prior volume is at most ``ndim * 2**-53`` and which the cube
    integrand cannot evaluate either. A nonfinite Gaussian coordinate still
    raises: it means a broken fit, not boundary rounding.
    """
    from scipy.special import ndtr

    fraction = _IMPORTANCE_DEFENSIVE_FRACTION
    uniform = rng.random(n_draws) < fraction
    draws = rng.random((n_draws, ndim))
    gaussian = ~uniform
    z = mixture["mean"] + rng.standard_normal((int(gaussian.sum()), ndim)) @ mixture["chol"].T
    if not np.isfinite(z).all():
        raise ArithmeticError("probit proposal produced a nonfinite Gaussian coordinate")
    draws[gaussian] = ndtr(z)
    return draws, np.flatnonzero(np.all((draws > 0.0) & (draws < 1.0), axis=1))


def _probit_log_density(mixture, points):
    """Log density of the probit-Gaussian/prior mixture at points inside the cube.

    Changing to ``z = Phi^{-1}(x)`` turns the Gaussian component into
    ``N(z; mean, C) dz`` with ``dz = dx / prod_i phi(z_i)``, so the exact
    Jacobian is ``-sum_i log phi(z_i)``; the uniform component has density 1
    inside the cube. Chunking bounds memory only.
    """
    from scipy.special import ndtri

    fraction = _IMPORTANCE_DEFENSIVE_FRACTION
    log_q = np.empty(len(points))
    chunk = _IMPORTANCE_DENSITY_CHUNK
    for lo in range(0, len(points), chunk):
        z = ndtri(points[lo:lo + chunk])
        whitened = (z - mixture["mean"]) @ mixture["chol_inv"].T
        log_gauss = mixture["log_norm"] - 0.5 * np.sum(whitened ** 2, axis=1)
        log_jacobian = -np.sum(-0.5 * z ** 2 - 0.5 * math.log(2.0 * math.pi), axis=1)
        log_q[lo:lo + chunk] = np.logaddexp(
            math.log(fraction), math.log1p(-fraction) + log_gauss + log_jacobian,
        )
    return log_q


def _draw_proposal(proposal, samples, weights, rng, n_draws, ndim):
    """Fit the selected defensive family and draw it; the default calls are unchanged."""
    if proposal == _PROPOSAL_PROBIT_GAUSSIAN:
        mixture = _probit_mixture(samples, weights, regularize=True)
        return mixture, *_probit_draws(mixture, rng, n_draws, ndim)
    mixture = _kernel_mixture(samples, weights, rng, regularize=True)
    return mixture, *_mixture_draws(mixture, rng, n_draws, ndim)


def _proposal_log_density(proposal, mixture, points):
    """Log density of the selected defensive family at points inside the cube."""
    if proposal == _PROPOSAL_PROBIT_GAUSSIAN:
        return _probit_log_density(mixture, points)
    return _mixture_log_density(mixture, points)


def _adapt_proposal(samples, weights, evaluate, rng, *, checkpoint=None, evaluation_chunk=None,
                    proposal="kernel-mixture"):
    """Move the proposal onto the integrand in small tempered passes.

    Population Monte Carlo (Cappe et al. 2004, JCGS 13, 907) with adaptive tempering: each pass
    keeps the importance weights raised to the largest power that still leaves the target ESS,
    so a pilot far from the mass widens instead of collapsing. These draws only shape the final
    proposal; the estimate uses fresh draws, so adaptation cannot bias it.
    An optional input-bound checkpoint retains complete passes and the ordered
    likelihood prefix of the current pass. Its RNG state precedes that pass's
    proposal draws, so resume reconstructs them without repeating evaluations.
    Every evaluated pass contributes to cost, even if it finds no finite weight.
    ``proposal`` selects the family each pass refits; the default kernel
    mixture keeps its original calls and RNG order. Adaptation only shapes the
    proposal: the estimate uses fresh draws.
    """
    _validate_proposal(proposal)
    from trex.evidence import nested_controls as controls
    from trex.evidence.mc_importance import tempering_exponent

    if controls._IMPORTANCE_ADAPTATION_ROUNDS == 0:
        return samples, weights, []
    ndim = samples.shape[1]
    passes = []
    values = np.empty(0, dtype=float)
    prefix, deadline, identity, estimation_started = (
        checkpoint if checkpoint is not None else (None, math.inf, None, False)
    )
    if prefix is not None:
        restored = load_importance_adaptation(
            prefix, identity, ndim, controls._IMPORTANCE_ADAPTATION_ROUNDS, controls._IMPORTANCE_ADAPTATION_DRAWS,
        )
        if estimation_started and (restored is None or not restored[-1]):
            raise ValueError("importance checkpoint is missing its completed adaptation")
        if restored is not None:
            samples, weights, passes, state, values, finished = restored
            rng.bit_generator.state = state
            if finished:
                return samples, weights, passes
    if evaluation_chunk is None:
        evaluation_chunk = _IMPORTANCE_EVALUATION_CHUNK
    last_saved = _monotonic() if checkpoint is not None else 0.0

    def retain(state, current, *, finished=False, force=False):
        nonlocal last_saved
        if checkpoint is None:
            return
        now = _monotonic()
        if force or now >= deadline or now - last_saved >= _IMPORTANCE_CHECKPOINT_SECONDS:
            if prefix is not None:
                save_importance_adaptation(
                    prefix, identity, samples, weights, passes, state, current, finished=finished,
                )
            last_saved = now
        if now >= deadline:
            raise NestedCheckpointPaused("nested wall deadline reached during importance adaptation")

    for round_index in range(len(passes), controls._IMPORTANCE_ADAPTATION_ROUNDS):
        state = rng.bit_generator.state
        retain(state, values)
        mixture, draws, inside = _draw_proposal(
            proposal, samples, weights, rng, controls._IMPORTANCE_ADAPTATION_DRAWS, ndim)
        if values.size > inside.size:
            raise ValueError("importance adaptation checkpoint has too many evaluated likelihoods")
        log_l = np.empty(inside.size, dtype=float)
        log_l[:values.size] = values
        for start in range(values.size, inside.size, evaluation_chunk):
            end = min(start + evaluation_chunk, inside.size)
            part = np.asarray(evaluate(draws[inside[start:end]]), dtype=float)
            if part.shape != (end - start,):
                raise ValueError("adaptive importance likelihood must return one value per point")
            invalid = np.isnan(part) | np.isposinf(part)
            if np.any(invalid):
                raise NumericalLikelihoodError("an adaptive importance pass returned NaN or +inf", int(invalid.sum()))
            log_l[start:end] = part
            retain(state, log_l[:end])
        log_w = log_l - _proposal_log_density(proposal, mixture, draws[inside])
        finite = np.flatnonzero(np.isfinite(log_w))
        exponent, ess = None, 0.0
        if finite.size:
            exponent = tempering_exponent(log_w[finite], controls._IMPORTANCE_ADAPTATION_ESS)
            tempered = np.exp(exponent * (log_w[finite] - np.max(log_w[finite])))
            samples, weights = draws[inside[finite]], tempered / tempered.sum()
            ess = float(1.0 / np.sum(weights ** 2))
        passes.append({"exponent": exponent, "effective_sample_size": ess,
                       "likelihood_calls": int(inside.size)})
        finished = finite.size == 0 or exponent == 1.0 or round_index + 1 == controls._IMPORTANCE_ADAPTATION_ROUNDS
        values = np.empty(0, dtype=float)
        retain(rng.bit_generator.state, values, finished=finished, force=True)
        if finished:
            break
    return samples, weights, passes


def _pool_evaluator(block, pool, chunk):
    """Evaluate unit-cube draws through the existing spawn-pool block mapping."""
    def evaluate(points):
        if not len(points):
            return np.empty(0, dtype=float)
        if pool is None:
            return np.asarray(block(points), dtype=float)
        blocks = [points[lo:lo + chunk] for lo in range(0, len(points), chunk)]
        return np.concatenate([
            np.asarray(part, dtype=float)
            for part in pool.map(_cached_importance_block, blocks, chunksize=1)
        ])
    return evaluate


def _run_importance_sampler(
    prior_transform,
    log_likelihood,
    posteriors,
    *,
    n_dimensions: int,
    random_seed: int,
    n_jobs: Optional[int],
    n_draws: int = _IMPORTANCE_DRAWS,
    native_integrand=None,
    cpu_native_integrand=None,
    proposal: str = "kernel-mixture",
) -> ImportanceEvidenceResult:
    """Defensive importance sampling of the unit-cube integral over a pooled nested posterior.

    ``posteriors`` is a sequence of ``(unit_samples, normalized_weights)``, one
    per independent nested run, pooled with equal run weight. The proposal is
    fixed before any draw: ``_IMPORTANCE_CENTRES`` equal-weight centres
    systematically resampled from the pool, Gaussian kernels with Scott's
    bandwidth on the pooled weighted covariance, mixed with the uniform prior
    at ``_IMPORTANCE_DEFENSIVE_FRACTION``. Draws outside the cube carry zero
    weight and count in the mean. A likelihood exception propagates: a failed
    evaluation is never read as zero support. ``native_integrand`` replaces
    only the ordered likelihood execution with the real scenario's JAX
    callback on an admitted accelerator. Without one, ``cpu_native_integrand``
    runs the same callback as a batched float64 JAX kernel on the CPU device;
    when the scenario has no native callback, the declared float64 scalar
    integrand is evaluated instead. Proposal draws, densities, RNG and all
    NumPy reductions below are unchanged. ``n_likelihood_calls`` includes both
    adaptation and fresh estimation points inside the cube; ``n_draws`` is
    only the final estimator budget. Device pilots/padding/retries are separate
    execution overhead and do not enter either estimator.

    ``proposal="probit-gaussian"`` is opt-in and replaces the kernel component
    with one Gaussian fit in probit coordinates ``z = Phi^{-1}(u)``, keeping the
    same uniform prior component. Its density divides by the exact probit
    Jacobian ``prod_i phi(z_i)``. Draws use the unmodified normal CDF; a draw
    that ``ndtr`` rounds onto the cube boundary is neither clipped nor
    redrawn: like a kernel draw outside the cube it is a zero contribution
    that still counts in ``n_draws`` (see ``_probit_draws``). It reports
    ``n_centres=1`` and adds its name to the
    execution provenance. The default family above is unchanged; the opt-in
    family has not been shown to be more accurate.
    """
    _validate_proposal(proposal)
    from trex.evidence.mc_importance import ESTIMATOR, native_stage

    for value, label, minimum in (
        (n_dimensions, "parameter count", 1), (n_draws, "importance draw count", 2),
        (random_seed, "random seed", 0),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    n_runs = len(posteriors)
    stage_identity = native_stage.get()
    if n_runs == 0:
        raise ValueError("importance refinement needs at least one nested posterior")
    ndim = int(n_dimensions)
    sample_parts, weight_parts = [], []
    for unit_samples, normalized_weights in posteriors:
        unit_samples, normalized_weights = np.asarray(unit_samples), np.asarray(normalized_weights)
        if (unit_samples.dtype.kind not in "iuf" or normalized_weights.dtype.kind not in "iuf"
                or unit_samples.ndim != 2 or unit_samples.shape[1] != ndim
                or normalized_weights.shape != (len(unit_samples),)
                or not np.isfinite(unit_samples).all() or np.any((unit_samples < 0) | (unit_samples > 1))
                or not np.isfinite(normalized_weights).all() or np.any(normalized_weights < 0)
                or not np.isclose(normalized_weights.sum(), 1.0, rtol=1e-12, atol=0.0)):
            raise ValueError("nested posterior requires finite unit-cube samples and normalized non-negative weights")
        sample_parts.append(np.asarray(unit_samples, dtype=float))
        weight_parts.append(np.asarray(normalized_weights, dtype=float) / n_runs)
    samples, weights = np.concatenate(sample_parts), np.concatenate(weight_parts)
    rng = np.random.default_rng(int(random_seed))
    pilots = {}

    accelerated = native_integrand is not None
    native = native_integrand if accelerated else cpu_native_integrand
    runtime = _checkpoint_context.get()
    completed_values = np.empty(0, dtype=float)
    executions = []
    adaptation_checkpoint = None
    if runtime is not None:
        prefix, deadline = runtime
        if prefix is None and stage_identity is None:
            raise RuntimeError("resumable escalation requires a candidate-owned checkpoint directory")
        identity_inputs = (
            "importance", prior_transform, log_likelihood, posteriors, n_dimensions,
            int(random_seed), int(n_draws), _IMPORTANCE_CENTRES, _IMPORTANCE_DEFENSIVE_FRACTION,
            stage_identity, "jax-native" if native is not None else "scalar-cpu",
        )
        # Only a non-default family changes the identity, so existing
        # kernel-mixture checkpoints keep their exact inputs.
        if proposal != _PROPOSAL_KERNEL_MIXTURE:
            identity_inputs += (proposal, "unclipped-ndtr-v1")
        identity = _checkpoint_identity(identity_inputs)
        if prefix is not None:
            completed_values, executions = _load_importance_progress(prefix, identity, n_draws)
        if _monotonic() >= deadline:
            raise NestedCheckpointPaused("nested wall deadline reached before refinement")
        adaptation_checkpoint = prefix, deadline, identity, bool(completed_values.size)
    cpu_device = None
    if native is not None and not accelerated:
        from trex.accelerators import _jax_models as models

        cpu_device = models.jax.devices("cpu")[0]

    def evaluate_accelerator(evaluator, subset):
        """Keep device leases and float64 parity while requiring accelerator execution."""
        from trex.accelerators import _jax_models as models
        from trex.runtime.accelerator_pool import accelerator_execution_scope, record_accelerator_event

        points = evaluator.points
        evaluator.points = points[subset]
        try:
            if not len(evaluator.points):
                return np.empty(0, dtype=float)
            with accelerator_execution_scope(models._usable_jax_devices()) as devices:
                if not devices:
                    raise RuntimeError("no accelerator lease was admitted for MC-pilot importance")
                unmeasured = [device for device in devices if device not in pilots]
                if unmeasured:
                    count = min(len(evaluator.points), models.DEVICE_PILOT_DRAWS, evaluator.block)
                    cpu = models.jax.devices("cpu")[0]
                    reference, cpu_seconds, count = models._timed_pilot(evaluator, cpu, count)
                    for device in unmeasured:
                        values, seconds, used = models._timed_pilot(evaluator, device, count)
                        parity = models._pilot_parity(reference[:used], values)
                        pilots[device] = {"device": str(device), "draws": int(used),
                                          "cpu_seconds": cpu_seconds, "accelerator_seconds": seconds,
                                          "parity": parity}
                        record_accelerator_event("mc-pilot-importance-device-pilot", device,
                                                 **{key: value for key, value in pilots[device].items() if key != "device"})
                admitted = tuple(device for device in devices if pilots[device]["parity"]["passed"])
                if not admitted:
                    raise RuntimeError("no accelerator passed MC-pilot importance float64 parity")
                result = models._run_jax_chunks(
                    len(evaluator.points), len(evaluator.reference.time), evaluator.reference.nsamples,
                    evaluator, 1, chunk_plan={device: len(evaluator.points) for device in admitted},
                    _allocated_devices=admitted,
                )
                if evaluator._integrity_failures:
                    if evaluator._numerical_failure is not None:
                        raise evaluator._numerical_failure
                    raise RuntimeError("native unit numerical/execution integrity failure: "
                                       + "; ".join(evaluator._integrity_failures))
                return result
        finally:
            evaluator.points = points

    def evaluate_native_batch(batch, subset=None, *, leased=True):
        """Ordered native evaluation: leased accelerator, bare native, or batched CPU kernel."""
        if accelerated and leased:
            return evaluate_accelerator(batch, slice(None) if subset is None else subset)
        if accelerated:
            return batch.evaluate() if subset is None else batch.evaluate(subset=subset)
        from trex.accelerators import _jax_models as models

        with models.jax.default_device(cpu_device):
            return batch.evaluate() if subset is None else batch.evaluate(subset=subset)

    adaptation = []
    if stage_identity is not None:
        if native is not None:
            from trex.accelerators._native_unit_batch import NativeUnitCubeBatch

            adaptive_batch = None

            def adapt(points):
                nonlocal adaptive_batch
                if adaptive_batch is None:
                    adaptive_batch = NativeUnitCubeBatch(native, points)
                else:
                    adaptive_batch.points = points
                return evaluate_native_batch(adaptive_batch)
            samples, weights, adaptation = _adapt_proposal(
                samples, weights, adapt, rng, checkpoint=adaptation_checkpoint,
                proposal=proposal,
            )
        else:
            # No native scenario callback: the identical proposal is shaped by
            # the float64 scalar unit-cube integrand the scenario declares.
            # The adaptive passes are the same cost as a quarter of the final
            # draw stage, so they use the existing worker mapping too.
            block = _LogLikelihoodBlock(prior_transform, log_likelihood)
            workers = resolve_n_jobs(n_jobs)
            spool = start_worker_warning_spool() if workers > 1 else None
            try:
                pool = _worker_pool(
                    workers, importance_block=block, warning_report=worker_warning_report(spool))
            except BaseException:
                collect_worker_warnings(spool, 0)
                raise
            successful = False
            try:
                adapt = _pool_evaluator(block, pool, _IMPORTANCE_EVALUATION_CHUNK)
                samples, weights, adaptation = _adapt_proposal(
                    samples, weights, adapt, rng, checkpoint=adaptation_checkpoint,
                    evaluation_chunk=_IMPORTANCE_EVALUATION_CHUNK * workers,
                    proposal=proposal,
                )
                successful = True
            finally:
                if pool is not None:
                    try:
                        if successful:
                            pool.close()
                        else:
                            pool.terminate()
                        pool.join()
                    finally:
                        collect_worker_warnings(spool, workers)
    mixture, draws, inside = _draw_proposal(proposal, samples, weights, rng, n_draws, ndim)

    if completed_values.size > inside.size:
        raise ValueError("importance checkpoint progress is incomplete or invalid")
    completed = completed_values.size
    log_l_parts = [completed_values] if completed else []
    last_saved = _monotonic()

    def pause_if_due():
        nonlocal last_saved
        if runtime is None:
            return
        now = _monotonic()
        if now >= deadline or now - last_saved >= _IMPORTANCE_CHECKPOINT_SECONDS:
            values = np.concatenate(log_l_parts) if log_l_parts else np.empty(0, dtype=float)
            if prefix is not None:
                _save_importance_progress(prefix, identity, values, executions)
            last_saved = _monotonic()
            if now >= deadline:
                raise NestedCheckpointPaused(
                    "nested wall deadline reached during importance refinement; "
                    "{0} of {1} likelihoods retained".format(values.size, inside.size)
                )

    pause_if_due()

    log_q = _proposal_log_density(proposal, mixture, draws[inside])

    if native is not None:
        from trex.accelerators._native_unit_batch import NativeUnitCubeBatch

        if runtime is None:
            evaluate = NativeUnitCubeBatch(native, draws[inside])
            log_l = evaluate_native_batch(evaluate, leased=stage_identity is not None)
            execution = evaluate.provenance()
            if not accelerated:
                execution["backend"] = "jax-native-cpu"
            if stage_identity is not None:
                execution["stage_device_pilots"] = list(pilots.values())
        else:
            # Keep the actual device-admission/pilot/lease scheduler in each
            # recovery wave; never call a bare callback on an unleased device.
            evaluate = NativeUnitCubeBatch(native, draws[inside])
            group = max(_IMPORTANCE_EVALUATION_CHUNK, evaluate.block)
            session_index = len(executions)
            for start in range(completed, inside.size, group):
                end = min(start + group, inside.size)
                part = evaluate_native_batch(evaluate, slice(start, end), leased=stage_identity is not None)
                log_l_parts.append(part)
                record = evaluate.provenance()
                if not accelerated:
                    record["backend"] = "jax-native-cpu"
                if stage_identity is not None:
                    record["stage_device_pilots"] = list(pilots.values())
                record["logical_likelihood_calls"] = end - completed
                if len(executions) == session_index:
                    executions.append(record)
                else:
                    executions[session_index] = record
                pause_if_due()
            log_l = np.concatenate(log_l_parts) if log_l_parts else np.empty(0, dtype=float)
            execution = {"backend": "jax-native-cpu" if not accelerated else "jax-native-unit-cube",
                         "logical_likelihood_calls": int(inside.size),
                         "reduction_backend": "numpy-cpu-original-order", "recovery_waves": executions}
    else:
        evaluate = _LogLikelihoodBlock(prior_transform, log_likelihood)
        workers = resolve_n_jobs(n_jobs)
        # Ship the block callable once per worker through the pool initializer;
        # mapping a module-level cached function keeps the integrand out of the
        # per-task pickle.  Blocks, order, chunking and reductions are
        # unchanged, so the executed draws are bitwise the same.
        warning_spool = start_worker_warning_spool() if workers > 1 else None
        try:
            pool = _worker_pool(
                workers, importance_block=evaluate, warning_report=worker_warning_report(warning_spool)
            )
        except BaseException:
            collect_worker_warnings(warning_spool, 0)  # Nothing ran; only remove the spool.
            raise
        successful = False
        try:
            # One block per allocated CPU per wave keeps parallelism and bounds
            # the unchecked interval without changing draws or reduction order.
            group = _IMPORTANCE_EVALUATION_CHUNK * workers
            for start in range(completed, inside.size, group):
                blocks = [draws[inside[lo:min(lo + _IMPORTANCE_EVALUATION_CHUNK, inside.size)]]
                          for lo in range(start, min(start + group, inside.size), _IMPORTANCE_EVALUATION_CHUNK)]
                if pool is None:
                    parts = list(map(evaluate, blocks))
                else:
                    parts = pool.map(_cached_importance_block, blocks, chunksize=1)
                log_l_parts.extend(np.asarray(part, dtype=float) for part in parts)
                pause_if_due()
            successful = True
        finally:
            if pool is not None:
                try:
                    if successful:
                        pool.close()
                    else:
                        pool.terminate()
                    pool.join()
                finally:
                    collect_worker_warnings(warning_spool, workers)
        log_l = np.concatenate(log_l_parts) if log_l_parts else np.empty(0)
        execution = {"backend": "scalar-cpu", "logical_likelihood_calls": int(inside.size),
                     "reduction_backend": "numpy-cpu-original-order"}
    if adaptation:
        execution["adaptation"] = adaptation
    adaptive_calls = sum(record["likelihood_calls"] for record in adaptation)
    total_calls = int(inside.size) + adaptive_calls
    execution.update(logical_likelihood_calls=total_calls, estimation_likelihood_calls=int(inside.size),
                     adaptation_likelihood_calls=adaptive_calls)
    if proposal != _PROPOSAL_KERNEL_MIXTURE:
        execution["proposal"] = proposal
    invalid = np.isnan(log_l) | np.isposinf(log_l)
    if np.any(invalid):
        raise NumericalLikelihoodError("the importance-refinement likelihood returned NaN or +inf", int(invalid.sum()))
    log_w = log_l - log_q
    finite = np.isfinite(log_w)
    if not np.any(finite):
        raise RuntimeError("no importance draw landed on the likelihood support")
    shift = float(np.max(log_w[finite]))
    w = np.zeros(n_draws)
    w[inside[finite]] = np.exp(log_w[finite] - shift)
    mean_w = float(np.mean(w))
    return ImportanceEvidenceResult(
        log_evidence=shift + math.log(mean_w),
        log_evidence_error=float(np.std(w, ddof=1) / math.sqrt(n_draws) / mean_w),
        n_draws=int(n_draws),
        n_likelihood_calls=total_calls,
        effective_sample_size=float(w.sum() ** 2 / np.sum(w ** 2)),
        maximum_weight_share=float(w.max() / w.sum()),
        n_centres=1 if proposal == _PROPOSAL_PROBIT_GAUSSIAN else _IMPORTANCE_CENTRES,
        defensive_fraction=_IMPORTANCE_DEFENSIVE_FRACTION,
        random_seed=int(random_seed),
        **({"estimator": ESTIMATOR} if stage_identity is not None else {}),
        execution_provenance=execution,
    )
