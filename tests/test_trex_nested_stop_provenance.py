"""The nested runner must certify convergence from dynesty's own base-run record.

``DynamicNestedSampler.combine_runs`` rebuilds ``saved_run`` after every batch,
interleaving base and batch samples by likelihood and recomputing ``logvol`` and
``logz`` for the merged run, so the base run's stopping quantity can no longer be
read from it.

dynesty tests ``delta_logz < dlogz`` at the top of each base iteration and stops
without yielding, so the last value it yields is the one that did *not* stop it.
Two properties follow that do not depend on how the value is recovered:
the recovered stop value lies below the tolerance while dynesty's last yielded
base value does not (the crossing), and adding dynamic batches to an identical
seeded run must not change the recovered base-run value.
"""

import math

import numpy as np
import pytest
from scipy import integrate

dynesty = pytest.importorskip("dynesty")

from trex.evidence.nested_evidence import (  # noqa: E402 - dynesty is optional
    _DEFAULT_EVIDENCE_TOLERANCE,
    _SamplerControls,
    _base_run_remaining_evidence,
    _run_nested_sampler,
)

_N_DIMENSIONS = 3
_N_LIVE = 60
_WIDTH = 0.05
_SEED = 11


def _log_likelihood(x):
    return -0.5 * float(np.sum(((np.asarray(x) - 0.5) / _WIDTH) ** 2))


def _prior_transform(u):
    return u


def _sampler():
    return dynesty.DynamicNestedSampler(
        _log_likelihood, _prior_transform, _N_DIMENSIONS,
        sample="rslice", rstate=np.random.default_rng(_SEED),
    )


def _base_only_sampler():
    """A seeded sampler whose base run is complete and no batch was added.

    Returns the sampler and the last ``delta_logz`` dynesty yielded in its base
    run (the value that did not satisfy the stopping rule).
    """
    sampler = _sampler()
    in_base = dynesty.dynamicsampler.DynamicSamplerStatesEnum.INBASE
    last_yielded = None
    for result in sampler.sample_initial(nlive=_N_LIVE, dlogz=_DEFAULT_EVIDENCE_TOLERANCE):
        if sampler.internal_state == in_base:
            last_yielded = result.delta_logz
    return sampler, last_yielded


def test_base_run_stop_value_survives_dynamic_batches():
    base_only, last_yielded = _base_only_sampler()
    with_batches = _sampler()
    with_batches.run_nested(
        nlive_init=_N_LIVE, dlogz_init=_DEFAULT_EVIDENCE_TOLERANCE, print_progress=False
    )

    before = _base_run_remaining_evidence(base_only)
    after = _base_run_remaining_evidence(with_batches)

    assert with_batches.batch > 0, "the regression needs a run that added dynamic batches"
    # The crossing: dynesty continued at its last yielded value and stopped at the next.
    assert last_yielded >= _DEFAULT_EVIDENCE_TOLERANCE > before
    # Batches never touch the base run, so its stop value must not move.
    assert after == pytest.approx(before, rel=np.sqrt(np.finfo(float).eps))


_PLATEAU_RADIUS = 0.3


def _plateau_log_likelihood(x):
    radius = float(np.hypot(x[0] - 0.5, x[1] - 0.5))
    return 0.0 if radius < _PLATEAU_RADIUS else -0.5 * ((radius - _PLATEAU_RADIUS) / _WIDTH) ** 2


def test_a_likelihood_plateau_stop_is_not_labelled_an_unresolved_integration():
    # dynesty 3.0.0 (sampler.py:1096-1100) ends the base run, with a warning,
    # once every live point shares one likelihood. At a flat maximum the
    # remaining mass is that constant times the remaining volume, so the
    # integral is complete up to volume noise. Truth: independent quadrature.
    truth, _ = integrate.dblquad(lambda y, x: math.exp(_plateau_log_likelihood((x, y))),
                                 0.0, 1.0, 0.0, 1.0, epsabs=1e-12, epsrel=1e-10)
    controls = _SamplerControls(n_live_points=200, evidence_tolerance=_DEFAULT_EVIDENCE_TOLERANCE,
                                max_likelihood_calls=None, random_seed=_SEED, n_jobs=1)
    with pytest.warns(UserWarning, match="plateau"):
        result = _run_nested_sampler(_prior_transform, _plateau_log_likelihood, controls, "plateau", n_dimensions=2)
    assert abs(result.log_evidence - math.log(truth)) < 3.0 * result.log_evidence_error
    assert result.stop_reason != "unresolved", "a complete plateau integration must not defer as unresolved"
