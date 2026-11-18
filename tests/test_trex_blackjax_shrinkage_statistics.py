"""Independent limiting cases for the native compression simulation."""

import math

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.evidence._blackjax_kernel import _compression_log_evidences


@pytest.mark.parametrize("deleted", [1, 200])
def test_joint_deletion_compression_has_order_statistic_moments(deleted):
    """Zero dead likelihood and unit live likelihood make logZ exactly logX."""
    jax, jnp = models.jax, models.jnp
    counts = np.arange(400, 400 - deleted, -1, dtype=np.int64)
    # For IID uniform constrained volumes, log(t_j) = -Exp(1)/n_j.
    # Thus E(logX) = -sum(1/n_j), Var(logX) = sum(1/n_j**2).
    expected_mean = -np.sum(1.0 / counts)
    expected_sd = math.sqrt(np.sum(1.0 / counts**2))
    values = np.asarray(jax.jit(_compression_log_evidences)(jax.random.PRNGKey(901),
        jnp.full(deleted, -jnp.inf, dtype=jnp.float64), jnp.asarray(counts),
        jnp.zeros(400, dtype=jnp.float64)))
    assert abs(values.mean() - expected_mean) < 4.0 * expected_sd / math.sqrt(len(values))
    assert 0.7 < values.var(ddof=1) / expected_sd**2 < 1.3
