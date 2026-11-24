"""Production TREX JAX EB batch parity and throughput on 5000 draws.

The measured path is the real memory-bounded ``batch_lnL_EB`` executor, not a
test-local vmap, and every returned log-likelihood is compared with the
independent CPU Batman reference from ``trex.models.likelihoods``. Batman and
jaxoplanet are separate quadrature implementations, so the 5 ppm forward-model
tolerance is propagated into an exact lnL error bound rather than assuming
machine-epsilon agreement.

Units: time, period and exposure in days; semimajor axis in cm; radii in
R_earth / R_sun; angles in degrees; flux normalized to unity; float64
throughout (jax_enable_x64 is set by ``trex.accelerators._jax_models``).
"""

from __future__ import annotations

import time as pytime
from fractions import Fraction

import numpy as np
import pytest

from trex.accelerators import _jax_models as models
from trex.accelerators._jax_models import batch_lnL_EB
from trex.foundation.constants import SECONDS_PER_DAY, au
from trex.models.likelihoods import simulate_EB

pytestmark = pytest.mark.skipif(
    not models.HAS_JAX_TREX,
    reason="JAX/jaxoplanet unavailable",
)

# Batman and jaxoplanet are independent quadrature implementations; 5 ppm forward-model parity bound.
PARITY_TOLERANCE = 5.0e-6


def test_jax_batch_eb_benchmark_5000():
    """All 5000 EB draws use the actual memory-bounded production batch.

    The previous test bypassed the executor with an unbounded test-local vmap,
    requested almost 4 GiB in one allocation, and checked just its first draw.
    Keep the draw count and strengthen parity to every returned likelihood.
    """
    n_draws = 5000
    time = np.linspace(-0.25, 4.25, 300)
    exptime_days = 120.0 / SECONDS_PER_DAY

    rng = np.random.default_rng(84)
    reb_draws = rng.uniform(0.2, 0.8, size=n_draws)
    feb_draws = rng.uniform(0.05, 0.40, size=n_draws)
    # One scenario has one observed period, matching the production interface.
    period = float(Fraction(5, 1))
    p_draws = np.full(n_draws, period)
    inc_draws = rng.uniform(85.0, 89.5, size=n_draws)
    a_draws = rng.uniform(0.04, 0.12, size=n_draws) * au
    rs_draws = rng.uniform(0.8, 1.3, size=n_draws)
    ecc_draws = rng.uniform(0.0, 0.3, size=n_draws)
    w_draws = rng.uniform(0.0, 360.0, size=n_draws)
    u1, u2 = 0.3, 0.2  # Same atmosphere assumed by this legacy reference.

    # 1. Benchmark Batman sequential (computes both primary and secondary eclipse)
    t_start_batman = pytime.perf_counter()
    batman_results = [
        simulate_EB(
            time, reb_draws[i], feb_draws[i], p_draws[i], inc_draws[i],
            a_draws[i], rs_draws[i], u1, u2, exptime_days,
            ecc=ecc_draws[i], argp_deg=w_draws[i], nsamples=20,
            u1_EB=u1, u2_EB=u2,
        )[0]
        for i in range(n_draws)
    ]
    t_batman = pytime.perf_counter() - t_start_batman

    # TEST_DESIGN: normalized OOT flux and a relative-flux uncertainty.
    observed = np.ones_like(time)
    sigma = float(Fraction(1, 1000))
    parameters = dict(
        time=time, flux_obs=observed, sigma=sigma, reb_arr=reb_draws,
        feb_arr=feb_draws, p_orb=period, inc_arr=inc_draws, a_cm_arr=a_draws,
        R_host=rs_draws, u1=u1, u2=u2, exptime_days=exptime_days,
        ecc_arr=ecc_draws, w_arr=w_draws, nsamples=20, u1_EB=u1, u2_EB=u2,
    )
    batch_lnL_EB(**parameters)  # real compiled-shape warmup, already synchronized

    # Benchmark JAX steady-state GPU execution
    t_start_jax = pytime.perf_counter()
    jax_results = batch_lnL_EB(**parameters)
    t_jax = pytime.perf_counter() - t_start_jax

    speedup = t_batman / max(t_jax, 1.0e-6)

    residual = observed - np.asarray(batman_results)
    reference = -np.sum(residual**2, axis=1) / (2 * sigma**2)
    # Propagate the unchanged 5-ppm flux-model tolerance into lnL exactly.
    error_bound = (PARITY_TOLERANCE * np.sum(np.abs(residual), axis=1)
                   + time.size * PARITY_TOLERANCE**2 / 2) / sigma**2
    assert jax_results.shape == (n_draws,)
    assert np.all(np.abs(jax_results - reference) <= error_bound)

    print(f"\n[BENCHMARK EB] 5000 Draws | Batman CPU: {t_batman:.4f}s | TREX JAX batch: {t_jax:.4f}s | Speedup: {speedup:.1f}x")
