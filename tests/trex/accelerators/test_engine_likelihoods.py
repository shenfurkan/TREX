"""TREX dual-backend likelihood and JAX batch execution tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trex.foundation.constants import SECONDS_PER_DAY
from tests.fixtures.synthetic_observations import SYNTHETIC_SOLAR_METALLICITY_DEX, synthetic_batch_likelihood_case


def synthetic_eb_ldc():
    """Representative cool-secondary coefficients from the retained TESS grid.

    Local to this module so the TREX-only tests do not import an EXONYM fixture.
    """
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    # TEST_DESIGN: select a retained cool-dwarf grid node, not invented LDC.
    u1, u2 = load_triceratops_tess_grid().coefficients(3500, 5, SYNTHETIC_SOLAR_METALLICITY_DEX)
    return {"u1_EB": u1, "u2_EB": u2}


# ============================================================================
# Dual-Backend Monte Carlo Evidence & Likelihood Tests
# ============================================================================

def test_eval_scenario_dual_backend():
    """Verify _eval_scenario executes and achieves parity across JAX and Batman backends."""
    from trex.evidence.marginal_likelihoods import _eval_scenario
    import trex.evidence.marginal_likelihoods as ml_mod

    time = np.linspace(-0.1, 0.1, 100)
    flux = np.ones_like(time)
    flux[45:55] = 0.995
    sigma = 0.001
    period_days = 3.5
    M_s = 1.0
    R_s = 1.0
    u1, u2 = 0.3, 0.2
    exptime_days = 120.0 / SECONDS_PER_DAY

    orig_has_jax = ml_mod.HAS_JAX_TREX
    try:
        ml_mod.HAS_JAX_TREX = False
        rng_batman = np.random.default_rng(12345)
        lnZ_batman = _eval_scenario(
            time, flux, sigma, period_days, M_s, R_s, u1, u2,
            n_draws=100, is_planet=True, is_EB=False, use_2xP=False,
            exptime_days=exptime_days, rng=rng_batman, nsamples=10,
        )

        ml_mod.HAS_JAX_TREX = orig_has_jax
        rng_jax = np.random.default_rng(12345)
        lnZ_jax = _eval_scenario(
            time, flux, sigma, period_days, M_s, R_s, u1, u2,
            n_draws=100, is_planet=True, is_EB=False, use_2xP=False,
            exptime_days=exptime_days, rng=rng_jax, nsamples=10,
        )
    finally:
        ml_mod.HAS_JAX_TREX = orig_has_jax

    assert math.isfinite(lnZ_batman)
    assert math.isfinite(lnZ_jax)
    if orig_has_jax:
        assert math.isclose(lnZ_batman, lnZ_jax, rel_tol=1e-3, abs_tol=1e-3)


def test_eval_scenario_eb_dual_backend():
    """Verify _eval_scenario EB executes and achieves parity across JAX and Batman backends."""
    from trex.evidence.marginal_likelihoods import _eval_scenario
    import trex.evidence.marginal_likelihoods as ml_mod

    time = np.linspace(-0.2, 0.2, 120)
    flux = np.ones_like(time)
    flux[50:70] = 0.95
    sigma = 0.002
    period_days = 4.0
    M_s = 1.0
    R_s = 1.0
    u1, u2 = 0.3, 0.2
    exptime_days = 120.0 / SECONDS_PER_DAY

    orig_has_jax = ml_mod.HAS_JAX_TREX
    try:
        ml_mod.HAS_JAX_TREX = False
        rng_batman = np.random.default_rng(54321)
        lnZ_batman = _eval_scenario(
            time, flux, sigma, period_days, M_s, R_s, u1, u2,
            n_draws=100, is_planet=False, is_EB=True, use_2xP=False,
            exptime_days=exptime_days, rng=rng_batman, nsamples=10,
            target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
        )

        ml_mod.HAS_JAX_TREX = orig_has_jax
        rng_jax = np.random.default_rng(54321)
        lnZ_jax = _eval_scenario(
            time, flux, sigma, period_days, M_s, R_s, u1, u2,
            n_draws=100, is_planet=False, is_EB=True, use_2xP=False,
            exptime_days=exptime_days, rng=rng_jax, nsamples=10,
            target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
        )
    finally:
        ml_mod.HAS_JAX_TREX = orig_has_jax

    assert math.isfinite(lnZ_batman)
    assert math.isfinite(lnZ_jax)
    if orig_has_jax:
        assert math.isclose(lnZ_batman, lnZ_jax, rel_tol=1e-3, abs_tol=1e-3)


@pytest.mark.parametrize("scenario", ("TP", "EB"))
def test_eval_scenario_warns_and_uses_batman_when_the_jax_batch_fails(monkeypatch, scenario):
    """A failing JAX batch must not be swallowed silently.

    The batman reference likelihood evaluates the same prior draws, so the
    evidence must equal a batman-only run exactly, and a RuntimeWarning must
    identify the failed JAX batch.
    """
    import trex.evidence.marginal_likelihoods as ml_mod

    is_eb = scenario == "EB"
    time = np.linspace(-0.2, 0.2, 120)
    flux = np.ones_like(time)
    flux[50:70] = 0.95 if is_eb else 0.995
    kwargs = dict(
        n_draws=100, is_planet=not is_eb, is_EB=is_eb, use_2xP=False,
        exptime_days=120.0 / SECONDS_PER_DAY, nsamples=10,
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )

    monkeypatch.setattr(ml_mod, "HAS_JAX_TREX", False)
    reference = ml_mod._eval_scenario(
        time, flux, 0.002, 4.0, 1.0, 1.0, 0.3, 0.2,
        rng=np.random.default_rng(2468), **kwargs,
    )

    attempts = []

    def failing_batch(*_args, **_kwargs):
        attempts.append(_kwargs)
        raise RuntimeError("synthetic JAX batch failure")

    monkeypatch.setattr(ml_mod, "HAS_JAX_TREX", True)
    monkeypatch.setattr(ml_mod, "batch_lnL_EB" if is_eb else "batch_lnL_TP", failing_batch)
    with pytest.warns(RuntimeWarning, match="JAX batched"):
        fallback = ml_mod._eval_scenario(
            time, flux, 0.002, 4.0, 1.0, 1.0, 0.3, 0.2,
            rng=np.random.default_rng(2468), **kwargs,
        )

    assert math.isfinite(reference)
    assert fallback == reference
    assert len(attempts) == 2
    assert attempts[0].get("chunk_size_divisor", 1) == 1
    assert attempts[1]["chunk_size_divisor"] == 2


def test_eval_scenario_records_the_jax_batch_fallback(monkeypatch):
    """The batman fallback is recorded as provenance, not only warned about.

    A runtime warning is gone once the run ends: a report read afterwards would
    imply the JAX batch path evaluated draws it did not. The same event must
    travel as a structured record naming the failed likelihood so the run's
    accelerator provenance cannot be reconstructed from a transient warning.
    """
    import trex.evidence.marginal_likelihoods as ml_mod

    time = np.linspace(-0.1, 0.1, 100)
    flux = np.ones_like(time)
    flux[45:55] = 0.995
    kwargs = dict(
        n_draws=100, is_planet=True, is_EB=False, use_2xP=False,
        exptime_days=120.0 / SECONDS_PER_DAY, nsamples=10,
    )

    def failing_batch(*_args, **_kwargs):
        raise RuntimeError("synthetic JAX batch failure")

    monkeypatch.setattr(ml_mod, "HAS_JAX_TREX", True)
    monkeypatch.setattr(ml_mod, "batch_lnL_TP", failing_batch)
    fallbacks: list = []
    with pytest.warns(RuntimeWarning, match="JAX batched"):
        lnZ = ml_mod._eval_scenario(
            time, flux, 0.001, 3.5, 1.0, 1.0, 0.3, 0.2,
            rng=np.random.default_rng(1357), jax_fallback_out=fallbacks,
            **kwargs,
        )

    assert math.isfinite(lnZ)
    assert len(fallbacks) == 1
    record = fallbacks[0]
    assert record["kind"] == "TP"
    assert record["draws"] > 0
    assert record["exception_type"] == "RuntimeError"
    assert record["message"] == "synthetic JAX batch failure"


@pytest.mark.parametrize("scenario", ("TP", "EB"))
def test_eval_scenario_retries_transient_memory_failure_before_batman(monkeypatch, scenario):
    """Exercise the real JAX batch and evidence reducer after a failed probe."""
    import warnings

    import trex.runtime.hardware as hardware
    import trex.evidence.marginal_likelihoods as ml_mod

    if not ml_mod.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    time, flux, params = synthetic_batch_likelihood_case()
    is_eb = scenario == "EB"
    # TEST_DESIGN: use the existing synthetic geometry and one draw per time.
    kwargs = dict(
        n_draws=time.size, is_planet=not is_eb, is_EB=is_eb, use_2xP=False,
        exptime_days=params["exptime_days"], nsamples=params["nsamples"],
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )
    args = (time, flux, params["exptime_days"], params["P_orb"],
            params["R_s_solar"], params["R_s_solar"], params["u1"], params["u2"])
    expected = ml_mod._eval_scenario(*args, rng=np.random.default_rng(time.size), **kwargs)
    original_probe = hardware.memory_resource_profile
    probes = []

    def transient_failure():
        probes.append(None)
        if len(probes) == 1:
            raise MemoryError("synthetic transient memory pressure")
        return original_probe()

    monkeypatch.delenv("TREX_JAX_BATCH_MEMORY_BYTES", raising=False)
    monkeypatch.setattr(hardware, "memory_resource_profile", transient_failure)
    fallbacks = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        actual = ml_mod._eval_scenario(
            *args, rng=np.random.default_rng(time.size), jax_fallback_out=fallbacks, **kwargs
        )
    assert len(probes) > 1
    assert not [warning for warning in caught if "JAX batched" in str(warning.message)]
    assert fallbacks == []
    np.testing.assert_allclose(actual, expected, rtol=np.finfo(float).eps * time.size)


@pytest.mark.parametrize("scenario", ("TP", "EB"))
def test_retry_cannot_grow_chunks_when_memory_pressure_lifts(monkeypatch, scenario):
    import trex.runtime.hardware as hardware
    import trex.evidence.marginal_likelihoods as ml_mod
    from trex.accelerators import _jax_models as models
    from trex.evidence import mc_scenario

    if not models.HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed")
    time, flux, params = synthetic_batch_likelihood_case()
    device = models.jax.devices("cpu")[0]
    monkeypatch.setattr(models, "JAX_USABLE_DEVICES", (device,))
    monkeypatch.delenv(models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False)
    # One full-grid batch, so the memory arithmetic below stays about that grid.
    monkeypatch.setattr(mc_scenario, "_SCREEN_POINT_FRACTION", 1.0)
    monkeypatch.setattr(mc_scenario, "_screened_lnL", lambda bound, *_: bound)
    # TEST_DESIGN: two-draw initial headroom and evaluation block, headroom
    # doubled after a transfer failure; the halved block must not grow back.
    initial_chunk = 2
    from trex.accelerators import _jax_batch

    monkeypatch.setattr(_jax_batch, "_BLOCK_ELEMENTS", initial_chunk * time.size * params["nsamples"])
    per_draw = time.size * np.dtype(float).itemsize * models._JAX_BATCH_MEMORY_OVERHEAD
    probes = []

    def available_memory(_device):
        probes.append(_device)
        factor = 1 if len(probes) == 1 else initial_chunk
        return factor * initial_chunk * per_draw * models._JAX_BATCH_MEMORY_DEVICE_DIVISOR

    monkeypatch.setattr(hardware, "device_memory_available_bytes", available_memory)
    original_asarray = models.jnp.asarray
    transfers = []

    def upload(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size != time.size:
            transfers.append(value.size)
            if len(transfers) == 1:
                raise MemoryError("synthetic device transfer failure")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(models.jnp, "asarray", upload)
    fallbacks = []
    evidence = ml_mod._eval_scenario(
        time, flux, params["exptime_days"], params["P_orb"],
        params["R_s_solar"], params["R_s_solar"], params["u1"], params["u2"],
        n_draws=time.size, is_planet=scenario == "TP", is_EB=scenario == "EB",
        use_2xP=False, exptime_days=params["exptime_days"], nsamples=params["nsamples"],
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
        rng=np.random.default_rng(time.size), jax_fallback_out=fallbacks,
    )
    assert math.isfinite(evidence)
    assert fallbacks == []
    assert transfers[0] == initial_chunk
    assert transfers[1:]
    assert max(transfers[1:]) <= initial_chunk // 2
    assert set(probes) == {device}


def test_jax_models_batch_likelihood_parity():
    """Verify batch_lnL_TP and batch_lnL_EB match lnL_TP and lnL_EB within 1e-4."""
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_TP, batch_lnL_EB
    from trex.models.likelihoods import lnL_TP, lnL_EB
    from trex.foundation.constants import au

    if not HAS_JAX_TREX:
        pytest.skip("JAX not available in current environment")

    time = np.linspace(-0.1, 0.1, 80)
    flux = np.ones_like(time)
    flux[35:45] = 0.99
    sigma = 0.001
    exptime_days = 120.0 / SECONDS_PER_DAY

    # 3 TP draws
    rp_arr = np.array([2.0, 2.5, 3.0])
    inc_arr = np.array([88.0, 89.0, 87.5])
    a_cm_arr = np.array([0.04, 0.05, 0.06]) * au
    ecc_arr = np.array([0.0, 0.1, 0.2])
    w_arr = np.array([90.0, 45.0, 120.0])

    batch_tp = batch_lnL_TP(
        time, flux, sigma, rp_arr, 3.5, inc_arr, a_cm_arr,
        1.0, 0.3, 0.2, exptime_days, ecc_arr, w_arr, nsamples=10,
    )
    for k in range(len(rp_arr)):
        single_tp = lnL_TP(
            time, flux, sigma, rp_arr[k], 3.5, inc_arr[k], a_cm_arr[k],
            1.0, 0.3, 0.2, exptime_days, ecc=ecc_arr[k], argp_deg=w_arr[k],
            nsamples=10,
        )
        assert math.isclose(batch_tp[k], single_tp, rel_tol=1e-4, abs_tol=1e-4)

    # 3 EB draws
    reb_arr = np.array([0.4, 0.5, 0.6])
    feb_arr = np.array([0.1, 0.15, 0.2])
    batch_eb = batch_lnL_EB(
        time, flux, sigma, reb_arr, feb_arr, 4.0, inc_arr, a_cm_arr,
        1.0, 0.3, 0.2, exptime_days, ecc_arr, w_arr, nsamples=10,
        **synthetic_eb_ldc(),
    )
    for k in range(len(reb_arr)):
        single_eb = lnL_EB(
            time, flux, sigma, reb_arr[k], feb_arr[k], 4.0, inc_arr[k], a_cm_arr[k],
            1.0, 0.3, 0.2, exptime_days, ecc=ecc_arr[k], argp_deg=w_arr[k],
            nsamples=10,
            **synthetic_eb_ldc(),
        )
        assert math.isclose(batch_eb[k], single_eb, rel_tol=1e-4, abs_tol=1e-4)


def test_eccentric_transit_condition_follows_winn_2010_equation_7():
    """The eccentricity correction scales the impact parameter, not the threshold.

    Winn (2010), *Transits and Occultations*, ADS ``2010exop.book...55W``,
    eq. 7: ``b_tra = (a cos i / R*) * (1 - e**2) / (1 + e sin w)``. Applying the
    factor to the ``1 + rp_rs`` side instead inverts the correction; the two
    agree only at zero eccentricity, which is why this pins a non-zero one.
    """
    from trex.evidence.marginal_likelihoods import _geometric_transit_mask

    ecc = np.array([0.4, 0.4])
    argp = np.array([90.0, 270.0])  # periastron and apastron conjunction
    a_rs = np.array([10.0, 10.0])
    rp_rs = np.array([0.1, 0.1])
    factor = (1.0 - ecc ** 2) / (1.0 + ecc * np.sin(np.radians(argp)))

    # A draw placed exactly on the Winn boundary must be admitted; one just
    # outside it must not.
    b_boundary = (1.0 + rp_rs) / factor
    assert np.all(_geometric_transit_mask(b_boundary * 0.999, rp_rs, ecc, argp, a_rs))
    assert not np.any(_geometric_transit_mask(b_boundary * 1.001, rp_rs, ecc, argp, a_rs))

    # The inverted form would have used (1 + rp_rs) * factor; for e = 0.4 the two
    # boundaries differ by factor**2, so this is a real discrimination.
    assert not np.allclose(b_boundary, (1.0 + rp_rs) * factor)

    # At zero eccentricity both conventions coincide.
    circular = np.zeros(2)
    assert np.all(
        _geometric_transit_mask(np.array([1.05, 1.05]), rp_rs, circular, argp, a_rs)
    )
    assert not np.any(
        _geometric_transit_mask(np.array([1.2, 1.2]), rp_rs, circular, argp, a_rs)
    )


def test_a_colliding_eccentric_orbit_has_no_transit_support():
    """Bodies that would overlap at periastron are not a physical draw.

    A Keplerian orbit's closest separation is ``a (1 - e)``; below the summed
    radii of host and occultor the bodies overlap. Upstream TRICERATOPS 1.0.20
    excludes exactly those draws in every eccentric scenario
    (``coll = (R_occ + R_s) > a (1 - e)``), computed here in physical units as
    the independent reference. Every draw is a central transit, so only the
    collision condition can reject it.
    """
    from trex.evidence.marginal_likelihoods import _geometric_transit_mask

    R_s, R_occ, a = 1.0, 0.1, 5.0
    ecc = np.array([0.5, 0.77, 0.79, 0.9])
    upstream_collision = (R_occ + R_s) > a * (1.0 - ecc)
    assert upstream_collision.tolist() == [False, False, True, True]

    mask = _geometric_transit_mask(
        np.zeros(4), np.full(4, R_occ / R_s), ecc, np.full(4, 90.0), np.full(4, a / R_s)
    )
    assert np.array_equal(mask, ~upstream_collision)


def test_jax_phase_bin_integration_matches_the_batman_reference():
    """The JAX batch path must reproduce batman when a phase-bin box applies.

    Production always passes a non-zero ``bin_integration_days`` (the phase-bin
    width), so before this was supported the accelerator was unreachable on
    every real candidate and every draw went through the one-at-a-time batman
    loop. The two paths use different limb-darkening implementations, so they
    agree to their own numerical parity, not to the last bit.
    """
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_EB, batch_lnL_TP
    from trex.models.likelihoods import lnL_EB, lnL_TP

    if not HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed in this environment")

    rng = np.random.default_rng(3)
    time = np.linspace(-0.12, 0.12, 241)
    sigma = np.full(time.size, 2e-4)
    exptime, bin_days, period = 0.000231, 0.0009, 3.836
    a_cm = np.full(3, 1.4e12)
    r_host = np.full(3, 0.83)
    u1 = np.full(3, 0.4)
    u2 = np.full(3, 0.2)
    ecc = np.zeros(3)
    argp = np.full(3, 90.0)
    cfr = np.zeros(3)

    flux = 1.0 - 0.004 * np.exp(-0.5 * (time / 0.02) ** 4) + rng.normal(0, 2e-4, time.size)
    rp = np.array([6.0, 7.5, 9.0])
    jax_tp = batch_lnL_TP(
        time, flux, sigma, rp, period, np.array([89.2, 88.6, 89.8]), a_cm, r_host,
        u1, u2, exptime, ecc_arr=ecc, w_arr=argp, companion_fluxratio=cfr,
        nsamples=20, bin_integration_days=bin_days,
    )
    batman_tp = np.array([
        lnL_TP(
            time, flux, sigma, float(rp[i]), period, float([89.2, 88.6, 89.8][i]),
            float(a_cm[i]), float(r_host[i]), float(u1[i]), float(u2[i]), exptime,
            ecc=0.0, argp_deg=90.0, companion_fluxratio=0.0, nsamples=20,
            bin_integration_days=bin_days,
        )
        for i in range(3)
    ])
    assert np.max(np.abs((jax_tp - batman_tp) / batman_tp)) < 1e-5

    eb_flux = 1.0 - 0.02 * np.exp(-0.5 * (time / 0.02) ** 4) + rng.normal(0, 3e-4, time.size)
    eb_sigma = np.full(time.size, 3e-4)
    r_eb = np.array([0.15, 0.22, 0.30])
    f_eb = np.array([0.01, 0.05, 0.12])
    jax_eb = batch_lnL_EB(
        time, eb_flux, eb_sigma, r_eb, f_eb, period, np.array([89.2, 88.6, 89.8]),
        a_cm, r_host, u1, u2, exptime, ecc_arr=ecc, w_arr=argp,
        companion_fluxratio=cfr, nsamples=20, bin_integration_days=bin_days,
        u1_EB=u1, u2_EB=u2,
    )
    batman_eb = np.array([
        lnL_EB(
            time, eb_flux, eb_sigma, float(r_eb[i]), float(f_eb[i]), period,
            float([89.2, 88.6, 89.8][i]), float(a_cm[i]), float(r_host[i]),
            float(u1[i]), float(u2[i]), exptime, ecc=0.0, argp_deg=90.0,
            companion_fluxratio=0.0, nsamples=20, bin_integration_days=bin_days,
            u1_EB=float(u1[i]), u2_EB=float(u2[i]),
        )
        for i in range(3)
    ])
    assert np.max(np.abs((jax_eb - batman_eb) / batman_eb)) < 1e-5

    # A zero bin width must leave the unbinned behaviour untouched.
    unbinned = batch_lnL_TP(
        time, flux, sigma, rp, period, np.array([89.2, 88.6, 89.8]), a_cm, r_host,
        u1, u2, exptime, ecc_arr=ecc, w_arr=argp, companion_fluxratio=cfr,
        nsamples=20, bin_integration_days=0.0,
    )
    assert not np.allclose(unbinned, jax_tp)


def test_two_box_quadrature_matches_a_converged_batman_convolution():
    """The trapezoid-kernel quadrature must reproduce the batman convolution limit.

    fixscientific.md F12.6: the nested ``nsamples x nsamples`` supersampling is
    replaced by the exact two-box trapezoid kernel integrated at
    Gauss-Legendre nodes -- ten model evaluations per bin instead of four
    hundred. That is admissible only if it still computes the same integral,
    so the test drives the nested batman reference to convergence in the
    supersampling count (n vs 2n) and then requires the quadrature to agree
    with the finer reference within that measured convergence floor.
    """
    pytest.importorskip("batman")
    import batman

    from trex.foundation._numerics import _two_box_convolution_quadrature
    from trex.models.likelihoods import _batman_transit

    period = 3.83616118
    a_rs = 20.0
    u1, u2 = 0.4, 0.2
    exptime = 20.0 / SECONDS_PER_DAY
    time = (np.arange(281) - 140.0) * exptime

    def make_params(rp_rs, inc_deg, ecc, argp):
        params = batman.TransitParams()
        params.t0 = 0.0
        params.per = period
        params.rp = rp_rs
        params.a = a_rs
        params.inc = inc_deg
        params.ecc = ecc
        params.w = argp
        params.limb_dark = "quadratic"
        params.u = [u1, u2]
        return params

    def nested_reference(params, n):
        offsets = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * (exptime / n)
        eval_times = (time[:, None] + offsets[None, :]).reshape(-1)
        model = batman.TransitModel(
            params, eval_times, supersample_factor=n, exp_time=exptime
        )
        flux = np.asarray(model.light_curve(params), dtype=float)
        return flux.reshape(time.size, n).mean(axis=1)

    def inclination_for(b):
        return 90.0 - np.degrees(np.arcsin(b / a_rs))

    for label, params in (
        ("b=0.4", make_params(0.0688, inclination_for(0.4), 0.0, 90.0)),
        ("b=0.98", make_params(0.0688, inclination_for(0.98), 0.0, 90.0)),
        ("e=0.3 w=60", make_params(0.0688, 89.7, 0.3, 60.0)),
    ):
        coarse = nested_reference(params, 128)
        reference = nested_reference(params, 256)
        reference_gap = float(np.max(np.abs(reference - coarse)))
        model = _batman_transit(
            time, period, params.rp, a_rs, params.inc, u1, u2, exptime,
            ecc=params.ecc, argp_deg=params.w, bin_integration_days=exptime,
        )
        error = float(np.max(np.abs(model - reference)))
        # The batman gold standard must be matched to better than one part in
        # 1e7 of the normalised flux, and within a few times the reference's
        # own 128-to-256 convergence gap.
        assert error < 1e-7, (label, error)
        assert error <= 5.0 * reference_gap + 1e-8, (label, error, reference_gap)

    offsets, weights = _two_box_convolution_quadrature(exptime, exptime)
    assert offsets.size <= 15
    # Photon conservation: the discrete kernel is a probability distribution
    # and must sum to one at machine precision (four ulp at most).
    assert abs(float(np.sum(weights)) - 1.0) <= 4.0 * np.finfo(float).eps

    # Analytic reference: for a known smooth profile the weighted node sum must
    # equal the exact trapezoid-kernel convolution computed by fine-grid
    # quadrature. This checks the kernel itself, independent of batman.
    def gaussian(t):
        return 1.0 - 0.0047 * np.exp(-0.5 * (t / 0.03) ** 2)

    half_span = 0.5 * (exptime + exptime)
    tau = np.linspace(-half_span, half_span, 200001)
    kernel = np.where(
        np.abs(tau) <= 0.0,
        1.0 / exptime,
        (half_span - np.abs(tau)) / exptime ** 2,
    )
    kernel = kernel / np.trapz(kernel, tau)
    sample_times = np.linspace(-0.02, 0.02, 21)
    analytic = np.array(
        [np.trapz(kernel * gaussian(t + tau), tau) for t in sample_times]
    )
    node_sum = np.array(
        [np.sum(weights * gaussian(t + offsets)) for t in sample_times]
    )
    assert float(np.max(np.abs(node_sum - analytic))) < 1e-12

    # A zero bin width must leave only the exposure box: the mean of the
    # instantaneous model at 20 sub-exposure midpoints (PyTransit 2.2's rule),
    # not batman's equal-weight endpoint grid.
    params = make_params(0.0688, 89.7, 0.3, 60.0)
    midpoints = exptime * ((np.arange(20) + 0.5) / 20 - 0.5)
    midpoint_times = (time[:, None] + midpoints[None, :]).reshape(-1)
    instantaneous = batman.TransitModel(params, midpoint_times)
    expected = np.asarray(instantaneous.light_curve(params), dtype=float)
    np.testing.assert_allclose(
        _batman_transit(
            time, period, 0.0688, a_rs, 89.7, u1, u2, exptime,
            ecc=0.3, argp_deg=60.0, nsamples=20, bin_integration_days=0.0,
        ),
        expected.reshape(time.size, 20).mean(axis=1),
        rtol=0.0,
        atol=1e-14,
    )


def test_jax_batch_chunking_does_not_change_the_result(monkeypatch):
    """Chunked and unchunked JAX batches must agree to float64 rounding.

    ``jax.vmap`` materializes every draw at once, so an unchunked batch of a
    production-sized draw count over a phase-binned curve asked XLA for hundreds
    of gigabytes and died, silently falling back to the batman loop. Chunking is
    a memory control only: the chunks are independent draws of the same
    function, so their concatenation reproduces the single-batch answer. The
    compiled kernels are not bit-identical -- a chunk boundary changes what XLA
    fuses -- so the contract is agreement to float64 rounding (measured here at
    a maximum relative deviation of 2.2e-14), tight enough that any indexing or
    weighting defect, whose error would be O(1), still fails.
    """
    from trex.accelerators import _jax_models
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_TP

    if not HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed in this environment")

    rng = np.random.default_rng(11)
    time = np.linspace(-0.1, 0.1, 61)
    flux = 1.0 - 0.004 * np.exp(-0.5 * (time / 0.02) ** 4) + rng.normal(0, 2e-4, time.size)
    sigma = np.full(time.size, 2e-4)
    n = 37
    kwargs = dict(
        p_orb=3.836, exptime_days=0.000231, nsamples=6, bin_integration_days=0.0009,
    )
    rp = rng.uniform(4.0, 9.0, n)
    inc = rng.uniform(88.5, 90.0, n)
    a_cm = np.full(n, 1.4e12)
    r_host = np.full(n, 0.83)
    u1 = np.full(n, 0.4)
    u2 = np.full(n, 0.2)
    ecc = np.zeros(n)
    argp = np.full(n, 90.0)
    cfr = np.zeros(n)

    def run():
        return batch_lnL_TP(
            time, flux, sigma, rp, kwargs["p_orb"], inc, a_cm, r_host, u1, u2,
            kwargs["exptime_days"], ecc_arr=ecc, w_arr=argp,
            companion_fluxratio=cfr, nsamples=kwargs["nsamples"],
            bin_integration_days=kwargs["bin_integration_days"],
        )

    # Drive the budget through the operator override rather than the module
    # constant: the resolved budget also scales with the device's reported
    # memory, so only the override pins the chunk size on every host.
    budget_variable = _jax_models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE
    monkeypatch.setenv(budget_variable, str(1 << 40))  # one batch
    single = run()
    # Force a chunk size well below n so several chunks are concatenated.
    monkeypatch.setenv(budget_variable, "1")
    chunked = run()

    assert _jax_models._jax_chunk_size(time.size * kwargs["nsamples"], kwargs["nsamples"]) >= 1
    assert single.shape == chunked.shape == (n,)
    np.testing.assert_allclose(single, chunked, rtol=1e-12, atol=1e-9)


def test_jax_batch_budget_scales_with_the_reported_device_memory(monkeypatch):
    """Larger devices scale; smaller devices must not inherit an unsafe floor.

    Contract change: the old lower bound could exceed measured headroom. An
    explicit operator override still wins, but derived budgets can shrink.
    """
    from trex.accelerators import _jax_models

    divisor = _jax_models._JAX_BATCH_MEMORY_DEVICE_DIVISOR
    # Isolate the OS probe too: actual available RAM changes between calls and
    # is now a real ceiling, not something the budget is allowed to ignore.
    from pathlib import Path

    large = 15 * 1024**3  # synthetic allocator-capacity fixture, not a real device
    host_bytes = large * divisor
    floor = large // divisor**2  # target-neutral counter fixture, not a production memory floor
    original_read = Path.read_text

    def read_memory(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemTotal: {host_bytes // 1024} kB\nMemAvailable: {host_bytes // 1024} kB\n"
        if str(path) == "/proc/self/cgroup":
            return ""
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_memory)
    monkeypatch.delenv(
        _jax_models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, raising=False
    )

    def report(value):
        monkeypatch.setattr(
            _jax_models, "_jax_batch_device_memory_bytes", lambda: value
        )

    def budget():
        with _jax_models.jax.default_device(_jax_models.jax.devices("cpu")[0]):
            return _jax_models._jax_batch_memory_budget_bytes()

    def chunk_size():
        with _jax_models.jax.default_device(_jax_models.jax.devices("cpu")[0]):
            return _jax_models._jax_chunk_size(4096, 8)

    # Contract change: real remaining host RAM, not a fixed 1-GiB cap, sizes
    # CPU batches when the backend itself supplies no memory counter.
    report(None)
    assert budget() == host_bytes // divisor

    # A small device must shrink the budget instead of overcommitting memory.
    report(floor)
    assert budget() == floor // divisor

    # A larger device scales, and so does the chunk it buys.
    report(large)
    assert budget() == large // divisor
    assert budget() > floor

    report(floor)
    small_chunk = chunk_size()
    report(large)
    assert chunk_size() > small_chunk


def test_jax_batch_budget_override_rejects_a_value_it_cannot_use(monkeypatch):
    """A misread override must fail, not silently serialize or exhaust a batch."""
    from trex.accelerators import _jax_models

    variable = _jax_models.JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE
    monkeypatch.setenv(variable, "8589934592")
    assert _jax_models._jax_batch_memory_budget_bytes() == 8589934592

    for rejected in ("not-an-integer", "0", "-1"):
        monkeypatch.setenv(variable, rejected)
        with pytest.raises(ValueError):
            _jax_models._jax_batch_memory_budget_bytes()


def test_jax_and_batman_agree_below_the_batman_solver_limit():
    """The two likelihood paths must agree where batman still solves the orbit.

    Batman stops returning a finite forward model for most arguments of
    periastron at e >= 0.98, so it can no longer be the settle reference for the
    near-unity tail (the router now sends every admitted draw to the device
    solver). Below the 0.98 limit this pins the two independent implementations
    together: 1e-4 relative where both convolved paths were measured to agree
    within 3.7e-6, and 1e-3 at 0.95/0.97 where batman's own eccentric-anomaly
    error dominates (measured 3.0e-5 and 2.6e-4). Independent accuracy of the
    device solver above 0.97 is established in test_jax_kepler_domain.py.
    """
    from trex.accelerators._jax_models import HAS_JAX_TREX, batch_lnL_TP
    from trex.models.likelihoods import lnL_TP

    if not HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet is not installed in this environment")

    rng = np.random.default_rng(7)
    time = np.linspace(-0.12, 0.12, 241)
    sigma = np.full(time.size, 2e-4)
    flux = 1.0 - 0.004 * np.exp(-0.5 * (time / 0.02) ** 4) + rng.normal(0, 2e-4, time.size)
    bin_days, exptime, period = 0.0009, 0.000231, 3.836

    for ecc, tolerance in ((0.0, 1e-4), (0.5, 1e-4), (0.8, 1e-4), (0.93, 1e-4),
                           (0.95, 1e-3), (0.97, 1e-3)):
        for argp in (0.0, 90.0, 180.0, 270.0):
            jax_value = batch_lnL_TP(
                time, flux, sigma, np.array([7.0]), period, np.array([89.2]),
                np.array([1.4e12]), np.array([0.83]), np.array([0.4]), np.array([0.2]),
                exptime, ecc_arr=np.array([ecc]), w_arr=np.array([argp]),
                companion_fluxratio=np.array([0.0]), nsamples=20,
                bin_integration_days=bin_days,
            )[0]
            reference = lnL_TP(
                time, flux, sigma, 7.0, period, 89.2, 1.4e12, 0.83, 0.4, 0.2, exptime,
                ecc=ecc, argp_deg=argp, companion_fluxratio=0.0, nsamples=20,
                bin_integration_days=bin_days,
            )
            assert math.isfinite(reference), (ecc, argp)
            relative = abs((jax_value - reference) / reference)
            assert relative < tolerance, (ecc, argp, relative, tolerance)


def test_every_admitted_eccentricity_is_routed_to_the_device_solver():
    """The router admits the whole prior support, not the batman-tested range.

    The replaced bound sent draws this side of one to batman, where most of the
    e >= 0.98 tail has no finite forward model; the device solver's accuracy
    over that range is covered by test_jax_kepler_domain.py.
    """
    from trex.evidence.marginal_likelihoods import _JAX_MAX_SUPPORTED_ECCENTRICITY

    assert _JAX_MAX_SUPPORTED_ECCENTRICITY == np.nextafter(1.0, 0.0)


def test_wide_eccentric_eb_scenario_resolves_the_prior_tail():
    """A detached near-unity-eccentricity tail must produce an evidence.

    A 100-day effective period around a solar-mass host puts about two percent
    of the Moe & Di Stefano (2017) eccentricity prior above e = 0.98; with
    a/R_s ~ 90, the low-mass companions there remain detached and reach the
    likelihood. Batman has no finite forward model there; before the router
    admitted this range the scenario ended as an unresolved numerical failure
    instead of an evidence.
    """
    from trex.evidence.marginal_likelihoods import _eval_scenario

    time = np.linspace(-0.25, 0.25, 41)
    flux = 1.0 - 0.15 * (np.abs(time) < 0.12)
    sigma = 1e-3
    fallbacks = []
    lnZ = _eval_scenario(
        time, flux, sigma, 50.0, 1.0, 1.0, 0.3, 0.2,
        n_draws=6000, is_planet=False, is_EB=True, use_2xP=True,
        exptime_days=30.0 / SECONDS_PER_DAY, rng=np.random.default_rng(20261114),
        nsamples=3, target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX, jax_fallback_out=fallbacks,
    )
    assert math.isfinite(lnZ)
    assert fallbacks == []


def test_the_wide_binary_prior_reaches_the_range_batman_cannot_solve():
    """Why the scenario above is a discriminator, stated independently of it.

    Uses the same prior call the scenario uses (``planet=False``, effective
    P = 100 d); the draw count is illustrative. The unsolvable tail is a
    property of the adopted eccentricity law, not an edge case added here.
    """
    from trex.populations.priors import sample_ecc

    ecc = sample_ecc(np.random.default_rng(11).random(200_000), planet=False, P_orb=100.0)
    assert np.count_nonzero(ecc > 0.98) > 1_000
