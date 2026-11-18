"""The eclipsed secondary's atmosphere must shape its own EB occultation."""

import inspect
from fractions import Fraction

import numpy as np
import pytest

from trex.models.likelihoods import lnL_EB, simulate_EB
from tests.fixtures.synthetic_observations import synthetic_secondary_ldc_case

# Batman and jaxoplanet are independent quadrature implementations; 5 ppm forward-model parity bound.
PARITY_TOLERANCE = 5.0e-6


@pytest.mark.parametrize("companion_is_host", [False, True])
def test_binary_likelihood_uses_each_eclipsed_bodys_grid_ldc(companion_is_host):
    time, expected, parameters = synthetic_secondary_ldc_case(companion_is_host=companion_is_host)
    _, coarse, _ = synthetic_secondary_ldc_case(companion_is_host=companion_is_host, radial_nodes=64)
    # Independent quadrature refinement bounds its numerical integration error.
    tolerance = 4 * np.max(np.abs(expected - coarse)) + np.sqrt(np.finfo(float).eps)
    assert (parameters["u1"], parameters["u2"]) != (parameters["u1_EB"], parameters["u2_EB"])
    for function in (simulate_EB, lnL_EB):
        assert {"u1_EB", "u2_EB"} <= inspect.signature(function).parameters.keys(), (
            "EB likelihood must accept the eclipsed secondary's own atmosphere"
        )
    actual, phase = simulate_EB(time, **parameters)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    assert phase == parameters["P_orb"] / (2 * parameters["P_orb"])
    corrected = lnL_EB(time, expected, tolerance, **parameters)
    legacy_parameters = {
        **parameters, "u1_EB": parameters["u1"], "u2_EB": parameters["u2"],
    }
    legacy = lnL_EB(time, expected, tolerance, **legacy_parameters)
    legacy_model, _ = simulate_EB(time, **legacy_parameters)
    assert corrected > legacy
    split = time.size // 2
    np.testing.assert_array_equal(actual[:split], legacy_model[:split])
    assert np.max(np.abs(actual[split:] - legacy_model[split:])) > tolerance


@pytest.mark.parametrize("companion_is_host", [False, True])
@pytest.mark.parametrize("phase_binned", [False, True])
def test_jax_batch_keeps_per_draw_secondary_ldc_through_chunking(monkeypatch, companion_is_host, phase_binned):
    from trex.accelerators._jax_models import HAS_JAX_TREX, JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, _JAX_BATCH_MEMORY_OVERHEAD, batch_lnL_EB

    if not HAS_JAX_TREX:
        pytest.skip("JAX/jaxoplanet unavailable")

    time, expected, parameters = synthetic_secondary_ldc_case(companion_is_host=companion_is_host)
    parameters["bin_integration_days"] = parameters["exptime_days"] if phase_binned else float(Fraction(0, 1))
    # TEST_DESIGN: fixed photometric uncertainty, not a relaxed parity tolerance.
    sigma = float(Fraction(1, 1000))
    secondary_u1 = np.array([parameters["u1_EB"], parameters["u1"]])
    secondary_u2 = np.array([parameters["u2_EB"], parameters["u2"]])
    kwargs = dict(
        time=time, flux_obs=expected, sigma=sigma,
        reb_arr=np.full(secondary_u1.size, parameters["R_EB_solar"]),
        feb_arr=np.full(secondary_u1.size, parameters["EB_fluxratio"]),
        p_orb=parameters["P_orb"],
        inc_arr=np.full(secondary_u1.size, parameters["inc_deg"]),
        a_cm_arr=parameters["a_cm"], R_host=parameters["R_s_solar"],
        u1=parameters["u1"], u2=parameters["u2"],
        exptime_days=parameters["exptime_days"],
        ecc_arr=np.zeros(secondary_u1.size),
        w_arr=np.full(secondary_u1.size, np.degrees(np.pi / 2)),
        companion_fluxratio=parameters["companion_fluxratio"],
        companion_is_host=companion_is_host, nsamples=parameters["nsamples"],
        u1_EB=secondary_u1, u2_EB=secondary_u2,
        bin_integration_days=parameters["bin_integration_days"],
    )
    batched = batch_lnL_EB(**kwargs)
    # Real memory-budget boundary forces one draw per chunk; no model mock.
    one_draw_bytes = time.size * parameters["nsamples"] * np.dtype(float).itemsize * _JAX_BATCH_MEMORY_OVERHEAD
    monkeypatch.setenv(JAX_BATCH_MEMORY_ENVIRONMENT_VARIABLE, str(one_draw_bytes))
    chunked = batch_lnL_EB(**kwargs)
    np.testing.assert_array_equal(chunked, batched)
    assert batched[0] > batched[1]
    models = np.array([
        simulate_EB(time, **{
            **parameters, "u1_EB": eb_u1, "u2_EB": eb_u2,
        })[0]
        for eb_u1, eb_u2 in zip(secondary_u1, secondary_u2)
    ])
    reference = -np.sum((expected - models)**2, axis=1) / (2 * sigma**2)
    # Compare odds at the declared photometric precision. Each backend must
    # prefer the independently injected atmosphere, without identical LDC.
    assert reference[0] > reference[1]
    # Batman and jaxoplanet are different quadrature implementations, not
    # bitwise twins. Propagate the existing 5-ppm forward-model parity
    # contract into lnL, rather than assuming machine-epsilon agreement.
    error_bound = (
        PARITY_TOLERANCE * np.sum(np.abs(expected - models), axis=1)
        + time.size * PARITY_TOLERANCE**2 / 2
    ) / sigma**2
    assert np.all(np.abs(batched - reference) <= error_bound)


@pytest.mark.parametrize("missing", ["u1_EB", "u2_EB"])
def test_missing_secondary_ldc_is_not_replaced_by_primary_ldc(missing):
    time, expected, parameters = synthetic_secondary_ldc_case()
    del parameters[missing]
    with pytest.raises(TypeError, match=missing):
        simulate_EB(time, **parameters)
    with pytest.raises(TypeError, match=missing):
        lnL_EB(time, expected, np.sqrt(np.finfo(float).eps), **parameters)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_secondary_ldc_is_refused(invalid):
    time, expected, parameters = synthetic_secondary_ldc_case()
    parameters["u1_EB"] = invalid
    with pytest.raises(ValueError, match="secondary.*finite"):
        lnL_EB(time, expected, np.sqrt(np.finfo(float).eps), **parameters)


@pytest.mark.parametrize("metallicity", [None, np.nan, np.inf, True])
def test_nested_binary_evidence_requires_retained_metallicity(metallicity):
    from trex.evidence.nested_evidence import eb_scenario_nested_evidence

    time, expected, parameters = synthetic_secondary_ldc_case()
    with pytest.raises(ValueError, match="target_feh"):
        eb_scenario_nested_evidence(
            time=time, flux=expected, sigma=np.sqrt(np.finfo(float).eps),
            period_days=parameters["P_orb"], M_s_Msun=float(Fraction(1, 1)),
            R_s_Rsun=parameters["R_s_solar"], u1=parameters["u1"], u2=parameters["u2"],
            exptime_days=parameters["exptime_days"],
            target_aperture_flux_fraction=float(Fraction(1, 1)),
            target_feh=metallicity, random_seed=time.size,
        )
