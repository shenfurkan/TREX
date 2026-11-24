"""MC-pilot importance and draw-stage parity; synthetic inputs only."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy import stats
from scipy.special import ndtr, ndtri

from trex.accelerators import _jax_models as models
from trex.evidence import evidence_escalation as escalation
from trex.evidence import evidence_escalation_run as escalation_run
from trex.stellar import stellar_relations as trex_stellar
from trex.evidence import marginal_likelihoods as ml
from trex.evidence import mc_scenario
from trex.populations import mass_ratio_priors
from trex.evidence import mc_importance as stage
from trex.evidence import nested_controls as controls
from trex.evidence import nested_evidence as nested
from trex.evidence import nested_importance as importance
from trex.populations import priors
from trex.evidence.mc_occultors import _distinct_rows, draw_eb_occultor, draw_planet_occultor

CELL = 2.0**-ml._QUANTILE_BISECTION_ITERATIONS


def _bisected_quantile(sampler, values):
    """The fixed-depth bisection the closed-form windows replaced."""
    lo, hi = np.zeros(values.size), np.ones(values.size)
    for _ in range(ml._QUANTILE_BISECTION_ITERATIONS):
        mid = 0.5*(lo+hi)
        below = np.asarray(sampler(mid)) < values
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5*(lo+hi)


def _unit_points(dimensions=4, value=0.5):
    return np.full((controls._IMPORTANCE_CENTRES, dimensions), value).tolist()


def test_centres_invert_the_inclination_band_and_sort_the_background(monkeypatch):
    monkeypatch.setattr(controls, "_IMPORTANCE_PILOT_ESS", 1.0)
    logw = np.array([-np.inf, -50.0, 0.0, -50.0])
    inclination = np.array([0.1, 0.2, 0.5, 0.9])
    band_low = np.full(4, 0.25)
    occultor, ecc, argument = (np.array([0.1, 0.2, 0.3, 0.4]) * scale for scale in (1.0, 2.0, 0.5))
    magnitudes = np.array([12.0, 10.0, 10.0, 11.0])
    rows = np.array([3, 0, 2, 1])
    centres = stage.make_centres(logw, occultor, ecc, argument, inclination, band_low,
                                 [(rows, np.array([0.1, 0.2, 0.7, 0.4]))], {"tess_mag": magnitudes},
                                 seed=5, scenario=("BTP", True, False))
    # Only draw 2 carries weight; row 2 is rank 1 of the stable sort [1, 2, 3, 0].
    np.testing.assert_allclose(centres, np.tile([0.3, 0.6, 0.15, (0.5-0.25)/0.75, 1.5/4, 0.7],
                                                (controls._IMPORTANCE_CENTRES, 1)))


def test_handoff_refuses_a_changed_seed_centre_or_version():
    state = stage.make_state(7, {"0": _unit_points()}, "input")
    assert stage.validate_state(state, "input", 15) == state
    with pytest.raises(ValueError):
        stage.validate_state(state, "other-input", 15)
    with pytest.raises(ValueError):
        stage.validate_state({**state, "seed": 8}, "input", 15)
    with pytest.raises(ValueError):
        stage.validate_state({**state, "centres": {"0": _unit_points(value=0.4)}}, "input", 15)
    with pytest.raises(ValueError):
        stage.validate_state({**state, "controls": {**state["controls"], "version": "old"}}, "input", 15)


def test_bound_importance_handoff_puts_the_primary_quantile_after_companion_coordinates():
    centres = stage.make_centres(
        np.array([0.]), np.array([.1]), np.array([.2]), np.array([.3]),
        np.array([.625]), np.array([.25]),
        [(np.array([.6]), np.array([.7]), np.array([.8]))], None,
        seed=5, scenario="PTP", host_quantile=np.array([.9]),
    )
    np.testing.assert_array_equal(
        centres, np.tile([.1, .2, .3, .5, .6, .7, .8, .9], (controls._IMPORTANCE_CENTRES, 1)))
    state = stage.make_state(7, {"3": centres}, "input")
    distribution = {"method": "synthetic joint target grid"}
    assert stage.validate_state(state, "input", 15, 1., target_distribution=distribution) == state
    with pytest.raises(ValueError, match="requires 7 unit dimensions"):
        stage.validate_state(state, "input", 15, 1.)


@pytest.mark.parametrize("mass", [0.3, 1.0])
def test_bound_handoff_requires_projection_coordinate_even_if_resigned(mass):
    # Bound periods and mass ratios cannot determine a sky projection.
    old = stage.make_state(7, {"3": _unit_points(dimensions=6)}, "input")
    with pytest.raises(ValueError, match="requires 7 unit dimensions"):
        stage.validate_state(old, "input", 15, target_mass=mass)
    new = stage.make_state(7, {"3": _unit_points(dimensions=7)}, "input")
    assert stage.validate_state(new, "input", 15, target_mass=mass) == new


def test_a_missed_mode_passes_ess_and_error_but_not_the_first_stage():
    width = 0.001
    modes = np.array([[0.25, 0.25], [0.75, 0.75]])

    def likelihood(point):
        # Two equal normalized Gaussians, so the true evidence is 1 and ln Z is 0.
        terms = -0.5*np.sum(((point-modes)/width)**2, axis=1)-np.log(2*np.pi*width**2)-np.log(2.0)
        return float(np.logaddexp.reduce(terms))

    rng = np.random.default_rng(4)
    pilot = rng.normal(modes[0], width, (500, 2))
    refined = nested._run_importance_sampler(lambda x: x, likelihood, [(pilot, np.full(500, 1/500))],
                                             n_dimensions=2, random_seed=3, n_jobs=1, n_draws=20_000)
    assert refined.log_evidence == pytest.approx(np.log(0.5), abs=0.1)
    assert refined.effective_sample_size >= 100 and refined.log_evidence_error <= 0.1
    assert escalation._mc_importance_refusal(refined, refined.log_evidence, np.log(0.5), 1000.0) is None
    assert "sits below" in escalation._mc_importance_refusal(refined, refined.log_evidence, 0.0, 1000.0)
    # Sub-floor Monte Carlo errs low: a stage above it found mass, it did not miss a mode.
    assert escalation._mc_importance_refusal(refined, refined.log_evidence, -5.0, 1000.0) is None


def _refined(log_evidence, error):
    return SimpleNamespace(log_evidence=log_evidence, log_evidence_error=error, effective_sample_size=500.0,
                           n_draws=controls._IMPORTANCE_DRAWS, n_likelihood_calls=10, n_centres=2000,
                           defensive_fraction=0.1, maximum_weight_share=0.01, execution_provenance={})


def test_stage_follows_probability_and_failures_keep_their_monte_carlo(monkeypatch, tmp_path):
    # An EB slot 60 nats down cannot gate or flip, but its estimate may be 42 nats low;
    # one 150 nats down is outside the margin.
    lnz = np.full(15, -np.inf)
    lnz[[0, 3, 1, 2]] = -1.0, -2.0, -60.0, -150.0
    ess = {0: 50.0, 3: 10.0, 1: 1.0, 2: 1.0}
    state = stage.make_state(7, {str(key): _unit_points() for key in ess}, "input")
    calls = []

    def nested_call(plan, **_kwargs):
        calls.append(plan["index"])
        return _refined(-1.1, 0.05) if plan["index"] == 0 else _refined(-2.0, 0.5)

    monkeypatch.setattr(models, "_usable_jax_devices", lambda: [object()])
    monkeypatch.setattr(escalation_run, "_population_blocks", lambda **_k: {})
    monkeypatch.setattr(escalation_run, "_target_plan_entry", lambda index: {"index": index, "log_prior": 0.0})
    monkeypatch.setattr(escalation_run, "_nested_call", nested_call)
    names = ("M_s_Msun", "R_s_Rsun", "plx", "contrast_separations", "contrast_values", "resolution_limit_arcsec",
             "background_population", "background_area_sq_deg", "target_tess_mag", "time", "flux", "sigma",
             "period_days", "u1", "u2", "exptime_days", "bin_integration_days", "target_aperture_flux_fraction",
             "target_feh", "nsamples")
    outcome = stage.run_mc_importance(lnZ=lnz, ess=ess, state=state, n_draws=1_000_000, input_identity="input",
                                      checkpoint_directory=tmp_path / "stage", **dict.fromkeys(names))
    assert calls == [0, 3, 1]
    assert outcome["attempts"]["TP"]["status"] == "accepted" and list(outcome["records"]) == ["TP"]
    assert outcome["attempts"][escalation._scenario_name(3)]["status"] == "unresolved"
    assert outcome["lnZ"][0] == -1.1 and outcome["ess"][0] == 500.0
    assert outcome["lnZ"][3] == -2.0 and outcome["ess"][3] == 10.0


def test_stage_orders_eb_and_2p_by_weighted_probability(monkeypatch, tmp_path):
    lnz = np.full(15, -np.inf)
    lnz[[0, 1, 2]] = 0.0, -15.55, -15.70
    ess = {0: 1000.0, 1: 1.0, 2: 1.0}
    state = stage.make_state(7, {str(key): _unit_points() for key in (1, 2)}, "input")
    calls = []
    monkeypatch.setattr(models, "_usable_jax_devices", lambda: [object()])
    monkeypatch.setattr(escalation_run, "_population_blocks", lambda **_kwargs: {})
    monkeypatch.setattr(escalation_run, "_target_plan_entry", lambda index: {"index": index, "log_prior": 0.0})
    monkeypatch.setattr(escalation_run, "_nested_call", lambda plan, **_kwargs: calls.append(plan["index"]) or _refined(-20, 0.5))
    names = ("M_s_Msun", "R_s_Rsun", "plx", "contrast_separations", "contrast_values", "resolution_limit_arcsec",
             "background_population", "background_area_sq_deg", "target_tess_mag", "time", "flux", "sigma",
             "u1", "u2", "exptime_days", "bin_integration_days", "target_aperture_flux_fraction", "target_feh", "nsamples")
    outcome = stage.run_mc_importance(
        lnZ=lnz, ess=ess, state=state, n_draws=controls._IMPORTANCE_DRAWS, input_identity="input",
        checkpoint_directory=tmp_path / "stage", period_days=1.0, **dict.fromkeys(names),
    )
    # G21 §2.4.1's EB prior at 2 days outweighs the slightly larger raw 1-day EB evidence.
    assert calls == [2, 1]
    np.testing.assert_array_equal(outcome["lnZ"], lnz)


def _radius_ratio(host_mass, host_radius):
    inverse = priors.host_mass_ratio_inverse_cdf(host_mass)

    def ratio(variates):
        secondary = inverse(variates)*host_mass
        value = np.zeros_like(secondary)
        inside = (secondary >= ml.STELLAR_RELATIONS_MASS_MIN_SOLAR) & (secondary <= ml.STELLAR_RELATIONS_MASS_MAX_SOLAR)
        value[inside] = trex_stellar.stellar_radii_in_domain(secondary[inside])/host_radius[inside]
        return value

    return ratio


@pytest.mark.parametrize("hosts", [(0.8,), (0.8, 0.5)], ids=["fixed-host", "two-hosts"])
def test_window_bounds_match_plain_bisection_for_every_draw(monkeypatch, hosts):
    rng = np.random.default_rng(9)
    n = 600
    mass = rng.choice(np.array(hosts), n)
    radius = mass**0.9
    # Repeats exercise the grouping; extremes leave the tabulated range on both sides.
    low = np.round(rng.uniform(-0.1, 1.2, n), 2)
    high = low+np.round(rng.uniform(0.0, 0.5, n), 2)
    captured = []

    class Stop(Exception):
        pass

    def window(_rng, _n, x_low, x_high, **_kwargs):
        captured.append((x_low, x_high))
        raise Stop

    monkeypatch.setattr(mc_scenario, "_defensive_window_draws", window)
    with pytest.raises(Stop):
        draw_eb_occultor(mass, radius, None, False, None, None, None, "EB", np.zeros(n), n, 0.5*(low+high), high, low,
                         rng, None, None, False)
    ratio = _radius_ratio(mass, radius)
    expected = [_bisected_quantile(ratio, values) for values in (low, high)]
    for found, reference in zip(captured[0], expected):
        assert np.max(np.abs(found - reference)) <= CELL


def test_window_inverses_undo_their_samplers():
    rng = np.random.default_rng(11)
    mass = rng.uniform(0.11, 3.0, 20000)
    u = rng.random(mass.size)
    np.testing.assert_allclose(priors.radius_quantile(priors.sample_rp(u, mass), mass), u, atol=1e-11)
    q = ml.host_mass_ratio_inverse_cdf(mass)
    np.testing.assert_allclose(q.cdf(q(u)), u, atol=1e-11)


def _q_branch_integral(host_mass, use_2xP, moment, q_interval=None):
    """Independent quadrature of the adopted G21 §2.4.2 broken-power-law density."""
    floor = max(priors.MOE2017_Q_BREAK_LOW, priors.MOE2017_Q_BREAK_LOW / host_mass)
    middle, twin = priors.MOE2017_Q_BREAK_MID, priors.MOE2017_Q_TWIN_THRESHOLD
    p1, p2 = priors.MOE2017_Q_SLOPE_LOW, priors.MOE2017_Q_SLOPE_HIGH
    continuity = middle ** (p1 - p2) if floor < middle else 1.0
    steep = quad(lambda q: q**p2, max(floor, middle), 1.0)[0]
    twin_base = quad(lambda q: q**p2, twin, 1.0)[0]
    boost = 1.0 + priors.MOE2017_TWIN_FRACTION / (1.0 - priors.MOE2017_TWIN_FRACTION) * steep / twin_base

    def density(q):
        return q**p1 if q < middle else continuity * q**p2 * (boost if q >= twin else 1.0)

    normalizer = quad(density, floor, 1.0, points=[q for q in (middle, twin) if floor < q < 1.0],
                      epsabs=1e-12, epsrel=1e-12)[0]
    low, high = (max(floor, twin), 1.0) if use_2xP else (floor, twin)
    if q_interval is not None:
        low, high = max(low, q_interval[0]), min(high, q_interval[1])
    if low >= high:
        return 0.0
    return quad(lambda q: density(q) * q**moment, low, high,
                points=[middle] if low < middle < high else None, epsabs=1e-12, epsrel=1e-12)[0] / normalizer


@pytest.mark.parametrize("use_2xP", [False, True])
@pytest.mark.parametrize("window", [None, (0.01, 0.02), (0.2, 0.8), (0.9, 1.1)],
                         ids=["no-depth", "below-stellar-floor", "ordinary-window", "twin-window"])
def test_eb_branch_proposal_preserves_prior_mass_and_moment(use_2xP, window, record_property, monkeypatch):
    """Real draw stage: both branches retain the full-q prior, including empty EB support."""
    n = 4096
    hosts = np.array([0.105, 0.2, 1.0])
    mass = np.repeat(hosts, n)
    radius, temperature = trex_stellar.stellar_relations(mass)
    low, high = (None, None) if window is None else (np.full(mass.size, value) for value in window)
    quantiles, support, proposals = [], [], []
    original = ml._defensive_window_draws

    def capture(rng, size, x_low, x_high, **broad):
        proposals.append((x_low, x_high, broad))
        return original(rng, size, x_low, x_high, **broad)

    monkeypatch.setattr(mc_scenario, "_defensive_window_draws", capture)
    drawn = draw_eb_occultor(
        mass, radius, np.zeros(mass.size), False, temperature, [], [], "EB",
        np.zeros(mass.size), mass.size, None if window is None else 0.5 * (low + high),
        high, low, np.random.default_rng(318), support, 0.0, use_2xP, quantile_out=quantiles,
    )
    supported, secondary_mass, log_weight = drawn[0], drawn[8], drawn[9]
    q = secondary_mass / mass
    assert np.all(priors.eb_mass_ratio_support(q[supported], use_2xP))
    # Pilots must retain coordinates in the full prior, not in a conditional branch.
    reproduced = priors.sample_q_host_mass_ratio(quantiles[0][supported], mass[supported])
    np.testing.assert_allclose(q[supported], reproduced, rtol=2e-15, atol=0.0)
    weights = np.where(supported, np.exp(log_weight), 0.0)
    for index, host in enumerate(hosts):
        group = slice(index * n, (index + 1) * n)
        branch_mass = _q_branch_integral(host, use_2xP, 0)
        for moment in (0, 2):
            values = weights[group] * q[group]**moment
            expected = _q_branch_integral(host, use_2xP, moment)
            prior_second = _q_branch_integral(host, use_2xP, 2 * moment)
            second = branch_mass * prior_second
            if window is not None and branch_mass > 0.0:
                x_low, x_high, broad = proposals[0]
                lower, upper = (broad[key][group.start] for key in ("broad_low", "broad_high"))
                lo, hi = np.clip([x_low[group.start], x_high[group.start]], lower, upper)
                if hi > lo:
                    q_interval = priors.sample_q_host_mass_ratio(np.array([lo, hi]), np.full(2, host))
                    narrow_second = _q_branch_integral(host, use_2xP, 2 * moment, q_interval)
                    broad_density = ml._DEFENSIVE_PRIOR_FRACTION / (upper - lower)
                    narrow_density = broad_density + (1.0 - ml._DEFENSIVE_PRIOR_FRACTION) / (hi - lo)
                    second = narrow_second / narrow_density + (prior_second - narrow_second) / broad_density
            # Quadrature gives E_q[(p*g/q)^2]; sampled variance misses rare dyadic-endpoint tails.
            tolerance = 6.0 * np.sqrt(max(second - expected**2, 0.0) / n) + 2e-12
            assert values.mean() == pytest.approx(expected, abs=tolerance)
        expected_nonempty = branch_mass > 0.0
        assert np.all(supported[group] == expected_nonempty)
    if window is not None:
        assert support[0]["draws_considered"] == n * sum(
            _q_branch_integral(host, use_2xP, 0) > 0.0 for host in hosts)
    else:
        assert support == [{"state": "not-requested", "draws_considered": 0}]
    original_q = priors.sample_q_host_mass_ratio(np.random.default_rng(318).random(mass.size), mass)
    record_property("full_prior_branch_fraction", float(np.mean(priors.eb_mass_ratio_support(original_q, use_2xP))))
    record_property("conditional_branch_fraction", float(np.mean(supported)))


def test_stellar_mass_inverse_undoes_the_relation_where_it_rises():
    masses = np.linspace(0.11, 30.0, 40001)
    # The relation falls by 0.11% on 0.622-0.630 solar masses, where a radius has three masses.
    masses = masses[(masses < 0.6219) | (masses > 0.6301)]
    np.testing.assert_allclose(ml.stellar_masses_of_radii(ml.stellar_radii_in_domain(masses)), masses, rtol=1e-6)


def test_planet_window_bounds_match_plain_bisection_for_every_draw(monkeypatch):
    rng = np.random.default_rng(10)
    n = 600
    mass = rng.choice(np.array([0.3, 0.8]), n)
    radius = mass**0.9
    low = np.round(rng.uniform(0.0, 0.2, n), 3)
    high = low+np.round(rng.uniform(0.0, 0.2, n), 3)
    captured = []

    class Stop(Exception):
        pass

    def window(_rng, _n, x_low, x_high):
        captured.append((x_low, x_high))
        raise Stop

    monkeypatch.setattr(mc_scenario, "_defensive_window_draws", window)
    with pytest.raises(Stop):
        draw_planet_occultor(mass, radius, n, 0.5*(low+high), high, low, rng, None, False)
    scale = radius*ml.Rsun/ml.Rearth
    for found, values in zip(captured[0], (low, high)):
        expected = _bisected_quantile(lambda x: priors.sample_rp(x, mass), values*scale)
        assert np.max(np.abs(found - expected)) <= CELL


@pytest.mark.parametrize("columns", [
    (np.full(50, 1.0), np.arange(50.0) % 7), (np.arange(50.0) % 3, np.arange(50.0) % 4),
    (np.full(50, 2.0), np.full(50, 3.0)), (np.arange(50.0) % 3, np.where(np.arange(50) % 5 == 0, np.nan, 1.0)),
    (np.where(np.arange(50) % 2 == 0, -0.0, 0.0), np.arange(50.0) % 2)])
def test_distinct_rows_match_numpy_unique(columns):
    rows, inverse = _distinct_rows(*columns)
    expected, expected_inverse = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(inverse.ravel(), expected_inverse.ravel())


def test_radius_polynomials_match_the_retained_splines():
    edges = np.concatenate((trex_stellar._Mass_nodes_cdwrf, trex_stellar._Mass_nodes_Torres))
    masses = np.sort(np.concatenate((np.linspace(edges.min(), edges.max(), 20_001), edges)))
    masses = masses[(masses >= trex_stellar.STELLAR_RELATIONS_MASS_MIN_SOLAR) & (masses <= trex_stellar.STELLAR_RELATIONS_MASS_MAX_SOLAR)]
    hot = masses > trex_stellar.STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR
    expected = np.where(hot, trex_stellar._Rad_spline_Torres(masses), trex_stellar._Rad_spline_cdwrf(masses))
    np.testing.assert_allclose(trex_stellar.stellar_radii_in_domain(masses), expected, rtol=4e-15, atol=4e-15)


def test_power_law_inverse_inverts_its_integral_with_and_without_zero_exponents():
    q = np.linspace(0.2, 0.95, 40)
    for exponent in (np.full(40, -1.3), np.where(np.arange(40) % 2 == 0, 0.0, 0.8)):
        integral = mass_ratio_priors._power_law_integral(0.1, q, exponent)
        np.testing.assert_allclose(mass_ratio_priors._power_law_inverse(0.1, integral, exponent), q, rtol=2e-13)


def test_planet_eccentricity_inverse_matches_the_beta_distribution():
    x = np.concatenate(([0.0], np.linspace(1e-9, 1-1e-9, 2001)))
    expected = stats.beta.ppf(x, priors.KIPPING_2013_ECCENTRICITY_BETA_A, priors.KIPPING_2013_ECCENTRICITY_BETA_B)
    np.testing.assert_allclose(priors.sample_ecc(x, True, 5.0), expected, rtol=2e-12)
    # The last range lies below the interpolation table, where a wrapped index once returned e near 1.
    tails = np.concatenate((2.0**-np.arange(1.0, 54.0), 1.0-2.0**-np.arange(1.0, 54.0), 10.0**-np.arange(17.0, 261.0)))
    reference = stats.beta.ppf(tails, priors.KIPPING_2013_ECCENTRICITY_BETA_A, priors.KIPPING_2013_ECCENTRICITY_BETA_B)
    np.testing.assert_allclose(priors.sample_ecc(tails, True, 5.0), reference, rtol=2e-12)


def test_handoff_identity_is_equal_across_hosts_and_bound_to_inputs(tmp_path):
    """A resume on another machine holds the same files, not the same path or SIMD build."""
    from trex.app.engine import _input_identity

    def scene(path, digest="a"*64):
        return SimpleNamespace(trilegal_cache=path, background_sha256=digest, M_s_Msun=0.45,
                               contrast_provenance=None)

    radii = np.array([0.5, 0.7])
    inputs = dict(flux=np.ones(3), background_population={"radius_solar": radii})
    reference = _input_identity(scene(tmp_path/"a.csv"), inputs, 1.0)
    # A resume must tolerate ulp differences in parsed radii.
    other_host = dict(inputs, background_population={"radius_solar": np.nextafter(radii, 1.0)})
    assert _input_identity(scene(tmp_path/"b.csv"), other_host, 1.0) == reference
    assert _input_identity(scene(tmp_path/"a.csv", "b"*64), inputs, 1.0) != reference
    assert _input_identity(scene(tmp_path/"a.csv"), dict(inputs, flux=np.full(3, 2.0)), 1.0) != reference


@pytest.mark.skipif(not models.HAS_JAX_TREX, reason="JAX is unavailable")
def test_device_eccentricity_matches_the_retained_beta_inverse():
    from trex.evidence._jax_unit_integrand import _inverse_planet_eccentricity

    x = np.concatenate(([0.0], 2.0**-np.arange(1.0, 54.0), 1.0-2.0**-np.arange(1.0, 54.0),
                        np.random.default_rng(12).random(5000)))
    found = np.asarray(models.jax.vmap(_inverse_planet_eccentricity)(x))
    np.testing.assert_allclose(found, priors.sample_ecc(x, True, 5.0), rtol=1e-13, atol=1e-300)
    # Below the table e = exp(L) with |L| up to 660, which scales double rounding to about 1.5e-13 each side.
    tail = 10.0**-np.arange(17.0, 261.0)
    found = np.asarray(models.jax.vmap(_inverse_planet_eccentricity)(tail))
    np.testing.assert_allclose(found, priors.sample_ecc(tail, True, 5.0), rtol=5e-13, atol=1e-300)
    assert np.isnan(np.asarray(models.jax.vmap(_inverse_planet_eccentricity)(np.array([-0.1, 1.0])))).all()


def test_tempering_exponent_reaches_the_target_or_stays_at_one():
    log_weights = np.array([0.0, -30.0, -30.0, -30.0, -30.0, -30.0])
    assert stage.tempering_exponent(np.zeros(6), 3.0) == 1.0
    power = stage.tempering_exponent(log_weights, 3.0)
    weights = np.exp(power*log_weights)
    assert 0.0 < power < 1.0 and weights.sum()**2/np.sum(weights**2) == pytest.approx(3.0, abs=1e-6)


def test_cpu_host_refines_instead_of_marking_scenarios_unavailable(monkeypatch, tmp_path):
    """A host without an admitted accelerator still runs the declared CPU stage."""
    lnz = np.array([-1.0])
    ess = {0: 1.0}
    state = stage.make_state(7, {"0": _unit_points()}, "input")
    calls = []

    def nested_call(plan, **_kwargs):
        calls.append(plan["index"])
        return _refined(-1.05, 0.05)

    monkeypatch.setattr(models, "_usable_jax_devices", lambda: ())
    monkeypatch.setattr(escalation_run, "_population_blocks", lambda **_k: {})
    monkeypatch.setattr(escalation_run, "_target_plan_entry", lambda index: {"index": index, "log_prior": 0.0})
    monkeypatch.setattr(escalation_run, "_nested_call", nested_call)
    names = ("M_s_Msun", "R_s_Rsun", "plx", "contrast_separations", "contrast_values", "resolution_limit_arcsec",
             "background_population", "background_area_sq_deg", "target_tess_mag", "time", "flux", "sigma",
             "period_days", "u1", "u2", "exptime_days", "bin_integration_days", "target_aperture_flux_fraction",
             "target_feh", "nsamples")
    outcome = stage.run_mc_importance(
        lnZ=lnz, ess=ess, state=state, n_draws=controls._IMPORTANCE_DRAWS, input_identity="input",
        checkpoint_directory=tmp_path / "stage", **dict.fromkeys(names),
    )
    assert calls == [0]
    assert outcome["attempts"]["TP"]["status"] == "accepted"
    assert outcome["attempts"]["TP"].get("reason") is None


def test_stage_scalar_integrand_recovers_a_known_unit_evidence(monkeypatch):
    """Without a native callback the stage integrates the scalar float64 likelihood.

    A normalized 2-D Gaussian wholly inside the unit cube integrates to one, so
    its evidence is exactly 0 nats; the defensive proposal is a different object.
    """
    mode, width = np.array([0.31, 0.72]), 0.02

    def log_likelihood(point):
        return float(-0.5*np.sum(((point-mode)/width)**2) - np.log(2.0*np.pi*width**2))

    rng = np.random.default_rng(21)
    pilot = rng.normal(mode, width, (controls._IMPORTANCE_CENTRES, 2))
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 20_000)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 2)
    token = stage.native_stage.set(("test-stage", "identity", stage.stage_controls()))
    try:
        refined = nested._run_importance_sampler(
            lambda point: point, log_likelihood,
            [(pilot, np.full(controls._IMPORTANCE_CENTRES, 1.0 / controls._IMPORTANCE_CENTRES))],
            n_dimensions=2, random_seed=3, n_jobs=1, n_draws=20_000, native_integrand=None)
    finally:
        stage.native_stage.reset(token)
    assert refined.log_evidence == pytest.approx(0.0, abs=0.1)
    assert refined.execution_provenance["backend"] == "scalar-cpu"
    assert refined.effective_sample_size >= 100


def test_ess_one_pilot_spreads_and_passes_reach_a_narrow_mode(monkeypatch):
    """The low-scatter control: one pilot draw dominated and the kernel covariance was singular."""
    rng = np.random.default_rng(13)
    mode, width = np.array([0.31, 0.72]), 0.002
    points = rng.random((200_000, 2))

    def log_likelihood(x):
        return -0.5*np.sum(((x-mode)/width)**2, axis=1) - np.log(2.0*np.pi*width**2)

    logw = log_likelihood(points)
    plain = np.exp(logw-np.max(logw))
    assert plain.sum()**2/np.sum(plain**2) < 200.0
    power = stage.tempering_exponent(logw, controls._IMPORTANCE_PILOT_ESS)
    weights = np.exp(power*(logw-np.max(logw)))
    weights /= weights.sum()
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 20_000)
    samples, adapted, passes = importance._adapt_proposal(points, weights, log_likelihood, rng)
    assert passes[-1]["exponent"] == 1.0 and len(passes) <= controls._IMPORTANCE_ADAPTATION_ROUNDS
    assert np.average(samples, axis=0, weights=adapted) == pytest.approx(mode, abs=3*width)
    # Fresh draws from the frozen proposal recover the unit integral: adaptation shaped, it did not bias.
    mixture = importance._kernel_mixture(samples, adapted, rng, regularize=True)
    draws, inside = importance._mixture_draws(mixture, rng, 200_000, 2)
    log_w = log_likelihood(draws[inside]) - importance._mixture_log_density(mixture, draws[inside])
    assert np.exp(log_w).sum()/200_000 == pytest.approx(1.0, abs=0.02)


def test_stage_covers_near_scenarios_however_small_their_estimate():
    """A scenario with ESS 1 is staged even when its first-stage estimate is negligible."""
    lnZ = np.array([-112.4, -153.8, -700.0, -np.inf])
    ess = {0: 5000.0, 1: 1.0, 2: 1.0, 3: 1.0}
    assert escalation.stage_scenarios(lnZ, ess) == [1, 3]
    # No finite draws do not prove either prior support is empty.
    assert escalation.stage_scenarios(np.full(2, -np.inf), {0: 1.0, 1: 1.0}) == [0, 1]


def test_stage_outcome_survives_the_handoff_exactly():
    from trex.evidence.monte_carlo_state import _portable_importance, _restore_importance

    outcome = {"lnZ": np.array([-70.73, -np.inf]), "ess": {0: 37772.0, 1: 0.0},
               "records": {"TP": {"log_evidence": -70.73}}, "attempts": {"TP": {"status": "accepted"}}}
    state = {"mc_importance_outcome": _portable_importance(outcome)}
    restored = _restore_importance(state, 2)
    np.testing.assert_array_equal(restored["lnZ"], outcome["lnZ"])
    assert restored["ess"] == outcome["ess"] and restored["records"] == outcome["records"]
    assert _restore_importance({}, 2) is None
    with pytest.raises(ValueError, match="MC importance outcome is malformed") as caught:
        _restore_importance(state, 3)
    assert "another scenario set" in str(caught.value.__cause__)


def _gaussian_scenario(mode, width):
    """A 2-D unit-cube native callback carrying the same normal density as the scalar reference."""
    class Gaussian:
        n_dimensions = 2
        time = np.zeros(9)
        nsamples = 1

        def jax_unit_integrand(self, *, device):
            from trex.evidence._jax_unit_integrand import UnitIntegrandEvaluation

            jnp = models.jnp
            centre = jnp.asarray(mode, dtype=jnp.float64)
            scale = jnp.asarray(width, dtype=jnp.float64)
            normalizer = jnp.asarray(-np.log(2.0*np.pi*width**2), dtype=jnp.float64)

            class Native:
                def evaluate(self, point):
                    value = -0.5*jnp.sum(((point-centre)/scale)**2) + normalizer
                    return UnitIntegrandEvaluation(value, jnp.isfinite(value))

            return Native()

    return Gaussian()


@pytest.mark.skipif(not models.HAS_JAX_TREX, reason="JAX is unavailable")
def test_cpu_native_stage_batches_the_same_float64_estimator(monkeypatch):
    """No accelerator: the scenario callback runs batched on the CPU device.

    The scalar and native backends execute the same estimator on the same
    proposal draws, so each scenario's evidence must agree far inside the
    0.1-nat integration bar and the record must name the CPU backend.
    """
    mode, width = np.array([0.31, 0.72]), 0.02

    def log_likelihood(point):
        return float(-0.5*np.sum(((point-mode)/width)**2) - np.log(2.0*np.pi*width**2))

    rng = np.random.default_rng(21)
    pilot = rng.normal(mode, width, (controls._IMPORTANCE_CENTRES, 2))
    posteriors = [(pilot, np.full(controls._IMPORTANCE_CENTRES, 1.0/controls._IMPORTANCE_CENTRES))]
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 5_000)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 2)
    token = stage.native_stage.set(("test-stage", "identity", stage.stage_controls()))
    try:
        scalar = nested._run_importance_sampler(
            lambda point: point, log_likelihood, posteriors,
            n_dimensions=2, random_seed=3, n_jobs=1, n_draws=4_000)
        native = nested._run_importance_sampler(
            lambda point: point, log_likelihood, posteriors,
            n_dimensions=2, random_seed=3, n_jobs=1, n_draws=4_000,
            cpu_native_integrand=_gaussian_scenario(mode, width))
    finally:
        stage.native_stage.reset(token)
    assert scalar.execution_provenance["backend"] == "scalar-cpu"
    assert native.execution_provenance["backend"] == "jax-native-cpu"
    assert native.log_evidence == pytest.approx(0.0, abs=0.05)
    assert native.log_evidence == pytest.approx(scalar.log_evidence, abs=1e-6)
    assert native.effective_sample_size == pytest.approx(scalar.effective_sample_size, rel=1e-9)
    assert native.n_likelihood_calls == scalar.n_likelihood_calls


def test_scenarios_without_a_native_callback_keep_the_scalar_stage(monkeypatch):
    class ScalarOnly:
        baseline_likelihood = object()

    class Native:
        baseline_likelihood = object()

        def jax_unit_integrand(self, *, device):
            raise AssertionError("capability selection must not execute the callback")

    monkeypatch.setattr(models, "HAS_JAX_TREX", True)
    assert nested._cpu_refinement_integrand(ScalarOnly()) is None
    assert nested._native_refinement_integrand(ScalarOnly()) is None
    candidate = Native()
    assert nested._cpu_refinement_integrand(candidate) is candidate
    monkeypatch.setattr(models, "HAS_JAX_TREX", False)
    assert nested._cpu_refinement_integrand(candidate) is None


@pytest.mark.skipif(not models.HAS_JAX_TREX, reason="JAX is unavailable")
def test_a_paused_stage_cannot_resume_across_integrand_backends(monkeypatch, tmp_path):
    """Scalar and native log-likelihood values are one estimate; never mix them on resume."""
    mode, width = np.array([0.31, 0.72]), 0.02

    def log_likelihood(point):
        return float(-0.5*np.sum(((point-mode)/width)**2) - np.log(2.0*np.pi*width**2))

    rng = np.random.default_rng(21)
    pilot = rng.normal(mode, width, (controls._IMPORTANCE_CENTRES, 2))
    posteriors = [(pilot, np.full(controls._IMPORTANCE_CENTRES, 1.0/controls._IMPORTANCE_CENTRES))]
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 2_000)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 1)
    captured = []
    monkeypatch.setattr(importance, "_checkpoint_identity", lambda value: captured.append(value) or "identity")
    token = stage.native_stage.set(("test-stage", "identity", stage.stage_controls()))
    try:
        for integrand in (None, _gaussian_scenario(mode, width)):
            with pytest.raises(nested.NestedCheckpointPaused):
                with nested.nested_checkpoint(tmp_path / "pending", deadline=0.0):
                    nested._run_importance_sampler(
                        lambda point: point, log_likelihood, posteriors,
                        n_dimensions=2, random_seed=3, n_jobs=1, n_draws=4_000,
                        cpu_native_integrand=integrand)
    finally:
        stage.native_stage.reset(token)
    assert captured[0][-1] == "scalar-cpu" and captured[1][-1] == "jax-native"


def _identity_transform(point):
    return point


def _checkpoint_gaussian(point):
    return float(-0.5*np.sum(((point - np.array([0.31, 0.72]))/0.02)**2) - np.log(2*np.pi*0.02**2))


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_scalar_only_scenario_runs_the_real_refinement_when_jax_is_installed(monkeypatch, seed):
    class ScalarOnly:
        n_dimensions = 2
        time = np.zeros(9)
        nsamples = 1
        prior_transform = staticmethod(_identity_transform)
        log_likelihood = staticmethod(_checkpoint_gaussian)

    monkeypatch.setattr(models, "HAS_JAX_TREX", True)
    monkeypatch.setattr(models, "_usable_jax_devices", lambda: [object()])
    monkeypatch.setattr(importance, "_IMPORTANCE_CENTRES", 32)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 2048)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 1)
    sampler = nested._run_importance_sampler
    monkeypatch.setattr(nested, "_run_importance_sampler",
                        lambda *args, **kwargs: sampler(*args, n_draws=4096, **kwargs))
    pilot = np.random.default_rng(21).normal([.31, .72], .02, (128, 2))
    posteriors = [(pilot, np.full(128, 1/128))]
    token = stage.native_stage.set(("test-stage", "identity", stage.stage_controls()))
    try:
        result = nested._integrate(ScalarOnly(), SimpleNamespace(random_seed=seed, n_jobs=1), "synthetic", 2, posteriors)
    finally:
        stage.native_stage.reset(token)
    # A narrow, normalized Gaussian has negligible mass outside the unit cube.
    assert result.log_evidence == pytest.approx(0., abs=.1)
    assert result.execution_provenance["backend"] == "scalar-cpu"


@pytest.mark.skipif(not models.HAS_JAX_TREX, reason="JAX is unavailable")
@pytest.mark.parametrize("stage_active", [False, True], ids=["nested-posterior", "mc-pilot"])
@pytest.mark.parametrize("first_native", [False, True], ids=["scalar-first", "native-first"])
def test_partial_importance_resume_refuses_another_integrand_model(monkeypatch, tmp_path, stage_active, first_native):
    """Real retained values may resume on the same model, never a different one."""
    from trex.accelerators import _native_unit_batch

    monkeypatch.setattr(importance, "_IMPORTANCE_CENTRES", 32)
    monkeypatch.setattr(importance, "_IMPORTANCE_EVALUATION_CHUNK", 8)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 0)
    monkeypatch.setattr(_native_unit_batch, "evaluation_block", lambda count, _size: min(count, 8))
    rng = np.random.default_rng(21)
    pilot = rng.normal([0.31, 0.72], 0.02, (64, 2))
    posteriors = [(pilot, np.full(64, 1/64))]
    native = _gaussian_scenario(np.array([0.31, 0.72]), 0.02)
    token = stage.native_stage.set(("test-stage", "identity", stage.stage_controls()) if stage_active else None)

    def run(use_native):
        return nested._run_importance_sampler(
            _identity_transform, _checkpoint_gaussian, posteriors,
            n_dimensions=2, random_seed=3, n_jobs=1, n_draws=128,
            cpu_native_integrand=native if use_native else None,
        )

    try:
        reference = run(first_native)
        ticks = iter([0.0, 0.0, 0.0, 2.0])
        monkeypatch.setattr(importance, "_monotonic", lambda: next(ticks, 2.0))
        prefix = tmp_path / "pending"
        with pytest.raises(nested.NestedCheckpointPaused):
            with nested.nested_checkpoint(prefix, deadline=1.0):
                run(first_native)
        with np.load(prefix.with_suffix(".importance.npz"), allow_pickle=False) as saved:
            assert 0 < saved["log_l"].size < 128
        monkeypatch.setattr(importance, "_monotonic", lambda: 0.0)
        with pytest.raises(ValueError, match="importance checkpoint inputs do not match"):
            with nested.nested_checkpoint(prefix, deadline=1.0):
                run(not first_native)
        with nested.nested_checkpoint(prefix, deadline=1.0):
            resumed = run(first_native)
        assert resumed.log_evidence == pytest.approx(reference.log_evidence, abs=1e-12)
        assert resumed.effective_sample_size == pytest.approx(reference.effective_sample_size, abs=1e-10)
        assert resumed.n_likelihood_calls == reference.n_likelihood_calls
    finally:
        stage.native_stage.reset(token)


def test_probit_gaussian_draws_fill_the_cube_and_its_density_normalizes():
    """Complete support and the exact Jacobian: the defensive mixture integrates to one."""
    rng = np.random.default_rng(41)
    samples = rng.beta(2.0, 5.0, (50_000, 1))
    mixture = importance._probit_mixture(samples, np.full(50_000, 1.0/50_000))
    draws, inside = importance._probit_draws(mixture, np.random.default_rng(42), 20_000, 1)
    assert inside.size == 20_000 and np.all((draws > 0.0) & (draws < 1.0))
    # Integrate in probit coordinates: x = Phi(z), dx = phi(z) dz. A uniform-x
    # grid under-resolves the integrable growth of the Gaussian near x = 1;
    # the change of variables is exact and avoids that quadrature artefact.
    z = np.linspace(-7.5, 7.5, 15_001)
    density = np.exp(importance._probit_log_density(mixture, ndtr(z)[:, None]))
    weight = np.exp(-0.5*z**2)/math.sqrt(2.0*math.pi)
    # The defensive mixture is 0.1 * prior + 0.9 * Gaussian and must normalize.
    # This trapezoid is an independent numerical oracle (it never reuses the
    # fit's log_norm); dz = 1e-3 and ndtri(ndtr(z)) rounding keep the error
    # below the declared 5e-4 bound.
    assert np.trapezoid(density*weight, z) == pytest.approx(1.0, abs=5e-4)


def test_probit_fit_refuses_weighted_boundary_samples():
    with pytest.raises(ValueError, match="strictly inside"):
        importance._probit_mixture(np.array([[0.0], [0.5], [1.0]]), np.full(3, 1.0/3.0))
    # A zero-weight boundary sample cannot influence the fit.
    mixture = importance._probit_mixture(np.array([[0.0], [0.5], [1.0]]), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(mixture["mean"], [0.0], atol=1e-15)


@pytest.mark.parametrize("z,prior_draw,outcome", [
    ([-10., -1., 0., 1.], .5, [0, 1, 2, 3]),
    # ndtr rounds these onto the float64 cube boundary: a zero contribution
    # that keeps its slot in n_draws, exactly like a kernel draw outside it.
    ([8.3, 0.], .5, [1]),
    ([-40.], .5, []),
    ([np.nan], .5, "raises"),
    ([np.inf], .5, "raises"),
    ([], 0., []),
    ([], 1., []),
    ([], np.nan, []),
])
def test_probit_draws_neither_clip_nor_redraw(z, prior_draw, outcome):
    # -10 remains representable under ndtr but the old +/-8.2 clip moved it.
    gaussian = bool(z)
    count = len(z) if gaussian else 1
    randoms = iter((np.full(count, float(gaussian)), np.full((count, 1), prior_draw)))
    normals = iter((np.asarray(z).reshape((-1, 1)),))
    rng = SimpleNamespace(random=lambda shape: next(randoms),
                          standard_normal=lambda shape: next(normals).reshape(shape))
    mixture = {"mean": np.zeros(1), "chol": np.eye(1)}
    if outcome == "raises":
        with pytest.raises(ArithmeticError, match="nonfinite Gaussian coordinate"):
            importance._probit_draws(mixture, rng, count, 1)
    else:
        draws, inside = importance._probit_draws(mixture, rng, count, 1)
        assert draws.shape == (count, 1)
        if gaussian:
            np.testing.assert_array_equal(draws[:, 0], ndtr(z))
        np.testing.assert_array_equal(inside, np.asarray(outcome, dtype=int))
    assert next(randoms, None) is None and next(normals, None) is None


def test_single_weight_probit_fit_stays_proper():
    mixture = importance._probit_mixture(np.array([[0.25], [0.7]]), np.array([1.0, 0.0]))
    # One atom has zero covariance plus the documented 1e-10 ridge.
    np.testing.assert_allclose(mixture["chol"], [[1e-5]], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(mixture["mean"], [ndtri(0.25)], atol=1e-14)
    points = np.linspace(np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), 101)[:, None]
    assert np.all(np.isfinite(importance._probit_log_density(mixture, points)))


def test_probit_adaptation_refits_a_normalized_tempered_sample(monkeypatch):
    rng = np.random.default_rng(71)
    points = rng.random((20_000, 1))
    log_weights = -0.5*((points[:, 0] - 0.3)/0.05)**2
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_DRAWS", 4_000)
    monkeypatch.setattr(controls, "_IMPORTANCE_ADAPTATION_ROUNDS", 1)
    samples, adapted, passes = importance._adapt_proposal(
        points, weights, lambda x: -0.5*((x[:, 0] - 0.3)/0.05)**2, rng,
        proposal="probit-gaussian",
    )
    assert len(passes) == 1 and passes[0]["likelihood_calls"] == 4_000
    assert passes[0]["exponent"] is not None and 0.0 <= passes[0]["exponent"] <= 1.0
    assert adapted.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all((samples > 0.0) & (samples < 1.0))
    # Tempered draws stay on the synthetic mode; 0.02 is five posterior widths,
    # far above the weighted-mean Monte Carlo error of this pass.
    assert np.average(samples, axis=0, weights=adapted) == pytest.approx(0.3, abs=0.02)


def test_both_proposal_families_recover_a_known_one_dimensional_evidence():
    rate = 3.0
    truth = math.log(1.0 - math.exp(-rate))
    rng = np.random.default_rng(51)
    # Exact unit-cube posterior draws: F(x) = (1 - exp(-rate x)) / (1 - exp(-rate)).
    pilot = -np.log1p(-rng.random(4_000)*(1.0 - math.exp(-rate)))/rate
    posteriors = [(pilot[:, None], np.full(pilot.size, 1.0/pilot.size))]

    def log_likelihood(point):
        return math.log(rate) - rate*point[0]

    results = {
        name: nested._run_importance_sampler(
            lambda point: point, log_likelihood, posteriors, n_dimensions=1,
            random_seed=17, n_jobs=1, n_draws=40_000, proposal=name)
        for name in ("kernel-mixture", "probit-gaussian")
    }
    for result in results.values():
        # Four reported standard errors plus 0.01 nat: the reported error is the
        # sample error of the importance weights, 4 sigma is the predeclared
        # Monte Carlo bound, and the floor covers the analytic value's rounding.
        assert abs(result.log_evidence - truth) <= 4.0*result.log_evidence_error + 0.01
    assert results["kernel-mixture"].n_centres == controls._IMPORTANCE_CENTRES
    assert results["probit-gaussian"].n_centres == 1
    assert "proposal" not in results["kernel-mixture"].execution_provenance
    assert results["probit-gaussian"].execution_provenance["proposal"] == "probit-gaussian"


def test_both_proposal_families_recover_a_known_correlated_rectangle_evidence():
    mode, sd, correlation, amplitude = np.array([0.35, 0.62]), np.array([0.06, 0.10]), 0.75, 0.2
    covariance = np.array([[sd[0]**2, correlation*sd[0]*sd[1]],
                           [correlation*sd[0]*sd[1], sd[1]**2]])
    # scipy's Genz rectangle probability carries a default 1e-5 absolute
    # tolerance; the 0.01-nat floor below is three orders of magnitude larger.
    rectangle = float(stats.multivariate_normal(mean=mode, cov=covariance).cdf(
        np.ones(2), lower_limit=np.zeros(2)))
    truth = math.log(amplitude*rectangle)
    inverse = np.linalg.inv(covariance)
    normalizer = -math.log(2.0*math.pi) - 0.5*math.log(np.linalg.det(covariance))

    def log_likelihood(point):
        delta = point - mode
        return math.log(amplitude) + normalizer - 0.5*delta @ inverse @ delta

    rng = np.random.default_rng(52)
    proposed = rng.multivariate_normal(mode, covariance, 8_000)
    pilot = proposed[np.all((proposed > 0.0) & (proposed < 1.0), axis=1)]
    posteriors = [(pilot, np.full(len(pilot), 1.0/len(pilot)))]
    results = {
        name: nested._run_importance_sampler(
            lambda point: point, log_likelihood, posteriors, n_dimensions=2,
            random_seed=18, n_jobs=1, n_draws=20_000, proposal=name)
        for name in ("kernel-mixture", "probit-gaussian")
    }
    for result in results.values():
        assert math.isfinite(result.log_evidence)
        assert abs(result.log_evidence - truth) <= 4.0*result.log_evidence_error + 0.01


def test_single_probit_gaussian_is_only_a_limitation_on_a_bimodal_target():
    """Limitation control, not validation: one Gaussian cannot represent two modes."""
    modes, width, n = np.array([0.25, 0.75]), 0.01, 20_000
    rng = np.random.default_rng(61)
    pilot = np.where(rng.integers(0, 2, n) == 0,
                     rng.normal(modes[0], width, n), rng.normal(modes[1], width, n))
    posteriors = [(pilot[:, None], np.full(n, 1.0/n))]
    normalizer = math.log(width*math.sqrt(2.0*math.pi)*2.0)

    def log_likelihood(point):
        # Two equal normalized modes fully inside the cube: ln Z is zero.
        return float(np.logaddexp.reduce(-0.5*((modes - point[0])/width)**2) - normalizer)

    result = nested._run_importance_sampler(
        lambda point: point, log_likelihood, posteriors, n_dimensions=1,
        random_seed=19, n_jobs=1, n_draws=40_000, proposal="probit-gaussian")
    # The defensive 0.1 prior component bounds the weights, so a misspecified
    # single Gaussian stays unbiased with a larger variance. 0.1 nats is loose.
    assert math.isfinite(result.log_evidence)
    assert abs(result.log_evidence) <= 0.1
    assert 0.0 < result.maximum_weight_share <= 1.0
    assert 0.0 < result.effective_sample_size <= result.n_draws
    mixture = importance._probit_mixture(pilot[:, None], posteriors[0][1])
    draws, inside = importance._probit_draws(mixture, np.random.default_rng(62), 20_000, 1)
    assert inside.size == 20_000 and np.all((draws > 0.0) & (draws < 1.0))


def test_non_default_proposal_changes_the_checkpoint_identity_only(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(importance, "_checkpoint_identity", lambda value: captured.append(value) or "identity")
    posteriors = [(np.linspace(0.2, 0.8, 16)[:, None], np.full(16, 1.0/16))]

    def attempt(**kwargs):
        with pytest.raises(nested.NestedCheckpointPaused):
            with nested.nested_checkpoint(tmp_path / "pending", deadline=0.0):
                nested._run_importance_sampler(
                    lambda point: point, lambda point: 0.0, posteriors,
                    n_dimensions=1, random_seed=2, n_jobs=1, n_draws=16, **kwargs)

    attempt()
    attempt(proposal="probit-gaussian")
    assert captured[0][-1] == "scalar-cpu"
    assert captured[1][-3:] == ("scalar-cpu", "probit-gaussian", "unclipped-ndtr-v1")
    assert len(captured[1]) == len(captured[0]) + 2


def test_unknown_proposal_family_is_refused_before_any_draw():
    with pytest.raises(ValueError, match="proposal must be one of"):
        nested._run_importance_sampler(
            lambda point: point, lambda point: 0.0, [],
            n_dimensions=1, random_seed=1, n_jobs=1, n_draws=8, proposal="kde")
    with pytest.raises(ValueError, match="proposal must be one of"):
        importance._adapt_proposal(
            np.full((4, 1), 0.5), np.full(4, 0.25), None, np.random.default_rng(1), proposal="kde")
