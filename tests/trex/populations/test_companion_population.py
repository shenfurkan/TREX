"""Literature-consistent bound-companion population for the TREX P/S scenarios.

Every value expected here is derived directly from Moe & Di Stefano (2017),
ApJS 230, 15 (DOI 10.3847/1538-4365/aa6fb6), read from the retained e-print:

* Eqn. 5-7  -- excess twin fraction ``F_twin(M1, P)``, its short-period value
  ``0.30 - 0.15 log10(M1/Msun)`` and its cutoff ``logP_twin = 8.0 - M1/Msun``.
* Eqn. 9    -- large-mass-ratio slope ``gamma_largeq`` for solar-type primaries:
  ``-0.5`` below ``logP = 5`` and ``-0.5 - 0.3(logP - 5)`` above it.
* Eqn. 13   -- small-mass-ratio slope ``gamma_smallq = 0.3`` across the whole
  ``0.2 <= logP < 8.0`` domain for solar-type primaries.
* Eqn. 23   -- companion frequency per decade of orbital period, whose
  cumulative integral is the companion occurrence prior.

None of these expectations is produced by running the code under test.
"""

from __future__ import annotations

import math

import numpy as np
from tests.fixtures.synthetic_observations import SYNTHETIC_SOLAR_METALLICITY_DEX
import pytest

from trex.populations import priors


# ---------------------------------------------------------------------------
# Mass-ratio distribution parameters as functions of orbital period
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "log_period,expected",
    [
        (0.2, -0.5),          # Eqn. 9, flat branch, domain minimum
        (4.9, -0.5),          # Eqn. 9, flat branch, just below the break
        (5.0, -0.5),          # Eqn. 9, at the break
        (6.0, -0.8),          # -0.5 - 0.3 * 1
        (6.5, -0.95),         # the value triceratops hardcodes
        (8.0, -1.4),          # -0.5 - 0.3 * 3, domain maximum
    ],
)
def test_largeq_slope_follows_moe2017_equation_9(log_period, expected):
    value = priors.moe2017_largeq_slope(np.array([log_period]))
    assert float(value[0]) == pytest.approx(expected)


def test_largeq_slope_rejects_periods_outside_the_measured_domain():
    for log_period in (0.1, 8.1):
        with pytest.raises(ValueError, match="orbital period"):
            priors.moe2017_largeq_slope(np.array([log_period]))


@pytest.mark.parametrize(
    "primary_mass,log_period,expected",
    [
        # Eqn. 6 at 1 Msun: F_twin;logP<1 = 0.30; Eqn. 7: logP_twin = 7.0.
        (1.0, 0.5, 0.30),
        (1.0, 1.0, 0.30),
        (1.0, 4.0, 0.30 * (1.0 - 3.0 / 6.0)),
        (1.0, 6.0, 0.30 * (1.0 - 5.0 / 6.0)),   # = 0.05, the triceratops value
        (1.0, 6.5, 0.30 * (1.0 - 5.5 / 6.0)),
        (1.0, 7.0, 0.0),
        (1.0, 7.5, 0.0),
    ],
)
def test_twin_fraction_follows_moe2017_equations_5_to_7(primary_mass, log_period, expected):
    value = priors.moe2017_twin_fraction(primary_mass, np.array([log_period]))
    assert float(value[0]) == pytest.approx(expected)


@pytest.mark.parametrize("primary_mass", [0.5, 0.6, 40.01])
def test_twin_fraction_refuses_masses_outside_the_calibrated_moe_branch(primary_mass):
    """The 0.5 Msun Eqn. 7 branch is not the adopted 0.60<M1<=40 branch."""
    with pytest.raises(ValueError, match="0.60|40"):
        priors.moe2017_twin_fraction(primary_mass, np.array([0.5]))


def test_triceratops_companion_constants_are_not_a_single_period_evaluation():
    """Records the comparison that motivated this per-draw model.

    ``triceratops.priors.sample_q_companion`` hardcodes ``p2 = -0.95`` and
    ``F_twin = 0.05``.  Against Moe & Di Stefano's own relations at one solar
    mass those correspond to different orbital periods, so the pair cannot be
    reproduced by evaluating the paper at any single period.
    """
    slope_period = 6.5
    twin_period = 6.0

    assert float(priors.moe2017_largeq_slope(np.array([slope_period]))[0]) == pytest.approx(-0.95)
    assert float(priors.moe2017_twin_fraction(1.0, np.array([twin_period]))[0]) == pytest.approx(0.05)
    # Cross-evaluated at the other's period, neither reproduces the constant.
    assert float(priors.moe2017_largeq_slope(np.array([twin_period]))[0]) != pytest.approx(-0.95)
    assert float(priors.moe2017_twin_fraction(1.0, np.array([slope_period]))[0]) != pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Cumulative companion frequency (Eqn. 23) and period sampling
# ---------------------------------------------------------------------------

def test_cumulative_companion_frequency_starts_at_the_domain_minimum():
    """Eqn. 23's first segment spans 0.2 <= logP < 1.0 and must be integrated.

    The frequency is flat at ``f_logP<1;q>0.3`` there, so the cumulative value
    at logP = 1.0 is exactly that constant times the 0.8 dex width.
    """
    mass = 1.0
    log10_mass = math.log10(mass)
    c0, c1, c2 = priors.MOE2017_COMPANION_FRACTION_F1_COEFFS
    f1 = c0 + c1 * log10_mass + c2 * log10_mass ** 2

    at_minimum = priors.moe2017_cumulative_companion_frequency(mass, np.array([0.2]))
    at_first_break = priors.moe2017_cumulative_companion_frequency(mass, np.array([1.0]))

    assert float(at_minimum[0]) == pytest.approx(0.0)
    assert float(at_first_break[0]) == pytest.approx(f1 * 0.8)


def _eqn23_density(log_period, mass):
    """Moe & Di Stefano (2017) Eqn. 23 with Eqns. 20-22, transcribed from the paper's TeX
    (binstat_ApJv3.tex, Eqns. 20-23), alpha = 0.018, Delta logP = 0.7."""
    x = math.log10(mass)
    f1, f27, f55 = 0.020 + 0.04 * x + 0.07 * x * x, 0.039 + 0.07 * x + 0.01 * x * x, 0.078 - 0.05 * x + 0.04 * x * x
    alpha, width = 0.018, 0.7
    if log_period < 1.0:
        return f1
    if log_period < 2.7 - width:
        return f1 + (log_period - 1.0) / (1.7 - width) * (f27 - f1 - alpha * width)
    if log_period < 2.7 + width:
        return f27 + alpha * (log_period - 2.7)
    if log_period < 5.5:
        return f27 + alpha * width + (log_period - 2.7 - width) / (2.8 - width) * (f55 - f27 - alpha * width)
    return f55 * math.exp(-0.3 * (log_period - 5.5))


@pytest.mark.parametrize("mass", [1.0, 3.5, 12.0])
def test_cumulative_companion_frequency_is_the_integral_of_the_published_equation_23(mass):
    """Regression for the 3.4 <= logP < 5.5 segment, once integrated with (logP - 2.0)/2.1
    instead of the paper's (logP - 3.4)/2.1: the solar q > 0.3 total read 0.404, not 0.367."""
    from scipy.integrate import quad

    for upper in (2.5, 3.4, 4.5, 5.5, 8.0):
        breaks = [edge for edge in (1.0, 2.0, 3.4, 5.5) if edge < upper]
        expected = quad(_eqn23_density, 0.2, upper, args=(mass,), points=breaks,
                        epsabs=1e-14, epsrel=1e-13, limit=200)[0]
        produced = float(priors.moe2017_cumulative_companion_frequency(mass, np.array([upper]))[0])
        assert produced == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("mass", [0.5, 0.6, 40.01])
def test_cumulative_companion_frequency_refuses_out_of_domain_masses(mass):
    """The Moe branch is calibrated only for 0.60 < M1 <= 40 Msun."""
    with pytest.raises(ValueError, match="0.60|40"):
        priors.moe2017_cumulative_companion_frequency(mass, np.array([1.0]))


def test_cumulative_companion_frequency_is_monotone_and_bounded():
    grid = np.linspace(0.2, 8.0, 200)
    values = priors.moe2017_cumulative_companion_frequency(1.0, grid)

    assert np.all(np.diff(values) >= 0.0)
    assert values[0] == pytest.approx(0.0)
    assert 0.0 < float(values[-1]) < 1.0


def _solar_q_support_correction(log_period):
    """Companions with q > 0.1 per companion with q > 0.3 at one solar mass.

    Written from MDS17 section 2, Eq. 2 and section 9.1, not from the production
    sampler: q**0.3 on [0.1, 0.3], continuous with q**gamma_largeq on [0.3, 1];
    the excess twins are a fraction F_twin of all q > 0.3 companions, so the
    q > 0.3 integral is J / (1 - F_twin) and the ratio is 1 + (1 - F_twin) I / J.
    """
    from scipy.integrate import quad

    gamma = float(priors.moe2017_largeq_slope(np.array([log_period]))[0])
    twins = float(priors.moe2017_twin_fraction(1.0, np.array([log_period]))[0])
    small = quad(lambda q: q ** 0.3, 0.1, 0.3)[0]
    large = quad(lambda q: 0.3 ** 0.3 * (q / 0.3) ** gamma, 0.3, 1.0)[0]
    return 1.0 + (1.0 - twins) * small / large


def _solar_supported_companion_frequency(upper_log_period):
    """Integral of the Eq. 23 q > 0.3 rate times the q-support correction from log P = 0.2."""
    from scipy.integrate import quad

    def rate(m, h=1e-6):  # dN/dlogP from the paper's own cumulative (Eq. 23 integral)
        values = priors.moe2017_cumulative_companion_frequency(1.0, np.array([m - h, m + h]))
        return float(values[1] - values[0]) / (2.0 * h)

    breaks = [b for b in (1.0, 2.0, 3.4, 5.0, 5.5, 20.0 / 3.0, 7.0) if 0.2 < b < upper_log_period]
    return quad(lambda m: rate(m) * _solar_q_support_correction(m), 0.2, upper_log_period,
                points=breaks or None, limit=400)[0]


def test_sampled_companion_periods_reproduce_the_frequency_distribution():
    """Inverse-transform samples follow Eq. 23 weighted by the drawn q support."""
    rng = np.random.default_rng(11)
    samples = priors.sample_companion_log_period(rng.random(200000), 1.0)

    assert samples.min() >= 0.2 and samples.max() <= 8.0
    total = _solar_supported_companion_frequency(8.0)
    for probe in (1.0, 3.0, 5.5, 7.0):
        expected = _solar_supported_companion_frequency(probe) / total
        # 200,000 draws: binomial sigma <= 0.0011, so 5e-3 is about 4.5 sigma.
        assert float(np.mean(samples <= probe)) == pytest.approx(expected, abs=5e-3)


def test_m_dwarf_companion_projected_separations_follow_the_winters_gaussian():
    """Winters et al. (2019) Sect. 5.4/Fig. 16: log-normal projected separation peaking at 20 AU, sigma 1.16 dex."""
    rng = np.random.default_rng(5)
    log_separations = priors.sample_companion_log_projected_separation(rng.random(200000), 0.3)
    assert float(np.mean(log_separations)) == pytest.approx(np.log10(20.0), abs=0.02)
    assert float(np.std(log_separations)) == pytest.approx(1.16, rel=0.02)


@pytest.mark.parametrize("separation_au", [2.0, 20.0])
def test_m_dwarf_mc_draw_projects_the_semimajor_axis_onto_the_sky(separation_au):
    """Log-normal a, then s = a sqrt(1-u^2) with u uniform (circular, isotropic orbit).

    The production MC draw decides unresolved retention from s. With the same seed, the
    fraction retained inside the limit over that with no limit is P(s <= S), because q
    (and hence the fainter-companion cut) is drawn independently of the orbit. Expected
    value: quadrature over u of the log-normal axis CDF, with the axis moments derived
    from Winters (2019) mean log10(20 AU), sigma 1.16 dex minus the projection moments
    (ln 2 - 1)/ln 10 and (1 - pi^2/12)/ln(10)^2. Without projection (s = a) the 20 AU
    fraction would be 0.454 instead of 0.499.
    """
    from scipy import integrate, stats
    from trex.evidence.mc_population import _draw_bound_companion

    mean = math.log10(20.0) - (math.log(2.0) - 1.0) / math.log(10.0)
    sigma = math.sqrt(1.16 ** 2 - (1.0 - math.pi ** 2 / 12.0) / math.log(10.0) ** 2)
    expected = integrate.quad(lambda u: stats.norm.cdf(
        (math.log10(separation_au / math.sqrt(1.0 - u * u)) - mean) / sigma), 0.0, 1.0)[0]
    plx_mas = 100.0  # 10 pc: 1 arcsec = 10 AU

    def retained(limit_arcsec):
        lnprior = _draw_bound_companion(np.random.default_rng(3), 20000, 0.3, 0.3, plx_mas, None, None,
                                        resolution_limit_arcsec=limit_arcsec)[1]
        return np.isfinite(lnprior)

    inside, everywhere = retained(separation_au * plx_mas / 1000.0), retained(1e9)
    assert np.all(everywhere[inside])
    # About 18,000 fainter companions: binomial sigma <= 0.0037, tolerance about 4 sigma.
    assert float(inside.sum() / everywhere.sum()) == pytest.approx(expected, abs=0.015)


@pytest.mark.parametrize("hosted", [False, True], ids=["target-hosted", "companion-hosted"])
def test_bound_draws_condition_on_the_paired_primary_properties(hosted):
    """Same unit variates as the fixed-host kernel, alternating across the Winters/Moe boundary."""
    from trex.evidence.mc_population import _draw_bound_companion

    n = 128
    primary_a = (0.38, 0.39, 3450.0)
    primary_b = (1.05, 1.50, 5750.0)
    mask = np.arange(n) % 2 == 0
    mass, radius, teff = (np.where(mask, a, b) for a, b in zip(primary_a, primary_b, strict=True))

    def draw(m, r, t):
        result = _draw_bound_companion(
            np.random.default_rng(703), n, m, r, 20.0, None, None,
            resolution_limit_arcsec=5.0, host_teff_k=t, observed_period_days=3.0,
            stability_hosted=hosted,
        )
        return np.asarray(result)

    fixed_a, fixed_b = draw(*primary_a), draw(*primary_b)
    expected = np.where(mask, fixed_a, fixed_b)
    actual = draw(mass, radius, teff)
    assert np.count_nonzero(np.isfinite(expected[1, mask])) > 0
    assert np.count_nonzero(np.isfinite(expected[1, ~mask])) > 0
    np.testing.assert_array_equal(np.isfinite(actual[1]), np.isfinite(expected[1]))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_m_dwarf_companion_period_uses_both_masses_and_the_dupuy_liu_factor():
    """a = 1.16 x projected separation (Dupuy & Liu 2011 Table 6), then Kepler's third law with the pair mass."""
    total_mass, log_projected = 0.45, np.log10(20.0)
    # P[yr] = (a[AU]^3 / M[Msun])^(1/2), Julian year 365.25 d.
    expected = np.log10(np.sqrt((1.16 * 20.0) ** 3 / total_mass) * 365.25)
    got = float(priors.companion_log_period_from_log_projected_separation(total_mass, np.array([log_projected]))[0])
    assert got == pytest.approx(expected, abs=2e-4)


def test_companion_population_frequency_matches_the_source_normalization():
    # Winters et al. (2019) section 5.2: 0.324 companions per primary down to
    # 0.075 Msun (a companion rate, not the 26.8% multiplicity fraction). With the
    # Reggiani & Meyer (2013) dN/dq ~ q^0.25 law only the part above the 0.1 Msun
    # stellar-grid floor is modelled: 0.324 * (1 - (0.1/0.3)^1.25) / (1 - (0.075/0.3)^1.25).
    supported = (1 - (0.1 / 0.3) ** 1.25) / (1 - (0.075 / 0.3) ** 1.25)
    assert priors.companion_population_frequency(0.3) == pytest.approx(0.324 * supported, rel=1e-12)
    # Moe branch: Eq. 23's q > 0.3 rate rescaled to the q > 0.1 support (0.49961 at 1 Msun).
    assert priors.companion_population_frequency(1.0) == pytest.approx(
        _solar_supported_companion_frequency(8.0), rel=1e-4)


# ---------------------------------------------------------------------------
# Mass-ratio draws conditioned on the sampled period
# ---------------------------------------------------------------------------

def test_companion_mass_ratios_respond_to_the_per_draw_period():
    """A steeper gamma_largeq at long periods must favour smaller mass ratios."""
    x = np.linspace(1e-6, 1.0 - 1e-6, 20000)
    short = priors.sample_q_companion(x, 1.0, np.full(x.size, 3.0))
    long = priors.sample_q_companion(x, 1.0, np.full(x.size, 7.5))

    assert np.all((short > 0.0) & (short <= 1.0))
    assert np.all((long > 0.0) & (long <= 1.0))
    assert float(np.mean(long)) < float(np.mean(short))


def test_companion_mass_ratio_twin_excess_disappears_beyond_the_twin_cutoff():
    """Above logP_twin (7.0 at 1 Msun) Eqn. 5 removes the excess twin population."""
    x = np.linspace(1e-6, 1.0 - 1e-6, 40000)
    with_twins = priors.sample_q_companion(x, 1.0, np.full(x.size, 1.0))
    without_twins = priors.sample_q_companion(x, 1.0, np.full(x.size, 7.5))

    threshold = priors.MOE2017_Q_TWIN_THRESHOLD
    assert float(np.mean(with_twins >= threshold)) > float(np.mean(without_twins >= threshold))


def test_companion_mass_ratio_rejects_a_mismatched_period_array():
    with pytest.raises(ValueError, match="per-draw"):
        priors.sample_q_companion(np.linspace(0.1, 0.9, 10), 1.0, np.full(3, 4.0))


def test_companion_mass_ratio_is_continuous_through_the_unit_slope_singularity():
    """gamma_largeq passes through -1, where the power-law integral is a log.

    Eqn. 9 reaches gamma_largeq = -1 at log P = 20/3, so the ``(q**s - a**s)/s``
    normalization hits a removable 0/0 there.  The draws must stay finite and
    vary continuously across it rather than producing nan or a discontinuity.
    """
    singular = 20.0 / 3.0
    assert float(priors.moe2017_largeq_slope(np.array([singular]))[0]) == pytest.approx(-1.0)

    x = np.linspace(1e-6, 1.0 - 1e-6, 2001)
    at = priors.sample_q_companion(x, 1.0, np.full(x.size, singular))
    below = priors.sample_q_companion(x, 1.0, np.full(x.size, singular - 1e-9))
    above = priors.sample_q_companion(x, 1.0, np.full(x.size, singular + 1e-9))

    assert np.all(np.isfinite(at)) and np.all((at > 0.0) & (at <= 1.0))
    np.testing.assert_allclose(at, below, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(at, above, rtol=1e-7, atol=1e-9)


def test_host_mass_ratio_floor_keeps_every_secondary_a_star():
    """A low-mass host cannot be given a sub-stellar 'stellar' secondary."""
    hosts = np.linspace(0.15, 1.5, 5000)
    q = priors.sample_q_host_mass_ratio(np.linspace(1e-6, 1.0 - 1e-6, hosts.size), hosts)

    assert np.all((q > 0.0) & (q <= 1.0))
    # Exact: a tolerance here once hid secondaries one ulp below the boundary.
    assert np.all(q * hosts >= priors.MOE2017_Q_BREAK_LOW)


def _hosts_whose_naive_floor_rounds_short(seed: int = 29, size: int = 20000) -> np.ndarray:
    """Sub-solar primaries for which ``(0.1 / M) * M`` falls below 0.1 in binary64."""
    hosts = np.random.default_rng(seed).uniform(
        priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR, 1.0, size
    )
    naive = (priors.MOE2017_Q_BREAK_LOW / hosts) * hosts
    short = hosts[naive < priors.MOE2017_Q_BREAK_LOW]
    assert short.size > 0, "the synthetic sample must exercise the rounding case"
    return short


def test_mass_ratio_floor_is_the_smallest_ratio_keeping_the_secondary_stellar():
    """The floor excludes only ratios whose secondary is below the stellar boundary."""
    hosts = _hosts_whose_naive_floor_rounds_short()
    floors = priors._stellar_mass_ratio_floors(hosts)

    assert np.all(floors * hosts >= priors.MOE2017_Q_BREAK_LOW)
    assert np.all(np.nextafter(floors, -np.inf) * hosts < priors.MOE2017_Q_BREAK_LOW)


def test_scalar_mass_ratio_prior_resolves_the_exact_secondary_mass_boundary():
    """The low-mass CDF has support at the inclusive q-floor boundary."""
    from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR
    from trex.stellar.stellar_relations import stellar_relations

    host_mass = priors.MOE2017_Q_DOMAIN_MINIMUM_PRIMARY_MASS_SOLAR
    quantiles = np.linspace(0.0, 1.0, 101)
    ratios = priors.sample_q(quantiles, host_mass)
    secondary_masses = ratios * host_mass

    assert np.all(np.isfinite(ratios))
    assert np.all((ratios >= priors.MOE2017_Q_TWIN_THRESHOLD) & (ratios <= 1.0))
    assert np.all(secondary_masses >= STELLAR_RELATIONS_MASS_MIN_SOLAR)
    radii, temperatures = stellar_relations(secondary_masses)
    assert np.all(np.isfinite(radii)) and np.all(np.isfinite(temperatures))


def test_no_eclipsing_secondary_support_below_the_mass_ratio_domain_floor():
    """A 0.1-solar-mass primary cannot enter the Moe q prior or its CDF."""
    with pytest.raises(ValueError, match="no valid q support"):
        priors.sample_q(np.array([0.0, 0.5, 1.0]), priors.MOE2017_Q_BREAK_LOW)


def test_floor_quantile_secondaries_stay_inside_the_stellar_relation_domain():
    """Draws at the proposal's bisection floor must evaluate, not crash the vet.

    A collapsed importance window places draws at the smallest bisection
    quantile, where the inverse CDF returns the floor itself.
    """
    from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR
    from trex.stellar.stellar_relations import stellar_relations
    from trex.evidence.marginal_likelihoods import _QUANTILE_BISECTION_ITERATIONS

    hosts = _hosts_whose_naive_floor_rounds_short()
    for quantile in (0.0, 2.0 ** -(_QUANTILE_BISECTION_ITERATIONS + 1)):
        u = np.full(hosts.size, quantile)
        secondary = priors.sample_q_host_mass_ratio(u, hosts) * hosts
        assert np.all(secondary >= STELLAR_RELATIONS_MASS_MIN_SOLAR)
        stellar_relations(secondary)

        companion = np.array(
            [
                priors.sample_q_companion(np.array([quantile]), float(host), np.array([3.0]))[0] * host
                for host in hosts[:500]
            ]
        )
        assert np.all(companion >= STELLAR_RELATIONS_MASS_MIN_SOLAR)


def test_eclipsing_pair_with_a_collapsed_proposal_window_evaluates(monkeypatch):
    """Production boundary: a planet-sized depth on a host whose floor rounds short.

    The smallest stellar secondary is larger than the observed occultor, so
    the importance window collapses onto the mass-ratio floor. Every such draw
    must still reach the likelihood instead of raising from the mass-radius
    relation.
    """
    import trex.evidence.marginal_likelihoods as marginal_likelihoods
    import trex.evidence.mc_scenario as mc_scenario
    from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR
    from trex.stellar.stellar_relations import stellar_relations

    host_mass = float(_hosts_whose_naive_floor_rounds_short()[0])
    host_radius = float(stellar_relations(np.array([host_mass]))[0][0])
    smallest_ratio = float(stellar_relations(np.array([STELLAR_RELATIONS_MASS_MIN_SOLAR]))[0][0]) / host_radius
    # An occultor a tenth the size of the smallest stellar secondary.
    depth_fraction = (smallest_ratio / 10.0) ** 2
    evaluated = []
    monkeypatch.setattr(
        mc_scenario,
        "lnL_EB",
        lambda *args, **_kwargs: evaluated.append(args[3]) or 0.0,
    )
    monkeypatch.setattr(mc_scenario, "HAS_JAX_TREX", False)

    n_draws = 64
    lnz = marginal_likelihoods._eval_scenario(
        np.array([0.0]), np.array([1.0]), 0.01, 5.0, host_mass, host_radius, 0.4, 0.2,
        n_draws, is_planet=False, is_EB=True, use_2xP=False, exptime_days=0.01,
        rng=np.random.default_rng(3), host_depth_fraction=depth_fraction,
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )

    assert not math.isnan(lnz)
    assert evaluated and np.all(np.isfinite(evaluated))


def test_host_mass_ratio_matches_sample_q_for_a_uniform_host():
    """The per-draw floor must not change the distribution for one host mass."""
    x = np.linspace(1e-6, 1.0 - 1e-6, 5000)
    uniform = priors.sample_q_host_mass_ratio(x, np.full(x.size, 1.0))
    reference = priors.sample_q(x.copy(), M_s=1.0)

    np.testing.assert_allclose(uniform, reference, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# The scenario evidences must marginalize the population, not read neighbours
# ---------------------------------------------------------------------------

_BOUND_COMPANION_SCENARIOS = slice(3, 9)   # PTP, PEB, PEBx2P, STP, SEB, SEBx2P


def _nearby_stars(aperture_flux_fraction):
    """One resolvable synthetic neighbour, or none for the absent case."""
    if aperture_flux_fraction is None:
        return None
    return [{
        "source_id": "synthetic-neighbor",
        "M_s": 0.6,
        "R_s": 0.6,
        "aperture_flux_fraction": aperture_flux_fraction,
    }]


def _evidences(aperture_flux_fraction):
    from trex.evidence.marginal_likelihoods import calc_target_evidences

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    lnZ, _ = calc_target_evidences(
        time=time,
        flux=flux,
        sigma=1e-4,
        period_days=4.0,
        depth_ppm=1000.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.4,
        u2=0.2,
        exptime_days=0.001,
        target_aperture_flux_fraction=1.0,
        n_draws=64,
        contrast_separations=np.array([0.2, 1.0, 4.0]),
        contrast_values=np.array([2.0, 5.0, 8.0]),
        nearby_stars=_nearby_stars(aperture_flux_fraction),
        plx=10.0,
        random_seed=3,
        nsamples=5,
        target_feh=-0.2,
    )
    return lnZ


def test_bound_companion_evidences_ignore_resolved_neighbour_contrasts():
    """A resolved catalog neighbour is not an unresolved bound companion.

    The P/S scenarios marginalize over a drawn companion population, so
    changing the resolved neighbour's aperture flux fraction -- or removing it
    entirely -- must leave those six evidences untouched.
    """
    faint = _evidences(0.01)
    bright = _evidences(0.05)
    absent = _evidences(None)

    np.testing.assert_allclose(
        faint[_BOUND_COMPANION_SCENARIOS], bright[_BOUND_COMPANION_SCENARIOS]
    )
    np.testing.assert_allclose(
        faint[_BOUND_COMPANION_SCENARIOS], absent[_BOUND_COMPANION_SCENARIOS]
    )
    assert np.all(np.isfinite(faint[_BOUND_COMPANION_SCENARIOS]))


def test_bound_companion_evidences_require_a_measured_contrast_curve():
    """Without imaging there is nothing bounding the unresolved population."""
    from trex.evidence.marginal_likelihoods import calc_target_evidences

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    lnZ, _ = calc_target_evidences(
        time=time,
        flux=flux,
        sigma=1e-4,
        period_days=4.0,
        depth_ppm=1000.0,
        M_s_Msun=1.0,
        R_s_Rsun=1.0,
        u1=0.4,
        u2=0.2,
        exptime_days=0.001,
        target_aperture_flux_fraction=1.0,
        n_draws=32,
        plx=10.0,
        random_seed=3,
        nsamples=5,
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
    )

    assert np.all(np.isneginf(lnZ[_BOUND_COMPANION_SCENARIOS]))


def test_bound_companion_draws_are_not_a_degenerate_estimator():
    """Marginalizing by sampling only works if enough draws carry weight.

    This model replaces an analytic occurrence integral with a Monte Carlo one,
    so the fraction of draws that survive the imaging-detection and
    already-resolved cuts sets the estimator's variance. For a representative
    solar-type target with a 0.2-4 arcsec contrast curve at 10 mas parallax the
    retained fraction must be a substantial part of the sample, not a handful
    of draws.
    """
    from trex.evidence.marginal_likelihoods import _draw_bound_companion

    n_draws = 20000
    _, lnprior, _, _, _ = _draw_bound_companion(
        np.random.default_rng(17),
        n_draws,
        1.0,
        1.0,
        10.0,
        np.array([0.2, 1.0, 4.0]),
        np.array([2.0, 5.0, 8.0]),
    )
    retained = float(np.mean(np.isfinite(lnprior)))

    assert 0.05 < retained < 1.0
    # Every retained draw carries the same occurrence weight, which is a
    # probability and therefore a non-positive log.
    finite = lnprior[np.isfinite(lnprior)]
    assert np.all(finite < 0.0)
    assert float(np.ptp(finite)) == 0.0


def test_measured_curve_outer_edge_does_not_truncate_unresolved_companions(monkeypatch):
    """Uncovered companions remain in the prior through the Gaia boundary."""
    from trex.foundation.constants import GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC
    from trex.foundation.physical_constants import MILLIARCSECONDS_PER_ARCSECOND
    from trex.evidence import marginal_likelihoods, mc_population

    host_mass = 1.0
    parallax_mas = 10.0
    angular_separations = np.array([1.5, GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC + 0.3])
    rng = np.random.default_rng(91)
    rng.random(2)  # Period quantiles precede the mass-ratio and sky-projection variates.
    rng.random(2)
    projection = np.sqrt(1.0 - rng.random(2) ** 2)
    log_separation_au = np.log10(
        angular_separations * MILLIARCSECONDS_PER_ARCSECOND / (parallax_mas * projection)
    )
    # Kepler's law uses both masses; a circular sky projection gives the specified separations.
    log_periods = priors._companion_log_period_from_log_separation(
        1.5 * host_mass, log_separation_au
    )
    monkeypatch.setattr(
        mc_population,
        "sample_companion_log_period",
        lambda _quantiles, _mass: log_periods.copy(),
    )
    monkeypatch.setattr(
        mc_population, "sample_q_companion",
        lambda quantiles, *_args: np.full_like(quantiles, 0.5),
    )

    _, log_prior, _, _, _ = marginal_likelihoods._draw_bound_companion(
        np.random.default_rng(91),
        angular_separations.size,
        host_mass,
        1.0,
        parallax_mas,
        np.array([0.1, 1.0]),
        np.array([0.0, 0.0]),
    )

    assert np.isfinite(log_prior[0]), "outside the measured curve but inside Gaia remains unresolved"
    assert np.isneginf(log_prior[1]), "outside Gaia resolution is not double-counted as unresolved"


def test_secondary_hosted_scenarios_use_the_companion_geometry(monkeypatch):
    """STP/SEB place the transit on the companion, not on the target star."""
    from trex.evidence import marginal_likelihoods, mc_scenario

    captured = {}
    real = marginal_likelihoods._draw_bound_companion

    def spy(rng, n_draws, M_s, R_s, plx, seps, contrasts, **kwargs):
        result = real(rng, n_draws, M_s, R_s, plx, seps, contrasts, **kwargs)
        captured["companion_radius"] = result[3]
        return result

    monkeypatch.setattr(mc_scenario, "_draw_bound_companion", spy)

    seen = []
    original = mc_scenario._eval_scenario

    def record(*args, **kwargs):
        if kwargs.get("bound_companion") == "secondary":
            seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(mc_scenario, "_eval_scenario", record)

    _evidences(None)

    assert len(seen) == 3
    assert all(entry["companion_is_host"] is True for entry in seen)
    # The host geometry varies draw by draw and is drawn from a population of
    # mostly lower-mass companions, so it cannot be the target's fixed radius.
    radii = captured["companion_radius"]
    assert float(np.ptp(radii)) > 0.0
    assert float(np.median(radii)) < 1.0


@pytest.mark.parametrize("primary_mass", [0.09, 0.3])
def test_bound_scenarios_below_the_stellar_companion_floor_are_recorded_unavailable(primary_mass):
    """A 0.1-Msun stellar companion cannot have q <= 1 around a 0.09-Msun primary.

    The gate must record the population prior's own reason before sampling instead of
    raising inside the draw; a supported primary keeps all six slots evaluated.
    """
    from trex.evidence.marginal_likelihoods import calc_target_evidences

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    fits = {}
    lnZ, _ = calc_target_evidences(
        time=time, flux=flux, sigma=1e-4, period_days=4.0, depth_ppm=1000.0,
        M_s_Msun=primary_mass, R_s_Rsun=0.12 if primary_mass < 0.1 else 0.3, u1=0.4, u2=0.2,
        exptime_days=0.001, target_aperture_flux_fraction=1.0, n_draws=8, plx=10.0, random_seed=3,
        nsamples=5, target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX, target_teff_k=2800.0 if primary_mass < 0.1 else 3400.0,
        contrast_separations=np.array([0.1, 0.5, 2.0]), contrast_values=np.array([2.0, 4.0, 6.0]),
        representative_fit_out=fits,
    )

    bound = range(3, 9)
    if primary_mass < 0.1:
        assert np.all(np.isneginf(lnZ[_BOUND_COMPANION_SCENARIOS]))
        assert {fits[index]["status"] for index in bound} == {"unavailable"}
        from trex.populations.companion_priors import companion_population_frequency

        with pytest.raises(ValueError) as owner:
            companion_population_frequency(primary_mass)
        assert {fits[index]["reason"] for index in bound} == {str(owner.value)}
        # The standalone summary rows and the report-level list carry the same record.
        from trex.app.engine import scenario_monte_carlo_evidence, unavailable_scenario_records
        from trex.evidence.diagnostics import generate_diagnostics

        result = generate_diagnostics(lnZ, scenario_fits=fits)
        rows = {row["scenario"]: row for row in scenario_monte_carlo_evidence(result)}
        names = ("PTP", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P")
        assert all(rows[name]["status"] == "unavailable" and rows[name]["reason"] == str(owner.value) for name in names)
        assert rows["TP"].get("status") != "unavailable"
        # The fixture supplies no background population, so D*/B* are disclosed too, naming those inputs.
        background = ("DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P")
        disclosed = {record["scenario"]: record["reason"] for record in unavailable_scenario_records(result)}
        assert list(disclosed) == [*names, *background]
        assert all("background population" in disclosed[name] for name in background)
        assert np.isfinite(lnZ[0])  # TP still evaluated
    else:
        assert all(fits.get(index, {}).get("status") != "unavailable" for index in bound)


_P_BOUND = ("PTP", "PEB", "PEBx2P")
_S_BOUND = ("STP", "SEB", "SEBx2P")
_BACKGROUND = ("DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P")


@pytest.mark.parametrize(
    ("missing", "excluded", "reason_word"),
    [
        ({"plx": None}, _P_BOUND + _S_BOUND, "parallax"),
        ({"target_feh": None}, _S_BOUND, "metallicity"),
        # One curve point is no angular constraint, and no resolution limit is given:
        # every unresolved population (bound and background) is excluded.
        ({"contrast_separations": np.array([0.5]), "contrast_values": np.array([4.0])},
         _P_BOUND + _S_BOUND + _BACKGROUND, "angular"),
    ],
)
def test_each_missing_input_is_disclosed_as_unavailable_with_its_reason(missing, excluded, reason_word):
    """An excluded scenario must reach the report-level list with the missing input named.

    The FPP silently dropping scenarios was the defect; the other scenarios' evidence must not move.
    """
    from trex.evidence.diagnostics import generate_diagnostics
    from trex.evidence.marginal_likelihoods import calc_target_evidences
    from trex.app.engine import unavailable_scenario_records

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    common = dict(
        time=time, flux=flux, sigma=1e-4, period_days=4.0, depth_ppm=1000.0, M_s_Msun=0.5, R_s_Rsun=0.5,
        u1=0.4, u2=0.2, exptime_days=0.001, target_aperture_flux_fraction=1.0, n_draws=8, random_seed=3,
        nsamples=5, target_teff_k=3800.0, plx=10.0, target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
        contrast_separations=np.array([0.1, 0.5, 2.0]), contrast_values=np.array([2.0, 4.0, 6.0]),
    )
    full_fits, fits = {}, {}
    full, _ = calc_target_evidences(**common, representative_fit_out=full_fits)
    lnZ, _ = calc_target_evidences(**{**common, **missing}, representative_fit_out=fits)

    disclosed = {r["scenario"]: r["reason"] for r in unavailable_scenario_records(generate_diagnostics(lnZ, scenario_fits=fits))}
    for name in excluded:
        assert reason_word in disclosed[name].lower(), (name, disclosed.get(name))
    # Any scenario the missing input removes (finite with it, -inf without it) must be disclosed too.
    from trex.evidence.marginal_likelihoods import _SCENARIO_NAMES
    removed = [_SCENARIO_NAMES[i] for i in range(len(lnZ)) if np.isfinite(full[i]) and not np.isfinite(lnZ[i])]
    assert set(removed) <= set(disclosed), sorted(set(removed) - set(disclosed))
    # Scenarios still evaluated must keep bit-identical evidence: disclosure changes no arithmetic.
    evaluated = [i for i in range(len(lnZ)) if np.isfinite(lnZ[i])]
    assert evaluated, "the target scenarios must stay evaluated"
    np.testing.assert_array_equal(lnZ[evaluated], full[evaluated])


@pytest.mark.parametrize(
    ("missing", "retired", "draws"),
    [
        # Parallax gates the bound branch only; the background families still
        # evaluate, so B* runs at the draw count proven for a feh-less scene.
        ({"plx": None}, _P_BOUND + _S_BOUND, 64),
        ({"background_population": None, "background_area_sq_deg": None}, _BACKGROUND, 8),
        ({"plx": None, "background_population": None, "background_area_sq_deg": None},
         _P_BOUND + _S_BOUND + _BACKGROUND, 8),
        # No metallicity anywhere: the target EBs, the hosted S* rows and the
        # resolved neighbour's scenarios retire beside the population families.
        # The neighbour rows are disclosed under their diagnostic keys N15-N17.
        ({"target_feh": None,
          "background_population": None, "background_area_sq_deg": None,
          "nearby_stars": [{"source_id": "synthetic-neighbor", "M_s": 0.6, "R_s": 0.6,
                            "aperture_flux_fraction": 0.5}]},
         ("EB", "EBx2P", "PEB", "PEBx2P", "STP", "SEB", "SEBx2P",
          "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P",
          "N15", "N16", "N17"), 8),
    ],
)
def test_retired_families_keep_the_run_status_ok(missing, retired, draws):
    """With scenario records, retirement does not change the whole-result status.

    The 2026-11-14 admission refactor coupled every missing-input retirement to
    a run-level ``unavailable`` (mc_target), so a scene without a parallax or a
    background population returned no usable run even though its remaining
    families were fully evaluated; the documented-contract tests (progress, host
    feasibility, aperture dilution) went red on 2026-11-18.
    Retirement is disclosed per scenario record and the report layer refuses
    an FPP on those records (``scenario-input-unavailable``); the run status
    itself must stay ``ok``.
    """
    from trex.app.engine import unavailable_scenario_records
    from trex.evidence.diagnostics import generate_diagnostics
    from trex.evidence.marginal_likelihoods import (
        N_NEIGHBOR_SCENARIOS,
        N_TARGET_SCENARIOS,
        _SCENARIO_NAMES,
        calc_target_evidences,
    )

    time = np.linspace(-0.15, 0.15, 60)
    flux = np.ones_like(time)
    flux[25:35] = 0.999
    common = dict(
        time=time, flux=flux, sigma=1e-4, period_days=4.0, depth_ppm=1000.0, M_s_Msun=0.5, R_s_Rsun=0.5,
        u1=0.4, u2=0.2, exptime_days=0.001, target_aperture_flux_fraction=1.0,
        random_seed=3, nsamples=5, target_teff_k=3800.0, plx=10.0,
        target_feh=SYNTHETIC_SOLAR_METALLICITY_DEX,
        contrast_separations=np.array([0.1, 0.5, 2.0]), contrast_values=np.array([2.0, 4.0, 6.0]),
        target_tess_mag=12.0, background_area_sq_deg=0.1,
        background_population={
            "tess_mag": np.array([14.0, 16.0, 18.0]),
            "mass_solar": np.array([0.9, 0.6, 0.3]),
            "logg_cgs": np.array([4.4, 4.6, 4.9]),
            "teff_k": np.array([5400.0, 4200.0, 3400.0]),
            "radius_solar": np.array([0.95, 0.60, 0.30]),
            "mh_dex": np.array([0.0, -0.5, 0.5]),
        },
    )
    fits: dict = {}
    lnZ, status = calc_target_evidences(
        **{**common, **missing, "n_draws": draws}, representative_fit_out=fits)

    assert status == "ok", (missing, status)
    assert np.isfinite(lnZ[0]), "TP stays evaluated in every case"
    if "target_feh" not in missing:
        without_records, status = calc_target_evidences(**{**common, **missing, "n_draws": draws})
        assert status == "unavailable", (missing, status)
        np.testing.assert_array_equal(without_records, lnZ)
    index_of = {name: index for index, name in _SCENARIO_NAMES.items()}
    if "nearby_stars" in missing:
        index_of.update({f"N{N_TARGET_SCENARIOS + offset}": N_TARGET_SCENARIOS + offset
                         for offset in range(N_NEIGHBOR_SCENARIOS)})
    disclosed = {record["scenario"]: record["reason"] for record in unavailable_scenario_records(
        generate_diagnostics(lnZ, scenario_fits=fits))}
    for name in retired:
        index = index_of[name]
        assert np.isneginf(lnZ[index]), (name, lnZ[index])
        assert name in disclosed, (name, sorted(disclosed))
