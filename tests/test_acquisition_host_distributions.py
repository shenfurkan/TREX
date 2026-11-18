"""SHBoost measures checked against independent solar and truncated-normal values."""

import math

import numpy as np
import pytest
from astropy import constants as const
from scipy import stats

from trex.acquisition.host_distributions import shboost_distribution, shboost_entry, shboost_target_values

SOLAR_LOGG = math.log10((const.G * const.M_sun / const.R_sun**2).cgs.value)
SOLAR_LOGTEFF = math.log10(5772.0)


def _shboost(**overrides):
    row = {
        "xgbdist_mass_mean": 1.0, "xgbdist_mass_std": 0.1,
        "xgbdist_logg_mean": SOLAR_LOGG, "xgbdist_logg_std": 0.05,
        "xgbdist_logteff_mean": SOLAR_LOGTEFF, "xgbdist_logteff_std": 0.01,
        "xgb_inputflag": "XP", "xgb_mass_outputflag": "0",
    }
    row.update(overrides)
    return row


def test_shboost_solar_gravity_and_mass_recover_one_solar_radius():
    tight = shboost_distribution(_shboost(xgbdist_mass_std=1e-6, xgbdist_logg_std=1e-6, xgbdist_logteff_std=1e-6))
    np.testing.assert_allclose(tight["radius_solar"], 1.0, rtol=1e-5)
    np.testing.assert_allclose(tight["teff_k"], 5772.0, rtol=1e-5)
    assert math.isclose(sum(tight["weights"]), 1.0, rel_tol=1e-12)


def test_shboost_mass_is_a_normal_conditioned_on_positive_mass():
    mean, sigma = 0.2, 0.2
    measure = shboost_distribution(_shboost(xgbdist_mass_mean=mean, xgbdist_mass_std=sigma, xgb_mass_outputflag="1"))
    mass, weights = np.asarray(measure["mass_solar"]), np.asarray(measure["weights"])
    assert np.all(mass > 0)
    reference = stats.truncnorm(-mean / sigma, np.inf, loc=mean, scale=sigma)
    assert np.sum(weights * mass) == pytest.approx(reference.mean(), rel=5e-3)
    assert np.sum(weights * mass**2) - np.sum(weights * mass)**2 == pytest.approx(reference.var(), rel=5e-2)


@pytest.mark.parametrize("overrides", [
    {"xgb_inputflag": "G"}, {"xgbdist_mass_std": -0.1}, {"xgbdist_mass_mean": None},
    {"xgbdist_logg_mean": float("nan")},
], ids=["no-xp", "negative-error", "missing-mass", "nonfinite-gravity"])
def test_shboost_refuses_rows_without_a_usable_measurement(overrides):
    with pytest.raises(ValueError):
        shboost_distribution(_shboost(**overrides))


def test_shboost_row_normalization_keeps_broad_errors_and_validates_flags():
    row = _shboost(source_id="123", xgb_mass_outputflag="1", xgb_logg_outputflag="0",
                   xgb_logteff_outputflag="0", xgbdist_met_mean="-0.2")
    entry = shboost_entry(row)
    assert entry["values"]["xgb_mass_outputflag"] == "1"
    assert entry["values"]["xgbdist_met_mean"] == -0.2
    assert "source_id" not in shboost_target_values(row)
    with pytest.raises(ValueError, match="flag"):
        shboost_entry({**row, "xgb_logg_outputflag": "2"})
def test_gaia_and_starhorse_rows_preserve_units_and_missing_values():
    import pytest
    from astropy import constants as const, units as u
    from trex.acquisition.host_distributions import gaia_ap_entry, starhorse_entry

    gaia = gaia_ap_entry({"mass_flame": 1.0, "radius_flame": None, "evolstage_flame": 2.0})
    assert gaia["mass_flame_solar"] == 1.0 and gaia["radius_flame_solar"] is None
    assert gaia["evolstage_flame"] == 2
    gravity = (const.G * const.M_sun / const.R_sun**2).to_value(u.cm / u.s**2)
    import math
    star = starhorse_entry({"mass50": 1.0, "logg50": math.log10(gravity)})
    assert star["radius_solar"] == pytest.approx(1.0)
    assert star["radius_lower_solar"] is None and star["radius_upper_solar"] is None


def test_starhorse_domain_requires_fidelity_and_relevant_spread():
    from trex.acquisition.host_distributions import starhorse_domain_refusal

    row = {"mass_solar": 1., "radius_solar": 1., "astrometric_fidelity": 0.9,
           "sh_outflag": "0010", "sh_photoflag": "valid", "mass_lower_solar": 0.9,
           "mass_upper_solar": 1.1, "logg_lower_cgs": 4.3, "logg_upper_cgs": 4.5}
    assert starhorse_domain_refusal(row) is None
    assert "fidelity" in starhorse_domain_refusal({**row, "astrometric_fidelity": 0.5})
    assert "large uncertainties" in starhorse_domain_refusal({**row, "mass_upper_solar": 3.0})


def test_starhorse_radius_holds_for_zero_and_negative_log_g():
    from trex.acquisition.host_distributions import starhorse_entry

    solar_gravity = (const.G * const.M_sun / const.R_sun**2).cgs.value
    for logg in (0.0, -0.3):
        entry = starhorse_entry({"mass50": 2.0, "logg50": logg, "mass16": 1.5, "mass84": 2.5,
                                 "logg16": logg - 0.1, "logg84": logg + 0.1})
        assert entry["radius_solar"] == pytest.approx(math.sqrt(2.0 * solar_gravity / 10.0**logg), rel=1e-12)
        assert entry["radius_lower_solar"] < entry["radius_solar"] < entry["radius_upper_solar"]


@pytest.mark.parametrize("value", [None, True, "nan", "x"])
def test_shboost_gravity_must_be_a_finite_number(value):
    with pytest.raises(ValueError, match="xgbdist_logg_mean"):
        shboost_distribution(_shboost(xgbdist_logg_mean=value))


def test_distribution_values_leave_the_input_untouched_and_report_the_same_percentiles():
    from trex.acquisition.host_distributions import distribution_values

    distribution = shboost_distribution(_shboost())
    before = dict(distribution)
    values = distribution_values(distribution, {"source": "shboost-xp-marginals"})
    assert distribution == before
    reference = stats.truncnorm(-1.0 / 0.1, np.inf, loc=1.0, scale=0.1)
    low, high = reference.ppf([0.16, 0.84])
    assert values["stellar_parameter_distribution"]["mass_solar_p16_p84"] == pytest.approx([low, high], rel=1e-12)
    assert values["mass_solar_err_lower"] == pytest.approx(reference.median() - low, rel=1e-12)
    assert values["mass_solar_err_upper"] == pytest.approx(high - reference.median(), rel=1e-12)
    assert values["radius_solar_err_lower"] > 0 and values["radius_solar_err_upper"] > 0


def test_weighted_median_of_a_symmetric_grid_is_its_centre():
    from trex.acquisition.host_distributions import weighted_quantiles

    assert weighted_quantiles([1.0, 2.0, 3.0, 4.0], [0.25] * 4, [0.5])[0] == pytest.approx(2.5)


def test_neighbor_record_states_why_shboost_was_not_used():
    from trex.acquisition.host_distributions import neighbor_parameter_entry

    entry = neighbor_parameter_entry(None, None, None, "SHBoost output flag missing", None,
                                     {"status": "available"}, {"reason": "timeout"})
    assert entry["status"] == "unavailable" and entry["shboost_refusal"] == "SHBoost output flag missing"
    assert entry["reason"] == ("gaia-dr3-astrophysical-parameters: no values; "
                               "starhorse-edr3: query unavailable (timeout); shboost-xp: SHBoost output flag missing")


def test_host_measure_medians_are_exact_or_midpoint_weighted():
    from trex.evidence.host_measure import HostMeasure

    measure = HostMeasure(shboost_distribution(_shboost()), 0.0)
    assert measure.marginal_median("teff_k") == pytest.approx(5772.0, rel=1e-12)
    assert measure.marginal_median("mass_solar") == pytest.approx(
        stats.truncnorm(-10.0, np.inf, loc=1.0, scale=0.1).median(), rel=1e-12)
    grid = {"method": "grid", "mass_solar": [1.0, 2.0, 3.0, 4.0], "radius_solar": [1.0] * 4,
            "teff_k": [4000.0, 5000.0, 6000.0, 7000.0], "weights": [0.25] * 4}
    assert HostMeasure(grid, 0.0).marginal_median("teff_k") == pytest.approx(5500.0)
