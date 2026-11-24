"""Native-time noise calibration with declared event windows.

These are synthetic controls: the injected event phase, depth and noise
realization are known, so the expected change in the binned-RMS factor does not
come from the estimator itself. They verify that a declared phase/duration is
excluded from calibration while every point remains available to the likelihood.
"""

import math

import numpy as np
import pytest

from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.noise import calibrate_trex_noise
from trex.models.red_noise import out_of_transit_mask

PERIOD_DAYS = 2.0
DURATION_DAYS = 0.1
EPOCH_BTJD = 1000.0
CADENCE_DAYS = 0.002
SIGMA = 200e-6
ORBITS = 20
DEPTH_SIGMAS = 20
SEED = 20261114


def _series(*, secondary_phase=None, secondary_duration=DURATION_DAYS, secondary_depth=0, correlated=False):
    """Primary box plus an optional secondary box on a regular native cadence."""
    time = EPOCH_BTJD + np.arange(ORBITS * PERIOD_DAYS / CADENCE_DAYS) * CADENCE_DAYS
    phase = (time - EPOCH_BTJD + PERIOD_DAYS / 2) % PERIOD_DAYS - PERIOD_DAYS / 2
    flux = 1.0 - (np.abs(phase) <= DURATION_DAYS / 2) * DEPTH_SIGMAS * SIGMA
    if secondary_depth:
        centre = (secondary_phase * PERIOD_DAYS + PERIOD_DAYS / 2) % PERIOD_DAYS - PERIOD_DAYS / 2
        distance = (phase - centre + PERIOD_DAYS / 2) % PERIOD_DAYS - PERIOD_DAYS / 2
        flux -= (np.abs(distance) <= secondary_duration / 2) * secondary_depth * SIGMA
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=time.size)
    if correlated:
        rho = math.exp(-CADENCE_DAYS / DURATION_DAYS)
        scale = math.sqrt(1.0 - rho**2)
        for index in range(1, noise.size):
            noise[index] = rho * noise[index - 1] + scale * noise[index]
    return time, flux + SIGMA * noise, np.full(time.size, SIGMA), np.ones(time.size, dtype=int)


def _calibrate(series, **kwargs):
    time, flux, sigma, sectors = series
    return calibrate_trex_noise(
        time, flux, sigma, sectors,
        period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD, duration_days=DURATION_DAYS,
        cadence_days_by_sector={1: CADENCE_DAYS},
        **kwargs,
    )


def test_default_mask_still_excludes_only_the_circular_half_phase():
    time, *_ = _series()
    record = _calibrate(_series())
    expected = np.count_nonzero(
        out_of_transit_mask(time, period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD,
                            duration_days=DURATION_DAYS, exclusion_factor=(DURATION_DAYS + CADENCE_DAYS) / DURATION_DAYS)
        & out_of_transit_mask(time, period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD + PERIOD_DAYS / 2,
                              duration_days=DURATION_DAYS, exclusion_factor=(DURATION_DAYS + CADENCE_DAYS) / DURATION_DAYS)
    )
    assert record["secondary_mask"] == "circular-half-phase-window"
    assert record["sectors"]["1"]["out_of_transit_sample_count"] == expected
    assert [window["role"] for window in record["event_windows"]] == ["primary", "circular-half-phase"]


def test_declared_non_half_phase_window_removes_an_eccentric_secondary_from_the_noise():
    window = [{"phase": 0.3, "duration_days": DURATION_DAYS, "role": "eccentric-secondary"}]
    plain = _calibrate(_series(), event_windows=window)
    eclipsing = _calibrate(
        _series(secondary_phase=0.3, secondary_depth=DEPTH_SIGMAS), event_windows=window
    )
    # The event is wholly inside the declared window, so the retained OOT sample
    # is byte-identical and the calibration record must not change.
    assert eclipsing == plain
    assert plain["secondary_mask"] == "caller-declared-event-windows"
    assert plain["event_windows"][1]["phase"] == 0.3

    # Without the declaration the phase-0.3 secondary sits in the OOT sample and
    # inflates the factor: the circular default is a stated limitation, not a
    # general eccentric-orbit solution.
    undeclared = _calibrate(_series(secondary_phase=0.3, secondary_depth=DEPTH_SIGMAS))
    assert undeclared["secondary_mask"] == "circular-half-phase-window"
    assert undeclared["native_uncertainty_factor"] > plain["native_uncertainty_factor"]


def test_declared_duration_removes_a_secondary_wider_than_the_primary_window():
    long_duration = 2.0 * DURATION_DAYS
    window = [{"phase": 0.5, "duration_days": long_duration, "role": "wide-secondary"}]
    plain = _calibrate(_series(), event_windows=window)
    eclipsing = _calibrate(
        _series(secondary_phase=0.5, secondary_duration=long_duration, secondary_depth=DEPTH_SIGMAS),
        event_windows=window,
    )
    assert eclipsing == plain
    # The default primary-duration half-phase window clips the long event and
    # leaves its wings in the calibration sample.
    clipped = _calibrate(
        _series(secondary_phase=0.5, secondary_duration=long_duration, secondary_depth=DEPTH_SIGMAS)
    )
    assert clipped["native_uncertainty_factor"] > plain["native_uncertainty_factor"]


def test_primary_only_white_control_keeps_the_factor_near_unity():
    record = _calibrate(_series())
    assert 1.0 <= record["native_uncertainty_factor"] < 1.2
    assert record["status"] == "passed"


def test_declared_events_do_not_change_correlated_noise_calibration():
    window = [{"phase": 0.27, "duration_days": DURATION_DAYS, "role": "declared-secondary"}]
    plain = _calibrate(_series(correlated=True), event_windows=window)
    eclipsing = _calibrate(
        _series(secondary_phase=0.27, secondary_depth=DEPTH_SIGMAS, correlated=True),
        event_windows=window,
    )
    assert eclipsing == plain
    assert plain["sectors"]["1"]["beta"] >= 1.0
    assert plain["sectors"]["1"]["bin_size_points"] == round(DURATION_DAYS / CADENCE_DAYS)


@pytest.mark.parametrize("windows", [
    [{"phase": 1.0, "duration_days": DURATION_DAYS}],
    [{"phase": -0.1, "duration_days": DURATION_DAYS}],
    [{"phase": float("nan"), "duration_days": DURATION_DAYS}],
    [{"phase": 0.2, "duration_days": 0.0}],
    [{"phase": 0.2, "duration_days": PERIOD_DAYS / 2}],
    [{"phase": 0.2, "duration_days": DURATION_DAYS, "role": ""}],
    [],
    {"phase": 0.2},
    "not-a-window",
])
def test_invalid_event_windows_are_refused(windows):
    with pytest.raises(ValueError):
        _calibrate(_series(), event_windows=windows)


def test_short_duration_refusal_survives_declared_windows():
    with pytest.raises(TrexEvidenceUnavailableError) as excinfo:
        calibrate_trex_noise(
            *_series(), period_days=PERIOD_DAYS, epoch_btjd=EPOCH_BTJD,
            duration_days=0.5 * CADENCE_DAYS, cadence_days_by_sector={1: CADENCE_DAYS},
            event_windows=[{"phase": 0.3, "duration_days": 0.5 * CADENCE_DAYS}],
        )
    # str(error) is only the message; the durable reason lives in diagnostics.
    record = excinfo.value.diagnostics["noise_model"]
    assert record["status"] == "deferred"
    assert record["native_uncertainty_factor"] is None
    assert record["sectors"]["1"]["reason"] == "insufficient native cadence resolution at T14"
