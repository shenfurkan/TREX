"""Execute the acquisition example's generated code cells on synthetic TESS curves.

The archive download and ``from_tic`` acquisition are replaced at their external
boundary; the fold, binning, noise calibration and handoff are the generator's
actual cell source. The controls inject known event phases and depths, so
retention and depth transfer are checked without a network request.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pytest
from astropy.time import Time

GENERATOR = Path(__file__).resolve().parents[1] / "examples" / "make_from_tic_id.py"
NOTEBOOK = GENERATOR.with_name("from_tic_id.ipynb")
DURATION_DAYS = 2.4927338 / 24.0
MODULE = None


def _module():
    global MODULE
    if MODULE is None:
        spec = importlib.util.spec_from_file_location("trex_make_from_tic_id", GENERATOR)
        MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MODULE)
    return MODULE


def _cells():
    return [source for kind, source in _module().CELLS if kind == "code"]


def test_notebook_matches_its_generator():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]] == list(
        _module().CELLS
    )


class _SearchResult:
    def __init__(self, curves):
        self._curves = list(curves)

    def download_all(self, flux_column):
        assert flux_column == "pdcsap_flux"
        return self._curves


def _curves(*, secondary_phase=None, secondary_depth=0.0, secondary_duration=None,
            sectors=(2, 3), gap_phase=0.72):
    period = 4.5320502
    epoch = 2458358.1182630 - 2457000.0
    duration = 2.4927338 / 24.0
    cadence = 120.0 / 86400.0
    sigma = 300e-6
    primary_depth = 7809.0577339e-6
    secondary_duration = duration if secondary_duration is None else secondary_duration
    count = int(6 * period / cadence)
    time = epoch + (np.arange(count) - count // 2) * cadence
    phase = (time - epoch + period / 2) % period - period / 2
    flux = 1.0 - (np.abs(phase) <= duration / 2) * primary_depth
    if secondary_phase is not None and secondary_depth:
        centre = (secondary_phase * period + period / 2) % period - period / 2
        distance = (phase - centre + period / 2) % period - period / 2
        flux -= (np.abs(distance) <= secondary_duration / 2) * secondary_depth
    rng = np.random.default_rng(20261114)
    flux = flux + sigma * rng.normal(size=time.size)
    curves = []
    for sector in sectors:
        keep = np.ones(time.size, dtype=bool)
        if sector == sectors[-1]:
            # A 0.4 d downlink gap away from both event windows.
            keep = np.abs(phase - gap_phase * period) > 0.2
        curves.append(lk.LightCurve(
            time=Time(time[keep], format="btjd", scale="tdb"),
            flux=flux[keep], flux_err=np.full(np.count_nonzero(keep), sigma),
            meta={"SECTOR": sector, "TIMEDEL": cadence},
        ))
    return curves, period, duration, sigma


def _run(monkeypatch, curves):
    cells = _cells()
    namespace = {"__name__": "trex_example_control"}
    exec(compile(cells[0], "<example-cell-0>", "exec"), namespace)
    namespace["SECTORS"] = sorted(curve.meta["SECTOR"] for curve in curves)
    monkeypatch.setattr(lk, "search_lightcurve", lambda *args, **kwargs: _SearchResult(curves))
    exec(compile(cells[1], "<example-cell-1>", "exec"), namespace)
    return namespace


def _depth(namespace, centre_days, half_width_days, exclude_days=0.0):
    time, flux = namespace["time"], namespace["flux"]
    baseline = np.abs(time - centre_days) > half_width_days + exclude_days
    # Unequal-width bins give the primary more samples, not more baseline.
    baseline &= np.abs(time) > DURATION_DAYS
    window = np.abs(time - centre_days) <= half_width_days
    assert np.any(window) and np.any(baseline)
    return float(np.median(flux[baseline]) - np.median(flux[window])) * 1e6


def test_depth_baseline_excludes_the_oversampled_primary():
    namespace = {
        "time": np.r_[np.linspace(-.04, .04, 11), -.5, -.3, .3, .5, 1.3],
        "flux": np.r_[np.full(11, .99), np.ones(4), .996],
    }
    assert _depth(namespace, 1.3, .1) == pytest.approx(4000., rel=0., abs=1e-6)


@pytest.mark.parametrize("secondary_phase", [0.3, 0.5])
def test_example_retains_the_full_orbit_and_transfers_event_depths(monkeypatch, secondary_phase):
    curves, period, duration, *_ = _curves(
        secondary_phase=secondary_phase, secondary_depth=4000e-6,
        secondary_duration=2 * DURATION_DAYS,
    )
    namespace = _run(monkeypatch, curves)
    time, flux, sigma = namespace["time"], namespace["flux"], namespace["sigma"]
    widths = np.asarray(namespace["bin_integration_days"])

    assert time.min() < -0.45 * period and time.max() > 0.45 * period
    assert widths.shape == time.shape and np.all(widths > 0.0)
    assert widths.max() <= duration + np.finfo(float).eps
    assert np.median(sigma) * 1e6 > 0.0

    # Primary depth transfer through the phase bins.
    primary = _depth(namespace, 0.0, 0.5 * duration, exclude_days=0.5 * duration)
    assert primary == pytest.approx(7809.0577339, rel=0.10)
    # The eccentric or half-phase secondary is retained at its own phase.
    centre = secondary_phase * period if secondary_phase < 0.5 else 0.5 * period
    if secondary_phase < 0.5:
        secondary = _depth(namespace, centre, 0.5 * duration)
        assert 0.35 * 4000.0 < secondary < 1.3 * 4000.0
    else:
        # The wrapped +-P/2 edge bins share a half-phase event.
        edge = np.array([0, -1])
        weight = widths[edge]
        edge_flux = float(np.sum(flux[edge] * weight) / np.sum(weight))
        baseline = np.abs(np.abs(time) - 0.5 * period) > 1.5 * duration
        baseline &= np.abs(time) > duration
        assert (float(np.median(flux[baseline])) - edge_flux) * 1e6 == pytest.approx(4000.0, rel=0.40)

    record = namespace["noise"]
    assert record["status"] == "passed"
    assert int(record["sectors"]["2"]["sample_count"]) > 0.5 * sum(
        len(curve) for curve in curves if curve.meta["SECTOR"] == 2
    )
    assert record["secondary_mask"] == "circular-half-phase-window"


def test_half_phase_secondary_does_not_change_the_native_noise_factor(monkeypatch):
    plain = _curves()
    eclipsing = _curves(secondary_phase=0.5, secondary_depth=4000e-6)
    plain_record = _run(monkeypatch, plain[0])["noise"]
    eclipsing_record = _run(monkeypatch, eclipsing[0])["noise"]
    assert eclipsing_record["event_windows"] == plain_record["event_windows"]
    assert eclipsing_record["native_uncertainty_factor"] == pytest.approx(
        plain_record["native_uncertainty_factor"], rel=1e-10
    )
    for sector, record in plain_record["sectors"].items():
        assert eclipsing_record["sectors"][sector]["sample_count"] == record["sample_count"]


def test_missing_phase_bins_are_absent_not_filled(monkeypatch):
    curves, period, duration, *_ = _curves(sectors=(2,), gap_phase=0.5)
    namespace = _run(monkeypatch, curves)
    time = namespace["time"]
    # The 0.4 d downlink gap is defined around +P/2 only: data stop 0.2 d
    # before +P/2, so a populated bin centre cannot lie beyond that boundary.
    # The -P/2 edge is still observed, so it must remain in the fold; empty
    # bins are omitted instead of fabricated as zero flux.
    cutoff = 0.5 * period - 0.2
    assert np.all(time < cutoff)
    assert np.any(time < -cutoff)
    assert namespace["noise"]["status"] == "passed"


def test_eccentric_secondary_visibility_and_primary_only_control(monkeypatch):
    plain = _run(monkeypatch, _curves()[0])["noise"]
    eccentric = _run(monkeypatch, _curves(secondary_phase=0.3, secondary_depth=4000e-6)[0])["noise"]
    assert plain["secondary_mask"] == "circular-half-phase-window"
    # The circular mask cannot exclude phase 0.3, so the event enters the noise
    # sample; the record states that assumption rather than claiming support.
    assert eccentric["native_uncertainty_factor"] > plain["native_uncertainty_factor"]
    assert eccentric["event_phase_support"].startswith("circular assumption")


@pytest.mark.parametrize("acquisition_fails", [False, True])
def test_example_handoff_passes_time_flux_sigma_and_bin_widths(monkeypatch, tmp_path, acquisition_fails):
    from trex.target import target

    curves, period, duration, _sigma = _curves()
    namespace = _run(monkeypatch, curves)
    namespace["work"] = tmp_path / "trex-toi-181"
    star = SimpleNamespace(
        stars="scene", trex_result=SimpleNamespace(status="ok"), FPP=0.1, NFPP=0.01, probs=[0.0],
        calls={},
    )

    def calc_probs(*args, **kwargs):
        star.calls["args"], star.calls["kwargs"] = args, kwargs

    def write_outputs(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text("{}")

    star.calc_probs, star.write_outputs = calc_probs, write_outputs
    seen = {}

    def acquire(cls, *args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        if acquisition_fails:
            raise RuntimeError("synthetic archive failure")
        return star

    monkeypatch.setattr(target, "from_tic", classmethod(acquire))
    if acquisition_fails:
        with pytest.raises(RuntimeError, match="synthetic archive failure"):
            exec(compile(_cells()[2], "<example-cell-2>", "exec"), namespace)
        assert json.loads((namespace["work"] / "noise.json").read_text()) == namespace["noise"]
        assert not (namespace["work"] / "result").exists()
        return
    exec(compile(_cells()[2], "<example-cell-2>", "exec"), namespace)
    assert seen["args"][0] == 76923707
    assert np.asarray(seen["kwargs"]["bin_integration_days"]).shape == namespace["time"].shape
    assert seen["kwargs"]["exptime_days"] == pytest.approx(curves[0].meta["TIMEDEL"])
    assert seen["kwargs"]["sector_weights"].keys() == {2, 3}

    exec(compile(_cells()[3], "<example-cell-3>", "exec"), namespace)
    assert np.asarray(star.calls["args"][0]).shape == namespace["time"].shape
    assert np.asarray(star.calls["args"][1]).shape == namespace["flux"].shape
    assert np.asarray(star.calls["args"][2]).shape == namespace["sigma"].shape

    exec(compile(_cells()[4], "<example-cell-4>", "exec"), namespace)
    assert (tmp_path / "trex-toi-181" / "result" / "summary.json").is_file()
    assert json.loads((tmp_path / "trex-toi-181" / "noise.json").read_text()) == namespace["noise"]


@pytest.mark.parametrize("draws", [4096, 1_000_000])
@pytest.mark.parametrize(("diagnostics", "category"), [
    ({"effective_sample_sizes": {"TP": 2.0}}, "convergence"),
    ({"input_requirements": [{"source_id": "100002"}]}, "missing-evidence"),
    ({"numerical_failures": {"TP": 1}}, "numerical-failure"),
])
def test_example_distinguishes_recorded_failures_from_deferrals(
    tmp_path, monkeypatch, capsys, draws, diagnostics, category,
):
    from importlib import import_module
    from trex.app.example import synthetic_inputs
    from trex.app.target import target
    from trex.foundation.errors import TrexEvidenceUnavailableError

    scene, binding, time, flux, sigma, period = synthetic_inputs(tmp_path / "inputs")
    star = target.from_scene(scene, **binding)
    original = TrexEvidenceUnavailableError("synthetic evidence stop", diagnostics)

    def stop(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(import_module("trex.app.engine"), "run_trex_vetting", stop)
    namespace = dict(star=star, time=time, flux=flux, sigma=sigma, PERIOD_DAYS=period,
                     exptime_days=binding["exptime_days"], N_DRAWS=draws, work=tmp_path)
    cell = compile(_cells()[3], "<example-evidence-cell>", "exec")
    if category == "numerical-failure":
        with pytest.raises(TrexEvidenceUnavailableError) as caught:
            exec(cell, namespace)
        assert caught.value is original
        assert "TREX deferred" not in capsys.readouterr().out
    else:
        exec(cell, namespace)
        text = capsys.readouterr().out
        assert f"{draws:,} draws per scenario ({category})" in text
        assert "FPP =" not in text and "production-budget run is pending" not in text
        exec(compile(_cells()[4], "<example-output-cell>", "exec"), namespace)
    record = json.loads((tmp_path / "result/failure.json").read_text())
    assert record["operational_diagnosis"]["category"] == category
    assert record["operational_diagnosis"]["status"] == (
        "failed" if category == "numerical-failure" else "deferred")
    assert record["n_draws"] == draws and record["error_type"] == type(original).__name__
    assert record["diagnostics"] == diagnostics
    assert not (tmp_path / "result/summary.json").exists()
