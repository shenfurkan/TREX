"""TRICERATOPS-packaged TESS host limb-darkening grid contract.

The reference TRICERATOPS 1.0.20 package selects host-specific quadratic
coefficients from its packaged ``ldc_tess.csv`` grid.  TREX must use that same
versioned grid rather than substitute target-star coefficients for another host.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_packaged_grid_selects_published_host_coefficients():
    """Exact published rows, not merely two different finite answers.

    TRICERATOPS 1.0.20 ``ldc_tess.csv`` (columns logg,Teff,Z,xi,aLSM,bLSM):
    4.5,5750,0.0 -> (0.3075, 0.2743) and 5.0,3500,0.0 -> (0.2594, 0.3847).
    """
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    assert grid.coefficients(5750.0, 4.5, 0.0) == pytest.approx((0.3075, 0.2743), abs=1e-12)
    assert grid.coefficients(3500.0, 5.0, 0.0) == pytest.approx((0.2594, 0.3847), abs=1e-12)
    assert grid.package_version == "1.0.20"


def test_packaged_grid_rejects_nonfinite_host_atmospheres():
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    with pytest.raises(ValueError, match="finite"):
        grid.coefficients(float("nan"), 4.5, 0.0)


def test_packaged_grid_surfaces_out_of_bounds_clamp_with_the_used_node():
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    # D-62: a 2949 K solar dwarf takes Claret (2017) PHOENIX-COND table 15, linear in
    # Teff between its published 2900/3000 K rows at log g 4.5 (aLSM 0.1863/0.1693,
    # bLSM 0.5198/0.5023), instead of the former clamp to the 3500 K ATLAS edge.
    cool = grid.select(2949.0, 4.5, 0.0)
    assert (cool.u1, cool.u2) == pytest.approx((0.1863 + 0.49 * (0.1693 - 0.1863),
                                                0.5198 + 0.49 * (0.5023 - 0.5198)), abs=1e-12)
    assert cool.used_teff_k == 2949.0 and cool.boundary_clamped is False
    # Below both tables (PHOENIX-COND starts at 2300 K) the reference clamp and its flag remain.
    clamped = grid.select(2000.0, 4.5, 0.0)
    assert clamped.used_teff_k == 3500.0
    assert clamped.boundary_clamped is True

    in_bounds = grid.select(5750.0, 4.5, 0.0)
    assert in_bounds.used_teff_k == 5750.0
    assert in_bounds.boundary_clamped is False


def test_population_lookup_vectorized_selection_matches_scalar_selection():
    """Consistency only: the vector path must reproduce the scalar lookup.

    Coefficient and clamp correctness is covered by the published-node and
    analytic controls above; this case pins the batched handoff at grid edges
    and half-step ties.
    """
    from fractions import Fraction

    from trex.stellar_light.host_limb_darkening import TESS_LOGG_GRID_MAX_DEX, TESS_LOGG_GRID_MIN_DEX, TESS_LOGG_GRID_STEP_DEX, TESS_TEFF_GRID_MAX_K, TESS_TEFF_GRID_MIN_K, TESS_TEFF_GRID_STEP_K, load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    # TEST_DESIGN: cover grid boundaries, half-step ties, and repeated hosts.
    half = float(Fraction(1, 2))
    teff = np.array([
        TESS_TEFF_GRID_MIN_K - TESS_TEFF_GRID_STEP_K,
        TESS_TEFF_GRID_MIN_K + half * TESS_TEFF_GRID_STEP_K,
        TESS_TEFF_GRID_MAX_K + TESS_TEFF_GRID_STEP_K,
    ])
    logg = np.array([
        TESS_LOGG_GRID_MIN_DEX - TESS_LOGG_GRID_STEP_DEX,
        TESS_LOGG_GRID_MIN_DEX + half * TESS_LOGG_GRID_STEP_DEX,
        TESS_LOGG_GRID_MAX_DEX + TESS_LOGG_GRID_STEP_DEX,
    ])
    teff, logg = np.tile(teff, teff.size), np.repeat(logg, logg.size)
    metallicity = float(Fraction(0, 1))
    expected = [grid.select(t, g, metallicity) for t, g in zip(teff, logg)]
    actual = grid.select_many(teff, logg, metallicity)
    for name in ("u1", "u2", "used_teff_k", "used_logg_cgs", "boundary_clamped"):
        np.testing.assert_array_equal(getattr(actual, name), [getattr(node, name) for node in expected])
    assert actual.used_metallicity_dex == expected[0].used_metallicity_dex


def test_population_lookup_rejects_invalid_shapes_and_missing_nodes():
    from dataclasses import replace

    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    node = grid.select(5750.0, 4.5, 0.0)
    for temperatures, gravities in (([np.nan], [node.used_logg_cgs]), ([node.used_teff_k], [])):
        with pytest.raises(ValueError):
            grid.select_many(temperatures, gravities, node.used_metallicity_dex)
    sparse = replace(grid, rows=tuple(row for row in grid.rows if row["Teff"] != node.used_teff_k))
    with pytest.raises(RuntimeError, match="exact host coefficient node"):
        sparse.select_many([node.used_teff_k], [node.used_logg_cgs], node.used_metallicity_dex)


def test_metal_poor_secondary_vector_matches_scalar_selection():
    """Consistency only: the demonstrated non-solar planes agree scalar vs vector.

    The private population table is deliberately not used as an oracle; it is
    the same implementation the vector path reads.
    """
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    teff, logg, mh = np.array([3500., 5750.]), np.array([5., 4.5]), np.array([-3., 0.])
    scalar = [grid.select(t, g, z) for t, g, z in zip(teff, logg, mh)]
    vector = grid.select_many(teff, logg, mh)
    for name in ("u1", "u2"):
        expected = np.array([getattr(node, name) for node in scalar])
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(getattr(vector, name), expected)
    np.testing.assert_array_equal(vector.used_metallicity_dex, mh)
    assert not vector.boundary_clamped.any()


def test_checkpoint_identity_binds_both_limb_darkening_tables(monkeypatch):
    from dataclasses import replace
    import trex.stellar_light.host_limb_darkening as tables
    from trex.evidence.nested_checkpoint import checkpoint_identity

    grid = tables.load_triceratops_tess_grid()
    before = checkpoint_identity(("inputs",))
    monkeypatch.setattr(tables, "load_triceratops_tess_grid", lambda: replace(grid, supplement_sha256="changed"))
    assert checkpoint_identity(("inputs",)) != before


@pytest.mark.parametrize("coordinates", [
    ([True], [4.5], 0.), ([5750.], [True], 0.), ([5750.], [4.5], True),
])
def test_population_lookup_refuses_boolean_physical_coordinates(coordinates):
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    with pytest.raises(ValueError, match="real numeric"):
        load_triceratops_tess_grid().select_many(*coordinates)


def test_valid_extension_does_not_require_an_unused_reference_node():
    from dataclasses import replace
    from trex.stellar_light.host_limb_darkening import load_triceratops_tess_grid

    grid = load_triceratops_tess_grid()
    expected = grid.select(11000., 4.5, 0.)
    assert expected.interpolated
    sparse = replace(grid, rows=tuple(row for row in grid.rows if row["Teff"] != 10000.))
    actual = sparse.select_many([11000.], [4.5], 0.)
    np.testing.assert_allclose([actual.u1[0], actual.u2[0]], [expected.u1, expected.u2], rtol=0, atol=0)
    assert actual.interpolated[0] and not actual.boundary_clamped[0]


def test_provenance_summary_separates_target_selection_from_unknown_mc():
    from trex.stellar_light.host_limb_darkening import TARGET_HOST_LDC_ROLE, summarize_ldc_grid_provenance

    empty = summarize_ldc_grid_provenance([])
    assert empty["clamped"] is None and empty["events"] is None
    assert empty["teff_k_lookup_bounds"] == [3500.0, 10000.0]
    assert empty["logg_cgs_lookup_bounds"] == [3.5, 5.0]

    target = {"role": TARGET_HOST_LDC_ROLE, "boundary_clamped": False}
    summary = summarize_ldc_grid_provenance([target])
    assert summary["clamped"] is None
    assert summary["events"] is None
    assert summary["monte_carlo_source_counts"]["total_draws"] is None
    assert summary["target_host"] == target
    clamped = summarize_ldc_grid_provenance([{**target, "boundary_clamped": True}])
    assert clamped["clamped"] is True


@pytest.mark.parametrize("missing", [
    "reference_draws", "atlas_draws", "phoenix_draws", "clamped_draws", "total_draws",
])
def test_incomplete_source_counts_stay_unknown_without_losing_known_events(missing):
    from trex.stellar_light.host_limb_darkening import MC_LDC_ROLE, summarize_ldc_grid_provenance

    marker = {"role": MC_LDC_ROLE, "status": "available", "source_counts_complete": True}
    event = {"stage": "monte-carlo", "reference_draws": 2, "atlas_draws": 3,
             "phoenix_draws": 4, "clamped_draws": 1, "total_draws": 10}
    del event[missing]
    summary = summarize_ldc_grid_provenance([marker, event])
    assert summary["monte_carlo_source_counts"]["total_draws"] is None
    assert summary["monte_carlo_source_counts"]["status"] == "unknown (legacy source counts incomplete)"
    if missing != "clamped_draws":
        assert summary["clamped"] is True


def test_host_grid_coefficients_count_every_lookup_by_its_source():
    """Producer-side accounting through the actual evidence handoff.

    D-62: every lookup input is counted once, including an in-bounds
    reference-only call. The 2949 K / 0.2 R_sun case is a published
    PHOENIX-COND hole and keeps the reference clamp.
    """
    from trex.evidence.marginal_likelihoods import _host_grid_coefficients

    def counts(event):
        return {key: event[key] for key in
                ("reference_draws", "atlas_draws", "phoenix_draws", "clamped_draws", "total_draws")}

    events = []
    _host_grid_coefficients(np.array([2949.0]), np.array([0.2]), np.array([0.2]), 0.0,
                            scenario="N26-EB", provenance=events)
    event, = events
    assert event["scenario"] == "N26-EB"
    assert counts(event) == dict(reference_draws=0, atlas_draws=0, phoenix_draws=0,
                                 clamped_draws=1, total_draws=1)
    assert event["used_nodes"] == [(3500.0, 5.0)]

    in_bounds = []
    _host_grid_coefficients(np.array([5750.0]), np.array([1.0]), np.array([1.0]), 0.0,
                            scenario="N26-EB", provenance=in_bounds)
    event, = in_bounds
    assert counts(event) == dict(reference_draws=1, atlas_draws=0, phoenix_draws=0,
                                 clamped_draws=0, total_draws=1)
    assert event["used_nodes"] == []


def test_complete_source_counts_are_partitioned_and_summed_per_event():
    """Summarizer contract for complete hand-written event records.

    Producer-side counting is covered by
    ``test_host_grid_coefficients_count_every_lookup_by_its_source`` above;
    this case pins only the report aggregation and completeness rule.
    """
    from trex.stellar_light.host_limb_darkening import MC_LDC_ROLE, summarize_ldc_grid_provenance

    marker = {"role": MC_LDC_ROLE, "status": "available", "source_counts_complete": True}
    event = {"stage": "monte-carlo", "reference_draws": 2, "atlas_draws": 3,
             "phoenix_draws": 4, "clamped_draws": 1, "total_draws": 10}
    summary = summarize_ldc_grid_provenance([marker, event, event])
    counts = summary["monte_carlo_source_counts"]
    assert counts == {"reference_draws": 4, "atlas_draws": 6, "phoenix_draws": 8,
                      "clamped_draws": 2, "total_draws": 20, "status": "available"}
    # A record that does not partition its own total is not complete.
    inconsistent = {**event, "total_draws": 11}
    assert summarize_ldc_grid_provenance([marker, inconsistent])[
        "monte_carlo_source_counts"]["total_draws"] is None


def test_eval_scenario_labels_clamp_provenance_with_the_scenario(monkeypatch):
    from trex.evidence import marginal_likelihoods

    n_draws = 4

    def background_draw(*_args, **_kwargs):
        _kwargs["metallicity_out"].append(np.full(n_draws, -0.2))
        return (
            np.full(n_draws, 0.1),
            np.zeros(n_draws),
            np.full(n_draws, 0.5),
            np.full(n_draws, 0.5),
            np.full(n_draws, 2000.0),
        )

    monkeypatch.setattr(marginal_likelihoods, "_draw_background_star", background_draw)
    monkeypatch.setattr(marginal_likelihoods, "sample_inc", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "sample_ecc", lambda draws, **_kwargs: np.zeros(draws.size))
    monkeypatch.setattr(marginal_likelihoods, "sample_w", lambda draws: np.full(draws.size, 90.0))
    monkeypatch.setattr(marginal_likelihoods, "lnL_TP", lambda *_args, **_kwargs: -1.0)

    events = []
    result = marginal_likelihoods._eval_scenario(
        time=np.array([0.0]), flux=np.array([1.0]), sigma=1e-4,
        period_days=3.0, M_s_Msun=1.0, R_s_Rsun=1.0, u1=0.3, u2=0.2,
        n_draws=n_draws, is_planet=True, is_EB=False, use_2xP=False,
        exptime_days=0.001, background_star="background",
        background_population={"unused": np.array([])}, background_area_sq_deg=1.0,
        target_tess_mag=10.0, contrast_separations=np.array([0.1, 1.0]),
        contrast_values=np.array([2.0, 5.0]), target_feh=-0.2,
        rng=np.random.default_rng(7), ldc_provenance=events, ldc_scenario="BEB",
    )

    assert np.isfinite(result)
    assert len(events) == 1
    assert events[0]["scenario"] == "BEB"
    assert events[0]["used_nodes"] == [(3500.0, 4.5)]
