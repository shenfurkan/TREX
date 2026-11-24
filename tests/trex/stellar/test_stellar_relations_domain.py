"""Target-neutral unit tests for stellar_relations() mass domain enforcement.

Covers:
  - Torres et al. (2010), A&ARv 18, 67, DOI 10.1007/s00159-009-0027-7 — M > 0.63 Msun spline
  - Chabrier CDWRF grid — M <= 0.63 Msun spline
  - Domain boundaries: [0.1, 40.0] Msun total coverage
  - CDWRF minimum: 0.1 Msun; Torres maximum: 40.0 Msun
  - Named domain boundary constants exported from funcs module

No candidate-specific TIC, sector, or ephemeris data appear here.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from trex.stellar.stellar_relations import stellar_relations
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MIN_SOLAR
from trex.stellar.stellar_relations import STELLAR_RELATIONS_MASS_MAX_SOLAR
from trex.stellar.stellar_relations import STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR


class TestStellarRelationsDomainEnforcement:
    """stellar_relations() must raise ValueError for out-of-domain masses."""

    def test_named_mass_domain_constants_exported(self):
        """Named mass domain constants must be exported."""
        assert math.isfinite(STELLAR_RELATIONS_MASS_MIN_SOLAR)
        assert math.isfinite(STELLAR_RELATIONS_MASS_MAX_SOLAR)
        assert math.isfinite(STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR)
        # Minimum is CDWRF grid lower node
        assert STELLAR_RELATIONS_MASS_MIN_SOLAR == pytest.approx(0.1)
        # Maximum is Torres grid upper node
        assert STELLAR_RELATIONS_MASS_MAX_SOLAR == pytest.approx(40.0)
        # Crossover between CDWRF and Torres
        assert STELLAR_RELATIONS_CDWRF_MASS_BOUNDARY_SOLAR == pytest.approx(0.63)

    def test_solar_mass_in_domain(self):
        """1.0 Msun (Torres regime) must produce physically sensible results."""
        r, t = stellar_relations(np.array([1.0]))
        assert 1.0 < r[0] < 1.2
        assert 5500 < t[0] < 6000

    def test_low_mass_m_dwarf_in_domain(self):
        """0.3 Msun (CDWRF regime) must produce a finite, positive result."""
        r, t = stellar_relations(np.array([0.3]))
        assert np.all(np.isfinite(r)) and np.all(r > 0.0)
        assert np.all(np.isfinite(t)) and np.all(t > 0.0)

    def test_lower_domain_boundary_accepted(self):
        """Exactly 0.1 Msun (CDWRF lower node) must be accepted."""
        r, t = stellar_relations(np.array([STELLAR_RELATIONS_MASS_MIN_SOLAR]))
        assert np.all(np.isfinite(r)) and np.all(r > 0.0)
        assert np.all(np.isfinite(t)) and np.all(t > 0.0)

    def test_upper_domain_boundary_accepted(self):
        """Exactly 40.0 Msun (Torres upper node) must be accepted."""
        r, t = stellar_relations(np.array([STELLAR_RELATIONS_MASS_MAX_SOLAR]))
        assert np.all(np.isfinite(r)) and np.all(r > 0.0)
        assert np.all(np.isfinite(t)) and np.all(t > 0.0)

    def test_below_domain_raises_value_error(self):
        """Mass < 0.1 Msun must raise ValueError (no silent extrapolation)."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([0.05]))

    def test_well_below_domain_raises_value_error(self):
        """Mass << 0.1 Msun must raise ValueError."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([0.001]))

    def test_above_domain_raises_value_error(self):
        """Mass > 40.0 Msun must raise ValueError (no silent extrapolation)."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([41.0]))

    def test_well_above_domain_raises_value_error(self):
        """Mass >> 40.0 Msun must raise ValueError."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([100.0]))

    def test_mixed_array_raises_if_any_out_of_domain(self):
        """Any out-of-domain mass in an array must raise ValueError."""
        # [0.5 (in), 0.05 (out)]
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([0.5, 0.05]))

    def test_negative_mass_raises_value_error(self):
        """Negative mass must raise ValueError."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([-1.0]))

    def test_zero_mass_raises_value_error(self):
        """Zero mass must raise ValueError."""
        with pytest.raises(ValueError, match="domain"):
            stellar_relations(np.array([0.0]))

    def test_nan_mass_raises_value_error(self):
        """NaN mass must raise ValueError."""
        with pytest.raises(ValueError):
            stellar_relations(np.array([float("nan")]))

    def test_finite_output_across_domain(self):
        """All in-domain masses produce finite positive radii and Teff."""
        masses = np.linspace(
            STELLAR_RELATIONS_MASS_MIN_SOLAR,
            STELLAR_RELATIONS_MASS_MAX_SOLAR,
            40,
        )
        r, t = stellar_relations(masses)
        assert np.all(np.isfinite(r)) and np.all(r > 0.0)
        assert np.all(np.isfinite(t)) and np.all(t > 0.0)

    def test_monotonic_radius_across_mass_range(self):
        """Radii should be broadly monotonically increasing with mass."""
        masses = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        r, _ = stellar_relations(masses)
        # Check that radii are broadly increasing (may not be perfectly monotonic
        # at the CDWRF/Torres crossover, but the trend should hold)
        assert r[0] < r[-1]

    def test_max_radii_clamp_still_works_within_domain(self):
        """max_radii parameter must still clip within-domain results."""
        r, _ = stellar_relations(np.array([1.0]), max_radii=np.array([0.5]))
        assert r[0] == pytest.approx(0.5)

    def test_vectorized_result_shape(self):
        """Output shapes must match input shape."""
        masses = np.array([0.3, 1.0, 2.0])
        r, t = stellar_relations(masses)
        assert r.shape == masses.shape
        assert t.shape == masses.shape
