"""Short CGS aliases for TREX physics code, plus one Gaia resolution assumption.

``physical_constants`` is the canonical owner of the standards: IAU 2015
Resolution B3 nominal values and CODATA 2018 constants, taken through
``astropy.constants`` / ``astropy.units`` as reference float64 quantities, never
hand-written or truncated literals. This module re-exports the standards TREX
physics uses and gives the CGS ones the short names of the astrophysical
literature (Giacalone et al. 2021). Importing it loads Astropy; code that only
needs ``_numerics`` does not.

Two values here are not Astropy standards: ``pi`` (``math.pi``, dimensionless)
and ``GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC``, an empirical assumption in
arcseconds documented at its definition.

References
----------
* Prša et al. (2016), AJ 152, 41, DOI
  ``10.3847/0004-6256/152/2/41``: IAU 2015 B3 nominal solar and planetary
  conversion constants.
* Tiesinga et al. (2021), Rev. Mod. Phys. 93, 025010, DOI
  ``10.1103/RevModPhys.93.025010``: CODATA 2018 fundamental constants.
* IAU 2012 Resolution B2: astronomical unit (149 597 870 700 m, exact).

Exported suffixes or inline comments specify CGS/SI/nominal-solar units.
"""

from __future__ import annotations

import math as _math

from trex.foundation.physical_constants import (
    ASTRONOMICAL_UNIT_CM,
    EARTH_TO_SOLAR_RADIUS_RATIO,
    GRAVITATIONAL_CONSTANT_CGS,
    NOMINAL_EARTH_EQUATORIAL_RADIUS_CM,
    NOMINAL_SOLAR_MASS_G,
    NOMINAL_SOLAR_MASS_PARAMETER_CGS,
    NOMINAL_SOLAR_RADIUS_CM,
    SECONDS_PER_DAY,
    SOLAR_MEAN_DENSITY_G_CM3,
    SPEED_OF_LIGHT_CGS,
)

Msun: float = NOMINAL_SOLAR_MASS_G
Rsun: float = NOMINAL_SOLAR_RADIUS_CM
Rearth: float = NOMINAL_EARTH_EQUATORIAL_RADIUS_CM
G: float = GRAVITATIONAL_CONSTANT_CGS
au: float = ASTRONOMICAL_UNIT_CM
pi: float = _math.pi

# Giacalone et al. (2021), AJ 161, 24, Sect. 1, DOI 10.3847/1538-3881/abc6af:
# Gaia DR2 consistently resolves sources separated by more than 2.2 arcsec
# (Brown et al. 2018, A&A 616, A1, DOI 10.1051/0004-6361/201833051).
# This is the unresolved bound when a contrast curve is available but no
# separate resolution limit is supplied; the resulting FPP is model-dependent.
GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC: float = 2.2


# Re-exported standards are public API even where TREX itself reads them
# through ``physical_constants`` or a short alias.
__all__ = [
    # Short CGS aliases and pi.
    "Msun",
    "Rsun",
    "Rearth",
    "G",
    "au",
    "pi",
    # Re-exported standards (owned by physical_constants).
    "ASTRONOMICAL_UNIT_CM",
    "EARTH_TO_SOLAR_RADIUS_RATIO",
    "GRAVITATIONAL_CONSTANT_CGS",
    "NOMINAL_EARTH_EQUATORIAL_RADIUS_CM",
    "NOMINAL_SOLAR_MASS_G",
    "NOMINAL_SOLAR_MASS_PARAMETER_CGS",
    "NOMINAL_SOLAR_RADIUS_CM",
    "SECONDS_PER_DAY",
    "SOLAR_MEAN_DENSITY_G_CM3",
    "SPEED_OF_LIGHT_CGS",
    # Empirical assumption.
    "GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC",
]
