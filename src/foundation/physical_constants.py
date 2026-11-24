"""Canonical portable standards and unit conversions used by TREX.

Dimensional names carry their unit as a suffix (``_CGS``, ``_SI``, ``_CM``,
``_G``, ``_M_K``, ``_G_CM3``); ratios and representation factors are
dimensionless. Astropy supplies the IAU 2015 B3 nominal values (Prša et al.
2016, AJ 152, 41, DOI 10.3847/0004-6256/152/2/41) and CODATA 2018 constants
(Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010, DOI
10.1103/RevModPhys.93.025010). The AU follows IAU 2012 B2. These are
standards, not measurements or calibrated priors.

IAU 2015 B3 defines a nominal solar mass parameter (GM), not a nominal mass:
``NOMINAL_SOLAR_MASS_PARAMETER_CGS`` is that GM [cm^3 s^-2], while
``NOMINAL_SOLAR_MASS_G`` is the derived mass GM / G [g] with CODATA 2018 G,
the same derivation as ``astropy.constants.M_sun``. Use GM where only the
product enters (Kepler's third law) to avoid G's measurement uncertainty.

Importing this module loads Astropy; ``trex.foundation._numerics`` does not.
"""

import math
from fractions import Fraction

from astropy import constants as _const
from astropy import units as _u
from astropy.constants import codata2018 as _codata
from astropy.constants import iau2015 as _iau


# Unit conversions.
SECONDS_PER_DAY: float = float(_u.day.to(_u.s))
MILLIARCSECONDS_PER_ARCSECOND: float = float(_u.arcsec.to(_u.mas))
SQUARE_ARCSECONDS_PER_SQUARE_DEGREE: float = float((_u.deg**2).to(_u.arcsec**2))
FULL_TURN_DEGREES: float = math.degrees(math.tau)

# CODATA 2018 constants.
GRAVITATIONAL_CONSTANT_SI: float = float(_codata.G.value)
GRAVITATIONAL_CONSTANT_CGS: float = GRAVITATIONAL_CONSTANT_SI * float(
    (1.0 * _u.m**3 / (_u.kg * _u.s**2)).to(_u.cm**3 / (_u.g * _u.s**2)).value
)
SPEED_OF_LIGHT_CGS: float = float(_const.c.cgs.value)
SECOND_RADIATION_CONSTANT_M_K: float = float(
    (_codata.h * _const.c / _codata.k_B).to(_u.m * _u.K).value
)

# IAU 2015 B3 nominal values (and the IAU 2012 AU) with derived quantities.
NOMINAL_SOLAR_RADIUS_M: float = float(_iau.R_sun.value)
NOMINAL_SOLAR_RADIUS_CM: float = float(_iau.R_sun.cgs.value)
NOMINAL_SOLAR_MASS_PARAMETER_CGS: float = float(_iau.GM_sun.cgs.value)
NOMINAL_SOLAR_MASS_G: float = float((_iau.GM_sun / _codata.G).cgs.value)
NOMINAL_EARTH_EQUATORIAL_RADIUS_M: float = float(_iau.R_earth.value)
NOMINAL_EARTH_EQUATORIAL_RADIUS_CM: float = float(_iau.R_earth.cgs.value)
ASTRONOMICAL_UNIT_CM: float = float(_const.au.cgs.value)
# Mean density of the nominal Sun, M / (4/3 pi R^3); see the Fraction note below.
SOLAR_MEAN_DENSITY_G_CM3: float = float(
    (
        (_iau.GM_sun / _codata.G)
        / (float(Fraction(4, 3)) * math.pi * _iau.R_sun**3)
    ).to(_u.g / _u.cm**3).value
)
EARTH_TO_SOLAR_RADIUS_RATIO: float = (
    NOMINAL_EARTH_EQUATORIAL_RADIUS_M / NOMINAL_SOLAR_RADIUS_M
)

# Representation factors, not empirical coefficients. A Fraction converts
# to the correctly rounded float (identical to 4 / 3 or -0.4); it records the
# exact ratio and adds no precision.
PARTS_PER_MILLION: float = 1_000_000.0
FRACTION_PER_PART_PER_MILLION: float = 1.0 / PARTS_PER_MILLION
POGSON_FLUX_EXPONENT: float = -float(Fraction(2, 5))

__all__ = [
    "SECONDS_PER_DAY",
    "MILLIARCSECONDS_PER_ARCSECOND",
    "SQUARE_ARCSECONDS_PER_SQUARE_DEGREE",
    "FULL_TURN_DEGREES",
    "GRAVITATIONAL_CONSTANT_SI",
    "GRAVITATIONAL_CONSTANT_CGS",
    "SPEED_OF_LIGHT_CGS",
    "SECOND_RADIATION_CONSTANT_M_K",
    "NOMINAL_SOLAR_RADIUS_M",
    "NOMINAL_SOLAR_RADIUS_CM",
    "NOMINAL_SOLAR_MASS_PARAMETER_CGS",
    "NOMINAL_SOLAR_MASS_G",
    "NOMINAL_EARTH_EQUATORIAL_RADIUS_M",
    "NOMINAL_EARTH_EQUATORIAL_RADIUS_CM",
    "ASTRONOMICAL_UNIT_CM",
    "SOLAR_MEAN_DENSITY_G_CM3",
    "EARTH_TO_SOLAR_RADIUS_RATIO",
    "PARTS_PER_MILLION",
    "FRACTION_PER_PART_PER_MILLION",
    "POGSON_FLUX_EXPONENT",
]
