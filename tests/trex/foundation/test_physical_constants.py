"""IAU/CODATA values and CGS consumer bindings, not frozen source expressions."""

import math
from fractions import Fraction

import pytest

from trex.foundation import constants, physical_constants as canonical

pytestmark = pytest.mark.slow


def test_values_match_astropy_standards_and_exact_representations():
    from astropy import units as u
    from astropy.constants import codata2018, iau2015

    exact = {
        "SECONDS_PER_DAY": u.day.to(u.s),
        "MILLIARCSECONDS_PER_ARCSECOND": u.arcsec.to(u.mas),
        "SQUARE_ARCSECONDS_PER_SQUARE_DEGREE": (u.deg**2).to(u.arcsec**2),
        "FULL_TURN_DEGREES": (math.tau * u.rad).to_value(u.deg),
        "GRAVITATIONAL_CONSTANT_SI": codata2018.G.to_value(u.m**3 / (u.kg * u.s**2)),
        "SPEED_OF_LIGHT_CGS": codata2018.c.to_value(u.cm / u.s),
        "NOMINAL_SOLAR_RADIUS_M": iau2015.R_sun.to_value(u.m),
        "NOMINAL_SOLAR_RADIUS_CM": iau2015.R_sun.to_value(u.cm),
        "NOMINAL_SOLAR_MASS_PARAMETER_CGS": iau2015.GM_sun.to_value(u.cm**3 / u.s**2),
        "NOMINAL_EARTH_EQUATORIAL_RADIUS_M": iau2015.R_earth.to_value(u.m),
        "NOMINAL_EARTH_EQUATORIAL_RADIUS_CM": iau2015.R_earth.to_value(u.cm),
        "ASTRONOMICAL_UNIT_CM": iau2015.au.to_value(u.cm),
        "PARTS_PER_MILLION": float(10**6),
        "FRACTION_PER_PART_PER_MILLION": float(Fraction(1, 10**6)),
        "POGSON_FLUX_EXPONENT": -float(Fraction(2, 5)),
    }
    for name, expected in exact.items():
        assert getattr(canonical, name) == expected, name

    derived = {
        "GRAVITATIONAL_CONSTANT_CGS": codata2018.G.to_value(u.cm**3 / (u.g * u.s**2)),
        "NOMINAL_SOLAR_MASS_G": (iau2015.GM_sun / codata2018.G).to_value(u.g),
        "SECOND_RADIATION_CONSTANT_M_K": (
            codata2018.h * codata2018.c / codata2018.k_B
        ).to_value(u.m * u.K),
        "EARTH_TO_SOLAR_RADIUS_RATIO": (
            iau2015.R_earth / iau2015.R_sun
        ).to_value(u.dimensionless_unscaled),
        "SOLAR_MEAN_DENSITY_G_CM3": (
            (iau2015.GM_sun / codata2018.G)
            / (float(Fraction(4, 3)) * math.pi * iau2015.R_sun**3)
        ).to_value(u.g / u.cm**3),
    }
    # Reordered unit conversions differ only at binary64 roundoff, not measurement uncertainty.
    for name, expected in derived.items():
        assert getattr(canonical, name) == pytest.approx(expected, rel=2e-15, abs=0.0), name


@pytest.mark.parametrize("alias,name", [
    ("Msun", "NOMINAL_SOLAR_MASS_G"),
    ("Rsun", "NOMINAL_SOLAR_RADIUS_CM"),
    ("Rearth", "NOMINAL_EARTH_EQUATORIAL_RADIUS_CM"),
    ("G", "GRAVITATIONAL_CONSTANT_CGS"),
    ("au", "ASTRONOMICAL_UNIT_CM"),
])
def test_short_cgs_aliases_use_the_canonical_standards(alias, name):
    assert getattr(constants, alias) is getattr(canonical, name)
