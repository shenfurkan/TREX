"""Gaia DR3 to TESS magnitude relation (Stassun et al. 2019, TIC v8).

Stassun et al. (2019), AJ 158, 138, DOI 10.3847/1538-3881/ab3467, Sect. 2.3.1:
Eq. (1) T = G + c3 x^3 + c2 x^2 + c1 x + c0 with x = G_BP - G_RP is valid for
-0.2 < x < 3.5, and TICv8 extrapolates the same polynomial to -1.0 < x < 6.0
with 0.1 mag added to its error. Without a color in that domain, Eq. (2)
T = G - 0.430 (a solar-color star, x = 0.82) applies with a conservative
0.6 mag error, "valid for all but the reddest M dwarfs".

Provenance boundary: Eq. (1) was fitted to PHOENIX synthetic colors in the
Gaia DR2 passbands for unreddened stars. Applying it to Gaia DR3 photometry
assumes the DR2->DR3 passband change is negligible at the relation's ~0.1 mag
systematic level; no DR3-specific transfer is applied, and colors are used as
observed (not de-reddened).
"""

import math
from typing import NamedTuple


class _Stassun2019Coeffs(NamedTuple):
    c3: float
    c2: float
    c1: float
    c0: float


STASSUN_2019_G_TO_T_COEFFS = _Stassun2019Coeffs(c3=-0.00522555, c2=0.0891337, c1=-0.633923, c0=0.0324473)

# Open color intervals (mag): Eq. (1)'s validity and TICv8's extrapolation.
STASSUN_2019_EQ1_VALID_BP_RP = (-0.2, 3.5)
STASSUN_2019_EQ1_TIC_EXTRAPOLATED_BP_RP = (-1.0, 6.0)
STASSUN_BP_RP_MIN, STASSUN_BP_RP_MAX = STASSUN_2019_EQ1_TIC_EXTRAPOLATED_BP_RP
STASSUN_2019_EQ2_G_MINUS_T = 0.430

# Eq. (2)'s error. Consumers use it as the half-width of a magnitude scenario
# interval, not as a Gaussian sigma; it does not cover the reddest M dwarfs.
STASSUN_2019_EQ2_CONSERVATIVE_ERROR_MAG = 0.6
# Model and empirical G-V relations differ by at most ~0.1 mag (Sect. 2.3.1): the
# relation's accuracy for measured-G sources, not a bound on any one source.
STASSUN_2019_RELATION_ACCURACY_MAG = 0.1
STASSUN_2019_EQ2_APPLICABILITY = (
    "Stassun et al. (2019) Sect. 2.3.1 Eq. (2): solar-color offset with a 0.6 mag "
    "conservative error, stated valid for all but the reddest M dwarfs"
)


def tess_magnitude_from_gaia_photometry(g_mag: object, bp_mag: object, rp_mag: object) -> tuple[float | None, str | None]:
    """TICv8 TESS magnitude and the label of the relation used.

    Labels: ``stassun2019-eq1`` (color inside Eq. (1)'s validity interval),
    ``stassun2019-eq1-tic-extrapolated`` (inside TICv8's extrapolation only) and
    ``stassun2019-eq2-no-valid-color`` (no color in the extrapolation domain,
    whether BP/RP is missing or the color lies outside it, as in TICv8). Both
    interval bounds are exclusive. A missing G magnitude gives ``(None, None)``.
    Inputs are magnitudes as numbers or numeric strings; booleans and
    non-finite values count as missing.
    """
    magnitude = _finite_float(g_mag)
    if magnitude is None:
        return None, None
    bp = _finite_float(bp_mag)
    rp = _finite_float(rp_mag)
    if bp is not None and rp is not None:
        color = bp - rp
        if STASSUN_BP_RP_MIN < color < STASSUN_BP_RP_MAX:
            c = STASSUN_2019_G_TO_T_COEFFS
            tess_mag = magnitude + ((c.c3 * color + c.c2) * color + c.c1) * color + c.c0
            valid_lower, valid_upper = STASSUN_2019_EQ1_VALID_BP_RP
            relation = "stassun2019-eq1" if valid_lower < color < valid_upper else "stassun2019-eq1-tic-extrapolated"
            return tess_mag, relation
    return magnitude - STASSUN_2019_EQ2_G_MINUS_T, "stassun2019-eq2-no-valid-color"


def _finite_float(value: object) -> float | None:
    """A finite magnitude from a number or numeric string, or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
