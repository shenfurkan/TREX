"""TIC colour calibration, Pogson magnitude conversions and native-band ratios."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from trex.foundation.physical_constants import POGSON_FLUX_EXPONENT
from trex.stellar_light.response import is_qualified_response_id

# Stassun et al. (2018, AJ 156 102; DOI 10.3847/1538-3881/aad050)
# Section 2.2.1.1 piecewise cubic polynomial J - Ks -> delta_T (T - J).
# Coefficients listed descending: [c3, c2, c1, c0] so that
#   poly(x) = c3*x^3 + c2*x^2 + c1*x + c0.
# Valid domain: -0.1 <= J - Ks <= 1.0 (2MASS mag). Fail-closed outside domain.
STASSUN_2018_JK_DOMAIN_MIN: float = -0.1   # mag (J - Ks lower bound)
STASSUN_2018_JK_DOMAIN_MAX: float = 1.0    # mag (J - Ks upper bound)
STASSUN_2018_JK_BREAK: float = 0.70        # mag (piecewise break point)

# First-segment (−0.1 ≤ J−Ks ≤ 0.70) polynomial coefficients.
STASSUN_2018_FIRST_REL_COEFFS: tuple[float, float, float, float] = (1.22163, -1.74299, 1.89115, 0.0563)

# Second-segment (0.70 < J−Ks ≤ 1.0) polynomial coefficients.
STASSUN_2018_SECOND_REL_COEFFS: tuple[float, float, float, float] = (-269.372, 668.453, -545.64, 147.811)


def _real_values(values: ArrayLike) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not values.size:
        raise ValueError("photometric conversions require non-empty real inputs, not boolean or text")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("photometric conversion inputs must be finite")
    return values


def J_Ks_to_Tmag(J_mags: ArrayLike, Ks_mags: ArrayLike) -> np.ndarray:
    """Convert 2MASS J and Ks magnitudes to TESS T magnitudes.

    Applies the Stassun et al. (2018) piecewise cubic polynomial in the
    2MASS ``J - Ks`` color.  Raises ``ValueError`` for any input outside the
    published calibration domain ``[-0.1, 1.0]`` mag (fail-closed; no
    extrapolation or silent clamp).

    Args:
        J_mags: Finite real 2MASS J-band magnitudes [mag], scalar or array.
        Ks_mags: Finite real Ks magnitudes [mag], broadcastable with J.

    Returns:
        TESS T magnitudes [mag], with the broadcast shape (0-D for scalar inputs).

    Raises:
        ValueError: If any J - Ks color lies outside [-0.1, 1.0], or if any
            input is non-finite, boolean, text, empty or not broadcastable.

    The two published cubics have a discontinuous join at J-Ks=0.70, not a
    continuous colour law. Values just redward use the second cubic directly;
    no colour-to-spectral-type classification or artificial join offset is added.

    References:
        Stassun et al. (2018), AJ 156 102,
        DOI ``10.3847/1538-3881/aad050``.
        Erratum: Stassun et al. (2018), AJ 156 183,
        DOI ``10.3847/1538-3881/aade86``. It corrects only the
        Candidate Target List selection (Figure 5 and Section 3.1); the
        Section 2.2.1.1 J - Ks transform used here is unchanged.
    """
    J, Ks = np.broadcast_arrays(_real_values(J_mags), _real_values(Ks_mags))
    with np.errstate(over="ignore", invalid="ignore"):
        JK = J - Ks
    out_of_domain = (JK < STASSUN_2018_JK_DOMAIN_MIN) | (JK > STASSUN_2018_JK_DOMAIN_MAX)
    if np.any(out_of_domain):
        raise ValueError(
            "J_Ks_to_Tmag: J - Ks outside the Stassun et al. (2018) "
            "calibration domain [-0.1, 1.0] mag; no extrapolation is performed"
        )
    Tmags = np.empty_like(J)
    m1 = JK <= STASSUN_2018_JK_BREAK
    m2 = ~m1
    Tmags[m1] = J[m1] + np.polyval(STASSUN_2018_FIRST_REL_COEFFS, JK[m1])
    # The published TIC relation defines the second cubic directly.  It is
    # not algebraically continuity-corrected at J-Ks=0.70; adding an offset
    # would create a new, unreferenced empirical relation.
    Tmags[m2] = J[m2] + np.polyval(STASSUN_2018_SECOND_REL_COEFFS, JK[m2])
    return Tmags


def delta_mag_to_flux_ratio(delta_mag: ArrayLike) -> np.ndarray:
    """Return F_source/F_reference = 10**(-0.4 * delta_mag), dimensionless.

    delta_mag = m_source-m_reference [mag]; positive means fainter. Finite
    real scalar/array inputs preserve shape; boolean/text/empty inputs raise
    ValueError, as do differences with no finite positive float64 flux ratio.
    There is no artificial astrophysical contrast limit or zero/inf replacement.
    """
    values = _real_values(delta_mag)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.power(10.0, POGSON_FLUX_EXPONENT * values)
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("magnitude difference has no finite positive flux ratio in float64")
    return ratio


def flux_ratio_to_delta_mag(flux_ratio: ArrayLike) -> np.ndarray:
    """Return m_source-m_reference [mag] from positive F_source/F_reference.

    Pogson's exponent is -0.4; ratios below one give positive (fainter) contrasts.
    Shape is preserved for finite positive real scalar/array inputs. Nonfinite,
    nonpositive, boolean/text and empty inputs raise ValueError.
    """
    flux_ratio = _real_values(flux_ratio)
    if np.any(flux_ratio <= 0.0):
        raise ValueError("flux ratios must be finite and strictly positive")
    return np.log10(flux_ratio) / POGSON_FLUX_EXPONENT


def measured_same_band_flux_ratio(
    delta_mag: ArrayLike,
    *,
    band_id: str,
    reference_band_id: str,
) -> np.ndarray:
    """Pogson flux ratio from a *measured* magnitude difference in one native band.

    Both identities must be canonical instrument-qualified response ids and must
    match exactly. Two magnitudes measured through different filters are not a
    flux ratio: converting them would require the source SED or colour terms
    this function does not have. No SDSS/PS1/WIRC/2MASS transformation is
    applied, and no empirical cross-band offset is invented.
    """
    for value, label in ((band_id, "band id"), (reference_band_id, "reference band id")):
        if not is_qualified_response_id(value):
            raise ValueError(
                "{0} must be a canonical instrument-qualified response id; a bare "
                "filter letter is ambiguous".format(label)
            )
    if band_id != reference_band_id:
        raise ValueError(
            "a measured magnitude difference is a flux ratio only within one native "
            "band ({0!r} != {1!r})".format(band_id, reference_band_id)
        )
    return delta_mag_to_flux_ratio(delta_mag)
