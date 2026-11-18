"""Validated caller-supplied interstellar attenuation on the A_V scale.

An attenuation curve is a measured or published table of
``kappa(lambda) = A_lambda/A_V``, explicitly normalized so ``kappa(V) = 1``.
No analytic law, polynomial coefficient or solar-neighborhood substitution is
embedded here: a sightline-dependent, distance-dependent or NIR/Bulge-specific
result must arrive as the caller's own table with its source and applicability.

The tabulated degree of freedom is ``A_V``. ``R_V = A_V/E(B-V)`` is recorded as
metadata only; a variable ``R_V`` alone is not a universal NIR or Bulge law, and
this module never rescales a curve for it. Attenuation is applied once: using
this curve and then another band extinction on the same source double-counts
the dust column.
"""

from dataclasses import dataclass

import numpy as np


class ExtinctionUnsupportedError(ValueError):
    """A passband grid extends outside the tabulated attenuation law."""


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExtinctionCurve:
    """Tabulated A_lambda/A_V with its declared source and validity domain.

    ``wavelength_cm`` is vacuum wavelength [cm] on a strictly increasing grid.
    ``a_lambda_over_av`` is ``kappa = A_lambda/A_V`` [dimensionless], normalized
    so the V-band value is exactly 1 (the explicit all-zero model is the sole
    exception); the supplied table must already implement that normalization,
    because this class does not define a V band, rescale the curve or
    synthesize a law. ``r_v = A_V/E(B-V)`` is the dimensionless
    total-to-selective ratio recorded as provenance, not a reshaping parameter;
    a sightline- or distance-dependent law still requires the caller's table.
    ``r_v=None`` is the explicit zero-extinction model and is accepted only when
    every ``kappa`` is exactly 0. ``source`` names the primary reference/URL and
    retrieval date; ``applicability`` names the sightline, distance range and
    domain where the table is valid.

    Outside the tabulated support :meth:`at` returns NaN and :meth:`on_grid`
    refuses the grid. Nothing is extrapolated, clipped, smoothed or replaced by
    a solar-neighborhood curve. Applying this curve and then an additional band
    extinction to the same source double-counts the attenuation.
    """

    wavelength_cm: np.ndarray
    a_lambda_over_av: np.ndarray
    r_v: float | None
    source: str
    applicability: str
    normalization: str = "A_V"

    def __post_init__(self):
        wave, kappa = (np.asarray(v) for v in (self.wavelength_cm, self.a_lambda_over_av))
        if (wave.ndim != 1 or wave.size < 2
                or wave.dtype.kind not in "iuf" or kappa.dtype.kind not in "iuf"
                or kappa.shape != wave.shape
                or not np.isfinite(wave).all() or not np.isfinite(kappa).all()
                or np.any(wave <= 0) or np.any(wave[1:] <= wave[:-1])
                or np.any(kappa < 0)):
            raise ValueError(
                "attenuation curve requires a finite strictly increasing positive "
                "vacuum wavelength grid and a same-shape finite nonnegative A_lambda/A_V"
            )
        r_v = None
        if self.r_v is None:
            if np.any(kappa != 0):
                raise ValueError(
                    "a nonzero attenuation curve must declare a positive finite R_V "
                    "(A_V/E(B-V)); r_v=None is reserved for the explicit "
                    "zero-extinction model"
                )
        else:
            r_v = np.asarray(self.r_v)
            if (r_v.ndim != 0 or r_v.dtype.kind not in "iuf"
                    or not np.isfinite(r_v) or r_v <= 0):
                raise ValueError("R_V must be a positive finite scalar or None")
            r_v = float(r_v)
        normalization = _text(self.normalization, "extinction normalization")
        if normalization != "A_V":
            raise ValueError(
                f"unsupported extinction normalization {normalization!r}; "
                "the curve must be tabulated as A_lambda/A_V"
            )
        for name, value in (("wavelength_cm", wave), ("a_lambda_over_av", kappa)):
            value = value.astype(float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "r_v", r_v)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "source", _text(self.source, "extinction source"))
        object.__setattr__(
            self, "applicability", _text(self.applicability, "extinction applicability")
        )

    def at(self, wavelength_cm):
        """kappa on a scalar or 1-D query grid, NaN outside the support.

        Linear interpolation between the tabulated nodes only; the closed
        support ``[first, last]`` includes the nodes. Vacuum wavelengths [cm]
        must be finite and nonnegative (zero lies outside the positive support
        and returns NaN). No extrapolation or clipping is performed.
        """
        query = np.asarray(wavelength_cm)
        if (query.ndim > 1 or query.dtype.kind not in "iuf"
                or not np.isfinite(query).all() or np.any(query < 0)):
            raise ValueError(
                "query wavelengths must be finite nonnegative real values on a "
                "scalar or 1-D grid [cm]"
            )
        values = np.interp(
            query, self.wavelength_cm, self.a_lambda_over_av,
            left=np.nan, right=np.nan,
        )
        return float(values) if query.ndim == 0 else values

    def on_grid(self, wavelength_cm):
        """kappa for a passband grid, refusing incomplete spectral coverage.

        Same interpolation contract as :meth:`at`, but any query outside the
        tabulated support raises :class:`ExtinctionUnsupportedError` instead of
        silently leaving NaN or treating the band as unreddened.
        """
        values = np.asarray(self.at(wavelength_cm), dtype=float)
        if not np.isfinite(values).all():
            low, high = self.wavelength_cm[[0, -1]]
            raise ExtinctionUnsupportedError(
                f"passband wavelength grid is not covered by the tabulated "
                f"extinction support [{low:.6g}, {high:.6g}] cm; no extrapolation, "
                f"clipping or zero-extinction substitution (source: {self.source})"
            )
        return values

    def attenuation(self, av_mag, wavelength_cm=None):
        """``10**(-0.4 * A_V * kappa)`` [dimensionless] on a query or the table.

        ``A_V`` [mag] must be a finite nonnegative scalar; ``kappa`` is
        dimensionless and ``R_V`` is the dimensionless ratio A_V/E(B-V).
        ``A_V=0`` returns exactly 1.0, including outside the support. For
        ``A_V>0`` an out-of-support query returns NaN; use :meth:`on_grid` when
        an incomplete passband must be refused instead.
        """
        av = np.asarray(av_mag)
        if (av.ndim != 0 or av.dtype.kind not in "iuf"
                or not np.isfinite(av) or av < 0):
            raise ValueError("A_V must be a finite nonnegative scalar [mag]")
        if wavelength_cm is None:
            kappa = self.a_lambda_over_av
        else:
            kappa = self.at(wavelength_cm)
        if av == 0:
            values = np.asarray(kappa, dtype=float)
            return 1.0 if values.ndim == 0 else np.ones_like(values)
        return np.power(10.0, -0.4 * float(av) * kappa)

    def as_passband_extinction(self, response):
        """Return ``(kappa, provenance)`` on a ``SpectralResponse`` grid.

        This is the narrow boundary to
        :meth:`trex.stellar_light.response.SpectralResponse.to_photon_passband`:
        the response supplies the wavelength grid, kappa is evaluated with
        :meth:`on_grid` so an incomplete band is refused, and the provenance
        combines the table source, applicability, A_V normalization and R_V.
        The returned kappa is applied once by the passband owner; applying a
        further band extinction to the same source double-counts.
        """
        from trex.stellar_light.response import SpectralResponse

        if not isinstance(response, SpectralResponse):
            raise ValueError(
                "passband extinction requires a trex.stellar_light.response."
                "SpectralResponse grid"
            )
        kappa = self.on_grid(response.wavelength_cm)
        r_v = (
            "None (explicit zero-extinction model)"
            if self.r_v is None
            else f"{self.r_v:g}"
        )
        provenance = (
            f"A_lambda/A_V extinction table: {self.source}; "
            f"applicability: {self.applicability}; "
            f"normalization: {self.normalization}; R_V={r_v}"
        )
        return kappa, provenance
