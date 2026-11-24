"""Exact passband identity and detector-convention conversion to the photon kernel.

A filter profile is not a detector signal. Two physically different integrals are
used by published response tables:

* energy counter: ``S_E = integral F_lambda(lambda) R(lambda) dlambda``;
* photon counter: ``S_P = integral (lambda / h c) F_lambda(lambda) R(lambda) dlambda``.

``trex.stellar_light.passband.integrate_photon_intensity`` implements the photon
kernel ``integral (lambda / h c) I_lambda R_kernel dlambda`` (Chen et al. 2019,
A&A 632, A105, DOI 10.1051/0004-6361/201936612, eq_Mag photon weighting). Feeding
an energy-counter response straight into it computes a photon count, not the
energy signal the detector records. Algebra gives the photon-kernel input that
reproduces the energy signal, ``R_kernel = R_E h c / lambda`` up to a constant
normalization; adding another wavelength factor would double-weight photons.
The same rule keeps a complete energy-weighted response distinct from a
filter-only transmission. ``components`` records how much of the optical train
the curve actually covers: a curve labelled ``Filter`` cannot certify an
end-to-end instrument calibration.

Wavelengths are stored in vacuum centimetres; ``wavelength_convention`` keeps
the source's own air/vacuum declaration rather than inventing a conversion.
No negative response is clipped: a table with out-of-band noise below zero must
be repaired by an explicit, logged caller decision. ``response_id`` is a
canonical, instrument-qualified identifier because a bare ``J`` or ``z`` names
different physical responses at different facilities.
"""

from dataclasses import dataclass
import hashlib
from typing import Iterable, Tuple

from astropy import constants as const
import numpy as np

RESPONSE_QUANTITIES = ("dimensionless_transmission", "effective_area_m2")
MAGNITUDE_SYSTEMS = ("AB", "Vega", "ST", "PLATO_P", "TESS_T")
_DETECTOR_LABELS = {0: "energy counter", 1: "photon counter"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{0} must be a non-empty string".format(label))
    return value.strip()


def is_qualified_response_id(value) -> bool:
    """True for ``facility-or-mission/instrument.filter`` style identities.

    A bare filter letter is not usable where an instrument response is required:
    WIRC J, 2MASS J and Roman F146 are different curves with different detector
    conventions and wavelength coverage.
    """
    if not isinstance(value, str) or "/" not in value:
        return False
    head, tail = value.split("/", 1)
    return bool(head.strip()) and bool(tail.strip()) and not any(ch.isspace() for ch in value)


@dataclass(frozen=True)
class ZeroPoint:
    """One declared photometric zero point with its own magnitude system.

    ``unit`` is ``"mag"`` for a zero-point magnitude or ``"Jy"`` for the flux
    density of a zero-magnitude source. Values from different systems are never
    interchangeable without the source's own conversion.
    """

    system: str
    value: float
    unit: str
    source: str
    release: str = ""

    def __post_init__(self):
        system = _text(self.system, "zero-point system")
        if system not in MAGNITUDE_SYSTEMS:
            raise ValueError("unsupported zero-point magnitude system {0!r}".format(system))
        unit = _text(self.unit, "zero-point unit")
        if unit not in ("mag", "Jy"):
            raise ValueError("zero-point unit must be 'mag' or 'Jy'")
        value = np.asarray(self.value)
        if value.ndim != 0 or value.dtype.kind not in "iuf" or not np.isfinite(value):
            raise ValueError("zero-point value must be finite")
        if unit == "Jy" and value <= 0:
            raise ValueError("a zero-magnitude flux density must be positive")
        if not isinstance(self.release, str):
            raise ValueError("zero-point release must be a string")
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "value", float(value))
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "source", _text(self.source, "zero-point source"))
        object.__setattr__(self, "release", self.release.strip())


@dataclass(frozen=True)
class SpectralResponse:
    """Validated tabulated response with its exact physical identity.

    ``wavelength_cm`` is vacuum wavelength in cm and strictly increasing.
    ``response`` is either a dimensionless transmission or a physical effective
    area in m^2, as declared by ``response_quantity``. ``detector_type`` is the
    source table's convention (0 energy counter, 1 photon counter), not a claim
    that the curve covers the full optical train. ``components`` lists the
    declared contributors, e.g. ``("Filter",)`` or
    ``("Filter", "Instrument", "Atmosphere")``. ``calibration_context`` states
    what the edition is (as-designed, as-built, simulator, calibrated, ...).
    """

    response_id: str
    wavelength_cm: np.ndarray
    response: np.ndarray
    response_quantity: str
    detector_type: int
    components: Tuple[str, ...]
    facility: str
    instrument: str
    filter_name: str
    edition: str
    calibration_context: str
    wavelength_convention: str
    source: str
    zero_point: ZeroPoint | None = None

    def __post_init__(self):
        response_id = _text(self.response_id, "response id")
        if not is_qualified_response_id(response_id):
            raise ValueError(
                "response id must be instrument-qualified (facility/instrument.filter); "
                "a bare band name is ambiguous"
            )
        arrays = [np.asarray(v) for v in (self.wavelength_cm, self.response)]
        if (arrays[0].ndim != 1 or arrays[0].size < 2
                or any(a.dtype.kind not in "iuf" or a.shape != arrays[0].shape
                       or not np.isfinite(a).all() for a in arrays)
                or np.any(arrays[0] <= 0) or np.any(arrays[0][1:] <= arrays[0][:-1])
                or np.any(arrays[1] < 0) or not np.any(arrays[1] > 0)):
            raise ValueError(
                "invalid response table: finite ordered positive wavelengths and "
                "nonnegative non-identical response required (no clipping)"
            )
        quantity = _text(self.response_quantity, "response quantity")
        if quantity not in RESPONSE_QUANTITIES:
            raise ValueError("unsupported response quantity {0!r}".format(quantity))
        detector_type = self.detector_type
        if isinstance(detector_type, (bool, np.bool_)) or detector_type not in (0, 1):
            raise ValueError("detector type must be 0 (energy) or 1 (photon)")
        components = tuple(_text(part, "response component") for part in self.components)
        if not components:
            raise ValueError("a response must declare its included components")
        convention = _text(self.wavelength_convention, "wavelength convention")
        if convention not in ("vacuum", "air"):
            raise ValueError("wavelength convention must be 'vacuum' or 'air'")
        if self.zero_point is not None and not isinstance(self.zero_point, ZeroPoint):
            raise ValueError("zero point must be a ZeroPoint record")
        for name, value in (
            ("wavelength_cm", arrays[0]),
            ("response", arrays[1]),
        ):
            value = value.astype(float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "response_id", response_id)
        object.__setattr__(self, "response_quantity", quantity)
        object.__setattr__(self, "detector_type", int(detector_type))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "facility", _text(self.facility, "facility"))
        object.__setattr__(self, "instrument", _text(self.instrument, "instrument"))
        object.__setattr__(self, "filter_name", _text(self.filter_name, "filter name"))
        object.__setattr__(self, "edition", _text(self.edition, "response edition"))
        object.__setattr__(self, "calibration_context", _text(
            self.calibration_context, "calibration context"))
        object.__setattr__(self, "wavelength_convention", convention)
        object.__setattr__(self, "source", _text(self.source, "response source"))

    @property
    def detector_label(self) -> str:
        return _DETECTOR_LABELS[self.detector_type]

    @property
    def input_id(self) -> str:
        """Digest of the exact response bytes for cache/wrong-input binding only.

        This detects a substituted curve or corrupted table; it is not a physics
        or calibration check (a digest identifies bytes).
        """
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.wavelength_cm, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.response, dtype="<f8").tobytes())
        digest.update(self.response_id.encode("utf-8"))
        digest.update(self.response_quantity.encode("utf-8"))
        digest.update(str(self.detector_type).encode("ascii"))
        return digest.hexdigest()

    def provenance(self) -> str:
        """Human-readable identity written into prepared products."""
        return (
            "{0} [{1}; {2}; {3}; components={4}; {5}]; {6}".format(
                self.response_id, self.edition, self.detector_label,
                self.response_quantity, "+".join(self.components),
                self.calibration_context, self.source,
            )
        )

    def photon_kernel_response(self, *, normalized: bool = True) -> np.ndarray:
        """Response to multiply into ``integrate_photon_intensity``'s photon kernel.

        With ``normalized=False`` the returned array keeps physical scale:
        dimensionless transmission is per unit collecting area, while an
        effective area is converted from m^2 to cm^2. A photon-counter response is
        used as tabulated; an energy-counter response is multiplied by ``h c /
        lambda`` so the photon kernel reproduces its energy-weighted signal.
        With ``normalized=True`` the peak is divided out for colors/limb
        darkening, where only the spectral shape matters.
        """
        response = np.array(self.response, dtype=float, copy=True)
        if self.response_quantity == "effective_area_m2":
            response *= 1.0e4  # m^2 -> cm^2, matching the CGS intensity kernel.
        if self.detector_type == 0:
            # Energy signal / photon kernel: inverse wavelength (up to hc).
            response *= const.h.cgs.value * const.c.cgs.value / self.wavelength_cm
        if normalized:
            response /= np.max(response)
        return response

    def to_photon_passband(
        self,
        *,
        a_lambda_over_av=None,
        extinction_source: str,
        normalized: bool = True,
    ):
        """Build the existing photon kernel input without changing its algebra.

        ``a_lambda_over_av`` is a tabulated ``A_lambda/A_V`` curve on this
        response's wavelength grid; ``None`` means an explicitly declared zero
        extinction model and still requires ``extinction_source``. The returned
        :class:`~trex.stellar_light.passband.PhotonPassband` records this
        response's identity and exact input digest so a prepared grid cannot be
        reused across bands. ``normalized=True`` is for shape-only ratios; use
        ``False`` for an absolute signal (dimensionless transmission then means
        per cm^2 of collecting area, effective area means cm^2).
        """
        from .passband import PhotonPassband

        extinction = _extinction_curve(self.wavelength_cm, a_lambda_over_av)
        return PhotonPassband(
            self.wavelength_cm,
            self.photon_kernel_response(normalized=normalized),
            extinction,
            self.provenance(),
            _text(extinction_source, "extinction source"),
            self.response_id,
        )


def _extinction_curve(wavelength_cm, a_lambda_over_av):
    if a_lambda_over_av is None:
        return np.zeros_like(wavelength_cm)
    values = np.asarray(a_lambda_over_av)
    if (values.dtype.kind not in "iuf" or values.shape != np.shape(wavelength_cm)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("A_lambda/A_V must be finite nonnegative and match the response grid")
    return values.astype(float, copy=False)


def require_common_zero_point_system(responses: Iterable[SpectralResponse]) -> str:
    """Return the shared magnitude system or refuse a silent cross-system mix."""
    systems = set()
    for response in responses:
        if not isinstance(response, SpectralResponse):
            raise ValueError("zero-point compatibility requires SpectralResponse inputs")
        if response.zero_point is None:
            raise ValueError(
                "response {0} declares no zero point; it cannot be combined with a "
                "magnitude-system calculation".format(response.response_id)
            )
        systems.add(response.zero_point.system)
    if len(systems) != 1:
        raise ValueError(
            "responses declare incompatible zero-point systems {0}; convert explicitly".format(
                sorted(systems)
            )
        )
    return next(iter(systems))


def require_response_zero_point_edition(
    response: SpectralResponse, zero_point: ZeroPoint
) -> ZeroPoint:
    """Refuse a response and zero point taken from different release editions.

    A zero point is only meaningful for the effective-area/throughput table it was
    computed from. The release tag must match the response's edition, and a zero
    point already attached to the response may not be silently replaced.
    """
    if not isinstance(response, SpectralResponse) or not isinstance(zero_point, ZeroPoint):
        raise ValueError("edition compatibility requires a response and a zero point")
    if not zero_point.release:
        raise ValueError("zero point declares no release edition")
    if not response.edition.startswith(zero_point.release):
        raise ValueError(
            "response edition {0!r} does not match zero-point release {1!r}".format(
                response.edition, zero_point.release
            )
        )
    attached = response.zero_point
    if attached is not None and attached != zero_point:
        raise ValueError(
            "response carries a different zero point than the one requested; "
            "do not mix releases"
        )
    return zero_point
