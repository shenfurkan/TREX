"""Exact catalogue identifiers and their native epoch/frame context.

A catalog identity is three separate facts: which catalogue (namespace), the
catalogue's own identifier text, and the coordinate frame/epoch those
coordinates belong to. They are kept together because a bare number is not an
identity: Gaia ``source_id`` 1234, TIC 1234 and PIC 1234 are different
sources, and a Gaia position at J2016 is not a J2000 position.

Contract
--------
``exact_identifier``/``gaia_identifier`` remain the strict signed-64-bit
decimal path used by TIC, MAST, Gaia and neighbour catalogues. ``native_identifier``
is the general path: it preserves a catalogue's own identifier text exactly
(PIC names, 2MASS designations, VIRAC identifiers) and never forces it into
Gaia's signed-64-bit decimal convention. Missing, malformed and ambiguous
identities are distinct states; no helper here chooses a nearest match, invents
a Gaia/TIC cross-identifier or fills a missing epoch.

Units: ``epoch_jyear`` is a Julian year (e.g. 2016.0 for Gaia DR3 positions),
never a calendar date or a modified Julian date. Frames are labels, not
conversions; this module does not transform coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np

# 2**63 - 1 has 19 decimal digits; a longer significant string cannot fit.
_MAX_DIGITS = 19

# Frames whose coordinate convention can be stated without conversion here.
# A caller holding Galactic or FK5 coordinates preserves that fact instead of
# relabelling them ICRS; conversion belongs to the coordinate owner.
CATALOG_FRAMES = ("ICRS", "FK5", "GALACTIC")

# Association outcomes are deliberately closed. ``ambiguous`` keeps every
# candidate: a tie must be reviewed, never resolved by brightness or distance.
ASSOCIATION_STATUSES = ("empty", "unique", "ambiguous")


def exact_identifier(value, label: str = "catalogue identifier") -> str:
    """Canonical decimal text of a positive signed-64-bit identifier.

    Accepts Python or NumPy integers and ASCII decimal strings (leading zeros
    are dropped). Floats, booleans and non-ASCII digits are refused because
    they cannot carry a 64-bit identifier exactly. ``label`` names the
    identifier in the error message.

    Raises:
        ValueError: The value is not an exact positive integer below 2**63.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an exact integer or decimal string")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, str):
        if not value.isascii() or not value.isdecimal() or len(value.lstrip("0")) > _MAX_DIGITS:
            raise ValueError(f"{label} must be a positive signed-64-bit decimal integer")
        number = int(value)
    else:
        raise ValueError(f"{label} must be an exact integer or decimal string")
    if not 0 < number < 2**63:
        raise ValueError(f"{label} must be a positive signed-64-bit decimal integer")
    return str(number)


def gaia_identifier(value) -> str:
    """Keep native signed-64-bit Gaia identifiers exact; reject floating-point IDs."""
    return exact_identifier(value, "Gaia source_id")


def native_identifier(value, label: str = "catalogue identifier") -> str:
    """Preserve an arbitrary catalogue's own identifier text.

    Unlike :func:`exact_identifier`, this does not impose a bit width or a
    decimal-only spelling: PIC names, 2MASS designations and VIRAC identifiers
    are legitimate native identities. Integers are accepted and rendered in
    decimal without a signed-64-bit bound; floats/bools/bytes are refused
    because their text is not the catalogue's identity. Surrounding whitespace
    is removed, embedded whitespace and control characters are refused, and an
    empty/blank value is *missing*, not a valid identifier.

    Raises:
        ValueError: empty, non-text, or containing ASCII control characters.
    """
    if isinstance(value, (bool, np.bool_)) or isinstance(value, (float, np.floating, bytes)):
        raise ValueError(f"{label} must be exact catalogue text")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueError(f"{label} must be exact catalogue text")
    if not text:
        raise ValueError(f"{label} is missing")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{label} contains control characters")
    return text


def identity_status(value, label: str = "catalogue identifier") -> str:
    """Classify a raw identity cell as ``missing``, ``malformed`` or ``exact``.

    ``None``/blank is missing; a value that :func:`native_identifier` refuses is
    malformed. This helper never raises for a bad cell so a caller can retain
    the distinction in a record instead of aborting the whole retrieval.
    """
    if value is None:
        return "missing"
    if isinstance(value, str) and not value.strip():
        return "missing"
    try:
        native_identifier(value, label)
    except ValueError:
        return "malformed"
    return "exact"


@dataclass(frozen=True)
class CatalogIdentity:
    """One source's exact native identity and the frame/epoch of its position.

    Attributes:
        namespace: Allowlisted catalogue namespace (``gaia-dr3``, ``tic``,
            ``plato-pic``, ``2mass`` ...). Lowercase text; no implicit alias.
        native_id: The catalogue's own identifier text, preserved exactly.
        frame: Coordinate frame label from :data:`CATALOG_FRAMES`. This is the
            frame of the coordinates the caller holds, not a conversion.
        epoch_jyear: Julian year of the coordinates, or ``None`` when the
            catalogue does not declare one. Never substituted with J2000.
        cross_identifiers: Optional ``(namespace, native_id)`` pairs retained by
            the provider. They are associations, not proof of the same source.
    """

    namespace: str
    native_id: str
    frame: str = "ICRS"
    epoch_jyear: Optional[float] = None
    cross_identifiers: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        namespace = self.namespace
        if not isinstance(namespace, str) or not namespace.strip() or namespace != namespace.strip().lower():
            raise ValueError("catalog namespace must be nonempty lowercase text")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "native_id", native_identifier(self.native_id, "native catalogue identifier"))
        frame = self.frame
        if not isinstance(frame, str) or frame.upper() not in CATALOG_FRAMES:
            raise ValueError("catalog frame must be one of: " + ", ".join(CATALOG_FRAMES))
        object.__setattr__(self, "frame", frame.upper())
        if self.epoch_jyear is not None:
            if isinstance(self.epoch_jyear, (bool, np.bool_)):
                raise ValueError("catalog coordinate epoch must be a Julian year or missing")
            epoch = float(self.epoch_jyear)
            if not np.isfinite(epoch):
                raise ValueError("catalog coordinate epoch must be finite or missing")
            object.__setattr__(self, "epoch_jyear", epoch)
        pairs = []
        for entry in self.cross_identifiers:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("cross identifiers must be (namespace, native_id) pairs")
            other_namespace, other_id = entry
            if not isinstance(other_namespace, str) or not other_namespace.strip() or (
                other_namespace.strip().lower() != other_namespace
            ):
                raise ValueError("cross-identifier namespace must be nonempty lowercase text")
            pairs.append((other_namespace, native_identifier(other_id, "cross-identifier")))
        namespaces = [pair[0] for pair in pairs]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("cross identifiers must name each namespace at most once")
        object.__setattr__(self, "cross_identifiers", tuple(pairs))

    def to_record(self) -> dict:
        """JSON-ready identity record; missing epoch stays ``None``."""
        return {
            "namespace": self.namespace,
            "native_id": self.native_id,
            "frame": self.frame,
            "epoch_jyear": self.epoch_jyear,
            "cross_identifiers": [{"namespace": namespace, "native_id": native_id}
                                  for namespace, native_id in self.cross_identifiers],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CatalogIdentity":
        """Revalidate a serialized identity; unknown fields are refused."""
        if not isinstance(record, Mapping) or set(record) - {
            "namespace", "native_id", "frame", "epoch_jyear", "cross_identifiers",
        }:
            raise ValueError("catalog identity record is malformed")
        cross = record.get("cross_identifiers", ())
        if not isinstance(cross, (tuple, list)):
            raise ValueError("catalog identity cross identifiers must be a list")
        pairs = []
        for entry in cross:
            if not isinstance(entry, Mapping) or set(entry) != {"namespace", "native_id"}:
                raise ValueError("catalog identity cross identifier is malformed")
            pairs.append((entry["namespace"], entry["native_id"]))
        return cls(
            namespace=record.get("namespace"),
            native_id=record.get("native_id"),
            frame=record.get("frame", "ICRS"),
            epoch_jyear=record.get("epoch_jyear"),
            cross_identifiers=tuple(pairs),
        )

    def matches(self, other: "CatalogIdentity") -> bool:
        """Whether two identities name the same catalogue entry exactly.

        Cross-identifiers are deliberately ignored: an associated TIC row does
        not make two different Gaia identities the same source.
        """
        return (
            isinstance(other, CatalogIdentity)
            and self.namespace == other.namespace
            and self.native_id == other.native_id
        )


def catalog_identity(
    namespace: str,
    native_id,
    *,
    frame: str = "ICRS",
    epoch_jyear: Optional[float] = None,
    cross_identifiers: Iterable[Tuple[str, str]] = (),
) -> CatalogIdentity:
    """Build a validated :class:`CatalogIdentity` from provider-declared fields."""
    return CatalogIdentity(
        namespace=namespace,
        native_id=native_id,
        frame=frame,
        epoch_jyear=epoch_jyear,
        cross_identifiers=tuple(cross_identifiers),
    )


def unique_association(candidates: Iterable[Mapping[str, Any]], *,
                       namespace: str, native_id) -> Tuple[str, tuple]:
    """Resolve an exact-identifier association without a nearest-match fallback.

    ``candidates`` are retained provider rows/provenance records carrying
    ``namespace`` and ``native_id``. Returns ``("empty", ())`` when no candidate
    matches, ``("unique", (candidate,))`` for exactly one, and
    ``("ambiguous", candidates)`` with every match when more than one does.
    A blank or malformed requested identity raises: it is a caller error, not
    a no-match result.
    """
    requested = native_identifier(native_id, "requested native catalogue identifier")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("requested catalog namespace must be nonempty text")
    wanted = namespace.strip().lower()
    matches = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("association candidates must be mappings")
        candidate_namespace = candidate.get("namespace")
        if not isinstance(candidate_namespace, str) or candidate_namespace.strip().lower() != wanted:
            continue
        candidate_id = candidate.get("native_id")
        if identity_status(candidate_id) != "exact":
            # A malformed retained candidate is not silently dropped: it is
            # reported as unusable rather than being treated as a non-match.
            raise ValueError("association candidate carries a malformed native identifier")
        if native_identifier(candidate_id) == requested:
            matches.append(candidate)
    if not matches:
        return "empty", ()
    if len(matches) > 1:
        return "ambiguous", tuple(matches)
    return "unique", (matches[0],)


__all__ = [
    "ASSOCIATION_STATUSES",
    "CATALOG_FRAMES",
    "CatalogIdentity",
    "catalog_identity",
    "exact_identifier",
    "gaia_identifier",
    "identity_status",
    "native_identifier",
    "unique_association",
]
