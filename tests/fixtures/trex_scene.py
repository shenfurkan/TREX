"""Portable scene-provenance fixtures for TREX's own test suite."""

from __future__ import annotations


def caller_provenance(
    value_dex: float,
    *,
    quantity: str = "[Fe/H]",
    uncertainty_dex: float = 0.1,
    citation: str = "Synthetic test value; not a measurement of a real source",
) -> dict:
    """Return the portable caller-supplied metallicity provenance shape."""
    return {
        "source": "caller-supplied",
        "quantity": quantity,
        "value_dex": value_dex,
        "uncertainty_dex": uncertainty_dex,
        "citation": citation,
    }
