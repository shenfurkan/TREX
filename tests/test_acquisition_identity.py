"""Catalogue identifiers stay exact across the acquisition boundary."""

import numpy as np
import pytest

from trex.acquisition.identity import (
    CatalogIdentity,
    catalog_identity,
    exact_identifier,
    gaia_identifier,
    identity_status,
    native_identifier,
    unique_association,
)


def test_gaia_identifier_is_exact_and_bounded():
    assert gaia_identifier("06000000000000000001") == "6000000000000000001"
    assert gaia_identifier(9000000000000000001) == "9000000000000000001"
    assert gaia_identifier(np.int64(9000000000000000001)) == "9000000000000000001"
    for value in (True, np.bool_(True), False, None, 1.0, "1.0", "０１", "", " 1", 0, -1, 2**63,
                  "1" * 20, "9" * 5000):
        with pytest.raises(ValueError, match="Gaia source_id"):
            gaia_identifier(value)


def test_identifier_errors_name_the_identifier_actually_checked():
    assert exact_identifier("198213332", "TIC ID") == "198213332"
    with pytest.raises(ValueError, match="^TIC ID must"):
        exact_identifier("TIC 198213332", "TIC ID")


# ---------------------------------------------------------------------------
# Native (non-Gaia) identity
# ---------------------------------------------------------------------------

def test_native_identifier_preserves_non_numeric_catalogue_names():
    assert native_identifier("2MASS J12345678+1234567") == "2MASS J12345678+1234567"
    assert native_identifier("PIC 123.45") == "PIC 123.45"
    assert native_identifier("  VIRAC2-123  ") == "VIRAC2-123"
    # An arbitrary-length integer is still an exact identity; the 64-bit Gaia
    # bound is a property of Gaia, not of every catalogue.
    assert native_identifier(10**30) == "1" + "0" * 30


def test_native_identifier_distinguishes_missing_from_malformed():
    assert identity_status(None) == "missing"
    assert identity_status("   ") == "missing"
    assert identity_status(1.5) == "malformed"
    assert identity_status("bad\x00id") == "malformed"
    assert identity_status("PIC 123.45") == "exact"
    with pytest.raises(ValueError, match="missing"):
        native_identifier("  ", "PIC name")


def test_catalog_identity_keeps_frame_and_native_epoch():
    identity = catalog_identity(
        "gaia-dr3", "6000000000000000001", frame="ICRS", epoch_jyear=2016.0,
        cross_identifiers=(("tic", "198213332"),),
    )
    assert identity.native_id == "6000000000000000001"
    assert identity.epoch_jyear == 2016.0
    assert identity.to_record() == {
        "namespace": "gaia-dr3",
        "native_id": "6000000000000000001",
        "frame": "ICRS",
        "epoch_jyear": 2016.0,
        "cross_identifiers": [{"namespace": "tic", "native_id": "198213332"}],
    }
    assert CatalogIdentity.from_record(identity.to_record()) == identity


def test_missing_epoch_stays_missing_and_is_not_j2000():
    identity = catalog_identity("plato-pic", "PIC 123.45")
    assert identity.epoch_jyear is None
    record = identity.to_record()
    assert record["epoch_jyear"] is None
    assert CatalogIdentity.from_record(record).epoch_jyear is None


def test_catalog_identity_refuses_unknown_frame_and_bad_epoch():
    with pytest.raises(ValueError, match="frame"):
        catalog_identity("gaia-dr3", "1", frame="J2000")
    with pytest.raises(ValueError, match="epoch"):
        catalog_identity("gaia-dr3", "1", epoch_jyear=float("nan"))


def test_cross_identifier_namespaces_are_unique():
    with pytest.raises(ValueError, match="at most once"):
        catalog_identity(
            "gaia-dr3", "1",
            cross_identifiers=(("tic", "2"), ("tic", "3")),
        )


def test_identity_match_ignores_cross_identifiers():
    first = catalog_identity("gaia-dr3", "1", cross_identifiers=(("tic", "2"),))
    second = catalog_identity("gaia-dr3", "1")
    other = catalog_identity("tic", "2", cross_identifiers=(("gaia-dr3", "1"),))
    assert first.matches(second)
    assert not first.matches(other)


# ---------------------------------------------------------------------------
# Association ambiguity
# ---------------------------------------------------------------------------

def test_unique_association_returns_the_only_exact_match():
    candidates = [
        {"namespace": "plato-pic", "native_id": "PIC 1"},
        {"namespace": "tic", "native_id": "PIC 1"},
    ]
    status, matches = unique_association(candidates, namespace="plato-pic", native_id="PIC 1")
    assert status == "unique" and len(matches) == 1
    assert matches[0]["namespace"] == "plato-pic"


def test_association_keeps_every_ambiguous_match():
    candidates = [
        {"namespace": "2mass", "native_id": "2MASS J00000001+0000001", "mag": 12.0},
        {"namespace": "2mass", "native_id": "2MASS J00000001+0000001", "mag": 15.0},
    ]
    status, matches = unique_association(candidates, namespace="2mass",
                                         native_id="2MASS J00000001+0000001")
    assert status == "ambiguous" and len(matches) == 2


def test_association_no_match_is_empty_not_an_error():
    assert unique_association([], namespace="tic", native_id="1") == ("empty", ())


def test_association_refuses_a_malformed_requested_identity():
    with pytest.raises(ValueError):
        unique_association([], namespace="tic", native_id="")
