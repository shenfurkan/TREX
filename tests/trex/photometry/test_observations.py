"""Physical observation identity, photon independence and portable transport."""

from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from trex.photometry.observations import ObservationSet
from tests.fixtures.joint_observations import REFERENCE, observation


def test_json_and_spawn_transport_preserve_raw_rows_unknown_shares_and_split_time():
    row = observation(shares={"target": .8, "absent": 0., "unknown": None})
    row = replace(row, raw_rows=np.array([1, 3, 7]), raw_count=8)
    original = ObservationSet((row,), REFERENCE)
    for restored in (ObservationSet.from_dict(json.loads(json.dumps(original.to_dict()))),
                     pickle.loads(pickle.dumps(original))):
        assert restored.to_dict() == original.to_dict()
        retained = restored.observations[0]
        assert not retained.flux.flags.writeable and not retained.source_shares["target"].flags.writeable
        assert retained.source_shares["unknown"] is None


def test_simultaneous_streams_are_legal_but_renamed_reused_photons_are_not():
    first, other = observation(), observation("ground")
    ObservationSet((first, other), REFERENCE)
    reused = replace(other, lineage=first.lineage)
    with pytest.raises(ValueError, match="overlapping"):
        ObservationSet((first, reused), REFERENCE)
    derived = replace(other, lineage=(("new-0",), ("new-1", "original-space-1"), ("new-2",)))
    with pytest.raises(ValueError, match="overlapping"):
        ObservationSet((first, derived), REFERENCE)


@pytest.mark.parametrize("changes", [
    {"source_shares": {"target": .9, "neighbor": .2}},
    {"source_shares": {"target": [0., np.nan, 1.]}},
    {"sigma": [1., 0., 1.]},
    {"raw_rows": np.array([0, 0, 1])},
    {"band_id": "R"},
])
def test_invalid_physical_observation_refuses(changes):
    with pytest.raises(ValueError):
        replace(observation(), **changes)
