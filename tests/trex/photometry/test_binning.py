"""Pin the extracted observed-bin arithmetic, missing bins and edge ownership."""

import math

import numpy as np

from trex.photometry.binning import weighted_phase_bins


def test_reported_weight_bin_extraction_preserves_exact_arithmetic_and_edges():
    phase = np.array([-.5, -.25, -.125, 0., .25, .5])
    flux = np.array([.99, 1., 1.01, 1.02, 1.03, 1.04])
    sigma = np.array([.01, .02, .03, .04, .05, .06])
    edges = np.array([-.5, -.25, 0., .125, .25, .5])
    result = weighted_phase_bins(phase, flux, sigma, edges)
    # Independent explicit index membership includes the final endpoint once.
    memberships = [[0], [1, 2], [3], [], [4, 5]]
    values = []
    for index, members in enumerate(memberships):
        if not members:
            continue
        weights = np.reciprocal(np.square(sigma[members]))
        total = float(np.sum(weights))
        values.append((index, float(np.sum(weights*phase[members])/total),
                       float(np.sum(weights*flux[members])/total), math.sqrt(1/total), len(members)))
    np.testing.assert_array_equal(result["bin_indices"], [v[0] for v in values])
    np.testing.assert_array_equal(result["time_days"], [v[1] for v in values])
    np.testing.assert_array_equal(result["flux"], [v[2] for v in values])
    np.testing.assert_array_equal(result["flux_err"], [v[3] for v in values])
    np.testing.assert_array_equal(result["counts"], [v[4] for v in values])
    assert sum(result["counts"]) == len(phase)
