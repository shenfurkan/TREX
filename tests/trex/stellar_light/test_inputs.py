"""Pin demonstrated TRILEGAL extinction loss with unchanged apparent magnitudes."""

import numpy as np

from trex.populations.background import load_trilegal_population
from trex.evidence.nested_population import background_star_population


def test_extinction_stays_aligned_through_population_and_draw(tmp_path):
    path = tmp_path/"stars.csv"
    path.write_text("TESS,logTe,logg,Mact,[M/H],Av,m-M0\n"
                    "14,3.7,4.5,.8,-.2,.17,10\n13,3.6,4.7,.5,.1,.03,8\n")
    rows = load_trilegal_population(path, 12.)
    np.testing.assert_array_equal(rows["av_mag"], [.17, .03])
    block = background_star_population(rows, 1., 12., resolution_limit_arcsec=2.)
    draw = block.resolve(np.array([.1, .5]))
    assert draw.av_mag == .03
    assert draw.mh_dex == .1
    assert draw.flux_fraction == 10**(-.4)/(1+10**(-.4))
