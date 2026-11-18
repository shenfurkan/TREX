"""Target-neutral broad synthetic atmospheres and independent aperture streams."""

import hashlib

import numpy as np
from astropy.time import Time

from trex.app.scene import TargetScene
from trex.photometry.exposure import ExposureResponse
from trex.photometry.observations import Observation, ObservationSet
from trex.photometry.time_axis import TimeAxis
from trex.stellar_light.components import StellarLight
from trex.stellar_light.interpolation import PhotonAtmosphereGrid
from trex.stellar_light.release import RELEASE
from tests.fixtures.trex_scene import caller_provenance

REFERENCE = Time(2459000., .125, format="jd", scale="tdb")


def observation(name="space", *, time=None, flux=None, sigma=.02, shares=None, baseline=None):
    time = np.array([-.04, 0., .04]) if time is None else np.asarray(time)
    return Observation(
        dataset_id=name, segment_id=name, band_id=f"synthetic/{name}", response_id=f"asset-{name}",
        time_axis=TimeAxis(time, REFERENCE, .001, "mid", "barycentric", "synthetic observer",
                           "synthetic target", "analytic test times"),
        flux=np.ones(time.size) if flux is None else flux, sigma=sigma,
        exposure_response=ExposureResponse.boxcar(.001, nsamples=4, source="synthetic box"),
        aperture_id=f"aperture-{name}", source_shares={"target": 1.} if shares is None else shares,
        lineage=tuple((f"original-{name}-{i}",) for i in range(time.size)),
        raw_rows=np.arange(time.size), raw_count=time.size, uncertainty_source="synthetic fixed Gaussian",
        baseline=baseline,
    )


def light(band="space", *, u1=.2, u2=.3):
    axes = tuple(np.array(a) for a in ([1000., 50000.], [-2., 8.], [-5., 2.], [0., 3.]))
    centre = np.broadcast_to(axes[0][:, None, None, None], (2, 2, 2, 2))
    table = centre[..., None] * np.array([np.pi*(1-u1/3-u2/6), 1., -u1, -u2])
    return StellarLight(PhotonAtmosphereGrid(axes, table, {
        "release": RELEASE, "test": "synthetic; not a stellar calibration",
        "response_id": f"synthetic/{band}", "passband_input_id": f"asset-{band}",
    }), 0.)


def joint_case(tmp_path, *, auxiliary_zero=False):
    path = tmp_path / "population.csv"
    path.write_text("TESS,RED,Mact,logg,logTe,[M/H],Av\n"
                    "13.,12.,.8,4.5,3.70,0.,.03\n"
                    "14.,13.,.7,4.6,3.65,-.2,.05\n#TRILEGAL\n", encoding="utf-8")
    rows = (
        observation(shares={"target": .9, "neighbor": .1}),
        observation("ground", time=np.array([-.07, -.02, .03, 1.48, 1.51]),
                    flux=np.array([1., .999, 1., 1.002, 1.]), sigma=.1,
                    shares={"target": 0. if auxiliary_zero else 1., "neighbor": 0.}),
    )
    datasets = [
        dict(dataset_id=row.dataset_id,
             band={"id": row.band_id, "population_column": column},
             aperture={"id": row.aperture_id, "kind": "synthetic", "denominator": "aperture-total"},
             shares={key: value.tolist() for key, value in row.source_shares.items()},
             magnitudes={"target": {"value": mag, "uncertainty": .01}})
        for row, column, mag in zip(rows, ("TESS", "RED"), (10., 9.), strict=True)]
    scene = TargetScene(
        tic_id=123, ra_deg=90., dec_deg=-60., M_s_Msun=1., R_s_Rsun=1., Teff_K=5700.,
        Tmag=10., plx_mas=10., feh_dex=0., metallicity_provenance=caller_provenance(0.),
        sectors=[1], target_aperture_flux_fractions=[{"sector": 1, "fraction": .9}],
        target_source_id="target", datasets=datasets,
        resolved_neighbors=[dict(source_id="neighbor", M_s=.8, R_s=.8, delta_mag=2.,
                                 separation_arcsec=1., aperture_flux_fractions=[{"sector": 1, "fraction": .1}],
                                 stellar_atmosphere=dict(teff_k=5000., mh_dex=0., av_mag=0., source="synthetic"))],
        N_background=2, trilegal_cache=path,
        contrast_separations=None, contrast_values=None,
        background_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), background_area_sq_deg=.1,
        unresolved_resolution_limit_arcsec=10.,
        # Two observed bands; the shared background faint limit must be
        # declared instead of following dataset order (D-79).
        background_selection_band="synthetic/space",
        background_completeness="not modeled: synthetic fixture population",
    )
    return scene, ObservationSet(rows, REFERENCE), {row.band_id: light(row.dataset_id) for row in rows}
