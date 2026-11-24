"""Runnable synthetic software demonstration: `python -m trex.example NEW_DIRECTORY`.

This exercises the installed public path with the real engine. The inputs are invented
and the budget is far below the 1,000,000 draws per scenario of production, so the
probabilities mean nothing: they show which files a run leaves, on success or failure.
"""

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np

from trex.foundation.physical_constants import SECONDS_PER_DAY
from trex.app.scene import TargetScene
from trex.app.target import target

DEMONSTRATION_DRAWS = 4096
DEMONSTRATION_SUPERSAMPLING = 4


def synthetic_inputs(directory):
    """Explicit low-information scene for API plumbing, not science calibration.

    Both sectors and every neighbor are retained. The faint missing-parameter
    neighbor cannot host the declared depth; its three rows remain in outputs.
    Large declared sigma tests orchestration without demanding a planet verdict.
    """
    directory.mkdir(parents=True, exist_ok=True)
    background = directory / "synthetic-trilegal.csv"
    background.write_text(
        ",TESS,logTe,logg,Mact,[M/H]\n0,14.0,3.70,4.5,0.80,0.0\n1,16.0,3.60,4.7,0.50,0.0\n"
        "2,#TRILEGAL,normally,terminated,\n", encoding="utf-8",
    )
    target_fractions = [{"sector": 1, "fraction": .85}, {"sector": 2, "fraction": .80}]
    neighbor_fractions = [{"sector": 1, "fraction": .1499}, {"sector": 2, "fraction": .1999}]
    faint_fractions = [{"sector": 1, "fraction": .0001}, {"sector": 2, "fraction": .0001}]
    calibration = directory / "synthetic-aperture-input.json"
    calibration.write_text(json.dumps({
        "synthetic_test_input": True, "claim_eligible": False,
        "target": target_fractions, "neighbor": neighbor_fractions, "faint": faint_fractions,
    }, allow_nan=False))
    scene = TargetScene(
        tic_id=100001, ra_deg=120.0, dec_deg=-30.0,
        M_s_Msun=1.02, R_s_Rsun=.98, Teff_K=5778.0, Tmag=10.0, plx_mas=10.0,
        feh_dex=0.0, sectors=[1, 2],
        metallicity_provenance={"source": "caller-supplied", "quantity": "[Fe/H]",
                                "value_dex": 0.0, "uncertainty_dex": 0.1,
                                "citation": "Invented value for synthetic TREX demonstration; not an observation"},
        contrast_separations=[.1, 1.0, 2.0], contrast_values=[1.0, 3.0, 5.0],
        target_aperture_flux_fractions=target_fractions,
        resolved_neighbors=[
            {"source_id": "100002", "M_s": .82, "R_s": .79, "delta_mag": 1.5,
             "separation_arcsec": 12.0, "aperture_flux_fractions": neighbor_fractions},
            {"source_id": "100003", "M_s": None, "R_s": None, "delta_mag": 10.0,
             "separation_arcsec": 14.4, "aperture_flux_fractions": faint_fractions,
             "stellar_parameter_unavailable_reason": "synthetic missing-observation control"},
        ],
        N_background=2, trilegal_cache=background,
        background_sha256=hashlib.sha256(background.read_bytes()).hexdigest(),
        background_area_sq_deg=.1,
    )
    time = np.linspace(-.08, .08, 80)
    flux = np.where(np.abs(time) < .03, .992, 1.0)
    binding = dict(
        sector_weights={1: 3.0, 2: 1.0}, depth_ppm=8000.0,
        exptime_days=120.0 / SECONDS_PER_DAY, bin_integration_days=60.0 / SECONDS_PER_DAY,
        random_seed=424242,
        input_artifacts={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (calibration, background)},
    )
    return scene, binding, time, flux, np.ones_like(time), 3.2


def main(arguments=None):
    parser = argparse.ArgumentParser(prog="python -m trex.example", description=__doc__)
    parser.add_argument("directory", type=Path, help="New output directory; an existing one is refused.")
    directory = parser.parse_args(arguments).directory
    directory.mkdir()  # Refuse to overwrite an earlier run.
    scene, binding, time, flux, sigma, period = synthetic_inputs(directory/"inputs")
    obj = target.from_scene(scene, **binding)
    try:
        obj.calc_probs(time, flux, sigma, period, exptime=binding["exptime_days"],
                       N=DEMONSTRATION_DRAWS, nsamples=DEMONSTRATION_SUPERSAMPLING,
                       escalation_checkpoint_directory=str(directory / "escalation"))
    except BaseException as error:
        if obj.failure_record is not None:
            try:
                obj.write_outputs(directory / "result")
            except Exception as problem:
                raise problem from error
        raise
    if obj.trex_result is None:
        raise RuntimeError("synthetic inference returned without a result")
    obj.write_outputs(directory / "result")
    print(f"Synthetic demonstration only; see {directory/'result'}. status={obj.trex_result.status}")
    return 0 if obj.trex_result.status == "ok" else 3


if __name__ == "__main__":
    main()
