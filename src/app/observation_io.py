"""Version-two portable observations and exact compiled stellar-light assets."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import stat

import numpy as np

from trex.acquisition.storage import write_bytes, write_json
from trex.evidence.joint_observations import ScenarioObservations
from trex.models.joint_likelihood import JointLikelihood
from trex.photometry.baseline import GaussianBaseline
from trex.photometry.observations import ObservationSet
from trex.stellar_light.components import StellarLight
from trex.stellar_light.interpolation import PhotonAtmosphereGrid


def load_observation_binding(directory, payload):
    """Read only named regular local assets; validate physical and byte identities."""
    directory = Path(directory)
    observations = ObservationSet.from_dict(payload["observations"])
    shared = payload.get("shared_baseline")
    shared = None if shared is None else GaussianBaseline(**shared)
    lights, artifacts = {}, {}
    for band, record in payload["band_light"].items():
        name = record["file"]
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("compiled light asset must name a file in the prepared input directory")
        path = directory / name
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("compiled light assets must be regular non-symlink files")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError("compiled light asset bytes changed")
        grid = PhotonAtmosphereGrid.from_file(
            path, expected_model_id=record["model_id"], expected_response_id=band,
            expected_passband_input_id=record["response_id"])
        lights[band] = StellarLight(grid, record["target_av_mag"])
        artifacts[str(path.absolute())] = digest
    likelihood = JointLikelihood(observations, shared_baseline=shared)
    joint = ScenarioObservations(likelihood, host_id=payload["scene"]["target_source_id"],
                                 band_light=lights, epoch_days=payload["epoch_days"])
    first = observations.observations[0]
    return {"joint_observations": joint, "stellar_light": lights[first.band_id]}, artifacts


def export_observation_input(scene, observations, output, *, period_days, depth_ppm,
                             epoch_days, band_light, shared_baseline=None, random_seed=None,
                             depth_sigma_ppm=None, duration_days=None, duration_sigma_days=None):
    """Exclusively publish v2 inputs for the existing ``python -m trex.run``.

    Arrays carry relative fractional aperture flux/errors and split TDB time;
    priors and temporal kernels are explicit, not reconstructed from labels.
    Compiled grids retain all selected band/extinction/angular arrays. No
    catalog or network operation occurs. An observable publication failure
    removes only this invocation's partial files, never a prior directory.
    """
    from trex.app.engine import prepare_vetting_inputs

    likelihood = JointLikelihood(observations, shared_baseline=shared_baseline)
    joint = ScenarioObservations(likelihood, host_id=scene.target_source_id,
                                 band_light=band_light, epoch_days=epoch_days)
    first = observations.observations[0]
    time = likelihood.times[0] - joint.epoch_days
    prepare_vetting_inputs(scene, time, first.flux, first.sigma, period_days, depth_ppm, None,
                          stellar_light=band_light[first.band_id], joint_observations=joint)
    binding = dict(sector_weights={}, exptime_days=None, depth_ppm=depth_ppm, random_seed=random_seed,
                   depth_sigma_ppm=depth_sigma_ppm, duration_days=duration_days,
                   duration_sigma_days=duration_sigma_days)
    scene_payload = scene.to_dict()
    background = Path(scene_payload.pop("trilegal_cache"))
    payload = dict(version=2, scene=scene_payload, period_days=period_days, binding=binding,
                   epoch_days=joint.epoch_days, observations=observations.to_dict(),
                   shared_baseline=None if shared_baseline is None else shared_baseline.to_dict(), band_light={})
    json.dumps(payload, allow_nan=False)
    buffer = io.BytesIO()
    np.savez(buffer, time=time, flux=first.flux, sigma=first.sigma, bin_integration_days=0.)
    output = Path(output)
    output.mkdir(parents=True)
    claimed = output.stat()
    owned = []
    try:
        owned.append("background.csv")
        with background.open("rb") as source, (output / "background.csv").open("xb") as destination:
            shutil.copyfileobj(source, destination)
        for index, (band, light) in enumerate(sorted(joint.band_light.items())):
            name = f"light-{index:03d}.npz"
            owned.append(name)
            light.grid.to_file(output / name)
            payload["band_light"][band] = {
                "file": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                "model_id": light.model_id, "response_id": light.grid.source["passband_input_id"],
                "target_av_mag": light.target_av_mag,
            }
        owned.append("arrays.npz")
        write_bytes(output / "arrays.npz", buffer.getvalue())
        owned.append("trex-input.json")
        write_json(output / "trex-input.json", payload)
    except BaseException as exc:
        try:
            current = output.stat()
            if (current.st_dev, current.st_ino) != (claimed.st_dev, claimed.st_ino):
                raise OSError("partial output directory was replaced; left untouched")
            for name in owned:
                (output / name).unlink(missing_ok=True)
            output.rmdir()
        except OSError as cleanup:
            exc.add_note(f"could not remove this invocation's partial export: {cleanup}")
        raise
    return output
