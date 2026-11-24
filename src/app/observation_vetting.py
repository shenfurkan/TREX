"""Public native-time, multi-aperture entry to the existing scenario engine."""

from trex.evidence.joint_observations import ScenarioObservations
from trex.models.joint_likelihood import JointLikelihood


def run_observation_vetting(scene, observations, *, period_days, depth_ppm, epoch_days,
                           band_light, shared_baseline=None, **controls):
    """Run every scenario on independent native-time aperture observations.

    ``observations`` is an ObservationSet, ``epoch_days`` is conjunction at its
    split-TDB reference, period is days and depth is ppm. Depth only positions
    defensive proposals; each stream keeps its actual times, fractional
    aperture flux/errors, exposure quadrature, light and calibrated shares.
    The first stream supplies proposal coordinates, never the joint density.
    Per-band StellarLight assets must match the observations' exact identities.
    ``shared_baseline`` is one proper Gaussian prior across all streams.

    Other controls are run_trex_vetting's draw/seed/worker/recovery controls.
    The scalar dynesty route evaluates every scenario. A native-only estimator
    is admitted only when the scene produces no population-hosted/diluted
    scenario; the fixed-host TP/EB scenarios then use the device twin of this
    same joint density, while an uncovered combination refuses before draws
    rather than silently computing a single-band model.
    """
    from trex.app.engine import run_trex_vetting

    likelihood = JointLikelihood(observations, shared_baseline=shared_baseline)
    joint = ScenarioObservations(likelihood, host_id=scene.target_source_id,
                                 band_light=band_light, epoch_days=epoch_days)
    primary = observations.observations[0]
    return run_trex_vetting(
        scene, likelihood.times[0] - joint.epoch_days, primary.flux, primary.sigma,
        period_days, depth_ppm, exptime_days=None,
        stellar_light=joint.band_light[primary.band_id], joint_observations=joint, **controls,
    )
