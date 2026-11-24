"""One physical orbital draw over explicitly independent aperture observations."""

from collections.abc import Mapping
import math

import numpy as np

from trex.foundation.errors import NumericalLikelihoodError, TrexEvidenceUnavailableError
from trex.foundation.constants import Rsun
from trex.models.likelihoods import simulate_EB, simulate_TP
from trex.models.orbital_mechanics import impact_parameter
from trex.photometry.baseline import GaussianBaseline
from trex.photometry.observations import ObservationSet
from trex.populations._validation import real_scalar, boolean


class JointLikelihood:
    """Prepared normalized aperture density over an ObservationSet.

    Each dataset keeps its own reported Gaussian sigma or explicit proper
    baseline prior. No residual-derived factor is shared across streams.
    ``shared_baseline`` uses the existing physical Gaussian convolution over
    concatenated dataset order, with explicit stream IDs. It must include all
    private/shared nuisance coefficients in one prior; multiplying it by
    separately marginalized dataset baselines would count nuisance priors twice.
    """

    def __init__(self, observations: ObservationSet, *, shared_baseline=None):
        if not isinstance(observations, ObservationSet):
            raise ValueError("joint likelihood requires an ObservationSet")
        self.observations = observations
        self.times = tuple(row.time_axis.relative_midpoints(observations.reference)
                           for row in observations.observations)
        self.baselines = tuple(None if row.baseline is None else row.baseline.prepare(time, row.flux, row.sigma)
                               for row, time in zip(observations.observations, self.times))
        self.normalizations = tuple(float(row.flux.size * np.log(2 * np.pi) + 2 * np.log(row.sigma).sum())
                                    for row in observations.observations)
        self.shared = None
        if shared_baseline is not None:
            if not isinstance(shared_baseline, GaussianBaseline) or any(p is not None for p in self.baselines):
                raise ValueError("shared baseline needs one joint prior, not additional per-dataset priors")
            streams = tuple(row.dataset_id for row in observations.observations for _ in row.flux)
            if shared_baseline.stream_ids != streams:
                raise ValueError("shared baseline must declare the observation set's exact stream order")
            self.shared = shared_baseline.prepare(np.concatenate(self.times),
                                                  np.concatenate([row.flux for row in observations.observations]),
                                                  np.concatenate([row.sigma for row in observations.observations]))
        if not np.isfinite(self.normalizations).all():
            raise NumericalLikelihoodError("joint Gaussian normalization cannot be represented finitely")

    def density(self, models: Mapping[str, np.ndarray]) -> tuple[float, dict]:
        """Return normalized log density [nat] and owned per-dataset diagnostics.

        Models are fractional total-aperture means. A shared covariance is
        evaluated once, not reported as a sum of independent marginals.
        """
        expected = {row.dataset_id for row in self.observations.observations}
        if not isinstance(models, Mapping) or set(models) != expected:
            raise ValueError("models must contain exactly the admitted datasets")
        diagnostics, aligned, total = {}, [], 0.
        for row, baseline, normalization in zip(self.observations.observations, self.baselines, self.normalizations):
            model = np.asarray(models[row.dataset_id])
            if model.dtype.kind not in "iuf" or model.shape != row.flux.shape:
                raise ValueError("one aperture model value is required per retained observation")
            if not np.isfinite(model).all():
                raise NumericalLikelihoodError("joint physical model is nonfinite")
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    residual = (row.flux - model) / row.sigma
                    chi2 = float(residual @ residual)
                    value = (float(baseline(model)) if baseline is not None else
                             -.5 * (chi2 + normalization))
            except FloatingPointError as exc:
                raise NumericalLikelihoodError("joint Gaussian residual calculation failed") from exc
            if not math.isfinite(value):
                raise NumericalLikelihoodError("joint Gaussian density is nonfinite")
            total += value
            aligned.append(model)
            diagnostics[row.dataset_id] = {
                "band_id": row.band_id, "response_id": row.response_id, "n_retained": row.flux.size,
                "n_raw": int(row.raw_count), "reported_error_chi2": chi2,
                "log_density": value if self.shared is None else None,
                "uncertainty_source": row.uncertainty_source,
            }
        if self.shared is not None:
            total = self.shared(np.concatenate(aligned))
        if not math.isfinite(total):
            raise NumericalLikelihoodError("joint log density sum is nonfinite")
        return float(total), diagnostics

    def models(self, *, host_id: str, light_by_dataset: Mapping, period_days: float,
               epoch_days: float, inc_deg: float, a_cm: float, host_radius_solar: float,
               planet_radius_earth=None, secondary_radius_solar=None, ecc=0., argp_deg=90.,
               companion_is_host=False):
        """Evaluate the same orbit/host/occultor over every admitted dataset.

        Geometry uses days, cm, solar/earth radii as named and degrees. Epoch is
        relative to the ObservationSet's TDB reference; it is common to all
        visits, not independent per-band timing jitter. ``light_by_dataset``
        supplies each host/body's own coefficients or angular profiles and,
        for EB, its secondary/primary flux ratio. The caller's physical
        population/atmosphere adapter owns those values, never this likelihood.
        """
        if ((planet_radius_earth is None) == (secondary_radius_solar is None)
                or isinstance(epoch_days, (bool, np.bool_)) or np.ndim(epoch_days)
                or not np.isfinite(epoch_days)):
            raise ValueError("supply one occultor family and a finite shared epoch in TDB days")
        for name, value in (("period_days", period_days), ("a_cm", a_cm), ("host_radius_solar", host_radius_solar),
                            ("occultor radius", planet_radius_earth if secondary_radius_solar is None else secondary_radius_solar)):
            real_scalar(value, name, positive=True)
        impact_parameter(inc_deg, a_cm/(host_radius_solar*Rsun), ecc, argp_deg)
        boolean(companion_is_host, "companion_is_host")
        dataset_ids = {row.dataset_id for row in self.observations.observations}
        if not isinstance(light_by_dataset, Mapping) or set(light_by_dataset) != dataset_ids:
            raise ValueError("per-dataset light must cover exactly the admitted observations")
        result = {}
        for row, time in zip(self.observations.observations, self.times):
            share = row.source_shares.get(host_id)
            if share is None:
                raise TrexEvidenceUnavailableError(
                    f"source {host_id!r} has unknown aperture light in dataset {row.dataset_id!r}")
            # A justified exact-zero aperture share predicts flat data; it never
            # deletes this host from the other observations or scenario table.
            if not np.any(share):
                result[row.dataset_id] = np.ones(time.shape)
                continue
            light = light_by_dataset[row.dataset_id]
            common = dict(time=time - epoch_days, P_orb=period_days, inc_deg=inc_deg,
                          a_cm=a_cm, R_s_solar=host_radius_solar, u1=light["u1"], u2=light["u2"],
                          exptime_days=None, ecc=ecc, argp_deg=argp_deg,
                          exposure_response=row.exposure_response,
                          angular_profile=light.get("angular_profile"),
                          companion_fluxratio=light.get("companion_fluxratio", 0.),
                          companion_is_host=companion_is_host)
            if planet_radius_earth is not None:
                intrinsic = simulate_TP(R_p_earth=planet_radius_earth, **common)
            else:
                intrinsic, _ = simulate_EB(
                    R_EB_solar=secondary_radius_solar, EB_fluxratio=None,
                    secondary_to_host_flux_ratio=light["secondary_to_host_flux_ratio"],
                    u1_EB=light["u1_EB"], u2_EB=light["u2_EB"],
                    secondary_angular_profile=light.get("secondary_angular_profile"), **common)
            result[row.dataset_id] = 1 + share * (intrinsic - 1)
        return result

    def __call__(self, **physical_draw):
        return self.density(self.models(**physical_draw))
