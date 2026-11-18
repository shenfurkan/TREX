"""Common observation/light binding for existing MC and nested scenario transforms."""

import copy
from collections.abc import Mapping

import numpy as np

from trex.foundation.errors import TrexEvidenceUnavailableError
from trex.models.joint_likelihood import JointLikelihood


class ScenarioObservations:
    """Bind exact physical band assets once, then evaluate one shared draw.

    ``epoch_days`` is relative to the observation set's TDB reference and is
    shared by all visits. ``host_id`` identifies the scene source group whose
    measured aperture shares contain the unresolved system. A population
    adapter must supply its SAME draw's per-dataset dilution fractions; this
    owner never draws another background star, position or occurrence prior.
    """

    def __init__(self, likelihood: JointLikelihood, *, host_id: str, band_light: Mapping,
                 epoch_days: float, target_magnitudes=None):
        if not isinstance(likelihood, JointLikelihood):
            raise ValueError("scenario observations require a prepared JointLikelihood")
        if (not isinstance(host_id, str) or not host_id.strip()
                or isinstance(epoch_days, (bool, np.bool_)) or np.ndim(epoch_days)
                or not np.isfinite(epoch_days)):
            raise ValueError("scenario observations require an exact source ID and finite shared TDB epoch")
        self.likelihood = likelihood
        self.host_id = host_id
        self.epoch_days = float(epoch_days)
        from trex.stellar_light.components import BandStellarLight

        if isinstance(band_light, BandStellarLight):
            self.band_light = {band: band_light.light(band) for band in band_light.bands}
        elif isinstance(band_light, Mapping):
            self.band_light = dict(band_light)
        else:
            raise ValueError("band_light requires per-band StellarLight values or BandStellarLight")
        self.target_magnitudes = dict(target_magnitudes or {})
        for row in likelihood.observations.observations:
            light = self.band_light.get(row.band_id)
            if light is None:
                raise TrexEvidenceUnavailableError(f"missing physical atmosphere for band {row.band_id!r}")
            source = light.grid.source
            if source.get("response_id") != row.band_id or source.get("passband_input_id") != row.response_id:
                raise ValueError(f"atmosphere response/model asset differs from dataset {row.dataset_id!r}")

    @property
    def population_datasets(self):
        return tuple({"dataset_id": row.dataset_id, "band": row.band_id,
                      "target_magnitude": self.target_magnitudes.get(row.dataset_id)}
                     for row in self.likelihood.observations.observations)

    def for_host(self, host_id, *, av_mag=None):
        other = copy.copy(self)
        other.host_id = str(host_id)
        if av_mag is not None:
            from dataclasses import replace

            other.band_light = {band: replace(light, target_av_mag=av_mag)
                                for band, light in self.band_light.items()}
        return other

    def proposal_fraction(self, host_id):
        """Positive proposal rescaling; every physical aperture share stays unchanged.

        Unity for an all-zero-share source is merely an invertible proposal
        coordinate. Its actual predictions stay flat in every dataset.
        """
        fractions = []
        for row in self.likelihood.observations.observations:
            fraction = row.source_shares.get(str(host_id))
            if fraction is None:
                raise TrexEvidenceUnavailableError(
                    f"source {host_id!r} has unknown aperture light in dataset {row.dataset_id!r}")
            fractions.append(float(np.max(fraction)))
        return max(fractions) or 1.

    def bind_scene(self, scene):
        """Refuse mismatched scene apertures/shares before starting any draws."""
        rows = self.likelihood.observations.observations
        if set(scene.dataset_ids) != {row.dataset_id for row in rows}:
            raise ValueError("scene and observation set must declare exactly the same datasets")
        if scene.target_source_id != self.host_id:
            raise ValueError("joint host must equal the scene's exact target source identity")
        sources = {self.host_id, *(str(n["source_id"]) for n in scene.resolved_neighbors)}
        for row in rows:
            dataset = scene.dataset(row.dataset_id)
            if dataset["band"]["id"] != row.band_id or dataset["aperture"]["id"] != row.aperture_id:
                raise ValueError("scene and observation dataset have different bands/apertures")
            if set(row.source_shares) != sources:
                raise ValueError("each observation must retain exactly the scene's sources")
            for source in sources:
                share = row.source_shares[source]
                if share is None:
                    raise TrexEvidenceUnavailableError(
                        f"source {source!r} has unknown light in dataset {row.dataset_id!r}")
                if not np.array_equal(share, np.broadcast_to(
                        scene.dataset_flux_fraction(row.dataset_id, source), share.shape)):
                    raise ValueError("observation source shares differ from the calibrated scene")
        magnitudes = {entry["dataset_id"]: entry["target_magnitude"]
                      for entry in scene.population_dataset_bands()}
        if self.target_magnitudes and self.target_magnitudes != magnitudes:
            raise ValueError("joint population magnitude anchors differ from the scene")
        self.target_magnitudes = magnitudes

    def provenance(self):
        return {
            "time_domain": "native-barycentric-TDB-days-from-shared-split-reference",
            "reference_jd1": float(self.likelihood.observations.reference.jd1),
            "reference_jd2": float(self.likelihood.observations.reference.jd2),
            "epoch_days": self.epoch_days,
            "density": "normalized-aperture-Gaussian",
            "datasets": [
                {"dataset_id": row.dataset_id, "band_id": row.band_id, "response_id": row.response_id,
                 "aperture_id": row.aperture_id, "n_retained": row.flux.size, "n_raw": int(row.raw_count),
                 "uncertainty_source": row.uncertainty_source,
                 "exposure_response": row.exposure_response.source,
                 "baseline": None if row.baseline is None else row.baseline.provenance()}
                for row in self.likelihood.observations.observations],
            "stellar_light": {band: light.model_configuration() for band, light in self.band_light.items()},
            "shared_baseline": (None if self.likelihood.shared is None else self.likelihood.shared.prior.provenance()),
        }

    def evaluate(self, *, host_mass, host_radius, host_teff, host_metallicity, period_days,
                 inc_deg, a_cm, ecc, argp_deg, planet_radius=None, secondary_mass=None,
                 secondary_radius=None, secondary_teff=None, host_av=None,
                 companion_is_host=False, dataset_dilutions=None, return_diagnostics=False):
        """Return normalized aperture log density using each body's own light.

        Temperature [K], mass/radius [solar], planet radius [earth], axis [cm],
        time/period [TDB days], angles [degrees], composition [dex], extinction
        [mag]. Missing host characterization/native-band dilution raises an
        evidence-unavailable error, not a fabricated color or zero likelihood.
        """
        if host_teff is None or host_metallicity is None:
            raise TrexEvidenceUnavailableError("joint observations require this draw's own temperature/composition")
        light_by_dataset = {}
        for row in self.likelihood.observations.observations:
            light = self.band_light[row.band_id]
            av = host_av if host_av is not None else light.target_av_mag
            if av is None:
                raise TrexEvidenceUnavailableError("joint atmosphere requires this source's own extinction")
            primary = (host_teff, host_mass, host_radius)
            if secondary_radius is None:
                first = light.prepare_component(*primary, host_metallicity, av)
                values = {"u1": float(first.u1), "u2": float(first.u2),
                          "angular_profile": first.angular_profile}
            else:
                if secondary_mass is None or secondary_teff is None:
                    raise TrexEvidenceUnavailableError("joint EB requires its actual secondary mass and temperature")
                ratio, first, second = light.prepare_pair(
                    primary, (secondary_teff, secondary_mass, secondary_radius), host_metallicity, av)
                values = {"u1": float(first.u1), "u2": float(first.u2),
                          "u1_EB": float(second.u1), "u2_EB": float(second.u2),
                          "secondary_to_host_flux_ratio": float(ratio),
                          "angular_profile": first.angular_profile,
                          "secondary_angular_profile": second.angular_profile}
            if dataset_dilutions is not None:
                fraction = dataset_dilutions.get(row.dataset_id)
                if fraction is None or not np.isfinite(fraction):
                    raise TrexEvidenceUnavailableError(
                        f"population draw has missing native-band dilution in {row.dataset_id!r}")
                values["companion_fluxratio"] = float(fraction)
            light_by_dataset[row.dataset_id] = values
        value, diagnostics = self.likelihood(
            host_id=self.host_id, light_by_dataset=light_by_dataset,
            period_days=period_days, epoch_days=self.epoch_days, inc_deg=inc_deg, a_cm=a_cm,
            host_radius_solar=host_radius, planet_radius_earth=planet_radius,
            secondary_radius_solar=secondary_radius, ecc=ecc, argp_deg=argp_deg,
            companion_is_host=companion_is_host)
        return (value, diagnostics) if return_diagnostics else value
