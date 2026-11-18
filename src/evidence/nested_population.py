"""Population priors for nested evidence."""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np

from trex.populations.companion_imaging import ContrastCurve, ImagingConstraint

from trex.evidence.nested_controls import _POPULATION_N_DIMENSIONS, _BOUND_POPULATION_N_DIMENSIONS


_facade_module = None


def _facade(name):
    global _facade_module
    if _facade_module is None:
        from trex.evidence import nested_evidence

        _facade_module = nested_evidence
    return getattr(_facade_module, name)

class UnresolvedPopulationDraw(NamedTuple):
    """One unresolved diluting source, as the production draw resolves it.

    ``dataset_dilutions`` carries this same physical draw's dilution in every
    bound observation dataset, as ``(dataset_id, flux_fraction)`` pairs. The
    legacy ``flux_fraction`` remains the TESS/first-band value for existing
    single-dataset consumers. An empty tuple means the caller declared no
    dataset bands.
    """

    flux_fraction: float
    mass_solar: float
    radius_solar: float
    teff_k: float
    supported: bool
    imaging: Optional[ImagingConstraint] = None
    mh_dex: Optional[float] = None
    av_mag: Optional[float] = None
    log_period_days: Optional[float] = None
    dataset_dilutions: tuple = ()


class PopulationDataset(NamedTuple):
    """One observation dataset bound to the same physical population draw.

    ``dataset_id`` is the observation identity (not a band alias), ``band`` is
    the native band whose light/magnitude decides the dilution, and
    ``target_magnitude`` is the target's magnitude in that band (required for a
    background star, ignored for a bound companion whose light comes from the
    atmosphere). Declaring a dataset does not add a second population or
    solid-angle prior: it selects a different aperture/band response for the
    one draw.
    """

    dataset_id: str
    band: str
    target_magnitude: Optional[float] = None


def _population_datasets(datasets) -> tuple:
    """Validate a dataset-binding tuple without inventing a band or a target."""
    if datasets is None:
        return ()
    normalized = []
    seen = set()
    for entry in datasets:
        if isinstance(entry, PopulationDataset):
            dataset = entry
        elif isinstance(entry, Mapping):
            unknown = set(entry) - {"dataset_id", "band", "target_magnitude"}
            if unknown:
                raise ValueError("population dataset has unknown fields")
            dataset = PopulationDataset(
                dataset_id=entry.get("dataset_id"),
                band=entry.get("band"),
                target_magnitude=entry.get("target_magnitude"),
            )
        else:
            raise ValueError("population datasets must be PopulationDataset records or mappings")
        if not isinstance(dataset.dataset_id, str) or not dataset.dataset_id.strip():
            raise ValueError("population dataset requires a nonempty dataset_id")
        if not isinstance(dataset.band, str) or not dataset.band.strip():
            raise ValueError("population dataset requires a native band id")
        if dataset.dataset_id in seen:
            raise ValueError("population dataset ids must be unique")
        seen.add(dataset.dataset_id)
        normalized.append(dataset)
    return tuple(normalized)


class _BoundCompanionPopulation(NamedTuple):
    """The unresolved bound-companion population as a prior block.

    Holds only plain data so the block survives pickling into a worker
    process: dynesty evolves points through ``pool.map``, which pickles the
    callable and its arguments, and a closure would not travel.
    """

    host_mass: float
    host_radius: float
    host_teff: float
    parallax_mas: float
    measured_curve: bool
    contrast_separations: Optional[np.ndarray]
    contrast_values: Optional[np.ndarray]
    outer_arcsec: float
    log_occurrence_weight: float
    n_dimensions: int = _BOUND_POPULATION_N_DIMENSIONS
    stellar_light: object = None
    target_feh: Optional[float] = None
    contrast_curve: Optional[ContrastCurve] = None
    datasets: tuple = ()
    band_light: Optional[Mapping[str, object]] = None

    def _band_pair_ratio(self, band, companion_mass, companion_radius, companion_teff):
        """F/(F_host) in one native band from Agent 1's per-band light.

        The TESS entry may fall back to the legacy single-band ``stellar_light``;
        any other band requires its own atmosphere grid. A missing band is not
        approximated from TESS, because that would fabricate the colour.
        """
        light = None if self.band_light is None else self.band_light.get(band)
        if light is None and band.strip().lower() == "tess":
            light = self.stellar_light
        if light is None:
            return None
        ratio, _, _ = light.pair(
            (self.host_teff, self.host_mass, self.host_radius),
            (companion_teff, companion_mass, companion_radius),
            self.target_feh, self.stellar_light.target_av_mag if self.stellar_light is not None
            else getattr(light, "target_av_mag", None),
        )
        return float(ratio)

    def resolve(self, unit_point: np.ndarray, *, defer_imaging=False) -> "UnresolvedPopulationDraw":
        """One companion draw, as ``_draw_bound_companion`` resolves it."""
        point = np.asarray(unit_point, dtype=float)
        unsupported = _facade("UnresolvedPopulationDraw")(
            0.0, self.host_mass, self.host_radius, self.host_teff, False
        )
        m_dwarf = self.host_mass <= _facade("WINTERS_2019_M_DWARF_MAXIMUM_MASS_SOLAR")
        expected = _BOUND_POPULATION_N_DIMENSIONS
        if point.shape != (expected,) or self.n_dimensions != expected:
            raise ValueError(f"bound population requires {expected} unit dimensions; old-dimensional checkpoint cannot be resumed")
        if not np.isfinite(point).all() or np.any(point < 0) or np.any(point > 1):
            raise ValueError("bound population variates must be finite and in [0, 1]")
        if m_dwarf:
            if point[0] == 0 or point[0] == 1:
                return unsupported
            log_axis = float(
                _facade("sample_companion_log_semimajor_axis")(
                    np.array([point[0]]), self.host_mass
                )[0]
            )
            if not math.isfinite(log_axis):
                return unsupported
            mass_ratio = float(
                _facade("sample_q_companion")(
                    np.array([point[1]]), self.host_mass, None
                )[0]
            )
            companion_mass = mass_ratio * self.host_mass
            log_period = float(
                _facade("_companion_log_period_from_log_separation")(
                    self.host_mass + companion_mass, np.array([log_axis])
                )[0]
            )
            if not math.isfinite(log_period):
                # A unit-cube edge maps to an infinite separation; it has zero measure
                # and no support, as on the MC and JAX paths.
                return unsupported
        else:
            log_period = float(
                _facade("sample_companion_log_period")(np.array([point[0]]), self.host_mass)[0]
            )
            if not (
                _facade("MOE2017_LOGPERIOD_DOMAIN_MIN_DAYS") <= log_period <= _facade("MOE2017_LOGPERIOD_MAX_DAYS")
            ):
                return unsupported
            mass_ratio = float(
                _facade("sample_q_companion")(
                    np.array([point[1]]), self.host_mass, np.array([log_period])
                )[0]
            )
            companion_mass = mass_ratio * self.host_mass
            # Kepler's law uses the total pair mass; the resulting axis is in AU.
            log_axis = float(
                _facade("companion_log_separation_au")(
                    self.host_mass + companion_mass, np.array([log_period])
                )[0]
            )
        separation_arcsec = (
            float(np.power(10.0, log_axis)) * math.sqrt((1.0 - point[2]) * (1.0 + point[2]))
            * self.parallax_mas / _facade("MILLIARCSECONDS_PER_ARCSECOND")
        )
        if separation_arcsec > self.outer_arcsec:
            return unsupported
        if not (
            _facade("STELLAR_RELATIONS_MASS_MIN_SOLAR")
            <= companion_mass
            <= _facade("STELLAR_RELATIONS_MASS_MAX_SOLAR")
        ):
            return unsupported
        radius_array, teff_array = _facade("stellar_relations")(np.array([companion_mass]))
        companion_radius = float(radius_array[0])
        companion_teff = float(teff_array[0])
        if self.stellar_light is None:
            flux_ratio = (companion_radius / self.host_radius) ** 2 * float(
                _facade("tess_surface_brightness_ratio")(
                    np.array([companion_teff]), np.array([self.host_teff])
                )[0]
            )
        else:
            ratio, _, _ = self.stellar_light.pair(
                (self.host_teff, self.host_mass, self.host_radius),
                (companion_teff, companion_mass, companion_radius),
                self.target_feh, self.stellar_light.target_av_mag,
            )
            flux_ratio = float(ratio)
        curve = self.contrast_curve
        if self.measured_curve and curve is None:
            curve = ContrastCurve(self.contrast_separations, self.contrast_values)
        limit = curve.at(np.array([separation_arcsec])) if curve is not None else None
        imaging = _facade("ImagingConstraint")(np.array([flux_ratio]), limit, require_fainter=True)
        supported = defer_imaging or bool(np.isfinite(imaging.apply(0.0)[0]))
        dataset_dilutions = []
        if self.datasets:
            # One position, one companion mass and one occurrence prior; only the
            # band response differs. A missing native band light leaves the draw
            # unsupported instead of reusing the TESS ratio.
            for dataset in self.datasets:
                band_ratio = self._band_pair_ratio(
                    dataset.band, companion_mass, companion_radius, companion_teff
                )
                if band_ratio is None or not math.isfinite(band_ratio):
                    supported = False
                    dataset_dilutions.append((dataset.dataset_id, None))
                    continue
                dataset_dilutions.append(
                    (dataset.dataset_id,
                     1.0 if math.isinf(band_ratio) and band_ratio > 0.0
                     else band_ratio / (1.0 + band_ratio))
                )
        return _facade("UnresolvedPopulationDraw")(
            flux_fraction=1.0 if math.isinf(flux_ratio) and flux_ratio > 0.0 else flux_ratio / (1.0 + flux_ratio),
            mass_solar=companion_mass,
            radius_solar=companion_radius,
            teff_k=companion_teff,
            supported=supported,
            imaging=imaging,
            av_mag=None if self.stellar_light is None else self.stellar_light.target_av_mag,
            log_period_days=log_period,
            dataset_dilutions=tuple(dataset_dilutions),
        )


class _BackgroundStarPopulation(NamedTuple):
    """The unresolved background-star population as a prior block.

    Plain data for the same pickling reason as
    :class:`_BoundCompanionPopulation`. The physical/metallicity column arrays
    are the retained rows selected in the declared observing band, ordered by
    that band's magnitude. ``tess_mag`` remains the legacy column only for a
    TESS selection; a native-band block withholds it (NaN) so a TESS-only
    consumer refuses instead of silently selecting on TESS. ``selection_mag``
    and ``selection_depth_mag`` carry the one shared faint-limit prior.
    ``completeness`` is the caller's statement verbatim, never a correction.
    """

    tess_mag: np.ndarray
    mass_solar: np.ndarray
    radius_solar: np.ndarray
    teff_k: np.ndarray
    target_tess_mag: Optional[float]
    measured_curve: bool
    contrast_separations: Optional[np.ndarray]
    contrast_values: Optional[np.ndarray]
    field_radius_arcsec: float
    log_occurrence_weight: float
    mh_dex: np.ndarray
    n_dimensions: int = _POPULATION_N_DIMENSIONS
    av_mag: Optional[np.ndarray] = None
    contrast_curve: Optional[ContrastCurve] = None
    datasets: tuple = ()
    band_magnitudes: Optional[Mapping[str, np.ndarray]] = None
    band_valid: Optional[Mapping[str, np.ndarray]] = None
    selection_mag: Optional[np.ndarray] = None
    selection_band: str = "TESS"
    selection_depth_mag: Optional[float] = None
    completeness: Optional[str] = None
    resolved_excluded: int = 0

    def resolve(self, unit_point: np.ndarray, *, defer_imaging=False) -> "UnresolvedPopulationDraw":
        """One background draw, as ``_draw_background_star`` resolves it."""
        point = np.asarray(unit_point, dtype=float)
        if point.shape != (_POPULATION_N_DIMENSIONS,) or self.n_dimensions != _POPULATION_N_DIMENSIONS:
            raise ValueError(f"background population requires {_POPULATION_N_DIMENSIONS} unit dimensions")
        if not np.isfinite(point).all() or np.any(point < 0.0) or np.any(point > 1.0):
            raise ValueError("background population variates must be finite and in [0, 1]")
        magnitudes = self.tess_mag if self.selection_mag is None else self.selection_mag
        depth = self.target_tess_mag if self.selection_depth_mag is None else self.selection_depth_mag
        star_count = int(magnitudes.size)
        index = min(star_count - 1, max(0, int(point[0] * star_count)))
        delta_mag = float(magnitudes[index]) - float(depth)
        with np.errstate(over="ignore"):
            flux_ratio = float(np.power(10.0, _facade("POGSON_FLUX_EXPONENT") * delta_mag))
        separation_arcsec = self.field_radius_arcsec * math.sqrt(
            max(float(point[1]), 0.0)
        )
        curve = self.contrast_curve
        if self.measured_curve and curve is None:
            curve = ContrastCurve(self.contrast_separations, self.contrast_values)
        limit = curve.at(np.array([separation_arcsec])) if curve is not None else None
        imaging = _facade("ImagingConstraint")(np.array([flux_ratio]), limit, require_fainter=False)
        supported = defer_imaging or bool(np.isfinite(imaging.apply(0.0)[0]))
        dataset_dilutions = []
        if self.datasets:
            # The same selected star and the same areal position feed every
            # dataset. Its magnitude in another native band changes the
            # dilution; an unknown cell makes the draw unsupported rather than
            # reusing the TESS magnitude.
            for dataset in self.datasets:
                values = None if self.band_magnitudes is None else self.band_magnitudes.get(dataset.band)
                valid = None if self.band_valid is None else self.band_valid.get(dataset.band)
                if (values is None or valid is None or dataset.target_magnitude is None
                        or not bool(valid[index]) or not math.isfinite(float(values[index]))):
                    supported = False
                    dataset_dilutions.append((dataset.dataset_id, None))
                    continue
                band_delta = float(values[index]) - float(dataset.target_magnitude)
                with np.errstate(over="ignore"):
                    band_ratio = float(np.power(10.0, _facade("POGSON_FLUX_EXPONENT") * band_delta))
                dataset_dilutions.append(
                    (dataset.dataset_id,
                     1.0 if math.isinf(band_ratio) and band_ratio > 0.0
                     else band_ratio / (1.0 + band_ratio))
                )
        return _facade("UnresolvedPopulationDraw")(
            flux_fraction=1.0 if math.isinf(flux_ratio) and flux_ratio > 0.0 else flux_ratio / (1.0 + flux_ratio),
            mass_solar=float(self.mass_solar[index]),
            radius_solar=float(self.radius_solar[index]),
            teff_k=float(self.teff_k[index]),
            supported=supported,
            imaging=imaging,
            mh_dex=float(self.mh_dex[index]),
            av_mag=None if self.av_mag is None else float(self.av_mag[index]),
            dataset_dilutions=tuple(dataset_dilutions),
        )


# An unresolved diluting population, in either of its two retained forms.
#
# ``resolve`` maps the block's own unit-cube variates to one draw the way the
# production ``_draw_bound_companion`` and ``_draw_background_star`` map their
# uniform variates: the same inverse CDFs, the same retention mask, the same
# occurrence rate.  ``supported`` is that retention mask -- a draw the imaging
# would have detected, one wide enough to be a resolved catalog neighbour
# instead, or (on the Moe branch) one whose period leaves the measured
# mass-ratio domain -- which the production path carries as a ``-inf`` prior
# weight and the nested likelihood carries as zero support.  Hierarchical
# stability is applied by the nested integrands once the inner binary's total
# mass exists, exactly as the production estimator applies it.
#
# ``log_occurrence_weight`` is the population's occurrence rate, a constant on
# the retained set.  The production estimator keeps it inside the expectation
# (``ln mean_i exp(lnL_i + lnprior_i)``); since it is constant there and
# ``-inf`` off the retained set, adding it to the converged ``lnZ`` is the
# identical quantity.
UnresolvedPopulation = Union[_BoundCompanionPopulation, _BackgroundStarPopulation]


def _angular_constraint(
    contrast_separations: Optional[np.ndarray],
    contrast_values: Optional[np.ndarray],
    resolution_limit_arcsec: Optional[float],
    label: str,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """Resolve imaging detectability and the unresolved-scene outer bound.

    A measured contrast curve is a detection constraint, not a population
    boundary. The unresolved population extends to the catalog resolution
    bound; if callers supply a measured curve without the bound, use the
    published Gaia close-pair resolution used by the production scene builder.
    """
    measured = contrast_separations is not None and contrast_values is not None
    if (contrast_separations is None) != (contrast_values is None):
        raise ValueError("contrast separations and values must be supplied together")
    if measured and resolution_limit_arcsec is None:
        resolution_limit_arcsec = _facade("GAIA_DR2_CLOSE_PAIR_RESOLUTION_ARCSEC")
    if resolution_limit_arcsec is None:
        raise ValueError(
            "{0} scenarios require a measured contrast curve or catalog resolution bound".format(label)
        )
    limit = _facade("_validate_scalar")(
        resolution_limit_arcsec, "resolution_limit_arcsec", positive=True
    )
    if not measured:
        return False, None, None, limit
    curve = ContrastCurve(contrast_separations, contrast_values)
    return True, curve.separations, curve.contrasts, limit


def bound_companion_population(
    M_s_Msun: float,
    R_s_Rsun: float,
    plx_mas: float,
    *,
    contrast_separations: Optional[np.ndarray] = None,
    contrast_values: Optional[np.ndarray] = None,
    resolution_limit_arcsec: Optional[float] = None,
    host_teff_k: Optional[float] = None,
    stellar_light=None,
    target_feh=None,
    datasets=(),
    band_light: Optional[Mapping[str, object]] = None,
) -> _BoundCompanionPopulation:
    """The unresolved bound-companion population of the ``P*``/``S*`` slots.

    Mirrors ``_draw_bound_companion`` variate for variate: for M dwarfs a
    moment-matched log-normal axis, mass ratio and independent circular-orbit
    projection (period derived afterwards with both masses); for higher masses a period from the Moe & Di Stefano (2017)
    Eq. 23 distribution, the mass ratio belonging to *that* period and the same independent sky projection; then
    the companion's TESS-band flux from the same radius-and-surface-brightness
    construction, and the resulting contrast decides retention.  On the Moe
    branch a period outside the measured domain, a companion mass outside the
    retained mass-radius domain, a draw the imaging would have detected, one
    wide enough to be a resolved catalog neighbour, or one brighter than the
    primary all carry no support rather than an extrapolated value; the
    nested integrands add the shared hierarchical-stability mask.
    """
    host_mass = _facade("_validate_scalar")(M_s_Msun, "M_s_Msun", positive=True)
    host_radius = _facade("_validate_scalar")(R_s_Rsun, "R_s_Rsun", positive=True)
    parallax = _facade("_validate_scalar")(plx_mas, "plx_mas", positive=True)
    measured, separations, values, outer_arcsec = _facade("_angular_constraint")(
        contrast_separations, contrast_values, resolution_limit_arcsec, "bound-companion"
    )
    if host_teff_k is None:
        if stellar_light is not None:
            raise ValueError("physical bound light requires the actual target temperature")
        _, host_teff_array = _facade("stellar_relations")(np.array([host_mass]))
        host_teff_k = float(host_teff_array[0])
    else:
        host_teff_k = _facade("_validate_scalar")(host_teff_k, "host_teff_k", positive=True)
    frequency = _facade("_validate_scalar")(
        _facade("companion_population_frequency")(host_mass), "companion population frequency", positive=True
    )
    bound_datasets = _population_datasets(datasets)
    if bound_datasets:
        if not band_light:
            raise ValueError("no native-band light for dataset-bound bound companions")
        missing = [
            dataset.band for dataset in bound_datasets
            if dataset.band not in band_light
            and not (dataset.band.strip().lower() == "tess" and stellar_light is not None)
        ]
        if missing:
            raise ValueError(
                "no native-band light for population dataset band(s): " + ", ".join(sorted(set(missing)))
            )
    return _facade("_BoundCompanionPopulation")(
        host_mass=host_mass,
        host_radius=host_radius,
        host_teff=host_teff_k,
        parallax_mas=parallax,
        measured_curve=measured,
        contrast_separations=separations,
        contrast_values=values,
        outer_arcsec=outer_arcsec,
        log_occurrence_weight=math.log(frequency),
        n_dimensions=_BOUND_POPULATION_N_DIMENSIONS,
        stellar_light=stellar_light,
        target_feh=target_feh,
        contrast_curve=ContrastCurve(separations, values) if measured else None,
        datasets=bound_datasets,
        band_light=dict(band_light) if band_light else None,
    )


def background_star_population(
    population: Mapping[str, np.ndarray],
    simulated_area_sq_deg: float,
    target_tess_mag=None,
    *,
    contrast_separations: Optional[np.ndarray] = None,
    contrast_values: Optional[np.ndarray] = None,
    resolution_limit_arcsec: Optional[float] = None,
    datasets=(),
    selection_band: Optional[str] = None,
    selection_magnitude=None,
    completeness: Optional[str] = None,
) -> _BackgroundStarPopulation:
    """The unresolved background-star population of the ``D*``/``B*`` slots.

    Mirrors ``_draw_background_star``: a star taken from the candidate's own
    retained population, placed at a separation drawn uniformly in area out to
    the field edge, and retained unless the imaging would have detected it.
    The production draw is ``rng.integers(0, N)`` -- a uniform choice over the
    selected table -- whose exact inverse CDF on the unit interval is
    ``floor(u * N)``; the table is ordered by the selection band's magnitude so
    this coordinate's step function varies monotonically in the dilution it
    controls rather than in catalog order. Reordering permutes a uniform
    measure, so the integral is unchanged, and the slice sampler gets a
    coordinate whose likelihood level sets are intervals it can bracket.

    ``selection_band`` is the one exact instrument-qualified faint-limit band
    shared by every dataset (for example ``Roman/WFI07.F146``); when omitted,
    the first declared dataset's band is used. Its native magnitudes and the
    target's magnitude from that dataset set the pool; an absent band refuses
    rather than reusing TESS. With no datasets and no band, the legacy TESS
    path keeps the loader's already-applied cut, so existing TESS evidence is
    unchanged. Dataset entries only change the per-aperture dilution share.
    ``completeness`` is stored verbatim and never turned into a factor; no
    completeness model is assumed.
    """
    area = _facade("_validate_scalar")(simulated_area_sq_deg, "simulated_area_sq_deg", positive=True)
    measured, separations, values, field_radius_arcsec = _facade("_angular_constraint")(
        contrast_separations, contrast_values, resolution_limit_arcsec, "background-star"
    )
    background_datasets = _population_datasets(datasets)
    from trex.populations.background import select_background_population

    selection = select_background_population(
        population,
        datasets=background_datasets,
        band=selection_band,
        depth_mag=selection_magnitude,
        target_tess_mag=target_tess_mag,
        completeness=completeness,
    )
    indices = selection.selected_indices
    selection_mag = np.asarray(selection.magnitudes, dtype=float)[indices]
    order = np.argsort(selection_mag, kind="stable")
    indices = indices[order]
    selection_mag = selection_mag[order]
    mass_solar = np.asarray(population["mass_solar"], dtype=float)[indices]
    radius_solar = np.asarray(population["radius_solar"], dtype=float)[indices]
    teff_k = np.asarray(population["teff_k"], dtype=float)[indices]
    if selection_mag.size == 0 or any(
        values.shape != selection_mag.shape for values in (mass_solar, radius_solar, teff_k)
    ):
        raise ValueError(
            "background population columns must be non-empty and of equal length"
        )
    if not all(np.isfinite(values).all() for values in (mass_solar, radius_solar, teff_k)):
        raise ValueError("background population columns must be finite")
    if any(np.any(values <= 0.0) for values in (mass_solar, radius_solar, teff_k)):
        raise ValueError("background population masses, radii and temperatures must be positive")
    from trex.populations.background import background_metallicity_dex

    mh_dex = background_metallicity_dex(population)[indices]
    from trex.stellar_light.inputs import background_extinction

    av_mag = background_extinction(population)[indices] if "av_mag" in population else None
    expectation = _facade("_validate_scalar")(
        _facade("background_population_expectation")(
            star_count=int(indices.size), simulated_area_sq_deg=area,
            field_radius_arcsec=field_radius_arcsec,
        ),
        "background population expectation", positive=True,
    )
    band_magnitudes = None
    band_valid = None
    if background_datasets:
        from trex.populations.background import background_dataset_magnitudes

        validated_datasets = []
        band_magnitudes, band_valid = {}, {}
        for dataset in background_datasets:
            if dataset.target_magnitude is None:
                raise ValueError(
                    "background dataset {0!r} requires the target magnitude in band {1!r}".format(
                        dataset.dataset_id, dataset.band
                    )
                )
            validated_datasets.append(dataset._replace(
                target_magnitude=_facade("_validate_scalar")(
                    dataset.target_magnitude, "background target magnitude", positive=False
                )
            ))
            magnitudes, valid = background_dataset_magnitudes(population, dataset.band)
            if magnitudes.shape != np.asarray(selection.magnitudes).shape:
                raise ValueError("background band magnitudes must align with the retained stars")
            band_magnitudes[dataset.band] = magnitudes[indices]
            band_valid[dataset.band] = valid[indices]
        background_datasets = tuple(validated_datasets)
    legacy = selection.source == "legacy-tess"
    return _facade("_BackgroundStarPopulation")(
        tess_mag=(selection_mag.copy() if legacy
                  else np.full(selection_mag.shape, np.nan, dtype=float)),
        mass_solar=mass_solar,
        radius_solar=radius_solar,
        teff_k=teff_k,
        target_tess_mag=(float(selection.depth_mag) if legacy else math.nan),
        measured_curve=measured,
        contrast_separations=separations,
        contrast_values=values,
        field_radius_arcsec=field_radius_arcsec,
        log_occurrence_weight=math.log(expectation),
        mh_dex=mh_dex,
        av_mag=None if av_mag is None else av_mag,
        contrast_curve=ContrastCurve(separations, values) if measured else None,
        datasets=background_datasets,
        band_magnitudes=band_magnitudes,
        band_valid=band_valid,
        selection_mag=selection_mag,
        selection_band=selection.band,
        selection_depth_mag=float(selection.depth_mag),
        completeness=selection.completeness,
        resolved_excluded=selection.resolved_excluded,
    )
