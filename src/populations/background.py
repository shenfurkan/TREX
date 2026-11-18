"""Retained TRILEGAL background-star population for the TREX D*/B* scenarios.

The unresolved background-star scenarios of Giacalone et al. (2021), AJ 161, 24,
DOI ``10.3847/1538-3881/abc6af``, are evaluated
against a simulated stellar population for the candidate's own field, generated
with TRILEGAL (Girardi et al. 2005, A&A 436, 895, DOI
``10.1051/0004-6361:20042352``).

Units and fail-closed contract
------------------------------
``tess_mag`` is TESS-band apparent magnitude, ``mass_solar`` is ``M_sun``,
``logg_cgs`` is ``log10(cm/s^2)``, ``teff_k`` is K, and ``radius_solar`` is
``R_sun`` derived from the simulated mass and surface gravity rather than from a
main-sequence mass-radius relation, so an evolved simulated star stays evolved.
Only stars at least as faint as the target are retained: a brighter unresolved
star would dominate the aperture and is not the modelled configuration.  A cache
without the required TRILEGAL columns, or with no star fainter than the target,
raises rather than yielding an empty or substituted population.

Requested extra bands (``extra_bands``) are retained with their native TRILEGAL
column names and magnitude unit on exactly the same rows as the TESS selection.
H and Ks stay distinct arrays: no band is dropped, substituted by another band,
or repaired from a colour because the cache lacks it.  An absent requested
column is reported in ``missing_bands``; a present column with an unusable cell
keeps the row, stores NaN for that band and records the row in ``band_valid``.
The retained population therefore states *which* per-band information exists,
which is what a multi-band scene needs; it never fills a missing band with the
TESS magnitude or a solar estimate.  ``target_tess_mag=None`` retains every
valid simulated row so a non-TESS consumer can apply its own declaring band's
faint limit instead of inheriting a TESS cut.  The legacy retention rule
(``TESS >= target Tmag``) is TESS-specific: it selects stars that are not
brighter than the target in TESS, and does not by itself establish a
completeness limit in J/H/Ks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np

from trex.foundation.constants import G, Msun, Rsun
from trex.populations._validation import real_array, real_scalar


# A legacy 2MASS-only cache cannot supply the required apparent TESS column.
TRILEGAL_TESS_COLUMN = "TESS"
TRILEGAL_MASS_COLUMN = "Mact"
TRILEGAL_LOGG_COLUMN = "logg"
TRILEGAL_LOGTE_COLUMN = "logTe"
TRILEGAL_METALLICITY_COLUMN = "[M/H]"
TRILEGAL_FOOTER_MARKER = "#TRILEGAL"
TRILEGAL_EXTINCTION_COLUMN = "Av"
TRILEGAL_DISTANCE_MODULUS_COLUMN = "m-M0"
TRILEGAL_MAGNITUDE_UNIT = "mag"


def _parse_header(line: str) -> Dict[str, int]:
    """Parse unique names; allow a legacy unnamed leading pandas index only."""
    names = [name.strip() for name in line.split(",")]
    if any(not name for name in names[1:]) or len(set(names)) != len(names):
        raise ValueError("TRILEGAL population cache has empty or duplicate column names")
    return {name: index for index, name in enumerate(names) if name}


def load_trilegal_population(
    path: Path,
    target_tess_mag,
    *,
    extra_bands=None,
) -> Dict[str, np.ndarray]:
    """Read a retained TRILEGAL cache into per-star background arrays.

    Args:
        path: Candidate-owned TRILEGAL population CSV.
        target_tess_mag: Target TESS magnitude; brighter simulated stars are
            excluded because they are not unresolved background contaminants.
            ``None`` keeps every valid row for a caller that selects in another
            native band; the TESS magnitude is then a convenience column, not
            the population's selection depth. A native-only cache (for example
            the TRILEGAL ``Roman2024`` system, whose columns are WFI filters
            and no TESS magnitude) is accepted only with ``None`` and returns a
            NaN ``tess_mag`` row axis.
        extra_bands: Optional requested magnitude bands. Either an iterable of
            TRILEGAL column names (the band identity is the column name) or a
            mapping ``{band_id: column_name}`` for a provider whose band label
            differs from the column, e.g. ``{"H": "H", "Ks": "Ks"}``. Requested
            bands are retained on the same rows as ``tess_mag``.

    Returns:
        Dict of equal-length arrays ``tess_mag``, ``mass_solar``, ``logg_cgs``,
        ``teff_k``, ``radius_solar``, ``mh_dex``. The latter is TRILEGAL
        [M/H]=log10(Z/Z_sun), not mass-fraction Z or measured iron abundance.
        Optional ``av_mag`` and ``distance_modulus_mag`` retain Av and m-M0
        in magnitudes when present; absent values are not replaced.
        With ``extra_bands``, also ``band_magnitudes`` (``{band_id: mag array}``;
        NaN marks a cell the cache left unusable), ``band_valid``
        (``{band_id: bool array}``; False on those cells), ``bands``
        (``{band_id: {"column", "unit", "status"}}``), and ``missing_bands``
        (requested ids with no such column). A band whose column exists for
        only some rows is ``partially-missing``, never silently complete.
        Raw cells and mass/temperature must be valid even for brighter excluded
        stars; derived radii are checked for the retained population.
        Read/schema/physical failures raise ValueError with path/row context.
    """
    target_tess_mag = (
        None if target_tess_mag is None
        else real_scalar(target_tess_mag, "target TESS magnitude")
    )
    requested = _requested_bands(extra_bands)
    path = Path(path)
    required = (
        TRILEGAL_MASS_COLUMN,
        TRILEGAL_LOGG_COLUMN,
        TRILEGAL_LOGTE_COLUMN,
        TRILEGAL_METALLICITY_COLUMN,
    )
    tess, mass, logg, logte, mh = [], [], [], [], []
    band_values = {band: [] for band in requested}
    band_seen = {band: False for band in requested}
    # Apparent TESS already includes distance/extinction (Girardi 2005:441-442).
    # Preserve available physical context, never substitute zero for an absent A_V.
    try:
        with path.open(encoding="utf-8-sig") as stream:
            lines = ((number, line.strip()) for number, line in enumerate(stream, 1) if line.strip())
            _, header = next(lines, (0, ""))
            if not header:
                raise ValueError("TRILEGAL population cache has no simulated stars")
            columns = _parse_header(header)
            missing = [name for name in required if name not in columns]
            if missing:
                raise ValueError(f"TRILEGAL population cache lacks required columns: {', '.join(missing)}")
            # A Roman2024 TRILEGAL cache has no TESS column at all. It is usable
            # only for an explicitly native selection; a TESS cut is refused.
            has_tess = TRILEGAL_TESS_COLUMN in columns
            if not has_tess and target_tess_mag is not None:
                raise ValueError(
                    "TRILEGAL population cache has no TESS column; a TESS retention cut cannot be applied"
                )
            optional = {key: [] for key in (TRILEGAL_EXTINCTION_COLUMN, TRILEGAL_DISTANCE_MODULUS_COLUMN) if key in columns}
            extra_columns = {band: name for band, name in requested.items() if name in columns}
            read_columns = (*required, *optional, *extra_columns.values())
            if has_tess:
                read_columns = (*read_columns, TRILEGAL_TESS_COLUMN)
            last_column = max(columns[name] for name in read_columns)
            outputs = (*zip(required, (mass, logg, logte, mh)), *optional.items())
            if has_tess:
                outputs = ((TRILEGAL_TESS_COLUMN, tess), *outputs)
            terminated = False
            for row, line in lines:
                fields = line.split(",")
                footer = fields[1:] if header.startswith(",") and fields[0].strip().isdigit() else fields
                tokens = [field.strip() for field in footer]
                # Accept only complete known footer forms, not a marker inside data.
                normal_footer = (line == f"{TRILEGAL_FOOTER_MARKER} normally terminated"
                                 or tokens == [TRILEGAL_FOOTER_MARKER]
                                 or (tokens[:3] == [TRILEGAL_FOOTER_MARKER, "normally", "terminated"]
                                     and all(not token for token in tokens[3:])))
                if terminated:
                    raise ValueError(f"TRILEGAL population cache has data after its footer at {path}:{row}")
                if normal_footer:
                    terminated = True
                    continue
                if len(fields) <= last_column:
                    raise ValueError(f"TRILEGAL population cache has a truncated row at {path}:{row}")
                if extra_columns:
                    for band, column in extra_columns.items():
                        band_seen[band] = True
                        cell = fields[columns[column]].strip()
                        try:
                            value = float(cell) if cell else math.nan
                        except ValueError as exc:
                            raise ValueError(
                                f"TRILEGAL population has a non-numeric {column} at {path}:{row}"
                            ) from exc
                        band_values[band].append(value if math.isfinite(value) else math.nan)
                for name, output in outputs:
                    try:
                        value = float(fields[columns[name]])
                    except ValueError as exc:
                        raise ValueError(f"TRILEGAL population has a non-numeric {name} at {path}:{row}") from exc
                    if not math.isfinite(value) or (name == TRILEGAL_EXTINCTION_COLUMN and value < 0):
                        raise ValueError(f"TRILEGAL population has invalid {name} at {path}:{row}")
                    output.append(value)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"TRILEGAL population cache is unreadable: {path}") from exc

    mass_solar = np.asarray(mass, dtype=float)
    # A native-only cache withholds TESS rather than inventing one; TESS-only
    # consumers then refuse the finite-magnitude requirement instead of
    # selecting on a substituted value.
    tess_mag = (np.asarray(tess, dtype=float) if has_tess
                else np.full(mass_solar.shape, np.nan))
    logg_cgs = np.asarray(logg, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        teff_k = np.power(10.0, np.asarray(logte, dtype=float))
    if mass_solar.size == 0:
        raise ValueError("TRILEGAL population cache has no simulated stars")
    if np.any(mass_solar <= 0.0) or np.any(teff_k <= 0.0) or not np.isfinite(teff_k).all():
        raise ValueError("TRILEGAL population cache has a non-physical mass or temperature")

    retained = (
        np.ones(tess_mag.shape, dtype=bool) if target_tess_mag is None
        else tess_mag >= float(target_tess_mag)
    )
    if not np.any(retained):
        raise ValueError(
            "TRILEGAL population has no simulated star fainter than the target"
        )

    mass_solar = mass_solar[retained]
    logg_cgs = logg_cgs[retained]
    # R = sqrt(G M / g): the simulated surface gravity, not a main-sequence
    # relation, sets the radius, so an evolved simulated star keeps its size.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        radius_solar = np.power(10.0, (
            math.log10(G * Msun) + np.log10(mass_solar) - logg_cgs
        ) / 2.0 - math.log10(Rsun))
    if not np.isfinite(radius_solar).all() or np.any(radius_solar <= 0):
        raise ValueError("TRILEGAL population has an unrepresentable stellar radius")
    result = {
        "tess_mag": tess_mag[retained],
        "mass_solar": mass_solar,
        "logg_cgs": logg_cgs,
        "teff_k": teff_k[retained],
        "radius_solar": radius_solar,
        # Girardi 2005, trilegal.tex:eq_mh: [M/H]=log10(Z/Z_sun) [dex], not Z or measured [Fe/H].
        "mh_dex": np.asarray(mh, dtype=float)[retained],
    }
    for name, key in ((TRILEGAL_EXTINCTION_COLUMN, "av_mag"), (TRILEGAL_DISTANCE_MODULUS_COLUMN, "distance_modulus_mag")):
        if name in optional:
            result[key] = np.asarray(optional[name], dtype=float)[retained]
    if requested:
        magnitudes, valid, bands = {}, {}, {}
        missing_bands = []
        for band, column in requested.items():
            if not band_seen[band]:
                missing_bands.append(band)
                bands[band] = {"column": column, "unit": TRILEGAL_MAGNITUDE_UNIT, "status": "missing-column"}
                continue
            values = np.asarray(band_values[band], dtype=float)[retained]
            magnitudes[band] = values
            valid[band] = np.isfinite(values)
            # A column present in the header but unusable on every retained row
            # is still retained as missing magnitudes, never replaced by TESS.
            status = "retained" if bool(valid[band].all()) else "partially-missing"
            bands[band] = {"column": column, "unit": TRILEGAL_MAGNITUDE_UNIT, "status": status}
        result["band_magnitudes"] = magnitudes
        result["band_valid"] = valid
        result["bands"] = bands
        result["missing_bands"] = tuple(missing_bands)
    return result


def _requested_bands(extra_bands) -> Dict[str, str]:
    """Normalize ``extra_bands`` to ``{band_id: TRILEGAL column}`` without guessing aliases."""
    if extra_bands is None:
        return {}
    if isinstance(extra_bands, Mapping):
        requested = {}
        for band, column in extra_bands.items():
            if not isinstance(band, str) or not band.strip() or not isinstance(column, str) or not column.strip():
                raise ValueError("extra band identities and columns must be nonempty text")
            requested[band] = column
        return requested
    if isinstance(extra_bands, (str, bytes)):
        raise ValueError("extra_bands must be an iterable of column names, not one string")
    requested = {}
    for column in extra_bands:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("extra band columns must be nonempty text")
        requested[column] = column
    return requested



def background_metallicity_dex(population) -> np.ndarray:
    """Require finite TRILEGAL [M/H]; never substitute target [Fe/H].

    A legacy population's ``tess_mag`` row axis, when present, must still
    align. A non-TESS population need not carry a TESS magnitude; the caller
    aligns this array to the rows selected in its own native band.
    """
    if not isinstance(population, Mapping) or "mh_dex" not in population:
        raise ValueError("background population requires retained TRILEGAL [M/H] (mh_dex)")
    metallicity = real_array(population["mh_dex"], "background mh_dex")
    if metallicity.ndim != 1 or not metallicity.size:
        raise ValueError("background mh_dex must be a finite nonempty vector")
    axis = population.get("tess_mag")
    if axis is not None:
        axis = np.asarray(axis)
        if axis.dtype.kind not in "iuf" or axis.ndim != 1 or axis.shape != metallicity.shape:
            raise ValueError("background mh_dex must be finite and match the retained stars")
    return metallicity


def background_band_magnitudes(population, band: str):
    """Return ``(magnitudes, valid_mask)`` for one requested native band.

    ``valid_mask`` is False where the retained cache left that star's band cell
    unusable. The caller must keep those rows and treat the band light as
    unknown; this helper never substitutes the TESS magnitude or another band.
    A band that was not requested/retained raises: silently falling back to
    TESS would turn "unknown H light" into a fabricated measurement.
    """
    if not isinstance(population, Mapping) or "band_magnitudes" not in population:
        raise ValueError(
            "background population was loaded without requested band magnitudes; "
            "reload it with extra_bands before using a non-TESS dataset"
        )
    if not isinstance(band, str) or not band.strip():
        raise ValueError("background band identity must be nonempty text")
    magnitudes = population["band_magnitudes"].get(band)
    if magnitudes is None:
        raise ValueError(f"background population does not retain band {band!r}")
    values = np.asarray(magnitudes, dtype=float)
    valid = np.asarray(population["band_valid"][band], dtype=bool)
    if values.ndim != 1 or not values.size or valid.shape != values.shape or not valid.any():
        raise ValueError(f"background band {band!r} has no usable magnitude for the retained stars")
    return values, valid


def background_dataset_magnitudes(population, band: str):
    """Magnitudes of one dataset band, accepting the legacy TESS column.

    A dataset whose band is TESS uses the retained ``tess_mag`` column when the
    population was loaded without explicit extra bands. Every other band must
    be a native retained column; there is no fallback to TESS or another band.
    """
    if isinstance(band, str) and band.strip().lower() == "tess" and "tess_mag" in population:
        values = np.asarray(population["tess_mag"], dtype=float)
        if values.ndim != 1 or not values.size:
            raise ValueError("legacy TESS band requires a nonempty tess_mag vector")
        return values, np.isfinite(values)
    return background_band_magnitudes(population, band)


class BackgroundSelection(NamedTuple):
    """One declared background faint limit shared by every observation dataset.

    ``band`` is the exact observing band id (``"TESS"`` for the legacy path);
    ``depth_mag`` is the target's magnitude in that band. ``magnitudes`` are
    the population's own native values, NaN where the cache left a cell
    unusable; ``valid`` marks usable cells; ``selected`` is the faint-limit
    mask after excluding the producer's resolved duplicates. ``source`` records
    whether the legacy loader cut or a native-band limit produced the pool.
    ``completeness`` is the caller's statement about the retained population,
    stored verbatim: no completeness correction is invented, and ``None``
    means "not stated", never a completeness of one.
    """

    band: str
    depth_mag: float
    magnitudes: np.ndarray
    valid: np.ndarray
    selected: np.ndarray
    source: str
    completeness: Optional[str] = None
    resolved_excluded: int = 0

    @property
    def selected_indices(self) -> np.ndarray:
        return np.flatnonzero(self.selected)

    @property
    def n_selected(self) -> int:
        return int(np.count_nonzero(self.selected))

    @property
    def n_invalid(self) -> int:
        return int(np.count_nonzero(~self.valid))


# The loader's retention cut is TESS-specific; the canonical instrument-qualified
# id (D-76) must take that same path instead of looking for a native column named
# "TESS/TESS.T" that a TRILEGAL cache never has.
LEGACY_TESS_BAND_IDS = frozenset({"tess", "tess/tess.t"})


def is_legacy_tess_band(band) -> bool:
    """Whether ``band`` names the legacy TESS column/cut, including D-76's id."""
    return isinstance(band, str) and band.strip().lower() in LEGACY_TESS_BAND_IDS


def _selection_value(dataset, name):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    return dataset.get(name)


def _unresolved_mask(population, shape):
    """Producer duplicate flag: resolved matches are kept but not re-counted."""
    if "unresolved_mask" in population:
        mask = np.asarray(population["unresolved_mask"], dtype=bool)
        if mask.shape != shape:
            raise ValueError("background unresolved_mask must align with the selected band")
        return mask, int(np.count_nonzero(~mask))
    if "resolved_match" in population:
        resolved = np.asarray(population["resolved_match"], dtype=bool)
        if resolved.shape != shape:
            raise ValueError("background resolved_match must align with the selected band")
        return ~resolved, int(np.count_nonzero(resolved))
    return np.ones(shape, dtype=bool), 0


class BackgroundSelectionParameters(NamedTuple):
    """One declared selection, resolved before any star is drawn.

    ``depth_mag`` is None when no constant in any band was declared; a gate can
    then leave the scenario unavailable instead of drawing from a guessed cut.
    ``legacy_tess`` marks the loader-applied TESS path.
    """

    band: Optional[str]
    depth_mag: Optional[float]
    completeness: Optional[str]
    legacy_tess: bool


def _declared_metadata(population, key):
    value = population.get(key) if isinstance(population, Mapping) else None
    if value is None:
        return None
    if key == "selection_depth_mag":
        return real_scalar(value, "declared background selection depth")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"background population {key} must be nonempty text or None")
    return value


def _resolve_background_selection(
    population,
    *,
    datasets=(),
    band: Optional[str] = None,
    depth_mag=None,
    target_tess_mag=None,
    completeness: Optional[str] = None,
) -> BackgroundSelectionParameters:
    """Resolve the population's declared band/depth/completeness, order-independently.

    A scene attaches ``selection_band``/``selection_depth_mag``/
    ``selection_completeness`` to the population it loaded. That declaration
    wins over any dataset ordering; without one, only a single distinct dataset
    band is unambiguous. Several undeclared bands refuse rather than silently
    following the first dataset. Every non-None depth statement must agree.
    """
    if not isinstance(population, Mapping):
        raise ValueError("background population must be a mapping")
    dataset_list = tuple(datasets or ())
    dataset_bands = [_selection_value(dataset, "band") for dataset in dataset_list]
    if any(not isinstance(value, str) or not value.strip() for value in dataset_bands):
        raise ValueError("background selection datasets require their native band ids")
    declared_band = _declared_metadata(population, "selection_band")
    declared_depth = _declared_metadata(population, "selection_depth_mag")
    declared_completeness = _declared_metadata(population, "selection_completeness")
    if band is not None:
        if not isinstance(band, str) or not band.strip():
            raise ValueError("background selection band must be nonempty text")
        if declared_band is not None and band != declared_band:
            raise ValueError(
                "background selection band {0!r} differs from the population's declared {1!r}".format(
                    band, declared_band)
            )
    band = band if band is not None else declared_band
    if band is None and dataset_bands:
        unique = list(dict.fromkeys(dataset_bands))
        if len(unique) != 1:
            raise ValueError(
                "background selection band is ambiguous across observation datasets ({0}); "
                "declare the scene's background selection band".format(
                    ", ".join(repr(value) for value in unique))
            )
        band = unique[0]
    if band is not None and dataset_bands and band not in dataset_bands:
        raise ValueError(
            f"background selection band {band!r} is not declared by any observation dataset"
        )
    legacy = band is None or is_legacy_tess_band(band)
    targets = [
        _selection_value(dataset, "target_magnitude")
        for dataset, dataset_band in zip(dataset_list, dataset_bands, strict=True)
        if dataset_band == band
    ]
    if any(target is None for target in targets):
        raise ValueError(f"background selection band {band!r} requires the dataset target magnitude")
    dataset_depth = None if not targets else float(targets[0])
    if any(float(target) != dataset_depth for target in targets[1:]):
        raise ValueError("datasets sharing a background selection band must agree on its target magnitude")
    candidates = [value for value in (
        dataset_depth, declared_depth,
        depth_mag, target_tess_mag if legacy else None,
    ) if value is not None]
    if not candidates:
        return BackgroundSelectionParameters(band, None, None, legacy)
    depth = real_scalar(candidates[0], "background selection depth")
    if any(float(value) != depth for value in candidates[1:]):
        raise ValueError("background selection depth declarations disagree")
    if completeness is not None and (not isinstance(completeness, str) or not completeness.strip()):
        raise ValueError("background completeness statement must be nonempty text or None")
    if (completeness is not None and declared_completeness is not None
            and completeness != declared_completeness):
        raise ValueError("background completeness declarations disagree")
    return BackgroundSelectionParameters(
        band, depth, completeness if completeness is not None else declared_completeness, legacy)


def declared_background_selection(
    population,
    *,
    datasets=(),
    target_tess_mag=None,
) -> BackgroundSelectionParameters:
    """Gate-level view of the same declaration used by the draw.

    Called by production scenario builders before any RNG work, so a native
    scene states its band/depth instead of inheriting the first dataset's band
    or a finite TESS magnitude.
    """
    return _resolve_background_selection(
        population, datasets=datasets, target_tess_mag=target_tess_mag)


def select_background_population(
    population,
    *,
    datasets=(),
    band: Optional[str] = None,
    depth_mag=None,
    target_tess_mag=None,
    completeness: Optional[str] = None,
) -> BackgroundSelection:
    """Select unresolved stars at one observing dataset's band and depth.

    ``band`` is the exact instrument-qualified observing band id (for example
    ``Roman/WFI07.F146``), never a TESS magnitude standing in for it. Its
    magnitudes come from the population's own native ``band_magnitudes``;
    an absent band raises instead of falling back to TESS. The population's
    scene-attached ``selection_band`` is the explicit declaration; without
    one, a single distinct dataset band is used, and several undeclared bands
    refuse as ambiguous. With no datasets the legacy TESS path uses
    ``tess_mag``/``target_tess_mag``. The canonical ``TESS/TESS.T`` id takes
    that same legacy path.

    ``depth_mag`` is the target magnitude in the selection band. It must agree
    with the matching dataset's target magnitude and the population's declared
    depth when more than one is stated. The native path keeps only
    ``magnitude >= depth_mag``; the legacy TESS path re-applies its own
    loader cut, which is a no-op for a population the loader already trimmed
    and makes an untrimmed cache still select at the declared depth.
    ``resolved_match``/``unresolved_mask`` from the producer is honoured
    without deleting rows. A selection that retains no star raises.
    """
    parameters = _resolve_background_selection(
        population, datasets=datasets, band=band, depth_mag=depth_mag,
        target_tess_mag=target_tess_mag, completeness=completeness)
    depth = parameters.depth_mag
    if depth is None:
        raise ValueError(
            "background selection requires the target magnitude in its observing band"
        )
    if parameters.legacy_tess:
        values = np.asarray(population.get("tess_mag"), dtype=float)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("legacy TESS selection requires finite retained tess_mag")
        magnitudes, valid = values, np.ones(values.shape, dtype=bool)
        band_name, source = "TESS", "legacy-tess"
    else:
        magnitudes, valid = background_band_magnitudes(population, parameters.band)
        band_name, source = parameters.band, "native-band"
    unresolved, resolved_excluded = _unresolved_mask(population, magnitudes.shape)
    selected = valid & unresolved & (magnitudes >= depth)
    if not np.any(selected):
        raise ValueError(
            "background selection in band {0!r} retains no unresolved star at depth {1:g} mag "
            "({2} usable, {3} unknown, {4} resolved duplicate)".format(
                band_name, depth, int(np.count_nonzero(valid)),
                int(np.count_nonzero(~valid)), resolved_excluded,
            )
        )
    return BackgroundSelection(
        band=band_name, depth_mag=depth, magnitudes=magnitudes, valid=valid,
        selected=selected, source=source, completeness=parameters.completeness,
        resolved_excluded=resolved_excluded,
    )


__all__ = [
    "BackgroundSelection",
    "BackgroundSelectionParameters",
    "background_band_magnitudes",
    "background_dataset_magnitudes",
    "background_metallicity_dex",
    "declared_background_selection",
    "is_legacy_tess_band",
    "load_trilegal_population",
    "select_background_population",
]
