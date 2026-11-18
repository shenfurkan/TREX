"""Retained-source physical input checks; no invented extinction/abundance."""

import numpy as np


def background_extinction(population):
    """Mapping input: align retained TRILEGAL A_V [mag] to its tess_mag row axis."""
    if "av_mag" not in population:
        raise ValueError("physical background atmosphere requires retained TRILEGAL A_V (av_mag)")
    if "tess_mag" not in population:
        raise ValueError("physical background atmosphere requires retained tess_mag rows")
    magnitudes = np.asarray(population["tess_mag"])
    values = np.asarray(population["av_mag"])
    if (magnitudes.dtype.kind not in "iuf" or magnitudes.ndim != 1
            or values.dtype.kind not in "iuf" or values.shape != magnitudes.shape
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("background av_mag must be finite, nonnegative and row aligned")
    return values.astype(float, copy=False)


def require_same_population_model(population, light):
    """Prepared population object: require the same model instance, including None.

    Equal labels need not identify equal compiled cells. This is distinct from
    background_extinction's raw catalog mapping; disabling an attached physical
    model with light=None is also a model change and must be refused.
    """
    if population is not None and hasattr(population, "stellar_light") and population.stellar_light is not light:
        raise ValueError("bound population and evidence integrand require the identical stellar_light input")
