"""Concrete population inputs; traced device arithmetic is validated by its caller."""

from numbers import Real

import numpy as np


def real_array(value, name: str) -> np.ndarray:
    """Finite float64 values, preserving shape; refuse boolean/text/object inputs."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite real numbers, not boolean or text")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers, not boolean or text")
    return array


def real_scalar(value, name: str, *, positive: bool = False) -> float:
    """Require one finite real number, optionally strictly positive."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be a finite real scalar" + (" greater than zero" if positive else ""))
    return value


def unit_quantiles(value, name: str, *, open_endpoints: bool = False) -> np.ndarray:
    """Admit finite real [0,1] quantiles, or (0,1) for unbounded normal laws."""
    value = real_array(value, name)
    outside = ((value <= 0) | (value >= 1)) if open_endpoints else ((value < 0) | (value > 1))
    if np.any(outside):
        interval = "(0, 1)" if open_endpoints else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return value


def boolean(value, name: str) -> bool:
    """Accept Python/NumPy boolean scalars, not truthy numbers or arrays."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)
