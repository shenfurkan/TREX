"""Shape-preserving adapters for supported MC binary draws."""

import numpy as np


def binary_light(light, host_t, host_m, host_r, secondary_t, secondary_m, secondary_r,
                 metallicity, av_mag, selected):
    """Select a 1-D boolean structural-support mask; no atmosphere/prior cuts.

    All coordinates broadcast to that mask. Both bodies share their source's
    composition/extinction, not that of an unrelated target or background row.
    """
    selected = np.asarray(selected)
    if selected.ndim != 1 or selected.dtype.kind != "b":
        raise ValueError("binary light selection must be a 1-D boolean mask")
    shape = selected.shape

    def take(value):
        if value is None:
            raise ValueError("physical binary light requires both stellar coordinates and source composition/extinction")
        return np.broadcast_to(value, shape)[selected]

    # Coefficient adapter only; the angular forward consumer prepares original rays per draw.
    return light.pair(
        tuple(take(v) for v in (host_t, host_m, host_r)),
        tuple(take(v) for v in (secondary_t, secondary_m, secondary_r)),
        take(metallicity), take(av_mag),
    )
