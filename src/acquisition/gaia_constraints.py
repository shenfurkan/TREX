"""Gaia detection inputs that cannot turn catalog absence into a null likelihood."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class GaiaDetectionConstraint:
    """One declared host-scope detection; probabilities and mass ratios are dimensionless.

    ``None`` is unavailable, not a non-detection. ``period_match`` must already
    account for the orbital/eclipsing period ambiguity; this record does not
    infer detectability from RUWE or convert a catalog absence to a limit.
    """

    target_host_scope: str
    detected: bool | None
    p_detect: float | None = None
    period_match: bool | None = None
    mass_ratio_limit: float | None = None

    def __post_init__(self):
        if not isinstance(self.target_host_scope, str) or not self.target_host_scope.strip():
            raise ValueError("Gaia constraints need an explicit source scope")
        for name in ("detected", "period_match"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"Gaia {name} must be boolean or unavailable")
        for name in ("p_detect", "mass_ratio_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Gaia {name} must be in [0,1] or unavailable")


def require_neutral_release_constraints(constraints, scene_source_ids):
    """Admit only evidence-neutral rows until a positive-detection model is approved."""
    if not isinstance(constraints, (list, tuple)):
        raise ValueError("Gaia release constraints must be an ordered sequence")
    for row in constraints:
        if not isinstance(row, GaiaDetectionConstraint) or row.target_host_scope not in scene_source_ids:
            raise ValueError("Gaia detection scope must name a retained scene source")
        if row.detected is True and row.period_match is True:
            raise ValueError("no approved Gaia release detection likelihood is available")
