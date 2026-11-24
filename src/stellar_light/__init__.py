"""Source-supported atmosphere photon light; historical reference APIs stay separate."""

from .components import BandStellarLight, ComponentLight, StellarLight
from .extinction import ExtinctionCurve, ExtinctionUnsupportedError
from .host_limb_darkening import LimbDarkeningUnavailableError, resolve_host_ldc
from .interpolation import PhotonAtmosphereGrid, UnsupportedAtmosphereError
from .passband import PhotonPassband, integrate_photon_intensity
from .release import acquire_phoenix_intensity, read_phoenix_intensity
from .response import (
    SpectralResponse,
    ZeroPoint,
    require_common_zero_point_system,
    require_response_zero_point_edition,
)
from .preparation import AtmospherePreparation, prepare_atmospheres
from .support import population_atmosphere_demand

__all__ = [
    "AtmospherePreparation", "BandStellarLight", "ComponentLight",
    "ExtinctionCurve", "ExtinctionUnsupportedError", "LimbDarkeningUnavailableError",
    "PhotonAtmosphereGrid", "PhotonPassband", "SpectralResponse", "StellarLight",
    "UnsupportedAtmosphereError", "ZeroPoint", "acquire_phoenix_intensity",
    "integrate_photon_intensity", "read_phoenix_intensity",
    "population_atmosphere_demand", "prepare_atmospheres", "resolve_host_ldc",
    "require_common_zero_point_system", "require_response_zero_point_edition",
]
