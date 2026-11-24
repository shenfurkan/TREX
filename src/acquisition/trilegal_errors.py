"""Typed failure of a background-population provider."""

from typing import Optional

from .transport import AcquisitionError, AcquisitionFailure

TRILEGAL_ENDPOINT = "https://stev.oapd.inaf.it/cgi-bin/trilegal_1.6"


class TrilegalUnavailableError(AcquisitionError):
    """A background population could not be produced and retained.

    Raised by TRILEGAL itself and by the services feeding or replacing it (the
    IRSA dust map, the LSST simulation fallback). With a transport ``failure``
    the manifest is that failure's record. Without one, ``endpoint`` (default:
    TRILEGAL), ``category`` and ``retryable`` describe the refusal; a provider
    answer that may succeed later (HTTP 5xx/429, busy, timeout) should pass
    ``retryable=True``.
    """

    def __init__(self, message: str, failure: Optional[AcquisitionFailure] = None, *,
                 endpoint: str = TRILEGAL_ENDPOINT, category: str = "provider",
                 retryable: Optional[bool] = None) -> None:
        if retryable is None:
            retryable = bool(failure is not None and failure.retryable)
        super().__init__(message, retryable=retryable)
        self.failure = failure
        self.endpoint = endpoint
        self.category = category

    def to_manifest(self) -> dict:
        """The failure record, in the same fields as ``AcquisitionFailure.to_manifest``."""
        if self.failure is not None:
            return self.failure.to_manifest()
        message = str(self)
        return {"status": "unavailable", "category": self.category, "event": "network." + self.category,
                "endpoint": self.endpoint, "exception_class": type(self).__name__, "retryable": self.retryable,
                "attempts": None, "message": message, "diagnostic_chain": [message]}
