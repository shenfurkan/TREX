"""Shared TREX evidence-engine failures: unresolved inputs and numerical likelihood errors."""

from __future__ import annotations

import copy
from numbers import Integral
from typing import Any, Mapping, Optional


class TrexEvidenceUnavailableError(ValueError):
    """Required evidence is absent, so TREX cannot resolve every live scenario.

    ``diagnostics`` carries source-specific ``input_requirements`` for a missing
    capable host, or the structured convergence record behind a numerical
    deferral (per-scenario effective sample sizes, the registered floor and
    unconverged probability mass). Input requirements are not a convergence
    measurement; callers must preserve that distinction.

    Top-level ``diagnostics`` keys are strings. The mapping is deep-copied, so
    later changes to the caller's records do not alter the evidence carried by
    the error. ``str(error)`` is only the message: a consumer that records the
    failure must read ``diagnostics`` explicitly. Pickling for spawned workers
    keeps ``diagnostics`` through the exception's ``__dict__``.
    """

    def __init__(
        self, message: str, diagnostics: Optional[Mapping[str, Any]] = None
    ) -> None:
        super().__init__(message)
        diagnostics = dict(diagnostics) if diagnostics else {}
        if not all(isinstance(key, str) for key in diagnostics):
            raise TypeError("evidence diagnostics keys must be strings")
        self.diagnostics: dict[str, Any] = copy.deepcopy(diagnostics)


class NumericalLikelihoodError(RuntimeError):
    """Supported likelihood evaluations that could not be computed, not prior zeros.

    ``count`` is the positive integer number of failed evaluations the error
    stands for; booleans, fractions and text are refused rather than coerced.
    """

    def __init__(self, message: str, count: int = 1) -> None:
        super().__init__(message)
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("numerical failure count must be an integer; booleans are not accepted")
        count = int(count)
        if count < 1:
            raise ValueError("numerical failure count must be at least one")
        self.count = count

    def __reduce__(self):
        # Rebuild through __init__ with the count, so an error raised in a
        # spawned worker is validated again when the parent unpickles it.
        return type(self), (str(self), self.count)


__all__ = ["TrexEvidenceUnavailableError", "NumericalLikelihoodError"]
