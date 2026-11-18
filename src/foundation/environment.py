"""Environment-variable reads that accept a retired spelling once, with a warning.

TREX and its embedding supervisors used the same configuration variables under
an earlier project name. A silently ignored rename would leave a supervisor's
explicit worker count, memory floor or recovery directory unapplied, so the old
name keeps working for one release and every use is reported.
"""

from __future__ import annotations

import os
import warnings

_warned: set[str] = set()


def read_environment(name: str, legacy: str | None = None) -> str | None:
    """Return ``name``'s value, falling back to its retired ``legacy`` spelling.

    The return value is the raw string ``os.environ`` holds, or ``None`` when
    neither name is set; callers keep their own validation and defaults.
    ``name`` always wins when both are set. Each ``legacy`` name warns once per
    process, so a 24-scenario run reports the rename once rather than per use.
    """
    value = os.environ.get(name)
    if value is not None:
        return value
    if legacy is None or legacy not in os.environ:
        return None
    if legacy not in _warned:
        _warned.add(legacy)
        warnings.warn(
            "{0} is retired and will stop being read; set {1} instead".format(legacy, name),
            FutureWarning,
            stacklevel=2,
        )
    return os.environ[legacy]
