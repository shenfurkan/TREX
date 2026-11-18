"""Compatibility identity for :mod:`trex.app.example`."""

import sys

from trex.app import example as _canonical

if __name__ == "__main__":
    from trex.runtime.logging import standalone_cli

    raise SystemExit(standalone_cli("trex.app.example"))
else:
    sys.modules[__name__] = _canonical
