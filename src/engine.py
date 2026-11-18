"""Compatibility identity for :mod:`trex.app.engine`."""

import sys

from trex.app import engine as _canonical

sys.modules[__name__] = _canonical
