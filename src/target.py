"""Compatibility identity for :mod:`trex.app.target`."""

import sys

from trex.app import target as _canonical

sys.modules[__name__] = _canonical
