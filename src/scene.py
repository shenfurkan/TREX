"""Compatibility identity for :mod:`trex.app.scene`."""

import sys

from trex.app import scene as _canonical

sys.modules[__name__] = _canonical
