"""Compatibility identity for :mod:`trex.app.run`."""

if __name__ == "__main__":
    from trex.runtime.logging import standalone_cli

    raise SystemExit(standalone_cli("trex.app.run"))
else:
    import sys
    import warnings

    _showwarning = warnings.showwarning
    try:
        from trex.app import run as _canonical
    finally:
        warnings.showwarning = _showwarning

    sys.modules[__name__] = _canonical
