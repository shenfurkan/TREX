"""Standalone operational checks: ``python -m trex.doctor --help``."""

if __name__ == "__main__":
    from trex.runtime.logging import standalone_cli

    raise SystemExit(standalone_cli("trex.app.doctor"))
else:
    import warnings

    _showwarning = warnings.showwarning
    try:
        from trex.app.doctor import group_failures, inspect_inputs, main as main, probe_mast
    finally:
        warnings.showwarning = _showwarning

    __all__ = ["group_failures", "inspect_inputs", "probe_mast"]
