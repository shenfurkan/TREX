"""The public TREX modules and ``-m`` bridges resolve to the app owners."""

import importlib
import subprocess
import sys

import pytest

from trex.app import run


def test_public_modules_share_the_app_owners():
    for name in ("scene", "engine", "target", "run", "example"):
        assert importlib.import_module("trex." + name) is importlib.import_module("trex.app." + name)


def test_public_cli_bridges_keep_usage_and_pause_exit_code():
    help_result = subprocess.run([sys.executable, "-m", "trex.run", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "--draws" in help_result.stdout
    example = subprocess.run([sys.executable, "-m", "trex.example"], capture_output=True, text=True)
    assert example.returncode != 0 and "usage: python -m trex.example" in example.stderr
    paused = subprocess.run([sys.executable, "-c", (
        "import runpy; import trex.app.run as run; "
        "run.main = lambda arguments=None: 75; runpy.run_module('trex.run', run_name='__main__')"
    )], capture_output=True, text=True)
    assert paused.returncode == 75, paused.stderr


@pytest.mark.parametrize("has_previous_result", [False, True])
def test_output_directory_is_never_reused(tmp_path, has_previous_result):
    output = tmp_path / "output"
    output.mkdir()
    if has_previous_result:
        (output / "summary.json").write_text("previous result", encoding="utf-8")
    assert run.main([str(tmp_path / "input"), str(output)]) == 2
    if has_previous_result:
        assert (output / "summary.json").read_text(encoding="utf-8") == "previous result"
    else:
        assert not list(output.iterdir())
