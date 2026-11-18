"""Atomic evidence writes publish a whole file or nothing."""

import errno
import json
import os
import stat

import pytest

from trex.acquisition import storage


def test_write_json_accepts_a_string_path_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "nested" / "record.json"
    storage.write_json(str(target), {"b": 1, "a": [1.5, None]})
    assert json.loads(target.read_text()) == {"a": [1.5, None], "b": 1}
    assert [path.name for path in target.parent.iterdir()] == ["record.json"]


def test_refused_payload_creates_no_file(tmp_path):
    with pytest.raises(ValueError):
        storage.write_json(tmp_path / "record.json", {"x": float("nan")})
    assert list(tmp_path.iterdir()) == []


def test_failed_write_keeps_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "record.bin"
    target.write_bytes(b"old")

    def failing_fsync(descriptor):
        raise OSError(errno.EIO, "disk failure")

    monkeypatch.setattr(storage.os, "fsync", failing_fsync)
    with pytest.raises(OSError):
        storage.write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert [path.name for path in tmp_path.iterdir()] == ["record.bin"]


@pytest.mark.parametrize("code, raises", [(errno.EINVAL, False), (errno.EIO, True)])
def test_directory_sync_skips_only_unsupported_filesystems(tmp_path, monkeypatch, code, raises):
    real_fsync = os.fsync

    def fsync(descriptor):
        # Mark the descriptor by the object it currently names: the closed
        # temp-file fd number is reused by the directory open, so an fd-number
        # set would let a real directory error through.
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(code, os.strerror(code))
        return real_fsync(descriptor)

    monkeypatch.setattr(storage.os, "fsync", fsync)
    target = tmp_path / "record.bin"
    if raises:
        with pytest.raises(OSError):
            storage.write_bytes(target, b"data")
    else:
        storage.write_bytes(target, b"data")
    # The rename happened before the directory sync either way.
    assert target.read_bytes() == b"data"
