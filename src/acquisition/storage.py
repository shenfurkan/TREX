"""Durably retain acquisition responses in the caller's output directory."""

import errno
import json
import os
import tempfile
from pathlib import Path

# Filesystems or platforms that cannot open or fsync a directory (some network
# and FUSE mounts, Windows). The renamed file is already complete there.
_DIRECTORY_SYNC_UNSUPPORTED = {getattr(errno, name) for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "EACCES", "EISDIR")
                               if hasattr(errno, name)}


def _sync_directory(directory: Path) -> None:
    """Persist a rename; skip only where directory sync is unsupported."""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError as exc:
        if exc.errno in _DIRECTORY_SYNC_UNSUPPORTED:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in _DIRECTORY_SYNC_UNSUPPORTED:
            raise
    finally:
        os.close(descriptor)


def write_bytes(path, content: bytes) -> None:
    """Atomically publish bytes at ``path`` and sync its directory entry.

    Readers see the old file or the complete new one, never a partial write.
    A symlink at ``path`` is replaced by the file, not followed. The file has
    ``tempfile.mkstemp``'s owner-only (0o600) permissions. A real I/O error,
    including one while syncing the directory, propagates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        published = True
        _sync_directory(path.parent)
    finally:
        if not published:
            temporary.unlink(missing_ok=True)


def write_json(path, payload) -> None:
    """Atomically publish a strict JSON evidence record.

    ``payload`` may contain only JSON types (dict, list, str, int, float,
    bool, None); NaN and infinities are refused before any file is created.
    """
    write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))

