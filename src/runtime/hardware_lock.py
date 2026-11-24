"""Short, OS-backed critical sections for host accelerator lease metadata.

The persistent sidecar inode must never be unlinked: deleting a locked inode
would let another process lock a different inode at the same path. The OS
releases these advisory locks on process exit. POSIX flock and Windows byte
range locking are local-host mechanisms, not distributed/NFS leases.
"""

from __future__ import annotations

import os
import errno
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from trex.runtime.logging import host_logger

PRIVATE_FILE_MODE = 0o600
LOCK_BYTE_COUNT = 1
LOCK_TIMEOUT_SECONDS = 5.0


@contextmanager
def lease_metadata_lock(path: Path, *, blocking: bool = False) -> Iterator[bool]:
    """Serialize locally; blocking requests wait at most five seconds.

    POSIX O_NOFOLLOW refuses sidecar symlinks and fchmod repairs old permissions.
    Sidecars persist to preserve lock identity; Windows locks beyond EOF without
    an unlocked initialization write. Closing also releases a failed explicit unlock.
    """
    descriptor = None
    unlock = None
    try:
        descriptor = os.open(
            str(path) + ".lock",
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            PRIVATE_FILE_MODE,
        )
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        if os.name == "nt":
            import msvcrt

            def lock() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, LOCK_BYTE_COUNT)

            def unlock() -> None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, LOCK_BYTE_COUNT)
        else:
            import fcntl

            def lock() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock() -> None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        deadline = time.monotonic() + (LOCK_TIMEOUT_SECONDS if blocking else 0)
        while True:
            try:
                lock()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    except (ImportError, OSError):
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        yield False
        return
    try:
        yield True
    finally:
        body_error = sys.exc_info()[1]
        cleanup_error = None
        try:
            unlock()
        except OSError as exc:
            cleanup_error = exc
            host_logger("hardware").warning(
                "Lease metadata unlock failed; closing descriptor", exc_info=True,
                extra={"event": "trex.hardware.lease_cleanup", "error_type": type(exc).__name__},
            )
        finally:
            try:
                os.close(descriptor)
                cleanup_error = None
            except OSError as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            if body_error is not None:
                body_error.add_note("lease metadata lock cleanup failed")
            else:
                raise cleanup_error


__all__ = [
    "LOCK_BYTE_COUNT",
    "PRIVATE_FILE_MODE",
    "lease_metadata_lock",
]
