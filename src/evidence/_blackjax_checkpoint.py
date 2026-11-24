"""Atomic, input-bound BlackJAX execution state; restore only trusted local files.

Like dynesty recovery, the payload uses pickle and is not an interchange format.
The atomic rename means a reader sees either the previous or the new complete
file; the ``.npz`` (zip) CRC-32 of each member, checked by ``np.load``, detects
a damaged file. Neither guards against a malicious pickle. Supported values are
nested dicts, lists, tuples (including named tuples), init-field dataclasses,
NumPy/JAX arrays and typed JAX PRNG keys.
"""
from dataclasses import fields, is_dataclass, replace
import json
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from trex.accelerators import _jax_models as models

# A stored typed PRNG key: its implementation name and raw key data.
_KEY_FIELDS = frozenset({"trex_key_impl", "key_data"})


def _to_host(value: Any) -> Any:
    """Device arrays to NumPy and typed keys to their stored form, recursively."""
    jax = models.jax
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _to_host(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    if isinstance(value, dict):
        return {key: _to_host(item) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [_to_host(item) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_to_host(item) for item in value]
    if isinstance(value, jax.Array):
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            return {"trex_key_impl": str(jax.random.key_impl(value)),
                    "key_data": np.asarray(jax.random.key_data(value))}
        return np.asarray(value)
    return value


def _to_device(value: Any, device: Any) -> Any:
    """Inverse of :func:`_to_host`: NumPy arrays and stored keys back onto ``device``."""
    jax = models.jax
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _to_device(getattr(value, field.name), device)
                                 for field in fields(value) if field.init})
    if isinstance(value, dict):
        if (set(value) == _KEY_FIELDS and isinstance(value["trex_key_impl"], str)
                and isinstance(value["key_data"], np.ndarray)):
            return jax.random.wrap_key_data(jax.device_put(value["key_data"], device), impl=value["trex_key_impl"])
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [_to_device(item, device) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, np.ndarray):
        return jax.device_put(value, device)
    return value


def _require_private_directory(directory: Path) -> None:
    if directory.is_symlink():
        raise ValueError("BlackJAX recovery requires an owner-private directory")
    status = directory.stat()
    if status.st_uid != os.geteuid() or status.st_mode & 0o077:
        raise ValueError("BlackJAX recovery requires an owner-private directory")


def save_native(path, identity: str, value: Any) -> None:
    """Publish a complete logical boundary, never a partially written payload.

    The temporary file is unique in the target directory, so two writers never
    share it; the directory is synced after the rename so the new name survives
    a crash.
    """
    path = Path(path)
    payload = pickle.dumps(_to_host(value), protocol=5)
    metadata = json.dumps({"identity": identity}, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_directory(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, metadata=np.array(metadata), payload=np.frombuffer(payload, dtype=np.uint8))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_native(path, identity: str, *, device: Optional[Any] = None) -> Any:
    """Restore native arrays and keys without evaluating the scientific integrand.

    Returns ``None`` when no checkpoint exists. A symlinked or foreign-owned file
    or directory, an identity mismatch, and any unreadable, incomplete or
    corrupt payload raise ``ValueError``.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError("BlackJAX recovery requires owner-controlled files in a private directory")
    if not path.exists():
        return None
    _require_private_directory(path.parent)
    if path.stat().st_uid != os.geteuid():
        raise ValueError("BlackJAX recovery requires owner-controlled files in a private directory")
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            payload = stored["payload"].tobytes()
        stored_identity = metadata["identity"]
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise ValueError("BlackJAX checkpoint is corrupt or incomplete") from error
    if stored_identity != identity:
        from trex.evidence.nested_checkpoint import CheckpointInputMismatch

        raise CheckpointInputMismatch("BlackJAX checkpoint inputs do not match this integration")
    try:
        restored = pickle.loads(payload)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, TypeError) as error:
        raise ValueError("BlackJAX checkpoint is corrupt or incomplete") from error
    return _to_device(restored, device)
