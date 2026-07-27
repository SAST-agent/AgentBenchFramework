"""Atomic JSON file writer: temp file + fsync + os.replace.

Guarantees the final path is never partially written. The new content is
written to a same-directory temp file, fsync'd, then atomically renamed over
the target with ``os.replace``. A crash anywhere before the rename leaves the
target with its prior complete content (or absent); a leftover temp file
(``.<name>.*.tmp``) is the only audit trace. Used for vendor result-json so a
force-killed vendor can never leave a half-written result.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_SENTINEL = object()
_WINDOWS_SHARING_WINERRORS = {5, 32}


def _is_windows() -> bool:
    """Return the host platform without making tests mutate global ``os``."""
    return os.name == "nt"


def _replace_with_windows_retry(tmp: str, target: str) -> None:
    """Retry only transient Windows sharing/access failures within 150ms."""
    for i, delay in enumerate((0.01, 0.02, 0.04, 0.08), start=1):
        try:
            os.replace(tmp, target)
            return
        except PermissionError as exc:
            if not _is_windows() or getattr(exc, "winerror", None) not in _WINDOWS_SHARING_WINERRORS or i == 4:
                raise
            time.sleep(delay)


def atomic_write_json(path, obj: Any, *, encoding: str = "utf-8",
                      indent: int = 2, default=_SENTINEL) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            if default is _SENTINEL:
                json.dump(obj, f, ensure_ascii=False, indent=indent)
            else:
                json.dump(obj, f, ensure_ascii=False, indent=indent, default=default)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_windows_retry(tmp, str(p))
    except BaseException:
        # never leave the target half-written; leave the temp for audit (do NOT unlink)
        raise
    return p
