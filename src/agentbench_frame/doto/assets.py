"""Paths and integrity checks for the vendored DOTO assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


class AssetIntegrityError(RuntimeError):
    """Raised when a vendored official asset differs from its manifest."""


def official_server_dir() -> Path:
    return PACKAGE_DIR / "official_server"


def sdk_dir() -> Path:
    return PACKAGE_DIR / "sdk"


def verify_assets() -> dict[str, str]:
    manifest = json.loads((PACKAGE_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    actual: dict[str, str] = {}
    for relative, expected in manifest["sha256"].items():
        path = PACKAGE_DIR / relative
        if not path.is_file():
            raise AssetIntegrityError(f"asset missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise AssetIntegrityError(f"asset hash mismatch: {relative}")
        actual[relative] = digest
    return actual
