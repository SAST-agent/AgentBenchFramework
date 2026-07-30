"""Immutable content-addressed candidate versions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


IGNORED_DIRS = frozenset(
    {"__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".git", ".agentbench"}
)
IGNORED_SUFFIXES = (".pyc", ".pyo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclasses.dataclass(frozen=True)
class Version:
    version_id: str
    content_hash: str
    parent_version_id: Optional[str]
    act_id: str
    edit_type: str
    created_at: str
    files: tuple[str, ...]


def _read_tree(root: Path) -> list[tuple[str, bytes]]:
    if not root.is_dir():
        raise FileNotFoundError(f"candidate workspace is not a directory: {root}")
    files: list[tuple[str, bytes]] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            if filename.endswith(IGNORED_SUFFIXES):
                continue
            path = Path(directory) / filename
            if path.is_symlink():
                raise ValueError(f"candidate workspace cannot contain symlink: {path}")
            relative = path.relative_to(root).as_posix()
            files.append((relative, path.read_bytes()))
    return sorted(files)


def _hash_tree(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, content in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


class VersionStore:
    """Snapshot and restore one candidate workspace without nested Git."""

    def __init__(self, workspace: str | Path, root: str | Path) -> None:
        self.workspace = Path(workspace)
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)

    def _next_version_id(self) -> str:
        numbers = []
        for path in self.manifests.glob("v*.json"):
            try:
                numbers.append(int(path.stem[1:]))
            except ValueError:
                continue
        return f"v{(max(numbers, default=-1) + 1):06d}"

    def snapshot(
        self,
        *,
        parent_version_id: Optional[str],
        act_id: str,
        edit_type: str = "candidate",
    ) -> Version:
        files = _read_tree(self.workspace)
        content_hash = _hash_tree(files)
        object_root = self.objects / content_hash
        if not object_root.exists():
            object_root.mkdir()
            for relative, content in files:
                destination = object_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        version = Version(
            version_id=self._next_version_id(),
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            act_id=act_id,
            edit_type="initial" if parent_version_id is None else edit_type,
            created_at=_now(),
            files=tuple(relative for relative, _ in files),
        )
        manifest_path = self.manifests / f"{version.version_id}.json"
        manifest_path.write_text(
            json.dumps(dataclasses.asdict(version), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return version

    def get(self, version_id: str) -> Version:
        path = self.manifests / f"{version_id}.json"
        if not path.is_file():
            raise KeyError(version_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        value["files"] = tuple(value["files"])
        return Version(**value)

    def restore(
        self,
        *,
        source_version_id: str,
        parent_version_id: str,
        act_id: str,
    ) -> Version:
        self.checkout(source_version_id)
        return self.snapshot(
            parent_version_id=parent_version_id,
            act_id=act_id,
            edit_type="rollback",
        )

    def checkout(self, version_id: str) -> None:
        """Place an immutable version in the workspace without creating a version."""

        source = self.get(version_id)
        object_root = self.objects / source.content_hash
        if not object_root.is_dir():
            raise FileNotFoundError(f"missing snapshot object: {source.content_hash}")
        self.workspace.mkdir(parents=True, exist_ok=True)
        for child in self.workspace.iterdir():
            if child.name in IGNORED_DIRS:
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        for relative, content in _read_tree(object_root):
            destination = self.workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
