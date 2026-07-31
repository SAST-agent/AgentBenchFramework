"""Immutable content-addressed candidate versions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_object(
    object_root: Path,
    *,
    content_hash: str,
    expected_files: tuple[str, ...],
) -> None:
    files = _read_tree(object_root)
    actual_files = tuple(relative for relative, _ in files)
    if actual_files != expected_files:
        raise ValueError(f"snapshot object file manifest mismatch: {content_hash}")
    if _hash_tree(files) != content_hash:
        raise ValueError(f"snapshot object content hash mismatch: {content_hash}")


def _load_version(root: Path, version_id: str) -> Version:
    path = root / "manifests" / f"{version_id}.json"
    if not path.is_file():
        raise KeyError(version_id)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"] = tuple(value["files"])
    version = Version(**value)
    if version.version_id != version_id:
        raise ValueError(f"manifest version id mismatch: {version_id}")
    object_root = root / "objects" / version.content_hash
    if not object_root.is_dir():
        raise FileNotFoundError(
            f"missing snapshot object: {version.content_hash}"
        )
    _verify_object(
        object_root,
        content_hash=version.content_hash,
        expected_files=version.files,
    )
    return version


def _replace_workspace(workspace: Path, object_root: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for child in workspace.iterdir():
        if child.name in IGNORED_DIRS:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    for relative, content in _read_tree(object_root):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


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

    def current_content_hash(self) -> str:
        """Hash the current candidate tree without creating a version."""

        return _hash_tree(_read_tree(self.workspace))

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
        expected_files = tuple(relative for relative, _ in files)
        if object_root.exists():
            _verify_object(
                object_root,
                content_hash=content_hash,
                expected_files=expected_files,
            )
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{content_hash}.", dir=self.objects)
            )
            try:
                for relative, content in files:
                    destination = temporary / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                _verify_object(
                    temporary,
                    content_hash=content_hash,
                    expected_files=expected_files,
                )
                os.replace(temporary, object_root)
                _fsync_directory(self.objects)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        version = Version(
            version_id=self._next_version_id(),
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            act_id=act_id,
            edit_type=(
                "initial"
                if parent_version_id is None and edit_type == "candidate"
                else edit_type
            ),
            created_at=_now(),
            files=expected_files,
        )
        manifest_path = self.manifests / f"{version.version_id}.json"
        encoded = (
            json.dumps(
                dataclasses.asdict(version),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
        _fsync_directory(self.manifests)
        return version

    def get(self, version_id: str) -> Version:
        return _load_version(self.root, version_id)

    def import_version(
        self,
        source_versions_root: str | Path,
        source_version_id: str,
        *,
        act_id: str = "imported-origin",
    ) -> tuple[Version, Version]:
        """Restore and snapshot a verified version from another run."""

        source_root = Path(source_versions_root)
        source = _load_version(source_root, source_version_id)
        _replace_workspace(
            self.workspace,
            source_root / "objects" / source.content_hash,
        )
        imported = self.snapshot(
            parent_version_id=None,
            act_id=act_id,
            edit_type="imported_origin",
        )
        if imported.content_hash != source.content_hash:
            raise ValueError("imported origin content hash mismatch")
        return source, imported

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
        _replace_workspace(self.workspace, object_root)
