"""Local workspace manifest snapshots for version tracking."""

import hashlib
import difflib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Union


PathLike = Union[str, os.PathLike]


@dataclass
class WorkspaceManifest:
    """Content-addressed metadata for one readable workspace state."""

    content_hash: str
    files: Dict[str, str]
    changed_files: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LocalWorkspaceSnapshotter:
    """Hash source files without copying or mutating the workspace."""

    DEFAULT_EXCLUDES = frozenset({".git", ".venv", "__pycache__", "agentbench_data"})

    def __init__(self, excludes: Optional[Iterable[str]] = None) -> None:
        self.excludes = frozenset(excludes or self.DEFAULT_EXCLUDES)

    def file_hash(self, path: PathLike) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def capture(
        self, root: PathLike, previous: Optional[WorkspaceManifest] = None
    ) -> WorkspaceManifest:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ValueError(f"workspace root is not a directory: {root}")
        files: Dict[str, str] = {}
        for current_root, dirnames, filenames in os.walk(root_path):
            dirnames[:] = sorted(name for name in dirnames if name not in self.excludes)
            for filename in sorted(filenames):
                path = Path(current_root) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(root_path).as_posix()
                files[relative] = self.file_hash(path)

        manifest_digest = hashlib.sha256()
        for relative, digest in sorted(files.items()):
            manifest_digest.update(relative.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(digest.encode("ascii"))
            manifest_digest.update(b"\n")

        previous_files = previous.files if previous is not None else {}
        changed = sorted(
            relative
            for relative in set(files) | set(previous_files)
            if files.get(relative) != previous_files.get(relative)
        )
        return WorkspaceManifest(
            content_hash=manifest_digest.hexdigest(),
            files=files,
            changed_files=changed,
        )

    def write_manifest(self, manifest: WorkspaceManifest, path: PathLike) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)

    def materialize_manifest(
        self,
        root: PathLike,
        destination: PathLike,
        manifest: WorkspaceManifest,
    ) -> WorkspaceManifest:
        """Copy exactly one verified manifest into a clean destination."""
        source_root = Path(root)
        if source_root.is_symlink() or not source_root.is_dir():
            raise ValueError(f"manifest source is not a directory: {root}")
        source_root = source_root.resolve()

        verified_sources: list[tuple[str, Path]] = []
        for relative, expected_hash in sorted(manifest.files.items()):
            manifest_path = PurePosixPath(relative)
            if (
                not relative
                or "\\" in relative
                or manifest_path.is_absolute()
                or manifest_path.as_posix() != relative
                or any(
                    part in {"", ".", ".."} or part in self.excludes
                    for part in manifest_path.parts
                )
            ):
                raise ValueError(f"unsafe manifest path: {relative!r}")

            source = source_root
            for part in manifest_path.parts:
                source = source / part
                if source.is_symlink():
                    raise ValueError(
                        f"manifest source is a symlink: {relative}"
                    )
            if not source.is_file():
                raise ValueError(f"manifest source file is missing: {relative}")
            actual_hash = self.file_hash(source)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"manifest source hash mismatch: {relative}"
                )
            verified_sources.append((relative, source))

        target_root = Path(destination)
        if target_root.exists():
            if target_root.is_symlink() or not target_root.is_dir():
                raise ValueError(
                    f"manifest destination is not a directory: {destination}"
                )
            unexpected = sorted(
                child.name
                for child in target_root.iterdir()
                if child.name != ".git"
            )
            if unexpected:
                raise ValueError(
                    "manifest destination is not pre-cleared: "
                    + ", ".join(unexpected)
                )
        else:
            target_root.mkdir(parents=True)

        for relative, source in verified_sources:
            output = target_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)

        actual = self.capture(target_root)
        if (
            actual.content_hash != manifest.content_hash
            or actual.files != manifest.files
        ):
            raise ValueError(
                "materialized content does not match manifest"
            )
        return actual

    def copy_snapshot(
        self, root: PathLike, destination: PathLike
    ) -> WorkspaceManifest:
        root_path = Path(root).resolve()
        manifest = self.capture(root_path)
        target = Path(destination)
        self.materialize_manifest(root_path, target, manifest)
        self.write_manifest(manifest, target.parent / "manifest.json")
        return manifest

    def write_unified_patch(
        self, before_root: PathLike, after_root: PathLike, destination: PathLike
    ) -> None:
        before = Path(before_root)
        after = Path(after_root)
        paths = sorted(
            {
                path.relative_to(before).as_posix()
                for path in before.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            | {
                path.relative_to(after).as_posix()
                for path in after.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
        )
        chunks: list[str] = []
        for relative in paths:
            old_bytes = (before / relative).read_bytes() if (before / relative).is_file() else b""
            new_bytes = (after / relative).read_bytes() if (after / relative).is_file() else b""
            if old_bytes == new_bytes:
                continue
            try:
                old_text = old_bytes.decode("utf-8").splitlines(keepends=True)
                new_text = new_bytes.decode("utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                chunks.append(f"Binary files a/{relative} and b/{relative} differ\n")
                continue
            chunks.extend(
                difflib.unified_diff(
                    old_text,
                    new_text,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(chunks), encoding="utf-8")
