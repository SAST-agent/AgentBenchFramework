"""Local workspace manifest snapshots for version tracking."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
