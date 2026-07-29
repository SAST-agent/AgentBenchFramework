"""
HLCodebase — the versioned, agent-editable artifact for heuristic learning.

The workspace is a plain directory the coding agent edits (today one
``agent.py``; extensible to a package via ``manifest.toml`` — see
``hl/manifest.py``). The controller never edits the workspace; it only
*snapshots* it.

Versioning: content-hash snapshot store (no git inside the agent tree).
- ``snapshot()`` content-hashes the tree, copies it into the store, returns a
  ``VersionHandle``. New ``version_id`` per call; identical ``content_hash``
  across versions is allowed (doc §12.1).
- ``restore()`` copies a prior snapshot back into the workspace and returns a
  NEW ``version_id`` with ``edit_type='rollback'`` — zero policy-KL but
  nonzero act budget, recorded (never silently folded).
- ``diff()`` is a structured file-level diff consumed by the EditType
  classifier.

Crash-safe: a partially-edited (but readable) tree still snapshots; the
controller records ``version_after=None`` only when the tree is unreadable,
and even then the act event persists (doc §12.1).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

#: Directories ignored when hashing/snapshotting (caches, not source).
IGNORED_DIRS = {"__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".git"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class VersionHandle:
    """One immutable version of the codebase."""
    version_id: str
    content_hash: str
    parent_version_id: Optional[str]
    edit_type: str            # initial | add_rule | reorder | parametrize | refactor | replace
                              #         | utility | search | planner | consolidate | rollback | noop
    created_at: str


@dataclass(frozen=True)
class StructuredDiff:
    added: List[str]
    removed: List[str]
    modified: List[str]
    unchanged: List[str]


def _walk(root: Path) -> List[Tuple[str, bytes]]:
    """Return sorted [(relpath_posix, bytes)] of source files in the tree."""
    out: List[Tuple[str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored dirs in-place
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fn in filenames:
            if any(fn.endswith(s) for s in IGNORED_SUFFIXES):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            try:
                data = full.read_bytes()
            except OSError:
                continue
            out.append((rel, data))
    out.sort(key=lambda t: t[0])
    return out


def _content_hash(files: List[Tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for rel, data in files:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "little"))
        h.update(b"\0")
        h.update(data)
    return h.hexdigest()[:32]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class HLCodebase:
    """A versioned, agent-editable codebase rooted at ``root``."""

    def __init__(self, root, store):
        self.root = Path(root)
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self._counter = 0  # for version_id uniqueness within this process

    def _next_version_id(self) -> str:
        self._counter += 1
        return f"v{_now_iso().replace(':', '').replace('.', '')}_{self._counter:06d}"

    def snapshot(self, parent_version_id: Optional[str],
                 edit_type: str = "noop") -> VersionHandle:
        """Hash + copy the current workspace into the store. Always succeeds
        for a readable tree (even a partial one). Returns a VersionHandle."""
        files = _walk(self.root)
        content_hash = _content_hash(files)
        snap_dir = self.store / content_hash
        if not snap_dir.exists():
            snap_dir.mkdir(parents=True)
            for rel, data in files:
                dst = snap_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
        et = edit_type if parent_version_id is not None else "initial"
        return VersionHandle(
            version_id=self._next_version_id(),
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            edit_type=et,
            created_at=_now_iso(),
        )

    def restore(self, content_hash: str,
                parent_version_id: str) -> VersionHandle:
        """Copy a prior snapshot back into the workspace. Returns a NEW
        version_id with edit_type='rollback'."""
        snap_dir = self.store / content_hash
        if not snap_dir.exists():
            raise FileNotFoundError(f"no snapshot for content_hash={content_hash}")
        # wipe the workspace (source-only files), then repopulate
        for child in self.root.iterdir():
            if child.name in IGNORED_DIRS:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for rel, data in _walk(snap_dir):
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        return VersionHandle(
            version_id=self._next_version_id(),
            content_hash=content_hash,
            parent_version_id=parent_version_id,
            edit_type="rollback",
            created_at=_now_iso(),
        )

    def diff(self, before_hash: str, after_hash: str) -> StructuredDiff:
        """Structured file-level diff between two snapshots."""
        before = {rel: data for rel, data in _walk(self.store / before_hash)}
        after = {rel: data for rel, data in _walk(self.store / after_hash)}
        b_keys, a_keys = set(before), set(after)
        added = sorted(a_keys - b_keys)
        removed = sorted(b_keys - a_keys)
        modified = sorted(k for k in (b_keys & a_keys) if before[k] != after[k])
        unchanged = sorted(k for k in (b_keys & a_keys) if before[k] == after[k])
        return StructuredDiff(
            added=added, removed=removed,
            modified=modified, unchanged=unchanged,
        )
