"""Lock-protected atomic storage for Codex-directed DOTO Runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Mapping


DOTO_SKILL_NAMES = (
    "doto-benchmark-run", "doto-game-rules",
    "doto-agent-authoring", "doto-replay-reader",
)


@dataclass(frozen=True)
class SkillSnapshot:
    name: str
    sha256: str


class InvalidTransition(RuntimeError):
    pass


class RunState(StrEnum):
    CREATED = "created"
    ITERATING = "iterating"
    FINAL_TEST_STARTED = "final_test_started"
    FINALIZED = "finalized"


class IterationState(StrEnum):
    OPEN = "iteration_open"
    BUILT = "candidate_built"
    BUILD_FAILED = "build_failed"
    EVALUATED = "training_evaluated"
    INCOMPLETE = "evaluation_incomplete"
    IG_RECORDED = "ig_recorded"
    IG_MISSING = "ig_missing"
    CLOSED = "iteration_closed"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in Path(path).rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(item.read_bytes() + b"\0")
    return digest.hexdigest()


def snapshot_skills(run_dir: Path, roots: Mapping[str, Path]) -> tuple[SkillSnapshot, ...]:
    """Atomically snapshot exactly the four DOTO Skill packages."""

    if set(roots) != set(DOTO_SKILL_NAMES):
        raise ValueError("Run requires exactly the four DOTO Skill packages")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "skills"
    if destination.exists():
        raise ValueError("Run Skill snapshot already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".skills-", dir=run_dir))
    rows: list[SkillSnapshot] = []
    try:
        for name in sorted(DOTO_SKILL_NAMES):
            root = Path(roots[name])
            skill_file = root / "SKILL.md"
            if not root.is_dir() or not skill_file.is_file():
                raise ValueError(f"Skill package is missing SKILL.md: {name}")
            if root.is_symlink() or any(item.is_symlink() for item in root.rglob("*")):
                raise ValueError(f"Skill package contains a symlink: {name}")
            declared = next((
                line.split(":", 1)[1].strip().strip("'\"")
                for line in skill_file.read_text(encoding="utf-8").splitlines()
                if line.startswith("name:")
            ), "")
            if declared != name:
                raise ValueError(f"Skill frontmatter name differs from package: {name}")
            target = temporary / name
            shutil.copytree(root, target)
            source_hash = hash_directory(root)
            target_hash = hash_directory(target)
            if source_hash != target_hash:
                raise ValueError(f"Skill snapshot hash mismatch: {name}")
            rows.append(SkillSnapshot(name, target_hash))
        DotoRunStore.write_json_atomic(temporary / "manifest.json", {
            "schema_version": 1,
            "skills": [{"name": row.name, "sha256": row.sha256} for row in rows],
        })
        os.replace(temporary, destination)
        return tuple(rows)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class DotoRunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.run_id = self.run_dir.name

    @classmethod
    def open(cls, run_dir: Path) -> "DotoRunStore":
        path = Path(run_dir)
        if not (path / "run.toml").is_file():
            raise ValueError(f"not a DOTO Run: {path}")
        return cls(path)

    @property
    def state(self) -> RunState:
        with (self.run_dir / "run.toml").open("rb") as stream:
            raw = tomllib.load(stream)
        return RunState(raw["state"])

    def reload(self) -> "DotoRunStore":
        return self.open(self.run_dir)

    @contextmanager
    def locked(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / ".run.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def write_text_atomic(path: Path, value: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def write_json_atomic(cls, path: Path | str, value) -> None:
        cls.write_text_atomic(Path(path), json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n")

    def _append_event_unlocked(self, event: str, **fields) -> None:
        row = {"event": event, "timestamp": time.time(), **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_event(self, event: str, **fields) -> None:
        with self.locked():
            self._append_event_unlocked(event, **fields)

    def iteration_dir(self, index: int) -> Path:
        path = self.run_dir / "iterations" / f"iteration-{index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path
