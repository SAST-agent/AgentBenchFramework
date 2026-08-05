"""Atomic five-file projection into the unchanged AgentBenchResults layout."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .run_store import DotoRunStore, InvalidTransition, hash_file


FILES = {
    "run.toml", "summary.json", "score_curve.json", "ig_curve.json",
    "doto_results_ref.json",
}


@dataclass(frozen=True)
class ProjectionResult:
    path: Path
    sealed_content_sha256: str


def _sealed_hash(run_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name in {"projection.json", ".run.lock"} or path.name.endswith(".tmp"):
            continue
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _strict_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"non-finite JSON constant {token}")))

    def finite(item) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if isinstance(item, dict):
            return all(finite(child) for child in item.values())
        if isinstance(item, list):
            return all(finite(child) for child in item)
        return True

    if not finite(value):
        raise ValueError(f"non-finite JSON in {path}")
    return value


def _write_projection_status(run_dir: Path, state: str, *, target: Path,
                             error: str | None = None) -> None:
    DotoRunStore.write_json_atomic(run_dir / "projection.json", {
        "schema_version": 1, "state": state, "target": str(target.resolve()),
        "error": error,
    })


def _existing_matches(destination: Path, run_dir: Path, sealed_hash: str) -> bool:
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != FILES:
        return False
    for name in ("run.toml", "summary.json", "score_curve.json", "ig_curve.json"):
        if (destination / name).read_bytes() != (run_dir / name).read_bytes():
            return False
    try:
        reference = _strict_json(destination / "doto_results_ref.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        reference.get("authoritative_run") == str(run_dir.resolve())
        and reference.get("sealed_content_sha256") == sealed_hash
        and reference.get("authoritative_summary_sha256") == hash_file(run_dir / "summary.json")
        and reference.get("authoritative_manifest_sha256") == hash_file(run_dir / "run.toml")
    )


def export_agentbench_projection(authoritative_run: Path,
                                 agentbench_results: Path) -> ProjectionResult:
    run_dir = Path(authoritative_run).resolve()
    required = {"run.toml", "summary.json", "score_curve.json", "ig_curve.json", "final-test.json"}
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        raise ValueError(f"authoritative Run is incomplete: {missing}")
    with (run_dir / "run.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    summary = _strict_json(run_dir / "summary.json")
    _strict_json(run_dir / "score_curve.json")
    _strict_json(run_dir / "ig_curve.json")
    if metadata.get("state") != "finalized" or summary.get("state") != "finalized":
        raise InvalidTransition("only a finalized authoritative Run can be projected")
    for field in ("wall_hours", "total_steps", "win_rate"):
        if (field not in summary or not isinstance(summary[field], (int, float))
                or isinstance(summary[field], bool)):
            raise ValueError(f"authoritative summary lacks scalar {field}")

    destination = (Path(agentbench_results) / "runs" / "23_doto"
                   / str(metadata["agent"]) / str(metadata["run_id"]))
    sealed_hash = _sealed_hash(run_dir)
    _write_projection_status(run_dir, "pending", target=destination)
    if destination.exists():
        if _existing_matches(destination, run_dir, sealed_hash):
            _write_projection_status(run_dir, "exported", target=destination)
            return ProjectionResult(destination, sealed_hash)
        error = "destination contains a different projection"
        _write_projection_status(run_dir, "failed", target=destination, error=error)
        raise InvalidTransition(error)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in ("run.toml", "summary.json", "score_curve.json", "ig_curve.json"):
            shutil.copy2(run_dir / name, temporary / name)
        DotoRunStore.write_json_atomic(temporary / "doto_results_ref.json", {
            "schema_version": 1,
            "authoritative_run": str(run_dir),
            "authoritative_run_id": str(metadata["run_id"]),
            "sealed_content_sha256": sealed_hash,
            "authoritative_summary_sha256": hash_file(run_dir / "summary.json"),
            "authoritative_manifest_sha256": hash_file(run_dir / "run.toml"),
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        if {path.name for path in temporary.iterdir()} != FILES:
            raise RuntimeError("projection staging file set is invalid")
        os.replace(temporary, destination)
        _write_projection_status(run_dir, "exported", target=destination)
        return ProjectionResult(destination, sealed_hash)
    except Exception as error:
        shutil.rmtree(temporary, ignore_errors=True)
        _write_projection_status(run_dir, "failed", target=destination,
                                 error=f"{type(error).__name__}: {error}")
        raise
