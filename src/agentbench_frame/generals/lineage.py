"""Validate and import the immutable first-run v1 into a new round-2 run."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from agentbench_frame.tracking.snapshot import (
    LocalWorkspaceSnapshotter,
    WorkspaceManifest,
)


@dataclass(frozen=True)
class ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    raw_score: float
    evo_score_1: float
    v0_source: Path
    v1_source: Path
    v1_manifest: WorkspaceManifest
    learning_budget: Mapping[str, object]
    first_diff: str


def _read_manifest(path: Path) -> WorkspaceManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return WorkspaceManifest(
            content_hash=str(value["content_hash"]),
            files={str(key): str(item) for key, item in value["files"].items()},
            changed_files=[str(item) for item in value.get("changed_files", [])],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read parent manifest: {exc}") from exc


def load_parent_lineage(
    parent_run_dir: Path,
    expected_hash: str,
) -> ParentLineage:
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read parent summary: {exc}") from exc
    if summary.get("status") != "complete":
        raise ValueError("parent run must be complete")
    if summary.get("raw_score") is None or summary.get("evo_score") is None:
        raise ValueError("parent run must contain complete v0 and v1 scores")
    v0_source = parent / "versions" / "v0" / "source"
    v1_source = parent / "versions" / "v1" / "source"
    if not v0_source.is_dir() or not v1_source.is_dir():
        raise ValueError("parent v0/v1 source snapshot is missing")
    declared = _read_manifest(parent / "versions" / "v1" / "manifest.json")
    actual = LocalWorkspaceSnapshotter().capture(v1_source)
    if declared.content_hash != actual.content_hash or declared.files != actual.files:
        raise ValueError("parent v1 manifest does not match source")
    if actual.content_hash != expected_hash:
        raise ValueError("unexpected parent v1 content hash")
    try:
        first_diff = (parent / "versions" / "v0-to-v1.patch").read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ValueError(f"parent v0-to-v1 diff is missing: {exc}") from exc
    budget = summary.get("budget")
    learning_budget = {
        str(key): value
        for key, value in (budget.items() if isinstance(budget, dict) else ())
        if str(key).startswith("learning_")
    }
    return ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v1",
        raw_score=float(summary["raw_score"]),
        evo_score_1=float(summary["evo_score"]),
        v0_source=v0_source,
        v1_source=v1_source,
        v1_manifest=WorkspaceManifest(
            content_hash=actual.content_hash,
            files=actual.files,
            changed_files=declared.changed_files,
        ),
        learning_budget=learning_budget,
        first_diff=first_diff,
    )


def import_parent_v1(
    lineage: ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    target_run = Path(run_dir)
    workspace = target_run / "workspace"
    version_source = target_run / "versions" / "v1" / "source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-2 import destination already exists")
    shutil.copytree(lineage.v1_source, workspace)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = snapshotter.capture(workspace)
    if (
        imported.content_hash != lineage.v1_manifest.content_hash
        or imported.files != lineage.v1_manifest.files
    ):
        raise ValueError("imported v1 content does not match parent")
    shutil.copytree(lineage.v1_source, version_source)
    preserved = WorkspaceManifest(
        content_hash=imported.content_hash,
        files=imported.files,
        changed_files=lineage.v1_manifest.changed_files,
    )
    snapshotter.write_manifest(
        preserved, target_run / "versions" / "v1" / "manifest.json"
    )
    lineage_record = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "content_hash": preserved.content_hash,
    }
    (target_run / "versions" / "v1" / "lineage.json").write_text(
        json.dumps(lineage_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved
