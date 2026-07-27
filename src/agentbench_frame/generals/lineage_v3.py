"""Validate and import the immutable pilot v2 into a round-3 rescue run."""

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
class Round3ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    raw_score: float
    evo_score_1: float
    evo_score_2: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v2_source: Path
    v2_manifest: WorkspaceManifest
    learning_budget: Mapping[str, object]
    second_diff: str


def _read_manifest(path: Path) -> WorkspaceManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return WorkspaceManifest(
            content_hash=str(value["content_hash"]),
            files={
                str(key): str(item)
                for key, item in value["files"].items()
            },
            changed_files=[
                str(item)
                for item in value.get("changed_files", [])
            ],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read parent v2 manifest: {exc}") from exc


def load_round3_parent(
    parent_run_dir: Path,
    expected_hash: str,
) -> Round3ParentLineage:
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads(
            (parent / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read round-3 parent summary: {exc}") from exc
    if summary.get("status") != "complete":
        raise ValueError("round-3 parent run must be complete")
    score_keys = ("raw_score", "evo_score_1", "evo_score_2")
    if any(summary.get(key) is None for key in score_keys):
        raise ValueError(
            "round-3 parent must contain complete v0, v1, and v2 scores"
        )
    source = parent / "versions" / "v2" / "source"
    if not source.is_dir():
        raise ValueError("parent v2 source snapshot is missing")
    declared = _read_manifest(
        parent / "versions" / "v2" / "manifest.json"
    )
    actual = LocalWorkspaceSnapshotter().capture(source)
    if (
        declared.content_hash != actual.content_hash
        or declared.files != actual.files
    ):
        raise ValueError("parent v2 manifest does not match source")
    if actual.content_hash != expected_hash:
        raise ValueError("unexpected parent v2 content hash")
    try:
        second_diff = (parent / "versions" / "v1-to-v2.patch").read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ValueError(f"parent v1-to-v2 diff is missing: {exc}") from exc
    recovery = bool(summary.get("recovery_from_run_id"))
    declared_acts = int(summary.get("act_count", 2))
    global_acts = max(declared_acts, 3) if recovery else declared_acts
    raw = float(summary["raw_score"])
    evo_1 = float(summary["evo_score_1"])
    evo_2 = float(summary["evo_score_2"])
    history = (
        (raw, evo_1, None, evo_2)
        if recovery
        else (raw, evo_1, evo_2)
    )
    preferred_budget = summary.get("cumulative_learning_budget")
    if not isinstance(preferred_budget, dict):
        preferred_budget = summary.get("budget")
    learning_budget = {
        str(key): value
        for key, value in (
            preferred_budget.items()
            if isinstance(preferred_budget, dict)
            else ()
        )
        if str(key).startswith("learning_")
    }
    return Round3ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v2",
        raw_score=raw,
        evo_score_1=evo_1,
        evo_score_2=evo_2,
        global_act_count=global_acts,
        prior_score_history=history,
        v2_source=source,
        v2_manifest=WorkspaceManifest(
            content_hash=actual.content_hash,
            files=actual.files,
            changed_files=declared.changed_files,
        ),
        learning_budget=learning_budget,
        second_diff=second_diff,
    )


def import_parent_v2(
    lineage: Round3ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    target_run = Path(run_dir)
    workspace = target_run / "workspace"
    version_source = target_run / "versions" / "v2" / "source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-3 import destination already exists")
    shutil.copytree(lineage.v2_source, workspace)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = snapshotter.capture(workspace)
    if (
        imported.content_hash != lineage.v2_manifest.content_hash
        or imported.files != lineage.v2_manifest.files
    ):
        raise ValueError("imported v2 content does not match parent")
    shutil.copytree(lineage.v2_source, version_source)
    preserved = WorkspaceManifest(
        content_hash=imported.content_hash,
        files=imported.files,
        changed_files=lineage.v2_manifest.changed_files,
    )
    snapshotter.write_manifest(
        preserved,
        target_run / "versions" / "v2" / "manifest.json",
    )
    lineage_record = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "content_hash": preserved.content_hash,
        "global_act_count": lineage.global_act_count,
    }
    (target_run / "versions" / "v2" / "lineage.json").write_text(
        json.dumps(lineage_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved
