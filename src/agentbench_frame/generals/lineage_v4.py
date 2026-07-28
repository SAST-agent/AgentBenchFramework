"""Validate and import an immutable v3 parent for the v4 HL act."""

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
class Round4ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    raw_score: float
    evo_score_1: float
    evo_score_2: float
    evo_score_3: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v3_source: Path
    v3_manifest: WorkspaceManifest
    learning_budget: Mapping[str, object]


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
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"cannot read parent v3 manifest: {exc}") from exc


def _learning_only(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key).startswith("learning_")
    }


def load_round4_parent(
    parent_run_dir: Path,
    expected_hash: str,
) -> Round4ParentLineage:
    """Read and verify v3 without modifying any finalized parent artifact."""
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads(
            (parent / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read round-4 parent summary: {exc}") from exc
    if summary.get("status") != "complete":
        raise ValueError("round-4 parent run must be complete")
    if summary.get("evaluation_status") != "complete":
        raise ValueError(
            "round-4 parent formal evaluation must be complete"
        )
    score_keys = (
        "raw_score",
        "evo_score_1",
        "evo_score_2",
        "evo_score_3",
    )
    if any(summary.get(key) is None for key in score_keys):
        raise ValueError(
            "round-4 parent must contain complete v0 through v3 scores"
        )
    raw = float(summary["raw_score"])
    evo_1 = float(summary["evo_score_1"])
    evo_2 = float(summary["evo_score_2"])
    evo_3 = float(summary["evo_score_3"])
    expected_history = (raw, evo_1, None, evo_2, evo_3)
    declared_history = summary.get("score_history")
    if (
        not isinstance(declared_history, list)
        or tuple(declared_history) != expected_history
    ):
        raise ValueError(
            "round-4 parent score history must preserve the failed-act gap"
        )
    global_acts = int(summary.get("act_count", -1))
    if global_acts != 4:
        raise ValueError("round-4 parent act count must equal four")
    learning_budget = _learning_only(
        summary.get("cumulative_learning_budget")
    )
    if not learning_budget:
        raise ValueError(
            "round-4 parent cumulative learning budget is missing"
        )
    recorded_acts = learning_budget.get("learning_coding_agent_acts")
    if recorded_acts is None or int(recorded_acts) != global_acts:
        raise ValueError(
            "round-4 parent learning act count does not match lineage"
        )

    source = parent / "versions" / "v3" / "source"
    if not source.is_dir():
        raise ValueError("parent v3 source snapshot is missing")
    declared = _read_manifest(
        parent / "versions" / "v3" / "manifest.json"
    )
    actual = LocalWorkspaceSnapshotter().capture(source)
    if (
        declared.content_hash != actual.content_hash
        or declared.files != actual.files
    ):
        raise ValueError("parent v3 manifest does not match source")
    if actual.content_hash != expected_hash:
        raise ValueError("unexpected parent v3 content hash")

    return Round4ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v3",
        raw_score=raw,
        evo_score_1=evo_1,
        evo_score_2=evo_2,
        evo_score_3=evo_3,
        global_act_count=global_acts,
        prior_score_history=expected_history,
        v3_source=source,
        v3_manifest=WorkspaceManifest(
            content_hash=actual.content_hash,
            files=actual.files,
            changed_files=declared.changed_files,
        ),
        learning_budget=learning_budget,
    )


def import_parent_v3(
    lineage: Round4ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    """Copy the exact v3 source into a new run and save a child receipt."""
    target_run = Path(run_dir)
    workspace = target_run / "workspace"
    version_source = target_run / "versions" / "v3" / "source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-4 import destination already exists")
    shutil.copytree(lineage.v3_source, workspace)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = snapshotter.capture(workspace)
    if (
        imported.content_hash != lineage.v3_manifest.content_hash
        or imported.files != lineage.v3_manifest.files
    ):
        raise ValueError("imported v3 content does not match parent")
    shutil.copytree(lineage.v3_source, version_source)
    preserved = WorkspaceManifest(
        content_hash=imported.content_hash,
        files=imported.files,
        changed_files=lineage.v3_manifest.changed_files,
    )
    snapshotter.write_manifest(
        preserved,
        target_run / "versions" / "v3" / "manifest.json",
    )
    receipt = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "content_hash": preserved.content_hash,
        "global_act_count": lineage.global_act_count,
    }
    (target_run / "versions" / "v3" / "lineage.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved
