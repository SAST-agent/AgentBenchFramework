"""Validate and import an immutable v5 parent for the v6 HL act."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping

from agentbench_frame.tracking.snapshot import (
    LocalWorkspaceSnapshotter,
    WorkspaceManifest,
)


@dataclass(frozen=True)
class Round6ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    raw_score: float
    evo_score_1: float
    evo_score_2: float
    evo_score_3: float
    evo_score_4: float
    evo_score_5: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v5_source: Path
    v5_manifest: WorkspaceManifest
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
                str(item) for item in value.get("changed_files", [])
            ],
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"cannot read parent v5 manifest: {exc}") from exc


def _learning_only(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key).startswith("learning_")
    }


def _is_exact_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_round6_parent(
    parent_run_dir: Path,
    expected_parent_hash: str,
) -> Round6ParentLineage:
    """Read and verify a complete v5 parent without modifying it."""
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads(
            (parent / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read round-6 parent summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("round-6 parent summary must be an object")
    if summary.get("status") != "complete":
        raise ValueError("round-6 parent run must be complete")
    if summary.get("evaluation_status") != "complete":
        raise ValueError(
            "round-6 parent formal evaluation must be complete"
        )
    score_keys = (
        "raw_score",
        "evo_score_1",
        "evo_score_2",
        "evo_score_3",
        "evo_score_4",
        "evo_score_5",
    )
    if any(summary.get(key) is None for key in score_keys):
        raise ValueError(
            "round-6 parent must contain complete v0 through v5 scores"
        )
    scores = tuple(float(summary[key]) for key in score_keys)
    expected_history = (
        scores[0],
        scores[1],
        None,
        scores[2],
        scores[3],
        scores[4],
        scores[5],
    )
    declared_history = summary.get("score_history")
    if (
        not isinstance(declared_history, list)
        or tuple(declared_history) != expected_history
    ):
        raise ValueError(
            "round-6 parent score history must preserve the failed-act gap"
        )
    global_acts = summary.get("act_count")
    if not _is_exact_integer(global_acts) or global_acts != 6:
        raise ValueError("round-6 parent act count must equal six")
    learning_budget = _learning_only(
        summary.get("cumulative_learning_budget")
    )
    if not learning_budget:
        raise ValueError(
            "round-6 parent cumulative learning budget is missing"
    )
    recorded_acts = learning_budget.get("learning_coding_agent_acts")
    if (
        not _is_exact_integer(recorded_acts)
        or recorded_acts != global_acts
    ):
        raise ValueError(
            "round-6 parent learning act count does not match lineage"
        )

    source = parent / "versions" / "v5" / "source"
    if not source.is_dir():
        raise ValueError("parent v5 source snapshot is missing")
    declared = _read_manifest(parent / "versions" / "v5" / "manifest.json")
    actual = LocalWorkspaceSnapshotter().capture(source)
    if (
        declared.content_hash != actual.content_hash
        or declared.files != actual.files
    ):
        raise ValueError("parent v5 manifest does not match source")
    if actual.content_hash != expected_parent_hash:
        raise ValueError("unexpected parent v5 content hash")

    return Round6ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v5",
        raw_score=scores[0],
        evo_score_1=scores[1],
        evo_score_2=scores[2],
        evo_score_3=scores[3],
        evo_score_4=scores[4],
        evo_score_5=scores[5],
        global_act_count=global_acts,
        prior_score_history=expected_history,
        v5_source=source,
        v5_manifest=WorkspaceManifest(
            content_hash=actual.content_hash,
            files=actual.files,
            changed_files=declared.changed_files,
        ),
        learning_budget=learning_budget,
    )


def import_round6_source(
    lineage: Round6ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    """Copy exact v5 source into a v6 run and preserve its lineage."""
    target = Path(run_dir)
    workspace = target / "workspace"
    version_source = target / "versions" / "v5" / "source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-6 import destination already exists")
    snapshotter.materialize_manifest(
        lineage.v5_source,
        workspace,
        lineage.v5_manifest,
    )
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = snapshotter.capture(workspace)
    if (
        imported.content_hash != lineage.v5_manifest.content_hash
        or imported.files != lineage.v5_manifest.files
    ):
        raise ValueError("imported v5 content does not match parent")
    snapshotter.materialize_manifest(
        lineage.v5_source,
        version_source,
        lineage.v5_manifest,
    )
    preserved = WorkspaceManifest(
        content_hash=lineage.v5_manifest.content_hash,
        files=dict(lineage.v5_manifest.files),
        changed_files=list(lineage.v5_manifest.changed_files),
    )
    snapshotter.write_manifest(
        preserved,
        target / "versions" / "v5" / "manifest.json",
    )
    lineage_record = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "parent_content_hash": preserved.content_hash,
        "starting_version": "v5",
    }
    (target / "versions" / "lineage.json").write_text(
        json.dumps(lineage_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved
