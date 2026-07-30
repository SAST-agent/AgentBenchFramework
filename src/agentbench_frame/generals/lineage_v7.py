"""Validate and import an immutable v6 parent for the v7 champion act."""

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
class Round7ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    expected_parent_hash: str
    raw_score: float
    evo_score_1: float
    evo_score_2: float
    evo_score_3: float
    evo_score_4: float
    evo_score_5: float
    evo_score_6: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v6_source: Path
    v6_manifest: WorkspaceManifest
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
        raise ValueError(f"cannot read parent v6 manifest: {exc}") from exc


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


def load_round7_parent(
    parent_run_dir: Path,
    expected_parent_hash: str,
) -> Round7ParentLineage:
    """Read and verify a complete v6 parent without modifying it."""
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads(
            (parent / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read round-7 parent summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("round-7 parent summary must be an object")
    if summary.get("status") != "complete":
        raise ValueError("round-7 parent run must be complete")
    if (
        summary.get("evaluation_status") != "complete"
        or summary.get("formal_attempted") is not True
    ):
        raise ValueError(
            "round-7 parent formal evaluation must be complete"
        )

    score_keys = (
        "raw_score",
        "evo_score_1",
        "evo_score_2",
        "evo_score_3",
        "evo_score_4",
        "evo_score_5",
        "evo_score_6",
    )
    if any(summary.get(key) is None for key in score_keys):
        raise ValueError(
            "round-7 parent must contain complete v0 through v6 scores"
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
        scores[6],
    )
    declared_history = summary.get("score_history")
    if (
        not isinstance(declared_history, list)
        or tuple(declared_history) != expected_history
    ):
        raise ValueError(
            "round-7 parent score history must preserve the failed-act gap"
        )

    global_acts = summary.get("act_count")
    if not _is_exact_integer(global_acts) or global_acts != 7:
        raise ValueError("round-7 parent act count must equal seven")
    learning_budget = _learning_only(
        summary.get("cumulative_learning_budget")
    )
    if not learning_budget:
        raise ValueError(
            "round-7 parent cumulative learning budget is missing"
        )
    recorded_acts = learning_budget.get("learning_coding_agent_acts")
    if (
        not _is_exact_integer(recorded_acts)
        or recorded_acts != global_acts
    ):
        raise ValueError(
            "round-7 parent learning act count does not match lineage"
        )

    source = parent / "versions" / "v6" / "source"
    if not source.is_dir():
        raise ValueError("parent v6 source snapshot is missing")
    declared = _read_manifest(parent / "versions" / "v6" / "manifest.json")
    actual = LocalWorkspaceSnapshotter().capture(source)
    if (
        declared.content_hash != actual.content_hash
        or declared.files != actual.files
    ):
        raise ValueError("parent v6 manifest does not match source")
    if actual.content_hash != expected_parent_hash:
        raise ValueError("unexpected parent v6 content hash")

    return Round7ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v6",
        expected_parent_hash=expected_parent_hash,
        raw_score=scores[0],
        evo_score_1=scores[1],
        evo_score_2=scores[2],
        evo_score_3=scores[3],
        evo_score_4=scores[4],
        evo_score_5=scores[5],
        evo_score_6=scores[6],
        global_act_count=global_acts,
        prior_score_history=expected_history,
        v6_source=source,
        v6_manifest=WorkspaceManifest(
            content_hash=actual.content_hash,
            files=actual.files,
            changed_files=declared.changed_files,
        ),
        learning_budget=learning_budget,
    )


def import_round7_source(
    lineage: Round7ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    """Copy exact v6 source into a v7 run and preserve full lineage."""
    target = Path(run_dir)
    workspace = target / "workspace"
    version_source = target / "versions" / "v6" / "source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-7 import destination already exists")
    snapshotter.materialize_manifest(
        lineage.v6_source,
        workspace,
        lineage.v6_manifest,
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
        imported.content_hash != lineage.v6_manifest.content_hash
        or imported.files != lineage.v6_manifest.files
    ):
        raise ValueError("imported v6 content does not match parent")
    snapshotter.materialize_manifest(
        lineage.v6_source,
        version_source,
        lineage.v6_manifest,
    )
    preserved = WorkspaceManifest(
        content_hash=lineage.v6_manifest.content_hash,
        files=dict(lineage.v6_manifest.files),
        changed_files=list(lineage.v6_manifest.changed_files),
    )
    snapshotter.write_manifest(
        preserved,
        target / "versions" / "v6" / "manifest.json",
    )
    lineage_record = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "expected_parent_hash": lineage.expected_parent_hash,
        "observed_parent_hash": preserved.content_hash,
        "starting_version": "v6",
        "global_act_count": lineage.global_act_count,
        "prior_score_history": list(lineage.prior_score_history),
        "inherited_learning_budget": dict(lineage.learning_budget),
    }
    lineage_path = target / "versions" / "lineage.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(
        json.dumps(lineage_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved
