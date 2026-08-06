"""Validate and import the immutable v7 parent for clean-room v8."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter, WorkspaceManifest


@dataclass(frozen=True)
class Round8ParentLineage:
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
    evo_score_7: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v7_source: Path
    v7_manifest: WorkspaceManifest
    learning_budget: Mapping[str, object]


def _read_manifest(path: Path) -> WorkspaceManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return WorkspaceManifest(
            content_hash=str(raw["content_hash"]),
            files={str(key): str(value) for key, value in raw["files"].items()},
            changed_files=[str(item) for item in raw.get("changed_files", [])],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read parent v7 manifest: {exc}") from exc


def _learning_only(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if str(key).startswith("learning_")}


def load_round8_parent(parent_run_dir: Path, expected_parent_hash: str) -> Round8ParentLineage:
    parent = Path(parent_run_dir).resolve()
    try:
        summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read round-8 parent summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("round-8 parent summary must be an object")
    if summary.get("status") != "complete":
        raise ValueError("round-8 parent run must be complete")
    if summary.get("evaluation_status") != "complete" or summary.get("formal_attempted") is not True:
        raise ValueError("round-8 parent formal evaluation must be complete")
    keys = ("raw_score", "evo_score_1", "evo_score_2", "evo_score_3", "evo_score_4", "evo_score_5", "evo_score_6", "evo_score_7")
    if any(summary.get(key) is None for key in keys):
        raise ValueError("round-8 parent must contain complete v0 through v7 scores")
    scores = tuple(float(summary[key]) for key in keys)
    history = (scores[0], scores[1], None, scores[2], scores[3], scores[4], scores[5], scores[6], scores[7])
    if summary.get("score_history") != list(history):
        raise ValueError("round-8 parent score history must preserve the failed-act gap")
    acts = summary.get("act_count")
    if type(acts) is not int or acts != 8:
        raise ValueError("round-8 parent act count must equal eight")
    budget = _learning_only(summary.get("cumulative_learning_budget"))
    if not budget:
        raise ValueError("round-8 parent cumulative learning budget is missing")
    if type(budget.get("learning_coding_agent_acts")) is not int or budget["learning_coding_agent_acts"] != acts:
        raise ValueError("round-8 parent learning act count does not match lineage")
    source = parent / "versions/v7/source"
    if not source.is_dir():
        raise ValueError("parent v7 source snapshot is missing")
    declared = _read_manifest(parent / "versions/v7/manifest.json")
    actual = LocalWorkspaceSnapshotter().capture(source)
    if declared.content_hash != actual.content_hash or declared.files != actual.files:
        raise ValueError("parent v7 manifest does not match source")
    if actual.content_hash != expected_parent_hash:
        raise ValueError("unexpected parent v7 content hash")
    return Round8ParentLineage(
        parent_run_dir=parent,
        parent_run_id=str(summary.get("run_id", parent.name)),
        parent_version="v7",
        expected_parent_hash=expected_parent_hash,
        raw_score=scores[0], evo_score_1=scores[1], evo_score_2=scores[2],
        evo_score_3=scores[3], evo_score_4=scores[4], evo_score_5=scores[5],
        evo_score_6=scores[6], evo_score_7=scores[7], global_act_count=acts,
        prior_score_history=history, v7_source=source,
        v7_manifest=WorkspaceManifest(actual.content_hash, actual.files, declared.changed_files),
        learning_budget=budget,
    )


def import_round8_source(
    lineage: Round8ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    target = Path(run_dir)
    workspace = target / "workspace"
    version_source = target / "versions/v7/source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-8 import destination already exists")
    snapshotter.materialize_manifest(lineage.v7_source, workspace, lineage.v7_manifest)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, capture_output=True, text=True)
    imported = snapshotter.capture(workspace)
    if imported.content_hash != lineage.v7_manifest.content_hash or imported.files != lineage.v7_manifest.files:
        raise ValueError("imported v7 content does not match parent")
    snapshotter.materialize_manifest(lineage.v7_source, version_source, lineage.v7_manifest)
    preserved = WorkspaceManifest(
        lineage.v7_manifest.content_hash,
        dict(lineage.v7_manifest.files),
        list(lineage.v7_manifest.changed_files),
    )
    snapshotter.write_manifest(preserved, target / "versions/v7/manifest.json")
    record = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": "v7",
        "expected_parent_hash": lineage.expected_parent_hash,
        "observed_parent_hash": preserved.content_hash,
        "starting_version": "v7",
        "global_act_count": lineage.global_act_count,
        "prior_score_history": list(lineage.prior_score_history),
        "inherited_learning_budget": dict(lineage.learning_budget),
    }
    path = target / "versions/lineage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return preserved
