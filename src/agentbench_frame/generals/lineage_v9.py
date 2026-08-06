"""Validate v7 as v9 source parent and v8 as chronological predecessor."""

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
class Round9Lineage:
    policy_parent_run_dir: Path
    policy_parent_run_id: str
    policy_parent_version: str
    policy_parent_hash: str
    iteration_predecessor_run_dir: Path
    iteration_predecessor_run_id: str
    iteration_predecessor_version: str
    iteration_predecessor_hash: str
    next_global_act_count: int
    score_history: tuple[float | None, ...]
    v7_source: Path
    v7_manifest: WorkspaceManifest
    v8_source: Path
    v8_manifest: WorkspaceManifest
    inherited_learning_budget: Mapping[str, object]


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _manifest(path: Path, label: str) -> WorkspaceManifest:
    raw = _read_object(path, label)
    try:
        return WorkspaceManifest(
            content_hash=str(raw["content_hash"]),
            files={str(key): str(value) for key, value in raw["files"].items()},
            changed_files=[str(item) for item in raw.get("changed_files", [])],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _verified_source(
    run_dir: Path,
    version: str,
) -> tuple[Path, WorkspaceManifest]:
    source = run_dir / f"versions/{version}/source"
    declared = _manifest(
        run_dir / f"versions/{version}/manifest.json",
        f"{version} manifest",
    )
    try:
        actual = LocalWorkspaceSnapshotter().capture(source)
    except ValueError as exc:
        raise ValueError(f"{version} manifest does not match source: {exc}") from exc
    if (
        actual.content_hash != declared.content_hash
        or actual.files != declared.files
    ):
        raise ValueError(f"{version} manifest does not match source")
    return source, WorkspaceManifest(
        actual.content_hash,
        dict(actual.files),
        list(declared.changed_files),
    )


def _history(raw: object, label: str) -> tuple[float | None, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} score history is missing")
    values: list[float | None] = []
    for value in raw:
        if value is None:
            values.append(None)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        else:
            raise ValueError(f"{label} score history is invalid")
    return tuple(values)


def load_round9_lineage(
    policy_parent_run_dir: Path,
    iteration_predecessor_run_dir: Path,
) -> Round9Lineage:
    v7_dir = Path(policy_parent_run_dir).resolve()
    v8_dir = Path(iteration_predecessor_run_dir).resolve()
    v7_summary = _read_object(v7_dir / "summary.json", "v7 source-parent summary")
    v8_summary = _read_object(v8_dir / "summary.json", "v8 predecessor summary")

    if v7_summary.get("status") != "complete":
        raise ValueError("v7 source-parent run must be complete")
    if (
        v7_summary.get("evaluation_status") != "complete"
        or v7_summary.get("formal_attempted") is not True
        or v7_summary.get("runnable") is not True
    ):
        raise ValueError("v7 source-parent evaluation must be complete and runnable")
    if v8_summary.get("status") != "complete":
        raise ValueError("v8 predecessor run must be complete")
    if (
        v8_summary.get("evaluation_status") != "complete"
        or v8_summary.get("formal_attempted") is not True
    ):
        raise ValueError("v8 predecessor evaluation must be complete")
    if v8_summary.get("runnable") is not True:
        raise ValueError("v8 predecessor must be runnable")
    if v7_summary.get("run_id") != v7_dir.name:
        raise ValueError("v7 source-parent run identity changed")
    if v8_summary.get("run_id") != v8_dir.name:
        raise ValueError("v8 predecessor run identity changed")

    v7_acts = v7_summary.get("act_count")
    v8_acts = v8_summary.get("act_count")
    if type(v7_acts) is not int or v7_acts != 8:
        raise ValueError("v7 source-parent act count must equal eight")
    if type(v8_acts) is not int or v8_acts != v7_acts + 1:
        raise ValueError("v8 predecessor act count must follow v7")

    v7_history = _history(v7_summary.get("score_history"), "v7")
    v8_history = _history(v8_summary.get("score_history"), "v8")
    if len(v7_history) != 9 or v7_history[2] is not None:
        raise ValueError("v7 score history must preserve the historical gap")
    if len(v8_history) != len(v7_history) + 1 or v8_history[:-1] != v7_history:
        raise ValueError("v8 score history must extend v7 exactly once")
    if v7_summary.get("evo_score_7") != v7_history[-1]:
        raise ValueError("v7 terminal score does not match score history")
    if v8_summary.get("evo_score_8") != v8_history[-1]:
        raise ValueError("v8 terminal score does not match score history")

    v7_source, v7_manifest = _verified_source(v7_dir, "v7")
    v8_source, v8_manifest = _verified_source(v8_dir, "v8")
    if (
        v8_summary.get("parent_run_id") != v7_dir.name
        or v8_summary.get("parent_version") != "v7"
        or v8_summary.get("parent_content_hash") != v7_manifest.content_hash
    ):
        raise ValueError("v8 predecessor parent must be the verified v7 authority")

    budget_raw = v7_summary.get("cumulative_learning_budget")
    budget = (
        {
            str(key): value
            for key, value in budget_raw.items()
            if str(key).startswith("learning_")
        }
        if isinstance(budget_raw, dict)
        else {}
    )
    if not budget or budget.get("learning_coding_agent_acts") != v7_acts:
        raise ValueError("v7 source-parent learning budget is inconsistent")

    return Round9Lineage(
        policy_parent_run_dir=v7_dir,
        policy_parent_run_id=v7_dir.name,
        policy_parent_version="v7",
        policy_parent_hash=v7_manifest.content_hash,
        iteration_predecessor_run_dir=v8_dir,
        iteration_predecessor_run_id=v8_dir.name,
        iteration_predecessor_version="v8",
        iteration_predecessor_hash=v8_manifest.content_hash,
        next_global_act_count=v8_acts + 1,
        score_history=v8_history,
        v7_source=v7_source,
        v7_manifest=v7_manifest,
        v8_source=v8_source,
        v8_manifest=v8_manifest,
        inherited_learning_budget=budget,
    )


def import_round9_parent(
    lineage: Round9Lineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> WorkspaceManifest:
    target = Path(run_dir)
    workspace = target / "workspace"
    version_source = target / "versions/v7/source"
    if workspace.exists() or version_source.exists():
        raise ValueError("round-9 import destination already exists")
    snapshotter.materialize_manifest(
        lineage.v7_source,
        workspace,
        lineage.v7_manifest,
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
        imported.content_hash != lineage.v7_manifest.content_hash
        or imported.files != lineage.v7_manifest.files
    ):
        raise ValueError("imported v7 content does not match source parent")
    snapshotter.materialize_manifest(
        lineage.v7_source,
        version_source,
        lineage.v7_manifest,
    )
    snapshotter.write_manifest(
        lineage.v7_manifest,
        target / "versions/v7/manifest.json",
    )
    record = {
        "policy_parent_run_id": lineage.policy_parent_run_id,
        "policy_parent_version": lineage.policy_parent_version,
        "policy_parent_hash": lineage.policy_parent_hash,
        "iteration_predecessor_run_id": lineage.iteration_predecessor_run_id,
        "iteration_predecessor_version": lineage.iteration_predecessor_version,
        "iteration_predecessor_hash": lineage.iteration_predecessor_hash,
        "starting_version": "v7",
        "next_version": "v9",
        "next_global_act_count": lineage.next_global_act_count,
        "score_history": list(lineage.score_history),
        "inherited_learning_budget": dict(lineage.inherited_learning_budget),
    }
    path = target / "versions/lineage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return lineage.v7_manifest
