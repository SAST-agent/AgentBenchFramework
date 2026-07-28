"""Validate v4 lineage and import the immutable v3 rollback source."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
class Round5ParentLineage:
    parent_run_dir: Path
    parent_run_id: str
    parent_version: str
    starting_version: str
    raw_score: float
    evo_score_1: float
    evo_score_2: float
    evo_score_3: float
    evo_score_4: float
    global_act_count: int
    prior_score_history: tuple[float | None, ...]
    v3_source: Path
    v4_source: Path
    v3_manifest: WorkspaceManifest
    v4_manifest: WorkspaceManifest
    learning_budget: Mapping[str, object]
    campaign_budget_receipt: Mapping[str, object]


def _read_manifest(path: Path, version: str) -> WorkspaceManifest:
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
        raise ValueError(
            f"cannot read parent {version} manifest: {exc}"
        ) from exc


def _verified_snapshot(
    parent: Path,
    version: str,
    expected_hash: str,
) -> tuple[Path, WorkspaceManifest]:
    source = parent / "versions" / version / "source"
    if not source.is_dir():
        raise ValueError(f"parent {version} source snapshot is missing")
    declared = _read_manifest(
        parent / "versions" / version / "manifest.json",
        version,
    )
    actual = LocalWorkspaceSnapshotter().capture(source)
    if (
        declared.content_hash != actual.content_hash
        or declared.files != actual.files
    ):
        raise ValueError(
            f"parent {version} manifest does not match source"
        )
    if actual.content_hash != expected_hash:
        raise ValueError(f"unexpected parent {version} content hash")
    return source, WorkspaceManifest(
        content_hash=actual.content_hash,
        files=actual.files,
        changed_files=declared.changed_files,
    )


def _load_campaign_receipt(
    path: Path,
    parent_summary_path: Path,
    parent_summary_raw: bytes,
    parent_run_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read v4 campaign budget receipt: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise ValueError("v4 campaign budget receipt must be an object")
    if (
        receipt.get("status") != "derived_prior_attempt_added"
        or receipt.get("mutation_policy")
        != "inputs_immutable_separate_derived_receipt"
    ):
        raise ValueError("v4 campaign budget receipt is not audited")
    if str(receipt.get("success_run_id")) != parent_run_id:
        raise ValueError("v4 campaign receipt success run mismatch")
    try:
        referenced_summary = Path(
            str(receipt["success_summary_ref"])
        ).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "v4 campaign receipt summary reference is invalid"
        ) from exc
    if referenced_summary != parent_summary_path.resolve():
        raise ValueError("v4 campaign receipt summary reference mismatch")
    actual_hash = hashlib.sha256(parent_summary_raw).hexdigest()
    if receipt.get("success_summary_sha256") != actual_hash:
        raise ValueError("v4 campaign receipt summary hash mismatch")
    after = receipt.get("after")
    if not isinstance(after, dict):
        raise ValueError("v4 campaign receipt after budget is missing")
    learning = {
        str(key): value
        for key, value in after.items()
        if str(key).startswith("learning_")
    }
    if int(learning.get("learning_coding_agent_acts", -1)) != 5:
        raise ValueError("v4 campaign receipt act count must equal five")
    return receipt, learning


def load_round5_parent(
    parent_run_dir: Path,
    expected_parent_hash: str,
    expected_rollback_hash: str,
    campaign_budget_receipt: Path,
) -> Round5ParentLineage:
    """Verify v4, its embedded v3, and the external audited budget."""
    parent = Path(parent_run_dir).resolve()
    summary_path = parent / "summary.json"
    try:
        summary_raw = summary_path.read_bytes()
        summary = json.loads(summary_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read round-5 parent summary: {exc}"
        ) from exc
    if summary.get("status") != "complete":
        raise ValueError("round-5 parent run must be complete")
    if summary.get("evaluation_status") != "complete":
        raise ValueError(
            "round-5 parent formal evaluation must be complete"
        )
    score_keys = (
        "raw_score",
        "evo_score_1",
        "evo_score_2",
        "evo_score_3",
        "evo_score_4",
    )
    if any(summary.get(key) is None for key in score_keys):
        raise ValueError(
            "round-5 parent must contain complete v0 through v4 scores"
        )
    scores = tuple(float(summary[key]) for key in score_keys)
    expected_history = (
        scores[0],
        scores[1],
        None,
        scores[2],
        scores[3],
        scores[4],
    )
    declared_history = summary.get("score_history")
    if (
        not isinstance(declared_history, list)
        or tuple(declared_history) != expected_history
    ):
        raise ValueError(
            "round-5 parent score history must preserve the failed-act gap"
        )
    global_acts = int(summary.get("act_count", -1))
    if global_acts != 5:
        raise ValueError("round-5 parent act count must equal five")
    run_id = str(summary.get("run_id", parent.name))
    receipt, learning_budget = _load_campaign_receipt(
        campaign_budget_receipt,
        summary_path,
        summary_raw,
        run_id,
    )
    v3_source, v3_manifest = _verified_snapshot(
        parent,
        "v3",
        expected_rollback_hash,
    )
    v4_source, v4_manifest = _verified_snapshot(
        parent,
        "v4",
        expected_parent_hash,
    )
    for protected in ("main.py", "state_view.py"):
        if (
            v3_manifest.files.get(protected) is None
            or v3_manifest.files.get(protected)
            != v4_manifest.files.get(protected)
        ):
            raise ValueError(
                f"protected file differs across v3 and v4: {protected}"
            )
    return Round5ParentLineage(
        parent_run_dir=parent,
        parent_run_id=run_id,
        parent_version="v4",
        starting_version="v3",
        raw_score=scores[0],
        evo_score_1=scores[1],
        evo_score_2=scores[2],
        evo_score_3=scores[3],
        evo_score_4=scores[4],
        global_act_count=global_acts,
        prior_score_history=expected_history,
        v3_source=v3_source,
        v4_source=v4_source,
        v3_manifest=v3_manifest,
        v4_manifest=v4_manifest,
        learning_budget=learning_budget,
        campaign_budget_receipt=receipt,
    )


def _preserved_manifest(
    manifest: WorkspaceManifest,
) -> WorkspaceManifest:
    return WorkspaceManifest(
        content_hash=manifest.content_hash,
        files=dict(manifest.files),
        changed_files=list(manifest.changed_files),
    )


def import_round5_sources(
    lineage: Round5ParentLineage,
    run_dir: Path,
    snapshotter: LocalWorkspaceSnapshotter,
) -> tuple[WorkspaceManifest, WorkspaceManifest]:
    """Start from v3 while retaining both audited source snapshots."""
    target = Path(run_dir)
    workspace = target / "workspace"
    v3_target = target / "versions" / "v3" / "source"
    v4_target = target / "versions" / "v4" / "source"
    if any(path.exists() for path in (workspace, v3_target, v4_target)):
        raise ValueError("round-5 import destination already exists")
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
        raise ValueError("imported v3 rollback content does not match parent")
    shutil.copytree(lineage.v3_source, v3_target)
    shutil.copytree(lineage.v4_source, v4_target)
    preserved_v3 = _preserved_manifest(lineage.v3_manifest)
    preserved_v4 = _preserved_manifest(lineage.v4_manifest)
    snapshotter.write_manifest(
        preserved_v3,
        target / "versions" / "v3" / "manifest.json",
    )
    snapshotter.write_manifest(
        preserved_v4,
        target / "versions" / "v4" / "manifest.json",
    )
    rollback = {
        "parent_run_id": lineage.parent_run_id,
        "parent_version": lineage.parent_version,
        "parent_content_hash": preserved_v4.content_hash,
        "starting_version": lineage.starting_version,
        "rollback_source_version": "v3",
        "rollback_content_hash": preserved_v3.content_hash,
    }
    rollback_path = target / "versions" / "rollback.json"
    rollback_path.parent.mkdir(parents=True, exist_ok=True)
    rollback_path.write_text(
        json.dumps(rollback, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return preserved_v3, preserved_v4
