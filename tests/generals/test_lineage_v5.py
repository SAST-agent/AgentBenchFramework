import hashlib
import json
from pathlib import Path

import pytest

from agentbench_frame.generals.lineage_v5 import (
    import_round5_sources,
    load_round5_parent,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _write_source(path: Path, version: str) -> None:
    (path / "tests").mkdir(parents=True)
    (path / "main.py").write_text("PROTECTED = True\n", encoding="utf-8")
    (path / "state_view.py").write_text(
        "def normalize(value): return value\n",
        encoding="utf-8",
    )
    (path / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        f"    return [[8]]  # {version}\n",
        encoding="utf-8",
    )
    (path / "STRATEGY.md").write_text(
        f"{version} strategy\n",
        encoding="utf-8",
    )
    (path / "EXPERIENCE.md").write_text(
        f"{version} experience\n",
        encoding="utf-8",
    )
    (path / "tests" / "test_strategy.py").write_text(
        "def test_strategy_contract(): assert True\n",
        encoding="utf-8",
    )


def _parent(tmp_path: Path):
    parent = tmp_path / "parent-v4"
    snapshotter = LocalWorkspaceSnapshotter()
    manifests = {}
    for version in ("v3", "v4"):
        source = parent / "versions" / version / "source"
        _write_source(source, version)
        manifest = snapshotter.capture(source)
        snapshotter.write_manifest(
            manifest,
            parent / "versions" / version / "manifest.json",
        )
        manifests[version] = manifest

    summary = {
        "run_id": "parent-v4",
        "status": "complete",
        "evaluation_status": "complete",
        "raw_score": 0.0,
        "evo_score_1": 0.0,
        "evo_score_2": 0.0,
        "evo_score_3": 7 / 18,
        "evo_score_4": 4 / 18,
        "score_history": [
            0.0,
            0.0,
            None,
            0.0,
            7 / 18,
            4 / 18,
        ],
        "act_count": 5,
        "cumulative_learning_budget": {
            "learning_coding_agent_acts": 5,
            "learning_episodes": 52,
        },
    }
    summary_path = parent / "summary.json"
    summary_raw = json.dumps(summary, indent=2).encode()
    summary_path.write_bytes(summary_raw)
    receipt = tmp_path / "derived" / "v4-budget.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({
            "schema_version": "1.0",
            "status": "derived_prior_attempt_added",
            "success_run_id": "parent-v4",
            "success_summary_ref": str(summary_path),
            "success_summary_sha256": hashlib.sha256(
                summary_raw
            ).hexdigest(),
            "mutation_policy": (
                "inputs_immutable_separate_derived_receipt"
            ),
            "after": {
                "learning_coding_agent_acts": 5,
                "learning_episodes": 58,
                "learning_env_steps": 30882,
                "learning_game_agent_decision_steps": 12091,
                "learning_primitive_commands": 50465,
                "learning_prompt_tokens": None,
                "learning_completion_tokens": None,
                "learning_total_tokens": None,
                "learning_time_s": 779.94,
            },
        }),
        encoding="utf-8",
    )
    return parent, manifests["v3"], manifests["v4"], receipt


def test_round5_parent_uses_v4_lineage_v3_workspace_and_audited_budget(
    tmp_path,
):
    parent, v3_manifest, v4_manifest, receipt = _parent(tmp_path)
    before_summary = (parent / "summary.json").read_bytes()
    before_receipt = receipt.read_bytes()

    lineage = load_round5_parent(
        parent,
        expected_parent_hash=v4_manifest.content_hash,
        expected_rollback_hash=v3_manifest.content_hash,
        campaign_budget_receipt=receipt,
    )

    assert lineage.parent_run_id == "parent-v4"
    assert lineage.parent_version == "v4"
    assert lineage.starting_version == "v3"
    assert lineage.global_act_count == 5
    assert lineage.prior_score_history == (
        0.0,
        0.0,
        None,
        0.0,
        7 / 18,
        4 / 18,
    )
    assert lineage.learning_budget["learning_episodes"] == 58
    assert (parent / "summary.json").read_bytes() == before_summary
    assert receipt.read_bytes() == before_receipt


def test_round5_parent_rejects_receipt_summary_hash_tampering(tmp_path):
    parent, v3_manifest, v4_manifest, receipt = _parent(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["success_summary_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="summary hash"):
        load_round5_parent(
            parent,
            v4_manifest.content_hash,
            v3_manifest.content_hash,
            receipt,
        )


def test_round5_parent_rejects_snapshot_or_protected_file_tampering(
    tmp_path,
):
    parent, v3_manifest, v4_manifest, receipt = _parent(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v4"):
        load_round5_parent(
            parent,
            "0" * 64,
            v3_manifest.content_hash,
            receipt,
        )

    (parent / "versions" / "v4" / "source" / "main.py").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        load_round5_parent(
            parent,
            v4_manifest.content_hash,
            v3_manifest.content_hash,
            receipt,
        )


def test_import_round5_sources_starts_workspace_from_v3_and_keeps_v4(
    tmp_path,
):
    parent, v3_manifest, v4_manifest, receipt = _parent(tmp_path)
    lineage = load_round5_parent(
        parent,
        v4_manifest.content_hash,
        v3_manifest.content_hash,
        receipt,
    )
    run_dir = tmp_path / "child"

    imported_v3, imported_v4 = import_round5_sources(
        lineage,
        run_dir,
        LocalWorkspaceSnapshotter(),
    )

    assert imported_v3.content_hash == v3_manifest.content_hash
    assert imported_v4.content_hash == v4_manifest.content_hash
    assert (run_dir / "workspace" / "strategy.py").read_text().endswith(
        "# v3\n"
    )
    assert (
        run_dir / "versions" / "v4" / "source" / "strategy.py"
    ).read_text().endswith("# v4\n")
    rollback = json.loads(
        (run_dir / "versions" / "rollback.json").read_text()
    )
    assert rollback == {
        "parent_content_hash": v4_manifest.content_hash,
        "parent_run_id": "parent-v4",
        "parent_version": "v4",
        "rollback_content_hash": v3_manifest.content_hash,
        "rollback_source_version": "v3",
        "starting_version": "v3",
    }
