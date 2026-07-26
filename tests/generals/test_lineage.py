import json
from pathlib import Path

import pytest

from agentbench_frame.generals.lineage import (
    import_parent_v1,
    load_parent_lineage,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def parent_run(tmp_path, status="complete"):
    parent = tmp_path / "parent"
    v0 = parent / "versions" / "v0" / "source"
    v1 = parent / "versions" / "v1" / "source"
    v0.mkdir(parents=True)
    v1.mkdir(parents=True)
    for root, policy in ((v0, "0"), (v1, "1")):
        (root / "strategy.py").write_text(
            f"POLICY = {policy}\n"
            "def choose_actions(round_number, my_seat, view): return [[8]]\n"
        )
        (root / "main.py").write_text("")
        (root / "STRATEGY.md").write_text(f"policy {policy}")
        (root / "tests").mkdir()
        (root / "tests" / "test_strategy.py").write_text(
            "from strategy import choose_actions\n"
            "def test_end(): assert choose_actions(1, 0, {}) == [[8]]\n"
        )
    snapshotter = LocalWorkspaceSnapshotter()
    v0_manifest = snapshotter.capture(v0)
    v1_manifest = snapshotter.capture(v1, previous=v0_manifest)
    snapshotter.write_manifest(
        v0_manifest, parent / "versions" / "v0" / "manifest.json"
    )
    snapshotter.write_manifest(
        v1_manifest, parent / "versions" / "v1" / "manifest.json"
    )
    (parent / "versions" / "v0-to-v1.patch").write_text("diff")
    (parent / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "parent-1",
                "status": status,
                "raw_score": 0.0,
                "evo_score": 0.0,
                "act_count": 1,
                "budget": {
                    "learning_coding_agent_acts": 1,
                    "learning_episodes": 12,
                    "learning_env_steps": 120,
                    "learning_game_agent_decision_steps": 60,
                    "learning_primitive_commands": 90,
                    "learning_total_tokens": 1000,
                    "learning_time_s": 10.0,
                },
            }
        )
    )
    return parent, v1_manifest


def test_load_parent_lineage_verifies_completed_v1_content(tmp_path):
    parent, manifest = parent_run(tmp_path)

    lineage = load_parent_lineage(parent, expected_hash=manifest.content_hash)

    assert lineage.parent_run_id == "parent-1"
    assert lineage.parent_version == "v1"
    assert lineage.raw_score == 0.0
    assert lineage.evo_score_1 == 0.0
    assert lineage.v1_manifest.content_hash == manifest.content_hash
    assert lineage.learning_budget["learning_episodes"] == 12
    assert lineage.first_diff == "diff"


def test_import_parent_v1_copies_exact_source_into_new_run(tmp_path):
    parent, manifest = parent_run(tmp_path)
    lineage = load_parent_lineage(parent, expected_hash=manifest.content_hash)
    child = tmp_path / "child"

    imported = import_parent_v1(
        lineage, child, LocalWorkspaceSnapshotter()
    )

    assert imported.content_hash == manifest.content_hash
    assert (child / "workspace" / ".git").is_dir()
    assert (
        child / "workspace" / "strategy.py"
    ).read_bytes() == (
        parent / "versions" / "v1" / "source" / "strategy.py"
    ).read_bytes()
    assert (
        child / "versions" / "v1" / "manifest.json"
    ).is_file()


def test_parent_lineage_rejects_non_complete_run(tmp_path):
    parent, manifest = parent_run(tmp_path, status="failed")

    with pytest.raises(ValueError, match="parent run must be complete"):
        load_parent_lineage(parent, expected_hash=manifest.content_hash)


def test_parent_lineage_rejects_expected_hash_mismatch(tmp_path):
    parent, _manifest = parent_run(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v1 content hash"):
        load_parent_lineage(parent, expected_hash="0" * 64)


def test_parent_lineage_rejects_tampered_v1_source(tmp_path):
    parent, manifest = parent_run(tmp_path)
    (parent / "versions" / "v1" / "source" / "strategy.py").write_text(
        "tampered"
    )

    with pytest.raises(ValueError, match="parent v1 manifest does not match source"):
        load_parent_lineage(parent, expected_hash=manifest.content_hash)

