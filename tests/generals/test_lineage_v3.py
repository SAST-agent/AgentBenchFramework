import json

import pytest

from agentbench_frame.generals.lineage_v3 import (
    import_parent_v2,
    load_round3_parent,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def parent_run(tmp_path, status="complete", recovery=True):
    parent = tmp_path / "20260727_parent"
    source = parent / "versions" / "v2" / "source"
    (source / "tests").mkdir(parents=True)
    (source / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view): return [[8]]\n",
        encoding="utf-8",
    )
    (source / "STRATEGY.md").write_text("v2", encoding="utf-8")
    (source / "main.py").write_text("", encoding="utf-8")
    (source / "state_view.py").write_text("", encoding="utf-8")
    (source / "tests" / "test_strategy.py").write_text(
        "from strategy import choose_actions\n"
        "def test_end(): assert choose_actions(1, 0, {}) == [[8]]\n",
        encoding="utf-8",
    )
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        parent / "versions" / "v2" / "manifest.json",
    )
    (parent / "versions" / "v1-to-v2.patch").write_text(
        "v2 diff",
        encoding="utf-8",
    )
    summary = {
        "run_id": "parent-v2",
        "status": status,
        "raw_score": 0.0,
        "evo_score_1": 0.0,
        "evo_score_2": 0.0,
        "act_count": 2,
        "round_act_count": 1,
        "budget": {
            "learning_coding_agent_acts": 1,
            "learning_episodes": 0,
            "learning_env_steps": 0,
            "learning_game_agent_decision_steps": 0,
            "learning_primitive_commands": 0,
            "learning_total_tokens": 150,
            "learning_time_s": 1.0,
        },
    }
    if recovery:
        summary["recovery_from_run_id"] = "failed-v2-run"
        summary["source_learning_budget"] = {
            "learning_coding_agent_acts": 1,
            "learning_episodes": 10,
            "learning_env_steps": 20,
            "learning_game_agent_decision_steps": 10,
            "learning_primitive_commands": 15,
            "learning_total_tokens": None,
            "learning_time_s": 2.0,
        }
        failed = tmp_path / "failed-v2-run"
        failed.mkdir()
        (failed / "summary.json").write_text(
            json.dumps({
                "run_id": "failed-v2-run",
                "status": "provider_failed",
                "budget": summary["source_learning_budget"],
                "parent_learning_budget": {
                    "learning_coding_agent_acts": 1,
                    "learning_episodes": 12,
                    "learning_env_steps": 24,
                    "learning_game_agent_decision_steps": 12,
                    "learning_primitive_commands": 24,
                    "learning_total_tokens": 300,
                    "learning_time_s": 4.0,
                },
            }),
            encoding="utf-8",
        )
    (parent / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return parent, manifest


def test_load_round3_parent_verifies_complete_v2_and_recovery_gap(tmp_path):
    parent, manifest = parent_run(tmp_path)

    lineage = load_round3_parent(parent, manifest.content_hash)

    assert lineage.parent_run_id == "parent-v2"
    assert lineage.parent_version == "v2"
    assert lineage.raw_score == 0.0
    assert lineage.evo_score_1 == 0.0
    assert lineage.evo_score_2 == 0.0
    assert lineage.global_act_count == 3
    assert lineage.prior_score_history == (0.0, 0.0, None, 0.0)
    assert lineage.learning_budget["learning_episodes"] == 22
    assert lineage.learning_budget["learning_env_steps"] == 44
    assert lineage.learning_budget["learning_coding_agent_acts"] == 3
    assert lineage.learning_budget["learning_total_tokens"] is None
    assert lineage.learning_budget["learning_time_s"] == 7.0


def test_import_parent_v2_preserves_exact_source_and_hash(tmp_path):
    parent, manifest = parent_run(tmp_path)
    lineage = load_round3_parent(parent, manifest.content_hash)

    imported = import_parent_v2(
        lineage,
        tmp_path / "child",
        LocalWorkspaceSnapshotter(),
    )

    assert imported.content_hash == manifest.content_hash
    assert (
        tmp_path / "child" / "versions" / "v2" / "lineage.json"
    ).is_file()
    assert (tmp_path / "child" / "workspace" / ".git").is_dir()
    assert (
        tmp_path / "child" / "workspace" / "strategy.py"
    ).read_bytes() == (
        parent / "versions" / "v2" / "source" / "strategy.py"
    ).read_bytes()


def test_round3_parent_rejects_non_complete_run(tmp_path):
    parent, manifest = parent_run(tmp_path, status="failed")

    with pytest.raises(ValueError, match="complete"):
        load_round3_parent(parent, manifest.content_hash)


def test_round3_parent_rejects_missing_v2_score(tmp_path):
    parent, manifest = parent_run(tmp_path)
    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evo_score_2"] = None
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="v0, v1, and v2 scores"):
        load_round3_parent(parent, manifest.content_hash)


def test_round3_parent_rejects_expected_hash_mismatch(tmp_path):
    parent, _manifest = parent_run(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v2 content hash"):
        load_round3_parent(parent, "0" * 64)


def test_round3_parent_rejects_tampered_v2_source(tmp_path):
    parent, manifest = parent_run(tmp_path)
    (parent / "versions" / "v2" / "source" / "strategy.py").write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest does not match"):
        load_round3_parent(parent, manifest.content_hash)
