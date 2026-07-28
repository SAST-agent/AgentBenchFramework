import json

import pytest

from agentbench_frame.generals.lineage_v4 import (
    import_parent_v3,
    load_round4_parent,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _parent(tmp_path):
    parent = tmp_path / "parent-v3"
    source = parent / "versions" / "v3" / "source"
    (source / "tests").mkdir(parents=True)
    (source / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[8]]\n",
        encoding="utf-8",
    )
    (source / "STRATEGY.md").write_text("v3 strategy", encoding="utf-8")
    (source / "EXPERIENCE.md").write_text("v3 experience", encoding="utf-8")
    (source / "main.py").write_text("", encoding="utf-8")
    (source / "state_view.py").write_text("", encoding="utf-8")
    (source / "tests" / "test_strategy.py").write_text(
        "def test_placeholder(): assert True\n",
        encoding="utf-8",
    )
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        parent / "versions" / "v3" / "manifest.json",
    )
    (parent / "summary.json").write_text(
        json.dumps({
            "run_id": "parent-v3",
            "status": "complete",
            "evaluation_status": "complete",
            "raw_score": 0.0,
            "evo_score_1": 0.0,
            "evo_score_2": 0.0,
            "evo_score_3": 7 / 18,
            "score_history": [0.0, 0.0, None, 0.0, 7 / 18],
            "act_count": 4,
            "cumulative_learning_budget": {
                "learning_coding_agent_acts": 4,
                "learning_episodes": 46,
                "learning_env_steps": 1000,
                "learning_game_agent_decision_steps": 500,
                "learning_primitive_commands": 2000,
                "learning_prompt_tokens": None,
                "learning_completion_tokens": None,
                "learning_total_tokens": None,
                "learning_time_s": 50.0,
            },
        }),
        encoding="utf-8",
    )
    return parent, manifest


def test_round4_parent_loads_exact_complete_v3_without_mutating_it(tmp_path):
    parent, manifest = _parent(tmp_path)
    before_summary = (parent / "summary.json").read_bytes()
    before_manifest = (
        parent / "versions" / "v3" / "manifest.json"
    ).read_bytes()

    lineage = load_round4_parent(parent, manifest.content_hash)

    assert lineage.parent_run_id == "parent-v3"
    assert lineage.parent_version == "v3"
    assert lineage.raw_score == 0.0
    assert lineage.evo_score_3 == 7 / 18
    assert lineage.global_act_count == 4
    assert lineage.prior_score_history == (
        0.0,
        0.0,
        None,
        0.0,
        7 / 18,
    )
    assert lineage.learning_budget["learning_episodes"] == 46
    assert (parent / "summary.json").read_bytes() == before_summary
    assert (
        parent / "versions" / "v3" / "manifest.json"
    ).read_bytes() == before_manifest


def test_round4_parent_rejects_hash_or_source_tampering(tmp_path):
    parent, manifest = _parent(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v3"):
        load_round4_parent(parent, "0" * 64)

    (parent / "versions" / "v3" / "source" / "strategy.py").write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match source"):
        load_round4_parent(parent, manifest.content_hash)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "must be complete"),
        ("evaluation_status", "incomplete", "formal evaluation"),
        ("evo_score_3", None, "complete v0 through v3 scores"),
        ("score_history", [0.0, 0.0, 0.0], "score history"),
        ("act_count", 3, "act count"),
    ],
)
def test_round4_parent_rejects_incomplete_lineage(
    tmp_path,
    field,
    value,
    message,
):
    parent, manifest = _parent(tmp_path)
    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_round4_parent(parent, manifest.content_hash)


def test_import_parent_v3_copies_exact_source_and_lineage_receipt(tmp_path):
    parent, manifest = _parent(tmp_path)
    lineage = load_round4_parent(parent, manifest.content_hash)
    run_dir = tmp_path / "child"

    imported = import_parent_v3(
        lineage,
        run_dir,
        LocalWorkspaceSnapshotter(),
    )

    assert imported.content_hash == manifest.content_hash
    assert (
        run_dir / "workspace" / "EXPERIENCE.md"
    ).read_text() == "v3 experience"
    assert (
        run_dir / "versions" / "v3" / "source" / "strategy.py"
    ).read_text() == (
        parent / "versions" / "v3" / "source" / "strategy.py"
    ).read_text()
    receipt = json.loads(
        (run_dir / "versions" / "v3" / "lineage.json").read_text()
    )
    assert receipt == {
        "content_hash": manifest.content_hash,
        "global_act_count": 4,
        "parent_run_id": "parent-v3",
        "parent_version": "v3",
    }
