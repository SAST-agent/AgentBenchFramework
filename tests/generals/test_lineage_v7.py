import json
from pathlib import Path

import pytest

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _module():
    from agentbench_frame.generals import lineage_v7

    return lineage_v7


def _write_source(path: Path) -> None:
    (path / "tests").mkdir(parents=True)
    (path / "policy").mkdir()
    (path / "main.py").write_text("V6 = True\n", encoding="utf-8")
    (path / "state_view.py").write_text(
        "def normalize(value): return value\n",
        encoding="utf-8",
    )
    (path / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[8]]\n",
        encoding="utf-8",
    )
    (path / "policy" / "planner.py").write_text(
        "MAX_MACRO = 4\n",
        encoding="utf-8",
    )
    (path / "STRATEGY.md").write_text("v6 strategy\n", encoding="utf-8")
    (path / "EXPERIENCE.md").write_text("v6 experience\n", encoding="utf-8")
    (path / "tests" / "test_strategy.py").write_text(
        "def test_strategy_contract(): assert True\n",
        encoding="utf-8",
    )


def _parent(tmp_path: Path):
    parent = tmp_path / "parent-v6"
    source = parent / "versions" / "v6" / "source"
    _write_source(source)
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        parent / "versions" / "v6" / "manifest.json",
    )
    (parent / "summary.json").write_text(
        json.dumps({
            "run_id": "parent-v6",
            "status": "complete",
            "evaluation_status": "complete",
            "formal_attempted": True,
            "raw_score": 0.0,
            "evo_score_1": 0.0,
            "evo_score_2": 0.0,
            "evo_score_3": 7 / 18,
            "evo_score_4": 4 / 18,
            "evo_score_5": 7 / 18,
            "evo_score_6": 12 / 18,
            "score_history": [
                0.0,
                0.0,
                None,
                0.0,
                7 / 18,
                4 / 18,
                7 / 18,
                12 / 18,
            ],
            "act_count": 7,
            "cumulative_learning_budget": {
                "learning_coding_agent_acts": 7,
                "learning_episodes": 88,
                "learning_env_steps": 40857,
            },
        }),
        encoding="utf-8",
    )
    return parent, manifest


def test_round7_parent_loads_exact_complete_v6_without_mutation(tmp_path):
    parent, manifest = _parent(tmp_path)
    summary_bytes = (parent / "summary.json").read_bytes()
    manifest_bytes = (
        parent / "versions" / "v6" / "manifest.json"
    ).read_bytes()

    lineage = _module().load_round7_parent(parent, manifest.content_hash)

    assert lineage.parent_version == "v6"
    assert lineage.global_act_count == 7
    assert lineage.evo_score_6 == 12 / 18
    assert lineage.prior_score_history[-1] == 12 / 18
    assert lineage.learning_budget["learning_coding_agent_acts"] == 7
    assert lineage.v6_manifest.content_hash == manifest.content_hash
    assert (parent / "summary.json").read_bytes() == summary_bytes
    assert (
        parent / "versions" / "v6" / "manifest.json"
    ).read_bytes() == manifest_bytes


def test_round7_parent_rejects_wrong_hash_and_changed_source(tmp_path):
    parent, manifest = _parent(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v6"):
        _module().load_round7_parent(parent, "0" * 64)

    (parent / "versions" / "v6" / "source" / "strategy.py").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match source"):
        _module().load_round7_parent(parent, manifest.content_hash)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "must be complete"),
        ("evaluation_status", "incomplete", "formal evaluation"),
        ("formal_attempted", False, "formal evaluation"),
        ("evo_score_6", None, "complete v0 through v6 scores"),
        (
            "score_history",
            [0.0, 0.0, 0.0, 0.0, 7 / 18, 4 / 18, 7 / 18, 12 / 18],
            "score history",
        ),
        ("act_count", 6, "act count"),
        ("act_count", 7.0, "act count"),
        ("cumulative_learning_budget", {}, "learning budget"),
        (
            "cumulative_learning_budget",
            {"learning_coding_agent_acts": 6},
            "learning act count",
        ),
    ],
)
def test_round7_parent_rejects_incomplete_or_altered_lineage(
    tmp_path,
    field,
    value,
    message,
):
    parent, manifest = _parent(tmp_path)
    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _module().load_round7_parent(parent, manifest.content_hash)


def test_round7_parent_rejects_non_object_summary(tmp_path):
    parent, manifest = _parent(tmp_path)
    (parent / "summary.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="summary"):
        _module().load_round7_parent(parent, manifest.content_hash)


def test_import_round7_source_copies_exact_v6_and_records_full_lineage(tmp_path):
    parent, manifest = _parent(tmp_path)
    lineage = _module().load_round7_parent(parent, manifest.content_hash)
    run_dir = tmp_path / "child"

    imported = _module().import_round7_source(
        lineage,
        run_dir,
        LocalWorkspaceSnapshotter(),
    )

    assert imported.content_hash == manifest.content_hash
    assert (run_dir / "workspace" / ".git").is_dir()
    assert (
        run_dir / "versions" / "v6" / "source" / "EXPERIENCE.md"
    ).read_text(encoding="utf-8") == "v6 experience\n"
    assert json.loads(
        (run_dir / "versions" / "lineage.json").read_text(encoding="utf-8")
    ) == {
        "expected_parent_hash": manifest.content_hash,
        "global_act_count": 7,
        "inherited_learning_budget": {
            "learning_coding_agent_acts": 7,
            "learning_env_steps": 40857,
            "learning_episodes": 88,
        },
        "observed_parent_hash": manifest.content_hash,
        "parent_run_id": "parent-v6",
        "parent_version": "v6",
        "prior_score_history": [
            0.0,
            0.0,
            None,
            0.0,
            7 / 18,
            4 / 18,
            7 / 18,
            12 / 18,
        ],
        "starting_version": "v6",
    }
