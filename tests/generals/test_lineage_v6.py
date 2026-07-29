import json
from pathlib import Path

import pytest

from agentbench_frame.generals.lineage_v6 import (
    import_round6_source,
    load_round6_parent,
)
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _write_source(path: Path) -> None:
    (path / "tests").mkdir(parents=True)
    (path / "main.py").write_text("V5 = True\n", encoding="utf-8")
    (path / "state_view.py").write_text(
        "def normalize(value): return value\n",
        encoding="utf-8",
    )
    (path / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[8]]\n",
        encoding="utf-8",
    )
    (path / "STRATEGY.md").write_text("v5 strategy\n", encoding="utf-8")
    (path / "EXPERIENCE.md").write_text("v5 experience\n", encoding="utf-8")
    (path / "tests" / "test_strategy.py").write_text(
        "def test_strategy_contract(): assert True\n",
        encoding="utf-8",
    )


def _parent(tmp_path: Path):
    parent = tmp_path / "parent-v5"
    source = parent / "versions" / "v5" / "source"
    _write_source(source)
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        parent / "versions" / "v5" / "manifest.json",
    )
    (parent / "summary.json").write_text(
        json.dumps({
            "run_id": "parent-v5",
            "status": "complete",
            "evaluation_status": "complete",
            "raw_score": 0.0,
            "evo_score_1": 0.0,
            "evo_score_2": 0.0,
            "evo_score_3": 7 / 18,
            "evo_score_4": 4 / 18,
            "evo_score_5": 7 / 18,
            "score_history": [
                0.0,
                0.0,
                None,
                0.0,
                7 / 18,
                4 / 18,
                7 / 18,
            ],
            "act_count": 6,
            "cumulative_learning_budget": {
                "learning_coding_agent_acts": 6,
                "learning_episodes": 64,
            },
        }),
        encoding="utf-8",
    )
    return parent, manifest


def test_round6_parent_loads_exact_complete_v5_without_mutating_it(tmp_path):
    parent, v5_manifest = _parent(tmp_path)
    before_summary = (parent / "summary.json").read_bytes()
    before_manifest = (
        parent / "versions" / "v5" / "manifest.json"
    ).read_bytes()

    lineage = load_round6_parent(parent, v5_manifest.content_hash)

    assert lineage.parent_version == "v5"
    assert lineage.global_act_count == 6
    assert lineage.evo_score_5 == 7 / 18
    assert lineage.prior_score_history[-1] == 7 / 18
    assert lineage.learning_budget["learning_coding_agent_acts"] == 6
    assert (parent / "summary.json").read_bytes() == before_summary
    assert (
        parent / "versions" / "v5" / "manifest.json"
    ).read_bytes() == before_manifest


def test_round6_parent_rejects_wrong_hash_or_modified_v5_source(tmp_path):
    parent, v5_manifest = _parent(tmp_path)

    with pytest.raises(ValueError, match="unexpected parent v5"):
        load_round6_parent(parent, "0" * 64)

    (parent / "versions" / "v5" / "source" / "strategy.py").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match source"):
        load_round6_parent(parent, v5_manifest.content_hash)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_status", "incomplete", "formal evaluation"),
        ("evo_score_5", None, "complete v0 through v5 scores"),
        (
            "score_history",
            [0.0, 0.0, 0.0, 0.0, 7 / 18, 4 / 18, 7 / 18],
            "score history",
        ),
        ("act_count", 5, "act count"),
        ("act_count", 6.5, "act count"),
        (
            "cumulative_learning_budget",
            {"learning_coding_agent_acts": 6.5},
            "learning act count",
        ),
    ],
)
def test_round6_parent_rejects_incomplete_or_altered_lineage(
    tmp_path,
    field,
    value,
    message,
):
    parent, v5_manifest = _parent(tmp_path)
    summary_path = parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_round6_parent(parent, v5_manifest.content_hash)


def test_round6_parent_rejects_non_object_summary(tmp_path):
    parent, v5_manifest = _parent(tmp_path)
    (parent / "summary.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="summary"):
        load_round6_parent(parent, v5_manifest.content_hash)


def test_import_round6_source_copies_exact_v5_and_writes_lineage(tmp_path):
    parent, v5_manifest = _parent(tmp_path)
    lineage = load_round6_parent(parent, v5_manifest.content_hash)
    run_dir = tmp_path / "child"

    imported = import_round6_source(
        lineage,
        run_dir,
        LocalWorkspaceSnapshotter(),
    )

    assert imported.content_hash == v5_manifest.content_hash
    assert (run_dir / "workspace" / ".git").is_dir()
    assert (
        run_dir / "workspace" / "EXPERIENCE.md"
    ).read_text() == "v5 experience\n"
    assert (
        run_dir / "versions" / "v5" / "source" / "strategy.py"
    ).read_text() == (
        parent / "versions" / "v5" / "source" / "strategy.py"
    ).read_text()
    assert json.loads(
        (run_dir / "versions" / "v5" / "manifest.json").read_text()
    ) == json.loads(
        (parent / "versions" / "v5" / "manifest.json").read_text()
    )
    assert json.loads((run_dir / "versions" / "lineage.json").read_text()) == {
        "parent_content_hash": v5_manifest.content_hash,
        "parent_run_id": "parent-v5",
        "parent_version": "v5",
        "starting_version": "v5",
    }
