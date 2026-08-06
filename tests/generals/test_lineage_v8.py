import json
from pathlib import Path

import pytest

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


def _module():
    from agentbench_frame.generals import lineage_v8

    return lineage_v8


def _parent(tmp_path: Path):
    parent = tmp_path / "parent-v7"
    source = parent / "versions/v7/source"
    (source / "tests").mkdir(parents=True)
    (source / "main.py").write_text("V7 = True\n", encoding="utf-8")
    (source / "state_view.py").write_text("def normalize(x): return x\n", encoding="utf-8")
    (source / "strategy.py").write_text("def choose_actions(*args): return [[8]]\n", encoding="utf-8")
    (source / "STRATEGY.md").write_text("v7 strategy\n", encoding="utf-8")
    (source / "EXPERIENCE.md").write_text("v7 experience\n", encoding="utf-8")
    (source / "tests/test_strategy.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(manifest, parent / "versions/v7/manifest.json")
    summary = {
        "run_id": "parent-v7",
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
        "evo_score_7": 12 / 18,
        "score_history": [0.0, 0.0, None, 0.0, 7 / 18, 4 / 18, 7 / 18, 12 / 18, 12 / 18],
        "act_count": 8,
        "cumulative_learning_budget": {
            "learning_coding_agent_acts": 8,
            "learning_episodes": 94,
            "learning_env_steps": 43034,
        },
    }
    (parent / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return parent, manifest


def test_round8_parent_is_exact_complete_v7(tmp_path):
    parent, manifest = _parent(tmp_path)
    before = (parent / "summary.json").read_bytes()

    lineage = _module().load_round8_parent(parent, manifest.content_hash)

    assert lineage.parent_version == "v7"
    assert lineage.global_act_count == 8
    assert lineage.evo_score_7 == 12 / 18
    assert lineage.prior_score_history[-1] == 12 / 18
    assert lineage.v7_manifest.content_hash == manifest.content_hash
    assert (parent / "summary.json").read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "must be complete"),
        ("formal_attempted", False, "formal evaluation"),
        ("evo_score_7", None, "complete v0 through v7"),
        ("act_count", 7, "act count"),
        ("act_count", 8.0, "act count"),
        ("cumulative_learning_budget", {}, "learning budget"),
    ],
)
def test_round8_parent_rejects_altered_lineage(tmp_path, field, value, message):
    parent, manifest = _parent(tmp_path)
    path = parent / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[field] = value
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _module().load_round8_parent(parent, manifest.content_hash)


def test_round8_parent_rejects_hash_or_source_change(tmp_path):
    parent, manifest = _parent(tmp_path)
    with pytest.raises(ValueError, match="unexpected parent v7"):
        _module().load_round8_parent(parent, "0" * 64)
    (parent / "versions/v7/source/strategy.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match source"):
        _module().load_round8_parent(parent, manifest.content_hash)


def test_import_round8_source_copies_v7_and_records_lineage(tmp_path):
    parent, manifest = _parent(tmp_path)
    lineage = _module().load_round8_parent(parent, manifest.content_hash)
    child = tmp_path / "child"

    imported = _module().import_round8_source(lineage, child, LocalWorkspaceSnapshotter())

    assert imported.content_hash == manifest.content_hash
    assert (child / "workspace/.git").is_dir()
    assert (child / "versions/v7/source/EXPERIENCE.md").read_text() == "v7 experience\n"
    record = json.loads((child / "versions/lineage.json").read_text())
    assert record["parent_version"] == "v7"
    assert record["starting_version"] == "v7"
    assert record["global_act_count"] == 8
