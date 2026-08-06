import json
from pathlib import Path

import pytest

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


V7_HISTORY = [
    0.0,
    0.0,
    None,
    0.0,
    7 / 18,
    4 / 18,
    7 / 18,
    12 / 18,
    12 / 18,
]
V8_HISTORY = [*V7_HISTORY, 7 / 18]


def _source(root: Path, version: str, marker: str):
    source = root / f"versions/{version}/source"
    (source / "tests").mkdir(parents=True)
    (source / "main.py").write_text("def agent(*args): return [[8]]\n", encoding="utf-8")
    (source / "state_view.py").write_text("def normalize(x): return x\n", encoding="utf-8")
    (source / "strategy.py").write_text(
        f"MARKER = {marker!r}\ndef choose_actions(*args): return [[8]]\n",
        encoding="utf-8",
    )
    (source / "STRATEGY.md").write_text(f"{version} strategy\n", encoding="utf-8")
    (source / "EXPERIENCE.md").write_text(f"{version} experience\n", encoding="utf-8")
    (source / "tests/test_strategy.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(manifest, root / f"versions/{version}/manifest.json")
    return source, manifest


def _authorities(tmp_path: Path):
    v7 = tmp_path / "20260730_1739_680b1632"
    v7_source, v7_manifest = _source(v7, "v7", "source-parent")
    (v7 / "summary.json").write_text(
        json.dumps({
            "run_id": v7.name,
            "status": "complete",
            "evaluation_status": "complete",
            "formal_attempted": True,
            "runnable": True,
            "act_count": 8,
            "score_history": V7_HISTORY,
            "evo_score_7": 12 / 18,
            "cumulative_learning_budget": {
                "learning_coding_agent_acts": 8,
                "learning_episodes": 94,
            },
        }),
        encoding="utf-8",
    )

    v8 = tmp_path / "20260806_1126_8e1b6795"
    _, v8_manifest = _source(v8, "v8", "chronological-predecessor")
    (v8 / "summary.json").write_text(
        json.dumps({
            "run_id": v8.name,
            "status": "complete",
            "evaluation_status": "complete",
            "formal_attempted": True,
            "runnable": True,
            "act_count": 9,
            "score_history": V8_HISTORY,
            "evo_score_8": 7 / 18,
            "parent_run_id": v7.name,
            "parent_version": "v7",
            "parent_content_hash": v7_manifest.content_hash,
        }),
        encoding="utf-8",
    )
    return v7, v8, v7_source, v7_manifest, v8_manifest


def test_round9_lineage_separates_source_parent_from_predecessor(tmp_path):
    from agentbench_frame.generals.lineage_v9 import load_round9_lineage

    v7, v8, _, v7_manifest, v8_manifest = _authorities(tmp_path)

    lineage = load_round9_lineage(v7, v8)

    assert lineage.policy_parent_version == "v7"
    assert lineage.policy_parent_hash == v7_manifest.content_hash
    assert lineage.iteration_predecessor_version == "v8"
    assert lineage.iteration_predecessor_hash == v8_manifest.content_hash
    assert lineage.next_global_act_count == 10
    assert lineage.score_history == tuple(V8_HISTORY)


@pytest.mark.parametrize(
    ("authority", "field", "value", "message"),
    [
        ("v7", "status", "failed", "v7 source-parent run must be complete"),
        ("v8", "runnable", False, "v8 predecessor must be runnable"),
        ("v8", "act_count", 8, "predecessor act count"),
        ("v8", "parent_version", "v6", "v8 predecessor parent"),
        ("v8", "score_history", V7_HISTORY, "score history"),
    ],
)
def test_round9_lineage_rejects_altered_authority(
    tmp_path, authority, field, value, message
):
    from agentbench_frame.generals.lineage_v9 import load_round9_lineage

    v7, v8, _, _, _ = _authorities(tmp_path)
    selected = v7 if authority == "v7" else v8
    summary_path = selected / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_round9_lineage(v7, v8)


def test_round9_lineage_rejects_source_tree_drift(tmp_path):
    from agentbench_frame.generals.lineage_v9 import load_round9_lineage

    v7, v8, v7_source, _, _ = _authorities(tmp_path)
    (v7_source / "strategy.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v7 manifest does not match source"):
        load_round9_lineage(v7, v8)


def test_import_round9_parent_materializes_only_v7_and_records_both_roles(tmp_path):
    from agentbench_frame.generals.lineage_v9 import (
        import_round9_parent,
        load_round9_lineage,
    )

    v7, v8, _, v7_manifest, _ = _authorities(tmp_path)
    lineage = load_round9_lineage(v7, v8)
    child = tmp_path / "child"

    imported = import_round9_parent(
        lineage,
        child,
        LocalWorkspaceSnapshotter(),
    )

    assert imported.content_hash == v7_manifest.content_hash
    assert (child / "workspace/.git").is_dir()
    assert (child / "versions/v7/source/EXPERIENCE.md").is_file()
    assert not (child / "versions/v8/source").exists()
    record = json.loads((child / "versions/lineage.json").read_text())
    assert record["policy_parent_version"] == "v7"
    assert record["iteration_predecessor_version"] == "v8"
    assert record["next_version"] == "v9"
    assert record["next_global_act_count"] == 10
