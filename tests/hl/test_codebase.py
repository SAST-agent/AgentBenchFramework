from pathlib import Path

import pytest


def _workspace(root: Path) -> Path:
    workspace = root / "candidate"
    workspace.mkdir()
    (workspace / "agent.py").write_text("def act(state):\n    return 0\n", encoding="utf-8")
    (workspace / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    return workspace


def test_snapshots_are_immutable_logical_versions(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore

    workspace = _workspace(tmp_path)
    store = VersionStore(workspace, tmp_path / "versions")
    v0 = store.snapshot(parent_version_id=None, act_id="act-0000")
    v1 = store.snapshot(parent_version_id=v0.version_id, act_id="act-0001")

    assert v0.version_id != v1.version_id
    assert v0.content_hash == v1.content_hash
    assert v1.parent_version_id == v0.version_id
    assert store.get(v0.version_id) == v0
    assert (tmp_path / "versions" / "manifests" / f"{v0.version_id}.json").exists()


def test_snapshot_ignores_run_artifacts_but_keeps_candidate_source(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore

    workspace = _workspace(tmp_path)
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "agent.pyc").write_bytes(b"cache")
    (workspace / ".agentbench").mkdir()
    (workspace / ".agentbench" / "events.jsonl").write_text("{}\n", encoding="utf-8")
    version = VersionStore(workspace, tmp_path / "versions").snapshot(
        parent_version_id=None,
        act_id="act-0000",
    )
    snapshot = tmp_path / "versions" / "objects" / version.content_hash

    assert (snapshot / "agent.py").exists()
    assert not (snapshot / "__pycache__").exists()
    assert not (snapshot / ".agentbench").exists()


def test_restore_historical_version_creates_new_rollback_version(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore

    workspace = _workspace(tmp_path)
    store = VersionStore(workspace, tmp_path / "versions")
    v0 = store.snapshot(parent_version_id=None, act_id="act-0000")
    (workspace / "agent.py").write_text("def act(state):\n    return 4\n", encoding="utf-8")
    v1 = store.snapshot(parent_version_id=v0.version_id, act_id="act-0001")

    rollback = store.restore(
        source_version_id=v0.version_id,
        parent_version_id=v1.version_id,
        act_id="act-0002",
    )

    assert rollback.version_id not in {v0.version_id, v1.version_id}
    assert rollback.content_hash == v0.content_hash
    assert rollback.edit_type == "rollback"
    assert (workspace / "agent.py").read_text(encoding="utf-8").endswith("return 0\n")
    assert store.get(v1.version_id).content_hash != rollback.content_hash


def test_missing_or_unreadable_workspace_fails_instead_of_fabricating_version(tmp_path):
    from agentbench_frame.hl.codebase import VersionStore

    store = VersionStore(tmp_path / "missing", tmp_path / "versions")
    with pytest.raises(FileNotFoundError):
        store.snapshot(parent_version_id=None, act_id="act-0000")

