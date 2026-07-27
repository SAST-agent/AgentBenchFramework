"""Tests for hl/codebase.py — the versioned, agent-editable HL codebase.

Contract (plan §codebase):
- snapshot() content-hashes the workspace tree, copies into a version store,
  returns a VersionHandle (version_id, content_hash, parent_version_id).
- Identical content across versions is ALLOWED: two snapshots of the same
  tree get different version_ids but the SAME content_hash.
- restore(content_hash) copies a prior snapshot back into the workspace and
  returns a NEW version_id with edit_type='rollback' (zero policy-KL, nonzero
  act budget — recorded, not folded).
- diff(before, after) -> structured file-level diff.
- Crash-safe: snapshot of a partially-edited tree still succeeds; an
  unreadable workspace yields version_after=None but no exception thrown by
  the controller (snapshot itself just records what's readable).
- manifest.toml is loaded as part of the tree (manifest.py).
"""
import os
from pathlib import Path

import pytest

from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.manifest import HLManifest, load_manifest


def _write(tree: Path, rel: str, content: str):
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    _write(ws, "agent.py", "print('v0')\n")
    _write(ws, "manifest.toml",
           'shape = "single_file"\nentrypoint = "agent.py"\n')
    return ws


def test_snapshot_returns_handle_with_hash_and_parent(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    assert isinstance(h, VersionHandle)
    assert h.content_hash
    assert h.version_id != h.content_hash  # distinct ids
    assert h.parent_version_id is None
    assert h.edit_type == "initial"
    assert (tmp_path / "store" / h.content_hash).exists()


def test_identical_content_same_hash_different_version_id(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    h1 = cb.snapshot(parent_version_id=h0.version_id)
    assert h0.content_hash == h1.content_hash  # same tree
    assert h0.version_id != h1.version_id       # distinct version
    assert h1.parent_version_id == h0.version_id


def test_different_content_different_hash(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    _write(ws, "agent.py", "print('v1')\n")
    h1 = cb.snapshot(parent_version_id=h0.version_id)
    assert h0.content_hash != h1.content_hash


def test_snapshot_copies_tree_into_store(tmp_path):
    ws = _make_workspace(tmp_path)
    _write(ws, "rules/expand.py", "def r():\n    pass\n")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    snap = tmp_path / "store" / h.content_hash
    assert (snap / "agent.py").exists()
    assert (snap / "rules/expand.py").exists()
    assert (snap / "manifest.toml").exists()


def test_restore_copies_old_content_back_and_tags_rollback(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    _write(ws, "agent.py", "print('v1-changed')\n")
    h1 = cb.snapshot(parent_version_id=h0.version_id)
    assert (ws / "agent.py").read_text() == "print('v1-changed')\n"

    hr = cb.restore(content_hash=h0.content_hash, parent_version_id=h1.version_id)
    assert hr.edit_type == "rollback"
    assert hr.content_hash == h0.content_hash  # content restored
    assert hr.version_id not in (h0.version_id, h1.version_id)  # new version
    # the workspace now holds the old content
    assert (ws / "agent.py").read_text() == "print('v0')\n"


def test_restore_records_zero_kl_intent(tmp_path):
    """A rollback is a distinct act (nonzero budget) but zero policy-KL.
    The handle carries edit_type=rollback so the controller can record both."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    hr = cb.restore(content_hash=h0.content_hash, parent_version_id=h0.version_id)
    assert hr.edit_type == "rollback"
    assert hr.parent_version_id == h0.version_id


def test_diff_is_structured_file_level(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    _write(ws, "rules/a.py", "a = 1\n")
    h0 = cb.snapshot(parent_version_id=None)
    _write(ws, "rules/a.py", "a = 2\n")          # modified
    _write(ws, "rules/b.py", "b = 1\n")          # added
    os.remove(ws / "manifest.toml")               # removed
    h1 = cb.snapshot(parent_version_id=h0.version_id)

    diff = cb.diff(h0.content_hash, h1.content_hash)
    assert set(diff.added) == {"rules/b.py"}
    assert set(diff.removed) == {"manifest.toml"}
    assert set(diff.modified) == {"rules/a.py"}
    assert set(diff.unchanged) == {"agent.py"}


def test_diff_identical_is_empty(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    h1 = cb.snapshot(parent_version_id=h0.version_id)
    diff = cb.diff(h0.content_hash, h1.content_hash)
    assert diff.added == [] and diff.removed == [] and diff.modified == []


def test_hash_is_deterministic_order_independent(tmp_path):
    """Content hash is stable regardless of filesystem enumeration order."""
    ws = tmp_path / "workspace"
    _write(ws, "a.py", "1\n")
    _write(ws, "b.py", "2\n")
    _write(ws, "manifest.toml", 'shape = "single_file"\nentrypoint = "a.py"\n')
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h0 = cb.snapshot(parent_version_id=None)
    # snapshot again -> same hash (content unchanged)
    h1 = cb.snapshot(parent_version_id=h0.version_id)
    assert h0.content_hash == h1.content_hash


def test_snapshot_ignores_cache_and_pycache(tmp_path):
    ws = _make_workspace(tmp_path)
    _write(ws, "__pycache__/junk.pyc", "x")
    _write(ws, ".cache/scratch.txt", "y")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    snap = tmp_path / "store" / h.content_hash
    assert not (snap / "__pycache__").exists()
    assert not (snap / ".cache").exists()


def test_partial_tree_still_snapshots(tmp_path):
    """A crash mid-edit leaves a readable (if partial) tree; snapshot still
    succeeds — version_after is the partial tree, never None for a readable ws."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    # simulate a partial edit: agent.py truncated but present
    (ws / "agent.py").write_text("", encoding="utf-8")
    h = cb.snapshot(parent_version_id=None)
    assert h.content_hash  # still produced a hash


# ---- manifest.py ----

def test_load_single_file_manifest(tmp_path):
    ws = _make_workspace(tmp_path)
    m = load_manifest(ws / "manifest.toml")
    assert m.shape == "single_file"
    assert m.entrypoint == "agent.py"
    assert m.rule_order == []


def test_load_package_manifest_with_rules(tmp_path):
    ws = tmp_path / "workspace"
    _write(ws, "manifest.toml",
           'shape = "package"\n'
           'entrypoint = "agent.py"\n'
           '[rules]\n'
           'order = ["expand", "attack", "escape", "fallback"]\n')
    m = load_manifest(ws / "manifest.toml")
    assert m.shape == "package"
    assert m.rule_order == ["expand", "attack", "escape", "fallback"]


def test_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.toml")
