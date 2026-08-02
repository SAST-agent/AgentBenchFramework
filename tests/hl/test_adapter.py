"""Tests for hl/adapter.py — run a versioned candidate through the eval contract.

The adapter takes a VersionHandle (a snapshot in the codebase store) and
produces the (command, cwd) pair that LostSpaceEvaluator already consumes as
``candidate_command``. It stages a *copy* of the snapshot into a fresh run
dir so the canonical store is never mutated by an eval.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentbench_frame.hl.adapter import stage_candidate, candidate_command
from agentbench_frame.hl.codebase import HLCodebase


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(
        "import json, sys\n"
        "line = sys.stdin.readline()\n"
        "print(json.dumps({'type':'id'}) if 'id' in line else '{}')\n",
        encoding="utf-8",
    )
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8",
    )
    return ws


def test_stage_candidate_copies_snapshot_to_run_dir(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)

    stage = tmp_path / "stage"
    staged = stage_candidate(h, store=cb.store, dest=stage)
    assert (staged / "agent.py").exists()
    assert (staged / "manifest.toml").exists()
    # the staged copy is independent of the canonical store
    assert staged.resolve() != (cb.store / h.content_hash).resolve()


def test_candidate_command_runs_python_on_entrypoint(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    stage = tmp_path / "stage"
    staged = stage_candidate(h, store=cb.store, dest=stage)

    cmd, cwd = candidate_command(h, store=cb.store, dest=stage)
    assert cmd[0] in (sys.executable, "python", "python3")
    assert cmd[1].endswith("agent.py")
    assert str(cwd) == str(staged)


def test_candidate_command_executes_and_responds(tmp_path):
    """The staged candidate actually runs and reads stdin / writes stdout —
    the wire contract the LostSpace match harness depends on."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    stage = tmp_path / "stage"
    cmd, cwd = candidate_command(h, store=cb.store, dest=stage)

    proc = subprocess.run(
        cmd, cwd=str(cwd),
        input='{"type":"id"}\n', capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    import json as _j
    assert _j.loads(out)["type"] == "id"


def test_stage_does_not_mutate_canonical_store(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    canonical = cb.store / h.content_hash
    original = (canonical / "agent.py").read_bytes()

    stage = tmp_path / "stage"
    stage_candidate(h, store=cb.store, dest=stage)
    # mutate the staged copy
    (stage / "agent.py").write_text("MUTATED\n", encoding="utf-8")

    assert (canonical / "agent.py").read_bytes() == original  # untouched


def test_relative_dest_resolves_to_absolute(tmp_path, monkeypatch):
    """Windows CreateProcess resolves a relative argv against cwd, so a
    relative dest must be made absolute before the command string is built.
    Otherwise the candidate path nests (``.hl_codebase/stage/.hl_codebase/...``),
    never starts, and the probe swallows the OSError as a ``None`` emission —
    collapsing every reference point to uniform and masking ``policy_kl = 0``.
    """
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)

    # Isolate the relative-dest resolution from the repo-root cwd so the test
    # never creates a stray ``./stage`` in the repository.
    monkeypatch.chdir(tmp_path)
    staged = stage_candidate(h, store=cb.store, dest="stage")
    cmd, cwd = candidate_command(h, store=cb.store, dest="stage")

    assert staged.is_absolute()
    assert cwd.is_absolute()
    entry = Path(cmd[1])
    assert entry.is_absolute()
    assert entry == staged / "agent.py"
    assert entry.exists()  # the argv target resolves under any cwd


def test_package_shape_stage_includes_rules(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("print('ok')\n", encoding="utf-8")
    (ws / "rules").mkdir()
    (ws / "rules/expand.py").write_text("def r(): pass\n", encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "package"\nentrypoint = "agent.py"\n'
        '[rules]\norder = ["expand"]\n', encoding="utf-8",
    )
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    stage = tmp_path / "stage"
    staged = stage_candidate(h, store=cb.store, dest=stage)
    assert (staged / "rules/expand.py").exists()
    assert (staged / "agent.py").exists()
