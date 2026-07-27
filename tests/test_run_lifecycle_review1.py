"""review #1 on framework: optional ``run_id`` in ``Run.start`` (backcompat TDD).

This covers the matrix-resume / migration scenario where the caller already
has a fixed ``run_id`` (e.g. ``20260722-001929_52bd14``) and must NOT let
``Run.start`` fabricate a different one — otherwise a new run directory is
created and resume integrity breaks (review #1 §1 constraint).

Sequence:
  1. Write a red-light test that asserts ``Run.start(run_id=...)`` accepts
     a caller-supplied id and reuses it (rewrite events.jsonl + run.toml).
  2. Apply the minimal back-compat patch on ``Run.start``.
  3. Re-run: must be GREEN.

Also covers idempotent re-write: ``Run.start(run_id=X)`` called twice in the
same directory must not produce a second run directory (run_dir reuses the
same path), and events must not be double-appended on a ``Run`` object reach
(although the normal lifecycle recreates the writer fresh each time).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_run_start_accepts_caller_run_id(tmp_path: Path):
    from agentbench_frame.tracking.run import Run
    rid = "20260722-001929_52bd14"
    run = Run.start(game="24_miracle", agent="miracle_ifelse", run_type="eval",
                    data_dir=str(tmp_path), run_id=rid,
                    config={"matrix": "plan_a_32"})
    assert run.run_id == rid
    run_dir = tmp_path / "runs" / "24_miracle" / "miracle_ifelse" / rid
    assert run.run_dir == str(run_dir)
    assert run_dir.exists()
    # write an event, finish, and confirm run_id round-trips to disk
    run.log_episode(reward=1.0, steps=5, winner=0)
    run.finish()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == rid
    toml = (run_dir / "run.toml").read_text(encoding="utf-8")
    assert f'run_id = "{rid}"' in toml
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert events and all(e.get("run_id") == rid for e in events), \
        "every event must carry the same caller-supplied run_id"


def test_run_start_default_run_id_unchanged(tmp_path: Path):
    """Backward compatibility: no run_id kwarg -> framework auto-generates."""
    from agentbench_frame.tracking.run import Run
    run = Run.start(game="28_generals", agent="x", run_type="eval", data_dir=str(tmp_path))
    assert run.run_id  # non-empty, framework-generated form "YYYYMMDD_HHMM_<8hex>"
    # must look like an auto-generated id (timestamp prefix + 8 hex suffix)
    parts = run.run_id.split("_")
    assert len(parts) == 3
    assert parts[0].isdigit() and parts[1].isdigit()
    assert len(parts[2]) == 8
    run.finish()


def test_run_start_run_id_reuses_same_dir_no_second(tmp_path: Path):
    """Two Run.start calls with the SAME caller run_id must reuse the same
    directory path; no second run directory is created."""
    from agentbench_frame.tracking.run import Run
    rid = "20260722-001929_52bd14"
    r1 = Run.start(game="24_miracle", agent="a", run_type="eval",
                   data_dir=str(tmp_path), run_id=rid)
    r2 = Run.start(game="24_miracle", agent="a", run_type="eval",
                   data_dir=str(tmp_path), run_id=rid)
    assert r1.run_dir == r2.run_dir
    siblings = list((tmp_path / "runs" / "24_miracle" / "a").iterdir())
    assert len([p for p in siblings if p.is_dir()]) == 1
    r1.finish()
    r2.finish()