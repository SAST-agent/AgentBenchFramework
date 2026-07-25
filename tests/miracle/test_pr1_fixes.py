"""PR#1 review#2-6 red-light tests (方案C: 基线 main，review#1 统一封装待定).

Each group must FAIL against current code (the gap), then turn green after fix.
review#1 (unified event envelope) is DEFERRED (framework-dependent).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------- review#4: atomic result-json write (temp + fsync + os.replace) ---------- #
def test_atomic_write_json_exists_and_atomic(tmp_path):
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    p = tmp_path / "r.json"
    atomic_write_json(p, {"a": 1, "b": "x"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "b": "x"}
    # no leftover temp file
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("r.json.*"))


def test_atomic_write_json_preserves_old_on_failure(tmp_path, monkeypatch):
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    p = tmp_path / "r.json"
    p.write_text('{"old": true}', encoding="utf-8")
    # inject failure during the replace step
    import agentbench_frame.games.miracle.atomicio as aio
    def boom(*a, **k):
        raise OSError("injected")
    monkeypatch.setattr(aio.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(p, {"new": 1})
    # old complete file intact (or file unchanged)
    assert json.loads(p.read_text(encoding="utf-8")) == {"old": True}


# ---------- review#5: first-hand process fields on GameOutcome / event record ---------- #
def test_gameoutcome_has_first_hand_fields():
    from agentbench_frame.games.miracle.result import GameOutcome, finalize
    o = finalize(GameOutcome(game_id="g", evaluated_agent="a", opponent="b", evaluated_agent_camp=0))
    for f in ("judge_exit", "ai0_exit", "ai1_exit", "replay_sha256", "process_cleanup"):
        assert hasattr(o, f), f"GameOutcome missing {f}"


def test_event_record_includes_first_hand_fields():
    from agentbench_frame.games.miracle.result import GameOutcome, finalize, to_event_record
    o = finalize(GameOutcome(game_id="g", evaluated_agent="a", opponent="b", evaluated_agent_camp=0,
                             judge_exit=0, ai0_exit=0, ai1_exit=1, replay_sha256="abc",
                             process_cleanup=[{"role": "judge", "cleanup_succeeded": True}]))
    rec = to_event_record(o)
    for f in ("judge_exit", "ai0_exit", "ai1_exit", "replay_sha256", "process_cleanup"):
        assert f in rec, f"event record missing {f}"


# ---------- review#6: classification of vendor/cleanup/result-json infra failures ---------- #
class _Att:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

def _base_att(**over):
    base = dict(wrapper_timeout=False, error_type=None, result_json_status="ok",
                discrepancies=[], realized_randomization={"map_type": 0, "day_time": 1},
                raw_winner=0, ai_crash_player=None, ai_timeout_player=None,
                judge_crash=False, normal_cleanup_nonzero=False, evaluated_agent_camp=0,
                evaluated_agent="ifelse", opponent="r", game_id="g")
    base.update(over)
    return _Att(**base)

def test_classify_cleanup_failure_is_infra_invalid():
    from agentbench_frame.games.miracle.match_runner import classify, ReplayInfo, TraceStats
    # valid end_info + replay ok, but cleanup failed -> infra invalid (not valid)
    ri = ReplayInfo(exists=True, header_valid=True)
    c = classify(rj_status="ok", rj={"end_info_received": True, "ai0": {"termination_requested": True, "final_returncode": 0},
                                     "ai1": {"termination_requested": True, "final_returncode": 0},
                                     "raw_winner": 0, "cleanup_all_succeeded": False, "run_match_returncode": 0},
                 ts=TraceStats(end_info_seen=True, end_info={"0": 5, "1": 2}), ri=ri, discrepancies=[],
                 vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert c.valid is False
    assert c.error_type in ("cleanup_failure", "infra_failure")


def test_classify_vendor_nonzero_with_end_info_is_infra():
    from agentbench_frame.games.miracle.match_runner import classify, ReplayInfo, TraceStats
    ri = ReplayInfo(exists=True, header_valid=True)
    c = classify(rj_status="ok", rj={"end_info_received": True,
                                     "ai0": {"termination_requested": False, "final_returncode": 0},
                                     "ai1": {"termination_requested": False, "final_returncode": 0},
                                     "raw_winner": 0, "cleanup_all_succeeded": True, "run_match_returncode": 1,
                                     "exception": "RuntimeError('x')"},
                 ts=TraceStats(end_info_seen=True, end_info={"0": 5, "1": 2}), ri=ri, discrepancies=[],
                 vendor_returncode=1, wrapper_timeout=False, evaluated_agent_camp=0)
    assert c.valid is False
    assert c.error_type in ("vendor_exception", "infra_failure")


# ---------- review#2: --resume <session_id> ---------- #
def test_matrix_runner_resume_method_exists():
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    assert hasattr(MatrixRunner, "resume"), "MatrixRunner missing resume()"


def test_resume_skips_done_no_rerun(tmp_path, monkeypatch):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle import matrix
    calls = []
    r = MatrixRunner(session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
                     opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
                     framework_src=tmp_path / "src", attempt_fn=lambda **k: calls.append(k["game_id"]) or _base_att())
    r.prepare_session()
    plan = matrix.make_attempt_plan()
    # mark rank01 both camps done already
    matrix.mark_done(r.progress, "m_rank01_camp0", {"valid": True, "normalized_result": "win"})
    matrix.mark_done(r.progress, "m_rank01_camp1", {"valid": True, "normalized_result": "loss"})
    matrix.write_progress_atomic(r.progress_path, r.progress)
    monkeypatch.setattr(r, "session_id", r.session_id)  # no-op; resume reuses existing session
    r.resume(r.session_id)
    r.execute_rank(rank=1)
    # done games not re-run
    assert "m_rank01_camp0" not in calls
    assert "m_rank01_camp1" not in calls


# ---------- review#3: POSIX reap (wait after terminate/kill) ---------- #
def test_proctree_cleanup_reaps_popen():
    from agentbench_frame.games.miracle.proctree import ProcessTreeManager
    # spawn a long-lived child via Popen
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        mgr = ProcessTreeManager()
        mp = mgr.register_popen(proc, "child")
        mgr.cleanup_all("test")
        # reap: Popen.returncode must be set (wait was called), not None (zombie/unreaped)
        assert proc.returncode is not None, "Popen not reaped (returncode None)"
    finally:
        if proc.returncode is None:
            proc.kill(); proc.wait(timeout=5)
