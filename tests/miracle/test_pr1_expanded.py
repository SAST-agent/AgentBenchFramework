"""PR#1 expanded tests: end-to-end fields, classification, resume, atomic interrupt."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.match_runner import MatchAttempt, classify, ReplayInfo, TraceStats
from agentbench_frame.games.miracle.runner import attempt_to_outcome
from agentbench_frame.games.miracle.result import to_event_record


# ---- helper: full MatchAttempt ---- #
def _full_att(**over):
    base = dict(
        game_id="m_rank01_camp0", evaluated_agent="ifelse", opponent="rank01",
        evaluated_agent_camp=0, valid=False, normalized_result="error",
        error_type="ai_crash", reason="opponent AI crashed",
        raw_winner=None, winner_agent=None,
        scores={"0": 5, "1": 2}, steps=42,
        realized_randomization={"map_type": 1, "day_time": 0},
        result_json_status="ok", discrepancies=[],
        ai_crash_player=1, ai_timeout_player=None,
        judge_crash=False, wrapper_timeout=False, normal_cleanup_nonzero=False,
        evidence_paths={"stdout": "rel/stdout", "stderr": "rel/stderr",
                        "trace": "rel/trace", "replay": "rel/replay",
                        "result_json": "rel/result.json"},
        collision_detected=False,
        process_cleanup=[{"role": "judge", "cleanup_succeeded": True, "final_returncode": 0},
                         {"role": "ai0", "cleanup_succeeded": True, "final_returncode": 0},
                         {"role": "ai1", "cleanup_succeeded": True, "final_returncode": 1}],
        vendor_returncode=0, started_at=1000.0, finished_at=1005.0, duration_s=5.0,
        exception="RuntimeError('AI read EOF')",
        judge_exit=0, ai0_exit=0, ai1_exit=1,
        replay_sha256="abc123",
    )
    base.update(over)
    return MatchAttempt(**base)


# ===== SECTION 1: end-to-end field propagation ===== #
def test_end_to_end_field_propagation(tmp_path):
    att = _full_att()
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    # write to disk and re-read (simulates events.jsonl)
    p = tmp_path / "event.json"
    p.write_text(json.dumps(rec, default=str), encoding="utf-8")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    # assert every key field
    assert loaded["judge_exit"] == 0
    assert loaded["ai0_exit"] == 0
    assert loaded["ai1_exit"] == 1
    assert loaded["exception"] == "RuntimeError('AI read EOF')"
    assert loaded["reason"] == "opponent AI crashed"
    assert loaded["replay_sha256"] == "abc123"
    assert loaded["process_cleanup"] == att.process_cleanup
    assert loaded["result_json_status"] == "ok"
    assert loaded["vendor_returncode"] == 0
    assert loaded["evidence_paths"]["replay"] == "rel/replay"
    assert loaded["steps"] == 42
    assert loaded["seed"]["realized_randomization"] == {"map_type": 1, "day_time": 0}
    assert loaded["error_type"] == att.error_type  # NOT overwritten by normalized_result


def test_exception_and_reason_separately_preserved():
    att = _full_att(exception="FileNotFoundError('no file')", reason="vendor file missing")
    o = attempt_to_outcome(att)
    rec = to_event_record(o)
    assert rec["exception"] == "FileNotFoundError('no file')"
    assert rec["reason"] == "vendor file missing"


# ===== SECTION 2: expanded classification (10 cases) ===== #
def _ri_ok():
    return ReplayInfo(exists=True, header_valid=True, map_type=0, day_time=1, sha256="x")


def _ts_ok(end_info):
    return TraceStats(end_info_seen=True, end_info=end_info, n_ai_operation=10)


def test_class_vendor_rc_nonzero_no_exception():
    ri = _ri_ok(); ts = _ts_ok({"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": True,
                                     "run_match_returncode": 0, "raw_winner": 0}, ts=ts, ri=ri,
                 discrepancies=[], vendor_returncode=1, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "vendor_exception"


def test_class_result_json_internal_rc_nonzero():
    ri = _ri_ok(); ts = _ts_ok({"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": True,
                                     "run_match_returncode": 1, "exception": None, "raw_winner": 0}, ts=ts, ri=ri,
                 discrepancies=[], vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "vendor_exception"


def test_class_vendor_exception_with_end_info():
    ri = _ri_ok(); ts = _ts_ok({"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": True,
                                     "exception": "ValueError('bad')", "raw_winner": 0}, ts=ts, ri=ri,
                 discrepancies=[], vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "vendor_exception"


def test_class_result_json_missing():
    ri = ReplayInfo(exists=False)
    c = classify(rj_status="missing", rj=None, ts=TraceStats(), ri=ri, discrepancies=[],
                 vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "result_json_missing"


def test_class_result_json_corrupt():
    c = classify(rj_status="corrupt", rj=None, ts=TraceStats(), ri=ReplayInfo(exists=False),
                 discrepancies=[], vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "result_json_corrupt"


def test_class_cleanup_failure():
    ri = _ri_ok(); ts = _ts_ok({"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": False,
                                     "raw_winner": 0}, ts=ts, ri=ri, discrepancies=[],
                 vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "cleanup_failure"


def test_class_normal_cleanup_nonzero_still_valid():
    ri = _ri_ok(); ts = _ts_ok({"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": True,
                                     "raw_winner": 0,
                                     "ai0": {"termination_requested": True, "final_returncode": 1},
                                     "ai1": {"termination_requested": True, "final_returncode": 1}},
                 ts=ts, ri=ri, discrepancies=[], vendor_returncode=0, wrapper_timeout=False,
                 evaluated_agent_camp=0)
    assert c.valid is True
    assert c.normal_cleanup_nonzero is True


def test_class_ai_crash_allows_continue():
    ri = ReplayInfo(exists=True, header_valid=True)
    ts = TraceStats(ai_error_players=[1], end_info_seen=True, end_info={"0": 5, "1": 2})
    c = classify(rj_status="ok", rj={"end_info_received": True, "cleanup_all_succeeded": True,
                                     "raw_winner": 0}, ts=ts, ri=ri, discrepancies=[],
                 vendor_returncode=0, wrapper_timeout=False, evaluated_agent_camp=0)
    assert not c.valid and c.error_type == "ai_crash"
    # matrix should_stop: ai_crash is NOT in INFRA_STOP → allows continue
    from agentbench_frame.games.miracle.matrix import should_stop
    assert should_stop(_full_att(error_type="ai_crash", valid=False))[0] != "stop"


def test_class_infra_triggers_matrix_stop():
    from agentbench_frame.games.miracle.matrix import should_stop
    assert should_stop(_full_att(error_type="cleanup_failure", valid=False))[0] == "stop"
    assert should_stop(_full_att(error_type="vendor_exception", valid=False))[0] == "stop"


# ===== SECTION 3: resume expanded ===== #
def test_resume_does_not_create_new_session(tmp_path):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    r = MatrixRunner(session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
                     opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
                     framework_src=tmp_path / "src", attempt_fn=lambda **k: _full_att())
    r.prepare_session()
    sid = r.session_id
    sessions_before = list(tmp_path.glob("*"))
    r.resume(sid)
    sessions_after = list(tmp_path.glob("*"))
    assert sessions_before == sessions_after  # no new session created


def test_resume_corrupted_progress_rejected(tmp_path):
    """Corrupt progress.json must be rejected by verify_session_for_resume (read-only)."""
    _wm(tmp_path)
    (tmp_path / "progress.json").write_text("NOT JSON {{{", encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok
    assert any("progress" in e.lower() and "corrupt" in e.lower() for e in errs)


def test_resume_running_state_uncertain(tmp_path):
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle import matrix
    r = MatrixRunner(session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
                     opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
                     framework_src=tmp_path / "src", attempt_fn=lambda **k: None)
    r.prepare_session()
    plan = matrix.make_attempt_plan()
    matrix.mark_running(r.progress, plan[0]["game_id"])
    matrix.write_progress_atomic(r.progress_path, r.progress)
    r.resume(r.session_id)
    res = r.execute()
    assert res["halted"] is True and "UNCERTAIN" in res["reason"]


# ===== SECTION 4: atomic write subprocess interrupt ===== #
def test_atomic_write_survives_subprocess_kill(tmp_path):
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    target = tmp_path / "target.json"
    # write old complete content
    atomic_write_json(target, {"old": True})
    old_content = target.read_text(encoding="utf-8")
    # simulate interrupted write: monkeypatch os.replace to raise (process killed before replace)
    import agentbench_frame.games.miracle.atomicio as aio
    original_replace = aio.os.replace
    def kill_before_replace(*a, **k):
        raise ProcessLookupError("process killed")
    aio.os.replace = kill_before_replace
    try:
        with pytest.raises(ProcessLookupError):
            atomic_write_json(target, {"new": False})
    finally:
        aio.os.replace = original_replace
    # target must still have old complete content (not partial)
    assert target.read_text(encoding="utf-8") == old_content
    # temp file exists and is auditable
    temps = list(tmp_path.glob(".target.json.*.tmp"))
    assert len(temps) >= 1, "temp file should be detectable after interrupt"


def test_atomic_write_first_write_killed_no_partial(tmp_path):
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    target = tmp_path / "first.json"
    assert not target.exists()
    import agentbench_frame.games.miracle.atomicio as aio
    original_replace = aio.os.replace
    aio.os.replace = lambda *a, **k: (_ for _ in ()).throw(ProcessLookupError("killed"))
    try:
        with pytest.raises(ProcessLookupError):
            atomic_write_json(target, {"data": 1})
    finally:
        aio.os.replace = original_replace
    # target must NOT exist (not a partial file)
    assert not target.exists()


# ===== SECTION 2: manifest verification (7 cases) ===== #
from agentbench_frame.games.miracle.matrix_runner import verify_session_for_resume

def _write_manifest(tmp_path, **over):
    plan = [{"rank": r, "camp": c, "game_id": f"m_rank{r:02d}_camp{c}"}
            for r in range(1, 17) for c in (0, 1)]
    m = {"protocol_sha256": "a"*64, "code_hashes": {"matrix": "abc"},
         "plan_count": 32, "plan": plan, "run_id": "RID",
         "timeout": 8.0, "wrapper_timeout_s": 180.0,
         "session_id": tmp_path.name}
    m.update(over)
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    if not (tmp_path / "progress.json").exists():
        (tmp_path / "progress.json").write_text('{"attempts": {}}', encoding="utf-8")

def test_manifest_missing(tmp_path):
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and "manifest missing" in errs[0]

def test_manifest_corrupt(tmp_path):
    (tmp_path / "manifest.json").write_text("NOT JSON {{{", encoding="utf-8")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and "corrupt" in errs[0]

def test_manifest_protocol_hash_change(tmp_path):
    _write_manifest(tmp_path, protocol_sha256="wrong")
    ok, errs = verify_session_for_resume(tmp_path, protocol_sha="a"*64)
    assert not ok and any("protocol" in e for e in errs)

def test_manifest_code_hash_change(tmp_path):
    codefile = tmp_path / "matrix.py"; codefile.write_text("x=1")
    _write_manifest(tmp_path, code_hashes={"matrix": "wrong"})
    ok, errs = verify_session_for_resume(tmp_path, code_files={"matrix": str(codefile)})
    assert not ok and any("code hash" in e for e in errs)

def test_manifest_plan_change(tmp_path):
    _write_manifest(tmp_path, plan=[{"rank": 1, "camp": 0, "game_id": "x"}], plan_count=1)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("plan" in e for e in errs)

def test_manifest_run_id_missing(tmp_path):
    _write_manifest(tmp_path, run_id="")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("run_id" in e for e in errs)

def test_manifest_timeout_change(tmp_path):
    _write_manifest(tmp_path, timeout=99.0)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("timeout" in e for e in errs)

def test_manifest_valid_passes(tmp_path):
    _write_manifest(tmp_path)
    ok, errs = verify_session_for_resume(tmp_path, protocol_sha="a"*64)
    assert ok and errs == []


# ===== SECTION: asset/session/python identity + write-safety + CLI mutex ===== #
import hashlib as _hl

def _wm(tmp_path, **over):
    """Full manifest with all identity fields."""
    plan = [{"rank": r, "camp": c, "game_id": f"m_rank{r:02d}_camp{c}"}
            for r in range(1, 17) for c in (0, 1)]
    m = {
        "protocol_sha256": "a"*64, "code_hashes": {"matrix": "abc"},
        "plan_count": 32, "plan": plan, "run_id": "RID",
        "timeout": 8.0, "wrapper_timeout_s": 180.0,
        "ifelse_sha256": "if_sha", "judge_sha256": "j_sha",
        "opponent_archive_sha256": {f"rank{r:02d}": f"opp_{r}" for r in range(1,17)},
        "cpp_build_sha256": {f"rank{r:02d}": f"bld_{r}" for r in [1,2,3,6]+list(range(8,17))},
        "session_id": tmp_path.name, "python_version": "3.11.15", "platform": "test_plat",
    }
    m.update(over)
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    if not (tmp_path / "progress.json").exists():
        (tmp_path / "progress.json").write_text('{"attempts": {}}', encoding="utf-8")


def test_verify_ifelse_sha_mismatch(tmp_path):
    _wm(tmp_path, ifelse_sha256="wrong")
    ok, errs = verify_session_for_resume(tmp_path, expected_ifelse_sha="if_sha")
    assert not ok and any("if-else" in e for e in errs)


def test_verify_judge_sha_mismatch(tmp_path):
    _wm(tmp_path, judge_sha256="wrong")
    ok, errs = verify_session_for_resume(tmp_path, expected_judge_sha="j_sha")
    assert not ok and any("judge" in e for e in errs)


def test_verify_opponent_sha_mismatch(tmp_path):
    _wm(tmp_path, opponent_archive_sha256={f"rank{r:02d}": f"opp_{r}" for r in range(1,17)},
        **{"opponent_archive_sha256.rank01": "wrong"})  # this won't work as **kw
    # simpler: just write a bad opponent hash directly
    import json as _j
    m = _j.loads((tmp_path / "manifest.json").read_text())
    m["opponent_archive_sha256"]["rank01"] = "WRONG"
    (tmp_path / "manifest.json").write_text(_j.dumps(m))
    ok, errs = verify_session_for_resume(tmp_path, expected_opponent_shas={1: "opp_1"})
    assert not ok and any("opponent" in e for e in errs)


def test_verify_build_sha_mismatch(tmp_path):
    _wm(tmp_path)
    import json as _j
    m = _j.loads((tmp_path / "manifest.json").read_text())
    m["cpp_build_sha256"]["rank01"] = "WRONG"
    (tmp_path / "manifest.json").write_text(_j.dumps(m))
    ok, errs = verify_session_for_resume(tmp_path, expected_build_shas={1: "bld_1"})
    assert not ok and any("build" in e for e in errs)


def test_verify_session_id_mismatch(tmp_path):
    _wm(tmp_path, session_id="WRONG_SID")
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("session_id" in e for e in errs)


def test_verify_python_version_mismatch(tmp_path):
    _wm(tmp_path, python_version="3.99.0")
    ok, errs = verify_session_for_resume(tmp_path, expected_python="3.11.15")
    assert not ok and any("python" in e.lower() for e in errs)


def test_verify_failure_leaves_session_unchanged(tmp_path):
    _wm(tmp_path, protocol_sha256="wrong")
    before = {p.name: _hl.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob("*") if p.is_file()}
    ok, errs = verify_session_for_resume(tmp_path, protocol_sha="a"*64)
    assert not ok
    after = {p.name: _hl.sha256(p.read_bytes()).hexdigest()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after, "session files changed during verify!"


def test_cli_resume_dry_run_mutex():
    """--resume and --dry-run together must fail immediately (exit 2)."""
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "AGENTBENCH_ROOT": str(repo), "MIRACLE_IFELSE_DIR": "dummy"}
    r = subprocess.run(
        [sys.executable, str(repo / "tools" / "miracle_matrix.py"),
         "--dry-run", "--resume", "xxx"],
        capture_output=True, text=True, timeout=15, cwd=str(repo), env=env)
    assert r.returncode == 2
    assert "mutually exclusive" in r.stderr.lower()


# ===== FINAL GAP: plan rank/camp + wrapper_timeout_s + platform ===== #
def test_plan_rank_tamper_rejected(tmp_path):
    _wm(tmp_path)
    import json as _j
    m = _j.loads((tmp_path / "manifest.json").read_text())
    m["plan"][0]["rank"] = 99  # game_id unchanged, rank wrong
    (tmp_path / "manifest.json").write_text(_j.dumps(m))
    from agentbench_frame.games.miracle.matrix import make_attempt_plan
    ok, errs = verify_session_for_resume(tmp_path, expected_plan=make_attempt_plan())
    assert not ok and any("rank" in e for e in errs)


def test_plan_camp_tamper_rejected(tmp_path):
    _wm(tmp_path)
    import json as _j
    m = _j.loads((tmp_path / "manifest.json").read_text())
    m["plan"][1]["camp"] = 99  # game_id unchanged, camp wrong
    (tmp_path / "manifest.json").write_text(_j.dumps(m))
    from agentbench_frame.games.miracle.matrix import make_attempt_plan
    ok, errs = verify_session_for_resume(tmp_path, expected_plan=make_attempt_plan())
    assert not ok and any("camp" in e for e in errs)


def test_wrapper_timeout_missing_rejected(tmp_path):
    _wm(tmp_path)
    import json as _j
    m = _j.loads((tmp_path / "manifest.json").read_text())
    del m["wrapper_timeout_s"]
    (tmp_path / "manifest.json").write_text(_j.dumps(m))
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("wrapper_timeout_s" in e for e in errs)


def test_wrapper_timeout_null_rejected(tmp_path):
    _wm(tmp_path, wrapper_timeout_s=None)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("wrapper_timeout_s" in e for e in errs)


def test_wrapper_timeout_value_change_rejected(tmp_path):
    _wm(tmp_path, wrapper_timeout_s=99.0)
    ok, errs = verify_session_for_resume(tmp_path)
    assert not ok and any("wrapper_timeout_s" in e for e in errs)


def test_platform_mismatch_rejected(tmp_path):
    _wm(tmp_path, platform="wrong_platform")
    ok, errs = verify_session_for_resume(tmp_path, expected_platform="test_plat")
    assert not ok and any("platform" in e for e in errs)


# ===== SECTION 2: partial-rank resume + run_id restore end-to-end ===== #
def test_resume_partial_rank_restores_run_id_and_skips_done(tmp_path):
    """rank01 camp0 done + 1 event, camp1 not_started, no audit → resume
    restores original run_id, executes only camp1, camp0 not re-run."""
    from agentbench_frame.games.miracle.matrix_runner import MatrixRunner
    from agentbench_frame.games.miracle import matrix

    calls = []
    def fake_fn(**kw):
        calls.append(kw["game_id"])
        return _full_att(game_id=kw["game_id"], valid=True, normalized_result="win")

    r = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=fake_fn, run_id="ORIGINAL_RID")
    r.prepare_session()
    original_run_id = r.run_id
    # write manifest with the run_id
    r.record_manifest(opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
                      build_hashes={i: "b" + str(i) for i in [1, 2, 3, 6] + list(range(8, 17))},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={})
    # simulate camp0 done
    matrix.mark_done(r.progress, "m_rank01_camp0",
                     {"valid": True, "normalized_result": "win", "rank": 1, "camp": 0})
    matrix.write_progress_atomic(r.progress_path, r.progress)
    matrix.append_event_atomic(r.events_path,
                              {"event": "game", "game_id": "m_rank01_camp0",
                               "valid": True, "normalized_result": "win"})

    # new runner with DIFFERENT run_id → resume should restore original
    r2 = MatrixRunner(
        session_root=tmp_path, judge_dir=tmp_path / "j", ifelse_dir=tmp_path / "i",
        opponent_dir_of=lambda rk: tmp_path / f"o{rk}", vendor_script=tmp_path / "v.py",
        framework_src=tmp_path / "src", attempt_fn=fake_fn, run_id="WRONG")
    r2.resume(r.session_id)
    assert r2.run_id == original_run_id, f"run_id not restored: {r2.run_id}"
    assert "WRONG" not in str(r2.run_dir)

    res = r2.execute_rank(rank=1)
    assert res.get("halted") is False
    assert "m_rank01_camp0" not in calls  # done NOT re-run
    assert "m_rank01_camp1" in calls       # camp1 executed
    ev = [json.loads(l)["game_id"] for l in r2.events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(ev) == 2 and len(set(ev)) == 2


# ===== SECTION 3: real subprocess kill atomic write test ===== #
def test_atomic_write_real_subprocess_kill(tmp_path):
    """Kill a real subprocess mid-write: target keeps old complete JSON."""
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    target = tmp_path / "target.json"
    atomic_write_json(target, {"old": True})
    old_content = target.read_text(encoding="utf-8")

    child = tmp_path / "child.py"
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    marker = str(tmp_path / "ready.marker")
    child.write_text(f"""
import sys, json, os, time
from pathlib import Path
sys.path.insert(0, {src_path!r})
from agentbench_frame.games.miracle.atomicio import atomic_write_json
import agentbench_frame.games.miracle.atomicio as aio
_marker = Path({marker!r})
_orig = aio.os.replace
def slow_replace(src, dst):
    _marker.write_text("ready")
    time.sleep(30)
    _orig(src, dst)
aio.os.replace = slow_replace
atomic_write_json({str(target)!r}, {{"new": False}})
""", encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(child)], cwd=str(tmp_path))
    mpath = tmp_path / "ready.marker"
    deadline = time.time() + 10
    while not mpath.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert mpath.exists(), "child never reached replace"
    proc.terminate()
    proc.wait(timeout=5)

    content = target.read_text(encoding="utf-8")
    assert content == old_content, "target changed during killed write!"
    json.loads(content)  # parses cleanly
    for t in tmp_path.glob(".target.json.*.tmp"):
        json.loads(t.read_text(encoding="utf-8"))  # temp is complete JSON


def test_atomic_write_first_write_killed_no_target(tmp_path):
    """First write killed mid-replace: target must not exist (no partial)."""
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    target = tmp_path / "first.json"
    assert not target.exists()

    child = tmp_path / "child2.py"
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    marker = str(tmp_path / "ready2.marker")
    child.write_text(f"""
import sys, time
from pathlib import Path
sys.path.insert(0, {src_path!r})
from agentbench_frame.games.miracle.atomicio import atomic_write_json
import agentbench_frame.games.miracle.atomicio as aio
_orig = aio.os.replace
def slow_replace(src, dst):
    Path({marker!r}).write_text("ready")
    time.sleep(30)
    _orig(src, dst)
aio.os.replace = slow_replace
atomic_write_json({str(target)!r}, {{"data": 1}})
""", encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(child)], cwd=str(tmp_path))
    mpath = tmp_path / "ready2.marker"
    deadline = time.time() + 10
    while not mpath.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert mpath.exists()
    proc.terminate()
    proc.wait(timeout=5)
    assert not target.exists(), "partial target created!"
