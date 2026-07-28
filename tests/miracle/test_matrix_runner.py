"""Tests for the matrix runner (tools/miracle_matrix.py logic).

Uses an injected FAKE attempt_fn (signature matches the runner's keyword call:
``attempt_fn(game_id=..., p0_dir=..., ...)``) — no Judge, no AI subprocess.
Covers the 15 runner safety requirements.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.games.miracle import matrix_runner
from agentbench_frame.games.miracle.matrix_runner import MatrixRunner


def _att(game_id, **over):
    rc = matrix_runner.parse_game_id(game_id)
    d = dict(valid=True, normalized_result="win", raw_winner=None, error_type=None,
             wrapper_timeout=False, result_json_status="ok", discrepancies=None,
             realized_randomization=None, scores=None, steps=10, ai_crash_player=None,
             judge_crash=False)
    d.update(over)
    rank, camp = rc
    rw = d["raw_winner"] if d["raw_winner"] is not None else (
        camp if d["normalized_result"] == "win" else 1 - camp)
    class A: pass
    a = A()
    a.game_id = game_id; a.rank = rank; a.camp = camp
    a.valid = d["valid"]; a.normalized_result = d["normalized_result"]; a.raw_winner = rw
    a.error_type = d["error_type"]; a.wrapper_timeout = d["wrapper_timeout"]
    a.result_json_status = d["result_json_status"]; a.discrepancies = d["discrepancies"] or []
    a.realized_randomization = d["realized_randomization"] or {"map_type": 0, "day_time": 1}
    a.scores = d["scores"] or {"0": 5, "1": 2}; a.steps = d["steps"]
    a.ai_crash_player = d["ai_crash_player"]; a.judge_crash = d["judge_crash"]
    a.winner_agent = "miracle_ifelse" if d["normalized_result"] == "win" else "opp"
    a.normal_cleanup_nonzero = False; a.reason = d["error_type"] or ""
    a.evidence_paths = {"stdout": "", "stderr": "", "trace": "", "replay": "",
                        "result_json": str(Path(game_id + ".rj"))}
    return a


def _runner(tmp_path, fn, **kw):
    return MatrixRunner(session_root=tmp_path, judge_dir=tmp_path / "j",
                        ifelse_dir=tmp_path / "ifelse", opponent_dir_of=lambda r: tmp_path / f"opp{r}",
                        vendor_script=tmp_path / "v.py", framework_src=tmp_path / "src",
                        timeout=8.0, wrapper_timeout_s=60.0, attempt_fn=fn,
                        protocol_sha="p" * 64, run_id="RID", **kw)


# 1. dry-run never starts Judge/AI
def test_dry_run_starts_nothing(tmp_path):
    called = []
    r = _runner(tmp_path, lambda **k: called.append(1))
    r.prepare_session()
    out = r.dry_run()
    assert called == []
    assert out["plan_count"] == 32


# 2. plan rank01-16 camp0/camp1, 32 unique
def test_dry_run_plan_correct(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    out = r.dry_run()
    ids = [a["game_id"] for a in out["plan"]]
    assert len(ids) == 32 and len(set(ids)) == 32
    assert out["plan"][0]["game_id"] == "m_rank01_camp0"
    assert out["plan"][-1]["game_id"] == "m_rank16_camp1"


# 3. existing session refused
def test_existing_session_refused(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    r2 = _runner(tmp_path, lambda **k: None)
    with pytest.raises(FileExistsError):
        r2.prepare_session_for_existing(r.session_id)


# 4+5. manifest atomic + hashes
def test_manifest_atomic_with_hashes(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    r.record_manifest(opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
                      build_hashes={i: "b" + str(i) for i in [1, 2, 3, 6] + list(range(8, 17))},
                      ifelse_sha="IF", judge_sha="JD", code_hashes={"matrix_runner": "MR"})
    m = json.loads((r.session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["protocol_sha256"] == "p" * 64
    assert m["plan_count"] == 32 and m["run_id"] == "RID"
    assert m["ifelse_sha256"] == "IF" and m["judge_sha256"] == "JD"
    assert not (r.session_dir / "manifest.json.tmp").exists()


def test_manifest_records_control_input_hashes(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    r.record_manifest(
        opponent_hashes={i: "h" + str(i) for i in range(1, 17)},
        build_hashes={i: "b" + str(i) for i in range(1, 17)},
        ifelse_sha="IF", judge_sha="JD", code_hashes={},
        control_inputs={"protocol": {"path": "/p.json", "sha256": "p"},
                        "roster": {"path": "/r.json", "sha256": "r"}},
    )
    manifest = json.loads((r.session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["control_inputs"]["protocol"]["sha256"] == "p"


# 6. running -> UNCERTAIN_IN_FLIGHT, no auto-rerun
def test_running_state_uncertain_no_rerun(tmp_path):
    r = _runner(tmp_path, lambda **k: None)
    r.prepare_session()
    plan = matrix_runner.make_attempt_plan()
    matrix_runner.mark_running(r.progress, plan[0]["game_id"])
    matrix_runner.write_progress_atomic(r.progress_path, r.progress)
    res = r.execute()
    assert res["halted"] is True
    assert "UNCERTAIN_IN_FLIGHT" in res["reason"]


# 7. done only after attempt returns
def test_done_only_after_attempt_returns(tmp_path):
    seen = []
    def fn(**k):
        seen.append(k["game_id"]); return _att(k["game_id"], normalized_result="win")
    r = _runner(tmp_path, fn)
    r.prepare_session()
    r.execute_rank(rank=1)
    prog = json.loads(r.progress_path.read_text(encoding="utf-8"))
    assert prog["attempts"]["m_rank01_camp0"]["state"] == "done"
    assert prog["attempts"]["m_rank01_camp1"]["state"] == "done"


# 8+9. valid/AI-invalid/infra distinct; AI-invalid continues, not in win-rate denom
def test_ai_invalid_continues_not_in_winrate(tmp_path):
    calls = [0]
    def fn(**k):
        i = calls[0]; calls[0] += 1
        if i == 0:
            return _att(k["game_id"], normalized_result="error", error_type="ai_crash", valid=False)
        return _att(k["game_id"], normalized_result="win")
    r = _runner(tmp_path, fn)
    r.prepare_session()
    r.execute_rank(rank=1)
    agg = r.aggregate_from_events()
    assert agg["total_attempts"] == 2 and agg["valid_games"] == 1 and agg["invalid_games"] == 1
    assert agg["wins"] == 1 and agg["win_rate"] == pytest.approx(1.0)


# 10. infra failure blocks next
def test_infra_failure_halts(tmp_path):
    calls = [0]
    def fn(**k):
        calls[0] += 1
        if calls[0] == 1:
            return _att(k["game_id"], error_type="evidence_mismatch", valid=False)
        return _att(k["game_id"])
    r = _runner(tmp_path, fn)
    r.prepare_session()
    res = r.execute_rank(rank=1)
    assert res["halted"] is True and res["state"] == "HALTED_INFRA_FAILURE"
    assert calls[0] == 1


# 11. rank audit after two games
def test_rank_audit_runs_after_two_games(tmp_path):
    r = _runner(tmp_path, lambda **k: _att(k["game_id"]))
    r.prepare_session()
    r.execute_rank(rank=1)
    assert (r.session_dir / "audit" / "rank01.json").exists()


# 12. progress/events no duplicate
def test_progress_and_events_no_duplicate(tmp_path):
    r = _runner(tmp_path, lambda **k: _att(k["game_id"]))
    r.prepare_session()
    r.execute_rank(rank=1)
    ev = [json.loads(l) for l in r.events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [e["game_id"] for e in ev]
    assert len(ids) == len(set(ids))


# 13. normal cleanup nonzero not crash
def test_normal_cleanup_nonzero_not_crash(tmp_path):
    def fn(**k):
        a = _att(k["game_id"]); a.normal_cleanup_nonzero = True; return a
    r = _runner(tmp_path, fn)
    r.prepare_session()
    res = r.execute_rank(rank=1)
    assert res["halted"] is False


# 14. PID residual blocks continue
def test_residual_blocks_continue(tmp_path):
    import os as _os, psutil
    me = psutil.Process(_os.getpid())
    def fn(**k):
        a = _att(k["game_id"])
        rjp = r.session_dir / (k["game_id"] + ".rj")
        rjp.write_text(json.dumps({"judge": {"pid": _os.getpid(), "started_at": me.create_time(), "role": "judge"}}))
        a.evidence_paths = {"result_json": str(rjp)}
        return a
    r = _runner(tmp_path, fn)
    r.prepare_session()
    res = r.execute_rank(rank=1)
    assert res["halted"] is True and "residual" in res["reason"]


# 15. partial/complete summary independently re-computable
def test_summary_recomputable_from_events(tmp_path):
    r = _runner(tmp_path, lambda **k: _att(k["game_id"]))
    r.prepare_session()
    r.execute_rank(rank=1)
    agg = r.aggregate_from_events()
    raw = [json.loads(l) for l in r.events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert agg["total_attempts"] == 2 and len(raw) == 2
    assert agg["wins"] == sum(1 for e in raw if e["normalized_result"] == "win")
