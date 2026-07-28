"""Red-light tests for the 32-game matrix orchestrator (阶段正式矩阵 §3 TDD gate).

Tests MUST pass before the first real game starts. They cover the spec's safety
requirements using synthesized MatchAttempt-like objects + tmp progress files —
no Judge, no real match.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.games.miracle import matrix  # red-light import
from agentbench_frame.games.miracle.matrix import (
    aggregate,
    classify_game,
    is_done,
    load_progress,
    make_attempt_plan,
    mark_done,
    next_incomplete,
    rank_audit,
    should_stop,
    write_progress_atomic,
)


# ---- fake MatchAttempt (same shape as match_runner.MatchAttempt for classify_game) ---- #
class FakeAtt:
    def __init__(self, *, game_id="g", rank=None, camp=None, valid=True, normalized_result="win",
                 raw_winner=None, error_type=None, wrapper_timeout=False,
                 result_json_status="ok", discrepancies=None, realized_randomization=None,
                 scores=None, steps=10, ai_crash_player=None, judge_crash=False,
                 cleanup_procs=None, evidence_paths=None):
        self.game_id = game_id; self.rank = rank; self.camp = camp
        self.valid = valid; self.normalized_result = normalized_result
        self.raw_winner = raw_winner if raw_winner is not None else (
            None if camp is None else (camp if normalized_result == "win" else 1 - camp))
        self.error_type = error_type; self.wrapper_timeout = wrapper_timeout
        self.result_json_status = result_json_status
        self.discrepancies = discrepancies or []
        self.realized_randomization = realized_randomization or {"map_type": 0, "day_time": 1}
        self.scores = scores or {"0": 5, "1": 2}; self.steps = steps
        self.ai_crash_player = ai_crash_player; self.judge_crash = judge_crash
        self.winner_agent = "miracle_ifelse" if normalized_result == "win" else "rank"
        self.normal_cleanup_nonzero = False; self.reason = error_type or ""
        self.evidence_paths = evidence_paths or {"result_json": ""}


# ---- 1. exactly 32 unique attempts ---- #
def test_plan_has_32_unique_attempts():
    plan = make_attempt_plan()
    ids = [a["game_id"] for a in plan]
    assert len(plan) == 32
    assert len(set(ids)) == 32


# ---- 2. each rank camp0/camp1 once ---- #
def test_plan_each_rank_both_camps_once():
    plan = make_attempt_plan()
    for rank in range(1, 17):
        camps = sorted(a["camp"] for a in plan if a["rank"] == rank)
        assert camps == [0, 1], f"rank{rank} camps={camps}"


def test_plan_order_is_rank_then_camp():
    plan = make_attempt_plan()
    seq = [(a["rank"], a["camp"]) for a in plan]
    assert seq[0] == (1, 0) and seq[1] == (1, 1) and seq[-1] == (16, 1)


# ---- 3. completed never rerun ---- #
def test_next_incomplete_skips_done():
    plan = make_attempt_plan()
    prog = {"attempts": {}}
    first = next_incomplete(prog, plan)
    assert first["game_id"] == plan[0]["game_id"]
    mark_done(prog, plan[0]["game_id"], {"valid": True, "normalized_result": "win"})
    assert is_done(prog, plan[0]["game_id"]) is True
    nxt = next_incomplete(prog, plan)
    assert nxt["game_id"] == plan[1]["game_id"]


def test_next_incomplete_none_when_all_done():
    plan = make_attempt_plan()
    prog = {"attempts": {}}
    for a in plan:
        mark_done(prog, a["game_id"], {"valid": True})
    assert next_incomplete(prog, plan) is None


# ---- 4. session exists not overwrite/delete (reuses smoke_audit) ---- #
def test_fresh_session_refused_if_exists(tmp_path):
    from agentbench_frame.games.miracle.smoke_audit import ensure_fresh_session
    sid = matrix.make_session_id()
    sd = ensure_fresh_session(tmp_path, sid)
    (sd / "x").write_text("keep")
    with pytest.raises(FileExistsError):
        ensure_fresh_session(tmp_path, sid)
    assert (sd / "x").read_text() == "keep"


# ---- 5. progress atomic write ---- #
def test_progress_atomic_write(tmp_path):
    prog = {"session_id": "s1", "attempts": {"g1": {"state": "done"}}}
    p = tmp_path / "progress.json"
    write_progress_atomic(p, prog)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == prog
    assert not (tmp_path / "progress.json.tmp").exists()


def test_progress_load_handles_missing(tmp_path):
    assert load_progress(tmp_path / "nope.json") == {"attempts": {}}


# ---- 6. result append no dup (events append) ---- #
def test_append_event_no_duplicate(tmp_path):
    evp = tmp_path / "events.jsonl"
    matrix.append_event_atomic(evp, {"event": "game", "game_id": "g1"})
    matrix.append_event_atomic(evp, {"event": "game", "game_id": "g1"})
    matrix.append_event_atomic(evp, {"event": "game", "game_id": "g2"})
    lines = [l for l in evp.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [json.loads(l)["game_id"] for l in lines]
    # no duplicate game_id entries
    assert len(ids) == len(set(ids)) == 2


# ---- 7 + 8. classify: valid/invalid; AI crash is NOT a valid win ---- #
def test_classify_valid_win_camp0():
    rec = classify_game(FakeAtt(game_id="m_rank01_camp0", rank=1, camp=0, normalized_result="win", raw_winner=0))
    assert rec["valid"] is True and rec["normalized_result"] == "win"
    assert rec["rank"] == 1 and rec["camp"] == 0


def test_classify_ai_crash_not_valid_win():
    # Judge may award ifelse the win (raw_winner=camp), but AI crash => invalid, not a capability win
    att = FakeAtt(game_id="m_rank03_camp0", rank=3, camp=0, normalized_result="error",
                  error_type="ai_crash", ai_crash_player=1, raw_winner=0, valid=False)
    rec = classify_game(att)
    assert rec["valid"] is False
    assert rec["normalized_result"] != "win"


# ---- 10. camp-swap winner normalization ---- #
def test_classify_camp_swap_normalization():
    # camp0 raw_winner=0 -> ifelse win; camp1 raw_winner=0 -> ifelse loss
    r0 = classify_game(FakeAtt(game_id="a", rank=1, camp=0, normalized_result="win", raw_winner=0))
    r1 = classify_game(FakeAtt(game_id="b", rank=1, camp=1, normalized_result="loss", raw_winner=0))
    assert r0["normalized_result"] == "win" and r1["normalized_result"] == "loss"
    assert r0["ifelse_camp"] == 0 and r1["ifelse_camp"] == 1


# ---- 11. recovery audit distinguishes not_started/running/done ---- #
def test_progress_states_distinguished():
    plan = make_attempt_plan()
    prog = {"attempts": {}}
    assert matrix.state_of(prog, plan[0]["game_id"]) == "not_started"
    matrix.mark_running(prog, plan[0]["game_id"])
    assert matrix.state_of(prog, plan[0]["game_id"]) == "running"
    matrix.mark_done(prog, plan[0]["game_id"], {"valid": True})
    assert matrix.state_of(prog, plan[0]["game_id"]) == "done"


# ---- 9 + 13. infra error blocks next batch; success/invalid/infra distinct ---- #
def test_should_stop_on_infra_anomalies():
    assert should_stop(FakeAtt(wrapper_timeout=True)) == ("stop", "wrapper_timeout")
    assert should_stop(FakeAtt(error_type="evidence_mismatch"))[0] == "stop"
    assert should_stop(FakeAtt(result_json_status="missing"))[0] == "stop"
    # a clean valid game does NOT stop
    assert should_stop(FakeAtt(normalized_result="win"))[0] != "stop"
    # an AI-crash invalid does NOT stop the matrix (recorded, continue)
    assert should_stop(FakeAtt(normalized_result="error", error_type="ai_crash"))[0] != "stop"


# ---- rank audit ---- #
def test_rank_audit_pass_for_two_clean_games():
    games = [FakeAtt(game_id="m_rank01_camp0", rank=1, camp=0, normalized_result="win"),
             FakeAtt(game_id="m_rank01_camp1", rank=1, camp=1, normalized_result="loss")]
    ok, reasons = rank_audit(1, games)
    assert ok is True, reasons


def test_rank_audit_fail_on_missing_camp():
    games = [FakeAtt(game_id="m_rank01_camp0", rank=1, camp=0, normalized_result="win"),
             FakeAtt(game_id="m_rank01_camp0b", rank=1, camp=0, normalized_result="win")]  # both camp0
    ok, reasons = rank_audit(1, games)
    assert ok is False


# ---- 12. PID residual check wired into stop ---- #
def test_residual_proc_triggers_stop(tmp_path):
    # write a result-json with a fake live-managed-proc identity pointing at this process
    import os, psutil
    me = psutil.Process(os.getpid())
    rj = tmp_path / "r.json"
    rj.write_text(json.dumps({"judge": {"pid": os.getpid(), "started_at": me.create_time(), "role": "judge"}}))
    att = FakeAtt(evidence_paths={"result_json": str(rj)})
    stop, reason = matrix.should_stop_with_residual(att)
    assert stop == "stop" and "residual" in reason


# ---- aggregate ---- #
def test_aggregate_counts_and_winrate():
    recs = []
    for rank in range(1, 4):
        recs.append({"rank": rank, "camp": 0, "valid": True, "normalized_result": "win", "raw_winner": 0, "steps": 40, "realized_randomization": {"map_type": 0, "day_time": 1}, "scores": {"0": 5, "1": 2}, "error_type": None})
        recs.append({"rank": rank, "camp": 1, "valid": True, "normalized_result": "loss", "raw_winner": 0, "steps": 38, "realized_randomization": {"map_type": 1, "day_time": 0}, "scores": {"0": 5, "1": 2}, "error_type": None})
    recs.append({"rank": 3, "camp": 0, "valid": False, "normalized_result": "error", "raw_winner": None, "steps": 5, "realized_randomization": None, "scores": None, "error_type": "ai_crash"})
    agg = aggregate(recs)
    assert agg["total_attempts"] == 7
    assert agg["valid_games"] == 6 and agg["invalid_games"] == 1
    assert agg["wins"] == 3 and agg["losses"] == 3
    assert agg["win_rate"] == pytest.approx(0.5)
    assert agg["per_rank"][1]["win_rate"] == pytest.approx(0.5)  # rank1: camp0 win + camp1 loss
    assert agg["per_rank"][3]["invalid"] == 1


def test_aggregate_no_valid_winrate_null():
    recs = [{"rank": 1, "camp": 0, "valid": False, "normalized_result": "error", "raw_winner": None, "steps": 1, "realized_randomization": None, "scores": None, "error_type": "ai_crash"}]
    agg = aggregate(recs)
    assert agg["win_rate"] is None
