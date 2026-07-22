"""Contract tests for the 24_miracle pure result module.

Covers SKILL.md 测试门槛 items that are pure logic (no subprocess, no Judge):
  1. Agent wins at camp0
  2. Agent wins at camp1
  3. side-swap tallying (raw_winner==0 is NOT always the evaluated agent's win)
  4. draw recorded separately, not a win
  5. opponent crash -> error (not a capability win)
  6. agent crash -> error
  7. judge crash -> error
  8. timeout -> error
  9. replay missing -> error
 10. duplicate game_id resumability
 13. h2h direction
plus end_info parsing, replay-header parsing, file hashing, and event-record
contract completeness.

Run with the 3.11+ interpreter:
    py -3.13 -m pytest tests/miracle/test_result.py -v
"""
from __future__ import annotations

import json
import struct

import pytest

from agentbench_frame.games.miracle.result import (
    DRAW,
    ERROR,
    GameOutcome,
    LOSS,
    VALID_RESULTS,
    WIN,
    build_seed_provenance,
    compute_h2h,
    compute_run_stats,
    compute_win_rate,
    derive_raw_winner,
    finalize,
    normalize,
    outcome_counts,
    read_replay_header,
    scores_from_end_info,
    select_games_to_run,
    sha256_file,
    to_event_record,
    would_rerun_successful,
)


# ---- helpers ------------------------------------------------------------- #
def outcome(eval_camp=0, raw_winner=None, **kw):
    """Build a finalized GameOutcome with sane defaults for tests."""
    o = GameOutcome(
        game_id=kw.pop("game_id", "g1"),
        evaluated_agent=kw.pop("evaluated_agent", "ifelse"),
        opponent=kw.pop("opponent", "rank04"),
        evaluated_agent_camp=eval_camp,
        raw_winner=raw_winner,
        **kw,
    )
    return finalize(o)


# ---- 1 & 2: win at camp0 / camp1 ----------------------------------------- #
def test_win_at_camp0():
    o = outcome(eval_camp=0, raw_winner=0)
    assert o.normalized_result == WIN
    assert o.winner_agent == "ifelse"


def test_win_at_camp1():
    # evaluated agent on camp 1, Judge says camp 1 won -> evaluated agent wins
    o = outcome(eval_camp=1, raw_winner=1)
    assert o.normalized_result == WIN
    assert o.winner_agent == "ifelse"


def test_loss_at_camp1_when_raw_winner_zero():
    # side-swapped: evaluated on camp1, raw_winner==0 (opponent camp) -> LOSS,
    # NOT a win. This is the exact bug the framework's Match has.
    o = outcome(eval_camp=1, raw_winner=0)
    assert o.normalized_result == LOSS
    assert o.winner_agent == "rank04"


# ---- 3: side-swap tallying ----------------------------------------------- #
def test_side_swap_win_rate_counts_both_sides():
    # two wins, one as camp0, one as camp1 (swapped) -> 100%
    games = [
        outcome(game_id="a", eval_camp=0, raw_winner=0),
        outcome(game_id="b", eval_camp=1, raw_winner=1),
    ]
    assert compute_win_rate(games) == 1.0


def test_side_swap_raw_winner_zero_is_not_always_eval_win():
    # raw_winner==0 once means eval win (camp0) and once means eval loss (camp1)
    games = [
        outcome(game_id="a", eval_camp=0, raw_winner=0),  # win
        outcome(game_id="b", eval_camp=1, raw_winner=0),  # loss (swapped)
    ]
    assert compute_win_rate(games) == pytest.approx(0.5)
    # naive "raw_winner==0 => win" would wrongly give 1.0


# ---- 4: draw ------------------------------------------------------------- #
def test_draw_recorded_not_a_win():
    # Miracle Judge never draws; simulate a hypothetical draw (raw_winner=-1).
    games = [
        outcome(game_id="a", eval_camp=0, raw_winner=0),          # win
        finalize(GameOutcome(game_id="b", evaluated_agent="ifelse",
                             opponent="rank04", evaluated_agent_camp=0,
                             raw_winner=-1)),                      # draw
    ]
    counts = outcome_counts(games)
    assert counts["win"] == 1 and counts["draw"] == 1
    # draw is a valid game but not a win -> win_rate = 1/2
    assert compute_win_rate(games) == pytest.approx(0.5)


# ---- 5-9: anomaly -> error, never a capability win ----------------------- #
def test_opponent_crash_is_error_not_win():
    # Judge gave camp0 the win (raw_winner=0, evaluated on camp0), but the
    # OPPONENT crashed -> invalid evidence, must NOT count as a capability win.
    o = outcome(eval_camp=0, raw_winner=0, ai_error_player=1)
    assert o.normalized_result == ERROR
    assert o.valid is False


def test_agent_crash_is_error():
    o = outcome(eval_camp=0, raw_winner=1, ai_error_player=0)
    assert o.normalized_result == ERROR


def test_judge_crash_is_error():
    o = outcome(eval_camp=0, judge_ok=False)
    assert o.normalized_result == ERROR
    assert o.raw_winner is None or o.normalized_result == ERROR


def test_timeout_is_error():
    o = outcome(eval_camp=0, raw_winner=0, ai_timeout_player=0)
    assert o.normalized_result == ERROR
    o2 = outcome(eval_camp=0, raw_winner=1, ai_timeout_player=1)
    assert o2.normalized_result == ERROR


def test_replay_missing_is_error():
    o = outcome(eval_camp=0, raw_winner=0, replay_ok=False)
    assert o.normalized_result == ERROR


def test_error_games_excluded_from_win_rate():
    games = [
        outcome(game_id="ok", eval_camp=0, raw_winner=0),               # win
        outcome(game_id="crash", eval_camp=0, raw_winner=0,
                ai_error_player=1),                                     # error
    ]
    # only 1 valid game (the win) -> win_rate 1.0, not 0.5
    assert compute_win_rate(games) == 1.0
    assert outcome_counts(games)["error"] == 1


# ---- 10: duplicate game_id resumability ---------------------------------- #
def test_select_games_skips_completed_successful():
    planned = ["g1", "g2", "g3", "g4"]
    done_valid = ["g2"]  # g2 already has a valid result
    assert select_games_to_run(planned, done_valid) == ["g1", "g3", "g4"]


def test_would_rerun_successful_detected():
    done = ["g2"]
    assert would_rerun_successful("g2", done) is True
    assert would_rerun_successful("g1", done) is False


def test_no_valid_games_means_rerun_all():
    planned = ["g1", "g2"]
    assert select_games_to_run(planned, []) == ["g1", "g2"]


# ---- 13: h2h direction --------------------------------------------------- #
def test_h2h_direction_no_draws_sums_to_one():
    # ifelse vs rank04: ifelse won 3, rank04 won 1, across side-swaps
    games = [
        outcome(game_id="1", eval_camp=0, raw_winner=0),   # ifelse win
        outcome(game_id="2", eval_camp=1, raw_winner=1),   # ifelse win (swapped)
        outcome(game_id="3", eval_camp=0, raw_winner=0),   # ifelse win
        outcome(game_id="4", eval_camp=0, raw_winner=1),   # rank04 win
    ]
    h2h = compute_h2h(games)
    assert h2h["ifelse"]["rank04"] == pytest.approx(0.75)
    assert h2h["rank04"]["ifelse"] == pytest.approx(0.25)
    # no draws -> symmetric pair sums to 1
    assert h2h["ifelse"]["rank04"] + h2h["rank04"]["ifelse"] == pytest.approx(1.0)


def test_h2h_with_draw_does_not_sum_to_one():
    games = [
        outcome(game_id="1", eval_camp=0, raw_winner=0),                          # win
        finalize(GameOutcome(game_id="2", evaluated_agent="ifelse",
                             opponent="rank04", evaluated_agent_camp=0,
                             raw_winner=-1)),                                     # draw
    ]
    h2h = compute_h2h(games)
    assert h2h["ifelse"]["rank04"] == pytest.approx(0.5)   # 1 win / 2 valid
    assert h2h["rank04"]["ifelse"] == pytest.approx(0.0)
    assert h2h["ifelse"]["rank04"] + h2h["rank04"]["ifelse"] == pytest.approx(0.5)


def test_h2h_excludes_error_games():
    games = [
        outcome(game_id="1", eval_camp=0, raw_winner=0),                  # win
        outcome(game_id="2", eval_camp=0, raw_winner=0, ai_error_player=1),  # error
    ]
    h2h = compute_h2h(games)
    assert h2h["ifelse"]["rank04"] == pytest.approx(1.0)   # only the valid game counts


# ---- end_info parsing (Judge main.py semantics) -------------------------- #
def test_derive_raw_winner_player0_wins():
    assert derive_raw_winner({"0": 5, "1": 2}) == 0


def test_derive_raw_winner_player1_wins_on_tie():
    # Judge breaks ties toward player1 (main.py:448-449)
    assert derive_raw_winner({"0": 3, "1": 3}) == 1
    assert derive_raw_winner({"0": 2, "1": 5}) == 1


def test_derive_raw_winner_none_when_missing():
    assert derive_raw_winner(None) is None
    assert derive_raw_winner({}) is None
    assert derive_raw_winner({"0": 1}) is None
    assert derive_raw_winner({"0": "x", "1": "y"}) is None


def test_scores_from_end_info():
    assert scores_from_end_info({"0": 7, "1": 4}) == (7, 4)
    assert scores_from_end_info(None) == (None, None)


# ---- replay header + file hashing ---------------------------------------- #
def test_read_replay_header(tmp_path):
    # [0,0,0,map_type,day_time,0,0] as big-endian signed int32
    blob = struct.pack(">7i", 0, 0, 0, 1, 0, 0, 0)
    p = tmp_path / "g.replay"
    p.write_bytes(blob)
    assert read_replay_header(p) == {"map_type": 1, "day_time": 0}


def test_read_replay_header_missing(tmp_path):
    assert read_replay_header(tmp_path / "nope.replay") is None


def test_read_replay_header_too_short(tmp_path):
    p = tmp_path / "short.replay"
    p.write_bytes(b"\x00" * 10)
    assert read_replay_header(p) is None


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # known sha256 of "hello"
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_file_missing(tmp_path):
    assert sha256_file(tmp_path / "nope") is None


# ---- event-record contract completeness ---------------------------------- #
def test_event_record_has_all_required_fields():
    o = outcome(game_id="g1", eval_camp=0, raw_winner=0,
                replay_path="/tmp/g1.replay", replay_sha256="abc",
                evaluated_source_sha256="d1", opponent_source_sha256="e1",
                judge_exit=0, ai0_exit=0, ai1_exit=0, steps=42,
                started_at=1.0, finished_at=2.0, duration_s=1.0)
    rec = to_event_record(o)
    missing = [f for f in (
        "game_id", "seed", "policy_ids", "policy_source_sha256", "camps",
        "raw_winner", "winner_agent", "normalized_result", "scores", "draw",
        "started_at", "finished_at", "duration", "judge_exit", "ai0_exit",
        "ai1_exit", "timeout_s", "exception", "replay_path", "replay_sha256",
        "valid", "is_resume", "is_rerun") if f not in rec]
    assert missing == []
    # direction-critical fields must be present and correct
    assert rec["raw_winner"] == 0
    assert rec["winner_agent"] == "ifelse"
    assert rec["normalized_result"] == WIN
    assert rec["evaluated_agent_camp"] == 0
    assert rec["valid"] is True


def test_event_record_seed_provenance_and_serializable():
    o = outcome(game_id="g1", eval_camp=0, raw_winner=0,
                realized_randomization={"map_type": 1, "day_time": 0})
    rec = to_event_record(o)
    # seed field is the structured provenance, NOT map_type/day_time called a seed
    assert rec["seed"] == {
        "requested_seed": None, "effective_seed": None,
        "deterministic_seed_supported": False, "reproducible_from_seed": False,
        "realized_randomization": {"map_type": 1, "day_time": 0},
    }
    # must round-trip through JSON (events.jsonl is newline-delimited JSON)
    s = json.dumps(rec, default=str)
    assert json.loads(s)["game_id"] == "g1"


# ---- run-level statistics ------------------------------------------------ #
def test_compute_run_stats_counts_and_steps():
    outcomes = [
        outcome(game_id="1", eval_camp=0, raw_winner=0, steps=40),   # win
        outcome(game_id="2", eval_camp=1, raw_winner=1, steps=38),   # win (swapped)
        outcome(game_id="3", eval_camp=0, raw_winner=1, steps=50),   # loss
        outcome(game_id="4", eval_camp=0, raw_winner=0, steps=10, ai_error_player=1),  # error
    ]
    stats = compute_run_stats(outcomes)
    assert stats["attempted_games"] == 4
    assert stats["valid_games"] == 3
    assert stats["invalid_games"] == 1
    assert (stats["wins"], stats["losses"], stats["draws"]) == (2, 1, 0)
    assert stats["win_rate_denominator"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["attempted_steps"] == 138   # 40+38+50+10
    assert stats["total_steps"] == 128       # 40+38+50 (valid only)
    assert stats["evaluation_status"] == "COMPLETE"


def test_compute_run_stats_no_valid_games_win_rate_null():
    outcomes = [
        outcome(game_id="1", eval_camp=0, raw_winner=0, steps=10, ai_error_player=1),
        outcome(game_id="2", eval_camp=0, judge_ok=False),
    ]
    stats = compute_run_stats(outcomes)
    assert stats["valid_games"] == 0
    assert stats["win_rate"] is None
    assert stats["evaluation_status"] == "NO_VALID_GAMES"
    assert stats["win_rate_denominator"] == 0


# ---- tie semantics: Judge resolves ties to player1, NOT a draw ----------- #
def test_tie_resolved_to_player1_not_draw():
    o_win = outcome(eval_camp=1, raw_winner=1, score0=4, score1=4)   # evaluated camp1 wins the tiebreak
    assert o_win.score_tie is True and o_win.judge_tiebreak_applied is True
    assert o_win.normalized_result == "win"
    assert o_win.draw is False
    o_loss = outcome(eval_camp=0, raw_winner=1, score0=4, score1=4)  # evaluated camp0 loses the tiebreak
    assert o_loss.normalized_result == "loss"
    assert o_loss.score_tie is True
    assert o_loss.draw is False


def test_build_seed_provenance_structure():
    p = build_seed_provenance({"map_type": 0, "day_time": 1})
    assert p == {
        "requested_seed": None, "effective_seed": None,
        "deterministic_seed_supported": False, "reproducible_from_seed": False,
        "realized_randomization": {"map_type": 0, "day_time": 1},
    }
