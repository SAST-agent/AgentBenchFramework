"""MiracleEvalRunner tests (阶段4b-6). Uses an injected fake attempt_fn so the
runner logic is exercised without running real matches (real-match behaviour is
covered by match_runner's own tests + smoke).

Verifies:
  * run-level statistics land in the persisted summary (attempted/valid/invalid/
    wins/losses/draws/win_rate_denominator/attempted_steps/total_steps/evaluation_status);
  * win_rate == wins/valid_games and matches an independent recompute from events;
  * valid_games == 0 -> win_rate is null, evaluation_status NO_VALID_GAMES;
  * side-swapped wins both count (raw_winner differs across games);
  * no runs/runs double-dir (risk #6); disk summary (not just in-memory) is correct;
  * Run is driven directly (no BaseRunner).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentbench_frame.games.miracle.match_runner import MatchAttempt
from agentbench_frame.games.miracle.runner import MiracleEvalRunner

SRC = Path(__file__).resolve().parents[2] / "src"


def _mk_att(*, game_id, evaluated_agent_camp, evaluated_agent="ifelse",
            opponent="rank04", normalized="error", error_type="ai_crash",
            raw_winner=None, scores=None, steps=30,
            realized=None, reason="synthetic") -> MatchAttempt:
    valid = normalized in ("win", "loss", "draw")
    if raw_winner is None and valid:
        raw_winner = evaluated_agent_camp if normalized == "win" else (1 - evaluated_agent_camp)
    winner = None
    if valid and raw_winner in (0, 1):
        winner = evaluated_agent if raw_winner == evaluated_agent_camp else opponent
    return MatchAttempt(
        game_id=game_id, evaluated_agent=evaluated_agent, opponent=opponent,
        evaluated_agent_camp=evaluated_agent_camp,
        valid=valid, normalized_result=normalized,
        error_type=(None if valid else error_type), reason=reason,
        raw_winner=raw_winner, winner_agent=winner, scores=scores, steps=steps,
        realized_randomization=realized,
        result_json_status=("ok" if error_type not in ("result_json_missing", "result_json_corrupt") else "missing"),
        discrepancies=[], ai_crash_player=(0 if error_type == "ai_crash" else None),
        ai_timeout_player=(0 if error_type == "ai_timeout" else None),
        judge_crash=(error_type == "judge_crash"),
        wrapper_timeout=(error_type == "wrapper_timeout"),
        normal_cleanup_nonzero=False,
        evidence_paths={"stdout": "", "stderr": "", "trace": "", "replay": "", "result_json": ""},
        collision_detected=False, process_cleanup=[], vendor_returncode=0,
        started_at=0.0, finished_at=1.0, duration_s=1.0, exception=None,
    )


def _runner_with_plan(tmp_path: Path, plan: List[Dict[str, Any]], *, n_games=None):
    counter = {"i": 0}
    n = n_games if n_games is not None else len(plan)

    def fake(*, game_id, evaluated_agent_camp, evaluated_agent, opponent, **_):
        i = counter["i"]
        counter["i"] += 1
        p = plan[i % len(plan)]
        return _mk_att(
            game_id=game_id, evaluated_agent_camp=evaluated_agent_camp,
            evaluated_agent=evaluated_agent, opponent=opponent, **p,
        )

    return MiracleEvalRunner(
        agent="ifelse", data_dir=str(tmp_path), judge_dir=tmp_path / "judge",
        vendor_script=tmp_path / "vendor.py", framework_src=str(SRC),
        evaluated_dir=tmp_path / "eval", opponent_dir=tmp_path / "opp", n_games=n,
        opponent="rank04", work_dir=tmp_path / "work", attempt_fn=fake,
        config={"source": "test"},
    )


def test_mixed_run_statistics_and_consistency(tmp_path):
    plan = [
        {"normalized": "win", "scores": {"0": 5, "1": 2}, "steps": 40, "realized": {"map_type": 0, "day_time": 1}},
        {"normalized": "win", "scores": {"0": 2, "1": 5}, "steps": 38, "realized": {"map_type": 1, "day_time": 0}},
        {"normalized": "loss", "scores": {"0": 2, "1": 5}, "steps": 50, "realized": {"map_type": 0, "day_time": 0}},
        {"normalized": "error", "error_type": "ai_crash", "reason": "ai crash", "steps": 10},
    ]
    runner = _runner_with_plan(tmp_path, plan)
    summary = runner.run()
    # persisted on disk
    run_dir = tmp_path / "runs" / "24_miracle" / "ifelse" / runner_run_id(runner, summary)
    disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for k in ("attempted_games", "valid_games", "invalid_games", "wins", "losses",
              "draws", "win_rate_denominator", "attempted_steps", "total_steps",
              "evaluation_status", "win_rate_available", "h2h"):
        assert k in disk, f"summary missing {k}"
    assert disk["attempted_games"] == 4
    assert disk["valid_games"] == 3 and disk["invalid_games"] == 1
    assert (disk["wins"], disk["losses"], disk["draws"]) == (2, 1, 0)
    assert disk["win_rate_denominator"] == 3
    assert disk["win_rate"] == pytest.approx(2 / 3)
    assert disk["attempted_steps"] == 138 and disk["total_steps"] == 128
    assert disk["evaluation_status"] == "COMPLETE" and disk["win_rate_available"] is True
    assert disk["total_episodes"] == 3                      # == valid_games
    assert disk["win_rate"] == pytest.approx(summary["_recompute_check"]["win_rate"])
    # no double-runs
    assert not (tmp_path / "runs" / "runs").exists()


def test_no_valid_games_win_rate_null(tmp_path):
    plan = [{"normalized": "error", "error_type": "ai_crash", "reason": "x", "steps": 5}]
    runner = _runner_with_plan(tmp_path, plan, n_games=2)
    summary = runner.run()
    assert summary["valid_games"] == 0
    assert summary["win_rate"] is None
    assert summary["evaluation_status"] == "NO_VALID_GAMES"
    assert summary["win_rate_available"] is False
    assert summary["_recompute_check"]["win_rate"] is None


def test_side_swap_both_wins_count(tmp_path):
    plan = [{"normalized": "win", "scores": {"0": 5, "1": 2}, "steps": 40,
             "realized": {"map_type": 0, "day_time": 0}}]
    runner = _runner_with_plan(tmp_path, plan, n_games=2)  # camp0 then camp1
    summary = runner.run()
    assert summary["wins"] == 2
    assert summary["win_rate"] == pytest.approx(1.0)


def runner_run_id(runner, summary) -> str:
    return summary["run_id"]
