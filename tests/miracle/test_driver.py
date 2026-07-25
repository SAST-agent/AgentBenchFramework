"""Run-contract tests for the 24_miracle driver (SKILL.md 测试门槛 items 11 & 12,
plus the disk-level side-swap correctness that defeats framework risks #2/#3/#5).

These instantiate the real framework ``Run`` (no Judge, no subprocess) and
assert that the persisted ``run.toml`` / ``summary.json`` are contract-complete
and that win_rate is correct on disk after side-swapping.

    py -3.13 -m pytest tests/miracle/test_driver.py -v
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.driver import feed_outcomes_to_run
from agentbench_frame.games.miracle.result import (
    GameOutcome,
    compute_h2h,
    compute_win_rate,
    finalize,
)
from agentbench_frame.tracking.run import Run


# ---- helpers ------------------------------------------------------------- #
def _outcome(gid, eval_camp, raw_winner, steps=40, score0=None, score1=None, **kw):
    s0 = score0 if score0 is not None else (5 if raw_winner == 0 else 2)
    s1 = score1 if score1 is not None else (2 if raw_winner == 0 else 5)
    return finalize(
        GameOutcome(
            game_id=gid,
            evaluated_agent="ifelse",
            opponent="rank04",
            evaluated_agent_camp=eval_camp,
            raw_winner=raw_winner,
            score0=s0,
            score1=s1,
            steps=steps,
            replay_path=f"/tmp/{gid}.replay",
            replay_sha256="x" * 64,
            **kw,
        )
    )


def _build_run(tmp_path: Path, outcomes, agent="ifelse", run_type="eval"):
    run = Run.start(
        game="24_miracle",
        agent=agent,
        run_type=run_type,
        data_dir=str(tmp_path),
        config={"opponent_set": "smoke", "n_games": str(len(outcomes))},
    )
    feed_outcomes_to_run(run, outcomes)
    summary = run.finish()
    run_dir = tmp_path / "runs" / "24_miracle" / agent / run.run_id
    return run, summary, run_dir


# ---- item 11: run.toml fields -------------------------------------------- #
def test_run_toml_has_required_fields(tmp_path):
    _, _, run_dir = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    meta = tomllib.loads((run_dir / "run.toml").read_text())
    run_section = meta["run"]
    for field in ("run_id", "game", "agent", "type", "created",
                  "started_at", "total_steps", "total_episodes"):
        assert field in run_section, f"run.toml missing [run].{field}"
    assert run_section["game"] == "24_miracle"
    assert run_section["agent"] == "ifelse"
    assert run_section["type"] == "eval"
    assert meta["config"]["opponent_set"] == "smoke"


def test_run_toml_finished_at_written(tmp_path):
    _, _, run_dir = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    meta = tomllib.loads((run_dir / "run.toml").read_text())
    assert "finished_at" in meta["run"]  # only after finish()


# ---- item 12: summary.json fields ---------------------------------------- #
REQUIRED_SUMMARY_FIELDS = (
    "run_id", "game", "agent", "run_type", "created", "git_commit",
    "wall_hours", "total_episodes", "total_steps", "win_rate",
    "best_elo", "final_elo", "elo_history", "h2h", "resource_summary", "config",
)


def test_summary_json_has_required_fields(tmp_path):
    _, summary, run_dir = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    disk = json.loads((run_dir / "summary.json").read_text())
    for f in REQUIRED_SUMMARY_FIELDS:
        assert f in disk, f"summary.json missing {f}"
    # eval run: Elo fields present but null, NOT omitted (SKILL.md)
    assert disk["best_elo"] is None
    assert disk["final_elo"] is None
    assert disk["elo_history"] == []


# ---- disk-level side-swap correctness (risks #2/#3/#5) ------------------- #
def test_persisted_win_rate_correct_under_side_swap(tmp_path):
    outcomes = [
        _outcome("g1", 0, 0, steps=40),   # win as camp0
        _outcome("g2", 1, 1, steps=38),   # win as camp1 (swapped)
        _outcome("g3", 0, 1, steps=50),   # loss
    ]
    _, summary, run_dir = _build_run(tmp_path, outcomes)
    expected = compute_win_rate(outcomes)          # 2/3
    # in-memory return is correct...
    assert summary["win_rate"] == pytest.approx(expected)
    # ...AND the value actually persisted to disk (this is the risk-#5 check)
    disk = json.loads((run_dir / "summary.json").read_text())
    assert disk["win_rate"] == pytest.approx(expected)
    assert disk["total_episodes"] == 3
    assert disk["total_steps"] == 128              # 40 + 38 + 50 (valid games only)


def test_error_games_excluded_from_persisted_counts(tmp_path):
    outcomes = [
        _outcome("g1", 0, 0, steps=40),                         # win
        _outcome("g2", 0, 0, steps=10, ai_error_player=1),      # error (opp crash)
    ]
    _, _, run_dir = _build_run(tmp_path, outcomes)
    disk = json.loads((run_dir / "summary.json").read_text())
    # only the valid game counts toward episodes/steps/win_rate
    assert disk["total_episodes"] == 1
    assert disk["total_steps"] == 40
    assert disk["win_rate"] == pytest.approx(1.0)
    # but the error game IS still in events.jsonl as an audit record
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    game_events = [e for e in events if e.get("event") == "game"]
    assert len(game_events) == 2
    assert any(e["normalized_result"] == "error" for e in game_events)


def test_no_runs_runs_double_dir(tmp_path):
    # risk #6: Run.start must write <data_root>/runs/... NOT <data_root>/runs/runs/...
    _, _, run_dir = _build_run(tmp_path, [_outcome("g1", 0, 0)])
    assert run_dir.exists()
    # the forbidden double-runs path must NOT exist
    assert not (tmp_path / "runs" / "runs").exists()


# ---- persisted h2h (item 13 integration) --------------------------------- #
def test_summary_h2h_matches_compute(tmp_path):
    outcomes = [
        _outcome("g1", 0, 0),
        _outcome("g2", 1, 1),
        _outcome("g3", 0, 1),
    ]
    _, _, run_dir = _build_run(tmp_path, outcomes)
    disk = json.loads((run_dir / "summary.json").read_text())
    expected = compute_h2h(outcomes)
    assert disk["h2h"] == expected
    assert disk["h2h"]["ifelse"]["rank04"] == pytest.approx(2 / 3)


# ---- NEW framework defect: Run._write_toml must escape backslashes -------- #
def test_run_toml_valid_with_windows_path_in_config(tmp_path):
    """A config value containing a Windows path (backslashes) must NOT produce
    invalid TOML. This is the root cause of the smoke `data check` failure
    (run.toml 'Invalid hex value' on judge_dir_resolved). Run._write_toml must
    escape backslashes and quotes in string values."""
    run = Run.start(
        game="24_miracle", agent="x", run_type="eval", data_dir=str(tmp_path),
        config={"judge_dir_resolved": r"C:\Users\example\judge_dev_logic"},
    )
    run.log_episode(reward=1.0, steps=5, winner=0)
    run.finish()
    run_dir = tmp_path / "runs" / "24_miracle" / "x" / run.run_id
    # run.toml must parse cleanly and round-trip the path
    meta = tomllib.loads((run_dir / "run.toml").read_text(encoding="utf-8"))
    assert meta["config"]["judge_dir_resolved"] == r"C:\Users\example\judge_dev_logic"


# ---- _write_toml escape regression: quotes / backslashes / mixed types ---- #
def _run_with_config(tmp_path, config):
    run = Run.start(game="24_miracle", agent="c", run_type="eval",
                    data_dir=str(tmp_path), config=config)
    run.log_episode(reward=1.0, steps=1, winner=0)
    run.finish()
    run_dir = tmp_path / "runs" / "24_miracle" / "c" / run.run_id
    return tomllib.loads((run_dir / "run.toml").read_text(encoding="utf-8"))


def test_run_toml_config_double_quotes_roundtrip(tmp_path):
    meta = _run_with_config(tmp_path, {"note": 'he said "hi"'})
    assert meta["config"]["note"] == 'he said "hi"'


def test_run_toml_config_backslashes_roundtrip(tmp_path):
    meta = _run_with_config(tmp_path, {"p": "a\\b\\c"})
    assert meta["config"]["p"] == "a\\b\\c"


def test_run_toml_config_mixed_types_roundtrip(tmp_path):
    meta = _run_with_config(tmp_path, {"s": "plain", "b": True, "i": 7, "f": 1.5,
                                       "path": r"C:\x\y"})
    c = meta["config"]
    assert c["s"] == "plain" and c["b"] is True and c["i"] == 7 and c["f"] == 1.5
    assert c["path"] == r"C:\x\y"


def test_run_toml_normal_string_not_over_escaped(tmp_path):
    # a plain ASCII value must round-trip unchanged (no double-escaping)
    meta = _run_with_config(tmp_path, {"group": "GROUP1"})
    assert meta["config"]["group"] == "GROUP1"


def test_run_toml_run_section_roundtrip(tmp_path):
    meta = _run_with_config(tmp_path, {})
    r = meta["run"]
    assert r["game"] == "24_miracle" and r["agent"] == "c" and r["type"] == "eval"
    assert r["total_steps"] == 1 and r["total_episodes"] == 1
    assert r["started_at"] == float(r["started_at"])  # numeric, not a quoted string
