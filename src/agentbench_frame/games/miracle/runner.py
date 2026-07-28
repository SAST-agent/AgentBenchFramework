"""MiracleEvalRunner — top-level evaluation driver for 24_miracle.

Does NOT inherit ``BaseRunner`` (that path has the win-attribution / run_type /
summary-ordering bugs — risks #2/#3/#4/#5/#6). Instead it drives ``Run`` directly:

    Run.start(game="24_miracle", agent=…, run_type="eval", data_dir=<root>)
      → run each game via match_runner.run_match_attempt (side-swapped)
      → map each MatchAttempt to a GameOutcome (match_runner's classification wins)
      → feed_outcomes_to_run (events for ALL attempts; log_episode for valid only)
      → log_h2h
      → Run.finish()
      → atomically enrich summary.json with run-level statistics
      → re-read disk summary + independently recompute from events.jsonl, assert equal

When ``valid_games == 0`` the persisted ``win_rate`` is ``null`` and
``evaluation_status`` is ``NO_VALID_GAMES`` (never a fabricated 0% conclusion).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentbench_frame.games.miracle.driver import feed_outcomes_to_run
from agentbench_frame.games.miracle.match_runner import MatchAttempt, run_match_attempt
from agentbench_frame.games.miracle.result import (
    DRAW,
    VALID_RESULTS,
    GameOutcome,
    compute_h2h,
    compute_run_stats,
)
from agentbench_frame.tracking.run import Run

GAME_ID = "24_miracle"


def attempt_to_outcome(att: MatchAttempt) -> GameOutcome:
    """Carry a MatchAttempt into a GameOutcome. match_runner.classify sees the
    full timeline (end_info order, cleanup vs crash, evidence consistency) so its
    normalized_result/valid are authoritative here — we do not re-run normalize."""
    scores = att.scores or {}
    o = GameOutcome(
        game_id=att.game_id,
        evaluated_agent=att.evaluated_agent,
        opponent=att.opponent,
        evaluated_agent_camp=att.evaluated_agent_camp,
        raw_winner=att.raw_winner,
        score0=scores.get("0") if isinstance(scores, dict) else None,
        score1=scores.get("1") if isinstance(scores, dict) else None,
        ai_error_player=att.ai_crash_player,
        ai_timeout_player=att.ai_timeout_player,
        judge_ok=(att.result_json_status == "ok" and not att.judge_crash
                  and not att.wrapper_timeout),
        replay_ok=(att.realized_randomization is not None),
        realized_randomization=att.realized_randomization,
        steps=att.steps,
        duration_s=att.duration_s,
        started_at=att.started_at,
        finished_at=att.finished_at,
        replay_path=att.evidence_paths.get("replay"),
        is_resume=att.collision_detected,
        exception=(att.reason if not att.valid else None),
    )
    o.normalized_result = att.normalized_result
    o.valid = att.valid
    o.draw = (att.normalized_result == DRAW)
    o.winner_agent = att.winner_agent
    o.score_tie = (o.score0 is not None and o.score1 is not None and o.score0 == o.score1)
    o.judge_tiebreak_applied = o.score_tie
    return o


def enrich_summary_atomically(run_dir: Path, outcomes: List[GameOutcome]) -> Dict[str, Any]:
    """Merge run-level statistics into summary.json with an atomic temp+replace."""
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = compute_run_stats(outcomes)
    summary.update(stats)
    summary["win_rate"] = stats["win_rate"]            # None when valid_games == 0
    summary["win_rate_available"] = (stats["valid_games"] > 0)
    summary["h2h"] = compute_h2h(outcomes)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, summary_path)
    return summary


def recompute_from_events(run_dir: Path) -> Dict[str, Any]:
    """Independently recompute win_rate / counts straight from events.jsonl."""
    games: List[dict] = []
    p = run_dir / "events.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "game":
                games.append(e)
    valid = [g for g in games if g.get("normalized_result") in VALID_RESULTS]
    wins = sum(1 for g in valid if g.get("normalized_result") == "win")
    return {
        "attempted_games": len(games),
        "valid_games": len(valid),
        "win_rate": (wins / len(valid)) if valid else None,
    }


class MiracleEvalRunner:
    def __init__(self, *, agent: str, data_dir: str, judge_dir, vendor_script,
                 framework_src, evaluated_dir, opponent_dir, n_games: int,
                 opponent: str = "opponent",
                 timeout: float = 12.0, wrapper_timeout_s: float = 60.0, work_dir,
                 config: Optional[Dict[str, Any]] = None,
                 attempt_fn: Optional[Callable] = None, prefix: str = "smoke"):
        self.agent = agent
        self.data_dir = data_dir
        self.judge_dir = judge_dir
        self.vendor_script = vendor_script
        self.framework_src = framework_src
        self.evaluated_dir = evaluated_dir
        self.opponent_dir = opponent_dir
        self.n_games = n_games
        self.opponent = opponent
        self.timeout = timeout
        self.wrapper_timeout_s = wrapper_timeout_s
        self.work_dir = work_dir
        self.config = config or {}
        self.prefix = prefix
        self.attempt_fn = attempt_fn or self._default_attempt_fn

    def _default_attempt_fn(self, *, game_id, evaluated_agent_camp,
                            evaluated_agent, opponent, **_):
        # TRUE side-swap: the evaluated agent is player0 on camp0 games and
        # player1 on camp1 games, so the Judge sees both camp assignments.
        if evaluated_agent_camp == 0:
            p0_dir, p1_dir = self.evaluated_dir, self.opponent_dir
            p0_name, p1_name = evaluated_agent, opponent
        else:
            p0_dir, p1_dir = self.opponent_dir, self.evaluated_dir
            p0_name, p1_name = opponent, evaluated_agent
        return run_match_attempt(
            game_id=game_id, p0_dir=p0_dir, p1_dir=p1_dir,
            p0_name=p0_name, p1_name=p1_name,
            judge_dir=self.judge_dir, work_dir=self.work_dir,
            vendor_script=self.vendor_script, framework_src=self.framework_src,
            timeout=self.timeout, wrapper_timeout_s=self.wrapper_timeout_s,
            evaluated_agent_camp=evaluated_agent_camp,
            evaluated_agent=evaluated_agent, opponent=opponent,
        )

    def run(self) -> Dict[str, Any]:
        run = Run.start(game=GAME_ID, agent=self.agent, run_type="eval",
                        data_dir=self.data_dir, config=self.config)
        outcomes: List[GameOutcome] = []
        attempts: List[MatchAttempt] = []
        for i in range(self.n_games):
            eval_camp = i % 2  # alternate sides
            game_id = f"{self.prefix}_{i:02d}_camp{eval_camp}"
            att = self.attempt_fn(
                game_id=game_id, evaluated_agent_camp=eval_camp,
                evaluated_agent=self.agent, opponent=self.opponent,
            )
            attempts.append(att)
            outcomes.append(attempt_to_outcome(att))
        self.attempts = attempts

        feed_outcomes_to_run(run, outcomes)
        run.finish()

        run_dir = Path(run.run_dir)
        summary = enrich_summary_atomically(run_dir, outcomes)

        # cross-check: re-read disk summary and recompute from events independently
        disk = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        recompute = recompute_from_events(run_dir)
        wr_disk, wr_recompute = disk.get("win_rate"), recompute["win_rate"]
        if wr_disk is None or wr_recompute is None:
            assert wr_disk is wr_recompute, f"win_rate None-mismatch: {wr_disk} vs {wr_recompute}"
        else:
            assert abs(wr_disk - wr_recompute) < 1e-9, \
                f"win_rate mismatch: disk={wr_disk} recompute={wr_recompute}"
        assert disk["total_episodes"] == recompute["valid_games"]
        # guard against the double-runs path bug (risk #6): run_dir must be
        # <data_dir>/runs/24_miracle/<agent>/<run_id>, never .../runs/runs/...
        rel = run_dir.relative_to(self.data_dir)
        assert rel.parts == ("runs", GAME_ID, self.agent, run.run_id), \
            f"unexpected run path shape (risk #6): {rel}"
        summary["_recompute_check"] = recompute
        return summary
