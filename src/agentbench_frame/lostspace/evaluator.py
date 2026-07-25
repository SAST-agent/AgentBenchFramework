"""Seat-balanced LostSpace evaluation with AgentBench-standard tracking.

Mirrors ``agentbench_frame.aquawar.evaluator`` but generalised to the 4-player
free-for-all table:

* the candidate occupies one of four seats (rotated when ``seats='all'``);
* each opponent gets its own bracket: candidate + that opponent + two filler
  seats, so head-to-head win rates stay meaningful;
* results are ranked (4 = 1st ... 1 = 4th); the candidate "wins" by ranking
  first, and we also report average rank and average rank-points.

Output layout (unchanged from AquaWar / the rest of the framework):

    $AGENTBENCH_DATA/runs/25_lostspace/{agent}/{run_id}/
        run.toml, summary.json, events.jsonl, matches.jsonl, artifacts/...
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from agentbench_frame.lostspace.match import LostSpaceMatchError, run_match
from agentbench_frame.tracking import Run


MatchRunner = Callable[..., dict[str, Any]]
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
GAME = "25_lostspace"
NUM_SEATS = 4


@dataclass(frozen=True)
class Opponent:
    name: str
    command: str


@dataclass(frozen=True)
class LostSpaceEvaluationResult:
    run_dir: Path
    summary: dict[str, Any]
    matches: list[dict[str, Any]]
    error_count: int


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        result: sum(record["candidate_result"] == result for record in records)
        for result in ("win", "loss", "error")
    }
    valid = counts["win"] + counts["loss"]
    ranks = [record["candidate_rank"] for record in records if "candidate_rank" in record]
    scores = [record["candidate_score"] for record in records if record.get("candidate_score") is not None]
    turns = [record["turns"] for record in records if "turns" in record]
    return {
        "wins": counts["win"],
        "losses": counts["loss"],
        "errors": counts["error"],
        "valid_games": valid,
        "attempted_games": len(records),
        "win_rate": (counts["win"] / valid) if valid else None,
        "avg_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "avg_turns": (sum(turns) / len(turns)) if turns else None,
    }


class LostSpaceEvaluator:
    """Run candidate-versus-pool evaluations through the official 4-player logic."""

    def __init__(
        self,
        logic_command: str,
        candidate_name: str,
        candidate_command: str,
        opponents: Sequence[Opponent],
        filler_command: str,
        *,
        pairs: int = 5,
        seats: str = "all",
        timeout: float = 10.0,
        data_dir: str | Path | None = None,
        save_replays: bool = False,
        save_traces: bool = False,
        match_runner: MatchRunner = run_match,
    ):
        if not logic_command.strip():
            raise ValueError("logic command must not be empty")
        if not SAFE_NAME.fullmatch(candidate_name):
            raise ValueError(
                "candidate name must contain only letters, digits, '.', '_', or '-'"
            )
        if not candidate_command.strip():
            raise ValueError("candidate command must not be empty")
        if not opponents:
            raise ValueError("at least one opponent is required")
        for opponent in opponents:
            if not SAFE_NAME.fullmatch(opponent.name):
                raise ValueError(
                    "opponent name must contain only letters, digits, '.', '_', or '-'"
                )
            if not opponent.command.strip():
                raise ValueError("opponent command must not be empty")
        opponent_names = [opponent.name for opponent in opponents]
        if len(set(opponent_names)) != len(opponent_names):
            raise ValueError("opponent names must be unique")
        if not filler_command.strip():
            raise ValueError("filler command must not be empty")
        if pairs <= 0:
            raise ValueError("pairs must be positive")
        if seats not in {"all", "0", "1", "2", "3"}:
            raise ValueError("seats must be one of: all, 0, 1, 2, 3")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.logic_command = logic_command
        self.candidate_name = candidate_name
        self.candidate_command = candidate_command
        self.opponents = list(opponents)
        self.filler_command = filler_command
        self.pairs = pairs
        self.seats = seats
        self.timeout = timeout
        self.data_dir = data_dir
        self.save_replays = save_replays
        self.save_traces = save_traces
        self.match_runner = match_runner

    def _seat_values(self) -> tuple[int, ...]:
        return tuple(range(NUM_SEATS)) if self.seats == "all" else (int(self.seats),)

    def _build_table(self, candidate_seat: int, opponent: Opponent, pair: int) -> list[str]:
        """candidate @ candidate_seat, opponent @ a rotated other seat, rest filler."""
        table: list[str] = [self.filler_command] * NUM_SEATS
        table[candidate_seat] = self.candidate_command
        other_seats = [s for s in range(NUM_SEATS) if s != candidate_seat]
        opponent_seat = other_seats[pair % len(other_seats)]
        table[opponent_seat] = opponent.command
        return table

    def evaluate(self) -> LostSpaceEvaluationResult:
        run = Run.start(
            game=GAME,
            agent=self.candidate_name,
            run_type="eval",
            data_dir=str(self.data_dir) if self.data_dir is not None else None,
            config={
                "pairs": self.pairs,
                "seats": self.seats,
                "timeout": self.timeout,
                "opponents": ",".join(opponent.name for opponent in self.opponents),
            },
        )
        run_dir = Path(run.run_dir)
        checkpoint_path = run_dir / "matches.jsonl"
        artifact_dir = run_dir / "artifacts"
        if self.save_replays or self.save_traces:
            artifact_dir.mkdir(exist_ok=True)

        records: list[dict[str, Any]] = []
        seat_values = self._seat_values()
        with tempfile.TemporaryDirectory(prefix="agentbench-lostspace-") as temp_dir:
            temp_root = Path(temp_dir)
            with checkpoint_path.open("a") as checkpoint:
                for opponent in self.opponents:
                    for pair in range(self.pairs):
                        for candidate_seat in seat_values:
                            stem = (
                                f"{opponent.name}-pair{pair:03d}"
                                f"-seat{candidate_seat}"
                            )
                            replay_path = (
                                artifact_dir / f"{stem}.json"
                                if self.save_replays
                                else temp_root / f"{stem}.json"
                            )
                            trace_path = (
                                artifact_dir / f"{stem}.trace.jsonl"
                                if self.save_traces
                                else None
                            )
                            ai_commands = self._build_table(candidate_seat, opponent, pair)
                            record: dict[str, Any] = {
                                "opponent": opponent.name,
                                "pair": pair,
                                "candidate_seat": candidate_seat,
                            }
                            started = time.monotonic()
                            try:
                                match = self.match_runner(
                                    self.logic_command,
                                    ai_commands,
                                    self.timeout,
                                    replay_path,
                                    trace_path=trace_path,
                                )
                                record.update(match)
                                ranking = match["ranking"]
                                candidate_rank = ranking.index(candidate_seat) + 1
                                candidate_score = match["end_info"].get(
                                    str(candidate_seat)
                                )
                                record["candidate_rank"] = candidate_rank
                                record["candidate_score"] = candidate_score
                                record["candidate_result"] = (
                                    "win" if candidate_rank == 1 else "loss"
                                )
                            except LostSpaceMatchError as exc:
                                record.update(
                                    candidate_result="error",
                                    error=str(exc),
                                )
                            record["seconds"] = round(time.monotonic() - started, 6)
                            if self.save_replays and replay_path.exists():
                                record["replay"] = replay_path.relative_to(run_dir).as_posix()
                            if trace_path is not None and trace_path.exists():
                                record["trace"] = trace_path.relative_to(run_dir).as_posix()

                            records.append(record)
                            checkpoint.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True)
                                + "\n"
                            )
                            checkpoint.flush()
                            os.fsync(checkpoint.fileno())
                            run.write("lostspace_match", **record)
                            result_name = record["candidate_result"]
                            tracked_winner = 0 if result_name == "win" else 1
                            run.log_episode(
                                reward=1.0 if result_name == "win" else 0.0,
                                steps=record.get("turns", 0),
                                winner=tracked_winner,
                                info={
                                    "opponent": opponent.name,
                                    "pair": pair,
                                    "candidate_seat": candidate_seat,
                                    "candidate_rank": record.get("candidate_rank"),
                                    "result": result_name,
                                },
                            )

        by_opponent = {
            opponent.name: _summarize(
                [
                    record
                    for record in records
                    if record["opponent"] == opponent.name
                ]
            )
            for opponent in self.opponents
        }
        run.log_h2h(
            {
                self.candidate_name: {
                    name: metrics["win_rate"]
                    for name, metrics in by_opponent.items()
                    if metrics["win_rate"] is not None
                }
            }
        )
        summary = run.finish()
        aggregate = _summarize(records)
        summary["win_rate"] = aggregate["win_rate"]
        summary["lostspace"] = {
            "aggregate": aggregate,
            "by_opponent": by_opponent,
            "by_seat": {
                str(seat): _summarize(
                    [
                        record
                        for record in records
                        if record["candidate_seat"] == seat
                    ]
                )
                for seat in seat_values
            },
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        return LostSpaceEvaluationResult(
            run_dir=run_dir,
            summary=summary,
            matches=records,
            error_count=aggregate["errors"],
        )
