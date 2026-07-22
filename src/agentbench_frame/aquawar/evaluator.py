"""Seat-balanced AquaWar evaluation with AgentBench-standard tracking."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from agentbench_frame.aquawar.match import AquaWarMatchError, run_match
from agentbench_frame.tracking import Run


MatchRunner = Callable[..., dict[str, Any]]
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Opponent:
    name: str
    command: str


@dataclass(frozen=True)
class AquaWarEvaluationResult:
    run_dir: Path
    summary: dict[str, Any]
    matches: list[dict[str, Any]]
    error_count: int


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        result: sum(record["candidate_result"] == result for record in records)
        for result in ("win", "draw", "loss", "error")
    }
    valid = counts["win"] + counts["draw"] + counts["loss"]
    turns = [record["turns"] for record in records if "turns" in record]
    return {
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "errors": counts["error"],
        "valid_games": valid,
        "attempted_games": len(records),
        "score_rate": (
            (counts["win"] + 0.5 * counts["draw"]) / valid if valid else None
        ),
        "avg_turns": sum(turns) / len(turns) if turns else None,
    }


class AquaWarEvaluator:
    """Run candidate-versus-pool evaluations through the official logic."""

    def __init__(
        self,
        logic_command: str,
        candidate_name: str,
        candidate_command: str,
        opponents: Sequence[Opponent],
        *,
        pairs: int = 5,
        seats: str = "both",
        timeout: float = 5.0,
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
        if pairs <= 0:
            raise ValueError("pairs must be positive")
        if seats not in {"both", "0", "1"}:
            raise ValueError("seats must be one of: both, 0, 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.logic_command = logic_command
        self.candidate_name = candidate_name
        self.candidate_command = candidate_command
        self.opponents = list(opponents)
        self.pairs = pairs
        self.seats = seats
        self.timeout = timeout
        self.data_dir = data_dir
        self.save_replays = save_replays
        self.save_traces = save_traces
        self.match_runner = match_runner

    def evaluate(self) -> AquaWarEvaluationResult:
        run = Run.start(
            game="25_aquawar",
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
        seat_values = (0, 1) if self.seats == "both" else (int(self.seats),)
        with tempfile.TemporaryDirectory(prefix="agentbench-aquawar-") as temp_dir:
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
                            ai_commands = [opponent.command, opponent.command]
                            ai_commands[candidate_seat] = self.candidate_command
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
                                winner = match["winner"]
                                if winner is None:
                                    candidate_result = "draw"
                                elif winner == candidate_seat:
                                    candidate_result = "win"
                                else:
                                    candidate_result = "loss"
                                record["candidate_result"] = candidate_result
                            except AquaWarMatchError as exc:
                                record.update(
                                    candidate_result="error",
                                    error=str(exc),
                                )
                            record["seconds"] = round(
                                time.monotonic() - started, 6
                            )
                            if self.save_replays and replay_path.exists():
                                record["replay"] = str(
                                    replay_path.relative_to(run_dir)
                                )
                            if trace_path is not None and trace_path.exists():
                                record["trace"] = str(
                                    trace_path.relative_to(run_dir)
                                )

                            records.append(record)
                            checkpoint.write(
                                json.dumps(record, ensure_ascii=False, sort_keys=True)
                                + "\n"
                            )
                            checkpoint.flush()
                            os.fsync(checkpoint.fileno())
                            run.write("aquawar_match", **record)
                            result_name = record["candidate_result"]
                            tracked_winner = {"win": 0, "loss": 1}.get(
                                result_name, -1
                            )
                            run.log_episode(
                                reward={
                                    "win": 1.0,
                                    "draw": 0.5,
                                    "loss": 0.0,
                                    "error": 0.0,
                                }[result_name],
                                steps=record.get("turns", 0),
                                winner=tracked_winner,
                                info={
                                    "opponent": opponent.name,
                                    "pair": pair,
                                    "candidate_seat": candidate_seat,
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
                    name: metrics["score_rate"]
                    for name, metrics in by_opponent.items()
                    if metrics["score_rate"] is not None
                }
            }
        )
        summary = run.finish()
        aggregate = _summarize(records)
        summary["win_rate"] = aggregate["score_rate"]
        summary["aquawar"] = {
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
        return AquaWarEvaluationResult(
            run_dir=run_dir,
            summary=summary,
            matches=records,
            error_count=aggregate["errors"],
        )
