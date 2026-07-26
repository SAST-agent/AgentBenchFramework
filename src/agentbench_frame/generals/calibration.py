"""Separate weak-opponent calibration cases and aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from agentbench_frame.eval.benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    GameResult,
    evaluate_benchmark,
)
from agentbench_frame.tracking.run import Run

from .assets import CALIBRATION_SELECTION_RULE
from .models import CalibrationConfig, CalibrationEvaluation, MatchResult


def build_calibration_cases(
    config: CalibrationConfig,
    split: str,
    mode: str,
) -> tuple[BenchmarkCase, ...]:
    if split not in {"development", "heldout"}:
        raise ValueError("calibration split must be 'development' or 'heldout'")
    if mode not in config.candidate_modes:
        raise ValueError(f"unsupported calibration mode: {mode}")
    seeds = (
        config.development_seeds if split == "development" else config.heldout_seeds
    )
    return tuple(
        BenchmarkCase(
            case_id=(
                f"calibration-{split}-{mode}-s{seed}-p{seat}"
            ),
            opponent=f"calibration-{mode}",
            seed=seed,
            first_player=seat,
            metadata={
                "suite": "calibration",
                "split": split,
                "mode": mode,
                "tier": "calibration",
            },
        )
        for seed in seeds
        for seat in (0, 1)
    )


def select_calibration_candidate(
    config: CalibrationConfig,
    results: Mapping[str, CalibrationEvaluation],
) -> str:
    if set(results) != set(config.candidate_modes) or any(
        result.status != "complete" or result.score is None
        for result in results.values()
    ):
        raise ValueError("all candidate evaluations must be complete")
    return min(
        config.candidate_modes,
        key=lambda mode: (
            round(abs(float(results[mode].score) - config.target_midpoint), 12),
            config.candidate_modes.index(mode),
        ),
    )


class CalibrationEvaluator:
    def __init__(
        self,
        config: CalibrationConfig,
        execute_match: Callable[[BenchmarkCase, Path, str, Path], MatchResult],
    ) -> None:
        self.config = config
        self.execute_match = execute_match

    def evaluate(
        self,
        workspace: Path,
        version: str,
        split: str,
        mode: str,
        run: Run,
        budget_phase: str,
    ) -> CalibrationEvaluation:
        cases = build_calibration_cases(self.config, split, mode)
        spec = BenchmarkSpec(self.config.benchmark_id, list(cases))
        raw: list[GameResult] = []
        matches: list[MatchResult] = []
        for case in cases:
            artifact_dir = (
                Path(run.run_dir)
                / "matches"
                / version
                / case.case_id
            )
            match = self.execute_match(case, workspace, version, artifact_dir)
            matches.append(match)
            if match.winner == -1:
                outcome = "draw"
            elif match.winner == case.first_player:
                outcome = "win"
            else:
                outcome = "loss"
            game = GameResult(
                case_id=case.case_id,
                outcome=outcome,
                valid=match.valid,
                error=match.error,
                metadata={
                    "benchmark_id": self.config.benchmark_id,
                    "suite": "calibration",
                    "split": split,
                    "mode": mode,
                    "version": version,
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                    "winner": match.winner,
                    "termination_type": match.termination_type,
                    "environment_steps": len(match.turns),
                    "artifact_dir": str(artifact_dir),
                },
            )
            raw.append(game)
            run.log_budget(
                budget_phase,
                episodes=1,
                env_steps=len(match.turns),
                game_agent_decision_steps=sum(
                    turn.player == match.evaluated_seat for turn in match.turns
                ),
                primitive_commands=sum(len(turn.commands) for turn in match.turns),
                time_s=match.elapsed_time_s,
            )
            run.log_game_result(
                budget_phase,
                version,
                {
                    "case_id": game.case_id,
                    "outcome": game.outcome,
                    "valid": game.valid,
                    "error": game.error,
                    **game.metadata,
                },
            )
        aggregate = evaluate_benchmark(spec, raw)
        per_seat: dict[int, float | None] = {0: None, 1: None}
        if aggregate.status == "complete":
            for seat in (0, 1):
                seat_cases = [case for case in cases if case.first_player == seat]
                seat_ids = {case.case_id for case in seat_cases}
                seat_results = [item for item in raw if item.case_id in seat_ids]
                per_seat[seat] = evaluate_benchmark(
                    BenchmarkSpec(self.config.benchmark_id, seat_cases),
                    seat_results,
                ).score
        in_target_range = (
            self.config.target_min <= aggregate.score <= self.config.target_max
            if split == "heldout" and aggregate.score is not None
            else None
        )
        return CalibrationEvaluation(
            mode=mode,
            split=split,
            status=aggregate.status,
            score=aggregate.score,
            wins=aggregate.wins,
            losses=aggregate.losses,
            draws=aggregate.draws,
            per_seat=per_seat,
            results=tuple(raw),
            matches=tuple(matches),
            in_target_range=in_target_range,
        )


__all__ = [
    "CALIBRATION_SELECTION_RULE",
    "CalibrationEvaluator",
    "build_calibration_cases",
    "select_calibration_candidate",
]
