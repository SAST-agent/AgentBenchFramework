"""Frozen Generals evaluation and learning case construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from agentbench_frame.eval.benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    GameResult,
    evaluate_benchmark,
)
from agentbench_frame.tracking.run import Run

from .models import MatchResult, PilotConfig
from .dense import persist_dense_diagnostics


@dataclass(frozen=True)
class GeneralsEvaluation:
    version: str
    status: str
    score: float | None
    wins: int
    losses: int
    draws: int
    per_tier: Mapping[str, float | None]
    seat_gap: float | None
    results: tuple[GameResult, ...]
    matches: tuple[MatchResult, ...]


def _case(opponent, seed: int, seat: int, phase: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=f"{phase}-{opponent.tier}-{opponent.opponent_id}-s{seed}-p{seat}",
        opponent=opponent.opponent_id,
        seed=seed,
        first_player=seat,
        metadata={"tier": opponent.tier, "phase": phase},
    )


def build_evaluation_spec(config: PilotConfig) -> BenchmarkSpec:
    cases = [
        _case(opponent, seed, seat, "eval")
        for opponent in config.opponents
        for seed in config.evaluation_seeds
        for seat in (0, 1)
    ]
    return BenchmarkSpec(version=config.benchmark_id, cases=cases)


def build_learning_cases(config: PilotConfig) -> tuple[BenchmarkCase, ...]:
    return tuple(
        _case(opponent, seed, seat, "learn")
        for opponent in config.learning_opponents
        for seed in config.learning_seeds
        for seat in (0, 1)
    )


class GeneralsEvaluator:
    def __init__(
        self,
        config: PilotConfig,
        execute_match: Callable[[BenchmarkCase, Path, str, Path], MatchResult],
    ):
        self.config = config
        self.execute_match = execute_match

    def evaluate(
        self,
        workspace: Path,
        version: str,
        phase: str,
        run: Run,
        cases: Sequence[BenchmarkCase] | None = None,
    ) -> GeneralsEvaluation:
        selected = tuple(cases or build_evaluation_spec(self.config).cases)
        spec = BenchmarkSpec(self.config.benchmark_id, list(selected))
        raw: list[GameResult] = []
        matches: list[MatchResult] = []
        for case in selected:
            artifact_dir = Path(run.run_dir) / "matches" / version / case.case_id
            match = self.execute_match(case, workspace, version, artifact_dir)
            matches.append(match)
            try:
                dense_trace, dense_summary = persist_dense_diagnostics(
                    match, artifact_dir
                )
                run.write(
                    "dense_trajectory",
                    case_id=case.case_id,
                    version=version,
                    phase=phase,
                    evaluated_seat=case.first_player,
                    trace=[asdict(sample) for sample in dense_trace],
                    artifact_ref=str(artifact_dir / "dense-trace.jsonl"),
                )
                run.write(
                    "dense_episode_summary",
                    version=version,
                    phase=phase,
                    artifact_ref=str(artifact_dir / "dense-summary.json"),
                    **asdict(dense_summary),
                )
            except Exception as exc:
                run.write(
                    "dense_metric_error",
                    case_id=case.case_id,
                    version=version,
                    phase=phase,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if match.valid:
                if match.winner == -1:
                    outcome = "draw"
                elif match.winner == case.first_player:
                    outcome = "win"
                else:
                    outcome = "loss"
            else:
                outcome = "loss"
            game = GameResult(
                case_id=case.case_id,
                outcome=outcome,
                valid=match.valid,
                error=match.error,
                metadata={
                    "version": version,
                    "phase": phase,
                    "opponent": case.opponent,
                    "tier": case.metadata["tier"],
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                    "winner": match.winner,
                    "termination_type": match.termination_type,
                    "environment_steps": len(match.turns),
                    "elapsed_time_s": match.elapsed_time_s,
                    "artifact_dir": str(artifact_dir),
                },
            )
            raw.append(game)
            run.log_budget(
                phase,
                episodes=1,
                env_steps=len(match.turns),
                game_agent_decision_steps=sum(
                    turn.player == match.evaluated_seat for turn in match.turns
                ),
                primitive_commands=sum(
                    len(turn.commands) for turn in match.turns
                ),
                time_s=match.elapsed_time_s,
            )
            run.log_game_result(phase, version, {
                "case_id": game.case_id,
                "outcome": game.outcome,
                "valid": game.valid,
                "error": game.error,
                **game.metadata,
            })
        aggregate = evaluate_benchmark(spec, raw)
        per_tier: dict[str, float | None] = {}
        for tier in ("high", "medium", "low"):
            tier_cases = [case for case in selected if case.metadata["tier"] == tier]
            if not tier_cases:
                per_tier[tier] = None
                continue
            tier_ids = {case.case_id for case in tier_cases}
            tier_results = [result for result in raw if result.case_id in tier_ids]
            per_tier[tier] = evaluate_benchmark(
                BenchmarkSpec(self.config.benchmark_id, tier_cases), tier_results
            ).score
        seat_gap = None
        if aggregate.status == "complete":
            scores = {}
            for seat in (0, 1):
                values = [
                    1.0 if result.outcome == "win" else 0.5 if result.outcome == "draw" else 0.0
                    for result, case in zip(raw, selected)
                    if case.first_player == seat
                ]
                scores[seat] = sum(values) / len(values)
            seat_gap = scores[0] - scores[1]
        return GeneralsEvaluation(
            version=version,
            status=aggregate.status,
            score=aggregate.score,
            wins=aggregate.wins,
            losses=aggregate.losses,
            draws=aggregate.draws,
            per_tier=per_tier,
            seat_gap=seat_gap,
            results=tuple(raw),
            matches=tuple(matches),
        )
