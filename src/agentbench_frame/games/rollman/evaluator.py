"""Role-fixed Rollman learning gates and all-human certification."""

from __future__ import annotations

import csv
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agentbench_frame.games.rollman.match import (
    DecisionStateTracker,
    MatchError,
    MatchResult,
    ProcessSpec,
    run_match,
)
from agentbench_frame.hl.codebase import Version
from agentbench_frame.hl.evaluator import CandidateEvaluation


@dataclasses.dataclass(frozen=True)
class Opponent:
    opponent_id: str
    rank: int
    archive: Path
    process: ProcessSpec | None = None
    username: str | None = None
    language: str | None = None
    version: str | None = None


def load_human_pool(manifest_path: str | Path) -> tuple[Opponent, ...]:
    """Read ranked metadata without opening or exposing human source code."""

    source = Path(manifest_path).resolve()
    opponents: list[Opponent] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rank = int(row["rank"])
            archive = (source.parent / row["archive"]).resolve()
            if not archive.is_file():
                raise FileNotFoundError(f"missing human archive: {archive}")
            opponents.append(
                Opponent(
                    opponent_id=f"rank{rank:02d}",
                    rank=rank,
                    archive=archive,
                    username=row.get("username") or None,
                    language=row.get("language") or None,
                    version=row.get("version") or None,
                )
            )
    opponents.sort(key=lambda opponent: opponent.rank)
    if [opponent.rank for opponent in opponents] != list(range(1, 17)):
        raise ValueError("Rollman human pool must contain ranks 1 through 16")
    return tuple(opponents)


MatchRunner = Callable[..., MatchResult]
CandidateFactory = Callable[[Version], ProcessSpec]
TrackerFactory = Callable[[], DecisionStateTracker]


class RollmanEvaluator:
    """Evaluate each iteration against rank 1 and certify against all 16."""

    def __init__(
        self,
        *,
        logic: ProcessSpec,
        candidate_factory: CandidateFactory,
        learning_opponent: Opponent,
        human_pool: Sequence[Opponent],
        fixed_gate_seeds: Iterable[int],
        certification_seeds: Iterable[int],
        artifact_root: str | Path,
        timeout_s: float = 2.0,
        match_runner: MatchRunner = run_match,
        state_tracker_factory: TrackerFactory | None = None,
        max_parallel_matches: int = 1,
    ) -> None:
        self.logic = logic
        self.candidate_factory = candidate_factory
        self.learning_opponent = learning_opponent
        self.human_pool = tuple(sorted(human_pool, key=lambda item: item.rank))
        self.fixed_gate_seeds = tuple(int(seed) for seed in fixed_gate_seeds)
        self.certification_seeds = tuple(
            int(seed) for seed in certification_seeds
        )
        self.artifact_root = Path(artifact_root)
        self.timeout_s = float(timeout_s)
        self.match_runner = match_runner
        self.state_tracker_factory = state_tracker_factory
        self.max_parallel_matches = int(max_parallel_matches)
        self.last_evaluation: CandidateEvaluation | None = None
        if learning_opponent.process is None:
            raise ValueError("learning opponent has not been prepared")
        if not self.fixed_gate_seeds:
            raise ValueError("fixed_gate_seeds cannot be empty")
        if not self.certification_seeds:
            raise ValueError("certification_seeds cannot be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_parallel_matches < 1:
            raise ValueError("max_parallel_matches must be positive")

    def evaluate(self, version: Version) -> CandidateEvaluation:
        self.last_evaluation = self._evaluate_cases(
            version,
            opponents=(self.learning_opponent,),
            seeds=self.fixed_gate_seeds,
            phase="learning",
        )
        return self.last_evaluation

    def quick_screen(self, version: Version) -> CandidateEvaluation:
        """Give every sibling one common, real feedback trajectory."""

        return self._evaluate_cases(
            version,
            opponents=(self.learning_opponent,),
            seeds=self.fixed_gate_seeds[:1],
            phase="quick_screen",
        )

    def evaluate_finalist(self, version: Version) -> CandidateEvaluation:
        """Evaluate a quick-screen finalist on the remaining target seeds."""

        seeds = self.fixed_gate_seeds[1:]
        if not seeds:
            return CandidateEvaluation(
                status="complete",
                score=0.0,
                matches=(),
            )
        return self._evaluate_cases(
            version,
            opponents=(self.learning_opponent,),
            seeds=seeds,
            phase="finalist",
        )

    @staticmethod
    def combine_stages(
        quick: CandidateEvaluation,
        finalist: CandidateEvaluation,
    ) -> CandidateEvaluation:
        matches = (*quick.matches, *finalist.matches)
        if quick.status != "complete" or finalist.status != "complete":
            return CandidateEvaluation(
                status="incomplete",
                score=None,
                error="one or more staged evaluation cases are incomplete",
                matches=matches,
            )
        results = [match.get("result") for match in matches]
        if any(result not in {"win", "draw", "loss"} for result in results):
            return CandidateEvaluation(
                status="incomplete",
                score=None,
                error="staged evaluation contains an invalid result",
                matches=matches,
            )
        if not results:
            return CandidateEvaluation(status="complete", score=0.0, matches=())
        score = (
            sum(result == "win" for result in results)
            + 0.5 * sum(result == "draw" for result in results)
        ) / len(results)
        return CandidateEvaluation(status="complete", score=score, matches=matches)

    def evaluate_reporting_panel(
        self,
        version: Version,
        *,
        seeds: Iterable[int],
    ) -> CandidateEvaluation:
        if any(opponent.process is None for opponent in self.human_pool):
            raise ValueError("every reporting opponent must be prepared")
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if not normalized_seeds:
            raise ValueError("reporting panel seeds cannot be empty")
        return self._evaluate_cases(
            version,
            opponents=self.human_pool,
            seeds=normalized_seeds,
            phase="reporting",
        )

    def set_learning_opponent(self, opponent: Opponent) -> None:
        """Switch the fixed-seed gate target without rebuilding the runtime."""

        if opponent.process is None:
            raise ValueError("learning opponent has not been prepared")
        self.learning_opponent = opponent

    def certify(self, version: Version) -> CandidateEvaluation:
        if len(self.human_pool) != 16:
            raise ValueError("certification requires all 16 ranked humans")
        if any(opponent.process is None for opponent in self.human_pool):
            raise ValueError("every certification opponent must be prepared")
        result = self._evaluate_cases(
            version,
            opponents=self.human_pool,
            seeds=self.certification_seeds,
            phase="certification",
        )
        return result

    def _evaluate_cases(
        self,
        version: Version,
        *,
        opponents: Sequence[Opponent],
        seeds: Sequence[int],
        phase: str,
    ) -> CandidateEvaluation:
        candidate = self.candidate_factory(version)
        cases = [
            (opponent, seed)
            for opponent in opponents
            for seed in seeds
        ]

        def run_case(opponent: Opponent, seed: int) -> dict[str, Any]:
            assert opponent.process is not None
            case_root = (
                self.artifact_root
                / version.version_id
                / phase
                / opponent.opponent_id
                / f"seed-{seed}"
            )
            replay_path = case_root / "replay.jsonl"
            trace_path = case_root / "trace.jsonl"
            record: dict[str, Any] = {
                "phase": phase,
                "version_id": version.version_id,
                "opponent": opponent.opponent_id,
                "opponent_rank": opponent.rank,
                "seed": seed,
            }
            try:
                match = self.match_runner(
                    logic=self.logic,
                    rollman=candidate,
                    ghosts=opponent.process,
                    seed=seed,
                    timeout_s=self.timeout_s,
                    replay_path=replay_path,
                    trace_path=trace_path,
                    state_tracker=(
                        self.state_tracker_factory()
                        if self.state_tracker_factory is not None
                        else None
                    ),
                )
            except MatchError as exc:
                record.update(status="incomplete", error=str(exc))
            else:
                record.update(
                    status="complete",
                    result=match.result,
                    end_state=list(match.end_state),
                    rollman_score=match.rollman_score,
                    ghosts_score=match.ghosts_score,
                    raw_replay_sha256=match.replay.raw_sha256,
                    normalized_replay_sha256=match.replay.normalized_sha256,
                    replay=str(replay_path),
                    trace=str(trace_path),
                    game_agent_decisions=len(match.rollman_decisions),
                )
            return record

        if self.max_parallel_matches == 1 or len(cases) <= 1:
            records = [run_case(opponent, seed) for opponent, seed in cases]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_parallel_matches, len(cases))
            ) as executor:
                futures = [
                    executor.submit(run_case, opponent, seed)
                    for opponent, seed in cases
                ]
                records = [future.result() for future in futures]

        complete = all(record.get("status") == "complete" for record in records)

        if not complete:
            return CandidateEvaluation(
                status="incomplete",
                score=None,
                error="one or more fixed evaluation cases are incomplete",
                matches=tuple(records),
            )
        wins = sum(record["result"] == "win" for record in records)
        draws = sum(record["result"] == "draw" for record in records)
        score = (wins + 0.5 * draws) / len(records)
        return CandidateEvaluation(
            status="complete",
            score=score,
            matches=tuple(records),
        )
