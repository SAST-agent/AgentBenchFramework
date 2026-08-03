"""Role-symmetric AntWar2 learning gates and human-pool certification."""

from __future__ import annotations

import csv
import dataclasses
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agentbench_frame.games.antwar2.match import (
    AntWarMatchError,
    NativeMatchArtifacts,
    ProcessSpec,
    run_native_match,
)
from agentbench_frame.hl.codebase import Version
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.match_record import MatchRecord


@dataclasses.dataclass(frozen=True)
class AntWarOpponent:
    opponent_id: str
    rank: int
    archive: Path
    process: ProcessSpec | None
    username: str | None = None
    score: int | None = None
    language: str | None = None
    version: str | None = None


def load_human_pool(
    manifest_path: str | Path,
    extracted_root: str | Path,
) -> tuple[AntWarOpponent, ...]:
    """Load ranked metadata and entry paths without reading policy source."""

    manifest = Path(manifest_path).resolve()
    extracted = Path(extracted_root).resolve()
    pool: list[AntWarOpponent] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rank = int(row["rank"])
            matches = tuple(sorted(extracted.glob(f"rank{rank:02d}__*")))
            if len(matches) != 1:
                raise ValueError(
                    f"rank{rank:02d} must map to exactly one extracted package"
                )
            package = matches[0]
            main = package / "main.py"
            process = (
                ProcessSpec((sys.executable, "main.py"), package)
                if main.is_file()
                else None
            )
            archive = (manifest.parent / row["archive"]).resolve()
            if not archive.is_file():
                raise FileNotFoundError(f"missing human archive: {archive}")
            pool.append(
                AntWarOpponent(
                    opponent_id=f"rank{rank:02d}",
                    rank=rank,
                    archive=archive,
                    process=process,
                    username=row.get("username") or None,
                    score=int(row["score"]) if row.get("score") else None,
                    language=row.get("language") or None,
                    version=row.get("version") or None,
                )
            )
    pool.sort(key=lambda item: item.rank)
    if [item.rank for item in pool] != list(range(1, len(pool) + 1)):
        raise ValueError("AntWar2 human ranks must be contiguous from rank01")
    return tuple(pool)


CandidateFactory = Callable[[Version], ProcessSpec]
MatchRunner = Callable[..., NativeMatchArtifacts]


class AntWar2Evaluator:
    """Evaluate every candidate in both official player roles."""

    def __init__(
        self,
        *,
        game: ProcessSpec,
        candidate_factory: CandidateFactory,
        learning_opponents: Sequence[AntWarOpponent],
        human_pool: Sequence[AntWarOpponent],
        fixed_gate_seeds: Iterable[int],
        certification_seeds: Iterable[int],
        artifact_root: str | Path,
        roles: Sequence[str] = ("P0", "P1"),
        timeout_s: float = 120.0,
        match_runner: MatchRunner = run_native_match,
        max_parallel_matches: int = 1,
        quick_screen_seed_count: int = 1,
        finalist_seed_count: int | None = None,
    ) -> None:
        self.game = game
        self.candidate_factory = candidate_factory
        self.learning_opponents = tuple(learning_opponents)
        self.human_pool = tuple(sorted(human_pool, key=lambda item: item.rank))
        self.fixed_gate_seeds = tuple(int(seed) for seed in fixed_gate_seeds)
        self.certification_seeds = tuple(int(seed) for seed in certification_seeds)
        self.artifact_root = Path(artifact_root)
        self.roles = tuple(roles)
        self.timeout_s = float(timeout_s)
        self.match_runner = match_runner
        self.max_parallel_matches = int(max_parallel_matches)
        self.quick_screen_seed_count = int(quick_screen_seed_count)
        self.finalist_seed_count = (
            max(0, len(self.fixed_gate_seeds) - self.quick_screen_seed_count)
            if finalist_seed_count is None
            else int(finalist_seed_count)
        )
        self.last_evaluation: CandidateEvaluation | None = None
        if not self.learning_opponents:
            raise ValueError("learning_opponents cannot be empty")
        if any(item.process is None for item in self.learning_opponents):
            raise ValueError("learning opponent has no runnable main.py")
        if not self.fixed_gate_seeds:
            raise ValueError("fixed_gate_seeds cannot be empty")
        if not self.certification_seeds:
            raise ValueError("certification_seeds cannot be empty")
        if not self.roles or any(role not in {"P0", "P1"} for role in self.roles):
            raise ValueError("roles must contain P0 and/or P1")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be unique")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_parallel_matches < 1:
            raise ValueError("max_parallel_matches must be positive")
        if not 1 <= self.quick_screen_seed_count <= len(self.fixed_gate_seeds):
            raise ValueError("quick_screen_seed_count is outside fixed seeds")
        if self.finalist_seed_count < 0 or (
            self.quick_screen_seed_count + self.finalist_seed_count
            > len(self.fixed_gate_seeds)
        ):
            raise ValueError("finalist_seed_count is outside fixed seeds")

    def evaluate(self, version: Version) -> CandidateEvaluation:
        self.last_evaluation = self._evaluate_cases(
            version,
            opponents=self.learning_opponents,
            seeds=self.fixed_gate_seeds,
            phase="learning",
        )
        return self.last_evaluation

    def quick_screen(self, version: Version) -> CandidateEvaluation:
        return self._evaluate_cases(
            version,
            opponents=self.learning_opponents,
            seeds=self.fixed_gate_seeds[: self.quick_screen_seed_count],
            phase="quick_screen",
        )

    def evaluate_finalist(self, version: Version) -> CandidateEvaluation:
        seeds = self.fixed_gate_seeds[
            self.quick_screen_seed_count :
            self.quick_screen_seed_count + self.finalist_seed_count
        ]
        if not seeds:
            return CandidateEvaluation(status="complete", score=0.0, matches=())
        return self._evaluate_cases(
            version,
            opponents=self.learning_opponents,
            seeds=seeds,
            phase="finalist",
        )

    @staticmethod
    def combine_stages(
        quick: CandidateEvaluation,
        finalist: CandidateEvaluation,
    ) -> CandidateEvaluation:
        unique: dict[tuple[str, str, int], dict[str, Any]] = {}
        for raw in (*quick.matches, *finalist.matches):
            record = MatchRecord.from_mapping(raw)
            key = record.comparison_key
            if key in unique and unique[key] != record.to_dict():
                return CandidateEvaluation(
                    status="incomplete",
                    score=None,
                    error="duplicate staged match has inconsistent outcome",
                    matches=tuple(unique.values()),
                )
            unique[key] = record.to_dict()
        records = tuple(unique.values())
        if quick.status != "complete" or finalist.status != "complete":
            return CandidateEvaluation(
                status="incomplete",
                score=None,
                error="one or more staged cases are incomplete",
                matches=records,
            )
        points = [float(record["points"]) for record in records]
        return CandidateEvaluation(
            status="complete",
            score=(sum(points) / len(points) if points else 0.0),
            matches=records,
        )

    def certify(self, version: Version) -> CandidateEvaluation:
        opponents = tuple(item for item in self.human_pool if item.process is not None)
        if not opponents:
            raise ValueError("human pool contains no runnable opponents")
        return self._evaluate_cases(
            version,
            opponents=opponents,
            seeds=self.certification_seeds,
            phase="certification",
        )

    def evaluate_reporting_panel(
        self,
        version: Version,
        *,
        seed_count: int = 1,
    ) -> CandidateEvaluation:
        if not 1 <= seed_count <= len(self.certification_seeds):
            raise ValueError("reporting seed_count is outside certification seeds")
        opponents = tuple(item for item in self.human_pool if item.process is not None)
        if not opponents:
            raise ValueError("human pool contains no runnable opponents")
        return self._evaluate_cases(
            version,
            opponents=opponents,
            seeds=self.certification_seeds[:seed_count],
            phase="reporting",
        )

    def set_learning_opponent(self, opponent: AntWarOpponent) -> None:
        if opponent.process is None:
            raise ValueError("learning opponent has no runnable main.py")
        self.learning_opponents = (opponent,)

    def current_learning_cases(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (opponent.opponent_id, role, seed)
            for opponent in self.learning_opponents
            for role in self.roles
            for seed in self.fixed_gate_seeds
        )

    def _failure_record(
        self,
        *,
        version: Version,
        opponent: AntWarOpponent,
        role: str,
        seed: int,
        error: str,
    ) -> MatchRecord:
        return MatchRecord.from_mapping(
            {
                "schema_version": "1.0",
                "game": "30_antwar2",
                "candidate": version.version_id,
                "opponent": opponent.opponent_id,
                "candidate_role": role,
                "seed": seed,
                "status": "failed",
                "result": None,
                "points": None,
                "candidate_score": None,
                "opponent_score": None,
                "dense_margin": None,
                "terminal_metrics": {},
                "rounds": None,
                "replay": None,
                "trace": None,
                "faults": [error],
                "live_opponent": True,
            }
        )

    def _evaluate_cases(
        self,
        version: Version,
        *,
        opponents: Sequence[AntWarOpponent],
        seeds: Sequence[int],
        phase: str,
    ) -> CandidateEvaluation:
        candidate_process = self.candidate_factory(version)
        cases = [
            (opponent, role, seed)
            for opponent in opponents
            for role in self.roles
            for seed in seeds
        ]

        def run_case(
            opponent: AntWarOpponent,
            role: str,
            seed: int,
        ) -> MatchRecord:
            assert opponent.process is not None
            root = (
                self.artifact_root
                / version.version_id
                / phase
                / opponent.opponent_id
                / role.lower()
                / f"seed-{seed}"
            )
            try:
                artifacts = self.match_runner(
                    game=self.game,
                    candidate_process=candidate_process,
                    opponent_process=opponent.process,
                    candidate=version.version_id,
                    opponent=opponent.opponent_id,
                    candidate_role=role,
                    seed=seed,
                    replay_path=root / "replay.json",
                    trace_path=root / "trace.jsonl",
                    events_path=root / "events.jsonl",
                    timeout_s=self.timeout_s,
                )
            except AntWarMatchError as exc:
                return self._failure_record(
                    version=version,
                    opponent=opponent,
                    role=role,
                    seed=seed,
                    error=str(exc),
                )
            return artifacts.record

        if self.max_parallel_matches == 1 or len(cases) <= 1:
            records = [run_case(*case) for case in cases]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_parallel_matches, len(cases))
            ) as executor:
                records = list(executor.map(lambda case: run_case(*case), cases))
        raw = tuple(record.to_dict() for record in records)
        if any(not record.promotable for record in records):
            return CandidateEvaluation(
                status="incomplete",
                score=None,
                error="one or more live evaluation cases are incomplete",
                matches=raw,
            )
        return CandidateEvaluation(
            status="complete",
            score=sum(record.points for record in records if record.points is not None)
            / len(records),
            matches=raw,
        )
