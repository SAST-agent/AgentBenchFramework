"""Complete seat-balanced DOTO population evaluation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .match import DotoMatchError, MatchResult, run_match
from .population import PopulationBundle
from .scheduler import AdaptiveScheduler


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    opponent_id: str
    opponent_name: str
    opponent_executable: Path
    seed: int
    candidate_seat: int


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    opponent_id: str
    opponent_name: str
    seed: int
    candidate_seat: int
    terminated_by: str
    scores: tuple[float, float] | None
    score_diff: float | None
    replay_path: str | None
    trace_path: str | None
    error: str | None

    def to_json(self) -> dict:
        value = asdict(self)
        value["scores"] = list(self.scores) if self.scores is not None else None
        return value


@dataclass(frozen=True)
class MatrixSpec:
    split: str
    bundle: PopulationBundle
    tasks: tuple[EvaluationTask, ...]

    @classmethod
    def formal_train(cls, bundle: PopulationBundle, seed: int = 11) -> "MatrixSpec":
        return cls._build(bundle, "train", seed, 15)

    @classmethod
    def final_test(cls, bundle: PopulationBundle, seed: int = 11) -> "MatrixSpec":
        return cls._build(bundle, "test", seed, 28)

    @classmethod
    def _build(cls, bundle: PopulationBundle, split: str, seed: int,
               expected_policies: int) -> "MatrixSpec":
        if bundle.split != split:
            raise ValueError(f"expected {split} population bundle")
        if bundle.seed != seed or seed != 11:
            raise ValueError("formal DOTO matrices require seed 11")
        if bundle.seats != (0, 1):
            raise ValueError("formal DOTO matrices require seats 0 and 1")
        if len(bundle.policies) != expected_policies:
            raise ValueError(f"{split} bundle must contain exactly {expected_policies} policies")
        tasks = tuple(
            EvaluationTask(
                task_id=f"{split}:{policy.policy_id}:seed{seed}:seat{seat}",
                opponent_id=policy.policy_id,
                opponent_name=policy.name,
                opponent_executable=policy.executable,
                seed=seed,
                candidate_seat=seat,
            )
            for policy in bundle.policies
            for seat in bundle.seats
        )
        return cls(split, bundle, tasks)


def _normal(row: EpisodeResult) -> bool:
    return (
        row.terminated_by == "normal"
        and row.error is None
        and row.scores is not None
        and row.score_diff is not None
        and math.isfinite(row.score_diff)
        and all(math.isfinite(value) for value in row.scores)
    )


def aggregate_matrix(rows: list[EpisodeResult] | tuple[EpisodeResult, ...],
                     expected_task_ids: tuple[str, ...]) -> dict:
    if len(set(expected_task_ids)) != len(expected_task_ids):
        raise ValueError("expected matrix task IDs must be unique")
    by_id = {row.task_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("matrix rows contain duplicate task IDs")
    unexpected = set(by_id) - set(expected_task_ids)
    if unexpected:
        raise ValueError(f"matrix contains unexpected task IDs: {sorted(unexpected)}")

    normal_rows = [by_id[task_id] for task_id in expected_task_ids
                   if task_id in by_id and _normal(by_id[task_id])]
    expected_count = len(expected_task_ids)
    completion_rate = len(normal_rows) / expected_count if expected_count else 0.0
    differences = [row.score_diff for row in normal_rows if row.score_diff is not None]
    formal_score = (
        round(sum(differences) / expected_count, 6)
        if expected_count and len(normal_rows) == expected_count else None
    )
    outcomes = [1.0 if diff > 0 else 0.0 if diff < 0 else 0.5 for diff in differences]

    grouped: dict[str, list[EpisodeResult]] = {}
    for row in rows:
        grouped.setdefault(row.opponent_id, []).append(row)
    policies = []
    for opponent_id in sorted(grouped):
        policy_rows = grouped[opponent_id]
        complete = (
            len(policy_rows) == 2
            and {row.candidate_seat for row in policy_rows} == {0, 1}
            and all(_normal(row) for row in policy_rows)
        )
        mean_difference = (
            sum(row.score_diff for row in policy_rows if row.score_diff is not None) / 2
            if complete else None
        )
        policies.append({
            "policy_id": opponent_id,
            "name": policy_rows[0].opponent_name,
            "status": "complete" if complete else "incomplete",
            "mean_score_diff": round(mean_difference, 6) if mean_difference is not None else None,
            "defeated": bool(complete and mean_difference is not None and mean_difference > 0),
            "task_ids": sorted(row.task_id for row in policy_rows),
        })

    counts = {
        "normal": len(normal_rows),
        "failed": sum(row.task_id in expected_task_ids and not _normal(row) for row in rows),
        "missing": len(set(expected_task_ids) - set(by_id)),
    }
    return {
        "expected_matches": expected_count,
        "attempted_matches": len(rows),
        "completion_rate": completion_rate,
        "formal_score": formal_score,
        "win_rate": round(sum(outcomes) / len(outcomes), 6) if outcomes else None,
        "defeated_policy_count": sum(policy["defeated"] for policy in policies),
        "policy_count": len(policies),
        "counts": counts,
        "policies": policies,
        "episodes": [by_id[task_id].to_json() for task_id in expected_task_ids if task_id in by_id],
    }


def _safe_task_dir(task_id: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-"
                   for character in task_id)


def evaluate_matrix(
    spec: MatrixSpec,
    candidate: Path,
    scheduler: AdaptiveScheduler,
    output_dir: Path,
    *,
    match_runner: Callable[..., MatchResult] = run_match,
    server_dir: Path | None = None,
    test_only: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def execute(task: EvaluationTask) -> EpisodeResult:
        cell_dir = output_dir / "cells" / _safe_task_dir(task.task_id)
        cell_dir.mkdir(parents=True, exist_ok=True)

        def persist(episode: EpisodeResult) -> EpisodeResult:
            temporary = cell_dir / ".episode.json.tmp"
            temporary.write_text(json.dumps(
                episode.to_json(), ensure_ascii=False, indent=2, allow_nan=False
            ) + "\n")
            os.replace(temporary, cell_dir / "episode.json")
            return episode

        agents = (
            (candidate, task.opponent_executable)
            if task.candidate_seat == 0
            else (task.opponent_executable, candidate)
        )
        try:
            result = match_runner(
                agents[0], agents[1], seed=task.seed, output_dir=cell_dir,
                tag="match", server_dir=server_dir, test_only=test_only,
            )
            score_diff = result.scores[task.candidate_seat] - result.scores[1 - task.candidate_seat]
            return persist(EpisodeResult(
                task.task_id, task.opponent_id, task.opponent_name, task.seed,
                task.candidate_seat, result.terminated_by, result.scores,
                score_diff, str(result.replay_path), str(result.trace_path),
                "; ".join(result.errors) or None,
            ))
        except DotoMatchError as error:
            return persist(EpisodeResult(
                task.task_id, task.opponent_id, task.opponent_name, task.seed,
                task.candidate_seat, error.stage, None, None, None, None, str(error),
            ))
        except Exception as error:
            return persist(EpisodeResult(
                task.task_id, task.opponent_id, task.opponent_name, task.seed,
                task.candidate_seat, "failed", None, None, None, None,
                f"{type(error).__name__}: {error}",
            ))

    existing: list[EpisodeResult] = []
    existing_ids: set[str] = set()
    expected_ids = {task.task_id for task in spec.tasks}
    for path in sorted((output_dir / "cells").glob("*/episode.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(raw.get("task_id", ""))
        if task_id not in expected_ids or task_id in existing_ids:
            raise ValueError(f"invalid immutable evaluation attempt: {task_id}")
        raw["scores"] = tuple(raw["scores"]) if raw.get("scores") is not None else None
        existing.append(EpisodeResult(**raw))
        existing_ids.add(task_id)
    missing_tasks = tuple(task for task in spec.tasks if task.task_id not in existing_ids)
    scheduled = scheduler.run(missing_tasks, execute)
    rows = existing + [outcome.result for outcome in scheduled.outcomes if outcome.result is not None]
    summary = aggregate_matrix(rows, tuple(task.task_id for task in spec.tasks))
    document = {
        "schema_version": 1,
        "benchmark_version": spec.bundle.benchmark_version,
        "split": spec.split,
        "seed": 11,
        "seats": [0, 1],
        "expected_task_ids": [task.task_id for task in spec.tasks],
        "summary": {key: value for key, value in summary.items() if key != "episodes"},
        "episodes": summary["episodes"],
        "scheduler": {
            "cpu_history": list(scheduled.cpu_history),
            "worker_history": list(scheduled.worker_history),
            "cancelled_task_ids": list(scheduled.cancelled_task_ids),
        },
    }
    temporary = output_dir / ".matrix.json.tmp"
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output_dir / "matrix.json")
    return document
