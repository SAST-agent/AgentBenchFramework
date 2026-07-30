"""Framework-owned HL act, candidate, evaluation, and rollback orchestration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

from agentbench_frame.arena.rating import RoleEloLedger
from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.config import IterationConfig, RollbackConfig
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.events import HLEventWriter
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.lineage import LineageManager, ParentDecision
from agentbench_frame.tracking.provider import ProviderInvocation


@dataclasses.dataclass(frozen=True)
class CandidateResult:
    act_id: str
    branch_index: int
    version: Version
    evaluation: CandidateEvaluation
    provider: ProviderInvocation


@dataclasses.dataclass(frozen=True)
class IterationResult:
    iteration_id: str
    parent_version_id: str
    candidates: tuple[CandidateResult, ...]
    selected: CandidateResult
    rollback: Optional[ParentDecision]


class HLController:
    """Run reproducible candidate iterations; the game stays behind evaluator."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        run_root: str | Path,
        provider: Any,
        evaluator: Any,
        version_store: VersionStore,
        lineage: LineageManager,
        events: HLEventWriter,
        iteration: IterationConfig,
        rollback: RollbackConfig,
        prompt_factory: Callable[..., str],
        experience_manager: ExperienceManager | None = None,
        elo_ledger: RoleEloLedger | None = None,
        elo_candidate_id: str = "hl-run",
        anchor_human_opponents: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.run_root = Path(run_root)
        self.provider = provider
        self.evaluator = evaluator
        self.version_store = version_store
        self.lineage = lineage
        self.events = events
        self.iteration = iteration
        self.rollback = rollback
        self.prompt_factory = prompt_factory
        self.experience_manager = experience_manager
        self.elo_ledger = elo_ledger or RoleEloLedger()
        self.elo_candidate_id = elo_candidate_id
        self.anchor_human_opponents = anchor_human_opponents
        self._iteration_count = 0
        self._coding_agent_acts = 0
        self._sessions: dict[str, str] = {}
        self._started = False

    def initialize(self, *, evaluate: bool = False) -> Version:
        if self._started:
            raise RuntimeError("controller already initialized")
        self.events.write("run_started", iteration_config=dataclasses.asdict(self.iteration))
        version = self.version_store.snapshot(
            parent_version_id=None,
            act_id="initial",
        )
        evaluation = (
            self.evaluator.evaluate(version)
            if evaluate
            else CandidateEvaluation(status="incomplete", score=None, error="not evaluated")
        )
        self.lineage.record_evaluation(
            version.version_id,
            parent_version_id=None,
            status=evaluation.status,
            score=evaluation.score,
        )
        self._write_version_event(version, evaluation, selected=True)
        self.events.write(
            "candidate_selected",
            iteration_id="iter-000000",
            version_id=version.version_id,
            act_id=version.act_id,
        )
        if evaluation.status == "complete":
            self.events.write(
                "champion_promoted",
                version_id=version.version_id,
                score=evaluation.score,
            )
        self._started = True
        return version

    def resume(self, historical_events: list[dict[str, Any]]) -> None:
        if self._started:
            raise RuntimeError("controller already initialized")
        if self.lineage.lineage_head_version_id is None:
            raise ValueError("resume requires lineage rebuilt from events")
        threads_by_act = {
            str(event["act_id"]): str(event["thread_id"])
            for event in historical_events
            if event.get("event_type") == "act_completed"
            and event.get("thread_id")
        }
        for event in historical_events:
            if event.get("event_type") != "version_created":
                continue
            thread_id = threads_by_act.get(str(event.get("act_id")))
            if thread_id:
                self._sessions[str(event["version_id"])] = thread_id
        self._coding_agent_acts = sum(
            event.get("event_type") == "act_completed"
            for event in historical_events
        )
        selected_iterations = {
            str(event.get("iteration_id"))
            for event in historical_events
            if event.get("event_type") == "candidate_selected"
            and event.get("iteration_id") != "iter-000000"
        }
        self._iteration_count = len(selected_iterations)
        self._started = True
        self.events.write(
            "run_resumed",
            coding_agent_acts=self._coding_agent_acts,
            iterations=self._iteration_count,
            lineage_head_version_id=self.lineage.lineage_head_version_id,
            champion_version_id=self.lineage.champion_version_id,
        )

    def run_act(self) -> IterationResult:
        if not self._started:
            raise RuntimeError("initialize must be called before run_act")
        parent_decision = self.lineage.select_next_parent()
        rollback = parent_decision if parent_decision.rollback else None
        parent_id = parent_decision.to_version_id
        if rollback is not None:
            self.events.write(
                "rollback_selected",
                from_version_id=rollback.from_version_id,
                to_version_id=rollback.to_version_id,
                reason=rollback.reason,
            )
        self._iteration_count += 1
        iteration_id = f"iter-{self._iteration_count:06d}"
        results: list[CandidateResult] = []
        pending_experience: dict[str, Path] = {}
        branch_count = self.iteration.candidates_per_act

        for branch_index in range(branch_count):
            self.version_store.checkout(parent_id)
            experience_update = (
                self.workspace / ".agentbench" / "experience_update.json"
            )
            if experience_update.exists():
                experience_update.unlink()
            act_id = f"act-{self._coding_agent_acts + 1:06d}-b{branch_index:02d}"
            prompt = self.prompt_factory(
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=branch_index,
                branch_count=branch_count,
                parent_version_id=parent_id,
            )
            raw_path = self.run_root / "provider" / f"{act_id}.jsonl"
            session_id = (
                self._sessions.get(parent_id)
                if branch_count == 1
                else None
            )
            invocation = self.provider.invoke(
                prompt=prompt,
                workspace=self.workspace,
                raw_output_path=raw_path,
                session_id=session_id,
            )
            self._write_checkpoint(
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=branch_index,
                parent_version_id=parent_id,
                prompt=prompt,
                invocation=invocation,
            )
            self._coding_agent_acts += 1
            if experience_update.is_file():
                pending_path = (
                    self.run_root
                    / "experience"
                    / "pending"
                    / f"{act_id}.json"
                )
                pending_path.parent.mkdir(parents=True, exist_ok=True)
                pending_path.write_bytes(experience_update.read_bytes())
                pending_experience[act_id] = pending_path
            version = self.version_store.snapshot(
                parent_version_id=parent_id,
                act_id=act_id,
                edit_type="candidate",
            )
            evaluation = (
                self.evaluator.evaluate(version)
                if invocation.status == "completed"
                else CandidateEvaluation(
                    status=invocation.status
                    if invocation.status in {"failed", "timeout"}
                    else "failed",
                    score=None,
                    error=invocation.error,
                )
            )
            thread_id = invocation.metadata.get("thread_id")
            if thread_id:
                self._sessions[version.version_id] = str(thread_id)
            self.lineage.register_candidate(
                version.version_id,
                parent_version_id=parent_id,
                status=evaluation.status,
                score=evaluation.score,
            )
            result = CandidateResult(
                act_id=act_id,
                branch_index=branch_index,
                version=version,
                evaluation=evaluation,
                provider=invocation,
            )
            results.append(result)
            self._write_act_event(iteration_id, result)
            self._write_version_event(version, evaluation, selected=False)

        completed = [
            result for result in results if result.evaluation.status == "complete"
        ]
        selected = (
            max(completed, key=lambda result: result.evaluation.score)
            if completed
            else results[0]
        )
        promoted = self.lineage.select_version(selected.version.version_id)
        self.version_store.checkout(selected.version.version_id)
        if promoted:
            self.events.write(
                "champion_promoted",
                version_id=selected.version.version_id,
                score=selected.evaluation.score,
            )
        self.events.write(
            "candidate_selected",
            iteration_id=iteration_id,
            version_id=selected.version.version_id,
            act_id=selected.act_id,
        )
        selected_experience = pending_experience.get(selected.act_id)
        if self.experience_manager is not None and selected_experience is not None:
            path = self.experience_manager.apply_file(
                selected.act_id,
                selected_experience,
            )
            self.events.write(
                "experience_updated",
                act_id=selected.act_id,
                version_id=selected.version.version_id,
                experience_path=str(path),
            )
        return IterationResult(
            iteration_id=iteration_id,
            parent_version_id=parent_id,
            candidates=tuple(results),
            selected=selected,
            rollback=rollback,
        )

    def reached_iteration_limit(self) -> bool:
        return (
            self.iteration.max_acts is not None
            and self._coding_agent_acts >= self.iteration.max_acts
        )

    def summary(self) -> dict[str, Any]:
        complete_scores = [
            value.score
            for value in self.lineage.versions.values()
            if value.status == "complete" and value.score is not None
        ]
        return {
            "coding_agent_acts": self._coding_agent_acts,
            "iterations": self._iteration_count,
            "best_score": max(complete_scores) if complete_scores else None,
            "champion_version_id": self.lineage.champion_version_id,
            "lineage_head_version_id": self.lineage.lineage_head_version_id,
        }

    def _write_act_event(
        self,
        iteration_id: str,
        result: CandidateResult,
    ) -> None:
        usage = result.provider.usage
        self.events.write(
            "act_completed",
            act_id=result.act_id,
            iteration_id=iteration_id,
            branch_index=result.branch_index,
            status=result.provider.status,
            prompt_tokens=usage.prompt_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            total_tokens=usage.total_tokens,
            elapsed_time_s=result.provider.elapsed_time_s,
            raw_output_ref=result.provider.raw_output_ref,
            thread_id=result.provider.metadata.get("thread_id"),
        )

    def _write_checkpoint(
        self,
        *,
        act_id: str,
        iteration_id: str,
        branch_index: int,
        parent_version_id: str,
        prompt: str,
        invocation: ProviderInvocation,
    ) -> None:
        path = self.run_root / "checkpoints" / f"{act_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "1.0",
            "act_id": act_id,
            "iteration_id": iteration_id,
            "branch_index": branch_index,
            "parent_version_id": parent_version_id,
            "prompt": prompt,
            "provider_status": invocation.status,
            "thread_id": invocation.metadata.get("thread_id"),
            "provider_fingerprint": invocation.metadata.get(
                "provider_fingerprint"
            ),
            "raw_output_ref": invocation.raw_output_ref,
            "usage": dataclasses.asdict(invocation.usage),
        }
        path.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.events.write(
            "checkpoint_created",
            act_id=act_id,
            iteration_id=iteration_id,
            path=str(path),
            parent_version_id=parent_version_id,
            thread_id=invocation.metadata.get("thread_id"),
        )

    def _write_version_event(
        self,
        version: Version,
        evaluation: CandidateEvaluation,
        *,
        selected: bool,
    ) -> None:
        self.events.write(
            "version_created",
            version_id=version.version_id,
            parent_version_id=version.parent_version_id,
            act_id=version.act_id,
            content_hash=version.content_hash,
            edit_type=version.edit_type,
            evaluation_status=evaluation.status,
            benchmark_score=evaluation.score,
            selected=selected,
        )
        if evaluation.status == "complete":
            match_records = [
                dict(match)
                for match in evaluation.matches
                if match.get("status", "complete") == "complete"
                and match.get("result") in {"win", "draw", "loss"}
            ]
            for index, match in enumerate(match_records):
                match_id = str(
                    match.get(
                        "match_id",
                        f"{version.version_id}-{index:04d}",
                    )
                )
                self.events.write(
                    "match_completed",
                    match_id=match_id,
                    version_id=version.version_id,
                    act_id=version.act_id,
                    opponent=match.get("opponent"),
                    seed=match.get("seed"),
                    result=match.get("result"),
                    rollman_score=match.get("rollman_score"),
                    ghosts_score=match.get("ghosts_score"),
                    valid=True,
                    replay=match.get("replay"),
                    trace=match.get("trace"),
                )
                elo_record = self.elo_ledger.update_game(
                    role="rollman",
                    candidate=self.elo_candidate_id,
                    opponent=str(match.get("opponent")),
                    result=str(match["result"]),
                    act_id=version.act_id,
                    version_id=version.version_id,
                    seed=int(match["seed"]),
                    anchor_opponent=self.anchor_human_opponents,
                )
                self.events.write(
                    "elo_updated",
                    version_id=version.version_id,
                    act_id=version.act_id,
                    opponent=elo_record.opponent,
                    seed=elo_record.seed,
                    result=elo_record.result,
                    rating=elo_record.rating_after,
                    opponent_rating=elo_record.opponent_rating_after,
                    role=elo_record.role,
                )
            self.events.write(
                "evaluation_completed",
                version_id=version.version_id,
                status=evaluation.status,
                benchmark_score=evaluation.score,
                wins=sum(match["result"] == "win" for match in match_records),
                draws=sum(match["result"] == "draw" for match in match_records),
                losses=sum(match["result"] == "loss" for match in match_records),
                matches=list(evaluation.matches),
            )
