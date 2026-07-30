"""Framework-owned HL act, candidate, evaluation, and rollback orchestration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional

from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.config import IterationConfig, RollbackConfig
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.events import HLEventWriter
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
        if evaluation.status == "complete":
            self.events.write(
                "champion_promoted",
                version_id=version.version_id,
                score=evaluation.score,
            )
        self._started = True
        return version

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
        branch_count = self.iteration.candidates_per_act

        for branch_index in range(branch_count):
            self.version_store.checkout(parent_id)
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
            self._coding_agent_acts += 1
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
            self.events.write(
                "evaluation_completed",
                version_id=version.version_id,
                status=evaluation.status,
                benchmark_score=evaluation.score,
                matches=list(evaluation.matches),
            )
