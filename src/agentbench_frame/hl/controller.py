"""Framework-owned HL act, candidate, evaluation, and rollback orchestration."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from agentbench_frame.arena.rating import RoleEloLedger
from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.config import IterationConfig, RollbackConfig
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.events import HLEventWriter
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.lineage import LineageManager, ParentDecision
from agentbench_frame.hl.proposal import BranchBrief, load_branch_briefs
from agentbench_frame.hl.research_state import ResearchState, apply_reducer_update
from agentbench_frame.hl.selection import (
    CandidateDiagnostics,
    select_linear_successor,
)
from agentbench_frame.tracking.provider import ProviderInvocation


@dataclasses.dataclass(frozen=True)
class CandidateResult:
    act_id: str
    branch_index: int
    version: Version
    evaluation: CandidateEvaluation
    provider: ProviderInvocation
    pending_experience_path: Optional[Path] = None


@dataclasses.dataclass(frozen=True)
class IterationResult:
    iteration_id: str
    parent_version_id: str
    candidates: tuple[CandidateResult, ...]
    selected: CandidateResult
    search_parent_version_id: str
    rollback: Optional[ParentDecision]
    finalists: tuple[CandidateResult, ...] = ()


@dataclasses.dataclass(frozen=True)
class ProposalCycleResult:
    iteration_id: str
    parent_version_id: str
    candidates: tuple[CandidateResult, ...]
    finalists: tuple[CandidateResult, ...]
    selected: CandidateResult
    search_parent_version_id: str
    planner: ProviderInvocation
    reducer: ProviderInvocation
    reducer_input_path: Path
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
        research_state_path: str | Path | None = None,
        research_state_max_bytes: int = 16384,
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
        self.research_state_path = (
            None if research_state_path is None else Path(research_state_path)
        )
        self.research_state_max_bytes = research_state_max_bytes
        self._iteration_count = 0
        self._coding_agent_acts = 0
        self._sessions: dict[str, str] = {}
        self._recorded_match_ids: set[str] = set()
        self._recorded_match_keys: set[
            tuple[str, str, str, int]
        ] = set()
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
        self._write_version_event(
            version,
            evaluation,
            selected=True,
            record_evaluation=evaluate,
        )
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

    def bootstrap(self) -> CandidateResult:
        """Use one coding-agent act to create the only scientific origin."""

        if self._started:
            raise RuntimeError("controller already initialized")
        self.events.write(
            "run_started",
            iteration_config=dataclasses.asdict(self.iteration),
        )
        iteration_id = "iter-000000"
        act_id = "act-000001-b00"
        parent_id = "bootstrap-scaffold"
        prompt = self.prompt_factory(
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=0,
            branch_count=1,
            parent_version_id=parent_id,
            bootstrap=True,
        )
        scaffold_content_hash = self.version_store.current_content_hash()
        raw_path = self.run_root / "provider" / f"{act_id}.jsonl"
        invocation = self.provider.invoke(
            prompt=prompt,
            workspace=self.workspace,
            raw_output_path=raw_path,
            session_id=None,
        )
        self._write_checkpoint(
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=0,
            parent_version_id=parent_id,
            prompt=prompt,
            invocation=invocation,
        )
        self._coding_agent_acts = 1
        if invocation.status != "completed":
            self._write_provider_event(
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=0,
                invocation=invocation,
            )
            raise RuntimeError(
                f"bootstrap provider did not complete: {invocation.status}"
            )
        if self.version_store.current_content_hash() == scaffold_content_hash:
            self._write_provider_event(
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=0,
                invocation=invocation,
            )
            raise RuntimeError(
                "bootstrap provider completed but did not change the candidate scaffold"
            )

        version = self.version_store.snapshot(
            parent_version_id=None,
            act_id=act_id,
        )
        evaluation = self.evaluator.evaluate(version)
        self.lineage.record_evaluation(
            version.version_id,
            parent_version_id=None,
            status=evaluation.status,
            score=evaluation.score,
        )
        result = CandidateResult(
            act_id=act_id,
            branch_index=0,
            version=version,
            evaluation=evaluation,
            provider=invocation,
        )
        thread_id = invocation.metadata.get("thread_id")
        if thread_id:
            self._sessions[version.version_id] = str(thread_id)
        self._write_act_event(iteration_id, result)
        self._write_version_event(version, evaluation, selected=True)
        self.events.write(
            "candidate_selected",
            iteration_id=iteration_id,
            version_id=version.version_id,
            act_id=act_id,
        )
        if evaluation.status == "complete":
            self.events.write(
                "champion_promoted",
                version_id=version.version_id,
                score=evaluation.score,
            )
        self._started = True
        return result

    def initialize_imported(
        self,
        *,
        source_run: str | Path,
        source_version_id: str,
    ) -> Version:
        """Register a verified external snapshot without a provider call."""

        if self._started:
            raise RuntimeError("controller already initialized")
        source_root = Path(source_run)
        self.events.write(
            "run_started",
            iteration_config=dataclasses.asdict(self.iteration),
        )
        source, version = self.version_store.import_version(
            source_root / "versions",
            source_version_id,
        )
        evaluation = CandidateEvaluation(
            status="incomplete",
            score=None,
            error="imported origin has not been evaluated",
        )
        self.lineage.record_evaluation(
            version.version_id,
            parent_version_id=None,
            status=evaluation.status,
            score=evaluation.score,
        )
        self._write_version_event(
            version,
            evaluation,
            selected=True,
            record_evaluation=False,
        )
        self.events.write(
            "candidate_selected",
            iteration_id="iter-000000",
            version_id=version.version_id,
            act_id=version.act_id,
        )
        self.events.write(
            "origin_imported",
            source_run_id=source_root.name,
            source_version_id=source.version_id,
            source_content_hash=source.content_hash,
            version_id=version.version_id,
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
        for event in historical_events:
            if (
                event.get("event_type") != "match_completed"
                or event.get("valid") is not True
                or event.get("result") not in {"win", "draw", "loss"}
            ):
                continue
            self._recorded_match_ids.add(str(event["match_id"]))
            self._recorded_match_keys.add(
                (
                    str(event["version_id"]),
                    str(event.get("phase") or "learning"),
                    str(event["opponent"]),
                    int(event["seed"]),
                )
            )
            self.elo_ledger.update_game(
                role=str(event.get("role") or "rollman"),
                candidate=str(event["version_id"]),
                opponent=str(event["opponent"]),
                result=str(event["result"]),
                act_id=str(event["act_id"]),
                version_id=str(event["version_id"]),
                seed=int(event["seed"]),
                anchor_opponent=self.anchor_human_opponents,
            )
        self._coding_agent_acts = sum(
            event.get("event_type") == "act_completed"
            for event in historical_events
        )
        selected_iterations = {
            str(event.get("iteration_id"))
            for event in historical_events
            if event.get("event_type")
            in {"candidate_selected", "search_parent_selected"}
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

    def retry_head_evaluation(self) -> CandidateEvaluation:
        """Retry local evaluation of an incomplete head without a provider call."""

        if not self._started:
            raise RuntimeError("controller must be initialized before evaluation retry")
        head = self.lineage.lineage_head_version_id
        if head is None:
            raise ValueError("evaluation retry requires a lineage head")
        version = self.version_store.get(head)
        evaluation = self.evaluator.evaluate(version)
        promoted = self.lineage.update_incomplete_evaluation(
            version.version_id,
            status=evaluation.status,
            score=evaluation.score,
        )
        self._write_evaluation_event(version, evaluation)
        if promoted:
            self.events.write(
                "champion_promoted",
                version_id=version.version_id,
                score=evaluation.score,
            )
        return evaluation

    def record_matches(
        self,
        *,
        version: Version,
        act_id: str,
        phase: str,
        matches: Any,
    ) -> int:
        """Append valid game and Elo events for an evaluated match series."""

        match_records = [
            dict(match)
            for match in matches
            if match.get("status", "complete") == "complete"
            and match.get("result") in {"win", "draw", "loss"}
        ]
        for index, match in enumerate(match_records):
            match_phase = str(match.get("phase") or phase)
            match_key = (
                version.version_id,
                match_phase,
                str(match.get("opponent")),
                int(match["seed"]),
            )
            if match_key in self._recorded_match_keys:
                continue
            fallback_match_id = (
                f"{version.version_id}-{match_phase}-{index:04d}"
            )
            if fallback_match_id in self._recorded_match_ids:
                fallback_match_id = (
                    f"{version.version_id}-{match_phase}-"
                    f"{match.get('opponent', 'unknown')}-"
                    f"{match.get('seed', index)}"
                )
            match_id = str(
                match.get(
                    "match_id",
                    fallback_match_id,
                )
            )
            if match_id in self._recorded_match_ids:
                continue
            self.events.write(
                "match_completed",
                match_id=match_id,
                version_id=version.version_id,
                act_id=act_id,
                phase=match_phase,
                role="rollman",
                opponent=match.get("opponent"),
                seed=match.get("seed"),
                result=match.get("result"),
                rollman_score=match.get("rollman_score"),
                ghosts_score=match.get("ghosts_score"),
                valid=True,
                replay=match.get("replay"),
                trace=match.get("trace"),
            )
            self._recorded_match_ids.add(match_id)
            self._recorded_match_keys.add(match_key)
            elo_record = self.elo_ledger.update_game(
                role="rollman",
                candidate=version.version_id,
                opponent=str(match.get("opponent")),
                result=str(match["result"]),
                act_id=act_id,
                version_id=version.version_id,
                seed=int(match["seed"]),
                anchor_opponent=self.anchor_human_opponents,
            )
            self.events.write(
                "elo_updated",
                match_id=match_id,
                version_id=version.version_id,
                act_id=act_id,
                phase=match_phase,
                candidate=elo_record.candidate,
                opponent=elo_record.opponent,
                seed=elo_record.seed,
                result=elo_record.result,
                rating_before=elo_record.rating_before,
                rating=elo_record.rating_after,
                opponent_rating=elo_record.opponent_rating_after,
                role=elo_record.role,
            )
        return len(match_records)

    def run_act(
        self,
        *,
        parent_version_id: str | None = None,
        defer_experience: bool = False,
        branch_briefs: tuple[BranchBrief, ...] | None = None,
        promote_champion: bool = True,
        parent_evaluation: CandidateEvaluation | None = None,
    ) -> IterationResult:
        if not self._started:
            raise RuntimeError("initialize must be called before run_act")
        parent_decision = (
            self.lineage.select_next_parent()
            if parent_version_id is None
            else self.lineage.force_parent(parent_version_id)
        )
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
        if branch_briefs is not None and len(branch_briefs) != branch_count:
            raise ValueError("branch briefs must match candidate count")
        staged_evaluation = (
            self.iteration.planner_enabled
            and hasattr(self.evaluator, "quick_screen")
            and hasattr(self.evaluator, "evaluate_finalist")
            and hasattr(self.evaluator, "combine_stages")
        )

        for branch_index in range(branch_count):
            self.version_store.checkout(parent_id)
            experience_update = (
                self.workspace / ".agentbench" / "experience_update.json"
            )
            if experience_update.exists():
                experience_update.unlink()
            act_id = f"act-{self._coding_agent_acts + 1:06d}-b{branch_index:02d}"
            prompt = self.prompt_factory(
                phase="candidate",
                act_id=act_id,
                iteration_id=iteration_id,
                branch_index=branch_index,
                branch_count=branch_count,
                parent_version_id=parent_id,
                branch_brief=(
                    None
                    if branch_briefs is None
                    else branch_briefs[branch_index].to_dict()
                ),
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
                (
                    self.evaluator.quick_screen(version)
                    if staged_evaluation
                    else self.evaluator.evaluate(version)
                )
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
                pending_experience_path=pending_experience.get(act_id),
            )
            results.append(result)
            self._write_act_event(iteration_id, result)
            self._write_version_event(version, evaluation, selected=False)

        def selection_key(result: CandidateResult) -> tuple[float, ...]:
            try:
                diagnostics = CandidateDiagnostics.from_matches(
                    version_id=result.version.version_id,
                    branch_index=result.branch_index,
                    matches=result.evaluation.matches,
                )
            except ValueError:
                return (
                    float(result.evaluation.score or 0.0),
                    float("-inf"),
                    float("-inf"),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(-result.branch_index),
                )
            return diagnostics.key()

        finalists: tuple[CandidateResult, ...] = ()
        if staged_evaluation:
            quick_completed = sorted(
                (
                    result
                    for result in results
                    if result.evaluation.status == "complete"
                ),
                key=selection_key,
                reverse=True,
            )
            finalist_ids = {
                result.version.version_id
                for result in quick_completed[: self.iteration.finalist_count]
            }
            updated_results: list[CandidateResult] = []
            for result in results:
                if result.version.version_id not in finalist_ids:
                    updated_results.append(result)
                    continue
                finalist_evaluation = self.evaluator.evaluate_finalist(
                    result.version
                )
                combined = self.evaluator.combine_stages(
                    result.evaluation,
                    finalist_evaluation,
                )
                updated = dataclasses.replace(result, evaluation=combined)
                updated_results.append(updated)
                self._write_evaluation_event(result.version, combined)
            results = updated_results
            finalists = tuple(
                result
                for result in results
                if result.version.version_id in finalist_ids
            )

        completed = [
            result for result in results if result.evaluation.status == "complete"
        ]

        selected = max(completed, key=selection_key) if completed else results[0]
        if not finalists:
            finalists = tuple(
                sorted(completed, key=selection_key, reverse=True)[
                    : self.iteration.finalist_count
                ]
            ) or (selected,)
        search_parent_version_id = selected.version.version_id
        if not promote_champion and parent_evaluation is not None:
            try:
                parent_diagnostics = CandidateDiagnostics.from_matches(
                    version_id=parent_id,
                    branch_index=-1,
                    matches=parent_evaluation.matches,
                )
                selected_diagnostics = CandidateDiagnostics.from_matches(
                    version_id=selected.version.version_id,
                    branch_index=selected.branch_index,
                    matches=selected.evaluation.matches,
                )
                search_parent_version_id = select_linear_successor(
                    parent_diagnostics,
                    (selected_diagnostics,),
                ).search_parent_version_id
            except ValueError:
                search_parent_version_id = selected.version.version_id
        promoted = (
            self.lineage.select_version(search_parent_version_id)
            if promote_champion
            else self.lineage.select_search_parent(search_parent_version_id)
        )
        self.version_store.checkout(search_parent_version_id)
        if promoted:
            self.events.write(
                "champion_promoted",
                version_id=selected.version.version_id,
                score=selected.evaluation.score,
            )
        self.events.write(
            "candidate_selected" if promote_champion else "search_parent_selected",
            iteration_id=iteration_id,
            version_id=search_parent_version_id,
            act_id=self.version_store.get(search_parent_version_id).act_id,
        )
        if (
            not defer_experience
            and search_parent_version_id == selected.version.version_id
        ):
            self.commit_experience(selected)
        return IterationResult(
            iteration_id=iteration_id,
            parent_version_id=parent_id,
            candidates=tuple(results),
            selected=selected,
            search_parent_version_id=search_parent_version_id,
            rollback=rollback,
            finalists=finalists,
        )

    def run_proposal_cycle(
        self,
        *,
        parent_version_id: str | None = None,
        defer_experience: bool = False,
        parent_evaluation: CandidateEvaluation | None = None,
    ) -> ProposalCycleResult:
        """Run one planner, four sibling candidates, and one reducer."""

        if not self._started:
            raise RuntimeError("initialize must be called before proposal cycle")
        if self.iteration.candidates_per_cycle != 4:
            raise ValueError("proposal cycle requires exactly four candidates")
        if not self.iteration.planner_enabled or not self.iteration.reducer_enabled:
            raise ValueError("proposal cycle requires planner and reducer")
        parent_id = (
            self.lineage.lineage_head_version_id
            if parent_version_id is None
            else parent_version_id
        )
        if parent_id is None:
            raise ValueError("proposal cycle requires a parent")
        self.version_store.checkout(parent_id)
        iteration_id = f"iter-{self._iteration_count + 1:06d}"
        self.events.write(
            "proposal_cycle_started",
            iteration_id=iteration_id,
            parent_version_id=parent_id,
            candidate_count=4,
        )

        control_root = self.workspace / ".agentbench"
        control_root.mkdir(parents=True, exist_ok=True)
        planner_output = control_root / "branch_briefs.json"
        if planner_output.exists():
            planner_output.unlink()
        planner_act_id = f"act-{self._coding_agent_acts + 1:06d}-planner"
        planner_prompt = self.prompt_factory(
            phase="planner",
            act_id=planner_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            branch_count=4,
            parent_version_id=parent_id,
            branch_brief=None,
            reducer_input=None,
        )
        planner_raw = self.run_root / "provider" / f"{planner_act_id}.jsonl"
        planner = self.provider.invoke(
            prompt=planner_prompt,
            workspace=self.workspace,
            raw_output_path=planner_raw,
            session_id=None,
        )
        self._coding_agent_acts += 1
        self._write_checkpoint(
            act_id=planner_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            parent_version_id=parent_id,
            prompt=planner_prompt,
            invocation=planner,
        )
        self._write_provider_event(
            act_id=planner_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            invocation=planner,
        )
        if planner.status != "completed" or not planner_output.is_file():
            raise RuntimeError("planner did not produce branch_briefs.json")
        proposal_root = self.run_root / "proposals" / iteration_id
        proposal_root.mkdir(parents=True, exist_ok=True)
        persisted_briefs = proposal_root / "branch_briefs.json"
        shutil.copy2(planner_output, persisted_briefs)
        briefs = load_branch_briefs(persisted_briefs, expected_count=4)
        self.events.write(
            "planner_completed",
            act_id=planner_act_id,
            iteration_id=iteration_id,
            status=planner.status,
            branch_briefs=str(persisted_briefs),
        )

        iteration = self.run_act(
            parent_version_id=parent_id,
            defer_experience=True,
            branch_briefs=briefs,
            promote_champion=False,
            parent_evaluation=parent_evaluation,
        )
        finalists = iteration.finalists
        if iteration.selected not in finalists:
            finalists = (iteration.selected, *finalists)[: self.iteration.finalist_count]
        self.events.write(
            "finalists_selected",
            iteration_id=iteration_id,
            version_ids=[candidate.version.version_id for candidate in finalists],
        )

        reducer_input = proposal_root / "reducer_input.json"
        reducer_input.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "iteration_id": iteration_id,
                    "parent_version_id": parent_id,
                    "selected_version_id": iteration.search_parent_version_id,
                    "best_candidate_version_id": (
                        iteration.selected.version.version_id
                    ),
                    "candidates": [
                        {
                            "branch_index": candidate.branch_index,
                            "act_id": candidate.act_id,
                            "version_id": candidate.version.version_id,
                            "status": candidate.evaluation.status,
                            "score": candidate.evaluation.score,
                            "brief": briefs[candidate.branch_index].to_dict(),
                            "matches": list(candidate.evaluation.matches),
                        }
                        for candidate in iteration.candidates
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.version_store.checkout(iteration.search_parent_version_id)
        reducer_output = control_root / "research_state_update.json"
        if reducer_output.exists():
            reducer_output.unlink()
        reducer_act_id = f"act-{self._coding_agent_acts + 1:06d}-reducer"
        reducer_prompt = self.prompt_factory(
            phase="reducer",
            act_id=reducer_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            branch_count=4,
            parent_version_id=parent_id,
            branch_brief=None,
            reducer_input=str(reducer_input),
        )
        reducer_raw = self.run_root / "provider" / f"{reducer_act_id}.jsonl"
        reducer = self.provider.invoke(
            prompt=reducer_prompt,
            workspace=self.workspace,
            raw_output_path=reducer_raw,
            session_id=None,
        )
        self._coding_agent_acts += 1
        self._write_checkpoint(
            act_id=reducer_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            parent_version_id=parent_id,
            prompt=reducer_prompt,
            invocation=reducer,
        )
        self._write_provider_event(
            act_id=reducer_act_id,
            iteration_id=iteration_id,
            branch_index=None,
            invocation=reducer,
        )
        persisted_reducer_output: str | None = None
        if reducer_output.is_file():
            target = proposal_root / "research_state_update.json"
            shutil.copy2(reducer_output, target)
            persisted_reducer_output = str(target)
            if self.research_state_path is not None:
                current_state = ResearchState.load_or_create(
                    self.research_state_path,
                    max_bytes=self.research_state_max_bytes,
                )
                next_state = apply_reducer_update(
                    current_state,
                    target,
                    proposal_cycle=self._iteration_count,
                    search_parent_version_id=(
                        iteration.search_parent_version_id
                    ),
                    official_champion_version_id=(
                        self.lineage.champion_version_id
                    ),
                    exploration_debt=current_state.exploration_debt + 1,
                )
                next_state.write(self.research_state_path)
        self.events.write(
            "reducer_completed",
            act_id=reducer_act_id,
            iteration_id=iteration_id,
            status=reducer.status,
            input_path=str(reducer_input),
            output_path=persisted_reducer_output,
        )
        self.events.write(
            "proposal_cycle_completed",
            iteration_id=iteration_id,
            parent_version_id=parent_id,
            selected_version_id=iteration.search_parent_version_id,
            candidate_version_ids=[
                candidate.version.version_id for candidate in iteration.candidates
            ],
        )
        if (
            not defer_experience
            and iteration.search_parent_version_id
            == iteration.selected.version.version_id
        ):
            self.commit_experience(iteration.selected)
        return ProposalCycleResult(
            iteration_id=iteration_id,
            parent_version_id=parent_id,
            candidates=iteration.candidates,
            finalists=finalists,
            selected=iteration.selected,
            search_parent_version_id=iteration.search_parent_version_id,
            planner=planner,
            reducer=reducer,
            reducer_input_path=reducer_input,
            rollback=iteration.rollback,
        )

    def commit_experience(self, candidate: CandidateResult) -> Path | None:
        """Commit a selected candidate's staged Experience update once."""

        pending = candidate.pending_experience_path
        if self.experience_manager is None or pending is None:
            return None
        path = self.experience_manager.apply_file(candidate.act_id, pending)
        self.events.write(
            "experience_updated",
            act_id=candidate.act_id,
            version_id=candidate.version.version_id,
            experience_path=str(path),
        )
        return path

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
        self._write_provider_event(
            act_id=result.act_id,
            iteration_id=iteration_id,
            branch_index=result.branch_index,
            invocation=result.provider,
        )

    def _write_provider_event(
        self,
        *,
        act_id: str,
        iteration_id: str,
        branch_index: int | None,
        invocation: ProviderInvocation,
    ) -> None:
        usage = invocation.usage
        self.events.write(
            "act_completed",
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=branch_index,
            status=invocation.status,
            prompt_tokens=usage.prompt_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            total_tokens=usage.total_tokens,
            elapsed_time_s=invocation.elapsed_time_s,
            raw_output_ref=invocation.raw_output_ref,
            thread_id=invocation.metadata.get("thread_id"),
        )

    def _write_checkpoint(
        self,
        *,
        act_id: str,
        iteration_id: str,
        branch_index: int | None,
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
        record_evaluation: bool = True,
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
        if record_evaluation:
            self._write_evaluation_event(version, evaluation)

    def _write_evaluation_event(
        self,
        version: Version,
        evaluation: CandidateEvaluation,
    ) -> None:
        match_records = [
            dict(match)
            for match in evaluation.matches
            if match.get("status", "complete") == "complete"
            and match.get("result") in {"win", "draw", "loss"}
        ]
        self.record_matches(
            version=version,
            act_id=version.act_id,
            phase="learning",
            matches=match_records,
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
