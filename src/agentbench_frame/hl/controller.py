"""Framework-owned HL act, candidate, evaluation, and rollback orchestration."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agentbench_frame.arena.rating import RoleEloLedger
from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.config import IterationConfig, RollbackConfig
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.events import HLEventWriter
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.experience_ledger import derive_experience_record
from agentbench_frame.hl.lineage import LineageManager, ParentDecision
from agentbench_frame.hl.proposal import (
    BranchBrief,
    build_candidate_code_index,
    branch_briefs_json_schema,
    load_branch_briefs,
)
from agentbench_frame.hl.repair import (
    RepairSelection,
    build_repair_packet,
    select_branch_representative,
)
from agentbench_frame.hl.research_state import (
    ResearchState,
    apply_reducer_update,
    prepend_framework_comparisons,
)
from agentbench_frame.hl.selection import (
    CandidateDiagnostics,
    select_linear_successor,
)
from agentbench_frame.tracking.provider import ProviderInvocation


def _counts_as_coding_act(
    *,
    status: str,
    total_tokens: int | None,
    tool_call_count: int,
) -> bool:
    """Count model work, but not a zero-work infrastructure failure."""

    return (
        status == "completed"
        or total_tokens is not None
        or tool_call_count > 0
    )


@dataclasses.dataclass(frozen=True)
class CandidateResult:
    act_id: str
    branch_index: int
    version: Version
    evaluation: CandidateEvaluation
    provider: ProviderInvocation
    pending_experience_path: Optional[Path] = None
    activation: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass(frozen=True)
class IterationResult:
    iteration_id: str
    parent_version_id: str
    candidates: tuple[CandidateResult, ...]
    selected: CandidateResult
    search_parent_version_id: str
    rollback: Optional[ParentDecision]
    finalists: tuple[CandidateResult, ...] = ()
    repairs: tuple[RepairSelection, ...] = ()
    representatives: tuple[CandidateResult, ...] = ()


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
    branch_briefs: tuple[BranchBrief, ...] = ()
    repairs: tuple[RepairSelection, ...] = ()
    representatives: tuple[CandidateResult, ...] = ()


def _positive_margin_deltas(
    parent_evaluation: CandidateEvaluation | None,
    representatives: tuple[CandidateResult, ...],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return the strongest same-opponent, same-seed gains over the parent."""

    if parent_evaluation is None:
        return []

    def margin(match: Mapping[str, Any]) -> float | None:
        dense = match.get("dense_margin")
        if isinstance(dense, (int, float)) and not isinstance(dense, bool):
            return float(dense)
        rollman = match.get("rollman_score")
        ghosts = match.get("ghosts_score")
        if not isinstance(rollman, (int, float)) or not isinstance(
            ghosts, (int, float)
        ):
            return None
        return float(rollman) - float(ghosts)

    def score_delta(
        candidate: Mapping[str, Any],
        parent: Mapping[str, Any],
        generic_field: str,
        legacy_field: str,
    ) -> float | None:
        candidate_value = candidate.get(generic_field)
        parent_value = parent.get(generic_field)
        if not isinstance(candidate_value, (int, float)) or not isinstance(
            parent_value, (int, float)
        ):
            candidate_value = candidate.get(legacy_field)
            parent_value = parent.get(legacy_field)
        if (
            isinstance(candidate_value, bool)
            or isinstance(parent_value, bool)
            or not isinstance(candidate_value, (int, float))
            or not isinstance(parent_value, (int, float))
        ):
            return None
        return float(candidate_value) - float(parent_value)

    def comparison_key(match: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return (
            match.get("opponent"),
            match.get("candidate_role") or match.get("role") or "rollman",
            match.get("seed"),
        )

    parent_by_match: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for match in parent_evaluation.matches:
        if match.get("status") == "complete" and margin(match) is not None:
            parent_by_match[comparison_key(match)] = match

    improvements: list[dict[str, Any]] = []
    for representative in representatives:
        for match in representative.evaluation.matches:
            parent = parent_by_match.get(comparison_key(match))
            candidate_margin = margin(match)
            parent_margin = None if parent is None else margin(parent)
            if (
                match.get("status") != "complete"
                or parent is None
                or candidate_margin is None
                or parent_margin is None
                or candidate_margin <= parent_margin
            ):
                continue
            improvement = {
                "version_id": representative.version.version_id,
                "branch_index": representative.branch_index,
                "opponent": match.get("opponent"),
                "seed": match.get("seed"),
                "parent_result": parent.get("result"),
                "candidate_result": match.get("result"),
                "parent_margin": parent_margin,
                "candidate_margin": candidate_margin,
                "margin_delta": candidate_margin - parent_margin,
            }
            if "dense_margin" in match or "candidate_role" in match:
                improvement.update(
                    {
                        "candidate_role": (
                            match.get("candidate_role")
                            or match.get("role")
                            or "rollman"
                        ),
                        "candidate_score_delta": score_delta(
                            match, parent, "candidate_score", "rollman_score"
                        ),
                        "opponent_score_delta": score_delta(
                            match, parent, "opponent_score", "ghosts_score"
                        ),
                    }
                )
            else:
                improvement.update(
                    {
                        "rollman_score_delta": score_delta(
                            match, parent, "candidate_score", "rollman_score"
                        ),
                        "ghosts_score_delta": score_delta(
                            match, parent, "opponent_score", "ghosts_score"
                        ),
                    }
                )
            improvements.append(improvement)
    return sorted(
        improvements,
        key=lambda row: (
            -row["margin_delta"],
            row["branch_index"],
            str(row["opponent"]),
            str(row["seed"]),
        ),
    )[:limit]


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
        summary_resolver: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
        activation_probe: Callable[..., Mapping[str, Any]] | None = None,
        candidate_smoke_verifier: (
            Callable[..., Mapping[str, Any]] | None
        ) = None,
        candidate_source_relative: str = "ai.py",
        policy_entry_symbol: str = "ai_func",
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
        self.summary_resolver = summary_resolver or (
            lambda match: {
                key: match.get(key)
                for key in ("replay", "trace")
                if match.get(key) is not None
            }
        )
        self.activation_probe = activation_probe
        self.candidate_smoke_verifier = candidate_smoke_verifier
        source = Path(candidate_source_relative)
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("candidate source must stay inside workspace")
        self.candidate_source_relative = source
        if not policy_entry_symbol.strip():
            raise ValueError("policy entry symbol cannot be empty")
        self.policy_entry_symbol = policy_entry_symbol.strip()
        self._iteration_count = 0
        self._coding_agent_acts = 0
        self._provider_attempts = 0
        self._sessions: dict[str, str] = {}
        self._recorded_match_ids: set[str] = set()
        self._recorded_match_keys: set[
            tuple[str, str, str, str, int]
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
            **(
                {"reasoning_effort": "high"}
                if getattr(self.provider, "supports_structured_output", False)
                else {}
            ),
        )
        self._write_checkpoint(
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=0,
            parent_version_id=parent_id,
            prompt=prompt,
            invocation=invocation,
        )
        self._provider_attempts = 1
        self._coding_agent_acts = int(
            _counts_as_coding_act(
                status=invocation.status,
                total_tokens=invocation.usage.total_tokens,
                tool_call_count=invocation.tool_call_count,
            )
        )
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

    def recover_bootstrap(
        self,
        *,
        failed_act_id: str,
        raw_output_ref: str,
        failure_reason: str,
    ) -> Version:
        """Adopt a fully materialized bootstrap after a terminal stream failure."""

        if self._started:
            raise RuntimeError("controller already initialized")
        if not any(path.is_file() for path in self.workspace.iterdir()):
            raise ValueError("recovered bootstrap workspace is empty")
        version = self.version_store.snapshot(
            parent_version_id=None,
            act_id=f"{failed_act_id}-recovered",
            edit_type="recovered_bootstrap",
        )
        evaluation = self.evaluator.evaluate(version)
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
        self.events.write(
            "bootstrap_recovered",
            failed_act_id=failed_act_id,
            raw_output_ref=raw_output_ref,
            failure_reason=failure_reason,
            version_id=version.version_id,
            content_hash=version.content_hash,
        )
        if evaluation.status == "complete":
            self.events.write(
                "champion_promoted",
                version_id=version.version_id,
                score=evaluation.score,
            )
        self._coding_agent_acts = 1
        self._provider_attempts = 1
        self._started = True
        return version

    def initialize_imported(
        self,
        *,
        source_run: str | Path,
        source_version_id: str,
        evaluate: bool = False,
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
        evaluation = (
            self.evaluator.evaluate(version)
            if evaluate
            else CandidateEvaluation(
                status="incomplete",
                score=None,
                error="imported origin has not been evaluated",
            )
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
        self.events.write(
            "origin_imported",
            source_run_id=source_root.name,
            source_version_id=source.version_id,
            source_content_hash=source.content_hash,
            version_id=version.version_id,
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
                    str(
                        event.get("candidate_role")
                        or event.get("role")
                        or "rollman"
                    ),
                    int(event["seed"]),
                )
            )
            self.elo_ledger.update_game(
                role=str(
                    event.get("candidate_role")
                    or event.get("role")
                    or "rollman"
                ),
                candidate=str(event["version_id"]),
                opponent=str(event["opponent"]),
                result=str(event["result"]),
                act_id=str(event["act_id"]),
                version_id=str(event["version_id"]),
                seed=int(event["seed"]),
                anchor_opponent=self.anchor_human_opponents,
            )
        act_events = [
            event
            for event in historical_events
            if event.get("event_type") == "act_completed"
        ]
        self._provider_attempts = len(act_events)
        self._coding_agent_acts = sum(
            _counts_as_coding_act(
                status=str(event.get("status") or "failed"),
                total_tokens=(
                    int(event["total_tokens"])
                    if isinstance(event.get("total_tokens"), int)
                    else None
                ),
                tool_call_count=int(event.get("tool_call_count") or 0),
            )
            for event in act_events
        )
        if self.iteration.planner_enabled and self.iteration.reducer_enabled:
            completed_iterations = {
                str(event.get("iteration_id"))
                for event in historical_events
                if event.get("event_type") == "proposal_cycle_completed"
            }
        else:
            completed_iterations = {
                str(event.get("iteration_id"))
                for event in historical_events
                if event.get("event_type")
                in {"candidate_selected", "search_parent_selected"}
                and event.get("iteration_id") != "iter-000000"
            }
        self._iteration_count = len(completed_iterations)
        self._started = True
        self.events.write(
            "run_resumed",
            coding_agent_acts=self._coding_agent_acts,
            provider_attempts=self._provider_attempts,
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

    def _retry_recovered_activation_failure(
        self,
        candidate: CandidateResult,
        *,
        parent_evaluation: CandidateEvaluation | None,
    ) -> CandidateResult:
        """Retry framework-owned activation/evaluation without another model act."""

        error = str(candidate.evaluation.error or "")
        if (
            not error.startswith("activation_probe_failed:")
            or self.activation_probe is None
            or parent_evaluation is None
            or not hasattr(self.evaluator, "quick_screen")
        ):
            return candidate
        parent_id = candidate.version.parent_version_id
        if parent_id is None:
            return candidate
        iteration_id = str(
            candidate.provider.metadata.get("iteration_id", "iter-unknown")
        )
        try:
            measured = dict(
                self.activation_probe(
                    new_version=candidate.version,
                    old_version=self.version_store.get(parent_id),
                    version_store=self.version_store,
                    parent_evaluation=parent_evaluation,
                )
            )
            decision_count = int(measured["decision_count"])
            changed_action_count = int(measured["changed_action_count"])
            if (
                decision_count < 1
                or changed_action_count < 0
                or changed_action_count > decision_count
            ):
                raise ValueError("invalid activation probe counts")
            activation = {
                **measured,
                "status": "complete",
                "decision_count": decision_count,
                "changed_action_count": changed_action_count,
                "changed_fraction": changed_action_count / decision_count,
                "episodes": list(measured.get("episodes", ())),
            }
            activation_error = (
                "no_parent_trace_action_change"
                if changed_action_count == 0
                else None
            )
        except Exception as exception:
            message = (
                " ".join(str(exception).split())
                or exception.__class__.__name__
            )
            activation_error = f"activation_probe_failed: {message}"
            activation = {
                "status": "failed",
                "decision_count": 0,
                "changed_action_count": 0,
                "changed_fraction": 0.0,
                "episodes": [],
                "error": message,
            }
        self.events.write(
            "candidate_activation_measured",
            iteration_id=iteration_id,
            act_id=candidate.act_id,
            branch_index=candidate.branch_index,
            version_id=candidate.version.version_id,
            parent_version_id=parent_id,
            status=str(activation["status"]),
            decision_count=int(activation["decision_count"]),
            changed_action_count=int(activation["changed_action_count"]),
            changed_fraction=float(activation["changed_fraction"]),
            episodes=list(activation["episodes"]),
            error=activation.get("error"),
        )
        if activation_error is None:
            try:
                evaluation = self.evaluator.quick_screen(candidate.version)
            except Exception as exception:
                message = (
                    " ".join(str(exception).split())
                    or exception.__class__.__name__
                )
                evaluation = CandidateEvaluation(
                    status="failed",
                    score=None,
                    error=f"evaluation_retry_failed: {message}",
                )
        else:
            evaluation = CandidateEvaluation(
                status="failed",
                score=None,
                error=activation_error,
            )
        self.lineage.update_incomplete_evaluation(
            candidate.version.version_id,
            status=evaluation.status,
            score=evaluation.score,
            select=False,
        )
        self._write_evaluation_event(candidate.version, evaluation)
        return dataclasses.replace(
            candidate,
            evaluation=evaluation,
            activation=activation,
        )

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
        emitted = 0
        for index, match in enumerate(match_records):
            match_phase = str(match.get("phase") or phase)
            candidate_role = str(
                match.get("candidate_role")
                or match.get("role")
                or "rollman"
            )
            match_key = (
                version.version_id,
                match_phase,
                str(match.get("opponent")),
                candidate_role,
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
                    f"{candidate_role}-"
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
                game=match.get("game"),
                role=candidate_role,
                candidate_role=candidate_role,
                opponent=match.get("opponent"),
                seed=match.get("seed"),
                result=match.get("result"),
                points=match.get("points"),
                candidate_score=match.get("candidate_score"),
                opponent_score=match.get("opponent_score"),
                dense_margin=match.get("dense_margin"),
                terminal_metrics=match.get("terminal_metrics", {}),
                rounds=match.get("rounds"),
                faults=match.get("faults", []),
                live_opponent=match.get("live_opponent", True),
                rollman_score=match.get("rollman_score"),
                ghosts_score=match.get("ghosts_score"),
                valid=True,
                replay=match.get("replay"),
                trace=match.get("trace"),
            )
            self._recorded_match_ids.add(match_id)
            self._recorded_match_keys.add(match_key)
            emitted += 1
            elo_record = self.elo_ledger.update_game(
                role=candidate_role,
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
        return emitted

    def _run_coding_candidate(
        self,
        *,
        iteration_id: str,
        act_id: str,
        branch_index: int,
        branch_count: int,
        parent_version_id: str,
        phase: str,
        prompt_values: dict[str, Any],
        edit_type: str,
        staged_evaluation: bool,
        parent_evaluation: CandidateEvaluation | None = None,
        session_id: str | None = None,
    ) -> CandidateResult:
        """Invoke, snapshot, quick-screen, and register one immutable edit."""

        self.version_store.checkout(parent_version_id)
        experience_update = self.workspace / ".agentbench" / "experience_update.json"
        if experience_update.exists():
            experience_update.unlink()
        prompt = self.prompt_factory(
            phase=phase,
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=branch_index,
            branch_count=branch_count,
            parent_version_id=parent_version_id,
            **prompt_values,
        )
        raw_path = self.run_root / "provider" / f"{act_id}.jsonl"
        phase_overrides = (
            {"reasoning_effort": "high"}
            if getattr(self.provider, "supports_structured_output", False)
            else {}
        )
        invocation = self.provider.invoke(
            prompt=prompt,
            workspace=self.workspace,
            raw_output_path=raw_path,
            session_id=session_id,
            **phase_overrides,
        )
        self._write_checkpoint(
            act_id=act_id,
            iteration_id=iteration_id,
            branch_index=branch_index,
            parent_version_id=parent_version_id,
            prompt=prompt,
            invocation=invocation,
        )
        self._provider_attempts += 1
        self._coding_agent_acts += int(
            _counts_as_coding_act(
                status=invocation.status,
                total_tokens=invocation.usage.total_tokens,
                tool_call_count=invocation.tool_call_count,
            )
        )
        pending_path: Path | None = None
        if experience_update.is_file():
            pending_path = (
                self.run_root / "experience" / "pending" / f"{act_id}.json"
            )
            pending_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path.write_bytes(experience_update.read_bytes())
        parent_content_hash = self.version_store.get(
            parent_version_id
        ).content_hash
        version = self.version_store.snapshot(
            parent_version_id=parent_version_id,
            act_id=act_id,
            edit_type=edit_type,
        )
        budget_exhausted = bool(
            invocation.metadata.get("rollout_budget_exhausted")
        )
        budget_source_eligible = (
            invocation.status == "failed"
            and budget_exhausted
            and version.content_hash != parent_content_hash
            and not invocation.metadata.get("access_policy_violations")
        )
        if budget_exhausted:
            invocation.metadata["accepted_after_budget_exhaustion"] = False
        provider_pre_smoke_eligible = (
            invocation.status == "completed" or budget_source_eligible
        )
        candidate_smoke_error: str | None = None
        if provider_pre_smoke_eligible and staged_evaluation:
            smoke: dict[str, Any]
            if self.candidate_smoke_verifier is None:
                candidate_smoke_error = (
                    "candidate_smoke_failed: framework smoke verifier is unavailable"
                )
                smoke = {
                    "status": "failed",
                    "error": "framework smoke verifier is unavailable",
                }
            else:
                try:
                    measured_smoke = dict(
                        self.candidate_smoke_verifier(
                            act_id=act_id,
                            iteration_id=iteration_id,
                            branch_index=branch_index,
                            version=version,
                            parent_version=self.version_store.get(
                                parent_version_id
                            ),
                            workspace=self.workspace,
                            run_root=self.run_root,
                        )
                    )
                    if measured_smoke.get("status") != "complete":
                        raise ValueError("verifier did not report complete")
                    audit_root = self.run_root / "smoke" / act_id
                    audit_root.mkdir(parents=True, exist_ok=True)
                    smoke = dict(measured_smoke)
                    for key, filename in (
                        ("scenario_path", "smoke_scenario.json"),
                        ("result_path", "candidate_smoke_result.json"),
                    ):
                        raw_source = measured_smoke.get(key)
                        if raw_source is None:
                            continue
                        source = Path(str(raw_source))
                        if not source.is_file():
                            raise ValueError(f"verifier {key} is missing")
                        target = audit_root / filename
                        shutil.copy2(source, target)
                        smoke[key] = str(target)
                    smoke["status"] = "complete"
                except Exception as error:
                    message = (
                        " ".join(str(error).split())
                        or error.__class__.__name__
                    )
                    candidate_smoke_error = (
                        f"candidate_smoke_failed: {message}"
                    )
                    smoke = {"status": "failed", "error": message}
            invocation.metadata["candidate_smoke"] = smoke
        provider_eligible = (
            provider_pre_smoke_eligible and candidate_smoke_error is None
        )
        budget_candidate_eligible = (
            budget_source_eligible and candidate_smoke_error is None
        )
        activation: dict[str, Any] | None = None
        activation_error: str | None = None
        if (
            provider_eligible
            and staged_evaluation
            and self.activation_probe is not None
            and parent_evaluation is not None
            and parent_evaluation.status == "complete"
        ):
            try:
                measured = dict(
                    self.activation_probe(
                        new_version=version,
                        old_version=self.version_store.get(parent_version_id),
                        version_store=self.version_store,
                        parent_evaluation=parent_evaluation,
                    )
                )
                decision_count = int(measured["decision_count"])
                changed_action_count = int(measured["changed_action_count"])
                if (
                    decision_count < 1
                    or changed_action_count < 0
                    or changed_action_count > decision_count
                ):
                    raise ValueError("invalid activation probe counts")
                activation = {
                    **measured,
                    "status": "complete",
                    "decision_count": decision_count,
                    "changed_action_count": changed_action_count,
                    "changed_fraction": changed_action_count / decision_count,
                    "episodes": list(measured.get("episodes", ())),
                }
                if changed_action_count == 0:
                    activation_error = "no_parent_trace_action_change"
            except Exception as error:
                message = " ".join(str(error).split()) or error.__class__.__name__
                activation_error = f"activation_probe_failed: {message}"
                activation = {
                    "status": "failed",
                    "decision_count": 0,
                    "changed_action_count": 0,
                    "changed_fraction": 0.0,
                    "episodes": [],
                    "error": message,
                }
            self.events.write(
                "candidate_activation_measured",
                iteration_id=iteration_id,
                act_id=act_id,
                branch_index=branch_index,
                version_id=version.version_id,
                parent_version_id=parent_version_id,
                status=str(activation["status"]),
                decision_count=int(activation["decision_count"]),
                changed_action_count=int(activation["changed_action_count"]),
                changed_fraction=float(activation["changed_fraction"]),
                episodes=list(activation["episodes"]),
                error=activation.get("error"),
            )
        evaluation = (
            (
                self.evaluator.quick_screen(version)
                if staged_evaluation
                else self.evaluator.evaluate(version)
            )
            if provider_eligible and activation_error is None
            else CandidateEvaluation(
                status=(
                    invocation.status
                    if not provider_eligible
                    and invocation.status in {"failed", "timeout"}
                    else "failed"
                ),
                score=None,
                error=(
                    candidate_smoke_error
                    or activation_error
                    or invocation.error
                ),
            )
        )
        if budget_candidate_eligible:
            if evaluation.status == "complete":
                invocation.metadata["original_provider_status"] = invocation.status
                invocation.metadata["original_provider_error"] = invocation.error
                invocation.metadata["accepted_after_budget_exhaustion"] = True
                invocation.status = "completed"
                invocation.error = None
            else:
                evaluation = CandidateEvaluation(
                    status="failed",
                    score=None,
                    error=(
                        "budget-terminated candidate failed safe quick screen: "
                        + str(evaluation.error or evaluation.status)
                    ),
                    matches=evaluation.matches,
                )
        self._refresh_checkpoint_outcome(
            act_id=act_id,
            invocation=invocation,
        )
        thread_id = invocation.metadata.get("thread_id")
        if thread_id:
            self._sessions[version.version_id] = str(thread_id)
        self.lineage.register_candidate(
            version.version_id,
            parent_version_id=parent_version_id,
            status=evaluation.status,
            score=evaluation.score,
        )
        result = CandidateResult(
            act_id=act_id,
            branch_index=branch_index,
            version=version,
            evaluation=evaluation,
            provider=invocation,
            pending_experience_path=pending_path,
            activation=activation,
        )
        self._write_act_event(iteration_id, result)
        self._write_version_event(version, evaluation, selected=False)
        return result

    def run_act(
        self,
        *,
        parent_version_id: str | None = None,
        defer_experience: bool = False,
        branch_briefs: tuple[BranchBrief, ...] | None = None,
        promote_champion: bool = True,
        parent_evaluation: CandidateEvaluation | None = None,
        candidate_recoveries: Mapping[int, CandidateResult] | None = None,
        repair_recoveries: Mapping[int, CandidateResult] | None = None,
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
            recovered_candidate = (candidate_recoveries or {}).get(branch_index)
            if recovered_candidate is not None:
                if (
                    recovered_candidate.branch_index != branch_index
                    or recovered_candidate.version.parent_version_id != parent_id
                ):
                    raise ValueError("recovered candidate lineage does not match")
                recovered_candidate = self._retry_recovered_activation_failure(
                    recovered_candidate,
                    parent_evaluation=parent_evaluation,
                )
                results.append(recovered_candidate)
                continue
            act_id = f"act-{self._provider_attempts + 1:06d}-b{branch_index:02d}"
            results.append(
                self._run_coding_candidate(
                    iteration_id=iteration_id,
                    act_id=act_id,
                    branch_index=branch_index,
                    branch_count=branch_count,
                    parent_version_id=parent_id,
                    phase="candidate",
                    prompt_values={
                        "branch_brief": (
                            None
                            if branch_briefs is None
                            else branch_briefs[branch_index].to_dict()
                        )
                    },
                    edit_type="candidate",
                    staged_evaluation=staged_evaluation,
                    parent_evaluation=parent_evaluation,
                    session_id=(
                        self._sessions.get(parent_id) if branch_count == 1 else None
                    ),
                )
            )

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

        repairs: list[RepairSelection] = []
        representatives: list[CandidateResult] = list(results)
        if self.iteration.repair_enabled and self.iteration.repair_rounds:
            if branch_briefs is None:
                raise ValueError("repair cycle requires branch briefs")
            if parent_evaluation is None:
                raise ValueError("repair cycle requires the parent evaluation")
            parent_result = CandidateResult(
                act_id=self.version_store.get(parent_id).act_id,
                branch_index=-1,
                version=self.version_store.get(parent_id),
                evaluation=parent_evaluation,
                provider=ProviderInvocation(status="completed"),
            )
            eligible = sorted(
                (
                    result
                    for result in results
                    if result.evaluation.status == "complete"
                ),
                key=selection_key,
                reverse=True,
            )
            repair_targets = eligible[: self.iteration.repair_top_k]
            self.events.write(
                "repairs_selected",
                iteration_id=iteration_id,
                branch_indices=[target.branch_index for target in repair_targets],
                version_ids=[
                    target.version.version_id for target in repair_targets
                ],
            )
            proposal_root = self.run_root / "proposals" / iteration_id
            for initial in repair_targets:
                repair_input = build_repair_packet(
                    output_path=(
                        proposal_root
                        / f"repair_input-b{initial.branch_index:02d}.json"
                    ),
                    iteration_id=iteration_id,
                    branch_brief=branch_briefs[initial.branch_index],
                    parent=parent_result,
                    candidate=initial,
                    summary_resolver=self.summary_resolver,
                )
                act_id = (
                    f"act-{self._provider_attempts + 1:06d}"
                    f"-repair-b{initial.branch_index:02d}"
                )
                recovered = (repair_recoveries or {}).get(initial.branch_index)
                if recovered is not None:
                    if recovered.branch_index != initial.branch_index:
                        raise ValueError("recovered repair branch does not match")
                    recovered_parent_id = recovered.version.parent_version_id
                    if recovered_parent_id is None:
                        raise ValueError("recovered repair requires a parent version")
                    recovered_parent = self.version_store.get(recovered_parent_id)
                    if recovered_parent.content_hash != initial.version.content_hash:
                        recovered = None
                if recovered is None:
                    self.events.write(
                        "repair_started",
                        iteration_id=iteration_id,
                        act_id=act_id,
                        branch_index=initial.branch_index,
                        initial_version_id=initial.version.version_id,
                        repair_input_path=str(repair_input),
                    )
                    repaired = self._run_coding_candidate(
                        iteration_id=iteration_id,
                        act_id=act_id,
                        branch_index=initial.branch_index,
                        branch_count=branch_count,
                        parent_version_id=initial.version.version_id,
                        phase="repair",
                        prompt_values={
                            "branch_brief": branch_briefs[
                                initial.branch_index
                            ].to_dict(),
                            "repair_input": str(repair_input),
                        },
                        edit_type="repair",
                        staged_evaluation=staged_evaluation,
                        parent_evaluation=initial.evaluation,
                    )
                    self.events.write(
                        "repair_completed",
                        iteration_id=iteration_id,
                        act_id=act_id,
                        branch_index=initial.branch_index,
                        initial_version_id=initial.version.version_id,
                        repaired_version_id=repaired.version.version_id,
                        status=repaired.provider.status,
                    )
                else:
                    repaired = recovered
                representative = select_branch_representative(initial, repaired)
                representatives[initial.branch_index] = representative
                repairs.append(
                    RepairSelection(
                        branch_index=initial.branch_index,
                        initial=initial,
                        repaired=repaired,
                        representative=representative,
                        repair_input_path=repair_input,
                    )
                )
            repair_by_branch = {repair.branch_index: repair for repair in repairs}
            for initial, representative in zip(results, representatives):
                repair = repair_by_branch.get(initial.branch_index)
                self.events.write(
                    "branch_representative_selected",
                    iteration_id=iteration_id,
                    branch_index=initial.branch_index,
                    initial_version_id=initial.version.version_id,
                    repaired_version_id=(
                        None
                        if repair is None or repair.repaired is None
                        else repair.repaired.version.version_id
                    ),
                    representative_version_id=representative.version.version_id,
                    reason=(
                        "not_repaired"
                        if repair is None
                        else "repair_strictly_improved"
                        if representative.version.version_id
                        == repair.repaired.version.version_id
                        else "initial_retained"
                    ),
                )

        finalists: tuple[CandidateResult, ...] = ()
        if staged_evaluation:
            quick_completed = sorted(
                (
                    result
                    for result in representatives
                    if result.evaluation.status == "complete"
                ),
                key=selection_key,
                reverse=True,
            )
            finalist_ids = {
                result.version.version_id
                for result in quick_completed[: self.iteration.finalist_count]
            }
            updated_representatives: list[CandidateResult] = []
            for result in representatives:
                if result.version.version_id not in finalist_ids:
                    updated_representatives.append(result)
                    continue
                self.version_store.checkout(result.version.version_id)
                finalist_evaluation = self.evaluator.evaluate_finalist(
                    result.version
                )
                combined = self.evaluator.combine_stages(
                    result.evaluation,
                    finalist_evaluation,
                )
                updated = dataclasses.replace(result, evaluation=combined)
                updated_representatives.append(updated)
                self._write_evaluation_event(result.version, combined)
            representatives = updated_representatives
            updated_by_version = {
                result.version.version_id: result for result in representatives
            }
            repairs = [
                dataclasses.replace(
                    repair,
                    initial=updated_by_version.get(
                        repair.initial.version.version_id, repair.initial
                    ),
                    repaired=(
                        None
                        if repair.repaired is None
                        else updated_by_version.get(
                            repair.repaired.version.version_id, repair.repaired
                        )
                    ),
                    representative=updated_by_version[
                        repair.representative.version.version_id
                    ],
                )
                for repair in repairs
            ]
            if not repairs:
                results = list(representatives)
            finalists = tuple(
                result
                for result in representatives
                if result.version.version_id in finalist_ids
            )

        completed = [
            result
            for result in representatives
            if result.evaluation.status == "complete"
        ]

        selected = (
            max(completed, key=selection_key) if completed else representatives[0]
        )
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
            repairs=tuple(repairs),
            representatives=tuple(representatives),
        )

    def run_proposal_cycle(
        self,
        *,
        parent_version_id: str | None = None,
        defer_experience: bool = False,
        parent_evaluation: CandidateEvaluation | None = None,
        planner_recovery: ProviderInvocation | None = None,
        candidate_recoveries: Mapping[int, CandidateResult] | None = None,
        repair_recoveries: Mapping[int, CandidateResult] | None = None,
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

        control_root = self.workspace / ".agentbench"
        control_root.mkdir(parents=True, exist_ok=True)
        planner_output = control_root / "branch_briefs.json"
        if planner_recovery is None:
            self.events.write(
                "proposal_cycle_started",
                iteration_id=iteration_id,
                parent_version_id=parent_id,
                candidate_count=4,
            )
            if planner_output.exists():
                planner_output.unlink()
            planner_act_id = f"act-{self._provider_attempts + 1:06d}-planner"
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
            if getattr(self.provider, "supports_structured_output", False):
                planner_raw.parent.mkdir(parents=True, exist_ok=True)
                planner_schema = planner_raw.with_suffix(".schema.json")
                planner_final = planner_raw.with_suffix(".final.json")
                planner_schema.write_text(
                    json.dumps(
                        branch_briefs_json_schema(
                            expected_count=4,
                            required_entry_symbol=self.policy_entry_symbol,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if planner_final.exists():
                    planner_final.unlink()
                structured_prompt = planner_prompt + f"""

结构化输出覆盖指令：不要写 workspace 文件，也不要修改任何文件。只读输入后，直接把最终答案返回为 JSON；CLI 将按 JSON Schema 保存到 `{planner_final}`。顶层必须是仅含 `branches` 的对象，`branches` 恰好四项。不要在最终 JSON 前后添加 Markdown 或解释。规划阶段使用 high 推理强度；把深入代码核查留给四个 xhigh 候选 act。
"""
                planner = self.provider.invoke(
                    prompt=structured_prompt,
                    workspace=self.workspace,
                    raw_output_path=planner_raw,
                    session_id=None,
                    output_schema_path=planner_schema,
                    output_last_message_path=planner_final,
                    reasoning_effort="high",
                    sandbox_mode="read-only",
                )
                planner_prompt = structured_prompt
                if planner.status == "completed" and planner_final.is_file():
                    load_branch_briefs(
                        planner_final,
                        expected_count=4,
                        known_code_symbols={
                            item["name"]
                            for item in build_candidate_code_index(
                                self.workspace / self.candidate_source_relative
                            )
                        },
                        required_entry_symbol=self.policy_entry_symbol,
                    )
                    shutil.copy2(planner_final, planner_output)
            else:
                planner = self.provider.invoke(
                    prompt=planner_prompt,
                    workspace=self.workspace,
                    raw_output_path=planner_raw,
                    session_id=None,
                )
            self._provider_attempts += 1
            self._coding_agent_acts += int(
                _counts_as_coding_act(
                    status=planner.status,
                    total_tokens=planner.usage.total_tokens,
                    tool_call_count=planner.tool_call_count,
                )
            )
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
        else:
            planner = planner_recovery
            planner_act_id = str(planner.metadata.get("act_id") or "")
            recovered_iteration = str(
                planner.metadata.get("iteration_id") or ""
            )
            if planner.status != "completed":
                raise ValueError("recovered planner must be completed")
            if not planner_act_id:
                raise ValueError("recovered planner requires metadata.act_id")
            if recovered_iteration != iteration_id:
                raise ValueError(
                    "recovered planner iteration does not match pending cycle"
                )
        if planner.status != "completed" or not planner_output.is_file():
            raise RuntimeError("planner did not produce branch_briefs.json")
        proposal_root = self.run_root / "proposals" / iteration_id
        proposal_root.mkdir(parents=True, exist_ok=True)
        persisted_briefs = proposal_root / "branch_briefs.json"
        shutil.copy2(planner_output, persisted_briefs)
        briefs = load_branch_briefs(
            persisted_briefs,
            expected_count=4,
            known_code_symbols={
                item["name"]
                for item in build_candidate_code_index(
                    self.workspace / self.candidate_source_relative
                )
            },
            required_entry_symbol=self.policy_entry_symbol,
        )
        self.events.write(
            "planner_completed",
            act_id=planner_act_id,
            iteration_id=iteration_id,
            status=(
                "recovered"
                if planner_recovery is not None
                else planner.status
            ),
            branch_briefs=str(persisted_briefs),
        )

        iteration = self.run_act(
            parent_version_id=parent_id,
            defer_experience=True,
            branch_briefs=briefs,
            promote_champion=False,
            parent_evaluation=parent_evaluation,
            candidate_recoveries=candidate_recoveries,
            repair_recoveries=repair_recoveries,
        )
        finalists = iteration.finalists
        if iteration.selected not in finalists:
            finalists = (iteration.selected, *finalists)[: self.iteration.finalist_count]
        self.events.write(
            "finalists_selected",
            iteration_id=iteration_id,
            version_ids=[candidate.version.version_id for candidate in finalists],
        )

        positive_margin_deltas = _positive_margin_deltas(
            parent_evaluation,
            iteration.representatives,
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
                    "parent_evaluation": (
                        None
                        if parent_evaluation is None
                        else {
                            "version_id": parent_id,
                            "status": parent_evaluation.status,
                            "score": parent_evaluation.score,
                            "matches": list(parent_evaluation.matches),
                        }
                    ),
                    "positive_margin_deltas": positive_margin_deltas,
                    "initial_candidates": [
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
                    "repairs": [
                        {
                            "branch_index": repair.branch_index,
                            "repair_input_path": str(repair.repair_input_path),
                            "initial": {
                                "act_id": repair.initial.act_id,
                                "version_id": repair.initial.version.version_id,
                                "status": repair.initial.evaluation.status,
                                "score": repair.initial.evaluation.score,
                                "matches": list(repair.initial.evaluation.matches),
                            },
                            "repaired": (
                                None
                                if repair.repaired is None
                                else {
                                    "act_id": repair.repaired.act_id,
                                    "version_id": repair.repaired.version.version_id,
                                    "status": repair.repaired.evaluation.status,
                                    "score": repair.repaired.evaluation.score,
                                    "matches": list(
                                        repair.repaired.evaluation.matches
                                    ),
                                }
                            ),
                            "representative_version_id": (
                                repair.representative.version.version_id
                            ),
                        }
                        for repair in iteration.repairs
                    ],
                    "representatives": [
                        {
                            "branch_index": candidate.branch_index,
                            "act_id": candidate.act_id,
                            "version_id": candidate.version.version_id,
                            "status": candidate.evaluation.status,
                            "score": candidate.evaluation.score,
                            "matches": list(candidate.evaluation.matches),
                        }
                        for candidate in iteration.representatives
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
        reducer_act_id = f"act-{self._provider_attempts + 1:06d}-reducer"
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
            **(
                {"reasoning_effort": "high"}
                if getattr(
                    self.provider, "supports_structured_output", False
                )
                else {}
            ),
        )
        self._provider_attempts += 1
        self._coding_agent_acts += int(
            _counts_as_coding_act(
                status=reducer.status,
                total_tokens=reducer.usage.total_tokens,
                tool_call_count=reducer.tool_call_count,
            )
        )
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
        if reducer.status == "completed" and reducer_output.is_file():
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
                next_state = prepend_framework_comparisons(
                    next_state,
                    positive_margin_deltas,
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
        if not defer_experience:
            self.consolidate_experience_cycle(
                iteration_id=iteration_id,
                parent_version_id=parent_id,
                parent_evaluation=parent_evaluation,
                representatives=iteration.representatives,
                branch_briefs=briefs,
                selected_version_id=iteration.search_parent_version_id,
            )
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
            branch_briefs=briefs,
            repairs=iteration.repairs,
            representatives=iteration.representatives,
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

    def consolidate_experience_cycle(
        self,
        *,
        iteration_id: str,
        parent_version_id: str,
        parent_evaluation: CandidateEvaluation | None,
        representatives: tuple[CandidateResult, ...],
        branch_briefs: tuple[BranchBrief, ...],
        selected_version_id: str,
    ) -> Path | None:
        """Persist all measured branch outcomes, then regenerate the Skill."""

        if self.experience_manager is None:
            return None
        parent_matches = (
            () if parent_evaluation is None else parent_evaluation.matches
        )
        records = tuple(
            derive_experience_record(
                iteration_id=iteration_id,
                act_id=candidate.act_id,
                branch_index=candidate.branch_index,
                parent_version_id=parent_version_id,
                candidate_version_id=candidate.version.version_id,
                selected=(candidate.version.version_id == selected_version_id),
                brief=branch_briefs[candidate.branch_index].to_dict(),
                parent_matches=parent_matches,
                candidate_matches=candidate.evaluation.matches,
                activation=candidate.activation,
            )
            for candidate in representatives
        )
        selected_notes = []
        for candidate in representatives:
            if (
                candidate.version.version_id != selected_version_id
                or candidate.pending_experience_path is None
            ):
                continue
            selected_notes.append(
                self.experience_manager.read_file(
                    candidate.pending_experience_path
                )
            )
        path = self.experience_manager.consolidate_cycle(
            iteration_id,
            records=records,
            notes=tuple(selected_notes),
        )
        self.events.write(
            "experience_cycle_consolidated",
            iteration_id=iteration_id,
            selected_version_id=selected_version_id,
            record_count=len(records),
            experience_path=str(path),
            ledger_path=str(self.experience_manager.ledger.path),
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
            "provider_attempts": self._provider_attempts,
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
            tool_call_count=invocation.tool_call_count,
            elapsed_time_s=invocation.elapsed_time_s,
            raw_output_ref=invocation.raw_output_ref,
            thread_id=invocation.metadata.get("thread_id"),
            **self._provider_outcome_fields(invocation),
        )

    @staticmethod
    def _provider_outcome_fields(
        invocation: ProviderInvocation,
    ) -> dict[str, Any]:
        return {
            key: invocation.metadata.get(key)
            for key in (
                "termination_reason",
                "timeout_kind",
                "weighted_tokens",
                "rollout_budget_limit_tokens",
                "rollout_budget_exhausted",
                "accepted_after_budget_exhaustion",
            )
        }

    def _refresh_checkpoint_outcome(
        self,
        *,
        act_id: str,
        invocation: ProviderInvocation,
    ) -> None:
        path = self.run_root / "checkpoints" / f"{act_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["provider_status"] = invocation.status
        record.update(self._provider_outcome_fields(invocation))
        if "candidate_smoke" in invocation.metadata:
            record["candidate_smoke"] = invocation.metadata["candidate_smoke"]
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
            **self._provider_outcome_fields(invocation),
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
            error=evaluation.error,
            wins=sum(match["result"] == "win" for match in match_records),
            draws=sum(match["result"] == "draw" for match in match_records),
            losses=sum(match["result"] == "loss" for match in match_records),
            matches=list(evaluation.matches),
        )
