"""Audited, non-gated v5-to-v6 Generals HL orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import WorkspaceManifest

from .action_profile import (
    action_profile_payload,
    summarize_action_profile,
)
from .assets import (
    load_pilot_config,
    load_round6_learning_config,
    require_valid_assets,
    resolve_assets,
    resolve_replay_skill,
)
from .dense import build_dense_trace, summarize_dense_trace
from .evaluator import (
    GeneralsEvaluation,
    GeneralsEvaluator,
    build_evaluation_spec,
    build_round6_learning_cases,
    build_round6_validation_cases,
)
from .lineage_v6 import import_round6_source, load_round6_parent
from .measurement import (
    ProbeState,
    measure_action_disagreement,
    summarize_decision_classes,
)
from .models import (
    AssetLayout,
    PilotConfig,
    ReplaySkillAsset,
    Round6LearningConfig,
)
from .pipeline import GeneralsHLPipeline
from .prompt import PromptBuildResult
from .prompt_v6 import (
    build_round6_prompt,
    validate_round6_static_context,
)
from .replay import (
    CriticalLearningEvidence,
    CriticalWindowSelection,
    build_critical_learning_evidence,
    build_learning_replay,
)


V6_SELECTION_REASONS = (
    "first_decision",
    "first_non_end_action",
    "first_main_pressure",
    "before_steepest_territory_drop",
    "before_steepest_army_drop",
    "first_strategic_opportunity",
    "final_decision",
)

V6_EDITABLE_FILES = frozenset(
    {
        "strategy.py",
        "state_view.py",
        "STRATEGY.md",
        "EXPERIENCE.md",
    }
)
V6_REQUIRED_FILES = frozenset(
    {
        "main.py",
        "strategy.py",
        "state_view.py",
        "STRATEGY.md",
        "EXPERIENCE.md",
    }
)


def _combine_round6_learning_budgets(
    parent: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, float | int | None]:
    """Add known lineage values without treating an absent value as zero."""
    keys = {
        str(key)
        for budget in (parent, current)
        for key in budget
        if str(key).startswith("learning_")
    }
    combined: dict[str, float | int | None] = {}
    for key in sorted(keys):
        if key not in parent or key not in current:
            combined[key] = None
            continue
        left = parent[key]
        right = current[key]
        if left is None or right is None:
            combined[key] = None
            continue
        value = float(left) + float(right)
        combined[key] = (
            value if key.endswith("_time_s") else int(value)
        )
    return combined


@dataclass(frozen=True)
class Round6PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score_1: float | None
    evo_score_2: float | None
    evo_score_3: float | None
    evo_score_4: float | None
    evo_score_5: float | None
    evo_score_6: float | None
    gain_6: float | None
    global_act_count: int
    round_act_count: int
    status: str
    runnable: bool


class _Round6Abort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class _Round6RecoverySource:
    run_dir: Path
    run_id: str
    mode: str
    summary: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]
    local_learning_budget: Mapping[str, object]
    cumulative_learning_budget: Mapping[str, object]
    prompt_text: str | None
    prompt_manifest: Mapping[str, object] | None
    v6_manifest: WorkspaceManifest | None


class GeneralsHLRound6Pipeline(GeneralsHLPipeline):
    """Run one high-only v6 act and formally evaluate every runnable result."""

    def __init__(
        self,
        config: PilotConfig,
        learning_config: Round6LearningConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        prompt_max_bytes: int = 131_072,
    ):
        super().__init__(
            config,
            assets,
            data_dir,
            provider,
            evaluator=evaluator,
            rules_path=rules_path,
        )
        self.learning_config = learning_config
        self.replay_skill = replay_skill
        self.parent_run_dir = Path(parent_run_dir)
        self.expected_parent_hash = expected_parent_hash
        self.prompt_max_bytes = int(prompt_max_bytes)

    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        learning_manifest_path: Path,
        replay_skill_path: Path,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
    ) -> "GeneralsHLRound6Pipeline":
        config = load_pilot_config(manifest_path)
        learning = load_round6_learning_config(
            learning_manifest_path,
            config,
        )
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        replay_skill = resolve_replay_skill(
            agentbench_root,
            replay_skill_path,
        )
        return cls(
            config=config,
            learning_config=learning,
            assets=assets,
            replay_skill=replay_skill,
            parent_run_dir=parent_run_dir,
            expected_parent_hash=expected_parent_hash,
            data_dir=data_dir,
            provider=provider,
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _result_rows(
        evaluations: Iterable[GeneralsEvaluation | None],
    ) -> list[dict[str, object]]:
        return [
            {
                "case_id": result.case_id,
                "outcome": result.outcome,
                "valid": result.valid,
                "error": result.error,
                **result.metadata,
            }
            for evaluation in evaluations
            if evaluation is not None
            for result in evaluation.results
        ]

    def _write_benchmark_artifacts(
        self,
        run_dir: Path,
        learning_cases: tuple,
        validation_cases: tuple,
        formal_cases: tuple,
    ) -> None:
        benchmark_dir = run_dir / "benchmark"
        skill_target = benchmark_dir / "replay-skill.md"
        skill_target.parent.mkdir(parents=True, exist_ok=True)
        skill_target.write_text(
            self.replay_skill.text,
            encoding="utf-8",
        )
        self._write_json(
            benchmark_dir / "v6-learning-spec.json",
            {
                "learning_id": self.learning_config.learning_id,
                "opponent_id": self.learning_config.opponent_id,
                "seeds": list(self.learning_config.seeds),
                "seats": list(self.learning_config.seats),
                "cases": [asdict(case) for case in learning_cases],
            },
        )
        self._write_json(
            benchmark_dir / "v6-validation-spec.json",
            {
                "benchmark_id": self.config.benchmark_id,
                "engine_hash": self.assets.engine_hash,
                "cases": [asdict(case) for case in validation_cases],
            },
        )
        self._write_json(
            benchmark_dir / "formal-spec.json",
            {
                "benchmark_id": self.config.benchmark_id,
                "engine_hash": self.assets.engine_hash,
                "evaluation_cases": [asdict(case) for case in formal_cases],
            },
        )
        self._write_json(
            benchmark_dir / "replay-skill.json",
            {
                "source_path": str(self.replay_skill.path),
                "sha256": self.replay_skill.sha256,
                "bytes": len(self.replay_skill.text.encode("utf-8")),
                "artifact_ref": str(skill_target),
            },
        )

    @staticmethod
    def _complete_learning(
        evaluation: GeneralsEvaluation,
        case_count: int,
    ) -> bool:
        return (
            evaluation.status == "complete"
            and len(evaluation.results) == case_count
            and sum(result.valid for result in evaluation.results)
            == case_count
        )

    @staticmethod
    def _critical_feedback(
        evaluation: GeneralsEvaluation,
        tier_by_case: Mapping[str, str],
    ) -> tuple[
        tuple[CriticalLearningEvidence, ...],
        tuple[CriticalWindowSelection, ...],
        tuple[ProbeState, ...],
    ]:
        evidence: list[CriticalLearningEvidence] = []
        selections: list[CriticalWindowSelection] = []
        probes: list[ProbeState] = []
        for match in evaluation.matches:
            replay_id = (
                f"learn6-high-s{match.seed}-p{match.evaluated_seat}"
            )
            raw = build_learning_replay(
                match,
                "v5",
                tier_by_case[match.case_id],
            )
            replay = replace(
                raw,
                replay_id=replay_id,
                decisions=tuple(
                    replace(
                        decision,
                        state_id=f"v5:{replay_id}-d{index}",
                    )
                    for index, decision in enumerate(
                        raw.decisions,
                        start=1,
                    )
                ),
            )
            trace = build_dense_trace(match)
            dense = summarize_dense_trace(match, trace)
            compact, selection = build_critical_learning_evidence(
                replay,
                dense,
                trace,
                max_decisions=6,
                selection_reasons=V6_SELECTION_REASONS,
                include_strategic_targets=True,
            )
            selected_state_ids = set(selection.selected_state_ids)
            evidence.append(compact)
            selections.append(selection)
            probes.extend(
                ProbeState(
                    decision.state_id,
                    {**decision.state, "my_seat": decision.seat},
                    decision.action,
                )
                for decision in replay.decisions
                if decision.state_id in selected_state_ids
            )
        return tuple(evidence), tuple(selections), tuple(probes)

    def _persist_action_profile(
        self,
        run: Run,
        evaluation: GeneralsEvaluation,
        *,
        phase: str,
        artifact_path: Path,
        required: bool,
    ) -> dict[str, object] | None:
        try:
            payload = action_profile_payload(
                summarize_action_profile(evaluation)
            )
            self._write_json(artifact_path, payload)
            run.write(
                "behavior_diagnostics",
                diagnostic="action_profile",
                phase=phase,
                version=evaluation.version,
                artifact_ref=str(artifact_path),
                **payload,
            )
            return payload
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write(
                "behavior_measurement_error",
                diagnostic="action_profile",
                phase=phase,
                version=evaluation.version,
                error=error,
            )
            if required:
                raise _Round6Abort(
                    "learning_profile_failed",
                    error,
                ) from exc
            return None

    @staticmethod
    def _scope_valid(changed_files: Iterable[str]) -> bool:
        return all(
            path in V6_EDITABLE_FILES
            or path.startswith("tests/")
            or path.startswith("policy/")
            for path in changed_files
        )

    def _frozen_manifest_matches(
        self,
        source: Path,
        manifest: WorkspaceManifest,
    ) -> bool:
        """Prove the saved source tree exactly matches its manifest."""
        try:
            actual = self.snapshotter.capture(source)
        except (OSError, ValueError):
            return False
        return (
            actual.content_hash == manifest.content_hash
            and actual.files == manifest.files
        )

    @staticmethod
    def _required_frozen_files_valid(
        source: Path,
        manifest: WorkspaceManifest,
    ) -> bool:
        return all(
            relative in manifest.files
            and (source / relative).is_file()
            and not (source / relative).is_symlink()
            for relative in V6_REQUIRED_FILES
        )

    def _materialize_verified_source(
        self,
        frozen_source: Path,
        destination: Path,
        manifest: WorkspaceManifest,
    ) -> Path:
        """Create a one-use copy and prove it is the frozen candidate."""
        if destination.exists():
            raise ValueError(
                f"isolated source destination already exists: {destination}"
            )
        shutil.copytree(frozen_source, destination)
        actual = self.snapshotter.capture(destination)
        if (
            actual.content_hash != manifest.content_hash
            or actual.files != manifest.files
        ):
            raise ValueError(
                "isolated v6 source does not match frozen manifest"
            )
        return destination

    def _verify_runtime_source(
        self,
        run: Run,
        source: Path,
        manifest: WorkspaceManifest,
        *,
        phase: str,
    ) -> None:
        """Persist the immediate pre-use content-addressed verification."""
        try:
            actual = self.snapshotter.capture(source)
            verified = (
                actual.content_hash == manifest.content_hash
                and actual.files == manifest.files
            )
            actual_hash: str | None = actual.content_hash
            files_match: bool | None = actual.files == manifest.files
            verification_error = None
        except Exception as exc:
            verified = False
            actual_hash = None
            files_match = None
            verification_error = f"{type(exc).__name__}: {exc}"
        run.write(
            "behavior_diagnostics",
            diagnostic="frozen_source_verification",
            phase=phase,
            version="v6",
            source=str(source),
            expected_manifest_hash=manifest.content_hash,
            actual_manifest_hash=actual_hash,
            files_match=files_match,
            verified=verified,
            error=verification_error,
        )
        if not verified:
            raise ValueError(
                f"{phase} source does not match frozen v6 manifest"
            )

    def _run_candidate_tests(
        self,
        workspace: Path,
    ) -> tuple[bool, str]:
        """Convert candidate-test timeouts into retained invalid results."""
        try:
            return self._baseline_tests(workspace)
        except subprocess.TimeoutExpired as exc:
            def output_text(value: object) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return str(value)

            return (
                False,
                (
                    "candidate tests timeout after "
                    f"{exc.timeout} seconds\n"
                    f"{output_text(exc.output)}"
                    f"{output_text(exc.stderr)}"
                )[-100_000:],
            )

    @staticmethod
    def _formal_success(
        formal_scores: Mapping[str, float | None],
    ) -> dict[str, bool]:
        high = formal_scores.get("high")
        medium = formal_scores.get("medium")
        low = formal_scores.get("low")
        return {
            "medium_at_least_2_of_6": (
                medium is not None and medium >= 2 / 6
            ),
            "low_retained_6_of_6": low is not None and low == 1.0,
            "high_breakthrough_at_least_1_of_6": (
                high is not None and high >= 1 / 6
            ),
        }

    @staticmethod
    def _read_json_object(path: Path, label: str) -> dict[str, object]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"recovery {label} is invalid: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"recovery {label} must be an object")
        return value

    @classmethod
    def _read_workspace_manifest(
        cls,
        path: Path,
        label: str,
    ) -> WorkspaceManifest:
        value = cls._read_json_object(path, label)
        try:
            files = {
                str(key): str(item)
                for key, item in value["files"].items()
            }
            changed_files = [
                str(item)
                for item in value.get("changed_files", [])
            ]
            content_hash = str(value["content_hash"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"recovery {label} is malformed: {exc}"
            ) from exc
        return WorkspaceManifest(
            content_hash=content_hash,
            files=files,
            changed_files=changed_files,
        )

    @staticmethod
    def _learning_budget(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): item
            for key, item in value.items()
            if str(key).startswith("learning_")
        }

    def _validate_recovery_source(
        self,
        failed_run_dir: Path,
        lineage: Any,
        seen: frozenset[Path] = frozenset(),
    ) -> _Round6RecoverySource:
        source_dir = Path(failed_run_dir).resolve()
        if source_dir in seen:
            raise ValueError("recovery lineage contains a cycle")
        seen = seen | {source_dir}
        summary = self._read_json_object(
            source_dir / "summary.json",
            "terminal summary",
        )
        terminal_status = summary.get("status")
        if terminal_status == "prompt_incomplete":
            mode = "prompt_rebuild"
        elif terminal_status == "provider_failed":
            mode = "provider_retry"
        elif terminal_status in {"formal_failed", "incomplete"}:
            mode = "post_act_evaluation"
        else:
            raise ValueError(
                "recovery terminal status is not recoverable"
            )

        config = summary.get("config")
        if not isinstance(config, dict):
            raise ValueError("recovery lineage config is missing")
        if (
            summary.get("benchmark_id") != self.config.benchmark_id
            or config.get("benchmark_id") != self.config.benchmark_id
            or config.get("engine_hash") != self.assets.engine_hash
            or config.get("round") != 6
            or config.get("lineage_mode")
            != "legacy_pilot_v5_to_v6"
        ):
            raise ValueError("recovery lineage benchmark does not match")
        if (
            summary.get("learning_id")
            != self.learning_config.learning_id
            or config.get("learning_id")
            != self.learning_config.learning_id
        ):
            raise ValueError("recovery learning id does not match")
        expected_scores = {
            "raw_score": lineage.raw_score,
            "evo_score_1": lineage.evo_score_1,
            "evo_score_2": lineage.evo_score_2,
            "evo_score_3": lineage.evo_score_3,
            "evo_score_4": lineage.evo_score_4,
            "evo_score_5": lineage.evo_score_5,
        }
        if (
            summary.get("parent_run_id") != lineage.parent_run_id
            or summary.get("parent_version") != "v5"
            or summary.get("starting_version") != "v5"
            or summary.get("parent_content_hash")
            != lineage.v5_manifest.content_hash
            or any(
                summary.get(key) != value
                for key, value in expected_scores.items()
            )
        ):
            raise ValueError("recovery parent lineage does not match")
        score_history = summary.get("score_history")
        if (
            not isinstance(score_history, list)
            or tuple(score_history[:7])
            != tuple(lineage.prior_score_history)
            or len(score_history) != 8
        ):
            raise ValueError("recovery parent score history does not match")

        try:
            source_events = tuple(
                json.loads(line)
                for line in (
                    source_dir / "events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"recovery lineage events are invalid: {exc}"
            ) from exc
        run_id = summary.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or not source_events
            or any(
                not isinstance(event, dict)
                or event.get("run_id") != run_id
                for event in source_events
            )
        ):
            raise ValueError("recovery lineage event IDs do not match")

        failed_run_id = summary.get("failed_run_id")
        declared_recovery_mode = summary.get("recovery_mode")
        config_failed_run_id = config.get("failed_run_id")
        config_recovery_mode = config.get("recovery_mode")
        previous_source: _Round6RecoverySource | None = None
        recovery_events = [
            event
            for event in source_events
            if event.get("event_type") == "recovery_import"
        ]
        if failed_run_id is None:
            if (
                declared_recovery_mode is not None
                or config_failed_run_id is not None
                or config_recovery_mode is not None
                or recovery_events
            ):
                raise ValueError(
                    "recovery lineage root metadata does not match"
                )
        else:
            if (
                not isinstance(failed_run_id, str)
                or not failed_run_id
                or declared_recovery_mode
                not in {
                    "prompt_rebuild",
                    "provider_retry",
                    "post_act_evaluation",
                }
                or config_failed_run_id != failed_run_id
                or config_recovery_mode != declared_recovery_mode
                or len(recovery_events) != 1
                or recovery_events[0].get("failed_run_id")
                != failed_run_id
                or recovery_events[0].get("recovery_mode")
                != declared_recovery_mode
            ):
                raise ValueError(
                    "recovery lineage link metadata does not match"
                )
            previous_dir = source_dir.parent / failed_run_id
            previous_source = self._validate_recovery_source(
                previous_dir,
                lineage,
                seen,
            )
            if previous_source.run_id != failed_run_id:
                raise ValueError(
                    "recovery lineage failed run ID does not match"
                )
            if declared_recovery_mode != previous_source.mode:
                raise ValueError(
                    "recovery mode does not match failed run kind"
                )

        learning_cases = tuple(
            build_round6_learning_cases(
                self.config,
                self.learning_config,
            )
        )
        validation_cases = tuple(
            build_round6_validation_cases(self.config)
        )
        formal_cases = tuple(build_evaluation_spec(self.config).cases)
        benchmark_specs = [
            event
            for event in source_events
            if event.get("event_type") == "benchmark_spec"
        ]
        if len(benchmark_specs) != 1:
            raise ValueError(
                "recovery learning benchmark spec is ambiguous"
            )
        benchmark_spec = benchmark_specs[0]
        if (
            benchmark_spec.get("benchmark_id")
            != self.config.benchmark_id
            or benchmark_spec.get("learning_id")
            != self.learning_config.learning_id
            or benchmark_spec.get("engine_hash")
            != self.assets.engine_hash
            or benchmark_spec.get("replay_skill_sha256")
            != self.replay_skill.sha256
            or benchmark_spec.get("learning_cases")
            != [asdict(case) for case in learning_cases]
            or benchmark_spec.get("validation_cases")
            != [asdict(case) for case in validation_cases]
            or benchmark_spec.get("evaluation_cases")
            != [asdict(case) for case in formal_cases]
        ):
            raise ValueError(
                "recovery learning benchmark spec does not match"
            )
        learning_spec = self._read_json_object(
            source_dir / "benchmark" / "v6-learning-spec.json",
            "learning spec",
        )
        if (
            learning_spec.get("learning_id")
            != self.learning_config.learning_id
            or learning_spec.get("opponent_id")
            != self.learning_config.opponent_id
            or learning_spec.get("seeds")
            != list(self.learning_config.seeds)
            or learning_spec.get("seats")
            != list(self.learning_config.seats)
            or learning_spec.get("cases")
            != [asdict(case) for case in learning_cases]
        ):
            raise ValueError("recovery learning spec does not match")
        learning_results = summary.get("learning_results")
        expected_learning = {
            case.case_id: case for case in learning_cases
        }
        if (
            not isinstance(learning_results, list)
            or len(learning_results) != len(learning_cases)
            or any(
                not isinstance(result, dict)
                or not isinstance(result.get("case_id"), str)
                for result in learning_results
            )
            or {
                result.get("case_id")
                for result in learning_results
                if isinstance(result, dict)
            }
            != set(expected_learning)
            or any(
                not isinstance(result, dict)
                or result.get("valid") is not True
                or result.get("error") is not None
                or result.get("version") != "v5"
                or result.get("phase") != "learning"
                or result.get("tier")
                != expected_learning[result["case_id"]].metadata["tier"]
                or result.get("seed")
                != expected_learning[result["case_id"]].seed
                or result.get("evaluated_seat")
                != expected_learning[result["case_id"]].first_player
                for result in learning_results
            )
        ):
            raise ValueError(
                "recovery learning results do not match declared suite"
            )
        learning_profile = self._read_json_object(
            source_dir
            / "diagnostics"
            / "v5-learning-action-profile.json",
            "learning action profile",
        )
        declared_profiles = summary.get("action_profiles")
        if (
            not isinstance(declared_profiles, dict)
            or declared_profiles.get("learning") != learning_profile
        ):
            raise ValueError(
                "recovery learning action profile does not match"
            )
        imported_v5 = self._read_workspace_manifest(
            source_dir / "versions" / "v5" / "manifest.json",
            "parent v5 manifest",
        )
        source_v5 = source_dir / "versions" / "v5" / "source"
        if (
            imported_v5.content_hash
            != lineage.v5_manifest.content_hash
            or imported_v5.files != lineage.v5_manifest.files
            or not self._frozen_manifest_matches(
                source_v5,
                imported_v5,
            )
        ):
            raise ValueError("recovery parent v5 source does not match")

        local_budget = self._learning_budget(
            summary.get("local_learning_budget")
        )
        declared_budget = self._learning_budget(summary.get("budget"))
        if not local_budget or local_budget != declared_budget:
            raise ValueError(
                "recovery learning budget does not match terminal record"
            )
        cumulative_budget = self._learning_budget(
            summary.get("cumulative_learning_budget")
        )
        base_learning_budget = (
            previous_source.cumulative_learning_budget
            if previous_source is not None
            else lineage.learning_budget
        )
        expected_cumulative = _combine_round6_learning_budgets(
            base_learning_budget,
            local_budget,
        )
        if cumulative_budget != expected_cumulative:
            raise ValueError(
                "recovery learning cumulative budget does not match"
            )

        acts = [
            event
            for event in source_events
            if event.get("event_type") == "coding_agent_act"
        ]
        expected_round_acts = (
            0
            if mode == "prompt_rebuild"
            or (
                mode == "post_act_evaluation"
                and declared_recovery_mode == "post_act_evaluation"
            )
            else 1
        )
        base_global_acts = (
            int(previous_source.summary["act_count"])
            if previous_source is not None
            else lineage.global_act_count
        )
        expected_global_acts = base_global_acts + expected_round_acts
        if (
            summary.get("round_act_count") != expected_round_acts
            or summary.get("act_count") != expected_global_acts
            or local_budget.get("learning_coding_agent_acts")
            != expected_round_acts
            or len(acts) != expected_round_acts
        ):
            raise ValueError(
                "recovery lineage act counters do not match"
            )
        if (
            recovery_events
            and recovery_events[0].get("new_provider_act")
            != (expected_round_acts == 1)
        ):
            raise ValueError(
                "recovery lineage provider act receipt does not match"
            )
        if mode == "prompt_rebuild":
            if (
                summary.get("runnable") is not False
                or summary.get("evaluation_status") != "not_run"
                or summary.get("formal_attempted") is not False
                or summary.get("evo_score_6") is not None
            ):
                raise ValueError(
                    "recovery terminal prompt status does not match"
                )
        elif mode == "provider_retry":
            if (
                acts[0].get("status") != "failed"
                or summary.get("runnable") is not False
                or summary.get("evaluation_status") != "not_run"
                or summary.get("formal_attempted") is not False
                or summary.get("evo_score_6") is not None
            ):
                raise ValueError(
                    "recovery terminal provider status does not match"
                )
        else:
            post_act_contract_valid = (
                (
                    expected_round_acts == 0
                    or acts[0].get("status") == "completed"
                )
                and summary.get("runnable") is True
                and summary.get("formal_attempted") is True
            )
            if terminal_status == "formal_failed":
                post_act_contract_valid = (
                    post_act_contract_valid
                    and summary.get("evaluation_status") == "error"
                    and summary.get("evo_score_6") is None
                    and summary.get("formal_score") is None
                    and score_history[-1] is None
                    and summary.get("formal_results") == []
                    and isinstance(summary.get("formal_error"), str)
                    and bool(summary.get("formal_error"))
                )
            else:
                formal_results = summary.get("formal_results")
                post_act_contract_valid = (
                    post_act_contract_valid
                    and summary.get("evaluation_status") == "incomplete"
                    and summary.get("formal_score")
                    == summary.get("evo_score_6")
                    and score_history[-1]
                    == summary.get("evo_score_6")
                    and isinstance(formal_results, list)
                    and len(formal_results) == len(formal_cases)
                    and all(
                        isinstance(result, dict)
                        and result.get("version") == "v6"
                        and result.get("phase") == "evaluation"
                        for result in formal_results
                    )
                    and summary.get("formal_error") is None
                )
            if not post_act_contract_valid:
                raise ValueError(
                    "recovery terminal post act status does not match"
                )

        prompt_text: str | None = None
        prompt_manifest: Mapping[str, object] | None = None
        if mode != "prompt_rebuild":
            provider_dir = source_dir / "provider"
            canonical_prompt = (
                provider_dir / "codex-act-v6.prompt.md"
            )
            prompt_copy = provider_dir / "prompt.txt"
            try:
                prompt_bytes = canonical_prompt.read_bytes()
                if prompt_copy.read_bytes() != prompt_bytes:
                    raise ValueError(
                        "recovery prompt copies do not match"
                    )
                prompt_text = prompt_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"recovery prompt is invalid: {exc}"
                ) from exc
            prompt_manifest = self._read_json_object(
                provider_dir / "prompt-manifest.json",
                "prompt manifest",
            )
            prompt_json = self._read_json_object(
                provider_dir / "codex-act-v6.prompt.json",
                "prompt JSON",
            )
            actual_prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
            expected_episode_ids = [
                f"learn6-high-s{seed}-p{seat}"
                for seed in self.learning_config.seeds
                for seat in self.learning_config.seats
            ]
            if (
                prompt_json != prompt_manifest
                or prompt_manifest.get("prompt_sha256")
                != actual_prompt_hash
                or prompt_manifest.get("prompt_bytes")
                != len(prompt_bytes)
                or prompt_manifest.get("replay_skill_sha256")
                != self.replay_skill.sha256
                or prompt_manifest.get("included_episode_ids")
                != expected_episode_ids
                or prompt_manifest.get("omitted_episode_ids") != []
                or prompt_manifest.get("truncated") is not False
            ):
                raise ValueError(
                    "recovery prompt provenance does not match"
                )
            receipt = self._read_json_object(
                provider_dir / "feedback-receipt.json",
                "prompt feedback receipt",
            )
            if summary.get("feedback_read") != receipt:
                raise ValueError(
                    "recovery prompt feedback receipt does not match"
                )

        v6_manifest: WorkspaceManifest | None = None
        v6_manifest_path = (
            source_dir / "versions" / "v6" / "manifest.json"
        )
        if v6_manifest_path.is_file():
            v6_manifest = self._read_workspace_manifest(
                v6_manifest_path,
                "v6 manifest",
            )
            source_v6 = source_dir / "versions" / "v6" / "source"
            if not self._frozen_manifest_matches(
                source_v6,
                v6_manifest,
            ):
                raise ValueError("recovery v6 source does not match manifest")
        if mode == "post_act_evaluation":
            if (
                v6_manifest is None
                or not self._scope_valid(v6_manifest.changed_files)
                or not self._required_frozen_files_valid(
                    source_dir / "versions" / "v6" / "source",
                    v6_manifest,
                )
                or v6_manifest.files.get("main.py")
                != lineage.v5_manifest.files.get("main.py")
                or (
                    source_dir
                    / "versions"
                    / "v6"
                    / "source"
                    / "main.py"
                ).read_bytes()
                != lineage.v5_source.joinpath("main.py").read_bytes()
            ):
                raise ValueError(
                    "recovery v6 runnable source does not match"
                )
            versions = [
                event
                for event in source_events
                if event.get("event_type") == "version"
                and event.get("version") == "v6"
            ]
            if (
                len(versions) != 1
                or versions[0].get("manifest_hash")
                != v6_manifest.content_hash
                or versions[0].get("status") != "available"
                or versions[0].get("runnable") is not True
                or versions[0].get("frozen_manifest_valid") is not True
                or versions[0].get("required_source_files_valid")
                is not True
                or versions[0].get("main_unchanged") is not True
                or versions[0].get("tests_passed") is not True
            ):
                raise ValueError(
                    "recovery v6 terminal version does not match"
                )

        return _Round6RecoverySource(
            run_dir=source_dir,
            run_id=run_id,
            mode=mode,
            summary=summary,
            events=source_events,
            local_learning_budget=local_budget,
            cumulative_learning_budget=cumulative_budget,
            prompt_text=prompt_text,
            prompt_manifest=prompt_manifest,
            v6_manifest=v6_manifest,
        )

    def run(self) -> Round6PipelineResult:
        return self._run()

    def _run(
        self,
        prompt_recovery: _Round6RecoverySource | None = None,
    ) -> Round6PipelineResult:
        lineage = load_round6_parent(
            self.parent_run_dir,
            self.expected_parent_hash,
        )
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "learning_id": self.learning_config.learning_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 6,
                "lineage_mode": "legacy_pilot_v5_to_v6",
                **(
                    {
                        "failed_run_id": prompt_recovery.run_id,
                        "recovery_mode": prompt_recovery.mode,
                    }
                    if prompt_recovery is not None
                    else {}
                ),
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = lineage.raw_score
        evo_score_1 = lineage.evo_score_1
        evo_score_2 = lineage.evo_score_2
        evo_score_3 = lineage.evo_score_3
        evo_score_4 = lineage.evo_score_4
        evo_score_5 = lineage.evo_score_5
        evo_score_6 = None
        gain_6 = None
        global_act_count = lineage.global_act_count
        round_act_count = 0
        status = "failed"
        runnable = False
        error: str | None = None
        v5_learning: GeneralsEvaluation | None = None
        validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        validation_error: str | None = None
        formal_error: str | None = None
        formal_attempted = False
        prompt: PromptBuildResult | None = None
        act = None
        action_profiles: dict[str, dict[str, object] | None] = {
            "learning": None,
            "validation": None,
            "formal": None,
        }
        behavior_diagnostics: dict[str, object] | None = None
        try:
            imported_v5 = import_round6_source(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            versions_v5_source = (
                run_dir / "versions" / "v5" / "source"
            )
            v5_strategy = (workspace / "strategy.py").read_text(
                encoding="utf-8"
            )
            v5_experience = (workspace / "EXPERIENCE.md").read_text(
                encoding="utf-8"
            )
            rules_text = self.rules_path.read_text(encoding="utf-8")
            validate_round6_static_context(
                v5_strategy=v5_strategy,
                v5_experience=v5_experience,
                rules_text=rules_text,
                replay_skill_text=self.replay_skill.text,
            )
            learning_cases = tuple(
                build_round6_learning_cases(
                    self.config,
                    self.learning_config,
                )
            )
            validation_cases = tuple(
                build_round6_validation_cases(self.config)
            )
            formal_cases = tuple(
                build_evaluation_spec(self.config).cases
            )
            self._write_benchmark_artifacts(
                run_dir,
                learning_cases,
                validation_cases,
                formal_cases,
            )
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v5",
                starting_version="v5",
                version="v5",
                manifest_hash=imported_v5.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=global_act_count,
            )
            run.write(
                "version",
                version="v5",
                status="lineage_parent",
                manifest_hash=imported_v5.content_hash,
                parent_run_id=lineage.parent_run_id,
            )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                learning_id=self.learning_config.learning_id,
                learning_cases=[asdict(case) for case in learning_cases],
                validation_cases=[
                    asdict(case) for case in validation_cases
                ],
                evaluation_cases=[asdict(case) for case in formal_cases],
                engine_hash=self.assets.engine_hash,
                replay_skill_sha256=self.replay_skill.sha256,
            )
            evaluator = (
                self.evaluator
                or self._production_evaluator(run_dir)
            )
            v5_learning = evaluator.evaluate(
                versions_v5_source,
                "v5",
                "learning",
                run,
                cases=learning_cases,
            )
            valid_learning_count = sum(
                result.valid for result in v5_learning.results
            )
            run.write(
                "learning_validation",
                version="v5",
                phase="learning",
                learning_id=self.learning_config.learning_id,
                status=v5_learning.status,
                case_count=len(learning_cases),
                valid_case_count=valid_learning_count,
            )
            if not self._complete_learning(
                v5_learning,
                len(learning_cases),
            ):
                raise _Round6Abort(
                    "learning_failed",
                    "v5 strongest-human suite is incomplete",
                )

            tiers = {
                case.case_id: str(case.metadata["tier"])
                for case in learning_cases
            }
            evidence, selections, probes = self._critical_feedback(
                v5_learning,
                tiers,
            )
            for selection in selections:
                run.write(
                    "critical_window_selection",
                    version="v5",
                    replay_id=selection.replay_id,
                    total_decision_count=selection.total_decision_count,
                    selected_state_ids=list(
                        selection.selected_state_ids
                    ),
                    reasons={
                        key: list(value)
                        for key, value in selection.reasons.items()
                    },
                    omitted_decision_count=(
                        selection.omitted_decision_count
                    ),
                )
            learning_profile = self._persist_action_profile(
                run,
                v5_learning,
                phase="learning",
                artifact_path=(
                    run_dir
                    / "diagnostics"
                    / "v5-learning-action-profile.json"
                ),
                required=True,
            )
            action_profiles["learning"] = learning_profile
            assert learning_profile is not None
            try:
                prompt = build_round6_prompt(
                    benchmark_id=self.config.benchmark_id,
                    v5_strategy=v5_strategy,
                    v5_experience=v5_experience,
                    rules_text=rules_text,
                    replay_skill_text=self.replay_skill.text,
                    replay_skill_sha256=self.replay_skill.sha256,
                    evidence=evidence,
                    action_profile=learning_profile,
                    max_bytes=self.prompt_max_bytes,
                )
            except ValueError as exc:
                if "prompt context exceeds max_bytes" not in str(exc):
                    raise
                raise _Round6Abort(
                    "prompt_incomplete",
                    str(exc),
                ) from exc
            expected_episode_ids = {
                f"learn6-high-s{seed}-p{seat}"
                for seed in self.learning_config.seeds
                for seat in self.learning_config.seats
            }
            if (
                prompt.truncated
                or set(prompt.included_episode_ids)
                != expected_episode_ids
                or prompt.omitted_episode_ids
            ):
                raise _Round6Abort(
                    "prompt_incomplete",
                    "v6 prompt omitted a declared learning episode",
                )

            feedback_receipt = {
                "episodes": prompt.feedback_episodes_read,
                "decision_records": (
                    prompt.feedback_decision_records_read
                ),
                "serialized_bytes": (
                    prompt.feedback_serialized_bytes_read
                ),
                "prompt_bytes": prompt.prompt_bytes,
                "estimated_tokens": prompt.estimated_tokens,
                "included_episode_ids": list(
                    prompt.included_episode_ids
                ),
                "omitted_episode_ids": list(
                    prompt.omitted_episode_ids
                ),
                "selection_policy": prompt.selection_policy,
            }
            run.write("feedback_read", **feedback_receipt)
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            canonical_prompt = (
                provider_dir / "codex-act-v6.prompt.md"
            )
            canonical_prompt.write_text(
                prompt.prompt,
                encoding="utf-8",
            )
            shutil.copy2(canonical_prompt, provider_dir / "prompt.txt")
            prompt_manifest = {
                key: value
                for key, value in asdict(prompt).items()
                if key != "prompt"
            }
            prompt_manifest["prompt_sha256"] = hashlib.sha256(
                prompt.prompt.encode("utf-8")
            ).hexdigest()
            prompt_manifest["replay_skill_sha256"] = (
                self.replay_skill.sha256
            )
            self._write_json(
                provider_dir / "codex-act-v6.prompt.json",
                prompt_manifest,
            )
            self._write_json(
                provider_dir / "prompt-manifest.json",
                prompt_manifest,
            )
            self._write_json(
                provider_dir / "feedback-receipt.json",
                feedback_receipt,
            )
            raw_path = provider_dir / "codex-act-v6.raw.jsonl"
            stderr_path = provider_dir / "codex-act-v6.stderr.log"
            raw_path.touch()
            stderr_path.touch()
            controller = run.create_coding_agent_controller(
                self.provider,
                snapshotter=self.snapshotter,
                budget_phase="learning",
            )
            act = controller.run_act(
                {
                    "prompt": prompt.prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(raw_path),
                    "stderr_output_path": str(stderr_path),
                    "max_artifact_bytes": (
                        self.config.limits.max_artifact_bytes
                    ),
                },
                workspace_root=str(workspace),
                version_before="v5",
                previous_manifest=imported_v5,
            )
            round_act_count = 1
            global_act_count = lineage.global_act_count + 1
            self._write_json(
                provider_dir / "provider-budget.json",
                run.budget_snapshot(),
            )

            v6 = self._write_version(
                run_dir,
                "v6",
                workspace,
                previous=imported_v5,
            )
            frozen_v6_source = (
                run_dir / "versions" / "v6" / "source"
            )
            self.snapshotter.write_unified_patch(
                versions_v5_source,
                frozen_v6_source,
                run_dir / "versions" / "v5-to-v6.patch",
            )
            scope_valid = self._scope_valid(v6.changed_files)
            frozen_manifest_valid = self._frozen_manifest_matches(
                frozen_v6_source,
                v6,
            )
            required_source_files_valid = (
                self._required_frozen_files_valid(
                    frozen_v6_source,
                    v6,
                )
            )
            main_unchanged = (
                "main.py" in v6.files
                and v6.files.get("main.py")
                == imported_v5.files.get("main.py")
                and "main.py" not in v6.changed_files
                and (frozen_v6_source / "main.py").is_file()
                and not (frozen_v6_source / "main.py").is_symlink()
                and (
                    frozen_v6_source / "main.py"
                ).read_bytes()
                == (versions_v5_source / "main.py").read_bytes()
            )
            provider_completed = act.status == "completed"
            strategy_documents_present = (
                required_source_files_valid
                and all(
                    name in v6.files
                    for name in ("STRATEGY.md", "EXPERIENCE.md")
                )
            )
            isolation_root = run_dir / "isolated-workspaces"
            test_workspace = None
            probe_workspace = None
            validation_workspace = None
            formal_workspace = None
            isolation_valid = False
            pretest_valid = (
                provider_completed
                and scope_valid
                and frozen_manifest_valid
                and required_source_files_valid
                and main_unchanged
                and strategy_documents_present
            )
            if pretest_valid:
                try:
                    test_workspace = self._materialize_verified_source(
                        frozen_v6_source,
                        isolation_root / "v6-tests",
                        v6,
                    )
                    self._verify_runtime_source(
                        run,
                        test_workspace,
                        v6,
                        phase="tests",
                    )
                    tests_ok, test_output = (
                        self._run_candidate_tests(test_workspace)
                    )
                    if tests_ok:
                        probe_workspace = (
                            self._materialize_verified_source(
                                frozen_v6_source,
                                isolation_root / "v6-probes",
                                v6,
                            )
                        )
                        validation_workspace = (
                            self._materialize_verified_source(
                                frozen_v6_source,
                                isolation_root / "v6-validation",
                                v6,
                            )
                        )
                        formal_workspace = (
                            self._materialize_verified_source(
                                frozen_v6_source,
                                isolation_root / "v6-formal",
                                v6,
                            )
                        )
                        isolation_valid = True
                except Exception as exc:
                    tests_ok = False
                    test_output = (
                        "candidate source isolation failed: "
                        f"{type(exc).__name__}: {exc}\n"
                    )
            else:
                tests_ok = False
                test_output = (
                    "candidate tests skipped because the provider did not "
                    "complete or the frozen source is invalid\n"
                )
            (run_dir / "versions" / "v6" / "tests.log").write_text(
                test_output,
                encoding="utf-8",
            )
            runnable = (
                provider_completed
                and scope_valid
                and main_unchanged
                and tests_ok
                and strategy_documents_present
                and frozen_manifest_valid
                and required_source_files_valid
                and isolation_valid
            )
            run.write(
                "version",
                version="v6",
                version_before="v5",
                status=(
                    "available"
                    if runnable
                    else (
                        "provider_failed"
                        if not provider_completed
                        else "invalid"
                    )
                ),
                manifest_hash=v6.content_hash,
                changed_files=v6.changed_files,
                provider_status=act.status,
                scope_valid=scope_valid,
                main_unchanged=main_unchanged,
                frozen_manifest_valid=frozen_manifest_valid,
                required_source_files_valid=(
                    required_source_files_valid
                ),
                isolation_valid=isolation_valid,
                strategy_documents_present=(
                    strategy_documents_present
                ),
                tests_passed=tests_ok,
                runnable=runnable,
            )
            if not runnable:
                status = (
                    "provider_failed"
                    if not provider_completed
                    else "invalid_version"
                )
            else:
                assert probe_workspace is not None
                assert validation_workspace is not None
                assert formal_workspace is not None
                action_disagreement = None
                decision_classes = None
                try:
                    self._verify_runtime_source(
                        run,
                        probe_workspace,
                        v6,
                        phase="probes",
                    )
                    new_actions = self._probe_actions(
                        probe_workspace,
                        probes,
                    )
                    behavior = measure_action_disagreement(
                        probes,
                        new_actions,
                    )
                    action_disagreement = behavior.mean
                    decision_classes = summarize_decision_classes(
                        probes,
                        new_actions,
                    )
                    run.write(
                        "behavior_change",
                        version_before="v5",
                        version_after="v6",
                        probe_set="v6_learning_states",
                        learning_id=self.learning_config.learning_id,
                        decision_count=len(probes),
                        action_disagreement_trace=list(
                            behavior.trace
                        ),
                        action_disagreement=behavior.mean,
                        policy_kl=behavior.policy_kl,
                        policy_kl_status=behavior.policy_kl_status,
                        occupancy_shift=None,
                    )
                    run.write(
                        "decision_class_summary",
                        version="v6",
                        probe_set="v6_learning_states",
                        learning_id=self.learning_config.learning_id,
                        total=decision_classes.total,
                        counts=dict(decision_classes.counts),
                        rates=dict(decision_classes.rates),
                    )
                except Exception as exc:
                    run.write(
                        "behavior_measurement_error",
                        version="v6",
                        diagnostic="decision_space",
                        error=f"{type(exc).__name__}: {exc}",
                    )

                try:
                    self._verify_runtime_source(
                        run,
                        validation_workspace,
                        v6,
                        phase="validation",
                    )
                    validation = evaluator.evaluate(
                        validation_workspace,
                        "v6",
                        "validation",
                        run,
                        cases=validation_cases,
                    )
                    action_profiles["validation"] = (
                        self._persist_action_profile(
                            run,
                            validation,
                            phase="validation",
                            artifact_path=(
                                run_dir
                                / "diagnostics"
                                / "v6-validation-action-profile.json"
                            ),
                            required=False,
                        )
                    )
                except Exception as exc:
                    validation_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    run.write(
                        "pipeline_error",
                        phase="validation",
                        error=validation_error,
                        status="validation_error_non_blocking",
                    )
                finally:
                    try:
                        self._verify_runtime_source(
                            run,
                            formal_workspace,
                            v6,
                            phase="formal",
                        )
                        if (
                            formal_workspace / "main.py"
                        ).read_bytes() != (
                            versions_v5_source / "main.py"
                        ).read_bytes():
                            raise ValueError(
                                "formal main.py differs from v5"
                            )
                        formal_attempted = True
                        formal = evaluator.evaluate(
                            formal_workspace,
                            "v6",
                            "evaluation",
                            run,
                            cases=formal_cases,
                        )
                        action_profiles["formal"] = (
                            self._persist_action_profile(
                                run,
                                formal,
                                phase="formal",
                                artifact_path=(
                                    run_dir
                                    / "diagnostics"
                                    / "v6-formal-action-profile.json"
                                ),
                                required=False,
                            )
                        )
                    except Exception as exc:
                        formal_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        run.write(
                            "pipeline_error",
                            phase="formal",
                            error=formal_error,
                            status="formal_error",
                        )

                valid_validation_count = (
                    sum(
                        result.valid
                        for result in validation.results
                    )
                    if validation is not None
                    else None
                )
                behavior_diagnostics = {
                    "action_disagreement": action_disagreement,
                    "probe_decision_count": len(probes),
                    "decision_classes": (
                        {
                            "total": decision_classes.total,
                            "counts": dict(decision_classes.counts),
                            "rates": dict(decision_classes.rates),
                        }
                        if decision_classes is not None
                        else None
                    ),
                    "validation_complete": (
                        validation is not None
                        and self._complete_learning(
                            validation,
                            len(validation_cases),
                        )
                    ),
                    "valid_validation_case_count": (
                        valid_validation_count
                    ),
                    "validation_case_count": len(validation_cases),
                    "formal_evaluation_blocking": False,
                    "policy_kl": None,
                    "policy_kl_status": (
                        "complete_macro_action_distribution_unavailable"
                    ),
                    "epistemic_information_gain": None,
                    "information_gain_status": (
                        "complete_action_distribution_unavailable"
                    ),
                }
                run.write(
                    "behavior_diagnostics",
                    **behavior_diagnostics,
                )
                if formal is not None:
                    evo_score_6 = formal.score
                    gain_6 = (
                        evo_score_6 - raw_score
                        if evo_score_6 is not None
                        else None
                    )
                    run.write(
                        "evaluation",
                        phase="evolved_6",
                        version="v6",
                        status=formal.status,
                        score=formal.score,
                        gain=gain_6,
                        global_coding_agent_act=global_act_count,
                        per_tier=dict(formal.per_tier),
                        seat_gap=formal.seat_gap,
                    )
                    run.record_act_evaluation(
                        act.act_id,
                        {
                            "benchmark_id": (
                                self.config.benchmark_id
                            ),
                            "version": "v6",
                            "score": formal.score,
                            "gain": gain_6,
                            "global_coding_agent_act": (
                                global_act_count
                            ),
                        },
                    )
                    status = (
                        "complete"
                        if formal.status == "complete"
                        else "incomplete"
                    )
                else:
                    status = "formal_failed"
                    error = formal_error
        except _Round6Abort as exc:
            status = exc.status
            error = str(exc)
            run.write(
                "pipeline_error",
                error=error,
                status=status,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", error=error)
            status = "failed"
        finally:
            if prompt_recovery is not None:
                run.write(
                    "recovery_import",
                    failed_run_id=prompt_recovery.run_id,
                    recovery_mode=prompt_recovery.mode,
                    reused_learning_budget=True,
                    reused_prompt=False,
                    new_provider_act=round_act_count == 1,
                    source_learning_budget=dict(
                        prompt_recovery.local_learning_budget
                    ),
                )
            run.writer.flush()
            quality = inspect_event_file(
                run_dir / "events.jsonl"
            ).to_dict()
            self._write_json(run_dir / "quality.json", quality)
            current_budget = run.budget_snapshot()
            cumulative_budget = _combine_round6_learning_budgets(
                (
                    prompt_recovery.cumulative_learning_budget
                    if prompt_recovery is not None
                    else lineage.learning_budget
                ),
                current_budget,
            )
            local_learning_budget = {
                key: value
                for key, value in current_budget.items()
                if key.startswith("learning_")
            }
            score_history = [
                *lineage.prior_score_history,
                evo_score_6,
            ]
            formal_scores = {
                tier: (
                    formal.per_tier.get(tier)
                    if formal is not None
                    else None
                )
                for tier in ("high", "medium", "low")
            }
            success = self._formal_success(formal_scores)
            feedback_summary = (
                {
                    "episodes": prompt.feedback_episodes_read,
                    "decision_records": (
                        prompt.feedback_decision_records_read
                    ),
                    "serialized_bytes": (
                        prompt.feedback_serialized_bytes_read
                    ),
                    "prompt_bytes": prompt.prompt_bytes,
                    "estimated_tokens": prompt.estimated_tokens,
                    "included_episode_ids": list(
                        prompt.included_episode_ids
                    ),
                    "omitted_episode_ids": list(
                        prompt.omitted_episode_ids
                    ),
                    "selection_policy": prompt.selection_policy,
                }
                if prompt is not None
                else None
            )
            run.finish(
                {
                    "total_episodes": None,
                    "total_steps": None,
                    "total_reward": None,
                    "win_rate": None,
                    "wins": None,
                    "losses": None,
                    "draws": None,
                    "status": status,
                    "benchmark_id": self.config.benchmark_id,
                    "learning_id": self.learning_config.learning_id,
                    "raw_score": raw_score,
                    "evo_score": evo_score_6,
                    "evo_score_1": evo_score_1,
                    "evo_score_2": evo_score_2,
                    "evo_score_3": evo_score_3,
                    "evo_score_4": evo_score_4,
                    "evo_score_5": evo_score_5,
                    "evo_score_6": evo_score_6,
                    "gain": gain_6,
                    "gain_1": evo_score_1 - raw_score,
                    "gain_2": evo_score_2 - raw_score,
                    "gain_3": evo_score_3 - raw_score,
                    "gain_4": evo_score_4 - raw_score,
                    "gain_5": evo_score_5 - raw_score,
                    "gain_6": gain_6,
                    "benchmark_score": evo_score_6,
                    "validation_score": (
                        validation.score
                        if validation is not None
                        else None
                    ),
                    "formal_score": (
                        formal.score if formal is not None else None
                    ),
                    "formal_scores": formal_scores,
                    "formal_high_score": formal_scores["high"],
                    "formal_medium_score": formal_scores["medium"],
                    "formal_low_score": formal_scores["low"],
                    "success": success,
                    "medium_at_least_2_of_6": (
                        success["medium_at_least_2_of_6"]
                    ),
                    "low_retained_6_of_6": (
                        success["low_retained_6_of_6"]
                    ),
                    "high_breakthrough_at_least_1_of_6": (
                        success[
                            "high_breakthrough_at_least_1_of_6"
                        ]
                    ),
                    "evaluation_status": (
                        formal.status
                        if formal is not None
                        else (
                            "error"
                            if formal_attempted
                            else "not_run"
                        )
                    ),
                    "formal_attempted": formal_attempted,
                    "validation_status": (
                        validation.status
                        if validation is not None
                        else (
                            "error"
                            if validation_error is not None
                            else "not_run"
                        )
                    ),
                    "score_history": score_history,
                    "AUC_coding_agent_act": None,
                    "AUC_episode": None,
                    "AUC_env_step": None,
                    "AUC_token": None,
                    "AUC_time": None,
                    "auc_status": (
                        "unavailable_missing_score_point"
                    ),
                    "act_count": global_act_count,
                    "round_act_count": round_act_count,
                    "parent_run_id": lineage.parent_run_id,
                    "parent_version": "v5",
                    "starting_version": "v5",
                    "parent_content_hash": (
                        lineage.v5_manifest.content_hash
                    ),
                    "failed_run_id": (
                        prompt_recovery.run_id
                        if prompt_recovery is not None
                        else None
                    ),
                    "recovery_mode": (
                        prompt_recovery.mode
                        if prompt_recovery is not None
                        else None
                    ),
                    "source_learning_budget": (
                        dict(prompt_recovery.local_learning_budget)
                        if prompt_recovery is not None
                        else None
                    ),
                    "runnable": runnable,
                    "feedback_read": feedback_summary,
                    "behavior_diagnostics": behavior_diagnostics,
                    "formal_evaluation_blocking": False,
                    "action_profiles": action_profiles,
                    "error": error,
                    "validation_error": validation_error,
                    "formal_error": formal_error,
                    "local_learning_budget": local_learning_budget,
                    "cumulative_learning_budget": cumulative_budget,
                    "learning_results": self._result_rows(
                        (v5_learning,)
                    ),
                    "validation_results": self._result_rows(
                        (validation,)
                    ),
                    "formal_results": self._result_rows((formal,)),
                    "benchmark_results": self._result_rows((formal,)),
                }
            )
        return Round6PipelineResult(
            run_dir=run_dir,
            raw_score=raw_score,
            evo_score_1=evo_score_1,
            evo_score_2=evo_score_2,
            evo_score_3=evo_score_3,
            evo_score_4=evo_score_4,
            evo_score_5=evo_score_5,
            evo_score_6=evo_score_6,
            gain_6=gain_6,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
            status=status,
            runnable=runnable,
        )

    def _copy_recovery_prompt(
        self,
        source: _Round6RecoverySource,
        provider_dir: Path,
    ) -> None:
        provider_dir.mkdir(parents=True, exist_ok=True)
        source_provider = source.run_dir / "provider"
        for name in (
            "codex-act-v6.prompt.md",
            "prompt.txt",
            "codex-act-v6.prompt.json",
            "prompt-manifest.json",
            "feedback-receipt.json",
        ):
            shutil.copy2(source_provider / name, provider_dir / name)
        source_budget = source_provider / "provider-budget.json"
        if source_budget.is_file():
            shutil.copy2(
                source_budget,
                provider_dir / "source-provider-budget.json",
            )

    @staticmethod
    def _replace_workspace_from_frozen(
        workspace: Path,
        frozen_source: Path,
    ) -> None:
        for child in workspace.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(
            frozen_source,
            workspace,
            dirs_exist_ok=True,
        )

    def _run_audited_recovery(
        self,
        source: _Round6RecoverySource,
        lineage: Any,
    ) -> Round6PipelineResult:
        provider_retry = source.mode == "provider_retry"
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "learning_id": self.learning_config.learning_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 6,
                "lineage_mode": "legacy_pilot_v5_to_v6",
                "failed_run_id": source.run_id,
                "recovery_mode": source.mode,
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = lineage.raw_score
        evo_score_1 = lineage.evo_score_1
        evo_score_2 = lineage.evo_score_2
        evo_score_3 = lineage.evo_score_3
        evo_score_4 = lineage.evo_score_4
        evo_score_5 = lineage.evo_score_5
        evo_score_6 = None
        gain_6 = None
        global_act_count = int(source.summary["act_count"])
        round_act_count = 0
        status = "failed"
        runnable = False
        error: str | None = None
        validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        validation_error: str | None = None
        formal_error: str | None = None
        formal_attempted = False
        act = None
        action_profiles: dict[str, dict[str, object] | None] = {
            "learning": None,
            "validation": None,
            "formal": None,
        }
        source_profiles = source.summary.get("action_profiles")
        if isinstance(source_profiles, dict):
            learning_profile = source_profiles.get("learning")
            if isinstance(learning_profile, dict):
                action_profiles["learning"] = dict(learning_profile)
        behavior_diagnostics: dict[str, object] | None = None
        learning_cases = tuple(
            build_round6_learning_cases(
                self.config,
                self.learning_config,
            )
        )
        validation_cases = tuple(
            build_round6_validation_cases(self.config)
        )
        formal_cases = tuple(build_evaluation_spec(self.config).cases)
        try:
            imported_v5 = import_round6_source(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            versions_v5_source = (
                run_dir / "versions" / "v5" / "source"
            )
            self._write_benchmark_artifacts(
                run_dir,
                learning_cases,
                validation_cases,
                formal_cases,
            )
            source_learning_matches = (
                source.run_dir / "matches" / "v5"
            )
            run.write(
                "recovery_import",
                failed_run_id=source.run_id,
                recovery_mode=source.mode,
                source_status=source.summary.get("status"),
                reused_learning_evidence=True,
                reused_learning_budget=True,
                reused_prompt=True,
                new_provider_act=provider_retry,
                learning_results_verified=True,
                learning_action_profile_verified=True,
                source_learning_match_artifacts_available=(
                    source_learning_matches.is_dir()
                ),
            )
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v5",
                starting_version="v5",
                version="v5",
                manifest_hash=imported_v5.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=lineage.global_act_count,
                recovery=True,
            )
            run.write(
                "version",
                version="v5",
                status="lineage_parent",
                manifest_hash=imported_v5.content_hash,
                parent_run_id=lineage.parent_run_id,
                recovery=True,
            )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                learning_id=self.learning_config.learning_id,
                learning_cases=[
                    asdict(case) for case in learning_cases
                ],
                validation_cases=[
                    asdict(case) for case in validation_cases
                ],
                evaluation_cases=[
                    asdict(case) for case in formal_cases
                ],
                engine_hash=self.assets.engine_hash,
                replay_skill_sha256=self.replay_skill.sha256,
                recovery=True,
            )
            if source_learning_matches.is_dir():
                shutil.copytree(
                    source_learning_matches,
                    run_dir / "matches" / "v5",
                )
            source_learning_profile = (
                source.run_dir
                / "diagnostics"
                / "v5-learning-action-profile.json"
            )
            target_profile = (
                run_dir
                / "diagnostics"
                / "v5-learning-action-profile.json"
            )
            target_profile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_learning_profile, target_profile)

            provider_dir = run_dir / "provider"
            self._copy_recovery_prompt(source, provider_dir)
            if provider_retry:
                run.write(
                    "provider_retry",
                    failed_run_id=source.run_id,
                    reused_prompt=True,
                    reused_learning_evidence=True,
                    source_act_count=source.summary.get("act_count"),
                )
                raw_path = (
                    provider_dir / "codex-act-v6.raw.jsonl"
                )
                stderr_path = (
                    provider_dir / "codex-act-v6.stderr.log"
                )
                raw_path.touch()
                stderr_path.touch()
                controller = run.create_coding_agent_controller(
                    self.provider,
                    snapshotter=self.snapshotter,
                    budget_phase="learning",
                )
                assert source.prompt_text is not None
                act = controller.run_act(
                    {
                        "prompt": source.prompt_text,
                        "workspace_root": str(workspace),
                        "raw_output_path": str(raw_path),
                        "stderr_output_path": str(stderr_path),
                        "max_artifact_bytes": (
                            self.config.limits.max_artifact_bytes
                        ),
                    },
                    workspace_root=str(workspace),
                    version_before="v5",
                    previous_manifest=imported_v5,
                )
                global_act_count += 1
                round_act_count = 1
                self._write_json(
                    provider_dir / "provider-budget.json",
                    run.budget_snapshot(),
                )
                v6 = self._write_version(
                    run_dir,
                    "v6",
                    workspace,
                    previous=imported_v5,
                )
            else:
                assert source.v6_manifest is not None
                v6 = WorkspaceManifest(
                    content_hash=source.v6_manifest.content_hash,
                    files=dict(source.v6_manifest.files),
                    changed_files=list(
                        source.v6_manifest.changed_files
                    ),
                )
                source_v6 = (
                    source.run_dir / "versions" / "v6" / "source"
                )
                self._replace_workspace_from_frozen(
                    workspace,
                    source_v6,
                )
                recovered_workspace_manifest = (
                    self.snapshotter.capture(workspace)
                )
                if (
                    recovered_workspace_manifest.content_hash
                    != v6.content_hash
                    or recovered_workspace_manifest.files != v6.files
                ):
                    raise ValueError(
                        "recovery v6 workspace copy does not match"
                    )
                frozen_target = (
                    run_dir / "versions" / "v6" / "source"
                )
                shutil.copytree(source_v6, frozen_target)
                self.snapshotter.write_manifest(
                    v6,
                    run_dir / "versions" / "v6" / "manifest.json",
                )

            frozen_v6_source = (
                run_dir / "versions" / "v6" / "source"
            )
            self.snapshotter.write_unified_patch(
                versions_v5_source,
                frozen_v6_source,
                run_dir / "versions" / "v5-to-v6.patch",
            )
            provider_completed = (
                act.status == "completed"
                if provider_retry
                else True
            )
            scope_valid = self._scope_valid(v6.changed_files)
            frozen_manifest_valid = self._frozen_manifest_matches(
                frozen_v6_source,
                v6,
            )
            required_source_files_valid = (
                self._required_frozen_files_valid(
                    frozen_v6_source,
                    v6,
                )
            )
            main_unchanged = (
                v6.files.get("main.py")
                == imported_v5.files.get("main.py")
                and "main.py" not in v6.changed_files
                and (frozen_v6_source / "main.py").is_file()
                and not (frozen_v6_source / "main.py").is_symlink()
                and (
                    frozen_v6_source / "main.py"
                ).read_bytes()
                == (versions_v5_source / "main.py").read_bytes()
            )
            strategy_documents_present = (
                required_source_files_valid
                and all(
                    name in v6.files
                    for name in ("STRATEGY.md", "EXPERIENCE.md")
                )
            )
            isolation_root = run_dir / "isolated-workspaces"
            tests_ok = False
            test_output = (
                "candidate tests skipped because the recovered source "
                "is invalid\n"
            )
            validation_workspace = None
            formal_workspace = None
            isolation_valid = False
            if (
                provider_completed
                and scope_valid
                and frozen_manifest_valid
                and required_source_files_valid
                and main_unchanged
                and strategy_documents_present
            ):
                test_workspace = self._materialize_verified_source(
                    frozen_v6_source,
                    isolation_root / "v6-tests",
                    v6,
                )
                self._verify_runtime_source(
                    run,
                    test_workspace,
                    v6,
                    phase="tests",
                )
                tests_ok, test_output = self._run_candidate_tests(
                    test_workspace
                )
                if tests_ok:
                    validation_workspace = (
                        self._materialize_verified_source(
                            frozen_v6_source,
                            isolation_root / "v6-validation",
                            v6,
                        )
                    )
                    formal_workspace = (
                        self._materialize_verified_source(
                            frozen_v6_source,
                            isolation_root / "v6-formal",
                            v6,
                        )
                    )
                    isolation_valid = True
            (run_dir / "versions" / "v6" / "tests.log").write_text(
                test_output,
                encoding="utf-8",
            )
            runnable = (
                provider_completed
                and scope_valid
                and frozen_manifest_valid
                and required_source_files_valid
                and main_unchanged
                and strategy_documents_present
                and tests_ok
                and isolation_valid
            )
            run.write(
                "version",
                version="v6",
                version_before="v5",
                status=(
                    "available"
                    if runnable
                    else (
                        "provider_failed"
                        if not provider_completed
                        else "invalid"
                    )
                ),
                manifest_hash=v6.content_hash,
                changed_files=v6.changed_files,
                provider_status=(
                    act.status
                    if act is not None
                    else "reused_completed_act"
                ),
                scope_valid=scope_valid,
                main_unchanged=main_unchanged,
                frozen_manifest_valid=frozen_manifest_valid,
                required_source_files_valid=(
                    required_source_files_valid
                ),
                isolation_valid=isolation_valid,
                strategy_documents_present=(
                    strategy_documents_present
                ),
                tests_passed=tests_ok,
                runnable=runnable,
                recovered_from_run_id=source.run_id,
            )
            if not runnable:
                status = (
                    "provider_failed"
                    if not provider_completed
                    else "invalid_version"
                )
            else:
                assert validation_workspace is not None
                assert formal_workspace is not None
                evaluator = (
                    self.evaluator
                    or self._production_evaluator(run_dir)
                )
                try:
                    self._verify_runtime_source(
                        run,
                        validation_workspace,
                        v6,
                        phase="validation",
                    )
                    validation = evaluator.evaluate(
                        validation_workspace,
                        "v6",
                        "validation",
                        run,
                        cases=validation_cases,
                    )
                    action_profiles["validation"] = (
                        self._persist_action_profile(
                            run,
                            validation,
                            phase="validation",
                            artifact_path=(
                                run_dir
                                / "diagnostics"
                                / "v6-validation-action-profile.json"
                            ),
                            required=False,
                        )
                    )
                except Exception as exc:
                    validation_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    run.write(
                        "pipeline_error",
                        phase="validation",
                        error=validation_error,
                        status="validation_error_non_blocking",
                        recovery=True,
                    )
                finally:
                    try:
                        self._verify_runtime_source(
                            run,
                            formal_workspace,
                            v6,
                            phase="formal",
                        )
                        if (
                            formal_workspace / "main.py"
                        ).read_bytes() != (
                            versions_v5_source / "main.py"
                        ).read_bytes():
                            raise ValueError(
                                "formal main.py differs from v5"
                            )
                        formal_attempted = True
                        formal = evaluator.evaluate(
                            formal_workspace,
                            "v6",
                            "evaluation",
                            run,
                            cases=formal_cases,
                        )
                        action_profiles["formal"] = (
                            self._persist_action_profile(
                                run,
                                formal,
                                phase="formal",
                                artifact_path=(
                                    run_dir
                                    / "diagnostics"
                                    / "v6-formal-action-profile.json"
                                ),
                                required=False,
                            )
                        )
                    except Exception as exc:
                        formal_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        run.write(
                            "pipeline_error",
                            phase="formal",
                            error=formal_error,
                            status="formal_error",
                            recovery=True,
                        )
                behavior_diagnostics = {
                    "validation_complete": (
                        validation is not None
                        and self._complete_learning(
                            validation,
                            len(validation_cases),
                        )
                    ),
                    "validation_case_count": len(validation_cases),
                    "formal_evaluation_blocking": False,
                    "recovery_mode": source.mode,
                    "frozen_source_reused": True,
                }
                run.write(
                    "behavior_diagnostics",
                    **behavior_diagnostics,
                )
                if formal is not None:
                    evo_score_6 = formal.score
                    gain_6 = (
                        evo_score_6 - raw_score
                        if evo_score_6 is not None
                        else None
                    )
                    run.write(
                        "evaluation",
                        phase="evolved_6",
                        version="v6",
                        status=formal.status,
                        score=formal.score,
                        gain=gain_6,
                        global_coding_agent_act=global_act_count,
                        per_tier=dict(formal.per_tier),
                        seat_gap=formal.seat_gap,
                        recovery=True,
                    )
                    if act is not None:
                        run.record_act_evaluation(
                            act.act_id,
                            {
                                "benchmark_id": (
                                    self.config.benchmark_id
                                ),
                                "version": "v6",
                                "score": formal.score,
                                "gain": gain_6,
                                "global_coding_agent_act": (
                                    global_act_count
                                ),
                                "recovery": True,
                            },
                        )
                    status = (
                        "complete"
                        if formal.status == "complete"
                        else "incomplete"
                    )
                else:
                    status = "formal_failed"
                    error = formal_error
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write(
                "pipeline_error",
                error=error,
                recovery=True,
            )
            status = "failed"
        finally:
            run.writer.flush()
            quality = inspect_event_file(
                run_dir / "events.jsonl"
            ).to_dict()
            self._write_json(run_dir / "quality.json", quality)
            current_budget = run.budget_snapshot()
            cumulative_budget = _combine_round6_learning_budgets(
                source.cumulative_learning_budget,
                current_budget,
            )
            local_learning_budget = self._learning_budget(
                current_budget
            )
            formal_scores = {
                tier: (
                    formal.per_tier.get(tier)
                    if formal is not None
                    else None
                )
                for tier in ("high", "medium", "low")
            }
            success = self._formal_success(formal_scores)
            run.finish(
                {
                    "total_episodes": None,
                    "total_steps": None,
                    "total_reward": None,
                    "win_rate": None,
                    "wins": None,
                    "losses": None,
                    "draws": None,
                    "status": status,
                    "benchmark_id": self.config.benchmark_id,
                    "learning_id": self.learning_config.learning_id,
                    "raw_score": raw_score,
                    "evo_score": evo_score_6,
                    "evo_score_1": evo_score_1,
                    "evo_score_2": evo_score_2,
                    "evo_score_3": evo_score_3,
                    "evo_score_4": evo_score_4,
                    "evo_score_5": evo_score_5,
                    "evo_score_6": evo_score_6,
                    "gain": gain_6,
                    "gain_1": evo_score_1 - raw_score,
                    "gain_2": evo_score_2 - raw_score,
                    "gain_3": evo_score_3 - raw_score,
                    "gain_4": evo_score_4 - raw_score,
                    "gain_5": evo_score_5 - raw_score,
                    "gain_6": gain_6,
                    "benchmark_score": evo_score_6,
                    "validation_score": (
                        validation.score
                        if validation is not None
                        else None
                    ),
                    "formal_score": (
                        formal.score if formal is not None else None
                    ),
                    "formal_scores": formal_scores,
                    "formal_high_score": formal_scores["high"],
                    "formal_medium_score": formal_scores["medium"],
                    "formal_low_score": formal_scores["low"],
                    "success": success,
                    "medium_at_least_2_of_6": (
                        success["medium_at_least_2_of_6"]
                    ),
                    "low_retained_6_of_6": (
                        success["low_retained_6_of_6"]
                    ),
                    "high_breakthrough_at_least_1_of_6": (
                        success[
                            "high_breakthrough_at_least_1_of_6"
                        ]
                    ),
                    "evaluation_status": (
                        formal.status
                        if formal is not None
                        else (
                            "error"
                            if formal_attempted
                            else "not_run"
                        )
                    ),
                    "formal_attempted": formal_attempted,
                    "validation_status": (
                        validation.status
                        if validation is not None
                        else (
                            "error"
                            if validation_error is not None
                            else "not_run"
                        )
                    ),
                    "score_history": [
                        *lineage.prior_score_history,
                        evo_score_6,
                    ],
                    "AUC_coding_agent_act": None,
                    "AUC_episode": None,
                    "AUC_env_step": None,
                    "AUC_token": None,
                    "AUC_time": None,
                    "auc_status": (
                        "unavailable_missing_score_point"
                    ),
                    "act_count": global_act_count,
                    "round_act_count": round_act_count,
                    "parent_run_id": lineage.parent_run_id,
                    "parent_version": "v5",
                    "starting_version": "v5",
                    "parent_content_hash": (
                        lineage.v5_manifest.content_hash
                    ),
                    "failed_run_id": source.run_id,
                    "recovery_mode": source.mode,
                    "source_learning_budget": dict(
                        source.local_learning_budget
                    ),
                    "runnable": runnable,
                    "feedback_read": source.summary.get(
                        "feedback_read"
                    ),
                    "behavior_diagnostics": behavior_diagnostics,
                    "formal_evaluation_blocking": False,
                    "action_profiles": action_profiles,
                    "error": error,
                    "validation_error": validation_error,
                    "formal_error": formal_error,
                    "local_learning_budget": local_learning_budget,
                    "cumulative_learning_budget": cumulative_budget,
                    "learning_results": list(
                        source.summary.get("learning_results") or []
                    ),
                    "validation_results": self._result_rows(
                        (validation,)
                    ),
                    "formal_results": self._result_rows((formal,)),
                    "benchmark_results": self._result_rows((formal,)),
                }
            )
        return Round6PipelineResult(
            run_dir=run_dir,
            raw_score=raw_score,
            evo_score_1=evo_score_1,
            evo_score_2=evo_score_2,
            evo_score_3=evo_score_3,
            evo_score_4=evo_score_4,
            evo_score_5=evo_score_5,
            evo_score_6=evo_score_6,
            gain_6=gain_6,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
            status=status,
            runnable=runnable,
        )

    def recover(
        self,
        failed_run_dir: Path,
    ) -> Round6PipelineResult:
        """Create a new audited run for one supported v6 recovery state."""
        lineage = load_round6_parent(
            self.parent_run_dir,
            self.expected_parent_hash,
        )
        source = self._validate_recovery_source(
            failed_run_dir,
            lineage,
        )
        if source.mode == "prompt_rebuild":
            return self._run(prompt_recovery=source)
        return self._run_audited_recovery(source, lineage)
