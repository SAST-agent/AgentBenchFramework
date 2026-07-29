"""Rollback-guided, non-gated v3/v4-to-v5 Generals orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run

from .assets import (
    load_pilot_config,
    load_round5_learning_config,
    require_valid_assets,
    resolve_assets,
    resolve_replay_skill,
)
from .dense import (
    DenseEpisodeSummary,
    build_dense_trace,
    summarize_dense_trace,
)
from .evaluator import (
    GeneralsEvaluation,
    GeneralsEvaluator,
    build_evaluation_spec,
    build_round5_learning_cases,
)
from .lineage_v5 import (
    Round5ParentLineage,
    import_round5_sources,
    load_round5_parent,
)
from .measurement import (
    DecisionClassSummary,
    ProbeState,
    measure_action_disagreement,
    summarize_decision_classes,
)
from .models import (
    AssetLayout,
    PilotConfig,
    ReplaySkillAsset,
    Round5LearningConfig,
)
from .pipeline import GeneralsHLPipeline
from .pipeline_v3 import paired_dense_deltas
from .pipeline_v4 import _combine_learning_budgets
from .prompt import PromptBuildResult
from .prompt_v5 import (
    VersionedCriticalEvidence,
    build_round5_prompt,
)
from .replay import (
    CriticalLearningEvidence,
    CriticalWindowSelection,
    build_critical_learning_evidence,
    build_learning_replay,
)


V5_SELECTION_REASONS = (
    "first_decision",
    "first_non_end_action",
    "first_main_pressure",
    "before_steepest_territory_drop",
    "before_steepest_army_drop",
    "final_decision",
)


@dataclass(frozen=True)
class Round5PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score_1: float | None
    evo_score_2: float | None
    evo_score_3: float | None
    evo_score_4: float | None
    evo_score_5: float | None
    gain_5: float | None
    global_act_count: int
    round_act_count: int
    status: str
    runnable: bool


class _Round5Abort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PriorRound5Attempt:
    run_dir: Path
    run_id: str
    learning_budget: Mapping[str, object]


def _load_prior_attempt(
    run_dir: Path,
    lineage: Round5ParentLineage,
    learning_id: str,
) -> PriorRound5Attempt:
    target = Path(run_dir).resolve()
    try:
        summary = json.loads(
            (target / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prior v5 attempt: {exc}") from exc
    if summary.get("status") != "prompt_incomplete":
        raise ValueError(
            "prior v5 attempt must have status prompt_incomplete"
        )
    if (
        summary.get("parent_run_id") != lineage.parent_run_id
        or summary.get("parent_version") != "v4"
        or summary.get("starting_version") != "v3"
    ):
        raise ValueError("prior v5 attempt lineage does not match")
    if summary.get("learning_id") != learning_id:
        raise ValueError("prior v5 attempt learning suite does not match")
    if (
        int(summary.get("act_count", -1)) != lineage.global_act_count
        or int(summary.get("round_act_count", -1)) != 0
    ):
        raise ValueError(
            "prior v5 prompt attempt must not contain a coding-agent act"
        )
    raw_budget = summary.get("budget")
    if not isinstance(raw_budget, dict):
        raise ValueError("prior v5 attempt budget is missing")
    learning_budget = {
        str(key): value
        for key, value in raw_budget.items()
        if str(key).startswith("learning_")
    }
    if (
        not learning_budget
        or int(
            learning_budget.get("learning_coding_agent_acts", -1)
        )
        != 0
    ):
        raise ValueError(
            "prior v5 prompt attempt learning budget is invalid"
        )
    return PriorRound5Attempt(
        run_dir=target,
        run_id=str(summary.get("run_id", target.name)),
        learning_budget=learning_budget,
    )


class GeneralsHLRound5Pipeline(GeneralsHLPipeline):
    """Run one rollback-guided v5 act and evaluate every runnable result."""

    def __init__(
        self,
        config: PilotConfig,
        learning_config: Round5LearningConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        parent_run_dir: Path,
        expected_parent_hash: str,
        expected_rollback_hash: str,
        campaign_budget_receipt: Path,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        prior_attempt_run_dir: Path | None = None,
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
        self.expected_rollback_hash = expected_rollback_hash
        self.campaign_budget_receipt = Path(campaign_budget_receipt)
        self.prior_attempt_run_dir = (
            Path(prior_attempt_run_dir)
            if prior_attempt_run_dir is not None
            else None
        )
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
        expected_rollback_hash: str,
        campaign_budget_receipt: Path,
        data_dir: Path,
        provider: ProviderAdapter,
        prior_attempt_run_dir: Path | None = None,
    ) -> "GeneralsHLRound5Pipeline":
        config = load_pilot_config(manifest_path)
        learning = load_round5_learning_config(
            learning_manifest_path,
            config,
        )
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        skill = resolve_replay_skill(
            agentbench_root,
            replay_skill_path,
        )
        return cls(
            config=config,
            learning_config=learning,
            assets=assets,
            replay_skill=skill,
            parent_run_dir=parent_run_dir,
            expected_parent_hash=expected_parent_hash,
            expected_rollback_hash=expected_rollback_hash,
            campaign_budget_receipt=campaign_budget_receipt,
            data_dir=data_dir,
            provider=provider,
            prior_attempt_run_dir=prior_attempt_run_dir,
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
                "case_id": item.case_id,
                "outcome": item.outcome,
                "valid": item.valid,
                "error": item.error,
                **item.metadata,
            }
            for evaluation in evaluations
            if evaluation is not None
            for item in evaluation.results
        ]

    def _write_benchmark_artifacts(self, run_dir: Path) -> None:
        formal = build_evaluation_spec(self.config).cases
        learning = build_round5_learning_cases(
            self.config,
            self.learning_config,
        )
        skill_target = run_dir / "benchmark" / "replay-skill-v1.md"
        skill_target.parent.mkdir(parents=True, exist_ok=True)
        skill_target.write_text(
            self.replay_skill.text,
            encoding="utf-8",
        )
        self._write_json(
            run_dir / "benchmark" / "formal-spec.json",
            {
                "benchmark_id": self.config.benchmark_id,
                "engine_hash": self.assets.engine_hash,
                "evaluation_cases": [asdict(case) for case in formal],
            },
        )
        self._write_json(
            run_dir / "benchmark" / "v5-learning-spec.json",
            {
                "learning_id": self.learning_config.learning_id,
                "opponent_id": self.learning_config.opponent_id,
                "seeds": list(self.learning_config.seeds),
                "seats": list(self.learning_config.seats),
                "cases": [asdict(case) for case in learning],
            },
        )
        self._write_json(
            run_dir / "benchmark" / "replay-skill-v1.json",
            {
                "source_path": str(self.replay_skill.path),
                "sha256": self.replay_skill.sha256,
                "bytes": len(
                    self.replay_skill.text.encode("utf-8")
                ),
                "artifact_ref": str(skill_target),
            },
        )

    @staticmethod
    def _critical_feedback(
        evaluation: GeneralsEvaluation,
        version: str,
        tier_by_case: Mapping[str, str],
    ) -> tuple[
        tuple[CriticalLearningEvidence, ...],
        tuple[CriticalWindowSelection, ...],
        tuple[ProbeState, ...],
        tuple[DenseEpisodeSummary, ...],
    ]:
        evidence: list[CriticalLearningEvidence] = []
        selections: list[CriticalWindowSelection] = []
        probes: list[ProbeState] = []
        dense: list[DenseEpisodeSummary] = []
        for match in evaluation.matches:
            replay_id = (
                f"learn5-high-s{match.seed}"
                f"-p{match.evaluated_seat}"
            )
            raw = build_learning_replay(
                match,
                "baseline",
                tier_by_case[match.case_id],
            )
            replay = replace(
                raw,
                replay_id=replay_id,
                decisions=tuple(
                    replace(
                        item,
                        state_id=(
                            f"{version}:{replay_id}-d{index}"
                        ),
                    )
                    for index, item in enumerate(
                        raw.decisions,
                        start=1,
                    )
                ),
            )
            trace = build_dense_trace(match)
            summary = summarize_dense_trace(match, trace)
            compact, selection = build_critical_learning_evidence(
                replay,
                summary,
                trace,
                max_decisions=6,
                selection_reasons=V5_SELECTION_REASONS,
                include_strategic_targets=True,
            )
            selected = set(selection.selected_state_ids)
            evidence.append(compact)
            selections.append(selection)
            dense.append(summary)
            probes.extend(
                ProbeState(
                    item.state_id,
                    {**item.state, "my_seat": item.seat},
                    item.action,
                )
                for item in replay.decisions
                if item.state_id in selected
            )
        return (
            tuple(evidence),
            tuple(selections),
            tuple(probes),
            tuple(dense),
        )

    @staticmethod
    def _dense_summaries(
        evaluation: GeneralsEvaluation,
    ) -> tuple[DenseEpisodeSummary, ...]:
        return tuple(
            summarize_dense_trace(match, build_dense_trace(match))
            for match in evaluation.matches
        )

    @staticmethod
    def _complete_learning(
        evaluation: GeneralsEvaluation,
        case_count: int,
    ) -> bool:
        return (
            evaluation.status == "complete"
            and len(evaluation.results) == case_count
            and sum(item.valid for item in evaluation.results)
            == case_count
        )

    def run(self) -> Round5PipelineResult:
        lineage = load_round5_parent(
            self.parent_run_dir,
            self.expected_parent_hash,
            self.expected_rollback_hash,
            self.campaign_budget_receipt,
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
                "round": 5,
                "lineage_mode": "pilot_v4_to_v5_rollback_v3",
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = lineage.raw_score
        evo_score_1 = lineage.evo_score_1
        evo_score_2 = lineage.evo_score_2
        evo_score_3 = lineage.evo_score_3
        evo_score_4 = lineage.evo_score_4
        evo_score_5 = None
        gain_5 = None
        global_act_count = lineage.global_act_count
        round_act_count = 0
        status = "failed"
        error = None
        runnable = False
        v3_learning: GeneralsEvaluation | None = None
        v4_learning: GeneralsEvaluation | None = None
        v5_validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        prompt: PromptBuildResult | None = None
        diagnostics: dict[str, object] | None = None
        prior_attempt: PriorRound5Attempt | None = None
        act = None
        try:
            if self.prior_attempt_run_dir is not None:
                prior_attempt = _load_prior_attempt(
                    self.prior_attempt_run_dir,
                    lineage,
                    self.learning_config.learning_id,
                )
                run.write(
                    "prior_attempt_import",
                    prior_attempt_run_id=prior_attempt.run_id,
                    prior_attempt_status="prompt_incomplete",
                    **dict(prior_attempt.learning_budget),
                )
            imported_v3, imported_v4 = import_round5_sources(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            self._write_benchmark_artifacts(run_dir)
            learning_cases = build_round5_learning_cases(
                self.config,
                self.learning_config,
            )
            formal_cases = build_evaluation_spec(self.config).cases
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v4",
                version="v4",
                manifest_hash=imported_v4.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=global_act_count,
            )
            run.write(
                "version_rollback",
                parent_run_id=lineage.parent_run_id,
                parent_version="v4",
                parent_content_hash=imported_v4.content_hash,
                starting_version="v3",
                rollback_source_version="v3",
                rollback_content_hash=imported_v3.content_hash,
            )
            for version, manifest, source_status in (
                ("v3", imported_v3, "rollback_source"),
                ("v4", imported_v4, "lineage_parent"),
            ):
                run.write(
                    "version",
                    version=version,
                    status=source_status,
                    manifest_hash=manifest.content_hash,
                    parent_run_id=lineage.parent_run_id,
                )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                evaluation_cases=[
                    asdict(case) for case in formal_cases
                ],
                learning_cases=[
                    asdict(case) for case in learning_cases
                ],
                learning_versions=["v3", "v4"],
                learning_id=self.learning_config.learning_id,
                engine_hash=self.assets.engine_hash,
                replay_skill_sha256=self.replay_skill.sha256,
            )
            evaluator = self.evaluator or self._production_evaluator(run_dir)
            v3_learning = evaluator.evaluate(
                run_dir / "versions" / "v3" / "source",
                "v3",
                "learning",
                run,
                cases=learning_cases,
            )
            v4_learning = evaluator.evaluate(
                run_dir / "versions" / "v4" / "source",
                "v4",
                "learning",
                run,
                cases=learning_cases,
            )
            for version, evaluation in (
                ("v3", v3_learning),
                ("v4", v4_learning),
            ):
                valid = sum(item.valid for item in evaluation.results)
                run.write(
                    "learning_validation",
                    version=version,
                    phase="learning",
                    learning_id=self.learning_config.learning_id,
                    status=evaluation.status,
                    case_count=len(learning_cases),
                    valid_case_count=valid,
                )
                if not self._complete_learning(
                    evaluation,
                    len(learning_cases),
                ):
                    raise _Round5Abort(
                        "learning_failed",
                        f"{version} strongest-human suite is incomplete",
                    )
            tiers = {
                case.case_id: str(case.metadata["tier"])
                for case in learning_cases
            }
            (
                v3_evidence,
                v3_selections,
                v3_probes,
                v3_dense,
            ) = self._critical_feedback(v3_learning, "v3", tiers)
            (
                v4_evidence,
                v4_selections,
                v4_probes,
                v4_dense,
            ) = self._critical_feedback(v4_learning, "v4", tiers)
            if not v3_probes or not v4_probes:
                raise _Round5Abort(
                    "learning_probe_failed",
                    "paired feedback produced no selected decisions",
                )
            for version, selections in (
                ("v3", v3_selections),
                ("v4", v4_selections),
            ):
                for selection in selections:
                    run.write(
                        "critical_window_selection",
                        version=version,
                        replay_id=selection.replay_id,
                        total_decision_count=(
                            selection.total_decision_count
                        ),
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
            v3_actions = {
                probe.state_id: probe.old_action
                for probe in v3_probes
            }
            v4_actions = {
                probe.state_id: probe.old_action
                for probe in v4_probes
            }
            v3_classes = summarize_decision_classes(
                v3_probes,
                v3_actions,
            )
            v4_classes = summarize_decision_classes(
                v4_probes,
                v4_actions,
            )
            for version, classes in (
                ("v3", v3_classes),
                ("v4", v4_classes),
            ):
                run.write(
                    "decision_class_summary",
                    version=version,
                    probe_set="v5_new_learning",
                    learning_id=self.learning_config.learning_id,
                    total=classes.total,
                    counts=dict(classes.counts),
                    rates=dict(classes.rates),
                )
            v4_minus_v3 = paired_dense_deltas(
                v3_dense,
                v4_dense,
            )
            versioned = tuple(
                VersionedCriticalEvidence("v3", item)
                for item in v3_evidence
            ) + tuple(
                VersionedCriticalEvidence("v4", item)
                for item in v4_evidence
            )
            try:
                prompt = build_round5_prompt(
                    benchmark_id=self.config.benchmark_id,
                    v3_strategy=(workspace / "strategy.py").read_text(
                        encoding="utf-8"
                    ),
                    v3_experience=(workspace / "EXPERIENCE.md").read_text(
                        encoding="utf-8"
                    ),
                    v4_strategy=(
                        lineage.v4_source / "strategy.py"
                    ).read_text(encoding="utf-8"),
                    v4_experience=(
                        lineage.v4_source / "EXPERIENCE.md"
                    ).read_text(encoding="utf-8"),
                    rules_text=self.rules_path.read_text(
                        encoding="utf-8"
                    ),
                    replay_skill_text=self.replay_skill.text,
                    replay_skill_sha256=self.replay_skill.sha256,
                    evidence=versioned,
                    decision_classes={
                        "v3": v3_classes,
                        "v4": v4_classes,
                    },
                    dense_deltas=v4_minus_v3,
                    max_bytes=self.prompt_max_bytes,
                )
            except ValueError as exc:
                if "prompt context exceeds max_bytes" not in str(exc):
                    raise
                raise _Round5Abort(
                    "prompt_incomplete",
                    str(exc),
                ) from exc
            expected_episode_ids = {
                f"{version}:learn5-high-s{seed}-p{seat}"
                for seed in self.learning_config.seeds
                for seat in self.learning_config.seats
                for version in ("v3", "v4")
            }
            if (
                prompt.truncated
                or set(prompt.included_episode_ids)
                != expected_episode_ids
            ):
                raise _Round5Abort(
                    "prompt_incomplete",
                    "v5 prompt omitted a declared versioned episode",
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
            canonical_prompt = provider_dir / "codex-act-v5.prompt.md"
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
                provider_dir / "codex-act-v5.prompt.json",
                prompt_manifest,
            )
            self._write_json(
                provider_dir / "prompt-manifest.json",
                prompt_manifest,
            )
            controller = run.create_coding_agent_controller(
                self.provider,
                snapshotter=self.snapshotter,
                budget_phase="learning",
            )
            act = controller.run_act(
                {
                    "prompt": prompt.prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(
                        provider_dir / "codex-act-v5.raw.jsonl"
                    ),
                    "stderr_output_path": str(
                        provider_dir / "codex-act-v5.stderr.log"
                    ),
                    "max_artifact_bytes": (
                        self.config.limits.max_artifact_bytes
                    ),
                },
                workspace_root=str(workspace),
                version_before="v3",
                previous_manifest=imported_v3,
            )
            round_act_count = 1
            global_act_count = lineage.global_act_count + 1
            v5 = self._write_version(
                run_dir,
                "v5",
                workspace,
                previous=imported_v3,
            )
            self.snapshotter.write_unified_patch(
                run_dir / "versions" / "v3" / "source",
                run_dir / "versions" / "v5" / "source",
                run_dir / "versions" / "v3-to-v5.patch",
            )
            self.snapshotter.write_unified_patch(
                run_dir / "versions" / "v4" / "source",
                run_dir / "versions" / "v5" / "source",
                run_dir / "versions" / "v4-to-v5.patch",
            )
            protected_unchanged = all(
                path
                in {
                    "strategy.py",
                    "STRATEGY.md",
                    "EXPERIENCE.md",
                }
                or path.startswith("tests/")
                or path.startswith("policy/")
                for path in v5.changed_files
            )
            experience_present = (workspace / "EXPERIENCE.md").is_file()
            provider_completed = act.status == "completed"
            if provider_completed and protected_unchanged:
                tests_ok, test_output = self._baseline_tests(workspace)
            else:
                tests_ok = False
                test_output = (
                    "strategy tests skipped because provider did not "
                    "complete or protected files changed\n"
                )
            (run_dir / "versions" / "v5" / "tests.log").write_text(
                test_output,
                encoding="utf-8",
            )
            runnable = (
                provider_completed
                and protected_unchanged
                and tests_ok
                and experience_present
            )
            run.write(
                "version",
                version="v5",
                version_before="v3",
                rollback_from="v4",
                status=(
                    "available"
                    if runnable
                    else (
                        "provider_failed"
                        if not provider_completed
                        else "invalid"
                    )
                ),
                manifest_hash=v5.content_hash,
                changed_files=v5.changed_files,
                provider_status=act.status,
                protected_files_unchanged=protected_unchanged,
                experience_present=experience_present,
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
                combined_probes = (*v3_probes, *v4_probes)
                new_actions: dict[
                    str,
                    tuple[tuple[int, ...], ...],
                ] = {}
                action_vs_v3 = None
                action_vs_v4 = None
                v5_classes: DecisionClassSummary = v3_classes
                try:
                    new_actions = self._probe_actions(
                        workspace,
                        combined_probes,
                    )
                    behavior_v3 = measure_action_disagreement(
                        v3_probes,
                        {
                            probe.state_id: new_actions[probe.state_id]
                            for probe in v3_probes
                        },
                    )
                    behavior_v4 = measure_action_disagreement(
                        v4_probes,
                        {
                            probe.state_id: new_actions[probe.state_id]
                            for probe in v4_probes
                        },
                    )
                    action_vs_v3 = behavior_v3.mean
                    action_vs_v4 = behavior_v4.mean
                    v5_classes = summarize_decision_classes(
                        combined_probes,
                        new_actions,
                    )
                    for compared, probes, behavior in (
                        ("v3", v3_probes, behavior_v3),
                        ("v4", v4_probes, behavior_v4),
                    ):
                        run.write(
                            "behavior_change",
                            version_before=compared,
                            version_after="v5",
                            probe_set=f"{compared}_learning_states",
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
                except Exception as exc:
                    run.write(
                        "behavior_measurement_error",
                        version="v5",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                run.write(
                    "decision_class_summary",
                    version="v5",
                    probe_set="combined_v3_v4_learning_states",
                    learning_id=self.learning_config.learning_id,
                    total=v5_classes.total,
                    counts=dict(v5_classes.counts),
                    rates=dict(v5_classes.rates),
                )
                validation_error = None
                try:
                    v5_validation = evaluator.evaluate(
                        workspace,
                        "v5",
                        "validation",
                        run,
                        cases=learning_cases,
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
                valid_v5 = (
                    sum(
                        item.valid
                        for item in v5_validation.results
                    )
                    if v5_validation is not None
                    else 0
                )
                v5_dense = (
                    self._dense_summaries(v5_validation)
                    if v5_validation is not None
                    else ()
                )
                dense_vs_v3 = paired_dense_deltas(
                    v3_dense,
                    v5_dense,
                )
                dense_vs_v4 = paired_dense_deltas(
                    v4_dense,
                    v5_dense,
                )
                validation_complete = (
                    v5_validation is not None
                    and self._complete_learning(
                        v5_validation,
                        len(learning_cases),
                    )
                )
                run.write(
                    "learning_validation",
                    version="v5",
                    phase="validation",
                    learning_id=self.learning_config.learning_id,
                    status=(
                        v5_validation.status
                        if v5_validation is not None
                        else "error"
                    ),
                    case_count=len(learning_cases),
                    valid_case_count=valid_v5,
                    dense_deltas_vs_v3=dense_vs_v3,
                    dense_deltas_vs_v4=dense_vs_v4,
                    error=validation_error,
                )
                diagnostics = {
                    "action_disagreement_vs_v3": action_vs_v3,
                    "action_disagreement_vs_v4": action_vs_v4,
                    "v3_probe_decision_count": len(v3_probes),
                    "v4_probe_decision_count": len(v4_probes),
                    "validation_complete": validation_complete,
                    "valid_validation_case_count": valid_v5,
                    "validation_case_count": len(learning_cases),
                    "dense_deltas_vs_v3": dense_vs_v3,
                    "dense_deltas_vs_v4": dense_vs_v4,
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
                run.write("behavior_diagnostics", **diagnostics)
                formal = evaluator.evaluate(
                    workspace,
                    "v5",
                    "evaluation",
                    run,
                    cases=formal_cases,
                )
                evo_score_5 = formal.score
                gain_5 = (
                    evo_score_5 - raw_score
                    if evo_score_5 is not None
                    else None
                )
                run.write(
                    "evaluation",
                    phase="evolved_5",
                    version="v5",
                    status=formal.status,
                    score=formal.score,
                    gain=gain_5,
                    global_coding_agent_act=global_act_count,
                    per_tier=dict(formal.per_tier),
                    seat_gap=formal.seat_gap,
                )
                run.record_act_evaluation(
                    act.act_id,
                    {
                        "benchmark_id": self.config.benchmark_id,
                        "version": "v5",
                        "score": formal.score,
                        "gain": gain_5,
                        "global_coding_agent_act": global_act_count,
                    },
                )
                status = (
                    "complete"
                    if formal.status == "complete"
                    else "incomplete"
                )
        except _Round5Abort as exc:
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
            run.writer.flush()
            quality = inspect_event_file(
                run_dir / "events.jsonl"
            ).to_dict()
            self._write_json(run_dir / "quality.json", quality)
            score_history = [
                *lineage.prior_score_history,
                evo_score_5,
            ]
            current_budget = run.budget_snapshot()
            prior_cumulative = _combine_learning_budgets(
                lineage.learning_budget,
                (
                    prior_attempt.learning_budget
                    if prior_attempt is not None
                    else {}
                ),
            )
            cumulative_budget = _combine_learning_budgets(
                prior_cumulative,
                current_budget,
            )
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
                    "status": status,
                    "benchmark_id": self.config.benchmark_id,
                    "learning_id": self.learning_config.learning_id,
                    "raw_score": raw_score,
                    "evo_score": evo_score_5,
                    "evo_score_1": evo_score_1,
                    "evo_score_2": evo_score_2,
                    "evo_score_3": evo_score_3,
                    "evo_score_4": evo_score_4,
                    "evo_score_5": evo_score_5,
                    "gain": gain_5,
                    "gain_1": evo_score_1 - raw_score,
                    "gain_2": evo_score_2 - raw_score,
                    "gain_3": evo_score_3 - raw_score,
                    "gain_4": evo_score_4 - raw_score,
                    "gain_5": gain_5,
                    "benchmark_score": evo_score_5,
                    "evaluation_status": (
                        formal.status
                        if formal is not None
                        else "not_run"
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
                    "parent_version": "v4",
                    "starting_version": "v3",
                    "rollback_source_version": "v3",
                    "parent_content_hash": (
                        lineage.v4_manifest.content_hash
                    ),
                    "rollback_content_hash": (
                        lineage.v3_manifest.content_hash
                    ),
                    "prior_attempt_run_id": (
                        prior_attempt.run_id
                        if prior_attempt is not None
                        else None
                    ),
                    "runnable": runnable,
                    "feedback_read": feedback_summary,
                    "behavior_diagnostics": diagnostics,
                    "error": error,
                    "cumulative_learning_budget": cumulative_budget,
                    "benchmark_results": self._result_rows((formal,)),
                    "learning_results": self._result_rows(
                        (v3_learning, v4_learning, v5_validation)
                    ),
                }
            )
        return Round5PipelineResult(
            run_dir=run_dir,
            raw_score=raw_score,
            evo_score_1=evo_score_1,
            evo_score_2=evo_score_2,
            evo_score_3=evo_score_3,
            evo_score_4=evo_score_4,
            evo_score_5=evo_score_5,
            gain_5=gain_5,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
            status=status,
            runnable=runnable,
        )
