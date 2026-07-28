"""Replay-guided, non-gated v3-to-v4 Generals HL orchestration."""

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
    load_round4_learning_config,
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
    build_round4_learning_cases,
)
from .lineage_v4 import (
    Round4ParentLineage,
    import_parent_v3,
    load_round4_parent,
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
    Round4LearningConfig,
)
from .pipeline import GeneralsHLPipeline
from .pipeline_v3 import paired_dense_deltas
from .prompt import PromptBuildResult, build_round4_prompt
from .replay import (
    CriticalLearningEvidence,
    CriticalWindowSelection,
    build_critical_learning_evidence,
    build_learning_replay,
)


@dataclass(frozen=True)
class Round4PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score_1: float | None
    evo_score_2: float | None
    evo_score_3: float | None
    evo_score_4: float | None
    gain_4: float | None
    global_act_count: int
    round_act_count: int
    status: str
    runnable: bool


class _Round4Abort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class PriorRound4Attempt:
    run_dir: Path
    run_id: str
    learning_budget: Mapping[str, object]


def _load_prior_attempt(
    run_dir: Path,
    lineage: Round4ParentLineage,
    learning_id: str,
) -> PriorRound4Attempt:
    target = Path(run_dir).resolve()
    try:
        summary = json.loads(
            (target / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prior v4 attempt: {exc}") from exc
    if summary.get("status") != "prompt_incomplete":
        raise ValueError(
            "prior v4 attempt must have status prompt_incomplete"
        )
    if (
        summary.get("parent_run_id") != lineage.parent_run_id
        or summary.get("parent_version") != "v3"
    ):
        raise ValueError("prior v4 attempt parent does not match")
    if summary.get("learning_id") != learning_id:
        raise ValueError("prior v4 attempt learning suite does not match")
    if (
        int(summary.get("act_count", -1)) != lineage.global_act_count
        or int(summary.get("round_act_count", -1)) != 0
    ):
        raise ValueError(
            "prior v4 prompt attempt must not contain a coding-agent act"
        )
    raw_budget = summary.get("budget")
    if not isinstance(raw_budget, dict):
        raise ValueError("prior v4 attempt budget is missing")
    learning_budget = {
        str(key): value
        for key, value in raw_budget.items()
        if str(key).startswith("learning_")
    }
    if not learning_budget:
        raise ValueError("prior v4 attempt learning budget is missing")
    if int(
        learning_budget.get("learning_coding_agent_acts", -1)
    ) != 0:
        raise ValueError(
            "prior v4 prompt attempt budget contains a coding-agent act"
        )
    return PriorRound4Attempt(
        run_dir=target,
        run_id=str(summary.get("run_id", target.name)),
        learning_budget=learning_budget,
    )


def _combine_learning_budgets(
    parent: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, float | int | None]:
    keys = {
        str(key)
        for budget in (parent, current)
        for key in budget
        if str(key).startswith("learning_")
    }
    combined: dict[str, float | int | None] = {}
    for key in sorted(keys):
        left = parent.get(key, 0)
        right = current.get(key, 0)
        if left is None or right is None:
            combined[key] = None
            continue
        value = float(left) + float(right)
        combined[key] = value if key.endswith("_time_s") else int(value)
    return combined


def derive_round4_campaign_budget(
    success_run_dir: Path,
    prior_attempt_run_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Write a separate budget correction receipt without mutating runs."""
    success_dir = Path(success_run_dir).resolve()
    prior_dir = Path(prior_attempt_run_dir).resolve()
    target = Path(output_path).resolve()
    if target.is_relative_to(success_dir) or target.is_relative_to(prior_dir):
        raise ValueError(
            "campaign budget receipt must live outside finalized runs"
        )
    success_path = success_dir / "summary.json"
    prior_path = prior_dir / "summary.json"
    try:
        success_raw = success_path.read_bytes()
        prior_raw = prior_path.read_bytes()
        success = json.loads(success_raw)
        prior = json.loads(prior_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v4 campaign summaries: {exc}") from exc
    if success.get("status") != "complete":
        raise ValueError("v4 campaign success run must be complete")
    if prior.get("status") != "prompt_incomplete":
        raise ValueError(
            "v4 campaign prior run must be prompt_incomplete"
        )
    for key in ("parent_run_id", "parent_version", "learning_id"):
        if success.get(key) != prior.get(key):
            raise ValueError(
                f"v4 campaign summaries disagree on {key}"
            )
    if success.get("prior_attempt_run_id") not in {
        None,
        "",
    }:
        raise ValueError(
            "v4 success summary already declares a prior attempt"
        )
    if int(prior.get("round_act_count", -1)) != 0:
        raise ValueError(
            "v4 prior prompt attempt must not contain a coding-agent act"
        )
    before = success.get("cumulative_learning_budget")
    prior_budget_raw = prior.get("budget")
    if not isinstance(before, dict) or not isinstance(
        prior_budget_raw,
        dict,
    ):
        raise ValueError("v4 campaign learning budgets are missing")
    prior_budget = {
        str(key): value
        for key, value in prior_budget_raw.items()
        if str(key).startswith("learning_")
    }
    if int(
        prior_budget.get("learning_coding_agent_acts", -1)
    ) != 0:
        raise ValueError(
            "v4 prior prompt attempt budget contains an act"
        )
    after = _combine_learning_budgets(before, prior_budget)
    receipt: dict[str, object] = {
        "schema_version": "1.0",
        "status": "derived_prior_attempt_added",
        "success_run_id": success.get("run_id", success_dir.name),
        "prior_attempt_run_id": prior.get("run_id", prior_dir.name),
        "parent_run_id": success.get("parent_run_id"),
        "learning_id": success.get("learning_id"),
        "success_summary_sha256": hashlib.sha256(
            success_raw
        ).hexdigest(),
        "prior_summary_sha256": hashlib.sha256(prior_raw).hexdigest(),
        "success_summary_ref": str(success_path),
        "prior_summary_ref": str(prior_path),
        "before": dict(before),
        "prior_attempt": prior_budget,
        "after": after,
        "mutation_policy": (
            "inputs_immutable_separate_derived_receipt"
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return receipt


class GeneralsHLRound4Pipeline(GeneralsHLPipeline):
    """Run one v4 coding act and formally evaluate every runnable candidate."""

    def __init__(
        self,
        config: PilotConfig,
        learning_config: Round4LearningConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        prior_attempt_run_dir: Path | None = None,
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
        self.prior_attempt_run_dir = (
            Path(prior_attempt_run_dir)
            if prior_attempt_run_dir is not None
            else None
        )

    @classmethod
    def from_paths(
        cls,
        agentbench_root: Path,
        manifest_path: Path,
        learning_manifest_path: Path,
        replay_skill_path: Path,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        prior_attempt_run_dir: Path | None = None,
    ) -> "GeneralsHLRound4Pipeline":
        config = load_pilot_config(manifest_path)
        learning = load_round4_learning_config(
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
        learning = build_round4_learning_cases(
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
                "evaluation_cases": [
                    asdict(case) for case in formal
                ],
            },
        )
        self._write_json(
            run_dir / "benchmark" / "v4-learning-spec.json",
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
                f"learn4-high-s{match.seed}"
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
                        state_id=f"{replay_id}-d{index}",
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

    def run(self) -> Round4PipelineResult:
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
                "round": 4,
                "lineage_mode": "pilot_v3_to_v4_replay_guided",
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = None
        evo_score_1 = None
        evo_score_2 = None
        evo_score_3 = None
        evo_score_4 = None
        gain_4 = None
        global_act_count = 0
        round_act_count = 0
        status = "failed"
        error = None
        runnable = False
        lineage: Round4ParentLineage | None = None
        v3_learning: GeneralsEvaluation | None = None
        v4_validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        prompt: PromptBuildResult | None = None
        diagnostics: dict[str, object] | None = None
        act = None
        prior_attempt: PriorRound4Attempt | None = None
        try:
            lineage = load_round4_parent(
                self.parent_run_dir,
                self.expected_parent_hash,
            )
            raw_score = lineage.raw_score
            evo_score_1 = lineage.evo_score_1
            evo_score_2 = lineage.evo_score_2
            evo_score_3 = lineage.evo_score_3
            global_act_count = lineage.global_act_count
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
            imported = import_parent_v3(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            self._write_benchmark_artifacts(run_dir)
            learning_cases = build_round4_learning_cases(
                self.config,
                self.learning_config,
            )
            formal_cases = build_evaluation_spec(self.config).cases
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v3",
                version="v3",
                manifest_hash=imported.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=global_act_count,
            )
            run.write(
                "version",
                version="v3",
                status="imported",
                manifest_hash=imported.content_hash,
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
                learning_id=self.learning_config.learning_id,
                engine_hash=self.assets.engine_hash,
                replay_skill_sha256=self.replay_skill.sha256,
            )
            evaluator = self.evaluator or self._production_evaluator(run_dir)
            v3_learning = evaluator.evaluate(
                workspace,
                "v3",
                "learning",
                run,
                cases=learning_cases,
            )
            valid_v3 = sum(item.valid for item in v3_learning.results)
            run.write(
                "learning_validation",
                version="v3",
                learning_id=self.learning_config.learning_id,
                status=v3_learning.status,
                case_count=len(learning_cases),
                valid_case_count=valid_v3,
            )
            if (
                v3_learning.status != "complete"
                or len(v3_learning.results) != len(learning_cases)
                or valid_v3 != len(learning_cases)
            ):
                raise _Round4Abort(
                    "learning_failed",
                    "v3 strongest-human feedback suite is incomplete",
                )
            tiers = {
                case.case_id: str(case.metadata["tier"])
                for case in learning_cases
            }
            (
                evidence,
                selections,
                probes,
                old_dense,
            ) = self._critical_feedback(v3_learning, tiers)
            if not probes:
                raise _Round4Abort(
                    "learning_probe_failed",
                    "v3 feedback produced no selected policy decisions",
                )
            for selection in selections:
                run.write(
                    "critical_window_selection",
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
            old_actions = {
                probe.state_id: probe.old_action
                for probe in probes
            }
            old_classes = summarize_decision_classes(
                probes,
                old_actions,
            )
            run.write(
                "decision_class_summary",
                version="v3",
                learning_id=self.learning_config.learning_id,
                total=old_classes.total,
                counts=dict(old_classes.counts),
                rates=dict(old_classes.rates),
            )
            prompt = build_round4_prompt(
                self.config.benchmark_id,
                (workspace / "STRATEGY.md").read_text(
                    encoding="utf-8"
                ),
                (workspace / "EXPERIENCE.md").read_text(
                    encoding="utf-8"
                ),
                self.rules_path.read_text(encoding="utf-8"),
                self.replay_skill.text,
                self.replay_skill.sha256,
                evidence,
                old_classes,
            )
            if prompt.truncated:
                raise _Round4Abort(
                    "prompt_incomplete",
                    "v4 prompt omitted a declared learning episode",
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
                provider_dir / "codex-act-v4.prompt.md"
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
                provider_dir / "codex-act-v4.prompt.json",
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
                        provider_dir / "codex-act-v4.raw.jsonl"
                    ),
                    "stderr_output_path": str(
                        provider_dir / "codex-act-v4.stderr.log"
                    ),
                    "max_artifact_bytes": (
                        self.config.limits.max_artifact_bytes
                    ),
                },
                workspace_root=str(workspace),
                version_before="v3",
                previous_manifest=imported,
            )
            round_act_count = 1
            global_act_count = lineage.global_act_count + 1
            v4 = self._write_version(
                run_dir,
                "v4",
                workspace,
                previous=imported,
            )
            self.snapshotter.write_unified_patch(
                run_dir / "versions" / "v3" / "source",
                run_dir / "versions" / "v4" / "source",
                run_dir / "versions" / "v3-to-v4.patch",
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
                for path in v4.changed_files
            )
            experience_present = (
                workspace / "EXPERIENCE.md"
            ).is_file()
            provider_completed = act.status == "completed"
            if provider_completed and protected_unchanged:
                tests_ok, test_output = self._baseline_tests(workspace)
            else:
                tests_ok = False
                test_output = (
                    "strategy tests skipped because provider did not "
                    "complete or protected files changed\n"
                )
            (run_dir / "versions" / "v4" / "tests.log").write_text(
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
                version="v4",
                version_before="v3",
                status=(
                    "available"
                    if runnable
                    else (
                        "provider_failed"
                        if not provider_completed
                        else "invalid"
                    )
                ),
                manifest_hash=v4.content_hash,
                changed_files=v4.changed_files,
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
                action_disagreement = None
                new_classes: DecisionClassSummary = old_classes
                try:
                    new_actions = self._probe_actions(
                        workspace,
                        probes,
                    )
                    behavior = measure_action_disagreement(
                        probes,
                        new_actions,
                    )
                    action_disagreement = behavior.mean
                    new_classes = summarize_decision_classes(
                        probes,
                        new_actions,
                    )
                    run.write(
                        "behavior_change",
                        version_before="v3",
                        version_after="v4",
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
                        version="v4",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                run.write(
                    "decision_class_summary",
                    version="v4",
                    learning_id=self.learning_config.learning_id,
                    total=new_classes.total,
                    counts=dict(new_classes.counts),
                    rates=dict(new_classes.rates),
                )

                validation_error = None
                try:
                    v4_validation = evaluator.evaluate(
                        workspace,
                        "v4",
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
                valid_v4 = (
                    sum(
                        item.valid
                        for item in v4_validation.results
                    )
                    if v4_validation is not None
                    else 0
                )
                new_dense = (
                    self._dense_summaries(v4_validation)
                    if v4_validation is not None
                    else ()
                )
                dense_deltas = paired_dense_deltas(
                    old_dense,
                    new_dense,
                )
                validation_complete = (
                    v4_validation is not None
                    and v4_validation.status == "complete"
                    and len(v4_validation.results)
                    == len(learning_cases)
                    and valid_v4 == len(learning_cases)
                )
                run.write(
                    "learning_validation",
                    version="v4",
                    phase="validation",
                    learning_id=self.learning_config.learning_id,
                    status=(
                        v4_validation.status
                        if v4_validation is not None
                        else "error"
                    ),
                    case_count=len(learning_cases),
                    valid_case_count=valid_v4,
                    dense_deltas=dense_deltas,
                    error=validation_error,
                )
                diagnostics = {
                    "action_disagreement": action_disagreement,
                    "decision_count": len(probes),
                    "validation_complete": validation_complete,
                    "valid_validation_case_count": valid_v4,
                    "validation_case_count": len(learning_cases),
                    "dense_deltas": dense_deltas,
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
                    "v4",
                    "evaluation",
                    run,
                    cases=formal_cases,
                )
                evo_score_4 = formal.score
                gain_4 = (
                    evo_score_4 - raw_score
                    if evo_score_4 is not None
                    and raw_score is not None
                    else None
                )
                run.write(
                    "evaluation",
                    phase="evolved_4",
                    version="v4",
                    status=formal.status,
                    score=formal.score,
                    gain=gain_4,
                    global_coding_agent_act=global_act_count,
                    per_tier=dict(formal.per_tier),
                    seat_gap=formal.seat_gap,
                )
                run.record_act_evaluation(
                    act.act_id,
                    {
                        "benchmark_id": self.config.benchmark_id,
                        "version": "v4",
                        "score": formal.score,
                        "gain": gain_4,
                        "global_coding_agent_act": global_act_count,
                    },
                )
                status = (
                    "complete"
                    if formal.status == "complete"
                    else "incomplete"
                )
        except _Round4Abort as exc:
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
            score_history = (
                [*lineage.prior_score_history, evo_score_4]
                if lineage is not None
                else []
            )
            current_budget = run.budget_snapshot()
            prior_cumulative = _combine_learning_budgets(
                lineage.learning_budget if lineage else {},
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
                    "evo_score": evo_score_4,
                    "evo_score_1": evo_score_1,
                    "evo_score_2": evo_score_2,
                    "evo_score_3": evo_score_3,
                    "evo_score_4": evo_score_4,
                    "gain": gain_4,
                    "gain_1": (
                        evo_score_1 - raw_score
                        if evo_score_1 is not None
                        and raw_score is not None
                        else None
                    ),
                    "gain_2": (
                        evo_score_2 - raw_score
                        if evo_score_2 is not None
                        and raw_score is not None
                        else None
                    ),
                    "gain_3": (
                        evo_score_3 - raw_score
                        if evo_score_3 is not None
                        and raw_score is not None
                        else None
                    ),
                    "gain_4": gain_4,
                    "benchmark_score": evo_score_4,
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
                    "parent_run_id": (
                        lineage.parent_run_id if lineage else None
                    ),
                    "parent_version": "v3" if lineage else None,
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
                        (v3_learning, v4_validation)
                    ),
                }
            )
        return Round4PipelineResult(
            run_dir=run_dir,
            raw_score=raw_score,
            evo_score_1=evo_score_1,
            evo_score_2=evo_score_2,
            evo_score_3=evo_score_3,
            evo_score_4=evo_score_4,
            gain_4=gain_4,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
            status=status,
            runnable=runnable,
        )
