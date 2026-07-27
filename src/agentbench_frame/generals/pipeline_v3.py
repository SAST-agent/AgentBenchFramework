"""Gated v2-to-v3 Generals heuristic-learning rescue orchestration."""

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
    load_round3_learning_config,
    require_valid_assets,
    resolve_assets,
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
    build_round3_learning_cases,
)
from .lineage_v3 import (
    Round3ParentLineage,
    import_parent_v2,
    load_round3_parent,
)
from .measurement import (
    BehaviorGateResult,
    DecisionClassSummary,
    ProbeState,
    evaluate_behavior_gate,
    measure_action_disagreement,
    summarize_decision_classes,
)
from .models import AssetLayout, PilotConfig, Round3LearningConfig
from .pipeline import GeneralsHLPipeline
from .prompt import build_round3_prompt
from .replay import build_compact_evidence, build_learning_replay


@dataclass(frozen=True)
class Round3PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score_1: float | None
    evo_score_2: float | None
    evo_score_3: float | None
    gain_3: float | None
    global_act_count: int
    round_act_count: int
    status: str
    gate_passed: bool


class _Round3Abort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


_DENSE_TERMINALS = {
    "terminal_territory_share": "territory_share",
    "terminal_army_margin": "army_margin",
    "terminal_coin_share": "coin_share",
    "terminal_net_main_pressure": "net_main_pressure",
}


def paired_dense_deltas(
    before: Iterable[DenseEpisodeSummary],
    after: Iterable[DenseEpisodeSummary],
) -> dict[str, float | None]:
    """Return paired v3-minus-v2 means on the predeclared dense metrics."""
    old = {item.case_id: item for item in before}
    new = {item.case_id: item for item in after}
    if not old or set(old) != set(new):
        return {
            "completed_rounds_survived": None,
            **{name: None for name in _DENSE_TERMINALS},
        }

    values: dict[str, list[float]] = {
        "completed_rounds_survived": [],
        **{name: [] for name in _DENSE_TERMINALS},
    }
    unavailable = {name: False for name in _DENSE_TERMINALS}
    for case_id in sorted(old):
        previous = old[case_id]
        current = new[case_id]
        values["completed_rounds_survived"].append(
            float(
                current.completed_rounds_survived
                - previous.completed_rounds_survived
            )
        )
        for output, field in _DENSE_TERMINALS.items():
            old_value = getattr(previous, field).terminal
            new_value = getattr(current, field).terminal
            if old_value is None or new_value is None:
                unavailable[output] = True
                continue
            values[output].append(float(new_value) - float(old_value))
    return {
        name: (
            sum(items) / len(items)
            if len(items) == len(old)
            and not unavailable.get(name, False)
            else None
        )
        for name, items in values.items()
    }


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
        combined[key] = (
            value
            if key.endswith("_time_s")
            else int(value)
        )
    return combined


class GeneralsHLRound3Pipeline(GeneralsHLPipeline):
    """Run one architecture-level act and gate formal evaluation."""

    def __init__(
        self,
        config: PilotConfig,
        learning_config: Round3LearningConfig,
        assets: AssetLayout,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        replay_guide_path: Path | None = None,
    ):
        super().__init__(
            config,
            assets,
            data_dir,
            provider,
            evaluator=evaluator,
            rules_path=rules_path,
            replay_guide_path=replay_guide_path,
        )
        self.learning_config = learning_config
        self.parent_run_dir = Path(parent_run_dir)
        self.expected_parent_hash = expected_parent_hash

    @classmethod
    def from_paths(
        cls,
        agentbench_root: Path,
        manifest_path: Path,
        learning_manifest_path: Path,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
    ) -> "GeneralsHLRound3Pipeline":
        config = load_pilot_config(manifest_path)
        learning = load_round3_learning_config(
            learning_manifest_path,
            config,
        )
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        return cls(
            config=config,
            learning_config=learning,
            assets=assets,
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

    @staticmethod
    def _compact_evidence(
        evaluation: GeneralsEvaluation,
        tier_by_case: Mapping[str, str],
    ) -> tuple[
        tuple[object, ...],
        tuple[ProbeState, ...],
        tuple[DenseEpisodeSummary, ...],
    ]:
        evidence = []
        probes = []
        dense = []
        for match in evaluation.matches:
            replay_id = (
                f"learn3-high-s{match.seed}"
                f"-p{match.evaluated_seat}"
            )
            raw_replay = build_learning_replay(
                match,
                "baseline",
                tier_by_case[match.case_id],
            )
            replay = replace(
                raw_replay,
                replay_id=replay_id,
                decisions=tuple(
                    replace(
                        item,
                        state_id=f"{replay_id}-d{index}",
                    )
                    for index, item in enumerate(
                        raw_replay.decisions,
                        start=1,
                    )
                ),
            )
            trace = build_dense_trace(match)
            summary = summarize_dense_trace(match, trace)
            evidence.append(
                build_compact_evidence(replay, summary, trace)
            )
            dense.append(summary)
            probes.extend(
                ProbeState(
                    item.state_id,
                    {**item.state, "my_seat": item.seat},
                    item.action,
                )
                for item in raw_replay.decisions
            )
        return tuple(evidence), tuple(probes), tuple(dense)

    def _write_benchmark_artifacts(self, run_dir: Path) -> None:
        formal = build_evaluation_spec(self.config).cases
        learning = build_round3_learning_cases(
            self.config,
            self.learning_config,
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
            run_dir / "benchmark" / "v3-learning-spec.json",
            {
                "learning_id": self.learning_config.learning_id,
                "opponent_id": self.learning_config.opponent_id,
                "seeds": list(self.learning_config.seeds),
                "seats": list(self.learning_config.seats),
                "cases": [asdict(case) for case in learning],
            },
        )

    def run(self) -> Round3PipelineResult:
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
                "round": 3,
                "lineage_mode": "pilot_v2_to_v3_rescue",
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = evo_score_1 = evo_score_2 = evo_score_3 = None
        gain_3 = None
        global_act_count = 0
        round_act_count = 0
        status = "failed"
        error = None
        lineage: Round3ParentLineage | None = None
        gate: BehaviorGateResult | None = None
        dense_deltas: dict[str, float | None] = {}
        formal: GeneralsEvaluation | None = None
        v2_learning: GeneralsEvaluation | None = None
        v3_learning: GeneralsEvaluation | None = None
        act = None
        try:
            lineage = load_round3_parent(
                self.parent_run_dir,
                self.expected_parent_hash,
            )
            raw_score = lineage.raw_score
            evo_score_1 = lineage.evo_score_1
            evo_score_2 = lineage.evo_score_2
            global_act_count = lineage.global_act_count
            imported = import_parent_v2(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            self._write_benchmark_artifacts(run_dir)
            learning_cases = build_round3_learning_cases(
                self.config,
                self.learning_config,
            )
            formal_cases = build_evaluation_spec(self.config).cases
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v2",
                version="v2",
                manifest_hash=imported.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=global_act_count,
            )
            run.write(
                "version",
                version="v2",
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
            )
            evaluator = self.evaluator or self._production_evaluator(run_dir)
            v2_learning = evaluator.evaluate(
                workspace,
                "v2",
                "learning",
                run,
                cases=learning_cases,
            )
            valid_v2 = sum(
                item.valid for item in v2_learning.results
            )
            run.write(
                "learning_validation",
                version="v2",
                learning_id=self.learning_config.learning_id,
                status=v2_learning.status,
                case_count=len(learning_cases),
                valid_case_count=valid_v2,
            )
            if (
                v2_learning.status != "complete"
                or len(v2_learning.results) != len(learning_cases)
                or valid_v2 != len(learning_cases)
            ):
                raise _Round3Abort(
                    "learning_failed",
                    "v2 strongest-human learning suite is incomplete",
                )
            tiers = {
                case.case_id: str(case.metadata["tier"])
                for case in learning_cases
            }
            evidence, probes, old_dense = self._compact_evidence(
                v2_learning,
                tiers,
            )
            if not probes:
                raise _Round3Abort(
                    "learning_probe_failed",
                    "v2 learning suite produced no policy decisions",
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
                version="v2",
                learning_id=self.learning_config.learning_id,
                total=old_classes.total,
                counts=dict(old_classes.counts),
                rates=dict(old_classes.rates),
            )
            prompt = build_round3_prompt(
                self.config.benchmark_id,
                (workspace / "STRATEGY.md").read_text(encoding="utf-8"),
                self.rules_path.read_text(encoding="utf-8"),
                self.replay_guide_path.read_text(encoding="utf-8"),
                evidence,
                old_classes,
            )
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            canonical_prompt = provider_dir / "codex-act-v3.prompt.md"
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
            self._write_json(
                provider_dir / "codex-act-v3.prompt.json",
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
                        provider_dir / "codex-act-v3.raw.jsonl"
                    ),
                    "stderr_output_path": str(
                        provider_dir / "codex-act-v3.stderr.log"
                    ),
                    "max_artifact_bytes": (
                        self.config.limits.max_artifact_bytes
                    ),
                },
                workspace_root=str(workspace),
                version_before="v2",
                previous_manifest=imported,
            )
            round_act_count = 1
            global_act_count = lineage.global_act_count + 1
            v3 = self._write_version(
                run_dir,
                "v3",
                workspace,
                previous=imported,
            )
            self.snapshotter.write_unified_patch(
                run_dir / "versions" / "v2" / "source",
                run_dir / "versions" / "v3" / "source",
                run_dir / "versions" / "v2-to-v3.patch",
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
                for path in v3.changed_files
            )
            experience_present = (
                workspace / "EXPERIENCE.md"
            ).is_file()
            tests_ok, test_output = self._baseline_tests(workspace)
            (run_dir / "versions" / "v3" / "tests.log").write_text(
                test_output,
                encoding="utf-8",
            )
            provider_completed = act.status == "completed"
            candidate_valid = (
                provider_completed
                and protected_unchanged
                and tests_ok
                and experience_present
            )

            new_actions = old_actions
            action_disagreement = None
            new_classes: DecisionClassSummary = old_classes
            if provider_completed and tests_ok:
                try:
                    new_actions = self._probe_actions(workspace, probes)
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
                        version_before="v2",
                        version_after="v3",
                        learning_id=self.learning_config.learning_id,
                        decision_count=len(probes),
                        action_disagreement_trace=list(behavior.trace),
                        action_disagreement=behavior.mean,
                        policy_kl=behavior.policy_kl,
                        policy_kl_status=behavior.policy_kl_status,
                        occupancy_shift=None,
                    )
                except Exception as exc:
                    run.write(
                        "behavior_measurement_error",
                        version="v3",
                        error=f"{type(exc).__name__}: {exc}",
                    )
            run.write(
                "decision_class_summary",
                version="v3",
                learning_id=self.learning_config.learning_id,
                total=new_classes.total,
                counts=dict(new_classes.counts),
                rates=dict(new_classes.rates),
            )

            new_dense: tuple[DenseEpisodeSummary, ...] = ()
            valid_v3 = 0
            if candidate_valid:
                v3_learning = evaluator.evaluate(
                    workspace,
                    "v3",
                    "learning",
                    run,
                    cases=learning_cases,
                )
                valid_v3 = sum(
                    item.valid for item in v3_learning.results
                )
                _, _, new_dense = self._compact_evidence(
                    v3_learning,
                    tiers,
                )
            dense_deltas = paired_dense_deltas(
                old_dense,
                new_dense,
            )
            run.write(
                "learning_validation",
                version="v3",
                learning_id=self.learning_config.learning_id,
                status=(
                    v3_learning.status
                    if v3_learning is not None
                    else "not_run_invalid_candidate"
                ),
                case_count=len(learning_cases),
                valid_case_count=valid_v3,
                dense_deltas=dense_deltas,
            )
            gate = evaluate_behavior_gate(
                provider_completed=provider_completed,
                protected_files_unchanged=(
                    protected_unchanged and experience_present
                ),
                tests_passed=tests_ok,
                action_disagreement=action_disagreement,
                decision_classes=new_classes,
                validation_case_count=len(learning_cases),
                valid_validation_case_count=valid_v3,
                dense_deltas=dense_deltas,
            )
            run.write(
                "behavior_gate",
                version_before="v2",
                version_after="v3",
                passed=gate.passed,
                conditions=dict(gate.conditions),
                improved_dense_metrics=list(
                    gate.improved_dense_metrics
                ),
                dense_deltas=dense_deltas,
                action_disagreement=action_disagreement,
            )
            run.write(
                "version",
                version="v3",
                version_before="v2",
                status="available" if gate.passed else "rejected",
                manifest_hash=v3.content_hash,
                changed_files=v3.changed_files,
                provider_status=act.status,
                protected_files_unchanged=protected_unchanged,
                experience_present=experience_present,
                tests_passed=tests_ok,
                behavior_gate_passed=gate.passed,
            )
            if not gate.passed:
                status = "behavior_gate_failed"
            else:
                formal = evaluator.evaluate(
                    workspace,
                    "v3",
                    "evaluation",
                    run,
                    cases=formal_cases,
                )
                evo_score_3 = formal.score
                gain_3 = (
                    evo_score_3 - raw_score
                    if evo_score_3 is not None
                    and raw_score is not None
                    else None
                )
                run.write(
                    "evaluation",
                    phase="evolved_3",
                    version="v3",
                    status=formal.status,
                    score=formal.score,
                    gain=gain_3,
                    global_coding_agent_act=global_act_count,
                    per_tier=dict(formal.per_tier),
                    seat_gap=formal.seat_gap,
                )
                run.record_act_evaluation(
                    act.act_id,
                    {
                        "benchmark_id": self.config.benchmark_id,
                        "version": "v3",
                        "score": formal.score,
                        "gain": gain_3,
                        "global_coding_agent_act": global_act_count,
                    },
                )
                status = (
                    "complete"
                    if formal.status == "complete"
                    else "incomplete"
                )
        except _Round3Abort as exc:
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
                [
                    *lineage.prior_score_history,
                    evo_score_3,
                ]
                if lineage is not None
                else []
            )
            current_budget = run.budget_snapshot()
            cumulative_budget = _combine_learning_budgets(
                lineage.learning_budget if lineage else {},
                current_budget,
            )
            gate_payload = (
                {
                    "passed": gate.passed,
                    "conditions": dict(gate.conditions),
                    "improved_dense_metrics": list(
                        gate.improved_dense_metrics
                    ),
                    "dense_deltas": dense_deltas,
                }
                if gate is not None
                else None
            )
            run.finish(
                {
                    "status": status,
                    "benchmark_id": self.config.benchmark_id,
                    "learning_id": self.learning_config.learning_id,
                    "raw_score": raw_score,
                    "evo_score": evo_score_3,
                    "evo_score_1": evo_score_1,
                    "evo_score_2": evo_score_2,
                    "evo_score_3": evo_score_3,
                    "gain": gain_3,
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
                    "gain_3": gain_3,
                    "benchmark_score": evo_score_3,
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
                    "auc_status": "unavailable_missing_score_point",
                    "act_count": global_act_count,
                    "round_act_count": round_act_count,
                    "parent_run_id": (
                        lineage.parent_run_id if lineage else None
                    ),
                    "parent_version": "v2" if lineage else None,
                    "behavior_gate": gate_payload,
                    "error": error,
                    "cumulative_learning_budget": cumulative_budget,
                    "benchmark_results": self._result_rows((formal,)),
                    "learning_results": self._result_rows(
                        (v2_learning, v3_learning)
                    ),
                }
            )
        return Round3PipelineResult(
            run_dir=run_dir,
            raw_score=raw_score,
            evo_score_1=evo_score_1,
            evo_score_2=evo_score_2,
            evo_score_3=evo_score_3,
            gain_3=gain_3,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
            status=status,
            gate_passed=bool(gate and gate.passed),
        )
