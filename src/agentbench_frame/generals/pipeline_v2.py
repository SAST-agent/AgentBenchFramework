"""Immutable second-act Generals HL orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run

from .assets import (
    CALIBRATION_SELECTION_RULE,
    calibration_source_hash,
    load_calibration_config,
    load_calibration_selection,
    load_pilot_config,
    prepare_opponents,
    require_valid_assets,
    resolve_assets,
    resolve_calibration_source,
)
from .calibration import (
    CalibrationEvaluator,
    select_calibration_candidate,
)
from .dense import build_dense_trace, summarize_dense_trace
from .evaluator import (
    GeneralsEvaluation,
    GeneralsEvaluator,
    build_evaluation_spec,
    build_round2_learning_cases,
)
from .lineage import import_parent_v1, load_parent_lineage
from .match import GeneralsMatchRunner
from .measurement import ProbeState, measure_action_disagreement
from .models import (
    AssetLayout,
    CalibrationConfig,
    CalibrationEvaluation,
    CalibrationSelection,
    MatchCase,
    PilotConfig,
)
from .pipeline import GeneralsHLPipeline
from .process import build_baseline_process, build_calibration_process
from .prompt import build_round2_prompt
from .replay import build_compact_evidence, build_learning_replay


@dataclass(frozen=True)
class Round2PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score_1: float | None
    evo_score_2: float | None
    gain_2: float | None
    calibration_score: float | None
    global_act_count: int
    round_act_count: int
    status: str


@dataclass(frozen=True)
class CalibrationDevelopmentResult:
    run_dir: Path
    selection_path: Path
    selected_mode: str | None
    development_scores: dict[str, float | None]
    status: str


def _production_calibration_evaluator(
    config: CalibrationConfig,
    assets: AssetLayout,
    source: Path,
) -> CalibrationEvaluator:
    sdk_root = next(
        item.source for item in assets.opponents if item.tier == "low"
    )
    runner = GeneralsMatchRunner(assets, config.benchmark_id)

    def execute(case, workspace, version, artifact_dir):
        baseline = build_baseline_process(
            workspace,
            assets.engine_root,
            Path(sys.executable),
            sdk_root=sdk_root,
        )
        calibration = build_calibration_process(
            source,
            assets.engine_root,
            Path(sys.executable),
            sdk_root,
            str(case.metadata["mode"]),
        )
        players = (
            (baseline, calibration)
            if case.first_player == 0
            else (calibration, baseline)
        )
        return runner.run(
            MatchCase(
                case.case_id,
                case.seed,
                case.first_player,
                case.opponent,
                "calibration",
            ),
            players,
            artifact_dir,
        )

    return CalibrationEvaluator(config, execute)


def _write_calibration_selection(
    path: Path,
    config: CalibrationConfig,
    selected_mode: str,
    source_hash: str,
    scores: dict[str, float],
) -> None:
    lines = [
        f'benchmark_id = "{config.benchmark_id}"',
        f'selected_mode = "{selected_mode}"',
        f'source_hash = "{source_hash}"',
        f'selection_rule = "{CALIBRATION_SELECTION_RULE}"',
        "",
        "[development_scores]",
        *[f"{mode} = {scores[mode]:.12g}" for mode in config.candidate_modes],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def calibrate_development(
    config: PilotConfig,
    assets: AssetLayout,
    calibration_config: CalibrationConfig,
    calibration_source: Path,
    parent_run_dir: Path,
    expected_parent_hash: str,
    data_dir: Path,
    selection_output: Path,
    evaluator: CalibrationEvaluator | Any | None = None,
) -> CalibrationDevelopmentResult:
    """Select one weak policy using development seeds only and freeze its hash."""
    run = Run.start(
        game="28_generals",
        agent="generals-hl-calibration",
        run_type="eval",
        data_dir=str(data_dir),
        config={
            "benchmark_id": calibration_config.benchmark_id,
            "budget_phase": "calibration",
            "engine_hash": assets.engine_hash,
            "split": "development",
        },
    )
    run_dir = Path(run.run_dir)
    selected_mode = None
    scores: dict[str, float | None] = {
        mode: None for mode in calibration_config.candidate_modes
    }
    status = "failed"
    error = None
    try:
        lineage = load_parent_lineage(parent_run_dir, expected_parent_hash)
        calibration = evaluator or _production_calibration_evaluator(
            calibration_config, assets, calibration_source
        )
        run.write(
            "calibration_spec",
            calibration_benchmark_id=calibration_config.benchmark_id,
            split="development",
            candidate_modes=list(calibration_config.candidate_modes),
            seeds=list(calibration_config.development_seeds),
            selection_rule=CALIBRATION_SELECTION_RULE,
        )
        evaluations = {}
        for mode in calibration_config.candidate_modes:
            result = calibration.evaluate(
                lineage.v0_source,
                "v0",
                "development",
                mode,
                run,
                budget_phase="calibration",
            )
            evaluations[mode] = result
            scores[mode] = result.score
            run.write(
                "calibration_result",
                calibration_benchmark_id=calibration_config.benchmark_id,
                version="v0",
                split="development",
                mode=mode,
                status=result.status,
                score=result.score,
                wins=result.wins,
                losses=result.losses,
                draws=result.draws,
                per_seat=dict(result.per_seat),
            )
        selected_mode = select_calibration_candidate(
            calibration_config, evaluations
        )
        complete_scores = {
            mode: float(score)
            for mode, score in scores.items()
            if score is not None
        }
        _write_calibration_selection(
            Path(selection_output),
            calibration_config,
            selected_mode,
            calibration_source_hash(calibration_source),
            complete_scores,
        )
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.write("pipeline_error", error=error)
    finally:
        run.finish(
            {
                "status": status,
                "calibration_benchmark_id": calibration_config.benchmark_id,
                "calibration_split": "development",
                "selected_mode": selected_mode,
                "development_scores": scores,
                "selection_output": str(selection_output),
                "error": error,
            }
        )
    return CalibrationDevelopmentResult(
        run_dir,
        Path(selection_output),
        selected_mode,
        scores,
        status,
    )


class GeneralsHLRound2Pipeline(GeneralsHLPipeline):
    """Run one leak-resistant v1-to-v2 act from an immutable parent snapshot."""

    def __init__(
        self,
        config: PilotConfig,
        assets: AssetLayout,
        calibration_config: CalibrationConfig,
        calibration_selection: CalibrationSelection,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        calibration_evaluator: CalibrationEvaluator | Any | None = None,
        calibration_source: Path | None = None,
        rules_path: Path | None = None,
        replay_guide_path: Path | None = None,
    ) -> None:
        super().__init__(
            config,
            assets,
            data_dir,
            provider,
            evaluator=evaluator,
            rules_path=rules_path,
            replay_guide_path=replay_guide_path,
        )
        self.calibration_config = calibration_config
        self.calibration_selection = calibration_selection
        self.parent_run_dir = Path(parent_run_dir)
        self.expected_parent_hash = expected_parent_hash
        self.calibration_evaluator = calibration_evaluator
        self.calibration_source = (
            Path(calibration_source) if calibration_source is not None else None
        )

    @classmethod
    def from_paths(
        cls,
        agentbench_root: Path,
        manifest_path: Path,
        calibration_manifest_path: Path,
        calibration_selection_path: Path,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
    ) -> "GeneralsHLRound2Pipeline":
        config = load_pilot_config(manifest_path)
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        calibration = load_calibration_config(calibration_manifest_path)
        selection = load_calibration_selection(
            calibration_selection_path, calibration
        )
        source = resolve_calibration_source(
            calibration, selection, agentbench_root
        )
        return cls(
            config=config,
            assets=assets,
            calibration_config=calibration,
            calibration_selection=selection,
            parent_run_dir=parent_run_dir,
            expected_parent_hash=expected_parent_hash,
            data_dir=data_dir,
            provider=provider,
            calibration_source=source,
        )

    def _production_evaluators(
        self, run_dir: Path
    ) -> tuple[GeneralsEvaluator, CalibrationEvaluator]:
        if self.calibration_source is None:
            raise ValueError("verified calibration source is required")
        prepared = prepare_opponents(
            self.assets, run_dir / "prepared-opponents", Path(sys.executable)
        )
        opponents = {item.agent_id: item for item in prepared}
        sdk_root = opponents["popular-rank16-xiaoaojianghu-v1"].cwd
        formal_runner = GeneralsMatchRunner(
            self.assets, self.config.benchmark_id
        )
        calibration_runner = GeneralsMatchRunner(
            self.assets, self.calibration_config.benchmark_id
        )

        def execute_formal(case, workspace, version, artifact_dir):
            baseline = build_baseline_process(
                workspace,
                self.assets.engine_root,
                Path(sys.executable),
                sdk_root=sdk_root,
            )
            opponent = opponents[case.opponent]
            players = (
                (baseline, opponent)
                if case.first_player == 0
                else (opponent, baseline)
            )
            match_case = MatchCase(
                case.case_id,
                case.seed,
                case.first_player,
                case.opponent,
                str(case.metadata["tier"]),
            )
            return formal_runner.run(match_case, players, artifact_dir)

        def execute_calibration(case, workspace, version, artifact_dir):
            baseline = build_baseline_process(
                workspace,
                self.assets.engine_root,
                Path(sys.executable),
                sdk_root=sdk_root,
            )
            calibration = build_calibration_process(
                self.calibration_source,
                self.assets.engine_root,
                Path(sys.executable),
                sdk_root,
                str(case.metadata["mode"]),
            )
            players = (
                (baseline, calibration)
                if case.first_player == 0
                else (calibration, baseline)
            )
            match_case = MatchCase(
                case.case_id,
                case.seed,
                case.first_player,
                case.opponent,
                "calibration",
            )
            return calibration_runner.run(match_case, players, artifact_dir)

        return (
            GeneralsEvaluator(self.config, execute_formal),
            CalibrationEvaluator(self.calibration_config, execute_calibration),
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
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
    def _compact_evidence(evaluation, tiers):
        compact = []
        probes = []
        for match in evaluation.matches:
            tier = tiers[match.case_id]
            replay = replace(
                build_learning_replay(match, "baseline", tier),
                replay_id=match.case_id,
            )
            trace = build_dense_trace(match)
            summary = summarize_dense_trace(match, trace)
            compact.append(build_compact_evidence(replay, summary, trace))
            probes.append(
                (
                    match.case_id,
                    tuple(
                        ProbeState(
                            item.state_id,
                            {**item.state, "my_seat": item.seat},
                            item.action,
                        )
                        for item in replay.decisions
                    ),
                )
            )
        return compact, probes

    def run(self) -> Round2PipelineResult:
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "calibration_benchmark_id": self.calibration_config.benchmark_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 2,
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = evo_score_1 = evo_score_2 = gain_2 = None
        calibration_score = None
        round_act_count = 0
        global_act_count = 1
        status = "failed"
        error = None
        v1_formal = v2_formal = None
        heldout: CalibrationEvaluation | None = None
        lineage = None
        try:
            lineage = load_parent_lineage(
                self.parent_run_dir, self.expected_parent_hash
            )
            raw_score = lineage.raw_score
            evo_score_1 = lineage.evo_score_1
            imported = import_parent_v1(lineage, run_dir, self.snapshotter)
            workspace = run_dir / "workspace"
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v1",
                version="v1",
                manifest_hash=imported.content_hash,
            )
            run.write(
                "version",
                version="v1",
                status="imported",
                manifest_hash=imported.content_hash,
                parent_run_id=lineage.parent_run_id,
            )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                evaluation_cases=[
                    asdict(case)
                    for case in build_evaluation_spec(self.config).cases
                ],
                learning_cases=[
                    asdict(case)
                    for case in build_round2_learning_cases(self.config)
                ],
                engine_hash=self.assets.engine_hash,
            )
            run.write(
                "calibration_spec",
                calibration_benchmark_id=self.calibration_config.benchmark_id,
                selected_mode=self.calibration_selection.selected_mode,
                source_hash=self.calibration_selection.source_hash,
                split="heldout",
                seeds=list(self.calibration_config.heldout_seeds),
                target_range=[
                    self.calibration_config.target_min,
                    self.calibration_config.target_max,
                ],
            )
            if self.evaluator is None or self.calibration_evaluator is None:
                formal, calibration = self._production_evaluators(run_dir)
                evaluator = self.evaluator or formal
                calibration_evaluator = self.calibration_evaluator or calibration
            else:
                evaluator = self.evaluator
                calibration_evaluator = self.calibration_evaluator

            heldout = calibration_evaluator.evaluate(
                lineage.v0_source,
                "v0",
                "heldout",
                self.calibration_selection.selected_mode,
                run,
                budget_phase="evaluation",
            )
            calibration_score = heldout.score
            run.write(
                "calibration_result",
                calibration_benchmark_id=self.calibration_config.benchmark_id,
                version="v0",
                split="heldout",
                mode=heldout.mode,
                status=heldout.status,
                score=heldout.score,
                wins=heldout.wins,
                losses=heldout.losses,
                draws=heldout.draws,
                per_seat=dict(heldout.per_seat),
                in_target_range=heldout.in_target_range,
            )
            if (
                heldout.status != "complete"
                or heldout.score is None
                or heldout.in_target_range is not True
            ):
                status = "calibration_failed"
            else:
                v1_formal = evaluator.evaluate(
                    workspace, "v1", "evaluation", run
                )
                run.write(
                    "evaluation",
                    phase="evolved_1_reproduction",
                    version="v1",
                    status=v1_formal.status,
                    score=v1_formal.score,
                    parent_score=evo_score_1,
                    reproducible=(
                        v1_formal.score == evo_score_1
                        if v1_formal.score is not None
                        else None
                    ),
                    per_tier=dict(v1_formal.per_tier),
                    seat_gap=v1_formal.seat_gap,
                )
                learning_cases = build_round2_learning_cases(self.config)
                learning = evaluator.evaluate(
                    workspace,
                    "v1",
                    "learning",
                    run,
                    cases=learning_cases,
                )
                learning_tiers = {
                    case.case_id: str(case.metadata["tier"])
                    for case in learning_cases
                }
                compact, episode_probes = self._compact_evidence(
                    learning, learning_tiers
                )
                calibration_feedback = calibration_evaluator.evaluate(
                    workspace,
                    "v1",
                    "development",
                    self.calibration_selection.selected_mode,
                    run,
                    budget_phase="learning",
                )
                feedback_tiers = {
                    match.case_id: "calibration"
                    for match in calibration_feedback.matches
                }
                feedback_compact, feedback_probes = self._compact_evidence(
                    calibration_feedback, feedback_tiers
                )
                compact.extend(feedback_compact)
                episode_probes.extend(feedback_probes)
                prompt = build_round2_prompt(
                    self.config.benchmark_id,
                    (workspace / "STRATEGY.md").read_text(encoding="utf-8"),
                    self.rules_path.read_text(encoding="utf-8"),
                    self.replay_guide_path.read_text(encoding="utf-8"),
                    lineage.first_diff,
                    compact,
                )
                provider_dir = run_dir / "provider"
                provider_dir.mkdir(parents=True, exist_ok=True)
                (provider_dir / "prompt.txt").write_text(
                    prompt.prompt, encoding="utf-8"
                )
                self._write_json(
                    provider_dir / "prompt-manifest.json",
                    asdict(prompt),
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
                            provider_dir / "codex.raw.jsonl"
                        ),
                        "stderr_output_path": str(
                            provider_dir / "codex.stderr.log"
                        ),
                        "max_artifact_bytes": self.config.limits.max_artifact_bytes,
                    },
                    workspace_root=str(workspace),
                    version_before="v1",
                    previous_manifest=imported,
                )
                round_act_count = 1
                global_act_count = 2
                v2 = self._write_version(
                    run_dir, "v2", workspace, previous=imported
                )
                self.snapshotter.write_unified_patch(
                    run_dir / "versions" / "v1" / "source",
                    run_dir / "versions" / "v2" / "source",
                    run_dir / "versions" / "v1-to-v2.patch",
                )
                allowed = all(
                    path in {"strategy.py", "STRATEGY.md"}
                    or path.startswith("tests/")
                    for path in v2.changed_files
                )
                tests_ok, test_output = self._baseline_tests(workspace)
                (run_dir / "versions" / "v2" / "tests.log").write_text(
                    test_output, encoding="utf-8"
                )
                version_status = (
                    "available"
                    if act.status == "completed" and allowed and tests_ok
                    else "invalid"
                )
                run.write(
                    "version",
                    version="v2",
                    version_before="v1",
                    status=version_status,
                    manifest_hash=v2.content_hash,
                    changed_files=v2.changed_files,
                    provider_status=act.status,
                    protected_files_unchanged=allowed,
                    tests_passed=tests_ok,
                )
                if version_status != "available":
                    status = (
                        "provider_failed"
                        if act.status != "completed"
                        else "invalid_version"
                    )
                else:
                    all_trace = []
                    for episode_index, (case_id, probes) in enumerate(
                        episode_probes, start=1
                    ):
                        if not probes:
                            continue
                        behavior = measure_action_disagreement(
                            probes, self._probe_actions(workspace, probes)
                        )
                        all_trace.extend(behavior.trace)
                        run.write(
                            "behavior_change_episode",
                            episode=episode_index,
                            case_id=case_id,
                            version_before="v1",
                            version_after="v2",
                            action_disagreement_trace=list(behavior.trace),
                            action_disagreement=behavior.mean,
                            policy_kl=None,
                            policy_kl_status=behavior.policy_kl_status,
                        )
                    run.write(
                        "behavior_change",
                        version_before="v1",
                        version_after="v2",
                        episode_count=len(episode_probes),
                        decision_count=len(all_trace),
                        action_disagreement=(
                            sum(all_trace) / len(all_trace)
                            if all_trace
                            else None
                        ),
                        policy_kl=None,
                        policy_kl_status=(
                            "complete_macro_action_distribution_unavailable"
                        ),
                        occupancy_shift=None,
                    )
                    v2_formal = evaluator.evaluate(
                        workspace, "v2", "evaluation", run
                    )
                    evo_score_2 = v2_formal.score
                    gain_2 = (
                        evo_score_2 - raw_score
                        if evo_score_2 is not None and raw_score is not None
                        else None
                    )
                    v1_states = [
                        turn.state_id_before
                        for match in v1_formal.matches
                        for turn in match.turns
                        if turn.player == match.evaluated_seat
                    ]
                    v2_states = [
                        turn.state_id_before
                        for match in v2_formal.matches
                        for turn in match.turns
                        if turn.player == match.evaluated_seat
                    ]
                    if v1_states and v2_states:
                        run.log_occupancy(
                            episode=2,
                            version_before="v1",
                            version_after="v2",
                            state_ids=v2_states,
                            reference_state_ids=v1_states,
                            context={"domain": "frozen_evaluation_matrix"},
                        )
                    run.write(
                        "evaluation",
                        phase="evolved_2",
                        version="v2",
                        status=v2_formal.status,
                        score=v2_formal.score,
                        gain=gain_2,
                        global_coding_agent_act=2,
                        per_tier=dict(v2_formal.per_tier),
                        seat_gap=v2_formal.seat_gap,
                    )
                    run.record_act_evaluation(
                        act.act_id,
                        {
                            "benchmark_id": self.config.benchmark_id,
                            "version": "v2",
                            "score": v2_formal.score,
                            "gain": gain_2,
                            "global_coding_agent_act": 2,
                        },
                    )
                    status = (
                        "complete"
                        if v2_formal.status == "complete"
                        else "incomplete"
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", error=error)
            status = "failed"
        finally:
            run.writer.flush()
            quality = inspect_event_file(run_dir / "events.jsonl").to_dict()
            self._write_json(run_dir / "quality.json", quality)
            auc = (
                0.5 * (raw_score + evo_score_1)
                + 0.5 * (evo_score_1 + evo_score_2)
                if (
                    raw_score is not None
                    and evo_score_1 is not None
                    and evo_score_2 is not None
                )
                else None
            )
            parent_budget = dict(lineage.learning_budget) if lineage else {}
            summary = {
                "status": status,
                "benchmark_id": self.config.benchmark_id,
                "raw_score": raw_score,
                "evo_score": evo_score_2,
                "evo_score_1": evo_score_1,
                "evo_score_2": evo_score_2,
                "gain": gain_2,
                "gain_1": (
                    evo_score_1 - raw_score
                    if evo_score_1 is not None and raw_score is not None
                    else None
                ),
                "gain_2": gain_2,
                "benchmark_score": evo_score_2,
                "evaluation_status": (
                    v2_formal.status if v2_formal is not None else None
                ),
                "AUC_coding_agent_act": auc,
                "act_count": global_act_count,
                "round_act_count": round_act_count,
                "parent_run_id": lineage.parent_run_id if lineage else None,
                "parent_version": "v1" if lineage else None,
                "calibration_benchmark_id": self.calibration_config.benchmark_id,
                "calibration_status": heldout.status if heldout else None,
                "calibration_score": calibration_score,
                "calibration_wins": heldout.wins if heldout else None,
                "calibration_losses": heldout.losses if heldout else None,
                "calibration_draws": heldout.draws if heldout else None,
                "calibration_per_seat": (
                    dict(heldout.per_seat) if heldout else None
                ),
                "calibration_target_range": [
                    self.calibration_config.target_min,
                    self.calibration_config.target_max,
                ],
                "calibration_in_target_range": (
                    heldout.in_target_range if heldout else None
                ),
                "parent_learning_budget": parent_budget,
                "error": error,
                "benchmark_results": self._result_rows(
                    (v1_formal, v2_formal)
                ),
            }
            run.finish(summary)
        return Round2PipelineResult(
            run_dir,
            raw_score,
            evo_score_1,
            evo_score_2,
            gain_2,
            calibration_score,
            global_act_count,
            round_act_count,
            status,
        )


__all__ = [
    "CalibrationDevelopmentResult",
    "GeneralsHLRound2Pipeline",
    "Round2PipelineResult",
    "calibrate_development",
]
