"""Immutable second-act Generals HL orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
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
from .pipeline import GeneralsHLPipeline, _load_match_result
from .process import build_baseline_process, build_calibration_process
from .prompt import _reject_leaks, build_round2_prompt
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


_LEARNING_AUC_AXES = {
    "episode": "learning_episodes",
    "env_step": "learning_env_steps",
    "token": "learning_total_tokens",
    "time": "learning_time_s",
}


def _combine_learning_budgets(
    *budgets: dict[str, object],
) -> dict[str, float | int | None]:
    keys = {
        "learning_coding_agent_acts",
        "learning_episodes",
        "learning_env_steps",
        "learning_game_agent_decision_steps",
        "learning_primitive_commands",
        "learning_prompt_tokens",
        "learning_completion_tokens",
        "learning_total_tokens",
        "learning_time_s",
    }
    combined: dict[str, float | int | None] = {}
    for key in keys:
        values = [budget.get(key) for budget in budgets]
        if any(value is None for value in values):
            combined[key] = None
        else:
            total = sum(float(value) for value in values)
            combined[key] = (
                int(total)
                if key != "learning_time_s"
                else total
            )
    return combined


def _learning_auc_fields(
    raw_score: float | None,
    evo_score_1: float | None,
    evo_score_2: float | None,
    parent_budget: dict[str, object],
    round2_budget: dict[str, object],
    *,
    interrupted: bool = False,
) -> tuple[dict[str, float | int | None], dict[str, float | None]]:
    cumulative = _combine_learning_budgets(parent_budget, round2_budget)
    aucs: dict[str, float | None] = {}
    for output, key in _LEARNING_AUC_AXES.items():
        if interrupted:
            # A failed coding-agent act has no score. The global curve breaks
            # there, so the full-axis AUC is unavailable (not a partial AUC).
            aucs[f"AUC_{output}"] = None
            continue
        before = parent_budget.get(key)
        after = cumulative.get(key)
        if (
            raw_score is None
            or evo_score_1 is None
            or evo_score_2 is None
            or before is None
            or after is None
        ):
            aucs[f"AUC_{output}"] = None
            continue
        x1 = float(before)
        x2 = float(after)
        aucs[f"AUC_{output}"] = (
            0.5 * (raw_score + evo_score_1) * x1
            + 0.5 * (evo_score_1 + evo_score_2) * (x2 - x1)
        )
    return cumulative, aucs


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
        selected_score = complete_scores[selected_mode]
        if (
            calibration_config.target_min
            <= selected_score
            <= calibration_config.target_max
        ):
            _write_calibration_selection(
                Path(selection_output),
                calibration_config,
                selected_mode,
                calibration_source_hash(calibration_source),
                complete_scores,
            )
            status = "complete"
        else:
            status = "calibration_failed"
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

    def _acquire_heldout_receipt(self, run: Run) -> Path:
        """Atomically consume a frozen calibration held-out suite once."""
        parent = self.parent_run_dir.resolve()
        study_root = next(
            (
                ancestor.parent
                for ancestor in parent.parents
                if ancestor.name == "runs"
            ),
            parent.parent,
        )
        receipt_dir = study_root / "calibration-heldout"
        receipt = receipt_dir / (
            f"{self.calibration_config.benchmark_id}-"
            f"{self.calibration_selection.source_hash}.json"
        )
        if receipt.exists():
            raise ValueError("held-out calibration was already opened")
        run_roots = {
            study_root / "runs",
            self.data_dir.resolve() / "runs",
        }
        for runs_root in run_roots:
            if not runs_root.is_dir():
                continue
            for events_path in runs_root.rglob("events.jsonl"):
                events = []
                try:
                    lines = events_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                except OSError:
                    continue
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                matching_spec = any(
                    event.get("event_type") == "calibration_spec"
                    and event.get("source_hash")
                    == self.calibration_selection.source_hash
                    for event in events
                )
                opened = any(
                    event.get("event_type") == "calibration_result"
                    and event.get("split") == "heldout"
                    for event in events
                )
                if matching_spec and opened:
                    raise ValueError("held-out calibration was already opened")
        receipt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark_id": self.calibration_config.benchmark_id,
            "source_hash": self.calibration_selection.source_hash,
            "selected_mode": self.calibration_selection.selected_mode,
            "run_id": run.run_id,
        }
        try:
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except FileExistsError as exc:
            raise ValueError(
                "held-out calibration was already opened"
            ) from exc
        return receipt

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _write_benchmark_artifacts(self, run_dir: Path) -> None:
        benchmark_dir = run_dir / "benchmark"
        self._write_json(
            benchmark_dir / "formal-spec.json",
            {
                "benchmark_id": self.config.benchmark_id,
                "engine_hash": self.assets.engine_hash,
                "evaluation_cases": [
                    asdict(case)
                    for case in build_evaluation_spec(self.config).cases
                ],
                "learning_cases": [
                    asdict(case)
                    for case in build_round2_learning_cases(self.config)
                ],
            },
        )
        self._write_json(
            benchmark_dir / "calibration-spec.json",
            {
                "benchmark_id": self.calibration_config.benchmark_id,
                "development_seeds": list(
                    self.calibration_config.development_seeds
                ),
                "heldout_seeds": list(
                    self.calibration_config.heldout_seeds
                ),
                "target_range": [
                    self.calibration_config.target_min,
                    self.calibration_config.target_max,
                ],
                "target_midpoint": self.calibration_config.target_midpoint,
            },
        )
        self._write_json(
            benchmark_dir / "calibration-opponent-manifest.json",
            {
                "benchmark_id": self.calibration_selection.benchmark_id,
                "selected_mode": self.calibration_selection.selected_mode,
                "source_hash": self.calibration_selection.source_hash,
                "selection_rule": self.calibration_selection.selection_rule,
                "development_scores": dict(
                    self.calibration_selection.development_scores
                ),
            },
        )

    def _validate_recovery_source(
        self,
        source_summary: dict[str, object],
        source_events: list[dict[str, object]],
        source_prompt: Path,
        source_prompt_manifest: Path,
        lineage,
        failed_act: dict[str, object],
    ) -> dict[str, float | int | None]:
        config = source_summary.get("config")
        if not isinstance(config, dict):
            raise ValueError("recovery source config is missing")
        if source_summary.get("benchmark_id") != self.config.benchmark_id:
            raise ValueError("recovery source benchmark ID does not match")
        if (
            source_summary.get("calibration_benchmark_id")
            != self.calibration_config.benchmark_id
        ):
            raise ValueError(
                "recovery source calibration benchmark ID does not match"
            )
        if config.get("engine_hash") != self.assets.engine_hash:
            raise ValueError("recovery source engine hash does not match")
        if source_summary.get("parent_run_id") != lineage.parent_run_id:
            raise ValueError("recovery source parent run does not match")
        if (
            source_summary.get("raw_score") != lineage.raw_score
            or source_summary.get("evo_score_1") != lineage.evo_score_1
        ):
            raise ValueError("recovery source lineage scores do not match")
        if (
            failed_act.get("snapshot_content_hash")
            != lineage.v1_manifest.content_hash
        ):
            raise ValueError("recovery source v1 snapshot hash does not match")
        if any(
            failed_act.get(key) is not None
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        ):
            raise ValueError(
                "recovery source pre-thread token usage must be unknown"
            )
        source_run_id = source_summary.get("run_id")
        if not source_run_id or any(
            event.get("run_id") != source_run_id
            for event in source_events
        ):
            raise ValueError("recovery source event lineage does not match")

        benchmark_specs = [
            event
            for event in source_events
            if event.get("event_type") == "benchmark_spec"
        ]
        if len(benchmark_specs) != 1:
            raise ValueError("recovery source benchmark spec is ambiguous")
        benchmark_spec = benchmark_specs[0]
        if (
            benchmark_spec.get("benchmark_id") != self.config.benchmark_id
            or benchmark_spec.get("engine_hash") != self.assets.engine_hash
            or benchmark_spec.get("evaluation_cases")
            != [
                asdict(case)
                for case in build_evaluation_spec(self.config).cases
            ]
            or benchmark_spec.get("learning_cases")
            != [
                asdict(case)
                for case in build_round2_learning_cases(self.config)
            ]
        ):
            raise ValueError("recovery source benchmark spec does not match")

        calibration_specs = [
            event
            for event in source_events
            if event.get("event_type") == "calibration_spec"
            and event.get("split") == "heldout"
        ]
        if len(calibration_specs) != 1:
            raise ValueError("recovery source held-out spec is ambiguous")
        calibration_spec = calibration_specs[0]
        if (
            calibration_spec.get("calibration_benchmark_id")
            != self.calibration_config.benchmark_id
            or calibration_spec.get("selected_mode")
            != self.calibration_selection.selected_mode
            or calibration_spec.get("source_hash")
            != self.calibration_selection.source_hash
            or calibration_spec.get("seeds")
            != list(self.calibration_config.heldout_seeds)
            or calibration_spec.get("target_range")
            != [
                self.calibration_config.target_min,
                self.calibration_config.target_max,
            ]
        ):
            raise ValueError("recovery source calibration spec does not match")
        heldout_results = [
            event
            for event in source_events
            if event.get("event_type") == "calibration_result"
            and event.get("split") == "heldout"
        ]
        if len(heldout_results) != 1:
            raise ValueError("recovery source held-out result is ambiguous")
        heldout = heldout_results[0]
        if (
            heldout.get("calibration_benchmark_id")
            != self.calibration_config.benchmark_id
            or heldout.get("mode")
            != self.calibration_selection.selected_mode
            or heldout.get("status") != "complete"
            or heldout.get("score")
            != source_summary.get("calibration_score")
            or heldout.get("in_target_range") is not True
        ):
            raise ValueError("recovery source held-out result does not match")

        try:
            prompt_manifest = json.loads(
                source_prompt_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"recovery source prompt manifest is invalid: {exc}"
            ) from exc
        prompt_text = source_prompt.read_text(encoding="utf-8")
        prompt_bytes = len(prompt_text.encode("utf-8"))
        if (
            prompt_manifest.get("prompt_bytes") != prompt_bytes
            or prompt_bytes > 262_144
        ):
            raise ValueError("recovery source prompt size does not match")
        embedded = prompt_manifest.get("prompt")
        if embedded is not None and embedded != prompt_text:
            raise ValueError("recovery source prompt manifest does not match")
        actual_sha256 = hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest()
        declared_sha256 = prompt_manifest.get("prompt_sha256")
        if declared_sha256 is not None:
            if declared_sha256 != actual_sha256:
                raise ValueError(
                    "recovery source prompt SHA-256 does not match"
                )
        elif embedded is None:
            raise ValueError(
                "recovery source prompt lacks a SHA-256 binding"
            )
        _reject_leaks(prompt_text)

        parent_acts = lineage.learning_budget.get(
            "learning_coding_agent_acts"
        )
        if not isinstance(parent_acts, int):
            raise ValueError("parent coding-agent act count is unavailable")
        expected_global_acts = parent_acts + 1
        if source_summary.get("act_count") != expected_global_acts:
            raise ValueError("recovery source act count does not match events")
        if source_summary.get("round_act_count") != 1:
            raise ValueError(
                "recovery source round act count does not match events"
            )

        budget_events = [
            event
            for event in source_events
            if event.get("event_type") == "budget"
            and event.get("phase") == "learning"
        ]
        if not budget_events:
            raise ValueError("recovery source learning budget events are missing")
        last_budget = budget_events[-1]
        keys = (
            "learning_episodes",
            "learning_env_steps",
            "learning_game_agent_decision_steps",
            "learning_primitive_commands",
        )
        derived_budget: dict[str, float | int | None] = {
            key: last_budget.get(key) for key in keys
        }
        derived_budget["learning_coding_agent_acts"] = 1
        for output in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            derived_budget[f"learning_{output}"] = failed_act.get(output)
        base_time = last_budget.get("learning_time_s")
        act_time = failed_act.get("elapsed_time_s")
        derived_budget["learning_time_s"] = (
            float(base_time) + float(act_time)
            if base_time is not None and act_time is not None
            else None
        )
        declared_budget = source_summary.get("budget")
        if not isinstance(declared_budget, dict) or any(
            declared_budget.get(key) != value
            for key, value in derived_budget.items()
        ):
            raise ValueError(
                "recovery source learning budget does not match events"
            )
        return derived_budget

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
            self._write_benchmark_artifacts(run_dir)
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

            receipt = self._acquire_heldout_receipt(run)
            run.write(
                "calibration_spec",
                calibration_benchmark_id=self.calibration_config.benchmark_id,
                split="heldout_receipt",
                source_hash=self.calibration_selection.source_hash,
                receipt_ref=str(receipt),
            )
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
                    global_coding_agent_act=1,
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
                canonical_prompt = provider_dir / "codex-act-2.prompt.md"
                canonical_prompt.write_text(
                    prompt.prompt, encoding="utf-8"
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
                    provider_dir / "codex-act-2.prompt.json",
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
                            provider_dir / "codex-act-2.raw.jsonl"
                        ),
                        "stderr_output_path": str(
                            provider_dir / "codex-act-2.stderr.log"
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
                    episode_means = []
                    for episode_index, (case_id, probes) in enumerate(
                        episode_probes, start=1
                    ):
                        if not probes:
                            continue
                        behavior = measure_action_disagreement(
                            probes, self._probe_actions(workspace, probes)
                        )
                        all_trace.extend(behavior.trace)
                        episode_means.append(behavior.mean)
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
                        episode_balanced_action_disagreement=(
                            sum(episode_means) / len(episode_means)
                            if episode_means
                            else None
                        ),
                        decision_balanced_action_disagreement=(
                            sum(all_trace) / len(all_trace)
                            if all_trace
                            else None
                        ),
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
            round2_budget = run.budget_snapshot()
            cumulative_learning_budget, learning_aucs = (
                _learning_auc_fields(
                    raw_score,
                    evo_score_1,
                    evo_score_2,
                    parent_budget,
                    round2_budget,
                )
            )
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
                "round2_learning_budget": {
                    key: value
                    for key, value in round2_budget.items()
                    if key.startswith("learning_")
                },
                "cumulative_learning_budget": cumulative_learning_budget,
                **learning_aucs,
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

    def recover_provider_init_failure(
        self, failed_run_dir: Path
    ) -> Round2PipelineResult:
        """Retry a pre-thread infrastructure failure without reopening held-out."""
        source_dir = Path(failed_run_dir).resolve()
        source_summary = json.loads(
            (source_dir / "summary.json").read_text(encoding="utf-8")
        )
        if source_summary.get("status") != "provider_failed":
            raise ValueError("recovery source must have provider_failed status")
        source_events = [
            json.loads(line)
            for line in (source_dir / "events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        failed_acts = [
            event
            for event in source_events
            if event.get("event_type") == "coding_agent_act"
        ]
        if len(failed_acts) != 1 or failed_acts[0].get("status") != "failed":
            raise ValueError("recovery requires exactly one failed provider act")
        metadata = failed_acts[0].get("provider_metadata") or {}
        if (
            metadata.get("thread_id") is not None
            or metadata.get("raw_event_count") not in {None, 0}
            or failed_acts[0].get("changed_files")
        ):
            raise ValueError("provider failure occurred after a Codex thread started")
        if source_summary.get("evo_score_2") is not None:
            raise ValueError("recovery source already contains a v2 score")
        if source_summary.get("calibration_in_target_range") is not True:
            raise ValueError("recovery source lacks a passed held-out calibration")
        source_prompt = source_dir / "provider" / "prompt.txt"
        source_prompt_manifest = (
            source_dir / "provider" / "prompt-manifest.json"
        )
        if not source_prompt.is_file() or not source_prompt_manifest.is_file():
            raise ValueError("recovery source prompt artifacts are missing")

        lineage = load_parent_lineage(
            self.parent_run_dir, self.expected_parent_hash
        )
        source_round2_budget = self._validate_recovery_source(
            source_summary,
            source_events,
            source_prompt,
            source_prompt_manifest,
            lineage,
            failed_acts[0],
        )
        parent_act_count = int(
            lineage.learning_budget["learning_coding_agent_acts"]
        )
        global_act_count = parent_act_count + len(failed_acts) + 1
        round_act_count = len(failed_acts) + 1
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "calibration_benchmark_id": (
                    self.calibration_config.benchmark_id
                ),
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 2,
                "recovery_from_run_id": source_summary.get("run_id"),
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = source_summary.get("raw_score")
        evo_score_1 = source_summary.get("evo_score_1")
        evo_score_2 = gain_2 = None
        calibration_score = source_summary.get("calibration_score")
        status = "failed"
        error = None
        v2_formal = None
        try:
            imported = import_parent_v1(lineage, run_dir, self.snapshotter)
            workspace = run_dir / "workspace"
            self._write_benchmark_artifacts(run_dir)
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v1",
                version="v1",
                manifest_hash=imported.content_hash,
            )
            run.write(
                "provider_retry",
                recovery_from_run_id=source_summary.get("run_id"),
                failed_act_id=failed_acts[0].get("act_id"),
                failure_stage="pre_thread_initialization",
                reused_prompt=True,
                reused_learning_evidence=True,
                reused_heldout_calibration=True,
            )
            run.write(
                "calibration_result",
                calibration_benchmark_id=self.calibration_config.benchmark_id,
                version="v0",
                split="heldout",
                status=source_summary.get("calibration_status"),
                score=calibration_score,
                wins=source_summary.get("calibration_wins"),
                losses=source_summary.get("calibration_losses"),
                draws=source_summary.get("calibration_draws"),
                per_seat=source_summary.get("calibration_per_seat"),
                in_target_range=True,
                reused_from_run_id=source_summary.get("run_id"),
            )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                evaluation_cases=[
                    asdict(case)
                    for case in build_evaluation_spec(self.config).cases
                ],
                engine_hash=self.assets.engine_hash,
                recovery=True,
            )
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                source_prompt, provider_dir / "codex-act-2.prompt.md"
            )
            shutil.copy2(source_prompt, provider_dir / "prompt.txt")
            shutil.copy2(
                source_prompt_manifest,
                provider_dir / "codex-act-2.prompt.json",
            )
            shutil.copy2(
                source_prompt_manifest,
                provider_dir / "prompt-manifest.json",
            )
            prompt_text = source_prompt.read_text(encoding="utf-8")
            controller = run.create_coding_agent_controller(
                self.provider,
                snapshotter=self.snapshotter,
                budget_phase="learning",
            )
            act = controller.run_act(
                {
                    "prompt": prompt_text,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(
                        provider_dir / "codex-act-2.raw.jsonl"
                    ),
                    "stderr_output_path": str(
                        provider_dir / "codex-act-2.stderr.log"
                    ),
                    "max_artifact_bytes": self.config.limits.max_artifact_bytes,
                },
                workspace_root=str(workspace),
                version_before="v1",
                previous_manifest=imported,
            )
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
                learning_root = source_dir / "matches" / "v1"
                episode_probes = []
                if learning_root.is_dir():
                    for artifact in sorted(learning_root.iterdir()):
                        if not (
                            artifact.name.startswith("learn2-")
                            or artifact.name.startswith(
                                "calibration-development-"
                            )
                        ):
                            continue
                        match = _load_match_result(artifact)
                        replay = build_learning_replay(
                            match, "baseline", "recovery"
                        )
                        probes = tuple(
                            ProbeState(
                                decision.state_id,
                                {
                                    **decision.state,
                                    "my_seat": decision.seat,
                                },
                                decision.action,
                            )
                            for decision in replay.decisions
                        )
                        episode_probes.append((match.case_id, probes))
                all_trace = []
                episode_means = []
                for episode_index, (case_id, probes) in enumerate(
                    episode_probes, start=1
                ):
                    if not probes:
                        continue
                    behavior = measure_action_disagreement(
                        probes, self._probe_actions(workspace, probes)
                    )
                    all_trace.extend(behavior.trace)
                    episode_means.append(behavior.mean)
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
                        source_run_id=source_summary.get("run_id"),
                    )
                run.write(
                    "behavior_change",
                    version_before="v1",
                    version_after="v2",
                    episode_count=len(episode_probes),
                    decision_count=len(all_trace),
                    episode_balanced_action_disagreement=(
                        sum(episode_means) / len(episode_means)
                        if episode_means
                        else None
                    ),
                    decision_balanced_action_disagreement=(
                        sum(all_trace) / len(all_trace)
                        if all_trace
                        else None
                    ),
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
                if self.evaluator is not None:
                    evaluator = self.evaluator
                else:
                    evaluator, _unused = self._production_evaluators(run_dir)
                v2_formal = evaluator.evaluate(
                    workspace, "v2", "evaluation", run
                )
                evo_score_2 = v2_formal.score
                gain_2 = (
                    evo_score_2 - raw_score
                    if evo_score_2 is not None and raw_score is not None
                    else None
                )
                v1_states = []
                if learning_root.is_dir():
                    for artifact in sorted(learning_root.iterdir()):
                        if not artifact.name.startswith("eval-"):
                            continue
                        match = _load_match_result(artifact)
                        v1_states.extend(
                            turn.state_id_before
                            for turn in match.turns
                            if turn.player == match.evaluated_seat
                        )
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
                        context={
                            "domain": "frozen_evaluation_matrix",
                            "v1_source_run_id": source_summary.get("run_id"),
                        },
                    )
                run.write(
                    "evaluation",
                    phase="evolved_2",
                    version="v2",
                    status=v2_formal.status,
                    score=v2_formal.score,
                    gain=gain_2,
                    global_coding_agent_act=global_act_count,
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
                        "global_coding_agent_act": global_act_count,
                    },
                )
                status = (
                    "complete"
                    if v2_formal.status == "complete"
                    else "incomplete"
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", error=error, recovery=True)
        finally:
            run.writer.flush()
            self._write_json(
                run_dir / "quality.json",
                inspect_event_file(run_dir / "events.jsonl").to_dict(),
            )
            auc = None
            benchmark_results = [
                item
                for item in source_summary.get("benchmark_results", [])
                if item.get("version") == "v1"
            ]
            benchmark_results.extend(
                self._result_rows((v2_formal,))
            )
            parent_learning_budget = dict(
                source_summary.get("parent_learning_budget") or {}
            )
            recovery_budget = {
                key: value
                for key, value in run.budget_snapshot().items()
                if key.startswith("learning_")
            }
            round2_learning_budget = _combine_learning_budgets(
                source_round2_budget, recovery_budget
            )
            cumulative_learning_budget, learning_aucs = (
                _learning_auc_fields(
                    raw_score,
                    evo_score_1,
                    evo_score_2,
                    parent_learning_budget,
                    round2_learning_budget,
                    interrupted=True,
                )
            )
            run.finish(
                {
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
                    "parent_run_id": lineage.parent_run_id,
                    "parent_version": "v1",
                    "recovery_from_run_id": source_summary.get("run_id"),
                    "reused_heldout_calibration": True,
                    "source_learning_budget": source_round2_budget,
                    "parent_learning_budget": parent_learning_budget,
                    "round2_learning_budget": round2_learning_budget,
                    "cumulative_learning_budget": (
                        cumulative_learning_budget
                    ),
                    **learning_aucs,
                    "calibration_benchmark_id": (
                        self.calibration_config.benchmark_id
                    ),
                    "calibration_status": source_summary.get(
                        "calibration_status"
                    ),
                    "calibration_score": calibration_score,
                    "calibration_wins": source_summary.get(
                        "calibration_wins"
                    ),
                    "calibration_losses": source_summary.get(
                        "calibration_losses"
                    ),
                    "calibration_draws": source_summary.get(
                        "calibration_draws"
                    ),
                    "calibration_per_seat": source_summary.get(
                        "calibration_per_seat"
                    ),
                    "calibration_target_range": source_summary.get(
                        "calibration_target_range"
                    ),
                    "calibration_in_target_range": True,
                    "error": error,
                    "benchmark_results": benchmark_results,
                }
            )
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
