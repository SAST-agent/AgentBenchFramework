"""One-act Generals HL vertical-loop orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

from .assets import (
    load_pilot_config,
    prepare_opponents,
    require_valid_assets,
    resolve_assets,
)
from .evaluator import GeneralsEvaluation, GeneralsEvaluator, build_evaluation_spec, build_learning_cases
from .match import GeneralsMatchRunner
from .measurement import ProbeState, measure_action_disagreement, measure_occupancy_shift
from .models import AssetLayout, MatchCase, PilotConfig
from .process import build_baseline_process
from .prompt import build_codex_prompt
from .replay import LearningReplay, build_learning_replay


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    raw_score: float | None
    evo_score: float | None
    gain: float | None
    act_count: int
    status: str


class GeneralsHLPipeline:
    def __init__(
        self,
        config: PilotConfig,
        assets: AssetLayout,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        replay_guide_path: Path | None = None,
    ):
        self.config = config
        self.assets = assets
        self.data_dir = Path(data_dir)
        self.provider = provider
        self.evaluator = evaluator
        benchmark_dir = assets.root / "backend_sources/corpus/28_generals/benchmark"
        self.rules_path = rules_path or benchmark_dir / "rules.md"
        self.replay_guide_path = replay_guide_path or benchmark_dir / "replay.md"
        self.snapshotter = LocalWorkspaceSnapshotter()

    @classmethod
    def from_paths(
        cls,
        agentbench_root: Path,
        manifest_path: Path,
        data_dir: Path,
        provider: ProviderAdapter,
    ) -> "GeneralsHLPipeline":
        config = load_pilot_config(manifest_path)
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        return cls(config, assets, data_dir, provider)

    def _production_evaluator(self, run_dir: Path) -> GeneralsEvaluator:
        prepared = prepare_opponents(
            self.assets, run_dir / "prepared-opponents", Path(sys.executable)
        )
        opponent_specs = {item.agent_id: item for item in prepared}
        sdk_root = opponent_specs["popular-rank16-xiaoaojianghu-v1"].cwd
        runner = GeneralsMatchRunner(self.assets, self.config.benchmark_id)

        def execute(case, workspace, version, artifact_dir):
            baseline = build_baseline_process(
                workspace,
                self.assets.engine_root,
                Path(sys.executable),
                sdk_root=sdk_root,
            )
            opponent = opponent_specs[case.opponent]
            players = (baseline, opponent) if case.first_player == 0 else (opponent, baseline)
            match_case = MatchCase(
                case_id=case.case_id,
                seed=case.seed,
                evaluated_seat=case.first_player,
                opponent_id=case.opponent,
                opponent_tier=case.metadata["tier"],
            )
            return runner.run(match_case, players, artifact_dir)

        return GeneralsEvaluator(self.config, execute)

    def _write_version(
        self,
        run_dir: Path,
        version: str,
        workspace: Path,
        previous=None,
    ):
        version_dir = run_dir / "versions" / version
        manifest = self.snapshotter.capture(workspace, previous=previous)
        source = version_dir / "source"
        source.mkdir(parents=True, exist_ok=True)
        for relative in manifest.files:
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(workspace / relative, target)
        self.snapshotter.write_manifest(manifest, version_dir / "manifest.json")
        return manifest

    def _baseline_tests(self, workspace: Path) -> tuple[bool, str]:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=workspace,
            env={**os.environ, "PYTHONPATH": str(workspace)},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr)[-100_000:]
        return completed.returncode == 0, output

    def _probe_actions(
        self, workspace: Path, probes: tuple[ProbeState, ...]
    ) -> dict[str, tuple[tuple[int, ...], ...]]:
        code = (
            "import json,sys\n"
            "from strategy import choose_actions\n"
            "for line in sys.stdin:\n"
            " d=json.loads(line); a=choose_actions(d['round'],d['seat'],d['state']);"
            " print(json.dumps({'id':d['id'],'actions':a}),flush=True)\n"
        )
        payload = "".join(
            json.dumps(
                {
                    "id": probe.state_id,
                    "round": int(probe.state.get("round", 1)),
                    "seat": int(probe.state.get("my_seat", 0)),
                    "state": probe.state,
                },
                ensure_ascii=False,
            )
            + "\n"
            for probe in probes
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workspace,
            env={**os.environ, "PYTHONPATH": str(workspace)},
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"strategy probe failed: {completed.stderr[-2000:]}")
        result = {}
        for line in completed.stdout.splitlines():
            item = json.loads(line)
            result[item["id"]] = tuple(tuple(int(v) for v in command) for command in item["actions"])
        return result

    @staticmethod
    def _replays(learning: GeneralsEvaluation) -> tuple[LearningReplay, ...]:
        items = []
        for match in learning.matches:
            tier = next(
                (
                    result.metadata.get("tier", "unknown")
                    for result in learning.results
                    if result.case_id == match.case_id
                ),
                "unknown",
            )
            replay = build_learning_replay(match, "baseline", str(tier))
            # Preserve all twelve episodes but bound prompt size deterministically.
            items.append(replace(replay, decisions=replay.decisions[:2]))
        return tuple(items)

    def run(self) -> PipelineResult:
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = evo_score = gain = None
        status = "failed"
        act_count = 0
        raw_eval = evolved = None
        error = None
        try:
            require_valid_assets(self.assets)
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                evaluation_cases=[asdict(case) for case in build_evaluation_spec(self.config).cases],
                learning_cases=[asdict(case) for case in build_learning_cases(self.config)],
                engine_hash=self.assets.engine_hash,
            )
            workspace = run_dir / "workspace"
            shutil.copytree(self.assets.baseline_root, workspace)
            v0 = self._write_version(run_dir, "v0", workspace)
            run.write("version", version="v0", status="available", manifest_hash=v0.content_hash)

            evaluator = self.evaluator or self._production_evaluator(run_dir)
            raw_eval = evaluator.evaluate(workspace, "v0", "evaluation", run)
            raw_score = raw_eval.score
            run.write(
                "evaluation",
                phase="raw",
                version="v0",
                status=raw_eval.status,
                score=raw_eval.score,
                per_tier=dict(raw_eval.per_tier),
                seat_gap=raw_eval.seat_gap,
            )
            learning = evaluator.evaluate(
                workspace,
                "v0",
                "learning",
                run,
                cases=build_learning_cases(self.config),
            )
            replays = self._replays(learning)
            prompt = build_codex_prompt(
                self.config.benchmark_id,
                (workspace / "STRATEGY.md").read_text(encoding="utf-8"),
                self.rules_path.read_text(encoding="utf-8"),
                self.replay_guide_path.read_text(encoding="utf-8"),
                replays,
            )
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            (provider_dir / "prompt.txt").write_bytes(
                prompt.encode("utf-8")[: self.config.limits.max_artifact_bytes]
            )
            controller = run.create_coding_agent_controller(
                self.provider, snapshotter=self.snapshotter, budget_phase="learning"
            )
            act = controller.run_act(
                {
                    "prompt": prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(provider_dir / "codex.raw.jsonl"),
                    "stderr_output_path": str(provider_dir / "codex.stderr.log"),
                    "max_artifact_bytes": self.config.limits.max_artifact_bytes,
                },
                workspace_root=str(workspace),
                version_before="v0",
                previous_manifest=v0,
            )
            act_count = 1
            v1 = self._write_version(run_dir, "v1", workspace, previous=v0)
            self.snapshotter.write_unified_patch(
                run_dir / "versions" / "v0" / "source",
                run_dir / "versions" / "v1" / "source",
                run_dir / "versions" / "v0-to-v1.patch",
            )
            allowed = all(
                path in {"strategy.py", "STRATEGY.md"} or path.startswith("tests/")
                for path in v1.changed_files
            )
            tests_ok, test_output = self._baseline_tests(workspace)
            (run_dir / "versions" / "v1" / "tests.log").write_text(
                test_output, encoding="utf-8"
            )
            version_status = (
                "available"
                if act.status == "completed" and allowed and tests_ok
                else "invalid"
            )
            run.write(
                "version",
                version="v1",
                status=version_status,
                manifest_hash=v1.content_hash,
                changed_files=v1.changed_files,
                provider_status=act.status,
                protected_files_unchanged=allowed,
                tests_passed=tests_ok,
            )
            if version_status == "available":
                probes = tuple(
                    ProbeState(
                        decision.state_id,
                        decision.state,
                        decision.action,
                    )
                    for replay in replays
                    for decision in replay.decisions
                )
                if probes:
                    new_actions = self._probe_actions(workspace, probes)
                    behavior = measure_action_disagreement(probes, new_actions)
                    old_states = tuple(probe.state_id for probe in probes)
                    new_states = tuple(
                        turn.state_id_before
                        for match in learning.matches
                        for turn in match.turns
                        if turn.player == match.evaluated_seat
                    )[: len(old_states)]
                    occupancy = (
                        measure_occupancy_shift(new_states, old_states)
                        if new_states
                        else None
                    )
                    run.write(
                        "behavior_change",
                        version_before="v0",
                        version_after="v1",
                        action_disagreement_trace=list(behavior.trace),
                        action_disagreement=behavior.mean,
                        policy_kl=behavior.policy_kl,
                        policy_kl_status=behavior.policy_kl_status,
                        occupancy_shift=occupancy,
                    )
                evolved = evaluator.evaluate(workspace, "v1", "evaluation", run)
                evo_score = evolved.score
                gain = (
                    evo_score - raw_score
                    if evo_score is not None and raw_score is not None
                    else None
                )
                run.write(
                    "evaluation",
                    phase="evolved",
                    version="v1",
                    status=evolved.status,
                    score=evolved.score,
                    gain=gain,
                    per_tier=dict(evolved.per_tier),
                    seat_gap=evolved.seat_gap,
                )
                status = "complete" if evolved.status == "complete" else "incomplete"
            else:
                status = "provider_failed" if act.status != "completed" else "invalid_version"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", error=error)
            status = "failed"
        finally:
            run.writer.flush()
            quality = inspect_event_file(str(run_dir / "events.jsonl")).to_dict()
            (run_dir / "quality.json").write_text(
                json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            auc = (
                (raw_score + evo_score) / 2
                if raw_score is not None and evo_score is not None
                else None
            )
            summary = {
                "status": status,
                "benchmark_id": self.config.benchmark_id,
                "raw_score": raw_score,
                "evo_score": evo_score,
                "gain": gain,
                "AUC_coding_agent_act": auc,
                "act_count": act_count,
                "error": error,
                "benchmark_results": [
                    {
                        "case_id": item.case_id,
                        "outcome": item.outcome,
                        "valid": item.valid,
                        "error": item.error,
                        **item.metadata,
                    }
                    for evaluation in (raw_eval, evolved)
                    if evaluation is not None
                    for item in evaluation.results
                ],
            }
            run.finish(summary)
        return PipelineResult(run_dir, raw_score, evo_score, gain, act_count, status)
