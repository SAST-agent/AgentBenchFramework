"""CLI surface for the official-engine Generals HL pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentbench_frame.tracking.provider import ProviderInvocation
from agentbench_frame.tracking.providers import CodexProvider

from .assets import (
    AssetValidationError,
    load_calibration_config,
    resolve_calibration_candidate_source,
    load_pilot_config,
    require_valid_assets,
    resolve_assets,
)
from .pipeline import GeneralsHLPipeline
from .pipeline_v2 import (
    GeneralsHLRound2Pipeline,
    calibrate_development,
)
from .pipeline_v3 import GeneralsHLRound3Pipeline
from .pipeline_v4 import GeneralsHLRound4Pipeline
from .pipeline_v5 import GeneralsHLRound5Pipeline
from .pipeline_v6 import GeneralsHLRound6Pipeline


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generals", help="Official-engine Generals HL benchmark"
    )
    commands = parser.add_subparsers(dest="generals_command", required=True)
    for name, help_text in (
        ("prepare", "Validate frozen assets and population"),
        ("eval", "Evaluate the frozen v0 matrix"),
        ("iterate", "Run v0 → learning → one Codex act → v1"),
        (
            "calibrate-dev",
            "Select and freeze a weak opponent on development seeds",
        ),
        (
            "iterate-v2",
            "Run held-out calibration → v1 learning → one Codex act → v2",
        ),
        (
            "recover-v2",
            "Retry a pre-thread provider failure without reopening held-out",
        ),
        (
            "iterate-v3",
            "Run strongest-human v2 learning → one gated Codex act → v3",
        ),
        (
            "iterate-v4",
            "Run replay-guided v3 learning → one non-gated Codex act → v4",
        ),
        (
            "iterate-v5",
            "Run paired v3/v4 learning → one rollback-guided Codex act → v5",
        ),
        (
            "iterate-v6",
            "Run strongest-human v5 learning → one macro-planner act → v6",
        ),
        (
            "recover-v6",
            "Recover an audited v6 prompt, provider, or evaluation failure",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--agentbench-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--data-dir", type=Path, required=True)
        if name == "eval":
            command.add_argument("--version", choices=("v0",), default="v0")
        if name in {
            "iterate",
            "iterate-v2",
            "recover-v2",
            "iterate-v3",
            "iterate-v4",
            "iterate-v5",
            "iterate-v6",
            "recover-v6",
        }:
            explicit_provider = name in {"iterate-v6", "recover-v6"}
            command.add_argument(
                "--codex-executable",
                default="codex",
                required=explicit_provider,
            )
            command.add_argument(
                "--provider-timeout",
                type=float,
                default=1800,
                required=explicit_provider,
            )
        if name in {"calibrate-dev", "iterate-v2", "recover-v2"}:
            command.add_argument(
                "--calibration-manifest", type=Path, required=True
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v3", "iterate-v4", "iterate-v5"}:
            command.add_argument(
                "--learning-manifest",
                type=Path,
                required=True,
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v6", "recover-v6"}:
            command.add_argument(
                "--learning-manifest",
                type=Path,
                required=True,
            )
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v4", "iterate-v5"}:
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument(
                "--prior-attempt-run",
                type=Path,
                help=(
                    "Optional finalized prompt_incomplete v4 run whose "
                    "pre-act learning budget must remain cumulative"
                ),
            )
        if name == "iterate-v5":
            command.add_argument(
                "--expected-rollback-hash",
                required=True,
            )
            command.add_argument(
                "--campaign-budget-receipt",
                type=Path,
                required=True,
            )
        if name == "calibrate-dev":
            command.add_argument("--selection-output", type=Path, required=True)
        if name in {"iterate-v2", "recover-v2"}:
            command.add_argument(
                "--calibration-selection", type=Path, required=True
            )
        if name == "recover-v2":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "recover-v6":
            command.add_argument("--failed-run", type=Path, required=True)
    return parser


def _assets(args):
    config = load_pilot_config(args.manifest)
    layout = resolve_assets(config, args.agentbench_root)
    require_valid_assets(layout)
    return config, layout


class _RawOnlyProvider:
    provider_name = "codex"

    def invoke(self, context):
        return ProviderInvocation(
            status="failed",
            error="v0-only evaluation intentionally stops before a coding-agent act",
        )


def handle(args) -> int:
    try:
        config, layout = _assets(args)
        if args.generals_command == "prepare":
            print(
                f"{config.benchmark_id}: {len(config.opponents)} opponents, "
                f"18 evaluation cases, 12 learning cases; engine "
                f"{layout.engine_hash[:12]}"
            )
            return 0
        if args.generals_command == "calibrate-dev":
            calibration = load_calibration_config(
                args.calibration_manifest
            )
            calibration_source = resolve_calibration_candidate_source(
                calibration, args.agentbench_root
            )
            result = calibrate_development(
                config=config,
                assets=layout,
                calibration_config=calibration,
                calibration_source=calibration_source,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                selection_output=args.selection_output,
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "selection_path": str(result.selection_path),
                "selected_mode": result.selected_mode,
                "development_scores": result.development_scores,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "eval":
            provider = _RawOnlyProvider()
        else:
            provider = CodexProvider(
                executable=args.codex_executable,
                timeout_s=args.provider_timeout,
                sandbox="workspace-write",
            )
        if args.generals_command in {"iterate-v2", "recover-v2"}:
            pipeline = GeneralsHLRound2Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                calibration_manifest_path=args.calibration_manifest,
                calibration_selection_path=args.calibration_selection,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v2"
                else pipeline.recover_provider_init_failure(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "gain_2": result.gain_2,
                "calibration_score": result.calibration_score,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v3":
            result = GeneralsHLRound3Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "gain_3": result.gain_3,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "behavior_gate_passed": result.gate_passed,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v4":
            result = GeneralsHLRound4Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
                prior_attempt_run_dir=args.prior_attempt_run,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "gain_4": result.gain_4,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v5":
            result = GeneralsHLRound5Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                expected_rollback_hash=args.expected_rollback_hash,
                campaign_budget_receipt=args.campaign_budget_receipt,
                data_dir=args.data_dir,
                provider=provider,
                prior_attempt_run_dir=args.prior_attempt_run,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "evo_score_5": result.evo_score_5,
                "gain_5": result.gain_5,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {"iterate-v6", "recover-v6"}:
            pipeline = GeneralsHLRound6Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v6"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "evo_score_5": result.evo_score_5,
                "evo_score_6": result.evo_score_6,
                "gain_6": result.gain_6,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        result = GeneralsHLPipeline(
            config=config,
            assets=layout,
            data_dir=args.data_dir,
            provider=provider,
        ).run()
        print(result.run_dir)
        if args.generals_command == "eval":
            return 0 if result.raw_score is not None else 1
        return 0 if result.status == "complete" else 1
    except (AssetValidationError, ValueError) as exc:
        print(f"Generals asset validation failed: {exc}")
        return 2
