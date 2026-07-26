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
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--agentbench-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--data-dir", type=Path, required=True)
        if name == "eval":
            command.add_argument("--version", choices=("v0",), default="v0")
        if name in {"iterate", "iterate-v2"}:
            command.add_argument("--codex-executable", default="codex")
            command.add_argument("--provider-timeout", type=float, default=1800)
        if name in {"calibrate-dev", "iterate-v2"}:
            command.add_argument(
                "--calibration-manifest", type=Path, required=True
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name == "calibrate-dev":
            command.add_argument("--selection-output", type=Path, required=True)
        if name == "iterate-v2":
            command.add_argument(
                "--calibration-selection", type=Path, required=True
            )
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
        if args.generals_command == "iterate-v2":
            result = GeneralsHLRound2Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                calibration_manifest_path=args.calibration_manifest,
                calibration_selection_path=args.calibration_selection,
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
                "gain_2": result.gain_2,
                "calibration_score": result.calibration_score,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
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
