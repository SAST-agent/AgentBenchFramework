"""CLI surface for the official-engine Generals HL pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentbench_frame.tracking.provider import ProviderInvocation
from agentbench_frame.tracking.providers import CodexProvider

from .assets import (
    AssetValidationError,
    load_pilot_config,
    require_valid_assets,
    resolve_assets,
)
from .pipeline import GeneralsHLPipeline


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generals", help="Official-engine Generals HL benchmark"
    )
    commands = parser.add_subparsers(dest="generals_command", required=True)
    for name, help_text in (
        ("prepare", "Validate frozen assets and population"),
        ("eval", "Evaluate the frozen v0 matrix"),
        ("iterate", "Run v0 → learning → one Codex act → v1"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--agentbench-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--data-dir", type=Path, required=True)
        if name == "eval":
            command.add_argument("--version", choices=("v0",), default="v0")
        if name == "iterate":
            command.add_argument("--codex-executable", default="codex")
            command.add_argument("--provider-timeout", type=float, default=1800)
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
        if args.generals_command == "eval":
            provider = _RawOnlyProvider()
        else:
            provider = CodexProvider(
                executable=args.codex_executable,
                timeout_s=args.provider_timeout,
                sandbox="workspace-write",
            )
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
    except AssetValidationError as exc:
        print(f"Generals asset validation failed: {exc}")
        return 2
