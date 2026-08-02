"""Single command-line surface for the DOTO harness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .build import build_candidate


class DotoNotImplementedError(RuntimeError):
    pass


def _pending(_args: argparse.Namespace) -> int:
    raise DotoNotImplementedError("this DOTO command is not implemented yet")


def _build(args: argparse.Namespace) -> int:
    result = build_candidate(args.player_ai, args.output_dir)
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    return 0 if result.exit_code == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.doto",
        description="DOTO build, match, replay, strict KL, and LLM loop harness",
    )
    subcommands = parser.add_subparsers(dest="cmd", required=True)
    build = subcommands.add_parser("build", help="compile a native playerAI.cpp")
    build.add_argument("--player-ai", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(func=_build)
    for name, help_text in (
        ("match", "run one official-protocol match"),
        ("replay", "parse an official replay ZIP"),
        ("ig", "compare native policies on one trace"),
        ("loop", "run the replay-driven LLM iteration loop"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.set_defaults(func=_pending)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
