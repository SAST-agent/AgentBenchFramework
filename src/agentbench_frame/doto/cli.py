"""Single command-line surface for the DOTO harness."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


class DotoNotImplementedError(RuntimeError):
    pass


def _pending(_args: argparse.Namespace) -> int:
    raise DotoNotImplementedError("this DOTO command is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.doto",
        description="DOTO build, match, replay, strict KL, and LLM loop harness",
    )
    subcommands = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("build", "compile a native playerAI.cpp"),
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
