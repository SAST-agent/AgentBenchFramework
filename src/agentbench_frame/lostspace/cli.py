"""Package-local command line interface for LostSpace evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Type

from agentbench_frame.lostspace.evaluator import (
    LostSpaceEvaluationResult,
    LostSpaceEvaluator,
    Opponent,
)

_BASELINE_SAMPLE_AI = (
    Path(__file__).resolve().parent / "baselines" / "sample_ai" / "main.py"
)


def _quote(parts: list[str]) -> str:
    """Build a shell command string that works on both POSIX and Windows."""
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(part) for part in parts)


def default_filler_command() -> str:
    """The bundled sample AI, used to pad the 4-player table when needed."""
    return _quote([sys.executable, str(_BASELINE_SAMPLE_AI)])


def parse_opponent(value: str) -> Opponent:
    if "=" not in value:
        raise ValueError("opponent must use NAME=COMMAND")
    name, command = value.split("=", 1)
    if not name or not command:
        raise ValueError("opponent must use NAME=COMMAND")
    return Opponent(name=name, command=command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.lostspace",
        description="Run seat-balanced LostSpace (4-player) subprocess evaluations.",
    )
    parser.add_argument("--logic", required=True, help="official logic command")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate", required=True, help="candidate AI command")
    parser.add_argument(
        "--opponent",
        action="append",
        type=parse_opponent,
        default=[],
        metavar="NAME=COMMAND",
        help="explicit opponent as NAME=COMMAND (repeatable)",
    )
    parser.add_argument(
        "--ladder-opponent",
        action="append",
        default=[],
        metavar="rank=NAME",
        help=(
            "add a ranked human algorithm as an opponent. NAME is a rank "
            "(rank=1), username (rank=omegafantasy) or display name "
            "(rank=最终幻想). C++ algorithms are built into a cache dir on "
            "first use. Repeatable."
        ),
    )
    parser.add_argument(
        "--filler",
        default=None,
        help="command used to pad empty table seats (default: bundled sample AI)",
    )
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--seats", choices=("all", "0", "1", "2", "3"), default="all")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_class: Type[LostSpaceEvaluator] = LostSpaceEvaluator,
) -> int:
    args = build_parser().parse_args(argv)
    from agentbench_frame.lostspace import ladder

    opponents = list(args.opponent)
    for selector in args.ladder_opponent:
        entry = ladder.resolve(selector)
        cmd, _cwd = ladder.launch_command(entry)
        opponents.append(Opponent(name=f"rank{entry.rank:02d}", command=cmd))
    if not opponents:
        build_parser().error(
            "at least one of --opponent or --ladder-opponent is required"
        )
    evaluator = evaluator_class(
        logic_command=args.logic,
        candidate_name=args.candidate_name,
        candidate_command=args.candidate,
        opponents=opponents,
        filler_command=args.filler or default_filler_command(),
        pairs=args.pairs,
        seats=args.seats,
        timeout=args.timeout,
        data_dir=args.data_dir,
        save_replays=args.save_replays,
        save_traces=args.save_traces,
    )
    result = evaluator.evaluate()
    print(result.run_dir)
    return 1 if result.error_count else 0
