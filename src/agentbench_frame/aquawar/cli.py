"""Package-local command line interface for AquaWar evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Type

from agentbench_frame.aquawar.evaluator import AquaWarEvaluator, Opponent


def parse_opponent(value: str) -> Opponent:
    if "=" not in value:
        raise ValueError("opponent must use NAME=COMMAND")
    name, command = value.split("=", 1)
    if not name or not command:
        raise ValueError("opponent must use NAME=COMMAND")
    return Opponent(name=name, command=command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.aquawar",
        description="Run seat-balanced AquaWar subprocess evaluations.",
    )
    parser.add_argument("--logic", required=True, help="official logic command")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate", required=True, help="candidate AI command")
    parser.add_argument(
        "--opponent",
        action="append",
        type=parse_opponent,
        required=True,
        metavar="NAME=COMMAND",
    )
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--seats", choices=("both", "0", "1"), default="both")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator_class: Type[AquaWarEvaluator] = AquaWarEvaluator,
) -> int:
    args = build_parser().parse_args(argv)
    evaluator = evaluator_class(
        logic_command=args.logic,
        candidate_name=args.candidate_name,
        candidate_command=args.candidate,
        opponents=args.opponent,
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
