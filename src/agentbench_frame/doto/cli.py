"""Single command-line surface for the DOTO harness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .build import build_candidate
from .ig import compare_policies_on_trace, write_ig_artifacts
from .match import run_match
from .replay import iter_replay, summarize_replay


class DotoNotImplementedError(RuntimeError):
    pass


def _pending(_args: argparse.Namespace) -> int:
    raise DotoNotImplementedError("this DOTO command is not implemented yet")


def _build(args: argparse.Namespace) -> int:
    result = build_candidate(args.player_ai, args.output_dir)
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    return 0 if result.exit_code == 0 else 1


def _match(args: argparse.Namespace) -> int:
    result = run_match(
        args.agent0,
        args.agent1,
        seed=args.seed,
        output_dir=args.output_dir,
        tag=args.tag,
        frame_timeout=args.frame_timeout,
        server_timeout=args.server_timeout,
        server_dir=args.server_dir,
        test_only=args.test_only,
    )
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    return 0


def _replay(args: argparse.Namespace) -> int:
    summary = summarize_replay(args.path)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as stream:
            for frame in iter_replay(args.path):
                for event in frame.events:
                    row = {"frame": frame.frame, "type": event[0], "args": event[1:]}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary.to_json(), ensure_ascii=False, indent=2))
    return 0


def _ig(args: argparse.Namespace) -> int:
    row = compare_policies_on_trace(
        args.trace, args.old, args.new,
        faction=args.faction,
        iteration=args.iteration,
        old_version=args.old_version,
        new_version=args.new_version,
        timeout=args.timeout,
    )
    paths = write_ig_artifacts(row, args.output_dir)
    result = {
        "episode_id": row["episode_id"],
        "unchanged_ratio": row["unchanged_ratio"],
        "infinite_ratio": row["infinite_ratio"],
        "missing_ratio": row["missing_ratio"],
        "finite_kl_mean": row["finite_kl_mean"],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


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
    match = subcommands.add_parser("match", help="run one official-protocol match")
    match.add_argument("--agent0", type=Path, required=True)
    match.add_argument("--agent1", type=Path, required=True)
    match.add_argument("--seed", type=int, default=11)
    match.add_argument("--output-dir", type=Path, required=True)
    match.add_argument("--tag", required=True)
    match.add_argument("--frame-timeout", type=float, default=1.0)
    match.add_argument("--server-timeout", type=float, default=330.0)
    match.add_argument("--server-dir", type=Path)
    match.add_argument("--test-only", action="store_true")
    match.set_defaults(func=_match)
    replay = subcommands.add_parser("replay", help="parse an official replay ZIP")
    replay.add_argument("--path", type=Path, required=True)
    replay.add_argument("--jsonl", type=Path)
    replay.set_defaults(func=_replay)
    ig = subcommands.add_parser("ig", help="compare native policies on one trace")
    ig.add_argument("--trace", type=Path, required=True)
    ig.add_argument("--old", type=Path, required=True)
    ig.add_argument("--new", type=Path, required=True)
    ig.add_argument("--faction", type=int, choices=(0, 1), required=True)
    ig.add_argument("--iteration", type=int, required=True)
    ig.add_argument("--old-version", required=True)
    ig.add_argument("--new-version", required=True)
    ig.add_argument("--output-dir", type=Path, required=True)
    ig.add_argument("--timeout", type=float, default=1.0)
    ig.set_defaults(func=_ig)
    loop = subcommands.add_parser("loop", help="run the replay-driven LLM iteration loop")
    loop.set_defaults(func=_pending)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
