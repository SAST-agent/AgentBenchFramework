#!/usr/bin/env python3
"""Fail-closed fake-only 24_miracle research preflight."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.iteration_protocol import (
    HumanAuthoredContentRequired,
    IterationBlockedError,
    LearningConfig,
    MatchConfig,
    default_bootstrap_template,
    preflight_learning,
    preflight_match,
    preflight_replay,
)


def build_research_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only fake-only 24_miracle iteration preflight stage. "
            "No Judge, opponent, provider, match, or session is started."
        )
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    def add_match_inputs(stage: argparse.ArgumentParser) -> None:
        stage.add_argument("--champion-manifest", type=Path, required=True)
        stage.add_argument("--human-replay-skill", type=Path, required=True)
        stage.add_argument("--match-plan", type=Path, required=True)

    plan = stages.add_parser("plan", help="validate the independently approved match plan")
    add_match_inputs(plan)
    replay = stages.add_parser(
        "replay", help="validate captured replay evidence and its independent approval"
    )
    add_match_inputs(replay)
    replay.add_argument("--replay-evidence-manifest", type=Path, required=True)
    learning = stages.add_parser(
        "learning", help="validate approved train evidence before an Agent may read it"
    )
    add_match_inputs(learning)
    learning.add_argument("--replay-evidence-manifest", type=Path, required=True)
    return parser


def _research_main(argv=None) -> int:
    args = build_research_parser().parse_args(argv)
    match_config = MatchConfig(
        game="24_miracle",
        bootstrap=default_bootstrap_template(),
        champion_manifest_path=args.champion_manifest,
        human_replay_skill_path=args.human_replay_skill,
        match_plan_path=args.match_plan,
    )
    if args.stage == "plan":
        context = preflight_match(match_config)
    else:
        learning_config = LearningConfig(
            match=match_config,
            replay_evidence_manifest_path=args.replay_evidence_manifest,
        )
        context = (
            preflight_replay(learning_config)
            if args.stage == "replay"
            else preflight_learning(learning_config)
        )
    print(
        json.dumps(
            {
                "status": "fake-only preflight complete",
                "stage": args.stage,
                "lifecycle_state": context.state.value,
                "authoritative_execution": "blocked",
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv=None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        return _research_main(values)
    except (HumanAuthoredContentRequired, IterationBlockedError, OSError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
