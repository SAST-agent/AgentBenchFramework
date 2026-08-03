"""Candidate-visible AntWar2 activation contract on frozen public states."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.games.antwar2.measurement import compare_behavior


def build_activation_check_command(
    *,
    parent_root: str | Path,
    candidate_root: str | Path,
    references: Sequence[tuple[str | Path, str]],
    references_path: str | Path,
    output_path: str | Path,
    epsilon: float,
    minimum_changed_actions: int,
) -> tuple[str, ...]:
    """Freeze reference paths and return the exact candidate check command."""

    if minimum_changed_actions < 1:
        raise ValueError("minimum_changed_actions must be >= 1")
    reference_values = [
        {"replay": str(Path(replay).resolve()), "role": str(role)}
        for replay, role in references
    ]
    if not reference_values:
        raise ValueError("activation check requires replay references")
    destination = Path(references_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(reference_values, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return (
        sys.executable,
        "-m",
        "agentbench_frame.games.antwar2.activation_check",
        "--parent",
        str(Path(parent_root).resolve()),
        "--candidate",
        str(Path(candidate_root).resolve()),
        "--references",
        str(destination.resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--epsilon",
        str(float(epsilon)),
        "--minimum-changed-actions",
        str(int(minimum_changed_actions)),
    )


def run_activation_check(
    *,
    parent_root: str | Path,
    candidate_root: str | Path,
    references_path: str | Path,
    output_path: str | Path,
    epsilon: float,
    minimum_changed_actions: int,
    comparator: Callable[..., Any] = compare_behavior,
) -> int:
    """Write a bounded result and fail unless the repair changes enough actions."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = json.loads(Path(references_path).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("references must be a non-empty list")
        references = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("reference entries must be objects")
            replay = item.get("replay")
            role = item.get("role")
            if not isinstance(replay, str) or not replay:
                raise ValueError("reference replay must be a non-empty string")
            if not isinstance(role, str) or not role:
                raise ValueError("reference role must be a non-empty string")
            references.append((Path(replay), role))
        comparison = comparator(
            Path(parent_root),
            Path(candidate_root),
            references=tuple(references),
            epsilon=float(epsilon),
        )
        changed = int(comparison.changed_action_count)
        decision_count = int(comparison.decision_count)
        passed = comparison.status == "complete" and changed >= int(
            minimum_changed_actions
        )
        payload = {
            "schema_version": "1.0",
            "status": "complete" if passed else "failed",
            "decision_count": decision_count,
            "changed_action_count": changed,
            "minimum_changed_actions": int(minimum_changed_actions),
            "changed_fraction": (
                changed / decision_count if decision_count else 0.0
            ),
            "details": dict(comparison.details),
            "error": (
                None
                if passed
                else "insufficient_parent_trace_action_change: "
                f"{changed} < {minimum_changed_actions}"
            ),
        }
    except Exception as error:
        payload = {
            "schema_version": "1.0",
            "status": "failed",
            "decision_count": 0,
            "changed_action_count": 0,
            "minimum_changed_actions": int(minimum_changed_actions),
            "changed_fraction": 0.0,
            "details": {},
            "error": " ".join(str(error).split()) or error.__class__.__name__,
        }
        passed = False
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--minimum-changed-actions", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_activation_check(
        parent_root=args.parent,
        candidate_root=args.candidate,
        references_path=args.references,
        output_path=args.output,
        epsilon=args.epsilon,
        minimum_changed_actions=args.minimum_changed_actions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
