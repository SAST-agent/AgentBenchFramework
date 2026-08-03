"""Strict planner artifact and reachable-activation validator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.proposal import load_branch_briefs


def _action_codes(game_digest: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    roles = game_digest.get("roles")
    if not isinstance(roles, Mapping):
        return result
    for role in roles.values():
        if not isinstance(role, Mapping):
            continue
        actions = role.get("actions")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            name = action.get("name")
            code = action.get("code", action.get("id"))
            if isinstance(name, str) and name:
                if code is None and name == "HOLD":
                    code = 0
                if isinstance(code, int) and not isinstance(code, bool):
                    result[name] = code
    return result


def run_planner_check(
    *,
    briefs_path: str | Path,
    planner_input_path: str | Path,
    output_path: str | Path,
    expected_count: int,
    entry_symbol: str,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    correction_hint: dict[str, Any] | None = None
    try:
        packet = json.loads(Path(planner_input_path).read_text(encoding="utf-8"))
        index = packet.get("candidate_code_index")
        if not isinstance(index, list):
            raise ValueError("planner input requires candidate_code_index")
        symbols = {
            str(item["name"])
            for item in index
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        briefs = load_branch_briefs(
            briefs_path,
            expected_count=int(expected_count),
            known_code_symbols=symbols,
            required_entry_symbol=entry_symbol,
        )
        occupancy = packet.get("parent_occupancy")
        if not isinstance(occupancy, Mapping):
            raise ValueError("planner input requires parent_occupancy")
        examples = occupancy.get("state_examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("parent_occupancy requires state_examples")
        legal_by_state: dict[str, set[int]] = {}
        for example in examples:
            if not isinstance(example, Mapping):
                continue
            state_id = example.get("state_id")
            legal = example.get("legal_operation_types")
            if isinstance(state_id, str) and isinstance(legal, list):
                legal_by_state[state_id] = {
                    int(code)
                    for code in legal
                    if isinstance(code, int) and not isinstance(code, bool)
                }
        action_codes = _action_codes(packet.get("game_digest") or {})

        def named_legal_operations(state: str) -> list[dict[str, Any]]:
            return [
                {"code": code, "name": name}
                for name, code in sorted(
                    action_codes.items(),
                    key=lambda item: (item[1], item[0]),
                )
                if code in legal_by_state[state]
            ]

        evidence: list[dict[str, Any]] = []
        for brief in briefs:
            text = " ".join(
                (brief.activation_condition, brief.mechanism)
            )
            cited_states = [state for state in legal_by_state if state in text]
            cited_actions = {
                name: code for name, code in action_codes.items() if name in text
            }
            legal_pairs = [
                {"state_id": state, "action": name, "operation_type": code}
                for state in cited_states
                for name, code in cited_actions.items()
                if code in legal_by_state[state]
            ]
            if not cited_states:
                correction_hint = {
                    "branch_index": brief.branch_index,
                    "available_states": [
                        {
                            "state_id": state,
                            "legal_operations": named_legal_operations(state),
                        }
                        for state in sorted(legal_by_state)[:8]
                    ],
                    "requirement": (
                        "Cite one listed state_id and name one of its legal "
                        "operations exactly."
                    ),
                }
                raise ValueError(
                    f"branch {brief.branch_index} cites no parent occupancy state"
                )
            if not legal_pairs:
                correction_hint = {
                    "branch_index": brief.branch_index,
                    "cited_states": [
                        {
                            "state_id": state,
                            "legal_operations": named_legal_operations(state),
                        }
                        for state in cited_states
                    ],
                    "requirement": (
                        "Name one listed legal operation exactly in mechanism or "
                        "activation_condition."
                    ),
                }
                raise ValueError(
                    f"branch {brief.branch_index} has no cited state with its "
                    "proposed atomic operation in legal_operation_types"
                )
            evidence.append(
                {"branch_index": brief.branch_index, "legal_pairs": legal_pairs}
            )
        payload = {
            "schema_version": "1.0",
            "status": "complete",
            "branch_count": len(briefs),
            "activation_evidence": evidence,
            "error": None,
            "correction_hint": None,
        }
        returncode = 0
    except Exception as error:
        payload = {
            "schema_version": "1.0",
            "status": "failed",
            "branch_count": 0,
            "activation_evidence": [],
            "error": " ".join(str(error).split()) or error.__class__.__name__,
            "correction_hint": correction_hint,
        }
        returncode = 2
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", required=True)
    parser.add_argument("--planner-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--entry-symbol", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_planner_check(
        briefs_path=args.briefs,
        planner_input_path=args.planner_input,
        output_path=args.output,
        expected_count=args.expected_count,
        entry_symbol=args.entry_symbol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
