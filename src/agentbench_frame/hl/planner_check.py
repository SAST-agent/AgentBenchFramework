"""Strict planner artifact and reachable-activation validator."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.proposal import load_branch_briefs


_ATOMIC_ACTION = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


def _atoms_in_text(value: str) -> set[tuple[int, int, int]]:
    return {
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        for match in _ATOMIC_ACTION.finditer(value)
    }


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
        legal_atoms_by_state: dict[str, set[tuple[int, int, int]]] = {}
        parent_atom_by_state: dict[str, tuple[int, int, int]] = {}
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
                raw_atoms = example.get("legal_atomic_actions")
                if not isinstance(raw_atoms, list) or not raw_atoms:
                    raise ValueError(
                        f"occupancy state {state_id} requires legal_atomic_actions"
                    )
                atomic_values = [
                    atom.get("atom") if isinstance(atom, Mapping) else atom
                    for atom in raw_atoms
                ]
                atoms = {
                    tuple(atom)
                    for atom in atomic_values
                    if isinstance(atom, list)
                    and len(atom) == 3
                    and all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in atom
                    )
                }
                if not atoms:
                    raise ValueError(
                        f"occupancy state {state_id} has no valid legal atomic actions"
                    )
                legal_atoms_by_state[state_id] = atoms
                parent_selected = example.get("parent_selected")
                if (
                    isinstance(parent_selected, list)
                    and len(parent_selected) == 3
                    and all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in parent_selected
                    )
                ):
                    parent_atom_by_state[state_id] = tuple(parent_selected)
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

        def named_legal_atoms(state: str) -> list[dict[str, Any]]:
            names_by_code = {code: name for name, code in action_codes.items()}
            return [
                {"atom": list(atom), "name": names_by_code[atom[0]]}
                for atom in sorted(legal_atoms_by_state[state])
                if atom[0] in names_by_code
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
            cited_atoms = _atoms_in_text(text)
            legal_pairs = [
                {
                    "state_id": state,
                    "action": name,
                    "operation_type": code,
                    "atom": list(atom),
                }
                for state in cited_states
                for name, code in cited_actions.items()
                for atom in cited_atoms
                if atom in legal_atoms_by_state[state] and atom[0] == code
            ]
            if not cited_states:
                correction_hint = {
                    "branch_index": brief.branch_index,
                    "available_states": [
                        {
                            "state_id": state,
                            "legal_operations": named_legal_operations(state),
                            "legal_atomic_actions": named_legal_atoms(state),
                        }
                        for state in sorted(legal_by_state)[:8]
                    ],
                    "requirement": (
                        "Put one listed state_id, one of its legal operation "
                        "names, and its exact proposed atom [code,arg0,arg1] "
                        "together in mechanism or activation_condition; "
                        "diagnosis-only citations are ignored."
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
                            "legal_atomic_actions": named_legal_atoms(state),
                        }
                        for state in cited_states
                    ],
                    "requirement": (
                        "Name one listed legal operation and its exact atomic "
                        "action [code,arg0,arg1] in mechanism or activation_condition."
                    ),
                }
                raise ValueError(
                    f"branch {brief.branch_index} has no cited state with its "
                    "proposed atomic operation in legal_operation_types"
                )
            divergent_pairs = [
                pair
                for pair in legal_pairs
                if parent_atom_by_state.get(pair["state_id"])
                != tuple(pair["atom"])
            ]
            if not divergent_pairs:
                correction_hint = {
                    "branch_index": brief.branch_index,
                    "cited_states": [
                        {
                            "state_id": state,
                            "parent_atomic_action": list(
                                parent_atom_by_state[state]
                            ),
                            "divergent_legal_atomic_actions": [
                                operation
                                for operation in named_legal_atoms(state)
                                if tuple(operation["atom"])
                                != parent_atom_by_state.get(state)
                            ],
                        }
                        for state in cited_states
                    ],
                    "requirement": (
                        "Propose one listed atomic action [code,arg0,arg1] that "
                        "differs from the full parent action on the same state."
                    ),
                }
                raise ValueError(
                    f"branch {brief.branch_index} only repeats the parent "
                    "atomic operation on its cited states"
                )
            evidence.append(
                {
                    "branch_index": brief.branch_index,
                    "legal_pairs": legal_pairs,
                    "divergent_legal_pairs": divergent_pairs,
                }
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
