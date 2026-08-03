"""Frozen-occupancy atomic behavior comparison for deterministic policies."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.game_profile import BehaviorComparison


HOLD = (0, -1, -1)


def summarize_probe_occupancy(
    probe: Mapping[str, Any],
    *,
    max_examples: int = 16,
) -> dict[str, Any]:
    """Summarize raw public occupancy without inventing tactical features."""

    cases = probe.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("probe output must contain cases")
    ordered = sorted(
        (case for case in cases if isinstance(case, Mapping)),
        key=lambda case: str(case.get("state_id") or ""),
    )
    if not ordered:
        raise ValueError("probe output has no valid cases")
    count = min(max(1, int(max_examples)), len(ordered))
    positions = (
        set(range(len(ordered)))
        if count == len(ordered)
        else {
            round(offset * (len(ordered) - 1) / (count - 1))
            for offset in range(count)
        }
    )
    range_fields = (
        ("round_index", ("round_index",)),
        ("self_coins", ("coins", "self")),
        ("enemy_coins", ("coins", "enemy")),
        ("self_camp_hp", ("camp_hp", "self")),
        ("enemy_camp_hp", ("camp_hp", "enemy")),
        ("self_generation_level", ("generation_level", "self")),
        ("enemy_generation_level", ("generation_level", "enemy")),
        ("self_ant_level", ("ant_level", "self")),
        ("enemy_ant_level", ("ant_level", "enemy")),
        ("self_tower_count", ("tower_count", "self")),
        ("enemy_tower_count", ("tower_count", "enemy")),
        ("self_ant_count", ("ant_count", "self")),
        ("enemy_ant_count", ("ant_count", "enemy")),
    )
    values_by_role: dict[str, dict[str, list[int]]] = {}
    actions_by_role: dict[str, dict[tuple[int, int, int], int]] = {}
    examples: list[dict[str, Any]] = []
    for index, case in enumerate(ordered):
        summary = case.get("public_summary")
        if not isinstance(summary, Mapping):
            summary = {}
        role = str(case.get("role") or summary.get("role") or "unknown")
        role_values = values_by_role.setdefault(role, {})
        for label, path in range_fields:
            current: Any = summary
            for key in path:
                if not isinstance(current, Mapping):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, int) and not isinstance(current, bool):
                role_values.setdefault(label, []).append(current)
        selected, _ = _step(case, 0)
        counts = actions_by_role.setdefault(role, {})
        counts[selected] = counts.get(selected, 0) + 1
        if index in positions:
            examples.append(
                {
                    "state_id": str(case.get("state_id") or ""),
                    "public_summary": dict(summary),
                    "parent_selected": list(selected),
                }
            )
    return {
        "schema_version": "1.0",
        "state_count": len(ordered),
        "roles": {
            role: {
                "observed_ranges": {
                    label: {"min": min(items), "max": max(items)}
                    for label, items in sorted(values.items())
                    if items
                },
                "first_atomic_action_counts": [
                    {"atom": list(atom), "count": count}
                    for atom, count in sorted(
                        actions_by_role.get(role, {}).items()
                    )
                ],
            }
            for role, values in sorted(values_by_role.items())
        },
        "state_examples": examples,
    }


def summarize_occupancy(
    *,
    candidate_root: str | Path,
    references: Sequence[tuple[str | Path, str]],
    max_states_per_reference: int = 64,
) -> dict[str, Any]:
    return summarize_probe_occupancy(
        probe_policy(
            candidate_root=candidate_root,
            references=references,
            max_states_per_reference=max_states_per_reference,
        )
    )


def _atom(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"invalid atomic operation: {value!r}")
    return tuple(value)


def _support(value: Any) -> set[tuple[int, int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("atomic support must be a non-empty list")
    result = {_atom(item) for item in value}
    result.add(HOLD)
    return result


def _deterministic_distribution(
    support: set[tuple[int, int, int]],
    selected: tuple[int, int, int],
    *,
    epsilon: float,
) -> dict[tuple[int, int, int], float]:
    support = set(support)
    support.add(selected)
    uniform = epsilon / len(support)
    return {
        action: uniform + (1.0 - epsilon if action == selected else 0.0)
        for action in support
    }


def _step(case: Mapping[str, Any], index: int) -> tuple[tuple[int, int, int], set[tuple[int, int, int]]]:
    steps = case.get("steps")
    if not isinstance(steps, list):
        raise ValueError("probe case steps must be a list")
    if index >= len(steps):
        return HOLD, _support(case.get("terminal_support"))
    step = steps[index]
    if not isinstance(step, Mapping):
        raise ValueError("probe step must be an object")
    return _atom(step.get("selected")), _support(step.get("support"))


def compare_probe_outputs(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    epsilon: float,
) -> BehaviorComparison:
    """Compute candidate||parent KL for literal atoms on identical states."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    parent_cases = parent.get("cases")
    candidate_cases = candidate.get("cases")
    if not isinstance(parent_cases, list) or not isinstance(candidate_cases, list):
        raise ValueError("probe outputs must contain case lists")
    parent_by_id = {
        str(case["state_id"]): case
        for case in parent_cases
        if isinstance(case, Mapping) and "state_id" in case
    }
    candidate_by_id = {
        str(case["state_id"]): case
        for case in candidate_cases
        if isinstance(case, Mapping) and "state_id" in case
    }
    if set(parent_by_id) != set(candidate_by_id):
        raise ValueError("parent and candidate probes use different frozen states")
    ordered_state_ids = sorted(parent_by_id)
    example_count = min(16, len(ordered_state_ids))
    if example_count == len(ordered_state_ids):
        example_ids = set(ordered_state_ids)
    elif example_count <= 1:
        example_ids = {ordered_state_ids[0]} if ordered_state_ids else set()
    else:
        example_ids = {
            ordered_state_ids[
                round(offset * (len(ordered_state_ids) - 1) / (example_count - 1))
            ]
            for offset in range(example_count)
        }
    values: list[float] = []
    changed = 0
    changed_examples: list[dict[str, Any]] = []
    state_examples: list[dict[str, Any]] = []
    role_values: dict[str, list[float]] = {}
    for state_id in ordered_state_ids:
        parent_case = parent_by_id[state_id]
        candidate_case = candidate_by_id[state_id]
        parent_steps = parent_case.get("steps")
        candidate_steps = candidate_case.get("steps")
        if not isinstance(parent_steps, list) or not isinstance(candidate_steps, list):
            raise ValueError("probe case steps must be lists")
        count = max(1, len(parent_steps), len(candidate_steps))
        role = state_id.rsplit(":", 1)[-1]
        if state_id in example_ids:
            public_summary = parent_case.get("public_summary")
            if not isinstance(public_summary, Mapping):
                public_summary = {}
            parent_selected, _ = _step(parent_case, 0)
            candidate_selected, _ = _step(candidate_case, 0)
            state_examples.append(
                {
                    "state_id": state_id,
                    "public_summary": dict(public_summary),
                    "parent_selected": list(parent_selected),
                    "candidate_selected": list(candidate_selected),
                }
            )
        for index in range(count):
            parent_selected, parent_support = _step(parent_case, index)
            candidate_selected, candidate_support = _step(candidate_case, index)
            support = (
                parent_support
                | candidate_support
                | {parent_selected, candidate_selected}
            )
            p = _deterministic_distribution(
                support,
                candidate_selected,
                epsilon=epsilon,
            )
            q = _deterministic_distribution(
                support,
                parent_selected,
                epsilon=epsilon,
            )
            kl = sum(p[action] * math.log(p[action] / q[action]) for action in support)
            values.append(kl)
            role_values.setdefault(role, []).append(kl)
            action_changed = candidate_selected != parent_selected
            changed += action_changed
            if action_changed and len(changed_examples) < 16:
                changed_examples.append(
                    {
                        "state_id": state_id,
                        "step_index": index,
                        "parent_selected": list(parent_selected),
                        "candidate_selected": list(candidate_selected),
                    }
                )
    mean = sum(values) / len(values) if values else 0.0
    return BehaviorComparison(
        status="complete",
        decision_count=len(values),
        changed_action_count=changed,
        details={
            "metric": "epsilon_smoothed_atomic_policy_kl",
            "direction": "candidate||parent",
            "epsilon": float(epsilon),
            "state_count": len(parent_by_id),
            "mean_kl_nats_per_decision": mean,
            "role_mean_kl": {
                role: sum(items) / len(items)
                for role, items in sorted(role_values.items())
            },
            "changed_examples": changed_examples,
            "state_examples": state_examples,
        },
    )


def probe_policy(
    *,
    candidate_root: str | Path,
    references: Sequence[tuple[str | Path, str]],
    max_states_per_reference: int = 64,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Execute one candidate on frozen public-state occupancy in a subprocess."""

    root = Path(candidate_root).resolve()
    script = Path(__file__).with_name("policy_probe.py").resolve()
    with tempfile.TemporaryDirectory(prefix="agentbench-antwar-probe-") as temporary:
        request = Path(temporary) / "request.json"
        output = Path(temporary) / "output.json"
        request.write_text(
            json.dumps(
                {
                    "candidate_root": str(root),
                    "references": [
                        {"replay": str(Path(path).resolve()), "role": role}
                        for path, role in references
                    ],
                    "max_states_per_reference": int(max_states_per_reference),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            (sys.executable, str(script), str(request), str(output)),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode != 0 or not output.is_file():
            diagnostic = (completed.stderr or completed.stdout)[-8000:]
            raise RuntimeError(f"policy probe failed: {diagnostic}")
        value = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("policy probe output must be an object")
        return value


def compare_behavior(
    parent_root: str | Path,
    candidate_root: str | Path,
    *,
    references: Sequence[tuple[str | Path, str]],
    epsilon: float = 0.05,
    max_states_per_reference: int = 64,
) -> BehaviorComparison:
    parent = probe_policy(
        candidate_root=parent_root,
        references=references,
        max_states_per_reference=max_states_per_reference,
    )
    candidate = probe_policy(
        candidate_root=candidate_root,
        references=references,
        max_states_per_reference=max_states_per_reference,
    )
    return compare_probe_outputs(parent, candidate, epsilon=epsilon)
