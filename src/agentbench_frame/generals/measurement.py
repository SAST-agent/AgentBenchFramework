"""Behavior-change measures that do not pretend a strict policy KL exists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from collections.abc import Sequence

from agentbench_frame.eval.information_gain import occupancy_shift


@dataclass(frozen=True)
class ProbeState:
    state_id: str
    state: Mapping[str, Any]
    old_action: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BehaviorChange:
    trace: tuple[float, ...]
    mean: float
    policy_kl: None = None
    policy_kl_status: str = "complete_macro_action_distribution_unavailable"


DECISION_CLASSES = (
    "end_only",
    "main_army_move",
    "non_main_army_move",
    "general_move",
    "general_upgrade",
    "skill",
    "technology",
    "super_weapon",
    "recruit",
    "other",
)


@dataclass(frozen=True)
class DecisionClassSummary:
    total: int
    counts: Mapping[str, int]
    rates: Mapping[str, float]


@dataclass(frozen=True)
class BehaviorGateResult:
    passed: bool
    conditions: Mapping[str, bool]
    improved_dense_metrics: tuple[str, ...]


def _canonical(commands) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in command) for command in commands)


def measure_action_disagreement(
    probes: Sequence[ProbeState],
    new_actions: Mapping[str, tuple[tuple[int, ...], ...]],
) -> BehaviorChange:
    if not probes:
        raise ValueError("probe set cannot be empty")
    expected = {probe.state_id for probe in probes}
    if set(new_actions) != expected:
        raise ValueError("new_actions must contain exactly the probe state IDs")
    trace = tuple(
        0.0
        if _canonical(probe.old_action) == _canonical(new_actions[probe.state_id])
        else 1.0
        for probe in probes
    )
    return BehaviorChange(trace=trace, mean=sum(trace) / len(trace))


def measure_occupancy_shift(
    new_state_ids: Sequence[str],
    old_state_ids: Sequence[str],
    smoothing: float = 1e-12,
) -> float:
    return occupancy_shift(new_state_ids, old_state_ids, smoothing=smoothing)


def _main_position(
    state: Mapping[str, Any],
    seat: int,
) -> tuple[int, int] | None:
    generals = state.get("generals")
    if not isinstance(generals, Mapping):
        return None
    candidates = []
    for general in generals.values():
        if (
            not isinstance(general, Mapping)
            or general.get("type") != "main"
            or int(general.get("player", -1)) != seat
        ):
            continue
        position = general.get("position")
        if (
            not isinstance(position, Sequence)
            or isinstance(position, (str, bytes))
            or len(position) != 2
        ):
            continue
        candidates.append(
            (
                int(general.get("id", 0)),
                (int(position[0]), int(position[1])),
            )
        )
    return min(candidates)[1] if candidates else None


def classify_macro_action(
    state: Mapping[str, Any],
    seat: int,
    action: Sequence[Sequence[int]],
) -> str:
    """Classify the first command using only externally observable fields."""
    try:
        commands = _canonical(action)
    except (TypeError, ValueError):
        return "other"
    if commands == ((8,),):
        return "end_only"
    if (
        not commands
        or commands[-1] != (8,)
        or sum(command == (8,) for command in commands) != 1
        or not commands[0]
    ):
        return "other"
    opcode = commands[0][0]
    if opcode == 1:
        if len(commands[0]) < 3:
            return "other"
        main = _main_position(state, seat)
        if main is None:
            return "other"
        source = (commands[0][1], commands[0][2])
        return (
            "main_army_move"
            if source == main
            else "non_main_army_move"
        )
    return {
        2: "general_move",
        3: "general_upgrade",
        4: "skill",
        5: "technology",
        6: "super_weapon",
        7: "recruit",
    }.get(opcode, "other")


def summarize_decision_classes(
    probes: Sequence[ProbeState],
    actions: Mapping[str, tuple[tuple[int, ...], ...]],
) -> DecisionClassSummary:
    if not probes:
        raise ValueError("probe set cannot be empty")
    expected = {probe.state_id for probe in probes}
    if set(actions) != expected:
        raise ValueError("actions must contain exactly the probe state IDs")
    counts = {name: 0 for name in DECISION_CLASSES}
    for probe in probes:
        seat = int(probe.state.get("my_seat", 0))
        category = classify_macro_action(
            probe.state,
            seat,
            actions[probe.state_id],
        )
        counts[category] += 1
    total = len(probes)
    return DecisionClassSummary(
        total=total,
        counts=counts,
        rates={name: count / total for name, count in counts.items()},
    )


def evaluate_behavior_gate(
    *,
    provider_completed: bool,
    protected_files_unchanged: bool,
    tests_passed: bool,
    action_disagreement: float | None,
    decision_classes: DecisionClassSummary,
    validation_case_count: int,
    valid_validation_case_count: int,
    dense_deltas: Mapping[str, float | None],
) -> BehaviorGateResult:
    improved = tuple(
        name
        for name, delta in dense_deltas.items()
        if delta is not None and float(delta) > 0.0
    )
    conditions = {
        "provider_completed": bool(provider_completed),
        "protected_files_unchanged": bool(protected_files_unchanged),
        "tests_passed": bool(tests_passed),
        "behavior_changed": (
            action_disagreement is not None
            and float(action_disagreement) > 0.0
        ),
        "frontline_move_observed": (
            int(decision_classes.counts.get("non_main_army_move", 0)) > 0
        ),
        "validation_complete": (
            validation_case_count > 0
            and valid_validation_case_count == validation_case_count
        ),
        "dense_progress_observed": bool(improved),
    }
    return BehaviorGateResult(
        passed=all(conditions.values()),
        conditions=conditions,
        improved_dense_metrics=improved,
    )
