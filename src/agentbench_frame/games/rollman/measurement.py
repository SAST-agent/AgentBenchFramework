"""Rollman-specific deterministic policy and occupancy measurements."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from agentbench_frame.eval.information_gain import (
    deterministic_measurement_distribution,
    occupancy_shift,
    policy_kl,
)
from agentbench_frame.eval.measurement import ActionSupport


ROLLMAN_ACTION_SUPPORT = ActionSupport(("0", "1", "2", "3", "4"))


def _indexed(
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    actions: dict[str, int] = {}
    for decision in decisions:
        state_id = str(decision["state_id"])
        if state_id in actions:
            raise ValueError(f"duplicate reference context: {state_id}")
        action = int(decision["action"])
        if str(action) not in ROLLMAN_ACTION_SUPPORT.action_ids:
            raise ValueError(f"Rollman action outside complete support: {action}")
        order.append(state_id)
        actions[state_id] = action
    if not order:
        raise ValueError("decision trace cannot be empty")
    return order, actions


def decision_trace_metrics(
    new_decisions: Iterable[Mapping[str, Any]],
    old_decisions: Iterable[Mapping[str, Any]],
    *,
    epsilon: float,
    new_occupancy_state_ids: Iterable[str] | None = None,
    old_occupancy_state_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Measure policies on common explicit contexts and occupancy separately."""

    new_order, new_actions = _indexed(new_decisions)
    old_order, old_actions = _indexed(old_decisions)
    if new_order != old_order:
        raise ValueError("new and old traces must share ordered reference contexts")

    trace = []
    for state_id in new_order:
        new_distribution = deterministic_measurement_distribution(
            new_actions[state_id], ROLLMAN_ACTION_SUPPORT, epsilon
        )
        old_distribution = deterministic_measurement_distribution(
            old_actions[state_id], ROLLMAN_ACTION_SUPPORT, epsilon
        )
        trace.append(
            policy_kl(
                [new_distribution[action] for action in ROLLMAN_ACTION_SUPPORT.action_ids],
                [old_distribution[action] for action in ROLLMAN_ACTION_SUPPORT.action_ids],
            )
        )

    new_occupancy = (
        list(new_occupancy_state_ids)
        if new_occupancy_state_ids is not None
        else new_order
    )
    old_occupancy = (
        list(old_occupancy_state_ids)
        if old_occupancy_state_ids is not None
        else old_order
    )
    return {
        "reference_state_ids": new_order,
        "local_policy_kl_trace": trace,
        "mean_local_policy_kl": sum(trace) / len(trace),
        "occupancy_shift": occupancy_shift(new_occupancy, old_occupancy),
        "epsilon": epsilon,
        "action_support": list(ROLLMAN_ACTION_SUPPORT.action_ids),
    }

