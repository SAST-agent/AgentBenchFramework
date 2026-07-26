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
