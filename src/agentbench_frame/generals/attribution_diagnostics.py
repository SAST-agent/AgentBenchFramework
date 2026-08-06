"""Deterministic replay-state selection and same-state attribution probes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter

from .ablation_v9 import AttributionPolicy
from .historical_policy import (
    HistoricalPolicySource,
    PolicyProbeResult,
    probe_historical_policy,
)
from .macro_action_space import GeneralsMacroActionSpaceV1
from .measurement_state import measurement_state_id
from .models import MatchResult, TurnRecord


DIAGNOSTIC_REASONS = (
    "first_main_general_danger",
    "first_enemy_contact",
    "first_economy_combat_conflict",
    "first_ineffective_large_stack",
    "first_action_divergence",
    "material_preterminal_decision",
)


@dataclass(frozen=True)
class DiagnosticState:
    pair_id: str
    source_version: str
    reason: str
    round_number: int
    actor: int
    measurement_state_id: str
    measurement_state: Mapping[str, Any]
    replay_ref: str


@dataclass(frozen=True)
class DiagnosticProbe:
    measurement_state_id: str
    policy_cell: str
    policy_id: str
    status: str
    deterministic: bool
    canonical_action: tuple[tuple[int, ...], ...] | None
    legal: bool | None
    latency_s: float
    stdout: str
    stderr: str
    error: str | None


@dataclass(frozen=True)
class TrajectoryDivergence:
    pair_id: str
    round_number: int
    actor: int
    state_id: str
    v7_action: tuple[tuple[int, ...], ...]
    v8_action: tuple[tuple[int, ...], ...]


def _pair_id(match: MatchResult) -> str:
    return f"s{match.seed}-p{match.evaluated_seat}"


def _canonical(action: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in command) for command in action)


def _position(key: str) -> tuple[int, int]:
    row, column = key.split(",", 1)
    return int(row), int(column)


def _cell(
    state: Mapping[str, Any],
    position: tuple[int, int],
) -> Mapping[str, Any] | None:
    cells = state.get("cells")
    if not isinstance(cells, Mapping):
        return None
    value = cells.get(f"{position[0]},{position[1]}")
    return value if isinstance(value, Mapping) else None


def _neighbors(position: tuple[int, int]):
    row, column = position
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


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
        if isinstance(position, Sequence) and len(position) == 2:
            candidates.append(
                (int(general.get("id", 0)), (int(position[0]), int(position[1])))
            )
    return min(candidates)[1] if candidates else None


def _contact_exists(state: Mapping[str, Any], seat: int) -> bool:
    cells = state.get("cells")
    if not isinstance(cells, Mapping):
        return False
    for key, raw in cells.items():
        if (
            not isinstance(raw, Mapping)
            or int(raw.get("player", -1)) != seat
            or int(raw.get("army", 0)) <= 1
        ):
            continue
        position = _position(str(key))
        for neighbor in _neighbors(position):
            other = _cell(state, neighbor)
            if (
                other is not None
                and int(other.get("type", 2)) != 2
                and int(other.get("player", -1)) not in (-1, seat)
            ):
                return True
    return False


def _main_danger(state: Mapping[str, Any], seat: int) -> bool:
    position = _main_position(state, seat)
    if position is None:
        return False
    main_cell = _cell(state, position)
    if main_cell is None:
        return False
    pressure = sum(
        max(int(other.get("army", 0)) - 1, 0)
        for neighbor in _neighbors(position)
        for other in (_cell(state, neighbor),)
        if other is not None
        and int(other.get("type", 2)) != 2
        and int(other.get("player", -1)) not in (-1, seat)
    )
    return pressure >= int(main_cell.get("army", 0)) and pressure > 0


def _large_stack_ineffective(turn: TurnRecord, seat: int) -> bool:
    cells = turn.state_before.get("cells")
    if not isinstance(cells, Mapping):
        return False
    large_positions = {
        _position(str(key))
        for key, cell in cells.items()
        if isinstance(cell, Mapping)
        and int(cell.get("player", -1)) == seat
        and int(cell.get("army", 0)) - 1 >= 24
    }
    if not large_positions:
        return False
    moved_positions = {
        (int(command[1]), int(command[2]))
        for command in turn.commands
        if len(command) == 5 and command[0] == 1
    }
    return large_positions.isdisjoint(moved_positions)


def _reason_indices(match: MatchResult) -> dict[str, int]:
    turns = [turn for turn in match.turns if turn.player == match.evaluated_seat]
    result: dict[str, int] = {}
    for index, turn in enumerate(turns):
        if (
            "first_main_general_danger" not in result
            and _main_danger(turn.state_before, match.evaluated_seat)
        ):
            result["first_main_general_danger"] = index
        contact = _contact_exists(turn.state_before, match.evaluated_seat)
        if "first_enemy_contact" not in result and contact:
            result["first_enemy_contact"] = index
        if (
            "first_economy_combat_conflict" not in result
            and contact
            and any(command and command[0] in {3, 5} for command in turn.commands)
        ):
            result["first_economy_combat_conflict"] = index
        if (
            "first_ineffective_large_stack" not in result
            and _large_stack_ineffective(turn, match.evaluated_seat)
        ):
            result["first_ineffective_large_stack"] = index
    if turns:
        result["material_preterminal_decision"] = len(turns) - 1
    return result


def earliest_trajectory_divergence(
    v7_match: MatchResult,
    v8_match: MatchResult,
) -> TrajectoryDivergence | None:
    if _pair_id(v7_match) != _pair_id(v8_match):
        raise ValueError("trajectory divergence requires the same pair ID")
    v8_turns = {
        (turn.round_number, turn.player, turn.state_id_before): turn
        for turn in v8_match.turns
        if turn.player == v8_match.evaluated_seat
    }
    for turn in v7_match.turns:
        if turn.player != v7_match.evaluated_seat:
            continue
        other = v8_turns.get(
            (turn.round_number, turn.player, turn.state_id_before)
        )
        if other is not None and _canonical(turn.commands) != _canonical(other.commands):
            return TrajectoryDivergence(
                pair_id=_pair_id(v7_match),
                round_number=turn.round_number,
                actor=turn.player,
                state_id=turn.state_id_before,
                v7_action=_canonical(turn.commands),
                v8_action=_canonical(other.commands),
            )
    return None


def _match_map(matches: Sequence[MatchResult], label: str) -> dict[str, MatchResult]:
    result = {_pair_id(match): match for match in matches}
    if len(result) != len(matches):
        raise ValueError(f"{label} contains duplicate pair IDs")
    return result


def diagnostic_missing_reasons(
    v7_matches: Sequence[MatchResult],
    v8_matches: Sequence[MatchResult],
) -> dict[str, tuple[str, ...]]:
    v7_by_pair = _match_map(v7_matches, "v7 matches")
    v8_by_pair = _match_map(v8_matches, "v8 matches")
    if set(v7_by_pair) != set(v8_by_pair):
        raise ValueError("v7 and v8 matches must contain identical pair IDs")
    missing = {}
    for pair_id in sorted(v7_by_pair):
        observed = set(_reason_indices(v7_by_pair[pair_id]))
        observed.update(_reason_indices(v8_by_pair[pair_id]))
        if earliest_trajectory_divergence(
            v7_by_pair[pair_id], v8_by_pair[pair_id]
        ) is not None:
            observed.add("first_action_divergence")
        absent = tuple(reason for reason in DIAGNOSTIC_REASONS if reason not in observed)
        if absent:
            missing[pair_id] = absent
    return missing


def select_diagnostic_states(
    v7_matches: Sequence[MatchResult],
    v8_matches: Sequence[MatchResult],
    *,
    measurement_states: Mapping[tuple[str, str, int], Mapping[str, Any]],
    max_states: int = 48,
) -> tuple[DiagnosticState, ...]:
    if type(max_states) is not int or max_states <= 0 or max_states > 48:
        raise ValueError("max_states must be an integer from 1 through 48")
    v7_by_pair = _match_map(v7_matches, "v7 matches")
    v8_by_pair = _match_map(v8_matches, "v8 matches")
    if set(v7_by_pair) != set(v8_by_pair):
        raise ValueError("v7 and v8 matches must contain identical pair IDs")

    selected: list[DiagnosticState] = []
    seen_ids: set[str] = set()
    for pair_id in sorted(v7_by_pair):
        divergence = earliest_trajectory_divergence(
            v7_by_pair[pair_id], v8_by_pair[pair_id]
        )
        for version, match in (
            ("v7", v7_by_pair[pair_id]),
            ("v8", v8_by_pair[pair_id]),
        ):
            policy_cell = {"v7": "A", "v8": "D"}[version]
            turns = [
                turn for turn in match.turns if turn.player == match.evaluated_seat
            ]
            reasons = _reason_indices(match)
            if divergence is not None:
                for index, turn in enumerate(turns):
                    if turn.state_id_before == divergence.state_id:
                        reasons["first_action_divergence"] = index
                        break
            candidates = []
            for priority, reason in enumerate(DIAGNOSTIC_REASONS):
                index = reasons.get(reason)
                if index is None:
                    continue
                turn = turns[index]
                snapshot = measurement_states.get((version, pair_id, turn.step))
                if snapshot is None:
                    raise ValueError(
                        f"missing measurement state for {version}/{pair_id}/{turn.step}"
                    )
                state_id = measurement_state_id(snapshot)
                candidates.append((priority, turn.round_number, turn.player, state_id, reason, turn, snapshot))
            chosen = 0
            for _, round_number, actor, state_id, reason, turn, snapshot in sorted(candidates):
                if state_id in seen_ids:
                    continue
                selected.append(
                    DiagnosticState(
                        pair_id=pair_id,
                        source_version=version,
                        reason=reason,
                        round_number=round_number,
                        actor=actor,
                        measurement_state_id=state_id,
                        measurement_state=snapshot,
                        replay_ref=(
                            f"matches/{policy_cell}/{match.case_id}/replay.jsonl"
                        ),
                    )
                )
                seen_ids.add(state_id)
                chosen += 1
                if chosen == 2 or len(selected) == max_states:
                    break
            if len(selected) == max_states:
                return tuple(selected)
    return tuple(selected)


def compare_same_state_actions(
    states: Sequence[DiagnosticState],
    policies: Sequence[AttributionPolicy],
    *,
    engine_root: Path,
    sdk_root: Path,
    probe_policy: Callable[..., PolicyProbeResult] = probe_historical_policy,
    canonicalize_action: Callable[
        [Mapping[str, Any], Sequence[Sequence[int]]],
        tuple[tuple[int, ...], ...],
    ]
    | None = None,
) -> tuple[DiagnosticProbe, ...]:
    if not states:
        raise ValueError("diagnostic states cannot be empty")
    if tuple(policy.cell for policy in policies) != ("A", "B", "C", "D"):
        raise ValueError("diagnostic policies must be ordered A, B, C, D")
    if len({item.measurement_state_id for item in states}) != len(states):
        raise ValueError("diagnostic measurement-state IDs must be unique")

    if canonicalize_action is None:
        action_space = GeneralsMacroActionSpaceV1(Path(engine_root))
        canonicalize_action = action_space.canonicalize

    resolved = []
    snapshotter = LocalWorkspaceSnapshotter()
    for policy in policies:
        manifest = snapshotter.capture(policy.source)
        if manifest.content_hash != policy.content_hash:
            raise ValueError(f"attribution policy {policy.cell} source hash changed")
        resolved.append(
            (
                policy,
                HistoricalPolicySource(
                    version=policy.policy_id,
                    run_id=policy.source_authority,
                    content_hash=policy.content_hash,
                    source=policy.source,
                    manifest=manifest,
                ),
            )
        )

    probes = []
    for state in states:
        if measurement_state_id(state.measurement_state) != state.measurement_state_id:
            raise ValueError("diagnostic measurement state hash changed")
        for policy, source in resolved:
            result = probe_policy(
                source,
                state.measurement_state,
                engine_root=engine_root,
                sdk_root=sdk_root,
                repeats=2,
            )
            canonical = None
            legal = None
            status = result.status
            error = result.error
            if result.status == "complete" and result.deterministic:
                try:
                    canonical = canonicalize_action(
                        state.measurement_state,
                        result.raw_actions[0],
                    )
                    legal = True
                except (KeyError, TypeError, ValueError) as exc:
                    status = "illegal_action"
                    legal = False
                    error = f"{type(exc).__name__}: {exc}"
            probes.append(
                DiagnosticProbe(
                    measurement_state_id=state.measurement_state_id,
                    policy_cell=policy.cell,
                    policy_id=policy.policy_id,
                    status=status,
                    deterministic=result.deterministic,
                    canonical_action=canonical,
                    legal=legal,
                    latency_s=result.elapsed_time_s,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=error,
                )
            )
    return tuple(probes)
