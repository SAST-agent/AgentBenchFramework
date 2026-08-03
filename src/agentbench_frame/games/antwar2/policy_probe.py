"""Execute one AntWar2 policy on frozen public states and atomize decisions."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HOLD = (0, -1, -1)


def _reconstruct_tower_snapshots(
    replay: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Expand the replay's tower delta stream into full public snapshots."""

    towers_by_id: dict[int, dict[str, Any]] = {}
    snapshots: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(replay):
        state = record.get("round_state")
        if not isinstance(state, dict):
            continue
        rows = state.get("towers", [])
        if not isinstance(rows, list):
            raise ValueError(f"replay round {index} towers must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"replay round {index} tower must be an object")
            tower_id = int(row["id"])
            if int(row["type"]) == -1:
                towers_by_id.pop(tower_id, None)
            else:
                towers_by_id[tower_id] = copy.deepcopy(row)
        snapshots[index] = [
            copy.deepcopy(towers_by_id[tower_id])
            for tower_id in sorted(towers_by_id)
        ]
    return snapshots


def _load_replay(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"invalid replay: {path}")
    return value


def _is_terminal_state(state: dict[str, Any]) -> bool:
    winner = state.get("winner")
    return isinstance(winner, int) and not isinstance(winner, bool) and winner in {0, 1}


def _sample_records(
    replay: list[dict[str, Any]],
    *,
    max_states: int,
) -> list[tuple[int, dict[str, Any]]]:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    valid = [
        (index, record)
        for index, record in enumerate(replay)
        if isinstance(record.get("round_state"), dict)
        and not _is_terminal_state(record["round_state"])
    ]
    if len(valid) <= max_states:
        return valid
    if max_states == 1:
        return [valid[0]]
    positions = {
        round(offset * (len(valid) - 1) / (max_states - 1))
        for offset in range(max_states)
    }
    return [valid[position] for position in sorted(positions)]


def _main(request_path: Path, output_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_root = Path(request["candidate_root"]).resolve()
    sys.path.insert(0, str(candidate_root))

    from SDK.backend.engine import PublicRoundState
    from SDK.backend.model import Operation
    from SDK.backend.runtime import MatchRuntime
    from SDK.utils.constants import (
        HIGHLAND_CELLS,
        OperationType,
        SuperWeaponType,
        TOWER_UPGRADE_TREE,
        VALID_CELLS,
    )

    def atom(operation: Operation) -> tuple[int, int, int]:
        return (
            int(operation.op_type),
            int(operation.arg0),
            int(operation.arg1),
        )

    def replay_operation(item: dict[str, Any]) -> Operation:
        operation_type = OperationType(int(item["type"]))
        if operation_type in {
            OperationType.BUILD_TOWER,
            OperationType.USE_LIGHTNING_STORM,
            OperationType.USE_EMP_BLASTER,
            OperationType.USE_DEFLECTOR,
            OperationType.USE_EMERGENCY_EVASION,
        }:
            position = item.get("pos") or {}
            return Operation(operation_type, int(position["x"]), int(position["y"]))
        if operation_type == OperationType.UPGRADE_TOWER:
            return Operation(operation_type, int(item["id"]), int(item["args"]))
        if operation_type == OperationType.DOWNGRADE_TOWER:
            return Operation(operation_type, int(item["id"]))
        return Operation(operation_type)

    def public_state(
        round_index: int,
        data: dict[str, Any],
        tower_snapshot: list[dict[str, Any]],
    ) -> PublicRoundState:
        towers = [
            (
                int(item["id"]),
                int(item["player"]),
                int(item["pos"]["x"]),
                int(item["pos"]["y"]),
                int(item["type"]),
                int(item.get("cd", 0)),
                int(item.get("hp", -1)),
            )
            for item in tower_snapshot
        ]
        ants = [
            (
                int(item["id"]),
                int(item["player"]),
                int(item["pos"]["x"]),
                int(item["pos"]["y"]),
                int(item["hp"]),
                int(item["level"]),
                int(item["age"]),
                int(item["status"]),
                int(item.get("behavior", 0)),
                int(item.get("kind", 0)),
            )
            for item in data.get("ants", [])
        ]
        effects = [
            (
                int(item["type"]),
                int(item["player"]),
                int(item["x"]),
                int(item["y"]),
                int(item.get("duration", item.get("remaining_turns", 0))),
            )
            for item in data.get("activeEffects", [])
        ]
        return PublicRoundState(
            round_index=round_index,
            towers=towers,
            ants=ants,
            coins=tuple(int(value) for value in data["coins"]),
            camps_hp=tuple(int(value) for value in data["camps"]),
            speed_lv=tuple(int(value) for value in data.get("speedLv", (0, 0))),
            anthp_lv=tuple(int(value) for value in data.get("anthpLv", (0, 0))),
            weapon_cooldowns=tuple(
                tuple(int(value) for value in row)
                for row in data.get("weaponCooldowns", ((0, 0, 0, 0),) * 2)
            ),
            active_effects=effects,
        )

    def legal_atoms(state, player: int, prefix: list[Operation]) -> list[list[int]]:
        support: set[tuple[int, int, int]] = {HOLD}
        for x, y in HIGHLAND_CELLS[player]:
            operation = Operation(OperationType.BUILD_TOWER, x, y)
            if state.can_apply_operation(player, operation, prefix):
                support.add(atom(operation))
        for tower in state.towers_of(player):
            operation = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
            if state.can_apply_operation(player, operation, prefix):
                support.add(atom(operation))
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()):
                operation = Operation(
                    OperationType.UPGRADE_TOWER,
                    tower.tower_id,
                    int(target),
                )
                if state.can_apply_operation(player, operation, prefix):
                    support.add(atom(operation))
        for weapon in SuperWeaponType:
            operation_type = OperationType(20 + int(weapon))
            for x, y in VALID_CELLS:
                operation = Operation(operation_type, x, y)
                if state.can_apply_operation(player, operation, prefix):
                    support.add(atom(operation))
        for operation_type in (
            OperationType.UPGRADE_GENERATION_SPEED,
            OperationType.UPGRADE_GENERATED_ANT,
        ):
            operation = Operation(operation_type)
            if state.can_apply_operation(player, operation, prefix):
                support.add(atom(operation))
        return [list(item) for item in sorted(support)]

    spec = importlib.util.spec_from_file_location("ai", candidate_root / "ai.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {candidate_root / 'ai.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ai"] = module
    spec.loader.exec_module(module)

    cases: list[dict[str, Any]] = []
    for reference_index, reference in enumerate(request["references"]):
        replay_path = Path(reference["replay"]).resolve()
        role = str(reference["role"])
        if role not in {"P0", "P1"}:
            raise ValueError(f"invalid role: {role}")
        player = 0 if role == "P0" else 1
        replay = _load_replay(replay_path)
        tower_snapshots = _reconstruct_tower_snapshots(replay)
        seed = int(replay[0].get("seed", 0))
        agent = module.AI()
        agent.on_match_start(player, seed)
        runtime = MatchRuntime.create(player=player, seed=seed, prefer_native=False)
        max_states = int(request.get("max_states_per_reference", 64))
        for index, record in _sample_records(replay, max_states=max_states):
            state_data = record.get("round_state")
            if not isinstance(state_data, dict):
                raise ValueError(f"replay round {index} has no public state")
            frozen = public_state(
                index + 1,
                state_data,
                tower_snapshots[index],
            )
            runtime.state.sync_public_round_state(frozen)
            agent.on_round_state(frozen)
            opponent_operations = [
                replay_operation(item)
                for item in record.get(f"op{1 - player}", [])
            ]
            agent.on_opponent_operations(opponent_operations)
            proposed = agent.choose_operations(runtime.state, player)
            if not isinstance(proposed, list):
                raise TypeError("AI.choose_operations must return a list")
            accepted: list[Operation] = []
            steps: list[dict[str, Any]] = []
            for operation in proposed:
                support = legal_atoms(runtime.state, player, accepted)
                if runtime.state.can_apply_operation(player, operation, accepted):
                    steps.append(
                        {"selected": list(atom(operation)), "support": support}
                    )
                    accepted.append(operation)
            terminal_support = legal_atoms(runtime.state, player, accepted)
            agent.on_self_operations(accepted)
            cases.append(
                {
                    "state_id": f"reference-{reference_index}:{index}:{role}",
                    "replay": str(replay_path),
                    "round": index + 1,
                    "role": role,
                    "steps": steps,
                    "terminal_support": terminal_support,
                }
            )
    output_path.write_text(
        json.dumps(
            {"schema_version": "1.0", "cases": cases},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: policy_probe.py REQUEST.json OUTPUT.json")
    _main(Path(sys.argv[1]), Path(sys.argv[2]))
