"""Lossless, canonical snapshots of the official Generals ``GameState``."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


MEASUREMENT_STATE_SCHEMA = "generals-measurement-state-v1"

_STATE_FIELDS = (
    "round",
    "coin",
    "active_super_weapon",
    "super_weapon_unlocked",
    "super_weapon_cd",
    "tech_level",
    "rest_move_step",
    "next_generals_id",
    "winner",
    "generals",
    "board",
)
_GENERAL_TYPES = ("Farmer", "MainGenerals", "SubGenerals")


class MeasurementStateError(ValueError):
    """A measurement snapshot is missing or violates official-state semantics."""


def _enum_int(value: Any) -> int:
    return int(value.value if hasattr(value, "value") else value)


def _int_pair(value: Any, field: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(type(item) is int for item in value)
    ):
        raise MeasurementStateError(f"{field} must contain two integers")
    return [int(value[0]), int(value[1])]


def _general_payload(general: Any) -> dict[str, Any]:
    general_type = type(general).__name__
    if general_type not in _GENERAL_TYPES:
        raise MeasurementStateError(
            f"unsupported official general type: {general_type}"
        )
    result: dict[str, Any] = {
        "type": general_type,
        "id": int(general.id),
        "player": int(general.player),
        "position": [int(item) for item in general.position],
        "skills_cd": [int(item) for item in general.skills_cd],
        "skill_duration": [int(item) for item in general.skill_duration],
        "rest_move": int(general.rest_move),
        "produce_level": general.produce_level,
        "defense_level": general.defense_level,
        "mobility_level": general.mobility_level,
    }
    if hasattr(general, "skills"):
        result["skills"] = [
            {"type": _enum_int(skill.type), "cd": int(skill.cd)}
            for skill in general.skills
        ]
    return result


def _weapon_payload(weapon: Any) -> dict[str, Any]:
    return {
        "type": _enum_int(weapon.type),
        "player": int(weapon.player),
        "cd": int(weapon.cd),
        "rest": int(weapon.rest),
        "position": [int(item) for item in weapon.position],
    }


def serialize_measurement_state(state: Any, actor: int) -> dict[str, Any]:
    """Serialize all state that can change an official legal transition."""

    if type(actor) is not int or actor not in (0, 1):
        raise MeasurementStateError("actor must be 0 or 1")

    generals = sorted(state.generals, key=lambda item: int(item.id))
    general_identity = {id(item): int(item.id) for item in generals}
    weapons = list(state.active_super_weapon)
    weapon_identity = {id(item): index for index, item in enumerate(weapons)}
    if len(weapon_identity) != len(weapons):
        raise MeasurementStateError(
            "active_super_weapon contains duplicate object references"
        )

    board = []
    for row in state.board:
        for cell in row:
            if cell.generals is None:
                general_id = None
            else:
                general_id = general_identity.get(id(cell.generals))
                if general_id is None:
                    raise MeasurementStateError(
                        "board references a general absent from state.generals"
                    )
            weapon_indices = []
            for weapon in cell.weapon_activate:
                index = weapon_identity.get(id(weapon))
                if index is None:
                    raise MeasurementStateError(
                        "board references a weapon absent from "
                        "active_super_weapon"
                    )
                weapon_indices.append(index)
            board.append(
                {
                    "position": [int(item) for item in cell.position],
                    "type": _enum_int(cell.type),
                    "player": int(cell.player),
                    "army": int(cell.army),
                    "general_id": general_id,
                    "weapon_indices": weapon_indices,
                }
            )
    board.sort(key=lambda item: tuple(item["position"]))

    return {
        "schema": MEASUREMENT_STATE_SCHEMA,
        "actor": actor,
        "state": {
            "round": int(state.round),
            "coin": [int(item) for item in state.coin],
            "active_super_weapon": [
                _weapon_payload(item) for item in weapons
            ],
            "super_weapon_unlocked": [
                bool(item) for item in state.super_weapon_unlocked
            ],
            "super_weapon_cd": [
                int(item) for item in state.super_weapon_cd
            ],
            "tech_level": [
                [int(item) for item in player]
                for player in state.tech_level
            ],
            "rest_move_step": [
                int(item) for item in state.rest_move_step
            ],
            "next_generals_id": int(state.next_generals_id),
            "winner": int(state.winner),
            "generals": [
                _general_payload(item) for item in generals
            ],
            "board": board,
        },
    }


def measurement_state_id(payload: Mapping[str, Any]) -> str:
    """Return the content hash for one canonical measurement snapshot."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MeasurementStateError("measurement state must be an object")
    if payload.get("schema") != MEASUREMENT_STATE_SCHEMA:
        raise MeasurementStateError(
            f"schema must be {MEASUREMENT_STATE_SCHEMA}"
        )
    if type(payload.get("actor")) is not int or payload["actor"] not in (0, 1):
        raise MeasurementStateError("actor must be 0 or 1")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise MeasurementStateError("state must be an object")
    for field in _STATE_FIELDS:
        if field not in state:
            raise MeasurementStateError(f"missing state field: {field}")
    return state


def _official_data_module(main_module: Any) -> Any:
    game_state_module = sys.modules.get(main_module.GameState.__module__)
    if game_state_module is None:
        raise MeasurementStateError("official GameState module is not loaded")
    return game_state_module


def _restore_general(module: Any, raw: Mapping[str, Any]) -> Any:
    general_type = raw.get("type")
    if general_type not in _GENERAL_TYPES:
        raise MeasurementStateError(
            f"unsupported official general type: {general_type}"
        )
    cls = getattr(module, general_type)
    general = cls()
    try:
        general.id = int(raw["id"])
        general.player = int(raw["player"])
        general.position = _int_pair(
            raw["position"],
            "general position",
        )
        general.skills_cd = [int(item) for item in raw["skills_cd"]]
        general.skill_duration = [
            int(item) for item in raw["skill_duration"]
        ]
        general.rest_move = int(raw["rest_move"])
        general.produce_level = raw["produce_level"]
        general.defense_level = raw["defense_level"]
        general.mobility_level = raw["mobility_level"]
        if hasattr(general, "skills"):
            skills_raw = raw["skills"]
            if len(skills_raw) != len(general.skills):
                raise MeasurementStateError(
                    "general skills length does not match official type"
                )
            for skill, item in zip(general.skills, skills_raw):
                skill.type = type(skill.type)(int(item["type"]))
                skill.cd = int(item["cd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementStateError(
            f"invalid general record: {exc}"
        ) from exc
    return general


def _restore_weapon(module: Any, raw: Mapping[str, Any]) -> Any:
    try:
        return module.SuperWeapon(
            type=module.WeaponType(int(raw["type"])),
            player=int(raw["player"]),
            cd=int(raw["cd"]),
            rest=int(raw["rest"]),
            position=_int_pair(raw["position"], "weapon position"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MeasurementStateError(
            f"invalid active weapon record: {exc}"
        ) from exc


def reconstruct_measurement_state(
    main_module: Any,
    payload: Mapping[str, Any],
    replay_path: Path,
) -> Any:
    """Rebuild an official object graph, including shared references."""

    raw = _validate_payload(payload)
    module = _official_data_module(main_module)
    state = main_module.GameState()
    try:
        state.round = int(raw["round"])
        state.coin = [int(item) for item in raw["coin"]]
        state.super_weapon_unlocked = [
            bool(item) for item in raw["super_weapon_unlocked"]
        ]
        state.super_weapon_cd = [
            int(item) for item in raw["super_weapon_cd"]
        ]
        state.tech_level = [
            [int(item) for item in player]
            for player in raw["tech_level"]
        ]
        state.rest_move_step = [
            int(item) for item in raw["rest_move_step"]
        ]
        state.next_generals_id = int(raw["next_generals_id"])
        state.winner = int(raw["winner"])
    except (TypeError, ValueError) as exc:
        raise MeasurementStateError(
            f"invalid state scalar field: {exc}"
        ) from exc

    if len(state.coin) != 2 or len(state.rest_move_step) != 2:
        raise MeasurementStateError(
            "coin and rest_move_step must contain both players"
        )

    generals = [_restore_general(module, item) for item in raw["generals"]]
    generals_by_id = {int(item.id): item for item in generals}
    if len(generals_by_id) != len(generals):
        raise MeasurementStateError("general IDs must be unique")
    weapons = [
        _restore_weapon(module, item)
        for item in raw["active_super_weapon"]
    ]

    cells = {}
    for item in raw["board"]:
        position = _int_pair(item.get("position"), "cell position")
        key = tuple(position)
        if key in cells:
            raise MeasurementStateError("cell positions must be unique")
        cells[key] = item
    official_positions = {
        tuple(cell.position)
        for row in state.board
        for cell in row
    }
    if set(cells) != official_positions:
        raise MeasurementStateError(
            "board positions must match the official board"
        )

    referenced_generals: set[int] = set()
    for row in state.board:
        for cell in row:
            item = cells[tuple(cell.position)]
            try:
                cell.type = module.CellType(int(item["type"]))
                cell.player = int(item["player"])
                cell.army = int(item["army"])
                general_id = item["general_id"]
                if general_id is None:
                    cell.generals = None
                else:
                    general_id = int(general_id)
                    cell.generals = generals_by_id[general_id]
                    referenced_generals.add(general_id)
                    if list(cell.generals.position) != list(cell.position):
                        raise MeasurementStateError(
                            "general position disagrees with board reference"
                        )
                indices = [int(index) for index in item["weapon_indices"]]
                cell.weapon_activate = [weapons[index] for index in indices]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise MeasurementStateError(
                    f"invalid board cell at {cell.position}: {exc}"
                ) from exc

    if referenced_generals != set(generals_by_id):
        raise MeasurementStateError(
            "every general must be referenced by exactly one board cell"
        )

    replay_path = Path(replay_path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.touch(exist_ok=True)
    state.replay_file = str(replay_path)
    state.changed_cells = []
    state.generals = generals
    state.active_super_weapon = weapons
    return state
