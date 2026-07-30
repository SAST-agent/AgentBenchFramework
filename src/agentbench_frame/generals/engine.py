"""Thin adapter around the preserved official Generals Python logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any
from collections.abc import Sequence

from .measurement_state import (
    measurement_state_id as canonical_measurement_state_id,
    reconstruct_measurement_state,
    serialize_measurement_state,
)


@dataclass(frozen=True)
class TurnOutcome:
    done: bool
    winner: int | None
    termination_type: str | None


@dataclass(frozen=True)
class PrimitiveOutcome:
    valid: bool
    terminal: bool
    winner: int | None


def _primitive(value: Any) -> Any:
    if is_dataclass(value):
        return _primitive(asdict(value))
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if hasattr(value, "value"):
        return _primitive(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class OfficialGeneralsEngine:
    def __init__(self, engine_root: Path, seed: int, replay_path: Path | None = None):
        self.engine_root = engine_root.resolve()
        self._main = self._load_official_main()
        random.seed(seed)
        self.state = self._main.GameState()
        self._main.update_map(self.state)
        self._main.init_generals(self.state)
        self.state.coin = [40, 40]
        if replay_path is None:
            replay_path = Path(tempfile.mkstemp(prefix="generals-official-", suffix=".jsonl")[1])
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.touch(exist_ok=True)
        self.state.replay_file = str(replay_path)

    def _load_official_main(self):
        root_text = str(self.engine_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        module_name = "agentbench_official_generals_main"
        loaded = sys.modules.get(module_name)
        if loaded is not None and Path(loaded.__file__).resolve() == self.engine_root / "main.py":
            return loaded
        spec = importlib.util.spec_from_file_location(module_name, self.engine_root / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load official engine from {self.engine_root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @property
    def round_number(self) -> int:
        return int(self.state.round)

    def initial_observation(self, player: int) -> dict:
        return _primitive(self.state.trans_state_to_init_json(player))

    def measurement_state(self, actor: int) -> dict:
        return serialize_measurement_state(self.state, actor)

    def measurement_state_id(self, actor: int) -> str:
        return canonical_measurement_state_id(
            self.measurement_state(actor)
        )

    @classmethod
    def from_measurement_state(
        cls,
        engine_root: Path,
        payload: dict,
        replay_path: Path,
    ) -> "OfficialGeneralsEngine":
        instance = cls.__new__(cls)
        instance.engine_root = Path(engine_root).resolve()
        instance._main = instance._load_official_main()
        instance.state = reconstruct_measurement_state(
            instance._main,
            payload,
            replay_path,
        )
        return instance

    def apply_primitive(
        self,
        player: int,
        command: Sequence[int],
    ) -> PrimitiveOutcome:
        if type(player) is not int or player not in (0, 1):
            return PrimitiveOutcome(False, False, None)
        try:
            normalized = tuple(int(item) for item in command)
        except (TypeError, ValueError):
            return PrimitiveOutcome(False, False, None)
        if not normalized or normalized[0] not in range(1, 8):
            return PrimitiveOutcome(False, False, None)
        try:
            valid = bool(
                self._main.execute_single_command(
                    player,
                    self.state,
                    normalized[0],
                    list(normalized[1:]),
                )
            )
        except Exception:
            valid = False
        if not valid:
            return PrimitiveOutcome(False, False, None)
        winner = int(self._main.is_game_over(self.state))
        if winner == -1:
            return PrimitiveOutcome(True, False, None)
        self.state.winner = winner
        return PrimitiveOutcome(True, True, winner)

    def apply_turn(self, player: int, commands: Sequence[Sequence[int]]) -> TurnOutcome:
        for raw in commands:
            command = tuple(int(item) for item in raw)
            if command == (8,):
                break
            if command[0] == 1 and len(command) == 5:
                command = command[:4] + (min(command[4], 2_100_000_000),)
            try:
                valid = self._main.execute_single_command(
                    player, self.state, command[0], list(command[1:])
                )
            except Exception:
                valid = False
            if not valid:
                self.state.winner = 1 - player
                return TurnOutcome(True, 1 - player, "illegal_action")
            winner = self._main.is_game_over(self.state)
            if winner != -1:
                self.state.winner = winner
                return TurnOutcome(True, winner, "normal")
        if player == 1:
            self._main.update_round(self.state)
            winner = self._main.is_game_over(self.state)
            if winner != -1:
                self.state.winner = winner
                return TurnOutcome(True, winner, "normal")
        return TurnOutcome(False, None, None)

    def normalized_state(self) -> dict:
        cells = {}
        for row in self.state.board:
            for cell in row:
                key = f"{int(cell.position[0])},{int(cell.position[1])}"
                cells[key] = {
                    "type": int(cell.type),
                    "player": int(cell.player),
                    "army": int(cell.army),
                    "general_id": int(cell.generals.id) if cell.generals else None,
                }
        generals = {}
        for general in sorted(self.state.generals, key=lambda item: int(item.id)):
            name = type(general).__name__.lower()
            general_type = "main" if "main" in name else "sub" if "sub" in name else "resource"
            generals[str(general.id)] = {
                "id": int(general.id),
                "type": general_type,
                "player": int(general.player),
                "position": [int(item) for item in general.position],
                "produce_level": int(getattr(general, "produce_level", 0)),
                "defense_level": int(getattr(general, "defense_level", 0)),
                "mobility_level": int(getattr(general, "mobility_level", 0)),
                "skills_cd": [int(item) for item in getattr(general, "skills_cd", [])],
                "skill_duration": [
                    int(item) for item in getattr(general, "skill_duration", [])
                ],
            }
        active_super_weapons = [
            {
                "type": int(weapon.type),
                "player": int(weapon.player),
                "rest": int(weapon.rest),
                "position": [
                    int(weapon.position[0]),
                    int(weapon.position[1]),
                ],
            }
            for weapon in sorted(
                self.state.active_super_weapon,
                key=lambda item: (
                    int(item.type),
                    int(item.player),
                    tuple(map(int, item.position)),
                    int(item.rest),
                ),
            )
        ]
        return {
            "round": int(self.state.round),
            "next_actor": 0,
            "cells": cells,
            "generals": generals,
            "coins": [int(item) for item in self.state.coin],
            "tech_level": _primitive(self.state.tech_level),
            "movement_budget": [
                int(item) for item in self.state.rest_move_step
            ],
            "active_super_weapons": active_super_weapons,
            "weapons": _primitive(self.state.active_super_weapon),
            "weapon_cds": [int(item) for item in self.state.super_weapon_cd],
        }

    def state_id(self) -> str:
        encoded = json.dumps(
            self.normalized_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
