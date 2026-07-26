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


@dataclass(frozen=True)
class TurnOutcome:
    done: bool
    winner: int | None
    termination_type: str | None


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
        board = [
            [
                {
                    "position": [int(cell.position[0]), int(cell.position[1])],
                    "type": int(cell.type),
                    "player": int(cell.player),
                    "army": int(cell.army),
                    "general_id": int(cell.generals.id) if cell.generals else None,
                }
                for cell in row
            ]
            for row in self.state.board
        ]
        generals = []
        for general in sorted(self.state.generals, key=lambda item: int(item.id)):
            generals.append(
                {
                    "id": int(general.id),
                    "class": type(general).__name__,
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
            )
        return {
            "round": int(self.state.round),
            "next_actor": 0,
            "board": board,
            "generals": generals,
            "coins": [int(item) for item in self.state.coin],
            "tech_level": _primitive(self.state.tech_level),
            "weapons": _primitive(self.state.active_super_weapon),
            "weapon_cds": [int(item) for item in self.state.super_weapon_cd],
        }

    def state_id(self) -> str:
        encoded = json.dumps(
            self.normalized_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
