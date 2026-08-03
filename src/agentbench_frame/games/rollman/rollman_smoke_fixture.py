#!/usr/bin/env python3
"""Framework-owned valid Rollman state fixture and public-entry verifier."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


WALL = 0
REGULAR_BEAN = 2
SPACE_CATEGORY = 10
ACTION_COUNT = 5
GHOST_COUNT = 3
SKILL_COUNT = 5


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass
class FixtureGameState:
    """Exact public field shape of frozen ``core.gamedata.GameState``."""

    space_info: dict[str, Any]
    level: int
    round: int
    board_size: int
    board: np.ndarray
    pacman_skill_status: list[int]
    pacman_pos: np.ndarray
    ghosts_pos: list[np.ndarray]
    pacman_score: int
    ghosts_score: int
    beannumber: int
    portal_available: bool
    portal_coord: np.ndarray

    def gamestate_to_statedict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "round": self.round,
            "board_size": self.board_size,
            "board": self.board,
            "pacman_skill_status": np.asarray(
                self.pacman_skill_status,
                dtype=int,
            ),
            "pacman_coord": self.pacman_pos,
            "ghosts_coord": self.ghosts_pos,
            "score": [self.pacman_score, self.ghosts_score],
            "beannumber": self.beannumber,
            "portal_available": self.portal_available,
            "portal_coord": self.portal_coord,
        }


def _coordinate(
    value: Sequence[int],
    *,
    name: str,
    board_size: int,
) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    row, column = (int(value[0]), int(value[1]))
    if not (0 <= row < board_size and 0 <= column < board_size):
        raise ValueError(f"{name} must be inside board")
    if row in {0, board_size - 1} or column in {0, board_size - 1}:
        raise ValueError(f"{name} must be on walkable interior")
    return row, column


def make_state(
    *,
    level: int,
    round_id: int,
    pacman: Sequence[int],
    ghosts: Sequence[Sequence[int]],
    score: Sequence[int] = (0, 0),
    skills: Sequence[int] = (0, 0, 0, 0, 0),
    portal_available: bool = False,
    portal_coord: Sequence[int] = (-1, -1),
    beannumber: int = 20,
    board_size: int = 22,
    board_overrides: Sequence[Sequence[int]] = (),
) -> FixtureGameState:
    """Build one valid, walkable, NumPy-compatible frozen-SDK state."""

    size = int(board_size)
    if size < 5:
        raise ValueError("board_size must be at least 5")
    if int(level) not in {1, 2, 3}:
        raise ValueError("level must be 1, 2, or 3")
    if int(round_id) < 0:
        raise ValueError("round_id cannot be negative")
    if len(ghosts) != GHOST_COUNT:
        raise ValueError("ghosts must contain exactly three coordinates")
    if len(score) != 2:
        raise ValueError("score must contain two integers")
    if len(skills) != SKILL_COUNT:
        raise ValueError("skills must contain five integers")

    board = np.full((size, size), REGULAR_BEAN, dtype=np.int64)
    board[0, :] = WALL
    board[-1, :] = WALL
    board[:, 0] = WALL
    board[:, -1] = WALL
    for override in board_overrides:
        if not isinstance(override, (list, tuple)) or len(override) != 3:
            raise ValueError("board override must contain row, column, tile")
        row, column, tile = (int(value) for value in override)
        if not (0 <= row < size and 0 <= column < size):
            raise ValueError("board override must be inside board")
        if tile not in range(SPACE_CATEGORY):
            raise ValueError("board override tile must be in 0..9")
        if (
            row in {0, size - 1} or column in {0, size - 1}
        ) and tile != WALL:
            raise ValueError("board boundary tile must remain a wall")
        board[row, column] = tile

    pacman_coord = _coordinate(pacman, name="pacman", board_size=size)
    ghost_coords = [
        _coordinate(value, name=f"ghosts[{index}]", board_size=size)
        for index, value in enumerate(ghosts)
    ]
    for name, coordinate in (
        ("pacman", pacman_coord),
        *(
            (f"ghosts[{index}]", coordinate)
            for index, coordinate in enumerate(ghost_coords)
        ),
    ):
        if int(board[coordinate]) == WALL:
            raise ValueError(f"{name} must be on a walkable tile")

    portal = np.asarray(portal_coord, dtype=np.int64)
    if portal.shape != (2,):
        raise ValueError("portal_coord must contain two coordinates")
    return FixtureGameState(
        space_info={
            "observation_space": SimpleNamespace(
                shape=(size, size),
                nvec=np.full((size, size), SPACE_CATEGORY, dtype=np.int64),
            ),
            "pacman_action_space": SimpleNamespace(n=ACTION_COUNT),
            "ghost_action_space": SimpleNamespace(
                nvec=np.full(GHOST_COUNT, ACTION_COUNT, dtype=np.int64)
            ),
        },
        level=int(level),
        round=int(round_id),
        board_size=size,
        board=board,
        pacman_skill_status=[int(value) for value in skills],
        pacman_pos=np.asarray(pacman_coord, dtype=np.int64),
        ghosts_pos=[
            np.asarray(value, dtype=np.int64) for value in ghost_coords
        ],
        pacman_score=int(score[0]),
        ghosts_score=int(score[1]),
        beannumber=int(beannumber),
        portal_available=bool(portal_available),
        portal_coord=portal,
    )


def _load_public_policy(workspace: Path):
    source = workspace / "ai.py"
    if not source.is_file():
        raise FileNotFoundError(f"candidate workspace has no ai.py: {source}")
    module_name = "agentbench_smoke_" + sha256_file(source)[:16]
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate policy: {source}")
    module = importlib.util.module_from_spec(spec)
    workspace_text = str(workspace)
    sys.path.insert(0, workspace_text)
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == workspace_text:
            sys.path.pop(0)
    policy = getattr(module, "ai_func", None)
    if not callable(policy):
        raise TypeError("candidate ai.py must define public ai_func(game_state)")
    return policy


def _decision(value: Any) -> tuple[int, str | None]:
    if isinstance(value, Mapping):
        action = value.get("action")
        memory_id = value.get("memory_id")
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        action = value[0]
        memory_id = None
    else:
        action = value
        memory_id = None
    if not isinstance(action, int) or isinstance(action, bool) or action not in range(5):
        raise ValueError("Rollman action must be an integer in 0..4")
    normalized = None if memory_id is None else str(memory_id)
    if normalized is not None and len(normalized) > 200:
        raise ValueError("memory_id cannot exceed 200 characters")
    return action, normalized


def _required_index(value: Any, *, name: str, state_count: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value not in range(state_count):
        raise ValueError(f"{name} must reference a scenario state")
    return value


def verify_scenario(
    *,
    workspace: str | Path,
    scenario_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Fresh-import ``ai.py`` and prove activation and preservation in order."""

    candidate = Path(workspace).resolve()
    scenario_file = Path(scenario_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    policy = _load_public_policy(candidate)
    value = json.loads(scenario_file.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != "1.0":
        raise ValueError("smoke scenario schema_version must be 1.0")
    raw_states = value.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("smoke scenario states must be a non-empty array")
    activation_index = _required_index(
        value.get("activation_state_index"),
        name="activation_state_index",
        state_count=len(raw_states),
    )
    preservation_index = _required_index(
        value.get("preservation_state_index"),
        name="preservation_state_index",
        state_count=len(raw_states),
    )
    activation_prefix = value.get("activation_memory_prefix")
    forbidden_prefix = value.get("preservation_forbidden_prefix")
    required_preservation = value.get("preservation_memory_prefix")
    if not isinstance(activation_prefix, str) or not activation_prefix:
        raise ValueError("activation_memory_prefix must be non-empty")
    if not isinstance(forbidden_prefix, str) or not forbidden_prefix:
        raise ValueError("preservation_forbidden_prefix must be non-empty")
    if required_preservation is not None and (
        not isinstance(required_preservation, str) or not required_preservation
    ):
        raise ValueError("preservation_memory_prefix must be non-empty")

    decisions: list[dict[str, Any]] = []
    for index, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, Mapping):
            raise ValueError("each smoke state must be an object")
        state = make_state(**dict(raw_state))
        action, memory_id = _decision(policy(state))
        decisions.append(
            {
                "state_index": index,
                "action": action,
                "memory_id": memory_id,
            }
        )

    activation_memory = decisions[activation_index]["memory_id"] or ""
    if not activation_memory.startswith(activation_prefix):
        raise ValueError(
            "activation state did not return activation_memory_prefix"
        )
    preservation_memory = decisions[preservation_index]["memory_id"] or ""
    if preservation_memory.startswith(forbidden_prefix):
        raise ValueError(
            "preservation state returned preservation_forbidden_prefix"
        )
    if required_preservation is not None and not preservation_memory.startswith(
        required_preservation
    ):
        raise ValueError(
            "preservation state did not return preservation_memory_prefix"
        )

    result = {
        "schema_version": "1.0",
        "status": "complete",
        "policy_sha256": sha256_file(candidate / "ai.py"),
        "scenario_sha256": sha256_file(scenario_file),
        "activation_state_index": activation_index,
        "preservation_state_index": preservation_index,
        "decisions": decisions,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        verify_scenario(
            workspace=args.workspace,
            scenario_path=args.scenario,
            output_path=args.output,
        )
    except Exception as error:
        message = " ".join(str(error).split()) or error.__class__.__name__
        print(f"smoke verification failed: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
