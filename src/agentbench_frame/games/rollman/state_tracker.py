"""Mirror the frozen environment to capture complete pre-decision states."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class FrozenStateTracker:
    """Maintain the same public GameState as the two official SDKs."""

    def __init__(self, logic_root: str | Path):
        root = Path(logic_root).resolve()
        if not (root / "core" / "GymEnvironment.py").is_file():
            raise ValueError(f"invalid Rollman logic root: {root}")
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        module = importlib.import_module("core.GymEnvironment")
        self._env = module.PacmanEnv()
        self._initialized = False

    def reset(self, state: Mapping[str, Any]) -> None:
        self._env.ai_reset(dict(state))
        self._initialized = True

    def state(self) -> Mapping[str, Any]:
        if not self._initialized:
            raise RuntimeError("state tracker has not received a level initialization")
        game_state = self._env.game_state().gamestate_to_statedict()
        return _plain(game_state)

    def step(self, rollman_action: int, ghosts_action: tuple[int, int, int]) -> None:
        if not self._initialized:
            raise RuntimeError("state tracker has not received a level initialization")
        self._env.step(int(rollman_action), [int(value) for value in ghosts_action])

