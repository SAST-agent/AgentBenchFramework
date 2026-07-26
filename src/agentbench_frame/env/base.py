"""
Base environment abstraction for Saiblo game environments.

Defines the core interfaces that all game environments must implement,
supporting both direct (in-process) and subprocess (stdio protocol) modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class EnvMode(Enum):
    """Execution mode for the environment."""
    DIRECT = auto()       # Run game logic in-process (fast, for training)
    SUBPROCESS = auto()   # Run game logic as subprocess via stdio (faithful to competition)


@dataclass
class ActionSpace:
    """Description of the action space for a game."""
    type: str  # "discrete", "multi_discrete", "continuous", "dict"
    n: Optional[int] = None  # for discrete spaces
    nvec: Optional[List[int]] = None  # for multi_discrete spaces
    shape: Optional[Tuple[int, ...]] = None  # for continuous/box spaces
    low: Optional[float] = None
    high: Optional[float] = None
    description: str = ""


@dataclass
class Observation:
    """
    Standardized observation from a game environment.

    Contains the game state in a structured format that agents can consume.
    """
    state: Dict[str, Any] = field(default_factory=dict)
    player_id: int = 0
    round_num: int = 0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> Any:
        """Convert observation to a flat vector for RL agents."""
        raise NotImplementedError("Subclasses must implement to_vector()")

    def to_dict(self) -> Dict[str, Any]:
        """Return observation as a dictionary for rule-based agents."""
        return {
            "state": self.state,
            "player_id": self.player_id,
            "round_num": self.round_num,
            "done": self.done,
            "info": self.info,
        }


class BaseEnv(ABC):
    """
    Abstract base class for all game environments.

    Provides a gym-like interface (reset, step) that wraps the underlying
    game logic, whether running in-process or via subprocess stdio.

    Subclasses must implement:
    - _reset_direct() or _reset_subprocess()
    - _step_direct() or _step_subprocess()
    - action_space property
    - observation_space property
    """

    metadata: Dict[str, Any] = {}

    def __init__(self, mode: EnvMode = EnvMode.DIRECT, **kwargs):
        self.mode = mode
        self.kwargs = kwargs
        self._current_player = 0
        self._round = 0
        self._done = False
        self._trajectory: List[Dict[str, Any]] = []

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """Return the action space specification."""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """Return the observation space specification."""
        ...

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players in this game."""
        ...

    @property
    @abstractmethod
    def game_name(self) -> str:
        """Return the human-readable game name."""
        ...

    def reset(self, seed: Optional[int] = None) -> Observation:
        """
        Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation
        """
        self._round = 0
        self._done = False
        self._trajectory = []

        if seed is not None and HAS_NUMPY:
            np.random.seed(seed)

        if self.mode == EnvMode.DIRECT:
            return self._reset_direct(seed)
        else:
            return self._reset_subprocess(seed)

    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The action to execute (format depends on game)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.mode == EnvMode.DIRECT:
            obs, reward, done, info = self._step_direct(action)
        else:
            obs, reward, done, info = self._step_subprocess(action)

        self._round += 1
        self._done = done

        # Record trajectory
        self._trajectory.append({
            "round": self._round,
            "player": self._current_player,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
        })

        return obs, reward, done, info

    def _reset_direct(self, seed: Optional[int] = None) -> Observation:
        """Reset in direct mode. Override in subclass."""
        raise NotImplementedError(f"Direct mode not supported for {self.game_name}")

    def _reset_subprocess(self, seed: Optional[int] = None) -> Observation:
        """Reset in subprocess mode. Override in subclass."""
        raise NotImplementedError(f"Subprocess mode not supported for {self.game_name}")

    def _step_direct(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute step in direct mode. Override in subclass."""
        raise NotImplementedError(f"Direct mode not supported for {self.game_name}")

    def _step_subprocess(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute step in subprocess mode. Override in subclass."""
        raise NotImplementedError(f"Subprocess mode not supported for {self.game_name}")

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return the recorded trajectory of this episode."""
        return self._trajectory.copy()

    def canonical_state_id(self, observation: Observation) -> str:
        """Return a stable ID for an adapter-visible observation."""
        from agentbench_frame.eval.measurement import canonical_state_id
        return canonical_state_id(observation)

    def render(self, mode: str = "text") -> Any:
        """Render the current game state."""
        raise NotImplementedError

    def close(self):
        """Clean up resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
