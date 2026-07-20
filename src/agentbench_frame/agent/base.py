"""
Base agent interface.

All agents (rule-based, RL, hybrid) must implement this interface,
enabling seamless swapping in the arena and training loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import random as _random


class BaseAgent(ABC):
    """
    Abstract base class for all game-playing agents.

    Each agent receives an observation and returns an action.
    The framework handles the environment interaction; the agent
    only needs to implement the decision-making logic.

    Subclasses must implement:
    - act(): Given an observation, return an action
    - reset(): Reset agent state for a new episode
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> Any:
        """
        Choose an action given the current observation.

        Args:
            observation: The current observation from the environment.
                        Format depends on the game environment.

        Returns:
            An action in the format expected by the environment.
        """
        ...

    def reset(self):
        """Reset agent state for a new episode."""
        pass

    def learn(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Optional: Update the agent's policy from a batch of experience.

        Args:
            batch: A batch of experience data.

        Returns:
            Dictionary of training metrics.
        """
        return {}

    def save(self, path: str):
        """Save agent state to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseAgent":
        """Load agent state from disk."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class RandomAgent(BaseAgent):
    """
    A baseline agent that selects random actions from the legal action set.

    Useful for:
    - Baseline evaluation
    - Initial opponent for self-play training
    - Testing environment integration
    """

    def __init__(self, name: str = "RandomAgent", seed: Optional[int] = None):
        super().__init__(name=name)
        self.rng = _random.Random(seed)

    def act(self, observation: Dict[str, Any]) -> Any:
        """Return a random end-turn action."""
        # For Generals, the simplest valid action is to end the turn
        return [[8]]  # END_TURN

    def reset(self):
        pass
