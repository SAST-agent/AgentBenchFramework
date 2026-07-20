"""
Registry for game environments.

Allows registration and discovery of game environment implementations.
"""

from typing import Dict, Type, Any
from agentbench_frame.env.base import BaseEnv

ENV_REGISTRY: Dict[str, Type[BaseEnv]] = {}


def register_env(name: str, env_class: Type[BaseEnv]):
    """Register a game environment class."""
    ENV_REGISTRY[name] = env_class


def make_env(name: str, **kwargs) -> BaseEnv:
    """
    Create an environment instance by name.

    Args:
        name: Game name (e.g., 'generals', 'lostspace', 'rollman')
        **kwargs: Arguments passed to the environment constructor

    Returns:
        Environment instance

    Raises:
        KeyError: If the game name is not registered
    """
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"Unknown game: '{name}'. Available: {list(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name](**kwargs)


def list_envs() -> Dict[str, str]:
    """List all registered environments with their game names."""
    return {name: cls.game_name.__get__(None, cls)
            for name, cls in ENV_REGISTRY.items()}
