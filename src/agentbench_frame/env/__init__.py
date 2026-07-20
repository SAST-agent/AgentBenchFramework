"""
Environment abstraction layer for Saiblo games.

Provides:
- BaseEnv: Abstract gym-like interface for game environments
- StdioProtocol: Handles the THUAC RFC stdio protocol
- make_env: Factory function for creating environment instances
"""

from agentbench_frame.env.base import BaseEnv, EnvMode, ActionSpace, Observation
from agentbench_frame.env.stdio_protocol import StdioProtocol
from agentbench_frame.env.generals_env import GeneralsEnv
from agentbench_frame.env.registry import ENV_REGISTRY, make_env, register_env

__all__ = [
    "BaseEnv",
    "EnvMode",
    "ActionSpace",
    "Observation",
    "StdioProtocol",
    "GeneralsEnv",
    "ENV_REGISTRY",
    "make_env",
    "register_env",
]
