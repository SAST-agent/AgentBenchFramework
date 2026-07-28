"""
Unified strategy interface for AntWAR2.

A strategy is any decision maker that can produce operations for a game
state. The same interface covers rule code, RL models and external processes,
so they can be swapped freely in Match / Arena / PayoffMatrix.
"""

from agentbench_frame.strategy.base import BaseStrategy, load_strategy, backend_from_obs
from agentbench_frame.strategy.rule_strategy import RuleStrategy
from agentbench_frame.strategy.rl_strategy import RLStrategy
from agentbench_frame.strategy.external_strategy import ExternalStrategy

__all__ = [
    "BaseStrategy",
    "RuleStrategy",
    "RLStrategy",
    "ExternalStrategy",
    "load_strategy",
    "backend_from_obs",
]
