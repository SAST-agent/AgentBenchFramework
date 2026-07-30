"""
Agent abstraction layer.

Provides:
- BaseAgent: Abstract agent interface
- RuleBasedAgent: Configurable rule-based agent
- RLAgent: Reinforcement learning agent wrapper
- AgentRegistry: Agent discovery and management
"""

from agentbench_frame.agent.base import BaseAgent, RandomAgent
from agentbench_frame.agent.rule_based import RuleBasedAgent
from agentbench_frame.agent.rl_agent import RLAgent, PolicyNetwork
from agentbench_frame.agent.registry import AgentRegistry, register_agent
from agentbench_frame.agent.snakego_agent import SnakeGoAgent, SnakeGoSubprocessAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "RLAgent",
    "PolicyNetwork",
    "AgentRegistry",
    "register_agent",
    "SnakeGoAgent",
    "SnakeGoSubprocessAgent",
]
