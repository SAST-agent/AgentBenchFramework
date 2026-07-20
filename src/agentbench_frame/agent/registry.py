"""
Agent registry for discovery, management, and instantiation.
"""

from typing import Dict, List, Optional, Type
from agentbench_frame.agent.base import BaseAgent


class AgentRegistry:
    """
    Registry for agent classes and instances.

    Supports:
    - Registration of agent types by name
    - Factory creation of agent instances
    - Listing available agents
    - Versioning and metadata
    """

    _registry: Dict[str, Type[BaseAgent]] = {}
    _instances: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, name: str, agent_cls: Type[BaseAgent]):
        """Register an agent class."""
        cls._registry[name] = agent_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAgent:
        """Create an agent instance by registered name."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown agent type: '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._registry.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict:
        """Get metadata about a registered agent."""
        if name not in cls._registry:
            raise KeyError(f"Unknown agent: '{name}'")
        agent_cls = cls._registry[name]
        return {
            "name": name,
            "class": agent_cls.__name__,
            "module": agent_cls.__module__,
        }

    @classmethod
    def register_instance(cls, name: str, agent: BaseAgent):
        """Register a specific agent instance."""
        cls._instances[name] = agent

    @classmethod
    def get_instance(cls, name: str) -> Optional[BaseAgent]:
        """Get a registered agent instance."""
        return cls._instances.get(name)

    @classmethod
    def list_instances(cls) -> List[str]:
        """List all registered agent instances."""
        return list(cls._instances.keys())


def register_agent(name: str):
    """Decorator to register an agent class."""
    def decorator(cls: Type[BaseAgent]):
        AgentRegistry.register(name, cls)
        return cls
    return decorator


# Register built-in agent types
from agentbench_frame.agent.base import RandomAgent
from agentbench_frame.agent.rule_based import RuleBasedAgent
from agentbench_frame.agent.rl_agent import RLAgent

AgentRegistry.register("random", RandomAgent)
AgentRegistry.register("rule_based", RuleBasedAgent)
AgentRegistry.register("rl", RLAgent)
