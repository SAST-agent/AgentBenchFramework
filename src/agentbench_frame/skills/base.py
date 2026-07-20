"""
Skills system for rule-based agents.

Skills are modular, reusable, versioned strategy components that can be:
- Composed into agent decision pipelines
- Tracked for performance analysis
- Iterated on independently (A/B testing)
- Exposed as MCP tools for external access

Each Skill has:
- Metadata (name, version, description, tags)
- An activation condition (when should it fire?)
- An execution function (what does it do?)
- Performance tracking (how well does it work?)

Design pattern:
    Skill = structured Rule with metadata + tracking + MCP bridge
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import json


@dataclass
class SkillMeta:
    """Metadata for a skill."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    game: str = ""  # which game this skill is for
    dependencies: List[str] = field(default_factory=list)  # other skill names


@dataclass
class SkillStats:
    """Performance tracking for a skill."""
    total_activations: int = 0
    successful_actions: int = 0
    total_reward_contributed: float = 0.0
    avg_decision_time_ms: float = 0.0
    last_used_round: int = -1
    per_opponent: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_activations == 0:
            return 0.0
        return self.successful_actions / self.total_activations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_activations": self.total_activations,
            "successful_actions": self.successful_actions,
            "success_rate": self.success_rate,
            "total_reward_contributed": self.total_reward_contributed,
            "avg_decision_time_ms": self.avg_decision_time_ms,
            "last_used_round": self.last_used_round,
            "per_opponent": self.per_opponent,
        }


class Skill(ABC):
    """
    Abstract base class for all skills.

    A Skill is a modular, reusable piece of agent logic. Unlike raw rule
    functions, skills carry metadata, track their own performance, and
    can be exposed as MCP tools.

    Subclasses must implement:
    - can_activate(): Whether this skill should fire given the current state
    - execute(): What action to take

    Usage:
        skill = ReplayReaderSkill()
        if skill.can_activate(observation, context):
            action = skill.execute(observation, context)
    """

    def __init__(self, meta: SkillMeta):
        self.meta = meta
        self.stats = SkillStats()
        self._context: Dict[str, Any] = {}

    # ---- Core interface ----

    @abstractmethod
    def can_activate(self, observation: Dict[str, Any],
                     context: Dict[str, Any]) -> bool:
        """
        Check whether this skill should activate given the current observation.

        Args:
            observation: Current game observation
            context: Shared agent context (persistent across turns)

        Returns:
            True if this skill wants to handle the current turn
        """
        ...

    @abstractmethod
    def execute(self, observation: Dict[str, Any],
                context: Dict[str, Any]) -> Any:
        """
        Execute this skill and return an action.

        Args:
            observation: Current game observation
            context: Shared agent context

        Returns:
            An action in the environment's expected format
        """
        ...

    # ---- Lifecycle ----

    def on_episode_start(self, observation: Dict[str, Any]):
        """Called when a new episode begins."""
        self._context = {}

    def on_episode_end(self, observation: Dict[str, Any],
                       winner: int, reward: float):
        """Called when an episode ends."""
        pass

    def reset(self):
        """Reset skill state between episodes."""
        self._context = {}

    # ---- Tracking ----

    def record_activation(self, success: bool, reward: float,
                          decision_time_ms: float, opponent: str = ""):
        """Record that this skill was activated."""
        self.stats.total_activations += 1
        if success:
            self.stats.successful_actions += 1
        self.stats.total_reward_contributed += reward
        self.stats.avg_decision_time_ms = (
            (self.stats.avg_decision_time_ms * (self.stats.total_activations - 1) +
             decision_time_ms) / self.stats.total_activations
        )

        if opponent and opponent not in self.stats.per_opponent:
            self.stats.per_opponent[opponent] = {"activations": 0, "successes": 0}
        if opponent:
            self.stats.per_opponent[opponent]["activations"] += 1
            if success:
                self.stats.per_opponent[opponent]["successes"] += 1

    # ---- MCP Bridge ----

    def as_mcp_tool_definition(self) -> Dict[str, Any]:
        """
        Export this skill as an MCP tool definition.

        Returns:
            MCP-compatible tool schema
        """
        return {
            "name": self.meta.name,
            "description": f"[Skill v{self.meta.version}] {self.meta.description}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation": {
                        "type": "object",
                        "description": "Current game observation"
                    },
                    "context": {
                        "type": "object",
                        "description": "Shared agent context"
                    },
                },
                "required": ["observation"],
            },
        }

    def call_as_mcp(self, **kwargs) -> Dict[str, Any]:
        """Call this skill through the MCP interface."""
        obs = kwargs.get("observation", {})
        ctx = kwargs.get("context", {})
        start = time.time()
        try:
            action = self.execute(obs, ctx)
            elapsed = (time.time() - start) * 1000
            return {
                "action": action,
                "skill_name": self.meta.name,
                "skill_version": self.meta.version,
                "decision_time_ms": elapsed,
                "success": True,
            }
        except Exception as e:
            return {
                "error": str(e),
                "skill_name": self.meta.name,
                "success": False,
            }

    # ---- Serialization ----

    def get_config(self) -> Dict[str, Any]:
        """Get skill configuration for serialization."""
        return {
            "name": self.meta.name,
            "version": self.meta.version,
            "description": self.meta.description,
            "tags": self.meta.tags,
            "stats": self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        return f"Skill({self.meta.name} v{self.meta.version})"


# ---- Function-based Skill (for quick composition) ----

class FunctionSkill(Skill):
    """
    A skill defined by a function pair (can_activate, execute).

    Useful for quick experimentation without subclassing.
    """

    def __init__(self,
                 meta: SkillMeta,
                 can_activate_fn: Callable[[Dict, Dict], bool],
                 execute_fn: Callable[[Dict, Dict], Any]):
        super().__init__(meta)
        self._can_activate_fn = can_activate_fn
        self._execute_fn = execute_fn

    def can_activate(self, observation: Dict[str, Any],
                     context: Dict[str, Any]) -> bool:
        return self._can_activate_fn(observation, context)

    def execute(self, observation: Dict[str, Any],
                context: Dict[str, Any]) -> Any:
        return self._execute_fn(observation, context)
