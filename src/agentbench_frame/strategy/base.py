"""
BaseStrategy: the single decision interface every AntWAR2 strategy implements.

Extends BaseAgent so any strategy drops straight into Match / Arena, while
carrying population metadata (strategy_id, version, source) and the
save/load contract used by the Population store.
"""

from __future__ import annotations

import json
import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from agentbench_frame.agent.base import BaseAgent


@dataclass
class StrategyMeta:
    strategy_id: str = ""
    game: str = "30_antwar2"
    kind: str = "rule"
    version: int = 0
    source: str = ""
    parent_id: str = ""
    created: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def backend_from_obs(observation: Any):
    """Pull the live backend state handle that AntWar2Env attaches to obs."""
    if observation is None:
        return None
    if isinstance(observation, dict):
        info = observation.get("info") or {}
        bs = info.get("_backend_state")
        if bs is not None:
            return bs
        if "_backend_state" in observation:
            return observation["_backend_state"]
        return None
    info = getattr(observation, "info", None) or {}
    return info.get("_backend_state")


def _player_from_obs(observation: Any) -> int:
    if isinstance(observation, dict):
        return int(observation.get("player_id", observation.get("current_player", 0)) or 0)
    return int(getattr(observation, "player_id", 0) or 0)


class BaseStrategy(BaseAgent):
    """Abstract strategy. Subclasses implement _decide(backend_state, player)."""

    kind: str = "base"

    def __init__(self, name: str = "BaseStrategy", meta: Optional[StrategyMeta] = None,
                 **kwargs):
        super().__init__(name=name)
        self.meta = meta or StrategyMeta(kind=self.kind)
        if not self.meta.strategy_id:
            self.meta.strategy_id = name
        if not self.meta.created:
            self.meta.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.metadata.update(self.meta.to_dict())

    @abstractmethod
    def _decide(self, backend_state, player: int) -> List[List[int]]:
        ...

    def act(self, observation: Any) -> Any:
        backend_state = backend_from_obs(observation)
        player = _player_from_obs(observation)
        if backend_state is None:
            return []
        try:
            return self._decide(backend_state, player)
        except Exception:
            return []

    def reset(self):
        pass

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.meta.kind = self.kind
        with open(os.path.join(path, "strategy.json"), "w") as f:
            json.dump(self.meta.to_dict(), f, indent=2)
        self._save_artifacts(path)

    def load(self, path: str):
        meta_path = os.path.join(path, "strategy.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = StrategyMeta.from_dict(json.load(f))
        self._load_artifacts(path)
        return self

    def _save_artifacts(self, path: str):
        pass

    def _load_artifacts(self, path: str):
        pass

    def to_meta(self) -> Dict[str, Any]:
        self.meta.kind = self.kind
        return self.meta.to_dict()


def load_strategy(path: str) -> BaseStrategy:
    """Reconstruct a strategy from a population directory using its manifest."""
    from agentbench_frame.strategy.rule_strategy import RuleStrategy
    from agentbench_frame.strategy.rl_strategy import RLStrategy
    from agentbench_frame.strategy.external_strategy import ExternalStrategy

    meta_path = os.path.join(path, "strategy.json")
    with open(meta_path) as f:
        meta = StrategyMeta.from_dict(json.load(f))
    kind = meta.kind
    if kind == "rule":
        s = RuleStrategy(name=meta.strategy_id, meta=meta)
    elif kind == "rl":
        s = RLStrategy(name=meta.strategy_id, meta=meta)
    elif kind == "external":
        s = ExternalStrategy(name=meta.strategy_id, meta=meta)
    else:
        raise ValueError(f"Unknown strategy kind: {kind}")
    s.load(path)
    return s
