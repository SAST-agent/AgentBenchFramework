"""
RuleStrategy: adapts rule-based decision code to the unified strategy interface.

Three construction modes:
  - named builtin policy: "hold", "random", "greedy", "pragmatic"
  - explicit callable: decide(backend_state, player) -> operations
  - python module path exposing a decide/choose_operations/choose_bundle symbol

Operations may be returned as Operation objects, integer token lists, or
ActionBundle objects; all are normalized by the environment.
"""

from __future__ import annotations

import importlib.util
import os
import random as _random
from typing import Any, Callable, List, Optional

from agentbench_frame.strategy.base import BaseStrategy, StrategyMeta
from agentbench_frame.env._antwar2.utils.actions import ActionBundle, ActionCatalog
from agentbench_frame.env._antwar2.utils.features import FeatureExtractor
from agentbench_frame.env._antwar2.utils.constants import MAX_ACTIONS

DecideFn = Callable[[Any, int], Any]


class RuleStrategy(BaseStrategy):
    kind = "rule"

    def __init__(self,
                 name: str = "RuleStrategy",
                 policy: Optional[str] = None,
                 decide: Optional[DecideFn] = None,
                 module_path: Optional[str] = None,
                 meta: Optional[StrategyMeta] = None,
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__(name=name, meta=meta)
        self.policy_name = policy or "greedy"
        self._decide_fn: Optional[DecideFn] = decide
        self._module_path = module_path
        self._rng = _random.Random(seed)
        self._feature = FeatureExtractor(max_actions=MAX_ACTIONS)
        self._catalog = ActionCatalog(max_actions=MAX_ACTIONS, feature_extractor=self._feature)
        if decide is None and module_path is None:
            self._decide_fn = self._builtin(self.policy_name)
        elif module_path is not None:
            self._decide_fn = self._load_module(module_path)
        self.meta.kind = self.kind
        if not self.meta.description:
            self.meta.description = self.policy_name

    @staticmethod
    def _builtin(name: str) -> DecideFn:
        def hold(state, player):
            return []

        def random_policy(state, player):
            catalog = ActionCatalog(feature_extractor=FeatureExtractor())
            bundles = catalog.build(state, player) or [ActionBundle(name="hold")]
            import random as r
            return list(r.choice(bundles).operations)

        def greedy(state, player):
            catalog = ActionCatalog(feature_extractor=FeatureExtractor())
            bundles = catalog.build(state, player) or [ActionBundle(name="hold")]
            return list(max(bundles, key=lambda b: b.score).operations)

        def pragmatic(state, player):
            catalog = ActionCatalog(feature_extractor=FeatureExtractor())
            bundles = catalog.build(state, player) or [ActionBundle(name="hold")]
            scored = sorted(bundles, key=lambda b: b.score, reverse=True)
            coin_budget = max(0, state.coins[player] - state.safe_coin_threshold(player))
            for bundle in scored:
                cost = sum(max(0, state.operation_income(player, op)) * -1 for op in bundle.operations)
                if cost <= coin_budget or not bundle.operations:
                    return list(bundle.operations)
            return list(scored[0].operations)

        table = {
            "hold": hold,
            "random": random_policy,
            "greedy": greedy,
            "pragmatic": pragmatic,
        }
        if name not in table:
            raise ValueError(f"Unknown builtin rule policy: {name}. Options: {list(table)}")
        return table[name]

    @staticmethod
    def _load_module(module_path: str) -> DecideFn:
        spec = importlib.util.spec_from_file_location("antwar2_rule_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load rule module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in ("decide", "choose_operations", "choose_bundle"):
            fn = getattr(module, attr, None)
            if callable(fn):
                return fn
        raise AttributeError(f"{module_path} must define decide/choose_operations/choose_bundle")

    def _decide(self, backend_state, player: int) -> List[List[int]]:
        if self._decide_fn is None:
            return []
        result = self._decide_fn(backend_state, player)
        return _normalize_ops(result)

    def _save_artifacts(self, path: str):
        if self._module_path and os.path.exists(self._module_path):
            import shutil
            shutil.copyfile(self._module_path, os.path.join(path, "rule.py"))
        self.meta.extra["policy"] = self.policy_name
        if self._module_path:
            self.meta.extra["module_path"] = "rule.py"

    def _load_artifacts(self, path: str):
        local = os.path.join(path, "rule.py")
        policy = self.meta.extra.get("policy")
        if os.path.exists(local):
            self._module_path = local
            self._decide_fn = self._load_module(local)
        elif policy:
            self.policy_name = policy
            self._decide_fn = self._builtin(policy)


def _normalize_ops(result: Any) -> List[List[int]]:
    if result is None:
        return []
    out: List[List[int]] = []
    if isinstance(result, ActionBundle):
        result = result.operations
    if not isinstance(result, (list, tuple)):
        return out
    for item in result:
        if hasattr(item, "to_protocol_tokens"):
            out.append([int(t) for t in item.to_protocol_tokens()])
        elif isinstance(item, (list, tuple)):
            out.append([int(t) for t in item])
    return out
