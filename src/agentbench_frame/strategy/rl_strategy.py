"""
RLStrategy: adapts a reinforcement-learning policy to the unified interface.

The policy maps a board/stat tensor plus an action mask to an action index
into the env's action-bundle list. When torch is unavailable the strategy
gracefully degrades to a greedy baseline so the rest of the pipeline
(eval / iteration) still runs end to end.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from agentbench_frame.strategy.base import BaseStrategy, StrategyMeta
from agentbench_frame.env._antwar2.utils.actions import ActionCatalog
from agentbench_frame.env._antwar2.utils.features import FeatureExtractor
from agentbench_frame.env._antwar2.utils.constants import MAX_ACTIONS

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False


if HAS_TORCH:
    class AntWar2Policy(nn.Module):
        def __init__(self, board_channels: int = 28, map_size: int = 19,
                     stats_dim: int = 42, num_actions: int = MAX_ACTIONS):
            super().__init__()
            self.num_actions = num_actions
            self.board_conv = nn.Sequential(
                nn.Conv2d(board_channels, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d((4, 4)),
            )
            self.stats_head = nn.Sequential(
                nn.Linear(stats_dim, 64), nn.ReLU(),
            )
            self.actor = nn.Linear(32 * 4 * 4 + 64, num_actions)
            self.critic = nn.Linear(32 * 4 * 4 + 64, 1)

        def forward(self, board, stats, mask=None):
            feat = torch.cat([self.board_conv(board).flatten(1), self.stats_head(stats)], dim=1)
            logits = self.actor(feat)
            if mask is not None:
                logits = logits.masked_fill(mask == 0, -1e9)
            value = self.critic(feat).squeeze(-1)
            return logits, value

        def act(self, board, stats, mask, deterministic=False):
            logits, value = self.forward(board, stats, mask)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                idx = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()
            return idx, torch.log_softmax(logits, dim=-1).gather(-1, idx.unsqueeze(-1)).squeeze(-1), value


class RLStrategy(BaseStrategy):
    kind = "rl"

    def __init__(self,
                 name: str = "RLStrategy",
                 policy: Any = None,
                 checkpoint_path: Optional[str] = None,
                 deterministic: bool = True,
                 fallback: str = "greedy",
                 meta: Optional[StrategyMeta] = None,
                 **kwargs):
        super().__init__(name=name, meta=meta)
        self.deterministic = deterministic
        self.fallback = fallback
        self._feature = FeatureExtractor(max_actions=MAX_ACTIONS)
        self._catalog = ActionCatalog(max_actions=MAX_ACTIONS, feature_extractor=self._feature)
        self.policy = policy
        self.checkpoint_path = checkpoint_path
        self._device = None
        if self.policy is None and HAS_TORCH:
            self.policy = AntWar2Policy()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy.to(self._device)
        if self.policy is not None and checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        self.meta.kind = self.kind

    def _load_checkpoint(self, path: str):
        if not HAS_TORCH or self.policy is None:
            return
        state = torch.load(path, map_location=self._device)
        self.policy.load_state_dict(state)

    def _decide(self, backend_state, player: int) -> List[List[int]]:
        bundles = self._catalog.build(backend_state, player) or []
        if not bundles:
            return []
        mask = self._catalog.action_mask(bundles)
        if not HAS_TORCH or self.policy is None or not HAS_NUMPY:
            return self._fallback_pick(bundles)
        board = self._feature.encode_board(backend_state, player)
        stats = self._feature.encode_stats(backend_state, player)
        with torch.no_grad():
            b = torch.from_numpy(board).float().unsqueeze(0).to(self._device)
            s = torch.from_numpy(stats).float().unsqueeze(0).to(self._device)
            m = torch.from_numpy(mask.astype("float32")).unsqueeze(0).to(self._device)
            idx, _, _ = self.policy.act(b, s, m, deterministic=self.deterministic)
            idx = int(idx.item())
        idx = min(idx, len(bundles) - 1)
        return _ops_from_bundle(bundles[idx])

    def _fallback_pick(self, bundles) -> List[List[int]]:
        from agentbench_frame.strategy.rule_strategy import _normalize_ops
        if self.fallback == "random":
            import random
            return _normalize_ops(random.choice(bundles))
        return _normalize_ops(max(bundles, key=lambda b: b.score))

    def _save_artifacts(self, path: str):
        if HAS_TORCH and self.policy is not None:
            torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
            self.meta.extra["checkpoint"] = "policy.pt"
        else:
            self.meta.extra["fallback"] = self.fallback

    def _load_artifacts(self, path: str):
        ck = self.meta.extra.get("checkpoint")
        if ck:
            self.checkpoint_path = os.path.join(path, ck)
            if HAS_TORCH and self.policy is None:
                self.policy = AntWar2Policy()
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.policy.to(self._device)
            self._load_checkpoint(self.checkpoint_path)


def _ops_from_bundle(bundle) -> List[List[int]]:
    out: List[List[int]] = []
    for op in getattr(bundle, "operations", ()):
        if hasattr(op, "to_protocol_tokens"):
            out.append([int(t) for t in op.to_protocol_tokens()])
    return out
