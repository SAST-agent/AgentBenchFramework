"""
BudgetRecorder: tracks cost dimensions for an experiment.

Accumulates four cost axes required by the framework contract:
  - environment interactions (steps, episodes)
  - wall-clock seconds
  - API / token cost (prompt tokens, completion tokens, estimated USD)
  - GPU cost (gpu-seconds, estimated USD)

Designed as a self-contained recorder so it can be embedded inside the
existing Run class without changing its public constructor signature.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostEntry:
    label: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_cost_usd: float = 0.0
    gpu_seconds: float = 0.0
    gpu_cost_usd: float = 0.0
    interactions: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_cost_usd": round(self.api_cost_usd, 6),
            "gpu_seconds": round(self.gpu_seconds, 3),
            "gpu_cost_usd": round(self.gpu_cost_usd, 6),
            "interactions": self.interactions,
            "timestamp": self.timestamp,
        }


class BudgetRecorder:
    """Accumulates cost entries and produces a summary block."""

    def __init__(self,
                 token_price_per_1m_prompt: float = 0.0,
                 token_price_per_1m_completion: float = 0.0,
                 gpu_price_per_hour: float = 0.0):
        self.entries: List[CostEntry] = []
        self.token_price_per_1m_prompt = token_price_per_1m_prompt
        self.token_price_per_1m_completion = token_price_per_1m_completion
        self.gpu_price_per_hour = gpu_price_per_hour
        self._start: Optional[float] = None

    def start_clock(self):
        self._start = time.time()

    def stop_clock(self) -> float:
        if self._start is None:
            return 0.0
        return time.time() - self._start

    def log_cost(self,
                 label: str = "default",
                 prompt_tokens: int = 0,
                 completion_tokens: int = 0,
                 api_cost_usd: Optional[float] = None,
                 gpu_seconds: float = 0.0,
                 gpu_cost_usd: Optional[float] = None,
                 interactions: int = 0):
        if api_cost_usd is None:
            api_cost_usd = (
                prompt_tokens / 1_000_000.0 * self.token_price_per_1m_prompt
                + completion_tokens / 1_000_000.0 * self.token_price_per_1m_completion
            )
        if gpu_cost_usd is None:
            gpu_cost_usd = gpu_seconds / 3600.0 * self.gpu_price_per_hour
        entry = CostEntry(
            label=label,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            api_cost_usd=api_cost_usd,
            gpu_seconds=gpu_seconds,
            gpu_cost_usd=gpu_cost_usd,
            interactions=interactions,
            timestamp=time.time(),
        )
        self.entries.append(entry)

    def log_interactions(self, interactions: int, label: str = "env"):
        if self.entries and self.entries[-1].label == label:
            self.entries[-1].interactions += interactions
        else:
            self.log_cost(label=label, interactions=interactions)

    def summary(self) -> Dict[str, Any]:
        total_prompt = sum(e.prompt_tokens for e in self.entries)
        total_completion = sum(e.completion_tokens for e in self.entries)
        total_api = sum(e.api_cost_usd for e in self.entries)
        total_gpu_s = sum(e.gpu_seconds for e in self.entries)
        total_gpu_cost = sum(e.gpu_cost_usd for e in self.entries)
        total_interactions = sum(e.interactions for e in self.entries)
        wall_s = self.stop_clock()
        return {
            "total_interactions": total_interactions,
            "wall_seconds": round(wall_s, 2),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "api_cost_usd": round(total_api, 6),
            "gpu_seconds": round(total_gpu_s, 3),
            "gpu_cost_usd": round(total_gpu_cost, 6),
            "total_cost_usd": round(total_api + total_gpu_cost, 6),
            "entries": [e.to_dict() for e in self.entries],
        }
