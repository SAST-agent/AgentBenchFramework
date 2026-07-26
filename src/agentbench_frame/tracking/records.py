"""
Dataclasses for all tracking record types.

Records are the atomic units of observability: each step, episode, and
resource sample is written as a JSON line to a tracking log.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class StepRecord:
    """One environment step."""
    event: str = "step"
    timestamp: float = 0.0
    episode: int = 0
    step: int = 0
    action: Any = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    wall_time_ms: float = 0.0
    agent_time_ms: float = 0.0
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None
    termination_reason: Optional[str] = None
    actor: Optional[str] = None


@dataclass
class EpisodeRecord:
    """Summary of a complete episode."""
    event: str = "episode"
    timestamp: float = 0.0
    episode: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    winner: int = -1
    wall_time_s: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    """Evaluation run summary."""
    event: str = "eval"
    timestamp: float = 0.0
    eval_index: int = 0
    episodes: int = 0
    win_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRecord:
    """System resource snapshot (CPU, RSS, GPU)."""
    event: str = "resource"
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    gpu_util: Optional[float] = None
    gpu_mem_mb: Optional[float] = None


@dataclass
class LogRecord:
    """Arbitrary key-value log event."""
    event: str = "log"
    timestamp: float = 0.0
    episode: Optional[int] = None
    step: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMeta:
    """Metadata about the run (written to run.toml)."""
    run_id: str = ""
    game: str = ""
    agent: str = ""
    run_type: str = "eval"
    created: str = ""
    git_commit: str = ""
    agent_type: str = ""
    runner: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    total_steps: int = 0
    total_episodes: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
