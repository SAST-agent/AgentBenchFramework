"""
Transparent observability and experiment tracking.

Provides:
- Run: Central experiment-run coordinator
- JSONLWriter: Buffered JSONL event writer
- TrackedEnv / TimedAgent: Drop-in wrappers for step/act telemetry
- ResourceSampler: Background CPU/RSS/GPU sampling
- Record dataclasses: StepRecord, EpisodeRecord, EvalRecord, ResourceRecord
"""

from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.records import (
    StepRecord,
    EpisodeRecord,
    EvalRecord,
    ResourceRecord,
    LogRecord,
    RunMeta,
)
from agentbench_frame.tracking.writer import JSONLWriter
from agentbench_frame.tracking.sampler import ResourceSampler
from agentbench_frame.tracking.wrappers import TrackedEnv, TimedAgent
from agentbench_frame.tracking.budget import BudgetRecorder

__all__ = [
    "Run",
    "StepRecord",
    "EpisodeRecord",
    "EvalRecord",
    "ResourceRecord",
    "LogRecord",
    "RunMeta",
    "JSONLWriter",
    "ResourceSampler",
    "TrackedEnv",
    "TimedAgent",
    "BudgetRecorder",
]
