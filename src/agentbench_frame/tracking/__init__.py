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
from agentbench_frame.tracking.budget import BudgetLedger
from agentbench_frame.tracking.iteration import ActRecord, VersionedActRecorder
from agentbench_frame.tracking.provider import ProviderAdapter, ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter, WorkspaceManifest
from agentbench_frame.tracking.controller import CodingAgentController

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
    "BudgetLedger",
    "ActRecord",
    "VersionedActRecorder",
    "ProviderAdapter",
    "ProviderInvocation",
    "ProviderUsage",
    "LocalWorkspaceSnapshotter",
    "WorkspaceManifest",
    "CodingAgentController",
]
