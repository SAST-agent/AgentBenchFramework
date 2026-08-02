"""Heuristic-Learning (HL) iteration layer.

Defines the interface an external coding agent iterates against:
- ``HLCodebase`` — the versioned, agent-editable artifact (content-hash store)
- ``CodebasePolicy`` / ``EditType`` — edit classification (enforcement deferred)
- ``HLDistribution`` — the shared measurement channel (epsilon-smoothed)
- ``BenchmarkSpec`` / ``ReferenceStateSet`` — frozen eval + KL reference
- ``HLResources`` — read-only views handed to the coding agent
- ``HLIterationController`` — owns the act → snapshot → eval → measure → emit loop
- ``HLEventWriter`` — public-schema append-only event log
"""
from agentbench_frame.hl.events import (
    HLEventWriter,
    read_events,
    PUBLIC_FIELDS,
    KNOWN_EVENT_TYPES,
    SCHEMA_VERSION,
)
from agentbench_frame.hl.manifest import HLManifest, load_manifest
from agentbench_frame.hl.codebase import (
    HLCodebase,
    VersionHandle,
    StructuredDiff,
)
from agentbench_frame.hl.distribution import (
    enumerate_legal_actions,
    epsilon_smoothed_distribution,
    policy_kl,
    local_policy_kl_trace,
    PolicyKLPoint,
    ok_kl_values,
    LegalActionSet,
    FINISH,
)
from agentbench_frame.hl.adapter import stage_candidate, candidate_command
from agentbench_frame.hl.reference import (
    BenchmarkSpec,
    ReferenceSample,
    ReferenceStateSet,
)
from agentbench_frame.hl.resources import (
    MatchHistoryView,
    ReplayView,
    VersionDiffView,
)
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction
from agentbench_frame.hl.runner import (
    CodingAgentRunner,
    FakeRunner,
    ClaudeCodeRunner,
    AgentRunResult,
)
from agentbench_frame.hl.controller import HLIterationController

__all__ = [
    "HLEventWriter",
    "read_events",
    "PUBLIC_FIELDS",
    "KNOWN_EVENT_TYPES",
    "SCHEMA_VERSION",
    "HLManifest",
    "load_manifest",
    "HLCodebase",
    "VersionHandle",
    "StructuredDiff",
    "enumerate_legal_actions",
    "epsilon_smoothed_distribution",
    "policy_kl",
    "local_policy_kl_trace",
    "PolicyKLPoint",
    "ok_kl_values",
    "LegalActionSet",
    "FINISH",
    "stage_candidate",
    "candidate_command",
    "BenchmarkSpec",
    "ReferenceSample",
    "ReferenceStateSet",
    "MatchHistoryView",
    "ReplayView",
    "VersionDiffView",
    "ReferenceProbe",
    "EmittedAction",
    "CodingAgentRunner",
    "FakeRunner",
    "ClaudeCodeRunner",
    "AgentRunResult",
    "HLIterationController",
]
