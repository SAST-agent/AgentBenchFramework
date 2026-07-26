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

__all__ = [
    "HLEventWriter",
    "read_events",
    "PUBLIC_FIELDS",
    "KNOWN_EVENT_TYPES",
    "SCHEMA_VERSION",
]
