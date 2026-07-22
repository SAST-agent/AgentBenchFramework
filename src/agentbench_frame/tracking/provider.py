"""Provider-neutral coding-agent invocation contracts.

The concrete process adapters live in :mod:`tracking.providers`.  This module
only contains the stable data boundary consumed by the controller so callers
can also provide their own in-process adapter.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable


@dataclass
class ProviderUsage:
    """First-hand provider usage; ``None`` means the provider did not expose it."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_accuracy: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_tokens is None and self.prompt_tokens is not None and self.completion_tokens is not None:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.token_accuracy not in {"exact", "estimated", "unknown"}:
            raise ValueError("token_accuracy must be exact, estimated, or unknown")


@dataclass
class ProviderInvocation:
    """Provider result consumed by the framework controller."""

    status: str
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    tool_call_count: int = 0
    elapsed_time_s: Optional[float] = None
    raw_output_ref: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "timeout", "cancelled"}:
            raise ValueError("invalid provider invocation status")
        if self.tool_call_count < 0:
            raise ValueError("tool_call_count must be non-negative")
        if self.elapsed_time_s is not None and self.elapsed_time_s < 0:
            raise ValueError("elapsed_time_s must be non-negative")


@runtime_checkable
class ProviderAdapter(Protocol):
    """The only provider boundary required by the framework."""

    provider_name: str

    def invoke(self, context: Mapping[str, Any]) -> ProviderInvocation:
        """Run one complete coding-agent invocation and return raw usage metadata."""
        ...
