"""Framework-owned coding-agent act orchestration."""

from typing import Any, Mapping, Optional

from agentbench_frame.tracking.iteration import ActRecord, VersionedActRecorder
from agentbench_frame.tracking.budget import BudgetLedger
from agentbench_frame.tracking.provider import ProviderAdapter, ProviderInvocation
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


class CodingAgentController:
    """Connect a provider adapter, act recorder, and local workspace snapshotter."""

    def __init__(
        self,
        provider: ProviderAdapter,
        recorder: Optional[VersionedActRecorder] = None,
        snapshotter: Optional[LocalWorkspaceSnapshotter] = None,
        budget: Optional[BudgetLedger] = None,
        budget_phase: str = "learning",
    ) -> None:
        if not isinstance(provider, ProviderAdapter):
            raise TypeError("provider must implement ProviderAdapter")
        self.provider = provider
        self.recorder = recorder or VersionedActRecorder()
        self.snapshotter = snapshotter or LocalWorkspaceSnapshotter()
        self.budget = budget
        self.budget_phase = budget_phase

    def run_act(
        self,
        context: Mapping[str, Any],
        workspace_root: Optional[str] = None,
        version_before: Optional[str] = None,
        act_id: Optional[str] = None,
        previous_manifest=None,
    ) -> ActRecord:
        """Run one provider invocation and retain its resulting version metadata."""
        act = self.recorder.begin_act(
            provider=self.provider.provider_name,
            version_before=version_before,
            act_id=act_id,
        )
        provider_context = dict(context)
        if workspace_root is not None:
            provider_context.setdefault("workspace_root", workspace_root)

        before = None
        snapshot_error = None
        if workspace_root is not None:
            try:
                before = previous_manifest or self.snapshotter.capture(workspace_root)
            except Exception as exc:
                snapshot_error = str(exc)

        try:
            invocation = self.provider.invoke(provider_context)
            if not isinstance(invocation, ProviderInvocation):
                raise TypeError("provider.invoke must return ProviderInvocation")
        except Exception as exc:
            invocation = ProviderInvocation(status="failed", error=str(exc))

        manifest = None
        if workspace_root is not None:
            try:
                manifest = self.snapshotter.capture(workspace_root, previous=before)
            except Exception as exc:
                snapshot_error = f"{snapshot_error}; {exc}" if snapshot_error else str(exc)

        error = invocation.error
        if snapshot_error:
            error = f"{error}; snapshot: {snapshot_error}" if error else f"snapshot: {snapshot_error}"
        usage = invocation.usage
        if self.budget is not None:
            self.budget.add(
                self.budget_phase,
                coding_agent_acts=1,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                time_s=invocation.elapsed_time_s,
            )
        return self.recorder.finish_act(
            act.act_id,
            status=invocation.status,
            snapshot_content_hash=manifest.content_hash if manifest else None,
            changed_files=manifest.changed_files if manifest else [],
            tool_call_count=invocation.tool_call_count,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            token_accuracy=usage.token_accuracy,
            elapsed_time_s=invocation.elapsed_time_s,
            raw_output_ref=invocation.raw_output_ref,
            error=error,
        )
