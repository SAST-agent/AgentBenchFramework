"""Provider-neutral coding-agent act and version lifecycle records."""

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ActRecord:
    act_id: str
    provider: str
    status: str = "running"
    version_before: Optional[str] = None
    version_after: Optional[str] = None
    version_status: str = "pending"
    changed_files: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    token_accuracy: str = "unknown"
    elapsed_time_s: Optional[float] = None
    raw_output_ref: Optional[str] = None
    snapshot_content_hash: Optional[str] = None
    error: Optional[str] = None
    started_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None


class VersionedActRecorder:
    """Record act lifecycle without owning provider or workspace processes."""

    def __init__(self, emit: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.emit = emit
        self._active: Dict[str, ActRecord] = {}
        self._next_version = 1

    def begin_act(
        self, provider: str, version_before: Optional[str] = None, act_id: Optional[str] = None
    ) -> ActRecord:
        if version_before and version_before.startswith("v"):
            try:
                self._next_version = max(self._next_version, int(version_before[1:]) + 1)
            except ValueError:
                pass
        record = ActRecord(
            act_id=act_id or f"act_{uuid4().hex}",
            provider=provider,
            version_before=version_before,
        )
        if record.act_id in self._active:
            raise ValueError(f"act_id already active: {record.act_id}")
        self._active[record.act_id] = record
        return record

    def finish_act(
        self,
        act_id: str,
        status: str,
        version_after: Optional[str] = None,
        snapshot_content_hash: Optional[str] = None,
        changed_files: Optional[List[str]] = None,
        tool_call_count: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        token_accuracy: str = "unknown",
        elapsed_time_s: Optional[float] = None,
        raw_output_ref: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ActRecord:
        if act_id not in self._active:
            raise KeyError(f"unknown active act: {act_id}")
        if status not in {"completed", "failed", "timeout", "cancelled"}:
            raise ValueError("invalid act status")
        record = self._active.pop(act_id)
        record.status = status
        record.version_after = version_after
        if version_after is None and snapshot_content_hash is not None:
            record.version_after = f"v{self._next_version}"
            self._next_version += 1
        record.version_status = "available" if record.version_after is not None else "missing"
        record.snapshot_content_hash = snapshot_content_hash
        record.changed_files = list(changed_files or [])
        record.tool_call_count = int(tool_call_count)
        record.prompt_tokens = prompt_tokens
        record.completion_tokens = completion_tokens
        record.total_tokens = total_tokens
        record.token_accuracy = token_accuracy
        record.elapsed_time_s = elapsed_time_s
        record.raw_output_ref = raw_output_ref
        record.error = error
        record.finished_at = _now_iso()
        self._emit(record)
        return record

    def record_evaluation(self, act_id: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"event_type": "act_evaluation", "act_id": act_id, **evaluation}
        if self.emit is not None:
            self.emit(payload)
        return payload

    def _emit(self, record: ActRecord) -> None:
        if self.emit is None:
            return
        payload = asdict(record)
        payload["event_type"] = "coding_agent_act"
        self.emit(payload)
