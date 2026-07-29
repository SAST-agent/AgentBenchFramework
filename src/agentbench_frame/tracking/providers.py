"""Concrete subprocess adapters for coding agents.

The adapters deliberately keep the provider boundary small: a prompt enters,
one invocation comes back, and the provider's own JSONL stream is optionally
written byte-for-byte to ``raw_output_path``.  The framework does not invent
token counts when a provider does not expose them.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Union

from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage


#: Type alias accepted by ``executable`` — a single command (string, split with
#: :func:`shlex.split`) or a pre-split argv list. Accepting argv lets callers
#: use ``[sys.executable, path_to_script]`` so provider tests run on Windows
#: (where ``#!`` script execution via the bare filename is not possible).
ExecutableLike = Union[str, Sequence[str]]


def _split_executable(executable: ExecutableLike) -> List[str]:
    """Normalize ``executable`` to a pre-split argv list.

    * ``str``  → :func:`shlex.split` (POSIX-style; equivalent on Windows when
                used as ``subprocess.run([argv0, argv1, ...])`` since we never
                shell out).
    * iterable/list → list(obj) verbatim.

    Keeps backward compatibility with the prior single-string form
    (``executable="codex"``) while enabling cross-platform argv pre-splitting
    (``executable=[sys.executable, '/abs/path/fake-codex.py']``), required so
    a real process can be invoked on Windows where ``WinError 193`` would
    otherwise be raised by the OS loader for a non-PE ``#!`` file.
    """
    if isinstance(executable, str):
        return list(shlex.split(executable, posix=os.name == "posix"))
    return list(executable)


def _lines(source: str | Iterable[str]) -> tuple[list[dict], int]:
    if isinstance(source, str):
        source = source.splitlines()
    records: list[dict] = []
    malformed = 0
    for line in source:
        if not str(line).strip():
            continue
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            malformed += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            malformed += 1
    return records, malformed


def _usage(value: Any) -> tuple[ProviderUsage, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return ProviderUsage(), {}
    prompt = value.get("input_tokens", value.get("prompt_tokens"))
    completion = value.get("output_tokens", value.get("completion_tokens"))
    prompt = int(prompt) if isinstance(prompt, (int, float)) else None
    completion = int(completion) if isinstance(completion, (int, float)) else None
    accuracy = "exact" if prompt is not None and completion is not None else "unknown"
    return (
        ProviderUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=(prompt + completion)
            if prompt is not None and completion is not None
            else None,
            token_accuracy=accuracy,
        ),
        dict(value),
    )


def parse_codex_jsonl(source: str | Iterable[str]) -> ProviderInvocation:
    """Parse the JSONL stream emitted by ``codex exec --json``."""

    records, malformed = _lines(source)
    thread_id = None
    usage = ProviderUsage()
    raw_usage: dict[str, Any] = {}
    tool_calls = 0
    error = None
    status = "failed"
    for record in records:
        event_type = record.get("type", "")
        if event_type == "thread.started":
            thread_id = record.get("thread_id")
        elif event_type == "turn.completed":
            status = "completed"
            usage, raw_usage = _usage(record.get("usage"))
        elif event_type in {"turn.failed", "error"}:
            status = "failed"
            error = record.get("message") or record.get("error") or str(record)
        item = record.get("item")
        item_type = item.get("type") if isinstance(item, Mapping) else None
        if item_type in {
            "command_execution", "mcp_tool_call", "web_search_call", "file_change"
        }:
            tool_calls += 1
    if not records and malformed:
        error = "provider emitted no valid JSONL events"
    metadata = {
        "thread_id": thread_id,
        "raw_event_count": len(records),
        "malformed_line_count": malformed,
        "event_types": [record.get("type", "unknown") for record in records],
        "raw_usage": raw_usage,
    }
    return ProviderInvocation(
        status=status,
        usage=usage,
        tool_call_count=tool_calls,
        error=error,
        metadata=metadata,
    )


def _count_claude_tools(record: Mapping[str, Any]) -> int:
    message = record.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list):
        return 0
    return sum(1 for block in content if isinstance(block, Mapping) and block.get("type") == "tool_use")


def parse_claude_stream_json(source: str | Iterable[str]) -> ProviderInvocation:
    """Parse the stream emitted by ``claude -p --output-format stream-json``."""

    records, malformed = _lines(source)
    session_id = None
    usage = ProviderUsage()
    raw_usage: dict[str, Any] = {}
    tool_calls = 0
    result_record = None
    error = None
    for record in records:
        event_type = record.get("type", "")
        if event_type == "system":
            session_id = record.get("session_id", session_id)
        tool_calls += _count_claude_tools(record)
        message = record.get("message")
        message_usage = message.get("usage") if isinstance(message, Mapping) else None
        if message_usage:
            usage, raw_usage = _usage(message_usage)
        if event_type == "result":
            result_record = record
            if record.get("usage"):
                usage, raw_usage = _usage(record.get("usage"))
            if record.get("is_error") or record.get("subtype") not in {None, "success"}:
                error = record.get("result") or record.get("error") or str(record)
    status = "completed" if result_record is not None and error is None else "failed"
    if result_record is None and not error:
        error = "provider stream did not contain a result event"
    metadata = {
        "session_id": session_id,
        "raw_event_count": len(records),
        "malformed_line_count": malformed,
        "event_types": [record.get("type", "unknown") for record in records],
        "raw_usage": raw_usage,
    }
    if result_record is not None:
        for key in ("num_turns", "duration_ms", "duration_api_ms", "total_cost_usd"):
            if key in result_record:
                metadata[key] = result_record[key]
    return ProviderInvocation(
        status=status,
        usage=usage,
        tool_call_count=tool_calls,
        error=error,
        metadata=metadata,
    )


class _SubprocessProvider:
    provider_name = "subprocess"
    parser = staticmethod(parse_codex_jsonl)

    def __init__(
        self,
        executable: ExecutableLike,
        timeout_s: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        # Canonical form is a pre-split argv list (list[str]). Backward
        # compat: a plain string (e.g. ``"codex"``) is split via shlex.
        self.executable: List[str] = _split_executable(executable)
        self.timeout_s = timeout_s
        self.env = dict(env) if env is not None else None
        self.extra_args = list(extra_args)

    def build_command(self, context: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    def invoke(self, context: Mapping[str, Any]) -> ProviderInvocation:
        command = self.build_command(context)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=context.get("workspace_root"),
                env=(None if self.env is None else {**os.environ, **self.env}),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            return ProviderInvocation(
                status="timeout",
                elapsed_time_s=elapsed,
                error=f"provider timed out after {self.timeout_s}s",
                metadata={"command": command, "stderr": str(exc.stderr or "")},
            )
        elapsed = time.monotonic() - started
        raw_output = completed.stdout or ""
        raw_path = context.get("raw_output_path")
        raw_ref = None
        if raw_path:
            target = Path(str(raw_path))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(raw_output, encoding="utf-8")
            raw_ref = str(target)
        result = self.parser(raw_output)
        result.elapsed_time_s = elapsed
        result.raw_output_ref = raw_ref
        result.metadata["command"] = command
        result.metadata["return_code"] = completed.returncode
        result.metadata["stderr"] = completed.stderr or ""
        if completed.returncode != 0:
            result.status = "failed"
            result.error = result.error or (completed.stderr.strip() or f"provider exited {completed.returncode}")
        return result


class CodexProvider(_SubprocessProvider):
    """Run the official Codex CLI in non-interactive JSONL mode."""

    provider_name = "codex"
    parser = staticmethod(parse_codex_jsonl)

    def __init__(self, executable: ExecutableLike = "codex",
                 sandbox: str = "workspace-write", **kwargs) -> None:
        super().__init__(executable, **kwargs)
        self.sandbox = sandbox

    def build_command(self, context: Mapping[str, Any]) -> list[str]:
        prompt = context.get("prompt", context.get("task"))
        if not prompt:
            raise ValueError("Codex provider requires context['prompt'] or context['task']")
        sandbox = context.get("sandbox", self.sandbox)
        return [*self.executable, "exec", "--json", "--sandbox", str(sandbox),
                *self.extra_args, str(prompt)]


class ClaudeCodeProvider(_SubprocessProvider):
    """Run Claude Code in print/stream-json mode with edit permissions."""

    provider_name = "claude-code"
    parser = staticmethod(parse_claude_stream_json)

    def __init__(
        self,
        executable: ExecutableLike = "claude",
        permission_mode: str = "acceptEdits",
        **kwargs,
    ) -> None:
        super().__init__(executable, **kwargs)
        self.permission_mode = permission_mode

    def build_command(self, context: Mapping[str, Any]) -> list[str]:
        prompt = context.get("prompt", context.get("task"))
        if not prompt:
            raise ValueError("Claude Code provider requires context['prompt'] or context['task']")
        permission_mode = context.get("permission_mode", self.permission_mode)
        command = [*self.executable, "-p", str(prompt), "--output-format", "stream-json",
                   "--verbose", "--permission-mode", str(permission_mode)]
        max_turns = context.get("max_turns")
        if max_turns is not None:
            command.extend(["--max-turns", str(max_turns)])
        command.extend(self.extra_args)
        return command
