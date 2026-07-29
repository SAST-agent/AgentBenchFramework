"""
Coding-agent runner interface — what the HL controller calls to mutate the
codebase.

The controller is agnostic to *how* the coding agent decides what to edit;
it only requires a runner that takes the workspace + a context dict and
returns what it did. Two implementations:

- ``FakeRunner``: scripted edits for tests/debugging. Lets the controller's
  act→snapshot→eval→measure→emit loop run end-to-end without a real LLM CLI.
  Not throwaway — it stays as the permanent test double.
- ``ClaudeCodeRunner``: shells out to the ``claude`` CLI against the workspace,
  parses its JSONL output. Real HL iteration. Requires the CLI installed and
  authenticated (the user has Claude Code installed).

Token / time usage: a real runner reports prompt/completion tokens from the
CLI's usage events. A fake runner has none and reports ``None`` (unknown),
never 0 (doc §13: unknown -> unknown, not 0).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class AgentRunResult:
    """What a coding-agent run produced.

    ``failure_reason`` is the run-classification failure: it is set when the
    coding agent could not complete its edit (timeout, CLI not found,
    non-zero exit). ``edit_type`` stays ``"noop"`` on those paths — the edit
    classification is "no change" — but ``failure_reason`` distinguishes a
    *clean* no-op (agent chose not to edit) from a *failed* run. The
    controller writes ``failure_reason`` into the ``version`` and ``budget``
    events so a silent timeout is never mistaken for a clean no-op in the
    research stream. ``None`` (unknown/none) on success.
    """
    edit_type: str                       # add_rule | reorder | parametrize | refactor | replace
                                       #   | utility | search | planner | consolidate | noop
    files_touched: List[str] = field(default_factory=list)
    prompt_tokens: Optional[int] = None      # None = unknown, not 0
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    time_s: Optional[float] = None
    error: Optional[str] = None               # legacy alias of failure_reason
    failure_reason: Optional[str] = None      # set if the run failed (timeout/not-found/non-zero)
    session_id: Optional[str] = None          # claude session id (opaque UUID)
    transcript_path: Optional[str] = None     # ~/.claude/projects/<slug>/<session_id>.jsonl


def _resolve_transcript_path(session_id: Optional[str]) -> Optional[str]:
    """Locate claude's transcript for a session by its globally-unique id.

    Claude writes one transcript per session to
    ``~/.claude/projects/<cwd-slug>/<session_id>.jsonl``. The cwd-slug is
    claude-internal (non-alphanumeric -> ``-``); rather than hand-roll it we
    glob ``~/.claude/projects/*/<session_id>.jsonl`` — the session_id is
    globally unique, so at most one match (``None`` if the file was rotated or
    the home isn't a real claude home, e.g. in tests). Recording this path in
    the run output lets each act's claude history be located without guessing
    by mtime.
    """
    if not session_id:
        return None
    try:
        matches = sorted(
            (Path.home() / ".claude" / "projects").glob(f"*/{session_id}.jsonl")
        )
    except OSError:
        return None
    return str(matches[0]) if matches else None


@runtime_checkable
class CodingAgentRunner(Protocol):
    """A coding agent that edits the workspace in place."""

    def run(self, *, workspace: Path, context: Dict[str, Any]
            ) -> AgentRunResult: ...


class FakeRunner:
    """A scripted coding agent for testing/debugging.

    ``transform(workspace)`` is called with the live workspace path; it may
    edit files however it likes. The runner reports the declared ``edit_type``
    and (best-effort) the files that changed by diffing the tree before/after.
    """

    def __init__(self, *, transform: Callable[[Path], None],
                 edit_type: str = "noop", session_id: Optional[str] = None):
        self._transform = transform
        self._edit_type = edit_type
        self._session_id = session_id

    def run(self, *, workspace: Path, context: Dict[str, Any]
            ) -> AgentRunResult:
        before = self._snapshot_files(workspace)
        self._transform(workspace)
        after = self._snapshot_files(workspace)
        touched = sorted(set(after) - set(before) |
                         {f for f in (set(before) & set(after))
                          if before[f] != after[f]})
        return AgentRunResult(
            edit_type=self._edit_type,
            files_touched=touched,
            prompt_tokens=None,  # fake -> unknown, not 0
            completion_tokens=None,
            total_tokens=None,
            time_s=0.0,
            session_id=self._session_id,
            transcript_path=_resolve_transcript_path(self._session_id),
        )

    @staticmethod
    def _snapshot_files(root: Path) -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                full = Path(dp) / fn
                rel = full.relative_to(root).as_posix()
                try:
                    out[rel] = full.read_bytes()
                except OSError:
                    continue
        return out


class ClaudeCodeRunner:
    """Run the real ``claude`` CLI against the workspace.

    Constructs even if the claude binary isn't found (fails on ``run()``
    instead) so importing this module never breaks in test environments
    without the CLI.

    Real-CLI knobs:
    - ``system_prompt`` → ``--append-system-prompt`` (the HL role + rules).
    - ``permission_mode`` → ``--permission-mode`` (default ``acceptEdits`` so
      the non-interactive run can edit files without prompting). Use
      ``bypassPermissions`` (= ``--dangerously-skip-permissions``) for a
      fully autonomous loop, but only when you trust the sandbox.
    - ``model`` → ``--model``.
    """

    def __init__(self, *, claude_path: str = "claude",
                 extra_args: Optional[List[str]] = None,
                 print_format: str = "json",
                 system_prompt: Optional[str] = None,
                 permission_mode: str = "acceptEdits",
                 dangerously_skip_permissions: bool = False,
                 model: Optional[str] = None):
        self.claude_path = claude_path
        self.extra_args = list(extra_args or [])
        self.print_format = print_format
        self.system_prompt = system_prompt
        self.permission_mode = permission_mode
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.model = model

    def _build_argv(self, prompt: str) -> List[str]:
        argv = [self.claude_path, "-p", prompt,
                "--output-format", self.print_format]
        if self.system_prompt:
            argv += ["--append-system-prompt", self.system_prompt]
        if self.dangerously_skip_permissions:
            argv.append("--dangerously-skip-permissions")
        elif self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        if self.model:
            argv += ["--model", self.model]
        argv.extend(self.extra_args)
        return argv

    def run(self, *, workspace: Path, context: Dict[str, Any]
            ) -> AgentRunResult:
        import json as _json
        import time as _time
        prompt = context.get("prompt") or self._default_prompt(context)
        argv = self._build_argv(prompt)
        started = _time.monotonic()
        try:
            proc = subprocess.run(
                argv, cwd=str(workspace),
                capture_output=True, text=True, timeout=context.get("timeout", 300),
            )
        except FileNotFoundError:
            reason = f"claude CLI not found: {self.claude_path}"
            return AgentRunResult(edit_type="noop",
                                  error=reason, failure_reason=reason)
        except subprocess.TimeoutExpired:
            reason = "claude CLI timed out"
            return AgentRunResult(edit_type="noop",
                                  error=reason, failure_reason=reason)
        elapsed = _time.monotonic() - started

        # The --output-format json stream yields one JSON object per line.
        edit_type = "refactor"  # default classification; the controller reclassifies via diff
        prompt_tokens = completion_tokens = total_tokens = None
        session_id = None
        try:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                evt = _json.loads(line)
                if evt.get("type") == "result":
                    usage = evt.get("usage") or {}
                    prompt_tokens = usage.get("input_tokens")
                    completion_tokens = usage.get("output_tokens")
                    total_tokens = usage.get("total_tokens")
                    session_id = evt.get("session_id")
        except _json.JSONDecodeError:
            pass

        if proc.returncode != 0:
            reason = f"claude exited {proc.returncode}: {proc.stderr[:500]}"
            return AgentRunResult(
                edit_type="noop",
                error=reason,
                failure_reason=reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                time_s=elapsed,
            )
        return AgentRunResult(
            edit_type=edit_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_s=elapsed,
            session_id=session_id,
            transcript_path=_resolve_transcript_path(session_id),
        )

    @staticmethod
    def _default_prompt(context: Dict[str, Any]) -> str:
        goal = context.get("goal", "Improve the agent's win rate.")
        resources = context.get("resources_summary", "")
        return (f"{goal}\n\nMatch history & replays are available in the "
                f"workspace context.\n{resources}\n\nEdit the agent files in "
                f"place to improve performance. Do not touch the manifest.toml "
                f"entrypoint.")
