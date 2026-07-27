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
    """What a coding-agent run produced."""
    edit_type: str                       # add_rule | reorder | parametrize | refactor | replace | noop
    files_touched: List[str] = field(default_factory=list)
    prompt_tokens: Optional[int] = None      # None = unknown, not 0
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    time_s: Optional[float] = None
    error: Optional[str] = None               # set if the run failed


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
                 edit_type: str = "noop"):
        self._transform = transform
        self._edit_type = edit_type

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
    """

    def __init__(self, *, claude_path: str = "claude",
                 extra_args: Optional[List[str]] = None,
                 print_format: str = "json"):
        self.claude_path = claude_path
        self.extra_args = list(extra_args or [])
        self.print_format = print_format

    def run(self, *, workspace: Path, context: Dict[str, Any]
            ) -> AgentRunResult:
        import json as _json
        import time as _time
        prompt = context.get("prompt") or self._default_prompt(context)
        argv = [self.claude_path, "-p", prompt, "--output-format", self.print_format]
        argv.extend(self.extra_args)
        started = _time.monotonic()
        try:
            proc = subprocess.run(
                argv, cwd=str(workspace),
                capture_output=True, text=True, timeout=context.get("timeout", 300),
            )
        except FileNotFoundError:
            return AgentRunResult(edit_type="noop",
                                  error=f"claude CLI not found: {self.claude_path}")
        except subprocess.TimeoutExpired:
            return AgentRunResult(edit_type="noop", error="claude CLI timed out")
        elapsed = _time.monotonic() - started

        # The --output-format json stream yields one JSON object per line.
        edit_type = "refactor"  # default classification; the controller reclassifies via diff
        prompt_tokens = completion_tokens = total_tokens = None
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
        except _json.JSONDecodeError:
            pass

        if proc.returncode != 0:
            return AgentRunResult(
                edit_type="noop",
                error=f"claude exited {proc.returncode}: {proc.stderr[:500]}",
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
        )

    @staticmethod
    def _default_prompt(context: Dict[str, Any]) -> str:
        goal = context.get("goal", "Improve the agent's win rate.")
        resources = context.get("resources_summary", "")
        return (f"{goal}\n\nMatch history & replays are available in the "
                f"workspace context.\n{resources}\n\nEdit the agent files in "
                f"place to improve performance. Do not touch the manifest.toml "
                f"entrypoint.")
