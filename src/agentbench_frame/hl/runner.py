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
import signal
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
    edit_type: Optional[str]            # add_rule | reorder | parametrize | refactor | replace
                                       #   | utility | search | planner | consolidate | noop
                                       # None = "unclassified": the controller will
                                       #   diff-classify (noop vs a real edit).
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


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Reap the *whole* claude process tree on a timeout.

    ``subprocess``'s own timeout handling only kills the **direct** child; the
    ``claude`` CLI is a launcher that spawns a runtime grandchild, and that
    grandchild survives as an orphan (run ``hl-run-0730`` leaked a
    ``claude.exe`` PID that had to be ``taskkill``'d by hand). We launch the
    child in its own process group / Windows process group so the tree can be
    killed wholesale:

    - Windows: ``taskkill /F /T /PID`` (``/T`` = the whole subtree).
    - POSIX: ``os.killpg(os.getpgid(pid), SIGKILL)`` (works because we launched
      with ``start_new_session=True``).

    Both fall back to ``proc.kill()`` (the direct child) if the platform
    tree-kill finds no group / fails — so a timed-out child is always reaped
    even when the tree-kill misses. Best-effort: never raises.
    """
    pid = getattr(proc, "pid", None)
    if pid is not None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=5.0,
                )
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


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
        timeout = context.get("timeout", 300)
        started = _time.monotonic()
        # Launch in its own process group so a timeout can reap the WHOLE tree
        # (claude is a launcher; its runtime grandchild otherwise orphans — see
        # ``_kill_tree``). start_new_session is POSIX-only; on Windows the
        # CREATE_NEW_PROCESS_GROUP flag serves the analogous role and ``taskkill
        # /T`` walks the subtree regardless.
        popen_kwargs: Dict[str, Any] = dict(
            cwd=str(workspace),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(argv, **popen_kwargs)
        except FileNotFoundError:
            reason = f"claude CLI not found: {self.claude_path}"
            return AgentRunResult(edit_type="noop",
                                  error=reason, failure_reason=reason)

        # edit_type=None signals "unclassified" — the controller diff-classifies
        # the snapshot (noop vs a real edit) because the CLI gives us no
        # semantic label. FakeRunner declares a concrete edit_type and the
        # controller trusts it; only None triggers diff-based classification.
        edit_type: Optional[str] = None
        prompt_tokens = completion_tokens = total_tokens = None
        session_id = None
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Reap the whole tree (direct child + grandchildren), then drain
            # the pipes so the child is collected and we can parse any partial
            # result streamed before the timeout. claude emits ``result`` only
            # at the end, so on timeout there usually isn't one (tokens stay
            # None = unknown, never 0 — doc §13).
            _kill_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=10.0)
            except Exception:
                stdout, stderr = "", ""
            elapsed = _time.monotonic() - started
            reason = "claude CLI timed out"
            return AgentRunResult(
                edit_type="noop",
                error=reason, failure_reason=reason,
                time_s=elapsed,
            )
        elapsed = _time.monotonic() - started

        try:
            for line in (stdout or "").splitlines():
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
            reason = f"claude exited {proc.returncode}: {(stderr or '')[:500]}"
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


class ApiCodingRunner:
    """Drive the HL edit via raw provider API calls (no claude CLI).

    Hybrid depth: a bounded tool-use loop exposing read-only tools
    (``read_file`` / ``list_replays`` / ``read_replay``) plus the single edit
    ``write_agent_py`` (full-file rewrite). The loop stops when the model calls
    ``write_agent_py``, emits a final message with no tool call, or hits the
    step cap (``max_turns``). The edit is ``ast.parse``-validated before apply;
    a SyntaxError leaves the workspace untouched and yields an honest ``noop``
    plus a ``failure_reason``.

    Failure contract matches ``ClaudeCodeRunner``: API error / step-cap /
    parse-failure -> ``edit_type="noop"`` + stable ``failure_reason`` so the
    controller records it in the event stream. A clean decision not to edit is
    ``noop`` with ``failure_reason=None``.
    """

    _READ_TOOLS = ("read_file", "list_replays", "read_replay")

    def __init__(self, *, client, system_prompt: str,
                 max_turns: int = 6, max_tokens: int = 8192,
                 timeout: float = 300.0):
        self._client = client
        self._system = system_prompt
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._timeout = timeout

    def run(self, *, workspace: Path, context: Dict[str, Any]) -> AgentRunResult:
        import time as _time
        from agentbench_frame.hl.llm import SHARED_TOOLS
        prompt = context.get("prompt") or self._default_prompt(context)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        prompt_tok = completion_tok = total_tok = 0
        edit_applied = False
        failure_reason: Optional[str] = None
        turn_records: List[Dict[str, Any]] = []
        started = _time.monotonic()
        try:
            for _ in range(self._max_turns):
                remaining = self._timeout - (_time.monotonic() - started)
                if remaining <= 0:
                    failure_reason = "timeout"
                    break
                resp = self._client.complete(
                    system=self._system, messages=messages,
                    tools=SHARED_TOOLS, max_tokens=self._max_tokens, timeout=remaining)
                u = resp.usage
                prompt_tok += u.prompt_tokens or 0
                completion_tok += u.completion_tokens or 0
                total_tok += u.total_tokens or 0

                rec: Dict[str, Any] = {
                    "kind": "turn", "turn": len(turn_records) + 1,
                    "finish_reason": resp.finish_reason,
                    "usage": {"prompt": u.prompt_tokens,
                              "completion": u.completion_tokens,
                              "total": u.total_tokens},
                    "tool_calls": [{"name": tc.name, "id": tc.id,
                                    "args_preview": self._preview(tc.arguments)}
                                   for tc in resp.tool_calls],
                    "assistant_text_preview": (resp.text or "")[:300],
                    "write_attempt": None,
                    "tool_results": [],
                }

                if not resp.tool_calls:
                    messages.append({"role": "assistant", "content": resp.text})
                    turn_records.append(rec)
                    break  # model chose to stop -> clean no-op unless it edited

                messages.append({"role": "assistant", "content": resp.text,
                                 "tool_calls": resp.tool_calls})
                wrote = False
                for tc in resp.tool_calls:
                    if tc.name == "write_agent_py":
                        content = tc.arguments.get("content", "")
                        # A length-capped generation (finish_reason="length")
                        # cut the tool args mid-stream, so ``content`` is
                        # incomplete — possibly empty. Never apply a
                        # truncated/empty write: it silently corrupts
                        # agent.py (empty file parses as valid Python). Fail
                        # the act honestly; the workspace stays on the last
                        # good version and the next act retries.
                        if resp.finish_reason == "length":
                            rec["write_attempt"] = {
                                "called": True, "content_len": len(content),
                                "ast_ok": False,
                                "reason": "write_truncated (max_tokens hit mid-write)",
                            }
                            failure_reason = "write_truncated"
                            wrote = True
                            break
                        if not content.strip():
                            rec["write_attempt"] = {
                                "called": True, "content_len": len(content),
                                "ast_ok": False, "reason": "empty_write",
                            }
                            failure_reason = "empty_write"
                            wrote = True
                            break
                        ok, reason = self._apply_edit(content, workspace)
                        rec["write_attempt"] = {
                            "called": True, "content_len": len(content),
                            "ast_ok": ok, "reason": reason,
                        }
                        if ok:
                            edit_applied = True
                        else:
                            failure_reason = reason
                        wrote = True
                        break
                    result = self._dispatch(tc.name, tc.arguments, workspace, context)
                    messages.append({"role": "tool", "tool_name": tc.name,
                                     "tool_call_id": tc.id, "content": result})
                    rec["tool_results"].append(
                        {"name": tc.name, "preview": result[:300]})
                turn_records.append(rec)
                if wrote:
                    break
            else:
                failure_reason = "step_cap_exceeded"
        except Exception as e:  # API/network/auth errors
            failure_reason = f"api_error: {type(e).__name__}: {str(e)[:200]}"

        elapsed = _time.monotonic() - started
        transcript_path = self._write_transcript(
            turn_records, edit_applied, failure_reason, workspace, context)
        if edit_applied:
            return AgentRunResult(
                edit_type=None,  # unclassified -> controller diff-classifies
                prompt_tokens=prompt_tok or None,
                completion_tokens=completion_tok or None,
                total_tokens=total_tok or None,
                time_s=elapsed, transcript_path=transcript_path)
        return AgentRunResult(
            edit_type="noop", failure_reason=failure_reason,
            prompt_tokens=prompt_tok or None,
            completion_tokens=completion_tok or None,
            total_tokens=total_tok or None,
            time_s=elapsed, transcript_path=transcript_path)

    def _write_transcript(self, turn_records: List[Dict[str, Any]],
                          edit_applied: bool, failure_reason: Optional[str],
                          workspace: Path, context: Dict[str, Any]
                          ) -> Optional[str]:
        """Write a per-act JSONL tool-call transcript beside the round root.

        One ``{kind: turn}`` record per tool-use turn (tool calls emitted,
        any ``write_agent_py`` attempt, read-tool result previews) plus a
        final ``{kind: terminal}`` record. Lets a run be diagnosed without a
        live debugger: did the model call ``write_agent_py``? did it loop on
        reads? did ``ast.parse`` reject the edit? ``transcript_dir`` in context
        overrides the default ``<workspace.parent>/transcripts``.
        Best-effort: any IO error -> None (never breaks the act loop).
        """
        import json as _json
        tdir = Path(context.get("transcript_dir")
                    or (workspace.parent / "transcripts"))
        try:
            tdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        act_id = context.get("act_id") or "last-run"
        path = tdir / f"{act_id}.jsonl"
        try:
            with path.open("w", encoding="utf-8") as f:
                for rec in turn_records:
                    f.write(_json.dumps(rec) + "\n")
                f.write(_json.dumps({
                    "kind": "terminal", "edit_applied": edit_applied,
                    "failure_reason": failure_reason,
                    "turns": len(turn_records),
                }) + "\n")
        except OSError:
            return None
        return str(path)

    @staticmethod
    def _preview(args: Any) -> str:
        import json as _json
        try:
            return _json.dumps(args)[:300]
        except (TypeError, ValueError):
            return str(args)[:300]

    # ---- edit + tool dispatch ----

    def _apply_edit(self, content: str, workspace: Path):
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"syntax_error: {e.msg} (line {e.lineno})"
        (workspace / "agent.py").write_text(content, encoding="utf-8")
        return True, None

    def _dispatch(self, name: str, args: Dict[str, Any],
                  workspace: Path, context: Dict[str, Any]) -> str:
        if name == "read_file":
            return self._tool_read_file(args.get("path", ""), workspace)
        if name == "list_replays":
            return self._tool_list_replays(context)
        if name == "read_replay":
            return self._tool_read_replay(args.get("id", ""), context)
        return f"error: unknown tool {name!r}"

    def _tool_read_file(self, rel: str, workspace: Path) -> str:
        rel = (rel or "").strip()
        if not rel:
            return "error: path required"
        target = (workspace / rel).resolve()
        try:
            target.relative_to(workspace.resolve())
        except ValueError:
            return "error: path escapes workspace"
        if not target.is_file():
            return f"error: not found: {rel}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:20000]  # bound context

    def _tool_list_replays(self, context: Dict[str, Any]) -> str:
        import json as _json
        replays = context.get("replays") or []
        if not replays:
            return "no replays available"
        return _json.dumps(replays[:20])

    def _tool_read_replay(self, rid: str, context: Dict[str, Any]) -> str:
        replays_dir = context.get("replays_dir")
        if not replays_dir:
            return "error: replays not available for this run"
        p = Path(replays_dir) / rid
        if not p.is_file():
            return f"error: replay not found: {rid}"
        return p.read_text(encoding="utf-8", errors="replace")[:20000]

    @staticmethod
    def _default_prompt(context: Dict[str, Any]) -> str:
        goal = context.get("goal", "Improve the agent's win rate.")
        resources = context.get("resources_summary", "")
        return (f"{goal}\n\n{resources}\n\nInspect the workspace and replays, "
                f"then call write_agent_py with the improved complete agent.py.")
