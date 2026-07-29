"""Tests for the real-CLI knobs on ClaudeCodeRunner.

No real ``claude`` binary is invoked — ``subprocess.Popen`` is monkeypatched to
capture the argv and return a canned proc whose ``communicate`` yields a
canned JSONL result line. (The runner uses ``Popen`` + ``communicate`` so it
owns the child PID and can reap the whole process tree on timeout — see
``test_timeout_kills_process_tree``.)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agentbench_frame.hl.runner import ClaudeCodeRunner


class _FakeProc:
    """Stand-in for a ``subprocess.Popen`` object. ``communicate`` may be
    scripted to raise ``TimeoutExpired`` on the first call (to drive the
    timeout path); a second ``communicate`` then returns the drained output
    after the (fake) kill."""

    def __init__(self, stdout="", stderr="", returncode=0, pid=99999,
                 comm_exc=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self._comm_exc = comm_exc
        self._comm_calls = 0
        self.killed = False

    def communicate(self, timeout=None):
        self._comm_calls += 1
        if self._comm_calls == 1 and self._comm_exc is not None:
            # Simulate the timeout: claude is still running, no result yet.
            self.returncode = None
            raise self._comm_exc
        return (self.stdout, self.stderr)

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = -9  # killed

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _patch_popen(monkeypatch, capture, proc):
    def fake_popen(argv, **kwargs):
        capture["argv"] = list(argv)
        capture["cwd"] = kwargs.get("cwd")
        return proc
    monkeypatch.setattr("agentbench_frame.hl.runner.subprocess.Popen", fake_popen)


def test_argv_includes_system_prompt_and_permission_mode(monkeypatch, tmp_path):
    cap = {}
    _patch_popen(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner(system_prompt="ROLE", permission_mode="acceptEdits")
    r.run(workspace=tmp_path, context={"prompt": "do thing", "timeout": 10})
    assert cap["argv"][0] == "claude"
    assert "-p" in cap["argv"]
    assert cap["argv"][cap["argv"].index("-p") + 1] == "do thing"
    assert "--append-system-prompt" in cap["argv"]
    assert cap["argv"][cap["argv"].index("--append-system-prompt") + 1] == "ROLE"
    assert "--permission-mode" in cap["argv"]
    assert cap["argv"][cap["argv"].index("--permission-mode") + 1] == "acceptEdits"


def test_dangerously_skip_permissions_replaces_permission_mode(monkeypatch, tmp_path):
    cap = {}
    _patch_popen(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner(dangerously_skip_permissions=True)
    r.run(workspace=tmp_path, context={"prompt": "go"})
    assert "--dangerously-skip-permissions" in cap["argv"]
    assert "--permission-mode" not in cap["argv"]


def test_model_flag_when_set(monkeypatch, tmp_path):
    cap = {}
    _patch_popen(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner(model="claude-fable-5")
    r.run(workspace=tmp_path, context={"prompt": "go"})
    assert "--model" in cap["argv"]
    assert cap["argv"][cap["argv"].index("--model") + 1] == "claude-fable-5"


def test_usage_parsed_from_result_event(monkeypatch, tmp_path):
    cap = {}
    result_line = json.dumps({
        "type": "result",
        "usage": {"input_tokens": 1234, "output_tokens": 56,
                  "total_tokens": 1290},
    })
    _patch_popen(monkeypatch, cap, _FakeProc(stdout=result_line + "\n",
                                             returncode=0))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.error is None
    assert res.prompt_tokens == 1234
    assert res.completion_tokens == 56
    assert res.total_tokens == 1290
    assert res.time_s is not None


def test_missing_binary_returns_error_result(monkeypatch, tmp_path):
    def boom(argv, **kwargs):
        raise FileNotFoundError(2, "no such file")
    monkeypatch.setattr("agentbench_frame.hl.runner.subprocess.Popen", boom)
    r = ClaudeCodeRunner(claude_path="claude-missing")
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.error is not None
    assert "claude-missing" in res.error
    assert res.edit_type == "noop"
    # failure_reason is what the controller writes to events; it must mirror
    # error on this failure path so a missing CLI is visible in the stream.
    assert res.failure_reason == res.error


def test_nonzero_exit_with_stderr(monkeypatch, tmp_path):
    cap = {}
    _patch_popen(monkeypatch, cap, _FakeProc(stdout="", stderr="boom: bad",
                                              returncode=2))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.error is not None
    assert "2" in res.error
    assert res.failure_reason == res.error


def test_timeout_sets_failure_reason(monkeypatch, tmp_path):
    """The TimeoutExpired path sets failure_reason so a silent timeout is
    visible in the event stream (spec scenario: claude CLI times out)."""
    proc = _FakeProc(stdout="", returncode=0,
                     comm_exc=subprocess.TimeoutExpired(cmd=["claude"], timeout=1))
    _patch_popen(monkeypatch, {}, proc)
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go", "timeout": 1})
    assert res.edit_type == "noop"
    assert res.failure_reason == "claude CLI timed out"


def test_timeout_kills_process_tree(monkeypatch, tmp_path):
    """Regression for run hl-run-0730: on a claude-CLI timeout the runner
    must reap the child (``proc.kill`` invoked) — previously the timed-out
    ``claude`` process leaked (a grandchild survived and had to be
    taskkill'd by hand). Also asserts the stable failure_reason is written."""
    proc = _FakeProc(stdout="", returncode=0,
                     comm_exc=subprocess.TimeoutExpired(cmd=["claude"], timeout=1))
    _patch_popen(monkeypatch, {}, proc)
    # _kill_tree falls back to proc.kill when the platform tree-kill (taskkill
    # / killpg) finds no real process — assert that fallback fired.
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go", "timeout": 1})
    assert proc.killed is True
    assert res.failure_reason == "claude CLI timed out"


def test_default_prompt_used_when_context_omits_prompt(monkeypatch, tmp_path):
    cap = {}
    _patch_popen(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner()
    r.run(workspace=tmp_path, context={"goal": "win more"})
    p = cap["argv"][cap["argv"].index("-p") + 1]
    assert "win more" in p


def test_session_id_captured_from_result_event(monkeypatch, tmp_path):
    """claude -p --output-format json returns session_id in the result event.
    The runner must capture it (was discarded) so each act's claude history is
    locatable in the run output."""
    cap = {}
    result_line = json.dumps({
        "type": "result",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "session_id": "abc12345-dead-beef-cafe-feedface0000",
    })
    _patch_popen(monkeypatch, cap, _FakeProc(stdout=result_line + "\n",
                                             returncode=0))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.session_id == "abc12345-dead-beef-cafe-feedface0000"


def test_transcript_path_resolved_by_globbing_claude_home(monkeypatch, tmp_path):
    """transcript_path is found by globbing ~/.claude/projects/*/<sid>.jsonl —
    no hand-rolled cwd-slug. Monkeypatch home so the lookup is deterministic."""
    sid = "deadbeef-0000-1111-2222-333333333333"
    fake_home = tmp_path / "home"
    proj_dir = fake_home / ".claude" / "projects" / "some-cwd-slug"
    proj_dir.mkdir(parents=True)
    transcript = proj_dir / f"{sid}.jsonl"
    transcript.write_text("[]")
    monkeypatch.setattr("agentbench_frame.hl.runner.Path.home",
                        staticmethod(lambda: fake_home))

    result_line = json.dumps({"type": "result",
                              "usage": {"input_tokens": 1, "output_tokens": 1,
                                        "total_tokens": 2}, "session_id": sid})
    _patch_popen(monkeypatch, {}, _FakeProc(stdout=result_line + "\n",
                                             returncode=0))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.session_id == sid
    assert res.transcript_path == str(transcript)


def test_transcript_path_none_when_session_absent(monkeypatch, tmp_path):
    """No matching transcript file -> transcript_path None (not an error)."""
    monkeypatch.setattr("agentbench_frame.hl.runner.Path.home",
                        staticmethod(lambda: tmp_path))
    result_line = json.dumps({"type": "result", "session_id": "no-such-sid-0000",
                              "usage": {"input_tokens": 1, "output_tokens": 1,
                                        "total_tokens": 2}})
    _patch_popen(monkeypatch, {}, _FakeProc(stdout=result_line + "\n",
                                             returncode=0))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.session_id == "no-such-sid-0000"
    assert res.transcript_path is None
