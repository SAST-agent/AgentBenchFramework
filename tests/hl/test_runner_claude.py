"""Tests for the real-CLI knobs on ClaudeCodeRunner.

No real ``claude`` binary is invoked — ``subprocess.run`` is monkeypatched to
capture the argv and return a canned JSONL result line.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_frame.hl.runner import ClaudeCodeRunner


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _patch_run(monkeypatch, capture, proc):
    def fake_run(argv, **kwargs):
        capture["argv"] = list(argv)
        capture["cwd"] = kwargs.get("cwd")
        capture["timeout"] = kwargs.get("timeout")
        return proc
    monkeypatch.setattr("agentbench_frame.hl.runner.subprocess.run", fake_run)


def test_argv_includes_system_prompt_and_permission_mode(monkeypatch, tmp_path):
    cap = {}
    _patch_run(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
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
    _patch_run(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner(dangerously_skip_permissions=True)
    r.run(workspace=tmp_path, context={"prompt": "go"})
    assert "--dangerously-skip-permissions" in cap["argv"]
    assert "--permission-mode" not in cap["argv"]


def test_model_flag_when_set(monkeypatch, tmp_path):
    cap = {}
    _patch_run(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
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
    _patch_run(monkeypatch, cap, _FakeProc(stdout=result_line + "\n",
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
    monkeypatch.setattr("agentbench_frame.hl.runner.subprocess.run", boom)
    r = ClaudeCodeRunner(claude_path="claude-missing")
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.error is not None
    assert "claude-missing" in res.error
    assert res.edit_type == "noop"


def test_nonzero_exit_with_stderr(monkeypatch, tmp_path):
    cap = {}
    _patch_run(monkeypatch, cap, _FakeProc(stdout="", stderr="boom: bad",
                                           returncode=2))
    r = ClaudeCodeRunner()
    res = r.run(workspace=tmp_path, context={"prompt": "go"})
    assert res.error is not None
    assert "2" in res.error


def test_default_prompt_used_when_context_omits_prompt(monkeypatch, tmp_path):
    cap = {}
    _patch_run(monkeypatch, cap, _FakeProc(stdout="", returncode=0))
    r = ClaudeCodeRunner()
    r.run(workspace=tmp_path, context={"goal": "win more"})
    p = cap["argv"][cap["argv"].index("-p") + 1]
    assert "win more" in p
