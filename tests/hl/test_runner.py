"""Tests for hl/runner.py — CodingAgentRunner interface + FakeRunner + ClaudeCodeRunner.

Contract:
- CodingAgentRunner: a pluggable interface. The controller calls run() with
  the workspace + resource views; the runner performs edits and returns a
  summary (edit_type, files touched, token usage if any).
- FakeRunner: scripted edits for tests/debugging — applies a deterministic
  transform to the workspace so the controller's loop is testable without
  a real CLI.
- ClaudeCodeRunner: shells out to the `claude` CLI (real). Not exercised
  here beyond import/construction; its behavior is verified in the E2E
  smoke if a CLI is present.
"""
from pathlib import Path

import pytest

from agentbench_frame.hl.runner import (
    CodingAgentRunner,
    FakeRunner,
    AgentRunResult,
)


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("THRESHOLD = 10\n", encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def test_fake_runner_is_a_coding_agent_runner():
    fr = FakeRunner(transform=lambda ws: None)
    assert isinstance(fr, CodingAgentRunner)


def test_fake_runner_applies_scripted_edit(tmp_path):
    ws = _ws(tmp_path)

    def bump_threshold(workspace: Path):
        p = workspace / "agent.py"
        p.write_text("THRESHOLD = 20\n", encoding="utf-8")

    fr = FakeRunner(transform=bump_threshold, edit_type="parametrize")
    result = fr.run(workspace=ws, context={"act_id": "a1"})
    assert isinstance(result, AgentRunResult)
    assert result.edit_type == "parametrize"
    assert (ws / "agent.py").read_text() == "THRESHOLD = 20\n"


def test_fake_runner_records_files_touched(tmp_path):
    ws = _ws(tmp_path)

    def add_rule(workspace: Path):
        (workspace / "rules").mkdir(exist_ok=True)
        (workspace / "rules/escape.py").write_text("def r(): pass\n",
                                                    encoding="utf-8")

    fr = FakeRunner(transform=add_rule, edit_type="add_rule")
    result = fr.run(workspace=ws, context={})
    assert "rules/escape.py" in result.files_touched


def test_fake_runner_reports_unknown_token_usage_as_none(tmp_path):
    """FakeRunner has no token usage; reports None, not 0 (doc §13)."""
    ws = _ws(tmp_path)
    fr = FakeRunner(transform=lambda w: None)
    result = fr.run(workspace=ws, context={})
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


def test_coder_interface_is_protocol_like():
    """CodingAgentRunner is an abstract interface: subclasses implement run()."""
    with pytest.raises(TypeError):
        CodingAgentRunner()  # type: ignore[abstract]


def test_claude_code_runner_imports_without_cli(tmp_path):
    """ClaudeCodeRunner constructs even when no claude binary is on PATH —
    it only fails on .run(), not on construction."""
    from agentbench_frame.hl.runner import ClaudeCodeRunner
    r = ClaudeCodeRunner(claude_path="claude-not-real")
    assert isinstance(r, CodingAgentRunner)
