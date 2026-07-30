from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

from agentbench_frame.hl.llm import LLMResponse, ToolCall, Usage
from agentbench_frame.hl.runner import ApiCodingRunner


class ScriptedClient:
    """Returns a scripted list of LLMResponses, one per complete() call."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_timeout = None

    def complete(self, *, system, messages, tools, max_tokens, timeout=None):
        self.calls += 1
        self.last_timeout = timeout
        return self._responses.pop(0)


def _seed_agent(workspace: Path, body: str = "def step():\n    pass\n") -> Path:
    p = workspace / "agent.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_applies_write_agent_py(tmp_path):
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(5, 1)),
        LLMResponse(text="done", tool_calls=[ToolCall("write_agent_py",
                    {"content": "VALUE = 1\n"})], usage=Usage(7, 2)),
    ])
    r = ApiCodingRunner(client=client, system_prompt="S")
    res = r.run(workspace=tmp_path, context={"prompt": "improve it"})
    assert res.edit_type is None          # unclassified -> controller diff-classifies
    assert res.failure_reason is None
    assert (tmp_path / "agent.py").read_text() == "VALUE = 1\n"
    assert res.prompt_tokens == 12 and res.completion_tokens == 3
    assert client.calls == 2


def test_invalid_python_not_applied(tmp_path):
    _seed_agent(tmp_path, "ORIGINAL = 1\n")
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("write_agent_py",
                    {"content": "def (\n"})], usage=Usage(1, 1)),
    ])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason and "syntax" in res.failure_reason.lower()
    assert (tmp_path / "agent.py").read_text() == "ORIGINAL = 1\n"


def test_step_cap_exceeded(tmp_path):
    _seed_agent(tmp_path)
    # always reads, never writes
    reader = LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                         usage=Usage(1, 1))
    client = ScriptedClient([reader] * 10)
    res = ApiCodingRunner(client=client, system_prompt="S", max_turns=3).run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason == "step_cap_exceeded"
    assert client.calls == 3


def test_read_file_path_escape_blocked(tmp_path):
    _seed_agent(tmp_path)
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    seen = {}

    class Probe:
        def __init__(self): self.calls = 0
        def complete(self, *, system, messages, tools, max_tokens, timeout=None):
            self.calls += 1
            # record the tool-result text the runner fed back
            for m in messages:
                if m.get("role") == "tool":
                    seen[m.get("tool_name")] = m.get("content")
            if self.calls == 1:
                return LLMResponse(text="", tool_calls=[
                    ToolCall("read_file", {"path": "../secret.txt"})], usage=Usage(1, 1))
            return LLMResponse(text="ok", tool_calls=[], usage=Usage(1, 1))

    ApiCodingRunner(client=Probe(), system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert "error" in seen.get("read_file", "").lower()
    assert "TOPSECRET" not in seen.get("read_file", "")


def test_api_error_is_visible_failure(tmp_path):
    _seed_agent(tmp_path)

    class Boom:
        def complete(self, *, system, messages, tools, max_tokens, timeout=None):
            raise RuntimeError("401 unauthorized")

    res = ApiCodingRunner(client=Boom(), system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason and "401" in res.failure_reason


def test_no_tool_calls_clean_noop(tmp_path):
    _seed_agent(tmp_path, "ORIG = 1\n")
    client = ScriptedClient([
        LLMResponse(text="I choose not to edit.", tool_calls=[], usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason is None              # clean no-op, not a failure
    assert (tmp_path / "agent.py").read_text() == "ORIG = 1\n"


def test_timeout_passed_to_client(tmp_path):
    """Test that timeout kwarg is passed to client complete() calls."""
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(5, 1)),
        LLMResponse(text="done", tool_calls=[ToolCall("write_agent_py",
                    {"content": "EDITED = 1\n"})], usage=Usage(7, 2)),
    ])
    runner = ApiCodingRunner(client=client, system_prompt="S", timeout=42.0)
    res = runner.run(workspace=tmp_path, context={"prompt": "improve it"})
    assert res.edit_type is None
    assert client.last_timeout is not None
    assert client.last_timeout <= 42.0


def test_zero_timeout_immediate_deadline(tmp_path):
    """Test that timeout=0.0 causes immediate deadline exit."""
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(1, 1))],
    )
    runner = ApiCodingRunner(client=client, system_prompt="S", timeout=0.0, max_turns=50)
    res = runner.run(workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason == "timeout"
    # Should exit immediately without calling the client more than once (or at all)
    assert client.calls <= 1
