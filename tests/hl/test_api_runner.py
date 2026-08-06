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


def test_applies_str_replace(tmp_path):
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(5, 1)),
        LLMResponse(text="done", tool_calls=[ToolCall("edit",
                    {"old_string": "def step():\n    pass\n",
                     "new_string": "def step():\n    return 1\n"})],
                    usage=Usage(7, 2)),
    ])
    r = ApiCodingRunner(client=client, system_prompt="S")
    res = r.run(workspace=tmp_path, context={"prompt": "improve it"})
    assert res.edit_type is None          # unclassified -> controller diff-classifies
    assert res.failure_reason is None
    assert (tmp_path / "agent.py").read_text() == "def step():\n    return 1\n"
    assert res.prompt_tokens == 12 and res.completion_tokens == 3
    assert client.calls == 2


def test_files_touched_recorded_on_edit(tmp_path):
    """Fix-F: a successful str_replace records agent.py in files_touched."""
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "def step():\n    pass\n",
                     "new_string": "def step():\n    return 1\n"})],
                    usage=Usage(7, 2))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.files_touched == ["agent.py"]


def test_files_touched_empty_when_no_edit(tmp_path):
    """Fix-F: a clean no-edit run records [] — never a fabricated path."""
    _seed_agent(tmp_path)
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(1, 1)),
        LLMResponse(text="I choose not to edit.", tool_calls=[], usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.files_touched == []


def test_str_replace_not_unique_not_applied(tmp_path):
    _seed_agent(tmp_path, "A = 1\nB = 1\n")  # "= 1\n" matches twice
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "= 1\n", "new_string": "= 2\n"})],
                    usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason and "not_unique" in res.failure_reason
    assert (tmp_path / "agent.py").read_text() == "A = 1\nB = 1\n"  # unchanged


def test_str_replace_not_found_not_applied(tmp_path):
    _seed_agent(tmp_path, "ORIG = 1\n")
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "NOT_PRESENT", "new_string": "x"})],
                    usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason and "not_found" in res.failure_reason
    assert (tmp_path / "agent.py").read_text() == "ORIG = 1\n"


def test_invalid_python_not_applied(tmp_path):
    _seed_agent(tmp_path, "ORIGINAL = 1\n")
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "ORIGINAL = 1\n", "new_string": "def (\n"})],
                    usage=Usage(1, 1))])
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


def test_read_file_returns_full_large_file(tmp_path):
    """read_file must return the whole file, not truncate at 20k chars — the
    LostSpace agent.py is ~30k chars; truncating left the model unable to see
    the whole file, driving 8+ redundant reads and step-cap waste."""
    big = "x = 1\n" * 5000  # 30k chars
    _seed_agent(tmp_path, big)
    captured = {}

    class C:
        def __init__(self): self.n = 0
        def complete(self, *, system, messages, tools, max_tokens, timeout=None):
            self.n += 1
            for m in messages:
                if m.get("role") == "tool" and m.get("tool_name") == "read_file":
                    captured["result"] = m.get("content")
            if self.n == 1:
                return LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                                   usage=Usage(1, 1))
            return LLMResponse(text="ok", tool_calls=[], usage=Usage(1, 1))

    ApiCodingRunner(client=C(), system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert len(captured.get("result", "")) > 20000  # not truncated at the old 20k cap


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
        LLMResponse(text="done", tool_calls=[ToolCall("edit",
                    {"old_string": "def step():\n    pass\n",
                     "new_string": "EDITED = 1\n"})], usage=Usage(7, 2)),
    ])
    runner = ApiCodingRunner(client=client, system_prompt="S", timeout=42.0)
    res = runner.run(workspace=tmp_path, context={"prompt": "improve it"})
    assert res.edit_type is None
    assert client.last_timeout is not None
    assert client.last_timeout <= 42.0


def _read_transcript(path):
    import json
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def test_transcript_records_successful_edit(tmp_path):
    _seed_agent(tmp_path)
    tdir = tmp_path / "transcripts"
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                    usage=Usage(5, 1)),
        LLMResponse(text="done", tool_calls=[ToolCall("edit",
                    {"old_string": "def step():\n    pass\n",
                     "new_string": "def step():\n    return 1\n"})],
                    usage=Usage(7, 2)),
    ])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path,
        context={"prompt": "x", "act_id": "act-1", "transcript_dir": str(tdir)})
    assert res.transcript_path is not None
    assert Path(res.transcript_path) == tdir / "act-1.jsonl"
    recs = _read_transcript(res.transcript_path)
    turns = [r for r in recs if r.get("kind") == "turn"]
    terms = [r for r in recs if r.get("kind") == "terminal"]
    assert len(turns) == 2
    assert turns[0]["tool_calls"][0]["name"] == "read_file"
    assert turns[0]["write_attempt"] is None
    wa = turns[1]["write_attempt"]
    assert wa["called"] is True and wa["ast_ok"] is True and wa["old_len"] > 0
    assert terms and terms[0]["edit_applied"] is True and terms[0]["failure_reason"] is None


def test_transcript_records_step_cap(tmp_path):
    _seed_agent(tmp_path)
    tdir = tmp_path / "t2"
    reader = LLMResponse(text="", tool_calls=[ToolCall("read_file", {"path": "agent.py"})],
                         usage=Usage(1, 1))
    client = ScriptedClient([reader] * 10)
    res = ApiCodingRunner(client=client, system_prompt="S", max_turns=3).run(
        workspace=tmp_path, context={"prompt": "x", "transcript_dir": str(tdir)})
    recs = _read_transcript(res.transcript_path)
    turns = [r for r in recs if r.get("kind") == "turn"]
    terms = [r for r in recs if r.get("kind") == "terminal"]
    assert len(turns) == 3
    assert all(t["write_attempt"] is None for t in turns)
    assert all(t["tool_calls"][0]["name"] == "read_file" for t in turns)
    assert terms[0]["failure_reason"] == "step_cap_exceeded"
    assert terms[0]["edit_applied"] is False


def test_transcript_records_syntax_failure(tmp_path):
    _seed_agent(tmp_path, "ORIG = 1\n")
    tdir = tmp_path / "t3"
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "ORIG = 1\n", "new_string": "def (\n"})],
                    usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x", "transcript_dir": str(tdir)})
    recs = _read_transcript(res.transcript_path)
    wa = recs[0]["write_attempt"]
    assert wa["called"] is True and wa["ast_ok"] is False
    assert wa["reason"] and "syntax" in wa["reason"].lower()


def test_empty_old_string_rejected(tmp_path):
    """An empty old_string must never be applied (silent corruption)."""
    _seed_agent(tmp_path, "ORIG = 1\n")
    client = ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("edit",
                    {"old_string": "", "new_string": "x"})],
                    usage=Usage(1, 1))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason == "empty_old_string"
    assert (tmp_path / "agent.py").read_text() == "ORIG = 1\n"  # unchanged


def test_truncated_write_rejected(tmp_path):
    """A length-capped generation (finish_reason=length) on an edit turn means
    the old/new strings were cut mid-stream -> never apply the partial edit."""
    _seed_agent(tmp_path, "ORIG = 1\n")
    client = ScriptedClient([
        LLMResponse(text="", finish_reason="length",
                    tool_calls=[ToolCall("edit",
                                {"old_string": "ORIG", "new_string": "partial"})],
                    usage=Usage(1, 4096))])
    res = ApiCodingRunner(client=client, system_prompt="S").run(
        workspace=tmp_path, context={"prompt": "x"})
    assert res.edit_type == "noop"
    assert res.failure_reason == "write_truncated"
    assert (tmp_path / "agent.py").read_text() == "ORIG = 1\n"  # unchanged


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
