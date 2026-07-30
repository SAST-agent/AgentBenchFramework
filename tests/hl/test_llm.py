from __future__ import annotations
import json
from types import SimpleNamespace

import pytest

from agentbench_frame.hl.llm import (
    LLMResponse, OpenAICompatClient, AnthropicClient, build_client, SHARED_TOOLS,
)
from agentbench_frame.hl.models_config import ModelEntry


# ---- fake SDK clients (no network) ----

class _FakeOpenAI:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


class _FakeAnthropic:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


def _openai_resp(text, tool_calls=None, usage=(10, 5)):
    tc = None
    if tool_calls:
        tc = [SimpleNamespace(id=f"c{i}", type="function",
                              function=SimpleNamespace(name=n, arguments=json.dumps(a)))
              for i, (n, a) in enumerate(tool_calls)]
    msg = SimpleNamespace(content=text, tool_calls=tc)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tc else "stop")],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1],
                              total_tokens=sum(usage)),
    )


def _anthropic_resp(text, tool_uses=None, usage=(10, 5)):
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    for n, a in (tool_uses or []):
        content.append(SimpleNamespace(type="tool_use", id="t", name=n, input=a))
    return SimpleNamespace(
        content=content,
        stop_reason="tool_use" if tool_uses else "end_turn",
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )


def test_openai_client_maps_response():
    resp = _openai_resp("hi", tool_calls=[("read_file", {"path": "agent.py"})], usage=(12, 3))
    client = OpenAICompatClient(api_key="k", model="m", base_url="u", _client=_FakeOpenAI(resp))
    out = client.complete(system="S", messages=[{"role": "user", "content": "q"}],
                          tools=SHARED_TOOLS, max_tokens=100)
    assert isinstance(out, LLMResponse)
    assert out.text == "hi"
    assert out.tool_calls[0].name == "read_file"
    assert out.tool_calls[0].arguments == {"path": "agent.py"}
    assert out.usage.prompt_tokens == 12
    assert out.usage.completion_tokens == 3
    assert out.usage.total_tokens == 15


def test_openai_client_no_tool_calls():
    resp = _openai_resp("done", tool_calls=None, usage=(5, 2))
    client = OpenAICompatClient(api_key="k", model="m", _client=_FakeOpenAI(resp))
    out = client.complete(system="S", messages=[], tools=SHARED_TOOLS, max_tokens=100)
    assert out.tool_calls == []
    assert out.finish_reason == "stop"


def test_openai_client_converts_neutral_history():
    fake = _FakeOpenAI(_openai_resp("ok"))
    client = OpenAICompatClient(api_key="k", model="m", _client=fake)
    neutral = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "thinking", "tool_calls": []},
        {"role": "tool", "tool_name": "read_file", "content": "FILE"},
    ]
    client.complete(system="S", messages=neutral, tools=SHARED_TOOLS, max_tokens=100)
    sent = fake.calls[0]
    # tool result is converted to an OpenAI "tool" role message
    roles = [m["role"] for m in sent["messages"]]
    assert "tool" in roles


def test_anthropic_client_maps_response():
    resp = _anthropic_resp("hi", tool_uses=[("write_agent_py", {"content": "print(1)"})],
                           usage=(8, 4))
    client = AnthropicClient(api_key="k", model="m", _client=_FakeAnthropic(resp))
    out = client.complete(system="S", messages=[{"role": "user", "content": "q"}],
                          tools=SHARED_TOOLS, max_tokens=100)
    assert out.tool_calls[0].name == "write_agent_py"
    assert out.tool_calls[0].arguments == {"content": "print(1)"}
    assert out.usage.prompt_tokens == 8
    assert out.usage.completion_tokens == 4
    assert out.usage.total_tokens == 12


def test_anthropic_client_system_top_level():
    fake = _FakeAnthropic(_anthropic_resp("ok"))
    client = AnthropicClient(api_key="k", model="m", _client=fake)
    client.complete(system="THE-SYSTEM", messages=[{"role": "user", "content": "q"}],
                    tools=SHARED_TOOLS, max_tokens=100)
    assert fake.calls[0]["system"] == "THE-SYSTEM"
    # tools are forwarded as anthropic-shaped specs
    assert all("input_schema" in t for t in fake.calls[0]["tools"])


def test_build_client_picks_backend():
    e_oai = ModelEntry("glm", "openai", "glm-4.6", "k", "https://u")
    e_ant = ModelEntry("claude", "anthropic", "claude-sonnet-5", "k")
    assert isinstance(build_client(e_oai), OpenAICompatClient)
    assert isinstance(build_client(e_ant), AnthropicClient)


def test_build_client_unknown_provider():
    with pytest.raises(ValueError):
        build_client(ModelEntry("x", "madeup", "m", "k"))


def test_shared_tools_have_four_names():
    assert {t["name"] for t in SHARED_TOOLS} == {
        "read_file", "list_replays", "read_replay", "write_agent_py"}


def test_openai_multi_call_roundtrip():
    """Test that tool call IDs are preserved across multiple tool calls in a conversation."""
    from agentbench_frame.hl.llm import _to_openai_messages, ToolCall

    # Simulate a conversation where the assistant makes two tool calls with distinct IDs
    neutral = [
        {"role": "user", "content": "Read two files"},
        {"role": "assistant", "content": "I'll read both files", "tool_calls": [
            ToolCall(name="read_file", arguments={"path": "file1.py"}, id="call_a"),
            ToolCall(name="read_file", arguments={"path": "file2.py"}, id="call_b"),
        ]},
        # Tool results matching the IDs above
        {"role": "tool", "tool_call_id": "call_a", "content": "FILE1"},
        {"role": "tool", "tool_call_id": "call_b", "content": "FILE2"},
    ]

    converted = _to_openai_messages("sys", neutral)

    # Check that assistant message has correct tool_calls with proper IDs
    assistant_msg = [m for m in converted if m["role"] == "assistant"][0]
    assert len(assistant_msg["tool_calls"]) == 2
    assert assistant_msg["tool_calls"][0]["id"] == "call_a"
    assert assistant_msg["tool_calls"][1]["id"] == "call_b"

    # Check that tool messages have matching tool_call_ids
    tool_msgs = [m for m in converted if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "call_a"
    assert tool_msgs[1]["tool_call_id"] == "call_b"


def test_anthropic_multi_call_roundtrip():
    """Test that tool_use IDs are preserved across multiple tool calls in a conversation."""
    from agentbench_frame.hl.llm import _to_anthropic_messages, ToolCall

    # Simulate a conversation where the assistant makes two tool calls with distinct IDs
    neutral = [
        {"role": "user", "content": "Read two files"},
        {"role": "assistant", "content": "I'll read both files", "tool_calls": [
            ToolCall(name="read_file", arguments={"path": "file1.py"}, id="toolu_a1b2c3"),
            ToolCall(name="read_file", arguments={"path": "file2.py"}, id="toolu_d4e5f6"),
        ]},
        # Tool results matching the IDs above
        {"role": "tool", "tool_call_id": "toolu_a1b2c3", "content": "FILE1"},
        {"role": "tool", "tool_call_id": "toolu_d4e5f6", "content": "FILE2"},
    ]

    converted = _to_anthropic_messages(neutral)

    # Check that assistant message has correct tool_use blocks with proper IDs
    assistant_msg = [m for m in converted if m["role"] == "assistant"][0]
    tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
    assert len(tool_use_blocks) == 2
    assert tool_use_blocks[0]["id"] == "toolu_a1b2c3"
    assert tool_use_blocks[1]["id"] == "toolu_d4e5f6"

    # Check that tool result messages have matching tool_use_ids
    user_msgs = [m for m in converted if m["role"] == "user"]
    tool_result_blocks = []
    for m in user_msgs:
        # Only process if content is a list (tool result messages), not a string (regular user messages)
        if isinstance(m["content"], list):
            tool_result_blocks.extend([b for b in m["content"] if b["type"] == "tool_result"])

    assert len(tool_result_blocks) == 2
    assert tool_result_blocks[0]["tool_use_id"] == "toolu_a1b2c3"
    assert tool_result_blocks[1]["tool_use_id"] == "toolu_d4e5f6"


def test_openai_timeout_forwarded():
    """Test that timeout kwarg is forwarded to OpenAI SDK create() call."""
    resp = _openai_resp("ok")
    fake = _FakeOpenAI(resp)
    client = OpenAICompatClient(api_key="k", model="m", _client=fake)
    client.complete(system="S", messages=[], tools=SHARED_TOOLS, max_tokens=100, timeout=12.5)
    assert fake.calls[0]["timeout"] == 12.5


def test_openai_timeout_omitted_when_none():
    """Test that timeout is not passed to SDK when None."""
    resp = _openai_resp("ok")
    fake = _FakeOpenAI(resp)
    client = OpenAICompatClient(api_key="k", model="m", _client=fake)
    client.complete(system="S", messages=[], tools=SHARED_TOOLS, max_tokens=100, timeout=None)
    assert "timeout" not in fake.calls[0]


def test_anthropic_timeout_forwarded():
    """Test that timeout kwarg is forwarded to Anthropic SDK create() call."""
    resp = _anthropic_resp("ok")
    fake = _FakeAnthropic(resp)
    client = AnthropicClient(api_key="k", model="m", _client=fake)
    client.complete(system="S", messages=[], tools=SHARED_TOOLS, max_tokens=100, timeout=12.5)
    assert fake.calls[0]["timeout"] == 12.5


def test_anthropic_timeout_omitted_when_none():
    """Test that timeout is not passed to SDK when None."""
    resp = _anthropic_resp("ok")
    fake = _FakeAnthropic(resp)
    client = AnthropicClient(api_key="k", model="m", _client=fake)
    client.complete(system="S", messages=[], tools=SHARED_TOOLS, max_tokens=100, timeout=None)
    assert "timeout" not in fake.calls[0]

