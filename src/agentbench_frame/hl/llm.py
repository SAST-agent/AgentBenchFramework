"""
Provider abstraction for the HL raw-API runner. One ``complete()`` interface,
two backends.

- ``OpenAICompatClient``: ``openai`` SDK ``chat.completions`` — covers any
  OpenAI-compatible endpoint (GLM/DeepSeek/Qwen/Moonshot/GPT) via ``base_url``.
- ``AnthropicClient``: ``anthropic`` SDK ``messages`` — Claude.

Messages are a NEUTRAL history the caller builds; each backend converts to its
provider's shape:
  {"role":"user","content":str}
  {"role":"assistant","content":str,"tool_calls":[ToolCall,...]}
  {"role":"tool","tool_name":str,"content":str}

SDK imports are lazy (inside constructors) so importing this module without the
opt-in ``hl`` extra does not crash.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from agentbench_frame.hl.models_config import ModelEntry


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"


# Neutral (Anthropic-style) tool schema; each backend converts.
SHARED_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the agent workspace (relative path).",
        "input_schema": {"type": "object",
                         "properties": {"path": {"type": "string"}},
                         "required": ["path"]},
    },
    {
        "name": "list_replays",
        "description": "List available match replay identifiers for this round.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_replay",
        "description": "Read one match replay by id (JSON).",
        "input_schema": {"type": "object",
                         "properties": {"id": {"type": "string"}},
                         "required": ["id"]},
    },
    {
        "name": "write_agent_py",
        "description": "Replace the entire agent.py with the given Python source. "
                       "Must be complete, syntactically valid Python.",
        "input_schema": {"type": "object",
                         "properties": {"content": {"type": "string"}},
                         "required": ["content"]},
    },
]


def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in tools]


def _to_openai_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool",
                        "tool_call_id": "c0",
                        "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant",
                        "content": m.get("content") or "",
                        "tool_calls": [{"id": f"c{i}", "type": "function",
                                        "function": {"name": tc.name,
                                                     "arguments": json.dumps(tc.arguments)}}
                                       for i, tc in enumerate(m["tool_calls"])]})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class OpenAICompatClient:
    def __init__(self, *, api_key: str, model: str,
                 base_url: Optional[str] = None, _client=None):
        self.model = model
        self.base_url = base_url
        if _client is not None:
            self._client = _client
        else:
            import openai  # lazy
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)

    def complete(self, *, system: str, messages, tools, max_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(system, messages),
            tools=_to_openai_tools(tools),
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: List[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args))
        u = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", None) if u else None,
            completion_tokens=getattr(u, "completion_tokens", None) if u else None,
            total_tokens=getattr(u, "total_tokens", None) if u else None,
        )
        return LLMResponse(text=(getattr(msg, "content", None) or ""),
                           tool_calls=tool_calls, usage=usage,
                           finish_reason=getattr(choice, "finish_reason", "stop") or "stop")


def _to_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": str(m.get("content", ""))}]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for i, tc in enumerate(m["tool_calls"]):
                blocks.append({"type": "tool_use", "id": f"t{i}", "name": tc.name,
                               "input": tc.arguments})
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class AnthropicClient:
    def __init__(self, *, api_key: str, model: str, _client=None):
        self.model = model
        if _client is not None:
            self._client = _client
        else:
            import anthropic  # lazy
            self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, *, system: str, messages, tools, max_tokens: int) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=_to_anthropic_messages(messages),
            tools=tools,  # already anthropic-shaped
            max_tokens=max_tokens,
        )
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(name=block.name,
                                           arguments=dict(getattr(block, "input", {}) or {})))
        u = getattr(resp, "usage", None)
        pt = getattr(u, "input_tokens", None) if u else None
        ct = getattr(u, "output_tokens", None) if u else None
        usage = Usage(prompt_tokens=pt, completion_tokens=ct,
                      total_tokens=(pt + ct if pt is not None and ct is not None else None))
        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls,
                           usage=usage, finish_reason=getattr(resp, "stop_reason", "stop") or "stop")


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, messages, tools, max_tokens: int) -> LLMResponse: ...


def build_client(entry: ModelEntry):
    if entry.provider == "openai":
        return OpenAICompatClient(api_key=entry.api_key, model=entry.model,
                                  base_url=entry.base_url)
    if entry.provider == "anthropic":
        return AnthropicClient(api_key=entry.api_key, model=entry.model)
    raise ValueError(f"unknown provider: {entry.provider!r} (label={entry.label!r})")
