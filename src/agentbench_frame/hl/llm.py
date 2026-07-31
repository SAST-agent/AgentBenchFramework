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
    id: Optional[str] = None


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
        "description": "Read a UTF-8 text file from the agent workspace (relative "
                       "path). Returns the ENTIRE file content (up to 64k chars) "
                       "— one read sees the whole file, no paging needed.",
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
        "name": "str_replace",
        "description": "Make a surgical edit to agent.py: replace exactly one "
                       "UNIQUE occurrence of old_string with new_string. "
                       "old_string must match the file exactly (indentation, "
                       "newlines) and appear exactly once. The result is "
                       "ast-validated before apply; a SyntaxError or a "
                       "non-unique/missing old_string is rejected and nothing "
                       "is written. Prefer many small str_replace calls over "
                       "rewriting the whole file.",
        "input_schema": {"type": "object",
                         "properties": {
                             "old_string": {"type": "string"},
                             "new_string": {"type": "string"},
                         },
                         "required": ["old_string", "new_string"]},
    },
]


def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in tools]


def _to_openai_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    last_assistant_tool_call_ids: List[str] = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            # Use provided tool_call_id if available, otherwise fall back to most recent assistant tool call
            tool_call_id = m.get("tool_call_id")
            if tool_call_id is None and last_assistant_tool_call_ids:
                # Fallback: use the first tool call id from the most recent assistant message
                # (correct for single-tool-call-per-turn case that existing tests use)
                tool_call_id = last_assistant_tool_call_ids[0]
            if tool_call_id is None:
                # Ultimate fallback if no context (shouldn't happen in well-formed sequences)
                tool_call_id = "unknown"
            out.append({"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(m.get("content", ""))})
        elif role == "assistant" and m.get("tool_calls"):
            assistant_tc_ids: List[str] = []
            tool_calls = []
            for i, tc in enumerate(m["tool_calls"]):
                # Use tc.id if provided, otherwise generate one
                tc_id = tc.id if tc.id else f"call_{i}"
                assistant_tc_ids.append(tc_id)
                tool_calls.append({"id": tc_id, "type": "function",
                                   "function": {"name": tc.name,
                                                "arguments": json.dumps(tc.arguments)}})
            last_assistant_tool_call_ids = assistant_tc_ids
            out.append({"role": "assistant",
                        "content": m.get("content") or "",
                        "tool_calls": tool_calls})
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

    def complete(self, *, system: str, messages, tools, max_tokens: int, timeout: Optional[float] = None) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(system, messages),
            "tools": _to_openai_tools(tools),
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: List[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id))
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
    last_assistant_tool_call_ids: List[str] = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            # Use provided tool_call_id if available, otherwise fall back to most recent assistant tool call
            tool_use_id = m.get("tool_call_id")
            if tool_use_id is None and last_assistant_tool_call_ids:
                # Fallback: use the first tool call id from the most recent assistant message
                # (correct for single-tool-call-per-turn case that existing tests use)
                tool_use_id = last_assistant_tool_call_ids[0]
            if tool_use_id is None:
                # Ultimate fallback if no context (shouldn't happen in well-formed sequences)
                tool_use_id = "unknown"
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": str(m.get("content", ""))}]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            assistant_tc_ids: List[str] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for i, tc in enumerate(m["tool_calls"]):
                # Use tc.id if provided, otherwise generate one
                tc_id = tc.id if tc.id else f"toolu_{i}"
                assistant_tc_ids.append(tc_id)
                blocks.append({"type": "tool_use", "id": tc_id, "name": tc.name,
                               "input": tc.arguments})
            last_assistant_tool_call_ids = assistant_tc_ids
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

    def complete(self, *, system: str, messages, tools, max_tokens: int, timeout: Optional[float] = None) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": _to_anthropic_messages(messages),
            "tools": tools,  # already anthropic-shaped
            "max_tokens": max_tokens,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = self._client.messages.create(**kwargs)
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(name=block.name,
                                           arguments=dict(getattr(block, "input", {}) or {}),
                                           id=getattr(block, "id", None)))
        u = getattr(resp, "usage", None)
        pt = getattr(u, "input_tokens", None) if u else None
        ct = getattr(u, "output_tokens", None) if u else None
        usage = Usage(prompt_tokens=pt, completion_tokens=ct,
                      total_tokens=(pt + ct if pt is not None and ct is not None else None))
        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls,
                           usage=usage, finish_reason=getattr(resp, "stop_reason", "stop") or "stop")


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, messages, tools, max_tokens: int, timeout: Optional[float] = None) -> LLMResponse: ...


def build_client(entry: ModelEntry):
    if entry.provider == "openai":
        return OpenAICompatClient(api_key=entry.api_key, model=entry.model,
                                  base_url=entry.base_url)
    if entry.provider == "anthropic":
        return AnthropicClient(api_key=entry.api_key, model=entry.model)
    raise ValueError(f"unknown provider: {entry.provider!r} (label={entry.label!r})")
