"""OpenAI-compatible Chat Completions transport for C++ policy proposals."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .loop_config import LLMConfig


ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class LLMRequestError(RuntimeError):
    def __init__(self, stage: str, message: str, raw_response: Any = None,
                 usage: dict | None = None, latency_seconds: float = 0.0):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.message = message
        self.raw_response = raw_response
        self.usage = usage or dict(ZERO_USAGE)
        self.latency_seconds = latency_seconds


@dataclass(frozen=True)
class PolicyProposal:
    analysis: str
    player_ai_cpp: str
    usage: dict
    latency_seconds: float
    request_body: dict
    raw_response: dict
    normalized_fence: bool = False


class ChatCompletionsClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def propose(self, messages: list[dict]) -> PolicyProposal:
        body: dict[str, Any] = {
            "model": self.config.model, "messages": messages,
            "temperature": self.config.temperature, "stream": self.config.stream,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        if self.config.stream:
            body["stream_options"] = {"include_usage": True}
        secret = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        headers = {"Content-Type": "application/json", "User-Agent": "AgentBenchFramework/0.1 doto"}
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers=headers, method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                if self.config.stream:
                    raw, latency = self._read_stream(response, started)
                else:
                    payload = response.read()
                    latency = time.monotonic() - started
                    try:
                        raw = json.loads(payload)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise LLMRequestError("response_json", str(exc), latency_seconds=latency) from exc
                    raw.update({"stream": False, "chunk_count": 1,
                                "first_chunk_seconds": latency,
                                "usage_missing": not bool(raw.get("usage"))})
        except LLMRequestError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            message = self._safe_error(exc)
            if secret:
                message = message.replace(secret, "[REDACTED]")
            raise LLMRequestError("request", message, latency_seconds=time.monotonic() - started) from exc
        if secret:
            raw = self._redact(raw, secret)
        usage = self._usage(raw.get("usage", {}))
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("assistant_content", "missing choices[0].message.content",
                                  raw, usage, latency) from exc
        if not isinstance(content, str):
            raise LLMRequestError("assistant_content", "assistant content is not a string",
                                  raw, usage, latency)
        candidate, fenced = self._strip_one_json_fence(content)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMRequestError("proposal_json", str(exc), raw, usage, latency) from exc
        if not isinstance(parsed, dict):
            raise LLMRequestError("proposal_json", "proposal must be an object", raw, usage, latency)
        analysis, source = parsed.get("analysis"), parsed.get("player_ai_cpp")
        if not isinstance(analysis, str) or not analysis.strip():
            raise LLMRequestError("proposal_json", "analysis must be a nonempty string", raw, usage, latency)
        if not isinstance(source, str) or not source.strip():
            raise LLMRequestError("proposal_json", "player_ai_cpp must be a nonempty string", raw, usage, latency)
        return PolicyProposal(analysis.strip(), source, usage, latency, body, raw, fenced)

    def _read_stream(self, response, started: float) -> tuple[dict, float]:
        content, reasoning, usage = [], [], {}
        metadata: dict[str, Any] = {}
        finish_reason = None
        chunk_count = 0
        first_chunk_seconds = None

        def normalized() -> dict:
            return {**metadata, "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "".join(content),
                "reasoning_content": "".join(reasoning)},
                "finish_reason": finish_reason}], "usage": usage, "stream": True,
                "chunk_count": chunk_count, "first_chunk_seconds": first_chunk_seconds,
                "usage_missing": not bool(usage)}

        for raw_line in response:
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise LLMRequestError("stream_chunk_json", str(exc), normalized(),
                                      self._usage(usage), time.monotonic() - started) from exc
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return normalized(), time.monotonic() - started
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise LLMRequestError("stream_chunk_json", str(exc), normalized(),
                                      self._usage(usage), time.monotonic() - started) from exc
            if not isinstance(chunk, dict):
                continue
            chunk_count += 1
            if first_chunk_seconds is None:
                first_chunk_seconds = time.monotonic() - started
            for name in ("id", "object", "created", "model", "system_fingerprint"):
                if chunk.get(name) is not None:
                    metadata[name] = chunk[name]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if choices and isinstance(choices[0], dict):
                choice = choices[0]
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    if isinstance(delta.get("reasoning_content"), str):
                        reasoning.append(delta["reasoning_content"])
                    if isinstance(delta.get("content"), str):
                        content.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        raise LLMRequestError("stream_incomplete", "stream ended before [DONE]", normalized(),
                              self._usage(usage), time.monotonic() - started)

    @staticmethod
    def _strip_one_json_fence(content: str) -> tuple[str, bool]:
        candidate = content.strip()
        if candidate.startswith("```json\n") and candidate.endswith("```"):
            return candidate[8:-3].strip(), True
        if candidate.startswith("```\n") and candidate.endswith("```"):
            return candidate[4:-3].strip(), True
        return candidate, False

    @staticmethod
    def _usage(raw: dict) -> dict:
        return {name: int(raw.get(name, 0) or 0) for name in ZERO_USAGE}

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                excerpt = exc.read(500).decode("utf-8", errors="replace")
            except OSError:
                excerpt = ""
            return f"HTTP {exc.code}: {excerpt}"
        return str(exc)

    @classmethod
    def _redact(cls, value: Any, secret: str) -> Any:
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        if isinstance(value, list):
            return [cls._redact(item, secret) for item in value]
        if isinstance(value, dict):
            return {key: cls._redact(item, secret) for key, item in value.items()}
        return value
