"""OpenAI-compatible Chat Completions transport for strategy proposals."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .loop_config import LLMConfig


class LLMRequestError(RuntimeError):
    def __init__(
        self, stage: str, reason: str, raw_response=None,
        usage: dict | None = None, latency_seconds: float = 0.0,
    ):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.raw_response = raw_response
        self.usage = usage or {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        self.latency_seconds = latency_seconds


@dataclass(frozen=True)
class StrategyProposal:
    analysis: str
    strategy_code: str
    usage: dict
    request_body: dict
    raw_response: dict
    latency_seconds: float
    normalized_fence: bool = False


class ChatCompletionsClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def propose_strategy(self, messages: list[dict]) -> StrategyProposal:
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": self.config.stream,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if self.config.stream:
            body["stream_options"] = {"include_usage": True}
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentBenchFramework/0.1",
        }
        api_key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                if self.config.stream:
                    raw, latency = self._read_stream(response, started)
                else:
                    raw_bytes = response.read()
                    latency = time.monotonic() - started
                    try:
                        raw = json.loads(raw_bytes)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        excerpt = raw_bytes[:500].decode("utf-8", errors="replace")
                        raise LLMRequestError(
                            "response_json", str(exc), excerpt,
                            latency_seconds=latency,
                        ) from exc
                    if isinstance(raw, dict):
                        raw.setdefault("stream", False)
                        raw.setdefault("chunk_count", 1)
                        raw.setdefault("first_chunk_seconds", latency)
                        raw.setdefault("usage_missing", not bool(raw.get("usage")))
        except LLMRequestError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMRequestError("request", self._safe_error(exc)) from exc
        raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        usage = {
            name: int(raw_usage.get(name, 0) or 0)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError(
                "assistant_content", "missing choices[0].message.content", raw,
                usage, latency,
            ) from exc
        if not isinstance(content, str):
            raise LLMRequestError(
                "assistant_content", "assistant content is not a string", raw,
                usage, latency,
            )

        normalized = False
        candidate = content.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            first_newline = candidate.find("\n")
            if first_newline >= 0:
                candidate = candidate[first_newline + 1:-3].strip()
                normalized = True
        try:
            proposal = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMRequestError("proposal_json", str(exc), raw, usage, latency) from exc
        if not isinstance(proposal, dict):
            raise LLMRequestError(
                "proposal_json", "proposal must be an object", raw, usage, latency,
            )
        analysis = proposal.get("analysis")
        strategy_code = proposal.get("strategy_code")
        if not isinstance(analysis, str) or not analysis.strip():
            raise LLMRequestError(
                "proposal_json", "analysis must be a nonempty string", raw, usage, latency,
            )
        if not isinstance(strategy_code, str) or not strategy_code.strip():
            raise LLMRequestError(
                "proposal_json", "strategy_code must be a nonempty string", raw,
                usage, latency,
            )
        return StrategyProposal(
            analysis.strip(), strategy_code, usage, body, raw, latency, normalized,
        )

    def _read_stream(self, response, started: float) -> tuple[dict, float]:
        content_parts = []
        reasoning_parts = []
        metadata = {}
        usage = {}
        finish_reason = None
        chunk_count = 0
        first_chunk_seconds = None

        def normalized() -> dict:
            return {
                **metadata,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts),
                        "reasoning_content": "".join(reasoning_parts),
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": usage,
                "stream": True,
                "chunk_count": chunk_count,
                "first_chunk_seconds": first_chunk_seconds,
                "usage_missing": not bool(usage),
            }

        for raw_line in response:
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                latency = time.monotonic() - started
                raise LLMRequestError(
                    "stream_chunk_json", str(exc), normalized(),
                    self._usage(usage), latency,
                ) from exc
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
                latency = time.monotonic() - started
                raise LLMRequestError(
                    "stream_chunk_json", str(exc), normalized(),
                    self._usage(usage), latency,
                ) from exc
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
                        reasoning_parts.append(delta["reasoning_content"])
                    if isinstance(delta.get("content"), str):
                        content_parts.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        latency = time.monotonic() - started
        raise LLMRequestError(
            "stream_incomplete", "stream ended before [DONE]", normalized(),
            self._usage(usage), latency,
        )

    @staticmethod
    def _usage(raw_usage: dict) -> dict:
        return {
            name: int(raw_usage.get(name, 0) or 0)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                excerpt = exc.read(500).decode("utf-8", errors="replace")
            except OSError:
                excerpt = ""
            return f"HTTP {exc.code}: {excerpt}"
        return str(exc)
