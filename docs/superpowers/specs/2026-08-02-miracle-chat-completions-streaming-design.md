# Miracle Chat Completions Streaming Design

## Goal

Make OpenAI-compatible Chat Completions streaming the default Miracle Harness transport so upstream proxies receive continuous response bytes during long reasoning/generation and do not terminate an otherwise active request at an idle 60-second boundary.

## Configuration

`LLMConfig` adds:

```toml
[llm]
stream = true
```

The default is `true`. Set `stream = false` only for endpoints that do not support SSE streaming. The client never silently retries a failed streaming request as non-streaming because that could issue two billable requests for one iteration.

## Request

Streaming requests add:

```json
{
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

All existing fields, including `reasoning_effort`, remain unchanged. Non-streaming requests add `"stream": false` and omit `stream_options`.

## SSE Parsing

The client reads UTF-8 Server-Sent Events line by line. It ignores blank lines and comment/heartbeat lines beginning with `:`. Every `data:` payload must be either `[DONE]` or one JSON object.

For each chunk it:

- appends `choices[0].delta.reasoning_content` when present;
- appends `choices[0].delta.content` when present;
- preserves the latest non-null `finish_reason`;
- preserves response `id`, `object`, `created`, `model`, and `system_fingerprint` when present;
- reads usage from any chunk containing `usage`, with the last value authoritative;
- counts parsed chunks and records first-chunk latency.

At `[DONE]`, the accumulated stream is normalized into the same Chat Completions response shape consumed by the existing proposal parser:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "...",
      "reasoning_content": "..."
    },
    "finish_reason": "stop"
  }],
  "usage": {}
}
```

The stored response additionally contains `stream=true`, `chunk_count`, `first_chunk_seconds`, and `usage_missing`.

## Error and Budget Semantics

- HTTP, connection, or idle socket failures remain `request` failures.
- Invalid SSE JSON is `stream_chunk_json` and preserves the accumulated response.
- EOF without `[DONE]` is `stream_incomplete`, preserving accumulated content, usage, and timing.
- A complete stream whose assistant content is empty or invalid remains `assistant_content` or `proposal_json` as today.
- Usage is charged even when later proposal parsing or strategy validation fails.
- If the endpoint omits final usage, token counters remain zero and `usage_missing=true`; the Harness does not estimate tokens and does not invent accounting data.
- `max_tokens` truncation remains visible through `finish_reason="length"`; streaming prevents idle gateway timeout but does not remove output limits.

## Compatibility Boundary

Only `ChatCompletionsClient` and `LLMConfig` change. The Loop continues to receive one `StrategyProposal`, and strategy snapshots, battles, replay parsing, IG, score curves, budgets, and AgentBenchResults schemas remain unchanged.

## Verification

Tests use a local HTTP SSE server and cover:

1. split reasoning and content deltas;
2. final usage and `[DONE]`;
3. heartbeat/comment lines;
4. malformed chunk JSON;
5. EOF before `[DONE]`;
6. explicit `stream=false` compatibility;
7. secret redaction and failed-response usage accounting;
8. a minimal real request to the temporary API followed by a full three-iteration Run.
