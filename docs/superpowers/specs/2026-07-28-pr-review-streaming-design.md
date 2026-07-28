# PR Review Responses Streaming Design

## Problem

The trusted PR reviewer sends a non-streaming Responses API request. The
configured HTTP proxy returns 504 after roughly 62 seconds, before the
reviewer's 90-second request timeout, when a high-reasoning review produces no
HTTP response bytes during that interval. Retrying the same non-streaming
request reproduces the same failure.

## Scope

Only `responses` API mode becomes an SSE client. `chat_completions` remains a
normal JSON request. The model, high reasoning effort, structured-output
schema, compatibility fallbacks, retry policy, and fail-closed review contract
remain unchanged.

## Request

Responses requests include `"stream": true` and send
`Accept: text/event-stream`. Compatibility fallback requests are also streamed
because they use the same Responses request builder.

## Response handling

The reviewer parses UTF-8 SSE records incrementally:

- `response.output_text.delta` contributes text to the structured review.
- `response.completed` is the only successful terminal event.
- If no deltas were received, the completed event's full `response` object is
  accepted as a compatibility fallback.
- `response.failed`, `response.incomplete`, and `error` terminate with failure.
- Invalid JSON events, invalid event shapes, or EOF before
  `response.completed` fail closed.

SSE comments and unrelated valid lifecycle events are ignored. A `[DONE]`
marker does not replace `response.completed`.

## Verification

Local HTTP server tests cover request headers/body, delta reconstruction,
completed-response fallback, terminal failure events, and premature EOF. After
merge, a historical PR review is rerun against the real endpoint to determine
whether the proxy forwards SSE without buffering.
