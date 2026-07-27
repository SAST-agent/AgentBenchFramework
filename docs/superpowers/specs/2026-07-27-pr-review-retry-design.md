# PR Review Upstream Retry Design

## Goal

Make the blocking PR reviewer tolerant of transient upstream gateway failures without changing the configured model or request semantics.

## Behavior

- The first request keeps the existing model, API mode, reasoning setting, and structured-output body.
- HTTP 502, 503, and 504, plus network timeout errors, are transient and retry the same request twice.
- Retries use a short exponential backoff; the delay is configurable for deterministic tests.
- After the same-request retry budget is exhausted, the existing no-reasoning and legacy structured-output fallbacks remain unchanged.
- HTTP 400, 401, 403, 422, 500, validation failures, and other non-transient errors do not enter the new retry loop.
- Every exhausted failure still fails closed.

## Testing

The CLI tests will verify that a transient gateway error followed by success sends identical strict requests, that repeated gateway errors still fail closed after the bounded retry budget, and that non-retryable HTTP errors are sent only once.
