# PR Review Upstream Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry transient upstream gateway failures with the unchanged review request before using the existing compatibility fallbacks.

**Architecture:** Add a small retry wrapper around the initial `_post_review_request` call in `tools/pr_review.py`. The wrapper recognizes only gateway statuses and timeout errors, applies bounded exponential backoff, and leaves the current fallback sequence and fail-closed behavior intact.

**Tech Stack:** Python standard library, `urllib`, pytest, local `ThreadingHTTPServer` test fixtures.

## Global Constraints

- Do not change `PR_REVIEW_MODEL`, API mode, request schema, or reasoning configuration.
- Retry only transient gateway and timeout failures.
- Preserve fail-closed behavior after all attempts fail.
- Keep the implementation standard-library-only.

---

### Task 1: Lock retry semantics with tests

**Files:**
- Modify: `tests/test_pr_review.py`

**Interfaces:**
- Consumes: the existing `_run_cli` test helper and local HTTP server fixture.
- Produces: executable tests for identical-request retries, bounded exhaustion, and non-retryable errors.

- [ ] **Step 1: Add a test for same-request retry**

  Add a handler that returns HTTP 504 once and then a valid review. Assert two requests were received and both bodies retain the original strict schema and `reasoning.effort=high`.

- [ ] **Step 2: Run the focused test and verify it fails**

  Run `pytest tests/test_pr_review.py::test_cli_retries_same_request_after_gateway_timeout -q` from the worktree. It should fail because the current second request removes reasoning.

- [ ] **Step 3: Add a bounded-exhaustion assertion**

  Extend the retry coverage with a handler that always returns 504 and assert the CLI exits non-zero after the configured same-request attempts rather than retrying indefinitely.

- [ ] **Step 4: Add a non-retryable assertion**

  Assert that an HTTP 401 handler receives exactly one request and the CLI fails.

### Task 2: Implement the bounded retry wrapper

**Files:**
- Modify: `tools/pr_review.py`

**Interfaces:**
- Consumes: `_post_review_request`, `GATEWAY_RETRY_STATUS_CODES`, and the existing fallback chain.
- Produces: a private helper that retries the unchanged request for gateway and timeout failures.

- [ ] **Step 1: Add retry constants and imports**

  Add `time`, a default retry-attempt count of three total attempts, and a default two-second backoff. Read the backoff from `PR_REVIEW_RETRY_BACKOFF_S` so tests can set it to zero.

- [ ] **Step 2: Add `_post_review_with_retry`**

  Catch only gateway `HTTPError` statuses, `URLError`, and `TimeoutError`; sleep with `backoff * 2**retry_index` between attempts; re-raise the final exception and all non-retryable HTTP errors.

- [ ] **Step 3: Route only the initial strict request through the helper**

  Replace the first `_post_review_request` call in `run_review()` with `_post_review_with_retry()`. Keep the existing no-reasoning and legacy fallback calls unchanged.

### Task 3: Verify and package the change

**Files:**
- Modify: `tools/pr_review.py`
- Modify: `tests/test_pr_review.py`

**Interfaces:**
- Consumes: the retry helper and its tests.
- Produces: a verified implementation with no model configuration changes.

- [ ] **Step 1: Run the complete reviewer test module**

  Run `pytest tests/test_pr_review.py -q` and require all tests to pass.

- [ ] **Step 2: Inspect the diff for scope**

  Run `git diff --check` and verify the diff contains no model or API-mode changes.

- [ ] **Step 3: Commit the focused change**

  Run `git add tools/pr_review.py tests/test_pr_review.py docs/superpowers/specs/2026-07-27-pr-review-retry-design.md docs/superpowers/plans/2026-07-27-pr-review-retry.md && git commit -m "ci: retry transient PR review failures"`.
