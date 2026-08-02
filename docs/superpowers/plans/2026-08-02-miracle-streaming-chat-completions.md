# Miracle Streaming Chat Completions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SSE streaming the default Chat Completions transport, remove the default completion-length limit, and account against a configurable one-million-token context budget before rerunning three real iterations.

**Architecture:** Extend `LLMConfig` with default streaming, optional `max_tokens`, and `max_context_tokens`. Split the client response path into JSON and SSE readers that normalize into the existing proposal parser, keeping the Loop and Results schemas stable except for additional stream telemetry.

**Tech Stack:** Python 3.11 standard library `urllib.request`, JSON, OpenAI-compatible SSE, pytest.

## Global Constraints

- `stream` defaults to `true`; explicit `false` retains non-streaming compatibility.
- Do not retry streaming failures as non-streaming.
- `max_tokens` defaults to absent and is sent only when configured.
- `max_context_tokens` defaults to `1_000_000`, is not sent to the API, and uses authoritative returned usage.
- Do not estimate tokens locally or add a tokenizer dependency.
- Preserve partial stream content, usage, timings, and failure reason.

---

### Task 1: Streaming and Context Configuration

**Files:**
- Modify: `src/agentbench_frame/miracle/loop_config.py`
- Modify: `tests/miracle/test_loop_config.py`
- Modify: `examples/miracle-loop.toml`

**Interfaces:**
- `LLMConfig.stream: bool = True`
- `LLMConfig.max_tokens: int | None = None`
- `LLMConfig.max_context_tokens: int = 1_000_000`

- [ ] Write tests asserting defaults, explicit `stream=false`, optional positive `max_tokens`, and rejection of nonpositive context limits.
- [ ] Run `uv run --with pytest python -m pytest tests/miracle/test_loop_config.py -q` and verify RED.
- [ ] Implement strict TOML parsing and public serialization.
- [ ] Rerun the configuration tests and verify PASS.
- [ ] Commit with `feat(miracle): configure default streaming context budget`.

### Task 2: SSE Reader and Normalized Response

**Files:**
- Modify: `src/agentbench_frame/miracle/llm_client.py`
- Modify: `tests/miracle/test_llm_client.py`

**Interfaces:**
- `_read_stream(response, started: float) -> tuple[dict, float]`
- Normalized response adds `stream`, `chunk_count`, `first_chunk_seconds`, and `usage_missing`
- New error stages: `stream_chunk_json`, `stream_incomplete`

- [ ] Add a local SSE fixture emitting heartbeat lines, split reasoning/content deltas, final usage, and `[DONE]`; assert reconstructed content and telemetry.
- [ ] Add tests for malformed JSON and EOF without `[DONE]`, including preserved partial response and usage.
- [ ] Add an explicit `stream=false` test asserting the JSON path still works and `max_tokens` is omitted by default.
- [ ] Run `uv run --with pytest python -m pytest tests/miracle/test_llm_client.py -q` and verify RED.
- [ ] Implement line-oriented SSE parsing, normalized response construction, and shared proposal parsing.
- [ ] Rerun client tests and verify PASS.
- [ ] Commit with `feat(miracle): stream Chat Completions by default`.

### Task 3: Context Budget and Stream Telemetry in Loop

**Files:**
- Modify: `src/agentbench_frame/miracle/loop.py`
- Modify: `src/agentbench_frame/miracle/run_store.py`
- Modify: `tests/miracle/test_loop.py`
- Modify: `skills/miracle-harness/SKILL.md`

**Interfaces:**
- `BudgetLedger.charge_context(total_tokens: int, limit: int) -> None`
- Saved LLM response contains stream telemetry and real usage on success or failure

- [ ] Add tests that successful streaming usage is charged, proposal failures retain usage, and reported usage above `max_context_tokens` creates a preserved `context_tokens` failure.
- [ ] Run Loop tests and verify RED.
- [ ] Implement context checking after charging API usage; save telemetry without API secrets.
- [ ] Document default streaming, optional non-streaming, omitted `max_tokens`, and the 1M context budget.
- [ ] Run Loop, store, and client tests and verify PASS.
- [ ] Commit with `feat(miracle): account streaming context usage`.

### Task 4: Real API Verification and Three-Iteration Run

**Files:**
- Modify only temporary `/tmp` configuration for the live run.
- Write Run artifacts beneath `AgentBenchResults/runs/24_miracle/temporary_deepseek_v4_flash/`.

- [ ] Run one minimal streaming request and verify multiple SSE chunks, `[DONE]`, content, usage, and no 60-second idle 504.
- [ ] Run the complete command with `stream=true`, no `max_tokens`, `max_context_tokens=1000000`, and three iterations.
- [ ] Inspect all iteration statuses, replay/trace files, score/IG curves, token/time accounting, and secret absence.
- [ ] Run `agentbench data check` against AgentBenchResults.
- [ ] Run `uv run --with pytest python -m pytest tests/miracle -q` and `git diff --check`.
- [ ] Remove generated Framework caches and commit code changes; do not delete failed Run records.
