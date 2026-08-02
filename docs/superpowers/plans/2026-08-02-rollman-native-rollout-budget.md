# Rollman Native Rollout Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce and audit a native 70k Codex rollout budget for every Rollman HL act, safely retain useful budget-terminated candidate code, and start the trusted k=4 curriculum run.

**Architecture:** A nested provider budget schema renders the exact Codex TOML and drives a local runtime preflight. The provider classifies native budget termination and derives weighted usage; the controller alone decides whether a changed, policy-compliant candidate earns a real quick screen and can be normalized into a completed scientific candidate.

**Tech Stack:** Python 3.13, dataclasses, JSON/JSONL, TOML emitted by the existing provider, pytest, Codex CLI Responses provider, frozen Rollman evaluator.

## Global Constraints

- Required runtime: `codex-cli 0.146.0-alpha.9.2`.
- Rollout budget: 70,000 weighted tokens per fresh act.
- Reminders: 20,000, 10,000, and 5,000 weighted tokens remaining.
- Token weights: prefill 1.0 and sampling 1.0; cached input excluded.
- Timeouts: 420 seconds hard and 120 seconds without stream progress.
- k=4 remains linear sibling exploration, not a four-branch tree.
- API credentials remain confined to provider subprocesses and absent from artifacts.

---

### Task 1: Strict Provider Budget Configuration and TOML

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/hl/provider.py`
- Modify: `tests/hl/test_config.py`
- Modify: `tests/hl/test_provider.py`

**Interfaces:**
- Produces: `RolloutBudgetConfig` nested at `ProviderConfig.rollout_budget`.
- Produces: generated `[features.rollout_budget]` TOML and provider fingerprint.

- [x] Write tests that parse valid nested values, reject invalid limits/reminders/weights, and assert exact secret-free TOML.
- [x] Run the focused tests and verify failures are caused by the missing schema and TOML.
- [x] Implement the dataclass, strict nested parsing, validation, and TOML rendering.
- [x] Run the focused tests to green.

### Task 2: Runtime Preflight and Budget Telemetry

**Files:**
- Modify: `src/agentbench_frame/hl/provider.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `src/agentbench_frame/tracking/provider.py`
- Modify: `tests/hl/test_provider.py`
- Modify: `tests/hl/test_cli.py`

**Interfaces:**
- Produces: `CodexSessionProvider.preflight() -> dict[str, object]`.
- Produces: provider metadata keys `termination_reason`, `weighted_tokens`, `rollout_budget_limit_tokens`, and `rollout_budget_exhausted`.

- [x] Write tests using an executable fixture that reports an exact version, reads the generated feature flag, and emits `SessionBudgetExceeded` JSONL.
- [x] Run the focused tests and verify the new behavior is absent.
- [x] Implement exact-version/feature preflight, secret-free preflight persistence, termination classification, partial-stream access audit, and weighted-token derivation.
- [x] Invoke preflight before the first paid act and persist its result.
- [x] Run the focused tests to green.

### Task 3: Safe Adoption of Budget-Terminated Candidate Code

**Files:**
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `tests/hl/test_controller.py`

**Interfaces:**
- Consumes: `ProviderInvocation.metadata["rollout_budget_exhausted"]`.
- Produces: completed candidate invocation with `accepted_after_budget_exhaustion=true` only after a changed snapshot and complete quick screen.

- [x] Write controller tests for a changed candidate that passes quick screen, an unchanged candidate, an access violation, and a failed quick screen.
- [x] Run the focused tests and verify the safe-adoption case fails before implementation.
- [x] Implement the minimal eligibility branch without changing ordinary completion, timeout, planner, or reducer semantics.
- [x] Persist budget facts in checkpoint and act events.
- [x] Run the focused tests to green.

### Task 4: Freeze Experiment Configuration and Verify

**Files:**
- Modify: `configs/hl/29_rollman-k4-repair.yaml`
- Modify: `docs/hl/rollman-k4-runbook.md`
- Test: `tests/hl/test_local_config.py`

**Interfaces:**
- Produces: the exact approved local experiment configuration.

- [x] Add a config integration test that loads the Rollman k=4 budget values.
- [x] Run it red, then freeze the version, native budget, and 420/120 deadlines in YAML.
- [x] Document the budget and safe-adoption semantics in the runbook.
- [x] Run provider/config/controller tests, then the entire repository suite.
- [x] Run provider preflight against the generated real Codex home without invoking a model.

### Task 5: Start and Monitor the Paid k=4 Cycle

**Files:**
- Runtime output: `.agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v3-official/`

**Interfaces:**
- Consumes: trusted imported `v000000`, active opponent rank15, and the approved provider budget.
- Produces: immutable candidates, JSONL/checkpoints, Rollman replays, curriculum facts, and integer-iteration curves.

- [ ] Resume the trusted run only if the frozen config matches; otherwise create a content-identical trusted continuation run with explicit provenance.
- [ ] Monitor planner, four candidates, top-two repairs, reducer, local matches, budget facts, and API errors.
- [ ] Regenerate IG, Elo, win-rate, and mean-margin curves after each completed cycle.
- [ ] Continue the weakest-failed curriculum until rank15 and rank14 pass five-seed certification and the champion is 16/16.
