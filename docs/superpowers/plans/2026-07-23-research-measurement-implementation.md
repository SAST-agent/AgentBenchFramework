# Research Measurement Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first framework-native measurement layer for versioned coding-agent evaluation, budgets, append-only events, benchmark score, and two-source policy-change data.

**Architecture:** Keep pure calculations in `eval`/`measurement`, event persistence in `tracking`, and leave provider process control outside the framework. Extend existing wrappers and `Run` additively, preserving the old `event` field and 4-tuple environment API while recording richer fields when available.

**Tech Stack:** Python standard library, dataclasses, JSONL, unittest; no new mandatory dependencies.

## Global Constraints

- Preserve existing public imports and old event fields.
- `events.jsonl` is the fact source and must be append-only.
- Unknown provider usage is `None`/`unknown`, never zero.
- Benchmark aggregate score is absent for incomplete evaluation.
- No accept/reject or automatic rollback.
- `trajectory_kl` is derived; the two primary measurement objects are local KL trace and occupancy shift.

### Task 1: Add pure benchmark and curve calculations

**Files:**
- Create: `src/agentbench_frame/eval/benchmark.py`
- Create: `src/agentbench_frame/eval/curves.py`
- Modify: `src/agentbench_frame/eval/__init__.py`
- Test: `tests/test_measurement_contracts.py`

**Interfaces:**
- `BenchmarkCase(case_id, opponent, seed, first_player, metadata)`.
- `GameResult(case_id, outcome, valid=True, error=None, metadata={})`, where outcome is `win`, `loss`, or `draw`.
- `BenchmarkSpec(version, cases)`.
- `BenchmarkEvaluation(results)` with `status`, `score`, `wins`, `losses`, `draws`, and `per_case`.
- `evaluate_benchmark(spec, results) -> BenchmarkEvaluation`.
- `trapezoid_auc(points) -> float`, where points are `(x, score)` and incomplete/None scores split the curve.

- [ ] Write failing tests for tie score, missing case/incomplete status, duplicate case rejection, and AUC trapezoid integration.
- [ ] Run `PYTHONPATH=src python -m unittest tests.test_measurement_contracts...` and confirm failures are missing symbols/behavior.
- [ ] Implement dataclasses and pure functions with explicit validation.
- [ ] Run the focused tests, then the existing suite.

### Task 2: Add policy-change measurement functions

**Files:**
- Create: `src/agentbench_frame/eval/information_gain.py`
- Modify: `src/agentbench_frame/eval/__init__.py`
- Test: `tests/test_measurement_contracts.py`

**Interfaces:**
- `epsilon_regularize(probs, legal_count, epsilon) -> list[float]`.
- `policy_kl(new_probs, old_probs, epsilon=None) -> float`.
- `episode_policy_kl_trace(new_policy, old_policy, contexts, legal_actions, epsilon=None) -> list[float]`.
- `occupancy_histogram(state_ids) -> dict[str, float]`.
- `occupancy_shift(new_state_ids, old_state_ids, smoothing=...) -> float`.
- `trajectory_kl_from_trace(trace) -> float`.

- [ ] Write failing tests for normalization, deterministic epsilon smoothing, zero unchanged KL, invalid support, per-context trace, and occupancy shift.
- [ ] Run the focused tests and confirm they fail before implementation.
- [ ] Implement numerically stable finite-horizon calculations using natural logarithms.
- [ ] Run focused and full tests.

### Task 3: Make event persistence schema-safe and append-only

**Files:**
- Modify: `src/agentbench_frame/tracking/writer.py`
- Modify: `src/agentbench_frame/tracking/run.py`
- Modify: `src/agentbench_frame/tracking/records.py`
- Test: `tests/test_tracking_contracts.py`

**Interfaces:**
- `JSONLWriter(path, ..., append=True)` appends by default and preserves one JSON object per line.
- `Run.write(event_type, **kwargs)` emits `schema_version`, `event_id`, `event_type`, `event` alias, `run_id`, `created_at`.
- Existing event-specific fields remain unchanged.

- [ ] Write failing tests for two writer instances appending, common fields, unique event ids, and old `event` compatibility.
- [ ] Run focused tests and confirm failure.
- [ ] Implement additive schema normalization and append mode without changing caller signatures.
- [ ] Run focused and full tests.

### Task 4: Add budget ledger and act/version records

**Files:**
- Create: `src/agentbench_frame/tracking/budget.py`
- Create: `src/agentbench_frame/tracking/iteration.py`
- Modify: `src/agentbench_frame/tracking/__init__.py`
- Test: `tests/test_tracking_contracts.py`

**Interfaces:**
- `BudgetLedger.add(phase, episodes=0, env_steps=0, prompt_tokens=None, completion_tokens=None, time_s=None)`.
- `BudgetLedger.snapshot() -> dict` with learning/evaluation/total fields.
- `ActRecord(act_id, provider, status, version_before, version_after, changed_files, tool_call_count, prompt_tokens, completion_tokens, total_tokens, token_accuracy, elapsed_time_s, raw_output_ref)`.
- `VersionedActRecorder.begin_act(...)`, `.finish_act(...)`, `.record_evaluation(...)`.

- [ ] Write failing tests for phase totals, unknown token propagation, unchanged content hash creating a new logical version, and failed act retention.
- [ ] Run focused tests and confirm failure.
- [ ] Implement small dataclasses and append-only event emission through an injected callback.
- [ ] Run focused and full tests.

### Task 5: Integrate raw lifecycle and benchmark fields into Run

**Files:**
- Modify: `src/agentbench_frame/tracking/run.py`
- Modify: `src/agentbench_frame/tracking/wrappers.py`
- Modify: `src/agentbench_frame/runner/eval_runner.py`
- Test: `tests/test_tracking_contracts.py`, `tests/test_benchmark_contracts.py`

- [ ] Write failing tests for `Run.log_budget`, `Run.log_act`, `Run.log_version`, and named-agent win/draw scoring.
- [ ] Run focused tests and confirm failure.
- [ ] Add additive methods and ensure summary includes the ledger snapshot without removing old fields.
- [ ] Update evaluation runner to retain per-game results and expose benchmark spec/results when supplied; do not invent a default frozen dataset.
- [ ] Run focused and full tests.

### Task 6: Document remaining external boundary and verify

**Files:**
- Modify: `docs/research/information-gain-design.md`
- Modify: `docs/research/research-methodology-summary.md`
- Test: all tests

- [ ] Add a short implementation-status section distinguishing framework-native support from provider adapter responsibilities.
- [ ] Run `PYTHONPATH=src python -m unittest discover -s tests -v`.
- [ ] Run `git diff --check` and inspect status/diff for unrelated changes.
