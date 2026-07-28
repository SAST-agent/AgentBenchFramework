# Generals HL v4 Replay-Guided Iteration Implementation Plan

> **Execution mode:** implement sequentially in this session with
> test-driven development and verification checkpoints. The user explicitly
> requested direct execution without a separate plan-review pause.

**Goal:** Produce one immutable, scientifically auditable `v3 → v4` Codex
heuristic-learning act and a real 18-case formal score.

**Architecture:** Add a v4 layer around the existing official evaluator,
provider controller, snapshotter, dense diagnostics, and report builder. Freeze
a new strongest-human learning suite and human replay skill, select
high-information decision windows deterministically, import the exact v3
parent, and treat behavior/dense changes as diagnostics rather than gates.

**Tech stack:** Python 3.11, standard library, pytest, official Generals engine,
Codex CLI provider, existing AgentBench tracking/report contracts.

## Global Constraints

- Parent v3 hash is
  `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`.
- Learning opponent is the frozen high-tier
  `advanced-rank02-robinliu-v18`.
- Learning seeds are `285101`, `285202`, `285303`, both seats.
- Formal benchmark remains `generals-hl-pilot-v1`, 18 cases.
- Every runnable v4 receives formal evaluation, regardless of diagnostics.
- No post-finalization summary or event mutation.
- Strict IG and policy KL remain missing; only deterministic behavior change is
  reported.

---

### Task 1: Freeze the v4 learning suite and replay-analysis skill

**Framework files:**

- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `src/agentbench_frame/generals/evaluator.py`
- Test: `tests/generals/test_assets.py`
- Test: `tests/generals/test_evaluator.py`

**Asset files:**

- Create:
  `backend_sources/corpus/28_generals/benchmark/v4-strongest-learning-v1.toml`
- Create:
  `backend_sources/corpus/28_generals/skills/replay-analysis-v1/SKILL.md`

**Interfaces:**

- `Round4LearningConfig`
- `load_round4_learning_config(path, pilot)`
- `build_round4_learning_cases(config, learning)`
- `resolve_replay_skill(agentbench_root, relative_path)`

- [ ] Add failing tests for exact learning ID, opponent, seeds, seats, and
  disjointness from every frozen seed set including v3.
- [ ] Add failing tests for a relative, readable replay skill with a stable
  SHA-256 digest and path-traversal rejection.
- [ ] Confirm the focused tests fail for missing v4 interfaces.
- [ ] Implement immutable records, loading, validation, case construction, and
  skill resolution.
- [ ] Add the frozen manifest and human-authored skill.
- [ ] Run the focused tests and commit Framework and asset changes separately.

### Task 2: Select critical replay windows and build the v4 prompt

**Files:**

- Modify: `src/agentbench_frame/generals/replay.py`
- Modify: `src/agentbench_frame/generals/prompt.py`
- Test: `tests/generals/test_replay_prompt.py`
- Create: `tests/generals/test_prompt_v4.py`

**Interfaces:**

- `CriticalDecision`
- `CriticalLearningEvidence`
- `CriticalWindowSelection`
- `build_critical_learning_evidence(replay, dense_summary, dense_trace)`
- `build_round4_prompt(...)`

- [ ] Add failing tests for first decision, first non-end action, first main
  pressure, steepest territory loss, steepest army loss, first strategic
  opportunity, penultimate/final decisions, de-duplication, and chronological
  ordering.
- [ ] Add failing tests proving derived state features are deterministic and
  source-readable.
- [ ] Add failing prompt tests for exact replay-skill bytes/hash, v3 experience,
  new-seed acceptance, formal/v3-seed rejection, byte cap, and no opponent ID
  leakage.
- [ ] Add tests for exact included episode, decision-record, and serialized-byte
  read counts.
- [ ] Implement selection, serialization, leak checks, and prompt construction.
- [ ] Run focused tests and commit.

### Task 3: Import the exact immutable v3 parent

**Files:**

- Create: `src/agentbench_frame/generals/lineage_v4.py`
- Create: `tests/generals/test_lineage_v4.py`

**Interfaces:**

- `Round4ParentLineage`
- `load_round4_parent(parent_run_dir, expected_hash)`
- `import_parent_v3(lineage, run_dir, snapshotter)`

- [ ] Add failing tests for required complete v3 summary fields, exact v3
  source/manifest equality, hash mismatch, missing v3 score, score history with
  the historical missing point, cumulative learning budget, and exact import.
- [ ] Confirm focused tests fail because the module is missing.
- [ ] Implement read-only validation and import without using the v3 audit
  mutation helper.
- [ ] Run focused tests and commit.

### Task 4: Orchestrate the non-gated v4 pipeline

**Files:**

- Create: `src/agentbench_frame/generals/pipeline_v4.py`
- Create: `tests/generals/test_pipeline_v4.py`

**Interfaces:**

- `Round4PipelineResult`
- `GeneralsHLRound4Pipeline.from_paths(...)`
- `GeneralsHLRound4Pipeline.run()`

- [ ] Add a fake-provider success test covering v3 import, six v3 feedback
  games, prompt/read receipts, one coding act, v4 snapshot/tests, six v4
  validations, diagnostics, and 18 formal cases.
- [ ] Add a runnable-but-no-behavior-change test proving formal evaluation still
  runs and v4 status remains available.
- [ ] Add a runnable-but-negative-dense-delta test proving formal evaluation
  still runs.
- [ ] Add provider-failure, protected-file, and test-failure paths with retained
  artifacts and missing formal score.
- [ ] Add incomplete-validation behavior proving formal evaluation is still
  attempted for a runnable policy.
- [ ] Add assertions for separate learning, validation, evaluation, feedback
  read, provider, and cumulative lineage budgets.
- [ ] Implement append-only events, artifacts, diagnostics, and single
  finalization.
- [ ] Run focused tests and commit.

### Task 5: Expose v4 through CLI, quality checks, and report

**Files:**

- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `tests/generals/test_cli.py`
- Create: `tests/generals/test_cli_v4.py`
- Modify: `tests/test_research_boundaries.py`
- Modify: `tests/test_local_report_research.py`

**Interfaces:**

- `agentbench generals iterate-v4`
- known `feedback_read`, `critical_window_selection`, and
  `behavior_diagnostics` event types
- report support for `evo_score_4`, v4 diagnostics, phase budgets, and a missing
  global AUC across the historical gap

- [ ] Add failing CLI tests for v4 paths, parent hash, skill path, provider
  timeout, and JSON result.
- [ ] Add failing event-quality and report tests.
- [ ] Register `iterate-v4` and known events.
- [ ] Extend report-derived research data without relabeling behavior change as
  information gain.
- [ ] Run focused tests and commit.

### Task 6: Verify implementation and run the real v4 experiment

**Files:**

- Modify: `README.md`
- Modify:
  `backend_sources/corpus/28_generals/README.md`
- Runtime:
  `agentbench_data/runs/28_generals/generals-hl/<v4_run_id>/`

- [ ] Document v4 as pilot-rescue continuation and retain the canonical
  from-scratch experiment as separate future work.
- [ ] Run the complete Framework suite and Generals asset suite.
- [ ] Execute `iterate-v4` with the exact v3 parent and expected hash.
- [ ] Inspect six v3 feedback games, prompt receipt, provider raw JSONL and
  usage, changed-file boundary, tests, v4 hash, six validation games, 18 formal
  games, diagnostics, phase budgets, and event quality.
- [ ] Regenerate the local CI report and inspect its research data.
- [ ] Write a result document with actual score, per-tier/per-seat outcomes,
  dense changes, decision-class changes, budgets, limitations, and reproducible
  command.
- [ ] Run final verification, commit documentation/result support, and report
  actual evidence without claiming unobserved improvement.
