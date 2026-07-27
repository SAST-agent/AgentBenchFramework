# Generals HL v3 Pilot Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one gated, strongest-human replay-driven Codex act from the exact
pilot v2 snapshot and produce an auditable v3 result.

**Architecture:** Add a small round-3 layer around the existing official
evaluator, provider controller, snapshotter, dense diagnostics, and report
builder. Keep strongest-human learning configuration separate from the frozen
formal manifest, and keep observable behavior classification separate from
strategy internals.

**Tech Stack:** Python 3.11, standard library, pytest, official Generals engine,
existing AgentBench tracking/provider/report contracts.

## Global Constraints

- The run is a rescue of the old pilot lineage, not the canonical
  from-scratch experiment.
- Formal benchmark `generals-hl-pilot-v1` is unchanged.
- Learning uses only high-tier opponent `advanced-rank02-robinliu-v18`, seeds
  `284101`, `284202`, `284303`, and both seats.
- Formal and calibration seeds or trajectories never enter the prompt.
- Strategy code may be restructured but must remain deterministic,
  source-readable, and dependency-free.
- The failed prior coding-agent act remains a missing score point.

---

### Task 1: Freeze and validate the v3 learning suite

**Files:**
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `src/agentbench_frame/generals/evaluator.py`
- Test: `tests/generals/test_assets.py`
- Test: `tests/generals/test_evaluator.py`
- Create: `../AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v3-strongest-learning-v1.toml`

**Interfaces:**
- Produces: `Round3LearningConfig`
- Produces: `load_round3_learning_config(path: Path, pilot: PilotConfig) -> Round3LearningConfig`
- Produces: `build_round3_learning_cases(config: PilotConfig, learning: Round3LearningConfig) -> tuple[BenchmarkCase, ...]`

- [ ] Add failing tests for exact learning ID, strongest opponent, three unique
  seeds, both seats, and disjointness from all known frozen seeds.
- [ ] Run the focused asset/evaluator tests and confirm missing symbols fail.
- [ ] Implement the immutable config, loader, validator, and six-case builder.
- [ ] Add the frozen learning manifest with the declared values.
- [ ] Run focused tests and the complete Generals asset tests.
- [ ] Commit with `feat(generals): freeze strongest-human v3 learning suite`.

### Task 2: Add observable decision diagnostics and behavior gate

**Files:**
- Modify: `src/agentbench_frame/generals/measurement.py`
- Test: `tests/generals/test_measurement.py`

**Interfaces:**
- Produces: `DecisionClassSummary`
- Produces: `classify_macro_action(state: Mapping[str, Any], seat: int, action: Sequence[Sequence[int]]) -> str`
- Produces: `summarize_decision_classes(probes: Sequence[ProbeState], actions: Mapping[str, tuple[tuple[int, ...], ...]]) -> DecisionClassSummary`
- Produces: `BehaviorGateResult`
- Produces: `evaluate_behavior_gate(...) -> BehaviorGateResult`

- [ ] Add failing tests for end-only, main army, non-main army, upgrade, skill,
  technology, super-weapon, recruit, malformed/other classifications.
- [ ] Add failing tests proving the gate rejects zero disagreement, zero
  non-main movement, invalid validation matches, and no dense improvement.
- [ ] Run focused tests and confirm expected assertion or import failures.
- [ ] Implement classification, stable counts/rates, paired dense deltas, and
  all-of gate semantics.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit with `feat(generals): gate v3 on observable behavior`.

### Task 3: Build compact v3 prompt and versioned experience contract

**Files:**
- Modify: `src/agentbench_frame/generals/prompt.py`
- Test: `tests/generals/test_prompt_v3.py`

**Interfaces:**
- Produces: `build_round3_prompt(...) -> PromptBuildResult`
- Consumes: `CompactLearningEvidence`, `DecisionClassSummary`

- [ ] Add failing tests that high-tier learning is accepted, formal and
  calibration-held-out seeds are rejected, prompt bytes stay at or below
  65,536, full board state is absent, and `EXPERIENCE.md` is required.
- [ ] Run focused tests and confirm `build_round3_prompt` is missing.
- [ ] Implement a prompt that permits policy restructuring, requires
  evidence-backed experience, includes the dominant-rule diagnosis, and omits
  “one coherent improvement.”
- [ ] Run focused tests and confirm they pass.
- [ ] Commit with `feat(generals): build high-signal v3 replay prompt`.

### Task 4: Import the exact parent v2 lineage

**Files:**
- Create: `src/agentbench_frame/generals/lineage_v3.py`
- Test: `tests/generals/test_lineage_v3.py`

**Interfaces:**
- Produces: `Round3ParentLineage`
- Produces: `load_round3_parent(parent_run_dir: Path, expected_hash: str) -> Round3ParentLineage`
- Produces: `import_parent_v2(lineage: Round3ParentLineage, run_dir: Path, snapshotter: LocalWorkspaceSnapshotter) -> WorkspaceManifest`

- [ ] Add failing tests for a complete v2 parent, source/manifest tampering,
  hash mismatch, missing scores, recovery linkage, and exact import.
- [ ] Run focused tests and confirm the new module is missing.
- [ ] Implement v2 validation/import without changing the round-2 lineage API.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit with `feat(generals): import immutable v2 rescue parent`.

### Task 5: Orchestrate the gated v3 pipeline

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v3.py`
- Test: `tests/generals/test_pipeline_v3.py`

**Interfaces:**
- Produces: `Round3PipelineResult`
- Produces: `GeneralsHLRound3Pipeline.from_paths(...)`
- Produces: `GeneralsHLRound3Pipeline.run() -> Round3PipelineResult`
- Consumes: round-3 learning cases, v2 lineage, prompt builder, decision gate,
  existing evaluator/provider/snapshot/run contracts.

- [ ] Add a fake-provider success test covering v2 learning, one coding act,
  v3 snapshot, tests, offline behavior measurement, v3 learning validation,
  passed gate, and formal evaluation.
- [ ] Add gate-failure tests proving formal evaluation is skipped while v3,
  provider logs, test output, and gate event remain saved.
- [ ] Run focused tests and confirm the pipeline module is missing.
- [ ] Implement the pipeline with append-only events and phase-separated
  budgets.
- [ ] Run focused tests and confirm all paths pass.
- [ ] Commit with `feat(generals): orchestrate gated v3 rescue`.

### Task 6: Expose v3 in CLI, quality diagnostics, and report

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Modify: `src/agentbench_frame/report/builder.py`
- Test: `tests/generals/test_cli.py`
- Test: `tests/test_research_boundaries.py`
- Test: `tests/test_local_report_research.py`

**Interfaces:**
- Produces: `agentbench generals iterate-v3`
- Produces: known `decision_class_summary` and `behavior_gate` events
- Produces: v0/v1/v2/v3 report history with the prior missing act gap retained

- [ ] Add failing CLI, quality, and report tests.
- [ ] Run focused tests and confirm missing v3 behavior.
- [ ] Register `iterate-v3` with learning manifest, parent, hash, data,
  executable, and timeout arguments.
- [ ] Add the two event types and expose gate/distribution data in research
  report JSON without calling it policy KL or information gain.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit with `feat(report): expose gated Generals v3 results`.

### Task 7: Verify implementation and run the live v3 rescue

**Files:**
- Modify: `README.md`
- Modify: `../AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/README.md`
- Create at runtime: `agentbench_data/runs/28_generals/generals-hl/<run_id>/`

**Interfaces:**
- Consumes: `agentbench generals iterate-v3`
- Produces: immutable v3 source, patch, provider JSONL, learning/formal
  artifacts, events, summary, quality report, and refreshed local CI report.

- [ ] Document the pilot-rescue command and distinguish it from the future
  strict from-scratch default.
- [ ] Run all Framework tests and all Generals asset tests.
- [ ] Run `iterate-v3` against the exact recovery v2 parent and expected hash.
- [ ] Inspect the gate event, v3 manifest, tests log, six validation cases,
  formal completeness when gated, provider usage, and event quality.
- [ ] Rebuild the local CI report and verify v3 and the failed-act gap are
  represented without interpolation.
- [ ] Commit documentation and report-support changes.
- [ ] Report actual v3 score and dense deltas; do not claim improvement unless
  fresh artifacts demonstrate it.

