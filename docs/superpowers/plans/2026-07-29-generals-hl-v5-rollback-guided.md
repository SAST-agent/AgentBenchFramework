# Generals HL v5 Rollback-Guided Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one audited v5 act that retains v4 in the lineage,
starts the editable policy from immutable v3, learns from paired v3/v4
strongest-human replays, and evaluates every runnable v5.

**Architecture:** Add a frozen v5 learning asset, a v5-specific lineage loader
and prompt builder, then orchestrate them in `GeneralsHLRound5Pipeline`. The
pipeline verifies the v4 parent and v3 rollback snapshot, inherits the separate
audited campaign receipt, runs 12 pre-act learning games, invokes Codex once
from v3 source, runs six v5 validation games and the unchanged 18-case formal
matrix, and exposes the immutable result in the existing report.

**Tech Stack:** Python 3.11+, dataclasses, TOML/JSON/JSONL, pytest, Jinja2,
official Generals Python logic, Codex CLI.

## Global Constraints

- Lineage parent is complete v4 run `20260728_1519_7b242596`, content hash
  `5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f`.
- Editable rollback source is embedded v3, content hash
  `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`.
- Learning opponent is `advanced-rank02-robinliu-v18`; seeds are
  `286101`, `286202`, `286303`; seats are ordered `[0, 1]`.
- Formal seeds, outcomes, trajectories, and opponent source never enter the
  Codex prompt.
- v3 and v4 each run all six pre-act cases; v5 runs the same six only after
  the act and then runs all 18 formal cases.
- Every runnable v5 is formally evaluated; behavior, dense values, and score
  improvement are non-blocking diagnostics.
- `main.py` and `state_view.py` remain protected and identical to rollback v3.
- Every run, including failures, retains raw matches, replay/protocol streams,
  provider JSONL/stderr, prompt, token/tool/time, patches, snapshots, tests,
  events, summary, quality, and budgets.
- Finalized run summaries are immutable. Corrections use separate hashed
  receipts.
- Historical score gaps, strict policy KL, and epistemic information gain
  remain missing and are never interpolated.

---

### Task 1: Freeze and validate the v5 learning suite

**Files:**
- Create in AgentBench assets:
  `backend_sources/corpus/28_generals/benchmark/v5-rollback-learning-v1.toml`
- Create: `tests/generals/fixtures/v5-rollback-learning-v1.toml`
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `tests/generals/test_assets.py`

**Interfaces:**
- Consumes: `PilotConfig`, its ordered high/medium/low opponents, and every
  previously frozen seed set.
- Produces:
  `Round5LearningConfig(learning_id, opponent_id, seeds, seats)` and
  `load_round5_learning_config(path: Path, pilot: PilotConfig)
  -> Round5LearningConfig`.

- [ ] **Step 1: Write the failing manifest tests**

```python
def test_round5_learning_manifest_is_new_strongest_human_matrix():
    config = load_round5_learning_config(
        FIXTURES / "v5-rollback-learning-v1.toml",
        load_pilot_config(FIXTURES / "pilot-v1.toml"),
    )
    assert config.learning_id == "generals-hl-v5-rollback-strongest-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.seeds == (286101, 286202, 286303)
    assert config.seats == (0, 1)


@pytest.mark.parametrize("seed", [280101, 281101, 284101, 285101])
def test_round5_learning_manifest_rejects_every_frozen_seed(tmp_path, seed):
    path = tmp_path / "v5.toml"
    path.write_text(
        'learning_id = "generals-hl-v5-rollback-strongest-v1"\n'
        'opponent_id = "advanced-rank02-robinliu-v18"\n'
        f"seeds = [{seed}, 286202, 286303]\n"
        "seats = [0, 1]\n"
    )
    with pytest.raises(AssetValidationError, match="previously frozen"):
        load_round5_learning_config(path, PILOT)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_assets.py -k round5`

Expected: collection/import failure because the round-5 type and loader do not
exist.

- [ ] **Step 3: Add the exact type, constants, loader, fixture, and asset**

```python
@dataclass(frozen=True)
class Round5LearningConfig:
    learning_id: str
    opponent_id: str
    seeds: tuple[int, ...]
    seats: tuple[int, ...]
```

The loader must require the exact ID, exactly three unique seeds, no overlap
with pilot/calibration/v3/v4 frozen sets, seats `(0, 1)`, and the pilot's
highest-tier opponent. The two TOML files contain the exact four values from
the global constraints.

- [ ] **Step 4: Run focused and asset tests**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_assets.py`

Expected: all asset tests pass.

- [ ] **Step 5: Commit each repository**

Framework:

```bash
git add src/agentbench_frame/generals/models.py \
  src/agentbench_frame/generals/assets.py \
  tests/generals/fixtures/v5-rollback-learning-v1.toml \
  tests/generals/test_assets.py
git commit -m "feat(generals): freeze v5 rollback learning suite"
```

AgentBench assets:

```bash
git add backend_sources/corpus/28_generals/benchmark/v5-rollback-learning-v1.toml
git commit -m "feat(generals): add v5 rollback learning matrix"
```

### Task 2: Load the exact v4 parent, v3 rollback source, and budget receipt

**Files:**
- Create: `src/agentbench_frame/generals/lineage_v5.py`
- Create: `tests/generals/test_lineage_v5.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Modify: `tests/test_research_boundaries.py`

**Interfaces:**
- Consumes: v4 parent `summary.json`, `versions/v3`, `versions/v4`, and the
  source-hashed v4 campaign receipt.
- Produces:
  `Round5ParentLineage`, `load_round5_parent(...) -> Round5ParentLineage`, and
  `import_round5_sources(...) -> tuple[WorkspaceManifest, WorkspaceManifest]`.

- [ ] **Step 1: Write failing parent and rollback tests**

```python
def test_round5_parent_uses_v4_lineage_v3_workspace_and_audited_budget(tmp_path):
    parent, v3_manifest, v4_manifest, receipt = make_v5_parent(tmp_path)
    lineage = load_round5_parent(
        parent,
        expected_parent_hash=v4_manifest.content_hash,
        expected_rollback_hash=v3_manifest.content_hash,
        campaign_budget_receipt=receipt,
    )
    assert lineage.parent_version == "v4"
    assert lineage.starting_version == "v3"
    assert lineage.global_act_count == 5
    assert lineage.prior_score_history == (
        0.0, 0.0, None, 0.0, 7 / 18, 4 / 18,
    )
    assert lineage.learning_budget["learning_episodes"] == 58


def test_round5_parent_rejects_receipt_or_snapshot_tampering(tmp_path):
    parent, v3_manifest, v4_manifest, receipt = make_v5_parent(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["success_summary_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary hash"):
        load_round5_parent(
            parent,
            v4_manifest.content_hash,
            v3_manifest.content_hash,
            receipt,
        )
```

- [ ] **Step 2: Run the lineage tests and verify RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_lineage_v5.py`

Expected: module import failure because `lineage_v5.py` does not exist.

- [ ] **Step 3: Implement immutable lineage loading and source import**

`load_round5_parent` verifies run status, complete v0-v4 scores, exact score
history, act count five, v3/v4 manifests against source bytes, both expected
hashes, protected-file equality, receipt mutation policy, receipt success run
ID, exact parent-summary SHA-256, and audited `after` budget.

`import_round5_sources` copies v3 into `workspace` and `versions/v3/source`,
copies v4 into `versions/v4/source`, initializes workspace git metadata, writes
both manifests, and writes:

```json
{
  "parent_version": "v4",
  "starting_version": "v3",
  "rollback_source_version": "v3"
}
```

It must never write into the parent run or receipt.

- [ ] **Step 4: Add and validate the rollback event type**

Add `version_rollback` to `KNOWN_EVENT_TYPES` and to the research-boundary
contract. The event requires IDs and carries only hashes/version labels, never
formal trajectory data.

- [ ] **Step 5: Run focused tests**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_lineage_v5.py tests/test_research_boundaries.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/lineage_v5.py \
  src/agentbench_frame/tracking/quality.py \
  tests/generals/test_lineage_v5.py tests/test_research_boundaries.py
git commit -m "feat(generals): import audited v3 rollback for v5"
```

### Task 3: Build complete paired v3/v4 replay feedback

**Files:**
- Create: `src/agentbench_frame/generals/prompt_v5.py`
- Create: `tests/generals/test_prompt_v5.py`
- Modify: `src/agentbench_frame/generals/replay.py`
- Modify: `tests/generals/test_replay_v4.py`

**Interfaces:**
- Consumes: `CriticalLearningEvidence` for versions v3 and v4,
  `DecisionClassSummary` per version, both source strategy/experience texts,
  official rules, and the replay skill.
- Produces:
  `VersionedCriticalEvidence(version, evidence)` and
  `build_round5_prompt(...) -> PromptBuildResult`.

- [ ] **Step 1: Write a failing six-window selection test**

```python
def test_v5_window_selection_keeps_at_most_six_declared_reasons():
    evidence, receipt = build_critical_learning_evidence(
        replay_with_all_critical_events(),
        dense_summary(),
        dense_trace(),
        max_decisions=6,
        selection_reasons=(
            "first_decision",
            "first_non_end_action",
            "first_main_pressure",
            "before_steepest_territory_drop",
            "before_steepest_army_drop",
            "final_decision",
        ),
    )
    assert len(evidence.decisions) <= 6
    assert receipt.selected_state_ids == tuple(
        item.state_id for item in evidence.decisions
    )
```

- [ ] **Step 2: Verify the selection test is RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_replay_v4.py -k v5`

Expected: failure because the selector does not accept the v5 reason contract.

- [ ] **Step 3: Parameterize the critical selector without changing v4**

Keep the current v4 default behavior unchanged. Add an explicit reason-order
argument used only by v5, deduplicate state IDs, order selected decisions
chronologically, and preserve exact omission counts.

- [ ] **Step 4: Write failing prompt completeness and leak tests**

```python
def test_v5_prompt_groups_all_twelve_versioned_episodes():
    records = tuple(
        VersionedCriticalEvidence(version, evidence(seed, seat))
        for seed in (286101, 286202, 286303)
        for seat in (0, 1)
        for version in ("v3", "v4")
    )
    result = build_round5_prompt(
        benchmark_id="generals-hl-pilot-v1",
        v3_strategy="global stack planner",
        v3_experience="retained v3 experience",
        v4_strategy="reserve planner",
        v4_experience="retained v4 experience",
        rules_text="official rules",
        replay_skill_text="human replay skill",
        replay_skill_sha256="a" * 64,
        evidence=records,
        decision_classes={"v3": classes(), "v4": classes()},
        dense_deltas={"terminal_army_margin": -10.0},
        max_bytes=131_072,
    )
    assert result.feedback_episodes_read == 12
    assert not result.omitted_episode_ids
    assert "v3:learn5-high-s286101-p0" in result.included_episode_ids
    assert "v4:learn5-high-s286303-p1" in result.included_episode_ids
    assert "formal score" not in result.prompt.lower()


@pytest.mark.parametrize("seed", [280101, 284101, 285101])
def test_v5_prompt_rejects_formal_or_historical_episode_records(seed):
    with pytest.raises(ValueError, match="forbidden round-5 evidence seed"):
        build_prompt_with_seed(seed)
```

- [ ] **Step 5: Verify prompt tests are RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_prompt_v5.py`

Expected: module import failure because `prompt_v5.py` does not exist.

- [ ] **Step 6: Implement the paired prompt builder**

Sort by seed, seat, then version. Validate versions are exactly v3/v4, every
record is high tier and uses a new v5 seed, and skill hash is lowercase
SHA-256. Prefix each episode receipt with its version, append only complete
episode records, count actual serialized decisions/bytes, reject leaks through
the existing `_reject_leaks`, and use a 131,072-byte default cap.

- [ ] **Step 7: Run replay and prompt tests**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_replay_v4.py tests/generals/test_prompt_v4.py tests/generals/test_prompt_v5.py`

Expected: v4 compatibility and all v5 tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/generals/replay.py \
  src/agentbench_frame/generals/prompt_v5.py \
  tests/generals/test_replay_v4.py tests/generals/test_prompt_v5.py
git commit -m "feat(generals): build paired v5 replay feedback"
```

### Task 4: Orchestrate the rollback-guided v5 lifecycle

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v5.py`
- Create: `tests/generals/test_pipeline_v5.py`

**Interfaces:**
- Consumes: `Round5ParentLineage`, `Round5LearningConfig`,
  `VersionedCriticalEvidence`, `PromptBuildResult`, `GeneralsEvaluator`, and
  `ProviderAdapter`.
- Produces: `Round5PipelineResult` and
  `GeneralsHLRound5Pipeline.from_paths(...).run()`.

- [ ] **Step 1: Write the failing end-to-end lifecycle test**

```python
def test_v5_runs_paired_learning_one_act_validation_and_formal(tmp_path):
    pipeline, evaluator, provider = make_pipeline(
        tmp_path,
        provider=RewritingProvider(),
    )
    result = pipeline.run()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert evaluator.calls == [
        ("v3", "learning", 6),
        ("v4", "learning", 6),
        ("v5", "validation", 6),
        ("v5", "evaluation", 18),
    ]
    assert provider.calls == 1
    assert summary["parent_version"] == "v4"
    assert summary["starting_version"] == "v3"
    assert summary["rollback_source_version"] == "v3"
    assert summary["evo_score_5"] == 0.5
    assert summary["score_history"] == [
        0.0, 0.0, None, 0.0, 7 / 18, 4 / 18, 0.5,
    ]
    assert summary["budget"]["learning_episodes"] == 12
    assert summary["budget"]["validation_episodes"] == 6
    assert summary["budget"]["evaluation_episodes"] == 18
    assert summary["feedback_read"]["episodes"] == 12
```

- [ ] **Step 2: Verify lifecycle test is RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_pipeline_v5.py`

Expected: module import failure because `pipeline_v5.py` does not exist.

- [ ] **Step 3: Implement pre-act setup and paired learning**

Load and verify lineage before run creation. Inside a normal `Run` lifecycle,
save formal/learning specs and replay skill, import v3/v4 snapshots, emit
`version_rollback`, prepare opponents, run all v3 learning cases followed by
all v4 learning cases, compute dense summaries, and abort before Codex unless
both suites are 6/6 valid.

- [ ] **Step 4: Implement prompt receipt and provider act**

Build critical evidence for all 12 episodes, require all 12 version-prefixed
IDs, write prompt JSON/Markdown/manifest, invoke the provider from the v3
workspace, retain raw JSONL/stderr/tool/token/time, validate edit boundaries,
run strategy tests, snapshot v5, and write both v3-to-v5 and v4-to-v5 patches.

- [ ] **Step 5: Implement non-gated validation and formal evaluation**

Evaluate v5 on six validation cases and then all 18 formal cases for every
runnable candidate, regardless of disagreement or dense deltas. Compute
v5-minus-v3 and v5-minus-v4 dense comparisons, three decision-class summaries,
and both action-disagreement rates. Record KL and epistemic IG as missing with
the complete-action-distribution reason.

- [ ] **Step 6: Implement all finalization paths**

Provider failure, prompt omission, protected-file changes, test failure,
incomplete learning, incomplete validation, and incomplete formal evaluation
each save available logs and exactly one finalized summary. A successful
summary includes v0-v5 scores, missing AUC, rollback fields, benchmark and
learning results, feedback receipt, behavior diagnostics, phase budgets, and
the audited cumulative budget.

- [ ] **Step 7: Add negative-path tests**

```python
def test_v5_formal_evaluation_is_not_blocked_by_bad_diagnostics(tmp_path):
    pipeline, evaluator, _ = make_pipeline(
        tmp_path,
        validation_dense_change=-100.0,
        disagreement=0.0,
    )
    result = pipeline.run()
    assert result.status == "complete"
    assert ("v5", "evaluation", 18) in evaluator.calls


def test_v5_prompt_omission_finalizes_zero_act_run(tmp_path):
    pipeline, evaluator, provider = make_pipeline(
        tmp_path,
        prompt_max_bytes=100,
    )
    result = pipeline.run()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert result.status == "prompt_incomplete"
    assert provider.calls == 0
    assert summary["round_act_count"] == 0
    assert summary["budget"]["learning_episodes"] == 12


def test_v5_prior_zero_act_attempt_is_added_to_cumulative_budget(tmp_path):
    failed = make_finalized_v5_prompt_attempt(
        tmp_path,
        learning_episodes=12,
        learning_coding_agent_acts=0,
    )
    pipeline, _, _ = make_pipeline(tmp_path, prior_attempt_run=failed)
    result = pipeline.run()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["prior_attempt_run_id"] == failed.name
    assert summary["cumulative_learning_budget"]["learning_episodes"] == 82
    assert (
        summary["cumulative_learning_budget"][
            "learning_coding_agent_acts"
        ]
        == 6
    )
```

- [ ] **Step 8: Run v5 and v4 pipeline tests**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_pipeline_v5.py tests/generals/test_pipeline_v4.py`

Expected: all tests pass and v4 behavior is unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/agentbench_frame/generals/pipeline_v5.py \
  tests/generals/test_pipeline_v5.py
git commit -m "feat(generals): orchestrate rollback-guided v5"
```

### Task 5: Expose v5 through CLI, event quality, and the Dashboard

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Create: `tests/generals/test_cli_v5.py`
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `tests/test_local_report_research.py`

**Interfaces:**
- Consumes: `GeneralsHLRound5Pipeline.from_paths`.
- Produces: `agentbench generals iterate-v5` and report fields for rollback,
  v5 scores, paired diagnostics, and cumulative budgets.

- [ ] **Step 1: Write the failing CLI surface test**

```python
def test_generals_cli_registers_v5_rollback_inputs():
    args = parser().parse_args([
        "generals", "iterate-v5",
        "--agentbench-root", "/assets",
        "--manifest", "/assets/pilot-v1.toml",
        "--learning-manifest", "/assets/v5-learning.toml",
        "--replay-skill", "skills/replay-analysis-v1/SKILL.md",
        "--parent-run", "/runs/v4",
        "--expected-parent-hash", "a" * 64,
        "--expected-rollback-hash", "b" * 64,
        "--campaign-budget-receipt", "/derived/v4-budget.json",
        "--data-dir", "/data",
    ])
    assert args.generals_command == "iterate-v5"
    assert str(args.expected_rollback_hash) == "b" * 64
    assert str(args.campaign_budget_receipt) == "/derived/v4-budget.json"
```

- [ ] **Step 2: Run CLI test and verify RED**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_cli_v5.py`

Expected: parser rejection because `iterate-v5` is not registered.

- [ ] **Step 3: Add the CLI command and exact routing**

Register the v5-specific flags plus provider timeout and optional
`--prior-attempt-run`. Route them only to `GeneralsHLRound5Pipeline`, print
status/run/score/act/runnable fields as JSON, and return nonzero for incomplete
or invalid runs.

- [ ] **Step 4: Write failing report assertions**

```python
def test_v5_dashboard_renders_rollback_and_three_way_result(tmp_path):
    build_v5_report_fixture(tmp_path)
    html = render_report(tmp_path)
    assert "Rollback source" in html
    assert "v3" in html
    assert "Evo score" in html
    assert "44.4%" in html
    assert "v5 versus v3" in html
    assert "v5 versus v4" in html
    assert "70" in html
    assert "AUC / act<br>—" in html
```

- [ ] **Step 5: Run report test and verify RED**

Run:
`.venv/bin/python -m pytest -q tests/test_local_report_research.py -k v5`

Expected: missing rollback and v5 comparison labels.

- [ ] **Step 6: Extend report derivation and template**

Recognize `evo_score_5`/`gain_5`, preserve the seven-point score curve, expose
rollback source/hash metadata, render v3/v4/v5 decision classes and both paired
diagnostic groups, and keep all missing AUC/IG/KL values visibly missing.
Continue applying the hashed campaign receipt without mutating summaries.

- [ ] **Step 7: Run CLI/report/full tests**

Run:
`.venv/bin/python -m pytest -q tests/generals/test_cli_v5.py tests/test_local_report_research.py`

Expected: all focused tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: the full Framework suite passes.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/generals/cli.py \
  src/agentbench_frame/report/builder.py \
  src/agentbench_frame/report/templates/index.html \
  tests/generals/test_cli_v5.py tests/test_local_report_research.py
git commit -m "feat(report): expose Generals v5 rollback results"
```

### Task 6: Run the real v5 act and audit every artifact

**Files:**
- Runtime:
  `agentbench_data/runs/28_generals/generals-hl/<v5-run-id>/`
- Runtime: `agentbench_data/report-v5/`
- Create after evaluation:
  `docs/experiments/2026-07-29-generals-v5-rollback-guided-result.md`
- Modify: `README.md`
- Modify in AgentBench assets:
  `backend_sources/corpus/28_generals/README.md`

**Interfaces:**
- Consumes: committed v5 pipeline and assets, authenticated local Codex CLI,
  exact v4/v3 hashes, and v4 campaign-budget receipt.
- Produces: one immutable real v5 result and its report/documentation.

- [ ] **Step 1: Verify both worktrees before live execution**

Run in Framework:

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short
```

Run in AgentBench assets:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m pytest -q \
  backend_sources/corpus/28_generals/baselines/hl_v0/tests \
  backend_sources/corpus/28_generals/calibration/weak_v1/tests
git diff --check
git status --short
```

Expected: all tests pass and both worktrees are clean.

- [ ] **Step 2: Run the exact real v5 command**

```bash
.venv/bin/python -m agentbench_frame.cli generals iterate-v5 \
  --agentbench-root \
    /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest \
    /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --learning-manifest \
    /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v5-rollback-learning-v1.toml \
  --replay-skill \
    backend_sources/corpus/28_generals/skills/replay-analysis-v1/SKILL.md \
  --parent-run \
    /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260728_1519_7b242596 \
  --expected-parent-hash \
    5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f \
  --expected-rollback-hash \
    a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815 \
  --campaign-budget-receipt \
    /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/derived/28_generals/generals-hl/20260728_v4_campaign-budget.json \
  --data-dir \
    /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

Expected: one JSON result identifying a finalized run. If it stops before an
act, preserve that run, diagnose from its logs, fix through a new red-green
test, and retry with `--prior-attempt-run`.

- [ ] **Step 3: Verify the real run**

Assert from primary artifacts:

```python
assert summary["parent_version"] == "v4"
assert summary["starting_version"] == "v3"
assert summary["rollback_source_version"] == "v3"
assert summary["budget"]["learning_episodes"] == 12
assert summary["feedback_read"]["episodes"] == 12
assert summary["round_act_count"] == 1
assert summary["runnable"] is True
assert len(summary["benchmark_results"]) == 18
assert summary["event_quality"]["warnings"] == []
assert summary["AUC_coding_agent_act"] is None
```

Recompute v3/v4/v5 manifest hashes, summary/receipt source hashes, tier/seat
scores, decision classes, dense deltas, budget totals, and event-quality
counts. Run the saved v5 strategy tests from `versions/v5/source`.

- [ ] **Step 4: Build and inspect the report**

```bash
.venv/bin/agentbench data check --data-dir agentbench_data
.venv/bin/agentbench report \
  --data-dir agentbench_data \
  --output-dir agentbench_data/report-v5
```

Verify the latest report shows the exact v5 score, seven-point curve with the
historical gap, rollback source, 12 feedback episodes, phase budgets,
three-way diagnostics, and missing AUC/IG/KL.

- [ ] **Step 5: Write the result document from measured values**

The result document records scope, hashes, prompt receipt, provider usage,
changed files, tests, v3/v4/v5 learning comparison, v5 formal tier/seat score,
budgets, event quality, any preserved failed attempts, causal diagnosis, and
the next evidence-backed hypothesis. It must state a regression plainly if v5
does not beat v3.

- [ ] **Step 6: Run final verification**

Framework:

```bash
.venv/bin/python -m pytest -q
.venv/bin/agentbench data check --data-dir agentbench_data
git diff --check
```

AgentBench assets:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m pytest -q \
  backend_sources/corpus/28_generals/baselines/hl_v0/tests \
  backend_sources/corpus/28_generals/calibration/weak_v1/tests
git diff --check
```

- [ ] **Step 7: Commit documentation**

Framework:

```bash
git add README.md \
  docs/experiments/2026-07-29-generals-v5-rollback-guided-result.md
git commit -m "docs(generals): record real v5 result"
```

AgentBench assets:

```bash
git add backend_sources/corpus/28_generals/README.md
git commit -m "docs(generals): document v5 rollback assets"
```

Both worktrees must end clean. Runtime `agentbench_data` remains an immutable
local experiment artifact even when gitignored.
