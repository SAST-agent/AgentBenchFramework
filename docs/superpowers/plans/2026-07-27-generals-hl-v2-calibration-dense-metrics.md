# Generals HL Round-2 Calibration and Dense Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate weak-opponent calibration suite, round-aligned dense Generals diagnostics, compact replay feedback, immutable `v1 → v2` lineage, a second real Codex act, and Framework/CI evidence without changing the frozen formal benchmark.

**Architecture:** `AgentBench` owns the deterministic calibration player and versioned calibration manifests. `AgentBenchFramework` adds generic calibration budget accounting, suite construction, dense trajectory measurement, compact prompts, parent-run import, and a focused `GeneralsHLRound2Pipeline`; the existing first-round pipeline remains compatible. Formal and calibration results flow through separate event and report fields.

**Tech Stack:** Python 3.11+, standard library (`dataclasses`, `hashlib`, `json`, `pathlib`, `shutil`, `subprocess`, `tomllib`), pytest, uv, official Generals Python SDK/engine, Jinja2 report templates, Codex CLI `exec --json`.

## Global Constraints

- The formal benchmark ID remains exactly `generals-hl-pilot-v1`.
- The calibration benchmark ID is exactly `generals-hl-calibration-v1`.
- The original formal opponents, evaluation seeds `280101`, `280202`, `280303`, both seats, engine hash, and 18-case score are unchanged.
- Calibration development seeds are exactly `282101`, `282202`, `282303`, `282404`, `282505`; held-out seeds are exactly `282601`, `282702`, `282803`, `282904`, `283005`; every seed uses both seats.
- New human-opponent learning seeds are exactly `283101`, `283202`, `283303`; only low and medium opponents are learning opponents.
- The original v0 and first Codex v1 are immutable. Round 2 imports parent run `20260726_1631_f56f789f`, verifies v1 content hash `17356a28682378c4877f7da7aeb8ac90b8891e2685012dec8cdec990cb282e79`, invokes Codex once, and snapshots v2.
- Calibration score, dense diagnostics, action disagreement, occupancy shift, and formal benchmark score remain separate.
- Dense diagnostics are never named reward, information gain, policy KL, or benchmark score.
- Strict policy KL remains missing with status `complete_macro_action_distribution_unavailable`.
- Prompt UTF-8 size is at most `262144` bytes and truncation occurs only at whole evidence-record boundaries.
- Candidate-selection games use calibration-construction budget; feedback games use learning budget; held-out calibration and formal cases use evaluation budget; total includes all three.
- Missing data remains missing. Incomplete fixed-case evaluation has no aggregate score, gain, or AUC point.
- Events are finalized-only and append-only. Existing run files are never edited.
- The real held-out calibration set is opened once. A miss is retained as a failed calibration version and stops the Codex act.
- Generated run data, prepared opponents, caches, and reports are not committed.
- Preserve all unrelated user changes in both repositories.

## Execution Prerequisites

The work already lives in isolated worktrees:

```text
Framework: /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
Assets:    /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
```

Before implementation, invoke `superpowers:using-git-worktrees`, verify both
paths are linked worktrees, and keep using their current branches:

```text
zhaoyicheng/generals-hl-implementation
zhaoyicheng/generals-assets
```

Use the existing Framework `.venv` when present. Otherwise create it without
writing a project lock:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e . 'pytest>=8,<9' 'jinja2>=3,<4'
```

## File Map

### AgentBench assets

- `backend_sources/corpus/28_generals/benchmark/calibration-v1.toml` — candidate modes, frozen dev/test seeds, target interval, and calibration source path.
- `backend_sources/corpus/28_generals/benchmark/calibration-v1-selection.toml` — empirically selected mode, source hash, dev scores, and selection rule.
- `backend_sources/corpus/28_generals/calibration/weak_v1/main.py` — official SDK-compatible calibration entrypoint.
- `backend_sources/corpus/28_generals/calibration/weak_v1/state_view.py` — stable SDK state normalization.
- `backend_sources/corpus/28_generals/calibration/weak_v1/strategy.py` — deterministic `passive`, `local-expander`, and `resource-greedy` modes.
- `backend_sources/corpus/28_generals/calibration/weak_v1/README.md` — policy limits and scientific role.
- `backend_sources/corpus/28_generals/calibration/weak_v1/tests/test_strategy.py` — deterministic strategy tests.

### AgentBenchFramework

- `src/agentbench_frame/tracking/budget.py` — add calibration-construction as a third budget phase while preserving learning/evaluation/total fields.
- `src/agentbench_frame/tracking/run.py` — restore the third phase on resume.
- `src/agentbench_frame/tracking/quality.py` — recognize the additive round-2 event types without weakening unknown-event diagnostics.
- `src/agentbench_frame/generals/models.py` — calibration configuration and round-2 result records.
- `src/agentbench_frame/generals/assets.py` — load/validate/resolve calibration manifests and selection.
- `src/agentbench_frame/generals/process.py` — construct the calibration player process with an allowlisted mode.
- `src/agentbench_frame/generals/calibration.py` — build calibration cases, aggregate candidates, choose the dev winner, and run frozen held-out cases.
- `src/agentbench_frame/generals/dense.py` — dense samples, summaries, graph-distance pressure, AUC, and JSONL persistence.
- `src/agentbench_frame/generals/evaluator.py` — emit dense artifacts/events for every evaluated match without changing formal aggregation.
- `src/agentbench_frame/generals/replay.py` — build compact per-episode evidence records.
- `src/agentbench_frame/generals/prompt.py` — bounded whole-record prompt assembly and prompt manifest.
- `src/agentbench_frame/generals/lineage.py` — validate and import an immutable parent v1 snapshot.
- `src/agentbench_frame/generals/pipeline_v2.py` — orchestrate frozen selection, held-out calibration, v1 learning, one Codex act, v2 evaluation, and final summary.
- `src/agentbench_frame/generals/cli.py` — add `calibrate-dev` and `iterate-v2`.
- `src/agentbench_frame/report/builder.py` — derive separate calibration and dense report structures and respect explicit global act indices.
- `src/agentbench_frame/report/templates/index.html` — render formal, calibration, and dense panels separately.
- `tests/generals/fixtures/calibration-v1.toml` — relocatable calibration config.
- `tests/generals/fixtures/calibration-v1-selection.toml` — deterministic fake selection.
- `tests/generals/test_calibration.py` — config, split, selection, score isolation, and held-out behavior.
- `tests/generals/test_dense.py` — state samples, pressure, survival, missing values, and AUC.
- `tests/generals/test_prompt_v2.py` — compact selection, byte bound, and leakage.
- `tests/generals/test_lineage.py` — parent validation and exact v1 import.
- `tests/generals/test_pipeline_v2.py` — fake-provider round-2 evidence chain.
- `tests/generals/test_cli.py` — new command arguments and exit semantics.
- `tests/test_tracking_contracts.py` — third-phase budget compatibility.
- `tests/test_local_report_research.py` — formal/calibration/dense rendering separation.
- `tests/generals/test_live.py` — gated calibration and round-2 live checks.

---

### Task 1: Add Calibration-Construction Budget Accounting

**Files:**
- Modify: `src/agentbench_frame/tracking/budget.py`
- Modify: `src/agentbench_frame/tracking/run.py`
- Modify: `tests/test_tracking_contracts.py`
- Modify: `tests/generals/test_pipeline.py`

**Interfaces:**
- Consumes: the existing public `BudgetLedger.add` keyword interface
- Produces: phase `"calibration"`, per-phase `game_agent_decision_steps` and `primitive_commands`, and snapshot keys `calibration_*`; existing `learning_*`, `evaluation_*`, and `total_*` keys retain their meanings

- [ ] **Step 1: Write failing three-phase ledger tests**

Add:

```python
def test_budget_ledger_separates_calibration_without_polluting_learning():
    from agentbench_frame.tracking.budget import BudgetLedger

    ledger = BudgetLedger()
    ledger.add(
        "calibration", episodes=3, env_steps=30,
        game_agent_decision_steps=15, primitive_commands=45, time_s=2.0,
    )
    ledger.add(
        "learning", episodes=2, env_steps=20,
        game_agent_decision_steps=10, primitive_commands=30, time_s=1.0,
    )
    ledger.add(
        "evaluation", episodes=1, env_steps=10,
        game_agent_decision_steps=5, primitive_commands=15, time_s=0.5,
    )
    snapshot = ledger.snapshot()

    assert snapshot["calibration_episodes"] == 3
    assert snapshot["learning_episodes"] == 2
    assert snapshot["evaluation_episodes"] == 1
    assert snapshot["total_episodes"] == 6
    assert snapshot["total_env_steps"] == 60
    assert snapshot["total_game_agent_decision_steps"] == 30
    assert snapshot["total_primitive_commands"] == 90
    assert snapshot["total_time_s"] == 3.5
```

Extend the run-resume test so a finalized run with calibration budget resumes
with the same `calibration_episodes`, `calibration_env_steps`, and
`calibration_time_s`.

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/test_tracking_contracts.py tests/generals/test_pipeline.py -q
```

Expected: FAIL because `"calibration"` is rejected and its fields are absent.

- [ ] **Step 3: Generalize the phase accumulator**

Use the exact phase order:

```python
PHASES = ("calibration", "learning", "evaluation")
```

Build `_phases` from `PHASES`, emit every phase's act/episode/environment-step/
game-agent-decision-step/primitive-command/token/time fields, and calculate
totals across every phase that has observed the corresponding optional value.
Unknown token usage in any observed phase keeps the total unknown; a phase
with no token observation does not fabricate zero.

Change `Run.resume()` to restore all three phases using the same optional-token
and optional-time logic already used for learning/evaluation, plus the integer
game-agent-decision-step and primitive-command fields.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tracking_contracts.py tests/generals/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the Framework change**

```bash
git add src/agentbench_frame/tracking/budget.py src/agentbench_frame/tracking/run.py tests/test_tracking_contracts.py tests/generals/test_pipeline.py
git commit -m "feat(tracking): separate calibration construction budget"
```

### Task 2: Add the Deterministic Calibration Player Assets

**Files:**
- Create [Assets]: `backend_sources/corpus/28_generals/benchmark/calibration-v1.toml`
- Create [Assets]: `backend_sources/corpus/28_generals/calibration/weak_v1/main.py`
- Create [Assets]: `backend_sources/corpus/28_generals/calibration/weak_v1/state_view.py`
- Create [Assets]: `backend_sources/corpus/28_generals/calibration/weak_v1/strategy.py`
- Create [Assets]: `backend_sources/corpus/28_generals/calibration/weak_v1/README.md`
- Create [Assets]: `backend_sources/corpus/28_generals/calibration/weak_v1/tests/test_strategy.py`

**Interfaces:**
- Consumes: official SDK initial JSON and peer command stream; environment variable `AGENTBENCH_CALIBRATION_MODE`
- Produces: `choose_actions(round_number: int, my_seat: int, view: dict, mode: str) -> list[list[int]]` and an SDK-compatible framed stdout response

- [ ] **Step 1: Write failing strategy tests**

Create tests covering:

```python
import pytest

from strategy import choose_actions


def minimal_view():
    return {
        "round": 1,
        "coins": [40, 40],
        "cells": {
            "1,1": {
                "type": 0, "player": 0, "army": 5, "general_id": 0,
            },
        },
        "generals": {
            "0": {
                "id": 0, "type": "main", "player": 0,
                "position": [1, 1],
            },
        },
    }


def view_with_resource_at(position):
    view = minimal_view()
    view["cells"]["1,2"] = {
        "type": 0, "player": -1, "army": 0, "general_id": None,
    }
    view["cells"][f"{position[0]},{position[1]}"] = {
        "type": 0, "player": -1, "army": 2, "general_id": 1,
    }
    view["generals"]["1"] = {
        "id": 1, "type": "resource", "player": -1,
        "position": list(position),
    }
    return view


def test_passive_always_ends_turn():
    assert choose_actions(1, 0, minimal_view(), "passive") == [[8]]


def test_local_expander_captures_only_an_adjacent_affordable_neutral():
    view = view_with_resource_at((1, 2))
    assert choose_actions(1, 0, view, "local-expander") == [
        [1, 1, 1, 4, 4],
        [8],
    ]


def test_resource_greedy_takes_deterministic_first_bfs_step():
    view = view_with_resource_at((1, 3))
    assert choose_actions(1, 0, view, "resource-greedy") == [
        [1, 1, 1, 4, 4],
        [8],
    ]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported calibration mode"):
        choose_actions(1, 0, minimal_view(), "rush")
```

- [ ] **Step 2: Run the tests and verify the red state**

Run from the Assets worktree:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest backend_sources/corpus/28_generals/calibration/weak_v1/tests -q
```

Expected: collection fails because the calibration package is absent.

- [ ] **Step 3: Implement the three bounded policies**

Use the baseline's stable state view and SDK framing. Implement:

```python
SUPPORTED_MODES = ("passive", "local-expander", "resource-greedy")


def choose_actions(round_number, my_seat, view, mode):
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported calibration mode: {mode}")
    if mode == "passive":
        return [[8]]
    adjacent = capture_adjacent_neutral(view, my_seat)
    if adjacent is not None:
        return [adjacent, [8]]
    if mode == "resource-greedy":
        routed = route_main_to_nearest_resource(view, my_seat)
        if routed is not None:
            return [routed, [8]]
    return [[8]]
```

No mode may upgrade technology, search for the enemy main, use randomness, or
read time. Neighbor and BFS tie-breaking use the fixed order
`up, down, left, right`.

Freeze the TOML values:

```toml
benchmark_id = "generals-hl-calibration-v1"
source = "backend_sources/corpus/28_generals/calibration/weak_v1"
candidate_modes = ["passive", "local-expander", "resource-greedy"]
development_seeds = [282101, 282202, 282303, 282404, 282505]
heldout_seeds = [282601, 282702, 282803, 282904, 283005]
target_min = 0.2
target_max = 0.6
target_midpoint = 0.4
```

- [ ] **Step 4: Run the asset tests**

Run:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest backend_sources/corpus/28_generals/calibration/weak_v1/tests -q
```

Expected: PASS.

- [ ] **Step 5: Commit the Assets change**

```bash
git add backend_sources/corpus/28_generals/benchmark/calibration-v1.toml backend_sources/corpus/28_generals/calibration/weak_v1
git commit -m "feat(generals): add weak calibration policy ladder"
```

### Task 3: Load, Validate, and Score the Calibration Suite

**Files:**
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `src/agentbench_frame/generals/process.py`
- Create: `src/agentbench_frame/generals/calibration.py`
- Create: `tests/generals/fixtures/calibration-v1.toml`
- Create: `tests/generals/fixtures/calibration-v1-selection.toml`
- Create: `tests/generals/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationConfig`, `CalibrationSelection`, `CalibrationEvaluation`
- Produces: `load_calibration_config(path: Path) -> CalibrationConfig`
- Produces: `load_calibration_selection(path: Path, config: CalibrationConfig) -> CalibrationSelection`
- Produces: `build_calibration_cases(config, split: str, mode: str) -> tuple[BenchmarkCase, ...]`
- Produces: `select_calibration_candidate(config, results: Mapping[str, CalibrationEvaluation]) -> str`
- Produces: `build_calibration_process(source_root, engine_root, python_executable, sdk_root, mode) -> AgentProcessSpec`

- [ ] **Step 1: Write failing config and split tests**

Add exact assertions:

```python
from agentbench_frame.generals.models import CalibrationEvaluation


def development_result(mode, score, status="complete"):
    return CalibrationEvaluation(
        mode=mode,
        split="development",
        status=status,
        score=score,
        wins=0,
        losses=0,
        draws=0,
        per_seat={0: score, 1: score},
        results=(),
        matches=(),
        in_target_range=None,
    )


def test_calibration_manifest_has_disjoint_frozen_splits():
    config = load_calibration_config(FIXTURE)
    assert config.benchmark_id == "generals-hl-calibration-v1"
    assert config.candidate_modes == (
        "passive", "local-expander", "resource-greedy"
    )
    assert len(build_calibration_cases(config, "development", "passive")) == 10
    assert len(build_calibration_cases(config, "heldout", "passive")) == 10
    assert set(config.development_seeds).isdisjoint(config.heldout_seeds)


def test_calibration_cases_never_enter_formal_spec():
    formal = build_evaluation_spec(load_pilot_config(PILOT_FIXTURE))
    calibration = build_calibration_cases(
        load_calibration_config(FIXTURE), "heldout", "passive"
    )
    assert formal.version == "generals-hl-pilot-v1"
    assert all(case.metadata["suite"] == "calibration" for case in calibration)
    assert {case.case_id for case in formal.cases}.isdisjoint(
        case.case_id for case in calibration
    )
```

Add validation failures for overlapping seeds, target range outside `[0, 1]`,
duplicate modes, unknown selected mode, and a mismatched source hash.

- [ ] **Step 2: Write failing deterministic-selection tests**

```python
def test_candidate_selection_uses_distance_to_point_four_then_manifest_order():
    results = {
        "passive": development_result("passive", 0.7),
        "local-expander": development_result("local-expander", 0.3),
        "resource-greedy": development_result("resource-greedy", 0.5),
    }
    assert select_calibration_candidate(CONFIG, results) == "local-expander"


def test_incomplete_candidate_cannot_be_selected():
    with pytest.raises(ValueError, match="all candidate evaluations must be complete"):
        select_calibration_candidate(
            CONFIG, {
                mode: development_result(mode, None, status="incomplete")
                for mode in CONFIG.candidate_modes
            }
        )
```

- [ ] **Step 3: Run tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_calibration.py -q
```

Expected: FAIL because calibration types and loaders are absent.

- [ ] **Step 4: Implement strict calibration contracts**

Use frozen dataclasses:

```python
@dataclass(frozen=True)
class CalibrationConfig:
    benchmark_id: str
    source: Path
    candidate_modes: tuple[str, ...]
    development_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    target_min: float
    target_max: float
    target_midpoint: float


@dataclass(frozen=True)
class CalibrationSelection:
    benchmark_id: str
    selected_mode: str
    source_hash: str
    development_scores: Mapping[str, float]
    selection_rule: str


@dataclass(frozen=True)
class CalibrationEvaluation:
    mode: str
    split: str
    status: str
    score: float | None
    wins: int
    losses: int
    draws: int
    per_seat: Mapping[int, float | None]
    results: tuple[GameResult, ...]
    matches: tuple[MatchResult, ...]
    in_target_range: bool | None
```

Case IDs include benchmark, split, mode, seed, and seat. Candidate selection
minimizes:

```python
(abs(result.score - config.target_midpoint),
 config.candidate_modes.index(mode))
```

Reject any incomplete candidate result rather than treating it as zero.
`build_calibration_process` passes only a validated manifest mode through the
process environment.

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_calibration.py tests/generals/test_assets.py tests/generals/test_process.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the Framework change**

```bash
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/assets.py src/agentbench_frame/generals/process.py src/agentbench_frame/generals/calibration.py tests/generals/fixtures/calibration-v1.toml tests/generals/fixtures/calibration-v1-selection.toml tests/generals/test_calibration.py
git commit -m "feat(generals): add isolated calibration suite"
```

### Task 4: Compute and Persist Dense Trajectory Diagnostics

**Files:**
- Create: `src/agentbench_frame/generals/dense.py`
- Modify: `src/agentbench_frame/generals/evaluator.py`
- Create: `tests/generals/test_dense.py`
- Modify: `tests/generals/test_evaluator.py`

**Interfaces:**
- Produces: `DenseStateSample`, `DenseMetricSummary`, `DenseEpisodeSummary`
- Produces: `dense_sample(state: Mapping[str, Any], evaluated_seat: int, sample_index: int, sample_kind: str) -> DenseStateSample`
- Produces: `build_dense_trace(match: MatchResult) -> tuple[DenseStateSample, ...]`
- Produces: `summarize_dense_trace(match, trace) -> DenseEpisodeSummary`
- Produces: `persist_dense_diagnostics(match, artifact_dir) -> tuple[trace, summary]`

- [ ] **Step 1: Write failing state-metric tests**

Use a small state with a mountain barrier and two main generals. Assert:

```python
sample = dense_sample(state, evaluated_seat=0, sample_index=0, sample_kind="initial")
assert sample.target_territory == 2
assert sample.opponent_territory == 1
assert sample.territory_margin == 1
assert sample.territory_share == pytest.approx(2 / 3)
assert sample.target_army == 9
assert sample.opponent_army == 4
assert sample.army_margin == 5
assert sample.target_coins == 30
assert sample.opponent_coins == 10
assert sample.coin_share == 0.75
```

Add symmetric seat-1 assertions and zero-denominator assertions where shares
are `None`.

- [ ] **Step 2: Write failing pressure and trace tests**

Cover:

- graph distance follows traversable cells and does not cross type-2 mountains;
- attacking mass uses `max(army - 1, 0)`;
- defense includes the main cell;
- a missing main yields missing pressure fields;
- initial, completed-round, and terminal sample kinds are ordered;
- a full round is counted only when player 1's `state_after["round"]` exceeds
  `state_before["round"]`;
- a terminal state after player 0 still appears;
- truncated external failures set `truncated=True`, while valid normal or
  illegal-action outcomes set `terminated=True`.

- [ ] **Step 3: Write failing episode-summary tests**

Construct a trace at official rounds `1`, `2`, and `3` whose territory shares
are `0.25`, `0.5`, and `0.75`:

```python
summary = summarize_dense_trace(match, trace)
assert summary.completed_rounds_survived == 2
assert summary.territory_share.terminal == 0.75
assert summary.territory_share.time_average == pytest.approx(0.5)
assert summary.territory_share.auc == pytest.approx(1.0)
```

Add a missing sample and assert every derived value requiring it is `None`;
the calculator must not interpolate.

- [ ] **Step 4: Run tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_dense.py tests/generals/test_evaluator.py -q
```

Expected: FAIL because `dense.py` is absent.

- [ ] **Step 5: Implement round-aligned dense measurement**

Use frozen, JSON-serializable dataclasses. BFS neighbors use
`((-1, 0), (1, 0), (0, -1), (0, 1))`. Derive territory, army, coins, pressure,
alive flags, outcome, termination, and per-metric terminal/min/max/mean/AUC.
Trapezoidal AUC uses official-round coordinates and returns `None` when any
required value is missing.

Write `dense-trace.jsonl` and `dense-summary.json` atomically through temporary
files in the same artifact directory followed by `Path.replace`.

- [ ] **Step 6: Integrate dense facts without changing formal scoring**

After each match, `GeneralsEvaluator.evaluate()` calls
`persist_dense_diagnostics`. On success it writes:

```python
run.write(
    "dense_trajectory",
    case_id=case.case_id,
    version=version,
    phase=phase,
    evaluated_seat=case.first_player,
    trace=[asdict(sample) for sample in trace],
    artifact_ref=str(artifact_dir / "dense-trace.jsonl"),
)
run.write("dense_episode_summary", **asdict(summary))
```

On dense calculation failure, retain the match result and write a
`dense_metric_error` event with a missing dense summary. Never replace missing
metrics with zero and never alter `GameResult.outcome`, validity, or aggregate
score. Each match budget observation additionally records:

```python
game_agent_decision_steps = sum(
    turn.player == match.evaluated_seat for turn in match.turns
)
primitive_commands = sum(len(turn.commands) for turn in match.turns)
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_dense.py tests/generals/test_evaluator.py tests/generals/test_match.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/generals/dense.py src/agentbench_frame/generals/evaluator.py tests/generals/test_dense.py tests/generals/test_evaluator.py
git commit -m "feat(generals): record dense trajectory diagnostics"
```

### Task 5: Build Compact, Leak-Resistant Round-2 Feedback

**Files:**
- Modify: `src/agentbench_frame/generals/replay.py`
- Modify: `src/agentbench_frame/generals/prompt.py`
- Create: `tests/generals/test_prompt_v2.py`
- Modify: `tests/generals/test_replay_prompt.py`

**Interfaces:**
- Produces: `CompactLearningEvidence`
- Produces: `PromptBuildResult`
- Produces: `build_compact_evidence(replay, dense_summary, dense_trace, max_decisions=4) -> CompactLearningEvidence`
- Produces: `build_round2_prompt(benchmark_id: str, strategy_doc: str, rules_text: str, replay_guide: str, first_diff: str, evidence: Sequence[CompactLearningEvidence], max_bytes: int = 262144) -> PromptBuildResult`

- [ ] **Step 1: Write failing compaction tests**

Assert that a loss retains:

- episode ID, seed, seat, tier, outcome, and termination;
- dense terminal/mean/AUC fields;
- the first target decision;
- the final three target decisions;
- no full board, complete cell dictionary, opponent source path, or formal
  evaluation result.

If the episode has four or fewer target decisions, retain every decision once.

- [ ] **Step 2: Write failing byte-bound tests**

```python
result = build_round2_prompt(
    benchmark_id="generals-hl-pilot-v1",
    strategy_doc="current strategy",
    rules_text="official rules",
    replay_guide="field guide",
    first_diff="v0 to v1 diff",
    evidence=large_records,
    max_bytes=1024,
)
assert len(result.prompt.encode("utf-8")) <= 1024
assert result.truncated is True
assert set(result.included_episode_ids).isdisjoint(result.omitted_episode_ids)
assert result.estimated_tokens == (result.prompt_bytes + 3) // 4
```

Parse the evidence JSON lines and prove no record is partially truncated.
Assert the prompt rejects a limit smaller than its mandatory header.

- [ ] **Step 3: Write leakage tests**

Build a prompt from allowed learning records and assert it does not contain:

```text
280101 280202 280303
282601 282702 282803 282904 283005
advanced-rank02-robinliu-v18
top_algorithms/
```

The first-round diff may be included, but evaluation artifacts may not.

- [ ] **Step 4: Run tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_prompt_v2.py tests/generals/test_replay_prompt.py -q
```

Expected: FAIL because compact round-2 APIs are absent.

- [ ] **Step 5: Implement deterministic whole-record selection**

Sort evidence by `(opponent_tier, seed, evaluated_seat, replay_id)`. Assemble
the mandatory instructions first, then append one JSON evidence record at a
time while the UTF-8 result stays within `max_bytes`. Return:

```python
@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    included_episode_ids: tuple[str, ...]
    omitted_episode_ids: tuple[str, ...]
    prompt_bytes: int
    estimated_tokens: int
    truncated: bool
    selection_policy: str = "sorted_whole_records_v1"
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_prompt_v2.py tests/generals/test_replay_prompt.py -q
```

Expected: PASS.

Commit:

```bash
git add src/agentbench_frame/generals/replay.py src/agentbench_frame/generals/prompt.py tests/generals/test_prompt_v2.py tests/generals/test_replay_prompt.py
git commit -m "feat(generals): compact round-2 learning feedback"
```

### Task 6: Import v1 and Orchestrate One v1-to-v2 Act

**Files:**
- Create: `src/agentbench_frame/generals/lineage.py`
- Create: `src/agentbench_frame/generals/pipeline_v2.py`
- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Create: `tests/generals/test_lineage.py`
- Create: `tests/generals/test_pipeline_v2.py`
- Modify: `tests/generals/test_cli.py`
- Modify: `tests/test_research_boundaries.py`

**Interfaces:**
- Produces: `ParentLineage`, `Round2PipelineResult`
- Produces: `load_parent_lineage(parent_run_dir, expected_hash) -> ParentLineage`
- Produces: `import_parent_v1(lineage, run_dir, snapshotter) -> WorkspaceManifest`
- Produces: `GeneralsHLRound2Pipeline.run() -> Round2PipelineResult`
- CLI: `agentbench generals calibrate-dev` with explicit asset, parent, data, and selection-output paths
- CLI: `agentbench generals iterate-v2` with explicit asset, parent, calibration, provider, and data paths

- [ ] **Step 1: Write failing lineage tests**

Build a synthetic completed parent run containing v0/v1 sources and manifests.
Assert:

```python
lineage = load_parent_lineage(parent, expected_hash=manifest.content_hash)
assert lineage.parent_run_id == "parent-1"
assert lineage.raw_score == 0.0
assert lineage.evo_score_1 == 0.0
assert lineage.parent_version == "v1"

imported = import_parent_v1(lineage, child_run, LocalWorkspaceSnapshotter())
assert imported.content_hash == manifest.content_hash
assert (child_run / "workspace" / "strategy.py").read_bytes() == (
    parent / "versions" / "v1" / "source" / "strategy.py"
).read_bytes()
```

Reject non-complete parent status, missing v1 files, manifest/file hash
mismatch, and an unexpected content hash.

- [ ] **Step 2: Write the fake round-2 pipeline test**

Use fake calibration and formal evaluators plus the existing `FakeProvider`.
Assert:

```python
assert result.raw_score == 0.0
assert result.evo_score_1 == 0.0
assert result.evo_score_2 == 0.25
assert result.gain_2 == 0.25
assert result.global_act_count == 2
assert result.round_act_count == 1
assert result.calibration_score == 0.4
assert result.status == "complete"
```

Verify artifacts:

```text
benchmark/formal-spec.json
benchmark/calibration-spec.json
benchmark/calibration-opponent-manifest.json
provider/codex-act-2.prompt.md
provider/codex-act-2.prompt.json
provider/codex-act-2.raw.jsonl
versions/v1/manifest.json
versions/v2/manifest.json
versions/v1-to-v2.patch
quality.json
summary.json
```

Read events and assert exactly one new `coding_agent_act`, one
`lineage_import`, separate `calibration_result`, dense events, and
`version_before="v1"`, `version_after="v2"`.

Group learning probes by replay and assert one `behavior_change_episode` event
per included episode plus one aggregate `behavior_change` event containing
both episode-balanced and decision-balanced values. Both retain
`policy_kl=None` and the unavailable status.

- [ ] **Step 3: Write failure-path tests**

Assert:

- held-out calibration score `0.1` yields `status="calibration_failed"`, saves
  the result, and does not invoke the provider;
- incomplete calibration yields missing score and no provider act;
- provider failure still snapshots readable v2;
- protected-file or baseline-test failure marks v2 invalid and skips formal v2
  evaluation;
- one invalid formal case produces missing `evo_score_2`, `gain_2`, and AUC.

- [ ] **Step 4: Run tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_lineage.py tests/generals/test_pipeline_v2.py tests/generals/test_cli.py -q
```

Expected: FAIL because lineage, pipeline, and CLI commands are absent.

- [ ] **Step 5: Implement parent import and the round-2 sequence**

The pipeline sequence is exact:

```text
start new run
→ write formal/calibration specs
→ verify and import parent v1
→ load frozen calibration selection
→ run original v0 on held-out calibration once
→ stop if score is missing or outside [0.2, 0.6]
→ re-evaluate imported v1 on the 18 formal cases for dense paired data
→ run v1 on new low/medium human learning cases
→ run v1 on selected calibration-development feedback cases
→ build compact prompt and prompt manifest
→ invoke Codex once with version_before=v1
→ snapshot v2 and write v1-to-v2 patch
→ run strategy tests and protected-file check
→ measure v1/v2 action disagreement on learning probes
→ evaluate v2 on the same 18 formal cases
→ record separate occupancy shift
→ finish quality and summary
```

For behavior measurement, call `measure_action_disagreement` independently
for each learning replay. The aggregate event stores:

```text
episode_balanced_action_disagreement
decision_balanced_action_disagreement
episode_count
decision_count
policy_kl = missing
policy_kl_status = complete_macro_action_distribution_unavailable
```

Do not concatenate traces and call the result episode-balanced.

The summary retains:

```python
{
    "benchmark_id": "generals-hl-pilot-v1",
    "calibration_benchmark_id": "generals-hl-calibration-v1",
    "parent_run_id": lineage.parent_run_id,
    "raw_score": lineage.raw_score,
    "evo_score_1": lineage.evo_score_1,
    "evo_score": evo_score_2,
    "evo_score_2": evo_score_2,
    "gain": gain_2,
    "gain_2": gain_2,
    "act_count": 2,
    "round_act_count": 1,
    "calibration_score": calibration.score,
    "calibration_in_target_range": calibration.in_target_range,
}
```

Calculate `AUC_coding_agent_act` only when v0, v1, and v2 scores all exist,
using trapezoids over global x coordinates `0`, `1`, `2`.

Load the parent's learning-only budget coordinates. Imported v0 and v1 score
events use the parent's recorded coordinates; v2 uses parent learning budget
plus only round-2 learning budget. Calibration-construction and evaluation
costs are excluded. Emit `AUC_episode`, `AUC_env_step`, `AUC_token`, and
`AUC_time` only when every required score and coordinate exists; unknown token
usage keeps token AUC missing.

- [ ] **Step 6: Add explicit CLI contracts**

`calibrate-dev` requires AgentBench root, formal manifest, calibration
manifest, parent run, data directory, and `--selection-output`. It evaluates
all three modes on dev cases, prints canonical JSON with scores, selected
mode, source hash, and selection rule, and atomically writes the same facts as
TOML to the selection output.

`iterate-v2` additionally requires a frozen selection file and Codex provider
arguments. It returns nonzero for calibration failure, provider failure,
invalid v2, or incomplete formal evaluation.

Extend `KNOWN_EVENT_TYPES` with exactly:

```text
calibration_spec
calibration_result
dense_trajectory
dense_episode_summary
dense_metric_error
lineage_import
behavior_change_episode
```

Add a quality test showing these are recognized while an unrelated future
event still increments `unknown_event_types`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_lineage.py tests/generals/test_pipeline_v2.py tests/generals/test_cli.py tests/generals/test_pipeline.py tests/test_research_boundaries.py -q
```

Expected: PASS, including first-round regression.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/generals/lineage.py src/agentbench_frame/generals/pipeline_v2.py src/agentbench_frame/generals/cli.py src/agentbench_frame/tracking/quality.py tests/generals/test_lineage.py tests/generals/test_pipeline_v2.py tests/generals/test_cli.py tests/test_research_boundaries.py
git commit -m "feat(generals): orchestrate second Codex HL act"
```

### Task 7: Separate Formal, Calibration, and Dense CI Panels

**Files:**
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `tests/test_local_report_research.py`

**Interfaces:**
- Consumes: `calibration_result`, `dense_episode_summary`, `dense_trajectory`, `behavior_change`, formal evaluation events, and summary fields
- Produces: `research["calibration"]`, `research["dense_history"]`, and global-act-aware `score_history`

- [ ] **Step 1: Write failing derivation tests**

Construct events containing one imported v1 score at global act 1, one v2 score
at global act 2, one calibration result, and two dense summaries. Assert:

```python
assert research["score_history"] == [
    {"x": 0, "score": 0.0, "act_id": None},
    {"x": 1, "score": 0.0, "act_id": "parent-act"},
    {"x": 2, "score": 0.25, "act_id": "act-2"},
]
assert research["calibration"]["score"] == 0.4
assert research["calibration"]["in_target_range"] is True
assert len(research["dense_history"]) == 2
```

Assert calibration events never enter `score_history` or `auc_points`.

- [ ] **Step 2: Write failing HTML tests**

Build the local report and assert the generated HTML contains:

```text
Formal performance
Calibration suite
Dense trajectory diagnostics
Survival rounds
Territory
Army
Coins
Main-general pressure
action disagreement
policy KL unavailable
```

Also assert it labels calibration score separately and preserves `missing`
for an incomplete dense field.

- [ ] **Step 3: Run tests and verify the red state**

Run:

```bash
.venv/bin/python -m pytest tests/test_local_report_research.py -q
```

Expected: FAIL because calibration and dense report structures are absent.

- [ ] **Step 4: Implement additive report derivation**

Honor an event's explicit `coding_agent_act` before phase defaults. Derive
calibration only from `calibration_result`; derive dense episode rows only
from `dense_episode_summary`. Keep policy KL, occupancy, and action
disagreement sections separate. Do not synthesize zeros or interpolate.

- [ ] **Step 5: Render three distinct panels**

Use different headings and table columns. The formal panel shows
raw/evo/gain/AUC; calibration shows benchmark ID, version, score, target band,
seat split, and status; dense shows one row per case/version with outcome,
survival, terminal and AUC values. The budget panel shows calibration,
learning, evaluation, and total episode/environment-step/game-agent-decision/
primitive-command counts while retaining unknown token/time values. Keep the
existing quality diagnostics.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_local_report_research.py -q
```

Expected: PASS.

Commit:

```bash
git add src/agentbench_frame/report/builder.py src/agentbench_frame/report/templates/index.html tests/test_local_report_research.py
git commit -m "feat(report): separate calibration and dense diagnostics"
```

### Task 8: Select and Freeze the Calibration Mode on Development Seeds

**Files:**
- Create [Assets]: `backend_sources/corpus/28_generals/benchmark/calibration-v1-selection.toml`
- Modify: `tests/generals/test_live.py`
- Generated, not committed: calibration-development run under `agentbench_data/`

**Interfaces:**
- Consumes: all three predeclared modes and only the five development seeds
- Produces: one frozen selection whose mode minimizes distance to `0.4`, with manifest-order tie-breaking

- [ ] **Step 1: Add a gated live development test**

The test reads `AGENTBENCH_ASSET_ROOT` and
`AGENTBENCH_GENERALS_PARENT_RUN`, executes every mode on all ten dev cases,
asserts complete results, and verifies the selected mode matches the pure
selection function. It never reads held-out seeds.

- [ ] **Step 2: Run non-live regressions**

Run:

```bash
.venv/bin/python -m pytest tests/generals -q
```

Expected: PASS with live tests skipped unless explicitly enabled.

- [ ] **Step 3: Execute calibration development**

Run:

```bash
.venv/bin/agentbench generals calibrate-dev \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --calibration-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/calibration-v1.toml \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260726_1631_f56f789f \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --selection-output /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/calibration-v1-selection.toml
```

Expected: exit `0`; canonical JSON reports 30 valid games, per-mode scores,
the selected mode, source hash, and rule
`closest_to_0.4_then_manifest_order`. No held-out case ID appears.

- [ ] **Step 4: Verify the atomically frozen empirical selection**

Load `calibration-v1-selection.toml` through
`load_calibration_selection`. Assert its `selected_mode`, `source_hash`,
`development_scores`, and `selection_rule` exactly equal the canonical JSON
printed by Step 3. Inspect the development run and assert no held-out case ID
or held-out seed appears.

- [ ] **Step 5: Validate and commit both repositories**

Run Framework config tests and asset strategy tests, then commit:

```bash
git add tests/generals/test_live.py
git commit -m "test(generals): gate live calibration development"
```

In Assets:

```bash
git add backend_sources/corpus/28_generals/benchmark/calibration-v1-selection.toml
git commit -m "data(generals): freeze calibration v1 selection"
```

### Task 9: Run the One-Shot Held-Out Calibration and Second Real Codex Act

**Files:**
- Generated, not committed: new round-2 run under `agentbench_data/runs/28_generals/generals-hl/`
- Generated, not committed: `_site/`

**Interfaces:**
- Consumes: frozen formal spec, frozen calibration selection, parent v1 snapshot, authenticated local Codex CLI
- Produces: one complete `v1 → v2` run and report, or one transparently failed run without retry

- [ ] **Step 1: Run a preflight without opening held-out cases**

Run:

```bash
.venv/bin/agentbench generals prepare \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml
```

Then validate the selection file through the config loader test. Expected:
assets, parent hash, Codex executable, and selection source hash are valid.

- [ ] **Step 2: Run round 2 exactly once**

Run:

```bash
.venv/bin/agentbench generals iterate-v2 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --calibration-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/calibration-v1.toml \
  --calibration-selection /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/calibration-v1-selection.toml \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260726_1631_f56f789f \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

Expected:

- held-out calibration has 10 valid cases and score in `[0.2, 0.6]`;
- exactly one new completed or transparently failed provider act exists;
- if calibration misses the target, the command stops before Codex and no
  alternate mode is tried;
- if Codex completes and v2 is valid, all 18 v2 formal cases run.

- [ ] **Step 3: Build the Framework report**

Run:

```bash
.venv/bin/agentbench report \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --output /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/_site
```

Expected: `_site/index.html` contains separate formal, calibration, dense,
behavior, occupancy, budget, and quality sections.

### Task 10: Verify the Complete Evidence Chain

**Files:**
- Verify: all source and test files changed in Tasks 1–8
- Generated, not committed: final run and report artifacts

**Interfaces:**
- Consumes: both feature worktrees and the completed round-2 run
- Produces: test evidence, data-quality evidence, clean Git states, and a concise scientific conclusion

- [ ] **Step 1: Run complete Framework tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass; only explicitly gated live tests skip.

- [ ] **Step 2: Run complete calibration asset tests**

Run from Assets:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest backend_sources/corpus/28_generals/calibration/weak_v1/tests -q
```

Expected: PASS.

- [ ] **Step 3: Validate run data**

Run:

```bash
.venv/bin/agentbench data check \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

Expected: both the first complete run and round-2 run are valid; new event
types are recognized; no malformed lines, duplicate event IDs, missing common
fields, or silent corrections exist.

- [ ] **Step 4: Audit the round-2 artifacts**

Check programmatically:

- exact parent run ID and v1 content hash;
- calibration/formal benchmark IDs are different;
- held-out calibration score and target flag agree;
- calibration case IDs never appear in formal results or score history;
- dense traces exist for every valid evaluated match;
- prompt bytes are at most `262144`;
- prompt excludes formal and held-out seeds;
- exactly one round-2 provider act exists;
- provider JSONL, usage, tool calls, elapsed time, stderr, and patch are saved;
- v2 manifest matches the final workspace;
- evaluation incompleteness, if any, leaves missing score/gain/AUC;
- quality warnings are explicit.

- [ ] **Step 5: Inspect repository hygiene**

Run in each worktree:

```bash
git status --short
git log -10 --oneline
```

Expected: no uncommitted implementation changes and no generated run/report
artifacts staged.

- [ ] **Step 6: Apply verification and review skills**

Invoke `superpowers:verification-before-completion` and
`superpowers:requesting-code-review`. Address only evidence-backed findings,
rerun the affected focused tests, then rerun Steps 1–5.

- [ ] **Step 7: Report the measured conclusion**

Report:

- calibration v0 score and whether it is inside 20%–60%;
- formal v0, v1, and v2 scores plus gain and AUC;
- paired dense changes for survival, territory, army, coins, and pressure;
- v1/v2 action disagreement and occupancy shift;
- learning/evaluation/calibration/total budgets;
- Codex tokens, tool calls, time, changed files, and version hash;
- event-quality state;
- limits on claims from one second act.
