# Generals HL v6 Macro Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run one audited Generals HL v6 iteration that learns only from fresh strongest-human replays, creates an explainable bounded macro planner, validates on fresh high/medium games, and formally evaluates every runnable candidate.

**Architecture:** Extend the existing versioned Generals pipeline with a v6-specific frozen learning configuration, verified v5 lineage import, high-only bounded prompt, post-hoc macro-action diagnostics, and a non-gated validation/formal workflow. Codex edits an isolated copy of the exact v5 source; framework tests and source hashing freeze v6 before any validation or formal game.

**Tech Stack:** Python 3.11, dataclasses, `tomllib`, pytest, the existing AgentBenchFrame tracking/provider/snapshot APIs, the official Generals Python engine, and non-interactive Codex CLI.

## Global Constraints

- Parent run is `/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_0818_e6bcb9b3`.
- Expected v5 source SHA-256 is `facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e`.
- Learning uses only high opponent `advanced-rank02-robinliu-v18`, seeds `287101`, `287202`, `287303`, and seats `0`, `1`.
- Validation uses only high and medium opponents, seeds `288101`, `288202`, `288303`, and seats `0`, `1`.
- Formal evaluation remains the existing immutable 18 cases and is never included in the Codex prompt.
- Every runnable candidate is formally evaluated; validation and performance never gate version saving or formal evaluation.
- v6 success is medium at least `2/6` and low exactly `6/6`; high at least `1/6` is an additional breakthrough.
- The provider is invoked once in the normal v6 run; every retry remains visible in cumulative act and budget records.
- No new runtime dependency is added.
- All source, prompt, raw provider JSONL, replay, dense, action-profile, budget, quality, diff, and version artifacts are retained.

---

### Task 1: Freeze v6 learning assets and case construction

**Files:**
- Create: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v6-strongest-learning-v1.toml`
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Modify: `src/agentbench_frame/generals/evaluator.py`
- Create: `tests/generals/fixtures/v6-strongest-learning-v1.toml`
- Create: `tests/generals/test_assets_v6.py`

**Interfaces:**
- Produces: `Round6LearningConfig(learning_id, opponent_id, seeds, seats)`.
- Produces: `load_round6_learning_config(path: Path, pilot: PilotConfig) -> Round6LearningConfig`.
- Produces: `build_round6_learning_cases(config, learning) -> tuple[BenchmarkCase, ...]`.
- Produces: `build_round6_validation_cases(config) -> tuple[BenchmarkCase, ...]`.

- [ ] **Step 1: Write failing asset and case tests**

```python
def test_v6_learning_manifest_is_exact_fresh_strongest_matrix():
    config = load_round6_learning_config(FIXTURE, PILOT)
    assert config.learning_id == "generals-hl-v6-macro-strongest-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.seeds == (287101, 287202, 287303)
    assert config.seats == (0, 1)


def test_v6_validation_is_high_medium_fresh_and_disjoint():
    learning = build_round6_learning_cases(PILOT, config)
    validation = build_round6_validation_cases(PILOT)
    assert len(learning) == 6
    assert len(validation) == 12
    assert {case.metadata["tier"] for case in validation} == {"high", "medium"}
    assert {case.seed for case in validation} == {288101, 288202, 288303}
    assert {case.seed for case in learning}.isdisjoint(
        case.seed for case in validation
    )
```

- [ ] **Step 2: Run tests and verify the missing interfaces fail**

Run: `.venv/bin/python -m pytest tests/generals/test_assets_v6.py -q`

Expected: collection fails because `Round6LearningConfig` and the v6 loaders/builders do not exist.

- [ ] **Step 3: Implement the exact immutable configuration**

Add this model:

```python
@dataclass(frozen=True)
class Round6LearningConfig:
    learning_id: str
    opponent_id: str
    seeds: tuple[int, ...]
    seats: tuple[int, ...]
```

Add constants and strict validation:

```python
ROUND5_LEARNING_SEEDS = frozenset({286101, 286202, 286303})
ROUND6_LEARNING_ID = "generals-hl-v6-macro-strongest-v1"
ROUND6_LEARNING_SEEDS = frozenset({287101, 287202, 287303})
ROUND6_VALIDATION_SEEDS = (288101, 288202, 288303)
FROZEN_BEFORE_ROUND6 = FROZEN_BEFORE_ROUND5 | ROUND5_LEARNING_SEEDS
```

`load_round6_learning_config` must require the exact learning ID, exact seed
set, ordered seats `(0, 1)`, no overlap with `FROZEN_BEFORE_ROUND6`, and the
pilot's first/high opponent. `build_round6_validation_cases` must derive only
the high and medium frozen opponents and use phase `validate6`.

Create both repository and test manifests with:

```toml
learning_id = "generals-hl-v6-macro-strongest-v1"
opponent_id = "advanced-rank02-robinliu-v18"
seeds = [287101, 287202, 287303]
seats = [0, 1]
```

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/generals/test_assets_v6.py tests/generals/test_assets.py -q`

Expected: PASS.

- [ ] **Step 5: Commit framework and asset changes separately**

Framework commit:

```bash
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/assets.py src/agentbench_frame/generals/evaluator.py tests/generals/fixtures/v6-strongest-learning-v1.toml tests/generals/test_assets_v6.py
git commit -m "feat(generals): freeze v6 learning and validation splits"
```

Assets worktree commit:

```bash
git add backend_sources/corpus/28_generals/benchmark/v6-strongest-learning-v1.toml
git commit -m "assets(generals): freeze v6 strongest learning suite"
```

### Task 2: Verify and import exact v5 lineage

**Files:**
- Create: `src/agentbench_frame/generals/lineage_v6.py`
- Create: `tests/generals/test_lineage_v6.py`

**Interfaces:**
- Produces: `Round6ParentLineage` with v0-v5 scores, v5 source/manifest, prior score history, cumulative learning budget, and global act count.
- Produces: `load_round6_parent(parent_run_dir: Path, expected_parent_hash: str) -> Round6ParentLineage`.
- Produces: `import_round6_source(lineage, run_dir, snapshotter) -> WorkspaceManifest`.

- [ ] **Step 1: Write failing lineage tests**

Create a synthetic complete v5 parent whose `score_history` is
`(0.0, 0.0, None, 0.0, 7/18, 4/18, 7/18)`, `act_count` is `6`, and v5
manifest matches its source. Tests must assert:

```python
lineage = load_round6_parent(parent, v5_manifest.content_hash)
assert lineage.parent_version == "v5"
assert lineage.global_act_count == 6
assert lineage.evo_score_5 == 7 / 18
assert lineage.prior_score_history[-1] == 7 / 18
assert lineage.learning_budget["learning_coding_agent_acts"] == 6
```

Also test rejection of a wrong expected hash, modified v5 source, incomplete
formal status, a missing score, an altered score-history gap, and an act count
other than six.

- [ ] **Step 2: Run lineage tests and verify failure**

Run: `.venv/bin/python -m pytest tests/generals/test_lineage_v6.py -q`

Expected: collection fails because `lineage_v6` does not exist.

- [ ] **Step 3: Implement verified v5 import**

`load_round6_parent` must independently capture `versions/v5/source`, compare
the captured file map and content hash to `versions/v5/manifest.json`, compare
the hash to `expected_parent_hash`, and read the cumulative learning budget
from the complete v5 summary without mutating it.

`import_round6_source` must copy v5 to both `workspace/` and
`versions/v5/source/`, initialize the isolated workspace git repository, write
the preserved v5 manifest, and write:

```json
{
  "parent_run_id": "<v5 run id>",
  "parent_version": "v5",
  "parent_content_hash": "<v5 hash>",
  "starting_version": "v5"
}
```

to `versions/lineage.json`.

- [ ] **Step 4: Run focused lineage tests**

Run: `.venv/bin/python -m pytest tests/generals/test_lineage_v6.py tests/generals/test_lineage_v5.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/generals/lineage_v6.py tests/generals/test_lineage_v6.py
git commit -m "feat(generals): verify v5 parent for v6"
```

### Task 3: Build the high-only, macro-planner Codex prompt

**Files:**
- Create: `src/agentbench_frame/generals/prompt_v6.py`
- Create: `tests/generals/test_prompt_v6.py`

**Interfaces:**
- Consumes: six `CriticalLearningEvidence` records created from v5 high-only learning games.
- Consumes: `action_profile: Mapping[str, object]`; Task 4 later produces this
  payload without creating a module-level dependency.
- Produces: `build_round6_prompt(...) -> PromptBuildResult`.

- [ ] **Step 1: Write failing prompt-isolation tests**

Tests must construct six evidence records for seeds 287101/287202/287303 and
both seats and assert:

```python
result = build_round6_prompt(...)
assert result.feedback_episodes_read == 6
assert result.omitted_episode_ids == ()
assert "bounded macro-action planner" in result.prompt
assert "You may edit strategy.py, state_view.py" in result.prompt
assert "formal score" not in result.prompt.lower()
```

Parameterized candidate-evidence tests must reject every formal seed, every
historical learning seed from 281101 through 287303 except the exact v6 set,
every validation seed 288101/288202/288303, non-high evidence, duplicate
episode IDs, incomplete episode sets, and prompts exceeding the byte cap.
These restrictions apply to the six `CriticalLearningEvidence` records, not
indiscriminately to inherited parent documentation.

Static-context tests must also load the exact manifest-verified v5 strategy
and experience, official rules, and frozen replay-analysis-v2 entry. Only the
v5 experience role may retain its declared round-5 learning replay/state-ID
citations for 286101/286202/286303. V5 strategy, rules, and replay Skill remain
under the strict denylist. The experience role must still reject formal,
validation, current-v6, undeclared historical, or arbitrary six-digit seeds,
as well as whole-word formal/validation material. The exact production bundle
must pass through the public static validator without creating run, gameplay,
evaluator, or provider artifacts.

The same exact four-file bundle must also pass through the complete
`build_round6_prompt` path with the frozen Skill hash, a valid action profile,
and the exact six 287101/287202/287303 evidence records. Assert that the final
prompt retains the legitimate v5 experience replay/state-ID citations, appends
only those six round-6 evidence records, and creates no artifacts. Add
negative tests proving that benchmark ID, strict static roles, action profile,
and serialized evidence cannot introduce formal, validation, or historical
material.

An integration regression must also rehash a manifest-valid synthetic v5
parent whose `EXPERIENCE.md` contains declared 286 replay/state citations, run
the real pipeline through the fake provider, and prove temporal isolation:
learning observes zero provider calls; validation and formal evaluation occur
only after the single provider act.

- [ ] **Step 2: Run prompt tests and verify failure**

Run: `.venv/bin/python -m pytest tests/generals/test_prompt_v6.py -q`

Expected: collection fails because `prompt_v6` does not exist.

- [ ] **Step 3: Implement the complete prompt builder**

Define:

```python
ROUND6_EVIDENCE_SEEDS = frozenset({287101, 287202, 287303})
FORBIDDEN_ROUND6_EVIDENCE_SEEDS = (
    FORBIDDEN_EVALUATION_SEEDS
    | frozenset({
        281101, 281202, 281303,
        282101, 282202, 282303, 282404, 282505,
        282601, 282702, 282803, 282904, 283005,
        283101, 283202, 283303,
        284101, 284202, 284303,
        285101, 285202, 285303,
        286101, 286202, 286303,
        288101, 288202, 288303,
    })
)
```

`FORBIDDEN_ROUND6_EVIDENCE_SEEDS` governs candidate round-6 evidence. Static
context uses the role policy from Step 1: strict checks for v5 strategy, rules,
and replay Skill, and a narrow exception for the declared round-5 learning
citations already present in the exact v5 experience.

Validate each untrusted input under its own role before interpolation:
benchmark ID, v5 strategy, rules, replay Skill, action profile, and each
serialized evidence line are strict; v5 experience alone gets the narrow
round-5 citation exception. Do not concatenate differently trusted roles and
then apply a context-agnostic denylist to the assembled mandatory prompt.
Replay Skill digest format, exact-six evidence structure, and the prompt byte
cap remain independent mandatory checks.

The mandatory prompt must:

- state that v5 is the exact editable parent;
- include only manifest-verified v5 strategy/experience static inputs, with
  role-aware preflight completed before learning;
- permit edits only to `strategy.py`, `state_view.py`, `STRATEGY.md`,
  `EXPERIENCE.md`, `tests/**`, and `policy/**`;
- prohibit edits to `main.py`;
- require exact one-final-`[8]`, movement-budget bounds, deterministic
  sequential state updates, main safety, resource/main production economics,
  valuable routing, and no opponent identity/seed/filesystem/network input;
- describe commands 1/3/5 as required-safe scope and commands 2/4/6/7 as
  optional only with full legality tests;
- require strategy tests and learning replay/state-ID-backed experience;
- label dense values and action profiles as diagnostics, not causal
  information gain or KL.

Append each evidence record atomically, sorted by seed and seat. Reject any
set other than exactly the six declared episodes before returning.

- [ ] **Step 4: Run prompt tests**

Run: `.venv/bin/python -m pytest tests/generals/test_prompt_v6.py tests/generals/test_prompt_v5.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/generals/prompt_v6.py tests/generals/test_prompt_v6.py
git commit -m "feat(generals): isolate v6 strongest replay prompt"
```

### Task 4: Add macro-action behavior diagnostics

**Files:**
- Create: `src/agentbench_frame/generals/action_profile.py`
- Create: `tests/generals/test_action_profile.py`

**Interfaces:**
- Produces: immutable `ActionProfile`.
- Produces: `summarize_action_profile(evaluation: GeneralsEvaluation) -> ActionProfile`.
- Produces: `action_profile_payload(profile: ActionProfile) -> dict[str, object]`.

- [ ] **Step 1: Write failing diagnostic tests**

Build synthetic evaluated-seat turns containing end-only, upgrade, technology,
one army move, and a two-move macro. Assert:

```python
profile = summarize_action_profile(evaluation)
assert profile.turn_count == 5
assert profile.primitive_command_count == 5
assert profile.mean_primitives_per_turn == 1.0
assert profile.max_primitives_per_turn == 2
assert profile.multi_command_turn_count == 1
assert profile.command_counts == {1: 3, 3: 1, 5: 1}
assert profile.end_only_turn_count == 1
assert profile.general_upgrade_count == 1
assert profile.technology_upgrade_count == 1
assert profile.move_destination_counts["neutral_plain"] == 1
```

Also test that opponent turns are excluded, malformed movement commands become
`unknown`, zero-move profiles have ratio `None`, and output ordering is stable.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/generals/test_action_profile.py -q`

Expected: collection fails because `action_profile` does not exist.

- [ ] **Step 3: Implement read-only profiles**

Use each evaluated-seat `TurnRecord.state_before` to classify a movement
destination as `owned`, `enemy`, `neutral_general`, `neutral_plain`, or
`unknown`. Exclude every `[8]` from the primitive count but count a turn whose
only command is `[8]` as end-only. Return sorted opcode and destination maps.
Calculate `neutral_plain_move_ratio` over classifiable movement commands only.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/generals/test_action_profile.py tests/generals/test_measurement.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/generals/action_profile.py tests/generals/test_action_profile.py
git commit -m "feat(generals): profile macro action behavior"
```

### Task 5: Implement the non-gated v5-to-v6 pipeline

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v6.py`
- Create: `tests/generals/test_pipeline_v6.py`

**Interfaces:**
- Produces: `Round6PipelineResult` with v0-v6 scores, `gain_6`, act counts,
  status, run directory, and runnable flag.
- Produces: `GeneralsHLRound6Pipeline.from_paths(...)`.
- Consumes: Tasks 1-4 interfaces and existing evaluator, snapshot, provider,
  dense, replay, measurement, budget, and quality APIs.

- [ ] **Step 1: Write the failing end-to-end pipeline contract**

Use a fake evaluator and rewriting provider. The complete-path test must
assert this exact call sequence:

```python
assert evaluator.calls == [
    ("v5", "learning", 6),
    ("v6", "validation", 12),
    ("v6", "evaluation", 18),
]
assert result.status == "complete"
assert result.runnable is True
assert result.global_act_count == 7
assert result.round_act_count == 1
```

It must also assert existence of v5/v6 manifests, `v5-to-v6.patch`, learning,
validation and formal specs, prompt/raw-provider artifacts, test log, action
profile JSON files for all three phases, quality report, and a summary whose
score history appends v6.

Add tests proving:

- incomplete validation still runs formal evaluation;
- provider failure saves an unrunnable v6 manifest and no validation/formal
  score;
- candidate test failure saves the source and skips gameplay;
- prompt omission performs zero acts;
- `main.py` changes invalidate the candidate;
- `state_view.py` changes are allowed;
- formal cases never enter the prompt manifest;
- action-profile failure is visible as a quality/error event but does not
  interpolate data;
- every runnable v6 is formally evaluated regardless of score.

- [ ] **Step 2: Run pipeline tests and verify failure**

Run: `.venv/bin/python -m pytest tests/generals/test_pipeline_v6.py -q`

Expected: collection fails because `pipeline_v6` does not exist.

- [ ] **Step 3: Implement artifacts, learning, and provider act**

`GeneralsHLRound6Pipeline.run()` must:

```python
lineage = load_round6_parent(parent_run_dir, expected_parent_hash)
imported_v5 = import_round6_source(lineage, run_dir, snapshotter)
learning_cases = build_round6_learning_cases(config, learning_config)
validation_cases = build_round6_validation_cases(config)
formal_cases = tuple(build_evaluation_spec(config).cases)
v5_learning = evaluator.evaluate(
    versions_v5_source, "v5", "learning", run, cases=learning_cases
)
```

Require all six learning games valid before constructing learning-only
critical evidence and v5 action profile. Write `v6-learning-spec.json`,
`v6-validation-spec.json`, and `formal-spec.json` before the act, but pass only
learning evidence to `build_round6_prompt`.

Persist `codex-act-v6.prompt.md`, `.prompt.json`, `prompt-manifest.json`,
`.raw.jsonl`, `.stderr.log`, prompt SHA-256, skill SHA-256, feedback receipt,
and provider budget. Invoke the coding-agent controller with
`version_before="v5"` and the imported v5 manifest.

- [ ] **Step 4: Implement version validation and non-gated evaluation**

Freeze v6 immediately after the act. Allow changes only in:

```python
{
    "strategy.py",
    "state_view.py",
    "STRATEGY.md",
    "EXPERIENCE.md",
}
```

plus `tests/**` and `policy/**`; require `main.py` unchanged. Run candidate
tests, save `versions/v6/tests.log`, save `v5-to-v6.patch`, and mark runnable
only when the provider completed, scope is valid, tests pass, and both strategy
documents exist.

For every runnable v6:

```python
validation = evaluator.evaluate(
    workspace, "v6", "validation", run, cases=validation_cases
)
formal = evaluator.evaluate(
    workspace, "v6", "evaluation", run, cases=formal_cases
)
```

The formal call must be in a `finally`-equivalent path after validation so a
validation exception cannot suppress it. Save post-act profiles without
feeding them back to the provider. Compute v6 score/gain/AUC/decision-space
records with existing framework semantics and append v6 to `score_history`.

- [ ] **Step 5: Implement finalization and quality audit**

The summary must separate `learning_results`, `validation_results`, and
`formal_results`; include high/medium/low formal scores; include success
booleans `medium_at_least_2_of_6`, `low_retained_6_of_6`, and
`high_breakthrough_at_least_1_of_6`; include local and cumulative learning
budgets; and never convert missing values to zero.

Finalize with `inspect_event_file`, write `quality.json`, and retain the run
for every terminal status.

- [ ] **Step 6: Run focused pipeline tests**

Run: `.venv/bin/python -m pytest tests/generals/test_pipeline_v6.py tests/generals/test_pipeline_v5.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/generals/pipeline_v6.py tests/generals/test_pipeline_v6.py
git commit -m "feat(generals): orchestrate audited v6 iteration"
```

### Task 6: Expose v6 CLI and audited recovery

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Create: `tests/generals/test_cli_v6.py`
- Modify: `src/agentbench_frame/generals/pipeline_v6.py`
- Modify: `tests/generals/test_pipeline_v6.py`

**Interfaces:**
- Produces CLI commands `generals iterate-v6` and `generals recover-v6`.
- Produces `GeneralsHLRound6Pipeline.recover(failed_run_dir: Path)`.

- [ ] **Step 1: Write failing CLI and recovery tests**

The parser test must require:

```text
--agentbench-root --manifest --learning-manifest --replay-skill
--data-dir --parent-run --expected-parent-hash
--codex-executable --provider-timeout
```

`recover-v6` additionally requires `--failed-run`. Routing tests must assert
all paths and hashes reach `from_paths`, and output includes `evo_score_6`,
`gain_6`, act counts, and runnable.

Recovery tests must prove:

- a `prompt_incomplete` zero-act run may reuse its audited learning budget but
  still rebuilds and hashes the prompt;
- a provider-failed run is retried as a new visible act and increments
  cumulative act count;
- a post-act run with a verified runnable v6 source and incomplete evaluation
  performs no provider act and runs validation/formal under a recovery run;
- mismatched parent, v6, prompt, learning ID, or terminal status is rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/generals/test_cli_v6.py tests/generals/test_pipeline_v6.py -q`

Expected: FAIL because the v6 commands and recovery entrypoint are absent.

- [ ] **Step 3: Implement CLI routing and recovery provenance**

Register both commands and use the same provider configuration as v5. Recovery
must create a new run linked to `failed_run_id`; it must never append to or
mutate a finalized run. A provider retry reuses only verified learning/prompt
inputs and records another act. A post-act evaluation recovery verifies the
saved v6 manifest/hash, copies the exact source, records zero new acts, and
runs the complete validation and formal matrices so the recovery result is
self-contained.

- [ ] **Step 4: Run focused CLI/recovery tests**

Run: `.venv/bin/python -m pytest tests/generals/test_cli_v6.py tests/generals/test_pipeline_v6.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/generals/cli.py src/agentbench_frame/generals/pipeline_v6.py tests/generals/test_cli_v6.py tests/generals/test_pipeline_v6.py
git commit -m "feat(generals): expose v6 iteration and recovery"
```

### Task 7: Integrate documentation and run regression verification

**Files:**
- Modify: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/README.md`
- Modify: `README.md`
- Create: `docs/experiments/2026-07-29-generals-v6-protocol.md`

**Interfaces:**
- Documents the exact production invocation, split isolation, recovery
  semantics, expected artifacts, and interpretation limits.

- [ ] **Step 1: Add executable commands and audit checklist**

Document the full `iterate-v6` invocation using the exact parent path/hash,
asset paths, local data directory, Codex executable, and 1800-second provider
timeout. State explicitly that validation/formal evidence is post-act only.

- [ ] **Step 2: Run formatting and focused Generals tests**

Run: `git diff --check`

Expected: no output.

Run: `.venv/bin/python -m pytest tests/generals -q`

Expected: PASS.

- [ ] **Step 3: Run the complete framework suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS with no new failures.

- [ ] **Step 4: Commit documentation**

Framework:

```bash
git add README.md docs/experiments/2026-07-29-generals-v6-protocol.md
git commit -m "docs(generals): publish v6 execution protocol"
```

Assets:

```bash
git add backend_sources/corpus/28_generals/README.md
git commit -m "docs(generals): document v6 learning manifest"
```

### Task 8: Execute the real v6 act, evaluation, and result audit

**Files:**
- Runtime artifacts: `agentbench_data/runs/28_generals/generals-hl/<new-run-id>/`
- Create after results: `docs/experiments/2026-07-29-generals-v6-macro-planner-result.md`

**Interfaces:**
- Consumes the exact frozen assets, parent hash, and local Codex CLI.
- Produces the immutable v6 source hash, full run artifacts, and evidence-based
  conclusion.

- [ ] **Step 1: Verify clean inputs**

Run in the framework worktree:

```bash
git status --short
```

Expected: no output.

Run in the assets worktree:

```bash
git status --short
```

Expected: no output.

- [ ] **Step 2: Run the real v6 command**

```bash
.venv/bin/python -m agentbench_frame.cli generals iterate-v6 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --learning-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v6-strongest-learning-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_0818_e6bcb9b3 \
  --expected-parent-hash facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

Expected: JSON with a terminal run directory. `complete` is the successful
harness outcome; a different status triggers only the audited recovery path
defined in Task 6.

- [ ] **Step 3: Audit artifacts before reading the performance conclusion**

Verify the v6 manifest against the captured source, prompt hash against prompt
bytes, exactly six learning cases, exactly twelve validation cases, exactly
eighteen formal cases, no seed overlap, one normal-run provider act, all game
artifact families, action profiles, budgets, and `quality.json` counts.

Run candidate tests directly from `versions/v6/source` and rerun the focused
framework v6 tests.

- [ ] **Step 4: Write the result report**

The report must include:

- v6 source hash and parent hash;
- learning, validation, and formal score tables;
- formal high/medium/low and seat/seed breakdown;
- v5-to-v6 score/gain/AUC comparison;
- primitive-action count, command histogram, upgrade count, end-only count,
  and neutral-plain movement ratio;
- dense metrics and information-gain records with correct non-causal labels;
- provider token/time/tool-call budget;
- quality diagnostics and all invalid/incomplete states;
- whether medium `>=2/6`, low `=6/6`, and high `>=1/6`;
- the next bottleneck supported by saved evidence.

- [ ] **Step 5: Verify and commit the result report**

Run: `git diff --check`

Run: `.venv/bin/python -m pytest tests/generals/test_pipeline_v6.py tests/generals/test_prompt_v6.py tests/generals/test_action_profile.py -q`

Expected: PASS.

```bash
git add docs/experiments/2026-07-29-generals-v6-macro-planner-result.md
git commit -m "docs(generals): publish audited v6 result"
```
