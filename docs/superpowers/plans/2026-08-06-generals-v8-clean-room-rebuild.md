# Generals v8 Clean-Room Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the superseded v8 experiment from local deliverable history and produce one new, leak-free, single-act v8 iteration from frozen v7 with complete evaluation, logs, and controlled-reference KL.

**Architecture:** Reconstruct both local branches across the contiguous old-v8 commit interval, verify that v0-v7 authorities are byte-identical, and delete the two inventoried old run trees. Build a fresh v8 contract and pipeline only from v7 interfaces: six learning games feed one Codex act, then twelve validation and eighteen unconditional formal games run against disjoint seeds. Extend the immutable v0-v7 KL run append-only by probing only the frozen new v8 source.

**Tech Stack:** Python 3.11, pytest, TOML/JSON/JSONL contracts, Git linked worktrees, official Generals Python SDK, AgentBench `Run` event tracking, Codex non-interactive provider, exact canonical macro-action enumeration.

## Global Constraints

- Do not push, create a pull request, or merge either branch during this plan.
- Do not read or reuse old-v8 code, prompts, patches, replays, summaries, scores, or analysis while building the new v8.
- Preserve v0-v7 source/run lineage, v7 run `20260730_1739_680b1632`, and controlled-reference KL run `20260731_1808_aedfbce7` byte for byte.
- The new v8 parent is the frozen v7 source with content hash `c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
- Invoke Codex exactly once; validation or formal evidence can never trigger another act within v8.
- Learning uses seeds `300101`, `300202`, `300303`; validation uses `301101`, `301202`, `301303`, `301404`, `301505`, `301606`; both use seats `0` and `1`.
- Formal evaluation remains the unchanged 18-case `pilot-v1.toml` matrix.
- A successful performance target requires high tier at least `2/6`, total at least `12/18`, and at least one high-tier win from each seat.
- Every runnable v8 receives all 18 formal cases even when validation fails.
- Policy KL uses the existing 12 states, exact uniform `U_s`, and epsilons `0.001`, `0.01`, `0.05`, `0.1` with `12/12` coverage.
- Do not interpolate the known missing historical score point; full-history AUC remains unavailable.
- Use `apply_patch` for hand-authored file changes and explicit validated paths for deletion.

---

### Task 1: Remove the old v8 history and data without changing v0-v7

**Files:**
- Delete through history reconstruction: old-v8 files under `src/agentbench_frame/generals/`, `tests/generals/`, `docs/superpowers/`, and old-v8 report changes
- Delete through history reconstruction: old-v8 files under `backend_sources/corpus/28_generals/benchmark/` and `backend_sources/corpus/28_generals/tests/`
- Preserve: `docs/superpowers/specs/2026-08-06-generals-v8-clean-room-rebuild-design.md`
- Preserve: `docs/experiments/2026-08-01-generals-controlled-policy-kl-v7-result.md`

**Interfaces:**
- Consumes: the current local Framework and Assets branches plus the two old run directories found by their `versions/v8` snapshot
- Produces: rewritten clean branches whose tips retain the approved design and v0-v7 KL work, with no old-v8 deliverable reference

- [ ] **Step 1: Capture retained scientific hashes and run focused pre-rewrite tests**

Run:

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
sha256sum agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632/summary.json
sha256sum agentbench_data/runs/28_generals/generals-policy-kl/20260731_1808_aedfbce7/summary.json
.venv/bin/pytest tests/generals/test_pipeline_v7.py tests/generals/test_policy_kl_extension.py tests/generals/test_policy_kl_figure.py -q
```

Expected: all focused tests pass; record both SHA-256 values in the execution log, not in a retained source file.

- [ ] **Step 2: Resolve the exact contiguous old-v8 commit interval without opening its contents**

Run:

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
git log --format='%H%x09%s' --reverse | rg 'design v8 transition-parity challenge|plot retried v8 at global act'
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
git log --format='%H%x09%s' --reverse | rg 'freeze v8 transition contract|freeze v8 champion challenge'
```

Expected: one first and one last boundary in each contiguous interval. Abort if ordering is not contiguous or either boundary is ambiguous.

- [ ] **Step 3: Rewrite the Framework branch across the resolved interval**

Run the following after replacing the shell variables with the exact hashes printed in Step 2 inside the same shell session:

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
generals_v8_first_commit=$(git log --format='%H%x09%s' --reverse | awk -F '\t' '$2 ~ /design v8 transition-parity challenge/ {print $1; exit}')
generals_v8_last_commit=$(git log --format='%H%x09%s' | awk -F '\t' '$2 ~ /plot retried v8 at global act/ {print $1; exit}')
test -n "$generals_v8_first_commit"
test -n "$generals_v8_last_commit"
git rebase --onto "${generals_v8_first_commit}^" "$generals_v8_last_commit" zhaoyicheng/generals-hl-implementation
```

Expected: later v0-v7 KL commits and the approved clean-room design replay successfully. Resolve a conflict only by keeping the pre-v8 side for removed CLI/report behavior and the later side for v0-v7 KL behavior; abort instead of guessing when those concerns cannot be separated.

- [ ] **Step 4: Rewrite the Assets branch across its resolved interval**

Run:

```bash
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
generals_assets_v8_first=$(git log --format='%H%x09%s' --reverse | awk -F '\t' '$2 ~ /freeze v8 transition contract/ {print $1; exit}')
generals_assets_v8_last=$(git log --format='%H%x09%s' | awk -F '\t' '$2 ~ /freeze v8 champion challenge/ {print $1; exit}')
test -n "$generals_assets_v8_first"
test -n "$generals_assets_v8_last"
git rebase --onto "${generals_assets_v8_first}^" "$generals_assets_v8_last" zhaoyicheng/generals-assets
```

Expected: the v7 champion and policy-KL v7 extension assets remain; old-v8 asset files are absent.

- [ ] **Step 5: Resolve and delete only the two inventoried old run trees**

Run a read-only resolution first:

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
find agentbench_data/runs/28_generals/generals-hl -mindepth 3 -maxdepth 3 -type d -path '*/versions/v8' -printf '%h\n' | sed 's#/versions$##' | sort -u
```

Expected: exactly two absolute or worktree-relative run roots. Verify each root is below `agentbench_data/runs/28_generals/generals-hl/`, contains `versions/v8/manifest.json`, and is not the v7 authority. Delete those two explicit roots individually; do not use a glob or unresolved variable as the deletion target.

- [ ] **Step 6: Verify removal and retained hashes**

Run:

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
test -z "$(find src tests -type f -iname '*v8*')"
test -z "$(rg -l 'transition-parity act|blind v8 retry' src tests docs --glob '!**/*clean-room*' || true)"
sha256sum agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632/summary.json
sha256sum agentbench_data/runs/28_generals/generals-policy-kl/20260731_1808_aedfbce7/summary.json
.venv/bin/pytest tests/generals/test_pipeline_v7.py tests/generals/test_policy_kl_extension.py tests/generals/test_policy_kl_figure.py -q
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
test -z "$(find backend_sources/corpus/28_generals -type f -iname '*v8*')"
```

Expected: the two hashes exactly match Step 1, all focused tests pass, and no old-v8 file remains. The approved clean-room design is allowed because its filename describes the new experiment.

### Task 2: Freeze the new clean-room challenge asset

**Files:**
- Create: `backend_sources/corpus/28_generals/benchmark/v8-clean-room-challenge-v1.toml`
- Create: `backend_sources/corpus/28_generals/tests/test_v8_clean_room_contract.py`

**Interfaces:**
- Consumes: `pilot-v1.toml`, official engine SHA-256, replay-analysis-v2 Skill SHA-256
- Produces: a frozen TOML contract with `challenge_id`, `opponent_id`, learning/validation seeds, seats, diagnostic validation thresholds, formal success thresholds, and runtime digests

- [ ] **Step 1: Write the failing asset contract test**

Add assertions equivalent to:

```python
def test_v8_clean_room_contract_is_exact_and_disjoint():
    raw = tomllib.loads(CHALLENGE.read_text(encoding="utf-8"))
    assert raw["challenge_id"] == "generals-hl-v8-clean-room-v1"
    assert raw["opponent_id"] == "advanced-rank02-robinliu-v18"
    assert raw["learning_seeds"] == [300101, 300202, 300303]
    assert raw["validation_seeds"] == [
        301101, 301202, 301303, 301404, 301505, 301606,
    ]
    assert raw["seats"] == [0, 1]
    assert raw["formal_high_min_wins"] == 2
    assert raw["formal_total_min_wins"] == 12
    assert raw["formal_high_min_wins_per_seat"] == 1
    assert set(raw["learning_seeds"]).isdisjoint(raw["validation_seeds"])
```

Also scan every frozen benchmark TOML and assert each new seed occurs only in this manifest.

- [ ] **Step 2: Run the test and verify the asset is missing**

Run: `pytest backend_sources/corpus/28_generals/tests/test_v8_clean_room_contract.py -q`

Expected: FAIL because `v8-clean-room-challenge-v1.toml` does not exist.

- [ ] **Step 3: Add the exact frozen manifest**

Create this content with the current official engine and Skill digests:

```toml
challenge_id = "generals-hl-v8-clean-room-v1"
opponent_id = "advanced-rank02-robinliu-v18"
learning_seeds = [300101, 300202, 300303]
validation_seeds = [301101, 301202, 301303, 301404, 301505, 301606]
seats = [0, 1]
validation_min_wins = 2
validation_min_wins_per_seat = 1
formal_high_min_wins = 2
formal_total_min_wins = 12
formal_high_min_wins_per_seat = 1
engine_sha256 = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
replay_skill_sha256 = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
```

- [ ] **Step 4: Run the asset suite**

Run: `pytest backend_sources/corpus/28_generals/tests/test_v8_clean_room_contract.py backend_sources/corpus/28_generals/tests/test_v7_champion_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the asset contract**

```bash
git add backend_sources/corpus/28_generals/benchmark/v8-clean-room-challenge-v1.toml backend_sources/corpus/28_generals/tests/test_v8_clean_room_contract.py
git commit -m "assets(generals): freeze clean-room v8 challenge"
```

### Task 3: Add clean-room challenge loading and exact v7 lineage

**Files:**
- Modify: `src/agentbench_frame/generals/models.py`
- Create: `src/agentbench_frame/generals/challenge_v8.py`
- Create: `src/agentbench_frame/generals/lineage_v8.py`
- Create: `tests/generals/fixtures/v8-clean-room-challenge-v1.toml`
- Create: `tests/generals/test_challenge_v8.py`
- Create: `tests/generals/test_lineage_v8.py`

**Interfaces:**
- Consumes: `PilotConfig`, v7 run summary and `versions/v7/manifest.json`
- Produces: `Round8ChallengeConfig`, `Round8ParentLineage`, `load_round8_challenge_config()`, `build_round8_learning_cases()`, `build_round8_validation_cases()`, `load_round8_parent()`, `import_round8_source()`

- [ ] **Step 1: Write failing challenge and lineage tests**

Use these core assertions:

```python
def test_round8_contract_loads_exact_clean_room_partitions():
    config = load_round8_challenge_config(
        CHALLENGE, pilot, engine_hash=ENGINE_HASH,
        replay_skill_sha256=SKILL_HASH,
    )
    assert config.learning_seeds == (300101, 300202, 300303)
    assert config.validation_seeds == (
        301101, 301202, 301303, 301404, 301505, 301606,
    )
    assert config.formal_success_threshold == (2, 12, 1)

def test_round8_parent_is_exact_complete_v7(tmp_path):
    parent, manifest = make_complete_v7_parent(tmp_path)
    lineage = load_round8_parent(parent, manifest.content_hash)
    assert lineage.parent_version == "v7"
    assert lineage.global_act_count == 8
    assert lineage.evo_score_7 == 12 / 18
    assert lineage.prior_score_history[-1] == 12 / 18
```

Mutation tests must reject changed seeds, seat order, thresholds, engine/Skill digests, incomplete formal status, an incorrect v7 hash, a score-history gap moved from its historical position, and any parent mutation.

- [ ] **Step 2: Run the focused tests and verify import failures**

Run: `.venv/bin/pytest tests/generals/test_challenge_v8.py tests/generals/test_lineage_v8.py -q`

Expected: FAIL because the new modules and model do not exist.

- [ ] **Step 3: Implement the immutable models and loaders**

Add this model shape:

```python
@dataclass(frozen=True)
class Round8ChallengeConfig:
    challenge_id: str
    opponent_id: str
    learning_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    seats: tuple[int, ...]
    validation_min_wins: int
    validation_min_wins_per_seat: int
    formal_high_min_wins: int
    formal_total_min_wins: int
    formal_high_min_wins_per_seat: int
    engine_sha256: str
    replay_skill_sha256: str

    @property
    def formal_success_threshold(self) -> tuple[int, int, int]:
        return (
            self.formal_high_min_wins,
            self.formal_total_min_wins,
            self.formal_high_min_wins_per_seat,
        )
```

Implement strict integer parsing without bool/float coercion, exact constant matching, strongest-opponent identity, pairwise seed disjointness, rejection against every previously frozen seed, and digest equality. Build six `learn8` and twelve `validate8` cases in seed-major, seat-minor order.

- [ ] **Step 4: Implement exact v7 import**

`load_round8_parent()` must require `status=evaluation_status=complete`, `formal_attempted=true`, `act_count=8`, the fixed historical score sequence through `evo_score_7`, and a manifest matching the v7 source tree. `import_round8_source()` materializes immutable v7 into both `workspace/` and `versions/v7/source/`, initializes the editable workspace repository, and writes `versions/lineage.json` with starting version `v7`.

- [ ] **Step 5: Run focused and regression tests**

Run: `.venv/bin/pytest tests/generals/test_challenge_v8.py tests/generals/test_lineage_v8.py tests/generals/test_challenge_v7.py tests/generals/test_lineage_v7.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/challenge_v8.py src/agentbench_frame/generals/lineage_v8.py tests/generals/fixtures/v8-clean-room-challenge-v1.toml tests/generals/test_challenge_v8.py tests/generals/test_lineage_v8.py
git commit -m "feat(generals): define clean-room v8 lineage"
```

### Task 4: Build leak-safe replay evidence and the single-act prompt

**Files:**
- Create: `src/agentbench_frame/generals/prompt_v8.py`
- Create: `tests/generals/test_prompt_v8.py`
- Modify: `src/agentbench_frame/generals/replay.py`
- Modify: `tests/generals/test_replay_prompt.py`

**Interfaces:**
- Consumes: exactly six `CriticalLearningEvidence` objects, v7 strategy/experience, official rules, replay Skill, and learning action profile
- Produces: `validate_round8_static_context()` and `build_round8_prompt() -> PromptBuildResult` with a complete provenance manifest

- [ ] **Step 1: Write failing leak and evidence-selection tests**

Cover exact learning episode IDs, seed/seat pairs, prompt order, and rejection of held-out material:

```python
@pytest.mark.parametrize("forbidden", [
    "301101", "280101", "controlled_policy_kl", "formal_score",
    "validation", "top_algorithms/", "transition-parity act",
])
def test_round8_prompt_rejects_nonlearning_material(forbidden):
    evidence = list(valid_learning_evidence())
    evidence[0] = replace(evidence[0], summary=forbidden)
    with pytest.raises(ValueError):
        build_round8_prompt(evidence=tuple(evidence), **valid_context())

def test_round8_prompt_reads_exactly_six_learning_episodes():
    result = build_round8_prompt(
        evidence=valid_learning_evidence(), **valid_context()
    )
    assert result.feedback_episodes_read == 6
    assert result.manifest["learning_seeds"] == [300101, 300202, 300303]
    assert result.manifest["provider_act_limit"] == 1
```

Add replay selector tests for first main-danger, large-stack inactivity, missed counter-capture/reinforcement/production capture, economy-defense conflict, first dense divergence, and low-value or unsafe long macro.

- [ ] **Step 2: Run tests and verify failures**

Run: `.venv/bin/pytest tests/generals/test_prompt_v8.py tests/generals/test_replay_prompt.py -q`

Expected: FAIL on missing v8 prompt builder and selectors.

- [ ] **Step 3: Implement deterministic evidence reasons**

Extend the replay selector with named reasons:

```python
ROUND8_SELECTION_REASONS = (
    "first_main_danger",
    "large_stack_inactive",
    "missed_counter_or_reinforcement",
    "economy_defense_conflict",
    "first_dense_divergence",
    "unsafe_or_low_value_macro",
    "final_decision",
)
```

Each reason must be derived only from the current learning replay and dense trace. Tie-breaking is by decision index; selected windows are bounded and deduplicated.

- [ ] **Step 4: Implement strict prompt construction**

The builder validates exactly the six declared `(seed, seat)` pairs, rejects every known nonlearning seed and phase marker, rejects opponent source, orders evidence by seed then seat, enforces `max_bytes=131072`, and records prompt SHA-256, Skill SHA-256, episode/state IDs, serialized bytes read, forbidden partitions, parent version/hash, and `provider_act_limit=1`.

The instruction permits explainable Python/search/planning refactors and requires compact `STRATEGY.md` and `EXPERIENCE.md`; it must not supply benchmark scores, KL actions, validation evidence, or old-v8 material.

- [ ] **Step 5: Run focused and v7 prompt regressions**

Run: `.venv/bin/pytest tests/generals/test_prompt_v8.py tests/generals/test_replay_prompt.py tests/generals/test_prompt_v7.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/prompt_v8.py src/agentbench_frame/generals/replay.py tests/generals/test_prompt_v8.py tests/generals/test_replay_prompt.py
git commit -m "feat(generals): build leak-safe clean-room v8 prompt"
```

### Task 5: Implement the one-act v8 pipeline and fail-closed recovery

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v8.py`
- Create: `tests/generals/test_pipeline_v8.py`
- Create: `tests/generals/test_pipeline_v8_recovery.py`

**Interfaces:**
- Consumes: `Round8ChallengeConfig`, `Round8ParentLineage`, provider, evaluator, replay Skill, pilot assets
- Produces: `Round8PipelineResult` and `GeneralsHLRound8Pipeline.run()` / `.recover()`

- [ ] **Step 1: Write failing happy-path and failure-path integration tests**

The happy path must assert:

```python
result = pipeline.run()
assert fake_provider.calls == 1
assert len(fake_evaluator.calls_for("learning")) == 6
assert len(fake_evaluator.calls_for("validation")) == 12
assert len(fake_evaluator.calls_for("formal")) == 18
assert result.global_act_count == 9
assert result.round_act_count == 1
assert result.runnable is True
assert result.formal_attempted is True
```

Add cases proving validation failure still runs 18 formal games, candidate test failure creates no v8 score, protected-file mutation aborts, a second provider call is impossible, both-seat high-tier success is calculated exactly, and all artifacts remain when formal performance regresses.

- [ ] **Step 2: Run tests and verify failures**

Run: `.venv/bin/pytest tests/generals/test_pipeline_v8.py tests/generals/test_pipeline_v8_recovery.py -q`

Expected: FAIL because the clean-room pipeline does not exist.

- [ ] **Step 3: Implement result and orchestration boundaries**

Use this result shape:

```python
@dataclass(frozen=True)
class Round8PipelineResult:
    run_dir: Path
    status: str
    runnable: bool
    raw_score: float | None
    evo_score_8: float | None
    gain_8: float | None
    validation_passed: bool
    formal_attempted: bool
    performance_target_met: bool
    global_act_count: int
    round_act_count: int
```

The pipeline verifies/imports v7, writes immutable suite and Skill receipts,
runs six v7 learning games, writes replay/dense/action-profile evidence, builds
the prompt, invokes the provider once, validates the editable scope, runs
candidate tests, and freezes v8 before validation. It then runs all validation
and formal games and writes score, gain, tier, seat, dense, behavior, budget,
feedback-read, source, patch, provider, and event-quality artifacts.

- [ ] **Step 4: Enforce runnable candidate gates**

Allow changes only to `strategy.py`, `state_view.py`, `STRATEGY.md`,
`EXPERIENCE.md`, `policy/**`, and `tests/**`. Require `main.py`, strategy,
state view, and both documents; reject symlinks, randomness, network,
subprocesses, environment dependence, seed/opponent/replay identifiers, files
over the existing workspace limit, nondeterministic output, illegal commands,
multiple final `[8]`, or latency beyond the official decision bound.

- [ ] **Step 5: Implement recovery identity checks**

`recover(failed_run)` may resume only after a provider act has a frozen prompt,
raw output, candidate source, and candidate manifest. It verifies all hashes,
does not invoke the provider, rejects a complete run, and appends only missing
validation/formal/report events with deterministic event IDs.

- [ ] **Step 6: Run focused and v7 pipeline regressions**

Run: `.venv/bin/pytest tests/generals/test_pipeline_v8.py tests/generals/test_pipeline_v8_recovery.py tests/generals/test_pipeline_v7.py tests/generals/test_pipeline_v7_recovery.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/generals/pipeline_v8.py tests/generals/test_pipeline_v8.py tests/generals/test_pipeline_v8_recovery.py
git commit -m "feat(generals): run one clean-room v8 act"
```

### Task 6: Add CLI and append-only v8 KL extension

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Create: `src/agentbench_frame/generals/policy_kl_v8_extension.py`
- Create: `tests/generals/test_cli_v8.py`
- Create: `tests/generals/test_policy_kl_v8_extension.py`

**Interfaces:**
- Consumes: clean-room v8 challenge, frozen v7 parent, completed new-v8 run, and authoritative v0-v7 KL run
- Produces: CLI commands `iterate-v8`, `recover-v8`, `extend-policy-kl-v8`, `recover-policy-kl-v8`; `GeneralsPolicyKLV8ExtensionPipeline`

- [ ] **Step 1: Write failing CLI and KL projection tests**

Assert parser requirements and exact append-only projection:

```python
def test_v8_kl_extension_reuses_v0_v7_and_probes_only_v8(tmp_path):
    result = make_pipeline(tmp_path).run()
    metric = result.summary["controlled_reference_policy_kl"]
    assert [item["version_after"] for item in metric["transitions"]][-1] == "v8"
    assert len(metric["transitions"]) == 8
    assert len(result.new_policy_actions) == 12
    assert len(result.new_kl_rows) == 48
    assert all(item["coverage"] == {"complete": 12, "total": 12}
               for item in metric["transitions"])
```

Tests must also reject a changed source tree, non-v8 target, target hash mismatch,
nondeterministic v8 probe, incomplete coverage, polluted recovered events, and
recovery of a complete run.

- [ ] **Step 2: Run tests and verify failures**

Run: `.venv/bin/pytest tests/generals/test_cli_v8.py tests/generals/test_policy_kl_v8_extension.py -q`

Expected: FAIL because the new CLI and extension pipeline do not exist.

- [ ] **Step 3: Implement the v8 CLI without a retry command**

`iterate-v8` and `recover-v8` accept challenge manifest, replay Skill, parent
run/hash, data directory, Codex executable, and timeout. `extend-policy-kl-v8`
accepts source KL run, target v8 run/hash, reference manifest v2, data directory,
pilot manifest, and AgentBench root. There is deliberately no `retry-v8` path.

- [ ] **Step 4: Implement append-only KL reuse**

Verify the entire v0-v7 source receipt/tree/events/transition projection, copy
it byte-exactly into a new measurement run, resolve only the target v8 source,
probe each state twice in fresh subprocesses, and append twelve v8 action events
plus forty-eight v7-to-v8 KL events. Preserve source event provenance and fail
closed on unconsumed or changed recovery events.

- [ ] **Step 5: Run focused and v7 KL regressions**

Run: `.venv/bin/pytest tests/generals/test_cli_v8.py tests/generals/test_policy_kl_v8_extension.py tests/generals/test_policy_kl_extension.py tests/generals/test_policy_kl_figure.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/cli.py src/agentbench_frame/generals/policy_kl_v8_extension.py tests/generals/test_cli_v8.py tests/generals/test_policy_kl_v8_extension.py
git commit -m "feat(generals): expose clean-room v8 and KL extension"
```

### Task 7: Run preflight and the one real clean-room act

**Files:**
- Generate under ignored data: one newly allocated `Run.run_id` directory below `agentbench_data/runs/28_generals/generals-hl/`
- Generate under ignored data: provider logs, replay artifacts, source snapshots, patch, events, and summary

**Interfaces:**
- Consumes: completed Tasks 1-6, frozen v7 parent, local Codex CLI/authentication, official engine
- Produces: exactly one real new-v8 run, runnable or explicitly failed

- [ ] **Step 1: Run preflight suites before any provider call**

Run:

```bash
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
pytest backend_sources/corpus/28_generals/tests -q
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
.venv/bin/pytest tests/generals/test_challenge_v8.py tests/generals/test_lineage_v8.py tests/generals/test_prompt_v8.py tests/generals/test_pipeline_v8.py tests/generals/test_pipeline_v8_recovery.py tests/generals/test_cli_v8.py -q
```

Expected: PASS with no run directory created and no provider invocation.

- [ ] **Step 2: Execute the single act**

Run:

```bash
.venv/bin/python -m agentbench_frame.cli generals iterate-v8 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --challenge-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v8-clean-room-challenge-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632 \
  --expected-parent-hash c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4 \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --codex-executable codex \
  --provider-timeout 1800
```

Expected: one new run and one provider act. Do not run `iterate-v8` a second time if performance is poor.

- [ ] **Step 3: Audit the terminal run**

Check `summary.json`, `events.jsonl`, provider receipt/raw JSONL, prompt manifest,
lineage, candidate tests, v7-to-v8 patch, six learning matches, twelve validation
matches, eighteen formal matches, budgets, dense diagnostics, action profiles,
and event quality. If the process stopped after a frozen candidate, invoke
`recover-v8` once; if it stopped before a frozen candidate, report a failed act
without retrying.

- [ ] **Step 4: Verify score semantics**

Assert `raw_score=0`, `evo_score_8=formal_wins/18`, `gain_8=evo_score_8`, all
18 formal results are valid, and `performance_target_met` equals the conjunction
of high wins at least two, total wins at least twelve, and high-tier wins from
both seats. No code or prompt may change after inspecting these results.

### Task 8: Measure v7-to-v8 KL and write the audited result

**Files:**
- Generate under ignored data: one newly allocated `Run.run_id` directory below `agentbench_data/runs/28_generals/generals-policy-kl/`
- Create: `docs/experiments/2026-08-06-generals-v8-clean-room-result.md`
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `src/agentbench_frame/generals/paper_figure.py`
- Modify: `tests/generals/test_policy_kl_figure.py`
- Modify: `tests/test_local_report_research.py`
- Regenerate: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.png`
- Regenerate: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg`

**Interfaces:**
- Consumes: authoritative v0-v7 KL run and completed runnable v8 source
- Produces: append-only v0-v8 measurement, English three-panel figure, and result report

- [ ] **Step 1: Run the v8 KL extension**

Resolve the new v8 run directory and content hash from the audited terminal
summary, then run `extend-policy-kl-v8` with those exact values and source KL run
`20260731_1808_aedfbce7`.

Expected: a complete run with 12 new policy actions, 48 new per-state KL rows,
all earlier transition objects unchanged, four epsilon curves complete, and
zero event-quality defects.

- [ ] **Step 2: Write a failing figure test for eight transitions**

```python
def test_three_panel_figure_includes_v7_to_v8(tmp_path):
    output = render_controlled_policy_kl_figure(V8_SUMMARY, tmp_path)
    assert output.transition_labels == (
        "v0→v1", "v1→v2", "v2→v3", "v3→v4",
        "v4→v5", "v5→v6", "v6→v7", "v7→v8",
    )
```

- [ ] **Step 3: Update and regenerate the figure**

Make the renderer consume the complete transition array without hard-coding
seven items while retaining strict finite, nonnegative, `12/12` validation.
Generate PNG and SVG from the new authoritative run.

- [ ] **Step 4: Write the audited English result report**

Report exact run/source/prompt hashes, formal high/medium/low and seat scores,
validation, dense changes, budgets, action profile, v7-to-v8 KL and epsilon
sensitivity, event quality, and whether the performance target was met. State
explicitly that controlled KL is not performance or epistemic information gain.

- [ ] **Step 5: Expose v8 without manufacturing missing report data**

Update the report projection and template to append the observed `evo_score_8`,
`gain_8`, budgets, validation status, formal tier/seat scores, and v7-to-v8 KL
only when the corresponding events or summary fields exist. A failed provider
run remains visibly failed and contributes no synthetic score point. Add a
fixture with one complete v8 and one provider-failed v8 and assert the report
distinguishes them.

- [ ] **Step 6: Run report and figure tests**

Run: `.venv/bin/pytest tests/generals/test_policy_kl_figure.py tests/generals/test_policy_kl_v8_extension.py tests/test_local_report_research.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the audited result**

```bash
git add src/agentbench_frame/generals/paper_figure.py src/agentbench_frame/report/builder.py src/agentbench_frame/report/templates/index.html tests/generals/test_policy_kl_figure.py tests/test_local_report_research.py docs/experiments/2026-08-06-generals-v8-clean-room-result.md docs/experiments/figures/generals-controlled-policy-kl-three-panel.png docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg
git commit -m "docs(generals): report clean-room v8 result"
```

### Task 9: Complete full verification and stop before push

**Files:**
- Modify only if verification exposes a scoped defect covered by this plan

**Interfaces:**
- Consumes: both final local branches and all retained/new run data
- Produces: clean verified worktrees and a user-facing handoff with no remote mutation

- [ ] **Step 1: Run complete Framework tests**

Run: `cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl && .venv/bin/pytest -q`

Expected: all tests pass; only documented optional skips are allowed.

- [ ] **Step 2: Run complete Generals asset contracts**

Run: `cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets && pytest backend_sources/corpus/28_generals/tests -q`

Expected: all tests pass.

- [ ] **Step 3: Validate all AgentBench data**

Run: `cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl && .venv/bin/python -m agentbench_frame.cli data check --data-dir agentbench_data`

Expected: zero invalid runs and zero malformed, unknown, duplicate, or missing-ID event defects for the new runs.

- [ ] **Step 4: Verify clean history and worktrees**

Run `git status --short`, `git log --all --oneline --grep` searches for the old
experiment subjects, and file/content searches for old run IDs/hashes captured
ephemerally in Task 1. Expected: both worktrees clean, old deliverable commits
unreferenced, old data absent, v0-v7 hashes unchanged, and only the new clean-room
v8 implementation/results present.

- [ ] **Step 5: Report and wait**

Provide the exact Framework and Assets commit IDs, new HL/KL run paths, score
breakdown, target status, test counts, data-quality result, and key limitations.
Do not push until the user explicitly authorizes it.
