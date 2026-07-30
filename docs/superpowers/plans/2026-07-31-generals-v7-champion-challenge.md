# Generals v7 Champion Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one audited Generals v7 iteration whose frozen explainable policy learns from six fresh champion replays and earns a champion-beating claim only by passing the separate 20-game sealed dual-seat challenge.

**Architecture:** Add a frozen three-way challenge contract (learning, validation, sealed), a v7-specific lineage/prompt/pipeline layer, and explicit gate/claim records while preserving the historical formal benchmark as the only source of raw/evo/gain/AUC. The coding agent receives the exact v6 source plus learning-only replay evidence once, produces a deterministic bounded planner, and the Framework freezes that source before validation, formal, or sealed gameplay.

**Tech Stack:** Python 3.11, dataclasses, `tomllib`, pytest, AgentBenchFrame tracking/provider/snapshot APIs, the official Generals Python engine, Jinja report templates, and the non-interactive Codex CLI.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-31-generals-v7-champion-challenge-design.md`.
- Parent run: `/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_1653_af8eda26`.
- Expected v6 source SHA-256: `974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b`.
- Opponent in all three new partitions: `advanced-rank02-robinliu-v18`.
- Learning: seeds `290101`, `290202`, `290303`, both seats, six v6 games visible to the single v7 Codex act.
- Validation: seeds `291101`, `291202`, `291303`, `291404`, `291505`, `291606`, both seats, twelve frozen-v7 games never visible to the prompt.
- Validation gate: all twelve games valid, at least `7/12` overall, and at least `3/6` from each seat.
- Sealed challenge: seeds `292101`, `292202`, `292303`, `292404`, `292505`, `292606`, `292707`, `292808`, `292909`, `292999`, both seats, twenty games executed only after the validation gate passes.
- Champion claim: all twenty games valid, at least `11/20` overall, and at least `5/10` from each seat.
- Historical formal evaluation is always attempted for a runnable v7 and remains the sole input to benchmark score, raw/evo/gain, and AUC.
- Frozen engine SHA-256 is `4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97`.
- Frozen Replay Analysis v2 SHA-256 is `0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48`.
- Exactly one coding-agent act occurs in a normal v7 run. Validation, formal, and sealed payloads cannot enter that act or candidate selection.
- No runtime dependency is added. The official two-second decision timeout is unchanged.
- v7 source is saved even when it fails validation or the sealed claim. A failed sealed suite is retired and cannot tune another candidate evaluated on those seeds.
- The implementation is not the goal boundary. The goal is achieved only if the final frozen policy passes the sealed overall and per-seat thresholds.

---

### Task 1: Freeze the v7 champion challenge asset

**Files:**
- Create in the assets worktree: `backend_sources/corpus/28_generals/benchmark/v7-champion-challenge-v1.toml`
- Create in the assets worktree: `backend_sources/corpus/28_generals/tests/test_v7_champion_contract.py`
- Modify: `src/agentbench_frame/generals/models.py`
- Create: `src/agentbench_frame/generals/challenge_v7.py`
- Create: `tests/generals/fixtures/v7-champion-challenge-v1.toml`
- Create: `tests/generals/test_challenge_v7.py`

**Interfaces:**
- Produces `Round7ChallengeConfig`.
- Produces `load_round7_challenge_config(path, pilot, *, engine_hash, replay_skill_sha256)`.
- Exposes frozen seed tuples and threshold constants from one module.

- [ ] **Step 1: Write failing Framework contract tests**

```python
def test_round7_challenge_loads_exact_frozen_contract():
    challenge = load_round7_challenge_config(
        FIXTURE,
        PILOT,
        engine_hash=ENGINE_SHA256,
        replay_skill_sha256=SKILL_SHA256,
    )
    assert challenge.challenge_id == "generals-hl-v7-champion-v1"
    assert challenge.opponent_id == "advanced-rank02-robinliu-v18"
    assert challenge.learning_seeds == (290101, 290202, 290303)
    assert challenge.validation_seeds == (
        291101, 291202, 291303, 291404, 291505, 291606,
    )
    assert challenge.sealed_seeds == (
        292101, 292202, 292303, 292404, 292505,
        292606, 292707, 292808, 292909, 292999,
    )
    assert challenge.seats == (0, 1)
    assert challenge.validation_threshold == (7, 3)
    assert challenge.sealed_threshold == (11, 5)
```

Add rejection tests for a changed opponent, reordered seats, non-integer seeds,
duplicate or overlapping seeds, overlap with any frozen v0-v6 or policy-KL
seed, changed thresholds, and either digest mismatch.

- [ ] **Step 2: Run the focused test and observe the missing module**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_challenge_v7.py -q
```

Expected: collection fails because `challenge_v7` and
`Round7ChallengeConfig` do not exist.

- [ ] **Step 3: Add the immutable model and strict parser**

Add to `models.py`:

```python
@dataclass(frozen=True)
class Round7ChallengeConfig:
    challenge_id: str
    opponent_id: str
    learning_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    sealed_seeds: tuple[int, ...]
    seats: tuple[int, ...]
    validation_min_wins: int
    validation_min_wins_per_seat: int
    sealed_min_wins: int
    sealed_min_wins_per_seat: int
    engine_sha256: str
    replay_skill_sha256: str

    @property
    def validation_threshold(self) -> tuple[int, int]:
        return self.validation_min_wins, self.validation_min_wins_per_seat

    @property
    def sealed_threshold(self) -> tuple[int, int]:
        return self.sealed_min_wins, self.sealed_min_wins_per_seat
```

The parser must require exact tuple values, the pilot's high/first opponent,
the two exact SHA-256 values, and disjointness against every seed constant
already frozen in `assets.py`, including the controlled-policy-KL seeds.

- [ ] **Step 4: Write the identical repository and fixture manifests**

```toml
challenge_id = "generals-hl-v7-champion-v1"
opponent_id = "advanced-rank02-robinliu-v18"
learning_seeds = [290101, 290202, 290303]
validation_seeds = [291101, 291202, 291303, 291404, 291505, 291606]
sealed_seeds = [292101, 292202, 292303, 292404, 292505, 292606, 292707, 292808, 292909, 292999]
seats = [0, 1]
validation_min_wins = 7
validation_min_wins_per_seat = 3
sealed_min_wins = 11
sealed_min_wins_per_seat = 5
engine_sha256 = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
replay_skill_sha256 = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
```

- [ ] **Step 5: Add the assets-repository contract test**

The assets test must parse the TOML, assert every exact value above, read
`pilot-v1.toml` to confirm the opponent is the high entry, hash the replay
skill, and fail if any new seed appears elsewhere in the Generals benchmark
manifests.

- [ ] **Step 6: Run both repositories' focused tests**

Framework:

```bash
.venv/bin/python -m pytest tests/generals/test_challenge_v7.py tests/generals/test_assets.py -q
```

Assets:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest backend_sources/corpus/28_generals/tests/test_v7_champion_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the two repositories separately**

Framework commit: `feat(generals): freeze v7 champion challenge contract`

Assets commit: `assets(generals): freeze v7 champion challenge`

### Task 2: Build exact cases and all-or-nothing gates

**Files:**
- Modify: `src/agentbench_frame/generals/challenge_v7.py`
- Create: `tests/generals/test_challenge_v7_gates.py`

**Interfaces:**
- Produces `build_round7_learning_cases`, `build_round7_validation_cases`, and `build_round7_sealed_cases`.
- Produces `ChampionGate` and `evaluate_champion_gate`.

- [ ] **Step 1: Write failing case-order and gate tests**

```python
def test_round7_partitions_have_exact_order_and_size():
    learning = build_round7_learning_cases(PILOT, CHALLENGE)
    validation = build_round7_validation_cases(PILOT, CHALLENGE)
    sealed = build_round7_sealed_cases(PILOT, CHALLENGE)
    assert len(learning) == 6
    assert len(validation) == 12
    assert len(sealed) == 20
    assert [(case.seed, case.first_player) for case in learning] == [
        (seed, seat)
        for seed in CHALLENGE.learning_seeds
        for seat in (0, 1)
    ]
    assert {case.metadata["phase"] for case in sealed} == {"sealed7"}
```

```python
def test_sealed_gate_requires_overall_and_each_seat():
    gate = evaluate_champion_gate(
        results=valid_results(seat_0_wins=6, seat_1_wins=5, games_per_seat=10),
        cases=SEALED_CASES,
        minimum_wins=11,
        minimum_wins_per_seat=5,
    )
    assert gate.status == "passed"
    assert gate.passed is True
    assert gate.wins == 11
    assert gate.per_seat_wins == {0: 6, 1: 5}
```

Also test 11 wins with only four from one seat, 10 total wins, one invalid
result, one absent result, a duplicate case ID, and an unexpected case ID.
Incomplete or invalid suites must have `score=None`, `passed=False`, and a
status that distinguishes `incomplete` from `invalid`.

- [ ] **Step 2: Run the tests and verify missing interfaces**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_challenge_v7_gates.py -q
```

Expected: FAIL because the builders and gate evaluator are absent.

- [ ] **Step 3: Implement deterministic builders and gate records**

```python
@dataclass(frozen=True)
class ChampionGate:
    status: str
    passed: bool
    score: float | None
    wins: int
    losses: int
    draws: int
    per_seat_wins: Mapping[int, int]
    expected_games: int
    valid_games: int
    reasons: tuple[str, ...]
```

Builders must use only the configured champion, preserve seed-then-seat
ordering, and write phase names `learn7`, `validate7`, and `sealed7`.
`evaluate_champion_gate` must compare exact case-ID sets before counting wins.
It must not coerce missing, invalid, or drawn games into losses for an
aggregate score.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_challenge_v7.py tests/generals/test_challenge_v7_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat(generals): add champion case builders and sealed gates`

### Task 3: Verify and import the exact v6 parent

**Files:**
- Create: `src/agentbench_frame/generals/lineage_v7.py`
- Create: `tests/generals/test_lineage_v7.py`

**Interfaces:**
- Produces `Round7ParentLineage`.
- Produces `load_round7_parent(parent_run_dir, expected_parent_hash)`.
- Produces `import_round7_source(lineage, run_dir, snapshotter)`.

- [ ] **Step 1: Write failing lineage tests**

Build a synthetic complete v6 run with the exact v0-v6 score history, seven
global coding-agent acts, a complete historical formal evaluation, and a
hash-matched `versions/v6/source`.

```python
lineage = load_round7_parent(parent, manifest.content_hash)
assert lineage.parent_version == "v6"
assert lineage.global_act_count == 7
assert lineage.prior_score_history[-1] == pytest.approx(12 / 18)
assert lineage.v6_manifest.content_hash == manifest.content_hash
```

Add rejection tests for wrong expected hash, changed source bytes, incomplete
v6, incomplete formal evaluation, changed score history, absent learning
budget, and global act count other than seven.

- [ ] **Step 2: Run and observe the missing module**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_lineage_v7.py -q
```

Expected: collection fails because `lineage_v7` does not exist.

- [ ] **Step 3: Implement strict parent loading and isolated import**

Reuse the verified snapshot comparison pattern from `lineage_v6.py`, but
require `parent_version == "v6"` and copy source into both:

```text
<new run>/workspace/
<new run>/versions/v6/source/
```

Write `lineage.json` with parent run ID, parent version, expected and observed
hashes, inherited score history, inherited cumulative budget, and
`global_act_count=7`. Never read a mutable policy workspace from the parent.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_lineage_v7.py tests/generals/test_lineage_v6.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat(generals): verify v6 lineage for v7`

### Task 4: Build a learning-only v7 prompt with a full planner contract

**Files:**
- Create: `src/agentbench_frame/generals/prompt_v7.py`
- Create: `tests/generals/test_prompt_v7.py`

**Interfaces:**
- Produces `validate_round7_static_context`.
- Produces `build_round7_prompt`.
- Produces a prompt manifest containing only six declared learning episodes.

- [ ] **Step 1: Write failing prompt-boundary tests**

Use six synthetic complete `learn7` evidence records and assert:

```python
prompt = build_round7_prompt(
    v6_strategy=V6_STRATEGY,
    v6_experience=V6_EXPERIENCE,
    rules_text=RULES,
    replay_skill=SKILL,
    evidence=SIX_LEARNING_EPISODES,
    action_profile=LEARNING_ACTION_PROFILE,
    max_bytes=131_072,
)
assert prompt.manifest["episode_count"] == 6
assert prompt.manifest["episode_ids"] == EXPECTED_LEARNING_IDS
assert "commands 1 through 7" in prompt.text
assert "at most eight non-end primitives" in prompt.text
assert "main.py is immutable" in prompt.text
```

Add rejection tests when any input contains a validation, sealed, historical
formal, calibration, or controlled-KL case ID, phase label, or frozen seed.
Also reject champion source text, incomplete evidence, duplicate learning
episodes, a wrong replay-skill digest, and an oversized prompt.

- [ ] **Step 2: Run and verify missing interfaces**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_prompt_v7.py -q
```

Expected: collection fails because `prompt_v7` does not exist.

- [ ] **Step 3: Implement the bounded prompt**

The prompt must require:

- a private normalized-state clone and exact modeled transitions;
- focused deterministic generators for commands 1 through 7;
- v6 commands 1/3/5 as the explicit uncertainty fallback;
- a documented phase-aware state evaluator;
- fixed beam width, per-family top-k, exact-state deduplication, and no random tie breaking;
- at most eight non-end primitives and exactly one final `[8]`;
- official legality, deterministic replay-state, and latency tests;
- editable paths limited to `strategy.py`, `state_view.py`, `STRATEGY.md`,
  `EXPERIENCE.md`, `tests/**`, and `policy/**`;
- no network, seed, replay-ID, path, wall-clock, or opponent-identity runtime conditioning;
- a compressed update to the strategy and experience documents instead of
  an append-only rule pile.

The prompt manifest must include the prompt digest, replay-skill digest,
learning case IDs, evidence decision IDs, included bytes, omitted bytes, and
an explicit tuple of forbidden partition names.

- [ ] **Step 4: Run focused and prior prompt tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_prompt_v7.py tests/generals/test_prompt_v6.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat(generals): build leak-safe v7 planner prompt`

### Task 5: Implement pre-act learning, evidence, and one provider act

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v7.py`
- Create: `tests/generals/test_pipeline_v7.py`

**Interfaces:**
- Produces `Round7PipelineResult`.
- Produces `GeneralsHLRound7Pipeline.from_paths`.
- Produces `GeneralsHLRound7Pipeline.run`.

- [ ] **Step 1: Write a failing happy-path skeleton test**

Use the existing fake evaluator/provider/snapshot fixtures from
`test_pipeline_v6.py`. Record all calls and assert this prefix:

```python
assert calls[:3] == [
    ("evaluate", "v6", "learning", 6),
    ("provider", "v7", 1),
    ("freeze", "v7"),
]
assert result.round_act_count == 1
assert result.global_act_count == 8
```

Assert the run contains learning case specs, six official and normalized
replays, dense traces, replay-skill copy and digest, compact evidence,
learning action profile, exact prompt and prompt manifest, raw provider JSONL,
stderr, token/time receipt, feedback receipt, v6 source manifest, and a
post-act candidate workspace.

- [ ] **Step 2: Run the test and verify the missing pipeline**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -q
```

Expected: collection fails because `pipeline_v7` does not exist.

- [ ] **Step 3: Implement the v7 result and constructor**

```python
@dataclass(frozen=True)
class Round7PipelineResult:
    run_dir: Path
    status: str
    runnable: bool
    raw_score: float | None
    evo_score_7: float | None
    gain_7: float | None
    validation_passed: bool
    sealed_status: str
    champion_claim: bool
    global_act_count: int
    round_act_count: int
```

`from_paths` must load the pilot, resolve and validate assets, resolve the
Replay Skill, then load the challenge while checking its engine and skill
digests.

- [ ] **Step 4: Implement the pre-act workflow**

The run must:

1. verify/import v6;
2. write all three challenge specs and the unchanged formal spec before play;
3. evaluate exactly six v6 learning cases;
4. abort with zero acts if any learning case is incomplete or invalid;
5. normalize replays and build state-anchored critical evidence using Replay
   Analysis v2;
6. build the learning action profile and the bounded v7 prompt;
7. invoke the provider exactly once; and
8. retain all provider records even on failure.

Adapt shared behavior from v6 without changing v6 artifacts or semantics.
Keep phase budgets separate: `learning`, `validation`, `formal`, and `sealed`.

- [ ] **Step 5: Add pre-act failure tests**

Cover parent-hash failure, one invalid learning match, replay-normalization
failure, static-context failure, prompt leakage, provider timeout, provider
non-zero exit, and a provider receipt without an exact post-act source.
Each test must assert status, act count, preserved artifacts, and that no
post-act gameplay occurred.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -q
```

Expected: all pre-act tests PASS.

- [ ] **Step 7: Commit**

Commit: `feat(generals): orchestrate v7 learning and provider act`

### Task 6: Freeze only a legal deterministic v7 policy

**Files:**
- Modify: `src/agentbench_frame/generals/pipeline_v7.py`
- Modify: `tests/generals/test_pipeline_v7.py`

- [ ] **Step 1: Add failing source-scope and policy-verification tests**

Accept changes below `tests/**` and `policy/**` plus the four named editable
root files. Reject a modified `main.py`, every other root file, absolute or
escaping symlink, network marker, undeclared executable, and source larger
than the asset limit.

Policy verification fixtures must assert:

- exact determinism on repeated frozen-state probes;
- valid command shapes and exactly one final `[8]`;
- no more than eight non-end primitives;
- official movement budget and one-army source reserve;
- legal and illegal cases for commands 2, 4, 6, and 7;
- fallback to the v6-safe prefix on missing transition fields; and
- every fixed probe completes with explicit headroom below two seconds.

- [ ] **Step 2: Run and observe failures**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -k "scope or probe or freeze" -q
```

Expected: FAIL because the v7 verifier and expanded editable scope are not
implemented.

- [ ] **Step 3: Implement post-act verification**

Run candidate-owned tests in the isolated workspace, then Framework-owned
policy probes in separate worker processes. Capture stdout, stderr, duration,
exit code, probe state IDs, command output, and failure reason.

Only after every check passes:

1. capture the exact workspace manifest;
2. write `versions/v7/source/` and `versions/v7/manifest.json`;
3. re-capture the frozen version and compare file map plus content hash;
4. write the v6-to-v7 patch; and
5. make all post-act evaluators use `versions/v7/source/`, never `workspace/`.

- [ ] **Step 4: Add candidate policy-contract assertions**

The saved `STRATEGY.md` must declare beam width, family top-k values, maximum
macro length, evaluator weights by phase, deterministic tie order, and
fallback rules. The saved `EXPERIENCE.md` must summarize the six learning
episodes without copying replay IDs or seed values into runtime code.

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -k "scope or probe or freeze or policy_contract" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `feat(generals): freeze verified v7 planner candidates`

### Task 7: Add validation, formal, and conditional sealed execution

**Files:**
- Modify: `src/agentbench_frame/generals/pipeline_v7.py`
- Modify: `tests/generals/test_pipeline_v7.py`

- [ ] **Step 1: Write failing phase-order and threshold tests**

For a passing fake candidate, assert:

```python
assert evaluation_calls == [
    ("v6", "learning", 6),
    ("v7", "validation", 12),
    ("v7", "formal", 18),
    ("v7", "sealed", 20),
]
assert result.validation_passed is True
assert result.champion_claim is True
```

For a validation failure, assert formal still runs and sealed never runs:

```python
assert evaluation_calls == [
    ("v6", "learning", 6),
    ("v7", "validation", 12),
    ("v7", "formal", 18),
]
assert result.sealed_status == "not_opened"
assert result.champion_claim is False
```

Add exact boundary cases: validation `7/12` with `3/6` each seat passes;
`7/12` with one seat at `2/6` fails; sealed `11/20` with `5/10` each seat
passes; `11/20` with one seat at `4/10` fails.

- [ ] **Step 2: Run and observe missing orchestration**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -k "validation or formal or sealed or champion" -q
```

Expected: FAIL until phase order and gates are implemented.

- [ ] **Step 3: Implement post-freeze gameplay**

Run validation first, write `validation-gate.json`, then always attempt the
historical formal benchmark for a runnable v7. Open the sealed suite only if
the immutable validation gate passed. Write `sealed-claim.json` even when the
suite was not opened, using:

```json
{
  "status": "not_opened",
  "champion_claim": false,
  "reason": "validation_gate_failed"
}
```

On a complete sealed suite, write exact overall/per-seat counts, thresholds,
source hash, engine hash, suite digest, and `champion_claim`. On an invalid or
incomplete sealed suite, preserve individual results but set aggregate score
to null and the claim to false.

- [ ] **Step 4: Keep formal research metrics independent**

`benchmark_score`, `evo_score_7`, `gain_7`, `score_history`, and AUC must use
only the 18 historical formal results. Store champion fields separately:

```python
"champion_validation": validation_gate_payload,
"champion_sealed": sealed_claim_payload,
"champion_claim": sealed_claim.champion_claim,
```

Never append validation or sealed results to the formal benchmark result
array.

- [ ] **Step 5: Add artifact, budget, and quality assertions**

Assert phase-specific action profiles, replay/dense artifacts, elapsed time,
episode and environment-step counts, formal/validation/sealed missingness,
event-quality diagnostics, source lineage, and final summary consistency.
No missing value may be converted to zero.

- [ ] **Step 6: Run the full v7 pipeline test**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Commit: `feat(generals): gate and audit v7 champion challenge`

### Task 8: Implement hash-safe recovery

**Files:**
- Modify: `src/agentbench_frame/generals/pipeline_v7.py`
- Create: `tests/generals/test_pipeline_v7_recovery.py`

- [ ] **Step 1: Write failing recovery tests**

Cover these exact modes:

1. retry provider from an immutable complete learning/prompt receipt;
2. resume verification from an exact provider-produced candidate;
3. resume validation from an exact frozen v7;
4. resume formal after validation was already recorded;
5. resume sealed only when the inherited validation gate passed; and
6. refuse reuse after any prompt, learning evidence, source, manifest, gate,
   engine, skill, case spec, or result hash changes.

Every recovery creates a new run and records the failed source run ID and
reused artifact hashes. It never edits the failed run.

- [ ] **Step 2: Run and observe missing recovery**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7_recovery.py -q
```

Expected: FAIL because `recover` is absent or incomplete.

- [ ] **Step 3: Implement recovery with the narrowest reusable boundary**

Expose:

```python
def recover(self, failed_run_dir: Path) -> Round7PipelineResult:
    ...
```

Reuse a coding-agent act only when the exact frozen post-act v7 source exists.
Otherwise, reuse at most the verified learning/prompt receipt and perform a
new visible act. Never reuse partial validation or sealed aggregates; reuse
only individually complete, hash-matched match artifacts.

- [ ] **Step 4: Run recovery and main pipeline tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_pipeline_v7.py tests/generals/test_pipeline_v7_recovery.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat(generals): recover v7 without reopening evidence`

### Task 9: Expose v7 in the CLI

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Create: `tests/generals/test_cli_v7.py`

- [ ] **Step 1: Write failing CLI tests**

Assert both `iterate-v7` and `recover-v7` require:

- `--challenge-manifest`
- `--replay-skill`
- `--parent-run`
- `--expected-parent-hash`
- `--codex-executable`
- `--provider-timeout`

Assert recovery additionally requires `--failed-run`. Mock the pipeline and
verify JSON output contains `evo_score_7`, `gain_7`, `validation_passed`,
`sealed_status`, `champion_claim`, both act counts, and the run path.

- [ ] **Step 2: Run and observe parser failure**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_cli_v7.py -q
```

Expected: FAIL because the commands are unknown.

- [ ] **Step 3: Register and dispatch the commands**

Import `GeneralsHLRound7Pipeline`, create explicit-provider arguments, call
`run()` or `recover(args.failed_run)`, print sorted JSON, and return zero only
for a structurally complete run. A complete run that validly fails the
champion threshold still returns zero; `champion_claim` conveys scientific
success separately.

- [ ] **Step 4: Run all Generals CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_cli_v7.py tests/generals/test_cli_v6.py tests/generals/test_cli_v5.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat(generals): expose v7 champion workflow`

### Task 10: Preserve v6/v7 scores and show champion evidence in CI reports

**Files:**
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Modify: `src/agentbench_frame/report/templates/agent.html`
- Modify: `tests/test_local_report_research.py`

- [ ] **Step 1: Write failing report tests**

Feed a v7 summary whose historical formal score differs from its validation
and sealed scores. Assert:

```python
assert research["benchmark_score"] == summary["evo_score_7"]
assert research["gain"] == summary["gain_7"]
assert research["champion"]["validation"]["score"] == 7 / 12
assert research["champion"]["sealed"]["score"] == 11 / 20
assert research["champion"]["claim"] is True
```

Also test a validation failure where sealed score is `None` and status is
`not_opened`, plus an incomplete sealed suite where the HTML renders
`missing/invalid` rather than `0%`.

- [ ] **Step 2: Run and observe stale latest-score selection**

Run:

```bash
.venv/bin/python -m pytest tests/test_local_report_research.py -k "champion or latest_generals" -q
```

Expected: FAIL because the builder currently stops at earlier numbered
`evo_score` and `gain` keys and has no champion projection.

- [ ] **Step 3: Implement version-generic latest-score selection**

Replace fixed nested lookups with a numeric-key selector:

```python
def _latest_numbered(summary, prefix):
    candidates = (
        (int(key.removeprefix(prefix)), value)
        for key, value in summary.items()
        if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
    )
    return max(candidates, default=(0, None))[1]
```

Use it for `evo_score_1` through `evo_score_7` and `gain_2` through `gain_7`,
while retaining legacy unnumbered fallbacks.

- [ ] **Step 4: Add a separate champion panel**

Render validation and sealed wins/games, each seat's wins, gate/claim status,
source hash, and a clear sentence that these metrics do not enter the formal
benchmark score. Preserve missing values exactly.

- [ ] **Step 5: Run report tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_local_report_research.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `feat(report): render Generals champion evidence`

### Task 11: Run repository verification and review the implementation

**Files:**
- Review every file changed in Tasks 1-10.

- [ ] **Step 1: Run focused Generals tests**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/generals -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete Framework suite**

```bash
AGENTBENCH_ASSET_ROOT=/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q
```

Expected: PASS.

- [ ] **Step 3: Run Generals assets tests**

From `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets`:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest backend_sources/corpus/28_generals/tests -q
```

Expected: PASS.

- [ ] **Step 4: Perform structural checks**

```bash
git diff --check
rg -n 'TO''DO|TB''D|FIX''ME|PLACE''HOLDER' src/agentbench_frame/generals/pipeline_v7.py src/agentbench_frame/generals/challenge_v7.py src/agentbench_frame/generals/prompt_v7.py tests/generals/test_pipeline_v7.py
```

Expected: `git diff --check` succeeds and the marker scan returns no matches.

- [ ] **Step 5: Inspect scientific boundaries**

Read the final prompt manifest, v7 pipeline summary construction, and report
projection side by side. Confirm:

- only six `learn7` episodes enter the prompt;
- validation/sealed case IDs and results never enter provider input;
- the source is frozen before validation;
- formal runs whether validation passes or fails;
- sealed runs only when validation passes;
- formal and champion aggregates are separate; and
- no invalid/incomplete phase becomes zero.

- [ ] **Step 6: Review for requirements and code quality**

Use `superpowers:requesting-code-review`. Address every high- or
medium-severity finding, rerun affected tests, then rerun Steps 1-3.

- [ ] **Step 7: Commit review corrections**

Commit: `fix(generals): harden v7 champion workflow`

### Task 12: Execute the real v7 experiment and adjudicate the claim

**Files produced under the run directory:**
- `benchmark/*.json`
- `matches/v6/learn7-*/**`
- `matches/v7/validate7-*/**`
- `matches/v7/eval-*/**`
- conditionally `matches/v7/sealed7-*/**`
- `provider/**`
- `versions/v6/**`
- `versions/v7/**`
- `validation-gate.json`
- `sealed-claim.json`
- `summary.json`
- `events.jsonl`

- [ ] **Step 1: Record clean code and asset heads**

```bash
git status --short
git rev-parse HEAD
git -C /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets status --short
git -C /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets rev-parse HEAD
```

Expected: both worktrees are clean and the two commit IDs are recorded in the
experiment note.

- [ ] **Step 2: Run the real iteration**

From the Framework worktree:

```bash
.venv/bin/python -m agentbench_frame.cli generals iterate-v7 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --challenge-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v7-champion-challenge-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260729_1653_af8eda26 \
  --expected-parent-hash 974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b \
  --codex-executable codex \
  --provider-timeout 1800
```

Expected: one new auditable run. It may validly finish without opening the
sealed suite if validation fails.

- [ ] **Step 3: Validate the real artifact graph**

Run the quality inspector and a dedicated artifact checker against the
reported run directory. Assert exact case counts, phase ordering, one act,
source hash stability, provider receipt completeness, replay/dense/action
profile counts, phase budgets, no unknown or malformed events, and exact
agreement between match results, gates, summary, and report projection.

- [ ] **Step 4: Render and inspect the CI report**

```bash
.venv/bin/python -m agentbench_frame.cli report \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data \
  --output-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_report
```

Open the generated report files locally and verify the historical formal
curve and champion panel show separate numbers and explicit missingness.

- [ ] **Step 5: Adjudicate without changing thresholds**

The v7 outcome is one of:

- `champion_claim=true`: all 20 sealed games are valid, wins are at least
  11/20, and each seat has at least 5/10; mark the active goal complete and
  report the exact run, source hash, counts, formal score, budgets, and logs.
- validation failed: preserve v7, report formal results, leave sealed unopened,
  and design v8 with entirely fresh learning/validation/sealed seeds.
- validation passed but sealed claim failed: retire the sealed v7 suite,
  preserve all results, and design v8 with entirely fresh splits; never use
  v7 sealed outcomes to select a candidate evaluated on the same suite.
- invalid/incomplete run: recover only through the hash-safe paths in Task 8,
  preserving the failed attempt and all receipts.

- [ ] **Step 6: Commit only durable code or documentation**

Do not commit generated run data unless the repository policy explicitly
tracks it. Commit the final experiment note and any reproducibility commands
with: `docs(generals): record v7 champion challenge result`.
