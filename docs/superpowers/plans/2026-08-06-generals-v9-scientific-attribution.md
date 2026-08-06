# Generals v9 Scientific Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a paired scientific attribution of the v8 regression, perform exactly one leak-free Codex act from frozen v7 to create v9, evaluate it unconditionally, and measure v0-v9 policy change on separately reported legacy-12 and expanded-24 exact-support domains.

**Architecture:** Add an append-only attribution run that materializes four auditable policy cells and compares them on identical strongest-human games plus same-state replay probes. Feed only the frozen attribution report and diagnosis evidence to a new single-act v9 pipeline whose source parent is v7 while its chronological predecessor is v8. Add a deterministic 12-state intervention pack and a new exact-KL run that hash-reuses legacy evidence but never mixes the legacy and expanded reference domains.

**Tech Stack:** Python 3.11, pytest, dataclasses, TOML/JSON/JSONL contracts, official Generals Python SDK, AgentBench `Run` event tracking, Codex non-interactive provider, exact canonical macro-action enumeration, Matplotlib, Jinja CI reports, Git linked worktrees.

## Global Constraints

- Do not push, create a pull request, merge, delete history, or modify any existing v0-v8 run.
- Frozen v7 source run: `20260730_1739_680b1632`, content hash `c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
- Frozen v8 diagnostic run: `20260806_1126_8e1b6795`, content hash `3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a`.
- Frozen legacy-12 KL run: `20260806_1129_6085e1a6`, tree hash `d1dcb9f058d2b09a2e6e66638bf30ea5a591c1fd62d345a4ffcf8b2076b32379`.
- v9 has `policy_parent_version=v7` and `iteration_predecessor_version=v8`; these two lineage roles must never be conflated.
- Attribution seeds are exactly `302101, 302202, 302303, 302404, 302505, 302606`; validation seeds are exactly `303101, 303202, 303303, 303404, 303505, 303606`; both use seats `0, 1`.
- Formal evaluation remains the unchanged 18-case `pilot-v1.toml` matrix and is never visible to the provider.
- Invoke Codex exactly once for v9. An explicit recovery may resume evaluation of an already frozen candidate but may never invoke the provider again.
- Every runnable v9 receives all 12 validation and all 18 formal games, regardless of validation performance.
- Champion promotion requires validation `>=2/12` with one win per seat, formal `>=13/18`, high tier `>=2/6` with one high-tier win per seat, and all 18 formal games valid.
- Attribution ablations are population/testcase policies, not HL versions and not coding-agent acts.
- Replay diagnostics contain at most 48 selected states and compare policies only on an identical reconstructed state.
- Exact KL uses uniform `U_s`, natural logarithms, and epsilons `0.001`, `0.01`, `0.05`, `0.1`; primary epsilon is `0.01`.
- Legacy-12 and expanded-24 are separate named domains and must never be joined into one apparent continuous measurement curve.
- Any non-exact support makes expanded-24 incomplete; do not sample, truncate, bound, impute, or interpolate it.
- Missing provider usage is `null`/unknown, never zero. Missing scores or diagnostics remain missing.
- Hand-authored edits use `apply_patch`; generated state packs and figures use their versioned generators.

## File and responsibility map

| File | Responsibility |
|---|---|
| `generals/models.py` | Immutable attribution, v9 challenge, intervention-state, and expanded-KL records |
| `generals/challenge_v9.py` | Exact seed/threshold/role contract and case construction |
| `generals/lineage_v9.py` | Verify dual v7 source-parent and v8 chronological-predecessor lineage |
| `generals/ablation_v9.py` | Materialize A/B/C/D policy trees from exact v7/v8 authorities |
| `generals/attribution.py` | Paired 2x2 effects and deterministic bootstrap summaries |
| `generals/attribution_diagnostics.py` | Critical-state selection, same-state probes, and earliest divergence |
| `generals/attribution_pipeline.py` | Append-only attribution run and diagnosis artifacts |
| `generals/prompt_v9.py` | Leak-safe diagnosis-to-v9 prompt and input receipt |
| `generals/pipeline_v9.py` | Single Codex act, candidate freeze, unconditional evaluation, champion decision |
| `generals/intervention_states.py` | Deterministic official-state recipes and scenario assertions |
| `generals/policy_kl_expanded.py` | Hash-reused legacy-12 plus newly computed intervention-12 exact KL |
| `generals/paper_figure_v9.py` | English attribution, score/IG, and dual-domain paper figures |
| `generals/cli.py` | Attribution, v9, recovery, expanded-KL, and figure command surfaces |
| `tracking/quality.py` | First-class event types for attribution and expanded measurement |
| `report/builder.py`, `report/templates/index.html` | CI projection without interpolation |

The subsystems remain in one plan because the attribution artifact is a frozen
input to the v9 act, and the resulting v9 source is a frozen input to expanded
KL. Each task still ends in an independently reviewable commit.

---

### Task 1: Freeze attribution and expanded-domain asset contracts

**Files:**
- Create: `backend_sources/corpus/28_generals/benchmark/v9-scientific-attribution-v1.toml`
- Create: `backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.toml`
- Create: `backend_sources/corpus/28_generals/tests/test_v9_scientific_attribution_contract.py`
- Create: `backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py`

**Interfaces:**
- Consumes: frozen `pilot-v1.toml`, engine hash, replay Skill hash, v7/v8/KL authority IDs and hashes
- Produces: exact manifests consumed by `load_round9_challenge_config()` and `load_expanded_kl_config()`

- [ ] **Step 1: Write failing asset contract tests**

Add exact assertions:

```python
def test_v9_attribution_contract_is_exact_and_disjoint():
    raw = tomllib.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    assert raw["challenge_id"] == "generals-hl-v9-scientific-attribution-v1"
    assert raw["opponent_id"] == "advanced-rank02-robinliu-v18"
    assert raw["attribution_seeds"] == [
        302101, 302202, 302303, 302404, 302505, 302606,
    ]
    assert raw["validation_seeds"] == [
        303101, 303202, 303303, 303404, 303505, 303606,
    ]
    assert raw["seats"] == [0, 1]
    assert raw["diagnostic_max_states"] == 48
    assert raw["validation_min_wins"] == 2
    assert raw["validation_min_wins_per_seat"] == 1
    assert raw["formal_total_min_wins"] == 13
    assert raw["formal_high_min_wins"] == 2
    assert raw["formal_high_min_wins_per_seat"] == 1
    assert set(raw["attribution_seeds"]).isdisjoint(raw["validation_seeds"])

def test_expanded_kl_contract_declares_six_scenarios_twice():
    raw = tomllib.loads(EXPANDED.read_text(encoding="utf-8"))
    assert raw["measurement_id"] == "generals-policy-kl-expanded-v1"
    assert raw["source_run_id"] == "20260806_1129_6085e1a6"
    assert raw["source_tree_hash"] == LEGACY_TREE_HASH
    states = raw["intervention_state"]
    assert len(states) == 12
    assert Counter(item["scenario"] for item in states) == {
        "contact": 2, "main_general_danger": 2,
        "large_stack_routing": 2, "economy_combat_conflict": 2,
        "counter_capture": 2, "mid_late_consolidation": 2,
    }
    assert Counter(item["actor"] for item in states) == {0: 6, 1: 6}
```

Also scan every Generals benchmark TOML and assert each attribution/validation
seed occurs only in the v9 manifest.

- [ ] **Step 2: Run the new asset tests and verify missing-file failures**

Run:

```bash
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
pytest backend_sources/corpus/28_generals/tests/test_v9_scientific_attribution_contract.py backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py -q
```

Expected: FAIL because both manifests are absent.

- [ ] **Step 3: Add exact frozen TOML manifests**

The attribution manifest contains these immutable values:

```toml
challenge_id = "generals-hl-v9-scientific-attribution-v1"
opponent_id = "advanced-rank02-robinliu-v18"
attribution_seeds = [302101, 302202, 302303, 302404, 302505, 302606]
validation_seeds = [303101, 303202, 303303, 303404, 303505, 303606]
seats = [0, 1]
diagnostic_max_states = 48
validation_min_wins = 2
validation_min_wins_per_seat = 1
formal_total_min_wins = 13
formal_high_min_wins = 2
formal_high_min_wins_per_seat = 1
parent_run_id = "20260730_1739_680b1632"
parent_content_hash = "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
predecessor_run_id = "20260806_1126_8e1b6795"
predecessor_content_hash = "3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a"
engine_sha256 = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
replay_skill_sha256 = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
```

The expanded manifest is exactly:

```toml
measurement_id = "generals-policy-kl-expanded-v1"
source_measurement_id = "generals-policy-kl-reference-v3"
source_run_id = "20260806_1129_6085e1a6"
source_tree_hash = "d1dcb9f058d2b09a2e6e66638bf30ea5a591c1fd62d345a4ffcf8b2076b32379"
state_pack = "policy-kl-expanded-v1.states.json"
epsilons = ["0.001", "0.01", "0.05", "0.1"]
primary_epsilon = "0.01"

[[intervention_state]]
state_key = "contact-p0-a"
scenario = "contact"
actor = 0
variant = 0

[[intervention_state]]
state_key = "contact-p1-b"
scenario = "contact"
actor = 1
variant = 1

[[intervention_state]]
state_key = "main-danger-p0-a"
scenario = "main_general_danger"
actor = 0
variant = 0

[[intervention_state]]
state_key = "main-danger-p1-b"
scenario = "main_general_danger"
actor = 1
variant = 1

[[intervention_state]]
state_key = "large-stack-p0-a"
scenario = "large_stack_routing"
actor = 0
variant = 0

[[intervention_state]]
state_key = "large-stack-p1-b"
scenario = "large_stack_routing"
actor = 1
variant = 1

[[intervention_state]]
state_key = "economy-conflict-p0-a"
scenario = "economy_combat_conflict"
actor = 0
variant = 0

[[intervention_state]]
state_key = "economy-conflict-p1-b"
scenario = "economy_combat_conflict"
actor = 1
variant = 1

[[intervention_state]]
state_key = "counter-capture-p0-a"
scenario = "counter_capture"
actor = 0
variant = 0

[[intervention_state]]
state_key = "counter-capture-p1-b"
scenario = "counter_capture"
actor = 1
variant = 1

[[intervention_state]]
state_key = "consolidation-p0-a"
scenario = "mid_late_consolidation"
actor = 0
variant = 0

[[intervention_state]]
state_key = "consolidation-p1-b"
scenario = "mid_late_consolidation"
actor = 1
variant = 1
```

- [ ] **Step 4: Run all Generals asset contracts**

Run:

```bash
pytest backend_sources/corpus/28_generals/tests -q
```

Expected: all asset contract tests pass.

- [ ] **Step 5: Commit the frozen assets**

```bash
git add backend_sources/corpus/28_generals/benchmark/v9-scientific-attribution-v1.toml backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.toml backend_sources/corpus/28_generals/tests/test_v9_scientific_attribution_contract.py backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py
git commit -m "assets(generals): freeze v9 attribution contracts"
```

### Task 2: Add strict v9 contracts and dual lineage

**Files:**
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Create: `src/agentbench_frame/generals/challenge_v9.py`
- Create: `src/agentbench_frame/generals/lineage_v9.py`
- Create: `tests/generals/fixtures/v9-scientific-attribution-v1.toml`
- Create: `tests/generals/fixtures/policy-kl-expanded-v1.toml`
- Create: `tests/generals/test_challenge_v9.py`
- Create: `tests/generals/test_lineage_v9.py`

**Interfaces:**
- Consumes: the two Task 1 manifests plus v7/v8 summaries and source manifests
- Produces: `Round9ChallengeConfig`, `ExpandedPolicyKLConfig`, `Round9Lineage`, `load_round9_challenge_config()`, `load_expanded_kl_config()`, `build_round9_attribution_cases()`, `build_round9_validation_cases()`, `load_round9_lineage()`, `import_round9_parent()`

- [ ] **Step 1: Write failing contract and lineage tests**

Use these required identities:

```python
def test_round9_has_distinct_source_parent_and_predecessor(tmp_path):
    v7 = make_v7_authority(tmp_path)
    v8 = make_v8_authority(tmp_path)
    lineage = load_round9_lineage(v7.run_dir, v8.run_dir)
    assert lineage.policy_parent_version == "v7"
    assert lineage.policy_parent_hash == V7_HASH
    assert lineage.iteration_predecessor_version == "v8"
    assert lineage.iteration_predecessor_hash == V8_HASH
    assert lineage.next_global_act_count == 10

def test_round9_cases_are_seed_major_and_dual_seat():
    cases = build_round9_attribution_cases(pilot, challenge)
    assert [(item.seed, item.first_player) for item in cases] == [
        (seed, seat) for seed in ATTRIBUTION_SEEDS for seat in (0, 1)
    ]
    assert all(item.metadata["pair_id"] == f"s{item.seed}-p{item.first_player}" for item in cases)
```

Mutation tests reject bool-as-int fields, seed overlap, altered order, any old
seed, wrong opponent/tier, wrong engine/Skill hash, wrong v7/v8 run or content
hash, incomplete authority summaries, non-runnable versions, and altered
source manifests.

- [ ] **Step 2: Run focused tests and verify import failures**

Run:

```bash
.venv/bin/pytest tests/generals/test_challenge_v9.py tests/generals/test_lineage_v9.py -q
```

Expected: FAIL because the v9 modules and records do not exist.

- [ ] **Step 3: Implement immutable records and exact loaders**

Add these central shapes:

```python
@dataclass(frozen=True)
class Round9ChallengeConfig:
    challenge_id: str
    opponent_id: str
    attribution_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    seats: tuple[int, ...]
    diagnostic_max_states: int
    validation_min_wins: int
    validation_min_wins_per_seat: int
    formal_total_min_wins: int
    formal_high_min_wins: int
    formal_high_min_wins_per_seat: int
    parent_run_id: str
    parent_content_hash: str
    predecessor_run_id: str
    predecessor_content_hash: str
    engine_sha256: str
    replay_skill_sha256: str

@dataclass(frozen=True)
class InterventionStateSpec:
    state_key: str
    scenario: str
    actor: int
    variant: int

@dataclass(frozen=True)
class ExpandedPolicyKLConfig:
    measurement_id: str
    source_measurement_id: str
    source_run_id: str
    source_tree_hash: str
    state_pack: Path
    epsilons: tuple[str, ...]
    primary_epsilon: str
    intervention_states: tuple[InterventionStateSpec, ...]

@dataclass(frozen=True)
class Round9Lineage:
    policy_parent_run_id: str
    policy_parent_version: str
    policy_parent_hash: str
    iteration_predecessor_run_id: str
    iteration_predecessor_version: str
    iteration_predecessor_hash: str
    next_global_act_count: int
    score_history: tuple[float | None, ...]
```

Parse TOML without coercing booleans or floats, compare every frozen constant,
and check every seed role against all historical manifests. Build exactly 12
`attribute9` and 12 `validate9` cases with stable `pair_id` metadata.

- [ ] **Step 4: Implement verified dual lineage and v7 materialization**

`load_round9_lineage(v7_run_dir, v8_run_dir)` verifies both summaries, their
run-directory names, source manifests, actual source tree hashes, completed
formal evaluations, and recorded chronology. `import_round9_parent()` copies
only v7 into `workspace/` and `versions/v7/source/`; it writes:

```json
{
  "starting_version": "v7",
  "policy_parent_version": "v7",
  "iteration_predecessor_version": "v8",
  "next_version": "v9"
}
```

It never copies v8 into the editable workspace.

- [ ] **Step 5: Run focused and v7/v8 regression tests**

Run:

```bash
.venv/bin/pytest tests/generals/test_challenge_v9.py tests/generals/test_lineage_v9.py tests/generals/test_challenge_v8.py tests/generals/test_lineage_v8.py tests/generals/test_challenge_v7.py tests/generals/test_lineage_v7.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/models.py src/agentbench_frame/generals/assets.py src/agentbench_frame/generals/challenge_v9.py src/agentbench_frame/generals/lineage_v9.py tests/generals/fixtures/v9-scientific-attribution-v1.toml tests/generals/fixtures/policy-kl-expanded-v1.toml tests/generals/test_challenge_v9.py tests/generals/test_lineage_v9.py
git commit -m "feat(generals): define v9 attribution lineage"
```

### Task 3: Materialize the four attribution policies reproducibly

**Files:**
- Create: `src/agentbench_frame/generals/ablation_v9.py`
- Create: `tests/generals/test_ablation_v9.py`

**Interfaces:**
- Consumes: verified v7 and v8 `HistoricalPolicySource` records
- Produces: `materialize_attribution_policies(v7, v8, destination) -> tuple[AttributionPolicy, ...]` for cells A/B/C/D

- [ ] **Step 1: Write failing source-materialization tests**

```python
def test_ablation_cells_have_exact_interventions(tmp_path):
    policies = materialize_attribution_policies(v7, v8, tmp_path)
    assert [item.cell for item in policies] == ["A", "B", "C", "D"]
    by_cell = {item.cell: item for item in policies}
    sources = {cell: (item.source / "strategy.py").read_text() for cell, item in by_cell.items()}
    assert "LARGE_STACK = 24" not in sources["A"]
    assert "LARGE_STACK = 24" in sources["B"]
    assert "def _contact_exists" not in sources["B"]
    assert "def _contact_exists" in sources["C"]
    assert "LARGE_STACK = 24" not in sources["C"]
    assert by_cell["D"].content_hash == V8_HASH

def test_materialization_rejects_changed_v7_preimage(tmp_path):
    changed = mutate_file(v7, "strategy.py", "# mutation\n")
    with pytest.raises(ValueError, match="v7 preimage"):
        materialize_attribution_policies(changed, v8, tmp_path)
```

Also assert only `strategy.py` and the generated `ABLATION.md` differ from v7
for B/C, both generated policies pass the frozen entrypoint probe twice, and D
is byte-identical to frozen v8.

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run: `.venv/bin/pytest tests/generals/test_ablation_v9.py -q`

Expected: FAIL because `ablation_v9.py` is absent.

- [ ] **Step 3: Implement exact-preimage transformations**

Define:

```python
@dataclass(frozen=True)
class AttributionPolicy:
    cell: str
    policy_id: str
    source: Path
    content_hash: str
    interventions: tuple[str, ...]
    source_authority: str

materialize_attribution_policies(
    v7: HistoricalPolicySource,
    v8: HistoricalPolicySource,
    destination: Path,
) -> tuple[AttributionPolicy, ...]
```

The B transformation adds only the v8 `LARGE_STACK = 24`, `force_priority`, and
score-tuple term to the exact v7 `strategy.py`. The C transformation adds only
the v8 `_contact_exists()` helper and its `_upgrade_candidate()` guard. Every
replacement requires an exact unique v7 preimage. A is an exact v7 copy; D is
an exact v8 copy. Capture every output with `LocalWorkspaceSnapshotter`, write
the transformation receipt, and reject any unexpected changed file.

- [ ] **Step 4: Run focused policy and historical-probe tests**

Run:

```bash
.venv/bin/pytest tests/generals/test_ablation_v9.py tests/generals/test_historical_policy.py tests/generals/test_pipeline_v8.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/generals/ablation_v9.py tests/generals/test_ablation_v9.py
git commit -m "feat(generals): materialize v8 attribution cells"
```

### Task 4: Implement paired attribution math and replay diagnostics

**Files:**
- Create: `src/agentbench_frame/generals/attribution.py`
- Create: `src/agentbench_frame/generals/attribution_diagnostics.py`
- Create: `tests/generals/test_attribution.py`
- Create: `tests/generals/test_attribution_diagnostics.py`

**Interfaces:**
- Consumes: four `GeneralsEvaluation` values with the same 12 `pair_id` values and v7/v8 `MatchResult` trajectories
- Produces: `compute_paired_attribution() -> AttributionReport`, `select_diagnostic_states() -> tuple[DiagnosticState, ...]`, `compare_same_state_actions() -> tuple[DiagnosticProbe, ...]`

- [ ] **Step 1: Write failing factorial and missingness tests**

```python
def test_factorial_effects_use_paired_case_values():
    cells = {
        "A": {"s1-p0": 0.0, "s2-p1": 1.0},
        "B": {"s1-p0": 1.0, "s2-p1": 1.0},
        "C": {"s1-p0": 0.0, "s2-p1": 0.0},
        "D": {"s1-p0": 1.0, "s2-p1": 0.0},
    }
    report = compute_paired_attribution({"win": cells}, bootstrap_seed=9049)
    effect = report.metrics["win"]
    assert effect.large_stack == pytest.approx(0.5)
    assert effect.contact == pytest.approx(-0.5)
    assert effect.interaction == pytest.approx(0.0)

def test_missing_case_stays_missing_instead_of_becoming_zero():
    cells = complete_cells()
    cells["B"]["s1-p0"] = None
    effect = compute_paired_attribution({"army": cells}).metrics["army"]
    assert effect.complete_pair_count == 1
    assert effect.case_effects["s1-p0"].large_stack is None
```

Bootstrap tests require identical intervals for repeated `bootstrap_seed=9049`
and reject a non-identical case set across cells.

- [ ] **Step 2: Write failing diagnostic-selection tests**

Create fixture trajectories containing all six event classes and assert:

```python
states = select_diagnostic_states(v7_matches, v8_matches, max_states=48)
assert len(states) <= 48
assert all(sum(item.pair_id == pair for item in states) <= 4 for pair in PAIRS)
assert [item.reason for item in states[:2]] == [
    "first_main_general_danger", "first_enemy_contact",
]
assert len({item.measurement_state_id for item in states}) == len(states)
```

Verify missing classes are emitted in `missing_reasons`, selection tie-breaking
is `(priority, round_number, actor, measurement_state_id)`, and probes compare
all four policies on the exact same measurement-state hash.

- [ ] **Step 3: Run tests and verify missing modules**

Run:

```bash
.venv/bin/pytest tests/generals/test_attribution.py tests/generals/test_attribution_diagnostics.py -q
```

Expected: FAIL on missing imports.

- [ ] **Step 4: Implement typed attribution summaries**

Use these public records:

```python
@dataclass(frozen=True)
class FactorialCaseEffect:
    pair_id: str
    large_stack: float | None
    contact: float | None
    interaction: float | None

@dataclass(frozen=True)
class FactorialMetricEffect:
    metric: str
    large_stack: float | None
    contact: float | None
    interaction: float | None
    complete_pair_count: int
    intervals: Mapping[str, tuple[float, float] | None]
    case_effects: Mapping[str, FactorialCaseEffect]

@dataclass(frozen=True)
class AttributionReport:
    pair_ids: tuple[str, ...]
    metrics: Mapping[str, FactorialMetricEffect]
    bootstrap_seed: int
    bootstrap_replicates: int

@dataclass(frozen=True)
class DiagnosticState:
    pair_id: str
    source_version: str
    reason: str
    round_number: int
    actor: int
    measurement_state_id: str
    measurement_state: Mapping[str, Any]
    replay_ref: str

@dataclass(frozen=True)
class DiagnosticProbe:
    measurement_state_id: str
    policy_cell: str
    status: str
    deterministic: bool
    canonical_action: tuple[tuple[int, ...], ...] | None
    legal: bool | None
    latency_s: float
    error: str | None

compare_same_state_actions(
    states: Sequence[DiagnosticState],
    policies: Sequence[AttributionPolicy],
    *,
    engine_root: Path,
    sdk_root: Path,
) -> tuple[DiagnosticProbe, ...]
```

For every complete pair calculate `B-A`, `C-A`, and `D-B-C+A`; aggregate by
the arithmetic mean. Run 10,000 paired resamples with local
`random.Random(9049)` and percentile indices 250 and 9749 after sorting. Never
convert missing values to zero and never emit a p-value.

- [ ] **Step 5: Implement deterministic state selection and same-state probes**

Define `DiagnosticState` with `pair_id`, source version, selection reason,
round, actor, measurement snapshot, ID, and source replay reference. Select at
most two v7 and two v8 states per pair by the frozen priority list. Probe each
`HistoricalPolicySource` using
`probe_historical_policy(policy, state.measurement_state,
engine_root=engine_root, sdk_root=sdk_root, repeats=2)`;
persist raw and canonical actions, legality, determinism, latency, stdout,
stderr, and error. `earliest_divergence()` returns the first shared decision
coordinate whose canonical actions differ, otherwise explicit `None`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
.venv/bin/pytest tests/generals/test_attribution.py tests/generals/test_attribution_diagnostics.py tests/generals/test_dense.py tests/generals/test_measurement.py -q
```

Expected: PASS.

```bash
git add src/agentbench_frame/generals/attribution.py src/agentbench_frame/generals/attribution_diagnostics.py tests/generals/test_attribution.py tests/generals/test_attribution_diagnostics.py
git commit -m "feat(generals): compute paired policy attribution"
```

### Task 5: Build the append-only scientific-attribution pipeline

**Files:**
- Create: `src/agentbench_frame/generals/attribution_pipeline.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Create: `tests/generals/test_attribution_pipeline.py`
- Modify: `tests/test_research_boundaries.py`

**Interfaces:**
- Consumes: Task 2 challenge, Task 3 policy cells, Task 4 analysis functions, `GeneralsEvaluator`
- Produces: `GeneralsAttributionPipeline.run() -> AttributionRunResult` and an immutable `generals-scientific-attribution` run

- [ ] **Step 1: Write a failing end-to-end fake-evaluator test**

```python
def test_attribution_pipeline_runs_four_cells_on_identical_cases(tmp_path):
    result = make_pipeline(tmp_path, evaluator=PairedEvaluator()).run()
    assert result.status == "complete"
    assert result.policy_order == ("A", "B", "C", "D")
    assert result.case_count_per_policy == 12
    assert result.diagnostic_state_count <= 48
    summary = read_json(result.run_dir / "summary.json")
    assert summary["coding_agent_act_count"] == 0
    assert summary["formal_benchmark_opened"] is False
    assert summary["attribution"]["estimands"] == [
        "B-A", "C-A", "D-B-C+A",
    ]
```

Failure tests cover one invalid game, mismatched pair IDs, source-hash drift,
more than 48 states, a nondeterministic probe, and unknown/malformed events.

- [ ] **Step 2: Run the test and verify the missing pipeline**

Run: `.venv/bin/pytest tests/generals/test_attribution_pipeline.py -q`

Expected: FAIL because the pipeline is absent.

- [ ] **Step 3: Add first-class event names**

Add these to `KNOWN_EVENT_TYPES` and the research-boundary test:

```python
{
    "attribution_policy_materialized",
    "attribution_game_result",
    "attribution_factorial_effect",
    "diagnostic_state_selected",
    "diagnostic_policy_probe",
    "trajectory_divergence",
    "attribution_report_frozen",
}
```

- [ ] **Step 4: Implement the pipeline and immutable report artifacts**

Define:

```python
@dataclass(frozen=True)
class AttributionRunResult:
    run_dir: Path
    status: str
    policy_order: tuple[str, ...]
    case_count_per_policy: int
    valid_case_count: int
    diagnostic_state_count: int
    report_hash: str | None

GeneralsAttributionPipeline.run(self) -> AttributionRunResult
```

Create a `Run(game="28_generals", agent="generals-scientific-attribution",
run_type="measurement")`; verify and materialize policies; evaluate A/B/C/D in
that order on the same cases; save action profiles and dense summaries; select
and probe diagnostics; calculate paired effects; then atomically write
`diagnosis/report.json`, `diagnosis/report.md`, `diagnosis/evidence.json`, and
`summary.json`. The report includes every case, missing field, trigger,
divergence, effect, bootstrap interval, policy/source hash, and artifact path.
No provider object is accepted by this class.

- [ ] **Step 5: Validate projection and run focused tests**

The finish gate requires exactly 48 game results, 48 dense episode summaries,
four source receipts, no formal phase, no act event, at most 48 unique selected
states, four probe records per selected state, and zero quality defects.

Run:

```bash
.venv/bin/pytest tests/generals/test_attribution_pipeline.py tests/test_research_boundaries.py tests/generals/test_evaluator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/generals/attribution_pipeline.py src/agentbench_frame/tracking/quality.py tests/generals/test_attribution_pipeline.py tests/test_research_boundaries.py
git commit -m "feat(generals): run scientific v8 attribution"
```

### Task 6: Build the leak-safe v9 diagnosis prompt

**Files:**
- Create: `src/agentbench_frame/generals/prompt_v9.py`
- Create: `tests/generals/test_prompt_v9.py`

**Interfaces:**
- Consumes: complete attribution run, frozen v7 source/docs, official rules, replay Skill
- Produces: `build_round9_prompt() -> PromptBuildResult` and `Round9PromptReceipt`

- [ ] **Step 1: Write failing allowlist and denylist tests**

```python
def test_v9_prompt_uses_attribution_and_v7_but_not_v8_source():
    result = build_round9_prompt(**valid_context())
    assert result.manifest["policy_parent_version"] == "v7"
    assert result.manifest["iteration_predecessor_version"] == "v8"
    assert result.manifest["provider_act_limit"] == 1
    assert result.manifest["diagnosis_report_hash"] == REPORT_HASH
    assert "versions/v8/source" not in result.text
    assert "formal_score" not in result.text

@pytest.mark.parametrize("forbidden", [
    "303101", "eval-high", "formal-spec", "controlled_reference_policy_kl",
    "versions/v8/source/strategy.py", "benchmark_score",
])
def test_v9_prompt_rejects_sealed_or_nonparent_material(forbidden):
    with pytest.raises(ValueError):
        build_round9_prompt(**context_with_injected_text(forbidden))
```

Also reject an incomplete attribution run, altered report/evidence hashes,
unselected replays, prompt size over 196,608 bytes, duplicate state IDs, wrong
v7 source hash, wrong rules/Skill hash, or any validation/formal/KL role.

- [ ] **Step 2: Run the test and verify the missing module**

Run: `.venv/bin/pytest tests/generals/test_prompt_v9.py -q`

Expected: FAIL on import.

- [ ] **Step 3: Implement prompt receipts and content validation**

Use:

```python
@dataclass(frozen=True)
class Round9PromptReceipt:
    prompt_sha256: str
    prompt_bytes: int
    diagnosis_report_sha256: str
    diagnosis_evidence_sha256: str
    included_replay_ids: tuple[str, ...]
    included_state_ids: tuple[str, ...]
    policy_parent_hash: str
    provider_act_limit: int = 1
```

Require an attribution `summary.json` with `status=complete`, matching run ID,
zero quality defects, exact A/B/C/D policy hashes, exact 12 pair IDs, and the
recorded report/evidence hashes. Serialize evidence deterministically and stop
before 196,608 bytes; omission is by fixed lower-priority tail and is recorded,
never silent.

- [ ] **Step 4: Add the actual v9 instruction**

The prompt states that the editable source is v7; the attribution is diagnostic
evidence rather than reward proof; formal/validation/KL are sealed; exactly one
act is permitted; explainable bounded search/scoring/FSM/refactoring are
allowed; rule compression is required; replay IDs and seeds cannot become
runtime features; and `STRATEGY.md` plus `EXPERIENCE.md` must record retained,
rejected, and new lessons.

- [ ] **Step 5: Run prompt regressions and commit**

Run:

```bash
.venv/bin/pytest tests/generals/test_prompt_v9.py tests/generals/test_prompt_v8.py tests/generals/test_replay_prompt.py -q
```

Expected: PASS.

```bash
git add src/agentbench_frame/generals/prompt_v9.py tests/generals/test_prompt_v9.py
git commit -m "feat(generals): build attribution-guided v9 prompt"
```

### Task 7: Implement the single-act v9 pipeline and frozen-candidate recovery

**Files:**
- Create: `src/agentbench_frame/generals/pipeline_v9.py`
- Create: `tests/generals/test_pipeline_v9.py`
- Create: `tests/generals/test_pipeline_v9_recovery.py`

**Interfaces:**
- Consumes: Task 2 lineage/challenge, Task 5 attribution run, Task 6 prompt, `ProviderAdapter`, `GeneralsEvaluator`
- Produces: `GeneralsHLRound9Pipeline.run() -> Round9PipelineResult`, `recover()`, frozen v9 testcase, optional atomic champion metadata

- [ ] **Step 1: Write failing happy-path and threshold tests**

```python
def test_v9_runs_one_act_then_all_validation_and_formal_cases(tmp_path):
    provider = CountingProvider(valid_v9_patch())
    result = make_pipeline(tmp_path, provider=provider, evaluator=WinningEvaluator()).run()
    assert provider.calls == 1
    assert result.runnable is True
    assert result.validation_attempted is True
    assert result.formal_attempted is True
    assert result.champion_claim is True
    summary = read_json(result.run_dir / "summary.json")
    assert summary["policy_parent_version"] == "v7"
    assert summary["iteration_predecessor_version"] == "v8"
    assert summary["round_act_count"] == 1
    assert len(summary["score_history"]) == len(PREDECESSOR_HISTORY) + 1

def test_validation_failure_cannot_hide_formal(tmp_path):
    result = make_pipeline(
        tmp_path, evaluator=ValidationLosingFormalEvaluator()
    ).run()
    assert result.validation_passed is False
    assert result.formal_attempted is True
    assert count_phase_games(result.run_dir, "formal9") == 18
```

Parameterize champion rejection for validation `1/12`, missing one seat,
formal `12/18`, high `1/6`, missing one high-tier seat, or one invalid formal
game. In every rejection v9 remains saved and champion metadata remains v7.

- [ ] **Step 2: Write failing safety and recovery tests**

Reject a second provider call, v8 source in the workspace, protected-file
mutation, random/network/subprocess runtime use, seed/state-ID branches,
nondeterminism, illegal commands, missing strategy documents, over-limit macro,
and latency failure. Recovery accepts only a frozen, hash-verified runnable v9
candidate and uses a provider whose `invoke()` raises if called.

- [ ] **Step 3: Run tests and verify missing pipeline**

Run:

```bash
.venv/bin/pytest tests/generals/test_pipeline_v9.py tests/generals/test_pipeline_v9_recovery.py -q
```

Expected: FAIL on import.

- [ ] **Step 4: Implement the pipeline result and execution state machine**

```python
@dataclass(frozen=True)
class Round9PipelineResult:
    run_dir: Path
    status: str
    runnable: bool
    raw_score: float | None
    evo_score_9: float | None
    gain_9: float | None
    validation_attempted: bool
    validation_passed: bool
    formal_attempted: bool
    performance_target_met: bool
    champion_claim: bool
    global_act_count: int
    round_act_count: int
```

Create the run; write attribution, validation, and formal specs; import exact
v7; validate the attribution receipt; build the prompt; invoke the provider
once; capture raw JSONL/tokens/tools/time and pre/post snapshots; enforce the
editable-file allowlist; run candidate tests/probes; freeze v9; evaluate all 12
validation and 18 formal cases; then calculate raw/evo/gain, tier/seat results,
budgets, and the champion decision. `score_history` appends the actual v9 score
to frozen v8 chronology; the known historical gap remains `null`, so affected
AUC values remain unavailable.

- [ ] **Step 5: Implement fail-closed frozen-candidate recovery**

`recover(failed_run_dir)` verifies run ID, prompt/report hashes, provider act
count exactly one, raw output, v7 preimage, frozen v9 manifest, protected files,
and every pre-existing event payload. It may resume only missing validation,
formal, summary, or quality-finalization stages. It never invokes a provider,
changes the v9 tree, reopens diagnosis inputs, or upgrades a failed candidate.

- [ ] **Step 6: Run focused and v8 regression tests**

Run:

```bash
.venv/bin/pytest tests/generals/test_pipeline_v9.py tests/generals/test_pipeline_v9_recovery.py tests/generals/test_pipeline_v8.py tests/generals/test_pipeline_v8_recovery.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/generals/pipeline_v9.py tests/generals/test_pipeline_v9.py tests/generals/test_pipeline_v9_recovery.py
git commit -m "feat(generals): run one attribution-guided v9 act"
```

### Task 8: Generate and freeze 12 official intervention states

**Files:**
- Create: `src/agentbench_frame/generals/intervention_states.py`
- Create: `tests/generals/test_intervention_states.py`
- Generate: `backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.states.json`
- Modify: `backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py`

**Interfaces:**
- Consumes: Task 1 recipe records and official engine
- Produces: `build_intervention_state_pack() -> InterventionStatePack` and a byte-stable 12-state asset

- [ ] **Step 1: Write failing recipe and round-trip tests**

```python
def test_intervention_pack_is_exact_and_balanced(engine_root, manifest):
    pack = build_intervention_state_pack(engine_root, manifest)
    assert len(pack.states) == 12
    assert Counter(item.scenario for item in pack.states) == EXPECTED_SCENARIOS
    assert Counter(item.actor for item in pack.states) == {0: 6, 1: 6}
    assert len({item.measurement_state_id for item in pack.states}) == 12
    for item in pack.states:
        restored = OfficialGeneralsEngine.from_measurement_state(
            engine_root, item.snapshot, tmp_path / f"{item.state_key}.jsonl"
        )
        assert restored.measurement_state(item.actor) == item.snapshot
        assert assert_intervention_scenario(item) is None
```

Also assert repeated generation is byte-identical, every state is nonterminal,
both mains exist, the acting side has a legal end macro, and its exact scenario
predicate is true.

- [ ] **Step 2: Run the test and verify missing implementation**

Run with `AGENTBENCH_ASSET_ROOT` set to the Generals-assets worktree:

```bash
AGENTBENCH_ASSET_ROOT=/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets .venv/bin/pytest tests/generals/test_intervention_states.py -q
```

Expected: FAIL on import.

- [ ] **Step 3: Implement deterministic official-state recipes**

Define `InterventionState` with key, scenario, actor, variant, snapshot, state
ID, construction receipt, and assertion receipt. Each recipe starts from a
fresh official engine with fixed recipe seed `304000 + recipe_index`, clears
non-main generals/weapons, resets technologies/cooldowns/movement budgets, and
uses plain walkable cells except scenario-specific terrain. Mirrored actor-0
and actor-1 positions use main coordinates `(7, 4)` and `(7, 10)`.

Construct these predicates:

```python
SCENARIO_ASSERTIONS = {
    "contact": owned_movable_stack_touches_hostile,
    "main_general_danger": hostile_pressure_exceeds_main_reserve,
    "large_stack_routing": two_routes_include_stack_at_least_24,
    "economy_combat_conflict": contact_and_affordable_upgrade,
    "counter_capture": exact_adjacent_counter_capture_exists,
    "mid_late_consolidation": round_at_least_100_and_owned_merge_exists,
}
```

Use armies `7/2` for contact, main `13` versus adjacent `20` for danger,
movable stacks `4` and `30` for large-stack routing, 40 coins plus `7/2`
contact for economy conflict, enemy `5` plus owned adjacent `9` for
counter-capture, and separated owned `18`/`14` stacks at round 120 for
consolidation. Variants mirror direction and swap a noncritical terrain cell.

- [ ] **Step 4: Generate the committed state pack and verify byte stability**

Add a module entrypoint that writes canonical sorted JSON with a trailing
newline. Run it twice, hash before/after, and require equality:

```bash
.venv/bin/python -m agentbench_frame.generals.intervention_states --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.toml --output /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.states.json
sha256sum /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.states.json
```

Record the resulting pack SHA-256 in every measurement-run receipt. The asset
contract regenerates canonical bytes and requires their SHA-256 to equal the
committed file. Run the generator again and require the hash unchanged; the
already frozen TOML recipe manifest does not change.

- [ ] **Step 5: Run Framework and asset tests**

Run:

```bash
AGENTBENCH_ASSET_ROOT=/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets .venv/bin/pytest tests/generals/test_intervention_states.py tests/generals/test_measurement_state.py tests/generals/test_macro_action_space.py -q
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
pytest backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Framework generator and generated asset separately**

Framework:

```bash
git add src/agentbench_frame/generals/intervention_states.py tests/generals/test_intervention_states.py
git commit -m "feat(generals): construct exact KL intervention states"
```

Assets:

```bash
git add backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.states.json backend_sources/corpus/28_generals/tests/test_policy_kl_expanded_contract.py
git commit -m "assets(generals): freeze expanded KL state pack"
```

### Task 9: Implement legacy-12 and expanded-24 exact KL measurement

**Files:**
- Create: `src/agentbench_frame/generals/policy_kl_expanded.py`
- Create: `tests/generals/test_policy_kl_expanded.py`
- Modify: `src/agentbench_frame/generals/policy_kl_reuse.py`
- Modify: `tests/generals/test_policy_kl_reuse.py`

**Interfaces:**
- Consumes: verified legacy v3 KL run, frozen 12-state pack, runnable v9 run/hash
- Produces: `GeneralsExpandedPolicyKLPipeline.run()/recover()` and separate `legacy-12` plus `expanded-24` metrics

- [ ] **Step 1: Write failing complete-measurement test**

```python
def test_expanded_pipeline_reuses_legacy_and_measures_24_states(tmp_path):
    result = make_expanded_pipeline(tmp_path).run()
    assert result.status == "complete"
    summary = result.summary
    assert summary["domains"]["legacy-12"]["reference_state_count"] == 12
    assert summary["domains"]["expanded-24"]["reference_state_count"] == 24
    assert len(summary["domains"]["expanded-24"]["transitions"]) == 9
    assert summary["domains"]["expanded-24"]["transitions"][-1][
        "version_after"
    ] == "v9"
    assert fake_counter.calls == 12
    assert fake_probe.calls == 12 * 10 + 12
```

The final `+12` is the v9 probe on legacy states; all legacy v0-v8 actions and
counts are hash-reused. Assert event totals: 24 selected states, 24 exact count
facts, 240 expanded policy actions, 864 expanded KL facts, and 48 legacy
v8-to-v9 KL facts.

- [ ] **Step 2: Write failing incompleteness and provenance tests**

Reject changed legacy tree hash, changed legacy event payload, altered state
pack hash, v9 target mismatch, duplicate scenario/key/state ID, wrong epsilon,
nonuniform smoothing, nondeterministic/illegal action, and approximate count.
One incomplete exact count must set expanded-24 aggregate values to `null`,
retain all completed facts, and return a resumable incomplete status.

- [ ] **Step 3: Run tests and verify missing pipeline**

Run:

```bash
.venv/bin/pytest tests/generals/test_policy_kl_expanded.py tests/generals/test_policy_kl_reuse.py -q
```

Expected: FAIL on missing imports.

- [ ] **Step 4: Generalize verified reuse without weakening v1-v3 checks**

Add a domain-filtered verifier whose exact interface is:

```python
verify_policy_kl_domain(
    run_dir: Path,
    *,
    expected_measurement_id: str,
    expected_tree_hash: str,
    expected_state_count: int,
    expected_versions: tuple[str, ...],
) -> VerifiedPolicyKLSource
```

The existing `verify_policy_kl_source()` delegates to it with its current
constants, so old extension tests remain byte-for-byte strict.

- [ ] **Step 5: Implement expanded measurement and recovery**

Materialize verified legacy artifacts append-only. Load and hash-check the
intervention pack. Resolve v0-v8 from frozen history and v9 from the target run.
Run `ExactMacroCounter` only on the 12 intervention states; probe v0-v9 twice
on each; probe v9 twice on legacy-12; call
`compute_controlled_policy_kl()` separately for both domain IDs. Persist domain
ID and reference-state count on every fact. `recover()` resumes only exact
count/probe work in the same append-only run and rejects a complete run or any
changed pre-existing fact.

- [ ] **Step 6: Run focused and legacy regression tests**

Run:

```bash
.venv/bin/pytest tests/generals/test_policy_kl_expanded.py tests/generals/test_policy_kl_reuse.py tests/generals/test_policy_kl_pipeline.py tests/generals/test_policy_kl_extension.py tests/generals/test_policy_kl_v8_extension.py tests/generals/test_policy_kl_math.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/generals/policy_kl_expanded.py src/agentbench_frame/generals/policy_kl_reuse.py tests/generals/test_policy_kl_expanded.py tests/generals/test_policy_kl_reuse.py
git commit -m "feat(generals): measure expanded exact policy KL"
```

### Task 10: Expose CLI, CI projections, and English paper figures

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `src/agentbench_frame/report/builder.py`
- Modify: `src/agentbench_frame/report/templates/index.html`
- Create: `src/agentbench_frame/generals/paper_figure_v9.py`
- Create: `tests/generals/test_cli_v9.py`
- Create: `tests/generals/test_policy_kl_expanded_figure.py`
- Create: `tests/generals/test_v9_paper_figures.py`
- Modify: `tests/test_local_report_research.py`

**Interfaces:**
- Consumes: attribution, v9, and expanded-KL run directories
- Produces: CLI commands, CI data without interpolation, PNG/SVG paper figures

- [ ] **Step 1: Write failing CLI routing tests**

Cover these exact commands and arguments:

```text
attribute-v8-regression --attribution-manifest --v7-run --v8-run
iterate-v9 --challenge-manifest --replay-skill --parent-run --predecessor-run --attribution-run --expected-parent-hash --expected-predecessor-hash --codex-executable --provider-timeout
recover-v9 --failed-run plus the same frozen authorities, using no provider
measure-policy-kl-expanded --reference-manifest --source-run --target-run --expected-target-hash --count-wall-time --count-max-states
recover-policy-kl-expanded plus --failed-run
plot-v9-paper --attribution-run --v9-run --legacy-kl-run --expanded-kl-run --output-dir
```

Assert each handler prints sorted JSON with status and run path and returns 1
for incomplete/invalid results.

- [ ] **Step 2: Write failing report and figure tests**

```python
def test_ci_keeps_attribution_missingness_and_domain_labels():
    report = build_report(run_with_missing_attribution_metric())
    assert report["attribution"]["army"]["large_stack"] is None
    assert report["policy_kl_domains"][0]["domain_id"] == "legacy-12"
    assert report["policy_kl_domains"][1]["domain_id"] == "expanded-24"

def test_paper_loader_refuses_domain_splicing(tmp_path):
    with pytest.raises(ValueError, match="reference domain"):
        load_v9_figure_data(spliced_run(tmp_path))
```

Figure tests assert English titles/labels, nine transition labels through
`v8→v9`, 12 versus 24 state annotations, all four epsilon traces, exact support
labels, 2x2 attribution cells, score gaps preserved as gaps, and deterministic
PNG/SVG creation.

- [ ] **Step 3: Run tests and verify missing routes/projections**

Run:

```bash
.venv/bin/pytest tests/generals/test_cli_v9.py tests/generals/test_policy_kl_expanded_figure.py tests/generals/test_v9_paper_figures.py tests/test_local_report_research.py -q
```

Expected: FAIL on missing commands and figure module.

- [ ] **Step 4: Implement CLI and CI projection**

Wire the five pipelines and figure command without changing existing command
arguments. Add CI panels for attribution case/effect tables, action disagreement
and earliest divergence, v9 validation/formal/tier/seat outcomes, and two
separate KL domain tables. Render `None` as `missing`; render incomplete status
prominently; never synthesize an absent point.

- [ ] **Step 5: Implement deterministic English figures**

`paper_figure_v9.py` validates all source run IDs/hashes/domain IDs before
plotting. It writes atomically:

```text
generals-v9-scientific-attribution.{png,svg}
generals-v0-v9-score-and-policy-change.{png,svg}
generals-legacy12-expanded24-kl.{png,svg}
```

The dual-domain figure places legacy-12 and expanded-24 in separate panels and
never draws a connecting segment between them. The score/IG figure labels
policy KL as behavioral information gain and preserves the historical missing
score point.

- [ ] **Step 6: Run focused and report regression tests**

Run:

```bash
.venv/bin/pytest tests/generals/test_cli_v9.py tests/generals/test_policy_kl_expanded_figure.py tests/generals/test_v9_paper_figures.py tests/generals/test_policy_kl_figure.py tests/generals/test_policy_kl_cli.py tests/test_local_report_research.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/generals/cli.py src/agentbench_frame/report/builder.py src/agentbench_frame/report/templates/index.html src/agentbench_frame/generals/paper_figure_v9.py tests/generals/test_cli_v9.py tests/generals/test_policy_kl_expanded_figure.py tests/generals/test_v9_paper_figures.py tests/test_local_report_research.py
git commit -m "feat(generals): report v9 attribution and KL"
```

### Task 11: Run implementation verification before real evidence

**Files:**
- Verify only; modify a failing component in its owning task before proceeding

**Interfaces:**
- Consumes: Tasks 1-10 implementation commits
- Produces: a clean, fully tested implementation ready for irreversible provider budget and real game time

- [ ] **Step 1: Run all focused Generals tests with live assets**

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
AGENTBENCH_ASSET_ROOT=/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets .venv/bin/pytest tests/generals -q
```

Expected: all Generals tests pass; only explicitly environment-gated tests may
skip.

- [ ] **Step 2: Run the complete Framework test suite**

```bash
.venv/bin/pytest -q
```

Expected: no failures.

- [ ] **Step 3: Run all Generals asset contracts**

```bash
cd /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets
pytest backend_sources/corpus/28_generals/tests -q
```

Expected: no failures.

- [ ] **Step 4: Verify frozen authorities and clean trees**

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
.venv/bin/python -c "from pathlib import Path; from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter as S; print(S().capture(Path('agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632/versions/v7/source')).content_hash); print(S().capture(Path('agentbench_data/runs/28_generals/generals-hl/20260806_1126_8e1b6795/versions/v8/source')).content_hash)"
git status --short --branch
git -C /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets status --short --branch
```

Expected: hashes equal the Global Constraints and both trees are clean.

### Task 12: Run real attribution, one v9 act, exact KL, and final reporting

**Files:**
- Generate outside Git: `agentbench_data/runs/28_generals/generals-scientific-attribution/<run-id>/`
- Generate outside Git: `agentbench_data/runs/28_generals/generals-hl/<run-id>/`
- Generate outside Git: `agentbench_data/runs/28_generals/generals-policy-kl-expanded/<run-id>/`
- Create: `docs/experiments/2026-08-06-generals-v9-scientific-attribution-result.md`
- Generate: `docs/experiments/figures/generals-v9-*`

**Interfaces:**
- Consumes: verified implementation, local Codex CLI/authentication, frozen manifests and authorities
- Produces: real auditable runs, paper figures, final conclusion, champion decision

- [ ] **Step 1: Run the real paired attribution**

```bash
cd /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl
.venv/bin/python -m agentbench_frame.cli generals attribute-v8-regression --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data --attribution-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v9-scientific-attribution-v1.toml --v7-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632 --v8-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260806_1126_8e1b6795 > /tmp/generals-v9-attribution-cli.json
sed -n '1,20p' /tmp/generals-v9-attribution-cli.json
```

Expected: status `complete`, 48/48 valid games, zero acts, at most 48 diagnostic
states, four probes per state, and zero event-quality defects. The JSON file is
the exact handoff receipt for the next command.

- [ ] **Step 2: Execute exactly one real Codex v9 act and all evaluations**

Resolve and validate the exact attribution run from the Step 1 receipt, then
run `iterate-v9` in the same shell:

```bash
generals_attribution_run=$(.venv/bin/python -c "import json; print(json.load(open('/tmp/generals-v9-attribution-cli.json', encoding='utf-8'))['run_dir'])")
test -d "$generals_attribution_run"
test "$(basename "$generals_attribution_run")" = "$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['run_id'])" "$generals_attribution_run/summary.json")"
.venv/bin/python -m agentbench_frame.cli generals iterate-v9 --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data --challenge-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/v9-scientific-attribution-v1.toml --replay-skill backend_sources/corpus/28_generals/benchmark/replay.md --parent-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260730_1739_680b1632 --predecessor-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-hl/20260806_1126_8e1b6795 --attribution-run "$generals_attribution_run" --expected-parent-hash c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4 --expected-predecessor-hash 3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a --codex-executable codex --provider-timeout 1800 > /tmp/generals-v9-cli.json
sed -n '1,20p' /tmp/generals-v9-cli.json
```

Expected: one provider act at most; if a provider or candidate gate fails,
retain the failed run and stop without another act. If v9 is runnable, require
12 validation plus 18 formal games even when the champion target is missed.

- [ ] **Step 3: Audit the v9 run before KL**

Inspect `summary.json`, `events.jsonl`, provider JSONL, prompt receipt, patch,
source manifest, validation/formal game counts, tier/seat table, quality report,
and budget totals. Require `policy_parent_version=v7`,
`iteration_predecessor_version=v8`, `round_act_count=1`, the new source hash to
match `versions/v9/manifest.json`, and the champion decision to exactly match
the predeclared thresholds.

- [ ] **Step 4: Run expanded exact KL**

Resolve the exact runnable v9 run and source hash from the Step 2 receipt and
verified manifest, then run expanded KL in the same shell:

```bash
generals_v9_run=$(.venv/bin/python -c "import json; print(json.load(open('/tmp/generals-v9-cli.json', encoding='utf-8'))['run_dir'])")
test -d "$generals_v9_run/versions/v9/source"
generals_v9_hash=$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['content_hash'])" "$generals_v9_run/versions/v9/manifest.json")
.venv/bin/python -m agentbench_frame.cli generals measure-policy-kl-expanded --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-expanded-v1.toml --source-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-policy-kl/20260806_1129_6085e1a6 --target-run "$generals_v9_run" --expected-target-hash "$generals_v9_hash" --count-wall-time 3600 --count-max-states 5000000 > /tmp/generals-v9-expanded-kl-cli.json
sed -n '1,20p' /tmp/generals-v9-expanded-kl-cli.json
```

If the counter returns incomplete, preserve the run, report it as incomplete,
and use `recover-policy-kl-expanded` only to resume exact counting with unchanged
inputs; never substitute an estimate.

- [ ] **Step 5: Validate every run and generate figures**

Run the full data validator:

```bash
.venv/bin/python -m agentbench_frame.cli data check --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

Then run `plot-v9-paper` with the exact four receipt-verified run paths.
Expected: zero malformed, unknown, duplicate, missing-ID, and missing-field
defects; three PNG/SVG figure pairs whose labels and values trace to source
events.

- [ ] **Step 6: Write and verify the final result report**

The English-data/Chinese-conclusion report records run IDs/paths, commits,
source hashes, act/token/tool/time budgets, all 48 attribution games, causal
limitations, v9 validation/formal/tier/seat results, raw/evo/gain/AUC status,
legacy-12 and expanded-24 KL, support sizes, epsilon sensitivity, event quality,
and whether v7 or v9 is champion. It explicitly distinguishes:

```text
performance target missed
measurement incomplete
pipeline failed
```

No one category may be substituted for another.

- [ ] **Step 7: Run final verification and commit only reports/figures**

Run the complete Framework suite, all Generals asset contracts, and data checks
again. Inspect every generated figure. Then commit the report and deterministic
figures; run directories remain untracked first-hand data.

```bash
git add docs/experiments/2026-08-06-generals-v9-scientific-attribution-result.md docs/experiments/figures/generals-v9-scientific-attribution.png docs/experiments/figures/generals-v9-scientific-attribution.svg docs/experiments/figures/generals-v0-v9-score-and-policy-change.png docs/experiments/figures/generals-v0-v9-score-and-policy-change.svg docs/experiments/figures/generals-legacy12-expanded24-kl.png docs/experiments/figures/generals-legacy12-expanded24-kl.svg
git commit -m "docs(generals): report v9 scientific attribution"
```

Expected final state: both worktrees clean, all required tests and data checks
pass, every runnable v9 is fully evaluated, champion metadata matches the
predeclared gate, and neither branch has been pushed.
