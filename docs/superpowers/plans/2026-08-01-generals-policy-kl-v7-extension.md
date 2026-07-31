# Generals Controlled Policy-KL v7 Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one auditable v0-v7 controlled-reference policy-KL run by cryptographically reusing the completed v1 domain, probing only v7, computing v6→v7, and regenerating the English three-panel paper figure.

**Architecture:** Keep the completed v1 pipeline and run immutable. Add a frozen v2 asset contract, a pure source-run verifier/materializer, a pure KL computation helper, and a dedicated extension pipeline that re-emits verified v1 facts with provenance and computes only the new transition. Dispatch figure validation by measurement ID so v1 remains reproducible and v2 requires seven transitions.

**Tech Stack:** Python 3.11, dataclasses, `pathlib`, SHA-256, JSON/JSONL/TOML, pytest, existing AgentBench `Run` tracking, existing Generals macro-action space and historical-policy subprocess probe, Matplotlib.

**Design Spec:** `docs/superpowers/specs/2026-08-01-generals-policy-kl-v7-extension-design.md`

## Global Constraints

- Work only in `/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl` and `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets`; preserve unrelated user changes.
- Keep `/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-policy-kl/20260730_1126_8d123b55` read-only and prove its canonical tree hash remains `6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225` before and after the extension.
- Freeze v7 to HL run `20260730_1739_680b1632` and source content hash `c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
- Keep the exact 12 v1 reference states: seeds `289101, 289202, 289303`, seats `0, 1`, decisions `2, 10`, strongest-human collector `advanced-rank02-robinliu-v18`.
- Keep strict uniform `U_s`, epsilons `0.001, 0.01, 0.05, 0.1`, primary epsilon `0.01`, and action-space spec ID `a383f61cba2b284623c0b377eaddee4ef7522b8e8adb53e0ea3efb7bc9bc329e`.
- Do not recollect reference states, recount supports, execute v0-v6, interpolate missing data, or include invalid v8 attempts.
- A complete v2 run contains seven ordered transitions, 12/12 coverage at every epsilon, 336 per-state KL facts, source provenance, and zero event-quality defects.
- Any source mismatch fails closed. Missing/nondeterministic/illegal v7 behavior remains explicit and produces an incomplete measurement with null affected aggregates.
- Follow TDD, use `apply_patch` for edits, run focused tests before full suites, and commit each independently reviewable task.

---

## File Structure

### Assets worktree

- Create `backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v2.toml`: frozen v2 measurement, source-run, tree-hash, and v0-v7 history contract.
- Create `backend_sources/corpus/28_generals/tests/test_policy_kl_reference_v2_contract.py`: byte-level v2 asset checks.
- Modify `backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py`: allow the intentionally shared controlled-state seeds in only v1/v2 policy-KL manifests.

### Framework worktree

- Modify `src/agentbench_frame/generals/models.py`: add `PolicyKLExtensionConfig`.
- Modify `src/agentbench_frame/generals/assets.py`: parse and strictly validate v2 without changing v1 validation.
- Create `tests/generals/fixtures/policy-kl-reference-v2.toml`: Framework-side loader fixture identical to the real asset.
- Modify `tests/generals/test_assets.py`: v2 loader positive and mutation tests.
- Create `src/agentbench_frame/generals/policy_kl_math.py`: pure per-state fact and transition aggregation logic extracted from the v1 pipeline.
- Create `tests/generals/test_policy_kl_math.py`: order, exact-decimal, missing, and aggregate tests.
- Modify `src/agentbench_frame/generals/policy_kl_pipeline.py`: delegate unchanged v1 math to the pure helper.
- Create `src/agentbench_frame/generals/policy_kl_reuse.py`: canonical tree hash, strict v1 source verification, artifact materialization, and receipt types.
- Create `tests/generals/test_policy_kl_reuse.py`: tree-hash, tamper, structural, copying, and old-run immutability tests.
- Create `src/agentbench_frame/generals/policy_kl_extension.py`: dedicated run/recovery orchestration, only-v7 probe, provenance events, and v2 summary.
- Create `tests/generals/test_policy_kl_extension.py`: fake end-to-end, only-v7, incomplete, provenance, counts, and recovery tests.
- Modify `src/agentbench_frame/tracking/quality.py`: register the two provenance event types.
- Modify `src/agentbench_frame/generals/cli.py`: add `extend-policy-kl-v7` and `recover-policy-kl-v7`.
- Modify `tests/generals/test_policy_kl_cli.py`: parser and routing tests for extension commands.
- Modify `src/agentbench_frame/generals/paper_figure.py`: measurement-ID-dispatched six/seven-transition validation.
- Modify `tests/generals/test_policy_kl_figure.py`: preserve v1 and exercise v2 rendering.
- Modify `docs/experiments/figures/generals-controlled-policy-kl-three-panel.{svg,png}`: regenerated v0-v7 outputs.
- Create `docs/experiments/2026-08-01-generals-controlled-policy-kl-v7-result.md`: first-hand result, provenance, limitations, and commands.

---

### Task 1: Freeze the v2 assets contract

**Files:**
- Create: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v2.toml`
- Create: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/tests/test_policy_kl_reference_v2_contract.py`
- Modify: `/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py`

**Interfaces:**
- Consumes: approved v1 manifest, completed v1 run ID/tree hash, retained v7 run ID/source hash.
- Produces: immutable TOML keys `measurement_id`, `source_measurement_id`, `source_run_id`, `source_tree_hash`, common domain fields, and eight ordered `history` tables.

- [ ] **Step 1: Write the failing v2 asset tests**

```python
from pathlib import Path
import tomllib

GAME_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_V2 = GAME_ROOT / "benchmark/policy-kl-reference-v2.toml"


def test_policy_kl_v2_freezes_source_and_v7():
    raw = tomllib.loads(REFERENCE_V2.read_text(encoding="utf-8"))
    assert raw["measurement_id"] == "generals-policy-kl-reference-v2"
    assert raw["source_measurement_id"] == "generals-policy-kl-reference-v1"
    assert raw["source_run_id"] == "20260730_1126_8d123b55"
    assert raw["source_tree_hash"] == (
        "6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225"
    )
    assert [item["version"] for item in raw["history"]] == [
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
    ]
    assert raw["history"][-1] == {
        "version": "v7",
        "run_id": "20260730_1739_680b1632",
        "content_hash": (
            "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
        ),
    }


def test_policy_kl_v2_reuses_exact_v1_domain():
    v1 = tomllib.loads(
        (GAME_ROOT / "benchmark/policy-kl-reference-v1.toml").read_text()
    )
    v2 = tomllib.loads(REFERENCE_V2.read_text())
    for key in (
        "opponent_id", "seeds", "seats", "decision_numbers",
        "epsilons", "primary_epsilon",
    ):
        assert v2[key] == v1[key]
    assert v2["history"][:-1] == v1["history"]
```

Update the existing seed-occurrence assertion to require each seed to appear
in exactly `policy-kl-reference-v1.toml` and `policy-kl-reference-v2.toml`.

- [ ] **Step 2: Run the asset tests and verify the new-file failure**

Run from the assets worktree:

```bash
/home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python -m pytest \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_v2_contract.py -q
```

Expected: FAIL because `policy-kl-reference-v2.toml` does not exist.

- [ ] **Step 3: Create the exact v2 TOML**

Copy the v1 common fields and seven history blocks byte-for-byte, prepend:

```toml
measurement_id = "generals-policy-kl-reference-v2"
source_measurement_id = "generals-policy-kl-reference-v1"
source_run_id = "20260730_1126_8d123b55"
source_tree_hash = "6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225"
```

Append:

```toml
[[history]]
version = "v7"
run_id = "20260730_1739_680b1632"
content_hash = "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
```

- [ ] **Step 4: Run the focused asset tests**

Run the command from Step 2. Expected: all tests PASS.

- [ ] **Step 5: Commit the asset contract**

```bash
git add backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v2.toml \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_contract.py \
  backend_sources/corpus/28_generals/tests/test_policy_kl_reference_v2_contract.py
git commit -m "feat(generals): freeze policy KL v7 extension"
```

---

### Task 2: Parse and reject mutations of the v2 contract

**Files:**
- Modify: `src/agentbench_frame/generals/models.py`
- Modify: `src/agentbench_frame/generals/assets.py`
- Create: `tests/generals/fixtures/policy-kl-reference-v2.toml`
- Modify: `tests/generals/test_assets.py`

**Interfaces:**
- Consumes: v2 TOML from Task 1 and `PilotConfig`.
- Produces: `PolicyKLExtensionConfig` and `load_policy_kl_extension_config(path: Path, pilot: PilotConfig) -> PolicyKLExtensionConfig`.

- [ ] **Step 1: Add failing loader tests and the exact fixture**

Copy the Task 1 TOML to the Framework fixture, then add:

```python
def test_policy_kl_extension_manifest_freezes_v1_source_and_v7():
    pilot = load_pilot_config(FIXTURE)
    config = load_policy_kl_extension_config(
        Path(__file__).parent / "fixtures/policy-kl-reference-v2.toml",
        pilot,
    )
    assert config.measurement_id == "generals-policy-kl-reference-v2"
    assert config.source_measurement_id == "generals-policy-kl-reference-v1"
    assert config.source_run_id == "20260730_1126_8d123b55"
    assert config.source_tree_hash == (
        "6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225"
    )
    assert tuple(item.version for item in config.history) == tuple(
        f"v{index}" for index in range(8)
    )
    assert config.history[-1].content_hash == (
        "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"
    )
```

Add parametrized mutations for source run ID, source tree hash, v7 run ID,
v7 content hash, domain seed, epsilon, and history order. Each must raise
`AssetValidationError` with a field-specific message.

- [ ] **Step 2: Run the loader test and verify import failure**

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
```

Expected: FAIL because the model/function do not exist.

- [ ] **Step 3: Add the dedicated immutable model**

```python
@dataclass(frozen=True)
class PolicyKLExtensionConfig:
    measurement_id: str
    source_measurement_id: str
    source_run_id: str
    source_tree_hash: str
    opponent_id: str
    seeds: tuple[int, ...]
    seats: tuple[int, ...]
    decision_numbers: tuple[int, ...]
    epsilons: tuple[str, ...]
    primary_epsilon: str
    history: tuple[HistoricalPolicyConfig, ...]
```

- [ ] **Step 4: Implement strict v2 parsing without weakening v1**

Add constants for the exact v2 ID, source ID/run/tree hash, and v0-v7 history.
Implement `load_policy_kl_extension_config` using the same TOML type checks as
v1, then require exact equality for every common-domain tuple and every
`(version, run_id, content_hash)` tuple. Reuse a private common parser only if
both public loaders retain their existing exact error behavior.

- [ ] **Step 5: Run v1 and v2 loader tests**

```bash
.venv/bin/python -m pytest tests/generals/test_assets.py -q
```

Expected: PASS, including all existing v1 tests.

- [ ] **Step 6: Commit the Framework loader**

```bash
git add src/agentbench_frame/generals/models.py \
  src/agentbench_frame/generals/assets.py \
  tests/generals/fixtures/policy-kl-reference-v2.toml \
  tests/generals/test_assets.py
git commit -m "feat(generals): load policy KL v7 extension contract"
```

---

### Task 3: Extract deterministic KL fact computation

**Files:**
- Create: `src/agentbench_frame/generals/policy_kl_math.py`
- Create: `tests/generals/test_policy_kl_math.py`
- Modify: `src/agentbench_frame/generals/policy_kl_pipeline.py`
- Test: `tests/generals/test_policy_kl_pipeline.py`

**Interfaces:**
- Consumes: ordered versions/state IDs, exact support counts, canonical deterministic actions, missing reasons, epsilons, primary epsilon, action-space spec ID.
- Produces: `PolicyKLComputation` and the fully typed
  `compute_controlled_policy_kl` function shown in Step 3.

- [ ] **Step 1: Write pure-computation tests**

```python
def test_compute_controlled_policy_kl_preserves_transition_state_epsilon_order():
    result = compute_controlled_policy_kl(
        versions=("v6", "v7"),
        reference_state_ids=("s0", "s1"),
        counts={"s0": 13, "s1": 1050},
        actions={
            ("v6", "s0"): ((5, 1), (8,)),
            ("v7", "s0"): ((5, 1), (8,)),
            ("v6", "s1"): ((5, 1), (8,)),
            ("v7", "s1"): ((5, 2), (8,)),
        },
        missing_actions={},
        epsilons=("0.001", "0.01", "0.05", "0.1"),
        primary_epsilon="0.01",
        action_space_spec_id="spec",
    )
    assert len(result.facts) == 8
    assert result.facts[0]["measurement_state_id"] == "s0"
    assert result.facts[0]["epsilon"] == "0.001"
    assert result.metric["transitions"][0]["coverage"] == {
        "complete": 2, "total": 2
    }
```

Add tests proving identical actions yield exact decimal `"0"`, disagreement
matches `uniform_smoothed_deterministic_kl`, one missing action emits four
missing facts and null aggregate, and `direction == "new||old"` remains
unchanged.

- [ ] **Step 2: Run the math tests and verify missing-module failure**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_math.py -q
```

Expected: FAIL because `policy_kl_math` does not exist.

- [ ] **Step 3: Implement the pure dataclass and function**

```python
@dataclass(frozen=True)
class PolicyKLComputation:
    facts: tuple[dict[str, Any], ...]
    metric: dict[str, Any]
    complete: bool


CanonicalAction = tuple[tuple[int, ...], ...]


def compute_controlled_policy_kl(
    *,
    versions: tuple[str, ...],
    reference_state_ids: tuple[str, ...],
    counts: Mapping[str, int],
    actions: Mapping[tuple[str, str], CanonicalAction],
    missing_actions: Mapping[tuple[str, str], str],
    epsilons: tuple[str, ...],
    primary_epsilon: str,
    action_space_spec_id: str,
) -> PolicyKLComputation:
```

Use the body of the existing `_measure_kl` computation loop, ending with
`return PolicyKLComputation(tuple(facts), metric, all_primary_complete)`.
Move the existing Decimal precision, fact ordering, disagreement counting,
coverage, support summary, and scientific-scope logic without changing field
names or numeric conversion. This is a move, not a rewrite: file/event writes
remain in the pipeline wrapper.

- [ ] **Step 4: Refactor v1 `_measure_kl` to delegate**

Call the pure helper, write each returned fact as the existing event, write
`per-state-kl.jsonl`, write `transition-summary.json`, and return
`(result.metric, result.complete)`. Do not change v1 event or artifact bytes
except for unavoidable JSON formatting already produced by the same code.

- [ ] **Step 5: Run math and v1 pipeline regression tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_policy_kl_math.py \
  tests/generals/test_policy_kl_pipeline.py -q
```

Expected: PASS; v1 still emits six transitions and 288 rows.

- [ ] **Step 6: Commit the math boundary**

```bash
git add src/agentbench_frame/generals/policy_kl_math.py \
  src/agentbench_frame/generals/policy_kl_pipeline.py \
  tests/generals/test_policy_kl_math.py
git commit -m "refactor(generals): isolate controlled policy KL math"
```

---

### Task 4: Verify and materialize the completed v1 source run

**Files:**
- Create: `src/agentbench_frame/generals/policy_kl_reuse.py`
- Create: `tests/generals/test_policy_kl_reuse.py`

**Interfaces:**
- Consumes: `source_run_dir: Path`, `PolicyKLExtensionConfig`.
- Produces: `canonical_tree_hash(root: Path) -> str`,
  `verify_policy_kl_source(source_run_dir: Path, config: PolicyKLExtensionConfig) -> VerifiedPolicyKLSource`, and
  `materialize_policy_kl_source(source: VerifiedPolicyKLSource, destination_run_dir: Path) -> tuple[ArtifactReceipt, ...]`.

- [ ] **Step 1: Build a complete synthetic v1 source fixture and failing tests**

The fixture must contain the exact directory classes used by the real run:
benchmark specs, 12 state files, 12 count files, v0-v6 source/manifests and 84
probe files, 288 facts, 396 events, transition summary, quality, summary, and a
non-imported cache file.

```python
def test_canonical_tree_hash_is_path_and_content_sensitive(tmp_path):
    root = tmp_path / "source"
    (root / "b").mkdir(parents=True)
    (root / "a.txt").write_text("one", encoding="utf-8")
    (root / "b/z.txt").write_text("two", encoding="utf-8")
    first = canonical_tree_hash(root)
    (root / "b/z.txt").write_text("changed", encoding="utf-8")
    assert canonical_tree_hash(root) != first


def test_verify_rejects_one_mutated_count(complete_source, extension_config):
    count = next((complete_source / "action-space/counts").glob("*.json"))
    count.write_text(count.read_text().replace('"complete"', '"broken"', 1))
    with pytest.raises(PolicyKLSourceError, match="count"):
        verify_policy_kl_source(complete_source, extension_config)
```

Add independent failures for wrong tree hash, incomplete summary, dirty event
quality, altered action-space spec ID, wrong state hash/coordinates, duplicate
state, missing count, invalid v0-v6 manifest provenance, nondeterministic or
empty canonical action, missing/duplicate fact coordinate, mismatched source
event, and transition-summary mismatch.

- [ ] **Step 2: Run the reuse tests and verify missing-module failure**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_reuse.py -q
```

Expected: FAIL because the verifier does not exist.

- [ ] **Step 3: Implement canonical tree hashing**

```python
def canonical_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise PolicyKLSourceError("source path contains a newline")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{file_hash}  ./{relative}\n".encode("utf-8"))
    return digest.hexdigest()
```

Add a real-run integration assertion guarded by path existence that the
result is exactly the frozen
`6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225`
hash.

- [ ] **Step 4: Implement strict source verification**

```python
@dataclass(frozen=True)
class VerifiedPolicyKLSource:
    root: Path
    tree_hash: str
    reference_records: tuple[dict[str, Any], ...]
    counts: dict[str, int]
    actions: dict[tuple[str, str], CanonicalAction]
    prior_facts: tuple[dict[str, Any], ...]
    reuse_events: tuple[dict[str, Any], ...]
    kl_source_events: dict[tuple[str, str, str, str], dict[str, Any]]
    prior_metric: dict[str, Any]
```

Define `PolicyKLSourceError(ValueError)`. Define
`verify_policy_kl_source(source_run_dir: Path, config: PolicyKLExtensionConfig)
-> VerifiedPolicyKLSource` immediately below the dataclass.

Require exact counts `12/12/84/288/396`, unique coordinate sets, all complete
statuses, valid state IDs via `measurement_state_id`, positive decimal support
strings, approved v0-v6 run/hash manifests, exact action-space spec ID, exact
six-transition metric equality between summary and measurement artifact, and
zero values in every event-quality defect field. Strip only Framework event
metadata when comparing source events to their artifact facts.

- [ ] **Step 5: Implement idempotent artifact materialization**

```python
@dataclass(frozen=True)
class ArtifactReceipt:
    source_relative_path: str
    destination_relative_path: str
    semantic_role: str
    sha256: str
```

Define `materialize_policy_kl_source(source: VerifiedPolicyKLSource,
destination_run_dir: Path) -> tuple[ArtifactReceipt, ...]` immediately below
the dataclass.

Copy the action spec, 12 states, 12 counts, and v0-v6 policy directories to
standard paths. Copy both source specs, old per-state file, old transition
summary, source events, quality, and summary to `provenance/source-run/`.
For an existing destination, verify identical bytes instead of overwriting.
Write sorted receipts to `provenance/imported-artifacts.jsonl`; exclude cache,
prepared opponents, and replays. Recompute every destination SHA-256 before
returning.

- [ ] **Step 6: Run reuse tests**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_reuse.py -q
```

Expected: PASS, including a before/after source tree-hash equality assertion.

- [ ] **Step 7: Commit the verifier/materializer**

```bash
git add src/agentbench_frame/generals/policy_kl_reuse.py \
  tests/generals/test_policy_kl_reuse.py
git commit -m "feat(generals): verify reusable policy KL source run"
```

---

### Task 5: Orchestrate the append-only v7 extension run

**Files:**
- Create: `src/agentbench_frame/generals/policy_kl_extension.py`
- Create: `tests/generals/test_policy_kl_extension.py`
- Modify: `src/agentbench_frame/tracking/quality.py`
- Test: `tests/generals/test_policy_kl_pipeline.py`

**Interfaces:**
- Consumes: v2 config, verified/materialized v1 source, existing historical-policy resolver/probe, and pure KL computation.
- Produces: `GeneralsPolicyKLExtensionPipeline.from_paths` with the exact Step
  6 signature, `.run()`, `.recover(failed_run)`, a self-contained v2 run, and
  known provenance events.

- [ ] **Step 1: Write a fake complete extension test**

Use fakes for source verification, materialization, v7 resolution, and probing;
make any attempt to resolve/probe v0-v6 fail the test.

```python
def test_extension_probes_only_v7_and_emits_complete_projection(tmp_path):
    pipeline, probes = make_fake_extension(tmp_path)
    result = pipeline.run()
    metric = result.summary["controlled_reference_policy_kl"]
    assert result.status == "complete"
    assert probes == [("v7", state_id) for state_id in EXPECTED_STATE_IDS]
    assert len(metric["transitions"]) == 7
    assert metric["transitions"][-1]["version_before"] == "v6"
    assert metric["transitions"][-1]["version_after"] == "v7"
    assert all(
        point["coverage"] == {"complete": 12, "total": 12}
        for point in metric["transitions"]
    )
    assert len((result.run_dir / "measurement/per-state-kl.jsonl")
               .read_text().splitlines()) == 336
```

Assert 396 imported events have new event/run IDs plus source IDs and
`verified_reuse`, exactly 12 new v7 action events and 48 new v6→v7 events use
`reuse_status == "new"`, both provenance event types are known, and event
quality has zero defects.

- [ ] **Step 2: Add failing incomplete and recovery tests**

Test one nondeterministic v7 state yields four missing v6→v7 facts, 11/12
coverage, null aggregate, and `incomplete_policy_measurement`. Simulate a stop
after imported events, recover the same run, and assert old event bytes remain
a prefix, deterministic event IDs prevent duplicates, and completion occurs.
Reject recovery of complete/wrong-measurement/wrong-source-tree runs.

- [ ] **Step 3: Run extension tests and verify missing-module failure**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_extension.py -q
```

Expected: FAIL because the extension pipeline does not exist.

- [ ] **Step 4: Register provenance event types**

Add exactly:

```python
"policy_kl_source_verified",
"policy_kl_artifact_reused",
```

to `KNOWN_EVENT_TYPES`.

- [ ] **Step 5: Implement deterministic append-only event emission**

```python
def extension_event_id(run_id: str, kind: str, key: str) -> str:
    value = hashlib.sha256(
        f"{run_id}\0{kind}\0{key}".encode("utf-8")
    ).hexdigest()
    return f"evt_{value[:32]}"
```

Load existing event IDs during recovery. Emit only absent deterministic IDs.
For imported events, remove old Framework metadata, preserve measurement
fields, add `source_event_id`, `source_run_id`, and
`reuse_status="verified_reuse"`. Remap artifact paths to the new run.

- [ ] **Step 6: Implement the extension pipeline**

```python
class GeneralsPolicyKLExtensionPipeline(GeneralsPolicyKLPipeline):
    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        reference_manifest_path: Path,
        source_run_dir: Path,
        data_dir: Path,
    ) -> "GeneralsPolicyKLExtensionPipeline":

    def run(self) -> PolicyKLMeasurementResult:

    def recover(self, failed_run: Path) -> PolicyKLMeasurementResult:
```

Implement all three method bodies in this task; the execution order below is
their normative algorithm, and no method is committed as a stub.

Execution order is fixed:

1. verify source tree;
2. materialize artifacts and receipts;
3. write v2 reference-state spec from verified coordinates/state IDs;
4. re-emit the 108 imported non-KL scientific events and provenance receipts;
5. resolve only `config.history[-1]` and verify v7 source hash;
6. probe v7 twice on 12 states, canonicalize, persist and emit 12 actions;
7. combine imported v0-v6 actions with v7 actions;
8. compute all 336 facts, byte/field-compare the first 288 to source facts,
   attach source event IDs, and emit/write them;
9. require the first six aggregates to equal the source metric;
10. recompute source tree hash and finish with source ID/hash/reuse mode in
    summary.

Write `provenance/source-run-receipt.json` containing source ID, directory,
pre/post tree hashes, receipt count, and `verified=true`.

- [ ] **Step 7: Run extension, v1 regression, and quality tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_policy_kl_extension.py \
  tests/generals/test_policy_kl_pipeline.py \
  tests/test_tracking_contracts.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the extension engine**

```bash
git add src/agentbench_frame/generals/policy_kl_extension.py \
  src/agentbench_frame/tracking/quality.py \
  tests/generals/test_policy_kl_extension.py
git commit -m "feat(generals): extend controlled policy KL through v7"
```

---

### Task 6: Add explicit extension CLI commands

**Files:**
- Modify: `src/agentbench_frame/generals/cli.py`
- Modify: `tests/generals/test_policy_kl_cli.py`

**Interfaces:**
- Consumes: `GeneralsPolicyKLExtensionPipeline` from Task 5.
- Produces: `generals extend-policy-kl-v7` and `generals recover-policy-kl-v7` with explicit `--source-run`.

- [ ] **Step 1: Write parser and routing tests**

```python
def extension_arguments(command):
    values = [
        "generals", command,
        "--agentbench-root", "/assets",
        "--manifest", "/benchmark/pilot.toml",
        "--reference-manifest", "/benchmark/policy-kl-reference-v2.toml",
        "--source-run", "/data/runs/v1",
        "--data-dir", "/data",
    ]
    if command == "recover-policy-kl-v7":
        values += ["--failed-run", "/data/runs/v2-incomplete"]
    return values


def test_extend_policy_kl_v7_requires_source_run():
    args = parser().parse_args(extension_arguments("extend-policy-kl-v7"))
    assert args.source_run == Path("/data/runs/v1")
```

Route tests must monkeypatch `GeneralsPolicyKLExtensionPipeline`, assert no
Codex provider is constructed, assert `.run()` versus `.recover()`, and check
the JSON result includes status, run directory, source run, and metric.

- [ ] **Step 2: Run CLI tests and verify unknown-command failure**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_cli.py -q
```

Expected: FAIL because the new commands are not registered.

- [ ] **Step 3: Register and route both commands**

Add help text that says “reuse verified v1 and probe only v7”. Require
`--reference-manifest`, `--source-run`, and `--data-dir`; recovery additionally
requires `--failed-run`. Do not expose exact-count limits because v2 never
counts supports.

- [ ] **Step 4: Run CLI tests**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_cli.py -q
```

Expected: PASS for old and new commands.

- [ ] **Step 5: Commit the CLI**

```bash
git add src/agentbench_frame/generals/cli.py \
  tests/generals/test_policy_kl_cli.py
git commit -m "feat(generals): expose policy KL v7 extension CLI"
```

---

### Task 7: Support strict v1/v2 paper-figure contracts

**Files:**
- Modify: `src/agentbench_frame/generals/paper_figure.py`
- Modify: `tests/generals/test_policy_kl_figure.py`

**Interfaces:**
- Consumes: finalized v1 or v2 run directory.
- Produces: unchanged `PolicyKLFigureData` and `render_policy_kl_three_panel`, with transitions selected strictly by measurement ID.

- [ ] **Step 1: Keep the v1 fixture explicit and add a v2 fixture**

Add `measurement_id="generals-policy-kl-reference-v1"` to the existing fake
summary. Parameterize its transition builder, then create v2 with:

```python
TRANSITIONS_V2 = TRANSITIONS_V1 + (("v6", "v7"),)


def test_loads_complete_v2_figure_data(tmp_path):
    data = load_policy_kl_figure_data(write_complete_run(tmp_path, version=2))
    assert data.transitions[-1] == "v6→v7"
    assert len(data.primary_kl) == 7
    assert len(data.sensitivity["0.01"]) == 7
    assert len(data.support_states) == 12
```

Add tests that a valid v1 six-transition run still loads, a v2 six-transition
run fails, a v1 seven-transition run fails, and an unknown measurement ID
fails. Extend SVG assertions to contain `v6→v7` for v2.

- [ ] **Step 2: Run figure tests and verify v2 rejection**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_figure.py -q
```

Expected: new v2 test FAILS with transition-order validation.

- [ ] **Step 3: Implement measurement-ID dispatch**

```python
EXPECTED_VERSION_PAIRS_BY_MEASUREMENT = {
    "generals-policy-kl-reference-v1": (
        ("v0", "v1"), ("v1", "v2"), ("v2", "v3"),
        ("v3", "v4"), ("v4", "v5"), ("v5", "v6"),
    ),
    "generals-policy-kl-reference-v2": (
        ("v0", "v1"), ("v1", "v2"), ("v2", "v3"),
        ("v3", "v4"), ("v4", "v5"), ("v5", "v6"),
        ("v6", "v7"),
    ),
}
```

Read `summary["measurement_id"]`, select the exact tuple, and use it for all
pair, label, and length checks. Preserve epsilon and 12-state validation.

- [ ] **Step 4: Run figure tests**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_figure.py -q
```

Expected: PASS; PNG remains at least 6000×1800.

- [ ] **Step 5: Commit figure compatibility**

```bash
git add src/agentbench_frame/generals/paper_figure.py \
  tests/generals/test_policy_kl_figure.py
git commit -m "feat(generals): plot controlled policy KL through v7"
```

---

### Task 8: Verify both repositories before real execution

**Files:**
- No new production files.
- Test all files changed in Tasks 1-7.

**Interfaces:**
- Consumes: complete implementation and frozen assets.
- Produces: clean focused/full-suite receipts and clean worktrees except planned commits.

- [ ] **Step 1: Run focused Framework tests**

```bash
.venv/bin/python -m pytest \
  tests/generals/test_assets.py \
  tests/generals/test_policy_kl_math.py \
  tests/generals/test_policy_kl_reuse.py \
  tests/generals/test_policy_kl_pipeline.py \
  tests/generals/test_policy_kl_extension.py \
  tests/generals/test_policy_kl_cli.py \
  tests/generals/test_policy_kl_figure.py \
  tests/test_tracking_contracts.py -q
```

Expected: all PASS.

- [ ] **Step 2: Run the complete Framework suite**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
```

Expected: all tests PASS; only pre-existing documented skips are allowed.

- [ ] **Step 3: Run the complete Generals assets suite**

From the assets worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python \
  -m pytest backend_sources/corpus/28_generals/tests -q
```

Expected: all tests PASS.

- [ ] **Step 4: Audit status and commit only test-driven corrections**

```bash
git status --short
git log --oneline -8
```

If verification required a source correction, commit only that correction and
its regression test with a narrowly scoped message. Do not squash the design
or task commits.

---

### Task 9: Run the real extension, audit it, and regenerate the figure

**Files:**
- Modify: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg`
- Modify: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.png`
- Create: `docs/experiments/2026-08-01-generals-controlled-policy-kl-v7-result.md`
- Runtime output: `agentbench_data/runs/28_generals/generals-policy-kl/<new-run-id>/`

**Interfaces:**
- Consumes: real v2 asset, immutable v1 measurement run, retained v7 source.
- Produces: real v0-v7 run, audited v6→v7 values, updated SVG/PNG, and reproduction note.

- [ ] **Step 1: Record the immutable source hash immediately before execution**

```bash
find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

Run inside the completed v1 run. Expected:

```text
6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225  -
```

- [ ] **Step 2: Execute the real v7 extension**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m agentbench_frame.cli \
  generals extend-policy-kl-v7 \
  --agentbench-root /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  --manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --reference-manifest /home/cathy/AgentBench/AgentBench/.worktrees/generals-assets/backend_sources/corpus/28_generals/benchmark/policy-kl-reference-v2.toml \
  --source-run /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data/runs/28_generals/generals-policy-kl/20260730_1126_8d123b55 \
  --data-dir /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/agentbench_data
```

Expected: exit 0 and JSON `status="complete"`. If interrupted, use
`recover-policy-kl-v7 --failed-run <new-run-dir>` with the same other inputs;
never edit or delete the incomplete run.

- [ ] **Step 3: Audit the finalized run programmatically**

Run a read-only script/assertion that verifies:

```python
assert summary["status"] == "complete"
assert summary["measurement_id"] == "generals-policy-kl-reference-v2"
assert summary["source_run_id"] == "20260730_1126_8d123b55"
assert len(metric["transitions"]) == 7
assert metric["transitions"][:6] == source_metric["transitions"]
assert metric["transitions"][-1]["coverage"] == {"complete": 12, "total": 12}
assert len(per_state_rows) == 336
assert event_quality == {
    "total_lines": len(events),
    "valid_events": len(events),
    "malformed_lines": 0,
    "invalid_events": 0,
    "unknown_event_types": 0,
    "duplicate_event_ids": 0,
    "missing_event_ids": 0,
    "missing_run_ids": 0,
    "warnings": [],
}
```

Also assert exactly 12 v7 action events, 48 v6→v7 KL events, 396 verified
reuse events, and matching pre/post source hashes in the provenance receipt.

- [ ] **Step 4: Recompute the source hash after execution**

Repeat Step 1. Expected: the same
`6203161e1056628d46bb974ed2ffaaf1b2d6a67581e80b57f0d92afa2bb41225`
hash.

- [ ] **Step 5: Render the updated paper figure**

```bash
MPLCONFIGDIR=/tmp/agentbench-matplotlib \
  .venv/bin/python scripts/plot_generals_controlled_policy_kl.py \
  --run-dir <new-run-dir> \
  --output-prefix docs/experiments/figures/generals-controlled-policy-kl-three-panel
```

Expected: one editable SVG and one 300-DPI PNG. Open the PNG with the local
image viewer and verify: seven legible transition labels, v6→v7 in panels (a)
and (b), unchanged 12 support bars in panel (c), no clipped title/legend/value,
and no v8 label.

- [ ] **Step 6: Write the result note from first-hand artifacts**

Record exact run ID, source tree hash, v7 source hash, primary v6→v7 KL,
four-epsilon sensitivity, disagreement rate, 12/12 coverage, unchanged first
six aggregates, support-size range, event counts/quality, source immutability,
scientific limitations, and exact reproduction/render commands. Call the
metric only `controlled_reference_policy_kl` or deterministic policy-change
KL, never epistemic information gain.

- [ ] **Step 7: Run final verification**

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_figure.py \
  tests/generals/test_policy_kl_extension.py -q
git diff --check
git status --short
```

Expected: tests PASS and only the result note/figure outputs are uncommitted.

- [ ] **Step 8: Commit the real result and figure**

```bash
git add docs/experiments/2026-08-01-generals-controlled-policy-kl-v7-result.md \
  docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg \
  docs/experiments/figures/generals-controlled-policy-kl-three-panel.png
git commit -m "docs: report Generals controlled policy KL through v7"
```

---

## Completion Evidence

Before claiming completion, report:

- both repository branches and final commits;
- focused/full test counts and skips;
- old v1 pre/post canonical tree hash;
- new v2 run ID and status;
- exact v6→v7 primary KL, epsilon sensitivity, disagreement rate, and coverage;
- equality of all six old transition aggregates;
- event and per-state row counts/quality;
- clickable paths to the new run summary, result note, SVG, and PNG;
- visual-inspection outcome and the limitation that 12 controlled early states
  do not prove global policy identity or benchmark improvement.
