# 29 Rollman Opponent Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `weakest_failed` HL curriculum that imports `run-20260731-gpt55-sota/v000001`, learns against rank15 and then rank14, preserves every locked opponent, and stops only at 16/16.

**Architecture:** Keep the fixed-rank1 workflow as the default. Add strict origin and curriculum configuration, a content-verified snapshot import boundary, and a focused `CurriculumManager` whose state is rebuilt from append-only events. The Rollman CLI coordinates target-specific gates, all-pool certification, stage promotion, immediate regression rollback, stagnation pauses, and report generation without exposing opponent source to the coding model.

**Tech Stack:** Python 3.9+, dataclasses, argparse, PyYAML, pytest/unittest, JSONL event sourcing, matplotlib, existing Codex CLI Responses provider.

## Global Constraints

- Source code snapshot: `run-20260731-gpt55-sota/v000001`.
- First target: `rank15`; target order is the largest rank among failed opponents.
- Completion requires all 16 frozen human opponents to pass at `required_win_rate`.
- Initial `candidates_per_act` is 1 and remains a configuration-only ablation.
- No total act cap; four consecutive non-improving acts pause as `stagnated`.
- A candidate may lock a target only if every locked opponent still passes.
- Imported runs use a fresh provider session and a fresh Experience Skill.
- Primitive actions and KL semantics remain unchanged.
- Human source code, API keys, seed-specific code, fixed replay coordinates, unsupported tactical labels, and blind grid search are forbidden.
- Fixed mode retains the rank1 bootstrap, evaluation, rollback, resume, and reporting behavior.
- Every implementation step begins with a failing test and ends with a focused commit.

---

## File Structure

- Create `src/agentbench_frame/hl/curriculum.py`: certification summaries, target selection, stage state, event replay, promotion, rejection, and stagnation decisions.
- Modify `src/agentbench_frame/hl/config.py`: `OriginConfig` and `CurriculumConfig`.
- Modify `src/agentbench_frame/hl/local_config.py`: repository-root resolution for `origin.source_run`.
- Modify `src/agentbench_frame/hl/codebase.py`: verified import of an immutable version into a fresh version store.
- Modify `src/agentbench_frame/hl/events.py`: strict schemas for curriculum lifecycle events.
- Modify `src/agentbench_frame/games/rollman/evaluator.py`: explicit target replacement without rebuilding game infrastructure.
- Modify `src/agentbench_frame/hl/lineage.py`: stage champion reset and explicit safe-parent rollback.
- Modify `src/agentbench_frame/hl/controller.py`: imported-origin registration and externally requested safe parent.
- Modify `src/agentbench_frame/hl/context.py`: target-aware incremental prompt.
- Modify `src/agentbench_frame/hl/cli.py`: imported start, target loop, certification, pause, resume, and completion.
- Modify `src/agentbench_frame/hl/report.py`: curriculum table and curve annotations.
- Create `configs/hl/29_rollman-curriculum.yaml`: the approved local experiment.
- Modify `docs/hl/rollman-local-run.md`: reproducible curriculum commands and statuses.
- Modify focused tests under `tests/hl/` and `tests/rollman/`.

### Task 1: Strict Origin and Curriculum Configuration

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/hl/local_config.py`
- Modify: `tests/hl/test_config.py`
- Create: `tests/hl/test_local_config.py`

**Interfaces:**
- Produces: `OriginConfig(mode, source_run, source_version, reset_session, reset_experience)`.
- Produces: `CurriculumConfig(mode, target_order, preserve_passed_opponents, required_human_opponents, stagnation_patience)`.
- Produces: `HLRunConfig.origin` and `HLRunConfig.curriculum`.
- Produces: `LocalHLConfig.load()` with an absolute `run.origin.source_run` for imported runs.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_imported_curriculum_config_is_strict_and_resolves_source_run():
    config = LocalHLConfig.load(CURRICULUM_CONFIG)
    assert config.run.origin.mode == "imported_version"
    assert Path(config.run.origin.source_run).is_absolute()
    assert config.run.origin.source_version == "v000001"
    assert config.run.curriculum.mode == "weakest_failed"
    assert config.run.curriculum.required_human_opponents == 16
    assert config.run.curriculum.stagnation_patience == 4


def test_imported_origin_requires_run_and_version():
    with pytest.raises(ValueError, match="source_run"):
        HLRunConfig.from_mapping({
            "game": "29_rollman",
            "provider": {"kind": "codex"},
            "origin": {"mode": "imported_version"},
        })


def test_fixed_defaults_do_not_change_rank1_workflow():
    config = HLRunConfig.from_mapping({
        "game": "29_rollman",
        "provider": {"kind": "codex"},
    })
    assert config.origin.mode == "model_bootstrap"
    assert config.curriculum.mode == "fixed"
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_config.py tests/hl/test_local_config.py
```

Expected: failures because `OriginConfig`, `CurriculumConfig`, and curriculum path resolution do not exist.

- [ ] **Step 3: Implement minimal strict dataclasses**

Add frozen dataclasses with exact validations:

```python
@dataclasses.dataclass(frozen=True)
class OriginConfig:
    mode: str = "model_bootstrap"
    source_run: Optional[str] = None
    source_version: Optional[str] = None
    reset_session: bool = True
    reset_experience: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"model_bootstrap", "imported_version"}:
            raise ValueError("origin.mode must be model_bootstrap or imported_version")
        if self.mode == "imported_version" and (
            not self.source_run or not self.source_version
        ):
            raise ValueError(
                "imported_version origin requires source_run and source_version"
            )
        if self.mode == "model_bootstrap" and (
            self.source_run is not None or self.source_version is not None
        ):
            raise ValueError("model_bootstrap origin cannot define a source")


@dataclasses.dataclass(frozen=True)
class CurriculumConfig:
    mode: str = "fixed"
    target_order: str = "lowest_rank_first"
    preserve_passed_opponents: bool = True
    required_human_opponents: int = 16
    stagnation_patience: int = 4
```

Validate `mode`, `target_order`, the 1..16 requirement, positive patience, and require `preserve_passed_opponents=True` for `weakest_failed`. Extend `_strict_values` construction in `HLRunConfig.from_mapping()`.

Resolve a relative `source_run` against `source.parents[2]` in `LocalHLConfig.load()` with nested `dataclasses.replace()`.

- [ ] **Step 4: Run the focused tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_config.py tests/hl/test_local_config.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/config.py src/agentbench_frame/hl/local_config.py \
  tests/hl/test_config.py tests/hl/test_local_config.py
git commit -m "feat: configure imported HL curricula"
```

### Task 2: Content-Verified Imported Origin

**Files:**
- Modify: `src/agentbench_frame/hl/codebase.py`
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `tests/hl/test_codebase.py`
- Modify: `tests/hl/test_controller.py`

**Interfaces:**
- Produces: `VersionStore.import_version(source_versions_root, source_version_id, act_id="imported-origin") -> tuple[Version, Version]`.
- Produces: `HLController.initialize_imported(source_versions_root, source_version_id) -> Version`.
- Consumes: existing `_verify_object`, `_read_tree`, `VersionStore.snapshot()`, and event writer.

- [ ] **Step 1: Write failing import tests**

```python
def test_import_version_restores_exact_content_and_creates_imported_origin(tmp_path):
    source_workspace = _workspace(tmp_path / "source")
    source_store = VersionStore(source_workspace, tmp_path / "source-versions")
    source = source_store.snapshot(parent_version_id=None, act_id="source")

    target_workspace = _workspace(tmp_path / "target")
    (target_workspace / "agent.py").write_text("VALUE = 999\n")
    target_store = VersionStore(target_workspace, tmp_path / "target-versions")
    imported_source, imported = target_store.import_version(
        tmp_path / "source-versions", source.version_id
    )

    assert imported_source.content_hash == source.content_hash
    assert imported.version_id == "v000000"
    assert imported.content_hash == source.content_hash
    assert imported.parent_version_id is None
    assert imported.edit_type == "imported_origin"
    assert target_store.current_content_hash() == source.content_hash


def test_import_rejects_tampered_source_object(tmp_path):
    # Create a source version, alter its immutable object, and require fail-closed.
    with pytest.raises(ValueError, match="content hash mismatch"):
        target_store.import_version(source_versions_root, "v000000")
```

Add a controller test asserting zero provider calls and one `origin_imported` event with source run id, source version, source hash, and imported version.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_codebase.py tests/hl/test_controller.py
```

Expected: failures because the import methods and event type do not exist.

- [ ] **Step 3: Implement verified import**

Implement a read-only source manifest loader that:

```python
source_manifest = source_versions_root / "manifests" / f"{source_version_id}.json"
source = _load_version_manifest(source_manifest, source_version_id)
source_object = source_versions_root / "objects" / source.content_hash
_verify_object(
    source_object,
    content_hash=source.content_hash,
    expected_files=source.files,
)
```

Clear only non-ignored target workspace entries, copy verified bytes from the source object, snapshot with `parent_version_id=None`, and preserve `edit_type="imported_origin"`. Adjust `snapshot()` so only its default candidate edit type maps a root version to `"initial"`.

Implement `HLController.initialize_imported()` to write `run_started`, import and register the root as incomplete, write `version_created`, `candidate_selected`, and `origin_imported`, set `_started=True`, and never invoke the provider.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_codebase.py tests/hl/test_controller.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/codebase.py src/agentbench_frame/hl/controller.py \
  tests/hl/test_codebase.py tests/hl/test_controller.py
git commit -m "feat: import verified HL origins"
```

### Task 3: Event-Sourced Curriculum State

**Files:**
- Create: `src/agentbench_frame/hl/curriculum.py`
- Modify: `src/agentbench_frame/hl/events.py`
- Create: `tests/hl/test_curriculum.py`
- Modify: `tests/hl/test_events.py`

**Interfaces:**
- Produces: `summarize_certification(matches, required_win_rate) -> CertificationSummary`.
- Produces: `select_weakest_failed(summary) -> Optional[str]`.
- Produces: `CurriculumDecision(kind, parent_version_id, next_target, lost_locked_opponents)`.
- Produces: `CurriculumManager.start(...)`, `CurriculumManager.from_events(...)`, `observe_gate(...)`, and `observe_certification(...)`.

- [ ] **Step 1: Write failing target and state tests**

```python
def _certification_matches(*, failed: tuple[int, ...]) -> list[dict[str, object]]:
    records = []
    for rank in range(1, 17):
        for seed in (201, 202, 203):
            records.append({
                "status": "complete",
                "opponent": f"rank{rank:02d}",
                "opponent_rank": rank,
                "seed": seed,
                "result": "loss" if rank in failed else "win",
            })
    return records


def _started_manager() -> CurriculumManager:
    return CurriculumManager.start(
        version_id="v000000",
        summary=summarize_certification(
            _certification_matches(failed=(14, 15)), 0.5
        ),
        required_human_opponents=16,
        stagnation_patience=4,
    )


def test_selects_lowest_rank_failed_opponent():
    summary = summarize_certification(
        matches=_certification_matches(failed=(14, 15)),
        required_win_rate=0.5,
    )
    assert summary.passing_opponents == 14
    assert select_weakest_failed(summary) == "rank15"


def test_promotes_rank15_then_selects_rank14():
    manager = _started_manager()
    decision = manager.observe_certification(
        version_id="v000003",
        summary=summarize_certification(
            _certification_matches(failed=(14,)), 0.5
        ),
    )
    assert decision.kind == "promote"
    assert decision.next_target == "rank14"
    assert "rank15" in manager.locked_opponents


def test_rejects_target_win_that_loses_a_locked_opponent():
    manager = _started_manager()
    decision = manager.observe_certification(
        version_id="v000004",
        summary=summarize_certification(
            _certification_matches(failed=(3, 14)), 0.5
        ),
    )
    assert decision.kind == "reject"
    assert decision.parent_version_id == manager.stage_origin_version_id
    assert decision.lost_locked_opponents == ("rank03",)


def test_four_non_improving_gates_pause_and_resume_identically():
    manager = _started_manager()
    for version_id in ("v1", "v2", "v3", "v4"):
        decision = manager.observe_gate(version_id=version_id, score=0.0)
    assert decision.kind == "stagnated"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_curriculum.py tests/hl/test_events.py
```

Expected: failures because curriculum state and strict event schemas do not exist.

- [ ] **Step 3: Implement certification summary and manager**

Use frozen dataclasses:

```python
@dataclasses.dataclass(frozen=True)
class CertificationSummary:
    pass_rates: Mapping[str, float]
    ranks: Mapping[str, int]
    passing_opponents: int
    failed_opponents: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CurriculumDecision:
    kind: str
    parent_version_id: str
    next_target: Optional[str] = None
    lost_locked_opponents: tuple[str, ...] = ()
```

Calculate rates as `(wins + 0.5 * draws) / complete_cases`. Reject missing or incomplete opponents. Select `max(failed_opponents, key=lambda id: ranks[id])`.

The manager must:

- initialize locked opponents from the complete source certification;
- set `stage_origin_version_id` and `stage_best_version_id` to the imported version;
- count only strict gate-score improvements;
- reset stagnation on improvement or stage promotion;
- reject a certification that loses any locked opponent;
- return `complete` only when `passing_opponents == required_human_opponents`;
- rebuild all fields from curriculum events.

Add a separate replay test that writes `curriculum_started`, four
`curriculum_gate_completed`, and one `curriculum_stagnated` event through
`HLEventWriter`, then asserts:

```python
rebuilt = CurriculumManager.from_events(
    read_events(events_path),
    required_human_opponents=16,
    stagnation_patience=4,
)
assert rebuilt.state == manager.state
```

Add strict schemas for `origin_imported`, `curriculum_started`, `curriculum_target_selected`, `curriculum_gate_completed`, `curriculum_stage_promoted`, `curriculum_candidate_rejected`, and `curriculum_stagnated`. Register list-valued fields such as `locked_opponents`, `lost_locked_opponents`, and `matches`.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_curriculum.py tests/hl/test_events.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/curriculum.py src/agentbench_frame/hl/events.py \
  tests/hl/test_curriculum.py tests/hl/test_events.py
git commit -m "feat: track opponent curriculum state"
```

### Task 4: Target Switching and Safe Parent Control

**Files:**
- Modify: `src/agentbench_frame/games/rollman/evaluator.py`
- Modify: `src/agentbench_frame/hl/lineage.py`
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `tests/rollman/test_evaluator.py`
- Modify: `tests/hl/test_lineage.py`
- Modify: `tests/hl/test_controller.py`

**Interfaces:**
- Produces: `RollmanEvaluator.set_learning_opponent(opponent: Opponent) -> None`.
- Produces: `LineageManager.begin_stage(version_id: str, score: float) -> None`.
- Produces: `LineageManager.force_parent(version_id: str) -> ParentDecision`.
- Extends: `HLController.run_act(parent_version_id: Optional[str] = None)`.

- [ ] **Step 1: Write failing evaluator and rollback tests**

```python
def test_learning_target_can_switch_without_rebuilding_evaluator():
    evaluator.set_learning_opponent(rank15)
    first = evaluator.evaluate(version)
    evaluator.set_learning_opponent(rank14)
    second = evaluator.evaluate(version)
    assert {match["opponent"] for match in first.matches} == {"rank15"}
    assert {match["opponent"] for match in second.matches} == {"rank14"}


def test_begin_stage_resets_noncomparable_champion_score():
    lineage.begin_stage("v000003", 0.0)
    assert lineage.champion_version_id == "v000003"
    candidate = _register_and_select(lineage, "v000004", parent="v000003", score=1/3)
    assert lineage.champion_version_id == "v000004"


def test_forced_parent_makes_next_act_start_from_stage_origin(tmp_path):
    result = controller.run_act(parent_version_id="v000000")
    assert result.parent_version_id == "v000000"
    assert result.rollback.reason == "curriculum_regression"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest -q tests/rollman/test_evaluator.py tests/hl/test_lineage.py \
  tests/hl/test_controller.py
```

Expected: failures because target switching and stage-parent APIs do not exist.

- [ ] **Step 3: Implement the minimal APIs**

`set_learning_opponent()` validates `opponent.process is not None` and assigns it. `begin_stage()` requires an existing complete version, replaces champion id/score, resets degradation and rollback flags, and moves the lineage head to the stage origin.

`force_parent()` returns:

```python
ParentDecision(
    rollback=version_id != self.lineage_head_version_id,
    from_version_id=self.lineage_head_version_id,
    to_version_id=version_id,
    reason="curriculum_regression",
)
```

It resets degradation state and moves the head. `HLController.run_act(parent_version_id=...)` uses `force_parent()` when supplied; fixed mode calls the method with `None` and retains `select_next_parent()`.

- [ ] **Step 4: Run focused tests and verify pass**

Run:

```bash
pytest -q tests/rollman/test_evaluator.py tests/hl/test_lineage.py \
  tests/hl/test_controller.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/games/rollman/evaluator.py \
  src/agentbench_frame/hl/lineage.py src/agentbench_frame/hl/controller.py \
  tests/rollman/test_evaluator.py tests/hl/test_lineage.py \
  tests/hl/test_controller.py
git commit -m "feat: switch curriculum targets safely"
```

### Task 5: Target-Aware Incremental Prompt

**Files:**
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `tests/hl/test_context.py`

**Interfaces:**
- Extends: `IterationContext.build_prompt(..., active_target: Optional[str], locked_opponents: Sequence[str]) -> str`.
- Keeps: `build_bootstrap_prompt()` unchanged for fixed model-bootstrap mode.

- [ ] **Step 1: Write a failing prompt test**

```python
def test_curriculum_prompt_contains_only_active_target_evidence(bundle, tmp_path):
    prompt = IterationContext(bundle).build_prompt(
        act_id="act-2",
        branch_index=0,
        branch_count=1,
        parent_version_id="v000001",
        workspace=tmp_path,
        replay_evidence=[
            {"opponent": "rank15", "seed": 101, "replay": "/r15.jsonl"}
        ],
        previous_measurements={"benchmark_score": 0.0},
        experience_path=experience,
        active_target="rank15",
        locked_opponents=("rank01", "rank16"),
    )
    assert "当前学习目标：rank15" in prompt
    assert "rank01" in prompt and "rank16" in prompt
    assert "固定坐标" in prompt
    assert "grid search" in prompt
    assert "rank14" not in prompt
```

Add a failing test that rejects replay evidence whose opponent differs from `active_target`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_context.py
```

Expected: failure because target metadata and evidence filtering are absent.

- [ ] **Step 3: Implement target prompt fields**

When `active_target` is not `None`:

- validate every evidence row has that opponent;
- include the active target and locked-opponent set;
- demand one falsifiable level/round/event diagnosis;
- permit interpretable condition handling, state machines, planning, graph search, and finite memory;
- require replacement/compression of superseded branches;
- forbid seed identity, fixed coordinates, opponent-source access, unsupported tactical labels, blind threshold enumeration, and grid search.

Serialize only the target replay index and incremental measurements; retain hashed file references for rules, action space, Replay Skill, and Experience.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_context.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/context.py tests/hl/test_context.py
git commit -m "feat: prompt target-specific replay learning"
```

### Task 6: Integrate the Curriculum Run Loop

**Files:**
- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `tests/hl/test_cli.py`
- Create: `tests/hl/test_curriculum_cli.py`

**Interfaces:**
- Consumes: `HLController.initialize_imported()`, `CurriculumManager`, evaluator target switching, and strict curriculum events.
- Produces: `_run_fixed(...)` and `_run_curriculum(...)` branches under `_run_real(...)`.
- Produces: statuses `certified`, `stagnated`, `provider_*`, and `evaluation_*`.

- [ ] **Step 1: Write failing orchestration tests with fakes**

```python
def test_imported_curriculum_certifies_before_first_provider_call(fake_runtime):
    result = _run_curriculum(fake_runtime, acts=0)
    assert fake_runtime.provider.calls == []
    assert fake_runtime.certification_calls == [("v000000", "all")]
    assert fake_runtime.gate_calls == [("v000000", "rank15")]
    assert result["active_target"] == "rank15"


def test_rank15_promotion_switches_to_rank14_and_rebaselines_gate(fake_runtime):
    fake_runtime.queue_candidate_gate("rank15", score=1.0)
    fake_runtime.queue_certification(failed=("rank14",))
    result = _run_curriculum(fake_runtime, acts=1)
    assert result["active_target"] == "rank14"
    assert fake_runtime.gate_calls[-1] == ("selected", "rank14")


def test_regression_certification_rolls_back_without_updating_parent(fake_runtime):
    fake_runtime.queue_candidate_gate("rank15", score=1.0)
    fake_runtime.queue_certification(failed=("rank03", "rank14"))
    result = _run_curriculum(fake_runtime, acts=1)
    assert result["next_parent_version_id"] == "v000000"
    assert result["status"] == "running"


def test_all_sixteen_complete_exactly_once(fake_runtime):
    fake_runtime.queue_candidate_gate("rank14", score=1.0)
    fake_runtime.queue_certification(failed=())
    result = _run_curriculum(fake_runtime, acts=1)
    assert result["certified"] is True
    assert fake_runtime.events_of_type("run_completed") == [{
        "reason": "all_human_opponents_defeated",
        "passing_human_opponents": 16,
    }]
```

Add resume coverage that rebuilds the active target, stage origin, best gate score, locked set, stagnation count, evaluator feedback, and provider session from events.

The test module defines a `FakeCurriculumRuntime` with explicit lists
`provider_calls`, `certification_calls`, and `gate_calls`, queue methods
`queue_candidate_gate(target: str, score: float)` and
`queue_certification(failed: tuple[str, ...])`, plus
`events_of_type(event_type: str)`. `_run_curriculum()` accepts this runtime
through injected provider, evaluator, match, and filesystem factories; tests
must not monkeypatch production results after the call starts.

- [ ] **Step 2: Run focused CLI tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_cli.py tests/hl/test_curriculum_cli.py
```

Expected: failures because imported curriculum orchestration does not exist.

- [ ] **Step 3: Refactor shared setup without changing fixed behavior**

Extract small helpers:

```python
def _prepare_rollman_runtime(config, run_dir, workspace) -> RollmanRuntime: ...
def _write_certification(... ) -> CertificationSummary: ...
def _evaluate_curriculum_baseline(... ) -> CandidateEvaluation: ...
def _run_fixed(... ) -> int: ...
def _run_curriculum(... ) -> int: ...
```

Keep provider construction after all config, source, backend, SDK, and human-pool validation. Do not construct or invoke the provider for `--dry-run`.
Permit `run --acts 0` for `imported_version` so origin import, target gate, and
full certification can run without an API credential. Retain the pre-state-change
credential failure for `model_bootstrap` and for every invocation requesting one
or more model acts.

- [ ] **Step 4: Implement imported curriculum initialization**

For a fresh run:

1. create the frozen config and source audit;
2. create a fresh Context Bundle and Experience Skill;
3. import the source as `v000000`;
4. prepare the full human pool;
5. certify `v000000` on seeds 201–203 and persist all match/Elo events;
6. start `CurriculumManager`, selecting rank15;
7. switch evaluator to rank15;
8. evaluate `v000000` on seeds 101–103;
9. finalize the incomplete origin evaluation and freeze measurement reference;
10. construct the provider only before the first requested model act.

- [ ] **Step 5: Implement each paid act**

For each act:

1. call `controller.run_act(parent_version_id=curriculum.next_parent_version_id)`;
2. stop immediately on provider/evaluation failure;
3. record target gate and KL/occupancy metrics;
4. call `curriculum.observe_gate()`;
5. pause on `stagnated`;
6. skip full certification when target gate is below threshold;
7. certify all 16 when target gate passes;
8. reject and force the stage origin when locked opponents regress;
9. promote and select the next target when preservation succeeds;
10. evaluate the promoted version against the new target and call `lineage.begin_stage()`;
11. write `run_completed(reason="all_human_opponents_defeated")` at 16/16.

On resume, refuse config drift, rebuild curriculum state from events, restore the active target, reconstruct the latest target evaluation, preserve provider session ids, and continue without rerunning a completed certification.

- [ ] **Step 6: Run focused CLI tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_cli.py tests/hl/test_curriculum_cli.py
```

Expected: all tests pass, including fixed-mode regression tests.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/cli.py tests/hl/test_cli.py \
  tests/hl/test_curriculum_cli.py
git commit -m "feat: run weakest-failed HL curricula"
```

### Task 7: Curriculum Metrics and Multi-Panel Report

**Files:**
- Modify: `src/agentbench_frame/hl/report.py`
- Modify: `tests/hl/test_report.py`

**Interfaces:**
- Produces: `derive_curriculum_rows(events) -> list[dict[str, Any]]`.
- Extends: `write_hl_report()` return mapping with `curriculum_csv`.
- Extends: `curves.csv` with active target, target gate score, passing humans, and curriculum event marker.

`tests/hl/test_report.py` defines `EVENT_FIXTURE` with one imported origin,
rank15 gate events, a rank15 promotion, a rank14 target selection, match/Elo
events, KL, occupancy, token, and final certification fields accepted by
`HLEventWriter`.

- [ ] **Step 1: Write failing report tests**

```python
def test_curriculum_rows_capture_target_progress_and_stage_events():
    rows = derive_curriculum_rows(EVENT_FIXTURE)
    assert rows[0]["active_target"] == "rank15"
    assert rows[0]["locked_opponents"] == 14
    assert rows[1]["event"] == "stage_promoted"
    assert rows[1]["passing_human_opponents"] == 15
    assert rows[2]["active_target"] == "rank14"


def test_report_writes_curriculum_csv_and_multimetric_figures(tmp_path):
    outputs = write_hl_report(EVENT_FIXTURE, tmp_path)
    assert outputs["curriculum_csv"].is_file()
    assert outputs["curves_png"].stat().st_size > 0
    assert outputs["curves_svg"].stat().st_size > 0
```

- [ ] **Step 2: Run report tests and verify failure**

Run:

```bash
pytest -q tests/hl/test_report.py
```

Expected: failures because curriculum rows and output are absent.

- [ ] **Step 3: Implement curriculum table and plot annotations**

Define exact fields:

```python
CURRICULUM_FIELDS = (
    "coding_agent_act",
    "act_id",
    "version_id",
    "active_target",
    "target_gate_score",
    "locked_opponents",
    "passing_human_opponents",
    "event",
)
```

Extend the plot with:

- performance and best score;
- role-scoped Elo;
- target gate win rate and current target label;
- local policy KL;
- occupancy shift;
- full-pool passing count and overall win rate;
- token usage.

Draw vertical markers for target selection, promotion, rejection, rollback, and stagnation. Preserve PNG/SVG headless rendering.

- [ ] **Step 4: Run report tests and verify pass**

Run:

```bash
pytest -q tests/hl/test_report.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/report.py tests/hl/test_report.py
git commit -m "feat: report HL curriculum progress"
```

### Task 8: Local Experiment Configuration and Reproduction Guide

**Files:**
- Create: `configs/hl/29_rollman-curriculum.yaml`
- Modify: `docs/hl/rollman-local-run.md`
- Modify: `tests/hl/test_cli.py`

**Interfaces:**
- Produces: approved `k=1`, 16/16, imported-v000001 local configuration.
- Produces: exact validate, dry-run, run, resume, report, and status-inspection commands.

- [ ] **Step 1: Write a failing config contract test**

```python
def test_rollman_curriculum_config_matches_approved_experiment():
    config = LocalHLConfig.load(CURRICULUM_CONFIG)
    assert config.run.origin.source_version == "v000001"
    assert config.run.curriculum.mode == "weakest_failed"
    assert config.run.curriculum.required_human_opponents == 16
    assert config.run.iteration.candidates_per_act == 1
    assert config.run.iteration.max_acts is None
```

- [ ] **Step 2: Run the config test and verify failure**

Run:

```bash
pytest -q tests/hl/test_cli.py::test_rollman_curriculum_config_matches_approved_experiment
```

Expected: failure because the experiment config does not exist.

- [ ] **Step 3: Add the config and documentation**

Create `configs/hl/29_rollman-curriculum.yaml` by retaining the provider, rollback, experience, measurement, fixed seeds, and local paths from `29_rollman.yaml`, and add the exact approved `origin` and `curriculum` blocks. Set `evaluation.required_human_opponents: 16` so reports and generic validation agree with curriculum completion.

Document:

```bash
agentbench hl validate --config configs/hl/29_rollman-curriculum.yaml
agentbench hl run --dry-run --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-rollman-curriculum-dry
agentbench hl run --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum
agentbench hl resume --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum
agentbench hl report \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum
```

Define `stagnated` as a recoverable supervision checkpoint and `all_human_opponents_defeated` as the only curriculum success.

- [ ] **Step 4: Run the config test and verify pass**

Run:

```bash
pytest -q tests/hl/test_cli.py::test_rollman_curriculum_config_matches_approved_experiment
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add configs/hl/29_rollman-curriculum.yaml docs/hl/rollman-local-run.md \
  tests/hl/test_cli.py
git commit -m "docs: expose Rollman curriculum workflow"
```

### Task 9: Full Verification and Zero-Cost Origin Validation

**Files:**
- Modify only when a failing verification exposes a scoped defect.

**Interfaces:**
- Consumes: all implementation tasks.
- Produces: passing suite, clean tracked worktree, validated imported origin, and no paid model act.

- [ ] **Step 1: Run the complete test suite**

Run:

```bash
pytest -q
```

Expected: every test passes.

- [ ] **Step 2: Run source audit and strict config validation**

Run:

```bash
agentbench hl audit --config configs/hl/29_rollman-curriculum.yaml
agentbench hl validate --config configs/hl/29_rollman-curriculum.yaml
```

Expected: frozen sources valid, 16 humans present, imported origin resolved, active mode `weakest_failed`, target requirement 16.

- [ ] **Step 3: Run dry-run without a provider call**

Run:

```bash
agentbench hl run --dry-run \
  --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-rollman-curriculum-dry
```

Expected: `would_call_model=false`; no provider directory or Codex session exists.

- [ ] **Step 4: Initialize and locally evaluate the imported origin**

Run:

```bash
agentbench hl run \
  --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum \
  --acts 0
```

Expected:

- imported `v000000` hash equals source `v000001`;
- a complete 16-human certification exists;
- rank15 is selected as active target;
- rank15 gate replay exists for seeds 101–103;
- no `act_completed` event or provider output exists.

- [ ] **Step 5: Generate and inspect the baseline report**

Run:

```bash
agentbench hl report \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum
```

Expected: nonempty `curves.csv`, `matches.csv`, `curriculum.csv`, PNG, and SVG.

- [ ] **Step 6: Confirm no tracked drift and commit scoped fixes if required**

Run:

```bash
git status --short
git log -10 --oneline
```

Expected: no uncommitted tracked changes.

### Task 10: Start and Supervise the Paid Curriculum

**Files:**
- Generated artifacts only under `.agentbench/29_rollman/runs/run-20260731-gpt55-curriculum/`.

**Interfaces:**
- Consumes: verified local baseline and `.env` credential.
- Produces: a monitored curriculum run that pauses on stagnation or completes at 16/16.

- [ ] **Step 1: Resume one paid act**

Run:

```bash
agentbench hl resume \
  --config configs/hl/29_rollman-curriculum.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum \
  --acts 1
```

Expected: exactly one new provider act, one immutable candidate version, rank15 gate matches, and either continued rank15 state, regression rollback, or rank15 promotion.

- [ ] **Step 2: Inspect the act before continuing**

Check:

```bash
jq -r '.event_type' \
  .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum/events.jsonl \
  | sort | uniq -c
tail -n 20 \
  .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum/events.jsonl
agentbench hl report \
  --run-dir .agentbench/29_rollman/runs/run-20260731-gpt55-curriculum
```

Verify provider completion, target-specific evidence, gate score, code growth, token usage, Experience update, certification decision, and absence of opponent-source access.

- [ ] **Step 3: Continue in bounded supervised acts**

Resume one act at a time while target score changes meaningfully. A `stagnated` event requires inspecting rank15 or rank14 replays and choosing whether to resume `k=1` or create a configuration-only `k` ablation. Do not suppress provider or evaluation failures.

- [ ] **Step 4: Verify final 16/16 result**

Require:

- exactly one `run_completed` with reason `all_human_opponents_defeated`;
- `passing_human_opponents=16`;
- complete certification artifacts for the final version;
- final report containing KL, occupancy, target win rate, full-pool passing count, Elo, token usage, and stage markers;
- preserved source/config/context/content hashes and full version lineage.
