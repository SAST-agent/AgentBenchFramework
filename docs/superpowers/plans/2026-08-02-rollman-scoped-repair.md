# Rollman Scoped Top-Two Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `k=4` Rollman proposal cycle in which every branch carries an observable scope contract and the two strongest initial siblings receive one bounded feedback-driven repair before finalist selection.

**Architecture:** Extend the strict branch schema and prompt layer first, then add a controller-owned linear repair phase between quick screening and finalist evaluation. Preserve initial and repaired versions as immutable lineage nodes, emit explicit repair events, and start a provenance-linked imported-version run from `v000037`. Aggregate reporting joins the source and repair runs on a deterministic global iteration axis.

**Tech Stack:** Python 3.13, dataclasses, JSON/JSONL event sourcing, pytest, YAML configuration, Matplotlib/CSV reporting, Codex CLI Responses provider, frozen Rollman evaluator.

## Global Constraints

- Four initial candidates start from one immutable parent.
- Exactly the configured top-k completed branches receive at most one linear repair act.
- The experiment uses `repair_top_k: 2` and `repair_rounds: 1`.
- A failed or regressing repair cannot erase its initial candidate.
- Seed, replay identity, absolute replay coordinates, opponent identity, and human source code are forbidden policy inputs.
- Strategy size and `if/else` count are not selection penalties.
- API credentials remain confined to provider subprocesses and absent from artifacts.
- Information gain, Elo, full-pool win rate, and score margin use integer global HL iteration.
- Every production change follows red-green TDD and receives an independent commit.

---

### Task 1: Configuration and Strict Scope-Contract Schema

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/hl/proposal.py`
- Test: `tests/hl/test_config.py`
- Test: `tests/hl/test_proposal.py`

**Interfaces:**
- Produces: `IterationConfig.scope_contract_required: bool`
- Produces: `IterationConfig.repair_enabled: bool`
- Produces: `IterationConfig.repair_top_k: int`
- Produces: `IterationConfig.repair_rounds: int`
- Produces: `BranchBrief.activation_condition: str`
- Produces: `BranchBrief.preservation_contract: str`
- Consumes: existing `IterationConfig.candidates_per_cycle` and `load_branch_briefs(path, expected_count)`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_repair_config_defaults_are_ablatable():
    value = IterationConfig(candidates_per_cycle=4)
    assert value.scope_contract_required is True
    assert value.repair_enabled is False
    assert value.repair_top_k == 2
    assert value.repair_rounds == 1


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"repair_top_k": 0}, "repair_top_k"),
        ({"repair_top_k": 5}, "repair_top_k"),
        ({"repair_rounds": -1}, "repair_rounds"),
        ({"repair_enabled": True, "planner_enabled": False}, "planner"),
    ],
)
def test_invalid_repair_config_fails_closed(overrides, message):
    values = {"candidates_per_cycle": 4, "planner_enabled": True,
              "reducer_enabled": True, "repair_enabled": True, **overrides}
    with pytest.raises(ValueError, match=message):
        IterationConfig(**values)
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_config.py -k repair`

Expected: FAIL because the repair fields do not exist.

- [ ] **Step 3: Implement configuration fields and validation**

```python
scope_contract_required: bool = True
repair_enabled: bool = False
repair_top_k: int = 2
repair_rounds: int = 1

if self.repair_rounds not in {0, 1}:
    raise ValueError("iteration.repair_rounds must be 0 or 1")
if self.repair_enabled:
    if self.repair_top_k < 1 or self.repair_top_k > resolved_candidates:
        raise ValueError("iteration.repair_top_k must be within candidate count")
    if not self.planner_enabled:
        raise ValueError("iteration repair requires planner_enabled")
```

- [ ] **Step 4: Write failing strict-schema tests**

```python
def test_branch_brief_requires_scope_contract(tmp_path):
    path = tmp_path / "briefs.json"
    path.write_text(json.dumps([{
        "branch_index": index,
        "diagnosis": f"level 3 round {index + 20}: capture",
        "mechanism": f"mechanism-{index}",
        "activation_condition": f"observable-condition-{index}",
        "preservation_contract": f"preserve-path-{index}",
        "expected_change": f"expected-{index}",
        "falsifier": f"falsifier-{index}",
    } for index in range(4)]), encoding="utf-8")

    briefs = load_branch_briefs(path, expected_count=4)

    assert briefs[2].activation_condition == "observable-condition-2"
    assert briefs[2].preservation_contract == "preserve-path-2"
```

- [ ] **Step 5: Run schema test and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_proposal.py::test_branch_brief_requires_scope_contract`

Expected: FAIL because the strict allowed-field set rejects the two fields.

- [ ] **Step 6: Implement strict scope fields**

```python
@dataclasses.dataclass(frozen=True)
class BranchBrief:
    branch_index: int
    diagnosis: str
    mechanism: str
    activation_condition: str
    preservation_contract: str
    expected_change: str
    falsifier: str
```

Add both names to the strict `allowed` set and normalize them with `_text`.

- [ ] **Step 7: Run focused tests**

Run: `.venv/bin/pytest -q tests/hl/test_config.py tests/hl/test_proposal.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/hl/config.py src/agentbench_frame/hl/proposal.py tests/hl/test_config.py tests/hl/test_proposal.py
git commit -m "feat: define scoped repair protocol config"
```

---

### Task 2: Planner, Candidate, and Repair Prompt Contracts

**Files:**
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_context.py`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Consumes: `BranchBrief.to_dict()` with seven strict fields
- Produces: `IterationContext.build_repair_prompt` with the exact signature in Step 6
- Produces: `prompt_factory(phase="repair", repair_input=repair_input_path) -> str`

- [ ] **Step 1: Write failing planner and candidate prompt tests**

```python
assert "activation_condition" in planner
assert "preservation_contract" in planner
assert "触发条件外" in candidate
assert "保持父代" in candidate
assert "不得修改全局 scorer" in candidate
```

- [ ] **Step 2: Run prompt tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_context.py -k 'proposal or candidate'`

Expected: FAIL on missing scope-contract language.

- [ ] **Step 3: Implement scope language in planner and candidate prompts**

Require the planner JSON fields and require the candidate to route through one observable gate:

```text
activation_condition 必须是可观测状态谓词；触发条件外保持父代 action-selection 路径。
preservation_contract 必须指出不受本机制影响的既有路径。
除非 activation_condition 为真，不得修改全局 scorer、预测器、权重或候选排序。
```

Add `scope_contract_required: bool` to the planner and candidate builders. When false, retain the two fields in the logged brief but label them diagnostic-only and omit the preservation enforcement from the candidate prompt. Route `config.run.iteration.scope_contract_required` through the CLI prompt factory so the ablation changes behavior without a code edit.

- [ ] **Step 4: Write failing repair-prompt test**

```python
repair = context.build_repair_prompt(
    act_id="act-repair",
    iteration_id="iter-000012",
    branch_index=1,
    workspace=workspace,
    game_digest_path=digest,
    research_state_path=research,
    repair_input_path=repair_input,
    experience_path=experience,
)
assert str(repair_input) in repair
assert "错误诊断" in repair
assert "过宽" in repair
assert "不得切换到其他 branch" in repair
assert "最多 2 个" in repair
```

- [ ] **Step 5: Run repair-prompt test and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_context.py -k repair_prompt`

Expected: FAIL because `build_repair_prompt` is absent.

- [ ] **Step 6: Implement `build_repair_prompt`**

```python
def build_repair_prompt(
    self, *, act_id: str, iteration_id: str, branch_index: int,
    workspace: str | Path, game_digest_path: str | Path,
    research_state_path: str | Path, repair_input_path: str | Path,
    experience_path: str | Path,
) -> str:
    packet = Path(repair_input_path).resolve()
    return f"""# Rollman scoped repair {act_id}
proposal cycle: {iteration_id}
branch: {branch_index}
repair packet: {packet}

Read the bounded packet and every summary path it names. Classify the failure
as a wrong diagnosis, an over-broad activation condition, or faulty integration.
Keep the same mechanism and preserve parent behavior outside the declared gate.
Use at most two allowed trace windows, then compile once and run one NumPy smoke.
"""
```

The prompt reads only the bounded packet and its explicit summary/trace paths, preserves branch identity, permits one scoped correction, writes `experience_update.json`, runs one compile and one NumPy smoke, and stops.

- [ ] **Step 7: Route `phase="repair"` through the CLI prompt factory**

```python
if phase == "repair":
    return iteration_context.build_repair_prompt(
        act_id=values["act_id"],
        iteration_id=values["iteration_id"],
        branch_index=values["branch_index"],
        workspace=workspace,
        game_digest_path=game_digest_path,
        research_state_path=research_state_path,
        repair_input_path=Path(values["repair_input"]),
        experience_path=experience.path,
    )
```

- [ ] **Step 8: Run focused prompt tests**

Run: `.venv/bin/pytest -q tests/hl/test_context.py tests/hl/test_cli.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/agentbench_frame/hl/context.py src/agentbench_frame/hl/cli.py tests/hl/test_context.py tests/hl/test_cli.py
git commit -m "feat: prompt scoped candidate repairs"
```

---

### Task 3: Repair Data Model and Feedback Packets

**Files:**
- Create: `src/agentbench_frame/hl/repair.py`
- Create: `tests/hl/test_repair.py`
- Modify: `src/agentbench_frame/hl/controller.py`

**Interfaces:**
- Produces: `RepairSelection`
- Produces: `build_repair_packet(output_path, iteration_id, branch_brief, parent, candidate, summary_resolver) -> Path`
- Produces: `select_branch_representative(initial, repaired) -> CandidateResult`
- Consumes: `CandidateDiagnostics.from_matches(version_id, branch_index, matches)`

- [ ] **Step 1: Write failing packet and representative tests**

```python
def test_repair_packet_contains_same_seed_parent_candidate_feedback(tmp_path):
    path = build_repair_packet(
        output_path=tmp_path / "repair.json",
        iteration_id="iter-000012",
        branch_brief=brief,
        parent=parent_result,
        candidate=candidate_result,
    )
    value = json.loads(path.read_text())
    assert value["parent"]["matches"][0]["seed"] == 101
    assert value["candidate"]["matches"][0]["seed"] == 101
    assert value["scope"]["activation_condition"] == brief.activation_condition


def test_regressing_repair_cannot_replace_stronger_initial():
    assert select_branch_representative(initial, repaired) is initial
```

- [ ] **Step 2: Run repair tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_repair.py`

Expected: ERROR because `agentbench_frame.hl.repair` is absent.

- [ ] **Step 3: Implement immutable repair types and packet builder**

```python
@dataclasses.dataclass(frozen=True)
class RepairSelection:
    branch_index: int
    initial: "CandidateResult"
    repaired: "CandidateResult | None"
    representative: "CandidateResult"
    repair_input_path: Path
```

`repair.py` imports `CandidateResult` only under `TYPE_CHECKING`, so `controller.py` can import `RepairSelection` without a runtime cycle. `build_repair_packet` serializes only version ids, scores, results, margins, bounded replay summaries, runtime errors, and the strict branch brief. It rejects mismatched seeds and branches. The real controller receives a `summary_resolver(match) -> Mapping[str, Any]` callback from the CLI; unit tests use a deterministic local resolver.

- [ ] **Step 4: Implement representative selection with existing diagnostics**

Use `CandidateDiagnostics.key()` for completed versions. A non-complete repair always loses to a complete initial candidate. Exact ties retain the initial candidate.

- [ ] **Step 5: Run repair tests**

Run: `.venv/bin/pytest -q tests/hl/test_repair.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/hl/repair.py src/agentbench_frame/hl/controller.py tests/hl/test_repair.py
git commit -m "feat: build bounded repair feedback packets"
```

---

### Task 4: Controller-Owned Top-Two Linear Repair Cycle

**Files:**
- Modify: `src/agentbench_frame/hl/controller.py`
- Test: `tests/hl/test_controller.py`

**Interfaces:**
- Consumes: `build_repair_packet`
- Consumes: `select_branch_representative`
- Extends: `ProposalCycleResult.repairs` as an immutable tuple of `RepairSelection`
- Extends: `ProposalCycleResult.representatives` as an immutable tuple of `CandidateResult`
- Consumes: controller constructor callback `summary_resolver(match) -> Mapping[str, Any]`

- [ ] **Step 1: Write failing successful-cycle test**

The fake provider emits one planner output, four initial edits, two repair edits, and one reducer output. The evaluator returns deterministic quick and finalist results.

```python
result = controller.run_proposal_cycle(parent_version_id=origin.version_id)
assert len(result.candidates) == 4
assert len(result.repairs) == 2
assert len(result.representatives) == 4
assert len(result.finalists) == 2
assert len(provider.calls) == 8  # planner + 4 initial + 2 repair + reducer
assert {r.repaired.version.parent_version_id for r in result.repairs} == {
    r.initial.version.version_id for r in result.repairs
}
```

- [ ] **Step 2: Run cycle test and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_controller.py -k top_two_linear_repair`

Expected: FAIL because proposal cycles do not run repair acts.

- [ ] **Step 3: Extract reusable candidate invocation helper**

Add a private controller method that checks out a requested parent, clears candidate control outputs, invokes the provider, checkpoints usage, snapshots one immutable version, evaluates quick screen, registers lineage, and writes events:

```python
def _run_coding_candidate(
    self, *, iteration_id: str, act_id: str, branch_index: int,
    parent_version_id: str, phase: str, prompt_values: dict[str, Any],
    edit_type: str,
) -> CandidateResult:
    self.version_store.checkout(parent_version_id)
    prompt = self.prompt_factory(
        phase=phase,
        act_id=act_id,
        iteration_id=iteration_id,
        branch_index=branch_index,
        parent_version_id=parent_version_id,
        **prompt_values,
    )
    invocation = self.provider.invoke(
        prompt=prompt,
        workspace=self.workspace,
        raw_output_path=self.run_root / "provider" / f"{act_id}.jsonl",
        session_id=None,
    )
    return self._snapshot_and_quick_screen(
        iteration_id=iteration_id,
        act_id=act_id,
        branch_index=branch_index,
        parent_version_id=parent_version_id,
        edit_type=edit_type,
        prompt=prompt,
        invocation=invocation,
    )
```

Extract `_snapshot_and_quick_screen` from the body of `run_act` without changing its checkpoint, experience staging, lineage registration, event, or evaluation behavior.

- [ ] **Step 4: Implement repair ranking and acts**

After four initial quick screens:

```python
eligible = sorted(completed_initials, key=selection_key, reverse=True)
repair_targets = eligible[: self.iteration.repair_top_k]
```

For each target, persist `repair_input-bNN.json`, invoke one repair act from the initial version, quick-screen it, and choose the stronger immutable representative.

- [ ] **Step 5: Make finalist selection consume representatives**

Run finalist seeds only for the strongest representatives. Keep the parent comparison through `select_linear_successor` unchanged.

- [ ] **Step 6: Write and pass regression tests**

Add five explicit tests with these assertions:

```python
assert failed_result.repairs[0].representative is failed_result.repairs[0].initial
assert regressed_result.repairs[0].representative is regressed_result.repairs[0].initial
assert len(partial_result.repairs) == 1
assert parent_result.search_parent_version_id == origin.version_id
assert all(
    repair.repaired.version.parent_version_id == repair.initial.version.version_id
    for repair in successful_result.repairs
)
```

Run: `.venv/bin/pytest -q tests/hl/test_controller.py -k 'repair or proposal_cycle'`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/controller.py tests/hl/test_controller.py
git commit -m "feat: run top-two linear candidate repairs"
```

---

### Task 5: Repair Events, Checkpoint Recovery, and Reducer Facts

**Files:**
- Modify: `src/agentbench_frame/hl/events.py`
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_events.py`
- Test: `tests/hl/test_controller.py`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Produces events: `repairs_selected`, `repair_started`, `repair_completed`, `branch_representative_selected`
- Produces: `recover_completed_repair(checkpoint, version_event, evaluation_event)`
- Extends reducer packet with `initial_candidates`, `repairs`, and `representatives`

- [ ] **Step 1: Write failing event-schema tests**

```python
writer.write("repairs_selected", iteration_id="iter-000012",
             branch_indices=[1, 3], version_ids=["v2", "v4"])
writer.write("repair_completed", iteration_id="iter-000012",
             act_id="act-repair", branch_index=1,
             initial_version_id="v2", repaired_version_id="v5",
             status="completed")
assert [e["event_type"] for e in read_events(path)] == [
    "repairs_selected", "repair_completed"
]
```

- [ ] **Step 2: Run event tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_events.py -k repair`

Expected: FAIL because strict event fields are undefined.

- [ ] **Step 3: Define strict repair event fields**

Register the four event types and validate branch indices, version id lists, act ids, and statuses with the existing event validator.

- [ ] **Step 4: Write failing resume test**

Persist a completed repair raw stream, checkpoint, version event, and evaluation. Resume the cycle and assert that the provider receives no second prompt for that repair act.

```python
assert recovered.metadata["recovered_from_persisted_output"] is True
assert provider.repair_call_count == 0
```

- [ ] **Step 5: Implement repair recovery**

The CLI reconstructs completed repair stages from event/checkpoint/version/evaluation facts. The controller accepts `repair_recoveries: Mapping[int, CandidateResult]` and reuses them after content-hash validation.

- [ ] **Step 6: Extend reducer packet**

```json
{
  "initial_candidates": [],
  "repairs": [],
  "representatives": [],
  "selected_version_id": "v000123"
}
```

Each repair row includes the initial and repaired scores, status, matches, and representative decision. Timeout facts remain operational facts rather than gameplay conclusions.

- [ ] **Step 7: Run recovery and reducer tests**

Run: `.venv/bin/pytest -q tests/hl/test_events.py tests/hl/test_controller.py tests/hl/test_cli.py -k 'repair or reducer or recover'`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/hl/events.py src/agentbench_frame/hl/controller.py src/agentbench_frame/hl/cli.py tests/hl/test_events.py tests/hl/test_controller.py tests/hl/test_cli.py
git commit -m "feat: recover and audit repair stages"
```

---

### Task 6: Branch and Aggregate Reporting

**Files:**
- Modify: `src/agentbench_frame/hl/report.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_report.py`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Extends `BRANCH_FIELDS` with `stage`, `parent_version_id`, `representative`
- Produces: `build_aggregate_curves(source_run, phase_run, global_origin_iteration) -> list[dict[str, Any]]`
- Produces CLI: `agentbench_frame.hl.cli aggregate-report`

- [ ] **Step 1: Write failing branch-report test**

```python
rows = build_branch_rows(events)
assert {(r["branch_index"], r["stage"]) for r in rows} == {
    (0, "initial"), (1, "initial"), (2, "initial"), (3, "initial"),
    (1, "repair-1"), (3, "repair-1"),
}
assert sum(bool(r["representative"]) for r in rows) == 4
```

- [ ] **Step 2: Run branch-report test and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_report.py -k repair`

Expected: FAIL on missing fields and rows.

- [ ] **Step 3: Extend branch reporting**

Derive initial and repair rows from immutable version, evaluation, and representative events. Do not infer a repair row from source code or filesystem state.

- [ ] **Step 4: Write failing aggregate-curve test**

```python
rows = build_aggregate_curves(source_events, phase_events, global_origin_iteration=11)
assert [r["global_iteration"] for r in rows] == list(range(12 + phase_cycles))
assert rows[11]["version_id"] == "v000037"
assert rows[12]["phase_iteration"] == 1
assert rows[12]["source_run"] == "repair-run"
```

- [ ] **Step 5: Implement deterministic aggregate join**

Use the source curve rows through iteration 11, drop the imported phase origin duplicate, append phase iterations `1..n` at global iterations `12..11+n`, and write:

- `aggregate-curves.csv`
- `aggregate-curves.png`
- `aggregate-curves.svg`

The plot contains four panels: mean local policy KL, Rollman Elo, full-pool win rate, and mean full-pool score margin.

- [ ] **Step 6: Add aggregate-report CLI**

```text
python -m agentbench_frame.hl.cli aggregate-report \
  --source-run SOURCE --phase-run PHASE --origin-iteration 11
```

- [ ] **Step 7: Run report and CLI tests**

Run: `.venv/bin/pytest -q tests/hl/test_report.py tests/hl/test_cli.py -k report`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agentbench_frame/hl/report.py src/agentbench_frame/hl/cli.py tests/hl/test_report.py tests/hl/test_cli.py
git commit -m "feat: report scoped repairs on global iterations"
```

---

### Task 7: Imported-v37 Experiment Configuration and End-to-End Verification

**Files:**
- Create: `configs/hl/29_rollman-k4-repair.yaml`
- Modify: `tests/hl/test_e2e_fake.py`
- Modify: `docs/hl/README.md`

**Interfaces:**
- Consumes: `origin.mode: imported_version`
- Consumes: source run `run-20260802-gpt55-k4-interface-v4`, source version `v000037`
- Produces: independently frozen repair-phase run configuration

- [ ] **Step 1: Create explicit experiment config**

Copy the frozen Rollman settings and change only the protocol and provenance fields:

```yaml
origin:
  mode: "imported_version"
  source_run: ".agentbench/29_rollman/runs/run-20260802-gpt55-k4-interface-v4"
  source_version: "v000037"
  reset_session: true
  reset_experience: false
iteration:
  candidates_per_cycle: 4
  planner_enabled: true
  reducer_enabled: true
  scope_contract_required: true
  repair_enabled: true
  repair_top_k: 2
  repair_rounds: 1
  quick_screen_seeds: 1
  finalist_count: 2
  finalist_seeds: 3
```

- [ ] **Step 2: Write failing fake end-to-end test**

The fake provider writes seven-field branch briefs and handles two repair prompts. Assert four initial versions, two repair descendants, four representatives, two finalists, and one aggregate curve point.

- [ ] **Step 3: Run fake end-to-end test and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_e2e_fake.py`

Expected: FAIL until all repair protocol components are connected.

- [ ] **Step 4: Complete integration wiring and documentation**

Document the start, resume, report, aggregate-report, repair ablation, and imported provenance commands in `docs/hl/README.md`. Keep credentials in `.env` only.

- [ ] **Step 5: Run focused HL suite**

Run: `.venv/bin/pytest -q tests/hl`

Expected: all HL tests and subtests pass.

- [ ] **Step 6: Run full repository suite outside the restricted socket sandbox**

Run: `.venv/bin/pytest -q`

Expected: all main tests and subtests pass with zero failures.

- [ ] **Step 7: Validate and dry-run configuration without a credential**

Run:

```bash
AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST \
.venv/bin/python -m agentbench_frame.hl.cli validate \
  --config configs/hl/29_rollman-k4-repair.yaml

AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST \
.venv/bin/python -m agentbench_frame.hl.cli run \
  --config configs/hl/29_rollman-k4-repair.yaml --dry-run
```

Expected: both commands succeed, show imported `v000037`, four candidates, top-two repair, one repair round, and no credential value.

- [ ] **Step 8: Commit**

```bash
git add configs/hl/29_rollman-k4-repair.yaml tests/hl/test_e2e_fake.py docs/hl/README.md
git commit -m "feat: freeze v37 scoped repair experiment"
```

- [ ] **Step 9: Start the real repair-phase run only after verification**

Run:

```bash
AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST \
MPLCONFIGDIR=/private/tmp/agentbench-mpl \
.venv/bin/python -m agentbench_frame.hl.cli run \
  --config configs/hl/29_rollman-k4-repair.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v1 \
  --workspace .agentbench/29_rollman/candidate-k4-repair-v1 \
  --acts 1
```

Expected: the imported origin evaluates successfully, one cycle records four initial siblings and two repair descendants, and the lineage remains rollback-safe.
