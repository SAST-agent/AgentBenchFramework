# Rollman Generalizable HL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-evaluation Experience Skill loop and stratified k=4 Rollman curriculum that passes rank15 and rank16 on at least four of five sealed certification seeds each.

**Architecture:** An append-only Framework-owned experience ledger stores exact per-branch comparisons after evaluation. A deterministic projector combines those verified outcomes with bounded replay-grounded candidate notes into the Skill every later act reads. Rollman evaluation uses rotating train/validation cases for both hard opponents while keeping a five-seed certification set sealed from all model context.

**Tech Stack:** Python 3.12, dataclasses, JSON/JSONL, pytest, YAML, existing AgentBench HL controller and Rollman evaluator.

## Global Constraints

- Four candidates remain siblings from one explicit parent; do not build a search tree.
- Rules, decision-space definitions, and replay instructions remain frozen hashed context.
- Never place certification seeds, matches, replay paths, or summaries in model input or Experience memory.
- Framework match measurements are authoritative; model-authored text is provisional.
- Certification requires rank15 wins >= 4/5 and rank16 wins >= 4/5 on valid sealed matches.
- Strategy code may grow and use observable-state interpretable mechanisms, but cannot branch on seed, replay identity, fixed replay coordinates, human identity, or opponent source.

---

### Task 1: Framework-owned experience ledger

**Files:**
- Create: `src/agentbench_frame/hl/experience_ledger.py`
- Create: `tests/hl/test_experience_ledger.py`

**Interfaces:**
- Produces: `ExperienceComparison`, `ExperienceRecord`, `derive_experience_record(...)`, and `ExperienceLedger.append(...)` / `ExperienceLedger.load()`.
- Consumes: branch brief mappings, parent/candidate `CandidateEvaluation`, activation mappings, and replay references already produced by the evaluator.

- [ ] **Step 1: Write failing verdict and persistence tests**

```python
def test_mixed_record_preserves_good_and_bad_match_conditions(tmp_path):
    record = derive_experience_record(
        iteration_id="iter-1",
        branch_index=2,
        parent_version_id="v1",
        candidate_version_id="v2",
        selected=False,
        brief={"activation_condition": "level == 3", "mechanism": "breakout", "preservation_contract": "otherwise parent"},
        parent_matches=(match("rank15", 11, 0, 100), match("rank15", 12, 0, 100)),
        candidate_matches=(match("rank15", 11, 60, 100), match("rank15", 12, -50, 100)),
        activation={"status": "complete", "decision_count": 20, "changed_action_count": 3},
    )
    assert record.verdict == "mixed"
    assert [row.margin_delta for row in record.comparisons] == [60.0, -50.0]

def test_ledger_rejects_secret_material_and_round_trips_jsonl(tmp_path):
    ledger = ExperienceLedger(tmp_path / "ledger.jsonl")
    ledger.append(valid_record())
    assert ledger.load() == (valid_record(),)
    with pytest.raises(ValueError, match="credential"):
        ledger.append(dataclasses.replace(valid_record(), mechanism="sk-secretvalue"))
```

- [ ] **Step 2: Run `pytest tests/hl/test_experience_ledger.py -q` and verify missing-module failure**
- [ ] **Step 3: Implement strict dataclasses, deterministic comparison join, verdict derivation, canonical JSONL, duplicate-record rejection, and credential validation**
- [ ] **Step 4: Run `pytest tests/hl/test_experience_ledger.py -q` and verify pass**
- [ ] **Step 5: Commit `feat: add verified HL experience ledger`**

### Task 2: Deterministic Skill projection

**Files:**
- Modify: `src/agentbench_frame/hl/experience.py`
- Modify: `tests/hl/test_experience.py`
- Test: `tests/hl/test_experience_ledger.py`

**Interfaces:**
- Consumes: `tuple[ExperienceRecord, ...]` and strict candidate-authored `ExperienceUpdate` notes.
- Produces: `ExperienceManager.consolidate_cycle(...) -> Path`, regenerated `SKILL.md`, and bounded `state.json`.

- [ ] **Step 1: Write failing projection tests**

```python
def test_skill_separates_verified_good_bad_and_mixed(tmp_path):
    manager = ExperienceManager(tmp_path)
    manager.consolidate_cycle(records=(good_record(), bad_record(), mixed_record()), notes=())
    text = manager.path.read_text()
    assert "## Verified good conditions" in text
    assert "margin_delta=+60" in text
    assert "## Verified bad conditions" in text
    assert "## Mixed or scope-sensitive findings" in text

def test_skill_projection_is_deterministic_and_bounded(tmp_path):
    first = build_skill(tmp_path / "a", reversed(records()))
    second = build_skill(tmp_path / "b", records())
    assert first == second
    assert len(first.encode()) <= 65536
```

- [ ] **Step 2: Run the two tests and verify they fail because post-evaluation consolidation is absent**
- [ ] **Step 3: Implement deterministic grouping, exact effect formatting, duplicate compaction, provisional replay observations, and the five required Skill sections**
- [ ] **Step 4: Preserve `apply_file` compatibility for imported runs and run `pytest tests/hl/test_experience.py tests/hl/test_experience_ledger.py -q`**
- [ ] **Step 5: Commit `feat: project verified outcomes into HL Skill`**

### Task 3: Controller post-evaluation consolidation

**Files:**
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `src/agentbench_frame/hl/events.py`
- Modify: `tests/hl/test_controller.py`
- Modify: `tests/hl/test_events.py`

**Interfaces:**
- Consumes: all four branch representatives, parent evaluation, branch briefs, selected search parent, and each representative's pending notes.
- Produces: four ledger records and one `experience_cycle_consolidated` event after reducer completion.

- [ ] **Step 1: Write a failing controller test proving all four outcomes are consolidated after finalist evaluation**

```python
def test_proposal_cycle_consolidates_all_representatives_after_evaluation(tmp_path):
    result = controller_with_four_measured_branches(tmp_path).run_proposal_cycle(parent_evaluation=parent_eval())
    records = ExperienceLedger(tmp_path / "experience/ledger.jsonl").load()
    assert {record.branch_index for record in records} == {0, 1, 2, 3}
    assert any(record.verdict == "verified_bad" for record in records)
    assert result.search_parent_version_id in manager.path.read_text()
```

- [ ] **Step 2: Run the test and verify the ledger is absent**
- [ ] **Step 3: Add `consolidate_experience_cycle` after selection/reducer, load pending notes strictly, write all representative records, regenerate Skill, and emit the event**
- [ ] **Step 4: Add resume/idempotency coverage so an interrupted cycle cannot append the same `(iteration, branch, candidate)` twice**
- [ ] **Step 5: Run `pytest tests/hl/test_controller.py tests/hl/test_events.py -q`**
- [ ] **Step 6: Commit `feat: consolidate branch outcomes after evaluation`**

### Task 4: Stratified k=4 evidence routing

**Files:**
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `tests/hl/test_context.py`
- Modify: `tests/hl/test_proposal.py`
- Modify: `tests/hl/test_cli.py`

**Interfaces:**
- Produces: `stratify_rollout_evidence(evidence, hard_opponents=("rank15", "rank16")) -> tuple[tuple[Mapping, ...], ...]` and branch input packets with isolated evidence.
- Branch roles are fixed as rank15 suppression, rank16 offense, cross-replay best response, and generalization/consolidation.

- [ ] **Step 1: Write failing evidence-isolation tests**

```python
def test_k4_packets_assign_complementary_hard_opponent_evidence():
    packets = stratify_rollout_evidence(evidence_fixture(), hard_opponents=("rank15", "rank16"))
    assert {row["opponent"] for row in packets[0]} == {"rank15"}
    assert {row["opponent"] for row in packets[1]} == {"rank16"}
    assert {row["opponent"] for row in packets[2]} == {"rank15", "rank16"}
    assert packets[3] != packets[0] and packets[3] != packets[1]
```

- [ ] **Step 2: Run focused context/proposal tests and verify failure**
- [ ] **Step 3: Implement deterministic failure-cluster ordering and branch packets; preserve at most two trace probes per branch and exclude certification phases**
- [ ] **Step 4: Update planner/candidate prompts to require the four fixed roles and cite each packet's evidence rather than a shared replay bundle**
- [ ] **Step 5: Run `pytest tests/hl/test_context.py tests/hl/test_proposal.py tests/hl/test_cli.py -q`**
- [ ] **Step 6: Commit `feat: stratify replay evidence across k4 rollouts`**

### Task 5: Dual-opponent train/validation and sealed certification

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/games/rollman/evaluator.py`
- Modify: `src/agentbench_frame/hl/selection.py`
- Modify: `src/agentbench_frame/hl/curriculum.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `tests/hl/test_config.py`
- Modify: `tests/rollman/test_evaluator.py`
- Modify: `tests/hl/test_selection.py`
- Modify: `tests/hl/test_curriculum.py`

**Interfaces:**
- Config adds explicit `training_seeds`, `validation_seeds`, `certification_seeds`, `hard_opponents`, `certification_wins_required`, and deterministic rotation stride.
- Evaluator adds common two-opponent quick/finalist cases and a certification summary per opponent.
- Selection compares per-opponent points, mean margins, and worst margins without allowing a material regression on the preserved opponent.

- [ ] **Step 1: Write failing config tests for disjoint seed roles and exactly five certification seeds**

```python
def test_seed_roles_are_disjoint_and_certification_has_five_seeds():
    config = EvaluationConfig(training_seeds=(1,2,3,4), validation_seeds=(11,12), certification_seeds=(91,92,93,94,95))
    assert set(config.training_seeds).isdisjoint(config.validation_seeds)

@pytest.mark.parametrize("bad", [(1,2,3,4), (1,2,3,4,4)])
def test_certification_requires_five_unique_seeds(bad):
    with pytest.raises(ValueError):
        EvaluationConfig(certification_seeds=bad)
```

- [ ] **Step 2: Write failing evaluator and selection tests proving common rank15/rank16 cases and 4/5 arithmetic**
- [ ] **Step 3: Implement seed-role validation, deterministic training rotation, dual-opponent stages, robust successor comparison, and sealed-context assertions**
- [ ] **Step 4: Add certification tests where rank15=4/5 and rank16=4/5 passes, while either opponent=3/5 fails**
- [ ] **Step 5: Run focused config/evaluator/selection/curriculum tests**
- [ ] **Step 6: Commit `feat: add dual-opponent sealed Rollman gates`**

### Task 6: Rollman experiment configuration and end-to-end verification

**Files:**
- Create: `configs/hl/29_rollman-k4-generalizable.yaml`
- Modify: `tests/hl/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Produces a resumable experiment imported from v77 with k=4, rank15/rank16 training, rotating seeds, and five sealed certification seeds.

- [ ] **Step 1: Write a failing configuration snapshot test asserting k=4, both opponents, disjoint seed pools, and 4/5 requirement**
- [ ] **Step 2: Run the snapshot test and verify the new config is missing**
- [ ] **Step 3: Add the config with training seeds `[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]`, validation seeds `[201, 202, 203, 204, 205, 206]`, sealed certification seeds `[910001, 910002, 910003, 910004, 910005]`, `hard_opponents: [rank15, rank16]`, and `certification_wins_required: 4`**
- [ ] **Step 4: Run `python -m pytest -q` and require the complete suite to pass**
- [ ] **Step 5: Run one local dry/resume cycle with a fake provider fixture and verify the next planner packet contains the regenerated Skill but no certification seeds**
- [ ] **Step 6: Commit `feat: configure generalizable Rollman k4 experiment`**

### Task 7: Live rank15/rank16 iteration and tracking

**Files:**
- Runtime outputs: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable/`
- Runtime report: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable/report/`

**Interfaces:**
- Consumes the approved API provider and v77 import.
- Produces reproducible events, provider logs, immutable versions, ledger, Skill history, IG/Elo/win-rate curves, and certification evidence.

- [ ] **Step 1: Start the new run from v77 and record its run ID and frozen config snapshot**
- [ ] **Step 2: After each cycle inspect all four branch verdicts, Skill regeneration, train/validation results, activation, and API failures**
- [ ] **Step 3: If three cycles show no validation improvement, change evidence allocation or parent archive selection through explicit configuration rather than parameter grid search**
- [ ] **Step 4: Run sealed certification only at readiness gates; never feed failed certification replays back into the same run**
- [ ] **Step 5: Continue until rank15 and rank16 independently achieve at least four valid wins in five sealed matches**
- [ ] **Step 6: Regenerate IG-iteration, Elo-iteration, and win-rate-iteration reports and audit the full evidence chain**
