# Rollman K=4 Linear Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver and run a reproducible Rollman-only HL loop that generates four mechanism-distinct candidates per proposal cycle, selects exactly one linear successor, keeps an independently certified champion, and learns from all four feedback trajectories without relying on a Codex App conversation.

**Architecture:** `agentbench_frame.hl` gains explicit proposal-cycle configuration, deterministic static context digests, bounded research-state checkpoints, structured planner/reducer acts, lexicographic linear search selection, and reporting-panel lifecycle events. `agentbench_frame.games.rollman` supplies staged target evaluation and dense game-grounded diagnostics. The existing append-only event log and immutable version store remain the factual audit trail.

**Tech Stack:** Python 3.11+, dataclasses, JSON/YAML, pytest, Codex CLI JSONL provider, frozen PacmanLogic/PacmanSDK, matplotlib.

## Global Constraints

- Candidate role is Rollman (`role_id=0`); Ghost HL is out of scope.
- Every proposal cycle creates exactly four code candidates from one identical parent and selects exactly one next search parent.
- Non-selected candidates are immutable audit artifacts and never become active descendants.
- Policy source size and number of if/else branches have no penalty.
- No opponent source, seed, replay ID, opponent ID, or fixed replay-coordinate hardcoding is allowed.
- Static game truth remains the human-reviewed rules, decision-space YAML, and Replay Skill.
- Cross-cycle semantic state is explicit and reconstructible from run artifacts.
- All code changes follow test-first red/green/refactor cycles.

---

### Task 1: K=4 Experiment Configuration Contract

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `tests/hl/test_config.py`
- Create: `configs/hl/29_rollman-k4.yaml`

**Interfaces:**
- Produces: `IterationConfig.candidates_per_cycle`, `planner_enabled`, `reducer_enabled`, `quick_screen_seeds`, `finalist_count`, `finalist_seeds`.
- Produces: `SelectionConfig(mode, exploration_debt_cycles, source_size_penalty)`.
- Produces: `ContextConfig(use_game_digest, research_state_max_bytes, reduction_token_threshold)`.
- Produces: reporting-panel fields on `EvaluationConfig`.
- Consumes: existing strict mapping loader and secret-free `HLRunConfig.to_dict()`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_k4_config_parses_linear_proposal_cycle():
    config = HLRunConfig.from_mapping({
        "game": "29_rollman",
        "provider": {"model": "gpt-5.5", "context_mode": "fresh"},
        "iteration": {
            "candidates_per_cycle": 4,
            "planner_enabled": True,
            "reducer_enabled": True,
            "quick_screen_seeds": 1,
            "finalist_count": 2,
            "finalist_seeds": 3,
        },
        "selection": {
            "mode": "linear_lexicographic",
            "exploration_debt_cycles": 3,
            "source_size_penalty": False,
        },
        "context": {
            "use_game_digest": True,
            "research_state_max_bytes": 16384,
            "reduction_token_threshold": 250000,
        },
        "evaluation": {
            "reporting_panel_every_cycle": True,
            "reporting_seeds_per_opponent": 1,
        },
    })
    assert config.iteration.candidates_per_cycle == 4
    assert config.selection.mode == "linear_lexicographic"
    assert config.selection.source_size_penalty is False
    assert config.context.research_state_max_bytes == 16384
    assert config.evaluation.reporting_panel_every_cycle is True

def test_linear_k4_rejects_tree_width_and_invalid_finalists():
    with pytest.raises(ValueError, match="finalist_count"):
        IterationConfig(candidates_per_cycle=4, finalist_count=5)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/hl/test_config.py -q`

Expected: FAIL because the new fields and dataclasses do not exist.

- [ ] **Step 3: Implement the strict dataclass schema**

Add the new dataclasses and mapping sections. Preserve backward parsing by accepting
`candidates_per_act` only for legacy configs and normalizing it to
`candidates_per_cycle`; new configs serialize only the new field. Validate:

```python
if self.candidates_per_cycle != 4:
    raise ValueError("Rollman linear search requires exactly four candidates")
if not 1 <= self.finalist_count <= self.candidates_per_cycle:
    raise ValueError("iteration.finalist_count must be within candidate count")
if self.source_size_penalty:
    raise ValueError("Rollman k4 policy source size must not be penalized")
```

- [ ] **Step 4: Add the portable K=4 YAML**

Create `configs/hl/29_rollman-k4.yaml` with `origin.mode: model_bootstrap`,
`provider.context_mode: fresh`, the exact K=4 settings from the design, relative
artifact paths, and `${AGENTBENCH_*}` machine-local source-root placeholders.

- [ ] **Step 5: Run configuration tests and validation**

Run: `.venv/bin/pytest tests/hl/test_config.py tests/hl/test_local_config.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/hl/config.py tests/hl/test_config.py configs/hl/29_rollman-k4.yaml
git commit -m "config: define Rollman k4 linear search"
```

### Task 2: Deterministic Game Digest and Bounded Research State

**Files:**
- Create: `src/agentbench_frame/hl/research_state.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `tests/hl/test_context.py`
- Create: `tests/hl/test_research_state.py`

**Interfaces:**
- Produces: `compile_game_digest(bundle: ContextBundle, destination: Path) -> Path`.
- Produces: `ResearchState.load_or_create(path, max_bytes)` and `ResearchState.write(path)`.
- Produces: `IterationContext.build_planner_prompt(...)`, `build_candidate_prompt(...)`, and `build_reducer_prompt(...)`.
- Consumes: context manifest hashes, decision-space YAML, rule Markdown heading index, Replay Skill front matter, replay evidence, branch briefs, and measured outcomes.

- [ ] **Step 1: Write failing digest and research-state tests**

```python
def test_game_digest_is_deterministic_and_contains_primitive_actions(tmp_path):
    bundle = ContextBundle.create(tmp_path / "bundle", _static_files(tmp_path / "assets"))
    first = compile_game_digest(bundle, tmp_path / "first.json")
    second = compile_game_digest(bundle, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text())
    assert value["context_bundle_hash"] == bundle.bundle_hash
    assert value["roles"]["rollman"]["actions"] == [0, 1, 2, 3, 4]

def test_research_state_rejects_secret_and_byte_overflow(tmp_path):
    state = ResearchState.empty(max_bytes=256)
    with pytest.raises(ValueError, match="credential"):
        state.with_findings(stable_knowledge=["sk-secret-value"])
    with pytest.raises(ValueError, match="max_bytes"):
        state.with_findings(stable_knowledge=["x" * 1000])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/hl/test_context.py tests/hl/test_research_state.py -q`

Expected: FAIL because digest and research-state APIs are missing.

- [ ] **Step 3: Implement deterministic digest compilation**

Parse YAML through `yaml.safe_load`, extract exact role/action/output-shape fields,
index Markdown headings without semantic rewriting, read Replay Skill front matter,
and emit canonical sorted JSON with the context bundle hash. Reject a Rollman
support other than `[0,1,2,3,4]`.

- [ ] **Step 4: Implement bounded research state**

Use a frozen dataclass with exact fields:

```python
@dataclasses.dataclass(frozen=True)
class ResearchState:
    schema_version: str
    proposal_cycle: int
    search_parent_version_id: str | None
    official_champion_version_id: str | None
    active_target: str | None
    locked_opponents: tuple[str, ...]
    stable_knowledge: tuple[str, ...]
    failed_hypotheses: tuple[str, ...]
    open_questions: tuple[str, ...]
    recent_comparisons: tuple[dict[str, Any], ...]
    exploration_debt: int
    max_bytes: int
```

Canonical JSON serialization performs secret scanning and enforces `max_bytes`.
Full reducer inputs remain in provider artifacts; only bounded findings enter the
next cycle.

- [ ] **Step 5: Split prompts by role**

Planner prompt consumes common evidence and writes
`.agentbench/branch_briefs.json`. Candidate prompt consumes one exact brief and
explicitly states that policy growth is allowed. Reducer prompt consumes all four
briefs and result summaries and writes `.agentbench/research_state_update.json`.
Remove mandatory full static rereads and source-compression requirements from
candidate prompts.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/hl/test_context.py tests/hl/test_research_state.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/context.py src/agentbench_frame/hl/research_state.py tests/hl/test_context.py tests/hl/test_research_state.py
git commit -m "feat: checkpoint compact Rollman research context"
```

### Task 3: Planner, Four Candidate Acts, and Comparative Reducer

**Files:**
- Create: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `src/agentbench_frame/hl/events.py`
- Modify: `tests/hl/test_controller.py`
- Modify: `tests/hl/test_events.py`

**Interfaces:**
- Produces: `BranchBrief(branch_index, diagnosis, mechanism, expected_change, falsifier)`.
- Produces: `ProposalCycleResult(iteration_id, planner, candidates, finalists, selected, reducer)`.
- Consumes: provider adapter, one common parent, prompt factory, staged evaluator, research-state manager.

- [ ] **Step 1: Write failing controller test for one-parent/four-candidate/reducer behavior**

```python
def test_k4_cycle_uses_one_parent_and_reducer_receives_all_feedback(tmp_path):
    controller = _k4_controller(tmp_path)
    origin = controller.initialize()
    result = controller.run_proposal_cycle(parent_version_id=origin.version_id)
    assert len(result.candidates) == 4
    assert {c.version.parent_version_id for c in result.candidates} == {origin.version_id}
    assert len(result.finalists) == 2
    assert result.selected in result.finalists
    assert controller.lineage.lineage_head_version_id == result.selected.version.version_id
    reducer_payload = json.loads(result.reducer.input_path.read_text())
    assert {row["branch_index"] for row in reducer_payload["candidates"]} == {0, 1, 2, 3}
```

Add a second test asserting that the fifth provider call cannot create a fifth
candidate and that a non-selected version is never used as the next parent.

- [ ] **Step 2: Run controller tests and verify RED**

Run: `.venv/bin/pytest tests/hl/test_controller.py tests/hl/test_events.py -q`

Expected: FAIL because `run_proposal_cycle` and proposal events do not exist.

- [ ] **Step 3: Implement structured brief validation**

`proposal.py` validates exactly four unique branch indices, non-empty mechanisms,
and mechanism-normalized uniqueness. It rejects threshold-only descriptions when
all normalized mechanisms match.

- [ ] **Step 4: Implement the proposal-cycle controller**

The invocation order is planner, four independent candidate acts, staged
evaluation, reducer. Every candidate checkout starts from the same content hash.
The controller writes:

```text
proposal_cycle_started
planner_completed
candidate_act_completed × 4
candidate_quick_screen_completed × 4
finalists_selected
candidate_finalist_evaluation_completed × 2
search_parent_selected
reducer_completed
proposal_cycle_completed
```

Planner/reducer acts count as coding-agent acts but do not create policy versions.
Candidate acts always use fresh provider sessions and fully persisted inputs.

- [ ] **Step 5: Persist reducer state only after framework validation**

Reject reducer updates that contradict match outcomes, contain secrets, omit any
candidate, or exceed the context byte limit. On reducer failure, persist the
failure event and generate a deterministic minimal research-state update from
framework facts so the run remains resumable.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/hl/test_controller.py tests/hl/test_events.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/proposal.py src/agentbench_frame/hl/controller.py src/agentbench_frame/hl/events.py tests/hl/test_controller.py tests/hl/test_events.py
git commit -m "feat: orchestrate four-candidate Rollman cycles"
```

### Task 4: Lexicographic Search Parent and Independent Champion

**Files:**
- Create: `src/agentbench_frame/hl/selection.py`
- Modify: `src/agentbench_frame/hl/lineage.py`
- Modify: `src/agentbench_frame/hl/curriculum.py`
- Create: `tests/hl/test_selection.py`
- Modify: `tests/hl/test_lineage.py`
- Modify: `tests/hl/test_curriculum.py`

**Interfaces:**
- Produces: `CandidateDiagnostics.from_matches(matches)`.
- Produces: `select_linear_successor(parent, finalists) -> SelectionDecision`.
- Produces: `SearchState(search_parent_version_id, official_champion_version_id, exploration_debt)`.
- Consumes: valid completed match records and immutable version IDs.

- [ ] **Step 1: Write failing lexicographic selection tests**

```python
def test_score_margin_improvement_advances_search_parent_without_promoting_champion():
    parent = diagnostics(version="v0", wins=0, margin=-500, worst=-700)
    candidate = diagnostics(version="v1", wins=0, margin=-120, worst=-250)
    decision = select_linear_successor(parent, [candidate])
    assert decision.search_parent_version_id == "v1"
    assert decision.promote_official_champion is False

def test_exploration_debt_falls_back_to_best_specialist_not_origin():
    state = SearchState(search_parent_version_id="v5", official_champion_version_id="v0", exploration_debt=3)
    decision = choose_debt_recovery(state, specialists=[diagnostics(version="v3", wins=0, margin=-50)])
    assert decision.search_parent_version_id == "v3"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/hl/test_selection.py tests/hl/test_lineage.py tests/hl/test_curriculum.py -q`

Expected: FAIL because dense diagnostics and independent pointers do not exist.

- [ ] **Step 3: Implement game-grounded diagnostics and lexicographic comparison**

Compute only from valid matches: points, mean Rollman-minus-Ghosts margin, worst
margin, completed level, survival decisions, captures, and behavioral novelty.
Compare common seed sets only. Do not use source length.

- [ ] **Step 4: Separate search-parent and champion transitions**

`candidate_selected` updates the search parent. `champion_promoted` updates the
official champion only after hidden certification and locked-opponent checks.
Rebuild both pointers from events. Replace unconditional stage-origin rejection
with nearest safe ancestor or best historical target specialist.

- [ ] **Step 5: Implement exploration debt**

Increment debt only when no target win or score-margin improvement occurs. Reset
on either improvement. At debt 3, select the best historical non-dominated target
specialist; use the official champion only if no specialist has a valid advantage.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/hl/test_selection.py tests/hl/test_lineage.py tests/hl/test_curriculum.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/selection.py src/agentbench_frame/hl/lineage.py src/agentbench_frame/hl/curriculum.py tests/hl/test_selection.py tests/hl/test_lineage.py tests/hl/test_curriculum.py
git commit -m "feat: separate Rollman search parent and champion"
```

### Task 5: Staged Rollman Evaluation, Opponent Audit, and Reporting Panel

**Files:**
- Modify: `src/agentbench_frame/games/rollman/evaluator.py`
- Modify: `src/agentbench_frame/games/rollman/opponents.py`
- Create: `src/agentbench_frame/games/rollman/diagnostics.py`
- Modify: `tests/rollman/test_evaluator.py`
- Modify: `tests/rollman/test_opponents.py`
- Create: `tests/rollman/test_diagnostics.py`

**Interfaces:**
- Produces: `quick_screen(version, target, seed)`.
- Produces: `evaluate_finalist(version, target, seeds)`.
- Produces: `evaluate_reporting_panel(version, opponents, seeds)`.
- Produces: `audit_opponents(reference_rollmen, opponents, seeds)`.
- Produces: `primitive_counterfactuals(trace_window) -> tuple[ActionDiagnostic, ...]` over actions 0–4.

- [ ] **Step 1: Write failing staged-evaluator tests**

```python
def test_all_four_candidates_get_one_quick_match_and_only_two_get_finalist_matches():
    evaluator = staged_evaluator(tmp_path)
    quick = [evaluator.quick_screen(v, target, 101) for v in versions[:4]]
    finalists = evaluator.select_finalists(quick, count=2)
    full = [evaluator.evaluate_finalist(v, target, (102, 103, 104)) for v in finalists]
    assert len(quick) == 4
    assert len(full) == 2

def test_infrastructure_timeout_is_incomplete_and_never_a_win():
    result = evaluator.evaluate_reporting_panel(version, [slow_opponent], (201,))
    assert result.status == "incomplete"
    assert result.score is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/rollman/test_evaluator.py tests/rollman/test_opponents.py tests/rollman/test_diagnostics.py -q`

Expected: FAIL because staged methods and diagnostics do not exist.

- [ ] **Step 3: Implement staged evaluation using the existing match runner**

All phase names, seeds, paths, and match results are explicit. Finalist comparison
uses the same seed set. Reporting-panel results are hidden from candidate prompts.

- [ ] **Step 4: Implement opponent execution audit and empirical calibration**

Compile/start every opponent, run it against frozen reference Rollman policies,
record invalid/TLE infrastructure cases, and sort valid opponents by win rate,
Ghost-minus-Rollman margin, worst margin, then stable ID. Write
`opponent-audit.json` and `opponent-order.json` into the run.

- [ ] **Step 5: Implement primitive one-step diagnostics**

For each selected replay decision state, evaluate protocol actions 0–4 through
the frozen state tracker using the recorded same-round Ghost action. Return
legality, resulting position, immediate collision/shield event, score delta, item,
and portal effect. Mark the diagnostic as replay-fixed one-step hindsight, not a
deployment policy or tactical action space.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/rollman/test_evaluator.py tests/rollman/test_opponents.py tests/rollman/test_diagnostics.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/games/rollman/evaluator.py src/agentbench_frame/games/rollman/opponents.py src/agentbench_frame/games/rollman/diagnostics.py tests/rollman/test_evaluator.py tests/rollman/test_opponents.py tests/rollman/test_diagnostics.py
git commit -m "feat: stage valid Rollman candidate evaluation"
```

### Task 6: CLI Integration, Resume, and K=4 Reports

**Files:**
- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `src/agentbench_frame/hl/report.py`
- Modify: `tests/hl/test_curriculum_cli.py`
- Modify: `tests/hl/test_e2e_fake.py`
- Modify: `tests/hl/test_report.py`
- Modify: `docs/hl/rollman-local-run.md`

**Interfaces:**
- Consumes: K=4 config, proposal controller, research state, staged evaluator, selection state.
- Produces: one-command run/resume, integer proposal-cycle events, `curves.csv`, and four-panel PNG/SVG.

- [ ] **Step 1: Write failing end-to-end tests**

```python
def test_fake_k4_run_and_resume_reconstruct_identical_next_inputs(tmp_path):
    first = run_fake_k4(tmp_path, cycles=1)
    resumed = resume_fake_k4(first.run_dir, cycles=1)
    uninterrupted = run_fake_k4(tmp_path / "control", cycles=2)
    assert resumed.second_cycle_input_hash == uninterrupted.second_cycle_input_hash
    assert resumed.events.count("candidate_act_completed") == 8

def test_report_uses_integer_cycle_and_four_candidate_scatter(tmp_path):
    report = write_fixture_report(tmp_path, cycles=2, candidates=4)
    rows = list(csv.DictReader(report.curves_csv.open()))
    assert {int(row["proposal_cycle"]) for row in rows} == {0, 1, 2}
    assert sum(row["series"] == "candidate" and row["proposal_cycle"] == "1" for row in rows) == 4
```

- [ ] **Step 2: Run integration/report tests and verify RED**

Run: `.venv/bin/pytest tests/hl/test_curriculum_cli.py tests/hl/test_e2e_fake.py tests/hl/test_report.py -q`

Expected: FAIL because the CLI still runs one-stage `run_act` and reports one point
per selected coding act.

- [ ] **Step 3: Integrate proposal-cycle execution**

Replace the curriculum loop invocation with `run_proposal_cycle`. Run the fixed
reporting panel once for the selected search parent each cycle. Run hidden
multi-seed certification only after target gate success. Persist all recovery
boundaries so incomplete planner, candidate, reducer, measurement, or reporting
stages resume without repeating completed API acts.

- [ ] **Step 4: Implement the four requested panels**

Write behavioral IG, Elo, reporting-panel win rate, and score margin against
integer proposal cycle. Candidate branches are light scatter; selected search
parent is solid; official champion is a step line only on performance panels.
Keep cumulative coding acts/tokens/time in separate resource columns and plots.

- [ ] **Step 5: Update local run documentation**

Document `.env`, portable path variables, opponent audit, validate/audit/dry-run,
run/resume/report commands, pause semantics, artifact locations, and the fact that
no Codex App conversation participates in the loop.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/pytest tests/hl/test_curriculum_cli.py tests/hl/test_e2e_fake.py tests/hl/test_report.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/cli.py src/agentbench_frame/hl/report.py tests/hl/test_curriculum_cli.py tests/hl/test_e2e_fake.py tests/hl/test_report.py docs/hl/rollman-local-run.md
git commit -m "feat: run and report Rollman k4 proposal cycles"
```

### Task 7: Full Verification and Paid Experiment Launch

**Files:**
- Modify if required by verified failures: files already named in Tasks 1–6
- Generate: `.agentbench/29_rollman/runs/RUN_ID/*`

**Interfaces:**
- Consumes: complete K=4 harness, frozen game sources, human Ghost corpus, `.env` credential.
- Produces: validated run manifest, opponent audit, initial policy, proposal-cycle logs, replays, measurements, curves, and resumable checkpoints.

- [ ] **Step 1: Run focused suites**

Run:

```bash
.venv/bin/pytest tests/hl tests/rollman -q
```

Expected: all tests PASS with no warnings caused by the implementation.

- [ ] **Step 2: Run complete repository suite**

Run: `.venv/bin/pytest -q`

Expected: all tests PASS.

- [ ] **Step 3: Validate and audit the experiment**

Run:

```bash
.venv/bin/agentbench hl validate --config configs/hl/29_rollman-k4.yaml
.venv/bin/agentbench hl audit --config configs/hl/29_rollman-k4.yaml
.venv/bin/agentbench hl prepare-opponents --config configs/hl/29_rollman-k4.yaml --force
```

Expected: frozen source hashes match, all registered opponents are classified as
valid or excluded with an explicit infrastructure reason, and no secret appears
in generated artifacts.

- [ ] **Step 4: Run a no-cost fake/dry proposal cycle**

Run:

```bash
.venv/bin/agentbench hl run --dry-run \
  --config configs/hl/29_rollman-k4.yaml \
  --run-dir .agentbench/29_rollman/runs/run-rollman-k4-dry
```

Expected: one bootstrap path and one four-candidate proposal-cycle plan are
materialized without a paid model call.

- [ ] **Step 5: Start the paid run**

Run:

```bash
.venv/bin/agentbench hl run \
  --config configs/hl/29_rollman-k4.yaml \
  --run-dir .agentbench/29_rollman/runs/run-rollman-k4-gpt55
```

Expected: model bootstrap writes `v000000`, opponent certification establishes
the first valid target, and proposal cycle 1 starts with one planner plus four
candidate acts.

- [ ] **Step 6: Track and report until completion**

Monitor append-only events, provider usage, match validity, target progress,
research-state size, selected search parent, official champion, and four requested
curves. Resume after recoverable provider/credit interruptions. Do not change
frozen evaluation cases during the run. Completion requires one champion to pass
every valid registered human Ghost on the hidden certification protocol.
