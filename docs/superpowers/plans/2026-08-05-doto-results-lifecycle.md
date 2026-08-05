# DOTO Results and Run Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create independent DotoResults, implement atomic Codex-driven Run/iteration commands, finalize the sealed test once, and export a compatible projection to unchanged AgentBenchResults.

**Architecture:** DotoResults owns schemas, validation, aggregation, and reporting; AgentBenchFramework owns execution and atomic Run writes. A lock-protected state machine attaches immutable builds, 30-cell matrices, and strict IG to explicit parent/child iterations. Finalization seals a 56-cell test before a retryable five-file projection.

**Tech Stack:** Python 3.11, TOML/JSON/JSONL, `jsonschema>=4.23`, `fcntl`, atomic filesystem replacement, Jinja2/Chart.js, pytest.

## Global Constraints

- Authoritative path: `DotoResults/runs/23_doto/<agent>/<run_id>/`.
- AgentBenchResults is unchanged and receives a derived projection only.
- Codex never hand-writes authority files or curves.
- Every non-baseline iteration declares a built, closed parent.
- Formal score is non-null only for a complete 30-cell matrix.
- Final test has exactly 56 expected cells and one selected candidate.
- Failed/incomplete iterations remain null curve points.
- No hard budgets; observed counts and times remain recorded.
- `projection.json` is the only post-seal mutable file and is excluded from the sealed hash.
- JSON never contains `Infinity` or `NaN`.

---

## File Map

- New sibling repo directories: `../DotoResults/schemas`, `../DotoResults/src/doto_results`, `../DotoResults/tests`, and `../DotoResults/.github`.
- Replace: `src/agentbench_frame/doto/run_store.py`.
- Create: `src/agentbench_frame/doto/lifecycle.py`, `projection.py`.
- Modify: `src/agentbench_frame/doto/score.py`, `src/agentbench_frame/doto/ig.py`, and `src/agentbench_frame/doto/cli.py`.
- Create: `tests/doto/test_lifecycle.py`, `test_projection.py`, `test_codex_run_e2e.py`.
- Modify: `tests/doto/test_run_store.py`, `test_score.py`, `test_ig.py`, `test_cli.py`.

### Task 1: Scaffold Independent DotoResults

**Files:**
- Create: `../DotoResults/pyproject.toml`
- Create: `../DotoResults/src/doto_results/__init__.py`
- Create: `../DotoResults/src/doto_results/cli.py`
- Create: `../DotoResults/tests/test_cli.py`
- Create: `../DotoResults/README.md`

**Interfaces:** Produces `doto-results validate|aggregate|build-report|check-projection`.

- [ ] **Step 1: Initialize the sibling repository**

Run `mkdir -p ../DotoResults` then `git -C ../DotoResults init`. Do not nest it in Framework Git.

- [ ] **Step 2: Write failing parser test**

```python
def test_cli_operations():
    choices = build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {"validate", "aggregate", "build-report", "check-projection"}
```

- [ ] **Step 3: Add package metadata and parser**

Use project name `doto-results`, Python `>=3.11`, dependencies `jsonschema>=4.23` and `jinja2>=3.1`, and script `doto-results = "doto_results.cli:main"`. Until implemented, handlers return nonzero with `command not implemented`.

- [ ] **Step 4: Test and commit**

Run `uv run --with pytest python -m pytest -v` in DotoResults; expect PASS. Commit `chore: scaffold DOTO results repository`.

### Task 2: Define Schemas and Authority Validation

**Files:**
- Create: `../DotoResults/schemas/run-v1.json`, `iteration-v1.json`, `matrix-v1.json`, `score-curve-v1.json`, `ig-curve-v1.json`, `final-test-v1.json`, and `projection-ref-v1.json`
- Create: `../DotoResults/src/doto_results/schema.py`
- Create: `../DotoResults/tests/test_schema.py`

**Interfaces:** Produces `validate_run(Path) -> ValidationReport`, `sealed_content_hash(Path) -> str`.

- [ ] **Step 1: Write failing semantic tests**

```python
def test_complete_run_validates(complete_run):
    report = validate_run(complete_run)
    assert report.valid and report.errors == ()

def test_nonfinite_and_broken_parent_fail(broken_run):
    codes = {e.code for e in validate_run(broken_run).errors}
    assert codes >= {"nonfinite_json", "parent_not_closed"}
```

- [ ] **Step 2: Run failing test**

Run `uv run --with pytest python -m pytest tests/test_schema.py -v`; expect missing module/schema failure.

- [ ] **Step 3: Implement closed schemas and semantic checks**

```python
@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[ValidationError, ...]
    sealed_content_sha256: str | None

def validate_run(run_dir: Path) -> ValidationReport:
    documents = load_authority_documents_strict(run_dir)
    errors = tuple(validate_documents(run_dir, documents))
    digest = None if errors else sealed_content_hash(run_dir)
    return ValidationReport(not errors, errors, digest)

def sealed_content_hash(run_dir: Path) -> str:
    """Hash relative path plus bytes for authority files, excluding projection.json."""
```

Require identities, hashes, parent/state validity, 30/56 unique tasks, seed 11, seats 0/1, replay/trace evidence, curve alignment, one test identity, and null instead of non-finite values.

- [ ] **Step 4: Test and commit**

Run schema tests; expect PASS. Commit `feat: validate authoritative DOTO runs` in DotoResults.

### Task 3: Replace Loop Store with Atomic Lifecycle Storage

**Files:**
- Replace: `src/agentbench_frame/doto/run_store.py`
- Create: `src/agentbench_frame/doto/lifecycle.py`
- Modify: `tests/doto/test_run_store.py`
- Create: `tests/doto/test_lifecycle.py`

**Interfaces:** Produces `DotoRunStore.create/open`, `RunState`, `IterationState`, `init_run`, `begin_iteration`, `record_build`, `record_evaluation`, `record_ig`, `close_iteration`.

- [ ] **Step 1: Write failing storage tests**

```python
def test_init_snapshots_source_skills_and_pools(tmp_path):
    run = init_run(tmp_path, "codex", SOURCE, TRAIN, TEST, SKILLS, run_id="r1")
    assert run.state == RunState.CREATED
    assert (run.run_dir / "iterations/iteration-0000/playerAI.cpp").is_file()
    assert len(load_json(run.run_dir / "skills/manifest.json")["skills"]) == 4

def test_closed_iteration_cannot_be_overwritten(closed_run):
    with pytest.raises(InvalidTransition, match="closed"):
        record_build(closed_run, 1, OTHER_BUILD)
```

- [ ] **Step 2: Verify old store fails**

Run lifecycle/store tests; expect failure because current store requires `LoopConfig` and budgets.

- [ ] **Step 3: Implement state enums and atomic writes**

```python
class RunState(StrEnum):
    CREATED = "created"
    ITERATING = "iterating"
    FINAL_TEST_STARTED = "final_test_started"
    FINALIZED = "finalized"

class IterationState(StrEnum):
    OPEN = "iteration_open"
    BUILT = "candidate_built"
    BUILD_FAILED = "build_failed"
    EVALUATED = "training_evaluated"
    INCOMPLETE = "evaluation_incomplete"
    IG_RECORDED = "ig_recorded"
    IG_MISSING = "ig_missing"
    CLOSED = "iteration_closed"
```

Use `fcntl.flock` on `.run.lock`; write/fsync temporary files and directories before `os.replace`; append events while locked. Failed/incomplete iterations close as null but cannot be parents.

- [ ] **Step 4: Test and commit**

Run `tests/doto/test_run_store.py tests/doto/test_lifecycle.py`; expect PASS. Commit `feat(doto): add atomic Codex-driven lifecycle`.

### Task 4: Add Init, Status, Begin, and Build CLI

**Files:**
- Modify: `src/agentbench_frame/doto/cli.py`
- Modify: `tests/doto/test_cli.py`, `test_lifecycle.py`

**Interfaces:** Produces `doto run init|status` and `doto iteration begin|build`.

- [ ] **Step 1: Write failing JSON test**

```python
def test_run_init_prints_json(tmp_path, capsys):
    assert main(["run", "init", "--agent", "codex", "--initial-player-ai", str(SOURCE),
                 "--doto-results", str(tmp_path), "--run-id", "r1"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "created"
```

- [ ] **Step 2: Implement exact commands**

```text
doto run init --agent NAME --initial-player-ai PATH --doto-results PATH --train-manifest PATH --test-manifest PATH
doto run status --run-dir PATH
doto iteration begin --run-dir PATH --parent N --source PATH --analysis-file PATH
doto iteration build --run-dir PATH --iteration N
```

Resolve output from flag or `DOTO_RESULTS`, never Framework-local default. Status is read-only. Empty analysis fails. Build failure removes stale executable, preserves logs, and exits nonzero.

- [ ] **Step 3: Test and commit**

Run CLI/lifecycle tests; expect PASS. Commit `feat(doto): expose atomic run and build commands`.

### Task 5: Attach Evaluation, IG, and Closure

**Files:**
- Modify: `src/agentbench_frame/doto/lifecycle.py`, `score.py`, `ig.py`, and `cli.py`
- Modify: `tests/doto/test_lifecycle.py`, `test_score.py`, `test_ig.py`, and `test_cli.py`

**Interfaces:** Produces `doto iteration evaluate|compare|close` and parent-aligned curves.

- [ ] **Step 1: Write failing alignment/null tests**

```python
def test_close_aligns_parent_score_and_ig(complete_child):
    close_iteration(complete_child, 1)
    score = load_json(RUN / "score_curve.json")["points"][1]
    ig = load_json(RUN / "ig_curve.json")["points"][1]
    assert score["parent_iteration"] == 0 and score["evo"] is not None
    assert ig["version"] == score["version"]

def test_29_cells_closes_null(incomplete_child):
    close_iteration(incomplete_child, 1)
    point = load_json(RUN / "score_curve.json")["points"][1]
    assert point["evo"] is None and point["completion_rate"] == 29 / 30
```

- [ ] **Step 2: Implement commands**

`iteration evaluate` resumes absent cells only and stores immutable attempts. `iteration compare` runs parent/current on identical current-formal observations. `iteration close` atomically rebuilds both curves, retaining null failures.

- [ ] **Step 3: Test and commit**

Run lifecycle/score/IG/CLI tests; expect PASS. Commit `feat(doto): persist formal score and IG iterations`.

### Task 6: Finalize Sealed Test Exactly Once

**Files:**
- Modify: `src/agentbench_frame/doto/lifecycle.py`, `cli.py`
- Modify: `tests/doto/test_lifecycle.py`, `test_cli.py`

**Interfaces:** Produces `finalize_run(store: DotoRunStore, candidate_iteration: int, sealed_bundle: PopulationBundle, scheduler: AdaptiveScheduler) -> dict` and `doto run finalize`.

- [ ] **Step 1: Write failing one-shot tests**

```python
def test_finalize_runs_56_and_seals(ready_run, sealed_bundle):
    summary = finalize_run(ready_run, 3, sealed_bundle, fake_scheduler)
    assert summary["test_policy_count"] == 28
    assert summary["attempted_test_matches"] == 56
    assert ready_run.reload().state == RunState.FINALIZED

def test_interruption_cannot_change_candidate(interrupted):
    with pytest.raises(InvalidTransition, match="selected candidate"):
        finalize_run(interrupted, 4, SEALED, fake_scheduler)
```

- [ ] **Step 2: Implement durable selection/resume**

Write started state, selected hash, pool hash, and 56 task IDs before launch. Resume missing cells for the same identity only. A policy is defeated only with two normal cells and positive mean difference. Seal even if incomplete, retaining failures.

- [ ] **Step 3: Test and commit**

Expose `doto run finalize --run-dir PATH --candidate-iteration N --sealed-bundle-fd FD --cpu-target 70 [--workers N]`. Run final/sealed tests; commit `feat(doto): finalize sealed human policy evaluation`.

### Task 7: Export AgentBenchResults Projection

**Files:**
- Create: `src/agentbench_frame/doto/projection.py`, `tests/doto/test_projection.py`
- Modify: `src/agentbench_frame/doto/cli.py`
- Create: `../DotoResults/src/doto_results/projection.py`, `tests/test_projection.py`

**Interfaces:** Produces `export_agentbench_projection`, `doto run export`, `check_projection`.

- [ ] **Step 1: Write failing file-set test**

```python
def test_projection_file_set(sealed_run, target):
    path = export_agentbench_projection(sealed_run, target).path
    assert {p.name for p in path.iterdir()} == {
        "run.toml", "summary.json", "score_curve.json", "ig_curve.json", "doto_results_ref.json"
    }
```

- [ ] **Step 2: Implement atomic derived export**

Write five files in a sibling temporary directory; add scalar `wall_hours`, `total_steps`, `win_rate`; write authoritative path/hash; validate and rename. Existing identical destination is idempotent, differing destination fails. Record attempts only in post-seal `projection.json`.

- [ ] **Step 3: Verify unchanged aggregator compatibility**

DotoResults validates file set/types/hashes, rejects non-finite JSON, runs unchanged `AgentBenchResults/scripts/aggregate.py` on a fixture, and requires Run discovery.

- [ ] **Step 4: Test and commit separately**

Run both repositories' projection tests. Commit `feat(doto): export generic result projection` in Framework and `feat: validate AgentBench result projections` in DotoResults.

### Task 8: Aggregate and Render DOTO Results

**Files:**
- Create: `../DotoResults/src/doto_results/aggregate.py` and `report.py`
- Create: `../DotoResults/src/doto_results/templates/index.html` and `run.html`
- Modify: `../DotoResults/src/doto_results/cli.py`, `README.md`
- Create: `../DotoResults/tests/test_aggregate.py`, `test_report.py`
- Create: `../DotoResults/.github/workflows/validate-and-publish.yml`

**Interfaces:** Produces `aggregate_runs(root) -> Registry`, `build_report(root, output)`.

- [ ] **Step 1: Write failing metric/report tests**

```python
def test_registry_surfaces_primary_metric(valid_root):
    run = aggregate_runs(valid_root).runs[0]
    assert (run.defeated_human_policy_count, run.test_policy_count) == (17, 28)

def test_page_has_score_ig_policy_sections(valid_root, tmp_path):
    build_report(valid_root, tmp_path)
    page = (tmp_path / "runs/r1.html").read_text()
    assert all(x in page for x in ("Score vs. iteration", "Strict KL status", "17 / 28"))
```

- [ ] **Step 2: Implement validation-gated aggregation/report/CI**

Invalid Runs fail CI and appear in an error report. Render curves, policy tables, parent graph, failures, CPU history, and provenance; link rather than embed replay/trace. CI tests, validates, aggregates, builds, then publishes.

- [ ] **Step 3: Test and commit**

Run all DotoResults tests; commit `feat: publish DOTO benchmark results`.

### Task 9: Lifecycle E2E

**Files:**
- Create: `tests/doto/test_codex_run_e2e.py`
- Modify: `docs/doto-official-acceptance.md`
- Modify: `../DotoResults/README.md`

**Interfaces:** Produces fake-server authoritative Run and valid projection fixture.

- [ ] **Step 1: Write orchestration-free E2E**

```python
def test_atomic_workflow_produces_both_targets(tmp_path):
    run_dir = execute_fake_codex_workflow(tmp_path)
    assert validate_with_doto_results(run_dir).valid
    assert (tmp_path / "AgentBenchResults/runs/23_doto/codex/r1/summary.json").is_file()
```

The helper calls lifecycle functions, not `loop.py`: baseline build/evaluate/close; child begin/build/evaluate/compare/close; 56-cell finalize; validate; export.

- [ ] **Step 2: Run both full suites**

Framework: `uv run --extra doto --with pytest python -m pytest -q`. DotoResults: `uv run --with pytest python -m pytest -q`. Expected: PASS.

- [ ] **Step 3: Document and commit**

Document sealed FD handoff, 70-percent scheduling, 30/56 matrices, validator, projection, and leak scans. Commit Framework test/docs and DotoResults README separately.
