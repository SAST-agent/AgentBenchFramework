# DOTO Population and Parallel Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the 43-policy DOTO population, expose 15 training policies and keep 28 hidden, sealed test policies evaluator-only, and evaluate complete seat-balanced matrices with adaptive parallelism targeting 70% CPU.

**Architecture:** Identify historical policies by complete-directory SHA-256, not `playerAI.cpp`. Build public training and evaluator-owned sealed bundles, represent every `(policy, seed, seat)` as an isolated task, and aggregate task outcomes in manifest order through a bounded adaptive scheduler.

**Tech Stack:** Python 3.11, standard library concurrency/filesystem modules, existing native DOTO protocol and match runner, `psutil>=6.0`, pytest.

## Global Constraints

- Exactly 43 distinct complete snapshots: 15 train and 28 hidden, sealed test policies.
- Formal seed is `11`; seats are `0` and `1`.
- Training matrix size is 30; final test matrix size is 56.
- Scheduler target/lower/upper CPU thresholds are 70/65/75 percent.
- `--workers` is a hard ceiling; running formal matches are never killed for throttling.
- Official matches remain 300 seconds, 20 FPS, `realtime_scale=1.0`.
- Test sources and executable paths never enter Codex-readable output.
- Failures remain explicit and aggregation order follows the manifest.

---

## File Map

- Modify `pyproject.toml`: DOTO extra and pytest import mode.
- Replace `src/agentbench_frame/doto/population.toml`: versioned 43-policy manifest.
- Modify `src/agentbench_frame/doto/population.py`: manifest/bundle validation and building.
- Create `src/agentbench_frame/doto/scheduler.py`: adaptive executor.
- Create `src/agentbench_frame/doto/evaluation.py`: formal matrix construction and aggregation.
- Modify `src/agentbench_frame/doto/cli.py`: population and evaluate commands.
- Create `scripts/import_doto_population.py`: maintainer-only corpus migration.
- Create `tests/doto/test_scheduler.py`, `tests/doto/test_evaluation.py`.
- Modify `tests/doto/test_population.py`, `test_cli.py`.
- Create `tests/doto/__init__.py`, `tests/miracle/__init__.py`.

### Task 1: Freeze Full-Snapshot Population Identity

**Files:**
- Replace: `src/agentbench_frame/doto/population.toml`
- Modify: `src/agentbench_frame/doto/population.py`
- Modify: `tests/doto/test_population.py`

**Interfaces:**
- Produces `PopulationManifest`, `PopulationPolicy`, `load_population(Path) -> PopulationManifest`, and `source_hash(Path) -> str`.

- [ ] **Step 1: Write failing identity tests**

```python
def test_manifest_freezes_43_distinct_complete_snapshots():
    manifest = load_population(MANIFEST)
    assert manifest.schema_version == 2
    assert len(manifest.policies) == 43
    assert len({p.source_sha256 for p in manifest.policies}) == 43
    assert sum(p.split == "train" for p in manifest.policies) == 15
    assert sum(p.split == "test" for p in manifest.policies) == 28

def test_auxiliary_source_changes_policy_identity(tmp_path):
    (tmp_path / "playerAI.cpp").write_text("same")
    (tmp_path / "Attack.cpp").write_text("a")
    before = source_hash(tmp_path)
    (tmp_path / "Attack.cpp").write_text("b")
    assert source_hash(tmp_path) != before
```

- [ ] **Step 2: Confirm failure against the old three-entry list**

Run: `uv run --with pytest python -m pytest tests/doto/test_population.py -v`

Expected: FAIL because the loader returns a list and the manifest has three rows.

- [ ] **Step 3: Implement exact manifest types**

```python
@dataclass(frozen=True)
class PopulationPolicy:
    policy_id: str
    name: str
    source: Path | None
    origin: str
    origin_split: str
    split: Literal["train", "test"]
    source_sha256: str
    leaderboard: bool
    enabled: bool = True

@dataclass(frozen=True)
class PopulationManifest:
    schema_version: int
    benchmark_version: str
    game: str
    seed: int
    seats: tuple[int, ...]
    policies: tuple[PopulationPolicy, ...]
```

Require schema 2, game `23_doto`, seed 11, seats `(0, 1)`, unique names/IDs/full hashes, lowercase 64-character hashes, and exact 15/28 counts.

- [ ] **Step 4: Populate all 43 rows**

Transcribe the sibling corpus manifest; map original `train` to train and original `validation|test` to test. Preserve original split and leaderboard. Compute policy ID and expected hash over relative path plus bytes for every file.

- [ ] **Step 5: Run and commit**

```bash
uv run --with pytest python -m pytest tests/doto/test_population.py -v
git add src/agentbench_frame/doto/population.py src/agentbench_frame/doto/population.toml tests/doto/test_population.py
git commit -m "feat(doto): freeze full-snapshot human population"
```

### Task 2: Add Maintainer-Only Migration and Sealed Bundles

**Files:**
- Create: `scripts/import_doto_population.py`
- Create: `src/agentbench_frame/doto/human_policies/train/.gitkeep`
- Modify: `src/agentbench_frame/doto/population.py`
- Modify: `.gitignore`
- Modify: `tests/doto/test_population.py`

**Interfaces:**
- Produces `audit_population`, `materialize_training_sources`, `build_sealed_test_bundle`, `load_sealed_bundle`.

- [ ] **Step 1: Write failing migration/redaction tests**

```python
def test_audit_detects_no_fixture_hash_mismatch(corpus_fixture):
    report = audit_population(corpus_fixture.root, corpus_fixture.manifest)
    assert report["hash_mismatches"] == []

def test_sealed_report_has_no_paths_or_source(sealed_report):
    text = json.dumps(sealed_report)
    assert "source" not in text
    assert "executable" not in text
    assert "/tmp/" not in text
```

- [ ] **Step 2: Confirm missing APIs**

Run: `uv run --with pytest python -m pytest tests/doto/test_population.py -k 'audit or sealed' -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement three importer subcommands**

```text
scripts/import_doto_population.py audit --corpus-root PATH
scripts/import_doto_population.py materialize-training --corpus-root PATH --destination PATH
scripts/import_doto_population.py build-sealed --corpus-root PATH --destination PATH
```

`materialize-training` copies only 15 train directories into the Framework. `build-sealed` copies test sources to a temporary build root, builds/smoke-tests, moves only runtime bundles into an evaluator-owned destination, and writes a path-redacted manifest. Add temporary/sealed outputs to `.gitignore`.

- [ ] **Step 4: Audit and materialize the real corpus**

```bash
uv run python scripts/import_doto_population.py audit --corpus-root ../AgentBench/backend_sources/corpus/23_doto
uv run python scripts/import_doto_population.py materialize-training \
  --corpus-root ../AgentBench/backend_sources/corpus/23_doto \
  --destination src/agentbench_frame/doto/human_policies/train
```

Expected: 43 verified, 15 copied, 28 sealed declarations, zero mismatches, and no test source under the destination.

- [ ] **Step 5: Run and commit**

```bash
uv run --with pytest python -m pytest tests/doto/test_population.py -v
git add .gitignore scripts/import_doto_population.py src/agentbench_frame/doto/population.py src/agentbench_frame/doto/human_policies/train tests/doto/test_population.py
git commit -m "feat(doto): migrate isolated human policy pools"
```

### Task 3: Expose Population Bundle Commands

**Files:**
- Modify: `src/agentbench_frame/doto/population.py`
- Modify: `src/agentbench_frame/doto/cli.py`
- Modify: `tests/doto/test_population.py`
- Modify: `tests/doto/test_cli.py`

**Interfaces:**
- Produces `build_training_bundle(manifest: PopulationManifest, source_root: Path, output_dir: Path) -> PopulationBuildResult` and CLI `population verify|build-train|verify-sealed`.

- [ ] **Step 1: Write failing bundle tests**

```python
def test_training_bundle_builds_and_hashes_every_policy(tiny_manifest, tmp_path):
    result = build_training_bundle(tiny_manifest, tiny_manifest.root, tmp_path)
    assert all(row.status == "ready" for row in result.policies)
    assert all(row.executable_sha256 for row in result.policies)

def test_sealed_bundle_rejects_wrong_benchmark_version(sealed_bundle):
    with pytest.raises(ValueError, match="benchmark version"):
        load_sealed_bundle(sealed_bundle, "wrong")
```

- [ ] **Step 2: Confirm failure**

Run: `uv run --with pytest python -m pytest tests/doto/test_population.py tests/doto/test_cli.py -k bundle -v`

- [ ] **Step 3: Implement explicit CLI and retire overloaded population build**

```text
doto population verify --manifest PATH --source-root PATH
doto population build-train --manifest PATH --source-root PATH --output-dir PATH
doto population verify-sealed --manifest PATH --bundle-fd FD
```

Keep `doto build --player-ai` for candidates. Every population build removes stale binaries, uses an isolated directory, performs protocol smoke, and records source/executable hashes. Sealed stdout exposes policy IDs/status/hashes only.

- [ ] **Step 4: Run and commit**

```bash
uv run --with pytest python -m pytest tests/doto/test_population.py tests/doto/test_cli.py -v
git add src/agentbench_frame/doto/population.py src/agentbench_frame/doto/cli.py tests/doto/test_population.py tests/doto/test_cli.py
git commit -m "feat(doto): build public and sealed policy bundles"
```

### Task 4: Implement Adaptive Scheduling

**Files:**
- Modify: `pyproject.toml`
- Create: `src/agentbench_frame/doto/scheduler.py`
- Create: `tests/doto/test_scheduler.py`
- Create: `tests/doto/__init__.py`
- Create: `tests/miracle/__init__.py`

**Interfaces:**
- Produces `SchedulerConfig`, `TaskOutcome`, `SchedulerReport`, `AdaptiveScheduler.run(tasks, worker)`.

- [ ] **Step 1: Add dependency/test configuration**

```toml
[project.optional-dependencies]
doto = ["psutil>=6.0"]

[tool.pytest.ini_options]
addopts = ["--import-mode=importlib"]
```

- [ ] **Step 2: Write deterministic fake-sampler tests**

```python
def test_scheduler_grows_holds_and_throttles_without_cancelling():
    scheduler = AdaptiveScheduler(
        SchedulerConfig(target_cpu=70, lower_cpu=65, upper_cpu=75,
                        max_workers=8, sample_seconds=.01),
        cpu_sampler=FakeCpuSampler([40, 62, 69, 81, 82, 70]),
    )
    report = scheduler.run(make_tasks(12), successful_worker)
    assert max(report.worker_history) > 1
    assert report.cancelled_task_ids == []
    assert [x.task_id for x in report.outcomes] == [str(i) for i in range(12)]
```

- [ ] **Step 3: Confirm failure**

Run: `uv run --extra doto --with pytest python -m pytest tests/doto/test_scheduler.py -v`

- [ ] **Step 4: Implement bounded submission**

Use `ThreadPoolExecutor` because workers supervise subprocesses. Limit outstanding futures to current concurrency; sample the current process plus recursive children with psutil; grow by one below 65; hold within band; reduce future submission ceiling after two samples above 75; never cancel running futures; return outcomes in input order.

- [ ] **Step 5: Run and commit**

```bash
uv run --extra doto --with pytest python -m pytest tests/doto/test_scheduler.py -v
git add pyproject.toml tests/doto/__init__.py tests/miracle/__init__.py src/agentbench_frame/doto/scheduler.py tests/doto/test_scheduler.py
git commit -m "feat(doto): schedule matches at adaptive CPU target"
```

### Task 5: Implement Complete Matrix Evaluation

**Files:**
- Create: `src/agentbench_frame/doto/evaluation.py`
- Create: `tests/doto/test_evaluation.py`
- Modify: `src/agentbench_frame/doto/cli.py`
- Modify: `tests/doto/test_cli.py`

**Interfaces:**
- Produces `MatrixSpec.formal_train`, `MatrixSpec.final_test`, `evaluate_matrix`, `aggregate_matrix`.
- Consumes `PopulationBundle`, `AdaptiveScheduler`, `run_match`.

- [ ] **Step 1: Write failing matrix tests**

```python
def test_matrix_sizes(train_bundle, sealed_bundle):
    assert len(MatrixSpec.formal_train(train_bundle, seed=11).tasks) == 30
    assert len(MatrixSpec.final_test(sealed_bundle, seed=11).tasks) == 56

def test_incomplete_cell_nulls_formal_score():
    summary = aggregate_matrix(two_seat_rows([2.0, None], error="ai_timeout"))
    assert summary["formal_score"] is None
    assert summary["completion_rate"] == .5
    assert summary["defeated_policy_count"] == 0
```

- [ ] **Step 2: Confirm failure**

Run: `uv run --extra doto --with pytest python -m pytest tests/doto/test_evaluation.py -v`

- [ ] **Step 3: Implement records and aggregation**

```python
@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    opponent_id: str
    opponent_name: str
    seed: int
    candidate_seat: int

@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    terminated_by: str
    scores: tuple[float, float] | None
    score_diff: float | None
    replay_path: str | None
    trace_path: str | None
    error: str | None
```

Give each task unique output paths. A formal score exists only at 100 percent completion. A policy is defeated only if all expected cells are normal and oriented mean score difference is positive.

- [ ] **Step 4: Expose evaluation CLI**

```text
doto evaluate --candidate PATH --pool PATH --split train|test \
  --output-dir PATH --seed 11 --cpu-target 70 [--workers N]
```

Test split requires an evaluator-supplied sealed FD and never prints paths.

- [ ] **Step 5: Run and commit**

```bash
uv run --extra doto --with pytest python -m pytest tests/doto/test_evaluation.py tests/doto/test_cli.py tests/doto/test_match.py -v
git add src/agentbench_frame/doto/evaluation.py src/agentbench_frame/doto/cli.py tests/doto/test_evaluation.py tests/doto/test_cli.py
git commit -m "feat(doto): evaluate complete seat-balanced matrices"
```

### Task 6: Phase Verification

**Files:**
- Modify: `docs/doto-official-acceptance.md`

**Interfaces:**
- Produces documented public 30-cell and sealed 56-cell commands for the lifecycle phase.

- [ ] **Step 1: Document exact fake and official commands**

Include pool verification, CPU telemetry, path isolation, 30/56 expected cells, and the statement that official-duration acceptance is opt-in.

- [ ] **Step 2: Run all tests together**

Run: `uv run --extra doto --with pytest python -m pytest -q`

Expected: all AquaWar, Miracle, and DOTO tests collect together and pass.

- [ ] **Step 3: Run a 30-cell fake-server smoke**

```bash
uv run --extra doto python -m agentbench_frame.doto evaluate \
  --candidate tests/doto/fixtures/scripted_ai.py \
  --pool tests/doto/fixtures/train-bundle --split train \
  --output-dir /tmp/doto-matrix-smoke --seed 11 \
  --cpu-target 70 --workers 8 --test-only
```

Expected: 30 ordered cells, unique artifacts, completion 1.0, and CPU telemetry.

- [ ] **Step 4: Commit docs**

```bash
git add docs/doto-official-acceptance.md
git commit -m "docs(doto): verify full parallel training matrix"
```
