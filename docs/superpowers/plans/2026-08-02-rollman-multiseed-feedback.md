# Rollman Multi-Seed Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make staged Rollman evaluation honor configurable common quick-screen and disjoint finalist seed counts, while preserving bounded research knowledge when an experiment imports a version into a new run.

**Architecture:** `RollmanEvaluator` owns a validated deterministic split of the fixed learning seeds. The CLI passes `IterationConfig` counts into that evaluator before any provider preflight. Imported-run initialization optionally copies the bounded reducer state in the same way Experience state is inherited, and a v6 experiment configuration enables the 2+2 protocol.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, pytest, append-only JSONL events, Codex Responses provider.

## Global Constraints

- Four siblings share identical quick-screen seeds.
- Finalist seeds are disjoint from quick-screen seeds.
- Fixed certification seeds remain inaccessible to coding acts.
- Complete replay and trace contents are never embedded in prompts.
- `quick_screen_seeds`, `finalist_seeds`, and research-state inheritance are configuration-controlled ablation parameters.
- No provider act may run when the staged seed split or imported state is invalid.

---

### Task 1: Deterministic staged seed split

**Files:**
- Modify: `src/agentbench_frame/games/rollman/evaluator.py`
- Test: `tests/rollman/test_evaluator.py`

**Interfaces:**
- Consumes: `fixed_gate_seeds: Iterable[int]`, `quick_screen_seed_count: int = 1`, `finalist_seed_count: int | None = None`.
- Produces: `RollmanEvaluator.quick_screen()` on the first configured seed slice and `evaluate_finalist()` on the following disjoint slice.

- [ ] **Step 1: Write failing evaluator tests**

Add a 2+2 test that constructs the evaluator with:

```python
quick_screen_seed_count=2,
finalist_seed_count=2,
```

and asserts quick seeds `[101, 102]`, finalist seeds `[103, 104]`, and combined seeds `[101, 102, 103, 104]`. Add parametrized invalid cases for zero counts and for a sum greater than four.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest -q tests/rollman/test_evaluator.py -k 'staged or seed_count'`

Expected: failure because the evaluator constructor has no staged-count parameters.

- [ ] **Step 3: Implement the validated split**

Store:

```python
self.quick_screen_seed_count = int(quick_screen_seed_count)
self.finalist_seed_count = (
    len(self.fixed_gate_seeds) - self.quick_screen_seed_count
    if finalist_seed_count is None
    else int(finalist_seed_count)
)
```

Reject non-positive counts and `quick + finalist > len(fixed_gate_seeds)`. Slice with:

```python
quick = self.fixed_gate_seeds[: self.quick_screen_seed_count]
start = self.quick_screen_seed_count
finalist = self.fixed_gate_seeds[start : start + self.finalist_seed_count]
```

- [ ] **Step 4: Run evaluator tests and verify GREEN**

Run: `.venv/bin/pytest -q tests/rollman/test_evaluator.py`

Expected: all evaluator tests pass and the existing default 1+remaining behavior is preserved.

- [ ] **Step 5: Commit the evaluator unit**

```bash
git add src/agentbench_frame/games/rollman/evaluator.py tests/rollman/test_evaluator.py
git commit -m "feat: configure Rollman staged seed split"
```

### Task 2: CLI wiring and preflight visibility

**Files:**
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_cli.py`
- Test: `tests/hl/test_config.py`

**Interfaces:**
- Consumes: `config.run.iteration.quick_screen_seeds` and `config.run.iteration.finalist_seeds`.
- Produces: evaluator constructor arguments and dry-run JSON fields with the exact configured counts.

- [ ] **Step 1: Write failing CLI/config tests**

Assert dry-run output contains:

```python
assert result["quick_screen_seeds"] == 2
assert result["finalist_seeds"] == 2
```

and assert the evaluator factory receives both counts. Add a config parse assertion for the 2+2 values.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_cli.py tests/hl/test_config.py -k 'quick_screen or finalist'`

Expected: failure because dry-run and evaluator construction do not expose or pass the counts.

- [ ] **Step 3: Wire the configuration**

Add to `_validate()` output:

```python
"quick_screen_seeds": config.run.iteration.quick_screen_seeds,
"finalist_seeds": config.run.iteration.finalist_seeds,
```

and construct `RollmanEvaluator` with:

```python
quick_screen_seed_count=config.run.iteration.quick_screen_seeds,
finalist_seed_count=config.run.iteration.finalist_seeds,
```

The evaluator constructor validation must execute before `_preflight_provider(provider)`.

- [ ] **Step 4: Run focused HL tests and verify GREEN**

Run: `.venv/bin/pytest -q tests/hl/test_cli.py tests/hl/test_config.py tests/rollman/test_evaluator.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit CLI wiring**

```bash
git add src/agentbench_frame/hl/cli.py tests/hl/test_cli.py tests/hl/test_config.py
git commit -m "feat: wire staged feedback counts"
```

### Task 3: Imported research-state continuation

**Files:**
- Modify: `src/agentbench_frame/hl/config.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_config.py`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Consumes: `OriginConfig.reset_research_state: bool = True` and `<source_run>/research_state.json`.
- Produces: a size-bounded copy at `<new_run>/research_state.json` before `ResearchState.empty()` is considered.

- [ ] **Step 1: Write failing inheritance tests**

Create a source run containing a valid research state and configure:

```python
origin={
    "mode": "imported_version",
    "source_run": str(source_run),
    "source_version": "v000000",
    "reset_research_state": False,
}
```

Assert the destination bytes equal the source bytes. Assert a missing source file and a file larger than `research_state_max_bytes` fail before provider use. Assert the default remains `True`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest -q tests/hl/test_config.py tests/hl/test_cli.py -k research_state`

Expected: failure because the origin field and copy helper do not exist.

- [ ] **Step 3: Implement bounded inheritance**

Add `reset_research_state: bool = True` to `OriginConfig`. Add a helper with the interface:

```python
def _prepare_research_state(
    config: LocalHLConfig, *, run_dir: Path, resume: bool
) -> Path:
    destination = run_dir / "research_state.json"
    if resume:
        return destination
    if (
        config.run.origin.mode == "imported_version"
        and not config.run.origin.reset_research_state
    ):
        source = Path(config.run.origin.source_run).resolve() / "research_state.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"imported origin research state is missing: {source}"
            )
        limit = config.run.context.research_state_max_bytes
        if source.stat().st_size > limit:
            raise ValueError(f"source research state exceeds max_bytes={limit}")
        ResearchState.load_or_create(source, max_bytes=limit)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    if not destination.is_file():
        ResearchState.empty(
            max_bytes=config.run.context.research_state_max_bytes
        ).write(destination)
    return destination
```

For a new imported run with reset disabled, require the source file, reject it when its size exceeds `config.run.context.research_state_max_bytes`, validate it with `ResearchState.load_or_create`, copy it to the destination, and return the destination. Otherwise write `ResearchState.empty(max_bytes=config.run.context.research_state_max_bytes)` only when the destination does not exist.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `.venv/bin/pytest -q tests/hl/test_config.py tests/hl/test_cli.py -k research_state`

Expected: all research-state tests pass.

- [ ] **Step 5: Commit state continuation**

```bash
git add src/agentbench_frame/hl/config.py src/agentbench_frame/hl/cli.py tests/hl/test_config.py tests/hl/test_cli.py
git commit -m "feat: inherit Rollman research state"
```

### Task 4: v6 experiment and end-to-end verification

**Files:**
- Create: `configs/hl/29_rollman-k4-repair-v6-multiseed.yaml`
- Test: `tests/hl/test_cli.py`

**Interfaces:**
- Consumes: v5 run `v000000`, Experience state, research state, and the frozen official game roots.
- Produces: a fresh reproducible run using `k=4`, quick seeds 2, finalist seeds 2, repair top-k 2, and hidden five-seed certification.

- [ ] **Step 1: Add the v6 config**

Copy the v5 experiment settings and set:

```yaml
origin:
  source_run: ".agentbench/29_rollman/runs/run-20260802-gpt55-k4-repair-v5-budget"
  source_version: "v000000"
  reset_experience: false
  reset_research_state: false
iteration:
  quick_screen_seeds: 2
  finalist_seeds: 2
paths:
  workspace: ".agentbench/29_rollman/candidate-k4-repair-v6-multiseed"
```

- [ ] **Step 2: Validate without a credential**

Run: `AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST .venv/bin/python -m agentbench_frame.hl.cli dry-run --config configs/hl/29_rollman-k4-repair-v6-multiseed.yaml --run-dir /private/tmp/rollman-v6-dry-run`

Expected: valid JSON containing quick-screen `2`, finalist `2`, and the exact source content hash.

- [ ] **Step 3: Run complete automated verification**

Run:

```bash
.venv/bin/pytest -q tests/hl tests/rollman
.venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit the runnable experiment**

```bash
git add configs/hl/29_rollman-k4-repair-v6-multiseed.yaml tests/hl/test_cli.py
git commit -m "exp: add Rollman multiseed continuation"
```

- [ ] **Step 5: Launch one proposal cycle and monitor artifacts**

Run the v6 config with a unique run directory and `--acts 1`. Verify four initial candidates each contain seeds 101 and 102, only two finalists contain seeds 103 and 104, the research state records v5 facts, and the aggregate report adds one integer iteration point.
