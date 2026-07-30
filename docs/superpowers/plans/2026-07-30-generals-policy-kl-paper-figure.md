# Generals Policy-KL Paper Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible, English-only, publication-ready horizontal three-panel figure for the finalized Generals controlled-reference policy-KL run.

**Architecture:** A small Generals figure module validates and projects the finalized `summary.json` and `events.jsonl` into typed chart data, then renders the three panels with Matplotlib. A thin command-line script supplies explicit input/output paths, and committed SVG/PNG artifacts are regenerated only from saved first-hand run data.

**Tech Stack:** Python 3.11, standard-library `json`/`dataclasses`, Matplotlib 3.8+, pytest, SVG, PNG.

## Global Constraints

- Source run is exactly `20260730_1126_8d123b55`.
- Scientific values are read from saved artifacts; plotting code contains no duplicated KL or support values.
- Expected transitions are `v0→v1` through `v5→v6`.
- Expected epsilons are `0.001`, `0.01`, `0.05`, and `0.1`; primary epsilon is `0.01`.
- Expected reference coverage is 12 states.
- Missing values remain gaps; incomplete coverage is never averaged or interpolated.
- Figure language is English only.
- Outputs are SVG and 300 DPI PNG.
- Use a white academic style with light gridlines and a colorblind-safe palette.
- Use `apply_patch` for source changes and TDD for behavior.

---

## File and responsibility map

- Create `src/agentbench_frame/generals/paper_figure.py` — artifact validation, typed chart data, and rendering.
- Create `scripts/plot_generals_controlled_policy_kl.py` — explicit CLI wrapper.
- Create `tests/generals/test_policy_kl_figure.py` — data integrity, missingness, SVG, and PNG contracts.
- Modify `pyproject.toml` — add a `figures` optional dependency and include it in `all`.
- Create `docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg` — paper vector artifact.
- Create `docs/experiments/figures/generals-controlled-policy-kl-three-panel.png` — 300 DPI preview artifact.

### Task 1: Validated figure-data projection

**Files:**
- Create: `src/agentbench_frame/generals/paper_figure.py`
- Create: `tests/generals/test_policy_kl_figure.py`

**Interfaces:**
- Consumes: `load_policy_kl_figure_data(run_dir: Path)`.
- Produces: immutable `PolicyKLFigureData` containing six transitions, four sensitivity series, and 12 `SupportStatePoint` records.

- [ ] **Step 1: Write failing complete-data projection test**

Create a temporary finalized run whose summary contains six transition records
and whose event stream contains 12 unique `action_space_count` events. Assert:

```python
data = load_policy_kl_figure_data(run_dir)
assert data.transitions == (
    "v0→v1", "v1→v2", "v2→v3",
    "v3→v4", "v4→v5", "v5→v6",
)
assert data.primary_kl == (0.0, 0.0, 5.4, 8.9, 8.9, 6.5)
assert tuple(data.sensitivity) == ("0.001", "0.01", "0.05", "0.1")
assert len(data.support_states) == 12
assert data.support_states[0].decision_number == 2
assert data.support_states[-1].decision_number == 10
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/generals/test_policy_kl_figure.py::test_loads_complete_figure_data -q
```

Expected: import failure because `paper_figure.py` does not exist.

- [ ] **Step 3: Implement immutable records and strict loader**

Define:

```python
@dataclass(frozen=True)
class SupportStatePoint:
    state_id: str
    seed: int
    seat: int
    decision_number: int
    support_size: int | None
    status: str


@dataclass(frozen=True)
class PolicyKLFigureData:
    run_id: str
    transitions: tuple[str, ...]
    primary_epsilon: str
    primary_kl: tuple[float | None, ...]
    sensitivity: dict[str, tuple[float | None, ...]]
    support_states: tuple[SupportStatePoint, ...]
```

`load_policy_kl_figure_data()` must require:

- `status == "complete"`;
- metric name `controlled_reference_policy_kl`;
- primary epsilon `0.01`;
- exact ordered transitions and epsilon set;
- 12/12 coverage for every saved complete aggregate;
- exactly 12 unique count events;
- count rows sorted by decision number, seed, then seat; and
- decimal support values parsed to arbitrary-precision Python integers.

- [ ] **Step 4: Add failing malformed-data tests**

Add separate tests that require clear `ValueError` messages for:

- duplicate `measurement_state_id`;
- missing transition;
- wrong metric name; and
- a transition with coverage 11/12 paired with a non-null aggregate.

- [ ] **Step 5: Run projection tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_figure.py -q
```

Expected: all loader tests pass.

- [ ] **Step 6: Commit the projection**

```bash
git add src/agentbench_frame/generals/paper_figure.py \
  tests/generals/test_policy_kl_figure.py
git commit -m "feat(generals): project policy KL figure data"
```

### Task 2: Three-panel SVG and PNG renderer

**Files:**
- Modify: `src/agentbench_frame/generals/paper_figure.py`
- Modify: `tests/generals/test_policy_kl_figure.py`
- Modify: `pyproject.toml`
- Create: `scripts/plot_generals_controlled_policy_kl.py`

**Interfaces:**
- Consumes: `PolicyKLFigureData`.
- Produces: `render_policy_kl_three_panel(data, output_prefix: Path) -> tuple[Path, Path]`.

- [ ] **Step 1: Write failing render contract test**

Render a complete fixture and assert:

```python
svg_path, png_path = render_policy_kl_three_panel(data, output_prefix)
assert svg_path.read_text().count("Exact Canonical Support Size") == 1
assert "Controlled-reference Policy KL over Iterations" in svg_path.read_text()
assert "Epsilon Sensitivity" in svg_path.read_text()
assert png_path.read_bytes().startswith(b"\\x89PNG\\r\\n\\x1a\\n")
assert read_png_dimensions(png_path)[0] >= 6000
```

Set `matplotlib.rcParams["svg.fonttype"] = "none"` so English labels remain
machine-verifiable SVG text.

- [ ] **Step 2: Run the render test and verify RED**

Expected: import failure for the missing renderer.

- [ ] **Step 3: Add the optional plotting dependency**

Update:

```toml
[project.optional-dependencies]
figures = ["matplotlib>=3.8"]
all = ["psutil", "torch", "numpy", "jinja2", "matplotlib>=3.8"]
```

Install into the worktree environment:

```bash
uv pip install --python .venv/bin/python "matplotlib>=3.8"
```

- [ ] **Step 4: Implement the three panels**

Use a `24 × 7.2` inch figure:

- Panel A: dark-blue primary KL line, circle markers, value annotations, and
  gaps for `None`.
- Panel B: four colorblind-safe sensitivity lines with distinct markers and a
  compact legend.
- Panel C: log-scale bars, compact state labels, orange tones keyed by
  decision 2/10, and exact integer annotations.

Apply panel letters, white background, dashed light-grey y grids, restrained
spines, `tight_layout`, and fixed 300 DPI output.

- [ ] **Step 5: Add explicit CLI wrapper**

Implement:

```text
plot_generals_controlled_policy_kl.py
  --run-dir PATH
  --output-prefix PATH
```

The wrapper loads, renders, and prints the two resulting paths. It must not
guess the latest run.

- [ ] **Step 6: Run figure tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/generals/test_policy_kl_figure.py -q
```

Expected: all loader and renderer tests pass.

- [ ] **Step 7: Commit the renderer**

```bash
git add pyproject.toml scripts/plot_generals_controlled_policy_kl.py \
  src/agentbench_frame/generals/paper_figure.py \
  tests/generals/test_policy_kl_figure.py
git commit -m "feat(generals): render policy KL paper figure"
```

### Task 3: Generate and visually verify the real paper artifact

**Files:**
- Create: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg`
- Create: `docs/experiments/figures/generals-controlled-policy-kl-three-panel.png`

**Interfaces:**
- Consumes: finalized run `20260730_1126_8d123b55`.
- Produces: committed paper-ready SVG and PNG.

- [ ] **Step 1: Generate both real outputs**

Run:

```bash
.venv/bin/python scripts/plot_generals_controlled_policy_kl.py \
  --run-dir agentbench_data/runs/28_generals/generals-policy-kl/20260730_1126_8d123b55 \
  --output-prefix docs/experiments/figures/generals-controlled-policy-kl-three-panel
```

- [ ] **Step 2: Perform artifact checks**

Verify:

```bash
test -s docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg
test -s docs/experiments/figures/generals-controlled-policy-kl-three-panel.png
rg -n "Controlled-reference Policy KL over Iterations|Epsilon Sensitivity|Exact Canonical Support Size" \
  docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg
```

Also compare every plotted KL and support label against the saved artifacts.

- [ ] **Step 3: Inspect the PNG visually**

Open the PNG with the available image viewer and check:

- no clipped titles, labels, legends, or annotations;
- the three panels have balanced widths and whitespace;
- zero KL points are visible;
- sensitivity lines are distinguishable;
- log-scale support bars and exact labels are readable; and
- the composition resembles the supplied academic reference.

If visual corrections are required, first add or update a render-contract
test for the affected layout behavior, then regenerate both formats.

- [ ] **Step 4: Run final verification**

Run:

```bash
git diff --check
AGENTBENCH_ASSET_ROOT=/home/cathy/AgentBench/AgentBench/.worktrees/generals-assets \
  PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -p no:cacheprovider -q
```

Expected: zero failures; environment-dependent skips remain explicit.

- [ ] **Step 5: Commit generated artifacts**

```bash
git add docs/experiments/figures/generals-controlled-policy-kl-three-panel.svg \
  docs/experiments/figures/generals-controlled-policy-kl-three-panel.png
git commit -m "docs(generals): add policy KL paper figure"
```

## Plan self-review

- Spec coverage: all three panels, artifact inputs, English labels, academic
  style, missingness, SVG, PNG, 300 DPI, and reproduction are assigned to
  concrete tasks.
- Type consistency: the loader produces `PolicyKLFigureData`; the renderer and
  CLI consume the same type and explicit paths.
- Scope: this plan generates the confirmed single Generals figure only; it
  does not introduce cross-game or cross-model performance profiles.
