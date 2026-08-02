# Plan: Meet the HL checklist (focus = item 5)

## Problem (verified)
- All real runs have `win_rate=0.0`, `edit_type="refactor"`, `policy_kl_trace=[0,0,0]`,
  `occupancy_shift=0.0`. **No valid policy update has ever occurred** → checklist
  item 5 unmet.
- `ClaudeCodeRunner` hardcodes `edit_type="refactor"` (`runner.py:203`) with a stale
  comment claiming the controller reclassifies via diff — it never does. Every real
  edit is mislabeled.
- **No plotting module exists** (README §22 admits it). The lone
  `figures/hl-v1_iteration_curves.png` has no source → irreproducible, and
  contradicts the all-zero IG reality.
- ν has only 3 hand-authored decision points → a real strategy edit is unlikely to
  touch any, so KL stays 0 even when the agent does change behavior elsewhere.
- Environment is ready: `claude` CLI ✓, antlr4 in conda env ✓, ν exists ✓.

## Approach
Two independent deliverables + one real run.

### A. Plotting module — `hl/plot_curves.py` (deterministic, satisfies "curves" half of item 5)
- `read_iteration_curves(events_path) -> CurveData`: walk `events.jsonl`, join by
  `act_id`: x = act index (1-based, in emission order); y_score = `eval.win_rate`
  (None = incomplete, kept as gap); y_ig = mean of `policy_kl.local_policy_kl_trace`
  (None when no policy_kl event, e.g. first act / failed eval); plus per-point
  `evaluation_status`, `failure_reason`, `edit_type` for annotation.
- `plot(events_path, out_dir)`: lazy-import matplotlib (it's a dev dep; raise an
  actionable `pip install matplotlib` error if absent). Emit two PNGs:
  `score_iteration.png` (win_rate vs iteration) and `ig_iteration.png`
  (policy_kl vs iteration). **Honest rendering**: incomplete evals drawn as gaps
  (no point), failures marked with a red 'x' at the iteration, a caption noting
  N incomplete / N failed. Version-aligned: each x tick labels `act_id`.
- CLI: `python -m agentbench_frame.hl.plot_curves --events PATH --out-dir DIR`
  (and `--name` convenience that resolves `.hl_codebase/<name>/events.jsonl`).
- Tests (`tests/hl/test_plot_curves.py`): synthetic events.jsonl with a complete
  eval, an incomplete eval (win_rate=None → gap), and a KL>0 entry → assert PNGs
  created, None preserved (not coerced to 0), and the IG curve reflects the KL>0.

### B. Honest edit_type classification (fixes the hardcoded-"refactor" bug)
- `ClaudeCodeRunner` success path returns `edit_type=None` ("unclassified").
  (Failure paths already correctly return `"noop"`; tests only assert failure
  paths, so safe.)
- Controller `act()`: after `snapshot`, if `run_result.edit_type is None` and a
  parent exists, classify by diffing `codebase.diff(parent.content_hash,
  child.content_hash)`:
  - identical content_hash → `"noop"` (agent made no edit — currently mislabeled
    "refactor"); 
  - else → `"edit"` is not in the taxonomy, so map: files added/removed →
    `"replace"`; only `agent.py` modified → `"parametrize"` (smallest honest
    non-noop label); else `"refactor"`.
  - When `run_result.edit_type` is a concrete value (FakeRunner / declared),
    trust it unchanged (preserves all existing controller tests).
- `noop` detection is the real fix here: it stops masking "agent edited nothing"
  as "refactor". The semantic label for changed content is best-effort and
  documented as such in the docstring.

### C. Make a valid policy update achievable + run it
- **Expand ν** (`reference_seed.py`): add ~5 more hand-authored decision points
  covering diverse situations (carry a Kit, on a KeyMachine, WAIT_FOR_ESCAPE,
  multiple attack targets, mid-board movement) so a real strategy edit is likely
  to register KL>0. Regenerate `agentbench_data/reference/nu-v1.json` (the seed
  is frozen-by-spec, so version it: write `nu-v2.json` and point runs at it).
- **Strengthen the prompt** (`context.py` "What to do now"): require a concrete
  *behavioral* change — "your edit MUST change the action the agent takes in at
  least one reachable game situation; a pure rename/restructure with no behavior
  change is a failed act." Tie it to the existing "NO MEASURABLE EFFECT" feedback.
- **Run a real HL round** (2–3 acts) with `--dangerously-skip-permissions`,
  `--reference nu-v2.json`, against `rank=6` + `rank=12`, `--acts 3 --pairs 3`.
  Watch events.jsonl for `policy_kl > 0` (or `win_rate` change). If the first run
  stays KL=0, inspect the probe emissions vs the diff to see whether the edit
  missed the reference states, adjust ν/prompt, and run once more. Cap at ~2 runs.
- **Generate the curves** from the new round's `events.jsonl` via the module
  from (A). Incomplete/failed acts preserved as-is (gaps + x marks).

## Files
- new: `src/agentbench_frame/hl/plot_curves.py`
- new: `tests/hl/test_plot_curves.py`
- edit: `src/agentbench_frame/hl/runner.py` (success-path edit_type → None)
- edit: `src/agentbench_frame/hl/controller.py` (diff-based reclassification)
- edit: `src/agentbench_frame/hl/reference_seed.py` (expand ν) + `nu-v2.json`
- edit: `src/agentbench_frame/hl/context.py` (demand behavioral change)
- generated: `.hl_codebase/<round>/events.jsonl`, `figures/score_iteration.png`,
  `figures/ig_iteration.png`

## Verification
- `uv run pytest tests/hl -q` green (existing + new plot test).
- `python -m agentbench_frame.hl.plot_curves --name <round>` produces both PNGs;
  incomplete/failed acts shown honestly.
- The new round's `events.jsonl` contains ≥1 `policy_kl` event with
  `local_policy_kl_trace` having a positive entry (a valid policy update), and a
  non-noop `edit_type`.

## Out of scope / not done here
- Item 2 (Agent-facing replay Skill polish + validation) — separate; materials
  already exist. Will note status.
- The "proper ν recorder" (instrument logic's `get_legal_actions`) — README §22
  follow-up #2; still hand-authored (expanded) here.
