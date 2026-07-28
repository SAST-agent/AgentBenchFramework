# Heuristic-Learning (HL) Iteration — Operator Guide

How to run the HL iteration loop: a real **Claude Code** agent edits a
versioned LostSpace agent, each version is evaluated and measured for
behavioral change, and the whole process is recorded for research.

This is the *manipulation manual* — read it to run, customize, and inspect a
run yourself. For the module API see the docstrings under `hl/`.

---

## 1. What the loop does

One **act** = one coding-agent edit cycle:

```
agent_act ─▶ ClaudeCodeRunner.run  (edits agent.py in place)
          ─▶ HLCodebase.snapshot   (content-hash version_after)
          ─▶ LostSpaceEvaluator    (frozen BenchmarkSpec → win_rate, replay)
          ─▶ ReferenceProbe        (BOTH v_{k-1} and v_k over the frozen ν)
          ─▶ local_policy_kl_trace + occupancy_shift
          ─▶ emit  version / eval / policy_kl / occupancy_shift / budget
```

The coding agent never reads `runs/` directly. Each act the framework builds it
a **prompt** containing: the role + game rules, the match-history table so far,
pointers to the latest replays, the inline playback recipe, and the previous
version handle. The agent edits `agent.py`; the framework does everything else.

Two things make the measurement rigorous:

- **Frozen ν.** Every version pair is probed over the *same* ReferenceStateSet,
  so policy-KL is a pure measurement with no occupancy-drift confound.
- **Missing stays missing.** An incomplete eval records `win_rate=None`,
  never `0`, never a loss (doc §12).

---

## 2. Prerequisites

```bash
cd E:/HL_Agent/AgentBenchFramework
uv sync                              # core env (zero hard runtime deps)
# optional, if you build RL comparisons:  uv sync --extra rl
```

Set two env vars. **Pick the block for your shell** — `export` is bash (Git
Bash / WSL / Linux / macOS), `$env:` is PowerShell:

```bash
# bash / Git Bash / WSL / Linux / macOS
export PYTHONPATH=src
export AGENTBENCH_DATA="./agentbench_data"
```

```powershell
# PowerShell (Windows)
$env:PYTHONPATH = "src"
$env:AGENTBENCH_DATA = ".\agentbench_data"
```

`PYTHONPATH=src` is **always required** (the package lives under `src/`).
`AGENTBENCH_DATA` is the run-output root.

> To make them permanent: bash → add to `~/.bashrc`; PowerShell →
> `[Environment]::SetEnvironmentVariable("PYTHONPATH","src","User")`
> (then restart the shell). Or use a `.env` file / direnv if you prefer.

The official LostSpace backend needs `antlr4-python3-runtime==4.9.*` for its
`python.exe` (3.10) child. The harness already scrubs `uv`'s poison
`PYTHONHOME`/`PYTHONPATH` from child envs — don't reintroduce them.

A `claude` CLI must be installed and authenticated (`claude --version` works).

---

## 3. Run procedure

### 3.1 Produce the frozen ReferenceStateSet ν (once per spec)

Assuming `PYTHONPATH=src` is already set in your shell (§2):

```bash
# bash
uv run python -m agentbench_frame.hl.reference_seed \
  --out ./agentbench_data/reference/nu-v1.json --spec-id hl-v1
```

```powershell
# PowerShell
uv run python -m agentbench_frame.hl.reference_seed `
  --out ./agentbench_data/reference/nu-v1.json --spec-id hl-v1
```

(If you skipped §2, set it inline first: bash
`PYTHONPATH=src uv run ...`; PowerShell `$env:PYTHONPATH="src"` then run.)

Writes 3 hand-authored decision points in the exact
`get_legal_actions()` shape. This ν is frozen — re-use the same file across
all versions you want comparable. Expand it later (see §6.3).

> **Honest note:** the seed ν is hand-authored. The *proper* ν comes from
> instrumenting the logic's `Player.get_legal_actions()` during a frozen
> reference roll — the wire frames alone don't carry `legal_actions` (it lives
> in the logic, not the frame). That recorder is a follow-up. The seed makes
> the loop runnable and KL measurable today; KL is coarse (3 points) but real.

### 3.2 Run the loop

**bash / Git Bash:**

```bash
BACKEND="E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
# A Python interpreter that has antlr4-python3-runtime==4.9.* installed
# (the logic imports antlr4 at startup; a bare 'python' without it dies
# and every eval match errors with "logic exited while reading 4 bytes").
LOGIC_PY="C:/Users/27364/.conda/envs/torchy/python.exe"

PYTHONPATH=src uv run python -m agentbench_frame.hl \
  --logic "cd /d \"$BACKEND\" && python main.py" \
  --logic-python "$LOGIC_PY" \
  --initial-candidate src/agentbench_frame/lostspace/candidates/v1 \
  --name hl-v1 \
  --reference ./agentbench_data/reference/nu-v1.json \
  --ladder-opponent rank=6 \
  --ladder-opponent rank=12 \
  --acts 5 --pairs 3 --seats 0 --timeout 15 \
  --dangerously-skip-permissions
```

**PowerShell:**

```powershell
$BACKEND = "E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
$LOGIC_PY = "C:/Users/27364/.conda/envs/torchy/python.exe"

uv run python -m agentbench_frame.hl `
  --logic "cd /d `"$BACKEND`" && python main.py" `
  --logic-python $LOGIC_PY `
  --initial-candidate src/agentbench_frame/lostspace/candidates/v1 `
  --name hl-v1 `
  --reference ./agentbench_data/reference/nu-v1.json `
  --ladder-opponent rank=6 `
  --ladder-opponent rank=12 `
  --acts 5 --pairs 3 --seats 0 --timeout 15 `
  --dangerously-skip-permissions
```

`--logic-python` rewrites the bare `python` token in `--logic` to the
given interpreter, so the logic subprocess runs under a Python that has
`antlr4-python3-runtime==4.9.*`. Before any act runs, the CLI probes that
this interpreter can `import antlr4` and fails fast with an actionable
message if not — instead of producing a silent all-error eval. If your
`--logic` already names an absolute interpreter with antlr4, you can omit
`--logic-python` (the probe still runs and validates it).

(PowerShell line continuation is the backtick `` ` `` at end of line; nested
double-quotes are escaped as `` `" ``. `PYTHONPATH`/`AGENTBENCH_DATA` are set
once per shell session per §2.)

**Critical:** the `--logic` cwd **must** be `gamecode_logic/` (it loads
`src/mapconf2.map` relatively and does `from src import main`).

`--dangerously-skip-permissions` makes `claude` fully autonomous (no prompts).
Use only in a trusted sandbox. Drop it (it then defaults to
`--permission-mode acceptEdits`) if you want to keep some control.

### 3.3 What each of the N acts does

1. `ContextBuilder` reads `runs/25_lostspace/hl-v1/*/` (history + last replay)
   and assembles the prompt.
2. `ClaudeCodeRunner` runs `claude -p <prompt> --append-system-prompt <role>
   --output-format json` with `cwd` at the workspace → Claude edits `agent.py`.
   Token usage is parsed from the `result` event.
3. `HLCodebase.snapshot()` content-hashes the tree → immutable `version_after`.
4. `LostSpaceEvaluator` runs the frozen BenchmarkSpec (pairs × seats ×
   opponents) through the real logic subprocess → writes a full run bundle
   (see §4).
5. `ReferenceProbe` drives **both** `v_{k-1}` and `v_k` over the same ν →
   `local_policy_kl_trace` + `occupancy_shift`.
6. All events append to `.hl_codebase/hl-v1/events.jsonl`.

After the last act a summary prints to stderr (acts, versions, final
`win_rate`, KL measurement count, events path).

---

## 4. Where the data lands

```
.hl_codebase/hl-v1/
├── events.jsonl          # ← the research stream (one JSON per line)
├── workspace/            # live, agent-editable codebase (current version)
├── store/<content_hash>/ # immutable snapshots, one per distinct content
└── stage/                # per-version eval/probe staging dirs

$AGENTBENCH_DATA/runs/25_lostspace/hl-v1/<run_id>/
├── run.toml              # metadata (carries agent_version = content_hash)
├── summary.json          # win_rate, h2h, lostspace aggregate
├── matches.jsonl         # one line per match (the headline)
├── events.jsonl          # match + episode events
└── artifacts/*.json      # frame-by-frame replays
```

---

## 5. Reading the results

### 5.1 The research stream — `events.jsonl`

Every record carries the public schema: `schema_version`, `event_id`,
`run_id`, `created_at`, `event_type`. Event types:

| event_type        | key fields |
|-------------------|------------|
| `agent_act`       | act_id, version_before, edit_policy_mode, budget_before |
| `version`         | version_id, content_hash, parent_version_id, edit_type, files_touched, **failure_reason** (null on success) |
| `eval`            | act_id, spec_id, version_after, evaluation_status, win_rate |
| `policy_kl`       | version_before/after, **local_policy_kl_trace** (raw per-decision list), epsilon |
| `occupancy_shift` | version_before/after, shift |
| `budget`          | scope=learning, act_id, coding_agent_acts, prompt/completion/total_tokens (None=unknown), runner_time_s, act_time_s, **failure_reason** |

`failure_reason` distinguishes a *clean* no-op (agent chose not to edit,
`edit_type="noop"`, `failure_reason=null`) from a *failed* run (timeout,
CLI not found, non-zero exit — `edit_type="noop"`,
`failure_reason="<stable reason string>"`). Stable reason strings:
`"claude CLI timed out"`, `"claude CLI not found: <path>"`,
`"claude exited <N>: <stderr>"`. A silent all-error eval that previously
masqueraded as a clean `noop` is now visible here.

```python
from agentbench_frame.hl.events import read_events
for e in read_events(".hl_codebase/hl-v1/events.jsonl"):
    if e["event_type"] == "eval":
        print(e["act_id"], e["win_rate"])
```

The headline research axis is **win_rate per act** (the learning curve) and
**policy_kl per act** (behavioral change). Token/time budget gives the
resource axes. (Plotting these into a figure is a follow-up — the data is all
here.)

### 5.2 The matches — read replays with the playback skill

Don't parse 20 KB of replay JSON by hand. Use
`AgentBenchResults/skills/lostspace-playback/SKILL.md`:

```bash
# bash
cd AgentBenchFramework && PYTHONPATH=src
python -m agentbench_frame.lostspace.replay_view \
  "$AGENTBENCH_DATA/runs/25_lostspace/hl-v1/<run_id>/artifacts/<opp>-pair000-seat0.json"
```

```powershell
# PowerShell
cd AgentBenchFramework; $env:PYTHONPATH="src"
python -m agentbench_frame.lostspace.replay_view `
  "$env:AGENTBENCH_DATA/runs/25_lostspace/hl-v1/<run_id>/artifacts/<opp>-pair000-seat0.json"
```

Renders a human-readable turn log so you don't parse JSON by hand. Quick
"did I win + standings" one-liner:

```bash
python -c "import json;print(json.load(open('REPLAY'))[-1])"
```

The skill documents the full replay schema, action types, the `+3` coord shift,
and the `ai_error`-on-seats-1-3-is-background-noise note.

---

## 6. Customizing a run

### 6.1 Opponents

Ladder opponents (the 16 ranked human algorithms), repeatable:

```bash
--ladder-opponent rank=6            # by rank number
--ladder-opponent rank=omegafantasy # by username
--ladder-opponent rank=最终幻想       # by display name
```

C++ entries are built into `.cache/ladder` on first use (overridable via
`LOSTSPACE_LADDER_CACHE`); Python entries launch directly. Explicit opponents:

```bash
--opponent NAME=COMMAND
```

The bundled sample AI pads empty seats (override with `--filler`).

### 6.2 Acts, budget, measurement

| flag | default | meaning |
|------|---------|---------|
| `--acts` | 5 | number of coding-agent acts |
| `--pairs` | 3 | match pairs per opponent |
| `--seats` | `"0"` | candidate seat (`all`/`0`..`3`) |
| `--timeout` | 15 | per-action judger TLE (seconds) |
| `--epsilon` | 0.1 | smoothing for the policy-KL channel |
| `--claude-timeout` | 600 | per-act `claude` CLI timeout (seconds) |

### 6.3 Your own ν

Hand-author a JSON file in the seed format (the easiest path). Save this as
`make_nu.py` (multi-line `python -c` is painful in PowerShell, so use a file):

```python
# make_nu.py
from agentbench_frame.hl.reference import ReferenceSample, ReferenceStateSet

nu = ReferenceStateSet(spec_id="hl-v1", samples=(
    ReferenceSample(
        observation={...},
        legal_actions={"attack": [], "move": [True] * 8,
                       "detect": True, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06"),
    # ... more decision points
))
nu.save("nu-v1.json")
```

Then run it (both shells, with `PYTHONPATH=src` set per §2):

```bash
# bash
PYTHONPATH=src uv run python make_nu.py
```
```powershell
# PowerShell
$env:PYTHONPATH="src"; uv run python make_nu.py
```

`legal_actions` must match `Player.get_legal_actions()` exactly:
`{'attack':[ids], 'move':[8 bools], 'detect':bool, 'interprops':[names]}`.
Only `status == 0` (Alive) states are decision points.

### 6.4 Codebase shape

`--initial-candidate` is a directory copied into the workspace. It **must**
contain `agent.py`. The bundled starting points live at
`src/agentbench_frame/lostspace/candidates/v{1..4}/` — v1 is the official
sample AI copy; v2–v4 are iterating candidates. Pick one, e.g.
`--initial-candidate src/agentbench_frame/lostspace/candidates/v1`.

A `manifest.toml` is optional — with none, the harness defaults the
entrypoint to `agent.py` (the bundled candidates ship without one and work).
If you add one (single-file shape):

```toml
shape = "single_file"
entrypoint = "agent.py"
```

For a multi-file package, set `shape = "package"` and point `entrypoint` at the
script the harness runs (see `hl/manifest.py`).

---

## 7. Versioning: rollback and diff

The codebase is a content-hash snapshot store — **no git inside the agent
tree**. From Python you can restore a prior version or diff two:

```python
from agentbench_frame.hl.codebase import HLCodebase
cb = HLCodebase(root=".hl_codebase/hl-v1/workspace",
                store=".hl_codebase/hl-v1/store")

# restore a prior content_hash into the workspace (new version_id, edit_type='rollback')
v = cb.restore(content_hash="<hash>", parent_version_id="<current>")

# structured file-level diff between two snapshots
d = cb.diff("<before_hash>", "<after_hash>")
print(d.added, d.removed, d.modified)
```

`restore()` records `edit_type='rollback'` — zero policy-KL but nonzero act
budget, never silently folded. The coding agent sees prior versions read-only
via `VersionDiffView`.

---

## 8. CLI flags — full reference

```
python -m agentbench_frame.hl
  --logic LOGIC                      required; cwd must be gamecode_logic/
  --initial-candidate DIR            required; must contain agent.py
  --name NAME                        required; HL agent name
  --reference PATH                   required; frozen ReferenceStateSet JSON (ν)

  --ladder-opponent rank=NAME        repeatable (rank / username / display name)
  --opponent NAME=COMMAND            repeatable; explicit opponent
  --filler COMMAND                   default: bundled sample AI

  --acts N            (5)     coding-agent acts
  --epsilon EPS       (0.1)   policy-KL smoothing
  --pairs N           (3)     match pairs per opponent
  --seats S           ("0")   candidate seat: all|0|1|2|3
  --timeout SEC       (15)    per-action judger TLE
  --spec-id ID        (hl-<name>)

  --data-dir DIR              default: $AGENTBENCH_DATA
  --codebase-root DIR         default: ./.hl_codebase/<name>
  --claude-path PATH          default: claude
  --model ID                  claude model override
  --permission-mode MODE      default|acceptEdits|bypassPermissions|plan
  --dangerously-skip-permissions  fully autonomous (overrides permission-mode)
  --claude-timeout SEC        (600)  per-act claude timeout
```

Helpers:

```
python -m agentbench_frame.hl.reference_seed --out nu.json --spec-id hl-v1
```

---

## 9. Gotchas (read before debugging)

- **`--logic` cwd must be `gamecode_logic/`.** It loads `src/mapconf2.map`
  relatively. Needs `antlr4-python3-runtime==4.9.*`. Use `--logic-python`
  to point at an interpreter that has it (e.g. a conda env); the CLI
  probes `import antlr4` at startup and fails fast if missing, instead of
  silently producing all-error evals.
- **An all-error eval (`win_rate=null`) is a harness problem, not a
  strategy problem.** The per-act prompt tells the coding agent not to
  debug the harness in that case — but if you see it persisting, check
  the `failure_reason` field on the `version`/`budget` events: it
  distinguishes a clean no-op from a timed-out / CLI-not-found / non-zero
  exit run (stable reason strings documented in §5.1).
- **Never use the official `saiblo-local-judger` (PyPI 0.0.2)** as the runner —
  it stalls (its TLE timer is commented out). The harness's own
  `_ACTION_REQUEST_TYPES` handling is more correct.
- **`uv run` poisons child Pythons** (sets `PYTHONHOME`/`PYTHONPATH`). The
  harness scrubs these via `_child_env`; don't reintroduce them.
- **Don't "fix" ranked algorithms that stall/crash.** Several *correctly*
  stall on saiblo and rely on the judger's TLE to continue. A high
  `ai_error` count on seats 1-3 is background noise, not your bug.
- **Windows `CreateProcess`** resolves a relative app name against the *caller's*
  cwd, not `cwd=`. Bare binaries are made absolute by the harness.
- **LostSpace work lives on branch `liuzhuo/lostspace` only — never push to
  `main`.**
- **Always run Python through `uv run`** and set `PYTHONPATH=src`.

---

## 10. Follow-ups (not yet built)

1. **Figure/dashboard rendering** from `events.jsonl` — the data for learning
   curves (win_rate vs act), policy-KL traces, and token/time AUC is all
   captured, but no plotting module exists yet.
2. **Proper ν recorder** — instrument the logic's `Player.get_legal_actions()`
   during a frozen reference roll so ν is populated from real play, not by
   hand.
3. **Evaluation-scope budget** — the controller currently records only the
   `learning` scope per act; an explicit evaluation-scope budget split is a
   straightforward extension.

---

## 11. Smoke check (no real claude / no real logic)

Before pointing the loop at the real CLI, sanity-check the wiring end-to-end
with stubs (the test suite does this):

```bash
# bash
cd AgentBenchFramework && PYTHONPATH=src
uv run pytest tests/hl/test_cli.py::test_main_runs_acts_with_stubbed_runner_and_eval -q
```

```powershell
# PowerShell
cd AgentBenchFramework; $env:PYTHONPATH="src"
uv run pytest tests/hl/test_cli.py::test_main_runs_acts_with_stubbed_runner_and_eval -q
```

That test builds the controller exactly as `main()` does, swaps a scripted
runner + stub evaluator, runs 2 acts, and asserts the full event chain
(`agent_act` → `version` → `eval` → `policy_kl` → `occupancy_shift` →
`budget`) lands in `events.jsonl`. If it's green, the wiring is sound and the
only step to a real run is the real `--logic` + real `claude`.
