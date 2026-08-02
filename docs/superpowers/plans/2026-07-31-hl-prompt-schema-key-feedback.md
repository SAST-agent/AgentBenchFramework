# HL Prompt Schema + Loud Key-Feedback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the HL loop's coding agent from silently disabling LostSpace's win condition, by putting the game's data schema (interprops are int/object-coded, not strings) into the prompt and making a "0 keys collected" result a loud, explicit instruction instead of a buried count.

**Architecture:** Two prompt layers change, no control-plane or eval change. (1) The durable **system prompt** (`hl/cli.py::_system_prompt`) gains a data-safety rule: never gate an emitted action on a type-unverified membership/equality check; prefer blind-call-then-check-`success`. (2) The per-act **user prompt** (`hl/context.py::ContextBuilder`) gains a concrete "Data schema" section, and the existing seat-0 replay digest grows a loud `ACTION:` callout when it sees 0 keys + no escape. Both are pure prompt-string changes backed by unit tests; no runner/evaluator/controller edits.

**Tech Stack:** Python ≥3.11, `uv` + `--extra hl`, pytest, no new deps.

## Background (why — read this first)

Run `hl-curriculum-0731` (deepseek-v4-pro, curriculum tier `rank16`) regressed: `avg_rank` 3.00 (seed) → 4.00 (dead last, acts 2–6) → 3.33, `win_rate` 0 throughout. Root cause **proven** from the logic source (`backend_sources/.../gamecode_logic/src/interactive_props.py:4-5`):

```
1  ->  EscapeCapsule
2  ->  KeyMachine
```

LostSpace sends each AI a view whose `node[1]` is the tile's `interprops` — a list of **integer codes / `InterProps` objects** (`Materials/Box/EscapeCapsule/KeyMachine` are classes in `interactive_props.py`). The agent client also appends the string `"Box"` itself (`agent.py:418`). deepseek rewrote `play()` to gate the win-condition actions on string membership:

```python
if "KeyMachine" in interprops:        # ALWAYS False — list of int/obj, not str
    self.interact("KeyMachine")
if "EscapeCapsule" in interprops:     # ALWAYS False
    self.interact("EscapeCapsule", 1)
```

→ `interact("KeyMachine")` is never called → **0 keys collected, ever** → the agent cannot satisfy the win condition (4 keys + escape) → guaranteed loss. `errors=0` confirms it is a *behavioral* regression, not a crash. The seed avoided this by **blind**-calling `interact("KeyMachine")` every tick and branching on the returned `["success"]`.

The seat-0 digest (`context.py::_seat0_digest_line`) **already** computes `keys=0 escaped=False` from the replay — the data was present but buried in the "Replays" section, with no instruction connecting it to the broken gate. The data **schema** (interprops int-coded) appears nowhere in the prompt. This plan fixes both gaps.

## Global Constraints

- Branch **`liuzhuo/lostspace`** only — never push to `main`. Framework repo is `E:\HL_Agent\AgentBenchFramework`.
- Run all Python through **`uv run --extra hl`**; set `PYTHONPATH=src` for any direct invocation. Never bare `python`/`pip`.
- **TDD**: write the failing test first, run it red, implement, run green, commit — one commit per task.
- Edit **only** `src/agentbench_frame/hl/cli.py`, `src/agentbench_frame/hl/context.py`, and `tests/hl/test_cli.py`, `tests/hl/test_context.py`. No runner/evaluator/controller changes. No new deps.
- **Secrets**: `.env` (with the DeepSeek key) stays gitignored; never log/print key values.
- LostSpace gotchas stand: do not "fix" ranked algorithms that stall/crash on saiblo (the judger's TLE advances the game by design). This plan touches none of that.
- The reference set, logic command, and ladder opponents are unchanged from the `hl-curriculum-0731` run.

## File Structure

- **Modify** `src/agentbench_frame/hl/cli.py` — `_system_prompt()` (≈ line 224): add the data-safety paragraph.
- **Modify** `src/agentbench_frame/hl/context.py` — add `_DATA_SCHEMA_BLURB` constant (near `_GAME_RULES_BLURB`, ≈ line 42); emit it in `_prompt()` after the game-rules line (≈ line 140); extend `_seat0_digest_line()` (≈ line 415) with the loud `ACTION:` callout.
- **Modify** `tests/hl/test_cli.py` — assert the system prompt carries the schema rule.
- **Modify** `tests/hl/test_context.py` — assert the data-schema section renders; assert the 0-keys callout fires on a synthetic keyless replay.

---

### Task 1: Data-safety rule in the system prompt

**Files:**
- Modify: `src/agentbench_frame/hl/cli.py` (`_system_prompt`, ≈ line 224)
- Test: `tests/hl/test_cli.py`

**Interfaces:** none new. `_system_prompt()` is a 0-arg function returning `str`; it is passed as `system_prompt=` to `ApiCodingRunner` in `cli.py::main` (≈ line 410).

- [ ] **Step 1: Write the failing test**

Append to `tests/hl/test_cli.py`:

```python
from agentbench_frame.hl.cli import _system_prompt

def test_system_prompt_states_interprops_schema_and_safe_pattern():
    """The durable system prompt must warn that interprops are int/object-coded
    (so 'X' in interprops is always False) and point to the blind-call-then-
    check-success pattern. This is the regression that zeroed key collection
    in hl-curriculum-0731."""
    s = _system_prompt()
    assert "interprops" in s.lower()
    assert "always false" in s.lower()          # 'X' in interprops is always False
    assert "success" in s.lower()               # branch on result['success']
    assert "KeyMachine" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_cli.py::test_system_prompt_states_interprops_schema_and_safe_pattern -v`
Expected: FAIL — `"interprops"` / `"always false"` not in current prompt.

- [ ] **Step 3: Implement — extend `_system_prompt()`**

In `src/agentbench_frame/hl/cli.py`, replace the `return (...)` of `_system_prompt()` so the final sentence (`"...Leave the agent runnable."`) is followed by a new data-safety paragraph. Keep every existing sentence verbatim; only append:

```python
        "Never break the Saiblo stdio protocol (4-byte big-endian length "
        "prefix + UTF-8 JSON). Leave the agent runnable. "
        "DATA-SAFETY RULE: never gate an action you emit (interact/attack/"
        "move/use_tool) on a membership or equality check against data whose "
        "RUNTIME type you have not confirmed from the code that produces it. "
        "In LostSpace `self.view.nodes[i].interprops` is a list of INTEGER "
        "CODES / objects (1=EscapeCapsule, 2=KeyMachine; the client also "
        "appends the string 'Box'), NOT a list of strings. So a guard like "
        "`if 'KeyMachine' in interprops` is ALWAYS False and silently disables "
        "key collection. Prefer the codebase's existing blind-call-then-check "
        "pattern: call `self.interact('KeyMachine')` and branch on "
        "`result['success']` (the server returns success only when the action "
        "is actually valid)."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_cli.py::test_system_prompt_states_interprops_schema_and_safe_pattern -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/cli.py tests/hl/test_cli.py
git commit -m "feat(hl): data-safety rule in system prompt (interprops int-coded)"
```

---

### Task 2: "Data schema" section in the per-act context prompt

**Files:**
- Modify: `src/agentbench_frame/hl/context.py` (new `_DATA_SCHEMA_BLURB` ≈ line 48; emit in `_prompt()` ≈ line 140)
- Test: `tests/hl/test_context.py`

**Interfaces:** none new. Internal constant + one `lines.append` in `_prompt()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/hl/test_context.py`:

```python
def test_prompt_has_data_schema_section(tmp_path):
    """The per-act prompt must carry a Data schema section stating interprops
    are int/object-coded (1=EscapeCapsule, 2=KeyMachine), that string
    membership is always False, and the blind-interact-then-check-success
    pattern. First-act prompt (no history) must still include it."""
    cb = HLCodebase(root=tmp_path / "ws", store=tmp_path / "store")
    (tmp_path / "ws").mkdir()
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "Data schema" in prompt
    assert "1=EscapeCapsule" in prompt
    assert "2=KeyMachine" in prompt
    assert "always False" in prompt or "always false" in prompt
    assert "interact('KeyMachine')" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_context.py::test_prompt_has_data_schema_section -v`
Expected: FAIL — no "Data schema" section.

- [ ] **Step 3: Implement — add the blurb + emit it**

In `src/agentbench_frame/hl/context.py`, immediately after the `_GAME_RULES_BLURB` definition (after line 47), add:

```python
_DATA_SCHEMA_BLURB = (
    "`self.view.nodes[i].interprops` is a list of INTEGER CODES / objects "
    "(1=EscapeCapsule, 2=KeyMachine); the agent client also appends the "
    "string 'Box'. It is NOT a list of strings like 'KeyMachine', so "
    "`if 'KeyMachine' in interprops` (or 'EscapeCapsule') is ALWAYS False — "
    "never gate an action on it. Safe pattern (already used by the seed): "
    "call the action blind, then branch on the returned `['success']`, e.g. "
    "`if self.interact('KeyMachine')['success']: return` — the server returns "
    "success only when the action is valid. Win condition: interact with each "
    "of the 4 corner KeyMachines (collect 4 keys), then interact with the "
    "center EscapeCapsule. An agent that never calls interact('KeyMachine') "
    "can never win."
)
```

Then in `_prompt()`, immediately after the game-rules append (after the line `lines.append(f"## Game rules\n{_GAME_RULES_BLURB}\n")`, ≈ line 140), add:

```python
        lines.append(f"## Data schema\n{_DATA_SCHEMA_BLURB}\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_context.py::test_prompt_has_data_schema_section -v`
Expected: PASS. Also re-run the whole context suite to confirm no regression: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_context.py -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/context.py tests/hl/test_context.py
git commit -m "feat(hl): add Data schema section to per-act prompt"
```

---

### Task 3: Loud "0 keys" callout in the seat-0 digest

**Files:**
- Modify: `src/agentbench_frame/hl/context.py` (`_seat0_digest_line`, ≈ line 415)
- Test: `tests/hl/test_context.py`

**Interfaces:** none new. `_seat0_digest_line(run_dir, latest) -> str` stays a staticmethod returning one line; it just appends an `ACTION:` segment when the digest shows 0 keys + no escape in a game long enough to have collected keys.

- [ ] **Step 1: Write the failing test**

Append to `tests/hl/test_context.py`. The replay has seat-0 `move` actions only (no `getkey`/`keymachine`), many rounds, so the digest is `keys=0 escaped=False`:

```python
def test_prompt_seat0_digest_loud_callout_on_zero_keys(tmp_path):
    """When the last game ran long but seat 0 collected 0 keys and did not
    escape, the digest must emit a loud ACTION callout naming the interprops
    gating regression — not just a buried keys=0 count. This is the exact
    failure mode of hl-curriculum-0731."""
    cb = _codebase(tmp_path)
    run_dir = tmp_path / "runs" / "25_lostspace" / "hl-v1" / "run-001"
    art = run_dir / "artifacts"
    art.mkdir(parents=True)
    # 25 rounds; seat 0 (playerid 0) only moves — never getkey/keymachine.
    seat0_turn = [{"type": "move", "playerid": 0, "pos": [0, 0, 1]}]
    rounds = [[seat0_turn,
               [{"type": "move", "playerid": 1, "pos": [6, 0, 1]}],
               [{"type": "move", "playerid": 2, "pos": [6, 6, 1]}],
               [{"type": "move", "playerid": 3, "pos": [0, 6, 1]}]]
              for _ in range(25)]
    replay = [[[0, 0, 1], [6, 0, 1], [6, 6, 1], [0, 6, 1]], *rounds,
              {"0": 1, "1": 4, "2": 3, "3": 2}]
    (art / "rank06-pair000-seat0.json").write_text(json.dumps(replay), encoding="utf-8")
    (run_dir / "matches.jsonl").write_text(
        json.dumps({"opponent": "rank06", "candidate_result": "loss",
                    "candidate_rank": 4, "candidate_score": 1, "turns": 2500,
                    "pair": 0, "candidate_seat": 0,
                    "replay": "artifacts/rank06-pair000-seat0.json"}) + "\n",
        encoding="utf-8")
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="hl-v1", spec=_spec())
    prompt = builder.build(version_before=None, act_id="r-act0001")["prompt"]
    assert "keys=0" in prompt
    assert "ACTION" in prompt
    assert "interprops" in prompt.lower()
    # sanity: a game WITH a getkey must NOT trigger the callout
    seat0_key = [{"type": "getkey", "playerid": 0}]
    replay2 = json.loads(json.dumps(replay))
    replay2[1][0] = [seat0_key]
    (art / "rank06-pair000-seat0.json").write_text(json.dumps(replay2), encoding="utf-8")
    prompt2 = builder.build(version_before=None, act_id="r-act0002")["prompt"]
    assert "keys=" in prompt2 and "keys=0" not in prompt2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_context.py::test_prompt_seat0_digest_loud_callout_on_zero_keys -v`
Expected: FAIL — current digest emits `keys=0` but no `ACTION` callout.

- [ ] **Step 3: Implement — extend `_seat0_digest_line()`**

In `src/agentbench_frame/hl/context.py`, inside `_seat0_digest_line`, after the `if d["last_action"]: bits.append(...)` block and before the final `return`, add:

```python
        # Loud callout: a game that ran long enough to collect keys but where
        # seat 0 collected NONE and did not escape is almost certainly a broken
        # win-condition path (the interprops-gating regression). Make it an
        # explicit instruction, not a buried count.
        if (d.get("keys", 0) == 0 and not d.get("escaped")
                and d.get("n_rounds", 0) >= 20):
            bits.append(
                "ACTION: 0 keys + no escape = key collection is broken. Do "
                "NOT gate interact('KeyMachine') on interprops membership "
                "(int/object-coded, never the string 'KeyMachine'). Call "
                "interact('KeyMachine') and branch on result['success']. See "
                "the Data schema section.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl/test_context.py::test_prompt_seat0_digest_loud_callout_on_zero_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agentbench_frame/hl/context.py tests/hl/test_context.py
git commit -m "feat(hl): loud 0-keys ACTION callout in seat-0 digest"
```

---

### Task 4: Full-suite green + manual prompt smoke

**Files:** none modified (verification only).

- [ ] **Step 1: Run the whole HL suite**

Run: `PYTHONPATH=src uv run --extra hl pytest tests/hl -q`
Expected: all green (was 213 passing before; now 213 + 3 new = 216).

- [ ] **Step 2: Render one real prompt and eyeball it**

Run (reuses the existing `hl-curriculum-0731` data on disk; builds the prompt the agent would see on act 2 against `rank16`):

```bash
PYTHONPATH=src uv run --extra hl python -c "
from pathlib import Path
from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.context import ContextBuilder
from agentbench_frame.hl.reference import BenchmarkSpec
data=Path('agentbench_data'); root=Path('.hl_codebase/hl-curriculum-0731')
cb=HLCodebase(root=root/'workspace', store=root/'store')
spec=BenchmarkSpec(spec_id='x', opponents=('rank16',), pairs=6, seats='0', timeout=15.0)
b=ContextBuilder(codebase=cb, data_root=data, game='25_lostspace', agent_name='hl-curriculum-0731', spec=spec)
p=b.build(version_before=None, act_id='smoke')['prompt']
print('--- DATA SCHEMA present:', 'Data schema' in p)
print('--- ACTION callout present:', 'ACTION: 0 keys' in p)
print('--- length:', len(p), 'chars')
"
```
Expected: both `True`; length within a reasonable prompt size (a few KB). If `ACTION` is `False`, the latest run's last replay had keys>0 — acceptable (the callout is conditional); confirm the digest line itself renders.

- [ ] **Step 3: Commit any incidental fixes (likely none)**

Only commit if Step 2 surfaced a real defect. Otherwise nothing to commit.

---

### Task 5: Verify end-to-end — re-run the curriculum and confirm key collection is restored

**Files:** none modified (experiment). This is the empirical proof the fix works.

- [ ] **Step 1: Re-launch the curriculum run (background)**

From `E:\HL_Agent\AgentBenchFramework`, with the same env as `hl-curriculum-0731` but a **new name** (reusing a name truncates the prior events stream — see the hl-events-jsonl-truncates memory):

```bash
PYTHONPATH=src AGENTBENCH_DATA=./agentbench_data \
uv run --extra hl python -m agentbench_frame.hl \
  --logic 'cd /d "E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic" && python main.py' \
  --logic-python C:/Users/27364/.conda/envs/torchy/python.exe \
  --reference ./agentbench_data/reference/nu-v2.json \
  --ladder-opponent rank=16 --ladder-opponent rank=12 --ladder-opponent rank=8 --ladder-opponent rank=4 --ladder-opponent rank=1 \
  --curriculum --promote-rank 2.0 \
  --pairs 6 --seats 0 --timeout 15 --acts 12 \
  --name hl-schema-0731 \
  --max-turns 6
```

(12 acts, not 20 — enough to see whether key collection resumes and `avg_rank` climbs off 4.00. Each act ≈ 3–4 min → ~45 min.)

- [ ] **Step 2: After ~3 acts, confirm key collection resumed**

Poll the latest eval's seat-0 digest / replay: the agent should now emit `getkey`/`keymachine` actions. Check one replay from `.hl_codebase/hl-schema-0731`:

```bash
python -c "
import json,glob
d=sorted(glob.glob('agentbench_data/runs/25_lostspace/hl-schema-0731/*/artifacts/*.json'))[-1]
r=json.load(open(d,encoding='utf-8'))
seats=[a for rnd in r[1:-1] for turn in rnd for a in turn if a.get('playerid')==0]
keys=sum(1 for a in seats if a.get('type') in ('getkey','keymachine'))
print('replay:',d); print('seat-0 key actions:',keys,'/',len(seats),'turns')
"
```
Expected: `seat-0 key actions` > 0 on at least some games (vs. 0 before). If still 0 after 3 acts, inspect the act transcript under `.hl_codebase/hl-schema-0731/transcripts/` to see whether the model still emitted a gating guard despite the prompt — that is a model-compliance issue, not a plan defect; record it.

- [ ] **Step 3: Compare avg_rank trajectory vs the regressed run**

After the run (or ≥8 acts), pull per-act `avg_rank` from each run dir's `summary.json` (`lostspace.aggregate.avg_rank`) as in the analysis. Success criterion: `avg_rank` at tier 0 (`rank16`) **climbs below 3.00** (the seed's symmetric null) on at least one act — i.e. the edited agent now beats at least one filler, which is only possible once it collects keys. Promotion to `rank12` (avg_rank ≤ 2.0) is the stretch goal.

- [ ] **Step 4: Commit the result artifacts pointer (optional)**

The run data lives under gitignored `.hl_codebase/` and `agentbench_data/` — do NOT commit those. If a findings summary is wanted, write it to `docs/superpowers/plans/2026-07-31-hl-prompt-schema-key-feedback.results.md` and commit only that.

---

## Self-Review (run after drafting, before handoff)

- **Spec coverage:** the two gaps from the analysis — (a) schema absent from prompt, (b) key signal buried — map to Tasks 1–2 and Task 3 respectively. Task 4 guards regressions; Task 5 proves it. ✔
- **Placeholder scan:** every step has real code or a real command; no TBD/TODO. ✔
- **Type consistency:** `_DATA_SCHEMA_BLURB` is referenced by the exact name in both Task 2's test and its emit line; `_seat0_digest_line` signature unchanged. ✔
- **Scope:** pure prompt changes — no runner/evaluator/controller/spec-format edits, so no risk to the CI data contract (`run.toml`/`summary.json`) or the just-archived `hl-raw-api-runner` specs. ✔

## Execution Handoff

Plan complete and saved to `AgentBenchFramework/docs/superpowers/plans/2026-07-31-hl-prompt-schema-key-feedback.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh implementer subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session, batch with checkpoints.

Which approach? (Per the project's OpenSpec workflow, this plan can also be promoted into a formal change with `/opsx:propose hl-prompt-schema-key-feedback` before implementation — optional.)
