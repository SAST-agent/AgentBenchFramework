# Rollman Candidate Context and Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make at least three of four Rollman candidates reach a changed, compilable, Framework-reverified public-entry smoke-tested policy within the existing 70,000 weighted-token act budget, then resume the same experiment run without weakening evaluation or certification.

**Architecture:** The planner receives a compact module-function index and selects exact symbols for each branch. Candidate packets contain deterministic bounded source slices for those symbols plus a reusable valid Rollman state fixture and one exact smoke command. The Framework independently reruns the smoke verifier before activation or match evaluation, while the provider enforces six pre-edit and twelve total tool calls for this packet contract.

**Tech Stack:** Python 3.13, dataclasses, `ast`, JSON Schema, NumPy-compatible Rollman state objects, subprocess verification, pytest, AgentBench Framework CLI, Codex Responses provider.

## Global Constraints

- Continue the existing run and immutable version lineage; do not create a replacement run.
- Preserve K=4, linear sibling selection, `xhigh`, 70,000 weighted tokens, frozen opponents and seeds, rollback behavior, and rank15/rank16 certification at at least 4/5 each.
- Do not expose secrets, human source, hidden evaluation data, certification seeds, or non-whitelisted filesystem paths.
- Keep model-call count at one planner act plus four coding acts per proposal cycle; no second candidate/reviewer model act.
- Candidate partial adoption remains fail-closed: changed source, clean access audit, successful compile, Framework-reverified smoke, activation probe, and quick screen are all required.
- Use `apply_patch` for edits. Run focused tests after every task and the full suite before any paid experiment call.

---

## Task 1: Strict planner symbol contract

**Files:**

- Modify: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `src/agentbench_frame/hl/cli.py`
- Test: `tests/hl/test_proposal.py`
- Test: `tests/hl/test_context.py`

- [ ] **Step 1: Add failing schema and loader tests**

Add tests proving every planner branch requires 2–8 unique exact module-level function names, contains `ai_func`, and rejects missing, duplicate, unknown, or out-of-range symbol arrays.

```python
def test_load_branch_briefs_requires_known_unique_code_symbols(tmp_path):
    index = {"ai_func", "_phase_memory", "_pressure_action"}
    # Valid branches include ["ai_func", "_pressure_action"].
    # Unknown, duplicate, missing-ai_func, one-item, and nine-item cases raise.
    briefs = load_branch_briefs(
        path,
        expected_count=4,
        known_code_symbols=index,
    )
    assert briefs[0].code_symbols == ("ai_func", "_pressure_action")
```

Update existing branch fixtures to include `code_symbols` so failures reflect the new contract rather than unrelated missing fields.

- [ ] **Step 2: Add failing planner-packet index test**

Pass a small `ai.py` into `write_planner_input_packet(...)` and assert `candidate_code_index` includes only module-level functions with exact name, signature, start line, and end line.

- [ ] **Step 3: Implement a shared deterministic code-index helper**

In `proposal.py`, extract the existing AST logic into:

```python
def build_candidate_code_index(source_path: str | Path) -> list[dict[str, Any]]:
    source = Path(source_path).read_text(encoding="utf-8")
    module = ast.parse(source)
    return [
        {
            "name": node.name,
            "signature": f"{node.name}({ast.unparse(node.args)})",
            "start_line": node.lineno,
            "end_line": node.end_lineno,
        }
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
```

Use it in both planner and candidate packet construction.

- [ ] **Step 4: Extend and validate `BranchBrief`**

Add `code_symbols: tuple[str, ...]` to `BranchBrief`, add the JSON-schema property, and change `load_branch_briefs` to accept `known_code_symbols`. Validate:

```python
if not 2 <= len(symbols) <= 8:
    raise ValueError("code_symbols must contain 2-8 names")
if len(set(symbols)) != len(symbols):
    raise ValueError("code_symbols must be unique")
if "ai_func" not in symbols:
    raise ValueError("code_symbols must include ai_func")
if set(symbols) - known_code_symbols:
    raise ValueError("code_symbols contain unknown module functions")
```

- [ ] **Step 5: Wire the exact workspace source into the planner**

Pass `workspace / "ai.py"` to `write_planner_input_packet` in `cli.py`. Pass the index names into `load_branch_briefs`. Update the planner prompt to explain the exact 2–8 symbol contract and forbid invented helper names.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/hl/test_proposal.py tests/hl/test_context.py -q
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/proposal.py src/agentbench_frame/hl/context.py src/agentbench_frame/hl/cli.py tests/hl/test_proposal.py tests/hl/test_context.py
git commit -m "feat: bind planner branches to policy symbols"
```

---

## Task 2: Bounded candidate source slices

**Files:**

- Modify: `src/agentbench_frame/hl/proposal.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Test: `tests/hl/test_proposal.py`
- Test: `tests/hl/test_context.py`

- [ ] **Step 1: Add failing slice-extraction tests**

Cover exact extraction order, complete bodies, `ai_func` preservation, deterministic truncation, unknown symbols, syntax errors, and the 24,000-character total-source cap.

```python
slices = build_candidate_code_slices(
    candidate_source,
    code_symbols=("ai_func", "large_helper", "small_helper"),
    source_limit=24_000,
)
assert [item["name"] for item in slices] == [
    "ai_func", "large_helper", "small_helper"
]
assert slices[0]["completeness"] == "complete"
assert sum(len(item["source"]) for item in slices) <= 24_000
```

- [ ] **Step 2: Implement `build_candidate_code_slices`**

Parse once, map module-level nodes by name, and extract `source.splitlines(keepends=True)[lineno - 1:end_lineno]`. Always emit:

```json
{
  "name": "ai_func",
  "signature": "ai_func(game_state)",
  "start_line": 1747,
  "end_line": 1931,
  "completeness": "complete",
  "source": "def ai_func(game_state): ..."
}
```

If selected non-entry helpers overflow the remaining cap, include a deterministic head/tail excerpt and mark `completeness: "truncated"`. Raise before an API call if `ai_func` alone cannot fit the configured cap.

- [ ] **Step 3: Replace candidate code index with selected slices**

Have `write_candidate_input_packet` read `branch_brief["code_symbols"]`, populate `candidate_code_slices`, and retain only the compact `candidate_code_index` for cross-reference. Add `smoke_contract` metadata placeholders for Task 3.

- [ ] **Step 4: Tighten the candidate prompt**

Add a stable marker line:

```text
candidate-context-contract: rollman-v2
```

Require the packet read first, at most one additional `ai.py` range read only for a slice marked `truncated`, at most two exact trace windows, and the first file patch no later than tool call five. Explicitly prohibit full-file printing, rediscovery, grid search, coordinate memorization, and seed/opponent identity branches.

- [ ] **Step 5: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/hl/test_proposal.py tests/hl/test_context.py -q
```

Expected: source order, caps, truncation flags, packet keys, and prompt contract pass.

- [ ] **Step 6: Commit**

```bash
git add src/agentbench_frame/hl/proposal.py src/agentbench_frame/hl/context.py tests/hl/test_proposal.py tests/hl/test_context.py
git commit -m "feat: embed bounded policy slices in candidates"
```

---

## Task 3: Reusable valid Rollman smoke verifier

**Files:**

- Create: `src/agentbench_frame/games/rollman/rollman_smoke_fixture.py`
- Create: `tests/rollman/test_smoke_fixture.py`
- Modify: `src/agentbench_frame/hl/proposal.py`
- Test: `tests/hl/test_proposal.py`

- [ ] **Step 1: Add failing fixture-state tests**

Verify boundary walls, walkable interior, walkable actor coordinates, SDK-compatible NumPy dtypes, stable dictionary conversion, and validation failures for actors on a wall or outside the board.

```python
state = make_state(
    level=1,
    round_id=7,
    pacman=(5, 5),
    ghosts=[(6, 5), (7, 5), (8, 5), (9, 5)],
)
assert np.all(state.board[0, :] == WALL)
assert state.board[5, 5] != WALL
assert state.gamestate_to_statedict()["pacman"] == [5, 5]
```

- [ ] **Step 2: Add failing verifier tests**

Create temporary policies and scenarios for:

- a passing ordered activation/preservation sequence;
- invalid action outside 0–4;
- missing activation prefix;
- forbidden preservation prefix;
- stale or manually fabricated output;
- malformed scenario and missing public `ai_func`.

The passing result must identify the freshly imported policy content hash and scenario hash so an old file cannot be reused.

- [ ] **Step 3: Implement the state fixture**

Implement `make_state(...)` with the frozen tile constants and NumPy-compatible fields. Supply `gamestate_to_statedict()` and ordered state construction without importing candidate policy code.

- [ ] **Step 4: Implement executable verification**

Expose:

```bash
python ROLLMAN_SMOKE_FIXTURE \
  --workspace CANDIDATE_WORKSPACE \
  --scenario CANDIDATE_WORKSPACE/.agentbench/smoke_scenario.json \
  --output CANDIDATE_WORKSPACE/.agentbench/candidate_smoke_result.json
```

The verifier must fresh-import `workspace/ai.py`, execute every state through public `ai_func`, require `(action, memory_id)` shape, validate action 0–4, apply activation and preservation prefix assertions, and atomically replace the result only on success. Include policy and scenario SHA-256 hashes in the result.

Scenario schema:

```json
{
  "schema_version": "1.0",
  "states": [{"level": 1, "round_id": 1, "pacman": [5, 5], "ghosts": [[6, 5], [7, 5], [8, 5], [9, 5]]}],
  "activation_state_index": 0,
  "preservation_state_index": 0,
  "activation_memory_prefix": "candidate-mechanism:",
  "preservation_forbidden_prefix": "candidate-mechanism:"
}
```

The activation and preservation indices may differ and sequential-state mechanisms may use multiple states.

- [ ] **Step 5: Put the exact smoke command in each packet**

Extend `write_candidate_input_packet` to accept the read-only fixture path and candidate workspace. Emit exact scenario, output, and invocation paths under `smoke_contract`; never ask the agent to invent a Python smoke script.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/rollman/test_smoke_fixture.py tests/hl/test_proposal.py -q
```

Expected: valid fixtures pass and all malformed/forged cases fail closed.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/games/rollman/rollman_smoke_fixture.py tests/rollman/test_smoke_fixture.py src/agentbench_frame/hl/proposal.py tests/hl/test_proposal.py
git commit -m "feat: add framework-owned Rollman smoke verifier"
```

---

## Task 4: Framework smoke gate and safe partial adoption

**Files:**

- Modify: `src/agentbench_frame/hl/controller.py`
- Modify: `tests/hl/test_controller.py`

- [ ] **Step 1: Add failing controller gate tests**

Extend the test controller factory with an optional `candidate_smoke_verifier`. Prove that staged candidates are rejected before activation and evaluator calls when the verifier is missing, raises, reports failure, has a stale hash, or lacks the scenario. Prove a successful verifier allows the existing activation and quick-screen chain.

- [ ] **Step 2: Add budget-exhaustion partial-adoption tests**

For a provider act that exhausts 70k after changing `ai.py`, assert:

- no smoke means `accepted_after_budget_exhaustion` is false;
- valid Framework smoke plus clean access, activation, and quick screen permits adoption;
- a smoke failure prevents all match calls.

- [ ] **Step 3: Add the verifier dependency**

Extend `HLController.__init__` with:

```python
candidate_smoke_verifier: Callable[..., Mapping[str, Any]] | None = None
```

For every `staged_evaluation=True` candidate, run it after snapshot creation and before activation probing. Store a normalized result or error in invocation metadata:

```python
invocation.metadata["candidate_smoke"] = {
    "status": "complete",
    "scenario_path": "...",
    "result_path": "...",
    "policy_sha256": "...",
    "scenario_sha256": "...",
}
```

Copy scenario and result into `run_root/smoke/<act_id>/` for immutable audit evidence. Do not trust a provider-written result; the callback must rerun the Framework fixture and compare hashes against the version snapshot/source.

- [ ] **Step 4: Put smoke into provider eligibility**

Require successful smoke for completed staged candidates and for `budget_candidate_eligible`. If verification fails, set a concise `candidate_smoke_failed: ...` evaluation error, skip activation and matches, and preserve the immutable failed version for audit.

- [ ] **Step 5: Preserve checkpoint/report evidence**

Extend `_refresh_checkpoint_outcome` to include normalized smoke metadata. Keep provider raw output, pending Experience update, activation event, lineage registration, and existing rollback behavior unchanged.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/hl/test_controller.py -q
```

Expected: the new gate fails closed while all existing selection, rollback, and staged-evaluation tests remain green.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/controller.py tests/hl/test_controller.py
git commit -m "feat: gate Rollman candidates on verified smoke"
```

---

## Task 5: CLI wiring and candidate-specific provider limits

**Files:**

- Modify: `src/agentbench_frame/hl/cli.py`
- Modify: `src/agentbench_frame/hl/context.py`
- Modify: `src/agentbench_frame/hl/provider.py`
- Modify: `tests/hl/test_cli.py`
- Modify: `tests/hl/test_context.py`
- Modify: `tests/hl/test_provider.py`

- [ ] **Step 1: Add failing provider-limit tests**

Keep legacy bootstrap/repair limits at 14/20, but assert a prompt containing `candidate-context-contract: rollman-v2` fails after more than six pre-edit or twelve total tool calls.

```python
assert _tool_limits(
    "# HL iteration act-b00\ncandidate-context-contract: rollman-v2"
) == (6, 12)
```

- [ ] **Step 2: Add failing CLI/context tests**

Assert real and dry-run context bundles contain `rollman_smoke_fixture.py`, candidate packets receive its exact read-only path, and `HLController` receives a verifier callback. Verify the candidate prompt tells the agent to run the exact packet command rather than authoring a custom fixture.

- [ ] **Step 3: Implement provider limits**

Check the stable marker before the generic `# HL iteration` branch:

```python
if "candidate-context-contract: rollman-v2" in prompt:
    return 6, 12
```

Leave native 20k/10k/5k token reminders and the provider-native 70k termination semantics untouched.

- [ ] **Step 4: Bundle the fixture and wire packets**

Add the fixture to `ContextBundle.create(...)` in real and dry-run paths. Pass its resolved path and candidate workspace into candidate packet construction.

- [ ] **Step 5: Implement the CLI verifier callback**

The callback must:

1. resolve only the candidate workspace, fixture, scenario, and audit output paths;
2. remove any stale result;
3. run the exact fixture command with a bounded timeout;
4. parse the new result;
5. compare policy and scenario hashes;
6. copy scenario/result to `run_root/smoke/<act_id>/`;
7. return normalized metadata or raise a concise validation error.

Do not run matches or read hidden evaluation data in this callback.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/pytest tests/hl/test_provider.py tests/hl/test_context.py tests/hl/test_cli.py -q
```

Expected: rollman-v2 acts use 6/12; unrelated coding acts retain 14/20; bundle and verifier wiring tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/agentbench_frame/hl/cli.py src/agentbench_frame/hl/context.py src/agentbench_frame/hl/provider.py tests/hl/test_cli.py tests/hl/test_context.py tests/hl/test_provider.py
git commit -m "feat: wire bounded Rollman candidate execution"
```

---

## Task 6: Regression verification and no-model preflight

**Files:**

- Modify if necessary: `README.md`
- Modify if necessary: `docs/hl.md`
- Test: entire repository

- [ ] **Step 1: Run formatting/static checks exposed by the repository**

Inspect `pyproject.toml` and run only configured checks. Fix findings without changing experimental semantics.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
.venv/bin/pytest -q
```

Expected: the complete suite passes, including existing Rollman measurement, Elo, information-gain, versioning, rollback, and certification tests.

- [ ] **Step 3: Run a no-model candidate-packet preflight**

Using the current v000016 workspace policy, construct one planner index and four representative candidate packets. Assert:

- each packet is valid JSON;
- `ai_func` is complete;
- selected-source text is at most 24,000 characters;
- exact smoke paths stay within the candidate workspace/context bundle;
- a minimal valid scenario compiles and passes the Framework verifier;
- no secret or certification seed appears in a packet.

- [ ] **Step 4: Review the diff and repository status**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional tracked changes.

- [ ] **Step 5: Commit any documentation or verification adjustments**

```bash
git add README.md docs/hl.md
git commit -m "docs: document bounded Rollman candidate contract"
```

Skip this commit if no documentation changes are necessary.

---

## Task 7: Resume cycle 7 and enforce the harness acceptance gate

**Files:**

- Observe: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable-v5-norepair/events.jsonl`
- Observe: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable-v5-norepair/provider/`
- Observe: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable-v5-norepair/smoke/`
- Observe: `.agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable-v5-norepair/versions/`

- [ ] **Step 1: Confirm the same run is resumable**

Verify the active configuration, checkpoint, champion v000000, search parent v000016, locked seeds/opponents, and no active duplicate process. Do not modify the run state manually.

- [ ] **Step 2: Resume the existing run**

Run with the API key loaded from `.env` without printing it:

```bash
/usr/bin/env AGENTBENCH_SAST_ROOT=/Users/qingle/Code/SAST \
  .venv/bin/agentbench hl resume \
  --config configs/hl/29_rollman-k4-generalizable-no-repair.yaml \
  --run-dir .agentbench/29_rollman/runs/run-20260803-gpt55-k4-generalizable-v5-norepair
```

- [ ] **Step 3: Monitor each paid act**

Track planner completion, each candidate's tool calls, token status, changed snapshot, compilation, smoke result, activation fraction, rank15/rank16 quick-screen result, Experience update, and selection. Do not leave the process unattended for more than 60 seconds without a progress check or user update.

- [ ] **Step 4: Apply the one-cycle harness gate**

Before authorizing another paid proposal cycle, require all of:

- at least 3/4 candidates changed `ai.py`;
- at least 3/4 compiled;
- at least 3/4 passed Framework-reverified activation and preservation smoke;
- zero invalid-board fixtures and zero full-file policy reads;
- no more than one provider-budget exhaustion;
- activation and match evaluation remained fail-closed.

If this gate fails, stop before another paid cycle and diagnose the exact packet/tool/smoke failure. If it passes, continue the same run toward rank15 and rank16 certification.

- [ ] **Step 5: Preserve experimental stopping semantics**

Do not declare success from search scores alone. Stop as successful only when sealed certification records at least 4/5 wins against rank15 and at least 4/5 wins against rank16. Otherwise keep the best immutable historical versions available for rollback and continue only when the harness gate supports another informative cycle.

- [ ] **Step 6: Produce the progress report and three primary curves**

Report iteration-indexed information gain, Elo, and win rate, with integer iteration x-axis, plus concise candidate/Experience evidence. Keep auxiliary plots out of the main report unless they directly diagnose a harness failure.
