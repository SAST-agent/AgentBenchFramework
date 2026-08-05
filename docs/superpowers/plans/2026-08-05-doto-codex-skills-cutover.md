# DOTO Codex Skills and Loop Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic DOTO harness instructions and framework-owned LLM loop with four validated Codex Skills, then demonstrate a complete Codex-directed official benchmark Run.

**Architecture:** A workflow Skill teaches atomic lifecycle commands and requires domain companions for rules, C++ authoring, and replay diagnosis. Skills are versioned inputs copied into every Run. After Skill and CLI examples pass validation, remove LLM/API loop code while retaining deterministic tools.

**Tech Stack:** Codex SKILL.md packages, Markdown references, Python CLI examples, native compile/replay fixtures, pytest, ripgrep scans.

## Global Constraints

- Invoke `skill-creator` and `superpowers:writing-skills` before authoring Skills.
- Exact packages: `doto-benchmark-run`, `doto-game-rules`, `doto-agent-authoring`, `doto-replay-reader`.
- No Skill contains a fixed iteration loop or LLM API configuration.
- Numeric rules point to vendored authoritative server/map/SDK locations.
- Replay guidance passes the historical short replay fixture and actual parse command.
- Remove LLM loop only after atomic lifecycle E2E passes.
- Retain build, population, match, replay, evaluate, IG, lifecycle, and projection tools.
- Every Run copies exact Skill packages and SHA-256 hashes.
- Skills never reveal sealed test policies.

---

## File Map

- Replace `skills/doto-harness/` with `skills/doto-benchmark-run/`.
- Create `skills/doto-game-rules/` and `skills/doto-agent-authoring/`.
- Refine `skills/doto-replay-reader/`.
- Create `tests/doto/test_skills.py`; modify `test_docs_examples.py` and `test_run_store.py`.
- Remove `doto/llm_client.py`, `doto/loop_config.py`, `doto/loop.py`, and their tests.
- Remove `examples/doto-loop.toml`.
- Rewrite `docs/doto-harness.md`, official acceptance, and README DOTO section.

### Task 1: Create the Authoritative Game Rules Skill

**Files:**
- Create: `skills/doto-game-rules/SKILL.md`
- Create: `skills/doto-game-rules/references/rules.md`
- Create: `skills/doto-game-rules/references/decision-space.md`
- Create: `skills/doto-game-rules/agents/openai.yaml`
- Create: `tests/doto/test_skills.py`

**Interfaces:** Produces rule/decision guidance consumed by all other DOTO Skills.

- [ ] **Step 1: Read required Skill authoring instructions**

Read complete `skill-creator` and `superpowers:writing-skills` instructions before editing. Add their validation commands to this task.

- [ ] **Step 2: Write failing source-consistency tests**

```python
def test_rules_match_official_constants():
    text = RULES_REFERENCE.read_text()
    assert all(x in text for x in ("320 × 320", "6000", "20 FPS", "80 points"))
    assert "seed 11" not in text

def test_decision_reference_names_joint_action():
    text = DECISION_REFERENCE.read_text()
    assert all(x in text for x in ("move[5]", "shoot[5]", "meteor[5]", "flash[5]"))
```

- [ ] **Step 3: Run failing test**

Run `uv run --extra doto --with pytest python -m pytest tests/doto/test_skills.py -v`; expect missing Skill failure.

- [ ] **Step 4: Write routed Skill/references**

Keep SKILL.md short. Put scoring, visibility, movement, attacks, revival, timing, and terminal semantics in `rules.md`. Put observation, canonical joint action, mask, continuous support, and termination in `decision-space.md`. Point every numeric claim to an authoritative file/function.

- [ ] **Step 5: Validate, test, and commit**

Run skill validation and the test above; expect PASS. Commit `feat(doto): teach Codex authoritative game rules`.

### Task 2: Create the C++ Agent Authoring Skill

**Files:**
- Create: `skills/doto-agent-authoring/SKILL.md`
- Create: `skills/doto-agent-authoring/references/sdk-api.md`
- Create: `skills/doto-agent-authoring/references/safe-policy-patterns.md`
- Create: `skills/doto-agent-authoring/agents/openai.yaml`
- Modify: `tests/doto/test_skills.py`

**Interfaces:** Produces complete candidate/SDK contract consumed when Codex writes `playerAI.cpp`.

- [ ] **Step 1: Write failing contract/compile tests**

```python
def test_authoring_owns_only_complete_player_ai():
    text = AUTHORING_SKILL.read_text()
    assert "complete playerAI.cpp" in text
    assert "void playerAI()" in text
    assert all(x in text for x in ("official_server", "sdk/main.cpp", "sealed test"))

def test_sdk_example_compiles(tmp_path):
    result = build_source_text(extract_cpp_example(SDK_REFERENCE), tmp_path)
    assert result.exit_code == 0, result.stderr
```

- [ ] **Step 2: Run failing test**

Run Skill tests with `-k authoring`; expect missing package failure.

- [ ] **Step 3: Write SDK/safety guidance**

Document `Logic::Instance()` fields/setters from fixed headers, local slot `i` to global ID `i*2+faction`, absolute coordinates, no-op sentinel, cooldown/use/death checks, finite coordinates, persistent state, frame timing, and compilation. Compile-backed examples must not encode a strong fixed strategy.

- [ ] **Step 4: Validate, test, and commit**

Run Skill validation plus compile-backed tests; expect PASS. Commit `feat(doto): teach Codex native policy authoring`.

### Task 3: Refine and Validate Replay Reader

**Files:**
- Modify: `skills/doto-replay-reader/SKILL.md`
- Create: `skills/doto-replay-reader/references/frame-schema.md`
- Create: `skills/doto-replay-reader/references/events.md`
- Create: `skills/doto-replay-reader/references/diagnosis.md`
- Modify: `tests/doto/test_skills.py`, `test_docs_examples.py`

**Interfaces:** Produces fixture-backed replay/trace diagnosis consumed after matches.

- [ ] **Step 1: Write failing fixture-backed tests**

```python
def test_replay_claims_match_parser(real_short_replay):
    claims = load_machine_readable_claims(REPLAY_SKILL)
    summary = summarize_replay(real_short_replay)
    assert claims["final_scores"] == list(summary.final_scores)
    assert claims["last_frame"] == summary.last_frame
    assert claims["event_counts"] == summary.event_counts

def test_events_are_not_described_as_observation():
    assert "events are replay-only" in REPLAY_SKILL.read_text()
```

- [ ] **Step 2: Run failing tests**

Run Skill/docs replay tests; expect missing references/claims failure.

- [ ] **Step 3: Split references without duplicating rules**

Keep artifact choice and workflow in SKILL.md. Move row schemas, events, and diagnosis to references; link mechanics to `doto-game-rules`. Include a machine-readable fixture claims block.

- [ ] **Step 4: Execute documented parser**

Run `uv run python -m agentbench_frame.doto replay --path tests/doto/fixtures/real_short_replay.zip --jsonl /tmp/doto-short.events.jsonl`. Expect scores `[12.0, 7.0]`, last frame 1, and expected event counts.

- [ ] **Step 5: Validate, test, and commit**

Run validation and replay tests; expect PASS. Commit `docs(doto): validate Codex replay diagnosis Skill`.

### Task 4: Replace Harness with Atomic Workflow Skill

**Files:**
- Remove: `skills/doto-harness/`
- Create: `skills/doto-benchmark-run/SKILL.md`
- Create: `skills/doto-benchmark-run/references/lifecycle.md`
- Create: `skills/doto-benchmark-run/references/results.md`
- Create: `skills/doto-benchmark-run/agents/openai.yaml`
- Modify: `tests/doto/test_skills.py`, `test_docs_examples.py`

**Interfaces:** Produces routing through all Run/iteration commands and three companion Skills.

- [ ] **Step 1: Write failing workflow invariant tests**

```python
def test_workflow_names_atomic_commands():
    text = WORKFLOW_SKILL.read_text()
    for command in ("run init", "run status", "iteration begin", "iteration build",
                    "iteration evaluate", "iteration compare", "iteration close",
                    "run finalize", "run export"):
        assert command in text

def test_workflow_has_no_llm_loop():
    text = WORKFLOW_SKILL.read_text()
    assert "final test exactly once" in text
    assert "Do not edit authoritative result files" in text
    assert "OpenAI API" not in text and "Chat Completions" not in text
```

- [ ] **Step 2: Run failing tests**

Run Skill tests with `-k workflow`; expect missing package failure.

- [ ] **Step 3: Write workflow routing/invariants**

Require rules+authoring before first candidate, replay-reader before interpreting replay, workflow Skill for transitions. Explain formal/diagnostic separation, explicit parents, failure closure, training-only selection, one-shot test, DotoResults authority, and AgentBenchResults projection.

- [ ] **Step 4: Parser-test every command example**

Extract fenced commands, substitute fixtures, and invoke parser/help. No example may mention `doto loop`.

- [ ] **Step 5: Validate, test, and commit**

Run validation plus Skill/docs tests; expect PASS. Commit `feat(doto): guide Codex through atomic benchmark runs`.

### Task 5: Snapshot Complete Skill Packages

**Files:**
- Modify: `src/agentbench_frame/doto/run_store.py`
- Modify: `tests/doto/test_run_store.py`, `test_skills.py`

**Interfaces:** Produces `snapshot_skills(run_dir: Path, roots: Mapping[str, Path]) -> tuple[SkillSnapshot, ...]`.

- [ ] **Step 1: Write failing exact-set/hash test**

```python
def test_run_snapshots_exact_four_packages(run_dir):
    rows = load_json(run_dir / "skills/manifest.json")["skills"]
    assert {r["name"] for r in rows} == {
        "doto-benchmark-run", "doto-game-rules",
        "doto-agent-authoring", "doto-replay-reader",
    }
    assert all(r["sha256"] == hash_directory(run_dir / "skills" / r["name"]) for r in rows)
```

- [ ] **Step 2: Implement atomic package snapshots**

Copy SKILL.md, references, and agents. Reject missing package, duplicate name, escaping symlink, or hash mismatch. Write manifest only after all copies complete.

- [ ] **Step 3: Test and commit**

Run store/Skill tests; expect PASS. Commit `feat(doto): preserve exact Codex Skill provenance`.

### Task 6: Remove Framework-Owned LLM Loop

**Files:**
- Remove: `src/agentbench_frame/doto/llm_client.py`, `loop_config.py`, and `loop.py`
- Remove: `tests/doto/test_llm_client.py`, `test_loop_config.py`, `test_loop.py`, and `test_loop_e2e.py`
- Remove: `examples/doto-loop.toml`
- Modify: `src/agentbench_frame/doto/cli.py`, `tests/doto/test_cli.py`

**Interfaces:** Removes `doto loop`, `ChatCompletionsClient`, `LoopConfig`; retains all deterministic tools.

- [ ] **Step 1: Write failing obsolete-surface scan**

```python
def test_doto_has_no_framework_llm_surface():
    text = "\n".join(p.read_text() for p in DOTO_PACKAGE.rglob("*.py"))
    for forbidden in ("ChatCompletionsClient", "OPENAI_API_KEY", "api_key_env",
                      "reasoning_effort", "max_total_tokens", "def _loop("):
        assert forbidden not in text
    assert "loop" not in build_parser()._subparsers._group_actions[0].choices
```

- [ ] **Step 2: Confirm scan fails**

Run CLI scan test; expect failure while loop exists.

- [ ] **Step 3: Delete orchestration only**

Remove listed files/parser/imports/examples. Do not delete IG, score, Run, population, evaluation, replay, or projection code.

- [ ] **Step 4: Test and commit**

Run all DOTO tests; expect PASS. Commit `refactor(doto): remove framework-owned LLM loop`.

### Task 7: Rewrite Operator Documentation

**Files:**
- Modify: `README.md`
- Replace: `docs/doto-harness.md`
- Modify: `docs/doto-official-acceptance.md`
- Modify: `tests/doto/test_docs_examples.py`

**Interfaces:** Produces one operator path through DotoResults validation and AgentBenchResults projection.

- [ ] **Step 1: Write failing documentation scan**

```python
def test_docs_name_codex_as_orchestrator():
    text = DOTO_DOC.read_text()
    assert all(x in text for x in ("Codex decides", "DotoResults", "AgentBenchResults projection"))
    assert "Chat Completions" not in text and "doto loop" not in text
```

- [ ] **Step 2: Rewrite exact workflow**

Document DOTO extra installation, pool preparation, four Skills, lifecycle commands, diagnostics, formal 30-cell iteration, final 56-cell test, validation/report, export, resume, and failures.

- [ ] **Step 3: Parser-test docs and commit**

Run docs examples; expect PASS. Commit `docs(doto): document Codex-orchestrated benchmark`.

### Task 8: Real Codex-Directed Acceptance

**Files:**
- Runtime create: `../DotoResults/runs/23_doto/<agent>/<run_id>/`
- Runtime export: `../AgentBenchResults/runs/23_doto/<agent>/<run_id>/`

**Interfaces:** Consumes completed Framework, four Skills, sealed pool, DotoResults, unchanged AgentBenchResults.

- [ ] **Step 1: Run both full automated suites**

Framework: `uv run --extra doto --with pytest python -m pytest -q`. DotoResults: `uv run --with pytest python -m pytest -q`. Expected: PASS.

- [ ] **Step 2: Prepare isolated Codex workspace**

Include Framework, four Skills, SDK/server, and 15 training policies. Exclude original AgentBench corpus, test source, sealed paths, credentials, and old hidden-test results. Build sealed bundle outside.

- [ ] **Step 3: Have Codex execute genuine parent/child closure**

Without a fixed orchestration script: baseline 30 matches; inspect replay; create non-identical child with analysis; build; 30 matches; IG; close both iterations.

- [ ] **Step 4: Finalize once**

Choose from training evidence, run 56 hidden matches at 70-percent CPU target, seal, validate, and export. Do not feed test result into another modification.

- [ ] **Step 5: Validate storage/science**

Run DotoResults validate/aggregate and unchanged AgentBenchResults aggregate. Require 30 cells per formal iteration, 56 final cells, aligned source/parent curves, strict KL status, and retained failures.

- [ ] **Step 6: Scan and publish**

Search both results for API key value, test source fragments, absolute sealed paths, and non-finite JSON; expect none. Commit authoritative Run and five-file projection separately; exclude binaries, caches, locks, and unredacted sealed metadata.
