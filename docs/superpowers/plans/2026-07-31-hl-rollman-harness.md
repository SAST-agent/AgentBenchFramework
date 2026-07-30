# HL Rollman Harness Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 AgentBenchFramework 内交付可复现、可回滚、token 高效、支持 `k` 候选和科研测量的通用 HL harness，并完整接入 29_rollman，但不启动模型长实验。

**Architecture:** 通用 `agentbench_frame.hl` 控制 coding-agent act、版本谱系、Codex 会话、回滚、评测调度、事件和曲线；`agentbench_frame.games.rollman` 封装冻结后端、协议、回放、决策空间和评测。所有事实写入 append-only JSONL，派生 CSV/图表从日志生成。

**Tech Stack:** Python 3.11+、标准库、pytest；可选 PyYAML、python-dotenv、matplotlib；Codex CLI JSONL；AgentBench 冻结 Python 后端。

---

## Task 1: Integrate the research measurement foundation

**Files:**
- Modify: `src/agentbench_frame/arena/match.py`
- Modify: `src/agentbench_frame/env/base.py`
- Modify: `src/agentbench_frame/runner/base.py`
- Modify: `src/agentbench_frame/runner/eval_runner.py`
- Modify: `src/agentbench_frame/tracking/`
- Create: `src/agentbench_frame/eval/benchmark.py`
- Create: `src/agentbench_frame/eval/measurement.py`
- Create: `src/agentbench_frame/eval/information_gain.py`
- Create: `src/agentbench_frame/eval/curves.py`
- Test: `tests/test_benchmark_runner_contracts.py`
- Test: `tests/test_core_regressions.py`
- Test: `tests/test_measurement_contracts.py`
- Test: `tests/test_tracking_contracts.py`

**Step 1: Add the proven foundation commits**

Run:

```bash
git cherry-pick 4a1e9e8 15da961 1a61320
```

Resolve conflicts without deleting Aquawar or PR-review functionality.

**Step 2: Run the imported contract tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_benchmark_runner_contracts.py \
  tests/test_core_regressions.py \
  tests/test_measurement_contracts.py \
  tests/test_tracking_contracts.py \
  tests/test_external_boundary_contracts.py \
  tests/test_provider_integration_contracts.py
```

Expected: PASS.

**Step 3: Verify deterministic-policy KL semantics**

Add a failing test proving that two deterministic actions over the same five-action support produce finite epsilon-regularized KL, identical actions produce zero, and missing/extra actions fail closed.

Run the single test and confirm failure for the missing contract.

**Step 4: Implement the minimal deterministic helper**

Add a helper that converts a chosen action ID and complete `ActionSupport` into the epsilon measurement distribution. Do not infer high-level intent.

Run the focused test and confirm PASS.

**Step 5: Commit**

```bash
git add src tests docs/research docs/integration
git commit -m "feat: integrate reproducible research measurement core"
```

## Task 2: Build the HL run schema, config, and append-only event model

**Files:**
- Create: `src/agentbench_frame/hl/__init__.py`
- Create: `src/agentbench_frame/hl/config.py`
- Create: `src/agentbench_frame/hl/events.py`
- Create: `src/agentbench_frame/hl/models.py`
- Test: `tests/hl/test_config.py`
- Test: `tests/hl/test_events.py`

**Step 1: Write failing config tests**

Cover:

- `max_acts: null`;
- `candidates_per_act >= 1`;
- default rollback enabled with patience 3 and score margin 0.05;
- context mode limited to `resumable|fresh`;
- API secret value rejected from serialized config;
- unknown fields fail closed.

Run:

```bash
.venv/bin/python -m pytest -q tests/hl/test_config.py
```

Expected: FAIL because modules do not exist.

**Step 2: Implement strict dataclass config**

Use standard-library dataclasses and explicit mapping parsers. Preserve unknown token values as `None`.

**Step 3: Write failing event tests**

Prove:

- finalized records append exactly once;
- common schema fields exist;
- old records are never rewritten;
- non-finite numbers and secret-like fields are rejected;
- unknown event types can be read with warnings.

**Step 4: Implement JSONL writer and reader**

Use atomic line append + flush/fsync. Keep `summary.json` derived.

**Step 5: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/hl/test_config.py tests/hl/test_events.py
git add src/agentbench_frame/hl tests/hl
git commit -m "feat: define HL run configuration and event schema"
```

## Task 3: Implement immutable versions, lineage, champion, and rollback

**Files:**
- Create: `src/agentbench_frame/hl/codebase.py`
- Create: `src/agentbench_frame/hl/lineage.py`
- Test: `tests/hl/test_codebase.py`
- Test: `tests/hl/test_lineage.py`

**Step 1: Write failing snapshot tests**

Cover content-addressed snapshots, logical versions with identical hashes, ignored run artifacts, restoration into a clean candidate workspace, and unreadable workspace failure.

**Step 2: Implement version store**

Store immutable manifests and file blobs under the run artifact root. Never overwrite a version.

**Step 3: Write failing rollback tests**

Use literal score sequences to prove:

- champion promotion follows completed benchmark score;
- three consecutive versions below champion by 0.05 select champion as next parent;
- incomplete evaluations do not count as zero or degradation;
- all candidate versions remain in lineage;
- rollback creates an event and does not delete descendants.

**Step 4: Implement lineage state machine**

Separate `latest_attempt`, `lineage_head`, `champion`, and `next_parent`.

**Step 5: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/hl/test_codebase.py tests/hl/test_lineage.py
git add src/agentbench_frame/hl tests/hl
git commit -m "feat: add immutable HL lineage and rollback"
```

## Task 4: Implement token-efficient Codex provider sessions

**Files:**
- Modify: `src/agentbench_frame/tracking/providers.py`
- Create: `src/agentbench_frame/hl/provider.py`
- Create: `src/agentbench_frame/hl/context.py`
- Test: `tests/hl/test_provider.py`
- Test: `tests/hl/test_context.py`

**Step 1: Write failing command-construction tests**

Prove:

- first act uses `codex exec --json`;
- resumed act uses the recorded thread ID;
- model/reasoning/provider flags are passed without embedding the key;
- key is read only from configured env var;
- raw JSONL, thread ID, exact/unknown token usage and failure status are retained.

Use a fake executable; do not call a real model.

**Step 2: Implement resumable provider**

Keep subprocess invocation provider-neutral. Whitelist environment variables. Record command with secret values redacted.

**Step 3: Write failing context tests**

Prove the incremental prompt references immutable context paths instead of embedding rule text or source code, includes replay evidence and previous measurements, requires a causal hypothesis, blocks blind grid search, and distinguishes `k` branches by mechanism.

**Step 4: Implement context bundle and checkpoint manifest**

Static files are hashed once. Every act saves exact prompt, selected replay IDs, parent version, experience hash and session metadata. `fresh` mode rehydrates from checkpoint.

**Step 5: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/hl/test_provider.py tests/hl/test_context.py
git add src tests/hl
git commit -m "feat: add resumable token-efficient Codex acts"
```

## Task 5: Implement controller, k candidates, gates, and Experience Skill

**Files:**
- Create: `src/agentbench_frame/hl/controller.py`
- Create: `src/agentbench_frame/hl/experience.py`
- Create: `src/agentbench_frame/hl/evaluator.py`
- Test: `tests/hl/test_controller.py`
- Test: `tests/hl/test_experience.py`
- Test: `tests/hl/test_e2e_fake.py`

**Step 1: Write failing controller tests**

Cover:

- `k=1` default;
- `k=3` creates three siblings from one parent;
- every invocation creates a logical version;
- fixed gate selects by complete score;
- failed candidates keep artifacts and missing score;
- rollback changes the next parent;
- no hard max acts when `max_acts=None`;
- success criteria stop the loop;
- stagnation requests strategy diagnosis rather than numeric enumeration.

**Step 2: Implement controller**

Use dependency injection for provider and game evaluator. Write finalized lifecycle events only.

**Step 3: Write failing Experience Skill tests**

Verify structured sections for stable knowledge, failed hypotheses, replay evidence, active questions and compression. Ensure source code and secrets cannot enter the Skill.

**Step 4: Implement Experience Skill manager**

Persist one version per act and a canonical compact file for the next prompt.

**Step 5: Add fake end-to-end test**

A deterministic fake provider edits a fixture candidate; a fake evaluator returns literal outcomes including degradation and recovery. Assert versions, rollback, tokens, events and summary.

**Step 6: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/hl
git add src/agentbench_frame/hl tests/hl
git commit -m "feat: orchestrate replay-driven HL candidate evolution"
```

## Task 6: Add Elo, research tables, and multi-panel curves

**Files:**
- Modify: `src/agentbench_frame/arena/rating.py`
- Create: `src/agentbench_frame/hl/report.py`
- Modify: `src/agentbench_frame/report/builder.py`
- Test: `tests/test_rating.py`
- Test: `tests/hl/test_report.py`

**Step 1: Write failing Elo tests**

Prove:

- each valid game updates once;
- candidate win/loss/draw directions are correct;
- role namespace is explicit;
- invalid games do not update;
- series order is deterministic;
- persisted history includes act, version, opponent, seed and rating.

**Step 2: Implement role-scoped Elo ledger**

Do not reuse the known seat-swap aggregation path. Keep human pool anchors fixed by config when requested.

**Step 3: Write failing report tests**

Use a hand-authored event fixture and assert exact CSV values for score, gain, best score, win rate, Elo, mean KL, occupancy shift and token totals. Assert missing values remain empty.

**Step 4: Implement report derivation and plotting**

Matplotlib is optional at import time. Emit CSV always and PNG/SVG when report extras are installed.

**Step 5: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/test_rating.py tests/hl/test_report.py
git add src tests
git commit -m "feat: report Elo and HL research curves"
```

## Task 7: Add the frozen Rollman contract and source audit

**Files:**
- Create: `src/agentbench_frame/games/__init__.py`
- Create: `src/agentbench_frame/games/rollman/__init__.py`
- Create: `src/agentbench_frame/games/rollman/contract.py`
- Create: `src/agentbench_frame/games/rollman/assets/rules.md`
- Create: `src/agentbench_frame/games/rollman/assets/decision_space.yaml`
- Create: `src/agentbench_frame/games/rollman/assets/source_manifest.json`
- Create: `tools/audit_rollman_sources.py`
- Test: `tests/rollman/test_contract.py`
- Test: `tests/rollman/test_source_audit.py`

**Step 1: Write failing contract tests**

Load the frozen backend from a configurable AgentBench root and assert literal values:

- directions 0..4;
- map sizes 41, 32, 22;
- max rounds 500, 400, 300;
- skills 8, 8, 8, 8, 2;
- portal rounds 60 and 50;
- board enum 0..9;
- event enum 0..3;
- score constants.

**Step 2: Implement contract parser and source manifest**

Parse Python source safely with AST; do not import and execute the backend for static audit.

**Step 3: Write rules and decision-space assets**

Use frozen backend semantics. Include authoritative URLs and fixed commits. Explicitly document collisions, effect timing, transition order, action legality and replay fields.

**Step 4: Add source audit CLI**

Compare frozen core SHA-256 with the fixed official Logic-core files and print machine-readable JSON.

**Step 5: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/rollman/test_contract.py tests/rollman/test_source_audit.py
git add src/agentbench_frame/games tools tests/rollman
git commit -m "feat: freeze the Rollman scientific contract"
```

## Task 8: Create and pressure-test the Rollman replay Skill

**Files:**
- Create: `src/agentbench_frame/games/rollman/assets/replay-skill/SKILL.md`
- Create: `src/agentbench_frame/games/rollman/assets/replay-skill/agents/openai.yaml`
- Create: `src/agentbench_frame/games/rollman/assets/replay-skill/scripts/summarize_replay.py`
- Test: `tests/rollman/test_replay_skill.py`

**Step 1: Generate the skill scaffold**

Use the system `skill-creator` scaffold script with the project asset directory as target.

**Step 2: Write a failing replay fixture test**

Construct a minimal JSONL replay with initialization, normal round, shield destruction, eaten event, level finish and terminal reason. Assert the script emits exact natural-language facts and decision records.

**Step 3: Implement the parser script**

Validate required fields and fail closed on malformed or truncated input. Emit JSON and concise Markdown modes.

**Step 4: Write SKILL.md**

Teach:

- initialization versus incremental frames;
- coordinates and board enums;
- events 0..3;
- paths and collision interpretation;
- item/score deltas;
- portal timing;
- failure diagnosis;
- how to cite exact round/level/evidence;
- how to avoid inferring hidden intent.

**Step 5: Pressure-test the Skill**

Use the required writing-skills baseline and Skill-enabled subagent scenarios with the same fixture and adversarial malformed replays. Refine only for observed failures.

**Step 6: Validate metadata and run tests**

Run:

```bash
.venv/bin/python /Users/qingle/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  src/agentbench_frame/games/rollman/assets/replay-skill
.venv/bin/python -m pytest -q tests/rollman/test_replay_skill.py
```

**Step 7: Commit**

```bash
git add src/agentbench_frame/games/rollman/assets/replay-skill tests/rollman
git commit -m "feat: add the Rollman replay analysis skill"
```

## Task 9: Implement the Rollman adapter and deterministic evaluator

**Files:**
- Create: `src/agentbench_frame/games/rollman/protocol.py`
- Create: `src/agentbench_frame/games/rollman/replay.py`
- Create: `src/agentbench_frame/games/rollman/match.py`
- Create: `src/agentbench_frame/games/rollman/evaluator.py`
- Create: `src/agentbench_frame/games/rollman/measurement.py`
- Test: `tests/rollman/fixtures/`
- Test: `tests/rollman/test_protocol.py`
- Test: `tests/rollman/test_replay.py`
- Test: `tests/rollman/test_match.py`
- Test: `tests/rollman/test_evaluator.py`
- Test: `tests/rollman/test_measurement.py`

**Step 1: Write protocol and replay tests**

Cover 4-byte/8-byte big-endian packets, role action payloads, initialization, incremental frames and terminal frames.

**Step 2: Implement protocol and replay modules**

Keep raw replay lines and parsed records. Never collapse infrastructure errors into losses.

**Step 3: Write deterministic match tests**

Use fixture AIs and frozen backend. Run the same seed twice and assert identical normalized replay and scores. This test must fail against unseeded backend construction.

**Step 4: Implement match runner seed injection**

Set both `random` and NumPy seeds before environment construction. Execute candidate/opponent with explicit timeouts and isolated working directories.

**Step 5: Write evaluator and measurement tests**

Cover rank-1 learning matches, 16-opponent certification specs, opponent-source isolation, score completeness, decision trace, five-action support, epsilon KL and occupancy state IDs.

**Step 6: Implement evaluator and measurement**

Extract the target Rollman decision emitted at every reached decision point. Compare versions on the same frozen cases.

**Step 7: Run and commit**

```bash
.venv/bin/python -m pytest -q tests/rollman
git add src/agentbench_frame/games/rollman tests/rollman
git commit -m "feat: evaluate Rollman against the frozen human pool"
```

## Task 10: Add CLI, default local config, and dry-run workflow

**Files:**
- Modify: `src/agentbench_frame/cli.py`
- Create: `src/agentbench_frame/hl/cli.py`
- Create: `configs/hl/29_rollman.yaml`
- Create: `.env.example`
- Modify: `.gitignore`
- Create: `docs/hl/rollman-local-run.md`
- Test: `tests/hl/test_cli.py`
- Test: `tests/rollman/test_cli_dry_run.py`

**Step 1: Write failing CLI tests**

Cover:

- config validation;
- `--dry-run`;
- `--audit-sources`;
- `--render-report`;
- missing API env var error only for real provider run;
- no secret in stdout or artifacts;
- default `k=1`, rollback enabled, `max_acts=null`.

**Step 2: Implement CLI**

Commands:

```text
agentbench hl validate
agentbench hl audit
agentbench hl run
agentbench hl resume
agentbench hl report
```

**Step 3: Add default config and operator guide**

Use `AGENTBENCH_API_KEY` as placeholder env key. Do not include a real key. Document the custom Responses provider fields and the explicit command the user will run after authorization.

**Step 4: Run dry-run**

Run:

```bash
.venv/bin/agentbench hl validate --config configs/hl/29_rollman.yaml
.venv/bin/agentbench hl audit --config configs/hl/29_rollman.yaml
.venv/bin/agentbench hl run --dry-run --config configs/hl/29_rollman.yaml
```

Expected: all PASS without model calls.

**Step 5: Commit**

```bash
git add src configs .env.example .gitignore docs tests
git commit -m "feat: expose the local Rollman HL workflow"
```

## Task 11: Full verification and handoff

**Files:**
- Modify as required by verification findings.

**Step 1: Install optional test/report dependencies**

```bash
.venv/bin/python -m pip install pytest pyyaml python-dotenv matplotlib
```

**Step 2: Run the full suite outside restricted socket sandbox**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests PASS. The existing PR-review HTTP tests require loopback socket permission.

**Step 3: Run source audit and deterministic integration**

```bash
.venv/bin/agentbench hl audit --config configs/hl/29_rollman.yaml
.venv/bin/python -m pytest -q tests/rollman/test_match.py tests/rollman/test_evaluator.py
```

**Step 4: Run static and secret checks**

```bash
.venv/bin/python -m compileall -q src tests
git grep -nE 'sk-[A-Za-z0-9_-]{12,}'
git status --short
```

Expected: compile succeeds, secret grep returns no matches, status contains only intended files.

**Step 5: Generate a fake-run report**

```bash
.venv/bin/python -m pytest -q tests/hl/test_e2e_fake.py tests/hl/test_report.py
```

Inspect PNG/SVG and CSV artifacts for the six required panels/columns.

**Step 6: Request code review**

Use `superpowers:requesting-code-review`, address blocking findings with `superpowers:receiving-code-review`, and rerun all relevant tests.

**Step 7: Finish the development branch**

Use `superpowers:verification-before-completion` and `superpowers:finishing-a-development-branch`. Do not start a real Codex/model experiment.

