# Generals HL Closed-Loop Design

**Date:** 2026-07-26
**Status:** Approved on 2026-07-26
**Benchmark version:** `generals-hl-pilot-v1`

## 1. Goal

Build one reproducible Generals heuristic-learning pilot:

```text
official game logic
→ human-opponent matches
→ structured learning replays
→ one non-interactive Codex act
→ immutable v1 snapshot
→ evaluation on the same frozen matrix
→ Framework events, metrics, quality diagnostics, and CI-compatible data
```

The pilot must produce a complete `v0 → v1` evidence chain. It is not a
multi-round leaderboard or an RL experiment.

## 2. Project Boundaries

### AgentBench

`AgentBench` is the source of game assets:

- the official 28th-contest Generals Python logic is the game-rule authority;
- a new deterministic Python rule baseline is built on the official SDK;
- three historical submissions form the frozen pilot population;
- benchmark configuration, game rules, replay interpretation, and local
  executable adapters live beside the Generals assets.

The existing historical archives and extracted submissions remain unchanged.
Generated binaries, temporary workspaces, replays, and run data are not
committed.

### AgentBenchFramework

`AgentBenchFramework` owns reusable orchestration:

- opponent preparation and process lifecycle;
- match and evaluation adapters;
- frozen benchmark cases;
- replay selection and prompt assembly;
- Codex invocation;
- workspace manifests, versions, and diffs;
- research events, budgets, metrics, diagnostics, and report input.

Implementation starts from commit `1a61320` on
`origin/worktree/framework`, which already contains the provider-neutral
controller and research measurement contracts.

### Existing `GeneralsEnv`

`agentbench_frame.env.GeneralsEnv` remains available for smoke tests and RL
prototyping. It does not generate results for `generals-hl-pilot-v1`.
Scientific results use the official logic under
`AgentBench/backend_sources/corpus/28_generals/logic/gamecode_logic`.

## 3. Considered Architectures

### Selected: official engine adapter plus Framework orchestration

A thin local judger drives the official `GameState`,
`execute_single_command`, `update_round`, and replay functions. Player
programs retain the historical SDK protocol. The Framework consumes
canonical match JSONL and never duplicates game rules.

This provides authoritative rules, historical-agent compatibility, and a
reusable Framework boundary.

### Rejected: port all official rules into `GeneralsEnv`

This is easier to embed but creates two game implementations whose behavior
must be proven equivalent. That validation is larger than the pilot and
would weaken the provenance of the first research run.

### Rejected: recreate the complete Saiblo judger

The preserved logic exposes the original judger protocol, but the complete
platform runtime is not present. Recreating it adds infrastructure that is
not required to run deterministic local matches.

## 4. Frozen Population

The pilot uses these exact opponents:

| Tier | Submission | Language | Source |
|---|---|---|---|
| high | `advanced-rank02-robinliu-v18` | C++14 | `top_algorithms/corpus/28_generals_advanced_final/extracted/rank02__robinliu__改修型__v18/generals_impact_sdk_cpp-master` |
| medium | `advanced-rank08-nashjunheng-v20` | Python | `top_algorithms/corpus/28_generals_advanced_final/extracted/rank08__nashjunheng__55__v20/55att_gene_sold` |
| low | `popular-rank16-xiaoaojianghu-v1` | Python | `top_algorithms/corpus/28_generals_popular_final/extracted/rank16__Xiaoaojianghu__痔取松弛肛__v1` |

All table paths are relative to the `AgentBench` repository root. Rank 2 is
used for the high tier because static inspection found that rank 1 seeds
strategy randomness from wall-clock time; rank 2 retains a high contest
score while avoiding that known reproducibility defect.

The C++ opponent is built with its preserved makefile. Python opponents run
from temporary copies of their preserved source roots. Wrappers may set
working directories and module paths, but may not change opponent strategy
code.

The Codex prompt never exposes opponent source code. It identifies opponents
only by stable tier and benchmark IDs.

## 5. Baseline Workspace

The editable baseline is a small deterministic strategy built on the
official Python SDK. Its behavior is intentionally decomposed into readable
rules:

1. validate and normalize the current official SDK state;
2. prioritize legal captures of adjacent neutral resource generals;
3. move the main general toward the nearest reachable neutral resource;
4. upgrade owned production when the exact official cost is affordable;
5. move spare army toward contested or enemy cells;
6. terminate every decision with `END_TURN`.

The baseline contains:

- an SDK-compatible `main.py` entrypoint;
- focused strategy modules;
- unit tests for deterministic rule selection and legal command formatting;
- a short strategy document for Codex.

For every run, the baseline template is copied to a new run-owned workspace.
The template is never edited. The initial copy is snapshotted as `v0`.

Codex receives `workspace-write` access only to that copy. The workspace does
not contain the Framework, official engine, opponent source, evaluation
seeds, or evaluation replays.

## 6. Official Match Adapter

### Deterministic initialization

For each case the adapter performs the same initialization sequence as
official `main.py`:

1. call `random.seed(case.seed)`;
2. construct `GameState`;
3. call `update_map`;
4. call `init_generals`;
5. set both players' coins to `40`;
6. serialize a player-specific initial state with
   `trans_state_to_init_json`.

### Player protocol

Each player is a separate subprocess. The adapter:

1. writes the player-specific initialization JSON line to stdin;
2. reads the four-byte big-endian response length;
3. reads exactly that many UTF-8 bytes;
4. requires a newline-delimited integer command list ending in command `8`;
5. applies each command with official `execute_single_command`;
6. forwards the accepted command list to the other player;
7. calls official `update_round` after player 1 finishes;
8. checks the official game-over rule after commands and round updates.

The adapter clamps `MOVE_ARMY` counts to `2_100_000_000`, matching official
`main.py`. Invalid framing, malformed commands, illegal commands, premature
exit, timeout, and output overflow are distinct termination reasons.

### Local process restrictions

Docker is not used. This is explicitly a reliability boundary, not a secure
sandbox. Processes use:

- `shell=False`;
- an allowlisted environment;
- an isolated temporary working directory for generated output;
- a 10-second startup timeout;
- a 2-second per-decision timeout;
- a 600-second whole-match timeout;
- a 64 KiB maximum command packet;
- a 10 MiB cap for each stdout/stderr artifact;
- process-group termination on timeout or controller exit.

Historical submissions are untrusted. The documentation states that formal
shared-machine execution requires an external sandbox even though this local
pilot does not use Docker.

## 7. Evaluation and Learning Protocol

### Frozen evaluation

Evaluation seeds are:

```text
280101
280202
280303
```

Each version plays every opponent on every seed from both seats:

```text
3 opponents × 3 seeds × 2 seats = 18 cases per version
```

The same ordered `BenchmarkSpec` is used for `v0` and `v1`. Case IDs encode
benchmark version, opponent ID, seed, and evaluated-agent seat. The
aggregate score is:

```text
(wins + 0.5 × draws) / 18
```

It is emitted only when all 18 cases have valid results. Per-tier scores and
seat gaps are derived secondary metrics.

### Learning feedback

Learning seeds are disjoint from evaluation seeds:

```text
281101
281202
281303
```

`v0` plays the low and medium opponents on all three learning seeds from
both seats:

```text
2 opponents × 3 seeds × 2 seats = 12 learning episodes
```

The high-tier opponent is held out from feedback. Codex receives the
official rules, replay interpretation guide, current strategy description,
and the 12 structured learning replays. It receives no evaluation replay,
evaluation result, evaluation seed, or opponent implementation.

### One Codex act

The controller invokes:

```text
codex exec --json --sandbox workspace-write
```

The installed Codex CLI path, CLI version, actual provider metadata, exact
prompt, raw JSONL, tool calls, token usage, elapsed time, exit status, and
stderr are retained. Provider usage absent from the stream remains
`unknown`, never `0`.

After invocation, the controller captures `v1` even when no files changed.
It writes both manifests and a unified `v0-to-v1.patch`. It runs baseline
unit and protocol smoke tests before evaluation. `v1` is evaluated only if
the workspace remains readable and executable.

There is no accept/reject gate, automatic rollback, or best-version
substitution. A weaker `v1` is valid evidence.

## 8. Replay and Measurement Model

Every match stores:

- `game_id`, benchmark version, engine content hash, agent version;
- opponent ID and tier;
- seed and evaluated-agent seat;
- round, acting player, accepted macro-action, and command list;
- pre-action state reference and post-action state reference;
- winner, score, termination type, runtime, and process status;
- raw official replay records.

An episode is one complete game. An environment step is one player's
accepted turn-level macro-action. Primitive commands inside that
macro-action are recorded but do not increment environment steps.

The official historical agents expose deterministic macro-actions, not a
normalized probability distribution over the complete legal macro-action
support. Therefore the pilot does not mislabel a finite proxy as strict
policy KL. It records:

- canonical state IDs;
- old/new macro-actions on a shared probe-state set derived only from
  learning trajectories;
- `action_disagreement_trace`;
- separate occupancy samples and `occupancy_shift`.

`policy_kl_trace` is absent with the explicit reason
`complete_macro_action_distribution_unavailable`. The CI report displays
action disagreement as behavioral change, not epistemic information gain.

## 9. Run Artifacts

The canonical run layout is:

```text
runs/28_generals/{run_id}/
├── run.toml
├── events.jsonl
├── summary.json
├── benchmark/
│   ├── spec.json
│   └── population.json
├── provider/
│   ├── codex.raw.jsonl
│   ├── prompt.md
│   └── stderr.log
├── matches/{match_id}/
│   ├── metadata.json
│   ├── replay.jsonl
│   └── players/{0,1}/
│       ├── agent.protocol.bin
│       └── agent.stderr.log
├── versions/
│   ├── v0/manifest.json
│   ├── v1/manifest.json
│   └── v0-to-v1.patch
└── quality.json
```

`events.jsonl` is the append-only fact source. `summary.json` and reports
are derived. Authentication data and unrelated environment variables are
never written.

## 10. Metrics and Budgets

The run emits:

- `raw_score = score(v0)`;
- `evo_score = score(v1)`;
- `gain = evo_score - raw_score`;
- wins, losses, draws, per-tier score, and seat gap;
- `AUC_coding_agent_act`;
- `AUC_episode`, `AUC_env_step`, `AUC_token`, and `AUC_time` when their axes
  are fully known;
- learning, evaluation, and total episodes, environment steps, tokens, and
  time as separate budget namespaces;
- action disagreement and occupancy-shift traces.

The main efficiency axes use learning-only budgets. Evaluation cost remains
available as a separate and total budget.

With one act, `AUC_coding_agent_act` is the trapezoid between the raw point
at act `0` and the evolved point at act `1`. No AUC is emitted for an axis
whose coordinate is unknown.

## 11. Failure Semantics

An official illegal-action (`IA`) result and a 2-second per-decision
time-limit (`TLE`) result are valid losses with their own termination types.
A legal in-game loss is also valid. Harness failures are invalid results:

- agent failed to start;
- malformed protocol;
- premature process exit;
- 600-second whole-match controller timeout;
- controller crash;
- unreadable workspace;
- provider failure;
- baseline smoke-test failure.

Every available artifact is retained after failure. An evaluation containing
any invalid or missing case has `evaluation_status=incomplete`; aggregate
score, gain, and corresponding AUC point are absent. Completed cases remain
visible. Missing values are not converted to losses and reports do not
interpolate gaps.

## 12. Framework Interfaces

The Generals integration is a bounded package rather than additions to the
generic environment:

```text
agentbench_frame.generals.assets
agentbench_frame.generals.protocol
agentbench_frame.generals.process
agentbench_frame.generals.match
agentbench_frame.generals.evaluator
agentbench_frame.generals.prompt
agentbench_frame.generals.pipeline
```

The package consumes an explicit `AgentBench` asset-root path and emits
generic `BenchmarkSpec`, `GameResult`, budget, version, and event objects.
Generic tracking and provider code receives only targeted fixes required by
these published interfaces.

The CLI exposes:

```text
agentbench generals prepare
agentbench generals eval
agentbench generals iterate
```

`prepare` validates assets and builds the C++ opponent. `eval` evaluates an
existing baseline workspace. `iterate` runs the complete one-act pilot.

## 13. Verification

Automated verification covers:

1. deterministic official initialization for every frozen seed;
2. SDK framing with fragmented reads, oversized packets, malformed UTF-8,
   missing end-turn, and premature EOF;
3. legal and illegal command handling against official logic;
4. round ordering and seat swaps;
5. timeout, process cleanup, and log truncation;
6. baseline rule determinism and protocol compatibility;
7. smoke startup for all three frozen opponents;
8. exact generation of 18 unique frozen evaluation cases;
9. complete and incomplete score semantics;
10. disjoint learning and evaluation seeds;
11. prompt exclusion of evaluation artifacts and opponent source;
12. fake-provider `v0 → v1` integration with manifests and diff;
13. raw Codex JSONL parsing and unknown-usage behavior;
14. events, summary, budget, quality, and local-report contracts;
15. a real Codex live test guarded by an explicit environment marker.

Completion requires the full unit/integration suite, all three opponent smoke
tests, one fake-provider closed loop, and one real Codex one-act run. The
real run may produce lower performance; completion concerns evidence
integrity and reproducibility, not positive gain.

## 14. Explicit Non-Goals

- Docker or a hardened untrusted-code sandbox;
- multi-act learning curves;
- Claude Code execution;
- RL training;
- all 31 historical algorithms;
- merging the full CI dashboard branch;
- proving equivalence between official logic and `GeneralsEnv`;
- strict policy KL without a complete normalized macro-action distribution.
