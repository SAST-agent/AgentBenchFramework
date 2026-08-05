# Codex-Orchestrated DOTO Benchmark Design

## Purpose

Replace the framework-owned DOTO LLM loop with Codex-directed iteration while
preserving the deterministic, auditable parts of the benchmark. Codex learns
the game and workflow through composable Skills, edits one candidate policy,
and chooses which atomic framework commands to invoke. The framework remains
responsible for official execution, evidence capture, state transitions,
scoring, strict deterministic KL status, and result persistence.

The benchmark measures how many held-out historical human policy snapshots a
Codex-designed agent defeats. It does not measure the quality of a custom
Chat Completions loop.

## Goals

1. Preserve an end-to-end HL iteration closure: official game, match, replay,
   diagnosis, candidate modification, immutable version, reevaluation, and
   traceable logs.
2. Preserve and improve human-authored game/replay guidance, then validate it
   against an actual replay fixture.
3. Preserve the observation, macro-action, action-mask, support, and terminal
   definitions.
4. Preserve per-episode and per-iteration strict deterministic KL status. A
   missing or undefined value must retain its reason and must not be replaced
   by a proxy metric.
5. Produce version-aligned score-versus-iteration and IG-versus-iteration
   curves with at least one successful policy update. Failed, missing, and
   incomplete points remain visible.
6. Evaluate every iteration against the complete training pool and evaluate
   the selected final candidate exactly once against a sealed test pool.
7. Store complete authoritative DOTO evidence in a dedicated `DotoResults`
   repository and export a compatible projection to `AgentBenchResults`.

## Non-goals

- The framework does not call an LLM API, construct messages, parse model
  proposals, or decide when Codex should stop.
- The framework does not impose iteration, rollout, token, or wall-time
  budgets.
- The framework does not decide which successful candidate Codex should use
  as the parent of a later iteration.
- The framework does not modify `AgentBenchResults` aggregation or reporting
  code.
- The benchmark does not claim that historical policy snapshots correspond
  one-to-one with distinct natural persons.

## Architecture

The system has three responsibility layers:

```text
Codex
  -> composable DOTO Skills
    -> atomic AgentBenchFramework CLI commands
      -> official server/SDK, build, match, replay, IG, state and persistence
        -> DotoResults authoritative Run
          -> AgentBenchResults compatibility projection
```

Codex owns reasoning and orchestration. Framework commands own facts and state.
Codex must not directly write authoritative summaries, curve points, episode
results, or state-transition files.

### Retained framework components

- Official server, map, SDK, provenance, and integrity checks.
- Native policy build and process lifecycle.
- Historical population import, validation, build, and protocol smoke tests.
- Official-protocol match execution, replay ZIP, exact trace, stderr, and
  component hashes.
- Replay parsing and semantic summaries.
- Decision-space parsing, canonical actions, action masks, and termination.
- Strict deterministic KL/IG comparison and aggregation.
- Candidate-oriented scoring and curve construction.

### Removed framework components

- `doto/llm_client.py`
- `doto/loop_config.py`
- `doto/loop.py`
- The `doto loop` CLI command.
- OpenAI-compatible endpoint, model, API-key, message, streaming, reasoning,
  token-budget, and proposal-response configuration.
- `examples/doto-loop.toml` and documentation that instructs the framework to
  drive an LLM.

### Refactored and new components

- Refactor `run_store.py` into an atomic Run/iteration state store.
- Extend the CLI with Run and iteration lifecycle commands.
- Add a parallel evaluation scheduler with adaptive CPU-target control.
- Replace the monolithic harness Skill with four composable Skills.
- Add a versioned full-snapshot human population manifest.
- Add a new independent `DotoResults` repository with schema validation,
  aggregation, and DOTO-specific reporting.
- Add an atomic projection exporter targeting the unchanged
  `AgentBenchResults` repository.

## Human Policy Population

### Identity and provenance

The source corpus manifest contains 43 historical policy directories. Several
directories have identical `playerAI.cpp` files but different auxiliary C++
sources, such as attack, map, movement, sampler, or common logic. Therefore a
policy is identified by the SHA-256 of its complete buildable directory, not
by `playerAI.cpp` alone.

All 43 complete directory snapshots have distinct hashes. The authoritative
manifest preserves:

- stable policy ID;
- complete snapshot SHA-256;
- original name and path;
- origin repository;
- original split;
- leaderboard marker;
- enabled/build/smoke status;
- aliases, if a future corpus introduces true full-snapshot duplicates.

The metric is named `defeated_human_policy_count`, not `defeated_human_count`.

### Train/test split

The benchmark freezes the existing chronological split as follows:

- training pool: the 15 original `train` snapshots;
- sealed test pool: the 27 original `validation` snapshots plus the one
  original `test` snapshot, for 28 total.

Changing the pool, hashes, seed, seats, or official assets creates a new
benchmark schema version. It does not mutate an existing version.

### Isolation

Codex may repeatedly build, battle, and inspect the training policies. It must
not read test sources, invoke test policies during iteration, or receive test
results before selecting the final candidate.

Formal benchmark environments exclude the original `AgentBench` test corpus
from the Codex-readable workspace. Test sources are built before the Codex
session into a sealed evaluator-owned pool. The finalization command refers to
the pool by benchmark version and does not reveal source or executable paths.
The sealed evaluator verifies every expected full-snapshot and executable hash.

## Atomic CLI

The low-level commands remain independently usable:

```text
doto build
doto match
doto replay
doto ig
```

They accept optional Run attachment parameters so that a diagnostic invocation
can be recorded without Codex moving artifacts or editing events manually:

```text
--run-dir PATH --iteration N --purpose diagnostic|formal
```

The lifecycle surface is:

```text
doto run init
doto run status
doto iteration begin
doto iteration build
doto iteration evaluate
doto iteration compare
doto iteration close
doto run finalize
```

Every mutating command is atomic, returns machine-readable JSON, uses a nonzero
exit status for invalid transitions, and preserves failure evidence. Completed
steps cannot be silently overwritten. Querying status is read-only and safe.

### State machine

```text
created
  -> iteration_open
  -> candidate_built | build_failed
  -> training_evaluated | evaluation_incomplete
  -> ig_recorded | ig_missing
  -> iteration_closed
  -> ... another iteration
  -> final_test_started
  -> finalized
```

An iteration stores an explicit `parent_iteration`. Codex may branch from any
successfully built and closed version. A build failure or incomplete evaluation
can be closed as a null point but cannot become a parent. The framework does
not implement implicit accept/reject or hill climbing.

`run finalize --candidate-iteration N` requires a closed, complete candidate
chosen using training evidence only. It starts the sealed test exactly once and
seals the Run when the test finishes. A finalized Run cannot create iterations
or rerun the test.

### Formal and diagnostic evaluations

Every iteration's formal score covers all 15 training policies, seed `11`, and
both candidate seats: 30 matches. This fixed matrix alone contributes the
score-versus-iteration point. Codex may run additional training matches marked
`diagnostic`; they are recorded but never mixed into the formal score.

Finalization covers all 28 sealed policies, seed `11`, and both candidate
seats: 56 matches. Test evidence is not returned until the Run is sealed and
cannot be used for another candidate update in that Run.

## Parallel Evaluation

Each `(opponent, seed, candidate_seat)` tuple is an isolated task with its own
official server, two AI processes, replay, trace, stderr files, and unique tag.
The parent process schedules tasks and performs deterministic aggregation in
manifest order rather than completion order.

The default scheduler targets 70 percent aggregate CPU use:

- sample the complete evaluator process tree periodically;
- increase concurrency gradually while rolling CPU is below approximately 65
  percent;
- hold concurrency between approximately 65 and 75 percent;
- stop launching new tasks above approximately 75 percent;
- reduce the later concurrency ceiling after sustained overload;
- never suspend or terminate a valid in-progress formal match merely to meet
  the target;
- support `--workers` as a hard operator-supplied ceiling for memory and file
  descriptor safety.

The Run records target CPU, observed mean/peak CPU, worker history, the hard
ceiling, per-match duration, and scheduling failures. One task failure does not
cancel unrelated tasks. It makes the affected formal matrix incomplete.

## Skills

### `doto-benchmark-run`

The top-level workflow Skill explains the benchmark objective, lifecycle
commands, state invariants, formal versus diagnostic evidence, train/test
isolation, failure preservation, parent selection, and finalization. It tells
Codex when the companion Skills are mandatory. It contains no LLM API loop.

### `doto-game-rules`

The authoritative domain Skill documents map and factions, five controlled
humans, crystals, kills, bonuses, movement, fireball, meteor, flash, revival,
invincibility, frame timing, observation visibility, scoring, and terminal
rules. It includes the observation, macro-action, mask, continuous support, and
terminal definitions and cross-references official configuration sources.

### `doto-agent-authoring`

The C++ development Skill documents the complete `playerAI.cpp` contract,
fixed SDK APIs, local-slot/global-ID mapping, state and numeric safety, build
workflow, and common native-policy failures. It prohibits modification of the
SDK, server, map, evaluator, or test pool.

### `doto-replay-reader`

The replay Skill distinguishes authoritative ZIP events from the exact
observation/action trace. It documents numeric schemas, event semantics,
joining rules, diagnostic workflow, process-versus-strategy failures, strict
KL interpretation, and common misreadings. A historical short replay fixture
must validate its field descriptions and example commands.

### Skill provenance

Run initialization copies the exact contents and hashes of all four Skills into
the authoritative Run. The required composition is:

```text
start/design candidate -> game-rules + agent-authoring
diagnose match          -> replay-reader
advance/close Run       -> benchmark-run
```

## Score and IG Contracts

### Training score

For every policy, the candidate score difference is computed in candidate
orientation for both seats. A policy is defeated only if both matches complete
normally, both replays parse, and the mean candidate score difference is
strictly positive.

An iteration's formal `evo` score is valid only when all 30 expected matches
complete normally and parse successfully. An incomplete matrix has `evo=null`
and retains all measured episode values, completion rate, and exact reasons.
The curve includes every closed iteration, including build, evaluation, and IG
failures.

### Final test score

For each held-out policy, all expected matches must complete normally and the
mean candidate score difference must be greater than zero. Draws, incomplete
matches, timeouts, crashes, corrupt replays, and missing evidence do not count
as defeated policies.

The primary final metric is:

```text
defeated_human_policy_count / 28
```

The final result also retains each policy's score difference, win rate,
completion rate, episode evidence, and failure reason.

### Strict deterministic KL status

IG comparison uses real observations from the current iteration's formal
evaluation traces and invokes the parent and current executable on each same
observation.

- identical canonical deterministic joint action: KL `0`, included in
  `unchanged_ratio`;
- different canonical deterministic joint action: strict KL divergence,
  included in `infinite_ratio`;
- invalid output, timeout, crash, missing alignment, or unsupported action:
  included in `missing_ratio` with an exact reason;
- `finite_kl_mean` averages only genuine finite values and is otherwise null.

No coordinate distance, action-change rate, score gain, or behavior feature may
be labeled KL or IG. JSON artifacts use ratios and null; they never serialize
`Infinity` or `NaN` numeric literals.

Iteration zero has no parent and records IG as not applicable. Every later IG
artifact records parent/current hashes and exact trace provenance.

## Authoritative `DotoResults` Repository

`DotoResults` is an independent repository and the authoritative DOTO data
source:

```text
DotoResults/
  runs/23_doto/<agent>/<run_id>/
    run.toml
    state.json
    events.jsonl
    summary.json
    score_curve.json
    ig_curve.json
    final_test.json
    projection.json
    pools/
    skills/
    iterations/iteration-NNNN/
      iteration.json
      playerAI.cpp
      analysis.md
      build/
      evaluation/matrix.json
      evaluation/episodes/
      diagnostics/
      ig/
  schemas/
  scripts/validate.py
  scripts/aggregate.py
  scripts/build_report.py
  reports/
  .github/workflows/
```

The complete evidence includes candidate sources, build reports and stderr,
episode JSON, official replay ZIPs, exact trace JSONL, per-episode IG, Skills,
pool manifests, hashes, and curves. Rebuildable executables and interpreter
caches are not committed.

The validator fails closed on invalid state, source/version mismatch, broken
parent links, incomplete required matrices, wrong seed/seat, asset mismatch,
invalid replay, curve misalignment, repeated final test, or an invalid JSON
number. Validation happens before a Run may be marked authoritative and sealed.

`projection.json` is an operational sidecar rather than gameplay evidence. It
is excluded from the sealed Run-content hash and is the only Run file that may
change after sealing. It records pending, exported, or failed projection
attempts and permits a safe export retry without changing benchmark results.

## `AgentBenchResults` Projection

The existing `AgentBenchResults` framework remains unchanged. After a
`DotoResults` Run is sealed and validates successfully, an atomic exporter
creates:

```text
AgentBenchResults/runs/23_doto/<agent>/<run_id>/
  run.toml
  summary.json
  score_curve.json
  ig_curve.json
  doto_results_ref.json
```

The projection conforms to the current permissive contract:

- `[run]` metadata in `run.toml`;
- valid JSON in `summary.json`;
- scalar `wall_hours`, `total_steps`, and `win_rate` compatibility fields;
- no `Infinity` or `NaN` values.

`summary.json` may additionally contain DOTO metrics even though the current
AgentBenchResults report ignores them. `doto_results_ref.json` records the
authoritative Run ID and path, schema version, authoritative summary hash, Run
manifest hash, and export time.

Projection files are derived from the sealed authoritative Run and never
compute independent metrics. Export writes to a temporary directory, validates
the projection, and atomically renames it. An existing projection cannot be
silently overwritten. Projection failure leaves the authoritative Run valid,
records `projection_failed` in `projection.json`, and supports a safe retry.

## Result Fields

Each iteration records at least:

- iteration and explicit parent;
- source and executable hashes;
- Codex-authored modification analysis;
- build state and logs;
- complete expected training matrix and per-episode artifacts;
- adaptive parallelism metadata;
- mean score difference, win rate, completion rate, and defeated training
  policy count;
- aggregated and per-episode strict KL status;
- compile, battle, and total wall time;
- complete, failed, or incomplete status and reason.

The final summary records at least:

- Run, schema, pool, Skill, server, map, SDK, and selected candidate identities;
- total iterations, builds, matches, frames, and wall time;
- training score history and IG history;
- best complete training iteration and selected final iteration;
- `defeated_human_policy_count`, test policy count, final mean score difference,
  final win rate, and completion rate;
- per-stage failure counts;
- token usage as null/unavailable unless supplied by a trustworthy external
  Codex execution record. It is never estimated from text length.

Projection state and the exported reference hash live in the post-seal
`projection.json` sidecar, not in the immutable benchmark summary.

## Error Handling and Recovery

- Every attempt writes an event before and after execution.
- Partial subprocess output, stderr, trace, and replay files remain attached to
  a failed attempt.
- A failed build cannot reuse a stale executable.
- Formal episode retries are new attempts and never erase consumed attempts. A
  matrix cell closes on its first valid complete attempt; after that, duplicate
  attempts for the cell are forbidden.
- An interrupted parallel evaluation can resume only missing matrix cells after
  verifying all existing cell hashes.
- Closing an incomplete iteration creates a visible null curve point.
- Final test start is durable before any test process launches. An interruption
  may resume missing cells for the same selected candidate but cannot select a
  new candidate or start a second test.
- Atomic writes use temporary files in the destination filesystem followed by
  replacement.

## Verification

Automated verification covers:

1. all 43 complete snapshot hashes and the frozen 15/28 split;
2. absence of test sources and paths from the Codex-readable workspace;
3. population build and native protocol smoke;
4. valid and invalid Run state transitions, idempotent status, and overwrite
   refusal;
5. isolated parallel outputs, deterministic manifest-order aggregation, task
   failure isolation, and adaptive CPU scheduling decisions;
6. existing build, process, protocol, match, and replay behavior;
7. actual short replay schema and event interpretation;
8. observation, macro-action, mask, support, and terminal contracts;
9. strict KL `0`/infinite/missing behavior and JSON-safe representation;
10. version-aligned curves containing failed and incomplete points;
11. test execution exactly once and resume-only recovery;
12. authoritative DotoResults schema validation and atomic projection;
13. projection acceptance by the unchanged AgentBenchResults aggregator;
14. removal of all framework LLM/API configuration and entry points;
15. a fake-server end-to-end Codex-style Run with a successful child update;
16. an opt-in official-duration acceptance covering a genuine parent/child
    iteration, the full 30-match training matrix, the full 56-match final test,
    both curves, DotoResults validation, and AgentBenchResults projection.

## Migration Sequence

1. Freeze and verify the 43-policy full-snapshot manifest and sealed split.
2. Add atomic Run state and lifecycle commands while the old loop still exists.
3. Add parallel full-matrix evaluation and recovery.
4. Refactor scoring and IG into lifecycle operations.
5. Create and validate the four Skills.
6. Create the independent DotoResults repository and validator.
7. Add the AgentBenchResults projection exporter and compatibility test.
8. Run a fake-server end-to-end closure.
9. Remove the old loop, LLM client/config, examples, and obsolete docs/tests.
10. Run official-duration acceptance and publish the first sealed Run.

## Acceptance Criteria

The migration is complete when:

- Codex can start from the Skills and atomic CLI only, without framework LLM
  code, and produce at least one valid strategy update;
- every formal iteration covers all 15 training snapshots in both seats;
- the chosen final candidate is evaluated exactly once against all 28 sealed
  snapshots in both seats;
- score and IG curves are version aligned and preserve failures;
- strict KL status never uses a proxy metric;
- the full Run validates in DotoResults;
- the unchanged AgentBenchResults aggregator accepts the exported projection;
- all automated tests and the opt-in official-duration acceptance pass.
