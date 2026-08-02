# DOTO Benchmark Harness Design

## 1. Goal and scope

Build a reproducible benchmark harness for the 23rd DOTO game from remote
`main`. The evaluated LLM maintains the original C++ contestant file
`playerAI.cpp`. The harness performs the complete loop:

1. save and compile the current C++ policy;
2. run the official Python server against a fixed opponent in both factions;
3. save the official replay ZIP and a readable protocol trace;
4. give budget-limited replay evidence and the current policy to an
   OpenAI-compatible LLM;
5. save and compile the returned `playerAI.cpp` as a new immutable version;
6. evaluate it on the aligned seed/seat set;
7. save score, strict information-gain status, budgets, failures, and curves in
   AgentBenchResults format.

This spec covers the original five benchmark requirements: one high-level
iteration closure, a hand-authored replay-reading Skill with a real validation,
the decision-space definition, strict information-gain recording, and aligned
score/IG curves after at least one valid policy update.

Leaderboard execution, opponent-order ablations, reward/context ablations, and
policy-editing ablations are future consumers of this loop. They are not
separate implementations in this change.

## 2. Non-negotiable compatibility decisions

### 2.1 Upstream base

Development occurs on `feature/doto`, forked directly from `origin/main` at
`98d5b0a`. No Miracle implementation commits are ancestors of this branch.

### 2.2 Official game logic

Vendor the public Arena DOTO server files and map from:

```text
AgentBench/backend_sources/corpus/23_doto/logic/
  public_Arena-Doto-AI/Arena-Doto-AI-master/server/
```

The vendored rule implementation and `Maps/0.json` remain source-equivalent to
the corpus. Compatibility-only launch behavior belongs in harness wrappers, not
in edited copies of game functions. Record source provenance and SHA-256 hashes
in the package.

The official map lasts 300 seconds at 20 FPS, and the server intentionally
sleeps 50 ms per frame. A benchmark episode uses the official realtime behavior
(`realtime_scale=1.0`). Tests may use a separately named short fixture or a
protocol fake. Test-only results must never be exported as benchmark scores.

### 2.3 Contestant interface

The only LLM-maintained policy file is a complete `playerAI.cpp`. A fixed copy
of the original client SDK supplies `main.cpp`, `playerAI.h`, `logic.*`,
`geometry.*`, `const.h`, and jsoncpp. The harness copies the SDK into an
isolated build directory, replaces only `playerAI.cpp`, and invokes the fixed
build recipe.

Historical population policies may carry additional source files. They are
compiled by a population builder from their complete original directories and
are never rewritten into the single-file candidate contract.

## 3. User-facing interface

The package exposes exactly one entry point:

```bash
uv run python -m agentbench_frame.doto <build|match|replay|ig|loop>
```

### 3.1 Build

```bash
uv run python -m agentbench_frame.doto build \
  --player-ai path/to/playerAI.cpp \
  --output-dir agentbench_data/builds/23_doto/candidate
```

The command creates a clean SDK build tree, compiles `main.out`, and writes
`build.json` containing source hash, compiler command, timestamps, exit code,
stdout, and stderr. It prints JSON with the executable path. A nonzero compiler
exit is a build failure and no stale executable may be reported as successful.

### 3.2 Match

```bash
uv run python -m agentbench_frame.doto match \
  --agent0 path/to/agent0/main.out \
  --agent1 path/to/agent1/main.out \
  --seed 11 \
  --output-dir agentbench_data/replays/23_doto \
  --tag candidate-vs-sample
```

`agent0` controls faction 0 and `agent1` controls faction 1. The host starts the
official server and both native AI processes, routes framed messages without
changing their JSON bodies, and records both directions in sequence.

Success requires all of the following:

- server reaches `frame=-1`;
- final score vector has two numeric values;
- replay ZIP exists and can be parsed;
- no process/protocol timeout or crash occurred.

The JSON result contains seed, factions, winner (`0`, `1`, or `null` for draw),
scores, score difference, frame count, duration, termination status, errors,
replay path, and trace path. Formal comparisons swap factions with the same
seed.

### 3.3 Replay

```bash
uv run python -m agentbench_frame.doto replay \
  --path path/to/replay.zip \
  --jsonl path/to/events.jsonl
```

The parser reads the replay ZIP without executing embedded text. Historical
list-valued fields use `ast.literal_eval`, never `eval`. It returns the final
scores, winner/draw, last frame, event counts, and candidate-oriented combat,
goal, bonus, and death metrics when a faction is provided.

### 3.4 IG

```bash
uv run python -m agentbench_frame.doto ig \
  --trace path/to/match.trace.jsonl \
  --old path/to/old/main.out \
  --new path/to/new/main.out \
  --faction 0 \
  --iteration 1 \
  --output-dir agentbench_data/ig/23_doto
```

The harness feeds the same initialization and observation sequence to isolated
old/new native processes, captures their actions, canonicalizes them, checks
schema/observation legality, and writes per-frame comparisons plus
`ig_curve.json`.

### 3.5 Loop

```bash
export OPENAI_API_KEY='<key>'
uv run python -m agentbench_frame.doto loop \
  --config examples/doto-loop.toml \
  --data-dir ../AgentBenchResults
```

The first implementation uses OpenAI-compatible
`POST /v1/chat/completions`. SSE streaming defaults to true. `max_tokens` is
absent unless explicitly configured. `max_context_tokens` defaults to
1,000,000 and is checked from provider-reported prompt plus completion usage.

The model returns exactly:

```json
{
  "analysis": "diagnosis and update rationale",
  "player_ai_cpp": "complete playerAI.cpp source"
}
```

Patches, shell commands, multiple files, empty fields, compilation failures,
and protocol-invalid binaries are rejected as candidate updates. Rejection does
not replace the last accepted policy.

## 4. Native process architecture

### 4.1 Framing

The DOTO server emits a four-byte big-endian length, four-byte message type,
four-byte target/faction for targeted messages, followed by JSON. Native AIs
consume a four-byte length plus JSON and emit the same. The host implements
bounded frame reads, per-process deadlines, maximum frame size, and process
group cleanup.

Trace rows have stable fields:

```json
{
  "seq": 42,
  "timestamp": 0.512,
  "kind": "observation|action|final",
  "faction": 0,
  "frame": 17,
  "payload": {}
}
```

Raw JSON bodies are preserved. Parsed convenience fields may be added without
removing raw payloads.

### 4.2 Determinism

The launcher seeds Python `random` before importing `Arguments.py`, so map and
bonus timing randomness are reproducible without editing rule functions. The
seed, map hash, server source hash, both policy hashes, compiler identity, and
realtime scale are recorded in every episode.

### 4.3 Failure handling

Every started process is placed in its own process group. On completion or
failure the harness terminates the complete group, escalates to SIGKILL after a
bounded grace period, closes pipes, and preserves partial trace and stderr.

Failure stages are explicit: `build`, `server_start`, `protocol`, `ai_timeout`,
`server_timeout`, `process_exit`, `replay_missing`, `replay_parse`,
`llm_request`, `proposal_json`, `source_validation`, and `budget`.

## 5. Replay Reader Skill

Create `skills/doto-replay-reader/SKILL.md` by adapting and validating the
human-authored corpus document rather than asking the LLM to infer the game.
The Skill specifies:

- 320×320 map coordinates, walls, two factions, and five humans per faction;
- faction ownership rule `human_id % faction_number`;
- 20 FPS, 300-second termination, death/revival, and invincibility;
- crystal pickup/drop/delivery, target areas, bonus generation, and scoring;
- movement, fireball, meteor, flash, range/cooldown/use limits, and no-op values;
- initialization, ordinary, and final frame schemas;
- all human/fireball/meteor/ball field indices;
- replay event types 1 through 12 and their argument meanings;
- the distinction among replay ZIP, replay frame JSON, and harness trace;
- common misreadings, including stringified lists, absolute target coordinates,
  faction-relative human ordering, dead humans, held-ball flash prohibition,
  and draw handling;
- a step-by-step self-consistency validation joining actions, following
  observations, events, and final score.

At least one integration fixture must demonstrate that the parser's final score
and event counts agree with a real server-produced replay.

## 6. Decision space

### 6.1 Observation

An initialization observation is:

```text
InitObservation(frame=0, map_id, faction)
```

An ordinary observation is the normalized form of:

```text
FrameObservation(
  frame,
  humans,
  fireballs,
  meteors,
  balls,
  scores,
  bonus,
  faction,
  map_metadata
)
```

The parser converts historical stringified lists to typed arrays while
retaining the raw message for audit.

### 6.2 Macro-action

One policy decision controls the faction's five humans jointly:

```text
DotoAction(
  move:    tuple[PointOrNoop, 5],
  shoot:   tuple[PointOrNoop, 5],
  meteor:  tuple[PointOrNoop, 5],
  flash:   tuple[bool, 5]
)
```

`PointOrNoop` is `(-1, -1)` or a finite `(x, y)` coordinate. Canonicalization
normalizes numeric representation and rejects NaN, infinity, wrong lengths,
wrong types, or nonzero protocol flags.

### 6.3 Action mask and termination

The mask reports schema legality and observation-known constraints per human:

- dead humans can only no-op;
- shoot requires a living human, expired fireball cooldown, and a non-self
  in-map aim direction;
- meteor requires a living human, remaining uses, expired cooldown, in-map
  target, and distance at most 30;
- flash requires a living human, remaining uses, expired cooldown, no held
  crystal, legal non-wall destination, and distance at most 20;
- ordinary movement requires a living human, legal non-wall destination, and
  distance at most 0.6;
- no-op is always schema-valid.

The action mask is explanatory. The official server remains the final legality
arbiter because asynchronous timing and exact geometry belong to the rules.

Terminal observations are `frame=-1`. Harness termination also occurs on
timeout, process exit, protocol violation, or missing/corrupt replay, all marked
non-normal.

### 6.4 Support set and strict KL

DOTO has a continuous action space; the harness must not pretend it can
enumerate all legal coordinate combinations.

- The theoretical support is the observation-dependent measurable set of legal
  joint `DotoAction` values.
- For one old/new comparison, the computational support is the finite union of
  their two canonical actions on that exact observation.
- Each deterministic policy is represented as a Dirac measure on this support.

Therefore:

- identical canonical joint actions: strict KL is `0`;
- different canonical joint actions: strict KL is infinite;
- missing output, process failure, misalignment, invalid schema, or action
  outside the observation-known mask: comparison status is `missing` with a
  reason;
- `finite_kl_mean` averages only actual finite KL values and is otherwise
  `null`.

Store `unchanged_ratio`, `infinite_ratio`, `missing_ratio`, and
`finite_kl_mean` per episode and iteration. Coordinate distance, action-change
rate, score gain, and behavioral features may be diagnostic fields but may not
be labeled KL or IG.

## 7. LLM context and iteration semantics

Every update request contains:

1. a system contract requiring one JSON response with complete C++ source;
2. the complete DOTO Harness Skill;
3. the complete DOTO Replay Reader Skill;
4. a concise fixed-SDK interface reference;
5. the current accepted `playerAI.cpp`;
6. the immediately previous iteration status and metrics;
7. budget-limited structured episodes and frames, including observations,
   candidate actions, following events, score changes, and errors;
8. cumulative budget state.

Replay selection is deterministic and logged. `max_episode_reads` and
`max_frame_reads` are cumulative Run budgets, not per-iteration allowances.
The request file saves the exact messages and non-secret request body. The API
key is read only from the configured environment variable and is never written.

Iteration 0 is the compiled/evaluated baseline. Each later index is one update
attempt. Failed attempts remain in `iterations/iteration-NNNN` and in curves.
Only a source that compiles and passes a protocol smoke check becomes the next
accepted version. A completed but lower-scoring version is still accepted and
recorded; the harness measures evolution rather than enforcing monotonic score.

## 8. Score, budgets, curves, and Results

For each candidate faction, preserve:

- raw candidate and opponent scores;
- `score_diff = candidate_score - opponent_score`;
- win/draw/loss/error and completion status;
- damage dealt/taken;
- candidate/opponent deaths;
- goals and bonuses.

The primary policy score is mean seat-balanced `score_diff` over the aligned
seed/opponent set. Also report win rate with draw value 0.5 and completion rate.

Version-aligned `score_curve.json` points contain `raw`, `evo`,
`gain=evo-raw`, score/win/completion metrics, version hash, status, rollouts,
tokens, episode reads, frame reads, compile time, battle time, API time, and
wall time. AUC is computed independently over iteration, rollout, token,
episode-read, frame-read, and wall-time axes using only valid measured points;
missing points and reasons remain in the curve.

`ig_curve.json` contains aligned strict-KL status points, including missing
iterations. No value is interpolated across failures.

Budget limits include:

- update attempts;
- build attempts;
- rollouts;
- episode reads;
- frame reads;
- provider-reported cumulative tokens;
- provider-reported per-request context tokens;
- compile, battle, API, and total wall time.

The standard output tree is:

```text
AgentBenchResults/runs/23_doto/<agent>/<run_id>/
├── run.toml
├── config.json
├── summary.json
├── events.jsonl
├── score_curve.json
├── ig_curve.json
├── skills/
└── iterations/
    ├── iteration-0000/
    │   ├── playerAI.cpp
    │   ├── build.json
    │   ├── main.out
    │   └── episodes/
    └── iteration-0001/
        ├── llm_request.json
        ├── llm_response.json
        ├── candidate.playerAI.cpp
        ├── playerAI.cpp
        ├── build.json
        ├── main.out
        ├── iteration.json
        ├── episodes/
        └── ig/
```

Generated binaries, build trees, replay ZIPs, traces, API responses, Run data,
and credentials are runtime artifacts and are not committed to source control.

## 9. Population

Read the corpus `population.toml`, retain its declared split and provenance,
and compile policies in isolated directories. Record per-policy build status and
source hash. An opponent is eligible only when its original source compiles and
passes a protocol smoke match. Failed historical policies remain in the build
report with reasons rather than disappearing.

The first loop example uses one known-good fixed opponent. Population-wide
leaderboards and opponent-order ablations remain configuration-level follow-up
work once the core loop is validated.

## 10. Test and acceptance strategy

Tests are layered so normal CI does not require a 300-second match:

1. framing and process cleanup tests use tiny fake server/AI programs;
2. observation/action parsing and masks use fixed official-format fixtures;
3. compiler tests build a minimal valid `playerAI.cpp` and preserve failures;
4. replay tests parse a checked-in small real replay fixture;
5. match smoke tests use a clearly marked short test map outside benchmark
   Results;
6. IG tests drive old/new deterministic binaries on the same trace and verify
   zero/infinite/missing cases;
7. loop tests use a local mock Chat Completions SSE server and short match
   runner while exercising real source validation, storage, curves, and CLI;
8. an opt-in manual acceptance command runs an official 300-second,
   seat-balanced episode pair and validates its exported Results.

Acceptance requires:

- one command builds a real original-interface candidate;
- one command completes a server/AI battle and saves parseable replay plus
  trace;
- the Replay Reader Skill has a real fixture-backed validation;
- observation, macro-action, mask, termination, and continuous support are
  documented and tested;
- strict zero/infinite/missing KL status is saved per episode/iteration;
- a mocked fast loop completes baseline, one accepted update, re-evaluation,
  versioned score/IG curves, and budget logs;
- a real official-duration acceptance run is documented separately and cannot
  be confused with test-only results.
