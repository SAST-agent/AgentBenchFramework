---
name: doto-harness
description: Use when building, battling, replaying, comparing, or iteratively improving an original-interface 23rd DOTO C++ playerAI.cpp policy inside AgentBenchFramework.
---

# DOTO Harness

Run all commands from the AgentBenchFramework root through the only entry point:

```bash
uv run python -m agentbench_frame.doto <build|match|replay|ig|loop>
```

## Policy ownership

Maintain exactly one complete C++ file, `playerAI.cpp`, containing:

```cpp
#include "playerAI.h"

void playerAI()
{
    // Set operations through Logic::Instance().
}
```

The harness supplies the fixed original SDK (`main.cpp`, `playerAI.h`,
`logic.*`, `geometry.*`, `const.h`, jsoncpp, and logging). Do not return a patch,
shell command, executable, header replacement, or multiple source files.

Historical population policies may use complete original source directories;
that exception applies to fixed opponents, not to the LLM-maintained candidate.

## Build

```bash
uv run python -m agentbench_frame.doto build \
  --player-ai examples/doto-initial-playerAI.cpp \
  --output-dir agentbench_data/builds/23_doto/candidate
```

The output directory contains the snapshotted `playerAI.cpp`, fixed SDK,
`build.json`, and `main.out` only on success. Inspect `exit_code`, stdout, and
stderr. A failed rebuild removes any stale `main.out` and must not be evaluated.

## Match and replay

```bash
uv run python -m agentbench_frame.doto match \
  --agent0 agentbench_data/builds/23_doto/candidate/main.out \
  --agent1 agentbench_data/builds/23_doto/sample/main.out \
  --seed 11 \
  --output-dir agentbench_data/replays/23_doto \
  --tag candidate-vs-sample
```

`agent0` is faction 0 and `agent1` is faction 1. Repeat with swapped factions
and the same seed for a comparable pair. A formal match uses the source-equivalent
official 300-second server/map and succeeds only with final frame, two scores,
parseable replay ZIP, and no process/protocol error.

The command saves official `*.zip`, exact `*.trace.jsonl`, and component stderr.
Read the ZIP with:

```bash
uv run python -m agentbench_frame.doto replay \
  --path agentbench_data/replays/23_doto/candidate-vs-sample_seed11.zip \
  --jsonl agentbench_data/replays/23_doto/candidate-vs-sample.events.jsonl
```

**REQUIRED COMPANION SKILL:** Use `doto-replay-reader` before interpreting
numeric arrays, events, actions, scores, or trace behavior.

## Decision space

An ordinary observation contains the normalized current frame, all humans,
fireballs, meteors, crystals, scores, bonuses, candidate faction, and fixed map
metadata. The native AI does not see replay-only `events`.

One deterministic macro-action jointly controls five local humans:

```text
DotoAction(move[5], shoot[5], meteor[5], flash[5])
```

Each point is a finite absolute `(x,y)` or no-op `(-1,-1)`. `flash[i]` applies
to `move[i]`. The explanatory mask checks:

- exact five-element schema and finite coordinates;
- faction-to-global-human mapping;
- alive/dead state;
- movement/meteor/flash range;
- map boundary and wall destination;
- fireball/meteor/flash cooldown and remaining uses;
- crystal-carrier flash prohibition.

Official asynchronous logic remains the final legality arbiter.

Terminal state is official `frame=-1`. Timeout, process exit, protocol violation,
or corrupt/missing replay is non-normal termination and must not become a loss.

## Strict deterministic KL status

DOTO coordinates form a continuous space. Do not invent a finite enumeration.

- Theoretical support: the observation-dependent measurable set of legal joint
  actions.
- Computational support for one comparison: the union of the old and new
  canonical joint actions on that exact observation.
- Identical actions: strict KL `0`, counted in `unchanged_ratio`.
- Different actions: strict KL infinity, counted in `infinite_ratio`.
- Invalid, missing, crashed, timed-out, or misaligned output: `missing_ratio`
  with an exact reason.
- `finite_kl_mean` averages only genuine finite KL values and is otherwise null.

Never call coordinate distance, action-change rate, score gain, or behavior
features KL/IG.

## Iteration and Results contract

The `loop` command will read an OpenAI-compatible TOML config, compile/evaluate
iteration 0, provide the current source plus budget-limited replay evidence to
the model, require one JSON object containing `analysis` and complete
`player_ai_cpp`, compile and evaluate each candidate, and save immutable
versions. SSE streaming is the default; API keys come only from the configured
environment variable.

Run output belongs under:

```text
AgentBenchResults/runs/23_doto/<agent>/<run_id>/
```

Preserve build/API/process failures and incomplete evaluations. Align every
score and IG point with iteration and source hash. Record rollouts, episode/frame
reads, provider token usage, compile/battle/API time, total wall time, raw score,
evolved score, gain, AUC, and strict KL statuses. Generated binaries, replay
ZIPs, traces, responses, temporary configs, and Results are runtime artifacts,
not repository source.
