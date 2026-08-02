# DOTO benchmark harness

## Scope and prerequisites

This harness keeps the 23rd DOTO official server/map and original native
`playerAI.cpp` protocol. It requires Python 3.11+, `uv`, `g++`, `make`, and ZIP
support. The only command surface is:

```bash
uv run python -m agentbench_frame.doto <build|match|replay|ig|loop>
```

The policy maintained by the evaluated LLM is exactly one complete
`playerAI.cpp`; the harness supplies the fixed SDK. See
[`doto-harness`](../skills/doto-harness/SKILL.md),
[`doto-replay-reader`](../skills/doto-replay-reader/SKILL.md), and the
[design](superpowers/specs/2026-08-02-doto-benchmark-harness-design.md).

## Build, population, match, and replay

```bash
uv run python -m agentbench_frame.doto build \
  --player-ai examples/doto-initial-playerAI.cpp \
  --output-dir agentbench_data/builds/23_doto/candidate

uv run python -m agentbench_frame.doto build \
  --population-manifest src/agentbench_frame/doto/population.toml \
  --corpus-root ../AgentBench/backend_sources/corpus/23_doto \
  --output-dir agentbench_data/population/23_doto

uv run python -m agentbench_frame.doto match \
  --agent0 agentbench_data/builds/23_doto/candidate/main.out \
  --agent1 agentbench_data/population/23_doto/official-sample/main.out \
  --seed 11 --tag candidate-seat0 \
  --output-dir agentbench_data/replays/23_doto

uv run python -m agentbench_frame.doto replay \
  --path agentbench_data/replays/23_doto/candidate-seat0_seed11.zip \
  --jsonl agentbench_data/replays/23_doto/candidate-seat0.events.jsonl
```

Swap `--agent0` and `--agent1` with the same seed for seat balance. A formal
match is valid only with normal final frame, two scores, no error, a parseable
replay, `realtime_scale=1.0`, and recorded server/map/policy hashes. Population
entries become `ready` only after hash verification, build, and native protocol
smoke; all failures remain in `population-build.json`.

## Strict deterministic KL/IG

```bash
uv run python -m agentbench_frame.doto ig \
  --trace agentbench_data/replays/23_doto/candidate-seat0_seed11.trace.jsonl \
  --old agentbench_data/builds/23_doto/old/main.out \
  --new agentbench_data/builds/23_doto/candidate/main.out \
  --faction 0 --iteration 1 --old-version v0 --new-version v1 \
  --output-dir agentbench_data/ig/23_doto
```

On the same observations, equal deterministic joint actions give KL 0;
different actions give infinite KL; crashes, invalid actions, timeouts, and
misalignment are missing with reasons. The curve reports
`unchanged_ratio`, `infinite_ratio`, `missing_ratio`, and `finite_kl_mean`.
Distance or score gain is never relabeled as KL.

## LLM loop and visible context

Copy `examples/doto-loop.toml`, update model/opponent paths and budgets, then:

```bash
export OPENAI_API_KEY='<secret>'
uv run python -m agentbench_frame.doto loop \
  --config examples/doto-loop.toml \
  --data-dir ../AgentBenchResults
```

Chat Completions SSE is default. `max_context_tokens` defaults to 1,000,000;
no `max_tokens` is sent unless configured. The API key is read from the named
environment variable and never saved. The model receives the two exact Skills,
fixed SDK reference, current accepted source, immediately previous metrics,
deterministically budgeted observation/action evidence, and cumulative budget.
It must return one JSON object with nonempty `analysis` and complete
`player_ai_cpp` strings.

Budgets are cumulative across the Run: iterations, builds, rollouts, episode
reads, frame reads, provider tokens, context tokens, compile/battle/API time,
and wall time. Failed API, build, protocol, or evaluation attempts remain in
their iteration directories and curves; rejected code is not promoted.

## Results and interpretation

Output is written directly to:

```text
AgentBenchResults/runs/23_doto/<agent>/<run_id>/
├── run.toml, config.json, events.jsonl, summary.json
├── score_curve.json, ig_curve.json, skills/
└── iterations/iteration-NNNN/
```

Score is mean seat-balanced candidate score difference. Each point preserves
raw, evo, gain (`evo-raw`), win/completion rates, source version, budget state,
and failure/null status. AUC is reported independently over iteration,
rollouts, tokens, episode/frame reads, and wall time using measured points only.
The run directory is already the AgentBenchResults export; commit that directory
in the Results repository after checking that no credential is present.
