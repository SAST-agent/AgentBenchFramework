# DOTO official-duration acceptance

This opt-in acceptance uses the unmodified official 300-second, 20 FPS map at
`realtime_scale=1.0`. Two swapped-seat episodes take about ten minutes, so they
are intentionally excluded from ordinary CI.

From `AgentBenchFramework`:

```bash
uv run python -m agentbench_frame.doto build \
  --player-ai examples/doto-initial-playerAI.cpp \
  --output-dir agentbench_data/acceptance/candidate

uv run python -m agentbench_frame.doto build \
  --population-manifest src/agentbench_frame/doto/population.toml \
  --corpus-root ../AgentBench/backend_sources/corpus/23_doto \
  --output-dir agentbench_data/acceptance/population

uv run python -m agentbench_frame.doto match \
  --agent0 agentbench_data/acceptance/candidate/main.out \
  --agent1 agentbench_data/acceptance/population/official-sample/main.out \
  --seed 11 --tag acceptance-seat0 \
  --output-dir agentbench_data/acceptance/replays

uv run python -m agentbench_frame.doto match \
  --agent0 agentbench_data/acceptance/population/official-sample/main.out \
  --agent1 agentbench_data/acceptance/candidate/main.out \
  --seed 11 --tag acceptance-seat1 \
  --output-dir agentbench_data/acceptance/replays

uv run python -m agentbench_frame.doto replay \
  --path agentbench_data/acceptance/replays/acceptance-seat0_seed11.zip
uv run python -m agentbench_frame.doto replay \
  --path agentbench_data/acceptance/replays/acceptance-seat1_seed11.zip
```

Accept only if both match JSON objects say `terminated_by: normal`, have empty
errors, numeric two-sided scores, parseable replays, and metadata containing
`realtime_scale: 1.0`, server/map hashes, and both policy hashes. Compare those
server/map hashes with `src/agentbench_frame/doto/PROVENANCE.json`; inspect
`population-build.json` for `official-sample: ready` and compiler output.

For a formal LLM Run, additionally require `summary.json`, `events.jsonl`, every
iteration record, `score_curve.json`, and `ig_curve.json`; failed or missing
points must remain explicit. Search the exported Results tree for the API key
before committing it to AgentBenchResults.
