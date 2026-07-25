# LostSpace baselines

Python AI clients used as reference opponents and table fillers. Each is a
self-contained script speaking the LostSpace Saiblo wire protocol (receive:
4 ASCII digits + JSON; send: 4-byte big-endian length + JSON).

## `sample_ai/main.py`

The **official contest sample AI**, copied verbatim from
`AgentBench/backend_sources/corpus/25_lostspace/logic/judge_dev_sample_ai/`.
Heuristic BFS-based strategy. Use as a strong-ish reference opponent and as the
default seat filler (the evaluator's `--filler` default points here).

```bash
python <this dir>/sample_ai/main.py
```

## `random_agent.py`

A minimal protocol-correct agent: one random adjacent move per turn, then
finish. Weak baseline / filler, useful for fast smoke tests.

```bash
python <this dir>/random_agent.py
# or, as a module:
python -m agentbench_frame.lostspace.baselines.random_agent
```

## Future: the 16 ranked algorithms

`AgentBench/top_algorithms/corpus/25_lostspace_final_ladder/extracted/` holds 16
top contest submissions (rank01..rank16), mostly **C++**. They are not wired in
here — porting them requires compiling each `ai_client.cpp` and adapting the
subprocess protocol. They are the natural next step for a high-fidelity
opponent pool. See that directory's `MANIFEST.tsv` for ranks/scores/languages.
