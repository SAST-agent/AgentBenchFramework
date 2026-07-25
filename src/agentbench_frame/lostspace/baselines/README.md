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

## The 16 ranked human algorithms

`AgentBench/top_algorithms/corpus/25_lostspace_final_ladder/` holds 16 top
contest submissions (rank01..rank16): 11 C++ (`make`), 4 single-file Python,
and 1 Python bundle with `.npy` data (rank 3). They are referenced **in place**
(never copied into this framework) and wired up via
[`ladder.py`](../ladder.py). C++ sources are compiled into a build cache at
`<repo>/.cache/ladder/` so the corpus tree stays clean.

Add one as an opponent with `--ladder-opponent rank=NAME`, where `NAME` is a
rank number (`rank=1`), username (`rank=omegafantasy`), or display name
(`rank=最终幻想`):

```bash
uv run python -m agentbench_frame.lostspace \
  --logic 'cd /d <gamecode_logic> && python main.py' \
  --candidate-name myagent --candidate 'python my_agent.py' \
  --ladder-opponent rank=1 \
  --pairs 5 --seats all
```

Build (and smoke-test) all 16 at once:

```bash
uv run python -m agentbench_frame.lostspace.scripts.build_ladder
uv run python -m agentbench_frame.lostspace.scripts.build_ladder --rank 1 3 6   # subset
uv run python -m agentbench_frame.lostspace.scripts.build_ladder --build-only    # just compile
```

### TLE / crash handling

The harness enforces a per-round time limit the same way the saiblo judger
does: an AI that does not answer an action request in time (or whose process
exits) is reported to the logic as an `ai_error` (`timeOutError` / `runError`),
the logic eliminates that player, and the match continues. Several ranked
algorithms legitimately stall in specific branches (e.g. rank01 returns from
its get-key strategy without sending `finish`) — replicating the TLE rather
than deadlocking the whole match is the faithful behaviour. See
[`match.py`](../match.py) for the routing rule and `_ACTION_REQUEST_TYPES`.
