# LostSpace — evaluation harness

LostSpace (contest 25): 4-player turn-based survival / collection / escape PvP
on a 7×7×3 map. Collect 4 keys, survive shrinking zones, reach the escape
capsule. This package is the AgentBenchFramework integration layer for it,
mirroring `agentbench_frame/aquawar/`.

```
agentbench_frame/lostspace/
├── __init__.py            # public API
├── __main__.py            # `python -m agentbench_frame.lostspace ...`
├── cli.py                 # argparse CLI
├── evaluator.py           # LostSpaceEvaluator: 4-seat rotation + Run tracking
├── match.py               # run_match(): spawn logic + 4 AIs, shuttle protocol
├── protocol.py            # Saiblo frame read/write (cross-platform)
├── replay_view.py         # text replay viewer
├── replay_view_html.py    # HTML viewer (placeholder, phase 2)
├── replay_format.md       # native replay format reference
├── baselines/             # Python baseline AI clients (sample_ai, random_agent)
├── candidates/            # your iterating agents (v1, v2, ...)
└── scripts/iterate_demo.py
```

## Run an evaluation

```bash
cd AgentBenchFramework
export PYTHONPATH="src"        # Windows cmd: set PYTHONPATH=src

python -m agentbench_frame.lostspace \
  --logic "<logic command>" \
  --candidate-name cand-v1 \
  --candidate "<candidate command>" \
  --opponent random="python .../baselines/random_agent.py" \
  --opponent sample="python .../baselines/sample_ai/main.py" \
  --pairs 3 --seats all --timeout 15 --save-replays
```

The **logic command** must run the official game logic with its working
directory at `gamecode_logic/` (it loads `src/mapconf2.map` via a relative
path and imports `from src import main`):

```
# POSIX
cd /path/to/25_lostspace/logic/gamecode_logic && python main.py
# Windows (cmd)
cd /d E:\path\to\25_lostspace\logic\gamecode_logic && python main.py
```

The official backend lives at
`AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic/`. It
needs `antlr4-python3-runtime==4.9.*` (`pip install antlr4-python3-runtime==4.9.3`).

### Options

| flag | meaning |
|------|---------|
| `--logic` | command that launches the game logic subprocess |
| `--candidate-name` | run name (versioned, e.g. `cand-v2`) |
| `--candidate` | command that launches the candidate AI |
| `--opponent NAME=CMD` | baseline opponent (repeatable; ≥1) |
| `--filler CMD` | seat filler when the table is short (default: bundled sample AI) |
| `--pairs N` | games per (opponent × seat) |
| `--seats` | `all` or a single seat `0`/`1`/`2`/`3` |
| `--timeout` | per-read seconds before a player times out |
| `--save-replays` | keep native replay JSON per match under `artifacts/` |
| `--save-traces` | keep per-turn protocol trace |

### Output

```
$AGENTBENCH_DATA/runs/25_lostspace/<candidate-name>/<run_id>/
├── run.toml          # metadata (game, agent, type, config)
├── summary.json      # win_rate, h2h, lostspace.{aggregate,by_opponent,by_seat}
├── events.jsonl      # step-level events
├── matches.jsonl     # one record per game (checkpointed, fsynced)
└── artifacts/        # <opp>-pairNNN-seatS.json (+ .trace.jsonl) if --save-replays
```

`summary.json` key metrics:
- `win_rate` — fraction of valid games where the candidate ranked 1st.
- `lostspace.aggregate.avg_rank` — mean finishing rank (1..4; lower is better).
- `lostspace.aggregate.avg_score` — mean rank-points (4..1; higher is better).
- `lostspace.by_opponent` / `lostspace.by_seat` — per-opponent / per-seat breakdowns.

## View a replay

```bash
python -m agentbench_frame.lostspace.replay_view <run_dir>/artifacts/<...>.json
python -m agentbench_frame.lostspace.replay_view replay.json --round 12 --player 0
```

See [`replay_format.md`](replay_format.md) for the native replay layout.

## The iteration loop

See [`ITERATE.md`](ITERATE.md) for the full recipe and `scripts/iterate_demo.py`
for a runnable example: run candidate v1, view a replay, edit & save v2,
re-evaluate, compare.

## Baselines

See [`baselines/README.md`](baselines/README.md). Bundled Python baselines:
official `sample_ai` (heuristic BFS) and `random_agent` (weak filler). The 16
ranked contest algorithms under `top_algorithms/corpus/25_lostspace_final_ladder/`
are wired in via `--ladder-opponent rank=NAME` (see
[`ladder.py`](ladder.py) and [`scripts/build_ladder.py`](scripts/build_ladder.py)).

## How it works (protocol)

The harness is the **judger**: it spawns the logic + 4 AI subprocesses and
shuttles the Saiblo wire protocol. LostSpace framing (identical to AquaWar's):

- judger→logic: `[4B BE length][json]`
- logic→judger: `[4B BE length][4B BE target][json]` (target -1 = to judger)
- judger→AI: the logic pre-wraps each `content` with 4 ASCII digits + json;
  the harness forwards those bytes verbatim.
- AI→judger: `[4B BE length][json]`

Routing rule (the one real difference from AquaWar): a player replies only when
it is both in `listen` **and** in `player` (i.e. the frame is addressed to the
in-turn player). Witness/off-round frames carry the in-turn id in `listen` but
address `player` to bystanders, so they expect no reply.
