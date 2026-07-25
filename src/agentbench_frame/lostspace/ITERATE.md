# Iteration loop — 「对战 → 回放 → 改 Agent → 存新版本 → 再评测」

This is the closed loop the LostSpace harness exists to support. One full pass:

```
   ┌─────────────────────────────────────────────────────────┐
   │  1. evaluate candidate version Vn  (python -m ...lostspace)│
   │  2. read summary.json (win_rate / avg_rank)                │
   │  3. view a replay of a loss (replay_view)                  │
   │  4. edit the agent                                         │
   │  5. save as V(n+1) under candidates/                       │
   │  6. re-evaluate V(n+1)                                     │
   │  7. compare Vn vs V(n+1) summaries                         │
   └─────────────────────────────────────────────────────────┘
```

## Step-by-step

Set up env once:

```bash
cd AgentBenchFramework
export PYTHONPATH="src"                 # Windows cmd: set PYTHONPATH=src
export AGENTBENCH_DATA="./agentbench_data"
LS=$(pwd)/src/agentbench_frame/lostspace
BACKEND=E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic
# logic command (Windows):   cd /d "$BACKEND/gamecode_logic" && python main.py
```

### 1–2. Evaluate v1

`candidates/v1/agent.py` is a copy of the official sample AI. Run it:

```bash
python -m agentbench_frame.lostspace \
  --logic "cd /d \"$BACKEND/gamecode_logic\" && python main.py" \
  --candidate-name cand-v1 \
  --candidate "python \"$LS/candidates/v1/agent.py\"" \
  --opponent random="python \"$LS/baselines/random_agent.py\"" \
  --pairs 1 --seats 0 --timeout 15 --save-replays
# -> prints run_dir, e.g. agentbench_data/runs/25_lostspace/cand-v1/<run_id>
```

Read the headline numbers:

```bash
RUN1=<that run_dir>
python -c "import json;s=json.load(open('$RUN1/summary.json'));print('v1 win_rate',s['win_rate'],'avg_rank',s['lostspace']['aggregate']['avg_rank'])"
```

### 3. View a losing replay

```bash
python -m agentbench_frame.lostspace.replay_view "$RUN1/artifacts/"*.json | less
# focus on a player / round:
python -m agentbench_frame.lostspace.replay_view "$RUN1/artifacts/"*.json --player 0 --round 50
```

Look for: did the candidate get keys? die early? waste turns? This drives the
edit in step 4.

### 4–5. Edit and save v2

```bash
cp -r "$LS/candidates/v1" "$LS/candidates/v2"
# edit candidates/v2/agent.py  (the bundled v2 already tweaks the Kit threshold
# as an example — replace it with your own change)
```

### 6–7. Re-evaluate and compare

```bash
python -m agentbench_frame.lostspace \
  --logic "cd /d \"$BACKEND/gamecode_logic\" && python main.py" \
  --candidate-name cand-v2 \
  --candidate "python \"$LS/candidates/v2/agent.py\"" \
  --opponent random="python \"$LS/baselines/random_agent.py\"" \
  --pairs 1 --seats 0 --timeout 15 --save-replays
RUN2=<that run_dir>
```

Compare:

```bash
python -c "
import json
a=json.load(open('$RUN1/summary.json'))['lostspace']['aggregate']
b=json.load(open('$RUN2/summary.json'))['lostspace']['aggregate']
print('v1', a['win_rate'], a['avg_rank'])
print('v2', b['win_rate'], b['avg_rank'])
"
```

`win_rate` up / `avg_rank` down ⇒ the change helped.

## Runnable demo

`scripts/iterate_demo.py` automates 1–2 and 6–7 for the bundled v1/v2 so you
can see the loop run end-to-end:

```bash
python src/agentbench_frame/lostspace/scripts/iterate_demo.py
```

(It runs real matches, so it takes several minutes. Tune with the
`LS_PAIRS` / `LS_SEATS` env vars.)

## Notes

- **Games are slow.** The official sample AI plays full ~100-round games with
  BFS; a single match is typically minutes. For fast iteration use `--pairs 1
  --seats 0` and the `random_agent` opponent, then scale up for final numbers.
- **Versions are first-class.** Each `--candidate-name` gets its own
  `runs/25_lostspace/<name>/<run_id>/` tree, so v1/v2/v3 never overwrite each
  other and you can diff summaries across runs.
- **Replays are per-match** under `artifacts/` when `--save-replays` is set;
  they are the native logic format (see `replay_format.md`).
