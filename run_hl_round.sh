#!/usr/bin/env bash
# Run one real HL iteration round (3 acts) with the fixed probe + expanded nu-v2.
set -u
cd "E:/HL_Agent/AgentBenchFramework"
export PYTHONPATH=src
export AGENTBENCH_DATA="./agentbench_data"
BACKEND="E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
LOGIC_PY="C:/Users/27364/.conda/envs/torchy/python.exe"
uv run python -m agentbench_frame.hl \
  --logic "cd /d \"$BACKEND\" && python main.py" \
  --logic-python "$LOGIC_PY" \
  --reference ./agentbench_data/reference/nu-v2.json \
  --ladder-opponent rank=6 \
  --ladder-opponent rank=12 \
  --name hl-v2-probefix \
  --acts 3 --pairs 3 --seats 0 --timeout 15 \
  --permission-mode acceptEdits 2>&1 | tee .hl_codebase/hl-v2-probefix.run.log
