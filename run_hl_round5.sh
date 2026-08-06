#!/usr/bin/env bash
# Full HL iteration round 5 — replicate round3/round4 winning recipe.
# deepseek + v12 seed (wr 0.7 baseline vs rank01) + dynamic-nu + action-freq.
# Auto-names hl-v26-08-07-round5 (last_round=4 -> 5).
set -u
cd "E:/HL_Agent/AgentBenchFramework"
export PYTHONPATH=src
export AGENTBENCH_DATA="./agentbench_data"
BACKEND="E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
LOGIC_PY="C:/Users/27364/.conda/envs/torchy/python.exe"
uv run python -m agentbench_frame.hl \
  --logic "cd /d \"$BACKEND\" && python main.py" \
  --logic-python "$LOGIC_PY" \
  --reference ./agentbench_data/reference/nu-v2-t3.json \
  --initial-candidate src/agentbench_frame/lostspace/candidates/v12 \
  --ladder-opponent rank=1 \
  --acts 6 --pairs 20 --seats 0 --timeout 15 \
  --dynamic-nu --action-freq --save-traces \
  --model-key deepseek --max-turns 6 --max-tokens 16000 --claude-timeout 600 \
  2>&1 | tee .hl_codebase/hl-round5.run.log
