#!/usr/bin/env bash
# Run a real HL iteration round against the raw DeepSeek API.
# 10 acts, curriculum ladder from rank10 up (rank10 -> rank8 -> ... -> rank1),
# early-stop when KL < 0.05 for the trailing acts after act 5.
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
  --ladder-opponent rank=10 \
  --ladder-opponent rank=8 \
  --ladder-opponent rank=6 \
  --ladder-opponent rank=4 \
  --ladder-opponent rank=2 \
  --ladder-opponent rank=1 \
  --name hl-deepseek-r8 \
  --acts 10 --pairs 2 --seats 0 --timeout 15 \
  --curriculum --promote-rank 2.0 \
  --model-key deepseek --max-turns 6 --max-tokens 16000 --claude-timeout 600 \
  --min-kl 0.05 --stall-after 5 \
  2>&1 | tee .hl_codebase/hl-deepseek-r8.run.log
