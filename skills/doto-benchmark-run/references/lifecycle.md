# Atomic lifecycle

Run commands from AgentBenchFramework with:

```bash
uv run --extra doto python -m agentbench_frame.doto <command>
```

## Initialize and inspect

The evaluator creates the authoritative Run, including the sealed pool identity,
before handing control to Codex:

```bash
python -m agentbench_frame.doto run init \
  --agent codex --initial-player-ai /workspace/playerAI.cpp \
  --doto-results ../DotoResults --run-id <run-id> \
  --train-bundle /workspace/doto-train-bundle \
  --test-bundle /proc/self/fd/<evaluator-fd> --skills-root skills
python -m agentbench_frame.doto run status \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id>
```

Initialization snapshots initial source, exact four Skill packages, public and
sealed pool hashes, benchmark version, and open baseline iteration. Do not keep
the sealed descriptor available to Codex after identity capture.

## Build, formally evaluate, compare, close

```bash
python -m agentbench_frame.doto iteration build \
  --run-dir <run-dir> --iteration 0
python -m agentbench_frame.doto iteration evaluate \
  --run-dir <run-dir> --iteration 0 --pool /workspace/doto-train-bundle \
  --cpu-target 70
python -m agentbench_frame.doto iteration compare \
  --run-dir <run-dir> --iteration 0
python -m agentbench_frame.doto iteration close \
  --run-dir <run-dir> --iteration 0
```

Baseline compare records `missing` because no parent exists. Evaluation declares
15 public policies × two candidate seats = 30 task IDs. It resumes only missing
cells and retains cell evidence. Formal score is null unless all 30 cells are
normal. Close a failed build or incomplete evaluation too; failure remains an
aligned curve point.

## Diagnose and create an explicit child

Use `match` for diagnostic battles and `replay` for parsing, outside the formal
matrix. After reading both the ZIP and trace with `doto-replay-reader`, write a
specific analysis file and a complete new source:

```bash
python -m agentbench_frame.doto iteration begin \
  --run-dir <run-dir> --parent 0 --source /workspace/child-playerAI.cpp \
  --analysis-file /workspace/iteration-analysis.md
```

Build, evaluate, compare, and close the returned iteration number. Strict compare
replays the closed parent and child policies on recorded observations. Identical
deterministic actions give 0; different supports give infinity with numeric null;
missing output records a reason. Never replace these statuses with another metric.

After each close, inspect `run status`, `score_curve.json`, `ig_curve.json`, the
matrix, and selected replay/trace evidence. Decide whether evidence supports a
new child; no command makes that decision automatically.

## Select and finalize

Choose any closed candidate using training evidence only. The evaluator opens
the sealed 28-policy bundle and passes its directory descriptor:

```bash
python -m agentbench_frame.doto run finalize \
  --run-dir <run-dir> --candidate-iteration <n> \
  --sealed-bundle-fd <evaluator-fd> --cpu-target 70
```

The command durably fixes candidate version/hash and test pool hash before
launching 28 policies × two seats = 56 tasks. Interruption may resume missing
cells for the same identities. It cannot select another candidate, and the final
result must never feed another iteration.

## Export

After DotoResults validates, export only the compatibility projection:

```bash
python -m agentbench_frame.doto run export \
  --run-dir <run-dir> --agentbench-results ../AgentBenchResults
```
