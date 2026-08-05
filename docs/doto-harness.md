# Codex-orchestrated DOTO benchmark

## What orchestrates the benchmark

AgentBenchFramework exposes deterministic atomic tools; it does not own an LLM
loop. **Codex decides** which training replay to inspect, what complete
`playerAI.cpp` change to make, which explicit parent to use, whether another
iteration is justified, and which complete training version enters the final
test. No iteration, token, or wall-time budget is imposed by the DOTO workflow.

The complete authority is stored in `DotoResults`. The unchanged
AgentBenchResults repository receives only the five-file **AgentBenchResults projection**
after the Run is sealed and validated.

Requirements: Python 3.11+, `uv`, `g++`, `make`, ZIP support, and:

```bash
uv sync --extra doto
```

## Four required Skills

Every Run snapshots and hashes exactly these packages:

- [`doto-benchmark-run`](../skills/doto-benchmark-run/SKILL.md): lifecycle,
  training selection, finalization, validation, and export.
- [`doto-game-rules`](../skills/doto-game-rules/SKILL.md): authoritative rules,
  observation, joint action, action mask, terminal, and strict KL support.
- [`doto-agent-authoring`](../skills/doto-agent-authoring/SKILL.md): fixed SDK and
  complete C++ candidate contract.
- [`doto-replay-reader`](../skills/doto-replay-reader/SKILL.md): replay/trace
  fields, events, evidence joins, and diagnosis.

Read rules and authoring before the baseline. Read replay-reader before drawing
any conclusion from a replay or trace. The workflow Skill governs every state
transition.

## Recommended Codex launch prompt

Initialize the authoritative Run before starting Codex, then replace the three
path placeholders in this lean prompt. Rules, schemas, commands, and failure
semantics stay in the versioned Skills rather than being duplicated here.

```text
Use $doto-benchmark-run, $doto-game-rules, $doto-agent-authoring, and
$doto-replay-reader to autonomously improve a native DOTO playerAI.cpp.

Authoritative Run: <RUN_DIR>
Public training bundle: <TRAIN_BUNDLE>
AgentBenchResults projection root: <AGENTBENCH_RESULTS>

Goal: use only public training evidence to maximize the number of human
policies defeated under the benchmark's two-seat rule. Start by inspecting the
durable Run status. Complete and close the baseline if needed. For every child,
inspect selected normal replay and trace evidence, save a specific analysis,
create a complete non-identical playerAI.cpp from an explicit closed parent,
build it, run the full 30-cell training matrix, record strict IG or its exact
missing reason, close the iteration, and inspect the aligned score/IG curves.

Decide yourself whether evidence justifies another child and which complete
training version to select. Do not ask for confirmation between ordinary atomic
steps. Do not use a fixed iteration count, token budget, or wall-time budget.
Never inspect or infer sealed policies, and never use hidden-test evidence to
change a candidate. Preserve every failure and incomplete result; do not edit
authoritative result files by hand and do not substitute another metric for
strict KL/IG.

If no sealed-bundle descriptor is available, stop after durably selecting and
reporting the training candidate for evaluator finalization. If the evaluator
provides the descriptor, finalize exactly once, make no further candidate
changes, validate DotoResults, export the five-file AgentBenchResults
projection, verify it, and report artifact paths plus explicit failures.
```

For an API-driven Codex session, pass this as the user task after mounting the
repository and prepared Run. The evaluator, not the prompt, owns the sealed
bundle descriptor and the one-shot transition into hidden evaluation.

## Prepare human pools

Verify all 43 frozen complete snapshots, then build the 15-policy public train
bundle. The evaluator separately builds and owns the 28-policy sealed bundle.

```bash
uv run --extra doto python -m agentbench_frame.doto population verify \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root ../AgentBench/backend_sources/corpus/23_doto
uv run --extra doto python -m agentbench_frame.doto population build-train \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root src/agentbench_frame/doto/human_policies/train \
  --output-dir /workspace/doto-train-bundle
```

Require 43/43 identities verified and 15/15 public policies ready. Do not expose
sealed source, policy names, paths, binaries, or old test outcomes to Codex.

## Initialize an authoritative Run

The evaluator briefly opens the sealed bundle to capture its hash, creates the
Run, and closes that descriptor before Codex begins training:

```bash
uv run --extra doto python -m agentbench_frame.doto run init \
  --agent codex --initial-player-ai examples/doto-initial-playerAI.cpp \
  --doto-results ../DotoResults --run-id <run-id> \
  --train-bundle /workspace/doto-train-bundle \
  --test-bundle /proc/self/fd/<evaluator-fd> --skills-root skills
uv run --extra doto python -m agentbench_frame.doto run status \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id>
```

Initialization snapshots the initial source, four complete Skills, pool hashes,
and benchmark version. Never hand-edit the new Run.

## Complete one training iteration

Build the baseline, run/resume the full **30-cell** matrix (15 humans × both
candidate seats, seed 11), record baseline IG as missing, and close:

```bash
uv run --extra doto python -m agentbench_frame.doto iteration build \
  --run-dir <run-dir> --iteration 0
uv run --extra doto python -m agentbench_frame.doto iteration evaluate \
  --run-dir <run-dir> --iteration 0 --pool /workspace/doto-train-bundle \
  --cpu-target 70
uv run --extra doto python -m agentbench_frame.doto iteration compare \
  --run-dir <run-dir> --iteration 0
uv run --extra doto python -m agentbench_frame.doto iteration close \
  --run-dir <run-dir> --iteration 0
```

The adaptive multi-process scheduler targets **70%** aggregate CPU, grows below
65%, holds at 65–75%, and pauses new launches above 75%. `--workers` is an
optional hard ceiling. Running matches are never cancelled.

A formal score exists only after 30 normal cells. Failed build, timeout, crash,
protocol error, corrupt replay, and missing cell remain explicit and produce a
null formal score. Close failed/incomplete iterations; do not erase them.

## Diagnose, change, and create a child

Use `match` only for diagnostics. Use the official ZIP for events/scores and the
trace for observations/actions:

```bash
uv run --extra doto python -m agentbench_frame.doto match \
  --agent0 <candidate> --agent1 <training-opponent> --seed 11 \
  --output-dir /workspace/doto-diagnostics --tag candidate-seat0
uv run --extra doto python -m agentbench_frame.doto replay \
  --path /workspace/doto-diagnostics/candidate-seat0_seed11.zip \
  --jsonl /workspace/doto-diagnostics/candidate-seat0.events.jsonl
```

After evidence review, save a nonempty analysis and a non-identical complete
source, then create the explicit child:

```bash
uv run --extra doto python -m agentbench_frame.doto iteration begin \
  --run-dir <run-dir> --parent 0 --source /workspace/child-playerAI.cpp \
  --analysis-file /workspace/iteration-analysis.md
```

Build, evaluate all 30 cells, compare strict deterministic KL on recorded
observations, and close the returned iteration. Equal actions give strict KL 0;
different deterministic support gives infinity with numeric null; missing or
invalid output stays missing with a reason. Never call distance, change rate, or
score gain IG.

After each close, Codex inspects `run status`, aligned score/IG curves, failures,
and selected evidence. It either justifies another explicit child or stops and
selects among complete training versions. There is no automatic repetition.

## Finalize the hidden matrix once

The evaluator opens the sealed bundle only after training selection:

```bash
uv run --extra doto python -m agentbench_frame.doto run finalize \
  --run-dir <run-dir> --candidate-iteration <selected-training-iteration> \
  --sealed-bundle-fd <evaluator-fd> --cpu-target 70
```

Candidate and pool identities are durably fixed before the **56-cell** matrix
(28 policies × both seats) starts. An interruption resumes only missing cells
for the same identities. The final Run seals even if incomplete; final evidence
must never create another candidate.

A human policy counts as defeated only when both seats terminate normally and
their candidate-oriented mean score difference is strictly positive.

## Validate, report, and export

```bash
uv --directory ../DotoResults run doto-results validate \
  runs/23_doto/codex/<run-id>
uv --directory ../DotoResults run doto-results aggregate .
uv --directory ../DotoResults run doto-results build-report . --output _site
uv run --extra doto python -m agentbench_frame.doto run export \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id> \
  --agentbench-results ../AgentBenchResults
uv --directory ../DotoResults run doto-results check-projection \
  ../AgentBenchResults/runs/23_doto/codex/<run-id>
```

The projection contains exactly `run.toml`, `summary.json`, `score_curve.json`,
`ig_curve.json`, and `doto_results_ref.json`. Before publication, scan both
targets for credentials, hidden source fragments, absolute sealed paths, `NaN`,
and `Infinity`; exclude binaries, caches, locks, and unredacted sealed metadata.
