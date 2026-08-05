# DOTO population and official-duration acceptance

Formal DOTO evaluation uses the vendored official server and map for 300 seconds
at 20 FPS with `realtime_scale=1.0`. The fixed benchmark seed is 11. Every human
policy is evaluated twice, once with the candidate in each seat. Official-duration
commands are opt-in because a complete 30-cell training matrix is expensive.

## Verify and build isolated populations

From `AgentBenchFramework`, verify all 43 complete source snapshots and build the
15-policy public pool. The other 28 policies remain evaluator-owned and hidden.

```bash
uv run --extra doto python -m agentbench_frame.doto population verify \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root ../AgentBench/backend_sources/corpus/23_doto

uv run --extra doto python -m agentbench_frame.doto population build-train \
  --manifest src/agentbench_frame/doto/population.toml \
  --source-root src/agentbench_frame/doto/human_policies/train.tar.gz \
  --output-dir /tmp/doto-train-bundle
```

Expected: 43 verified snapshots, exactly 15 ready training policies, distinct
complete-source hashes, executable hashes, and protocol-smoke success. A sealed
bundle is built outside the Codex-readable workspace with
`scripts/import_doto_population.py build-sealed`; its public report must say
28/28 ready and must not contain source paths or executable paths.

## Fast complete-matrix regression

This command uses fixture AIs and the fixture server. It validates 30 unique cells,
seat orientation, replay/trace isolation, ordered aggregation, and CPU telemetry.
It must never be stored as an official Run.

```bash
uv run --extra doto python -m agentbench_frame.doto evaluate \
  --candidate tests/doto/fixtures/scripted_ai.py \
  --pool tests/doto/fixtures/train-bundle --split train \
  --output-dir /tmp/doto-matrix-smoke --seed 11 \
  --cpu-target 70 --workers 8 \
  --server-dir tests/doto/fixtures/fake_server --test-only
```

Expected: 30 attempted and normal matches, completion rate 1.0, 30 distinct replay
and trace pairs, no cancellations, and worker/CPU histories in `matrix.json`.

## Formal 30-cell training evaluation

Build the candidate and run the complete public matrix. Omitting `--workers` lets
the scheduler use the machine CPU count as its hard ceiling; it grows below 65%,
holds between 65–75%, pauses new launches above 75%, and never kills a running match.

```bash
uv run --extra doto python -m agentbench_frame.doto build \
  --player-ai examples/doto-initial-playerAI.cpp \
  --output-dir /tmp/doto-candidate

uv run --extra doto python -m agentbench_frame.doto evaluate \
  --candidate /tmp/doto-candidate/main.out \
  --pool /tmp/doto-train-bundle --split train \
  --output-dir /tmp/doto-formal-train --seed 11 --cpu-target 70
```

A formal score exists only when all 30 cells terminate normally. Timeouts, process
failures, missing replays, and incomplete matrices remain explicit and yield a null
formal score. A human policy counts as defeated only when both seats are normal and
the candidate-oriented mean score difference is strictly positive.

## Authoritative Codex-directed lifecycle

The evaluator initializes the Run in `DotoResults` and snapshots the four Skill
packages, initial source, public-pool identity, and sealed-pool identity. Codex
then uses only the atomic `run` and `iteration` commands: it chooses explicit
parents, builds, launches complete 30-cell training matrices, reads replays,
records strict KL or an explicit missing reason, and closes every iteration.
There is no framework-owned LLM loop and no benchmark budget.

Every valid child has a closed, completely evaluated parent. Score and IG curves
are rebuilt at close with the same iteration/version identities. Failed builds,
29/30 matrices, missing decisions, and infinite strict KL remain recorded; no
surrogate metric is written as IG.

The adaptive scheduler targets 70% aggregate CPU, launches more work below 65%,
holds between 65% and 75%, and pauses new launches above 75%. `--workers` is a
hard ceiling; already running matches are never cancelled.

## One-shot hidden 56-cell final evaluation

Only the evaluator may open the sealed directory and pass an inherited directory
descriptor. The final command must be issued once for the training-selected candidate;
its result must not be used to create another iteration.

```bash
exec {DOTO_SEALED_FD}</evaluator/doto-sealed
uv run --extra doto python -m agentbench_frame.doto population verify-sealed \
  --manifest src/agentbench_frame/doto/population.toml \
  --bundle-fd "$DOTO_SEALED_FD"
uv run --extra doto python -m agentbench_frame.doto run finalize \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id> \
  --candidate-iteration <training-selected-iteration> \
  --sealed-bundle-fd "$DOTO_SEALED_FD" --cpu-target 70
```

Expected: exactly 56 attempted cells, 28 policy summaries, fixed candidate/bundle
identity, no source or absolute sealed path in stdout, and all failures retained.
Candidate identity is durably selected before work starts. An interruption may
resume missing cells for that same candidate, but cannot select another candidate.
The Run is sealed even when the final matrix is incomplete so that failures cannot
be silently retried into a better reported result.

## Validate, export, and scan

The full Run is authoritative in DotoResults. AgentBenchResults remains unchanged
and receives only the five-file projection.

```bash
uv --directory ../DotoResults run doto-results validate \
  runs/23_doto/codex/<run-id>
uv run --extra doto python -m agentbench_frame.doto run export \
  --run-dir ../DotoResults/runs/23_doto/codex/<run-id> \
  --agentbench-results ../AgentBenchResults
uv --directory ../DotoResults run doto-results check-projection \
  ../AgentBenchResults/runs/23_doto/codex/<run-id>
```

Before publication, scan both outputs for API-key material, hidden policy source
fragments, absolute sealed-bundle paths, `NaN`, and `Infinity`. The authoritative
validator must report 30 declared tasks for every formal iteration and 56 for the
final test, verify normal-cell replay/trace links, aligned score/IG versions, pool
and candidate identities, and the sealed-content hash. The projection directory
must contain exactly `run.toml`, `summary.json`, `score_curve.json`,
`ig_curve.json`, and `doto_results_ref.json`.
