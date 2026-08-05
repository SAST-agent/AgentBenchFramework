---
name: doto-benchmark-run
description: Use when Codex must direct, resume, inspect, finalize, validate, or export an authoritative 23rd DOTO benchmark Run using AgentBenchFramework atomic tools.
---

# DOTO Benchmark Run

Codex decides what training evidence to inspect, whether a new candidate is
warranted, which closed training version to select, and when to stop training.
There is no fixed iteration loop and no benchmark budget.

**REQUIRED COMPANION SKILLS:** Read `doto-game-rules` and
`doto-agent-authoring` before the first candidate. Read `doto-replay-reader`
before interpreting any replay, trace, numeric frame, event, or ignored action.

## Atomic command surface

| Intent | Command |
|---|---|
| Snapshot a new authority | `run init` |
| Inspect durable state | `run status` |
| Create an explicit child | `iteration begin` |
| Compile and attach candidate | `iteration build` |
| Run/resume all 30 training cells | `iteration evaluate` |
| Record strict old/new KL status | `iteration compare` |
| Seal iteration and align curves | `iteration close` |
| Lock candidate and run 56 hidden cells | `run finalize` |
| Write five-file compatibility view | `run export` |

Read [references/lifecycle.md](references/lifecycle.md) before changing Run
state. Read [references/results.md](references/results.md) before selecting a
candidate, interpreting a failure, validating DotoResults, or exporting the
AgentBenchResults projection.

## Non-negotiable invariants

- Use official rules, SDK, server, map, fixed seed 11, and both seats.
- Keep diagnostic matches separate from formal matrices.
- Begin a child only from an explicit closed parent with a ready build and
  complete formal score. Save a nonempty evidence-based analysis.
- Record every build/evaluation/IG failure and close the iteration. Never erase,
  overwrite, fabricate, or silently rerun a consumed result.
- Never substitute score change, coordinate distance, or action-change rate for
  strict KL/IG. Record an exact missing reason when strict KL is unavailable.
- Select the final candidate from training evidence only; run the **final test exactly once**.
  After selection is durable, resume only missing cells for the same identity.
- Do not use final-test evidence to create another candidate.
- **Do not edit authoritative result files.** Use atomic commands and validator
  output; `projection.json` is the only post-seal mutable sidecar.
- Never reveal sealed policies, source, identities, paths, or earlier outcomes.

## Decision checkpoint

After closing an iteration, inspect status, aligned score/IG curves, failures,
and selected replays. Either justify a concrete child against that evidence or
select among complete training versions. More iterations are allowed but are
never automatic.
