# Generals API Leaderboard Qualification Gate

**Date:** 2026-08-08
**Status:** Verifier implemented; no iteration harness or provider has qualified
**Contract:** `generals-api-leaderboard-qualification-v1`

## Purpose

The API leaderboard may be published only after the common iteration harness
demonstrates that a frontier model can produce a policy that statistically
beats the frozen strongest reproducible human-authored opponent. This is a
capability gate for the harness, not a leaderboard score and not a claim that
any provider has already passed.

An infinite iteration claim cannot be tested literally. The operational
replacement is a finite, predeclared sequence of cumulative coding-agent act
budgets. A failure at the largest checkpoint means the harness remains
unqualified; it does not prove that success is impossible at every larger
budget.

## Frozen Authorities

The asset manifest
`benchmark/leaderboard-qualification-v1.toml` freezes all of the following:

- opponent `advanced-rank02-robinliu-v18`, the strongest reproducible
  human-authored opponent in the canonical pilot;
- 20 qualification-only sealed seeds, each evaluated from both seats;
- official engine SHA-256;
- opponent source-tree SHA-256;
- replay-reading Skill SHA-256;
- common initial v7 policy SHA-256;
- act checkpoints `1, 2, 4, 8, 16, 32`; and
- three independent replicates, of which two must qualify.

The qualification seeds may not occur in learning, validation, policy
selection, prompt construction, or another benchmark manifest. Qualification
results may not be exposed to the provider or used to select a policy.

## Statistical Threshold

Each replicate evaluates exactly 40 games at every checkpoint: 20 sealed seeds
times two seats. A checkpoint passes only when both conditions hold:

1. the one-sided 95% Wilson lower confidence bound on the 40-game win
   probability is strictly greater than `0.5`; and
2. the policy wins at least 11 of the 20 games from each seat.

At 40 games the exact integer boundary is 26 wins. A 25-15 result has lower
bound approximately `0.49497` and fails; a 26-14 result has lower bound
approximately `0.52007` and passes, provided the seat minimum also passes.
Draws contribute one half to the reported game score but do not count as wins
for this conservative superiority test.

A replicate qualifies at its first passing checkpoint. The harness qualifies
when at least two of the three replicates qualify. Its qualification act budget
is the second-smallest replicate crossing budget. Every checkpoint in the
three-by-six matrix must still be present and complete, so an early crossing
cannot hide later missing evidence.

## Receipt Contract

The verifier accepts one JSON receipt with exact top-level, replicate,
checkpoint, budget, and game-result fields. Each replicate must have a distinct
non-empty run ID and a SHA-256 of its raw provider trace. Every complete
checkpoint must contain the exact 40 case identities. Cumulative budget fields
are:

- `coding_agent_acts`;
- `total_tokens`;
- `wall_time_s`;
- `learning_episodes`; and
- `cost_usd`.

The result status is one of:

- `qualified`: the complete matrix establishes the threshold in at least two
  replicates;
- `unqualified`: the matrix is complete but fewer than two replicates cross;
- `incomplete`: a checkpoint or valid game is missing; or
- `invalid`: an identity, hash, schema, case, or cumulative budget contract was
  changed.

Invalid games remain missing evidence. They are never converted into losses,
zeros, or a score on a smaller denominator. Leaderboard publication requires
the exact status `qualified`.

## Verification Command

Run from the Framework worktree:

```bash
.venv/bin/python -m agentbench_frame.cli generals \
  check-leaderboard-qualification \
  --agentbench-root /path/to/AgentBench \
  --manifest /path/to/AgentBench/backend_sources/corpus/28_generals/benchmark/pilot-v1.toml \
  --qualification-manifest /path/to/AgentBench/backend_sources/corpus/28_generals/benchmark/leaderboard-qualification-v1.toml \
  --replay-skill backend_sources/corpus/28_generals/skills/replay-analysis-v2/SKILL.md \
  --receipt /path/to/provider-qualification-receipt.json \
  --output /path/to/leaderboard-qualification-result.json
```

The command independently re-hashes frozen assets, evaluates the receipt, and
writes `generals-api-leaderboard-qualification-result-v1` JSON containing the
manifest and receipt hashes. Exit code `0` means qualified, `1` means complete
but unqualified or incomplete, and `2` means invalid input or contract failure.

## Verified Capability Leaderboard

`build-leaderboard` accepts only `qualified` result files produced against the
same qualification manifest and harness hash. At every 1/2/4/8/16/32 act
checkpoint, it combines the three complete replicate rows into 120 games per
provider/model/revision. Rows are ordered by one-sided 95% Wilson lower bound,
then score and wins; the output keeps both per-replicate and aggregate token,
wall-time, learning-episode, and USD-cost budgets.

The aggregator rejects invalid or unqualified inputs, duplicate
provider/model/revision identities, incomplete replicate matrices, mutated
manifests, and mixed harnesses. It is a reporting boundary only: its ranks and
scores cannot enter campaign prompts, policy selection, or the next act.

## Publication Boundary

This gate answers only whether the common harness has demonstrated the required
frontier capability. The verified capability leaderboard reports the frozen
single-round and cumulative multi-round checkpoints from that gate. A broader
provider benchmark may add separately sealed measurements, but it must not
retroactively change this qualification result or expose its sealed scores to a
provider.
