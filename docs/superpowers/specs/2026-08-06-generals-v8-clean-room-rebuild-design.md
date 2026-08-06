# Generals v8 clean-room rebuild design

## Purpose

Remove the superseded Generals v8 experiment completely, including its
deliverable Git history, then create one new v8 heuristic-learning iteration
from the frozen v7 policy. The new iteration must be blind to every old-v8
artifact and must produce an auditable single-act result against the strongest
frozen human-written opponent.

The new v8 succeeds scientifically even when its performance target is missed:
every runnable candidate is preserved and formally evaluated. The performance
target is at least `2/6` formal high-tier wins, at least `12/18` total formal
wins, and at least one high-tier win from each evaluated seat.

## Repositories and retained authorities

The work spans two local branches and remains unpushed until final user review:

- Framework: `zhaoyicheng/generals-hl-implementation` in
  `AgentBenchFramework/.worktrees/generals-hl`.
- Assets: `zhaoyicheng/generals-assets` in
  `AgentBench/.worktrees/generals-assets`.

The retained scientific authorities are:

- the complete v0-v7 source and run lineage;
- v7 run `20260730_1739_680b1632` and its frozen source hash;
- controlled-reference policy-KL run `20260731_1808_aedfbce7`;
- the frozen 12-state canonical action-space domain and exact support counts;
- the existing high, medium, and low 18-case formal benchmark.

Their relevant source, summary, and tree hashes must be recorded before the
rewrite and reproduced afterward.

## Old-v8 removal and history reconstruction

The old v8 is removed from the current trees and from deliverable branch
history. No backup branch, tag, or hidden repository reference is retained.
History reconstruction must remove old-v8 Framework implementation, CLI,
retry, transition-parity, report, tests, specifications, and plans. It must
also remove the old-v8 asset transition contract and champion challenge.
Later v0-v7 policy-KL work is retained.

The two old-v8 run directories identified by the pre-rewrite inventory are
deleted. Their identifiers are used only as ephemeral deletion inputs and are
not copied into retained specifications, plans, reports, or source files.

Derived data, caches, indices, and report references tied to those runs are
deleted as well. Deletion verification searches for the old run IDs, source
hashes, manifest IDs, and old implementation identifiers. Generic occurrences
of the label `v8` are not an error after the new clean-room v8 is introduced.

The rewritten branches must preserve v0-v7 and the final v7 KL assets byte for
byte. Only after that invariant passes may new-v8 implementation begin.

## Clean-room boundary

The new experiment is called v8 but has no lineage relationship to the deleted
v8. It receives new manifest IDs, seeds, run IDs, prompt hash, source hash, and
event IDs. Its sole policy parent is the frozen v7 source.

The Codex act may read only:

- the verified v7 source, tests, strategy description, and experience summary;
- the official rules and frozen replay-analysis Skill;
- new learning-only replay evidence against the strongest frozen human-written
  opponent;
- structured critical windows and dense diagnostics derived solely from those
  learning episodes.

The act must not read old-v8 code, prompts, logs, patches, replays, summaries,
scores, commits, or analysis. It must not read validation or formal seeds,
states, replays, outcomes, dense metrics, or scores. It must not read canonical
actions from the controlled-reference KL domain.

The prompt receipt records every included artifact and hash. A denylist and
role-aware seed audit fail closed before provider invocation if the boundary is
violated.

## Experiment suites and execution order

The suites are disjoint:

- Learning: three new seeds, both seats, six games total, strongest opponent.
- Validation: six new seeds, both seats, twelve games total, strongest
  opponent.
- Formal: the unchanged three-tier, three-seed, both-seat benchmark, eighteen
  games total.

Execution order is fixed:

1. Verify and freeze v7, the engine, the replay Skill, and suite contracts.
2. Run the six v7 learning episodes.
3. Extract deterministic, redacted replay evidence and critical windows.
4. Invoke Codex exactly once in an isolated editable workspace.
5. Run candidate legality, determinism, latency, and strategy tests.
6. If runnable, freeze the new v8 source, manifest, patch, prompt, and hashes.
7. Run the twelve validation episodes.
8. Run all eighteen formal cases regardless of validation performance.
9. Extend policy-change measurement in a new run with only the v8 probes and
   v7-to-v8 KL records.
10. Generate the updated score, dense-metric, budget, and KL reports.

Formal or validation evidence can never trigger another act within v8.

## Replay evidence and policy scope

Replay extraction prioritizes these learning-only events:

- the first transition into material main-general danger;
- large stacks failing to contribute to effective attack or defense;
- missed counter-capture, reinforcement, or production-point capture windows;
- economy purchases that consume resources needed for defense;
- the first material divergence in territory, army, coins, or main pressure;
- long macros whose executed prefix has low value or exposes the main general.

The Codex act may refactor explainable Python, planning, search, decision trees,
state evaluators, and compressed strategy structure. It is not restricted to
if-else accumulation. The intended focus is:

- main-general safety and counterattack priority;
- large-stack consolidation, routing, and attack timing;
- contact-phase allocation between economy and combat;
- safe-prefix behavior when later macro actions become invalid;
- deterministic behavior and bounded runtime from both seats.

The provider may update strategy code, candidate tests, `STRATEGY.md`, and
`EXPERIENCE.md`. Protected engine, harness, benchmark, and Skill files remain
immutable inside the act workspace.

## Measurement and interpretation

Every game preserves raw replay, structured summary, critical windows, outcome,
and dense measurements including survival, territory, army, economy or coins,
and main-general pressure. The run also records action profiles, macro lengths,
command families, and per-seat results.

Provider evidence includes the exact prompt, raw JSONL, stderr, token usage,
tool calls, elapsed time, pre/post workspace snapshots, source manifests,
content hashes, and the v7-to-v8 patch.

Budget accounting includes coding-agent acts, episodes, environment steps,
game-agent decision steps, primitive commands, tokens, and wall time. Reports
record raw, evo, and gain. Historical AUC remains unavailable across the known
missing score point and must not be interpolated.

Policy change reuses the exact frozen 12-state domain and exact canonical
support set without modifying the existing v0-v7 run. A new extension run:

- probes v8 twice in fresh subprocesses and requires identical actions;
- uses the document-mandated uniform `U_s`;
- computes `epsilon` values `0.001`, `0.01`, `0.05`, and `0.1`;
- requires `12/12` finite, non-negative coverage for every epsilon;
- adds only v8 actions and v7-to-v8 transition measurements.

This KL is controlled-state policy change, not performance, causal information
gain, or global policy equivalence.

## Failure and recovery semantics

- A provider failure preserves its failed run, raw logs, and budget but creates
  no v8 score.
- A candidate that fails tests is preserved as candidate evidence and is not
  marked as a runnable version.
- Validation failure cannot suppress formal evaluation of a runnable v8.
- Formal regression is reported and retained; v7 remains the historical
  champion when v8 does not beat it.
- Recovery verifies parent, prompt, provider output, candidate, event stream,
  and suite hashes. Any mismatch fails closed rather than combining evidence.
- Interrupted recovery cannot invoke a second Codex act for the same v8 run.

## Verification

Removal verification proves that no deliverable reference retains old-v8 run
IDs, hashes, manifests, files, or implementation identifiers. It also proves
the two old run directories are absent and all retained v0-v7 and KL hashes are
unchanged.

New-v8 contract tests prove:

- exact v7 parent identity;
- pairwise-disjoint learning, validation, and formal seeds;
- exactly one provider act;
- prompt allowlist and leak denylist enforcement;
- protected-file integrity;
- legal, deterministic, bounded candidate behavior;
- unconditional formal evaluation for every runnable candidate;
- exact event counts and provenance;
- complete, deterministic policy-KL probes.

The final gate runs the complete Framework test suite, the Generals asset
contract suite, and `agentbench data check`. Event quality must report zero
malformed lines, invalid events, unknown event types, duplicate IDs, missing
event IDs, and missing run IDs.

## Delivery

The final report states the new v8 formal result, high/medium/low and per-seat
breakdown, champion validation result, dense changes, budgets, KL, run paths,
and commits. It distinguishes failure to reach the target from pipeline
failure.

Both worktrees must be clean. The branches remain local until the user reviews
the result and explicitly authorizes push. No pull request or merge is created
without separate authorization.
