# Generals v9 scientific-attribution design

## Purpose

Explain why the clean-room v8 changed behavior on the frozen controlled states
by zero measured KL yet regressed from the v7 champion, then perform exactly
one scientifically attributable v7-to-v9 heuristic-learning act. The design
separates diagnosis, learning, validation, formal evaluation, and policy-change
measurement so that neither benchmark leakage nor a changed measurement domain
can explain the result.

The experiment succeeds as a scientific run whenever it preserves a complete,
auditable result for every runnable v9 candidate. It succeeds as a champion
challenge only if v9 meets every predeclared performance criterion:

- at least `2/12` wins in strongest-human validation, including at least one
  win from each evaluated seat;
- at least `13/18` wins on the unchanged formal benchmark, strictly exceeding
  the v7 result of `12/18`;
- at least `2/6` high-tier formal wins, including at least one high-tier win
  from each evaluated seat; and
- all 18 formal games are valid.

If any criterion is missed, v9 remains an immutable historical testcase and
v7 remains the champion.

## Repositories, authorities, and non-goals

The work remains on the existing local Framework and Generals-assets branches.
No push, pull request, merge, deletion, or history rewrite is part of this
experiment. Existing v0-v8 code, commits, assets, runs, reports, and figures
remain immutable.

The experiment treats these as frozen authorities:

- the successful v0-v8 version snapshots and their recorded content hashes;
- v7 as the sole policy parent and current champion;
- the clean-room v8 candidate as a population member and diagnostic testcase,
  not as the parent of v9;
- the unchanged 18-case high/medium/low formal benchmark;
- the official Generals Python engine and frozen strongest human-written
  opponent;
- the human-authored and Agent-polished rules/replay Skill;
- the legacy 12-state canonical action-space specification, exact support
  counts, and strict uniform-epsilon KL convention; and
- `附件/information_gain.md` as the terminology and measurement authority.

This design does not alter historical scores, relabel policy KL as epistemic
information gain, approximate an exact action support, tune on validation or
formal evidence, perform more than one v9 coding-agent act, or promote a
candidate merely because it improves a dense proxy metric.

## Experimental boundaries and seed isolation

Five roles remain logically and physically disjoint:

1. historical runs and the frozen formal benchmark;
2. attribution games used only to explain v8's regression;
3. v9 learning evidence derived from the attribution run;
4. new strongest-human validation games; and
5. controlled states used only for policy-change measurement.

The attribution seeds are the exact tuple:

```text
302101, 302202, 302303, 302404, 302505, 302606
```

Each seed runs both seats, yielding 12 paired cases per policy. The v9
validation seeds are the exact disjoint tuple:

```text
303101, 303202, 303303, 303404, 303505, 303606
```

They also run both seats, yielding 12 games against the strongest frozen human
opponent. Asset contracts audit role labels, seeds, maps, opponents, seats, and
content hashes against all historical learning, validation, formal, and KL
inputs. Any overlap fails closed before a provider invocation or evaluation.

The unchanged formal benchmark is sealed throughout attribution and the v9
act. Its states, replays, actions, outcomes, dense metrics, and aggregate scores
are not provider inputs.

## Paired 2x2 scientific attribution

The diagnosis isolates the two substantive clean-room v8 interventions while
holding the frozen v7 policy constant:

| Cell | Policy |
|---|---|
| A | frozen v7 |
| B | v7 plus large-stack priority only |
| C | v7 plus contact-before-economy only |
| D | both interventions, representing the v8 core change |

Cells B and C are deterministic, reviewable ablation policies constructed from
v7 solely for diagnosis. They are not formal HL versions, do not increment the
iteration count, and consume no Codex act. Cell D is checked for behavioral
parity with the frozen v8 policy on the intervention surface; any remaining
v8-only behavior is named as a residual rather than silently attributed to the
two factors.

All four cells run the identical 12 seed-seat cases. Every game saves the full
official replay, normalized replay, dense trajectory, action profile,
termination data, and first occurrences of predefined critical events.
Measurements include:

- win, loss, and validity;
- survival rounds;
- territory, army, and coin trajectories;
- main-general pressure;
- canonical macro-action disagreement; and
- earliest paired trajectory divergence.

The paired descriptive effects are:

```text
large-stack effect = B - A
contact effect     = C - A
interaction       = D - B - C + A
```

They are computed per case for binary outcomes and dense summaries. The report
provides all case-level values, paired differences, effect summaries, and
paired bootstrap intervals. With only 12 cases it does not make an unsupported
null-hypothesis significance claim.

A rule is identified as harmful only when its trigger, same-state action
disagreement, subsequent trajectory deterioration, and outcome or dense
measurements form a consistent trace. Win rate alone is not accepted as causal
evidence. Conflicting evidence remains explicitly inconclusive.

## Replay diagnostic state domain

The attribution run may select at most 48 diagnostic states. For each of the
12 seed-seat cases it selects at most four pre-action states: up to two from
the v7 trajectory and up to two from the v8 trajectory. Selection uses this
fixed priority order:

1. first material main-general danger;
2. first enemy contact;
3. first economy-versus-combat conflict;
4. first ineffective use of a large stack;
5. first v7/v8 action divergence; and
6. a material pre-terminal decision.

Within a priority class, deterministic round, actor, and state-hash ordering
breaks ties. A missing event class is recorded as missing; it is not imputed or
replaced by a convenient later point.

The official game state for each selected point is reconstructed once. All
four policies are then probed on that same state in fresh isolated processes.
Canonical macro-actions, legality, latency, trigger attribution, and policy
hash are saved. Actions observed at different states on divergent trajectories
are never compared as if they were a same-state policy disagreement.

This replay domain supports action-disagreement and earliest-divergence
diagnostics. It is not the exact-KL domain and does not require exact support
enumeration for late-game states.

## Expanded exact policy-KL domain

Policy-change measurement has two separately reported domains:

- `legacy-12`: the existing 12 controlled states, retained byte for byte; and
- `expanded-24`: those same 12 states plus 12 new official-valid intervention
  states.

The 12 new states contain two examples for each of six scenario classes and
cover both player seats:

- enemy contact;
- main-general danger;
- large-stack routing;
- economy/combat conflict;
- counter-capture; and
- mid/late-game consolidation.

Each intervention state has an official construction receipt, round-trip state
hash, scenario assertion, actor/seat label, engine hash, and deterministic
selection receipt. Selection is frozen before any v9 policy probe and cannot
depend on v9 actions or performance.

Every state uses the existing complete canonical macro-action specification and
an exact support cardinality. For deterministic HL action `f(s)`, measurement
uses the document-mandated channel

\[
\widetilde\pi(a\mid s)
=(1-\epsilon)\mathbf 1[a=f(s)]+\epsilon U_s(a),
\qquad U_s(a)=\frac{1}{|A(s)|},
\]

with natural logarithms and epsilon values `0.001`, `0.01`, `0.05`, and `0.1`.
The primary reported epsilon is `0.01`. Any state whose support cannot be
enumerated exactly makes the expanded measurement incomplete; sampling,
truncation, bounds, and approximate replacements are forbidden.

The old legacy-12 v0-to-v8 curve remains unchanged. A new measurement probes
v0 through v9 on the 12 intervention states and probes v9 on the legacy states.
Legacy actions and supports may be reused only after their source, state,
action-space, policy, and result hashes match their immutable receipts.
Expanded-24 results live in a new run and are never appended to the old curve
as though the reference distribution had not changed. The report presents
legacy-12 and expanded-24 side by side.

This epsilon-regularized policy KL measures controlled-state behavioral change
in nats. It is not performance, global policy equivalence, trajectory KL, or
epistemic information gain. A zero value means equal canonical actions on the
measured states only.

## v9 learning boundary and single act

The v9 workspace begins from the verified frozen v7 source. It does not inherit
v8 code. The provider may read only:

- frozen v7 strategy source, tests, strategy description, and experience Skill;
- official rules and the frozen replay-analysis Skill;
- the scientific-attribution report and its explicitly selected diagnosis
  replays, same-state action probes, and critical windows; and
- immutable schemas and interfaces required to produce a runnable candidate.

It may not read formal, validation, or KL outcomes, scores, canonical policy
actions, or post-hoc v9 evidence. The prompt receipt enumerates every readable
artifact and hash; an allowlist and seed-role audit fail closed before the act.

The Framework invokes Codex exactly once. Explainable Python, bounded search,
scoring functions, finite-state control, target selection, rule compression,
and refactoring are allowed. The candidate must integrate or remove rules
rather than merely accumulating patches. It must update `STRATEGY.md` and
`EXPERIENCE.md` with retained, rejected, and newly inferred lessons. Protected
engine, harness, benchmark, attribution, validation, KL, and replay-Skill files
remain immutable in the act workspace.

The ablation policies and v8 remain population/testcase artifacts. They do not
become intermediate versions, alternative parents, or additional acts.

## v9 execution and champion decision

Execution order is fixed:

1. verify frozen authorities and all seed/domain separation contracts;
2. run and validate the paired 2x2 attribution experiment;
3. extract the deterministic diagnosis report and replay diagnostic states;
4. invoke Codex once from frozen v7;
5. run candidate syntax, legality, determinism, latency, and strategy tests;
6. if runnable, freeze v9 source, documentation, patch, manifests, and hashes;
7. run all 12 strongest-human validation games;
8. run all 18 unchanged formal games regardless of validation performance;
9. complete the expanded-24 v0-to-v9 policy measurement; and
10. generate reports, tables, and figures, then evaluate the predeclared
   champion criteria.

A runnable v9 is never suppressed because validation is poor. Conversely,
validation success cannot promote v9 without the complete valid formal result
and all champion criteria. Promotion is an atomic metadata update after all
checks pass. Otherwise v7 remains champion and v9 remains a fully reported
testcase.

## Runs, provenance, and budgets

The work produces separate immutable run families:

1. `generals-scientific-attribution` for the four policies, paired diagnostics,
   replay states, and attribution report;
2. `generals-hl` for the sole v7-to-v9 coding-agent act, candidate, validation,
   and formal evaluation; and
3. `generals-policy-kl-expanded` for legacy-12 verification and expanded-24
   v0-to-v9 policy-change measurement.

Every game retains its official replay, normalized replay, pre-action states,
dense trajectory, action profile, stdout/stderr, process/protocol logs,
termination reason, engine and policy hashes, and event provenance.

The v9 act retains the exact prompt, raw provider JSONL, stderr, token usage,
tool calls, elapsed time, pre/post workspace snapshots, protected-file audit,
source manifest, content hash, and v7-to-v9 patch. Missing provider facts are
recorded as unknown, never zero.

Budget accounting distinguishes coding-agent acts, episodes, environment
steps, game-agent decision steps, primitive commands, tokens, and wall time.
The formal score uses the unchanged benchmark definition. Raw, evo, gain, and
all valid budget-indexed AUC values are reported without interpolating missing
historical score points.

## Failure semantics and data quality

- A malformed, incomplete, crashed, or otherwise invalid game remains invalid;
  no score or dense value is invented for it.
- A provider failure preserves raw logs and consumed budget but creates no v9
  policy version.
- The controller does not silently retry or invoke a second Codex act.
- A candidate that fails runnable-version checks remains candidate evidence and
  receives no fabricated evaluation result.
- A runnable candidate is saved and formally evaluated even when it misses
  validation or champion thresholds.
- Any exact-support failure marks the entire expanded-KL run incomplete; no
  approximate value replaces the missing result.
- Missing diagnostic event classes remain missing and visible in reports.
- A failed or incomplete run cannot modify v7 champion metadata or any frozen
  historical artifact.

Event-quality diagnostics require explicit counts for malformed lines, invalid
events, unknown event types, duplicate event IDs, missing fields, missing event
IDs, and missing run IDs. Reports preserve incomplete and missing states rather
than silently interpolating them.

## Verification strategy

Unit tests cover paired effects, interaction terms, bootstrap determinism,
critical-state selection, same-state action comparison, canonical action
normalization, uniform `U_s`, strict KL, epsilon sensitivity, and atomic
champion decisions.

Asset and contract tests prove:

- exact attribution and validation tuples and both-seat coverage;
- complete separation among historical, attribution, validation, formal, and
  KL roles;
- exact v7 parent identity and a single provider act;
- provider allowlist, evidence receipts, and protected-file integrity;
- official validity and reproducibility of all 12 intervention states;
- exact action support, state hashes, and policy probe determinism;
- unconditional formal evaluation of every runnable v9 candidate; and
- complete version, testcase, and champion lineage.

Integration fixtures exercise attribution, diagnosis extraction, one-act v9
creation, validation, formal evaluation, and expanded KL without replacing the
real run. A bounded smoke test precedes real execution. Final verification runs
the complete Framework suite, Generals asset contracts, and `agentbench data
check`. Every plotted value is traced back to a run ID and source event.

## Reports and figures

Delivery includes:

- a 2x2 attribution table, case-level appendix, dense paired effects, action
  disagreement, and earliest-divergence evidence;
- v9 strategy source, updated experience Skill, patch, hashes, and complete act
  logs;
- validation and formal results with overall, seat, and high/medium/low
  breakdowns;
- v0-to-v9 raw/evo/gain and score-versus-iteration results;
- policy/behavioral information-gain results and iteration curves under their
  explicitly named fixed reference domains;
- legacy-12 and expanded-24 KL curves, epsilon sensitivity, and exact support
  sizes; and
- an English paper-ready figure set plus a conclusion report that distinguishes
  performance failure, measurement incompleteness, and pipeline failure.

All generated data and figures cite immutable run paths, manifest IDs, source
hashes, and measurement-domain identifiers. Both worktrees must be clean at
handoff. Branches remain local until the user separately authorizes a push.
