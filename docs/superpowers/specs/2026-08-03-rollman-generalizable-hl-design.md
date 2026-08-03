# Rollman Generalizable HL Design

## Objective

Build a reproducible Rollman HL loop that learns conditional, replay-grounded lessons across iterations and passes both rank15 and rank16 on at least four of five sealed certification seeds per opponent without materially regressing the training pool.

## Scientific invariants

- Static rules, decision-space definitions, and replay-reading instructions remain content-hashed frozen context.
- Strategy code may use observable-state conditionals, finite memory, graph search, path planning, opponent prediction, and other interpretable mechanisms.
- Strategy code must not branch on seed, replay identity, absolute replay coordinates, human identity, or hidden opponent source.
- Four candidates are siblings from one explicit search parent. The run remains a linear search lineage rather than a four-ary tree.
- Candidate promotion uses comparable matches on shared seeds. Sealed certification seeds never enter prompts or training memory.
- Framework-measured outcomes are authoritative. Model-authored explanations cannot overwrite match facts.

## Experience architecture

`experience/ledger.jsonl` is an append-only source of empirical facts. Each completed proposal cycle records one outcome per branch representative, including:

- iteration, act, branch, parent, candidate, opponent, and seed identifiers;
- the branch's observable activation condition, mechanism, and preservation contract;
- activation count and decision count;
- parent and candidate result, score margin, and margin delta on comparable matches;
- evidence replay and trace references;
- a framework-owned verdict: `verified_good`, `verified_bad`, `mixed`, `inconclusive`, or `invalid`;
- whether the candidate became the next search parent.

`experience/SKILL.md` is a deterministic bounded projection of the ledger plus replay-grounded candidate notes. It contains five sections:

1. Verified good conditions and mechanisms.
2. Verified bad conditions and mechanisms.
3. Mixed or scope-sensitive findings.
4. Replay-grounded observations and open questions.
5. Current target failure profile for rank15 and rank16.

Every planner, candidate, and repair act reads the active Skill. Candidate-authored notes remain provisional until the Framework joins them with evaluation results. The ledger retains exact facts; Skill compaction may merge duplicate semantic findings but must retain condition, mechanism, verdict, and measured effect.

## Stratified k=4 rollout

Each cycle assigns distinct roles and evidence packets:

- Branch 0 — rank15 loss-source suppression: diagnose the largest Ghost scoring source in rank15 failures.
- Branch 1 — rank16 offensive progress: increase Rollman score or level completion on rank16 while preserving survival.
- Branch 2 — cross-replay opponent modelling: distill coordinate-independent Ghost action patterns and implement an interpretable best response.
- Branch 3 — generalization and consolidation: use a different failure cluster to repair the scope of an existing mechanism or combine compatible verified lessons.

Each branch receives the shared compact Skill and digest but only its assigned bounded replay summaries. It may inspect at most two exact trace windows. Thus one cycle can inspect up to eight targeted windows without copying full replays into prompts. All four candidates are quick-screened on common cases so their outcomes are comparable.

## Seed and opponent split

The evaluator maintains three disjoint seed roles:

- Training seeds rotate across cycles and may contribute replay evidence and Skill observations.
- Validation seeds are shared by finalists and may contribute evidence only after being retired from validation and replaced.
- Certification seeds are generated or configured as a sealed set of five seeds per opponent. They never appear in planner, candidate, reducer, repair, research-state, or Experience inputs.

The active curriculum covers rank15 and rank16 together. A candidate can advance the search lineage only when it improves the robust lexicographic objective across the two opponents or preserves one while materially improving the other. Historical checkpoints retain the official champion, the current stable search parent, the best rank15 specialist, and the best rank16 specialist, but every k=4 cycle still has one common parent.

Certification succeeds only when each opponent independently has at least four wins in five valid matches. Incomplete matches, timeouts caused by infrastructure, and opponent faults do not count as wins and must be retried or reported separately.

## Data flow

1. Select one explicit parent from the stable archive.
2. Evaluate or load rotating training evidence for rank15 and rank16.
3. Build four branch-specific evidence packets and one four-role planner request.
4. Run four sibling coding acts and bounded activation probes.
5. Run common quick-screen cases, then shared validation cases for finalists.
6. Select the next search parent with a robust two-opponent comparison and retain rollback checkpoints.
7. Append deterministic branch outcomes to the experience ledger.
8. Run the reducer on complete measured outcomes, then regenerate `SKILL.md`.
9. Feed the regenerated Skill into the next planner and all four coding acts.
10. Run sealed certification only when the candidate reaches the configured readiness gate.

## Failure handling

- Compile, policy-probe, malformed output, access-policy, and zero-activation failures receive an `invalid` ledger verdict.
- A branch that improves some common cases and regresses others receives `mixed`; the Skill records both good and bad conditions.
- Missing parent comparisons yield `inconclusive`, never verified improvement.
- Failed candidates cannot become parents, but their measured failure facts remain available to prevent repetition.
- Reducer failure cannot erase or alter ledger facts; Skill regeneration remains deterministic from the last valid state.
- Rollback selects a retained stable checkpoint rather than automatically returning to the origin.

## Verification

Unit tests cover ledger validation, deterministic ordering, secret rejection, verdict derivation, Skill projection, branch-role evidence isolation, seed disjointness, two-opponent selection, and certification arithmetic. Controller tests prove that ledger and Skill updates occur after complete evaluation and include all four branch representatives. An end-to-end fixture proves that a resumed cycle reads the regenerated Skill. The Rollman experiment is successful only after rank15 and rank16 each pass four of five sealed valid matches.
