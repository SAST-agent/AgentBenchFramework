# Generals v7 Champion Challenge Design

Date: 2026-07-31

Status: Approved direction; written-spec review pending

Framework branch: `zhaoyicheng/generals-hl-implementation`

Parent policy: frozen v6 from run `20260729_1653_af8eda26`

Parent content hash:
`974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b`

## 1. Objective and success claim

The objective is to produce an explainable Generals policy that defeats the
frozen strongest runnable human opponent,
`advanced-rank02-robinliu-v18`, under the official engine.

The claim is deliberately stronger than obtaining one lucky win. A candidate
may be called a champion-beating policy only when all of the following hold:

1. the candidate source is frozen and hash-verified before gameplay;
2. the sealed challenge contains ten previously unused seeds and both seats,
   for twenty valid games;
3. the candidate wins at least 11 of 20 sealed games;
4. the candidate wins at least 5 of 10 sealed games from each seat;
5. no sealed replay, result, dense diagnostic, or score entered the candidate
   prompt or any earlier selection decision; and
6. the official engine reports no invalid episode in the sealed suite.

The existing three-seed formal benchmark remains an independent historical
score track. It is not sufficient evidence for the champion claim because its
seeds and outcomes have already been observed during v0-v6 development.

## 2. Empirical motivation

The v6 formal evaluation lost all six high-tier games. This is not explained
by too few army-movement primitives:

| Primitive family across six high games | v6 | champion |
| --- | ---: | ---: |
| Army movement (1) | 2,480 | 2,257 |
| General movement (2) | 0 | 190 |
| General upgrade (3) | 64 | 274 |
| General skill (4) | 0 | 32 |
| Technology (5) | 3 | 24 |
| Super weapon (6) | 0 | 16 |
| Recruit sub-general (7) | 0 | 19 |

The mean v6 terminal margins were -61.3 territory, -3,876.8 army, and
-5,654.2 coins. In five of six games v6 never acquired a sub-general. It
controlled at most two or three resource generals per game, while the champion
eventually controlled all ten resource generals and reached six to ten
sub-generals through capture and recruitment.

The design therefore addresses the missing economy, general, skill, and
weapon capabilities rather than adding another movement-only rule.

These figures are development evidence from the now-retired original formal
seeds. They justify harness and policy architecture work but will not be
reused as sealed claim evidence.

## 3. Alternatives considered

### 3.1 Economy-only patch

Prioritize neutral generals and add command 7 recruitment while retaining the
v6 planner. This is the smallest change, but it leaves general relocation,
skills, and super weapons entirely unanswered. The replay evidence indicates
that this is unlikely to close the champion-scale gap.

### 3.2 Full explainable planner (selected)

Retain the verified v6 safety fallback, add exact transition models and
candidate generators for commands 2, 4, 6, and 7, and select bounded macros
with a deterministic state evaluator and beam search. This is more work, but
it directly targets every observed capability gap and remains inspectable.

### 3.3 Opponent-specific replay templates

Encode hand-written round and map patterns that imitate the champion replays.
This may improve quickly against one opponent, but is more prone to overfit
and produces weaker scientific evidence. Replay observations may inform
general scoring terms, but replay IDs, seeds, or opponent-specific templates
must not become runtime inputs.

## 4. Experimental data partitions

Every listed seed is currently unused by the frozen benchmark, calibration,
v1-v6 learning and validation, or controlled-policy-KL reference manifests.

### 4.1 v7 learning

- Opponent: `advanced-rank02-robinliu-v18`
- Seeds: `290101`, `290202`, `290303`
- Seats: `0`, `1`
- Games: 6
- Policy under evaluation: frozen v6
- Prompt visibility: all six complete normalized learning episodes may enter
  the single v7 Codex act through the frozen Replay Skill and compact,
  state-anchored critical evidence.

### 4.2 v7 validation

- Opponent: `advanced-rank02-robinliu-v18`
- Seeds: `291101`, `291202`, `291303`, `291404`, `291505`, `291606`
- Seats: `0`, `1`
- Games: 12
- Policy under evaluation: frozen v7
- Prompt visibility: forbidden
- Sealed-suite gate: at least 7/12 wins overall and at least 3/6 from each
  seat, with all twelve games valid.

Validation results are development data. A candidate that fails the gate is
saved and reported but never consumes the sealed challenge.

### 4.3 Sealed champion challenge

- Opponent: `advanced-rank02-robinliu-v18`
- Seeds: `292101`, `292202`, `292303`, `292404`, `292505`, `292606`,
  `292707`, `292808`, `292909`, `292999`
- Seats: `0`, `1`
- Games: 20
- Policy under evaluation: the exact frozen v7 source that passed validation
- Prompt visibility: forbidden
- Execution: only after the validation gate passes
- Claim threshold: 11/20 overall and 5/10 from each seat

If the sealed suite is incomplete, no aggregate score or champion claim is
produced. If a complete candidate fails the claim threshold, the suite is
retired; its outcomes may be reported but cannot be used to tune a later
candidate that is evaluated on the same suite.

### 4.4 Historical formal benchmark

The unchanged `generals-hl-pilot-v1` high/medium/low matrix is evaluated for
every runnable v7 candidate, even when validation fails. It preserves the
v0-v7 score curve and checks medium/low retention. Its score is reported
separately from validation and the sealed champion result.

## 5. Policy architecture

### 5.1 Boundaries

The coding agent receives only:

- the exact frozen v6 policy and its retained strategy/experience;
- the official rule summary;
- the frozen Replay Analysis v2 skill;
- six declared high-only v7 learning episodes and their learning-only action
  profile; and
- an explicit implementation contract.

It does not receive validation, sealed, historical formal, calibration, or
policy-KL gameplay payloads. It does not receive champion source code.
Runtime policy input remains only
`choose_actions(round_number, my_seat, view)`.

The agent may edit `strategy.py`, `state_view.py`, `STRATEGY.md`,
`EXPERIENCE.md`, `tests/**`, and modules below `policy/**`. It may not edit
`main.py`, access the network, use randomness, inspect replay IDs or seeds at
runtime, or condition on filesystem paths, wall-clock time, or opponent
identity.

### 5.2 Deterministic state model

The v7 policy owns a private normalized-state clone. Every proposed primitive
is checked and applied to that clone before the next primitive is generated.
The transition model must cover every enabled command's official:

- ownership, level, affordability, cooldown, range, and terrain preconditions;
- army, territory, coin, general, technology, duration, cooldown, and weapon
  effects visible within the current macro; and
- movement budgets and one-army source reserve.

If an enabled command cannot be modeled with known state fields, it is omitted
from that state. An unknown transition terminates the extended search and
falls back to a verified safe prefix; uncertainty never becomes permission.

### 5.3 Candidate generators

The planner exposes focused, independently testable generators:

1. **Army movement:** immediate enemy-main capture, main defense,
   counter-capture, neutral-general capture, consolidation, and valuable
   routing.
2. **General movement:** relocate an owned main or sub-general only to a legal,
   owned, general-free cell when the move improves safety, production
   coverage, or tactical reach.
3. **General upgrade:** exact production, defense, and mobility costs and
   levels, with payback and reserve checks.
4. **General skills:** surprise attack, rout, command, defense, and weaken,
   with exact cooldown, cost, range, target, and frozen-state checks.
5. **Technology:** movement, climbing, swamp immunity, and super-weapon
   unlock, scored against current reachability and remaining horizon.
6. **Super weapons:** legal nuclear, strengthen, transmission, and time-stop
   candidates. High-value targets are selected from visible material,
   generals, main pressure, and clustered armies.
7. **Recruitment:** place a sub-general on an owned, empty, safe cell when its
   expected production/tactical value exceeds the 50-coin cost and required
   safety reserve.

The v6 commands 1/3/5 planner remains an explicit fallback until each new
family passes official-transition parity tests.

### 5.4 State evaluator

Candidate states are scored by a documented weighted vector, in this order:

1. terminal enemy-main capture or own-main loss;
2. immediate own-main pressure and forced opponent access;
3. owned resource/sub-general production capacity;
4. army and defensible territory;
5. coins, recurring income, and upgrade/recruit payback;
6. skill, technology, and weapon readiness;
7. objective distance, terrain exposure, and recapture risk; and
8. deterministic command and coordinate tie-breaks.

The evaluator is phase-aware (opening expansion, economic growth, contact,
and main assault) using only visible state and round number. Phase changes
select documented weight sets; they do not introduce learned opaque
parameters.

### 5.5 Bounded search and latency

The champion's observed high-game macros have a 95th-percentile length of
seven and a maximum of twenty primitives. v7 therefore uses:

- at most eight non-end primitives per macro;
- a fixed deterministic beam width declared in `STRATEGY.md`;
- focused top-k candidates per command family before beam expansion;
- exact duplicate-state elimination; and
- immediate termination after an enemy-main capture or an uncertain
  transition.

The final macro contains exactly one terminal `[8]`. Tests enforce the official
two-second decision limit with headroom on frozen replay states. No timeout
increase is permitted to make the candidate appear valid.

## 6. Framework workflow

The v7 pipeline is a new, explicit round rather than a mutation of v6
artifacts:

1. verify the complete parent v6 run and exact parent content hash;
2. materialize and re-hash v6 into a fresh workspace;
3. run six v6 learning games against the champion;
4. build learning-only replay evidence and one prompt;
5. execute exactly one Codex coding-agent act;
6. run policy tests, deterministic probes, source-scope checks, and manifest
   capture;
7. freeze v7 before all post-act gameplay;
8. run the twelve-game champion validation;
9. run the unchanged eighteen-game historical formal benchmark;
10. run the twenty-game sealed challenge only if validation passed;
11. compute separate learning, validation, formal, and sealed budgets and
    results; and
12. finalize the run, immutable lineage, quality diagnostics, and CI inputs.

Independent matches remain deterministic and artifact-isolated. Parallel
execution is not part of v7: correctness and the champion policy are the
scope, while evaluation concurrency is a separate framework optimization.

## 7. Artifacts and reporting

Every run preserves:

- learning, validation, formal, and sealed case specifications;
- official and normalized replays plus metadata and dense traces;
- replay-skill text, digest, and source reference;
- exact provider prompt, prompt manifest, raw JSONL, stderr, token usage, and
  elapsed time;
- feedback receipt with episodes, decision records, bytes, and omissions;
- v6 and v7 source manifests, content hashes, lineage, tests, and v6-to-v7
  patch;
- action profiles for all phases;
- explicit validation-gate and sealed-claim records; and
- event-quality diagnostics with no interpolation of missing results.

`benchmark_score`, `raw`, `evo`, `gain`, and AUC continue to use only the
unchanged historical formal benchmark. Validation and sealed champion scores
are separate named metrics and never enter the formal benchmark aggregate.
Controlled policy KL and dense metrics are diagnostics, not evidence of a
champion win.

## 8. Failure and recovery semantics

- Invalid or incomplete learning data produces zero coding-agent acts.
- Provider failure preserves its raw receipt and an unrunnable candidate.
- Source-scope, symlink, deterministic-probe, or test failure freezes the
  candidate as invalid and runs no gameplay.
- Validation failure still preserves v7 and runs the historical formal
  benchmark, but it skips the sealed suite.
- Validation incompleteness cannot satisfy the sealed gate.
- A sealed invalid episode removes the sealed aggregate and champion claim.
- Formal, validation, or sealed absence is reported as missing, never zero or
  interpolated.
- Recovery creates a new run, verifies all inherited hashes, and reuses a
  coding-agent act only when an exact frozen post-act v7 source exists.

## 9. Test strategy

### Asset and contract tests

- exact v7 learning, validation, and sealed matrices;
- all seed sets disjoint from every historical frozen set and each other;
- strongest-human-only learning, validation, and sealed opponent;
- ordered dual-seat coverage;
- immutable engine and replay-skill digests.

### Policy requirements in the provider prompt

- focused legal and illegal fixtures for every enabled command family;
- official-transition parity for commands 2, 4, 6, and 7;
- macro length, movement budget, one-final-`[8]`, ownership, affordability,
  cooldown, and deterministic tie invariants;
- fallback behavior on missing or uncertain fields;
- fixed-state latency checks with decision-time headroom.

### Pipeline tests

- exact phase ordering and one coding-agent act;
- prompt leakage rejection for all non-learning seeds and payload labels;
- freeze-before-play and isolated-workspace hashes;
- validation gate, unconditional historical formal evaluation, and
  conditional sealed execution;
- no aggregate on incomplete/invalid sealed results;
- per-seat and overall champion thresholds;
- provider, test, validation, formal, sealed, and repeated recovery paths;
- budget and event-quality completeness.

### Real verification

A real v7 run must include:

1. all Framework and Generals asset contract tests passing;
2. six valid v6 learning games;
3. one auditable Codex act;
4. a runnable hash-frozen v7;
5. twelve valid champion validation games;
6. eighteen valid historical formal games;
7. twenty valid sealed champion games if and only if validation passed; and
8. a champion claim only when the frozen thresholds are satisfied.

## 10. Completion boundary

Implementing the v7 harness is not completion. Producing a legal v7 is not
completion. Improving KL, dense metrics, or the historical formal score is not
completion.

The objective is complete only when a frozen policy satisfies the sealed
twenty-game overall and per-seat thresholds against the champion, with all
required provenance and quality checks. If v7 fails, it remains a valid
iteration asset and the same protocol advances to a fresh-seed v8 rather than
weakening the claim.
