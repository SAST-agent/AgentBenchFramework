# Generals HL v6: strongest-opponent macro-planner design

Date: 2026-07-29

Status: design approved; written specification pending user review

Experiment line: legacy `v5 -> v6` pilot

## 1. Purpose

Generals HL v6 will test whether a replay-guided coding agent can improve the
existing v5 policy against medium and high human algorithms by replacing its
single-primitive turn policy with a bounded, explainable macro-action planner.

The formal success criterion is:

- medium opponent: at least 2 wins in 6 formal games;
- low opponent: retain 6 wins in 6 formal games;
- high opponent: at least 1 win in 6 formal games is an additional
  breakthrough, not a hidden acceptance gate.

Every runnable v6 candidate is versioned and formally evaluated even when it
misses these thresholds. The framework must not select, hide, or relabel a
candidate based on formal results.

This is a continuation of the legacy v3-v5 pilot and therefore uses v5 as its
parent. It does not redefine the canonical future default, which remains a
from-scratch policy learned from the strongest human opponent.

## 2. Evidence motivating v6

The saved v5 formal result is 7/18: high 0/6, medium 1/6, and low 6/6.
Separate strongest-opponent learning games show two recurring loss modes:

1. rapid main-general loss while coins or local army are still available;
2. long games that end with large army and economy deficits.

The v5 source and learning traces identify a structural action-throughput gap:

- v5 emits at most one primitive command per turn;
- the strongest human opponent averages approximately 1.6-3.0 primitive
  commands per turn in the inspected learning games;
- that opponent uses as many as 12 primitives in a turn and combines army
  movement, general movement, upgrades, skills, technology, weapons, and
  recruitment;
- v5 mostly emits army moves, performs only one or two main-production
  upgrades per game, and often continues expanding neutral plain cells late
  in losing games.

The v6 hypothesis is therefore narrower than "add more rules": a conservative
sequential planner, plus value-aware economy and routing, should use the
official per-turn action budget more effectively without sacrificing
legality, determinism, or interpretability.

## 3. Considered approaches

### A. Retune the existing single-step scoring

This is the smallest code change and has the lowest legality risk. It cannot
close the observed primitive-action throughput gap, so it is rejected as the
primary v6 design.

### B. Strongest-opponent learning with a bounded macro planner

This preserves the adopted rule that HL learns from the strongest human,
addresses the largest observed structural gap, and remains explainable. This
is the selected design.

### C. Mixed high/medium learning curriculum

This could target the medium score more directly, but it changes the
strongest-opponent default and makes the source of improvement harder to
attribute. Medium remains a validation and formal-evaluation opponent, not a
v6 learning opponent.

## 4. Data isolation and experiment protocol

### 4.1 Immutable inputs

- Parent source: the exact saved v5 version identified by its manifest hash.
- Official logic and protocol: the repository's frozen Generals game logic.
- Human opponents:
  - high: `advanced-rank02-robinliu-v18`;
  - medium: `advanced-rank08-nashjunheng-v20`;
  - low: `popular-rank16-xiaoaojianghu-v1`.
- Formal benchmark manifest: unchanged from v5.

The v6 command must verify the expected v5 source hash before any game or
provider invocation.

### 4.2 Learning split

Run v5 against only the high opponent using fresh seeds:

- `287101`, both seats;
- `287202`, both seats;
- `287303`, both seats.

These six games produce the only gameplay trajectories that the v6 Codex act
may read. The prompt may include official rules, the replay-analysis skill,
v5 source and documentation, learning replays, learning-only dense summaries,
and learning-only critical windows.

### 4.3 Validation split

After Codex creates v6 and local tests pass, run v6 on a fresh, non-formal
validation split:

- opponents: high and medium;
- seeds: `288101`, `288202`, `288303`;
- both seats.

This produces 12 validation games. Validation results diagnose generalization
but do not decide whether v6 is saved or formally evaluated.

### 4.4 Formal split

Run v6 on the existing immutable 18-game formal benchmark:

- high, medium, and low opponents;
- the existing three formal seeds;
- both seats.

No formal replay, trajectory, critical window, dense trace, action profile, or
outcome may be present in the provider prompt. Formal events and diagnostics
are generated only after the v6 source has been frozen.

Historical v5 formal results may be used in the final human-facing comparison,
but are not learning data for the v6 Codex act.

## 5. v6 policy design

Codex is allowed to refactor the policy instead of monotonically appending
rules. The resulting policy must remain deterministic and explainable.

### 5.1 Normalized state

`state_view.py` may be extended with official state already visible to the
player and required for legality or combat estimates, including:

- remaining army movement steps;
- general skill cooldowns and active skill durations;
- weapon cooldowns and active weapon fields;
- existing general production, defense, mobility, coin, and technology
  levels.

Normalization must use stable primitive JSON-compatible values and stable
ordering. It must not read opponent identity, replay ID, seed, filesystem
state, wall clock, network state, or future information.

### 5.2 Bounded sequential planning

The planner operates on a cloned normalized state. It selects a primitive,
applies a conservative local transition to the clone, then evaluates the next
primitive. This prevents the second action from assuming the original army or
coin totals.

The v6 required command surface is deliberately limited to:

- command 1: army movement;
- command 3: production/defense/mobility upgrades when exact affordability
  and ownership are known;
- command 5: technology upgrades when exact affordability and level
  preconditions are known.

Commands 2, 4, 6, and 7 are optional only if the implementation can establish
their complete official legality from normalized state and cover that logic
with tests. They are not required for v6 success.

The macro has these bounds and invariants:

- no more army moves than the current official remaining movement allowance;
- at most one development purchase before movement planning unless a tested
  transition model proves multiple purchases remain legal;
- stop planning when the next transition is uncertain;
- terminate with exactly one `[8]`, with no earlier end command;
- never emit a zero/negative movement or a movement from a non-owned stack;
- deterministic row/column and command tie-breaking.

### 5.3 Ordered objectives

The planner uses three explicit layers:

1. **Main safety.** Detect immediate adjacent capture pressure, prefer a legal
   counter-capture or reinforcement, and preserve an explainable local
   reserve. Do not spend coins before an actionable urgent defense.
2. **Economy and force growth.** Evaluate main and captured resource-general
   production upgrades using exact costs, safety, expected remaining horizon,
   and a coin reserve. Consider army-movement technology when the policy can
   use the additional movement budget. Avoid indefinite unplanned coin
   hoarding.
3. **Strategic movement.** Score destinations by enemy-main, enemy-sub,
   resource-general, enemy-territory, owned-consolidation, and neutral-plain
   value. Include defender cost, remaining attacking surplus, route distance,
   swamp/terrain cost, and main exposure. Neutral plain expansion must lose to
   a reachable higher-value objective unless it is a necessary route step.

Shortest geometric distance alone is not an adequate objective. A
combat/value-aware route search or an equivalent documented deterministic
scoring method is required.

### 5.4 Compression and experience

`STRATEGY.md` must describe the final compact policy hierarchy and invariants.
`EXPERIENCE.md` must distinguish:

- replay-backed observations;
- hypotheses tested by v6;
- retained v5 behavior;
- rejected or compressed v5 behavior;
- remaining risks.

Claims must cite learning replay IDs/state IDs. Validation and formal evidence
is added only after the provider act, outside the learned strategy experience.

## 6. Framework workflow

Add a first-class `generals iterate-v6` workflow consistent with prior
Generals iterations:

1. validate manifests, opponent assets, parent run, and expected v5 hash;
2. create a new immutable run directory;
3. save run configuration and learning/validation/formal specs;
4. execute the six v5 strongest-opponent learning games;
5. generate learning-only dense metrics, critical windows, and action
   diagnostics;
6. build and hash the bounded provider prompt;
7. invoke non-interactive Codex once and save raw provider artifacts;
8. validate the workspace diff and run candidate unit tests;
9. freeze v6 source, manifest, hash, parent link, and v5-to-v6 patch;
10. run the 12-game high/medium validation split;
11. unconditionally run the unchanged 18-game formal benchmark;
12. compute score, raw/evo/gain/AUC, information-gain records, budget records,
    action diagnostics, and quality diagnostics;
13. save a result document that clearly separates learning, validation, and
    formal evidence.

Add `generals recover-v6` if the existing recovery abstraction cannot safely
resume a failed provider act or post-act evaluation while preserving the same
run lineage and prompt hash.

## 7. Logs and saved assets

The v6 run must preserve:

- `run.toml`, quality report, summary, and event JSONL;
- learning, validation, and formal spec JSON;
- official replay, normalized replay, dense trace, dense summary, metadata,
  process record, protocol bytes, and stderr for every game;
- provider prompt text/JSON/manifest/hash, raw provider JSONL, stderr, token
  usage, tool calls, and elapsed time;
- v5 and v6 source snapshots, manifests, source hashes, tests, test logs, and
  exact patch;
- validation and formal score breakdowns by opponent, seat, and seed;
- decision-space/information-gain and budget events already required by the
  framework.

New action diagnostics must include, per version and split:

- primitive commands per turn: mean, maximum, and multi-command-turn count;
- command-code histogram;
- end-only turn count;
- production/technology upgrade count;
- moves into enemy, neutral-general, neutral-plain, and owned cells;
- neutral-plain movement ratio.

These diagnostics describe behavior change; they must not be mislabeled as
causal information gain or policy KL.

## 8. Failure handling

- A provider crash or timeout preserves the partial run and raw logs.
- A source-hash mismatch stops before learning games.
- Candidate test failure preserves the candidate workspace and test output;
  recovery must not silently replace it.
- Illegal official actions are valid strategic losses and remain in the
  reported score.
- Process crashes, protocol damage, or harness timeouts remain invalid
  experiments under the existing contracts.
- Missing or malformed events remain visible in quality diagnostics and are
  never interpolated.
- Validation or formal regression never deletes v6 or rewrites it as a
  different version.

## 9. Verification strategy

Implementation follows test-driven development:

1. failing CLI/config tests for `iterate-v6` and its split/seed defaults;
2. failing prompt-isolation tests proving only learning artifacts are passed
   to Codex;
3. failing lineage/hash/recovery tests;
4. failing action-diagnostic unit tests;
5. candidate strategy tests for macro bounds, sequential state updates,
   legality, determinism, main defense, resource upgrades, valuable routing,
   and exact final end command;
6. focused tests, then the complete Generals/framework suite;
7. a cheap non-provider smoke run where supported;
8. the real Codex act, validation run, and formal evaluation;
9. final artifact and event-quality audit before publishing conclusions.

No performance claim is made until the saved v6 formal summary and replay
artifacts have been checked against the immutable formal spec.

## 10. Expected interpretation

If v6 improves medium/high score while preserving low performance, the result
supports the limited claim that replay-guided HL benefited from an
action-throughput and value-planning refactor in this Generals environment.
It does not establish that macro planning alone caused the gain.

If scores do not improve but action diagnostics show more legal, valuable
multi-command turns, v6 is still informative: the next bottleneck is likely
combat/skill modeling or opponent anticipation rather than primitive-action
throughput. If both score and action quality regress, the v5 rollback point
remains intact and the v6 evidence is retained as a rejected policy
hypothesis.
