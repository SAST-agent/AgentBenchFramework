# Generals HL v5 Rollback-Guided Iteration Design

**Date:** 2026-07-29  
**Status:** Design direction approved by the user on 2026-07-29; written spec
awaiting final review  
**Lineage parent run:**
`agentbench_data/runs/28_generals/generals-hl/20260728_1519_7b242596`  
**Lineage parent version:** `v4`  
**Expected v4 content hash:**
`5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f`  
**Rollback source version:** `v3` inside the v4 parent run  
**Expected v3 content hash:**
`a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`

## Purpose

Run one real rollback-guided heuristic-learning act that keeps v4 in the
lineage and experiment ledger while using retained v3 source as the editable
starting policy. The act learns from new, paired v3/v4 replays against the
strongest frozen human opponent, produces v5, and formally evaluates every
runnable v5 on the unchanged 18-case pilot benchmark.

The engineering success condition is a complete, reproducible v5 point with
all raw logs, replay evidence, rollback metadata, version snapshots, and
phase-separated budgets. The performance target is to exceed the historical
v3 score of `7/18 = 38.89%`, but performance improvement is not a gate.

This is still a legacy pilot-rescue lineage. Repeated formal evaluation makes
it unsuitable for a clean final test claim. The later canonical experiment
must use the agreed from-scratch default and a separately held-out test
population.

## Selected Approach

The lineage parent remains v4 because v4 was a real coding-agent act and formal
evaluation. Its source and negative result must not disappear. The editable
workspace starts from the immutable v3 snapshot already embedded in the v4
run. This is an explicit rollback event, not a rewrite of v4.

The experiment-design rollback decision is supported by the already separated
v4 learning/validation diagnostics. The coding-agent prompt receives v3/v4
source and retained experience plus only the newly declared v5 learning
episodes as new trajectory evidence. Exact formal outcomes, formal seeds, and
formal trajectories are never placed in the prompt.

Two alternatives were rejected:

- A narrow v4 patch could preserve the double-counted reserve architecture and
  its high end-only rate.
- A full v4 rewrite would make it harder to distinguish a repaired hypothesis
  from another unrelated architecture replacement.

## Frozen Experimental Flow

```text
immutable v4 lineage parent
  -> verify embedded immutable v3 rollback snapshot
  -> run v3 and v4 on six new strongest-human learning cases
  -> extract compact paired critical windows
  -> start editable workspace from v3
  -> human replay skill + v3/v4 code/experience + paired evidence
  -> one Codex coding-agent act
  -> runnable v5 snapshot and tests
  -> run v5 on the same six cases for three-way paired diagnostics
  -> unchanged 18-case formal benchmark
  -> immutable logs, budgets, result, and report
```

## Learning and Formal Data Boundaries

Add a separate asset manifest:

```toml
learning_id = "generals-hl-v5-rollback-strongest-v1"
opponent_id = "advanced-rank02-robinliu-v18"
seeds = [286101, 286202, 286303]
seats = [0, 1]
```

The six cases are declared before play and are disjoint from:

- formal seeds `280101`, `280202`, `280303`;
- legacy learning seeds `281101`, `281202`, `281303`;
- calibration development and held-out seeds;
- v3 learning seeds `284101`, `284202`, `284303`;
- v4 learning seeds `285101`, `285202`, `285303`.

Before the act, run both frozen v3 and frozen v4 on all six cases. This is 12
learning episodes. After the act, run v5 on the same six cases as validation.
New episode-level records in the prompt may come only from the pre-act v3/v4
games in this learning suite. Post-act v5 validation is evidence for future
acts, not the completed v5 act.

Formal opponent source, formal seeds, formal outcomes, and formal trajectories
remain unavailable to Codex. The formal 18-case matrix is opened only after a
runnable v5 snapshot has been saved.

## Rollback Contract

The v5 CLI accepts:

- the v4 parent run and expected v4 hash;
- the expected embedded v3 rollback hash;
- the frozen v5 learning manifest;
- the frozen replay-analysis skill;
- the audited v4 campaign-budget receipt;
- an optional finalized zero-act v5 prompt attempt for cumulative-budget
  recovery.

The pipeline verifies that:

1. the parent run is complete and its v4 hash matches;
2. the parent contains immutable v3 and v4 source snapshots;
3. the embedded v3 content hash matches the approved rollback hash;
4. protected `main.py` and `state_view.py` agree across v3 and v4;
5. the campaign receipt references and hashes the exact v4 parent summary;
6. the receipt's audited `after` budget is used as the global starting point.

The run records:

- `parent_run_id` and `parent_version = "v4"`;
- `starting_version = "v3"`;
- `rollback_source_version = "v3"`;
- rollback source and parent hashes;
- an explicit `version_rollback` event;
- immutable v3, v4, and v5 snapshots;
- both `v3-to-v5.patch` and `v4-to-v5.patch`.

No historical run, summary, snapshot, or campaign receipt is mutated.

## Paired Replay Evidence

Reuse the frozen human-authored
`skills/replay-analysis-v1/SKILL.md`. Copy its exact bytes and hash into the
run.

For each of the six cases and for each of v3 and v4, select at most six
chronological, unique decisions:

1. first evaluated-agent decision;
2. first non-end action;
3. first observable main-general pressure;
4. decision before the steepest territory-share drop;
5. decision before the steepest army-margin drop;
6. final evaluated-agent decision.

Missing criteria create no synthetic record. Every serialized decision keeps
the v4 compact strategic features:

- own and visible enemy main positions and armies;
- adjacent and one-step-reachable main pressure;
- territory, total army, army margin, and coin share;
- owned general counts and visible levels;
- movable-stack count and the four largest movable stacks;
- at most six opportunity-relevant strategic general targets;
- observable decision class and exact macro-action.

The prompt groups v3 and v4 evidence by case and labels divergent trajectories
as separate observations, never as aligned states. It includes three-way
aggregate context only from the learning suite: outcomes, dense summaries,
decision classes, and v4-minus-v3 deltas.

Every one of the 12 declared pre-act episodes must be represented. Record exact
episode IDs, selected decision IDs, omission counts, serialized bytes, total
prompt bytes, and estimated tokens. If a byte cap cannot include every
episode, finalize as `prompt_incomplete` before provider invocation.

## Codex Act Contract

The editable workspace is a byte-for-byte copy of v3 source. The prompt also
contains the relevant frozen v4 strategy and experience as a failed candidate,
the paired learning evidence, official rules, and the replay skill.

Codex may edit:

- `strategy.py`;
- `STRATEGY.md`;
- `EXPERIENCE.md`;
- `tests/**`;
- Python modules under `policy/**`.

Codex may delete, merge, compress, or replace rules and may use deterministic,
explainable search, scoring, planning, or state machines. It may not edit
`main.py`, `state_view.py`, manifests, replays, framework code, opponents, or
evaluation code. It may not use randomness, wall clock, network access, hidden
files, opponent source knowledge, formal evaluation material, or opaque
learned weights.

The prompt asks Codex to preserve v3's active global-stack enumeration while
testing a single coherent reserve calculation. In particular, it must inspect
the v4 failure hypothesis that adjacent hostile pressure was subtracted once
inside `_surplus()` and compared a second time inside urgent defense. It must
avoid turning an inconclusive defense calculation into a broad unconditional
end-turn rule.

The prompt requires targeted tests for:

- a main general with one defeatable adjacent threat;
- a main threat that cannot be defeated but can be reinforced;
- no double counting of the same visible hostile army;
- continued safe capture and routing from non-main stacks;
- deterministic output and exactly one final end command.

Codex must update `EXPERIENCE.md` with learning replay/state-ID citations,
retained and rejected v3/v4 hypotheses, v5 structural changes, and remaining
risks.

## Version and Evaluation Semantics

A v5 is runnable when:

1. the provider invocation completes;
2. changed files remain inside the editable policy boundary;
3. strategy tests pass;
4. `EXPERIENCE.md` exists.

Every runnable v5 is saved, validated on all six learning cases, and formally
evaluated on all 18 unchanged benchmark cases. Action disagreement, dense
improvement, learning wins, and formal score improvement are diagnostics only.
They cannot reject a runnable version, suppress its score, or rewrite v3/v4.

If v5 is weaker, the run remains complete and v3 remains the best retained
historical version. Rollback is an input-selection mechanism for a later act,
not post-evaluation deletion or automatic score replacement.

## Measurement

Report:

- `raw_score`, `evo_score_1` through `evo_score_5`;
- `gain_5 = evo_score_5 - raw_score`;
- per-tier and per-seat formal results;
- v3/v4/v5 outcomes and dense metrics on the six learning cases;
- v5-minus-v3 and v5-minus-v4 paired survival, territory, army, coin, and
  main-pressure deltas;
- v3/v4/v5 action classes on the frozen paired probe states;
- v5-versus-v3 and v5-versus-v4 action disagreement.

Action disagreement remains behavior change, not policy KL or epistemic
information gain. Strict policy KL and information gain stay missing because
the deterministic strategies expose macro-actions rather than a complete
probability distribution over the legal decision space.

The score curve preserves every point:
`[v0, v1, failed act, v2, v3, v4, v5]`. The historical failed-act gap remains
missing, so global AUC remains unavailable and is never interpolated.

## Budget and Log Contract

The current v5 run separates:

- learning: 12 pre-act v3/v4 episodes, prompt construction, one Codex act;
- validation: 6 post-act v5 episodes;
- evaluation: 18 formal v5 episodes;
- total: the explicitly labeled sum of all available phases.

The audited v4 campaign receipt starts v5 at 5 acts and 58 learning episodes.
A successful v5 therefore reaches 6 acts and at least 70 cumulative learning
episodes, with exact environment-step, decision, primitive-command, token, and
time values determined by the run. Historical cumulative token usage remains
missing if any prior failed act lacks token accounting.

Every run, including pre-act and provider failures, independently saves:

- `events.jsonl`, `summary.json`, `quality.json`, and `run.toml`;
- all match metadata, replays, protocol streams, and stderr;
- the replay skill, learning/formal specifications, and prompt manifest;
- exact prompt text, raw provider JSONL, provider stderr, token/tool/time usage;
- workspace diff, editable workspace, version manifests, source snapshots, and
  test logs;
- phase and cumulative budgets plus event-quality diagnostics.

Summaries are finalized once. Later corrections use a separate derived,
source-hashed receipt.

## Error Handling

- Parent, rollback, or receipt hash mismatch: stop before gameplay.
- Learning seed overlap or wrong opponent: reject configuration.
- Incomplete v3 or v4 pre-act suite: save partial matches and stop before
  Codex.
- Prompt omission or byte-cap failure: finalize `prompt_incomplete`, record a
  zero-act learning budget, and do not invoke Codex.
- Provider failure: retain raw output and a missing v5 score.
- Protected-file change or test failure: save the candidate as invalid and do
  not execute it.
- Incomplete v5 validation: retain diagnostics and still run formal evaluation
  when the strategy itself is runnable.
- Incomplete formal matrix: retain individual cases while aggregate score,
  gain, and dependent metrics remain missing.

## Verification

Automated tests must prove:

- exact v4 parent and embedded v3 rollback hash validation;
- protected-file equality and byte-for-byte v3 workspace initialization;
- v5 learning manifest identity, strongest opponent, both seats, and complete
  seed disjointness;
- complete paired v3/v4 learning execution and phase budgets;
- deterministic paired critical-window selection and prompt leak resistance;
- all 12 episodes represented within the prompt contract;
- explicit rollback events, lineage fields, dual patches, and three snapshots;
- provider edit boundary and runnable-version classification;
- v5 formal evaluation independent of every diagnostic value;
- three-way paired dense and behavior diagnostics;
- audited campaign-budget inheritance and optional zero-act recovery;
- missing AUC/KL/information gain semantics;
- CLI, event-quality, and Dashboard support.

The real run must additionally verify:

- 6/6 valid v3 and 6/6 valid v4 learning games;
- one completed Codex act and passing strategy tests;
- 6/6 valid v5 validation games;
- 18/18 valid formal v5 games;
- exact prompt, replay-skill, version, summary, and receipt hashes;
- zero malformed, unknown, duplicate-ID, or missing-ID events.
