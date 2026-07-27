# Generals HL v3 Pilot Rescue Design

**Date:** 2026-07-28  
**Status:** Direction approved by the user on 2026-07-28  
**Parent:** completed pilot v2 snapshot
`1e8a2ce4fa38b4171f6f8d77d42b80d48438f5213b08d543714d8241e292c565`

## Goal

Produce one scientifically auditable `v2 → v3` rescue iteration tonight. This
run diagnoses and repairs the existing pilot lineage; it is not presented as
the newly agreed strict from-scratch default experiment.

The run should answer:

1. Can replay feedback from the strongest runnable human opponent cause a real
   strategy change?
2. Does the changed strategy use non-main frontier armies instead of repeatedly
   streaming one or two armies from the main-general cell?
3. Does v3 improve any paired dense progress metric on its learning matrix?
4. Does that progress translate to the unchanged 18-case formal benchmark?

## Considered Approaches

### Selected: rescue the existing v2 with one architecture-level Codex act

Import the exact v2 snapshot, generate new high-tier learning games, give Codex
a compact root-cause diagnosis and permission to rewrite the policy
architecture, gate the candidate on observable behavior, and only then run the
frozen formal benchmark.

This preserves the requested v3 lineage and can produce a result tonight.

### Deferred: restart the canonical default experiment from scratch

This is the correct next main experiment, but it would create a new `v000`
lineage rather than the requested v3. Its v0 will contain only the official
SDK, legal-action interface, and minimal end-turn behavior.

### Rejected: manually patch the BFS rule

A researcher-authored strategy change could improve the score, but it would not
measure a Codex HL act and would confound attribution.

## Frozen Learning Contract

Add a separate learning manifest:

```toml
learning_id = "generals-hl-v3-strongest-v1"
opponent_id = "advanced-rank02-robinliu-v18"
seeds = [284101, 284202, 284303]
seats = [0, 1]
```

The strongest runnable human source is the already frozen high-tier opponent.
The learning seeds are declared before play and do not overlap formal,
round-1, round-2, calibration-development, or calibration-held-out seeds.

The same six cases are run for:

- v2 replay generation before the coding-agent act;
- v3 validation after the coding-agent act.

Only v2 learning artifacts and summaries may enter the Codex prompt. Formal
evaluation seeds, outcomes, trajectories, and opponent source files remain
absent.

## Editable Policy Boundary

Codex may:

- replace or compress existing rules;
- delete obsolete rules;
- implement deterministic search, scoring, planning, or state machines;
- edit `strategy.py`, `STRATEGY.md`, `EXPERIENCE.md`, `tests/**`, and
  Python modules under `policy/**`.

Codex may not edit `main.py`, `state_view.py`, SDK/engine code, manifests,
opponents, evaluation code, or saved replays. It may not use network, wall
clock, randomness, opponent-specific source knowledge, hidden evaluation
material, or opaque learned weights.

`strategy.choose_actions(round_number, my_seat, view)` remains the sole runtime
entry point. Explainability means the chosen action can be traced to readable
source logic and state-derived scores; it does not mean the policy must be a
flat if/else list.

## Replay Feedback and Experience

The prompt is capped at 64 KiB and contains:

- current v2 strategy documentation;
- official-rule summary and replay field guide;
- paired dense summaries from six strongest-human learning games;
- externally measured v2 decision-class counts;
- the observed dominant-main-streaming diagnosis;
- representative learning decisions without full board dumps;
- explicit permission to restructure the policy.

The prompt does not say “make one coherent improvement.” It asks for a
cohesive redesign with tests.

Codex must create or update `EXPERIENCE.md` with:

- evidence-backed observations and replay IDs;
- retained strategic hypotheses;
- rejected or failed hypotheses;
- the structural changes made in v3;
- risks that future iterations should check.

This file is versioned with the policy. In the future strict from-scratch
experiment it persists only inside one lineage and is reset for a new run.

## Observable Decision Classes

Framework classifies the first non-end command of each macro-action as:

- `end_only`;
- `main_army_move`;
- `non_main_army_move`;
- `general_move`;
- `general_upgrade`;
- `skill`;
- `technology`;
- `super_weapon`;
- `recruit`;
- `other`.

For army movement, the source coordinate is compared with the acting seat's
main-general coordinate in the probe state. This is an external observable
classification, not a claim about which internal Python function caused the
action.

## Candidate Gate

v3 is eligible for formal evaluation only if all conditions pass:

1. provider invocation completed;
2. only editable policy files changed;
3. strategy tests passed;
4. action disagreement against v2 on the complete v2 learning probe set is
   greater than zero;
5. v3 produces at least one `non_main_army_move` on that probe set;
6. all six v3 strongest-human validation games are valid;
7. at least one paired mean improves among completed rounds survived,
   terminal territory share, terminal army margin, terminal coin share, and
   terminal net-main pressure.

The gate writes an append-only `behavior_gate` event containing every
condition, counts, metric deltas, and the final decision. A failed gate saves
the candidate, provider logs, tests, probes, and budgets but does not run the
formal benchmark.

## Formal Evaluation and Measurement

If the gate passes, run the unchanged `generals-hl-pilot-v1` 18-case matrix:
three human tiers, three frozen formal seeds, and both evaluated-agent seats.

The summary reports:

- `raw_score`, `evo_score_1`, `evo_score_2`, `evo_score_3`;
- `gain_3 = evo_score_3 - raw_score`;
- per-tier and per-seat formal results;
- learning and evaluation budgets;
- v2/v3 action disagreement and decision-class distributions;
- paired learning dense deltas;
- provider JSONL, usage, tool calls, time, source snapshots, and patch.

The failed pre-thread act in the earlier recovery lineage remains a missing
score point. The global score AUC must remain missing across that gap; a local
post-recovery AUC may be shown only when explicitly labeled partial.

Calibration remains separate and is not reopened or included in formal score.

## Error Handling

- Invalid parent hash: stop before playing or invoking Codex.
- Invalid/incomplete learning case: stop before Codex and retain partial
  artifacts with missing aggregate feedback.
- Provider failure: save the failed act and unchanged candidate snapshot.
- Test/protected-file failure: mark v3 invalid and skip gameplay.
- Behavior-gate failure: retain v3 as rejected and skip formal evaluation.
- Incomplete formal matrix: keep per-game results and set aggregate score,
  gain, and associated AUC point to missing.

## Verification

Automated tests cover:

- learning-manifest validation and seed separation;
- strongest-human case construction;
- high-tier prompt acceptance with formal-seed rejection;
- prompt byte limit and experience-file requirement;
- decision-class counting;
- every behavior-gate condition;
- exact v2 lineage import;
- pipeline success and gate-failure paths;
- CLI arguments, known event types, and report v3 history.

The live run must additionally verify source hashes, strategy tests, six
learning validations, 18 formal cases when gated, event quality, and a clean
Framework worktree after committing implementation changes.

