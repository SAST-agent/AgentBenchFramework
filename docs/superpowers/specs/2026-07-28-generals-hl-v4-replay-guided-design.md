# Generals HL v4 Replay-Guided Iteration Design

**Date:** 2026-07-28  
**Status:** Approved by the user on 2026-07-28; user waived a separate
document-review checkpoint  
**Parent run:**
`agentbench_data/runs/28_generals/generals-hl/20260728_1122_57f647d5`  
**Parent version:** `v3`  
**Expected parent content hash:**
`a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`

## Purpose

Run one real, replay-guided `v3 → v4` heuristic-learning act against the
strongest frozen human opponent, then evaluate every runnable v4 on the
unchanged 18-case formal benchmark.

The engineering success condition is a complete, reproducible v4 score point
with intact logs and phase-separated budgets. The performance target is to
exceed v3's `7/18 = 38.89%`, but score improvement is not a version-retention
or formal-evaluation gate.

This remains part of the existing pilot-rescue lineage. It does not retroactively
turn v0 into the newly agreed canonical from-scratch default experiment.

## Considered Approaches

### Selected: replay-guided architecture redesign from frozen v3

Use new strongest-human games, a human-authored replay-analysis skill, critical
decision-window selection, and a single Codex act that may compress or replace
the current policy architecture. This preserves the requested v4 lineage while
repairing the most important protocol deviations found in v3.

### Deferred: narrow reserve-rule patch

Adding a few minimum-garrison conditions would be faster, but would not test
whether replay evidence can drive a coherent strategy redesign. It also risks
moving the all-in failure from one rule branch to another.

### Deferred: canonical from-scratch restart

The next clean main experiment should start from a minimal official-SDK v000.
It must be a new lineage rather than being mislabeled as v4.

## Frozen Experimental Flow

```text
frozen v3
  -> six new strongest-human learning games
  -> deterministic critical-window extraction
  -> human replay-analysis skill + v3 experience + compact evidence
  -> one Codex coding-agent act
  -> runnable v4 snapshot and tests
  -> six paired v4 validation games
  -> behavior and dense diagnostics
  -> unchanged 18-case formal benchmark
  -> immutable result, report, and budget artifacts
```

## Learning and Formal Data Boundaries

Add a separate v4 learning manifest:

```toml
learning_id = "generals-hl-v4-strongest-v1"
opponent_id = "advanced-rank02-robinliu-v18"
seeds = [285101, 285202, 285303]
seats = [0, 1]
```

These six cases are declared before play and are disjoint from:

- formal seeds `280101`, `280202`, `280303`;
- legacy learning seeds `281101`, `281202`, `281303`;
- calibration development and held-out seeds;
- v3 strongest-human seeds `284101`, `284202`, `284303`.

The same six cases are used once before the act to create v3 feedback and once
after the act for paired v4 validation. The prompt may contain only v3 feedback
from these cases. Formal outcomes, formal trajectories, formal seeds, and
opponent source files remain unavailable to Codex.

## Human-Authored Replay-Analysis Skill

Create a versioned, game-specific replay skill owned by the Generals benchmark
assets. The framework copies its exact bytes and hash into the run before
building the prompt.

The skill explains:

- how board cells, generals, ownership, armies, coins, technology, commands,
  terminal states, and evaluated seats map to game meaning;
- how to distinguish survival from actual strategic progress;
- how to inspect main-general danger, reserve sufficiency, force
  concentration, economic timing, and high-value targets;
- how to tie every observation to a replay ID and decision/state ID;
- that dense diagnostics are evidence, not reward or benchmark score;
- that a deterministic action-disagreement trace is behavior change, not
  epistemic information gain or policy KL.

The skill is human-authored input. Codex may update the version-local
`EXPERIENCE.md`, but may not modify the frozen skill during the act.

## Critical Decision Windows

Replace the v3 whole-record truncation policy with deterministic,
episode-local window selection. For each complete v3 learning episode, retain
at most eight unique policy decisions, ordered chronologically after selection:

1. first evaluated-agent decision;
2. first non-end action;
3. first state with observable main-general pressure;
4. decision immediately before the steepest territory-share drop;
5. decision immediately before the steepest army-margin drop;
6. first resource-, sub-general-, or production-relevant opportunity;
7. penultimate evaluated-agent decision;
8. final evaluated-agent decision.

If a criterion is absent or selects an already chosen decision, no synthetic
replacement is added. Selection criteria, selected state IDs, omitted counts,
and serialized byte counts are saved. No values are interpolated.

Each selected record includes the pre-action state fields already exposed by
`state_view.py`, the chosen macro-action, and derived, source-readable features:

- own and enemy main-general armies and positions when visible;
- adjacent and one-step-reachable enemy pressure on the own main;
- owned territory and army totals;
- coin share;
- owned general counts and visible levels;
- movable stack count and largest movable stack;
- the selected action's observable decision class.

The existing state view is sufficient for v4 and remains protected.

## Codex Act Contract

Codex may edit:

- `strategy.py`;
- `STRATEGY.md`;
- `EXPERIENCE.md`;
- `tests/**`;
- Python modules under `policy/**`.

Codex may delete, merge, compress, or replace existing rules and may implement
deterministic search, scoring, planning, or state machines. It may not edit
`main.py`, `state_view.py`, benchmark assets, manifests, replay artifacts,
framework code, opponents, or evaluation code. It may not use randomness,
wall clock, network access, hidden files, opponent-specific source knowledge,
evaluation material, or opaque learned weights.

The prompt asks for a cohesive, explainable redesign around:

- explicit main-general danger and emergency defense;
- dynamic reserves on the main, front line, and valuable generals;
- avoiding unconditional `army - 1` moves;
- force concentration and controlled merging before difficult attacks;
- high-value general and resource targets;
- safe production or mobility investment when no urgent threat exists;
- compression of obsolete or conflicting v3 rules.

These are hypotheses and design objectives, not hard-coded answers or a
researcher-authored replacement policy. Codex remains responsible for the
actual v4 implementation and tests.

Codex must update `EXPERIENCE.md` with replay/state-ID-backed observations,
retained and rejected hypotheses, v4 structural changes, and remaining risks.

## Version Lifecycle and Evaluation Semantics

A v4 is **runnable** when:

1. the provider invocation completes;
2. changed files stay inside the editable policy boundary;
3. strategy tests pass;
4. `EXPERIENCE.md` exists.

Every runnable v4 is:

1. saved with its own manifest and content hash;
2. run on all six paired validation cases;
3. measured on behavior-change and dense diagnostics;
4. run on the unchanged 18-case formal benchmark.

Behavior disagreement, non-main movement, dense improvement, and score
improvement are diagnostics only. They cannot reject a runnable version,
prevent formal evaluation, or trigger automatic rollback. Version status uses
`available`, `invalid`, or `provider_failed`; it never uses `rejected` for a
runnable but strategically weak candidate.

If the provider or policy tests fail, the candidate and all available logs are
still retained, but v4 has a missing score because it is not runnable. If the
formal matrix is incomplete, individual games remain visible while aggregate
score, gain, and dependent AUC values remain missing.

The finalized run summary is written once by the normal run lifecycle. No
post-finalization mutation is permitted. Corrections, if ever needed, must be
represented by a separate derived run or report.

## Measurement and Budget Semantics

### Feedback actually read

Record explicit, pre-act learning-read counters:

- `feedback_episodes_read`;
- `feedback_decision_records_read`;
- `feedback_serialized_bytes_read`;
- prompt bytes and estimated tokens.

Only episodes and decision records actually serialized into the Codex prompt
count as read. Running a match is not equivalent to reading every state in it.

### Phase separation

- **Learning:** six v3 feedback games, prompt construction, and one Codex act.
- **Validation:** six post-act v4 paired games.
- **Evaluation:** 18 formal benchmark games.
- **Total:** an explicitly labeled sum of the available phase budgets.

Post-act validation is not counted as learning-read evidence for the completed
act. Provider prompt, completion, total tokens, tool calls, elapsed time, raw
JSONL, stderr, workspace diff, and source snapshots remain separately
available.

### Scientific outputs

Report:

- `raw_score`, `evo_score_1` through `evo_score_4`;
- `gain_4 = evo_score_4 - raw_score`;
- per-tier and per-seat formal results;
- paired v4-minus-v3 survival, territory, army, coin, and main-pressure deltas;
- v3/v4 decision-class distributions;
- per-decision action-disagreement trace and mean on the fixed v3 feedback
  probe set.

Action disagreement is labeled `behavior_change`. Strict policy KL and
epistemic information gain remain missing because this deterministic policy
does not expose a complete action distribution over the legal decision space.

The historical global AUC remains missing across the earlier failed-act score
gap. v4 must not silently bridge or interpolate that gap.

## Error Handling

- Parent run or hash mismatch: stop before gameplay or provider invocation.
- Seed overlap or wrong opponent: reject configuration before starting a run.
- Incomplete v3 feedback suite: retain partial artifacts and stop before Codex.
- Prompt leak or prompt-size failure: stop before Codex and save the error.
- Provider failure: save raw output, unchanged or partial candidate, and a
  missing v4 score.
- Protected-file change or test failure: mark v4 invalid, save the diff, and do
  not execute unsafe strategy code.
- Incomplete v4 validation: preserve results and diagnostics; formal evaluation
  still runs when the v4 strategy itself is runnable.
- Incomplete formal matrix: preserve games and use missing aggregate metrics.

## Verification

Automated tests must prove:

- exact parent-v3 lineage import and hash validation;
- v4 learning manifest identity, strongest opponent, both seats, and complete
  seed disjointness;
- deterministic critical-window selection and de-duplication;
- prompt leak resistance, byte cap, skill inclusion, and exact read counters;
- runnable-version classification independent of diagnostic improvement;
- formal evaluation occurs for a runnable v4 even when behavior disagreement
  or every dense delta is non-positive;
- phase-separated learning, validation, and evaluation budgets;
- summary finalization without post-finish mutation;
- CLI, event-quality, and report support for v4.

The live run must additionally verify:

- six complete v3 feedback games;
- one completed Codex act and passing strategy tests;
- six complete v4 validation games;
- all 18 complete formal games;
- source, prompt, skill, manifest, and patch hashes;
- provider usage and event-quality diagnostics;
- a regenerated local CI report containing the v4 point without interpolation.
