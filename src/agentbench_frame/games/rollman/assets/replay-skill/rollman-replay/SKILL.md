---
name: rollman-replay
description: Analyze frozen 29_rollman JSONL replays and turn visible state, paths, scores, and protocol-defined events into evidence for a causal, generalizable Rollman code edit. Use for replay diagnosis, candidate review, failure analysis, Experience Skill updates, and deciding whether an HL candidate should be revised. Reject malformed or incomplete evidence instead of guessing.
---

# Rollman Replay Analyst

Use this Skill to translate a frozen-backend replay into auditable decision evidence. Keep observations, rule-grounded interpretations, and hypotheses separate. The replay exposes behavior and outcomes, not hidden intent.

## Required workflow

1. Read `../../rules.md` and `../../decision_space.yaml`.
2. Run:

   ```bash
   python scripts/summarize_replay.py REPLAY.jsonl --format markdown
   ```

3. If the command fails, mark the replay invalid and stop. Do not treat infrastructure failure as a loss, zero reward, or strategy defect.
4. Check `Round coverage`. Any reported gap means the omitted rounds are unknown. Do not summarize actions, loops, threats, or causality across that interval.
5. Locate score changes and named events at exact `(level, round)` coordinates.
6. Inspect Rollman and Ghost paths for the same complete round before diagnosing collision or escape behavior.
7. Compare the visible pre-decision state with the submitted direction and the resulting path. Do not invent an item target, route goal, tactic label, belief, or intent.
8. Propose the smallest code-level policy change that generalizes to an explicit visible-state condition.
9. State what replay evidence would falsify the hypothesis. Prefer one causal edit over a parameter sweep.

## Frame semantics

- A level initialization frame supplies the board and state before that level's first decision.
- A normal round frame records the completed round. Its path and event fields are outcome evidence, not a fresh decision request.
- A terminal frame has non-null `StopReason`. It is bookkeeping after play and is never a decision point.
- A replay with a terminal frame but missing ordinary rounds is not evidence about actions in those missing rounds.
- `portal_available` in a completed round is the post-round value. A portal that becomes available in that frame can only be used by a later decision.
- Never recommend an action "at" a terminal frame. If no later decision frame exists, there is no observed opportunity to take it.

## Frozen numeric meanings

Directions:

| Code | Meaning |
|---:|---|
| 0 | STAY |
| 1 | UP |
| 2 | LEFT |
| 3 | DOWN |
| 4 | RIGHT |

Replay events:

| Code | Meaning |
|---:|---|
| 0 | EATEN_BY_GHOST |
| 1 | SHIELD_DESTROYED |
| 2 | FINISH_LEVEL |
| 3 | TIMEOUT |

Board and skill codes must come from `../../rules.md`. If a numeric value is absent from that frozen contract, label it unknown and stop that line of interpretation.

## Collision evidence

Do not diagnose a capture from endpoints alone. For the same complete round:

1. Normalize wall sentinels to the last valid coordinate.
2. Compare the three path samples used by the frozen backend.
3. Account for shield and invincibility state.
4. Use the event field to confirm `EATEN_BY_GHOST` or `SHIELD_DESTROYED`.

An endpoint near a Ghost, a score change, or a reset coordinate alone does not prove the causal path.

## Portal and timeout evidence

- Levels 1 and 2 open portals only after their configured round threshold is settled.
- The first legal opportunity to enter is a later decision.
- `TIMEOUT` proves the level reached its limit; it does not prove Rollman repeatedly chose STAY.
- A sparse replay containing initialization and round 400 proves neither a 400-round loop nor the action distribution of the omitted rounds.
- Do not create an arbitrary "wait N rounds" watchdog. A threshold must come from a frozen rule, measured validation result, or an explicitly recorded experiment parameter.

## Score evidence

Use score deltas only as outcome evidence. Attribute a delta to a game event only when the frozen rule and same-round fields jointly support it. Do not infer which item Rollman meant to collect or which Ghost behavior it predicted.

## From evidence to a code edit

Write the diagnosis in this order:

1. **Observed fact** — exact level, round, field, path, event, or score delta.
2. **Rule-grounded interpretation** — only what follows from the frozen rules.
3. **Hypothesis** — a falsifiable explanation of a policy weakness.
4. **Policy edit** — visible-state predicate and interpretable action-selection change.
5. **Validation** — fixed seeds/opponents and metrics that could reject the edit.

Good policy edits change how code ranks legal directions under a recurring state condition. They may use search, graph distance, risk maps, finite-state memory, or other interpretable algorithms. They need not be a flat `if/else` list.

Bad edits include:

- blind grid search or enumerating many constants without a causal hypothesis;
- memorizing a seed, round number, coordinate, replay, or opponent identity;
- adding a tactic category to the KL decision space;
- accumulating exceptions without compressing overlapping logic;
- claiming hidden opponent intent;
- changing multiple unrelated subsystems from one replay.

## Required analysis output

Return:

```text
Replay validity:
Coverage limitations:
Observed facts:
Rule-grounded interpretation:
Falsifiable hypothesis:
Proposed policy edit:
Why this is not grid search:
Validation and rollback condition:
Experience Skill update:
```

The Experience Skill update must be a concise reusable lesson with supporting replay locations and a falsifier. If evidence is incomplete, record a coverage warning instead of a strategy lesson.

## Final checks

Before returning, verify:

- Every event name came from the frozen table.
- Every claimed action came from a recorded decision or submitted-action trace, not an outcome frame.
- No behavior was inferred across a round gap.
- No terminal frame was treated as actionable.
- Collision claims use complete same-round paths and event confirmation.
- The proposal contains no arbitrary threshold or parameter sweep.
- The suggested edit can be reverted if fixed-seed evaluation degrades against the historical champion.
