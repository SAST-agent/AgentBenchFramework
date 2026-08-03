---
name: antwar2-replay
description: Use when diagnosing AntWar2 replay.json or trace.jsonl files, translating protocol numbers into public causal evidence, preparing HL candidate changes, or updating replay-grounded Experience knowledge.
---

# AntWar2 Replay Analyst

Translate public records into auditable evidence for an interpretable edit. Replays expose operations and effects, not intent.

## Workflow

1. Read the frozen rules and `decision_space.yaml`.
2. Generate the bounded summary:

   ```bash
   python scripts/summarize_replay.py REPLAY.json --output match_summary.json
   ```

3. Reject incomplete replays, protocol faults, timeouts without a valid terminal state, and unknown operation codes. Infrastructure failure is not a strategy result.
4. Start with terminal camps, winner, breaches, operation counts, and event index. Select at most two hypotheses.
5. For each hypothesis, record exactly:

   ```text
   public condition
   -> accepted atomic operation
   -> immediate public effect
   -> delayed public effect
   -> candidate response and falsifier
   ```

6. Inspect only a bounded round window when the summary is insufficient:

   ```bash
   python scripts/inspect_trace_window.py TRACE.jsonl --round 80 --radius 2
   ```

7. Implement a visible-state condition and interpretable response. Large code growth and additional branches are allowed.
8. Validate against the unchanged opponent in a live match. A fixed replay is diagnostic only and cannot authorize promotion.

## Evidence rules

- Treat `op0` and `op1` as accepted atomic operation sequences for the recorded round.
- `BUILD_TOWER` is code 11 with `(x, y)`; it has no tower-type argument and creates a Basic tower.
- `UPGRADE_TOWER` is code 12 with `(tower_id, target_type)`.
- `DOWNGRADE_TOWER` is code 13 with `(tower_id)`.
- Codes 21–24 target `(x, y)`; codes 31–32 have no arguments.
- Reconstruct towers as a delta stream: `type == -1` removes a tower; otherwise update it by global tower ID.
- Attribute a breach only from a decrease in the defender's public `camps` value.
- Separate immediate effects in the next public state from delayed effects several rounds later.
- Compare candidate and opponent responses in both candidate roles, P0 and P1.
- Do not infer intent, hidden belief, objective, or a named tactic from an action sequence.
- Do not invent a tactical decision space. Behavioral KL uses the legal protocol atoms in `decision_space.yaml`.
- Do not branch policy code on seed, replay ID, opponent identity, or opponent source.

## From replay to Experience

Write one condition-scoped finding with role, opponent, seed, rounds, parent and candidate results, dense-margin change, activation evidence, and failure boundary. Keep good, bad, mixed, inconclusive, and invalid findings. Candidate notes stay provisional until joined with Framework facts.

## Common mistakes

| Mistake | Correction |
|---|---|
| Treating an empty bundle as missing data | Record the atomic decision as `HOLD`. |
| Reading a tower delta as the complete tower set | Reconstruct by global tower ID across rounds. |
| Calling `BUILD_TOWER(x,y,type)` | Use `BUILD_TOWER(x,y)`; upgrade in a later accepted atom. |
| Explaining one score swing with intent | State the public condition, operation, observed effects, and falsifier. |
| Promoting after replay simulation | Require a live adaptive match against the unchanged package. |
