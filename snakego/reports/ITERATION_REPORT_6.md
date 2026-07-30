> **✅ HL 主线 (HL-2)** — 结汄化代码变化（space-max GROW 重写），计入 HL 迭代曲线。

# SnakeGo HL Iteration Report 6 -- v9 (survive-and-multiply)

## What changed (structural, not weight-tuning)

v9 rewrites the GROW phase of v8's phase machine. v8's GROW had two **hard
rejects** (enemy head within 2 cells, room < length+4) that, on contested
boards, emptied the candidate set entirely and fell back to the v7 *territory*
weighted scorer -- which optimises for sealing/items, not survival. The snake
walked into a wall and died before it could split or seal. This is why v8 was
bimodal: dominant on open boards (seed 23: 0.821), dead on contested ones
(seed 1: 0.097, seed 7: 0.020).

v9's GROW phase is a **two-tier space-maximiser**:

1. Tier A: greedy growth -- score each legal move by the flood-fill free space
   it leaves, with straight-line and item tiebreaks. Enemy-head proximity is a
   **soft** penalty (never removes the move). This guarantees the candidate set
   is never emptied by enemy pressure.
2. Tier B: pure survival -- the most spacious legal move, bar nothing. Replaces
   the v7 territory-scorer fallback that caused early deaths.

SPLIT threshold lowered from 10 to 8 (insurance comes first).

SEAL and HUNT_LOOP inherited unchanged from v8.

## Results vs frozen greedy-scorer opponent (same eval set as all prior iters)

| seed | v8 score | v8 ratio | v9 score | v9 ratio | delta |
|------|----------|----------|----------|----------|-------|
| 1    | 12 / 112 | 0.097    | 78 / 104 | 0.429    | +0.332 |
| 2    | 30 / 32  | 0.484    | 24 / 168 | 0.125    | -0.359 |
| 3    | 36 / 42  | 0.462    | 66 / 74  | 0.471    | +0.009 |
| 7    | 4 / 198  | 0.020    | 54 / 100 | 0.351    | +0.331 |
| 11   | 76 / 46  | 0.623    | 24 / 38  | 0.387    | -0.236 |
| 23   | 156 / 34 | 0.821    | 104 / 48 | 0.684    | -0.137 |
| avg  |          | 0.418    |          | 0.408    | -0.010 |

**The survival fix works**: v8's worst-case seeds (1 and 7) went from near-zero
to competitive (0.43 and 0.35). But v9 lost aggressiveness on the seeds where
v8 was already winning (2, 11, 23) because pure space-maximisation does not
form loops as effectively as v8's item-seeking growth. Net average is flat.

## Information gain: v8 -> v9

Per-state epsilon-regularized KL (eps=0.05), 2189 shared decision states:

  IG = 0.343 nats/decision
  disagree_rate = 8.7%

This is a genuine structural change (not the ~0.0003 nats of v1-v4 weight
tuning). The two strategies disagree on ~1 in 12 decisions.

## Rejected attempts (retained per research doc sec 6/12)

Two further structural changes were attempted and rejected because they
regressed:

- **v10 (TERRITORY phase)**: once a backup snake exists, hunt aggressively at
  threshold length 10. Regressed to 0.284 -- the aggressive hunt drove snakes
  into traps.
- **v11 (CURVE-forcing)**: after 4+ straight segments, force a 90-degree turn
  to build a loop-forming path. Regressed to 0.358 -- uncoordinated turns led
  into worse positions than straight growth.

Both are kept in the data as honestly-reported failures; the main curve tracks
only accepted HL structural iterations.

## Control group (weight-tuning ablation)

v0-v7 are pure weight/score tuning with no structural decision change. Their
IG is ~0.0003 nats (effectively zero), confirming they are "manual RL" rather
than HL. They serve as the control group for the Policy Ablation (research plan
item 10): structural HL produces non-trivial IG and territory improvement where
weight-tuning does not.

| version | ratio | IG (nats) |
|---------|-------|-----------|
| v0-v7   | ~0.34 | ~0.0003   |
| v8      | 0.418 | 1.87      |
| v9      | 0.408 | 0.343     |

## Next direction

v9 proved survival but lost sealing aggressiveness. The next iteration should
combine: keep v9's space-maximising survival for the early game, but once a
backup snake exists AND the board has enough open space, switch the longest
snake to an **item-seeking curve** that forms loops naturally -- not the blind
hunt of v10 or the uncoordinated turn of v11. The key lesson from v10/v11 is
that geometric loop-formation needs to be coordinated with body position, not
just forced mechanically.
