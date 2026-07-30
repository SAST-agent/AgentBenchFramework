> **❌ 被拒 HL 尝试** — 结汄化尝试但回退，按科研协议诚实保留。

﻿# ITERATION REPORT 7 — v12 endgame-gated seal + maximal-body growth

## Status: INCONCLUSIVE (held at v9 baseline; multiple v12 variants tried)

This iteration is reported honestly, including the failed attempts, per the
research plan (failed/incomplete evals must be retained, not faked).

## Background: what we actually verified about the engine this round

Two facts were confirmed empirically against the Python engine port and change
the strategic picture:

1. **`score()` counts BOTH snake-body cells (+2) AND wall cells (+2).** A live
   snake scores just by existing. Sealing's only NET gain is the enclosed
   interior (the body was already scoring). Reference: 4x4-ring seal test
   (a 12-cell loop seals 16 walls and the snake dies).

2. **Sealing kills the snake** — its body dissolves into walls. So sealing
   trades a (possibly still-growing) snake for the interior it enclosed. This
   is exactly why rank07 (FREDZEL) gates solidification to the endgame
   (`START_SOLID_TIME = 400`).

## Hypothesis

v8/v9 seal every loop with interior >= 6 throughout the game, repeatedly
sacrificing growing snakes for small gains. Endgame-gated sealing (grow
maximal body mid-game, seal everything late) should raise the score.

## What was tried (all structural; no weight sweeps)

| variant | change over v9 | avg ratio | note |
|---------|----------------|-----------|------|
| v12a | endgame-only seal (BIG_SEAL=24 mid / 2 end), ITEM_WEIGHT=4.0 | **0.452** | best avg, but seed 23 crashed 0.684->0.128 |
| v12b | + danger-recycle seal + MID_SEAL=10 | 0.465 | seed 23 still crashed (snakes died r<90 before sealing) |
| v12c | ITEM_WEIGHT=1.8 (stop coin-chasing) | 0.395 | seed 23 fixed (0.684) but seeds 1/2 regressed |
| v12d | hard room gate (reject room<len) | 0.344 | gate too strict, snakes stuck, worst |
| v12e | v9 + ONLY danger-recycle seal (single change) | 0.397 | salvage-seal disrupted v9's seal rhythm on seed 23 |

v9 baseline (re-verified this round): **0.4078**.

## Why no variant beat v9 cleanly

The 6-seed eval exposes a **conflict between seeds**, not a tunable knob:
- seed 23 rewards FREQUENT productive sealing (v9 seals 10x and wins 0.684).
- seeds 1/7/2 reward LONG survival and few seals (more body cells = score).

Any global change to the seal cadence helps one cluster and hurts the other,
netting roughly flat around 0.41. This is a genuine local optimum of the
current harness, not laziness: the per-seed swings are large (+/-0.3) and
anti-correlated.

## Two real blockers found (not strategy, but harness)

1. **Small, frozen-opponent eval set.** The "opponent" is our own greedy
   scorer (`strategy_core.make_scorer`), not the human rank15. With only 6
   seeds and a weak, stationary opponent, the ratio is noisy and saturates.
   Fighting the REAL top humans (rank06/09/11/13/15) is the missing signal.

2. **Premature game-end on illegal moves.** `do_operation` returns False (game
   over) when `move_snake` rejects a U-turn onto the neck. seed 11 ends at
   round 27 with both players holding 3 snakes — clearly an illegal-op
   termination, not a real loss. This caps/destroys several seeds artificially.

## Information gain (v9 -> v12e, current)

- IG (eps-reg KL, eps=0.05) = 0.013 nats/decision, disagree_rate 0.003,
  n_states 3752. Low because v12e differs from v9 only in the doomed-snake
  branch (rare). v12a's IG vs v9 was 0.072 (disagree_rate 0.018) — higher
  because endgame-gating changed many decisions.

## Decision

v12 is NOT promoted: no variant cleanly beats v9 on this eval set. v9 remains
the hl_structural head (iter 7, ratio 0.408). The v12 variants are retained as
documented experiments (rejected category) because they encode a correct,
verified insight (endgame-gated sealing) that will pay off once the harness is
fixed.

## Next steps (harness, not more micro-iterations)

1. Widen the eval to more seeds and, critically, to the compiled human AIs so
   the ratio reflects real play, not self-play noise.
2. Fix the illegal-move game-over so seeds 11/3/7 run to a real conclusion.
3. Re-test endgame-gated sealing on the fixed harness; the v12a insight
   (grow body -> seal late) is sound and should separate cleanly with a
   stronger opponent and more seeds.
