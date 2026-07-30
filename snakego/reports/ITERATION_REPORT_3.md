> **⚠ CONTROL GROUP (权重调参，非 HL)** — 本报告属于 Policy Ablation 对照组。所有改动只是调整权重数值，IG≈0证实是“手动 RL”，不计入 HL 主线。详见 reports/README.md。

# Iteration Report 3: Strategy Bug Fixes + Persisted Iteration Loop

Date: 2026-07-28

## What was broken

1. **`curves.py` produced empty output.** The `runs/` directory had no persisted
   replays because the previous session's sandbox was read-only. All iteration
   data existed only in memory and was lost.

2. **The strategy never split.** Two interacting bugs:
   - `space_per_body_len` was uncapped (`w * room / length`). Early game this
     produced scores of ~26, swamping the split term (6.0). Split never won
     the argmax.
   - The split gate required `room >= snake.length + 12`, which combined with
     the uncapped space term meant splitting was effectively impossible. The
     snake grew to length 28 without splitting, then boxed itself in and died
     at round 76.

3. **The Experience seed corrupted weights.** `seed_experience()` initialized
   `weights_snapshot` as an empty dict, then applied seed deltas on top of 0.0.
   This replaced the tuned base defaults with raw offsets (e.g. `split_value`
   became 1.5 instead of the intended 7.5).

## What I fixed

### Strategy (`strategy_core.py`)

| Term | Before | After | Why |
|------|--------|-------|-----|
| `space_per_body_len` | 1.0, uncapped | 0.4, capped at 4.0 | stop it from swamping all other terms |
| `trap_penalty` | -3.0 | -8.0 (amplified) | survival is the ceiling |
| `lethal_trap` | (none) | -1000.0 when room < len | hard gate: never walk into a trap |
| `split` | hard gate room>=len+12 | length-urgency: base 8.0 + urgency per len beyond 6 | humans split at len 9-16 |
| `max_growth_length` | (none) | 14.0, decay beyond | stop overgrowing into death traps |
| `overcommit_corner` | -1.5 | -2.0 | avoid 1-exit pockets |

### Experience seed (`experience.py`)

`seed_experience()` now initializes `weights_snapshot` from the full default
`Weights()` dataclass, so seed deltas ADD to tuned defaults instead of
replacing them.

### Learning (`loop.py`)

Replaced the simplistic `learn()` (which only counted my own splits) with a
trajectory analyzer that examines BOTH players:
- counts opponent splits vs mine (split tempo race)
- detects my snake deaths with death-round and max length (survival)
- measures opponent's big territory gains (sealing effectiveness)
- compares territory scores (deficit detection)

Each lesson is traceable to replay evidence and produces targeted weight deltas.

### Persistence (`loop.py` + `run_iterations.py`)

- `persist_all()` saves `versions.json`, `experience.json`, `curves.json`
- Replays saved as `iter{N}_{learn,eval}_s{S}.json`
- `curves.py` now reads `curves.json` for the eval series, falling back to
  scanning learn replays
- Fixed eval to use seed=1 consistently so scores are comparable

## Iteration results (4 iterations, learn vs rank15, eval vs sample_ai seed 1)

| Iter | Learn (vs rank15) | Eval my | Eval hu | Win | IG(KL) | Lesson |
|------|--------------------|---------|---------|-----|--------|--------|
| 0 | 24-30, r30 | 170 | 28 | W | n/a | no clear deficit |
| 1 | 32-52, r79 | 170 | 28 | W | 0.183 | opp split 4x vs my 3x |
| 2 | 22-180*, r318 | 170 | 28 | W | 0.021 | split more; seal more |
| 3 | 24-30, r30 | 170 | 28 | W | 0.201 | holding weights |

*iter2 learn had a timeout error (desync), honestly preserved.

Score is flat at 170 vs sample_ai because the weight changes weren't large
enough to flip decisions on seed 1. IG(KL) confirms the action distribution
DID change each iteration. Weights evolved: split 9.5->10.4, seal 3.3->3.7.

## Milestone eval (final weights, seed 1)

| Opponent | My score | Hu score | Result | Rounds |
|----------|----------|----------|--------|--------|
| sample_ai | 170 | 28 | **WIN** | 306 |
| rank06 | 22 | 32 | lose | 28 |
| rank11 | 24 | 30 | lose | 30 |
| rank15 | 24 | 30 | lose | 30 |
| rank04 | 24 | 62 | lose* | 76 |
| rank01 | 18 | 48 | lose* | 54 |

*error (subprocess crash/desync). Complete game vs rank01 seed 3: 26-90 lose.

## Honest assessment

**Dramatic improvement vs weak humans.** Before fixes: 32-8 vs sample_ai in a
76-round game (both died early). After: 170-28, 306 rounds, decisive win.

**Still far from ranked humans.** Against rank01/04/06/11/15, my snakes die at
rounds 28-76. The root cause is survival: against opponents that seal territory
efficiently, my snakes get boxed in within 30-80 rounds. The single-step
flood-fill survival gate doesn't anticipate multi-step traps.

## Next steps (priority order)

1. **2-step survival lookahead**: simulate each candidate move, check if the
   snake still has room 2 steps ahead. Catches traps the 1-step gate misses.
2. **Opponent-aware pathing**: detect when the opponent is building a seal
   toward my territory and preemptively escape or counter-seal.
3. **Active big-loop sealing**: humans build U-shaped paths then close them for
   +6 to +20 territory bursts. Implement as a planned-path mode.
4. **Fix rank01/rank04 desync**: some seeds cause the human subprocess to crash.
   Likely an edge case in `seal_region` or `split_snake` turn ordering.
5. **Multi-seed eval population**: average scores over 5+ seeds to reduce
   variance and detect small improvements.

## Deliverables status

1. HL closed loop: working (play -> learn -> snapshot -> eval -> IG -> persist)
2. Replay-watching skill: exists (`REPLAY_SKILL.md` + `replay2.py`), verified
3. Decision space: defined (`decision_space.py`: 6 macros, mask, support)
4. IG: strict KL over {1..6} + Laplace, desync preserved as null
5. Score/IG curves: rendering with 4 real iterations, version-aligned

## Files changed this session

- `snakego/strategy_core.py` — survival gate, split fix, growth moderation
- `snakego/experience.py` — seed weights start from full defaults
- `snakego/loop.py` — trajectory-analyzing learn(), persist_all(), fixed eval seed
- `snakego/curves.py` — reads curves.json for eval series
- `snakego/run_iterations.py` — new runner script
- `snakego/runs/` — 8 replays + versions.json + experience.json + curves.json
