# Generals pilot v5 rollback-guided result

## Scope

This is the approved legacy-pilot v5 iteration, not a new canonical
from-scratch experiment. Its immutable lineage parent is the regressed v4 run,
while its editable source is the retained v3 snapshot:

- v4 parent run: `20260728_1519_7b242596`;
- v4 parent hash:
  `5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f`;
- v3 rollback hash:
  `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`;
- completed v5 run: `20260729_0818_e6bcb9b3`;
- v5 hash:
  `facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e`.

The rollback does not rewrite or delete v4. The run stores v3, v4, and v5
source snapshots, both v3-to-v5 and v4-to-v5 patches, and the explicit
`version_rollback` event.

## Frozen protocol

- Run v3 and v4 independently against the strongest frozen human opponent on
  new seeds 286101/286202/286303 and both seats: 12 pre-act games.
- Keep the two versions' divergent state IDs separate and select no more than
  six declared critical windows per episode.
- Give Codex both retained strategies and experience files, the official
  rules, the frozen human replay skill, observable decision classes, and
  paired dense diagnostics.
- Start the editable workspace from exact v3, permit one deterministic and
  explainable code act, and preserve `main.py` and `state_view.py`.
- Run six v5 validation games and the unchanged 18-case formal benchmark for
  every runnable v5. No behavior, dense, or performance gate can hide a
  regression.

The completed prompt contains all 12 versioned episodes and all 58 selected
decisions:

- serialized evidence: 69,831 bytes;
- total prompt: 104,224 bytes;
- estimated prompt tokens: 26,056;
- omitted episodes: 0;
- prompt SHA-256:
  `750ae6c554aece5b576cfcb5b17cbd43bc5fd8e7ad001ca333d55276dc180dca`;
- replay-skill SHA-256:
  `f6f3a6eadfd092d476e73c034bce9452b1c548d80d3ce0d4ebefdc9865799df4`.

## Preserved zero-act attempt

The first real attempt, run `20260728_1837_9b65a3d6`, completed all 12
learning games but could fit only 9 of 12 episode records under the 131,072
byte cap. It finalized as `prompt_incomplete`, used zero new coding-agent acts,
and produced no v5 score.

The fix kept the cap unchanged. It removed a duplicated pressure field,
retained terminal/time-average/AUC dense statistics, and bounded each selected
state's strategic targets and movable-stack details. Replaying the real
evidence then fit all 12 episodes and 58 decisions. The successful retry names
the failed run as `prior_attempt_run_id`, so both sets of 12 learning games
remain in the cumulative budget.

## What Codex changed

Codex changed only:

- `strategy.py`;
- `STRATEGY.md`;
- `EXPERIENCE.md`;
- `tests/test_strategy.py`.

V5 keeps v3's global enumeration of all movable stacks and replaces the
all-but-one rule with one contextual reserve calculation. A source retains one
occupier plus visible adjacent hostile pressure. When attacking an adjacent
threat, that exact defender is excluded from the reserve and paid once by the
strict winning comparison, removing v4's duplicate accounting.

Urgent main defense first attempts a defeatable counter-capture, then a
reserve-safe adjacent reinforcement. If neither is possible, the policy
continues to global capture, production, and routing instead of v4's broad
end-only branch. The saved source passes 19/19 strategy tests.

## Validation and behavior

V3, v4, and v5 each remained 0/6 on the new strongest-human validation suite.
The paired v5 validation deltas are:

| Dense metric | v5 − v3 | v5 − v4 |
| --- | ---: | ---: |
| Completed rounds survived | 0.00 | +0.67 |
| Terminal territory share | -0.0386 | +0.0289 |
| Terminal army margin | -45.50 | +397.33 |
| Terminal coin share | +0.0146 | +0.0297 |
| Terminal net-main pressure | missing | missing |

The original finalized run records missing action disagreement because the
measurement caller passed the combined v3/v4 action dictionary into each
version-specific exact-ID comparison. This did not affect policy execution,
validation, or formal scoring. The pipeline has a regression test and now
filters the action dictionary per source version.

Because finalized summaries are immutable, recovered values live in the
hash-verified separate receipt
`agentbench_data/derived/28_generals/generals-hl/20260729_v5_behavior-diagnostics.json`.
Replaying the saved v3/v4 selected states against the immutable v5 snapshot
gives:

| Comparison | Changed decisions | Action disagreement |
| --- | ---: | ---: |
| v5 vs v3 | 7/30 | 23.33% |
| v5 vs v4 | 16/28 | 57.14% |

Across all 58 probes, v5 emits 16 main-army moves, 30 non-main-army moves, and
12 end-only actions. Strict policy KL and epistemic information gain remain
missing because the deterministic strategy exposes actions, not a complete
legal-action probability distribution.

## Formal result

V5 restores v3's best score and reverses v4's regression:

| Version | Score | High | Medium | Low | Seat 0 | Seat 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 7/18 (38.89%) | 0/6 | 1/6 | 6/6 | 4/9 | 3/9 |
| v4 | 4/18 (22.22%) | 0/6 | 0/6 | 4/6 | 2/9 | 2/9 |
| v5 | 7/18 (38.89%) | 0/6 | 1/6 | 6/6 | 3/9 | 4/9 |

V5 gains 3/18 wins and 16.67 percentage points over v4. Its gain over raw v0
is 38.89 percentage points. It does not exceed v3.

The equal v3/v5 totals do not mean identical behavior. V3's medium-tier win was
seed 280202 as seat 0; v5's medium-tier win is seed 280303 as seat 1. The
formal change therefore agrees with the measured 23.33% disagreement against
v3.

## Logs, quality, and budgets

The completed run preserves prompt, provider raw JSONL/stderr, token usage,
tool calls, elapsed time, workspace diff, tests, all match replays/protocols,
dense traces, events, quality, and every version snapshot.

The provider act used:

- 921,498 input tokens, including 862,208 cached input tokens;
- 9,562 output tokens and 752 reasoning-output tokens;
- 931,060 total billed/accounted tokens;
- 34 tool calls;
- 245.93 seconds.

The successful run itself used:

- learning: 12 episodes, 3,758 environment steps, 1,876 game-agent decisions,
  9,233 primitive commands, one act, and 266.11 seconds;
- validation: 6 episodes, 1,883 environment steps, 940 decisions, 4,661
  primitive commands, and 11.77 seconds;
- formal evaluation: 18 episodes, 10,758 environment steps, 5,377 decisions,
  19,355 primitive commands, and 62.40 seconds;
- total: 36 episodes, 16,399 environment steps, 8,193 decisions, 33,249
  primitive commands, and 340.28 seconds.

The cumulative learning budget, including prior rounds and the zero-act v5
attempt, is 6 coding-agent acts, 82 episodes, 38,398 environment steps, 15,843
game-agent decisions, 68,931 primitive commands, and 1,063.19 seconds.
Cumulative tokens remain missing because older failed acts did not expose
complete token usage.

All 18 formal cases and all 18 v3/v4/v5 learning/validation rows are valid.
The event stream is 175/175 valid, with no malformed events, unknown types,
duplicate IDs, missing IDs, or quality warnings. Snapshot manifests recompute
exactly for v3, v4, and v5. `agentbench data check` reports 13 valid and 0
invalid runs.

The global score curve is
`[0, 0, missing, 0, 0.3889, 0.2222, 0.3889]`. All full-history AUC values
remain missing because the historical gap is not interpolated.

## Conclusion

One rollback-guided HL act recovered all formal wins lost by v4 while keeping
the failed v4 hypothesis and its evidence available for comparison. This is
evidence that version retention plus replay-guided code repair can correct a
specific heuristic regression without stacking another opaque threshold.

It is not yet evidence that v5 learned to beat the strongest human: v5 remains
0/6 in both the new strongest-human validation suite and the unchanged formal
high tier. The next iteration should retain v5, target medium/high opponents,
and learn multi-turn main safety and routing without using the formal
trajectories as training material.
