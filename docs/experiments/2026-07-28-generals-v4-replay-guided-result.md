# Generals pilot v4 replay-guided result

## Scope

This is the approved legacy-pilot `v3 → v4` replay-guided iteration. It is not
a canonical from-scratch experiment. The immutable parent was:

- run: `20260728_1122_57f647d5`
- version: `v3`
- content hash:
  `a9f27eb2a02ba452e9363022d97ffa5556c0f3a02eef120061e73e93c68ce815`

The completed child run is `20260728_1519_7b242596`. Its v4 content hash is
`5c12e7c92843cbc18b742ab7be71c684cfa0a0aac8e5dbb5a24fc7b905fe959f`.

## Frozen protocol

- Learn from six v3 games against the strongest frozen human opponent, using
  seeds 285101/285202/285303 and both seats.
- Select deterministic episode-local critical windows and include every
  declared learning episode in the completed prompt.
- Combine the frozen human-authored replay-analysis skill, v3 experience, and
  compact replay evidence in one Codex act.
- Allow deterministic, explainable policy redesign and strategy compression.
- Save every runnable v4, run six paired validation games, and then run the
  unchanged 18-case high/medium/low formal benchmark.
- Treat behavior and dense changes as diagnostics, never as a formal-evaluation
  or automatic-rollback gate.

The completed prompt contained all 6 episodes, 36 selected decisions, 46,667
serialized evidence bytes, and 60,409 total bytes. Its estimated prompt size
was 15,103 tokens; actual provider accounting was 565,981 input and 8,457
output tokens. No declared episode was omitted.

- prompt SHA-256:
  `78ddb650120d4657c92f9b753cc1fe24ee5e09e38047aaf31050524bf1181bb7`
- replay-skill SHA-256:
  `f6f3a6eadfd092d476e73c034bce9452b1c548d80d3ce0d4ebefdc9865799df4`

## Preserved pre-act failure

The first attempt, run `20260728_1504_0603955b`, completed all six v3 learning
games but stopped before Codex because the prompt could include only 2/6
episodes within the byte contract. Repeated full general lists dominated the
evidence. The run remains finalized as `prompt_incomplete`, with zero
coding-agent acts and no v4 score.

The prompt serializer was then compacted to keep opportunity-relevant strategic
targets and the four largest movable stacks. This reduced complete six-episode
evidence to 46,667 bytes. The second attempt performed the one real Codex act.

## What Codex changed

Codex changed only:

- `strategy.py`;
- `STRATEGY.md`;
- `EXPERIENCE.md`;
- `tests/test_strategy.py`.

It replaced v3's global-stack scoring with a deterministic
reserve-and-concentration planner. The new policy added visible local threat
reserves, urgent main-general defense, controlled merging, safer production
upgrades, and explicit high-value target priorities. Its snapshot passes 19
strategy tests.

## Behavior and paired diagnostics

On the 36 frozen critical-window probe decisions, v4 disagreed with v3 on
22 decisions (`61.11%`).

| Observable action class | v3 | v4 |
| --- | ---: | ---: |
| Main-army move | 8/36 (22.2%) | 3/36 (8.3%) |
| Non-main-army move | 22/36 (61.1%) | 20/36 (55.6%) |
| General upgrade | 0/36 (0.0%) | 0/36 (0.0%) |
| End only | 6/36 (16.7%) | 13/36 (36.1%) |

Paired v4-minus-v3 validation deltas against the strongest opponent:

| Dense metric | Mean delta |
| --- | ---: |
| Completed rounds survived | +20.83 |
| Terminal territory share | -0.0093 |
| Terminal army margin | -666.83 |
| Terminal coin share | -0.1213 |
| Terminal net-main pressure | missing |

Both v3 and v4 remained `0/6` against the strongest opponent. v4 survived
longer, but army, economy, and territory evidence became worse.

## Formal result

The complete formal benchmark regressed from v3's `7/18` to v4's `4/18`:

| Version | Score | High | Medium | Low | Seat 0 | Seat 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | 7/18 (38.89%) | 0/6 | 1/6 | 6/6 | 4/9 | 3/9 |
| v4 | 4/18 (22.22%) | 0/6 | 0/6 | 4/6 | 2/9 | 2/9 |

The v4 gain over v0 is `+22.22` percentage points, but its change from v3 is
`-16.67` percentage points. All 18 cases are valid. v4 is retained as a real
evaluated version; v3 remains the best historical version and no score-driven
rollback rewrites either artifact.

## Failure analysis

The regression is consistent with an over-conservative implementation rather
than a lack of behavior change:

1. `_reserve()` includes the full visible adjacent hostile pressure.
2. `_surplus()` subtracts that reserve from the main army.
3. `_urgent_main_defense()` then requires this already protected surplus to be
   larger than the same hostile army before counter-capturing.

For one adjacent threat of army `T`, the general buffer makes the effective
counterattack condition approximately `main army > 2T + 2`. The threat is
therefore counted twice. When this test fails and no adjacent disposable field
or resource stack can reinforce the main, `choose_actions()` ends the turn
instead of considering a controlled defense or productive move.

This matches the measured evidence: end-only decisions more than doubled,
survival improved, and army/economy/territory plus three formal wins were lost.
This is a diagnosis from the saved code and replay-level aggregate evidence;
it is not a post-hoc replacement score.

## Measurement integrity and budgets

The successful run itself used:

- learning: 6 episodes, 1,609 environment steps, 803 game-agent decisions,
  4,244 primitive commands, one coding-agent act, and 191.89 seconds;
- validation: 6 episodes, 1,859 environment steps, 928 decisions, 4,586
  primitive commands, and 14.89 seconds;
- evaluation: 18 episodes, 9,104 environment steps, 4,549 decisions, 16,855
  primitive commands, and 67.78 seconds;
- total: 30 episodes, 12,572 environment steps, 6,280 decisions, 25,685
  primitive commands, and 274.56 seconds.

The successful run was finalized before a CLI routing defect was discovered:
its summary did not reference the zero-act failed prompt attempt. The two
finalized summaries remain immutable. A separate hashed campaign receipt at
`agentbench_data/derived/28_generals/generals-hl/20260728_v4_campaign-budget.json`
adds that attempt and gives the audited global coordinates:

- 5 coding-agent acts;
- 58 learning episodes;
- 30,882 learning environment steps;
- 12,091 game-agent decisions;
- 50,465 primitive commands;
- 779.94 learning seconds.

Cumulative tokens remain missing because historical failed acts do not expose
complete token usage. The Dashboard labels the overlay as a derived campaign
budget instead of silently changing the finalized summary.

The completed run has 140/140 valid events, with no malformed events, unknown
types, duplicate IDs, missing IDs, or warnings. The global score curve is
`[0, 0, missing, 0, 0.3889, 0.2222]`; every global AUC remains unavailable
because the historical missing score point is not interpolated. Strict policy
KL and epistemic information gain also remain missing because the strategy
does not expose a complete legal-action distribution.

## Conclusion and next iteration

v4 demonstrates that replay evidence can drive a large, explainable
architecture change in one act, but it also shows that behavior change and
longer survival are not sufficient proxies for playing strength. Saving and
formally evaluating every runnable version exposed the failed hypothesis
instead of converting diagnostics into a hidden acceptance gate.

A v5 should start from retained v3 or selectively repair v4, remove the
double-counted threat reserve, and add regression tests for urgent defense
with one adjacent attacker. It should preserve v3's active global-stack
breadth while using a single coherent reserve calculation, then repeat the
same non-gated formal evaluation protocol.
