# Generals clean-room heuristic-learning v8 result

## Status and lineage

- Outcome: `complete`, runnable policy, performance target not met.
- Authoritative terminal run: `20260806_1126_8e1b6795`.
- Effective coding-agent act run: `20260806_1117_a83805bc`.
- Parent: v7 run `20260730_1739_680b1632`, content hash
  `c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4`.
- Frozen v8 content hash:
  `3f69b81a1b4831a980084f8419b39292fa34feac43cb6b88aefe26da3898f35a`.
- Prompt-manifest SHA-256:
  `0cf634608a9c8336d5ff7120053218f77363b0ecef3e72e2615e8be276c6b044`.
- Feedback-receipt SHA-256:
  `6b13b52c6eb3d138cd209f9efed2a4e18db3d361065f3339a35bbfc9bc214f81`.
- v8 manifest-file SHA-256:
  `8539dd01eaf0cc27fd9be1a2a8c7e2faff53759841d6919f8d6483659beeeb66`.
- Effective global act count: 9; v8 round effective act count: 1.

The first attempted run stopped before the provider because a dense AUC value
was numerically equal to an old seed. A second provider initialization failed
before model execution because the sandbox made the Codex state database
read-only. Both runs remain preserved. With explicit approval, that
infrastructure failure was excluded from the scientific act count. Exactly one
Codex turn then completed. Two later recoveries reused the same frozen source
without invoking the provider: the first corrected an over-specific document
gate, and the second regenerated the terminal summary with the inherited act
count. No policy change was made after the completed act.

## Learning evidence and budget

The act read exactly six learning games against
`advanced-rank02-robinliu-v18`: seeds `300101`, `300202`, and `300303`, both
seats. It received 16 selected decision records, 40,568 serialized evidence
bytes, and a 72,638-byte prompt. The provider reported 797,217 prompt tokens,
5,502 completion tokens, 802,719 total tokens, 28 tool calls, and 123.70 seconds
inside the Codex invocation. Learning games used 1,917 environment steps, 957
policy decisions, and 7,462 primitive engine commands.

The authoritative recovery evaluation adds 12 validation and 18 formal games:
11,270 environment steps, 5,630 policy decisions, 34,498 engine commands, and
165.60 seconds. The campaign receipt reports 9 effective learning acts and 100
learning episodes through v8. Token totals before v8 remain unavailable and
are not interpolated.

## Validation and formal benchmark

Validation against the strongest human was 0/12, with 0 wins from either
seat. The validation threshold was therefore not met, but the protocol still
ran all unchanged 18 formal cases.

| Tier | Seat 0 | Seat 1 | Combined |
|---|---:|---:|---:|
| High | 0/3 | 0/3 | 0/6 |
| Medium | 0/3 | 1/3 | 1/6 |
| Low | 3/3 | 3/3 | 6/6 |
| Total | 3/9 | 4/9 | 7/18 |

`raw_score = 0`, `evo_score_8 = gain_8 = 7/18 = 38.89%`. The target required
at least 2/6 high-tier wins, 12/18 total wins, and one high-tier win from each
seat; none of those conditions was satisfied. For comparison, frozen v7 was
12/18. Thus this clean-room edit is a measured regression, not a new champion.

## Behavioral and dense diagnostics

The v8 change prioritized contact before economy and large-stack routing. In
formal play it emitted 8,758 moves, 192 general upgrades, five technology
upgrades, and averaged 2.476 non-end primitives per decision. Of move
destinations, 819 were enemy cells, 81 neutral generals, 2,124 neutral plains,
and 5,734 owned cells.

The formal dense aggregates also degraded relative to v7. Mean completed
rounds survived fell from 372.06 to 200.33; mean terminal territory margin
fell from +22.06 to +10.44; mean terminal army margin fell from +2,445.72 to
-78.83; and mean terminal coin margin fell from +4,522.83 to -47.94. These are
descriptive paired-suite diagnostics, not replacement rewards or causal
attribution. They are consistent with the observed loss of high- and
medium-tier strength.

## Controlled-reference policy KL through v8

- Measurement run: `20260806_1129_6085e1a6`.
- Source v0-v7 run: `20260731_1808_aedfbce7`.
- Source tree SHA-256 before and after:
  `455f1f33314b82f254316883eb6743b4a211b1884eb72b6f94894875974968f5`.
- Source receipt SHA-256:
  `96618bcd3f6e30656c29979e2a15a3ff41cebdf506f8855e37b23988098e91d9`.
- Exact support coverage: 12/12; support range: 7–24,596.
- New v8 actions: 12; new v7→v8 KL facts: 48.
- Total per-state facts: 384 = 8 transitions × 12 states × 4 epsilons.

| Transition | ε=0.001 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---:|---:|---:|---:|
| v0→v1 | 0.000 | 0.000 | 0.000 | 0.000 |
| v1→v2 | 0.000 | 0.000 | 0.000 | 0.000 |
| v2→v3 | 6.599 | 5.396 | 4.394 | 3.827 |
| v3→v4 | 11.127 | 8.930 | 7.134 | 6.144 |
| v4→v5 | 11.127 | 8.930 | 7.134 | 6.144 |
| v5→v6 | 8.103 | 6.505 | 5.198 | 4.478 |
| v6→v7 | 0.000 | 0.000 | 0.000 | 0.000 |
| v7→v8 | 0.000 | 0.000 | 0.000 | 0.000 |

All v8 probes were deterministic in two fresh subprocesses. The v7 and v8
canonical actions were identical on all 12 controlled early states. This zero
is observed equality on the frozen domain, not missingness or interpolation.
It clearly did not predict the formal regression, demonstrating the present
12-state domain's limited behavioral coverage.

Controlled-reference policy KL is a policy-change diagnostic. It is neither a
performance score nor epistemic information gain, and it must be interpreted
beside formal outcomes, replay evidence, and dense trajectories.

## Data quality

The authoritative terminal run contains 130 valid events and the KL run 698
valid events. Both have zero malformed or invalid events, unknown types,
duplicate IDs, missing event IDs, missing run IDs, and warnings. The KL run
contains 181 artifact-reuse receipts, 108 policy-action events, and 384 KL
events. Earlier v0-v7 transition objects remain unchanged.

The updated English three-panel figure is available as
`docs/experiments/figures/generals-controlled-policy-kl-three-panel.{svg,png}`.
