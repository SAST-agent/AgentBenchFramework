# Generals HL v6 macro-planner audited result

Date: 2026-07-30

Run: `20260729_1653_af8eda26`

Status: **complete and runnable**. The run used one provider act, saved all 36
declared games, passed the medium and low success gates, and missed the
additional high-tier breakthrough gate.

## Audit verdict

The frozen v6 scored `12/18` (`0.666667`) on formal evaluation, versus the
exact parent v5's `7/18` (`0.388889`). This is an absolute same-matrix
improvement of `5/18` (`+0.277778`): medium changed from `1/6` to `6/6`, low
remained `6/6`, and high remained `0/6`.

This is an observed before/after result, not a causal estimate of the
macro-planner change. The saved AUC and epistemic information-gain fields are
unavailable, and the dense/action-profile diagnostics are descriptive only.

## Frozen lineage, prompt, and Skill

| Artifact | Recomputed SHA-256 | Audit |
|---|---|---|
| Exact v5 parent source | `facd39c8a0c064823da68f3dd3eba4a832c08815f75369407112ca087f1ecd9e` | Matches lineage, v5 manifest, and parent run `20260729_0818_e6bcb9b3` |
| Frozen v6 source | `974050ee1a3d4b4c4f96e61f5af39b4e52e2cfbc50f146f9e8b4ac96c4ac798b` | Matches captured source, v6 manifest, provider act, and all isolated-workspace verification events |
| Provider prompt | `7e0e950732dffd8c67967ba2b7e1f3b2b51c173820a22c0819dacb56f7927e66` | Matches the 91,519 prompt bytes and prompt manifest |
| replay-analysis-v2 entry | `0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48` | Matches source Skill, frozen copy, receipt, and prompt manifest |

The prompt was not truncated. It included exactly six whole learning episodes,
31 selected decision records, and no omitted episode. The included IDs were
the two seats for each of `287101`, `287202`, and `287303`. Formal seeds
`280101/280202/280303` and validation seeds `288101/288202/288303` do not
occur in the prompt. Declared round-5 replay/state citations remain in the
inherited V5 experience role.

The v6 manifest changes only `strategy.py`, `state_view.py`, `STRATEGY.md`,
`EXPERIENCE.md`, and `tests/test_strategy.py`. `main.py` is byte-identical to
v5. The saved version event records `scope_valid`, `main_unchanged`,
`isolation_valid`, `tests_passed`, and `runnable` as true.

## Phase results

| Phase | Evaluated version | Opponent tiers | Wins | Losses | Score/status |
|---|---|---|---:|---:|---|
| Learning | v5 | high only | 0 | 6 | Evidence collection; all six valid |
| Validation | v6 | high + medium | 6 | 6 | `6/12 = 0.500000`, complete |
| Formal | v6 | high + medium + low | 12 | 6 | `12/18 = 0.666667`, complete |

Validation was balanced across seats: each seat had three high losses and
three medium wins. Each validation seed likewise had two losses and two wins.
All six learning games were high-tier losses, two per seed and three per seat.

## Formal tier, seed, and seat breakdown

`W/L` cells are ordered `seat 0 / seat 1`.

| Version | Tier | 280101 | 280202 | 280303 | Seat 0 | Seat 1 | Tier total |
|---|---|---|---|---|---:|---:|---:|
| v5 | High | L/L | L/L | L/L | 0/3 | 0/3 | 0/6 |
| v5 | Medium | L/L | L/L | L/W | 0/3 | 1/3 | 1/6 |
| v5 | Low | W/W | W/W | W/W | 3/3 | 3/3 | 6/6 |
| v6 | High | L/L | L/L | L/L | 0/3 | 0/3 | 0/6 |
| v6 | Medium | W/W | W/W | W/W | 3/3 | 3/3 | 6/6 |
| v6 | Low | W/W | W/W | W/W | 3/3 | 3/3 | 6/6 |

Across all formal tiers, v6 was `6/9` from seat 0 and `6/9` from seat 1.
Every formal seed contributed four wins and two losses. All 18 results were
valid, normally terminated, and error-free.

## Score, gain, and AUC

| Record | v5 | v6 | Difference |
|---|---:|---:|---:|
| Formal wins | 7/18 | 12/18 | +5/18 |
| Formal score | 0.388889 | 0.666667 | +0.277778 |
| High | 0/6 | 0/6 | 0 |
| Medium | 1/6 | 6/6 | +5 |
| Low | 6/6 | 6/6 | 0 |
| Saved gain vs raw score 0 | 0.388889 | 0.666667 | +0.277778 |
| All five AUC fields | `null` | `null` | Not computable |
| AUC status | `unavailable_missing_score_point` | `unavailable_missing_score_point` | — |

The saved `gain_6` is `0.666667`, because the harness defines the per-version
gain against `raw_score=0.0`; it is not the incremental v5-to-v6 difference.
The same-matrix incremental difference is the independently computed
`+0.277778` above.

`AUC_coding_agent_act`, `AUC_episode`, `AUC_env_step`, `AUC_token`, and
`AUC_time` are all `null`. Their status is
`unavailable_missing_score_point`, consistent with the missing value in the
saved score history:
`[0, 0, null, 0, 7/18, 4/18, 7/18, 12/18]`.
No numeric AUC is claimed or imputed.

## Action profiles

These profiles count normalized non-end primitives for the evaluated seat
only. They are not the same population as the budget's all-player primitive
command counts.

| Phase/version | Turns | Primitive commands | Command histogram | General upgrades | Tech upgrades | End-only turns | Multi-command turns | Mean/max primitives | Neutral-plain move ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Learning/v5 | 1,228 | 1,214 | `1:1207, 3:7` | 7 | 0 | 14 | 0 | 0.9886 / 1 | 0.867439 |
| Validation/v6 | 3,496 | 11,475 | `1:11250, 3:208, 5:17` | 208 | 17 | 24 | 3,469 | 3.2823 / 4 | 0.085867 |
| Formal/v6 | 6,716 | 23,288 | `1:22820, 3:441, 5:27` | 441 | 27 | 36 | 6,668 | 3.4675 / 4 | 0.099167 |

The v6 profiles show that the bounded policy emitted multi-primitive macros,
production/general upgrades, and technology upgrades while greatly reducing
the aggregate neutral-plain routing ratio relative to the v5 learning
episodes. These phase populations differ, so the table does not isolate a
causal policy effect.

## Dense evidence

The following values are unweighted means over the six saved cases in each
row. Formal v5 and v6 rows use the same tier/seed/seat cases. Only populated
dense fields are reported.

| Matrix/version | Tier | Completed rounds | Terminal army margin | Time-average army margin | Terminal coin margin | Terminal territory margin | Time-average territory share |
|---|---|---:|---:|---:|---:|---:|---:|
| Formal/v5 | High | 124.67 | -2,474.67 | -797.22 | -2,434.50 | -80.50 | 0.2454 |
| Formal/v6 | High | 186.17 | -3,876.83 | -1,205.97 | -5,654.17 | -61.33 | 0.3138 |
| Formal/v5 | Medium | 270.67 | -75.50 | -90.78 | -2,240.67 | -1.00 | 0.4563 |
| Formal/v6 | Medium | 432.50 | 3,135.50 | 1,014.57 | 5,639.00 | 52.17 | 0.6811 |
| Formal/v5 | Low | 500.00 | 2,512.00 | 922.61 | -335.83 | 54.67 | 0.6831 |
| Formal/v6 | Low | 500.00 | 8,171.83 | 2,862.06 | 11,750.67 | 77.50 | 0.8193 |
| Validation/v6 | High | 156.00 | -2,633.50 | -888.96 | -4,322.50 | -46.17 | 0.3433 |
| Validation/v6 | Medium | 425.83 | 2,716.67 | 879.77 | 6,977.50 | 61.67 | 0.7247 |

On the same formal cases, medium moved from mostly losing with negative
economic/army aggregates to six wins with positive terminal margins. Low
retained all wins with larger positive aggregates. High survived longer and
held a larger average territory share, but still lost all six games and ended
with large army and coin deficits. These associations do not establish that a
specific macro-planner mechanism caused the score change. Dense `army_share`
is null throughout. The three saved pressure metrics are fully null for every
high-tier case and only partially populated for medium/low, so they are not
aggregated or inferred from other fields here.

## Information-gain and behavior diagnostics

`epistemic_information_gain` is `null` with status
`complete_action_distribution_unavailable`. `policy_kl` is also `null`, with
status `complete_macro_action_distribution_unavailable`. Neither quantity was
measured numerically.

The saved deterministic `action_disagreement` is `0.8064516129` over 31 probe
decisions. All 31 probe decision classes are `end_only`; every other class has
count zero. This is a limited behavior-disagreement diagnostic, not epistemic
information gain, policy KL, or broad decision-space coverage.

## Provider and budget audit

| Field | Saved value |
|---|---:|
| Normal-run provider acts | 1 |
| Provider return code | 0 |
| Provider elapsed time | 369.026 s |
| Tool calls | 40 |
| Raw provider events / malformed lines | 49 / 0 |
| Input tokens | 2,055,195 |
| Cached input tokens | 1,946,880 |
| Output tokens | 15,937 |
| Reasoning output tokens | 2,461 |
| Total tokens | 2,071,132 |
| Local learning time, including games/act | 377.298 s |
| Validation time | 144.230 s |
| Formal time | 381.258 s |
| Total recorded phase time | 902.786 s |

The summary's phase budgets use the broader gameplay accounting:

| Phase | Episodes | Environment steps | Evaluated-agent decision steps | All-player primitive commands | Time |
|---|---:|---:|---:|---:|---:|
| Learning | 6 | 2,459 | 1,228 | 6,168 | 377.298 s |
| Validation | 12 | 6,994 | 3,496 | 22,763 | 144.230 s |
| Formal | 18 | 13,435 | 6,716 | 42,348 | 381.258 s |
| Total | 36 | 22,888 | 11,440 | 71,279 | 902.786 s |

The all-player primitive column is intentionally larger than the
evaluated-seat action-profile counts and must not be used interchangeably.

Token accounting is marked exact. Historical cumulative learning tokens are
`null` because earlier lineage records lack compatible token points; the
current local provider usage above is complete and must not be promoted into a
historical cumulative total.

Provider stderr is non-empty (2,141 bytes). It records one safely rejected
`rm -f` cleanup command and one file-watcher unwatch warning. The provider then
used `unlink`/`rmdir` successfully, returned code 0, and left a runnable frozen
candidate. The quality record's zero warnings therefore means zero event-log
quality warnings, not empty provider stderr.

## Quality, invalid, and incomplete states

- `quality.json` and the summary agree on 169 total lines, 169 valid events,
  and zero malformed lines, invalid events, duplicate/missing IDs, unknown
  event types, or event-quality warnings.
- Exactly 6 learning, 12 validation, and 18 formal matches exist. Their seed
  sets are pairwise disjoint. Every match has metadata, official and normalized
  replay, dense summary/trace, and both players' process, protocol, and stderr
  artifacts.
- All 36 match results are valid, normally terminated, and error-free.
  Validation and formal statuses are complete; their error fields are null.
- The prompt is complete and untruncated. The provider act is completed, the
  source scope is valid, `main.py` is unchanged, and validation/formal frozen
  source checks match the v6 hash.
- The saved candidate log reports `27 passed`; an independent rerun from
  `versions/v6/source` reports `27 passed`, and all four frozen Python files
  compile without writing to the source.
- That independent audit rerun created two excluded bytecode files at
  `2026-07-30 01:12:49 +0800`:
  `versions/v6/source/__pycache__/strategy.cpython-311.pyc` and
  `versions/v6/source/tests/__pycache__/test_strategy.cpython-311-pytest-8.4.2.pyc`.
  The final summary predates them (`2026-07-30 01:08:20 +0800`), they are
  absent from the frozen manifest, and they did not exist during the 36-game
  evaluation. They therefore did not affect the recorded score or success
  gates. They are retained unchanged as post-finalization audit provenance.
- Subsequent review hardening makes v6 isolation, lineage import, and
  post-act recovery materialize only hash-verified manifest files, preventing
  excluded cache or environment files from propagating into future policy
  workspaces. Future candidate audits use
  `PYTHONDONTWRITEBYTECODE=1
  /home/cathy/AgentBench/AgentBenchFramework/.worktrees/generals-hl/.venv/bin/python
  -m pytest -p no:cacheprovider tests/test_strategy.py -q` from the frozen
  source working directory.
- Unavailable diagnostics remain explicit: all AUC fields, epistemic
  information gain, and policy KL are null; several dense pressure/share
  fields are null; cumulative historical token totals are null; and
  `resource_summary` is empty. None is replaced with zero.
- Provider stderr contains the two non-fatal diagnostics described above even
  though event quality is clean.

Attempt 1 remains immutable at `20260729_1555_c23d1075`: status `failed`, six
learning episodes, zero round provider acts, 36 events, and no provider
directory, v6 source, or score. Its `summary.json` and `events.jsonl` hashes
remain respectively
`e92bbeb54f32e5db1c737ee4f44da2e9c8c2e51c4cc65b522e28895f23b0279f`
and
`139ab46bd3b756ee3bc0662a9d081fc36c7ccf707760b64187782a227be9fc90`.

## Success gates and next bottleneck

| Gate | Result | Pass |
|---|---:|---|
| Medium at least 2/6 | 6/6 | Yes |
| Low exactly 6/6 | 6/6 | Yes |
| High at least 1/6 | 0/6 | **No** |

The next evidence-supported bottleneck is conversion against the strongest
opponent, not medium/low retention. High was `0/6` in learning, validation, and
formal evaluation across all saved seeds and both seats. On the same formal
cases, v6 increased survival and territory share but finished with larger army
and coin deficits; the validation high cases show the same negative terminal
army/coin pattern. The aggregate formal action profile confirms that the new
policy actively uses bounded macros and upgrades, but it is not tier-specific
and cannot identify which choice caused the high losses.

A follow-up should therefore use fresh, isolated high-tier learning evidence
to test resource accumulation and conversion into decisive pressure while
preserving the achieved medium/low behavior. Because pressure fields and full
action distributions are unavailable, this report cannot narrow the cause to
main pressure, upgrade timing, routing, or another mechanism without new
instrumentation and evidence.
