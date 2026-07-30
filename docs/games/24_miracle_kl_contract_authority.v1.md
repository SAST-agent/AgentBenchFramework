# 24_miracle KL contract authority v1

Status: **target contract approved; documentation-only labeling; runtime not migrated**.

This document records the game-owner decision for the target 24_miracle action
and KL contract. It has authority over the target meaning of 24_miracle terms.
It does not rewrite history, change any current Python behavior, approve an
experiment, or make legacy output valid evidence for this target contract.

The isolated calculator described in
[24_miracle Decision Space + KL Core v1](24_miracle_decision_space_kl.v1.md)
already implements this target calculation for synthetic/fake-only inputs. The
research lifecycle, generic KL implementation, tracking, reporting, Results,
and real runtime remain unmigrated unless a later separately authorized stage
changes and verifies them.

## 1. Authoritative target decision

The target constants are exactly:

```text
ACTION_BOUNDARY = ATOMIC_JUDGE_OPERATION
LEGAL_ACTION_REPRESENTATION = STATE_LOCAL_COMPLETE_ACTION_SUPPORT

KL_DIRECTION = old||new
LOGARITHM = natural
SMOOTHING = none
OCCUPANCY = new_policy
TRAJECTORY_PRIMARY = arithmetic_mean
UNIT = nats / decision
ACCEPTANCE_THRESHOLD = 0.01
```

For one decision state `s`, the target local value is:

```text
D_KL(old || new) = sum_a old(a|s) * ln(old(a|s) / new(a|s))
```

There is no normalization, epsilon smoothing, implicit zero filling, or
substitute metric. A term with `old=0` contributes zero. If any action has
`old>0,new=0`, the result is a structured threshold failure with no JSON
Infinity or NaN. If complete support or either strict distribution is
unavailable, the scalar stays absent and the structured reason is retained.
Decision-change rate, Jensen-Shannon divergence, occupancy KL, win-rate change,
or any other quantity must not fill that absence.

Under new-policy occupancy, the primary trajectory statistic is the arithmetic
mean of all complete target-agent local values. Its unit is `nats / decision`.
The acceptance comparison is `mean <= 0.01`; sums, maxima, and percentiles may
be diagnostics only and must not replace the primary value.

## 2. Action boundary and step names

The following counters name different layers and must not be merged:

- `hl_iteration_step`: one outer heuristic-learning/coding lifecycle step that
  may inspect evidence and propose a strategy change. It is not a game action.
- `game_agent_decision_step`: one target game agent decision after one visible
  observation. It counts only decisions by the measured/evaluated policy.
- `judge_operation_step`: one atomic operation submitted to the Miracle Judge.

In the current Miracle runtime, one observation produces one atomic Judge
operation. Therefore `game_agent_decision_step` and `judge_operation_step` are
currently one-to-one for the measured agent. Miracle has no runtime layer in
which one action is a command list or multi-command macro-action. The generic
Generals definition of a command-list macro-action remains valid for Generals
and does not apply to Miracle.

The trace-wide count named `ai_operation` may include both players. It must not
be used as the measured policy's decision count without actor filtering and a
verified one-to-one binding.

## 3. Legal action representation

For Miracle, the Checklist phrase "action mask" means a state-local, complete,
ordered, and verifiable `ActionSupport + support_id` contract:

- the support is rebuilt from the current visible observation;
- it contains every atomic operation whose legality can be proved for that
  observation;
- action IDs, schema identity, order, and `support_id` are verified together;
- old and new distributions use exactly that same support;
- missing, extra, duplicated, or identity-drifted actions fail closed;
- if completeness cannot be proved or the legal domain cannot be finitely
  enumerated, strict KL is unavailable and records a structured reason.

This representation is not a fixed global action index and is not a Boolean or
0/1 vector. No documentation may describe it as such. Conversely, legal means
accepted by the action contract; it does not mean tactically sensible, optimal,
or evidence of a particular rationale.

## 4. Unmigrated legacy implementation and documentation

The following active code still implements or validates the earlier
`new||old + epsilon smoothing + episode local-KL sum` contract:

| Path | Current legacy role |
|---|---|
| `src/agentbench_frame/eval/information_gain.py` | generic epsilon-regularized `KL(new||old)` calculation |
| `src/agentbench_frame/eval/trajectory_kl.py` | generic online wrapper whose primary episode value is the local-KL sum |
| `src/agentbench_frame/games/miracle/research_protocol.py` | current manifest fields still declare the old Miracle KL identity |
| `src/agentbench_frame/games/miracle/iteration_protocol.py` | current fake-only lifecycle validators still require the old identity |
| `src/agentbench_frame/tracking/run.py` | current event persistence defaults and sum/mean derivation |
| `src/agentbench_frame/report/builder.py` | current report derivation from persisted legacy traces |

The following documentation contains historical or generic descriptions of
that behavior and is scoped accordingly:

- [24_miracle research protocol v1](24_miracle_research_protocol.v1.md);
- [Trajectory KL downstream integration](../integration/trajectory-kl.md);
- [Information-gain design](../research/information-gain-design.md);
- [Research methodology summary](../research/research-methodology-summary.md).

This authority document does not change any of those implementations. Their
existing output must not be represented as evidence conforming to the target
24_miracle contract. Schema, trusted collection, persistence, presentation,
and runtime migration require separate scopes, changes, and verification.
Historical plans and specifications remain historical records and are not
rewritten by this documentation stage.

## 5. Migration and evidence boundary

Migration must be split into independently authorized stages:

1. schema and validator identity;
2. trusted atomic-decision collection and real old/new policy binding;
3. persistence of evidence and structured missing reasons;
4. report and Results presentation;
5. Miracle runtime wiring and fake-only end-to-end verification;
6. separately authorized real execution and evidence review.

Until a stage is implemented and independently verified, it remains legacy or
not done. A mixture of target labels and legacy scalars is not target evidence.

## 6. Checklist and readiness status

```text
Decision Space/KL = MECHANISM_ONLY
Replay Reading = MECHANISM_ONLY
Action mask = state-local ActionSupport + support_id, not a global Boolean mask
Runtime wiring = NOT_DONE
Human Replay Skill = NOT_DONE
Real replay = NOT_DONE
Authoritative readiness = false
```

This decision does not complete Checklist items 3 or 4. The synthetic core is
not a Human Replay Skill, real replay validation, benchmark result, production
approval, or authorization to run Judge, Provider, opponents, matches, matrix,
or sessions.
