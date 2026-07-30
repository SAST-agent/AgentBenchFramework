# 24_miracle Decision Space + KL Core v1

## Scope

This fake-only core defines one decision space and a pure trusted KL
calculator. It does not run a policy, Judge, environment, opponent, Provider,
match, or session. It is not connected to tracking, Results, reports,
benchmarks, or the authoritative evaluation lifecycle.

## Decision point and action families

One decision is one legal atomic Judge command at a visible target-agent
observation. Public operation families are:

| Operation | Parameters |
|---|---|
| `init` | one artifact and three distinct ordered creatures |
| `move` | mover ID and cube position |
| `attack` | attacker ID and unit/miracle target ID |
| `summon` | creature type, level, and cube position |
| `use` | artifact ID and unit/cube target |
| `endround` | none |
| `surrender` | none |

Parser-internal `forbid`, `select`, and `startround` are not decisions. There
is no separate `continue` action.

`build_trusted_action_support(observation)` wraps the existing game-rule
enumerator. The calculator does not trust a policy or evidence producer to
declare legal actions. Each canonical command is validated, serialized with
stable key order, hashed into its versioned action ID, checked for uniqueness,
and sorted by that ID. The supplied support identity must exactly match the
trusted schema version, support digest, and ordered action-ID sequence.

The initial support retains all 840 ordered-card choices, including the
Inferno creature. Runtime artifact domains remain those of the existing
enumerator: SalamanderShield uses its Judge-compatible unit domain and
InfernoFlame uses the current finite position domain. Available WindBlessing
fails closed because its authoritative position support is unbounded.

## Distribution contract

Old and new distributions use the same complete ActionSupport. Both must be
exact `action_id -> probability` mappings:

- no missing or extra action IDs;
- values are `int` or `float`, never `bool` or strings;
- every value is finite and non-negative;
- total mass equals one with absolute tolerance `1e-9`;
- no normalization, smoothing, or implicit zero filling.

A submitted `local_policy_kl` is never accepted as input. The core recomputes
the value from the captured distributions and trusted support identity.

`compute_trajectory_kl()` accepts only `DecisionKLEvidence`: the decision
step, complete `state_before`, captured support identity, and complete old/new
distributions. For every item it regenerates ActionSupport from the state,
checks the supplied identity, and recomputes local KL before aggregation.
`DecisionKLRecord` is computation output only; passing a publicly constructed
record, mapping, uploaded scalar, `local_kl`, `reported_local_kl`, or a
decision-change value to the trajectory API is rejected.

## Frozen KL definition

At decision state `s`:

```text
D_KL(old || new) = sum_a old(a|s) * ln(old(a|s) / new(a|s))
```

The logarithm is natural and local values are measured in nats. There is no
epsilon smoothing. An `old=0` term contributes zero. If `old>0` and `new=0`,
the record is `threshold_failed` with reason `old_positive_new_zero`; its
structured scalar is null, never Infinity or NaN.

For a non-empty trajectory under new-policy occupancy, the primary value is
the arithmetic mean of all complete local KL values:

```text
trajectory_kl = mean(local_kl_1, ..., local_kl_N)
```

Its unit is `nats / decision`. Sum, maximum, p50, and p95 are diagnostic only.
The fixed acceptance threshold is `0.01`; a mean equal to the threshold passes,
while a greater value fails. An empty trajectory is `incomplete`. Any
incomplete decision keeps the trajectory scalar absent, and a structured
infinite local result has threshold-failure priority.

## Current boundary

This mechanism is only a synthetic/fake-only calculator. It does not prove
real old/new policy binding, controlled Judge randomness, opponent
qualification, benchmark completion, or authoritative readiness. Formal
evaluation, tracking persistence, report rendering, Results integration, and
real strategy execution are explicitly deferred.
