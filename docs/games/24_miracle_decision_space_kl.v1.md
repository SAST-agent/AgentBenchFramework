# 24_miracle Decision Space + KL Core v1

Authority: [24_miracle KL contract authority v1](24_miracle_kl_contract_authority.v1.md).
That decision fixes the target contract while keeping this committed core an
isolated synthetic/fake-only mechanism; it does not assert runtime migration.

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

For Checklist terminology, one complete Miracle game action is this one atomic
Judge operation. `hl_iteration_step` is the outer learning/coding lifecycle,
`game_agent_decision_step` is the measured agent's choice after an observation,
and `judge_operation_step` is the submitted atomic command. The last two are
currently one-to-one for the measured agent. Miracle has no command-list
macro-action runtime; the Generals command-list definition is out of scope.

`build_trusted_action_support(observation)` wraps the existing game-rule
enumerator. The calculator does not trust a policy or evidence producer to
declare legal actions. Each canonical command is validated, serialized with
stable key order, hashed into its versioned action ID, checked for uniqueness,
and sorted by that ID. The supplied support identity must exactly match the
trusted schema version, support digest, and ordered action-ID sequence.

For Miracle, this state-local, complete, ordered `ActionSupport + support_id` is
the Checklist action-mask representation. It is not a fixed global index or a
Boolean/0-1 vector. Failure to prove complete finite support fails closed, and
membership in the legal support does not imply tactical value or rationale.

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

The `1e-9` envelope is the public distribution-validation boundary, not
permission to treat non-unit mass as a probability distribution. Before KL
arithmetic, each accepted total must still equal one within a fixed
eight-ULP roundoff bound. Evidence outside that bound is preserved as
`incomplete` with an `old_distribution_mass_not_strict` or
`new_distribution_mass_not_strict` reason. It is never normalized.

A submitted `local_policy_kl` is never accepted as input. The core recomputes
the value from the captured distributions and trusted support identity.

`compute_trajectory_kl()` accepts only `DecisionKLEvidence`: the decision
step, complete `state_before`, captured support identity, and complete old/new
distributions. For every item it regenerates ActionSupport from the state,
checks the supplied identity, and recomputes local KL before aggregation.
`DecisionKLRecord` is computation output only; passing a publicly constructed
record, mapping, uploaded scalar, `local_kl`, `reported_local_kl`, or a
decision-change value to the trajectory API is rejected.

Expected domain unavailability is represented by a JSON-safe, enumerated
reason and a null trace position. This includes non-finite ActionSupport,
missing or extra distribution actions, invalid probability values, invalid
mass, and a negative computed value beyond the documented floating-point
roundoff bound. Support-identity mismatch, wrong evidence types, and
non-continuous decision steps remain fail-closed API errors rather than
trusted incomplete evidence.

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

Active generic and lifecycle code that still uses `KL(new||old)`, epsilon
smoothing, or an episode sum is legacy for the 24_miracle target contract; see
the authority document. This file does not claim that code has been migrated.
