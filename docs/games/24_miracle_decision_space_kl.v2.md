# 24_miracle Decision Space + policy-KL Core v2

Authority: [24_miracle policy information-gain contract](24_miracle_kl_contract_authority.v2.md).

## Scope

The core is a pure, synthetic/fake-only calculator. It never runs a policy,
Judge, environment, opponent, provider, match or session, and does not claim
real old/new policy provenance or authoritative readiness.

One Miracle decision is one legal atomic Judge operation at a target-agent
pre-action observation. Public families are `init`, `move`, `attack`,
`summon`, `use`, `endround` and `surrender`. Parser-internal operations are not
decisions. The complete support is rebuilt from the observation, canonicalized,
assigned stable action IDs, sorted and bound to `support_id`. WindBlessing
continues to fail closed where its authoritative position support cannot be
finitely enumerated.

## Strict evidence boundary

`DecisionKLEvidence` contains only:

- continuous `decision_step`;
- frozen `state_before`;
- frozen complete support identity;
- complete old and new action-ID probability mappings.

Nested caller-owned data is copied and frozen. The calculator regenerates the
support, checks schema/support/action order, rejects missing or extra actions,
rejects bool/string/NaN/Infinity/negative probabilities, and requires unit
mass without normalization or implicit zero filling. A submitted local KL or
summary is not accepted as input.

Issuer-only decision records and summaries are bound to closure-owned snapshots.
Construction, copying, replacement or post-issue mutation cannot be reused as
trusted aggregation evidence. Expected support/distribution unavailability is
represented as a structured `incomplete` record with a null scalar. API type,
identity and ordering violations fail closed.

## Formal calculation

For complete current support `A(s)` and fixed `epsilon=0.01`:

```text
pi_epsilon(a|s) = 0.99 * pi(a|s) + 0.01 / |A(s)|

local_policy_kl(s)
  = sum_a new_epsilon(a|s) * ln(new_epsilon(a|s) / old_epsilon(a|s))
```

The logarithm is natural. Both policies use the same support and identical
smoothing channel. Equal deterministic actions produce zero; changed
deterministic actions produce a finite positive value. Epsilon affects the
measurement only and never the executed policy.

The summary retains ordered `decision_records` and `trace`. For a complete,
non-empty trace:

```text
information_gain = trajectory_kl = arithmetic_mean(trace)
sum_local_kl = sum(trace)
```

`information_gain`/`trajectory_kl` use `nats / decision`; `sum_local_kl` uses
`nats / episode`. Maximum and percentiles remain diagnostics. There is no
policy-change acceptance threshold; the deprecated compatibility fields
`acceptance_threshold` and `threshold_passed` are always null. Empty or partly
incomplete traces never expose an episode scalar.

Every machine-readable summary remains labeled
`evidence_scope=synthetic_fake_only`, `authoritative_readiness=false`,
`verified_rollout_source=null` and `policy_binding_verified=false`.
`rollout_source_contract=new_policy` states the target contract only; it does
not turn synthetic states into verified occupancy evidence.

## Remaining real-evidence gap

The current synthetic input records visible state only. If a real HL policy is
stateful, authoritative comparison requires canonical `z=(s,m)` evidence.
Until internal memory `m` is captured and bound, real stateful policy KL is
incomplete even when visible-state support and distributions are otherwise
valid.
