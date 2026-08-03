# 24_miracle policy information-gain contract authority v2

Status: **measurement contract migrated locally; authoritative execution remains blocked**.

This document records the group-approved scientific definition for 24_miracle.
It governs measurement semantics, not production approval, real replay status,
benchmark completion, or permission to start an experiment.

## Fixed scientific identity

```text
ACTION_BOUNDARY = one legal atomic Judge operation
SUPPORT = same complete canonical state-local ActionSupport A(s)
KL_DIRECTION = new||old
LOGARITHM = natural
SMOOTHING = symmetric epsilon mixture with uniform(A(s))
MAIN_EPSILON = 0.01
SENSITIVITY_EPSILONS = 0.001, 0.01, 0.05
OCCUPANCY_SOURCE = actual new-policy rollout
PRIMARY_EPISODE_AGGREGATE = arithmetic mean
PRIMARY_UNIT = nats / decision
OPTIONAL_SUM_UNIT = nats / episode
TERMINAL_STATE_INCLUDED = false
```

For `v_{k-1} -> v_k`, at each actual target-agent decision context `z_t`:

```text
pi_epsilon(a|z_t) = (1-epsilon) * pi(a|z_t) + epsilon / |A(s_t)|

local_policy_kl_k(z_t)
  = KL(pi_k,epsilon(.|z_t) || pi_k-1,epsilon(.|z_t))
```

The same epsilon and the same ordered, complete support are used for both
policies. Smoothing is a measurement layer only; it never changes the action
executed by the new policy. The main result always uses epsilon `0.01`.
Sensitivity values use the fixed panel above and cannot replace or relabel the
main result.

Each episode retains the ordered source trace:

```text
local_policy_kl_trace = [k_0, ..., k_(T-1)]
information_gain = mean(local_policy_kl_trace)       # nats / decision
local_policy_kl_sum = sum(local_policy_kl_trace)     # nats / episode
```

The terminal state is excluded because it produces no action. Empty or
incomplete evidence produces `null/incomplete`, never zero. There is no KL
magnitude acceptance threshold: epsilon `0.01` is not a performance gate, and
large policy change does not imply score improvement.

## Action and probability boundary

For Miracle, the action-mask concept is a state-local, complete, ordered and
verifiable `ActionSupport + support_id`, not a fixed global Boolean vector.
The support is regenerated from the visible observation where the synthetic
core can do so. Old/new action IDs must match it exactly. Missing, extra,
duplicated or drifted identities fail closed.

Probabilities must be exact numeric values (`int` or `float`, never `bool` or
strings), finite, non-negative and unit mass. The framework does not normalize,
zero-fill or coerce submitted distributions. Uploaded local KL, trace summary,
episode mean or sum is never authoritative: validators recompute the values
from the ordered distributions and reject disagreement.

## Occupancy and interpretation

`occupancy_shift` is an independent measurement of state visitation change.
It is never added to local policy KL or episode information gain. Policy KL is
behavioral/policy information gain, not epistemic information gain and not a
performance score. Performance remains governed by the frozen evaluation
win-rate/Elo contract.

The occupancy source for the main episode measure is the actual rollout of the
unsmoothed new policy. Consequently the saved local trace is an
epsilon-regularized policy-change measurement under new-policy occupancy; the
optional sum is not presented as a third independent KL.

## Decision context and current data gap

If action choice depends on internal memory, the decision context is
`z=(s,m)`. The current Miracle adapters can advance stateful callbacks through
`reset()` and `observe_transition()`, but the evidence record only proves the
visible observation/state identity. It does not yet serialize a canonical
policy-memory identity. A stateful real run therefore has an explicit
`internal_memory_evidence=not_collected` gap and cannot claim authoritative
policy KL until that evidence is supplied. Stateless fake tests do not close
this real-data gap.

## Security and lifecycle boundary

The migration preserves support completeness, chosen-in-support checks,
ordered decision records, strict probability validation, immutable evidence
snapshots, issuer/provenance checks, replay/path safety, revocation checks and
lifecycle revalidation. Evaluation-case `information_gain` must equal the mean
derived from its authoritative ordered trajectory evidence. Acceptance derives
its aggregate from those validated per-case means rather than self-reported
case fields.

Production approval tables remain empty. The bootstrap file and approved
bootstrap SHA are unchanged. Historical v0/v1 data remains `information_gain =
null` because it excluded replay, decision trace, complete action support and
old/new policy distributions; no formal IG can be reconstructed from it.
