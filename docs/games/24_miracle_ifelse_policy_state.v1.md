# 24_miracle explicit if-else policy state v1

Status: **local deterministic adapter candidate; H08 v2 regeneration still
required**.

This adapter closes the implementation gap between the stateful legacy
`IfElseAI.play()` control flow and the policy-information-gain contract. It
does not approve a replay, change an allowlist, or turn historical post-hoc
data into formal experiment evidence.

## Decision boundary

One decision is one atomic Judge operation. `summon`, `move`, `attack`, and
`use` send an operation and then read a fresh Judge observation before the
legacy Python function continues. A single `play()` can therefore adaptively
emit artifact, repeated pre-move attacks, moves, post-move attacks, summons,
and end-round operations. The complete turn-level command sequence is not a
finite action known at turn start, so a turn-level macro-action cannot supply
the required complete canonical `A(s)`.

The formal comparison context is consequently `z=(s,m)`, where `s` is the
pre-operation observation and `m` is `PolicyMemoryV1`.

## Frozen policy identity

The policy identity binds:

```text
verified six-file source identity
+ canonical 62-input configuration identity
+ 24-miracle-ifelse-explicit-state-machine-v1
+ provider version
```

The 62 inputs are 59 strict booleans plus `MIRACLE_CAMP1_OPENING`,
`MIRACLE_ARTIFACT`, and `MIRACLE_DECK`. Opening is one of `FF/SF/IF`;
artifact and creature names must be members of the Judge `Data.json`
enumerations. An explicit configuration must contain all 62 inputs. A
historical unknown is retained as the literal `unknown` and makes formal
provider construction incomplete.

The source reader binds `Data.json`, `ai_client.py`, `calculator.py`,
`card.py`, `gameunit.py`, and `main.py`; all must be non-empty strict UTF-8,
BOM-free regular files. It revalidates the source before every selection and
loads with bytecode writes disabled.

## Serializable memory

`PolicyMemoryV1` contains only canonical serializable values:

```text
schema_version, state_machine_version
policy_identity, policy_config_identity
episode_id, decision_step, lifecycle
phase, instruction_label, attack_pass, preserve_for_move, acted_iteration
ordered_unit_ids, ordered_unit_snapshots, current_unit_cursor
ordered_target_ids, current_target_cursor
ordered_positions, position_cursor
remaining_capacities, local_mana, local_unit_counts
camp, rng_mode, rng_state, previous_transition_sha256
```

The move-phase snapshots are necessary because legacy `move_phase()` freezes
its sorted unit-object iteration at phase entry. Recomputing that order from a
later observation changes behavior. No Python frame, iterator, policy object,
address, file handle, clock, environment lookup, or RNG object is retained.
The only supported RNG identity is `rng_mode=none`, `rng_state=null`.

Memory and pair-decision evidence are issuer-bound immutable snapshots. Every
episode starts from one explicit reset state. The machine keeps an issuer-side
transition tip only to reject replayed, skipped, or spliced evidence; all
policy decision state remains serialized in `m`. Old and new providers are
recomputed on the same exact `m_before`, observation, and regenerated complete
ActionSupport. They return unsmoothed one-hot base distributions. Epsilon
regularization belongs exclusively to the formal IG measurement layer.

## Local replay evidence

The H08 trace with SHA-256
`e0e280338ae3b06818a923733fb4a04a3730d931ea800e144ecd0aadf4241948`
was replayed sequentially from one reset under an explicit all-false/default
configuration assumption:

```text
399 / 399 chosen operations reproduced
399 / 399 complete ActionSupport values rebuilt
399 / 399 recorded operations were in support
399 / 399 old/new evaluations shared m_before
```

The trace exercised init, opening, artifact, repeated pre-move attack, move,
post-move attack, summon, and endround behavior; the longest consecutive attack
run was five operations.

All eight retained historical v0/v1 traces also replayed completely for both
camps and rank04/rank09 when evaluated under explicit default assumptions. The
observed action counts were 403, 621, 342, and 375 for each version. This is a
behavior-equivalence diagnostic only: those sessions did not record all 62
environment values, so their configuration evidence remains unknown and
cannot be promoted to formal IG evidence.

The verified H08/v0 source has canonical tree SHA-256
`c27a8b646e57b902f527061b4f041b24e000420c7e9bfeacd9d7563ab0de7254`,
legacy tree SHA-256
`209f182637e1abaee4ff50de6b1a37777fbba9fbab5f611aae58a06a109c0a3b`,
and full `main.py` SHA-256
`98199fae8875de63b41d2eacd92ad5c58b4d5aa95b01b05aeacedd0b55402f4b`.
The shorter value ending in `...402f4` is not a valid SHA-256.

A deterministic v0/v1 diagnostic on the fixed camp-0 H08 episode used
different provider identities and explicit FF/SF configurations. Both chose
the same 399 operations, so the ordered fixed-epsilon trace contained 399
zeros, with mean `0.0 nats / decision` and sum `0.0 nats / episode`. This is
not a formal curve or performance result.

## Remaining gate

`CURRENT_INTERNAL_MEMORY_EVIDENCE` remains `not_collected`. A new H08 v2 run
must explicitly record all 62 configuration values and their canonical digest,
the verified source/provider identities, `m_before` and transition evidence at
every decision, and the actual new-policy rollout identity. Only that new
evidence can bind `internal_memory_evidence` and permit a formal IG preflight.
